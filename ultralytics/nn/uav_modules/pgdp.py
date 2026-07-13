"""Position Gaussian Distribution Prediction for tiny-object feature enhancement.

Adapted for UAV-DETR from Bian et al., "Feature Information Driven Position
Gaussian Distribution Estimation for Tiny Object Detection" (CVPR 2025).
The original method is implemented on an FPN.  Here PGDP consumes the raw
backbone P2/P3/P4 features and only replaces the existing P2 skip entering the
high-resolution neck, while the FDR decoder remains unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import CBAM

__all__ = ["PGDPEnhance"]


def _conv_block(c1: int, c2: int, repeats: int = 2) -> nn.Sequential:
    layers = []
    for i in range(repeats):
        layers.extend(
            [
                nn.Conv2d(c1 if i == 0 else c2, c2, 3, padding=1, bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
            ]
        )
    return nn.Sequential(*layers)


class PGDPEnhance(nn.Module):
    """Supervised P2 enhancement with a multi-scale position Gaussian map.

    Inputs are ``[P2, P3, P4]``. During training, ``set_targets`` supplies the
    normalized xywh boxes used to generate the label-derived Gaussian map.
    Three side outputs receive the weighted-MSE deep supervision described in
    the paper. The learned P2 map is fused through a bounded residual so the
    original UAV-DETR P2 skip remains an explicit fallback path.
    """

    def __init__(
        self,
        channels,
        hidden: int = 64,
        pred_gain: float = 1.0,
        gamma: float = 0.1,
        gamma_max: float = 1.0,
    ):
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"PGDPEnhance expects [P2, P3, P4] channels, got {channels}")
        if not 0.0 < gamma < gamma_max:
            raise ValueError(f"Expected 0 < gamma < gamma_max, got {gamma=} and {gamma_max=}")

        c2, c3, c4 = channels
        self.p2_proj = nn.Conv2d(c2, hidden, 1, bias=False)
        self.p3_proj = nn.Conv2d(c3, hidden, 1, bias=False)
        self.p4_proj = nn.Conv2d(c4, hidden, 1, bias=False)

        self.conv4 = _conv_block(hidden, hidden, repeats=2)
        self.up4 = nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1, bias=False)
        self.conv3 = _conv_block(hidden, hidden, repeats=2)
        self.up3 = nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1, bias=False)
        self.conv2 = _conv_block(hidden, hidden, repeats=2)

        self.map4 = nn.Conv2d(hidden, 1, 3, padding=1)
        self.map3 = nn.Conv2d(hidden, 1, 3, padding=1)
        self.map2 = nn.Conv2d(hidden, 1, 3, padding=1)
        self.cbam = CBAM(c2)

        ratio = gamma / gamma_max
        self.raw_gamma = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))
        self.register_buffer("gamma_max", torch.tensor(float(gamma_max)))
        self.pred_gain = float(pred_gain)

        self._targets = None
        self._image_size = None
        self._aux_loss = None

    @property
    def effective_gamma(self):
        return self.gamma_max * torch.sigmoid(self.raw_gamma)

    def set_targets(self, targets: dict, image_size) -> None:
        """Attach one training batch of normalized xywh targets."""
        self._targets = {
            "bboxes": targets["bboxes"].detach(),
            "batch_idx": targets["batch_idx"].detach(),
        }
        self._image_size = tuple(int(v) for v in image_size)
        self._aux_loss = None

    def consume_aux_loss(self):
        """Return the most recent PGDP auxiliary loss and clear cached state."""
        loss = self._aux_loss
        self._aux_loss = None
        self._targets = None
        self._image_size = None
        return loss

    @staticmethod
    def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        weight = torch.where(target > threshold, 10.0, 0.1)
        return (weight * (pred - target).square()).mean()

    @torch.no_grad()
    def _build_gaussian_target(self, p2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the paper's label-conditioned position Gaussian map at P2."""
        batch, _, height, width = p2.shape
        target = p2.new_zeros((batch, 1, height, width))
        threshold = p2.new_zeros((batch, 1, 1, 1))
        if self._targets is None:
            return target, threshold

        bboxes = self._targets["bboxes"].to(device=p2.device, dtype=p2.dtype)
        batch_idx = self._targets["batch_idx"].to(device=p2.device, dtype=torch.long)
        image_h, image_w = self._image_size
        yy, xx = torch.meshgrid(
            torch.arange(height, device=p2.device, dtype=p2.dtype),
            torch.arange(width, device=p2.device, dtype=p2.dtype),
            indexing="ij",
        )

        for b in range(batch):
            boxes = bboxes[batch_idx == b]
            if boxes.numel() == 0:
                continue

            cx = boxes[:, 0] * width
            cy = boxes[:, 1] * height
            bw = (boxes[:, 2] * width).clamp_min(0.25)
            bh = (boxes[:, 3] * height).clamp_min(0.25)
            object_size = torch.sqrt((boxes[:, 2] * image_w) * (boxes[:, 3] * image_h))
            alpha = torch.where(
                object_size <= 8,
                4.0,
                torch.where(object_size <= 16, 6.0, torch.where(object_size <= 32, 8.0, 10.0)),
            ).to(dtype=p2.dtype)
            sx = (bw / alpha).clamp_min(0.25)
            sy = (bh / alpha).clamp_min(0.25)

            gaussian_sum = p2.new_zeros((height, width))
            for start in range(0, len(boxes), 64):
                sl = slice(start, start + 64)
                dx = (xx.unsqueeze(0) - cx[sl, None, None]) / sx[sl, None, None]
                dy = (yy.unsqueeze(0) - cy[sl, None, None]) / sy[sl, None, None]
                norm = 1.0 / (2.0 * math.pi * sx[sl, None, None] * sy[sl, None, None])
                gaussian_sum.add_((norm * torch.exp(-0.5 * (dx.square() + dy.square()))).sum(0))

            gaussian_sum.div_(gaussian_sum.amax().clamp_min(1e-6))
            th = gaussian_sum.mean()
            target[b, 0] = gaussian_sum + 0.5 * (gaussian_sum > th).to(gaussian_sum.dtype)
            threshold[b, 0, 0, 0] = th

        return target, threshold

    def forward(self, features):
        p2, p3, p4 = features

        f4 = self.conv4(self.p4_proj(p4))
        logits4 = self.map4(f4)
        f3 = self.conv3(self.p3_proj(p3) + self.up4(f4) + F.interpolate(logits4, p3.shape[-2:]))
        logits3 = self.map3(f3)
        f2 = self.conv2(self.p2_proj(p2) + self.up3(f3) + F.interpolate(logits3, p2.shape[-2:]))
        logits2 = self.map2(f2)

        map2 = torch.sigmoid(logits2)
        attended = self.cbam(p2 * (1.0 + map2))
        output = p2 + self.effective_gamma * (attended - p2)

        if self.training and self._targets is not None:
            target, threshold = self._build_gaussian_target(p2)
            pred2 = map2
            pred3 = torch.sigmoid(F.interpolate(logits3, p2.shape[-2:], mode="bilinear", align_corners=False))
            pred4 = torch.sigmoid(F.interpolate(logits4, p2.shape[-2:], mode="bilinear", align_corners=False))
            self._aux_loss = self.pred_gain * sum(
                self._weighted_mse(pred, target, threshold) for pred in (pred2, pred3, pred4)
            )

        return output
