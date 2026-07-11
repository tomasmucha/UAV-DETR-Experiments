"""Fine-grained Distribution Refinement adapted from D-FINE (ICLR 2025).

This is a focused adaptation for UAV-DETR's Ultralytics RT-DETR head. It keeps
the existing encoder, query selection and denoising paths, while replacing
iterative four-coordinate regression with residual distribution refinement.

Portions Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Upstream: https://github.com/Peterande/D-FINE (Apache-2.0)
This adaptation changes the module interface and training-output contract.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_

from ..modules.head import RTDETRDecoder
from ..modules.transformer import DeformableTransformerDecoderLayer, MLP
from ..modules.utils import inverse_sigmoid

__all__ = ['RTDETRFDRDecoder']


def _weighting_function(reg_max=32, up=0.5, reg_scale=4.0):
    """Build D-FINE's non-uniform bin-to-offset weighting function."""
    if reg_max % 2:
        raise ValueError('FDR reg_max must be even.')
    upper_bound1 = abs(up * reg_scale)
    upper_bound2 = upper_bound1 * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left = [-step**i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right = [step**i - 1 for i in range(1, reg_max // 2)]
    return torch.tensor([-upper_bound2, *left, 0.0, *right, upper_bound2], dtype=torch.float32)


def _distance2bbox(points, distance, reg_scale):
    """Decode left/top/right/bottom offsets around cxcywh reference boxes."""
    scale = abs(float(reg_scale))
    x1 = points[..., 0] - (0.5 * scale + distance[..., 0]) * (points[..., 2] / scale)
    y1 = points[..., 1] - (0.5 * scale + distance[..., 1]) * (points[..., 3] / scale)
    x2 = points[..., 0] + (0.5 * scale + distance[..., 2]) * (points[..., 2] / scale)
    y2 = points[..., 1] + (0.5 * scale + distance[..., 3]) * (points[..., 3] / scale)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), -1)


class FDRIntegral(nn.Module):
    """Convert a discrete edge-offset distribution into continuous offsets."""

    def __init__(self, reg_max=32, up=0.5, reg_scale=4.0):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('project', _weighting_function(reg_max, up, reg_scale), persistent=True)

    def forward(self, logits):
        shape = logits.shape
        prob = F.softmax(logits.reshape(-1, self.reg_max + 1), dim=-1)
        offsets = F.linear(prob, self.project.to(dtype=prob.dtype).reshape(1, -1))
        return offsets.reshape(*shape[:-1], 4)


class FineGrainedDeformableTransformerDecoder(nn.Module):
    """RT-DETR decoder with D-FINE-style residual distribution refinement."""

    def __init__(self, hidden_dim, decoder_layer, num_layers, integral, reg_scale=4.0, eval_idx=-1):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.integral = integral
        self.reg_scale = reg_scale

    def forward(self, embed, refer_bbox, feats, shapes, bbox_head, score_head, pos_mlp, pre_bbox_head,
                attn_mask=None, padding_mask=None):
        output = embed
        dec_bboxes, dec_cls, dec_corners, dec_refs = [], [], [], []
        accumulated_corners = None
        refer_bbox = refer_bbox.sigmoid()
        initial_bbox = None
        pre_bboxes = pre_scores = None

        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))
            if i == 0:
                pre_bboxes = torch.sigmoid(pre_bbox_head(output) + inverse_sigmoid(refer_bbox))
                pre_scores = score_head[0](output)
                initial_bbox = pre_bboxes.detach()

            corners = bbox_head[i](output)
            corners = corners if accumulated_corners is None else corners + accumulated_corners
            refined_bbox = _distance2bbox(initial_bbox, self.integral(corners), self.reg_scale)

            if self.training or i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                dec_corners.append(corners)
                dec_refs.append(initial_bbox)
                if not self.training:
                    break

            accumulated_corners = corners
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return (torch.stack(dec_bboxes), torch.stack(dec_cls), torch.stack(dec_corners),
                torch.stack(dec_refs), pre_bboxes, pre_scores)


class RTDETRFDRDecoder(RTDETRDecoder):
    """UAV-DETR head combining the existing RT-DETR path with FDR localization."""

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4, nh=8, ndl=6,
                 d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(), nd=100,
                 label_noise_ratio=0.5, box_noise_scale=1.0, learnt_init_query=False,
                 reg_max=32, reg_scale=4.0, up=0.5):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx, dropout, act, nd,
                         label_noise_ratio, box_noise_scale, learnt_init_query)
        self.reg_max = reg_max
        self.reg_scale = float(reg_scale)
        self.up = float(up)
        self.fdr_integral = FDRIntegral(reg_max, up, reg_scale)
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = FineGrainedDeformableTransformerDecoder(
            hd, decoder_layer, ndl, self.fdr_integral, reg_scale, eval_idx)
        self.pre_bbox_head = MLP(hd, hd, 4, num_layers=3)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hd, hd, 4 * (reg_max + 1), num_layers=3) for _ in range(ndl)])

        constant_(self.pre_bbox_head.layers[-1].weight, 0.)
        constant_(self.pre_bbox_head.layers[-1].bias, 0.)
        for head in self.dec_bbox_head:
            constant_(head.layers[-1].weight, 0.)
            constant_(head.layers[-1].bias, 0.)

    def forward(self, x, batch=None):
        """Return standard RT-DETR outputs plus FDR distributions and references."""
        from ultralytics.models.utils.ops import get_cdn_group

        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch, self.nc, self.num_queries, self.denoising_class_embed.weight,
            self.num_denoising, self.label_noise_ratio, self.box_noise_scale, self.training)
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        dec_bboxes, dec_scores, dec_corners, dec_refs, pre_bboxes, pre_scores = self.decoder(
            embed, refer_bbox, feats, shapes, self.dec_bbox_head, self.dec_score_head,
            self.query_pos_head, self.pre_bbox_head, attn_mask=attn_mask)
        outputs = (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, dec_corners, dec_refs,
                   pre_bboxes, pre_scores)
        if self.training:
            return outputs
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, outputs)
