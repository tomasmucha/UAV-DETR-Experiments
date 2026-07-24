"""Feature suppression and enhancement for the UAV-DETR feature pyramid."""

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import RepC3
from ..modules.conv import Conv

__all__ = ["FSEM", "FSEMSelect"]


class _DownAlign(nn.Module):
    """Align a shallow feature to the next deeper pyramid level."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.down = Conv(c1, c2, 3, 2)
        self.refine = RepC3(c2, c2, n=1, e=0.5)

    def forward(self, x: torch.Tensor, size) -> torch.Tensor:
        x = self.refine(self.down(x))
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x


class _UpAlign(nn.Module):
    """Align a deep feature to the next shallower pyramid level."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.project = Conv(c1, c2, 1)
        self.refine = RepC3(c2, c2, n=1, e=0.5)

    def forward(self, x: torch.Tensor, size) -> torch.Tensor:
        x = F.interpolate(self.project(x), size=size, mode="bilinear", align_corners=False)
        return self.refine(x)


class FSEM(nn.Module):
    """Paper-derived FSEM adaptation for a three-level ``[P3, P4, P5]`` pyramid.

    The topology follows FSENet's two ordered operations:

    1. FEM adds transformed shallow features to the deeper P4/P5 levels.
    2. FSM subtracts transformed non-local deep features from P4/P3.

    This is an adaptation rather than an official source-code reproduction. It
    uses the repository's RepC3 blocks for scale alignment and bounded,
    non-negative learnable coefficients to preserve a recoverable identity
    path. IKLD and the paper's DSConv are deliberately excluded so FSEM remains
    the only new variable in the structural ablation.
    """

    def __init__(
        self,
        channels: Sequence[int],
        suppress_init: float = 0.1,
        enhance_init: float = 0.1,
        coefficient_max: float = 1.0,
    ):
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"FSEM expects [P3, P4, P5] channels, got {channels}")
        if coefficient_max <= 0.0:
            raise ValueError(f"coefficient_max must be positive, got {coefficient_max}")
        for name, value in (("suppress_init", suppress_init), ("enhance_init", enhance_init)):
            if not 0.0 < value < coefficient_max:
                raise ValueError(
                    f"{name} must satisfy 0 < {name} < coefficient_max, "
                    f"got {value} and {coefficient_max}"
                )

        c3, c4, c5 = (int(c) for c in channels)

        # FEM: P3 -> P4/P5 and P4 -> P5.
        self.p3_to_p4 = _DownAlign(c3, c4)
        self.p3_to_p5 = _DownAlign(c4, c5)
        self.p4_to_p5 = _DownAlign(c4, c5)

        # FSM: P5 -> P4/P3 and P4 -> P3.
        self.p5_to_p4 = _UpAlign(c5, c4)
        self.p5_to_p3 = _UpAlign(c4, c3)
        self.p4_to_p3 = _UpAlign(c4, c3)

        self.raw_suppress = nn.Parameter(
            torch.tensor(self._inverse_bounded_coefficient(suppress_init, coefficient_max))
        )
        self.raw_enhance = nn.Parameter(
            torch.tensor(self._inverse_bounded_coefficient(enhance_init, coefficient_max))
        )
        self.register_buffer("coefficient_max", torch.tensor(float(coefficient_max)))
        self.channels = (c3, c4, c5)

    @staticmethod
    def _inverse_bounded_coefficient(value: float, maximum: float) -> float:
        ratio = value / maximum
        return math.log(ratio / (1.0 - ratio))

    @property
    def suppress_coefficient(self) -> torch.Tensor:
        return self.coefficient_max * torch.sigmoid(self.raw_suppress)

    @property
    def enhance_coefficient(self) -> torch.Tensor:
        return self.coefficient_max * torch.sigmoid(self.raw_enhance)

    def _validate_features(self, features) -> None:
        if not isinstance(features, (list, tuple)) or len(features) != 3:
            raise TypeError("FSEM forward expects a three-tensor [P3, P4, P5] sequence")
        for index, (feature, channels) in enumerate(zip(features, self.channels)):
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                raise TypeError(f"FSEM level {index} must be a 4D tensor")
            if feature.shape[1] != channels:
                raise ValueError(
                    f"FSEM level {index} expected {channels} channels, got {feature.shape[1]}"
                )
        p3_size, p4_size, p5_size = (feature.shape[-2:] for feature in features)
        if not (
            p3_size[0] >= p4_size[0] >= p5_size[0]
            and p3_size[1] >= p4_size[1] >= p5_size[1]
        ):
            raise ValueError(
                f"FSEM expects descending P3/P4/P5 resolutions, got "
                f"{p3_size}, {p4_size}, {p5_size}"
            )

    def forward(self, features):
        self._validate_features(features)
        p3, p4, p5 = features

        # FEM: retain P3 and inject its detail into P4/P5, while P4 also
        # contributes its scale-appropriate information to P5.
        p3_at_p4 = self.p3_to_p4(p3, p4.shape[-2:])
        p3_at_p5 = self.p3_to_p5(p3_at_p4, p5.shape[-2:])
        p4_at_p5 = self.p4_to_p5(p4, p5.shape[-2:])
        enhance = self.enhance_coefficient
        p4_enhanced = p4 + enhance * p3_at_p4
        p5_enhanced = p5 + enhance * (p4_at_p5 + p3_at_p5)

        # FSM: remove non-local deep responses that can cover weak tiny-object
        # evidence after multi-level fusion.
        p5_at_p4 = self.p5_to_p4(p5_enhanced, p4.shape[-2:])
        p5_at_p3 = self.p5_to_p3(p5_at_p4, p3.shape[-2:])
        p4_at_p3 = self.p4_to_p3(p4_enhanced, p3.shape[-2:])
        suppress = self.suppress_coefficient
        p4_suppressed = p4_enhanced - suppress * p5_at_p4
        p3_suppressed = p3 - suppress * (p4_at_p3 + p5_at_p3)

        return p3_suppressed, p4_suppressed, p5_enhanced


class FSEMSelect(nn.Module):
    """Parameter-free graph adapter that selects one FSEM pyramid output."""

    def __init__(self, index: int, expected_channels: int):
        super().__init__()
        if index not in (0, 1, 2):
            raise ValueError(f"FSEMSelect index must be 0, 1, or 2, got {index}")
        self.index = int(index)
        self.expected_channels = int(expected_channels)

    def forward(self, features) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != 3:
            raise TypeError("FSEMSelect expects the three outputs returned by FSEM")
        output = features[self.index]
        if output.ndim != 4 or output.shape[1] != self.expected_channels:
            raise ValueError(
                f"FSEMSelect[{self.index}] expected a 4D tensor with "
                f"{self.expected_channels} channels, got {tuple(output.shape)}"
            )
        return output
