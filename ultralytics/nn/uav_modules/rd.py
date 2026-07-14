"""Retriever-Dictionary feature module for UAV-DETR.

Adapted from Tsui et al., "YOLO-RD: Introducing Relevant and Compact
Explicit Knowledge to YOLO by Retriever-Dictionary" (ICLR 2025) and the
authors' MIT-licensed implementation at https://github.com/henrytsui000/YOLO.

The module retrieves a compact representation, extracts spatial context,
normalizes each position across dictionary atoms, projects it through a
learnable dictionary, and fuses the result with the input feature.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..modules.conv import Conv

__all__ = ["RetrieverDictionary"]


class RetrieverDictionary(nn.Module):
    """Dataset-level semantic retrieval and dictionary projection.

    Args:
        channels: Input and output channels.
        atoms: Width of the retrieved dictionary representation.
        alpha: Weight of the dictionary branch in the residual mixture.
        eps: Numerical stability term for position normalization.
    """

    def __init__(self, channels: int, atoms: int = 512, alpha: float = 0.8, eps: float = 1e-5):
        super().__init__()
        if channels <= 0 or atoms <= 1:
            raise ValueError(f"Expected positive channels and atoms > 1, got {channels=} and {atoms=}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Expected alpha in [0, 1], got {alpha}")

        self.retriever = Conv(channels, atoms, 1)
        self.global_information = Conv(atoms, atoms, 5, g=atoms, act=False)
        self.dictionary = Conv(atoms, channels, 1, act=False)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def _position_normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + self.eps)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        retrieved = self.retriever(residual)
        retrieved = self.global_information(retrieved)
        retrieved = self._position_normalize(retrieved)
        retrieved = self.dictionary(retrieved)
        return self.alpha * retrieved + (1.0 - self.alpha) * residual
