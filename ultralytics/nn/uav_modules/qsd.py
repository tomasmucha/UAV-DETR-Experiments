"""Query-level semantic dictionary modules for UAV-DETR.

The dictionary-retrieval idea is inspired by YOLO-RD (ICLR 2025), but this is
a query-level implementation rather than a copy of its dense P5 feature block.
Each decoder query retrieves a compact semantic prototype and receives a
bounded residual update. The same module can run on its own or consume
detached localization uncertainty supplied by D-FINE FDR (ICLR 2025).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.head import RTDETRDecoder
from ..modules.transformer import DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from ..modules.utils import inverse_sigmoid

__all__ = ["QuerySemanticDictionary", "RTDETRQSDDecoder", "normalized_distribution_entropy"]


def normalized_distribution_entropy(logits: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Return mean normalized entropy over four FDR edge distributions.

    Args:
        logits: Tensor shaped ``[batch, queries, 4 * num_bins]``.
        num_bins: Number of bins in each edge distribution.

    Returns:
        Tensor shaped ``[batch, queries, 1]`` with values in ``[0, 1]``.
    """
    if logits.shape[-1] != 4 * num_bins:
        raise ValueError(f"Expected {4 * num_bins} FDR logits, got {logits.shape[-1]}")
    probabilities = logits.float().reshape(*logits.shape[:-1], 4, num_bins).softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
    entropy = entropy.mean(dim=-1, keepdim=True) / math.log(num_bins)
    return entropy.clamp_(0.0, 1.0).to(dtype=logits.dtype)


class QuerySemanticDictionary(nn.Module):
    """Retrieve dataset-level semantic prototypes for decoder queries.

    FDR uncertainty is optional so this remains an independently executable
    structural module. When uncertainty is provided, it amplifies (rather than
    replaces) the base dictionary residual for ambiguous queries. The gate is
    detached to keep the information flow directed from localization to
    semantic compensation.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_prototypes: int = 64,
        temperature: float = 0.07,
        initial_scale: float = 0.10,
        max_scale: float = 0.50,
        uncertainty_gain: float = 1.0,
    ):
        super().__init__()
        if hidden_dim <= 0 or num_prototypes <= 1:
            raise ValueError(
                f"Expected positive hidden_dim and num_prototypes > 1, got {hidden_dim=}, {num_prototypes=}"
            )
        if temperature <= 0:
            raise ValueError(f"Expected temperature > 0, got {temperature}")
        if max_scale <= 0 or not 0 < initial_scale < max_scale:
            raise ValueError(f"Expected 0 < initial_scale < max_scale, got {initial_scale=}, {max_scale=}")
        if uncertainty_gain < 0:
            raise ValueError(f"Expected uncertainty_gain >= 0, got {uncertainty_gain}")

        self.hidden_dim = int(hidden_dim)
        self.num_prototypes = int(num_prototypes)
        self.temperature = float(temperature)
        self.max_scale = float(max_scale)
        self.uncertainty_gain = float(uncertainty_gain)

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.dictionary_keys = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.dictionary_values = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.retrieved_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        scale_ratio = initial_scale / max_scale
        self.raw_scale = nn.Parameter(torch.tensor(math.log(scale_ratio / (1.0 - scale_ratio))))

        nn.init.normal_(self.dictionary_keys, std=0.02)
        nn.init.normal_(self.dictionary_values, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=1e-3)
        nn.init.zeros_(self.output_proj.bias)

    @property
    def effective_scale(self) -> torch.Tensor:
        """Bounded scalar controlling the dictionary residual."""
        return self.max_scale * self.raw_scale.sigmoid()

    def forward(self, query: torch.Tensor, uncertainty: torch.Tensor | None = None) -> torch.Tensor:
        """Enhance ``[batch, queries, hidden_dim]`` decoder embeddings."""
        if query.ndim != 3 or query.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected query shape [B, Q, {self.hidden_dim}], got {tuple(query.shape)}")

        normalized_query = F.normalize(self.query_norm(query), dim=-1)
        normalized_keys = F.normalize(self.dictionary_keys, dim=-1)
        attention = torch.matmul(normalized_query, normalized_keys.t()) / self.temperature
        retrieved = torch.matmul(attention.softmax(dim=-1), self.dictionary_values)
        delta = self.output_proj(F.gelu(self.retrieved_norm(retrieved)))

        gate = 1.0
        if uncertainty is not None:
            if uncertainty.ndim == 2:
                uncertainty = uncertainty.unsqueeze(-1)
            if uncertainty.shape != query.shape[:2] + (1,):
                raise ValueError(
                    f"Expected uncertainty shape {query.shape[:2] + (1,)}, got {tuple(uncertainty.shape)}"
                )
            gate = 1.0 + self.uncertainty_gain * uncertainty.detach().to(dtype=query.dtype).clamp(0.0, 1.0)

        return query + self.effective_scale.to(dtype=query.dtype) * gate * delta


class QuerySemanticTransformerDecoder(DeformableTransformerDecoder):
    """Standard RT-DETR refinement with a shared query semantic dictionary."""

    def __init__(self, hidden_dim, decoder_layer, num_layers, query_dictionary, eval_idx=-1):
        super().__init__(hidden_dim, decoder_layer, num_layers, eval_idx)
        self.query_dictionary = query_dictionary

    def forward(
        self,
        embed,
        refer_bbox,
        feats,
        shapes,
        bbox_head,
        score_head,
        pos_mlp,
        attn_mask=None,
        padding_mask=None,
    ):
        """Apply dictionary retrieval after each layer's localization prediction."""
        output = embed
        dec_bboxes, dec_cls = [], []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()

        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))
            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))
            output = self.query_dictionary(output)

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class RTDETRQSDDecoder(RTDETRDecoder):
    """RT-DETR head with an independently executable query semantic dictionary."""

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,
        nq=300,
        ndp=4,
        nh=8,
        ndl=6,
        d_ffn=1024,
        eval_idx=-1,
        dropout=0.0,
        act=nn.ReLU(),
        nd=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        qsd_num_prototypes=64,
        qsd_temperature=0.07,
        qsd_initial_scale=0.10,
        qsd_max_scale=0.50,
    ):
        super().__init__(
            nc,
            ch,
            hd,
            nq,
            ndp,
            nh,
            ndl,
            d_ffn,
            eval_idx,
            dropout,
            act,
            nd,
            label_noise_ratio,
            box_noise_scale,
            learnt_init_query,
        )
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        dictionary = QuerySemanticDictionary(
            hd,
            qsd_num_prototypes,
            qsd_temperature,
            qsd_initial_scale,
            qsd_max_scale,
            uncertainty_gain=0.0,
        )
        self.decoder = QuerySemanticTransformerDecoder(hd, decoder_layer, ndl, dictionary, eval_idx)
