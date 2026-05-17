"""
Epistemic State Module (ESM).

read-only attachment that taps the residual stream at every layer of a
transformer backbone and produces (i) a per-token epistemic embedding e_t and
(ii) a per-token confidence score c_t in [0, 1].
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpistemicStateModule(nn.Module):
    """ESM as defined in Section 3.2.

    Parameters
    ----------
    num_layers : int
        Number of transformer layers L in the backbone.
    d_model : int
        Hidden size of the backbone.
    d_e : int
        Epistemic embedding dimension (paper uses 64).
    """

    def __init__(self, num_layers: int, d_model: int, d_e: int = 64):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_e = d_e

        # Per-layer linear projection W_e^l h_t^l + b_e^l followed by LayerNorm.
        # Implemented as one parameter tensor to avoid Python-side loops.
        self.proj_weight = nn.Parameter(torch.empty(num_layers, d_e, d_model))
        self.proj_bias = nn.Parameter(torch.zeros(num_layers, d_e))
        for l in range(num_layers):
            nn.init.kaiming_uniform_(self.proj_weight[l], a=math.sqrt(5))

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_e) for _ in range(num_layers)]
        )

        # Layer-aggregation weights a in R^L (softmaxed at use time).
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

        # Confidence head v in R^{d_e}, scalar bias b.
        self.conf_v = nn.Parameter(torch.zeros(d_e))
        nn.init.normal_(self.conf_v, std=0.02)
        self.conf_b = nn.Parameter(torch.zeros(1))

    @property
    def layer_weights(self) -> torch.Tensor:
        """Softmax-normalised layer weights alpha_l."""
        return F.softmax(self.layer_logits, dim=0)

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        return_per_layer: bool = False,
    ) -> dict:
        """Compute epistemic embeddings and confidence scores.

        Parameters
        ----------
        hidden_states : list of [B, T, d_model]
            One tensor per transformer layer (excluding embedding output).
        return_per_layer : bool
            Whether to return the per-layer projections (useful for analysis).

        Returns
        -------
        dict with keys:
            'epistemic' : [B, T, d_e]      eps_t
            'confidence': [B, T]           c_t
            'per_layer' : [B, T, L, d_e]   only if return_per_layer
        """
        assert len(hidden_states) == self.num_layers, (
            f"Expected {self.num_layers} hidden states, got {len(hidden_states)}"
        )

        # Stack: [B, T, L, d_model]
        H = torch.stack(hidden_states, dim=2)
        B, T, L, _ = H.shape

        # Project per layer: e_hat_t^l = LayerNorm(W_e^l h_t^l + b_e^l)
        # Use einsum for efficiency: [B, T, L, d_e]
        projected = torch.einsum("btld,led->btle", H, self.proj_weight)
        projected = projected + self.proj_bias  # broadcast over B, T

        # Per-layer LayerNorm (loop is small: L ~ 32)
        normed = torch.empty_like(projected)
        for l in range(self.num_layers):
            normed[:, :, l, :] = self.layer_norms[l](projected[:, :, l, :])

        # Aggregate with softmax-weighted sum across layers.
        alpha = self.layer_weights.view(1, 1, L, 1)
        epistemic = (alpha * normed).sum(dim=2)  # [B, T, d_e]

        # Confidence head: c_t = sigmoid(v^T eps_t + b)
        logits = epistemic @ self.conf_v + self.conf_b
        confidence = torch.sigmoid(logits)

        out = {"epistemic": epistemic, "confidence": confidence}
        if return_per_layer:
            out["per_layer"] = normed
        return out

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
