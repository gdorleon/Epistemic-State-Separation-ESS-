"""
Epistemic Gating Mechanism (EGM).

z~_t = z_t + max(0, tau - c_t) * beta * h_hedge

h_hedge is a learned per-vocabulary-token vector that captures the linguistic
direction of uncertainty expression. It is up-weighted (positive entries) on
hedging tokens and down-weighted on overconfident assertion tokens.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# Default seed lists used for h_hedge initialisation.
HEDGE_TOKENS = [
    "approximately", "roughly", "around", "likely", "probably",
    "perhaps", "maybe", "possibly", "might", "may",
    "could", "seems", "appears", "suggests",
    "I", "believe", "think", "not", "certain", "sure",
    "according", "based", "reportedly", "allegedly",
    "uncertain", "unclear", "unknown",
]

ASSERTIVE_TOKENS = [
    "definitely", "certainly", "obviously", "clearly", "undoubtedly",
    "absolutely", "always", "never", "must", "exactly",
]


class EpistemicGatingMechanism(nn.Module):
    """EGM as defined in Section 3.4.

    The hedging vector is a free parameter in R^|V|, learned in Phase 3.
    """

    def __init__(
        self,
        vocab_size: int,
        tau: float = 0.45,
        beta: float = 2.5,
        tokenizer=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tau = tau
        self.beta = beta

        # h_hedge in R^|V|; learnable.
        h = torch.zeros(vocab_size)
        if tokenizer is not None:
            h = self._init_from_tokenizer(h, tokenizer)
        self.h_hedge = nn.Parameter(h)

    @staticmethod
    def _init_from_tokenizer(h: torch.Tensor, tokenizer) -> torch.Tensor:
        """Initialise h_hedge by giving small positive/negative weight to seed
        tokens. This is a soft prior, not a hard constraint."""
        with torch.no_grad():
            for tok in HEDGE_TOKENS:
                for variant in (tok, " " + tok, tok.capitalize(), " " + tok.capitalize()):
                    ids = tokenizer.encode(variant, add_special_tokens=False)
                    for i in ids:
                        if 0 <= i < h.numel():
                            h[i] += 0.5
            for tok in ASSERTIVE_TOKENS:
                for variant in (tok, " " + tok, tok.capitalize(), " " + tok.capitalize()):
                    ids = tokenizer.encode(variant, add_special_tokens=False)
                    for i in ids:
                        if 0 <= i < h.numel():
                            h[i] -= 0.5
        return h

    def forward(
        self,
        logits: torch.Tensor,
        confidence: torch.Tensor,
        tau: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply the gate.

        Parameters
        ----------
        logits : [B, T, V] or [B, V]
        confidence : [B, T] or [B]
        """
        tau = self.tau if tau is None else tau
        beta = self.beta if beta is None else beta

        # gate = max(0, tau - c) * beta -> shape matching logits' batch/time dims.
        gate = (tau - confidence).clamp_min(0.0) * beta
        # broadcast gate to vocab dim
        gate = gate.unsqueeze(-1)
        return logits + gate * self.h_hedge

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, tau={self.tau}, beta={self.beta}"
