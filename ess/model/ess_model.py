"""
Wrapper that combines a Hugging Face causal LM backbone with the ESM and EGM.

The wrapper:
  * exposes a ``forward`` that returns standard LM logits, hidden states from
    every layer, the ESM epistemic embedding/confidence, and gated logits;
  * supports parameter-group freezing for the three training phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ess.model.esm import EpistemicStateModule
from ess.model.egm import EpistemicGatingMechanism


@dataclass
class ESSOutput:
    logits: torch.Tensor                   # [B, T, V]
    gated_logits: torch.Tensor             # [B, T, V]
    epistemic: torch.Tensor                # [B, T, d_e]
    confidence: torch.Tensor               # [B, T]
    hidden_states: List[torch.Tensor]      # length L of [B, T, d_model]
    loss_lm: Optional[torch.Tensor] = None


class ESSModel(nn.Module):
    """ESS = backbone + ESM + EGM."""

    def __init__(
        self,
        backbone_name: str,
        d_e: int = 64,
        tau: float = 0.45,
        beta: float = 2.5,
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
        device_map: Optional[str] = "auto",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_name,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            output_hidden_states=True,
        )
        self.backbone.config.output_hidden_states = True

        cfg = self.backbone.config
        # Hidden states from HF include the embedding layer output -> L+1 tensors.
        self.num_layers = cfg.num_hidden_layers
        self.d_model = cfg.hidden_size

        self.esm = EpistemicStateModule(
            num_layers=self.num_layers,
            d_model=self.d_model,
            d_e=d_e,
        )
        self.egm = EpistemicGatingMechanism(
            vocab_size=cfg.vocab_size,
            tau=tau,
            beta=beta,
            tokenizer=self.tokenizer,
        )

    # ---------- freezing helpers ----------

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def freeze_esm_projections(self, freeze: bool = True) -> None:
        for name, p in self.esm.named_parameters():
            if name.startswith(("proj_", "layer_norms", "layer_logits")):
                p.requires_grad = not freeze

    def freeze_conf_head(self, freeze: bool = True) -> None:
        self.esm.conf_v.requires_grad = not freeze
        self.esm.conf_b.requires_grad = not freeze

    def freeze_egm(self, freeze: bool = True) -> None:
        self.egm.h_hedge.requires_grad = not freeze

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # ---------- forward ----------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> ESSOutput:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states is a tuple of (L+1) tensors: embeddings + per layer.
        # Drop the embedding output so we have exactly L tensors.
        hidden_states = list(out.hidden_states[1:])
        # Cast to ESM dtype for stability.
        hidden_states = [h.to(self.esm.proj_weight.dtype) for h in hidden_states]

        esm_out = self.esm(hidden_states)
        gated_logits = self.egm(out.logits, esm_out["confidence"])

        return ESSOutput(
            logits=out.logits,
            gated_logits=gated_logits,
            epistemic=esm_out["epistemic"],
            confidence=esm_out["confidence"],
            hidden_states=hidden_states,
            loss_lm=out.loss,
        )

    # ---------- saving ----------

    def save_ess_modules(self, path: str) -> None:
        """Save only ESM+EGM weights (the lightweight delta)."""
        torch.save(
            {"esm": self.esm.state_dict(), "egm": self.egm.state_dict()},
            path,
        )

    def load_ess_modules(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu")
        self.esm.load_state_dict(sd["esm"])
        self.egm.load_state_dict(sd["egm"])
