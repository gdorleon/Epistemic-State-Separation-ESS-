"""
ECO training dataset.

Each example contains:
  * one positive claim c+,
  * one in-batch positive c++ (a different verified fact about the same entity),
  * K counterfactual / hard negatives c1-, ..., cK-.
Verbalised and tokenized for the backbone.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from ess.data.wikidata import Triple
from ess.data.verbalize import verbalize_triple
from ess.data.negatives import build_object_pools, sample_counterfactuals


class ECODataset(Dataset):
    """Yields claim_span-tagged examples for InfoNCE training."""

    def __init__(
        self,
        triples_path: str,
        self_negatives_path: Optional[str] = None,
        k_negatives: int = 8,
        seed: int = 0,
    ):
        super().__init__()
        self.rng = random.Random(seed)
        self.k_negatives = k_negatives

        triples = [Triple(**json.loads(l)) for l in Path(triples_path).read_text().splitlines()]
        # Group triples by subject for c++ selection.
        self.by_subject: Dict[str, List[Triple]] = defaultdict(list)
        for t in triples:
            self.by_subject[t.subject].append(t)
        self.triples = [t for t in triples if len(self.by_subject[t.subject]) >= 2]

        self.pools = build_object_pools(triples)

        self.self_negatives: List[str] = []
        if self_negatives_path and Path(self_negatives_path).exists():
            for line in Path(self_negatives_path).read_text().splitlines():
                rec = json.loads(line)
                if isinstance(rec, dict) and "text" in rec:
                    self.self_negatives.append(rec["text"])
                elif isinstance(rec, str):
                    self.self_negatives.append(rec)

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict:
        t_pos = self.triples[idx]

        # c+: verbalise t_pos
        c_pos = verbalize_triple(t_pos)

        # c++: another verified fact about same subject
        other = [x for x in self.by_subject[t_pos.subject] if x is not t_pos]
        t_other = self.rng.choice(other) if other else t_pos
        c_pos_pos = verbalize_triple(t_other)

        # negatives: counterfactual + (optionally) self-generated
        cfs = sample_counterfactuals(t_pos, self.pools, k=self.k_negatives, rng=self.rng)
        if self.self_negatives and self.rng.random() < 0.5:
            # mix in 1-2 hard self-negatives
            n_hard = min(2, len(self.self_negatives), self.k_negatives)
            cfs[:n_hard] = self.rng.sample(self.self_negatives, n_hard)

        return {
            "c_pos": c_pos,
            "c_pos_pos": c_pos_pos,
            "negatives": cfs[: self.k_negatives],
            "subject": t_pos.subject,
        }


class ECOCollator:
    """Tokenise and pad a batch of ECO examples, recording the claim span
    (here: the entire tokenised sequence, i.e. start_token_offset:end_token_offset)
    that the ESM should pool over to produce eps(c)."""

    def __init__(self, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _tokenise(self, texts: List[str]) -> dict:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, batch: List[dict]) -> dict:
        c_pos = [b["c_pos"] for b in batch]
        c_pp = [b["c_pos_pos"] for b in batch]
        negatives_flat = [n for b in batch for n in b["negatives"]]
        K = len(batch[0]["negatives"])

        pos_tok = self._tokenise(c_pos)
        pp_tok = self._tokenise(c_pp)
        neg_tok = self._tokenise(negatives_flat)

        return {
            "pos_input_ids": pos_tok["input_ids"],
            "pos_attention_mask": pos_tok["attention_mask"],
            "pp_input_ids": pp_tok["input_ids"],
            "pp_attention_mask": pp_tok["attention_mask"],
            "neg_input_ids": neg_tok["input_ids"],
            "neg_attention_mask": neg_tok["attention_mask"],
            "K": K,
            "batch_size": len(batch),
        }
