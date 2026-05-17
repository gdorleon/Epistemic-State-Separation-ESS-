"""
Negative-sample construction.

Two modes:
  (1) counterfactual: swap object o with o' sampled from the same semantic
      type (e.g. capital -> some other city).
  (2) self-generated: prompt the backbone for completions, parse atomic claims,
      verify against Wikidata, keep rejected claims as hard negatives.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ess.data.wikidata import Triple, RELATIONS
from ess.data.verbalize import verbalize_triple, verbalize_counterfactual


# ---------- counterfactual ----------

def build_object_pools(triples: List[Triple]) -> Dict[Tuple[str, str], List[str]]:
    """Group object surface strings by (relation, object_type)."""
    pools: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for t in triples:
        pools[(t.relation, t.obj_type)].append(t.obj)
    # dedup
    for k, v in pools.items():
        pools[k] = list(dict.fromkeys(v))
    return pools


def sample_counterfactuals(
    triple: Triple,
    pools: Dict[Tuple[str, str], List[str]],
    k: int = 8,
    rng: Optional[random.Random] = None,
) -> List[str]:
    rng = rng or random.Random()
    pool = pools.get((triple.relation, triple.obj_type), [])
    pool = [o for o in pool if o != triple.obj]
    if len(pool) < k:
        # fall back to any object of same type
        all_same_type = [
            o for (r, t), os_ in pools.items() if t == triple.obj_type
            for o in os_ if o != triple.obj
        ]
        pool = pool + all_same_type
    rng.shuffle(pool)
    fakes = pool[:k]
    return [verbalize_counterfactual(triple, f) for f in fakes]


# ---------- self-generated ----------

ATOMIC_CLAIM_RE = re.compile(r"([^.!?]*[.!?])", re.MULTILINE)


def split_atomic_claims(text: str) -> List[str]:
    """Naive sentence splitter for atomic-claim extraction."""
    return [c.strip() for c in ATOMIC_CLAIM_RE.findall(text) if c.strip()]


def verify_claim_against_wikidata(
    claim: str,
    true_triples_index: Dict[str, List[Triple]],
) -> Optional[bool]:
    """Cheap heuristic verifier: look for any (s, o) pair from true triples
    appearing together in the claim. Returns:
        True   -> claim consistent with a known true triple,
        False  -> contains a subject from known triples but no matching object,
        None   -> insufficient evidence to judge.
    """
    cl = claim.lower()
    for subj, triples in true_triples_index.items():
        if subj.lower() not in cl:
            continue
        for t in triples:
            if t.obj.lower() in cl:
                return True
        # subject present but no known object found -> probable hallucination
        return False
    return None


def mine_self_negatives(
    model,
    tokenizer,
    prompts: Iterable[str],
    true_triples: List[Triple],
    device: str = "cuda",
    max_new_tokens: int = 80,
    temperature: float = 0.9,
) -> List[str]:
    """Generate completions, extract atomic claims, return those that fail
    Wikidata verification.
    """
    import torch

    index: Dict[str, List[Triple]] = defaultdict(list)
    for t in true_triples:
        index[t.subject].append(t)

    hard_negatives: List[str] = []
    model.eval()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for claim in split_atomic_claims(text):
            verdict = verify_claim_against_wikidata(claim, index)
            if verdict is False:
                hard_negatives.append(claim)
    return hard_negatives


def save_jsonl(path: str, records: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
