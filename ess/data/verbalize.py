"""
Template-based verbalisation of (s, r, o) triples to natural-language claims.
"""

from __future__ import annotations

from typing import Dict

from ess.data.wikidata import Triple


TEMPLATES: Dict[str, list] = {
    "place of birth":          ["{s} was born in {o}.", "The birthplace of {s} is {o}."],
    "place of death":          ["{s} died in {o}.", "The place of death of {s} is {o}."],
    "country of citizenship":  ["{s} is a citizen of {o}.", "{s} holds citizenship of {o}."],
    "capital":                 ["The capital of {s} is {o}.", "{o} is the capital of {s}."],
    "author":                  ["{s} was written by {o}.", "The author of {s} is {o}."],
    "director":                ["{s} was directed by {o}.", "{o} directed {s}."],
    "composer":                ["{s} was composed by {o}.", "{o} composed {s}."],
    "occupation":              ["{s} works as a {o}.", "The occupation of {s} is {o}."],
    "founded by":              ["{s} was founded by {o}.", "{o} founded {s}."],
    "headquarters location":   ["{s} is headquartered in {o}."],
    "performer":               ["{s} was performed by {o}."],
    "manufacturer":            ["{s} is manufactured by {o}."],
    "part of":                 ["{s} is part of {o}."],
    "country of origin":       ["{s} originates from {o}."],
    "diplomatic relation":     ["{s} maintains diplomatic relations with {o}."],
    "date of birth":           ["{s} was born on {o}.", "The date of birth of {s} is {o}."],
    "date of death":           ["{s} died on {o}.", "The date of death of {s} is {o}."],
    "inception":               ["{s} was founded in {o}.", "{s} began in {o}."],
    "publication date":        ["{s} was published in {o}."],
    "notable work":            ["A notable work of {s} is {o}."],
}


def verbalize_triple(t: Triple, idx: int = 0) -> str:
    """Verbalise a triple using a deterministic template (selected by idx)."""
    tpls = TEMPLATES.get(t.relation, ["{s} is related to {o} via " + t.relation + "."])
    return tpls[idx % len(tpls)].format(s=t.subject, o=t.obj)


def verbalize_counterfactual(t: Triple, fake_obj: str, idx: int = 0) -> str:
    """Verbalise (s, r, o') where o' is a counterfactual object."""
    tpls = TEMPLATES.get(t.relation, ["{s} is related to {o} via " + t.relation + "."])
    return tpls[idx % len(tpls)].format(s=t.subject, o=fake_obj)
