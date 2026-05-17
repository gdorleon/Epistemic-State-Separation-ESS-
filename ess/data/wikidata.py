"""
Wikidata triple sampling.

So, here this is a minimal implementation; production use should respect rate limits
and use a local Wikidata dump for large-scale extraction.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Curated relations: (P-id, surface name, expected object type)
RELATIONS = [
    ("P19", "place of birth", "location"),
    ("P20", "place of death", "location"),
    ("P27", "country of citizenship", "country"),
    ("P36", "capital", "city"),
    ("P50", "author", "person"),
    ("P57", "director", "person"),
    ("P86", "composer", "person"),
    ("P106", "occupation", "occupation"),
    ("P112", "founded by", "person"),
    ("P159", "headquarters location", "location"),
    ("P175", "performer", "person"),
    ("P176", "manufacturer", "organisation"),
    ("P361", "part of", "entity"),
    ("P495", "country of origin", "country"),
    ("P530", "diplomatic relation", "country"),
    ("P569", "date of birth", "date"),
    ("P570", "date of death", "date"),
    ("P571", "inception", "date"),
    ("P577", "publication date", "date"),
    ("P800", "notable work", "work"),
]


@dataclass
class Triple:
    subject: str
    subject_qid: str
    relation_pid: str
    relation: str
    obj: str
    obj_qid: str
    obj_type: str

    def to_dict(self) -> dict:
        return self.__dict__


class WikidataTripleSampler:
    """Sample (s, r, o) triples grouped by relation and object type."""

    def __init__(self, cache_dir: str = "data/wikidata_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._headers = {
            "User-Agent": "ESS-research/0.1 (mailto:anonymous@review.org)",
            "Accept": "application/sparql-results+json",
        }

    def _cache_path(self, pid: str) -> Path:
        return self.cache_dir / f"{pid}.jsonl"

    def fetch_relation(
        self,
        pid: str,
        relation_name: str,
        obj_type: str,
        limit: int = 5000,
    ) -> List[Triple]:
        cache = self._cache_path(pid)
        if cache.exists():
            return [Triple(**json.loads(l)) for l in cache.read_text().splitlines()]

        query = f"""
        SELECT ?s ?sLabel ?o ?oLabel WHERE {{
          ?s wdt:{pid} ?o .
          ?s rdfs:label ?sLabel . FILTER(LANG(?sLabel) = "en")
          ?o rdfs:label ?oLabel . FILTER(LANG(?oLabel) = "en")
        }} LIMIT {limit}
        """
        triples: List[Triple] = []
        try:
            resp = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=self._headers,
                timeout=120,
            )
            resp.raise_for_status()
            for b in resp.json()["results"]["bindings"]:
                triples.append(
                    Triple(
                        subject=b["sLabel"]["value"],
                        subject_qid=b["s"]["value"].rsplit("/", 1)[-1],
                        relation_pid=pid,
                        relation=relation_name,
                        obj=b["oLabel"]["value"],
                        obj_qid=b["o"]["value"].rsplit("/", 1)[-1],
                        obj_type=obj_type,
                    )
                )
        except Exception as e:
            print(f"[wikidata] {pid} failed: {e}; retrying in 30s")
            time.sleep(30)

        with cache.open("w") as f:
            for t in triples:
                f.write(json.dumps(t.to_dict()) + "\n")
        return triples

    def sample_all(self, per_relation: int = 2500) -> List[Triple]:
        out: List[Triple] = []
        for pid, name, obj_type in RELATIONS:
            triples = self.fetch_relation(pid, name, obj_type, limit=per_relation)
            random.shuffle(triples)
            out.extend(triples[:per_relation])
            time.sleep(1.0)        # be polite to the public endpoint
        return out
