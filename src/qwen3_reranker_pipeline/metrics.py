"""Ranking metrics over a query–candidate-list dataset, a random floor and a lexical baseline.

Every query comes with a candidate list — its positive document and its negatives — and a scorer ranks the
list; the rank of the positive gives **recall@1**, **recall@3**, **recall@5** (the positive is within the
first *k*) and **MRR** (mean of 1 / rank). Ties are resolved pessimistically (a tied candidate counts as
ranked above the positive). The **random floor** is what a uniformly random ordering of each query's list
achieves in expectation, averaged over the queries; the **lexical baseline** orders each list by the
Jaccard overlap of lower-cased alphanumeric tokens with the query — what a system with no model gets
from shared words.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

RECALL_AT = (1, 3, 5)
METRIC_DEFINITIONS = {
    "recall@k": (
        "fraction of queries whose positive document is ranked within the first k of the query's candidate "
        "list; ties count against the positive"
    ),
    "mrr": "mean over queries of 1 / rank of the positive document within its candidate list",
    "candidates": "each query's list is its positive followed by its negatives; lists may differ in length",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def record_seed(record_id: str, seed: int) -> int:
    """A stable per-record seed so seeded choices depend only on the record and the base seed."""
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def rank_of_positive(scores: Sequence[float], positive_index: int = 0) -> int:
    """1-based rank of `positive_index` under descending `scores`; ties are ranked above the positive."""
    if not 0 <= positive_index < len(scores):
        raise ValueError("positive_index is outside the candidate list")
    target = float(scores[positive_index])
    return 1 + sum(1 for i, s in enumerate(scores) if i != positive_index and float(s) >= target)


def ranking_metrics(ranks: Sequence[int], list_sizes: Sequence[int]) -> dict[str, Any]:
    """Aggregate 1-based ranks of each query's positive (with its list size) into recall@k and MRR."""
    if not ranks:
        raise ValueError("no queries to score")
    if len(ranks) != len(list_sizes):
        raise ValueError("ranks and list_sizes must align")
    for rank, size in zip(ranks, list_sizes, strict=True):
        if not isinstance(size, int) or size < 2:
            raise ValueError("every candidate list needs at least two entries")
        if not isinstance(rank, int) or not 1 <= rank <= size:
            raise ValueError("every rank must be an int in 1..list size")
    out: dict[str, Any] = {
        "n_queries": len(ranks),
        "candidates": {
            "min": min(list_sizes),
            "max": max(list_sizes),
            "mean": sum(list_sizes) / len(list_sizes),
        },
    }
    for k in RECALL_AT:
        out[f"recall@{k}"] = sum(1 for r in ranks if r <= k) / len(ranks)
    out["mrr"] = sum(1.0 / r for r in ranks) / len(ranks)
    out["median_rank"] = sorted(ranks)[len(ranks) // 2]
    out["definitions"] = dict(METRIC_DEFINITIONS)
    return out


def random_floor(list_sizes: Sequence[int]) -> dict[str, Any]:
    """Expected metrics of a uniformly random ordering of every query's candidate list."""
    if not list_sizes or any(not isinstance(n, int) or n < 2 for n in list_sizes):
        raise ValueError("every candidate list needs at least two entries")
    out: dict[str, Any] = {"n_queries": len(list_sizes)}
    for k in RECALL_AT:
        out[f"recall@{k}"] = sum(min(k, n) / n for n in list_sizes) / len(list_sizes)
    out["mrr"] = sum(sum(1.0 / r for r in range(1, n + 1)) / n for n in list_sizes) / len(list_sizes)
    out["baseline"] = "uniformly random ordering of each candidate list (expected values)"
    return out


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def candidate_list(record: Mapping[str, Any]) -> list[str]:
    """A record's candidate list: its positive first, then its negatives."""
    return [str(record["positive"]), *(str(n) for n in record["negatives"])]


def lexical_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Order every candidate list by Jaccard token overlap with the query — the no-model floor."""
    ranks, sizes = [], []
    for record in records:
        candidates = candidate_list(record)
        scores = [jaccard(record["query"], doc) for doc in candidates]
        ranks.append(rank_of_positive(scores, 0))
        sizes.append(len(candidates))
    result = ranking_metrics(ranks, sizes)
    result["baseline"] = "Jaccard overlap of lower-cased alphanumeric tokens between query and candidate"
    return result
