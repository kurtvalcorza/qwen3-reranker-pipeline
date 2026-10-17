"""Ranking metrics over a query–candidate-list dataset, a random floor and a lexical baseline.

Every query comes with a candidate list — its positive document and its negatives — and a scorer ranks the
list; the rank of the positive gives **recall@1**, **recall@3**, **recall@5** (the positive is within the
first *k*) and **MRR** (mean of 1 / rank). Every scorer is reported under two tie policies, side by
side: **pessimistic** (a tied candidate counts as ranked above the positive; the headline numbers) and
**expected** (`tie_expected`: the exact expectation over uniformly random orderings of the tied candidates,
the convention of the random floor). `ties_with_positive` counts the lists where any negative ties the
positive. The **random floor** is what a uniformly random ordering of each query's list
achieves in expectation, averaged over the queries; the **lexical baseline** orders each list by the
Jaccard overlap of lower-cased alphanumeric tokens with the query — what a system with no model gets
from shared words.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

RECALL_AT = (1, 3, 5)
# docs/qwen3-reranker-preregistration-amendment-1.md section 1: the fresh test's paired bootstrap.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261018
METRIC_DEFINITIONS = {
    "recall@k": (
        "fraction of queries whose positive document is ranked within the first k of the query's candidate "
        "list; ties count against the positive"
    ),
    "mrr": "mean over queries of 1 / rank of the positive document within its candidate list",
    "tie_expected": (
        "the same metrics as the exact expectation over uniformly random orderings of the candidates tied "
        "with the positive (the random floor's convention); equal to the pessimistic values when nothing ties"
    ),
    "ties_with_positive": (
        "number of candidate lists in which at least one negative scores exactly the positive's score"
    ),
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


def tie_counts(scores: Sequence[float], positive_index: int = 0) -> tuple[int, int]:
    """(candidates scored strictly above the positive, candidates tied with it)."""
    if not 0 <= positive_index < len(scores):
        raise ValueError("positive_index is outside the candidate list")
    target = float(scores[positive_index])
    others = [float(s) for i, s in enumerate(scores) if i != positive_index]
    return sum(1 for s in others if s > target), sum(1 for s in others if s == target)


def expected_metrics(counts: Sequence[tuple[int, int]]) -> dict[str, float]:
    """Exact expected recall@k and MRR when each positive's ties are broken uniformly at random."""
    if not counts:
        raise ValueError("no queries to score")
    out = dict.fromkeys([f"recall@{k}" for k in RECALL_AT] + ["mrr"], 0.0)
    for higher, tied in counts:
        ranks = [higher + 1 + j for j in range(tied + 1)]
        for k in RECALL_AT:
            out[f"recall@{k}"] += sum(1 for r in ranks if r <= k) / len(ranks)
        out["mrr"] += sum(1.0 / r for r in ranks) / len(ranks)
    return {k: v / len(counts) for k, v in out.items()}


def score_lists_metrics(score_lists: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Both tie policies for lists whose positive is at index 0: the pessimistic `ranking_metrics`, plus
    `tie_expected` and `ties_with_positive`."""
    counts = [tie_counts(scores, 0) for scores in score_lists]
    result = ranking_metrics([h + t + 1 for h, t in counts], [len(s) for s in score_lists])
    result["tie_policy"] = "pessimistic (ties count against the positive); see tie_expected"
    result["tie_expected"] = expected_metrics(counts)
    result["ties_with_positive"] = sum(1 for _h, t in counts if t)
    result["ranks"] = [h + t + 1 for h, t in counts]  # per-query pessimistic ranks, for the paired bootstrap
    return result


def _interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    last = len(ordered) - 1
    return [ordered[round(0.025 * last)], ordered[round(0.975 * last)]]


def ci_verdict(ci: Sequence[float]) -> str:
    """`improved` when the 95 % CI of a difference lies above 0, `worse` below 0, else `inconclusive`."""
    if ci[0] > 0:
        return "improved"
    if ci[1] < 0:
        return "worse"
    return "inconclusive"


def paired_bootstrap(
    ranks_by_system: Mapping[str, Sequence[int]],
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    pair: tuple[str, str] = ("adapted", "frozen"),
) -> dict[str, Any]:
    """Paired bootstrap 95 % CIs over queries: each resample draws query indices with replacement
    (`random.Random` seeded by `seed`) and uses them for every system. Reports MRR and recall@1 per system
    and, when both systems of `pair` are present, their difference with a CI-based verdict. The interval
    covers the choice of queries only."""
    names = list(ranks_by_system)
    if not names:
        raise ValueError("no systems to resample")
    n = len(ranks_by_system[names[0]])
    if n == 0 or any(len(ranks_by_system[k]) != n for k in names):
        raise ValueError("every system needs the same non-zero number of per-query ranks")
    rr = {k: [1.0 / r for r in ranks_by_system[k]] for k in names}
    hit = {k: [1.0 if r == 1 else 0.0 for r in ranks_by_system[k]] for k in names}
    paired = pair[0] in rr and pair[1] in rr
    draws: dict[tuple[str, str], list[float]] = {(k, s): [] for k in names for s in ("mrr", "recall@1")}
    diffs: dict[str, list[float]] = {"mrr": [], "recall@1": []}
    rng = random.Random(seed)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        means = {}
        for k in names:
            means[(k, "mrr")] = sum(rr[k][i] for i in idx) / n
            means[(k, "recall@1")] = sum(hit[k][i] for i in idx) / n
            draws[(k, "mrr")].append(means[(k, "mrr")])
            draws[(k, "recall@1")].append(means[(k, "recall@1")])
        if paired:
            for s in ("mrr", "recall@1"):
                diffs[s].append(means[(pair[0], s)] - means[(pair[1], s)])
    out: dict[str, Any] = {
        "method": "paired bootstrap over queries (pessimistic ranks); percentile 95 % CI",
        "n_boot": n_boot,
        "seed": seed,
        "n_queries": n,
        "covers": "the choice of shortlists only, not split-to-split or seed-to-seed variation",
        "systems": {
            k: {
                s: {"point": sum(src[k]) / n, "ci95": _interval(draws[(k, s)])}
                for s, src in (("mrr", rr), ("recall@1", hit))
            }
            for k in names
        },
    }
    if paired:
        out["difference"] = {"pair": f"{pair[0]} - {pair[1]}"}
        for s, src in (("mrr", rr), ("recall@1", hit)):
            ci = _interval(diffs[s])
            point = (sum(src[pair[0]]) - sum(src[pair[1]])) / n
            out["difference"][s] = {"point": point, "ci95": ci, "verdict": ci_verdict(ci)}
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
    result = score_lists_metrics(
        [[jaccard(record["query"], doc) for doc in candidate_list(record)] for record in records]
    )
    result["baseline"] = "Jaccard overlap of lower-cased alphanumeric tokens between query and candidate"
    return result
