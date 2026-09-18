"""Query / positive / negatives dataset contract for reranker fine-tuning: the pinned Banking77 sample,
validation, seeded splitting, BYOD loaders and CSV export.

The default dataset is **real** and a reranking task the model was not tuned for: Banking77 (Casanueva et
al., 2020; CC BY 4.0), 13,083 customer-support messages labelled with 77 fine-grained banking intents. Two
CSV files (`train.csv`, `test.csv`) are fetched from the PolyAI `task-specific-datasets` repository at a
pinned commit and refused on any byte-size or SHA-256 mismatch. Every intent name becomes a short
**document** (`card_arrival` → `card arrival`); each message is a **query** whose positive document is its
intent phrase, and its **negatives** are a seeded shortlist of other intent phrases — the hardest few by
token overlap with the message plus random ones — so reranking the shortlist is intent detection over a
candidate list. Training and validation queries are drawn from `train.csv`, test queries from `test.csv` —
the release's own partition — balanced over the 77 intents.

A record is ``{id, query, positive, negatives}``; its candidate list is the positive then the negatives.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .metrics import jaccard, record_seed
from .pipeline import MAX_TEXT_CHARS, MODEL_ID

CORPUS_NAME = "Banking77 (messages → intent-phrase shortlists)"
CORPUS_RELEASE = "PolyAI-LDN/task-specific-datasets @ 57ec275d8078af65b7731c2a98be812d844a6d6b"
CORPUS_BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/"
)
CORPUS_FILES = {
    "train": ("train.csv", 839_073, "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"),
    "test": ("test.csv", 239_961, "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"),
}
CORPUS_LICENSE = "CC BY 4.0 (Casanueva et al. 2020; PolyAI-LDN/task-specific-datasets)"
CORPUS_ROWS = {"train": 10_003, "test": 3_080}
CORPUS_INTENTS = 77
DEFAULT_CACHE_DIR = Path("weights") / "banking77"
TASK_INSTRUCTION = (
    "Given a customer support message, judge whether the document names the banking intent it expresses"
)
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 231, "validation": 77, "test": 154}  # 3 / 1 / 2 per intent, balanced over 77
SAMPLE_NEGATIVES = 5  # per query: SAMPLE_HARD_NEGATIVES highest-overlap other intents plus seeded random ones
SAMPLE_HARD_NEGATIVES = 3
MAX_NEGATIVES = 15
NEGATIVE_SEPARATOR = " | "  # the CSV column format of `negatives`
MIN_RECORDS = 8
MAX_RECORDS = 20_000
MAX_DOCUMENT_CHARS = 1_000
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def intent_phrase(intent: str) -> str:
    """The document text of an intent: its snake_case name as words (`card_arrival` → `card arrival`)."""
    return " ".join(intent.strip().split("_"))


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, bytes]:
    """Return the two pinned Banking77 CSVs (bytes) from the cache or the project repository, verified."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, (name, size, digest) in CORPUS_FILES.items():
        local = cache / name
        data = local.read_bytes() if local.is_file() else b""
        if len(data) != size or _sha256_bytes(data) != digest:
            url = CORPUS_BASE_URL + name
            if fetcher is not None:
                data = fetcher(url)
            else:
                with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (pinned https URL)
                    data = response.read()
            if len(data) != size or _sha256_bytes(data) != digest:
                raise ValueError(
                    f"{name}: fetched {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, "
                    f"pinned {size} / {digest[:16]}…"
                )
            local.write_bytes(data)
        out[split] = data
    return out


def read_corpus(files: Mapping[str, bytes]) -> dict[str, list[dict[str, Any]]]:
    """Parse the CSV members (columns `text`, `category`) into flat records keeping the raw intent name."""
    out = {}
    for split in CORPUS_FILES:
        if split not in files:
            raise ValueError(f"corpus is missing the {split} file")
        rows = list(csv.DictReader(io.StringIO(files[split].decode("utf-8"))))
        if not rows or {"text", "category"} - set(rows[0]):
            raise ValueError(f"{split}: expected columns text and category")
        if len(rows) != CORPUS_ROWS[split]:
            raise ValueError(f"{split}: {len(rows)} rows, expected {CORPUS_ROWS[split]}")
        out[split] = [
            {"id": f"{split}-{i:05d}", "text": r["text"].strip(), "intent": r["category"].strip()}
            for i, r in enumerate(rows)
        ]
        intents = {r["intent"] for r in out[split]}
        if len(intents) != CORPUS_INTENTS:
            raise ValueError(f"{split}: {len(intents)} intents, expected {CORPUS_INTENTS}")
    return out


def filter_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Turn corpus rows into query–positive pairs; drop empty, over-long and repeated queries."""
    seen: set[str] = set()
    kept = []
    for record in records:
        query = str(record["text"]).strip()
        key = query.lower()
        if not query or key in seen or len(query) > MAX_TEXT_CHARS:
            continue
        seen.add(key)
        intent = str(record["intent"])
        kept.append({"id": record["id"], "query": query, "positive": intent_phrase(intent), "intent": intent})
    return kept


def sample_negatives(
    query: str,
    positive: str,
    phrases: Sequence[str],
    *,
    seed: int,
    n_negatives: int = SAMPLE_NEGATIVES,
    n_hard: int = SAMPLE_HARD_NEGATIVES,
) -> list[str]:
    """A seeded shortlist of other phrases: the `n_hard` with the highest token overlap with the query (ties
    broken by phrase order), then random others up to `n_negatives`."""
    others = [p for p in phrases if p != positive]
    if len(others) < n_negatives:
        raise ValueError(f"need {n_negatives} other phrases, have {len(others)}")
    hard = sorted(others, key=lambda p: (-jaccard(query, p), p))[:n_hard]
    rest = [p for p in others if p not in hard]
    random.Random(seed).shuffle(rest)
    return hard + rest[: n_negatives - len(hard)]


def build_sample_dataset(
    corpus: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Balanced seeded draws over all 77 intents with a seeded shortlist of negatives per query: training and
    validation from `train` (disjoint queries), test from `test`."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    for name, size in sizes.items():
        if size % CORPUS_INTENTS:
            raise ValueError(f"{name} size {size} is not a multiple of the {CORPUS_INTENTS} intents")
    rng = random.Random(seed)
    pools = {"train": filter_records(corpus["train"]), "test": filter_records(corpus["test"])}
    intents = sorted({r["intent"] for r in pools["train"]})
    phrases = [intent_phrase(intent) for intent in intents]
    by_intent = {
        split: {intent: [r for r in pool if r["intent"] == intent] for intent in intents}
        for split, pool in pools.items()
    }
    for split in by_intent.values():
        for records in split.values():
            rng.shuffle(records)
    cursor = dict.fromkeys(intents, 0)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, size in sizes.items():
        source = "test" if name == "test" else "train"
        per_intent = size // CORPUS_INTENTS
        picked = []
        for intent in intents:
            pool = by_intent[source][intent]
            start = cursor[intent] if source == "train" else 0
            chunk = pool[start : start + per_intent]
            if len(chunk) < per_intent:
                raise ValueError(
                    f"{name}: only {len(chunk)} records available for {intent!r}, need {per_intent}"
                )
            picked.extend(chunk)
            if source == "train":
                cursor[intent] = start + per_intent
        rng.shuffle(picked)
        out[name] = []
        for i, r in enumerate(picked):
            rid = f"{name}-{i:04d}"
            out[name].append(
                {
                    "id": rid,
                    "query": r["query"],
                    "positive": r["positive"],
                    "negatives": sample_negatives(
                        r["query"], r["positive"], phrases, seed=record_seed(rid, seed)
                    ),
                    "intent": r["intent"],
                }
            )
    return out


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Any = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus."""
    corpus = read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher))
    return build_sample_dataset(corpus, seed=seed, sizes=sizes)


def _check_text(value: Any, label: str, ceiling: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} is empty")
    if len(value) > ceiling:
        raise ValueError(f"{label} has {len(value)} chars; ceiling is {ceiling}")
    return value.strip()


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/query/positive/negatives")
    for key in ("id", "query", "positive", "negatives"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label}: id must match {_ID_RE.pattern}")
    item = {
        "id": rid,
        "query": _check_text(record["query"], f"{label}: query", MAX_TEXT_CHARS),
        "positive": _check_text(record["positive"], f"{label}: positive", MAX_DOCUMENT_CHARS),
    }
    negatives = record["negatives"]
    if isinstance(negatives, str | bytes) or not isinstance(negatives, Sequence):
        raise ValueError(f"{label}: negatives must be a list of 1..{MAX_NEGATIVES} documents")
    if not 1 <= len(negatives) <= MAX_NEGATIVES:
        raise ValueError(f"{label}: negatives must hold 1..{MAX_NEGATIVES} documents, got {len(negatives)}")
    checked = [
        _check_text(n, f"{label}: negatives[{j}]", MAX_DOCUMENT_CHARS) for j, n in enumerate(negatives)
    ]
    if len(set(checked)) != len(checked):
        raise ValueError(f"{label}: negatives repeat a document")
    if item["positive"] in checked:
        raise ValueError(f"{label}: a negative equals the positive")
    item["negatives"] = checked
    if "intent" in record:
        item["intent"] = str(record["intent"])
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a query / positive / negatives dataset; raises before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of {id, query, positive, negatives} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    queries: set[str] = set()
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        queries.add(item["query"].lower())
        checked.append(item)
    sizes = [1 + len(r["negatives"]) for r in checked]
    return {
        "records": checked,
        "n_records": len(checked),
        "unique_queries": len(queries),
        "n_documents": len(documents(checked)),
        "candidates": {"min": min(sizes), "max": max(sizes), "mean": sum(sizes) / len(sizes)},
        "query_chars": {
            "min": min(len(r["query"]) for r in checked),
            "max": max(len(r["query"]) for r in checked),
        },
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def documents(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The sorted unique documents (positives and negatives) of a dataset."""
    positives = {str(r["positive"]).strip() for r in records}
    negatives = {str(n).strip() for r in records for n in r["negatives"]}
    return sorted(positives | negatives)


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], r["query"], r["positive"], list(r["negatives"])] for r in records]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no lower-cased query appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = str(record["query"]).lower()
            if key in seen and seen[key] != name:
                raise ValueError(f"query {record['query'][:60]!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test after de-duplicating queries."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = record["query"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(record)
    random.Random(seed).shuffle(unique)
    n_test = max(1, round(len(unique) * test_fraction))
    n_val = round(len(unique) * val_fraction)
    splits = {
        "test": unique[:n_test],
        "validation": unique[n_test : n_test + n_val],
        "train": unique[n_test + n_val :],
    }
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, query, positive, negatives}` records from CSV (columns id, query, positive, negatives — the
    negatives separated by ` | `), a JSON array or JSONL (negatives as a list)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        missing = {"id", "query", "positive", "negatives"} - set(rows[0].keys() if rows else set())
        if missing:
            raise ValueError(f"CSV is missing columns {sorted(missing)}")
        return [
            {
                "id": r["id"],
                "query": r["query"],
                "positive": r["positive"],
                "negatives": [
                    n.strip() for n in r["negatives"].split(NEGATIVE_SEPARATOR.strip()) if n.strip()
                ],
            }
            for r in rows
        ]
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of records")
        return data
    raise ValueError("BYOD datasets must be .csv, .json or .jsonl")


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "query", "positive", "negatives"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "query": record["query"],
                    "positive": record["positive"],
                    "negatives": NEGATIVE_SEPARATOR.join(record["negatives"]),
                }
            )
    return out
