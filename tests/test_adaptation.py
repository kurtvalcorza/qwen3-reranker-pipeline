"""Offline tests for the query / positive / negatives dataset contract, the pinned Banking77 reader with its
seeded shortlists, ranking metrics, the random floor and lexical baseline, BYOD loaders, the injected-runner
evaluation path, artifact-manifest rejections and adapt() argument validation. Nothing here imports torch or
transformers; the corpus is two crafted CSV files served through an injected fetcher and the reranker is a
bag-of-words runner."""

from __future__ import annotations

import csv
import hashlib
import io
import json

import numpy as np
import pytest

from qwen3_reranker_pipeline import (
    ARTIFACT_FORMAT,
    DECODER_LAYERS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_NEGATIVES,
    SAMPLE_SPLIT,
    WEIGHT_SHA256,
    Qwen3RerankerPipeline,
    build_sample_dataset,
    candidate_list,
    check_split_disjoint,
    dataset_digest,
    documents,
    fetch_sample_dataset,
    intent_phrase,
    jaccard,
    lexical_baseline,
    load_byod_dataset,
    random_floor,
    rank_of_positive,
    ranking_metrics,
    record_seed,
    sample_negatives,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from qwen3_reranker_pipeline import pipeline as pl
from qwen3_reranker_pipeline import samples as sm
from qwen3_reranker_pipeline.samples import fetch_corpus, filter_records, read_corpus

INTENTS = [
    f"intent_{i:02d}_{w}"
    for i, w in enumerate(["card", "pin", "rate", "atm", "fee", "limit", "top", "close"])
]
PER_INTENT = {"train": 4, "test": 2}


def _rows(split):
    """Crafted Banking77 rows: every message names its intent word so a bag-of-words reranker can find it."""
    rows = []
    for intent in INTENTS:
        word = intent.split("_")[-1]
        for j in range(PER_INTENT[split]):
            rows.append({"text": f"{split} message {j} about my {word} please", "category": intent})
    return rows


def _csv(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["text", "category"])
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _files():
    return {"train": _csv(_rows("train")), "test": _csv(_rows("test"))}


def _pin(monkeypatch, files):
    monkeypatch.setattr(
        sm, "CORPUS_FILES", {k: (f"{k}.csv", len(v), hashlib.sha256(v).hexdigest()) for k, v in files.items()}
    )
    monkeypatch.setattr(sm, "CORPUS_ROWS", {k: len(INTENTS) * PER_INTENT[k] for k in files})
    monkeypatch.setattr(sm, "CORPUS_INTENTS", len(INTENTS))


WORDS = ["card", "pin", "rate", "atm"]


def _records(n=12):
    return [
        {
            "id": f"r{i:03d}",
            "query": f"message {i} about my {WORDS[i % 4]}",
            "positive": f"{WORDS[i % 4]} intent",
            "negatives": [f"{w} intent" for w in WORDS if w != WORDS[i % 4]],
        }
        for i in range(n)
    ]


def _bow_runner(bodies):
    """Bag-of-words reranker: the yes logit is the token overlap between the <Query> and <Document> lines."""
    logits = np.zeros((len(bodies), 2), dtype=np.float32)
    for row, body in enumerate(bodies):
        query = body.split("<Query>: ", 1)[1].split("\n<Document>: ", 1)[0]
        document = body.split("<Document>: ", 1)[1]
        logits[row, 1] = 4.0 * jaccard(query, document)
    return logits, [len(b.split()) for b in bodies]


def _pipeline_without_model():
    return Qwen3RerankerPipeline(_bow_runner, "cpu")


# --- corpus reader ----------------------------------------------------------------------------------


def test_pinned_corpus_constants():
    assert sm.CORPUS_BASE_URL.startswith(
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d"
    )
    assert {k: v[1] for k, v in sm.CORPUS_FILES.items()} == {"train": 839_073, "test": 239_961}
    assert all(len(v[2]) == 64 for v in sm.CORPUS_FILES.values())
    assert all(v % 77 == 0 for v in SAMPLE_SPLIT.values()) and set(SAMPLE_SPLIT) == {
        "train",
        "validation",
        "test",
    }
    assert intent_phrase("card_arrival") == "card arrival" and SAMPLE_NEGATIVES == 5


def test_fetch_corpus_verifies_each_file_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    calls = []

    def fetcher(url):
        calls.append(url)
        return files[url.rsplit("/", 1)[1].removesuffix(".csv")]

    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files
    assert len(calls) == 2 and all(u.startswith(sm.CORPUS_BASE_URL) for u in calls)
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")


def test_read_corpus_and_pairs(monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    corpus = read_corpus(files)
    assert len(corpus["train"]) == 32 and corpus["train"][0]["id"] == "train-00000"
    pairs = filter_records(corpus["train"])
    assert pairs[0]["positive"] == "intent 00 card" and pairs[0]["intent"] == INTENTS[0]
    assert len(filter_records([*corpus["train"], {**corpus["train"][0], "id": "dup"}])) == 32
    with pytest.raises(ValueError, match="missing the test file"):
        read_corpus({"train": files["train"]})
    monkeypatch.setattr(sm, "CORPUS_ROWS", {"train": 99, "test": 16})
    with pytest.raises(ValueError, match="expected 99"):
        read_corpus(files)


def test_sample_negatives_are_hard_then_seeded(forbid_model_imports):
    phrases = [intent_phrase(i) for i in INTENTS]
    negatives = sample_negatives(
        "my card and my pin", "intent 00 card", phrases, seed=1, n_negatives=5, n_hard=2
    )
    assert negatives[0] == "intent 01 pin" and len(negatives) == 5 and "intent 00 card" not in negatives
    assert len(set(negatives)) == 5
    assert negatives == sample_negatives(
        "my card and my pin", "intent 00 card", phrases, seed=1, n_negatives=5, n_hard=2
    )
    assert (
        negatives[2:]
        != sample_negatives("my card and my pin", "intent 00 card", phrases, seed=2, n_negatives=5, n_hard=2)[
            2:
        ]
    )
    with pytest.raises(ValueError, match="other phrases"):
        sample_negatives("x", phrases[0], phrases[:3], seed=1, n_negatives=5)


def test_sample_split_is_balanced_seeded_and_disjoint(monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    corpus = read_corpus(files)
    sizes = {"train": 16, "validation": 8, "test": 8}
    splits = build_sample_dataset(corpus, seed=1, sizes=sizes)
    assert {k: len(v) for k, v in splits.items()} == sizes
    assert all(sum(r["intent"] == i for r in splits["train"]) == 2 for i in INTENTS)
    first = splits["train"][0]
    assert first["id"] == "train-0000" and set(first) == {"id", "query", "positive", "negatives", "intent"}
    assert len(first["negatives"]) == SAMPLE_NEGATIVES and first["positive"] not in first["negatives"]
    assert first["negatives"] == sample_negatives(
        first["query"],
        first["positive"],
        [intent_phrase(i) for i in INTENTS],
        seed=record_seed("train-0000", 1),
    )
    assert check_split_disjoint(splits) == sizes and len(documents(splits["test"])) == 8
    assert build_sample_dataset(corpus, seed=1, sizes=sizes) == splits
    assert build_sample_dataset(corpus, seed=2, sizes=sizes) != splits
    with pytest.raises(ValueError, match="multiple"):
        build_sample_dataset(corpus, sizes={"train": 9, "validation": 8, "test": 8})
    with pytest.raises(ValueError, match="only"):
        build_sample_dataset(corpus, sizes={"train": 80, "validation": 8, "test": 8})
    leaky = {"train": splits["train"], "test": [{**splits["train"][0], "id": "leak"}]}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint(leaky)


def test_fetch_sample_dataset_end_to_end_with_injected_fetcher(tmp_path, monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    splits = fetch_sample_dataset(
        cache_dir=tmp_path,
        fetcher=lambda url: files[url.rsplit("/", 1)[1].removesuffix(".csv")],
        sizes={"train": 16, "validation": 8, "test": 8},
    )
    assert validate_dataset(splits["train"])["n_records"] == 16


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(forbid_model_imports):
    report = validate_dataset(_records())
    assert report["n_records"] == 12 and report["unique_queries"] == 12 and report["n_documents"] == 4
    assert report["candidates"] == {"min": 4, "max": 4, "mean": 4.0} and report["query_chars"]["min"] > 10
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    tagged = validate_dataset([{**_records()[0], "intent": 7}, *_records()[1:]])
    assert tagged["records"][0]["intent"] == "7" and "intent" not in tagged["records"][1]
    assert candidate_list(report["records"][0]) == ["card intent", "pin intent", "rate intent", "atm intent"]
    good = _records()
    for bad, message in (
        (good[:7], "8..20000"),
        ([{**good[0], "id": "bad id"}, *good[1:]], "id must match"),
        ([{**good[0], "id": good[1]["id"]}, *good[1:]], "duplicate id"),
        ([{**good[0], "query": " "}, *good[1:]], "query is empty"),
        ([{**good[0], "query": 5}, *good[1:]], "query must be a string"),
        ([{**good[0], "query": "x" * 100_001}, *good[1:]], "ceiling is 100000"),
        ([{**good[0], "positive": "x" * 1_001}, *good[1:]], "ceiling is 1000"),
        ([{**good[0], "negatives": []}, *good[1:]], "1..15 documents"),
        ([{**good[0], "negatives": "pin intent"}, *good[1:]], "must be a list"),
        ([{**good[0], "negatives": ["pin intent", "pin intent"]}, *good[1:]], "repeat a document"),
        ([{**good[0], "negatives": ["card intent"]}, *good[1:]], "equals the positive"),
        ([{**good[0], "negatives": [""]}, *good[1:]], "negatives\\[0\\] is empty"),
        ([{"id": "a", "query": "b", "positive": "c"}, *good[1:]], "missing 'negatives'"),
        (["not a mapping", *good[1:]], "must be a mapping"),
        ({"a": 1}, "must be a list"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset(bad)


def test_split_dataset_deduplicates_and_is_seeded(forbid_model_imports):
    records = [*_records(), {**_records()[0], "id": "dup"}]
    splits = split_dataset(records, val_fraction=0.1, test_fraction=0.2, seed=3)
    assert sum(len(v) for v in splits.values()) == 12 and len(splits["test"]) == 2
    assert check_split_disjoint(splits)
    assert split_dataset(records, val_fraction=0.1, test_fraction=0.2, seed=3) == splits
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="at least"):
        split_dataset(records, val_fraction=0.0, test_fraction=0.9)


# --- metrics, floor and baseline ----------------------------------------------------------------------


def test_rank_and_ranking_metrics(forbid_model_imports):
    assert (
        rank_of_positive([0.9, 0.1, 0.5]) == 1 and rank_of_positive([0.9, 0.9, 0.5]) == 2
    )  # ties count against
    assert rank_of_positive([0.1, 0.2, 0.3], 0) == 3
    with pytest.raises(ValueError, match="outside"):
        rank_of_positive([0.1], 3)
    metrics = ranking_metrics([1, 2, 4, 6], [6, 6, 6, 6])
    assert metrics["recall@1"] == 0.25 and metrics["recall@3"] == 0.5 and metrics["recall@5"] == 0.75
    assert metrics["mrr"] == pytest.approx((1 + 0.5 + 0.25 + 1 / 6) / 4) and metrics["median_rank"] == 4
    assert (
        metrics["n_queries"] == 4 and metrics["candidates"]["mean"] == 6.0 and "mrr" in metrics["definitions"]
    )
    with pytest.raises(ValueError, match="no queries"):
        ranking_metrics([], [])
    with pytest.raises(ValueError, match="1..list size"):
        ranking_metrics([7], [6])
    with pytest.raises(ValueError, match="align"):
        ranking_metrics([1], [6, 6])
    floor = random_floor([6, 6, 4])
    assert floor["recall@1"] == pytest.approx((1 / 6 + 1 / 6 + 1 / 4) / 3) and floor[
        "recall@5"
    ] == pytest.approx((5 / 6 + 5 / 6 + 1) / 3)
    assert floor["mrr"] == pytest.approx(
        (2 * sum(1 / r for r in range(1, 7)) / 6 + sum(1 / r for r in range(1, 5)) / 4) / 3
    )
    with pytest.raises(ValueError, match="at least two"):
        random_floor([1])


def test_lexical_baseline_orders_by_token_overlap(forbid_model_imports):
    assert jaccard("my card please", "card arrival") == pytest.approx(1 / 4) and jaccard("", "x") == 0.0
    baseline = lexical_baseline(_records())
    assert (
        baseline["recall@1"] == 1.0
        and "Jaccard" in baseline["baseline"]
        and baseline["candidates"]["max"] == 4
    )
    assert Qwen3RerankerPipeline.lexical_baseline(_records())["mrr"] == 1.0


# --- BYOD loaders and CSV -----------------------------------------------------------------------------


def test_byod_csv_json_jsonl_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = _records()
    csv_path = write_dataset_csv(records, tmp_path / "data.csv")
    assert load_byod_dataset(csv_path) == records
    assert "card intent | pin intent" in csv_path.read_text(encoding="utf-8")
    (tmp_path / "data.json").write_text(json.dumps(records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.json") == records
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.jsonl") == records
    (tmp_path / "bad.csv").write_text("id,query,positive\nx,y,z\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_byod_dataset(tmp_path / "bad.csv")
    (tmp_path / "obj.json").write_text('{"records": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="array of records"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "data.csv.bak").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="csv, .json or .jsonl"):
        load_byod_dataset(tmp_path / "data.csv.bak")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "missing.csv")


# --- evaluation, adaptation and artifacts without a model ---------------------------------------------


def test_evaluate_reranks_through_the_public_contract(forbid_model_imports):
    pipe = _pipeline_without_model()
    records = _records(40)  # 160 pairs: more than one MAX_PAIRS chunk
    metrics = pipe.evaluate(records, instruction="does the document name the intent")
    assert metrics["n_queries"] == 40 and metrics["n_pairs"] == 160 and metrics["candidates"]["mean"] == 4.0
    assert metrics["recall@1"] == 1.0 and metrics["mrr"] == 1.0 and metrics["median_rank"] == 1
    assert metrics["verdict"] == "measured-small-sample" and metrics["adapted"] is False
    assert metrics["instruction"] == "does the document name the intent" and metrics["model_id"] == MODEL_ID
    swapped = [
        {**r, "positive": r["negatives"][0], "negatives": [r["positive"], *r["negatives"][1:]]}
        for r in records[:12]
    ]
    worse = pipe.evaluate(swapped)
    assert (  # the intent word wins; the positive ties with the zero-overlap candidates (ties count against)
        worse["recall@1"] == 0.0 and worse["recall@3"] == 0.0 and worse["median_rank"] == 4
    )
    assert pipe.evaluate(_records(60))["verdict"] == "measured"
    with pytest.raises(ValueError, match="instruction"):
        pipe.evaluate(records[:12], instruction="  ")


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(_records(), batch_size=0)
    with pytest.raises(ValueError, match="train_candidates"):
        pipe.adapt(_records(), train_candidates=1)
    with pytest.raises(ValueError, match="instruction"):
        pipe.adapt(_records(), instruction="")
    with pytest.raises(ValueError, match="trainable_layers"):
        pipe.adapt(_records(), trainable_layers=DECODER_LAYERS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(_records())
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["model.layers.27.mlp.down_proj.weight"],
        "adapter": {"trainable_layers": 1},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)


def test_load_artifact_refuses_unsupported_versions_extra_files_and_traversal(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    good = {
        "format": ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": [],
        "adapter": {"trainable_layers": 1},
    }

    def write(manifest):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))

    write({**good, "format_version": "0.9"})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": good["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "other.safetensors"}]})
    with pytest.raises(ValueError, match="must name exactly"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]})
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**good, "base_model": {**good["base_model"], "weight_file": "other.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {}})
    with pytest.raises(ValueError, match="trainable_layers"):
        pipe.load_artifact(tmp_path)
    write(good)  # every manifest check passes; the weights file is still missing, and no model was imported
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
