"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): ranking metrics
of the frozen reranker against the lexical baseline, a one-epoch listwise adaptation of the last decoder layer
on a dozen shortlists, and the artifact round trip. Skipped when the weights are absent."""

from __future__ import annotations

import json

import pytest

from qwen3_reranker_pipeline import DEFAULT_WEIGHTS_DIR, TASK_INSTRUCTION, WEIGHT_FILE, Qwen3RerankerPipeline

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

DOCS = [
    "card arrival",
    "change pin",
    "exchange rate",
    "atm support",
    "declined card payment",
    "terminate account",
]
QUERIES = [
    ("I still have not received my new card, when will it arrive?", 0),
    ("My card never showed up in the post.", 0),
    ("How do I change the PIN on my card?", 1),
    ("I forgot my PIN and want a new one.", 1),
    ("What exchange rate do you use for euros?", 2),
    ("Is the rate for converting to dollars competitive?", 2),
    ("The ATM ate my card and I cannot get it back.", 3),
    ("Which cash machines can I use abroad?", 3),
    ("My payment at the shop was declined.", 4),
    ("Why was my card refused at the checkout?", 4),
    ("I want to close my account for good.", 5),
    ("How do I delete my account?", 5),
]
RECORDS = [
    {"id": f"p{i:02d}", "query": q, "positive": DOCS[g], "negatives": [d for d in DOCS if d != DOCS[g]][:3]}
    for i, (q, g) in enumerate(QUERIES)
]


@pytest.fixture(scope="module")
def pipe():
    return Qwen3RerankerPipeline.from_pretrained(device="cpu")


def test_frozen_ranking_beats_the_lexical_baseline(pipe):
    metrics = pipe.evaluate(RECORDS, instruction=TASK_INSTRUCTION)
    baseline = pipe.lexical_baseline(RECORDS)
    assert metrics["n_queries"] == 12 and metrics["n_pairs"] == 48 and metrics["adapted"] is False
    assert metrics["mrr"] >= baseline["mrr"] and metrics["recall@1"] >= 0.5


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, tmp_path):
    result = pipe.adapt(
        RECORDS[:8], RECORDS[8:], instruction=TASK_INSTRUCTION, epochs=1, trainable_layers=1, batch_size=4
    )
    assert result["n_trainable"] == 15_730_944 and result["history"][0]["note"] == "frozen model"
    assert all(name.startswith("model.layers.27.") for name in result["trainable_names"])
    assert result["n_total"] == 595_776_512 and "mrr" in result["history"][1]["val"]
    assert result["history"][1]["train_loss"] >= 0.0 and result["n_train_pairs"] == 32
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert (
        len(manifest["tensors"]) == len(result["trainable_names"])
        and manifest["adapter"]["instruction"] == TASK_INSTRUCTION
    )
    reloaded = Qwen3RerankerPipeline.from_artifact(artifact, device="cpu")
    pairs = [(r["query"], r["positive"]) for r in RECORDS[:3]]
    assert (
        reloaded.rerank(pairs, TASK_INSTRUCTION)["scores"] == pipe.rerank(pairs, TASK_INSTRUCTION)["scores"]
    )
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
