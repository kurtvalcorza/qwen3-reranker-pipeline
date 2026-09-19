"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): ranking metrics
of the frozen reranker against the lexical baseline, a one-epoch listwise adaptation of the last decoder layer
on a dozen shortlists, and the artifact round trip. Skipped when the weights are absent."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

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


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, tmp_path):
    """Without a validation split the recorded policy is "final epoch": the state after the last of three
    epochs is what stays in memory and what the artifact carries."""
    import torch

    result = pipe.adapt(
        RECORDS[:8], None, instruction=TASK_INSTRUCTION, epochs=3, trainable_layers=1, batch_size=4
    )
    assert result["best_epoch"] == 3 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 4
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = Qwen3RerankerPipeline.from_artifact(artifact, device="cpu")
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["best_epoch"] == 3 and reloaded.adapter["trainable_layers"] == 1


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, tmp_path):
    """The loader derives the exact tensor set from the recorded layer count: a manifest listing fewer,
    more or other tensors — or one whose safetensors payload differs from its list — is refused."""
    import json as _json
    import shutil

    from safetensors.torch import load_file, save_file

    pipe.adapt(RECORDS[:8], None, instruction=TASK_INSTRUCTION, epochs=1, trainable_layers=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = _json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(_json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        Qwen3RerankerPipeline.from_artifact(fewer, device="cpu")
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["zz.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    size = (extra / "adapter.safetensors").stat().st_size
    files = [{**manifest["files"][0], "bytes": size, "sha256": digest}]
    (extra / "manifest.json").write_text(_json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        Qwen3RerankerPipeline.from_artifact(extra, device="cpu")
    other_layers = tmp_path / "other_layers"
    shutil.copytree(artifact, other_layers)
    adapter = {**manifest["adapter"], "trainable_layers": 2}
    (other_layers / "manifest.json").write_text(_json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        Qwen3RerankerPipeline.from_artifact(other_layers, device="cpu")


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe):
    """A failure inside training leaves the base exactly as it was, frozen, with no adapter attached."""
    import torch

    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(
            RECORDS[:8],
            None,
            instruction=TASK_INSTRUCTION,
            epochs=2,
            trainable_layers=1,
            batch_size=4,
            progress=boom,
        )
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter is None
    assert not any(p.requires_grad for p in pipe._model.parameters())
