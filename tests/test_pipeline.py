import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from qwen3_reranker_pipeline import (
    DEFAULT_INSTRUCTION,
    DEFAULT_WEIGHTS_DIR,
    MAX_PAIRS,
    MAX_TEXT_CHARS,
    MAX_TEXT_TOKENS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    Qwen3RerankerPipeline,
    stage_missing_files,
    verify_snapshot,
)
from qwen3_reranker_pipeline.pipeline import NO_TOKEN_ID, YES_TOKEN_ID, format_pair

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"


def _fake_pipeline(logits=None, calls=None, n_tokens=None):
    def runner(bodies):
        if calls is not None:
            calls.append(list(bodies))
        out = np.zeros((len(bodies), 2), dtype=np.float32) if logits is None else np.asarray(logits)
        return out, n_tokens or [50] * len(bodies)

    return Qwen3RerankerPipeline(runner, "cpu")


def test_identity_constants_are_40_hex_and_match_manifest():
    assert re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION)
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert manifest["modelId"] == MODEL_ID
        assert manifest["revision"] == MODEL_REVISION
    logit_cfg = REPO / "weights" / MODEL_KEY / "1_LogitScore" / "config.json"
    if logit_cfg.is_file():
        cfg = json.loads(logit_cfg.read_text(encoding="utf-8"))
        assert (cfg["true_token_id"], cfg["false_token_id"]) == (YES_TOKEN_ID, NO_TOKEN_ID)


def _write_snapshot(tmp_path: Path, content: bytes, sha256: str, revision: str = MODEL_REVISION) -> Path:
    (tmp_path / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": revision,
        "files": [{"path": "config.json", "bytes": len(content), "sha256": sha256}],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_verify_snapshot_accepts_matching_digest(tmp_path):
    content = b'{"a": 1}'
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    assert verify_snapshot(root)["revision"] == MODEL_REVISION


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"a": 1}'
    good = hashlib.sha256(content).hexdigest()
    bad = ("0" if good[0] != "0" else "1") + good[1:]
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(_write_snapshot(tmp_path, content, bad))


def test_verify_snapshot_rejects_wrong_revision_and_size(tmp_path):
    content = b"xyz"
    digest = hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(_write_snapshot(tmp_path, content, digest, revision="f" * 40))
    root = _write_snapshot(tmp_path, content, digest)
    (tmp_path / "config.json").write_bytes(b"xyzw")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(root)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_from_pretrained_refuses_without_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError):
        Qwen3RerankerPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


@pytest.mark.parametrize(
    ("pairs", "kwargs", "exc"),
    [
        ("query document", {}, TypeError),
        ([], {}, ValueError),
        ([("q", "d")] * (MAX_PAIRS + 1), {}, ValueError),
        ([("q", "d", "extra")], {}, TypeError),
        (["q"], {}, TypeError),
        ([("q", 42)], {}, TypeError),
        ([("q", "   ")], {}, ValueError),
        ([("a" * (MAX_TEXT_CHARS + 1), "d")], {}, ValueError),
        ([("q", "d")], {"instruction": ""}, ValueError),
    ],
)
def test_rerank_rejects_bad_input(pairs, kwargs, exc):
    calls = []
    pipe = _fake_pipeline(calls=calls)
    with pytest.raises(exc):
        pipe.rerank(pairs, **kwargs)
    assert calls == []  # the model never ran


def test_rerank_output_fields_softmax_and_ranking():
    calls = []
    # row 0: yes logit 2 vs no 0 -> sigmoid(2); row 1: yes -3 vs no 0 -> sigmoid(-3)
    pipe = _fake_pipeline(logits=[[0.0, 2.0], [0.0, -3.0]], calls=calls)
    pairs = [
        ("What is the capital of China?", "The capital of China is Beijing."),
        ("Explain gravity", "Beijing"),
    ]
    result = pipe.rerank(pairs)
    expected_body = format_pair(pairs[0][0], pairs[0][1])
    assert calls[0][0] == expected_body
    assert expected_body == (
        f"<Instruct>: {DEFAULT_INSTRUCTION}\n<Query>: What is the capital of China?\n"
        "<Document>: The capital of China is Beijing."
    )
    assert abs(result["scores"][0] - 1 / (1 + np.exp(-2.0))) < 1e-9
    assert abs(result["scores"][1] - 1 / (1 + np.exp(3.0))) < 1e-9
    assert result["ranking"] == [0, 1]
    assert all(0.0 <= s <= 1.0 for s in result["scores"])
    assert result["score_kind"] == "relevance score, not a calibrated probability"
    assert result["instruction"] == DEFAULT_INSTRUCTION
    assert result["n_tokens"] == [50, 50]
    assert result["truncated"] == [False, False]
    assert result["model_id"] == MODEL_ID
    assert result["model_revision"] == MODEL_REVISION


def test_rerank_custom_instruction_and_truncation_flag():
    calls = []
    pipe = _fake_pipeline(logits=[[0.0, 0.0]], calls=calls, n_tokens=[MAX_TEXT_TOKENS])
    result = pipe.rerank([["q", "d"]], instruction="Retrieve code")
    assert calls == [["<Instruct>: Retrieve code\n<Query>: q\n<Document>: d"]]
    assert result["scores"] == [0.5]  # equal logits -> 0.5, no threshold applied
    assert result["truncated"] == [True]
