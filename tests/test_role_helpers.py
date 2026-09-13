"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from qwen3_reranker_pipeline import (
    DEFAULT_INSTRUCTION,
    INPUT_SCHEMA,
    MAX_PAIRS,
    MAX_TEXT_CHARS,
    MAX_TEXT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)

QUERY = "What is the capital of China?"


def _pairs(n: int = 2) -> list[tuple[str, str]]:
    return [(QUERY, f"document number {i}") for i in range(n)]


def _result(scores: list[float]) -> dict:
    return {
        "scores": scores,
        "ranking": sorted(range(len(scores)), key=lambda i: -scores[i]),
        "score_kind": "relevance score, not a calibrated probability",
        "instruction": DEFAULT_INSTRUCTION,
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_pairs(), names=["doc00", "doc01"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["pairs"] == [1, MAX_PAIRS]
    assert manifest["schema"]["text_chars"] == [1, MAX_TEXT_CHARS]
    assert manifest["schema"]["prompt_tokens"] == [1, MAX_TEXT_TOKENS]
    assert manifest["schema"]["score_range"] == [0.0, 1.0]
    assert manifest["inputs"] == [
        {"id": "doc00", "query_chars": len(QUERY), "document_chars": 17},
        {"id": "doc01", "query_chars": len(QUERY), "document_chars": 17},
    ]
    assert manifest["n_pairs"] == 2
    assert manifest["instruction"] == DEFAULT_INSTRUCTION
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids_and_custom_instruction() -> None:
    manifest = validate_inputs(_pairs(1), "Find the passage that answers the question")
    assert [entry["id"] for entry in manifest["inputs"]] == ["pair-0"]
    assert manifest["instruction"] == "Find the passage that answers the question"


def test_validate_inputs_rejects_like_rerank() -> None:
    with pytest.raises(TypeError, match="list of \\(query, document\\) pairs"):
        validate_inputs("not a pair list")
    with pytest.raises(ValueError, match=f"1\\.\\.{MAX_PAIRS}"):
        validate_inputs(_pairs(MAX_PAIRS + 1))
    with pytest.raises(TypeError, match="pair of two str"):
        validate_inputs([(QUERY,)])
    with pytest.raises(ValueError, match="document is empty"):
        validate_inputs([(QUERY, "   ")])
    with pytest.raises(ValueError, match="ceiling is"):
        validate_inputs([(QUERY, "x" * (MAX_TEXT_CHARS + 1))])
    with pytest.raises(ValueError, match="instruction must be a non-empty str"):
        validate_inputs(_pairs(1), "")
    with pytest.raises(ValueError, match="names must have one entry per pair"):
        validate_inputs(_pairs(1), names=["a", "b"])


def test_evaluation_report_is_always_not_measurable() -> None:
    report = evaluation_report(_result([0.9, 0.2, 0.1]))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["n_pairs"] == 3
    assert report["sample_kind"] == "synthetic"
    assert "the evaluated sample carries none" in report["reason"]
    assert "nDCG@k, MRR or precision@k" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_stays_not_measurable_when_judgements_are_supplied() -> None:
    report = evaluation_report(_result([0.9, 0.1]), [1, 0], sample_kind="BYOD upload")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["sample_kind"] == "BYOD upload"
    assert "judgements were supplied but no metric helper exists" in report["reason"]
