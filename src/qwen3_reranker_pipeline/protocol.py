"""The tutorial's pre-registered evaluation protocol (docs/qwen3-reranker-preregistration.md, 2026-10-17).

A three-value learning-rate grid is compared on validation only, and a challenger replaces the default 2e-5 only by a
margin; the fresh test is scored only when every pre-registered setting holds (SCORE_ONCE_RULE). Any other setting is
exploratory: it reads validation and the labelled exposed development test, and writes separate
`*_exploratory_<n>.json` files.
"""
# ruff: noqa: E501  -- written at the fleet's protocol-module width (as in the sibling pipelines' protocol.py)

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import MAX_TRAIN_TOKENS
from .samples import FRESH_TEST_DIGEST, SAMPLE_SEED, TASK_INSTRUCTION

PREREGISTRATION = "docs/qwen3-reranker-preregistration.md"
GRID_MARGIN = 0.010  # kept-epoch validation MRR on 77 queries; a practical tolerance, not a significance threshold
DEFAULT_LR = 2e-5
GRID_LRS = (1e-5, 2e-5, 5e-5)
SAMPLE_DATA_SOURCE = "Banking77 sample"
# Every setting the pre-registration fixes; `lr` must also equal the grid's selection in the same run.
PREREGISTERED_SETTINGS: dict[str, Any] = {
    "data_source": SAMPLE_DATA_SOURCE,
    "split_seed": SAMPLE_SEED,
    "fresh_test_digest": FRESH_TEST_DIGEST,
    "instruction": TASK_INSTRUCTION,
    "epochs": 2,
    "batch_size": 4,
    "trainable_layers": 2,
    "train_candidates": 4,
    "adapt_seed": 0,
    "max_train_tokens": MAX_TRAIN_TOKENS,
    "grid_run": True,
}
# SCORE_ONCE_RULE item 8: a runtime cannot know about other runtimes, so the labels say "(this runtime)"; which run is
# the evaluation of record (the first hosted pre-registered run) is stated only in docs/release-verification.md.
TEST_LOOKS = (
    "pre-registered evaluation (this runtime)",
    "reproduction (this runtime)",
    "not scored (exploratory)",
    "not scored (fresh test masked for a development check)",
)
# The evaluation of record: the first hosted T4 run with the pre-registered settings, copied here from
# docs/release-verification.md once it exists (date, commit, notebook blob, runtime, fresh-test numbers). None until then;
# the notebook prints it beside its own (this runtime) scoring and never uses it to change a setting.
RECORDED_EVALUATION: dict[str, Any] | None = None
# The pipeline's record (after `adapt`) must equal Section 7's selected run on these keys before the fresh test is read.
_ADAPTER_KEYS = ("lr", "epochs", "batch_size", "trainable_layers", "train_candidates", "seed", "best_epoch", "n_train")
MODEL_STATE_MISMATCH = "model state is not the selected adapted model"


def run_fingerprint(
    *,
    data_source: str,
    split_seed: int,
    train_digest: str,
    validation_digest: str,
    instruction: str,
    epochs: int,
    batch_size: int,
    trainable_layers: int,
    train_candidates: int,
    adapt_seed: int,
) -> dict[str, Any]:
    """The inputs a grid selection is valid for: a selection made on other data or settings does not carry over."""
    return {
        "data_source": data_source,
        "split_seed": split_seed,
        "train_digest": train_digest,
        "validation_digest": validation_digest,
        "instruction": instruction,
        "epochs": epochs,
        "batch_size": batch_size,
        "trainable_layers": trainable_layers,
        "train_candidates": train_candidates,
        "adapt_seed": adapt_seed,
    }


def select_lr(
    scores: Mapping[float, float],
    frozen_val_mrr: float,
    *,
    margin: float = GRID_MARGIN,
    fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the pre-registered rule to the kept-epoch validation MRR of each grid learning rate.

    2e-5 stays unless a challenger beats it by >= margin; if both challengers clear it, the higher wins unless they are
    within the margin of each other, when the lower learning rate wins. The selected run must beat the frozen model's
    validation MRR by the margin, else the frozen model is kept as the evaluated system."""
    keys = {float(k) for k in scores}
    if keys != set(GRID_LRS):
        raise ValueError(f"grid scores must cover exactly {list(GRID_LRS)}")
    by_lr = {float(k): float(v) for k, v in scores.items()}
    base = by_lr[DEFAULT_LR]
    clear = [lr for lr in GRID_LRS if lr != DEFAULT_LR and by_lr[lr] >= base + margin - 1e-12]
    if not clear:
        selected, reason = DEFAULT_LR, f"no challenger beat {DEFAULT_LR:g} by {margin}"
    elif len(clear) == 1:
        selected, reason = clear[0], f"{clear[0]:g} beat {DEFAULT_LR:g} by >= {margin}"
    else:
        low, high = sorted(clear)
        if abs(by_lr[low] - by_lr[high]) < margin:
            selected, reason = low, f"{low:g} and {high:g} both beat {DEFAULT_LR:g} and are within {margin}: lower lr"
        else:
            selected = low if by_lr[low] > by_lr[high] else high
            reason = f"{low:g} and {high:g} both beat {DEFAULT_LR:g}; {selected:g} is higher by >= {margin}"
    kept = by_lr[selected] >= float(frozen_val_mrr) + margin - 1e-12
    return {
        "selected_lr": selected,
        "adaptation_kept": kept,
        "evaluated_system": "adapted" if kept else "frozen",
        "reason": reason
        + ("" if kept else f"; adaptation did not beat the frozen model on validation by {margin}: frozen is kept"),
        "scores": {f"{lr:g}": round(by_lr[lr], 4) for lr in GRID_LRS},
        "frozen_val_mrr": round(float(frozen_val_mrr), 4),
        "margin": margin,
        "metric": "kept-epoch validation MRR (pessimistic ties)",
        "fingerprint": dict(fingerprint) if fingerprint is not None else None,
    }


def adapter_mismatch(adapter: Mapping[str, Any] | None, adapt_result: Mapping[str, Any] | None) -> list[str]:
    """Why the model in memory is not the one Section 7 trained (empty = it is)."""
    if adapter is None or adapt_result is None:
        return [MODEL_STATE_MISMATCH + " (no adapter on the pipeline, or no Section 7 result)"]
    bad = [k for k in _ADAPTER_KEYS if adapter.get(k) != adapt_result.get(k)]
    return [f"{MODEL_STATE_MISMATCH} (differs in {bad})"] if bad else []


def settings_differing(
    settings: Mapping[str, Any],
    selection: Mapping[str, Any] | None,
    fingerprint: Mapping[str, Any] | None = None,
    model_state: Sequence[str] | None = None,
) -> list[str]:
    """The settings that differ from the pre-registration (empty = pre-registered): every fixed setting, the grid's
    selection and the inputs it ran on (`fingerprint`), and the model state (`model_state`, from `adapter_mismatch`)."""
    differing = [k for k, v in PREREGISTERED_SETTINGS.items() if settings.get(k) != v]
    differing += list(model_state or [])
    if selection is None:
        differing.append("lr (no grid selection in this run)")
    else:
        if fingerprint is None or selection.get("fingerprint") != dict(fingerprint):
            differing.append("grid selection (it ran on other data or settings than this run; re-run Section 7)")
        if settings.get("lr") != selection["selected_lr"]:
            differing.append(f"lr (used {settings.get('lr')}, grid selected {selection['selected_lr']})")
    return differing


def look_at_test(differing: Sequence[str], prior: Mapping[str, Any] | None = None, *, masked: bool = False) -> str:
    """`not scored (exploratory)` when any setting differs; the masked label for a development check; `reproduction
    (this runtime)` when an earlier report in this runtime carries a pre-registered scoring; else the pre-registered
    evaluation (this runtime)."""
    if differing:
        return TEST_LOOKS[2]
    if masked:
        return TEST_LOOKS[3]
    if (prior or {}).get("test_look") in TEST_LOOKS[:2]:
        return TEST_LOOKS[1]
    return TEST_LOOKS[0]


def fresh_test_scored(test_look: str) -> bool:
    return test_look in TEST_LOOKS[:2]


def output_suffix(path: str | Path, test_look: str) -> str:
    """One suffix per run for every file it writes: '' for the first pre-registered scoring in this runtime,
    `_reproduction_<n>` or `_exploratory_<n>` (first free n beside `path`) otherwise, so no run overwrites another."""
    if test_look == TEST_LOOKS[0]:
        return ""
    kind = "reproduction" if test_look == TEST_LOOKS[1] else "exploratory"
    path = Path(path)
    n = 1
    while path.with_name(f"{path.stem}_{kind}_{n}{path.suffix}").exists():
        n += 1
    return f"_{kind}_{n}"
