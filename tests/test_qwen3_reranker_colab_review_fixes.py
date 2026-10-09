"""Regression tests for the Notebook Review Framework v1 findings on qwen3_reranker_colab.ipynb (RR-M2, RR-M3,
RR-M4, RR-m1, RR-m2, RR-m3, RR-m4, RR-m5). RR-M1 and RR-m6 are covered by tests/test_sweep_fixes.py (SWP-R, SWP-B)
and tests/test_worker_colab_stubs.py.

Only CI's dependencies are used: the notebook's own cell sources run against the carried modules (no model) or
stand-ins. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import os
import types
from pathlib import Path

import pytest

from qwen3_reranker_pipeline import metrics, samples
from qwen3_reranker_pipeline import pipeline as pl

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "qwen3_reranker_colab.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _cell(notebook: dict, marker: str) -> str:
    found = [_source(c) for c in notebook["cells"] if c["cell_type"] == "code" and marker in _source(c)]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _records(n: int, k: int = 4) -> list[dict]:
    topics = [f"topic {j}" for j in range(k)]
    return [
        {"id": f"r{i:05d}", "query": f"question number {i} about {topics[i % k]}", "positive": topics[i % k],
         "negatives": [t for t in topics if t != topics[i % k]]}
        for i in range(n)
    ]


# --- RR-M3: the BYOD size contract is checked once, on the user's dataset, and stated truly -------------------------


def test_rr_m3_bounds_at_the_default_fractions():
    assert samples.split_size_bounds() == (12, 10_002)
    assert pl.MAX_EVAL_RECORDS == 2_000


@pytest.mark.parametrize("n", [12, 20, 49, 50, 10_002])
def test_rr_m3_accepted_sizes_validate_every_split_and_fit_evaluate(n):
    splits = samples.split_dataset(_records(n), seed=0)
    samples.validate_dataset(splits["train"])
    for name in ("validation", "test"):
        samples.validate_dataset(splits[name], min_records=1, max_records=pl.MAX_EVAL_RECORDS)
    assert samples.check_split_disjoint(splits) == {name: len(part) for name, part in splits.items()}


@pytest.mark.parametrize("n", [3, 8, 11, 10_003, 20_000])
def test_rr_m3_rejected_sizes_name_the_dataset_size_and_the_range(n):
    with pytest.raises(ValueError, match=rf"the dataset has {n} distinct queries \(of {n} records\).*12\.\.10002 distinct queries are required"):
        samples.split_dataset(_records(n), seed=0)


def test_rr_m3_section_4_byod_path_reaches_the_split_with_12_records(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "shortlists.csv"
    with data.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "query", "positive", "negatives"])
        writer.writeheader()
        for r in _records(12):
            writer.writerow({**r, "negatives": samples.NEGATIVE_SEPARATOR.join(r["negatives"])})
    source = _cell(notebook, "USE_BYOD = False")
    source = source.replace("USE_BYOD = False", "USE_BYOD = True").replace("BYOD_PATH = ''", f"BYOD_PATH = {str(data)!r}")
    namespace = {name: getattr(samples, name) for name in dir(samples) if not name.startswith("__")}
    namespace.update(os=os, Path=Path)
    exec(compile(source, "<section 4 byod>", "exec"), namespace)
    assert namespace["disjoint"] == {"test": 2, "validation": 2, "train": 8}
    assert namespace["dataset_manifests"]["validation"]["n_records"] == 2
    assert namespace["data_source"] == "BYOD (shortlists.csv)" and (tmp_path / "outputs" / "qwen3_reranker_train.csv").is_file()


def test_rr_m3_prerequisites_and_troubleshooting_state_the_true_range(notebook):
    md = _markdown(notebook)
    assert "a BYOD dataset needs **12..10,002 distinct queries**" in md
    assert "12..10002 distinct queries are required" in md
    assert "a dataset needs 8..20,000 records" not in md and "at least eight records" not in md


def _m(r1: float, mrr: float) -> dict:
    return {"recall@1": r1, "recall@3": r1, "recall@5": r1, "mrr": mrr, "median_rank": 1, "n_queries": 154}


def _section_6_tail(notebook: dict) -> str:
    source = _cell(notebook, "frozen_vs_floor = ")
    return source[source.index("# Contract integrity: both systems ranked the same shortlists.") :]


def test_rr_m3_sample_path_still_stops_on_a_broken_frozen_model(notebook):
    namespace = {"frozen_test": _m(0.1, 0.3), "baseline_lexical": _m(0.27, 0.46), "floor": _m(0.1667, 0.4083), "USE_BYOD": False}
    with pytest.raises(RuntimeError, match="frozen model must rank above the random floor"):
        exec(compile(_section_6_tail(notebook), "<section 6>", "exec"), namespace)
    namespace.update(USE_BYOD=True)
    exec(compile(_section_6_tail(notebook), "<section 6>", "exec"), namespace)
    assert namespace["frozen_vs_floor"] == "not above"


def _section_8_namespace(adapter, adapted_mrr, use_byod, default_settings):
    return {
        "json": json, "pipe": types.SimpleNamespace(adapter=adapter, evaluate=lambda records, **kw: _m(0.75, adapted_mrr)), "test_records": [], "val_records": [],
        "INSTRUCTION": "i", "floor": _m(0.1667, 0.4083), "baseline_lexical": _m(0.27, 0.46), "frozen_test": _m(0.73, 0.8429), "frozen_vs_floor": "above",
        "MODEL_ID": "m", "MODEL_REVISION": "r", "MODEL_KEY": "k", "data_source": "stand-in", "dataset_manifests": {"test": {"digest": "d", "candidates": 6}},
        "disjoint": {}, "adapt_result": {"history": [], "best_epoch": 0}, "adapt_seconds": 0.0,
        "USE_BYOD": use_byod, "DEFAULT_SETTINGS": default_settings,
    }


def test_rr_m3_section_8_holds_only_the_default_sample_run_to_an_improvement(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    source = _cell(notebook, "delta_mrr = ")
    with pytest.raises(RuntimeError, match="adapted MRR must be above the frozen MRR"):
        exec(compile(source, "<section 8>", "exec"), _section_8_namespace({}, 0.8, False, True))
    for use_byod, default_settings in ((True, True), (False, False)):
        namespace = _section_8_namespace({}, 0.8, use_byod, default_settings)
        exec(compile(source, "<section 8>", "exec"), namespace)
        assert namespace["comparison"]["verdict"]["adapted_vs_frozen_mrr"] == "worse"
    report = json.loads((tmp_path / "outputs" / "qwen3_reranker_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["verdict"]["adapted_vs_frozen_mrr"] == "worse"


def test_rr_m3_default_settings_flag_matches_the_form_defaults(notebook):
    source = _cell(notebook, "DEFAULT_SETTINGS = ")
    head = source[: source.index("def report(")]
    namespace: dict = {}
    exec(compile(head, "<section 7 fields>", "exec"), namespace)
    assert namespace["DEFAULT_SETTINGS"] is True


# --- RR-M2: measurements of the frozen model always use the pinned base; adapted cells refuse the base ------------


class _Stop(Exception):
    pass


class _Pipe:
    def __init__(self, adapted: bool):
        self.adapter = {"trainable_names": ["model.layers.27.w"]} if adapted else None
        self.calls: list[str] = []

    def restore_base(self):
        self.calls.append("restore_base")
        was = self.adapter is not None
        self.adapter = None
        return ["model.layers.27.w"] if was else []

    def rerank(self, pairs, instruction=None):
        self.calls.append(f"rerank:adapted={self.adapter is not None}")
        raise _Stop


def test_rr_m2_section_5_puts_the_pinned_base_back_before_reranking(notebook, tmp_path, monkeypatch, capsys):
    source = _cell(notebook, "probe_records = test_records[:3]")
    assert source.index("pipe.restore_base()") < source.index("pipe.rerank(")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    pipe = _Pipe(adapted=True)
    namespace = {name: getattr(pl, name) for name in dir(pl) if not name.startswith("__")}
    namespace.update(pipe=pipe, test_records=_records(3), INSTRUCTION="i", json=json, candidate_list=metrics.candidate_list)
    with pytest.raises(_Stop):
        exec(compile(source, "<section 5>", "exec"), namespace)
    assert pipe.calls[:2] == ["restore_base", "rerank:adapted=False"]
    out = capsys.readouterr().out
    assert "restored_pinned_base" in out


def test_rr_m2_section_6_restores_before_the_frozen_evaluation(notebook):
    source = _cell(notebook, "frozen_vs_floor = ")
    assert source.index("pipe.restore_base()") < source.index("frozen_test = pipe.evaluate(")


@pytest.mark.parametrize("marker", ["delta_mrr = ", "pipe.save_artifact("])
def test_rr_m2_sections_8_and_9_refuse_the_pinned_base(notebook, marker):
    source = _cell(notebook, marker)
    guard = source[source.index("if pipe.adapter is None:") :]
    guard = guard[: guard.index("\n", guard.index("raise RuntimeError")) + 1]
    with pytest.raises(RuntimeError, match="holds the pinned base"):
        exec(compile(guard, "<guard>", "exec"), {"pipe": types.SimpleNamespace(adapter=None)})
    exec(compile(guard, "<guard>", "exec"), {"pipe": types.SimpleNamespace(adapter={})})
    assert source.index("if pipe.adapter is None:") < source.index("pipe.")


def test_rr_m2_opening_and_activity_say_which_cells_to_rerun(notebook):
    md = _markdown(notebook)
    assert "Sections 5 and 6 put the pinned base back before they measure" in md
    assert "Run Section 7, then Section 8." in md
    assert "Section 8 or 9 says the pipeline holds the pinned base" in md
    assert "re-run from Section 5 to re-read the frozen numbers" in md


# --- RR-M4: a Predict -> Change -> Run -> Observe -> Explain activity --------------------------------------------


def test_rr_m4_activity_follows_the_five_steps(notebook):
    md = _markdown(notebook)
    start = md.index("## Activity: how much does the second trainable layer buy?")
    steps = [md.index(f"{i}. **{name}.**", start) for i, name in enumerate(("Predict", "Change", "Run", "Observe", "Explain"), 1)]
    assert steps == sorted(steps)
    assert "<details><summary>Check your reasoning</summary>" in md[steps[-1] :]
    assert "To put the notebook back to the recorded state" in md
    assert "Optional experiments (they do not affect the default path)" not in md


# --- RR-m1: device-labelled figures; RR-m2: no template escapes; RR-m3: lexical baseline; RR-m4; RR-m5 ---------------


def test_rr_m1_quoted_figures_name_their_device(notebook):
    md = _markdown(notebook)
    assert "+4.5 points in a few minutes on CPU" not in md and "a 126 MB adapter" not in md
    assert "+4.5 in the float32 CPU build record, +3.2 on the recorded Kaggle Tesla T4 run" in md
    assert "The CPU float32 build record's sweep on this sample" in md
    assert "the recorded Kaggle Tesla T4 run of the default reached 76.6 %" in md
    assert "about 126 MB in float32 on CPU" in md and "63 MB in bfloat16 on CUDA" in md


def test_rr_m2_minor_no_template_escapes_in_markdown(notebook):
    md = _markdown(notebook)
    assert "{{" not in md and "}}" not in md
    assert "`[A-Za-z0-9_.:-]{1,64}`" in md


def test_rr_m3_minor_lexical_baseline_is_described_as_weakened_by_construction(notebook):
    md = _markdown(notebook)
    assert "deliberately hard to beat" not in md
    assert "weakened by construction and can score **below the random floor**" in md
    assert "(0.4156 against 0.5) and recall@5 (0.7532 against 0.8333)" in md
    assert "a sanity reference, not a" in md


def test_rr_m4_minor_printed_ceilings_match_what_adapt_enforces(notebook):
    source = _cell(notebook, "probe_records = test_records[:3]")
    assert "'TRAIN_CANDIDATES_RANGE': TRAIN_CANDIDATES_RANGE" in source
    assert "'defaults': {'TRAIN_CANDIDATES': MAX_TRAIN_CANDIDATES}" in source
    assert "'MAX_TRAIN_CANDIDATES': MAX_TRAIN_CANDIDATES}, 'yes_no" not in source
    assert pl.TRAIN_CANDIDATES_RANGE == (2, 8) and pl.MAX_TRAIN_CANDIDATES == 4
    pipe = pl.Qwen3RerankerPipeline(lambda bodies: None, "cpu")
    for bad in (1, 9):
        with pytest.raises(ValueError, match=r"train_candidates must be an int in 2\.\.8"):
            pipe.adapt(_records(12), _records(2), train_candidates=bad)


def test_rr_m5_minor_notebook_declares_spec_2_2(notebook):
    assert notebook["metadata"]["dimer"]["notebook_spec"] == "2.2"
    md = _markdown(notebook)
    assert "DIMER Notebook Specification 2.2" in md and "DIMER Notebook Specification 2.0" not in md
    assert "NOTEBOOK_SPEC 2.0" not in md
