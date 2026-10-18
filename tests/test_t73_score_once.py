"""Row 73 tutorial review (2026-10-17): the pre-registered grid, the fresh test, both tie policies and the
score-once wiring (SCORE_ONCE_RULE items 1-8), the DAT9 / data-source / REL13 wording, and provenance.

Only CI's dependencies are used: the notebook's own cell sources run against the carried modules with stand-ins
for the model. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import os
import subprocess
import types
from pathlib import Path

import pytest

from qwen3_reranker_pipeline import metrics, protocol, samples

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "qwen3_reranker_colab.ipynb"
CACHE = ROOT / "weights" / "banking77"


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


# --- T73-3: both tie policies ------------------------------------------------------------------------------------


def test_t73_3_expected_tie_metrics_are_exact():
    # positive tied with two negatives, one negative above: ranks 2, 3 or 4 with probability 1/3 each
    m = metrics.score_lists_metrics([[0.5, 0.9, 0.5, 0.5, 0.1, 0.0]])
    assert m["recall@1"] == 0.0 and m["mrr"] == pytest.approx(1 / 4)  # pessimistic: rank 4
    assert m["tie_expected"]["recall@1"] == 0.0
    assert m["tie_expected"]["recall@3"] == pytest.approx(2 / 3)
    assert m["tie_expected"]["mrr"] == pytest.approx((1 / 2 + 1 / 3 + 1 / 4) / 3)
    assert m["ties_with_positive"] == 1
    untied = metrics.score_lists_metrics([[0.9, 0.5, 0.1]])
    assert untied["tie_expected"]["mrr"] == untied["mrr"] == 1.0 and untied["ties_with_positive"] == 0


def test_t73_3_all_tied_list_matches_the_random_floor():
    m = metrics.score_lists_metrics([[0.0] * 6])
    floor = metrics.random_floor([6])
    for k in ("recall@1", "recall@3", "recall@5", "mrr"):
        assert m["tie_expected"][k] == pytest.approx(floor[k])


def test_t73_3_notebook_prints_both_policies_and_names_ties(notebook):
    source = _cell(notebook, "def brief(m):")
    assert "'tie_expected'" in source and "'ties_with_positive'" in source
    md = _markdown(notebook)
    assert "crowd its top three" not in md
    assert "86 of the 154 shortlists have a negative tied with the gold phrase" in md
    assert "(the bfloat16 T4 run of 2026-10-18 had one such tie), so read that count for the model too" in md


# --- T73-1 / T73-5: grid selection on validation ------------------------------------------------------------------


def test_t73_1_default_stays_unless_beaten_by_the_margin():
    sel = protocol.select_lr({1e-5: 0.920, 2e-5: 0.923, 5e-5: 0.930}, 0.881)
    assert sel["selected_lr"] == 2e-5 and sel["adaptation_kept"] and sel["evaluated_system"] == "adapted"
    sel = protocol.select_lr({1e-5: 0.920, 2e-5: 0.923, 5e-5: 0.933}, 0.881)
    assert sel["selected_lr"] == 5e-5


def test_t73_1_both_challengers_within_margin_pick_the_lower_lr():
    sel = protocol.select_lr({1e-5: 0.940, 2e-5: 0.923, 5e-5: 0.945}, 0.881)
    assert sel["selected_lr"] == 1e-5
    sel = protocol.select_lr({1e-5: 0.935, 2e-5: 0.923, 5e-5: 0.950}, 0.881)
    assert sel["selected_lr"] == 5e-5


def test_t73_1_keep_frozen_unless_beaten_by_the_margin():
    sel = protocol.select_lr({1e-5: 0.880, 2e-5: 0.885, 5e-5: 0.884}, 0.881)
    assert not sel["adaptation_kept"] and sel["evaluated_system"] == "frozen"
    with pytest.raises(ValueError, match="grid scores must cover exactly"):
        protocol.select_lr({2e-5: 0.9}, 0.8)


def test_t73_1_grid_reads_validation_only(notebook):
    source = _cell(notebook, "grid_selection = select_lr(")
    grid = source[source.index("if RUN_GRID:") : source.index("adapt_result = pipe.adapt(train_records, val_records, instruction=INSTRUCTION, epochs=EPOCHS, lr=lr_used")]
    assert "test_records" not in grid and "fresh_test_records" not in grid
    md = _markdown(notebook)
    assert "The test split was never used for training or epoch selection" not in md
    assert "on an independent split" not in md and "independent test split" not in md
    assert "**exposed development test**" in md


# --- T73-5: score-once wiring (rule items 1-8) ----------------------------------------------------------------------


FINGERPRINT = protocol.run_fingerprint(data_source="Banking77 sample", split_seed=42, train_digest="t", validation_digest="v", instruction=samples.TASK_INSTRUCTION, epochs=2, batch_size=4, trainable_layers=2, train_candidates=4, adapt_seed=0)
SETTINGS = {"data_source": "Banking77 sample", "split_seed": 42, "fresh_test_digest": samples.FRESH_TEST_DIGEST, "instruction": samples.TASK_INSTRUCTION, "epochs": 2, "batch_size": 4, "trainable_layers": 2, "train_candidates": 4, "adapt_seed": 0, "max_train_tokens": 192, "grid_run": True, "lr": 2e-5}


def _selection():
    return protocol.select_lr({1e-5: 0.91, 2e-5: 0.923, 5e-5: 0.92}, 0.881, fingerprint=FINGERPRINT)


def test_t73_5_identical_settings_are_preregistered_and_then_a_reproduction():
    assert protocol.settings_differing(SETTINGS, _selection(), FINGERPRINT) == []
    first = protocol.look_at_test([])
    assert first == "pre-registered evaluation (this runtime)"
    assert protocol.look_at_test([], {"test_look": first}) == "reproduction (this runtime)"


@pytest.mark.parametrize(("key", "value"), [("epochs", 3), ("batch_size", 2), ("adapt_seed", 1), ("data_source", "BYOD"), ("split_seed", 7), ("trainable_layers", 1), ("train_candidates", 6), ("instruction", "other"), ("grid_run", False)])
def test_t73_5_a_changed_setting_never_scores_the_fresh_test(key, value):
    differing = protocol.settings_differing({**SETTINGS, key: value}, _selection(), FINGERPRINT)
    assert key in differing
    look = protocol.look_at_test(differing)
    assert look == "not scored (exploratory)" and not protocol.fresh_test_scored(look)


def test_t73_5_a_changed_lr_or_a_stale_selection_never_scores_the_fresh_test():
    assert any(d.startswith("lr (used") for d in protocol.settings_differing({**SETTINGS, "lr": 5e-5}, _selection(), FINGERPRINT))
    stale = protocol.settings_differing(SETTINGS, _selection(), {**FINGERPRINT, "validation_digest": "other"})
    assert any(d.startswith("grid selection") for d in stale)
    assert protocol.settings_differing(SETTINGS, None, FINGERPRINT) == ["lr (no grid selection in this run)"]


def test_t73_5_masked_development_check_does_not_score():
    look = protocol.look_at_test([], masked=True)
    assert look == "not scored (fresh test masked for a development check)" and not protocol.fresh_test_scored(look)


def test_t73_5_output_suffixes_never_overwrite(tmp_path):
    path = tmp_path / "r.json"
    assert protocol.output_suffix(path, protocol.TEST_LOOKS[0]) == ""
    assert protocol.output_suffix(path, protocol.TEST_LOOKS[2]) == "_exploratory_1"
    (tmp_path / "r_exploratory_1.json").write_text("{}")
    assert protocol.output_suffix(path, protocol.TEST_LOOKS[2]) == "_exploratory_2"
    assert protocol.output_suffix(path, protocol.TEST_LOOKS[1]) == "_reproduction_1"


def test_t73_5_adapter_mismatch_names_the_differing_keys():
    run = {"lr": 2e-5, "epochs": 2, "batch_size": 4, "trainable_layers": 2, "train_candidates": 4, "seed": 0, "best_epoch": 2, "n_train": 231}
    assert protocol.adapter_mismatch(dict(run), run) == []
    assert "best_epoch" in protocol.adapter_mismatch({**run, "best_epoch": 1}, run)[0]
    assert protocol.adapter_mismatch(None, run)


def _metrics_stub(r1, mrr, ranks=(1, 2, 1)):
    return {"recall@1": r1, "recall@3": r1, "recall@5": r1, "mrr": mrr, "median_rank": 1, "n_queries": 3, "ties_with_positive": 0, "tie_expected": {"recall@1": r1, "recall@3": r1, "recall@5": r1, "mrr": mrr}, "ranks": list(ranks)}


RUN = {"lr": 2e-5, "epochs": 2, "batch_size": 4, "trainable_layers": 2, "train_candidates": 4, "seed": 0, "best_epoch": 2, "n_train": 231, "instruction": samples.TASK_INSTRUCTION, "max_train_tokens": 192, "history": [], "trainable_names": []}


def _section_8(notebook, tmp_path, run_overrides=None, exposed_adapted=(0.8, 0.88), **overrides):
    fresh = [{"id": f"fresh-{i:04d}", "query": f"q{i}", "positive": "p", "negatives": ["a", "b"]} for i in range(3)]
    calls = []
    run = {**RUN, **(run_overrides or {})}

    def evaluate(records, **kw):
        calls.append(("adapted", records))
        return _metrics_stub(*exposed_adapted) if records == ["dev"] else _metrics_stub(0.8, 0.88, (1, 1, 2))

    def evaluate_base(records, **kw):
        calls.append(("frozen", records))
        return _metrics_stub(0.7, 0.84)

    pipe = types.SimpleNamespace(adapter=dict(run), device="cuda:0", evaluate=evaluate, evaluate_base=evaluate_base, lexical_baseline=lambda records: _metrics_stub(0.3, 0.5))
    ns = {
        **{k: v for k, v in vars(protocol).items() if not k.startswith("__")},
        "json": json, "os": os, "compute_dtype": __import__("qwen3_reranker_pipeline.pipeline", fromlist=["compute_dtype"]).compute_dtype, "Path": Path, "random_floor": metrics.random_floor, "pipe": pipe, "paired_bootstrap": metrics.paired_bootstrap, "BOOTSTRAP_RESAMPLES": metrics.BOOTSTRAP_RESAMPLES, "BOOTSTRAP_SEED": metrics.BOOTSTRAP_SEED,
        "test_records": ["dev"], "val_records": ["val"], "fresh_test_records": fresh, "fresh_draw": {"digest": samples.FRESH_TEST_DIGEST},
        "INSTRUCTION": samples.TASK_INSTRUCTION, "floor": _metrics_stub(0.1667, 0.4083), "baseline_lexical": _metrics_stub(0.27, 0.46), "frozen_test": _metrics_stub(0.73, 0.84), "frozen_vs_floor": "above",
        "MODEL_ID": "m", "MODEL_REVISION": "r", "MODEL_KEY": "k", "data_source": "stand-in", "data_source_kind": "Banking77 sample",
        "dataset_manifests": {"train": {"digest": "t"}, "validation": {"digest": "v"}, "test": {"digest": "d", "candidates": 6}}, "disjoint": {}, "adapt_result": dict(run), "adapt_seconds": 0.0,
        "grid_selection": _selection(), "fingerprint": FINGERPRINT, "USE_BYOD": False, "DEFAULT_SETTINGS": True,
        "SPLIT_SEED": 42, "EPOCHS": 2, "BATCH_SIZE": 4, "TRAINABLE_LAYERS": 2, "TRAIN_CANDIDATES": 4, "ADAPT_SEED": 0, "RUN_GRID": True, "MAX_TRAIN_TOKENS": 192,
    }
    ns.update(overrides)
    exec(compile(_cell(notebook, "delta_mrr = "), "<section 8>", "exec"), ns)
    fresh_calls = [who for who, records in calls if records is fresh]
    return ns, fresh_calls


def test_t73_5_section_8_scores_the_fresh_test_only_when_preregistered(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DIMER_MASK_FRESH_TEST", raising=False)
    (tmp_path / "outputs").mkdir()
    ns, fresh_calls = _section_8(notebook, tmp_path)
    assert ns["test_look"] == "pre-registered evaluation (this runtime)" and fresh_calls == ["frozen", "adapted"]
    report = json.loads((tmp_path / "outputs" / "qwen3_reranker_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["test_look"] == ns["test_look"] and report["settings_differing"] == [] and report["fresh_test"]["n_queries"] == 3
    # an identical second scoring in the same runtime is a reproduction with its own file
    ns2, fresh_calls = _section_8(notebook, tmp_path, fresh_test_look=ns["fresh_test_look"])
    assert ns2["test_look"] == "reproduction (this runtime)" and fresh_calls == ["frozen", "adapted"]
    assert (tmp_path / "outputs" / "qwen3_reranker_evaluation_report_reproduction_1.json").is_file()


@pytest.mark.parametrize(("run_override", "override"), [({"epochs": 3}, {}), ({"batch_size": 2}, {}), ({"seed": 1}, {}), ({"lr": 5e-5}, {}), ({}, {"data_source_kind": "BYOD"}), ({}, {"RUN_GRID": False}), ({}, {"grid_selection": None})])
def test_t73_5_section_8_exploratory_runs_never_touch_the_fresh_test(notebook, tmp_path, monkeypatch, run_override, override):
    # F8: the training settings are read from the model in memory (pipe.adapter), not from the form fields.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    ns, fresh_calls = _section_8(notebook, tmp_path, run_overrides=run_override, **override)
    assert ns["test_look"] == "not scored (exploratory)" and fresh_calls == [] and ns["fresh_result"] is None
    assert not (tmp_path / "outputs" / "qwen3_reranker_evaluation_report.json").exists()
    assert (tmp_path / "outputs" / "qwen3_reranker_evaluation_report_exploratory_1.json").is_file()


def test_t73_5_section_8_masked_and_wrong_model_state(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    monkeypatch.setenv("DIMER_MASK_FRESH_TEST", "1")
    ns, fresh_calls = _section_8(notebook, tmp_path)
    assert ns["test_look"].startswith("not scored (fresh test masked") and fresh_calls == []
    monkeypatch.delenv("DIMER_MASK_FRESH_TEST")
    ns, fresh_calls = _section_8(notebook, tmp_path, adapt_result={**RUN, "best_epoch": 1})
    assert ns["test_look"] == "not scored (exploratory)" and fresh_calls == []
    assert any("model state" in d for d in ns["differing"])


def test_t73_5_frozen_kept_scores_only_the_frozen_model(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DIMER_MASK_FRESH_TEST", raising=False)
    (tmp_path / "outputs").mkdir()
    selection = protocol.select_lr({1e-5: 0.880, 2e-5: 0.885, 5e-5: 0.884}, 0.881, fingerprint=FINGERPRINT)
    ns, fresh_calls = _section_8(notebook, tmp_path, grid_selection=selection)
    assert fresh_calls == ["frozen"] and ns["fresh_result"]["evaluated_system"] == "frozen"


def test_t73_r2_f8_form_fields_do_not_decide_and_a_stale_fingerprint_fires(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DIMER_MASK_FRESH_TEST", raising=False)
    (tmp_path / "outputs").mkdir()
    ns, fresh_calls = _section_8(notebook, tmp_path, EPOCHS=7, BATCH_SIZE=1)  # globals edited after training: irrelevant
    assert ns["test_look"] == "pre-registered evaluation (this runtime)" and fresh_calls == ["frozen", "adapted"]
    ns, fresh_calls = _section_8(notebook, tmp_path, dataset_manifests={"train": {"digest": "t"}, "validation": {"digest": "other"}, "test": {"digest": "d", "candidates": 6}})
    assert any(d.startswith("grid selection") for d in ns["differing"]) and fresh_calls == []


def test_t73_r2_f2_report_is_written_and_exposed_test_never_stops(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DIMER_MASK_FRESH_TEST", raising=False)
    (tmp_path / "outputs").mkdir()
    source = _cell(notebook, "delta_mrr = ")
    assert source.index("fresh_systems = ") < source.index("json.dump(evaluation_report_payload") and "raise RuntimeError" not in source[source.index("fresh_systems = "):]
    ns, fresh_calls = _section_8(notebook, tmp_path, exposed_adapted=(0.6, 0.70))  # adapted worse than frozen on the exposed test
    assert ns["comparison"]["verdict"]["adapted_vs_frozen_mrr"] == "worse" and fresh_calls == ["frozen", "adapted"]
    report = json.loads((tmp_path / "outputs" / "qwen3_reranker_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["fresh_test"]["n_queries"] == 3 and report["fresh_test"]["bootstrap"]["n_boot"] == metrics.BOOTSTRAP_RESAMPLES


def test_t73_r2_f7_device_and_dtype_are_recorded(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DIMER_MASK_FRESH_TEST", raising=False)
    (tmp_path / "outputs").mkdir()
    ns, _ = _section_8(notebook, tmp_path)
    assert ns["fresh_result"]["device"] == "cuda:0" and ns["fresh_result"]["dtype"] == "float32"  # amendment 2
    assert ns["run_settings"]["dtype"] == "float32" and "dtype" not in protocol.PREREGISTERED_SETTINGS


def test_t73_r2_f6_retrain_check():
    selection = {**_selection(), "runs": {2e-5: {"val_mrr": 0.9232}}}
    history = [{"val": {"mrr": 0.88}}, {"val": {"mrr": 0.90}}, {"val": {"mrr": 0.9230}}]
    ok = protocol.retrain_check(selection, {**RUN, "history": history})
    assert ok["within"] and protocol.retrain_mismatch(ok) == []
    history[2]["val"]["mrr"] = 0.905
    bad = protocol.retrain_check(selection, {**RUN, "history": history})
    assert not bad["within"] and "re-train" in protocol.retrain_mismatch(bad)[0]
    assert protocol.retrain_check(None, RUN) is None


def test_t73_r2_f5_paired_bootstrap_is_seeded_paired_and_gives_ci_verdicts():
    frozen = [1] * 60 + [2] * 40
    adapted = [1] * 90 + [2] * 10
    a = metrics.paired_bootstrap({"frozen": frozen, "adapted": adapted}, n_boot=500, seed=7)
    b = metrics.paired_bootstrap({"frozen": frozen, "adapted": adapted}, n_boot=500, seed=7)
    assert a == b and a["difference"]["recall@1"]["point"] == pytest.approx(0.3)
    lo, hi = a["difference"]["recall@1"]["ci95"]
    assert 0 < lo <= 0.3 <= hi and a["difference"]["recall@1"]["verdict"] == "improved"
    same = metrics.paired_bootstrap({"frozen": frozen, "adapted": list(frozen)}, n_boot=200, seed=7)
    assert same["difference"]["mrr"]["ci95"] == [0.0, 0.0] and same["difference"]["mrr"]["verdict"] == "inconclusive"
    assert metrics.ci_verdict([-0.2, -0.01]) == "worse"
    only = metrics.paired_bootstrap({"frozen": frozen}, n_boot=50, seed=7)
    assert "difference" not in only and only["systems"]["frozen"]["recall@1"]["point"] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        metrics.paired_bootstrap({"frozen": frozen, "adapted": adapted[:5]}, n_boot=10)


def test_t73_r2_prose_amendment_items(notebook):
    md = _markdown(notebook)
    assert "**paired bootstrap 95 % CI**" in md and "decides nothing — no stop reads a test split" in md
    assert "is trained **again**" in md and "*keep the default unless clearly beaten*" in md
    assert "10 / 385 fresh" in md and "about 11.5 minutes on a T4" in md and "75 minutes or more" in md
    assert "give no dispersion estimate" not in md and "7–8 minutes" not in md
    assert (ROOT / "docs" / "qwen3-reranker-preregistration-amendment-1.md").is_file()


def test_t73_5_no_permanent_scored_once_wording(notebook):
    text = json.dumps(notebook) + (ROOT / "src" / "qwen3_reranker_pipeline" / "protocol.py").read_text(encoding="utf-8")
    assert "scored once" not in text.lower() and "SCORED_ONCE" not in text


# --- T73-10: the fresh test draw -------------------------------------------------------------------------------------


def _synthetic_corpus():
    intents = [f"intent_{i:02d}" for i in range(samples.CORPUS_INTENTS)]
    train = [{"id": f"train-{i:05d}", "text": f"train message {it} r{k} word{k}", "intent": it} for i, (it, k) in enumerate((it, k) for it in intents for k in range(6))]
    test = [{"id": f"test-{i:05d}", "text": f"test message {it} t{k} term{k}", "intent": it} for i, (it, k) in enumerate((it, k) for it in intents for k in range(12))]
    test.append({"id": "test-dup", "text": "TRAIN message intent_00 r0 word0!", "intent": intents[0]})  # equals a train.csv text
    return {"train": train, "test": test}


def test_t73_10_fresh_test_leaves_the_original_splits_unchanged_and_is_disjoint():
    corpus = _synthetic_corpus()
    sizes = {"train": 154, "validation": 77, "test": 154}
    splits = samples.build_sample_dataset(corpus, seed=42, sizes=sizes)
    fresh = samples.build_fresh_test(corpus, splits, seed=42, per_intent=5)
    assert samples.build_sample_dataset(corpus, seed=42, sizes=sizes) == splits
    assert len(fresh["records"]) == 385 and fresh["offset"] == 2
    used = {samples.normalise(r["query"]) for part in splits.values() for r in part}
    assert not used & {samples.normalise(r["query"]) for r in fresh["records"]}
    assert all(samples.normalise(r["query"]) != "train message intent_00 r0 word0" for r in fresh["records"])
    assert {r["id"] for r in fresh["records"]} == {f"fresh-{i:04d}" for i in range(385)}
    with pytest.raises(ValueError, match="not the pinned draw"):
        samples.check_fresh_test(fresh, splits)


def test_t73_10_check_fresh_test_refuses_overlap_and_near_duplicates(monkeypatch):
    corpus = _synthetic_corpus()
    splits = samples.build_sample_dataset(corpus, seed=42, sizes={"train": 154, "validation": 77, "test": 154})
    fresh = samples.build_fresh_test(corpus, splits, seed=42, per_intent=1)
    monkeypatch.setattr(samples, "FRESH_TEST_SOURCE_IDS_SHA256", fresh["source_ids_sha256"])
    monkeypatch.setattr(samples, "FRESH_TEST_DIGEST", fresh["digest"])
    assert samples.check_fresh_test(fresh, splits)["n_records"] == 77
    near = splits["train"][0]["query"] + " please"
    bad = {**fresh, "records": [*fresh["records"][:-1], {**fresh["records"][-1], "query": near}]}
    with pytest.raises(ValueError, match="near-duplicates"):
        samples.check_fresh_test(bad, splits)
    bad = {**fresh, "records": [*fresh["records"][:-1], {**fresh["records"][-1], "query": splits["test"][0]["query"]}]}
    with pytest.raises(ValueError, match="already in a split"):
        samples.check_fresh_test(bad, splits)


@pytest.mark.skipif(not (CACHE / "test.csv").is_file(), reason="Banking77 cache not present (CI); the notebook checks the pins at runtime")
def test_t73_10_pinned_draw_on_the_real_corpus():
    corpus = samples.read_corpus(samples.fetch_corpus(cache_dir=CACHE))
    splits = samples.build_sample_dataset(corpus, seed=42)
    assert {k: samples.dataset_digest(v)[:8] for k, v in splits.items()} == {"train": "6a5c384c", "validation": "32cc0b97", "test": "a67a92bd"}
    fresh = samples.build_fresh_test(corpus, splits, seed=42)
    check = samples.check_fresh_test(fresh, splits)
    assert check["n_records"] == 385 and check["skipped"] == {"already_in_a_split": 0, "text_in_train_csv": 2, "near_duplicate_of_train_or_validation": 4}
    lexical = metrics.lexical_baseline(splits["test"])  # the exposed development test, never the fresh one
    assert round(lexical["recall@1"], 4) == 0.2662 and round(lexical["tie_expected"]["recall@1"], 4) == 0.3335
    assert lexical["ties_with_positive"] == 86


# --- T73-2 / T73-7 / T73-4 / T73-8 / T73-6: wording, evidence logs, provenance ---------------------------------------


def test_t73_2_dat9_statement_replaces_never_tuned_for(notebook):
    md = _markdown(notebook)
    corpus = md + (ROOT / "src" / "qwen3_reranker_pipeline" / "samples.py").read_text(encoding="utf-8") + (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "never tuned for" not in corpus and "not tuned for" not in corpus
    assert "**Possible training overlap (DAT9):** Banking77 has been public since 2020 and is an MTEB task" in md
    assert "overlap with these messages cannot be ruled out" in md


def test_t73_7_data_source_names_github(notebook):
    md = _markdown(notebook)
    assert "from the project repository" not in md and "No GitHub access" not in md
    assert "GitHub raw content (`raw.githubusercontent.com`)" in md


def test_t73_8_no_local_paths_in_execution_evidence():
    for path in (ROOT / "docs" / "execution-evidence").rglob("*"):
        if path.is_file() and path.suffix in {".log", ".json", ".ipynb", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            assert "/home/" not in text and "Users\\\\Kurt" not in text and "Users/Kurt" not in text, path


def test_t73_6_stamped_revision_carries_the_embedded_modules(notebook):
    stamp = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    source_cell = _cell(notebook, "NOTEBOOK_SOURCE = {")
    assert f"'repository_revision': {stamp!r}" in source_cell
    modules = json.loads(json.dumps(notebook["metadata"]["dimer"]["generated_from"]))
    rels = [m for m in modules.get("modules", [])] or ["src/qwen3_reranker_pipeline/" + m for m in ("pipeline.py", "metrics.py", "samples.py", "protocol.py")]
    for rel in rels:
        try:
            at_stamp = subprocess.run(["git", "-C", str(ROOT), "show", f"{stamp}:{rel}"], capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip(f"revision {stamp[:12]} is not in this clone (shallow CI checkout)")
        assert at_stamp.replace(b"\r\n", b"\n") == (ROOT / rel).read_bytes().replace(b"\r\n", b"\n"), f"{rel} differs from the stamped revision {stamp[:12]}"
