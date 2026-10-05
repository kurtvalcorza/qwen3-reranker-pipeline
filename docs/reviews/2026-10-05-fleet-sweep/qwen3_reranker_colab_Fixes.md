# Fleet-sweep fixes: `qwen3_reranker_colab.ipynb` (2026-10-05)

A targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report; each flag was first
confirmed in the cell source at `main` `f13a58e`. Changes are made in the generator (`tools/build_notebook.py`,
`tools/notebook_template.py`) and, for SWP-F, in the carried `pipeline.py`; the notebook is regenerated. Status and release
labels are unchanged.

**Readiness: Verification pending** (until a hosted Run all of the regenerated notebook is recorded).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed: Section 1 pip-installed the pins into the kernel and raised "Restart the runtime" on stale modules. Generator → `build_notebook.py/2.2`; the template opts in. One kernel cell verifies and runs the pinned `uv` 0.12.15, builds a managed CPython 3.12.12 environment from `tutorials/requirements-colab.lock.txt` (47 packages; same pins and lock as qwen3-embedding-pipeline; `--require-hashes --only-binary :all:`), keys the folder on the lock digest and reuses it, keeps a live worker on re-run, forces `MPLBACKEND=Agg` and drops `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. | Section 1; generator, template, validator, new lock | `test_swp_r_*` (3 tests) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED with 0 of 9 guided markers. Added audience, Input → Model → Output, How to use, roadmap, Predict prompts (Sections 4–8), What to notice + Check your reasoning after Sections 4–9 quoting the recorded Kaggle T4 run of 2026-09-19 (recall@1 floor 0.1667 / lexical 0.2662 / frozen 0.7338 / adapted 0.7662; recall@3 lexical 0.4156 below the 0.5 floor; MRR 0.4083 / 0.457 / 0.8429 / 0.8634; reload MRR 0.9125 both) and naming the float32 CPU build record the prose quotes, Troubleshooting, Glossary, Conclusion template; infrastructure labelled and collapsed. Literal `{{…}}` in the BYOD statement and the Prerequisites now render as `{…}`. | opening, Sections 4–9 markdown, closing, header, Prerequisites | `test_swp_g_*` (2 tests) |
| SWP-A (quality asserts) | Fixed | The sweep's one flagged line is the Section 9 reload-parity check (`parity['scores_identical'] …`): a contract check, kept. The cell source also asserted `frozen MRR > random floor` (Section 6) and `adapted MRR > frozen MRR` (Section 8), which abort a BYOD run before export; both are recorded verdicts now (evaluation report and `result.json`), and the shortlist-count equality in Section 6 stays a hard contract check. | Sections 6, 8, 9 | `test_swp_a_*` (5 tests) |
| SWP-F (frozen re-run) | Fixed | Confirmed: `adapt()` trained the last decoder layers in place from whatever weights the model held, so a Section 7 re-run (the closing's optional experiments) continued training while epoch 0 was labelled "frozen model". `pipeline.py` now applies siglip-v1's `restore_base` pattern (dataclass field `_base_state`); a failed adapt leaves the weights as before; the result records `started_from` (printed in Section 7). | `src/qwen3_reranker_pipeline/pipeline.py`, Section 7 | `test_swp_f_*` (2 tests) |
| SWP-B (BYOD upload only) | Fixed | Confirmed: BYOD used only `files.upload()`. Added `BYOD_PATH` (one .csv/.json/.jsonl file; Kaggle/Jupyter); guarded upload fallback (off Colab, cancelled, multi-file, wrong extension each name the file or rule). | Section 4 | `test_swp_b_*` (3 tests) |

## User-visible changes

- Section 1 installs nothing into the kernel and never asks for a restart (first build takes several minutes; reused afterwards). Linux x86_64 only.
- `Qwen3RerankerPipeline.adapt()` and `load_artifact()` always start from the pinned base; new `restore_base()`; `adapt()` result has `started_from`.
- New `BYOD_PATH` field; Sections 6 and 8 print verdicts instead of raising; the evaluation report and `result.json` carry `verdict`.
- Guided-layer cells; infrastructure collapsed.

## Verification (offline; not clean-runtime evidence)

- No model stage can run here (Hub unreachable). The Section 1 cell runs for real against a stand-in environment; the BYOD block and Sections 6 and 8 run with stand-ins; `restore_base` runs on a NumPy stand-in model. Plumbing evidence, not model evidence.
- `pytest` with CI's dependencies only (pytest, ruff, numpy; torch absent): 51 passed, 1 skipped before → 68 passed, 1 skipped after.
- `build_notebook.py --check` up to date; `validate_release_assets.py` PASS; `ruff check src tests tools` clean.
- Sweep re-check on the regenerated notebook: isolated runtime, guided markers 9/9; the only line its metric pattern still matches is the reload-parity contract check.

## Remaining gates

- A hosted **Run all in one pass** in a fresh Colab T4 runtime (no restart expected), then a re-run of the Section 9 export cell.
- The REL12 BYOD run (`USE_BYOD = True` with `BYOD_PATH`).
- A full Notebook Review Framework v1 review has not been done.
