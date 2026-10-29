# Release verification

`tutorials/qwen3_reranker_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.5 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 13-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the single kernel cell that builds (or reuses) the
  hash-locked uv environment and routes every later cell to it, with no in-kernel install and no restart request; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned Banking77
  commit is the one allowed second hash);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Qwen3RerankerPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus` + `build_sample_dataset(seed=SPLIT_SEED)` / `load_byod_dataset`, `validate_dataset` per split,
  `check_split_disjoint`, `write_dataset_csv`, `validate_inputs` with the empty-document refusal probe, `pipe.rerank`
  on one shortlist with the sanity checks plus the frozen orderings of three shortlists, `random_floor`,
  `pipe.lexical_baseline`, `pipe.evaluate` on the frozen model with the floor assertion and on the validation and
  test splits after adaptation with the MRR assertion, `pipe.adapt` with its explicit hyperparameters,
  `trainable_layers=TRAINABLE_LAYERS`, `train_candidates=TRAIN_CANDIDATES` and `instruction=INSTRUCTION`, the
  per-batch `evaluation_report`, `pipe.save_artifact`, `Qwen3RerankerPipeline.from_artifact` and the reload-parity
  assertion, and the provenance fields `weight_format`, `weight_sha256` and the `corpus` block), the six expected
  `outputs/` paths, the learner-facing statements (uncalibrated score, adaptation measured by ranking, the random
  floor, the lexical baseline, listwise, no dispersion estimate, scores not comparable across the frozen and adapted
  models, named exclusions, the CC BY 4.0 corpus licence) and the gated-off BYOD default; forbidden patterns
  (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a mutable
  `revision='main'`, direct `from transformers import` / `AutoModelForCausalLM` / `AutoTokenizer` / `.logits` /
  `from huggingface_hub import` / `urllib.request` / `safetensors` / `torch.optim` / `.backward(` / `pipe._model` /
  `cross_entropy(` use **outside the carried module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(`
  without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI installs only `pytest`, `ruff` and `numpy` plus the package without its model dependencies (no torch, no
transformers), runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`, `tests/test_import_boundary.py`,
`tests/test_notebook_parity.py`; injected bag-of-words runner and corpus fetcher, temporary manifests, no weights —
`tests/test_model_backed.py` is skipped without `transformers` and the snapshot). These are source/provenance and unit
checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; float32 on every device since amendment 2) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy or torch that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/qwen3-reranker-0.6b/` or the corpus cache `weights/banking77/` (the standalone path writes the
   manifest itself, stages the missing file from the Hub, and fetches the two pinned Banking77 files from the
   `PolyAI-LDN/task-specific-datasets` repository on GitHub, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `INSTRUCTION = 'Given a customer support message, judge whether the document
   names the banking intent it expresses'`, `EPOCHS = 2`, `RUN_GRID = True`, `LEARNING_RATE = 2e-5` (used only without the grid), `BATCH_SIZE = 4`,
   `TRAINABLE_LAYERS = 2`, `TRAIN_CANDIDATES = 4`, `ADAPT_SEED = 0`; the pre-registered settings of
   `docs/qwen3-reranker-preregistration.md`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`,
   `numpy==2.5.3`, as built from `tutorials/requirements-colab.lock.txt` in the isolated environment (nothing is
   installed into the kernel, and **no restart is allowed**: a run that needs one does not meet RUN10/ENV6);
5. verify every default-path stage completes:
   - isolated runtime built from the hash lock (`--require-hashes --only-binary :all:`) with no GitHub access;
   - the four carried module cells (`pipeline.py`, `metrics.py`, `samples.py`, `protocol.py`) execute (defining `Qwen3RerankerPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `format_pair`, `record_seed`, `rank_of_positive`,
     `ranking_metrics`, `random_floor`, `candidate_list`, `lexical_baseline`, `fetch_corpus`, `read_corpus`,
     `sample_negatives`, `build_sample_dataset`, `validate_dataset`, `documents`, `check_split_disjoint`,
     `split_dataset`, `load_byod_dataset`, `write_dataset_csv` and the ceilings) with no import of the repository
     package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['model.safetensors']` (and any other absent entry) fetched from
     `Qwen/Qwen3-Reranker-0.6B` at the immutable revision, and `verify_snapshot` returning its dict (13 files);
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with the `yes`/`no` ids checked;
   - Section 4: `fetch_corpus` fetching the two pinned files (839,073 / 239,961 bytes) from `raw.githubusercontent.com`
     into `weights/banking77/`, 10,003 + 3,080 raw rows read, and the seeded balanced draw of 231 / 77 / 154 records
     over 77 intents with six candidates per query, `check_split_disjoint` reporting no shared message, and the three
     dataset digests `6a5c384c…` / `32cc0b97…` / `a67a92bd…`; the fresh test drawn and checked (385 records, source-id SHA-256
     `f59a5459…`, digest `9bc26a46…`, 2 + 4 skipped) but not scored; `outputs/…_train.csv` written; the four dataset
     refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MAX_PAIRS` 32, `MAX_TEXT_TOKENS` 8192, `MAX_TEXT_CHARS` 100000, `MAX_TRAIN_TOKENS`
     192, `MAX_TRAIN_CANDIDATES` 4) and the `yes`/`no` ids surfaced; `validate_inputs` writing
     `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the empty-document
     probe); `rerank` on the first test shortlist with all five sanity checks `True` and the ranked shortlist printed
     with the gold marked, then three test shortlists ordered by the frozen model;
   - Section 6 (exposed development test): the random floor (recall@1 16.7 %, MRR ≈ 0.408), the lexical baseline
     (26.6 % recall@1 pessimistic, 33.4 % with ties broken at random; 86 of 154 shortlists tie with the gold phrase)
     and the frozen reranker's metrics (≈ 73.4 %
     recall@1, MRR ≈ 0.843, 924 pairs) on CPU float32, with the cell's assertion that the query counts agree
     and the frozen MRR is above the floor;
   - Section 7: the grid table (three runs on validation only) and `grid_selection`, then `pipe.adapt` printing epoch 0 as the frozen model, 31,461,888 trainable of 595,776,512 parameters,
     924 training pairs, and a two-epoch history with validation MRR rising (0.881 → 0.902 → 0.923 in the recorded run;
     `best_epoch` 2);
   - Section 8: `test_look` = `pre-registered evaluation (this runtime)` with an empty `settings_differing`; the
     fresh test scored for the random floor, the lexical baseline, the frozen model and the evaluated system (both tie
     policies); `pipe.evaluate` on validation and the exposed development test with the four-way comparison, and
     `outputs/…_evaluation_report.json` written right after the fresh test is scored, with the paired bootstrap
     95 % CIs (10,000 resamples, seed 20261018), the CI-based fresh-test verdict, `retrain_check`, `device` and `dtype`
     (amendment 1); the exposed-development verdict is printed and stops nothing;
   - Section 9: the three test shortlists ordered again by the adapted reranker and printed beside the frozen order
     with the gold rank and score in each, the per-batch `evaluation_report` verdict `not-measurable`,
     `outputs/…_reranking.csv` written; `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,
     manifest.json}` (22 tensors, about 126 MB, `instruction` recorded) and
     `Qwen3RerankerPipeline.from_artifact` reloading it with identical scores on the probe shortlist and an identical
     20-shortlist validation MRR (the cell asserts both); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the
     model identity and licence, the snapshot block (`weight_format`, `weight_sha256`), the instruction, the `corpus`
     block, the inference-contract items, the comparison, the before/after orderings, the artifact digest, the reload
     parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `qwen3_reranker_colab.ipynb` (`E2E`) | `ae03f56` / `91402f81` | 2026-10-18 | Colab CLI 0.7.4, fresh Colab Tesla T4 (session `suite-qwen3-ae03f56-745f`; CLI sequential execution, not a browser Run all; `torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda:0`, float32) | **PASSED — REL13 reproduction of the pre-registered evaluation** (of the `ff6da3a` record): 13/13 code cells, 0 errors, no restart, 994.7 s; fresh test (385) recall@1 0.7455 → 0.7922, MRR 0.851 → 0.879 (paired Δ MRR +0.0279, 95 % CI 0.0086 to 0.0476), equal to the record; see *Hosted T4 reproduction* below; evidence `docs/execution-evidence/2026-10-18-ae03f56/` |
| `qwen3_reranker_colab.ipynb` (`E2E`) | `ff6da3a` / `2d6e5607` | 2026-10-18 | Colab CLI 0.7.4, fresh Colab Tesla T4 (session `suite-qwen3-ff6da3a-745f`; CLI sequential execution, not a browser Run all; `torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda:0`, float32) | **PASSED — REL13 pre-registered evaluation of record**: 13/13 code cells, 0 errors, no restart, 995.4 s; fresh test (385) recall@1 0.7455 → 0.7922, MRR 0.851 → 0.879 (paired Δ MRR +0.0279, 95 % CI 0.0086 to 0.0476); see *Hosted T4 evaluation of record* below; evidence `docs/execution-evidence/2026-10-18-ff6da3a/` |
| `qwen3_reranker_colab.ipynb` (`E2E`) | `995cd8e` / `99efdc6a` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-qwen3-reranker` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 30 files, 1209 MB staged from the Hub into a clean cache; comparison {recall@1: {random_floor: 0.1667, lexical: 0.2662, frozen: 0.7338, adapted: 0.7662}, recall@3: {random_floor: 0.5, lexical: 0.4156, frozen: 0.9416, adapted: 0.9545}, recall@5: {random_floor: 0.8333, lexical: 0.7532, frozen: 0.9935, adapted: 0.9935}, mrr: {random_floor: 0.4083, lexical: 0.457, frozen: 0.8429, adapted: 0.8634}, median_rank: {lexical: 4, frozen: 1, adapted: 1}, delta_vs_frozen: {recall@1: 0.0325, recall@3: 0.013, recall@5: 0, mrr: 0.0206}}; reload parity {scores_identical: True, mrr_in_memory: 0.9125, mrr_reloaded: 0.9125}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-qwen3-reranker/v3/evidence/` in the workspace |
| `qwen3_reranker_colab.ipynb` (`E2E`) | `d79994f` / `75fbc7d7` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `qwen3_reranker_colab.ipynb` (`TASK-INFERENCE`, superseded) | `0c652a2` / `a21e5a860c0f` | 2026-09-14 | Kaggle T4 (`kurtvalcorza/dimer-nb2-qwen3-reranker` v2) | PASSED — 8/8 code cells, 236.4 s (1 restart after the install cell), 1,207 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/qwen3_reranker_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/qwen3_reranker_colab.ipynb`). Wall times are the sum of per-cell times reported by
the executor and include the model download where it occurred; they are measurements for the stated runtime, not
general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-18 (01:39:23 UTC start) | commit `ae03f56` / blob `91402f813fb03439e05b06237c1602dadd89ba8d` (`NOTEBOOK_SOURCE.repository_revision` `482d078`; generator `build_notebook.py/2.2`, `notebook_spec` 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab VM (session `suite-qwen3-ae03f56-745f`), Tesla T4; blob fetched at the 40-char SHA and Git-blob verified, no repository checkout; kernel Python 3.13.15; uv isolated environment CPython 3.12.12 (47 locked packages, 56 s); `torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda:0`, float32 | Default sample path only (`USE_BYOD = False`, default form values, `RUN_GRID = True`); 13 snapshot files fetched and digest-verified | 994.7 s | **PASSED — REL13 reproduction of the pre-registered evaluation** (`test_look` "reproduction (this runtime)", `settings_differing` [], outputs `_reproduction_1`): 13/13 code cells, 0 errors; every printed metric equal to the `ff6da3a` evaluation of record (grid 0.9165 / 0.9232 / 0.9229 → 2e-5; re-train 0.92316 both; fresh test frozen 0.7455 / 0.851, adapted 0.7922 / 0.879, Δ MRR +0.0279 [0.0086, 0.0476], Δ recall@1 +0.0468 [0.0130, 0.0805]; exposed adapted 0.7792 / 0.8725; reload parity 0.975; adapter 125,850,032 B, SHA-256 prefix `fd5a7d95eb8c4eb47` equal). Evidence [`docs/execution-evidence/2026-10-18-ae03f56/`](execution-evidence/2026-10-18-ae03f56/): executed notebook SHA-256 `501c3573926f32240143352dcc55c9cb83a0cb68e9886c6bf9e0b339417ab17b`, `run_summary.json` `04213176c96cc9421477b5ed0712117899c4e4a5c74be865442dec9250d559e3`, `exec.log` captured `d02c752a…`, committed (home-directory prefix replaced) `29ca03f9…`. Not exercised: BYOD (REL12), the activity and the optional experiments, a browser Run all |
| 2026-10-18 (00:21:48 UTC start) | commit `ff6da3a` / blob `2d6e5607e1354f209ff862a34d798810c6093d54` (`NOTEBOOK_SOURCE.repository_revision` `003e228`; generator `build_notebook.py/2.2`, `notebook_spec` 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab VM (session `suite-qwen3-ff6da3a-745f`), Tesla T4; blob fetched at the 40-char SHA and Git-blob verified, no repository checkout; kernel Python 3.13.15; uv isolated environment CPython 3.12.12 (47 locked packages, 52 s); `torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda:0`, float32 | Default sample path only (`USE_BYOD = False`, default form values, `RUN_GRID = True`); 13 snapshot files fetched and digest-verified | 995.4 s | **PASSED — REL13 pre-registered evaluation of record** (`test_look` "pre-registered evaluation (this runtime)", `settings_differing` []): 13/13 code cells, 0 errors; grid kept-epoch validation MRR 0.9165 / 0.9232 / 0.9229 → 2e-5; re-train 0.92316 both; fresh test (385) frozen 0.7455 / 0.851, adapted 0.7922 / 0.879 (recall@1 / MRR), Δ MRR +0.0279 [0.0086, 0.0476], Δ recall@1 +0.0468 [0.0130, 0.0805]; exposed development test adapted 0.7792 / 0.8725; reload parity exact (0.975); adapter 125,850,032 B. Evidence [`docs/execution-evidence/2026-10-18-ff6da3a/`](execution-evidence/2026-10-18-ff6da3a/): executed notebook SHA-256 `32d7ced5a4bda2ce215e75d6cd7f1e7c7ba14792d1326187cd5e4383ca18a2bc`, `run_summary.json` `276969abdb2c5619c9bdcf3b0daa11a6787e8881dc78f1e359739e4b07d3d901`, `exec.log` captured `e3eeb766…`, committed (home-directory prefix replaced) `c6870306…`. Not exercised: BYOD (REL12), the activity and the optional experiments, a browser Run all |
| 2026-10-09 (02:49:17 UTC start) | commit `d064746` / blob `93f1a9d552131fc1382151ce1a7e99261ea0cfb6` (`NOTEBOOK_SOURCE.repository_revision` `c628da8`, the source revision the notebook was generated from; `c628da8..d064746` is the review-fix commit; generator `build_notebook.py/2.2`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab VM (session `suite-qwen3-d064746-745f`), Tesla T4; committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout; kernel Python 3.13.15; uv isolated environment CPython 3.12.12 (47 locked packages, built in 75 s), `torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda:0`, bfloat16 | Default sample path only (`USE_BYOD = False`, default form values); 13 snapshot files fetched from the Hub and digest-verified | 513.7 s | **PASSED — one pass, no restart, 0 errors**: 12/12 code cells (the three carried-module cells print nothing); splits 231 / 77 / 154, four refusal probes rejected; ceilings print `TRAIN_CANDIDATES_RANGE` (2, 8) and `MAX_EVAL_RECORDS` 2000 with `TRAIN_CANDIDATES` 4 as the default; random floor recall@1 0.1667 / MRR 0.4083; lexical 0.2662 / 0.457 (recall@3 0.4156 below the 0.5 floor, recall@5 0.7532 below 0.8333); frozen 0.7338 / 0.8429 (`adapted: False`, 46.7 s); validation MRR 0.8699 → 0.8872 → 0.8937, best epoch 2 (`started_from` pinned base, 229.0 s); adapted 0.7662 / 0.8634 (Δ recall@1 +0.0325, MRR +0.0206, recall@5 0.9935 unchanged, verdict *improved*); 2 of 3 probe shortlists reordered, gold rank 1 in all; adapter 22 tensors, 62,926,256 B; reload parity exact (scores identical, MRR 0.9125 both ways); six exports. These match the worked answers. Evidence [`docs/execution-evidence/2026-10-09-d064746/`](execution-evidence/2026-10-09-d064746/): executed notebook SHA-256 `1b087bd151ea134ca42b2c65b40a9c891c11a7bfa4c18b4f60ca153a12137562`, `run_summary.json` `eace86e87a65159011b8651fcbcd44de0640cc65a8007f124ae95a5b3761d9ac`, `exec.log` as captured `a0f70489b797baf860ce599b029a9b117fa56c708c133e7cf153f5b0e191e734` (the blob before `9ed7d8a`); redacted 2026-10-17 in commit `9ed7d8a`, where one string class (a local home-directory prefix, replaced by `<local-run-dir>/` on lines 1 and 65) was replaced and nothing else changed, giving `17ed4e1bb33ec525c069b8c4abae09f98c2bda08bb274cfaa727087103fa1b3a` (verified: applying only that substitution to the captured file reproduces this hash). Not exercised: BYOD (REL12), the activity and the optional experiments, a browser Run all |
| 2026-09-19 | `995cd8e` / `99efdc6a` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-qwen3-reranker` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 588.2 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 30 files, 1209 MB staged from the Hub into a clean cache; comparison {recall@1: {random_floor: 0.1667, lexical: 0.2662, frozen: 0.7338, adapted: 0.7662}, recall@3: {random_floor: 0.5, lexical: 0.4156, frozen: 0.9416, adapted: 0.9545}, recall@5: {random_floor: 0.8333, lexical: 0.7532, frozen: 0.9935, adapted: 0.9935}, mrr: {random_floor: 0.4083, lexical: 0.457, frozen: 0.8429, adapted: 0.8634}, median_rank: {lexical: 4, frozen: 1, adapted: 1}, delta_vs_frozen: {recall@1: 0.0325, recall@3: 0.013, recall@5: 0, mrr: 0.0206}}; reload parity {scores_identical: True, mrr_in_memory: 0.9125, mrr_reloaded: 0.9125}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-qwen3-reranker/v3/evidence/` in the workspace |
| 2026-09-19 | `d79994f` / `75fbc7d7` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 13 entries because the snapshot was pre-staged → `verify_snapshot` 13 files → `from_pretrained` on CPU with the `yes`/`no` ids checked → `fetch_corpus` served from the pre-staged cache after its digest checks → 10,003 + 3,080 rows read, 231 / 77 / 154 drawn over 77 intents with six candidates per query, `check_split_disjoint` clean and digests `6a5c384c…` / `32cc0b97…` / `a67a92bd…` → four dataset refusals → input manifest with the empty-document refusal → `rerank` on the first test shortlist (gold `extra charge on statement` scored 0.9697 and ranked first; 0.85 s, 95–98 tokens per pair) with all five sanity checks `True` → three shortlists ordered → random floor → lexical baseline → frozen evaluation → `adapt` → validation + test evaluation → orderings after adaptation → adapter export → reload parity) | 925.2 s | **PASSED** — 11/11 code cells; random floor recall@1 16.7 % / MRR 0.408; lexical baseline recall@1 26.6 %, recall@3 41.6 % (below the random floor: ties count against the positive and the hard negatives share its tokens), MRR 0.457, median rank 4; frozen test recall@1 73.4 %, recall@3 94.2 %, recall@5 99.4 %, MRR 0.843, median rank 1 (924 pairs, 131.7 s); `adapt` 31,461,888 of 595,776,512 params, 231 shortlists (924 training pairs), 2 epochs, 537.0 s incl. three 462-pair validation passes, validation MRR 0.881 → 0.902 → 0.923 (recall@1 79.2 → 83.1 → 85.7 %; `best_epoch` 2, train loss 0.799 → 0.669); **adapted test recall@1 77.9 %, recall@3 96.1 %, recall@5 99.4 %, MRR 0.873 (Δ +4.5 / +1.9 / 0.0 points, +0.030 MRR)** (129.2 s); the three probe shortlists kept the gold first before and after while the order below it changed and the gold scores moved (0.9697 → 0.9365, 0.2088 → 0.4085, 0.1343 → 0.0548); per-batch report `not-measurable`; adapter 125,850,032 B / 22 tensors, SHA-256 `9026a0ab…`; reload parity exact (probe scores identical, 20-shortlist validation MRR 0.975 both ways); six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-14 | `0c652a2` / `a21e5a860c0f` (`TASK-INFERENCE`, superseded) | Kaggle T4 (`kurtvalcorza/dimer-nb2-qwen3-reranker` v2) | Default sample path of the inference-only notebook: one synthetic query with three candidates, `stage_missing_files` fetching `model.safetensors` from the Hub, `verify_snapshot` over 13 files, `rerank` with its sanity checks, `not-measurable` report, CSV + JSON exports | 236.4 s | **PASSED** — 8/8 code cells (1 restart after the install cell), 1,207 MB staged; history only |

## Pre-registered fresh-test evaluation (2026-10-17)

The row 73 tutorial review found that the default learning rate had been kept over 5e-5 on the 154 test shortlists
(77.9 % against 76.0 % recall@1 in the `d79994f` CPU build record, validation MRR tied at 0.923). Those shortlists are
now the **exposed development test**; `docs/qwen3-reranker-preregistration.md` (committed alone, before any run)
fixes the grid, the fresh test, the tie policy and the scoring rules.

- **Evaluation of record (REL13):** the hosted Colab T4 run of 2026-10-18, commit `ff6da3a`, blob `2d6e5607`, cuda
  float32 — section *Hosted T4 evaluation of record* below. It is the first hosted T4 run of the regenerated notebook with
  the pre-registered settings and the first scoring of the fresh test by any model; later identical hosted runs are
  reproductions. `protocol.RECORDED_EVALUATION` is still `None` at this blob: it is set by the requote that followed
  this run (`1d3beb0`, notebook regenerated at `ae03f56`), and a reproduction is defined by identical pre-registered settings and module code except
  `RECORDED_EVALUATION` (amendment 1, section 10). Earlier hosted numbers came from bfloat16 training, stay labelled
  bfloat16 and are not comparable to it.
- **Amendment 2 (2026-10-18, `docs/qwen3-reranker-preregistration-amendment-2.md`, committed alone in `3d84c35`):** the
  whole model loads, runs and trains in float32 on CUDA as on CPU (AdamW on bfloat16 parameters had no float32 master
  copy). T4 estimate: about 4 min per two-epoch `adapt`, 25–35 min for the whole path, about 7–9 GB peak of 16 GB;
  measured on the record: 144.7–150.5 s per `adapt` and 995.4 s for the session (994.7 s on the reproduction); peak memory
  is not printed, so it stays unmeasured. Grid, margins, fresh test, bootstrap and score-once rules unchanged.
- **Aborted selection-only hosted look (2026-10-18):** a hosted T4 run of revision `b669f41` (bfloat16 training) was
  stopped by the orchestrator in Section 7, during the grid and before Section 8. It printed Section 6's
  exposed-development numbers and the T4 grid run for lr 1e-5 (kept-epoch validation MRR 0.8872; 0.8699 / 0.8786 /
  0.8872 by epoch). No fresh-test number exists. It is a validation look under the superseded precision and chooses
  nothing.
- **Amendment 1 (2026-10-18, `docs/qwen3-reranker-preregistration-amendment-1.md`, committed alone in `3eb2c57`):**
  paired bootstrap CIs, the exposed test no longer stops Section 8, device/dtype recorded, the re-train check, the
  fingerprint from the adapter, the margin's meaning, the T73-9 fallback disclosure and timings.
- **Reproductions:** one. The hosted Colab T4 run of 2026-10-18, commit `ae03f56`, blob `91402f81`, cuda float32
  (section *Hosted T4 reproduction* below): identical pre-registered settings; the module code differs from `ff6da3a`
  only in `RECORDED_EVALUATION` and its comment, and the notebook only in requoted markdown and the Section 8 line that
  labels an identical scoring a reproduction once `RECORDED_EVALUATION` is set. Every printed metric equals the record.
- **Development looks at the fresh test:** one, of a parameter-free baseline only. On 2026-10-18 the peer reviewer's data
  script (against `967977b`) ran the lexical baseline on the 385 fresh shortlists and read only
  `ties_with_positive` = 175 / 385; no model ran, no recall@k or MRR was printed or read, and no setting changed.
  No model has scored the fresh test. CPU checks run with `DIMER_MASK_FRESH_TEST=1` (SCORE_ONCE_RULE item 8).

### CPU development check, fresh test masked (2026-10-18; build record, not release evidence)

Commit `488c954` (code of this PR before the docs and regeneration commits), the package API driven in the notebook's
default order on the build workstation (Windows, CPython 3.12, `torch 2.14.0` CPU wheel, float32,
`CUDA_VISIBLE_DEVICES=-1`), pre-staged snapshot and corpus cache, 3,855 s. The fresh test was drawn and its pins and
disjointness checked, and **not scored**: `test_look` = `not scored (fresh test masked for a development check)`,
`settings_differing` = `[]`. Nothing in it changed a setting.

| lr | val MRR by epoch (0 = frozen) | kept epoch | kept val MRR | seconds |
|---|---|---|---|---|
| 1e-5 | 0.8807 / 0.9002 / 0.9165 | 2 | 0.9165 | 1043 |
| **2e-5** | 0.8807 / 0.9019 / 0.9232 | 2 | **0.9232** | 712 |
| 5e-5 | 0.8807 / 0.9229 / 0.9080 | 1 | 0.9229 | 599 |

Selection: **2e-5** ("no challenger beat 2e-05 by 0.01"); adaptation kept (0.9232 against frozen 0.8807, margin
0.010). The final 2e-5 run reproduced the grid run (validation MRR 0.9232, recall@1 0.8571).

Exposed development test (154 shortlists; old -> new for every number: the old column is the 2026-10-09 Colab T4
bfloat16 run of blob `93f1a9d5`, the new one this CPU float32 check):

| system | recall@1 | recall@3 | recall@5 | MRR | tie-expected recall@1 / MRR | ties_with_positive |
|---|---|---|---|---|---|---|
| random floor | 0.1667 -> 0.1667 | 0.5 -> 0.5 | 0.8333 -> 0.8333 | 0.4083 -> 0.4083 | (an expectation) | — |
| lexical | 0.2662 -> 0.2662 | 0.4156 -> 0.4156 | 0.7532 -> 0.7532 | 0.457 -> 0.457 | — -> 0.3335 / 0.5301 | — -> 86 |
| frozen | 0.7338 -> 0.7338 | — -> 0.9416 | — -> 0.9935 | 0.8429 -> 0.8429 | — -> 0.7338 / 0.8429 | — -> 0 |
| adapted (2e-5) | 0.7662 -> 0.7792 | — -> 0.9610 | 0.9935 -> 0.9935 | 0.8634 -> 0.8725 | — -> 0.7792 / 0.8725 | — -> 0 |

`evaluate_base` check: the base scored with `adapted: False` and the adapter kept afterwards (20-query validation MRR
0.9125 base, 0.975 adapted). The fresh-test numbers belong to the first hosted pre-registered run.

### 2026-10-18 hosted T4 evaluation of record (blob `2d6e5607`, PR #10 head `ff6da3a`)

**REL13: pre-registered evaluation of record.** This is the first hosted run of the pre-registered settings under
amendment 2 (float32 training) and the first scoring of the fresh test by any model (the CPU check above masked it; the
only earlier look was the parameter-free lexical tie count). Later hosted runs with identical settings are
reproductions; a run with any other setting is exploratory and never scores the fresh test.

- **Files.** [`docs/execution-evidence/2026-10-18-ff6da3a/`](execution-evidence/2026-10-18-ff6da3a/): executed notebook
  (SHA-256 `32d7ced5a4bda2ce215e75d6cd7f1e7c7ba14792d1326187cd5e4383ca18a2bc`) and `run_summary.json`
  (`276969abdb2c5619c9bdcf3b0daa11a6787e8881dc78f1e359739e4b07d3d901`), byte for byte (`-text`); `exec.log` as captured
  `e3eeb766fdccea39e8f772fb5580cc26f90f000f066d590b415be852297f897e`, committed with one string class replaced (the local
  home-directory prefix, replaced by `<local-run-dir>/` on lines 1 and 76, as in `9ed7d8a`) and nothing else changed:
  `c687030600346ae970fdb98d43fd25ab0b2ee7b187b8bc1b0f10b399410c3692`.
- **Source match.** The fetched notebook's `git hash-object` is `2d6e5607e1354f209ff862a34d798810c6093d54`, the blob at
  `ff6da3a`, and it is byte-equal to the committed file. All 36 executed cells have the same ids and order, with no source
  differences and no `# @param` edits. `NOTEBOOK_SOURCE.repository_revision` `003e228`, `module_sha256` `413253e4…`,
  generator `build_notebook.py/2.2`, spec 2.5.
- **Executor.** Colab CLI 0.7.4 (`colab exec -f`) on a fresh Colab VM, Tesla T4 (session `suite-qwen3-ff6da3a-745f`,
  started 2026-10-18 00:21:48 UTC). This is sequential CLI execution, not a browser Run all: forms and uploads were not
  exercised, and the file has no execution counts (order from `exec.log`, cells 1..13). Kernel Python 3.13.15; uv
  isolated environment CPython 3.12.12, 47 locked packages, built in 52 s; `torch 2.14.0+cu130`, `transformers 4.57.6`;
  `cuda:0`, **float32** (printed by Section 8 and in the fresh-test block).
- **Execution.** 13/13 code cells (the three carried-module cells print nothing), 0 error outputs, no restart, session wall
  995.4 s. 13 snapshot files (1,207,486,774 B) fetched from the Hub and digest-verified. Grid 144.7 + 150.5 + 150.2 s;
  Section 7 total 595.6 s (selected re-train about 150 s); exposed frozen evaluation 31.9 s.
- **Pre-registration.** `settings_differing` [], `test_look` "pre-registered evaluation (this runtime)", evaluated system
  `adapted` (lr 2e-5), re-train check 0.92316 against the grid run's 0.92316 (difference 0.0, tolerance 0.010).

**Validation grid (77 queries).**

| lr | Val MRR by epoch (0 = frozen) | Kept epoch | Kept val MRR | Seconds |
|---|---|---|---|---|
| 1e-5 | 0.8807 / 0.9002 / 0.9165 | 2 | 0.9165 | 144.7 |
| **2e-5** | 0.8807 / 0.9019 / 0.9232 | 2 | **0.9232** | 150.5 |
| 5e-5 | 0.8807 / 0.9229 / 0.9080 | 1 | 0.9229 | 150.2 |

Selection **2e-5** ("no challenger beat 2e-05 by 0.01": 1e-5 −0.0067, 5e-5 −0.0003); adaptation kept (+0.0425 over
frozen 0.8807, margin 0.010). Selected re-train: validation recall@1 0.7922 → 0.8312 → 0.8571, recall@3 0.961 → 0.974 →
0.987, MRR 0.8807 → 0.9019 → 0.9232, train loss 0.7989 → 0.6694, best epoch 2.

**Fresh test (385 shortlists, digest `9bc26a46…`), the evaluation of record.** Pessimistic ranks (ties count against the
positive); the tie-expected values are equal for both models because no model score tied the positive (`ties_with_positive`
0 and 0).

| System | Recall@1 | Recall@3 | Recall@5 | MRR | 95 % CI, MRR | 95 % CI, recall@1 |
|---|---|---|---|---|---|---|
| Random floor | 0.1667 | 0.5 | 0.8333 | 0.4083 | (an expectation) | — |
| Lexical | 0.3299 | 0.4597 | 0.8104 | 0.5069 | 0.4711 to 0.5428 | 0.2831 to 0.3766 |
| Lexical, tie-expected (175 ties) | 0.381 | 0.5888 | 0.9335 | 0.5644 | — | — |
| Frozen | 0.7455 | 0.9506 | 0.9974 | 0.851 | 0.8240 to 0.8765 | 0.7013 to 0.7870 |
| Adapted (2e-5) | 0.7922 | 0.974 | 0.9948 | 0.879 | 0.8544 to 0.9019 | 0.7506 to 0.8312 |

Adapted − frozen (paired bootstrap over the 385 shortlists, 10,000 resamples, seed 20261018, percentile intervals):
MRR **+0.0279, 95 % CI 0.0086 to 0.0476**; recall@1 **+0.0468 (18 / 385), 95 % CI 0.0130 to 0.0805**. Both intervals
exclude 0; verdict *improved* for both. Recall@5 falls by one shortlist (384 → 383). The intervals cover only the choice
of shortlists, not split-to-split or seed-to-seed variation.

Re-derived from the printed values (the per-query ranks stayed on the released VM, so the bootstrap was not re-run):
every recall × 385 is a whole count (frozen 287 / 366 / 384, adapted 305 / 375 / 383, lexical 127 / 177 / 312); each MRR
lies inside the bounds its recall@1/3/5 allow for six candidates; the floor is H₆ / 6 = 0.4083; every bootstrap endpoint
is a whole multiple of 1 / 385 (recall@1) or 1 / 23,100 (MRR), as resamples of 385 shortlists must be; the marginal
recall@1 intervals agree with the normal approximation to within 0.002; and the paired recall@1 interval implies about 31
shortlists improved and 13 worse at rank 1. The lexical tie count (175 / 385) equals the disclosed development look.

**Exposed development test (154; not independent — an earlier learning-rate choice was made on it).** Random floor
0.1667 / 0.4083, lexical 0.2662 / 0.457 (tie-expected 0.3335 / 0.5301, 86 ties), frozen 0.7338 / 0.8429, adapted 0.7792 /
0.8725 (recall@1 / MRR; recall@3 0.9416 → 0.961, recall@5 0.9935 both; Δ recall@1 +0.0455, Δ MRR +0.0297; model ties 0).
This split prints no interval.

**Against the CPU development check (float32, fresh test masked, `488c954`).** Every number both runs print is equal at
printed precision:

| Measure | CPU development check | Hosted T4 record | Difference |
|---|---|---|---|
| Grid val MRR by epoch, 1e-5 / 2e-5 / 5e-5 | 0.8807/0.9002/0.9165; 0.8807/0.9019/0.9232; 0.8807/0.9229/0.9080 | same | 0 |
| Kept epochs / selection / adaptation kept | 2, 2, 1 / 2e-5 / yes | same | none |
| Selected re-train val MRR / recall@1 | 0.9232 / 0.8571 | 0.9232 / 0.8571 | 0 |
| Exposed lexical, frozen, adapted (all metrics, both tie policies, tie counts) | table above | same | 0 |
| 20-query validation MRR, adapted (CPU `evaluate_base` check; T4 reload parity) | 0.975 | 0.975 | 0 |
| Grid seconds 1e-5 / 2e-5 / 5e-5 | 1,043 / 712 / 599 | 144.7 / 150.5 / 150.2 | device |
| Fresh test | masked | scored (above) | — |

The grid selection matches. The CPU check scored no model on the fresh test, so there is nothing there to compare.

**Against the bfloat16 hosted runs.** The 2026-10-09 run of blob `93f1a9d5` (bfloat16 training): frozen validation MRR
0.8699 → now 0.8807; exposed adapted 0.7662 / 0.8634 → 0.7792 / 0.8725; exposed frozen 0.7338 / 0.8429 equal; reload
parity 0.9125 → 0.975; adapter 62,926,256 → 125,850,032 B (float32: 31,461,888 parameters × 4 B plus a 2,480 B header).
The aborted selection-only look at `b669f41` (bfloat16, disclosed above, no fresh-test number) gave lr 1e-5 0.8699 /
0.8786 / 0.8872; float32 gives 0.8807 / 0.9002 / 0.9165. Float32 on CUDA removed the CPU/T4 gap.

**Measured against the amendment 2 estimate.**

| Quantity | Estimate (fixer, amendment 2) | Measured | Reading |
|---|---|---|---|
| One two-epoch `adapt` (incl. validation passes) | about 4 min | 144.7–150.5 s (2.4–2.5 min) | faster |
| Whole path | 25–35 min | 995.4 s session wall (16.6 min, incl. the 52 s environment build and the 1.2 GB download) | faster |
| Peak GPU memory | about 7–9 GB of 16 | not measured (the notebook prints no memory figure) | open |

**Evidence boundary.** Default path only, by CLI sequential execution. BYOD (REL12), the activity and the optional
experiments were not run. The `outputs/` files stayed on the released VM, so their bytes (per-query ranks, the evaluation
report) were not inspected.

| Journey | Verdict |
|---|---|
| Clean default path (isolated env, no restart; CLI sequential) | PASS |
| Activity / optional experiments | not assessed in this run |
| BYOD | not assessed in this run |

Status stays **Candidate**.

### 2026-10-18 hosted T4 reproduction (blob `91402f81`, PR #10 head `ae03f56`)

**REL13 reproduction of the pre-registered evaluation.** A later hosted run with the pre-registered settings of the `ff6da3a` evaluation of record (rule item 3).
Nothing was changed in response to the fresh-test number: `ff6da3a..ae03f56` sets `protocol.RECORDED_EVALUATION` to the
record, requotes markdown to the record, and adds the Section 8 line that labels an identical scoring a reproduction when
`RECORDED_EVALUATION` is set. The record above stays THE result.

- **Files.** [`docs/execution-evidence/2026-10-18-ae03f56/`](execution-evidence/2026-10-18-ae03f56/): executed notebook (SHA-256 `501c3573926f32240143352dcc55c9cb83a0cb68e9886c6bf9e0b339417ab17b`) and
  `run_summary.json` (`04213176c96cc9421477b5ed0712117899c4e4a5c74be865442dec9250d559e3`), byte for byte (`-text`); `exec.log` as captured `d02c752a20bbc928b1a67c70f0b5e3ddeafc4af5bd1f9dd2b4cc791ad6adb32e`, committed with one string class
  replaced (the local home-directory prefix, replaced by `<local-run-dir>/` on lines 1 and 76, as in `9ed7d8a`) and nothing
  else changed: `29ca03f9548b0860f7ded9b9d1f19c4f819e30094cf62ca3b4322ca26c6e30eb`.
- **Source match.** The fetched notebook's `git hash-object` is `91402f813fb03439e05b06237c1602dadd89ba8d`, the blob at
  `ae03f56`, byte-equal to the committed file; 36/36 executed cells have the same ids and order, no source differences and
  no `# @param` edits. `NOTEBOOK_SOURCE.repository_revision` `482d078`, `module_sha256` `5f47d52e…`, generator
  `build_notebook.py/2.2`, spec 2.5.
- **Executor.** Colab CLI 0.7.4 (`colab exec -f`), fresh Colab VM, Tesla T4 (session `suite-qwen3-ae03f56-745f`, started
  2026-10-18 01:39:23 UTC), CLI sequential execution, not a browser Run all; kernel Python 3.13.15, uv isolated
  environment CPython 3.12.12 (47 locked packages, 56 s); `torch 2.14.0+cu130`, `transformers 4.57.6`; `cuda:0`, float32.
- **Execution.** 13/13 code cells, 0 error outputs, no restart, session wall 994.7 s; grid 133.7 + 140.7 + 142.9 s;
  Section 7 total 560.5 s; exposed frozen evaluation 30.3 s.
- **Pre-registration.** `settings_differing` [], `test_look` "reproduction (this runtime)", files suffixed
  `_reproduction_1`, evaluated system `adapted` (lr 2e-5), re-train check 0.92316 against 0.92316 (difference 0.0).

**Against the evaluation of record (`ff6da3a`).** All 61 printed output lines are equal except timings, the worker pid,
the embedded revision and module hash, `test_look`, the `_reproduction_1` file suffix and the printed
`evaluation_of_record` block (`none recorded yet` at `ff6da3a`, the record at `ae03f56`).

| Measure | Record (`ff6da3a`) | Reproduction (`ae03f56`) |
|---|---|---|
| Grid kept val MRR 1e-5 / 2e-5 / 5e-5 (kept epochs) | 0.9165 / 0.9232 / 0.9229 (2, 2, 1) | same |
| Selection / re-train val MRR | 2e-5 / 0.92316 | same |
| Fresh MRR frozen → adapted | 0.851 → 0.879 | same |
| Fresh Δ MRR, 95 % CI | +0.0279, 0.0086 to 0.0476 | same (full precision 0.027922, 0.008615 to 0.047576) |
| Fresh recall@1 frozen → adapted | 0.7455 → 0.7922 | same |
| Fresh Δ recall@1, 95 % CI | +0.0468, 0.0130 to 0.0805 | same (0.046753, 0.012987 to 0.080519) |
| Exposed adapted recall@1 / MRR (Δ) | 0.7792 / 0.8725 (+0.0455 / +0.0297) | same |
| Reload parity / adapter bytes / SHA-256 prefix | 0.975 / 125,850,032 / `fd5a7d95eb8c4eb47` | same |
| Session wall / env build / grid seconds | 995.4 / 52 / 144.7, 150.5, 150.2 | 994.7 / 56 / 133.7, 140.7, 142.9 |

**Evidence boundary.** Default path only, by CLI sequential execution; BYOD (REL12), the activity and the optional
experiments were not run; `outputs/` stayed on the released VM. Status stays **Candidate**.

## Current status

**Candidate.** The current blob `e8889991` (code commit `b28964b`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes the earlier run's results and `outputs/` files and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); its hosted Colab T4 runs (the default path and a scripted refusal, `USE_BYOD=True;BYOD_PATH='/content/sample_data/california_housing_test.csv'`) are pending, and the hosted runs of earlier blobs, including `91402f81`, are history for it (REL14). Before that change: 2026-10-18: blob `2d6e5607` (commit `ff6da3a`, amendment 2, float32 on every device) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 (Colab CLI sequential execution, 13/13 code cells, 995.4 s). It is the REL13 pre-registered evaluation of record: fresh test (385) MRR 0.851 → 0.879, paired Δ +0.0279 (95 % CI 0.0086 to 0.0476), recall@1 0.7455 → 0.7922 (Δ +0.0468, CI 0.0130 to 0.0805). Blob `91402f81` (commit `ae03f56`) reproduced it on a second fresh Colab T4 the same day (REL13 reproduction of the pre-registered evaluation; 13/13 cells, 0 errors, 994.7 s; every printed metric equal). BYOD on a hosted runtime (REL12) and the activity were not run, so status stays **Candidate** (REL14). History: the `E2E` notebook blob `99efdc6a` (committed at `995cd8e`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok, 588.2 s, 30 files, 1209 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout, but it needed **one manual restart after the install cell** (pass 1 raised the stale-import guard after the in-kernel pip install). A restart-assisted run does not meet the no-restart gate (RUN10/ENV6), so that record is history, not release evidence; the earlier "Release-grade" label is withdrawn.

### 2026-10-09 review fixes (RR-M1..M4, RR-m1..m6)

The Notebook Review Framework v1 review of `f13a58e` (`docs/reviews/2026-10-02-notebook-review/`) found four Major and six Minor findings. On top of the cherry-picked fleet-sweep commit (`docs/reviews/2026-10-05-fleet-sweep/`), the notebook was regenerated by `tools/build_notebook.py` (/2.2) from `tools/notebook_template.py`:

- **RR-M1:** Section 1 installs nothing into the kernel. It verifies a pinned `uv` 0.12.15 wheel, builds a managed CPython 3.12.12 environment from `tutorials/requirements-colab.lock.txt` (47 hash-locked packages, `--require-hashes --only-binary :all:`) and routes every later cell to a worker there; the `google.colab` stubs in that worker carry a `ModuleSpec` (`tests/test_worker_colab_stubs.py`). Linux x86_64 only.
- **RR-M2:** `adapt()` and `load_artifact()` start from the pinned base (`restore_base`); Sections 5 and 6 put the base back before they measure; Sections 8 and 9 refuse to score or export the base as adapted.
- **RR-M3:** `split_dataset` checks a BYOD file's size once and names it (12..10,002 distinct queries at the default fractions: 8 stay for training, one record per evaluated split, at most `MAX_EVAL_RECORDS` = 2,000 per evaluated split); validation and test are validated with a one-record floor; on BYOD the quality comparisons are verdicts, while the Banking77 sample at the default settings still stops on a failed comparison.
- **RR-M4:** guided layer (from the sweep) plus one Predict → Change → Run → Observe → Explain activity (`TRAINABLE_LAYERS = 1`) and troubleshooting for the new refusals.
- **RR-m1..m6:** device-labelled figures (CPU float32 build record vs Kaggle T4 bfloat16; adapter about 126 MB in float32, 63 MB in bfloat16); no `{{`/`}}` in markdown; the lexical baseline is described as weakened by construction and its below-random recall@3/@5 explained; `TRAIN_CANDIDATES_RANGE` (2..8) printed as the enforced ceiling and `MAX_TRAIN_CANDIDATES` as the default; NOTEBOOK_SPEC 2.2; `BYOD_PATH`.

Local checks (pytest, Ruff, the validator and the generator `--check`) are not clean-runtime evidence. On 2026-10-09 the review-fix blob `93f1a9d5` (commit `d064746`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-09 (Colab CLI sequential execution, 12/12 code cells, 513.7 s; held-out recall@1 0.7338 → 0.7662, MRR 0.8429 → 0.8634; reload parity exact) — row at the top of *Recorded executions*. That closes the no-restart gate (RUN10/ENV6) for the default path on Colab T4. BYOD on a hosted runtime (REL12) and the activity were not run, so status stays **Candidate** (REL14).
