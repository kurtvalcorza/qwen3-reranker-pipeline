# Release verification

`tutorials/qwen3_reranker_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate
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
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; bfloat16 there, float32 on CPU) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy or torch that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/qwen3-reranker-0.6b/` or the corpus cache `weights/banking77/` (the standalone path writes the
   manifest itself, stages the missing file from the Hub, and fetches the two pinned Banking77 files from the project
   repository, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `INSTRUCTION = 'Given a customer support message, judge whether the document
   names the banking intent it expresses'`, `EPOCHS = 2`, `LEARNING_RATE = 2e-5`, `BATCH_SIZE = 4`,
   `TRAINABLE_LAYERS = 2`, `TRAIN_CANDIDATES = 4`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`,
   `numpy==2.5.3`, as built from `tutorials/requirements-colab.lock.txt` in the isolated environment (nothing is
   installed into the kernel, and **no restart is allowed**: a run that needs one does not meet RUN10/ENV6);
5. verify every default-path stage completes:
   - isolated runtime built from the hash lock (`--require-hashes --only-binary :all:`) with no GitHub access;
   - the three carried module cells execute (defining `Qwen3RerankerPipeline`, `verify_snapshot`,
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
     dataset digests `6a5c384c…` / `32cc0b97…` / `a67a92bd…`; `outputs/…_train.csv` written; the four dataset
     refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MAX_PAIRS` 32, `MAX_TEXT_TOKENS` 8192, `MAX_TEXT_CHARS` 100000, `MAX_TRAIN_TOKENS`
     192, `MAX_TRAIN_CANDIDATES` 4) and the `yes`/`no` ids surfaced; `validate_inputs` writing
     `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the empty-document
     probe); `rerank` on the first test shortlist with all five sanity checks `True` and the ranked shortlist printed
     with the gold marked, then three test shortlists ordered by the frozen model;
   - Section 6: the random floor (recall@1 16.7 %, MRR ≈ 0.408), the lexical baseline (≈ 26.6 % recall@1 on the
     sample — the hard negatives were chosen by overlap) and the frozen reranker's test metrics (≈ 73.4 %
     recall@1, MRR ≈ 0.843, 924 pairs) on CPU float32, with the cell's assertion that the query counts agree
     and the frozen MRR is above the floor;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 31,461,888 trainable of 595,776,512 parameters,
     924 training pairs, and a two-epoch history with validation MRR rising (0.881 → 0.902 → 0.923 in the recorded run;
     `best_epoch` 2);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison and
     `outputs/…_evaluation_report.json` written (the cell asserts the adapted test MRR exceeds the frozen one — on the
     sample recall@1 ≈ 77.9 % versus ≈ 73.4 %);
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
| 2026-10-09 (02:49:17 UTC start) | commit `d064746` / blob `93f1a9d552131fc1382151ce1a7e99261ea0cfb6` (`NOTEBOOK_SOURCE.repository_revision` `c628da8`, the source revision the notebook was generated from; `c628da8..d064746` is the review-fix commit; generator `build_notebook.py/2.2`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab VM (session `suite-qwen3-d064746-745f`), Tesla T4; committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout; kernel Python 3.13.15; uv isolated environment CPython 3.12.12 (47 locked packages, built in 75 s), `torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda:0`, bfloat16 | Default sample path only (`USE_BYOD = False`, default form values); 13 snapshot files fetched from the Hub and digest-verified | 513.7 s | **PASSED — one pass, no restart, 0 errors**: 12/12 code cells (the three carried-module cells print nothing); splits 231 / 77 / 154, four refusal probes rejected; ceilings print `TRAIN_CANDIDATES_RANGE` (2, 8) and `MAX_EVAL_RECORDS` 2000 with `TRAIN_CANDIDATES` 4 as the default; random floor recall@1 0.1667 / MRR 0.4083; lexical 0.2662 / 0.457 (recall@3 0.4156 below the 0.5 floor, recall@5 0.7532 below 0.8333); frozen 0.7338 / 0.8429 (`adapted: False`, 46.7 s); validation MRR 0.8699 → 0.8872 → 0.8937, best epoch 2 (`started_from` pinned base, 229.0 s); adapted 0.7662 / 0.8634 (Δ recall@1 +0.0325, MRR +0.0206, recall@5 0.9935 unchanged, verdict *improved*); 2 of 3 probe shortlists reordered, gold rank 1 in all; adapter 22 tensors, 62,926,256 B; reload parity exact (scores identical, MRR 0.9125 both ways); six exports. These match the worked answers. Evidence [`docs/execution-evidence/2026-10-09-d064746/`](execution-evidence/2026-10-09-d064746/): executed notebook SHA-256 `1b087bd151ea134ca42b2c65b40a9c891c11a7bfa4c18b4f60ca153a12137562`, `run_summary.json` `eace86e87a65159011b8651fcbcd44de0640cc65a8007f124ae95a5b3761d9ac`, `exec.log` `a0f70489b797baf860ce599b029a9b117fa56c708c133e7cf153f5b0e191e734`. Not exercised: BYOD (REL12), the activity and the optional experiments, a browser Run all |
| 2026-09-19 | `995cd8e` / `99efdc6a` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-qwen3-reranker` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 588.2 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 30 files, 1209 MB staged from the Hub into a clean cache; comparison {recall@1: {random_floor: 0.1667, lexical: 0.2662, frozen: 0.7338, adapted: 0.7662}, recall@3: {random_floor: 0.5, lexical: 0.4156, frozen: 0.9416, adapted: 0.9545}, recall@5: {random_floor: 0.8333, lexical: 0.7532, frozen: 0.9935, adapted: 0.9935}, mrr: {random_floor: 0.4083, lexical: 0.457, frozen: 0.8429, adapted: 0.8634}, median_rank: {lexical: 4, frozen: 1, adapted: 1}, delta_vs_frozen: {recall@1: 0.0325, recall@3: 0.013, recall@5: 0, mrr: 0.0206}}; reload parity {scores_identical: True, mrr_in_memory: 0.9125, mrr_reloaded: 0.9125}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-qwen3-reranker/v3/evidence/` in the workspace |
| 2026-09-19 | `d79994f` / `75fbc7d7` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 13 entries because the snapshot was pre-staged → `verify_snapshot` 13 files → `from_pretrained` on CPU with the `yes`/`no` ids checked → `fetch_corpus` served from the pre-staged cache after its digest checks → 10,003 + 3,080 rows read, 231 / 77 / 154 drawn over 77 intents with six candidates per query, `check_split_disjoint` clean and digests `6a5c384c…` / `32cc0b97…` / `a67a92bd…` → four dataset refusals → input manifest with the empty-document refusal → `rerank` on the first test shortlist (gold `extra charge on statement` scored 0.9697 and ranked first; 0.85 s, 95–98 tokens per pair) with all five sanity checks `True` → three shortlists ordered → random floor → lexical baseline → frozen evaluation → `adapt` → validation + test evaluation → orderings after adaptation → adapter export → reload parity) | 925.2 s | **PASSED** — 11/11 code cells; random floor recall@1 16.7 % / MRR 0.408; lexical baseline recall@1 26.6 %, recall@3 41.6 % (below the random floor: ties count against the positive and the hard negatives share its tokens), MRR 0.457, median rank 4; frozen test recall@1 73.4 %, recall@3 94.2 %, recall@5 99.4 %, MRR 0.843, median rank 1 (924 pairs, 131.7 s); `adapt` 31,461,888 of 595,776,512 params, 231 shortlists (924 training pairs), 2 epochs, 537.0 s incl. three 462-pair validation passes, validation MRR 0.881 → 0.902 → 0.923 (recall@1 79.2 → 83.1 → 85.7 %; `best_epoch` 2, train loss 0.799 → 0.669); **adapted test recall@1 77.9 %, recall@3 96.1 %, recall@5 99.4 %, MRR 0.873 (Δ +4.5 / +1.9 / 0.0 points, +0.030 MRR)** (129.2 s); the three probe shortlists kept the gold first before and after while the order below it changed and the gold scores moved (0.9697 → 0.9365, 0.2088 → 0.4085, 0.1343 → 0.0548); per-batch report `not-measurable`; adapter 125,850,032 B / 22 tensors, SHA-256 `9026a0ab…`; reload parity exact (probe scores identical, 20-shortlist validation MRR 0.975 both ways); six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-14 | `0c652a2` / `a21e5a860c0f` (`TASK-INFERENCE`, superseded) | Kaggle T4 (`kurtvalcorza/dimer-nb2-qwen3-reranker` v2) | Default sample path of the inference-only notebook: one synthetic query with three candidates, `stage_missing_files` fetching `model.safetensors` from the Hub, `verify_snapshot` over 13 files, `rerank` with its sanity checks, `not-measurable` report, CSV + JSON exports | 236.4 s | **PASSED** — 8/8 code cells (1 restart after the install cell), 1,207 MB staged; history only |

## Current status

**Candidate.** The `E2E` notebook blob `99efdc6a` (committed at `995cd8e`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok, 588.2 s, 30 files, 1209 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout, but it needed **one manual restart after the install cell** (pass 1 raised the stale-import guard after the in-kernel pip install). A restart-assisted run does not meet the no-restart gate (RUN10/ENV6), so that record is history, not release evidence; the earlier "Release-grade" label is withdrawn.

### 2026-10-09 review fixes (RR-M1..M4, RR-m1..m6)

The Notebook Review Framework v1 review of `f13a58e` (`docs/reviews/2026-10-02-notebook-review/`) found four Major and six Minor findings. On top of the cherry-picked fleet-sweep commit (`docs/reviews/2026-10-05-fleet-sweep/`), the notebook was regenerated by `tools/build_notebook.py` (/2.2) from `tools/notebook_template.py`:

- **RR-M1:** Section 1 installs nothing into the kernel. It verifies a pinned `uv` 0.12.15 wheel, builds a managed CPython 3.12.12 environment from `tutorials/requirements-colab.lock.txt` (47 hash-locked packages, `--require-hashes --only-binary :all:`) and routes every later cell to a worker there; the `google.colab` stubs in that worker carry a `ModuleSpec` (`tests/test_worker_colab_stubs.py`). Linux x86_64 only.
- **RR-M2:** `adapt()` and `load_artifact()` start from the pinned base (`restore_base`); Sections 5 and 6 put the base back before they measure; Sections 8 and 9 refuse to score or export the base as adapted.
- **RR-M3:** `split_dataset` checks a BYOD file's size once and names it (12..10,002 distinct queries at the default fractions: 8 stay for training, one record per evaluated split, at most `MAX_EVAL_RECORDS` = 2,000 per evaluated split); validation and test are validated with a one-record floor; on BYOD the quality comparisons are verdicts, while the Banking77 sample at the default settings still stops on a failed comparison.
- **RR-M4:** guided layer (from the sweep) plus one Predict → Change → Run → Observe → Explain activity (`TRAINABLE_LAYERS = 1`) and troubleshooting for the new refusals.
- **RR-m1..m6:** device-labelled figures (CPU float32 build record vs Kaggle T4 bfloat16; adapter about 126 MB in float32, 63 MB in bfloat16); no `{{`/`}}` in markdown; the lexical baseline is described as weakened by construction and its below-random recall@3/@5 explained; `TRAIN_CANDIDATES_RANGE` (2..8) printed as the enforced ceiling and `MAX_TRAIN_CANDIDATES` as the default; NOTEBOOK_SPEC 2.2; `BYOD_PATH`.

Local checks (pytest, Ruff, the validator and the generator `--check`) are not clean-runtime evidence. On 2026-10-09 the review-fix blob `93f1a9d5` (commit `d064746`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-09 (Colab CLI sequential execution, 12/12 code cells, 513.7 s; held-out recall@1 0.7338 → 0.7662, MRR 0.8429 → 0.8634; reload parity exact) — row at the top of *Recorded executions*. That closes the no-restart gate (RUN10/ENV6) for the default path on Colab T4. BYOD on a hosted runtime (REL12) and the activity were not run, so status stays **Candidate** (REL14).
