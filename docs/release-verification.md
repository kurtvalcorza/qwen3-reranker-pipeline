# Release verification

`tutorials/qwen3_reranker_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 13-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
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
   `numpy==2.5.3` (an interpreter restart after the install is expected where the runtime's preinstalled torch or
   numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
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
| `qwen3_reranker_colab.ipynb` (`E2E`) | `d79994f` / `75fbc7d7` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `qwen3_reranker_colab.ipynb` (`TASK-INFERENCE`, superseded) | `0c652a2` / `a21e5a860c0f` | 2026-09-14 | Kaggle T4 (`kurtvalcorza/dimer-nb2-qwen3-reranker` v2) | PASSED — 8/8 code cells, 236.4 s (1 restart after the install cell), 1,207 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/qwen3_reranker_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/qwen3_reranker_colab.ipynb`). Wall times are the sum of per-cell times reported by
the executor and include the model download where it occurred; they are measurements for the stated runtime, not
general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-19 | `d79994f` / `75fbc7d7` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 13 entries because the snapshot was pre-staged → `verify_snapshot` 13 files → `from_pretrained` on CPU with the `yes`/`no` ids checked → `fetch_corpus` served from the pre-staged cache after its digest checks → 10,003 + 3,080 rows read, 231 / 77 / 154 drawn over 77 intents with six candidates per query, `check_split_disjoint` clean and digests `6a5c384c…` / `32cc0b97…` / `a67a92bd…` → four dataset refusals → input manifest with the empty-document refusal → `rerank` on the first test shortlist (gold `extra charge on statement` scored 0.9697 and ranked first; 0.85 s, 95–98 tokens per pair) with all five sanity checks `True` → three shortlists ordered → random floor → lexical baseline → frozen evaluation → `adapt` → validation + test evaluation → orderings after adaptation → adapter export → reload parity) | 925.2 s | **PASSED** — 11/11 code cells; random floor recall@1 16.7 % / MRR 0.408; lexical baseline recall@1 26.6 %, recall@3 41.6 % (below the random floor: ties count against the positive and the hard negatives share its tokens), MRR 0.457, median rank 4; frozen test recall@1 73.4 %, recall@3 94.2 %, recall@5 99.4 %, MRR 0.843, median rank 1 (924 pairs, 131.7 s); `adapt` 31,461,888 of 595,776,512 params, 231 shortlists (924 training pairs), 2 epochs, 537.0 s incl. three 462-pair validation passes, validation MRR 0.881 → 0.902 → 0.923 (recall@1 79.2 → 83.1 → 85.7 %; `best_epoch` 2, train loss 0.799 → 0.669); **adapted test recall@1 77.9 %, recall@3 96.1 %, recall@5 99.4 %, MRR 0.873 (Δ +4.5 / +1.9 / 0.0 points, +0.030 MRR)** (129.2 s); the three probe shortlists kept the gold first before and after while the order below it changed and the gold scores moved (0.9697 → 0.9365, 0.2088 → 0.4085, 0.1343 → 0.0548); per-batch report `not-measurable`; adapter 125,850,032 B / 22 tensors, SHA-256 `9026a0ab…`; reload parity exact (probe scores identical, 20-shortlist validation MRR 0.975 both ways); six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-14 | `0c652a2` / `a21e5a860c0f` (`TASK-INFERENCE`, superseded) | Kaggle T4 (`kurtvalcorza/dimer-nb2-qwen3-reranker` v2) | Default sample path of the inference-only notebook: one synthetic query with three candidates, `stage_missing_files` fetching `model.safetensors` from the Hub, `verify_snapshot` over 13 files, `rerank` with its sanity checks, `not-measurable` report, CSV + JSON exports | 236.4 s | **PASSED** — 8/8 code cells (1 restart after the install cell), 1,207 MB staged; history only |

## Current status

The `E2E` notebook source is complete and passes all static checks, including the generator parity checks
(`--check` OK). A local pre-flight execution of the committed blob completed the whole default path on CPU — corpus
read from the cache, validation and split, the reranking contract, the random floor, the lexical baseline and the
frozen ranking metrics, two epochs of listwise fine-tuning, held-out evaluation, before/after orderings, adapter export
and reload parity — which catches defects but is **not** a supported runtime under REL1/REL10, and it ran with the
snapshot and the two Banking77 files pre-staged, so neither the 1.19 GB Hub fetch nor the corpus download has been
exercised by this notebook end to end; the earlier `TASK-INFERENCE` Kaggle run did exercise the Hub fetch and digest
check of the same snapshot. The repository stays at **Candidate** until a Colab or fresh-container run of the exact
`E2E` release revision is recorded above.
