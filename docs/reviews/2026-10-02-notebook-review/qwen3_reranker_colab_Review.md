# Qwen3-Reranker-0.6B E2E Reranker Fine-tuning Notebook — Review

Notebook Review Framework v1 review of `tutorials/qwen3_reranker_colab.ipynb`. Review only: this document changes
nothing in the notebook, the generator or the package.

## Executive assessment

**Readiness: Needs revision.** The technical core is strong. The snapshot is digest-pinned and verified. The corpus
is real and pinned. The split is balanced and query-disjoint, and the probe found no normalised or near-duplicate
leakage. Recall@k and MRR are reported beside a random floor and a lexical baseline, a bounded listwise adaptation
runs, and the adapter reloads with exact parity. The documented Kaggle T4 run of this exact blob passed. Four Major
findings block the intended self-paced use:

1. **RR-M1:** a fresh hosted runtime stops after the install cell and must be restarted by hand. The recorded
   release run needed that restart, which RUN10/ENV6 forbid.
2. **RR-M2:** `pipe` is adapted in place and never reset. Every documented rerun (the four optional experiments and
   the BYOD "re-run from that cell" instruction) starts from the already-adapted model, labels it "frozen", and
   stacks a second fine-tuning on it.
3. **RR-M3:** the BYOD size contract is wrong at both ends. The notebook states 8..20,000 records; cell 13 needs at
   least 50, and more than 10,000 fails in Section 6. The rejection messages describe a split, not the user's file,
   and two unconditional asserts stop a BYOD run before export when adaptation does not help.
4. **RR-M4:** the notebook declares `GUIDED` but has no guided layer (no predictions, checkpoints, worked answers,
   roadmap, glossary or troubleshooting).

## 1. Review contract and evidence

| Item | Scope |
|---|---|
| Repository | `kurtvalcorza/qwen3-reranker-pipeline` |
| Notebook | `tutorials/qwen3_reranker_colab.ipynb`, Git blob `99efdc6a3ca4a058be7a18c6cc7a88947a7531a9` |
| Reviewed revision | `origin/main` = `f13a58e65a7ee54343e8fa262c166308699c4f11` (confirmed with the GitHub API). The notebook was generated from `d778a7b`; the blob is unchanged since `995cd8e`. |
| Declared profile / mode / spec | `E2E` / `GUIDED` / NOTEBOOK_SPEC `2.0`, standalone carrier (`metadata.dimer`) |
| Specification baseline | NOTEBOOK_SPEC **2.2** (2026-09-26) at `ml-worker` `origin/main`; requirements cited below are from 2.2. The guided layer (§3.5, GDL1–GDL15) is new in 2.1/2.2 and is a `SHOULD`. |
| Intended learner | Stated: basic Python; knows what a cross-encoder, recall@k, MRR and a listwise softmax loss are (Prerequisites). |
| Supported runtime | Google Colab or Jupyter, Python 3.12. CPU float32 is the stated default; CUDA bfloat16 is used automatically. |
| Promised outcomes | Pinned install; verified snapshot; pinned Banking77 corpus validated and split without leakage (231/77/154); inference contract with an input manifest and refusal probe; random floor, lexical baseline and frozen recall@1/3/5 + MRR; bounded listwise fine-tuning of the last two decoder layers with validation-MRR selection; held-out four-way comparison; before/after orderings; safetensors adapter export with verified reload parity; BYOD through the same stages. |
| Scope examined | All 25 cells (markdown, the 11 code cells, form fields, the optional-experiment text, the exports), `tutorials/README.md`, `docs/release-verification.md`, `STATUS.md`, `tools/build_notebook.py`, `tools/notebook_template.py`, and the three carried modules. |

### Evidence actually obtained

| Journey | Evidence basis | What it shows |
|---|---|---|
| First-time learner | Source inspection | Strong prose on the contract, score semantics and limits. No guided layer. Literal `{{…}}` template escapes leak into the text. |
| Clean default | Documented execution evidence | Kaggle Tesla T4, blob `99efdc6a` at `995cd8e`, 2026-09-19: **PASSED, 11/11 cells, but `passes[0]` failed on the stale-import guard and needed one restart** (`run_summary.json` in `.agent/backups/kaggle-e2e-2026-09-19/…/v3/evidence/`). The executed outputs match the recorded comparison. There is **no Colab run** and no run on the declared CPU default runtime. The local CPU pre-flight row is for an older blob (`75fbc7d7`). |
| Active learning | Direct execution (local, CPU float32, the pinned snapshot digest-verified, stand-in sizes: 8 training / 8 validation records, 1 epoch, lr 1e-4) | Rerunning Section 6 after Section 7 reports `adapted: True` under the label `frozen_model_test`. A second `adapt` labels the adapted model's metrics "frozen model" at epoch 0. Probe scores are never restored. **Fails.** |
| Reuse and recovery | Direct execution of the carried data contract (no model), plus documented evidence for reload | BYOD size boundary: first passing size is 50; above 10,000 fails in Section 6. The four dataset refusals and the empty-document refusal are documented in the Kaggle outputs. Reload parity is exact in the Kaggle run. The upload dialog itself was **not verified**. |

**Limitations.** I did no GPU or hosted execution. The local model probe is CPU-only with stand-in subset sizes
and a higher learning rate, chosen to make state changes visible; it is not a run of the default path. I made no
learner observations.

## 2. Separate judgments

| Dimension | Assessment |
|---|---|
| Technical correctness | The carried modules are sound: transactional `adapt`, digest checks before deserialisation, refusal of non-decoder tensors on reload. Notebook-level state handling is wrong across reruns (RR-M2). The BYOD size contract is mis-stated and unconditional asserts can halt BYOD (RR-M3). The install needs a restart (RR-M1). |
| Promise fulfilment | Every default-path promise is delivered on the documented T4 run. The BYOD promise is delivered only for 50..10,000 unique queries, and only when adaptation happens to beat the frozen model. The optional experiments do not test what they say (RR-M2). |
| Scientific / experimental validity | The split is query-disjoint and balanced. Validation selects the epoch and test is used once. Probe: 0 normalised-exact and 0 Jaccard ≥ 0.8 near-duplicates between test and train/validation, and 0 between validation and train. The lexical baseline is weak by construction and falls below the random floor at recall@3/@5; the notebook does not explain this (RR-m3). The prose quotes CPU results as fixed facts (RR-m1). |
| Learner orientation / progression | Clear opening, prerequisites, data contract and "Look for" cues in most sections. No roadmap, no infrastructure labels on the 35 KB carried-module cells, and no glossary (RR-M4). |
| Explanations / interpretation | Score semantics (uncalibrated, not comparable across models), ties-against-positive and dispersion caveats are well explained. The interpretation hard-codes "+4.5 points … in a few minutes on CPU" and "126 MB", which the T4 record does not show (+3.25 points, 62.9 MB) (RR-m1). |
| Meaningful activity | Only a list of optional experiments. There are no predictions or checkpoints, and the experiments are invalid on rerun (RR-M2, RR-M4). |
| Interaction / pacing / recovery | Install restart (RR-M1). Timings are given per stage. BYOD errors are misleading (RR-M3). BYOD is upload-only with no location field (RR-m6). |
| Completion / transfer | Good: six exports, a provenance-rich `result.json`, an adapter with a manifest, a reload-parity assertion, and "three things to carry to real data". |
| Spec conformance | **MUST failures:** RUN10 / ENV6 (RR-M1); DAT12 / DAT19 (RR-M3). **SHOULD gaps:** GDL1–GDL15 (RR-M4), EXE2 (RR-m6), GDL8 (RR-m1). The declared spec 2.0 is stale against 2.2 (RR-m5). `STATUS.md` / `tutorials/README.md` claim **Release-grade** on a run that needed a manual restart. |

## 3. Prioritized findings

### RR-M1 — Major: Run all stops after the install cell and needs a manual restart

- **Cell/section:** Section 1, cell 3 (`pip install` of `PINS` into the live kernel, then the stale-import guard
  that raises `RuntimeError('… Restart the runtime, then rerun from the top.')`).
- **Observed issue:** on any hosted image whose preloaded torch/numpy/transformers differ from the pins (Kaggle:
  torch 2.10 / numpy 2.0.2 / transformers 5.0.0; Colab is similar), the guard stops the run. The learner must
  restart and run everything again. `docs/release-verification.md` step 4 calls this restart "expected".
- **Consequence:** the default path is not `Run all`. The release record and `STATUS.md` call the notebook
  Release-grade on a run that needed manual intervention.
- **Evidence:** documented execution: `run_summary.json` `runner.passes[0].ok = false` (the guard fired after pip),
  and the release-verification row says "1 restart after install cell". Source: `tools/build_notebook.py:69`.
- **Recommended correction:** replace the in-kernel install with the fleet's **uv isolated-environment pattern**.
  A carrier cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a
  hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in
  that environment, so the kernel's preloaded NumPy/torch are never replaced. Reference:
  `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on `origin/main`.
  Change the install-cell emitter in `tools/build_notebook.py` (around line 50–70) and drop the "restart is
  expected" wording from `docs/release-verification.md`.
- **Acceptance check:** a fresh Colab (or Kaggle) runtime runs the regenerated notebook with **Run all** to the
  final cell with **zero** restarts. The run summary shows one pass with `ok: true`, and that run is recorded in
  `docs/release-verification.md` against the new blob.
- **Spec:** RUN10, ENV6, §5 (closing paragraph), RUN1.

### RR-M2 — Major: reruns never reset to the pretrained model, so the experiments and BYOD reuse an adapted model

- **Cell/section:** Section 3 (cell 11 builds `pipe` once). Section 7 (cell 19: `pipe.adapt` overwrites the last
  decoder layers in place, in `pipeline.py` `adapt`, `model.load_state_dict(merged)`). Sections 5/6 reuse `pipe`.
  The optional-experiment paragraph sits in the Interpretation cell, and the BYOD instruction in the opening cell.
- **Observed issue:** the notebook asks the learner to rerun Section 7 with `TRAINABLE_LAYERS = 1`,
  `TRAIN_CANDIDATES = 6` or `EPOCHS = 3`, to "change `INSTRUCTION` and re-read the frozen numbers", and, for BYOD,
  to "re-run from that cell" (Section 4). Each rerun uses the already-adapted `pipe`:
  - Section 6 prints the adapted model as `frozen_model_test`, with `adapted: True` in the same line.
  - `adapt`'s epoch 0, noted `'frozen model'`, is the previous adaptation.
  - The new run trains on top of it, and Section 8's `delta_vs_frozen` compares against whatever `frozen_test`
    was last computed.
- **Consequence:** "watch the gain shrink with one layer" and similar comparisons measure stacked fine-tuning, not
  the stated variable. A BYOD learner's "frozen" baseline is a Banking77-adapted model, and the adapter then encodes
  two fine-tunings, while the manifest records only the second.
- **Evidence:** direct execution (P5; CPU float32; pinned snapshot verified; stand-in sizes):
  - True frozen validation MRR was 0.9062, and the probe gold score was 0.9697.
  - After one `adapt`, a Section-6-style `evaluate` returned MRR 0.9167 with `adapted: True`.
  - A second `adapt` (`trainable_layers=1`) reported epoch 0 as `'frozen model'` with MRR 0.9167, not 0.9062.
  - The probe gold score stayed at 0.6589, not restored.
  - Source: `pipe` is constructed only in cell 11 (P1 `pipe_constructed_only_in_section3: true`).
- **Recommended correction:** give the pipeline a reset path and use it. Either
  `pipe = Qwen3RerankerPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` at the top of Section 4, or a
  `pipe.reset_adapter()` that restores the base tensors `adapt` already snapshots (`initial_state`). Run it at the
  start of Section 4 and of Section 7. Then state in the experiments paragraph which section to rerun from. Emit
  this from `tools/notebook_template.py` (Section 4/7 cells, lines ~140–160 and ~300; experiments text ~450–455).
- **Acceptance check:** after a full run, rerunning from Section 4 (or Section 7 with `TRAINABLE_LAYERS = 1`) must:
  - print Section 6 `adapted: False` with the same frozen metrics as the first run (within float tolerance);
  - make the new `adapt` epoch-0 validation metrics equal the first run's epoch 0;
  - return the probe shortlist's frozen scores to their original values.
- **Spec:** framework dimension 7 (a control or exercise leaves the notebook in an inconsistent state); GDL10;
  RUN9; DAT13/DAT14 (BYOD must go through the same stages as the sample, from the same starting model).

### RR-M3 — Major: the BYOD size contract is wrong and BYOD can halt on legitimate results

- **Cell/section:** Prerequisites ("a dataset needs 8..20,000 records"); Section 4 cell 13 (`split_dataset`, then
  `validate_dataset(part)` with its default `min_records=8` on **each** split, and the refusal probes on
  `train_records[:8]`); Section 6 / 8 (`lexical_baseline` / `evaluate` validate with `max_records=2000`); the asserts
  in cells 17 and 21.
- **Observed issue:**
  - **Lower bound.** Validation needs at least 8 records, which takes ≥ 50 unique queries at the 15 % default.
    Datasets of 8–49 records are rejected with "split leaves 5 training records…" or "7 records; 8..20000 are
    required". That message reports the **validation split's** size, not the user's file.
  - **Upper bound.** Above 10,000 records the 20 % test split exceeds `MAX_EVAL_RECORDS` (2,000), and Section 6
    raises "2002 records; 1..2000 are required".
  - **Asserts.** `frozen_test['mrr'] > floor['mrr']` and `adapted_test['mrr'] > frozen_test['mrr']` are
    unconditional. A BYOD run where adaptation does not help, a legitimate result the notebook itself says to
    expect elsewhere, stops in Section 8 before the reranking, export and reload stages.
- **Consequence:** a learner following the stated contract gets confusing rejections or a crash that reads as a
  defect. The promised BYOD route through export and reload is not reached.
- **Evidence:** direct execution (P3, carried `samples.py`): n = 8 → "split leaves 5 training records";
  n = 12/20/40/49 → "2/4/6/7 records; 8..20000 are required"; **first passing n = 50** (10/8/32). n = 10,000 passes
  (test 2,000); n = 10,008 / 20,000 → "2002 / 4000 records; 1..2000 are required". The asserts come from source
  (P1 `unconditional_asserts`).
- **Recommended correction:**
  - State the real range: at least 50 unique queries at the default fractions, and at most `MAX_EVAL_RECORDS / 0.2`.
  - Better, validate the user's dataset once with an explicit per-split minimum and name the split and the fraction
    in the message.
  - Keep the two MRR assertions on the sample path only. On BYOD, print the comparison with a "did not improve"
    note and continue.
  - Generator: `tools/notebook_template.py` Prerequisites (~line 121), Section 4 cell (~150–170), asserts at
    ~268 and ~347; `samples.split_dataset`.
- **Acceptance check:**
  - A 49-record BYOD file is rejected before any model call, with a message that names the dataset size and the
    required minimum.
  - A 50-record file and a 10,000-record file both reach `outputs/qwen3_reranker_result.json`.
  - A BYOD run whose adapted MRR ≤ frozen MRR also completes export and reload.
- **Spec:** DAT12, DAT19 (MUST), DAT14, RUN9.

### RR-M4 — Major, learner-facing: `GUIDED` notebook without the guided layer

- **Cell/section:** whole notebook.
- **Observed issue:** the notebook declares `GUIDED` but has none of the following:
  - "How to use this notebook", a roadmap, or a glossary;
  - an Input → Model → Output contract set apart from the dense opening paragraph;
  - a prediction before the frozen/adapted comparisons;
  - "What to notice" notes (there are "Look for" lines, which partly serve this) or "Check your reasoning"
    checkpoints with sample answers;
  - a Predict → Change → Run → Observe → Explain activity;
  - **Infrastructure** labels on the three carried-module cells (≈58 KB of code a learner meets in Section 2);
  - a troubleshooting section;
  - an evidence-based conclusion scaffold.

  Learner activity is a single paragraph of optional experiments, and RR-M2 breaks those.
- **Consequence:** the self-paced learner the mode promises is not asked to predict, interpret or diagnose
  anything. Objectives such as "read recall@k and MRR … and understand what they do and do not measure" have no
  activity that exercises them.
- **Evidence:** source inspection; P1 `guided_layer_markers` all false.
- **Recommended correction:** add the §3.5 layer in `tools/notebook_template.py`, following the 2.2 reference
  notebook (`prithvi-flood-segmentation-pipeline/tutorials/DIMER_Philippines_Flood_Mapping_Capstone.ipynb`). At
  minimum, include:
  - a prediction before Sections 6 and 8 ("will the frozen reranker beat the lexical baseline? by how much will
    two layers help?");
  - a collapsible check after each, reading the actual comparison;
  - one Predict → Change one thing → Run → Observe → Explain activity built on the RR-M2 reset;
  - Infrastructure callouts on Sections 1–3;
  - a short troubleshooting list (restart, Hub download, CPU time, BYOD size);
  - a conclusion template.
- **Acceptance check:** a reviewer can point to a cell implementing each of GDL1–GDL14. Each principal result
  (Sections 6, 7, 8) is preceded by a question and followed by worked guidance.
- **Spec:** GDL1–GDL15 (SHOULD; not by itself a release MUST failure), UX5.

### RR-m1 — Minor: CPU results are written into the prose as fixed facts

- **Cell/section:** Interpretation ("lifts held-out recall@1 by +4.5 points in a few minutes on CPU, with a 126 MB
  adapter"); Section 7 ("reached recall@1 77.9 %"); Section 9 ("about 126 MB").
- **Observed issue / evidence:** the only hosted record of this blob (T4, bfloat16) shows +3.25 points recall@1
  (0.7338 → 0.7662) and a 62,926,256-byte adapter (documented execution). On CPU the local pre-flight took 537 s
  to adapt plus 2 × 130 s to evaluate, which is not "a few minutes".
- **Consequence:** a learner whose run differs reads the prose as contradicting their result.
- **Recommended correction:** print the interpretation numbers from `comparison` and the artifact manifest, or
  phrase them as "the build record saw … on CPU float32; your device and precision will differ". Generator:
  `tools/notebook_template.py` lines ~284, ~361, ~427.
- **Acceptance check:** no fixed delta, accuracy or adapter size appears in markdown without a runtime/precision
  qualifier, or the values are printed from the run.
- **Spec:** GDL8, ENV8.

### RR-m2 — Minor: template escapes leak into the text (`{{id, query, positive, negatives}}`, `{{1,64}}`)

- **Cell/section:** opening cell (BYOD paragraph); Prerequisites (data contract and id pattern).
- **Observed issue:** the doubled braces render literally. The id pattern reads `[A-Za-z0-9_.:-]{{1,64}}`, which
  is a different regular expression if a user copies it.
- **Evidence:** source inspection; P1 lists 3 occurrences; `tools/notebook_template.py` lines 42, 121, 133.
- **Recommended correction:** un-escape the strings that are not passed through `str.format`, or format them.
- **Acceptance check:** the regenerated notebook contains no `{{` or `}}` in markdown.

### RR-m3 — Minor: the lexical baseline is weak by construction, and its below-random scores are unexplained

- **Cell/section:** Section 6 markdown ("this floor is deliberately hard to beat by words alone") and output.
- **Observed issue:** the three hard negatives are, by definition, the phrases with the highest Jaccard overlap
  with the query, so the Jaccard baseline is set up to fail. It scores **below the random floor** at recall@3
  (0.4156 vs 0.5) and recall@5 (0.7532 vs 0.8333). The positive strictly beats every hard negative in only 41/154
  test shortlists and has zero overlap in 37. The notebook calls this baseline "hard to beat", which inverts the
  point, and never explains the below-random values; only the release record does.
- **Consequence:** a learner may read "frozen far above lexical" as evidence of semantic skill rather than of
  negative selection.
- **Evidence:** direct execution P2 (`lexical_below_floor`, `lexical_selection`); documented Kaggle output.
- **Recommended correction:** say that the negatives were selected to beat this baseline, that it can therefore
  fall below random, and that it is a sanity reference, not a competitor.
- **Acceptance check:** the Section 6 markdown explains the below-random recall@3/@5 and describes the baseline as
  weakened by construction.
- **Spec:** EVAL10, EVAL15; framework dimension 5.

### RR-m4 — Minor: `MAX_TRAIN_CANDIDATES` is presented as a ceiling but is only a default

- **Cell/section:** Section 5 prints it among the `ceilings`. README "Ceilings (VAL6)". The optional experiment
  says "set `TRAIN_CANDIDATES = 6`".
- **Observed issue:** `adapt` accepts `2 <= train_candidates <= 8`. The value 4 bounds nothing.
- **Evidence:** source and P4.
- **Recommended correction:** call it the default list length, or enforce it.
- **Acceptance check:** the printed ceilings match what `adapt` enforces.
- **Spec:** VAL6.

### RR-m5 — Minor: stale specification declaration

- **Cell/section:** metadata, opening cell, `NOTEBOOK_SOURCE`, References, `tutorials/README.md`.
- **Observed issue:** declares NOTEBOOK_SPEC 2.0; the current version is 2.2.
- **Recommended correction:** re-declare against 2.2 when RR-M1–M4 are fixed.
- **Acceptance check:** metadata and text say 2.2, and `tools/validate_release_assets.py` checks 2.2.
- **Spec:** §3.4, §30.

### RR-m6 — Minor: BYOD is upload-only, with no location field

- **Cell/section:** Section 4 cell 13 (`files.upload()` whenever `USE_BYOD`).
- **Observed issue:** there is no `BYOD_PATH`-style form field, so neither an executor nor a Jupyter (non-Colab)
  user can drive BYOD; `google.colab` is imported unconditionally on that branch.
- **Recommended correction:** add `BYOD_PATH = ''  # @param {type:"string"}`, read it when set, and upload only
  when empty.
- **Acceptance check:** with `USE_BYOD = True` and `BYOD_PATH` set, cell 13 runs without importing
  `google.colab`.
- **Spec:** EXE1, EXE2 (SHOULD).

### Suggestions

- **RR-S1:** de-duplicate BYOD queries on a normalised key (case, punctuation, whitespace) in `split_dataset` /
  `check_split_disjoint`. The Banking77 release has 25 test messages with a normalised twin in train. The seeded
  draw happens to include none (P2), but a user log easily could.
- **RR-S2:** offer a CPU fast path, for example a 77-shortlist test subset and one epoch, flagged as such. The
  full default is about 15 minutes of model time on CPU.

## 4. Positive findings and non-findings

- **Leakage:** none found. Probe P2 found 0 normalised-exact and 0 Jaccard ≥ 0.8 matches between test and
  train/validation, and 0 between validation and train. Validation-only epoch selection and single use of the test
  split are correct.
- **Supply chain:** inline manifest asserted against the module, `revision` immutable, `verify_snapshot` before
  load, `trust_remote_code=False`, adapter digest checked before deserialisation, non-decoder tensors refused.
- **Score semantics:** uncalibrated, no threshold, and not comparable across frozen and adapted models, each stated
  where the learner meets it (UNC1–UNC4).
- **Reload:** loads a fresh base from files and asserts exact score parity plus a 20-shortlist MRR (VER1–VER5).
- **Data:** the corpus and splits reproduce exactly in the probe: raw rows 10,003 / 3,080, splits 231 / 77 / 154,
  digests `6a5c384c` / `32cc0b97` / `a67a92bd`, and random floor and lexical baseline values equal to the record.

## 5. Promise-to-evidence matrix

| Promise | Implementation | Observable result | Judgment |
|---|---|---|---|
| Pinned runtime, Run all | cell 3 | Kaggle: guard fired, 1 restart | **Not met** (RR-M1) |
| Verified snapshot | cell 11 | 13 files verified, `cuda:0` (Kaggle) | Met (documented) |
| Real corpus, leak-free split | cell 13 | 231/77/154, digests, 4 refusals | Met (documented + direct) |
| Inference contract | cell 15 | 5 sanity checks True, manifest with refusal | Met (documented) |
| Floor / lexical / frozen | cell 17 | 0.1667 / 0.2662 / 0.7338 recall@1 | Met. Interpretation caveat RR-m3. |
| Bounded listwise fine-tuning | cell 19 | 31.46 M of 595.8 M trainable, val MRR 0.870 → 0.894 | Met (documented) |
| Held-out comparison | cell 21 | +0.0325 recall@1, +0.0206 MRR (T4) | Met. Prose numbers differ (RR-m1). |
| Before/after, export, reload | cell 23 | 2/3 reordered; parity exact | Met (documented) |
| Optional experiments | Interpretation | stack on the adapted model | **Not met** (RR-M2) |
| BYOD through all stages | cell 13 → 23 | 50..10,000 only; asserts can halt | **Partly met** (RR-M3); upload not verified |

| Objective | Learner activity | Evidence it was exercised |
|---|---|---|
| Read recall@k / MRR beside floors | Read printed tables | None asked (RR-M4) |
| Run bounded fine-tuning with explicit hyperparameters | Form fields + optional experiments | Experiments invalid on rerun (RR-M2) |
| Compare rankings before and after | Read the side-by-side output | No prompt to explain the change (RR-M4) |
| Export and reload with parity | Read the parity line | Asserted by the cell |

## 6. Readiness

**Needs revision.** Remaining gates:

1. RR-M1–RR-M3 fixed (RUN10/ENV6 and DAT12/DAT19 are applicable MUSTs).
2. RR-M4 guided layer added.
3. A fresh clean-runtime run of the regenerated blob recorded with zero restarts, preferably on Colab.

## 7. Verified versus inferred

- **Verified by direct execution:** the corpus and split reproduction, the absence of leakage, the
  lexical-below-floor values, the BYOD size boundaries, and the stacked-adaptation rerun behaviour (CPU, stand-in
  sizes).
- **Verified by documented execution:** the default path on Kaggle T4 for this exact blob, including the restart.
- **Inferred:** that a fresh Colab image also trips the guard. Colab's preinstalled torch/numpy differ from the
  pins, but this review did not run Colab.
- **Most likely to be wrong:** RR-M4's severity. The guided layer is a `SHOULD`, and §33 says older notebooks are
  not nonconformant without it. A reviewer weighing the existing "Look for" cues and the strong interpretation
  section could reasonably call it Minor.

Probe bundle: `qwen3_reranker_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
