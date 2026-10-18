# Pre-registration: learning-rate selection and the fresh test (qwen3_reranker_colab)

Date: 2026-10-17. Committed alone, before any run of the grid or any scoring of the fresh test.
Applies to `tutorials/qwen3_reranker_colab.ipynb` from the commit after this one.

## Why

The default learning rate (2e-5) was kept over 5e-5 in the CPU build record of PR #3 (commit `d79994f`,
notebook blob `75fbc7d7`) because test recall@1 was higher (77.9 % against 76.0 %), while the kept-epoch
validation MRR was tied at 0.923. The 154 test shortlists therefore informed a setting. From now on they are
the **exposed development test**: they are still shown, for continuity, but they decide nothing and they are not
an independent estimate. A **fresh test** that has never been scored replaces them as the held-out evaluation.

## Fixed settings

Every one of these must hold for the fresh test to be scored (`protocol.PREREGISTERED_SETTINGS`).

| Setting | Value |
|---|---|
| Data source | the pinned Banking77 sample (`USE_BYOD = False`), PolyAI-LDN/task-specific-datasets @ `57ec275d`, `train.csv` SHA-256 `b06e26ac…`, `test.csv` SHA-256 `d12d6e3b…` |
| Split seed | `SPLIT_SEED = 42`; splits 231 / 77 / 154 (3 / 1 / 2 per intent), digests `6a5c384c…` / `32cc0b97…` / `a67a92bd…` (unchanged) |
| Fresh test | 385 shortlists (5 per intent), see below; source-id SHA-256 `f59a5459…`, dataset digest `9bc26a46…` |
| Instruction | `Given a customer support message, judge whether the document names the banking intent it expresses` |
| Epochs | 2 (kept epoch = highest validation MRR; epoch 0 is the frozen model) |
| Batch size | 4 queries per step |
| Trainable layers | the last 2 decoder layers |
| Training candidates | 4 per query (positive + first 3 negatives) |
| Adapt seed | 0 |
| Training token budget | `MAX_TRAIN_TOKENS = 192` |
| Grid | run in the same notebook run (`RUN_GRID = True`), on the same data and settings (fingerprint) |
| Learning rate | the grid's selection |

## The grid (selection on validation only)

- **Configs:** learning rate ∈ {1e-5, **2e-5 (default)**, 5e-5}; everything else as in the table.
- **Selection metric:** the kept-epoch validation MRR (77 validation shortlists, pessimistic ties) of each run.
- **Margin:** 2e-5 stays unless another learning rate beats it by **≥ 0.010**. This is a practical tolerance, not
  a significance threshold: 77 validation queries cannot separate small differences from noise.
- **Tie rules:** if both challengers clear the margin, the higher wins, unless the two are within 0.010 of each
  other, when the lower learning rate wins.
- **Keep-frozen rule:** if the selected run's kept-epoch validation MRR does not beat the frozen model's validation
  MRR (epoch 0) by ≥ 0.010, the frozen model is the evaluated system and the adapted model is not scored on the fresh
  test.
- The grid never reads either test. It is ON by default; the review's estimate is about +7.6 min on a T4
  (three extra two-epoch runs).

## The fresh test

- **Source:** `test.csv` of the same pinned commit, which the tutorial had drawn only 2 messages per intent from.
- **Draw:** the same seeded per-intent shuffle (`SPLIT_SEED = 42`) that produced the original test, taking the
  positions **after** the original 2 per intent, so the original splits do not change; 5 per intent = 385.
- **Exclusions:** a message is skipped when its normalised text (lower-cased alphanumeric tokens) is already in a
  split, appears in `train.csv` (23 such messages in the pool), or is a near-duplicate, token Jaccard ≥ 0.8, of a
  training or validation message (20 in the pool). In the draw this skipped 2 and 4 messages respectively.
- **Pins:** the SHA-256 of the sorted source ids and the dataset digest are constants in `samples.py`, checked at
  runtime together with disjointness (`check_fresh_test`). Shortlists are built exactly as for the other splits.

## Tie policy (all systems)

Every system is reported under both policies, side by side: **pessimistic** (ties count against the positive; the
headline numbers) and **expected** (`tie_expected`, the exact expectation over random orderings of the tied
candidates, the random floor's convention). `ties_with_positive` counts the shortlists where a negative ties the
positive. The lexical baseline ties often (86 of the 154 exposed shortlists); model scores can tie only when two
scores are bit-identical, which bfloat16 makes possible, so the count is printed for every model evaluation too.

## Scoring rules (SCORE_ONCE_RULE)

- The fresh test is scored only when every fixed setting holds, the grid ran on the same inputs, the learning rate is
  the grid's selection, and the model in memory is the selected run (`preregistered`). Its output carries
  `test_look` and `settings_differing`.
- Anything else is **exploratory**: validation and the exposed development test only, written to separate
  `*_exploratory_<n>.json` files. Form-field changes, BYOD, the activity and the "More experiments" all make a run
  exploratory, and the notebook says so.
- In a notebook the label is "pre-registered evaluation (this runtime)"; a second identical scoring in the same
  runtime is a "reproduction". The first hosted T4 run with these settings is **the evaluation of record**, and
  `docs/release-verification.md` records it and any later reproductions.
- Development checks on CPU run with the fresh test **masked** (`DIMER_MASK_FRESH_TEST=1`): they report validation
  and the exposed development test only and change no setting.

## Reported per run

Random floor, lexical baseline, frozen model and the evaluated system on the fresh test (both tie policies, with
`ties_with_positive`); the same on the exposed development test, labelled as such; the grid table; the kept epoch;
`test_look`; `settings_differing`.
