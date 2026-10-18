# Pre-registration amendment 1 (qwen3_reranker_colab), 2026-10-18

Amends `docs/qwen3-reranker-preregistration.md` (committed alone in `5b3aaa2`). This amendment is also committed
alone, before the code that implements it, and before any model has scored the fresh test. No model score on the
fresh test exists. The only development look is described in section 9 below: a reviewer read the lexical baseline's
tie count on the fresh test, and no model or metric was involved. It follows the peer review of PR #10 (findings
F1–F11). Nothing below changes the grid, the margin, the split, the fresh-test draw or any fixed setting.

## 1. Uncertainty on the fresh test (F5)

- **Method:** a **paired bootstrap** over the fresh-test shortlists, with **n_boot = 10,000** resamples and a fixed
  seed of **20261018** (`random.Random`).
  - Each resample draws 385 query indices with replacement.
  - It uses the same indices for every system, so differences are paired.
  - The per-query statistics are the pessimistic reciprocal rank (MRR) and the recall@1 hit.
  - The interval is the 2.5th–97.5th percentile of the resampled means.
- **Reported:**
  - recall@1 and MRR with a 95 % CI for each model scored on the fresh test;
  - the adapted − frozen difference in MRR and in recall@1, each with its 95 % CI, beside its point estimate.
- **Verdict (fresh test, per metric):**
  - `improved` if the CI's lower bound is > 0;
  - `worse` if its upper bound is < 0;
  - `inconclusive` otherwise.
  - MRR is the headline metric, since it is also the selection metric. If the frozen model is kept, no difference
    is computed.
- **What the interval covers:** only the choice of shortlists. It does not estimate split-to-split or seed-to-seed
  variation, which one run cannot measure.

## 2. The exposed development test decides nothing (F2)

The adapted-vs-frozen verdict on the exposed development test is printed and recorded. It no longer stops the cell.
No stop in Section 8 reads a test split.

The only remaining stop is the model-state refusal, and it fires before anything is scored. The evaluation report,
including the fresh-test block, is written immediately after the fresh test is scored, so a pre-registered result is
never printed without a file.

## 3. Device and precision (F7)

`device` and `dtype` are recorded in the run settings and in the fresh-test output. They are **not** pre-registered
settings: a CPU float32 run with every pre-registered setting is still labelled "pre-registered evaluation (this
runtime)", and its output names float32.

The **evaluation of record is the first hosted T4 run (cuda, bfloat16)**. CPU float32 and T4 bfloat16 numbers differ
and are not comparable. For example, the adapted 2e-5 run reached validation MRR 0.8937 on the 2026-10-09 T4 run and
0.9232 in the 2026-10-18 CPU check. A reproduction of the record is a later hosted bfloat16 run with identical
settings.

## 4. The evaluated model is a re-train of the selected learning rate (F6)

Section 7 trains the three grid runs, selects the learning rate, and then trains that learning rate **again** with the
same seed. The fresh test scores this re-train, not the grid run whose validation MRR justified the selection.

Both kept-epoch validation MRRs are recorded. If they differ by more than **0.010** (the grid margin), the model state
counts as differing: the run is not pre-registered and does not score the fresh test. On CPU the re-train reproduced
the grid run exactly (0.9232). On CUDA bfloat16, bit-exactness is not guaranteed.

## 5. The fingerprint is recomputed in Section 8 (F8)

Section 8 recomputes the grid fingerprint from the current dataset digests and from the training settings recorded on
the adapter (`pipe.adapter`), not from the form fields. It then compares the result with the fingerprint the grid
selection was made on. The training keys of the settings check are also read from the adapter.

## 6. What the margin means (F4)

The 0.010 margin is unchanged. It is a **"keep the default unless clearly beaten"** tolerance, deliberately at or below
the sampling noise of 77 validation queries (one rank-2→1 flip moves MRR by 0.0065). It is not a detection threshold,
and a run that does not clear it says nothing about whether two learning rates differ.

## 7. A known property of the shortlists (F3; T73-9, not fixed)

When a message shares no token with any other intent phrase, `sample_negatives` falls back to case-sensitive sort order
for its three "hard" negatives: `Refund not showing up`, `activate my card`, `age limit`. This affects:

| Split | Shortlists affected |
|---|---|
| train | 7 / 231 |
| validation | 1 / 77 |
| exposed development test | 3 / 154 |
| fresh test | 10 / 385 |

All systems see the same shortlists, so comparisons between systems are not biased, but those shortlists are easier
than described. Fixing this changes every digest and the fresh-test pins, so it needs a new data revision and a new
pre-registration.

## 8. Timings (F10)

These estimates come from recorded timings, replacing "about +7.6 min".

| | T4 | CPU |
|---|---|---|
| One default `adapt` | 229 s | — |
| The grid (three extra runs) | about 11.5 min | about 39 min (2,354 s) |
| Fresh-test scoring (2 × 2,310 pairs) | about 4 min | about 11 min |
| Whole path | about 25 min | 75 min or more |

The whole-path CPU figure assumes the masked CPU check's 3,855 s plus the fresh test.

## 9. Development look (SCORE_ONCE_RULE item 8)

On 2026-10-18 the peer reviewer's data script ran the **lexical baseline** on the 385 fresh shortlists and read only
its `ties_with_positive` count (**175 / 385**). No model was run, no recall@k or MRR was printed or read, and no setting
was changed because of it. This is recorded as a development look in `docs/release-verification.md`.

## 10. The record and `RECORDED_EVALUATION` (F9)

`protocol.RECORDED_EVALUATION` is set only by the requote that follows the hosted T4 run. `docs/release-verification.md`
names the run of record. A reproduction is defined by identical pre-registered settings and identical module code,
except for `RECORDED_EVALUATION`.
