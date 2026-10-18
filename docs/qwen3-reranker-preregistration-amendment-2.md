# Pre-registration amendment 2 (qwen3_reranker_colab): training precision, 2026-10-18

This amends `docs/qwen3-reranker-preregistration.md` (`5b3aaa2`) and amendment 1 (`3eb2c57`). Like them, it is
committed **alone**, before the code that implements it.

**Status of the fresh test:** still unscored by any model. The only development look remains the one recorded in
amendment 1, section 9.

## 1. Why

`Qwen3RerankerPipeline.from_pretrained` loaded the model in `torch.bfloat16` on CUDA, and `adapt` ran AdamW directly
on those bfloat16 parameters. There was no float32 master copy, so updates smaller than bfloat16's resolution were
rounded away.

The sibling row 72 has the same design, and on a T4 it reached validation MRR 0.788, against 0.828 in CPU float32.
For this tutorial, the 2026-10-09 T4 run reached validation MRR 0.8937 for the adapted 2e-5 run, against 0.9232 on
CPU (amendment 1, section 3).

## 2. Hosted look disclosed (aborted, selection-only)

On 2026-10-18 the orchestrator started a hosted T4 run of notebook revision `b669f41` (bfloat16 training) and stopped
it in Section 7, during the grid and **before Section 8**:

- **Read before the stop:** the T4 grid run for lr 1e-5, kept-epoch validation MRR 0.8872 (by epoch 0.8699, 0.8786,
  0.8872), and Section 6's exposed-development numbers.
- **Not read:** no fresh-test number. Section 8 never ran.

This look is validation-only and selection-only. It was made under the training precision this amendment replaces, and
it chooses nothing. It is recorded in `docs/release-verification.md`.

## 3. The precision recipe (chosen: whole model in float32 on every device)

The options were:

| Option | Recipe |
|---|---|
| (a) | the whole model in float32 |
| (b) | frozen layers in bf16/fp16, trainable layers upcast to float32, with autocast |
| (c) | fp16 autocast with GradScaler and float32 trainable parameters |

**Chosen: (a).** On CUDA, as on CPU:

- The model is loaded in **float32**.
- The forward pass runs in float32.
- The trainable parameters (the last two decoder layers, 31.5 M) are float32 and are updated in float32.
- AdamW's state (`exp_avg`, `exp_avg_sq`) is float32.

`adapt` refuses to train non-float32 trainable parameters. The saved adapter is float32 (about 126 MB on every device),
and `load_artifact` refuses an adapter whose tensors are not float32.

**Reasons:**

- **Turing has no bfloat16 tensor-core path.** On a T4, bfloat16 matrix multiplies do not use tensor cores, so
  bfloat16 buys little speed over float32 there. The 2026-10-09 bfloat16 run's timings (229–240 s per two-epoch
  `adapt`) are therefore a fair guide to float32.
- **Memory fits.** The 0.6 B model is about 2.4 GB of weights in float32. Gradients and AdamW state for 31.5 M
  parameters add about 0.4 GB. The full-vocabulary logits of a 16-sequence training batch, and the gradient flowing
  back through them, need about 4 GB at the 192-token ceiling. Estimated peak: **about 7–9 GB of the T4's 16 GB**.
- **(b) is unsafe here.** Mixing dtypes inside one module stack needs per-layer casting at the boundary between the
  bf16 and float32 layers.
- **(c) is unsafe here.** fp16 has a narrow exponent range, which risks overflow in Qwen3's activations. It would also
  need GradScaler machinery for a 31.5 M-parameter adapter.
- **Comparability.** With (a), CPU and CUDA run the same arithmetic precision. Kernel-level differences remain, so
  numbers can still differ in low-order digits.

## 4. Estimates (replacing amendment 1, section 8 for T4)

- **T4:** about 4 min per two-epoch `adapt` (about 12–16 min for the grid plus the re-train). The whole path takes
  **about 25–35 min**. The first hosted run measures this, and the release record reports it.
- **CPU:** unchanged (75 min or more), because CPU was already float32.

## 5. Recorded, not pre-registered

`device` and `dtype` remain recorded in the run settings, the fresh-test block and the adapter record (`precision`),
and they remain not pre-registered settings.

The **evaluation of record is now the first hosted T4 run with float32 training** (cuda, float32) of a notebook
revision that implements this amendment. Earlier hosted numbers came from bfloat16 runs:

- the 2026-10-09 exposed-development numbers;
- the aborted 2026-10-18 grid look.

They stay labelled as bfloat16 and are not comparable.

## 6. Unchanged

The following are all unchanged:

- the grid ({1e-5, 2e-5, 5e-5}, kept-epoch validation MRR, margin 0.010, tie rule and keep-frozen rule);
- the re-train check (0.010);
- the fixed settings;
- the fresh test (385 shortlists, pins `f59a5459…` / `9bc26a46…`);
- the tie policies;
- the paired bootstrap (10,000 resamples, seed 20261018) and its CI-based verdict;
- the score-once rules.
