# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/qwen3-reranker-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/qwen3-reranker-pipeline/blob/main/tutorials/qwen3_reranker_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen%2FQwen3--Reranker--0.6B-ffcc4d?style=flat)](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
[![Upstream](https://img.shields.io/badge/Upstream-QwenLM%2FQwen3--Embedding-181717?style=flat&logo=github&logoColor=white)](https://github.com/QwenLM/Qwen3-Embedding)
[![arXiv](https://img.shields.io/badge/arXiv-2506.05176-b31b1b.svg)](https://arxiv.org/abs/2506.05176)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `qwen3_reranker_colab.ipynb` | `TASK-INFERENCE` | Qwen3-Reranker-0.6B pointwise query-document reranking on a synthetic query with three authored candidates; `yes`-share softmax relevance scores (not calibrated, no threshold) and a stable ranking; no metric exists or is reported | CPU float32 (CUDA bfloat16 used automatically when available) | one UTF-8 text file (query line + document lines), gated off by default | **Candidate** — static checks pass; the clean-runtime execution row in `../docs/release-verification.md` is pending and must be recorded for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `Qwen3RerankerPipeline` from the repository public API rather than reimplementing model loading; the pipeline pins the immutable upstream revision, stages the missing snapshot file through the package's `stage_missing_files(..., allow_download=True)`, loads only from a digest-verified local snapshot (`verify_snapshot`), and refuses remote model code. The notebook never calls `transformers` or `huggingface_hub` directly and never reaches text generation.
- Score semantics (UNC1/UNC2/UNC4): the score is the `yes` share of a two-way softmax over the last-position `yes`/`no` logits — a relevance score, not a calibrated probability; `ranking` is an ordering, not a decision; the pipeline ships no threshold and the caller owns any cut-off. The `instruction` is surfaced as part of the prompt.
- No intrinsic metric exists (EVAL9): the repository ships no metric helper; the notebook says so, names what a real evaluation needs (relevance judgements per query-candidate pair and the caller's own nDCG/MRR code over many queries), and presents the synthetic ordering as a falsifiable plumbing check only. Recorded `SHOULD` deviation: EVAL11 (no baseline — none is meaningful without judgements).
- Ceilings `MAX_PAIRS`, `MAX_TEXT_CHARS`, `MAX_TEXT_TOKENS` are surfaced before the model runs; per-pair `n_tokens` and `truncated` flags are printed and exported after it (DAT22/DAT23).
- The default sample is synthetic text authored in code; `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
