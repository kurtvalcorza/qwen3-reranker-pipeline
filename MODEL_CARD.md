---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: text-ranking
base_model: Qwen/Qwen3-Reranker-0.6B
date_published: "2025-05-29"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/Qwen/Qwen3-Reranker-0.6B)"
---

# Qwen3-Reranker-0.6B — Text Reranking Model (Cross-Encoder)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen%2FQwen3--Reranker--0.6B-ffcc4d?style=flat)](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-QwenLM%2FQwen3--Embedding-181717?style=flat&logo=github&logoColor=white)](https://github.com/QwenLM/Qwen3-Embedding)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2506.05176-b31b1b.svg)](https://arxiv.org/abs/2506.05176)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/qwen3-reranker-pipeline/blob/main/tutorials/qwen3_reranker_colab.ipynb) [`qwen3_reranker_colab.ipynb`](https://github.com/kurtvalcorza/qwen3-reranker-pipeline/blob/main/tutorials/qwen3_reranker_colab.ipynb)  
  *Pointwise query–document reranking with the pinned `Qwen/Qwen3-Reranker-0.6B` weights on a synthetic query with three candidates: `yes`-share softmax relevance scores (uncalibrated, no threshold) and a stable ranking; no metric is reported.*

---

#### Description

`Qwen/Qwen3-Reranker-0.6B` is the smallest reranker of the Qwen3 Embedding series (Zhang et al., arXiv:2506.05176), pinned here to revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`. The upstream README states it is built on `Qwen/Qwen3-0.6B-Base`; the pinned `config.json` is a `Qwen3ForCausalLM` decoder with 28 layers, hidden size 1024, 16 attention heads over 8 key-value heads, and a 40,960-token position budget. It is used as a pointwise cross-encoder: the query and one document are written into a fixed chat prompt that asks whether the document meets the query's requirements, the decoder runs one forward pass, and the score is the softmax over the logits of the tokens `no` and `yes` (ids 2152 and 9693, pinned by the snapshot's `1_LogitScore/config.json`) at the last position — the model never generates text. Nothing is trained or conditioned in this repository. What it adds is the `Qwen3RerankerPipeline` class in `src/qwen3_reranker_pipeline/pipeline.py`: manifest verification (`verify_snapshot`), fresh-clone staging (`stage_missing_files`), a loader that refuses remote code and cross-checks the yes/no token ids, input validation with named ceilings, the README's exact prompt and scoring contract, and provenance fields in every result.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is query-document relevance scoring: input is a list of up to `MAX_PAIRS = 32` `(query, document)` string pairs plus an optional task instruction; output is one float score in [0, 1] per pair, aligned to the input order, together with a `ranking` of pair indices by descending score. Envisioned applications are second-stage reranking of the top candidates returned by a first-stage retriever — the sibling `qwen3-embedding-pipeline`, BM25, or an existing search index — for enterprise document search, question-answering over a corpus, code search, and cross-lingual retrieval across the 100+ languages upstream claims. In a larger system the pipeline is the precision stage of a retrieval stack: it is too expensive to score a whole corpus (one forward pass per pair) and is meant to reorder tens of candidates per query.

###### Primary Intended Users

Intended users are machine-learning engineers, search engineers, and application developers integrating reranking into research prototypes, internal enterprise search, or the DIMER model workbench. The pipeline assumes its users understand that the score is a relative relevance signal and not a calibrated probability, that scores are comparable within one query's candidate list but not across queries or instructions, that cost grows linearly with the number of pairs and with prompt length up to 8,192 tokens, that the instruction text changes the scores, and that ranking quality on their own corpus must be measured with their own relevance judgements before deployment.

###### Out-of-scope use cases

1. **Capability boundary:** the model does not embed text, retrieve from a corpus, generate answers, or extract spans; first-stage retrieval is the sibling `qwen3-embedding-pipeline`, and text generation belongs to the language-model pipelines. It scores one document against one query at a time and has no notion of a document set beyond the batch it is given.
2. **Input boundary:** `rerank()` takes a list of 1 to 32 pairs (`MAX_PAIRS`) of two non-empty strings (`TypeError` for a bare string, a non-pair, or a non-string element; `ValueError` for an empty list or blank text); either text above 100,000 characters (`MAX_TEXT_CHARS`) is rejected before tokenisation; a prompt that exceeds 8,192 tokens (`MAX_TEXT_TOKENS`, including the fixed prefix and suffix) is truncated longest-first and flagged `truncated: True`. Images, audio, and structured records are not inputs.
3. **Decision boundary:** not for autonomous accept/reject decisions about people or claims from a score — résumé screening, moderation, fraud, eligibility — without a human reviewing the ranked items; the score is uncalibrated and no threshold is shipped.

#### Factors

###### Groups

The pipeline is not human-centric by design — it scores text pairs — but the queries and documents it ranks routinely describe or are written by people, and neither the base model's pretraining corpus nor the reranker's fine-tuning pairs are enumerated or group-audited by the Qwen team in the pinned README. No group-level audit exists here either. Reranking bias transfers directly into the operator's retrieval pipeline: if documents mentioning certain names, dialects, or groups are systematically scored higher or lower for otherwise-equivalent queries, the final ranking is skewed for those groups. The downstream operator must run a fairness audit on their own retrieval task — for example, score and rank position for paired queries or documents that differ only in a name, gender marker, or language variety — before relying on the ranking.

###### Instrumentation

There is no physical sensor: the training data is text produced by keyboards, web crawls, transcription, and machine translation, tokenised by the Qwen2 byte-pair tokenizer (`tokenizer.json`, 151,669-entry vocabulary). The characteristics that materially affect the data are encoding, normalisation, and language mix at collection time, none of which the upstream README describes beyond the "100+ languages" claim and the note that most training instructions were written in English. At inference the pipeline wraps the caller's strings verbatim in the fixed prompt: no case folding, whitespace normalisation, HTML stripping, or language detection. Defects in the operator's text pipeline — mis-decoded bytes, boilerplate, concatenated records, wrong-language text — reach the model unchanged; the pipeline detects only the shape errors it validates (type, emptiness, length ceilings) and reports the truncation it applies.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0` (the venv build is `2.14.0+cu130`), `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3` (`pyproject.toml`). The loader runs float32 on CPU and bfloat16 on CUDA (the checkpoint's native `torch_dtype`); the recorded smoke ran on CPU only (`CUDA_VISIBLE_DEVICES=-1`, `device="cpu"`), loading 1.19 GB of weights in 5.11 s and scoring four short pairs in 1.0 s on the host described under "Runtime". Flash attention and the CUDA path were not exercised, and CPU latency scales with prompt length, so long documents near the 8,192-token ceiling will be much slower than the smoke. Data environment: the model assumes natural-language or code queries and documents resembling its undisclosed training mix, with an English task instruction as the README recommends; scores degrade, without any error, on domains far from that mix, on languages with thin coverage, and when the instruction does not describe the actual task.

#### Metrics

###### Performance Measures

The pipeline reports no performance measure. `rerank()` returns a relevance score per pair and the induced ranking; there is no ground truth in a batch of pairs against which to score them, and the repository ships no relevance judgements. Upstream reports retrieval-subset scores for this checkpoint (for example MTEB-R 65.80 in the pinned README's evaluation table, computed on top-100 candidates from Qwen3-Embedding-0.6B); those are upstream-reported and were not reproduced by this pipeline. An operator who needs a measure must supply queries, candidate documents, and graded relevance judgements, then compute a ranking metric such as nDCG@10 or MRR over the reranked lists — ranking metrics because reordering candidates is the intended use, and a score with no fixed cutoff has no discrete correctness to count. The public `evaluation_report()` helper makes that absence machine-readable rather than silent: it always returns a report whose `verdict` is `not-measurable` with an empty `metrics` list, and whose `needs` field names the per-query relevance judgements and caller-side ranking-metric code that would be required to score this model.

###### Decision thresholds

No decision threshold is shipped. `rerank()` applies a two-way softmax to the `no`/`yes` logits and returns the `yes` share as the score; it applies no cutoff, asserts no pair relevant or irrelevant, and the `ranking` field is an ordering, not an acceptance rule. The threshold was withheld because the useful cutoff depends on the corpus, the instruction, and the relative cost of surfacing an irrelevant document versus hiding a relevant one, none of which this repository can see; upstream ships none either. The operator owns it: choose a score cutoff or a top-k from precision-recall curves on labelled pairs from the deployment, lowering it where a hidden relevant document costs more than reviewing extra candidates and raising it where noise is expensive. The smoke run's values (0.9995 and 0.9994 for the README's matched pairs, 0.0000 for the swapped pairs) are four observations, not calibration points.

###### Approaches to uncertainty and variability

No metric is reported, so no estimation procedure or dispersion applies; the operator who computes nDCG or MRR on their own judgements owns the split design and any bootstrap or repeated-run estimate. Within the pipeline the forward pass is deterministic on a fixed device and dtype — no sampling, dropout inactive under `torch.inference_mode`, no seed needed — but bfloat16 on CUDA versus float32 on CPU, kernel selection, and left-padding differences between batch compositions can move a score in its low-order digits and, for near-tied pairs, swap adjacent ranks. The score is a softmax over two logits from a model fine-tuned to emit `yes`/`no`; it is not calibrated, and the smoke shows it saturating near 1.0 and 0.0 on easy pairs. A caller who needs a calibrated relevance probability must fit a mapping from score to relevance on their own labelled pairs.

#### Ethical considerations and biases

###### Data

The upstream README discloses only that the model is built on `Qwen3-0.6B-Base`, supports 100+ languages, and that its training instructions were mostly English; the technical report (arXiv:2506.05176) describes large-scale synthetic and public retrieval pairs for training, and the base model's pretraining corpus is web-scale text that Qwen does not enumerate. Whether personal, copyrighted, or otherwise restricted text is included is therefore unknown, not ruled out. This repository distributes code, tests, and documentation; it does not vendor the weights in Git (`weights/qwen3-reranker-0.6b/model.safetensors` is git-ignored and reproduced from the pinned revision via `stage_missing_files`), and it ships no query or document corpus. The operator must audit the queries and documents they score for personal, confidential, or legally restricted content and must recognise that a query log paired with ranked documents is itself personal data where the queries are; the pipeline performs no such check.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by anyone; the only validation performed is the offline unit suite and one CPU smoke recorded under "Runtime". Use in such domains is foreseeable — ranking candidates against a job description, ordering case files or clinical notes by relevance to a question — and would be admissible only as a retrieval aid whose ordering a qualified person reviews, after an independent evaluation of ranking quality and group-level behaviour on that deployment's own data, and with whatever regulatory clearance the domain requires.

###### Mitigations

1. **Supply-chain integrity:** `MODEL_ID` and the 40-hex `MODEL_REVISION` are module constants; `verify_snapshot()` checks `modelId`, `revision`, and the byte size and SHA-256 of all 13 manifest entries before any load, raising on the first mismatch (a test flips one hex digit and asserts it raises). `stage_missing_files()` fetches only manifest-listed files absent on disk, only at `MODEL_REVISION`, only when `allow_download=True`, and refuses a manifest naming another model. The loader passes `local_files_only=True` for the verified directory, `trust_remote_code=False` always, and raises if the tokenizer's `yes`/`no` ids differ from `YES_TOKEN_ID`/`NO_TOKEN_ID` (a test checks those constants against `1_LogitScore/config.json`).
2. **Input integrity:** `rerank()` rejects a bare string, an empty or oversize batch, a non-pair, non-string or blank elements, text above `MAX_TEXT_CHARS`, and an empty instruction before the model runs (one test per check), and checks the backend's logit shape after. The public `validate_inputs()` helper applies those same checks through the same private `_check_inputs()` function and returns an input manifest (schema, ceilings, per-pair observations, verdict, findings), so a caller can record exactly what was accepted or rejected without duplicating the validation logic.
3. **Statistical mitigations:** none are implemented; the pipeline does no training, so there is no balancing or subsampling to apply.
4. **Reproducibility:** every runtime dependency is pinned with `==`; each result carries `model_id`, `model_revision`, `instruction`, `score_kind`, `n_tokens`, and `truncated`.
5. **Refusals:** no threshold, top-k cutoff, or generation is exposed; the model's language-modelling head is used only to read two logits, and free-text generation is deliberately not reachable through this package.

###### Risks and harms

1. **Silent truncation:** a prompt past 8,192 tokens is cut longest-first, so a long document is judged on its head only; the `truncated` flag is set, and an operator who ignores it ranks on partial content. Likely for long reports and transcripts.
2. **Instruction mismatch:** scoring with an instruction that does not describe the task shifts scores without any error; the operator bears the harm as silently worse ranking.
3. **Bias amplification in ranking:** systematic score differences for documents mentioning certain names, dialects, or groups reorder who or what is surfaced; the data subject bears the harm, and it compounds when the ranking feeds a human decision.
4. **Saturated confidence:** scores cluster near 0 and 1 (smoke), so a wrong judgement looks as certain as a right one; an operator reading the score as a probability is misled.
5. **Cost and denial of service:** each pair is a full forward pass over up to 8,192 tokens; unbounded candidate lists or adversarially long documents exhaust CPU time, which `MAX_PAIRS` and `MAX_TEXT_CHARS` bound per call but not across calls. Magnitude ranges from degraded search to a wrongful screening outcome when the decision boundary above is ignored.

###### Use cases

The pipeline must not be used to rank people against a prototype for employment, housing, credit, insurance, education, or healthcare access, to profile individuals from their queries or documents, for surveillance or social scoring, or for unlawful discrimination in any setting. It must not be used to rank content for manipulation or deception, to score text obtained without authorisation, or in any way that breaches the Apache-2.0 terms of the upstream weights or the DIMER deployment terms. These prohibitions hold even where the model would produce a plausible ordering.

## Immutable provenance

- Model: `Qwen/Qwen3-Reranker-0.6B`
- Revision: `e61197ed45024b0ed8a2d74b80b4d909f1255473`
- Manifest: `weights/qwen3-reranker-0.6b/dimer-base-manifest.json`, format `dimer_hf_snapshot` v1, 13 files, `totalBytes` 1207486774
- `model.safetensors` (1,191,588,280 bytes) SHA-256: `27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b`
- `config.json` (727 bytes) SHA-256: `d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0`
- `1_LogitScore/config.json` (57 bytes) SHA-256: `73e3156450564d8a98b7e47bcf5aace0f29600828b51937da545571e84db3ff3`
- Upstream reference: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B

## Input/output contract

- `Qwen3RerankerPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`: stages missing manifest files (only with `allow_download=True`), verifies the snapshot, loads `AutoTokenizer` (`padding_side="left"`) and `AutoModelForCausalLM` with `local_files_only=True`, checks the `yes`/`no` token ids; `device` defaults to `cuda:0` when visible, else `cpu`.
- `rerank(pairs: list[tuple[str, str]], instruction: str = DEFAULT_INSTRUCTION) -> dict` with keys `scores` (list of floats in [0, 1], aligned to `pairs`), `ranking` (pair indices by descending score, stable), `score_kind` (`"relevance score, not a calibrated probability"`), `instruction`, `n_tokens` (per pair, prefix and suffix included), `truncated` (per pair), `model_id`, `model_revision`.
- Prompt: `PREFIX` (system turn) + `<Instruct>: …\n<Query>: …\n<Document>: …` + `SUFFIX` (assistant turn with an empty `<think>` block), exactly as the pinned README's `format_instruction`/`process_inputs`.
- Constants: `MAX_PAIRS = 32`, `MAX_TEXT_TOKENS = 8192`, `MAX_TEXT_CHARS = 100000`, `YES_TOKEN_ID = 9693`, `NO_TOKEN_ID = 2152`, `DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"`.
- No metric helper is shipped; ranking evaluation needs relevance judgements the repository does not have.

## Runtime

- Pins (`pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`; dev `pytest==8.4.2`, `ruff==0.16.6`. Python 3.12, Windows venv `dimer-next16`.
- Executed 2026-09-12: `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -o addopts= tests` — 18 passed, exit 0; `ruff check src tests` clean.
- Smoke, executed on CPU (`CUDA_VISIBLE_DEVICES=-1`, `from_pretrained(device="cpu")`, float32): the README's two query/document pairs plus the two swapped pairs; load 5.11 s, scoring 0.999 s, 6.11 s total; scores [0.9995, 0.9994, 0.0000, 0.0000], `ranking` [0, 1, 3, 2], `n_tokens` [85, 104, 108, 81], no truncation. Transformers logs one informational line about calling `pad` after `__call__` on a fast tokenizer; that is the README's own two-step encoding and is expected.
- Not executed: the CUDA/bfloat16 path, the `allow_download=True` Hub path, prompts near the 8,192-token ceiling against the real model, and any labelled ranking evaluation.

## References

- Zhang, Y. et al. (2025). Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. https://arxiv.org/abs/2506.05176
- Upstream model card: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B (pinned README, revision above)
- Upstream code: https://github.com/QwenLM/Qwen3-Embedding
- Sibling embedder: https://github.com/kurtvalcorza/qwen3-embedding-pipeline
