# Qwen3 Reranker Pipeline

DIMER-oriented inference wrapper for **Qwen3-Reranker-0.6B**, pinned to an immutable Hugging Face revision. The repository exposes pointwise query-document relevance scoring with the upstream README's exact prompt and `yes`/`no` logit-softmax contract, a supply-chain check of the local weight snapshot, and machine-readable provenance.

## Upstream alignment

- Model: `Qwen/Qwen3-Reranker-0.6B`
- Revision: `e61197ed45024b0ed8a2d74b80b4d909f1255473`
- Upstream weight license: Apache-2.0
- Upstream task: text reranking (second-stage relevance scoring of retrieved candidates)
- Repository adaptation: **none**; inference only

## Quick start

```python
from qwen3_reranker_pipeline import Qwen3RerankerPipeline

pipe = Qwen3RerankerPipeline.from_pretrained()          # verifies weights/qwen3-reranker-0.6b first
pairs = [
    ("What is the capital of China?", "The capital of China is Beijing."),
    ("What is the capital of China?", "Gravity is a force that attracts two bodies."),
]
result = pipe.rerank(pairs)
print(result["scores"], result["ranking"])   # [~0.9995, ~0.0], [0, 1]
```

`rerank()` takes 1..32 `(query, document)` pairs (`MAX_PAIRS`) and an optional `instruction` (default from the upstream README). Each pair is wrapped in the fixed system/user/assistant prompt; the score is the softmax share of the `yes` logit against the `no` logit at the last position — a relevance score, not a calibrated probability, and no threshold is applied. Prompts beyond 8,192 tokens (`MAX_TEXT_TOKENS`) are truncated longest-first and flagged per pair in `truncated`; texts above 100,000 characters (`MAX_TEXT_CHARS`) are rejected. Every result carries `ranking`, `score_kind`, `n_tokens`, `model_id` and `model_revision`.

## Weights layout

```
weights/qwen3-reranker-0.6b/
  config.json  1_LogitScore/config.json  chat_template.jinja  model.safetensors  tokenizer.json
  tokenizer_config.json  vocab.json  merges.txt  generation_config.json  modules.json
  sentence_bert_config.json  config_sentence_transformers.json  README.md  dimer-base-manifest.json
```

`from_pretrained()` calls `stage_missing_files()` (fetches absent manifest entries at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (size + SHA-256 of every entry), loads with `local_files_only=True`, and raises if the tokenizer's `yes`/`no` ids differ from the pinned 9693/2152. Without a manifest it raises unless `allow_download=True`. See `docs/WEIGHTS.md`.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: they use an injected fake runner and temporary manifests, never the weights.

## Release status

**Candidate / source-complete** (`STATUS.md`). Card pass only; no tutorial notebook yet.

## Licensing

This repository's code is Apache-2.0 (`LICENSE`). The packaged upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.
