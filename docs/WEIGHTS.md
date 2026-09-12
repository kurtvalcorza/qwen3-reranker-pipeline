# Weight provenance and DIMER hosting

- Upstream: `Qwen/Qwen3-Reranker-0.6B`
- Immutable revision: `e61197ed45024b0ed8a2d74b80b4d909f1255473`
- Weight format: SafeTensors (`model.safetensors`, 1,191,588,280 bytes, bfloat16)
- Upstream weight license: Apache-2.0 (`license: apache-2.0` in the pinned upstream README)
- Local layout: `weights/qwen3-reranker-0.6b/` holds the 13 files listed in `dimer-base-manifest.json` (`config.json`, `1_LogitScore/config.json` with the yes/no token ids, tokenizer files, `chat_template.jinja`, `model.safetensors`, upstream `README.md`, sentence-transformers configs) with byte size and SHA-256 for each. `verify_snapshot()` in `src/qwen3_reranker_pipeline/pipeline.py` checks all of them before any load; `stage_missing_files(allow_download=True)` fetches only absent entries at the pinned revision into that directory. `.safetensors` files are git-ignored; the Git repository does not vendor the checkpoint.
- DIMER hosting: Apache-2.0 permits use, modification, redistribution and commercial use subject to preservation of the license and notices. DIMER may mirror the pinned checkpoint in its model store under those terms.
- Loader trust boundary: Transformers `AutoModelForCausalLM` (`Qwen3ForCausalLM`, used only to read two logits) + `AutoTokenizer` with `trust_remote_code=False`; the loader reads only the verified local directory (`local_files_only=True`) and falls back to the Hub at the pinned revision only when `allow_download=True` is passed explicitly.
