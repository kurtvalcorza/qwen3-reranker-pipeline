"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "qwen3_reranker_pipeline",
    "repo_name": "qwen3-reranker-pipeline",
    "stem": "qwen3_reranker",
    "notebook_name": "qwen3_reranker_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "Qwen3RerankerPipeline",
    "weights_key": "qwen3-reranker-0.6b",
    "runtime_imports": ["torch", "transformers"],
    "title": "Qwen3-Reranker-0.6B — DIMER query-document reranking tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/qwen3-reranker-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/qwen3-reranker-pipeline/blob/main/tutorials/qwen3_reranker_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen%2FQwen3--Reranker--0.6B-ffcc4d?style=flat",
            "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-QwenLM%2FQwen3--Embedding-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/QwenLM/Qwen3-Embedding",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2506.05176-b31b1b.svg", "https://arxiv.org/abs/2506.05176"),
    ],
    "capability": "pointwise query-document relevance reranking using the pinned Qwen3-Reranker-0.6B weights",
    "intro": (
        "At inference each `(query, document)` pair is wrapped in the fixed upstream system/user/assistant prompt "
        "together with an instruction, one forward pass of the causal language model reads the last-position logits of "
        "the `yes` and `no` tokens, and a two-way softmax turns them into a **relevance score** in [0, 1]: the `yes` "
        "share. **The score is not a calibrated probability**, no threshold is shipped, and the `ranking` the pipeline "
        "returns is an ordering of the supplied pairs, not an acceptance decision — the caller owns any cut-off. **No "
        "adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting — the pinned "
        "checkpoint is used as published. What the upstream checkpoint supplies is the model, tokenizer and prompt "
        "convention; what the carried pipeline module adds is manifest verification, input validation and ceilings, "
        "prompt assembly, the two-logit read-out, a fixed output contract and the `validate_inputs` and "
        "`evaluation_report` stage helpers. Free-text generation is deliberately not reachable through this package."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, author a synthetic query and "
        "candidate set (or upload your own), surface the pipeline's ceilings, stage and digest-verify the immutable "
        "upstream snapshot, validate the pairs into an input manifest, rerank through the public API, read scores and "
        "ranking correctly, understand from the machine-readable evaluation report why no metric is reported and what "
        "labelled judgements a real evaluation needs, and export the ranking with identifiers plus provenance."
    ),
    "exclusions": (
        "embedding or retrieval over a corpus (the Qwen3-Embedding sibling covers first-stage retrieval), text "
        "generation, listwise or pairwise comparison between documents, multilingual quality claims, or any calibrated "
        "relevance threshold. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (bfloat16 there, the checkpoint's native dtype); the model card's CPU smoke scored four short pairs in about 1 s after a 5 s load, so the three-pair default runs in seconds on a hosted CPU runtime. The pinned `torch==2.14.0` install and the 1.19 GB checkpoint are the largest downloads of the run.",
        "- **Knowledge:** basic Python; what a softmax over two logits is and why it is not a calibrated probability; what a relevance judgement is.",
        "- **Data:** the default sample is a synthetic query and three candidate passages authored in code, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 text file whose first non-empty line is the query and whose remaining non-empty lines are candidate documents (one per line, at most 32 documents, each at most 100,000 characters). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded text remains in the notebook runtime; this pipeline does not send it to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Author the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**: one query and three candidate passages written in this cell — one "
                "that answers the query, one on the same topic that does not answer it, and one unrelated — so it needs no "
                "download and contains no personal data. It ships **no relevance judgements** beyond the author's intent, "
                "so the scores it produces are smoke/sanity evidence that the code path works (the answering passage is "
                "expected to outrank the unrelated one), never a retrieval-quality measurement and never benchmark "
                "evidence.\n\n"
                "BYOD is optional and disabled by default. Expected BYOD input: one UTF-8 text file whose first non-empty "
                "line is the query and whose remaining non-empty lines are candidate documents; every document is paired "
                "with the query. The upload stays inside this runtime. If you also hold relevance judgements for your "
                "candidates, keep them outside the notebook — Section 7 explains what to compute with them."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    lines = [line.strip() for line in io.StringIO(uploaded[sample_name].decode('utf-8')) if line.strip()]\n"
                "    if len(lines) < 2:\n"
                "        raise ValueError(f'{{sample_name}}: expected a query line followed by at least one document line')\n"
                "    query, documents = lines[0], lines[1:]\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    query = 'What is the capital of China?'\n"
                "    documents = [\n"
                "        'The capital of China is Beijing, which has been the seat of government since 1949.',\n"
                "        'Shanghai is the largest city in China by population and a major financial centre.',\n"
                "        'Gravity is the force by which a planet or other body draws objects toward its centre.',\n"
                "    ]\n"
                "    sample_name = 'synthetic_capital_query'\n"
                "    sample_kind = 'synthetic (authored in this cell)'\n"
                "doc_ids = [f'doc{{index:02d}}' for index in range(len(documents))]\n"
                "pairs = [(query, document) for document in documents]\n"
                "sample_sha256 = hashlib.sha256('\\n'.join([query, *documents]).encode('utf-8')).hexdigest()\n"
                "print({{'sample': sample_name, 'sample_kind': sample_kind, 'query': query, 'documents': len(documents), 'text_sha256': sample_sha256}})\n"
                "for doc_id, document in zip(doc_ids, documents, strict=True):\n"
                "    print(f'{{doc_id}}: {{document[:100]}}')"
            ),
        },
        {
            "md": (
                "## 5. Validate the pairs → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `rerank` "
                "applies — both route through the same private `_check_inputs` — so the pair count 1..`MAX_PAIRS`, the "
                "pair shape, non-empty query and document, the character ceiling `MAX_TEXT_CHARS` and a non-empty "
                "instruction are enforced identically. It returns an **input manifest** naming the schema and ceilings, "
                "each pair's identifier and character counts, the instruction in force, and the verdict; the manifest is "
                "written to `outputs/{stem}_input_manifest.json`. The ceilings are surfaced before any model work: "
                "`MAX_PAIRS` (pairs per `rerank` call), `MAX_TEXT_CHARS` (characters per query or document, checked before "
                "tokenisation) and `MAX_TEXT_TOKENS` (total prompt tokens including the fixed prefix/suffix; longer "
                "prompts are truncated longest-first and flagged per pair in `truncated`). The default `instruction` is "
                "printed because it is part of the prompt and changes the scores, so a deployment must fix it "
                "deliberately. To show what rejection looks like, the cell also validates a pair with an empty document "
                "and records the pipeline's own error message as a finding. The notebook never trims or alters the texts; "
                "any token-level truncation happens inside the pipeline and is reported after the call."
            ),
            "code": (
                "import json\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "ceilings = {{'MAX_PAIRS': MAX_PAIRS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS}}\n"
                "instruction = DEFAULT_INSTRUCTION\n"
                "print(ceilings)\n"
                "print({{'instruction': instruction}})\n"
                "input_manifest = validate_inputs(pairs, instruction, names=doc_ids)\n"
                "# Demonstrate rejection on an input that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs([(query, '   ')], instruction)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'empty-document-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Rerank and interpret the scores\n\n"
                "`rerank(pairs, instruction=…)` returns `scores` (one float in [0, 1] per pair, **aligned with the input "
                "order**), `ranking` (pair indices sorted by descending score, stable), `score_kind`, the `instruction` "
                "used, `n_tokens` per prompt, `truncated` flags, and the model identity. **Score semantics:** each score "
                "is the softmax share of the `yes` logit against the `no` logit — a relevance score that orders candidates "
                "for one query; it is not a calibrated probability that the document is relevant, scores for different "
                "queries are not comparable as absolute values, and the pipeline ships no threshold. Any accept/reject "
                "cut-off (for example \"show only candidates above 0.5\") is owned by the caller and must be set on their "
                "own labelled pairs. Scores differ slightly between the CPU float32 and CUDA bfloat16 paths and can "
                "reorder near-tied candidates. The runtime figure is measured on the runtime identified in Section 1 for "
                "this batch and includes the first-call warm-up."
            ),
            "code": (
                "import time\n\n"
                "started = time.perf_counter()\n"
                "result = pipe.rerank(pairs, instruction=instruction)\n"
                "elapsed = time.perf_counter() - started\n"
                "scores = result['scores']\n"
                "checks = {{\n"
                "    'one_score_per_pair': len(scores) == len(pairs),\n"
                "    'scores_in_unit_interval': all(0.0 <= s <= 1.0 for s in scores),\n"
                "    'ranking_is_permutation': sorted(result['ranking']) == list(range(len(pairs))),\n"
                "    'nothing_truncated': not any(result['truncated']),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'rerank output failed a sanity check: {{checks}}')\n"
                "print({{key: value for key, value in result.items() if key not in ('scores', 'ranking')}})\n"
                "print({{'seconds': round(elapsed, 3), 'checks': checks}})\n"
                "print(f'query: {{query}}')\n"
                "for rank, index in enumerate(result['ranking'], start=1):\n"
                "    print(f\"{{rank:>2}}. {{doc_ids[index]}}  score {{scores[index]:.4f}}  tokens {{result['n_tokens'][index]:>4}}  {{documents[index][:80]}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report — even, as "
                "here, when nothing is measurable. The repository ships **no metric helper and reports no performance "
                "measure**, so the verdict is always `not-measurable` and the report states what would make the task "
                "measurable: per-query relevance judgements (binary or graded) over enough queries to state a dispersion, "
                "scored with the caller's own nDCG@k, MRR or precision@k code. Supplying judgements does not change the "
                "verdict, because there is no metric helper to score them with; the helper records that fact in `reason` "
                "rather than inventing a number, and the upstream benchmark figures quoted in the model card remain "
                "upstream claims, not measurements made here. On the synthetic default sample the cell also prints one "
                "falsifiable plumbing check — the answering passage should outrank the unrelated one — which is a check on "
                "one query, not a retrieval result. The report is written to "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "sanity = {{}}\n"
                "if not USE_BYOD:\n"
                "    sanity = {{'answering_outranks_unrelated': scores[0] > scores[2]}}\n"
                "    print({{'sanity_check': sanity, 'note': 'falsifiable plumbing check on one synthetic query; not a metric'}})\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric is reported: the sample has no relevance judgements and the repository ships no metric helper; compute nDCG/MRR on your own judged pairs.')"
            ),
        },
        {
            "md": (
                "## 8. Export the ranking and provenance\n\n"
                "The ranking is written as CSV (`outputs/{stem}_ranking.csv`) with explicit `rank`, `id`, `score`, "
                "`n_tokens`, `truncated` and `document` columns, so the ordering survives downstream use. Machine-readable "
                "JSON preserves an `items` list with, per pair, its identifier, document text, score, rank, token count "
                "and truncation flag (so every score maps back to its input), the query, the instruction, the `ranking`, "
                "the `score_kind`, the sanity checks, the ceilings in force, the input manifest, the evaluation report, "
                "the sample identity and digest, the notebook's source (repository, revision, embedded module digest, "
                "generator), the model identifier, the immutable model revision, the model licence, the verified snapshot "
                "summary, and the runtime identity (Python, `torch`, `transformers`, device, dtype). No credentials are "
                "involved in any step, so none can reach the export."
            ),
            "code": (
                "import csv\n\n"
                "rank_of = {{index: rank for rank, index in enumerate(result['ranking'], start=1)}}\n"
                "with open('outputs/{stem}_ranking.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['rank', 'id', 'score', 'n_tokens', 'truncated', 'document'])\n"
                "    for index in result['ranking']:\n"
                "        writer.writerow([rank_of[index], doc_ids[index], f'{{scores[index]:.6f}}', result['n_tokens'][index], result['truncated'][index], documents[index]])\n"
                "payload = {{\n"
                "    'query': query,\n"
                "    'instruction': result['instruction'],\n"
                "    'items': [\n"
                "        {{'id': doc_ids[index], 'document': documents[index], 'score': scores[index], 'rank': rank_of[index], 'n_tokens': result['n_tokens'][index], 'truncated': result['truncated'][index]}}\n"
                "        for index in range(len(pairs))\n"
                "    ],\n"
                "    'ranking': [doc_ids[index] for index in result['ranking']],\n"
                "    'score_kind': result['score_kind'],\n"
                "    'sanity_checks': checks,\n"
                "    'plumbing_check': sanity,\n"
                "    'ceilings': ceilings,\n"
                "    'ranking_file': 'outputs/{stem}_ranking.csv',\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sample': {{'name': sample_name, 'kind': sample_kind, 'text_sha256': sample_sha256}},\n"
                "    'seconds': round(elapsed, 3),\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes')}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': 'bfloat16' if pipe.device.startswith('cuda') else 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Each score is the `yes` share of a two-way softmax over the reranker's last-position logits: a relevance score "
        "that orders candidates for one query, not a calibrated probability, not comparable as an absolute value across "
        "queries or instructions, and never converted to a decision by the pipeline — the caller owns any threshold and "
        "must set it on their own judged pairs. On the synthetic sample the ordering is plumbing evidence only; the "
        "evaluation report is `not-measurable` because no metric can be computed without relevance judgements, and a "
        "real evaluation needs judged candidates for many queries and the caller's own nDCG/MRR code. The instruction is "
        "part of the prompt and changes the scores; prompts beyond `MAX_TEXT_TOKENS` are truncated longest-first and "
        "flagged; the pipeline exposes no generation, no listwise comparison, and no corpus retrieval. The forward pass "
        "is deterministic on a fixed device and dtype, but CPU (float32) and CUDA (bfloat16) scores differ slightly and "
        "can reorder near-tied candidates.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model snapshot, validate the demonstrated pairs against the enforced "
        "ceilings, execute the public pipeline path, and emit the shown machine-readable outputs in the tested runtime — "
        "without the repository being reachable. It does **not** establish benchmark superiority, retrieval quality on "
        "any domain, a usable relevance threshold, safety for high-consequence decisions, or production fitness on an "
        "unseen domain.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: the "
        "pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the top. "
        "`FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/{MODEL_KEY}/` and rerun Section 3. A `ValueError` naming "
        "`MAX_PAIRS` or `MAX_TEXT_CHARS` in Section 5: reduce or shorten the BYOD lines and rerun from Section 4. A "
        "`truncated` flag set to `True` in Section 6: that prompt exceeded `MAX_TEXT_TOKENS` and was cut longest-first — "
        "shorten the document if the cut matters.\n\n"
        "**Next experiments.** Upload a query with a dozen candidates you can judge yourself and compare the pipeline's "
        "ranking with your judgements — that is exactly the labelled data the evaluation report asks for; change "
        "`instruction` in Section 5 to a task-specific one and observe how the scores move; run the same batch on a CUDA "
        "runtime and compare the bfloat16 scores with the CPU float32 ones. None of these turns the sample result into "
        "evidence of production fitness.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/qwen3-reranker-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/qwen3-reranker-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/qwen3-reranker-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/QwenLM/Qwen3-Embedding\n"
        "- Qwen3 Embedding technical report: https://arxiv.org/abs/2506.05176"
    ),
}
