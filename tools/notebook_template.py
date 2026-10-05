"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E reranker fine-tuning workflow: the pinned Qwen3-Reranker-0.6B snapshot is
digest-verified and loaded, a digest-pinned real corpus (Banking77 messages with seeded intent-phrase
shortlists) is fetched, validated and split, the reranking contract is exercised on one shortlist, the frozen
model's ranking metrics on held-out shortlists are read beside a random floor and a lexical baseline, a bounded
listwise fine-tuning of the last decoder layers adapts the reranker in the kernel, the held-out split is scored
again, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "qwen3_reranker_pipeline",
    "repo_name": "qwen3-reranker-pipeline",
    "stem": "qwen3_reranker",
    "notebook_name": "qwen3_reranker_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # SWP-R (2026-10-05 fleet sweep): nothing is pip-installed into the notebook kernel. The fleet's uv isolated-environment
    # mechanism (build_notebook.py/2.2): managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the
    # pyproject pins with `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28
    # --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt` (uv 0.12.15).
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
        "pinned Qwen3-Reranker-0.6B snapshot (safetensors, 1.19 GB), fetches the two digest-pinned Banking77 CSV files from "
        "the project repository (1.1 MB, no credential), pairs every customer message with its intent phrase and a seeded "
        "shortlist of five other intent phrases and draws 231 / 77 / 154 training, validation and test records balanced "
        "over the 77 intents from the release's own partition, reranks one test shortlist through the inference contract "
        "with an input manifest and a rejection probe, scores the frozen reranker on the test shortlists by recall@1 / "
        "recall@3 / recall@5 and MRR beside the random floor and a lexical (token-overlap) baseline, runs a bounded "
        "listwise fine-tuning of the last two decoder layers with validation-MRR epoch selection, scores the "
        "held-out split again, reranks the same shortlists with the adapted model, exports the adapter as safetensors with "
        "a manifest, and reloads that artifact into a fresh pipeline to verify parity. The default path needs no "
        "repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about fifteen minutes of model time after the downloads — "
        "a cross-encoder scores every query–candidate pair with a full forward pass, so evaluation dominates; a CUDA "
        "runtime is used automatically when present (bfloat16 there, float32 on CPU)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "records — as `BYOD_PATH` (a path in the runtime, which works on Colab, Kaggle and Jupyter) or, when it is empty, through the "
        "Colab upload dialog — as a CSV (columns `id`, `query`, `positive`, `negatives` with the negatives separated by ` | `), a JSON "
        "array or a JSONL file of `{id, query, positive, negatives}` records; set `INSTRUCTION` to a one-line description "
        "of your relevance judgement. They pass through the same validation, seeded query-disjoint split, floor and "
        "baseline, frozen scoring, fine-tuning, held-out evaluation, reranking, artifact export and reload-parity cells as "
        "the Banking77 sample. The expected schema and the ceilings are stated in the Prerequisites and in Section 4, and "
        "uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "Qwen3RerankerPipeline",
    "guided": {
        "opening": [
            '**Who this notebook is for.** The intended audience is a learner who knows basic Python, has used Colab or Jupyter, and wants to see how a '
            'cross-encoder reranker orders a shortlist of candidate documents for a query, how ranking quality is measured against honest floors (including a '
            'lexical baseline the hard negatives were chosen to defeat), and what a bounded listwise fine-tuning changes. No prior experience with Qwen3, rerankers '
            'or ranking losses is assumed; terms are explained where they first matter and again in the **Glossary** at the end. CPU works but is slow (about '
            'fifteen minutes of model time, because a cross-encoder runs one forward pass per pair); a GPU runtime is much faster.\n\n**Input → Model → Output.**\n\n| '
            '| Reranking | Bounded listwise fine-tuning |\n|---|---|---|\n| Input | (query, document) pairs with a one-line relevance instruction | Banking77 '
            'messages, each with its intent phrase and five shortlisted negatives (231 / 77 / 154, query-disjoint) |\n| Model | the Qwen3-Reranker-0.6B decoder, '
            'reading the `yes` / `no` logits at the last position | the same model; only the last two decoder layers train (31.5 M of 595.8 M parameters) with a '
            'listwise cross-entropy |\n| Output | one relevance score per pair (the `yes` share — not a calibrated probability) and a ranking | recall@1/3/5 and MRR '
            'beside a random floor, a lexical baseline and the frozen model, and a 126 MB safetensors adapter that reloads with parity |\n\n**How to use this '
            'notebook.** Choose a runtime (**Runtime → Change runtime type → T4 GPU** is strongly recommended), then **Runtime → Run all**. Run all completes in '
            "one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated "
            'environment, the carried package and the model snapshot — and their cells are collapsed; you may run them without studying them. The learning path '
            'starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each '
            'principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked '
            "answer from the recorded run (the Kaggle Tesla T4 run of 19 September 2026 recorded in `docs/release-verification.md`, bfloat16 on CUDA; the prose's "
            'build-record numbers are the float32 CPU run, so the two differ in the last digits). Every adaptation starts from the pinned base, so re-running '
            'Section 7 with other settings is a fresh experiment. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your '
            'predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the Banking77 corpus, shortlists, validation and the split *(evaluation practice)* '
            '→ 5 the reranking contract *(core concept: relevance scores, not probabilities)* → 6 the random floor, the lexical baseline and the frozen model '
            '*(evaluation practice)* → 7 bounded listwise fine-tuning *(core concept)* → 8 held-out evaluation *(evaluation practice)* → 9 reranking before and '
            'after, export and fresh reload *(engineering)* → interpretation, troubleshooting, glossary and your conclusion.'
        ],
    },
    "weights_key": "qwen3-reranker-0.6b",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "transformers"],
    "title": "Qwen3-Reranker-0.6B — DIMER E2E reranker fine-tuning tutorial: intent shortlists on Banking77 (standalone)",
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
    "capability": "pointwise query-document relevance reranking and bounded listwise fine-tuning of the last decoder layers on query / positive / negatives records, measured by held-out ranking recall@k and MRR, using the pinned Qwen3-Reranker-0.6B weights",
    "intro": (
        "At inference each `(query, document)` pair is wrapped in the fixed upstream system/user/assistant prompt "
        "together with an instruction, one forward pass of the causal language model reads the last-position logits of "
        "the `yes` and `no` tokens, and a two-way softmax turns them into a **relevance score** in [0, 1]: the `yes` "
        "share. **The score is not a calibrated probability**, no threshold is shipped, and the `ranking` the pipeline "
        "returns is an ordering of the supplied pairs, not an acceptance decision — the caller owns any cut-off. What the "
        "upstream checkpoint supplies is the model, tokenizer and prompt convention; what the carried pipeline module adds "
        "is manifest verification, input validation and ceilings, prompt assembly, the two-logit read-out, a fixed output "
        "contract and the `validate_inputs` and `evaluation_report` stage helpers. Free-text generation is deliberately "
        "not reachable through this package.\n\n"
        "What this notebook adds to inference is **adaptation measured by ranking**. The dataset is real: Banking77 "
        "(Casanueva et al., 2020; CC BY 4.0) ships 13,083 customer-support messages labelled with 77 fine-grained banking "
        "intents as two digest-pinned CSV files fetched from the project repository at a pinned commit. Every intent name "
        "becomes a short **document** (`card_arrival` → `card arrival`), every message a **query** whose positive is its "
        "intent phrase, and every query carries a seeded **shortlist** of five other intent phrases — the three with the "
        "highest token overlap with the message plus two random ones — so reranking the six-entry shortlist is intent "
        "detection over hard candidates, a task the reranker was never tuned for. The carried `metrics.py` orders each "
        "shortlist by the relevance score and reads the rank of the positive: **recall@1**, **recall@3**, **recall@5** "
        "and **MRR**, ties counted against the positive; a **random floor** (1 / 6 recall@1) and a **lexical baseline** "
        "(Jaccard token overlap between message and phrase — the very signal the hard negatives were chosen by) frame the "
        "frozen number. The fine-tuning question is whether a bounded listwise adaptation of the last decoder layers on "
        "231 shortlists raises ranking on messages the model has not seen. Nothing here is a quality claim about your "
        "reranking task: it is one seeded split of one corpus."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned real corpus and validate and split it "
        "without leakage; rerank a shortlist through the public API with the instruction contract and read the score and "
        "ranking contract correctly; read recall@k and MRR beside a random floor and a lexical baseline and understand "
        "what they do and do not measure; run a bounded listwise fine-tuning with explicit hyperparameters and "
        "validation-based epoch selection; evaluate on an independent test split; compare rankings before and after; and "
        "export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "embedding or first-stage retrieval (see the sibling Qwen3 embedding pipeline), text generation or chat, "
        "calibrated probabilities or a relevance threshold, graded (non-binary) relevance, hard-negative mining beyond the "
        "seeded shortlists of the dataset contract, full-model or embedding-table training, and any claim that a "
        "Banking77 intent shortlist stands in for your reranking task. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available. **Precision differs by device:** the pipeline runs float32 on CPU and bfloat16 on CUDA, so scores and the recorded metrics can differ between the two. CPU is adequate but not fast for a cross-encoder: the build record measured about 5 s to load and digest-verify the 1.19 GB snapshot, about 132 s to score the 154 test shortlists (924 pairs) and about 170 s per training epoch over 231 shortlists plus a validation pass per epoch. Building the isolated environment (the pinned `torch==2.14.0` among its packages; reused on a re-run) and the 1.19 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python; what a cross-encoder does and why it scores one pair per forward pass; what recall@k and mean reciprocal rank measure over a candidate list; what a listwise softmax loss does.",
        "- **Data contract:** records are `{id, query, positive, negatives}` — a query of 1..100,000 characters, a positive document of 1..1,000 characters and 1..15 distinct negative documents (the pair prompt is truncated longest-first to 8,192 tokens at inference and to 192 tokens **during training only**), ids matching `[A-Za-z0-9_.:-]{1,64}` and unique; a dataset needs 8..20,000 records; queries are de-duplicated case-insensitively before splitting so the same message never sits in two splits. BYOD accepts CSV (negatives separated by ` | `), JSON or JSONL in that shape.",
        "- **Validation is structural, not semantic:** nothing checks that a positive is relevant to its query, that the negatives are not, or that the instruction describes the judgement — a mislabelled record set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — an internal query log with its relevance labels is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches two pinned objects (`train.csv` 839,073 bytes, `test.csv` 239,961 bytes; SHA-256 `b06e26ac…` / `d12d6e3b…`) from `raw.githubusercontent.com` at the pinned `PolyAI-LDN/task-specific-datasets` commit over HTTPS, each refused on any mismatch before it is read; Banking77 is CC BY 4.0 (Casanueva et al., 2020).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Referenced corpus, validation and split\n\n"
                "`fetch_corpus` downloads the two pinned Banking77 CSV files (or reads them from the cache), refuses a "
                "byte-size or SHA-256 mismatch per file before it is parsed, and `read_corpus` checks the columns, row "
                "counts and the 77 intents of each member. `build_sample_dataset` turns every message into a "
                "`{{id, query, positive, negatives}}` record whose positive is the intent name as words and whose negatives "
                "are a seeded shortlist (`sample_negatives`: the three other phrases with the highest token overlap with the "
                "message, then two seeded random ones — the seed is a function of the record id, so the shortlist is "
                "reproducible), drops repeated messages, and draws 3 training and 1 validation record per intent from the "
                "`train` member (disjoint messages) and 2 test records per intent from the `test` member by a seeded "
                "shuffle — the release's own partition, balanced over all 77 intents. `validate_dataset` then checks every "
                "record against the contract, `check_split_disjoint` asserts no message appears in two splits, and the "
                "training split is written to `outputs/{stem}_train.csv` in the shape BYOD expects. `INSTRUCTION` is "
                "the judgement every pair carries in later cells.\n\n"
                "Look for: 10,003 + 3,080 raw rows, two digests, splits 231 / 77 / 154, six candidates per query, and "
                "four refusal probes — a duplicate id, an empty query, a negative equal to the positive and a dataset too "
                "small to split — each rejected before `torch` does anything.\n\n"
                "*Evaluation practice.* **Predict before running:** three of the five negatives are the phrases with the most words in "
                "common with the message. Who does that make life hard for — the lexical baseline or the reranker?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n"
                "INSTRUCTION = 'Given a customer support message, judge whether the document names the banking intent it expresses'  # @param {{type:\"string\"}}\n"
                "# A file already in the runtime (works on Colab, Kaggle and Jupyter); empty = the Colab upload dialog.\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_path = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_path.is_file():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} is not a file (relative paths start at {{Path.cwd()}}): give one .csv, .json, .jsonl file holding id, query, positive and negatives.')\n"
                "        file_name = byod_path.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the file in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith(('.csv', '.json', '.jsonl')):\n"
                "            raise ValueError(f'{{file_name}}: upload one .csv, .json, .jsonl file holding id, query, positive and negatives.')\n"
                "        byod_path = Path('work') / file_name\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/banking77'))\n"
                "    raw_rows = {{name: len(part) for name, part in corpus.items()}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'instruction': INSTRUCTION, 'raw_rows': raw_rows, 'splits': disjoint, 'n_documents': len(documents(test_records)), 'file_sha256': {{k: v[2][:12] + '...' for k, v in CORPUS_FILES.items()}}}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_queries': manifest['unique_queries'], 'candidates': manifest['candidates'], 'query_chars': manifest['query_chars'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "print({{'example': {{k: train_records[0][k] for k in ('id', 'query', 'positive', 'negatives')}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'empty query': [{{**train_records[0], 'query': '   '}}, *train_records[1:8]],\n"
                "    'negative equals positive': [{{**train_records[0], 'negatives': [train_records[0]['positive']]}}, *train_records[1:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                '**What to notice:** 10,003 + 3,080 raw rows, splits 231 / 77 / 154, six candidates per query, and the four refusals.\n\n<details><summary>Check your '
                'reasoning</summary>Mostly the lexical baseline: word-overlap negatives are exactly what a word counter confuses with the answer, so the floor is '
                'deliberately hard to beat by words alone, and a reranker has to judge meaning. The shortlist is seeded per record id, so it reproduces; the splits are '
                'query-disjoint. The refusals (duplicate id, empty query, a negative equal to the positive, too small) stop before `torch` runs.</details>'
            ),
        },
        {
            "md": (
                "## 5. Rerank one shortlist through the inference contract\n\n"
                "Before any adaptation, the reranking contract is exercised as it always was, on the first test record's "
                "shortlist. `validate_inputs` applies exactly the checks `rerank` applies — both route through the same "
                "private `_check_inputs` — so pair shape, batch size 1..`MAX_PAIRS`, non-empty texts, the character "
                "ceiling and a non-empty instruction are enforced identically; it returns an input manifest, and an "
                "empty-document pair is validated too and its rejection recorded as a finding. `rerank` returns one "
                "`score` per pair (the `yes` share of a two-way softmax — a **relevance score, not a calibrated "
                "probability**), the `ranking` as a permutation of the pairs, `n_tokens` and `truncated` flags. The "
                "ranked shortlist is printed with the gold intent marked; the frozen rankings of three test shortlists are "
                "kept as the *before* column for Section 9.\n\n"
                "**Predict before running:** will the six scores of one shortlist add up to 1?"
            ),
            "code": (
                "import time\n\n"
                "probe_records = test_records[:3]\n"
                "record = probe_records[0]\n"
                "candidates = candidate_list(record)\n"
                "pairs = [(record['query'], doc) for doc in candidates]\n"
                "print({{'ceilings': {{'MAX_PAIRS': MAX_PAIRS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MAX_TRAIN_TOKENS': MAX_TRAIN_TOKENS, 'MAX_TRAIN_CANDIDATES': MAX_TRAIN_CANDIDATES}}, 'yes_no_token_ids': (YES_TOKEN_ID, NO_TOKEN_ID)}})\n"
                "input_manifest = validate_inputs(pairs, INSTRUCTION, names=[f'{{record[\"id\"]}}-cand-{{i}}' for i in range(len(pairs))])\n"
                "try:\n"
                "    validate_inputs([(record['query'], '   ')], INSTRUCTION)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'empty-document-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "started = time.perf_counter()\n"
                "result = pipe.rerank(pairs, instruction=INSTRUCTION)\n"
                "rerank_seconds = round(time.perf_counter() - started, 3)\n"
                "scores = result['scores']\n"
                "checks = {{\n"
                "    'one_score_per_pair': len(scores) == len(pairs),\n"
                "    'scores_in_unit_interval': all(0.0 <= s <= 1.0 for s in scores),\n"
                "    'ranking_is_permutation': sorted(result['ranking']) == list(range(len(pairs))),\n"
                "    'nothing_truncated': not any(result['truncated']),\n"
                "    'instruction_echoed': result['instruction'] == INSTRUCTION and result['score_kind'] == SCORE_KIND,\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'rerank output failed a sanity check: {{checks}}')\n"
                "print({{'query': record['query'], 'gold': record['positive'], 'seconds': rerank_seconds, 'n_tokens': result['n_tokens'], 'checks': checks, 'findings': len(input_manifest['findings']), 'score_semantics': result['score_kind'] + '; no threshold shipped'}})\n"
                "for rank, index in enumerate(result['ranking'], start=1):\n"
                "    print(f\"{{rank}}. {{'*' if index == 0 else ' '}} {{candidates[index]:<40}} score {{scores[index]:.4f}}\")\n\n"
                "def ranked_shortlist(rec):\n"
                "    docs = candidate_list(rec)\n"
                "    out = pipe.rerank([(rec['query'], d) for d in docs], instruction=INSTRUCTION)\n"
                "    return [(docs[i], round(out['scores'][i], 4)) for i in out['ranking']]\n\n"
                "before = {{r['id']: ranked_shortlist(r) for r in probe_records}}\n"
                "print({{'shortlists_ranked_by_the_frozen_model': len(before)}})"
            ),
        },
        {
            "md": (
                '**What to notice:** one score per pair in [0, 1], `ranking` as a permutation, `checks`, the empty-document finding, and where the gold intent (`*`) lands.\n\n<details><summary>Check '
                'your reasoning</summary>No. Each score is the `yes` share of a two-way softmax for that pair alone, so the six are independent and need not sum to '
                'anything; only their order matters. A score of 0.9 is a relevance score, not a 90 % probability, and no threshold ships.</details>'
            ),
        },
        {
            "md": (
                "## 6. The random floor, the lexical baseline and the frozen model on the test split\n\n"
                "Three numbers frame the adaptation, all over the same 154 six-entry shortlists. The **random floor** is "
                "what a uniformly random ordering achieves in expectation (recall@1 = 1 / 6, MRR ≈ 0.41). The **lexical "
                "baseline** orders each shortlist by Jaccard token overlap with the message — and because three of the "
                "five negatives were chosen for high overlap, this floor is deliberately hard to beat by words alone. "
                "`pipe.evaluate` scores every query–candidate pair through `rerank` (`MAX_PAIRS` at a time), orders each "
                "shortlist by the relevance score and reads the rank of the positive; ties are counted against it. Look "
                "for the frozen reranker well above both — the build record saw recall@1 around 73.4 % and MRR "
                "around 0.843 — and read `median_rank` beside the means. About 132 s on CPU: 924 forward "
                "passes. Whether the frozen model beats the random floor is recorded as a **verdict**, not asserted, so a run on your "
                "own shortlists continues either way.\n\n"
                "*Evaluation practice.* **Predict before running:** on these hard-negative shortlists, will the lexical baseline even beat "
                "the random floor on recall@1?"
            ),
            "code": (
                "def brief(m):\n"
                "    return {{k: round(m[k], 4) for k in ('recall@1', 'recall@3', 'recall@5', 'mrr')}} | {{'median_rank': m.get('median_rank')}}\n\n"
                "floor = random_floor([1 + len(r['negatives']) for r in test_records])\n"
                "print({{'random_floor': {{k: round(floor[k], 4) for k in ('recall@1', 'recall@3', 'recall@5', 'mrr')}}, 'baseline': floor['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "baseline_lexical = pipe.lexical_baseline(test_records)\n"
                "print({{'lexical_baseline': brief(baseline_lexical), 'baseline': baseline_lexical['baseline'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, instruction=INSTRUCTION)\n"
                "print({{'frozen_model_test': brief(frozen_test), 'n_queries': frozen_test['n_queries'], 'n_pairs': frozen_test['n_pairs'], 'verdict': frozen_test['verdict'], 'adapted': frozen_test['adapted'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "# Contract integrity: both systems ranked the same shortlists.\n"
                "if frozen_test['n_queries'] != baseline_lexical['n_queries']:\n"
                "    raise RuntimeError(f\"contract: the frozen model ranked {{frozen_test['n_queries']}} shortlists, the lexical baseline {{baseline_lexical['n_queries']}}\")\n"
                "# SWP-A: the quality comparison is a recorded verdict, not an assert, so a BYOD run still reaches adaptation and export.\n"
                "frozen_vs_floor = 'above' if frozen_test['mrr'] > floor['mrr'] else 'not above'\n"
                "print({{'verdict_frozen_vs_random_floor_mrr': frozen_vs_floor}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the three systems on recall@1/3/5 and MRR, `median_rank`, and the verdict line.\n\n<details><summary>Check your '
                'reasoning</summary>Barely on recall@1, and not on recall@3. In the recorded run recall@1 was 0.1667 for the random floor, 0.2662 for the lexical '
                'baseline and 0.7338 for the frozen reranker; at recall@3 the lexical baseline (0.4156) fell *below* the random floor (0.5), because the overlap '
                'negatives crowd its top three. MRR: 0.4083 / 0.457 / 0.8429. The frozen model judges meaning, not words.</details>'
            ),
        },
        {
            "md": (
                "## 7. Bounded listwise fine-tuning\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_LAYERS` decoder layers — two by default, "
                "31,461,888 of 595,776,512 parameters; the token embeddings (which the output projection shares), the "
                "earlier layers and the final norm stay frozen — with a **listwise** loss: each training query contributes "
                "its positive and its first `TRAIN_CANDIDATES - 1` negatives, formatted exactly as `rerank` formats them, "
                "the relevance logit of every candidate is the `yes`-minus-`no` logit at the last position, and the loss "
                "is the cross-entropy of the positive over its list. `BATCH_SIZE` counts queries per step; AdamW at a "
                "fixed learning rate, gradient clipping at 1.0, seeded shuffling and no scheduler; prompts are truncated to "
                "`MAX_TRAIN_TOKENS` (192) **during training only**. Epoch 0 records the frozen model's validation ranking "
                "metrics; every epoch is scored the same way, and the epoch with the highest validation MRR is kept.\n\n"
                "Watch validation recall@1 climb over two epochs (about 170 s of training plus a validation pass per "
                "epoch on CPU). The build record's sweep on this sample: two layers at 2e-5 reached recall@1 77.9 % (validation MRR still rising at epoch 2); two layers at 5e-5 peaked at epoch 1 and ended at 76.0 % — the default keeps 2e-5. "
                "Every call starts from the pinned base (`started_from` in the printed result), so a re-run with other settings is a "
                "fresh experiment, not continued training, and epoch 0 is always the frozen model.\n\n"
                "**Predict before running:** the frozen reranker already ranks about three in four shortlists correctly. How much room "
                "is left for two epochs on 231 shortlists?"
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 2e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_LAYERS = 2  # @param {{type:\"integer\"}}\n"
                "TRAIN_CANDIDATES = 4  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_recall@1'] = round(entry['val']['recall@1'], 4)\n"
                "        row['val_recall@3'] = round(entry['val']['recall@3'], 4)\n"
                "        row['val_mrr'] = round(entry['val']['mrr'], 4)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, instruction=INSTRUCTION, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_layers=TRAINABLE_LAYERS, train_candidates=TRAIN_CANDIDATES, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'objective': adapt_result['objective'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'train_pairs': adapt_result['n_train_pairs'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'started_from': adapt_result['started_from'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '**What to notice:** epoch 0 (`note: frozen model`), the training loss, `val_recall@1` and `val_mrr` per epoch, `best_epoch`, and `started_from`.\n\n<details><summary>Check '
                "your reasoning</summary>Not much: the gains are a few points. The build record's sweep found 2e-5 still rising at epoch 2 while 5e-5 peaked at epoch 1 "
                'and fell back — a higher learning rate overshoots a model that is already good. Validation MRR, not training loss, picks the kept epoch.</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or epoch selection, and no message in it appears in the "
                "training or validation splits. The adapted reranker is scored exactly as the frozen one was in Section 6 "
                "— same shortlists, same instruction — and the four rows are put side by side. Look for recall@1 up by "
                "several points and MRR up accordingly; the cell records whether the adapted MRR is above the frozen MRR as a "
                "**verdict** (`improved`, `no change` or `worse`) in the report and `result.json` instead of asserting it, so a run on "
                "your own shortlists that does not improve still exports and reloads. 154 "
                "shortlists from one seeded split of one corpus give no dispersion estimate; the deltas are sample-sanity "
                "evidence that the adaptation contract works, not a benchmark, and a gain on Banking77 intent shortlists "
                "says nothing about your reranking task until you measure it there. Another 132 s on CPU.\n\n"
                "*Evaluation practice.* **Predict before running:** recall@5 on a six-entry shortlist — can it move at all?"
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, instruction=INSTRUCTION)\n"
                "adapted_val = pipe.evaluate(val_records, instruction=INSTRUCTION)\n"
                "comparison = {{\n"
                "    metric: {{'random_floor': round(floor[metric], 4), 'lexical': round(baseline_lexical[metric], 4), 'frozen': round(frozen_test[metric], 4), 'adapted': round(adapted_test[metric], 4)}}\n"
                "    for metric in ('recall@1', 'recall@3', 'recall@5', 'mrr')\n"
                "}}\n"
                "comparison['median_rank'] = {{'lexical': baseline_lexical['median_rank'], 'frozen': frozen_test['median_rank'], 'adapted': adapted_test['median_rank']}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 4) for metric in ('recall@1', 'recall@3', 'recall@5', 'mrr')}}\n"
                "delta_mrr = adapted_test['mrr'] - frozen_test['mrr']\n"
                "# SWP-A: the direction is a recorded verdict, not an assert, so a BYOD run always reaches export, reload and result.json.\n"
                "comparison['verdict'] = {{'adapted_vs_frozen_mrr': 'improved' if delta_mrr > 0 else ('no change' if delta_mrr == 0 else 'worse'), 'frozen_vs_random_floor_mrr': frozen_vs_floor}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'instruction': INSTRUCTION,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'candidates_per_query': dataset_manifests['test']['candidates'],\n"
                "    'baselines': {{'random_floor': floor, 'lexical': baseline_lexical}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'verdict': comparison['verdict'],\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the four rows side by side, `delta_vs_frozen`, and the `verdict`.\n\n<details><summary>Check your reasoning</summary>Hardly: with '
                'the positive almost always in the top five already, recall@5 has nowhere to go. In the recorded run recall@5 stayed 0.9935, recall@1 rose 0.7338 → '
                '0.7662 and MRR 0.8429 → 0.8634 (verdict *improved*). A three-point gain on 154 shortlists from one seed is sample-sanity evidence, not a measured '
                'improvement.</details>'
            ),
        },
        {
            "md": (
                "## 9. Rerank before and after, export the adapter and reload it\n\n"
                "The three test shortlists ranked by the frozen model in Section 5 are ranked again by the adapted model "
                "through the same `rerank` contract and printed side by side with the gold intent's rank in each. Read "
                "them as observations: the metric is Section 8, and the scores of the adapted model live on a different "
                "scale from the frozen ones — the `yes` share moves for every pair, so a score is not comparable across "
                "the two models. The per-batch `evaluation_report` helper — the inference-stage helper — is written for "
                "the first shortlist and stays `not-measurable`, because a batch of scores has no metric without "
                "relevance labels; `pipe.evaluate` is that labelled evaluation.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last two decoder layers, about 126 MB — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the instruction it was trained with, the tensor "
                "names, the file size and SHA-256, the training configuration and the epoch history (OUT8). "
                "`Qwen3RerankerPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest and "
                "digest **before** deserialising, refuses any tensor that is not a decoder-layer tensor of the base, and "
                "overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). "
                "The cell asserts identical scores on the probe shortlists and an identical validation MRR (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "after = {{r['id']: ranked_shortlist(r) for r in probe_records}}\n"
                "rows = []\n"
                "for r in probe_records:\n"
                "    rows.append({{'id': r['id'], 'query': r['query'], 'gold': r['positive'], 'frozen_order': ' | '.join(d for d, _s in before[r['id']]), 'adapted_order': ' | '.join(d for d, _s in after[r['id']]), 'gold_rank_frozen': next(i + 1 for i, (d, _s) in enumerate(before[r['id']]) if d == r['positive']), 'gold_rank_adapted': next(i + 1 for i, (d, _s) in enumerate(after[r['id']]) if d == r['positive']), 'gold_score_frozen': next(s for d, s in before[r['id']] if d == r['positive']), 'gold_score_adapted': next(s for d, s in after[r['id']] if d == r['positive'])}})\n"
                "    print({{k: rows[-1][k] for k in ('id', 'gold', 'frozen_order', 'adapted_order', 'gold_rank_frozen', 'gold_rank_adapted', 'gold_score_frozen', 'gold_score_adapted')}})\n"
                "single_report = evaluation_report(pipe.rerank(pairs, instruction=INSTRUCTION), sample_kind='one Banking77 test shortlist' if not USE_BYOD else 'one BYOD test shortlist')\n"
                "print({{'batch_report_verdict': single_report['verdict'], 'shortlists_reordered': sum(r['frozen_order'] != r['adapted_order'] for r in rows), 'of': len(rows)}})\n"
                "with open('outputs/{stem}_reranking.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(rows)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'instruction': artifact_manifest['adapter']['instruction'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = Qwen3RerankerPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_scores = reloaded.rerank(pairs, instruction=INSTRUCTION)['scores']\n"
                "in_memory_scores = pipe.rerank(pairs, instruction=INSTRUCTION)['scores']\n"
                "reloaded_val = reloaded.evaluate(val_records[:20], instruction=INSTRUCTION)\n"
                "in_memory_val = pipe.evaluate(val_records[:20], instruction=INSTRUCTION)\n"
                "parity = {{'scores_identical': reloaded_scores == in_memory_scores, 'mrr_in_memory': round(in_memory_val['mrr'], 6), 'mrr_reloaded': round(reloaded_val['mrr'], 6)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['scores_identical'] and abs(in_memory_val['mrr'] - reloaded_val['mrr']) < 1e-9\n\n"
                "weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'instruction': INSTRUCTION,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'files': {{k: {{'name': v[0], 'bytes': v[1], 'sha256': v[2]}} for k, v in CORPUS_FILES.items()}}, 'license': CORPUS_LICENSE}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'query': record['query'], 'candidates': candidates, 'scores': scores, 'ranking': result['ranking'], 'n_tokens': result['n_tokens'], 'seconds': rerank_seconds}},\n"
                "    'comparison': comparison,\n"
                "    'verdict': comparison['verdict'],\n"
                "    'reranking_before_after': rows,\n"
                "    'batch_report': single_report,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'bfloat16' if pipe.device.startswith('cuda') else 'float32'}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "**What to notice:** the gold intent's rank before and after for the three shortlists, the artifact's size, and `reload_parity`.\n\n<details><summary>Check "
                'your reasoning</summary>Scores from the adapted model are on a different scale from the frozen ones — the `yes` share moves for every pair — so '
                'compare ranks, not scores. In the recorded run the reloaded adapter gave identical scores and the same validation MRR on the 20-query parity subset '
                '(0.9125 in memory and reloaded).</details>'
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen reranker already puts the right intent phrase first for most unseen messages (recall@1 in the "
        "seventies against a lexical baseline that the hard negatives were chosen to defeat and a random floor of "
        "16.7 %), and a bounded listwise fine-tuning of the last two decoder layers on 231 shortlists lifts held-out "
        "recall@1 by +4.5 points in a few minutes on CPU, with a 126 MB adapter that reloads to identical scores. "
        "That is the claim: the adaptation contract can adapt the reranker to a relevance judgement end to end on a real "
        "labelled corpus, and the numbers it produces are read against a random floor, a lexical baseline and the frozen "
        "model rather than in isolation.\n\n"
        "The test split is 154 six-entry shortlists from one seeded split of one corpus with no dispersion estimate; "
        "recall@k and MRR say whether the gold document is ordered first within a shortlist the dataset supplied, not how "
        "the model would rank a real candidate pool. The adapter changes the last layers, which every pair shares, so "
        "every score shifts and scores are not comparable across the frozen and adapted models; nothing here measures the "
        "effect on other judgements. On CUDA the model runs in bfloat16 and the recorded float32 CPU numbers will not "
        "reproduce to the last digit. A cross-encoder scores one pair per forward pass, so evaluation cost grows with the "
        "shortlist length — a first-stage retriever (the sibling embedding pipeline) is what makes the shortlist short.\n\n"
        "Three things to carry to real data. **Floors first:** the random floor and the lexical baseline on *your* "
        "shortlists are the numbers to read before any reranker's; a lexical baseline near the frozen model means the "
        "task is mostly keyword matching. **Leakage:** de-duplicate queries across splits (the contract does this "
        "case-insensitively) and split by user or session when your queries come from one. **Negatives:** the quality of "
        "the shortlist decides what the reranker learns — draw negatives from the retriever you will deploy behind, not "
        "at random.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled corpus, "
        "validate the demonstrated dataset contract without leakage, execute the reranking contract and a bounded listwise "
        "fine-tuning, evaluate by ranking metrics against a random floor, a lexical baseline and the frozen model on an "
        "independent split, and emit the shown machine-readable artifacts — without the repository being reachable. It "
        "does **not** establish benchmark superiority, ranking quality on any other task, a usable acceptance threshold, "
        "or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_LAYERS = 1` and watch the gain "
        "shrink; set `TRAIN_CANDIDATES = 6` to train on the whole shortlist and compare; set `EPOCHS = 3` and watch "
        "whether validation MRR keeps rising or turns (the best epoch is kept either way); change `INSTRUCTION` and re-read "
        "the frozen numbers — the judgement is instruction-conditioned; or bring your own shortlists through BYOD and read "
        "the lexical baseline before the adapted number.\n\n"
        '## Troubleshooting\n\n- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google '
        'Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 '
        'again; a complete environment built from the same lock is reused, an incomplete one is finished. If it repeats, the network is blocking or altering '
        '`files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and '
        'choose **Run all**.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it '
        'keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message '
        'names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again.\n- **Section 4 reports a size or SHA-256 mismatch for a '
        'Banking77 file** — the download from `raw.githubusercontent.com` was cut short or altered. Run Section 4 again; a verified cached file is reused.\n- '
        '**Sections 6–8 are very slow** — a cross-encoder runs one forward pass per query–candidate pair (924 for the test split); on CPU that is minutes per '
        'evaluation. Switch to a T4 GPU.\n- **Your numbers differ slightly from the recorded run** — CUDA runs bfloat16 and CPU float32; the comparison between '
        'systems, not the last digit, is the result.\n- **BYOD: a column or record refusal** — the message names the rule: columns `id`, `query`, `positive`, '
        '`negatives` (separated by ` | `), 1..15 distinct negatives per record, at least eight records; set `INSTRUCTION` to describe your relevance judgement.\n- '
        '**BYOD: "BYOD_PATH … is not a file"** — the path is relative to the working directory printed in the message; give one .csv, .json or .jsonl file.\n- '
        '**BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the file in the runtime and set `BYOD_PATH` to its path.\n- '
        '**BYOD: "Upload exactly one file"** — the dialog was cancelled or several files were chosen; run the cell again.\n\n## Glossary\n\n- **Reranker / '
        'cross-encoder:** a model that reads the query and one candidate together and scores their relevance; accurate, but one forward pass per pair.\n- '
        '**Shortlist:** the small candidate list a first-stage retriever would hand the reranker; here the positive plus five negatives.\n- **Hard negatives:** '
        "wrong candidates chosen to look right (here, the highest word overlap with the message).\n- **Relevance score:** the `yes` share of the model's yes/no "
        'answer for one pair; it orders candidates and is not a calibrated probability.\n- **Instruction:** the one-line relevance judgement every pair carries; '
        'the model is instruction-conditioned.\n- **Recall@k / MRR:** the share of queries whose positive is in the top k; the mean of 1 / rank of the positive.\n- '
        "**Random floor / lexical baseline:** a random ordering's expected score; ordering by shared-word (Jaccard) overlap.\n- **Listwise loss:** the "
        'cross-entropy of the positive over its whole candidate list, rather than one pair at a time.\n- **Frozen / adapted / pinned base:** the packaged model; '
        'the model after Section 7; the verified packaged weights every adaptation starts from (`restore_base`).\n- **Isolated environment:** the separate '
        'Python environment Section 1 builds from the hash lock; every later cell runs there.\n\n## Conclusion (your notes)\n\nComplete these in your own words; '
        "the recorded run's values are in the **Check your reasoning** answers above.\n\n- On the 154 test shortlists the frozen reranker reached recall@1 ___ "
        "against the lexical baseline's ___ and the random floor's ___.\n- Listwise fine-tuning moved recall@1 to ___ and MRR to ___ (verdict: ___), which I "
        'read as ___.\n- The number I would not trust on its own is ___, because ___.\n- Before adapting on my own shortlists I would draw negatives from ___, '
        'de-duplicate by ___, and compare against ___.\n\n'
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/qwen3-reranker-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/qwen3-reranker-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/qwen3-reranker-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/QwenLM/Qwen3-Embedding\n"
        "- Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models (2025): https://arxiv.org/abs/2506.05176\n"
        "- Efficient Intent Detection with Dual Sentence Encoders (Casanueva et al., 2020; Banking77, CC BY 4.0): https://arxiv.org/abs/2003.04807\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
