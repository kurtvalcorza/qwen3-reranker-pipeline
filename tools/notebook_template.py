"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.5 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py, protocol.py), and the model pin/stage/verify cells are produced by
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
    "notebook_kind": "TUTORIAL",
    "use_notice": (
        "> **⚠️ Notice.** For learning, research and evaluation only; not validated for production, customer-facing or "
        "safety-critical use, and the relevance score is not a calibrated probability or an acceptance decision. Running "
        "this notebook downloads third-party model weights and a dataset under their own licenses (Apache-2.0 for the model, "
        "CC BY 4.0 for Banking77; see the model card). Do not upload confidential or personal data to a hosted runtime unless "
        "you are authorized to process it there."
    ),
    "external_access": (
        "the Hugging Face Hub, to fetch the pinned `{MODEL_ID}` snapshot (~{total_mb:.0f} MB in total) at revision "
        "`{MODEL_REVISION}`, and GitHub raw content (`raw.githubusercontent.com`) for the two pinned Banking77 CSV files of "
        "the `PolyAI-LDN/task-specific-datasets` repository (see the data bullet above). No credentials are required; nothing "
        "is installed from this repository."
    ),
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
        "the `PolyAI-LDN/task-specific-datasets` repository on GitHub at a pinned commit (1.1 MB, no credential), pairs every "
        "customer message with its intent phrase and a seeded shortlist of five other intent phrases, draws 231 / 77 / 154 "
        "training, validation and exposed-development-test records and a pinned 385-record fresh test balanced over the 77 "
        "intents from the release's own partition, reranks one test shortlist through the inference contract "
        "with an input manifest and a rejection probe, scores the frozen reranker on the exposed development test by recall@1 / "
        "recall@3 / recall@5 and MRR beside the random floor and a lexical (token-overlap) baseline under two tie policies, "
        "selects the learning rate with a pre-registered three-value grid on validation, runs a bounded listwise "
        "fine-tuning of the last two decoder layers with validation-MRR epoch selection, scores the fresh test only under "
        "the pre-registered settings, reranks the same shortlists with the adapted model, exports the adapter as safetensors with "
        "a manifest, and reloads that artifact into a fresh pipeline to verify parity. The default path needs no "
        "repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.5 §5). On CPU the whole path takes 75 minutes or more of model time after the downloads (the recorded T4 run took 16.6 minutes for the whole session, 995 s including the environment build and the 1.2 GB download) — "
        "a cross-encoder scores every query–candidate pair with a full forward pass, so evaluation dominates; a CUDA "
        "runtime is used automatically when present (float32 on every device; pre-registration amendment 2)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "records — as `BYOD_PATH` (a path in the runtime, which works on Colab, Kaggle and Jupyter) or, when it is empty, through the "
        "Colab upload dialog — as a CSV (columns `id`, `query`, `positive`, `negatives` with the negatives separated by ` | `), a JSON "
        "array or a JSONL file of `{id, query, positive, negatives}` records; set `INSTRUCTION` to a one-line description "
        "of your relevance judgement. They pass through the same validation, seeded query-disjoint split, floor and "
        "baseline, frozen scoring, fine-tuning, held-out evaluation, reranking, artifact export and reload-parity cells as "
        "the Banking77 sample, except that a BYOD run is exploratory: it reads validation and its own test split and never the fresh test. The expected schema and the ceilings are stated in the Prerequisites and in Section 4, and "
        "uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "Qwen3RerankerPipeline",
    "guided": {
        "opening": [
            '**Who this notebook is for.** The intended audience is a learner who knows basic Python, has used Colab or Jupyter, and wants to see how a '
            'cross-encoder reranker orders a shortlist of candidate documents for a query, how ranking quality is measured against honest floors (a random '
            'floor and a lexical baseline, read under the same tie policy), and what a bounded listwise fine-tuning changes. No prior experience with Qwen3, rerankers '
            'or ranking losses is assumed; terms are explained where they first matter and again in the **Glossary** at the end. CPU works but is slow (75 minutes '
            'or more of model time, because a cross-encoder runs one forward pass per pair); a GPU runtime is much faster.\n\n**Input → Model → Output.**\n\n| '
            '| Reranking | Bounded listwise fine-tuning |\n|---|---|---|\n| Input | (query, document) pairs with a one-line relevance instruction | Banking77 '
            'messages, each with its intent phrase and five shortlisted negatives (231 / 77 / 154 plus a 385-record fresh test, query-disjoint) |\n| Model | the Qwen3-Reranker-0.6B decoder, '
            'reading the `yes` / `no` logits at the last position | the same model; only the last two decoder layers train (31.5 M of 595.8 M parameters) with a '
            'listwise cross-entropy |\n| Output | one relevance score per pair (the `yes` share — not a calibrated probability) and a ranking | recall@1/3/5 and MRR '
            'beside a random floor, a lexical baseline and the frozen model, and a safetensors adapter (about 126 MB in float32 on every device) that reloads with parity |\n\n**How to use this '
            'notebook.** Choose a runtime (**Runtime → Change runtime type → T4 GPU** is strongly recommended), then **Runtime → Run all**. Run all completes in '
            "one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated "
            'environment, the carried package and the model snapshot — and their cells are collapsed; you may run them without studying them. The learning path '
            'starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each '
            'principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked '
            "answer from the recorded run: the hosted Colab T4 run of notebook blob `2d6e5607` (commit `ff6da3a`, 18 October 2026; `docs/execution-evidence/2026-10-18-ff6da3a/`), float32, the pre-registered evaluation of record (`docs/release-verification.md`). Most worked answers are on the 154 shortlists called the "
            "**exposed development test**; on the 385-shortlist fresh test that run gave MRR 0.851 → 0.879 (paired difference +0.028, 95 % CI 0.009 to 0.048) and recall@1 +0.047 (95 % CI 0.013 to 0.081), verdict *improved*. "
            'This notebook runs float32 on every device (amendment 2), and the recorded T4 run equals the CPU float32 build at printed precision on the grid and the exposed development test. Every adaptation starts from the pinned base, so re-running '
            'Section 7 with other settings is a fresh, **exploratory** experiment (it never scores the fresh test); Sections 5 and 6 put the pinned base back before they measure, so their numbers are always the '
            "frozen model's, and after re-running them you re-run Section 7 before Sections 8 and 9 (those two cells refuse to treat the base as \"adapted\"). "
            '**Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your '
            'predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the Banking77 corpus, shortlists, validation and the split *(evaluation practice)* '
            '→ 5 the reranking contract *(core concept: relevance scores, not probabilities)* → 6 the random floor, the lexical baseline and the frozen model '
            '*(evaluation practice)* → 7 learning-rate selection on validation and bounded listwise fine-tuning *(core concept)* → 8 the fresh test, scored only under the pre-registered settings *(evaluation practice)* → 9 reranking before and '
            'after, export and fresh reload *(engineering)* → interpretation, troubleshooting, glossary and your conclusion.'
        ],
    },
    "weights_key": "qwen3-reranker-0.6b",
    "modules": ["pipeline.py", "metrics.py", "samples.py", "protocol.py"],
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
        "intents as two digest-pinned CSV files fetched from the `PolyAI-LDN/task-specific-datasets` repository on GitHub "
        "(`raw.githubusercontent.com`) at a pinned commit. Every intent name "
        "becomes a short **document** (`card_arrival` → `card arrival`), every message a **query** whose positive is its "
        "intent phrase, and every query carries a seeded **shortlist** of five other intent phrases — the three with the "
        "highest token overlap with the message plus two random ones — so reranking the six-entry shortlist is intent "
        "detection over hard candidates. **Possible training overlap (DAT9):** Banking77 has been public since 2020 and is "
        "an MTEB task (Banking77Classification). The Qwen3 Embedding report [1] names MS MARCO, NQ, HotpotQA and synthetic "
        "classification pairs among its supervised data, but it does not list the reranker's training sets or describe any "
        "decontamination, and the base model's pretraining text is not enumerated, so overlap with these messages cannot "
        "be ruled out and the frozen number may include exposure. The carried `metrics.py` orders each "
        "shortlist by the relevance score and reads the rank of the positive: **recall@1**, **recall@3**, **recall@5** "
        "and **MRR** under two tie policies side by side — **pessimistic** (ties count against the positive, the headline "
        "numbers) and **expected** (ties broken at random, in expectation); a **random floor** (1 / 6 recall@1, an expectation) "
        "and a **lexical baseline** (Jaccard token overlap between message and phrase — the signal the hard negatives were "
        "chosen by) frame the frozen number. The fine-tuning question is whether a bounded listwise adaptation of the last decoder layers on "
        "231 shortlists raises ranking on messages the model has not seen. The learning rate is chosen on validation by a "
        "pre-registered grid (`docs/qwen3-reranker-preregistration.md`); the 154 original test shortlists informed an earlier "
        "learning-rate choice and are now the **exposed development test**, and a **fresh test** of 385 never-scored "
        "shortlists is scored only under the pre-registered settings. Nothing here is a quality claim about your "
        "reranking task: it is one seeded split of one corpus."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned real corpus and validate and split it "
        "without leakage; rerank a shortlist through the public API with the instruction contract and read the score and "
        "ranking contract correctly; read recall@k and MRR beside a random floor and a lexical baseline and understand "
        "what they do and do not measure; run a bounded listwise fine-tuning with explicit hyperparameters and "
        "validation-based epoch and learning-rate selection; tell an exposed development test from a fresh test scored only under pre-registered settings; compare rankings before and after; and "
        "export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "embedding or first-stage retrieval (see the sibling Qwen3 embedding pipeline), text generation or chat, "
        "calibrated probabilities or a relevance threshold, graded (non-binary) relevance, hard-negative mining beyond the "
        "seeded shortlists of the dataset contract, full-model or embedding-table training, and any claim that a "
        "Banking77 intent shortlist stands in for your reranking task. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available. **Precision:** the pipeline loads, runs and trains the model in float32 on CPU and CUDA alike (pre-registration amendment 2); earlier recorded T4 runs used bfloat16 and are labelled so. CPU is adequate but not fast for a cross-encoder: the build record measured about 5 s to load and digest-verify the 1.19 GB snapshot, about 132 s to score the 154 test shortlists (924 pairs) and about 170 s per training epoch over 231 shortlists plus a validation pass per epoch. Building the isolated environment (the pinned `torch==2.14.0` among its packages; reused on a re-run) and the 1.19 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python; what a cross-encoder does and why it scores one pair per forward pass; what recall@k and mean reciprocal rank measure over a candidate list; what a listwise softmax loss does.",
        "- **Data contract:** records are `{id, query, positive, negatives}` — a query of 1..100,000 characters, a positive document of 1..1,000 characters and 1..15 distinct negative documents (the pair prompt is truncated longest-first to 8,192 tokens at inference and to 192 tokens **during training only**), ids matching `[A-Za-z0-9_.:-]{1,64}` and unique; a training set needs 8..20,000 records, and a BYOD dataset needs **12..10,002 distinct queries** (`split_size_bounds()`: 20 % go to test and 15 % to validation, each scored in one call of at most `MAX_EVAL_RECORDS` = 2,000 records, and at least 8 must stay for training); queries are de-duplicated case-insensitively before splitting so the same message never sits in two splits. BYOD accepts CSV (negatives separated by ` | `), JSON or JSONL in that shape.",
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
                "shuffle — the release's own partition, balanced over all 77 intents. Those 154 test records are the **exposed "
                "development test**: an earlier build compared two learning rates on them, so they are shown for continuity but "
                "decide nothing. `build_fresh_test` draws the **fresh test** from the same seeded shuffle of the `test` member at the "
                "positions after those 2 per intent (5 per intent = 385), skipping messages already in a split, messages whose text "
                "is in `train.csv` and near-duplicates (token Jaccard ≥ 0.8) of training and validation messages; `check_fresh_test` "
                "refuses it unless its source ids and records match the pinned digests. Drawing it scores nothing: Section 8 scores "
                "it only under the pre-registered settings. A known property, not fixed (it would change every digest): when a message "
                "shares no word with any other intent phrase, its three \"hard\" negatives fall back to sort order (`Refund not showing "
                "up`, `activate my card`, `age limit`) — 7 / 231 training, 1 / 77 validation, 3 / 154 exposed and 10 / 385 fresh "
                "shortlists, the same for every system. `validate_dataset` then checks every "
                "record against the contract, `check_split_disjoint` asserts no message appears in two splits, and the "
                "training split is written to `outputs/{stem}_train.csv` in the shape BYOD expects. A BYOD file is split by "
                "`split_dataset`, which checks its size once and names it: 12..10,002 distinct queries at the default "
                "fractions. The 8-record minimum applies to the training split; validation and test need one record each. `INSTRUCTION` is "
                "the judgement every pair carries in later cells.\n\n"
                "Look for: 10,003 + 3,080 raw rows, two digests, splits 231 / 77 / 154, the fresh test's 385 records and its pins, six candidates per query, and "
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
                "data_source_kind = 'BYOD' if USE_BYOD else SAMPLE_DATA_SOURCE\n"
                "# The fresh test exists only for the pinned sample at the default split seed; drawing it scores nothing.\n"
                "fresh_draw = fresh_test_records = None\n"
                "if not USE_BYOD and SPLIT_SEED == SAMPLE_SEED:\n"
                "    fresh_draw = build_fresh_test(corpus, splits, seed=SPLIT_SEED)\n"
                "    fresh_check = check_fresh_test(fresh_draw, splits)\n"
                "    fresh_test_records = fresh_draw['records']\n"
                "    print({{'fresh_test': {{'n': fresh_check['n_records'], 'source_ids_sha256': fresh_check['source_ids_sha256'][:16] + '...', 'digest': fresh_check['digest'][:16] + '...', 'skipped': fresh_check['skipped'], 'scored': 'only in Section 8, under the pre-registered settings'}}}})\n"
                "else:\n"
                "    print({{'fresh_test': 'not drawn: BYOD or a non-default SPLIT_SEED makes this run exploratory (validation and its own test split only)'}})\n"
                "# A new dataset invalidates any earlier grid selection or evaluation in this runtime.\n"
                "grid_selection = adapt_result = None\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "# RR-M3: the 8-record floor is the training floor; validation and test need one record each (split_dataset checked the sizes).\n"
                "dataset_manifests = {{name: validate_dataset(part) if name == 'train' else validate_dataset(part, min_records=1, max_records=MAX_EVAL_RECORDS) for name, part in splits.items()}}\n"
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
                'reasoning</summary>The lexical baseline is handicapped: three of the five negatives are, by construction, the phrases with the most words in common with the '
                'message, so a word counter has to beat its own best guesses, while a reranker can judge meaning. Section 6 shows how much of its weakness is really '
                'ties. The shortlist is seeded per record id, so it reproduces; the splits are '
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
                "kept as the *before* column for Section 9. The cell first puts the pinned base back (`restore_base`) if an "
                "earlier Section 7 adapted the model, so a re-run — for example with your own data — always shows the frozen model "
                "here. The printed `ceilings` are what the pipeline enforces; `TRAIN_CANDIDATES` = `MAX_TRAIN_CANDIDATES` (4) is only "
                "the default list length, and `adapt` accepts any value in `TRAIN_CANDIDATES_RANGE` (2..8).\n\n"
                "**Predict before running:** will the six scores of one shortlist add up to 1?"
            ),
            "code": (
                "import time\n\n"
                "# RR-M2: this cell describes the frozen model, so an earlier adaptation is undone first.\n"
                "restored_tensors = pipe.restore_base()\n"
                "if restored_tensors:\n"
                "    print({{'restored_pinned_base': len(restored_tensors), 'note': 'an earlier Section 7 had adapted the model; run Section 7 again before Sections 8 and 9'}})\n"
                "probe_records = test_records[:3]\n"
                "record = probe_records[0]\n"
                "candidates = candidate_list(record)\n"
                "pairs = [(record['query'], doc) for doc in candidates]\n"
                "print({{'ceilings': {{'MAX_PAIRS': MAX_PAIRS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MAX_TRAIN_TOKENS': MAX_TRAIN_TOKENS, 'TRAIN_CANDIDATES_RANGE': TRAIN_CANDIDATES_RANGE, 'MAX_EVAL_RECORDS': MAX_EVAL_RECORDS}}, 'defaults': {{'TRAIN_CANDIDATES': MAX_TRAIN_CANDIDATES}}, 'yes_no_token_ids': (YES_TOKEN_ID, NO_TOKEN_ID)}})\n"
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
                "## 6. The random floor, the lexical baseline and the frozen model on the exposed development test\n\n"
                "Three numbers frame the adaptation, all over the same 154 six-entry shortlists of the **exposed development "
                "test** (shown for continuity; it decides nothing). The **random floor** is what a uniformly random ordering "
                "achieves in expectation (recall@1 = 1 / 6, MRR ≈ 0.41). The **lexical baseline** orders each shortlist by "
                "Jaccard token overlap with the message. It is a sanity reference, not a competitor: three of the five negatives "
                "were *chosen* as the phrases with the highest overlap. **Ties matter here:** the headline metrics count a tie "
                "against the positive (pessimistic), while the floor is an expectation over random orderings. Jaccard ties a lot "
                "(86 of the 154 shortlists have a negative tied with the gold phrase; 37 gold phrases share no word with the "
                "message), so the pessimistic lexical recall@3 (0.4156) and recall@5 (0.7532) fall **below the floor** (0.5, "
                "0.8333) — a tie-handling artefact, not word overlap from the hard negatives. Broken at random like the floor "
                "(`tie_expected`), the same baseline reaches recall@1 0.3335, recall@3 0.5640, recall@5 0.9118 and MRR 0.5301, "
                "above the floor at every k. Both are printed side by side, for every system, with `ties_with_positive`; model "
                "scores tie only when two are bit-identical (the bfloat16 T4 run of 2026-10-18 had one such tie), so read that count for the "
                "model too. Beating the lexical baseline shows the reranker is not just counting words; it is not evidence of "
                "skill on its own. "
                "`pipe.evaluate` scores every query–candidate pair through `rerank` (`MAX_PAIRS` at a time), orders each "
                "shortlist by the relevance score and reads the rank of the positive; ties are counted against it. Look "
                "for the frozen reranker well above both — the hosted Colab T4 float32 run of notebook blob 2d6e5607 (commit ff6da3a, 18 October 2026; docs/execution-evidence/2026-10-18-ff6da3a/), the evaluation of record printed recall@1 0.7338 and MRR "
                "0.8429 here — and read `median_rank` beside the means. About 132 s on CPU: 924 forward "
                "passes. Like Section 5, the cell puts the pinned base back first, so `frozen_model_test` is always the "
                "pretrained model (`adapted: False`). Whether the frozen model beats the random floor is recorded as a "
                "**verdict**; on your own shortlists the run continues either way, while on the Banking77 sample a frozen model "
                "at or below the floor stops the cell, because that would mean a broken run.\n\n"
                "*Evaluation practice.* **Predict before running:** if the lexical baseline lands below the random floor at recall@3, is "
                "that because of the hard negatives, or because of how ties are counted?"
            ),
            "code": (
                "def brief(m):\n"
                "    out = {{k: round(m[k], 4) for k in ('recall@1', 'recall@3', 'recall@5', 'mrr')}} | {{'median_rank': m.get('median_rank')}}\n"
                "    if 'tie_expected' in m:  # T73-3: the same metrics with ties broken at random (the floor's convention)\n"
                "        out |= {{'ties_with_positive': m['ties_with_positive'], 'tie_expected': {{k: round(v, 4) for k, v in m['tie_expected'].items()}}}}\n"
                "    return out\n\n"
                "if pipe.restore_base():  # RR-M2: the frozen numbers are always the pinned base's\n"
                "    print({{'restored_pinned_base': True, 'note': 'an earlier Section 7 had adapted the model; run Section 7 again before Sections 8 and 9'}})\n"
                "floor = random_floor([1 + len(r['negatives']) for r in test_records])\n"
                "print({{'random_floor': {{k: round(floor[k], 4) for k in ('recall@1', 'recall@3', 'recall@5', 'mrr')}}, 'baseline': floor['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "baseline_lexical = pipe.lexical_baseline(test_records)\n"
                "print({{'split': 'exposed development test', 'lexical_baseline': brief(baseline_lexical), 'baseline': baseline_lexical['baseline'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, instruction=INSTRUCTION)\n"
                "print({{'frozen_model_exposed_development_test': brief(frozen_test), 'n_queries': frozen_test['n_queries'], 'n_pairs': frozen_test['n_pairs'], 'verdict': frozen_test['verdict'], 'adapted': frozen_test['adapted'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "# Contract integrity: both systems ranked the same shortlists.\n"
                "if frozen_test['n_queries'] != baseline_lexical['n_queries']:\n"
                "    raise RuntimeError(f\"contract: the frozen model ranked {{frozen_test['n_queries']}} shortlists, the lexical baseline {{baseline_lexical['n_queries']}}\")\n"
                "# SWP-A: the quality comparison is a recorded verdict, not an assert, so a BYOD run still reaches adaptation and export.\n"
                "frozen_vs_floor = 'above' if frozen_test['mrr'] > floor['mrr'] else 'not above'\n"
                "print({{'verdict_frozen_vs_random_floor_mrr': frozen_vs_floor}})\n"
                "if not USE_BYOD and frozen_vs_floor != 'above':\n"
                "    raise RuntimeError(f\"On the Banking77 sample the frozen model must rank above the random floor (MRR {{frozen_test['mrr']:.4f}} vs {{floor['mrr']:.4f}}): the snapshot or the corpus is not the pinned one. Run all again from the top.\")"
            ),
        },
        {
            "md": (
                '**What to notice:** the three systems on recall@1/3/5 and MRR, `median_rank`, and the verdict line.\n\n<details><summary>Check your '
                'reasoning</summary>Ties. In the hosted Colab T4 float32 run of notebook blob 2d6e5607 (commit ff6da3a, 18 October 2026; docs/execution-evidence/2026-10-18-ff6da3a/), the evaluation of record (exposed development test) recall@1 was 0.1667 for the random floor, 0.2662 for the '
                'pessimistic lexical baseline and 0.7338 for the frozen reranker; at recall@3 the pessimistic lexical value (0.4156) fell below the floor (0.5). '
                'The floor is an expectation over random orderings, while the pessimistic rule ranks every tied negative above the gold phrase, and 86 of 154 '
                'shortlists tie. Counted the floor\'s way, lexical is 0.3335 / 0.5640 / 0.9118 at recall@1/3/5 — above the floor everywhere. Compare systems under '
                'the same tie policy. MRR: 0.4083 / 0.457 (0.5301 expected) / 0.8429.</details>'
            ),
        },
        {
            "md": (
                "## 7. Learning-rate selection on validation, then bounded listwise fine-tuning\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_LAYERS` decoder layers — two by default, "
                "31,461,888 of 595,776,512 parameters; the token embeddings (which the output projection shares), the "
                "earlier layers and the final norm stay frozen — with a **listwise** loss: each training query contributes "
                "its positive and its first `TRAIN_CANDIDATES - 1` negatives, formatted exactly as `rerank` formats them, "
                "the relevance logit of every candidate is the `yes`-minus-`no` logit at the last position, and the loss "
                "is the cross-entropy of the positive over its list. `BATCH_SIZE` counts queries per step; AdamW at a "
                "fixed learning rate, gradient clipping at 1.0, seeded shuffling and no scheduler; prompts are truncated to "
                "`MAX_TRAIN_TOKENS` (192) **during training only**. Epoch 0 records the frozen model's validation ranking "
                "metrics; every epoch is scored the same way, and the epoch with the highest validation MRR is kept.\n\n"
                "**Choosing the learning rate on validation (pre-registered).** An earlier build kept 2e-5 over 5e-5 because the "
                "*test* recall@1 was higher while validation MRR was tied, which is why the original test is now the exposed "
                "development test. Now, with `RUN_GRID = True` (the default), the cell first trains one run per learning rate in "
                "{{1e-5, 2e-5, 5e-5}} with every other setting fixed and reads each run's kept-epoch **validation** MRR; 2e-5 stays "
                "unless another beats it by at least 0.010 — a *keep the default unless clearly beaten* tolerance, at or below the "
                "sampling noise of 77 validation queries by design, not a detection threshold — and the selected run must beat the frozen model's validation MRR by 0.010 or the frozen model is kept "
                "as the evaluated system (`docs/qwen3-reranker-preregistration.md`). Then the selected learning rate is trained **again** "
                "with the same seed: Sections 8 and 9 use this re-train, not the grid run, and Section 8 checks that its kept-epoch "
                "validation MRR is within 0.010 of the grid run's. The grid adds three runs (7.4 minutes, 445 s, on the recorded T4 run; about 40 on CPU). With "
                "`RUN_GRID = False` the cell uses `LEARNING_RATE` and the run is exploratory. Any change to these fields makes the "
                "run exploratory: Section 8 then reads validation and the exposed development test only, never the fresh test. "
                "Every call starts from the pinned base (`started_from` in the printed result), so a re-run with other settings is a "
                "fresh experiment, not continued training, and epoch 0 is always the frozen model.\n\n"
                "**Predict before running:** the frozen reranker already ranks about three in four shortlists correctly. Will the "
                "three learning rates differ on validation MRR by more than the 0.010 margin?"
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "RUN_GRID = True  # @param {{type:\"boolean\"}}\n"
                "LEARNING_RATE = 2e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_LAYERS = 2  # @param {{type:\"integer\"}}\n"
                "TRAIN_CANDIDATES = 4  # @param {{type:\"integer\"}}\n"
                "ADAPT_SEED = 0  # @param {{type:\"integer\"}}\n"
                "# The pre-registered defaults (LEARNING_RATE is used only when RUN_GRID is False); Section 8 holds the Banking77 sample to an improvement only at these settings.\n"
                "DEFAULT_SETTINGS = (EPOCHS, RUN_GRID, LEARNING_RATE, BATCH_SIZE, TRAINABLE_LAYERS, TRAIN_CANDIDATES, ADAPT_SEED) == (2, True, 2e-5, 4, 2, 4, 0)\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_recall@1'] = round(entry['val']['recall@1'], 4)\n"
                "        row['val_recall@3'] = round(entry['val']['recall@3'], 4)\n"
                "        row['val_mrr'] = round(entry['val']['mrr'], 4)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "fingerprint = run_fingerprint(data_source=data_source_kind, split_seed=SPLIT_SEED, train_digest=dataset_manifests['train']['digest'], validation_digest=dataset_manifests['validation']['digest'], instruction=INSTRUCTION, epochs=EPOCHS, batch_size=BATCH_SIZE, trainable_layers=TRAINABLE_LAYERS, train_candidates=TRAIN_CANDIDATES, adapt_seed=ADAPT_SEED)\n"
                "grid_selection = adapt_result = None\n"
                "t0 = time.perf_counter()\n"
                "if RUN_GRID:  # validation only: no test split is read here\n"
                "    grid_runs = {{}}\n"
                "    for grid_lr in GRID_LRS:\n"
                "        run = pipe.adapt(train_records, val_records, instruction=INSTRUCTION, epochs=EPOCHS, lr=grid_lr, batch_size=BATCH_SIZE, trainable_layers=TRAINABLE_LAYERS, train_candidates=TRAIN_CANDIDATES, seed=ADAPT_SEED)\n"
                "        grid_runs[grid_lr] = {{'kept_epoch': run['best_epoch'], 'val_mrr': run['history'][run['best_epoch']]['val']['mrr'], 'val_mrr_by_epoch': [round(h['val']['mrr'], 4) for h in run['history']]}}\n"
                "        print({{'grid_lr': grid_lr, 'kept_epoch': run['best_epoch'], 'kept_val_mrr': round(grid_runs[grid_lr]['val_mrr'], 4), 'val_mrr_by_epoch': grid_runs[grid_lr]['val_mrr_by_epoch'], 'seconds': run['seconds']}})\n"
                "    grid_selection = select_lr({{lr: row['val_mrr'] for lr, row in grid_runs.items()}}, run['history'][0]['val']['mrr'], fingerprint=fingerprint)\n"
                "    grid_selection['runs'] = grid_runs\n"
                "    print({{'grid_selection': {{k: grid_selection[k] for k in ('selected_lr', 'evaluated_system', 'reason', 'scores', 'frozen_val_mrr', 'margin')}}}})\n"
                "    lr_used = grid_selection['selected_lr']\n"
                "else:\n"
                "    lr_used = LEARNING_RATE\n"
                "    print({{'grid': 'skipped (RUN_GRID = False): LEARNING_RATE is used and this run is exploratory'}})\n"
                "adapt_result = pipe.adapt(train_records, val_records, instruction=INSTRUCTION, epochs=EPOCHS, lr=lr_used, batch_size=BATCH_SIZE, trainable_layers=TRAINABLE_LAYERS, train_candidates=TRAIN_CANDIDATES, seed=ADAPT_SEED, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'objective': adapt_result['objective'], 'lr': adapt_result['lr'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'train_pairs': adapt_result['n_train_pairs'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'started_from': adapt_result['started_from'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '**What to notice:** epoch 0 (`note: frozen model`), the training loss, `val_recall@1` and `val_mrr` per epoch, `best_epoch`, and `started_from`.\n\n<details><summary>Check '
                "your reasoning</summary>Often not. In the earlier CPU build (commit d79994f) 2e-5 and 5e-5 reached the same kept-epoch validation MRR (0.923), and on the recorded T4 run 0.9232 and 0.9229, "
                'with 5e-5 peaking at epoch 1 — on 77 validation queries such differences sit inside the margin, so the default stays unless a challenger clearly '
                'wins. Validation MRR, not training loss and never a test split, picks both the learning rate and the kept epoch. Your grid table is the answer for '
                'this runtime.</details>'
            ),
        },
        {
            "md": (
                "## 8. The fresh test (pre-registered) and the exposed development test\n\n"
                "The cell first decides whether this run may read the fresh test. It compares every pre-registered setting — data "
                "source, split seed, the fresh test's pins, instruction, epochs, batch size, trainable layers, training candidates, "
                "adapt seed, training token budget, the grid having run on these same inputs, and the learning rate equal to the "
                "grid's selection — and checks that the model in memory is the run Section 7 trained. `settings_differing` lists "
                "anything that differs and `test_look` names the outcome: **pre-registered evaluation (this runtime)** scores the "
                "385 fresh shortlists for the random floor, the lexical baseline, the frozen model and the evaluated system; a "
                "second identical scoring in the same runtime, and any identical scoring once the carried protocol names the evaluation of record (`RECORDED_EVALUATION`, the 2026-10-18 hosted T4 run), is a **reproduction** with `_reproduction_<n>` files; anything else is **not scored "
                "(exploratory)**, and its files get an `_exploratory_<n>` suffix so they never overwrite the pre-registered ones. "
                "Which hosted run is the evaluation of record is stated in `docs/release-verification.md`, not here. The exposed "
                "development test (the original 154 shortlists, which informed an earlier learning-rate choice) is always shown "
                "for continuity, and no message in either test appears in the training or validation splits. The adapted reranker "
                "is scored exactly as the frozen one was in Section 6 — same shortlists, same instruction — and the four rows are "
                "put side by side. Look for recall@1 up by "
                "several points and MRR up accordingly; the cell records whether the adapted MRR is above the frozen MRR as a "
                "**verdict** (`improved`, `no change` or `worse`) in the report and `result.json` instead of asserting it, so a run on "
                "your own shortlists that does not improve still exports and reloads; on the Banking77 sample at the default "
                "settings the exposed-development verdict is printed and decides nothing — no stop reads a test split, and the "
                "report is written right after the fresh test is scored. The cell refuses to run if the pipeline holds the pinned "
                "base (Section 5 or 6 re-run after Section 7). The fresh-test block adds a **paired bootstrap 95 % CI** (10,000 "
                "resamples of the 385 shortlists, seed 20261018) for each model and for adapted − frozen, and its verdict "
                "(`improved` / `worse` / `inconclusive`) is read from that interval, with the point estimate beside it; the interval "
                "covers only the choice of shortlists, not split-to-split or seed-to-seed variation. `device` and `dtype` are "
                "recorded: the evaluation of record is the first hosted T4 run with float32 training (amendment 2). The deltas are "
                "evidence that the adaptation contract works, not a benchmark, and a gain on Banking77 intent shortlists "
                "says nothing about your reranking task until you measure it there. About 2 minutes on CPU for the exposed "
                "development test, and about 5 more per model for the fresh test.\n\n"
                "*Evaluation practice.* **Predict before running:** if you change one field in Section 7 and re-run, which numbers can this cell still show you?"
            ),
            "code": (
                "if pipe.adapter is None:  # RR-M2: never score the pinned base as 'adapted'\n"
                "    raise RuntimeError('The pipeline holds the pinned base, not an adapted model (Section 5 or 6 was re-run after Section 7 and put the base back): run Section 7, then this cell.')\n"
                "# SCORE_ONCE_RULE: the fresh test is read only when every pre-registered setting holds and the model is the selected run.\n"
                "retrain = retrain_check(grid_selection, adapt_result)  # the fresh test scores a re-train of the selected lr (amendment 1, s4)\n"
                "model_state = adapter_mismatch(pipe.adapter, adapt_result) + retrain_mismatch(retrain)\n"
                "trained = adapter_settings(pipe.adapter)  # training settings from the model in memory, not the form fields (amendment 1, s5)\n"
                "fingerprint_now = run_fingerprint(data_source=data_source_kind, split_seed=SPLIT_SEED, train_digest=dataset_manifests['train']['digest'], validation_digest=dataset_manifests['validation']['digest'], instruction=trained['instruction'], epochs=trained['epochs'], batch_size=trained['batch_size'], trainable_layers=trained['trainable_layers'], train_candidates=trained['train_candidates'], adapt_seed=trained['adapt_seed'])\n"
                "device = str(getattr(pipe, 'device', 'cpu'))\n"
                "dtype = compute_dtype(device)  # float32 on every device (amendment 2); recorded, not pre-registered: the record (2026-10-18, T4) ran float32 (amendment 2)\n"
                "run_settings = {{'data_source': data_source_kind, 'split_seed': SPLIT_SEED, 'fresh_test_digest': fresh_draw['digest'] if fresh_draw else None, **trained, 'grid_run': RUN_GRID, 'device': device, 'dtype': dtype}}\n"
                "differing = settings_differing(run_settings, grid_selection, fingerprint_now, model_state)\n"
                "preregistered = not differing  # every fixed setting, the grid selection and the model state\n"
                "FRESH_MASKED = os.environ.get('DIMER_MASK_FRESH_TEST') == '1'  # development checks only (SCORE_ONCE_RULE item 8)\n"
                "# With an evaluation of record in the carried protocol, an identical pre-registered scoring is its reproduction (rule item 3).\n"
                "prior_look = globals().get('fresh_test_look') or (TEST_LOOKS[1] if RECORDED_EVALUATION else None)\n"
                "test_look = look_at_test(differing, {{'test_look': prior_look}}, masked=FRESH_MASKED)\n"
                "suffix = output_suffix('outputs/{stem}_evaluation_report.json', test_look)\n"
                "def suffixed(path):\n"
                "    p = Path(path)\n"
                "    return str(p.with_name(p.stem + suffix + p.suffix))\n"
                "print({{'test_look': test_look, 'settings_differing': differing, 'output_suffix': suffix or '(none)', 'device': device, 'dtype': dtype, 'retrain_check': retrain, 'evaluation_of_record': RECORDED_EVALUATION or 'none recorded yet: the first hosted pre-registered run will be (docs/release-verification.md)'}})\n"
                "if differing:\n"
                "    print('This run is exploratory: it reads validation and the exposed development test only; the fresh test is not scored.')\n"
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
                "comparison['split'] = 'exposed development test (informed an earlier learning-rate choice; shown for continuity)'\n"
                "comparison['tie_expected'] = {{name: {{k: round(v, 4) for k, v in m['tie_expected'].items()}} for name, m in (('lexical', baseline_lexical), ('frozen', frozen_test), ('adapted', adapted_test))}}\n"
                "comparison['ties_with_positive'] = {{'lexical': baseline_lexical['ties_with_positive'], 'frozen': frozen_test['ties_with_positive'], 'adapted': adapted_test['ties_with_positive']}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "fresh_result = None\n"
                "if fresh_test_scored(test_look):\n"
                "    if model_state:\n"
                "        raise RuntimeError(f'refusing to score the fresh test: {{model_state}}')\n"
                "    fresh_floor = random_floor([1 + len(r['negatives']) for r in fresh_test_records])\n"
                "    fresh_lexical = pipe.lexical_baseline(fresh_test_records)\n"
                "    fresh_frozen = pipe.evaluate_base(fresh_test_records, instruction=INSTRUCTION)\n"
                "    fresh_systems = {{'random_floor': fresh_floor, 'lexical': fresh_lexical, 'frozen': fresh_frozen}}\n"
                "    if grid_selection['adaptation_kept']:\n"
                "        fresh_systems['adapted'] = pipe.evaluate(fresh_test_records, instruction=INSTRUCTION)\n"
                "    fresh_result = {{'test_look': test_look, 'evaluated_system': grid_selection['evaluated_system'], 'n_queries': len(fresh_test_records), 'digest': fresh_draw['digest'], 'metrics': {{name: {{k: round(m[k], 4) for k in ('recall@1', 'recall@3', 'recall@5', 'mrr')}} for name, m in fresh_systems.items()}}, 'tie_expected': {{name: {{k: round(v, 4) for k, v in m['tie_expected'].items()}} for name, m in fresh_systems.items() if 'tie_expected' in m}}, 'ties_with_positive': {{name: m['ties_with_positive'] for name, m in fresh_systems.items() if 'ties_with_positive' in m}}, 'device': device, 'dtype': dtype}}\n"
                "    fresh_result['bootstrap'] = paired_bootstrap({{name: m['ranks'] for name, m in fresh_systems.items() if 'ranks' in m}}, n_boot=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)\n"
                "    if 'difference' in fresh_result['bootstrap']:\n"
                "        fresh_result['verdict'] = {{s: fresh_result['bootstrap']['difference'][s]['verdict'] for s in ('mrr', 'recall@1')}}\n"
                "    fresh_test_look = test_look\n"
                "    print({{'fresh_test': fresh_result}})\n"
                "else:\n"
                "    print({{'fresh_test': test_look}})\n"
                "# Amendment 1, s2: the exposed development test decides nothing, so its verdict is recorded, never a stop; the report\n"
                "# (with the fresh-test block) is written next, and no stop after scoring can lose it.\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'instruction': INSTRUCTION,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'preregistration': PREREGISTRATION,\n"
                "    'test_look': test_look,\n"
                "    'preregistered': preregistered,\n"
                "    'settings_differing': differing,\n"
                "    'settings': run_settings,\n"
                "    'retrain_check': retrain,\n"
                "    'grid_selection': grid_selection,\n"
                "    'fresh_test': fresh_result,\n"
                "    'exposed_development_test': {{'note': 'the original 154 test shortlists; they informed an earlier learning-rate choice', 'frozen': frozen_test, 'adapted': adapted_test}},\n"
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
                "report_path = suffixed('outputs/{stem}_evaluation_report.json')\n"
                "with open(report_path, 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'report': report_path}})\n"
                "if comparison['verdict']['adapted_vs_frozen_mrr'] != 'improved':\n"
                "    print({{'note': 'on the exposed development test the adapted MRR is not above the frozen MRR; this split decides nothing (amendment 1)'}})"
            ),
        },
        {
            "md": (
                '**What to notice:** `test_look` and `settings_differing`, the four rows on the exposed development test, `delta_vs_frozen`, the `verdict`, and the fresh-test block.\n\n<details><summary>Check your reasoning</summary>Validation and the exposed '
                'development test only: any changed field makes the run exploratory, the fresh test stays unscored and the files get an `_exploratory_<n>` suffix. '
                'For reference, in the hosted Colab T4 float32 run of notebook blob 2d6e5607 (commit ff6da3a, 18 October 2026; docs/execution-evidence/2026-10-18-ff6da3a/), the evaluation of record the exposed development test gave recall@1 0.7338 → 0.7792, MRR 0.8429 → 0.8725 and recall@5 '
                '0.9935 (verdict *improved*). A 4.5-point gain on 154 shortlists from one seed is sample-sanity evidence, not a measured improvement. On the fresh '
                'test the same run gave MRR 0.851 → 0.879 (paired difference +0.028, 95 % CI 0.009 to 0.048) and recall@1 +0.047 (95 % CI 0.013 to 0.081), verdict *improved*; the interval covers the choice of shortlists only (`docs/release-verification.md`).</details>'
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
                "`pipe.save_artifact` writes the trained tensors — the last two decoder layers, about 126 MB in float32 on every "
                "device (the cell prints the size) — as "
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
                "if pipe.adapter is None:  # RR-M2: the export must be the model Section 8 evaluated\n"
                "    raise RuntimeError('The pipeline holds the pinned base, not an adapted model (Section 5 or 6 was re-run after Section 7): run Section 7 and Section 8, then this cell.')\n"
                "after = {{r['id']: ranked_shortlist(r) for r in probe_records}}\n"
                "rows = []\n"
                "for r in probe_records:\n"
                "    rows.append({{'id': r['id'], 'query': r['query'], 'gold': r['positive'], 'frozen_order': ' | '.join(d for d, _s in before[r['id']]), 'adapted_order': ' | '.join(d for d, _s in after[r['id']]), 'gold_rank_frozen': next(i + 1 for i, (d, _s) in enumerate(before[r['id']]) if d == r['positive']), 'gold_rank_adapted': next(i + 1 for i, (d, _s) in enumerate(after[r['id']]) if d == r['positive']), 'gold_score_frozen': next(s for d, s in before[r['id']] if d == r['positive']), 'gold_score_adapted': next(s for d, s in after[r['id']] if d == r['positive'])}})\n"
                "    print({{k: rows[-1][k] for k in ('id', 'gold', 'frozen_order', 'adapted_order', 'gold_rank_frozen', 'gold_rank_adapted', 'gold_score_frozen', 'gold_score_adapted')}})\n"
                "single_report = evaluation_report(pipe.rerank(pairs, instruction=INSTRUCTION), sample_kind='one Banking77 test shortlist' if not USE_BYOD else 'one BYOD test shortlist')\n"
                "print({{'batch_report_verdict': single_report['verdict'], 'shortlists_reordered': sum(r['frozen_order'] != r['adapted_order'] for r in rows), 'of': len(rows)}})\n"
                "with open(suffixed('outputs/{stem}_reranking.csv'), 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(rows)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter' + suffix)\n"
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
                "    'preregistered': preregistered,\n"
                "    'test_look': test_look,\n"
                "    'settings_differing': differing,\n"
                "    'grid_selection': grid_selection,\n"
                "    'fresh_test': fresh_result,\n"
                "    'reranking_before_after': rows,\n"
                "    'batch_report': single_report,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': compute_dtype(pipe.device)}},\n"
                "}}\n"
                "with open(suffixed('outputs/{stem}_result.json'), 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "**What to notice:** the gold intent's rank before and after for the three shortlists, the artifact's size, and `reload_parity`.\n\n<details><summary>Check "
                'your reasoning</summary>Scores from the adapted model are on a different scale from the frozen ones — the `yes` share moves for every pair — so '
                'compare ranks, not scores. In the hosted Colab T4 float32 run of notebook blob 2d6e5607 (commit ff6da3a, 18 October 2026; docs/execution-evidence/2026-10-18-ff6da3a/), the evaluation of record the reloaded adapter gave identical scores and the same validation MRR on the 20-query parity subset '
                '(0.975 in memory and reloaded).</details>'
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen reranker already puts the right intent phrase first for most unseen messages (recall@1 in the "
        "seventies against a lexical baseline and a random floor of 16.7 %), and a bounded listwise fine-tuning of the "
        "last two decoder layers on 231 shortlists lifted recall@1 on the exposed development test by a few points (+4.5 in "
        "the hosted Colab T4 float32 run of notebook blob 2d6e5607 (commit ff6da3a, 18 October 2026; docs/execution-evidence/2026-10-18-ff6da3a/), the evaluation of record; that split informed an earlier learning-rate choice, so read it as a "
        "development number). On the fresh test the pre-registered settings gave MRR 0.851 → 0.879 on that run (paired difference "
        "+0.028, 95 % CI 0.009 to 0.048; `docs/release-verification.md`), with an adapter of about 126 MB in float32 that reloads to "
        "identical scores. "
        "That is the claim: the adaptation contract can adapt the reranker to a relevance judgement end to end on a real "
        "labelled corpus, and the numbers it produces are read against a random floor, a lexical baseline and the frozen "
        "model rather than in isolation, under one tie policy at a time.\n\n"
        "The fresh test is 385 six-entry shortlists (the exposed development test 154) from one seeded split of one corpus "
        "with a bootstrap interval over shortlists only (no estimate of split or seed variation), and Banking77 may overlap the model's training data (see the introduction); "
        "recall@k and MRR say whether the gold document is ordered first within a shortlist the dataset supplied, not how "
        "the model would rank a real candidate pool. The adapter changes the last layers, which every pair shares, so "
        "every score shifts and scores are not comparable across the frozen and adapted models; nothing here measures the "
        "effect on other judgements. The model now runs in float32 on every device (amendment 2); the float32 T4 record equals the CPU float32 build at printed precision "
        "(the earlier bfloat16 T4 runs had differed by up to one or two points of recall@1). A cross-encoder scores one pair per forward pass, so evaluation cost grows with the "
        "shortlist length — a first-stage retriever (the sibling embedding pipeline) is what makes the shortlist short.\n\n"
        "Three things to carry to real data. **Floors first:** the random floor and the lexical baseline on *your* "
        "shortlists, under the same tie policy, are the numbers to read before any reranker's; a lexical baseline near the frozen model means the "
        "task is mostly keyword matching. **Leakage:** de-duplicate queries across splits (the contract does this "
        "case-insensitively) and split by user or session when your queries come from one. **Negatives:** the quality of "
        "the shortlist decides what the reranker learns — draw negatives from the retriever you will deploy behind, not "
        "at random.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled corpus, "
        "validate the demonstrated dataset contract without leakage, execute the reranking contract and a bounded listwise "
        "fine-tuning, evaluate by ranking metrics against a random floor, a lexical baseline and the frozen model on a "
        "fresh test scored only under pre-registered settings, and emit the shown machine-readable artifacts — without the repository being reachable. It "
        "does **not** establish benchmark superiority, ranking quality on any other task, a usable acceptance threshold, "
        "or production fitness.\n\n"
        "## Activity: how much does the second trainable layer buy?\n\n"
        "Run this only after the default Run all has finished. It changes a Section 7 field, so it is **exploratory**: Section 8 reads validation and "
        "the exposed development test only, never the fresh test, and writes `_exploratory_<n>` files beside the pre-registered ones.\n\n"
        "1. **Predict.** Write down the recall@1 you expect from the adapted reranker if only the **last one** decoder layer trains "
        "(half the trainable parameters and half the adapter), on the exposed development test. Above or below the default's? By how much?\n"
        "2. **Change.** In Section 7 set `TRAINABLE_LAYERS = 1` and, to save time, `RUN_GRID = False` (it uses `LEARNING_RATE` = 2e-5); leave every other field as it is.\n"
        "3. **Run.** Run Section 7, then Section 8. Every adaptation starts from the pinned base (`started_from` in Section 7's output), "
        "so this is a fresh experiment, not more training on top of the first run.\n"
        "4. **Observe.** Section 7's epoch 0 (`note: frozen model`) must print the same validation metrics as in the default run — the same "
        "starting point. Then read `test_look` (*not scored (exploratory)*) and `delta_vs_frozen` in Section 8 and compare it with the default's.\n"
        "5. **Explain.** Say in one sentence what the second layer added, and whether that is worth twice the adapter size for this task.\n\n"
        "To put the notebook back to the pre-registered state, set `TRAINABLE_LAYERS = 2` and `RUN_GRID = True` and run Sections 7, 8 and 9 again "
        "(an identical pre-registered scoring is labelled a reproduction of the recorded evaluation).\n\n"
        "<details><summary>Check your reasoning</summary>There is no recorded run of this setting, so there is no worked number to match. "
        "The default itself moved exposed-development recall@1 by only +4.5 points on the recorded Colab T4 run (7 of 154 shortlists), because the frozen reranker already "
        "puts the right intent first in about three of four shortlists; one layer has less capacity to reorder the rest, so a change no larger "
        "than the default's — possibly none — is the reasonable expectation. On 154 shortlists from one seed a difference of a point or two is "
        "within run-to-run noise: what you can conclude is the direction and the adapter-size trade-off, not a ranking of the two settings.</details>\n\n"
        "**More experiments (same procedure; each is exploratory and never scores the fresh test):** set `TRAIN_CANDIDATES = 6` to train on the whole shortlist "
        "and compare; set `EPOCHS = 3` and watch whether validation MRR keeps rising or turns (the best epoch is kept either way); change "
        "`INSTRUCTION` in Section 4 and re-run from Section 5 to re-read the frozen numbers — the judgement is instruction-conditioned; or bring "
        "your own shortlists through BYOD (re-run from Section 4) and read the lexical baseline before the adapted number. On the Banking77 "
        "sample with changed Section 7 settings, Section 8 reports the verdict instead of stopping. A smaller `BATCH_SIZE` to fit memory is exploratory too.\n\n"
        '## Troubleshooting\n\n- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google '
        'Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 '
        'again; a complete environment built from the same lock is reused, an incomplete one is finished. If it repeats, the network is blocking or altering '
        '`files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and '
        'choose **Run all**.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it '
        'keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message '
        'names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again.\n- **Section 4 reports a size or SHA-256 mismatch for a '
        'Banking77 file** — the download from `raw.githubusercontent.com` was cut short or altered. Run Section 4 again; a verified cached file is reused.\n- '
        '**Sections 6–8 are very slow** — a cross-encoder runs one forward pass per query–candidate pair (924 for the test split); on CPU that is minutes per '
        'evaluation. Switch to a T4 GPU.\n- **Your numbers differ slightly from the recorded run** — the recorded run (2026-10-18, T4, float32) equals the CPU float32 build at printed precision, so the default path should '
        'reproduce it; if a number differs, read `settings_differing` and the package versions Section 2 prints. The comparison between systems is the result.\n- **BYOD: a column or record refusal** — the message names the rule: columns `id`, `query`, `positive`, '
        '`negatives` (separated by ` | `), 1..15 distinct negatives per record; set `INSTRUCTION` to describe your relevance judgement.\n- '
        '**BYOD: "the dataset has N distinct queries … 12..10002 distinct queries are required"** — the split keeps 20 % for test and 15 % for '
        'validation and needs 8 for training, and each evaluated split is scored in one call of at most 2,000 records: add records, or split a larger '
        'file into runs of at most 10,002 distinct queries.\n- '
        '**BYOD: "BYOD_PATH … is not a file"** — the path is relative to the working directory printed in the message; give one .csv, .json or .jsonl file.\n- '
        '**BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the file in the runtime and set `BYOD_PATH` to its path.\n- '
        '**BYOD: "Upload exactly one file"** — the dialog was cancelled or several files were chosen; run the cell again.\n- **Section 8 or 9 says the '
        'pipeline holds the pinned base** — you re-ran Section 5 or 6 after Section 7, and they put the pretrained weights back so their numbers are '
        "the frozen model's. Run Section 7 again, then Section 8 and Section 9.\n- **Section 6 stops on the Banking77 sample** — the frozen "
        'model did not beat the random floor; that did not happen in the recorded runs. Run all again from the top; if it repeats, report it with the printed numbers.\n- **Section 8 says `not scored (exploratory)`** — a '
        'setting differs from the pre-registration (`settings_differing` names it), or Section 4 was re-run after Section 7. Restore the defaults and run '
        'Sections 7 and 8 again; the exploratory files stay beside the pre-registered ones.\n- **Section 4 says the fresh test is not the pinned draw** — the '
        'corpus or the sampling code differs from the pinned one; Run all again from the top.\n\n## Glossary\n\n- **Reranker / '
        'cross-encoder:** a model that reads the query and one candidate together and scores their relevance; accurate, but one forward pass per pair.\n- '
        '**Shortlist:** the small candidate list a first-stage retriever would hand the reranker; here the positive plus five negatives.\n- **Hard negatives:** '
        "wrong candidates chosen to look right (here, the highest word overlap with the message).\n- **Relevance score:** the `yes` share of the model's yes/no "
        'answer for one pair; it orders candidates and is not a calibrated probability.\n- **Instruction:** the one-line relevance judgement every pair carries; '
        'the model is instruction-conditioned.\n- **Recall@k / MRR:** the share of queries whose positive is in the top k; the mean of 1 / rank of the positive.\n- '
        "**Random floor / lexical baseline:** a random ordering's expected score; ordering by shared-word (Jaccard) overlap.\n- **Listwise loss:** the "
        'cross-entropy of the positive over its whole candidate list, rather than one pair at a time.\n- **Frozen / adapted / pinned base:** the packaged model; '
        'the model after Section 7; the verified packaged weights every adaptation starts from (`restore_base`).\n- **Isolated environment:** the separate '
        'Python environment Section 1 builds from the hash lock; every later cell runs there.\n- **Exposed development test / fresh test:** a test split that has informed a choice (shown, decides nothing); one never scored before, scored only under pre-registered settings.\n- **Tie policy:** how a candidate scored exactly like the gold one is ranked: against it (pessimistic) or at random (expected).\n\n## Conclusion (your notes)\n\nComplete these in your own words; '
        "the recorded run's values are in the **Check your reasoning** answers above.\n\n- On the 385 fresh-test shortlists the frozen reranker reached recall@1 ___ "
        "against the lexical baseline's ___ and the random floor's ___.\n- Listwise fine-tuning moved recall@1 to ___ and MRR to ___ (verdict: ___), which I "
        'read as ___.\n- The number I would not trust on its own is ___, because ___.\n- Before adapting on my own shortlists I would draw negatives from ___, '
        'de-duplicate by ___, and compare against ___.\n\n'
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/qwen3-reranker-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/qwen3-reranker-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/qwen3-reranker-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/QwenLM/Qwen3-Embedding\n"
        "- Pre-registration: https://github.com/kurtvalcorza/qwen3-reranker-pipeline/blob/main/docs/qwen3-reranker-preregistration.md\n\n"
        "[1] Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, F. Huang and J. Zhou, \"Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models,\" arXiv:2506.05176, 2025. [Online]. Available: https://arxiv.org/abs/2506.05176\n\n"
        "[2] I. Casanueva, T. Temčinas, D. Gerz, M. Henderson and I. Vulić, \"Efficient Intent Detection with Dual Sentence Encoders,\" in Proc. 2nd Workshop on NLP for Conversational AI, ACL, 2020, pp. 38–45. [Online]. Available: https://arxiv.org/abs/2003.04807 (Banking77, CC BY 4.0)\n\n"
        "[3] N. Muennighoff, N. Tazi, L. Magne and N. Reimers, \"MTEB: Massive Text Embedding Benchmark,\" in Proc. EACL, 2023, pp. 2014–2037. [Online]. Available: https://arxiv.org/abs/2210.07316\n\n"
        "[4] DIMER Notebook Specification 2.5 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)."
    ),
}
