from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "qwen3-reranker-0.6b"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Prompt contract from the pinned upstream README ("Using Transformers"): a fixed system prompt, the
# instruction/query/document block, and the assistant prefix with an empty <think> block; the score is
# the softmax over the "no"/"yes" logits at the last position. Token ids are pinned by the snapshot's
# 1_LogitScore/config.json and cross-checked against the tokenizer at load time.
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
YES_TOKEN, NO_TOKEN = "yes", "no"
YES_TOKEN_ID, NO_TOKEN_ID = 9693, 2152
MAX_TEXT_TOKENS = 8192  # total prompt length incl. prefix/suffix; the pair is truncated longest-first to fit
MAX_TEXT_CHARS = 100_000  # per query or document, pre-tokenisation guard
MAX_PAIRS = 32  # (query, document) pairs per rerank() call
SCORE_KIND = "relevance score, not a calibrated probability"
WEIGHT_FILE = "model.safetensors"
WEIGHT_SHA256 = (
    "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b"  # manifest digest of WEIGHT_FILE
)
PARAMETER_COUNT = 595_776_512  # Qwen3ForCausalLM with the output projection tied to the token embeddings
DECODER_LAYERS = 28  # config.json num_hidden_layers
DEFAULT_TRAINABLE_LAYERS = 2  # the last two decoder layers (31,461,888 parameters)
MAX_TRAIN_TOKENS = 192  # training-only prompt ceiling (inference truncates the pair to MAX_TEXT_TOKENS)
MAX_TRAIN_CANDIDATES = 4  # per query: the positive plus the first negatives, scored together in one list
MAX_EVAL_RECORDS = 2_000
MIN_SCORED_RECORDS = 50  # below this a scored dataset is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.qwen3-reranker-0.6b.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest.hexdigest()} != manifest {entry['sha256']}")
    return manifest


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def format_pair(query: str, document: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
    """Upstream `format_instruction`: the user-turn body between PREFIX and SUFFIX."""
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


INPUT_SCHEMA: dict[str, Any] = {
    "input": "sequence of (query, document) pairs of two non-empty str; one score is returned per pair",
    "pairs": [1, MAX_PAIRS],
    "text_chars": [1, MAX_TEXT_CHARS],
    "prompt_tokens": [1, MAX_TEXT_TOKENS],
    "score_range": [0.0, 1.0],
    "preprocessing": (
        "each pair becomes '<Instruct>: <instruction>\\n<Query>: …\\n<Document>: …' between the fixed "
        "upstream PREFIX and SUFFIX, truncated longest-first to fit MAX_TEXT_TOKENS; the score is the "
        "two-way softmax share of the yes logit against the no logit at the last position"
    ),
}


def _check_inputs(pairs: Any, instruction: str) -> list[tuple[str, str]]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the pairs as a list."""
    if isinstance(pairs, str | bytes) or not isinstance(pairs, Sequence):
        raise TypeError("pairs must be a list of (query, document) pairs")
    if not 1 <= len(pairs) <= MAX_PAIRS:
        raise ValueError(f"pairs must hold 1..{MAX_PAIRS} items, got {len(pairs)}")
    for i, pair in enumerate(pairs):
        if isinstance(pair, str | bytes) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise TypeError(f"pairs[{i}] must be a (query, document) pair of two str")
        for name, text in zip(("query", "document"), pair, strict=True):
            if not isinstance(text, str):
                raise TypeError(f"pairs[{i}] {name} must be str, got {type(text).__name__}")
            if not text.strip():
                raise ValueError(f"pairs[{i}] {name} is empty")
            if len(text) > MAX_TEXT_CHARS:
                raise ValueError(f"pairs[{i}] {name} has {len(text)} chars; ceiling is {MAX_TEXT_CHARS}")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty str")
    return [(pair[0], pair[1]) for pair in pairs]


def validate_inputs(
    pairs: Sequence[Sequence[str]],
    instruction: str = DEFAULT_INSTRUCTION,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-pair observations, verdict).

    Rejection is reported by raising exactly as ``rerank`` would — both route through
    ``_check_inputs``. Prompt-level truncation cannot be observed here because it happens inside
    the tokenizer; ``rerank`` reports it in ``truncated``.
    """
    checked = _check_inputs(pairs, instruction)
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per pair")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[i] if names else f"pair-{i}",
                "query_chars": len(query),
                "document_chars": len(document),
            }
            for i, (query, document) in enumerate(checked)
        ],
        "n_pairs": len(checked),
        "instruction": instruction,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any], judgements: Sequence[Any] | None = None, *, sample_kind: str = "synthetic"
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even though no metric exists here.

    The repository ships no ranking-metric helper, so the verdict is always ``not-measurable``
    (EVAL9), including when ``judgements`` is supplied: the parameter exists for interface parity
    with the fleet's other pipelines and is recorded in ``reason`` rather than scored. Inventing
    nDCG or MRR here would hide the fact that a real evaluation needs judged candidates over many
    queries and the caller's own metric code.
    """
    scores = result["scores"]
    supplied = judgements is not None
    return {
        "task": "pointwise query-document relevance reranking",
        "score_semantics": (
            f"{SCORE_KIND}: the softmax share of the yes logit against the no logit, in [0, 1]; it "
            "orders candidates for one query, is not comparable as an absolute value across queries "
            "or instructions, and carries no shipped acceptance threshold"
        ),
        "sample_kind": sample_kind,
        "n_pairs": len(scores),
        "metrics": [],
        "baselines": [],
        "verdict": "not-measurable",
        "reason": (
            "ranking quality needs relevance judgements and the repository ships no metric helper"
            + (
                "; judgements were supplied but no metric helper exists to score them here"
                if supplied
                else "; the evaluated sample carries none"
            )
        ),
        "needs": (
            "per-query relevance judgements (binary or graded) over enough queries to state a "
            "dispersion, scored with the caller's own nDCG@k, MRR or precision@k code; a single "
            "query's ordering is a plumbing check, not a retrieval measurement"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


@dataclass
class Qwen3RerankerPipeline:
    """Pointwise reranker. `_runner` maps pair bodies to ([no, yes] last-position logits, token counts)."""

    _runner: Callable[[list[str]], tuple[np.ndarray, list[int]]]
    device: str
    adapter: dict[str, Any] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Qwen3RerankerPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), dict(local_files_only=True)
        elif allow_download:
            source, kwargs = MODEL_ID, dict(revision=MODEL_REVISION)
        else:
            raise FileNotFoundError(f"no verified snapshot at {root} and allow_download=False")
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(
            source, padding_side="left", trust_remote_code=False, **kwargs
        )
        ids = (tokenizer.convert_tokens_to_ids(YES_TOKEN), tokenizer.convert_tokens_to_ids(NO_TOKEN))
        if ids != (YES_TOKEN_ID, NO_TOKEN_ID):
            raise RuntimeError(f"tokenizer maps yes/no to {ids}, expected {(YES_TOKEN_ID, NO_TOKEN_ID)}")
        model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype, trust_remote_code=False, **kwargs)
        model = model.to(resolved_device).eval()
        prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
        suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
        body_budget = MAX_TEXT_TOKENS - len(prefix_ids) - len(suffix_ids)

        def runner(bodies: list[str]) -> tuple[np.ndarray, list[int]]:
            enc = tokenizer(
                bodies,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=body_budget,
            )
            enc["input_ids"] = [prefix_ids + row + suffix_ids for row in enc["input_ids"]]
            batch = tokenizer.pad(enc, padding=True, return_tensors="pt")
            batch = batch.to(resolved_device)
            with torch.inference_mode():
                last = model(**batch).logits[:, -1, :]
            pair_logits = torch.stack([last[:, NO_TOKEN_ID], last[:, YES_TOKEN_ID]], dim=1)
            counts = batch["attention_mask"].sum(dim=1).tolist()
            return pair_logits.float().cpu().numpy(), [int(c) for c in counts]

        return cls(runner, resolved_device, _model=model, _tokenizer=tokenizer)

    def _validate(self, pairs: Any, instruction: str) -> list[tuple[str, str]]:
        return _check_inputs(pairs, instruction)

    def rerank(
        self,
        pairs: Sequence[Sequence[str]],
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> dict[str, Any]:
        """Score up to MAX_PAIRS (query, document) pairs; `scores` align with `pairs`; no threshold."""
        pairs = self._validate(pairs, instruction)
        bodies = [format_pair(q, d, instruction) for q, d in pairs]
        logits, n_tokens = self._runner(bodies)
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != (len(pairs), 2):
            raise RuntimeError(f"backend returned {logits.shape}, expected ({len(pairs)}, 2)")
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        scores = probs[:, 1]
        return {
            "scores": [float(s) for s in scores],
            "ranking": [int(i) for i in np.argsort(-scores, kind="stable")],
            "score_kind": SCORE_KIND,
            "instruction": instruction,
            "n_tokens": list(n_tokens),
            "truncated": [n >= MAX_TEXT_TOKENS for n in n_tokens],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    # ---- adaptation -----------------------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._tokenizer

    def _score_pairs(self, pairs: Sequence[tuple[str, str]], instruction: str) -> list[float]:
        """Score any number of pairs through the public contract, MAX_PAIRS at a time."""
        scores: list[float] = []
        for start in range(0, len(pairs), MAX_PAIRS):
            scores.extend(self.rerank(list(pairs[start : start + MAX_PAIRS]), instruction)["scores"])
        return scores

    def evaluate(
        self, records: Sequence[Mapping[str, Any]], *, instruction: str = DEFAULT_INSTRUCTION
    ) -> dict[str, Any]:
        """Rerank every record's candidate list (its positive followed by its negatives) with `rerank` and
        read the rank of the positive: recall@1 / recall@3 / recall@5 and MRR, ties against the positive."""
        from .metrics import candidate_list, rank_of_positive, ranking_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        _check_inputs([("x", "y")], instruction)
        started = time.perf_counter()
        pairs: list[tuple[str, str]] = []
        sizes = []
        for record in checked:
            candidates = candidate_list(record)
            sizes.append(len(candidates))
            pairs.extend((record["query"], doc) for doc in candidates)
        scores = self._score_pairs(pairs, instruction)
        ranks = []
        cursor = 0
        for size in sizes:
            ranks.append(rank_of_positive(scores[cursor : cursor + size], 0))
            cursor += size
        metrics = ranking_metrics(ranks, sizes)
        metrics.update(
            {
                "instruction": instruction,
                "n_pairs": len(pairs),
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    @staticmethod
    def lexical_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """The no-model floor: candidate lists ordered by token overlap with the query (see metrics.py)."""
        from .metrics import lexical_baseline
        from .samples import validate_dataset

        return lexical_baseline(
            validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        )

    def _trainable_names(self, trainable_layers: int) -> list[str]:
        if not isinstance(trainable_layers, int) or not 1 <= trainable_layers <= DECODER_LAYERS:
            raise ValueError(f"trainable_layers must be an int in 1..{DECODER_LAYERS}")
        model, _ = self._require_model()
        first = DECODER_LAYERS - trainable_layers
        prefixes = tuple(f"model.layers.{k}." for k in range(first, DECODER_LAYERS))
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        epochs: int = 2,
        lr: float = 2e-5,
        batch_size: int = 4,
        trainable_layers: int = DEFAULT_TRAINABLE_LAYERS,
        train_candidates: int = MAX_TRAIN_CANDIDATES,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded listwise fine-tuning on validated query / positive / negatives records.

        Only the last `trainable_layers` decoder layers train (2 by default; the token embeddings — which the
        output projection shares — the earlier layers and the final norm stay frozen). Each query contributes
        one list: its positive and its first `train_candidates - 1` negatives, formatted exactly as `rerank`
        formats them; the relevance logit of every candidate is the yes-minus-no logit at the last position,
        and the loss is the cross-entropy of the positive over its list (listwise softmax). `batch_size`
        counts queries per step; AdamW at a fixed learning rate, gradient clipping at 1.0, seeded shuffling,
        no scheduler; prompts are truncated to MAX_TRAIN_TOKENS **during training only**. Epoch 0 records the
        frozen model's validation ranking metrics; the epoch with the highest validation MRR is kept."""
        from .metrics import candidate_list
        from .samples import validate_dataset

        if not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 16:
            raise ValueError("batch_size must be an int in 1..16")
        if not isinstance(train_candidates, int) or not 2 <= train_candidates <= 8:
            raise ValueError("train_candidates must be an int in 2..8")
        _check_inputs([("x", "y")], instruction)
        names = self._trainable_names(trainable_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        )
        import torch

        torch.manual_seed(seed)
        model, tokenizer = self._require_model()
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        device = torch.device(self.device)
        prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
        suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
        body_budget = MAX_TRAIN_TOKENS - len(prefix_ids) - len(suffix_ids)

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            keep = ("recall@1", "recall@3", "recall@5", "mrr", "n_pairs")
            return {k: v for k, v in self.evaluate(val_checked, instruction=instruction).items() if k in keep}

        def relevance(bodies: list[str]) -> torch.Tensor:
            enc = tokenizer(
                bodies,
                padding=False,
                truncation="longest_first",
                max_length=body_budget,
                return_attention_mask=False,
            )
            enc["input_ids"] = [prefix_ids + row + suffix_ids for row in enc["input_ids"]]
            batch = tokenizer.pad(enc, padding=True, return_tensors="pt").to(device)
            last = model(**batch).logits[:, -1, :].float()
            return last[:, YES_TOKEN_ID] - last[:, NO_TOKEN_ID]

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress:
            progress(entry)
        best_mrr = entry["val"]["mrr"] if entry["val"] else -math.inf
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        lists = [candidate_list(r)[:train_candidates] for r in train_checked]
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(train_checked), generator=generator).tolist()
            losses = []
            for start in range(0, len(order), batch_size):
                chosen = order[start : start + batch_size]
                bodies, spans = [], []
                for i in chosen:
                    spans.append((len(bodies), len(lists[i])))
                    bodies.extend(
                        format_pair(train_checked[i]["query"], doc, instruction) for doc in lists[i]
                    )
                logits = relevance(bodies)
                loss = torch.stack(
                    [
                        torch.nn.functional.cross_entropy(
                            logits[a : a + n][None], torch.zeros(1, dtype=torch.long, device=device)
                        )
                        for a, n in spans
                    ]
                ).mean()
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimiser.step()
                losses.append(float(loss.detach()))
            model.eval()
            entry = {"epoch": epoch, "train_loss": sum(losses) / max(len(losses), 1), "val": score_val()}
            history.append(entry)
            if progress:
                progress(entry)
            current = entry["val"]["mrr"] if entry["val"] else math.inf
            if current > best_mrr or not entry["val"]:
                best_mrr = current
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                best_epoch = epoch
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "objective": "listwise cross-entropy of the positive over its candidates (yes-minus-no logit)",
            "instruction": instruction,
            "trainable_layers": trainable_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation MRR" if val_checked else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "train_candidates": train_candidates,
            "max_train_tokens": MAX_TRAIN_TOKENS,
            "n_train": len(train_checked),
            "n_train_pairs": sum(len(lst) for lst in lists),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted decoder-layer tensors as safetensors with a manifest naming the base."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest and digest, then overwrite exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        entry = manifest["files"][0]
        weights_path = root / entry["path"]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        model, _ = self._require_model()
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != manifest["tensors"]:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        for key, value in tensors.items():
            if key not in state or not key.startswith("model.layers."):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable decoder-layer tensor of the base"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key}: shape {tuple(value.shape)} != {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        model.load_state_dict(merged, strict=True)
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Qwen3RerankerPipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline
