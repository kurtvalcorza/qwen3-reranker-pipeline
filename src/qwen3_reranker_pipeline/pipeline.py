from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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


@dataclass
class Qwen3RerankerPipeline:
    """Pointwise reranker. `_runner` maps pair bodies to ([no, yes] last-position logits, token counts)."""

    _runner: Callable[[list[str]], tuple[np.ndarray, list[int]]]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Qwen3RerankerPipeline:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), dict(local_files_only=True)
        elif allow_download:
            source, kwargs = MODEL_ID, dict(revision=MODEL_REVISION)
        else:
            raise FileNotFoundError(f"no verified snapshot at {root} and allow_download=False")
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

        return cls(runner, resolved_device)

    def rerank(
        self,
        pairs: Sequence[Sequence[str]],
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> dict[str, Any]:
        """Score up to MAX_PAIRS (query, document) pairs; `scores` align with `pairs`; no threshold."""
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
