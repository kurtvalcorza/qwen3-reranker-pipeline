"""Row 73 round 3 (pre-registration amendment 2): the model is loaded, run and trained in float32 on every device.

The torch-backed tests use tiny stand-in modules (no model weights) and are skipped where torch is not installed (CI's
lightweight job); the source-level checks always run.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

from qwen3_reranker_pipeline import pipeline as pl

ROOT = Path(__file__).resolve().parents[1]


def test_compute_dtype_is_float32_on_every_device():
    assert pl.compute_dtype("cuda:0") == pl.compute_dtype("cpu") == "float32" == pl.COMPUTE_DTYPE == pl.ADAPTER_DTYPE
    with pytest.raises(ValueError):
        pl.compute_dtype("")
    source = inspect.getsource(pl.Qwen3RerankerPipeline.from_pretrained)
    assert "bfloat16" not in source and "compute_dtype(resolved_device)" in source
    assert "check_float32_training(params)" in inspect.getsource(pl.Qwen3RerankerPipeline.adapt)


def test_notebook_records_float32_not_a_device_dependent_dtype():
    notebook = json.loads((ROOT / "tutorials" / "qwen3_reranker_colab.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in notebook["cells"] if c["cell_type"] == "code")
    assert "'bfloat16' if" not in code and "dtype = compute_dtype(device)" in code
    assert (ROOT / "docs" / "qwen3-reranker-preregistration-amendment-2.md").is_file()


def test_cuda_code_path_loads_float32(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    seen = {}

    class _Tok:
        def convert_tokens_to_ids(self, token):
            return pl.YES_TOKEN_ID if token == pl.YES_TOKEN else pl.NO_TOKEN_ID

        def encode(self, text, add_special_tokens=False):
            return [1, 2]

    class _Model:
        def to(self, device):
            seen["device"] = device
            return self

        def eval(self):
            return self

    def _model_from_pretrained(source, dtype=None, **kw):
        seen["dtype"] = dtype
        return _Model()

    fake = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tok()),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=_model_from_pretrained),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    pipe = pl.Qwen3RerankerPipeline.from_pretrained(device="cuda:0", weights_dir=tmp_path, allow_download=True)
    assert seen == {"dtype": torch.float32, "device": "cuda:0"} and pipe.device == "cuda:0"


def test_trainable_parameters_and_adamw_state_are_float32():
    torch = pytest.importorskip("torch")
    layer = torch.nn.Linear(4, 4)
    params = list(layer.parameters())
    optimiser = torch.optim.AdamW(params, lr=1e-3)
    assert pl.check_float32_training(params)["optimizer_state"] == "float32"
    layer(torch.ones(2, 4)).sum().backward()
    optimiser.step()
    assert pl.check_float32_training(params, optimiser)["trainable"] == "float32"
    state = optimiser.state[params[0]]
    assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    state["exp_avg"] = state["exp_avg"].to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="optimiser state must be float32"):
        pl.check_float32_training(params, optimiser)
    with pytest.raises(RuntimeError, match="trainable parameters must be float32"):
        pl.check_float32_training([p.detach().to(torch.bfloat16) for p in params])


def _stand_in_pipeline(torch, dtype):
    class _Stack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(pl.DECODER_LAYERS)])

    module = _Stack().to(dtype)
    names = [n for n, _p in module.named_parameters() if n.startswith(f"model.layers.{pl.DECODER_LAYERS - 1}.")]
    adapter = {"trainable_layers": 1, "trainable_names": names, "history": [], "lr": 2e-5, "precision": {"trainable": "float32"}}
    return pl.Qwen3RerankerPipeline(lambda bodies: None, "cuda:0", adapter=adapter, _model=module, _tokenizer=object())


def test_saved_adapter_is_float32_and_reloads(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file, save_file

    pipe = _stand_in_pipeline(torch, torch.float32)
    out = pipe.save_artifact(tmp_path / "adapter")
    tensors = load_file(str(out / pl.ARTIFACT_WEIGHTS_NAME))
    assert {str(v.dtype) for v in tensors.values()} == {"torch.float32"}
    manifest = pipe.load_artifact(out)
    assert manifest["adapter"]["precision"]["trainable"] == "float32"
    # a bfloat16 adapter is refused on save and on reload
    with pytest.raises(RuntimeError, match="adapter tensors must be float32"):
        _stand_in_pipeline(torch, torch.bfloat16).save_artifact(tmp_path / "bf16")
    weights = out / pl.ARTIFACT_WEIGHTS_NAME
    save_file({k: v.to(torch.bfloat16) for k, v in tensors.items()}, str(weights), metadata={"format": "pt"})
    manifest_path = out / pl.ARTIFACT_MANIFEST_NAME
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["files"][0].update(bytes=weights.stat().st_size, sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact tensors must be float32"):
        _stand_in_pipeline(torch, torch.float32).load_artifact(out)
