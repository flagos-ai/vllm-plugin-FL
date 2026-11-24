# SPDX-License-Identifier: Apache-2.0
"""Real-import contracts for DeepSeek-V4 initiation and NPU install paths.

``vllm_fl/models/deepseek_v4.py`` cannot be imported through its package on a
build without upstream DeepSeek-V4, so the actual source file is loaded with
``importlib`` while every upstream dependency is provided as an explicit stub.
The module top level (including the NPU conditionals) is executed for real.
"""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_REL = "vllm_fl/models/deepseek_v4.py"


def _source_path():
    candidates = []
    override = os.environ.get("VLLM_FL_SOURCE_ROOT")
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [Path("/tmp/flsrc"), Path.cwd(), Path(__file__).resolve().parents[2]]
    )
    for root in candidates:
        target = root / SOURCE_REL
        if target.is_file():
            return target
    raise FileNotFoundError("could not locate " + SOURCE_REL)


def _ensure_pkg(monkeypatch, name):
    import sys
    import types

    existing = sys.modules.get(name)
    if isinstance(existing, types.ModuleType):
        return existing
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    return module


def _install(monkeypatch, name, **attrs):
    import sys
    import types

    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _cfg(
    *,
    ratios=(0, 1, 4, 128),
    layers=4,
    sliding=128,
    max_len=128,
    topk=512,
):
    hf = SimpleNamespace(
        compress_ratios=list(ratios),
        num_hidden_layers=layers,
        sliding_window=sliding,
        index_topk=topk,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, max_model_len=max_len)
    )


def _load(monkeypatch, device_type, npu_event=None):
    base_calls = []

    class _BaseAttention:
        def __init__(self, *args, **kwargs):
            base_calls.append((args, kwargs))
            self.init_args = args
            self.init_kwargs = kwargs
            self.event_seen = torch.cuda.Event

    class _SWABuilder:
        pass

    class _SWABackend:
        pass

    class _MLABackend:
        pass

    lm_calls = []

    class _BaseCausalLM:
        def __init__(self, **kwargs):
            lm_calls.append(kwargs)
            self.init_kwargs = kwargs

    _ensure_pkg(monkeypatch, "vllm.models.deepseek_v4.xpu")
    xpu_model = _install(
        monkeypatch,
        "vllm.models.deepseek_v4.xpu.model",
        DeepseekV4ForCausalLM=_BaseCausalLM,
        DeepseekV4XPUAttention="ORIGINAL",
    )
    _install(
        monkeypatch,
        "vllm.models.deepseek_v4.attention",
        DeepseekV4Attention=_BaseAttention,
    )
    _install(
        monkeypatch,
        "vllm.models.deepseek_v4.sparse_mla",
        DeepseekV4FlashMLAMetadata=object,
        DeepseekV4SparseMLABackend=_MLABackend,
        DeepseekV4SparseMLAMetadataBuilder=_SWABuilder,
    )
    _install(
        monkeypatch,
        "vllm.v1.attention.backends.mla.sparse_swa",
        DeepseekSparseSWABackend=_SWABackend,
        DeepseekSparseSWAMetadata=object,
        DeepseekSparseSWAMetadataBuilder=_SWABuilder,
    )
    _install(
        monkeypatch,
        "vllm.forward_context",
        get_forward_context=lambda: SimpleNamespace(attn_metadata=None),
    )
    _install(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(device_type=device_type),
    )
    if npu_event is not None:
        monkeypatch.setattr(
            torch, "npu", SimpleNamespace(Event=npu_event), raising=False
        )

    hc_head = None
    if device_type == "npu":

        class _HCHeadOp:
            pass

        hc_head = _HCHeadOp
        _install(monkeypatch, "vllm.model_executor.layers.mhc", HCHeadOp=hc_head)

    path = _source_path()
    name = "_w17_dsv4_" + device_type
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return SimpleNamespace(
        module=module,
        xpu_model=xpu_model,
        hc_head=hc_head,
        base_attention=_BaseAttention,
        base_calls=base_calls,
        lm_calls=lm_calls,
        event=npu_event,
    )


def test_attention_init_positional_config_swaps_event(monkeypatch):
    sentinel = object()
    probe = _load(monkeypatch, "cuda", npu_event=sentinel)
    original_event = torch.cuda.Event
    cfg = _cfg()

    attn = probe.module.DeepseekV4FLAttention(cfg, 3)

    assert attn.init_args == (cfg, 3)
    assert attn.init_kwargs == {}
    assert attn.event_seen is sentinel
    assert torch.cuda.Event is original_event
    assert len(probe.base_calls) == 1


def test_attention_init_keyword_config_swaps_event(monkeypatch):
    sentinel = object()
    probe = _load(monkeypatch, "cuda", npu_event=sentinel)
    original_event = torch.cuda.Event
    cfg = _cfg()

    attn = probe.module.DeepseekV4FLAttention(vllm_config=cfg, prefix="p")

    assert attn.init_kwargs == {"vllm_config": cfg, "prefix": "p"}
    assert attn.event_seen is sentinel
    assert torch.cuda.Event is original_event


def test_attention_init_requires_a_config(monkeypatch):
    probe = _load(monkeypatch, "cuda", npu_event=object())
    with pytest.raises(AssertionError):
        probe.module.DeepseekV4FLAttention(prefix="p")
    assert probe.base_calls == []


def test_attention_init_validates_before_base(monkeypatch):
    probe = _load(monkeypatch, "cuda", npu_event=object())
    with pytest.raises(ValueError):
        probe.module.DeepseekV4FLAttention(_cfg(max_len=256))
    assert probe.base_calls == []


def test_attention_init_restores_event_when_base_raises(monkeypatch):
    probe = _load(monkeypatch, "cuda", npu_event=object())
    original_event = torch.cuda.Event

    def _boom(self, *args, **kwargs):
        raise RuntimeError("base init failed")

    monkeypatch.setattr(probe.base_attention, "__init__", _boom)
    with pytest.raises(RuntimeError):
        probe.module.DeepseekV4FLAttention(_cfg())
    assert torch.cuda.Event is original_event


def test_causal_lm_forwards_config_and_prefix(monkeypatch):
    probe = _load(monkeypatch, "cuda")
    cfg = _cfg()
    model = probe.module.DeepseekV4ForCausalLM(vllm_config=cfg, prefix="layer")
    assert model.init_kwargs == {"vllm_config": cfg, "prefix": "layer"}


def test_causal_lm_defaults_prefix_and_validates_first(monkeypatch):
    probe = _load(monkeypatch, "cuda")
    cfg = _cfg()
    model = probe.module.DeepseekV4ForCausalLM(vllm_config=cfg)
    assert model.init_kwargs["prefix"] == ""
    assert probe.lm_calls == [{"vllm_config": cfg, "prefix": ""}]

    with pytest.raises(ValueError):
        probe.module.DeepseekV4ForCausalLM(vllm_config=_cfg(max_len=256))
    assert len(probe.lm_calls) == 1


def test_npu_import_installs_hc_head_and_swaps_attention(monkeypatch):
    probe = _load(monkeypatch, "npu", npu_event=object())
    hc = probe.hc_head
    assert getattr(hc, "_vllm_fl_ascend", False) is True
    assert callable(hc.forward_native)
    assert (
        probe.xpu_model.DeepseekV4XPUAttention
        is probe.module.DeepseekV4FLAttention
    )
    assert probe.module.__all__ == ["DeepseekV4ForCausalLM", "DeepseekV4FLAttention"]


def test_non_npu_import_leaves_hc_head_and_attention(monkeypatch):
    probe = _load(monkeypatch, "cuda", npu_event=object())
    assert probe.hc_head is None
    assert probe.xpu_model.DeepseekV4XPUAttention == "ORIGINAL"


def test_hc_head_fallback_forward_native_runs_kernel(monkeypatch):
    probe = _load(monkeypatch, "npu", npu_event=object())
    recorded = []

    def _kernel(flat, hc_fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, mult):
        recorded.append(
            (flat.shape, hc_fn, hidden_size, rms_eps, hc_eps, mult, out.shape)
        )
        out.fill_(1.0)

    _install(monkeypatch, "flaggems_vllm", hc_head_fused_kernel=_kernel)

    hidden = torch.zeros(2, 3, 4, 8)
    out = probe.hc_head.forward_native(
        None, hidden, "hc_fn", "hc_scale", "hc_base", 1e-5, 1e-6
    )
    assert out.shape == (2, 3, 8)
    assert torch.all(out == 1.0)
    assert recorded == [
        (torch.Size([6, 4, 8]), "hc_fn", 8, 1e-5, 1e-6, 4, torch.Size([6, 8]))
    ]


def test_hc_head_fallback_is_idempotent(monkeypatch):
    probe = _load(monkeypatch, "npu", npu_event=object())
    installed = probe.hc_head.forward_native
    probe.module._install_hc_head_fallback()
    assert probe.hc_head.forward_native is installed
    assert probe.hc_head._vllm_fl_ascend is True
