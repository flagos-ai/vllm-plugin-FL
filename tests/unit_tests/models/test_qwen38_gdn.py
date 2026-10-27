# SPDX-License-Identifier: Apache-2.0
"""CPU branch contracts for the Qwen-local GDN layout boundary."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "case,spec,prefills,decodes,expected_calls",
    [
        ("profile", False, 0, 0, 0),
        ("prefill", False, 2, 0, 0),
        ("spec_only", True, 0, 0, 0),
        ("mixed_prefill", True, 2, 0, 1),
        ("mixed_decode", True, 0, 1, 1),
    ],
)
def test_layout_provider_runs_only_before_mixed_native_gather(
    monkeypatch, case, spec, prefills, decodes, expected_calls
):
    class NativeMetadata:
        def __init__(self):
            self.spec_sequence_masks = object() if spec else None
            self.num_prefills = prefills
            self.num_decodes = decodes

    class NativeAttention(torch.nn.Module):
        def __init__(self, config, vllm_config, prefix, gqa_interleaved_layout):
            super().__init__()
            self.prefix = prefix

        def _forward_core(self, mixed_qkv, b, a, core_attn_out):
            self.received = (mixed_qkv, b, a, core_attn_out)
            return core_attn_out

    prefix = "model.layers.0.linear_attn"
    metadata = None if case == "profile" else {prefix: NativeMetadata()}
    context_module = ModuleType("vllm.forward_context")
    context_module.get_forward_context = lambda: SimpleNamespace(attn_metadata=metadata)
    native_module = ModuleType(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    native_module.GDNAttentionMetadata = NativeMetadata
    native_module.QwenGatedDeltaNetAttention = NativeAttention
    native_module.causal_conv1d_update = None
    native_module.is_conv_state_dim_first = None
    layout_calls, resolved = [], []

    def layout(value):
        layout_calls.append(value)
        return value.contiguous()

    def resolve(name):
        resolved.append(name)
        return layout if name == "qwen38_gdn_mixed_qkv" else lambda **kwargs: None

    dispatch_module = ModuleType("vllm_fl.dispatch")
    dispatch_module.resolve_op = resolve
    monkeypatch.setitem(sys.modules, context_module.__name__, context_module)
    monkeypatch.setitem(sys.modules, native_module.__name__, native_module)
    monkeypatch.setitem(sys.modules, dispatch_module.__name__, dispatch_module)
    module_name = "_qwen38_gdn_layout_test"
    path = ROOT / "vllm_fl/models/qwen3_8_flash_next/gdn.py"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    module_spec.loader.exec_module(module)
    layer = module.Qwen38GatedDeltaNetAttention(None, None, prefix=prefix)

    qkv = torch.arange(42 * 2048, dtype=torch.float32).reshape(42, 2048)[:, :1280]
    b, a, output = torch.empty(42, 6), torch.empty(42, 6), torch.empty(42, 6, 128)
    assert layer._forward_core(qkv, b, a, output) is output
    assert resolved == ["gdn_prefill", "gdn_packed_decode", "qwen38_gdn_mixed_qkv"]
    assert len(layout_calls) == expected_calls
    actual, sent_b, sent_a, sent_output = layer.received
    assert sent_b is b and sent_a is a and sent_output is output
    torch.testing.assert_close(actual, qkv, atol=0, rtol=0)
    if expected_calls:
        assert layout_calls[0] is qkv
        assert actual.is_contiguous()
    else:
        assert actual is qkv
        assert actual.stride() == (2048, 1)
