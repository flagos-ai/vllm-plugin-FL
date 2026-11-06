# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU regressions for the vLLM 0.32 / external FLA state-layout bridge."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

PATCH = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl/dispatch/backends/vendor/metax/patches/gdn_linear_attn.py"
)


@pytest.fixture
def patched(monkeypatch):
    registered = {}
    calls = []

    class ChunkGatedDeltaRule:
        def __init__(self):
            # Match vLLM 0.32, which overrides CustomOp's OOT dispatch here.
            self._forward_method = self.forward_native

        def __call__(self, **kwargs):
            return self._forward_method(**kwargs)

        @classmethod
        def register_oot(cls, replacement):
            registered[cls.__name__] = replacement
            return replacement

        def forward_native(self, **kwargs):
            calls.append(kwargs)
            return "native-output", "native-state"

    upstream = ModuleType("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
    upstream.ChunkGatedDeltaRule = ChunkGatedDeltaRule
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    monkeypatch.delenv("MACA_VLLM_USE_EXTERNAL_FLA", raising=False)
    spec = importlib.util.spec_from_file_location("metax_gdn_test_patch", PATCH)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    return SimpleNamespace(patch=patch, registered=registered, calls=calls)


def inputs():
    # K != V exposes accidental double transposes, including when dimensions
    # happen to be equal in the deployed model.
    return {
        "q": torch.randn(1, 5, 2, 3),
        "k": torch.randn(1, 5, 2, 3),
        "v": torch.randn(1, 5, 2, 4),
        "g": torch.randn(1, 5, 2),
        "beta": torch.randn(1, 5, 2),
        "initial_state": torch.arange(48, dtype=torch.float32).reshape(2, 2, 4, 3),
        "output_final_state": True,
        "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.int32),
        "chunk_indices": torch.tensor([[0, 0], [1, 0]], dtype=torch.int32),
        "chunk_offsets": torch.tensor([0, 1, 2], dtype=torch.int32),
        "use_qk_l2norm_in_kernel": False,
        "core_attn_out": torch.full((7, 2, 4), -123.0),
        "aiter_prefill_metadata": object(),
    }


def test_032_chunk_registration_keeps_native_dispatch_and_all_metadata(patched):
    cls = patched.registered["ChunkGatedDeltaRule"]
    assert cls is patched.patch.MacaChunkGatedDeltaRule
    assert list(patched.registered) == ["ChunkGatedDeltaRule"]
    args = inputs()
    assert cls()(**args) == ("native-output", "native-state")
    assert patched.calls[0].keys() == args.keys()
    for name, value in args.items():
        assert patched.calls[0][name] is value


@pytest.mark.parametrize("layout", ["old", "state_v_first", "transpose_state_layout"])
@pytest.mark.parametrize("output_final_state", [True, False])
def test_external_fla_preserves_state_layout_and_writes_only_live_output(
    patched, monkeypatch, layout, output_final_state
):
    args = inputs()
    args["output_final_state"] = output_final_state
    initial = args["initial_state"].clone()
    expected_state = initial + 7
    expected_output = torch.arange(40, dtype=torch.float32).reshape(1, 5, 2, 4)
    calls = []

    def external_call(kwargs, v_first):
        calls.append(kwargs)
        # Real FLA consumes these kwargs; new vLLM-only metadata must not leak.
        assert set(kwargs) == {
            "q",
            "k",
            "v",
            "g",
            "beta",
            "initial_state",
            "output_final_state",
            "cu_seqlens",
            "use_qk_l2norm_in_kernel",
        }
        for name in ("q", "k", "v", "g", "beta", "cu_seqlens"):
            assert kwargs[name] is args[name]
        assert kwargs["use_qk_l2norm_in_kernel"] is False
        state = kwargs["initial_state"]
        assert state.is_contiguous()
        if not v_first:
            state = state.transpose(-1, -2)
        torch.testing.assert_close(state, initial)
        final_state = expected_state if v_first else expected_state.transpose(-1, -2)
        return expected_output, final_state if output_final_state else None

    def old(**kwargs):
        return external_call(kwargs, False)

    def state_v_first(*, state_v_first=False, **kwargs):
        assert state_v_first is True
        return external_call(kwargs, True)

    def transpose_state_layout(*, transpose_state_layout=False, **kwargs):
        assert transpose_state_layout is True
        return external_call(kwargs, True)

    external = ModuleType("fla.ops.gated_delta_rule")
    external.chunk_gated_delta_rule = {
        "old": old,
        "state_v_first": state_v_first,
        "transpose_state_layout": transpose_state_layout,
    }[layout]
    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setenv("MACA_VLLM_USE_EXTERNAL_FLA", "1")
    op = patched.patch.MacaChunkGatedDeltaRule()
    output, state = op(**args)
    assert len(calls) == 1
    assert patched.calls == []
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(args["core_attn_out"][:5], expected_output.squeeze(0))
    assert torch.all(args["core_attn_out"][5:] == -123)
    torch.testing.assert_close(args["initial_state"], initial)
    if output_final_state:
        assert state.is_contiguous()
        torch.testing.assert_close(state, expected_state)
    else:
        assert state is None


def test_external_fla_accepts_absent_state_and_output_buffer(patched, monkeypatch):
    external = ModuleType("fla.ops.gated_delta_rule")

    def chunk(**kwargs):
        assert kwargs["initial_state"] is None
        return kwargs["v"], None

    external.chunk_gated_delta_rule = chunk
    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setenv("MACA_VLLM_USE_EXTERNAL_FLA", "1")
    args = inputs()
    args.update(initial_state=None, core_attn_out=None, output_final_state=False)
    op = patched.patch.MacaChunkGatedDeltaRule()
    output, state = op.forward_oot(**args)
    assert output is args["v"]
    assert state is None
