import types

import torch

from vllm_fl.dispatch.backends.vendor.iluvatar.impl import mm_silu_mul


def test_fused_gate_up_matches_only_supported_shape(monkeypatch):
    proj = types.SimpleNamespace(
        quant_config=None,
        weight=torch.empty(12288, 2048, dtype=torch.bfloat16),
    )
    mlp = types.SimpleNamespace(gate_up_proj=proj)
    x = torch.empty(64, 2048, dtype=torch.bfloat16)
    expected = object()
    calls = []

    def fake_mm(a, b):
        calls.append((a.shape, b.shape, b.is_contiguous()))
        return expected

    monkeypatch.setattr(torch.ops.fl, "mm_silu_mul", fake_mm)
    assert mm_silu_mul._fused_gate_up(mlp, x) is expected
    assert calls == [(x.shape, (2048, 12288), False)]
    assert mm_silu_mul._fused_gate_up(mlp, torch.empty(512, 2048, dtype=torch.bfloat16)) is None

    proj.quant_config = object()
    assert mm_silu_mul._fused_gate_up(mlp, x) is None


def test_fused_gate_up_honors_disable_switch(monkeypatch):
    monkeypatch.setattr(mm_silu_mul, "_DISABLED", True)
    proj = types.SimpleNamespace(
        quant_config=None,
        weight=torch.empty(12288, 2048, dtype=torch.bfloat16),
    )
    mlp = types.SimpleNamespace(gate_up_proj=proj)
    x = torch.empty(64, 2048, dtype=torch.bfloat16)
    assert mm_silu_mul._fused_gate_up(mlp, x) is None


def test_apply_patches_minicpm_mlp(monkeypatch):
    from vllm.model_executor.models.llama import LlamaMLP
    from vllm.model_executor.models.minicpm import MiniCPMMLP

    monkeypatch.setattr(mm_silu_mul, "_applied", False)
    monkeypatch.setattr(LlamaMLP, "forward", lambda self, x: x)
    monkeypatch.setattr(MiniCPMMLP, "forward", lambda self, x: x)
    mm_silu_mul.apply()
    assert LlamaMLP.forward is mm_silu_mul._forward
    assert MiniCPMMLP.forward is mm_silu_mul._forward
