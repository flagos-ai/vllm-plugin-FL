"""Check custom-op metadata and decode-only compilation boundaries."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm_fl.models.hygon import deepseek_v4 as model


@pytest.mark.parametrize("batch,heads", [(1, 8), (64, 16)])
def test_o_proj_fake_matches_hidden_projection(monkeypatch, batch, heads):
    torch.manual_seed(42)
    weight = torch.randn(4096, 512, device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(
        hidden_size=4096,
        o_lora_rank=1024,
        _o_proj=lambda o, positions: torch.nn.functional.linear(o.mean(1), weight),
    )
    context = SimpleNamespace(no_compile_layers={"test.attn": layer})
    monkeypatch.setattr(model, "get_forward_context", lambda: context)
    o = torch.randn(batch, heads, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(batch, device="cuda")
    torch.library.opcheck(
        torch.ops.vllm.fl_hygon_dv4_o_proj.default,
        (o, positions, "test.attn"),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.parametrize(
    "mode,expected_splits",
    [(CUDAGraphMode.FULL_DECODE_ONLY, 0), (CUDAGraphMode.FULL_AND_PIECEWISE, 1)],
)
def test_attention_boundaries_respect_capture_mode(mode, expected_splits):
    config = SimpleNamespace(cudagraph_mode=mode, splitting_ops=[])
    model.configure_attention_graph_boundaries(config)
    model.configure_attention_graph_boundaries(config)
    assert len(config.splitting_ops) == expected_splits
