# Copyright (c) 2026 BAAI. All rights reserved.
"""Exercise the real decoder orchestration with cheap attention/FFN stubs.

This checks mHC, norm, residual/mix propagation and fullgraph boundaries. It
is deliberately not an end-to-end model quality or iteration benchmark.
"""
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP required")
MODEL = Path("/model/DeepSeek-V4-Flash-0731-W8A8-INT8")


class Attention(torch.nn.Module):
    def forward(self, positions, x, scaling=None):
        return x * 0.125


class FFN(torch.nn.Module):
    def forward(self, x, input_ids=None):
        return x * -0.25


@pytest.fixture(scope="module")
def layer():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.models.deepseek_v4.amd.model import DeepseekV4DecoderLayer
    from vllm_fl.models.hygon.deepseek_v4 import (
        _MHCPreGraphOp, _MHCPostGraphOp, _MHCFusedPostPreGraphOp,
    )
    from vllm_fl.ops.layernorm import RMSNormFL

    obj = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
    torch.nn.Module.__init__(obj)
    obj.hidden_size = 4096
    obj.rms_norm_eps = obj.hc_eps = 1e-6
    obj.hc_post_alpha = 2.
    obj.hc_sinkhorn_iters = 20
    obj.mhc_pre = _MHCPreGraphOp()
    obj.mhc_post = _MHCPostGraphOp()
    obj.mhc_fused_post_pre = _MHCFusedPostPreGraphOp()
    obj.attn = Attention()
    obj.ffn = FFN()
    with set_current_vllm_config(VllmConfig()):
        obj.attn_norm = RMSNormFL(4096, dtype=torch.bfloat16).cuda()
        obj.ffn_norm = RMSNormFL(4096, dtype=torch.bfloat16).cuda()
    weights = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    for branch in ("attn", "ffn"):
        for suffix in ("fn", "scale", "base"):
            name = f"hc_{branch}_{suffix}"
            key = "layers.0." + name
            with safe_open(MODEL / weights[key], framework="pt", device="cpu") as f:
                setattr(obj, name, f.get_tensor(key).cuda())
        key = f"layers.0.{branch}_norm.weight"
        with safe_open(MODEL / weights[key], framework="pt", device="cpu") as f:
            getattr(obj, branch + "_norm").weight.data.copy_(f.get_tensor(key).cuda())
    # Use the actual FL dispatch method, the same one installed by the worker.
    for norm in (obj.attn_norm, obj.ffn_norm):
        norm._forward_method = norm.forward_oot
    return obj


@pytest.mark.parametrize("m", [16, 24, 32, 48, 64, 1024, 4096, 8131, 8192])
@pytest.mark.parametrize("fused_post_pre", [False, True])
def test_decoder_chain_graph(layer, m, fused_post_pre):
    from vllm.models.deepseek_v4.amd.model import DeepseekV4DecoderLayer
    from vllm_fl.models.hygon.prefill_mhc import install_prefill_mhc_norm
    from vllm_fl.dispatch.backends.vendor.hygon.impl.native_ops.int8_quant import dynamic_scaled_int8_quant_hygon

    layer.has_tilelang = fused_post_pre
    assert install_prefill_mhc_norm(layer)
    x = torch.randn(m, 4, 4096, device="cuda", dtype=torch.bfloat16) * .25
    positions = torch.arange(m, device="cuda")

    def chain(fast):
        if fast:
            first = layer(x, positions, None)
            return layer(first[0], positions, None, first[2], first[3], first[1])
        first = DeepseekV4DecoderLayer.forward(layer, x, positions, None)
        return DeepseekV4DecoderLayer.forward(
            layer, first[0], positions, None, first[2], first[3], first[1])

    chain(True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = chain(True)
    for scale in (.25, 10., 0.):
        x.normal_(std=scale)
        graph.replay()
        expected = chain(False)
        for a, b in zip(actual, expected):
            if b is None:
                assert a is None
            else:
                assert torch.equal(a.view(torch.int16), b.view(torch.int16))
                assert a.isfinite().all()
        # Check the next quantization boundary, including its scale.
        a, b = actual[0].reshape(m, -1), expected[0].reshape(m, -1)
        aq, bq = torch.empty_like(a, dtype=torch.int8), torch.empty_like(b, dtype=torch.int8)
        asc, bsc = torch.empty(m, 1, device="cuda"), torch.empty(m, 1, device="cuda")
        dynamic_scaled_int8_quant_hygon(aq, a, asc, None)
        dynamic_scaled_int8_quant_hygon(bq, b, bsc, None)
        assert torch.equal(aq, bq) and torch.equal(asc, bsc)


def test_prefill_custom_op_fullgraph_and_schema(layer):
    from vllm_fl.dispatch.backends.vendor.hygon.impl.other import prefill_mhc  # noqa: F401
    from vllm_fl.models.hygon.prefill_mhc import install_prefill_mhc_norm
    assert install_prefill_mhc_norm(layer)
    op = torch.ops.vllm.fl_hygon_dv4_prefill_mhc_pre_norm
    x = torch.randn(1024, 4, 4096, device="cuda", dtype=torch.bfloat16)
    args = (x, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
            layer.attn_norm.weight, 1e-6, 1e-6, 1e-6, 2., 20, 1e-6)
    torch.library.opcheck(op.default, args, test_utils=("test_schema", "test_faketensor"))
    compiled = torch.compile(lambda *a: op(*a), fullgraph=True, backend="eager")
    for a, b in zip(compiled(*args), op(*args)):
        assert torch.equal(a, b)

    layer.has_tilelang = True
    positions = torch.arange(x.shape[0], device='cuda')
    compiled_layer = torch.compile(layer, fullgraph=True, backend='eager')
    for a, b in zip(compiled_layer(x, positions, None), layer(x, positions, None)):
        assert torch.equal(a, b)


def test_switch_and_backend_guard(layer, monkeypatch):
    import copy
    from vllm_fl.models.hygon.prefill_mhc import install_prefill_mhc_norm
    from vllm_fl.dispatch import get_default_manager

    monkeypatch.setenv("VLLM_FL_HYGON_PREFILL_MHC_NORM", "0")
    assert not install_prefill_mhc_norm(layer)
    monkeypatch.setenv("VLLM_FL_HYGON_PREFILL_MHC_NORM", "1")
    new = copy.copy(layer)
    new.__dict__.pop("_hygon_original_mhc_forward", None)
    monkeypatch.setattr(get_default_manager(), "resolve_candidates", lambda name: [])
    assert not install_prefill_mhc_norm(new)
