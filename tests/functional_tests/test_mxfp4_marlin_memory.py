# SPDX-License-Identifier: Apache-2.0
"""Measure one REAL TP shard layer's live allocated storage after raw retirement."""

import gc
import json
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config

from vllm_fl.ops.fused_moe.mxfp4_marlin import _make_experts_class


@pytest.mark.parametrize(
    "name,e,h,padded_i,compact_i",
    [
        ("Flash_TP8", 256, 4096, 256, 0),
        ("Pro_TP16", 384, 7168, 256, 192),
    ],
)
def test_real_shard_live_memory(name, e, h, padded_i, compact_i, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires Hopper")
    # Earlier test modules deliberately create expert/layer reference cycles;
    # collect them BEFORE measuring this layer's allocation delta.
    gc.collect()
    monkeypatch.setenv("VLLM_FL_MXFP4_COMPACT_I", str(compact_i))
    device, dtype = "cuda", torch.bfloat16
    w1 = torch.nn.Parameter(
        torch.zeros((e, 2 * padded_i, h // 2), device=device, dtype=torch.uint8),
        requires_grad=False,
    )
    w2 = torch.nn.Parameter(
        torch.zeros((e, h, padded_i // 2), device=device, dtype=torch.uint8),
        requires_grad=False,
    )
    s1 = torch.nn.Parameter(
        torch.full((e, 2 * padded_i, h // 32), 120, device=device, dtype=torch.uint8),
        requires_grad=False,
    )
    s2 = torch.nn.Parameter(
        torch.full((e, h, padded_i // 32), 120, device=device, dtype=torch.uint8),
        requires_grad=False,
    )
    valid_i = compact_i or padded_i
    w1[:, :valid_i] = 0x11
    w1[:, padded_i : padded_i + valid_i] = 0x22
    w2[:, :, : valid_i // 2] = 0x33
    params = (w1, w2, s1, s2)
    raw_bytes = sum(p.untyped_storage().nbytes() for p in params)
    references = [p.untyped_storage()._weak_ref() for p in params]
    identities = [id(p) for p in params]
    layer = SimpleNamespace(
        w13_weight=w1, w2_weight=w2, w13_weight_scale=s1, w2_weight_scale=s2
    )
    moe = SimpleNamespace(
        hidden_dim=h,
        in_dtype=dtype,
        swiglu_limit=10.0,
        swiglu_alpha=None,
        swiglu_beta=None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        num_experts=e,
        num_local_experts=e,
        activation=MoEActivation.SILU,
        intermediate_size_per_partition=padded_i,
    )
    quant = mxfp4_w4a16_moe_quant_config(
        w1_scale=s1, w2_scale=s2, gemm1_clamp_limit=10.0
    )
    experts = _make_experts_class()(moe_config=moe, quant_config=quant)
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    before_requested = torch.cuda.memory_stats()["requested_bytes.all.current"]
    experts.process_weights_after_loading(layer)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    after_requested = torch.cuda.memory_stats()["requested_bytes.all.current"]
    packed_bytes = sum(p.untyped_storage().nbytes() for p in experts.packed)
    assert all(torch.UntypedStorage._expired(r) for r in references)
    for ref in references:
        torch.UntypedStorage._free_weak_ref(ref)
    assert [id(p) for p in params] == identities
    assert all(
        p.untyped_storage().data_ptr() == q.untyped_storage().data_ptr()
        for p, q in zip(params, experts.packed)
    )
    # Native config and model TP are unchanged; only proven-zero padding retired.
    assert moe.intermediate_size_per_partition == padded_i
    expected_delta = packed_bytes - raw_bytes
    print(
        json.dumps(
            dict(
                model=name,
                E=e,
                H=h,
                loaded_I=padded_i,
                packed_I=valid_i,
                raw_bytes=raw_bytes,
                packed_bytes=packed_bytes,
                measured_allocation_delta=after - before,
                expected_delta=expected_delta,
                requested_bytes_delta=after_requested - before_requested,
                allocator_slack_delta=(after - before)
                - (after_requested - before_requested),
                original_storage_expired=True,
                parameters_alias_pack=True,
            )
        )
    )
    # A reused CUDA block may contain allocator slack (observed 1 MiB for
    # the 21 MiB Pro scale buffer). requested_bytes excludes that slack.
    assert after_requested - before_requested == expected_delta
