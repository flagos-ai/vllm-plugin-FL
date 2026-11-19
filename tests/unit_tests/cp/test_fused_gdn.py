"""Check GDN output fusion against native arithmetic and token ownership."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.cp.fused_gdn import strided_gated_norm, try_prefill_gdn
from vllm_fl.cp.layout import ZigZagPlan


@pytest.mark.gpu
@pytest.mark.parametrize("tokens", [0, 1, 31, 257, 4096, 65536])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_strided_norm_matches_native_and_preserves_inputs(
    tokens,
    activation,
    weight_dtype,
):
    from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

    torch.manual_seed(42)
    x = torch.randn(tokens, 8, 128, device="cuda", dtype=torch.bfloat16)
    storage = torch.randn(tokens, 3 * 8 * 128, device="cuda", dtype=x.dtype)
    z = storage[:, -8 * 128 :].view(tokens, 8, 128)
    # Exercise small and large magnitudes while keeping inputs finite.
    if tokens:
        scales = torch.logspace(-3, 3, tokens, device="cuda").to(x.dtype)
        x.mul_(scales[:, None, None])
        z.mul_(3)
    weight = torch.randn(128, device="cuda", dtype=weight_dtype)
    norm = SimpleNamespace(weight=weight, eps=1e-6, activation=activation)
    before_x, before_z = x.clone(), z.clone()
    if tokens:
        expected = rmsnorm_fn(
            x.reshape(-1, 128),
            weight,
            None,
            z=z.reshape(-1, 128),
            eps=norm.eps,
            norm_before_gate=True,
            activation=activation,
        ).view(tokens, 8, 128)
    else:
        expected = torch.empty_like(x)
    actual = strided_gated_norm(norm, x, z)
    assert torch.equal(actual, expected)
    order = torch.randperm(tokens, device="cuda")
    permuted = strided_gated_norm(norm, x, z, order)
    assert torch.equal(permuted, expected.index_select(0, order))
    assert actual.is_contiguous() and permuted.is_contiguous()
    assert torch.equal(x, before_x) and torch.equal(z, before_z)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "lengths",
    [[32], [4096], [65536], [8192] * 8, [9, 17, 31], [65535], [65535, 1], [1]],
)
def test_norm_and_projection_match_native_zigzag_partials(lengths):
    from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    tokens = starts[-1]
    plans = ZigZagPlan.build_all(starts, lengths, 4)
    order = plans[0].padded_rank_major_rows_tensor(plans, "cuda")
    torch.manual_seed(123)
    x = torch.randn(tokens, 8, 128, device="cuda", dtype=torch.bfloat16)
    storage = torch.randn(tokens, 3072, device="cuda", dtype=x.dtype)
    z = storage[:, 2048:].view(tokens, 8, 128)
    norm = SimpleNamespace(
        weight=torch.randn(128, device="cuda", dtype=x.dtype),
        eps=1e-6,
        activation="silu",
    )
    weight = torch.randn(2048, 1024, device="cuda", dtype=x.dtype) / 32
    native_norm = rmsnorm_fn(
        x.reshape(-1, 128),
        norm.weight,
        None,
        z=z.reshape(-1, 128),
        eps=norm.eps,
        norm_before_gate=True,
        activation=norm.activation,
    ).view(tokens, -1)
    expected = torch.nn.functional.linear(native_norm, weight).index_select(
        0, order.clamp_min(0)
    )
    expected[order < 0] = 0
    fused = strided_gated_norm(norm, x, z, order).flatten(1)
    actual = torch.nn.functional.linear(fused, weight)
    assert torch.equal(actual, expected)


def test_cpu_gdn_falls_back_without_accessing_layer():
    assert try_prefill_gdn(None, torch.ones(16, 2048)) is None


@pytest.mark.gpu
def test_unsupported_dtype_and_empty_input_fall_back():
    assert try_prefill_gdn(None, torch.ones(16, 2048, device="cuda")) is None
    assert (
        try_prefill_gdn(
            None,
            torch.empty(
                0,
                2048,
                device="cuda",
                dtype=torch.bfloat16,
            ),
        )
        is None
    )


def test_decode_and_mixed_batches_do_not_enter_gdn_fusion(monkeypatch):
    from vllm_fl.cp import prstyle

    sentinel = object()
    monkeypatch.setattr(prstyle, "get_forward_context", lambda: None)
    monkeypatch.setattr(prstyle, "_context_metadata", lambda _: None)
    monkeypatch.setattr(prstyle, "_NATIVE_LAYER_FORWARD", lambda *a, **k: sentinel)
    monkeypatch.setattr(
        prstyle,
        "try_prefill_gdn",
        lambda *a: pytest.fail("Fallback entered GDN fusion"),
    )
    assert prstyle._pr_layer_forward(None, None, None) is sentinel


@pytest.mark.parametrize("helper_falls_back", [False, True])
def test_gdn_restores_reduction_after_exception(monkeypatch, helper_falls_back):
    from vllm_fl.cp import prstyle

    plans = ZigZagPlan.build_all([0, 32], [32], 4)
    group = SimpleNamespace(
        world_size=4, rank_in_group=0, all_gather=lambda x, dim: x.repeat(4, 1)
    )
    projection = SimpleNamespace(reduce_results=True)

    class GDN:
        out_proj = projection

        def __call__(self, **kwargs):
            assert not self.out_proj.reduce_results
            raise RuntimeError("native failure")

    layer = SimpleNamespace(
        input_layernorm=None, layer_type="linear_attention", linear_attn=GDN()
    )
    monkeypatch.setattr(prstyle, "get_forward_context", lambda: None)
    monkeypatch.setattr(prstyle, "_context_metadata", lambda _: None)
    monkeypatch.setattr(prstyle, "_is_sequence_prefill", lambda _: True)
    monkeypatch.setattr(prstyle, "get_tp_group", lambda: group)
    monkeypatch.setattr(prstyle, "_plans", lambda *_: plans)
    monkeypatch.setattr(prstyle, "prefill_gemma_norm", lambda norm, x: x)

    def candidate(*args):
        assert not projection.reduce_results
        if helper_falls_back:
            return None
        raise RuntimeError("candidate failure")

    monkeypatch.setattr(prstyle, "try_prefill_gdn", candidate)
    with pytest.raises(RuntimeError, match="failure"):
        prstyle._pr_layer_forward(layer, torch.ones(8, 2048), None)
    assert projection.reduce_results
