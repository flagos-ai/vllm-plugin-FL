# SPDX-License-Identifier: Apache-2.0
"""Bounded native/reference NPU experts checks, without a model or collectives."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        current_platform.device_type != "npu", reason="Requires Ascend NPU"
    ),
]


def _providers():
    from vllm_fl.dispatch.backends.reference.impl import w8a8_moe as reference
    from vllm_fl.dispatch.backends.vendor.ascend.impl import w8a8_moe as native

    return native, reference


def _case(dtype, ids, first_expert=0, global_num_experts=2):
    native, reference = _providers()
    torch.manual_seed(83)
    w1 = torch.randint(-24, 24, (2, 128, 128), dtype=torch.int8, device="npu")
    w2 = torch.randint(-24, 24, (2, 128, 64), dtype=torch.int8, device="npu")
    s1 = torch.full((2, 128, 1), 0.015625, dtype=dtype, device="npu")
    s2 = torch.full((2, 128, 1), 0.015625, dtype=dtype, device="npu")
    weights = native.modelslim_w8a8_prepare_experts(w1, w2, s1, s2)
    oracle_weights = reference.modelslim_w8a8_prepare_experts(w1, w2, s1, s2)
    for canonical, original in zip(weights[:2], (w1, w2)):
        assert canonical.shape == original.shape
        torch.testing.assert_close(canonical, original)
        transposed = canonical.transpose(1, 2)
        assert transposed.is_contiguous()
        assert transposed.data_ptr() == canonical.data_ptr()
    assert (
        weights[2].dtype
        == weights[3].dtype
        == (torch.float32 if dtype == torch.float16 else torch.bfloat16)
    )
    assert weights[2].shape == s1.shape
    assert weights[3].shape == s2.shape
    ids = torch.tensor(ids, dtype=torch.int32, device="npu")
    if ids.numel() == 0:
        ids = ids.reshape(0, 1)
    probs = torch.rand(ids.shape, dtype=torch.float32, device="npu")
    probs = probs / probs.sum(dim=-1, keepdim=True)
    x = torch.randn(ids.shape[0], 128, dtype=dtype, device="npu")
    return (
        native,
        reference,
        x,
        weights,
        oracle_weights,
        probs,
        ids,
        {
            "first_expert": first_expert,
            "global_num_experts": global_num_experts,
        },
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "routing", ["local", "local_single", "mixed_offset", "remote_single"]
)
def test_native_dynamic_qdq_reference(dtype, routing):
    if routing == "local":
        ids, first, total = [[0, 1], [1, 0], [0, 1]], 0, 2
    elif routing == "local_single":
        ids, first, total = [[3]], 2, 4
    elif routing == "mixed_offset":
        ids, first, total = [[0, 2], [3, 1], [2, 3]], 2, 4
    else:
        # One remote route exercises the -1 row-index case where abs(-1)
        # would address out of bounds, as well as the empty local prefix.
        ids, first, total = [[0]], 2, 4
    native, ref, x, w, rw, probs, ids, kwargs = _case(dtype, ids, first, total)
    if x.shape[0] > 1:
        x[0].zero_()
    actual = native.modelslim_w8a8_experts(x, *w, probs, ids, **kwargs)
    expected = ref.modelslim_w8a8_experts(x, *rw, probs, ids, **kwargs)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.06)
    if routing == "remote_single":
        assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_zero_tokens(dtype):
    native, ref, x, w, rw, probs, ids, kwargs = _case(dtype, [], 2, 4)
    actual = native.modelslim_w8a8_experts(x, *w, probs, ids, **kwargs)
    expected = ref.modelslim_w8a8_experts(x, *rw, probs, ids, **kwargs)
    assert actual.shape == (0, 128)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("activation", ["silu", "swiglustep"])
def test_clamp_contract(activation):
    native, ref, x, w, rw, probs, ids, kwargs = _case(
        torch.bfloat16, [[2, 3], [0, 2]], 2, 4
    )
    x.mul_(3)
    kwargs.update(activation=activation, limit=0.5)
    actual = native.modelslim_w8a8_experts(x, *w, probs, ids, **kwargs)
    expected = ref.modelslim_w8a8_experts(x, *rw, probs, ids, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.04)
    other = "swiglustep" if activation == "silu" else "silu"
    other_output = ref.modelslim_w8a8_experts(
        x, *rw, probs, ids, **{**kwargs, "activation": other}
    )
    assert not torch.equal(expected, other_output)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_changed_input_graph_replay(dtype):
    import torch_npu  # noqa: F401

    native, _, x, w, _, probs, ids, kwargs = _case(
        dtype, [[0, 2], [3, 1], [2, 3]], 2, 4
    )
    for _ in range(3):
        native.modelslim_w8a8_experts(x, *w, probs, ids, **kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = native.modelslim_w8a8_experts(x, *w, probs, ids, **kwargs)
    torch.npu.synchronize()
    previous = output.clone()
    for _ in range(2):
        x.copy_(torch.randn_like(x))
        expected = native.modelslim_w8a8_experts(x, *w, probs, ids, **kwargs)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected)
        assert torch.isfinite(output).all()
        assert not torch.equal(previous, output)
        previous = output.clone()


def test_unsupported_activation_rejected():
    native, ref, x, w, rw, probs, ids, kwargs = _case(torch.bfloat16, [[0]])
    for provider, weights in ((native, w), (ref, rw)):
        with pytest.raises(NotImplementedError, match="activation"):
            provider.modelslim_w8a8_experts(
                x, *weights, probs, ids, activation="gelu", **kwargs
            )
