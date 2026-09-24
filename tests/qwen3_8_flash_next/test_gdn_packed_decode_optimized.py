# Copyright (c) 2026 BAAI. All rights reserved.

"""GPU correctness gates for the optimized packed-GDN decode replacement.

This is intentionally separate from the capability/idempotence test.  It
executes the actual vLLM wrapper after installing the replacement, so a test
cannot pass by compiling an unreferenced kernel while the model still points
at the old symbol.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.model_executor.layers.fla.ops import (  # noqa: E402
    fused_recurrent as fused_recurrent_mod,
)

from vllm_fl.patches import gdn_packed_decode  # noqa: E402

H = 2
HV = 6
DEVICE = "cuda"


def _install_candidate():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for packed-GDN kernel validation")

    # The model patch calls this hook too; invoking it explicitly makes this
    # standalone test prove the same selection path without a full model boot.
    gdn_packed_decode.patch_vllm_packed_gdn_beta()
    selected = fused_recurrent_mod.fused_recurrent_gated_delta_rule_packed_decode_kernel
    candidate = gdn_packed_decode._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
    assert selected is candidate, (
        "the vLLM wrapper is not bound to the optimized FP32-beta kernel"
    )
    assert getattr(selected, "_fl_fp32_beta", False)
    return fused_recurrent_mod.fused_recurrent_gated_delta_rule_packed_decode


def _make_case(
    batch: int,
    key_size: int = 128,
    value_size: int = 128,
    dtype=None,
    seed: int = 1234,
):
    if dtype is None:
        dtype = torch.bfloat16
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    qkv_size = 2 * H * key_size + HV * value_size
    mixed_qkv = torch.randn(
        (batch, qkv_size), device=DEVICE, dtype=dtype, generator=generator
    )
    a = torch.randn((batch, HV), device=DEVICE, dtype=dtype, generator=generator)
    b = torch.randn((batch, HV), device=DEVICE, dtype=dtype, generator=generator)
    A_log = torch.randn((HV,), device=DEVICE, dtype=torch.float32, generator=generator)
    dt_bias = torch.randn((HV,), device=DEVICE, dtype=torch.float32, generator=generator)
    state = torch.randn(
        (batch + 2, HV, value_size, key_size),
        device=DEVICE,
        dtype=torch.float32,
        generator=generator,
    )
    indices = torch.arange(1, batch + 1, device=DEVICE, dtype=torch.int32)
    out = torch.empty(
        (batch, 1, HV, value_size), device=DEVICE, dtype=dtype
    )
    return mixed_qkv, a, b, A_log, dt_bias, state, indices, out


def _reference(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    initial_state,
    indices,
    out_dtype,
    use_qk_l2norm_in_kernel: bool,
):
    """Scalar PyTorch reference with the same FP32 beta/state semantics."""

    batch = mixed_qkv.shape[0]
    key_size = initial_state.shape[-1]
    value_size = initial_state.shape[-2]
    heads = mixed_qkv.shape[1] - HV * value_size
    heads //= 2 * key_size
    scale = key_size**-0.5
    state = initial_state.clone()
    out = torch.zeros(
        (batch, 1, HV, value_size), device=mixed_qkv.device, dtype=out_dtype
    )
    index_list = indices.detach().cpu().tolist()

    for n, state_idx in enumerate(index_list):
        if state_idx <= 0:
            continue
        for hv in range(HV):
            h = state[state_idx, hv].float()
            head = hv // (HV // heads)
            q_start = head * key_size
            k_start = heads * key_size + q_start
            v_start = 2 * heads * key_size + hv * value_size
            q = mixed_qkv[n, q_start : q_start + key_size].float()
            k = mixed_qkv[n, k_start : k_start + key_size].float()
            v = mixed_qkv[n, v_start : v_start + value_size].float()
            if use_qk_l2norm_in_kernel:
                q = q / torch.sqrt(torch.sum(q * q) + 1e-6)
                k = k / torch.sqrt(torch.sum(k * k) + 1e-6)
            q = q * scale
            x = a[n, hv].float() + dt_bias[hv].float()
            softplus_x = torch.where(
                x <= 20.0, torch.log(1.0 + torch.exp(x)), x
            )
            decay = torch.exp(-torch.exp(A_log[hv].float()) * softplus_x)
            beta = torch.sigmoid(b[n, hv].float())
            h = h * decay
            v = v - torch.sum(h * k[None, :], dim=1)
            v = v * beta
            h = h + v[:, None] * k[None, :]
            out[n, 0, hv] = torch.sum(h * q[None, :], dim=1).to(out_dtype)
            state[state_idx, hv].copy_(h)
    return out, state


def _run(wrapper, case, use_qk_l2norm_in_kernel=True):
    mixed_qkv, a, b, A_log, dt_bias, state, indices, out = case
    return wrapper(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=state.shape[-1] ** -0.5,
        initial_state=state,
        out=out,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def _assert_close(actual_out, actual_state, expected_out, expected_state):
    out_diff = (actual_out.float() - expected_out.float()).abs()
    state_diff = (actual_state - expected_state).abs()
    out_max = out_diff.max().item()
    out_rmse = out_diff.square().mean().sqrt().item()
    state_max = state_diff.max().item()
    state_rmse = state_diff.square().mean().sqrt().item()
    print(
        "GDN packed decode error metrics: "
        f"out_max_abs={out_max:.8g} out_rmse={out_rmse:.8g} "
        f"state_max_abs={state_max:.8g} state_rmse={state_rmse:.8g}"
    )
    assert torch.allclose(
        actual_out.float(), expected_out.float(), atol=1e-3, rtol=1e-3
    ), (
        "packed-GDN output mismatch: "
        f"out_max_abs={out_max:.8g}, out_rmse={out_rmse:.8g}"
    )
    assert torch.allclose(
        actual_state, expected_state, atol=1e-5, rtol=1e-5
    ), (
        "packed-GDN FP32 state mismatch: "
        f"state_max_abs={state_max:.8g}, state_rmse={state_rmse:.8g}"
    )


@pytest.mark.parametrize(
    "batch,value_size,key_size,dtype",
    [
        (1, 1, 127, torch.float16),
        (2, 17, 128, torch.bfloat16),
        (8, 33, 127, torch.bfloat16),
        (64, 128, 128, torch.bfloat16),
        (3, 129, 128, torch.float16),
    ],
)
def test_optimized_packed_decode_matches_fp32_reference(
    batch, value_size, key_size, dtype
):
    wrapper = _install_candidate()
    case = _make_case(
        batch,
        key_size=key_size,
        value_size=value_size,
        dtype=dtype,
        seed=7000 + batch + value_size + key_size,
    )
    initial_state = case[5].clone()
    actual_out, actual_state = _run(wrapper, case)
    expected_out, expected_state = _reference(
        case[0],
        case[1],
        case[2],
        case[3],
        case[4],
        initial_state,
        case[6],
        case[7].dtype,
        True,
    )
    _assert_close(actual_out, actual_state, expected_out, expected_state)


def test_invalid_indices_keep_state_and_zero_outputs():
    wrapper = _install_candidate()
    case = list(_make_case(4, key_size=127, value_size=33, seed=8111))
    case[6] = torch.tensor([0, -1, 1, 2], device=DEVICE, dtype=torch.int32)
    initial_state = case[5].clone()
    actual_out, actual_state = _run(wrapper, tuple(case))
    expected_out, expected_state = _reference(
        case[0],
        case[1],
        case[2],
        case[3],
        case[4],
        initial_state,
        case[6],
        case[7].dtype,
        True,
    )
    _assert_close(actual_out, actual_state, expected_out, expected_state)
    assert torch.count_nonzero(actual_out[:2]).item() == 0


def test_long_decode_preserves_fp32_state_update():
    wrapper = _install_candidate()
    state_case = _make_case(
        8, key_size=128, value_size=32, dtype=torch.bfloat16, seed=9001
    )
    expected_state = state_case[5].clone()
    max_state_abs = 0.0
    max_state_rmse = 0.0
    max_out_abs = 0.0
    max_out_rmse = 0.0
    for step in range(32):
        case = _make_case(
            8,
            key_size=128,
            value_size=32,
            dtype=torch.bfloat16,
            seed=10000 + step,
        )
        case = list(case)
        case[5] = state_case[5]
        case[6] = state_case[6]
        actual_out, actual_state = _run(wrapper, tuple(case))
        expected_out, expected_state = _reference(
            case[0],
            case[1],
            case[2],
            case[3],
            case[4],
            expected_state,
            case[6],
            case[7].dtype,
            True,
        )
        out_diff = (actual_out.float() - expected_out.float()).abs()
        state_diff = (actual_state - expected_state).abs()
        out_max = out_diff.max().item()
        out_rmse = out_diff.square().mean().sqrt().item()
        state_max = state_diff.max().item()
        state_rmse = state_diff.square().mean().sqrt().item()
        max_out_abs = max(max_out_abs, out_max)
        max_out_rmse = max(max_out_rmse, out_rmse)
        max_state_abs = max(max_state_abs, state_max)
        max_state_rmse = max(max_state_rmse, state_rmse)
        assert torch.allclose(
            actual_out.float(), expected_out.float(), atol=1e-3, rtol=1e-3
        ), (
            f"long decode output mismatch at step {step}: "
            f"out_max_abs={out_max:.8g}, out_rmse={out_rmse:.8g}"
        )
        assert torch.allclose(
            actual_state, expected_state, atol=1e-5, rtol=1e-5
        ), (
            f"long decode FP32 state mismatch at step {step}: "
            f"state_max_abs={state_max:.8g}, state_rmse={state_rmse:.8g}"
        )
        state_case[5].copy_(actual_state)
    print(
        "GDN long-decode max error metrics: "
        f"out_max_abs={max_out_abs:.8g} out_rmse={max_out_rmse:.8g} "
        f"state_max_abs={max_state_abs:.8g} state_rmse={max_state_rmse:.8g}"
    )
    assert torch.allclose(state_case[5], expected_state, atol=1e-5, rtol=1e-5)


def test_cuda_graph_replay_accepts_changed_inputs_and_indices():
    wrapper = _install_candidate()
    case = list(
        _make_case(
            8, key_size=128, value_size=128, dtype=torch.bfloat16, seed=12001
        )
    )
    # Warm up and restore the captured state before graph construction.
    initial_state = case[5].clone()
    _run(wrapper, tuple(case))
    case[5].copy_(initial_state)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        _run(wrapper, tuple(case))

    replay_case = list(
        _make_case(
            8, key_size=128, value_size=128, dtype=torch.bfloat16, seed=12002
        )
    )
    replay_initial_state = replay_case[5].clone()
    expected_out, expected_state = _reference(
        replay_case[0],
        replay_case[1],
        replay_case[2],
        replay_case[3],
        replay_case[4],
        replay_initial_state,
        replay_case[6],
        replay_case[7].dtype,
        True,
    )
    for destination, source in zip(case[:5], replay_case[:5]):
        destination.copy_(source)
    case[5].copy_(replay_initial_state)
    case[6].copy_(replay_case[6])
    case[7].zero_()
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(case[7], case[5], expected_out, expected_state)

    invalid_indices = torch.tensor([0, -1, 3, 4, 5, 6, 7, 8], device=DEVICE, dtype=torch.int32)
    invalid_case = list(
        _make_case(
            8, key_size=128, value_size=128, dtype=torch.bfloat16, seed=12003
        )
    )
    invalid_case[6] = invalid_indices
    invalid_initial_state = invalid_case[5].clone()
    invalid_out, invalid_state = _reference(
        invalid_case[0],
        invalid_case[1],
        invalid_case[2],
        invalid_case[3],
        invalid_case[4],
        invalid_initial_state,
        invalid_case[6],
        invalid_case[7].dtype,
        True,
    )
    for destination, source in zip(case[:5], invalid_case[:5]):
        destination.copy_(source)
    case[5].copy_(invalid_initial_state)
    case[6].copy_(invalid_indices)
    case[7].zero_()
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(case[7], case[5], invalid_out, invalid_state)
