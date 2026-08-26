from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)

from vllm_fl.ops.fused_moe import fused_moe_utils


def _import_ascend_backend_module():
    try:
        from vllm_fl.dispatch.backends.vendor.ascend import ascend
    except (AttributeError, ImportError) as exc:
        pytest.skip(f"Ascend backend dependencies are unavailable: {exc}")
    return ascend


def test_ascend_uses_fl_experts_when_flaggems_is_disabled(monkeypatch):
    """Ascend vendor kernels do not require global FlagGems ATen patches."""
    platform = SimpleNamespace(
        is_cpu=lambda: False,
        is_tpu=lambda: False,
        is_out_of_tree=lambda: True,
    )
    monkeypatch.setattr(fused_moe_utils, "current_platform", platform)
    monkeypatch.setattr(fused_moe_utils, "get_platform_name", lambda: "ascend")
    monkeypatch.setattr(fused_moe_utils, "use_flaggems", lambda: False)

    backend, experts_cls = fused_moe_utils.select_unquantized_moe_backend_oot(
        SimpleNamespace()
    )

    assert backend is UnquantizedMoeBackend.TRITON
    assert experts_cls is fused_moe_utils.TritonExpertsFL


def test_ascend_topk_returns_native_bf16_and_int32_outputs(monkeypatch):
    """The simple router must not round-trip native BF16 weights via FP32."""
    ascend = _import_ascend_backend_module()
    backend = ascend.AscendBackend()

    gating_output = torch.randn(3, 8, dtype=torch.bfloat16)
    topk_weights = torch.zeros(3, 2, dtype=torch.float32)
    topk_indices = torch.full((3, 2), -1, dtype=torch.int32)
    token_expert_indices = torch.full((3, 2), -2, dtype=torch.int32)
    native_weights = torch.randn(3, 2, dtype=torch.bfloat16)
    native_indices = torch.tensor([[1, 3], [2, 4], [0, 7]], dtype=torch.int32)
    calls = []

    def fake_moe_gating_top_k(x, **kwargs):
        calls.append((x, kwargs))
        return native_weights, native_indices, torch.empty(0)

    monkeypatch.setattr(
        ascend,
        "_get_moe_gating_top_k",
        lambda: fake_moe_gating_top_k,
    )

    actual_weights, actual_indices = backend.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=True,
    )

    assert actual_weights is native_weights
    assert actual_indices is native_indices
    assert actual_weights.dtype == torch.bfloat16
    assert actual_indices.dtype == torch.int32
    # Prove that the generic preallocations were not used as copy targets.
    assert torch.count_nonzero(topk_weights) == 0
    assert torch.equal(topk_indices, torch.full_like(topk_indices, -1))
    assert len(calls) == 1
    assert calls[0][0] is gating_output
    assert calls[0][1] == {
        "k": 2,
        "k_group": 1,
        "group_count": 1,
        "group_select_mode": 1,
        "renorm": 1,
        "norm_type": 0,
        "out_flag": False,
        "routed_scaling_factor": 1.0,
        "eps": 1e-20,
        "bias_opt": None,
    }


@pytest.mark.parametrize(
    ("gating_dtype", "indices_dtype", "renormalize"),
    [
        (torch.float64, torch.int32, False),
        (torch.bfloat16, torch.int64, False),
        (torch.bfloat16, torch.int32, 1),
    ],
    ids=("unsupported-dtype", "requested-int64-ids", "non-bool-renorm"),
)
def test_ascend_topk_unsupported_contract_uses_torch_fallback(
    monkeypatch,
    gating_dtype,
    indices_dtype,
    renormalize,
):
    ascend = _import_ascend_backend_module()
    backend = ascend.AscendBackend()

    gating_output = torch.tensor(
        [[3.0, 1.0, 0.0, 2.0], [0.0, 4.0, 1.0, 2.0]],
        dtype=gating_dtype,
    )
    topk_weights = torch.empty(2, 2, dtype=torch.float32)
    topk_indices = torch.empty(2, 2, dtype=indices_dtype)
    token_expert_indices = torch.empty(2, 2, dtype=torch.int32)

    def unexpected_custom_op(*args, **kwargs):
        pytest.fail("unsupported contracts must not call moe_gating_top_k")

    monkeypatch.setattr(
        ascend,
        "_get_moe_gating_top_k",
        lambda: unexpected_custom_op,
    )

    actual_weights, actual_indices = backend.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=renormalize,
    )

    expected_weights, expected_indices = torch.topk(
        torch.softmax(gating_output.float(), dim=-1),
        k=2,
        dim=-1,
    )
    if renormalize:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    assert actual_weights is topk_weights
    assert actual_indices is topk_indices
    assert actual_weights.dtype == torch.float32
    assert actual_indices.dtype == indices_dtype
    torch.testing.assert_close(actual_weights, expected_weights)
    assert torch.equal(actual_indices, expected_indices.to(indices_dtype))


def test_ascend_topk_missing_custom_op_uses_torch_fallback(monkeypatch):
    ascend = _import_ascend_backend_module()
    backend = ascend.AscendBackend()

    gating_output = torch.randn(2, 4, dtype=torch.bfloat16)
    topk_weights = torch.empty(2, 2, dtype=torch.float32)
    topk_indices = torch.empty(2, 2, dtype=torch.int32)
    token_expert_indices = torch.empty(2, 2, dtype=torch.int32)
    monkeypatch.setattr(ascend, "_get_moe_gating_top_k", lambda: None)

    actual_weights, actual_indices = backend.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
    )

    assert actual_weights is topk_weights
    assert actual_indices is topk_indices
    assert actual_weights.dtype == torch.float32
    assert actual_indices.dtype == torch.int32


def test_ascend_topk_old_custom_op_contract_keeps_preallocated_outputs(
    monkeypatch,
):
    """A stale local extension must preserve the former public contract."""
    ascend = _import_ascend_backend_module()
    backend = ascend.AscendBackend()

    gating_output = torch.randn(2, 4, dtype=torch.bfloat16)
    topk_weights = torch.empty(2, 2, dtype=torch.float32)
    topk_indices = torch.empty(2, 2, dtype=torch.int32)
    token_expert_indices = torch.empty(2, 2, dtype=torch.int32)
    old_weights = torch.randn(2, 2, dtype=torch.float32)
    old_indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)

    monkeypatch.setattr(
        ascend,
        "_get_moe_gating_top_k",
        lambda: (
            lambda *args, **kwargs: (old_weights, old_indices, torch.empty(0))
        ),
    )

    actual_weights, actual_indices = backend.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
    )

    assert actual_weights is topk_weights
    assert actual_indices is topk_indices
    torch.testing.assert_close(actual_weights, old_weights)
    assert torch.equal(actual_indices, old_indices.to(torch.int32))


def test_ascend_native_topk_fast_path_is_not_registered_for_other_routers():
    """Grouped/bias/scoring routers retain their existing fallback dispatch."""
    ascend = _import_ascend_backend_module()

    assert "grouped_topk" not in ascend.AscendBackend.__dict__
    assert "fused_topk_bias" not in ascend.AscendBackend.__dict__
