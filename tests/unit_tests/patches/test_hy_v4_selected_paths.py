# SPDX-License-Identifier: Apache-2.0
"""Selected dependencies and implementation failures at the HY4 boundary."""

import importlib
import runpy
from types import SimpleNamespace

import pytest

from .test_hy_v4_review_contracts import _prefill_config
from vllm_fl.models import hy_v4
from vllm_fl.patches import hy_v4_runtime as runtime


@pytest.fixture
def native_prefill(monkeypatch):
    import vllm.model_executor.layers.attention.mla_attention as mla
    from vllm import platforms

    backend = SimpleNamespace(is_available=lambda: True, get_name=lambda: "NATIVE")
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    monkeypatch.setattr(mla, "get_mla_prefill_backend", lambda cfg: backend)
    monkeypatch.setattr(runtime, "native_hy4_available", lambda *_: False)
    monkeypatch.setattr(runtime, "use_flaggems_op", lambda name: True)
    monkeypatch.setattr(
        runtime,
        "has_device_kernel",
        lambda name, device: (
            name in {"_C_cache_ops::concat_and_cache_mla", "_C_cache_ops::concat_mla_q"}
        ),
    )
    return backend


@pytest.mark.parametrize("name", ["flash_attn_varlen_func", "concat_and_cache_mla"])
@pytest.mark.parametrize("absence", ["blacklisted", "missing"])
def test_unused_flaggems_dependency_does_not_reject_plan(
    monkeypatch, native_prefill, name, absence
):
    import vllm._custom_ops as ops
    import vllm.model_executor.layers.attention.mla_attention as mla

    module = importlib.import_module(
        "flag_gems" if name == "flash_attn_varlen_func" else "flag_gems.fused"
    )
    if absence == "blacklisted":
        monkeypatch.setattr(runtime, "use_flaggems_op", lambda op: op != name)
    else:
        monkeypatch.setattr(module, name, None)
    plan = runtime.validate_hy4_runtime(_prefill_config())
    assert plan.provider == "flaggems"
    assert plan.prefill_backend is native_prefill
    assert name not in plan.operations
    assert "concat_mla_q" not in plan.operations
    native_cache = ops.concat_and_cache_mla
    with runtime.patch_transaction() as tx:
        try:
            runtime._install_fallback(tx, plan)
            assert mla.get_mla_prefill_backend(_prefill_config()) is native_prefill
            assert ops.concat_and_cache_mla is native_cache
        finally:
            tx.rollback()


@pytest.mark.parametrize("absence", ["blacklisted", "missing"])
def test_selected_cache_replacement_must_be_available(
    monkeypatch, native_prefill, absence
):
    import flag_gems.fused as fused

    monkeypatch.setattr(runtime, "has_device_kernel", lambda *a: False)
    if absence == "blacklisted":
        monkeypatch.setattr(
            runtime, "use_flaggems_op", lambda op: op != "concat_and_cache_mla"
        )
    else:
        monkeypatch.setattr(fused, "concat_and_cache_mla", None)
    with pytest.raises(RuntimeError, match="concat_and_cache_mla"):
        runtime.validate_hy4_runtime(_prefill_config())


def _native_capabilities(monkeypatch, missing, hopper=True):
    from vllm import platforms
    from vllm.utils import deep_gemm
    from vllm.v1.attention.ops import flashmla

    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, has_device_capability=lambda cap: hopper),
    )
    monkeypatch.setattr(deep_gemm, "has_deep_gemm", lambda: True)
    monkeypatch.setattr(flashmla, "is_flashmla_sparse_supported", lambda: (True, None))
    seen = []

    def has_kernel(name, device):
        seen.append(name)
        return name not in missing

    monkeypatch.setattr(runtime, "has_device_kernel", has_kernel)
    return seen


@pytest.mark.parametrize(
    "topk,missing,hopper,available",
    [
        (128, "persistent_topk", True, True),
        (128, "cooperative_topk", True, True),
        (128, "top_k_per_row_decode", True, False),
        (2048, "persistent_topk", True, False),
        (2048, "cooperative_topk", True, False),
        (2048, "cooperative_topk", False, True),
        (2048, "top_k_per_row_decode", True, True),
    ],
)
def test_native_topk_checks_only_reachable_branches(
    monkeypatch, topk, missing, hopper, available
):
    seen = _native_capabilities(monkeypatch, {"_C::" + missing}, hopper)
    assert runtime.native_hy4_available(topk) is available
    if topk == 128:
        assert "_C::persistent_topk" not in seen
        assert "_C::cooperative_topk" not in seen
    else:
        assert "_C::top_k_per_row_decode" not in seen


def test_topk_128_resolves_native_without_any_flaggems_dependency(monkeypatch):
    import vllm.model_executor.layers.attention.mla_attention as mla

    _native_capabilities(monkeypatch, {"_C::persistent_topk", "_C::cooperative_topk"})
    backend = SimpleNamespace(is_available=lambda: True, get_name=lambda: "NATIVE")
    monkeypatch.setattr(mla, "get_mla_prefill_backend", lambda cfg: backend)
    monkeypatch.setattr(runtime, "use_flaggems_op", lambda name: False)
    config = _prefill_config()
    config.model_config.hf_text_config.index_topk = 128
    plan = runtime.validate_hy4_runtime(config)
    assert plan.provider == "native"
    assert not plan.operations
    assert plan.prefill_backend is backend


@pytest.mark.parametrize("topk,available", [(128, False), (2048, True)])
def test_native_generic_decode_callable_is_required_only_when_used(
    monkeypatch, topk, available
):
    import vllm._custom_ops as ops

    _native_capabilities(monkeypatch, set())
    monkeypatch.setattr(ops, "top_k_per_row_decode", None)
    assert runtime.native_hy4_available(topk) is available


@pytest.mark.parametrize("error_type", [TypeError, AttributeError])
@pytest.mark.parametrize("during_iteration", [False, True])
def test_expert_mapping_helper_errors_propagate(
    monkeypatch, error_type, during_iteration
):
    error = error_type("helper implementation failed")

    def iterator():
        yield ("partial", "entry", 0, "w1")
        raise error

    def helper(*args, **kwargs):
        if during_iteration:
            return iterator()
        raise error

    monkeypatch.setattr(hy_v4, "fused_moe_make_expert_params_mapping", helper)
    model = SimpleNamespace(config=SimpleNamespace(n_routed_experts=2))
    with pytest.raises(error_type) as exc:
        hy_v4._make_hyv4_expert_params_mapping(model)
    assert exc.value is error


def test_missing_expert_helper_keeps_no_eplb_mapping(monkeypatch):
    monkeypatch.setattr(hy_v4, "fused_moe_make_expert_params_mapping", None)
    model = SimpleNamespace(config=SimpleNamespace(n_routed_experts=2))
    mapping = hy_v4._make_hyv4_expert_params_mapping(model)
    assert len(mapping) == 6
    assert {row[2] for row in mapping} == {0, 1}
    model.n_redundant_experts = 1
    with pytest.raises(RuntimeError, match="EPLB loading requires"):
        hy_v4._make_hyv4_expert_params_mapping(model)


@pytest.mark.parametrize(
    "module_name,fail_at", [("hy4_hc_projection", 1), ("hy_v4_hc", 1), ("hy_v4_hc", 2)]
)
@pytest.mark.parametrize(
    "error_type", [ImportError, RuntimeError, AttributeError, TypeError]
)
def test_hc_registration_errors_propagate(
    monkeypatch, module_name, fail_at, error_type
):
    from vllm.utils import torch_utils

    module = importlib.import_module("vllm_fl.ops." + module_name)
    calls = []
    error = error_type("registration implementation failed")

    def register(**kwargs):
        calls.append(kwargs["op_name"])
        if len(calls) == fail_at:
            raise error

    # Execute the actual module registration with a fake registrar; no Torch
    # operator is registered by this test, including before second-op failure.
    monkeypatch.setattr(torch_utils, "direct_register_custom_op", register)
    with pytest.raises(error_type) as exc:
        runpy.run_path(module.__file__)
    assert exc.value is error
    assert len(calls) == fail_at
