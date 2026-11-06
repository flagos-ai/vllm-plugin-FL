# SPDX-License-Identifier: Apache-2.0
"""Selected dependencies and implementation failures at the HY4 boundary."""

import importlib
from types import SimpleNamespace

import pytest

from .test_hy_v4_review_contracts import _prefill_config
from vllm_fl.models.hy4 import model as hy_v4, runtime


@pytest.mark.parametrize(
    "vendor", ["ascend", "metax", "musa", "hygon", "thead", "kunlunxin", "enflame"]
)
def test_non_cuda_preflight_rejects_before_operator_resolution(monkeypatch, vendor):
    from vllm import platforms

    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(vendor_name=vendor, is_cuda=lambda: False),
    )
    monkeypatch.setattr(
        runtime,
        "_resolve_registered_operator",
        lambda *a, **k: pytest.fail("unsupported platform reached operator resolution"),
    )
    with pytest.raises(ValueError, match="requires an NVIDIA CUDA runtime"):
        runtime.validate_hy4_runtime(_prefill_config())


@pytest.mark.parametrize(
    "options,selected",
    [
        ({}, "flagos"),
        ({"prefer": "vendor"}, "vendor"),
        ({"per_op_order": {"hy4_test": ["vendor", "flagos"]}}, "vendor"),
        ({"prefer": "vendor", "deny_vendors": {"cuda"}}, "flagos"),
        ({"prefer": "vendor", "allow_vendors": {"ascend"}}, "flagos"),
    ],
)
def test_runtime_resolution_obeys_common_policy(monkeypatch, options, selected):
    from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

    calls = []

    def flagos():
        calls.append("flagos")

    def vendor():
        calls.append("vendor")

    with policy_context(SelectionPolicy.from_dict(**options)):
        from vllm_fl.dispatch.manager import get_default_manager
        from vllm_fl.dispatch.types import BackendImplKind, OpImpl

        manager = get_default_manager()
        manager.ensure_initialized()
        manager.registry.register_many(
            [
                OpImpl("hy4_test", "test.flagos", BackendImplKind.DEFAULT, flagos),
                OpImpl(
                    "hy4_test",
                    "test.vendor",
                    BackendImplKind.VENDOR,
                    vendor,
                    vendor="cuda",
                ),
            ]
        )
        implementation = runtime._resolve_registered_operator("hy4_test").fn
        assert not calls
        implementation()
    assert calls == [selected]


@pytest.mark.parametrize("error_type", [NotImplementedError, RuntimeError])
def test_selected_runtime_failure_is_not_retried(monkeypatch, error_type):
    import torch

    from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

    out = torch.zeros(1)
    calls = []
    error = error_type("failure after output mutation")

    def flagos():
        calls.append("flagos")
        out.fill_(1)
        raise error

    def vendor():
        calls.append("vendor")
        out.fill_(2)

    with policy_context(SelectionPolicy()):
        from vllm_fl.dispatch.manager import get_default_manager
        from vllm_fl.dispatch.types import BackendImplKind, OpImpl

        manager = get_default_manager()
        manager.ensure_initialized()
        manager.registry.register_many(
            [
                OpImpl("hy4_test", "test.flagos", BackendImplKind.DEFAULT, flagos),
                OpImpl(
                    "hy4_test",
                    "test.vendor",
                    BackendImplKind.VENDOR,
                    vendor,
                    vendor="cuda",
                ),
            ]
        )
        implementation = runtime._resolve_registered_operator("hy4_test").fn
        with pytest.raises(error_type) as exc:
            implementation()
    assert exc.value is error
    assert calls == ["flagos"]
    assert out.item() == 1


def test_native_paths_obey_vendor_filter_and_per_op_order():
    from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

    with policy_context(SelectionPolicy()):
        assert runtime._native_operator_allowed("concat_mla_q")
    with policy_context(SelectionPolicy.from_dict(deny_vendors={"cuda"})):
        assert not runtime._native_operator_allowed("concat_mla_q")
    with policy_context(
        SelectionPolicy.from_dict(per_op_order={"concat_mla_q": ["flagos"]})
    ):
        assert not runtime._native_operator_allowed("concat_mla_q")


@pytest.mark.parametrize("order,allowed", [(["flagos"], True), (["vendor"], False)])
def test_explicit_flaggems_prefill_obeys_per_op_order(monkeypatch, order, allowed):
    import vllm.model_executor.layers.attention.mla_attention as mla

    from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

    backend = SimpleNamespace(
        is_available=lambda: True, get_name=lambda: "FLAGGEMS_MLA_PREFILL"
    )
    monkeypatch.setattr(mla, "get_mla_prefill_backend", lambda cfg: backend)
    config = _prefill_config()
    config.attention_config.mla_prefill_backend = "CUSTOM"
    policy = SelectionPolicy.from_dict(per_op_order={"mla_prefill_backend": order})
    with policy_context(policy):
        if allowed:
            plan = runtime.validate_hy4_runtime(config)
            assert plan.prefill_backend.get_name() == "FLAGGEMS_MLA_PREFILL"
        else:
            with pytest.raises(RuntimeError, match="disabled by dispatch policy"):
                runtime.validate_hy4_runtime(config)


def test_portable_indexer_rejects_graph_mode_before_weight_loading(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.impl import mla_ops

    original = mla_ops.get_operator
    monkeypatch.setattr(
        mla_ops,
        "get_operator",
        lambda name: (
            (lambda *a, **k: None)
            if name == "fp8_fp4_paged_mqa_logits"
            else original(name)
        ),
    )
    monkeypatch.setattr(runtime, "native_hy4_available", lambda *_: False)
    config = _prefill_config()
    config.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="requires enforce_eager=True"):
        runtime.validate_hy4_runtime(config)


def test_portable_indexer_accepts_public_graph_safe_provider(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.impl import mla_ops

    monkeypatch.setattr(runtime, "native_hy4_available", lambda *_: False)
    original = mla_ops.get_operator
    graph_fn = lambda *args, **kwargs: None
    graph_fn.supports_cuda_graph = True
    monkeypatch.setattr(
        mla_ops,
        "get_operator",
        lambda name: graph_fn if name == "fp8_fp4_paged_mqa_logits" else original(name),
    )
    config = _prefill_config()
    config.model_config.enforce_eager = False
    plan = runtime.validate_hy4_runtime(config)
    assert plan.provider == "flaggems"
    assert plan.operations["fp8_fp4_paged_mqa_logits"]._supports_cuda_graph()


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
    from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

    with policy_context(SelectionPolicy.from_dict(prefer="vendor")):
        yield backend


@pytest.mark.parametrize("name", ["flash_attn_varlen_func", "concat_and_cache_mla"])
@pytest.mark.parametrize("absence", ["blacklisted", "missing"])
def test_unused_flaggems_dependency_does_not_reject_plan(
    monkeypatch, native_prefill, name, absence
):
    import vllm._custom_ops as ops
    import vllm.model_executor.layers.attention.mla_attention as mla

    module = importlib.import_module("flaggems_vllm")
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
    import flaggems_vllm as fused

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


@pytest.mark.usefixtures("prefer_native")
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


@pytest.mark.parametrize("prefer,expected", [("flagos", 1), ("vendor", 2)])
def test_registered_concat_compares_available_backends(monkeypatch, prefer, expected):
    import flaggems_vllm
    import torch

    from vllm import _custom_ops

    from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

    def flagos(a, b, out):
        out.fill_(1)

    def native(a, b, out):
        out.fill_(2)

    monkeypatch.setattr(flaggems_vllm, "concat_mla_q", flagos)
    monkeypatch.setattr(_custom_ops, "concat_mla_q", native)
    monkeypatch.setattr(runtime, "has_device_kernel", lambda *a: True)
    monkeypatch.setattr(runtime, "use_flaggems_op", lambda *a: True)
    with policy_context(SelectionPolicy.from_dict(prefer=prefer)):
        implementation = runtime._resolve_registered_operator("concat_mla_q")
        assert implementation.impl_id == (
            "default.flagos" if prefer == "flagos" else "vendor.cuda"
        )
        out = torch.zeros(1)
        implementation.fn(out, out, out)
        assert out.item() == expected
