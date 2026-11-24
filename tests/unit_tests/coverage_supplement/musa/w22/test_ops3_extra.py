# Coverage tests: vllm_fl/ops/layernorm.py, vllm_fl/ops/fused_moe/activation.py,
# plus remaining branch gaps in fused_moe_utils.py / router.py (CPU only).

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

import vllm_fl.ops.fused_moe.activation as fm_activation
import vllm_fl.ops.fused_moe.fused_moe_utils as mu
import vllm_fl.ops.fused_moe.router as router_mod
import vllm_fl.ops.layernorm as layernorm


# ---------------------------------------------------------------------------
# layernorm.py
# ---------------------------------------------------------------------------


def _norm_instance(amd=False):
    instance = object.__new__(layernorm.RMSNormFL)
    instance._use_amd_graph_ops = amd
    instance.weight = torch.ones(4)
    instance.variance_epsilon = 1e-6
    return instance


def test_rmsnorm_init_sets_amd_flag(monkeypatch):
    def fake_init(self, hidden_size, eps=1e-6, var_hidden_size=None,
                  has_weight=True, dtype=None):
        self.variance_epsilon = eps
        self.weight = torch.ones(hidden_size)

    monkeypatch.setattr(layernorm.RMSNorm, "__init__", fake_init)

    monkeypatch.setattr(
        layernorm, "current_platform", SimpleNamespace(vendor_name="amd")
    )
    assert layernorm.RMSNormFL(8, eps=1e-5)._use_amd_graph_ops is True

    monkeypatch.setattr(
        layernorm, "current_platform", SimpleNamespace(vendor_name="mthreads")
    )
    assert layernorm.RMSNormFL(8)._use_amd_graph_ops is False

    # No vendor_name attribute at all -> getattr default.
    monkeypatch.setattr(layernorm, "current_platform", SimpleNamespace())
    assert layernorm.RMSNormFL(8)._use_amd_graph_ops is False


def test_rmsnorm_forward_oot_dispatches_to_cached_op(monkeypatch):
    instance = _norm_instance(amd=False)
    seen = []
    monkeypatch.setattr(
        layernorm,
        "_rms_norm",
        lambda layer, x, residual: seen.append((layer, x, residual)) or x,
    )
    x = torch.zeros(2, 4)
    assert layernorm.RMSNormFL.forward_oot(instance, x) is x
    assert len(seen) == 1
    assert seen[0][0] is instance
    assert seen[0][1] is x
    assert seen[0][2] is None


def _stub_amd_graph_ops(monkeypatch, op):
    package = ModuleType("vllm_fl.dispatch.backends.vendor.amd")
    package.__path__ = []
    graph_ops = ModuleType("vllm_fl.dispatch.backends.vendor.amd.graph_ops")
    graph_ops.rms_norm_custom_op = op
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, graph_ops.__name__, graph_ops)


def test_rmsnorm_amd_not_compiling_uses_cached_op(monkeypatch):
    instance = _norm_instance(amd=True)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    sentinel = torch.ones(2, 4)
    monkeypatch.setattr(layernorm, "_rms_norm", lambda *a, **k: sentinel)
    assert layernorm.RMSNormFL.forward_oot(instance, torch.zeros(2, 4)) is sentinel


def test_rmsnorm_amd_compiling_without_residual_returns_tensor(monkeypatch):
    instance = _norm_instance(amd=True)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    out = torch.ones(2, 4)
    calls = []
    _stub_amd_graph_ops(
        monkeypatch,
        lambda x, residual, weight, eps: calls.append((x, residual, weight, eps))
        or (out, None),
    )
    x = torch.zeros(2, 4)
    assert layernorm.RMSNormFL.forward_oot(instance, x) is out
    assert calls[0][0] is x
    assert calls[0][1] is None


def test_rmsnorm_amd_compiling_with_residual_returns_pair(monkeypatch):
    instance = _norm_instance(amd=True)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    out = torch.ones(2, 4)
    residual_out = torch.full((2, 4), 2.0)
    _stub_amd_graph_ops(monkeypatch, lambda *a: (out, residual_out))
    result = layernorm.RMSNormFL.forward_oot(
        instance, torch.zeros(2, 4), torch.zeros(2, 4)
    )
    assert result[0] is out
    assert result[1] is residual_out


def test_rmsnorm_forward_cuda_delegates(monkeypatch):
    instance = _norm_instance()
    sentinel = object()
    monkeypatch.setattr(
        layernorm.RMSNormFL, "forward_oot", lambda self, x, residual=None: sentinel
    )
    assert layernorm.RMSNormFL.forward_cuda(instance, torch.zeros(1, 4)) is sentinel


# ---------------------------------------------------------------------------
# fused_moe/activation.py
# ---------------------------------------------------------------------------


def test_apply_moe_activation_requires_2d():
    with pytest.raises(AssertionError):
        fm_activation.apply_moe_activation(
            MoEActivation.SILU, torch.zeros(1, 4), torch.zeros(1, 1, 8)
        )
    with pytest.raises(AssertionError):
        fm_activation.apply_moe_activation(
            MoEActivation.SILU, torch.zeros(1, 1, 4), torch.zeros(1, 8)
        )


def test_apply_moe_activation_gated_size_mismatch():
    with pytest.raises(AssertionError):
        fm_activation.apply_moe_activation(
            MoEActivation.SILU, torch.zeros(2, 3), torch.zeros(2, 8)
        )


def test_apply_moe_activation_non_gated_size_mismatch():
    fake = SimpleNamespace(is_gated=False, value="linear")
    with pytest.raises(AssertionError):
        fm_activation.apply_moe_activation(
            fake, torch.zeros(2, 5), torch.zeros(2, 4)
        )


def test_apply_moe_activation_silu_fast_path(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fm_activation,
        "_silu_and_mul",
        lambda prefix, x: calls.append((prefix, x)) or torch.full((2, 4), 2.0),
    )
    out = torch.zeros(2, 4)
    inp = torch.ones(2, 8)
    result = fm_activation.apply_moe_activation(MoEActivation.SILU, out, inp)
    assert result is out
    assert calls[0][0] is None
    assert calls[0][1] is inp
    assert torch.equal(out, torch.full((2, 4), 2.0))


def test_apply_moe_activation_silu_none_clamp_uses_fast_path(monkeypatch):
    monkeypatch.setattr(
        fm_activation, "_silu_and_mul", lambda prefix, x: torch.ones(2, 4)
    )
    out = torch.zeros(2, 4)
    config = SimpleNamespace(clamp_limit=None)
    fm_activation.apply_moe_activation(
        MoEActivation.SILU, out, torch.ones(2, 8), activation_config=config
    )
    assert torch.equal(out, torch.ones(2, 4))


def test_apply_moe_activation_silu_with_clamp_delegates(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        fm_activation,
        "upstream_apply_moe_activation",
        lambda *a, **k: seen.update(args=a, kwargs=k),
    )
    monkeypatch.setattr(
        fm_activation, "_silu_and_mul", lambda *a: pytest.fail("must not run")
    )
    config = SimpleNamespace(clamp_limit=3.0)
    out = torch.zeros(2, 4)
    fm_activation.apply_moe_activation(
        MoEActivation.SILU,
        out,
        torch.ones(2, 8),
        activation_config=config,
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
        expert_map=None,
    )
    assert seen["kwargs"]["activation_config"] is config
    assert seen["kwargs"]["topk_ids"] is not None


def test_apply_moe_activation_gelu_uses_kernel(monkeypatch):
    seen = []
    monkeypatch.setattr(
        fm_activation,
        "_gelu_and_mul",
        lambda prefix, x: seen.append((prefix, x)) or torch.full((2, 4), 3.0),
    )
    out = torch.zeros(2, 4)
    # GELU is a gated activation in this vLLM build: output 4 needs input 8.
    fm_activation.apply_moe_activation(MoEActivation.GELU, out, torch.ones(2, 8))
    assert seen[0][0] is None
    assert torch.equal(out, torch.full((2, 4), 3.0))


def test_apply_moe_activation_other_variant_delegates(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        fm_activation,
        "upstream_apply_moe_activation",
        lambda *a, **k: seen.update(args=a, kwargs=k),
    )
    fake = SimpleNamespace(is_gated=True, value="swigluoai")
    out = torch.zeros(2, 4)
    result = fm_activation.apply_moe_activation(
        fake,
        out,
        torch.ones(2, 8),
        activation_config=None,
        topk_ids=None,
        expert_map=None,
    )
    assert result is out
    assert seen["args"][0] is fake
    assert seen["args"][1] is out


# ---------------------------------------------------------------------------
# fused_moe_utils.py remaining branches
# ---------------------------------------------------------------------------


def _plat(**overrides):
    values = {
        "is_tpu": lambda: False,
        "is_rocm": lambda: False,
        "is_cuda": lambda: False,
        "is_xpu": lambda: False,
        "is_cpu": lambda: False,
        "is_out_of_tree": lambda: False,
        "is_device_capability_family": lambda cap: False,
        "device_type": "cuda",
        "vendor_name": "nvidia",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _moe_config(**overrides):
    values = {
        "moe_parallel_config": SimpleNamespace(
            dp_size=1, use_ep=False, use_batched_activation_format=False
        ),
        "activation": mu.MoEActivation.SILU,
        "moe_backend": "auto",
        "is_lora_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _kernel_cls(supported, reason=None):
    class Kernel:
        def is_supported_config(self, config, a, b, activation_format):
            return supported, reason

    return Kernel


def test_should_use_fl_triton_experts_out_of_tree(monkeypatch):
    monkeypatch.setattr(
        mu, "current_platform", _plat(is_out_of_tree=lambda: True)
    )
    monkeypatch.setattr(mu, "use_flaggems", lambda: True)
    monkeypatch.setattr(mu, "get_oot_blacklist", lambda: None)
    assert mu._should_use_fl_triton_experts() is True


def test_prepare_expert_assignment_zero_block_width_keeps_naive_path():
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    sorted_ids, flat_ids, padded = mu._prepare_expert_assignment(
        topk_ids,
        {"BLOCK_SIZE_M": 4},
        num_tokens=1,
        top_k_num=2,
        global_num_experts=100,
        expert_map=None,
        use_int8_w8a16=True,
        block_shape=[128, 0],
    )
    assert sorted_ids is None
    assert torch.equal(flat_ids, topk_ids.view(-1))
    assert padded.tolist() == [2 * 4]


def test_prepare_expert_assignment_forwards_ignore_invalid(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        mu,
        "_moe_align_block_size",
        lambda *args, **kwargs: captured.update(kwargs) or (1, 2, 3),
    )
    result = mu._prepare_expert_assignment(
        torch.zeros((2, 2), dtype=torch.int32),
        {"BLOCK_SIZE_M": 8},
        num_tokens=2,
        top_k_num=2,
        global_num_experts=4,
        expert_map=torch.arange(4),
        ignore_invalid_experts=True,
    )
    assert result == (1, 2, 3)
    assert captured["ignore_invalid_experts"] is True


def test_select_backend_humming_uses_priority_list(monkeypatch):
    monkeypatch.setattr(mu, "current_platform", _plat(is_cuda=lambda: True))
    monkeypatch.setattr(mu, "_should_use_fl_triton_experts", lambda: False)
    monkeypatch.setattr(mu.envs, "is_set", lambda name: False)
    monkeypatch.setattr(
        mu, "_get_priority_backends", lambda config: [mu.UnquantizedMoeBackend.TRITON]
    )
    kernel = _kernel_cls(True)
    monkeypatch.setattr(mu, "backend_to_kernel_cls", lambda backend: [kernel])

    assert mu.select_unquantized_moe_backend_oot(
        _moe_config(moe_backend="humming")
    ) == (mu.UnquantizedMoeBackend.TRITON, kernel)


def test_select_explicit_backend_tries_next_kernel(monkeypatch):
    monkeypatch.setattr(mu, "current_platform", _plat(is_cuda=lambda: True))
    monkeypatch.setattr(mu, "_should_use_fl_triton_experts", lambda: False)
    monkeypatch.setattr(
        mu, "map_unquantized_backend", lambda name: mu.UnquantizedMoeBackend.TRITON
    )
    bad = _kernel_cls(False, "no shape support")
    good = _kernel_cls(True)
    monkeypatch.setattr(mu, "backend_to_kernel_cls", lambda backend: [bad, good])

    assert mu.select_unquantized_moe_backend_oot(
        _moe_config(moe_backend="custom")
    ) == (mu.UnquantizedMoeBackend.TRITON, good)


def test_select_backend_skips_aiter_when_moe_flag_disabled(monkeypatch):
    monkeypatch.setattr(mu, "current_platform", _plat(is_cuda=lambda: True))
    monkeypatch.setattr(mu, "_should_use_fl_triton_experts", lambda: False)
    monkeypatch.setattr(mu.envs, "is_set", lambda name: True)
    monkeypatch.setattr(mu.envs, "VLLM_ROCM_USE_AITER", True, raising=False)
    monkeypatch.setattr(mu.envs, "VLLM_ROCM_USE_AITER_MOE", False, raising=False)
    monkeypatch.setattr(
        mu.rocm_aiter_ops, "is_rdna_aiter_enabled", lambda: False
    )
    kernel = _kernel_cls(True)
    monkeypatch.setattr(mu, "backend_to_kernel_cls", lambda backend: [kernel])
    monkeypatch.setattr(
        mu,
        "_get_priority_backends",
        lambda config: [mu.UnquantizedMoeBackend.AITER, mu.UnquantizedMoeBackend.TRITON],
    )
    assert mu.select_unquantized_moe_backend_oot(_moe_config()) == (
        mu.UnquantizedMoeBackend.TRITON,
        kernel,
    )


def _quant_config(**overrides):
    values = {
        "quant_dtype": None,
        "weight_quant_dtype": None,
        "use_fp8_w8a8": False,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "config_name": lambda dtype: "cfg",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _experts(**overrides):
    values = {
        "_lora_context": None,
        "quant_config": _quant_config(),
        "per_act_token_quant": False,
        "block_shape": None,
        "w1_scale": None,
        "w2_scale": None,
        "w1_bias": None,
        "w2_bias": None,
        "quant_dtype": None,
        "quantization_emulation": False,
        "activation_config": SimpleNamespace(clamp_limit=None),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _apply_args(hidden):
    return {
        "output": torch.zeros(hidden.shape, dtype=hidden.dtype),
        "hidden_states": hidden,
        "w1": torch.zeros((2, 8, 4), dtype=hidden.dtype),
        "w2": torch.zeros((2, 4, 4), dtype=hidden.dtype),
        "topk_weights": torch.ones((hidden.shape[0], 1), dtype=torch.float32),
        "topk_ids": torch.zeros((hidden.shape[0], 1), dtype=torch.int64),
        "activation": mu.MoEActivation.SILU,
        "global_num_experts": -1,
        "expert_map": None,
        "a1q_scale": None,
        "a2_scale": None,
        "workspace13": torch.empty(0),
        "workspace2": torch.empty(0),
        "expert_tokens_meta": None,
        "apply_router_weight_on_input": False,
    }


def _patch_pipeline(monkeypatch, instance, captured):
    monkeypatch.setattr(
        mu, "try_get_optimal_moe_config", lambda *a, **k: {"BLOCK_SIZE_M": 16}
    )
    monkeypatch.setattr(
        mu, "_resize_cache", lambda buffer, shape: torch.empty(shape)
    )

    def prepare(*args, **kwargs):
        captured["prepare_args"] = args
        captured["prepare_kwargs"] = kwargs
        return (None, None, None)

    monkeypatch.setattr(mu, "_prepare_expert_assignment", prepare)
    monkeypatch.setattr(
        mu,
        "_invoke_fused_moe_triton_kernel",
        lambda *a, **k: captured.update(kernel_kwargs=k),
    )
    monkeypatch.setattr(mu, "apply_moe_activation", lambda *a, **k: None)
    monkeypatch.setattr(
        mu, "moe_kernel_quantize_input", lambda A, *a, **k: (A, None)
    )
    instance.moe_problem_size = lambda hidden, w1, w2, ids: (2, 2, 8, 4, 1)
    instance.adjust_N_for_activation = lambda N, activation: N // 2
    instance.moe_sum = lambda tensor, output: None


@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.int8],
)
def test_apply_pipeline_low_precision_compute_type(monkeypatch, dtype):
    captured = {}
    instance = _experts()
    _patch_pipeline(monkeypatch, instance, captured)
    monkeypatch.setattr(mu, "current_platform", _plat())

    hidden = torch.zeros((2, 4), dtype=dtype)
    args = _apply_args(hidden)
    mu.TritonExpertsFL.apply(instance, **args)

    assert captured["prepare_args"][4] == 2
    assert captured["kernel_kwargs"]["compute_type"] is mu.tl.bfloat16
    assert captured["kernel_kwargs"]["use_int4_w4a16"] is False


def test_apply_pipeline_int4_checks_packed_hidden_size(monkeypatch):
    captured = {}
    instance = _experts(quant_config=_quant_config(use_int4_w4a16=True))
    _patch_pipeline(monkeypatch, instance, captured)
    monkeypatch.setattr(mu, "current_platform", _plat())

    ok = _apply_args(torch.zeros((2, 8), dtype=torch.bfloat16))
    mu.TritonExpertsFL.apply(instance, **ok)
    assert captured["prepare_kwargs"]["use_int4_w4a16"] is True

    bad = _apply_args(torch.zeros((2, 4), dtype=torch.bfloat16))
    with pytest.raises(AssertionError):
        mu.TritonExpertsFL.apply(instance, **bad)


# ---------------------------------------------------------------------------
# router.py remaining branches
# ---------------------------------------------------------------------------


def test_sqrtsoftplus_hash_table_matching_dtype():
    hash_table = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
    weights, ids = router_mod._sqrtsoftplus_topk(
        gating_output=torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32),
        e_score_correction_bias=None,
        topk=2,
        renormalize=False,
        indices_type=None,
        input_tokens=torch.tensor([1], dtype=torch.int64),
        hash_indices_table=hash_table,
        routed_scaling_factor=1.0,
    )
    assert torch.equal(ids, torch.tensor([[1, 2]], dtype=torch.int32))
    assert ids.dtype == torch.int32
    assert weights.shape == (1, 2)


def test_fused_topk_honours_indices_type(monkeypatch):
    monkeypatch.setattr(router_mod, "_is_ascend_npu_tensor", lambda value: False)
    monkeypatch.setattr(router_mod, "_has_native_moe_topk", lambda: False)
    monkeypatch.setattr(router_mod, "_get_padding_mask", lambda n: None)
    monkeypatch.setattr(
        router_mod, "_topk_softmax", lambda w, i, t, g, r: (w, i)
    )
    weights, ids, token_expert_indices = router_mod.fused_topk(
        hidden_states=torch.zeros(2, 4),
        gating_output=torch.zeros(2, 3),
        topk=1,
        renormalize=False,
        indices_type=torch.int64,
    )
    assert ids.dtype == torch.int64
    assert token_expert_indices.dtype == torch.int32
    assert weights.shape == (2, 1)


def test_grouped_router_invalid_grouping_when_not_divisible(monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "rocm_aiter_ops",
        SimpleNamespace(is_fused_moe_enabled=lambda: False),
    )
    monkeypatch.setattr(
        router_mod,
        "fused_topk",
        lambda **kwargs: (
            torch.ones(1, 2),
            torch.zeros(1, 2, dtype=torch.int32),
            None,
        ),
    )
    router = SimpleNamespace(
        num_expert_group=2,
        topk_group=1,
        top_k=2,
        renormalize=True,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        num_fused_shared_experts=0,
    )
    # 5 experts > 2 groups, but 5 % 2 != 0 -> invalid grouping fallback.
    weights, ids = router_mod.GroupedTopKRouterFL._compute_routing(
        router, torch.zeros(1, 5), torch.zeros(1, 5), torch.int32
    )
    assert weights.shape == (1, 2)
    assert ids.shape == (1, 2)


def test_fl_grouped_topk_no_bias_unit_scaling(monkeypatch):
    gating = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=torch.float32
    )
    weights, ids = router_mod._fl_grouped_topk(
        hidden_states=torch.zeros(2, 4),
        gating_output=gating,
        topk=2,
        renormalize=False,
        num_expert_group=2,
        topk_group=1,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
    )
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    assert weights.shape == (2, 2)
