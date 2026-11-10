from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


_HYGON_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl"
    / "dispatch"
    / "backends"
    / "vendor"
    / "hygon"
)
_MODEL_PATCH_PATH = _HYGON_PATH / "patches" / "models" / "deepseek_v4.py"
_PATCH_SPEC = importlib.util.spec_from_file_location(
    "_vllm_fl_hygon_model_patch_test_module", _MODEL_PATCH_PATH
)
assert _PATCH_SPEC is not None and _PATCH_SPEC.loader is not None
_PATCH_MODULE = importlib.util.module_from_spec(_PATCH_SPEC)
_PATCH_SPEC.loader.exec_module(_PATCH_MODULE)

_CUSTOM_OP_PATCH_PATH = _HYGON_PATH / "patches" / "custom_ops.py"
_CUSTOM_OP_PATCH_SPEC = importlib.util.spec_from_file_location(
    "_vllm_fl_hygon_custom_op_patch_test_module", _CUSTOM_OP_PATCH_PATH
)
assert _CUSTOM_OP_PATCH_SPEC is not None and _CUSTOM_OP_PATCH_SPEC.loader is not None
_CUSTOM_OP_PATCH_MODULE = importlib.util.module_from_spec(_CUSTOM_OP_PATCH_SPEC)
_CUSTOM_OP_PATCH_SPEC.loader.exec_module(_CUSTOM_OP_PATCH_MODULE)

_install_compressed_tensors_scale_fallback = (
    _PATCH_MODULE._install_compressed_tensors_scale_fallback
)
_scale_inv_alias = _PATCH_MODULE._scale_inv_alias
_patch_deepseek_v4_cutedsl_selection = (
    _PATCH_MODULE._patch_deepseek_v4_cutedsl_selection
)
_patch_deepseek_v4_bf16_indexer_cache = (
    _PATCH_MODULE._patch_deepseek_v4_bf16_indexer_cache
)
_patch_deepseek_v4_qnorm_rope_kv_insert = (
    _PATCH_MODULE._patch_deepseek_v4_qnorm_rope_kv_insert
)
_patch_sparse_attn_indexer = _PATCH_MODULE._patch_sparse_attn_indexer
_patch_mhc_ops = _PATCH_MODULE._patch_mhc_ops
_patch_custom_op_dispatch = _CUSTOM_OP_PATCH_MODULE._patch_custom_op_dispatch

_TOPK_IMPL_PATH = _HYGON_PATH / "impl" / "native_ops" / "topk_softplus_sqrt.py"
_TOPK_IMPL_SPEC = importlib.util.spec_from_file_location(
    "_vllm_fl_hygon_topk_test_module", _TOPK_IMPL_PATH
)
assert _TOPK_IMPL_SPEC is not None and _TOPK_IMPL_SPEC.loader is not None
_TOPK_IMPL_MODULE = importlib.util.module_from_spec(_TOPK_IMPL_SPEC)
_TOPK_IMPL_SPEC.loader.exec_module(_TOPK_IMPL_MODULE)

_ALIGN_IMPL_PATH = _HYGON_PATH / "impl" / "native_ops" / "moe_align_block_size.py"
_ALIGN_IMPL_SPEC = importlib.util.spec_from_file_location(
    "_vllm_fl_hygon_align_test_module", _ALIGN_IMPL_PATH
)
assert _ALIGN_IMPL_SPEC is not None and _ALIGN_IMPL_SPEC.loader is not None
_ALIGN_IMPL_MODULE = importlib.util.module_from_spec(_ALIGN_IMPL_SPEC)
_ALIGN_IMPL_SPEC.loader.exec_module(_ALIGN_IMPL_MODULE)

_ACTIVATION_IMPL_PATH = _HYGON_PATH / "impl" / "native_ops" / "activation.py"
_ACTIVATION_IMPL_SPEC = importlib.util.spec_from_file_location(
    "_vllm_fl_hygon_activation_test_module", _ACTIVATION_IMPL_PATH
)
assert _ACTIVATION_IMPL_SPEC is not None and _ACTIVATION_IMPL_SPEC.loader is not None
_ACTIVATION_IMPL_MODULE = importlib.util.module_from_spec(_ACTIVATION_IMPL_SPEC)
_ACTIVATION_IMPL_SPEC.loader.exec_module(_ACTIVATION_IMPL_MODULE)

_NORMALIZATION_IMPL_PATH = _HYGON_PATH / "impl" / "custom_ops" / "normalization.py"
_NORMALIZATION_IMPL_SPEC = importlib.util.spec_from_file_location(
    "_vllm_fl_hygon_normalization_test_module", _NORMALIZATION_IMPL_PATH
)
assert (
    _NORMALIZATION_IMPL_SPEC is not None
    and _NORMALIZATION_IMPL_SPEC.loader is not None
)
_NORMALIZATION_IMPL_MODULE = importlib.util.module_from_spec(
    _NORMALIZATION_IMPL_SPEC
)
_NORMALIZATION_IMPL_SPEC.loader.exec_module(_NORMALIZATION_IMPL_MODULE)

_MOE_SUM_IMPL_PATH = _HYGON_PATH / "impl" / "native_ops" / "moe_sum.py"
_MOE_SUM_IMPL_SPEC = importlib.util.spec_from_file_location(
    "_vllm_fl_hygon_moe_sum_test_module", _MOE_SUM_IMPL_PATH
)
assert _MOE_SUM_IMPL_SPEC is not None and _MOE_SUM_IMPL_SPEC.loader is not None
_MOE_SUM_IMPL_MODULE = importlib.util.module_from_spec(_MOE_SUM_IMPL_SPEC)
_MOE_SUM_IMPL_SPEC.loader.exec_module(_MOE_SUM_IMPL_MODULE)

_MHC_IMPL_PATH = _HYGON_PATH / "impl" / "other" / "mhc.py"
_MHC_IMPL_SPEC = importlib.util.spec_from_file_location(
    "_vllm_fl_hygon_mhc_test_module", _MHC_IMPL_PATH
)
assert _MHC_IMPL_SPEC is not None and _MHC_IMPL_SPEC.loader is not None
_MHC_IMPL_MODULE = importlib.util.module_from_spec(_MHC_IMPL_SPEC)
_MHC_IMPL_SPEC.loader.exec_module(_MHC_IMPL_MODULE)


class _QuantConfig:
    def __init__(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name


class _VllmConfig:
    def __init__(self, quantization: str, parameters: dict[str, object]):
        self.quant_config = _QuantConfig(quantization)
        self.parameters = parameters


def test_hygon_custom_op_dispatch_prefers_only_explicit_oot_replacements(
    monkeypatch,
):
    class FakePlatform:
        vendor_name = "hygon"

        @staticmethod
        def is_out_of_tree():
            return True

    class FakeCustomOp:
        name = "FakeCustomOp"

        def __init__(self, enabled=True):
            self.enabled = enabled

        def forward_native(self):
            return "native"

        def forward_hip(self):
            return "hip"

        def forward_oot(self):
            return self.forward_native()

        def dispatch_forward(self, compile_native):
            del compile_native
            if not self.enabled:
                return self.forward_native
            return self.forward_hip

    class ExplicitOOTOp(FakeCustomOp):
        name = "ExplicitOOTOp"

        def forward_oot(self):
            return "oot"

    class RegisteredWithoutOOT(FakeCustomOp):
        name = "RegisteredWithoutOOT"

    class UnregisteredOp(FakeCustomOp):
        name = "UnregisteredOp"

    fake_custom_op_mod = ModuleType("vllm.model_executor.custom_op")
    fake_custom_op_mod.CustomOp = FakeCustomOp
    fake_custom_op_mod.current_platform = FakePlatform()
    fake_custom_op_mod.op_registry_oot = {
        "ExplicitOOTOp": ExplicitOOTOp,
        "RegisteredWithoutOOT": RegisteredWithoutOOT,
    }

    real_import_module = importlib.import_module

    def fake_import_module(name):
        if name == "vllm.model_executor.custom_op":
            return fake_custom_op_mod
        return real_import_module(name)

    monkeypatch.setattr(_PATCH_MODULE.importlib, "import_module", fake_import_module)

    _patch_custom_op_dispatch()
    _patch_custom_op_dispatch()  # idempotent

    assert ExplicitOOTOp().dispatch_forward(False)() == "oot"
    assert UnregisteredOp().dispatch_forward(False)() == "hip"
    assert RegisteredWithoutOOT().dispatch_forward(False)() == "hip"
    assert ExplicitOOTOp(enabled=False).dispatch_forward(False)() == "native"

    fake_custom_op_mod.current_platform.vendor_name = "amd"
    assert ExplicitOOTOp().dispatch_forward(False)() == "hip"


def _make_model_class():
    class FakeDeepseekV4Model:
        def __init__(self, *, vllm_config):
            self._parameters = vllm_config.parameters
            self.loaded_parameter_objects: dict[str, object] = {}

        def named_parameters(self, *args, **kwargs):
            yield from self._parameters.items()

        def load_weights(self, weights):
            params_dict = dict(self.named_parameters())
            loaded_params = set()
            for name, _ in weights:
                self.loaded_parameter_objects[name] = params_dict[name]
                loaded_params.add(name)
            return loaded_params

    return FakeDeepseekV4Model


def test_scale_inv_alias_supports_linear_and_moe_names():
    assert _scale_inv_alias("attn.weight_scale") == "attn.weight_scale_inv"
    assert (
        _scale_inv_alias("experts.w13_weight_scale") == "experts.w13_weight_scale_inv"
    )
    assert _scale_inv_alias("attn.weight_scale_inv") is None
    assert _scale_inv_alias("weight") is None


def test_compressed_tensors_fallback_is_target_aware_and_mixed_safe():
    model_cls = _make_model_class()
    _install_compressed_tensors_scale_fallback(model_cls)
    _install_compressed_tensors_scale_fallback(model_cls)

    int8_linear = object()
    int8_moe = object()
    fp8_linear = object()
    model = model_cls(
        vllm_config=_VllmConfig(
            "compressed-tensors",
            {
                "attn.weight_scale": int8_linear,
                "experts.w13_weight_scale": int8_moe,
                "compressor.weight_scale_inv": fp8_linear,
            },
        )
    )

    loaded = model.load_weights(
        [
            ("attn.weight_scale_inv", object()),
            ("experts.w13_weight_scale_inv", object()),
            ("compressor.weight_scale_inv", object()),
        ]
    )

    assert loaded == {
        "attn.weight_scale",
        "experts.w13_weight_scale",
        "compressor.weight_scale_inv",
    }
    assert model.loaded_parameter_objects["attn.weight_scale_inv"] is int8_linear
    assert model.loaded_parameter_objects["experts.w13_weight_scale_inv"] is int8_moe
    assert model.loaded_parameter_objects["compressor.weight_scale_inv"] is fp8_linear
    assert "named_parameters" not in vars(model)


def test_fallback_is_disabled_for_other_quantization_methods():
    model_cls = _make_model_class()
    _install_compressed_tensors_scale_fallback(model_cls)
    model = model_cls(
        vllm_config=_VllmConfig(
            "fp8",
            {"attn.weight_scale": object()},
        )
    )

    with pytest.raises(KeyError, match="attn.weight_scale_inv"):
        model.load_weights([("attn.weight_scale_inv", object())])

    assert "named_parameters" not in vars(model)


def test_fallback_restores_named_parameters_after_loading_error():
    model_cls = _make_model_class()
    _install_compressed_tensors_scale_fallback(model_cls)
    model = model_cls(vllm_config=_VllmConfig("compressed-tensors", {}))

    with pytest.raises(KeyError, match="missing.weight_scale_inv"):
        model.load_weights([("missing.weight_scale_inv", object())])

    assert "named_parameters" not in vars(model)


def test_topk_softplus_sqrt_does_not_call_flaggems(monkeypatch):
    fake_flaggems = ModuleType("flag_gems")

    def fake_topk(*args):
        raise AssertionError("FlagGems MoE topk must not be called")

    fake_flaggems.topk_softplus_sqrt = fake_topk
    monkeypatch.setitem(sys.modules, "flag_gems", fake_flaggems)

    logits = torch.tensor([[0.0, 1.0, -1.0, 2.0]])
    weights = torch.empty(1, 2)
    indices = torch.empty(1, 2, dtype=torch.int32)
    token_expert_indices = torch.empty_like(indices)
    bias = torch.tensor([0.0, 10.0, 0.0, 0.0])

    _TOPK_IMPL_MODULE.topk_softplus_sqrt_hygon(
        weights,
        indices,
        token_expert_indices,
        logits,
        True,
        1.5,
        bias,
    )

    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    expected_indices = torch.topk(scores + bias, k=2, dim=-1).indices
    expected_weights = scores.gather(1, expected_indices)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(indices, expected_indices.to(torch.int32))
    torch.testing.assert_close(weights, expected_weights * 1.5)


def test_topk_softplus_sqrt_matches_vllm_semantics():
    logits = torch.tensor([[0.0, 1.0, -1.0, 2.0]])
    bias = torch.tensor([[0.0, 10.0, 0.0, 0.0]]).squeeze(0)
    weights = torch.empty(1, 2)
    indices = torch.empty(1, 2, dtype=torch.int32)
    token_expert_indices = torch.empty_like(indices)

    _TOPK_IMPL_MODULE.topk_softplus_sqrt_hygon(
        weights,
        indices,
        token_expert_indices,
        logits,
        True,
        2.0,
        bias,
        None,
        None,
    )

    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    expected_indices = torch.topk(scores + bias, k=2, dim=-1).indices
    expected_weights = scores.gather(1, expected_indices)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(indices, expected_indices.to(torch.int32))
    torch.testing.assert_close(weights, expected_weights * 2.0)


def test_topk_softplus_sqrt_preserves_hash_routing_semantics():
    logits = torch.tensor(
        [
            [0.0, 1.0, -1.0, 2.0],
            [1.5, -0.5, 0.5, 0.0],
        ]
    )
    input_ids = torch.tensor([1, 3], dtype=torch.int32)
    tid2eid = torch.tensor(
        [
            [0, 1],
            [3, 0],
            [1, 2],
            [2, 1],
        ],
        dtype=torch.int32,
    )
    weights = torch.empty(2, 2)
    indices = torch.empty(2, 2, dtype=torch.int32)
    token_expert_indices = torch.empty_like(indices)

    _TOPK_IMPL_MODULE.topk_softplus_sqrt_hygon(
        weights,
        indices,
        token_expert_indices,
        logits,
        True,
        1.25,
        None,
        input_ids,
        tid2eid,
    )

    expected_indices = tid2eid[input_ids.long()]
    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    expected_weights = scores.gather(1, expected_indices.long())
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights * 1.25)


def _align_tensors():
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    sorted_token_ids = torch.empty(16, dtype=torch.int32)
    expert_ids = torch.empty(4, dtype=torch.int32)
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32)
    return topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad


def test_moe_align_block_size_prefers_lightop(monkeypatch):
    called = {}
    fake_lightop = ModuleType("lightop")
    fake_lightop_moe = ModuleType("lightop.moe")

    def fake_align(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs

    fake_lightop_moe.moe_align_block_size_out = fake_align
    monkeypatch.setitem(sys.modules, "lightop", fake_lightop)
    monkeypatch.setitem(sys.modules, "lightop.moe", fake_lightop_moe)

    topk_ids, sorted_ids, expert_ids, num_post_pad = _align_tensors()
    _ALIGN_IMPL_MODULE.moe_align_block_size_hygon_out(
        topk_ids,
        2,
        4,
        sorted_ids,
        expert_ids,
        num_post_pad,
    )

    assert called["args"] == (
        topk_ids,
        2,
        4,
        sorted_ids,
        expert_ids,
        num_post_pad,
        None,
    )
    assert called["kwargs"] == {"is_ep": False, "is_fuse_fill": True}


def test_moe_align_block_size_requires_lightop(monkeypatch):
    fake_flaggems = ModuleType("flag_gems")

    def fake_align(*args):
        raise AssertionError("vendor:hygon must not call FlagGems")

    fake_flaggems.moe_align_block_size_triton = fake_align
    monkeypatch.setitem(sys.modules, "lightop", None)
    monkeypatch.setitem(sys.modules, "flag_gems", fake_flaggems)

    topk_ids, sorted_ids, expert_ids, num_post_pad = _align_tensors()
    with pytest.raises(ModuleNotFoundError, match="lightop"):
        _ALIGN_IMPL_MODULE.moe_align_block_size_hygon_out(
            topk_ids,
            2,
            4,
            sorted_ids,
            expert_ids,
            num_post_pad,
        )


def test_moe_silu_prefers_vllm_hcu_lightop(monkeypatch):
    called = {}
    fake_lightop = ModuleType("lightop")
    fake_activation = ModuleType("lightop.activation")

    def fake_silu_and_mul(output, input):
        called["args"] = (output, input)
        output.fill_(7)

    fake_activation.silu_and_mul_opt = fake_silu_and_mul
    fake_lightop.activation = fake_activation
    monkeypatch.setitem(sys.modules, "lightop", fake_lightop)
    monkeypatch.setitem(sys.modules, "lightop.activation", fake_activation)

    input = torch.randn(3, 8)
    output = torch.empty(3, 4)
    _ACTIVATION_IMPL_MODULE.silu_and_mul_hygon_out(output, input)

    assert called["args"] == (output, input)
    torch.testing.assert_close(output, torch.full_like(output, 7))


def test_moe_silu_does_not_fallback_to_flaggems(monkeypatch):
    fake_lightop = ModuleType("lightop")
    fake_activation = ModuleType("lightop.activation")
    fake_lightop.activation = fake_activation
    monkeypatch.setitem(sys.modules, "lightop", fake_lightop)
    monkeypatch.setitem(sys.modules, "lightop.activation", fake_activation)

    called = {"flaggems": False}
    fake_flaggems = ModuleType("flag_gems")
    fake_modules = ModuleType("flag_gems.modules")
    fake_gems_activation = ModuleType("flag_gems.modules.activation")

    def fake_gems_silu_and_mul(gate, up):
        called["flaggems"] = True
        return gate + up

    fake_gems_activation.gems_silu_and_mul = fake_gems_silu_and_mul
    monkeypatch.setitem(sys.modules, "flag_gems", fake_flaggems)
    monkeypatch.setitem(sys.modules, "flag_gems.modules", fake_modules)
    monkeypatch.setitem(
        sys.modules, "flag_gems.modules.activation", fake_gems_activation
    )

    input = torch.randn(3, 8)
    output = torch.empty(3, 4)
    with pytest.raises(AttributeError, match="silu_and_mul_opt"):
        _ACTIVATION_IMPL_MODULE.silu_and_mul_hygon_out(output, input)

    assert called["flaggems"] is False


def test_moe_sum_uses_hygon_kernel(monkeypatch):
    called = {}
    module_name = (
        "vllm_fl.dispatch.backends.vendor.hygon.impl.moe.bf16_moe_fusions"
    )
    fake_fusions = ModuleType(module_name)

    def fake_reduce(input, output):
        called["args"] = (input, output)
        output.fill_(3)
        return True

    fake_fusions.try_hygon_fixed_topk8_reduce = fake_reduce
    monkeypatch.setitem(sys.modules, module_name, fake_fusions)
    monkeypatch.setattr(
        _MOE_SUM_IMPL_MODULE,
        "__package__",
        "vllm_fl.dispatch.backends.vendor.hygon.impl.native_ops",
    )

    input = torch.randn(2, 8, 4)
    output = torch.empty(2, 4)
    _MOE_SUM_IMPL_MODULE.moe_sum_hygon_out(input, output)

    assert called["args"] == (input, output)
    torch.testing.assert_close(output, torch.full_like(output, 3))


def test_moe_sum_rejects_inputs_not_supported_by_hygon_kernel(monkeypatch):
    module_name = (
        "vllm_fl.dispatch.backends.vendor.hygon.impl.moe.bf16_moe_fusions"
    )
    fake_fusions = ModuleType(module_name)
    fake_fusions.try_hygon_fixed_topk8_reduce = lambda input, output: False
    monkeypatch.setitem(sys.modules, module_name, fake_fusions)
    monkeypatch.setattr(
        _MOE_SUM_IMPL_MODULE,
        "__package__",
        "vllm_fl.dispatch.backends.vendor.hygon.impl.native_ops",
    )

    input = torch.randn(2, 4, 6)
    output = torch.empty(2, 6)

    with pytest.raises(NotImplementedError, match="fixed-topk=8"):
        _MOE_SUM_IMPL_MODULE.moe_sum_hygon_out(input, output)


def test_rms_norm_uses_hygon_lightop(monkeypatch):
    called = {}
    fake_lightop = ModuleType("lightop")
    fake_norm = ModuleType("lightop.norm")

    def fake_rms(x, weight, epsilon, training):
        called["args"] = (x, weight, epsilon, training)
        return x + 1

    fake_norm.rmsnorm_forward_autograd = fake_rms
    fake_norm.fused_add_rms_norm = lambda *args: None
    fake_lightop.norm = fake_norm
    monkeypatch.setitem(sys.modules, "lightop", fake_lightop)
    monkeypatch.setitem(sys.modules, "lightop.norm", fake_norm)

    obj = type(
        "RMSNorm",
        (),
        {
            "weight": torch.ones(4),
            "variance_epsilon": 1e-6,
            "training": False,
        },
    )()
    x = torch.randn(2, 4)

    result = _NORMALIZATION_IMPL_MODULE.rms_norm_hygon(obj, x)

    assert called["args"] == (x, obj.weight, 1e-6, False)
    torch.testing.assert_close(result, x + 1)


def test_hygon_mhc_patch_routes_hip_methods_to_fl_impl(monkeypatch):
    called = {}

    class FakeMHCPreOp:
        def forward_hip(self, *args, **kwargs):
            raise AssertionError("upstream MHC pre must not be called")

    class FakeMHCPostOp:
        def forward_hip(self, *args, **kwargs):
            raise AssertionError("upstream MHC post must not be called")

    class FakeMHCFusedPostPreOp:
        def forward_hip(self, *args, **kwargs):
            raise AssertionError("upstream fused MHC must not be called")

    layers_name = "vllm.model_executor.layers.mhc"
    impl_name = "vllm_fl.dispatch.backends.vendor.hygon.impl.other.mhc"
    fake_layers = ModuleType(layers_name)
    fake_layers.MHCPreOp = FakeMHCPreOp
    fake_layers.MHCPostOp = FakeMHCPostOp
    fake_layers.MHCFusedPostPreOp = FakeMHCFusedPostPreOp
    fake_impl = ModuleType(impl_name)

    def fake_pre(*args):
        called["pre"] = args
        return "pre-output"

    def fake_post(*args):
        called["post"] = args
        return "post-output"

    def fake_fused(*args):
        called["fused"] = args
        return "fused-output"

    fake_impl.mhc_pre_hygon = fake_pre
    fake_impl.mhc_post_hygon = fake_post
    fake_impl.mhc_fused_post_pre_hygon = fake_fused

    real_import_module = importlib.import_module

    def fake_import_module(name):
        if name == layers_name:
            return fake_layers
        if name == impl_name:
            return fake_impl
        return real_import_module(name)

    monkeypatch.setattr(_PATCH_MODULE.importlib, "import_module", fake_import_module)

    _patch_mhc_ops()
    _patch_mhc_ops()

    pre_result = FakeMHCPreOp().forward_hip(
        "residual",
        "fn",
        "scale",
        "base",
        1e-6,
        1e-6,
        1e-6,
        2.0,
        3,
    )
    post_result = FakeMHCPostOp().forward_hip(
        "x",
        "residual",
        "post-mix",
        "comb-mix",
    )
    fused_result = FakeMHCFusedPostPreOp().forward_hip(
        "x",
        "residual",
        "post-mix",
        "comb-mix",
        "fn",
        "scale",
        "base",
        1e-6,
        1e-6,
        1e-6,
        2.0,
        3,
    )

    assert pre_result == "pre-output"
    assert post_result == "post-output"
    assert fused_result == "fused-output"
    assert called["pre"][-3:] == (1, None, 0.0)
    assert called["post"] == ("x", "residual", "post-mix", "comb-mix")
    assert called["fused"][-4:] == (1, 1, None, 0.0)
    assert getattr(FakeMHCPreOp.forward_hip, "_vllm_fl_hygon", False)
    assert getattr(FakeMHCPostOp.forward_hip, "_vllm_fl_hygon", False)
    assert getattr(FakeMHCFusedPostPreOp.forward_hip, "_vllm_fl_hygon", False)


def test_hygon_mhc_impl_prefers_aiter_vendor_ops(monkeypatch):
    called = {}
    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []
    fake_aiter_ops = ModuleType("aiter.ops")
    fake_aiter_ops.__path__ = []
    fake_tilelang = ModuleType("aiter.ops.tilelang")

    def fake_pre(**kwargs):
        called["pre"] = kwargs
        residual = kwargs["residual"]
        num_tokens, hc_mult, hidden_size = residual.shape
        return (
            torch.ones(num_tokens, hc_mult),
            torch.ones(num_tokens, hc_mult, hc_mult),
            torch.ones(num_tokens, hidden_size, dtype=residual.dtype),
        )

    def fake_post(x, residual, post_mix, comb_mix):
        called["post"] = (x, residual, post_mix, comb_mix)
        return residual + 1

    fake_tilelang.mhc_pre_big_fuse = fake_pre
    fake_tilelang.mhc_post_fwd = fake_post
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.ops", fake_aiter_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.tilelang", fake_tilelang)

    residual = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    fn = torch.zeros(24, 32, dtype=torch.float32)
    scale = torch.ones(3, dtype=torch.float32)
    base = torch.zeros(24, dtype=torch.float32)
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    post_mix = torch.ones(2, 4, 1, dtype=torch.float32)
    comb_mix = torch.eye(4).expand(2, -1, -1).contiguous()

    _MHC_IMPL_MODULE._get_aiter_mhc_ops.cache_clear()
    try:
        assert _MHC_IMPL_MODULE.get_mhc_backend_name() == "aiter"
        pre_result = _MHC_IMPL_MODULE.mhc_pre_hygon(
            residual,
            fn,
            scale,
            base,
            1e-6,
            1e-6,
            1e-6,
            2.0,
            3,
        )
        post_result = _MHC_IMPL_MODULE.mhc_post_hygon(
            x,
            residual,
            post_mix,
            comb_mix,
        )
    finally:
        _MHC_IMPL_MODULE._get_aiter_mhc_ops.cache_clear()

    assert pre_result[0].shape == (2, 4, 1)
    assert pre_result[1].shape == (2, 4, 4)
    assert pre_result[2].shape == (2, 8)
    assert called["pre"]["residual"].shape == (2, 4, 8)
    assert called["pre"]["mhc_scale"] is scale
    assert called["post"][0].shape == (2, 8)
    torch.testing.assert_close(post_result, residual + 1)


def test_hygon_disables_nvidia_cutedsl_for_deepseek_v4_common_ops(monkeypatch):
    indexer_module_name = "vllm.models.deepseek_v4.common.ops.fused_indexer_q"
    cache_module_name = "vllm.models.deepseek_v4.common.ops.cache_utils"
    indexer_module = ModuleType(indexer_module_name)
    cache_module = ModuleType(cache_module_name)
    indexer_module.has_cutedsl = lambda: True
    cache_module.has_cutedsl = lambda: True
    monkeypatch.setitem(sys.modules, indexer_module_name, indexer_module)
    monkeypatch.setitem(sys.modules, cache_module_name, cache_module)

    _patch_deepseek_v4_cutedsl_selection()

    assert indexer_module.has_cutedsl() is False
    assert cache_module.has_cutedsl() is False


def test_hygon_bf16_indexer_patches_cache_q_and_compressor(monkeypatch):
    called = {}

    class FakeDeepseekV4Indexer:
        def __init__(self):
            self.use_fp4_kv = False
            self.head_dim = 128
            self.k_cache = type(
                "FakeIndexerCache",
                (),
                {"head_dim": 132, "dtype": torch.uint8},
            )()
            self.compressor = type("FakeCompressor", (), {})()
            self.indexer_op = type("FakeIndexerOp", (), {})()

    attention_name = "vllm.models.deepseek_v4.attention"
    compressor_name = "vllm.models.deepseek_v4.compressor"
    common_q_name = "vllm.models.deepseek_v4.common.ops.fused_indexer_q"
    impl_name = "vllm_fl.dispatch.backends.vendor.hygon.impl.attention.bf16_indexer"

    attention_module = ModuleType(attention_name)
    attention_module.DeepseekV4Indexer = FakeDeepseekV4Indexer

    def original_q(*args, **kwargs):
        called["original_q"] = (args, kwargs)
        return "fp8-q"

    attention_module.fused_indexer_q_rope_quant = original_q
    common_q_module = ModuleType(common_q_name)
    common_q_module.fused_indexer_q_rope_quant = original_q
    compressor_module = ModuleType(compressor_name)

    def original_store(**kwargs):
        called["original_store"] = kwargs

    compressor_module.compress_norm_rope_store_triton = original_store
    impl_module = ModuleType(impl_name)
    impl_module.use_bf16_indexer_cache = lambda: True

    def bf16_q(*args):
        called["bf16_q"] = args
        return "bf16-q"

    def bf16_store(**kwargs):
        called["bf16_store"] = kwargs

    impl_module.fused_indexer_q_rope_bf16 = bf16_q
    impl_module.compress_norm_rope_store_bf16 = bf16_store

    modules = {
        attention_name: attention_module,
        compressor_name: compressor_module,
        common_q_name: common_q_module,
        impl_name: impl_module,
    }
    real_import_module = importlib.import_module

    def fake_import_module(name):
        if name in modules:
            return modules[name]
        return real_import_module(name)

    monkeypatch.setattr(_PATCH_MODULE.importlib, "import_module", fake_import_module)

    _patch_deepseek_v4_bf16_indexer_cache()

    indexer = FakeDeepseekV4Indexer()
    assert indexer.use_bf16_cache is True
    assert indexer.k_cache.head_dim == 128
    assert indexer.k_cache.dtype == torch.bfloat16
    assert indexer.compressor.use_bf16_cache is True
    assert indexer.indexer_op.use_bf16_cache is True

    result = attention_module.fused_indexer_q_rope_quant(
        "positions", "q", "cos-sin", "weights", 0.5, 0.125
    )
    assert result == "bf16-q"
    assert called["bf16_q"] == (
        "positions",
        "q",
        "cos-sin",
        "weights",
        0.5,
        0.125,
    )
    assert common_q_module.fused_indexer_q_rope_quant is (
        attention_module.fused_indexer_q_rope_quant
    )

    bf16_cache = torch.empty(1, dtype=torch.bfloat16)
    compressor_module.compress_norm_rope_store_triton(kv_cache=bf16_cache)
    assert called["bf16_store"] == {"kv_cache": bf16_cache}
    uint8_cache = torch.empty(1, dtype=torch.uint8)
    compressor_module.compress_norm_rope_store_triton(kv_cache=uint8_cache)
    assert called["original_store"] == {"kv_cache": uint8_cache}


def test_hygon_deepseek_v4_qnorm_insert_uses_lightop_8_arg_abi(monkeypatch):
    called = {}

    class FakeAttention:
        def _fused_qnorm_rope_kv_insert(self, *args):
            called["original"] = args
            return "original-output"

    rocm_module_name = "vllm.models.deepseek_v4.amd.rocm"
    fake_rocm_module = ModuleType(rocm_module_name)
    fake_rocm_module.DeepseekV4ROCMAiterMLAAttention = FakeAttention

    real_import_module = importlib.import_module

    def fake_import_module(name):
        if name == rocm_module_name:
            return fake_rocm_module
        return real_import_module(name)

    monkeypatch.setattr(_PATCH_MODULE.importlib, "import_module", fake_import_module)

    fake_lightop = ModuleType("lightop")
    fake_lightop_op = ModuleType("lightop.op")

    def fake_kernel(*args):
        called["lightop"] = args

    fake_lightop_op.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert = fake_kernel
    fake_lightop.op = fake_lightop_op
    monkeypatch.setitem(sys.modules, "lightop", fake_lightop)
    monkeypatch.setitem(sys.modules, "lightop.op", fake_lightop_op)

    _patch_deepseek_v4_qnorm_rope_kv_insert()

    q = torch.randn(2, 4, 8)
    kv = torch.randn(2, 8)
    cache = torch.empty(3, 2, 12, dtype=torch.uint8)
    positions = torch.tensor([1, 2], dtype=torch.int32)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
    metadata = type(
        "FakeMetadata",
        (),
        {"slot_mapping": slot_mapping, "block_size": 2},
    )()

    attention = FakeAttention()
    attention.swa_cache_layer = type(
        "FakeCacheLayer", (), {"prefix": "layer.swa", "kv_cache": cache}
    )()
    attention.rotary_emb = type(
        "FakeRotary", (), {"cos_sin_cache": torch.randn(16, 4)}
    )()
    attention.eps = 1e-6
    attention.n_local_heads = 4
    attention.padded_heads = 4

    result = attention._fused_qnorm_rope_kv_insert(
        q,
        kv,
        positions,
        {"layer.swa": metadata},
    )

    assert result is q
    assert len(called["lightop"]) == 8
    assert called["lightop"][0] is q
    assert called["lightop"][2].shape == (3, 24)
    assert called["lightop"][4].dtype == torch.int64
    assert "original" not in called


def test_hygon_sparse_indexer_uses_vllm_native_rocm_with_fl_ops(monkeypatch):
    called = {}

    class FakeSparseAttnIndexer:
        def forward_hip(self, *args):
            called["original"] = args

    fake_sparse_module = ModuleType("vllm.model_executor.layers.sparse_attn_indexer")
    fake_sparse_module.SparseAttnIndexer = FakeSparseAttnIndexer
    fake_aiter_ops = ModuleType("fake_aiter_ops")
    fake_aiter_ops.is_enabled = lambda: False
    fake_sparse_module.rocm_aiter_ops = fake_aiter_ops

    def fake_native(*args, **kwargs):
        called["native"] = (args, kwargs)
        return "topk-output"

    fake_sparse_module.sparse_attn_indexer = fake_native
    fake_sparse_module.ops = ModuleType("fake_sparse_ops")

    impl_module_name = (
        "vllm_fl.dispatch.backends.vendor.hygon.impl.attention.sparse_attn_indexer"
    )
    bf16_impl_module_name = (
        "vllm_fl.dispatch.backends.vendor.hygon.impl.attention.bf16_indexer"
    )
    fake_impl_module = ModuleType(impl_module_name)
    fake_bf16_impl_module = ModuleType(bf16_impl_module_name)

    def fake_install(module):
        called["installed"] = module

    fake_impl_module.install_sparse_indexer_hygon_ops = fake_install

    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_utils = ModuleType("vllm.utils")
    fake_utils.__path__ = []
    fake_torch_utils = ModuleType("vllm.utils.torch_utils")
    fake_torch_utils._encode_layer_name = lambda name: name
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", fake_torch_utils)

    real_import_module = importlib.import_module

    def fake_import_module(name):
        if name == "vllm.model_executor.layers.sparse_attn_indexer":
            return fake_sparse_module
        if name == impl_module_name:
            return fake_impl_module
        if name == bf16_impl_module_name:
            return fake_bf16_impl_module
        return real_import_module(name)

    monkeypatch.setattr(_PATCH_MODULE.importlib, "import_module", fake_import_module)

    _patch_sparse_attn_indexer()

    indexer = FakeSparseAttnIndexer()
    indexer.skip_k_cache_insert = True
    indexer.use_fp4_cache = False
    cache = torch.empty(1, dtype=torch.uint8)
    indexer.k_cache = type(
        "FakeCache", (), {"prefix": "layer.indexer", "kv_cache": cache}
    )()
    indexer.quant_block_size = 128
    indexer.scale_fmt = "ue8m0"
    indexer.topk_tokens = 2048
    indexer.head_dim = 128
    indexer.max_model_len = 4096
    indexer.max_total_seq_len = 8192
    indexer.topk_indices_buffer = "topk-buffer"

    result = indexer.forward_hip("hidden", "q", "k", "weights")

    assert result == "topk-output"
    assert called["installed"] is fake_sparse_module
    assert called["native"][0][0] == "hidden"
    assert called["native"][0][2] is cache
    assert called["native"][0][-2:] == (True, False)
    assert called["native"][1] == {}
    assert "original" not in called


def test_hygon_sparse_indexer_routes_bf16_cache_to_fl_branch(monkeypatch):
    called = {}

    class FakeSparseAttnIndexer:
        def forward_hip(self, *args):
            called["original"] = args

    fake_sparse_module = ModuleType("vllm.model_executor.layers.sparse_attn_indexer")
    fake_sparse_module.SparseAttnIndexer = FakeSparseAttnIndexer
    fake_sparse_module.rocm_aiter_ops = type(
        "FakeAiterOps", (), {"is_enabled": staticmethod(lambda: True)}
    )()

    sparse_impl_name = (
        "vllm_fl.dispatch.backends.vendor.hygon.impl.attention.sparse_attn_indexer"
    )
    bf16_impl_name = (
        "vllm_fl.dispatch.backends.vendor.hygon.impl.attention.bf16_indexer"
    )
    fake_sparse_impl = ModuleType(sparse_impl_name)
    fake_sparse_impl.install_sparse_indexer_hygon_ops = lambda module: None
    fake_bf16_impl = ModuleType(bf16_impl_name)

    def fake_bf16(*args):
        called["bf16"] = args
        return "bf16-topk-output"

    fake_bf16_impl.sparse_attn_indexer_bf16_hygon = fake_bf16

    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_utils = ModuleType("vllm.utils")
    fake_utils.__path__ = []
    fake_torch_utils = ModuleType("vllm.utils.torch_utils")
    fake_torch_utils._encode_layer_name = lambda name: f"encoded:{name}"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", fake_torch_utils)

    real_import_module = importlib.import_module

    def fake_import_module(name):
        if name == "vllm.model_executor.layers.sparse_attn_indexer":
            return fake_sparse_module
        if name == sparse_impl_name:
            return fake_sparse_impl
        if name == bf16_impl_name:
            return fake_bf16_impl
        return real_import_module(name)

    monkeypatch.setattr(_PATCH_MODULE.importlib, "import_module", fake_import_module)

    _patch_sparse_attn_indexer()

    indexer = FakeSparseAttnIndexer()
    indexer.use_bf16_cache = True
    cache = torch.empty(1, 64, 128, dtype=torch.bfloat16)
    indexer.k_cache = type(
        "FakeCache", (), {"prefix": "layer.indexer", "kv_cache": cache}
    )()
    indexer.topk_tokens = 2048
    indexer.head_dim = 128
    indexer.max_model_len = 4096
    indexer.max_total_seq_len = 8192
    indexer.topk_indices_buffer = "topk-buffer"
    q_bf16 = torch.empty(2, 64, 128, dtype=torch.bfloat16)

    result = indexer.forward_hip("hidden", q_bf16, None, "weights")

    assert result == "bf16-topk-output"
    assert called["bf16"] == (
        "hidden",
        "encoded:layer.indexer",
        cache,
        q_bf16,
        "weights",
        2048,
        128,
        4096,
        8192,
        "topk-buffer",
    )
    assert "original" not in called
