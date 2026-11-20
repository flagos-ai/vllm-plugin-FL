# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for ``vllm_fl.quantization.modelslim_w8a8``.

``modelslim_w8a8`` imports the external ``vllm`` layer/parameter graph at
module level. On CPU-only machines that graph is absent, so the helper below
installs a minimal stub tree in ``sys.modules`` before importing the module
under test; the same import succeeds unchanged against real vLLM. The
lazily-imported ``torch_npu`` boundary inside ``_quantize``/``_quant_matmul``
and the ``get_tensor_model_parallel_rank`` helper are the only other things
stubbed — the scheme/config logic under test always runs unmodified.

Existing core tests (tests/unit_tests/quantization/test_modelslim_w8a8.py)
cover the prefix-group mapping and the static scheme's create/process/apply
basics with numerical torch_npu stubs. This module pins NEW ground: the
config-level metadata API, the dynamic-linear scheme's offset parameter,
row-parallel ``tp_rank`` fallback through the module-level helper, pre-
quantized input rejection, single-vs-multi partition output shapes, the MoE
method's extra offset parameters, and the ``get_quant_method`` matrix for
linear and routed-expert layers.
"""

import sys
import types
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.supplementary


def _make_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


class _ModelSlimVllmStubs:
    """Install only the vllm graph ``modelslim_w8a8`` imports at import time.

    On machines with real vLLM this class is never used (see
    ``_import_modelslim``); the stubs exist purely so the module imports on
    CPU-only Windows hosts. Attribute values mirror the real vLLM symbols'
    constructor contracts (``output_dim``/``input_dim`` keywords and a
    ``weight_loader`` callable).
    """

    def __init__(self) -> None:
        self._installed: list[str] = []
        self._previous: dict[str, Any] = {}

        ct_root = _make_module("compressed_tensors")
        ct_quant = _make_module("compressed_tensors.quantization")

        class _Args:
            def __init__(self, **kwargs) -> None:
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class _Strategy(_Args):
            CHANNEL = "channel"
            TOKEN = "token"
            GROUP = "group"
            TENSOR = "tensor"

        ct_quant.QuantizationArgs = _Args
        ct_quant.QuantizationType = type(
            "QuantizationType", (), {"INT": "int", "FLOAT": "float"}
        )
        ct_quant.QuantizationStrategy = _Strategy

        vllm = _make_module("vllm")
        distributed = _make_module("vllm.distributed")
        distributed.get_tensor_model_parallel_rank = lambda: 0
        fused_moe = _make_module("vllm.model_executor.layers.fused_moe")
        fused_moe.FusedMoeWeightScaleSupported = type(
            "FusedMoeWeightScaleSupported",
            (),
            {"CHANNEL": type("C", (), {"value": "channel"})},
        )
        fused_moe.RoutedExperts = type("RoutedExperts", (), {})
        fused_moe.UnquantizedFusedMoEMethod = type(
            "UnquantizedFusedMoEMethod",
            (),
            {
                "__init__": lambda self, config: None,
            },
        )
        linear = _make_module("vllm.model_executor.layers.linear")
        linear.LinearBase = type("LinearBase", (), {})
        linear.RowParallelLinear = type("RowParallelLinear", (linear.LinearBase,), {})
        linear.UnquantizedLinearMethod = type(
            "UnquantizedLinearMethod",
            (),
            {
                "__init__": lambda self: None,
            },
        )
        quant = _make_module("vllm.model_executor.layers.quantization")
        quant.register_quantization_config = lambda name: lambda cls: cls
        base_config = _make_module(
            "vllm.model_executor.layers.quantization.base_config"
        )
        base_config.QuantizationConfig = type(
            "QuantizationConfig", (), {"__init__": lambda self: None}
        )
        base_config.QuantizeMethodBase = type("QuantizeMethodBase", (), {})
        ct = _make_module(
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors"
        )
        ct.CompressedTensorsLinearMethod = type(
            "CompressedTensorsLinearMethod",
            (),
            {
                "__init__": lambda self, config: None,
            },
        )
        ct_moe = _make_module(
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
        )
        ct_moe.CompressedTensorsW8A8Int8MoEMethod = type(
            "CompressedTensorsW8A8Int8MoEMethod",
            (),
            {
                "__init__": lambda self, *a, **k: None,
                "create_weights": classmethod(lambda cls, *a, **k: None),
            },
        )
        schemes = _make_module(
            "vllm.model_executor.layers.quantization.compressed_tensors.schemes"
        )
        schemes.CompressedTensorsScheme = type("CompressedTensorsScheme", (), {})
        schemes.CompressedTensorsW8A8Int8 = type(
            "CompressedTensorsW8A8Int8",
            (),
            {
                "__init__": lambda self, **kw: None,
                "create_weights": classmethod(lambda cls, *a, **kw: None),
            },
        )
        parameter = _make_module("vllm.model_executor.parameter")

        class _Parameter(torch.nn.Parameter):
            def __new__(cls, data, **kwargs):
                return torch.nn.Parameter._make_subclass(cls, data, False)

            def __init__(self, data, **kwargs) -> None:
                self.weight_loader = kwargs.get("weight_loader")
                self.input_dim = kwargs.get("input_dim")
                self.output_dim = kwargs.get("output_dim")
                self.packed_factor = kwargs.get("packed_factor")
                self.packed_dim = kwargs.get("packed_dim")

        parameter.BasevLLMParameter = _Parameter
        parameter.ModelWeightParameter = _Parameter
        parameter.ChannelQuantScaleParameter = _Parameter
        parameter.PerTensorScaleParameter = _Parameter
        parameter.PackedvLLMParameter = _Parameter
        parameter.GroupQuantScaleParameter = _Parameter
        parameter.get_tensor_model_parallel_rank = lambda: 0
        parameter.get_tensor_model_parallel_world_size = lambda: 1
        utils = _make_module("vllm.model_executor.utils")
        utils.set_weight_attrs = lambda param, attrs: setattr(
            param, "loaded_attrs", attrs
        )

        entries = {
            "compressed_tensors": ct_root,
            "compressed_tensors.quantization": ct_quant,
            "vllm": vllm,
            "vllm.distributed": distributed,
            "vllm.model_executor": _make_module("vllm.model_executor"),
            "vllm.model_executor.layers": _make_module("vllm.model_executor.layers"),
            "vllm.model_executor.layers.fused_moe": fused_moe,
            "vllm.model_executor.layers.linear": linear,
            "vllm.model_executor.layers.quantization": quant,
            "vllm.model_executor.layers.quantization.base_config": base_config,
            "vllm.model_executor.layers.quantization.compressed_tensors": _make_module(
                "vllm.model_executor.layers.quantization.compressed_tensors"
            ),
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors": ct,
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors_moe": _make_module(
                "vllm.model_executor.layers.quantization.compressed_tensors."
                "compressed_tensors_moe"
            ),
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8": ct_moe,
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "schemes": schemes,
            "vllm.model_executor.parameter": parameter,
            "vllm.model_executor.utils": utils,
        }
        # Wire parent attributes for attribute-style access.
        entries["vllm.model_executor"].layers = entries["vllm.model_executor.layers"]
        entries["vllm.model_executor.layers"].fused_moe = fused_moe
        entries["vllm.model_executor.layers"].linear = linear
        entries["vllm.model_executor.layers"].quantization = quant
        entries[
            "vllm.model_executor.layers.quantization.compressed_tensors"
        ].compressed_tensors = ct
        quant.base_config = base_config

        for name, module in entries.items():
            self._previous[name] = sys.modules.get(name)
            sys.modules[name] = module
            self._installed.append(name)

    def restore(self) -> None:
        """Put back every sys.modules entry replaced by the stub tree."""
        for name in self._installed:
            previous = self._previous[name]
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        self._installed.clear()
        self._previous.clear()


try:
    # Import the module under test at file scope, before the directory
    # conftest takes its first per-test sys.modules snapshot. Importing it
    # inside a test loads a vllm chain that registers torch custom ops
    # (vllm::all_reduce, vllm::marlin_gemm, ...); the conftest then removes
    # those freshly cached modules after the test, and the next import
    # would re-register the ops, which torch rejects. A file-scope import
    # puts the whole chain in every snapshot so the restore always puts it
    # back. On hosts without vllm (CPU-only Windows) the stub tree above
    # provides the external boundary instead.
    import vllm_fl.quantization.modelslim_w8a8 as module  # noqa: F811

    _REAL_VLLM = True
except ImportError:  # pragma: no cover - exercised only without vllm
    _stubs = _ModelSlimVllmStubs()
    import vllm_fl.quantization.modelslim_w8a8 as module  # noqa: F811

    # The module under test holds every vllm name it needs as attributes of
    # its own namespace, so it stays fully functional after the stub tree is
    # removed. Restoring is what makes the file order-independent: leaving
    # the stubs installed keeps a fake "vllm" in sys.modules, so any later
    # file's ``import vllm`` probe succeeds and files gated on HAS_VLLM
    # import against stale partial stubs instead of skipping.
    _stubs.restore()

    _REAL_VLLM = False


def _patch_real_vllm_runtime(monkeypatch):
    """Neutralize vLLM's global-runtime requirements on a bare process.

    Against real vLLM, the parameter classes consult the tensor-parallel
    process group at construction and the INT8 MoE scheme constructor probes
    hardware backends — none of which accept the CPU test device. Both are
    external vLLM boundaries, not logic under test: pin TP rank 0 / world 1
    (mirroring the core suite's ``_single_rank_tensor_parallel`` fixture) and
    return a fixed backend tuple so scheme construction works off-hardware.
    """
    import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 as ct_moe
    import vllm.model_executor.parameter as vllm_parameter

    monkeypatch.setattr(vllm_parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vllm_parameter, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        ct_moe, "select_int8_moe_backend", lambda **kwargs: (object(), object())
    )


@pytest.fixture
def modelslim(fl_sys_modules, monkeypatch):
    """Yield the file-scope-imported module under test.

    The module was imported at file scope (against real vllm when present,
    against the stub tree otherwise); the directory conftest snapshots those
    collection-time installs and re-instates them after every test, so the
    module object and its layer-class swaps persist per test. On real vLLM
    the fixture also pins the TP rank helpers and the INT8 MoE backend
    selector. Teardown restores any layer-class swap made by
    ``_swap_layers`` so the next test sees the original (real or stubbed)
    vllm classes.
    """
    if _REAL_VLLM:
        _patch_real_vllm_runtime(monkeypatch)
    yield module
    saved = vars(module).pop("_vllm_fl_saved_layers", None)
    if saved is not None:
        module.LinearBase = saved["LinearBase"]
        module.RowParallelLinear = saved["RowParallelLinear"]
        module.RoutedExperts = saved["RoutedExperts"]


def _install_torch_npu(monkeypatch, quantize=None, quant_matmul=None):
    """Install a fake torch_npu module for the two lazy NPU imports."""
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.npu_quantize = quantize or (lambda *a, **k: a[0].to(torch.int8))
    torch_npu.npu_quant_matmul = quant_matmul or (
        lambda *a, **k: torch.zeros((a[0].shape[0], a[1].shape[1]))
    )
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    return torch_npu


class TestConfigMetadataApi:
    """The QuantizationConfig metadata surface vLLM calls at load time."""

    def test_class_metadata_constants(self, modelslim):
        """get_name / filenames / dtypes / capability match the checkpoint."""
        module = modelslim
        config = module.ModelSlimW8A8Config({})
        assert config.get_name() == "fl_modelslim_w8a8"
        assert config.get_config_filenames() == ["quant_model_description.json"]
        assert config.get_min_capability() == 0
        assert module.ModelSlimW8A8Config.get_min_capability() == 0
        assert set(config.get_supported_act_dtypes()) == {
            torch.bfloat16,
            torch.float16,
        }
        assert module.ModelSlimW8A8Config.from_config({"a": 1}).description == {"a": 1}
        assert config.weight_quant.num_bits == 8
        assert config.input_quant.dynamic is True

    def test_default_unknown_prefix_is_float(self, modelslim):
        """A prefix absent from the description is not quantized."""
        module = modelslim
        config = module.ModelSlimW8A8Config({})
        assert config._quant_kind("model.layers.5.mlp.down_proj") is None
        assert config._is_dynamic_w8a8("model.layers.5.mlp.down_proj") is False
        assert config._is_static_w8a8("model.layers.5.mlp.down_proj") is False

    def test_unsupported_kind_rejected(self, modelslim):
        """An unknown quantization kind fails with NotImplementedError."""
        module = modelslim
        config = module.ModelSlimW8A8Config({"layers.0.mlp.down_proj.weight": "INT4"})
        with pytest.raises(NotImplementedError, match="not supported"):
            config._quant_kind("model.layers.0.mlp.down_proj")

    def test_root_prefixes_bidirectional_model_prefix(self, modelslim):
        """_root_prefixes maps both rooted and unrooted checkpoint names."""
        module = modelslim
        config = module.ModelSlimW8A8Config({})
        assert config._root_prefixes("model.layers.0.mlp") == [
            "model.layers.0.mlp",
            "layers.0.mlp",
        ]
        assert config._root_prefixes("layers.0.mlp") == [
            "layers.0.mlp",
            "model.layers.0.mlp",
        ]

    def test_down_proj_groups_w1_alias_deduplicated(self, modelslim):
        """down_proj maps to itself plus the w2 alias; duplicates collapse."""
        module = modelslim
        config = module.ModelSlimW8A8Config({"layers.0.mlp.down_proj.weight": "W8A8"})
        groups = config._checkpoint_prefix_groups("model.layers.0.mlp.down_proj")
        # model.-rooted and bare-rooted groups collapse to the same tuples.
        assert ("layers.0.mlp.down_proj",) in groups
        assert ("layers.0.mlp.w2",) in groups
        assert config._is_static_w8a8("model.layers.0.mlp.down_proj")

    def test_shared_gate_up_and_experts_groups(self, modelslim):
        """gate_up_proj / experts each expand into their checkpoint tuples."""
        module = modelslim
        config = module.ModelSlimW8A8Config(
            {
                "layers.0.mlp.gate_proj.weight": "W8A8_DYNAMIC",
                "layers.0.mlp.up_proj.weight": "W8A8_DYNAMIC",
                "layers.0.ffn.experts.0.w1.weight": "W8A8_DYNAMIC",
                "layers.0.ffn.experts.0.w2.weight": "W8A8_DYNAMIC",
                "layers.0.ffn.experts.0.w3.weight": "W8A8_DYNAMIC",
            }
        )
        assert config._quant_kind("model.layers.0.mlp.gate_up_proj") == "W8A8_DYNAMIC"
        assert config._quant_kind("model.layers.0.ffn.experts") == "W8A8_DYNAMIC"


class TestGetQuantMethod:
    """get_quant_method dispatch across linear and routed-expert layers."""

    def test_linear_float_layer_gets_unquantized_method(self, modelslim):
        """None/FLOAT linear prefixes return the unquantized method."""
        module = modelslim
        config = module.ModelSlimW8A8Config({})
        result = config.get_quant_method(_swap_layers(module, "linear"), "model.x.y")
        assert type(result).__name__ == "UnquantizedLinearMethod"

    def test_linear_static_layer_uses_static_scheme(self, modelslim):
        """W8A8 linear prefixes install the static asymmetric scheme."""
        module = modelslim
        config = module.ModelSlimW8A8Config({"x.weight": "W8A8"})
        layer = _swap_layers(module, "linear")
        config.get_quant_method(layer, "model.x")
        assert type(layer.scheme).__name__ == "_ModelSlimW8A8StaticLinearScheme"

    def test_linear_dynamic_layer_uses_offset_scheme(self, modelslim):
        """W8A8_DYNAMIC linear prefixes install the offset-retaining scheme."""
        module = modelslim
        config = module.ModelSlimW8A8Config({"x.weight": "W8A8_DYNAMIC"})
        layer = _swap_layers(module, "linear")
        result = config.get_quant_method(layer, "model.x")
        assert type(layer.scheme).__name__ == "_ModelSlimW8A8LinearScheme"
        assert type(result).__name__ == "CompressedTensorsLinearMethod"

    def test_routed_experts_dynamic_uses_moe_method(self, modelslim):
        """Dynamic routed experts get the ModelSlim MoE method."""
        module = modelslim
        config = module.ModelSlimW8A8Config(
            {
                "ffn.experts.0.w1.weight": "W8A8_DYNAMIC",
                "ffn.experts.0.w2.weight": "W8A8_DYNAMIC",
                "ffn.experts.0.w3.weight": "W8A8_DYNAMIC",
            }
        )
        layer = _swap_layers(module, "experts")
        result = config.get_quant_method(layer, "model.ffn.experts")
        assert type(result).__name__ == "_ModelSlimW8A8MoEMethod"

    def test_routed_experts_static_rejected(self, modelslim):
        """Static W8A8 routed experts raise NotImplementedError."""
        module = modelslim
        config = module.ModelSlimW8A8Config(
            {
                "ffn.experts.0.gate_proj.weight": "W8A8",
                "ffn.experts.0.up_proj.weight": "W8A8",
                "ffn.experts.0.down_proj.weight": "W8A8",
            }
        )
        layer = _swap_layers(module, "experts")
        with pytest.raises(NotImplementedError, match="static W8A8 routed"):
            config.get_quant_method(layer, "model.ffn.experts")

    def test_routed_experts_float_gets_unquantized_moe(self, modelslim, monkeypatch):
        """FLOAT routed experts return the unquantized fused-MoE method."""
        module = modelslim
        config = module.ModelSlimW8A8Config({})
        if _REAL_VLLM:
            # Real UnquantizedFusedMoEMethod is a vLLM CustomOp whose
            # construction requires a live global vLLM config and a
            # hardware-backed unquantized MoE backend — both unavailable in
            # this process. The dispatch branch under test is only the class
            # selection, so record the class the constructor received.
            from types import SimpleNamespace

            import vllm.model_executor.custom_op as vllm_custom_op

            monkeypatch.setattr(
                vllm_custom_op,
                "get_cached_compilation_config",
                lambda: SimpleNamespace(
                    custom_ops=["all"],
                    enabled_custom_ops=set(),
                    disabled_custom_ops=set(),
                ),
            )
            import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as unquantized_moe

            monkeypatch.setattr(
                unquantized_moe,
                "select_unquantized_moe_backend",
                lambda **kwargs: (object(), object()),
            )
        else:
            monkeypatch.setattr(
                module,
                "UnquantizedFusedMoEMethod",
                type(
                    "UnquantizedFusedMoEMethod",
                    (),
                    {"__init__": lambda self, config: None},
                ),
            )
        layer = _swap_layers(module, "experts")
        result = config.get_quant_method(layer, "model.e")
        assert type(result).__name__ == "UnquantizedFusedMoEMethod"

    def test_unknown_layer_returns_none(self, modelslim):
        """A layer that is neither linear nor routed experts returns None."""
        module = modelslim
        config = module.ModelSlimW8A8Config({})
        assert config.get_quant_method(torch.nn.Module(), "model.whatever") is None


class _PlainInstance:
    """Marker instance class for fake linear/expert layers."""


class _FakeLinearBase:
    """isinstance target standing in for vLLM's LinearBase."""


class _FakeRowParallelLinear(_FakeLinearBase):
    """isinstance target standing in for vLLM's RowParallelLinear."""


class _FakeRoutedExperts:
    """isinstance target standing in for vLLM's RoutedExperts."""


def _swap_layers(module, kind):
    """Point the module's layer classes at fakes and return a fake layer.

    ``get_quant_method`` dispatches with ``isinstance`` against the classes
    the module imported, so substituting the module attributes with plain
    marker classes (restored by the ``modelslim`` fixture teardown) lets the
    dispatch logic run unmodified on hosts without vLLM.
    """
    saved = {
        "LinearBase": module.LinearBase,
        "RowParallelLinear": module.RowParallelLinear,
        "RoutedExperts": module.RoutedExperts,
    }
    module.LinearBase = _FakeLinearBase
    module.RowParallelLinear = _FakeRowParallelLinear
    module.RoutedExperts = _FakeRoutedExperts
    module._vllm_fl_saved_layers = saved
    if kind == "linear":
        instance = _FakeLinearBase()
        instance.scheme = None
    else:
        instance = _FakeRoutedExperts()
        instance.moe_config = object()
    return instance


class TestDynamicLinearSchemeOffset:
    """_ModelSlimW8A8LinearScheme adds the weight_offset parameter."""

    def test_create_weights_registers_offset_parameter(self, modelslim, monkeypatch):
        """create_weights appends a [output_size, 1] float32 offset."""
        module = modelslim
        calls = {}

        def fake_super_create(layer, **kwargs):
            calls.update(kwargs)
            layer.register_parameter(
                "weight", torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            )
            layer.register_parameter(
                "weight_scale", torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            )

        scheme_cls = module._ModelSlimW8A8LinearScheme
        base_create = scheme_cls.__mro__[1].create_weights
        monkeypatch.setattr(
            scheme_cls.__mro__[1],
            "create_weights",
            staticmethod(fake_super_create),
            raising=False,
        )
        try:
            scheme = object.__new__(scheme_cls)
            layer = torch.nn.Module()
            scheme.create_weights(
                layer,
                output_partition_sizes=[3, 5],
                input_size_per_partition=4,
                params_dtype=torch.bfloat16,
                weight_loader=lambda *a, **k: None,
            )
        finally:
            monkeypatch.setattr(scheme_cls.__mro__[1], "create_weights", base_create)

        assert calls["output_partition_sizes"] == [3, 5]
        assert layer.weight_offset.shape == (8, 1)
        assert layer.weight_offset.dtype == torch.float32


class TestStaticSchemeApplyEdges:
    """apply_weights branches not exercised by the core suite."""

    @staticmethod
    def _loaded(module, widths=(3, 5), k=4):
        """Build a loaded static scheme layer with deterministic data."""
        scheme = module._ModelSlimW8A8StaticLinearScheme()
        layer = torch.nn.Module()
        scheme.create_weights(
            layer,
            output_partition_sizes=list(widths),
            input_size_per_partition=k,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *a, **kw: None,
        )
        layer.weight.data.copy_(
            torch.arange(sum(widths) * k, dtype=torch.int8).reshape(sum(widths), k)
        )
        scale_values = [0.25 + 0.25 * i for i in range(len(widths))]
        offset_values = [1 - 3 * i for i in range(len(widths))]
        layer.input_scale.data.copy_(torch.tensor(scale_values))
        layer.input_offset.data.copy_(torch.tensor(offset_values, dtype=torch.float32))
        layer.deq_scale.data.copy_(torch.ones(sum(widths)))
        layer.quant_bias.data.copy_(torch.zeros(sum(widths), dtype=torch.int32))
        scheme.process_weights_after_loading(layer)
        return scheme, layer

    def test_prequantized_multi_partition_rejected(self, modelslim, monkeypatch):
        """int8 input with multiple logical matrices raises ValueError."""
        module = modelslim
        _install_torch_npu(monkeypatch)
        scheme, layer = self._loaded(module, widths=(3, 5))
        x = torch.ones((2, 4), dtype=torch.int8)
        with pytest.raises(ValueError, match="single"):
            scheme.apply_weights(layer, x, bias=None)

    def test_prequantized_single_partition_skips_quantize(self, modelslim, monkeypatch):
        """int8 input on a single partition bypasses _quantize entirely."""
        module = modelslim
        seen = []
        torch_npu = _install_torch_npu(
            monkeypatch,
            quantize=lambda *a, **k: seen.append("quantize") or a[0].to(torch.int8),
            quant_matmul=lambda x, w, s, *, bias, output_dtype: torch.full(
                (x.shape[0], w.shape[1]), 7, dtype=output_dtype
            ),
        )
        assert torch_npu is not None
        scheme, layer = self._loaded(module, widths=(8,))
        x = torch.ones((2, 4), dtype=torch.int8)
        output = scheme.apply_weights(layer, x, bias=None)
        assert seen == []
        assert output.shape == (2, 8)
        assert torch.equal(output, torch.full((2, 8), 7, dtype=torch.bfloat16))

    def test_row_parallel_tp_rank_none_uses_helper(self, modelslim, monkeypatch):
        """A RowParallelLinear without tp_rank asks the module helper."""
        module = modelslim
        helper_rank = []

        def fake_rank():
            helper_rank.append("called")
            return 3

        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", fake_rank)
        _install_torch_npu(
            monkeypatch,
            quant_matmul=lambda x, w, s, *, bias, output_dtype: torch.zeros(
                (x.shape[0], w.shape[1]), dtype=output_dtype
            ),
        )
        scheme, layer = self._loaded(module, widths=(8,))

        class FakeRow(torch.nn.Module):
            pass

        layer.__class__ = FakeRow
        layer.__class__ = type("Row", (module.RowParallelLinear, torch.nn.Module), {})
        assert isinstance(layer, module.RowParallelLinear)
        assert getattr(layer, "tp_rank", None) is None
        output = scheme.apply_weights(
            layer, torch.ones((1, 4), dtype=torch.bfloat16), None
        )
        assert helper_rank == ["called"]
        # tp_rank 3 (not 0): no quant bias slice is forwarded to the matmul.
        assert output.shape == (1, 8)

    def test_single_partition_output_not_concatenated(self, modelslim, monkeypatch):
        """One logical matrix returns the matmul result directly."""
        module = modelslim
        _install_torch_npu(
            monkeypatch,
            quant_matmul=lambda x, w, s, *, bias, output_dtype: torch.full(
                (x.shape[0], w.shape[1]), 3, dtype=output_dtype
            ),
        )
        scheme, layer = self._loaded(module, widths=(8,))
        output = scheme.apply_weights(
            layer, torch.ones((2, 4), dtype=torch.bfloat16), None
        )
        assert output.shape == (2, 8)
        assert torch.equal(output, torch.full((2, 8), 3, dtype=torch.bfloat16))

    def test_process_weights_rejects_scale_count_mismatch(self, modelslim):
        """Fewer input scales than logical matrices raises ValueError."""
        module = modelslim
        scheme = module._ModelSlimW8A8StaticLinearScheme()
        layer = torch.nn.Module()
        scheme.create_weights(
            layer,
            output_partition_sizes=[3, 5],
            input_size_per_partition=4,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *a, **kw: None,
        )
        layer.weight.data.copy_(torch.zeros(8, 4, dtype=torch.int8))
        # Shrink the scale/offset tensors to a single logical matrix while
        # keeping them registered, so the numel check fires after loading.
        layer.input_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        layer.input_offset = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.int8), requires_grad=False
        )
        with pytest.raises(ValueError, match="one input scale per logical"):
            scheme.process_weights_after_loading(layer)


class TestMoeMethodOffsets:
    """_ModelSlimW8A8MoEMethod registers the extra offset tensors."""

    def test_create_weights_registers_offset_parameters(self, modelslim, monkeypatch):
        """w13/w2 offsets are created and carry the channel-scale attrs."""
        module = modelslim
        captured = {}

        def fake_super_create(layer, *args, **kwargs):
            captured["super"] = (args, kwargs)

        def record_attrs(param, attrs):
            captured.setdefault("attrs", []).append((param, attrs))

        moe_cls = module._ModelSlimW8A8MoEMethod
        base_cls = moe_cls.__mro__[1]
        original = base_cls.create_weights
        base_cls.create_weights = fake_super_create
        try:
            # The real vLLM set_weight_attrs writes attributes straight onto
            # the tensor (rejecting overwrites), while the file-scope stub
            # records them in loaded_attrs; intercept the module's own symbol
            # so the assertion below is identical on both hosts.
            monkeypatch.setattr(module, "set_weight_attrs", record_attrs)
            method = object.__new__(moe_cls)
            layer = torch.nn.Module()
            method.create_weights(
                layer,
                num_experts=2,
                hidden_size=6,
                intermediate_size_per_partition=4,
                params_dtype=torch.bfloat16,
            )
        finally:
            base_cls.create_weights = original

        args, kwargs = captured["super"]
        assert args[1:] == (2, 6, 4, torch.bfloat16)
        assert layer.w13_weight_offset.shape == (2, 8, 1)
        assert layer.w13_weight_offset.dtype == torch.float32
        assert layer.w2_weight_offset.shape == (2, 6, 1)
        assert layer.w2_weight_offset.dtype == torch.float32
        # Both offsets carry the channel quant_method plus extra weight attrs.
        by_name = {
            name: attrs
            for _, attrs in captured["attrs"]
            for name in [attrs.get("quant_method")]
        }
        assert by_name["channel"] is not None
        recorded_params = [param for param, _ in captured["attrs"]]
        assert recorded_params[0] is layer.w13_weight_offset
        assert recorded_params[1] is layer.w2_weight_offset
