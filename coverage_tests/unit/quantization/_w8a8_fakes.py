# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fake vLLM / compressed-tensors / FlagGems boundary for the W8A8 adapters.

``vllm_fl.quantization.w8a8`` imports vLLM 0.24 kernels, compressed-tensors
metadata classes and FlagGems inside ``apply`` at call time.  The tests build
per-case stand-in module trees here, shadow them in ``sys.modules`` for the
duration of a test, and keep every call recorder so marshalling can be
asserted without a GPU or a real vLLM installation.

Each ``build_*_environment`` function returns a ``SimpleNamespace`` holding
both the recorders and the ``modules`` mapping consumed by
``import_under_fakes``.
"""

from __future__ import annotations

import contextlib
import enum
import importlib
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import torch

MISSING = object()


class _FakeParameter(torch.nn.Parameter):
    """Parameter stand-in accepting vLLM construction keywords."""

    def __new__(cls, data=None, requires_grad=False, **extra):
        instance = super().__new__(cls, data, requires_grad=requires_grad)
        for key, value in extra.items():
            setattr(instance, key, value)
        return instance


def _parameter_type(name: str) -> type:
    return type(name, (_FakeParameter,), {})


def _module(name: str) -> types.ModuleType:
    return types.ModuleType(name)


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _register(modules: dict[str, types.ModuleType], *items) -> None:
    for module in items:
        modules[module.__name__] = module


def _link(parent: types.ModuleType, name: str, child: types.ModuleType) -> None:
    setattr(parent, name, child)


@contextlib.contextmanager
def import_under_fakes(
    module_name: str,
    modules: dict[str, types.ModuleType],
    purge: tuple[str, ...] = (),
):
    """Import ``module_name`` with ``modules`` shadowing ``sys.modules``.

    The shadow tree (and the imported module itself) stay installed for the
    whole ``with`` block so call-time lazy imports resolve to the fakes.  On
    exit every saved entry is restored, including the modules listed in
    ``purge`` (which are removed before the import so a real earlier import
    cannot leak into the test).
    """
    saved = {name: sys.modules.get(name, MISSING) for name in modules}
    saved_module = sys.modules.get(module_name, MISSING)
    purged = {name: sys.modules.pop(name, MISSING) for name in purge}
    sys.modules.update(modules)
    try:
        importlib.invalidate_caches()
        yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if saved_module is not MISSING:
            sys.modules[module_name] = saved_module
        for name, value in purged.items():
            if value is not MISSING:
                sys.modules[name] = value
        for name, value in saved.items():
            if value is MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


# ---------------------------------------------------------------------------
# compressed-tensors metadata + packed-scheme boundary
# ---------------------------------------------------------------------------


class QuantizationType(enum.Enum):
    INT = "int"
    FLOAT = "float"


class QuantizationStrategy(enum.Enum):
    CHANNEL = "channel"
    GROUP = "group"
    TOKEN = "token"


@dataclass
class QuantizationArgs:
    num_bits: int = 8
    type: QuantizationType = QuantizationType.INT
    symmetric: bool = True
    dynamic: bool = False
    strategy: QuantizationStrategy = QuantizationStrategy.CHANNEL
    group_size: int | None = None


class CompressionFormat:
    pack_quantized = SimpleNamespace(value="pack-quantized")


def build_ct_environment() -> SimpleNamespace:
    """Stand-ins for compressed-tensors metadata and the vLLM INT8 scheme."""
    env = SimpleNamespace()
    env.base_init_calls = []
    env.base_create_calls = []
    env.base_process_calls = []
    env.original_scheme_calls = []

    class CompressedTensorsW8A8Int8:
        def __init__(
            self,
            *,
            strategy=None,
            is_static_input_scheme=False,
            input_symmetric=True,
        ):
            env.base_init_calls.append(
                {
                    "strategy": strategy,
                    "is_static_input_scheme": is_static_input_scheme,
                    "input_symmetric": input_symmetric,
                }
            )
            self.strategy = strategy
            self.is_static_input_scheme = is_static_input_scheme
            self.input_symmetric = input_symmetric

        def create_weights(
            self,
            layer,
            output_partition_sizes,
            input_size_per_partition,
            params_dtype,
            weight_loader,
            **kwargs,
        ):
            env.base_create_calls.append(
                {
                    "layer": layer,
                    "output_partition_sizes": output_partition_sizes,
                    "input_size_per_partition": input_size_per_partition,
                    "params_dtype": params_dtype,
                    "weight_loader": weight_loader,
                    "kwargs": kwargs,
                }
            )
            total = sum(output_partition_sizes)
            layer.register_parameter(
                "weight",
                torch.nn.Parameter(
                    torch.zeros(total, input_size_per_partition, dtype=torch.int8),
                    requires_grad=False,
                ),
            )

        def process_weights_after_loading(self, layer):
            env.base_process_calls.append(layer)

    env.CompressedTensorsW8A8Int8 = CompressedTensorsW8A8Int8

    class FakeCompressedTensorsConfig:
        def __init__(self, quant_format="pack-quantized"):
            self.quant_format = quant_format

        def _get_scheme_from_parts(
            self,
            weight_quant,
            input_quant,
            output_quant=None,
            format=None,
            layer_name=None,
        ):
            env.original_scheme_calls.append(
                {
                    "weight_quant": weight_quant,
                    "input_quant": input_quant,
                    "output_quant": output_quant,
                    "format": format,
                    "layer_name": layer_name,
                }
            )
            return "original-scheme"

    env.CompressedTensorsConfig = FakeCompressedTensorsConfig
    env.BasevLLMParameter = _parameter_type("BasevLLMParameter")
    env.PackedvLLMParameter = _parameter_type("PackedvLLMParameter")

    modules: dict[str, types.ModuleType] = {}
    compressed = _package("compressed_tensors")
    compressed_config = _module("compressed_tensors.config")
    compressed_config.CompressionFormat = CompressionFormat
    quantization = _module("compressed_tensors.quantization")
    quantization.QuantizationArgs = QuantizationArgs
    quantization.QuantizationStrategy = QuantizationStrategy
    quantization.QuantizationType = QuantizationType
    _link(compressed, "config", compressed_config)
    _link(compressed, "quantization", quantization)

    vllm = _package("vllm")
    executor = _package("vllm.model_executor")
    layers = _package("vllm.model_executor.layers")
    layers_quant = _package("vllm.model_executor.layers.quantization")
    ct_pkg = _package(
        "vllm.model_executor.layers.quantization.compressed_tensors"
    )
    ct_config = _module(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors"
    )
    ct_config.CompressedTensorsConfig = FakeCompressedTensorsConfig
    ct_schemes = _package(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes"
    )
    ct_schemes.CompressedTensorsW8A8Int8 = CompressedTensorsW8A8Int8
    parameter = _module("vllm.model_executor.parameter")
    parameter.BasevLLMParameter = env.BasevLLMParameter
    parameter.PackedvLLMParameter = env.PackedvLLMParameter
    _link(ct_pkg, "compressed_tensors", ct_config)
    _link(ct_pkg, "schemes", ct_schemes)
    _link(layers_quant, "compressed_tensors", ct_pkg)
    _link(layers, "quantization", layers_quant)
    _link(executor, "layers", layers)
    _link(executor, "parameter", parameter)
    _link(vllm, "model_executor", executor)

    _register(
        modules,
        compressed,
        compressed_config,
        quantization,
        vllm,
        executor,
        layers,
        layers_quant,
        ct_pkg,
        ct_config,
        ct_schemes,
        parameter,
    )
    env.modules = modules
    return env


# ---------------------------------------------------------------------------
# W8A8 MoE expert boundary
# ---------------------------------------------------------------------------


class MoEActivation(enum.Enum):
    SILU = "silu"
    GELU = "gelu"
    GELU_TANH = "gelu_tanh"

    @property
    def is_gated(self) -> bool:
        return self in (MoEActivation.SILU, MoEActivation.GELU, MoEActivation.GELU_TANH)


def build_moe_experts_environment() -> SimpleNamespace:
    """Stand-ins for the vLLM fused-MoE base classes and FlagGems kernel."""
    env = SimpleNamespace()
    env.use_flaggems_vllm = False
    env.experts_init_calls = []
    env.super_apply_calls = []
    env.vllm_fused_experts_calls = []
    env.flaggems_calls = []
    env.flaggems_vllm_calls = []
    env.vllm_result = None
    env.flaggems_result = None

    class ExpertTokensMetadata:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    env.ExpertTokensMetadata = ExpertTokensMetadata

    class TritonExperts:
        def __init__(self, moe_config, quant_config):
            env.experts_init_calls.append((moe_config, quant_config))
            self.moe_config = moe_config
            self.quant_config = quant_config

        def apply(self, **kwargs):
            env.super_apply_calls.append(kwargs)
            return None

    env.TritonExperts = TritonExperts

    def fused_experts(**kwargs):
        env.vllm_fused_experts_calls.append(kwargs)
        return env.vllm_result

    env.fused_experts = fused_experts

    def fused_experts_impl(**kwargs):
        env.flaggems_calls.append(kwargs)
        return env.flaggems_result

    def fused_experts_impl_vllm(**kwargs):
        env.flaggems_vllm_calls.append(kwargs)
        return env.flaggems_result

    modules: dict[str, types.ModuleType] = {}
    vllm = _package("vllm")
    executor = _package("vllm.model_executor")
    layers = _package("vllm.model_executor.layers")
    fused_moe = _package("vllm.model_executor.layers.fused_moe")
    modular_kernel = _module(
        "vllm.model_executor.layers.fused_moe.modular_kernel"
    )
    modular_kernel.ExpertTokensMetadata = ExpertTokensMetadata
    activation = _module("vllm.model_executor.layers.fused_moe.activation")
    activation.MoEActivation = MoEActivation
    experts = _package("vllm.model_executor.layers.fused_moe.experts")
    triton_moe = _module(
        "vllm.model_executor.layers.fused_moe.experts.triton_moe"
    )
    triton_moe.TritonExperts = TritonExperts
    fused_moe_fn = _module("vllm.model_executor.layers.fused_moe.fused_moe")
    fused_moe_fn.fused_experts = fused_experts
    _link(experts, "triton_moe", triton_moe)
    _link(fused_moe, "modular_kernel", modular_kernel)
    _link(fused_moe, "activation", activation)
    _link(fused_moe, "experts", experts)
    _link(fused_moe, "fused_moe", fused_moe_fn)
    _link(layers, "fused_moe", fused_moe)
    _link(executor, "layers", layers)
    _link(vllm, "model_executor", executor)

    flag_gems = _module("flag_gems")
    flag_gems.fused_experts_impl = fused_experts_impl
    flaggems_vllm = _module("flaggems_vllm")
    flaggems_vllm.fused_experts_impl = fused_experts_impl_vllm

    fl_envs = _module("vllm_fl.envs")
    fl_envs.__dict__["VLLM_FL_USE_FLAGGEMS_VLLM"] = False

    _register(
        modules,
        vllm,
        executor,
        layers,
        fused_moe,
        modular_kernel,
        activation,
        experts,
        triton_moe,
        fused_moe_fn,
        flag_gems,
        flaggems_vllm,
        fl_envs,
    )
    env.modules = modules
    env.fl_envs = fl_envs
    return env


# ---------------------------------------------------------------------------
# W8A8 MoE selector boundary
# ---------------------------------------------------------------------------


@dataclass
class QuantConfig:
    """Quant config viewed by the W8A8 experts adapters."""

    use_int8_w8a8: bool = True
    per_act_token_quant: bool = True
    block_shape: tuple | None = None
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None
    w1_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None
    gemm1_clamp_limit: float | None = None
    gemm1_alpha: float | None = None
    gemm1_beta: float | None = None


class Int8MoeBackend(enum.Enum):
    TRITON = "triton"
    CUTLASS = "cutlass"


def build_moe_selector_environment() -> SimpleNamespace:
    """Stand-ins for the vLLM 0.24 INT8 MoE oracle and scheme modules."""
    env = SimpleNamespace()
    env.is_cuda = True
    env.is_out_of_tree = False
    env.oot_enabled = True
    env.flaggems_default = True
    env.flaggems_ops = {}
    env.logger_calls = []
    env.make_quant_calls = []
    env.int8_config_calls = []
    env.oracle_selector_calls = []
    env.oracle_selector_result = ("native-backend", object())

    class Logger:
        def info_once(self, *args, **kwargs):
            env.logger_calls.append((args, kwargs))

    env.logger = Logger()

    def init_logger(name):
        env.logger_name = name
        return env.logger

    def make_int8_moe_quant_config(
        w1_scale=None,
        w2_scale=None,
        a1_scale=None,
        a2_scale=None,
        w1_bias=None,
        w2_bias=None,
        per_act_token_quant=False,
    ):
        env.make_quant_calls.append(
            {
                "w1_scale": w1_scale,
                "w2_scale": w2_scale,
                "a1_scale": a1_scale,
                "a2_scale": a2_scale,
                "w1_bias": w1_bias,
                "w2_bias": w2_bias,
                "per_act_token_quant": per_act_token_quant,
            }
        )
        return "original-quant-config"

    def int8_w8a8_moe_quant_config(**kwargs):
        env.int8_config_calls.append(kwargs)
        return "w8a8-quant-config"

    def oracle_selector(config, weight_key=None, activation_key=None):
        env.oracle_selector_calls.append(
            {
                "config": config,
                "weight_key": weight_key,
                "activation_key": activation_key,
            }
        )
        return env.oracle_selector_result

    class TritonExperts:
        pass

    class FlagGemsW8A8Experts:
        pass

    class VllmFunctionalW8A8Experts:
        pass

    kInt8DynamicTokenSym = object()
    kInt8StaticChannelSym = object()
    env.kInt8DynamicTokenSym = kInt8DynamicTokenSym
    env.kInt8StaticChannelSym = kInt8StaticChannelSym
    env.TritonExperts = TritonExperts
    env.FlagGemsW8A8Experts = FlagGemsW8A8Experts
    env.VllmFunctionalW8A8Experts = VllmFunctionalW8A8Experts

    modules: dict[str, types.ModuleType] = {}
    vllm = _package("vllm")
    logger = _module("vllm.logger")
    logger.init_logger = init_logger
    executor = _package("vllm.model_executor")
    layers = _package("vllm.model_executor.layers")
    fused_moe = _package("vllm.model_executor.layers.fused_moe")
    oracle = _package(
        "vllm.model_executor.layers.fused_moe.oracle"
    )
    oracle_int8 = _module(
        "vllm.model_executor.layers.fused_moe.oracle.int8"
    )
    oracle_int8.Int8MoeBackend = Int8MoeBackend
    oracle_int8.select_int8_moe_backend = oracle_selector
    moe_config_mod = _module("vllm.model_executor.layers.fused_moe.config")
    moe_config_mod.int8_w8a8_moe_quant_config = int8_w8a8_moe_quant_config
    activation = _module("vllm.model_executor.layers.fused_moe.activation")
    activation.MoEActivation = MoEActivation
    experts = _package("vllm.model_executor.layers.fused_moe.experts")
    triton_moe = _module(
        "vllm.model_executor.layers.fused_moe.experts.triton_moe"
    )
    triton_moe.TritonExperts = TritonExperts
    _link(oracle, "int8", oracle_int8)
    _link(experts, "triton_moe", triton_moe)
    _link(fused_moe, "oracle", oracle)
    _link(fused_moe, "config", moe_config_mod)
    _link(fused_moe, "activation", activation)
    _link(fused_moe, "experts", experts)
    _link(layers, "fused_moe", fused_moe)
    layers_quant = _package("vllm.model_executor.layers.quantization")
    quant_utils = _module(
        "vllm.model_executor.layers.quantization.utils.quant_utils"
    )
    quant_utils.kInt8DynamicTokenSym = kInt8DynamicTokenSym
    quant_utils.kInt8StaticChannelSym = kInt8StaticChannelSym
    quant_utils_pkg = _package(
        "vllm.model_executor.layers.quantization.utils"
    )
    _link(quant_utils_pkg, "quant_utils", quant_utils)
    ct_moe = _package(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe"
    )
    scheme_module = _module(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
    )
    scheme_module.make_int8_moe_quant_config = make_int8_moe_quant_config
    scheme_module.select_int8_moe_backend = oracle_selector
    _link(ct_moe, "compressed_tensors_moe_w8a8_int8", scheme_module)
    _link(layers_quant, "quant_utils", quant_utils_pkg)
    _link(layers, "quantization", layers_quant)
    _link(executor, "layers", layers)
    _link(vllm, "model_executor", executor)
    platforms = _module("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        is_cuda=lambda: env.is_cuda,
        is_out_of_tree=lambda: env.is_out_of_tree,
        device_type="cuda",
    )
    _link(vllm, "platforms", platforms)

    fl_utils = _module("vllm_fl.utils")
    fl_utils.is_oot_enabled = lambda: env.oot_enabled
    fl_utils.use_flaggems_op = lambda op: env.flaggems_ops.get(
        op, env.flaggems_default
    )

    fl_moe_experts = _module("vllm_fl.quantization.w8a8.moe_experts")
    fl_moe_experts.FlagGemsW8A8Experts = FlagGemsW8A8Experts
    fl_moe_experts.VllmFunctionalW8A8Experts = VllmFunctionalW8A8Experts

    _register(
        modules,
        vllm,
        logger,
        executor,
        layers,
        fused_moe,
        oracle,
        oracle_int8,
        moe_config_mod,
        activation,
        experts,
        triton_moe,
        layers_quant,
        quant_utils,
        quant_utils_pkg,
        ct_moe,
        scheme_module,
        platforms,
        fl_utils,
        fl_moe_experts,
    )
    env.modules = modules
    env.scheme_module = scheme_module
    env.oracle_module = oracle_int8
    env.fl_utils = fl_utils
    return env


# ---------------------------------------------------------------------------
# W8A8 linear kernel boundary
# ---------------------------------------------------------------------------


class CachedOpRecorder:
    """Callable stand-in for ``vllm_fl.dispatch.CachedOp``."""

    def __init__(self, op_name, result=None):
        self.op_name = op_name
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def build_linear_environment() -> SimpleNamespace:
    """Stand-ins for the vLLM INT8 linear kernel and FlagGems scaled-mm."""
    env = SimpleNamespace()
    env.oot_enabled = True
    env.flaggems_default = True
    env.flaggems_ops = {}
    env.base_init_calls = []
    env.replace_calls = []
    env.scaled_mm_calls = []
    env.quant_result = None
    env.scaled_mm_result = None
    env.cached_op = CachedOpRecorder("dynamic_per_token_quant_int8")

    class Int8ScaledMMLinearLayerConfig:
        def __init__(
            self,
            *,
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=True,
        ):
            self.is_channelwise = is_channelwise
            self.is_static_input_scheme = is_static_input_scheme
            self.input_symmetric = input_symmetric

    env.Int8ScaledMMLinearLayerConfig = Int8ScaledMMLinearLayerConfig

    class Int8ScaledMMLinearKernel:
        layer_param_names = (
            "weight",
            "weight_scale",
            "input_scale",
            "input_zp",
            "azp_adj",
        )

        def __init__(self):
            env.base_init_calls.append(True)

        def _get_layer_params(self, layer):
            return tuple(getattr(layer, name) for name in self.layer_param_names)

    env.Int8ScaledMMLinearKernel = Int8ScaledMMLinearKernel

    def replace_parameter(module, name, new_param):
        env.replace_calls.append((module, name, new_param))
        setattr(module, name, new_param)

    def scaled_mm(*args, **kwargs):
        env.scaled_mm_calls.append((args, kwargs))
        return env.scaled_mm_result

    modules: dict[str, types.ModuleType] = {}
    vllm = _package("vllm")
    executor = _package("vllm.model_executor")
    kernels = _package("vllm.model_executor.kernels")
    linear = _module("vllm.model_executor.kernels.linear")
    linear.Int8ScaledMMLinearKernel = Int8ScaledMMLinearKernel
    linear.Int8ScaledMMLinearLayerConfig = Int8ScaledMMLinearLayerConfig
    layers = _package("vllm.model_executor.layers")
    layers_quant = _package("vllm.model_executor.layers.quantization")
    utils = _module("vllm.model_executor.layers.quantization.utils")
    utils.replace_parameter = replace_parameter
    platforms = _module("vllm.platforms")

    class PlatformEnum(enum.Enum):
        OOT = "oot"

    platforms.PlatformEnum = PlatformEnum
    _link(layers_quant, "utils", utils)
    _link(layers, "quantization", layers_quant)
    _link(executor, "kernels", kernels)
    _link(executor, "layers", layers)
    _link(vllm, "model_executor", executor)
    _link(vllm, "platforms", platforms)

    flag_gems = _module("flag_gems")
    flag_gems.scaled_mm = scaled_mm

    dispatch = _module("vllm_fl.dispatch")

    class CachedOp:
        def __new__(cls, op_name):
            env.cached_op.op_name = op_name
            return env.cached_op

    dispatch.CachedOp = CachedOp

    fl_utils = _module("vllm_fl.utils")
    fl_utils.is_oot_enabled = lambda: env.oot_enabled
    fl_utils.use_flaggems_op = lambda op: env.flaggems_ops.get(
        op, env.flaggems_default
    )

    _register(
        modules,
        vllm,
        executor,
        kernels,
        linear,
        layers,
        layers_quant,
        utils,
        platforms,
        flag_gems,
        dispatch,
        fl_utils,
    )
    env.modules = modules
    env.PlatformEnum = PlatformEnum
    env.Int8ScaledMMLinearLayerConfig = Int8ScaledMMLinearLayerConfig
    return env
