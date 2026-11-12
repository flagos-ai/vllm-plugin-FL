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

"""Fake vLLM / compressed-tensors / FlagGems boundary for ARM W4A8 tests.

``vllm_fl.quantization.arm_cpu_w4a8`` imports vLLM 0.24, compressed-tensors
and FlagGems lazily: once in ``install_arm_cpu_packed_w4a8`` and again inside
the runtime hooks it installs.  None of those packages is importable (or
CPU-safe) outside the ARM CPU container, so every test builds a fresh set of
stand-in modules and classes here and temporarily shadows the real names in
``sys.modules``.

The stand-ins are recorders, not simulations: every "original" vLLM method
the adapter is expected to delegate to appends its arguments to
``FakeEnvironment`` lists, so tests can assert the adapter's forwarding
behavior without touching vLLM internals.  The adapter-owned classes (the
scheme, its config, and the dynamic-4bit kernel) are re-created for every
test, which keeps the monkey patches the adapter installs from leaking into
another test or into a real vLLM installation.

``flag_gems.quantized_linear`` gets two deterministic CPU stand-ins whose
results tests can re-derive from their inputs:

* ``unpack_from_int32`` expands each int32 lane to eight int8 nibbles: the
  fake repeats every value eight times and reduces it modulo seven, so the
  expected tensor is easy to compute independently.
* ``pack_rhs_qsi4c128p`` returns a uint8 ``(n, k // 8)`` tensor of a fixed
  sentinel value.
* ``w4a8_g128_linear`` broadcasts each input row's sum across ``n`` output
  columns.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import torch

_MISSING = object()


class _FakeParameter(torch.nn.Parameter):
    """Parameter stand-in that accepts and stores vLLM-specific keywords."""

    def __new__(cls, data=None, requires_grad=False, **extra):
        instance = super().__new__(cls, data, requires_grad=requires_grad)
        for key, value in extra.items():
            setattr(instance, key, value)
        return instance


def _parameter_type(name: str) -> type:
    return type(name, (_FakeParameter,), {})


@dataclass
class _MPLinearLayerConfig:
    """Subset of vLLM's MPLinearLayerConfig the adapter constructs."""

    full_weight_shape: tuple[int, int]
    partition_weight_shape: tuple[int, int]
    weight_type: torch.dtype
    act_type: torch.dtype
    group_size: int
    zero_points: bool
    has_g_idx: bool


def build_environment() -> SimpleNamespace:
    """Create an isolated fake boundary and its call recorders."""
    env = SimpleNamespace()

    # Recorders populated by the fake originals and FlagGems kernels.
    env.original_init_calls = []
    env.original_create_weights_calls = []
    env.original_get_scheme_calls = []
    env.original_process_calls = []
    env.original_apply_calls = []
    env.choose_calls = []
    env.unpack_calls = []
    env.pack_calls = []
    env.linear_calls = []
    env.dynamic_token_calls = []
    env.replace_calls = []

    env.dynamic_result = False
    env.original_get_scheme_result = "original-scheme"

    # -- compressed_tensors -------------------------------------------------
    class CompressionFormat:
        """Mirror of compressed_tensors.config.CompressionFormat."""

        pack_quantized = SimpleNamespace(value="pack-quantized")

    env.CompressionFormat = CompressionFormat

    def unpack_from_int32(tensor: torch.Tensor, num_bits: int, shape):
        env.unpack_calls.append((tensor, num_bits, tuple(shape)))
        nibbles = tensor.repeat_interleave(8, dim=-1)
        return nibbles.to(torch.int64).remainder(7).to(torch.int8).reshape(tuple(shape))

    env.unpack_from_int32 = unpack_from_int32

    # -- vLLM kernels -------------------------------------------------------
    env.MPLinearLayerConfig = _MPLinearLayerConfig

    class FakeKernel:
        def __init__(self, config, **names):
            self.config = config
            self.init_names = names

    env.KernelClass = FakeKernel

    def choose_mp_linear_kernel(config):
        env.choose_calls.append(config)
        return FakeKernel

    env.choose_mp_linear_kernel = choose_mp_linear_kernel

    class Dynamic4bitLinearKernel:
        def process_weights_after_loading(self, layer):
            env.original_process_calls.append(layer)

        def apply_weights(self, layer, input, bias=None):
            env.original_apply_calls.append((layer, input, bias))
            return torch.full((input.shape[0], 2), -1.25, dtype=input.dtype)

    env.Dynamic4bitLinearKernel = Dynamic4bitLinearKernel

    # -- vLLM compressed-tensors scheme / config ----------------------------
    class CompressedTensorsW4A8Int:
        def __init__(
            self,
            strategy: str,
            num_bits: int,
            group_size: int | None = None,
            is_static_input_scheme: bool = False,
            input_symmetric: bool = True,
        ):
            env.original_init_calls.append(
                {
                    "strategy": strategy,
                    "num_bits": num_bits,
                    "group_size": group_size,
                    "is_static_input_scheme": is_static_input_scheme,
                    "input_symmetric": input_symmetric,
                }
            )
            self.strategy = strategy
            self.num_bits = num_bits
            self.group_size = group_size
            self.is_static_input_scheme = is_static_input_scheme
            self.input_symmetric = input_symmetric
            self.quant_type = torch.int8

        def create_weights(
            self,
            layer,
            output_size,
            input_size,
            output_partition_sizes,
            input_size_per_partition,
            params_dtype,
            weight_loader,
            **kwargs,
        ):
            env.original_create_weights_calls.append(
                {
                    "layer": layer,
                    "output_size": output_size,
                    "input_size": input_size,
                    "output_partition_sizes": output_partition_sizes,
                    "input_size_per_partition": input_size_per_partition,
                    "params_dtype": params_dtype,
                    "weight_loader": weight_loader,
                    "kwargs": kwargs,
                }
            )

    env.CompressedTensorsW4A8Int = CompressedTensorsW4A8Int

    class CompressedTensorsConfig:
        def _is_dynamic_token_w4a8_int(self, weight_quant, input_quant):
            env.dynamic_token_calls.append((weight_quant, input_quant))
            return env.dynamic_result

        def _get_scheme_from_parts(
            self,
            weight_quant,
            input_quant,
            output_quant=None,
            format=None,
            layer_name=None,
        ):
            env.original_get_scheme_calls.append(
                {
                    "weight_quant": weight_quant,
                    "input_quant": input_quant,
                    "output_quant": output_quant,
                    "format": format,
                    "layer_name": layer_name,
                }
            )
            return env.original_get_scheme_result

    env.CompressedTensorsConfig = CompressedTensorsConfig

    # -- vLLM parameter classes and replace_parameter -----------------------
    env.BasevLLMParameter = _parameter_type("BasevLLMParameter")
    env.ChannelQuantScaleParameter = _parameter_type("ChannelQuantScaleParameter")
    env.GroupQuantScaleParameter = _parameter_type("GroupQuantScaleParameter")
    env.PackedvLLMParameter = _parameter_type("PackedvLLMParameter")

    def replace_parameter(module, param_name, new_param):
        env.replace_calls.append((module, param_name, new_param))
        setattr(module, param_name, new_param)

    env.replace_parameter = replace_parameter

    # -- flag_gems.quantized_linear -----------------------------------------
    def pack_rhs_qsi4c128p(weight, scale):
        env.pack_calls.append((weight, scale))
        rows, cols = weight.shape
        return torch.full((rows, cols // 8), 11, dtype=torch.uint8)

    def w4a8_g128_linear(input, weight, n, k):
        env.linear_calls.append((input, weight, n, k))
        return input.sum(dim=-1, keepdim=True).expand(-1, n).clone()

    env.pack_rhs_qsi4c128p = pack_rhs_qsi4c128p
    env.w4a8_g128_linear = w4a8_g128_linear

    env.modules = _build_modules(env)
    return env


def _module(name: str) -> types.ModuleType:
    return types.ModuleType(name)


def _build_modules(env: SimpleNamespace) -> dict[str, types.ModuleType]:
    """Assemble the fake module tree, wiring child attributes as vLLM does."""
    modules: dict[str, types.ModuleType] = {}

    compressed = _module("compressed_tensors")
    compressed_config = _module("compressed_tensors.config")
    compressed_config.CompressionFormat = env.CompressionFormat
    compressors = _module("compressed_tensors.compressors")
    pack_quantized = _module("compressed_tensors.compressors.pack_quantized")
    helpers = _module("compressed_tensors.compressors.pack_quantized.helpers")
    helpers.unpack_from_int32 = env.unpack_from_int32
    pack_quantized.helpers = helpers
    compressors.pack_quantized = pack_quantized
    compressed.config = compressed_config
    compressed.compressors = compressors

    vllm = _module("vllm")
    executor = _module("vllm.model_executor")
    kernels = _module("vllm.model_executor.kernels")
    linear = _module("vllm.model_executor.kernels.linear")
    mixed_precision = _module("vllm.model_executor.kernels.linear.mixed_precision")
    dynamic_4bit = _module(
        "vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit"
    )
    dynamic_4bit.Dynamic4bitLinearKernel = env.Dynamic4bitLinearKernel
    linear.MPLinearLayerConfig = env.MPLinearLayerConfig
    linear.choose_mp_linear_kernel = env.choose_mp_linear_kernel
    linear.mixed_precision = mixed_precision
    mixed_precision.dynamic_4bit = dynamic_4bit
    kernels.linear = linear
    executor.kernels = kernels

    layers = _module("vllm.model_executor.layers")
    quantization = _module("vllm.model_executor.layers.quantization")
    compressed_tensors = _module(
        "vllm.model_executor.layers.quantization.compressed_tensors"
    )
    config_module = _module(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
    )
    config_module.CompressedTensorsConfig = env.CompressedTensorsConfig
    schemes = _module(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes"
    )
    scheme_module = _module(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a8_int"
    )
    scheme_module.CompressedTensorsW4A8Int = env.CompressedTensorsW4A8Int
    schemes.compressed_tensors_w4a8_int = scheme_module
    compressed_tensors.compressed_tensors = config_module
    compressed_tensors.schemes = schemes
    quantization.compressed_tensors = compressed_tensors
    quantization_utils = _module(
        "vllm.model_executor.layers.quantization.utils"
    )
    quantization_utils.replace_parameter = env.replace_parameter
    quantization.utils = quantization_utils
    layers.quantization = quantization
    executor.layers = layers

    parameter = _module("vllm.model_executor.parameter")
    parameter.BasevLLMParameter = env.BasevLLMParameter
    parameter.ChannelQuantScaleParameter = env.ChannelQuantScaleParameter
    parameter.GroupQuantScaleParameter = env.GroupQuantScaleParameter
    parameter.PackedvLLMParameter = env.PackedvLLMParameter
    executor.parameter = parameter

    vllm.model_executor = executor

    flag_gems = _module("flag_gems")
    flag_gems_linear = _module("flag_gems.quantized_linear")
    flag_gems_linear.pack_rhs_qsi4c128p = env.pack_rhs_qsi4c128p
    flag_gems_linear.w4a8_g128_linear = env.w4a8_g128_linear
    flag_gems.quantized_linear = flag_gems_linear

    for module in (
        compressed,
        compressed_config,
        compressors,
        pack_quantized,
        helpers,
        vllm,
        executor,
        kernels,
        linear,
        mixed_precision,
        dynamic_4bit,
        layers,
        quantization,
        compressed_tensors,
        config_module,
        schemes,
        scheme_module,
        quantization_utils,
        parameter,
        flag_gems,
        flag_gems_linear,
    ):
        modules[module.__name__] = module
    return modules


class InjectedModules:
    """Context manager that shadows and restores ``sys.modules`` entries."""

    def __init__(self, env: SimpleNamespace):
        self._env = env
        self._saved: dict[str, object] = {}

    def __enter__(self) -> SimpleNamespace:
        for name, module in self._env.modules.items():
            self._saved[name] = sys.modules.get(name, _MISSING)
            sys.modules[name] = module
        return self._env

    def __exit__(self, exc_type, exc, tb) -> bool:
        for name, original in self._saved.items():
            if original is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        return False


def injected_modules(env: SimpleNamespace) -> InjectedModules:
    return InjectedModules(env)


def make_scheme(
    env: SimpleNamespace,
    *,
    packed: bool = False,
    num_bits: int = 4,
    group_size: int | None = 128,
) -> object:
    """Instantiate the (patched) scheme after ``install`` has run."""
    return env.CompressedTensorsW4A8Int(
        strategy="group",
        num_bits=num_bits,
        group_size=group_size,
        is_static_input_scheme=False,
        input_symmetric=True,
        packed=packed,
    )


def make_kernel(
    env: SimpleNamespace,
    *,
    packed: bool = True,
    group_size: int = 128,
    zero_points: bool = False,
    k: int = 256,
    n: int = 48,
    w_q_name: str = "weight_packed",
    w_s_name: str = "weight_scale",
) -> object:
    """Instantiate the (patched) dynamic-4bit kernel with a fake config."""
    kernel = env.Dynamic4bitLinearKernel()
    kernel.config = SimpleNamespace(
        group_size=group_size,
        zero_points=zero_points,
        partition_weight_shape=(k, n),
    )
    kernel.w_q_name = w_q_name
    kernel.w_s_name = w_s_name
    if packed:
        kernel._vllm_fl_arm_packed_checkpoint = True
    return kernel


def make_layer(
    env: SimpleNamespace,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    weight_name: str = "weight_packed",
    weight_shape: torch.Tensor | object = _MISSING,
    alias: torch.Tensor | object = _MISSING,
) -> torch.nn.Module:
    """Build a layer carrying the compressed-tensors G128 parameters."""
    layer = torch.nn.Module()
    layer.register_parameter(weight_name, env.BasevLLMParameter(data=weight))
    layer.register_parameter("weight_scale", env.BasevLLMParameter(data=scale))
    if weight_shape is not _MISSING:
        layer.register_parameter(
            "weight_shape", env.BasevLLMParameter(data=weight_shape)
        )
    if alias is not _MISSING:
        layer.register_parameter("weight", env.BasevLLMParameter(data=alias))
    return layer
