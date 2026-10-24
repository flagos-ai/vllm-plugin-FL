"""Qwen FP8 capability, checkpoint scale and existing BF16 path contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_fl.models.qwen3_8_flash_next.gpu.fp8_fallback import (
    QwenFp8Bf16Config,
    QwenFp8Bf16LinearMethod,
    QwenFp8Bf16MoEMethod,
)
from vllm_fl.models.qwen3_8_flash_next.gpu.quantization import (
    with_platform_quantization,
)


def fp8_config(*, block=None, serialized=True, activation="dynamic"):
    config = Fp8Config(
        is_checkpoint_fp8_serialized=serialized,
        weight_block_size=block,
        activation_scheme=activation,
    )
    config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    return config


@pytest.mark.parametrize(
    "capability,expected",
    [(75, False), (80, False), (89, True), (90, True), (100, True)],
)
def test_native_fp8_requires_capability(monkeypatch, capability, expected):
    from vllm.platforms.interface import DeviceCapability

    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    monkeypatch.setattr(
        PlatformFL,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(capability // 10, capability % 10),
    )
    assert PlatformFL.supports_fp8() is expected
    assert PlatformFL().supports_fp8() is expected


@pytest.mark.parametrize(
    "vendor", ["hygon", "mthreads", "metax", "iluvatar", "kunlunxin", "ascend"]
)
def test_cuda_compatibility_is_not_fp8_support(monkeypatch, vendor):
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    probe = Mock(side_effect=AssertionError("must not probe NVIDIA capabilities"))
    monkeypatch.setattr(PlatformFL, "get_device_capability", probe)
    assert not PlatformFL.supports_fp8()
    probe.assert_not_called()


@pytest.mark.parametrize(
    "supported,fnuz", [(False, False), (True, False), (True, True)]
)
def test_amd_uses_native_fp8_capability_and_format(monkeypatch, supported, fnuz):
    import sys
    from types import ModuleType

    from vllm_fl.platform import PlatformFL

    native = ModuleType("vllm.platforms.rocm")
    native.RocmPlatform = SimpleNamespace(
        supports_fp8=lambda: supported, is_fp8_fnuz=lambda: fnuz
    )
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(PlatformFL, "vendor_name", "amd")
    assert PlatformFL.supports_fp8() is supported
    assert PlatformFL.is_fp8_fnuz() is fnuz
    assert PlatformFL.fp8_dtype() == (
        torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    )


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_model_policy_preserves_native_fp8_and_updates_runner_cache(
    monkeypatch, supported, quantized
):
    import vllm.platforms

    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(supports_fp8=lambda: supported),
    )
    original = fp8_config() if quantized else None
    cache = SimpleNamespace(cache_dtype="fp8")
    config = SimpleNamespace(
        quant_config=original,
        cache_config=cache,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    assert with_platform_quantization(config) is config
    assert cache.cache_dtype == ("fp8" if supported else "bfloat16")
    if quantized and not supported:
        assert isinstance(config.quant_config, QwenFp8Bf16Config)
        assert (
            config.quant_config.packed_modules_mapping
            == original.packed_modules_mapping
        )
    else:
        assert config.quant_config is original
    selected = config.quant_config
    assert with_platform_quantization(config).quant_config is selected


@pytest.mark.parametrize("name", [None, "compressed-tensors"])
def test_non_fp8_policy_does_not_probe_or_rewrite(monkeypatch, name):
    import vllm.platforms

    monkeypatch.setattr(vllm.platforms, "current_platform", SimpleNamespace())
    quant = None if name is None else SimpleNamespace(get_name=lambda: name)
    config = SimpleNamespace(
        quant_config=quant, cache_config=SimpleNamespace(cache_dtype="auto")
    )
    assert with_platform_quantization(config).quant_config is quant


@pytest.fixture
def tp1(monkeypatch):
    import vllm.distributed
    import vllm.distributed.parallel_state as parallel_state
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.quantization import fp8

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )

    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(fp8, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)


@pytest.mark.parametrize(
    "block,activation", [(None, "dynamic"), (None, "static"), ([4, 4], "dynamic")]
)
@pytest.mark.parametrize("backend", ["default", "rocm"])
def test_fp8_checkpoint_linear_uses_bf16_gemm(
    tp1, monkeypatch, block, activation, backend
):
    from vllm.model_executor.layers import linear, utils

    # Keep upstream GEMM selection and LinearMethod.apply real. Only replace
    # the GPU-only ROCm op at its boundary, since this test uses CPU tensors.
    # Exercise both choices on every CI host, including non-ROCm builders.
    platform = SimpleNamespace(
        is_cpu=lambda: False,
        is_rocm=lambda: backend == "rocm",
        is_cuda_alike=lambda: True,
    )
    monkeypatch.setattr(linear, "current_platform", platform)
    monkeypatch.setattr(utils, "current_platform", platform)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    rocm_gemm = Mock(side_effect=torch.nn.functional.linear)
    monkeypatch.setattr(torch.ops.vllm, "rocm_unquantized_gemm", rocm_gemm)

    config = QwenFp8Bf16Config(fp8_config(block=block, activation=activation))
    layer = ReplicatedLinear(
        5, 7, bias=False, params_dtype=torch.bfloat16, quant_config=config
    )
    weight = (torch.arange(35).reshape(7, 5).float() / 16 - 1).to(torch.float8_e4m3fn)
    layer.weight.weight_loader(layer.weight, weight)
    assert layer.weight.dtype == torch.bfloat16
    if block:
        scale = torch.tensor([[0.25, 0.5], [1.5, 2.0]])
        parameter = layer.weight_scale_inv
        expected = (
            weight.float()
            * scale.repeat_interleave(4, 0).repeat_interleave(4, 1)[:7, :5]
        )
    else:
        scale = torch.tensor(0.25)
        parameter = layer.weight_scale
        expected = weight.float() * scale
    parameter.weight_loader(parameter, scale)
    layer.quant_method.process_weights_after_loading(layer)
    layer.quant_method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight, expected.bfloat16(), rtol=0, atol=0)
    x = torch.arange(15).reshape(3, 5).bfloat16() / 8
    actual, _ = layer(x)
    torch.testing.assert_close(
        actual, torch.nn.functional.linear(x, expected.bfloat16()), rtol=0, atol=0
    )
    if backend == "rocm":
        rocm_gemm.assert_called_once()
        args = rocm_gemm.call_args.args
        assert args[0] is x
        assert args[1] is layer.weight
        assert args[0].dtype == args[1].dtype == torch.bfloat16
    else:
        rocm_gemm.assert_not_called()
    assert QwenFp8Bf16LinearMethod.apply is UnquantizedLinearMethod.apply


def test_fused_linear_retains_distinct_checkpoint_scales(tp1):
    config = QwenFp8Bf16Config(fp8_config())
    layer = MergedColumnParallelLinear(
        4, [4, 8], bias=False, params_dtype=torch.bfloat16, quant_config=config
    )
    expected = []
    for shard, width in enumerate([4, 8]):
        weight = torch.full((width, 4), 2.0 + shard).to(torch.float8_e4m3fn)
        scale = torch.tensor(0.5 + shard)
        layer.weight.weight_loader(layer.weight, weight, shard)
        layer.weight_scale.weight_loader(layer.weight_scale, scale, shard)
        expected.append(weight.float() * scale)
    layer.quant_method.process_weights_after_loading(layer)
    torch.testing.assert_close(
        layer.weight, torch.cat(expected).bfloat16(), rtol=0, atol=0
    )


def test_online_fp8_and_ignored_layers_use_existing_unquantized_method(tp1):
    config = QwenFp8Bf16Config(fp8_config(serialized=False))
    layer = ReplicatedLinear(4, 4, quant_config=config, prefix="proj")
    assert type(layer.quant_method) is UnquantizedLinearMethod
    config = QwenFp8Bf16Config(fp8_config())
    config.ignored_layers = ["proj"]
    layer = ReplicatedLinear(4, 4, quant_config=config, prefix="proj")
    assert type(layer.quant_method) is UnquantizedLinearMethod


@pytest.mark.parametrize("block", [None, [4, 4]])
def test_moe_checkpoint_scales_are_per_expert_and_gate_up(tp1, monkeypatch, block):
    from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod

    # Exercise upstream FP8 allocation/metadata and our post-load conversion;
    # backend setup is covered separately by the MoE routing tests.
    method = object.__new__(QwenFp8Bf16MoEMethod)
    torch.nn.Module.__init__(method)
    method.quant_config = fp8_config(block=block)
    method.weight_block_size = block
    method.block_quant = block is not None
    method.weight_scale_name = "weight_scale_inv" if block else "weight_scale"
    method.moe = SimpleNamespace(has_bias=False)
    layer = torch.nn.Module()
    loader = lambda p, w: p.data.copy_(w)
    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=5,
        intermediate_size_per_partition=4,
        params_dtype=torch.bfloat16,
        weight_loader=loader,
    )
    setup = Mock()
    monkeypatch.setattr(
        UnquantizedFusedMoEMethod, "process_weights_after_loading", setup
    )
    expected = {}
    for name in ("w13", "w2"):
        weight = getattr(layer, f"{name}_weight")
        values = (
            torch.linspace(-2, 2, weight.numel())
            .reshape(weight.shape)
            .to(torch.float8_e4m3fn)
        )
        weight.weight_loader(weight, values)
        assert weight.dtype == torch.bfloat16
        scales = getattr(layer, f"{name}_{method.weight_scale_name}")
        scales.data.copy_(torch.arange(scales.numel()).reshape(scales.shape) / 8 + 0.25)
        rows, cols = weight.shape[-2:]
        if block:
            expanded = scales.repeat_interleave(4, -2).repeat_interleave(4, -1)[
                :, :rows, :cols
            ]
        else:
            expanded = scales.reshape(2, -1, 1).repeat_interleave(
                rows // (2 if name == "w13" else 1), -2
            )
        expected[name] = (values.float() * expanded).bfloat16()
    method.process_weights_after_loading(layer)
    for name, values in expected.items():
        torch.testing.assert_close(
            getattr(layer, f"{name}_weight"), values, rtol=0, atol=0
        )
    setup.assert_called_once_with(layer)
