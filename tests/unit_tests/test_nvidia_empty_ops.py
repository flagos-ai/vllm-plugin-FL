# Copyright (c) 2026 BAAI. All rights reserved.

"""Exercise capabilities and real extension dispatch on NVIDIA EMPTY builds."""

from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def nvidia_empty_runtime():
    if not torch.cuda.is_available() or version("vllm") != "0.32.0+empty":
        pytest.skip("requires the NVIDIA vLLM 0.32 EMPTY stack")
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "nvidia":
        pytest.skip("requires the NVIDIA FL platform")


@pytest.mark.parametrize(
    "name",
    [
        "cutlass_scaled_mm_supports_fp8",
        "cutlass_scaled_mm_supports_block_fp8",
        "cutlass_group_gemm_supported",
        "cutlass_scaled_mm_supports_fp4",
    ],
)
@pytest.mark.parametrize("capability", [90, 100])
def test_absent_cutlass_kernels_are_not_advertised(name, capability):
    assert getattr(torch.ops._C, name)(capability) is False


def test_weak_reference_uses_plugin_cuda_extension():
    import vllm_fl._C  # noqa: F401

    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "_C::weak_ref_tensor", "CUDA"
    )
    original = torch.arange(12, device="cuda").reshape(3, 4)
    weak = torch.ops._C.weak_ref_tensor(original)
    assert weak is not original
    assert weak.data_ptr() == original.data_ptr()
    assert weak.stride() == original.stride()
    original.add_(2)
    torch.testing.assert_close(weak, original)


def test_missing_uva_does_not_return_a_cpu_tensor_as_cuda():
    from vllm.platforms import current_platform
    from vllm.utils.platform_utils import is_uva_available

    assert not current_platform.is_pin_memory_available()
    assert not is_uva_available()
    with pytest.raises(RuntimeError, match="NonUvaBuffer"):
        torch.ops._C.get_cuda_view_from_cpu_tensor(torch.ones(4))


def test_optional_native_schemas_do_not_suppress_real_fallbacks():
    from vllm_fl.ops._C_ops_registry import _OPTIONAL_NATIVE_ONLY_OPS

    for name in _OPTIONAL_NATIVE_ONLY_OPS:
        assert not hasattr(torch.ops._C, name), name


def test_eager_ir_defaults_use_executable_rms_norm():
    from vllm import ir
    from vllm.config.compilation import CompilationMode

    from vllm_fl.nvidia_platform import NvidiaPlatformFL

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            backend="inductor", mode=CompilationMode.NONE
        )
    )
    priorities = NvidiaPlatformFL.get_default_ir_op_priority(config)
    assert "vllm_c" not in priorities.rms_norm
    assert "vllm_c" not in priorities.fused_add_rms_norm
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    with priorities.set_priority():
        result = ir.ops.rms_norm(x, weight, 1e-6, None)
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = (normalized * weight.float()).to(x.dtype)
    torch.testing.assert_close(result, expected, atol=0.02, rtol=0.02)
