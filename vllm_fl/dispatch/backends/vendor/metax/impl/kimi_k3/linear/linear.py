"""K3 shape dispatch registered by the plugin; FlagGems remains independent."""

from math import prod

import torch
import flag_gems
from triton.runtime.jit import JITFunction
from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.ops.linear import linear as native_linear, linear_kernel
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.linear.projections import plan, try_run

# Prefill fixed tilings reuse this canonical kernel and keep one registered entry.
def _k3_raw_prefill_linear():
    kernel = linear_kernel
    for _ in range(5):
        if isinstance(kernel, JITFunction):
            return kernel
        kernel = kernel.fn
    raise RuntimeError("Unexpected linear launch-wrapper depth")


_K3_PREFILL_LINEAR_KERNEL = _k3_raw_prefill_linear()
_K3_MM_BATCHES = (1, 2, 4, 8, 16, 32, 64)


def _k3_mm_shape(m, n, k):
    return (
        (n, k) == (2112, 7168) and (m in _K3_MM_BATCHES or m in (3553, 4096))
    ) or (m == 64 and (n, k) == (896, 7168))


def _k3_mm_eligible(x, weight, bias, n, k):
    return (
        bias is None
        and x.device == weight.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and weight.shape == (n, k)
    )


def linear(input, weight, bias=None, *, tp8_compute=True):
    """Select K3 launches; delegate the ordinary transformation to FlagGems."""
    m, k, n = prod(input.shape[:-1]), input.shape[-1], weight.shape[0]
    local_device = runtime.device.vendor_name == "metax" and input.is_cuda
    selected = local_device and tp8_compute and plan(m, n, k) is not None
    mm_shape = local_device and _k3_mm_shape(m, n, k)
    if not selected and not mm_shape:
        return native_linear(input, weight, bias)

    input_flat = input.view(m, k)
    weight = weight.contiguous()
    if selected:
        with torch_device_fn.device(input.device):
            output = try_run(input_flat, weight, bias, _K3_PREFILL_LINEAR_KERNEL)
        if output is not None:
            return output.view(*input.shape[:-1], n)
    if mm_shape and _k3_mm_eligible(input_flat, weight, bias, n, k):
        output = torch.empty((m, n), device=input.device, dtype=input.dtype)
        with torch_device_fn.device(input.device):
            flag_gems.mm_out(input_flat, weight.t(), out=output)
        return output.view(*input.shape[:-1], n)
    return native_linear(input, weight, bias)
