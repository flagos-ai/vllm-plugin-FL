"""ARM CPU W8A16 Linear via torch-native fused int8-weight GEMM.

Uses ``torch._weight_int8pack_mm`` (the gpt-fast / llama.cpp style CPU int8
kernel): it streams the int8 weight and dequantizes on the fly inside the GEMM,
never materializing a bf16 weight.  Decode is memory-bound, so halving the
weight-read bandwidth gives a large speedup over torchao's dequant+bf16-mm path
(which materializes bf16 and reads full bandwidth).

Pure official torch op — no third-party kernel, no TLE, no flag_gems/triton.
Weights are quantized online (per-row symmetric int8) when the model loads, so
no torchao checkpoint is required; the original BF16 model is used directly.

Installed by register_model() when ``FL_CPU_INT8=1`` and
``FL_CPU_INT8_BACKEND=torchpack``, hooking vLLM's
``dispatch_cpu_unquantized_gemm`` (same mechanism as the other quantized CPU
backends).  Runs under CpuPlatformFL.
"""
import logging
import os

import torch

from vllm_fl.ops.cpu_quant_linear import install_cpu_quantized_linear

logger = logging.getLogger("vllm_fl.cpu_int8_pack")
INCLUDE_LM_HEAD = os.environ.get("FL_INT8_LMHEAD", "0") == "1"
STRICT = os.environ.get("FL_CPU_INT8_STRICT", "1") != "0"


def _quantize_int8(weight):
    """[N,K] bf16 weight -> (int8 [N,K] row-major, bf16 scale [N]), per-row symmetric."""
    w = weight.detach().to(torch.float32)
    scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    qw = (w / scale[:, None]).round().clamp(-128, 127).to(torch.int8).contiguous()
    return qw, scale.to(torch.bfloat16).contiguous()


def _make_cpu_linear(qweight, scale, N, K):
    def cpu_linear(x, weight, bias):
        shape = x.shape
        xb = x.to(torch.bfloat16).reshape(-1, K).contiguous()
        out = torch._weight_int8pack_mm(xb, qweight, scale)  # fused, [M,N] bf16
        out = out.reshape(*shape[:-1], N)
        return out + bias.to(out.dtype) if bias is not None else out

    return cpu_linear


def _prepare_linear(weight):
    N, K = weight.shape
    qweight, scale = _quantize_int8(weight.to(torch.bfloat16))
    return _make_cpu_linear(qweight, scale, N, K)


def enable_int8(verbose=True):
    installed = install_cpu_quantized_linear(
        backend="ARM W8A16 torchpack",
        prepare_linear=_prepare_linear,
        supports_shape=lambda n, k: k % 4 == 0,
        include_lm_head=INCLUDE_LM_HEAD,
        strict=STRICT,
        logger=logger,
    )
    if installed and verbose:
        logger.info(
            "[vllm_fl] ARM int8 W8A16 enabled (torch _weight_int8pack_mm, fused)"
        )
