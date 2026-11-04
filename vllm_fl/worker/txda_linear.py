# Copyright (c) 2026 BAAI. All rights reserved.
"""Use the audited TXDA BF16 GEMM without broad ATen replacement."""

import functools
import importlib
import os

import torch


def _make_txda_gemm(original, vendor_mm, *, device_type="txda", on_first_use=None):
    reported = False

    @functools.wraps(original)
    def unquantized_gemm(layer, x, weight, bias=None):
        nonlocal reported
        # Keep other dtypes, biased layers and autograd on their original path.
        # The vendor FP32 kernel does not meet the precision gate.
        supported = (
            x.device.type == device_type
            and weight.device == x.device
            and x.dtype == weight.dtype == torch.bfloat16
            and x.layout == weight.layout == torch.strided
            and x.ndim >= 1
            and weight.ndim == 2
            and x.shape[-1] == weight.shape[1]
            and x.numel() > 0
            and weight.numel() > 0
            and bias is None
            and not (
                torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
            )
        )
        if not supported:
            return original(layer, x, weight, bias)
        output = vendor_mm(x.reshape(-1, weight.shape[1]), weight.t())
        if output.device != x.device or output.dtype != x.dtype:
            raise RuntimeError("TXDA vendor GEMM returned an incorrect device or dtype")
        output = output.reshape(*x.shape[:-1], weight.shape[0])
        if not reported and on_first_use is not None:
            on_first_use(x, weight)
            reported = True
        return output

    unquantized_gemm._fl_txda_bf16_gemm = True
    return unquantized_gemm


def patch_txda_unquantized_gemm(rank=None):
    """Install the BF16-only provider before V2 model layer construction."""
    if os.getenv("FL_TXDA_USE_VENDOR_GEMM", "1") == "0":
        return False
    from vllm.platforms import current_platform

    if current_platform.device_type != "txda":
        return False
    from vllm.logger import init_logger
    from vllm.model_executor.layers import utils

    original = utils.default_unquantized_gemm
    if getattr(original, "_fl_txda_bf16_gemm", False):
        return True
    # Select the vendor implementation explicitly. The generic linear kernel
    # and its ATen registrations have not passed this adapter's hardware gate.
    vendor_mm = importlib.import_module(
        "flag_gems.runtime.backend._tsingmicro.ops.mm"
    ).mm
    provider = f"{vendor_mm.__module__}.{vendor_mm.__name__}"
    logger = init_logger("vllm.fl.txda.linear")

    def report(x, weight):
        logger.info(
            "TXDA BF16 GEMM active: rank=%s provider=%s input=%s weight=%s "
            "device=%s threads=%d",
            rank,
            provider,
            tuple(x.shape),
            tuple(weight.shape),
            x.device,
            torch.get_num_threads(),
        )

    utils.default_unquantized_gemm = _make_txda_gemm(
        original, vendor_mm, on_first_use=report
    )
    logger.info("TXDA BF16 GEMM installed: rank=%s provider=%s", rank, provider)
    return True
