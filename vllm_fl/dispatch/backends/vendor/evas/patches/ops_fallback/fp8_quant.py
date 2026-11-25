# Copyright (c) 2026 BAAI. All rights reserved.
"""Monkey-patch the vLLM fp8 activation-quant entry points on LPU.

Two rebinds make the upstream MoE activation-quant flow yield the LPU container
contract — uint8 E4M3 bytes plus an fp32 scale — with zero forks in
``TritonExpertsFL.apply``: ``vllm._custom_ops.scaled_fp8_quant`` (per-token
uint8 E4M3 + fp32 scale, W8A16 no-op, static raise) and the block branch of
``per_token_group_quant_fp8`` once a ``block_shape`` is declared.  The evas
experts fast path quantizes inside its own kernel, so the census of runs that
reach these rebinds lives in the op pin.

"""

from __future__ import annotations

import functools
from typing import Callable

import torch


def apply_lpu_fp8_quant_patch() -> None:
    """Replace ``vllm._custom_ops.scaled_fp8_quant`` with the LPU pure-PyTorch
    implementation.  Idempotent — safe to call multiple times.
    """
    from vllm import _custom_ops as ops

    if getattr(ops, "_lpu_scaled_fp8_quant_patched", False):
        return

    # Platform-local pure-PyTorch encoder (vendor/evas/local/): _evas/fused is
    # the core-op mirror surface and must not carry non-mirroring files, so the
    # deep flag_gems import is gone.
    from ...local.e4m3 import quant_act_per_token

    def _scaled_fp8_quant_lpu(
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
        num_token_padding: int | None = None,
        scale_ub: torch.Tensor | None = None,
        use_per_token_if_dynamic: bool = False,
        output: torch.Tensor | None = None,
        group_shape: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert input.ndim == 2
        if scale is not None:
            raise NotImplementedError(
                "scaled_fp8_quant static scaling is unreachable on LPU "
                "(DSV4 uses dynamic per-token activation)"
            )
        if not use_per_token_if_dynamic:
            # W8A16: A stays bf16, no scale — upstream would emit a per-tensor
            # fp8 quant, but LPU has no float8 compute, so "unquantized" is the
            # only meaningful semantics (the evas op self-detects W8A16 from
            # bf16 A + A_scale None).
            return input, None
        # Per-token dynamic: quant_act_per_token == upstream semantics
        # (amax/448 per token, rounded+clamped E4M3, uint8 bytes).
        assert num_token_padding is None and output is None, (
            "LPU scaled_fp8_quant does not support padding/output buffers "
            "(unused in the MoE flow)"
        )
        q, s = quant_act_per_token(input)
        return q, s

    ops.scaled_fp8_quant = _scaled_fp8_quant_lpu
    # Latch ONLY after both rebinds have succeeded.  The latch is the sole
    # reason a retry skips, so setting it before the group-quant redirect would
    # make a half-applied patch unrepairable: with a block_shape declared the
    # MoE flow then always takes the fp8 branch, which would still be native
    # and would die on a platform with no fp8 cast -- and it dies deep inside
    # the quantizer instead of reporting the import error that caused it.
    _redirect_group_quant()
    ops._lpu_scaled_fp8_quant_patched = True  # type: ignore[attr-defined]


def _redirect_group_quant() -> None:
    """Point the MoE block-quant at the LPU encoder.

    ``_fp8_quantize`` takes the ``per_token_group_quant_fp8`` branch whenever a
    ``block_shape`` and an fp8 quant dtype are both declared, and the LPU fp8 MoE
    config declares both (``[128, 128]``, both target checkpoints'
    ``weight_block_size``).  On LPU the two activations that branch quantizes are
    both the fork's own work, not the prepare stage's: the MoE declares
    ``expects_unquantized_inputs``, so ``triton_experts_fl.py`` quantizes the
    first activation in its fall-through body and the second one (the activation
    output) right after the fused activation.  Without this redirect those calls
    would run vLLM's CUDA/Triton body, on a platform that has no fp8 cast.
    Idempotent.

    Scoped to this ONE module attribute on purpose: it is the binding that
    ``vllm.model_executor.layers.fused_moe.utils`` imported with ``from``, which
    is exactly the call the MoE flow makes -- ``_fp8_quantize`` and
    ``moe_kernel_quantize_input`` both live in that module and resolve the name
    as a module global, so the forks that ``from``-import those functions inherit
    the redirect without touching it.  ``deepseek_v2``, the GLM DSA indexer,
    ``input_quant_fp8`` and ``deep_gemm_moe`` each bind their own copy and need
    the native fp8 contract, so rebinding the definition in ``fp8_utils`` (or any
    broader site) would break them.

    What is installed is the adopted op wrapped by :func:`_as_uint8_container`:
    the op itself is shared with ``flag_gems.<op>`` callers, which take the
    generic native-fp8 contract, so the container the consumer here asserts on
    is settled at this call site rather than inside the op.
    """
    from vllm.model_executor.layers.fused_moe import utils as fused_moe_utils

    if getattr(fused_moe_utils, "_lpu_group_quant_patched", False):
        return

    # Vendor-qualified on purpose: this is the FlagGems _evas twin, the copy
    # whose own adopt() call owns the sha256 pin for the vendored Triton
    # kernel. There is deliberately no fallback here: a FlagGems without _evas
    # must fail loudly rather than silently take another binding (02doc 12.2).
    from flag_gems.runtime.backend._evas.ops.per_token_group_quant_fp8 import (
        per_token_group_quant_fp8,
    )

    fused_moe_utils.per_token_group_quant_fp8 = _as_uint8_container(
        per_token_group_quant_fp8
    )
    fused_moe_utils._lpu_group_quant_patched = True  # type: ignore[attr-defined]


def _as_uint8_container(group_quant: Callable) -> Callable:
    """Wrap an adopted group-quant so the MoE always sees the uint8 container.

    The op has two arms and they do not agree on the container: the pure
    PyTorch twin returns the LPU uint8 E4M3 bytes (LPU aten has no float8 cast,
    so it cannot build the native tensor), while the adopted evas kernel returns
    native ``float8_e4m3fn`` -- the generic contract, and the reason the op
    cannot simply return uint8 itself.  The element bytes are identical either
    way, only the dtype tag differs, so the kernel's container is one
    byte-preserving view away from the LPU activation container: uint8 E4M3
    bytes, which is what the fork carries for LPU E4M3 activations
    (``triton_experts_fl.py``) and what the evas fused experts kernel keys its
    W8A8 self-detect on (``fused_moe.py`` in the FlagGems evas backend).

    Same reinterpretation as ``_o_proj`` runs in the other direction
    (impl/attention/dsv4_attention.py), and a no-op on the twin.
    """

    @functools.wraps(group_quant)
    def _quant(x: torch.Tensor, group_size: int, *args, **kwargs):
        q, scales = group_quant(x, group_size, *args, **kwargs)
        if q.dtype != torch.uint8:
            q = q.view(torch.uint8)
        return q, scales

    return _quant
