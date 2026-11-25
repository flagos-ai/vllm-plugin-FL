# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU (evas) FP8 quantization-method overrides for any FP8 checkpoint.

``Fp8LinearMethodLPU`` / ``Fp8MoEMethodLPU`` mirror the upstream fp8 methods
minus their CUDA-only bits: weights stay native ``float8_e4m3fn`` with fp32
block scales; ``apply`` dequants per call (W8A16, default) or runs the evas
Triton fp8 GEMM under ``VLLM_FL_FP8_W8A8_GEMM=1``.  The weight patch
(``dsv4_weights``) rebinds ``get_quant_method`` to them for DSV4 and plain
``Fp8Config``; else GLM-5.2-FP8 dies in ``choose_scaled_mm_linear_kernel``,
whose candidates are CUDA/XPU-only.

"""

from __future__ import annotations

import os
from collections import OrderedDict

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.fp8 import (
    Fp8LinearMethod,
    Fp8MoEMethod,
    validate_fp8_block_shape,
)
from vllm.model_executor.utils import set_weight_attrs


def wo_a_grouped_views(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    n_groups: int,
    o_lora_rank: int,
    in_width: int,
    block_n: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """View the wo_a FP8 weight + block scale as the grouped-einsum operands.

    ``create_weights`` (and the checkpoint) store ``wo_a.weight`` as 2-D
    ``[out, in]`` = ``[n_groups * o_lora_rank, in_width]``, with
    ``weight_scale_inv`` as ``[out // block_n, in // block_k]`` — the FLOOR
    form is only sound while ``o_lora_rank``/``in_width`` stay block-aligned
    (DSV4 wo_a out=7168, GLM o_proj out=6144, both %128==0); a ragged wo_a
    would need the ceil + expand treatment now used in
    ``_dequant_block_scale``.  The grouped
    einsum wants 3-D operands:

    - upstream ``deepgemm_post_process_fp8_weight_block`` reshapes the very
      same bytes with ``wq.view(g, r, d)`` where ``g = bmm_batch_size``
      (``n_local_groups``), ``r = wq.size(0) // g`` (``o_lora_rank``) and
      ``d = wq.size(1)`` (``in_width``);
    - ``flag_gems.fp8_einsum`` consumes ``y`` as ``[h, d_out, d_contract]``
      with the contraction over the **last** dim (``einsum("bmk,bnk->bmn")``).

    So the grouped view is a plain row-major reshape of the ``[out, in]``
    storage -- **no transpose**.  Transposing makes the contracted axis the
    output axis, which scrambles the weight into a tensor whose row index
    depends on the contracted index; the scramble differs with the shard
    width, so a TP1 and a TP4 run disagree on the same checkpoint (it shows
    up as a first-token divergence).
    """
    y = weight.reshape(n_groups, o_lora_rank, in_width)
    ys = weight_scale_inv.reshape(
        n_groups, o_lora_rank // block_n, in_width // block_k
    )
    return y, ys


# The E4M3 bit-unpack (``dequant_e4m3``) is a pure-PyTorch unpack that holds
# ~12 fp32 temporaries of the input shape live at once (sign/exponent/mantissa
# splits, three ``torch.where`` results, the zero template), so one whole-weight
# dequant costs ~48 bytes per fp8 byte.  That is fine for the FP8 checkpoints
# this path was written for (DSV4/GLM attention Linears top out at 32 MiB) but
# not for a dense 27B MLP: ``gate_up_proj`` [34816, 5120] is 178 MB of fp8 and
# peaks near 9 GiB, which no 48 GiB card can absorb next to a resident 28 GiB
# FP8 model.  Dequantize a row band at a time instead — the transient then
# scales with the band, and the bf16 output is filled in place.
#   The banding is only a win while ``w_u8[r0:r1]`` stays a cheap view: LPU
# materializes the whole storage for any sub-read of a single buffer above
# 1.5 GiB, so a weight that large would pay the full buffer once per band.
# Keep this for weights below that line; above it, bands have to copy rather
# than slice.
#   0 disables chunking (one whole-tensor dequant, the pre-existing behaviour).
_FP8_DEQUANT_CHUNK_ROWS = int(os.environ.get("VLLM_FL_FP8_DEQUANT_CHUNK_ROWS", "2048"))


def _dequant_block_scale(
    w_u8: torch.Tensor, scale: torch.Tensor, block: int = 128
) -> torch.Tensor:
    """Dequant E4M3 bytes (uint8 or native fp8) + E8M0 block scale to bf16.

    Mirrors the evas MoE W8A16 path (``_evas/fused/fused_moe.py``
    ``dequant_e4m3`` + block-scale broadcast): E4M3 grid * 2^(s-127) is exactly
    bf16-representable, so the result matches the BF16 baseline losslessly.

    Weights taller than the band height are processed a row band at a time to
    bound the unpack transient (see the knob's comment). The band height is
    ``VLLM_FL_FP8_DEQUANT_CHUNK_ROWS`` rounded down to a multiple of ``block``,
    lifted to one block if the knob is below that, and chunking is off at 0.
    The arithmetic per band is identical, so the result is bit-exact either way.

    Args:
        w_u8: [out, in] E4M3 bytes — uint8, or native ``float8_e4m3fn``
            (viewed to uint8 inside ``dequant_e4m3``).
        scale: [ceil(out/128), ceil(in/128)] fp32 (E8M0 decoded to float) —
            upstream block scales are ceil-div (a merged gemm's last shard may
            be non-aligned, e.g. GLM ``fused_qkv_a_proj`` 2624 rows).
        block: block size (128).
    Returns:
        [out, in] bf16 dequantized weight.
    """
    out = w_u8.shape[0]
    rows = _FP8_DEQUANT_CHUNK_ROWS
    # Bands must start on a scale-block boundary; a value below one block is
    # lifted to one block rather than silently disabling the chunking.
    rows = max(block, (rows // block) * block) if rows > 0 else 0
    if rows <= 0 or out <= rows:
        return _dequant_block_scale_band(w_u8, scale, block)
    result = torch.empty(
        w_u8.shape, dtype=torch.bfloat16, device=w_u8.device
    )
    for r0 in range(0, out, rows):
        r1 = min(r0 + rows, out)
        # r0 is a multiple of block, so the scale rows line up without any
        # re-indexing; the ragged tail band keeps the ceil-div scale slice.
        result[r0:r1] = _dequant_block_scale_band(
            w_u8[r0:r1], scale[r0 // block:(r1 + block - 1) // block], block
        )
    return result


def _dequant_block_scale_band(
    w_u8: torch.Tensor, scale: torch.Tensor, block: int
) -> torch.Tensor:
    """Dequant one row band of ``w_u8``; the original whole-tensor math."""
    out, inp = w_u8.shape
    n_out = (out + block - 1) // block
    n_in = (inp + block - 1) // block
    assert scale.shape == (n_out, n_in), (
        f"scale {scale.shape} != {(n_out, n_in)}"
    )
    w_deq = _dequant_e4m3(w_u8)  # [out, in] fp32
    if out % block == 0 and inp % block == 0:
        w_deq = (
            w_deq.view(n_out, block, n_in, block)
            * scale[:, None, :, None]
        ).reshape(out, inp)
    else:
        # Partial last block (rare: merged non-aligned total). Expand the
        # block scale to elementwise and multiply; the result is cached by
        # _dequant_linear_b_bf16, so this runs once per weight.
        s = (
            scale.repeat_interleave(block, dim=0)[:out]
            .repeat_interleave(block, dim=1)[:, :inp]
        )
        w_deq = w_deq * s
    return w_deq.to(torch.bfloat16)


def _dequant_e4m3(u8: torch.Tensor) -> torch.Tensor:
    """E4M3 bit-unpack (pure-PyTorch, LPU-safe)."""
    # Platform-local encoder (vendor/evas/local/); _evas/fused is the core-op
    # mirror surface and must not carry non-mirroring files (02doc §12.5). The
    # old ``flag_gems.fused._e4m3`` fallback was a dead path -- core has no
    # such module.
    from ..local.e4m3 import dequant_e4m3
    return dequant_e4m3(u8)


# FP8 attention Linear weights (fused_wqa_wkv / wq_b / wo_b — the ones routed
# through ``Fp8LinearMethodLPU``; wo_a goes through the flaggems einsum cache
# instead) are constant for an engine's lifetime, but the naive ``apply``
# dequantized the full weight (E4M3 + E8M0 128x128 block scale -> bf16)
# on EVERY call.  On LPU that dequant is the entire per-call cost — and for
# wq_b [1024, 32768] it is also a 384 MiB fp32 spike that, combined with the
# resident model + KV caches, OOMs the allocator (strict-mode e2e:
# "Tried to allocate 384.00 MiB ... LPU malloc failed", process at 20.08 GiB).
#
# So cache the dequantized bf16 weight per (layer.weight, layer.scale)
# identity, keyed by (data_ptr, shape, stride) of the fp8 weight AND the
# scale's data_ptr — static for the engine's lifetime, so a plain dict is safe
# (same data_ptr-alias/strong-ref discipline as the wo_a cache in
# fp8_einsum.py and the MoE per-expert cache in fused_moe.py).  Entries are
# LRU-evicted beyond a byte budget (default 2048 MB, knob
# VLLM_FL_FP8_LINEAR_CACHE_MB).  Each layer's weight is a distinct tensor, so
# the key naturally separates the 4 attention linears.  The budget must cover
# the WHOLE working set: decode sweeps layers 0..42 every step, so a cyclic
# scan against an LRU that is one entry short runs at ~0% hits.  2048 MB holds
# the whole 236-entry working set (1.75 GiB); 1024 MB holds only 135 of them
# and measures 0 hits out of 2124 lookups (attention scope -11.8%, tpot
# 29.9 -> 27.6 s between the two budgets).
_FP8_LINEAR_CACHE: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()

# Hit/miss mix of the bf16 linear-weight cache (dumped at process exit like
# the fused_moe _B_STAT): decode sweeps all 43 layers in order, so a budget
# smaller than that working set gives a cyclic-scan LRU ~0% steady-state hit
# and the "cached" dequant runs on every call anyway — the dump is the only
# field way to tell which of the two regimes a run landed in.
_FP8_LINEAR_STAT = {"hit": 0, "miss": 0, "evict": 0}


def _dump_fp8_linear_stat():
    # entries/bytes/budget at exit distinguish the 0-hit regimes: evict>0
    # with entries>0 = stable keys over budget (grow the budget), evict=0
    # with entries~0 = keys never match (fix keying instead).
    if not any(_FP8_LINEAR_STAT.values()) and not _FP8_LINEAR_CACHE:
        return
    total = sum(v[2].numel() * v[2].element_size()
                for v in _FP8_LINEAR_CACHE.values())
    budget = int(os.environ.get("VLLM_FL_FP8_LINEAR_CACHE_MB", "2048")) << 20
    print(
        f"[fp8-linear-stat] {_FP8_LINEAR_STAT}"
        f" entries={len(_FP8_LINEAR_CACHE)} bytes={total} budget={budget}",
        flush=True,
    )


try:
    import atexit as _atexit_fp8

    _atexit_fp8.register(_dump_fp8_linear_stat)
except Exception:  # pragma: no cover
    pass


def _dequant_linear_b_bf16(
    w_u8: torch.Tensor, scale: torch.Tensor, block: int = 128
) -> torch.Tensor:
    """Dequantize one FP8 attention Linear weight to bf16 [out, in], cached.

    Same math as ``_dequant_block_scale`` (bit-exact).  Keyed on w_u8's
    data_ptr/shape/stride AND the scale's data_ptr so a re-quantized weight
    with the same layout still misses; the entry holds strong refs to w_u8 +
    scale (a freed + reallocated same-shape weight can never alias a stale
    entry).  LRU-evicted beyond ``VLLM_FL_FP8_LINEAR_CACHE_MB`` (see module
    comment).
    """
    key = (w_u8.data_ptr(), w_u8.shape, w_u8.stride(),
           scale.data_ptr(), scale.shape)
    hit = _FP8_LINEAR_CACHE.get(key)
    if hit is not None:
        _FP8_LINEAR_CACHE.move_to_end(key)
        _FP8_LINEAR_STAT["hit"] += 1
        return hit[2]
    w_bf16 = _dequant_block_scale(w_u8, scale, block)
    _FP8_LINEAR_CACHE[key] = (w_u8, scale, w_bf16)
    _FP8_LINEAR_CACHE.move_to_end(key)
    _FP8_LINEAR_STAT["miss"] += 1
    # LRU-evict until under the byte budget (always keep >= 1 entry).
    budget = int(os.environ.get("VLLM_FL_FP8_LINEAR_CACHE_MB", "2048")) << 20
    while len(_FP8_LINEAR_CACHE) > 1:
        total = sum(v[2].numel() * v[2].element_size()
                    for v in _FP8_LINEAR_CACHE.values())
        if total <= budget:
            break
        _FP8_LINEAR_CACHE.popitem(last=False)
        _FP8_LINEAR_STAT["evict"] += 1
    return w_bf16


# W8A8 GEMM: quantise the activation per token group and let the evas Triton
# fp8 kernel contract fp8 x fp8 (the same block-scaled fp8 ``tl.dot`` the DSV4
# attention wo_a already runs), instead of dequantising the weight per call.
# Off by default: it changes a shipped numerical path, so it stays a knob until
# a model-level A/B says the trade is worth it.  Per-shape cost and the
# correctness criterion live with the op (``_evas/ops/w8a8_block_fp8_matmul.py``
# module docstring).
_W8A8_STAT = {"gemm": 0, "fallback": 0}


def _w8a8_gemm_enabled() -> bool:
    return os.environ.get("VLLM_FL_FP8_W8A8_GEMM", "0") == "1"


def _dump_w8a8_stat():
    if not (_W8A8_STAT["gemm"] or _W8A8_STAT["fallback"]):
        return
    print(f"[fp8-w8a8-stat] {_W8A8_STAT}", flush=True)


try:
    import atexit as _atexit_w8a8

    _atexit_w8a8.register(_dump_w8a8_stat)
except Exception:  # pragma: no cover
    pass


_W8A8_OP_MISSING_WARNED = {"done": False}


def _w8a8_ops():
    """Resolve the evas fp8 GEMM pieces, or None when this FlagGems lacks them.

    ``VLLM_FL_FP8_W8A8_GEMM`` is a performance dial and the W8A16 path is
    numerically complete without it, so a FlagGems tree that predates the op
    must not turn the first FP8 Linear into an ImportError deep inside
    ``apply``.  The warning is printed once per process; the atexit
    ``[fp8-w8a8-stat]`` line then reads all-fallback, so a run that lost the
    GEMM is identifiable from its own log rather than looking like a slow
    W8A8 measurement.
    """
    # Activation quant is platform-local (vendor/evas/local/, 02doc §12.5);
    # the feature probe covers only the flag_gems op surface.
    from ..local.e4m3 import quant_act_per_block
    try:
        from flag_gems import w8a8_block_fp8_matmul
        # kernel_supported has no public binding, so this probe needs the evas
        # path; a FlagGems without _evas lands in the handler below and the
        # fp8 GEMM keeps W8A16.
        from flag_gems.runtime.backend._evas.ops.w8a8_block_fp8_matmul import (
            kernel_supported,
        )
    except ImportError as exc:
        if not _W8A8_OP_MISSING_WARNED["done"]:
            _W8A8_OP_MISSING_WARNED["done"] = True
            print(
                f"[fp8-w8a8] fp8 GEMM op unavailable ({exc}); "
                "VLLM_FL_FP8_W8A8_GEMM=1 keeps W8A16 for this run",
                flush=True,
            )
        return None
    return w8a8_block_fp8_matmul, quant_act_per_block, kernel_supported


def _fp8_linear_w8a8(x, layer, block_size):
    """Run one FP8 Linear through the evas Triton fp8 GEMM, or None.

    None means "keep the W8A16 path" and is returned unless every constraint
    the kernel asserts holds -- checked with the op's own ``kernel_supported``
    so there is one predicate, not two.  The check has to happen before the
    call: the op answers an unsupported shape with a pure-torch dequant, which
    on a large weight is strictly worse than the W8A16 path this helper is
    meant to beat (same arithmetic, plus an fp32 expansion the W8A16 path does
    not have).  So a layer the kernel cannot take declines here rather than
    silently paying that.
    """
    if block_size is None or list(block_size) != [128, 128]:
        return None
    weight = layer.weight
    if weight.dim() != 2:
        return None
    out_features, in_features = weight.shape
    if out_features % 128 or in_features % 128:
        return None
    if x.dim() not in (2, 3) or x.shape[-1] != in_features:
        return None
    if x.dtype != torch.bfloat16:
        return None

    ops = _w8a8_ops()
    if ops is None:
        return None
    w8a8_block_fp8_matmul, quant_act_per_block, kernel_supported = ops

    x = x if x.is_contiguous() else x.contiguous()
    x_u8, x_s = quant_act_per_block(x, 128)
    scale = layer.weight_scale_inv
    if not kernel_supported(x_u8, weight, x_s, scale, [128, 128]):
        if os.environ.get("VLLM_FL_FP8_W8A8_GEMM_DEBUG") == "1":
            print(
                f"[fp8-w8a8-skip] {tuple(weight.shape)} "
                f"weight_contig={weight.is_contiguous()} "
                f"w_stride={tuple(weight.stride())} "
                f"scale={tuple(scale.shape)}",
                flush=True,
            )
        return None
    return w8a8_block_fp8_matmul(
        x_u8, weight, x_s, scale, [128, 128], output_dtype=torch.bfloat16,
    )


class Fp8LinearMethodLPU(Fp8LinearMethod):
    """LPU FP8 Linear method: native float8_e4m3fn bytes + fp32 block scale.

    Subclass of the upstream ``Fp8LinearMethod``; overrides only the pieces
    that are CUDA-bound or that require ``torch.float8`` compute (unsupported
    on LPU): ``create_weights`` (native fp8 weight param, no
    ``init_fp8_linear_kernel``),
    ``process_weights_after_loading`` (no-op), ``apply`` (pure-PyTorch
    dequant + ``F.linear``). The layer still exposes ``weight`` and
    ``weight_scale_inv`` so the generic MLA wrapper's O-projection
    (``deepseek_v4_attention.py`` ``wo_a.weight`` / ``wo_a.weight_scale_inv``)
    reads the same attributes.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size

        if self.block_quant:
            assert self.weight_block_size is not None
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        # Native float8_e4m3fn E4M3 bytes (byte-identical to uint8; LPU has no
        # fp8 compute/cast, so dequant always views to uint8 first via
        # dequant_e4m3 — the native fp8 storage keeps the dtype contract honest
        # for the upcoming real Triton FP8 ops). Upstream helper constructs the
        # identical ModelWeightParameter (empty, float8_e4m3fn, input=1, out=0).
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            create_fp8_weight_parameter,
        )

        layer.register_parameter(
            "weight",
            create_fp8_weight_parameter(
                output_size_per_partition, input_size_per_partition, weight_loader
            ),
        )

        # Scale params intentionally NOT upstream create_fp8_scale_parameter:
        # that helper inits with empty (here: ones) and uses a different
        # per-tensor layout (len(output_partition_sizes) with no input_dim vs.
        # here (out,) with input_dim=1/output_dim=0).  The BLOCK scale below
        # must nonetheless keep upstream's CEIL-div block shapes: vLLM's
        # MergedColumn loader narrows per shard with ceil(shard/128), and a
        # merged output whose total is not block-aligned (GLM fused_qkv_a_proj
        # = [q_lora_rank 2048, kv_lora+rope 576] -> 2624 = 20.5*128) needs
        # ceil(2624/128)=21 rows — floor gives 20 and the loader dies with
        # "start (16) + length (5) exceeds dimension size (20)" (GLM-5.2-FP8
        # TP16 run lpu-glm5-tp16-func2).  Note ragged merged partitions
        # pre-arm allow_fp8_block_shape_mismatch in ColumnParallelLinear, so
        # validate_fp8_block_shape is SKIPPED for them entirely (it does not
        # gate this); ceil is correct because it matches upstream
        # create_fp8_scale_parameter regardless.  Aligned outputs (all DSV4
        # linears) are identical either way.
        # BLOCK scale: fp32 E8M0 block scale (weight_scale_inv).
        if not self.block_quant:
            from vllm.model_executor.parameter import PerTensorScaleParameter

            scale = PerTensorScaleParameter(
                data=torch.ones(output_size_per_partition, dtype=torch.float32),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", scale)
        else:
            from vllm.model_executor.parameter import BlockQuantScaleParameter

            block_n, block_k = self.weight_block_size
            scale = BlockQuantScaleParameter(
                data=torch.ones(
                    (output_size_per_partition + block_n - 1) // block_n,
                    (input_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale_inv", scale)

        # No init_fp8_linear_kernel — no CUDA FP8 kernel on LPU.
        self.use_marlin = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Params are already the LPU-friendly native fp8 + fp32 block scale;
        # no kernel (re)formatting needed.
        return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _w8a8_gemm_enabled():
            out = _fp8_linear_w8a8(x, layer, self.weight_block_size)
            if out is not None:
                _W8A8_STAT["gemm"] += 1
                return out if bias is None else out + bias
            _W8A8_STAT["fallback"] += 1

        # Pure-PyTorch W8A16: dequant weight to bf16, then F.linear.  A stays
        # bf16 (W8A16).  The weight is constant for the engine's lifetime, so
        # the dequant is cached per (layer.weight, layer.scale) identity
        # (see _dequant_linear_b_bf16) — this removes the per-call 384 MiB
        # spike for wq_b and turns the dequant into a one-time cost.
        w_fp8 = layer.weight
        w_bf16 = _dequant_linear_b_bf16(
            w_fp8, layer.weight_scale_inv, self.weight_block_size[0]
        )
        return F.linear(x, w_bf16, bias)


def _alloc_expert_weight(shape, dtype, split_dim, double_die):
    """Allocate one fp8 expert weight or block scale in the kernel's layout.

    The evas fp8 MoE kernel splits the experts along the **feature** axis, not
    the expert axis: ``_moe_weight_shards`` reshapes w13 to ``[E, 2, I, H]``
    and w2 to ``[E, H, I]``, then requires the tensor to already be split at
    dim 2 resp. dim 1 -- the shards it derives are "die0 owns gate0|up0, die1
    owns gate1|up1" (w13) and a half-width N shard (w2).  A weight allocated
    with the device default (row split, ``split_dim=0``) fails that test on
    every forward of every MoE layer, and the kernel then re-lays it with
    ``clone_with_split_dim`` -- a whole-weight copy of several GiB per call.

    Allocating in the kernel's layout here makes that copy unnecessary.  The
    caller passes the *reshaped* logical shape and views it back down, so the
    parameter the framework sees is unchanged.

    ``empty_with_format`` forces a double-die allocation even while the device
    is in single-die mode, so it may only be used when double die is actually
    on; otherwise the plain allocation is the correct one.
    """
    if not double_die:
        return torch.empty(shape, dtype=dtype)
    return torch.ops.lpu.empty_with_format(
        shape, dtype=dtype, device="lpu", split_dim=split_dim
    )


class Fp8MoEMethodLPU(Fp8MoEMethod):
    """LPU FP8 MoE method: native float8_e4m3fn expert weights + fp32 block scales.

    Subclass of the upstream ``Fp8MoEMethod``; the param shapes match upstream
    block-quant exactly (``w13_weight``/``w2_weight`` ``[E, N, K]`` native fp8 +
    ``w13_weight_scale_inv``/``w2_weight_scale_inv`` ``[E, N/128, K/128]``
    fp32) — which is precisely the stash shape the evas FP8 MoE path
    (``_evas/fused/fused_moe.py``) consumes. Overrides:

    - ``create_weights`` — native fp8 weights.
    - ``process_weights_after_loading`` — no-op (no ``_setup_kernel``: no
      ``convert_to_fp8_moe_kernel_format`` CUDA shuffle, no ``make_fp8_moe_kernel``).
    - ``apply`` — forward through ``self.moe_kernel`` (``TritonExpertsFL``, built
      by ``_lpu_setup_moe_kernel`` with a real FP8 ``FusedMoEQuantConfig``: the
      per-layer block scales + activation contract live in the config,
      and the A-quant flows through the patched ``scaled_fp8_quant``), reusing
      the evas ``invoke_fused_moe_triton_kernel`` override (B dequant per-expert;
      the FP8-quantized A is dequantized too).
    """

    def __init__(self, quant_config, layer: torch.nn.Module):
        # Bypass upstream select_fp8_moe_backend (iterates CUDA kernel classes,
        # may raise on LPU). Set the fields upstream apply/process read; the
        # actual LPU kernel (TritonExpertsFL) is built by _lpu_setup_moe_kernel.
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
            FusedMoEMethodBase,
        )

        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )
        self.fp8_backend = None
        self.experts_cls = None
        self.moe_kernel = None
        self.moe_quant_config = None
        # NOTE: no `self.is_monolithic` — the base FusedMoEMethodBase defines it
        # as a read-only @property (settable attr raises AttributeError). It is
        # computed: when moe_kernel is None it reflects experts_cls.is_monolithic();
        # after _lpu_setup_moe_kernel sets moe_kernel (TritonExpertsFL,
        # non-monolithic OOT), it returns moe_kernel.is_monolithic == False, which
        # is exactly what the non-monolithic apply() path asserts.

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size

        assert self.quant_config.is_checkpoint_fp8_serialized
        # Native float8_e4m3fn E4M3 bytes (byte-identical to uint8; dequant
        # views to uint8 first via dequant_e4m3 — no fp8 compute/cast on LPU).
        params_dtype_fp8 = torch.float8_e4m3fn

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )

        import torch_lpu

        # Allocate in the evas fp8 MoE kernel's own feature-split layout (see
        # _alloc_expert_weight): a default-layout weight is re-laid per
        # forward, per layer, for the life of the engine.
        double_die = bool(torch_lpu.lpu.is_double_die_enabled())

        # WEIGHTS (native float8_e4m3fn E4M3 bytes).
        w13_weight = torch.nn.Parameter(
            _alloc_expert_weight(
                [num_experts, 2, intermediate_size_per_partition, hidden_size],
                params_dtype_fp8,
                2,
                double_die,
            ).view(num_experts, 2 * intermediate_size_per_partition, hidden_size),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            _alloc_expert_weight(
                [num_experts, hidden_size, intermediate_size_per_partition],
                params_dtype_fp8,
                1,
                double_die,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES (fp32 block scales).
        if not self.block_quant:
            # Per-tensor scales: not reachable on LPU (both fp8 stacks this
            # method serves -- DSV4-Flash fp8 and GLM-5.2-FP8 -- declare
            # weight_block_size [128, 128]), so no layout work is done here.
            w13_scale_data = torch.ones(num_experts, 2, dtype=torch.float32)
            w2_scale_data = torch.ones(num_experts, dtype=torch.float32)
        else:
            scale_i = (intermediate_size_per_partition + block_n - 1) // block_n
            scale_h = (hidden_size + block_k - 1) // block_k
            # Block scales travel with their weights and are sharded by the
            # same call, so they need the same split dims.
            w13_scale_data = _alloc_expert_weight(
                [num_experts, 2, scale_i, scale_h], torch.float32, 2, double_die
            ).view(num_experts, 2 * scale_i, scale_h)
            w2_scale_data = _alloc_expert_weight(
                [num_experts, scale_h, scale_i], torch.float32, 1, double_die
            )
            w13_scale_data.fill_(1)
            w2_scale_data.fill_(1)
        w13_weight_scale = torch.nn.Parameter(w13_scale_data, requires_grad=False)
        w2_weight_scale = torch.nn.Parameter(w2_scale_data, requires_grad=False)
        layer.register_parameter(
            f"w13_{self.weight_scale_name}", w13_weight_scale
        )
        layer.register_parameter(f"w2_{self.weight_scale_name}", w2_weight_scale)

        from vllm.model_executor.layers.fused_moe.routed_experts import (
            FusedMoeWeightScaleSupported,
        )

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES (dynamic scheme: none).
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # No CUDA _setup_kernel. Build the LPU moe_kernel (TritonExpertsFL)
        # instead, so self.moe_kernel.apply is available at forward time.
        # The params stay as-loaded native fp8 + fp32 block scale (single
        # residency).
        _lpu_setup_moe_kernel(self, layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: object | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Route through the FP8 expert path (TritonExpertsFL).  The moe_kernel
        # (built by _lpu_setup_moe_kernel) carries a real FP8 quant config: the
        # per-layer block scales + per-token activation contract live in the
        # config, and the A-quant flows through the patched scaled_fp8_quant
        # (patches/ops_fallback/fp8_quant.py).  The evas invoke op dequants both
        # A and B (E4M3 + E8M0 block scale) per-expert.
        assert self.moe_kernel is not None, "moe_kernel not built (LPU setup)"
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def _assert_block_scale_layout(layer: torch.nn.Module) -> None:
    """Fail loudly when the expert block scales do not match a 128x128 layout.

    ``FusedMoEQuantConfig.make`` only checks its flags against each other --
    nothing validates ``w1_scale``/``w2_scale`` against the block layout, so a
    wrong-shaped scale constructs fine and then broadcasts or fails deep in the
    kernel instead of saying what is wrong.  Declaring ``block_shape`` makes the
    layout an explicit part of the contract, so check it here, where both shapes
    are known and the failure is cheap to read.

    Raised explicitly rather than with ``assert``: this guards shapes that come
    from an external checkpoint, and a bare ``assert`` is stripped under
    ``python -O``, which would leave exactly the silent wrong-shape path this
    function exists to close.  The exception type is kept as ``AssertionError``
    to match the rest of the LPU layout guards.
    """
    for label, weight, scale in (
        ("w13", layer.w13_weight, layer.w13_weight_scale_inv),
        ("w2", layer.w2_weight, layer.w2_weight_scale_inv),
    ):
        want = (
            weight.shape[0],
            (weight.shape[1] + 127) // 128,
            (weight.shape[2] + 127) // 128,
        )
        if tuple(scale.shape) != want:
            raise AssertionError(
                f"LPU fp8 MoE {label} block scales are {tuple(scale.shape)}, "
                f"expected {want} for a 128x128 weight-block layout of "
                f"{tuple(weight.shape)}"
            )


def _lpu_setup_moe_kernel(method: Fp8MoEMethodLPU, layer: torch.nn.Module) -> None:
    """Build ``self.moe_kernel`` on LPU with a real FP8 quant config.

    No CUDA format conversion (``convert_to_fp8_moe_kernel_format`` /
    ``make_fp8_moe_kernel``): the OOT unquantized kernel machinery
    (``TritonExpertsFL``) is reused, but built with a genuine
    ``FusedMoEQuantConfig`` instead of the ``FUSED_MOE_UNQUANTIZED_CONFIG``
    singleton — fp8 weights + the per-layer block scales + the activation
    contract all live in the config, so ``per_act_token_quant`` /
    ``use_fp8_w8a8`` / ``w1_scale`` / ``w2_scale`` are honest (no instance-attr
    threading, no property overrides).  The A-quant then flows through the
    upstream ``moe_kernel_quantize_input`` -> ``_fp8_quantize``, which resolves
    to the patched ``per_token_group_quant_fp8`` once the block shape below is
    declared (see ``patches/ops_fallback/fp8_quant.py``).

    ``block_shape`` is the *activation* group shape in
    ``FusedMoEQuantConfig`` (``__post_init__`` asserts a per-token activation is
    incompatible with a block shape), and it is declared as the checkpoint's own
    ``weight_block_size`` -- ``[128, 128]`` -- rather than left ``None``.  Both
    real FP8 stacks this path targets declare that layout
    (``weight_block_size [128, 128]`` + a dynamic activation scheme), meaning
    one activation scale per (token, 128-channel block); declaring per-token
    asked for a coarser unit than the checkpoint declares, and than the frozen
    upstream evas kernel implements, which then had to be translated at the
    adoption boundary.  ``patches/ops_fallback/fp8_quant.py`` redirects the
    ``per_token_group_quant_fp8`` this reaches to the LPU encoder.
    """
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
        make_unquantized_moe_kernel,
    )

    from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL

    # The activation is FP8-quantized in the checkpoint's own unit -- one scale
    # per (token, 128-channel block) -- and the op dequants both A and B back to
    # bf16 for the LPU matmul.
    quant_config = FusedMoEQuantConfig.make(
        torch.float8_e4m3fn,
        per_act_token_quant=False,
        block_shape=[128, 128],
        w1_scale=layer.w13_weight_scale_inv,
        w2_scale=layer.w2_weight_scale_inv,
    )
    _assert_block_scale_layout(layer)
    method.moe_quant_config = quant_config
    method.moe_kernel = make_unquantized_moe_kernel(
        quant_config=quant_config,
        moe_config=method.moe,
        backend=UnquantizedMoeBackend.OOT,
        experts_cls=TritonExpertsFL,
        routing_tables=layer._expert_routing_tables(),
    )
