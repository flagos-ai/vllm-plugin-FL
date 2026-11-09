# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Triton-free KV block zeroing for the Kunlunxin XPU backend.

vLLM 0.28 zeroes freshly allocated KV cache blocks inside ``KVBlockZeroer``
via the ``_zero_kv_blocks_kernel`` Triton kernel, which dereferences raw
absolute addresses held in a uint64 tensor::

    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))

``patches.triton_compat`` only made that kernel *compile* on the XPU backend.
Launching it still kills the device on P800 at the first real request::

    xpuLaunchKernel("_zero_kv_blocks_kernel", ...)
      -> Exception in kernel execution (err_code: -714)

so ``POST /v1/chat/completions`` answered HTTP 500.  Reimplement
``zero_block_ids`` with ordinary tensor ops instead.

The first tensor-op implementation still died with ``error code= 700, an
illegal memory access was encountered`` at the same call site, but at a
different line: ``torch.tensor(physical, dtype=torch.long, device=kv.device)``.
Building the index on the device is a host->device copy, and that copy kernel
is what faults on P800 (the same ``eager_customized/copy_kernel.cpp`` path that
breaks the 35B-A3B profile run).  So the zeroing must not touch that path at
all: contiguous physical kernel blocks are merged into spans and cleared with
``kv.narrow(block_dim, start, len).zero_()``, which issues only fill kernels.

Semantics are preserved, not weakened.  Upstream builds one segment per
(outer index, virtual index): segment start = ``data_ptr + outer_offset +
virtual_index * block_stride_bytes``, block stride = ``block_stride_bytes *
ratio``, zeroed span = one kernel page (one kernel block's K/V bytes).  Hence
physical kernel block ``block_id * ratio + virtual_index``.  Zeroing exactly
that index set zeroes the same bytes for every block vLLM hands us, so a
freshly allocated block still reads back as zeros.

TODO: remove once the XPU Triton backend supports absolute-address pointers
or vLLM ships a non-Triton zeroing path.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False


def _collect_kv_segments(
    attn_groups,
    kernel_block_sizes,
    cache_dtype,
    runner_only_attn_layers,
    static_forward_context,
):
    """Map each distinct KV cache tensor to ``(tensor, block_dim, ratio)``.

    Mirrors ``KVBlockZeroer.__init__`` so exactly the same tensors, block
    dimension and virtual-block ratio are covered.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec

    segments = []
    seen_ptrs = set()
    for group in attn_groups:
        spec = group.kv_cache_spec
        if not isinstance(spec, AttentionSpec):
            # Mamba/SSM state groups are skipped upstream too.
            continue
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue
        kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
        if not kernel_bs or spec.block_size % kernel_bs:
            continue
        ratio = spec.block_size // kernel_bs
        block_dim = group.backend.get_kv_cache_block_dim(
            kernel_bs,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype,
        )
        for layer_name in group.layer_names:
            if layer_name in runner_only_attn_layers:
                continue
            kv = static_forward_context[layer_name].kv_cache
            if not isinstance(kv, torch.Tensor):
                continue
            ptr = kv.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            if kv.shape[block_dim] % ratio:
                continue
            segments.append((kv, block_dim, ratio))
    return segments


def _zero_block_ids_pytorch(self, block_ids) -> None:
    """Triton-free, copy-free replacement for ``KVBlockZeroer.zero_block_ids``.

    Zeroing used to be done with ``index_fill_`` over an index tensor built as
    ``torch.tensor(physical, dtype=torch.long, device=kv.device)``.  That is a
    host->device copy on every call, and on P800 the copy kernel itself faults
    (``eager_customized/copy_kernel.cpp``) which surfaces as
    ``error code= 700, an illegal memory access was encountered`` at the first
    ``POST /v1/chat/completions`` -> HTTP 500.

    Zero contiguous spans with ``narrow`` + ``zero_`` instead: no index tensor
    is materialized, no host->device copy is issued, and only plain fill
    kernels touch the KV cache.  The bytes written are identical to the
    ``index_fill_`` version (same per-dimension slices, value zero).
    """
    segments = getattr(self, "_kunlunxin_kv_segments", None)
    if not block_ids or not segments:
        return
    ids = sorted({int(b) for b in block_ids})
    for kv, block_dim, ratio in segments:
        num_kernel_blocks = kv.shape[block_dim]
        run_start = -1
        run_len = 0
        for block_id in ids:
            start = block_id * ratio
            if start < 0 or start + ratio > num_kernel_blocks:
                continue
            if run_start < 0:
                run_start, run_len = start, ratio
            elif start == run_start + run_len:
                run_len += ratio
            else:
                kv.narrow(block_dim, run_start, run_len).zero_()
                run_start, run_len = start, ratio
        if run_start >= 0:
            kv.narrow(block_dim, run_start, run_len).zero_()


def apply_kv_block_zero_compat() -> None:
    """Replace the Triton KV-block zeroing kernel with tensor-op zeroing."""
    global _patched
    if _patched:
        return
    try:
        from vllm.v1.worker.utils import KVBlockZeroer
    except Exception:  # pragma: no cover - vLLM without the zeroer
        return

    if getattr(KVBlockZeroer.zero_block_ids, "_kunlunxin_pytorch_zero", False):
        _patched = True
        return

    original_init = KVBlockZeroer.__init__

    def init_and_collect(
        self,
        device,
        attn_groups_iter,
        kernel_block_sizes,
        cache_dtype,
        static_forward_context,
        runner_only_attn_layers=None,
    ):
        # Materialize first: upstream consumes the iterable.
        groups = list(attn_groups_iter)
        original_init(
            self,
            device,
            iter(groups),
            kernel_block_sizes,
            cache_dtype,
            static_forward_context,
            runner_only_attn_layers,
        )
        try:
            self._kunlunxin_kv_segments = _collect_kv_segments(
                groups,
                kernel_block_sizes,
                cache_dtype,
                runner_only_attn_layers or set(),
                static_forward_context,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "kunlunxin kv_block_zero_compat: segment collection failed: %s", exc
            )
            self._kunlunxin_kv_segments = None

    _zero_block_ids_pytorch._kunlunxin_pytorch_zero = True
    KVBlockZeroer.__init__ = init_and_collect
    KVBlockZeroer.zero_block_ids = _zero_block_ids_pytorch
    _patched = True
    logger.info(
        "kunlunxin kv_block_zero_compat: KV block zeroing uses tensor ops "
        "instead of the Triton absolute-address kernel"
    )
