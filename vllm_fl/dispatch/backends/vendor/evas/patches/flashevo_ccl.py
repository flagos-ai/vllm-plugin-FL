# Copyright (c) 2026 BAAI. All rights reserved.
"""flashevo XCCL collective routing for ``CommunicatorFL`` (LPU TP).

Split out of ``vllm_fl/distributed/communicator.py``;
``apply_lpu_flashevo_ccl_patch()`` rebinds the wrappers (same pattern as
vendor/sunrise). flashevo XCCL is the LPU TP communication path (vllm-evas
EPOCHCommunicator); the wrappers take over only on an LPU device with
``pyflagcx_comm.disabled`` and otherwise delegate upstream. Single-die
inputs (``split_dim == -1``) fall back to torch.distributed. The
process-global communicator is drained and destroyed on shutdown.

"""

from __future__ import annotations

import atexit
import functools
import os
from typing import Optional, Union

import torch
import torch_lpu

from vllm.distributed.device_communicators.base_device_communicator import \
    DeviceCommunicatorBase

# `_flash_evo_dist()` is the single choke point every flashevo branch checks;
# it returns None only when flash_evo is not importable.
def _flash_evo_dist():
    try:
        import flash_evo
        return flash_evo
    except ImportError:
        return None


# Post-collective drain is intentionally dropped.  A device-wide
# ``torch.lpu.synchronize()`` after every fe.dist_* call below orders nothing in
# this configuration — it only stalls the CPU launcher between two enqueues:
#
#   * flash_evo's Dispatch{AllReduce,AllGather,ReduceScatter} pass the global
#     ``g_stream`` to xccl*; ``g_stream`` is written only by
#     ``flash_evo::SetStream`` (tvm-ffi ``set_stream``), which has no caller in
#     this project, so it stays 0 == the default LPU stream.  torch compute
#     runs on ``current_stream()``, also ``lpu_stream=0x0`` (no evas-path code
#     enters ``torch.lpu.stream(...)``).
#   * flash_evo inits the communicator with the EP-proxy flag clear (it passes
#     ecx=0 to ``xcclCommInitRank``), so ``xcclEnqueueCheck`` takes the direct
#     path and launches the collective synchronously (``evConfigureCall``) on
#     the calling thread.
#
# Compute and collectives are therefore FIFO on one stream; the drain costs a
# full device drain per collective (~2 per layer -> ~86 per step at TP8
# 43-layer) for no ordering benefit.  Escape hatches preserve the legacy
# behavior whenever the verified topology does not hold: ``VLLM_FL_FE_POST_SYNC=1``
# forces the drain, and a non-default current stream (side-stream caller) keeps
# it too — the collective still lands on the default stream there, so the drain
# is the conservative choice.
def _fe_post_collective_sync() -> None:
    if (os.environ.get("VLLM_FL_FE_POST_SYNC", "0") == "0"
            and torch.lpu.current_stream() == torch.lpu.default_stream()):
        return
    torch.lpu.synchronize()


_FLASH_EVO_DIST_INITIALIZED = False
_FLASH_EVO_CLEANUP_REGISTERED = False


def _fe_probe(op: str, tensors: tuple = ()) -> None:
    """VLLM_FL_DSV4_DEBUG_ARGS=1: log every flash_evo xccl collective with its
    caller stack.  The TP4 prefill wedge (`p(2) ... cidb 0` copy/queue-engine
    dump) lands right after the embedding lookup — i.e. at/near the run's first
    TP collective — and flashevo is TP4-only (TP1 has no CCL), so this path is
    the one remaining un-probed candidate (the earlier dist-probe only wrapped
    ``torch.distributed``; flashevo calls ``fe.dist_*`` directly in C).
    """
    if os.environ.get("VLLM_FL_DSV4_DEBUG_ARGS") != "1":
        return
    import traceback

    def _shape(v):
        if isinstance(v, torch.Tensor):
            return f"{tuple(v.shape)}/{v.dtype}"
        return type(v).__name__

    stack = " | ".join(
        line.strip().split("\n")[0]
        for line in traceback.format_stack(limit=5)[:-1]
    )
    args = ", ".join(_shape(t) for t in tensors)
    print(f"[fe-probe] {op}({args}) from {stack}", flush=True)


def _ensure_flash_evo_dist(rank: int, world_size: int) -> bool:
    """Init flash_evo exactly once per process (idempotent).

    Mirrors vllm-evas worker.init_worker: flash_evo keeps a single global xccl
    communicator per process, so the first caller (TP group) initializes it
    with the GLOBAL rank/world_size and everyone else reuses it.  Init is
    two-step (both required, same order as vllm-evas model_runner.load_model +
    worker.py): ``vllm_fusedop_init(device)`` sets up the LIB on the LPU device,
    then ``dist_init_process_group`` creates the xccl process group.
    """
    global _FLASH_EVO_DIST_INITIALIZED
    if _FLASH_EVO_DIST_INITIALIZED:
        return True
    fe = _flash_evo_dist()
    if fe is None:
        return False
    try:
        # flash_evo's LIB needs a one-time per-process device init before any
        # of its ops (including the xccl collectives below) can run — mirrors
        # vllm-evas model_runner.load_model (`vllm_fusedop_init(current_device)`
        # before weight preprocessing).  Without it the dist_* calls below
        # operate on an uninitialized LIB.
        current_device = torch.lpu.current_device()
        _fe_probe("fusedop_init")
        fe.LIB.vllm_fusedop_init(current_device)
        fe.LIB.dist_init_process_group(rank, world_size, "")
    except Exception as exc:  # noqa: BLE001
        print(f"[flashevo_ccl] flashevo dist init failed: {exc}", flush=True)
        return False
    _FLASH_EVO_DIST_INITIALIZED = True
    print(f"[flashevo_ccl] flashevo xccl dist initialized "
          f"(rank={rank}, world_size={world_size}) — flashevo CCL active",
          flush=True)
    _register_flash_evo_cleanup()
    return True


def _destroy_flash_evo_dist() -> None:
    """Tear down the process-wide flash_evo xccl communicator.

    The init above is one-way and process-global, so without this the
    communicator and its device-side state outlive every engine teardown and
    are reclaimed only by process exit -- and an exit that races a queued
    collective is the shape that leaves the board non-idle.  vllm-evas does
    the same pair (``worker.py::_destroy_flash_evo_dist``: drain, then
    ``dist_destroy_process_group``) and reaches it from both its shutdown hook
    and ``atexit``.

    Idempotent: the drain runs at most once per init, and a later init re-arms
    it, so an engine restarted in the same process re-initializes cleanly.
    """
    global _FLASH_EVO_DIST_INITIALIZED
    if not _FLASH_EVO_DIST_INITIALIZED:
        return
    # Clear the flag first: a destroy that raises must not leave the process
    # believing the communicator is still usable.
    _FLASH_EVO_DIST_INITIALIZED = False
    fe = _flash_evo_dist()
    if fe is None:
        return
    try:
        # A collective already queued on the stream must retire before the
        # communicator goes away; destroying under it is what wedges.
        torch.lpu.synchronize()
    except Exception:  # noqa: BLE001 - a dead device has nothing left to drain
        pass
    try:
        fe.LIB.dist_destroy_process_group()
    except Exception as exc:  # noqa: BLE001 - teardown must not mask a clean exit
        print(f"[flashevo_ccl] flashevo dist destroy failed: {exc}",
              flush=True)


def _register_flash_evo_cleanup() -> None:
    """Arrange for the communicator to be destroyed on every exit path.

    vLLM's graceful shutdown calls ``cleanup_dist_env_and_memory`` (bound in
    ``vllm.distributed.parallel_state`` and imported by name into
    ``vllm.v1.engine.core``), and a worker killed by a signal only runs
    ``atexit``.  Both are covered here; the upstream function itself is
    wrapped, not replaced, so its own teardown still runs afterwards.
    """
    global _FLASH_EVO_CLEANUP_REGISTERED
    if _FLASH_EVO_CLEANUP_REGISTERED:
        return
    _FLASH_EVO_CLEANUP_REGISTERED = True
    atexit.register(_destroy_flash_evo_dist)

    from vllm.distributed import parallel_state

    orig = parallel_state.cleanup_dist_env_and_memory

    def _cleanup_with_flashevo(*args, **kwargs):
        _destroy_flash_evo_dist()
        return orig(*args, **kwargs)

    parallel_state.cleanup_dist_env_and_memory = _cleanup_with_flashevo
    try:
        # The engine core took its own binding at import time, so rebinding
        # the origin module alone would not reach the one call site (core.py).
        from vllm.v1.engine import core as engine_core

        engine_core.cleanup_dist_env_and_memory = _cleanup_with_flashevo
    except ImportError:
        pass


# xccl dtype whitelist for the flashevo ops (GetXcclDataType). bf16 is not in
# the list, so it is widened to fp16 before the collective (bf16->fp16 widening
# is lossless; the reduce then runs in fp16).
_FLASH_EVO_DTYPES = (torch.float32, torch.float16, torch.int32, torch.int8)


def _to_flashevo_dtype(t: torch.Tensor) -> tuple[torch.Tensor, torch.dtype]:
    """Return (tensor in an xccl-supported dtype, original dtype)."""
    if t.dtype in _FLASH_EVO_DTYPES:
        return t, t.dtype
    return t.to(torch.float16), t.dtype


def _dd_is_row_split(t: torch.Tensor) -> bool:
    """True if `t` is a row-split double-die tensor (the default under
    ENABLE_DOUBLE_DIE=1), i.e. exactly what flashevo's dist_* ops consume.
    Single-die tensors (split_dim == -1) are handled by the torch.distributed
    fallback, not flashevo."""
    if t.dim() == 0 or t.numel() == 0:
        return False
    return getattr(t, "split_dim", lambda: -1)() == 0


def _dd_cast_back(dd: torch.Tensor, orig_dtype: torch.dtype) -> torch.Tensor:
    """Cast a row-split double-die collective result back to the original dtype.

    The logical view of a row-split result already reads the full result
    (die0 ++ die1) and `.to()` dtype-cast preserves the die layout (verified
    on-device), so we just cast the widened fp16 back to the original dtype.
    """
    return dd.to(orig_dtype) if orig_dtype != dd.dtype else dd


def _dd_gather_permute(out_dd: torch.Tensor, world_size: int) -> torch.Tensor:
    """Convert flashevo's all_gather output (die-major) to vLLM rank-major.

    flashevo's dist_all_gather runs an independent ring per die: with a
    row-split double-die input (bs, h) where die0 = first bs/2 rows, die1 =
    last bs/2 rows, the LOGICAL output (ws*bs, h) is die-major:

        [rank0.die0; rank1.die0; ...; rank_{ws-1}.die0 | rank0.die1; ...]

    i.e. all die0 halves concatenated first, then all die1 halves. vLLM's TP
    all_gather expects the standard NCCL rank-major concat:

        [rank0.(die0|die1); rank1.(die0|die1); ...]

    This local row-permutation rewrites the former into the latter. `out_dd`
    is the logical row-split view; the permuted result is a plain single-die
    tensor of the same (logical) shape. Verified on-device (ws=4 probe) and
    on CPU for ws in {2,4}, bs in {4,8}, h in {4,512,1024}.
    """
    bs = out_dd.shape[0] // world_size   # input rows per rank (logical)
    rest = out_dd.shape[1:]
    if bs < 2 or bs % 2 == 1:
        # flashevo's per-die rings require symmetric halves (die0 = first bs/2
        # rows, die1 = last bs/2 rows), which only exist for even bs.  An odd
        # per-rank row count (decode bs == 1) cannot be represented by that
        # geometry — the caller must mirror-pad the input to an even row count
        # first (see _dd_all_gather_rowsplit).  Surface a clear error rather
        # than letting the geometry mismatch crash later with a confusing
        # empty-slice/reshape error.
        raise RuntimeError(
            f"_dd_gather_permute: odd per-rank rows (bs={bs}) unsupported by "
            "flashevo's symmetric ring geometry; mirror-pad the input to an "
            "even row count before gathering")
    bsh = bs // 2                        # rows per die
    h = 1
    for d in rest:
        h *= d
    rm = torch.empty((world_size * bs,) + rest,
                     dtype=out_dd.dtype, device=out_dd.device)
    # interleave per-rank [die0; die1] rows from the die-major buffer
    for r in range(world_size):
        rm[r * bs:(r + 1) * bs] = torch.cat([
            out_dd[r * bsh:(r + 1) * bsh].reshape(bsh, h),
            out_dd[world_size * bsh + r * bsh:
                   world_size * bsh + (r + 1) * bsh].reshape(bsh, h),
        ], dim=0).reshape(bs, *rest)
    return rm


def _dd_all_gather_rowsplit(
    input_: torch.Tensor,
    world_size: int,
    global_rank: int,
    global_world_size: int,
) -> torch.Tensor | None:
    """Flashevo all-gather for a row-split double-die input, returned rank-major.

    Handles odd per-rank row counts (decode ``bs == 1``): flashevo's per-die
    rings need symmetric halves (die0 = first bs/2 rows, die1 = last bs/2
    rows), which don't exist for odd ``bs``.  For those we pad the input to
    ``2*bs`` rows by PLAIN DUPLICATION (``torch.cat([input_, input_], dim=0)``)
    -- not a byte-reversed mirror: die0 = the real rows, die1 = the same rows
    repeated.  Both dies then carry the same data, the ring geometry matches
    the verified even-bs case, and the permuted result has each rank's two
    slots equal — we drop the duplicated slot and keep the real one (the
    ``[::2]`` below).  The caller's reshape/movedim then sees a pure rank-major
    result (each rank contributes ``bs`` unique rows).  Returns ``None`` when
    flashevo is unavailable so the caller can fall back.

    PRECONDITION: the ``[::2]`` drop below is only correct when each rank's
    permuted block is exactly 2 rows, i.e. ``bs == 1`` (the decode case).  For
    odd ``bs >= 3`` the duplicated rows interleave with the real ones and
    ``[::2]`` would silently drop real rows; no such call site exists today, so
    this is a documented precondition rather than a handled case.
    """
    if input_.shape[0] % 2 == 1:
        # odd bs: pad to an even row count.  (bs, *rest) -> (2*bs, *rest)
        # with the real rows in die0 (first half) and the SAME rows repeated in
        # die1 (second half) -- a plain cat, not a reversed mirror.  die1's
        # content is irrelevant — the permuted result keeps only each rank's
        # die0 slot (see the `[::2]` below, valid only for bs == 1) — so a
        # plain cat is sufficient and matches the verified even-bs ring
        # geometry.
        t_pad = torch.cat([input_, input_], dim=0)
        t, orig_dtype = _to_flashevo_dtype(t_pad)
        bs_logical = t_pad.shape[0]
    else:
        t, orig_dtype = _to_flashevo_dtype(input_)
        bs_logical = input_.shape[0]
    output_size = (bs_logical * world_size,) + t.shape[1:]
    out_dd = torch_lpu.empty_with_format(
        output_size, dtype=t.dtype, device=t.device, split_dim=0)
    fe = _flash_evo_dist()
    if fe is None or not _ensure_flash_evo_dist(global_rank, global_world_size):
        return None
    _fe_probe("dist_all_gather", (input_, out_dd))
    fe.dist_all_gather(out_dd, t)
    _fe_post_collective_sync()
    out_dd = _dd_cast_back(out_dd, orig_dtype)
    rm = _dd_gather_permute(out_dd, world_size)  # (ws*bs_logical, *rest)
    if bs_logical == 2 * input_.shape[0]:
        # bs == 1: keep each rank's single real row, drop the duplicated slot
        # (see the PRECONDITION note in the docstring -- this pairwise drop is
        # not the general de-duplication for odd bs >= 3).
        rm = rm[::2]
    return rm


# ---- CommunicatorFL method implementations (LPU + flagcx-disabled only) ----


def _lpu_all_reduce(self, input_):
    # Debug probe (VLLM_FL_DEBUG_ALLREDUCE=1, default off): dump every
    # all_reduce's per-rank pre/post mean + absmax so the caller can tell
    # (a) whether each rank feeds a full/identical or a partial/different
    # tensor (compare pre stats across ranks) and (b) whether the
    # collective actually sums it (post/pre absmax ratio, ~1 for a
    # non-summing no-op, ~4 for a sum of 4 identical full copies). The
    # [M,4096] calls are the MoE layer-end / attention all_reduces.
    _dbg = os.environ.get("VLLM_FL_DEBUG_ALLREDUCE") == "1"
    if _dbg:
        _pre = input_.float()
        _pre = (_pre.mean().item(), _pre.abs().max().item())
    # flagcx lib missing (LPU with zccl backend): torch.distributed's
    # zccl all_reduce crashes inside the vLLM multiproc worker env
    # (LPUEvent.record -> evError 2181), so route through flashevo's
    # xccl all_reduce instead. Row-split double-die input (the default)
    # is consumed in-place with a full reduce; single-die input falls
    # back to torch.distributed. bf16 is widened to fp16 (lossless).
    fe = _flash_evo_dist()
    if (fe is not None
            and _dd_is_row_split(input_)
            and _ensure_flash_evo_dist(self.global_rank,
                                       self.global_world_size)):
        # flashevo's per-die ring needs symmetric halves (die0 =
        # first bs/2 rows, die1 = last bs/2 rows); bs == 1 (decode)
        # leaves die1 EMPTY, which dist_all_reduce mis-handles for
        # the one-rank-non-zero embedding case (TP4 e2e first token
        # diverges).  Mirror-pad to (2, h) so die1 mirrors die0 (the
        # verified even-bs geometry), reduce, then drop the mirror
        # row.  bs > 1 is already the row-split geometry dist_*
        # expects: in-place full reduce, result returned as-is.
        # One code path for both -- the two cases differ only in the
        # source tensor and the trailing row-drop.
        mirror = input_.shape[0] == 1
        src = torch.cat([input_, input_], dim=0) if mirror else input_
        t, orig_dtype = _to_flashevo_dtype(src)
        _fe_probe("dist_all_reduce", (input_, t))
        fe.dist_all_reduce(t, t, "sum")
        _fe_post_collective_sync()
        out = _dd_cast_back(t, orig_dtype)
        if mirror:
            out = out[::2]
    else:
        # fall through to torch.distributed if flashevo unavailable
        # or input is single-die (not the default double-die runtime)
        out = input_.clone()
        torch.distributed.all_reduce(out, group=self.device_group)
    if _dbg:
        _post = out.float()
        _post = (_post.mean().item(), _post.abs().max().item())
        print(
            f"[AR-DBG] rank={self.global_rank} shape={tuple(input_.shape)} "
            f"dtype={input_.dtype} "
            f"pre_mean={_pre[0]:.6g} pre_absmax={_pre[1]:.6g} "
            f"post_mean={_post[0]:.6g} post_absmax={_post[1]:.6g} "
            f"absmax_ratio={_post[1] / max(_pre[1], 1e-12):.4f}",
            flush=True,
        )
    return out


def _lpu_reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
    world_size = self.world_size
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()

    # Note: This will produce an incorrect answer if we don't make
    # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
    input_tensor = input_.movedim(0, dim).contiguous()

    assert input_tensor.shape[0] % world_size == 0
    chunk_size = input_tensor.shape[0] // world_size

    # flashevo's xccl reduce_scatter, else torch.distributed zccl fallback.
    #
    # NOTE: flashevo's dist_reduce_scatter has die-major semantics —
    # it reduce-scatters each die's own segment independently, giving
    # rank r `[reduced[r*c/2:(r+1)*c/2], reduced[M/2+r*c/2:...]]` (probe
    # on ws=4). vLLM's reduce_scatter wants the global rank-major chunk
    # `reduced[r*c:(r+1)*c]`, which crosses die boundaries and is NOT
    # recoverable by local permutation (the needed rows live in other
    # ranks' output). So we implement the exact vLLM semantics with a
    # composite: flashevo all_reduce (in-place, full logical shape) then
    # a local slice of rank r's chunk. Correct for any dim/ws.
    fe = _flash_evo_dist()
    if (fe is not None
            and _dd_is_row_split(input_tensor)
            and _ensure_flash_evo_dist(self.global_rank,
                                       self.global_world_size)):
        t, orig_dtype = _to_flashevo_dtype(input_tensor)
        _fe_probe("dist_all_reduce", (input_, t))
        fe.dist_all_reduce(t, t, "sum")
        _fe_post_collective_sync()
        out = _dd_cast_back(t, orig_dtype)
        # local slice: rank r keeps rows [r*c, (r+1)*c)
        out = out[self.rank_in_group * chunk_size:
                  (self.rank_in_group + 1) * chunk_size]
        return out.movedim(0, dim).contiguous()
    output_tensor = torch.empty_like(input_tensor)
    torch.distributed.reduce_scatter(output_tensor, input_tensor,
                                     group=self.device_group)
    return output_tensor.movedim(0, dim).contiguous()


def _lpu_reduce_scatterv(self,
                         input_: torch.Tensor,
                         dim: int = -1,
                         sizes: Optional[list[int]] = None):
    world_size = self.world_size
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()

    # Note: This will produce an incorrect answer if we don't make
    # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
    input_tensor = input_.movedim(0, dim).contiguous()

    if sizes is not None:
        assert len(sizes) == world_size
        assert input_tensor.shape[0] == sum(sizes)
        chunk_size = sizes[self.rank_in_group]
    else:
        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size

    # flashevo's xccl reduce_scatter (double-die input in xccl-supported
    # dtype), else torch.distributed zccl fallback. `sizes` is unsupported by
    # flashevo (xccl has no variable-size reduce_scatter), so uneven
    # gathers fall back to torch.distributed.
    #
    # Like reduce_scatter, the flashevo die-major output cannot produce
    # vLLM's global rank-major chunk, so we use all_reduce + local slice
    # (exact semantics for equal chunk sizes; sizes=None is guaranteed
    # equal here because flashevo path only runs when sizes is None).
    fe = _flash_evo_dist()
    if (fe is not None and sizes is None
            and _dd_is_row_split(input_tensor)
            and _ensure_flash_evo_dist(self.global_rank,
                                       self.global_world_size)):
        t, orig_dtype = _to_flashevo_dtype(input_tensor)
        _fe_probe("dist_all_reduce", (input_, t))
        fe.dist_all_reduce(t, t, "sum")
        _fe_post_collective_sync()
        out = _dd_cast_back(t, orig_dtype)
        out = out[self.rank_in_group * chunk_size:
                  (self.rank_in_group + 1) * chunk_size]
        return out.movedim(0, dim).contiguous()
    output_tensor = torch.empty_like(input_tensor)
    torch.distributed.reduce_scatter(output_tensor, input_tensor,
                                     group=self.device_group)
    return output_tensor.movedim(0, dim).contiguous()


def _lpu_all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """TP all-gather (GroupCoordinator.all_gather -> here when ws>1).

    The base DeviceCommunicatorBase.all_gather goes through
    dist.all_gather_into_tensor (zccl), which crashes in the vLLM multiproc
    worker env on LPU. Under the default double-die runtime the input is a
    row-split tensor, so we use flashevo's dist_all_gather (which consumes
    row-split double-die directly) and then locally permute the die-major
    result into vLLM's rank-major concat (see _dd_gather_permute).
    """
    world_size = self.world_size
    if dim < 0:
        dim += input_.dim()
    if _dd_is_row_split(input_) and _flash_evo_dist() is not None:
        rm = _dd_all_gather_rowsplit(
            input_, world_size,
            self.global_rank, self.global_world_size)
        if rm is not None:
            input_size = input_.size()
            output_tensor = rm.reshape((world_size,) + input_size)
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (world_size * input_size[dim],)
                + input_size[dim + 1:])
            return output_tensor
    # single-die input or flashevo unavailable: base fallback
    return DeviceCommunicatorBase.all_gather(self, input_, dim)


def _lpu_all_gatherv(self,
                     input_: Union[torch.Tensor, list[torch.Tensor]],
                     dim: int = 0,
                     sizes: Optional[list[int]] = None):
    if dim != 0:
        raise NotImplementedError("only dim 0 all-gatherv is supported")
    world_size = self.world_size

    # 'sizes' is not needed if all inputs in the same group have the same
    # shape
    if sizes is not None and all(s == sizes[0] for s in sizes):
        sizes = None

    def _all_gather_single(input_: torch.Tensor,
                           sizes: Optional[list[int]] = None):
        input_size = input_.size()
        if sizes is not None:
            assert len(sizes) == world_size
            assert input_.shape[dim] == sizes[self.rank_in_group], (
                f"{input_.shape[dim]} != {sizes[self.rank_in_group]}")
            output_size = (sum(sizes), ) + input_size[1:]
        else:
            output_size = (input_size[0] * world_size, ) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(output_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # flashevo's xccl all_gather (double-die in xccl-supported dtype),
        # else torch.distributed zccl fallback. sizes==None here is the
        # equal-shape all-gather.
        if (sizes is None and _dd_is_row_split(input_)
                and _flash_evo_dist() is not None):
            # output buffer: logical output shape
            # (input_size[0] * world_size, ...) with split_dim=0
            # (EPOCHCommunicator style). flashevo fills it in
            # die-major layout; we permute to rank-major before
            # returning (see _dd_gather_permute).
            rm = _dd_all_gather_rowsplit(
                input_, world_size,
                self.global_rank, self.global_world_size)
            if rm is not None:
                return rm
        torch.distributed.all_gather_into_tensor(
            output_tensor, input_, group=self.device_group)
        return output_tensor

    if isinstance(input_, torch.Tensor):
        return _all_gather_single(input_, sizes)

    return [_all_gather_single(inp, sizes) for inp in input_]


# ---- Patch entry ----

_FLASHEVO_CCL_PATCHED = False


def _is_lpu_ccl_case(comm) -> bool:
    """True for exactly one case: LPU device + flagcx communicator present but
    disabled (no libflagcx.so; its collectives are no-ops). Everything else
    delegates upstream."""
    return (comm.device is not None
            and comm.device.type == "lpu"
            and comm.pyflagcx_comm is not None
            and comm.pyflagcx_comm.disabled)


def _wrap(orig, lpu_fn):
    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        if _is_lpu_ccl_case(self):
            return lpu_fn(self, *args, **kwargs)
        return orig(self, *args, **kwargs)
    return wrapper


def apply_lpu_flashevo_ccl_patch() -> None:
    """Rebind CommunicatorFL TP collectives to the flashevo XCCL path.

    Idempotent.  Non-LPU devices are unreachable here (the evas patch entry
    is gated on ``device_type == "lpu"``); within LPU, non-disabled flagcx
    keeps the upstream flagcx path.
    """
    global _FLASHEVO_CCL_PATCHED
    if _FLASHEVO_CCL_PATCHED:
        return
    from vllm_fl.distributed.communicator import CommunicatorFL

    CommunicatorFL.all_reduce = _wrap(CommunicatorFL.all_reduce, _lpu_all_reduce)
    CommunicatorFL.reduce_scatter = _wrap(CommunicatorFL.reduce_scatter,
                                          _lpu_reduce_scatter)
    CommunicatorFL.reduce_scatterv = _wrap(CommunicatorFL.reduce_scatterv,
                                           _lpu_reduce_scatterv)
    CommunicatorFL.all_gather = _wrap(CommunicatorFL.all_gather,
                                      _lpu_all_gather)
    CommunicatorFL.all_gatherv = _wrap(CommunicatorFL.all_gatherv,
                                       _lpu_all_gatherv)
    _FLASHEVO_CCL_PATCHED = True
