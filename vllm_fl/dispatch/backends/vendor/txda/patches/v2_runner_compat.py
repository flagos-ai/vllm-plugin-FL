# Copyright (c) 2026 BAAI. All rights reserved.

"""TXDA adaptations for the upstream V2 model runner.

The upstream V2 runner (``vllm/v1/worker/gpu``) is written for a CUDA build.
These assumptions break on TXDA with a ``VLLM_TARGET_DEVICE=empty`` build:

* ``torch.cuda`` stream/event/graph constructors are dummy base classes. On
  TXDA they are only functional after ``torch_txda.transfer_to_txda`` has
  aliased the whole ``torch.cuda`` namespace, which the TXDA launchers do not
  necessarily import (``vllm serve`` and library users never touch it). The
  V2 runner constructs ``torch.cuda.Stream``/``Event``/``CUDAGraph`` itself,
  so the aliases have to be installed explicitly.
* ``buffer_utils`` stages host-side updates through ``UvaBuffer``, which needs
  pinned host memory that also maps into the device address space. TXDA has
  neither, so the V2 buffers must take the explicit-copy ``NonUvaBuffer``
  path.
* ``cudagraph_utils`` binds ``graph_capture`` at import time from
  ``vllm.distributed.parallel_state``, which opens a CUDA stream and asserts on
  an NCCL communicator. TXDA runs on FlagCX, so the device-agnostic capture
  context that ``ModelRunnerFL`` resolves for this platform is used instead.
* The TXDA Triton backend miscompiles kernels that cast a runtime-loaded
  integer (a ``data_ptr`` held in a tensor) to a pointer: the MLIR lowering
  aborts, or the load silently reads address 0. The V2 runner does this in
  its block-table write/gather, slot mapping and Mamba "align" kernels, so
  ``block_table_native`` and ``mamba_align`` replace those kernels with
  native tensor indexing.

Everything is applied from ``WorkerFL.__init__`` before the V2 runner is
constructed and leaves the upstream behaviour untouched wherever a probe
succeeds.

Removal condition: delete this module once the vLLM wheel maps host memory
into the device address space on TXDA, ships a ``torch.cuda`` that works with
the PrivateUse1 backend, FlagCX satisfies the communicator assertion in
``cudagraph_utils``, and the TXDA Triton backend supports casting loaded
integers to pointers.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_installed = False


def _txda_cuda_aliases_installed() -> bool:
    """Whether ``torch.cuda`` already resolves to the TXDA implementation."""
    try:
        import torch_txda
    except ImportError:  # pragma: no cover - only on non-TXDA hosts
        return False
    return bool(getattr(torch_txda, "is_transfer_to_txda", False))


def install_cuda_aliases() -> None:
    """Point the ``torch.cuda`` APIs the V2 runner constructs at ``torch.txda``.

    ``torch_txda`` ships ``torch.txda`` (streams, events, graphs) but only
    exposes it under the ``torch.cuda`` name after
    ``transfer_to_txda`` runs. Install the subset the V2 runner uses
    directly; skip entirely when the alias is already in place so a launcher
    that imported ``transfer_to_txda`` keeps its full namespace.
    """
    if _txda_cuda_aliases_installed():
        return

    import torch.txda as txda

    from vllm_fl.dispatch.backends.vendor.txda.patch import (
        patch_cuda_stream_apis_for_txda,
    )

    # stream()/set_stream()/current_stream() first: the aliases below build on
    # them and the patch is idempotent.
    patch_cuda_stream_apis_for_txda()

    cuda = torch.cuda
    # Only replace the constructors that are CUDA-only. Anything the torch
    # build resolves natively (memory info, device properties, ...) is left
    # alone: vLLM reaches those through torch.accelerator, which already
    # dispatches to PrivateUse1.
    for name, impl in (
        ("Stream", txda.Stream),
        ("ExternalStream", txda.ExternalStream),
        ("Event", txda.Event),
        ("CUDAGraph", txda.TXDAGraph),
        ("graph", txda.graph),
        ("graph_pool_handle", txda.graph_pool_handle),
        ("is_current_stream_capturing", txda.is_current_stream_capturing),
    ):
        if hasattr(txda, name):
            setattr(cuda, name, getattr(txda, name))

    # ``ubatch_utils`` calls this purely to force a stream association before
    # launching; TXDA has no cuBLAS handle, so a no-op is faithful.
    cuda.current_blas_handle = lambda *args, **kwargs: 0

    _wrap_graph_capture_mode()
    logger.info("Installed torch.cuda -> torch.txda aliases for the V2 runner")


def _wrap_graph_capture_mode() -> None:
    """Default TXDA graph captures to ``capture_error_mode="relaxed"``.

    ``torch.txda.graph`` forwards ``capture_error_mode`` to
    ``txStreamBeginCapture``. In the default ``"global"`` mode the TXDA
    runtime rejects kernel launches that it can otherwise record
    (``txLaunchKernel err=0x47000003``), which is why the V1 capture path
    uses ``"relaxed"``. Wrap whatever ``torch.cuda.graph`` currently is so the
    V2 capture path gets the same default without touching its call site.
    """
    current = torch.cuda.graph
    if getattr(current, "_fl_txda_relaxed_capture", False):
        return

    def graph(txda_graph, pool=None, stream=None, capture_error_mode="relaxed"):
        return current(
            txda_graph,
            pool=pool,
            stream=stream,
            capture_error_mode=capture_error_mode,
        )

    graph._fl_txda_relaxed_capture = True
    torch.cuda.graph = graph


def mapped_host_buffers_available() -> bool:
    """Whether this build can expose pinned host memory as a device view.

    Mirrors the T-Head probe: ``is_uva_available()`` only reports the
    pinned-memory capability, while the V2 buffers also need the memory to be
    *mapped*. Probe the op itself.
    """
    from vllm.utils.platform_utils import is_uva_available
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    if not is_uva_available():
        return False
    try:
        get_accelerator_view_from_cpu_tensor(
            torch.zeros(1, dtype=torch.int32, pin_memory=True)
        )
    except (RuntimeError, ValueError, NotImplementedError):
        # ``get_accelerator_view_from_cpu_tensor`` raises ValueError on the
        # PrivateUse1 build ("currently not supported in: txda"); accept every
        # "cannot map" spelling so a probe never aborts startup.
        logger.warning_once(
            "Pinned host memory cannot be mapped into the device address space "
            "by this vLLM build; the V2 runner will copy its UVA buffers "
            "explicitly instead."
        )
        return False
    return True


def _uva_unavailable() -> bool:
    """Stand-in for ``is_uva_available`` on builds without mapped host memory."""
    return False


def install_v2_runner_compat() -> None:
    """Adapt the upstream V2 runner internals that assume a CUDA build/runtime.

    Call before the V2 runner is constructed so that the buffers, graph
    capture and stream primitives are created with the adapted helpers.

    The body is transactional: it snapshots every global it is about to
    replace, and a failure restores them all so a retry starts clean. The
    completion flag is only set once every step succeeded; otherwise a partially
    applied install would make the next call return early and leave the process
    with mismatched buffers, graphs and kernels.
    """
    global _installed
    if _installed:
        return

    from vllm.v1.worker.gpu import buffer_utils, cudagraph_utils

    from vllm_fl.worker.model_runner import graph_capture as fl_graph_capture

    from . import block_table_native, mamba_align

    # Snapshot every global a step may replace, so a later failure can restore
    # them all. The sub-installers return their own undo, so their class-level
    # mutations are reverted too -- restoring only the ``_installed`` flags left
    # half the classes patched and made a retry return early.
    prev_is_uva = buffer_utils.is_uva_available
    prev_graph_capture = cudagraph_utils.graph_capture
    _undo_stack: list = []
    undo = None

    def _rollback() -> None:
        # Reverse order, so a step that depends on an earlier one is undone
        # first.
        while _undo_stack:
            step_undo = _undo_stack.pop()
            try:
                step_undo()
            except Exception:  # pragma: no cover - best-effort restoral
                logger.warning("V2 runner adaptation undo step failed", exc_info=True)
        buffer_utils.is_uva_available = prev_is_uva
        cudagraph_utils.graph_capture = prev_graph_capture

    try:
        install_cuda_aliases()

        # TXDA has no pinned/mapped host memory, so the V2 buffers always take
        # the explicit-copy fallback. Override only when the probe fails, so a
        # build that grows real UVA keeps the upstream path.
        if not mapped_host_buffers_available():
            buffer_utils.is_uva_available = _uva_unavailable

        # ``cudagraph_utils`` binds the process-group context manager at import
        # time; reuse the device-agnostic context that the V1 ``ModelRunnerFL``
        # resolves for this platform (FlagCX uses the accelerator stream APIs).
        if getattr(cudagraph_utils.graph_capture, "__name__", "") != getattr(
            fl_graph_capture, "__name__", None
        ):
            cudagraph_utils.graph_capture = fl_graph_capture

        # The "align" Mamba state kernels address memory through
        # runtime-loaded pointers, which the TXDA Triton backend miscompiles;
        # swap in native tensor copies. See mamba_align.py.
        undo = mamba_align.install_mamba_align_native()
        if undo is not None:
            _undo_stack.append(undo)

        # The per-group block-table gather and the per-token slot mapping have
        # the same defect; without them every token maps to KV slot 0.
        undo = block_table_native.install_block_table_native()
        if undo is not None:
            _undo_stack.append(undo)
    except Exception:
        _rollback()
        raise

    _installed = True
    logger.info("Installed V2 runner adaptations (non-UVA buffers, FL graph capture)")
