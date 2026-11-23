# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU platform compatibility patches (merged).

Four independent firmware/driver workarounds — ``device_properties``,
``events``, ``scatter``, ``bind_kv_cache`` — each a disable-able ``PatchSpec``,
all installed by ``apply_lpu_platform_compat_patch()`` (idempotent).

"""

from __future__ import annotations

import torch

_PATCHED = False


# ---- device_properties ----


def _platform_device_properties() -> None:
    """Add ``multi_processor_count=1`` to ``_LPUDeviceProperties``.

    LPU driver only exposes ``name`` + ``total_memory``.  vLLM code reads
    ``device_properties.multi_processor_count`` for internal scheduling
    decisions (e.g. ``num_scheduler_steps``).  Without this attribute,
    it raises ``AttributeError``.
    """
    import torch_lpu._C

    if not hasattr(torch_lpu._C._LPUDeviceProperties, "multi_processor_count"):
        torch_lpu._C._LPUDeviceProperties.multi_processor_count = 1


# ---- events ----


def _patch_event_synchronize() -> None:
    """Monkey-patch torch.Event on LPU.

    LPU's native Event.synchronize() / record() / wait() throw "Backend
    doesn't support synchronizing events" because the LPU firmware/driver
    doesn't implement zrtlEventSynchronize (and related).  Workaround:
    fall back to torch.lpu.synchronize() (global device sync), which the
    LPU runtime does support.  This is more conservative than per-event
    sync but is the closest correct behavior.

    Since torch.Event is a C extension type (immutable), we cannot set
    attributes on it directly.  Instead, we replace ``torch.Event`` with a
    Python subclass that overrides the three failing methods.  All existing
    references to ``torch.Event`` (e.g. in vLLM's model_runner) will pick
    up the patched class because they resolve ``torch.Event`` at call time.
    """
    _OrigEvent = torch.Event

    def _event_err_match(e: RuntimeError) -> bool:
        """True if a torch.Event RuntimeError is an LPU event-unsupported error
        that we should swallow (fall back), False if it is an unrelated error
        that must propagate.

        Matches both known LPU rejection messages:
        * ``Backend doesn't support synchronizing events`` (TP1: single device,
          single stream — the LPU runtime lacks zrtl event/stream-sync
          primitives);
        * ``record:CopyKernel.cpp ... evError_t=2181 ... operation not
          supported`` (TP4/multi-device: LPUEvent.record() on a second device /
          stream fails with evError 2181).  Only ``record()`` raises 2181 —
          ``synchronize()``/``wait()`` on LPU raise the "synchronizing events"
          message — but the match is kept uniform so a future driver change
          can't silently re-crash.
        """
        msg = str(e)
        return (
            "doesn't support synchronizing events" in msg
            or "operation not supported" in msg
        )

    class _LPUEvent(_OrigEvent):
        """torch.Event subclass with LPU-safe event methods."""

        def synchronize(self):
            try:
                return super().synchronize()
            except RuntimeError as e:
                if not _event_err_match(e):
                    raise
                torch.lpu.synchronize()

        def record(self, stream=None):
            try:
                return super().record(stream)
            except RuntimeError as e:
                if not _event_err_match(e):
                    raise
                # record() rejected -> no-op.  The synchronize() fallback
                # (global device sync) in get_output() is sufficient
                # because it drains all streams.
                return None

        def wait(self, stream=None):
            try:
                return super().wait(stream)
            except RuntimeError as e:
                if not _event_err_match(e):
                    raise
                # wait() rejected -> no-op.  Same reasoning as record().
                return None

    # Replace torch.Event globally so that any code doing
    # ``torch.Event()`` gets the patched subclass.
    torch.Event = _LPUEvent


def _patch_accelerator_synchronize() -> None:
    """Monkey-patch torch.accelerator.synchronize on LPU.

    torch.accelerator.synchronize() delegates to the C++ function
    ``_accelerator_synchronizeDevice``, which throws "Backend doesn't support
    synchronizing all streams on device" on LPU.  torch.lpu.synchronize()
    bypasses the accelerator framework and calls the LPU runtime directly,
    which does work.

    We patch torch.accelerator.synchronize to fall back to
    torch.lpu.synchronize() when the native call fails with this error.
    """
    _orig_sync = torch.accelerator.synchronize

    def _lpu_accelerator_synchronize(device=None):
        try:
            return _orig_sync(device)
        except RuntimeError as e:
            if "synchronizing all streams" not in str(e):
                raise
            torch.lpu.synchronize()

    torch.accelerator.synchronize = _lpu_accelerator_synchronize


def _patch_accelerator_empty_cache() -> None:
    """Route torch.accelerator.empty_cache to the LPU-native empty_cache.

    vLLM calls ``torch.accelerator.empty_cache()`` (no-arg) when resizing a
    workspace buffer (vllm/v1/worker/workspace.py), around the memory-profile /
    KV-budget snapshots (vllm/utils/mem_utils.py, gpu_worker.py), and at several
    other best-effort memory-flush sites.  The generic ``torch.accelerator.
    empty_cache`` on torch >= 2.10 would trip a C++ assert (``Allocator for lpu
    is not a DeviceAllocator``) because the LPU allocator is not a
    ``torch._C.DeviceAllocator``.  The native ``torch_lpu.lpu.empty_cache``
    releases cached (reserved-but-unallocated) caching-allocator segments back
    to the driver and is safe; routing to it is required because vLLM relies on
    it to return cache fragments (workspace.py explicitly frees the old
    workspace before growing).  Measured: the call is reachable in the worker
    (workspace resize released 2.41 GiB) and the accounting probe's own
    ``torch_lpu.lpu.empty_cache()`` released 6.72 GiB of load-peak cache
    (38.52 -> 31.79 GiB reserved) -- but the memory_profiling entry/exit
    empty_cache calls never reach the worker's patched function (worker-side
    call log shows only the workspace resize and shutdown), so the load cache
    stays unflushed until the first manual flush, and the DSV4 fp8 4-layer TP1
    decode OOM persists: the lm_head chunked einsum needs a large contiguous
    transient that a fragmented 8.95 GiB-free driver heap cannot provide.
    """

    def _lpu_empty_cache() -> None:
        try:
            import torch_lpu

            torch_lpu.lpu.empty_cache()
        except Exception:  # noqa: BLE001  (flush failure must not break inference)
            return

    torch.accelerator.empty_cache = _lpu_empty_cache


def _platform_events() -> None:
    """Apply all LPU event/accelerator compatibility patches.

    Covers torch.Event (synchronize/record/wait) and
    torch.accelerator.synchronize/empty_cache via a single unified mechanism.
    """
    _patch_event_synchronize()
    _patch_accelerator_synchronize()
    _patch_accelerator_empty_cache()


# ---- scatter ----


def _platform_scatter() -> None:
    """Monkey-patch torch.Tensor.scatter_ for LPU int64 index tolerance.

    The LPU backend rejects ``scatter_`` with int64 index tensors:
    ``"scatter expected dtype int32 for index"``.  We transparently cast
    int64 indices to int32 on LPU devices so upstream code can write
    ``torch.int64`` without a per-platform branch.
    """
    _orig = torch.Tensor.scatter_

    def _lpu_scatter_(self, dim, index, src, *, reduce=None):
        if index.device.type == "lpu" and index.dtype == torch.int64:
            index = index.to(torch.int32)
        if reduce is None:
            return _orig(self, dim, index, src)
        return _orig(self, dim, index, src, reduce=reduce)

    torch.Tensor.scatter_ = _lpu_scatter_


# ---- bind_kv_cache ----


_ORIGINAL_BIND_KV_CACHE = None


def _bind_kv_cache_patched(
    kv_caches, forward_context, runner_kv_caches, num_attn_module=1,
):
    """Patched bind_kv_cache that adds LPU to the multi-layer allowlist."""
    from collections import defaultdict
    from vllm.v1.worker.utils import extract_layer_index
    from vllm.platforms import current_platform

    assert len(runner_kv_caches) == 0

    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(
            layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            if (
                current_platform.is_cuda_alike()
                or current_platform.is_xpu()
                or current_platform.is_cpu()
                or current_platform.device_type == "lpu"
            ):
                pass
            else:
                raise NotImplementedError
        for layer_name in layer_names:
            runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache


def _platform_bind_kv_cache() -> None:
    """Replace bind_kv_cache with LPU-aware version."""
    global _ORIGINAL_BIND_KV_CACHE
    import vllm.v1.worker.utils as _utils
    _ORIGINAL_BIND_KV_CACHE = _utils.bind_kv_cache
    _utils.bind_kv_cache = _bind_kv_cache_patched


def _platform_cuda_stream() -> None:
    """Redirect torch.cuda.Stream to torch.lpu.Stream on LPU.

    Upstream model code (e.g. DeepseekV4Model) unconditionally creates
    ``torch.cuda.Stream()`` objects.  On LPU the CUDA driver is incompatible
    with the CUDA runtime, so ``torch.cuda.Stream()`` raises:

        RuntimeError: CUDA error: CUDA driver version is insufficient

    ``torch.lpu.Stream()`` works on LPU.  By redirecting the name at the
    module level, all upstream ``torch.cuda.Stream()`` call sites work
    unmodified — no model-level ``if lpu: aux_stream_list = None`` gate
    needed.

    The LPU Stream class inherits from ``torch.Stream`` (not
    ``torch.cuda.Stream``) but the name rebinding at ``torch.cuda.Stream``
    is sufficient because all callers resolve the name at call time.
    """
    torch.cuda.Stream = torch.lpu.Stream


# ---- torch.compile ----


_ORIGINAL_TORCH_COMPILE = None


def _lpu_torch_compile(*args, **kwargs):
    """Identity decorator: return the callable unchanged (no-op compile)."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        # bare ``@torch.compile`` -> return the function itself
        return args[0]
    # ``@torch.compile(...)`` -> return an identity decorator
    return lambda fn: fn


def _platform_torch_compile() -> None:
    """Disable torch.compile on LPU (no-op).

    torch.compile always runs the dynamo tracing / fake-tensor pass before
    the backend executes.  On LPU the platform already runs everything eager
    (``simple_compile_backend="eager"``, enforce_eager, no CUDAGraphs), so
    tracing buys nothing -- but under dynamic/symbolic shapes (enabled by
    newer torch + vLLM's model-runner context) the fake-tensor pass calls
    LPU ops like ``reshape`` with SymInt sizes, which torch_lpu rejects::

        SymIntArrayRef expected to contain only concrete integers

    Upstream model code (e.g. ``DeepseekV4ForCausalLM.hc_head``) carries raw
    ``@torch.compile(...)`` decorators that vLLM's enforce_eager does NOT
    disable, so we neutralize ``torch.compile`` itself for LPU.  The
    decorated function simply runs eagerly, which is what the platform
    already intends.
    """
    global _ORIGINAL_TORCH_COMPILE
    if _ORIGINAL_TORCH_COMPILE is None:
        _ORIGINAL_TORCH_COMPILE = torch.compile
        torch.compile = _lpu_torch_compile


# ---- Public entry point ----
# Application order matters:
# bind_kv_cache MUST come before events: events imports model_runner which
# captures bind_kv_cache at module level (as of vLLM v1.x).


def apply_lpu_platform_compat_patch() -> None:
    """Apply all LPU platform compatibility patches.  Idempotent.

    Order: device_properties -> scatter -> bind_kv_cache -> events ->
    cuda_stream -> torch_compile.  bind_kv_cache before events is required
    (see comment above).
    """
    global _PATCHED
    if _PATCHED:
        return
    _platform_device_properties()
    _platform_scatter()
    _platform_bind_kv_cache()
    _platform_events()
    _platform_cuda_stream()
    _platform_torch_compile()
    _PATCHED = True
