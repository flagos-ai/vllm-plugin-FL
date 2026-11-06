# Copyright (c) 2025 BAAI. All rights reserved.
"""txda runtime patches, mirroring the MUSA patch convention
(vendor/musa/patch.py): platform-specific workarounds that cannot live
upstream live here and are applied once at platform init.

``apply_txda_patches()`` is called from PlatformFL.import_kernels(), which
runs while `import vllm` is still in progress (vllm/_custom_ops.py calls it
from its module body). Only torch / torch_txda / vllm.utils patches are
safe here. Late-path patches that need a fully-imported vllm belong in
register_builtins() (register_ops.py), which the dispatch system loads
only after vllm is fully imported.

``capture_guard_context()`` is consumed by
vllm_fl.compilation.graph.GraphWrapper around cudagraph captures on txda
(see its docstring); importing it is safe from anywhere because this
module only imports torch / vllm.logger at module level.

IMPORTANT: this module must not import anything from `.impl` / `.patchs` —
those modules import vllm.model_executor / vllm._custom_ops, which are still
partially initialized when import_kernels() runs during `import vllm`
(circular import). They are imported from register_ops.py instead, which the
dispatch system loads only after vllm is fully imported.
"""

import contextlib
import os
import sys

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_patches_applied = False

# Reuse Stream wrapper objects per (device_index, stream_id): every
# torch.txda.current_stream() call constructs a new Stream object, and
# current_stream() is called per collective (e.g. flagcx all_reduce runs it
# for every layer of every eager forward). The wrapper is stateless around
# the handle, so reusing it for the same handle is safe — this mirrors the
# caching vLLM itself does on CUDA to avoid per-call Stream construction.
_stream_cache: dict[tuple[int, int], object] = {}
_STREAM_CACHE_MAX = 64


def _current_stream_txda(device=None):
    stream = torch.txda.current_stream(device)
    key = (stream.device_index, stream.stream_id)
    cached = _stream_cache.get(key)
    if cached is None:
        if len(_stream_cache) >= _STREAM_CACHE_MAX:
            _stream_cache.clear()
        _stream_cache[key] = stream
        return stream
    return cached


def patch_current_stream_for_txda():
    """Patch vllm.utils.torch_utils.current_stream to query txda live.

    vLLM's ``current_stream()`` caches the stream in thread-local storage and
    only refreshes it through ``torch.cuda.set_stream``. On txda torch is
    built without CUDA, so ``torch.cuda.set_stream`` raises and the cache is
    never updated: the first call resolves ``current_platform.current_stream``
    (which delegates to ``torch.txda.current_stream`` via
    ``Platform.__getattr__``) and caches that stream forever — the default
    stream at process start.

    During cudagraph capture vLLM switches to a dedicated capture stream via
    ``torch.txda.stream(...)`` which bypasses the cache, so any caller of
    ``current_stream()`` (e.g. the flagcx communicator, which launches every
    all_reduce on it) would use the stale default stream. Under
    ``capture_error_mode="relaxed"`` the collective is then silently launched
    outside the capture stream and never recorded into the graph — at replay
    the TP reduction is skipped and outputs are wrong.

    Replacing the function with a live ``torch.txda.current_stream()``
    lookup (plus wrapper-object reuse, see _stream_cache) makes every
    consumer observe the real current stream, including inside graph capture.
    """
    import torch_txda  # noqa: F401  # registers torch.txda

    import vllm.utils.torch_utils as torch_utils

    if getattr(torch_utils, "_fl_txda_stream_patched", False):
        return

    _orig_current_stream = torch_utils.current_stream

    torch_utils.current_stream = _current_stream_txda

    # `from vllm.utils.torch_utils import current_stream` binds the function
    # object at import time — rebind every module that already imported it
    # (pynccl, cuda_graph, ubatching, ray_communicator, our flagcx, ...).
    # Modules imported later pick up the patched attribute automatically.
    rebound = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            if mod.current_stream is _orig_current_stream:
                mod.current_stream = _current_stream_txda
                rebound.append(mod_name)
        except AttributeError:
            continue
    torch_utils._fl_txda_stream_patched = True
    logger.info(
        "Patched vllm.utils.torch_utils.current_stream -> torch.txda"
        ".current_stream (live lookup; rebound: %s)",
        rebound or "none",
    )


def patch_cuda_stream_apis_for_txda():
    """Delegate torch.cuda stream APIs to torch.txda.

    On txda torch is built without CUDA, so every ``torch.cuda`` stream
    primitive raises once touched. vLLM code that is not platform-aware
    (e.g. v1/worker/gpu/async_utils.py, shared_experts.py) still calls
    ``torch.cuda.stream/set_stream/current_stream`` — delegate them to the
    txda equivalents so those paths work instead of crashing. This mirrors
    musa's patch_cuda_stream_for_musa().
    """
    import torch_txda  # noqa: F401

    import torch.cuda as torch_cuda

    if getattr(torch_cuda, "_fl_txda_stream_patched", False):
        return

    import contextlib

    _orig_stream = torch_cuda.stream
    _orig_set_stream = torch_cuda.set_stream
    _orig_current_stream = torch_cuda.current_stream

    def _stream_txda(stream=None):
        if stream is None:
            return contextlib.nullcontext()
        return torch.txda.stream(stream)

    def _set_stream_txda(stream):
        torch.txda.set_stream(stream)

    def _cuda_current_stream_txda(device=None):
        return _current_stream_txda(device)

    torch_cuda.stream = _stream_txda
    torch_cuda.set_stream = _set_stream_txda
    torch_cuda.current_stream = _cuda_current_stream_txda
    torch_cuda._fl_txda_stream_patched = True
    # Keep references for potential un-patching / debugging.
    torch_cuda._fl_txda_orig = (_orig_stream, _orig_set_stream,
                                _orig_current_stream)
    logger.info("Patched torch.cuda stream APIs -> torch.txda for txda")


def patch_aux_stream_for_txda():
    """Opt-in: enable MoE shared-expert overlap via a dedicated txda stream.

    vLLM's ``aux_stream()`` only creates a stream on cuda-alike platforms,
    so on txda it returns None and the shared-expert/router overlap runs
    synchronously (``SharedExpertRunner._stream is None``). txda streams are
    functional (graph capture already runs on a dedicated stream), so when
    ``VLLM_FL_TXDA_AUX_STREAM=1`` we return a dedicated ``torch.txda.Stream``
    — the same overlap CUDA gets. Requires torch.cuda.stream delegation,
    applied by patch_cuda_stream_apis_for_txda().

    Default is off: ``aux_stream()`` returns None so SharedExperts runs on
    the current stream, same as MUSA. Verify on a MoE model before enabling.
    """
    import vllm.utils.torch_utils as torch_utils

    _orig_aux_stream = torch_utils.aux_stream
    enabled = os.environ.get("VLLM_FL_TXDA_AUX_STREAM", "0").lower() in (
        "1", "true", "on",
    )

    if enabled:
        import torch_txda  # noqa: F401

        _aux_stream = torch.txda.Stream()

        def _aux_stream_txda():
            return _aux_stream

        torch_utils._aux_stream = _aux_stream
    else:
        def _aux_stream_txda():
            return None

        torch_utils._aux_stream = None
    torch_utils.aux_stream = _aux_stream_txda

    # Rebind by-name importers already loaded; later importers pick up the
    # patched torch_utils.aux_stream directly.
    rebound = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            if mod.aux_stream is _orig_aux_stream:
                mod.aux_stream = _aux_stream_txda
                rebound.append(mod_name)
        except AttributeError:
            continue
    logger.info(
        "Patched aux_stream -> %s for txda (rebound: %s)",
        "dedicated torch.txda.Stream" if enabled else "None (SharedExperts sync path)",
        rebound or "none",
    )


def apply_txda_patches():
    """Apply the early-path txda runtime patches (idempotent).

    Called from PlatformFL.import_kernels(), which runs at vllm/_custom_ops
    import time — only patches that touch torch / torch_txda / already-loaded
    vllm.utils modules are safe here. Anything importing vllm.model_executor,
    vllm.compilation, ... belongs in register_ops.py instead, which the
    dispatch system loads only once vllm is fully imported
    (same convention as vendor/musa/patch.py).
    """
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True
    patch_current_stream_for_txda()
    patch_cuda_stream_apis_for_txda()
    patch_noop_device_guard_for_txda()
    patch_aux_stream_for_txda()


class _NoopDevice:
    """Drop-in no-op replacement for the torch.txda device guard class."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _noop_exchange_device(idx: int) -> int:
    """No-op _exchange_device: report the current device, switch nothing."""
    try:
        return torch.txda.current_device()
    except Exception:
        return idx


def patch_noop_device_guard_for_txda():
    """Perf: replace the torch.txda device guard with a no-op permanently.

    torch_txda's device context (torch.txda.device -> _exchange_device ->
    ExchangeDevice -> txSetDevice) resets the current stream to the device
    default stream and costs a driver round-trip on every enter/exit.
    FlagGems wraps *every* triton kernel launch in
    ``with torch_device_fn.device(...)``, so in eager mode each triton
    kernel pays that overhead (and silently resets the current stream —
    a live hazard for any code running on a non-default stream).

    vLLM workers are single-device processes: the guard can never
    legitimately switch devices, so a no-op is semantically equivalent.
    This is also what makes cudagraph capture work on txda (together with
    capture_guard_context); applying it globally removes the
    per-launch overhead in eager mode as well.

    Set VLLM_FL_TXDA_NOOP_DEVICE_GUARD=0 to restore the real device guard
    (e.g. when debugging driver-side stream behaviour).
    """
    if os.environ.get("VLLM_FL_TXDA_NOOP_DEVICE_GUARD", "1").lower() in (
        "0", "false", "off",
    ):
        return

    import torch_txda  # noqa: F401  # registers torch.txda

    if not hasattr(torch, "txda"):
        return
    if getattr(torch.txda, "_fl_txda_guard_neutralized", False):
        return

    torch.txda.device = _NoopDevice
    if hasattr(torch.txda, "_exchange_device"):
        torch.txda._exchange_device = _noop_exchange_device
    torch.txda._fl_txda_guard_neutralized = True
    logger.info(
        "Neutralized torch.txda device guard (no-op; single-device worker)")


@contextlib.contextmanager
def capture_guard_context():
    """Neutralize the torch.txda device guard for the duration of a capture.

    Consumed by vllm_fl.compilation.graph.GraphWrapper around cudagraph
    captures on txda. torch_txda's ExchangeDevice -> txSetDevice resets the
    current stream to the device's default stream. FlagGems wraps every
    triton kernel launch in `with torch_device_fn.device(...)`, so during
    graph capture triton kernels would be launched on the default stream
    instead of the capture stream, breaking the capture (txStreamEndCapture
    "Unknown error" / "Failed to txLaunchKernel"). vLLM workers are
    single-device processes, so no device switch can legitimately happen
    inside capture.

    Redundant with patch_noop_device_guard_for_txda() when that is active
    (the default), but kept so capture stays correct even when the global
    no-op guard is disabled via VLLM_FL_TXDA_NOOP_DEVICE_GUARD=0.
    """
    if not hasattr(torch, "txda"):
        yield
        return

    orig_device = torch.txda.device
    orig_exchange = getattr(torch.txda, "_exchange_device", None)

    torch.txda.device = _NoopDevice
    if orig_exchange is not None:
        torch.txda._exchange_device = _noop_exchange_device
    try:
        yield
    finally:
        torch.txda.device = orig_device
        if orig_exchange is not None:
            torch.txda._exchange_device = orig_exchange
