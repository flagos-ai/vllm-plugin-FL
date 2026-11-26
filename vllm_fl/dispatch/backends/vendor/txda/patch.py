# Copyright (c) 2026 BAAI. All rights reserved.
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
from torch.types import Device

from vllm.logger import init_logger

logger = init_logger(__name__)

_patches_applied = False

# Undo callables for every patch installed by ``apply_txda_patches``. The
# installer appends here as each patch succeeds, so a later failure can restore
# every already-replaced global and a retry starts from a clean slate.
_undo_stack: list[tuple[str, "callable"]] = []

# Original torch.txda guard objects, kept for rollback and capture scopes.
_orig_device_cls = None
_orig_exchange_device = None

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
        return None

    _orig_current_stream = torch_utils.current_stream

    torch_utils.current_stream = _current_stream_txda

    # `from vllm.utils.torch_utils import current_stream` binds the function
    # object at import time — rebind every module that already imported it
    # (pynccl, cuda_graph, ubatching, ray_communicator, our flagcx, ...).
    # Modules imported later pick up the patched attribute automatically.
    rebound = []

    def _undo() -> None:
        torch_utils.current_stream = _orig_current_stream
        torch_utils._fl_txda_stream_patched = False
        for mod_name in rebound:
            mod = sys.modules.get(mod_name)
            if mod is not None and (
                getattr(mod, "current_stream", None) is _current_stream_txda
            ):
                mod.current_stream = _orig_current_stream

    # Register the undo before mutating anything, so a module attribute that
    # raises something other than AttributeError still leaves the global
    # restorable instead of half-patched.
    try:
        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            try:
                if mod.current_stream is _orig_current_stream:
                    mod.current_stream = _current_stream_txda
                    rebound.append(mod_name)
            except AttributeError:
                continue
    except Exception:
        _undo()
        raise

    torch_utils._fl_txda_stream_patched = True
    logger.info(
        "Patched vllm.utils.torch_utils.current_stream -> torch.txda"
        ".current_stream (live lookup; rebound: %s)",
        rebound or "none",
    )
    return _undo


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
        return None

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

    def _undo() -> None:
        torch_cuda.stream = _orig_stream
        torch_cuda.set_stream = _orig_set_stream
        torch_cuda.current_stream = _orig_current_stream
        torch_cuda._fl_txda_stream_patched = False
        torch_cuda._fl_txda_orig = None

    return _undo


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
    _orig_aux_stream_attr = getattr(torch_utils, "_aux_stream", None)
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

    def _undo() -> None:
        torch_utils.aux_stream = _orig_aux_stream
        torch_utils._aux_stream = _orig_aux_stream_attr
        for mod_name in rebound:
            mod = sys.modules.get(mod_name)
            if mod is not None and getattr(mod, "aux_stream", None) is _aux_stream_txda:
                mod.aux_stream = _orig_aux_stream

    return _undo


def _preflight() -> None:
    """Fail before touching any global if a step's inputs are unusable.

    ``import_kernels`` calls this while ``import vllm`` is still running, so a
    missing ``torch.txda`` here means the platform was selected for the wrong
    backend; better to raise than to half-patch the process.
    """
    if not hasattr(torch, "txda"):
        raise RuntimeError(
            "TXDA runtime patches require torch.txda; is torch_txda installed "
            "and registered for this platform?"
        )


def rollback_txda_patches() -> None:
    """Undo every patch applied so far, newest first."""
    global _patches_applied
    while _undo_stack:
        name, undo = _undo_stack.pop()
        try:
            undo()
        except Exception:  # pragma: no cover - best-effort restore
            logger.exception("Failed to roll back txda patch '%s'", name)
    _patches_applied = False


def apply_txda_patches():
    """Apply the early-path txda runtime patches (idempotent, rollback-safe).

    Called from PlatformFL.import_kernels(), which runs at vllm/_custom_ops
    import time — only patches that touch torch / torch_txda / already-loaded
    vllm.utils modules are safe here. Anything importing vllm.model_executor,
    vllm.compilation, ... belongs in register_ops.py instead, which the
    dispatch system loads only once vllm is fully imported
    (same convention as vendor/musa/patch.py).

    Every step records how to undo itself, so a failure halfway through
    restores all previously replaced globals instead of leaving the process
    half-patched; the completion flag is only set once every step succeeded,
    which keeps a retry working.
    """
    global _patches_applied
    if _patches_applied:
        return

    _preflight()
    try:
        for name, patch in _PATCH_STEPS:
            undo = patch()
            if undo is not None:
                _undo_stack.append((name, undo))
    except Exception:
        rollback_txda_patches()
        raise
    _patches_applied = True



def _txda_get_device_index(device: Device) -> int:
    """Resolve a device spec to a TXDA index.

    Mirrors ``torch.txda.device.__init__``: ``optional=True`` keeps
    ``"txda"`` (no index) and ``None`` valid, which FlagGems passes when the
    target is simply "the current device".
    """
    from torch_txda.txda import _get_device_index

    idx = _get_device_index(device, optional=True)
    if idx is None or idx < 0:
        try:
            return int(torch.txda.current_device())
        except Exception:
            return 0
    return idx


class _CurrentDeviceGuard:
    """Device guard that skips the driver call when already on ``idx``.

    ``torch.txda.device`` is a real context manager: FlagCX's abort/cleanup
    threads and any explicit cross-device call rely on it actually selecting
    the device. What is expensive is the unconditional
    ``txSetDevice`` round-trip (plus the stream reset) that FlagGems pays on
    every Triton kernel launch, even though workers are single-device
    processes and the device is almost always already current.

    So instead of neutering the guard, keep the switch but make it a no-op
    when the current device already matches. Cross-device use stays correct.
    """

    def __init__(self, device=None):
        self.idx = int(_txda_get_device_index(device))
        self._inner = None
        self._switched = False

    def __enter__(self):
        try:
            current = torch.txda.current_device()
        except Exception:
            current = None
        if current == self.idx:
            # Already on the target device: no driver round-trip, no stream
            # reset.
            return None
        # Must use the class captured before patching: ``torch.txda.device``
        # may already be this guard, which would recurse forever.
        self._inner = _orig_device_cls(self.idx)
        self._switched = True
        return self._inner.__enter__()

    def __exit__(self, *args):
        if not self._switched or self._inner is None:
            return False
        return self._inner.__exit__(*args)


def _exchange_device_skip_current(idx: int) -> int:
    """``_exchange_device`` that avoids the driver call on a no-op switch."""
    try:
        current = torch.txda.current_device()
        # ``idx`` arrives as a tensor from some call sites; normalize both
        # sides so the comparison is never a tensor-vs-int bool.
        idx_int = int(idx)
        current_int = int(current)
    except Exception:
        return _orig_exchange_device(idx)
    if idx_int < 0 or current_int == idx_int:
        return current
    return _orig_exchange_device(idx)


def patch_noop_device_guard_for_txda():
    """Make the txda device guard a no-op *only* when already on the device.

    torch_txda's guard (``torch.txda.device`` -> ``_exchange_device`` ->
    ``ExchangeDevice`` -> ``txSetDevice``) resets the current stream to the
    device default stream and costs a driver round-trip on every enter/exit.
    FlagGems wraps *every* Triton kernel launch in
    ``with torch_device_fn.device(...)``, so eager mode pays that on every
    launch (and silently drops to the default stream).

    Workers are single-device processes, so in the common case the target
    device is already current and the switch has no effect; skipping just
    that case removes the overhead while leaving genuine device switches —
    including FlagCX's per-device cleanup threads — intact.

    Set VLLM_FL_TXDA_NOOP_DEVICE_GUARD=0 to keep the upstream guard verbatim.
    """
    global _orig_exchange_device, _orig_device_cls

    if os.environ.get("VLLM_FL_TXDA_NOOP_DEVICE_GUARD", "1").lower() in (
        "0", "false", "off",
    ):
        return None

    import torch_txda  # noqa: F401  # registers torch.txda

    if not hasattr(torch, "txda"):
        return None
    if getattr(torch.txda, "_fl_txda_guard_patched", False):
        return None

    _orig_device_cls = torch.txda.device
    _orig_exchange_device = getattr(torch.txda, "_exchange_device", None)

    torch.txda.device = _CurrentDeviceGuard
    if _orig_exchange_device is not None:
        torch.txda._exchange_device = _exchange_device_skip_current
    torch.txda._fl_txda_guard_patched = True
    logger.info(
        "Patched torch.txda device guard (skips the driver call when the "
        "target device is already current; cross-device switches preserved)")

    def _undo() -> None:
        torch.txda.device = _orig_device_cls
        if _orig_exchange_device is not None:
            torch.txda._exchange_device = _orig_exchange_device
        torch.txda._fl_txda_guard_patched = False

    return _undo


def patch_torch_compile_off_for_txda():
    """Make ``torch.compile`` a no-op on TXDA.

    vLLM decorates several helpers with ``@torch.compile`` at import time,
    using ``simple_compile_backend`` (``"eager"`` on TXDA). "eager" only skips
    Inductor code generation; Dynamo still traces the function and emits an FX
    graph. On TXDA that graph is wrong: it inserts ``_to_copy(..., device="cpu")``
    for intermediates, so ``get_masked_input_and_mask`` turns its int64 masks
    into host tensors. Under cudagraph capture those D2H copies synchronize the
    capture stream and the capture is aborted:

        get_masked_input_and_mask -> _to_copy(device=cpu) -> txStreamSynchronize
        while the stream is capturing

    Tracing buys nothing on TXDA (the backend is eager either way), so replace
    ``torch.compile`` with the identity decorator. Modules decorated before this
    runs keep their wrapper; callers that import them later get the plain
    function. Set VLLM_FL_TXDA_TORCH_COMPILE=1 to keep the upstream behaviour.
    """
    if os.environ.get("VLLM_FL_TXDA_TORCH_COMPILE", "0").lower() in (
        "1", "true", "on",
    ):
        return None

    # Unlike the other steps this one only touches ``torch``, so it has no
    # implicit ``import torch_txda`` guard. Require the TXDA platform
    # explicitly: ``torch.compile`` is process-global, and neutering it on a
    # CUDA/NPU host would silently break every other backend's Dynamo tests
    # that share the process (``test_musa_gems_compile`` and friends).
    #
    # ``FL_TEST_PLATFORM`` is the variable ``tests/run.py`` exports from
    # ``--platform``. It lets a TXDA host run another platform's test selection
    # without this step leaking into their ``torch.compile`` assertions.
    from vllm.platforms import current_platform

    if current_platform.device_type != "txda":
        return None
    test_platform = os.environ.get("FL_TEST_PLATFORM", "").strip().lower()
    if test_platform and test_platform != "txda":
        return None

    if getattr(torch, "_fl_txda_compile_disabled", False):
        return None

    original_compile = torch.compile

    def _identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            return lambda f: f
        return fn

    torch.compile = _identity_compile
    torch._fl_txda_compile_disabled = True
    logger.info(
        "Disabled torch.compile on TXDA (Dynamo graphs insert CPU copies that "
        "break graph capture)"
    )

    def _undo() -> None:
        torch.compile = original_compile
        torch._fl_txda_compile_disabled = False

    return _undo


@contextlib.contextmanager
def capture_guard_context():
    """Keep TXDA kernels on the capture stream for the duration of a capture.

    Consumed by vllm_fl.compilation.graph.GraphWrapper around cudagraph
    captures on txda. torch_txda's ExchangeDevice -> txSetDevice resets the
    current stream to the device's default stream. FlagGems wraps every
    Triton kernel launch in ``with torch_device_fn.device(...)``, so during
    graph capture triton kernels would be launched on the default stream
    instead of the capture stream, breaking the capture (txStreamEndCapture
    "Unknown error" / "Failed to txLaunchKernel").

    Only the no-op-device case is skipped here (the target is already the
    capture device); a real switch still goes through the SDK guard so a
    capture is never silently redirected to another device.

    Redundant with patch_noop_device_guard_for_txda() when that is active
    (the default), but kept so capture stays correct even when the global
    guard patch is disabled via VLLM_FL_TXDA_NOOP_DEVICE_GUARD=0.
    """
    if not hasattr(torch, "txda"):
        yield
        return

    orig_device = torch.txda.device
    orig_exchange = getattr(torch.txda, "_exchange_device", None)

    torch.txda.device = _CurrentDeviceGuard
    if orig_exchange is not None:
        torch.txda._exchange_device = _exchange_device_skip_current
    try:
        yield
    finally:
        torch.txda.device = orig_device
        if orig_exchange is not None:
            torch.txda._exchange_device = orig_exchange


_PATCH_STEPS = (
    ("current_stream", patch_current_stream_for_txda),
    ("cuda_stream_apis", patch_cuda_stream_apis_for_txda),
    ("device_guard", patch_noop_device_guard_for_txda),
    ("aux_stream", patch_aux_stream_for_txda),
    ("torch_compile", patch_torch_compile_off_for_txda),
)
