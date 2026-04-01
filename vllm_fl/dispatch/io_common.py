# Copyright (c) 2026 BAAI. All rights reserved.

"""
Shared utilities for IO dumper.

Provides common formatting, filtering, feature detection,
re-entrancy guard, execution order tracking, and YAML config
parsing used by io_dumper.py.
"""

from __future__ import annotations

import fnmatch
import functools
import inspect
import logging
import os
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)

# ── Feature detection (done once at import) ──

try:
    from torch.overrides import TorchFunctionMode
    HAS_TORCH_FUNC_MODE = True
except ImportError:
    TorchFunctionMode = None  # type: ignore[misc,assignment]
    HAS_TORCH_FUNC_MODE = False

try:
    from torch.utils._python_dispatch import TorchDispatchMode  # type: ignore[attr-defined]
    HAS_TORCH_DISPATCH_MODE = True
except ImportError:
    TorchDispatchMode = None  # type: ignore[misc,assignment]
    HAS_TORCH_DISPATCH_MODE = False


_io_active: bool = False


def set_io_active(active: bool) -> None:
    """Set the IO-active flag used by managed_inference_mode.

    When True, managed_inference_mode uses torch.no_grad() instead of
    torch.inference_mode() so that TorchDispatchMode can intercept ops.
    """
    global _io_active
    _io_active = active


def is_io_active() -> bool:
    """Return True if the IO dumper is active."""
    return _io_active


# ── Shared dispatch / function mode lifecycle ──


class ModeManager:
    """LIFO-safe manager for PyTorch TorchDispatchMode / TorchFunctionMode.

    The dumper pushes mode instances
    onto PyTorch's global mode stack.  PyTorch requires these to be exited
    in strict LIFO (reverse-entry) order — exiting a mode that is not on
    top of the stack corrupts internal state and segfaults.

    ``ModeManager`` centralises entry/exit of all IO modes so that:
    * Each subsystem calls ``enter(name, instance)`` / ``request_exit(name)``.
    * ``request_exit`` marks the mode as *pending*.  If it is on top of the
      stack it is exited immediately; otherwise it stays entered (the
      ``__torch_dispatch__`` / ``__torch_function__`` handler short-circuits
      via ``_enabled=False``).
    * After every exit the manager *drains* — repeatedly exiting the new
      top of the stack while it also has a pending exit.  This ensures that
      disabling the *last* subsystem always cleans up the entire stack
      regardless of disable order.

    One ``ModeManager`` instance is created for TorchDispatchMode and
    another for TorchFunctionMode.
    """

    def __init__(self) -> None:
        self._stack: list[tuple[str, Any]] = []   # (name, instance), bottom-first
        self._pending: set[str] = set()
        self._lock = threading.Lock()

    # ── public API ──

    def enter(self, name: str, mode_instance: Any) -> None:
        """Enter *mode_instance* under *name* (idempotent)."""
        with self._lock:
            if any(n == name for n, _ in self._stack):
                return  # already entered
            mode_instance.__enter__()
            self._stack.append((name, mode_instance))

    def request_exit(self, name: str) -> None:
        """Mark *name* for exit and drain from the top while possible."""
        with self._lock:
            if not any(n == name for n, _ in self._stack):
                return
            self._pending.add(name)
            self._drain_top()

    def exit_all(self) -> None:
        """Unconditionally exit every mode in reverse LIFO order."""
        with self._lock:
            while self._stack:
                _name, inst = self._stack.pop()
                inst.__exit__(None, None, None)
            self._pending.clear()

    def is_entered(self, name: str) -> bool:
        """True if *name* has an active (entered) mode on the stack."""
        return any(n == name for n, _ in self._stack)

    # ── internals ──

    def _drain_top(self) -> None:
        """Pop and exit modes from the top while they have a pending exit."""
        while self._stack and self._stack[-1][0] in self._pending:
            name, inst = self._stack.pop()
            self._pending.discard(name)
            inst.__exit__(None, None, None)


dispatch_mode_mgr = ModeManager()
func_mode_mgr = ModeManager()


def managed_inference_mode():
    """Drop-in replacement for ``@torch.inference_mode()``.

    When IO dispatch modes are enabled, downgrades to ``torch.no_grad()``
    so that CompositeImplicitAutograd ops (e.g. ``aten.linear``) decompose
    into leaf ops (``aten.mm``, ``aten.t``) visible to TorchDispatchMode.

    ``torch.no_grad()`` (vs removing inference mode entirely) is critical:
    it prevents autograd from building computation graphs for every op,
    which would otherwise exhaust GPU memory and cause segfaults.

    When IO is not enabled, behaves identically to
    ``@torch.inference_mode()``.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if is_io_active():
                with torch.no_grad():
                    return fn(*args, **kwargs)
            with torch.inference_mode():
                return fn(*args, **kwargs)

        return wrapper

    return decorator


# ── Distributed rank ──

_rank: Optional[int] = None  # cached after first call


def get_rank() -> int:
    """Get the current distributed rank.

    Resolution order:
    1. ``torch.distributed.get_rank()`` if the default process group is initialized
    2. ``RANK`` environment variable
    3. ``LOCAL_RANK`` environment variable
    4. 0 (single-process fallback)

    The result is cached once a definitive source (dist or env var) is found.
    The fallback ``0`` is never cached so that later calls can pick up a
    process group that was initialized after the first call.
    """
    global _rank
    if _rank is not None:
        return _rank
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            _rank = dist.get_rank()
            return _rank
    except Exception:
        pass
    for var in ("RANK", "LOCAL_RANK"):
        val = os.environ.get(var)
        if val is not None:
            try:
                _rank = int(val)
                return _rank
            except ValueError:
                pass
    return 0


def reset_rank() -> None:
    """Reset cached rank (for testing only)."""
    global _rank
    _rank = None


def parse_rank_filter(value: str) -> Optional[Set[int]]:
    """Parse a rank filter string.

    Args:
        value: ``"all"`` or ``""`` for all ranks, or comma-separated
            rank numbers like ``"0"`` or ``"0,2,4"``.

    Returns:
        None for all ranks, or a set of rank integers.
    """
    value = value.strip().lower()
    if not value or value == "all":
        return None
    ranks: Set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if token:
            try:
                ranks.add(int(token))
            except ValueError:
                pass
    return ranks if ranks else None


# ── Re-entrancy guard (per-caller, per-thread) ──
#
# Each subsystem creates its own guard via
# ``make_guard()`` so they do not block each other.  The guard is
# thread-local to prevent cross-thread interference.


def make_guard():
    """Create an independent re-entrancy guard (thread-local).

    Returns ``(guard_active, set_guard)`` functions.  Each caller gets
    its own guard so different subsystems do not interfere.
    """
    _tls = threading.local()

    def guard_active() -> bool:
        return getattr(_tls, "active", False)

    def set_guard(active: bool) -> None:
        _tls.active = active

    return guard_active, set_guard


# ── Execution order tracking ──

_exec_order_counter: int = 0
_exec_order_lock = threading.Lock()


def next_exec_order() -> int:
    """Get the next global execution order number (thread-safe).

    This provides a monotonically increasing counter across all
    intercepted ops, allowing users to align output with model
    definition order.
    """
    global _exec_order_counter
    with _exec_order_lock:
        _exec_order_counter += 1
        return _exec_order_counter


def reset_exec_order() -> None:
    """Reset the execution order counter (e.g. at step boundaries)."""
    global _exec_order_counter
    with _exec_order_lock:
        _exec_order_counter = 0


def get_exec_order() -> int:
    """Get the current execution order counter value (without incrementing)."""
    return _exec_order_counter


# ── Per-step counters ──
#
# module counter: each unique module class name gets a monotonic index (0,1,2,…)
# and a per-class call count, both reset each step.
# op counter: same for ops/torch funcs.
# seen sets: track which modules/ops were encountered this step (for summary).

_module_type_index: Dict[str, int] = {}
_module_type_count: Dict[str, int] = {}
_module_next_idx: int = 0

_op_type_index: Dict[str, int] = {}
_op_type_count: Dict[str, int] = {}
_op_next_idx: int = 0

_seen_modules: Set[str] = set()
_seen_ops: Set[str] = set()

_counter_lock = threading.Lock()


def next_module_counter(cls_name: str) -> Tuple[int, int]:
    """Get (type_index, call_count) for a module class this step, incrementing call count."""
    global _module_next_idx
    with _counter_lock:
        if cls_name not in _module_type_index:
            _module_type_index[cls_name] = _module_next_idx
            _module_next_idx += 1
            _module_type_count[cls_name] = 0
        _module_type_count[cls_name] += 1
        return _module_type_index[cls_name], _module_type_count[cls_name]


def next_op_counter(op_name: str) -> Tuple[int, int]:
    """Get (type_index, call_count) for an op this step, incrementing call count."""
    global _op_next_idx
    with _counter_lock:
        if op_name not in _op_type_index:
            _op_type_index[op_name] = _op_next_idx
            _op_next_idx += 1
            _op_type_count[op_name] = 0
        _op_type_count[op_name] += 1
        return _op_type_index[op_name], _op_type_count[op_name]


def _reset_per_step_counters() -> Tuple[Set[str], Set[str]]:
    """Reset module/op counters and seen sets for a new step.

    Returns:
        (seen_modules, seen_ops) from the completed step.
    """
    global _module_next_idx, _op_next_idx
    with _counter_lock:
        seen_modules = _seen_modules.copy()
        seen_ops = _seen_ops.copy()
        _seen_modules.clear()
        _seen_ops.clear()
        _module_type_index.clear()
        _module_type_count.clear()
        _module_next_idx = 0
        _op_type_index.clear()
        _op_type_count.clear()
        _op_next_idx = 0
    return seen_modules, seen_ops


# ── Step tracking ──

_step_counter: int = 0  # first forward pass runs at step 0
_step_lock = threading.Lock()
_step_callbacks: List[Any] = []  # called on each step advance


def get_step() -> int:
    """Get the current step counter value."""
    return _step_counter


def advance_step() -> int:
    """Increment the step counter and reset execution order (thread-safe).

    Returns the new step number.  Also resets per-step counters and
    fires registered step callbacks with the summary of the completed step.
    """
    global _step_counter
    with _step_lock:
        prev_step = _step_counter
        _step_counter += 1
        result = _step_counter
        callbacks = list(_step_callbacks)
    reset_exec_order()
    seen_modules, seen_ops = _reset_per_step_counters()
    for cb in callbacks:
        try:
            cb(prev_step, seen_modules, seen_ops)
        except Exception:
            _logger.warning("Step callback %s failed", cb.__name__, exc_info=True)
    return result


def reset_step() -> None:
    """Reset the step counter to 0 (for testing)."""
    global _step_counter
    with _step_lock:
        _step_counter = 0
    _reset_per_step_counters()


def register_step_callback(cb) -> None:
    """Register a callback to be called on each step advancement.

    Callbacks receive (step, seen_modules, seen_ops) for the completed step.
    """
    with _step_lock:
        if cb not in _step_callbacks:
            _step_callbacks.append(cb)


def unregister_step_callback(cb) -> None:
    """Remove a previously registered step callback."""
    with _step_lock:
        try:
            _step_callbacks.remove(cb)
        except ValueError:
            pass


def parse_step_range(value) -> Optional[Tuple[int, int]]:
    """Parse a step range string into a half-open tuple for internal use.

    Accepted formats (all inclusive):
    - ``"start-end"``: ``"0-2"`` → ``(0, 3)`` (steps 0, 1, 2)
    - Bare integer: ``"5"`` → ``(5, 6)`` (step 5 only)

    Returns ``None`` if the value is ``None``, empty, or cannot be parsed.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    if "-" in value:
        parts = value.split("-", 1)
        try:
            return (int(parts[0].strip()), int(parts[1].strip()) + 1)
        except ValueError:
            return None
    elif value.isdigit():
        s = int(value)
        return (s, s + 1)
    return None


def parse_step_range_env(*env_vars: str) -> Optional[Tuple[int, int]]:
    """Parse a step range from the first non-empty env var.

    Checks each env var in order; returns the first valid ``(start, end)``
    half-open tuple or ``None``.  Accepts ``"start-end"`` inclusive dash
    format (e.g. ``"0-1"`` → ``(0, 2)``) or a bare integer (e.g. ``"0"``
    → ``(0, 1)``).
    """
    for var in env_vars:
        value = os.environ.get(var, "").strip()
        if not value:
            continue
        result = parse_step_range(value)
        if result is not None:
            return result
    return None


# ── Module path registry ──
#
# Maps ``id(module)`` → full dotted path (e.g. ``"model.layers.0.self_attn"``).
# Populated by ``register_module_paths(model)`` after the model is loaded.
# Used by ``push_module_context`` and ``layer_path_matches`` for layer filtering.

_module_path_map: Dict[int, str] = {}
_module_path_lock = threading.Lock()


def register_module_paths(model: torch.nn.Module) -> None:
    """Build ``id(module) → path`` mapping from a model's named_modules.

    Call after the model is loaded (e.g. in ``load_model()``) so that
    global module hooks can look up each module's full path.
    """
    with _module_path_lock:
        for name, mod in model.named_modules():
            _module_path_map[id(mod)] = name


def get_module_path(module: torch.nn.Module) -> str:
    """Look up the full dotted path of a module, or empty string."""
    return _module_path_map.get(id(module), "")


# ── Module context tracking (per-thread) ──
#
# When module filtering is active, global module hooks push/pop module
# class names onto a thread-local stack.  This lets op hooks and
# TorchFunctionMode know which module(s) are currently executing,
# enabling "only log ops/torch_funcs within these modules" semantics.
#
# The stack stores (cls_name, module_type_idx, module_call_count) tuples
# so that op hooks can include the enclosing module's counter in output.

_module_context = threading.local()


def push_module_context(cls_name: str, module: Optional[torch.nn.Module] = None) -> None:
    """Push a module class name onto the current-module stack.

    Also tracks per-step module type index and call count, and the
    module's full path if registered via :func:`register_module_paths`.
    """
    stack = getattr(_module_context, "stack", None)
    if stack is None:
        _module_context.stack = []
        stack = _module_context.stack
    path = get_module_path(module) if module is not None else ""
    # Track module counter for this step
    mod_idx, mod_count = next_module_counter(cls_name)
    stack.append((cls_name, mod_idx, mod_count, path))


def pop_module_context() -> None:
    """Pop the most recent module class name from the stack."""
    stack = getattr(_module_context, "stack", None)
    if stack:
        stack.pop()


def get_current_module() -> Optional[str]:
    """Get the innermost (most recent) module class name, or None."""
    stack = getattr(_module_context, "stack", None)
    if stack:
        return stack[-1][0]
    return None


def get_current_module_counter() -> Optional[Tuple[int, int]]:
    """Get (module_type_idx, call_count) for the innermost module, or None."""
    stack = getattr(_module_context, "stack", None)
    if stack:
        return (stack[-1][1], stack[-1][2])
    return None


def module_context_matches(filter_set: Set[str]) -> bool:
    """Check if any module in the current call stack matches the filter."""
    stack = getattr(_module_context, "stack", None)
    if not stack:
        return False
    # Check from innermost to outermost for fast match
    for entry in reversed(stack):
        if entry[0] in filter_set:
            return True
    return False


def get_current_module_path() -> str:
    """Get the full path of the innermost module, or empty string."""
    stack = getattr(_module_context, "stack", None)
    if stack:
        return stack[-1][3]
    return ""


# ── Layer filtering ──

# Range pattern for layer specs like "0-3", "10-15"
_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")

# Default prefix for integer shorthand expansion
DEFAULT_LAYER_PREFIX = "model.layers."


def _is_glob(s: str) -> bool:
    """Check if a string contains glob wildcard characters."""
    return "*" in s or "?" in s


def expand_layer_specs(
    specs: Set[str],
    prefix: str = DEFAULT_LAYER_PREFIX,
) -> Set[str]:
    """Expand user-friendly layer specifications to full paths.

    Accepts several shorthand forms in addition to full paths:

    - **Integer**: ``"0"`` → ``"model.layers.0"``
    - **Range**: ``"0-3"`` → ``"model.layers.0"``, ..., ``"model.layers.3"``
    - **Full path**: ``"model.layers.0.self_attn"`` → kept as-is
    - **Glob pattern**: ``"model.layers.*.self_attn"`` → kept as-is

    Args:
        specs: Raw layer specifications from the user.
        prefix: Prefix for integer/range expansion.
            Defaults to ``"model.layers."``.

    Returns:
        Expanded set of layer path prefixes and/or glob patterns.
    """
    result: Set[str] = set()
    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue
        # Pure integer → single layer
        if spec.isdigit():
            result.add(prefix + spec)
            continue
        # Range like "0-3"
        m = _RANGE_RE.match(spec)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            for i in range(start, end + 1):
                result.add(prefix + str(i))
            continue
        # Full path or glob pattern — keep as-is
        result.add(spec)
    return result


def list_model_layers(
    model: torch.nn.Module,
    max_depth: Optional[int] = None,
) -> List[str]:
    """List all module paths in a model, useful for discovering layer names.

    Logs the paths at INFO level and returns them as a list.

    Args:
        model: The model to inspect.
        max_depth: Maximum nesting depth to show.  ``None`` = unlimited.
            Depth 1 shows direct children, depth 2 shows grandchildren, etc.

    Returns:
        List of ``(name, class_name)`` strings like
        ``"model.layers.0 (Qwen3DecoderLayer)"``.

    Example::

        from vllm_fl.dispatch.io_common import list_model_layers
        list_model_layers(model, max_depth=2)
    """
    paths: List[str] = []
    for name, mod in model.named_modules():
        if not name:
            continue  # skip root
        if max_depth is not None:
            depth = name.count(".") + 1
            if depth > max_depth:
                continue
        cls = type(mod).__name__
        paths.append(f"{name} ({cls})")
    if paths:
        _logger.info(
            "Model layer paths (%d modules):\n  %s",
            len(paths),
            "\n  ".join(paths),
        )
    return paths


def layer_path_matches(filter_set: Set[str]) -> bool:
    """Check if any module in the current call stack has a matching layer path.

    Supports three matching modes (determined per filter entry):
    - **Exact/prefix**: ``"model.layers.0"`` matches ``"model.layers.0"``
      and ``"model.layers.0.self_attn"`` but NOT ``"model.layers.00"``.
    - **Glob/wildcard**: entries containing ``*`` or ``?`` are matched
      with :func:`fnmatch.fnmatch`.  E.g. ``"model.layers.*.self_attn"``
      matches ``"model.layers.0.self_attn"``, ``"model.layers.1.self_attn"``.
    """
    stack = getattr(_module_context, "stack", None)
    if not stack:
        return False
    for entry in reversed(stack):
        path = entry[3]
        if path:
            for pattern in filter_set:
                if _is_glob(pattern):
                    if fnmatch.fnmatch(path, pattern):
                        return True
                else:
                    if path == pattern or path.startswith(pattern + "."):
                        return True
    return False


def parse_layers_env(*env_vars: str) -> Set[str]:
    """Parse layer path filters from the first non-empty env var.

    Checks each env var in order; returns the first valid set of
    layer specs after expansion (integers, ranges, globs, full paths).
    """
    for var in env_vars:
        value = os.environ.get(var, "").strip()
        if value:
            raw = {t.strip() for t in value.split(",") if t.strip()}
            return expand_layer_specs(raw)
    return set()


# ── Torch function filtering ──

SKIP_TORCH_FUNCS = frozenset({
    "size", "dim", "is_contiguous", "contiguous", "stride",
    "storage_offset", "numel", "element_size", "is_floating_point",
    "is_complex", "requires_grad_", "data_ptr", "device",
    "dtype", "shape", "ndim", "is_cuda", "is_cpu",
})


def get_torch_func_name(func) -> str:
    """Extract short name from a torch function."""
    return getattr(func, "__name__", str(func))


def parse_torch_funcs_config(value: str) -> Tuple[bool, Set[str]]:
    """
    Parse a torch funcs env var value (e.g. "1", "matmul,softmax").

    Returns:
        (enabled, func_filter) — func_filter empty means "all"
    """
    value = value.strip()
    if not value or value == "0":
        return False, set()
    if value == "1":
        return True, set()
    funcs = {t.strip() for t in value.split(",") if t.strip()}
    return True, funcs


def should_inspect_torch_func(
    func_name: str,
    torch_funcs_enabled: bool,
    torch_func_filter: Set[str],
    match_all: bool,
    op_filter: Set[str],
) -> bool:
    """Check if a torch function should be intercepted."""
    if not torch_funcs_enabled:
        return False
    if torch_func_filter:
        return func_name in torch_func_filter
    # "all" mode: skip dunder and trivial ops
    if func_name.startswith("_"):
        return False
    if func_name in SKIP_TORCH_FUNCS:
        return False
    # When specific ops are set (not match_all), only match those
    if op_filter and func_name not in op_filter:
        return False
    return True


# ── Dispatch op filtering (TorchDispatchMode) ──

def get_dispatch_op_name(func) -> str:
    """Extract short name from an ATen OpOverload.

    Examples::

        aten::mm           -> mm
        aten::_softmax     -> _softmax
        vllm::rms_norm     -> rms_norm
        mylib::my_add      -> my_add
    """
    if hasattr(func, "name"):
        full = func.name()  # e.g. "aten::mm"
        return full.split("::")[-1] if "::" in full else full
    return str(func)


def get_dispatch_op_namespace(func) -> str:
    """Extract namespace from an ATen OpOverload (e.g. 'aten', 'vllm')."""
    return getattr(func, "namespace", "aten")


# ── Dispatch key & backend detection ──
#
# Uses ``torch._C._dispatch_dump_table`` as the single source of truth.
# The table is parsed once per op (cached) to extract all registered
# ``[kernel]`` and ``[default backend kernel]`` entries as
# ``(dispatch_key, impl, is_default)`` triples.  Third-party backends
# like FlagGems are detected from the kernel registration path.

# Cache: op_qualified_name -> List[(dispatch_key, impl, is_default)]
_dispatch_table_cache: Dict[str, List[Tuple[str, str, bool]]] = {}


# Regex to extract backend name from PyTorch C++ registration paths like:
#   /pytorch/build/aten/src/ATen/RegisterCPU_0.cpp:3456
#   /pytorch/build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:7805
_REGISTER_PATTERN = re.compile(r"Register(\w+?)_\d+\.cpp")

# Known third-party package names → display labels.
# Checked in order; first match wins.  "triton" is intentionally last so that
# more specific packages (e.g. flag_gems, which *uses* triton) win first.
_THIRD_PARTY_BACKENDS = {
    "flag_gems": "FlagGems",
    "triton": "Triton",
}

def _infer_backend_from_path(reg_path: str) -> str:
    """Infer the backend name from a kernel registration path.

    Examples::

        /pytorch/build/.../RegisterCPU_0.cpp:3456           -> "CPU"
        /pytorch/build/.../RegisterCUDA_0.cpp:16060          -> "CUDA"
        /pytorch/build/.../RegisterCompositeExplicit..._0.cpp -> "CompositeExplicitAutogradNonFunctional"
        /.../flag_gems/__init__.py:20                        -> "FlagGems"
        /.../triton/ops/matmul.py:42                         -> "Triton"
        /.../torch/_meta_registrations.py:50                 -> "Meta"
        /.../torch/csrc/autograd/generated/TraceType_3.cpp   -> "Tracer"
    """
    path_lower = reg_path.lower()

    # Check third-party packages first
    for pkg, label in _THIRD_PARTY_BACKENDS.items():
        if pkg in path_lower:
            return label

    # Try Register<BackendName>_N.cpp pattern (C++ registrations)
    # Extract filename from path
    fname = reg_path.rsplit("/", 1)[-1] if "/" in reg_path else reg_path
    m = _REGISTER_PATTERN.match(fname)
    if m:
        return m.group(1)

    # Python meta registrations
    if "_meta_registrations" in path_lower:
        return "Meta"

    return "unknown"


def _parse_dispatch_table(func) -> List[Tuple[str, str, bool]]:
    """Parse ``_dispatch_dump_table`` for an op, returning all registered kernels.

    Returns a cached list of ``(dispatch_key, impl, is_default)`` triples:

    - **dispatch_key**: The dispatch key name (e.g. ``"CUDA"``, ``"CPU"``).
    - **impl**: The backend implementation inferred from the registration
      path (e.g. ``"CPU"``, ``"FlagGems"``,
      ``"CompositeExplicitAutogradNonFunctional"``).
    - **is_default**: ``True`` when the entry is a ``[default backend kernel]``
      (fallback), ``False`` for a dedicated ``[kernel]``.

    Excludes fallthrough entries.
    """
    op_qname = func.name() if hasattr(func, "name") else None
    if op_qname is None:
        return []

    cached = _dispatch_table_cache.get(op_qname)
    if cached is not None:
        return cached

    entries: List[Tuple[str, str, bool]] = []
    try:
        table = torch._C._dispatch_dump_table(op_qname)
        for line in table.split("\n"):
            stripped = line.strip()
            is_kernel = "[kernel]" in stripped
            is_default = "[default backend kernel]" in stripped
            if not is_kernel and not is_default:
                continue
            if "fallthrough" in stripped:
                continue
            colon_idx = stripped.find(": registered at ")
            if colon_idx < 0:
                continue
            key = stripped[:colon_idx].strip()
            reg_path = stripped[colon_idx + 16:].split(" [")[0]
            impl = _infer_backend_from_path(reg_path)
            entries.append((key, impl, is_default))
    except Exception:
        pass

    _dispatch_table_cache[op_qname] = entries
    return entries


def get_dispatch_keys(
    func, args: tuple = (), kwargs: dict = None,
) -> List[Tuple[str, str, bool]]:
    """Return all registered dispatch keys for an op.

    Returns the cached ``(dispatch_key, impl, is_default)`` list from
    :func:`_parse_dispatch_table`.  See that function for field descriptions.

    Example return value::

        [("CPU", "CPU", False),
         ("CUDA", "FlagGems", False),
         ("HIP", "CompositeExplicitAutogradNonFunctional", True),
         ("Meta", "Meta", False)]
    """
    entries = _parse_dispatch_table(func)
    if not entries:
        return [("unknown", "unknown", True)]
    return entries



def should_inspect_dispatch_op(
    op_name: str,
    match_all: bool,
    op_filter: Set[str],
) -> bool:
    """Check if a dispatch op should be intercepted.

    Args:
        op_name: Short op name (e.g. 'mm', 'rms_norm').
        match_all: Whether all ops are being captured.
        op_filter: Specific op names to match (empty means all).
    """
    if op_filter and op_name not in op_filter:
        return False
    return True


# ── Stack-based module context extraction ──
#
# Used by TorchDispatchMode to derive module context from the Python call
# stack, without requiring global module hooks.  Works in both eager and
# compiled modes.

def get_module_context_from_stack() -> List[Tuple[str, str]]:
    """Walk the Python call stack to find enclosing nn.Module instances.

    Returns a list of ``(class_name, module_path)`` pairs, innermost first.
    ``module_path`` is looked up from the module path registry (populated
    by :func:`register_module_paths`); it is empty string if not found.

    This is a debugging tool — the stack walk has a small performance cost
    (~0.18ms per call for a 10-layer model).
    """
    frame = inspect.currentframe()
    result: List[Tuple[str, str]] = []
    seen_ids: Set[int] = set()
    try:
        f = frame.f_back if frame is not None else None
        while f is not None:
            local_self = f.f_locals.get("self")
            if (
                local_self is not None
                and isinstance(local_self, nn.Module)
                and f.f_code.co_name in ("forward", "_call_impl")
                and id(local_self) not in seen_ids
            ):
                seen_ids.add(id(local_self))
                cls_name = type(local_self).__name__
                path = _module_path_map.get(id(local_self), "")
                result.append((cls_name, path))
            f = f.f_back
    finally:
        del frame
    return result


def layer_path_matches_from_stack(
    filter_set: Set[str],
    module_ctx: Optional[List[Tuple[str, str]]] = None,
) -> bool:
    """Check if any module in the given context matches the layer filter.

    Like :func:`layer_path_matches` but uses stack-derived context instead
    of the global module hook context stack.

    Args:
        filter_set: Set of layer path patterns (prefix or glob).
        module_ctx: Output of :func:`get_module_context_from_stack`.
            If None, derives it automatically.
    """
    if module_ctx is None:
        module_ctx = get_module_context_from_stack()
    for _cls_name, path in module_ctx:
        if not path:
            continue
        for pattern in filter_set:
            if _is_glob(pattern):
                if fnmatch.fnmatch(path, pattern):
                    return True
            else:
                if path == pattern or path.startswith(pattern + "."):
                    return True
    return False


def module_context_matches_from_stack(
    filter_set: Set[str],
    module_ctx: Optional[List[Tuple[str, str]]] = None,
) -> bool:
    """Check if any module class name in the stack context matches the filter.

    Like :func:`module_context_matches` but uses stack-derived context.
    """
    if module_ctx is None:
        module_ctx = get_module_context_from_stack()
    for cls_name, _path in module_ctx:
        if cls_name in filter_set:
            return True
    return False


# ── Tensor statistics ──
#
# Extensible registry: users can add custom stats via register_tensor_stat().


def _stat_min(t: torch.Tensor) -> float:
    return t.min().item()


def _stat_max(t: torch.Tensor) -> float:
    return t.max().item()


def _stat_mean(t: torch.Tensor) -> float:
    return t.mean().item()


def _stat_std(t: torch.Tensor) -> float:
    return t.std().item()


# Each entry: (name, fn, float_only).
# float_only=True → fn receives t.detach().float(); skipped for integer tensors.
# float_only=False → fn receives the original tensor (int or float).
_TENSOR_STAT_REGISTRY: List[Tuple[str, Callable[[torch.Tensor], Any], bool]] = [
    ("min", _stat_min, False),
    ("max", _stat_max, False),
    ("mean", _stat_mean, True),
    ("std", _stat_std, True),
]
_tensor_stat_lock = threading.Lock()


def register_tensor_stat(
    name: str,
    fn: Callable[[torch.Tensor], Any],
    float_only: bool = True,
) -> None:
    """Register a custom tensor statistic for the IO dumper.

    The function receives a tensor and should return a JSON-serializable
    value — scalar, list, or dict.  Scalars are displayed inline;
    lists are shown truncated (e.g. ``top_k=[1.2, 0.8, ...]``).

    For ``float_only=True`` (default), the tensor is ``t.detach().float()``;
    for ``float_only=False``, the original tensor is passed.

    If a stat with the same *name* already exists, it is replaced.
    Raises ``ValueError`` if *name* is empty.

    Example::

        from vllm_fl.dispatch import register_tensor_stat
        # Scalar stats
        register_tensor_stat("l2_norm", lambda t: t.norm(2).item())
        register_tensor_stat("sparsity",
                             lambda t: (t == 0).float().mean().item())
        # List stats (for future top-k, first-n, last-n, etc.)
        register_tensor_stat("top_5",
                             lambda t: t.flatten().topk(5).values.tolist(),
                             float_only=False)
        register_tensor_stat("first_4",
                             lambda t: t.flatten()[:4].tolist(),
                             float_only=False)
    """
    if not name or not name.strip():
        raise ValueError("Tensor stat name must be a non-empty string")
    with _tensor_stat_lock:
        for i, (existing_name, _, _) in enumerate(_TENSOR_STAT_REGISTRY):
            if existing_name == name:
                _TENSOR_STAT_REGISTRY[i] = (name, fn, float_only)
                return
        _TENSOR_STAT_REGISTRY.append((name, fn, float_only))


def tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
    """Compute shape, dtype, device, and all registered statistics for a tensor.

    Used by the dumper for text formatting and JSON metadata.
    Returns a dict like ``{"shape": [...], "dtype": "...", "min": ..., ...}``.
    """
    meta: Dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
    }
    if t.numel() == 0 or t.is_complex():
        return meta
    is_fp = t.is_floating_point()
    fp_tensor: Optional[torch.Tensor] = None  # lazy
    for name, fn, float_only in _TENSOR_STAT_REGISTRY:
        if float_only and not is_fp:
            continue
        try:
            if float_only:
                if fp_tensor is None:
                    fp_tensor = t.detach().float()
                meta[name] = fn(fp_tensor)
            else:
                meta[name] = fn(t)
        except Exception:
            pass
    return meta


# ── Formatting ──

_STAT_LIST_DISPLAY_LIMIT = 4  # max items shown inline for list-valued stats


def _format_stat_value(v: Any) -> str:
    """Format a stat value for inline display.

    Scalars: ``0.123456``.  Lists: ``[1.2, 0.8, ...]`` (truncated).
    """
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        limit = _STAT_LIST_DISPLAY_LIMIT
        formatted = [f"{x:.6g}" if isinstance(x, float) else repr(x)
                     for x in v[:limit]]
        if len(v) > limit:
            formatted.append("...")
        return f"[{', '.join(formatted)}]"
    return repr(v)


def format_value(value: Any) -> str:
    """Format a single value for display, including tensor statistics."""
    if isinstance(value, torch.Tensor):
        stats = tensor_stats(value)
        parts = [
            f"shape={stats['shape']}",
            f"dtype={stats['dtype']}",
            f"device={stats['device']}",
        ]
        for name, _, _ in _TENSOR_STAT_REGISTRY:
            if name in stats:
                v = stats[name]
                parts.append(f"{name}={_format_stat_value(v)}")
        return f"Tensor({', '.join(parts)})"
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        type_name = type(value).__name__
        if len(value) <= 4:
            items = ", ".join(format_value(v) for v in value)
            return f"{type_name}([{items}])"
        return f"{type_name}(len={len(value)})"
    return f"{type(value).__name__}(...)"


def format_result(result: Any) -> str:
    """Format operator output for display."""
    if isinstance(result, tuple):
        lines = []
        for i, v in enumerate(result):
            lines.append(f"  result[{i}]: {format_value(v)}")
        return "\n".join(lines)
    return f"  result: {format_value(result)}"


def get_module_class_name(args: tuple) -> Optional[str]:
    """Extract module class name from args[0] if it is an nn.Module."""
    if args and isinstance(args[0], torch.nn.Module):
        return type(args[0]).__name__
    return None


def make_module_tag() -> str:
    """Build ``[module=X,Y]`` from current module context, or empty string."""
    counter = get_current_module_counter()
    if counter is not None:
        return f"[module={counter[0]},{counter[1]}]"
    return ""


def make_module_tag_from_ctx(
    mod_name: str, mod_path: str,
    for_json: bool = False,
) -> str:
    """Build a module tag from stack-derived module context.

    Args:
        mod_name: Module class name (e.g. ``"Linear"``).
        mod_path: Full dotted path (e.g. ``"model.layers.0.self_attn.q_proj"``).
        for_json: If True, return bare ``"ClassName:path"`` for JSON fields.
            If False, return ``"[ClassName:path]"`` for log display.

    Returns:
        Empty string if *mod_name* is empty.

    Examples::

        make_module_tag_from_ctx("Linear", "model.layers.0.self_attn.q_proj", for_json=True)
        # => "Linear:model.layers.0.self_attn.q_proj"

        make_module_tag_from_ctx("Linear", "model.layers.0.self_attn.q_proj")
        # => "[Linear:model.layers.0.self_attn.q_proj]"

        make_module_tag_from_ctx("Linear", "")
        # => "[Linear]" or "Linear"
    """
    if not mod_name:
        return ""
    tag = f"{mod_name}:{mod_path}" if mod_path else mod_name
    if for_json:
        return tag
    return f"[{tag}]"


def make_op_tag(op_name: str) -> str:
    """Build ``[op=X,Y]``, incrementing the per-step op counter."""
    idx, count = next_op_counter(op_name)
    return f"[op={idx},{count}]"


def make_label(
    op_name: str,
    args: tuple = (),
    module_name: Optional[str] = None,
    layer_path: Optional[str] = None,
    dispatch_keys: Optional[List[Tuple[str, str, bool]]] = None,
) -> str:
    """Build a display label for an op call.

    Example output::

        mm (module=Linear, layer=model.layers.0, dispatch_keys=[(CUDA, FlagGems), (CPU, CPU)])

    Args:
        op_name: Operator name.
        args: Operator args (used to detect module from args[0] via old hooks).
        module_name: Explicit module class name (from stack-based context).
        layer_path: Explicit layer path (from stack-based context).
        dispatch_keys: Full ``(key, impl, is_default)`` list from
            :func:`get_dispatch_keys`.
    """
    if module_name is None:
        module_name = get_module_class_name(args) or get_current_module()
    if layer_path is None:
        layer_path = get_current_module_path()
    parts = [f"module={module_name or ''}"]
    if layer_path:
        parts.append(f"layer={layer_path}")
    if dispatch_keys is not None:
        dk_items = [f"({k}, {impl})" for k, impl, _default in dispatch_keys]
        parts.append(f"dispatch_keys=[{', '.join(dk_items)}]")
    return f"{op_name} ({', '.join(parts)})"


def record_seen(
    op_name: str,
    args: tuple = (),
    module_name: Optional[str] = None,
) -> None:
    """Record an op and its enclosing module as seen this step (for summary).

    Args:
        op_name: Operator name.
        args: Operator args (used to detect module from args[0] via old hooks).
        module_name: Explicit module class name (from stack-based context).
            Overrides args-based and global-hook-based detection.
    """
    if module_name is None:
        module_name = get_module_class_name(args) or get_current_module()
    with _counter_lock:
        _seen_ops.add(op_name)
        if module_name:
            _seen_modules.add(module_name)


# ── Torch func tag management ──
#
# When multiple TorchFunctionMode handlers are stacked, the outer handler
# calls func() which enters the inner.  Without coordination, both would
# call make_op_tag() / next_exec_order(), double-incrementing the shared
# counters.  These helpers ensure that tags and exec_order are generated
# once (by the first handler) and reused by the nested handler.

_tf_tags = threading.local()


def acquire_torch_func_tags(op_name: str) -> Tuple[str, str, int]:
    """Get or generate (module_tag, op_tag, exec_order) for a torch function call.

    The first caller (outermost TorchFunctionMode) generates and caches
    the values.  Nested callers reuse the cached values, preventing
    double-increment of shared counters.

    Must be paired with :func:`release_torch_func_tags`.
    """
    depth = getattr(_tf_tags, "depth", 0)
    _tf_tags.depth = depth + 1

    if depth == 0:
        # First (outermost) caller: generate and cache
        module_tag = make_module_tag()
        op_tag = make_op_tag(op_name)
        order = next_exec_order()
        _tf_tags.module_tag = module_tag
        _tf_tags.op_tag = op_tag
        _tf_tags.exec_order = order
    else:
        # Nested caller: reuse cached values
        module_tag = getattr(_tf_tags, "module_tag", "")
        op_tag = getattr(_tf_tags, "op_tag", "")
        order = getattr(_tf_tags, "exec_order", 0)

    return module_tag, op_tag, order


def release_torch_func_tags() -> None:
    """Release torch function tags.  Clears the cache when the outermost caller exits."""
    depth = getattr(_tf_tags, "depth", 0)
    if depth > 0:
        _tf_tags.depth = depth - 1
    if depth <= 1:
        _tf_tags.module_tag = None
        _tf_tags.op_tag = None
        _tf_tags.exec_order = None


# ── YAML config parsing ──


def parse_io_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Parse IO dump settings from a YAML config file.

    Expected YAML structure (all fields optional)::

        io_dump:
          dir: /tmp/io_dump
          with_print: true        # enable console logging
          ops:
            - rms_norm
          modules:
            - Linear
          max_calls: 100
          step_range: "5-15"      # inclusive "start-end"
          with_torch_funcs: true
          with_metas: true        # write per-op JSON files
          with_values: true       # write per-call .pt files

    Returns:
        Dict with key "io_dump" containing parsed settings.
        Returns empty dict if file not found or has no io_dump section.
    """
    if not os.path.isfile(config_path):
        return {}

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return {}

    result: Dict[str, Any] = {}

    dump_cfg = config.get("io_dump")
    if isinstance(dump_cfg, dict):
        result["io_dump"] = _parse_dump_section(dump_cfg)

    return result


def _parse_step_range_yaml(cfg: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Parse ``step_range`` from a YAML section.

    Accepts a dash-string like ``"0-2"`` (inclusive, same format as env vars).
    Returns a half-open ``(start, end+1)`` tuple for internal use,
    or ``None`` if unset / invalid.
    """
    step_range = cfg.get("step_range")
    if step_range is None:
        return None
    # Convert int/list to string for uniform handling
    if isinstance(step_range, int):
        step_range = str(step_range)
    elif isinstance(step_range, (list, tuple)):
        # Legacy [start, end] format — convert to "start-end" string
        if len(step_range) == 2:
            step_range = f"{step_range[0]}-{step_range[1]}"
        else:
            return None
    return parse_step_range(step_range)


def _parse_dump_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the io_dump section of a YAML config."""
    parsed: Dict[str, Any] = {}

    dump_dir = cfg.get("dir", "")
    parsed["dir"] = str(dump_dir).strip() if dump_dir else ""

    parsed["ops"] = _parse_string_list(cfg.get("ops"))
    parsed["modules"] = _parse_string_list(cfg.get("modules"))
    parsed["layers"] = expand_layer_specs(_parse_string_list(cfg.get("layers")))

    max_calls = cfg.get("max_calls", 0)
    try:
        parsed["max_calls"] = int(max_calls)
    except (ValueError, TypeError):
        parsed["max_calls"] = 0

    parsed["step_range"] = _parse_step_range_yaml(cfg)

    # Only include with_torch_funcs when explicitly set in YAML so that
    # _init_from_env can apply the match-all default for absent keys.
    if "with_torch_funcs" in cfg:
        parsed["with_torch_funcs"] = _parse_torch_funcs_yaml(cfg["with_torch_funcs"])
    parsed["ranks"] = _parse_ranks_yaml(cfg.get("ranks"))
    parsed["with_values"] = bool(cfg.get("with_values", False))
    if "with_metas" in cfg:
        parsed["with_metas"] = bool(cfg["with_metas"])
    if "with_print" in cfg:
        parsed["with_print"] = bool(cfg["with_print"])

    return parsed


def _parse_string_list(value: Any) -> Set[str]:
    """Parse a YAML value into a set of strings.

    Accepts:
      - list of strings: ["rms_norm", "silu_and_mul"]
      - comma-separated string: "rms_norm,silu_and_mul"
      - None / empty → empty set
    """
    if not value:
        return set()
    if isinstance(value, (list, tuple)):
        return {str(v).strip() for v in value if v}
    if isinstance(value, str):
        return {t.strip() for t in value.split(",") if t.strip()}
    return set()


def _parse_torch_funcs_yaml(value: Any) -> Tuple[bool, Set[str]]:
    """Parse torch_funcs from YAML config.

    Accepts:
      - true/false: enable/disable all
      - list: ["matmul", "softmax"] — enable specific
      - string: "matmul,softmax" or "1"
    """
    if value is None or value is False:
        return False, set()
    if value is True:
        return True, set()
    if isinstance(value, (list, tuple)):
        funcs = {str(v).strip() for v in value if v}
        return bool(funcs), funcs
    if isinstance(value, str):
        return parse_torch_funcs_config(value)
    return False, set()


def _parse_ranks_yaml(value: Any) -> Optional[Set[int]]:
    """Parse ``ranks`` from YAML config.

    Accepts:
      - None / "all" → None (all ranks)
      - list of ints: [0, 1, 2]
      - single int: 0
      - string: "0,2,4" or "all"
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        ranks = set()
        for v in value:
            try:
                ranks.add(int(v))
            except (ValueError, TypeError):
                pass
        return ranks if ranks else None
    if isinstance(value, int):
        return {value}
    if isinstance(value, str):
        return parse_rank_filter(value)
    return None
