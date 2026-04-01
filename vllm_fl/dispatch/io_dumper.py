# Copyright (c) 2026 BAAI. All rights reserved.

"""
IO Dumper for vllm-plugin-FL dispatch.

Handles both file dumping and console printing (logging) of operator I/O.
The ``with_print`` toggle enables console logging of inputs/outputs alongside
(or instead of) file dumping.

Saves operator input/output tensors to disk as PyTorch .pt files.

Only dumps at the **op level** (dispatch-managed ops and bare torch functions).
Module-level dumping is not produced; module names are included in dump
metadata via the thread-local context set by module hooks.  The ``module_tag``
field is always populated when ops run inside a tracked module.

When a ``modules`` filter is set (e.g., ``modules={"RMSNormFL"}``), module
context is read from the thread-local set by module hooks.  Only ops that run
inside the specified modules are dumped.

Supports three interception mechanisms:
1. Dispatch-managed ops: Automatic via OpManager.call() hooks
2. ATen-level ops: Via TorchDispatchMode (default, works in both eager and compile modes)
3. Bare torch functions: Via TorchFunctionMode (opt-in, eager mode only)

Initialized via ``init_io_dump_from_env()`` (called from ``model_runner.load_model()``)
or the programmatic API ``enable_io_dump()``.  Both parse config and activate hooks
immediately; the env-var path is a no-op in graph mode (``enforce_eager=False``).

Configuration (priority order):
    1. Python API: enable_io_dump(dump_dir=..., ops=..., step_range=..., layers=..., with_print=True)
    2. YAML config file (via VLLM_FL_CONFIG):
        io_dump:
          dir: /tmp/io_dump
          with_print: true          # enable console logging of I/O
          ops: [rms_norm, silu_and_mul]
          modules: [Linear]
          layers: [0, 1-3, model.layers.*.self_attn]
          max_calls: 100
          step_range: "5-15"        # inclusive "start-end"
          with_torch_funcs: true    # intercept bare torch functions
          with_metas: true          # write per-op input/output JSON (default: summary only)
          with_values: true         # write per-call .pt tensor files (default: off)
    3. Environment variables:
        VLLM_FL_IO_DUMP                  - Directory path or "1" for ./io_dump
        VLLM_FL_IO_DUMP_OPS              - Comma-separated op names
        VLLM_FL_IO_DUMP_MODULES          - Comma-separated module class names
        VLLM_FL_IO_DUMP_LAYERS           - Layer specs: integers, ranges, globs, paths
        VLLM_FL_IO_DUMP_MAX_CALLS        - Max calls per op (0 = unlimited)
        VLLM_FL_IO_DUMP_STEP_RANGE       - "start-end" inclusive range (e.g. "0-4")
        VLLM_FL_IO_DUMP_RANK             - Rank filter: "all", "0", "0,2,4"
        VLLM_FL_IO_DUMP_WITH_PRINT       - "1" to enable console logging of I/O
        VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS - "1" or "matmul,softmax" to intercept torch funcs
        VLLM_FL_IO_DUMP_WITH_METAS       - "1" to write per-op input/output JSON files
        VLLM_FL_IO_DUMP_WITH_VALUES      - "1" to write per-call .pt tensor data files

    Note: Setting any filter env var (step_range or layers) auto-enables
    dumping for all ops, even without VLLM_FL_IO_DUMP=1.

    Quick start (env-var only, no Python API needed):
        VLLM_FL_IO_DUMP_STEP_RANGE=0-2 VLLM_FL_IO_DUMP_LAYERS=1-3 VLLM_FL_IO_DUMP_WITH_METAS=1 python script.py

All filter dimensions are composable (AND logic): step_range, layers, modules,
and ops are orthogonal gates.  When multiple filters are set, an op must pass
ALL of them.  Unset filters are pass-through.

File layout:
    dump_dir/rank_0000/
        summary.json               # op summary: flaggems / non-flaggems classification
        step_0005/rms_norm/
            input.json             # merged metadata for all calls' inputs
            output.json            # merged metadata for all calls' outputs
            call_1_input.pt        # tensor data for call 1 inputs (with_values=True)
            call_1_output.pt       # tensor data for call 1 outputs
            call_2_input.pt
            call_2_output.pt
        step_0005/torch.matmul/
            input.json
            output.json
            ...

    JSON keys (``call_1``, ``call_2``, ...) match the PT file names for
    easy cross-reference.  When ``with_values=False`` (default), only JSON
    files are written; when ``with_metas=False`` (default), only summary.json
    is written.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

# Sentinel to distinguish "caller didn't set this" from explicit values.
_UNSET: Any = object()

import torch

from .io_common import (
    HAS_TORCH_DISPATCH_MODE,
    HAS_TORCH_FUNC_MODE,
    TorchDispatchMode,
    TorchFunctionMode,
    acquire_torch_func_tags,
    advance_step,
    dispatch_mode_mgr,
    expand_layer_specs,
    format_result,
    format_value,
    func_mode_mgr,
    get_current_module,
    get_current_module_path,
    get_dispatch_keys,
    get_dispatch_op_name,
    get_dispatch_op_namespace,
    get_module_class_name,
    get_rank,
    get_step,
    get_torch_func_name,
    layer_path_matches,
    make_guard,
    make_label,
    make_module_tag_from_ctx,
    make_op_tag,
    module_context_matches,
    next_exec_order,
    pop_module_context,
    push_module_context,
    register_module_paths,
    parse_io_config_from_yaml,
    parse_layers_env,
    parse_rank_filter,
    parse_step_range,
    parse_step_range_env,
    parse_torch_funcs_config,
    record_seen,
    register_step_callback,
    release_torch_func_tags,
    set_io_active,
    should_inspect_dispatch_op,
    should_inspect_torch_func,
    tensor_stats,
    unregister_step_callback,
)
from .logger_manager import get_logger

logger = get_logger("vllm_fl.dispatch.io_dump")
_print_logger = get_logger("vllm_fl.dispatch.io_print")

# ── Module-level state ──

# Independent re-entrancy guard so dumper doesn't block other hooks
guard_active, set_guard = make_guard()

_enabled: bool = False
_dump_dir: str = ""
_match_all: bool = False
_op_filter: Set[str] = set()
_module_filter: Set[str] = set()
_layer_filter: Set[str] = set()
_max_calls: int = 0  # 0 = unlimited
_step_range: Optional[Tuple[int, int]] = None
_with_values: bool = False  # write per-call .pt tensor data files
_with_metas: bool = False   # write per-op input/output JSON files (summary-only when False)

# Print (console logging) state
_print_enabled: bool = False

_call_counters: Dict[str, int] = {}
_lock = threading.Lock()

# ── Summary accumulator ──
# Collects per-op metadata across all steps for the final summary.json.
# op_name → {dispatch_keys: [...], call_count: int}
_op_summary: Dict[str, Dict[str, Any]] = {}

# ── Async I/O state ──
# Background executor for stats computation and file writes.
# Threading is sufficient: PyTorch releases the GIL during tensor ops and I/O.
_bg_workers: int = 8  # configurable via enable_io_dump(bg_workers=...) / VLLM_FL_IO_DUMP_BG_WORKERS
_io_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_pending_futures: List[concurrent.futures.Future] = []
_pending_lock = threading.Lock()

# Streaming JSON Lines file handles, one file per json_path per step.
# _WipHandle tracks an open file handle plus an ordered-write reorder buffer
# so that lines are always written in call_num order even if background tasks
# complete out of order.  Step end closes all handles; no finalization needed.


class _WipHandle:
    """Per-file state for in-order JSON Lines streaming."""

    __slots__ = ("fh", "lock", "pending", "next_write")

    def __init__(self, fh: Any) -> None:
        self.fh = fh
        self.lock = threading.Lock()
        self.pending: Dict[int, str] = {}  # call_num -> JSON line
        self.next_write: int = 1  # _next_call_num always starts at 1 per op per step


_open_file_handles: Dict[str, "_WipHandle"] = {}
_open_files_lock = threading.Lock()

# Thread-local storage for pairing dump_before → dump_after.
# Each thread stores a dict of op_name → list of (call_num, exec_order, op_dir).
_dump_pairing = threading.local()

# Thread-local storage for pairing print inputs → outputs (console logging).
_print_pairing = threading.local()

_torch_funcs_enabled: bool = False
_torch_func_filter: Set[str] = set()
_rank_filter: Optional[Set[int]] = None  # None = all ranks


def _rank_ok() -> bool:
    """Check if the current rank passes this subsystem's rank filter."""
    if _rank_filter is None:
        return True
    return get_rank() in _rank_filter


# Hook state
_hooks_activated: bool = False  # True once dispatch/func modes have been entered
_module_hook_handles: List[Any] = []  # PyTorch forward hook handles for module context


# ── Print formatting / pairing helpers ──


def _format_inputs(args: tuple, kwargs: dict,
                   skip_module_arg: bool = False) -> List[str]:
    """Format operator/module inputs into lines for console logging."""
    lines = []
    for i, arg in enumerate(args):
        if skip_module_arg and i == 0 and isinstance(arg, torch.nn.Module):
            continue
        lines.append(f"  arg[{i}]: {format_value(arg)}")
    for k, v in kwargs.items():
        lines.append(f"  {k}: {format_value(v)}")
    return lines


def _log_combined(label: str, input_lines: List[str], result: Any,
                  op_tag: str = "",
                  exec_order: int = 0) -> None:
    """Log a consolidated INPUTS + OUTPUTS block for an op call."""
    rank = get_rank()
    step = get_step()
    sep = "-" * 60
    parts = [
        f"\n{sep}",
        f"[IO_PRINT][rank={rank}][step={step}][exec_order={exec_order}]{op_tag} {label}",
        f"{sep}",
        "  INPUTS:",
    ]
    parts.extend(f"  {line}" for line in input_lines)
    parts.append("  OUTPUTS:")
    parts.append(f"  {format_result(result)}")
    parts.append(sep)
    _print_logger.info("\n".join(parts))


def _log_outputs_only(label: str, result: Any,
                      op_tag: str = "",
                      exec_order: int = 0) -> None:
    """Log operator/module outputs only (fallback when no pairing)."""
    rank = get_rank()
    step = get_step()
    _print_logger.info(f"[IO_PRINT][rank={rank}][step={step}][exec_order={exec_order}]{op_tag} {label} OUTPUTS:\n{format_result(result)}")


def _push_print_pairing(op_name: str, label: str, order: int,
                           input_lines: List[str], op_tag: str) -> None:
    """Store print pairing info in thread-local for dump_after to consume."""
    stack = getattr(_print_pairing, "stack", None)
    if stack is None:
        _print_pairing.stack = {}
        stack = _print_pairing.stack
    stack.setdefault(op_name, []).append((label, order, input_lines, op_tag))


def _pop_print_pairing(op_name: str):
    """Retrieve print pairing info stored by the most recent dump_before."""
    stack = getattr(_print_pairing, "stack", None)
    if stack is None:
        return None
    entries = stack.get(op_name)
    if entries:
        return entries.pop()
    return None


def _on_step_advance(step: int, seen_modules: Set[str], seen_ops: Set[str]) -> None:
    """Callback to clear per-op call counters, update summary, and log on step advance.

    Also handles lazy activation/deactivation of dispatch modes when
    ``_step_range`` defers activation past step 0.
    """
    next_step = step + 1  # step just completed; next step is about to begin

    # Lazy activation: enter dispatch mode when next step enters range.
    if (_step_range is not None
            and not dispatch_mode_mgr.is_entered("dump")
            and next_step >= _step_range[0]
            and next_step < _step_range[1]):
        _enter_dispatch_modes()

    with _lock:
        _call_counters.clear()
    # Drain background tasks and flush buffered JSON before writing summary
    if _dump_dir:
        _wait_and_flush()
        _write_summary()
    if seen_modules or seen_ops:
        rank = get_rank()
        logger.info(
            f"[IO_DUMP][rank={rank}][step={step}] Step summary: "
            f"modules={sorted(seen_modules)}, ops={sorted(seen_ops)}"
        )
        if _print_enabled:
            _print_logger.info(
                f"[IO_PRINT][rank={rank}][step={step}] Step summary: "
                f"modules={sorted(seen_modules)}, ops={sorted(seen_ops)}"
            )


# ── Filtering ──


def _check_limits(op_name: str) -> bool:
    """Check step_range and max_calls limits (no filter logic)."""
    if _step_range is not None:
        step = get_step()
        if step < _step_range[0] or step >= _step_range[1]:
            return False
    if _max_calls > 0:
        with _lock:
            if _call_counters.get(op_name, 0) >= _max_calls:
                return False
    return True


def _should_dump(op_name: str, args: tuple) -> bool:
    """Check if this dispatch-managed op call should be dumped.

    Filters are composable AND gates — each active filter must pass:
    - ``_step_range`` / ``_max_calls``: checked via ``_check_limits``
    - ``_layer_filter``: current layer path must match (prefix)
    - ``_module_filter``: must be inside a matching module
    - ``_op_filter``: op name must be in the filter set
    """
    if not _check_limits(op_name):
        return False
    if _layer_filter and not layer_path_matches(_layer_filter):
        return False
    if _match_all:
        return True
    if _module_filter:
        cls = get_module_class_name(args)
        if not (cls and cls in _module_filter) and not module_context_matches(_module_filter):
            return False
    if _op_filter:
        if op_name not in _op_filter:
            return False
    return bool(_op_filter) or bool(_module_filter)


def _should_dump_torch_func(func_name: str) -> bool:
    """Check if a torch function should be dumped.

    Layer and module filters are AND gates (same as ``_should_dump``).
    Uses thread-local module context set by module hooks.
    """
    if _step_range is not None:
        step = get_step()
        if step < _step_range[0] or step >= _step_range[1]:
            return False
    if _layer_filter and not layer_path_matches(_layer_filter):
        return False
    if not should_inspect_torch_func(
        func_name, _torch_funcs_enabled, _torch_func_filter,
        _match_all, _op_filter,
    ):
        return False
    # When module filter is active, only match torch funcs inside those modules
    if _module_filter and not _match_all:
        return module_context_matches(_module_filter)
    return True


# ── Serialization ──


def _serialize_value(value: Any) -> Any:
    """Prepare a value for torch.save(). Tensors -> CPU, Modules -> string."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, torch.nn.Module):
        return f"<module:{type(value).__name__}>"
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_serialize_value(v) for v in value)
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    return f"<{type(value).__name__}>"


def _build_data(args: tuple, kwargs: dict, *, is_output: bool = False) -> Dict[str, Any]:
    """Build tensor data dict for torch.save (PT file).

    Keys use the same scheme as ``_dump_input`` / ``_dump_output`` metadata
    (``arg_N``, ``kwarg_<name>``, ``result`` / ``result_N``) for easy
    cross-reference between .pt files and JSON.
    """
    data: Dict[str, Any] = {}
    if is_output:
        result = args[0] if args else None
        if isinstance(result, tuple):
            for i, v in enumerate(result):
                data[f"result_{i}"] = _serialize_value(v)
        else:
            data["result"] = _serialize_value(result)
    else:
        for i, arg in enumerate(args):
            data[f"arg_{i}"] = _serialize_value(arg)
        for k, v in kwargs.items():
            data[f"kwarg_{k}"] = _serialize_value(v)
    return data


def _sanitize_path_component(name: str) -> str:
    """Sanitize a name for safe use as a single path component.

    Replaces path separators and '..' to prevent directory traversal.
    """
    # Replace OS path separators with underscores
    safe = name.replace(os.sep, "_")
    if os.altsep:
        safe = safe.replace(os.altsep, "_")
    # Collapse any remaining '..' sequences
    safe = safe.replace("..", "__")
    # Strip leading/trailing whitespace and dots
    safe = safe.strip(". ")
    return safe or "_unnamed_"


def _push_pairing(op_name: str, call_num: int, exec_order: int, op_dir: str,
                   label: Optional[str] = None,
                   module_tag: str = "",
                   op_tag: str = "") -> None:
    """Store pairing info in thread-local for dump_after to consume."""
    stack = getattr(_dump_pairing, "stack", None)
    if stack is None:
        _dump_pairing.stack = {}
        stack = _dump_pairing.stack
    stack.setdefault(op_name, []).append(
        (call_num, exec_order, op_dir, label or op_name, module_tag, op_tag)
    )


def _pop_pairing(op_name: str):
    """Retrieve pairing info stored by the most recent dump_before for this op."""
    stack = getattr(_dump_pairing, "stack", None)
    if stack is None:
        return None
    entries = stack.get(op_name)
    if entries:
        return entries.pop()
    return None


def _get_op_dir(op_name: str) -> str:
    """Build the per-op directory path: dump_dir/rank_XXXX/step_XXXX/op_name."""
    safe_name = _sanitize_path_component(op_name)
    rank = get_rank()
    rank_dir = os.path.join(_dump_dir, f"rank_{rank:04d}")
    step_dir = os.path.join(rank_dir, f"step_{get_step():04d}")
    return os.path.join(step_dir, safe_name)


def _next_call_num(op_name: str) -> int:
    """Increment and return the per-op call counter (thread-safe)."""
    with _lock:
        count = _call_counters.get(op_name, 0) + 1
        _call_counters[op_name] = count
        return count


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _io_executor
    if _io_executor is None or _io_executor._shutdown:
        _io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_bg_workers, thread_name_prefix="io_dump_bg")
    return _io_executor


def _submit_bg(fn, *args, **kwargs) -> None:
    """Submit work to background executor; track future for later drain."""
    if not _enabled:
        return
    fut = _get_executor().submit(fn, *args, **kwargs)
    with _pending_lock:
        _pending_futures.append(fut)


def _record_device_event(
    tensor_refs: Dict[str, "torch.Tensor"],
) -> Optional[Any]:
    """Record a current-stream event on whatever accelerator is in use.

    Uses ``getattr(torch, device_type).Event()`` — portable to CUDA, XPU,
    and any PyTorch device that follows the standard device-module convention.
    Returns None for CPU-only tensors or devices that don't support events.
    """
    for t in tensor_refs.values():
        if t.device.type == "cpu":
            continue
        device_mod = getattr(torch, t.device.type, None)
        if device_mod is not None and hasattr(device_mod, "Event"):
            try:
                ev = device_mod.Event()
                ev.record()
                return ev
            except Exception:
                pass
        break
    return None


def _extract_tensor_refs(
    args: tuple, kwargs: dict, *, is_output: bool = False
) -> Dict[str, "torch.Tensor"]:
    """Extract detached tensor references using the same key scheme as ``_dump_input`` / ``_dump_output`` metadata."""
    refs: Dict[str, torch.Tensor] = {}
    if is_output:
        result = args[0] if args else None
        if isinstance(result, tuple):
            for i, v in enumerate(result):
                if isinstance(v, torch.Tensor):
                    refs[f"result_{i}"] = v.detach()
        elif isinstance(result, torch.Tensor):
            refs["result"] = result.detach()
    else:
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                refs[f"arg_{i}"] = arg.detach()
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                refs[f"kwarg_{k}"] = v.detach()
    return refs


def _get_or_open_handle(json_path: str) -> "_WipHandle":
    """Return the ``_WipHandle`` for *json_path*, opening it if necessary."""
    with _open_files_lock:
        if json_path not in _open_file_handles:
            fh = open(json_path, "a", encoding="utf-8")  # noqa: WPS515
            _open_file_handles[json_path] = _WipHandle(fh)
        return _open_file_handles[json_path]


def _bg_compute_and_write(
    json_path: str,
    call_num: int,
    meta_skeleton: Dict[str, Any],
    tensor_refs: Dict[str, "torch.Tensor"],
    device_event: Optional[Any],
) -> None:
    """Background: sync device stream, compute tensor stats, append JSON line.

    Lines are written to *json_path* in ascending ``call_num`` order even when
    background tasks complete out of order.  A small per-file reorder buffer
    holds at most ``max_workers`` entries at a time (effectively O(1)).
    """
    if device_event is not None:
        device_event.synchronize()
    tensors_meta: Dict[str, Any] = {}
    for k, t in tensor_refs.items():
        try:
            tensors_meta[k] = tensor_stats(t)
        except Exception:
            pass
    meta_skeleton["tensors"] = tensors_meta

    call_key = f"call_{call_num}"
    line = json.dumps({call_key: meta_skeleton}, ensure_ascii=False) + "\n"
    handle = _get_or_open_handle(json_path)
    with handle.lock:
        handle.pending[call_num] = line
        while handle.next_write in handle.pending:
            handle.fh.write(handle.pending.pop(handle.next_write))
            handle.next_write += 1
        handle.fh.flush()


def _bg_save_pt(
    args: tuple,
    kwargs: dict,
    *,
    is_output: bool,
    pt_path: str,
    device_event: Optional[Any],
) -> None:
    """Background: sync device stream, serialize tensors to CPU, save .pt."""
    if device_event is not None:
        device_event.synchronize()
    torch.save(_build_data(args, kwargs, is_output=is_output), pt_path)


def _close_open_files() -> None:
    """Close all open JSON Lines file handles at step end."""
    with _open_files_lock:
        handles = dict(_open_file_handles)
        _open_file_handles.clear()
    for handle in handles.values():
        try:
            handle.fh.flush()
            handle.fh.close()
        except OSError:
            pass


def _wait_and_flush() -> None:
    """Drain all background futures, log any exceptions, then close file handles."""
    with _pending_lock:
        pending = list(_pending_futures)
        _pending_futures.clear()
    if pending:
        concurrent.futures.wait(pending)
        for fut in pending:
            exc = fut.exception()
            if exc is not None:
                logger.warning("Background IO task failed: %s", exc, exc_info=exc)
    _close_open_files()


# ── Dump I/O ──


def _dump_input(op_name: str, args: tuple, kwargs: dict,
                exec_order: int,
                label: Optional[str] = None,
                module_tag: str = "",
                op_tag: str = "",
                dispatch_keys: Optional[List[Tuple[str, str, bool]]] = None) -> None:
    """Queue operator inputs for async stats computation and JSON buffering.

    Records a device stream event in the main thread, then submits background
    tasks to compute tensor statistics and buffer the JSON entry.  Optionally
    queues a ``.pt`` save task when ``_with_values`` is enabled.
    """
    display = label or op_name
    try:
        order = exec_order
        call_num = _next_call_num(op_name)
        op_dir = _get_op_dir(op_name)
        os.makedirs(op_dir, exist_ok=True)

        _record_op_summary(op_name, dispatch_keys)
        _push_pairing(op_name, call_num, order, op_dir, label=display,
                      module_tag=module_tag, op_tag=op_tag)

        call_key = f"call_{call_num}"
        meta_skeleton: Dict[str, Any] = {
            "op_name": op_name,
            "exec_order": order,
            "call_num": call_num,
            "step": get_step(),
            "rank": get_rank(),
            "module_tag": module_tag,
            "op_tag": op_tag,
        }
        tensor_refs = _extract_tensor_refs(args, kwargs)
        device_event = _record_device_event(tensor_refs)
        json_path = os.path.join(op_dir, "input.json")
        _submit_bg(_bg_compute_and_write, json_path, call_num,
                   meta_skeleton, tensor_refs, device_event)

        if _with_values:
            pt_path = os.path.join(op_dir, f"call_{call_num}_input.pt")
            _submit_bg(_bg_save_pt, args, kwargs,
                       is_output=False, pt_path=pt_path, device_event=device_event)

        logger.debug(f"Queued input dump: {op_dir} [{call_key}]")
    except Exception as e:
        logger.warning(f"Failed to queue input dump for '{display}': {e}")


def _dump_output(op_name: str, result: Any) -> None:
    """Queue operator outputs for async stats computation and JSON buffering.

    Mirrors ``_dump_input``: records a device stream event, submits background
    tasks for stats/JSON buffering, and optionally queues a ``.pt`` save.
    """
    try:
        pairing = _pop_pairing(op_name)
        if pairing is None:
            logger.warning(f"No pairing info for dump output '{op_name}', skipping")
            return
        (call_num, order, op_dir, label, module_tag, op_tag) = pairing

        call_key = f"call_{call_num}"
        meta_skeleton: Dict[str, Any] = {
            "op_name": op_name,
            "exec_order": order,
            "call_num": call_num,
            "step": get_step(),
            "rank": get_rank(),
            "module_tag": module_tag,
            "op_tag": op_tag,
        }
        tensor_refs = _extract_tensor_refs((result,), {}, is_output=True)
        device_event = _record_device_event(tensor_refs)
        json_path = os.path.join(op_dir, "output.json")
        _submit_bg(_bg_compute_and_write, json_path, call_num,
                   meta_skeleton, tensor_refs, device_event)

        if _with_values:
            pt_path = os.path.join(op_dir, f"call_{call_num}_output.pt")
            _submit_bg(_bg_save_pt, (result,), {},
                       is_output=True, pt_path=pt_path, device_event=device_event)

        logger.debug(f"Queued output dump: {op_dir} [{call_key}]")
    except Exception as e:
        logger.warning(f"Failed to queue output dump for '{op_name}': {e}")


# ── Public API ──


def is_dump_enabled() -> bool:
    """Check if IO dumping is enabled (fast path)."""
    return _enabled


def advance_io_step() -> None:
    """Advance the step counter if IO dumping is enabled.

    Called from model_runner after each inference cycle (forward + sampling).
    No-op when dumping is disabled — safe to call unconditionally.
    """
    if _enabled:
        advance_step()


def init_io_dump_from_env(eager: bool) -> None:
    """Initialize IO dumper from environment variables (called from model_runner).

    Checks eager mode, detects env/yaml config, parses it, and activates hooks.
    No-op when IO is not configured or when not in eager mode.
    """
    if _enabled or not eager:
        return
    if not (any(k.startswith("VLLM_FL_IO_DUMP") for k in os.environ)
            or os.environ.get("VLLM_FL_CONFIG", "").strip()):
        return
    _init_from_env()   # sets _enabled = True, parses config
    _activate_hooks()  # registers step callback, enters dispatch modes


def dump_before(op_name: str, args: tuple, kwargs: dict,
                exec_order: Optional[int] = None,
                module_tag: Optional[str] = None,
                op_tag: Optional[str] = None) -> None:
    """Dump operator inputs and optionally capture them for print logging.

    Args:
        exec_order: Pre-allocated execution order shared across subsystems.
        module_tag: Accepted for API compatibility with OpManager but not
            used — structured module tag is always built from hook context.
        op_tag: Pre-computed op counter tag (e.g. ``[op=3,2]``).
            When *None*, generated internally.
    """
    if guard_active():
        return
    if not _rank_ok():
        return
    if not _should_dump(op_name, args):
        return

    # Summary-only mode: just record the op for summary, skip per-op dump
    if not _with_metas:
        record_seen(op_name, args)
        _record_op_summary(op_name, None)
        return

    label = make_label(op_name, args)
    # Always build structured module_tag from current hook context
    mod_name = get_module_class_name(args) or get_current_module() or ""
    mod_path = get_current_module_path()
    _module_tag = make_module_tag_from_ctx(mod_name, mod_path, for_json=True)
    _op_tag = op_tag if op_tag is not None else make_op_tag(op_name)
    # Allocate exec_order once so file metadata and print pairing share the same value.
    order = exec_order if exec_order is not None else next_exec_order()
    record_seen(op_name, args)
    set_guard(True)
    try:
        # File dump (only if dump_dir is set)
        if _dump_dir:
            _dump_input(op_name, args, kwargs, exec_order=order, label=label,
                        module_tag=_module_tag, op_tag=_op_tag)
        # Print: capture inputs for console logging in dump_after
        if _print_enabled:
            input_lines = _format_inputs(args, kwargs, skip_module_arg=True)
            _push_print_pairing(op_name, label, order, input_lines, _op_tag)
    finally:
        set_guard(False)


def dump_after(op_name: str, args: tuple, result: Any) -> None:
    """Dump operator outputs and optionally log them to console."""
    if guard_active():
        return
    if not _rank_ok():
        return
    if not _should_dump(op_name, args):
        return
    if not _with_metas:
        return
    set_guard(True)
    try:
        # File dump (only if dump_dir is set)
        if _dump_dir:
            _dump_output(op_name, result)
        # Print: log consolidated INPUTS+OUTPUTS
        if _print_enabled:
            pairing = _pop_print_pairing(op_name)
            if pairing:
                label, order, input_lines, op_tag = pairing
                _log_combined(label, input_lines, result,
                              op_tag=op_tag, exec_order=order)
            else:
                op_tag = make_op_tag(op_name)
                _log_outputs_only(make_label(f"Op '{op_name}'", args), result,
                                  op_tag=op_tag)
    finally:
        set_guard(False)


def dump_cleanup(op_name: str) -> None:
    """Pop stale pairings left by dump_before when the op raises.

    Called from ``_call_with_hooks`` (and TorchFunctionMode) when the
    actual operator execution fails so that the pairing stacks stay
    clean for subsequent calls.
    """
    _pop_pairing(op_name)
    _pop_print_pairing(op_name)


def io_dump_step() -> int:
    """Increment step counter and reset per-op call counters.

    Uses the shared step counter from io_common.  The ``_on_step_advance``
    callback registered by ``_activate_hooks`` clears per-op call counters.
    """
    return advance_step()


def enable_io_dump(
    dump_dir=_UNSET,
    ops=_UNSET,
    modules=_UNSET,
    layers=_UNSET,
    max_calls=_UNSET,
    step_range=_UNSET,
    with_torch_funcs=_UNSET,
    ranks=_UNSET,
    with_values=_UNSET,
    with_metas=_UNSET,
    with_print=_UNSET,
    bg_workers=_UNSET,
) -> None:
    """
    Programmatically enable IO dumping (and optionally console printing).

    All filter dimensions are composable (AND logic): when multiple
    filters are set, an op must satisfy ALL of them to be dumped.

    Uses a 3-layer merge strategy (env < yaml < api): each parameter is
    resolved by starting with sensible defaults, overlaying env-var values,
    then YAML config values, then API values (if not ``_UNSET``).  This
    means env vars and YAML settings are respected for any parameter not
    explicitly passed by the caller.

    Args:
        dump_dir: Directory to save dump files.  Unset → ``./io_dump``.
        ops: Dispatch-managed op names to dump. Unset/``None`` = all.
        modules: nn.Module class names to scope dumping to.
            Unset/``None`` = no module scoping (dump everywhere).
        layers: Layer specifications to scope dumping to.  Supports
            integer shorthand (``"0"`` → ``"model.layers.0"``),
            ranges (``"0-3"``), glob patterns (``"model.layers.*.self_attn"``),
            and full paths.  Unset/``None`` = no layer scoping.
        max_calls: Max calls per op to dump.  Unset/``0`` = unlimited.
        step_range: Inclusive step range string.  ``"0-2"`` means
            steps 0, 1, 2.  Unset/``None`` = all steps.
        with_torch_funcs: Intercept bare torch functional ops via
            TorchFunctionMode (eager mode only).  Unset → ``False``.
        ranks: Set of ranks to dump on.  Unset/``None`` = all ranks.
        with_values: If True, write per-call ``.pt`` tensor data files.
            Unset → ``False``.
        with_metas: If True, write per-op input/output ``.json`` files.
            When False, only ``summary.json`` is written.  Unset → ``False``.
        with_print: If True, enable console logging of operator
            inputs/outputs.  Unset → ``False``.
        bg_workers: Number of background threads for stats/I/O work.
            Unset → ``8`` (or ``VLLM_FL_IO_DUMP_BG_WORKERS`` env var).
    """
    global _enabled, _dump_dir, _match_all, _op_filter, _module_filter, _layer_filter
    global _max_calls, _step_range, _torch_funcs_enabled, _torch_func_filter, _with_values
    global _rank_filter, _with_metas, _print_enabled, _bg_workers

    # ── Layer 0: defaults ──
    r_dump_dir = os.path.join(os.getcwd(), "io_dump")
    r_ops: Optional[Set[str]] = None
    r_modules: Optional[Set[str]] = None
    r_layers = None          # None = all
    r_max_calls = 0          # 0 = unlimited
    r_step_range: Optional[Tuple[int, int]] = None  # half-open tuple
    r_torch_funcs = False
    r_ranks: Optional[Set[int]] = None
    r_with_values = False
    r_with_metas = False
    r_print_io = False
    r_bg_workers = 8

    # ── Layer 1: env vars ──
    env_dump_dir = os.environ.get("VLLM_FL_IO_DUMP", "").strip()
    if env_dump_dir and env_dump_dir not in ("0", "1"):
        r_dump_dir = env_dump_dir
    elif env_dump_dir == "1":
        pass  # keep default cwd-based dir

    env_ops = os.environ.get("VLLM_FL_IO_DUMP_OPS", "").strip()
    if env_ops:
        r_ops = {t.strip() for t in env_ops.split(",") if t.strip()}

    env_modules = os.environ.get("VLLM_FL_IO_DUMP_MODULES", "").strip()
    if env_modules:
        r_modules = {t.strip() for t in env_modules.split(",") if t.strip()}

    env_layers = parse_layers_env("VLLM_FL_IO_DUMP_LAYERS")
    if env_layers:
        r_layers = env_layers

    env_max_calls = os.environ.get("VLLM_FL_IO_DUMP_MAX_CALLS", "").strip()
    if env_max_calls:
        try:
            r_max_calls = int(env_max_calls)
        except ValueError:
            pass

    env_sr = parse_step_range_env("VLLM_FL_IO_DUMP_STEP_RANGE")
    if env_sr is not None:
        r_step_range = env_sr

    r_torch_func_filter: Set[str] = set()
    env_tf = os.environ.get("VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", "").strip()
    if env_tf:
        r_torch_funcs, r_torch_func_filter = parse_torch_funcs_config(env_tf)

    env_rank = os.environ.get("VLLM_FL_IO_DUMP_RANK", "").strip()
    if env_rank:
        r_ranks = parse_rank_filter(env_rank)

    if os.environ.get("VLLM_FL_IO_DUMP_WITH_VALUES", "").strip().lower() in ("1", "true"):
        r_with_values = True

    if os.environ.get("VLLM_FL_IO_DUMP_WITH_METAS", "").strip().lower() in ("1", "true"):
        r_with_metas = True

    env_print = os.environ.get("VLLM_FL_IO_DUMP_WITH_PRINT", "").strip().lower()
    if env_print in ("1", "true"):
        r_print_io = True
    elif env_print in ("0", "false"):
        r_print_io = False

    env_bg_workers = os.environ.get("VLLM_FL_IO_DUMP_BG_WORKERS", "").strip()
    if env_bg_workers:
        try:
            r_bg_workers = max(1, int(env_bg_workers))
        except ValueError:
            pass

    # ── Layer 2: YAML config ──
    config_path = os.environ.get("VLLM_FL_CONFIG", "").strip()
    if config_path:
        io_cfg = parse_io_config_from_yaml(config_path).get("io_dump")
        if io_cfg is not None:
            if io_cfg.get("dir"):
                r_dump_dir = io_cfg["dir"]
            if io_cfg.get("ops"):
                r_ops = set(io_cfg["ops"])
            if io_cfg.get("modules"):
                r_modules = set(io_cfg["modules"])
            if io_cfg.get("layers"):
                r_layers = set(io_cfg["layers"])
            if io_cfg.get("max_calls", 0):
                r_max_calls = io_cfg["max_calls"]
            if io_cfg.get("step_range") is not None:
                r_step_range = io_cfg["step_range"]
            if "with_torch_funcs" in io_cfg:
                r_torch_funcs, r_torch_func_filter = io_cfg["with_torch_funcs"]
            if io_cfg.get("ranks") is not None:
                r_ranks = io_cfg["ranks"]
            if "with_values" in io_cfg:
                r_with_values = io_cfg["with_values"]
            if "with_metas" in io_cfg:
                r_with_metas = io_cfg["with_metas"]
            if "with_print" in io_cfg:
                r_print_io = io_cfg["with_print"]

    # ── Layer 3: API overrides (only when not _UNSET) ──
    if dump_dir is not _UNSET:
        r_dump_dir = dump_dir if dump_dir else os.path.join(os.getcwd(), "io_dump")
    if ops is not _UNSET:
        r_ops = ops
    if modules is not _UNSET:
        r_modules = modules
    if layers is not _UNSET:
        r_layers = layers
    if max_calls is not _UNSET:
        r_max_calls = max_calls
    if step_range is not _UNSET:
        r_step_range = parse_step_range(step_range) if isinstance(step_range, str) else step_range
    if with_torch_funcs is not _UNSET:
        r_torch_funcs = with_torch_funcs
    if ranks is not _UNSET:
        r_ranks = ranks
    if with_values is not _UNSET:
        r_with_values = with_values
    if with_metas is not _UNSET:
        r_with_metas = with_metas
    if with_print is not _UNSET:
        r_print_io = with_print
    if bg_workers is not _UNSET:
        r_bg_workers = max(1, int(bg_workers))

    # If print-only (no explicit dump_dir), skip file I/O
    if r_print_io and dump_dir is _UNSET and not os.environ.get("VLLM_FL_IO_DUMP", "").strip():
        r_dump_dir = ""

    # ── Apply resolved config ──
    _dump_dir = r_dump_dir
    if _dump_dir:
        os.makedirs(_dump_dir, exist_ok=True)

    if r_ops is None and r_modules is None:
        _match_all = True
        _op_filter = set()
        _module_filter = set()
    else:
        _match_all = False
        _op_filter = set(r_ops) if r_ops else set()
        _module_filter = set(r_modules) if r_modules else set()

    if r_layers is None:
        _layer_filter = set()
    else:
        if isinstance(r_layers, str):
            r_layers = {r_layers}
        _layer_filter = expand_layer_specs(r_layers)

    _max_calls = r_max_calls
    _step_range = r_step_range
    _torch_funcs_enabled = r_torch_funcs
    _torch_func_filter = r_torch_func_filter
    _with_values = r_with_values
    _with_metas = r_with_metas
    _rank_filter = r_ranks
    _print_enabled = r_print_io
    _bg_workers = r_bg_workers
    _enabled = True
    set_io_active(True)
    _activate_hooks()

    # Propagate resolved config to env vars so child processes
    # (e.g. vLLM EngineCore workers) inherit via _init_from_env().
    _resolved_ops = _op_filter if not _match_all else None
    _resolved_modules = _module_filter if not _match_all else None
    _set_env_vars(_dump_dir, _resolved_ops, _resolved_modules, _layer_filter,
                  _max_calls, _step_range, _torch_funcs_enabled, _rank_filter,
                  _with_values, _with_metas, _print_enabled)

    logger.info(
        f"IO Dump enabled: rank={get_rank()}, "
        f"rank_filter={_rank_filter or 'all'}, dir={_dump_dir}, "
        f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
        f"layers={_layer_filter or 'all'}, "
        f"max_calls={_max_calls}, step_range={_step_range}, "
        f"with_torch_funcs={_torch_funcs_enabled}, with_values={_with_values}, "
        f"with_metas={_with_metas}, with_print={_print_enabled}"
    )


def disable_io_dump() -> None:
    """Programmatically disable IO dumping and remove all hooks."""
    # Set _enabled=False first so handlers and _submit_bg stop accepting new work
    # immediately, even if dispatch/function modes can't be exited right away due
    # to LIFO constraints (they short-circuit on _enabled=False).
    global _enabled
    _enabled = False
    set_io_active(False)
    _deactivate_hooks()
    # Drain all in-flight background tasks (no new submissions possible now).
    _wait_and_flush()
    # Write summary before _reset_state clears _op_summary.
    _write_summary()
    _reset_state()
    _clear_env_vars()


# ── Env-var propagation for child processes ──


def _set_env_vars(
    dump_dir: str,
    ops: Optional[Set[str]],
    modules: Optional[Set[str]],
    layers: Set[str],
    max_calls: int,
    step_range: Optional[Tuple[int, int]],
    torch_funcs: bool,
    ranks: Optional[Set[int]],
    with_values: bool,
    with_metas: bool,
    print_on: bool = False,
) -> None:
    """Set VLLM_FL_IO_DUMP* env vars so child processes inherit the resolved config."""
    os.environ["VLLM_FL_IO_DUMP"] = dump_dir

    if ops:
        os.environ["VLLM_FL_IO_DUMP_OPS"] = ",".join(sorted(ops))
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_OPS", None)

    if modules:
        os.environ["VLLM_FL_IO_DUMP_MODULES"] = ",".join(sorted(modules))
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_MODULES", None)

    if layers:
        os.environ["VLLM_FL_IO_DUMP_LAYERS"] = ",".join(sorted(layers))
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_LAYERS", None)

    if max_calls > 0:
        os.environ["VLLM_FL_IO_DUMP_MAX_CALLS"] = str(max_calls)
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_MAX_CALLS", None)

    if step_range is not None:
        os.environ["VLLM_FL_IO_DUMP_STEP_RANGE"] = f"{step_range[0]}-{step_range[1] - 1}"
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_STEP_RANGE", None)

    if torch_funcs:
        os.environ["VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS"] = "1"
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", None)

    if ranks is not None:
        os.environ["VLLM_FL_IO_DUMP_RANK"] = ",".join(str(r) for r in sorted(ranks))
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_RANK", None)

    if with_values:
        os.environ["VLLM_FL_IO_DUMP_WITH_VALUES"] = "1"
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_WITH_VALUES", None)

    if with_metas:
        os.environ["VLLM_FL_IO_DUMP_WITH_METAS"] = "1"
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_WITH_METAS", None)

    if print_on:
        os.environ["VLLM_FL_IO_DUMP_WITH_PRINT"] = "1"
    else:
        os.environ.pop("VLLM_FL_IO_DUMP_WITH_PRINT", None)


def _clear_env_vars() -> None:
    """Remove VLLM_FL_IO_DUMP* env vars."""
    for key in [
        "VLLM_FL_IO_DUMP", "VLLM_FL_IO_DUMP_OPS", "VLLM_FL_IO_DUMP_MODULES",
        "VLLM_FL_IO_DUMP_LAYERS", "VLLM_FL_IO_DUMP_MAX_CALLS",
        "VLLM_FL_IO_DUMP_STEP_RANGE", "VLLM_FL_IO_DUMP_RANK",
        "VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", "VLLM_FL_IO_DUMP_WITH_VALUES",
        "VLLM_FL_IO_DUMP_WITH_METAS", "VLLM_FL_IO_DUMP_WITH_PRINT",
    ]:
        os.environ.pop(key, None)


def _format_dispatch_keys_for_json(
    dispatch_keys: List[Tuple[str, str, bool]],
) -> str:
    """Format dispatch keys as a string for JSON metadata."""
    items = [f"({key}, {impl}, {is_default})" for key, impl, is_default in dispatch_keys]
    return f"[{', '.join(items)}]"


# ── Summary accumulator ──


def _record_op_summary(
    op_name: str,
    dispatch_keys: Optional[List[Tuple[str, str, bool]]],
) -> None:
    """Accumulate op metadata for the final summary.json."""
    with _lock:
        entry = _op_summary.get(op_name)
        if entry is None:
            dk_str = (_format_dispatch_keys_for_json(dispatch_keys)
                      if dispatch_keys else "")
            entry = {
                "dispatch_keys": dk_str,
                "call_count": 0,
            }
            _op_summary[op_name] = entry
        entry["call_count"] += 1


def _is_flaggems_op(op_name: str, dispatch_keys: str) -> bool:
    """Check if the op is backed by FlagGems.

    Two detection paths:
    1. Dispatch table: the dispatch_keys string contains "FlagGems"
       (ATen ops that FlagGems registered a kernel for).
    2. OpManager: the resolved implementation is a flagos backend
       (dispatch-managed ops like rms_norm, silu_and_mul, rotary_embedding).
    """
    if "FlagGems" in dispatch_keys:
        return True
    # Fallback: check OpManager for dispatch-managed ops
    try:
        from .manager import get_default_manager
        from .types import BackendImplKind
        mgr = get_default_manager()
        impl_id = mgr._called_ops.get(op_name)
        if impl_id:
            snap = mgr._registry.snapshot()
            for imp in snap.impls_by_op.get(op_name, []):
                if imp.impl_id == impl_id:
                    return imp.kind == BackendImplKind.DEFAULT
    except Exception:
        pass
    return False


def _write_summary() -> None:
    """Write summary.json under the rank directory.

    Produces two sections:
    - ``flaggems_ops``: operators with at least one FlagGems dispatch key
    - ``non_flaggems_ops``: all other operators
    """
    if not _dump_dir or not _op_summary:
        return

    rank_dir = os.path.join(_dump_dir, f"rank_{get_rank():04d}")
    os.makedirs(rank_dir, exist_ok=True)

    flaggems_ops: List[str] = []
    non_flaggems_ops: List[str] = []

    for op_name in sorted(_op_summary):
        entry = _op_summary[op_name]
        if _is_flaggems_op(op_name, entry["dispatch_keys"]):
            flaggems_ops.append(op_name)
        else:
            non_flaggems_ops.append(op_name)

    summary = {
        "rank": get_rank(),
        "flaggems_ops": flaggems_ops,
        "non_flaggems_ops": non_flaggems_ops,
    }

    summary_path = os.path.join(rank_dir, "summary.json")
    try:
        tmp_path = summary_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp_path, summary_path)
        logger.info(
            f"[IO_DUMP] Summary written: {summary_path} "
            f"({len(flaggems_ops)} FlagGems, "
            f"{len(non_flaggems_ops)} non-FlagGems)"
        )
    except OSError as exc:
        logger.warning(f"Failed to write summary: {exc}")


# ── TorchDispatchMode (default — works in both eager and compile modes) ──


if HAS_TORCH_DISPATCH_MODE:
    class _DumpDispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if not _enabled or guard_active() or not _rank_ok():
                return func(*args, **kwargs)

            op_name = get_dispatch_op_name(func)
            if not should_inspect_dispatch_op(op_name, _match_all, _op_filter):
                return func(*args, **kwargs)

            # Step range check
            if _step_range is not None:
                step = get_step()
                if step < _step_range[0] or step >= _step_range[1]:
                    return func(*args, **kwargs)

            # Summary-only mode: just record the op for summary.json
            if not _with_metas:
                ns = get_dispatch_op_namespace(func)
                raw_name = f"{ns}.{op_name}"
                dispatch_keys = get_dispatch_keys(func, args, kwargs)
                _record_op_summary(raw_name, dispatch_keys)
                record_seen(raw_name)
                return func(*args, **kwargs)

            # Layer filter check
            if _layer_filter and not layer_path_matches(_layer_filter):
                return func(*args, **kwargs)

            # Module filter check
            if _module_filter and not _match_all:
                if not module_context_matches(_module_filter):
                    return func(*args, **kwargs)

            mod_name = get_current_module() or ""
            mod_path = get_current_module_path() or ""

            ns = get_dispatch_op_namespace(func)
            raw_name = f"{ns}.{op_name}"
            if not _check_limits(raw_name):
                return func(*args, **kwargs)

            dispatch_keys = get_dispatch_keys(func, args, kwargs)
            label = make_label(raw_name, module_name=mod_name or None,
                               layer_path=mod_path or None)
            order = next_exec_order()
            module_tag = make_module_tag_from_ctx(mod_name, mod_path, for_json=True)
            op_tag = make_op_tag(raw_name)
            record_seen(raw_name, module_name=mod_name or None)

            set_guard(True)
            try:
                if _dump_dir:
                    _dump_input(raw_name, args, kwargs, exec_order=order,
                                label=label, module_tag=module_tag, op_tag=op_tag,
                                dispatch_keys=dispatch_keys)
                if _print_enabled:
                    input_lines = _format_inputs(args, kwargs)
            finally:
                set_guard(False)

            try:
                result = func(*args, **kwargs)
            except Exception:
                _pop_pairing(raw_name)
                raise

            set_guard(True)
            try:
                if _dump_dir:
                    _dump_output(raw_name, result)
                if _print_enabled:
                    _log_combined(label, input_lines, result,
                                  op_tag=op_tag, exec_order=order)
            finally:
                set_guard(False)

            return result


# ── TorchFunctionMode (opt-in for bare torch ops) ──


if HAS_TORCH_FUNC_MODE:
    class _DumpTorchFuncMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if torch.compiler.is_compiling() or not _enabled or guard_active() or not _rank_ok():
                return func(*args, **kwargs)

            func_name = get_torch_func_name(func)

            # Summary-only mode: just record and pass through
            if not _with_metas:
                raw_name = f"torch.{func_name}"
                record_seen(raw_name)
                return func(*args, **kwargs)

            if not _should_dump_torch_func(func_name):
                return func(*args, **kwargs)

            # Use raw name for pairing/limits, annotated label for metadata
            raw_name = f"torch.{func_name}"
            if not _check_limits(raw_name):
                return func(*args, **kwargs)
            mod_name = get_current_module() or ""
            mod_path = get_current_module_path() or ""
            label = make_label(raw_name, module_name=mod_name or None,
                               layer_path=mod_path or None)
            _, op_tag, order = acquire_torch_func_tags(raw_name)
            record_seen(raw_name, module_name=mod_name or None)

            set_guard(True)
            try:
                if _dump_dir:
                    _dump_input(raw_name, args, kwargs, exec_order=order,
                                label=label,
                                module_tag=make_module_tag_from_ctx(mod_name, mod_path, for_json=True),
                                op_tag=op_tag)
                if _print_enabled:
                    input_lines = _format_inputs(args, kwargs)
            finally:
                set_guard(False)

            try:
                result = func(*args, **kwargs)
            except Exception:
                # Clean up stale pairing pushed by _dump_input
                _pop_pairing(raw_name)
                release_torch_func_tags()
                raise

            set_guard(True)
            try:
                if _dump_dir:
                    _dump_output(raw_name, result)
                if _print_enabled:
                    _log_combined(label, input_lines, result,
                                  op_tag=op_tag, exec_order=order)
            finally:
                set_guard(False)

            release_torch_func_tags()
            return result


# ── Hook lifecycle ──


def _enter_dispatch_modes():
    """Enter dispatch/function modes via the shared ModeManager."""
    if HAS_TORCH_DISPATCH_MODE:
        dispatch_mode_mgr.enter("dump", _DumpDispatchMode())
    if _torch_funcs_enabled and HAS_TORCH_FUNC_MODE:
        func_mode_mgr.enter("dump", _DumpTorchFuncMode())


def _exit_dispatch_modes():
    """Request exit of dump modes via the shared ModeManager."""
    dispatch_mode_mgr.request_exit("dump")
    func_mode_mgr.request_exit("dump")


def pause_dispatch_modes():
    """Temporarily exit dispatch modes (e.g. during CUDA graph capture).

    Call ``resume_dispatch_modes()`` to re-enter after the incompatible
    phase completes.  Safe to call when modes are not entered (no-op).
    """
    if dispatch_mode_mgr.is_entered("dump"):
        dispatch_mode_mgr.request_exit("dump")
    if func_mode_mgr.is_entered("dump"):
        func_mode_mgr.request_exit("dump")


def resume_dispatch_modes():
    """Re-enter dispatch modes after ``pause_dispatch_modes()``.

    Only re-enters if dumping is still enabled and hooks were activated.
    """
    if _enabled and _hooks_activated and not dispatch_mode_mgr.is_entered("dump"):
        _enter_dispatch_modes()


def register_io_module_hooks(model: torch.nn.Module) -> None:
    """Register module paths and install forward hooks for thread-local context.

    Builds the ``id(module) → path`` map (needed by ``get_module_path`` and
    ``push_module_context``) then installs pre/post forward hooks on every
    module so that ``get_current_module`` / ``get_current_module_path`` return
    the correct values inside ``__torch_dispatch__``.

    No-op when IO dumping is not enabled.
    Safe to call multiple times — removes any previously installed handles first.
    Must be called after the model is fully constructed.
    """
    if not _enabled:
        return
    register_module_paths(model)
    _remove_module_hooks()
    for module in model.modules():
        cls_name = type(module).__name__
        h_pre = module.register_forward_pre_hook(
            lambda m, _args, cls=cls_name: push_module_context(cls, m)
        )
        h_post = module.register_forward_hook(
            lambda _m, _args, _out: pop_module_context(),
            always_call=True,
        )
        _module_hook_handles.append(h_pre)
        _module_hook_handles.append(h_post)


def _remove_module_hooks() -> None:
    """Remove all module forward hooks installed by ``register_io_module_hooks``."""
    for handle in _module_hook_handles:
        handle.remove()
    _module_hook_handles.clear()


def _activate_hooks():
    """Activate TorchDispatchMode and optionally TorchFunctionMode."""
    global _hooks_activated

    # Register callback first — always needed for step tracking
    register_step_callback(_on_step_advance)

    # Defer dispatch mode if step_range starts later than step 0
    if _step_range is not None and _step_range[0] > 0:
        _hooks_activated = True
        return  # will be lazily activated by step callback

    # Activate immediately (no step_range, or starts at step 0)
    _enter_dispatch_modes()
    _hooks_activated = True


def _deactivate_hooks():
    """Remove step callback, module forward hooks, and request exit of dispatch modes.

    The shared ``ModeManager`` handles LIFO ordering — modes are only
    actually exited when they are on top of the stack.
    """
    global _hooks_activated

    unregister_step_callback(_on_step_advance)
    _remove_module_hooks()
    _exit_dispatch_modes()
    _hooks_activated = False


# ── State management ──


def _reset_state() -> None:
    """Reset all module-level state to defaults."""
    global _enabled, _dump_dir, _match_all, _op_filter, _module_filter, _layer_filter
    global _max_calls, _step_range, _with_values, _with_metas
    global _torch_funcs_enabled, _torch_func_filter, _rank_filter
    global _print_enabled, _io_executor, _bg_workers

    _enabled = False
    _dump_dir = ""
    _match_all = False
    _op_filter = set()
    _module_filter = set()
    _layer_filter = set()
    _max_calls = 0
    _step_range = None
    _with_values = False
    _with_metas = False
    _torch_funcs_enabled = False
    _torch_func_filter = set()
    _rank_filter = None
    _print_enabled = False
    _bg_workers = 8
    with _lock:
        _call_counters.clear()
        _op_summary.clear()
    # Shutdown background executor and clear async state
    exec_ = _io_executor
    _io_executor = None
    if exec_ is not None and not exec_._shutdown:
        exec_.shutdown(wait=False)
    with _pending_lock:
        _pending_futures.clear()
    with _open_files_lock:
        for handle in _open_file_handles.values():
            try:
                handle.fh.close()
            except OSError:
                pass
        _open_file_handles.clear()


def _init_from_env() -> None:
    """Initialize from environment variables and/or YAML config.

    Uses a 2-layer merge (env < yaml) — the same config strategy as
    ``enable_io_dump()`` but without the API layer.

    Skipped when the programmatic API (``enable_io_dump``) has already
    been called — the Python API has the highest priority.
    """
    global _enabled, _dump_dir, _match_all, _op_filter, _module_filter, _layer_filter
    global _max_calls, _step_range, _torch_funcs_enabled, _torch_func_filter, _with_values
    global _rank_filter, _with_metas, _print_enabled

    if _enabled:
        return

    _deactivate_hooks()

    # ── Determine if enabled ──
    env_dump_dir = os.environ.get("VLLM_FL_IO_DUMP", "").strip()
    yaml_enabled = False
    config_path = os.environ.get("VLLM_FL_CONFIG", "").strip()
    yaml_cfg = None
    if config_path:
        yaml_cfg = parse_io_config_from_yaml(config_path).get("io_dump")
        if yaml_cfg is not None:
            yaml_enabled = bool(yaml_cfg.get("dir"))

    # Check if print-only mode is requested via env
    env_print_val = os.environ.get("VLLM_FL_IO_DUMP_WITH_PRINT", "").strip().lower()
    _env_print_requested = env_print_val in ("1", "true")

    if env_dump_dir == "0":
        _reset_state()
        return
    if not env_dump_dir and not yaml_enabled and not _env_print_requested:
        # Auto-enable when dumper-specific filter env vars are set
        _has_filters = any(
            os.environ.get(v, "").strip()
            for v in (
                "VLLM_FL_IO_DUMP_STEP_RANGE", "VLLM_FL_IO_DUMP_LAYERS",
            )
        )
        if not _has_filters:
            _reset_state()
            return
        env_dump_dir = "1"

    # ── Layer 0: defaults ──
    r_dump_dir = os.path.join(os.getcwd(), "io_dump")
    r_ops: Optional[Set[str]] = None
    r_modules: Optional[Set[str]] = None
    r_layers: Optional[Set[str]] = None
    r_max_calls = 0
    r_step_range: Optional[Tuple[int, int]] = None
    r_torch_funcs = False
    r_torch_func_filter: Set[str] = set()
    r_ranks: Optional[Set[int]] = None
    r_with_values = False
    r_with_metas = False
    r_print_io = False

    # ── Layer 1: env vars ──
    if env_dump_dir and env_dump_dir not in ("0", "1"):
        r_dump_dir = env_dump_dir
    # "1" keeps default cwd-based dir

    env_ops = os.environ.get("VLLM_FL_IO_DUMP_OPS", "").strip()
    if env_ops:
        r_ops = {t.strip() for t in env_ops.split(",") if t.strip()}

    env_modules = os.environ.get("VLLM_FL_IO_DUMP_MODULES", "").strip()
    if env_modules:
        r_modules = {t.strip() for t in env_modules.split(",") if t.strip()}

    env_layers = parse_layers_env("VLLM_FL_IO_DUMP_LAYERS")
    if env_layers:
        r_layers = env_layers

    env_max_calls = os.environ.get("VLLM_FL_IO_DUMP_MAX_CALLS", "").strip()
    if env_max_calls:
        try:
            r_max_calls = int(env_max_calls)
        except ValueError:
            pass

    env_sr = parse_step_range_env("VLLM_FL_IO_DUMP_STEP_RANGE")
    if env_sr is not None:
        r_step_range = env_sr

    env_tf = os.environ.get("VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", "").strip()
    if env_tf:
        r_torch_funcs, r_torch_func_filter = parse_torch_funcs_config(env_tf)

    env_rank = os.environ.get("VLLM_FL_IO_DUMP_RANK", "").strip()
    if env_rank:
        r_ranks = parse_rank_filter(env_rank)

    if os.environ.get("VLLM_FL_IO_DUMP_WITH_VALUES", "").strip().lower() in ("1", "true"):
        r_with_values = True

    if os.environ.get("VLLM_FL_IO_DUMP_WITH_METAS", "").strip().lower() in ("1", "true"):
        r_with_metas = True

    env_print_new = os.environ.get("VLLM_FL_IO_DUMP_WITH_PRINT", "").strip().lower()
    if env_print_new in ("1", "true"):
        r_print_io = True
    elif env_print_new in ("0", "false"):
        r_print_io = False

    # ── Layer 2: YAML config (overrides env) ──
    if yaml_cfg is not None:
        if yaml_cfg.get("dir"):
            r_dump_dir = yaml_cfg["dir"]
        if yaml_cfg.get("ops"):
            r_ops = set(yaml_cfg["ops"])
        if yaml_cfg.get("modules"):
            r_modules = set(yaml_cfg["modules"])
        if yaml_cfg.get("layers"):
            r_layers = set(yaml_cfg["layers"])
        if yaml_cfg.get("max_calls", 0):
            r_max_calls = yaml_cfg["max_calls"]
        if yaml_cfg.get("step_range") is not None:
            r_step_range = yaml_cfg["step_range"]
        if "with_torch_funcs" in yaml_cfg:
            r_torch_funcs, r_torch_func_filter = yaml_cfg["with_torch_funcs"]
        if yaml_cfg.get("ranks") is not None:
            r_ranks = yaml_cfg["ranks"]
        if "with_values" in yaml_cfg:
            r_with_values = yaml_cfg["with_values"]
        if "with_metas" in yaml_cfg:
            r_with_metas = yaml_cfg["with_metas"]
        if "with_print" in yaml_cfg:
            r_print_io = yaml_cfg["with_print"]

    # ── Apply resolved config ──
    # For print-only mode (no dump_dir), skip directory creation
    if r_print_io and not env_dump_dir and not yaml_enabled:
        _dump_dir = ""
    else:
        _dump_dir = r_dump_dir
        try:
            os.makedirs(_dump_dir, exist_ok=True)
        except OSError as exc:
            logger.warning(
                f"Cannot create dump directory '{_dump_dir}': {exc}. "
                "File dumping disabled."
            )
            _dump_dir = ""
            if not r_print_io:
                _reset_state()
                return

    if r_ops is None and r_modules is None:
        _match_all = True
        _op_filter = set()
        _module_filter = set()
    else:
        _match_all = False
        _op_filter = set(r_ops) if r_ops else set()
        _module_filter = set(r_modules) if r_modules else set()

    _layer_filter = expand_layer_specs(r_layers) if r_layers is not None else set()

    _max_calls = r_max_calls
    _step_range = r_step_range
    _torch_funcs_enabled = r_torch_funcs
    _torch_func_filter = r_torch_func_filter
    _with_values = r_with_values
    _with_metas = r_with_metas
    _rank_filter = r_ranks
    _print_enabled = r_print_io

    _enabled = True
    set_io_active(True)

    logger.info(
        f"IO Dump enabled (env/yaml): rank={get_rank()}, "
        f"rank_filter={_rank_filter or 'all'}, "
        f"dir={_dump_dir or '(none)'}, "
        f"ops={_op_filter or 'all'}, modules={_module_filter or 'all'}, "
        f"layers={_layer_filter or 'all'}, "
        f"max_calls={_max_calls}, step_range={_step_range}, "
        f"with_torch_funcs={_torch_funcs_enabled}, with_values={_with_values}, "
        f"with_metas={_with_metas}, with_print={_print_enabled}"
    )
