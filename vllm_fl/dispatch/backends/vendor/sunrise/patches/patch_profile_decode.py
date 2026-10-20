# Copyright (c) 2026 BAAI. All rights reserved.

"""Optional per-operator decode profiler (VLLM_FL_SUNRISE_PROFILE_DECODE=1)."""

from __future__ import annotations

import atexit
import os
import statistics
import sys
import time
from collections import defaultdict

import torch

_PROFILE_ENABLED = os.environ.get(
    "VLLM_FL_SUNRISE_PROFILE_DECODE", "0"
).lower() in ("1", "true", "yes", "on")

# Number of decode steps to capture after the first prefill step. The
# first step in any sequence is prefill (not interesting for decode
# perf); we mark prefills externally via ``mark_prefill_step()`` and
# only count post-prefill timings.
MAX_DECODE_STEPS = int(os.environ.get("VLLM_FL_SUNRISE_PROFILE_MAX_STEPS", "50"))


# Per-call durations in microseconds, keyed by op name. Each list entry
# is one CALL. Per-step rollups happen at summary time using the mark
# from ``begin_decode_step``.
_call_records: dict[str, list[float]] = defaultdict(list)
_step_call_indices: list[dict[str, slice]] = []
_step_total_us: list[float] = []
_total_steps_seen = 0
_decode_step_count = 0
_in_decode_step = False
_decode_step_start_us = 0.0
_step_start_call_indices: dict[str, int] = {}


def _now_us() -> float:
    return time.perf_counter() * 1e6


def _device_sync(device: torch.device | None) -> None:
    if device is None:
        return
    if device.type == "ptpu":
        try:
            torch.ptpu.synchronize(device.index)
        except Exception:
            pass


def _wrap_call(name: str, fn, *, infer_device_from=None):
    """Return a timed wrapper around ``fn``.

    ``infer_device_from`` is a callable ``(*args, **kwargs) -> torch.device``
    used to pick which PTPU device to sync. If None, no sync is performed
    (the caller is responsible for ensuring measurements are well-defined).
    """

    def _wrapped(*args, **kwargs):
        if not _in_decode_step:
            return fn(*args, **kwargs)
        device = None
        if infer_device_from is not None:
            try:
                device = infer_device_from(*args, **kwargs)
            except Exception:
                device = None
        _device_sync(device)
        t0 = _now_us()
        out = fn(*args, **kwargs)
        _device_sync(device)
        t1 = _now_us()
        _call_records[name].append(t1 - t0)
        return out

    return _wrapped


def begin_decode_step(device: torch.device | None) -> None:
    """Mark the start of a model.forward call. Call once per step."""
    global _in_decode_step, _decode_step_start_us, _step_start_call_indices, _total_steps_seen
    _total_steps_seen += 1
    if _decode_step_count >= MAX_DECODE_STEPS:
        _in_decode_step = False
        return
    _in_decode_step = True
    _device_sync(device)
    _decode_step_start_us = _now_us()
    _step_start_call_indices = {n: len(v) for n, v in _call_records.items()}


def end_decode_step(device: torch.device | None, was_decode_only: bool) -> None:
    """Mark the end of a model.forward call. ``was_decode_only`` filters
    out prefill steps; only decode-only steps are aggregated."""
    global _in_decode_step, _decode_step_count
    if not _in_decode_step:
        return
    _device_sync(device)
    t1 = _now_us()
    _in_decode_step = False
    if not was_decode_only:
        # Roll back: drop everything we recorded this step (it was prefill).
        for n, start_idx in _step_start_call_indices.items():
            del _call_records[n][start_idx:]
        return
    _step_total_us.append(t1 - _decode_step_start_us)
    # Capture per-step slice indices for later rollup.
    end_indices = {n: len(v) for n, v in _call_records.items()}
    slices = {}
    for n, start_idx in _step_start_call_indices.items():
        slices[n] = slice(start_idx, end_indices.get(n, start_idx))
    # Also include any new ops that fired only this step.
    for n, end_idx in end_indices.items():
        if n not in slices:
            slices[n] = slice(0, end_idx)
    _step_call_indices.append(slices)
    _decode_step_count += 1
    # Emit partial summaries so we get data without needing the process
    # to shut down gracefully (vLLM is often killed via SIGTERM/SIGKILL,
    # which doesn't trigger atexit). Print once at quarter, half, and
    # full capture; also dump to a sidecar file.
    if _decode_step_count == max(1, MAX_DECODE_STEPS // 4):
        _print_summary(prefix=f"[partial @ {_decode_step_count} steps]")
    if _decode_step_count == max(1, MAX_DECODE_STEPS // 2):
        _print_summary(prefix=f"[partial @ {_decode_step_count} steps]")
    if _decode_step_count == MAX_DECODE_STEPS:
        _print_summary(prefix=f"[final @ {_decode_step_count} steps]")


def _format_summary() -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(
        f"Sunrise decode-step profile (decode-only steps captured: "
        f"{_decode_step_count} / {MAX_DECODE_STEPS}, total forwards: {_total_steps_seen})"
    )
    if _decode_step_count == 0:
        lines.append("(no decode-only steps captured — profiler did not see any)")
        lines.append("=" * 78)
        return "\n".join(lines)

    total_avg = statistics.mean(_step_total_us) / 1000.0
    total_med = statistics.median(_step_total_us) / 1000.0
    total_p90 = sorted(_step_total_us)[int(len(_step_total_us) * 0.9)] / 1000.0
    lines.append(
        f"Total step time:   avg={total_avg:7.2f} ms   "
        f"median={total_med:7.2f} ms   p90={total_p90:7.2f} ms"
    )
    lines.append("-" * 78)
    lines.append(
        f"{'Operator':<45} {'count/step':>10} {'avg ms/call':>12} {'avg ms/step':>12} {'%step':>6}"
    )
    lines.append("-" * 78)

    # For each op, compute average count per step + average time per call.
    rows = []
    for name in sorted(_call_records.keys()):
        per_step_counts = []
        per_step_times = []
        for sl in _step_call_indices:
            s = sl.get(name)
            if s is None:
                per_step_counts.append(0)
                per_step_times.append(0.0)
                continue
            calls = _call_records[name][s]
            per_step_counts.append(len(calls))
            per_step_times.append(sum(calls))
        avg_count = statistics.mean(per_step_counts) if per_step_counts else 0
        avg_time_us = statistics.mean(per_step_times) if per_step_times else 0
        avg_per_call = avg_time_us / avg_count if avg_count > 0 else 0
        avg_time_ms = avg_time_us / 1000.0
        pct = 100.0 * avg_time_ms / total_avg if total_avg > 0 else 0.0
        rows.append((avg_time_ms, name, avg_count, avg_per_call / 1000.0, avg_time_ms, pct))

    rows.sort(reverse=True)  # sort by avg_time_ms descending
    sum_pct = 0.0
    for _, name, avg_count, ms_per_call, ms_per_step, pct in rows:
        sum_pct += pct
        lines.append(
            f"{name:<45} {avg_count:>10.1f} {ms_per_call:>12.3f} {ms_per_step:>12.3f} {pct:>5.1f}%"
        )
    lines.append("-" * 78)
    residual_ms = total_avg - sum_pct * total_avg / 100.0
    residual_pct = 100.0 - sum_pct
    lines.append(
        f"{'(uninstrumented residual / double-count noise)':<45} "
        f"{'':>10} {'':>12} {residual_ms:>12.3f} "
        f"{residual_pct:>5.1f}%"
    )
    if residual_pct > 90.0:
        lines.append(
            "Captured-graph runs attribute most model time to residual; "
            "use enforce-eager for per-op breakdown."
        )
    elif residual_pct < -20.0:
        lines.append(
            "Nested timers overlap; do not sum all rows for step time."
        )

    # P0.5: roll up native-INT8 hotspots so decode vs Linear/MoE/quant
    # attribution does not require grepping many shape buckets by hand.
    int8_prefixes = (
        ("int8.scaled_int8_quant", "int8.scaled_int8_quant"),
        ("int8.triton_scaled_mm", "int8.triton_scaled_mm"),
        ("int8.TritonExpertsFL.apply", "int8.TritonExpertsFL.apply"),
        ("op.invoke_fused_moe_triton_kernel", "op.invoke_fused_moe_triton_kernel"),
        ("MoERunner.forward", "MoERunner.forward"),
        ("FusedMoE.forward", "FusedMoE.forward"),
    )
    rollup_rows = []
    for prefix, label in int8_prefixes:
        matched = [
            (ms, name, cnt, mpc, mps, pct)
            for ms, name, cnt, mpc, mps, pct in rows
            if name == prefix or name.startswith(prefix + "[")
        ]
        if not matched:
            continue
        ms_step = sum(r[4] for r in matched)
        pct_sum = sum(r[5] for r in matched)
        calls = sum(r[2] for r in matched)
        rollup_rows.append((ms_step, label, calls, pct_sum))
    if rollup_rows:
        lines.append("-" * 78)
        lines.append("Native INT8 / MoE rollup (sum of matching buckets):")
        lines.append(
            f"{'Bucket':<45} {'count/step':>10} {'avg ms/step':>12} {'%step':>6}"
        )
        for ms_step, label, calls, pct_sum in sorted(rollup_rows, reverse=True):
            lines.append(
                f"{label:<45} {calls:>10.1f} {ms_step:>12.3f} {pct_sum:>5.1f}%"
            )
    lines.append("=" * 78)
    return "\n".join(lines)


_SUMMARY_FILE = os.environ.get(
    "VLLM_FL_SUNRISE_PROFILE_FILE", "/tmp/sunrise_profile.txt"
)


def _append_summary_file(text: str) -> None:
    try:
        with open(_SUMMARY_FILE, "a") as f:
            f.write(text)
    except Exception:
        pass


def _print_summary(prefix: str = "[at exit]") -> None:
    if not _PROFILE_ENABLED:
        return
    text = _format_summary()
    sys.stderr.write(f"\n{prefix}\n{text}\n")
    sys.stderr.flush()
    # Also append to a sidecar file so a hard kill -9 still leaves us
    # readable artifacts.
    _append_summary_file(f"\n{prefix}\n{text}\n")


atexit.register(_print_summary)


def install() -> None:
    """Apply all monkey-patches. Called from ``patches/__init__.py``."""
    if not _PROFILE_ENABLED:
        return

    sys.stderr.write(
        "[sunrise profile] VLLM_FL_SUNRISE_PROFILE_DECODE=1: "
        f"capturing first {MAX_DECODE_STEPS} decode-only steps; "
        f"summary printed at process exit and appended to {_SUMMARY_FILE}.\n"
    )
    _append_summary_file(
        "\n[startup]\n"
        "Sunrise decode-step profile installed. If this file never grows "
        "past this marker, the server did not execute any decode-only steps "
        "or the execute_model wrapper did not install.\n"
    )

    _patch_attention_impl()
    _patch_gdn()
    _patch_kv_write()
    _patch_misc()
    _patch_oot_layers()
    _patch_collectives()
    _patch_moe_runner()
    _patch_moe_details()
    _patch_native_int8()
    _patch_model_forward()


def _patch_native_int8() -> None:
    """Instrument compressed-tensors INT8 Linear hot path (quant + GEMM + MoE).

    Must run after ``enable_native_int8()`` so wrappers sit on the FlagGems
    rebinds. Safe no-op when those symbols are absent (BF16 checkpoints).
    """
    # 1) Activation quant — vLLM custom op (patched to FlagGems Triton).
    try:
        import vllm._custom_ops as _vllm_ops

        orig = getattr(_vllm_ops, "scaled_int8_quant", None)
        if orig is not None and not getattr(
            orig, "__sunrise_profile_wrapped__", False
        ):

            def _wrap_quant(fn):
                def _w(input, *a, **kw):
                    if not _in_decode_step:
                        return fn(input, *a, **kw)
                    device = input.device if isinstance(input, torch.Tensor) else None
                    _device_sync(device)
                    t0 = _now_us()
                    out = fn(input, *a, **kw)
                    _device_sync(device)
                    t1 = _now_us()
                    try:
                        tag = (
                            f"int8.scaled_int8_quant"
                            f"[M={input.shape[0]},K={input.shape[-1]}]"
                        )
                    except Exception:
                        tag = "int8.scaled_int8_quant[?]"
                    _call_records[tag].append(t1 - t0)
                    return out

                _w.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
                return _w

            _vllm_ops.scaled_int8_quant = _wrap_quant(orig)
            sys.stderr.write(
                "[sunrise profile] hooked vllm._custom_ops.scaled_int8_quant\n"
            )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(
            f"[sunrise profile] could not hook scaled_int8_quant: {e}\n"
        )

    # 2) Linear INT8 GEMM — wrap whatever currently sits on triton_scaled_mm
    #    (FlagGems scaled_mm after patch_int8_native, else stock Triton).
    try:
        import sys as _sys

        import vllm.model_executor.kernels.linear.scaled_mm.triton as _mm_mod

        orig_mm = getattr(_mm_mod, "triton_scaled_mm", None)
        if orig_mm is not None and not getattr(
            orig_mm, "__sunrise_profile_wrapped__", False
        ):

            def _wrap_mm(fn):
                def _w(input, weight, *a, **kw):
                    if not _in_decode_step:
                        return fn(input, weight, *a, **kw)
                    device = input.device if isinstance(input, torch.Tensor) else None
                    _device_sync(device)
                    t0 = _now_us()
                    out = fn(input, weight, *a, **kw)
                    _device_sync(device)
                    t1 = _now_us()
                    try:
                        m, k = int(input.shape[0]), int(input.shape[1])
                        # weight may be [N,K] or [K,N]
                        if weight.shape[1] == k:
                            n = int(weight.shape[0])
                        else:
                            n = int(weight.shape[1])
                        tag = f"int8.triton_scaled_mm[M={m},K={k},N={n}]"
                    except Exception:
                        tag = "int8.triton_scaled_mm[?]"
                    _call_records[tag].append(t1 - t0)
                    return out

                _w.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
                # Preserve FlagGems markers so rebinding logic still recognizes
                # the patched callable.
                for attr in ("_fl_native_int8_mm", "_fl_original"):
                    if hasattr(fn, attr):
                        setattr(_w, attr, getattr(fn, attr))
                return _w

            wrapped = _wrap_mm(orig_mm)
            _mm_mod.triton_scaled_mm = wrapped
            for mod in list(_sys.modules.values()):
                if mod is None:
                    continue
                if getattr(mod, "triton_scaled_mm", None) is orig_mm:
                    mod.triton_scaled_mm = wrapped
            sys.stderr.write(
                "[sunrise profile] hooked triton_scaled_mm (Linear INT8 GEMM)\n"
            )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(
            f"[sunrise profile] could not hook triton_scaled_mm: {e}\n"
        )

    # 3) MoE experts apply — exclusive envelope for TritonExpertsFL (INT8/BF16).
    try:
        from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL

        orig_apply = TritonExpertsFL.apply
        if not getattr(orig_apply, "__sunrise_profile_wrapped__", False):

            def _wrapped_apply(self, output, hidden_states, w1=None, *a, **kw):
                if not _in_decode_step:
                    return orig_apply(self, output, hidden_states, w1, *a, **kw)
                device = (
                    hidden_states.device
                    if isinstance(hidden_states, torch.Tensor)
                    else None
                )
                _device_sync(device)
                t0 = _now_us()
                out = orig_apply(self, output, hidden_states, w1, *a, **kw)
                _device_sync(device)
                t1 = _now_us()
                try:
                    m = int(hidden_states.shape[0])
                    k = int(hidden_states.shape[-1])
                    q = getattr(self, "quant_config", None)
                    if q is None:
                        kind = "no_qcfg"
                    elif getattr(q, "use_int8_w8a8", False):
                        kind = "w8a8"
                    elif getattr(q, "use_int8_w8a16", False):
                        kind = "w8a16"
                    else:
                        kind = "bf16"
                    wdt = getattr(w1, "dtype", None)
                    wdt_s = str(wdt).replace("torch.", "") if wdt is not None else "?"
                    tag = (
                        f"int8.TritonExpertsFL.apply"
                        f"[{kind},w={wdt_s},M={m},K={k}]"
                    )
                except Exception:
                    tag = "int8.TritonExpertsFL.apply[?]"
                _call_records[tag].append(t1 - t0)
                return out

            _wrapped_apply.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
            TritonExpertsFL.apply = _wrapped_apply
            sys.stderr.write(
                "[sunrise profile] hooked TritonExpertsFL.apply\n"
            )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(
            f"[sunrise profile] could not hook TritonExpertsFL.apply: {e}\n"
        )


def _patch_oot_layers() -> None:
    """Time RMSNormFL/RotaryEmbeddingFL forward directly.

    These OOT layers route through ``call_op`` -> ``OpManager.call`` in
    eager mode, but the vLLM-side custom-op path can shortcut to a
    cached ``_forward_method`` that bypasses our dispatcher hook on
    some configurations (notably when CustomOp.dispatch_forward picks
    forward_oot at construction time but then later replaces it via
    inline torch.compile boundaries). To make sure we capture them,
    hook the OOT classes' ``forward_oot`` directly.
    """
    try:
        from vllm_fl.ops.layernorm import RMSNormFL
        from vllm_fl.ops.activation import SiluAndMulFL
        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL
    except Exception:
        return

    def _wrap(cls, attr, tag):
        orig = getattr(cls, attr, None)
        if orig is None or getattr(orig, "__sunrise_profile_wrapped__", False):
            return

        def _w(self, *a, **kw):
            if not _in_decode_step:
                return orig(self, *a, **kw)
            device = None
            for v in a:
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
            _device_sync(device)
            t0 = _now_us()
            out = orig(self, *a, **kw)
            _device_sync(device)
            t1 = _now_us()
            _call_records[tag].append(t1 - t0)
            return out

        _w.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
        setattr(cls, attr, _w)

    _wrap(RMSNormFL, "forward_oot", "RMSNormFL.forward_oot")
    _wrap(SiluAndMulFL, "forward_oot", "SiluAndMulFL.forward_oot")
    _wrap(RotaryEmbeddingFL, "forward_oot", "RotaryEmbeddingFL.forward_oot")
    sys.stderr.write(
        "[sunrise profile] hooked RMSNormFL/SiluAndMulFL/RotaryEmbeddingFL.forward_oot\n"
    )




def _patch_attention_impl() -> None:
    from ..impl import attention as _attn_mod

    cls = _attn_mod.AttentionFLImpl
    orig_forward = cls.forward

    def wrapped_forward(self, query, key, value, kv_cache, *args, **kwargs):
        if not _in_decode_step:
            return orig_forward(self, query, key, value, kv_cache, *args, **kwargs)
        device = query.device if isinstance(query, torch.Tensor) else None
        _device_sync(device)
        t0 = _now_us()
        out = orig_forward(self, query, key, value, kv_cache, *args, **kwargs)
        _device_sync(device)
        t1 = _now_us()
        _call_records["full_attention.layer (8 layers)"].append(t1 - t0)
        return out

    cls.forward = wrapped_forward

    # Also wrap flag_gems.flash_attn_varlen_func at the call-site import.
    if hasattr(_attn_mod, "flash_attn_varlen_func"):
        orig = _attn_mod.flash_attn_varlen_func

        def _wrapped(*args, **kwargs):
            if not _in_decode_step:
                return orig(*args, **kwargs)
            q = args[0] if args else kwargs.get("q")
            device = q.device if isinstance(q, torch.Tensor) else None
            _device_sync(device)
            t0 = _now_us()
            out = orig(*args, **kwargs)
            _device_sync(device)
            t1 = _now_us()
            _call_records[
                "  └─ flash_attn_varlen_func (FlagGems)"
            ].append(t1 - t0)
            return out

        _attn_mod.flash_attn_varlen_func = _wrapped


def _patch_kv_write() -> None:
    from ..impl import attention as _attn_mod

    orig = getattr(_attn_mod, "reshape_and_cache_flash", None)
    if orig is None:
        return

    def _wrapped(key, value, *args, **kwargs):
        if not _in_decode_step:
            return orig(key, value, *args, **kwargs)
        device = key.device if isinstance(key, torch.Tensor) else None
        _device_sync(device)
        t0 = _now_us()
        out = orig(key, value, *args, **kwargs)
        _device_sync(device)
        t1 = _now_us()
        _call_records[
            "  └─ kv_write (FlagGems reshape_and_cache_flash)"
        ].append(t1 - t0)
        return out

    _attn_mod.reshape_and_cache_flash = _wrapped


def _patch_gdn() -> None:
    """Hook the GDN body operators at every module that uses them.

    Symbols like ``fused_recurrent_gated_delta_rule_packed_decode`` and
    ``fused_sigmoid_gating_delta_rule_update`` are imported with ``from
    ... import ...`` at module-load time inside
    ``vllm.model_executor.layers.mamba.gdn_linear_attn``. So patching only
    the source module misses the actual callsite. We patch at every
    namespace where these names live.
    """

    def _make_wrap(orig, tag):
        if getattr(orig, "__sunrise_profile_wrapped__", False):
            return orig

        def _wrapped(*args, **kwargs):
            if not _in_decode_step:
                return orig(*args, **kwargs)
            device = None
            for a in args:
                if isinstance(a, torch.Tensor):
                    device = a.device
                    break
            if device is None:
                for v in kwargs.values():
                    if isinstance(v, torch.Tensor):
                        device = v.device
                        break
            _device_sync(device)
            t0 = _now_us()
            out = orig(*args, **kwargs)
            _device_sync(device)
            t1 = _now_us()
            _call_records[tag].append(t1 - t0)
            return out

        _wrapped.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
        return _wrapped

    # Patch sunrise's PTPU rebind in fused_sigmoid_gating module (the one
    # actually called for the spec/mixed path).
    try:
        from ..impl.fla import fused_sigmoid_gating as _fsg_mod

        orig = _fsg_mod.fused_sigmoid_gating_delta_rule_update
        _fsg_mod.fused_sigmoid_gating_delta_rule_update = _make_wrap(
            orig, "GDN spec/mixed: fused_sigmoid_gating (PTPU)"
        )
    except Exception:
        pass

    # Patch the GDN call-site in vllm so that ALL paths (spec/mixed and
    # steady packed_decode) end up timed via the rebinding done at the
    # gdn_linear_attn import. ``patch_fla_ops.apply_patch`` runs BEFORE us
    # (see ``patches/__init__.py``), so the bound callable is already
    # the PTPU wrapper.
    packed_decode_tag = "GDN steady decode: packed_decode (PTPU)"

    try:
        from vllm.model_executor.layers.mamba import (
            gdn_linear_attn as _gdn_mod,
        )

        for sym, tag in (
            (
                "fused_sigmoid_gating_delta_rule_update",
                "GDN spec/mixed: fused_sigmoid_gating (PTPU)",
            ),
            (
                "fused_recurrent_gated_delta_rule_packed_decode",
                packed_decode_tag,
            ),
            (
                "fla_chunk_gated_delta_rule",
                "GDN prefill: chunk_gated_delta_rule (FLA Triton)",
            ),
        ):
            f = getattr(_gdn_mod, sym, None)
            if f is None:
                continue
            setattr(_gdn_mod, sym, _make_wrap(f, tag))
    except Exception:
        pass

    # Same for the FLA package re-exports (in case some consumer imports
    # them from there). The module attribute has already been rebound to
    # the PTPU wrapper by ``patch_fla_ops.apply_patch``, so ``_make_wrap``
    # wraps the PTPU path.
    try:
        import fla.ops.fused_recurrent as _fla_fr_mod

        f = _fla_fr_mod.fused_recurrent_gated_delta_rule_packed_decode
        _fla_fr_mod.fused_recurrent_gated_delta_rule_packed_decode = (
            _make_wrap(f, packed_decode_tag)
        )
    except Exception:
        pass


def _patch_misc() -> None:
    """Hook the dispatch manager + ``F.linear`` to time all gemms.

    Wrapping ``OpManager.call`` catches every flagos / vendor op call
    (rms_norm, silu_and_mul, rotary_embedding, etc.). However, gemms in
    Linear layers go through ``torch.nn.functional.linear`` directly,
    bypassing OpManager.call. We hook both call sites so that gemms are
    broken out of the residual.

    Hook also breaks down by gemm output-feature size, which is useful
    for telling the QKV / o_proj gemms apart from gate_up / down_proj.
    """
    try:
        from vllm_fl.dispatch.manager import OpManager
    except Exception:
        return

    if getattr(OpManager.call, "__sunrise_profile_wrapped__", False):
        return

    orig_call = OpManager.call

    def _wrapped_call(self, op_name, *args, **kwargs):
        if not _in_decode_step:
            return orig_call(self, op_name, *args, **kwargs)
        device = None
        for a in args:
            if isinstance(a, torch.Tensor):
                device = a.device
                break
        if device is None:
            for v in kwargs.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
        _device_sync(device)
        t0 = _now_us()
        out = orig_call(self, op_name, *args, **kwargs)
        _device_sync(device)
        t1 = _now_us()
        _call_records[f"op.{op_name}"].append(t1 - t0)
        return out

    _wrapped_call.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
    OpManager.call = _wrapped_call
    sys.stderr.write("[sunrise profile] OpManager.call wrapped\n")

    # Hook causal_conv1d_update / causal_conv1d_fn at the gdn import
    # site so we can break out the GDN front-end conv that is NOT
    # routed through OpManager.call (it is directly imported as a
    # symbol).
    try:
        from vllm.model_executor.layers.mamba import (
            gdn_linear_attn as _gdn_src_mod,
        )
    except Exception:
        return

    for fn_name in ("causal_conv1d_update", "causal_conv1d_fn"):
        orig = getattr(_gdn_src_mod, fn_name, None)
        if orig is None:
            continue
        if getattr(orig, "__sunrise_profile_wrapped__", False):
            continue

        def _make(fn, full_name):
            def _w(*a, **kw):
                if not _in_decode_step:
                    return fn(*a, **kw)
                device = None
                for v in (*a, *kw.values()):
                    if isinstance(v, torch.Tensor):
                        device = v.device
                        break
                _device_sync(device)
                t0 = _now_us()
                out = fn(*a, **kw)
                _device_sync(device)
                t1 = _now_us()
                _call_records[full_name].append(t1 - t0)
                return out

            _w.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
            return _w

        setattr(_gdn_src_mod, fn_name, _make(orig, f"GDN.{fn_name}"))

    # Hook torch.nn.functional.linear so we time every gemm done by a
    # Linear layer (q/k/v/o_proj, gate_up_proj, down_proj, lm_head, ...).
    # We bucket calls by the output feature size so that the report
    # tells QKV/MLP gemms apart from each other.
    import torch.nn.functional as _F

    if not getattr(_F.linear, "__sunrise_profile_wrapped__", False):
        _orig_linear = _F.linear

        def _wrapped_linear(input, weight, bias=None):
            if not _in_decode_step:
                return _orig_linear(input, weight, bias)
            device = (
                input.device if isinstance(input, torch.Tensor) else None
            )
            _device_sync(device)
            t0 = _now_us()
            out = _orig_linear(input, weight, bias)
            _device_sync(device)
            t1 = _now_us()
            try:
                in_feat = weight.shape[-1]
                out_feat = weight.shape[-2]
                tag = f"F.linear[in={in_feat},out={out_feat}]"
            except Exception:
                tag = "F.linear[?]"
            _call_records[tag].append(t1 - t0)
            return out

        _wrapped_linear.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
        _F.linear = _wrapped_linear
        # Also patch torch.nn.functional.linear at the module level
        # because the symbol may already be cached at the call site.
        torch.nn.functional.linear = _wrapped_linear


def _patch_collectives() -> None:
    """Instrument TP/PP/world collectives for decode profiling."""
    try:
        from vllm.distributed.parallel_state import GroupCoordinator
    except Exception as e:
        sys.stderr.write(
            f"[sunrise profile] could not import GroupCoordinator: {e}\n"
        )
        return

    def _wrap_collective(attr: str, default_tag: str):
        orig = getattr(GroupCoordinator, attr, None)
        if orig is None or getattr(orig, "__sunrise_profile_wrapped__", False):
            return

        def _w(self, *a, **kw):
            if not _in_decode_step:
                return orig(self, *a, **kw)
            # world_size == 1 short-circuit is essentially free; don't
            # bother timing those (they don't actually hit the device).
            if getattr(self, "world_size", 1) == 1:
                return orig(self, *a, **kw)
            # Find the first tensor arg to pick a device for sync.
            device = None
            for v in a:
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
            if device is None:
                for v in kw.values():
                    if isinstance(v, torch.Tensor):
                        device = v.device
                        break
            # Group tag — TP / PP / DP / EP / world. ``unique_name``
            # looks like "tp:1" / "pp:0" / "world:0"; we strip the
            # numeric suffix to get a clean "tp" / "pp" / ... tag.
            try:
                uname = getattr(self, "unique_name", default_tag)
                tag = uname.split(":", 1)[0] if uname else default_tag
            except Exception:
                tag = default_tag
            _device_sync(device)
            t0 = _now_us()
            out = orig(self, *a, **kw)
            _device_sync(device)
            t1 = _now_us()
            _call_records[f"op.{attr}.{tag}"].append(t1 - t0)
            return out

        _w.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
        setattr(GroupCoordinator, attr, _w)

    _wrap_collective("all_reduce", default_tag="?")
    _wrap_collective("all_gather", default_tag="?")
    _wrap_collective("reduce_scatter", default_tag="?")
    sys.stderr.write(
        "[sunrise profile] hooked GroupCoordinator.{all_reduce,all_gather,reduce_scatter}\n"
    )


def _patch_moe_runner() -> None:
    """Instrument MoE forward paths for decode profiling."""
    try:
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
            MoERunner,
        )
        from vllm.model_executor.layers.fused_moe import FusedMoE
    except Exception as e:
        sys.stderr.write(
            f"[sunrise profile] could not import MoE classes: {e}\n"
        )
        return

    def _wrap_method(cls, attr: str, tag: str, infer_device_arg_idx: int = 1):
        orig = getattr(cls, attr, None)
        if orig is None or getattr(orig, "__sunrise_profile_wrapped__", False):
            return

        def _w(self, *a, **kw):
            if not _in_decode_step:
                return orig(self, *a, **kw)
            device = None
            if len(a) > infer_device_arg_idx - 1 and isinstance(
                a[infer_device_arg_idx - 1], torch.Tensor
            ):
                device = a[infer_device_arg_idx - 1].device
            if device is None:
                for v in (*a, *kw.values()):
                    if isinstance(v, torch.Tensor):
                        device = v.device
                        break
            _device_sync(device)
            t0 = _now_us()
            out = orig(self, *a, **kw)
            _device_sync(device)
            t1 = _now_us()
            _call_records[tag].append(t1 - t0)
            return out

        _w.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
        setattr(cls, attr, _w)

    _wrap_method(MoERunner, "forward", "MoERunner.forward")
    _wrap_method(FusedMoE, "forward", "FusedMoE.forward (incl. runner)")
    sys.stderr.write(
        "[sunrise profile] hooked MoERunner.forward + FusedMoE.forward\n"
    )


def _patch_moe_details() -> None:
    """Instrument MoE routing and shared experts.

    These are nested below ``MoERunner.forward``. They make the eager profile
    actionable when the top-level MoE envelope dominates: we can separate
    router/top-k, shared experts, and the INT8 native stack before deciding
    which cudagraph-visible device kernels are worth optimizing.
    """
    try:
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            BaseRouter,
        )
    except Exception as e:
        sys.stderr.write(
            f"[sunrise profile] could not import MoE BaseRouter: {e}\n"
        )
        BaseRouter = None

    if BaseRouter is not None:
        orig_select = getattr(BaseRouter, "select_experts", None)
        if orig_select is not None and not getattr(
            orig_select, "__sunrise_profile_wrapped__", False
        ):

            def _wrapped_select_experts(
                self,
                hidden_states,
                router_logits,
                *,
                input_ids=None,
            ):
                if not _in_decode_step:
                    return orig_select(
                        self,
                        hidden_states,
                        router_logits,
                        input_ids=input_ids,
                    )
                device = (
                    hidden_states.device
                    if isinstance(hidden_states, torch.Tensor)
                    else None
                )
                _device_sync(device)
                t0 = _now_us()
                out = orig_select(
                    self,
                    hidden_states,
                    router_logits,
                    input_ids=input_ids,
                )
                _device_sync(device)
                t1 = _now_us()
                _call_records["MoE.router.select_experts"].append(t1 - t0)
                return out

            _wrapped_select_experts.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
            BaseRouter.select_experts = _wrapped_select_experts

    try:
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExperts,
        )
    except Exception as e:
        sys.stderr.write(
            f"[sunrise profile] could not import MoE SharedExperts: {e}\n"
        )
        SharedExperts = None

    if SharedExperts is not None:
        orig_apply = getattr(SharedExperts, "apply", None)
        if orig_apply is not None and not getattr(
            orig_apply, "__sunrise_profile_wrapped__", False
        ):

            def _wrapped_shared_apply(self, shared_experts_input, order):
                if not _in_decode_step:
                    return orig_apply(self, shared_experts_input, order)
                device = (
                    shared_experts_input.device
                    if isinstance(shared_experts_input, torch.Tensor)
                    else None
                )
                _device_sync(device)
                t0 = _now_us()
                out = orig_apply(self, shared_experts_input, order)
                _device_sync(device)
                t1 = _now_us()
                try:
                    order_name = getattr(order, "name", str(order))
                except Exception:
                    order_name = "?"
                _call_records[f"MoE.shared_experts.apply[{order_name}]"].append(
                    t1 - t0
                )
                return out

            _wrapped_shared_apply.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
            SharedExperts.apply = _wrapped_shared_apply

    def _wrap_mlp_forward(cls, tag: str) -> None:
        orig = getattr(cls, "forward", None)
        if orig is None or getattr(orig, "__sunrise_profile_wrapped__", False):
            return

        def _wrapped_forward(self, x, *args, **kwargs):
            if not _in_decode_step:
                return orig(self, x, *args, **kwargs)
            device = x.device if isinstance(x, torch.Tensor) else None
            _device_sync(device)
            t0 = _now_us()
            out = orig(self, x, *args, **kwargs)
            _device_sync(device)
            t1 = _now_us()
            _call_records[tag].append(t1 - t0)
            return out

        _wrapped_forward.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
        cls.forward = _wrapped_forward

    try:
        from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

        _wrap_mlp_forward(
            Qwen2MoeMLP, "MoE.shared_expert.Qwen2MoeMLP.forward"
        )
    except Exception as e:
        sys.stderr.write(
            f"[sunrise profile] could not hook Qwen2MoeMLP.forward: {e}\n"
        )

    try:
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP

        _wrap_mlp_forward(
            Qwen3MoeMLP, "MoE.shared_expert.Qwen3MoeMLP.forward"
        )
    except Exception as e:
        sys.stderr.write(
            f"[sunrise profile] could not hook Qwen3MoeMLP.forward: {e}\n"
        )

    try:
        from vllm.model_executor.layers.activation import SiluAndMul
    except Exception as e:
        sys.stderr.write(
            f"[sunrise profile] could not import SiluAndMul: {e}\n"
        )
        SiluAndMul = None

    if SiluAndMul is not None:
        orig_forward = getattr(SiluAndMul, "forward", None)
        if orig_forward is not None and not getattr(
            orig_forward, "__sunrise_profile_wrapped__", False
        ):

            def _wrapped_silu_and_mul(self, x, *args, **kwargs):
                if not _in_decode_step:
                    return orig_forward(self, x, *args, **kwargs)
                device = x.device if isinstance(x, torch.Tensor) else None
                _device_sync(device)
                t0 = _now_us()
                out = orig_forward(self, x, *args, **kwargs)
                _device_sync(device)
                t1 = _now_us()
                try:
                    in_feat = x.shape[-1]
                    out_feat = in_feat // 2
                    tag = f"SiluAndMul.forward[in={in_feat},out={out_feat}]"
                except Exception:
                    tag = "SiluAndMul.forward[?]"
                _call_records[tag].append(t1 - t0)
                return out

            _wrapped_silu_and_mul.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
            SiluAndMul.forward = _wrapped_silu_and_mul

    sys.stderr.write(
        "[sunrise profile] hooked MoE router/shared-experts/shared-MLP\n"
    )


_runner_patch_state = {"installed": False, "execute_model_calls": 0}


def _make_execute_model_wrapper(orig, cls_name):
    state = _runner_patch_state

    def _wrapped(self, scheduler_output, intermediate_tensors=None, *a, **kw):
        state["execute_model_calls"] += 1
        was_decode_only = False
        try:
            counts = getattr(scheduler_output, "num_scheduled_tokens", None)
            if counts is not None and len(counts) > 0:
                # Pure single-token decode: every scheduled request asks
                # for exactly one token. Anything else (prefill chunks)
                # is excluded so the per-op aggregation stays apples-to-
                # apples for steady-state decode timing.
                was_decode_only = all(int(c) == 1 for c in counts.values())
        except Exception:
            pass

        if state["execute_model_calls"] <= 3:
            try:
                counts = getattr(scheduler_output, "num_scheduled_tokens", None)
                sys.stderr.write(
                    f"[sunrise profile] {cls_name}.execute_model call "
                    f"#{state['execute_model_calls']}: was_decode_only="
                    f"{was_decode_only} num_scheduled_tokens="
                    f"{dict(counts) if counts else None}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass

        device = getattr(self, "device", None)
        begin_decode_step(device)
        try:
            return orig(self, scheduler_output, intermediate_tensors, *a, **kw)
        finally:
            end_decode_step(device, was_decode_only)

    return _wrapped


def _try_install_execute_model_patch() -> bool:
    """Install ``execute_model`` patches if the runner classes are already loaded.

    Returns True if the patch was installed (or had been installed previously).
    Importing ``vllm_fl.worker.model_runner`` early (e.g. at sunrise/__init__
    time) snapshots ``current_platform.dist_backend`` BEFORE vLLM finishes
    initializing the platform, which then breaks ``graph_capture`` on PTPU. So
    we instead poll ``sys.modules`` and patch lazily once the runner module
    has been loaded by vLLM's normal startup path.
    """
    state = _runner_patch_state
    if state["installed"]:
        return True

    candidates = []
    fl_mod = sys.modules.get("vllm_fl.worker.model_runner")
    if fl_mod is not None and hasattr(fl_mod, "ModelRunnerFL"):
        candidates.append(("ModelRunnerFL", fl_mod.ModelRunnerFL))
    core_mod = sys.modules.get("vllm.v1.worker.gpu_model_runner")
    if core_mod is not None and hasattr(core_mod, "GPUModelRunner"):
        candidates.append(("GPUModelRunner", core_mod.GPUModelRunner))

    if not candidates:
        return False

    for cls_name, cls in candidates:
        if getattr(cls.execute_model, "__sunrise_profile_wrapped__", False):
            continue
        orig = cls.execute_model
        wrapped = _make_execute_model_wrapper(orig, cls_name)
        wrapped.__sunrise_profile_wrapped__ = True  # type: ignore[attr-defined]
        cls.execute_model = wrapped
        sys.stderr.write(
            f"[sunrise profile] {cls_name}.execute_model wrapper installed (lazy)\n"
        )
    state["installed"] = True
    return True


def _patch_model_forward() -> None:
    """Schedule the ``execute_model`` patch.

    We do NOT eagerly import the runner module here. Instead, we install a
    short polling thread that retries every few hundred milliseconds until
    the runner class shows up in ``sys.modules`` (it always does, just
    after vLLM finishes initializing the platform). This avoids forcing
    ``vllm_fl.worker.model_runner`` to be imported before
    ``current_platform`` is ready, which previously broke ``graph_capture``
    by capturing the wrong branch (non-PTPU) at module-load time.
    """
    import threading

    if _try_install_execute_model_patch():
        return

    def _poll():
        for _ in range(600):  # up to ~60 s
            if _try_install_execute_model_patch():
                return
            time.sleep(0.1)
        sys.stderr.write(
            "[sunrise profile] timed out waiting for ModelRunner; total-step "
            "timing disabled.\n"
        )

    t = threading.Thread(target=_poll, name="sunrise-profile-patcher", daemon=True)
    t.start()
    sys.stderr.write(
        "[sunrise profile] execute_model patch scheduled (will install once "
        "the runner module loads)\n"
    )
