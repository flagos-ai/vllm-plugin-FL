# Copyright (c) 2026 BAAI. All rights reserved.

"""Startup selection for reusing prefill cumulative KV lengths."""

import statistics
import threading
import time
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.vllm_fl.attention.prefill_cu_kv")

ENV_PREFILL_CU_KV_ONCE = "VLLM_FL_ATTN_PREFILL_CU_KV_ONCE"
MIN_AUTO_SPEEDUP = 1.10
_TIMING_REPEATS = 3


@dataclass(frozen=True)
class PrefillCuKvDecision:
    enabled: bool
    reason: str


_decisions: dict[tuple[str, int, int, bool | None], PrefillCuKvDecision] = {}
_decision_lock = threading.Lock()


def cumulative_kv_lengths_once(seq_lens: torch.Tensor) -> torch.Tensor:
    """Build an int32 cumulative-length vector without copying values to host."""
    result = torch.zeros(
        seq_lens.numel() + 1, dtype=torch.int32, device=seq_lens.device
    )
    torch.cumsum(seq_lens, dim=0, dtype=torch.int32, out=result[1:])
    return result


def _canonical_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _validate_numeric_path(device, max_num_seqs):
    for lengths in ((), (0, 15, 16, 17, 4096, 16384), (131072,) * max_num_seqs):
        actual = cumulative_kv_lengths_once(
            torch.tensor(lengths, dtype=torch.int32, device=device)
        )
        expected = [0]
        for length in lengths:
            expected.append(expected[-1] + length)
        torch.testing.assert_close(
            actual.cpu(), torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0
        )


def _can_fallback_runtime_error(exc: Exception) -> bool:
    if isinstance(exc, (MemoryError, NotImplementedError)):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "not implemented",
            "not supported",
            "unsupported",
            "could not run",
        )
    )


def try_cumulative_kv_lengths_once(
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor | None, Exception | None]:
    """Return None only for allocation or unsupported-operation failures."""
    try:
        return cumulative_kv_lengths_once(seq_lens), None
    except Exception as exc:
        if _can_fallback_runtime_error(exc):
            return None, exc
        raise


def _legacy_per_layer(seq_lens: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [0] + seq_lens.tolist(), dtype=torch.int32, device=seq_lens.device
    ).cumsum(dim=0, dtype=torch.int32)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, device: torch.device) -> float:
    _synchronize(device)
    start = time.perf_counter()
    result = fn()
    _synchronize(device)
    elapsed = time.perf_counter() - start
    del result
    return elapsed


def _benchmark_speedup(
    device: torch.device, layer_count: int, max_num_seqs: int
) -> tuple[float, tuple[tuple[int, float, float], ...]]:
    if layer_count < 1 or max_num_seqs < 1:
        raise ValueError("layer count and maximum sequence count must be positive")
    counts = sorted({1, min(8, max_num_seqs), max_num_seqs})
    measurements = []
    for count in counts:
        lengths = [4096 + 16 * i for i in range(count)]
        seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)

        def old_step(lengths=seq_lens):
            result = None
            for _ in range(layer_count):
                result = _legacy_per_layer(lengths)
            return result

        def new_step(lengths=seq_lens):
            return cumulative_kv_lengths_once(lengths)

        # Warm both implementations before the interleaved wall-time samples.
        old_step()
        new_step()
        _synchronize(device)
        old_samples = []
        new_samples = []
        for repeat in range(_TIMING_REPEATS):
            calls = ((old_step, old_samples), (new_step, new_samples))
            if repeat % 2:
                calls = tuple(reversed(calls))
            for fn, samples in calls:
                samples.append(_time_call(fn, device))
        old_median = statistics.median(old_samples)
        new_median = statistics.median(new_samples)
        measurements.append((count, old_median, new_median))

    # Sum the representative batches so small-count noise cannot dominate the gate.
    speedup = sum(old for _, old, _ in measurements) / max(
        sum(new for _, _, new in measurements), 1e-12
    )
    return speedup, tuple(measurements)


def resolve_prefill_cu_kv_once(device, layer_count, max_num_seqs, requested):
    if requested is False:
        return PrefillCuKvDecision(False, "explicitly disabled")
    device = _canonical_device(device)
    if device.type != "cuda":
        return PrefillCuKvDecision(False, "unsupported device type")
    key = (str(device), layer_count, max_num_seqs, requested)
    with _decision_lock:
        if key in _decisions:
            return _decisions[key]
        try:
            _validate_numeric_path(device, max_num_seqs)
            if requested is True:
                decision = PrefillCuKvDecision(
                    True, "explicitly enabled after safety check"
                )
            else:
                speedup, _ = _benchmark_speedup(device, layer_count, max_num_seqs)
                decision = PrefillCuKvDecision(
                    speedup >= MIN_AUTO_SPEEDUP, f"startup speedup {speedup:.2f}x"
                )
        except Exception as error:
            decision = PrefillCuKvDecision(False, f"startup check failed: {error}")
        _decisions[key] = decision
        logger.info(
            "%s enabled=%s: %s",
            ENV_PREFILL_CU_KV_ONCE,
            decision.enabled,
            decision.reason,
        )
        return decision


def disable_cached_decision(device, layer_count, max_num_seqs, requested, reason):
    key = (str(_canonical_device(device)), layer_count, max_num_seqs, requested)
    with _decision_lock:
        _decisions[key] = PrefillCuKvDecision(
            False, f"runtime fallback after {type(reason).__name__}"
        )
    logger.warning(
        "%s disabled after runtime fallback: %s", ENV_PREFILL_CU_KV_ONCE, reason
    )
