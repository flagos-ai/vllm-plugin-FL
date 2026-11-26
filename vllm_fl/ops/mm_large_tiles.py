"""Startup gate for large-M BF16 NT matrix-multiply configurations."""

import gc
import hashlib
import math
import statistics
from contextlib import nullcontext
from pathlib import Path

import torch
import triton

from vllm.logger import init_logger

from vllm_fl import device_profiles, envs
from vllm_fl.ops.mm_tuning import _find_tuner, _get_mm_module

logger = init_logger("vllm.vllm_fl.mm_large_tiles")
_SWITCH = "VLLM_FL_MM_LARGE_TILES"
_M = 2048
_SHAPES = (
    ("down", 2048, 6144),
    ("qkv", 2560, 2048),
    ("o", 2048, 2048),
)
_TAIL_SHAPE = (1031, 257, 257)
_SELF_CHECK_SEED = 72931
_BENCHMARK_SEED = 72932
_MAX_SHARED_BYTES = 65536


def source_hash():
    """Hash the selected backend and this startup gate for compile caches."""
    module = _get_mm_module()
    digest = hashlib.sha256()
    for path in (Path(module.__file__), Path(__file__)):
        digest.update(path.resolve().read_bytes())
    return digest.hexdigest()


def disable():
    """Restore the backend's original NT candidate set."""
    _get_mm_module().set_mm_large_tiles(False)


def _clear_tuner_startup_args(module):
    tuner = _find_tuner(module.mm_kernel_nt)
    tuner.nargs = None
    tuner._last_benchmark_args = None
    tuner._last_benchmark_meta = None


def _candidate_configs(module):
    configs = tuple(getattr(module, "_MM_LARGE_TILE_CONFIGS", ()))
    if len(configs) != 2:
        raise RuntimeError("selected mm backend does not expose two large-M configs")
    return configs


def _raw_nt_kernel(tuner):
    """Return the heuristics-wrapped JIT below LibTuner, bypassing autotune."""
    kernel = getattr(tuner, "fn", None)
    if kernel is None or not callable(getattr(kernel, "__getitem__", None)):
        raise TypeError("mm_kernel_nt tuner does not wrap a launchable JIT")
    return kernel


def _raw_launch(tuner, config, a, weight, output):
    m, k = a.shape
    n = weight.shape[0]
    block_m = config.kwargs["BLOCK_M"]
    block_n = config.kwargs["BLOCK_N"]
    block_k = config.kwargs["BLOCK_K"]
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    compiled = _raw_nt_kernel(tuner)[grid](
        a,
        weight.T,
        output,
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        EVEN_M=m % block_m == 0,
        EVEN_N=n % block_n == 0,
        EVEN_K=k % block_k == 0,
        pipeline=config.kwargs["pipeline"],
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    metadata = getattr(compiled, "metadata", None)
    shared_bytes = getattr(metadata, "shared", None)
    if shared_bytes is None:
        raise RuntimeError("raw mm compile did not expose shared-memory metadata")
    if shared_bytes > _MAX_SHARED_BYTES:
        raise RuntimeError(
            f"large-M tile uses {shared_bytes} shared bytes; limit is {_MAX_SHARED_BYTES}"
        )
    return shared_bytes


def _sample_rows(m, block_m):
    """Sample the ends and every output-tile boundary for the CPU oracle."""
    rows = set(range(min(16, m)))
    rows.update(range(max(0, m - 32), m))
    for boundary in range(block_m, m, block_m):
        rows.update(
            index for index in (boundary - 1, boundary, boundary + 1) if 0 <= index < m
        )
    return tuple(sorted(rows))


def _validate_raw_candidates(module, device):
    import flag_gems

    tuner = _find_tuner(module.mm_kernel_nt)
    configs = _candidate_configs(module)
    generator = torch.Generator(device="cpu").manual_seed(_SELF_CHECK_SEED)
    cases = [("tail", *_TAIL_SHAPE)] + [(name, _M, n, k) for name, n, k in _SHAPES]
    shared_by_case = {}
    dispatch = (
        nullcontext()
        if getattr(flag_gems, "current_work_registrar", None) is not None
        else flag_gems.use_gems()
    )

    with dispatch, torch.cuda.device(device):
        for name, m, n, k in cases:
            a_cpu = torch.randn((m, k), generator=generator, dtype=torch.bfloat16)
            weight_cpu = torch.randn((n, k), generator=generator, dtype=torch.bfloat16)
            a = a_cpu.to(device)
            weight = weight_cpu.to(device)
            for config in configs:
                rows = _sample_rows(m, config.kwargs["BLOCK_M"])
                row_indices = torch.tensor(rows, dtype=torch.long)
                reference = torch.mm(
                    a_cpu.index_select(0, row_indices).float(), weight_cpu.float().T
                )
                output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
                shared_bytes = _raw_launch(tuner, config, a, weight, output)
                torch.cuda.synchronize(device)
                if not torch.isfinite(output).all().item():
                    raise RuntimeError(
                        f"{name} raw candidate produced non-finite output"
                    )
                torch.testing.assert_close(
                    output.index_select(0, row_indices.to(device)).cpu().float(),
                    reference,
                    rtol=0.02,
                    atol=0.05,
                )
                shared_by_case[(name, _config_label(config))] = shared_bytes
                del output
            del a, weight, a_cpu, weight_cpu, reference
    logger.info(
        "Large-M raw-candidate CPU FP32 checks passed: shared_bytes=%s", shared_by_case
    )
    return shared_by_case


def _config_label(config):
    kwargs = config.kwargs
    return (
        f"{kwargs['BLOCK_M']}x{kwargs['BLOCK_N']}x{kwargs['BLOCK_K']}"
        f"/w{config.num_warps}/s{config.num_stages}/{kwargs['pipeline']}"
    )


def _make_buffers(device, n, k):
    generator = torch.Generator(device="cpu").manual_seed(_BENCHMARK_SEED)
    buffers = []
    input_bytes = 0
    for _ in range(3):
        a_cpu = torch.randn((_M, k), generator=generator, dtype=torch.bfloat16)
        weight_cpu = torch.randn((n, k), generator=generator, dtype=torch.bfloat16)
        a = a_cpu.to(device)
        weight = weight_cpu.to(device)
        buffers.append((a, weight))
        input_bytes += (
            a.numel() * a.element_size() + weight.numel() * weight.element_size()
        )
        del a_cpu, weight_cpu
    return buffers, input_bytes


def _prepare_arm(flag_gems, module, tuner, device, buffers, enabled):
    module.set_mm_large_tiles(enabled)
    a, weight = buffers[0]
    eager = flag_gems.mm(a, weight.T)
    torch.cuda.synchronize(device)
    if not torch.isfinite(eager).all().item():
        raise RuntimeError(
            f"mm output was non-finite with large tiles enabled={enabled}"
        )
    eager_outputs = [eager]
    for source, weight_source in buffers[1:]:
        eager_outputs.append(flag_gems.mm(source, weight_source.T))
    torch.cuda.synchronize(device)

    args = {
        "A": a,
        "B": weight.T,
        "C": eager,
        "M": a.shape[0],
        "N": weight.shape[0],
        "K": a.shape[1],
    }
    selected = tuner.cache.get(tuple(tuner.get_key(args)))
    if selected is None:
        raise RuntimeError("mm autotune completed without a selected configuration")

    outputs = []
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for input_id in (0, 1, 2) * 4:
            source, weight_source = buffers[input_id]
            outputs.append(flag_gems.mm(source, weight_source.T))
    return {
        "graph": graph,
        "outputs": outputs,
        "eager_outputs": eager_outputs,
        "tile": _config_label(selected),
    }


def _measure_once(graph):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise RuntimeError(f"invalid graph event time: {elapsed_ms}")
    return elapsed_ms


def _abba(graph_a, graph_b):
    samples = {"baseline": [], "candidate": []}
    for name, graph in (
        ("baseline", graph_a),
        ("candidate", graph_b),
        ("candidate", graph_b),
        ("baseline", graph_a),
    ):
        samples[name].append(_measure_once(graph))
    return samples


def _benchmark_shape(flag_gems, module, tuner, device, name, n, k, l2_bytes):
    buffers, input_bytes = _make_buffers(device, n, k)
    if input_bytes <= l2_bytes:
        raise RuntimeError(
            f"{name} rotating inputs ({input_bytes} bytes) do not exceed L2 ({l2_bytes})"
        )

    baseline = _prepare_arm(flag_gems, module, tuner, device, buffers, enabled=False)
    candidate = _prepare_arm(flag_gems, module, tuner, device, buffers, enabled=True)

    baseline["graph"].replay()
    candidate["graph"].replay()
    torch.cuda.synchronize(device)
    for arm_name, arm in (("baseline", baseline), ("candidate", candidate)):
        for output_id, output in enumerate(arm["outputs"]):
            if not torch.isfinite(output).all().item():
                raise RuntimeError(
                    f"{name} captured {arm_name} produced non-finite output"
                )
            torch.testing.assert_close(
                output.float(),
                arm["eager_outputs"][output_id % 3].float(),
                rtol=0.02,
                atol=0.05,
            )
    for baseline_output, candidate_output in zip(
        baseline["outputs"], candidate["outputs"]
    ):
        torch.testing.assert_close(
            candidate_output.float(), baseline_output.float(), rtol=0.02, atol=0.05
        )
    del baseline["eager_outputs"], candidate["eager_outputs"]

    _abba(baseline["graph"], candidate["graph"])
    formal = _abba(baseline["graph"], candidate["graph"])
    baseline_graph_ms = statistics.median(formal["baseline"])
    candidate_graph_ms = statistics.median(formal["candidate"])
    replay_calls = 12
    baseline_ms = baseline_graph_ms / replay_calls
    candidate_ms = candidate_graph_ms / replay_calls
    result = {
        "shape": name,
        "M": _M,
        "N": n,
        "K": k,
        "input_bytes": input_bytes,
        "rotating_buffers": len(buffers),
        "graph_calls": replay_calls,
        "baseline_tile": baseline["tile"],
        "candidate_tile": candidate["tile"],
        "baseline_event_ms": formal["baseline"],
        "candidate_event_ms": formal["candidate"],
        "baseline_event_per_call_ms": [
            value / replay_calls for value in formal["baseline"]
        ],
        "candidate_event_per_call_ms": [
            value / replay_calls for value in formal["candidate"]
        ],
        "baseline_median_ms": baseline_ms,
        "candidate_median_ms": candidate_ms,
        "speedup_percent": (baseline_ms - candidate_ms) * 100.0 / baseline_ms,
    }
    logger.info("Large-M graph ABBA: %s", result)
    del baseline, candidate, buffers
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _benchmark(module, device):
    import flag_gems

    properties = torch.cuda.get_device_properties(device)
    l2_bytes = int(
        getattr(properties, "L2_cache_size", 0)
        or getattr(properties, "l2_cache_size", 0)
    )
    if l2_bytes <= 0:
        raise RuntimeError("device L2 cache size is unavailable")
    tuner = _find_tuner(module.mm_kernel_nt)
    results = []
    dispatch = (
        nullcontext()
        if getattr(flag_gems, "current_work_registrar", None) is not None
        else flag_gems.use_gems()
    )
    with dispatch, torch.cuda.device(device):
        for name, n, k in _SHAPES:
            results.append(
                _benchmark_shape(flag_gems, module, tuner, device, name, n, k, l2_bytes)
            )
    passes = _passes_measurement_gate(results)
    logger.info("Large-M candidate gate: enabled=%s results=%s", passes, results)
    return passes, results


def _passes_measurement_gate(results):
    if len(results) != len(_SHAPES):
        return False
    if any(
        not math.isfinite(result["baseline_median_ms"])
        or not math.isfinite(result["candidate_median_ms"])
        or result["baseline_median_ms"] <= 0
        or result["candidate_median_ms"] <= 0
        for result in results
    ):
        return False
    return all(
        result["candidate_median_ms"] <= result["baseline_median_ms"] * 1.02
        for result in results
    ) and any(
        result["candidate_median_ms"] < result["baseline_median_ms"]
        for result in results
    )


def install(device) -> bool:
    """Enable the added candidates only after correctness and graph-time gates."""
    if device.type != "cuda" or torch.version.hip is not None:
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    profile = device_profiles.lookup(index) or {}
    requested = envs.switch(_SWITCH, auto=profile.get("mm_large_tiles") is True)
    if not requested and profile.get("mm_large_tiles") is not True:
        return False

    module = None
    try:
        module = _get_mm_module()
        module.set_mm_large_tiles(False)
        if not requested:
            return False
        _validate_raw_candidates(module, device)
        enabled, _ = _benchmark(module, device)
        module.set_mm_large_tiles(enabled)
        return enabled
    except Exception:
        if module is not None:
            try:
                module.set_mm_large_tiles(False)
            except Exception:
                logger.debug(
                    "Could not disable large-M mm configs after failure", exc_info=True
                )
        logger.warning(
            "Large-M mm startup gate failed; using original candidates", exc_info=True
        )
        return False
    finally:
        if module is not None:
            try:
                _clear_tuner_startup_args(module)
            except Exception:
                logger.debug(
                    "Could not release large-M mm autotune arguments", exc_info=True
                )
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.debug("Could not release large-M mm gate allocations", exc_info=True)
