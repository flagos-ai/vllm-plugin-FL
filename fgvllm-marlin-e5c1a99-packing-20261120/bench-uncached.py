"""Read-only validation harness. Torch references are never production imports."""
import gc
import importlib
import json
import statistics
import time

import torch

m = importlib.import_module("flaggems_vllm.ops.fused_marlin_moe")


def old_weight(w, block_size_k):
    e, n, k_half = w.shape
    k, b = 2 * k_half, block_size_k // 8
    shifts = torch.tensor((0, 16, 4, 20, 8, 24, 12, 28), dtype=torch.int32, device=w.device)
    out = torch.empty(e, k // 8, n, dtype=torch.int32, device=w.device)
    for i in range(e):
        low = (w[i] & 0xF).to(torch.uint8)
        high = ((w[i] >> 4) & 0xF).to(torch.uint8)
        unpacked = torch.stack((low, high), -1).reshape(n, k)
        tiled = unpacked.reshape(n, k // (8 * b), 8, b).transpose(-1, -2)
        packed = (tiled.to(torch.int32) << shifts).sum(-1, dtype=torch.int32)
        out[i].copy_(packed.reshape(n, k // 8).transpose(0, 1))
    return out


def old_scale(s, dtype, fold):
    s = s.view(torch.uint8) if s.dtype != torch.uint8 else s
    exp = s.to(torch.int32) - 127
    if fold:
        exp = exp + (126 if dtype == torch.bfloat16 else 14)
    out = torch.exp2(exp.to(torch.float32)).to(dtype).transpose(-2, -1).contiguous()
    if fold:
        bits = out.view(torch.int16).to(torch.int32) & 0xFFFF
        out = ((bits << 16) | bits).contiguous()
    return out


def timing(fn, iterations):
    # Every call allocates and computes a real result. No cached wrappers.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        output = fn()
        del output
    end.record()
    end.synchronize()
    return {"cuda_ms": start.elapsed_time(end) / iterations,
            "wall_ms": 1000 * (time.perf_counter() - wall_start) / iterations}


rows = []


def compare(name, shape, old, new, iterations):
    expected, actual = old(), new()
    # Bit-level comparison also proves the pre-timed output layout.
    if actual.dtype in (torch.float16, torch.bfloat16):
        assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))
    else:
        assert torch.equal(expected, actual)
    del expected, actual
    for fn in (old, new):
        for _ in range(2):
            output = fn()
            del output
    torch.cuda.synchronize()
    pairs = []
    for round_id in range(3):
        order = (("old", old), ("new", new)) if round_id != 1 else (("new", new), ("old", old))
        pair = {}
        for key, fn in order:
            pair[key] = timing(fn, iterations)
        pairs.append(pair)
    old_ms = statistics.median(p["old"]["cuda_ms"] for p in pairs)
    new_ms = statistics.median(p["new"]["cuda_ms"] for p in pairs)
    row = {"name": name, "shape": shape, "iterations_per_pair": iterations,
           "pairs": pairs, "old_median_cuda_ms": old_ms,
           "new_median_cuda_ms": new_ms, "speedup": old_ms / new_ms,
           "correctness": "bitwise PASS", "cache": "direct helper; uncached"}
    rows.append(row)
    print(json.dumps(row), flush=True)
    gc.collect()
    torch.cuda.empty_cache()


torch.manual_seed(42)
for label, shape, iterations in (("small", (2, 32, 64), 100),
                                  ("mixtral_gateup", (8, 28672, 2048), 3)):
    w = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    compare("weight_" + label, shape, lambda: old_weight(w, 128),
            lambda: m._pack_w_interleave(w, 128), iterations)
    del w
for label, shape, iterations in (("small", (2, 32, 4), 100),
                                  ("mixtral_gateup", (8, 28672, 128), 10)):
    s = torch.randint(125, 129, shape, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    for dtype in (torch.bfloat16, torch.float16):
        for fold in (False, True):
            helper = m._pack_scale_e8m0_fold if fold else m._pack_scale_e8m0
            compare(f"scale_{label}_{dtype}_{fold}", shape,
                    lambda: old_scale(s, dtype, fold),
                    lambda: helper(s, dtype), iterations)
    del s
receipt = {"rows": rows, "gpu": torch.cuda.get_device_name(0),
           "method": "3 alternating pairs, JIT warmed; direct uncached helper; real allocations",
           "max_memory_allocated_bytes": torch.cuda.max_memory_allocated()}
with open("/probe/uncached-timing.json", "w") as f:
    json.dump(receipt, f, indent=2)
