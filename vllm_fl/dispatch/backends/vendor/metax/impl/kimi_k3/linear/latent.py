"""Group D/U rows; preserve the original FlagGems 64-row GEMM arithmetic.

MetaX backend modules are loaded by FlagGems under its canonical _metax name.
Reuse each process's selected 64-row configuration; do not retune for larger M.
"""

import torch
import triton
import triton.language as tl
import flag_gems
from vllm_fl.dispatch.backends.flaggems.impl.kimi_k3_mm import (
    mm_kernel_splitk_partial,
    mm_kernel_splitk_reduce,
    _generic_two_step_split_k,
    splitk_mm_scenario,
    mm_kernel_nt,
    mm_kernel,
)


from vllm_fl.dispatch.backends.flaggems.impl.kimi_k3_mm import (
    PARTIAL_KERNEL as _PARTIAL,
    REDUCE_KERNEL as _REDUCE,
    NT_KERNEL as _NT,
    MM_KERNEL as _MM,
)

_ROW_CONFIGS = {}


def _row_config(x, wt, out):
    # Only CPU launch metadata is cached; no tensor or storage pointer is retained.
    n, k = out.shape[1], x.shape[1]
    key = (x.device.index, x.stride(0), n, k)
    if key not in _ROW_CONFIGS:
        # Warm the exact existing public operator and retrieve its selected config.
        # The production warmup performs this before Graph capture.
        flag_gems.mm_out(x[:64], wt, out=out[:64])
        if splitk_mm_scenario(64, n, k):
            split_k = _generic_two_step_split_k(64, n, k)
            config_key = (
                64,
                n,
                k,
                x.stride(0),
                1,
                "torch.bfloat16",
                "torch.bfloat16",
                "torch.float32",
            )
            config = mm_kernel_splitk_partial.fn.cache[config_key]
            meta = config.all_kwargs()
            assert meta["BLOCK_K"] == 64 and 64 % meta["BLOCK_M"] == 0
            kind = "splitk"
        else:
            # Preserve the public 64-row dense GEMM's K-reduction and tile.
            # U has more output columns and selects this path on TP8.
            kind = "nt" if x.is_contiguous() else "general"
            prefix = (64, n, k) if kind == "nt" else (64, n, k, x.stride(0), 1)
            config_key = prefix + ("torch.bfloat16",) * 3
            kernel = mm_kernel_nt if kind == "nt" else mm_kernel
            meta = kernel.fn.cache[config_key].all_kwargs()
            split_k = 1
        _ROW_CONFIGS[key] = {"kind": kind, "meta": dict(meta), "split_k": split_k}
    return _ROW_CONFIGS[key]


def old_project(x, weight, *, out_dtype=None):
    out = torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=out_dtype or x.dtype
    )
    for start in range(0, x.shape[0], 64):
        flag_gems.mm_out(x[start : start + 64], weight.t(), out=out[start : start + 64])
    return out


def project_grouped(x, weight, *, group_rows=4096, out_dtype=None):
    m = x.shape[0]
    if not (
        x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and (out_dtype is None or out_dtype == torch.bfloat16)
        and x.ndim == weight.ndim == 2
        and tuple(weight.shape) in ((448, 7168), (896, 3584))
        and x.shape[1] == weight.shape[1]
        and weight.is_contiguous()
        and x.stride(1) == 1
        and 65 <= m <= 4096
    ):
        return old_project(x, weight, out_dtype=out_dtype)
    if m < 128:
        return old_project(x, weight, out_dtype=out_dtype)
    assert group_rows in (64, 128, 256, 512, 1024, 4096)
    n, k = weight.shape
    group_rows = min(group_rows, 256)  # Qualified eager TP8 path; D partials <=7MiB.
    full_rows = m // 64 * 64
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    wt = weight.t()
    cfg = _row_config(x, wt, out)
    kind, meta, split_k = cfg["kind"], cfg["meta"], cfg["split_k"]
    partials = (
        torch.empty(
            (split_k, min(full_rows, group_rows), n),
            device=x.device,
            dtype=torch.float32,
        )
        if kind == "splitk"
        else None
    )
    for start in range(0, full_rows, group_rows):
        count = min(group_rows, full_rows - start)
        a, c = x[start : start + count], out[start : start + count]
        grid = (triton.cdiv(count, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
        if kind == "splitk":
            _PARTIAL[(grid[0], split_k)](
                a,
                wt,
                partials,
                count,
                n,
                k,
                a.stride(0),
                1,
                1,
                k,
                SPLIT_K=split_k,
                **meta,
            )
            _REDUCE[(triton.cdiv(count * n, 256),)](
                partials,
                c,
                count,
                n,
                c.stride(0),
                1,
                SPLIT_K=split_k,
                BLOCK_SIZE=256,
                num_warps=4,
            )
        else:
            flags = {
                "EVEN_M": count % meta["BLOCK_M"] == 0,
                "EVEN_N": n % meta["BLOCK_N"] == 0,
                "EVEN_K": k % meta["BLOCK_K"] == 0,
            }
            if kind == "nt":
                _NT[grid](a, wt, c, count, n, k, GROUP_M=8, **flags, **meta)
            else:
                _MM[grid](
                    a,
                    wt,
                    c,
                    count,
                    n,
                    k,
                    a.stride(0),
                    1,
                    1,
                    k,
                    n,
                    1,
                    dot_out_dtype=tl.float32,
                    GROUP_M=8,
                    UPGRADE=False,
                    UPGRADE_A_OFFS=False,
                    UPGRADE_B_OFFS=False,
                    UPGRADE_C_OFFS=False,
                    **flags,
                    **meta,
                )
    if full_rows < m:
        flag_gems.mm_out(x[full_rows:], wt, out=out[full_rows:])
    return out
