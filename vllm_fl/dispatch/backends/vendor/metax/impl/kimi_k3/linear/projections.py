"""Reuse FlagGems' chosen BF16 GEMM directly, caching only CPU launch metadata."""

import torch, triton
import triton.language as tl
import flag_gems
from vllm_fl.ops.kimi_k3.graph_context import in_full_graph
from vllm_fl.dispatch.backends.flaggems.impl.kimi_k3_mm import (
    mm_kernel_nt,
    mm_kernel,
    mm_kernel_splitk_partial,
    mm_kernel_splitk_reduce,
    _select_two_step_split_k,
    _generic_two_step_split_k,
    splitk_mm_scenario,
    nt_mm_scenario,
    _prefer_dense_nt_over_generic_splitk,
)

from vllm_fl.dispatch.backends.flaggems.impl.kimi_k3_mm import (
    NT_KERNEL as _NT,
    MM_KERNEL as _MM,
    PARTIAL_KERNEL as _PARTIAL,
    REDUCE_KERNEL as _REDUCE,
)

_CONFIGS = {}
# Qualified on real TP8 weights and full native dense MLP, including changing Graph inputs.
_QUALIFIED_DECODE_SPLITS = {(8, 7168, 4224): 2}


def _kv_expansion_tile(m):
    # Keep only measured eager gains; small prefill projections use the old path.
    if m in (4096, 6144, 7680, 8191) or 8193 <= m <= 66048:
        return (128, 128, 32, 8, 2)
    return None


_F32_DTYPE_MARKER = torch.empty(0, dtype=torch.float32)


def run(x, w, out, record=None):
    m, k = x.shape
    n = w.shape[0]
    assert (
        x.dtype == w.dtype == out.dtype == torch.bfloat16
        and x.stride(1) == 1
        and w.is_contiguous()
        and out.is_contiguous()
        and n >= 896
    )
    wt = w.T
    key = (x.device.index, m, n, k, x.stride(0))
    if key not in _CONFIGS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm qualified GEMM metadata before Graph capture")
        fixed_split = _QUALIFIED_DECODE_SPLITS.get((m, n, k))
        if fixed_split is not None:
            _CONFIGS[key] = {
                "kind": "splitk",
                "split": fixed_split,
                "meta": {
                    "BLOCK_M": 16,
                    "BLOCK_N": 64,
                    "BLOCK_K": 64,
                    "pipeline": "null",
                    "num_warps": 4,
                    "num_stages": 2,
                    "num_ctas": 1,
                },
                "config_key": "qualified-real-TP8-M8",
            }
        elif (n, k) == (3072, 512) and _kv_expansion_tile(m) is not None:
            bm, bn, bk, warps, stages = _kv_expansion_tile(m)
            _CONFIGS[key] = {
                "kind": "general",
                "split": None,
                "meta": {
                    "BLOCK_M": bm,
                    "BLOCK_N": bn,
                    "BLOCK_K": bk,
                    "pipeline": "basic",
                    "num_warps": warps,
                    "num_stages": stages,
                    "num_ctas": 1,
                },
                "config_key": "screened-TP8-KV-expansion",
            }
        else:
            flag_gems.mm_out(x, wt, out=out)
            nt = nt_mm_scenario(x, wt, out, m, n, k)
            split = _select_two_step_split_k(m, n, k)
            if (
                split is None
                and splitk_mm_scenario(m, n, k)
                and not (nt and _prefer_dense_nt_over_generic_splitk(m, n, k))
            ):
                split = _generic_two_step_split_k(m, n, k)
            kind = "splitk" if split is not None else ("nt" if nt else "general")
            tuner = (
                mm_kernel_splitk_partial
                if kind == "splitk"
                else mm_kernel_nt
                if kind == "nt"
                else mm_kernel
            ).fn
            params = {
                "A": x,
                "B": wt,
                ("P" if kind == "splitk" else "C"): (
                    _F32_DTYPE_MARKER if kind == "splitk" else out
                ),
                "M": m,
                "N": n,
                "K": k,
                "stride_am": x.stride(0),
                "stride_ak": 1,
                "stride_bk": 1,
                "stride_bn": k,
                "stride_cm": n,
                "stride_cn": 1,
            }
            config_key = tuner.get_key(params)
            meta = tuner.cache[config_key].all_kwargs()
            _CONFIGS[key] = {
                "kind": kind,
                "split": split,
                "meta": dict(meta),
                "config_key": repr(config_key),
            }
    cfg = _CONFIGS[key]
    kind = cfg["kind"]
    meta = cfg["meta"]
    grid = (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    if kind == "splitk":
        splits = cfg["split"]
        tmp = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
        kernel = _PARTIAL[(grid[0], splits)](
            x, wt, tmp, m, n, k, x.stride(0), 1, 1, k, SPLIT_K=splits, **meta
        )
        _REDUCE[(triton.cdiv(m * n, 256),)](
            tmp, out, m, n, n, 1, SPLIT_K=splits, BLOCK_SIZE=256, num_warps=4
        )
    else:
        flags = {
            "EVEN_M": m % meta["BLOCK_M"] == 0,
            "EVEN_N": n % meta["BLOCK_N"] == 0,
            "EVEN_K": k % meta["BLOCK_K"] == 0,
        }
        if kind == "nt":
            kernel = _NT[grid](x, wt, out, m, n, k, GROUP_M=8, **flags, **meta)
        else:
            kernel = _MM[grid](
                x,
                wt,
                out,
                m,
                n,
                k,
                x.stride(0),
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
    if record is not None:
        record.update(
            **cfg,
            n_regs=kernel.n_regs,
            n_spills=kernel.n_spills,
            shared=kernel.metadata.shared,
        )
    return out


_DECODE_MM = frozenset(
    {
        (2304, 1536),
        (7168, 1536),
        (2112, 7168),
        (6288, 7168),
        (8448, 7168),
        (7168, 768),
        (20480, 7168),
    }
)
_FIXED = {
    (128, 7168, 1536): (128, 128, 64, 8, 2),
    (512, 2304, 1536): (128, 128, 64, 8, 2),
    (512, 6288, 7168): (128, 128, 64, 8, 2),
    (128, 8448, 7168): (128, 128, 64, 8, 2),
    (128, 7168, 768): (128, 128, 64, 8, 2),
}

# PP keeps all local decode tokens: qualify the M32/M64 projection shapes.
_PP_GRAPH_MM = frozenset(
    {(2304, 1536), (7168, 1536), (1536, 7168), (7168, 768), (7168, 4224)}
)
_PP_ALL_MODE_MM = frozenset({(8448, 7168), (20480, 7168)})


def plan(m, n, k):
    # Dense M64 gate/up changed cancellation in the complete MLP; keep old path.
    if m == 64 and (n, k) == (8448, 7168):
        return None
    if m in (32, 64) and (n, k) in _PP_ALL_MODE_MM:
        return ("mm", None)
    if m in (24, 32, 64) and (n, k) in _PP_GRAPH_MM:
        return ("mm", None) if in_full_graph() else None
    if (n, k) == (3072, 512):
        return ("mm", None) if m == 8192 or _kv_expansion_tile(m) is not None else None
    if (m, n, k) in _QUALIFIED_DECODE_SPLITS:
        return ("mm", None) if in_full_graph() else None
    # Native KDA state-chain gates reject these input projection reorderings.
    if (n, k) == (6288, 7168) and m in (2, 128, 512):
        return None
    if (m, n, k) in _FIXED:
        return ("fixed", _FIXED[(m, n, k)])
    if 1 <= m <= 16 and (n, k) in _DECODE_MM:
        # R4 already uses the same MM kernels for these captured powers of two.
        if (n, k) == (2112, 7168) and m in (1, 2, 4, 8, 16):
            return None
        return ("mm", None)
    if (n, k) == (1536, 7168) and (m == 128 or 450 <= m <= 511):
        return ("mm", None)
    if m == 128 and (n, k) in ((2112, 7168), (6288, 7168)):
        return ("mm", None)
    return None


def try_run(x, w, bias, out, fixed_kernel):
    if bias is not None or x.ndim != 2 or w.ndim != 2:
        return False
    m, k = x.shape
    n = w.shape[0]
    selected = plan(m, n, k)
    if selected is None:
        return False
    if (
        x.dtype != torch.bfloat16
        or w.dtype != x.dtype
        or out.dtype != x.dtype
        or not x.is_cuda
        or x.device != w.device
        or out.device != x.device
        or not w.is_contiguous()
        or not out.is_contiguous()
        or not (x.is_contiguous() or ((n, k) == (3072, 512) and x.stride() == (576, 1)))
    ):
        return False
    if (
        (n, k) == (3072, 512)
        and _kv_expansion_tile(m) is not None
        and x.is_contiguous()
        and m not in (4096, 65024)
    ):
        return False
    kind, meta = selected
    if kind == "mm":
        run(x, w, out)
    else:
        bm, bn, bk, warps, stages = meta
        fixed_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            x,
            w,
            w,
            out,
            m,
            n,
            k,
            x.stride(0),
            1,
            k,
            1,
            n,
            1,
            0,
            BIAS=False,
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            num_warps=warps,
            num_stages=stages,
        )
    return True
