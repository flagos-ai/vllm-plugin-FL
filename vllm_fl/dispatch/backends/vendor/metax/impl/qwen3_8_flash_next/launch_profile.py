# SPDX-License-Identifier: Apache-2.0
"""Opt-in Qwen C550 launch presets over the installed, unchanged kernels.

All tensor computations remain in FG/FGV/vLLM. This module supplies reviewed
launch geometry and native LibTuner configurations; it never edits sources.
"""

from __future__ import annotations

from importlib import import_module
from importlib.resources import files
import json
import hashlib
from pathlib import Path
import math

import torch
import triton

_enabled = False


def _jit(entry):
    while not isinstance(entry, triton.runtime.JITFunction):
        entry = entry.fn
    return entry


class _Launch:
    """Select launch metadata before the existing library cache sees a call."""

    def __init__(self, entry, select, grid=None):
        self.fn = entry
        self.arg_names = _jit(entry).arg_names
        self.select = select
        self.grid = grid

    def __getattr__(self, name):
        return getattr(self.fn, name)

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(
            *args, grid=grid, warmup=False, **kwargs
        )

    def run(self, *args, grid, **kwargs):
        meta = dict(zip(self.arg_names, args))
        meta.update(kwargs)
        selected = self.select(meta)
        if selected is None:
            return self.fn.run(*args, grid=grid, **kwargs)
        args = list(args)
        for name, value in selected.items():
            index = self.arg_names.index(name) if name in self.arg_names else len(args)
            if index < len(args):
                args[index] = value
            else:
                kwargs[name] = value
        meta.update(selected)
        if self.grid is not None:
            grid = self.grid(meta, grid)
        return self.fn.run(*args, grid=grid, **kwargs)


def _mean_inner(meta):
    return meta["input_ptr"].dtype == torch.float32 and meta["N"] in (128, 256, 2560)


def _mean_vec(meta):
    if (
        meta["input_ptr"].dtype == torch.bfloat16
        and meta["N"] == 4
        and meta["K"] == 2560
    ):
        small = meta["M"] < 32
        return dict(
            BLOCK_SIZE_K=128,
            VEC_SIZE=1 if small else 4,
            num_warps=2 if small else 1,
            num_stages=1,
        )
    return None


def _ple_conv(meta):
    if (
        meta["x"].dtype != torch.bfloat16
        or meta["channels"] != 10240
        or meta["width"] != 4
        or meta["dilation"] != 3
    ):
        return None
    m = meta["tokens"]
    return dict(
        BT=1 if m <= 2 else 4 if m <= 4 else 8,
        BC=128 if m == 1 else 64 if m <= 4 else 256 if m >= 32 else 128,
        num_warps=2 if 1 < m <= 4 else 4,
        num_stages=1,
    )


def _ple_state(meta):
    # The opt-in model contract below fixes width=4 and dilation=3.
    if meta["x"].dtype == torch.bfloat16 and meta["channels"] == 10240:
        return dict(num_warps=1, num_stages=1)
    return None


def _qsa_attention(meta):
    if (
        meta["HEAD_DIM"] == 256
        and meta["GROUP_SIZE"] == 3
        and meta["q_ptr"].dtype == meta["k_cache_ptr"].dtype == torch.bfloat16
    ):
        return dict(
            BLOCK_M=max(meta["BLOCK_M"], 16),
            num_warps=4 if meta["num_rows"] <= 64 else 2,
            num_stages=1,
        )
    return None


def _wy(meta):
    if (
        meta["T"] >= 512
        and meta["H"] == 6
        and meta["Hg"] == 2
        and meta["K"] == meta["V"] == 128
        and meta["cu_seqlens"] is not None
        and meta["k"].dtype == torch.bfloat16
    ):
        return dict(BK=32)
    return None


def _sigmoid(meta):
    if (
        meta["H"] == 2
        and meta["HV"] == 6
        and meta["K"] == meta["V"] == 128
        and meta["q"].dtype == meta["k"].dtype == meta["v"].dtype == torch.bfloat16
        and not meta["IS_KDA"]
    ):
        return dict(BV=8, num_warps=1, num_stages=1)
    return None


def _config(row):
    return triton.Config(
        row["META"], num_warps=row["num_warps"], num_stages=row["num_stages"]
    )


class _GDNOutput:
    """Keep a native dispatch cache per T bucket without adding a kernel argument."""

    def __init__(self, entry, configs):
        self.fn = entry
        self.arg_names = _jit(entry).arg_names
        self.configs = configs
        self.entries = {}

    def __getattr__(self, name):
        return getattr(self.fn, name)

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(
            *args, grid=grid, warmup=False, **kwargs
        )

    def run(self, *args, grid, **kwargs):
        meta = dict(zip(self.arg_names, args))
        meta.update(kwargs)
        bucket = triton.next_power_of_2(meta["T"])
        entry = self.entries.get(bucket)
        if entry is None:
            from flaggems_vllm.utils import libentry, libtuner
            from flaggems_vllm.utils.libentry import libcache

            heuristics = self.fn.fn
            native_tuner = heuristics.fn
            keys = list(native_tuner.keys) + ["T"]
            tuner = libtuner(
                configs=self.configs,
                key=keys,
                strategy=["default"] * len(native_tuner.keys)
                + [triton.next_power_of_2],
            )(native_tuner.fn)
            # Separate the seven-component key from historical six-component tables.
            tuner.config_table_name += "_qwen38_context"
            tuner.cache = libcache[tuner.config_table_name]
            entry = libentry()(triton.heuristics(heuristics.values)(tuner))
            self.entries[bucket] = entry
        return entry.run(*args, grid=grid, **kwargs)


def qsa_topk(x, k, dim=-1, largest=True, sorted=False):
    """Use FG's existing two-stage kernels with the reviewed QSA launch geometry."""
    import flag_gems

    n = x.shape[-1]
    if (
        not _enabled
        or x.dtype != torch.float32
        or k != 512
        or n not in (512, 2048, 8192, 17408, 17600)
        or dim not in (-1, x.ndim - 1)
        or not largest
        or sorted
    ):
        return flag_gems.topk(x, k, dim=dim, largest=largest, sorted=sorted)
    module = import_module("flag_gems.ops.topk")
    m = math.prod(x.shape[:-1])
    chunk = min(n, 2048 if 32 < m <= 128 else 4096)
    chunks = triton.cdiv(n, chunk)
    values = torch.empty(m * chunks * k, device=x.device, dtype=x.dtype)
    indices = torch.empty(m * chunks * k, device=x.device, dtype=torch.int64)
    out = torch.empty((*x.shape[:-1], k), device=x.device, dtype=x.dtype)
    out_indices = torch.empty_like(out, dtype=torch.int64)
    stages = 1 if m > 128 else 3
    with module.torch_device_fn.device(x.device):
        module.topk_stage1_kernel[(m, chunks)](
            values,
            indices,
            x,
            k,
            n,
            chunk,
            True,
            num_warps=2 if 32 < m <= 128 else 4,
            num_stages=stages,
        )
        module.topk_stage2_kernel[(m,)](
            out,
            out_indices,
            values,
            indices,
            dim % x.ndim,
            k,
            chunks * k,
            triton.next_power_of_2(chunks * k),
            True,
            num_warps=4,
            num_stages=stages,
        )
    return out, out_indices


def install_qwen38_launch_profile(vllm_config=None):
    """Install before model construction/capture; geometry misses retain native defaults."""
    global _enabled
    if _enabled:
        return
    from flag_gems import runtime
    from flag_gems.utils.libentry import clear_libentry_dispatch_cache

    if (
        runtime.device.vendor_name != "metax"
        or "C550" not in torch.cuda.get_device_name()
    ):
        raise ValueError("The Qwen launch profile requires MetaX C550")
    if vllm_config is not None:
        cfg = vllm_config.model_config.hf_text_config
        if (
            vllm_config.model_config.dtype != torch.bfloat16
            or cfg.hidden_size != 2560
            or cfg.hc_count != 4
            or cfg.ple_conv_kernel_size != 4
            or cfg.ngram_size != 3
        ):
            raise ValueError(
                "The Qwen launch profile requires the reviewed BF16 geometry"
            )
    profile = json.loads(
        files("vllm_fl.dispatch.config")
        .joinpath("qwen3_8_flash_next_metax_tuning.json")
        .read_text()
    )
    for package, requirements in profile["source_files"].items():
        package_root = Path(import_module(package).__file__).parent
        for requirement in requirements:
            path = package_root / requirement["path"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != requirement["sha256"]:
                raise RuntimeError(f"Qwen launch profile dependency mismatch: {path}")
    for operator in ("linear", "mv"):
        entry = getattr(
            import_module("flag_gems.ops." + operator), operator + "_kernel"
        )
        tuner = entry.fn
        candidates = (
            [_config(row) for row in profile["linear"]]
            if operator == "linear"
            else (list(tuner.configs) + [_config(row) for row in profile["mv_extra"]])
        )
        tuner._set_configs_and_strategy(candidates, tuner.strategy)
        clear_libentry_dispatch_cache(entry)
    # Seed the same reviewed, dtype-qualified ConfigCache entries used by the formal run.
    dummy = torch.empty(1, device="cuda", dtype=torch.bfloat16)
    for row in profile["fg_shapes"]:
        operator, shape, selected = row["operator"], row["shape"], row["config"]
        tuner = getattr(
            import_module("flag_gems.ops." + operator), operator + "_kernel"
        ).fn
        args = (
            dict(
                input_ptr=dummy,
                weight_ptr=dummy,
                bias_ptr=dummy,
                output_ptr=dummy,
                **shape,
            )
            if operator == "linear"
            else (dict(A=dummy, B=dummy, C=dummy, N=shape["M"], M=shape["K"]))
        )
        cfg = triton.Config(
            {k: v for k, v in selected.items() if k not in ("num_warps", "num_stages")},
            num_warps=selected["num_warps"],
            num_stages=selected["num_stages"],
        )
        key = tuner.get_key(args)
        existing = tuner.cache.get(key)
        if existing is not None and existing.all_kwargs() != cfg.all_kwargs():
            raise RuntimeError(
                "Conflicting Qwen ConfigCache entry; use a separate cache"
            )
        tuner.cache[key] = cfg

    mean = import_module("flag_gems.ops.mean")
    heuristics = dict(mean.mean_dim_kernel_inner.fn.values)
    mean.mean_dim_kernel_inner.fn.values = heuristics
    old_tile, old_warps = heuristics["TILE_N"], heuristics["num_warps"]
    old_stages = heuristics.get("num_stages", lambda meta: 3)
    heuristics["TILE_N"] = lambda m: (
        (
            m["N"]
            if m["N"] <= 256
            else 4096
            if m["M"] <= 128
            else 1024
            if m["M"] <= 512
            else 512
        )
        if _mean_inner(m)
        else old_tile(m)
    )
    heuristics["num_warps"] = lambda m: 1 if _mean_inner(m) else old_warps(m)
    heuristics["num_stages"] = lambda m: 1 if _mean_inner(m) else old_stages(m)
    clear_libentry_dispatch_cache(mean.mean_dim_kernel_inner)
    mean.mean_dim_kernel_non_inner_vec = _Launch(
        mean.mean_dim_kernel_non_inner_vec,
        _mean_vec,
        lambda m, grid: (
            m["M"],
            triton.cdiv(m["K"], m["BLOCK_SIZE_K"] * m["VEC_SIZE"]),
        ),
    )
    ple = import_module("flaggems_vllm.ops.qwen4.ple_conv")
    ple._ple_conv_kernel = _Launch(
        ple._ple_conv_kernel,
        _ple_conv,
        lambda m, grid: (
            triton.cdiv(m["tokens"], m["BT"]),
            triton.cdiv(m["channels"], m["BC"]),
        ),
    )
    ple._ple_conv_state_kernel = _Launch(ple._ple_conv_state_kernel, _ple_state)
    qsa = import_module("flaggems_vllm.ops.qwen4.qsa_attention")
    qsa._qsa_sparse_paged_gqa_kernel = _Launch(
        qsa._qsa_sparse_paged_gqa_kernel, _qsa_attention
    )
    wy = import_module("flaggems_vllm.ops.FLA.wy_fast")
    wy.recompute_w_u_fwd_kernel = _Launch(wy.recompute_w_u_fwd_kernel, _wy)
    original_wy = wy.recompute_w_u_fwd

    def recompute_w_u(k, v, beta, g_cumsum, A, cu_seqlens, block_v=None, pipe_stages=2):
        if (
            block_v is None
            and k.shape[1] >= 512
            and v.shape[2] == 6
            and k.shape[2:] == (2, 128)
            and v.shape[-1] == 128
            and cu_seqlens is not None
            and k.dtype == torch.bfloat16
        ):
            block_v = 32
        return original_wy(k, v, beta, g_cumsum, A, cu_seqlens, block_v, pipe_stages)

    wy.recompute_w_u_fwd = recompute_w_u
    chunk = import_module("flaggems_vllm.runtime.backend._metax.fused.gdn_chunk")
    chunk.recompute_w_u_fwd = recompute_w_u
    chunk.chunk_fwd_kernel_o = _GDNOutput(
        chunk.chunk_fwd_kernel_o,
        [_config(row) for row in profile["gdn_chunk_o"]],
    )
    sigmoid = import_module("vllm.model_executor.layers.fla.ops.fused_sigmoid_gating")
    sigmoid.fused_sigmoid_gating_delta_rule_update_kernel = _Launch(
        sigmoid.fused_sigmoid_gating_delta_rule_update_kernel,
        _sigmoid,
        lambda m, grid: (
            triton.cdiv(m["K"], m["BK"]),
            triton.cdiv(m["V"], m["BV"]),
            m["N"] * m["HV"],
        ),
    )
    _enabled = True
