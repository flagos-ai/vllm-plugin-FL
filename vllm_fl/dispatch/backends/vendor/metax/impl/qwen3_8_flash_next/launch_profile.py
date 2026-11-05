# SPDX-License-Identifier: Apache-2.0
"""Opt-in Qwen C550 launch presets over the installed, unchanged kernels.

All tensor computations remain in FG/FGV/vLLM. This module supplies reviewed
launch geometry and native LibTuner configurations; it never edits sources.
"""

from __future__ import annotations

import hashlib
from copy import copy
from importlib import import_module
from pathlib import Path
from threading import RLock

import torch
import triton

from .shape_limits import load_tuning_config, max_prefill_rows

_enabled = False
_install_lock = RLock()


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


def _qsa_score(meta, presets):
    if (
        meta["q_ptr"].dtype == meta["k_cache_ptr"].dtype == torch.bfloat16
        and meta["NUM_HEADS"] == 4
        and meta["HEAD_DIM"] == 128
        and 8 <= meta["num_rows"] <= max_prefill_rows()
        and 512 < meta["num_columns"] <= 17600
    ):
        row = next(p for p in presets if meta["num_columns"] <= p["max_columns"])
        return {k: v for k, v in row.items() if k != "max_columns"}
    return None


def _qsa_radix(meta, preset):
    if (
        meta["logits"].dtype == torch.float32
        and 128 < meta["logits"].shape[0] <= max_prefill_rows()
        and meta["width"] == 2048
        and meta["k"] == 512
    ):
        return preset
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


class _PresetConfigCache:
    """Process-local immutable presets over the native cache.

    Startup never writes preset rows to SQLite. Unspecified keys retain the
    native cache's read/write behavior. Rewriting a pinned key is a no-op,
    matching ConfigCache.put_config for an existing record.
    """

    def __init__(self, native, presets):
        self.native = native
        self.presets = dict(presets)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def get(self, key):
        return self.presets[key] if key in self.presets else self.native.get(key)

    def __contains__(self, key):
        return self.get(key) is not None

    def __getitem__(self, key):
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __setitem__(self, key, config):
        if key not in self.presets:
            self.native[key] = config

    def set(self, key, config):
        self[key] = config


def _copy_entry(entry, fn):
    """Use a fresh dispatch cache without clearing the published entry's caches."""
    result = copy(entry)
    result.fn = fn
    result.kernel_cache = [dict() for _ in entry.kernel_cache]
    result._cpu_cache = {}
    return result


def _prepare_launch_profile(profile, vllm_config):
    """Resolve dependencies and construct all replacements before publication."""
    changes = []
    entries = {}
    for operator in ("linear", "mv"):
        module = import_module("flag_gems.ops." + operator)
        original = getattr(module, operator + "_kernel")
        tuner = copy(original.fn)
        candidates = (
            [_config(row) for row in profile["linear"]]
            if operator == "linear"
            else list(tuner.configs) + [_config(row) for row in profile["mv_extra"]]
        )
        tuner._set_configs_and_strategy(candidates, tuner.strategy)
        entries[operator] = tuner
        changes.append((module, operator + "_kernel", _copy_entry(original, tuner)))

    dummy = torch.empty(1, device="cuda", dtype=torch.bfloat16)
    presets = {name: {} for name in entries}
    for row in profile["fg_shapes"]:
        operator, shape, selected = row["operator"], row["shape"], row["config"]
        tuner = entries[operator]
        args = (
            dict(
                input_ptr=dummy,
                weight_ptr=dummy,
                bias_ptr=dummy,
                output_ptr=dummy,
                **shape,
            )
            if operator == "linear"
            else dict(A=dummy, B=dummy, C=dummy, N=shape["M"], M=shape["K"])
        )
        config = triton.Config(
            {k: v for k, v in selected.items() if k not in ("num_warps", "num_stages")},
            num_warps=selected["num_warps"],
            num_stages=selected["num_stages"],
        )
        key = tuner.get_key(args)
        existing = presets[operator].get(key)
        if existing is None:
            existing = tuner.cache.get(key)
        if existing is not None and existing.all_kwargs() != config.all_kwargs():
            raise RuntimeError(
                "Conflicting Qwen ConfigCache entry; use a separate cache"
            )
        presets[operator][key] = config
    for name, tuner in entries.items():
        tuner.cache = _PresetConfigCache(tuner.cache, presets[name])

    mean = import_module("flag_gems.ops.mean")
    original = mean.mean_dim_kernel_inner
    inner = copy(original.fn)
    heuristics = dict(inner.values)
    inner.values = heuristics
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
    changes.append((mean, "mean_dim_kernel_inner", _copy_entry(original, inner)))
    changes.append(
        (
            mean,
            "mean_dim_kernel_non_inner_vec",
            _Launch(
                mean.mean_dim_kernel_non_inner_vec,
                _mean_vec,
                lambda m, grid: (
                    m["M"],
                    triton.cdiv(m["K"], m["BLOCK_SIZE_K"] * m["VEC_SIZE"]),
                ),
            ),
        )
    )

    ple = import_module("flaggems_vllm.ops.qwen4.ple_conv")
    changes.extend(
        [
            (
                ple,
                "_ple_conv_kernel",
                _Launch(
                    ple._ple_conv_kernel,
                    _ple_conv,
                    lambda m, grid: (
                        triton.cdiv(m["tokens"], m["BT"]),
                        triton.cdiv(m["channels"], m["BC"]),
                    ),
                ),
            ),
            (
                ple,
                "_ple_conv_state_kernel",
                _Launch(ple._ple_conv_state_kernel, _ple_state),
            ),
        ]
    )
    qsa = import_module("flaggems_vllm.ops.qwen4.qsa_attention")
    for name in ("_qsa_sparse_paged_gqa_kernel", "_qsa_sparse_paged_gqa_split_kernel"):
        changes.append((qsa, name, _Launch(getattr(qsa, name), _qsa_attention)))
    if vllm_config is None or (
        vllm_config.speculative_config is None
        and vllm_config.additional_config.get("qwen38_qsa_long_context", False)
    ):
        indexer = import_module("flaggems_vllm.ops.qwen4.qsa")
        changes.append(
            (
                indexer,
                "_qsa_mqa_paged_dot_kernel",
                _Launch(
                    indexer._qsa_mqa_paged_dot_kernel,
                    lambda meta: _qsa_score(meta, profile["qsa_indexer_launch"]),
                    lambda m, grid: (
                        m["num_rows"],
                        triton.cdiv(m["num_columns"], m["BLOCK_N"]),
                    ),
                ),
            )
        )
        radix = import_module("flaggems_vllm.runtime.backend._metax.ops.radix_topk")
        changes.append(
            (
                radix,
                "_radix_topk_indices",
                _Launch(
                    radix._radix_topk_indices,
                    lambda meta: _qsa_radix(meta, profile["qsa_radix_short"]),
                ),
            )
        )
    wy = import_module("flaggems_vllm.ops.FLA.wy_fast")
    changes.append(
        (wy, "recompute_w_u_fwd_kernel", _Launch(wy.recompute_w_u_fwd_kernel, _wy))
    )
    original_wy = wy.recompute_w_u_fwd

    def recompute_w_u(
        k,
        v,
        beta,
        g_cumsum,
        A,
        cu_seqlens,
        block_v=None,
        pipe_stages=2,
        chunk_indices=None,
    ):
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
        return original_wy(
            k,
            v,
            beta,
            g_cumsum,
            A,
            cu_seqlens,
            block_v,
            pipe_stages,
            chunk_indices=chunk_indices,
        )

    chunk = import_module("flaggems_vllm.runtime.backend._metax.fused.gdn_chunk")
    changes.extend(
        [
            (wy, "recompute_w_u_fwd", recompute_w_u),
            (chunk, "recompute_w_u_fwd", recompute_w_u),
            (
                chunk,
                "chunk_fwd_kernel_o",
                _GDNOutput(
                    chunk.chunk_fwd_kernel_o,
                    [_config(row) for row in profile["gdn_chunk_o"]],
                ),
            ),
        ]
    )
    sigmoid = import_module("vllm.model_executor.layers.fla.ops.fused_sigmoid_gating")
    name = "fused_sigmoid_gating_delta_rule_update_kernel"
    changes.append(
        (
            sigmoid,
            name,
            _Launch(
                getattr(sigmoid, name),
                _sigmoid,
                lambda m, grid: (
                    triton.cdiv(m["K"], m["BK"]),
                    triton.cdiv(m["V"], m["BV"]),
                    m["N"] * m["HV"],
                ),
            ),
        )
    )
    changes.extend(
        [
            (torch.backends.cuda.matmul, "allow_tf32", False),
            (
                torch.backends.cuda.matmul,
                "allow_bf16_reduced_precision_reduction",
                False,
            ),
        ]
    )
    return changes


def install_qwen38_launch_profile(vllm_config=None):
    """Atomically install the scoped profile before construction or capture."""
    global _enabled
    from flag_gems import runtime

    from vllm_fl.patches._qwen38_installation import apply_attributes

    with _install_lock:
        if _enabled:
            return
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
        profile = load_tuning_config()
        for package, requirements in profile["source_files"].items():
            package_root = Path(import_module(package).__file__).parent
            for requirement in requirements:
                path = package_root / requirement["path"]
                if (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    != requirement["sha256"]
                ):
                    raise RuntimeError(
                        f"Qwen launch profile dependency mismatch: {path}"
                    )
        changes = _prepare_launch_profile(profile, vllm_config)
        apply_attributes(changes)
        _enabled = True
