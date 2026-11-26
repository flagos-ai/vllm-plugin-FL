# Copyright (c) 2026 BAAI. All rights reserved.
"""Loader for the native (matrix-core) decode attention kernel.

The kernel lives in ``native/gqa_decode_native.cu`` next to this file and is compiled once with
torch's cpp_extension into a per-user build directory, then cached across processes.
``VLLM_FL_NATIVE_DECODE=0`` disables it; if the build fails for any reason the Triton kernels
are used instead, so serving never depends on the native build.
"""

import logging
import os
import threading

import torch

logger = logging.getLogger(__name__)
_ENABLED = os.environ.get("VLLM_FL_NATIVE_DECODE", "1") == "1"
_lock = threading.Lock()
_mod = None
_failed = False


def _build():
    from torch.utils.cpp_extension import load

    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native")
    build_dir = os.environ.get("VLLM_FL_NATIVE_BUILD_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "vllm_fl_native", "gqa_decode_native_v5"
    )
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="vllm_fl_gqa_decode_native_v5",
        sources=[
            os.path.join(here, "gqa_decode_native_binding.cpp"),
            os.path.join(here, "gqa_decode_native.cu"),
        ],
        extra_cuda_cflags=["-O3"],
        build_directory=build_dir,
        verbose=False,
    )


def get_module():
    """The compiled extension module, or None when disabled / unavailable."""
    global _mod, _failed
    if not _ENABLED or _failed:
        return None
    if _mod is not None:
        return _mod
    with _lock:
        if _mod is None and not _failed:
            try:
                _mod = _build()
                logger.info("native GQA decode attention kernel loaded")
            except Exception as e:  # noqa: BLE001
                _failed = True
                logger.warning(
                    "native GQA decode attention kernel unavailable (%s); using the Triton kernel",
                    e,
                )
    return _mod


def gqa_decode_native(
    q,
    k_cache,
    v_cache,
    out,
    seq_lens,
    block_table,
    softmax_scale,
    combine_kernel,
    num_splits,
    workspace,
    min_tiles=8,
    i8=False,
    mode: int | None = None,
):
    """Same contract as the Triton decode kernels; returns False if the native kernel is unavailable."""
    mod = get_module()
    if mod is None:
        return False
    S, H, D = q.shape
    eff_mode = int(mode) if mode is not None else (2 if i8 else 0)
    if eff_mode == 2:
        if k_cache.dim() != 5 or k_cache.shape[2] != 16:
            return False
        Hkv = k_cache.shape[3]
    else:
        if k_cache.dim() != 4 or k_cache.shape[1] != 16:
            return False
        Hkv = k_cache.shape[2]
    if Hkv <= 0 or H // Hkv != 8 or D != 128:
        return False
    ws_acc, ws_m, ws_l = workspace.get(S, H, num_splits, D, q.device)
    mod.gqa_decode_native(
        q,
        k_cache,
        v_cache,
        out,
        seq_lens,
        block_table,
        ws_acc,
        ws_m,
        ws_l,
        softmax_scale * 1.4426950408889634,
        num_splits,
        min_tiles,
        eff_mode,
    )
    if num_splits > 1:
        import triton

        combine_kernel[(S, H)](
            ws_acc,
            ws_m,
            ws_l,
            out,
            seq_lens,
            out.stride(0),
            out.stride(1),
            NUM_HEADS=H,
            MAX_SPLITS=num_splits,
            SPLITS_POW2=triton.next_power_of_2(num_splits),
            MIN_TILES=min_tiles,
            TILE_N=16,
            HEAD_DIM=D,
            num_warps=1,
        )
    return True
