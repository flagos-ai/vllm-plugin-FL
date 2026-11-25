# Copyright (c) 2026 BAAI. All rights reserved.

"""vLLM official Custom AllReduce, source-built for Hygon (gfx936).

The upstream wheel installs vLLM without compiled extensions
(``VLLM_TARGET_DEVICE=empty``), so ``torch.ops._C_custom_ar`` is unregistered
and vLLM's :class:`CustomAllreduce` disables itself, leaving every TP
allreduce on NCCL.  NCCL carries a ~70us fixed latency at decode message
sizes (<=4MB); the IPC one/two-shot kernels cut that to ~20us (measured on
BW1000, see ``performance_opt/dv4_model/TP-commu``).

``csrc/`` holds the upstream kernel (``custom_all_reduce.cuh``, already
HIP-aware) plus a classic-at::Tensor binding that registers the same op
namespace/ABI vLLM's python layer expects, so the stock
:class:`vllm.distributed.device_communicators.custom_all_reduce.CustomAllreduce`
works unchanged (8MB size threshold, NCCL fallback for larger messages,
CUDA-graph buffer registration).

Algorithm default: the upstream heuristic picks one-shot for <512KB (TP<=4),
but on BW1000 two-shot wins everywhere (up to -16% at 288KB, and one-shot is
up to 69% slower at 4MB), so ``2stage`` is the default here.  Users can still
override with ``VLLM_CUSTOM_ALLREDUCE_ALGO=1stage``/``2stage``/unset.
"""

from __future__ import annotations

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "csrc")

# First build takes ~1-2 min; afterwards torch caches the extension and the
# ninja no-op check costs seconds even with concurrent workers (file baton).
_EXT_NAME = "vllm_fl_ca"


def _target_arch() -> str:
    """gfx string for PYTORCH_ROC_ARCH (fall back to the env / gfx936)."""
    arch = os.environ.get("PYTORCH_ROC_ARCH")
    if arch:
        return arch
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"gfx{major}{minor}"
    except Exception:  # pragma: no cover - defensive
        pass
    return "gfx936"  # BW1000


def _build_extension():
    from torch.utils.cpp_extension import load

    old = os.environ.get("PYTORCH_ROC_ARCH")
    os.environ["PYTORCH_ROC_ARCH"] = _target_arch()
    try:
        return load(
            name=_EXT_NAME,
            sources=[os.path.join(_CSRC, "ca_bindings.cu")],
            extra_include_paths=[_CSRC],
            extra_cuda_cflags=["-O3", "-DUSE_ROCM=1"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    finally:
        if old is None:
            os.environ.pop("PYTORCH_ROC_ARCH", None)
        else:
            os.environ["PYTORCH_ROC_ARCH"] = old


@functools.lru_cache(maxsize=1)
def _ensure_ops_impl() -> bool:
    if os.environ.get("VLLM_FL_CUSTOM_AR", "1").lower() in ("0", "false", "no"):
        logger.info("Hygon custom allreduce disabled via VLLM_FL_CUSTOM_AR")
        return False

    backend = os.environ.get("VLLM_FL_CUSTOM_AR_BACKEND", "hcu")

    # Preferred: the Hygon-built hcu_ops binary.  The vLLM-0.24 kernel this
    # plugin can source-build is numerically correct eager but corrupts
    # ~99% of elements under CUDA-graph capture/replay (verified with an
    # fp64 reference, 8 GPUs, 40/40 replays); hcu_ops passes both eager and
    # graph mode with the identical vLLM python protocol, and it is what
    # the native Hygon vLLM ships.  Same op ABI, so we bridge vLLM's
    # ``_custom_ops`` attribute access to the hcu_ops namespace.
    if backend == "hcu":
        try:
            import vllm_hcu.hcu_ops  # noqa: F401

            ns = torch.ops.hcu_ops
            int(ns.meta_size())
        except Exception as exc:
            logger.warning(
                "hcu_ops custom allreduce unavailable (%s: %s); trying the "
                "source-built kernel (CUDA-graph correctness NOT validated)",
                type(exc).__name__,
                exc,
            )
            backend = "source"

    if backend == "source":
        try:
            _build_extension()
            torch.ops._C_custom_ar.meta_size()
        except Exception as exc:
            logger.warning(
                "Hygon custom-allreduce extension unavailable "
                "(%s: %s); TP allreduce stays on NCCL",
                type(exc).__name__,
                exc,
            )
            return False

    # BW1000 interconnect: two-shot >= one-shot at every measured size.
    # setdefault => user can still override explicitly.
    os.environ.setdefault("VLLM_CUSTOM_ALLREDUCE_ALGO", "2stage")

    # Point vLLM's CustomAllreduce at the chosen op namespace and (re)arm
    # its module-level availability probe.
    try:
        import vllm.distributed.device_communicators.custom_all_reduce as ca_mod

        if backend == "hcu":
            target = torch.ops.hcu_ops
            for name in (
                "meta_size",
                "init_custom_ar",
                "all_reduce",
                "register_buffer",
                "register_graph_buffers",
                "get_graph_buffer_ipc_meta",
                "allocate_shared_buffer_and_handle",
                "open_mem_handle",
                "free_shared_buffer",
                "dispose",
            ):
                setattr(ca_mod.ops, name, getattr(target, name))
        if not ca_mod.custom_ar:
            ca_mod.custom_ar = True
    except Exception:  # pragma: no cover - vllm import order safety
        pass

    logger.info(
        "Hygon custom allreduce enabled (backend=%s; algo=%s)",
        backend,
        os.environ.get("VLLM_CUSTOM_ALLREDUCE_ALGO", "auto"),
    )
    return True


def ensure_custom_allreduce_ops() -> bool:
    """Idempotent: build+register ``torch.ops._C_custom_ar`` once per process."""
    return _ensure_ops_impl()
