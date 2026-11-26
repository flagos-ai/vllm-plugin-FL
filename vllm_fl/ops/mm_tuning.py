"""Startup gate for reusing nearby matrix-multiply tuning keys."""

import gc
import hashlib
import importlib
from contextlib import nullcontext
from pathlib import Path

import torch

from vllm.logger import init_logger

from vllm_fl import device_profiles, envs

logger = init_logger("vllm.vllm_fl.mm_tuning")
_SWITCH = "VLLM_FL_MM_TUNING_BUCKETS"


def _get_mm_module():
    import flag_gems

    registered_mm = getattr(flag_gems, "mm", None)
    module_name = getattr(registered_mm, "__module__", None)
    if not module_name:
        raise RuntimeError("FlagGems has no registered mm implementation")
    module = importlib.import_module(module_name)
    expected = (
        Path(flag_gems.__file__).resolve().parent
        / "runtime"
        / "backend"
        / "_metax"
        / "ops"
        / "mm.py"
    ).resolve()
    if getattr(module, "mm", None) is not registered_mm:
        raise RuntimeError("selected FlagGems mm does not match its imported module")
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError(
            f"selected mm module is outside the expected backend path: {module.__file__}"
        )
    if not callable(getattr(module, "set_mm_tuning_buckets", None)) or not hasattr(
        module, "mm_kernel_nt"
    ):
        raise RuntimeError("selected mm module lacks the tuning-bucket interface")
    return module


def source_hash():
    """Hash the backend selector and startup gate for compile caches."""
    module = _get_mm_module()
    digest = hashlib.sha256()
    for path in (Path(module.__file__), Path(__file__)):
        digest.update(path.resolve().read_bytes())
    return digest.hexdigest()


def disable():
    """Disable the bucket strategy on the selected registered backend."""
    _get_mm_module().set_mm_tuning_buckets(False)


def _find_tuner(kernel):
    candidate = getattr(kernel, "fn", None)
    visited = set()
    while candidate is not None and id(candidate) not in visited:
        visited.add(id(candidate))
        if hasattr(candidate, "keys") and hasattr(candidate, "cache"):
            return candidate
        candidate = getattr(candidate, "fn", None)
    raise TypeError("mm_kernel_nt.fn chain has no config tuner")


def _check_bucket_group(module, tuner, a_cpu, w_cpu, a, w, anchor_m, tail_ms):
    module.set_mm_tuning_buckets(False)
    anchor_a = a[:anchor_m]
    anchor = torch.mm(anchor_a, w.T)
    torch.cuda.synchronize(a.device)
    anchor_args = {
        "A": anchor_a,
        "B": w.T,
        "C": anchor,
        "M": anchor_m,
        "N": 128,
        "K": 128,
    }
    anchor_key = tuple(tuner.get_key(anchor_args))
    if not torch.isfinite(anchor).all().item():
        raise RuntimeError(f"M={anchor_m} self-check produced non-finite output")
    module.set_mm_tuning_buckets(True)
    tail_keys = []
    for tail_m in tail_ms:
        tail_a = a[:tail_m]
        tail = torch.mm(tail_a, w.T)
        torch.cuda.synchronize(a.device)
        tail_args = {"A": tail_a, "B": w.T, "C": tail, "M": tail_m, "N": 128, "K": 128}
        tail_key = tuple(tuner.get_key(tail_args))
        benchmark_key = tuple(tuner.get_benchmark_key(tail_args))
        if tail_key != anchor_key:
            raise RuntimeError(f"M={tail_m} config key did not reuse M={anchor_m}")
        if not benchmark_key or benchmark_key[0] != tail_m:
            raise RuntimeError(f"M={tail_m} exact benchmark key did not retain raw M")
        if not torch.isfinite(tail).all().item():
            raise RuntimeError(f"M={tail_m} self-check produced non-finite output")
        reference = torch.mm(a_cpu[:tail_m].float(), w_cpu.float().T)
        torch.testing.assert_close(
            tail.cpu().float(), reference, rtol=0.02, atol=0.0005
        )
        counts = (
            int(getattr(tuner, "benchmark_success_count", 0) or 0),
            int(getattr(tuner, "benchmark_cache_hit_count", 0) or 0),
        )
        if any(counts):
            raise RuntimeError(
                f"M={tail_m} self-check performed tuning work: counts={counts}"
            )
        tail_keys.append(tail_key)
    return tail_keys


def _self_check(module, device):
    import flag_gems

    tuner = _find_tuner(module.mm_kernel_nt)
    if tuple(getattr(tuner, "keys", ())) != ("M", "N", "K"):
        raise RuntimeError("mm_kernel_nt tuner keys are not M/N/K")
    dispatch = (
        nullcontext()
        if getattr(flag_gems, "current_work_registrar", None) is not None
        else flag_gems.use_gems()
    )
    with dispatch:
        generator = torch.Generator(device="cpu").manual_seed(72931)
        a_cpu = torch.randn((2048, 128), generator=generator, dtype=torch.bfloat16)
        w_cpu = torch.randn((128, 128), generator=generator, dtype=torch.bfloat16)
        a, w = a_cpu.to(device), w_cpu.to(device)
        tail_keys = _check_bucket_group(
            module, tuner, a_cpu, w_cpu, a, w, 2048, (1031,)
        )
        tail_keys.extend(
            _check_bucket_group(module, tuner, a_cpu, w_cpu, a, w, 512, (477, 769))
        )
    return tail_keys


def install(device) -> bool:
    if device.type != "cuda" or torch.version.hip is not None:
        envs.switch(_SWITCH, auto=False)
        return False

    module = None
    try:
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        profile = device_profiles.lookup(index) or {}
        if not envs.switch(_SWITCH, auto=profile.get("mm_tuning_buckets") is True):
            return False
        module = _get_mm_module()
        module.set_mm_tuning_buckets(False)
        tail_keys = _self_check(module, device)
        logger.info(
            "MM tuning bucket CPU/config reuse self-check passed: "
            "anchor_tail_pairs=2048:1031,512:477,512:769 normalized_keys=%s",
            tail_keys,
        )
        return True
    except Exception:
        if module is not None:
            try:
                module.set_mm_tuning_buckets(False)
            except Exception:
                logger.debug(
                    "Could not disable MM tuning buckets after failure", exc_info=True
                )
        logger.warning(
            "MM tuning bucket CPU/config reuse self-check failed", exc_info=True
        )
        return False
    finally:
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.debug(
                "Could not release MM tuning self-check allocations", exc_info=True
            )
