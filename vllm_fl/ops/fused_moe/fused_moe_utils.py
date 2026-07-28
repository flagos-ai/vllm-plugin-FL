
from enum import Enum
import json
import os
import time
from typing import Any

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
from vllm.model_executor.layers.fused_moe.oracle.unquantized import UnquantizedMoeBackend, map_unquantized_backend, backend_to_kernel_cls
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts, try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.utils import _resize_cache, moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    get_flashinfer_moe_backend,
)
from vllm.triton_utils import tl, triton
from vllm_fl.dispatch import call_op
from vllm_fl.ops.fused_moe.activation import apply_moe_activation
from vllm_fl.utils import use_flaggems

logger = init_logger(__name__)

_MOE_CONFIG_KEYS = {
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "SPLIT_K",
    "num_warps",
    "num_stages",
}

_MOE_RUNTIME_PROFILE_LOGGED: set[tuple[str, int]] = set()
_MOE_RUNTIME_PROFILE_TRACE_LOGGED: set[tuple[Any, ...]] = set()

_MOE_SMALL_DECODE_CONFIGS: dict[int, dict[str, dict[str, int]]] = {
    1: {
        "stage1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 32, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
        "stage2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    },
    2: {
        "stage1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
        "stage2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 64, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    },
    4: {
        "stage1": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
        "stage2": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    },
    8: {
        "stage1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 32, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
        "stage2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    },
    16: {
        "stage1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
        "stage2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    },
    24: {
        "stage1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
        "stage2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    },
    32: {
        "stage1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
        "stage2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    },
    40: {
        "stage1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
        "stage2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    },
    48: {
        "stage1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
        "stage2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    },
    56: {
        "stage1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
        "stage2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    },
    64: {
        "stage1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
        "stage2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    },
}


def _sanitize_moe_config(config: dict[str, Any]) -> dict[str, Any]:
    sanitized = {key: value for key, value in config.items() if key in _MOE_CONFIG_KEYS}
    sanitized.setdefault("SPLIT_K", 1)
    return sanitized


def _limit_config_for_mcoplib(config: dict[str, Any]) -> dict[str, Any]:
    limited = config.copy()
    if limited.get("BLOCK_SIZE_K", 0) > 64:
        limited["BLOCK_SIZE_K"] = 64
    if limited.get("num_stages", 0) > 3:
        limited["num_stages"] = 3
    return limited


def _split_staged_moe_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage1 = config.get("stage1", config)
    stage2 = config.get("stage2", config)
    stage1_config = _limit_config_for_mcoplib(_sanitize_moe_config(stage1))
    stage2_config = _limit_config_for_mcoplib(_sanitize_moe_config(stage2))

    # moe_align_block_size pads according to the stage1 BLOCK_SIZE_M. Keep w2
    # on the same M block contract unless the assignment path is redesigned.
    if stage2_config.get("BLOCK_SIZE_M") != stage1_config.get("BLOCK_SIZE_M"):
        stage2_config = stage2_config.copy()
        stage2_config["BLOCK_SIZE_M"] = stage1_config["BLOCK_SIZE_M"]
    return stage1_config, stage2_config


def _enable_moe_oot_without_flaggems() -> bool:
    value = os.environ.get("VLLM_FL_ENABLE_MOE_METHOD_OOT", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _enable_v626_moe_cfg() -> bool:
    value = os.environ.get("VLLM_FL_MOE_V626_CFG", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _enable_mcop_moe_sum() -> bool:
    value = os.environ.get("VLLM020_MCOP_MOE_SUM", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _moe_stage2_token_sum_max_m() -> int:
    try:
        return int(os.environ.get("VLLM020_MOE_STAGE2_TOKEN_SUM_MAX_M", "0"))
    except ValueError:
        return 0


def _moe_stage2_token_sum_block_n() -> int:
    try:
        return int(os.environ.get("VLLM020_MOE_STAGE2_TOKEN_SUM_BLOCK_N", "64"))
    except ValueError:
        return 64


def _moe_stage1_token_max_m() -> int:
    try:
        return int(os.environ.get("VLLM020_MOE_STAGE1_TOKEN_MAX_M", "0"))
    except ValueError:
        return 0


def _moe_stage1_token_block_n() -> int:
    try:
        return int(os.environ.get("VLLM020_MOE_STAGE1_TOKEN_BLOCK_N", "64"))
    except ValueError:
        return 64


def _enable_stage2_realign() -> bool:
    value = os.environ.get("VLLM020_MOE_STAGE2_REALIGN", "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def _force_stage2_realign() -> bool:
    value = os.environ.get("VLLM020_MOE_STAGE2_FORCE_REALIGN", "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def _moe_stage2_realign_min_m() -> int:
    try:
        return int(os.environ.get("VLLM020_MOE_STAGE2_REALIGN_MIN_M", "0"))
    except ValueError:
        return 0


def _env_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning_once("Ignore invalid %s=%s; expected int", name, value)
        return None


def _apply_stage2_gemm_override(
    stage2_config: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    env_to_key = {
        "VLLM020_MOE_STAGE2_BLOCK_M": "BLOCK_SIZE_M",
        "VLLM020_MOE_STAGE2_BLOCK_N": "BLOCK_SIZE_N",
        "VLLM020_MOE_STAGE2_BLOCK_K": "BLOCK_SIZE_K",
        "VLLM020_MOE_STAGE2_GROUP_M": "GROUP_SIZE_M",
        "VLLM020_MOE_STAGE2_SPLIT_K": "SPLIT_K",
        "VLLM020_MOE_STAGE2_WARPS": "num_warps",
        "VLLM020_MOE_STAGE2_STAGES": "num_stages",
    }
    updated = stage2_config.copy()
    changed = False
    for env_name, key in env_to_key.items():
        value = _env_int(env_name)
        if value is None:
            continue
        if value <= 0:
            logger.warning_once("Ignore invalid %s=%d; expected > 0", env_name, value)
            continue
        updated[key] = value
        changed = True
    if changed:
        updated = _limit_config_for_mcoplib(_sanitize_moe_config(updated))
        logger.info_once(
            "VLLM020 stage2 GEMM override enabled: %s",
            _cfg_signature(updated),
        )
    return updated, changed


def _moe_smallm_block_m_max_m() -> int:
    try:
        return int(os.environ.get("VLLM020_MOE_SMALLM_BLOCK_M_MAX_M", "0"))
    except ValueError:
        return 0


def _moe_smallm_block_m() -> int:
    try:
        return int(os.environ.get("VLLM020_MOE_SMALLM_BLOCK_M", "0"))
    except ValueError:
        return 0


def _apply_smallm_block_m_override(
    num_tokens: int,
    stage1_config: dict[str, Any],
    stage2_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    max_m = _moe_smallm_block_m_max_m()
    block_m = _moe_smallm_block_m()
    if max_m <= 0 or block_m <= 0 or int(num_tokens) > max_m:
        return stage1_config, stage2_config, False

    if block_m not in (4, 8, 16, 32, 64, 128):
        logger.warning_once(
            "Ignore invalid VLLM020_MOE_SMALLM_BLOCK_M=%s; expected one of "
            "4/8/16/32/64/128",
            block_m,
        )
        return stage1_config, stage2_config, False

    stage1_config = stage1_config.copy()
    stage2_config = stage2_config.copy()
    old_stage1_m = stage1_config.get("BLOCK_SIZE_M")
    old_stage2_m = stage2_config.get("BLOCK_SIZE_M")
    stage1_config["BLOCK_SIZE_M"] = block_m
    stage2_config["BLOCK_SIZE_M"] = block_m
    logger.info_once(
        "VLLM020_MOE_SMALLM_BLOCK_M override enabled: max_m=%d block_m=%d "
        "(stage1 %s -> %d, stage2 %s -> %d)",
        max_m,
        block_m,
        old_stage1_m,
        block_m,
        old_stage2_m,
        block_m,
    )
    return stage1_config, stage2_config, True


def _cfg_signature(config: dict[str, Any]) -> str:
    keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
            "num_warps", "num_stages")
    return "/".join(str(config.get(key, "-")) for key in keys)


def _moe_runtime_profile() -> str:
    return os.environ.get("VLLM020_MOE_RUNTIME_PROFILE", "").strip().lower()


def _trace_moe_runtime_config(
    num_tokens: int,
    stage1_config: dict[str, Any],
    stage2_config: dict[str, Any],
    selected_by_runtime_profile: bool,
) -> None:
    path = os.environ.get("VLLM020_MOE_RUNTIME_PROFILE_TRACE", "").strip()
    if not path:
        return

    key = (
        int(num_tokens),
        _cfg_signature(stage1_config),
        _cfg_signature(stage2_config),
        selected_by_runtime_profile,
    )
    if key in _MOE_RUNTIME_PROFILE_TRACE_LOGGED:
        return
    _MOE_RUNTIME_PROFILE_TRACE_LOGGED.add(key)

    max_records = int(os.environ.get("VLLM020_MOE_RUNTIME_PROFILE_TRACE_MAX", "128"))
    if len(_MOE_RUNTIME_PROFILE_TRACE_LOGGED) > max_records:
        return

    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "profile": _moe_runtime_profile(),
        "num_tokens": int(num_tokens),
        "selected_by_runtime_profile": selected_by_runtime_profile,
        "stage1": {k: stage1_config.get(k) for k in sorted(_MOE_CONFIG_KEYS)},
        "stage2": {k: stage2_config.get(k) for k in sorted(_MOE_CONFIG_KEYS)},
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        logger.warning_once("Failed to write MoE runtime profile trace %s: %s", path, exc)


def _select_runtime_moe_profile_config(
    num_tokens: int,
    stage1_config: dict[str, Any],
    stage2_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    profile = _moe_runtime_profile()
    if not profile:
        return stage1_config, stage2_config, False

    if profile in ("1k", "short1k"):
        enabled_buckets = {40, 48, 56, 64}
    elif profile in ("4k", "short4k", "decode"):
        enabled_buckets = set(_MOE_SMALL_DECODE_CONFIGS)
    else:
        logger.warning_once(
            "Unknown VLLM020_MOE_RUNTIME_PROFILE=%s; using default MoE config",
            profile,
        )
        return stage1_config, stage2_config, False

    if int(num_tokens) not in enabled_buckets:
        return stage1_config, stage2_config, False

    selected = _MOE_SMALL_DECODE_CONFIGS.get(int(num_tokens))
    if selected is None:
        return stage1_config, stage2_config, False

    tuned_stage1, tuned_stage2 = _split_staged_moe_config(selected)
    log_key = (profile, int(num_tokens))
    if log_key not in _MOE_RUNTIME_PROFILE_LOGGED:
        _MOE_RUNTIME_PROFILE_LOGGED.add(log_key)
        logger.info(
            "VLLM020_MOE_RUNTIME_PROFILE=%s selected M=%d stage1=%s stage2=%s",
            profile,
            int(num_tokens),
            _cfg_signature(tuned_stage1),
            _cfg_signature(tuned_stage2),
        )
    return tuned_stage1, tuned_stage2, True


def _apply_v626_moe_cfg(B: torch.Tensor, config: dict[str, Any]) -> dict[str, Any]:
    try:
        k_dim = B.shape[2]
    except Exception:
        return config
    if k_dim < 1024:
        return config
    tuned = config.copy()
    tuned["BLOCK_SIZE_N"] = 128
    tuned["BLOCK_SIZE_K"] = 64
    tuned["GROUP_SIZE_M"] = 1
    return tuned


def _v626_base_moe_cfg() -> dict[str, int]:
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 3,
    }


def _get_priority_backends(moe_config: FusedMoEConfig) -> list[UnquantizedMoeBackend]:
    """
    Get available backends in priority order based on platform and config.

    This function can be extended to become more complex as needed.
    """

    def _move_to_back(
        backends: list[UnquantizedMoeBackend],
        backend: UnquantizedMoeBackend,
    ) -> None:
        backends.append(backends.pop(backends.index(backend)))

    if current_platform.is_out_of_tree():
        _AVAILABLE_BACKENDS = [
            UnquantizedMoeBackend.TRITON,
            UnquantizedMoeBackend.BATCHED_TRITON,
        ]
    elif current_platform.is_rocm():
        _AVAILABLE_BACKENDS = [
            UnquantizedMoeBackend.AITER,
            UnquantizedMoeBackend.TRITON,
            UnquantizedMoeBackend.BATCHED_TRITON,
        ]
    elif current_platform.is_cuda():
        _AVAILABLE_BACKENDS = [
            UnquantizedMoeBackend.FLASHINFER_TRTLLM,
            UnquantizedMoeBackend.FLASHINFER_CUTLASS,
            UnquantizedMoeBackend.TRITON,
            UnquantizedMoeBackend.BATCHED_TRITON,
        ]

        # HACK: Qwen3.5 has crash with FLASHINFER_CUTLASS BF16 if DEP.
        # Updating the oracle querying logic is out of the scope of this
        # PR. Need to fix the kernel or update structure in follow up.
        if moe_config.moe_parallel_config.dp_size > 1:
            _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_CUTLASS)

    elif current_platform.is_xpu():
        _AVAILABLE_BACKENDS = [UnquantizedMoeBackend.XPU]
    elif current_platform.is_cpu():
        _AVAILABLE_BACKENDS = [UnquantizedMoeBackend.CPU]
    return _AVAILABLE_BACKENDS

## Adopt from select_unquantized_moe_backend
def select_unquantized_moe_backend_oot(moe_config: FusedMoEConfig,
) -> tuple[UnquantizedMoeBackend, type[mk.FusedMoEExperts] | None]:
    """
    Select the primary Unquantized MoE backend.
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    if current_platform.is_cpu():
        # TODO: migrate to MK structure.
        return UnquantizedMoeBackend.CPU, None

    if current_platform.is_tpu():
        return UnquantizedMoeBackend.TPU, None

    if current_platform.is_out_of_tree() and (
        use_flaggems() or _enable_moe_oot_without_flaggems()
    ):
        return UnquantizedMoeBackend.TRITON, TritonExpertsFL

    if moe_config.is_lora_enabled:
        return UnquantizedMoeBackend.TRITON, backend_to_kernel_cls(
            UnquantizedMoeBackend.TRITON
        )

    # NOTE: the kernels are selected in the following order.
    AVAILABLE_BACKENDS = _get_priority_backends(moe_config)

    # NOTE(rob): We need to peak into the P/F selection to determine
    # if we are using the batched or standard expert format, which
    # if not ideal. Once we unify TP + DP/EP, we can select P/F first.
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if moe_config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: UnquantizedMoeBackend) -> str:
        available_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} Unquantized MoE backend out "
            f"of potential backends: {available_strs}."
        )

    def _make_log_unsupported(
        backend: UnquantizedMoeBackend, reason: str | None
    ) -> str:
        if reason:
            return (
                f"Unquantized MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        return (
            f"Unquantized MoE backend '{backend.value}' does not support the "
            "deployment configuration."
        )

    def _return_or_raise(
        backend: UnquantizedMoeBackend,
        config: FusedMoEConfig,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[UnquantizedMoeBackend, type[mk.FusedMoEExperts] | None]:
        k_cls = backend_to_kernel_cls(backend)
        supported, reason = k_cls.is_supported_config(
            k_cls, config, None, None, activation_format
        )
        if supported:
            logger.info_once(_make_log_backend(backend))
            return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    runner_backend = moe_config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_unquantized_backend(runner_backend)
        if (
            activation_format == mk.FusedMoEActivationFormat.BatchedExperts
            and requested_backend == UnquantizedMoeBackend.TRITON
        ):
            requested_backend = UnquantizedMoeBackend.BATCHED_TRITON

        return _return_or_raise(requested_backend, moe_config, activation_format)

    # Handle explicit FlashInfer FP16 configuration.
    if envs.is_set("VLLM_USE_FLASHINFER_MOE_FP16"):
        if not envs.VLLM_USE_FLASHINFER_MOE_FP16:
            if UnquantizedMoeBackend.FLASHINFER_TRTLLM in AVAILABLE_BACKENDS:
                AVAILABLE_BACKENDS.remove(UnquantizedMoeBackend.FLASHINFER_TRTLLM)
            if UnquantizedMoeBackend.FLASHINFER_CUTLASS in AVAILABLE_BACKENDS:
                AVAILABLE_BACKENDS.remove(UnquantizedMoeBackend.FLASHINFER_CUTLASS)

        elif envs.is_set("VLLM_FLASHINFER_MOE_BACKEND"):
            # If user is explicit about backend, validate it.
            fi_backend = get_flashinfer_moe_backend()
            if fi_backend == FlashinferMoeBackend.CUTLASS:
                backend = UnquantizedMoeBackend.FLASHINFER_CUTLASS
            elif fi_backend == FlashinferMoeBackend.TENSORRT_LLM:
                backend = UnquantizedMoeBackend.FLASHINFER_TRTLLM
            else:
                raise ValueError(
                    f"FlashInfer MOE backend {fi_backend} "
                    "does not support unquantized MoE."
                )
            k_cls = backend_to_kernel_cls(backend)
            return _return_or_raise(backend, moe_config, activation_format)
        else:
            # If the user is not explicit about the backend, try both.
            for backend in [
                UnquantizedMoeBackend.FLASHINFER_TRTLLM,
                UnquantizedMoeBackend.FLASHINFER_CUTLASS,
            ]:
                k_cls = backend_to_kernel_cls(backend)
                supported, reason = k_cls.is_supported_config(
                    k_cls, moe_config, None, None, activation_format
                )
                if supported:
                    logger.info_once(_make_log_backend(backend))
                    return backend, k_cls
                else:
                    logger.debug_once(_make_log_unsupported(backend, reason))

            raise NotImplementedError(
                "Found VLLM_USE_FLASHINFER_MOE_FP16=1, but no "
                "FlashInfer unquantized MoE backend supports the configuration."
            )

    # Handle explicit AITER FP8 configuration.
    if envs.is_set("VLLM_ROCM_USE_AITER") or envs.is_set("VLLM_ROCM_USE_AITER_MOE"):
        if not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:
            if UnquantizedMoeBackend.AITER in AVAILABLE_BACKENDS:
                AVAILABLE_BACKENDS.remove(UnquantizedMoeBackend.AITER)
        else:
            backend = UnquantizedMoeBackend.AITER
            return _return_or_raise(backend, moe_config, activation_format)

    for backend in AVAILABLE_BACKENDS:
        k_cls = backend_to_kernel_cls(backend)
        supported, reason = k_cls.is_supported_config(
            k_cls, moe_config, None, None, activation_format
        )
        if supported:
            logger.info_once(_make_log_backend(backend))
            return backend, k_cls

        logger.debug_once(_make_log_unsupported(backend, reason))

    raise NotImplementedError(
        "No Unquantized MoE backend supports the deployment configuration."
    )

def _prepare_expert_assignment(
    topk_ids: torch.Tensor,
    config: dict[str, Any],
    num_tokens: int,
    top_k_num: int,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    *,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    block_shape: list[int] | None = None,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Prepare expert assignments for the aligned and low-latency Triton paths."""
    # SPARSITY_FACTOR is a heuristic margin ensuring tokens_in_chunk * top_k
    # activates only a small fraction of total experts
    # Skips moe_align_block_size and activates the `sorted_token_ids is None`
    # path of the fused_moe_kernel kernel
    naive_block_assignment = (
        expert_map is None
        and num_tokens * top_k_num * 4 <= global_num_experts
        and not (
            (use_int8_w8a16 or use_int4_w4a16)
            and block_shape is not None
            and block_shape[1] > 0
        )
    )

    if naive_block_assignment:
        return (
            None,
            topk_ids.view(-1),
            torch.full(
                (1,),
                topk_ids.numel() * config["BLOCK_SIZE_M"],
                dtype=torch.int32,
                device=topk_ids.device,
            ),
        )

    return call_op("moe_align_block_size",
        topk_ids,
        config["BLOCK_SIZE_M"],
        global_num_experts,
        expert_map,
        ignore_invalid_experts=ignore_invalid_experts,
    )

class TritonExpertsFL(TritonExperts):
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # Fast path (no LoRA, NVIDIA only): single fused FlagGems call.
        if self._lora_context is None and current_platform.is_cuda():
            import flag_gems

            output.copy_(flag_gems.fused_experts_impl(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                inplace=False,
                activation=activation.value,
                apply_router_weight_on_input=apply_router_weight_on_input,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                per_channel_quant=self.per_act_token_quant,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                w1_scale=self.w1_scale,
                w2_scale=self.w2_scale,
                a1_scale=a1q_scale,
                a2_scale=a2_scale,
                block_shape=self.block_shape,
                w1_bias=self.w1_bias,
                w2_bias=self.w2_bias,
            ))
            return

        # LoRA path: step-by-step pipeline (call_op dispatch) so LoRA
        # adapters can be injected between GEMM1/activation/GEMM2.
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )
        if _enable_v626_moe_cfg():
            base_config = _v626_base_moe_cfg()
            stage1_config = _apply_v626_moe_cfg(w1, base_config)
            stage2_config = base_config
            selected_by_runtime_profile = False
        else:
            stage1_config, stage2_config = _split_staged_moe_config(config)
            stage1_config, stage2_config, selected_by_runtime_profile = (
                _select_runtime_moe_profile_config(
                    num_tokens, stage1_config, stage2_config
                )
            )
        _trace_moe_runtime_config(
            num_tokens, stage1_config, stage2_config, selected_by_runtime_profile
        )
        stage1_config, stage2_config, smallm_block_m_override = (
            _apply_smallm_block_m_override(num_tokens, stage1_config, stage2_config)
        )
        if smallm_block_m_override:
            _trace_moe_runtime_config(
                num_tokens, stage1_config, stage2_config, True
            )
        stage2_config, stage2_override = _apply_stage2_gemm_override(stage2_config)
        if stage2_override:
            _trace_moe_runtime_config(
                num_tokens, stage1_config, stage2_config, True
            )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif (
            hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e4m3fnuz
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # Note that the output tensor might be in workspace1
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        cache2_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, cache2_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                stage1_config,
                num_tokens,
                top_k_num,
                global_num_experts,
                expert_map,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                block_shape=self.block_shape,
            )
        )
        sorted_token_ids_stage2 = sorted_token_ids
        expert_ids_stage2 = expert_ids
        num_tokens_post_padded_stage2 = num_tokens_post_padded
        stage2_block_m_differs = (
            stage2_config.get("BLOCK_SIZE_M") != stage1_config.get("BLOCK_SIZE_M")
        )
        stage2_realign_enabled = (
            _enable_stage2_realign()
            and num_tokens >= _moe_stage2_realign_min_m()
        )
        if (
            stage2_realign_enabled
            and (stage2_block_m_differs or _force_stage2_realign())
        ):
            (
                sorted_token_ids_stage2,
                expert_ids_stage2,
                num_tokens_post_padded_stage2,
            ) = _prepare_expert_assignment(
                topk_ids,
                stage2_config,
                num_tokens,
                top_k_num,
                global_num_experts,
                expert_map,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                block_shape=self.block_shape,
            )
        elif stage2_block_m_differs:
            stage2_config = stage2_config.copy()
            stage2_config["BLOCK_SIZE_M"] = stage1_config["BLOCK_SIZE_M"]

        # LoRA w13: applied to intermediate_cache1 before activation, using
        # hidden_states as the lora_a input.  moe_lora_align_block_size is
        # called once here and results reused for the w2 LoRA below.
        sorted_token_ids_lora = None
        expert_ids_lora = None
        num_tokens_post_padded_lora = None
        token_lora_mapping = None
        lora_context = self._lora_context

        stage1_token_used = False
        stage1_token_max_m = _moe_stage1_token_max_m()
        if (
            stage1_token_max_m > 0
            and num_tokens <= stage1_token_max_m
            and lora_context is None
            and expert_map is None
            and hidden_states.dtype == torch.bfloat16
            and w1.dtype == torch.bfloat16
            and intermediate_cache1.dtype == torch.bfloat16
            and a1q_scale is None
            and self.w1_scale is None
            and self.w1_bias is None
            and not self.quant_config.use_fp8_w8a8
            and not self.quant_config.use_int8_w8a8
            and not self.quant_config.use_int8_w8a16
            and not self.quant_config.use_int4_w4a16
        ):
            try:
                from mcoplib.triton_fused_moe import moe_stage1_token_triton

                moe_stage1_token_triton(
                    hidden_states,
                    w1,
                    intermediate_cache1,
                    topk_ids,
                    compute_type,
                    block_size_n=_moe_stage1_token_block_n(),
                    block_size_k=stage1_config.get("BLOCK_SIZE_K", 64),
                )
                stage1_token_used = True
            except Exception as err:
                logger.warning_once(
                    "mcoplib moe_stage1_token_triton failed, fallback to "
                    "stage1 fused_moe: %s",
                    err,
                )

        if not stage1_token_used:
            call_op("invoke_fused_moe_triton_kernel",
                hidden_states,
                w1,
                intermediate_cache1,
                a1q_scale,
                self.w1_scale,
                None,  # topk_weights
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,  # mul_routed_weights
                top_k_num,
                stage1_config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                per_channel_quant=self.per_act_token_quant,
                block_shape=self.block_shape,
                B_bias=self.w1_bias,
                _topk_ids=topk_ids,
            )

        if lora_context is not None:
            (
                sorted_token_ids_lora,
                expert_ids_lora,
                num_tokens_post_padded_lora,
                token_lora_mapping,
            ) = self.apply_w13_lora(
                lora_context,
                y=intermediate_cache1,
                x=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                expert_map=expert_map,
                w1=w1,
                w2=w2,
                num_tokens=num_tokens,
                top_k_num=top_k_num,
            )

        apply_moe_activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        a2q_scale: torch.Tensor | None = None

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
            quantization_emulation=self.quantization_emulation,
        )

        stage2_token_sum_used = False
        stage2_token_sum_max_m = _moe_stage2_token_sum_max_m()
        if (
            stage2_token_sum_max_m > 0
            and num_tokens <= stage2_token_sum_max_m
            and lora_context is None
            and topk_weights is not None
            and not apply_router_weight_on_input
            and qintermediate_cache2.dtype == torch.bfloat16
            and w2.dtype == torch.bfloat16
            and output.dtype == torch.bfloat16
            and a2q_scale is None
            and self.w2_scale is None
            and self.w2_bias is None
            and not self.quant_config.use_fp8_w8a8
            and not self.quant_config.use_int8_w8a8
            and not self.quant_config.use_int8_w8a16
            and not self.quant_config.use_int4_w4a16
        ):
            try:
                from mcoplib.triton_fused_moe import moe_stage2_token_sum_triton

                moe_stage2_token_sum_triton(
                    qintermediate_cache2,
                    w2,
                    output,
                    topk_weights,
                    topk_ids,
                    True,
                    compute_type,
                    block_size_n=_moe_stage2_token_sum_block_n(),
                    block_size_k=stage2_config.get("BLOCK_SIZE_K", 64),
                )
                stage2_token_sum_used = True
            except Exception as err:
                logger.warning_once(
                    "mcoplib moe_stage2_token_sum_triton failed, fallback to "
                    "stage2 fused_moe + moe_sum: %s",
                    err,
                )

        if not stage2_token_sum_used:
            call_op("invoke_fused_moe_triton_kernel",
                qintermediate_cache2,
                w2,
                intermediate_cache3,
                a2q_scale,
                self.w2_scale,
                topk_weights,
                sorted_token_ids_stage2,
                expert_ids_stage2,
                num_tokens_post_padded_stage2,
                not apply_router_weight_on_input,
                1,
                stage2_config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                per_channel_quant=self.per_act_token_quant,
                block_shape=self.block_shape,
                B_bias=self.w2_bias,
                _topk_ids=topk_ids,
            )

        # LoRA w2: applied to intermediate_cache3 before moe_sum, using the
        # unquantized intermediate_cache2 as the lora_a input.  Reuses the
        # sorted_token_ids_lora computed above.
        if lora_context is not None:
            self.apply_w2_lora(
                lora_context,
                y=intermediate_cache3,
                x=intermediate_cache2,
                topk_weights=topk_weights,
                sorted_token_ids_lora=sorted_token_ids_lora,
                expert_ids_lora=expert_ids_lora,
                num_tokens_post_padded_lora=num_tokens_post_padded_lora,
                token_lora_mapping=token_lora_mapping,
                num_tokens=num_tokens,
                w1=w1,
                w2=w2,
                top_k_num=top_k_num,
            )

        # separate function is required for MoE + LoRA
        if not stage2_token_sum_used:
            self.moe_sum(intermediate_cache3, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        if _enable_mcop_moe_sum():
            try:
                from mcoplib.triton_fused_moe import moe_sum_reduce_triton

                moe_sum_reduce_triton(input, output, 1.0)
                return
            except Exception as err:
                logger.warning_once(
                    "mcoplib moe_sum_reduce_triton failed, fallback to dispatch "
                    "moe_sum: %s",
                    err,
                )
        call_op("moe_sum", input, output)
