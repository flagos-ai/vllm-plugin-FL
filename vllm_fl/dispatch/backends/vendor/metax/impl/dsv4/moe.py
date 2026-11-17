# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# Minimal independent migration of the DEPLOYED W8A8 host path.
# Active math bodies are retained; no vendor Triton kernel implementation is copied.
import functools
import json
import os
from typing import Any
import torch
import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from flaggems_vllm import moe_align_block_size_no_tle as moe_align_block_size
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.fused_moe.fused_moe import logger
from vllm.platforms import current_platform
from vllm_fl import dsv4_config as _dsv4
from . import mctlass_ops


def get_config_file_name(
    E: int, N: int, dtype: str | None, block_shape: list[int] | None = None
) -> str:
    device_name = current_platform.get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    ).replace(" ", "")
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"  # noqa: E501


@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: str | None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> dict[int, Any] | None:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # Avoid optimizing for the batch invariant case. Use default config
    if envs.VLLM_BATCH_INVARIANT:
        return None

    # First look up if an optimized configuration is available in the configs
    # directory
    block_shape = [block_n, block_k] if block_n and block_k else None
    json_file_name = get_config_file_name(E, N, dtype, block_shape)

    config_file_paths = []

    # note that we prioritize user defined config
    # --------------------------------------------------------
    # Metax Note: Metax defined additional parameter:
    #       - H dimension
    # in the config file name. This is to address the scenario
    # where different models (with different H values) share
    # the same E and N values.
    #
    # The priority order for searching config in maca is:
    #   1. user_defined_config_folder of `configs/H={hidden_size}/`
    #   2. default config folder `configs/`
    # --------------------------------------------------------
    user_defined_config_folder = _dsv4.moe.tuned_config_folder
    logger.info_once("User defined config folder: %s", user_defined_config_folder)
    if user_defined_config_folder is not None:
        json_file_name_WITH_H_PREFIX = f"H={_dsv4.moe.hidden_size},{json_file_name}"
        user_defined_config_file_path = os.path.join(
            user_defined_config_folder, json_file_name_WITH_H_PREFIX
        )
        config_file_paths.append(user_defined_config_file_path)

    default_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    config_file_paths.append(default_config_file_path)

    for config_file_path in config_file_paths:
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                logger.info_once(
                    "Using configuration from %s for MoE layer.",
                    config_file_path,
                    scope="global",
                )
                # If a configuration has been found, return it
                tuned_config = json.load(f)
                # Delete triton_version from tuned_config
                tuned_config.pop("triton_version", None)
                return {int(key): val for key, val in tuned_config.items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning_once(
        "Using default MoE config. Performance might be sub-optimal! "
        "Config file not found at %s",
        ", ".join(config_file_paths),
        scope="local",
    )
    return None


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    if envs.VLLM_BATCH_INVARIANT:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "SPLIT_K": 1,
        }

    # num_stages can cause triton.runtime.errors.OutOfResources on Maca.
    num_stages_maca = 2

    if dtype == "fp8_w8a8" and block_shape is not None:
        # Block-wise quant: tile sizes are constrained by block_shape.
        # Use a small M tile for decode-like batches where tokens are
        # spread thin across experts. Larger batches benefit from
        # GROUP_SIZE_M > 1 because the per-block scales add memory
        # traffic that benefits from L2 tile reuse.
        config = {
            "BLOCK_SIZE_M": 16 if M <= 64 else 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 1 if M <= 16 else 32,
            "SPLIT_K": 1,
            "num_warps": 4,
            "num_stages": 3 if not current_platform.is_rocm() else num_stages_maca,
        }
    if dtype in ["int4_w4a16", "int8_w8a16"] and block_shape is not None:
        raise NotImplementedError("WNA16 is outside the DSV4 W8A8 adapter")
    else:
        if M <= 32:
            block_m = 16
        elif M <= 96:
            block_m = 32
        elif M <= 512:
            block_m = 64
        else:
            block_m = 128
        block_n = 64 if M <= 64 else 128
        block_k = 128 if dtype == "fp8_w8a8" or M <= 64 else 64
        tokens_per_expert = M // max(E, 1)
        group_m = 16 if tokens_per_expert > 128 else 1
        num_warps = 4 if M <= 128 else 8
        if current_platform.is_rocm():
            num_stages = num_stages_maca
        elif M <= 32:
            num_stages = 4
        else:
            num_stages = 3
        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "SPLIT_K": 1,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    dtype: str | None,
    M: int,
    block_shape: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    from vllm.model_executor.layers.fused_moe import get_config

    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        # TODO: This is a temporary fix for deepseek-r1-awq.
        # The N = N * 2 adjustment causes issues with Triton compilation for
        # this model. A proper fix should be investigated.
        # if dtype == "int4_w4a16":
        #     N = N * 2
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        configs = get_moe_configs(E, N, dtype, block_n, block_k)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype, block_shape)
    return config


def initialize_staged_config(
    config: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Metax Modification: prepare staged config for MACA
    stage1_config = config.get("stage1", config).copy()
    stage1_config.setdefault("SPLIT_K", 1)
    stage1_config.pop("ACCF32", None)

    stage2_config = config.get("stage2", config).copy()
    stage2_config.setdefault("SPLIT_K", 1)
    stage2_config.pop("ACCF32", None)
    return stage1_config, stage2_config


def dispatch_fused_moe_kernel(
    A,
    B,
    C,
    A_scale,
    B_scale,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight,
    top_k,
    config,
):
    """Invoke the W8A8 GEMM with the accepted routing and padding contract."""
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1
    assert B_scale is not None
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(EM, A.size(0) * top_k * config["BLOCK_SIZE_M"])
    mctlass_ops.cutlass_moe_mm_w8a8(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        EM,
        top_k,
        mul_routed_weight,
    )


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a8: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if not (use_int8_w8a8 and per_channel_quant and (activation == "silu")):
        raise NotImplementedError("DSV4 W8A8/SiLU only")
    if any((use_fp8_w8a8, use_int8_w8a16, use_int4_w4a8, use_int4_w4a16)):
        raise NotImplementedError(
            "other quantization formats are outside this migration"
        )
    if any(
        (
            x is not None
            for x in (
                ocp_mx_scheme,
                expert_map,
                w1_zp,
                w2_zp,
                a1_scale,
                a2_scale,
                block_shape,
                w1_bias,
                w2_bias,
            )
        )
    ):
        raise NotImplementedError("only current symmetric dynamic-token TP-only shape")
    if not _dsv4.moe.use_mctlass or not _dsv4.moe.python_api:
        raise NotImplementedError("MCTlass Python W8A8 path must remain enabled")
    if hidden_states.dtype != torch.bfloat16:
        raise NotImplementedError("only deployed BF16 activations validated")
    activation_enum = MoEActivation.from_str(activation)
    assert hidden_states.size(1) == w1.size(
        2
    ), f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens = hidden_states.size(0)
    (E, N, _) = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)
    CHUNK_SIZE = _dsv4.moe.chunk_size
    M = min(num_tokens, CHUNK_SIZE)
    config_dtype = "int8_w8a8"
    quant_dtype = torch.int8
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=None,
    )
    config = get_config_func(M)
    (stage1_config, stage2_config) = initialize_staged_config(config)
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)
    activation_out_dim = mk.FusedMoEExpertsModular.adjust_N_for_activation(
        N, activation_enum
    )
    intermediate_cache2 = torch.empty(
        (M * top_k_num, activation_out_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    out_hidden_states = hidden_states if inplace else torch.empty_like(hidden_states)
    for chunk in range(num_tokens // CHUNK_SIZE + 1):
        (begin_chunk_idx, end_chunk_idx) = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        (tokens_in_chunk, _) = curr_hidden_states.size()
        if tokens_in_chunk == 0:
            break
        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[
                : tokens_in_chunk * topk_ids.size(1)
            ]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)
        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        (qcurr_hidden_states, a1q_scale) = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=None,
            quant_dtype=quant_dtype,
            per_act_token_quant=True,
            block_shape=None,
            ocp_mx_scheme=None,
        )
        kernel_m = mctlass_ops.cutlass_moe_mm_w8a8_get_kernel_m(
            curr_hidden_states, w1, intermediate_cache1, top_k_num
        )
        assert kernel_m > 0, "cutlass_moe_w8a8 BLOCK_SIZE_M must greater than zero."
        stage1_config["BLOCK_SIZE_M"] = kernel_m
        stage2_config["BLOCK_SIZE_M"] = kernel_m
        (sorted_token_ids, expert_ids, num_tokens_post_padded) = moe_align_block_size(
            curr_topk_ids, stage1_config["BLOCK_SIZE_M"], global_num_experts, None
        )
        if stage1_config.get("SPLIT_K", 1) > 1:
            intermediate_cache1.zero_()
        dispatch_fused_moe_kernel(
            qcurr_hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            w1_scale,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k_num,
            stage1_config,
        )
        apply_moe_activation(
            activation_enum, intermediate_cache2, intermediate_cache1.view(-1, N)
        )
        (qintermediate_cache2, a2q_scale) = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=None,
            quant_dtype=quant_dtype,
            per_act_token_quant=True,
            block_shape=None,
            ocp_mx_scheme=None,
        )
        if stage2_config["BLOCK_SIZE_M"] != stage1_config["BLOCK_SIZE_M"]:
            (sorted_token_ids, expert_ids, num_tokens_post_padded) = (
                moe_align_block_size(
                    curr_topk_ids,
                    stage2_config["BLOCK_SIZE_M"],
                    global_num_experts,
                    None,
                )
            )
        if stage2_config.get("SPLIT_K", 1) > 1:
            intermediate_cache3.zero_()
        dispatch_fused_moe_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            w2_scale,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            stage2_config,
        )
        ops.moe_sum(
            intermediate_cache3.view(*intermediate_cache3.size()),
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
        )
    return out_hidden_states
