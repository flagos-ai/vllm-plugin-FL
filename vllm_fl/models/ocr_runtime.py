# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unlimited-OCR startup validation and token-to-cache metadata."""

import inspect
import os
import sys

import torch

from vllm_fl.models.unlimited_ocr import get_ocr_rswa_window, is_unlimited_ocr

_REQUIRED_OOT = frozenset(("rms_norm", "silu_and_mul", "rotary_embedding", "fused_moe"))


def validate_ocr_config(config, *, device_type, vendor):
    """Check the supported runtime before enabling any OCR behavior."""
    model = config.model_config
    if not is_unlimited_ocr(model):
        return False

    from vllm.config import CUDAGraphMode

    from vllm_fl.utils import (
        get_oot_blacklist,
        get_oot_whitelist,
        is_oot_enabled,
        use_flaggems,
    )

    if device_type != "npu" or vendor != "ascend":
        raise ValueError("Unlimited-OCR requires the FL Ascend NPU platform")
    if any(name.partition(".")[0] == "vllm_ascend" for name in sys.modules):
        raise RuntimeError("Unlimited-OCR requires VLLM_PLUGINS=fl")
    if config.use_v2_model_runner:
        raise ValueError("Unlimited-OCR requires the FL V1 runner")
    attention = config.attention_config
    if attention.backend is not None or getattr(attention, "use_non_causal", False):
        raise ValueError("Unlimited-OCR requires the FL causal R-SWA attention backend")
    if getattr(attention, "use_prefill_query_quantization", False):
        raise ValueError("Unlimited-OCR requires BF16 attention query inputs")
    for name in ("VLLM_FL_RSWA_GROUPED_MASK", "VLLM_FL_OCR_BATCH_NGRAM"):
        if os.getenv(name, "1") not in ("0", "1"):
            raise ValueError(f"{name} accepts 0 or 1 for Unlimited-OCR")
    if (
        config.speculative_config is not None
        or config.quant_config is not None
        or model.quantization is not None
        or model.dtype != torch.bfloat16
    ):
        raise ValueError(
            "Unlimited-OCR requires unquantized BF16 with no speculative decoding"
        )

    cache = config.cache_config
    if cache.cache_dtype not in ("auto", "bfloat16") or cache.calculate_kv_scales:
        raise ValueError("Unlimited-OCR requires an unquantized BF16 KV cache")
    if cache.enable_prefix_caching or cache.sliding_window is not None:
        raise ValueError(
            "Unlimited-OCR R-SWA requires full KV retention and prefix caching disabled"
        )

    window = get_ocr_rswa_window(model)
    if type(window) is not int or not 0 < window <= model.max_model_len:
        raise ValueError("Unlimited-OCR requires a valid R-SWA window")
    if (
        model.use_mla
        or getattr(model.hf_text_config, "topk_method", None) == "noaux_tc"
    ):
        raise ValueError("Unlimited-OCR requires MHA and ordinary expert routing")
    if model.use_fp64_gumbel:
        raise ValueError("Unlimited-OCR does not support FP64 Gumbel sampling on NPU")

    parallel = config.parallel_config
    if (
        parallel.tensor_parallel_size not in (1, 2)
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.enable_expert_parallel
        or parallel.use_ubatching
        or config.lora_config is not None
    ):
        raise ValueError("Unlimited-OCR supports ordinary TP1 or TP2 without LoRA")
    if config.scheduler_config.async_scheduling:
        raise ValueError("Unlimited-OCR requires synchronous scheduling")
    mode = config.compilation_config.cudagraph_mode
    if mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY):
        raise ValueError(
            "Unlimited-OCR supports eager or FULL_DECODE_ONLY graph execution"
        )
    if (
        mode == CUDAGraphMode.FULL_DECODE_ONLY
        and os.getenv("ASCEND_LAUNCH_BLOCKING") == "1"
    ):
        raise ValueError(
            "Unlimited-OCR graph execution requires ASCEND_LAUNCH_BLOCKING=0"
        )

    whitelist = get_oot_whitelist()
    if (
        use_flaggems()
        or not is_oot_enabled()
        or _REQUIRED_OOT.intersection(get_oot_blacklist() or ())
        or (whitelist is not None and not _REQUIRED_OOT.issubset(whitelist))
    ):
        raise ValueError("Unlimited-OCR requires FL OOT dispatch and USE_FLAGGEMS=0")
    return True


def enable_ocr_custom_ops(compilation_config):
    """Enable both upstream and concrete OOT registration names."""
    names = (
        "rms_norm",
        "RMSNorm",
        "silu_and_mul",
        "SiluAndMul",
        "rotary_embedding",
        "RotaryEmbedding",
    )
    custom_ops = compilation_config.custom_ops
    if any("-" + name in custom_ops for name in names):
        raise ValueError(
            "An operator required by Unlimited-OCR was explicitly disabled"
        )
    compilation_config.custom_ops = [
        *custom_ops,
        *("+" + name for name in names if "+" + name not in custom_ops),
    ]


def initialize_ocr_worker(config):
    """Validate and install OCR dispatch once before model construction."""
    if not is_unlimited_ocr(config.model_config):
        return False

    from vllm.platforms import current_platform
    from vllm.v1.sample.ops import topk_topp_sampler

    from vllm_fl.activation import (
        ActivationPlan,
        MoEDispatchDefaults,
        PendingPatch,
        activate,
        preflight_patches,
        validate_plan_capability,
    )
    from vllm_fl.dispatch import get_default_manager
    from vllm_fl.dispatch.ops import apply_top_k_top_p
    from vllm_fl.dispatch.policy import PolicyManager
    from vllm_fl.ops.custom_ops import register_ocr_oot_ops

    validate_ocr_config(
        config,
        device_type=current_platform.device_type,
        vendor=current_platform.vendor_name,
    )

    original = topk_topp_sampler.apply_top_k_top_p
    if original is not apply_top_k_top_p and (
        original.__module__ != topk_topp_sampler.__name__
        or tuple(inspect.signature(original).parameters) != ("logits", "k", "p")
    ):
        raise RuntimeError("Unsupported vLLM top-k/top-p sampler API")
    try:
        inspect.signature(topk_topp_sampler.apply_top_k_top_p_pytorch).bind(
            None, None, None, allow_cpu_sync=False
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Unsupported vLLM PyTorch sampling reference API") from exc

    binding = PendingPatch(
        target=f"{topk_topp_sampler.__name__}.apply_top_k_top_p",
        owner=topk_topp_sampler,
        attr="apply_top_k_top_p",
        replacement=apply_top_k_top_p,
        fingerprint="unlimited-ocr-ascend-v0.24",
        pristine=original,
        expected_params=("logits", "k", "p"),
    )
    # Check the locked upstream API and real operator availability before
    # installing the public OOT registrations and the worker sampler adapter.
    preflight_patches((binding,))
    vendor = ("impl:vendor.ascend",)
    reference = ("impl:reference.torch",)
    vendor_ops = (
        "attention_backend",
        "rms_norm",
        "silu_and_mul",
        "topk_softmax",
        "fused_experts",
    )
    ordered = (
        *((name, vendor) for name in vendor_ops),
        ("rotary_embedding", ("impl:vendor.ascend.torch_npu",)),
        ("scaled_dot_product_attention", vendor + reference),
        ("apply_top_k_top_p", reference),
    )
    required = (
        *((name, ("vendor:ascend",)) for name in (*vendor_ops, "rotary_embedding")),
        ("scaled_dot_product_attention", ("vendor:ascend", "reference")),
        ("apply_top_k_top_p", ("reference",)),
    )
    defaults = MoEDispatchDefaults(per_op_order=ordered, required_impls=required)

    def apply():
        register_ocr_oot_ops(_REQUIRED_OOT, binding)

    plan = ActivationPlan(
        name="Unlimited-OCR on Ascend",
        fingerprint=binding.fingerprint,
        apply=apply,
        moe_defaults=defaults,
    )
    policy_manager = PolicyManager.get_instance()
    policy = policy_manager.policy_for_plan(defaults)
    if not policy.strict:
        raise ValueError("Unlimited-OCR requires VLLM_FL_STRICT=1")
    manager = get_default_manager()
    with policy_manager.create_policy_context(policy):
        validate_plan_capability(
            plan, manager.resolve_candidates, policy.get_per_op_order
        )
        for op, expected in (
            ("attention_backend", "vendor.ascend"),
            ("rotary_embedding", "vendor.ascend.torch_npu"),
        ):
            ids = [impl.impl_id for impl in manager.resolve_candidates(op)]
            if ids != [expected]:
                raise ValueError(
                    f"Unexpected Unlimited-OCR {op} implementations: {ids}"
                )
    activate(plan)
    return True


def _supported_device(device):
    return device.type in ("npu", "privateuseone")


def compute_ocr_slot_mapping(block_table_manager, num_reqs, req_indices, positions):
    """Return False without writes when any cache group is unsupported.

    Request indices and positions must already satisfy the runner's indexing
    contract. No device values are read back or validated on the host.
    """
    if not isinstance(req_indices, torch.Tensor) or not isinstance(
        positions, torch.Tensor
    ):
        return False
    if (
        req_indices.ndim != 1
        or positions.ndim != 1
        or req_indices.shape != positions.shape
        or req_indices.dtype not in (torch.int32, torch.int64)
        or positions.dtype != torch.int64
        or req_indices.device != positions.device
        or not _supported_device(positions.device)
        or not isinstance(num_reqs, int)
        or num_reqs < 0
    ):
        return False
    tables = getattr(block_table_manager, "block_tables", None)
    if not isinstance(tables, (list, tuple)) or not tables:
        return False
    num_tokens = positions.numel()
    contracts = []
    for group in tables:
        mode = getattr(group, "slot_mapping_mode", None)
        # Older core BlockTable versions have only ordinary token slot mapping.
        if mode is not None and getattr(mode, "name", None) != "TOKEN_TO_KV_SLOT":
            return False
        block_size = getattr(group, "block_size", None)
        capacity = getattr(group, "max_num_batched_tokens", None)
        table = getattr(getattr(group, "block_table", None), "gpu", None)
        slots = getattr(getattr(group, "slot_mapping", None), "gpu", None)
        if (
            getattr(group, "pcp_world_size", None) != 1
            or getattr(group, "dcp_world_size", None) != 1
            or getattr(group, "blocks_per_kv_block", None) != 1
            or getattr(group, "use_hybrid_blocks", None) is not False
            or not isinstance(block_size, int)
            or block_size <= 0
            or getattr(group, "kv_cache_block_size", block_size) != block_size
            or not isinstance(capacity, int)
            or capacity < num_tokens
            or not isinstance(table, torch.Tensor)
            or not isinstance(slots, torch.Tensor)
        ):
            return False
        if (
            table.ndim != 2
            or table.shape[0] < num_reqs
            or table.stride(1) != 1
            or table.dtype not in (torch.int32, torch.int64)
            or table.device != positions.device
            or slots.ndim != 1
            or slots.numel() < capacity
            or slots.dtype != torch.int64
            or slots.device != positions.device
            or not slots.is_contiguous()
        ):
            return False
        contracts.append((table, slots, block_size, capacity))
    # Preflight all groups before touching any graph-visible output storage.
    for table, slots, block_size, capacity in contracts:
        if num_tokens:
            block_indices = positions // block_size
            physical_blocks = table[req_indices, block_indices].to(torch.int64)
            slots[:num_tokens].copy_(
                physical_blocks * block_size + positions % block_size
            )
        slots[num_tokens:capacity].fill_(-1)
    return True
