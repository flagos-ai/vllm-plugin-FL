# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Kimi K3 AttnRes PR737 provider, installed before model loading.

Set VLLM_FL_ATTN_RES_PROVIDER=1 on every worker to enable. The original vLLM
provider remains selected by default. An enabled, applicable provider fails
explicitly on dependency/kernel failures: it never silently runs native code.
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger("vllm.fl.attn_res_provider")
_ENV = "VLLM_FL_ATTN_RES_PROVIDER"
_BASE = "vllm.models.kimi_k3.nvidia"
_ARCHITECTURES = {"KimiK3ForConditionalGeneration", "KimiLinearForCausalLM"}
_HITS: dict[str, int] = {}
_CUSTOM_OP = None
_SCHEMA = (
    "(Tensor(a!) prefix, Tensor? delta, Tensor(b!) blocks, Tensor norm_weight, "
    "Tensor qk_weight, Tensor? output_norm_weight, int num_blocks, "
    "int block_write_idx, float eps, float output_norm_eps) -> Tensor"
)


def attn_res_provider_hits() -> dict[str, int]:
    """Host invocations, including capture; graph replays don't run Python."""
    return dict(_HITS)


def get_attn_res_triton_warmup_profiles(max_blocks: int):
    """Cover all five provider semantics, including native bypasses replaced here.

    vLLM's standard warmup imports this function at call time. The model's real
    dummy runs still warm each scheduled token shape before graph capture.
    """
    if max_blocks != 8:
        raise ValueError("PR737 requires exactly eight residual block slots")
    profiles = [(b, False, -1, True) for b in range(max_blocks + 1)]
    profiles += [(b, b != 0, b, True) for b in range(max_blocks)]
    profiles += [(b, True, -1, True) for b in range(max_blocks + 1)]
    profiles.append((max_blocks, True, -1, False))
    return tuple(profiles)


def _variant(delta, output_norm_weight, block_write_idx):
    if block_write_idx >= 0:
        return "block0_write" if delta is None else "write"
    if delta is None:
        return "post"
    return "final" if output_norm_weight is None else "common"


def _load_provider():
    from vllm_fl.third_party.attn_res_pr737.attn_res import attn_res

    return attn_res


def _create_dispatch(raw_provider, torch):
    """Give Dynamo a single opaque, accurately annotated mutation boundary."""
    global _CUSTOM_OP
    if _CUSTOM_OP is not None:
        return _CUSTOM_OP

    def execute(prefix, delta, blocks, norm_weight, qk_weight,
                output_norm_weight, num_blocks, block_write_idx, eps,
                output_norm_eps):
        # The provider validates every tensor's dtype/device/layout/alias and
        # enforces SM90 before launching. Do not catch failures or fall back.
        output = raw_provider(
            prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight,
            num_blocks, block_write_idx, eps, output_norm_eps,
        )
        variant = _variant(delta, output_norm_weight, block_write_idx)
        capture = torch.cuda.is_current_stream_capturing()
        key = f"{variant}:{'capture' if capture else 'eager'}"
        _HITS[key] = _HITS.get(key, 0) + 1
        if _HITS[key] == 1:
            logger.info(
                "FL_ATTN_RES_PR737_EXECUTED variant=%s tokens=%d blocks=%d "
                "capture=%s source=5b5329338f0275c021e647971e421330880c8981",
                variant, prefix.shape[0], num_blocks, capture,
            )
        return output

    op = torch.library.custom_op(
        "vllm_fl::attn_res_pr737", execute, mutates_args=("prefix", "blocks"),
        device_types="cuda", schema=_SCHEMA,
    )

    @op.register_fake
    def fake(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight,
             num_blocks, block_write_idx, eps, output_norm_eps):
        return prefix.new_empty(prefix.shape)

    _CUSTOM_OP = op
    return op


def patch_attn_res_provider(model_config, device) -> bool:
    """Call after selecting the worker GPU, before model construction/compile.

    Both the ops export and the model's from-import binding are replaced.
    Native warmup imports from the submodule at execution time and therefore
    uses the same custom op, with an expanded set of semantic profiles.
    """
    enabled = os.environ.get(_ENV, "0")
    if enabled not in {"0", "1"}:
        raise ValueError(f"{_ENV} must be 0 or 1, got {enabled!r}")
    if enabled == "0":
        return False
    if getattr(model_config, "architecture", None) not in _ARCHITECTURES:
        return False

    import torch

    config = getattr(model_config, "hf_text_config", None)
    block_size = getattr(config, "attn_res_block_size", None)
    layers = getattr(config, "num_hidden_layers", 0)
    max_blocks = (layers + block_size - 1) // block_size if block_size else 0
    reason = None
    if device.type != "cuda" or torch.version.hip is not None:
        reason = "requires CUDA"
    elif torch.cuda.get_device_capability(device) != (9, 0):
        reason = "requires SM90"
    elif model_config.dtype != torch.bfloat16:
        reason = "requires BF16"
    elif max_blocks != 8:
        reason = "requires eight residual block slots and AttnRes enabled"
    if reason is not None:
        logger.info("FL_ATTN_RES_PR737_NOT_APPLICABLE: %s", reason)
        return False

    provider = _create_dispatch(_load_provider(), torch)
    native = importlib.import_module(f"{_BASE}.ops.attn_res")
    ops = importlib.import_module(f"{_BASE}.ops")
    model = importlib.import_module(f"{_BASE}.model")
    if getattr(model.attn_res, "_fl_attn_res_pr737", False):
        return False
    # Keep callable metadata on the thin Python entry, not on CustomOpDef;
    # vLLM uses keyword arguments at these call sites.
    def dispatch(prefix, delta, blocks, norm_weight, qk_weight,
                 output_norm_weight, num_blocks, block_write_idx, eps,
                 output_norm_eps):
        return provider(prefix, delta, blocks, norm_weight, qk_weight,
                        output_norm_weight, num_blocks, block_write_idx, eps,
                        output_norm_eps)

    dispatch._fl_attn_res_pr737 = True
    dispatch._fl_native_attn_res = native.attn_res
    native.attn_res = dispatch
    native.get_attn_res_triton_warmup_profiles = get_attn_res_triton_warmup_profiles
    ops.attn_res = dispatch
    model.attn_res = dispatch
    logger.info(
        "FL_ATTN_RES_PR737_INSTALLED device=%s dtype=%s model=%s "
        "provider=vllm_fl::attn_res_pr737; execution requires EXECUTED evidence",
        device, model_config.dtype, model_config.architecture,
    )
    return True
