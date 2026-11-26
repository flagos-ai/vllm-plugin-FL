# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Vendor-neutral model topology walker and untied LM-head resolver.

Provides structural introspection over HuggingFace / vLLM causal decoder models
so load-time quantization (`apply_online_quant_transform`) can safely locate
eligible `(input_layernorm, qkv, o_proj, post_attention_layernorm, gate_up, down_proj)`
tuples and untied `ParallelLMHead` layers without hardcoding model class names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DecoderLayerLinearView:
    """Canonical structural handle for a single Pre-LN causal decoder block.

    Abstracts away whether QKV and MLP projections are fused (`qkv_proj`,
    `gate_up_proj`) or split (`q_proj/k_proj/v_proj`, `gate_proj/up_proj`) while
    exposing adjacent `RMSNorm` modules required for Rademacher sign absorption.
    """

    layer_idx: int
    block: nn.Module
    input_layernorm: nn.Module
    post_attention_layernorm: nn.Module
    self_attn: nn.Module
    mlp: nn.Module
    qkv_mods: tuple[tuple[str, nn.Module], ...]
    o_proj: nn.Module
    gate_up_mods: tuple[tuple[str, nn.Module], ...]
    is_fused_qkv: bool
    is_fused_gate_up: bool
    down_proj: nn.Module


def _locate_raw_decoder_blocks(model: nn.Module) -> list[nn.Module]:
    """Discover the ordered decoder layer list across common vLLM/HF wrapper layouts."""
    inner = getattr(model, "model", None)
    if inner is not None and isinstance(getattr(inner, "layers", None), (nn.ModuleList, list, tuple)):
        return list(inner.layers)
    if isinstance(getattr(model, "layers", None), (nn.ModuleList, list, tuple)):
        return list(model.layers)
    return [
        mod
        for _, mod in model.named_modules()
        if all(
            hasattr(mod, attr)
            for attr in ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp")
        )
    ]


def iter_causal_decoder_layers(
    model: nn.Module,
    *,
    require_vllm_parallel_linear: bool = False,
    require_unquantized: bool = False,
    div_n: int = 1,
    div_k: int = 1,
) -> list[DecoderLayerLinearView]:
    """Enumerate causal decoder layers satisfying structural, dtype, and tile-alignment rules.

    Filters candidate decoder blocks so only layers with 2D floating-point GPU
    weights divisible by `(div_n, div_k)` (e.g., 64x64 for block-64 FWHT and
    W8A8 GEMM tiles) and unquantized `UnquantizedLinearMethod` (when requested)
    are yielded for load-time transformation.
    """
    if require_vllm_parallel_linear or require_unquantized:
        from vllm.model_executor.layers.linear import (
            MergedColumnParallelLinear,
            QKVParallelLinear,
            RowParallelLinear,
            UnquantizedLinearMethod,
        )

    views: list[DecoderLayerLinearView] = []
    for block in _locate_raw_decoder_blocks(model):
        ln1 = getattr(block, "input_layernorm", None)
        ln2 = getattr(block, "post_attention_layernorm", None)
        attn = getattr(block, "self_attn", None)
        mlp = getattr(block, "mlp", None)
        if ln1 is None or ln2 is None or attn is None or mlp is None:
            continue

        o_proj = getattr(attn, "o_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if o_proj is None or down_proj is None:
            continue

        if hasattr(attn, "qkv_proj"):
            qkv_mods: tuple[tuple[str, nn.Module], ...] = (("qkv_proj", attn.qkv_proj),)
            is_fused_qkv = True
        elif all(hasattr(attn, k) for k in ("q_proj", "k_proj", "v_proj")):
            qkv_mods = (("q_proj", attn.q_proj), ("k_proj", attn.k_proj), ("v_proj", attn.v_proj))
            is_fused_qkv = False
        else:
            continue

        if hasattr(mlp, "gate_up_proj"):
            gate_up_mods: tuple[tuple[str, nn.Module], ...] = (("gate_up_proj", mlp.gate_up_proj),)
            is_fused_gate_up = True
        elif all(hasattr(mlp, k) for k in ("gate_proj", "up_proj")):
            gate_up_mods = (("gate_proj", mlp.gate_proj), ("up_proj", mlp.up_proj))
            is_fused_gate_up = False
        else:
            continue

        if require_vllm_parallel_linear:
            # Online W8A8 transformation requires fused TP=1 vLLM parallel linear layers.
            if not (is_fused_qkv and is_fused_gate_up):
                continue
            qkv_p, gu_p = qkv_mods[0][1], gate_up_mods[0][1]
            if not (
                isinstance(qkv_p, QKVParallelLinear)
                and isinstance(o_proj, RowParallelLinear)
                and isinstance(gu_p, MergedColumnParallelLinear)
                and isinstance(down_proj, RowParallelLinear)
            ):
                continue
            if any(getattr(p, "tp_size", 1) != 1 for p in (qkv_p, o_proj, gu_p, down_proj)):
                continue

        all_projs = [m for _, m in qkv_mods] + [o_proj] + [m for _, m in gate_up_mods] + [down_proj]
        valid = True
        for proj in all_projs:
            if require_unquantized and type(getattr(proj, "quant_method", None)) is not UnquantizedLinearMethod:
                valid = False
                break
            w = getattr(proj, "weight", None)
            if not isinstance(w, torch.Tensor) or w.ndim != 2:
                valid = False
                break
            if (require_vllm_parallel_linear or require_unquantized) and (
                not w.is_cuda or w.dtype not in (torch.bfloat16, torch.float16)
            ):
                valid = False
                break
            if (
                w.shape[0] < div_n
                or w.shape[1] < div_k
                or (w.shape[0] % div_n != 0)
                or (w.shape[1] % div_k != 0)
            ):
                valid = False
                break
        if not valid:
            continue

        views.append(
            DecoderLayerLinearView(
                layer_idx=len(views),
                block=block,
                input_layernorm=ln1,
                post_attention_layernorm=ln2,
                self_attn=attn,
                mlp=mlp,
                qkv_mods=qkv_mods,
                o_proj=o_proj,
                gate_up_mods=gate_up_mods,
                is_fused_qkv=is_fused_qkv,
                is_fused_gate_up=is_fused_gate_up,
                down_proj=down_proj,
            )
        )
    return views


def resolve_untied_lm_head(
    model: nn.Module,
    model_config: Any = None,
    *,
    require_parallel_lm_head: bool = False,
    div_n: int = 1,
    div_k: int = 1,
) -> nn.Module | None:
    """Return `model.lm_head` iff its weights are untied, GPU-resident, and tile-aligned.

    Enforces two independent untied-embedding checks before allowing in-place
    `lm_head` quantization:
    1. Configuration check: `tie_word_embeddings` must be False across all
       attached HF/vLLM config objects.
    2. Storage-aliasing check: `lm_head.weight.data_ptr()` must not alias any
       `*embed*` submodule's weight tensor, preventing accidental overwrite of
       the input token embedding table.
    """
    for cfg_obj in (
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
        getattr(model, "config", None),
    ):
        if cfg_obj is not None and getattr(cfg_obj, "tie_word_embeddings", False):
            return None

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return None

    if require_parallel_lm_head:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            UnquantizedEmbeddingMethod,
        )

        if not isinstance(lm_head, ParallelLMHead):
            return None
        if type(getattr(lm_head, "quant_method", None)) is not UnquantizedEmbeddingMethod:
            return None

    lm_w = getattr(lm_head, "weight", None)
    if (
        not isinstance(lm_w, torch.Tensor)
        or lm_w.ndim != 2
        or not lm_w.is_cuda
        or not lm_w.is_floating_point()
    ):
        return None
    if require_parallel_lm_head and lm_w.dtype not in (torch.bfloat16, torch.float16):
        return None
    if (
        lm_w.shape[0] < div_n
        or lm_w.shape[1] < div_k
        or (lm_w.shape[0] % div_n != 0)
        or (lm_w.shape[1] % div_k != 0)
    ):
        return None

    # Check 2: verify `lm_head.weight` does not physically alias any `*embed*` table.
    lm_ptr = lm_w.data_ptr()
    for name, submod in model.named_modules():
        if submod is lm_head or "embed" not in name.lower():
            continue
        ew = getattr(submod, "weight", None)
        if isinstance(ew, torch.Tensor) and ew.data_ptr() == lm_ptr:
            return None
    return lm_head


__all__ = [
    "DecoderLayerLinearView",
    "iter_causal_decoder_layers",
    "resolve_untied_lm_head",
]
