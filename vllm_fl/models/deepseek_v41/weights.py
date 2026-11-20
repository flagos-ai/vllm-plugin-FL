# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 BAAI. All rights reserved.
"""Exact text checkpoint mapping and complete local fusion/expert loading.

Adapted from vLLM e77daef89e18. Routing biases retain their original names;
the outer multimodal/draft loaders own tower and MTP keys. Unknown keys fail.
"""

import re

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import WeightsMapper, is_pp_missing_parameter

WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        "layers.": "model.layers.",
        "embed.": "model.embed_tokens.",
        "norm.": "model.norm.",
        "head.": "lm_head.",
        "vision.": None,
        "aligner.": None,
        "image_": None,
        "mtp.": None,
    },
    orig_to_new_regex={
        re.compile(r"(\.engram\.embed)(?=\.|$)"): r"\1_tokens",
        re.compile(r"\.scale$"): ".weight_scale",
    },
    orig_to_new_substr={".shared_experts.w2": ".shared_experts.down_proj"},
    orig_to_new_suffix={
        ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
    },
)

STACKED = (
    (".shared_experts.gate_up_proj.", ".shared_experts.w1.", 0),
    (".shared_experts.gate_up_proj.", ".shared_experts.w3.", 1),
    ("attn.fused_wqa_wkv", "attn.wq_a", 0),
    ("attn.fused_wqa_wkv", "attn.wkv", 1),
    ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
    ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
)


def _check_integer_storage(parameter, tensor, name):
    if tensor.dtype in (torch.int8, torch.int32) and parameter.dtype != tensor.dtype:
        raise TypeError(
            f"checkpoint integer storage mismatch: {name}: {tensor.dtype} -> {parameter.dtype}"
        )


def expert_mapping(model):
    return fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=model.config.n_routed_experts,
    )


def load_text_model_weights(model, weights, *, draft=False):
    from vllm_fl.dispatch import resolve_op
    from vllm_fl.quantization.modelslim import ModelSlimConfig

    quant = getattr(model, "quant_config", None)
    modelslim = quant if isinstance(quant, ModelSlimConfig) else None
    parameters = dict(model.named_parameters())
    loaded, seen, fused_parts, expert_parts = set(), set(), {}, {}
    mapping = expert_mapping(model)
    tp = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    for original_name, tensor in weights:
        if original_name in seen:
            raise ValueError(f"duplicate text checkpoint key: {original_name}")
        seen.add(original_name)
        if is_pp_missing_parameter(original_name, model):
            continue
        if modelslim is not None and modelslim.auxiliary(
            original_name, tensor, draft=draft
        ):
            continue
        repack = modelslim is not None and modelslim.is_packed_expert(
            original_name, draft=draft
        )
        name = original_name
        if repack:
            name = name.removesuffix(".weight") + ".weight_packed"
        if ".experts." in name:
            success = False
            for target, source, expert, shard in mapping:
                if source not in name:
                    continue
                destination = name.replace(source, target)
                if is_pp_missing_parameter(destination, model):
                    continue
                if destination not in parameters:
                    raise KeyError(f"unknown expert parameter: {destination}")
                param = parameters[destination]
                owner = model.get_submodule(destination.rsplit(".", 1)[0])
                if owner._map_global_expert_id_to_local_expert_id(expert) < 0:
                    success = True
                    break
                if repack:
                    tensor = resolve_op("repack_int4_n_to_k")(tensor)
                _check_integer_storage(param, tensor, original_name)
                success = param.weight_loader(
                    param,
                    tensor,
                    destination,
                    shard_id=shard,
                    expert_id=expert,
                    return_success=True,
                )
                if success:
                    loaded.add(destination)
                    expert_parts.setdefault(destination, set()).add((expert, shard))
                    break
            if not success:
                raise ValueError(
                    f"expert checkpoint key was not consumed: {original_name}"
                )
            continue
        for target, source, shard in STACKED:
            if source not in name:
                continue
            destination = name.replace(source, target)
            if destination not in parameters:
                raise KeyError(f"unknown fused text parameter: {destination}")
            param = parameters[destination]
            _check_integer_storage(param, tensor, original_name)
            param.weight_loader(param, tensor, shard)
            loaded.add(destination)
            fused_parts.setdefault(destination, set()).add(shard)
            break
        else:
            if name not in parameters:
                raise KeyError(f"unknown text checkpoint parameter: {name}")
            param = parameters[name]
            _check_integer_storage(param, tensor, original_name)
            if name.endswith(".attn_sink"):
                expected = (model.config.num_attention_heads,)
                if tuple(tensor.shape) != expected:
                    raise ValueError("attention sink checkpoint shape mismatch")
                heads = model.config.num_attention_heads // tp
                param.data[:heads].copy_(tensor[rank * heads : (rank + 1) * heads])
            else:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, tensor)
            loaded.add(name)
    missing = set(parameters) - loaded
    if missing:
        raise ValueError(f"missing local text parameters: {sorted(missing)}")
    for name, parts in fused_parts.items():
        required = {0, 1}
        if ".compressor.fused_wkv_wgate." in name:
            layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
            required = set(range(model.config.compress_ratios[layer]))
        if parts != required:
            raise ValueError(
                f"incomplete checkpoint fusion: {name}: {parts} != {required}"
            )
    for name, parts in expert_parts.items():
        shards = ("w1", "w3") if ".w13_" in name else ("w2",)
        required = {
            (expert, shard)
            for expert in range(model.config.n_routed_experts)
            for shard in shards
            if model.get_submodule(
                name.rsplit(".", 1)[0]
            )._map_global_expert_id_to_local_expert_id(expert)
            >= 0
        }
        if parts != required:
            raise ValueError(f"incomplete local expert slices: {name}")
    return loaded
