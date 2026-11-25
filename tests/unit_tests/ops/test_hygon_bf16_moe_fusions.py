# Copyright (c) 2026 BAAI. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from vllm_fl.dispatch.backends.vendor.hygon.impl.moe.bf16_moe_gemm2 import (
    HygonBf16MoeGemm2Config,
)
from vllm_fl.dispatch.backends.vendor.hygon.impl.moe.bf16_moe_fusions import (
    invoke_hygon_bf16_gemm1_silu,
    supports_hygon_bf16_gemm1_silu,
    supports_hygon_fixed_topk8_reduce,
)
from vllm_fl.dispatch.backends.vendor.hygon.impl.moe.stage_config import (
    requires_separate_expert_assignment,
    resolve_moe_stage_configs,
)


GEMM1_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 1,
}


def test_hygon_moe_stage_config_supports_independent_tiles():
    config = {
        "shared": {"GROUP_SIZE_M": 1},
        "gemm1": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "num_warps": 4,
            "num_stages": 1,
        },
        "gemm2": {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "num_warps": 8,
            "num_stages": 1,
        },
    }

    gemm1, gemm2 = resolve_moe_stage_configs(config)

    assert gemm1["BLOCK_SIZE_M"] == 128
    assert gemm2["BLOCK_SIZE_M"] == 16
    assert requires_separate_expert_assignment(gemm1, gemm2)


def test_hygon_moe_stage_config_keeps_legacy_flat_config():
    gemm1, gemm2 = resolve_moe_stage_configs(GEMM1_CONFIG)

    assert gemm1 is gemm2
    assert not requires_separate_expert_assignment(gemm1, gemm2)


def test_hygon_bf16_gemm2_reads_stage_config():
    route_config = {
        "BLOCK_SIZE_M": 8,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 2,
        "num_warps": 4,
        "num_stages": 2,
        "workers_per_cu": 2,
    }

    assert HygonBf16MoeGemm2Config.for_small_decode(route_config) == (
        HygonBf16MoeGemm2Config(
            block_m=8,
            block_n=64,
            block_k=32,
            num_warps=4,
            num_stages=2,
            workers_per_cu=2,
        )
    )


def test_hygon_bf16_gemm2_defaults_optional_workers_per_cu():
    route_config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 1,
    }

    config = HygonBf16MoeGemm2Config.for_small_decode(route_config)

    assert config.block_n == 256
    assert config.block_k == 64
    assert config.workers_per_cu == 4


def _gemm1_inputs(
    *,
    tokens: int = 8192,
    top_k: int = 8,
    experts: int = 256,
    hidden_size: int = 2048,
    intermediate_size: int = 256,
):
    activation = torch.empty(
        (tokens, hidden_size),
        device="meta",
        dtype=torch.bfloat16,
    )
    weight = torch.empty(
        (experts, 2 * intermediate_size, hidden_size),
        device="meta",
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (tokens * top_k, intermediate_size),
        device="meta",
        dtype=torch.bfloat16,
    )
    sorted_token_ids = torch.empty(
        (tokens * top_k + experts * 127,),
        device="meta",
        dtype=torch.int32,
    )
    expert_ids = torch.empty(
        (sorted_token_ids.numel() // 128 + 1,),
        device="meta",
        dtype=torch.int32,
    )
    return activation, weight, output, sorted_token_ids, expert_ids


def _supports_gemm1(input_overrides=None, **overrides) -> bool:
    input_overrides = input_overrides or {}
    activation, weight, output, sorted_token_ids, expert_ids = _gemm1_inputs(
        **input_overrides
    )
    arguments = {
        "top_k": input_overrides.get("top_k", 8),
        "activation_name": "silu",
        "apply_router_weight_on_input": False,
        "has_quantization": False,
        "has_bias": False,
        "has_expert_map": False,
        "has_lora": False,
    }
    arguments.update(overrides)
    return supports_hygon_bf16_gemm1_silu(
        activation,
        weight,
        output,
        sorted_token_ids,
        expert_ids,
        GEMM1_CONFIG,
        **arguments,
    )


def test_hygon_bf16_gemm1_silu_accepts_original_shape():
    assert _supports_gemm1()


def test_hygon_bf16_gemm1_silu_accepts_dynamic_shapes():
    for shape in (
        {
            "tokens": 64,
            "top_k": 4,
            "experts": 64,
            "hidden_size": 1024,
            "intermediate_size": 128,
        },
        {
            "tokens": 257,
            "top_k": 2,
            "experts": 17,
            "hidden_size": 1000,
            "intermediate_size": 190,
        },
        {
            "tokens": 16384,
            "top_k": 8,
            "experts": 512,
            "hidden_size": 4096,
            "intermediate_size": 512,
        },
    ):
        assert _supports_gemm1(input_overrides=shape)


def test_hygon_bf16_gemm1_silu_rejects_unsupported_semantics():
    for override in (
        {"activation_name": "gelu"},
        {"apply_router_weight_on_input": True},
        {"has_quantization": True},
        {"has_bias": True},
        {"has_expert_map": True},
        {"has_lora": True},
    ):
        assert not _supports_gemm1(**override)


@pytest.mark.parametrize("missing_assignment", ("sorted_token_ids", "expert_ids"))
def test_hygon_bf16_gemm1_silu_rejects_missing_expert_assignment(
    missing_assignment,
):
    activation, weight, output, sorted_token_ids, expert_ids = _gemm1_inputs()
    if missing_assignment == "sorted_token_ids":
        sorted_token_ids = None
    else:
        expert_ids = None

    assert not supports_hygon_bf16_gemm1_silu(
        activation,
        weight,
        output,
        sorted_token_ids,
        expert_ids,
        GEMM1_CONFIG,
        top_k=8,
        activation_name="silu",
        apply_router_weight_on_input=False,
        has_quantization=False,
        has_bias=False,
        has_expert_map=False,
        has_lora=False,
    )


def test_hygon_bf16_gemm1_silu_rejects_incompatible_shapes():
    activation, weight, output, sorted_token_ids, expert_ids = _gemm1_inputs()

    assert not supports_hygon_bf16_gemm1_silu(
        activation,
        weight[:, :-1, :],
        output,
        sorted_token_ids,
        expert_ids,
        GEMM1_CONFIG,
        top_k=8,
        activation_name="silu",
        apply_router_weight_on_input=False,
        has_quantization=False,
        has_bias=False,
        has_expert_map=False,
        has_lora=False,
    )
    assert not supports_hygon_bf16_gemm1_silu(
        activation,
        weight,
        output[:-1],
        sorted_token_ids,
        expert_ids,
        GEMM1_CONFIG,
        top_k=8,
        activation_name="silu",
        apply_router_weight_on_input=False,
        has_quantization=False,
        has_bias=False,
        has_expert_map=False,
        has_lora=False,
    )


def test_hygon_bf16_gemm1_silu_rejects_other_config():
    activation, weight, output, sorted_token_ids, expert_ids = _gemm1_inputs()
    config = dict(GEMM1_CONFIG)
    config["GROUP_SIZE_M"] = 3
    assert not supports_hygon_bf16_gemm1_silu(
        activation,
        weight,
        output,
        sorted_token_ids,
        expert_ids,
        config,
        top_k=8,
        activation_name="silu",
        apply_router_weight_on_input=False,
        has_quantization=False,
        has_bias=False,
        has_expert_map=False,
        has_lora=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    "tokens,top_k,experts,hidden_size,intermediate_size",
    (
        (17, 4, 7, 250, 70),
        (129, 2, 5, 192, 96),
    ),
)
def test_hygon_bf16_gemm1_silu_dynamic_shape_correctness(
    tokens,
    top_k,
    experts,
    hidden_size,
    intermediate_size,
):
    torch.manual_seed(7)
    device = torch.device("cuda")
    activation = torch.randn(
        (tokens, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        (experts, 2 * intermediate_size, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )
    topk_ids = torch.randint(
        experts,
        (tokens, top_k),
        device=device,
        dtype=torch.int64,
    )
    flat_expert_ids = topk_ids.flatten()
    num_routes = flat_expert_ids.numel()

    sorted_routes = []
    block_experts = []
    flat_expert_ids_cpu = flat_expert_ids.cpu()
    for expert in range(experts):
        routes = torch.nonzero(
            flat_expert_ids_cpu == expert,
            as_tuple=False,
        ).flatten()
        if routes.numel() == 0:
            continue
        padded_routes = (
            (routes.numel() + GEMM1_CONFIG["BLOCK_SIZE_M"] - 1)
            // GEMM1_CONFIG["BLOCK_SIZE_M"]
            * GEMM1_CONFIG["BLOCK_SIZE_M"]
        )
        padding = torch.full(
            (padded_routes - routes.numel(),),
            num_routes,
            dtype=torch.int64,
        )
        sorted_routes.append(torch.cat((routes, padding)))
        block_experts.extend([expert] * (padded_routes // GEMM1_CONFIG["BLOCK_SIZE_M"]))

    sorted_token_ids = torch.cat(sorted_routes).to(
        device=device,
        dtype=torch.int32,
    )
    expert_ids = torch.tensor(
        block_experts,
        device=device,
        dtype=torch.int32,
    )
    num_tokens_post_padded = torch.tensor(
        [sorted_token_ids.numel()],
        device=device,
        dtype=torch.int32,
    )
    output = torch.empty(
        (num_routes, intermediate_size),
        device=device,
        dtype=torch.bfloat16,
    )

    invoke_hygon_bf16_gemm1_silu(
        activation,
        weight,
        output,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k=top_k,
    )

    routed_activation = activation.repeat_interleave(top_k, dim=0)
    routed_weight = weight[flat_expert_ids]
    projected = torch.bmm(
        routed_activation.unsqueeze(1),
        routed_weight.transpose(1, 2),
    ).squeeze(1)
    gate, up = projected.chunk(2, dim=-1)
    expected = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_hygon_fixed_topk8_reduce_accepts_validated_shape():
    input_tensor = torch.empty(
        (8192, 8, 2048),
        device="meta",
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (8192, 2048),
        device="meta",
        dtype=torch.bfloat16,
    )
    assert supports_hygon_fixed_topk8_reduce(input_tensor, output)


def test_hygon_fixed_topk8_reduce_rejects_decode_shape():
    input_tensor = torch.empty(
        (64, 8, 2048),
        device="meta",
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (64, 2048),
        device="meta",
        dtype=torch.bfloat16,
    )
    assert not supports_hygon_fixed_topk8_reduce(input_tensor, output)
