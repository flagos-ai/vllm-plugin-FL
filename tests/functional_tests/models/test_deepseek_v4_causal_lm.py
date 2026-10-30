# SPDX-License-Identifier: Apache-2.0
"""Small actual CausalLM and scheduler lifecycle, without checkpoint downloads."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(current_platform.device_type != "npu", reason="Requires Ascend"),
]


@pytest.fixture(autouse=True)
def strict_dispatch():
    from vllm_fl.dispatch import get_default_manager, with_strict_mode

    # Quantized native providers must not silently fall back. The ordinary
    # DeepSeek reference-only math keeps its selected reference implementation.
    with with_strict_mode():
        manager = get_default_manager()
        for name in (
            "modelslim_w8a8_prepare_weight",
            "modelslim_w8a8_dynamic_linear",
            "modelslim_w8a8_prepare_experts",
            "modelslim_w8a8_experts",
            "deepseek_v4_index_quant",
        ):
            assert manager.get_selected_impl_id(name) == "vendor.ascend"
        yield


def _hf_config():
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=128,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        qk_rope_head_dim=4,
        sliding_window=8,
        rms_norm_eps=1e-6,
        compress_ratios=[0, 4, 128],
        index_head_dim=8,
        index_n_heads=2,
        index_topk=2,
        moe_intermediate_size=64,
        intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        routed_scaling_factor=1.5,
        num_hash_layers=1,
        hc_mult=4,
        hc_sinkhorn_iters=4,
        hc_eps=1e-4,
        hidden_act="silu",
        swiglu_limit=10.0,
        max_position_embeddings=256,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_scaling={
            "factor": 1.0,
            "original_max_position_embeddings": 256,
            "beta_fast": 32,
            "beta_slow": 1,
            "type": "yarn",
        },
        tie_word_embeddings=False,
        num_nextn_predict_layers=0,
    )


def _description():
    result = {"embed.weight": "FLOAT", "head.weight": "FLOAT", "norm.weight": "FLOAT"}
    for layer, ratio in enumerate((0, 4, 128)):
        prefix = f"layers.{layer}.attn."
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            result[prefix + name + ".weight"] = (
                "W8A8_DYNAMIC" if name in ("wq_a", "wq_b", "wkv") else "FLOAT"
            )
        if ratio:
            for name in ("wkv", "wgate"):
                result[prefix + "compressor." + name + ".weight"] = "FLOAT"
        if ratio == 4:
            result[prefix + "indexer.wq_b.weight"] = "W8A8_DYNAMIC"
            result[prefix + "indexer.weights_proj.weight"] = "FLOAT"
            for name in ("wkv", "wgate"):
                result[prefix + "indexer.compressor." + name + ".weight"] = "FLOAT"
        prefix = f"layers.{layer}.ffn."
        result[prefix + "gate.weight"] = "FLOAT"
        for projection in ("w1", "w2", "w3"):
            result[prefix + f"shared_experts.{projection}.weight"] = "W8A8_DYNAMIC"
            for expert in range(4):
                result[prefix + f"experts.{expert}.{projection}.weight"] = (
                    "W8A8_DYNAMIC"
                )
    return result


def _runtime_config(pipeline_parallel_size=1):
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        CompilationMode,
        DeviceConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    from vllm_fl.quantization.w8a8.modelslim import AscendModelSlimConfig

    config = VllmConfig(
        device_config=DeviceConfig(device="npu"),
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, custom_ops=["all"]
        ),
        parallel_config=ParallelConfig(pipeline_parallel_size=pipeline_parallel_size),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=256,
            max_num_batched_tokens=256,
            max_num_seqs=2,
        ),
        cache_config=CacheConfig(block_size=128, enable_prefix_caching=False),
    )
    # Only the attributes read by the real component chain are supplied; rank
    # getters and group creation always use the real distributed runtime.
    hf_config = _hf_config()
    config.model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
        dtype=torch.bfloat16,
        max_model_len=256,
        is_moe=True,
        enforce_eager=False,
    )
    config.quant_config = AscendModelSlimConfig(_description(), weight_nz_mode=0)
    return config


@pytest.fixture(scope="module")
def runtime():
    from tests.utils.device_utils import ascend_single_rank_runtime

    with ascend_single_rank_runtime(_runtime_config, workspace=True) as config:
        yield config


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="Requires isolated torchrun with two Ascend processes for PP2",
)
def test_pp_hc_head_checkpoint_loading():
    """Native root HC-head parameters load on both real pipeline ranks."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_pp_group,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    from vllm_fl import _patch_ascend_torch_accelerator
    from vllm_fl.models.deepseek_v4 import DeepseekV4ForCausalLM

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(local_rank)
    _patch_ascend_torch_accelerator()
    config = _runtime_config(pipeline_parallel_size=2)
    with set_current_vllm_config(config), torch.inference_mode():
        model = None
        try:
            init_distributed_environment(
                world_size=2, rank=rank, local_rank=local_rank, backend="hccl"
            )
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=2,
                backend="hccl",
            )
            init_workspace_manager(device)
            assert get_pp_group().world_size == 2
            assert get_pp_group().rank_in_group == rank
            assert get_tp_group().world_size == 1
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                model = initialize_model(config, model_class=DeepseekV4ForCausalLM).to(
                    device
                )
            finally:
                torch.set_default_dtype(previous_dtype)
            checkpoint = []
            expected = {}
            params = dict(model.named_parameters())
            hf_config = config.model_config.hf_config
            head_shapes = (
                ("fn", (hf_config.hc_mult, hf_config.hc_mult * hf_config.hidden_size)),
                ("base", (hf_config.hc_mult,)),
                ("scale", (1,)),
            )
            for index, (suffix, shape) in enumerate(head_shapes):
                raw_name = f"hc_head_{suffix}"
                size = 1
                for dimension in shape:
                    size *= dimension
                value = (
                    torch.arange(size, device=device, dtype=torch.float32)
                    .reshape(shape)
                    .mul_(0.001)
                    .add_(index + 0.25)
                )
                if f"model.{raw_name}" in params:
                    params[f"model.{raw_name}"].fill_(float("nan"))
                checkpoint.append((raw_name, value))
                expected[f"model.{raw_name}"] = value
            loaded = model.load_weights(iter(checkpoint))
            assert loaded == set(expected)
            for name, value in expected.items():
                assert params[name].dtype == torch.float32
                assert params[name].device == device
                torch.testing.assert_close(params[name], value, rtol=0, atol=0)
            print(
                f"PP_HC_HEAD_LOAD_OK rank={rank} pp_rank={get_pp_group().rank_in_group} "
                f"last={get_pp_group().is_last_rank} loaded={sorted(loaded)}",
                flush=True,
            )
        finally:
            del model
            try:
                reset_workspace_manager()
            finally:
                try:
                    destroy_model_parallel()
                finally:
                    destroy_distributed_environment()


def _value(name, param, index):
    values = (
        torch.arange(param.numel(), device="npu").reshape(param.shape) + index * 13
    ) % 47 - 23
    if param.dtype in (torch.int32, torch.int64) or name.endswith("tid2eid"):
        # Each vocab entry has a distinct deterministic global expert pair.
        return (torch.arange(param.numel(), device="npu").reshape(param.shape) % 4).to(
            param.dtype
        )
    if param.dtype == torch.int8:
        return values.to(torch.int8)
    if name.endswith("weight_offset"):
        return torch.zeros_like(param)
    if name.endswith("weight_scale"):
        return torch.full_like(param, 0.015625)
    if "norm.weight" in name:
        return (values.float().abs() * 0.01 + 0.8).to(param.dtype)
    if "hc_" in name and name.endswith("scale"):
        return torch.full_like(param, 0.3)
    if name.endswith("attn_sink"):
        return torch.full_like(param, -0.3)
    factor = 0.002 if "hc_" in name else 0.005
    return (values.float() * factor).to(param.dtype)


def _native_name(name):
    if name == "model.embed_tokens.weight":
        return "embed.weight"
    if name == "lm_head.weight":
        return "head.weight"
    if name == "model.norm.weight":
        return "norm.weight"
    if name.startswith("model."):
        name = name[6:]
    return (
        (name + ".")
        .replace(".self_attn.", ".attn.")
        .replace(".mlp.", ".ffn.")
        .replace(".post_attention_layernorm.", ".ffn_norm.")
        .replace(".input_layernorm.", ".attn_norm.")
        .replace(".gate.e_score_correction_bias", ".gate.bias")
    )[:-1]


def _native_checkpoint(model):
    checkpoint = []
    expected = {}
    tensors = list(model.named_parameters())
    tensors += [
        (name, value)
        for name, value in model.named_buffers()
        if name.endswith("tid2eid") and name not in dict(tensors)
    ]
    for index, (name, param) in enumerate(tensors):
        value = _value(name, param, index)
        expected[name] = value.clone()
        if ".experts.routed_experts." in name:
            base, suffix = name.split(".experts.routed_experts.", 1)
            base = _native_name(base) + ".experts."
            for expert in range(value.shape[0]):
                if suffix.startswith("w13_"):
                    tail = suffix[len("w13_") :]
                    middle = value.shape[1] // 2
                    checkpoint.extend(
                        [
                            (base + f"{expert}.w1." + tail, value[expert, :middle]),
                            (base + f"{expert}.w3." + tail, value[expert, middle:]),
                        ]
                    )
                elif suffix.startswith("w2_"):
                    checkpoint.append(
                        (base + f"{expert}.w2." + suffix[len("w2_") :], value[expert])
                    )
                else:
                    raise AssertionError(f"Unexpected expert parameter {name}")
        elif ".gate_up_proj." in name:
            base, suffix = name.split(".gate_up_proj.")
            middle = value.shape[0] // 2
            checkpoint.extend(
                [
                    (_native_name(base) + ".w1." + suffix, value[:middle]),
                    (_native_name(base) + ".w3." + suffix, value[middle:]),
                ]
            )
        elif ".shared_experts.down_proj." in name:
            checkpoint.append(
                (_native_name(name).replace(".down_proj.", ".w2."), value)
            )
        else:
            checkpoint.append((_native_name(name), value))
    return checkpoint, expected


@pytest.fixture
def model_env(runtime):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_groups,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
    from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache

    from vllm_fl.models.deepseek_v4 import DeepseekV4ForCausalLM
    from vllm_fl.worker.kv_cache_utils import bind_kv_cache
    from vllm_fl.worker.model_runner import ModelRunnerFL

    context = runtime.compilation_config.static_forward_context
    original_names = set(context)
    original_moe_layers = list(runtime.compilation_config.static_all_moe_layers)
    request_list = []
    manager = None
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        try:
            from vllm.model_executor.model_loader.utils import initialize_model

            model = initialize_model(runtime, model_class=DeepseekV4ForCausalLM).to(
                "npu"
            )
        finally:
            torch.set_default_dtype(dtype)
        checkpoint, expected = _native_checkpoint(model)
        loaded = model.load_weights(iter(checkpoint))
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, expected[name], rtol=0, atol=0)
        for module in model.modules():
            if isinstance(module, (LinearBase, RoutedExperts)):
                module.quant_method.process_weights_after_loading(module)
        leaves = {
            name: leaf
            for name, leaf in context.items()
            if getattr(leaf, "_fl_deepseek_v4_cache", False)
        }
        specs = {name: leaf.get_kv_cache_spec(runtime) for name, leaf in leaves.items()}
        groups = get_kv_cache_groups(runtime, specs)
        worker = KVCacheConfig(
            192,
            [
                KVCacheTensor(192 * spec.page_size_bytes, [name])
                for name, spec in specs.items()
            ],
            groups,
        )
        owner = SimpleNamespace(
            device=torch.device("npu:0"),
            runner_only_attn_layers=set(),
            use_ascend_hybrid_cache=False,
        )
        raw = ModelRunnerFL._allocate_kv_cache_tensors(owner, worker)
        views = {
            name: _reshape_attention_kv_cache(
                raw[name],
                spec,
                (192, spec.storage_block_size, spec.head_size),
                (0, 1, 2),
                192,
                None,
            )
            for name, spec in specs.items()
        }
        runner = []
        bind_kv_cache(views, leaves, runner)
        manager = KVCacheManager(
            generate_scheduler_kv_cache_config([worker]),
            256,
            128,
            128,
            max_num_batched_tokens=256,
            enable_caching=False,
        )
        initial = manager.block_pool.get_num_free_blocks()
        yield SimpleNamespace(
            model=model,
            config=runtime,
            expected=expected,
            loaded=loaded,
            checkpoint=checkpoint,
            leaves=leaves,
            specs=specs,
            groups=groups,
            manager=manager,
            requests=request_list,
            initial=initial,
            builders={},
        )
    finally:
        torch.set_default_dtype(dtype)
        try:
            if manager is not None:
                for request in request_list:
                    manager.free(request)
                assert manager.block_pool.get_num_free_blocks() == initial
        finally:
            for name in set(context) - original_names:
                context.pop(name, None)
            runtime.compilation_config.static_all_moe_layers[:] = original_moe_layers


def _request(env, name):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    request = Request(name, list(range(128)), SamplingParams(max_tokens=8), None)
    env.requests.append(request)
    return request


def _metadata(env, request, count, padded=None):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    from vllm_fl.attention.deepseek_v4 import DeepseekV4MetadataBuilder

    result = {}
    snapshots = []
    end = request.num_computed_tokens + count
    for name, leaf in env.leaves.items():
        spec = env.specs[name]
        group = next(i for i, g in enumerate(env.groups) if name in g.layer_names)
        pages = list(env.manager.get_blocks(request.request_id).get_block_ids()[group])
        capacity = (256 + spec.block_size - 1) // spec.block_size
        pages += [0] * max(0, capacity - len(pages))
        slots = []
        for position in range(request.num_computed_tokens, end):
            page = pages[position // spec.block_size]
            slots.append(
                page * spec.block_size + position % spec.block_size if page else -1
            )
        q = count if padded is None else padded
        slots += [-1] * (q - count)
        starts = torch.tensor([0, count], dtype=torch.int32)
        lengths = torch.tensor([end], dtype=torch.int32)
        common = CommonAttentionMetadata(
            query_start_loc=starts.to("npu"),
            query_start_loc_cpu=starts,
            seq_lens=lengths.to("npu"),
            num_reqs=1,
            num_actual_tokens=q,
            max_query_len=count,
            max_seq_len=end,
            block_table_tensor=torch.tensor([pages], device="npu", dtype=torch.int32),
            slot_mapping=torch.tensor(slots, device="npu"),
            _seq_lens_cpu=lengths,
        )
        if name not in env.builders:
            env.builders[name] = DeepseekV4MetadataBuilder(
                spec, [name], env.config, torch.device("npu:0")
            )
        result[name] = env.builders[name].build(0, common)
        snapshots.append(
            (common, common.block_table_tensor.clone(), common.slot_mapping.clone())
        )
    return result, snapshots


def _run(env, request, tokens, count, padded=None, inputs_embeds=None):
    from vllm.forward_context import set_forward_context

    assert env.manager.allocate_slots(request, count) is not None
    metadata, snapshots = _metadata(env, request, count, padded)
    start = request.num_computed_tokens
    positions = torch.arange(start, start + count, device="npu")
    if padded is not None:
        positions = torch.cat(
            [positions, torch.full((padded - count,), -1, device="npu")]
        )
    with set_forward_context(metadata, env.config, num_tokens=len(tokens)):
        hidden = env.model(tokens, positions, inputs_embeds=inputs_embeds)
        logits = env.model.compute_logits(hidden)
    for common, table, slots in snapshots:
        torch.testing.assert_close(common.block_table_tensor, table)
        torch.testing.assert_close(common.slot_mapping, slots)
    request.num_computed_tokens += count
    return hidden, logits


def test_native_loader_maps_complete_parameter_set(model_env):
    from vllm.model_executor.models.interfaces import is_mixture_of_experts

    env = model_env
    assert is_mixture_of_experts(env.model)
    assert env.model.num_local_physical_experts == 4
    assert len(env.model.moe_layers) == env.model.num_moe_layers == 3
    assert set(dict(env.model.named_parameters())) <= env.loaded
    assert all("mtp" not in name for name in env.loaded)
    for index, layer in enumerate(env.model.model.layers):
        assert layer.self_attn.compress_ratio == (0, 4, 128)[index]
        assert layer.mlp.gate.weight.dtype == torch.float32
        assert layer.mlp.shared_experts is not None
        routed = layer.mlp.experts.routed_experts
        assert routed.w13_weight.dtype == routed.w2_weight.dtype == torch.int8
        assert routed.quant_method.supports_internal_mk
    assert len(env.leaves) == 10

    before = model_env.model.model.layers[0].self_attn.wq_a.weight.clone()
    result = model_env.model.load_weights(
        iter(
            [
                (
                    "mtp.0.attn.wq_a.weight",
                    torch.ones(32, 128, device="npu", dtype=torch.bfloat16),
                )
            ]
        )
    )
    assert not result
    torch.testing.assert_close(
        model_env.model.model.layers[0].self_attn.wq_a.weight, before
    )

    with pytest.raises((KeyError, ValueError)):
        model_env.model.load_weights(
            iter([("layers.0.not_a_parameter.weight", torch.ones(1, device="npu"))])
        )


def test_profile_is_cache_free(model_env):
    from vllm.forward_context import set_forward_context

    env = model_env
    before = {name: leaf.kv_cache.clone() for name, leaf in env.leaves.items()}
    tokens = torch.tensor([1, 2, 3, 4], device="npu")
    with set_forward_context(None, env.config, num_tokens=4):
        logits = env.model.compute_logits(
            env.model(tokens, torch.arange(4, device="npu"))
        )
    assert torch.isfinite(logits).all()
    for name, leaf in env.leaves.items():
        torch.testing.assert_close(leaf.kv_cache, before[name], rtol=0, atol=0)


def test_full_vs_chunk_decode_token_to_logits(model_env):
    env = model_env
    tokens = (torch.arange(128, device="npu") * 7 % 128).long()

    def scenario(name, cuts):
        request = _request(env, name)
        outputs = []
        for begin, end in zip(cuts, cuts[1:]):
            _, logits = _run(env, request, tokens[begin:end], end - begin)
            assert torch.isfinite(logits).all()
            outputs.append(logits.clone())
        env.manager.free(request)
        env.requests.remove(request)
        return torch.cat(outputs)

    full = scenario("full", [0, 127, 128])
    for leaf in env.leaves.values():
        leaf.kv_cache.zero_()
    chunk = scenario("chunk", [0, 3, 4, 127, 128])
    torch.testing.assert_close(full, chunk, rtol=0.03, atol=0.03)
    assert full.shape == (128, 128)
    assert env.manager.block_pool.get_num_free_blocks() == env.initial


def _quant_linear(x, weight, scale):
    maximum = x.float().abs().amax(-1, keepdim=True)
    activation_scale = maximum / 127
    quant = (
        (x.float() / torch.where(activation_scale > 0, activation_scale, 1))
        .round()
        .clamp(-128, 127)
    )
    return (
        (quant @ weight.float().t()) * activation_scale * scale.float().reshape(1, -1)
    ).to(x.dtype)


@pytest.mark.parametrize("layer_index", [0, 1])
def test_hash_or_bias_router_and_shared_scaling_once(model_env, layer_index):
    env = model_env
    mlp = env.model.model.layers[layer_index].mlp
    x = (
        (torch.arange(3 * 128, device="npu").reshape(3, 128) % 41 - 20) * 0.02
    ).bfloat16()
    tokens = torch.tensor([0, 1, 3], device="npu")
    logits = F.linear(x.float(), mlp.gate.weight.float())
    torch.testing.assert_close(mlp.gate(x)[0], logits)
    scores = F.softplus(logits).sqrt()
    if layer_index == 0:
        ids = mlp.gate.tid2eid[tokens].long()
    else:
        ids = (scores + mlp.gate.e_score_correction_bias.float()).argsort(
            -1, descending=True
        )[:, :2]
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(-1, keepdim=True)
    router = mlp.experts.router
    assert router.e_score_correction_bias is mlp.gate.e_score_correction_bias
    assert router._hash_indices_table is mlp.gate.tid2eid
    actual_weights, actual_ids = router.select_experts(x, logits, input_ids=tokens)
    torch.testing.assert_close(actual_ids.sort(-1).values.long(), ids.sort(-1).values)
    route_weights = scores.gather(1, actual_ids.long())
    route_weights = route_weights / route_weights.sum(-1, keepdim=True) * 1.5
    torch.testing.assert_close(actual_weights, route_weights)
    prefix = f"model.layers.{layer_index}.mlp.experts.routed_experts."
    w1, w2 = (env.expected[prefix + name] for name in ("w13_weight", "w2_weight"))
    s1, s2 = (
        env.expected[prefix + name] for name in ("w13_weight_scale", "w2_weight_scale")
    )
    routed = torch.zeros_like(x, dtype=torch.float32)
    for row in range(3):
        for route in range(2):
            expert = int(ids[row, route].item())
            gate, up = _quant_linear(x[row : row + 1], w1[expert], s1[expert]).chunk(
                2, -1
            )
            gate = gate.clamp(max=10.0)
            activated = (
                gate.float()
                * torch.sigmoid(gate.float())
                * up.clamp(-10.0, 10.0).float()
            ).to(x.dtype)
            down = _quant_linear(activated, w2[expert], s2[expert])
            routed[row] += down[0].float() * weights[row, route] * 1.5
    sp = f"model.layers.{layer_index}.mlp.shared_experts."
    shared_gate, shared_up = _quant_linear(
        x,
        env.expected[sp + "gate_up_proj.weight"],
        env.expected[sp + "gate_up_proj.weight_scale"],
    ).chunk(2, -1)
    shared = (
        shared_gate.clamp(max=10.0).float()
        * torch.sigmoid(shared_gate.clamp(max=10.0).float())
        * shared_up.clamp(-10.0, 10.0).float()
    ).to(x.dtype)
    shared = _quant_linear(
        shared,
        env.expected[sp + "down_proj.weight"],
        env.expected[sp + "down_proj.weight_scale"],
    )
    from vllm_fl.dispatch import CachedOp

    experts = mlp.experts.routed_experts
    actual_routed = CachedOp("modelslim_w8a8_experts")(
        x,
        experts.w13_weight,
        experts.w2_weight,
        experts.w13_weight_scale,
        experts.w2_weight_scale,
        actual_weights,
        actual_ids,
        activation=experts.activation.value,
        limit=10.0,
    )
    torch.testing.assert_close(actual_routed, routed.to(x.dtype), rtol=0.03, atol=0.06)
    actual_shared = mlp.shared_experts(x)
    torch.testing.assert_close(actual_shared, shared, rtol=0.03, atol=0.06)
    from vllm.forward_context import get_forward_context, set_forward_context

    with set_forward_context(None, env.config, num_tokens=3):
        # This test calls one layer, not the full model's ordered MoE chain.
        # On torch <2.11 the custom op resolves its owner through this cursor.
        get_forward_context().all_moe_layers = [mlp.experts.layer_name]
        actual = mlp(x, input_ids=tokens)
    # Each component is checked independently above. The runner adds their
    # BF16 outputs, not an unrounded FP32 routed accumulator to the shared path.
    torch.testing.assert_close(actual, actual_routed + actual_shared, rtol=0, atol=0)


def test_padded_positions_preserve_cache_safety(model_env):
    request = _request(model_env, "padding")
    tokens = torch.tensor([1, 2, 3, 0], device="npu")
    hidden, logits = _run(model_env, request, tokens, 3, padded=4)
    assert torch.isfinite(hidden[:3]).all()
    assert torch.isfinite(logits[:3]).all()
    for leaf in model_env.leaves.values():
        assert torch.isfinite(leaf.kv_cache).all()
        assert torch.count_nonzero(leaf.kv_cache[0]) == 0


def test_complete_model_changed_tokens_graph(model_env):
    import torch_npu  # noqa: F401

    from vllm.forward_context import set_forward_context

    env = model_env
    request = _request(env, "graph")
    assert env.manager.allocate_slots(request, 4) is not None
    metadata, _ = _metadata(env, request, 4)
    tokens = torch.tensor([1, 2, 3, 4], device="npu")
    positions = torch.arange(4, device="npu")

    def run():
        with set_forward_context(metadata, env.config, num_tokens=4):
            return env.model.compute_logits(env.model(tokens, positions))

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    torch.npu.synchronize()
    previous = output.clone()
    for _ in range(2):
        tokens.add_(7)
        expected = run().clone()
        graph.replay()
        torch.npu.synchronize()
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, rtol=0.03, atol=0.03)
        assert not torch.equal(output, previous)
        previous = output.clone()


def test_fl_graph_wrapper_dummy_capture_and_changed_decode_boundaries(
    model_env, monkeypatch
):
    import torch_npu  # noqa: F401

    import vllm.compilation.monitor as graph_monitor
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, set_forward_context

    from vllm_fl.compilation.graph import GraphWrapper
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    env = model_env
    request = _request(env, "fl-wrapper-boundaries")
    _run(env, request, torch.arange(126, device="npu"), 126)
    assert request.num_computed_tokens == 126
    assert env.manager.allocate_slots(request, 1) is not None
    _, snapshots = _metadata(env, request, 1, padded=2)
    assert len(snapshots) == len(env.builders)
    dummy = {
        name: builder.build_for_dummy(common)
        for (name, builder), (common, _, _) in zip(env.builders.items(), snapshots)
    }
    layout = {
        (name, field): (tensor.data_ptr(), tensor.shape, tensor.stride())
        for name, plan in dummy.items()
        for field, tensor in vars(plan).items()
        if isinstance(tensor, torch.Tensor)
    }
    # Include padding between typed pages, not just the logical cache elements.
    raw_caches = {}
    for name, leaf in env.leaves.items():
        storage = leaf.kv_cache.untyped_storage()
        raw_caches[name] = torch.empty(
            0, dtype=torch.uint8, device=leaf.kv_cache.device
        ).set_(storage, 0, (storage.nbytes(),), (1,))
    before_dummy = {name: raw.clone() for name, raw in raw_caches.items()}

    tokens = torch.tensor([19, 0], device="npu")
    positions = torch.tensor([126, -1], device="npu")
    descriptor = BatchDescriptor(num_tokens=2, num_reqs=2, uniform=True)
    monkeypatch.setattr(
        env.config.compilation_config,
        "cudagraph_mode",
        CUDAGraphMode.FULL_DECODE_ONLY,
    )
    monkeypatch.setattr(graph_monitor, "cudagraph_capturing_enabled", True)
    controller = GraphRuntimeController(vllm_config=env.config)
    controller.bind_attention_groups(
        [
            [
                SimpleNamespace(backend=leaf.get_attn_backend())
                for leaf in env.leaves.values()
            ]
        ]
    )
    assert controller.attention_impls == ()
    python_calls = []

    def runnable(input_ids, input_positions):
        python_calls.append("forward")
        return env.model(input_ids, input_positions)

    wrapper = GraphWrapper(
        runnable,
        env.config,
        runtime_mode=CUDAGraphMode.FULL,
        graph_runtime=controller,
    )
    try:
        # Forced eager dummy warmups execute the same static numerical chain.
        for _ in range(3):
            with set_forward_context(
                dummy,
                env.config,
                num_tokens=2,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=descriptor,
            ):
                wrapper(tokens, positions)
            torch.npu.synchronize()
            for name, raw in raw_caches.items():
                torch.testing.assert_close(raw, before_dummy[name], rtol=0, atol=0)
        assert len(python_calls) == 3
        with set_forward_context(
            dummy,
            env.config,
            num_tokens=2,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=descriptor,
        ):
            captured_hidden = wrapper(tokens, positions)
        torch.npu.synchronize()
        assert len(python_calls) == 4
        assert wrapper.concrete_graph_entries[descriptor].graph is not None
        for name, raw in raw_caches.items():
            torch.testing.assert_close(raw, before_dummy[name], rtol=0, atol=0)

        for position in range(126, 132):
            token = (position * 13 + 7) % 128
            if position >= request.num_tokens:
                request.append_output_token_ids(token)
            assert env.manager.allocate_slots(request, 1) is not None
            real, public = _metadata(env, request, 1, padded=2)
            for name, plan in real.items():
                for field, tensor in vars(plan).items():
                    if isinstance(tensor, torch.Tensor):
                        assert (
                            tensor.data_ptr(),
                            tensor.shape,
                            tensor.stride(),
                        ) == layout[name, field]
            for ratio in (4, 128):
                prefix = "model.layers.1" if ratio == 4 else "model.layers.2"
                plan = real[prefix + ".self_attn.compact_cache"]
                assert plan.num_completed == int((position + 1) % ratio == 0)
                assert torch.all(plan.positions[1:] == -1)
                assert torch.all(plan.output_slots[plan.num_completed :] == -1)
            tokens.copy_(torch.tensor([token, 0], device="npu"))
            positions.copy_(torch.tensor([position, -1], device="npu"))
            before_real = {name: raw.clone() for name, raw in raw_caches.items()}
            with set_forward_context(
                real,
                env.config,
                num_tokens=2,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                expected_hidden = env.model(tokens, positions)
                expected_logits = env.model.compute_logits(expected_hidden).clone()
            expected_cache = {
                name: leaf.kv_cache.clone() for name, leaf in env.leaves.items()
            }
            for name, raw in raw_caches.items():
                raw.copy_(before_real[name])
            with set_forward_context(
                real,
                env.config,
                num_tokens=2,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=descriptor,
            ):
                hidden = wrapper(tokens, positions)
            torch.npu.synchronize()
            logits = env.model.compute_logits(hidden)
            assert len(python_calls) == 4, "Replay re-entered the Python model"
            assert wrapper.concrete_graph_entries[descriptor].graph is not None
            assert hidden.data_ptr() == captured_hidden.data_ptr()
            assert torch.isfinite(logits[:1]).all()
            torch.testing.assert_close(
                logits[:1], expected_logits[:1], rtol=0.03, atol=0.03
            )
            for name, leaf in env.leaves.items():
                torch.testing.assert_close(
                    leaf.kv_cache, expected_cache[name], rtol=0.03, atol=0.03
                )
                assert torch.count_nonzero(leaf.kv_cache[0]) == 0
            for common, table, slots in public:
                torch.testing.assert_close(common.block_table_tensor, table)
                torch.testing.assert_close(common.slot_mapping, slots)
            request.num_computed_tokens += 1
    finally:
        torch.npu.synchronize()
        wrapper.clear_graphs()
        controller.clear_decoder_graphs()
