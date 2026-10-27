"""Opt-in full QSA projection/cache/index/attention chain on an assigned GPU.

Uses real H2560/Q24/KV2/D256 and index4x128, TP1 on one assigned device.
No checkpoint weights are loaded. History below the 2048-token sparse budget
has an independent dense causal oracle. This does not validate TP8 collectives.

Set QWEN38_ENGINE_CONFIG to an EngineArgs JSON fixture whose model is a local
checkpoint directory. The fixture supplies the model/tokenizer paths; this
component test retains its fixed TP1, BF16, no-speculation runtime settings.
Missing config/model fixtures skip before accelerator imports. Optional
QWEN38_TEST_RESULTS_DIR selects result storage; otherwise pytest tmp_path is used.
"""

import json
import math
import os
from pathlib import Path

import pytest

if os.environ.get("QWEN38_RUN_DEVICE_TESTS") != "1":
    pytest.skip("requires assigned GPU window", allow_module_level=True)

_engine_config = os.environ.get("QWEN38_ENGINE_CONFIG")
if not _engine_config or not Path(_engine_config).expanduser().is_file():
    pytest.skip(
        "set QWEN38_ENGINE_CONFIG to a local EngineArgs JSON", allow_module_level=True
    )
_engine_path = Path(_engine_config).expanduser().resolve()
_ENGINE_ARGS = json.loads(_engine_path.read_text())
if not isinstance(_ENGINE_ARGS, dict):
    raise ValueError("QWEN38_ENGINE_CONFIG must contain an EngineArgs JSON object")
_model_path = _ENGINE_ARGS.get("model")
if not isinstance(_model_path, str) or not _model_path:
    pytest.skip(
        "engine fixture requires a local model directory", allow_module_level=True
    )
_model_path = Path(_model_path).expanduser()
if not _model_path.is_absolute():
    _model_path = _engine_path.parent / _model_path
if not (_model_path / "config.json").is_file():
    pytest.skip("local model config.json fixture is missing", allow_module_level=True)
_ENGINE_ARGS["model"] = str(_model_path.resolve())

import flag_gems
import torch
import torch.nn.functional as F

from vllm.config import set_current_vllm_config
from vllm.distributed import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context

import vllm_fl
from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import (
    PreparedQSAForwardMetadata as QSAForwardMetadata,
)
from vllm_fl.models.qwen3_8_flash_next.gpu.qsa import Qwen3_8FlashNextQSAAttention


def rms(x, weight, eps):
    f = x.float()
    return (
        f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps) * (1 + weight.float())
    ).to(x.dtype)


def rope(x, positions, layer):
    dim = layer.rotary_emb.rotary_dim
    cache = layer.rotary_emb.cos_sin_cache.to(x.dtype)[positions]
    cos, sin = cache.float().chunk(2, -1)
    first, second = x[..., :dim].float().chunk(2, -1)
    rotated = torch.cat(
        (
            first * cos[:, None] - second * sin[:, None],
            second * cos[:, None] + first * sin[:, None],
        ),
        -1,
    )
    return torch.cat((rotated.to(x.dtype), x[..., dim:]), -1)


def dense_oracle(layer, all_x):
    """Independent ATen math outside global FG, including projection/norm/RoPE."""
    count = all_x.shape[0]
    positions = torch.arange(count, device=all_x.device)
    qkv = F.linear(all_x, layer.qkv_proj.weight)
    q_gate, k, v = qkv.split([layer.q_size * 2, layer.kv_size, layer.kv_size], -1)
    q, gate = q_gate.view(count, layer.num_heads, 512).chunk(2, -1)
    q = rope(
        rms(q, layer.q_norm.weight, layer.q_norm.variance_epsilon), positions, layer
    )
    k = k.view(count, layer.num_kv_heads, 256)
    k = rope(
        rms(k, layer.k_norm.weight, layer.k_norm.variance_epsilon), positions, layer
    )
    v = v.view(count, layer.num_kv_heads, 256)
    repeat = layer.num_heads // layer.num_kv_heads
    scores = (
        torch.einsum("thd,shd->hts", q.float(), k.repeat_interleave(repeat, 1).float())
        / 16
    )
    causal = (
        torch.arange(count, device=all_x.device)[None, :]
        <= torch.arange(count, device=all_x.device)[:, None]
    )
    scores.masked_fill_(~causal[None], -torch.inf)
    probabilities = torch.softmax(scores, -1)
    output = torch.einsum(
        "hts,shd->thd", probabilities, v.repeat_interleave(repeat, 1).float()
    ).to(all_x.dtype)
    output = (output * torch.sigmoid(gate)).flatten(1)
    projected = F.linear(output, layer.o_proj.weight)
    index_qk = F.linear(all_x, layer.indexer.index_qk_proj.weight)
    raw_index = index_qk[:, -128:].view(count, 1, 128)
    return projected, k, v, raw_index


def padded_cache(shape, padding):
    logical = math.prod(shape[1:])
    backing = torch.zeros(
        shape[0], logical + padding, device="cuda", dtype=torch.bfloat16
    )
    strides = [logical + padding]
    strides.extend(math.prod(shape[i + 1 :]) for i in range(1, len(shape)))
    return backing.as_strided(shape, strides)


@pytest.fixture(scope="module")
def layer_and_config(tmp_path_factory):
    torch.manual_seed(9024)
    torch.cuda.set_device(0)
    vllm_fl.register_model()
    engine_args = dict(_ENGINE_ARGS)
    # Preserve the established single-device numerical fixture, even when the
    # supplied EngineArgs JSON describes a multi-rank deployment.
    engine_args.update(
        tensor_parallel_size=1,
        dtype="bfloat16",
        language_model_only=True,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_ssm_cache_dtype="float32",
        async_scheduling=False,
        disable_custom_all_reduce=True,
        compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=False,
        quantization=None,
        speculative_config=None,
    )
    config = EngineArgs(**engine_args).create_engine_config()
    rendezvous = tmp_path_factory.mktemp("qsa-distributed") / "rendezvous"
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=rendezvous.as_uri(),
            backend="gloo",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, backend="gloo"
        )
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                layer = Qwen3_8FlashNextQSAAttention(
                    vllm_config=config,
                    config=config.model_config.hf_text_config,
                    layer_id=3,
                    prefix="qsa_full_test.layers.3.self_attn",
                )
        finally:
            torch.set_default_dtype(old_dtype)
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            parameter.normal_(
                std=0.02 if "norm" in name else 1 / math.sqrt(parameter.shape[-1])
            )
    layer.kv_cache = padded_cache((5, 2, 128, 2, 256), 1024)
    layer.indexer.raw_key_cache.kv_cache = padded_cache((5, 128, 1, 128), 128)
    layer.indexer.compressed_key_cache.kv_cache = padded_cache((5, 32, 1, 128), 128)
    print(
        {
            "device": torch.cuda.get_device_name(0),
            "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "scope": "real QSA layer, TP1; all FG ATen enabled during forward",
            "rotary_dim": layer.rotary_emb.rotary_dim,
        },
        flush=True,
    )
    yield layer, config
    destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def metadata(layer, start, count, pages):
    pos = torch.arange(start, start + count, device="cuda", dtype=torch.int64)
    seq = torch.tensor([start + count], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, count], device="cuda", dtype=torch.int32)
    req = torch.zeros(count, device="cuda", dtype=torch.int32)
    result = {}
    for name, table_pages, compressed in [
        (layer.layer_name, pages[0], False),
        (layer.indexer.raw_key_cache.prefix, pages[1], False),
        (layer.indexer.compressed_key_cache.prefix, pages[2], True),
    ]:
        table = torch.tensor([table_pages], device="cuda", dtype=torch.int32)
        logical = pos // 4 if compressed else pos
        size = 32 if compressed else 128
        slots = table[0, logical // size].long() * size + logical % size
        if compressed:
            slots.masked_fill_((pos + 1) % 4 != 0, -1)
        result[name] = QSAForwardMetadata(
            block_table=table,
            slot_mapping=slots,
            seq_lens=seq,
            query_start_loc=offsets,
            token_to_req=req,
            logical_positions=pos,
            num_actual_tokens=count,
            storage_block_size=size,
            compress_ratio=4 if compressed else 1,
        )
    return pos, result


def clear_caches(layer):
    for cache in (
        layer.kv_cache,
        layer.indexer.raw_key_cache.kv_cache,
        layer.indexer.compressed_key_cache.kv_cache,
    ):
        # Clear allocator storage, not its intentionally strided logical view.
        cache._base.zero_()


@pytest.mark.parametrize("m", [1, 2, 8, 128, 256])
def test_full_qsa_eager_changed_graph_global_flaggems(layer_and_config, m, tmp_path):
    layer, config = layer_and_config
    prefix = 7
    all_x = torch.randn(prefix + m, 2560, device="cuda", dtype=torch.bfloat16) * 0.2
    changed = torch.randn_like(all_x) * 0.2
    with torch.no_grad():
        expected, _, _, _ = dense_oracle(layer, all_x)
        expected_changed, changed_k, changed_v, changed_raw = dense_oracle(
            layer, changed
        )
    eager_only = os.environ.get("QWEN38_QSA_EAGER_ONLY") == "1"
    pages = ([2, 1, 3, 4], [1, 3, 2, 4], [3, 1, 2, 4])
    prefix_pos, prefix_md = metadata(layer, 0, prefix, pages)
    positions, md = metadata(layer, prefix, m, pages)
    x = all_x[prefix:].clone()

    def forward(hidden, pos, meta):
        with set_forward_context(meta, config, num_tokens=hidden.shape[0]):
            return layer(pos, hidden)

    with torch.no_grad(), flag_gems.use_gems():
        assert flag_gems.current_work_registrar is not None
        clear_caches(layer)
        forward(all_x[:prefix], prefix_pos, prefix_md)
        eager = forward(x, positions, md)
        torch.cuda.synchronize()
        for _ in range(2):
            forward(x, positions, md)
        graph = None
        if not eager_only:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = forward(x, positions, md)
        clear_caches(layer)
        forward(changed[:prefix], prefix_pos, prefix_md)
        x.copy_(changed[prefix:])
        if graph is None:
            captured = forward(x, positions, md)
        else:
            graph.replay()
        torch.cuda.synchronize()
    difference = (eager.float() - expected[prefix:].float()).abs()
    graph_difference = (captured.float() - expected_changed[prefix:].float()).abs()
    torch.testing.assert_close(eager, expected[prefix:], atol=0.02, rtol=0.03)
    torch.testing.assert_close(
        captured, expected_changed[prefix:], atol=0.02, rtol=0.03
    )
    relative_l2 = float(
        (captured.float() - expected_changed[prefix:].float()).norm()
        / expected_changed[prefix:].float().norm().clamp_min(1e-12)
    )
    assert relative_l2 < 0.03
    # Check raw-index and compressed-index contents too: dense attention alone
    # cannot detect incorrect index scores when every history block fits TopK.
    logical = torch.arange(prefix + m, device="cuda")
    table_main = torch.tensor(pages[0], device="cuda")
    table_raw = torch.tensor(pages[1], device="cuda")
    observed_k = layer.kv_cache[table_main[logical // 128], 0, logical % 128]
    observed_v = layer.kv_cache[table_main[logical // 128], 1, logical % 128]
    observed_raw = layer.indexer.raw_key_cache.key_cache[
        table_raw[logical // 128], logical % 128
    ]
    torch.testing.assert_close(observed_k, changed_k, atol=0.02, rtol=0.02)
    torch.testing.assert_close(observed_v, changed_v, atol=0.003, rtol=0.02)
    torch.testing.assert_close(observed_raw, changed_raw, atol=0.003, rtol=0.02)
    groups = (prefix + m) // 4
    pooled = (
        changed_raw[: groups * 4].float().view(groups, 4, 1, 128).mean(1).bfloat16()
    )
    normed = rms(
        pooled,
        layer.indexer.k_layernorm.weight,
        layer.indexer.k_layernorm.variance_epsilon,
    )
    grouped_positions = torch.arange(groups, device="cuda") * 4
    expected_compressed = rope(normed, grouped_positions, layer)
    gi = torch.arange(groups, device="cuda")
    table_comp = torch.tensor(pages[2], device="cuda")
    observed_compressed = layer.indexer.compressed_key_cache.kv_cache[
        table_comp[gi // 32], gi % 32
    ]
    torch.testing.assert_close(
        observed_compressed, expected_compressed, atol=0.03, rtol=0.03
    )
    compressed_error = float(
        (observed_compressed.float() - expected_compressed.float()).abs().max()
    )
    result = {
        "M": m,
        "prefix": prefix,
        "eager_max_abs": float(difference.max()),
        "changed_max_abs": float(graph_difference.max()),
        "changed_relative_l2": relative_l2,
        "full_flaggems": True,
        "mode": "eager_only" if eager_only else "eager_and_changed_graph",
        "compressed_cache_max_abs": compressed_error,
        "raw_and_main_cache_checked": True,
        "whole_qsa_forward": True,
        "dense_oracle": True,
    }
    print(result, flush=True)
    results_dir = Path(
        os.environ.get("QWEN38_TEST_RESULTS_DIR") or str(tmp_path)
    ).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "qsa-full-forward-results.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(result) + "\n")
