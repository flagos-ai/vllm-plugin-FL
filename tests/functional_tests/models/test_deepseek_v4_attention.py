# SPDX-License-Identifier: Apache-2.0
"""Actual V4 attention modules over real scheduler-owned NPU cache pages."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(current_platform.device_type != "npu", reason="Requires Ascend"),
]


@pytest.fixture(scope="module")
def runtime():
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        CompilationMode,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.network_utils import get_open_port

    from vllm_fl import _patch_ascend_torch_accelerator

    torch.npu.set_device(0)
    _patch_ascend_torch_accelerator()
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, custom_ops=["all"]
        ),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=2,
        ),
        cache_config=CacheConfig(block_size=128, enable_prefix_caching=False),
    )
    with set_current_vllm_config(config), torch.inference_mode():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="hccl",
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        )
        initialize_model_parallel(1, backend="hccl")
        try:
            yield config
        finally:
            try:
                destroy_model_parallel()
            finally:
                destroy_distributed_environment()


@pytest.fixture(autouse=True)
def strict_dispatch(runtime):
    from vllm_fl.dispatch import get_default_manager, with_strict_mode

    with with_strict_mode():
        assert (
            get_default_manager().get_selected_impl_id("deepseek_v4_index_quant")
            == "vendor.ascend"
        )
        yield


def _values(shape, dtype=torch.bfloat16, offset=0):
    size = 1
    for dim in shape:
        size *= dim
    return (
        ((torch.arange(size, device="npu").reshape(shape) + offset) % 37 - 18) * 0.021
    ).to(dtype)


@pytest.fixture
def make_attention(runtime):
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_groups,
    )
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheTensor,
        MLAAttentionSpec,
        SlidingWindowMLASpec,
    )
    from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache

    from vllm_fl.kv_cache.deepseek_v4 import DeepseekV4CacheLayer
    from vllm_fl.models.deepseek_v4 import DeepseekV4Attention
    from vllm_fl.worker.kv_cache_utils import bind_kv_cache
    from vllm_fl.worker.model_runner import ModelRunnerFL

    environments = []
    context = runtime.compilation_config.static_forward_context
    original_names = set(context)

    def create(ratio, mixed=False, topk=2):
        cfg = SimpleNamespace(
            hidden_size=8,
            num_attention_heads=2,
            q_lora_rank=4,
            o_lora_rank=2,
            head_dim=4,
            qk_rope_head_dim=2,
            o_groups=2,
            sliding_window=8,
            rms_norm_eps=1e-6,
            compress_ratios=[ratio],
            index_head_dim=4,
            index_n_heads=2,
            index_topk=topk,
        )
        prefix = "model.layers.0.self_attn"
        common = dict(
            block_size=128,
            num_kv_heads=1,
            model_version="deepseek_v4",
            page_size_padded=8192,
        )
        specs = {
            "swa": SlidingWindowMLASpec(
                **common, head_size=4, dtype=torch.bfloat16, sliding_window=8
            )
        }
        if ratio:
            specs.update(
                {
                    "compact": MLAAttentionSpec(
                        **common,
                        head_size=4,
                        dtype=torch.bfloat16,
                        compress_ratio=ratio,
                    ),
                    "state": SlidingWindowMLASpec(
                        **common,
                        head_size=16 if ratio == 4 else 8,
                        dtype=torch.float32,
                        sliding_window=8 if ratio == 4 else 128,
                    ),
                }
            )
        if ratio == 4:
            specs.update(
                {
                    "index_key": MLAAttentionSpec(
                        **common, head_size=4, dtype=torch.int8, compress_ratio=4
                    ),
                    "index_scale": MLAAttentionSpec(
                        **common, head_size=1, dtype=torch.float16, compress_ratio=4
                    ),
                    "index_state": SlidingWindowMLASpec(
                        **common, head_size=16, dtype=torch.float32, sliding_window=8
                    ),
                }
            )
        full_specs = {
            prefix + "." + key + "_cache": spec for key, spec in specs.items()
        }
        groups = get_kv_cache_groups(runtime, full_specs)
        worker = KVCacheConfig(
            64,
            [
                KVCacheTensor(64 * spec.page_size_bytes, [name])
                for name, spec in full_specs.items()
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
                (64, spec.storage_block_size, spec.head_size),
                (0, 1, 2),
                64,
                None,
            )
            for name, spec in full_specs.items()
        }
        manager = KVCacheManager(
            generate_scheduler_kv_cache_config([worker]),
            512,
            128,
            128,
            max_num_batched_tokens=512,
            enable_caching=False,
        )
        leaves = {}
        caches = []
        env = SimpleNamespace(
            cfg=cfg,
            ratio=ratio,
            leaves=leaves,
            specs=specs,
            groups=groups,
            manager=manager,
            config=runtime,
            requests=[],
            builders={},
            caches=caches,
            free_before=manager.block_pool.get_num_free_blocks(),
        )
        # Register cleanup before creating a leaf or constructing model modules:
        # a failed constructor must not poison the next test's static context.
        environments.append(env)
        for key, spec in specs.items():
            leaves[key] = DeepseekV4CacheLayer(
                prefix + "." + key + "_cache", spec, vllm_config=runtime
            )
        bind_kv_cache(views, {leaf.prefix: leaf for leaf in leaves.values()}, caches)
        angle = torch.arange(512, device="npu").float() * 0.013
        rope = torch.stack([angle.cos(), angle.sin()], -1)
        old_quant = runtime.quant_config
        if mixed:
            from vllm_fl.quantization.w8a8.modelslim import AscendModelSlimConfig

            runtime.quant_config = AscendModelSlimConfig(
                {
                    prefix + "." + name + ".weight": "W8A8_DYNAMIC"
                    if name in ("wq_a", "wq_b", "wkv", "indexer.wq_b")
                    else "FLOAT"
                    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b", "indexer.wq_b")
                },
                weight_nz_mode=0,
            )
        try:
            module = DeepseekV4Attention(runtime, cfg, leaves, rope, prefix=prefix).to(
                "npu"
            )
        finally:
            runtime.quant_config = old_quant
        checkpoint = {}
        for index, (name, param) in enumerate(module.named_parameters()):
            if param.dtype == torch.int8:
                value = (
                    (_values(tuple(param.shape), torch.float32, index * 7) * 47)
                    .round()
                    .to(torch.int8)
                )
            elif name.endswith("weight_offset"):
                value = torch.zeros_like(param)
            elif name.endswith("weight_scale"):
                value = torch.full_like(param, 0.015625)
            elif "norm.weight" in name:
                value = _values(tuple(param.shape), param.dtype, index).abs() + 0.7
            elif name == "attn_sink":
                value = torch.tensor([0.2, -0.3], device="npu")
            else:
                value = _values(tuple(param.shape), param.dtype, index * 7)
            getattr(param, "weight_loader", default_weight_loader)(param, value)
            checkpoint[name] = value.clone()
        for linear in module.modules():
            if isinstance(linear, LinearBase):
                linear.quant_method.process_weights_after_loading(linear)
        env.module, env.checkpoint, env.rope = module, checkpoint, rope
        return env

    yield create
    try:
        for env in environments:
            for request in env.requests:
                env.manager.free(request)
        for env in environments:
            assert env.manager.block_pool.get_num_free_blocks() == env.free_before
    finally:
        for name in set(context) - original_names:
            context.pop(name, None)


def _request(env, name="request"):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    request = Request(name, list(range(256)), SamplingParams(max_tokens=8), None)
    env.requests.append(request)
    return request


def _metadata(env, requests, counts, padded=None):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    from vllm_fl.attention.deepseek_v4 import DeepseekV4MetadataBuilder

    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    ends = [
        request.num_computed_tokens + count for request, count in zip(requests, counts)
    ]
    output = {}
    public = []
    for key, leaf in env.leaves.items():
        group_index = next(
            i for i, g in enumerate(env.groups) if leaf.prefix in g.layer_names
        )
        tables = []
        slots = []
        for request, count, end in zip(requests, counts, ends):
            pages = list(
                env.manager.get_blocks(request.request_id).get_block_ids()[group_index]
            )
            pages += [0] * (4 - len(pages))
            tables.append(pages)
            for pos in range(end - count, end):
                page = pages[pos // 128]
                slots.append(page * 128 + pos % 128 if page else -1)
        n = starts[-1] if padded is None else padded
        slots += [-1] * (n - len(slots))
        seq = torch.tensor(ends, dtype=torch.int32)
        begin = torch.tensor(starts, dtype=torch.int32)
        common = CommonAttentionMetadata(
            query_start_loc=begin.to("npu"),
            query_start_loc_cpu=begin,
            seq_lens=seq.to("npu"),
            num_reqs=len(requests),
            num_actual_tokens=n,
            max_query_len=max(counts),
            max_seq_len=max(ends),
            block_table_tensor=torch.tensor(tables, device="npu", dtype=torch.int32),
            slot_mapping=torch.tensor(slots, device="npu"),
            _seq_lens_cpu=seq,
        )
        if key not in env.builders:
            env.builders[key] = DeepseekV4MetadataBuilder(
                leaf.spec, [leaf.prefix], env.config, torch.device("npu:0")
            )
        output[leaf.prefix] = env.builders[key].build(0, common)
        public.append(
            (common, common.block_table_tensor.clone(), common.slot_mapping.clone())
        )
    return output, public


def _linear(env, name, x):
    weight = env.checkpoint[name + ".weight"]
    if weight.dtype == torch.int8:
        maximum = x.float().abs().amax(-1, keepdim=True)
        scale = maximum / 127
        quant = (x.float() / torch.where(scale > 0, scale, 1)).round().clamp(-128, 127)
        result = (
            (quant @ weight.float().t())
            * scale
            * env.checkpoint[name + ".weight_scale"].float().reshape(1, -1)
        )
        return result.to(x.dtype)
    return F.linear(x.float(), weight.float()).to(x.dtype)


def _rms(x, weight=None):
    value = x.float() * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    return (value if weight is None else value * weight.float()).to(x.dtype)


def _rope(x, position, cache, inverse=False):
    c, s = cache[position]
    if inverse:
        s = -s
    result = x.float().clone()
    a, b = x[..., -2].float(), x[..., -1].float()
    result[..., -2], result[..., -1] = a * c - b * s, b * c + a * s
    return result.to(x.dtype)


def _compress(env, prefix, history, end):
    ratio = env.ratio
    coff = 2 if ratio == 4 else 1
    kv = F.linear(history.float(), env.checkpoint[prefix + ".wkv.weight"].float())
    gates = F.linear(history.float(), env.checkpoint[prefix + ".wgate.weight"].float())
    values, scores = [], []
    for time in range(max(0, end - ratio * coff + 1), end + 1):
        half = int(coff == 2 and time >= end - ratio + 1)
        values.append(kv[time, half * 4 : (half + 1) * 4])
        scores.append(
            (gates[time] + env.checkpoint[prefix + ".ape"][time % ratio])[
                half * 4 : (half + 1) * 4
            ]
        )
    value = (torch.stack(values) * torch.stack(scores).softmax(0)).sum(0)
    value = (
        value
        * (value.square().mean() + 1e-6).rsqrt()
        * env.checkpoint[prefix + ".norm.weight"].float()
    )
    # Native compression applies RoPE in FP32 before its final BF16 store.
    return _rope(value, (end // ratio) * ratio, env.rope).to(history.dtype)


def _hadamard(x):
    matrix = torch.tensor(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
        device="npu",
        dtype=torch.bfloat16,
    )
    return (F.linear(x, matrix) * 0.5).to(x.dtype)


def _quant(x):
    scale = x.float().abs().amax(-1, keepdim=True) / 127
    quant = (
        (x.float() / torch.where(scale > 0, scale, 1))
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
    )
    return quant, scale.half()


def _oracle(env, history, query_positions):
    qr = _rms(_linear(env, "wq_a", history), env.checkpoint["q_norm.weight"])
    q = _rms(_linear(env, "wq_b", qr).reshape(-1, 2, 4))
    kv = _rms(_linear(env, "wkv", history), env.checkpoint["kv_norm.weight"])
    q = torch.stack([_rope(row, p, env.rope) for p, row in enumerate(q)])
    kv = torch.stack([_rope(row, p, env.rope) for p, row in enumerate(kv)])
    compressed = (
        [
            _compress(env, "compressor", history, end)
            for end in range(env.ratio - 1, len(history), env.ratio)
        ]
        if env.ratio
        else []
    )
    index_keys = []
    index_query = None
    if env.ratio == 4:
        index_keys = [
            _quant(_hadamard(_compress(env, "indexer.compressor", history, end)))
            for end in range(3, len(history), 4)
        ]
        iq = _linear(env, "indexer.wq_b", qr).reshape(-1, 2, 4)
        iq = torch.stack(
            [_hadamard(_rope(row, p, env.rope)) for p, row in enumerate(iq)]
        )
        iq, iqscale = _quant(iq)
        iw = (
            _linear(env, "indexer.weights_proj", history) * (4**-0.5 * 2**-0.5)
        ).half()
        index_query = iq, iqscale, iw
    outputs = []
    for position in query_positions:
        rows = list(kv[max(0, position - 7) : position + 1].float())
        count = (position + 1) // env.ratio if env.ratio else 0
        selected = list(range(count))
        if index_query is not None and count:
            iq, iqscale, iw = index_query
            scores = []
            for key, kscale in index_keys[:count]:
                logits = (
                    (
                        iq[position].float()
                        * iqscale[position].float()
                        * (key.float() * kscale.float())
                    )
                    .sum(-1)
                    .relu()
                )
                scores.append((logits * iw[position].float()).sum())
            selected = (
                torch.stack(scores)
                .argsort(descending=True)[: env.cfg.index_topk]
                .tolist()
            )
        rows += [compressed[index].float() for index in selected]
        matrix = torch.stack(rows)
        logits = q[position].float() @ matrix.t() / 2
        denom = (
            logits.exp().sum(-1, keepdim=True)
            + env.checkpoint["attn_sink"].float().exp()[:, None]
        )
        attended = ((logits.exp() / denom) @ matrix).to(history.dtype)
        attended = _rope(attended, position, env.rope, inverse=True).reshape(2, 4)
        weights = env.checkpoint["wo_a.weight"].reshape(2, 2, 4)
        grouped = (
            torch.stack(
                [
                    F.linear(attended[group].float(), weights[group].float())
                    for group in range(2)
                ]
            )
            .to(history.dtype)
            .reshape(1, 4)
        )
        outputs.append(_linear(env, "wo_b", grouped)[0])
    return torch.stack(outputs)


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("chunked", [False, True])
def test_attention_full_chunk_and_decode(make_attention, ratio, chunked):
    env = make_attention(ratio)
    request = _request(env)
    length = 256 if ratio == 128 else 8
    history = _values((length, 8))
    cuts = (
        [0, length - 1, length]
        if not chunked
        else sorted(set([0, 3, 4, length - 1, length]))
    )
    for begin, end in zip(cuts, cuts[1:]):
        count = end - begin
        assert env.manager.allocate_slots(request, count) is not None
        metadata, public = _metadata(env, [request], [count])
        positions = torch.arange(begin, end, device="npu")
        actual = env.module(positions, history[begin:end], metadata=metadata)
        expected = _oracle(env, history[:end], list(range(begin, end)))
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)
        for common, table, slots in public:
            torch.testing.assert_close(common.block_table_tensor, table)
            torch.testing.assert_close(common.slot_mapping, slots)
        assert all(
            torch.count_nonzero(leaf.kv_cache[0]) == 0 for leaf in env.leaves.values()
        )
        request.num_computed_tokens = end


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_checkpoint_names_and_standard_loaders(make_attention, ratio):
    env = make_attention(ratio)
    names = dict(env.module.named_parameters())
    assert {
        "wq_a.weight",
        "q_norm.weight",
        "wq_b.weight",
        "wkv.weight",
        "kv_norm.weight",
        "wo_a.weight",
        "wo_b.weight",
        "attn_sink",
    } <= names.keys()
    assert callable(env.module.wq_a.weight.weight_loader)
    assert env.module.wo_a.weight.shape == (4, 4)
    assert env.module.q_norm_without_weight.weight.device.type == "npu"
    assert "q_norm_without_weight.weight" not in env.module.state_dict()
    assert "q_norm_without_weight.weight" not in names
    assert env.module.swa_cache_layer is env.leaves["swa"]
    if ratio:
        assert env.module.compressor.state_cache is env.leaves["state"]
        assert env.module.compact_cache is env.leaves["compact"]
    if ratio == 4:
        assert env.module.indexer.k_cache is env.leaves["index_key"]
        assert env.module.indexer.scale_cache is env.leaves["index_scale"]
        assert env.module.indexer.compressor.state_cache is env.leaves["index_state"]


@pytest.mark.parametrize("ratio", [0, 4])
def test_mixed_modelslim_qkv_actual_module(make_attention, ratio):
    env = make_attention(ratio, mixed=True)
    request = _request(env)
    assert env.manager.allocate_slots(request, 4) is not None
    metadata, _ = _metadata(env, [request], [4])
    history = _values((4, 8))
    actual = env.module(torch.arange(4, device="npu"), history, metadata=metadata)
    torch.testing.assert_close(
        actual, _oracle(env, history, [0, 1, 2, 3]), rtol=0.02, atol=0.01
    )
    assert env.module.wq_a.weight.dtype == env.module.wq_b.weight.dtype == torch.int8
    assert env.module.wkv.weight.dtype == torch.int8
    assert (
        env.module.wo_a.weight.dtype == env.module.wo_b.weight.dtype == torch.bfloat16
    )
    if ratio == 4:
        assert env.module.indexer.wq_b.weight.dtype == torch.int8


def test_two_requests_do_not_share_attention_history(make_attention):
    env = make_attention(4, topk=1)
    requests = [_request(env, "first"), _request(env, "second")]
    counts = [4, 8]
    for request, count in zip(requests, counts):
        assert env.manager.allocate_slots(request, count) is not None
    metadata, _ = _metadata(env, requests, counts)
    a, b = _values((4, 8)), _values((8, 8), offset=13)
    actual = env.module(
        torch.tensor(list(range(4)) + list(range(8)), device="npu"),
        torch.cat([a, b]),
        metadata=metadata,
    )
    expected = torch.cat(
        [_oracle(env, a, list(range(4))), _oracle(env, b, list(range(8)))]
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)
    assert env.leaves["index_key"].kv_cache.dtype == torch.int8
    assert env.leaves["index_scale"].kv_cache.dtype == torch.float16


def test_attention_changed_input_graph(make_attention):
    import torch_npu  # noqa: F401

    env = make_attention(4)
    request = _request(env)
    assert env.manager.allocate_slots(request, 8) is not None
    metadata, _ = _metadata(env, [request], [8])
    history = _values((8, 8))
    positions = torch.arange(8, device="npu")

    def run():
        return env.module(positions, history, metadata=metadata)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    torch.npu.synchronize()
    previous = output.clone()
    for _ in range(2):
        history.add_(0.03125)
        expected = _oracle(env, history, list(range(8)))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.01)
        assert not torch.equal(output, previous)
        previous = output.clone()


def test_padded_nan_input_does_not_enter_cache(make_attention):
    env = make_attention(4)
    request = _request(env)
    assert env.manager.allocate_slots(request, 3) is not None
    metadata, public = _metadata(env, [request], [3], padded=4)
    history = _values((3, 8))
    padded = torch.cat([history, torch.full_like(history[:1], float("nan"))])
    actual = env.module(
        torch.tensor([0, 1, 2, -1], device="npu"), padded, metadata=metadata
    )
    torch.testing.assert_close(
        actual[:3], _oracle(env, history, [0, 1, 2]), rtol=0.02, atol=0.01
    )
    assert torch.count_nonzero(actual[3]) == 0
    assert torch.isfinite(actual).all()
    for leaf in env.leaves.values():
        assert torch.isfinite(leaf.kv_cache).all()
        assert torch.count_nonzero(leaf.kv_cache[0]) == 0
    for common, table, slots in public:
        torch.testing.assert_close(common.block_table_tensor, table)
        torch.testing.assert_close(common.slot_mapping, slots)


def test_default_forward_context_metadata(make_attention):
    from vllm.forward_context import set_forward_context

    env = make_attention(0)
    request = _request(env)
    assert env.manager.allocate_slots(request, 4) is not None
    metadata, _ = _metadata(env, [request], [4])
    history = _values((4, 8))
    positions = torch.arange(4, device="npu")
    with set_forward_context(metadata, env.config, num_tokens=4):
        actual = env.module(positions, history)
    torch.testing.assert_close(
        actual, _oracle(env, history, [0, 1, 2, 3]), rtol=0.02, atol=0.01
    )
