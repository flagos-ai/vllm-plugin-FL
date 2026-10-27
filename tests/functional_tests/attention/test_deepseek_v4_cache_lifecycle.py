# SPDX-License-Identifier: Apache-2.0
"""Real scheduler allocations with installed NPU cache-layer components."""

from math import prod
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(current_platform.device_type != "npu", reason="Requires Ascend"),
]


@pytest.mark.parametrize("mode_name", ["NONE", "PIECEWISE"])
def test_non_full_graph_binding_skips_cache_factory(mode_name):
    from vllm.config import CUDAGraphMode

    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    calls = []

    class RaisingBackend:
        @staticmethod
        def get_impl_cls():
            calls.append("factory")
            raise NotImplementedError("cache-only backend")

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode[mode_name])
    )
    controller = GraphRuntimeController(vllm_config=config)
    controller.bind_attention_groups([[SimpleNamespace(backend=RaisingBackend)]])
    assert controller.attention_impls == ()
    assert calls == []


@pytest.mark.parametrize("mode_name", ["FULL", "FULL_DECODE_ONLY"])
@pytest.mark.parametrize("factory_kind", ["valid", "missing_updater", "raises"])
def test_full_graph_binding_validates_factory(mode_name, factory_kind):
    from vllm.config import CUDAGraphMode

    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    calls = []

    class Impl:
        @staticmethod
        def update_graph_params(*args):
            pass

    class Backend:
        @staticmethod
        def get_impl_cls():
            calls.append("factory")
            if factory_kind == "raises":
                raise NotImplementedError("cache-only backend")
            return Impl if factory_kind == "valid" else object

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode[mode_name])
    )
    controller = GraphRuntimeController(vllm_config=config)
    groups = [[SimpleNamespace(backend=Backend), SimpleNamespace(backend=Backend)]]
    if factory_kind == "valid":
        controller.bind_attention_groups(groups)
        assert controller.attention_impls == (Impl,)
    else:
        message = (
            "lacks update_graph_params"
            if factory_kind == "missing_updater"
            else "Failed to resolve"
        )
        with pytest.raises(RuntimeError, match=message) as error:
            controller.bind_attention_groups(groups)
        if factory_kind == "raises":
            assert isinstance(error.value.__cause__, NotImplementedError)
    assert calls == ["factory"]


@pytest.mark.parametrize("mode_name", ["FULL", "FULL_DECODE_ONLY"])
def test_full_graph_binding_cache_provider_needs_no_task_updater(
    mode_name, monkeypatch
):
    from vllm.config import CUDAGraphMode

    from vllm_fl.attention.deepseek_v4 import DeepseekV4CacheBackend
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    calls = []

    def forbidden_factory():
        calls.append("cache_factory")
        raise AssertionError("Cache-only FULL execution must not resolve an impl")

    monkeypatch.setattr(
        DeepseekV4CacheBackend, "get_impl_cls", staticmethod(forbidden_factory)
    )
    assert DeepseekV4CacheBackend.requires_graph_task_updates is False
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode[mode_name])
    )
    controller = GraphRuntimeController(vllm_config=config)
    cache_group = SimpleNamespace(backend=DeepseekV4CacheBackend)
    controller.bind_attention_groups([[cache_group]])
    assert controller.attention_impls == ()
    assert calls == []

    class OrdinaryBackend:
        @staticmethod
        def get_impl_cls():
            calls.append("ordinary_factory")
            return object

    # A cache-only group must not suppress another backend's FULL validation.
    with pytest.raises(RuntimeError, match="lacks update_graph_params"):
        controller.bind_attention_groups(
            [[cache_group, SimpleNamespace(backend=OrdinaryBackend)]]
        )
    assert calls == ["ordinary_factory"]


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


def _specs():
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

    # Match the planner's page-size coverage contract even with toy head widths.
    # Padded pages also exercise the actual strided-view reshape/write path.
    common = dict(
        block_size=128,
        num_kv_heads=1,
        model_version="deepseek_v4",
        page_size_padded=8192,
    )
    return {
        "state4": SlidingWindowMLASpec(
            **common, head_size=16, dtype=torch.float32, sliding_window=8
        ),
        "state128": SlidingWindowMLASpec(
            **common, head_size=8, dtype=torch.float32, sliding_window=128
        ),
        "swa": SlidingWindowMLASpec(
            **common, head_size=4, dtype=torch.bfloat16, sliding_window=8
        ),
        "compact4": MLAAttentionSpec(
            **common, head_size=4, dtype=torch.bfloat16, compress_ratio=4
        ),
        "compact128": MLAAttentionSpec(
            **common, head_size=4, dtype=torch.bfloat16, compress_ratio=128
        ),
        "key4": MLAAttentionSpec(
            **common, head_size=4, dtype=torch.int8, compress_ratio=4
        ),
        "scale4": MLAAttentionSpec(
            **common, head_size=1, dtype=torch.float16, compress_ratio=4
        ),
    }


@pytest.fixture
def pools(runtime):
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_groups,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
    from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache

    from vllm_fl.kv_cache.deepseek_v4 import DeepseekV4CacheLayer
    from vllm_fl.worker.kv_cache_utils import bind_kv_cache
    from vllm_fl.worker.model_runner import ModelRunnerFL

    specs = {f"model.layers.0.{key}": spec for key, spec in _specs().items()}
    groups = get_kv_cache_groups(runtime, specs)
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            KVCacheTensor(size=64 * spec.page_size_bytes, shared_by=[name])
            for name, spec in specs.items()
        ],
    )
    owner = SimpleNamespace(
        device=torch.device("npu:0"),
        runner_only_attn_layers=set(),
        use_ascend_hybrid_cache=False,
    )
    raw = ModelRunnerFL._allocate_kv_cache_tensors(owner, config)
    views = {
        name: _reshape_attention_kv_cache(
            raw[name],
            spec,
            (64, spec.storage_block_size, spec.head_size),
            (0, 1, 2),
            64,
            None,
        )
        for name, spec in specs.items()
    }
    manager = KVCacheManager(
        generate_scheduler_kv_cache_config([config]),
        max_model_len=512,
        scheduler_block_size=128,
        hash_block_size=128,
        max_num_batched_tokens=512,
        enable_caching=False,
    )
    context = runtime.compilation_config.static_forward_context
    original_names = set(context)
    leaves = {}
    requests = []
    initial = manager.block_pool.get_num_free_blocks()
    try:
        for name, spec in specs.items():
            leaves[name] = DeepseekV4CacheLayer(name, spec, vllm_config=runtime)
        runner_caches = []
        bind_kv_cache(views, leaves, runner_caches)
        yield SimpleNamespace(
            config=runtime,
            specs=specs,
            groups=groups,
            views=views,
            raw=raw,
            leaves=leaves,
            runner_caches=runner_caches,
            manager=manager,
            requests=requests,
            initial=initial,
        )
    finally:
        try:
            for request in requests:
                manager.free(request)
            assert manager.block_pool.get_num_free_blocks() == initial
        finally:
            for name in set(context) - original_names:
                context.pop(name, None)


def _request(env, name="request-0"):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    request = Request(name, list(range(256)), SamplingParams(max_tokens=8), None)
    env.requests.append(request)
    return request


def _allocate(env, request, count):
    assert env.manager.allocate_slots(request, num_new_tokens=count) is not None


def _common(env, request, leaf_name, count, *, padded=None):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    name = "model.layers.0." + leaf_name
    index = next(i for i, group in enumerate(env.groups) if name in group.layer_names)
    physical = env.manager.get_blocks(request.request_id).get_block_ids()[index]
    # IDs come exclusively from the real scheduler manager. Zero is its null page.
    pages = list(physical) + [0] * max(0, 4 - len(physical))
    end = request.num_computed_tokens + count
    qpad = count if padded is None else padded
    slots = []
    for position in range(request.num_computed_tokens, end):
        page = pages[position // 128]
        slots.append(page * 128 + position % 128 if page else -1)
    slots.extend([-1] * (qpad - count))
    starts = torch.tensor([0, count], dtype=torch.int32)
    seq = torch.tensor([end], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=starts.to("npu"),
        query_start_loc_cpu=starts,
        seq_lens=seq.to("npu"),
        num_reqs=1,
        num_actual_tokens=qpad,
        max_query_len=count,
        max_seq_len=end,
        block_table_tensor=torch.tensor([pages], device="npu", dtype=torch.int32),
        slot_mapping=torch.tensor(slots, device="npu", dtype=torch.int64),
        _seq_lens_cpu=seq,
    ), pages


def _builder(env, leaf_name):
    from vllm_fl.attention.deepseek_v4 import DeepseekV4MetadataBuilder

    name = "model.layers.0." + leaf_name
    return DeepseekV4MetadataBuilder(
        env.specs[name], [name], env.config, torch.device("npu:0")
    )


def _common_snapshot(common):
    names = (
        "query_start_loc",
        "query_start_loc_cpu",
        "seq_lens",
        "_seq_lens_cpu",
        "_num_computed_tokens_cpu",
        "block_table_tensor",
        "slot_mapping",
    )
    return {
        name: getattr(common, name).clone()
        for name in names
        if getattr(common, name) is not None
    }


def _assert_common_unchanged(common, snapshot):
    for name, expected in snapshot.items():
        torch.testing.assert_close(getattr(common, name), expected, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("capture", [False, True])
def test_dummy_plan_is_inert_and_real_build_restores_private_buffers(
    pools, ratio, capture, monkeypatch
):
    from vllm.v1.attention.backend import AttentionCGSupport

    request = _request(pools)
    count = ratio + 1
    _allocate(pools, request, count)
    common, _ = _common(pools, request, f"compact{ratio}", count, padded=count + 2)
    common._num_computed_tokens_cpu = torch.tensor([0], dtype=torch.int32)
    builder = _builder(pools, f"compact{ratio}")
    assert (
        builder.get_cudagraph_support(pools.config, builder.kv_cache_spec)
        == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    real = builder.build(0, common)
    tensors = {
        name: tensor
        for name, tensor in vars(real).items()
        if isinstance(tensor, torch.Tensor)
    }
    expected = {name: tensor.clone() for name, tensor in tensors.items()}
    layout = {
        name: (tensor.data_ptr(), tensor.shape, tensor.stride())
        for name, tensor in tensors.items()
    }
    cache_bytes = {name: raw.clone() for name, raw in pools.raw.items()}
    # Dummy construction must not validate synthetic/stale computed counts.
    common._num_computed_tokens_cpu.fill_(777)
    public = _common_snapshot(common)
    calls = []
    original_dummy = builder.build_for_dummy

    def record_dummy(*args, **kwargs):
        calls.append("dummy")
        return original_dummy(*args, **kwargs)

    monkeypatch.setattr(builder, "build_for_dummy", record_dummy)
    dummy = (
        builder.build_for_cudagraph_capture(common)
        if capture
        else builder.build_for_dummy(common)
    )
    assert calls == ["dummy"]
    assert dummy.num_completed == 0
    for name, previous in layout.items():
        tensor = getattr(dummy, name)
        assert (tensor.data_ptr(), tensor.shape, tensor.stride()) == previous
        fill = 0 if name in ("seq_lens", "key_lengths") else -1
        assert torch.all(tensor == fill), name
    # The full reusable capacities, not only the returned bucket views, are PAD.
    for name, tensor in builder.buffers.items():
        assert torch.all(tensor == (0 if name == "seq_lens" else -1)), name
    assert torch.all(builder.slots == -1)
    assert torch.all(builder.output_slots == -1)
    assert torch.all(builder.key_lengths == 0)
    _assert_common_unchanged(common, public)
    for name, raw in pools.raw.items():
        torch.testing.assert_close(raw, cache_bytes[name], rtol=0, atol=0)

    common._num_computed_tokens_cpu.zero_()
    public = _common_snapshot(common)
    restored = builder.build(0, common)
    assert restored.num_completed == 1
    for name, previous in layout.items():
        tensor = getattr(restored, name)
        assert (tensor.data_ptr(), tensor.shape, tensor.stride()) == previous
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    _assert_common_unchanged(common, public)


@pytest.mark.parametrize("ratio", [4, 128])
def test_zero_end_padded_request_is_inactive_but_active_counts_are_validated(
    pools, ratio
):
    request = _request(pools)
    _allocate(pools, request, ratio)
    common, _ = _common(pools, request, f"compact{ratio}", ratio, padded=ratio + 2)
    common.num_reqs = 2
    common.query_start_loc_cpu = torch.tensor([0, ratio, ratio + 1], dtype=torch.int32)
    common.query_start_loc = common.query_start_loc_cpu.to("npu")
    common._seq_lens_cpu = torch.tensor([ratio, 0], dtype=torch.int32)
    common.seq_lens = common._seq_lens_cpu.to("npu")
    common._num_computed_tokens_cpu = torch.tensor([0, 777], dtype=torch.int32)
    # A padded row may retain a real-looking page ID; private planning hides it.
    common.block_table_tensor = common.block_table_tensor.repeat(2, 1)
    public = _common_snapshot(common)
    builder = _builder(pools, f"compact{ratio}")
    meta = builder.build(0, common)
    assert meta.num_completed == 1
    assert torch.all(meta.request_ids[:ratio] == 0)
    torch.testing.assert_close(
        meta.positions[:ratio], torch.arange(ratio, device="npu"), rtol=0, atol=0
    )
    for name in ("request_ids", "positions", "slot_mapping"):
        assert torch.all(getattr(meta, name)[ratio:] == -1), name
    assert meta.output_request_ids[0] == 0
    assert meta.output_positions[0] == ratio - 1
    assert meta.output_slots[0] >= 0
    for name in ("output_request_ids", "output_positions", "output_slots"):
        assert torch.all(getattr(meta, name)[1:] == -1), name
    assert torch.all(meta.block_table[1] == -1)
    assert meta.seq_lens[1] == 0
    assert meta.key_lengths[1] == 0
    _assert_common_unchanged(common, public)

    # Ignore stale computed counts only for the zero-end inactive row, never
    # for a scheduled request (including a negative active computed count).
    common._num_computed_tokens_cpu[0] = -1
    public = _common_snapshot(common)
    with pytest.raises(ValueError, match="Inconsistent exact CPU scheduler lengths"):
        builder.build(0, common)
    _assert_common_unchanged(common, public)


def test_scheduler_groups_typed_views_and_binding(pools):
    key_group = next(
        group for group in pools.groups if "model.layers.0.key4" in group.layer_names
    )
    assert "model.layers.0.scale4" in key_group.layer_names
    assert len(pools.runner_caches) == len(pools.leaves)
    for name, leaf in pools.leaves.items():
        spec = pools.specs[name]
        assert leaf.kv_cache is pools.views[name]
        assert leaf.get_kv_cache_spec(pools.config) is spec
        assert pools.views[name].shape == (
            64,
            128 // spec.compress_ratio,
            spec.head_size,
        )
        assert pools.views[name].dtype == spec.dtype
        assert (
            pools.views[name].untyped_storage().data_ptr()
            == pools.raw[name].untyped_storage().data_ptr()
        )
        assert any(view is pools.views[name] for view in pools.runner_caches)


@pytest.mark.parametrize("ratio", [4, 128])
def test_scheduler_completed_row_arithmetic_and_private_metadata(pools, ratio):
    request = _request(pools)
    builder = _builder(pools, f"compact{ratio}")
    for count in (ratio - 1, 1, ratio):
        _allocate(pools, request, count)
        common, pages = _common(pools, request, f"compact{ratio}", count)
        public_table, public_slots = (
            common.block_table_tensor.clone(),
            common.slot_mapping.clone(),
        )
        meta = builder.build(0, common)
        start, end = request.num_computed_tokens, request.num_computed_tokens + count
        complete = list(range(start // ratio, end // ratio))
        assert meta.num_completed == len(complete)
        expected_ends = [(index + 1) * ratio - 1 for index in complete]
        expected_slots = [
            pages[(index * ratio) // 128] * (128 // ratio) + index % (128 // ratio)
            for index in complete
        ]
        torch.testing.assert_close(
            meta.output_positions[: len(complete)],
            torch.tensor(
                expected_ends, device="npu", dtype=meta.output_positions.dtype
            ),
        )
        torch.testing.assert_close(
            meta.output_slots[: len(complete)],
            torch.tensor(expected_slots, device="npu", dtype=meta.output_slots.dtype),
        )
        assert torch.all(meta.output_positions[len(complete) :] == -1)
        assert torch.all(meta.output_slots[len(complete) :] == -1)
        torch.testing.assert_close(
            meta.key_lengths[0],
            torch.tensor(end // ratio, device="npu"),
            check_dtype=False,
        )
        torch.testing.assert_close(common.block_table_tensor, public_table)
        torch.testing.assert_close(common.slot_mapping, public_slots)
        expected_table = torch.tensor(
            [page if page else -1 for page in pages],
            device="npu",
            dtype=meta.block_table.dtype,
        )
        torch.testing.assert_close(meta.block_table[0, : len(pages)], expected_table)
        assert meta.block_table.data_ptr() != common.block_table_tensor.data_ptr()
        request.num_computed_tokens = end


def test_two_scheduler_requests_keep_completed_slots_isolated(pools):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    first, second = _request(pools, "first"), _request(pools, "second")
    _allocate(pools, first, 4)
    _allocate(pools, second, 8)
    ca, pa = _common(pools, first, "compact4", 4)
    cb, pb = _common(pools, second, "compact4", 8)
    starts = torch.tensor([0, 4, 12], dtype=torch.int32)
    lengths = torch.tensor([4, 8], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=starts.to("npu"),
        query_start_loc_cpu=starts,
        seq_lens=lengths.to("npu"),
        num_reqs=2,
        num_actual_tokens=12,
        max_query_len=8,
        max_seq_len=8,
        block_table_tensor=torch.cat([ca.block_table_tensor, cb.block_table_tensor]),
        slot_mapping=torch.cat([ca.slot_mapping, cb.slot_mapping]),
        _seq_lens_cpu=lengths,
    )
    metadata = _builder(pools, "compact4").build(0, common)
    assert metadata.num_completed == 3
    torch.testing.assert_close(
        metadata.output_request_ids[:3],
        torch.tensor([0, 1, 1], device="npu", dtype=metadata.output_request_ids.dtype),
    )
    torch.testing.assert_close(
        metadata.output_positions[:3],
        torch.tensor([3, 3, 7], device="npu", dtype=metadata.output_positions.dtype),
    )
    torch.testing.assert_close(
        metadata.output_slots[:3],
        torch.tensor(
            [pa[0] * 32, pb[0] * 32, pb[0] * 32 + 1],
            device="npu",
            dtype=metadata.output_slots.dtype,
        ),
    )
    assert pa[0] != pb[0]


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int8]
)
def test_cache_write_typed_padding_and_null_page(pools, dtype):
    from vllm_fl.dispatch import call_op, get_default_manager

    # The Triton scatter_nd_update_v2 implementation is the Ascend vendor
    # implementation of this op and is selected by the normal priority policy
    # (no environment gating).
    assert (
        get_default_manager().get_selected_impl_id("deepseek_v4_cache_write")
        == "vendor.ascend"
    )
    cache = torch.zeros(3, 4, 4, device="npu", dtype=dtype)
    values = torch.arange(12, device="npu").reshape(3, 4).to(dtype)
    if dtype.is_floating_point:
        values[-1].fill_(float("nan"))
    call_op(
        "deepseek_v4_cache_write", values, torch.tensor([4, 9, -1], device="npu"), cache
    )
    torch.testing.assert_close(cache[1, 0], values[0])
    torch.testing.assert_close(cache[2, 1], values[1])
    assert torch.count_nonzero(cache[0]) == 0


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int8]
)
@pytest.mark.parametrize("strided", [False, True])
def test_cache_write_padding_and_strides_graph_replay(dtype, strided):
    from vllm_fl.dispatch.backends.reference.impl.deepseek_v4 import (
        deepseek_v4_cache_write,
    )

    shape = (5, 4, 8) if strided else (3, 4, 4)
    backing = torch.arange(prod(shape), device="npu").to(dtype).reshape(shape)
    cache = backing[1:4, :, ::2] if strided else backing
    original = backing.cpu()
    values = torch.arange(16, device="npu").reshape(4, 4).to(dtype)
    slots = torch.tensor([0, -1, 9, 99], device="npu")
    deepseek_v4_cache_write(values, slots, cache)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        deepseek_v4_cache_write(values, slots, cache)
    for raw_slots in ([0, -1, 9, 99], [-1, 4, 11, -1], [-1, -1, -1, -1]):
        backing.copy_(original)
        slots.copy_(torch.tensor(raw_slots, device="npu"))
        values.copy_(torch.arange(16, device="npu").reshape(4, 4).to(dtype) + 20)
        if dtype.is_floating_point:
            for index, slot in enumerate(raw_slots):
                if not 0 <= slot < 12:
                    values[index].fill_(float("nan"))
            cache[0, 0].fill_(float("nan"))
        expected = backing.cpu()
        expected_cache = expected[1:4, :, ::2] if strided else expected
        host_values = values.cpu()
        for index, slot in enumerate(raw_slots):
            if 0 <= slot < 12:
                expected_cache[slot // 4, slot % 4].copy_(host_values[index])
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            backing.cpu(), expected, rtol=0, atol=0, equal_nan=True
        )


def test_cache_write_workspace_is_independent_of_cache_capacity():
    from vllm_fl.dispatch.backends.reference.impl.deepseek_v4 import (
        deepseek_v4_cache_write,
    )

    values = torch.ones(2, 128, dtype=torch.bfloat16, device="npu")
    slots = torch.tensor([0, -1], device="npu")
    small = torch.zeros(2, 128, 128, dtype=torch.bfloat16, device="npu")
    deepseek_v4_cache_write(values, slots, small)
    large = torch.zeros(1024, 128, 128, dtype=torch.bfloat16, device="npu")
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    deepseek_v4_cache_write(values, slots, large)
    torch.npu.synchronize()
    assert torch.npu.max_memory_allocated() - before < 8 * 1024 * 1024
    torch.testing.assert_close(large[0, 0], values[0], rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["seq_lens", "query_start"])
def test_missing_exact_cpu_metadata_rejected(pools, missing):
    request = _request(pools)
    _allocate(pools, request, 3)
    common, _ = _common(pools, request, "compact4", 3)
    if missing == "seq_lens":
        common._seq_lens_cpu = None
    else:
        common.query_start_loc_cpu = None
    with pytest.raises((ValueError, RuntimeError), match="CPU|cpu"):
        _builder(pools, "compact4").build(0, common)


def test_duplicate_prefix_and_invalid_spec_rejected(pools):
    from vllm_fl.kv_cache.deepseek_v4 import DeepseekV4CacheLayer

    name = "model.layers.0.compact4"
    with pytest.raises((ValueError, RuntimeError)):
        DeepseekV4CacheLayer(name, pools.specs[name], vllm_config=pools.config)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        DeepseekV4CacheLayer(
            "model.layers.0.invalid", object(), vllm_config=pools.config
        )


def _compressor(ratio):
    from vllm_fl.models.deepseek_v4 import DeepseekV4Compressor

    module = DeepseekV4Compressor(8, 4, ratio).to("npu")
    for name, param in module.named_parameters():
        values = (
            torch.arange(param.numel(), device="npu").reshape(param.shape) % 23 - 11
        ).float() * 0.017
        param.copy_(
            values.to(param.dtype) if name != "norm.weight" else values.abs() + 0.7
        )
    return module


def _compression_oracle(module, history, end, ratio, rope):
    coff = 2 if ratio == 4 else 1
    projected = F.linear(history.float(), module.wkv.weight.float())
    gates = F.linear(history.float(), module.wgate.weight.float())
    values, logits = [], []
    for time in range(max(0, end - ratio * coff + 1), end + 1):
        half = int(coff == 2 and time >= end - ratio + 1)
        values.append(projected[time, half * 4 : (half + 1) * 4])
        logits.append(
            (gates[time] + module.ape[time % ratio])[half * 4 : (half + 1) * 4]
        )
    result = (torch.stack(values) * torch.stack(logits).softmax(0)).sum(0)
    result = (
        result * (result.square().mean() + 1e-6).rsqrt() * module.norm.weight.float()
    )
    c, s = rope[(end // ratio) * ratio]
    return torch.stack(
        [
            result[0],
            result[1],
            result[2] * c - result[3] * s,
            result[3] * c + result[2] * s,
        ]
    ).to(history.dtype)


@pytest.mark.parametrize("ratio", [4, 128])
def test_scheduler_compressor_chunk_decode_and_store(pools, ratio):
    request = _request(pools)
    module = _compressor(ratio)
    state_leaf = pools.leaves[f"model.layers.0.state{ratio}"]
    compact_leaf = pools.leaves[f"model.layers.0.compact{ratio}"]
    state_builder, compact_builder = (
        _builder(pools, f"state{ratio}"),
        _builder(pools, f"compact{ratio}"),
    )
    history = (
        (torch.arange(2 * ratio * 8, device="npu").reshape(2 * ratio, 8) % 29 - 14)
        * 0.03
    ).bfloat16()
    angle = torch.arange(2 * ratio, device="npu").float() * 0.013
    rope = torch.stack([angle.cos(), angle.sin()], -1)
    for count in (ratio - 1, 1, ratio):
        _allocate(pools, request, count)
        state_common, _ = _common(pools, request, f"state{ratio}", count)
        compact_common, _ = _common(pools, request, f"compact{ratio}", count)
        smeta, cmeta = (
            state_builder.build(0, state_common),
            compact_builder.build(0, compact_common),
        )
        start, end = request.num_computed_tokens, request.num_computed_tokens + count
        output = module.forward_from_cache(
            history[start:end], state_leaf, compact_leaf, smeta, cmeta, rope
        )
        completed = [(j + 1) * ratio - 1 for j in range(start // ratio, end // ratio)]
        for row, boundary in enumerate(completed):
            expected = _compression_oracle(module, history, boundary, ratio, rope)
            torch.testing.assert_close(output[row], expected, rtol=0.01, atol=0.01)
            slot = int(cmeta.output_slots[row].item())
            cache = compact_leaf.kv_cache
            torch.testing.assert_close(
                cache[slot // cache.shape[1], slot % cache.shape[1]],
                expected,
                rtol=0.01,
                atol=0.01,
            )
        assert torch.count_nonzero(state_leaf.kv_cache[0]) == 0
        assert torch.count_nonzero(compact_leaf.kv_cache[0]) == 0
        request.num_computed_tokens = end


def test_stable_plans_rebuild_and_changed_input_graph(pools):
    request = _request(pools)
    module = _compressor(4)
    state_leaf, compact_leaf = (
        pools.leaves["model.layers.0." + name] for name in ("state4", "compact4")
    )
    sb, cb = _builder(pools, "state4"), _builder(pools, "compact4")
    _allocate(pools, request, 3)
    scommon, _ = _common(pools, request, "state4", 3)
    ccommon, _ = _common(pools, request, "compact4", 3)
    rope = torch.tensor([[1.0, 0.0]] * 8, device="npu")
    x0 = torch.full((3, 8), 0.1, device="npu", dtype=torch.bfloat16)
    module.forward_from_cache(
        x0, state_leaf, compact_leaf, sb.build(0, scommon), cb.build(0, ccommon), rope
    )
    request.num_computed_tokens = 3
    _allocate(pools, request, 1)
    scommon, _ = _common(pools, request, "state4", 1, padded=4)
    ccommon, _ = _common(pools, request, "compact4", 1, padded=4)
    smeta, cmeta = sb.build(0, scommon), cb.build(0, ccommon)
    pointers = (
        smeta.positions.data_ptr(),
        smeta.slot_mapping.data_ptr(),
        cmeta.output_positions.data_ptr(),
        cmeta.output_slots.data_ptr(),
        smeta.block_table.data_ptr(),
        cmeta.block_table.data_ptr(),
    )
    x = torch.full((4, 8), 0.2, device="npu", dtype=torch.bfloat16)

    def run():
        return module.forward_from_cache(
            x, state_leaf, compact_leaf, smeta, cmeta, rope
        )

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    request.num_computed_tokens = 4
    _allocate(pools, request, 4)
    scommon, _ = _common(pools, request, "state4", 4, padded=4)
    ccommon, _ = _common(pools, request, "compact4", 4, padded=4)
    updated_s, updated_c = sb.build(0, scommon), cb.build(0, ccommon)
    assert pointers == (
        updated_s.positions.data_ptr(),
        updated_s.slot_mapping.data_ptr(),
        updated_c.output_positions.data_ptr(),
        updated_c.output_slots.data_ptr(),
        updated_s.block_table.data_ptr(),
        updated_c.block_table.data_ptr(),
    )
    assert int(updated_c.output_positions[0].item()) == 7
    for step in range(2):
        x.copy_(
            (
                (torch.arange(32, device="npu").reshape(4, 8) % 13 - 6) * 0.03
                + step * 0.05
            ).bfloat16()
        )
        expected = run().clone()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
        assert torch.count_nonzero(compact_leaf.kv_cache[0]) == 0
