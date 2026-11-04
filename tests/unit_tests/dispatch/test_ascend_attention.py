# Copyright (c) 2026 BAAI. All rights reserved.

import pytest

pytest.importorskip("torch_npu")

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.utils import select_common_block_size

from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendCommonAttentionMetadata,
    AscendMetadata,
)


def test_ascend_backend_resolves_through_upstream_registry():
    backend = AttentionBackendEnum[AscendAttentionBackend.get_name()]
    assert backend.get_class() is AscendAttentionBackend


def test_ascend_attention_advertises_full_decode_graph_support():
    assert (
        AscendAttentionMetadataBuilder.get_cudagraph_support(None, None)
        is AttentionCGSupport.ALWAYS
    )


def test_upstream_creates_ascend_metadata_builder():
    from types import SimpleNamespace

    import torch

    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.utils import AttentionGroup

    spec = FullAttentionSpec(
        block_size=256, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=2048),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False, max_num_seqs=4),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=None,
    )
    group = AttentionGroup(AscendAttentionBackend, ["attention"], spec, 0)
    group.create_metadata_builders(config, torch.device("cpu"), kernel_block_size=128)
    builder = group.get_metadata_builder()

    assert builder.kv_cache_spec.block_size == 128
    assert builder.layer_names == ["attention"]
    assert not builder.supports_update_block_table
    assert group.kernel_block_table.shape == (1, 4, 16)
    assert group.kernel_block_offsets.tolist() == [0, 1]


def test_ascend_phase_survives_upstream_kernel_metadata_mapping():
    from dataclasses import replace

    import torch

    metadata = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 3]),
        query_start_loc_cpu=torch.tensor([0, 1, 3]),
        seq_lens=torch.tensor([5, 8]),
        num_reqs=2,
        num_actual_tokens=3,
        max_query_len=2,
        max_seq_len=8,
        block_table_tensor=torch.tensor([[1], [2]], dtype=torch.int32),
        slot_mapping=torch.tensor([128, 256, 257]),
        attn_state=AscendAttentionState.ChunkedPrefill,
        _seq_lens_cpu=torch.tensor([5, 8]),
        _num_computed_tokens_cpu=torch.tensor([4, 6]),
    )
    # vLLM 0.32 AttentionGroup replaces the common metadata when virtual
    # kernel blocks need a mapped block table. Dynamic attributes are lost.
    mapped = replace(metadata, block_table_tensor=metadata.block_table_tensor * 2)
    assert mapped.attn_state is AscendAttentionState.ChunkedPrefill
    assert mapped._seq_lens_cpu is metadata._seq_lens_cpu
    unpadded = mapped.unpadded(num_actual_tokens=1, num_actual_reqs=1)
    assert unpadded.attn_state is AscendAttentionState.ChunkedPrefill
    assert unpadded.num_reqs == unpadded.num_actual_tokens == 1
    assert unpadded._seq_lens_cpu.tolist() == [5]
    assert unpadded._num_computed_tokens_cpu.tolist() == [4]
    assert unpadded.block_table_tensor.tolist() == [[2]]


@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_rejects_blocks_smaller_than_ascend_kernel(block_size):
    assert not AscendAttentionBackend.supports_block_size(block_size)
    assert AscendAttentionBackend.get_preferred_block_size(block_size) == 128


@pytest.mark.parametrize("manager_block_size", [128, 256, 384])
def test_hybrid_blocks_use_128_token_kernel_blocks(manager_block_size):
    import torch

    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = FullAttentionSpec(
        block_size=manager_block_size,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )
    assert AscendAttentionBackend.supports_block_size(manager_block_size)
    assert AscendAttentionBackend.get_supported_kernel_block_sizes(spec) == [128]
    assert (
        select_common_block_size(manager_block_size, [AscendAttentionBackend], [spec])
        == 128
    )


@pytest.mark.parametrize("state", list(AscendAttentionState))
@pytest.mark.parametrize("kv_heads", [2, 8])
def test_fia_preserves_cache_blocks_and_model_kv_width(state, kv_heads):
    import torch

    impl = object.__new__(AscendAttentionBackendImpl)
    impl.key_cache = torch.zeros(3, 128, kv_heads, 128)
    impl.value_cache = torch.ones_like(impl.key_cache)
    key = torch.zeros(5, kv_heads, 128)
    value = torch.ones_like(key)
    metadata = AscendMetadata(
        attn_state=state,
        seq_lens=torch.tensor([130, 3]),
        seq_lens_list=[130, 3],
        actual_seq_lengths_q=[2, 5],
        block_tables=torch.tensor([[0, 1], [2, 0]], dtype=torch.int32),
    )
    k, v, block_size, block_table, lengths = impl._get_fia_params(key, value, metadata)

    assert block_size == 128
    if state == AscendAttentionState.PrefillNoCache:
        assert k is key and v is value
        assert block_table is None
        assert lengths == [2, 5]
    else:
        assert k.shape == v.shape == (3, 128, kv_heads * 128)
        assert k.data_ptr() == impl.key_cache.data_ptr()
        assert v.data_ptr() == impl.value_cache.data_ptr()
        torch.testing.assert_close(block_table, metadata.block_tables)
        assert lengths == [130, 3]


@pytest.mark.gpu
@pytest.mark.parametrize("via_forward", [False, True])
def test_hybrid_kv_cache_updates_shared_storage(via_forward):
    from types import SimpleNamespace

    import torch

    from vllm.forward_context import override_forward_context
    from vllm.v1.attention.backend import AttentionType

    impl = object.__new__(AscendAttentionBackendImpl)
    impl.key_cache = impl.value_cache = None
    # vLLM hybrid models physically interleave K/V blocks.
    backing = torch.zeros(3, 2, 128, 2, 128, dtype=torch.bfloat16, device="npu")
    kv_cache = backing.permute(1, 0, 2, 3, 4)
    assert not kv_cache[0].is_contiguous()
    key = (
        torch.arange(3 * 2 * 128, device="npu", dtype=torch.float32)
        .view(3, 2, 128)
        .to(torch.bfloat16)
    )
    value = -key
    metadata = AscendMetadata(
        num_actual_tokens=3,
        slot_mapping=torch.tensor([0, 131, -1], dtype=torch.int32, device="npu"),
    )
    if via_forward:
        impl.attn_type = AttentionType.DECODER
        impl.forward_impl = lambda *args: args[-1]
        impl.forward(
            SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0),
            key,
            key,
            value,
            kv_cache,
            metadata,
            output=torch.empty_like(key),
        )
    else:
        impl.reshape_and_cache(key, value, kv_cache, metadata)
    torch.testing.assert_close(backing[0, 0, 0].cpu(), key[0].cpu())
    torch.testing.assert_close(backing[1, 0, 3].cpu(), key[1].cpu())
    torch.testing.assert_close(backing[0, 1, 0].cpu(), value[0].cpu())
    torch.testing.assert_close(backing[1, 1, 3].cpu(), value[1].cpu())
    assert backing[2].cpu().count_nonzero().item() == 0
    impl.num_heads = impl.num_kv_heads = 2
    impl.scale = 128**-0.5
    output = torch.empty_like(key[:1])
    with override_forward_context(SimpleNamespace(capturing=False)):
        impl.forward_paged_attention(
            torch.zeros_like(output),
            AscendMetadata(
                seq_lens=torch.tensor([1], dtype=torch.int32),
                block_tables=torch.tensor([[0]], dtype=torch.int32, device="npu"),
            ),
            output,
        )
    torch.testing.assert_close(output[0].cpu(), value[0].cpu())


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("slot_dtype", ["int32", "int64"])
def test_interleaved_kv_cache_graph_replays_shared_storage(dtype, slot_dtype):
    import torch

    dtype, slot_dtype = getattr(torch, dtype), getattr(torch, slot_dtype)
    impl = object.__new__(AscendAttentionBackendImpl)
    impl.key_cache = impl.value_cache = None
    backing = torch.zeros(3, 2, 128, 2, 128, dtype=dtype, device="npu")
    kv_cache = backing.permute(1, 0, 2, 3, 4)
    key = torch.zeros(3, 2, 128, dtype=dtype, device="npu")
    value = torch.zeros_like(key)
    metadata = AscendMetadata(
        num_actual_tokens=3,
        slot_mapping=torch.full((3,), -1, dtype=slot_dtype, device="npu"),
    )
    impl.reshape_and_cache(key, value, kv_cache, metadata)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        impl.reshape_and_cache(key, value, kv_cache, metadata)

    expected = torch.zeros(backing.shape, dtype=dtype)
    for step, slots in enumerate(
        ([0, 131, -1], [-1, 0, 257], [-1, -1, -1], [129, -1, -1])
    ):
        cpu_key = torch.arange(3 * 2 * 128).view(3, 2, 128).to(dtype) + step
        key.copy_(cpu_key)
        value.copy_(-cpu_key)
        metadata.slot_mapping.copy_(torch.tensor(slots, dtype=slot_dtype))
        graph.replay()
        torch.npu.synchronize()
        for token, slot in enumerate(slots):
            if slot >= 0:
                block, offset = divmod(slot, 128)
                expected[block, 0, offset] = cpu_key[token]
                expected[block, 1, offset] = -cpu_key[token]
        torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)


@pytest.mark.gpu
def test_native_planar_graph_padding_never_writes_dummy_cache():
    """Negative write slots stay inactive even when null block zero is nonzero."""
    import torch

    impl = object.__new__(AscendAttentionBackendImpl)
    impl.key_cache = impl.value_cache = None
    cache = torch.full((2, 3, 128, 2, 128), 7, dtype=torch.bfloat16, device="npu")
    assert cache[0].is_contiguous() and cache[1].is_contiguous()
    key = torch.zeros(4, 2, 128, dtype=torch.bfloat16, device="npu")
    value = torch.zeros_like(key)
    metadata = AscendMetadata(
        num_actual_tokens=4,
        slot_mapping=torch.full((4,), -1, dtype=torch.int64, device="npu"),
    )
    impl.reshape_and_cache(key, value, cache, metadata)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        impl.reshape_and_cache(key, value, cache, metadata)
    expected = torch.full(cache.shape, 7, dtype=cache.dtype)
    for step, slots in enumerate(
        ([128, 129, 130, -1], [256, 257, -1, -1], [131, -1, -1, -1], [-1] * 4)
    ):
        cpu_key = torch.arange(4 * 2 * 128).view(4, 2, 128).to(cache.dtype) + step
        # Inactive queries deliberately carry unrelated values. Their slots,
        # rather than zero-valued query/cache contents, suppress cache writes.
        cpu_key[-1].fill_(1000 + step)
        key.copy_(cpu_key)
        value.copy_(-cpu_key)
        metadata.slot_mapping.copy_(torch.tensor(slots, dtype=torch.int64))
        graph.replay()
        torch.npu.synchronize()
        for token, slot in enumerate(slots):
            if slot >= 0:
                block, offset = divmod(slot, 128)
                expected[0, block, offset] = cpu_key[token]
                expected[1, block, offset] = -cpu_key[token]
        torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
        assert torch.all(cache[:, 0].cpu() == 7)
        assert metadata.num_actual_tokens == 4
        assert metadata.slot_mapping.cpu().tolist() == slots
