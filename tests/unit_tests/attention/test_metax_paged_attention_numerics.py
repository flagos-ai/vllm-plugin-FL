# Copyright (c) 2026 BAAI. All rights reserved.
"""Independent FA2 KV-read checks, including 528-to-16 block subdivision.

Run on the MACA image with --noconftest; this builds real MetaX metadata and
calls the real backend, without loading a model or a distributed process group.
"""

import os
from types import SimpleNamespace

import pytest
import torch


def _reference(query, key, value, scale):
    """Float32 causal GQA oracle; no FlashAttention or SDPA implementation."""
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1).float()
    value = value.repeat_interleave(repeats, dim=1).float()
    logits = torch.einsum("qhd,khd->hqk", query.float(), key) * scale
    qlen, klen = query.shape[0], key.shape[0]
    allowed = torch.arange(klen, device=query.device)[None, :] <= (
        klen - qlen + torch.arange(qlen, device=query.device)[:, None]
    )
    logits.masked_fill_(~allowed[None], -torch.inf)
    return torch.einsum("hqk,khd->qhd", logits.softmax(dim=-1), value).to(query.dtype)


def _metadata_builder(device):
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn import (
        FlashAttentionMetadataBuilder,
    )
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.common import (
        QueryLenSupport,
    )

    builder = object.__new__(FlashAttentionMetadataBuilder)
    builder.reorder_batch_threshold = 1
    builder.query_len_support = QueryLenSupport.UNIFORM
    builder.group_decodes_by_query_len = False
    builder.aot_schedule = False
    builder.aot_sliding_window = (-1, -1)
    builder.use_full_cuda_graph = False
    builder.max_cudagraph_size = None
    builder.max_num_splits = 0
    builder.dcp_world_size = 1
    builder._split_prefill_cu_seqlens_k = torch.zeros(
        3, dtype=torch.int32, device=device
    )
    builder.mm_prefix_query_ranges_np = None
    builder.rswa_window = None
    return builder


def _allocate_real_cache(device, dtype, layout, blocks, block_size, heads, dim):
    """Use native 0.32 allocation, including a second layer and nonzero offset."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.worker.utils import (
        AttentionGroup,
        allocate_kv_cache,
        map_kv_caches_to_kernel_blocks,
    )

    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn import (
        MacaFlashAttentionBackend,
    )

    names = ["untouched", "target"]
    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=heads, head_size=dim, dtype=dtype
    )
    layer_stride = blocks * spec.page_size_bytes
    guard_bytes = 4096
    size = guard_bytes + len(names) * layer_stride + guard_bytes
    placement = KVCacheTensor(
        size=size,
        layers=names,
        layer_stride=layer_stride,
        block_stride=spec.page_size_bytes,
        offset=guard_bytes,
    )
    config = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=[placement],
        kv_cache_groups=[KVCacheGroupSpec(names, spec)],
    )
    caches = allocate_kv_cache(config, device, KVCacheLayout[layout], [16])
    group = AttentionGroup(
        backend=MacaFlashAttentionBackend,
        layer_names=names,
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        kernel_block_size=16,
    )
    mapped = map_kv_caches_to_kernel_blocks(caches, [group])
    cache = mapped["target"]
    assert cache.data_ptr() == caches["target"].data_ptr()
    assert cache.storage_offset() > 0
    cache.fill_(-7)
    mapped["untouched"].fill_(-11)
    raw = torch.empty(0, dtype=torch.int8, device=device).set_(
        cache.untyped_storage(), 0, (size,), (1,)
    )
    raw[:guard_bytes].fill_(12)
    raw[-guard_bytes:].fill_(12)
    return cache, mapped["untouched"], raw, guard_bytes


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["prefill", "decode", "mixed"])
@pytest.mark.parametrize("layout", ["LBNHC", "LBHNC"])
@pytest.mark.parametrize("split_forward", [False, True])
@pytest.mark.parametrize("q_heads,kv_heads,head_size", [(4, 2, 128), (16, 4, 256)])
@torch.inference_mode()
def test_maca_fa2_reads_subdivided_packed_cache_against_float32_oracle(
    monkeypatch, mode, layout, split_forward, q_heads, kv_heads, head_size
):
    if not torch.cuda.is_available():
        pytest.skip("Requires a MACA GPU with flash_attn installed")
    from tests.utils.device_utils import get_backend

    if get_backend() != "metax":
        pytest.skip("MetaX-only paged attention numerics")
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn import (
        FlashAttentionImpl,
    )
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils import fa_utils

    # Diagnostic A/B without changing the production writer preference.
    if os.getenv("METAX_TEST_CACHE_WRITER") == "triton":
        monkeypatch.setattr(fa_utils, "_get_flaggems_cache_writer", lambda: None)

    monkeypatch.setenv("VLLM_METAX_ENABLE_FA_SPLIT_FORWARD", str(int(split_forward)))
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    manager_block_size, kernel_block_size = 528, 16
    subdivisions = manager_block_size // kernel_block_size
    manager_blocks = 4
    kernel_blocks = manager_blocks * subdivisions
    cache, untouched_layer, raw, guard_bytes = _allocate_real_cache(
        device, dtype, layout, manager_blocks, manager_block_size, kv_heads, head_size
    )
    assert cache.shape == (kernel_blocks, kv_heads, kernel_block_size, 2 * head_size)
    query_lens = {"prefill": [7, 5], "decode": [1, 1], "mixed": [1, 7]}[mode]
    context_lens = [17, 531]  # Cross a kernel page and the second manager block.
    seq_lens = [c + q for c, q in zip(context_lens, query_lens)]
    manager_tables = [[3], [1, 0]]  # Non-contiguous physical manager IDs.
    tables_cpu = torch.zeros((2, 2 * subdivisions), dtype=torch.int32)
    queries, new_keys, new_values, expected = [], [], [], []
    all_keys, all_values, all_slots = [], [], []
    scale = head_size**-0.5
    for request, (qlen, seq_len, managers) in enumerate(
        zip(query_lens, seq_lens, manager_tables)
    ):
        expanded = [
            block * subdivisions + sub
            for block in managers
            for sub in range(subdivisions)
        ]
        tables_cpu[request, : len(expanded)] = torch.tensor(expanded, dtype=torch.int32)
        key = torch.randn(seq_len, kv_heads, head_size, dtype=dtype, device=device)
        value = torch.randn_like(key)
        query = torch.randn(qlen, q_heads, head_size, dtype=dtype, device=device)
        all_keys.append(key)
        all_values.append(value)
        all_slots.extend(
            managers[token // manager_block_size] * manager_block_size
            + token % manager_block_size
            for token in range(seq_len)
        )
        queries.append(query)
        new_keys.append(key[-qlen:])
        new_values.append(value[-qlen:])
        expected.append(_reference(query, key, value, scale))
    query = torch.cat(queries)
    key, value = torch.cat(new_keys), torch.cat(new_values)
    expected = torch.cat(expected)
    query_start_cpu = torch.tensor(
        [0, query_lens[0], sum(query_lens)], dtype=torch.int32
    )
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        query_start_loc=query_start_cpu.to(device),
        query_start_loc_cpu=query_start_cpu,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        block_table_tensor=tables_cpu.to(device),
        slot_mapping=torch.full(
            (sum(query_lens),), -1, dtype=torch.long, device=device
        ),
        causal=True,
        mm_req_doc_ranges=None,
        rswa_prefix_lens=None,
    )
    builder = _metadata_builder(device)
    metadata = builder.build(0, common)
    impl = object.__new__(FlashAttentionImpl)
    impl.vllm_flash_attn_version = 2
    impl.attn_type = "decoder"
    impl.head_size = head_size
    impl.num_heads, impl.num_kv_heads = q_heads, kv_heads
    impl.kv_cache_dtype = "auto"
    impl.supports_quant_query_input = False
    impl.dcp_world_size = 1
    impl.scale = scale
    impl.sliding_window = (-1, -1)
    impl.alibi_slopes = None
    impl.logits_soft_cap = 0.0
    impl.sinks = None
    impl.batch_invariant_enabled = False
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0, device=device),
        _k_scale=torch.tensor(1.0, device=device),
        _v_scale=torch.tensor(1.0, device=device),
    )
    # Fill complete request histories through the real selected writer, using
    # manager-block slot IDs as ModelRunnerFL does. No direct cache assignment.
    all_keys, all_values = torch.cat(all_keys), torch.cat(all_values)
    slots = torch.tensor(all_slots, dtype=torch.long, device=device)
    writer = fa_utils._get_flaggems_cache_writer()
    print(
        f"CACHE-WRITER native={fa_utils.has_cuda_or_composite_kernel('_C_cache_ops::reshape_and_cache_flash')} "
        f"bridge={getattr(writer, '__module__', None)}.{getattr(writer, '__qualname__', None)} "
        f"shape={cache.shape} stride={cache.stride()} offset={cache.storage_offset()}"
    )
    impl.do_kv_cache_update(
        layer,
        all_keys,
        all_values,
        cache,
        slots,
    )
    # Check the precise writer/read-address contract before entering FA. This
    # distinguishes cache corruption from an attention arithmetic failure.
    written_key = cache[
        slots // kernel_block_size, :, slots % kernel_block_size, :head_size
    ]
    written_value = cache[
        slots // kernel_block_size, :, slots % kernel_block_size, head_size:
    ]
    torch.cuda.synchronize()
    print(
        f"CACHE-WRITER key_error={(written_key.float() - all_keys.float()).abs().max().item():.6g} "
        f"value_error={(written_value.float() - all_values.float()).abs().max().item():.6g}"
    )
    torch.testing.assert_close(written_key, all_keys, rtol=0, atol=0)
    torch.testing.assert_close(written_value, all_values, rtol=0, atol=0)
    output = torch.empty_like(query)
    impl.forward(layer, query, key, value, cache, metadata, output)
    torch.cuda.synchronize()
    max_error = (output.float() - expected.float()).abs().max().item()
    print(
        f"FA2 {mode=} {layout=} {split_forward=} {q_heads=} {kv_heads=} {head_size=} cache_stride={cache.stride()} max_abs_error={max_error:.6g}"
    )
    torch.testing.assert_close(output, expected, rtol=0.03, atol=0.03)
    torch.testing.assert_close(untouched_layer, torch.full_like(untouched_layer, -11))
    assert (raw[:guard_bytes] == 12).all().item()
    assert (raw[-guard_bytes:] == 12).all().item()

    # Exercise the actual metadata update path after moving every physical page.
    moved_cache = cache.flip(0)
    moved_tables = (kernel_blocks - 1) - common.block_table_tensor
    moved = builder.update_block_table(metadata, moved_tables, common.slot_mapping)
    updated_output = torch.empty_like(output)
    impl.forward(layer, query, key, value, moved_cache, moved, updated_output)
    torch.cuda.synchronize()
    torch.testing.assert_close(updated_output, expected, rtol=0.03, atol=0.03)
