# Copyright (c) 2026 BAAI. All rights reserved.
"""Default R-SWA dispatch, paged attention and cache insertion on C550."""

import json
from types import SimpleNamespace

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        get_backend() != "metax" or not torch.cuda.is_available(),
        reason="MetaX GPU and MACA runtime required",
    ),
]


@pytest.fixture(scope="module")
def rswa_config(tmp_path_factory):
    from vllm.engine.arg_utils import EngineArgs

    model = tmp_path_factory.mktemp("rswa-config")
    text = {
        "hidden_size": 1280,
        "num_attention_heads": 10,
        "num_key_value_heads": 10,
        "num_hidden_layers": 12,
        "max_position_embeddings": 32768,
        "use_mla": False,
        "q_lora_rank": None,
        "kv_lora_rank": None,
        "qk_nope_head_dim": 0,
        "qk_rope_head_dim": 0,
        "v_head_dim": 128,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "num_experts_per_tok": 6,
        "first_k_dense_replace": 1,
    }
    config = {
        "model_type": "unlimited_ocr",
        "architectures": ["UnlimitedOCRForCausalLM"],
        "language_config": text,
        "candidate_resolutions": [[1024, 1024]],
        "torch_dtype": "bfloat16",
    }
    (model / "config.json").write_text(json.dumps(config))
    return EngineArgs(
        model=str(model),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=640,
        max_num_seqs=4,
        max_num_batched_tokens=640,
        block_size=16,
        enforce_eager=True,
        enable_prefix_caching=False,
    ).create_engine_config()


@pytest.mark.parametrize("generated", [127, 128, 129, 223])
@pytest.mark.parametrize("reuse_pages", [False, True])
@torch.inference_mode()
def test_default_rswa_keeps_prefix_and_masks_evicted_decode(
    rswa_config, generated, reuse_pages, monkeypatch
):
    import flaggems_vllm

    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.attention.rswa_attention import RSWAAttention
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.rswa import (
        MetaXRSWABackend,
        vendor_rswa_available,
    )

    available = vendor_rswa_available()
    calls = []
    if available:
        original = flaggems_vllm.paged_prefix_window_attention

        def track(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(flaggems_vllm, "paged_prefix_window_attention", track)

    device, dtype = torch.device("cuda"), torch.bfloat16
    prefixes, window, block_size, heads, dim = (273, 301), 128, 16, 10, 128
    ends = [p + generated for p in prefixes]
    seq_lens = [end + 1 for end in ends]
    pages = (max(seq_lens) + block_size - 1) // block_size
    tables = [[1 + 2 * page + row for page in range(pages)] for row in range(2)]
    freed = []
    if reuse_pages:
        for row, (prefix, end) in enumerate(zip(prefixes, ends)):
            # Only complete pages beyond the prefix and before the window retire.
            for page in range((prefix + 15) // 16, (end - window + 1) // 16):
                freed.append(tables[row][page])
                tables[row][page] = 0
        if freed:
            tables[1][ends[1] // block_size] = freed[0]

    query = torch.zeros((2, heads, dim), device=device, dtype=dtype)
    cache = torch.zeros(
        (1 + 2 * pages, heads, block_size, 2 * dim), device=device, dtype=dtype
    )
    values, slots = [], []
    reference = torch.zeros((2, heads, dim), dtype=torch.float32)
    for row, (prefix, end) in enumerate(zip(prefixes, ends)):
        count = prefix + min(window, generated + 1)
        for position in range(end + 1):
            value = torch.zeros(heads * dim, dtype=dtype)
            value[position] = row + 1
            values.append(value.reshape(heads, dim))
            page = tables[row][position // block_size]
            slots.append(page * block_size + position % block_size if page else -1)
            if position < prefix or end - position < window:
                assert page != 0
                reference[row].flatten()[position] = (row + 1) / count
    value = torch.stack(values).to(device)
    key = torch.zeros_like(value)
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)
    starts_cpu = torch.tensor([0, 1, 2], dtype=torch.int32)
    metadata = CommonAttentionMetadata(
        query_start_loc=starts_cpu.to(device),
        query_start_loc_cpu=starts_cpu,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.tensor(tables, dtype=torch.int32, device=device),
        slot_mapping=torch.full((2,), -1, dtype=torch.int64, device=device),
        rswa_prefix_lens=torch.tensor(prefixes, dtype=torch.int32),
    )
    name = f"rswa_{generated}_{reuse_pages}"
    with set_current_vllm_config(rswa_config):
        layer = RSWAAttention(
            heads,
            dim,
            dim**-0.5,
            num_kv_heads=heads,
            rswa_window=window,
            prefix=name,
            cache_config=rswa_config.cache_config,
        ).to(device)
        expected_backend = MetaXRSWABackend if available else TritonAttentionBackend
        assert layer.get_attn_backend() is expected_backend
        layer.kv_cache = cache
        builder = layer.get_attn_backend().get_builder_cls()(
            layer.get_kv_cache_spec(rswa_config), [name], rswa_config, device
        )
        attention_metadata = builder.build(0, metadata)
        assert attention_metadata.rswa_prefix_lens.tolist() == list(prefixes)
        # Populate through the actual Triton backend's installed writer.
        layer.impl.do_kv_cache_update(layer, key, value, cache, slot_mapping)
        with set_forward_context({name: attention_metadata}, rswa_config, num_tokens=2):
            result = layer(query, None, None).reshape(2, heads, dim).float().cpu()
        torch.testing.assert_close(result, reference, atol=2e-5, rtol=1e-2)
        if available:
            assert len(calls) == 1
        for row, prefix in enumerate(prefixes):
            assert torch.all(result[row].flatten()[:prefix] > 0)
            assert (
                torch.count_nonzero(
                    result[row].flatten()[prefix : ends[row] - window + 1]
                )
                == 0
            )


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@torch.inference_mode()
def test_triton_kv_writer_replays_updated_inputs(layout):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    heads, dim, block_size = 10, 128, 16
    device, dtype = torch.device("cuda"), torch.bfloat16
    impl = TritonAttentionImpl(heads, dim, dim**-0.5, heads, None, None, "auto")
    layer = SimpleNamespace(
        _k_scale=torch.ones((), device=device), _v_scale=torch.ones((), device=device)
    )
    shape = (
        (4, heads, block_size, 2 * dim)
        if layout == "HND"
        else (4, block_size, heads, 2 * dim)
    )
    cache = torch.full(shape, -99, dtype=dtype, device=device)
    if layout == "NHD":
        cache = cache.transpose(1, 2)
    slots = torch.tensor([16, 31, 32, -1, 47, 60], dtype=torch.int64, device=device)
    key = torch.zeros((6, heads, dim), dtype=dtype, device=device)
    value = torch.zeros_like(key)
    impl.do_kv_cache_update(layer, key, value, cache, slots)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        impl.do_kv_cache_update(layer, key, value, cache, slots)
    for epoch in (1, 2):
        key.fill_(epoch)
        value.fill_(-epoch)
        slots.copy_(
            torch.tensor([0, 15, 33, -1, 46, 61], dtype=slots.dtype, device=device)
        )
        cache.fill_(-99)
        expected = torch.full(cache.shape, -99, dtype=dtype)
        for slot in (0, 15, 33, 46, 61):
            page, offset = divmod(slot, block_size)
            expected[page, :, offset, :dim] = epoch
            expected[page, :, offset, dim:] = -epoch
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(cache.cpu(), expected, atol=0, rtol=0)
