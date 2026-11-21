# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch

from tests.unit_tests.attention.test_metax_rswa import rswa_config as _rswa_config
from tests.utils.device_utils import get_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        get_backend() != "metax" or not torch.cuda.is_available(),
        reason="MetaX CUDA runtime required",
    ),
]


@pytest.fixture(scope="module")
def mixed_config(tmp_path_factory):
    config = _rswa_config.__wrapped__(tmp_path_factory)
    config.scheduler_config.max_num_seqs = 8
    return config


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("generated", [127, 128, 129, 223])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("causal_available", [False, True])
@torch.inference_mode()
def test_actual_mixed_dispatch_matches_fp32(
    mixed_config, dtype, generated, interleaved, causal_available, monkeypatch
):
    import flaggems_vllm

    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.attention.rswa_attention import RSWAAttention
    from vllm.v1.attention.backend import CommonAttentionMetadata

    from vllm_fl.dispatch.backends.vendor.metax.impl.attention import rswa
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.rswa import (
        MetaXRSWABackend,
    )

    monkeypatch.setattr(rswa, "vendor_causal_available", lambda: causal_available)

    prefixes = [273, 301, 257, 301, 257]
    query_counts = [1, 1, 257, 37, 3]
    lengths = [273 + generated + 1, 301 + generated + 1, 257, 301, 257 + generated + 3]
    order = [2, 0, 3, 1, 4] if interleaved else list(range(5))
    prefixes = [prefixes[i] for i in order]
    query_counts = [query_counts[i] for i in order]
    lengths = [lengths[i] for i in order]
    heads, dim, block = 10, 128, 16
    pages = (max(lengths) + block - 1) // block
    generator = torch.Generator().manual_seed(9317)
    logical = torch.randn((5, pages * block, heads, 2 * dim), generator=generator).to(
        dtype
    )
    qcpu = torch.randn((sum(query_counts), heads, dim), generator=generator).to(dtype)
    permutation = torch.randperm(5 * pages, generator=generator) + 1
    table_cpu = permutation.reshape(5, pages).to(torch.int32)
    cache = torch.full(
        (1 + 5 * pages, heads, block, 2 * dim), 30.0, device="cuda", dtype=dtype
    )
    cache.index_copy_(
        0,
        permutation.cuda(),
        logical.reshape(5 * pages, block, heads, 2 * dim).cuda().transpose(1, 2),
    )
    retired = []
    for row, (count, prefix, length) in enumerate(zip(query_counts, prefixes, lengths)):
        if count != 1:
            continue
        for page in range((prefix + block - 1) // block, (length - 128) // block):
            retired.append(int(table_cpu[row, page]))
            table_cpu[row, page] = 0
    if retired:
        row = next(i for i, count in enumerate(query_counts) if count == 1)
        last = (lengths[row] - 1) // block
        old = int(table_cpu[row, last])
        new = retired[0]
        cache[new].copy_(cache[old])
        cache[old].fill_(-31)
        table_cpu[row, last] = new
    starts = [0]
    for count in query_counts:
        starts.append(starts[-1] + count)
    start_cpu = torch.tensor(starts, dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=start_cpu.cuda(),
        query_start_loc_cpu=start_cpu,
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        num_reqs=5,
        num_actual_tokens=sum(query_counts),
        max_query_len=max(query_counts),
        max_seq_len=max(lengths),
        block_table_tensor=table_cpu.cuda(),
        slot_mapping=torch.full(
            (sum(query_counts),), -1, dtype=torch.int64, device="cuda"
        ),
        rswa_prefix_lens=torch.tensor(prefixes, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor(lengths, dtype=torch.int32),
    )
    expected = []
    start = 0
    for row, (count, prefix, length) in enumerate(zip(query_counts, prefixes, lengths)):
        q = qcpu[start : start + count].float()
        kv = logical[row, :length].float()
        scores = torch.einsum("qhd,khd->hqk", q, kv[..., :dim]) / math.sqrt(dim)
        positions = torch.arange(length - count, length)[:, None]
        keys = torch.arange(length)[None, :]
        mask = (keys <= positions) & ((keys < prefix) | (keys >= positions - 127))
        scores.masked_fill_(~mask[None, :, :], -float("inf"))
        expected.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), kv[..., dim:]))
        start += count
    expected = torch.cat(expected).to(dtype)
    calls = {"prefill": 0, "decode": 0}
    for key, api in [
        ("prefill", "paged_causal_attention"),
        ("decode", "paged_prefix_window_attention"),
    ]:
        original = getattr(flaggems_vllm, api)

        def track(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(flaggems_vllm, api, track)
    name = f"mixed_{dtype}_{generated}_{interleaved}_{causal_available}"
    with set_current_vllm_config(mixed_config):
        layer = RSWAAttention(
            heads,
            dim,
            dim**-0.5,
            num_kv_heads=heads,
            rswa_window=128,
            prefix=name,
            cache_config=mixed_config.cache_config,
        ).cuda()
        assert layer.get_attn_backend() is MetaXRSWABackend
        layer.kv_cache = cache
        builder = MetaXRSWABackend.get_builder_cls()(
            layer.get_kv_cache_spec(mixed_config),
            [name],
            mixed_config,
            torch.device("cuda"),
        )
        meta = builder.build(0, common)
        assert [r.kind for r in meta.query_runs] == (
            ["prefill", "decode", "prefill", "decode", "native"]
            if interleaved
            else ["decode", "prefill", "native"]
        )
        for run in meta.query_runs:
            cu = run.metadata.query_start_loc.cpu().tolist()
            assert cu[0] == 0 and cu[-1] == run.token_end - run.token_start
        before = cache.clone()
        with set_forward_context(
            {name: meta}, mixed_config, num_tokens=sum(query_counts)
        ):
            actual = layer(qcpu.cuda(), None, None).reshape(-1, heads, dim).cpu()
        torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.02)
        torch.testing.assert_close(cache, before, atol=0, rtol=0)
    assert calls == dict(
        prefill=(2 if interleaved else 1) if causal_available else 0,
        decode=0 if interleaved else 1,
    )
    if causal_available:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

        failures = []

        def fail(*args, **kwargs):
            failures.append(True)
            raise RuntimeError("test prefill execution failure")

        def forbidden(*args, **kwargs):
            raise AssertionError("Execution error retried native attention")

        monkeypatch.setattr(flaggems_vllm, "paged_causal_attention", fail)
        monkeypatch.setattr(TritonAttentionImpl, "forward", forbidden)
        with (
            set_current_vllm_config(mixed_config),
            set_forward_context(
                {name: meta}, mixed_config, num_tokens=sum(query_counts)
            ),
            pytest.raises(RuntimeError, match="test prefill execution failure"),
        ):
            layer(qcpu.cuda(), None, None)
        assert failures == [True]


def test_missing_cpu_plan_is_not_copied_from_device():
    from types import SimpleNamespace

    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.rswa import (
        MetaXRSWAMetadataBuilder,
    )

    builder = object.__new__(MetaXRSWAMetadataBuilder)
    builder._split_requests = True
    base = SimpleNamespace(max_query_len=4)
    common = SimpleNamespace(
        query_start_loc_cpu=torch.zeros(3, dtype=torch.int32),
        seq_lens_cpu_upper_bound=None,
        rswa_prefix_lens=torch.ones(2, dtype=torch.int32),
    )
    assert builder._build_query_runs(base, common, object()) == ()


def test_full_mixed_capture_does_not_use_python_plan():
    from types import SimpleNamespace

    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.rswa import (
        MetaXRSWAMetadataBuilder,
    )

    builder = object.__new__(MetaXRSWAMetadataBuilder)
    builder._split_requests = False
    assert (
        builder._build_query_runs(SimpleNamespace(max_query_len=4), None, object())
        == ()
    )


@torch.inference_mode()
def test_normalized_query_offsets_graph_updates():
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.rswa import (
        _normalize_query_starts,
    )

    starts = torch.tensor([0, 3, 11, 29, 31], dtype=torch.int32, device="cuda")
    out = torch.full((3,), -77, dtype=torch.int32, device="cuda")
    _normalize_query_starts[(1,)](starts, out, 1, 3, 3, 128, num_warps=4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _normalize_query_starts[(1,)](starts, out, 1, 3, 3, 128, num_warps=4)
    for values in [[0, 3, 17, 41, 51], [0, 3, 9, 19, 25]]:
        starts.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
        out.fill_(-77)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out.cpu(),
            torch.tensor([0, values[2] - 3, values[3] - 3], dtype=torch.int32),
            atol=0,
            rtol=0,
        )
