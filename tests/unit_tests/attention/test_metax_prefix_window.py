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

from types import SimpleNamespace

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        get_backend() != "metax" or not torch.cuda.is_available(),
        reason="MetaX CUDA runtime required",
    ),
]


@pytest.mark.parametrize("available", [False, True])
def test_selector_keeps_native_when_provider_is_missing(monkeypatch, available):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_fl.dispatch.backends.vendor.metax.impl.attention import rswa
    from vllm_fl.dispatch.backends.vendor.metax.metax import MacaBackend

    monkeypatch.setattr(rswa, "vendor_rswa_available", lambda: available)
    result = MacaBackend().attention_backend(use_rswa=True)
    expected = (
        "vllm_fl.dispatch.backends.vendor.metax.impl.attention.rswa.MetaXRSWABackend"
        if available
        else AttentionBackendEnum.TRITON_ATTN.get_path()
    )
    assert result == expected
    assert (
        MacaBackend().attention_backend() == AttentionBackendEnum.FLASH_ATTN.get_path()
    )


def test_rswa_rejects_mla_before_execution():
    from vllm_fl.dispatch.backends.vendor.metax.metax import MacaBackend

    with pytest.raises(NotImplementedError, match="dense MHA"):
        MacaBackend().attention_backend(use_mla=True, use_rswa=True)


def test_metadata_capacity_rejected_before_staging():
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.rswa import (
        MetaXRSWAMetadataBuilder,
    )

    builder = object.__new__(MetaXRSWAMetadataBuilder)
    builder._prefix_counts = torch.empty(4)
    with pytest.raises(ValueError, match="request capacity"):
        builder.build(0, SimpleNamespace(num_reqs=5))


def test_single_request_uses_native_before_vendor_work(monkeypatch):
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.rswa import MetaXRSWAImpl

    calls = []
    sentinel = object()

    def native_forward(*args, **kwargs):
        calls.append(True)
        return sentinel

    monkeypatch.setattr(TritonAttentionImpl, "forward", native_forward)
    impl = object.__new__(MetaXRSWAImpl)
    impl.attn_type = AttentionType.DECODER
    meta = SimpleNamespace(max_query_len=1, num_actual_tokens=1)
    assert impl.forward(None, None, None, None, None, meta, None) is sentinel
    assert calls == [True]


@torch.inference_mode()
def test_disjoint_metadata_graph_replay_updates_early_and_late_positions():
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.rswa import (
        _disjoint_prefix_counts,
    )

    prefixes = torch.tensor([273, 301], dtype=torch.int32, device="cuda")
    lengths = torch.empty_like(prefixes)
    counts = torch.full_like(prefixes, -7)
    lengths.copy_(prefixes)
    _disjoint_prefix_counts[(1,)](prefixes, lengths, counts, 2, 128, 128, num_warps=4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _disjoint_prefix_counts[(1,)](
            prefixes, lengths, counts, 2, 128, 128, num_warps=4
        )
    for generated in [0, 1, 127, 128, 129, 223]:
        values = [273 + generated + 1, 301 + generated + 1]
        lengths.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
        counts.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.tensor(
            [min(p, max(l - 128, 0)) for p, l in zip([273, 301], values)],
            dtype=torch.int32,
        )
        torch.testing.assert_close(counts.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(
            prefixes.cpu(), torch.tensor([273, 301], dtype=torch.int32), atol=0, rtol=0
        )
