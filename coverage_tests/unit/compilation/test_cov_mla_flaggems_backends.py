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

"""CPU tests for the FlagGems dense, prefill and sparse MLA adapters.

The FlagGems kernels (``flash_attn_varlen_func``, ``flash_mla``,
``flash_mla_sparse_fwd``, ``contiguous_copy``, ``pad_attention_heads``) are
the external boundary: recorders here return closed-form tensors so the
argument plumbing and slicing/padding contracts are asserted against
explicit references.  The upstream ``MLACommonImpl`` constructor (whose
signature differs between vLLM builds) is stubbed because these tests target
the plugin's post-construction validation and forward logic.  The sparse
physical-index layout conversion is checked against a plain-python reference.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# The shim import is load-bearing: it must precede vllm_fl/vllm imports.
# isort: split
from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)

# isort: split
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType

import vllm_fl.dispatch.backends.flaggems.impl.mla as mla


def _ensure_provider_symbols():
    """Give the selected FlagGems provider module the two attention symbols.

    On the A100 image ``flaggems_vllm`` supplies them; on hosts where it is
    absent (or where the inert placeholder owns ``flag_gems``) the symbols
    are installed as raise-if-called placeholders.  Every test below
    replaces them with a recorder before use.
    """
    import importlib

    names = ("flash_attn_varlen_func", "flash_mla")
    for module_name in ("flaggems_vllm", "flag_gems"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            module = types.ModuleType(module_name)
            sys.modules[module_name] = module
        for name in names:
            if not hasattr(module, name):

                def _unset(*args, __name=module_name + "." + name, **kwargs):
                    raise AssertionError(f"{__name} must be faked by the test")

                setattr(module, name, _unset)


_ensure_provider_symbols()


class _ConcreteMLAFL(mla.MLAFLImpl):
    """Concrete subclass: this vLLM image leaves forward_mqa abstract."""

    def forward_mqa(self, *args, **kwargs):
        raise AssertionError("dense MLA forward_mqa is not exercised here")

    def forward_mha(self, *args, **kwargs):
        raise AssertionError("dense MLA forward_mha is not exercised here")


class _Kernel:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if callable(self.result):
            return self.result(*args, **kwargs)
        return self.result


def _dense_impl(monkeypatch, **overrides):
    """Construct MLAFLImpl with the upstream base constructor stubbed."""

    def base_init(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        logits_soft_cap,
        attn_type,
        kv_sharing_target_layer_name,
        **mla_args,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank = mla_args.get("kv_lora_rank")
        self.mla_args = mla_args

    monkeypatch.setattr(mla.MLACommonImpl, "__init__", base_init)
    kwargs = {
        "num_heads": 2,
        "head_size": 64,
        "scale": 0.125,
        "num_kv_heads": 1,
        "alibi_slopes": None,
        "sliding_window": None,
        "kv_cache_dtype": "auto",
        "logits_soft_cap": None,
        "attn_type": AttentionType.DECODER,
        "kv_sharing_target_layer_name": None,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
    }
    kwargs.update(overrides)
    return _ConcreteMLAFL(**kwargs)


def test_dense_backend_identity():
    """The dense adapter advertises MLAFL and returns its implementation."""
    assert mla.MLAFLBackend.get_name() == "MLAFL"
    assert mla.MLAFLBackend.get_impl_cls() is mla.MLAFLImpl


@pytest.mark.parametrize(
    "override",
    [
        {"alibi_slopes": [0.5]},
        {"sliding_window": 32},
        {"logits_soft_cap": 1.0},
    ],
)
def test_dense_impl_rejects_unsupported_features(monkeypatch, override):
    """alibi/sliding-window/soft-cap raise the documented error."""
    with pytest.raises(NotImplementedError, match="does not support"):
        _dense_impl(monkeypatch, **override)


def test_dense_impl_rejects_encoder_attention(monkeypatch):
    """Only decoder self-attention is implemented."""
    with pytest.raises(NotImplementedError, match="encoder/decoder cross-attention"):
        _dense_impl(monkeypatch, attn_type="encoder")


def test_dense_impl_rejects_quantized_kv_cache(monkeypatch):
    """FP8 KV-cache storage is not supported by the dense adapter."""
    monkeypatch.setattr(
        mla, "is_quantized_kv_cache", lambda dtype: str(dtype).startswith("fp8")
    )
    with pytest.raises(NotImplementedError, match="FP8 KV cache"):
        _dense_impl(monkeypatch, kv_cache_dtype="fp8")


def test_dense_impl_keeps_valid_configuration(monkeypatch):
    """A supported configuration is stored without further validation errors."""
    impl = _dense_impl(monkeypatch)
    assert impl.kv_cache_dtype == "auto"
    assert impl.kv_lora_rank == 8
    assert impl.scale == 0.125


def test_dense_flash_attn_pads_value_heads_and_unpacks_lse(monkeypatch):
    """A narrower v head is zero-padded; tuple results yield the LSE pair."""
    kernel = _Kernel(
        result=lambda *a, **k: (
            torch.full((3, 2, 8), 7.0),
            torch.full((3, 2), 1.5),
        )
    )
    monkeypatch.setattr(mla, "flash_attn_varlen_func", kernel)
    impl = object.__new__(_ConcreteMLAFL)
    impl._pad_v = True
    q = torch.zeros(3, 2, 8)
    v = torch.zeros(5, 2, 4)
    out, lse = impl._flash_attn_varlen_diff_headdims(
        q,
        torch.zeros(5, 2, 8),
        v,
        return_softmax_lse=True,
        softmax_scale=0.25,
        cu_seqlens_q="cq",
        cu_seqlens_k="ck",
        max_seqlen_q=3,
        max_seqlen_k=5,
    )
    kwargs = kernel.calls[0][1]
    assert kwargs["v"].shape == (5, 2, 8)
    assert kwargs["return_softmax_lse"] is True
    assert kwargs["softmax_scale"] == 0.25
    assert kwargs["cu_seqlens_q"] == "cq"
    assert out.shape == (3, 2, 8)
    assert torch.all(lse == 1.5)


def test_dense_flash_attn_without_padding_returns_the_provider_tensor(monkeypatch):
    """Without padding the provider tensor is returned (tuple LSE dropped)."""
    output = torch.full((3, 2, 8), 4.0)
    kernel = _Kernel(result=(output, torch.zeros(3, 2)))
    monkeypatch.setattr(mla, "flash_attn_varlen_func", kernel)
    impl = object.__new__(_ConcreteMLAFL)
    impl._pad_v = False
    v = torch.zeros(5, 2, 8)
    result = impl._flash_attn_varlen_diff_headdims(
        torch.zeros(3, 2, 8),
        torch.zeros(5, 2, 8),
        v,
        return_softmax_lse=False,
    )
    assert result is output
    assert kernel.calls[0][1]["v"] is v
    assert kernel.calls[0][1]["return_softmax_lse"] is False


def _decode_metadata():
    return SimpleNamespace(
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            seq_lens=torch.tensor([6, 5], dtype=torch.int32),
        )
    )


def test_dense_forward_decode_uses_page_size_and_metadata(monkeypatch):
    """flash_mla receives the unsqueezed cache, page size, batch and metadata."""
    output = torch.full((2, 2, 8), 9.0)
    kernel = _Kernel(result=output)
    monkeypatch.setattr(mla, "flash_mla", kernel)
    impl = object.__new__(_ConcreteMLAFL)
    impl.kv_cache_dtype = "auto"
    q = torch.zeros(2, 2, 8)
    cache = torch.zeros(4, 3, 8)
    metadata = _decode_metadata()
    out, lse = mla.MLAFLImpl._forward_decode(impl, q, cache, metadata, None)
    args = kernel.calls[0][0]
    assert out is output
    assert lse.shape == (2, 2) and torch.all(lse == 0)
    assert args[0] is q
    assert args[1] is metadata.decode.block_table
    assert args[2].shape == (4, 3, 1, 8)
    assert args[3] is None
    assert args[4] == 3  # PAGE_SIZE
    assert args[5] == 2  # batch
    assert args[6] == 1
    assert args[7] is metadata.decode.seq_lens
    assert args[8] == 2  # q heads
    assert args[9] is None
    assert args[11] == 0  # dense q: no separate v head dim
    assert args[12] is True


def test_dense_forward_decode_concatenates_tuple_query(monkeypatch):
    """A (q_nope, q_pe) query is concatenated on the last dimension."""
    kernel = _Kernel(result=torch.zeros(2, 2, 8))
    monkeypatch.setattr(mla, "flash_mla", kernel)
    impl = object.__new__(_ConcreteMLAFL)
    impl.kv_cache_dtype = "auto"
    q_nope = torch.full((2, 2, 4), 1.0)
    q_pe = torch.full((2, 2, 4), 2.0)
    cache = torch.zeros(2, 2, 8)
    mla.MLAFLImpl._forward_decode(impl, (q_nope, q_pe), cache, _decode_metadata(), None)
    joined = kernel.calls[0][0][0]
    assert joined.shape == (2, 2, 8)
    assert torch.all(joined[..., :4] == 1.0)
    assert torch.all(joined[..., 4:] == 2.0)


def test_dense_forward_decode_rejects_fp8_and_empty_inputs(monkeypatch):
    """FP8 storage and empty caches/decode metadata are rejected."""
    kernel = _Kernel(result=torch.zeros(1))
    monkeypatch.setattr(mla, "flash_mla", kernel)
    impl = object.__new__(_ConcreteMLAFL)
    impl.kv_cache_dtype = "fp8"
    cache = torch.zeros(2, 2, 8)
    with pytest.raises(NotImplementedError, match="FP8 MLA"):
        mla.MLAFLImpl._forward_decode(
            impl, torch.zeros(1, 1, 8), cache, _decode_metadata(), None
        )
    impl.kv_cache_dtype = "auto"
    with pytest.raises(AssertionError):
        mla.MLAFLImpl._forward_decode(
            impl, torch.zeros(1, 1, 8), torch.zeros(0), _decode_metadata(), None
        )
    with pytest.raises(AssertionError):
        mla.MLAFLImpl._forward_decode(
            impl,
            torch.zeros(1, 1, 8),
            cache,
            SimpleNamespace(decode=None),
            None,
        )


def test_sparse_cudagraph_support_is_never():
    """Both the helper and the builder/backend report eager-only support."""
    support, reason = mla.sparse_mla_cudagraph_support(576)
    assert support is AttentionCGSupport.NEVER
    assert "eager attention boundary" in reason
    spec = SimpleNamespace(head_size=576)
    assert (
        mla.FlagGemsSparseMLAMetadataBuilder.get_cudagraph_support(None, spec)
        is AttentionCGSupport.NEVER
    )
    assert (
        mla.FlagGemsSparseMLABackend.get_cudagraph_support(None, spec)
        is AttentionCGSupport.NEVER
    )


def _builder():
    spec = SimpleNamespace(block_size=64, head_size=512)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(index_topk=2048)),
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    return mla.FlagGemsSparseMLAMetadataBuilder(
        spec, ["layer.0"], config, torch.device("cpu")
    )


def test_sparse_metadata_builder_allocates_buffer_and_builds_ids():
    """Per-token request ids repeat each request's scheduled segment length."""
    builder = _builder()
    assert builder.topk_tokens == 2048
    assert builder.req_id_per_token_buffer.shape == (8,)
    assert builder.req_id_per_token_buffer.dtype == torch.int32
    common = SimpleNamespace(
        query_start_loc_cpu=np.array([0, 2, 5], dtype=np.int32),
        num_reqs=2,
        max_query_len=3,
        max_seq_len=9,
        num_actual_tokens=5,
        query_start_loc=torch.tensor([0, 2, 5]),
        slot_mapping=torch.arange(5),
        block_table_tensor=torch.zeros(2, 4, dtype=torch.int32),
    )
    metadata = builder.build(0, common)
    assert metadata.req_id_per_token.tolist() == [0, 0, 1, 1, 1]
    assert metadata.num_reqs == 2
    assert metadata.max_query_len == 3
    assert metadata.max_seq_len == 9
    assert metadata.num_actual_tokens == 5
    assert metadata.query_start_loc is common.query_start_loc
    assert metadata.slot_mapping is common.slot_mapping
    assert metadata.block_table is common.block_table_tensor
    assert metadata.block_size == 64
    assert metadata.topk_tokens == 2048


def test_sparse_metadata_builder_handles_empty_segments():
    """A build with no scheduled tokens leaves the request-id buffer zeroed."""
    builder = _builder()
    builder.req_id_per_token_buffer.fill_(7)
    common = SimpleNamespace(
        query_start_loc_cpu=np.array([0, 0], dtype=np.int32),
        num_reqs=1,
        max_query_len=0,
        max_seq_len=4,
        num_actual_tokens=0,
        query_start_loc=torch.tensor([0, 0]),
        slot_mapping=torch.empty(0, dtype=torch.int64),
        block_table_tensor=torch.zeros(1, 2, dtype=torch.int32),
    )
    metadata = builder.build(0, common)
    assert metadata.req_id_per_token.numel() == 0
    assert torch.all(builder.req_id_per_token_buffer == 0)


def test_sparse_backend_class_contract():
    """Names, dtypes, head sizes, block sizes and sparse/MLA flags are fixed."""
    backend = mla.FlagGemsSparseMLABackend
    assert backend.get_name() == "FLAGGEMS_MLA_SPARSE"
    assert backend.get_builder_cls() is mla.FlagGemsSparseMLAMetadataBuilder
    assert backend.get_impl_cls() is mla.FlagGemsSparseMLAImpl
    assert backend.get_supported_head_sizes() == [512, 576]
    assert backend.get_supported_kernel_block_sizes() == [64]
    assert backend.is_mla() is True
    assert backend.is_sparse() is True
    assert backend.supports_compute_capability(SimpleNamespace(major=7)) is True
    assert backend.supported_dtypes == [torch.bfloat16]
    assert backend.supported_kv_cache_dtypes == ["auto", "bfloat16"]
    assert backend.get_kv_cache_shape(4, 64, 1, 512) == (4, 64, 512)
    with pytest.raises(ValueError, match="one latent KV head"):
        backend.get_kv_cache_shape(4, 64, 2, 512)


def _physical_reference(request_ids, block_table, token_indices, block_size):
    rows = []
    lengths = []
    for row in range(request_ids.shape[0]):
        valid = []
        for column in range(token_indices.shape[1]):
            token = int(token_indices[row, column])
            block = token // block_size if token >= 0 else -1
            if token < 0 or block >= block_table.shape[1]:
                continue
            physical_block = int(block_table[row, block])
            if physical_block < 0:
                continue
            valid.append(physical_block * block_size + token % block_size)
        width = token_indices.shape[1]
        rows.append(valid + [-1] * (width - len(valid)))
        lengths.append(len(valid))
    return torch.tensor(rows, dtype=torch.int32), torch.tensor(
        lengths, dtype=torch.int32
    )


def test_physical_index_conversion_compacts_valid_slots():
    """Invalid blocks sink to the tail; later valid tokens move forward."""
    request_ids = torch.tensor([0, 1])
    block_table = torch.tensor([[0, 1, 2], [2, -1, 0]], dtype=torch.int32)
    token_indices = torch.tensor([[0, 1, 4], [2, 3, 5]], dtype=torch.int32)
    compact, lengths = mla._convert_request_to_physical_indices(
        request_ids, block_table, token_indices, 2
    )
    expected, expected_lengths = _physical_reference(
        request_ids, block_table, token_indices, 2
    )
    assert compact.dtype == torch.int32
    assert compact.tolist() == expected.tolist()
    assert lengths.tolist() == expected_lengths.tolist() == [3, 1]


def test_physical_index_conversion_rejects_negative_tokens():
    """Negative and out-of-range logical tokens compact to the -1 sink."""
    request_ids = torch.tensor([0])
    block_table = torch.tensor([[1, 2]], dtype=torch.int32)
    token_indices = torch.tensor([[-1, 3, 9]], dtype=torch.int32)
    compact, lengths = mla._convert_request_to_physical_indices(
        request_ids, block_table, token_indices, 2
    )
    expected, expected_lengths = _physical_reference(
        request_ids, block_table, token_indices, 2
    )
    assert compact.tolist() == expected.tolist()
    assert lengths.tolist() == expected_lengths.tolist() == [1]


def _sparse_impl(monkeypatch, **overrides):
    monkeypatch.setattr(mla, "_resolve_sparse_mla_kernels", lambda: ("writer", "attn"))
    kwargs = {
        "num_heads": 2,
        "head_size": 8,
        "scale": 0.5,
        "num_kv_heads": 1,
        "alibi_slopes": None,
        "sliding_window": None,
        "kv_cache_dtype": "auto",
        "logits_soft_cap": None,
        "attn_type": "decoder",
        "kv_sharing_target_layer_name": None,
        "topk_indices_buffer": torch.zeros(4, 3, dtype=torch.int32),
        "indexer": None,
        "kv_lora_rank": 8,
    }
    kwargs.update(overrides)
    return mla.FlagGemsSparseMLAImpl(**kwargs)


def test_sparse_impl_rejects_quantized_cache_and_binds_indexer_buffer(monkeypatch):
    """Only auto/BF16 storage is accepted; the indexer buffer takes precedence."""
    with pytest.raises(NotImplementedError, match="BF16 KV cache"):
        _sparse_impl(monkeypatch, kv_cache_dtype="fp8")
    indexer_buffer = torch.ones(2, 2, dtype=torch.int32)
    impl = _sparse_impl(
        monkeypatch,
        indexer=SimpleNamespace(topk_indices_buffer=indexer_buffer),
    )
    assert impl.topk_indices_buffer is indexer_buffer
    assert impl.kv_lora_rank == 8
    assert impl.softmax_scale == 0.5
    assert impl._cache_writer == "writer"
    assert impl._sparse_attn == "attn"


def test_sparse_cache_update_skips_empty_cache_and_normalizes_dtype():
    """An empty cache is a no-op; bfloat16 asks the writer for auto storage."""
    impl = object.__new__(mla.FlagGemsSparseMLAImpl)
    calls = []
    impl._cache_writer = lambda *a, **k: calls.append((a, k))
    kv_cache = torch.zeros(0)
    impl.do_kv_cache_update(
        torch.zeros(1, 8), torch.zeros(1, 1, 4), kv_cache, torch.zeros(1), "auto", None
    )
    assert calls == []

    kv_cache = torch.zeros(2, 4, 8)
    kv_c = torch.ones(3, 8)
    k_pe = torch.ones(3, 1, 4)
    slots = torch.tensor([[0, 2, 5]])
    scale = torch.ones(1)
    impl.do_kv_cache_update(kv_c, k_pe, kv_cache, slots, "bfloat16", scale)
    args, kwargs = calls[0]
    assert args[0] is kv_c
    assert args[1].shape == (3, 4)
    assert args[2] is kv_cache
    assert args[3].tolist() == [0, 2, 5]
    assert kwargs["kv_cache_dtype"] == "auto"
    assert kwargs["scale"] is scale


def _flaggems_module(monkeypatch):
    module = sys.modules.get("flaggems_vllm")
    if module is None:
        module = types.ModuleType("flaggems_vllm")
        monkeypatch.setitem(sys.modules, "flaggems_vllm", module)
    return module


def test_sparse_forward_mqa_tensor_query_pads_heads_and_slices(monkeypatch):
    """A tensor query pads to 64 heads, compacts indices, slices the output."""
    provider = _flaggems_module(monkeypatch)
    sparse_calls = []

    def contiguous_copy(cache):
        sparse_calls.append(("copy", cache.shape))
        return cache

    def pad_heads(q, padded):
        sparse_calls.append(("pad", q.shape[1], padded))
        return torch.zeros(q.shape[0], padded, q.shape[2])

    def sparse_attn(q, cache, indices, scale, **kwargs):
        sparse_calls.append(("attn", q.shape, cache.shape, indices, scale, kwargs))
        return (
            torch.arange(q.shape[0] * q.shape[1] * kwargs["d_v"]).view(
                q.shape[0], q.shape[1], kwargs["d_v"]
            ),
            None,
        )

    monkeypatch.setattr(provider, "contiguous_copy", contiguous_copy)
    monkeypatch.setattr(provider, "pad_attention_heads", pad_heads)
    impl = object.__new__(mla.FlagGemsSparseMLAImpl)
    impl.kv_lora_rank = 4
    impl.softmax_scale = 0.25
    impl.topk_indices_buffer = torch.tensor([[0, 1, 4], [2, 3, 5]], dtype=torch.int32)
    impl._sparse_attn = sparse_attn
    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0, 1]),
        block_table=torch.tensor([[0, 1, 2], [2, -1, 0]], dtype=torch.int32),
        block_size=2,
    )
    q = torch.zeros(2, 2, 8)
    cache = torch.zeros(6, 2, 8)
    output, lse = mlf_forward(impl, q, cache, metadata)
    expected_indices, expected_lengths = _physical_reference(
        metadata.req_id_per_token,
        metadata.block_table,
        impl.topk_indices_buffer,
        metadata.block_size,
    )
    attn = [entry for entry in sparse_calls if entry[0] == "attn"][0]
    assert lse is None
    assert output.shape == (2, 2, 4)
    assert output.tolist() == [
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        [[256, 257, 258, 259], [260, 261, 262, 263]],
    ]
    assert ("pad", 2, 64) in sparse_calls
    assert attn[1] == (2, 64, 8)
    assert attn[2] == (12, 1, 8)  # contiguous copy flattened to one row/block
    # The kernel consumes only the first column of the (tokens, 1, topk) view.
    indices = attn[3]
    assert indices.shape == (2, 1, 3)
    assert indices[:, 0, :].tolist() == expected_indices.tolist()
    assert attn[5]["topk_length"].tolist() == expected_lengths.tolist()
    assert attn[4] == 0.25


def mlf_forward(impl, q, cache, metadata):
    return mla.FlagGemsSparseMLAImpl.forward_mqa(impl, q, cache, metadata, None)


def test_sparse_forward_mqa_zero_rope_uses_nope_without_concat(monkeypatch):
    """A zero-width RoPE query skips concatenation and keeps q_nope."""
    provider = _flaggems_module(monkeypatch)
    calls = []
    monkeypatch.setattr(provider, "contiguous_copy", lambda cache: cache)
    monkeypatch.setattr(
        provider,
        "pad_attention_heads",
        lambda q, padded: calls.append(("pad", padded)) or q,
    )
    monkeypatch.setattr(
        provider,
        "concat_query",
        lambda nope, rope: pytest.fail("zero-width RoPE must not concatenate"),
    )
    impl = object.__new__(mla.FlagGemsSparseMLAImpl)
    impl.kv_lora_rank = 4
    impl.softmax_scale = 0.5
    impl.topk_indices_buffer = torch.zeros(1, 2, dtype=torch.int32)
    impl._sparse_attn = lambda q, *a, **k: (torch.zeros(1, q.shape[1], 4), None)
    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0]),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        block_size=1,
    )
    q_nope = torch.ones(1, 2, 8)
    output, _ = mlf_forward(
        impl, (q_nope, torch.zeros(1, 2, 0)), torch.zeros(1, 1, 8), metadata
    )
    assert output.shape == (1, 2, 4)
    assert calls == [("pad", 64)]


def test_sparse_forward_mqa_concats_query_and_uses_128_head_padding(monkeypatch):
    """A non-zero RoPE width concatenates; more than 64 heads pad to 128."""
    provider = _flaggems_module(monkeypatch)
    calls = []
    monkeypatch.setattr(provider, "contiguous_copy", lambda cache: cache)
    monkeypatch.setattr(
        provider,
        "pad_attention_heads",
        lambda q, padded: calls.append(("pad", q.shape[1], padded)) or q,
    )
    monkeypatch.setattr(
        provider,
        "concat_query",
        lambda nope, rope: (
            calls.append(("concat", nope.shape, rope.shape))
            or torch.cat((nope, rope), dim=-1)
        ),
    )
    impl = object.__new__(mla.FlagGemsSparseMLAImpl)
    impl.kv_lora_rank = 4
    impl.softmax_scale = 0.5
    impl.topk_indices_buffer = torch.zeros(1, 2, dtype=torch.int32)
    impl._sparse_attn = lambda q, *a, **k: (torch.zeros(1, q.shape[1], 4), None)
    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0]),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        block_size=1,
    )
    q_nope = torch.ones(1, 65, 4)
    q_pe = torch.ones(1, 65, 4)
    output, _ = mlf_forward(impl, (q_nope, q_pe), torch.zeros(1, 1, 8), metadata)
    assert calls == [("concat", (1, 65, 4), (1, 65, 4)), ("pad", 65, 128)]
    assert output.shape == (1, 65, 4)


def test_sparse_forward_mqa_requires_the_indexer_buffer(monkeypatch):
    """Without the indexer's top-k buffer the sparse path refuses to run."""
    impl = object.__new__(mla.FlagGemsSparseMLAImpl)
    impl.topk_indices_buffer = None
    with pytest.raises(RuntimeError, match="top-k buffer"):
        mlf_forward(
            impl,
            torch.zeros(1, 2, 8),
            torch.zeros(1, 1, 8),
            SimpleNamespace(),
        )


def test_flag_op_delegates_to_the_glm5_loader(monkeypatch):
    """_flag_op forwards the module/name pair to the loader table."""
    import vllm_fl.models.glm5.indexer_backend as backend

    seen = []
    monkeypatch.setattr(
        backend,
        "_load_flaggems_op",
        lambda module, name: seen.append((module, name)) or "OP",
    )
    assert mla._flag_op("concat_and_cache_mla", "concat_and_cache_mla") == "OP"
    assert seen == [("concat_and_cache_mla", "concat_and_cache_mla")]


def test_resolve_sparse_kernels_registers_both_operators(monkeypatch):
    """Both sparse entry points are resolved and registered as bindings."""
    requests = []
    monkeypatch.setattr(
        mla, "_flag_op", lambda module, name: requests.append(name) or (lambda *a: name)
    )
    monkeypatch.setattr(
        "vllm_fl.utils.use_flaggems_op", lambda name: name.endswith("fwd")
    )
    import vllm_fl.dispatch.binding as binding_module

    monkeypatch.setattr(
        binding_module.OperatorBinding,
        "describe",
        lambda self: {"op": self.op_name, "strict": True},
    )
    bindings = mla._resolve_sparse_mla_kernels()
    assert requests == ["concat_and_cache_mla", "flash_mla_sparse_fwd"]
    assert len(bindings) == 2


def test_resolve_sparse_kernels_rejects_missing_operator(monkeypatch):
    """A missing FlagGems-vllm op stops construction with a clear error."""
    monkeypatch.setattr(mla, "_flag_op", lambda module, name: None)
    with pytest.raises(RuntimeError, match="requires FlagGems-vllm"):
        mla._resolve_sparse_mla_kernels()


def _prefill_backend(monkeypatch, *, qk_nope=4, qk_rope=4, v_head=8, kernel=None):
    provider = sys.modules.get("flag_gems")
    if provider is None:
        provider = types.ModuleType("flag_gems")
        monkeypatch.setitem(sys.modules, "flag_gems", provider)
    if kernel is not None:
        monkeypatch.setattr(provider, "flash_attn_varlen_func", kernel)
    elif not hasattr(provider, "flash_attn_varlen_func"):
        monkeypatch.setattr(provider, "flash_attn_varlen_func", lambda *a, **k: None)
    return mla.FlagGemsMLAPrefillBackend(
        num_heads=2,
        scale=0.25,
        kv_lora_rank=8,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=v_head,
        vllm_config=MagicMock(),
    )


def test_prefill_backend_identity_and_availability(monkeypatch):
    """The prefill backend reports FLAGGEMS and probes the FlagGems symbol."""
    assert mla.FlagGemsMLAPrefillBackend.get_name() == "FLAGGEMS"
    provider = types.ModuleType("flag_gems")
    provider.flash_attn_varlen_func = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "flag_gems", provider)
    assert mla.FlagGemsMLAPrefillBackend.is_available() is True
    empty = types.ModuleType("flag_gems")
    monkeypatch.setitem(sys.modules, "flag_gems", empty)
    assert mla.FlagGemsMLAPrefillBackend.is_available() is False


@pytest.mark.parametrize(
    ("v_head", "expected"),
    [(8, False), (4, True)],
)
def test_prefill_backend_requires_padding_only_for_narrow_values(
    monkeypatch, v_head, expected
):
    """v_head_dim != qk_head_dim enables padding; equality disables it."""
    backend = _prefill_backend(monkeypatch, v_head=v_head, kernel=lambda *a, **k: None)
    assert backend.requires_v_padding is expected


def test_prefill_run_flash_attn_forwards_and_rejects_output_scale(monkeypatch):
    """The varlen call forwards metadata; quantized output is rejected."""
    kernel = _Kernel(result=torch.full((3, 2, 8), 6.0))
    backend = _prefill_backend(monkeypatch, kernel=kernel)
    q = torch.zeros(3, 2, 8)
    out = torch.zeros(3, 2, 8)
    result = backend._run_flash_attn(
        q,
        torch.zeros(5, 2, 8),
        torch.zeros(5, 2, 8),
        cu_seqlens_q="cq",
        cu_seqlens_k="ck",
        max_seqlen_q=3,
        max_seqlen_k=5,
        causal=True,
        return_softmax_lse=False,
        out=out,
    )
    assert result is not out
    kwargs = kernel.calls[0][1]
    assert kwargs["q"] is q
    assert kwargs["out"] is out
    assert kwargs["softmax_scale"] == 0.25
    assert kwargs["causal"] is True
    assert kwargs["fa_version"] == 2
    assert kwargs["cu_seqlens_q"] == "cq"
    assert kwargs["max_seqlen_k"] == 5
    with pytest.raises(NotImplementedError, match="quantized output"):
        backend._run_flash_attn(
            q,
            q,
            q,
            cu_seqlens_q="cq",
            cu_seqlens_k="ck",
            max_seqlen_q=3,
            max_seqlen_k=5,
            causal=True,
            return_softmax_lse=False,
            output_scale=torch.ones(1),
        )


def test_prefill_run_flash_attn_pads_slices_and_copies_into_out(monkeypatch):
    """Padding lets the provider allocate, then the trimmed width is copied."""

    def provider(**kwargs):
        assert kwargs["v"].shape == (5, 2, 8)
        assert kwargs["out"] is None
        return torch.full((3, 2, 8), 4.0)

    backend = _prefill_backend(monkeypatch, qk_nope=4, qk_rope=4, v_head=4)
    backend.flash_attn_varlen_func = provider
    out = torch.zeros(3, 2, 4)
    result = backend._run_flash_attn(
        torch.zeros(3, 2, 8),
        torch.zeros(5, 2, 8),
        torch.zeros(5, 2, 4),
        cu_seqlens_q="cq",
        cu_seqlens_k="ck",
        max_seqlen_q=3,
        max_seqlen_k=5,
        causal=False,
        return_softmax_lse=False,
        out=out,
    )
    assert result is out
    assert torch.all(out == 4.0)


def test_prefill_run_flash_attn_returns_lse_and_validates_it(monkeypatch):
    """return_softmax_lse yields the provider pair and rejects a missing LSE."""
    backend = _prefill_backend(
        monkeypatch,
        kernel=lambda **k: (
            torch.zeros(3, 2, 8),
            torch.full((3, 2), 2.0),
        ),
    )
    result, lse = backend._run_flash_attn(
        torch.zeros(3, 2, 8),
        torch.zeros(5, 2, 8),
        torch.zeros(5, 2, 8),
        cu_seqlens_q="cq",
        cu_seqlens_k="ck",
        max_seqlen_q=3,
        max_seqlen_k=5,
        causal=True,
        return_softmax_lse=True,
    )
    assert result.shape == (3, 2, 8)
    assert torch.all(lse == 2.0)

    backend.flash_attn_varlen_func = lambda **k: torch.zeros(3, 2, 8)
    with pytest.raises(RuntimeError, match="did not return softmax LSE"):
        backend._run_flash_attn(
            torch.zeros(3, 2, 8),
            torch.zeros(5, 2, 8),
            torch.zeros(5, 2, 8),
            cu_seqlens_q="cq",
            cu_seqlens_k="ck",
            max_seqlen_q=3,
            max_seqlen_k=5,
            causal=True,
            return_softmax_lse=True,
        )


def test_prefill_new_tokens_plumbs_query_metadata(monkeypatch):
    """New-token prefill uses the query start locations for both sides."""
    calls = []
    backend = _prefill_backend(
        monkeypatch, kernel=lambda **k: calls.append(k) or torch.zeros(3, 2, 8)
    )
    starts = torch.tensor([0, 3])
    backend._prefill_metadata = SimpleNamespace(
        query_start_loc=starts, max_query_len=3, chunked_context=None
    )
    backend.run_prefill_new_tokens(
        torch.zeros(3, 2, 8),
        torch.zeros(3, 2, 8),
        torch.zeros(3, 2, 8),
        return_softmax_lse=False,
    )
    call = calls[0]
    assert call["cu_seqlens_q"] is starts
    assert call["cu_seqlens_k"] is starts
    assert call["max_seqlen_q"] == call["max_seqlen_k"] == 3
    assert call["causal"] is True


def test_prefill_context_chunk_uses_chunk_metadata(monkeypatch):
    """Chunked context reads per-chunk offsets, is non-causal and returns LSE."""
    calls = []
    backend = _prefill_backend(
        monkeypatch,
        kernel=lambda **k: calls.append(k) or (torch.zeros(5, 2, 8), torch.zeros(5, 2)),
    )
    offsets = torch.tensor([0, 2, 5])
    backend._prefill_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 5]),
        max_query_len=5,
        chunked_context=SimpleNamespace(cu_seq_lens=[offsets], max_seq_lens=[5]),
    )
    result, lse = backend.run_prefill_context_chunk(
        0,
        torch.zeros(5, 2, 8),
        torch.zeros(5, 2, 8),
        torch.zeros(5, 2, 8),
    )
    call = calls[0]
    assert call["cu_seqlens_k"] is offsets
    assert call["max_seqlen_k"] == 5
    assert call["causal"] is False
    assert call["return_softmax_lse"] is True
    assert result.shape == (5, 2, 8)
    assert lse.shape == (5, 2)
