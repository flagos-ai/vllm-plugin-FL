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

"""Coverage for the FlagGems attention backend implementation.

Pure metadata/layout helpers and the cascade heuristic run on CPU.  Forward
and cache-update paths need tiny CUDA tensors and carry ``gpu``; the external
FlagGems kernel is replaced by a recorder that evaluates the same paged /
dense attention with an explicit matmul-softmax reference written here, so
the numerics and the exact kernel arguments (slices, shapes, strides, dtype,
layout, descales) are both asserted.  DCP, cascade and encoder entry points
are driven separately.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from vllm_fl.dispatch.backends.flaggems.impl import attention as fa

RTOL = 1e-2
ATOL = 1e-2


def _attn_type(name):
    """Resolve a vLLM ``AttentionType`` member after the import shims ran."""
    from vllm.v1.attention.backend import AttentionType

    return getattr(AttentionType, name)


def _value_fields(item):
    """Field-name/value mapping of a vLLM ``MultipleOf`` value object.

    vLLM 0.24's ``MultipleOf`` does not define ``__eq__``, so two instances
    constructed with the same argument compare by identity.  Read the stored
    fields from the class definition (``__slots__`` or instance ``__dict__``)
    and compare those values instead.
    """
    slots = getattr(type(item), "__slots__", None)
    if slots:
        names = (slots,) if isinstance(slots, str) else slots
        return {name: getattr(item, name) for name in names}
    return dict(vars(item))


def _impl(
    num_heads=2,
    head_size=8,
    num_kv_heads=1,
    sliding_window=None,
    attn_type=None,
    kv_cache_dtype="auto",
    alibi_slopes=None,
    logits_soft_cap=None,
):
    if attn_type is None:
        attn_type = _attn_type("DECODER")
    impl = fa.AttentionFLImpl(
        num_heads,
        head_size,
        head_size**-0.5,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        logits_soft_cap,
        attn_type,
    )
    # vLLM's Attention layer normally publishes this onto the impl.
    impl.dcp_world_size = 1
    return impl


def _layer(q_scale=0.5, k_scale=0.25, v_scale=0.125):
    return SimpleNamespace(
        _q_scale=torch.tensor(q_scale, dtype=torch.float32),
        _k_scale=torch.tensor(k_scale, dtype=torch.float32),
        _v_scale=torch.tensor(v_scale, dtype=torch.float32),
    )


class TestBackendMetadata:
    """Static backend identity, capabilities and KV-cache layout."""

    def test_identity_and_support_flags(self):
        """Name, impl/builder classes and the fixed support flags."""
        backend = fa.AttentionFLBackend
        assert backend.get_name() == "CUSTOM"
        assert backend.get_impl_cls() is fa.AttentionFLImpl
        assert backend.get_builder_cls() is fa.AttentionFLMetadataBuilder
        assert backend.supports_sink() is False
        assert backend.accept_output_buffer is True
        assert backend.forward_includes_kv_cache_update is False
        assert backend.supported_dtypes == [torch.float16, torch.bfloat16]

    def test_supports_attn_type(self):
        """All four vLLM attention types are accepted, others are not."""
        cases = [
            ("DECODER", True),
            ("ENCODER", True),
            ("ENCODER_ONLY", True),
            ("ENCODER_DECODER", True),
            ("cross", False),
        ]
        for name, expected in cases:
            attn_type = _attn_type(name) if name != "cross" else name
            assert (
                fa.AttentionFLBackend.supports_attn_type(attn_type) is expected
            )

    def test_kv_cache_dtype_support(self):
        """Only unquantized (auto) KV cache is supported."""
        support = fa.AttentionFLBackend.supports_kv_cache_dtype
        assert support(None) is True
        assert support("auto") is True
        assert support("fp8") is False
        assert support("fp8_e4m3") is False

    def test_kv_cache_shape_and_block_size_validation(self):
        """KV cache is [2, blocks, block, heads, head_size]; block%16 == 0."""
        shape = fa.AttentionFLBackend.get_kv_cache_shape(4, 16, 2, 8)
        assert shape == (2, 4, 16, 2, 8)
        with pytest.raises(ValueError, match="multiple of 16"):
            fa.AttentionFLBackend.get_kv_cache_shape(4, 8, 2, 8)

    @pytest.mark.parametrize(
        "head_size,expected",
        [(8, True), (16, True), (256, True), (257, False), (12, False)],
    )
    def test_supports_head_size(self, head_size, expected):
        """Head size must be a multiple of 8 and at most 256."""
        assert fa.AttentionFLBackend.supports_head_size(head_size) is expected

    def test_combination_rejects_sinks_only(self):
        """Only a requested sink makes the combination unsupported."""
        kwargs = dict(
            head_size=128,
            dtype=torch.bfloat16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            use_sparse=False,
            device_capability=None,
        )
        assert (
            fa.AttentionFLBackend.supports_combination(has_sink=True, **kwargs)
            == "not support sink"
        )
        assert fa.AttentionFLBackend.supports_combination(has_sink=False, **kwargs) is None

    @pytest.mark.parametrize(
        "layout,include_layers,expected",
        [
            ("NHD", False, (0, 1, 2, 3, 4)),
            ("NHD", True, (2, 0, 1, 3, 4, 5)),
            ("HND", False, (0, 1, 3, 2, 4)),
            ("HND", True, (2, 4, 0, 1, 3, 5)),
        ],
    )
    def test_kv_cache_stride_order(self, monkeypatch, layout, include_layers, expected):
        """Stride permutations match the documented NHD/HND memory layouts."""
        monkeypatch.setattr(fa, "get_kv_cache_layout", lambda: layout)
        order = fa.AttentionFLBackend.get_kv_cache_stride_order(include_layers)
        assert order == expected

    def test_unknown_cache_layout_is_rejected(self, monkeypatch):
        """An unrecognized cache layout raises instead of guessing."""
        monkeypatch.setattr(fa, "get_kv_cache_layout", lambda: "XYZ")
        with pytest.raises(ValueError, match="Unknown cache layout"):
            fa.AttentionFLBackend.get_kv_cache_stride_order()

    def _config(self, hybrid, mamba_dtype):
        return SimpleNamespace(
            model_config=SimpleNamespace(is_hybrid=hybrid),
            cache_config=SimpleNamespace(
                mamba_ssm_cache_dtype=mamba_dtype,
                mamba_cache_dtype=mamba_dtype,
            ),
        )

    def test_hybrid_float32_mamba_uses_fixed_block_sizes(self, monkeypatch):
        """Hybrid Mamba-float32 models use the NaN-safe [16, 32, 64] set."""
        monkeypatch.setattr(
            fa, "get_current_vllm_config", lambda: self._config(True, "float32")
        )
        assert fa.AttentionFLBackend.get_supported_kernel_block_sizes() == [16, 32, 64]

    @pytest.mark.parametrize(
        "hybrid,mamba",
        [(False, "float32"), (True, "auto"), (True, None)],
    )
    def test_default_block_size_is_multiple_of_16(self, monkeypatch, hybrid, mamba):
        """Every other configuration only requires a multiple of 16."""
        monkeypatch.setattr(
            fa, "get_current_vllm_config", lambda: self._config(hybrid, mamba)
        )
        from vllm.v1.attention.backend import MultipleOf

        result = fa.AttentionFLBackend.get_supported_kernel_block_sizes()
        assert len(result) == 1
        assert isinstance(result[0], MultipleOf)
        assert _value_fields(result[0]) == _value_fields(MultipleOf(16))


class TestImplInit:
    """``AttentionFLImpl`` constructor normalises the sliding/alibi contract."""

    def test_defaults(self):
        """Default construction collapses every optional to its neutral form."""
        impl = _impl()
        assert impl.sliding_window == (-1, -1)
        assert impl.logits_soft_cap == 0
        assert impl.num_queries_per_kv == 2
        assert impl.alibi_slopes is None
        assert impl.kv_sharing_target_layer_name is None
        assert impl.vllm_flash_attn_version == 2
        assert impl.supports_quant_query_input is False
        assert impl.attn_type == _attn_type("DECODER")

    def test_decoder_sliding_window_is_leftward(self):
        """Decoder sliding window is (window - 1, 0)."""
        assert _impl(sliding_window=8).sliding_window == (7, 0)

    def test_encoder_only_sliding_window_is_symmetric(self):
        """Encoder-only sliding window is symmetric around the token."""
        impl = _impl(sliding_window=8, attn_type=_attn_type("ENCODER_ONLY"))
        assert impl.sliding_window == (7, 7)

    def test_alibi_slopes_become_float32_tensor(self):
        """A slope list is materialised as a float32 tensor."""
        impl = _impl(alibi_slopes=[0.1, 0.2])
        assert impl.alibi_slopes.dtype == torch.float32
        torch.testing.assert_close(
            impl.alibi_slopes, torch.tensor([0.1, 0.2], dtype=torch.float32)
        )

    def test_logits_soft_cap_is_preserved(self):
        """A non-zero soft cap is stored as given."""
        assert _impl(logits_soft_cap=5.0).logits_soft_cap == 5.0

    @pytest.mark.parametrize("kv_dtype", ["fp8", "fp8_e4m3"])
    def test_quantized_kv_cache_is_rejected(self, kv_dtype):
        """Quantized KV caches are rejected at construction time."""
        with pytest.raises(NotImplementedError, match="quantization kv-cache"):
            _impl(kv_cache_dtype=kv_dtype)


class TestCascadeHeuristic:
    """``use_cascade_attention`` thresholds against hand-computed decisions."""

    def _decide(self, prefix, lens, num_q=8, num_kv=2, alibi=False,
                sliding=False, local=False, sms=132, dcp=1):
        return fa.use_cascade_attention(
            prefix,
            np.asarray(lens),
            num_q,
            num_kv,
            alibi,
            sliding,
            local,
            sms,
            dcp,
        )

    def test_short_prefix_is_rejected(self):
        """Common prefixes below 256 tokens never use cascade."""
        assert self._decide(255, [2] * 8) is False

    @pytest.mark.parametrize(
        "flags",
        [
            {"alibi": True},
            {"sliding": True},
            {"local": True},
        ],
    )
    def test_unsupported_variants_are_rejected(self, flags):
        """ALiBi, sliding window and local attention disable cascade."""
        assert self._decide(300, [2] * 8, **flags) is False

    def test_too_few_queries_is_rejected(self):
        """Fewer than eight query rows never uses cascade."""
        assert self._decide(300, [2] * 7) is False

    def test_dcp_disables_cascade(self):
        """Decode context parallelism disables cascade attention."""
        assert self._decide(300, [2] * 8, dcp=2) is False

    def test_non_decode_queries_use_cascade(self):
        """Multi-token queries cannot use FlashDecoding, so cascade wins."""
        assert self._decide(300, [2] * 8) is True

    def test_single_query_per_kv_uses_cascade(self):
        """With one query per KV head there is no FlashDecoding split."""
        assert self._decide(300, [1] * 8, num_q=2, num_kv=2) is True

    def test_flash_decoding_cost_model(self):
        """Decode batches pick cascade only when its wave count is lower.

        prefix 300 -> 3 KV tiles; 8 queries; 8 query heads / 2 KV heads:
        cascade_ctas = 8, flash_decoding_ctas = 8 * 2 = 16 (times 3 tiles).
        With 4 SMs cascade takes 2 waves (6 units) versus 12 for
        FlashDecoding; with 132 SMs both take one wave (3 vs 1), so cascade
        loses.
        """
        assert self._decide(300, [1] * 8, sms=4) is True
        assert self._decide(300, [1] * 8, sms=132) is False


def _builder(dcp_world_size=1, dcp_rank=0):
    builder = object.__new__(fa.AttentionFLMetadataBuilder)
    builder.kv_cache_spec = SimpleNamespace(dtype=torch.bfloat16, block_size=16)
    builder.layer_names = ["model.layers.0.attn"]
    builder.vllm_config = SimpleNamespace()
    builder.device = torch.device("cpu")
    builder.model_config = SimpleNamespace(
        get_num_attention_heads=lambda parallel: 8,
        get_num_kv_heads=lambda parallel: 2,
        get_head_size=lambda: 128,
    )
    builder.parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=1)
    builder.cache_config = SimpleNamespace()
    builder.compilation_config = SimpleNamespace()
    builder.num_heads_q = 8
    builder.num_heads_kv = 2
    builder.kv_cache_dtype = torch.bfloat16
    builder.headdim = 128
    builder.block_size = 16
    builder.max_num_splits = 0
    builder.aot_schedule = False
    builder.dcp_world_size = dcp_world_size
    builder.dcp_rank = dcp_rank
    builder.cp_kv_cache_interleave_size = 1
    builder.use_full_cuda_graph = False
    builder.max_cudagraph_size = 0
    builder.aot_sliding_window = (-1, -1)
    return builder


def _common(query_lens, seq_lens, num_reqs=None):
    num_reqs = num_reqs if num_reqs is not None else len(query_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.cumsum(torch.tensor(query_lens, dtype=torch.int32), 0)
    seq = torch.tensor(seq_lens, dtype=torch.int32)
    num_actual = int(sum(query_lens))
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_actual_tokens=num_actual,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        query_start_loc=query_start_loc,
        seq_lens=seq,
        block_table_tensor=torch.zeros(num_reqs, 4, dtype=torch.int32),
        slot_mapping=torch.arange(num_actual, dtype=torch.int32),
        causal=True,
    )


class TestMetadataBuilderInit:
    """Constructor resolves config surfaces and the DCP fallback."""

    def _build_init(self, monkeypatch, dcp_present):
        import vllm.distributed.parallel_state as ps

        def _get_dcp_group():
            if not dcp_present:
                raise AssertionError("DCP not initialized")
            return SimpleNamespace(world_size=4, rank_in_group=2)

        monkeypatch.setattr(ps, "get_dcp_group", _get_dcp_group)
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                get_num_attention_heads=lambda parallel: 16,
                get_num_kv_heads=lambda parallel: 4,
                get_head_size=lambda: 128,
            ),
            parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=2),
            cache_config=SimpleNamespace(),
            compilation_config=SimpleNamespace(
                cudagraph_mode=SimpleNamespace(
                    has_full_cudagraphs=lambda: False
                ),
                max_cudagraph_capture_size=0,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=8),
            attention_config=SimpleNamespace(
                flash_attn_max_num_splits_for_cuda_graph=0
            ),
        )
        spec = SimpleNamespace(dtype=torch.bfloat16, block_size=16)
        return fa.AttentionFLMetadataBuilder(spec, ["layer"], config,
                                            torch.device("cpu"))

    def test_falls_back_to_single_rank_without_dcp(self, monkeypatch):
        """A missing DCP group yields world size 1 / rank 0."""
        builder = self._build_init(monkeypatch, dcp_present=False)
        assert builder.dcp_world_size == 1
        assert builder.dcp_rank == 0
        assert builder.num_heads_q == 16
        assert builder.num_heads_kv == 4
        assert builder.headdim == 128
        assert builder.max_num_splits == 0
        assert builder.aot_schedule is False
        assert builder.use_full_cuda_graph is False
        from vllm.v1.attention.backend import AttentionCGSupport

        assert (
            fa.AttentionFLMetadataBuilder._cudagraph_support
            is AttentionCGSupport.UNIFORM_BATCH
        )

    def test_reads_initialized_dcp_group(self, monkeypatch):
        """An initialized DCP group supplies world size and rank."""
        builder = self._build_init(monkeypatch, dcp_present=True)
        assert builder.dcp_world_size == 4
        assert builder.dcp_rank == 2
        assert builder.cp_kv_cache_interleave_size == 2


class TestMetadataBuilderBuild:
    """``build`` turns common metadata into cascade/DCP-aware metadata."""

    def test_plain_decode_batch(self):
        """A plain batch copies the common tensors unchanged."""
        builder = _builder()
        common = _common([1, 1, 1], [4, 5, 6])
        meta = builder.build(0, common)

        assert meta.num_actual_tokens == 3
        assert meta.max_query_len == 1
        assert meta.max_seq_len == 6
        assert meta.query_start_loc is common.query_start_loc
        assert meta.seq_lens is common.seq_lens
        assert meta.block_table is common.block_table_tensor
        assert meta.slot_mapping is common.slot_mapping
        assert meta.use_cascade is False
        assert meta.common_prefix_len == 0
        assert meta.dcp_context_kv_lens is None
        assert meta.max_dcp_context_kv_len == 0
        assert meta.cu_prefix_query_lens is None
        assert meta.prefix_kv_lens is None
        assert meta.suffix_kv_lens is None
        assert meta.scheduler_metadata is None
        assert meta.max_num_splits == 0
        assert meta.causal is True

    def test_cascade_batch_tensors(self):
        """A shared prefix produces the cascade prefix/suffix tensors."""
        builder = _builder()
        common = _common([2, 2], [16, 16])
        meta = builder.build(32, common)

        assert meta.use_cascade is True
        assert meta.common_prefix_len == 32
        assert meta.cu_prefix_query_lens.tolist() == [0, 4]
        assert meta.cu_prefix_query_lens.dtype == torch.int32
        assert meta.prefix_kv_lens.tolist() == [32]
        expected_suffix = common.seq_lens[: common.num_reqs] - 32
        torch.testing.assert_close(meta.suffix_kv_lens, expected_suffix)
        assert meta.scheduler_metadata is None

    def test_dcp_batch_splits_context_lengths(self):
        """DCP context lengths use the interleave/rank split formula."""
        builder = _builder(dcp_world_size=2, dcp_rank=1)
        common = _common([1, 1, 1], [7, 8, 9])
        meta = builder.build(0, common)

        # context = seq - query = [6, 7, 8]; split over rank 1 of 2 with
        # interleave 1: base [3, 3, 4], remainder [0, 0, 0].
        assert meta.dcp_context_kv_lens.tolist() == [3, 3, 4]
        assert meta.max_dcp_context_kv_len == 5  # ceil(9 / 2)
        assert meta.use_cascade is False
        assert meta.cu_prefix_query_lens is None

    def test_full_cudagraph_splits_are_capped_by_batch(self):
        """Only small batches under a full CUDA graph take max_num_splits."""
        builder = _builder()
        builder.use_full_cuda_graph = True
        builder.max_cudagraph_size = 8
        builder.max_num_splits = 4
        small = _common([1, 1], [4, 4])
        assert builder.build(0, small).max_num_splits == 4
        large = _common([1] * 16, [4] * 16)
        assert builder.build(0, large).max_num_splits == 0


def _request_attention(q, k, v, scale, causal):
    """Explicit GQA attention for one request, computed in float32 on CPU."""
    q = q.detach().float().cpu()
    k = k.detach().float().cpu()
    v = v.detach().float().cpu()
    repeat = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("qhd,shd->hqs", q, k) * scale
    if causal:
        kv_len = k.shape[0]
        query_len = q.shape[0]
        mask = torch.arange(kv_len)[None, :] > torch.arange(
            kv_len - query_len, kv_len
        )[:, None]
        scores = scores.masked_fill(mask[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqs,shd->qhd", probs, v)


def _gather_cache(cache, block_table, length, block_size):
    blocks = block_table[: (length + block_size - 1) // block_size].tolist()
    flat = cache[blocks].reshape(
        -1, cache.shape[-2], cache.shape[-1]
    )
    return flat[:length]


def _paged_reference(kwargs):
    q = kwargs["q"]
    k_cache = kwargs["k"]
    v_cache = kwargs["v"]
    block_table = kwargs["block_table"]
    seqused = kwargs["seqused_k"]
    cu = kwargs["cu_seqlens_q"]
    block_size = k_cache.shape[-3]
    pieces = []
    for batch in range(seqused.shape[0]):
        length = int(seqused[batch])
        q_b = q[int(cu[batch]) : int(cu[batch + 1])]
        k_b = _gather_cache(k_cache, block_table[batch], length, block_size)
        v_b = _gather_cache(v_cache, block_table[batch], length, block_size)
        pieces.append(
            _request_attention(q_b, k_b, v_b, kwargs["softmax_scale"],
                               kwargs["causal"])
        )
    return torch.cat(pieces, dim=0)


def _kernel_reference(kwargs):
    """Reference for either a paged cache call or a direct Q/K/V call."""
    if "block_table" in kwargs:
        return _paged_reference(kwargs)
    return _request_attention(
        kwargs["q"], kwargs["k"], kwargs["v"], kwargs["softmax_scale"],
        kwargs["causal"],
    )


class _FlashRecorder:
    """Stand-in for the external FlagGems kernel; records every argument."""

    def __init__(self, compute):
        self.compute = compute
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        result = self.compute(kwargs)
        out = kwargs.get("out")
        if out is not None:
            out.copy_(result.to(device=out.device, dtype=out.dtype))
        if kwargs.get("return_softmax_lse"):
            batch = kwargs["cu_seqlens_q"].shape[0] - 1
            lse = torch.zeros(
                result.shape[1], batch, dtype=torch.float32, device=out.device
                if out is not None else result.device,
            )
            return result, lse
        return None if out is not None else result


def _attention_metadata(seq_len, num_actual, block_table, causal=True):
    return SimpleNamespace(
        num_actual_tokens=num_actual,
        max_query_len=seq_len,
        query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32),
        max_seq_len=seq_len,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        block_table=block_table,
        slot_mapping=torch.arange(num_actual, dtype=torch.int64),
        use_cascade=False,
        causal=causal,
        scheduler_metadata=None,
        dcp_context_kv_lens=None,
        max_dcp_context_kv_len=0,
    )


class TestForwardDecode:
    """Paged decoder forward: slicing, kernel arguments and numerics."""

    @pytest.mark.gpu
    def test_forward_matches_reference_and_writes_output(self, monkeypatch, device):
        """Only the real tokens are attended and written into ``output``."""
        torch.manual_seed(4)
        num_actual, total, heads, kv_heads, head_dim, block_size = 4, 6, 2, 1, 8, 16
        query = torch.randn(total, heads, head_dim, device=device, dtype=torch.bfloat16)
        key_cache = torch.randn(
            1, block_size, kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        value_cache = torch.randn_like(key_cache)
        kv_cache = torch.stack([key_cache, value_cache], dim=0)
        block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
        metadata = _attention_metadata(num_actual, num_actual, block_table)
        output = torch.zeros(total, heads, head_dim, device=device,
                             dtype=torch.bfloat16)

        recorder = _FlashRecorder(_paged_reference)
        monkeypatch.setattr(fa, "flash_attn_varlen_func", recorder)
        impl = _impl(num_heads=heads, head_size=head_dim, num_kv_heads=kv_heads)
        layer = _layer()

        returned = impl.forward(
            layer, query, torch.empty(0), torch.empty(0), kv_cache,
            metadata, output=output,
        )

        assert returned is output
        expected = _request_attention(
            query[:num_actual], key_cache[0, :num_actual].reshape(
                num_actual, kv_heads, head_dim
            ), value_cache[0, :num_actual].reshape(num_actual, kv_heads, head_dim),
            impl.scale, True,
        )
        torch.testing.assert_close(
            output[:num_actual].float().cpu(), expected, rtol=RTOL, atol=ATOL
        )
        # Padding rows stay untouched.
        assert torch.count_nonzero(output[num_actual:]) == 0

        call = recorder.calls[0]
        assert call["q"].shape == (num_actual, heads, head_dim)
        assert call["q"].dtype == torch.bfloat16
        assert call["q"].is_contiguous()
        assert call["k"].data_ptr() == kv_cache[0].data_ptr()
        assert call["v"].data_ptr() == kv_cache[1].data_ptr()
        torch.testing.assert_close(call["k"], key_cache)
        torch.testing.assert_close(call["v"], value_cache)
        assert call["out"].data_ptr() == output[:num_actual].data_ptr()
        assert call["cu_seqlens_q"] is metadata.query_start_loc
        assert call["seqused_k"] is metadata.seq_lens
        assert call["max_seqlen_q"] == num_actual
        assert call["max_seqlen_k"] == num_actual
        assert call["block_table"] is block_table
        assert call["softmax_scale"] == impl.scale
        assert call["causal"] is True
        assert call["alibi_slopes"] is None
        assert call["window_size"] == (-1, -1)
        assert call["softcap"] == 0
        assert call["scheduler_metadata"] is None
        assert call["fa_version"] == 2
        assert call["num_splits"] == 0
        assert call["s_aux"] is None
        for name, scale in (
            ("q_descale", 0.5),
            ("k_descale", 0.25),
            ("v_descale", 0.125),
        ):
            assert call[name].shape == (1, kv_heads)
            assert call[name].dtype == torch.float32
            torch.testing.assert_close(
                call[name], torch.full((1, kv_heads), scale)
            )

    @pytest.mark.gpu
    def test_none_metadata_zero_fills_output(self, device):
        """A profiling call (no metadata) only zero-fills the output."""
        impl = _impl()
        output = torch.ones(3, 2, 8, device=device)
        returned = impl.forward(
            _layer(), torch.empty(0, device=device), torch.empty(0, device=device),
            torch.empty(0, device=device), None, None, output=output,
        )
        assert returned is output
        assert torch.count_nonzero(output) == 0

    @pytest.mark.gpu
    def test_missing_output_is_rejected(self, device):
        """forward requires an output buffer."""
        impl = _impl()
        with pytest.raises(AssertionError, match="Output tensor"):
            impl.forward(
                _layer(), torch.empty(0, device=device),
                torch.empty(0, device=device), torch.empty(0, device=device),
                None, None, output=None,
            )

    @pytest.mark.gpu
    def test_fused_output_quantization_is_rejected(self, device):
        """Fused output quantization is not implemented."""
        impl = _impl()
        with pytest.raises(NotImplementedError, match="fused output quantization"):
            impl.forward(
                _layer(), torch.empty(0, device=device),
                torch.empty(0, device=device), torch.empty(0, device=device),
                None, None, output=torch.empty(0, device=device),
                output_scale=torch.ones(1, device=device),
            )


class TestForwardEncoder:
    """Encoder attention runs directly on Q/K/V without the KV cache."""

    @pytest.mark.gpu
    def test_encoder_forward_matches_reference(self, monkeypatch, device):
        """Bidirectional dense attention is applied to the uncached tensors."""
        torch.manual_seed(8)
        tokens, heads, kv_heads, head_dim = 4, 2, 1, 8
        query = torch.randn(tokens, heads, head_dim, device=device,
                            dtype=torch.bfloat16)
        key = torch.randn(tokens, kv_heads, head_dim, device=device,
                          dtype=torch.bfloat16)
        value = torch.randn_like(key)
        output = torch.zeros_like(query)
        metadata = SimpleNamespace(
            num_actual_tokens=tokens,
            query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
            max_query_len=tokens,
        )

        def _dense(kwargs):
            return _request_attention(
                kwargs["q"], kwargs["k"], kwargs["v"], kwargs["softmax_scale"],
                False,
            )

        recorder = _FlashRecorder(_dense)
        monkeypatch.setattr(fa, "flash_attn_varlen_func", recorder)
        impl = _impl(
            num_heads=heads, head_size=head_dim, num_kv_heads=kv_heads,
            attn_type=_attn_type("ENCODER_ONLY"),
        )
        returned = impl.forward(
            _layer(), query, key, value, None, metadata, output=output
        )

        assert returned.data_ptr() == output.data_ptr()
        assert returned.shape == output.shape
        expected = _request_attention(query, key, value, impl.scale, False)
        torch.testing.assert_close(
            output.float().cpu(), expected, rtol=RTOL, atol=ATOL
        )
        call = recorder.calls[0]
        assert call["q"].data_ptr() == query.data_ptr()
        assert call["k"].data_ptr() == key.data_ptr()
        assert call["v"].data_ptr() == value.data_ptr()
        assert call["out"].data_ptr() == output.data_ptr()
        assert call["cu_seqlens_q"] is call["cu_seqlens_k"]
        assert call["max_seqlen_q"] == tokens
        assert call["max_seqlen_k"] == tokens
        assert call["causal"] is False
        assert call["q_descale"].shape == (1, kv_heads)

    @pytest.mark.gpu
    def test_encoder_fp8_cache_is_rejected(self, device):
        """The encoder path refuses an FP8 cache dtype."""
        impl = _impl(attn_type=_attn_type("ENCODER_ONLY"))
        impl.kv_cache_dtype = "fp8"
        with pytest.raises(NotImplementedError, match="encoder attention"):
            impl._forward_encoder_attention(
                torch.empty(0, device=device), torch.empty(0, device=device),
                torch.empty(0, device=device), torch.empty(0, device=device),
                SimpleNamespace(query_start_loc=torch.zeros(1, dtype=torch.int32)),
                _layer(),
            )


class TestKvCacheUpdate:
    """``do_kv_cache_update`` writes through the FlagGems cache kernel."""

    @pytest.mark.gpu
    def test_decoder_writes_key_value_and_scales(self, monkeypatch, device):
        """The cache kernel receives both halves and the layer scales."""
        impl = _impl()
        key = torch.randn(3, 1, 8, device=device, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        kv_cache = torch.zeros(2, 4, 16, 1, 8, device=device,
                               dtype=torch.bfloat16)
        slot_mapping = torch.tensor([0, 17, 33], dtype=torch.int64, device=device)
        layer = _layer()
        calls = []

        def _reshape(key, value, key_cache, value_cache, slots, dtype, k_scale,
                     v_scale):
            calls.append((key, value, key_cache, value_cache, slots, dtype,
                          k_scale, v_scale))

        monkeypatch.setattr(fa, "reshape_and_cache_flash", _reshape)
        impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

        assert len(calls) == 1
        assert calls[0][0] is key
        assert calls[0][1] is value
        assert calls[0][2].data_ptr() == kv_cache[0].data_ptr()
        assert calls[0][3].data_ptr() == kv_cache[1].data_ptr()
        assert calls[0][4] is slot_mapping
        assert calls[0][5] == "auto"
        assert calls[0][6] is layer._k_scale
        assert calls[0][7] is layer._v_scale

    @pytest.mark.gpu
    def test_encoder_is_a_no_op(self, monkeypatch, device):
        """Encoder attention never updates the paged cache."""
        impl = _impl(attn_type=_attn_type("ENCODER"))
        calls = []
        monkeypatch.setattr(
            fa, "reshape_and_cache_flash", lambda *args, **kwargs: calls.append(1)
        )
        impl.do_kv_cache_update(
            _layer(), torch.empty(0, device=device), torch.empty(0, device=device),
            None, None,
        )
        assert calls == []


class TestCascadeForward:
    """Prefix/suffix cascade attention splits the block table correctly."""

    @pytest.mark.gpu
    def test_cascade_attention_routes_residual_kv(self, monkeypatch, device):
        """The prefix uses block 0..1 and the suffix starts at block 2."""
        block_size, kv_heads, head_dim = 16, 2, 8
        num_blocks = 4
        key_cache = torch.randn(num_blocks, block_size, kv_heads, head_dim,
                                device=device, dtype=torch.bfloat16)
        value_cache = torch.randn_like(key_cache)
        block_table = torch.arange(num_blocks, dtype=torch.int32,
                                   device=device).reshape(1, num_blocks)
        query = torch.randn(2, 2, head_dim, device=device, dtype=torch.bfloat16)
        output = torch.zeros_like(query)
        cu_query_lens = torch.tensor([0, 1, 2], dtype=torch.int32)
        prefix_kv_lens = torch.tensor([32], dtype=torch.int32)
        suffix_kv_lens = torch.tensor([3, 4], dtype=torch.int32)
        merged = []

        def _compute(kwargs):
            flag = 1.0 if kwargs["causal"] is False else 2.0
            return torch.full_like(
                kwargs["q"].float().cpu(), flag
            )

        recorder = _FlashRecorder(_compute)
        monkeypatch.setattr(fa, "flash_attn_varlen_func", recorder)

        def _merge(out, prefix_out, prefix_lse, suffix_out, suffix_lse):
            merged.append((prefix_out, suffix_out))
            out.copy_(prefix_out + suffix_out)

        monkeypatch.setattr(fa, "merge_attn_states", _merge)

        fa.cascade_attention(
            output,
            query,
            key_cache,
            value_cache,
            cu_query_lens=cu_query_lens,
            max_query_len=1,
            cu_prefix_query_lens=torch.tensor([0, 2], dtype=torch.int32),
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            max_kv_len=40,
            softmax_scale=0.5,
            alibi_slopes=None,
            sliding_window=(-1, -1),
            logits_soft_cap=0,
            block_table=block_table,
            common_prefix_len=32,
            fa_version=2,
        )

        prefix_call, suffix_call = recorder.calls
        assert prefix_call["block_table"].tolist() == [[0, 1, 2, 3]]
        assert prefix_call["seqused_k"] is prefix_kv_lens
        assert prefix_call["cu_seqlens_q"].tolist() == [0, 2]
        assert prefix_call["max_seqlen_q"] == 2
        assert prefix_call["max_seqlen_k"] == 32
        assert prefix_call["causal"] is False
        assert prefix_call["return_softmax_lse"] is True
        assert suffix_call["block_table"].tolist() == [[2, 3]]
        assert suffix_call["seqused_k"] is suffix_kv_lens
        assert suffix_call["cu_seqlens_q"] is cu_query_lens
        assert suffix_call["max_seqlen_q"] == 1
        assert suffix_call["max_seqlen_k"] == 8
        assert suffix_call["causal"] is True
        assert len(merged) == 1
        assert merged[0][0].shape == query.shape
        assert merged[0][1].shape == query.shape
        # Merge stand-in wrote prefix (1.0) + suffix (2.0) into every element.
        torch.testing.assert_close(output, torch.full_like(output, 3.0))

    @pytest.mark.gpu
    def test_cascade_rejects_alibi_and_window(self, device):
        """Cascade attention refuses ALiBi and sliding windows outright."""
        kwargs = dict(
            output=torch.empty(1, 2, 8, device=device),
            query=torch.empty(1, 2, 8, device=device),
            key_cache=torch.empty(1, 16, 2, 8, device=device),
            value_cache=torch.empty(1, 16, 2, 8, device=device),
            cu_query_lens=torch.tensor([0, 1], dtype=torch.int32),
            max_query_len=1,
            cu_prefix_query_lens=torch.tensor([0, 1], dtype=torch.int32),
            prefix_kv_lens=torch.tensor([16], dtype=torch.int32),
            suffix_kv_lens=torch.tensor([1], dtype=torch.int32),
            max_kv_len=17,
            softmax_scale=0.5,
            sliding_window=(-1, -1),
            logits_soft_cap=0,
            block_table=torch.tensor([[0]], dtype=torch.int32),
            common_prefix_len=16,
            fa_version=2,
        )
        with pytest.raises(AssertionError, match="ALiBi"):
            fa.cascade_attention(alibi_slopes=torch.zeros(1), **kwargs)
        kwargs["alibi_slopes"] = None
        kwargs["sliding_window"] = (-1, 0)
        with pytest.raises(AssertionError, match="sliding window"):
            fa.cascade_attention(**kwargs)
        kwargs["sliding_window"] = (-1, -1)
        kwargs["common_prefix_len"] = 8
        with pytest.raises(AssertionError):
            fa.cascade_attention(**kwargs)


class TestDcpForward:
    """Decode context-parallel forward issues both flash calls and merges."""

    @pytest.mark.gpu
    def test_forward_with_dcp_arguments(self, monkeypatch, device):
        """Context attention is non-causal over DCP lengths; the query path
        uses the full Q/K/V tensors and both LSEs are merged."""
        torch.manual_seed(21)
        tokens, heads, kv_heads, head_dim, block_size = 3, 2, 1, 8, 16
        query = torch.randn(tokens, heads, head_dim, device=device,
                            dtype=torch.bfloat16)
        key = torch.randn(tokens, kv_heads, head_dim, device=device,
                          dtype=torch.bfloat16)
        value = torch.randn_like(key)
        key_cache = torch.randn(1, block_size, kv_heads, head_dim, device=device,
                                dtype=torch.bfloat16)
        value_cache = torch.randn_like(key_cache)
        output = torch.zeros(tokens, heads, head_dim, device=device,
                             dtype=torch.bfloat16)
        metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
            max_query_len=tokens,
            block_table=torch.tensor([[0]], dtype=torch.int32, device=device),
            dcp_context_kv_lens=torch.tensor([6], dtype=torch.int32),
            max_dcp_context_kv_len=6,
            causal=True,
            scheduler_metadata=None,
        )
        recorder = _FlashRecorder(_kernel_reference)
        gathers = []
        cp_calls = []
        merge_calls = []
        group = SimpleNamespace(
            all_gather=lambda tensor, dim: (
                gathers.append((tensor, dim)) or tensor
            )
        )

        def _cp(out, lse, dcp_group, return_lse=False):
            cp_calls.append((out, lse, dcp_group, return_lse))
            return out, lse

        def _merge(out, context_out, context_lse, query_out, query_lse):
            merge_calls.append((context_out, context_lse, query_out, query_lse))
            out.fill_(-1)

        monkeypatch.setattr(fa, "get_dcp_group", lambda: group)
        monkeypatch.setattr(fa, "flash_attn_varlen_func", recorder)
        monkeypatch.setattr(fa, "cp_lse_ag_out_rs", _cp)
        monkeypatch.setattr(fa, "merge_attn_states", _merge)
        impl = _impl(num_heads=heads, head_size=head_dim, num_kv_heads=kv_heads)

        impl._forward_with_dcp(
            query, key, value, key_cache, value_cache, output, metadata,
            q_descale=torch.ones(1, kv_heads),
            k_descale=torch.ones(1, kv_heads),
            v_descale=torch.ones(1, kv_heads),
        )

        assert gathers[0][1] == 1
        assert gathers[0][0].is_contiguous()
        context_call, query_call = recorder.calls
        assert context_call["k"] is key_cache
        assert context_call["v"] is value_cache
        assert context_call["out"] is None
        assert context_call["seqused_k"] is metadata.dcp_context_kv_lens
        assert context_call["max_seqlen_k"] == 6
        assert context_call["causal"] is False
        assert context_call["return_softmax_lse"] is True
        assert max(context_call["cu_seqlens_q"].tolist()) == tokens
        assert query_call["k"] is key
        assert query_call["v"] is value
        assert query_call["cu_seqlens_k"] is metadata.query_start_loc
        assert query_call["max_seqlen_k"] == tokens
        assert query_call["causal"] is True
        assert len(cp_calls) == 1
        assert cp_calls[0][2] is group
        assert len(merge_calls) == 1
        assert merge_calls[0][0].shape == (tokens, heads, head_dim)
        assert merge_calls[0][2].shape == (tokens, heads, head_dim)
        assert torch.all(output == -1)


class TestSlidingWindowConfigs:
    """``_get_sliding_window_configs`` collects the per-layer windows."""

    def _impl_with_window(self, window):
        impl = object.__new__(fa.AttentionFLImpl)
        impl.sliding_window = window
        return impl

    def test_collects_unique_windows(self, monkeypatch):
        """Duplicate windows collapse into one set entry per configuration."""
        layers = {
            "a": SimpleNamespace(impl=self._impl_with_window((-1, -1))),
            "b": SimpleNamespace(impl=self._impl_with_window((7, 0))),
            "c": SimpleNamespace(impl=self._impl_with_window((7, 0))),
        }
        calls = []
        monkeypatch.setattr(
            fa,
            "get_layers_from_vllm_config",
            lambda config, cls: calls.append((config, cls)) or layers,
        )
        windows = fa._get_sliding_window_configs("config-sentinel")
        assert windows == {(-1, -1), (7, 0)}
        assert calls == [("config-sentinel", fa.Attention)]

    def test_non_fl_impl_is_rejected(self, monkeypatch):
        """A layer whose impl is not AttentionFLImpl trips the assertion."""
        layers = {"a": SimpleNamespace(impl=object())}
        monkeypatch.setattr(
            fa, "get_layers_from_vllm_config", lambda config, cls: layers
        )
        with pytest.raises(AssertionError):
            fa._get_sliding_window_configs("config")


class TestBuilderCascadeDelegate:
    """The builder forwards the cascade decision to the module helper."""

    def test_use_cascade_attention_delegates(self, monkeypatch):
        """Arguments reach the module-level heuristic unchanged."""
        calls = []

        def _decide(*args, **kwargs):
            calls.append((args, kwargs))
            return "decision"

        monkeypatch.setattr(fa, "use_cascade_attention", _decide)
        builder = _builder()
        result = builder.use_cascade_attention(300, [1], num_sms=4)
        assert result == "decision"
        assert calls == [((300, [1]), {"num_sms": 4})]
