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

"""Attention-sink plumbing of the HY4 FlashMLA sparse backend.

``HYV4FlashMLASparseImpl`` threads a per-head float32 sink through the two
FlashMLA kernel wrappers: the sink is validated at construction, laid out
for the kernel's (possibly padded) head count, and forwarded verbatim to the
vendored kernels. These tests exercise that plumbing on the GPU with small
head counts, comparing against references computed independently in plain
PyTorch, and confirm the backend class-level contract that
``HYV4MLAAttention`` construction relies on.

The FlashMLA kernels themselves need SM90+, so the kernel wrappers are driven
with module-bound kernel recorders (the same seam the plugin's own
``_install_fallback`` rebinding uses), letting the wrapper logic - query
padding, output slicing, sink forwarding - run on real CUDA tensors.
"""

from types import SimpleNamespace

import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.hy4 import flashmla_sparse as sparse

pytestmark = [pytest.mark.gpu]

RTOL = 1e-4
ATOL = 1e-4


def _impl(num_heads: int, sinks, scale: float = 0.125):
    """Build an impl carrying only the attributes the wrapper methods read."""
    impl = object.__new__(sparse.HYV4FlashMLASparseImpl)
    impl.num_heads = num_heads
    impl.softmax_scale = scale
    impl.prefill_padding = 64
    impl.fp8_decode_padded_heads = 64 if num_heads <= 64 else 128
    impl.sinks = sinks
    return impl


class TestValidateSinks:
    """Construction-time rejection of sink tensors the kernels cannot eat."""

    def test_none_is_accepted(self):
        """None is accepted."""
        assert sparse.HYV4FlashMLASparseImpl._validate_sinks(None, 4) is None

    def test_float32_per_head_shape_is_accepted(self):
        """Float32 per-head shape is accepted."""
        sinks = torch.arange(4, dtype=torch.float32)
        assert sparse.HYV4FlashMLASparseImpl._validate_sinks(sinks, 4) is None

    def test_wrong_dtype_is_rejected(self):
        """Wrong dtype is rejected."""
        sinks = torch.arange(4, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="must have dtype"):
            sparse.HYV4FlashMLASparseImpl._validate_sinks(sinks, 4)

    @pytest.mark.parametrize("shape", [(4, 1), (3,), (2, 2), ()])
    def test_wrong_shape_is_rejected(self, shape):
        """Wrong shape is rejected."""
        sinks = torch.ones(shape, dtype=torch.float32)
        with pytest.raises(ValueError, match=r"must have shape \(4,\)"):
            sparse.HYV4FlashMLASparseImpl._validate_sinks(sinks, 4)


class TestSinksForQuery:
    """Sink layout resolution for the kernel head count."""

    def test_no_sink_layer_yields_none(self, device):
        """No sink layer yields none."""
        impl = _impl(4, None)
        q = torch.zeros(2, 4, 8, device=device)
        assert impl._sinks_for_query(q, head_dim=1, kernel_heads=4) is None

    def test_exact_match_passes_the_same_tensor(self, device):
        """Exact match passes the same tensor."""
        sinks = torch.linspace(-1.0, 2.0, 4, device=device)
        impl = _impl(4, sinks)
        q = torch.zeros(2, 4, 8, device=device)
        assert impl._sinks_for_query(q, head_dim=1, kernel_heads=4) is sinks

    def test_padding_appends_minus_inf_lanes(self, device):
        # 5 query heads, kernel wants 8: lanes 4..7 must be -inf (no-op
        # sinks), lanes 0..4 must keep the trained values exactly.
        """Padding appends minus inf lanes."""
        sinks = torch.linspace(-2.0, 3.0, 5, device=device)
        impl = _impl(5, sinks)
        q = torch.zeros(3, 5, 8, device=device)
        padded = impl._sinks_for_query(q, head_dim=1, kernel_heads=8)
        reference = torch.full((8,), float("-inf"), device=device)
        reference[:5] = sinks
        assert padded.shape == (8,)
        torch.testing.assert_close(padded, reference, rtol=0, atol=0)

    def test_head_count_mismatch_is_rejected(self, device):
        """Head count mismatch is rejected."""
        sinks = torch.zeros(4, device=device)
        impl = _impl(4, sinks)
        q = torch.zeros(2, 3, 8, device=device)
        with pytest.raises(ValueError, match="sink head count must match"):
            impl._sinks_for_query(q, head_dim=1, kernel_heads=4)

    def test_cross_device_sink_is_rejected(self, device):
        """Cross-device sink is rejected."""
        sinks = torch.zeros(4, dtype=torch.float32)
        impl = _impl(4, sinks)
        q = torch.zeros(2, 4, 8, device=device)
        with pytest.raises(ValueError, match="same device"):
            impl._sinks_for_query(q, head_dim=1, kernel_heads=4)

    def test_kernel_below_query_heads_is_rejected(self, device):
        """Kernel below query heads is rejected."""
        sinks = torch.zeros(8, device=device)
        impl = _impl(8, sinks)
        q = torch.zeros(2, 8, 4, device=device)
        with pytest.raises(ValueError, match="cannot be smaller than"):
            impl._sinks_for_query(q, head_dim=1, kernel_heads=4)


def _fp8_metadata(device, num_tokens: int):
    """Kernel metadata stand-in carrying only the fields the wrapper reads."""
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseMetadata,
    )

    return FlashMLASparseMetadata.FP8KernelMetadata(
        scheduler_metadata=SimpleNamespace(),
        dummy_block_table=torch.zeros(num_tokens, 1, dtype=torch.int32, device=device),
        cache_lens=torch.full((num_tokens,), 64, dtype=torch.int32, device=device),
    )


class TestFP8KernelWrapper:
    """Query padding/slicing and sink forwarding of the FP8 decode wrapper."""

    def test_pads_query_and_slices_output_back(self, device, monkeypatch):
        # 32 query heads but the FP8 kernel wants 64: the recorder checks
        # the padded head count and that the real lanes survive the round
        # trip; padded lanes must be exactly zero.
        """Pads query and slices output back."""
        num_heads, num_tokens, head_dim = 32, 2, 576
        impl = _impl(num_heads, torch.linspace(-1.0, 1.0, num_heads, device=device))

        seen = {}

        def kernel(**kwargs):
            seen.update(kwargs)
            q = kwargs["q"]
            out = q.sum(dim=-1, keepdim=True).expand(-1, -1, -1, head_dim)
            return out, torch.zeros(q.shape[:3], device=q.device)

        monkeypatch.setattr(sparse, "flash_mla_with_kvcache", kernel)
        q = torch.randn(1, num_tokens, num_heads, head_dim, device=device)
        cache = torch.zeros(256, 684, dtype=torch.uint8, device=device)
        indices = torch.zeros(num_tokens, 8, dtype=torch.int32, device=device)
        out, _ = impl._fp8_flash_mla_kernel(
            q, cache, indices, _fp8_metadata(device, num_tokens)
        )

        assert seen["q"].shape == (1, num_tokens, 64, head_dim)
        torch.testing.assert_close(seen["q"][:, :, :num_heads], q, rtol=0, atol=0)
        assert bool((seen["q"][:, :, num_heads:] == 0).all())
        assert seen["is_fp8_kvcache"] is True
        assert seen["head_dim_v"] == 512
        assert seen["softmax_scale"] == impl.softmax_scale
        assert seen["indices"] is indices
        torch.testing.assert_close(
            seen["attn_sink"],
            torch.cat([impl.sinks, torch.full((32,), float("-inf"), device=device)]),
            rtol=0,
            atol=0,
        )
        # Output is sliced back to the real head count; equals per-head sums.
        assert out.shape == (1, num_tokens, num_heads, head_dim)
        reference = q.sum(dim=-1, keepdim=True).expand(-1, -1, -1, head_dim)
        torch.testing.assert_close(out, reference, rtol=RTOL, atol=ATOL)

    def test_no_padding_when_heads_already_match_kernel(self, device, monkeypatch):
        # 64 heads: no padding, the query passes through untouched and the
        # sink reaches the kernel unmodified.
        """No padding when heads already match kernel."""
        num_heads, num_tokens, head_dim = 64, 1, 576
        impl = _impl(num_heads, torch.linspace(0.5, 1.5, num_heads, device=device))

        seen = {}

        def kernel(**kwargs):
            seen.update(kwargs)
            q = kwargs["q"]
            return q[..., :1].contiguous(), torch.zeros(1, device=q.device)

        monkeypatch.setattr(sparse, "flash_mla_with_kvcache", kernel)
        q = torch.randn(1, num_tokens, num_heads, head_dim, device=device)
        cache = torch.zeros(256, 684, dtype=torch.uint8, device=device)
        indices = torch.zeros(num_tokens, 8, dtype=torch.int32, device=device)
        out, _ = impl._fp8_flash_mla_kernel(
            q, cache, indices, _fp8_metadata(device, num_tokens)
        )

        assert seen["q"] is q
        assert seen["attn_sink"] is impl.sinks
        assert out.shape == (1, num_tokens, num_heads, 1)
        torch.testing.assert_close(out, q[..., :1], rtol=0, atol=0)


class TestBF16KernelWrapper:
    """Query padding and sink forwarding of the BF16 prefill wrapper."""

    def test_pads_query_to_prefill_padding_and_slices_back(self, device, monkeypatch):
        # 4 heads with prefill_padding 64: the source zero-pads (not
        # new_empty) because the kernel reduces across the head group and
        # NaNs from uninitialized lanes would leak into the real heads; the
        # recorder reduces the same way so a NaN would change the result.
        """Pads query to prefill padding and slices back."""
        num_heads, num_tokens, head_dim = 4, 3, 576
        impl = _impl(num_heads, torch.linspace(-1.0, 1.5, num_heads, device=device))

        seen = {}

        def kernel(q, cache, indices, scale, *, attn_sink, topk_length):
            seen.update(
                q=q,
                cache=cache,
                indices=indices,
                scale=scale,
                attn_sink=attn_sink,
                topk_length=topk_length,
            )
            return (
                q.sum(dim=-1, keepdim=True) * torch.sigmoid(-attn_sink)[None, :, None],
            )

        monkeypatch.setattr(sparse, "flash_mla_sparse_fwd", kernel)
        q = torch.randn(num_tokens, num_heads, head_dim, device=device)
        cache = torch.zeros(256, 684, dtype=torch.uint8, device=device)
        indices = torch.zeros(num_tokens, 16, dtype=torch.int32, device=device)
        length = torch.full((num_tokens,), 16, dtype=torch.int32, device=device)
        out = impl._bf16_flash_mla_kernel(q, cache, indices, length)

        assert seen["q"].shape == (num_tokens, 64, head_dim)
        torch.testing.assert_close(seen["q"][:, :num_heads], q, rtol=0, atol=0)
        assert bool((seen["q"][:, num_heads:] == 0).all())
        # Cache viewed to [blocks, 1, width]; indices viewed to [tokens, 1, k].
        assert seen["cache"].shape == (256, 1, 684)
        assert seen["indices"].shape == (num_tokens, 1, 16)
        assert seen["scale"] == impl.softmax_scale
        assert seen["topk_length"] is length
        reference = (
            q.sum(dim=-1, keepdim=True) * torch.sigmoid(-impl.sinks)[None, :, None]
        )
        assert out.shape == (num_tokens, num_heads, 1)
        torch.testing.assert_close(out, reference, rtol=RTOL, atol=ATOL)

    def test_multiple_of_padding_passes_query_unchanged(self, device, monkeypatch):
        """Multiple of padding passes query unchanged."""
        num_heads, num_tokens, head_dim = 64, 2, 576
        impl = _impl(num_heads, torch.linspace(0.0, 1.0, num_heads, device=device))

        seen = {}

        def kernel(q, cache, indices, scale, *, attn_sink, topk_length):
            seen.update(q=q, attn_sink=attn_sink)
            return (q[..., :2].contiguous(),)

        monkeypatch.setattr(sparse, "flash_mla_sparse_fwd", kernel)
        q = torch.randn(num_tokens, num_heads, head_dim, device=device)
        cache = torch.zeros(16, 684, dtype=torch.uint8, device=device)
        indices = torch.zeros(num_tokens, 4, dtype=torch.int32, device=device)
        out = impl._bf16_flash_mla_kernel(q, cache, indices)

        assert seen["q"] is q
        assert seen["attn_sink"] is impl.sinks
        assert out.shape == (num_tokens, num_heads, 2)
        torch.testing.assert_close(out, q[..., :2], rtol=0, atol=0)


class TestBackendContract:
    """Class-level contract consumed by HYV4MLAAttention construction."""

    def test_impl_class_and_sink_support(self):
        """Expose the HY4 FlashMLA implementation and advertise sink support."""
        assert sparse.HYV4FlashMLASparseBackend.get_impl_cls() is (
            sparse.HYV4FlashMLASparseImpl
        )
        assert sparse.HYV4FlashMLASparseBackend.supports_sink() is True

    def test_parent_name_is_kept_for_kv_cache_promotion(self):
        # The name deliberately matches the platform backend so
        # _canonicalize_sparse_mla_kv_cache_dtype keeps promoting the KV
        # cache to fp8_ds_mla; a new name would silently change behavior.
        """Parent name is kept for KV cache promotion."""
        from vllm.v1.attention.backends.mla.flashmla_sparse import (
            FlashMLASparseBackend,
        )

        assert sparse.HYV4FlashMLASparseBackend.get_name() == "FLASHMLA_SPARSE"
        assert (
            sparse.HYV4FlashMLASparseBackend.get_name()
            == FlashMLASparseBackend.get_name()
        )
