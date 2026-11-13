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

"""Single-GPU tests for vllm_fl/models/glm5/sparse_indexer.py helpers (M1).

The module-level helpers are pure index/geometry math over small tensors.
Each test builds tiny decode batches and compares against scalar references
computed with independent loops. ``_kpool_compress_insert`` routes through
``INDEXER_BACKEND``; its contract is checked by asserting the gathered pool
windows and the write mask passed to the backend, using the real backend
object (its preflight-mode calls run on any device).
"""

import pytest
import torch

from . import _import_shim  # noqa: F401  (must run before vllm_fl import)
from vllm_fl.models.glm5.sparse_indexer import (
    _build_decode_scatter_indices,
    _decode_topk_seq_lens,
    _decode_write_layout,
    _gather_workspace_shapes,
    _kpool_compress_insert,
    _scatter_decode_tokens_by_request,
    kv_cache_as_quant_view,
)

pytestmark = [pytest.mark.gpu]


class TestBuildDecodeScatterIndices:
    """Per-token (request id, intra-request index) from decode lengths."""

    def test_matches_scalar_reference_with_empty_request(self, device):
        """Matches scalar reference with empty request."""
        decode_lens = torch.tensor([2, 0, 3], dtype=torch.int32, device=device)
        req_id, intra = _build_decode_scatter_indices(decode_lens, 3, 5)
        lens = decode_lens.cpu().tolist()
        ref_req = [r for r, length in enumerate(lens) for _ in range(length)]
        ref_intra = [i for length in lens for i in range(length)]
        assert req_id.tolist() == ref_req
        assert intra.tolist() == ref_intra
        assert req_id.dtype == torch.int64 and intra.dtype == torch.int64

    def test_lengths_sum_must_match_n(self, device):
        """Lengths sum must match n."""
        decode_lens = torch.tensor([1, 1, 1], dtype=torch.int32, device=device)
        req_id, intra = _build_decode_scatter_indices(decode_lens, 3, 3)
        assert req_id.tolist() == [0, 1, 2]
        assert intra.tolist() == [0, 0, 0]


class TestScatterDecodeTokensByRequest:
    """Int tokens scatter into a padded per-request grid."""

    def test_int_tokens_scatter_per_request(self, device):
        """Integer tokens scatter per request."""
        decode_lens = torch.tensor([2, 0, 3], dtype=torch.int32, device=device)
        req_id, intra = _build_decode_scatter_indices(decode_lens, 3, 5)
        tokens = torch.arange(5, dtype=torch.int32, device=device)
        out = _scatter_decode_tokens_by_request(tokens, -1, 3, 3, (req_id, intra))
        # request 0 -> tokens 0,1; request 1 empty; request 2 -> 2,3,4
        assert out.tolist() == [[0, 1, -1], [-1, -1, -1], [2, 3, 4]]
        assert out.dtype == torch.int32

    def test_multidimensional_tokens_keep_trailing_shape(self, device):
        """Multidimensional tokens keep trailing shape."""
        decode_lens = torch.tensor([2, 1], dtype=torch.int32, device=device)
        req_id, intra = _build_decode_scatter_indices(decode_lens, 2, 3)
        tokens = torch.arange(12, dtype=torch.int64, device=device).reshape(3, 2, 2)
        out = _scatter_decode_tokens_by_request(tokens, -1, 2, 2, (req_id, intra))
        assert out.shape == (2, 2, 2, 2)
        # Request 0 owns tokens 0 and 1 (rows 0 and 1 of the grid).
        assert out[0, 0].tolist() == tokens[0].tolist()
        assert out[0, 1].tolist() == tokens[1].tolist()
        # Request 1 owns only token 2; its second grid row stays padded.
        assert out[1, 0].tolist() == tokens[2].tolist()
        assert (out[1, 1] == -1).all()

    def test_float_tokens_route_to_backend(self, device):
        """Floating tensors delegate to INDEXER_BACKEND.scatter_decode_tokens."""
        from vllm_fl.models.glm5 import sparse_indexer

        decode_lens = torch.tensor([1, 1], dtype=torch.int32, device=device)
        req_id, intra = _build_decode_scatter_indices(decode_lens, 2, 2)
        tokens = torch.randn(2, 4, device=device)
        captured = {}

        def fake_scatter(t, pad, num_requests, lmax, indices):
            captured["shape"] = tuple(t.shape)
            captured["pad"] = pad
            captured["grid"] = (num_requests, lmax)
            return "ROUTED"

        original = sparse_indexer.INDEXER_BACKEND.scatter_decode_tokens
        sparse_indexer.INDEXER_BACKEND.scatter_decode_tokens = fake_scatter
        try:
            out = _scatter_decode_tokens_by_request(tokens, 0.0, 2, 2, (req_id, intra))
        finally:
            sparse_indexer.INDEXER_BACKEND.scatter_decode_tokens = original
        assert out == "ROUTED"
        assert captured == {"shape": (2, 4), "pad": 0.0, "grid": (2, 2)}


class TestDecodeWriteLayout:
    """One decode token per request; graph padding adds empty rows."""

    @staticmethod
    def _metadata(requires_padding, decode_lens):
        return type(
            "Meta",
            (),
            {"requires_padding": requires_padding, "decode_lens": decode_lens},
        )()

    def test_uniform_layout_is_flagged_and_len_one(self, device):
        """Uniform layout is flagged and length one."""
        lens = torch.tensor([1, 1, 1], device=device)
        uniform, out_lens, tokens_per_req = _decode_write_layout(
            self._metadata(False, lens), 3, 3
        )
        assert uniform is True
        assert out_lens is lens
        assert tokens_per_req == 1

    def test_padded_layout_with_fewer_tokens(self, device):
        """Padded layout with fewer tokens."""
        lens = torch.tensor([1, 1, 1, 0], device=device)
        uniform, out_lens, tokens_per_req = _decode_write_layout(
            self._metadata(True, lens), 4, 3
        )
        assert uniform is False
        assert tokens_per_req == 1

    def test_more_tokens_than_requests_rejected(self, device):
        """vllm_fl/models/glm5/sparse_indexer.py:155 layout guard."""
        for requires_padding in (False, True):
            lens = torch.tensor(
                [2, 1, 2] if requires_padding else [1, 1, 1], device=device
            )
            with pytest.raises(ValueError, match="one decode token per request"):
                _decode_write_layout(self._metadata(requires_padding, lens), 3, 5)

    def test_non_padded_count_mismatch_rejected(self, device):
        """Unpadded count mismatch rejected."""
        lens = torch.tensor([1, 1], device=device)
        with pytest.raises(ValueError, match="one decode token per request"):
            _decode_write_layout(self._metadata(False, lens), 2, 3)
        with pytest.raises(ValueError, match="one decode token per request"):
            _decode_write_layout(self._metadata(False, lens), 3, 2)


class TestDecodeTopkSeqLens:
    """Token-granular seq_len (pos+1) per pool-topk row, layout-aware."""

    def test_uniform_layout_slices_flat_positions(self, device):
        """Uniform layout slices flat positions."""
        positions = torch.tensor(
            [10, 11, 20, 21, 30, 99], dtype=torch.int32, device=device
        )
        decode_lens = torch.tensor([2, 0, 3], dtype=torch.int32, device=device)
        out = _decode_topk_seq_lens(positions, decode_lens, 6, 2, 3, False)
        assert out.tolist() == [11, 12, 21, 22, 31, 100]
        assert out.dtype == torch.int32

    def test_padded_layout_scatters_per_request_grid(self, device):
        """Row (b, t) reads flat decode token offset_b + t, not b*next_n+t."""
        positions = torch.tensor(
            [10, 11, 20, 21, 30, 99], dtype=torch.int32, device=device
        )
        decode_lens = torch.tensor([2, 0, 3], dtype=torch.int32, device=device)
        out = _decode_topk_seq_lens(positions, decode_lens, 5, 3, 3, True)
        # grid rows: b0=(10,11,pad), b1=pad row, b2=(20,21,30)
        assert out.tolist() == [11, 12, 0, 0, 0, 0, 21, 22, 31]

    def test_padded_layout_with_three_nonempty_requests(self, device):
        """Padded layout with three nonempty requests."""
        positions = torch.tensor(
            [10, 11, 20, 21, 30, 99], dtype=torch.int32, device=device
        )
        decode_lens = torch.tensor([2, 3, 1], dtype=torch.int32, device=device)
        out = _decode_topk_seq_lens(positions, decode_lens, 6, 3, 3, True)
        # b0=(10,11), b1=(20,21,30), b2=(99)
        assert out.tolist() == [11, 12, 0, 21, 22, 31, 100, 0, 0]


class TestGatherWorkspaceShapes:
    """Workspace shapes per cache format."""

    @pytest.mark.parametrize("head_dim", [128, 576])
    def test_fp8_shapes(self, device, head_dim):
        """Verify the operand dimensions for the FP8 indexer layout."""
        values, scales = _gather_workspace_shapes(
            17, head_dim, torch.float8_e4m3fn, False
        )
        assert values == ((17, head_dim), torch.float8_e4m3fn)
        assert scales == ((17, 4), torch.uint8)

    @pytest.mark.parametrize("head_dim", [128, 576])
    def test_fp4_shapes(self, device, head_dim):
        """Verify the packed operand dimensions for the FP4 indexer layout."""
        values, scales = _gather_workspace_shapes(
            17, head_dim, torch.float8_e4m3fn, True
        )
        # MXFP4 packs 2 values per byte; one ue8m0 scale per 32 values.
        assert values == ((17, head_dim // 2), torch.uint8)
        assert scales == ((17, head_dim // 32), torch.uint8)


class TestKvCacheAsQuantView:
    """4D DeepGEMM views from the 3D indexer cache allocation."""

    def test_fp4_view_shape_and_strides(self, device):
        """FP4 view shape and strides."""
        fp4_bytes = 128 // 2 + 128 // 32  # 64 packed + 4 scale bytes
        blocks, block_size = 3, 8
        # The canonical allocation: [num_blocks, block_size, fp4_bytes].
        cache = torch.full(
            (blocks, block_size, fp4_bytes), 7, dtype=torch.uint8, device=device
        )
        view = kv_cache_as_quant_view(cache, 128, True)
        assert view.shape == (blocks, block_size, 1, fp4_bytes)
        assert view.stride() == (block_size * fp4_bytes, fp4_bytes, fp4_bytes, 1)
        assert view.data_ptr() == cache.data_ptr()
        assert (view == 7).all()
        # The block stride is read from the allocation's own stride(0):
        # a wider (non-contiguous) page layout keeps its page stride.
        wide = cache.transpose(0, 1).contiguous().transpose(0, 1)
        wide_view = kv_cache_as_quant_view(wide, 128, True)
        assert wide_view.stride()[0] == int(wide.stride(0))

    def test_fp8_view_is_simple_unsqueeze(self, device):
        """FP8 view is simple unsqueeze."""
        cache = torch.randn(3, 8, 128, device=device)
        view = kv_cache_as_quant_view(cache, 128, False)
        assert view.shape == (3, 8, 1, 128)
        assert torch.equal(view.squeeze(2), cache)


class TestKpoolCompressInsert:
    """Pool-window gather and write-mask contract of the prefill helper."""

    def _capture(self, device):
        from vllm_fl.models.glm5 import sparse_indexer

        captured = {}

        class RecordingBackend:
            def gather_rows(self, k, idx):
                captured.setdefault("gather", []).append(idx.cpu().clone())
                return k[idx.cpu().to(k.device)]

            def kpool_compress_and_write_cache(
                self, kv_cache, k_pooled, gate_pooled, ape, slot_mapping, **kw
            ):
                captured["write"] = {
                    "k_pooled": k_pooled.cpu().clone(),
                    "slot_dtype": slot_mapping.dtype,
                    "pool_size": kw["pool_size"],
                    "write_mask": kw["write_mask"].cpu().clone(),
                    "write_cache": kw["write_cache"],
                }

        original = sparse_indexer.INDEXER_BACKEND
        return sparse_indexer, original, RecordingBackend(), captured

    def test_passes_pool_windows_and_write_mask(self, device):
        """Pass complete pool windows and their valid write slots to the cache kernel."""
        torch.manual_seed(5)
        n, head_dim, kpool = 12, 8, 4
        k = torch.randn(n, head_dim, device=device)
        gate = torch.randn(n, device=device)
        cache = torch.zeros(4, 64, head_dim, device=device)
        slot = torch.tensor(
            [-1, -1, -1, 64, -1, -1, -1, 65, -1, -1, -1, 66], device=device
        )
        module, original, recording, captured = self._capture(device)
        module.INDEXER_BACKEND = recording
        try:
            _kpool_compress_insert(
                k,
                gate,
                torch.zeros((), device=device),
                cache,
                slot,
                kpool,
                head_dim,
                False,
            )
        finally:
            module.INDEXER_BACKEND = original
        # Every position becomes a completion candidate: idx rows are
        # (pos-(kpool-1)).clamp_min(0) + [0..kpool).
        idx = captured["gather"][0]
        expected_rows = [
            [max(pos - (kpool - 1), 0) + off for off in range(kpool)]
            for pos in range(n)
        ]
        assert idx.tolist() == expected_rows
        # The pooled K passed downstream is exactly the gathered windows.
        pooled_k = captured["write"]["k_pooled"]
        assert torch.equal(pooled_k, k[idx.to(k.device)].cpu())
        # Only positions >= kpool-1 with a valid slot are written.
        write_mask = captured["write"]["write_mask"]
        valid = (slot.cpu() >= 0) & (torch.arange(n) >= kpool - 1)
        assert torch.equal(write_mask, valid)
        assert captured["write"]["slot_dtype"] == torch.int64
        assert captured["write"]["pool_size"] == kpool
        assert captured["write"]["write_cache"] is True

    def test_batch_smaller_than_pool_returns_without_calls(self, device):
        """Leave the cache untouched when the batch has no complete pool."""
        torch.manual_seed(5)
        k = torch.randn(3, 8, device=device)
        gate = torch.randn(3, device=device)
        cache = torch.zeros(4, 64, 8, device=device)
        slot = torch.full((3,), -1, dtype=torch.int64, device=device)
        module, original, recording, captured = self._capture(device)
        module.INDEXER_BACKEND = recording
        try:
            _kpool_compress_insert(
                k, gate, torch.zeros((), device=device), cache, slot, 4, 8, False
            )
        finally:
            module.INDEXER_BACKEND = original
        assert captured == {}
