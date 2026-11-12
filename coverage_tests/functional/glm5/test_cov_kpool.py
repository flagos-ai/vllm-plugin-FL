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

"""Single-GPU tests for vllm_fl/models/glm5/kpool.py (M1).

The kpool module carries the tail-ring slot mapper, the tail spec/backend
static contracts and the DeepGEMM page-alignment logic. Every mapping test
compares against an independent scalar loop; static-method tests pin the
declared constants; the platform-dependent layout tests patch
``current_platform.get_device_capability`` (a sanctioned boundary) and
restore it deterministically.
"""

import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5.kpool import (
    KpoolTailBackend,
    KpoolTailMetadataBuilder,
    KpoolTailSpec,
    compute_kpool_tail_slot_mapping,
    glm5_indexer_page_alignment,
)

pytestmark = [pytest.mark.gpu]


def _make_spec(**overrides):
    kwargs = dict(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.uint8,
        sliding_window=None,
    )
    kwargs.update(overrides)
    return KpoolTailSpec(**kwargs)


class TestKpoolTailSpec:
    """The tail spec admits one block per request and never prefix-caches."""

    def test_admission_cap_is_one_block(self):
        """Admission cap is one block."""
        spec = _make_spec()
        assert spec.max_admission_blocks_per_request(4096, 131072) == 1
        assert spec.max_admission_blocks_per_request(1, 1) == 1

    def test_never_participates_in_prefix_caching(self):
        """Keep the rolling KPool tail outside prefix caching."""
        assert _make_spec().participates_in_prefix_caching is False

    def test_uniformity_requires_all_tail_specs(self):
        """Uniformity requires all tail specifications."""
        spec = _make_spec()
        assert spec.is_uniform_with_collection({"a": _make_spec(), "b": spec}) is True
        assert spec.is_uniform_with_collection({"a": spec, "b": 5}) is False


class TestKpoolTailBackend:
    """Static contracts of the tail attention backend."""

    def test_kv_cache_shape_packs_two_fields_per_head(self):
        """(num_blocks, 2, block_size, head_size // 2)."""
        assert KpoolTailBackend.get_kv_cache_shape(7, 16, 1, 128) == (7, 2, 16, 64)
        assert KpoolTailBackend.get_kv_cache_shape(3, 32, 1, 576) == (3, 2, 32, 288)

    def test_kv_cache_shape_guards(self, device):
        """num_kv_heads must be 1 and head_size even."""
        for args in ((7, 16, 2, 128), (7, 16, 1, 127)):
            with pytest.raises(AssertionError):
                KpoolTailBackend.get_kv_cache_shape(*args)

    def test_declared_constants(self):
        """Check the KPool backend's advertised names and dimensions."""
        assert KpoolTailBackend.get_name() == "KPOOL_TAIL"
        assert KpoolTailBackend.get_supported_head_sizes() == []
        blocks = KpoolTailBackend.get_supported_kernel_block_sizes()
        assert len(blocks) == 1 and blocks[0].base == 1
        assert KpoolTailBackend.get_builder_cls() is KpoolTailMetadataBuilder
        assert KpoolTailBackend.get_kv_cache_stride_order() == (0, 1, 2, 3)
        assert KpoolTailBackend.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        ) == (0, 1, 2, 3)
        assert KpoolTailBackend.indexes_kv_by_block_stride() is True

    def test_cudagraph_always_supported(self):
        """CUDA graph always supported."""
        from vllm.v1.attention.backend import AttentionCGSupport

        assert (
            KpoolTailMetadataBuilder.get_cudagraph_support(None, None)
            == AttentionCGSupport.ALWAYS
        )


class TestComputeKpoolTailSlotMapping:
    """Tokens map into their request's single circular tail block."""

    def test_matches_scalar_reference(self, device):
        """block = table[req, 0]; slot = block*kpool + pos % kpool."""
        num_tokens, kpool = 7, 4
        slot = torch.full((10,), -9, dtype=torch.int64, device=device)
        block_table = torch.tensor(
            [[100, 0, 0], [200, 0, 0]], dtype=torch.int32, device=device
        )
        query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32, device=device)
        positions = torch.tensor(
            [0, 1, 2, 0, 1, 2, 3], dtype=torch.int64, device=device
        )
        out = compute_kpool_tail_slot_mapping(
            slot, block_table, query_start_loc, positions, num_tokens, 2, kpool
        )
        expected = [-9] * 10
        for token in range(num_tokens):
            request = 0 if token < 3 else 1
            block = int(block_table[request, 0].item())
            position = token if request == 0 else token - 3
            expected[token] = block * kpool + position % kpool
        assert out.tolist() == expected
        assert expected == [400, 401, 402, 800, 801, 802, 803, -9, -9, -9]

    def test_positions_wrap_within_the_ring(self, device):
        """pos >= kpool wraps back to slot pos % kpool."""
        kpool = 4
        slot = torch.zeros(5, dtype=torch.int64, device=device)
        block_table = torch.tensor([[10]], dtype=torch.int32, device=device)
        query_start_loc = torch.tensor([0, 5], dtype=torch.int32, device=device)
        positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
        out = compute_kpool_tail_slot_mapping(
            slot, block_table, query_start_loc, positions, 5, 1, kpool
        )
        assert out.tolist() == [40, 41, 42, 43, 40]

    def test_zero_tokens_returns_clone(self, device):
        """Zero tokens returns clone."""
        slot = torch.full((6,), -7, dtype=torch.int64, device=device)
        out = compute_kpool_tail_slot_mapping(
            slot,
            torch.zeros(1, 2, dtype=torch.int32, device=device),
            torch.tensor([0, 1], dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
            0,
            1,
            4,
        )
        assert torch.equal(out, slot)
        assert out.data_ptr() != slot.data_ptr()


class TestPageAlignment:
    """DeepGEMM page size by compute capability."""

    def test_sm90_and_below_require_64(self):
        """SM90 and below require 64."""

        class Capability:
            def __init__(self, major):
                self.major = major

        assert glm5_indexer_page_alignment(None) == 32
        for major in (8, 9):
            assert glm5_indexer_page_alignment(Capability(major)) == 64
        for major in (10, 12):
            assert glm5_indexer_page_alignment(Capability(major)) == 32

    def test_physical_layout_follows_capability(self):
        """Kernel page = storage when aligned; misalignment raises."""
        from vllm.platforms import current_platform

        from vllm_fl.models.glm5.kpool import Glm5NextIndexerAttentionBackend

        class Capability:
            def __init__(self, major):
                self.major = major

        class FakeSpec:
            def __init__(self, storage):
                self.block_size = 64
                self.storage_block_size = storage

        original = current_platform.get_device_capability
        try:
            cases = [
                (Capability(9), 64, 64),
                (Capability(10), 32, 32),
                (Capability(10), 64, 64),
                (Capability(8), 64, 64),
            ]
            for capability, storage, expected_page in cases:
                current_platform.get_device_capability = lambda c=capability: c
                layout = Glm5NextIndexerAttentionBackend.get_physical_cache_layout(
                    FakeSpec(storage)
                )
                assert layout.kernel_block_size == expected_page
                assert layout.storage_block_size == storage
            # Storage below the capability minimum is rejected.
            current_platform.get_device_capability = lambda: Capability(9)
            with pytest.raises(ValueError, match="must align to 64"):
                Glm5NextIndexerAttentionBackend.get_physical_cache_layout(FakeSpec(32))
        finally:
            current_platform.get_device_capability = original
