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

"""Single-GPU tests for the glm5 BF16 indexer adapters and host dispatch (M1).

``Bf16Indexer`` composes the pooling math (float32 softmax gates over the
pool window) with backend dispatch; every numerical test is checked against
an independent scalar reference computed in the test itself. The
``indexer_backend`` host surface tests pin the framework-metadata routing
and the boolean capability probes.
"""

import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5.indexer_bf16 import Bf16Indexer

pytestmark = [pytest.mark.gpu]


class _RecordingBackend:
    """Backend double capturing ``_kpool`` dispatches."""

    def __init__(self):
        self.calls = []

    def _kpool(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return args[0] if args else None


class TestBf16IndexerQuery:
    """``query`` returns the rotated values and unit fp32 scales."""

    def test_query_routes_hadamard_and_returns_unit_scales(self, device):
        """Query routes hadamard and returns unit scales."""
        backend = _RecordingBackend()
        indexer = Bf16Indexer(backend)
        q = torch.randn(6, 8, device=device)
        values, scales = indexer.query(q)
        assert backend.calls[0][0] == "hadamard128"
        assert backend.calls[0][1] == (q,)
        assert scales.shape == (6, 1) and scales.dtype == torch.float32
        assert torch.equal(scales, torch.ones_like(scales))


class TestBf16IndexerInsert:
    """``insert`` drops the quantization arguments and stores bf16 rows."""

    def test_insert_drops_quant_args_and_stores_bf16(self, device):
        """Insert drops quant args and stores BF16."""
        backend = _RecordingBackend()
        indexer = Bf16Indexer(backend)
        k = torch.randn(4, 8, device=device)
        cache = torch.zeros(4, 64, 8, device=device)
        slots = torch.tensor([64, 65, 66, 67], device=device)
        indexer.insert(k, cache, slots, quant_block_size=16, scale_fmt="ue8m0")
        assert [call[0] for call in backend.calls] == ["qsa_store_cache_rows"]
        _, args, _ = backend.calls[0]
        cache_arg, slots_arg, rows = args
        assert torch.equal(rows, k.to(torch.bfloat16))
        assert slots_arg.tolist() == [64, 65, 66, 67]
        # The cache view is the same storage, widened with a head dim.
        assert cache_arg.data_ptr() == cache.data_ptr()
        assert cache_arg.shape == (4, 64, 1, 8)


class TestBf16IndexerCompress:
    """``compress`` pools the window with float32 softmax gates."""

    @staticmethod
    def _inputs(device, seed=3):
        torch.manual_seed(seed)
        return (
            torch.randn(5, 4, 8, device=device),  # keys [n, pool, head]
            torch.randn(5, 4, 1, device=device),  # scores
            torch.randn(5, 4, 1, device=device),  # ape
            torch.zeros(8, 64, 8, device=device),  # cache
            torch.arange(100, 105, device=device),  # slots
        )

    def test_pooled_rows_match_scalar_reference(self, device):
        """pooled = sum(keys * softmax(scores + ape)) cast to bf16."""
        keys, scores, ape, cache, slots = self._inputs(device)
        backend = _RecordingBackend()
        Bf16Indexer(backend).compress(cache, keys, scores, ape, slots, 4, head_dim=8)
        _, args, _ = backend.calls[-1]
        _, slots_arg, rows = args
        gates = torch.softmax((scores + ape).float(), dim=1)
        expected = (keys.float() * gates).sum(dim=1).to(torch.bfloat16)
        assert torch.equal(rows, expected)
        assert slots_arg.tolist() == [100, 101, 102, 103, 104]

    def test_write_mask_masks_slots_to_minus_one(self, device):
        """Write mask masks slots to -1."""
        keys, scores, ape, cache, slots = self._inputs(device)
        backend = _RecordingBackend()
        write_mask = torch.tensor([True, False, True, False, True], device=device)
        out = Bf16Indexer(backend).compress(
            cache,
            keys,
            scores,
            ape,
            slots,
            4,
            head_dim=8,
            write_mask=write_mask,
            return_compressed=True,
        )
        _, args, _ = backend.calls[-1]
        _, slots_arg, _ = args
        assert slots_arg.tolist() == [100, -1, 102, -1, 104]
        rotated, ones = out
        assert rotated.shape == (5, 8)
        assert torch.equal(ones, torch.ones(5, dtype=torch.float32, device=device))

    def test_write_cache_false_skips_the_store(self, device):
        """Write cache false skips the store."""
        keys, scores, ape, cache, slots = self._inputs(device)
        backend = _RecordingBackend()
        Bf16Indexer(backend).compress(
            cache,
            keys,
            scores,
            ape,
            slots,
            4,
            head_dim=8,
            write_cache=False,
            return_compressed=False,
        )
        assert "qsa_store_cache_rows" not in [c[0] for c in backend.calls]


class TestIndexerBackendHost:
    """Host dispatch: metadata ops route directly; probes are boolean."""

    def test_metadata_ops_route_to_kpool_metadata(self, device):
        """Metadata operators route to kpool metadata."""
        from vllm_fl.models.glm5.indexer_backend import INDEXER_BACKEND

        group_ids = torch.tensor([[0, 1], [2, 3]], device=device)
        valid = torch.ones(2, 2, dtype=torch.bool, device=device)
        got = INDEXER_BACKEND._kpool("expand_pools_to_tokens", group_ids, valid, 4, 2)
        # token = pool * pool_size + offset with pool_size=2: pools 2,3 -> 4..7.
        assert got.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert got.dtype == torch.int32

    def test_metadata_preflight_description(self):
        """Describe the metadata implementation without running a kernel."""
        from vllm_fl.models.glm5.indexer_backend import INDEXER_BACKEND

        for op in ("expand_pools_to_tokens", "append_tail_to_topk"):
            desc = INDEXER_BACKEND._kpool(op, _preflight=True)
            assert desc == {"op": op, "implementation": "framework.metadata"}

    def test_capability_probes_are_boolean(self):
        """Capability probes are boolean."""
        from vllm_fl.models.glm5.indexer_backend import INDEXER_BACKEND

        assert isinstance(INDEXER_BACKEND.use_fp8, bool)
        assert isinstance(INDEXER_BACKEND.use_torch_topk, bool)
        assert isinstance(INDEXER_BACKEND.is_nvidia, bool)
        assert isinstance(INDEXER_BACKEND.has_nvidia_deep_gemm, bool)

    def test_gather_rows_without_implementation_raises(self, device, monkeypatch):
        """Gather rows without implementation raises."""
        from vllm_fl.models.glm5.indexer_backend import Glm5NextIndexerBackend

        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "gather_rows")
        values = torch.zeros(2, 2, device=device)
        indices = torch.tensor([0], device=device)
        with pytest.raises(RuntimeError, match="gather_rows"):
            Glm5NextIndexerBackend().gather_rows(values, indices)
