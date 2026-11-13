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

"""Single-GPU tests for vllm_fl.worker.common_attention_metadata.

The CPU-side builder behavior lives in ``coverage_tests/unit/worker/
test_cov_common_attention_metadata.py``. These tests exercise the real
accelerator path: device-resident pointer tables and the real Triton kernels
against explicit reference values, so both carry ``pytest.mark.gpu`` and live
in the functional tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

# The shim import is load-bearing: it must precede vllm_fl imports.
# isort: split
from . import _import_shim  # noqa: F401

# isort: split
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

import vllm_fl.worker.common_attention_metadata as cam


class _FakeTable:
    """BlockTable group stand-in with real CPU tensors for pointer math."""

    def __init__(
        self,
        *,
        rows=2,
        cols=4,
        block_size=16,
        max_tokens=64,
        pcp_world=1,
        dcp_world=1,
        pcp_rank=0,
        dcp_rank=0,
        interleave=1,
        slot_len=8,
        device="cpu",
    ):
        self.block_table = SimpleNamespace(
            gpu=torch.zeros(rows, cols, dtype=torch.int32, device=device)
        )
        self.slot_mapping = SimpleNamespace(
            gpu=torch.zeros(slot_len, dtype=torch.int64, device=device)
        )
        self.block_size = block_size
        self.max_num_batched_tokens = max_tokens
        self.pcp_world_size = pcp_world
        self.dcp_world_size = dcp_world
        self.pcp_rank = pcp_rank
        self.dcp_rank = dcp_rank
        self.cp_kv_cache_interleave_size = interleave


class _FakeBlockTable:
    def __init__(self, tables):
        self.block_tables = tables


@pytest.mark.gpu
def test_layout_pointer_tables_are_built_on_cuda():
    """Pointer/stride/size tables live on the device of the cache groups."""
    table = _FakeTable(slot_len=8, device="cuda")
    layout = cam._create_common_attention_metadata_layout(_FakeBlockTable([table]))

    assert layout.block_table_ptrs.device.type == "cuda"
    assert layout.block_table_strides.device.type == "cuda"
    assert layout.block_sizes.device.type == "cuda"
    assert layout.slot_mapping_ptrs.device.type == "cuda"
    assert layout.block_table_ptrs.tolist() == [table.block_table.gpu.data_ptr()]
    assert layout.block_table_strides.tolist() == [table.block_table.gpu.stride(0)]


@pytest.mark.gpu
def test_compute_real_kernels_match_reference_on_cuda():
    """The real Triton kernels produce the reference slot/computed mapping."""
    table = _FakeTable(
        rows=2,
        cols=4,
        block_size=4,
        max_tokens=8,
        slot_len=8,
        device="cuda",
    )
    table.block_table.gpu.copy_(
        torch.tensor([[10, 20, 0, 0], [30, 40, 0, 0]], dtype=torch.int32, device="cuda")
    )
    block_table = _FakeBlockTable([table])
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")
    positions = torch.tensor([0, 1, 2, 3, 4, 0, 0, 0], dtype=torch.int64, device="cuda")
    seq_lens = torch.tensor([5, 7], dtype=torch.int32, device="cuda")
    num_computed = torch.full((2,), -1, dtype=torch.int32, device="cuda")

    cam.compute_common_attention_metadata(
        block_table, 2, query_start_loc, positions, seq_lens, num_computed
    )

    # block_size=4, world/rank=1: slot = block_number * 4 + position % 4.
    expected_slots = torch.tensor(
        [40, 41, 122, 123, 160, PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID],
        dtype=torch.int64,
        device="cuda",
    )
    assert torch.equal(table.slot_mapping.gpu, expected_slots)
    # num_computed = seq_len - scheduled tokens: [5-2, 7-3].
    assert num_computed.tolist() == [3, 4]
