# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import pytest
import torch

from vllm_fl.kv_cache.ascend.state import state_row_indices


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("shape", [(9, 2, 3, 4), (9, 2, 3, 4, 5)])
def test_state_row_plan_preserves_shared_storage_and_offsets(shape, transpose):
    backing = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    state = backing[:, 1]
    if transpose:
        state = state.transpose(-1, -2)
    original = backing.clone()
    slots = torch.tensor([8, 0, 2])
    offsets = state_row_indices(state, slots)
    storage = backing.flatten()
    expected = torch.stack([state[int(slot)].clone() for slot in slots])
    torch.testing.assert_close(storage[offsets].reshape_as(expected), expected)
    storage.index_copy_(0, offsets, torch.full_like(offsets, -7))
    for slot in slots:
        original[slot, 1].fill_(-7)
    torch.testing.assert_close(backing, original)


def test_state_row_plan_rejects_unsupported_rank():
    with pytest.raises(ValueError, match="two or three dimensions"):
        state_row_indices(torch.zeros(2, 3), torch.tensor([1]))
