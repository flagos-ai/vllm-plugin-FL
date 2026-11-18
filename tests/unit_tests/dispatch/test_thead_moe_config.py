# Copyright (c) 2026 BAAI. All rights reserved.

"""PPU MoE padding must not reach an unchecked Triton expert index."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_fl.dispatch.backends.reference.reference import ReferenceBackend
from vllm_fl.dispatch.config.utils import get_effective_config
from vllm_fl.dispatch.policy import PolicyManager


def test_detected_thead_selects_safe_moe_alignment(monkeypatch):
    monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
    with patch("vllm.platforms.current_platform", SimpleNamespace(vendor_name="thead")):
        config = get_effective_config()
        policy = PolicyManager()._policy_from_env()
    assert policy.per_op_order_dict["moe_align_block_size"] == ["reference"]
    assert config["op_backends"]["moe_align_block_size"] == ["reference"]
    # This is an operator-local repair, not a global switch away from FlagOS.
    assert config["prefer"] == "flagos"
    assert config["op_backends"]["attention_backend"][0] == "vendor:thead"


@pytest.mark.parametrize("block_size", [1, 2, 16])
@pytest.mark.parametrize(
    "routes",
    [
        [[0, -1], [2, -1], [1, 0]],
        [[-1, -1], [-1, -1], [-1, -1]],
        [[3, -2], [0, 1], [2, 99]],
    ],
)
def test_thead_reference_alignment_preserves_valid_routes(routes, block_size):
    ids = torch.tensor(routes, dtype=torch.int32)
    sorted_ids, experts, total = ReferenceBackend().moe_align_block_size(
        ids, block_size, 3
    )
    flat = ids.flatten().tolist()
    keep = int(total.item())
    placed = sorted_ids[:keep].tolist()
    assert sorted(slot for slot in placed if slot < ids.numel()) == [
        slot for slot, expert in enumerate(flat) if 0 <= expert < 3
    ]
    for offset in range(0, keep, block_size):
        owner = int(experts[offset // block_size])
        assert 0 <= owner < 3
        for slot in placed[offset : offset + block_size]:
            if slot < ids.numel():
                assert flat[slot] == owner
    assert (sorted_ids[keep:] == ids.numel()).all()
