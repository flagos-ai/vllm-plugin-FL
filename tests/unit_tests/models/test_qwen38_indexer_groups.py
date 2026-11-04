# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager

import pytest
import torch
from _qwen38_cpu import ROOT, declarations

packing = declarations(
    ROOT / "vllm_fl/models/qwen3_8_flash_next/common/qsa_packing.py",
    "qwen38_indexer_groups_cpu",
    namespace={"contextmanager": contextmanager},
)


def test_indexer_groups_preserve_request_boundaries_and_step_lifetime():
    starts = torch.tensor([0, 3, 3, 20], dtype=torch.int32)
    with packing.qsa_metadata_build_scope():
        rows = packing._cached_indexer_pack_rows(starts, 20, torch.device("cpu"))
        assert rows is packing._cached_indexer_pack_rows(
            starts.clone(), 20, torch.device("cpu")
        )
        active = rows[rows >= 0]
        assert active.tolist() == list(range(20))
        assert rows.shape == (3, 16)
        for group in rows.tolist():
            requests = {0 if x < 3 else 2 for x in group if x >= 0}
            assert len(requests) <= 1
    with packing.qsa_metadata_build_scope():
        fresh = packing._cached_indexer_pack_rows(starts, 20, torch.device("cpu"))
        assert fresh is not rows
        assert torch.equal(fresh, rows)


def test_indexer_groups_include_padded_rows_without_gpu_readback():
    rows = packing._cached_indexer_pack_rows(
        torch.tensor([0, 3, 3, 20]), 24, torch.device("cpu")
    )
    assert rows[rows >= 0].tolist() == list(range(24))
    with pytest.raises(ValueError, match="already be on CPU"):
        packing._cached_indexer_pack_rows(
            torch.empty(3, device="meta"), 24, torch.device("cpu")
        )


@pytest.mark.parametrize(
    "starts,tokens,pack",
    [([], 0, 16), ([1, 2], 2, 16), ([0, 4, 3], 4, 16), ([0, 5], 4, 16), ([0, 4], 4, 4)],
)
def test_indexer_rejects_invalid_request_boundaries(starts, tokens, pack):
    with pytest.raises(ValueError):
        packing._cached_indexer_pack_rows(
            torch.tensor(starts, dtype=torch.int32), tokens, torch.device("cpu"), pack
        )


def test_indexer_empty_batch_is_valid():
    assert (
        packing._cached_indexer_pack_rows(torch.tensor([0]), 0, torch.device("cpu"))
        is None
    )
