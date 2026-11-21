# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for padding, mutation and nonfinite dummy rows."""

import importlib.util
from pathlib import Path

import pytest
import torch

_path = Path(__file__).parents[2] / "vllm_fl/ops/fused_moe/padding.py"
_spec = importlib.util.spec_from_file_location("fl_moe_padding_test", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_padding_preserves_active_routes_and_never_mutates_inputs(dtype, id_dtype):
    hidden = torch.arange(20, dtype=dtype).view(5, 4)
    weights = torch.full((5, 3), 1 / 3, dtype=torch.float32)
    ids = torch.tensor(
        [[0, 1, 2], [-1, -1, -1], [2, 0, 1], [-1, -1, -1], [1, 2, 0]],
        dtype=id_dtype,
    )
    mask = torch.tensor([False, True, False, True, False])
    hidden[1] = float("nan")
    hidden[3] = float("inf")
    weights[3] = float("nan")
    originals = [x.clone() for x in [hidden, weights, ids]]
    h, w, i = _module.prepare_padded_moe_inputs(hidden, weights, ids, mask)
    for original, prepared in zip(originals, [h, w, i]):
        torch.testing.assert_close(prepared[~mask], original[~mask], rtol=0, atol=0)
        assert torch.count_nonzero(prepared[mask]) == 0
    for actual, original in zip([hidden, weights, ids], originals):
        torch.testing.assert_close(actual, original, rtol=0, atol=0, equal_nan=True)
    output = originals[0].clone()
    result = _module.zero_padded_moe_rows(output, mask)
    assert result is output
    assert torch.count_nonzero(result[mask]) == 0
    torch.testing.assert_close(result[~mask], originals[0][~mask], rtol=0, atol=0)


def test_no_mask_retains_aliases_and_data():
    args = [torch.ones(2, 4), torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.int32)]
    prepared = _module.prepare_padded_moe_inputs(*args, None)
    assert all(a is b for a, b in zip(args, prepared))
    assert _module.zero_padded_moe_rows(args[0], None) is args[0]


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(2),
        torch.zeros(2, 1, dtype=torch.bool),
        torch.zeros(3, dtype=torch.bool),
    ],
)
def test_bad_mask_is_rejected_before_launch(mask):
    with pytest.raises(ValueError, match="padding mask"):
        _module.prepare_padded_moe_inputs(
            torch.ones(2, 4),
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.int32),
            mask,
        )


def test_alignment_binding_is_idempotent():
    from types import SimpleNamespace
    from unittest.mock import patch

    def original():
        return None

    def replacement():
        return None

    moe = SimpleNamespace(moe_align_block_size=original)
    align = SimpleNamespace(moe_align_block_size_no_tle=replacement)
    with patch("importlib.import_module", side_effect=[moe, align]):
        assert _module.install_no_tle_moe_alignment()
    assert moe.moe_align_block_size is replacement
    with patch("importlib.import_module", side_effect=[moe, align]):
        assert not _module.install_no_tle_moe_alignment()


def test_missing_aligner_preserves_existing_dependency():
    from types import SimpleNamespace
    from unittest.mock import patch

    def original():
        return None

    moe = SimpleNamespace(moe_align_block_size=original)
    with patch("importlib.import_module", side_effect=[moe, SimpleNamespace()]):
        assert not _module.install_no_tle_moe_alignment()
    assert moe.moe_align_block_size is original
