"""Reference routing remains available when the FlagGems implementation is disabled."""

import pytest
import torch

from vllm_fl.dispatch.backends.reference.impl.grouped_topk import grouped_topk_torch


@pytest.mark.parametrize("topk", [3, 6, 7])
@pytest.mark.parametrize("scoring_func", [0, 1])
@pytest.mark.parametrize("renormalize", [False, True])
def test_non_power_of_two_topk(topk, scoring_func, renormalize):
    scores = torch.linspace(-2, 2, 128).reshape(1, -1)
    bias = torch.zeros(128)
    weights, ids = grouped_topk_torch(scores, 1, 1, topk, renormalize, 2.0, bias, scoring_func)
    expected_ids = torch.arange(127, 127 - topk, -1, dtype=torch.int32).reshape(1, -1)
    processed = scores.sigmoid() if scoring_func else scores
    expected = processed[:, 128 - topk:].flip(-1)
    if renormalize:
        expected = expected / expected.sum(-1, keepdim=True)
    torch.testing.assert_close(ids, expected_ids)
    torch.testing.assert_close(weights, expected * 2)
    assert weights.dtype == torch.float32


def test_bias_selects_groups_but_does_not_change_weights():
    scores = torch.tensor([[1., 2., 3., 4., 10., 20., 30., 40.]])
    bias = torch.tensor([100., 100., 100., 100., 0., 0., 0., 0.])
    weights, ids = grouped_topk_torch(scores, 2, 1, 3, True, 2., bias)
    torch.testing.assert_close(ids, torch.tensor([[3, 2, 1]], dtype=torch.int32))
    torch.testing.assert_close(weights, torch.tensor([[8 / 9, 6 / 9, 4 / 9]]))


def test_equal_scores_choose_lower_expert_indices():
    weights, ids = grouped_topk_torch(torch.ones(1, 8), 2, 1, 3, False, 1., torch.zeros(8))
    torch.testing.assert_close(ids, torch.tensor([[0, 1, 2]], dtype=torch.int32))
    torch.testing.assert_close(weights, torch.ones(1, 3))


def test_invalid_grouping_is_an_explicit_error():
    with pytest.raises(ValueError, match="divisible"):
        grouped_topk_torch(torch.ones(1, 7), 2, 1, 3, True, 1., torch.zeros(7))
