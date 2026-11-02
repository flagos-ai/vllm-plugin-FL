import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("padding", [[False, True, False], [False] * 3, [True] * 3])
def test_softmax_padding_stays_on_fl(monkeypatch, dtype, padding):
    from vllm_fl.ops.fused_moe import router

    weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
    ids = torch.tensor([[2, 0], [1, 2], [0, 1]], dtype=dtype)
    padding_mask = torch.tensor(padding)

    def kernel(output_weights, output_ids, source_rows, logits, renormalize):
        output_weights.copy_(weights)
        output_ids.copy_(ids)
        source_rows.copy_(torch.arange(6, dtype=torch.int32).reshape(2, 3).T)
        assert renormalize
        return output_weights, output_ids

    monkeypatch.setattr(router, "_topk_softmax", kernel)
    monkeypatch.setattr(router, "_get_padding_mask", lambda count: padding_mask)
    monkeypatch.setattr(router, "upstream_fused_topk", lambda **kwargs: pytest.fail("CUDA fallback"))
    actual_weights, actual_ids, source_rows = router.fused_topk(
        torch.zeros(3, 4), torch.zeros(3, 3), 2, True, dtype)
    torch.testing.assert_close(actual_weights, weights)
    torch.testing.assert_close(actual_ids, torch.where(padding_mask[:, None], -1, ids))
    assert actual_ids.dtype == dtype
    torch.testing.assert_close(source_rows, torch.arange(6, dtype=torch.int32).reshape(2, 3).T)


def test_sigmoid_keeps_upstream_path(monkeypatch):
    from vllm_fl.ops.fused_moe import router

    expected = object()
    monkeypatch.setattr(router, "upstream_fused_topk", lambda **kwargs: expected)
    assert router.fused_topk(torch.zeros(1, 4), torch.zeros(1, 3), 1, False,
                             scoring_func="sigmoid") is expected
