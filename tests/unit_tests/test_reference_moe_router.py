"""Shared reference routing must preserve the vLLM output-buffer contract."""

import pytest
import torch

from vllm_fl.dispatch.backends.reference.reference import ReferenceBackend


@pytest.mark.parametrize(
    "shape,k", [((1, 3), 1), ((5, 7), 3), ((3, 4), 4), ((0, 4), 2)]
)
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_router(shape, k, normalize, bias, dtype):
    logits = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
    logits = torch.sin(logits * 1.7).to(dtype)
    original = logits.clone()
    correction = torch.arange(shape[1], dtype=torch.float32) * 0.3 if bias else None
    weights = torch.full((shape[0], k), -99.0)
    indices = torch.full((shape[0], k), -99, dtype=torch.int32)
    source = torch.full_like(indices, -99)
    result = ReferenceBackend().topk_softmax(
        weights,
        indices,
        source,
        logits,
        normalize,
        correction,
    )
    assert result[0] is weights and result[1] is indices
    probabilities = logits.float().softmax(-1)
    selection = probabilities + correction if bias else probabilities
    # Sorting independently checks selection without relying on topk's tie order.
    expected_ids = selection.argsort(dim=-1, descending=True)[:, :k]
    assert torch.equal(indices.long(), expected_ids)
    expected_weights = probabilities.gather(1, expected_ids)
    if normalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(logits, original)
    expected_source = torch.arange(shape[0])[:, None] + torch.arange(k) * shape[0]
    assert torch.equal(source.long(), expected_source)


def test_reference_registered_callable():
    from vllm_fl.dispatch.backends.reference.register_ops import register_builtins

    class Registry:
        def register_many(self, implementations):
            self.implementations = implementations

    registry = Registry()
    register_builtins(registry)
    implementation = next(
        item for item in registry.implementations if item.op_name == "topk_softmax"
    )
    logits = torch.tensor([[0.0, 2.0, 1.0]])
    weights = torch.empty(1, 1)
    indices = torch.empty(1, 1, dtype=torch.int32)
    source = torch.empty_like(indices)
    result = implementation.fn(weights, indices, source, logits, True)
    assert result[0] is weights and result[1] is indices
    assert indices.item() == 1
    assert weights.item() == 1.0
    assert source.item() == 0
