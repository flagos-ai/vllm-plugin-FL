import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.gcu.impl import vocab_mask


def test_install_preserves_abi_and_is_idempotent():
    module = SimpleNamespace(
        get_masked_input_and_mask=SimpleNamespace(
            _torchdynamo_orig_callable=vocab_mask.get_masked_input_and_mask
        )
    )
    vocab_mask._install_vocab_mask(module)
    assert module.get_masked_input_and_mask is vocab_mask.get_masked_input_and_mask
    vocab_mask._install_vocab_mask(module)
    assert (
        tuple(inspect.signature(module.get_masked_input_and_mask).parameters)
        == vocab_mask._PARAMETERS
    )


def test_unknown_abi_is_not_replaced():
    def incompatible(input_):
        return input_

    module = SimpleNamespace(get_masked_input_and_mask=incompatible)
    with pytest.raises(RuntimeError, match="Unknown vLLM"):
        vocab_mask._install_vocab_mask(module)
    assert module.get_masked_input_and_mask is incompatible


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_empty_does_not_launch(monkeypatch, dtype):
    monkeypatch.setattr(vocab_mask, "_mask_kernel", None)
    result, mask = vocab_mask.get_masked_input_and_mask(
        torch.empty((0, 2), dtype=dtype), 0, 10, 2, 20, 24
    )
    assert result.shape == mask.shape == (0, 2)
    assert result.dtype == torch.int64
    assert mask.dtype == torch.bool


@pytest.mark.parametrize(
    "input_", [torch.zeros(2), torch.zeros(2, 3, dtype=torch.int64).T]
)
def test_invalid_input_rejected(input_):
    with pytest.raises(ValueError, match="contiguous int32 or int64"):
        vocab_mask.get_masked_input_and_mask(input_, 0, 10, 0, 10, 10)


def test_other_vendor_is_untouched(monkeypatch):
    import vllm.platforms

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="other")
    )
    monkeypatch.setattr(
        vocab_mask,
        "_install_vocab_mask",
        lambda module: pytest.fail("unexpected patch"),
    )
    vocab_mask.apply_vocab_mask_gcu_patch()
