from unittest.mock import patch
import torch


def get_masked_input_and_mask_eager(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # torch.compile will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index
    )
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


def apply_patch_get_masked_input_and_mask():
    patch("vllm.model_executor.layers.vocab_parallel_embedding.get_masked_input_and_mask",get_masked_input_and_mask_eager).start()
