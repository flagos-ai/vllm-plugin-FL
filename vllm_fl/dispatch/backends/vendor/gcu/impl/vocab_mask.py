"""GCU vocabulary masking without unsupported bool-to-int64 arithmetic."""

import inspect

import torch
from vllm.triton_utils import tl, triton

_PARAMETERS = (
    "input_", "org_vocab_start_index", "org_vocab_end_index",
    "num_org_vocab_padding", "added_vocab_start_index", "added_vocab_end_index",
)


@triton.jit
def _mask_kernel(INPUT, OUTPUT, MASK, COUNT: tl.constexpr,
                 START: tl.constexpr, END: tl.constexpr, PADDING: tl.constexpr,
                 ADDED_START: tl.constexpr, ADDED_END: tl.constexpr,
                 BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(INPUT + offsets, offsets < COUNT, other=0).to(tl.int64)
    original = (values >= START) & (values < END)
    added = (values >= ADDED_START) & (values < ADDED_END)
    added_offset = ADDED_START - (END - START) - PADDING
    offset = tl.where(original, START, 0) + tl.where(added, added_offset, 0)
    valid = original | added
    tl.store(OUTPUT + offsets, tl.where(valid, values - offset, 0), offsets < COUNT)
    tl.store(MASK + offsets, ~valid, offsets < COUNT)


def get_masked_input_and_mask(
    input_, org_vocab_start_index, org_vocab_end_index, num_org_vocab_padding,
    added_vocab_start_index, added_vocab_end_index,
):
    if input_.dtype not in (torch.int32, torch.int64) or not input_.is_contiguous():
        raise ValueError("GCU vocabulary mask requires contiguous int32 or int64 IDs")
    output = torch.empty_like(input_, dtype=torch.int64)
    mask = torch.empty_like(input_, dtype=torch.bool)
    count = input_.numel()
    if count:
        _mask_kernel[(triton.cdiv(count, 256),)](
            input_, output, mask, count, org_vocab_start_index, org_vocab_end_index,
            num_org_vocab_padding, added_vocab_start_index, added_vocab_end_index,
            BLOCK=256, enable_i64=True,
        )
    return output, mask


def _install_vocab_mask(embedding) -> None:
    current = embedding.get_masked_input_and_mask
    original = getattr(current, "_torchdynamo_orig_callable", current)
    if tuple(inspect.signature(original).parameters) != _PARAMETERS:
        raise RuntimeError("Unknown vLLM vocabulary mask ABI")
    embedding.get_masked_input_and_mask = get_masked_input_and_mask


def apply_vocab_mask_gcu_patch() -> None:
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "enflame":
        return
    from vllm.model_executor.layers import vocab_parallel_embedding

    _install_vocab_mask(vocab_parallel_embedding)
