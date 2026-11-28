"""GCU vocabulary masking without unsupported bool-to-int64 arithmetic."""

import dataclasses
import functools
import inspect

import torch

_PARAMETERS = (
    "input_",
    "org_vocab_start_index",
    "org_vocab_end_index",
    "num_org_vocab_padding",
    "added_vocab_start_index",
    "added_vocab_end_index",
)


@functools.lru_cache(maxsize=1)
def _get_kernel():
    from triton.backends.enflame.compiler import GCUOptions

    if "enable_i64" not in {field.name for field in dataclasses.fields(GCUOptions)}:
        raise RuntimeError(
            "GCU vocabulary masking requires compiler enable_i64 support"
        )
    from .vocab_mask_kernel import _mask_kernel

    return _mask_kernel


def get_masked_input_and_mask(
    input_,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    if input_.dtype not in (torch.int32, torch.int64) or not input_.is_contiguous():
        raise ValueError("GCU vocabulary mask requires contiguous int32 or int64 IDs")
    output = torch.empty_like(input_, dtype=torch.int64)
    mask = torch.empty_like(input_, dtype=torch.bool)
    count = input_.numel()
    if count:
        _get_kernel()[((count + 255) // 256,)](
            input_,
            output,
            mask,
            count,
            org_vocab_start_index,
            org_vocab_end_index,
            num_org_vocab_padding,
            added_vocab_start_index,
            added_vocab_end_index,
            BLOCK=256,
            enable_i64=True,
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
    # Validate compiler options and create the lazy JIT function during backend
    # setup, before model loading. Importing this module on another backend still
    # does not import the vendor compiler or Triton kernel.
    _get_kernel()
    from vllm.model_executor.layers import vocab_parallel_embedding

    _install_vocab_mask(vocab_parallel_embedding)
