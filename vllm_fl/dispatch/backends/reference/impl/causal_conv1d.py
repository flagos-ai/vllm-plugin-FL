# SPDX-License-Identifier: Apache-2.0
"""PyTorch depthwise causal-convolution reference."""

from __future__ import annotations

import torch


def _apply_activation(value, activation):
    if activation is True or activation in ("silu", "swish"):
        return torch.nn.functional.silu(value)
    if activation in (False, None):
        return value
    raise ValueError(f"Unsupported causal-conv activation: {activation}")


def _causal_conv_sequence(sequence, state, weight, bias, activation):
    width = weight.shape[-1]
    history = state[..., -(width - 1) :].clone()
    outputs = []
    for token in range(sequence.shape[-1]):
        window = torch.cat((history, sequence[:, token : token + 1]), dim=-1)
        value = (window.float() * weight.float()).sum(dim=-1)
        if bias is not None:
            value = value + bias.float()
        outputs.append(_apply_activation(value, activation))
        history = window[:, 1:]
    output = (
        torch.stack(outputs, dim=-1).to(sequence.dtype)
        if outputs
        else sequence.new_empty(sequence.shape)
    )
    next_state = state.clone()
    next_state[..., -(width - 1) :] = history.to(next_state.dtype)
    return output, next_state


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: bool | str | None = "silu",
    pad_slot_id: int = -1,
    null_block_id: int = -1,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int = 0,
    metadata=None,
    validate_data: bool = False,
) -> torch.Tensor:
    """Regular-prefill subset of vLLM's causal-convolution contract."""
    del block_size_to_align, metadata, validate_data
    if any(
        value is not None
        for value in (
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            num_computed_tokens,
        )
    ):
        raise NotImplementedError(
            "Reference causal_conv1d_fn does not support APC metadata"
        )
    if x.ndim != 2:
        raise ValueError("Reference causal_conv1d_fn expects [dim, total_tokens]")
    boundaries = query_start_loc.detach().to("cpu", torch.int64).tolist()
    output = torch.empty_like(x)
    for request, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        state_id = request if cache_indices is None else int(cache_indices[request].item())
        if state_id in (pad_slot_id, null_block_id) or state_id < 0:
            output[:, start:end].zero_()
            continue
        use_state = has_initial_state is None or bool(has_initial_state[request].item())
        state = conv_states[state_id] if use_state else torch.zeros_like(conv_states[state_id])
        request_output, next_state = _causal_conv_sequence(
            x[:, start:end].to(conv_states.dtype), state, weight, bias, activation
        )
        output[:, start:end] = request_output.to(output.dtype)
        conv_states[state_id].copy_(next_state)
    return output


__all__ = ["causal_conv1d_fn"]
