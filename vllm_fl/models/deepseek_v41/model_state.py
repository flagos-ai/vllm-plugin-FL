# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 BAAI. All rights reserved.
"""V2 request/token history metadata for DeepSeek V4.1.

Uses the same device token-history source as upstream e77daef89e18's
DeepseekV41ModelState. This kernel copies raw token ids only (plugin:A3).
Compressed-id mapping, n-gram hashing and table values belong to FGV.
"""

from typing import Any

import torch

from vllm.distributed import get_ep_group
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from .compressor_metadata import prepare_compressor_metadata


@triton.jit
def _history_kernel(
    IDS,
    PADDING,
    START,
    MAPPING,
    COMPUTED,
    ALL_IDS,
    OUT,
    N,
    NREQS,
    NSTATES: tl.constexpr,
    CAPACITY: tl.constexpr,
    ALL_STRIDE: tl.constexpr,
    IMAGE_ID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    end = tl.load(START + NREQS)
    valid = (token < N) & (token < end)
    padded = tl.load(PADDING + token, token < N, other=True)
    valid &= ~padded
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), NREQS, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        active = lo < hi
        mid = (lo + hi) // 2
        stop = tl.load(START + mid + 1, active, other=0)
        right = token >= stop
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    request = tl.minimum(lo, NREQS - 1).to(tl.int64)
    begin = tl.load(START + request, valid, other=0)
    state = tl.load(MAPPING + request, valid, other=-1).to(tl.int64)
    valid &= (state >= 0) & (state < NSTATES)
    computed = tl.load(COMPUTED + state, valid, other=0).to(tl.int64)
    relative = token - begin
    for shift in tl.static_range(4):
        in_chunk = valid & (relative >= shift)
        source = token - shift
        chunk_pad = tl.load(PADDING + source, in_chunk, other=True)
        current = tl.load(IDS + source, in_chunk & ~chunk_pad, other=-1)
        position = computed + relative - shift
        from_history = valid & ~in_chunk & (position >= 0) & (position < CAPACITY)
        previous = tl.load(
            ALL_IDS + state * ALL_STRIDE + position, from_history, other=-1
        )
        value = tl.where(in_chunk, current, previous)
        value = tl.where((value >= IMAGE_ID) & (value < IMAGE_ID + 5), -1, value)
        tl.store(OUT + token * 4 + shift, value, token < N)


def prepare_engram_history(
    input_ids,
    is_padding,
    query_start_loc,
    idx_mapping,
    num_computed_tokens,
    all_token_ids,
    num_reqs,
    output,
    image_token_id=129264,
):
    """Fill caller-owned INT32 [padded_tokens,4], current token then predecessors.

    Native V2 query boundaries are monotone, start at zero and end at the
    actual token count. The caller owns this verified request metadata.
    Padding/unknown request slots and unavailable history produce -1.
    Current chunk tokens take precedence over the device's committed history,
    including speculative input tokens not yet written to all_token_ids.
    No CPU token table or scheduler placeholder is read.
    """
    n = input_ids.numel()
    if (
        input_ids.ndim != 1
        or is_padding.shape != (n,)
        or is_padding.dtype != torch.bool
    ):
        raise ValueError("V2 input ids and padding must be aligned vectors")
    if (
        output.shape != (n, 4)
        or output.dtype != torch.int32
        or not output.is_contiguous()
    ):
        raise ValueError("Engram output must be contiguous INT32 [tokens,4]")
    if not isinstance(num_reqs, int) or isinstance(num_reqs, bool) or num_reqs < 0:
        raise ValueError("num_reqs must be a nonnegative integer")
    if query_start_loc.ndim != 1 or query_start_loc.numel() < num_reqs + 1:
        raise ValueError("V2 query boundaries are too short")
    if idx_mapping.ndim != 1 or idx_mapping.numel() < num_reqs:
        raise ValueError("V2 request mapping is too short")
    if all_token_ids.ndim != 2 or num_computed_tokens.shape != (
        all_token_ids.shape[0],
    ):
        raise ValueError("V2 committed history and computed counts must align")
    if all_token_ids.shape[0] <= 0 or all_token_ids.shape[1] <= 0:
        raise ValueError("V2 history capacity must be positive")
    tensors = (
        input_ids,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
    )
    for tensor in tensors:
        if tensor.dtype not in (torch.int32, torch.int64) or not tensor.is_contiguous():
            raise ValueError("V2 token metadata must be contiguous INT32/INT64")
    if output.device.type != "cuda" or any(
        t.device != output.device for t in (*tensors, is_padding)
    ):
        raise ValueError("V2 history metadata must share one CUDA-compatible device")
    with torch.cuda.device(output.device):
        if n and num_reqs:
            _history_kernel[(triton.cdiv(n, 128),)](
                input_ids,
                is_padding,
                query_start_loc,
                idx_mapping,
                num_computed_tokens,
                all_token_ids,
                output,
                n,
                num_reqs,
                all_token_ids.shape[0],
                all_token_ids.shape[1],
                all_token_ids.stride(0),
                image_token_id,
                128,
            )
        elif n:
            output.fill_(-1)
    return output


class DeepseekV41ModelState(DefaultModelState):
    """Prepare stable V2 raw history buffers before model graph execution."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        # MetaX module loading can synchronize the device. Keep idle DP ranks
        # out of EP kernels while a peer prepares uncached vision/metadata
        # kernels. The native CPU-group barrier stays outside graph capture.
        self.ep_prepare_fence = (
            get_ep_group()
            if getattr(current_platform, "vendor_name", None) == "metax"
            and vllm_config.parallel_config.data_parallel_size > 1
            and self.supports_mm_inputs
            else None
        )
        depth = model.token_lookback_depth
        if depth not in (0, 3):
            raise NotImplementedError("V4.1 currently supports 4-gram Engram only")
        self.engram_history_ids = (
            torch.full((self.max_num_tokens, 4), -1, device=device, dtype=torch.int32)
            if depth
            else None
        )

        capacity = getattr(model, "compressor_state_capacity", 0)
        if (
            not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity not in (0, 8)
        ):
            raise NotImplementedError("V4.1 compressor supports ring capacity 8 only")
        self.compressor_state_capacity = capacity
        self._compressor_model = model
        self.compressor_state_slots = None
        self.compressor_token_to_req = None
        self.compressor_query_start_loc = None
        if capacity:
            if not callable(getattr(model, "reset_compressor_state", None)):
                raise TypeError("compressor model must provide reset_compressor_state")
            speculation = getattr(vllm_config, "speculative_config", None)
            if speculation is not None and speculation.num_speculative_tokens > 5:
                raise NotImplementedError(
                    "V4.1 ring retains history for at most five speculative tokens"
                )
            self.compressor_state_slots = torch.full(
                (self.max_num_tokens,), -1, device=device, dtype=torch.int64
            )
            self.compressor_token_to_req = torch.full(
                (self.max_num_tokens,), -1, device=device, dtype=torch.int32
            )
            self.compressor_query_start_loc = torch.zeros(
                (self.max_num_reqs + 1,), device=device, dtype=torch.int32
            )

        self.token_is_padding = (
            torch.empty(self.max_num_tokens, device=device, dtype=torch.bool)
            if depth or capacity
            else None
        )

    def add_request(self, req_index, new_req_data):
        if self.compressor_state_capacity:
            computed = new_req_data.num_computed_tokens
            if (
                not isinstance(computed, int)
                or isinstance(computed, bool)
                or computed < 0
                or computed % 2
            ):
                raise NotImplementedError(
                    "new V4.1 compressor request must start at an even position"
                )
            if (
                not isinstance(req_index, int)
                or isinstance(req_index, bool)
                or not 0 <= req_index < self.max_num_reqs
            ):
                raise ValueError("compressor request index exceeds allocated capacity")
        super().add_request(req_index, new_req_data)
        if self.compressor_state_capacity:
            self._compressor_model.reset_compressor_state(req_index)

    def prepare_inputs(self, input_batch, req_states):
        inputs = super().prepare_inputs(input_batch, req_states)
        padding = None
        if self.token_is_padding is not None:
            n = input_batch.num_tokens_after_padding
            actual = input_batch.num_tokens
            if not 0 <= actual <= n <= self.max_num_tokens:
                raise ValueError("V4.1 batch exceeds allocated token capacity")
            # Native is_padding is maintained only for VLLM_MOE_SKIP_PADDING.
            # Dummy runs can leave it True, so derive our mask from batch sizes.
            padding = self.token_is_padding[:n]
            padding[:actual].fill_(False)
            padding[actual:].fill_(True)
        if self.engram_history_ids is not None:
            n = input_batch.num_tokens_after_padding
            if n > self.engram_history_ids.shape[0]:
                raise ValueError("Engram batch exceeds the allocated token capacity")
            history = self.engram_history_ids[:n]
            prepare_engram_history(
                input_batch.input_ids,
                padding,
                input_batch.query_start_loc,
                input_batch.idx_mapping,
                req_states.num_computed_tokens.gpu,
                req_states.all_token_ids.gpu,
                input_batch.num_reqs,
                history,
            )
            inputs["engram_history_ids"] = history
        if self.compressor_state_capacity:
            n = input_batch.num_tokens_after_padding
            if n > self.max_num_tokens:
                raise ValueError("compressor batch exceeds allocated token capacity")
            slots = self.compressor_state_slots[:n]
            requests = self.compressor_token_to_req[:n]
            prepare_compressor_metadata(
                input_batch.positions,
                padding,
                input_batch.query_start_loc,
                input_batch.idx_mapping,
                input_batch.num_reqs,
                self.compressor_state_capacity,
                slots,
                requests,
                self.compressor_query_start_loc,
            )
            inputs.update(
                compressor_state_slots=slots,
                compressor_token_to_req=requests,
                compressor_query_start_loc=self.compressor_query_start_loc,
            )
        if self.ep_prepare_fence is not None:
            self.ep_prepare_fence.barrier()
        return inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.engram_history_ids is not None:
            if num_tokens > self.engram_history_ids.shape[0]:
                raise ValueError("Engram dummy batch exceeds token capacity")
            history = self.engram_history_ids[:num_tokens]
            history.fill_(-1)
            inputs["engram_history_ids"] = history
        if self.compressor_state_capacity:
            if num_tokens > self.max_num_tokens:
                raise ValueError(
                    "compressor dummy batch exceeds allocated token capacity"
                )
            slots = self.compressor_state_slots[:num_tokens]
            requests = self.compressor_token_to_req[:num_tokens]
            slots.fill_(-1)
            requests.fill_(-1)
            self.compressor_query_start_loc.zero_()
            inputs.update(
                compressor_state_slots=slots,
                compressor_token_to_req=requests,
                compressor_query_start_loc=self.compressor_query_start_loc,
            )
        return inputs
