# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# vLLM 0.24.0 (empty wheel) compatibility shims for MetaX MACA.
#
# Four issues hit on vLLM 0.24.0 that did not exist in 0.20.2. Each is fixed
# by monkey-patching here so the vLLM wheel stays pristine (same architecture
# as the 0.20.2 line: all vendor fixes live in the plugin).
#
# 1. _load_ptr constexpr elem_dtype — a dtype passed into a nested @triton.jit
#    function arrives as a tl.constexpr wrapper, and tl.pointer_type() rejects
#    constexpr element types ("element_ty is a constexpr."). Unwrap with .value.
# 2. _penalties_kernel chained boolean — metax triton 3.0.0 codegen rejects
#    "chained boolean operators (A or B or C) are not supported". Operands are
#    scalar loads, so parenthesization is semantically identical.
# 3. SamplingStates.get_top_k_top_p — metax UVA-backed views are CPU-typed
#    tensors over device-accessible pinned memory; indexing them with a device
#    tensor fails ("indices should be either on cpu or on the same device as
#    the indexed tensor"). Gather on CPU, then move the result back to the
#    accelerator so downstream ops (pytorch top-k fallback on metax) see
#    device tensors.
# 4. PoolingRunner.pool — same UVA issue for prompt_len (compared against
#    input_batch.seq_lens, a device tensor).
#
# Each patch is version-gated: a missing module/attr in an older vLLM silently
# skips. TODO: remove each shim once upstream / MetaX fixes the root cause.

import inspect

import numpy as np
import torch
from torch.nn import functional as F

from vllm.triton_utils import tl, triton


def _patch_load_ptr() -> None:
    """Patch _load_ptr: unwrap constexpr elem_dtype (metax triton 3.0.0)."""
    try:
        import vllm.v1.worker.gpu.buffer_utils as buffer_utils
    except (ImportError, ModuleNotFoundError):
        return

    @triton.jit
    def _load_ptr(ptr_to_ptr, elem_dtype):
        ptr = tl.load(ptr_to_ptr)
        # metax triton: a dtype passed to a nested @jit fn arrives as a
        # constexpr wrapper, and tl.pointer_type() rejects constexpr element
        # types. Unwrap.
        # (isinstance(elem_dtype, tl.constexpr) can't guard this - triton
        # unwraps constexpr args before calling builtins, so that check is
        # always False.)
        elem_dtype = elem_dtype.value
        ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
        return tl.multiple_of(ptr, 16)

    buffer_utils._load_ptr = _load_ptr
    try:
        # block_table.py binds _load_ptr via `from buffer_utils import ...` at
        # import time; reassign its module global too so the multi-group fused
        # writers (gather_block_tables / compute_slot_mappings) use the shim.
        import vllm.v1.worker.gpu.block_table as block_table

        block_table._load_ptr = _load_ptr
    except (ImportError, AttributeError):
        # The block_table module is absent or renamed in some vllm 0.24.x
        # builds; the buffer_utils shim above already covers the common path,
        # so silently skip patching it.
        pass


def _patch_penalties_kernel() -> None:
    """Patch _penalties_kernel: split chained boolean for metax codegen."""
    try:
        import vllm.v1.worker.gpu.sample.penalties as penalties_mod
    except (ImportError, ModuleNotFoundError):
        return

    @triton.jit
    def _penalties_kernel(
        logits_ptr,
        logits_stride,
        expanded_idx_mapping_ptr,
        token_ids_ptr,
        expanded_local_pos_ptr,
        repetition_penalty_ptr,
        frequency_penalty_ptr,
        presence_penalty_ptr,
        prompt_bin_mask_ptr,
        prompt_bin_mask_stride,
        output_bin_counts_ptr,
        output_bin_counts_stride,
        vocab_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
        rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
        freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
        pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)

        use_rep_penalty = rep_penalty != 1.0
        use_freq_penalty = freq_penalty != 0.0
        use_pres_penalty = pres_penalty != 0.0
        use_penalty = (use_rep_penalty or use_freq_penalty) or use_pres_penalty
        if not use_penalty:
            # Early return to avoid loading logits.
            return

        block_idx = tl.program_id(1)
        block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
        logits = logits.to(tl.float32)

        base_output_counts = tl.load(
            output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
            mask=mask,
            other=0,
        )

        # Accumulate draft token counts from previous positions directly into
        # output_bin_counts (preserves its native tensor layout, avoiding an
        # expensive shared-memory layout conversion after the loop).
        pos = tl.load(expanded_local_pos_ptr + token_idx)
        start_idx = token_idx - pos
        output_bin_counts = base_output_counts
        for prev_pos in tl.range(pos):
            prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
            token_match = block == prev_token
            output_bin_counts = output_bin_counts + token_match.to(tl.int32)
        output_bin_mask = output_bin_counts > 0

        # Apply repetition penalties.
        if use_rep_penalty:
            packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
            packed_mask = tl.load(
                prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
                mask=packed_block < tl.cdiv(vocab_size, 32),
                other=0,
            )
            prompt_bin_mask = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
            prompt_bin_mask = prompt_bin_mask.to(tl.int1)
            prompt_bin_mask = prompt_bin_mask.reshape(BLOCK_SIZE)

            # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
            scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
            # If logits are positive, divide by penalty, otherwise multiply by penalty.
            logits *= tl.where(logits > 0, 1.0 / scale, scale)

        # Apply frequency penalties.
        logits -= freq_penalty * output_bin_counts
        # Apply presence penalties.
        logits -= pres_penalty * output_bin_mask
        # Store back to logits.
        tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)

    penalties_mod._penalties_kernel = _penalties_kernel


def _patch_get_top_k_top_p() -> None:
    """Patch SamplingStates.get_top_k_top_p: UVA CPU-index + device roundtrip."""
    try:
        from vllm.v1.worker.gpu.sample.states import SamplingStates
    except (ImportError, ModuleNotFoundError):
        return
    # Version gate: only the 0.24.0 signature is shimmed.
    try:
        params = list(inspect.signature(SamplingStates.get_top_k_top_p).parameters)
    except (TypeError, ValueError):
        return
    if params != ["self", "expanded_idx_mapping", "idx_mapping_np"]:
        return

    def get_top_k_top_p(
        self, expanded_idx_mapping: torch.Tensor, idx_mapping_np: np.ndarray
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        do_top_k = np.any(self.top_k.np[idx_mapping_np] != self.vocab_size)
        do_top_p = np.any(self.top_p.np[idx_mapping_np] != 1.0)
        # metax triton: UVA-backed views are CPU-typed tensors over
        # device-accessible pinned memory; indexing them with a device tensor
        # fails ("indices should be either on cpu or on the same device as the
        # indexed tensor"). Gather on CPU, then move the result back to the
        # accelerator so downstream ops (pytorch top-k fallback on metax) see
        # device tensors. On CUDA the view is device-typed; device index kept.
        if not self.top_k.gpu.is_cuda:
            device = expanded_idx_mapping.device
            top_k = (
                self.top_k.gpu[expanded_idx_mapping.cpu()].to(device)
                if do_top_k else None
            )
            top_p = (
                self.top_p.gpu[expanded_idx_mapping.cpu()].to(device)
                if do_top_p else None
            )
        else:
            top_k = self.top_k.gpu[expanded_idx_mapping] if do_top_k else None
            top_p = self.top_p.gpu[expanded_idx_mapping] if do_top_p else None
        return top_k, top_p

    SamplingStates.get_top_k_top_p = get_top_k_top_p


def _patch_pool() -> None:
    """Patch PoolingRunner.pool: prompt_len UVA CPU-index + device roundtrip."""
    try:
        from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner
    except (ImportError, ModuleNotFoundError):
        return
    # Version gate: only the 0.24.0 signature is shimmed.
    try:
        params = list(inspect.signature(PoolingRunner.pool).parameters)
    except (TypeError, ValueError):
        return
    if params != ["self", "hidden_states", "input_batch", "req_states"]:
        return

    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch,
        req_states,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO(woosuk): Support different types of pooling tasks.
        last_hidden_states = hidden_states[input_batch.logits_indices]
        # TODO(woosuk): Make normalization optional.
        last_hidden_states = F.normalize(last_hidden_states, p=2, dim=-1)

        # metax triton: UVA-backed views are CPU-typed; index with CPU indices
        # and move back to the accelerator (seq_lens is a device tensor).
        if not req_states.prompt_len.gpu.is_cuda:
            idx_mapping = input_batch.idx_mapping.cpu()
            prompt_len = req_states.prompt_len.gpu[idx_mapping].to(
                input_batch.seq_lens.device
            )
        else:
            prompt_len = req_states.prompt_len.gpu[input_batch.idx_mapping]
        is_valid = input_batch.seq_lens == prompt_len
        return last_hidden_states, is_valid

    PoolingRunner.pool = pool


_patch_load_ptr()
_patch_penalties_kernel()
_patch_get_top_k_top_p()
_patch_pool()
