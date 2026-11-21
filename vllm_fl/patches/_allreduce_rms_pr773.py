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
#
# The Lamport all-reduce protocol is adapted from the Apache-2.0
# FlashInfer/TensorRT-LLM CUDA implementation.

"""vLLM-compatible fused all-reduce, residual add, RMSNorm, and quantization.

The Triton kernels support a symmetric peer-buffer backend and an NVLink
multicast (MNNVL) backend.  FlashInfer is used only to allocate and exchange
MNNVL virtual-memory handles; FlagGems executes the collective and post-ops.
"""

# Vendored from flagos-ai/FlagGems-vllm PR #773, merge commit
# 61f39dbf17c1e9262fdc72cae4ffe32f1d193bf3.
# Source SHA256: e6e31753e3bf41e4718bde44f68a85e03dfe7b0fda7e5b09e39407a7d752af5c
# Local adaptation: the public entry point requires an explicit, eagerly
# allocated workspace; implicit TP/WORLD resolution and lazy allocation were
# removed. The forward-only guard honors no_grad/inference_mode, allowing
# immutable nn.Parameter weights in normal inference. The two tiled one-shot
# kernels additionally acknowledge phase consumption from every CTA before
# program 0 resets the phase counter. Without this acknowledgment, an arrived
# but delayed CTA can miss GRID_SIZE and spin forever after the reset. Launch
# policies and numerical contracts are unchanged. Controlled CTA-delay tests
# reproduce the old race on peer/MNNVL, and same-host old/fixed dense-model
# controls distinguish the fix from a driver change.
# This single-file provider avoids upgrading the pinned FlagGems-vllm package.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

try:
    import torch.distributed._symmetric_memory as symm_mem
except ImportError:
    symm_mem = None

try:
    from flashinfer.comm.comm_backend import TorchDistBackend
    from flashinfer.comm.mnnvl import (
        McastGPUBuffer,
        all_ranks_support_mnnvl,
        create_tensor_from_cuda_memory,
        is_multicast_supported,
    )

    _HAS_MNNVL_RUNTIME = True
except (ImportError, OSError):
    TorchDistBackend = None
    McastGPUBuffer = None
    all_ranks_support_mnnvl = None
    create_tensor_from_cuda_memory = None
    is_multicast_supported = None
    _HAS_MNNVL_RUNTIME = False

try:
    from triton.language.extra.cuda.gdc import gdc_launch_dependents, gdc_wait

    _HAS_PDL = True
except ImportError:
    _HAS_PDL = False


_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_BACKENDS = ("auto", "mnnvl", "peer")
_SLOT_COUNT = 3
_DEFAULT_HIDDEN_SIZE = 6144
_MNNVL_ONESHOT_THRESHOLD_BYTES = 1024 * 1024


class AllReduceFusionPattern:
    """Pattern values consumed by vLLM's FlashInfer all-reduce wrapper."""

    kARResidualRMSNorm = 1
    kARResidualRMSNormFP8Quant = 2
    kARResidualRMSNormFP4Quant = 3


_SUPPORTED_PATTERNS = (
    AllReduceFusionPattern.kARResidualRMSNorm,
    AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
    AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
)

# Autotune exemption: these launch shapes and resident-grid sizes are part of a
# stateful multi-rank protocol. Trying configs independently on each rank can
# deadlock, so the validated parameters are dispatched explicitly.
_ONE_SHOT_MAX_SIZE_MIB = {
    90: {2: 32, 4: 2, 8: 0.5},
    100: {2: 32, 4: 4, 8: 1},
    103: {2: 32, 4: 4, 8: 2},
    107: {2: 32, 4: 4, 8: 2},
}
_MIB = 1024 * 1024
_WORKSPACE_CACHE = {}


if _HAS_PDL:

    @triton.jit
    def _pdl_wait():
        gdc_wait()

    @triton.jit
    def _pdl_launch_dependents():
        gdc_launch_dependents()

else:

    @triton.jit
    def _pdl_wait():
        return

    @triton.jit
    def _pdl_launch_dependents():
        return


def _check_world_size(world_size: int) -> None:
    if world_size not in _SUPPORTED_WORLD_SIZES:
        raise NotImplementedError(
            "the validated Triton all-reduce backends support world_size in "
            f"{_SUPPORTED_WORLD_SIZES}, got {world_size}"
        )


@triton.jit
def _finish_row(
    reduced,
    token,
    columns,
    column_mask,
    residual_in,
    rms_gamma,
    residual_out,
    norm_scratch,
    rms_eps,
    weight_bias,
    H: tl.constexpr,
):
    offsets = token * H + columns
    residual = (
        reduced.to(tl.float32)
        + tl.load(residual_in + offsets, mask=column_mask, other=0.0).to(tl.float32)
    ).to(residual_in.dtype.element_ty)
    tl.store(residual_out + offsets, residual, mask=column_mask)

    residual_f = residual.to(tl.float32)
    square_sum = tl.sum(
        tl.where(column_mask, residual_f * residual_f, 0.0),
        axis=0,
    )
    reciprocal_rms = tl.rsqrt(square_sum / H + rms_eps)
    gamma = tl.load(rms_gamma + columns, mask=column_mask, other=0.0).to(tl.float32)
    normalized = residual_f * reciprocal_rms * (gamma + weight_bias)
    tl.store(
        norm_scratch + offsets,
        normalized.to(norm_scratch.dtype.element_ty),
        mask=column_mask,
    )


@libentry()
@triton.jit
def _peer_initialize_sentinel_workspace_kernel(
    communication_buffer,
    state,
    communication_numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < communication_numel
    sentinel = tl.full((BLOCK_SIZE,), 0x8000, tl.uint16).to(
        communication_buffer.dtype.element_ty, bitcast=True
    )
    tl.store(communication_buffer + offsets, sentinel, mask=mask)
    if tl.program_id(0) == 0:
        # [arrival, slot, previous payload size, strategy, phase arrival]
        tl.store(state + 0, 0)
        tl.store(state + 1, 0)
        tl.store(state + 2, 0)
        tl.store(state + 3, 0)
        tl.store(state + 4, 0)


@libentry()
@triton.jit
def _mnnvl_initialize_protocol_state_kernel(state):
    tl.store(state + 0, 0)
    tl.store(state + 1, 0)
    tl.store(state + 2, 0)
    tl.store(state + 3, 0)
    tl.store(state + 4, 0)


@libentry()
@triton.jit
def _peer_rowwise_oneshot_allreduce_rms_norm_kernel(
    allreduce_in,
    residual_in,
    rms_gamma,
    residual_out,
    norm_scratch,
    local_communication_buffer,
    peer_communication_buffers,
    state,
    rms_eps,
    weight_bias,
    M: tl.constexpr,
    H: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCATTER_RANK_CAPACITY: tl.constexpr,
    STAGE_CAPACITY: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FP32_ACC: tl.constexpr,
    USE_PDL: tl.constexpr,
    TRIGGER_COMPLETION_AT_END: tl.constexpr,
    CHAIN_QUANT: tl.constexpr,
):
    """Execute a resident-grid Lamport all-reduce and fused post-operations."""
    tl.static_assert(WORLD_SIZE >= 2)
    tl.static_assert(WORLD_SIZE <= 8)
    tl.static_assert((WORLD_SIZE & (WORLD_SIZE - 1)) == 0)
    tl.static_assert(0 <= RANK)
    tl.static_assert(RANK < WORLD_SIZE)

    pid = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    column_mask = columns < H

    if USE_PDL:
        _pdl_wait()

    # Every resident program snapshots the same generation before publishing
    # its arrival. Program 0 advances the slot only after all programs arrive.
    slot = tl.load(state + 1)
    previous_numel = tl.load(state + 2)
    previous_strategy = tl.load(state + 3)
    # Every warp must snapshot metadata before the CTA publishes arrival.
    tl.debug_barrier()
    tl.atomic_add(state, 1, sem="release", scope="gpu")

    slot_stride = WORLD_SIZE * CAPACITY
    data_slot_base = slot * slot_stride
    clear_slot_base = ((slot + 2) % 3) * slot_stride

    if USE_PDL and not CHAIN_QUANT and not TRIGGER_COMPLETION_AT_END:
        _pdl_launch_dependents()

    # Push this rank's rows directly into every peer's symmetric allocation.
    token = pid
    while token < M:
        row_offsets = token * H + columns
        value = tl.load(
            allreduce_in + row_offsets,
            mask=column_mask,
            other=0.0,
        )
        input_bits = value.to(tl.uint16, bitcast=True)
        # Negative zero is reserved as the empty-slot sentinel. Numerically it
        # is positive zero, so canonicalizing the payload preserves the value.
        value = tl.where(input_bits == 0x8000, 0.0, value)
        destination = data_slot_base + RANK * CAPACITY + row_offsets
        for peer_rank in tl.static_range(0, WORLD_SIZE):
            tl.store(
                peer_communication_buffers[peer_rank] + destination,
                value,
                mask=column_mask,
            )
        token += GRID_SIZE

    # Clear according to the preceding launch's actual communication layout.
    _clear_previous_protocol_slot(
        local_communication_buffer,
        clear_slot_base,
        previous_numel,
        previous_strategy,
        columns,
        pid,
        H,
        WORLD_SIZE,
        CAPACITY,
        SCATTER_RANK_CAPACITY,
        STAGE_CAPACITY,
        GRID_SIZE,
        BLOCK_SIZE,
    )

    # Poll the local symmetric allocation. Peer writes target this allocation
    # directly; volatile loads prevent a cached sentinel from stalling progress.
    token = pid
    while token < M:
        row_offsets = token * H + columns
        local_row = data_slot_base + row_offsets
        pending = 1
        while pending != 0:
            missing = tl.zeros((BLOCK_SIZE,), tl.int1)
            if FP32_ACC:
                rank_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
            else:
                rank_sum = tl.zeros((BLOCK_SIZE,), allreduce_in.dtype.element_ty)
            for peer_rank in tl.static_range(0, WORLD_SIZE):
                payload = tl.load(
                    local_communication_buffer + peer_rank * CAPACITY + local_row,
                    mask=column_mask,
                    other=0.0,
                    volatile=True,
                )
                bits = payload.to(tl.uint16, bitcast=True)
                is_missing = bits == 0x8000
                missing |= is_missing
                payload = tl.where(is_missing, 0.0, payload)
                if FP32_ACC:
                    rank_sum += payload.to(tl.float32)
                else:
                    rank_sum = (rank_sum.to(tl.float32) + payload.to(tl.float32)).to(
                        allreduce_in.dtype.element_ty
                    )
            pending = tl.sum((missing & column_mask).to(tl.int32))

            if pending == 0:
                reduced = rank_sum.to(allreduce_in.dtype.element_ty)
                _finish_row(
                    reduced,
                    token,
                    columns,
                    column_mask,
                    residual_in,
                    rms_gamma,
                    residual_out,
                    norm_scratch,
                    rms_eps,
                    weight_bias,
                    H,
                )
        token += GRID_SIZE

    if pid == 0:
        arrived = tl.load(state, volatile=True)
        while arrived != GRID_SIZE:
            arrived = tl.load(state, volatile=True)
        # A scalar poll runs in multiple warps: no warp may reset the counter
        # before its siblings have observed this generation.
        tl.debug_barrier()
        tl.store(state + 1, (slot + 1) % 3)
        tl.store(state + 2, M * H)
        tl.store(state + 3, 0)
        tl.atomic_xchg(state, 0, sem="release", scope="gpu")

    if USE_PDL and (CHAIN_QUANT or TRIGGER_COMPLETION_AT_END):
        _pdl_launch_dependents()


@libentry()
@triton.jit
def _peer_tiled_oneshot_allreduce_rms_norm_kernel(
    allreduce_in,
    residual_in,
    rms_gamma,
    residual_out,
    norm_scratch,
    local_communication_buffer,
    peer_communication_buffers,
    partial_sums,
    state,
    rms_eps,
    weight_bias,
    M: tl.constexpr,
    H: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCATTER_RANK_CAPACITY: tl.constexpr,
    STAGE_CAPACITY: tl.constexpr,
    PARTIAL_SUM_SLOT_CAPACITY: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    TILES_PER_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FP32_ACC: tl.constexpr,
    USE_PDL: tl.constexpr,
    TRIGGER_COMPLETION_AT_END: tl.constexpr,
    CHAIN_QUANT: tl.constexpr,
):
    """One-shot peer all-reduce tiled across the hidden dimension."""
    tl.static_assert(WORLD_SIZE >= 2)
    tl.static_assert(WORLD_SIZE <= 8)
    tl.static_assert((WORLD_SIZE & (WORLD_SIZE - 1)) == 0)

    pid = tl.program_id(0)
    lanes = tl.arange(0, BLOCK_SIZE)

    if USE_PDL:
        _pdl_wait()

    slot = tl.load(state + 1)
    previous_numel = tl.load(state + 2)
    previous_strategy = tl.load(state + 3)
    # Every warp must snapshot metadata before the CTA publishes arrival.
    tl.debug_barrier()
    tl.atomic_add(state, 1, sem="release", scope="gpu")

    slot_capacity = WORLD_SIZE * CAPACITY
    data_slot_base = slot * slot_capacity
    clear_slot_base = ((slot + 2) % 3) * slot_capacity
    partial_sum_base = slot * PARTIAL_SUM_SLOT_CAPACITY

    if USE_PDL and not CHAIN_QUANT and not TRIGGER_COMPLETION_AT_END:
        _pdl_launch_dependents()

    # Publish tiles to every rank in parallel. Unlike MNNVL this backend has no
    # multicast mapping, so each program explicitly stores to every peer.
    work_index = pid
    while work_index < M * TILES_PER_ROW:
        token = work_index // TILES_PER_ROW
        tile = work_index % TILES_PER_ROW
        columns = tile * BLOCK_SIZE + lanes
        column_mask = columns < H
        row_offsets = token * H + columns
        value = tl.load(allreduce_in + row_offsets, mask=column_mask, other=0.0)
        input_bits = value.to(tl.uint16, bitcast=True)
        value = tl.where(input_bits == 0x8000, 0.0, value)
        destination = data_slot_base + RANK * CAPACITY + row_offsets
        for peer_rank in tl.static_range(0, WORLD_SIZE):
            tl.store(
                peer_communication_buffers[peer_rank] + destination,
                value,
                mask=column_mask,
            )
        work_index += GRID_SIZE

    # Clear according to the preceding launch's actual communication layout.
    _clear_previous_protocol_slot(
        local_communication_buffer,
        clear_slot_base,
        previous_numel,
        previous_strategy,
        lanes,
        pid,
        H,
        WORLD_SIZE,
        CAPACITY,
        SCATTER_RANK_CAPACITY,
        STAGE_CAPACITY,
        GRID_SIZE,
        BLOCK_SIZE,
    )

    # Reduce each tile and publish its partial square sum for row-wide RMSNorm.
    work_index = pid
    while work_index < M * TILES_PER_ROW:
        token = work_index // TILES_PER_ROW
        tile = work_index % TILES_PER_ROW
        columns = tile * BLOCK_SIZE + lanes
        column_mask = columns < H
        row_offsets = token * H + columns
        local_row = data_slot_base + row_offsets
        pending = 1
        while pending != 0:
            missing = tl.zeros((BLOCK_SIZE,), tl.int1)
            if FP32_ACC:
                rank_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
            else:
                rank_sum = tl.zeros((BLOCK_SIZE,), allreduce_in.dtype.element_ty)
            for source_rank in tl.static_range(0, WORLD_SIZE):
                payload = tl.load(
                    local_communication_buffer + source_rank * CAPACITY + local_row,
                    mask=column_mask,
                    other=0.0,
                    volatile=True,
                )
                bits = payload.to(tl.uint16, bitcast=True)
                is_missing = bits == 0x8000
                missing |= is_missing
                payload = tl.where(is_missing, 0.0, payload)
                if FP32_ACC:
                    rank_sum += payload.to(tl.float32)
                else:
                    rank_sum = (rank_sum.to(tl.float32) + payload.to(tl.float32)).to(
                        allreduce_in.dtype.element_ty
                    )
            pending = tl.sum((missing & column_mask).to(tl.int32))
            if pending == 0:
                reduced = rank_sum.to(allreduce_in.dtype.element_ty)
                residual = (
                    reduced.to(tl.float32)
                    + tl.load(
                        residual_in + row_offsets,
                        mask=column_mask,
                        other=0.0,
                    ).to(tl.float32)
                ).to(residual_in.dtype.element_ty)
                tl.store(residual_out + row_offsets, residual, mask=column_mask)
                residual_f = residual.to(tl.float32)
                partial_sum = tl.sum(
                    tl.where(column_mask, residual_f * residual_f, 0.0), axis=0
                )
                tl.store(
                    partial_sums + partial_sum_base + token * TILES_PER_ROW + tile,
                    partial_sum,
                )
        work_index += GRID_SIZE

    # The launch is limited to one resident wave, making this global phase
    # barrier safe without relying on scheduling progress from another CTA.
    tl.atomic_add(state + 4, 1, sem="release", scope="gpu")
    phase_arrived = tl.load(state + 4, volatile=True)
    while phase_arrived != GRID_SIZE:
        phase_arrived = tl.load(state + 4, volatile=True)

    work_index = pid
    while work_index < M * TILES_PER_ROW:
        token = work_index // TILES_PER_ROW
        tile = work_index % TILES_PER_ROW
        columns = tile * BLOCK_SIZE + lanes
        column_mask = columns < H
        row_offsets = token * H + columns
        square_sum = 0.0
        for partial_tile in tl.static_range(0, TILES_PER_ROW):
            square_sum += tl.load(
                partial_sums + partial_sum_base + token * TILES_PER_ROW + partial_tile
            )
        reciprocal_rms = tl.rsqrt(square_sum / H + rms_eps)
        residual_f = tl.load(
            residual_out + row_offsets, mask=column_mask, other=0.0
        ).to(tl.float32)
        gamma = tl.load(rms_gamma + columns, mask=column_mask, other=0.0).to(tl.float32)
        normalized = residual_f * reciprocal_rms * (gamma + weight_bias)
        tl.store(
            norm_scratch + row_offsets,
            normalized.to(norm_scratch.dtype.element_ty),
            mask=column_mask,
        )
        work_index += GRID_SIZE

    # All warps must finish consuming the phase counter before their CTA
    # publishes its completion acknowledgment.
    tl.debug_barrier()
    tl.atomic_add(state, 1, sem="release", scope="gpu")
    if pid == 0:
        arrived = tl.load(state, volatile=True)
        while arrived != 2 * GRID_SIZE:
            arrived = tl.load(state, volatile=True)
        tl.debug_barrier()
        tl.store(state + 1, (slot + 1) % 3)
        tl.store(state + 2, M * H)
        tl.store(state + 3, 0)
        tl.store(state + 4, 0)
        tl.atomic_xchg(state, 0, sem="release", scope="gpu")

    if USE_PDL and (CHAIN_QUANT or TRIGGER_COMPLETION_AT_END):
        _pdl_launch_dependents()


@triton.jit
def _clear_previous_protocol_slot(
    local_communication_buffer,
    clear_slot_base,
    previous_numel,
    previous_strategy,
    columns,
    pid,
    H: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCATTER_RANK_CAPACITY: tl.constexpr,
    STAGE_CAPACITY: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    sentinel = tl.full((BLOCK_SIZE,), 0x8000, tl.uint16).to(
        local_communication_buffer.dtype.element_ty, bitcast=True
    )
    clear_offset = pid * BLOCK_SIZE
    clear_stride = GRID_SIZE * BLOCK_SIZE

    if previous_strategy == 0:
        while clear_offset < previous_numel:
            clear_indices = clear_offset + columns
            clear_mask = clear_indices < previous_numel
            for source_rank in tl.static_range(0, WORLD_SIZE):
                tl.store(
                    local_communication_buffer
                    + clear_slot_base
                    + source_rank * CAPACITY
                    + clear_indices,
                    sentinel,
                    mask=clear_mask,
                )
            clear_offset += clear_stride
    else:
        previous_tokens = previous_numel // H
        previous_scatter_numel = ((previous_tokens + WORLD_SIZE - 1) // WORLD_SIZE) * H
        while clear_offset < previous_scatter_numel:
            clear_indices = clear_offset + columns
            clear_mask = clear_indices < previous_scatter_numel
            for source_rank in tl.static_range(0, WORLD_SIZE):
                tl.store(
                    local_communication_buffer
                    + clear_slot_base
                    + source_rank * SCATTER_RANK_CAPACITY
                    + clear_indices,
                    sentinel,
                    mask=clear_mask,
                )
            clear_offset += clear_stride

        clear_offset = pid * BLOCK_SIZE
        while clear_offset < previous_numel:
            clear_indices = clear_offset + columns
            clear_mask = clear_indices < previous_numel
            tl.store(
                local_communication_buffer
                + clear_slot_base
                + STAGE_CAPACITY
                + clear_indices,
                sentinel,
                mask=clear_mask,
            )
            clear_offset += clear_stride


@libentry()
@triton.jit
def _mnnvl_tiled_oneshot_allreduce_rms_norm_kernel(
    allreduce_in,
    residual_in,
    rms_gamma,
    residual_out,
    norm_scratch,
    local_communication_buffer,
    multicast_communication_buffer,
    partial_sums,
    state,
    rms_eps,
    weight_bias,
    M: tl.constexpr,
    H: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    SLOT_CAPACITY: tl.constexpr,
    SCATTER_RANK_CAPACITY: tl.constexpr,
    STAGE_CAPACITY: tl.constexpr,
    PARTIAL_SUM_SLOT_CAPACITY: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    TILES_PER_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMM_BF16: tl.constexpr,
    FP32_ACC: tl.constexpr,
    USE_PDL: tl.constexpr,
    TRIGGER_COMPLETION_AT_END: tl.constexpr,
    CHAIN_QUANT: tl.constexpr,
):
    """Tiled one-shot MNNVL all-reduce, residual add, and RMSNorm."""
    tl.static_assert(WORLD_SIZE >= 2)
    tl.static_assert(WORLD_SIZE <= 8)
    tl.static_assert((WORLD_SIZE & (WORLD_SIZE - 1)) == 0)

    pid = tl.program_id(0)
    lanes = tl.arange(0, BLOCK_SIZE)

    if USE_PDL:
        _pdl_wait()

    slot = tl.load(state + 1)
    previous_numel = tl.load(state + 2)
    previous_strategy = tl.load(state + 3)
    # Every warp must snapshot metadata before the CTA publishes arrival.
    tl.debug_barrier()
    tl.atomic_add(state, 1, sem="release", scope="gpu")

    data_slot_base = slot * SLOT_CAPACITY
    partial_sum_base = slot * PARTIAL_SUM_SLOT_CAPACITY
    clear_slot_base = ((slot + 2) % 3) * SLOT_CAPACITY

    if USE_PDL and not CHAIN_QUANT and not TRIGGER_COMPLETION_AT_END:
        _pdl_launch_dependents()

    # A store through the CUDA multicast mapping is replicated into the local
    # physical allocation of every rank in the multicast group. Every resident
    # program publishes all of its tiles before any program starts polling.
    work_index = pid
    while work_index < M * TILES_PER_ROW:
        token = work_index // TILES_PER_ROW
        tile = work_index % TILES_PER_ROW
        columns = tile * BLOCK_SIZE + lanes
        column_mask = columns < H
        row_offsets = token * H + columns
        value = tl.load(
            allreduce_in + row_offsets,
            mask=column_mask,
            other=0.0,
        )
        input_bits = value.to(tl.uint16, bitcast=True)
        value = tl.where(input_bits == 0x8000, 0.0, value)
        if COMM_BF16:
            value = value.to(tl.uint16, bitcast=True).to(
                multicast_communication_buffer.dtype.element_ty,
                bitcast=True,
            )
        destination = data_slot_base + RANK * CAPACITY + row_offsets
        tl.store(
            multicast_communication_buffer + destination,
            value,
            mask=column_mask,
        )
        work_index += GRID_SIZE

    _clear_previous_protocol_slot(
        local_communication_buffer,
        clear_slot_base,
        previous_numel,
        previous_strategy,
        lanes,
        pid,
        H,
        WORLD_SIZE,
        CAPACITY,
        SCATTER_RANK_CAPACITY,
        STAGE_CAPACITY,
        GRID_SIZE,
        BLOCK_SIZE,
    )

    # Each tile reduces its rank slices, writes the rounded residual, and
    # publishes one FP32 partial square sum for the row-wide RMS reduction.
    work_index = pid
    while work_index < M * TILES_PER_ROW:
        token = work_index // TILES_PER_ROW
        tile = work_index % TILES_PER_ROW
        columns = tile * BLOCK_SIZE + lanes
        column_mask = columns < H
        row_offsets = token * H + columns
        local_row = data_slot_base + row_offsets
        pending = 1
        while pending != 0:
            missing = tl.zeros((BLOCK_SIZE,), tl.int1)
            if FP32_ACC:
                rank_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
            else:
                rank_sum = tl.zeros(
                    (BLOCK_SIZE,), local_communication_buffer.dtype.element_ty
                )
            for source_rank in tl.static_range(0, WORLD_SIZE):
                payload = tl.load(
                    local_communication_buffer + source_rank * CAPACITY + local_row,
                    mask=column_mask,
                    other=0.0,
                    volatile=True,
                )
                bits = payload.to(tl.uint16, bitcast=True)
                is_missing = bits == 0x8000
                missing |= is_missing
                if COMM_BF16:
                    payload_value = bits.to(tl.bfloat16, bitcast=True)
                else:
                    payload_value = payload
                payload_value = tl.where(is_missing, 0.0, payload_value)
                if FP32_ACC:
                    rank_sum += payload_value.to(tl.float32)
                else:
                    rank_sum = (
                        rank_sum.to(tl.float32) + payload_value.to(tl.float32)
                    ).to(allreduce_in.dtype.element_ty)
            pending = tl.sum((missing & column_mask).to(tl.int32))

            if pending == 0:
                reduced = rank_sum.to(allreduce_in.dtype.element_ty)
                residual = (
                    reduced.to(tl.float32)
                    + tl.load(
                        residual_in + row_offsets,
                        mask=column_mask,
                        other=0.0,
                    ).to(tl.float32)
                ).to(residual_in.dtype.element_ty)
                tl.store(residual_out + row_offsets, residual, mask=column_mask)
                residual_f = residual.to(tl.float32)
                partial_sum = tl.sum(
                    tl.where(column_mask, residual_f * residual_f, 0.0),
                    axis=0,
                )
                tl.store(
                    partial_sums + partial_sum_base + token * TILES_PER_ROW + tile,
                    partial_sum,
                )
        work_index += GRID_SIZE

    # The grid is capped at one program per SM, so all programs can reach this
    # device-wide phase barrier without starving an unscheduled producer.
    tl.atomic_add(state + 4, 1, sem="release", scope="gpu")
    phase_arrived = tl.load(state + 4, volatile=True)
    while phase_arrived != GRID_SIZE:
        phase_arrived = tl.load(state + 4, volatile=True)

    # Every tile now sees all row partials and normalizes only its own slice.
    work_index = pid
    while work_index < M * TILES_PER_ROW:
        token = work_index // TILES_PER_ROW
        tile = work_index % TILES_PER_ROW
        columns = tile * BLOCK_SIZE + lanes
        column_mask = columns < H
        row_offsets = token * H + columns
        square_sum = 0.0
        for partial_tile in tl.static_range(0, TILES_PER_ROW):
            square_sum += tl.load(
                partial_sums + partial_sum_base + token * TILES_PER_ROW + partial_tile
            )
        reciprocal_rms = tl.rsqrt(square_sum / H + rms_eps)
        residual_f = tl.load(
            residual_out + row_offsets,
            mask=column_mask,
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(
            rms_gamma + columns,
            mask=column_mask,
            other=0.0,
        ).to(tl.float32)
        normalized = residual_f * reciprocal_rms * (gamma + weight_bias)
        tl.store(
            norm_scratch + row_offsets,
            normalized.to(norm_scratch.dtype.element_ty),
            mask=column_mask,
        )
        work_index += GRID_SIZE

    # All warps must finish consuming the phase counter before their CTA
    # publishes its completion acknowledgment.
    tl.debug_barrier()
    tl.atomic_add(state, 1, sem="release", scope="gpu")
    if pid == 0:
        arrived = tl.load(state, volatile=True)
        while arrived != 2 * GRID_SIZE:
            arrived = tl.load(state, volatile=True)
        tl.debug_barrier()
        tl.store(state + 1, (slot + 1) % 3)
        tl.store(state + 2, M * H)
        tl.store(state + 3, 0)
        tl.store(state + 4, 0)
        tl.atomic_xchg(state, 0, sem="release", scope="gpu")

    if USE_PDL and (CHAIN_QUANT or TRIGGER_COMPLETION_AT_END):
        _pdl_launch_dependents()


@triton.jit
def _tiled_twoshot_allreduce_rms_norm_body(
    allreduce_in,
    residual_in,
    rms_gamma,
    residual_out,
    norm_scratch,
    local_communication_buffer,
    peer_communication_buffers,
    multicast_communication_buffer,
    state,
    rms_eps,
    weight_bias,
    M: tl.constexpr,
    H: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    SLOT_CAPACITY: tl.constexpr,
    SCATTER_RANK_CAPACITY: tl.constexpr,
    STAGE_CAPACITY: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    TILES_PER_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NORM_BLOCK_SIZE: tl.constexpr,
    COMM_BF16: tl.constexpr,
    USE_MULTICAST: tl.constexpr,
    FP32_ACC: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Tiled two-shot all-reduce fused with residual add and RMSNorm."""
    tl.static_assert(WORLD_SIZE >= 2)
    tl.static_assert(WORLD_SIZE <= 8)
    tl.static_assert((WORLD_SIZE & (WORLD_SIZE - 1)) == 0)

    pid = tl.program_id(0)
    lanes = tl.arange(0, BLOCK_SIZE)

    if USE_PDL:
        _pdl_wait()

    slot = tl.load(state + 1)
    previous_numel = tl.load(state + 2)
    previous_strategy = tl.load(state + 3)
    # Every warp must snapshot metadata before the CTA publishes arrival.
    tl.debug_barrier()
    tl.atomic_add(state, 1, sem="release", scope="gpu")

    data_slot_base = slot * SLOT_CAPACITY
    scatter_base = data_slot_base
    broadcast_base = data_slot_base + STAGE_CAPACITY
    clear_slot_base = ((slot + 2) % 3) * SLOT_CAPACITY

    # Shot 1: tile each row and send it to the token's owner rank. Each source
    # rank has a disjoint slice in the owner's local allocation.
    work_index = pid
    while work_index < M * TILES_PER_ROW:
        token = work_index // TILES_PER_ROW
        tile = work_index % TILES_PER_ROW
        columns = tile * BLOCK_SIZE + lanes
        column_mask = columns < H
        row_offsets = token * H + columns
        value = tl.load(
            allreduce_in + row_offsets,
            mask=column_mask,
            other=0.0,
        )
        input_bits = value.to(tl.uint16, bitcast=True)
        value = tl.where(input_bits == 0x8000, 0.0, value)
        if COMM_BF16:
            value = value.to(tl.uint16, bitcast=True).to(
                local_communication_buffer.dtype.element_ty,
                bitcast=True,
            )
        destination_rank = token % WORLD_SIZE
        destination_token = token // WORLD_SIZE
        destination = (
            scatter_base
            + RANK * SCATTER_RANK_CAPACITY
            + destination_token * H
            + columns
        )
        for peer_rank in tl.static_range(0, WORLD_SIZE):
            tl.store(
                peer_communication_buffers[peer_rank] + destination,
                value,
                mask=column_mask & (destination_rank == peer_rank),
            )
        work_index += GRID_SIZE

    _clear_previous_protocol_slot(
        local_communication_buffer,
        clear_slot_base,
        previous_numel,
        previous_strategy,
        lanes,
        pid,
        H,
        WORLD_SIZE,
        CAPACITY,
        SCATTER_RANK_CAPACITY,
        STAGE_CAPACITY,
        GRID_SIZE,
        BLOCK_SIZE,
    )

    # The owner rank reduces its tiles and broadcasts the results.
    owned_token_count = (M + WORLD_SIZE - 1 - RANK) // WORLD_SIZE
    owned_work_index = pid
    while owned_work_index < owned_token_count * TILES_PER_ROW:
        destination_token = owned_work_index // TILES_PER_ROW
        tile = owned_work_index % TILES_PER_ROW
        owned_token = RANK + destination_token * WORLD_SIZE
        columns = tile * BLOCK_SIZE + lanes
        column_mask = columns < H
        scatter_row = destination_token * H + columns
        pending = 1
        while pending != 0:
            missing = tl.zeros((BLOCK_SIZE,), tl.int1)
            if FP32_ACC:
                rank_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
            else:
                rank_sum = tl.zeros((BLOCK_SIZE,), allreduce_in.dtype.element_ty)
            for source_rank in tl.static_range(0, WORLD_SIZE):
                payload = tl.load(
                    local_communication_buffer
                    + scatter_base
                    + source_rank * SCATTER_RANK_CAPACITY
                    + scatter_row,
                    mask=column_mask,
                    other=0.0,
                    volatile=True,
                )
                bits = payload.to(tl.uint16, bitcast=True)
                is_missing = bits == 0x8000
                missing |= is_missing
                if COMM_BF16:
                    payload_value = bits.to(tl.bfloat16, bitcast=True)
                else:
                    payload_value = payload
                payload_value = tl.where(is_missing, 0.0, payload_value)
                if FP32_ACC:
                    rank_sum += payload_value.to(tl.float32)
                else:
                    rank_sum = (
                        rank_sum.to(tl.float32) + payload_value.to(tl.float32)
                    ).to(allreduce_in.dtype.element_ty)
            pending = tl.sum((missing & column_mask).to(tl.int32))

            if pending == 0:
                reduced = rank_sum.to(allreduce_in.dtype.element_ty)
                reduced_bits = reduced.to(tl.uint16, bitcast=True)
                reduced = tl.where(reduced_bits == 0x8000, 0.0, reduced)
                if COMM_BF16:
                    communication_reduced = reduced.to(tl.uint16, bitcast=True).to(
                        multicast_communication_buffer.dtype.element_ty,
                        bitcast=True,
                    )
                else:
                    communication_reduced = reduced
                broadcast_offset = broadcast_base + owned_token * H + columns
                if USE_MULTICAST:
                    tl.store(
                        multicast_communication_buffer + broadcast_offset,
                        communication_reduced,
                        mask=column_mask,
                    )
                else:
                    for peer_rank in tl.static_range(0, WORLD_SIZE):
                        tl.store(
                            peer_communication_buffers[peer_rank] + broadcast_offset,
                            communication_reduced,
                            mask=column_mask,
                        )
        owned_work_index += GRID_SIZE

    # Consume complete broadcast rows in this resident grid. Programs that
    # finish their owned communication tiles early can overlap post-processing
    # with communication still running in other programs.
    norm_columns = tl.arange(0, NORM_BLOCK_SIZE)
    norm_mask = norm_columns < H
    token = pid
    while token < M:
        row_offsets = token * H + norm_columns
        pending = 1
        while pending != 0:
            reduced = tl.load(
                local_communication_buffer + broadcast_base + row_offsets,
                mask=norm_mask,
                other=0.0,
                volatile=True,
            )
            bits = reduced.to(tl.uint16, bitcast=True)
            missing = (bits == 0x8000) & norm_mask
            pending = tl.sum(missing.to(tl.int32))
            if pending == 0:
                if COMM_BF16:
                    reduced_value = bits.to(tl.bfloat16, bitcast=True)
                else:
                    reduced_value = reduced
                _finish_row(
                    reduced_value,
                    token,
                    norm_columns,
                    norm_mask,
                    residual_in,
                    rms_gamma,
                    residual_out,
                    norm_scratch,
                    rms_eps,
                    weight_bias,
                    H,
                )
        token += GRID_SIZE

    if pid == 0:
        arrived = tl.load(state, volatile=True)
        while arrived != GRID_SIZE:
            arrived = tl.load(state, volatile=True)
        # A scalar poll runs in multiple warps: no warp may reset the counter
        # before its siblings have observed this generation.
        tl.debug_barrier()
        tl.store(state + 1, (slot + 1) % 3)
        tl.store(state + 2, M * H)
        tl.store(state + 3, 1)
        tl.atomic_xchg(state, 0, sem="release", scope="gpu")

    if USE_PDL:
        _pdl_launch_dependents()


@libentry()
@triton.jit
def _peer_tiled_twoshot_allreduce_rms_norm_kernel(
    allreduce_in,
    residual_in,
    rms_gamma,
    residual_out,
    norm_scratch,
    local_communication_buffer,
    peer_communication_buffers,
    state,
    rms_eps,
    weight_bias,
    M: tl.constexpr,
    H: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    SLOT_CAPACITY: tl.constexpr,
    SCATTER_RANK_CAPACITY: tl.constexpr,
    STAGE_CAPACITY: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    TILES_PER_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NORM_BLOCK_SIZE: tl.constexpr,
    FP32_ACC: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    _tiled_twoshot_allreduce_rms_norm_body(
        allreduce_in,
        residual_in,
        rms_gamma,
        residual_out,
        norm_scratch,
        local_communication_buffer,
        peer_communication_buffers,
        local_communication_buffer,
        state,
        rms_eps,
        weight_bias,
        M,
        H,
        RANK,
        WORLD_SIZE,
        CAPACITY,
        SLOT_CAPACITY,
        SCATTER_RANK_CAPACITY,
        STAGE_CAPACITY,
        GRID_SIZE,
        TILES_PER_ROW,
        BLOCK_SIZE,
        NORM_BLOCK_SIZE,
        False,
        False,
        FP32_ACC,
        USE_PDL,
    )


@libentry()
@triton.jit
def _mnnvl_tiled_twoshot_allreduce_rms_norm_kernel(
    allreduce_in,
    residual_in,
    rms_gamma,
    residual_out,
    norm_scratch,
    local_communication_buffer,
    peer_communication_buffers,
    multicast_communication_buffer,
    state,
    rms_eps,
    weight_bias,
    M: tl.constexpr,
    H: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    SLOT_CAPACITY: tl.constexpr,
    SCATTER_RANK_CAPACITY: tl.constexpr,
    STAGE_CAPACITY: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    TILES_PER_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NORM_BLOCK_SIZE: tl.constexpr,
    COMM_BF16: tl.constexpr,
    FP32_ACC: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    _tiled_twoshot_allreduce_rms_norm_body(
        allreduce_in,
        residual_in,
        rms_gamma,
        residual_out,
        norm_scratch,
        local_communication_buffer,
        peer_communication_buffers,
        multicast_communication_buffer,
        state,
        rms_eps,
        weight_bias,
        M,
        H,
        RANK,
        WORLD_SIZE,
        CAPACITY,
        SLOT_CAPACITY,
        SCATTER_RANK_CAPACITY,
        STAGE_CAPACITY,
        GRID_SIZE,
        TILES_PER_ROW,
        BLOCK_SIZE,
        NORM_BLOCK_SIZE,
        COMM_BF16,
        True,
        FP32_ACC,
        USE_PDL,
    )


@libentry()
@triton.jit
def _static_fp8_quant_kernel(
    norm_input,
    quant_out,
    scale_factor,
    numel,
    BLOCK_SIZE: tl.constexpr,
    USE_PDL: tl.constexpr,
    TRIGGER_COMPLETION_AT_END: tl.constexpr,
):
    if USE_PDL:
        _pdl_wait()
        if not TRIGGER_COMPLETION_AT_END:
            _pdl_launch_dependents()

    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    scale = tl.load(scale_factor).to(tl.float32)
    values = tl.load(norm_input + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.clamp(values / scale, -448.0, 448.0)
    tl.store(quant_out + offsets, values.to(tl.float8e4nv), mask=mask)

    if USE_PDL and TRIGGER_COMPLETION_AT_END:
        _pdl_launch_dependents()


@triton.jit
def _fp32x2_to_e2m1x2(low, high):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 packed;
            cvt.rn.satfinite.e2m1x2.f32 packed, $2, $1;
            cvt.u32.u8 $0, packed;
        }
        """,
        constraints="=r,f,f",
        args=[low, high],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@libentry()
@triton.jit
def _static_nvfp4_quant_kernel(
    norm_input,
    quant_out,
    scale_out,
    scale_factor,
    H: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    USE_PDL: tl.constexpr,
    TRIGGER_COMPLETION_AT_END: tl.constexpr,
):
    if USE_PDL:
        _pdl_wait()
        if not TRIGGER_COMPLETION_AT_END:
            _pdl_launch_dependents()

    row = tl.program_id(0)
    pairs = tl.arange(0, BLOCK_PAIRS)
    low_columns = pairs * 2
    high_columns = low_columns + 1
    pair_mask = high_columns < H
    row_base = row * H
    low = tl.load(norm_input + row_base + low_columns, mask=pair_mask, other=0.0).to(
        tl.float32
    )
    high = tl.load(norm_input + row_base + high_columns, mask=pair_mask, other=0.0).to(
        tl.float32
    )

    # One NVFP4 scale covers 16 values, i.e. eight packed pairs.
    block_groups = BLOCK_PAIRS // 8
    pair_amax = tl.maximum(tl.abs(low), tl.abs(high))
    group_amax = tl.max(tl.reshape(pair_amax, (block_groups, 8)), axis=1)
    global_scale = tl.load(scale_factor).to(tl.float32)
    sf = (global_scale * group_amax * (1.0 / 6.0)).to(tl.float8e4nv)
    sf_f32 = sf.to(tl.float32)
    output_scale = tl.where(sf_f32 == 0.0, 0.0, global_scale / sf_f32)
    output_scale = tl.broadcast_to(output_scale[:, None], (block_groups, 8))

    low = tl.reshape(tl.reshape(low, (block_groups, 8)) * output_scale, (BLOCK_PAIRS,))
    high = tl.reshape(
        tl.reshape(high, (block_groups, 8)) * output_scale, (BLOCK_PAIRS,)
    )
    packed = _fp32x2_to_e2m1x2(low, high)
    tl.store(quant_out + row * (H // 2) + pairs, packed, mask=pair_mask)

    group = tl.arange(0, block_groups)
    valid_groups = group < H // 16
    inner_k = group % 4
    inner_m = (row % 128) // 32
    outer_m = row % 32
    k_tile = group // 4
    num_k_tiles = (H + 63) // 64
    m_tile = row // 128
    scale_offset = (
        m_tile * num_k_tiles * 512 + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k
    )
    tl.store(
        scale_out + scale_offset,
        sf.to(tl.uint8, bitcast=True),
        mask=valid_groups,
    )

    if USE_PDL and TRIGGER_COMPLETION_AT_END:
        _pdl_launch_dependents()


@dataclass
class FusedAllReduceRMSNormWorkspace:
    """Communication buffers and protocol state owned by one TP rank."""

    backend: str
    world_size: int
    rank: int
    max_token_num: int
    hidden_dim: int
    dtype: torch.dtype
    group: dist.ProcessGroup
    local_buffer: Optional[torch.Tensor]
    peer_buffers: tuple[torch.Tensor, ...]
    multicast_buffer: Optional[torch.Tensor]
    partial_sums: Optional[torch.Tensor]
    state: torch.Tensor
    norm_buffer: torch.Tensor
    resident_program_limit: int
    slot_capacity: int
    scatter_rank_capacity: int
    stage_capacity: int
    partial_sum_slot_capacity: int
    _handle: object
    _destroyed: bool = False

    @property
    def capacity(self) -> int:
        return self.max_token_num * self.hidden_dim

    def is_buffer_size_sufficient(
        self,
        world_size: int,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
    ) -> bool:
        return (
            not self._destroyed
            and world_size == self.world_size
            and 0 < num_tokens <= self.max_token_num
            and hidden_dim == self.hidden_dim
            and dtype == self.dtype
        )

    def destroy(self) -> None:
        self.peer_buffers = ()
        self.multicast_buffer = None
        self.local_buffer = None
        self.partial_sums = None
        self._handle = None
        self._destroyed = True


def create_fused_allreduce_rms_norm_workspace(
    *,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    max_token_num: int,
    hidden_dim: int = _DEFAULT_HIDDEN_SIZE,
    dtype: torch.dtype = torch.bfloat16,
    group: Optional[dist.ProcessGroup] = None,
    backend: str = "auto",
) -> FusedAllReduceRMSNormWorkspace:
    """Allocate a peer-buffer or MNNVL communication workspace.

    ``auto`` selects MNNVL when CUDA multicast is available on every rank and
    otherwise uses the peer-buffer backend. Explicit ``mnnvl`` never falls
    back silently.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    backend = backend.lower()
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")

    group = dist.group.WORLD if group is None else group
    group_world_size = dist.get_world_size(group)
    group_rank = dist.get_rank(group)
    world_size = group_world_size if world_size is None else world_size
    rank = group_rank if rank is None else rank
    if world_size != group_world_size or rank != group_rank:
        raise ValueError(
            "world_size/rank must match the supplied process group: "
            f"got ({world_size},{rank}), group is ({group_world_size},{group_rank})"
        )
    _check_world_size(world_size)
    if max_token_num <= 0:
        raise ValueError(f"max_token_num must be positive, got {max_token_num}")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    if dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError(
            f"only torch.float16 and torch.bfloat16 are supported, got {dtype}"
        )
    if not torch_device_fn.is_available():
        raise NotImplementedError("a CUDA device is required")

    device_index = torch_device_fn.current_device()
    capability = torch_device_fn.get_device_capability(device_index)
    if capability[0] < 9:
        raise NotImplementedError(
            f"NVIDIA compute capability >= 9.0 is required, got {capability}"
        )

    device = torch.device("cuda", device_index)
    capacity = max_token_num * hidden_dim
    comm_backend = None
    mnnvl_supported = False
    if backend in ("auto", "mnnvl") and _HAS_MNNVL_RUNTIME:
        comm_backend = TorchDistBackend(group=group)
        local_supported = bool(is_multicast_supported(device_index))
        mnnvl_supported = bool(
            all_ranks_support_mnnvl(
                local_supported,
                world_size,
                comm_backend=comm_backend,
                group=group,
            )
        )
    if backend == "mnnvl" and not mnnvl_supported:
        raise NotImplementedError(
            "the MNNVL backend requires FlashInfer's MNNVL runtime and CUDA "
            "multicast support on every rank"
        )
    actual_backend = "mnnvl" if mnnvl_supported and backend != "peer" else "peer"

    state = torch.empty((5,), dtype=torch.int32, device=device)
    norm_buffer = torch.empty((max_token_num, hidden_dim), dtype=dtype, device=device)
    partial_sum_tiles = triton.cdiv(hidden_dim, 1024)
    partial_sum_slot_capacity = max_token_num * partial_sum_tiles
    partial_sums = torch.empty(
        (_SLOT_COUNT * partial_sum_slot_capacity,),
        dtype=torch.float32,
        device=device,
    )
    scatter_rank_capacity = (
        (max_token_num + world_size - 1) // world_size
    ) * hidden_dim
    stage_capacity = world_size * scatter_rank_capacity

    if actual_backend == "mnnvl":
        slot_capacity = max(world_size * capacity, 2 * stage_capacity)
        communication_numel = _SLOT_COUNT * slot_capacity
        element_size = torch.empty((), dtype=dtype).element_size()
        handle = McastGPUBuffer(
            buf_size=communication_numel * element_size,
            group_size=world_size,
            group_rank=rank,
            device=device,
            comm_backend_for_handle_transfer=comm_backend,
        )
        if handle.buf_size < communication_numel * element_size:
            raise RuntimeError(
                "MNNVL allocation is smaller than the requested workspace"
            )
        handle.lamport_initialize(rank, dtype)
        local_buffer = create_tensor_from_cuda_memory(
            handle.get_unicast_ptr(rank),
            (communication_numel,),
            dtype,
            device_index,
        )
        peer_buffers = tuple(
            create_tensor_from_cuda_memory(
                handle.get_unicast_ptr(peer_rank),
                (communication_numel,),
                dtype,
                device_index,
            )
            for peer_rank in range(world_size)
        )
        multicast_buffer = create_tensor_from_cuda_memory(
            handle.get_multicast_ptr(),
            (communication_numel,),
            dtype,
            device_index,
        )
        with torch_device_fn.device(device):
            _mnnvl_initialize_protocol_state_kernel[(1,)](state, num_warps=1)
        torch_device_fn.synchronize(device)
        dist.barrier(group=group)
    else:
        if symm_mem is None:
            raise NotImplementedError("PyTorch symmetric memory is unavailable")
        buffer_shape = (_SLOT_COUNT, world_size, capacity)
        local_buffer = symm_mem.empty(buffer_shape, dtype=dtype, device=device)
        multicast_buffer = None
        slot_capacity = world_size * capacity
        init_block_size = 256
        communication_numel = local_buffer.numel()
        init_grid = (triton.cdiv(communication_numel, init_block_size),)
        with torch_device_fn.device(device):
            _peer_initialize_sentinel_workspace_kernel[init_grid](
                local_buffer,
                state,
                communication_numel,
                BLOCK_SIZE=init_block_size,
                num_warps=4,
            )
        torch_device_fn.synchronize(device)
        handle = symm_mem.rendezvous(local_buffer, group)
        peer_buffers = tuple(
            handle.get_buffer(peer_rank, buffer_shape, dtype)
            for peer_rank in range(world_size)
        )
    resident_program_limit = int(
        torch_device_fn.get_device_properties(device_index).multi_processor_count
    )
    dist.barrier(group=group)
    workspace = FusedAllReduceRMSNormWorkspace(
        backend=actual_backend,
        world_size=world_size,
        rank=rank,
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        group=group,
        local_buffer=local_buffer,
        peer_buffers=peer_buffers,
        multicast_buffer=multicast_buffer,
        partial_sums=partial_sums,
        state=state,
        norm_buffer=norm_buffer,
        resident_program_limit=resident_program_limit,
        slot_capacity=slot_capacity,
        scatter_rank_capacity=scatter_rank_capacity,
        stage_capacity=stage_capacity,
        partial_sum_slot_capacity=partial_sum_slot_capacity,
        _handle=handle,
    )
    _WORKSPACE_CACHE[
        _workspace_key(group, world_size, max_token_num, hidden_dim, dtype)
    ] = workspace
    return workspace


def _check_tensor(
    tensor: Optional[torch.Tensor],
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if tensor is None:
        raise ValueError(f"{name} is required")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name}.shape must be {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name}.device must be {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _check_flat_output(
    tensor: Optional[torch.Tensor],
    name: str,
    numel: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if tensor is None:
        raise ValueError(f"{name} is required")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.numel() != numel:
        raise ValueError(f"{name}.numel must be {numel}, got {tensor.numel()}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name}.device must be {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _memory_interval(tensor: torch.Tensor) -> tuple[int, int]:
    start = tensor.data_ptr()
    return start, start + tensor.numel() * tensor.element_size()


def _tensors_overlap(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    lhs_start, lhs_end = _memory_interval(lhs)
    rhs_start, rhs_end = _memory_interval(rhs)
    return lhs_start < rhs_end and rhs_start < lhs_end


def _workspace_key(
    group: dist.ProcessGroup,
    world_size: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
) -> tuple:
    return (
        id(group),
        torch_device_fn.current_device(),
        world_size,
        max_token_num,
        hidden_dim,
        dtype,
    )


def destroy_fused_allreduce_rms_norm_workspaces() -> None:
    """Release every cached symmetric-memory workspace in this process."""
    for workspace in _WORKSPACE_CACHE.values():
        workspace.destroy()
    _WORKSPACE_CACHE.clear()


def _select_use_oneshot(
    device_capability: tuple[int, int],
    world_size: int,
    tensor_size: int,
) -> bool:
    capability = device_capability[0] * 10 + device_capability[1]
    capability_thresholds = _ONE_SHOT_MAX_SIZE_MIB.get(capability)
    if capability_thresholds is None:
        raise NotImplementedError(
            "one-shot/two-shot selection is not defined for compute "
            f"capability {device_capability}"
        )
    max_size_mib = capability_thresholds.get(world_size)
    if max_size_mib is None:
        raise NotImplementedError(
            "one-shot/two-shot selection is not defined for the Triton "
            f"peer-buffer backend at world_size={world_size}"
        )
    return tensor_size <= max_size_mib * _MIB


def _select_mnnvl_use_oneshot(world_size: int, tensor_size: int) -> bool:
    """Match FlashInfer's MNNVL strategy threshold."""
    return world_size * tensor_size <= _MNNVL_ONESHOT_THRESHOLD_BYTES


def _launch_fused_allreduce_rms_norm(
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    residual_out: torch.Tensor,
    norm_scratch: torch.Tensor,
    workspace: FusedAllReduceRMSNormWorkspace,
    *,
    rms_eps: float,
    weight_bias: float,
    launch_with_pdl: bool,
    trigger_completion_at_end: bool,
    fp32_acc: bool,
    use_oneshot: bool,
    chain_quant: bool,
) -> None:
    m, hidden_dim = allreduce_in.shape
    block_size = triton.next_power_of_2(hidden_dim)
    grid_size = min(m, workspace.resident_program_limit)
    launch_meta = {
        "M": m,
        "H": hidden_dim,
        "RANK": workspace.rank,
        "WORLD_SIZE": workspace.world_size,
        "CAPACITY": workspace.capacity,
        "SCATTER_RANK_CAPACITY": workspace.scatter_rank_capacity,
        "STAGE_CAPACITY": workspace.stage_capacity,
        "GRID_SIZE": grid_size,
        "BLOCK_SIZE": block_size,
        "FP32_ACC": fp32_acc,
        "USE_PDL": launch_with_pdl,
        "TRIGGER_COMPLETION_AT_END": trigger_completion_at_end,
        "CHAIN_QUANT": chain_quant,
        "num_warps": 32 if m >= 128 else 8,
        "num_stages": 1,
        "launch_pdl": launch_with_pdl,
    }
    if workspace.backend == "mnnvl":
        if use_oneshot:
            communication_block_size = 1024
            tiles_per_row = triton.cdiv(hidden_dim, communication_block_size)
            communication_grid_size = min(
                m * tiles_per_row,
                workspace.resident_program_limit,
            )
            _mnnvl_tiled_oneshot_allreduce_rms_norm_kernel[(communication_grid_size,)](
                allreduce_in,
                residual_in,
                rms_gamma,
                residual_out,
                norm_scratch,
                workspace.local_buffer,
                workspace.multicast_buffer,
                workspace.partial_sums,
                workspace.state,
                rms_eps,
                weight_bias,
                M=m,
                H=hidden_dim,
                RANK=workspace.rank,
                WORLD_SIZE=workspace.world_size,
                CAPACITY=workspace.capacity,
                SLOT_CAPACITY=workspace.slot_capacity,
                SCATTER_RANK_CAPACITY=workspace.scatter_rank_capacity,
                STAGE_CAPACITY=workspace.stage_capacity,
                PARTIAL_SUM_SLOT_CAPACITY=workspace.partial_sum_slot_capacity,
                GRID_SIZE=communication_grid_size,
                TILES_PER_ROW=tiles_per_row,
                BLOCK_SIZE=communication_block_size,
                COMM_BF16=workspace.dtype == torch.bfloat16,
                FP32_ACC=fp32_acc,
                USE_PDL=launch_with_pdl,
                TRIGGER_COMPLETION_AT_END=trigger_completion_at_end,
                CHAIN_QUANT=chain_quant,
                num_warps=4,
                num_stages=1,
                launch_pdl=launch_with_pdl,
            )
        else:
            communication_block_size = 2048 if workspace.world_size == 2 else 1024
            tiles_per_row = triton.cdiv(hidden_dim, communication_block_size)
            communication_grid_size = min(
                m * tiles_per_row,
                workspace.resident_program_limit,
            )
            _mnnvl_tiled_twoshot_allreduce_rms_norm_kernel[(communication_grid_size,)](
                allreduce_in,
                residual_in,
                rms_gamma,
                residual_out,
                norm_scratch,
                workspace.local_buffer,
                workspace.peer_buffers,
                workspace.multicast_buffer,
                workspace.state,
                rms_eps,
                weight_bias,
                M=m,
                H=hidden_dim,
                RANK=workspace.rank,
                WORLD_SIZE=workspace.world_size,
                CAPACITY=workspace.capacity,
                SLOT_CAPACITY=workspace.slot_capacity,
                SCATTER_RANK_CAPACITY=workspace.scatter_rank_capacity,
                STAGE_CAPACITY=workspace.stage_capacity,
                GRID_SIZE=communication_grid_size,
                TILES_PER_ROW=tiles_per_row,
                BLOCK_SIZE=communication_block_size,
                NORM_BLOCK_SIZE=block_size,
                COMM_BF16=workspace.dtype == torch.bfloat16,
                FP32_ACC=fp32_acc,
                USE_PDL=launch_with_pdl,
                num_warps=32,
                num_stages=1,
                launch_pdl=launch_with_pdl,
            )
        return

    peer_tile_size = 1024
    peer_tiles_per_row = triton.cdiv(hidden_dim, peer_tile_size)
    peer_tile_count = m * peer_tiles_per_row
    if use_oneshot and peer_tile_count <= workspace.resident_program_limit:
        _peer_tiled_oneshot_allreduce_rms_norm_kernel[(peer_tile_count,)](
            allreduce_in,
            residual_in,
            rms_gamma,
            residual_out,
            norm_scratch,
            workspace.local_buffer,
            workspace.peer_buffers,
            workspace.partial_sums,
            workspace.state,
            rms_eps,
            weight_bias,
            M=m,
            H=hidden_dim,
            RANK=workspace.rank,
            WORLD_SIZE=workspace.world_size,
            CAPACITY=workspace.capacity,
            SCATTER_RANK_CAPACITY=workspace.scatter_rank_capacity,
            STAGE_CAPACITY=workspace.stage_capacity,
            PARTIAL_SUM_SLOT_CAPACITY=workspace.partial_sum_slot_capacity,
            GRID_SIZE=peer_tile_count,
            TILES_PER_ROW=peer_tiles_per_row,
            BLOCK_SIZE=peer_tile_size,
            FP32_ACC=fp32_acc,
            USE_PDL=launch_with_pdl,
            TRIGGER_COMPLETION_AT_END=trigger_completion_at_end,
            CHAIN_QUANT=chain_quant,
            num_warps=4,
            num_stages=1,
            launch_pdl=launch_with_pdl,
        )
        return

    if not use_oneshot:
        peer_twoshot_tile_size = 1024
        peer_twoshot_tiles_per_row = triton.cdiv(hidden_dim, peer_twoshot_tile_size)
        peer_twoshot_grid_size = min(
            m * peer_twoshot_tiles_per_row,
            workspace.resident_program_limit,
        )
        _peer_tiled_twoshot_allreduce_rms_norm_kernel[(peer_twoshot_grid_size,)](
            allreduce_in,
            residual_in,
            rms_gamma,
            residual_out,
            norm_scratch,
            workspace.local_buffer,
            workspace.peer_buffers,
            workspace.state,
            rms_eps,
            weight_bias,
            M=m,
            H=hidden_dim,
            RANK=workspace.rank,
            WORLD_SIZE=workspace.world_size,
            CAPACITY=workspace.capacity,
            SLOT_CAPACITY=workspace.slot_capacity,
            SCATTER_RANK_CAPACITY=workspace.scatter_rank_capacity,
            STAGE_CAPACITY=workspace.stage_capacity,
            GRID_SIZE=peer_twoshot_grid_size,
            TILES_PER_ROW=peer_twoshot_tiles_per_row,
            BLOCK_SIZE=peer_twoshot_tile_size,
            NORM_BLOCK_SIZE=block_size,
            FP32_ACC=fp32_acc,
            USE_PDL=launch_with_pdl,
            num_warps=32,
            num_stages=1,
            launch_pdl=launch_with_pdl,
        )
        return

    _peer_rowwise_oneshot_allreduce_rms_norm_kernel[(grid_size,)](
        allreduce_in,
        residual_in,
        rms_gamma,
        residual_out,
        norm_scratch,
        workspace.local_buffer,
        workspace.peer_buffers,
        workspace.state,
        rms_eps,
        weight_bias,
        **launch_meta,
    )


def _launch_quantization(
    pattern_code: int,
    norm_input: torch.Tensor,
    quant_out: torch.Tensor,
    scale_out: Optional[torch.Tensor],
    scale_factor: torch.Tensor,
    *,
    launch_with_pdl: bool,
    trigger_completion_at_end: bool,
) -> None:
    m, hidden_dim = norm_input.shape
    if pattern_code == AllReduceFusionPattern.kARResidualRMSNormFP8Quant:
        block_size = 256
        _static_fp8_quant_kernel[(triton.cdiv(m * hidden_dim, block_size),)](
            norm_input,
            quant_out,
            scale_factor,
            m * hidden_dim,
            BLOCK_SIZE=block_size,
            USE_PDL=launch_with_pdl,
            TRIGGER_COMPLETION_AT_END=trigger_completion_at_end,
            num_warps=4,
            launch_pdl=launch_with_pdl,
        )
        return

    block_pairs = triton.next_power_of_2(hidden_dim // 2)
    _static_nvfp4_quant_kernel[(m,)](
        norm_input,
        quant_out,
        scale_out,
        scale_factor,
        H=hidden_dim,
        BLOCK_PAIRS=block_pairs,
        USE_PDL=launch_with_pdl,
        TRIGGER_COMPLETION_AT_END=trigger_completion_at_end,
        num_warps=8,
        launch_pdl=launch_with_pdl,
    )


def _check_forward_only(*tensors) -> None:
    # A Parameter retains requires_grad=True inside no_grad/inference_mode;
    # that flag alone does not imply that this invocation requests autograd.
    if torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    ):
        raise NotImplementedError("fused_allreduce_rms_norm is forward-only")


def fused_allreduce_rms_norm(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    world_size: int,
    launch_with_pdl: bool,
    fp32_acc: bool,
    max_token_num: int,
    pattern_code: int,
    norm_out: Optional[torch.Tensor] = None,
    quant_out: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    scale_factor: Optional[torch.Tensor] = None,
    weight_bias: float = 0.0,
    *,
    workspace: FusedAllReduceRMSNormWorkspace,
) -> None:
    """Match vLLM's fused all-reduce custom-op contract for TP2/TP4/TP8.

    An explicit eagerly allocated workspace selects the Triton MNNVL or
    peer-buffer backend. The integration never selects a backend implicitly. Each backend
    selects its own one-shot or two-shot strategy.

    For the non-quantized pattern, omitting ``norm_out`` writes normalized values
    to ``allreduce_in`` and residual values to ``residual``. Supplying
    ``norm_out`` writes residual values to ``allreduce_in`` and normalized values
    to ``norm_out``. The quantized patterns write only the residual destination
    selected by the same rule plus ``quant_out`` (and FP4 ``scale_out``), matching
    the vLLM/FlashInfer pattern traits.
    """
    if not isinstance(allreduce_in, torch.Tensor):
        raise TypeError("allreduce_in must be a torch.Tensor")
    if allreduce_in.ndim != 2:
        hidden_dim = allreduce_in.shape[-1]
        allreduce_in = allreduce_in.view(-1, hidden_dim)
        residual = residual.view(-1, hidden_dim)
        if norm_out is not None:
            norm_out = norm_out.view(-1, hidden_dim)
    m, hidden_dim = allreduce_in.shape
    if m <= 0 or hidden_dim <= 0:
        raise ValueError(
            f"allreduce_in must have positive [M,H], got {(m, hidden_dim)}"
        )
    _check_world_size(world_size)
    if max_token_num <= 0 or m > max_token_num:
        raise ValueError(
            f"M={m} must not exceed positive max_token_num={max_token_num}"
        )
    if pattern_code not in _SUPPORTED_PATTERNS:
        raise NotImplementedError(f"unsupported pattern_code={pattern_code}")
    if allreduce_in.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError(
            f"only FP16/BF16 inputs are supported, got {allreduce_in.dtype}"
        )
    if not math.isfinite(rms_eps) or rms_eps <= 0:
        raise ValueError(f"rms_eps must be finite and positive, got {rms_eps}")
    if not math.isfinite(weight_bias):
        raise ValueError(f"weight_bias must be finite, got {weight_bias}")
    if launch_with_pdl and not _HAS_PDL:
        raise NotImplementedError("this Triton build does not provide CUDA PDL")

    shape = (m, hidden_dim)
    dtype = allreduce_in.dtype
    device = allreduce_in.device
    if device.type != "cuda":
        raise NotImplementedError("only NVIDIA CUDA tensors are supported")
    allreduce_in = _check_tensor(allreduce_in, "allreduce_in", shape, dtype, device)
    residual = _check_tensor(residual, "residual", shape, dtype, device)
    rms_gamma = _check_tensor(rms_gamma, "rms_gamma", (hidden_dim,), dtype, device)
    if norm_out is not None:
        norm_out = _check_tensor(norm_out, "norm_out", shape, dtype, device)
    _check_forward_only(allreduce_in, residual, rms_gamma, norm_out)
    if _tensors_overlap(allreduce_in, residual):
        raise ValueError("allreduce_in and residual must not overlap")
    if norm_out is not None and any(
        _tensors_overlap(norm_out, tensor)
        for tensor in (allreduce_in, residual, rms_gamma)
    ):
        raise ValueError("explicit norm_out must not alias an input")

    is_quant = pattern_code != AllReduceFusionPattern.kARResidualRMSNorm
    residual_out = residual if norm_out is None else allreduce_in

    if is_quant:
        if scale_factor is None:
            raise ValueError("scale_factor is required for quantization")
        scale_factor = _check_flat_output(
            scale_factor, "scale_factor", 1, torch.float32, device
        )
        if pattern_code == AllReduceFusionPattern.kARResidualRMSNormFP8Quant:
            quant_out = _check_flat_output(
                quant_out,
                "quant_out",
                m * hidden_dim,
                torch.float8_e4m3fn,
                device,
            )
            if scale_out is not None:
                raise ValueError("scale_out is not used by FP8 quantization")
        else:
            capability = torch_device_fn.get_device_capability(device.index)
            if capability[0] < 10:
                raise NotImplementedError("NVFP4 quantization requires SM100 or newer")
            if hidden_dim % 16 != 0:
                raise ValueError("NVFP4 quantization requires H divisible by 16")
            quant_out = _check_flat_output(
                quant_out, "quant_out", m * hidden_dim // 2, torch.uint8, device
            )
            scale_dtype = torch.float8_e4m3fn
            required_scales = ((m + 127) // 128 * 128) * (
                (hidden_dim // 16 + 3) // 4 * 4
            )
            if scale_out is not None and not isinstance(scale_out, torch.Tensor):
                raise TypeError("scale_out must be a torch.Tensor")
            if scale_out is None or scale_out.numel() < required_scales:
                actual = 0 if scale_out is None else scale_out.numel()
                raise ValueError(
                    "scale_out needs at least "
                    f"{required_scales} FP8 elements, got {actual}"
                )
            if scale_out.dtype != scale_dtype or scale_out.device != device:
                raise TypeError(
                    f"scale_out must be contiguous {scale_dtype} on {device}"
                )
            if not scale_out.is_contiguous():
                raise ValueError("scale_out must be contiguous")
    elif any(value is not None for value in (quant_out, scale_out, scale_factor)):
        raise ValueError("quantization outputs are invalid for non-quant pattern")

    if not workspace.is_buffer_size_sufficient(world_size, m, hidden_dim, dtype):
        raise ValueError("the eagerly allocated TP workspace does not fit this call")
    if workspace.max_token_num != max_token_num:
        raise ValueError("max_token_num does not match the eagerly allocated workspace")
    if workspace.local_buffer.device != device:
        raise ValueError("inputs and cached workspace must be on the same device")
    if is_quant:
        norm_scratch = workspace.norm_buffer[:m]
    elif norm_out is None:
        norm_scratch = allreduce_in
    else:
        norm_scratch = norm_out

    capability = torch_device_fn.get_device_capability(device.index)
    tensor_size = m * hidden_dim * allreduce_in.element_size()
    if workspace.backend == "mnnvl":
        use_oneshot = _select_mnnvl_use_oneshot(world_size, tensor_size)
    else:
        use_oneshot = _select_use_oneshot(capability, world_size, tensor_size)
    trigger_completion_at_end = use_oneshot or m > 16

    with torch_device_fn.device(device):
        _launch_fused_allreduce_rms_norm(
            allreduce_in,
            residual,
            rms_gamma,
            residual_out,
            norm_scratch,
            workspace,
            rms_eps=rms_eps,
            weight_bias=weight_bias,
            launch_with_pdl=launch_with_pdl,
            trigger_completion_at_end=trigger_completion_at_end,
            fp32_acc=fp32_acc,
            use_oneshot=use_oneshot,
            chain_quant=is_quant,
        )
        if is_quant:
            _launch_quantization(
                pattern_code,
                norm_scratch,
                quant_out,
                scale_out,
                scale_factor,
                launch_with_pdl=launch_with_pdl,
                trigger_completion_at_end=trigger_completion_at_end,
            )
