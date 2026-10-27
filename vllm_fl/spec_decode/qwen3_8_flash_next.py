# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native vLLM 0.24 K1 MTP with separate QSA main/raw/compressed caches.

Sampling and first-pass token/hidden alignment remain in EagleProposer. Unlike
Step3p5's multi-group adapter, every slot here uses its own logical block size
and compression ratio, rebuilt from the post-first-pass request coordinates.
"""

from __future__ import annotations

from copy import copy

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID
from vllm.v1.worker.utils import AttentionGroup


def packed_request_positions(common: CommonAttentionMetadata):
    """Map the transformed packed token rows to unchanged request-table rows."""
    count = common.num_actual_tokens
    requests = common.num_reqs
    if common.query_start_loc_cpu.device.type != "cpu":
        raise ValueError("QSA draft query_start_loc_cpu must reside on CPU")
    mapped = int(common.query_start_loc_cpu[-1])
    if mapped < 0 or mapped > count:
        raise ValueError("QSA draft query offsets exceed the actual token count")
    device = common.query_start_loc.device
    token_rows = torch.arange(count, dtype=torch.int64, device=device)
    request_rows = torch.full((count,), -1, dtype=torch.int64, device=device)
    positions = torch.full_like(request_rows, -1)
    if mapped:
        lengths = torch.diff(common.query_start_loc)
        row_ids = torch.repeat_interleave(
            torch.arange(requests, device=device, dtype=torch.int64),
            lengths,
            output_size=mapped,
        )
        request_rows[:mapped].copy_(row_ids)
        offsets = token_rows[:mapped] - common.query_start_loc[row_ids]
        positions[:mapped].copy_(
            common.seq_lens[row_ids].long() - lengths[row_ids] + offsets
        )
    return token_rows, request_rows, positions


def qsa_group_slot_mapping(
    block_table: torch.Tensor,
    request_rows: torch.Tensor,
    positions: torch.Tensor,
    valid_tokens: torch.Tensor,
    *,
    logical_block_size: int,
    compress_ratio: int,
    max_model_len: int,
) -> torch.Tensor:
    """Build physical slots; compressed caches store only complete microblocks."""
    if logical_block_size <= 0 or compress_ratio <= 0:
        raise ValueError("QSA draft block size and compression must be positive")
    if logical_block_size % compress_ratio:
        raise ValueError("QSA logical block size must divide by compression ratio")
    if block_table.ndim != 2:
        raise ValueError("QSA draft block tables must be two-dimensional")
    if not (request_rows.shape == positions.shape == valid_tokens.shape):
        raise ValueError("QSA draft token coordinates have inconsistent shapes")
    if not all(block_table.shape):
        return torch.full_like(positions, PADDING_SLOT_ID)
    storage_block_size = logical_block_size // compress_ratio
    block_columns = torch.div(
        positions.clamp_min(0), logical_block_size, rounding_mode="floor"
    )
    valid = (
        valid_tokens
        & (request_rows >= 0)
        & (request_rows < block_table.shape[0])
        & (positions >= 0)
        & (positions < max_model_len)
        & (block_columns < block_table.shape[1])
    )
    safe_rows = request_rows.clamp(0, block_table.shape[0] - 1)
    safe_columns = block_columns.clamp(0, block_table.shape[1] - 1)
    physical_blocks = block_table[safe_rows, safe_columns].long()
    # vLLM's hybrid allocator reserves block zero for padded/null requests.
    valid &= physical_blocks > 0
    if compress_ratio != 1:
        valid &= (positions + 1).remainder(compress_ratio) == 0
    offsets = torch.div(
        positions.clamp_min(0).remainder(logical_block_size),
        compress_ratio,
        rounding_mode="floor",
    )
    slots = physical_blocks * storage_block_size + offsets
    return torch.where(valid, slots, torch.full_like(slots, PADDING_SLOT_ID))


class Qwen38MTPProposer(EagleProposer):
    """K1 Qwen MTP; target graph scheduling and native sampling stay unchanged."""

    _qwen38_multi_group_mtp = True

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        spec = vllm_config.speculative_config
        if spec is None or spec.method != "mtp" or spec.num_speculative_tokens != 1:
            raise NotImplementedError("Qwen3.8 multi-group MTP currently requires K=1")
        if not getattr(spec, "disable_padded_drafter_batch", False):
            raise NotImplementedError(
                "Qwen3.8 MTP requires disable_padded_drafter_batch=True "
                "for chunked-prefill token alignment"
            )
        if spec.parallel_drafting:
            raise NotImplementedError("Qwen3.8 MTP requires sequential drafting")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("Qwen3.8 MTP currently requires PP=1")
        super().__init__(vllm_config, device, runner)
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._group_specs: dict[int, FullAttentionSpec] = {}
        self._per_group_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._first_pass_sample_indices: torch.Tensor | None = None
        self._slot_mappings_prepared = False
        self._prepared_slot_count = 0

    def load_model(self, target_model):
        super().load_model(target_model)
        if self.vllm_config.additional_config.get("qwen38_mtp_graph", False):
            if self.speculative_config.enforce_eager:
                raise ValueError("qwen38_mtp_graph conflicts with speculative enforce_eager")
            if self.vllm_config.parallel_config.data_parallel_size != 1:
                raise NotImplementedError("Qwen MTP graph currently requires DP=1")
            from vllm_fl.spec_decode.qwen38_draft_graph import Qwen38DraftGraph
            self.model = Qwen38DraftGraph(self.model, self.vllm_config)

    @staticmethod
    def unpad_target_metadata(
        common: CommonAttentionMetadata,
        num_actual_tokens: int,
        num_actual_reqs: int,
    ) -> CommonAttentionMetadata:
        """Preserve the native CPU upper bound across target graph unpadding.

        vLLM 0.24's ``unpadded`` omits this field, but native ``prepare_inputs``
        needs it when removing rejected draft rows. Slice the original bound;
        neither device seq_lens nor the deprecated CPU copy is a substitute.
        """
        unpadded = common.unpadded(num_actual_tokens, num_actual_reqs)
        if common.seq_lens_cpu_upper_bound is not None:
            unpadded.seq_lens_cpu_upper_bound = common.seq_lens_cpu_upper_bound[
                :num_actual_reqs
            ]
        return unpadded

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        locations = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                if name not in self._draft_attn_layer_names:
                    continue
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    if name not in spec.kv_cache_specs:
                        raise ValueError(f"Missing QSA draft cache spec for {name}")
                    spec = spec.kv_cache_specs[name]
                if name in locations:
                    raise ValueError(
                        f"QSA draft layer occurs in multiple groups: {name}"
                    )
                locations[name] = (gid, spec)
        missing = self._draft_attn_layer_names - locations.keys()
        if missing:
            raise ValueError(f"QSA draft cache groups omit layers: {sorted(missing)}")
        if not locations:
            raise ValueError("Qwen3.8 MTP has no registered attention cache layers")
        groups = {}
        main_gids = set()
        for name in sorted(locations):
            gid, spec = locations[name]
            if not isinstance(spec, FullAttentionSpec):
                raise TypeError(
                    f"Unsupported QSA draft cache spec: {type(spec).__name__}"
                )
            if name not in layers:
                raise ValueError(f"QSA draft layer missing from static context: {name}")
            if not isinstance(spec, MLAAttentionSpec) and spec.head_size_v > 0:
                main_gids.add(gid)
            backend = layers[name].get_attn_backend()
            key = (gid, backend.full_cls_name())
            if key in groups:
                if groups[key].kv_cache_spec != spec:
                    raise ValueError(
                        "QSA draft group/backend has incompatible cache specs"
                    )
                groups[key].layer_names.append(name)
                continue
            kernel_size = None
            if kernel_block_sizes is not None:
                if gid >= len(kernel_block_sizes):
                    raise ValueError("QSA draft kernel block-size list is incomplete")
                kernel_size = kernel_block_sizes[gid]
            group = AttentionGroup(
                backend=backend,
                layer_names=[name],
                kv_cache_spec=spec,
                kv_cache_group_id=gid,
            )
            group.create_metadata_builders(
                self.vllm_config,
                self.device,
                kernel_block_size=kernel_size,
            )
            groups[key] = group
        if len(main_gids) != 1:
            raise ValueError("Qwen3.8 MTP requires one primary main-K/V cache group")
        self.kv_cache_gid = main_gids.pop()
        self.draft_attn_groups = list(groups.values())
        self._group_specs.clear()
        self._per_group_block_tables.clear()
        self._per_group_slot_mapping_buffers.clear()
        for group in self.draft_attn_groups:
            gid = group.kv_cache_group_id
            spec = group.get_metadata_builder().kv_cache_spec
            if gid in self._group_specs and self._group_specs[gid] != spec:
                raise ValueError(
                    "QSA draft group has inconsistent effective block sizes"
                )
            self._group_specs[gid] = spec
            self._per_group_slot_mapping_buffers[gid] = (
                self._slot_mapping_buffer
                if gid == self.kv_cache_gid
                else torch.empty(
                    self.max_positions, dtype=torch.int64, device=self.device
                )
            )
            self._per_group_slot_mapping_buffers[gid].fill_(PADDING_SLOT_ID)
        self.block_size = self._group_specs[self.kv_cache_gid].block_size
        self._slot_mappings_prepared = False
        self._prepared_slot_count = 0

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if gid not in self._group_specs:
            return
        if block_table.ndim != 2 or slot_mapping.ndim != 1:
            raise ValueError("Invalid runner QSA group block-table/slot layout")
        # Slot values belong to the target's original packed input. Only the
        # request table is reusable after native first-pass token compaction.
        self._per_group_block_tables[gid] = block_table
        self._first_pass_sample_indices = None
        self._slot_mappings_prepared = False

    def set_inputs_first_pass(
        self,
        target_token_ids,
        next_token_ids,
        target_positions,
        target_hidden_states,
        token_indices_to_sample,
        cad,
        num_rejected_tokens_gpu,
    ):
        if self.num_speculative_tokens not in (0, 1) or self.needs_extra_input_slots:
            raise NotImplementedError("Qwen3.8 multi-group drafting only supports K1")
        result = super().set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            cad,
            num_rejected_tokens_gpu,
        )
        # Padded drafting retains rejected token rows. Never write their K/V or
        # compressed-index groups; the native sampler still selects the same row.
        self._first_pass_sample_indices = result[1]
        self._slot_mappings_prepared = False
        return result

    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        if draft_index != 0 or self.num_speculative_tokens not in (0, 1):
            raise NotImplementedError("Qwen3.8 K>1 cache evolution is not implemented")
        common = common_attn_metadata
        count, nreq = common.num_actual_tokens, common.num_reqs
        if count > self.max_positions:
            raise ValueError("QSA draft token count exceeds fixed slot buffers")
        if common.slot_mapping.numel() < count:
            raise ValueError("QSA draft validity mask is shorter than packed tokens")
        tokens, requests, positions = packed_request_positions(common)
        valid = (requests >= 0) & (common.slot_mapping[:count] >= 0)
        if self._first_pass_sample_indices is None:
            raise RuntimeError(
                "QSA draft metadata must follow native first-pass inputs"
            )
        if self._first_pass_sample_indices.numel() != nreq:
            raise ValueError("QSA draft sample rows do not match request-table rows")
        if nreq:
            valid &= tokens <= self._first_pass_sample_indices[requests.clamp_min(0)]
        else:
            valid.zero_()
        per_group, per_layer = [], {}
        for group in self.draft_attn_groups:
            gid = group.kv_cache_group_id
            if gid not in self._per_group_block_tables:
                raise RuntimeError(f"Runner did not provide QSA draft group {gid}")
            table = self._per_group_block_tables[gid]
            if table.shape[0] < nreq:
                raise ValueError("QSA group block table has fewer rows than requests")
            table = table[:nreq]
            spec = self._group_specs[gid]
            ratio = spec.compress_ratio if isinstance(spec, MLAAttentionSpec) else 1
            slots = qsa_group_slot_mapping(
                table,
                requests,
                positions,
                valid,
                logical_block_size=spec.block_size,
                compress_ratio=ratio,
                max_model_len=self.max_model_len,
            )
            buffer = self._per_group_slot_mapping_buffers[gid]
            buffer.fill_(PADDING_SLOT_ID)
            buffer[:count].copy_(slots)
            cm = copy(common)
            cm.block_table_tensor = table
            cm.slot_mapping = buffer[:count]
            metadata = group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cm,
                draft_index=0,
            )
            per_group.append(metadata)
            for name in group.layer_names:
                per_layer[name] = metadata
        self._prepared_slot_count = count
        self._slot_mappings_prepared = True
        return per_group, per_layer

    def _get_slot_mapping(self, num_tokens: int, slot_mapping=None):
        if num_tokens > self.max_positions:
            raise ValueError("QSA padded draft tokens exceed fixed slot capacity")
        if slot_mapping is None:
            # Native dummy_run has no attention metadata and must not retain a
            # real request's slots from a previous invocation.
            for buffer in self._per_group_slot_mapping_buffers.values():
                buffer.fill_(PADDING_SLOT_ID)
            self._slot_mappings_prepared = False
        elif not self._slot_mappings_prepared:
            raise RuntimeError("QSA per-group slots were not rebuilt for this draft")
        elif num_tokens < self._prepared_slot_count:
            raise ValueError("QSA graph token padding is smaller than actual tokens")
        result = {}
        for group in self.draft_attn_groups:
            buffer = self._per_group_slot_mapping_buffers[group.kv_cache_group_id]
            if slot_mapping is not None:
                buffer[self._prepared_slot_count : num_tokens].fill_(PADDING_SLOT_ID)
            view = buffer[:num_tokens]
            for name in group.layer_names:
                result[name] = view
        return result
