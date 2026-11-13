# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

"""Use tensor copies for V2 Mamba align state migration on Kunlunxin."""

from dataclasses import dataclass

import torch


@dataclass
class MambaGroupStates:
    block_table: torch.Tensor
    conv_states: list[torch.Tensor]
    temporal_states: list[torch.Tensor]


def _copy_conv_state(
    state: torch.Tensor,
    src_block: int,
    dst_block: int,
    token_bias: int,
    dim_first: bool,
) -> None:
    source, destination = state[src_block], state[dst_block]
    if token_bias:
        if dim_first:
            source = source[..., token_bias:]
            destination = destination[..., :-token_bias]
        else:
            source = source[token_bias:]
            destination = destination[:-token_bias]
    if src_block != dst_block:
        destination.copy_(source)
    elif token_bias:
        # Same-block window shifts overlap.
        destination.copy_(source.clone())


def _copy_states(
    groups: list[MambaGroupStates],
    req_indices: torch.Tensor,
    src_cols: torch.Tensor,
    dst_cols: torch.Tensor,
    token_biases: torch.Tensor,
    needs_copy: torch.Tensor,
    *,
    dim_first: bool,
) -> None:
    # Mask inactive entries before looking up physical block IDs. This also
    # avoids using a speculative source offset for rows that need no copy.
    req_indices = torch.where(needs_copy, req_indices, 0)
    src_cols = torch.where(needs_copy, src_cols, 0)
    dst_cols = torch.where(needs_copy, dst_cols, 0)
    token_biases = torch.where(needs_copy, token_biases, 0)
    block_ids = []
    for group in groups:
        table = group.block_table
        block_ids.extend(
            (
                table[req_indices, src_cols],
                table[req_indices, dst_cols],
                table[req_indices, src_cols + token_biases],
            )
        )

    # Copies run before the model forward or after sampling, outside graph
    # capture/replay. Move only the selected block IDs to the host.
    rows = torch.stack((needs_copy, token_biases, *block_ids), dim=1).tolist()
    for needs_copy_row, token_bias, *blocks in rows:
        if not needs_copy_row:
            continue
        for group, offset in zip(groups, range(0, len(blocks), 3)):
            src_block, dst_block, temporal_src_block = blocks[offset : offset + 3]
            for state in group.conv_states:
                _copy_conv_state(state, src_block, dst_block, token_bias, dim_first)
            if temporal_src_block != dst_block:
                for state in group.temporal_states:
                    state[dst_block].copy_(state[temporal_src_block])


def apply_mamba_align_compat() -> None:
    """Replace the V2 model state's copy context without changing FL/V1."""

    from vllm.model_executor.layers.mamba.mamba_utils import (
        get_conv_copy_spec,
        get_temporal_copy_spec,
        is_conv_state_dim_first,
    )
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    from vllm.v1.worker.gpu.model_states import mamba_hybrid
    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

    if mamba_hybrid.MambaSpecDecodeGPUContext is not MambaSpecDecodeGPUContext:
        return

    class KunlunxinMambaCopyContext(MambaSpecDecodeGPUContext):
        def initialize_from_forward_context(
            self,
            kv_cache_config,
            forward_context,
            mamba_state_copy_funcs,
            block_tables,
            aligned_index_block_tables=None,
        ):
            if self.is_initialized:
                return
            super().initialize_from_forward_context(
                kv_cache_config,
                forward_context,
                mamba_state_copy_funcs,
                block_tables,
                aligned_index_block_tables,
            )

            groups = []
            for gid, table in zip(self.mamba_group_ids, block_tables, strict=True):
                group = kv_cache_config.kv_cache_groups[gid]
                conv_states = []
                temporal_states = []
                for layer_name in group.layer_names:
                    spec = group.kv_cache_spec
                    if isinstance(spec, UniformTypeKVCacheSpecs):
                        spec = spec.kv_cache_specs[layer_name]
                    funcs = mamba_state_copy_funcs[spec.mamba_type]
                    caches = forward_context[layer_name].kv_cache
                    for state, copy_func in zip(caches, funcs):
                        if copy_func is get_conv_copy_spec:
                            conv_states.append(state)
                        else:
                            assert copy_func is get_temporal_copy_spec
                            temporal_states.append(state)
                groups.append(MambaGroupStates(table, conv_states, temporal_states))
            self._kunlunxin_states = groups

        def run_fused_precopy(
            self, num_reqs, state_idx_gpu, src_col_gpu, token_bias_gpu, idx_mapping
        ):
            if num_reqs == 0 or not self.is_initialized:
                return
            assert idx_mapping is not None
            req_indices = idx_mapping[:num_reqs]
            valid_rows = req_indices >= 0
            safe_indices = req_indices.clamp_min(0)
            src_cols = src_col_gpu[safe_indices]
            dst_cols = state_idx_gpu[safe_indices]
            needs_copy = valid_rows & (src_cols >= 0) & (src_cols != dst_cols)
            with gpu_sync_allowed():
                _copy_states(
                    self._kunlunxin_states,
                    safe_indices,
                    src_cols,
                    dst_cols,
                    token_bias_gpu[safe_indices],
                    needs_copy,
                    dim_first=is_conv_state_dim_first(),
                )

        def run_fused_postprocess_align(
            self,
            num_reqs,
            num_accepted_tokens_gpu,
            state_idx_gpu,
            new_num_computed_tokens_gpu,
            idx_mapping,
        ):
            if num_reqs == 0 or not self.is_initialized:
                return
            req_indices = idx_mapping[:num_reqs]
            safe_indices = req_indices.clamp_min(0)
            accepted = num_accepted_tokens_gpu[safe_indices]
            src_cols = state_idx_gpu[safe_indices]
            computed = new_num_computed_tokens_gpu[safe_indices]
            aligned = (computed // self.block_size) * self.block_size
            dst_cols = aligned // self.block_size - 1
            token_biases = aligned - (computed - accepted + 1)
            at_boundary = (req_indices >= 0) & (dst_cols >= 0) & (token_biases >= 0)
            needs_copy = at_boundary & ((src_cols != dst_cols) | (token_biases != 0))
            with gpu_sync_allowed():
                _copy_states(
                    self._kunlunxin_states,
                    safe_indices,
                    src_cols,
                    dst_cols,
                    token_biases,
                    needs_copy,
                    dim_first=is_conv_state_dim_first(),
                )
                # Match the upstream per-request update, including exact
                # self-copies where no bytes needed to move.
                reset_rows = req_indices[at_boundary & (src_cols == dst_cols)]
                num_accepted_tokens_gpu[reset_rows] = 1

    mamba_hybrid.MambaSpecDecodeGPUContext = KunlunxinMambaCopyContext
