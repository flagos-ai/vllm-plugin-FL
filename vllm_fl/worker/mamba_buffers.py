# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner-owned align-mode buffers for models with heterogeneous Mamba states.

Keep the upstream homogeneous path, including its speculative-decode support.
Heterogeneous normal decoding reuses upstream copy planning and kernels once
per equal-spec partition, with independent request/block-index bookkeeping.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker import mamba_utils


@dataclass
class GroupedMambaCopyBuffers:
    groups: list[tuple[mamba_utils.MambaCopyBuffers, tuple, dict[str, int]]]
    validated: bool = field(default=False, init=False)

    def validate_states(self, kv_cache_config, forward_context) -> None:
        """Reject an inconsistent model contract before any cache copy."""
        if self.validated:
            return
        for buffers, copy_funcs, _ in self.groups:
            spec = buffers.mamba_spec
            for gid in buffers.mamba_group_ids:
                for name in kv_cache_config.kv_cache_groups[gid].layer_names:
                    states = forward_context[name].kv_cache
                    if len(states) != len(copy_funcs):
                        raise ValueError(
                            f"Mamba layer {name} has {len(states)} states but "
                            f"{len(copy_funcs)} copy functions"
                        )
                    for state, shape, dtype in zip(states, spec.shapes, spec.dtypes):
                        if (
                            tuple(state.shape[1:]) != tuple(shape)
                            or state.dtype != dtype
                        ):
                            raise ValueError(
                                f"Mamba state does not match spec for {name}"
                            )
        self.validated = True


@dataclass
class GroupedMambaBuffers:
    preprocess: GroupedMambaCopyBuffers
    postprocess_align: None = None


def create_mamba_buffers(
    *,
    model,
    max_num_reqs,
    kv_cache_config,
    make_buffer,
    device,
    with_postprocess_align,
) -> mamba_utils.MambaBuffers | GroupedMambaBuffers:
    partitions: list[tuple[MambaSpec, list[int]]] = []
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if not isinstance(spec, MambaSpec):
            continue
        for existing_spec, group_ids in partitions:
            if spec == existing_spec:
                group_ids.append(gid)
                break
        else:
            partitions.append((spec, [gid]))

    if len(partitions) <= 1:
        return mamba_utils.MambaBuffers.create(
            max_num_reqs=max_num_reqs,
            kv_cache_config=kv_cache_config,
            copy_funcs=model.get_mamba_state_copy_func(),
            make_buffer=make_buffer,
            device=device,
            with_postprocess_align=with_postprocess_align,
        )

    # Validate the entire contract before allocating buffers or launching copies.
    if with_postprocess_align or any(s.num_speculative_blocks for s, _ in partitions):
        raise NotImplementedError(
            "Heterogeneous Mamba align caching does not support speculative decoding"
        )
    if len({spec.block_size for spec, _ in partitions}) != 1:
        raise ValueError("Mamba align partitions must share the scheduler block size")
    get_copy_funcs = getattr(model, "get_mamba_state_copy_funcs", None)
    if not callable(get_copy_funcs):
        raise ValueError(
            "Heterogeneous Mamba caching requires get_mamba_state_copy_funcs(types)"
        )
    funcs_by_type = get_copy_funcs({spec.mamba_type for spec, _ in partitions})
    for spec, _ in partitions:
        funcs = funcs_by_type.get(spec.mamba_type, ())
        if (
            spec.mamba_cache_mode != "align"
            or spec.block_size <= 0
            or len(funcs) != len(spec.shapes)
            or len(funcs) != len(spec.dtypes)
            or not all(callable(func) for func in funcs)
        ):
            raise ValueError(f"Invalid Mamba align copy contract for {spec.mamba_type}")

    groups = []
    for spec, group_ids in partitions:
        copy_funcs = tuple(funcs_by_type[spec.mamba_type])
        entries = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names) for gid in group_ids
        ) * len(copy_funcs)
        n = max_num_reqs * entries
        buffers = mamba_utils.MambaCopyBuffers(
            src_ptrs=make_buffer(n, dtype=torch.uint64),
            dst_ptrs=make_buffer(n, dtype=torch.uint64),
            sizes=make_buffer(n, dtype=torch.int32),
            mamba_group_ids=group_ids,
            mamba_spec=spec,
        )
        groups.append((buffers, copy_funcs, {}))
    return GroupedMambaBuffers(GroupedMambaCopyBuffers(groups))


def preprocess_mamba(
    scheduler_output,
    kv_cache_config,
    cache_config,
    mamba_state_idx,
    input_batch,
    requests,
    forward_context,
    mamba_state_copy_funcs,
    copy_bufs,
) -> None:
    if not isinstance(copy_bufs, GroupedMambaCopyBuffers):
        mamba_utils.preprocess_mamba(
            scheduler_output,
            kv_cache_config,
            cache_config,
            mamba_state_idx,
            input_batch,
            requests,
            forward_context,
            mamba_state_copy_funcs,
            copy_bufs,
        )
        return

    if any(
        buffers.mamba_spec.block_size != cache_config.block_size
        for buffers, _, _ in copy_bufs.groups
    ):
        raise ValueError("Mamba align partitions do not match the scheduler block size")
    copy_bufs.validate_states(kv_cache_config, forward_context)
    num_reqs = len(input_batch.req_ids)
    accepted = input_batch.num_accepted_tokens_cpu[:num_reqs]
    original_accepted = accepted.copy()
    final_accepted = original_accepted.copy()
    for buffers, copy_funcs, state_indices in copy_bufs.groups:
        # Each partition must see the same accepted-token bias. Upstream resets
        # it after copying; letting that reset leak would shift later states.
        accepted[:] = original_accepted
        mamba_utils.preprocess_mamba(
            scheduler_output,
            kv_cache_config,
            cache_config,
            state_indices,
            input_batch,
            requests,
            forward_context,
            copy_funcs,
            buffers,
        )
        np.minimum(final_accepted, accepted, out=final_accepted)
    accepted[:] = final_accepted
