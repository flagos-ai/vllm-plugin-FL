# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4 runtime policy and scoped method patches for official vLLM 0.28.

The host adapters insert shared input-packet, metadata-graph and sequential
MTP3 execution into the fixed V1 runtime. Other instances call the captured
native methods. Constructors always delegate to the native constructor once.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import torch

from vllm.tracing import instrument

from ..metadata import set_current_cpu_seq_lens_proof, validated_cpu_seq_lens_proof
from ..runtime import (
    _ABSENT,
    _CURRENT_RUNNER,
    _LOCK,
    Hy4ExecuteModelState,
    apply_patch_plan,
    make_common_attention_metadata,
    runnersynchronize_input_prep,
)

_INSTALLED = False


def create_hy4_runtime(config):
    """Decline unsupported configurations before creating a graph or buffer."""
    hf = config.model_config.hf_config
    if hf.model_type not in ("hy_v4", "hy_v4_vl"):
        return None
    from vllm.config import CUDAGraphMode
    from vllm.platforms import current_platform

    from ..indexer_policy import use_int8_indexer
    from .graphs import supports_mixed_config

    if not current_platform.is_cuda():
        return None
    if current_platform.get_device_capability() != (9, 0):
        return None
    parallel = config.parallel_config
    spec = config.speculative_config
    compilation = config.compilation_config
    if (
        spec is None
        or spec.method != "mtp"
        or spec.num_speculative_tokens != 3
        or spec.enforce_eager
        or spec.parallel_drafting
        or spec.disable_padded_drafter_batch
        or parallel.tensor_parallel_size != 16
        or not parallel.enable_expert_parallel
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.use_ubatching
        or config.lora_config is not None
        or compilation.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
    ):
        return None
    if not use_int8_indexer():
        return None
    if not supports_mixed_config(config):
        return None
    return Hy4RuntimeHooks()


class Hy4RuntimeHooks:
    """Framework resource factories; numerical providers retain their policy."""

    mixed_graphs = True

    @staticmethod
    def make_common_metadata_graph():
        from .metadata import CommonAttentionMetadataGraphRunner

        return CommonAttentionMetadataGraphRunner()

    @staticmethod
    def make_input_staging(runner):
        from .metadata import make_async_input_staging

        return make_async_input_staging(runner)

    @staticmethod
    def make_draft_graph(proposer, runner):
        from .graphs import make_hyv4_draft_graph

        return make_hyv4_draft_graph(proposer, runner)

    @staticmethod
    def select_decode_deltas(*args, **kwargs):
        from ..metadata import select_decode_deltas

        return select_decode_deltas(*args, **kwargs)

    @staticmethod
    def decode_lens_from_common(*args, **kwargs):
        from ..metadata import decode_lens_from_common

        return decode_lens_from_common(*args, **kwargs)

    @staticmethod
    def wrap_breakable(*args, **kwargs):
        from .graphs import wrap_native_breakable

        return wrap_native_breakable(*args, **kwargs)

    @staticmethod
    def warm_full_decode_metadata(*args, **kwargs):
        from .indexer import warm_full_decode_metadata_before_capture

        return warm_full_decode_metadata_before_capture(*args, **kwargs)


def prepare_runtime_for_worker(config):
    """Select and install before constructing this NVIDIA worker's V1 runner."""
    if config is None:
        return False
    hf = config.model_config.hf_config
    if hf.model_type not in ("hy_v4", "hy_v4_vl"):
        return False
    if create_hy4_runtime(config) is None:
        return False
    if config.use_v2_model_runner:
        raise ValueError(
            "Hy4 MTP3 graph optimization requires VLLM_USE_V2_MODEL_RUNNER=0"
        )

    install_runtime_patches()
    return True


def _scoped_method(original, implementation, module, kind, name):
    @wraps(original if original is not _ABSENT else implementation)
    def scoped(self, *args, **kwargs):
        enabled = getattr(self, "_hy4_runtime_enabled", False)
        if name == "__init__":
            self._hy4_runtime_enabled = False
            config = kwargs.get("vllm_config", args[0] if args else None)
            runtime = create_hy4_runtime(config) if config is not None else None
            enabled = runtime is not None
            if kind == "proposer":
                runner = kwargs.get("runner", args[3] if len(args) > 3 else None)
                enabled = (
                    enabled and getattr(runner, "_model_runtime", None) is not None
                )
            if enabled:
                self._hy4_runtime_module = module
                self._hy4_runtime_enabled = True
                if kind == "runner":
                    self._model_runtime = runtime
                    self.common_attention_metadata_graph = None
                    self._fl_async_input_staging = None
                    self._mrope_decode_state = None
                    self._mrope_decode_active = False
                else:
                    self._model_draft_graph = None
        if not enabled:
            if original is _ABSENT:
                raise AttributeError(name)
            return original(self, *args, **kwargs)
        runner_token = None
        proof_token = None
        if kind == "runner":
            runner_token = _CURRENT_RUNNER.set(self)
            if name in ("execute_model", "sample_tokens", "_dummy_run"):
                from ..metadata import push_current_cpu_seq_lens_proof

                proof_token = push_current_cpu_seq_lens_proof(None)
        try:
            if name == "__init__":
                # Delegate the real constructor, retaining every installed
                # wrapper and native initialization side effect exactly once.
                result = original(self, *args, **kwargs)
                if kind == "runner":
                    runtime = self._model_runtime
                    self.common_attention_metadata_graph = (
                        runtime.make_common_metadata_graph()
                    )
                    if self.uses_mrope:
                        self._mrope_decode_state = self._make_buffer(
                            self.max_num_reqs + 1, dtype=module.torch.int64
                        )
                    self._fl_async_input_staging = runtime.make_input_staging(self)
                else:
                    self._model_draft_graph = runner._model_runtime.make_draft_graph(
                        self, runner
                    )
                return result
            return implementation(self, *args, **kwargs)
        finally:
            if proof_token is not None:
                from ..metadata import reset_current_cpu_seq_lens_proof

                reset_current_cpu_seq_lens_proof(proof_token)
            if runner_token is not None:
                _CURRENT_RUNNER.reset(runner_token)

    return scoped


def runner_prepare_input_ids(
    self, scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens
):
    """Prepare the input IDs for the current batch.

    Carefully handles the `prev_sampled_token_ids` which can be cached
    from the previous engine iteration, in which case those tokens on the
    GPU need to be copied into the corresponding slots into input_ids.

    Uses self.prev_positions[:num_reqs] which maps current pos -> prev pos
    (-1 for new requests).
    """
    _native = self._hy4_runtime_module
    if self._pp_recv_work is not None:
        self._pp_recv_work.wait()
        self._pp_recv_work = None
    if self.input_batch.prev_sampled_token_ids is None:
        self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
        if self.enable_prompt_embeds:
            self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
            self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        return
    stager = self._fl_async_input_staging
    if stager is not None and stager.apply_if_prepared(
        scheduler_output, num_reqs, total_num_scheduled_tokens
    ):
        return
    prev_positions = self.prev_positions.np[:num_reqs]
    scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    sample_flattened_indices = []
    spec_flattened_indices = []
    prev_draft_token_indices = []
    prev_indices = []
    common_indices_match = True
    max_flattened_index = -1
    total_num_spec_tokens = 0
    for cur_index in range(num_reqs):
        prev_index = prev_positions[cur_index]
        if prev_index < 0:
            continue
        prev_indices.append(prev_index)
        req_id = self.input_batch.req_ids[cur_index]
        draft_len = len(scheduled_spec_tokens.get(req_id, ()))
        total_num_spec_tokens += draft_len
        flattened_index = cu_num_tokens[cur_index].item() - 1
        sample_flattened_indices.append(flattened_index - draft_len)
        spec_flattened_indices.extend(
            range(flattened_index - draft_len + 1, flattened_index + 1)
        )
        start = prev_index * self.prev_num_spec_tokens
        prev_draft_token_indices.extend(range(start, start + draft_len))
        common_indices_match &= prev_index == flattened_index
        max_flattened_index = max(max_flattened_index, flattened_index)
    num_common_tokens = len(sample_flattened_indices)
    total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
    if self.enable_prompt_embeds:
        self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
    if num_common_tokens < total_without_spec:
        self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
        if self.enable_prompt_embeds:
            self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
    if num_common_tokens == 0:
        return
    if common_indices_match and max_flattened_index == num_common_tokens - 1:
        self.input_ids.gpu[:num_common_tokens].copy_(
            self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0],
            non_blocking=True,
        )
        return
    sampled_tokens_index_tensor = _native.torch.tensor(
        sample_flattened_indices,
        dtype=_native.torch.int64,
        pin_memory=_native.PIN_MEMORY,
    ).to(self.device, non_blocking=True)
    prev_common_req_indices_tensor = _native.torch.tensor(
        prev_indices, dtype=_native.torch.int64, pin_memory=_native.PIN_MEMORY
    ).to(self.device, non_blocking=True)
    self.input_ids.gpu.scatter_(
        dim=0,
        index=sampled_tokens_index_tensor,
        src=self.input_batch.prev_sampled_token_ids[prev_common_req_indices_tensor, 0],
    )
    if self._draft_token_ids is None or not spec_flattened_indices:
        return
    assert isinstance(self._draft_token_ids, _native.torch.Tensor)
    draft_tokens_index_tensor = _native.torch.tensor(
        spec_flattened_indices, dtype=_native.torch.int64, pin_memory=_native.PIN_MEMORY
    ).to(self.device, non_blocking=True)
    prev_draft_token_indices_tensor = _native.torch.tensor(
        prev_draft_token_indices,
        dtype=_native.torch.int64,
        pin_memory=_native.PIN_MEMORY,
    ).to(self.device, non_blocking=True)
    draft_token_ids = self._draft_token_ids.to(dtype=_native.torch.int32)
    self.input_ids.gpu.scatter_(
        dim=0,
        index=draft_tokens_index_tensor,
        src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
    )


def runner_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
    """
    Returns:
        tuple[logits_indices, spec_decode_metadata, max_num_sampled_tokens]
    """
    _native = self._hy4_runtime_module
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    assert total_num_scheduled_tokens > 0
    num_reqs = self.input_batch.num_reqs
    assert num_reqs > 0
    self.input_batch.block_table.commit_block_table(num_reqs)
    req_indices = _native.np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
    cu_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)
    positions_np = (
        self.input_batch.num_computed_tokens_cpu[req_indices]
        + self.query_pos.np[: cu_num_tokens[-1]]
    )
    if self.uses_mrope and (not self._prepare_decode_mrope(scheduler_output)):
        self._calc_mrope_positions(scheduler_output)
    if self.uses_xdrope_dim > 0:
        self._calc_xdrope_positions(scheduler_output)
    token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
    token_indices_tensor = _native.torch.from_numpy(token_indices)
    _native.torch.index_select(
        self.input_batch.token_ids_cpu_tensor.flatten(),
        0,
        token_indices_tensor,
        out=self.input_ids.cpu[:total_num_scheduled_tokens],
    )
    if self.enable_prompt_embeds:
        is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
        _native.torch.index_select(
            is_token_ids,
            0,
            token_indices_tensor,
            out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
        )
    if self.input_batch.req_prompt_embeds:
        output_idx = 0
        for req_idx in range(num_reqs):
            num_sched = num_scheduled_tokens[req_idx]
            if req_idx not in self.input_batch.req_prompt_embeds:
                output_idx += num_sched
                continue
            if num_sched <= 0:
                output_idx += num_sched
                continue
            req_embeds = self.input_batch.req_prompt_embeds[req_idx]
            start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]
            if start_pos >= req_embeds.shape[0]:
                output_idx += num_sched
                continue
            end_pos = start_pos + num_sched
            actual_end = min(end_pos, req_embeds.shape[0])
            actual_num_sched = actual_end - start_pos
            if actual_num_sched > 0:
                self.inputs_embeds.cpu[
                    output_idx : output_idx + actual_num_sched
                ].copy_(req_embeds[start_pos:actual_end])
            output_idx += num_sched
    self.query_start_loc.np[0] = 0
    self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
    self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
    self.query_start_loc.copy_to_gpu()
    query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]
    _native.torch.add(
        self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
        _native.torch.from_numpy(num_scheduled_tokens),
        out=self.optimistic_seq_lens_cpu[:num_reqs],
    )
    self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
    prev_req_id_to_index = self.input_batch.prev_req_id_to_index
    self._compute_prev_positions(num_reqs)
    num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
    num_tokens_np = _native.np.array(num_tokens, dtype=_native.np.int32)
    self.discard_request_mask.np[:num_reqs] = (
        self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
    )
    self.discard_request_mask.copy_to_gpu(num_reqs)
    needs_cpu_accepted_counts = self.num_accepted_tokens_event is not None and (
        not (
            self.use_async_scheduling and self.cache_config.mamba_cache_mode != "align"
        )
    )
    if needs_cpu_accepted_counts:
        assert self.num_accepted_tokens_event is not None
        self.num_accepted_tokens_event.synchronize()
        if self.use_async_scheduling and prev_req_id_to_index:
            prev_idx = self.prev_positions.np[:num_reqs]
            new_mask = prev_idx < 0
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[
                    _native.np.where(new_mask, 0, prev_idx)
                ]
            )
            self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
            self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                self.num_accepted_tokens.np[:num_reqs]
            )
        else:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )
        self.num_accepted_tokens.np[num_reqs:].fill(1)
        self.num_accepted_tokens.copy_to_gpu()
    else:
        self.num_accepted_tokens.np.fill(1)
        self.num_accepted_tokens.gpu.fill_(1)
    if self.mamba_prev_last_scheduled_idx is not None:
        _native.mamba_utils.preprocess_mamba_all_specdec(
            scheduler_output,
            self.input_batch,
            self.mamba_state_idx,
            num_reqs,
            self.mamba_prev_last_scheduled_idx,
        )
    if (
        self.use_async_spec_decode
        and self.valid_sampled_token_count_gpu is not None
        and prev_req_id_to_index
    ):
        stager = self._fl_async_input_staging
        if stager is None or not stager.prepare_and_correct(
            scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens
        ):
            self.prev_positions.copy_to_gpu(num_reqs)
            self.prev_num_draft_tokens.copy_to_gpu()
            cpu_values = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
                device=self.device, non_blocking=True
            )
            _native.update_num_computed_tokens_for_batch_change(
                self.num_computed_tokens,
                self.num_accepted_tokens.gpu[:num_reqs],
                self.prev_positions.gpu[:num_reqs],
                self.valid_sampled_token_count_gpu,
                self.prev_num_draft_tokens.gpu,
                cpu_values,
            )
    else:
        self.num_computed_tokens[:num_reqs].copy_(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )
    self.req_indices.np[:total_num_scheduled_tokens] = req_indices
    self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
    req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]
    self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
    self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
    self.num_scheduled_tokens.copy_to_gpu(num_reqs)
    num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
    if self.common_attention_metadata_graph is None:
        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(_native.torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )
        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        )
        self.seq_lens[num_reqs:].fill_(0)
    if self.common_attention_metadata_graph is None:
        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )
    self._prepare_input_ids(
        scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens
    )
    if self.uses_mrope and (not self._mrope_decode_active):
        for row in range(self.mrope_positions.gpu.shape[0]):
            self.mrope_positions.gpu[row, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[row, :total_num_scheduled_tokens],
                non_blocking=True,
            )
    elif self.uses_xdrope_dim > 0:
        self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
            self.xdrope_positions.cpu[:, :total_num_scheduled_tokens], non_blocking=True
        )
    if self.use_async_spec_decode and (
        self.uses_mrope and (not self._mrope_decode_active) or self.uses_xdrope_dim > 0
    ):
        drift = self.num_computed_tokens[req_indices_gpu].to(
            _native.torch.int64
        ) - self.input_batch.num_computed_tokens_cpu_tensor[req_indices].to(
            device=self.device, dtype=_native.torch.int64, non_blocking=True
        )
        target = self.mrope_positions if self.uses_mrope else self.xdrope_positions
        target.gpu[:, :total_num_scheduled_tokens] += drift
    use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
    if not use_spec_decode:
        logits_indices = query_start_loc[1:] - 1
        spec_decode_metadata = None
        num_sampled_tokens = _native.np.ones(num_reqs, dtype=_native.np.int32)
    else:
        num_draft_tokens = _native.np.zeros(num_reqs, dtype=_native.np.int32)
        num_decode_draft_tokens = _native.np.full(num_reqs, -1, dtype=_native.np.int32)
        for (
            req_id,
            draft_token_ids,
        ) in scheduler_output.scheduled_spec_decode_tokens.items():
            req_idx = self.input_batch.req_id_to_index[req_id]
            draft_len = len(draft_token_ids)
            num_draft_tokens[req_idx] = draft_len
            if num_scheduled_tokens[req_idx] == draft_len + 1:
                num_decode_draft_tokens[req_idx] = draft_len
        spec_decode_metadata = self._calc_spec_decode_metadata(
            num_draft_tokens, cu_num_tokens
        )
        logits_indices = spec_decode_metadata.logits_indices
        num_sampled_tokens = num_draft_tokens + 1
        self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
        self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
        self.num_decode_draft_tokens.copy_to_gpu()
    if self.lora_config:
        assert (
            _native.np.sum(num_sampled_tokens)
            <= self.vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.set_active_loras(
            self.input_batch, num_scheduled_tokens, num_sampled_tokens
        )
    return (logits_indices, spec_decode_metadata, int(num_sampled_tokens.max()))


def runner_build_attention_metadata(
    self,
    num_tokens,
    num_reqs,
    max_query_len,
    num_tokens_padded=None,
    num_reqs_padded=None,
    ubatch_slices=None,
    logits_indices=None,
    max_num_sampled_tokens=None,
    use_spec_decode=False,
    for_cudagraph_capture=False,
    num_scheduled_tokens=None,
    cascade_attn_prefix_lens=None,
    slot_mappings=None,
    block_table_rows_are_current=False,
):
    """
    Returns:
        tuple[attn_metadata, spec_decode_common_attn_metadata]
    """
    _native = self._hy4_runtime_module
    set_current_cpu_seq_lens_proof(None)
    if len(self.kv_cache_config.kv_cache_groups) == 0:
        return ({}, None)
    num_tokens_padded = num_tokens_padded or num_tokens
    num_reqs_padded = num_reqs_padded or num_reqs
    assert num_reqs_padded is not None and num_tokens_padded is not None
    attn_metadata = {}
    if ubatch_slices is not None:
        attn_metadata = [dict() for _ in range(len(ubatch_slices))]
    if for_cudagraph_capture:
        max_seq_len = self.max_model_len
    else:
        max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()
    kv_cache_groups = self.kv_cache_config.kv_cache_groups

    def _get_block_table(kv_cache_gid: int):
        assert num_reqs_padded is not None and num_tokens_padded is not None
        kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
        if isinstance(kv_cache_spec, _native.EncoderOnlyAttentionSpec):
            blk_table_tensor = _native.torch.zeros(
                (num_reqs_padded, 1), dtype=_native.torch.int32, device=self.device
            )
        else:
            blk_table = self.input_batch.block_table[kv_cache_gid]
            blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)
        if not block_table_rows_are_current or isinstance(
            kv_cache_spec, _native.EncoderOnlyAttentionSpec
        ):
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(_native.NULL_BLOCK_ID)
        return blk_table_tensor

    assert slot_mappings is not None
    block_table_gid_0 = _get_block_table(0)
    slot_mapping_gid_0 = slot_mappings[0]
    if self.routed_experts_initialized:
        slot_mapping_attn = slot_mappings[self.routed_experts_capturer.attn_gid]
        self.routed_experts_slot_mapping_device[:num_tokens].copy_(
            slot_mapping_attn[:num_tokens]
        )
    num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
        :num_reqs_padded
    ]
    num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[
        :num_reqs_padded
    ]
    seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
    seq_lens_cpu_upper_bound = seq_lens_cpu
    is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
    is_prefilling[num_reqs:] = False
    if self.use_async_spec_decode:
        seq_lens_cpu = None
        num_computed_tokens_cpu = None
    req_doc_ranges = None
    if self.is_mm_prefix_lm:
        req_doc_ranges = {}
        hf_text_config = self.model_config.hf_text_config
        _bidi_sw = getattr(hf_text_config, "sliding_window", None)
        _clamps_in_kernel = getattr(self.model, "mm_prefix_clamp_sliding_window", False)
        for req_id in self.input_batch.req_ids:
            image_doc_ranges = []
            req_state = self.requests[req_id]
            for mm_feature in req_state.mm_features:
                if mm_feature.modality == "audio":
                    continue
                pos_info = mm_feature.mm_position
                img_doc_range = pos_info.extract_embeds_range()
                for r in img_doc_range:
                    if (
                        not _clamps_in_kernel
                        and _bidi_sw is not None
                        and (r[1] - r[0] + 1 > _bidi_sw)
                    ):
                        continue
                    image_doc_ranges.append(r)
            req_idx = self.input_batch.req_id_to_index[req_id]
            req_doc_ranges[req_idx] = image_doc_ranges
    rswa_prefix_lens = None
    if self.model_config.rswa_window is not None:
        rswa_prefix_lens = num_prompt_tokens_cpu
    replayssm_decode_base_cpu = None
    if self.cache_config.use_replayssm:
        replayssm_decode_base_cpu = self.input_batch.replayssm_decode_base_cpu_tensor[
            :num_reqs_padded
        ]
    # Upper bounds alone are not proof of exact decode lengths. Validate the
    # immutable packet generation and reordered request identities first.
    cpu_proof = None
    if self._model_runtime is not None:
        cpu_proof = validated_cpu_seq_lens_proof(
            self._fl_async_input_staging,
            self.input_batch.req_ids,
            self.input_batch.prev_req_id_to_index,
            self.query_start_loc.cpu,
            seq_lens_cpu_upper_bound,
            is_prefilling,
            self.input_batch.num_prompt_tokens_cpu_tensor,
            num_reqs,
            num_reqs_padded,
            num_tokens,
            num_tokens_padded,
            for_cudagraph_capture,
            ubatch_slices,
        )
    set_current_cpu_seq_lens_proof(cpu_proof)
    cm_base = make_common_attention_metadata(
        query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
        query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
        seq_lens=self.seq_lens[:num_reqs_padded],
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        _num_computed_tokens_cache=(
            self.num_computed_tokens[:num_reqs_padded]
            if self.common_attention_metadata_graph is not None
            else None
        ),
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        cpu_seq_lens_proof=cpu_proof,
        replayssm_decode_base_cpu=replayssm_decode_base_cpu,
        num_reqs=num_reqs_padded,
        num_actual_tokens=num_tokens_padded,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_gid_0,
        slot_mapping=slot_mapping_gid_0,
        causal=True,
        is_prefilling=is_prefilling,
        positions=self.positions[:num_tokens_padded],
        mm_req_doc_ranges=req_doc_ranges,
        rswa_prefix_lens=rswa_prefix_lens,
    )
    # Only this metadata instance carries plugin state; native classes and
    # modules retain their official definitions. Shallow group copies retain it.
    cm_base._model_runtime = self._model_runtime
    if self.common_attention_metadata_graph is not None:
        self.common_attention_metadata_graph.attach_req_id_per_token(cm_base)
    if self.dcp_world_size > 1:
        self.dcp_local_seq_lens.cpu[:num_reqs] = _native.get_dcp_local_seq_lens(
            self.optimistic_seq_lens_cpu[:num_reqs],
            self.dcp_world_size,
            self.dcp_rank,
            self.parallel_config.cp_kv_cache_interleave_size,
        )
        self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
        self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)
        cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
        cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[:num_reqs_padded]
    if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
        cm_base.num_logits_indices = logits_indices.size(0)
        cm_base.max_logits_per_req = max_num_sampled_tokens
        cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
            logits_indices
        )
    cached_attn_metadata = {}

    def _build_attn_group_metadata(
        kv_cache_gid: int,
        attn_gid: int,
        common_attn_metadata: Any,
        ubid: int | None = None,
    ) -> None:
        attn_group = self.attn_groups[kv_cache_gid][attn_gid]
        builder = attn_group.get_metadata_builder(ubid or 0)
        kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
        if isinstance(kv_cache_spec, _native.UniformTypeKVCacheSpecs):
            kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
        cache_key = (kv_cache_spec, type(builder))
        cascade_attn_prefix_len = (
            cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
            if cascade_attn_prefix_lens
            else 0
        )
        extra_attn_metadata_args = {}
        if use_spec_decode and isinstance(
            builder,
            (
                _native.Mamba2AttentionMetadataBuilder,
                _native.GDNAttentionMetadataBuilder,
                _native.BailingLinearAttentionMetadataBuilder,
            ),
        ):
            assert ubid is None, "UBatching not supported with GDN or linear attn yet"
            extra_attn_metadata_args = dict(
                num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                    :num_reqs_padded
                ],
            )
            if (
                isinstance(builder, _native.Mamba2AttentionMetadataBuilder)
                and self.mamba_prev_last_scheduled_idx is not None
            ):
                extra_attn_metadata_args["prev_last_scheduled_idx"] = (
                    self.mamba_prev_last_scheduled_idx.gpu[:num_reqs_padded]
                )
        if for_cudagraph_capture:
            attn_metadata_i = builder.build_for_cudagraph_capture(common_attn_metadata)
        elif cache_key in cached_attn_metadata and builder.supports_update_block_table:
            attn_metadata_i = builder.update_block_table(
                cached_attn_metadata[cache_key],
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.slot_mapping,
            )
        else:
            attn_metadata_i = builder.build(
                common_prefix_len=cascade_attn_prefix_len,
                common_attn_metadata=common_attn_metadata,
                **extra_attn_metadata_args,
            )
            if builder.supports_update_block_table:
                cached_attn_metadata[cache_key] = attn_metadata_i
        if ubid is None:
            assert isinstance(attn_metadata, dict)
            attn_metadata_dict = attn_metadata
        else:
            assert isinstance(attn_metadata, list)
            attn_metadata_dict = attn_metadata[ubid]
        for layer_name in attn_group.layer_names:
            attn_metadata_dict[layer_name] = attn_metadata_i

    spec_decode_common_attn_metadata = None
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
        cm = _native.copy(cm_base)
        cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
            num_scheduled_tokens or {},
            kv_cache_group.kv_cache_spec,
            num_reqs_padded,
            for_cudagraph_capture=for_cudagraph_capture,
        )
        if kv_cache_gid > 0:
            cm.block_table_tensor = _get_block_table(kv_cache_gid)
            cm.slot_mapping = slot_mappings[kv_cache_gid]
        if self.speculative_config and spec_decode_common_attn_metadata is None:
            if isinstance(
                self.drafter,
                (
                    _native.EagleProposer,
                    _native.DFlashProposer,
                    _native.Gemma4Proposer,
                    _native.ExtractHiddenStatesProposer,
                ),
            ):
                if self.drafter.kv_cache_gid == kv_cache_gid:
                    spec_decode_common_attn_metadata = cm
            else:
                spec_decode_common_attn_metadata = cm
        if self.speculative_config and isinstance(
            self.drafter, _native.Step3p5MTPProposer
        ):
            self.drafter.set_per_group_attn_metadata(
                kv_cache_gid, cm.block_table_tensor, cm.slot_mapping
            )
        elif self.speculative_config and isinstance(
            self.drafter, _native.Gemma4Proposer
        ):
            self.drafter.set_per_group_block_table(kv_cache_gid, cm.block_table_tensor)
        for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
            if ubatch_slices is not None:
                for ubid, _cm in enumerate(
                    _native.split_attn_metadata(ubatch_slices, cm)
                ):
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)
            else:
                _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)
    if spec_decode_common_attn_metadata is not None and (
        num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
    ):
        spec_decode_common_attn_metadata = spec_decode_common_attn_metadata.unpadded(
            num_tokens, num_reqs
        )
    return (attn_metadata, spec_decode_common_attn_metadata)


def runner_bookkeeping_sync(
    self,
    scheduler_output,
    sampler_output,
    logits,
    hidden_states,
    num_scheduled_tokens,
    prompt_logprobs_hidden_states=None,
):
    _native = self._hy4_runtime_module
    num_nans = None
    num_nans_in_logits = {}
    if _native.envs.VLLM_COMPUTE_NANS_IN_LOGITS:
        if self.use_async_scheduling:
            num_nans = None if logits is None else _native.count_nans_per_row(logits)
        else:
            num_nans_in_logits = self._get_nans_in_logits(logits)
    num_reqs = self.input_batch.num_reqs
    discard_sampled_tokens_req_indices = _native.np.nonzero(
        self.discard_request_mask.np[:num_reqs]
    )[0]
    for i in discard_sampled_tokens_req_indices:
        gen = self.input_batch.generators.get(int(i))
        if gen is not None:
            gen.set_offset(gen.get_offset() - 4)
    req_ids_output_copy = self.input_batch.req_ids.copy()
    req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()
    num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
    sampled_token_ids = sampler_output.sampled_token_ids
    logprobs_tensors = sampler_output.logprobs_tensors
    invalid_req_indices = []
    logprobs_lists = None
    if not self.use_async_scheduling:
        if self.routed_experts_initialized:
            buf = self.routed_experts_capturer.get_device_buffer()
            total = scheduler_output.total_num_scheduled_tokens
            self.routed_experts_cpu[:total].copy_(buf[:total], non_blocking=True)
            self.routed_experts_slot_mapping_cpu[:total].copy_(
                self.routed_experts_slot_mapping_device[:total], non_blocking=True
            )
        with _native.gpu_sync_allowed():
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()
                if logprobs_tensors is not None:
                    logprobs_lists = logprobs_tensors.tolists()
            else:
                valid_sampled_token_ids, logprobs_lists = (
                    _native.RejectionSampler.parse_output(
                        sampled_token_ids,
                        self.input_batch.vocab_size,
                        discard_sampled_tokens_req_indices,
                        logprobs_tensors=logprobs_tensors,
                    )
                )
    else:
        valid_sampled_token_ids = []
        invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
        invalid_req_indices_set = set(invalid_req_indices)
        if self.input_batch.prev_sampled_token_ids is None:
            assert sampled_token_ids.shape[-1] == 1
            self.input_batch.prev_sampled_token_ids = sampled_token_ids
        self.input_batch.prev_req_id_to_index = {
            req_id: i
            for i, req_id in enumerate(self.input_batch.req_ids)
            if i not in invalid_req_indices_set
        }
    req_ids = self.input_batch.req_ids
    for req_idx in range(num_sampled_tokens):
        if self.use_async_scheduling:
            sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
        else:
            sampled_ids = valid_sampled_token_ids[req_idx]
        num_sampled_ids = len(sampled_ids) if sampled_ids else 0
        if not sampled_ids:
            continue
        start_idx = self.input_batch.num_tokens_no_spec[req_idx]
        end_idx = start_idx + num_sampled_ids
        assert end_idx <= self.max_model_len, (
            f"Sampled token IDs exceed the max model length. Total number of tokens: {end_idx} > max_model_len: {self.max_model_len}"
        )
        self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
        self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
        self.input_batch.num_tokens_no_spec[req_idx] = end_idx
        req_id = req_ids[req_idx]
        req_state = self.requests[req_id]
        req_state.output_token_ids.extend(sampled_ids)
    prompt_logprobs_dict = self._get_prompt_logprobs_dict(
        hidden_states[:num_scheduled_tokens],
        scheduler_output.num_scheduled_tokens,
        prompt_logprobs_hidden_states,
    )
    return (
        num_nans_in_logits,
        num_nans,
        logprobs_lists,
        valid_sampled_token_ids,
        prompt_logprobs_dict,
        req_ids_output_copy,
        req_id_to_index_output_copy,
        invalid_req_indices,
    )


def runner_prepare_decode_mrope(self, scheduler_output):
    _native = self._hy4_runtime_module
    self._mrope_decode_active = False
    state = self._mrope_decode_state
    if state is None:
        return False
    deltas = None
    if (
        self.dcp_world_size == 1
        and self.parallel_config.prefill_context_parallel_size == 1
        and (not self.parallel_config.use_ubatching)
    ):
        deltas = self._model_runtime.select_decode_deltas(
            self.input_batch.req_ids,
            self.requests,
            scheduler_output.num_scheduled_tokens,
            self.input_batch.num_computed_tokens_cpu,
            _native.length_from_prompt_token_ids_or_embeds,
            uses_mrope=self.uses_mrope,
            uses_xdrope_dim=self.uses_xdrope_dim,
            multimodal_pruning=self.is_multimodal_pruning_enabled,
            scheduled_encoder_inputs=scheduler_output.scheduled_encoder_inputs,
        )
    state.np.fill(0)
    if deltas is not None:
        state.np[0] = 1
        state.np[1 : len(deltas) + 1] = deltas
        self._mrope_decode_active = True
    state.copy_to_gpu()
    return self._mrope_decode_active


def runner_run_common_attention_metadata(
    self, num_reqs, cudagraph_mode, *, capture=False
):
    _native = self._hy4_runtime_module
    if self.common_attention_metadata_graph is None:
        return False
    num_reqs = self.max_num_reqs
    use_graph = cudagraph_mode != _native.CUDAGraphMode.NONE and (
        not self.parallel_config.use_ubatching
    )
    return self.common_attention_metadata_graph.run(
        self.input_batch.block_table,
        num_reqs,
        self.query_start_loc.gpu[: num_reqs + 1],
        self.positions,
        self.seq_lens[:num_reqs],
        self.num_computed_tokens[:num_reqs],
        use_graph=use_graph,
        capture=capture,
        extra_tensors=(
            (self._mrope_decode_state.gpu, self.mrope_positions.gpu)
            if self._mrope_decode_state is not None
            else ()
        ),
    )


def runner_get_slot_mappings(
    self,
    num_tokens_padded,
    num_reqs_padded,
    num_tokens_unpadded,
    ubatch_slices=None,
    slot_mapping_is_current=False,
):
    """
    Build slot mappings in both formats needed by the system.

    Args:
        num_tokens_padded: Total number of tokens (padded)
        num_reqs_padded: Total number of requests (padded)
        num_tokens_unpadded: Actual number of tokens (unpadded)
        ubatch_slices: Optional ubatch slicing info for DBO

    Returns:
        A tuple of:
        - slot_mappings_by_gid: dict[int, torch.Tensor] for attention metadata
        - slot_mappings_by_layer: dict[str, torch.Tensor] or list for ForwardContext
    """
    _native = self._hy4_runtime_module
    if not (
        hasattr(self, "kv_cache_config")
        and self.kv_cache_config is not None
        and (len(self.kv_cache_config.kv_cache_groups) > 0)
    ):
        return (None, None)

    def _get_slot_mapping(kv_cache_gid: int):
        assert num_reqs_padded is not None and num_tokens_padded is not None
        kv_cache_spec = self.kv_cache_config.kv_cache_groups[kv_cache_gid].kv_cache_spec
        if isinstance(kv_cache_spec, _native.EncoderOnlyAttentionSpec):
            slot_mapping = _native.torch.zeros(
                (num_tokens_padded,), dtype=_native.torch.int64, device=self.device
            )
        else:
            blk_table = self.input_batch.block_table[kv_cache_gid]
            slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]
        if not slot_mapping_is_current or isinstance(
            kv_cache_spec, _native.EncoderOnlyAttentionSpec
        ):
            slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)
        return slot_mapping

    slot_mappings_by_gid = {
        gid: _get_slot_mapping(gid)
        for gid, _ in enumerate(self.kv_cache_config.kv_cache_groups)
    }
    slot_mappings_by_layer = {}
    for gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
        slot_mapping = slot_mappings_by_gid[gid]
        for layer_name in kv_cache_group.layer_names:
            slot_mappings_by_layer[layer_name] = slot_mapping
    if ubatch_slices is not None:
        result = []
        for ubatch in ubatch_slices:
            sliced_mappings = {}
            for layer_name, slot_mapping in slot_mappings_by_layer.items():
                sliced_mappings[layer_name] = slot_mapping[ubatch.token_slice]
            result.append(sliced_mappings)
        return (slot_mappings_by_gid, result)
    return (slot_mappings_by_gid, slot_mappings_by_layer)


@torch.inference_mode()
def runnerexecute_model(self, scheduler_output, intermediate_tensors=None):
    _native = self._hy4_runtime_module
    if self.execute_model_state is not None:
        raise RuntimeError(
            "State error: sample_tokens() must be called after execute_model() returns None."
        )
    if self.speculative_config is not None and self.speculative_config.use_ngram_gpu():
        num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
        spec_decode_tokens_copy = scheduler_output.scheduled_spec_decode_tokens.copy()
        scheduler_output = _native.replace(
            scheduler_output,
            num_scheduled_tokens=num_scheduled_tokens_copy,
            scheduled_spec_decode_tokens=spec_decode_tokens_copy,
        )
    if _native.has_kv_transfer_group():
        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        _native.get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    with (
        _native.record_function_or_nullcontext("gpu_model_runner: preprocess"),
        self.synchronize_input_prep(),
    ):
        deferred_state_corrections_fn = self._update_states(scheduler_output)
        if _native.has_ec_transfer() and (not _native.get_ec_transfer().is_consumer):
            with self.maybe_get_ec_connector_output(
                scheduler_output, encoder_cache=self.encoder_cache
            ) as ec_connector_output:
                self._execute_mm_encoder(scheduler_output)
                return _native.make_empty_encoder_model_runner_output(scheduler_output)
        if not num_scheduled_tokens:
            if (
                self.parallel_config.distributed_executor_backend == "external_launcher"
                and self.parallel_config.data_parallel_size > 1
            ):
                self._dummy_run(1)
            if not _native.has_kv_transfer_group():
                return _native.EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
        if self.cache_config.kv_sharing_fast_prefill:
            assert not self.num_prompt_logprobs, (
                "--kv-sharing-fast-prefill produces incorrect logprobs for prompt tokens, tokens, please disable it when the requests need prompt logprobs"
            )
        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens_np = _native.np.array(tokens, dtype=_native.np.int32)
        max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
        num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
        logits_indices, spec_decode_metadata, max_num_sampled_tokens = (
            self._prepare_inputs(scheduler_output, num_scheduled_tokens_np)
        )
        cascade_attn_prefix_lens = None
        if self.cascade_attn_enabled and (not self.parallel_config.use_ubatching):
            cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                num_scheduled_tokens_np,
                self.input_batch.num_computed_tokens_cpu[:num_reqs],
                scheduler_output.num_common_prefix_blocks,
            )
        (
            cudagraph_mode,
            batch_desc,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        ) = self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            use_cascade_attn=cascade_attn_prefix_lens is not None,
            num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            allow_microbatching=self._allow_microbatching(
                num_reqs, num_scheduled_tokens_np
            ),
        )
        _native.logger.debug(
            "Running batch with cudagraph_mode: %s, batch_descriptor: %s, should_ubatch: %s, num_tokens_across_dp: %s",
            cudagraph_mode,
            batch_desc,
            should_ubatch,
            num_tokens_across_dp,
        )
        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        ubatch_slices, ubatch_slices_padded = _native.maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens_np,
            num_tokens_padded,
            num_reqs_padded,
            self.parallel_config.num_ubatches,
        )
        _native.logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )
        has_separate_kv_update = not all(
            (
                all(
                    (
                        g.backend.forward_includes_kv_cache_update
                        for g in self.attn_groups[id]
                    )
                )
                for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
                if not isinstance(spec.kv_cache_spec, _native.EncoderOnlyAttentionSpec)
            )
        )
        pad_attn = cudagraph_mode == _native.CUDAGraphMode.FULL
        if self.cache_config.mamba_cache_mode == "align":
            if deferred_state_corrections_fn:
                deferred_state_corrections_fn()
                deferred_state_corrections_fn = None
            mamba_bufs = self._get_mamba_bufs()
            _native.mamba_utils.preprocess_mamba(
                scheduler_output,
                self.kv_cache_config,
                self.cache_config,
                self.mamba_state_idx,
                self.input_batch,
                self.requests,
                self.compilation_config.static_forward_context,
                self.model.get_mamba_state_copy_func(),
                mamba_bufs.preprocess,
                align_ctx=mamba_bufs.postprocess_align,
            )
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )
            self.num_accepted_tokens.copy_to_gpu(num_reqs)
            if mamba_bufs.postprocess_align is not None:
                _native.mamba_utils.stage_postprocess_inputs_to_gpu(
                    mamba_bufs.postprocess_align,
                    scheduler_output,
                    self.input_batch.req_ids,
                    num_reqs,
                    self.requests,
                    self.mamba_state_idx,
                )
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices
        self._run_common_attention_metadata(num_reqs_padded, cudagraph_mode)
        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=(
                num_tokens_padded
                if pad_attn or has_separate_kv_update
                else num_tokens_unpadded
            ),
            num_reqs_padded=(
                num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
            ),
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices_padded,
            slot_mapping_is_current=self.common_attention_metadata_graph is not None,
        )
        attn_metadata, spec_decode_common_attn_metadata = (
            self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded if pad_attn else None,
                max_query_len=max_num_scheduled_tokens,
                ubatch_slices=ubatch_slices_attn,
                logits_indices=logits_indices,
                max_num_sampled_tokens=max_num_sampled_tokens,
                use_spec_decode=use_spec_decode,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                slot_mappings=slot_mappings_by_group,
                block_table_rows_are_current=self.common_attention_metadata_graph
                is not None,
            )
        )
        (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        ) = self._preprocess(scheduler_output, num_tokens_padded, intermediate_tensors)
    num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
    has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0
    defer_kv_connector_finalize = self.speculative_config is not None
    if self.eplb_state is not None:
        self.eplb_state.prepare_forward(
            self.model_config, num_tokens_unpadded, ubatch_slices_padded
        )
    with (
        _native.set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_mode,
            batch_descriptor=batch_desc,
            ubatch_slices=ubatch_slices_padded,
            slot_mapping=slot_mappings,
            skip_compiled=has_encoder_input,
        ),
        _native.record_function_or_nullcontext("gpu_model_runner: forward"),
        self.maybe_get_kv_connector_output(
            scheduler_output, defer_finalize=defer_kv_connector_finalize
        ) as kv_connector_output,
    ):
        model_output = self._model_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
    with _native.record_function_or_nullcontext("gpu_model_runner: postprocess"):
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None
        prompt_logprobs_hidden_states = self._retain_prompt_logprobs_hidden_states(
            hidden_states, scheduler_output.num_scheduled_tokens
        )
        if not self.broadcast_pp_output:
            if not _native.get_pp_group().is_last_rank:
                assert isinstance(hidden_states, _native.IntermediateTensors)
                self.kv_connector_output = kv_connector_output
                return hidden_states
            if self.is_pooling_model:
                return self._pool(
                    hidden_states,
                    num_scheduled_tokens,
                    num_scheduled_tokens_np,
                    kv_connector_output,
                )
            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)
        else:
            assert not self.is_pooling_model
            sample_hidden_states = hidden_states[logits_indices]
            if not _native.get_pp_group().is_last_rank:
                all_gather_tensors = {
                    "residual": not _native.is_residual_scattered_for_sp(
                        self.vllm_config, num_tokens_padded
                    )
                }
                _native.get_pp_group().send_tensor_dict(
                    hidden_states.tensors,
                    all_gather_group=_native.get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
                logits = None
            else:
                logits = self.model.compute_logits(sample_hidden_states)
            model_output_broadcast_data = {}
            if logits is not None:
                model_output_broadcast_data["logits"] = logits.contiguous()
            broadcasted = _native.get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(_native.get_pp_group().ranks) - 1
            )
            assert broadcasted is not None
            logits = broadcasted["logits"]
    self.execute_model_state = Hy4ExecuteModelState(
        scheduler_output,
        logits,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        hidden_states,
        prompt_logprobs_hidden_states,
        sample_hidden_states,
        aux_hidden_states,
        ec_connector_output,
        cudagraph_stats,
        slot_mappings,
    )
    self.kv_connector_output = kv_connector_output
    if deferred_state_corrections_fn:
        deferred_state_corrections_fn()
    return None


@torch.inference_mode
def runnersample_tokens(self, grammar_output):
    _native = self._hy4_runtime_module
    if self.execute_model_state is None:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None
        if self.use_async_scheduling and (not _native.get_pp_group().is_last_rank):
            self._pp_receive_prev_sampled_token_ids_to_input_batch()
        return _native.ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
    (
        scheduler_output,
        logits,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        hidden_states,
        prompt_logprobs_hidden_states,
        sample_hidden_states,
        aux_hidden_states,
        ec_connector_output,
        cudagraph_stats,
        slot_mappings,
    ) = self.execute_model_state
    self.execute_model_state = None
    if grammar_output is not None:
        _native.apply_grammar_bitmask(
            scheduler_output, grammar_output, self.input_batch, logits
        )
    with _native.record_function_or_nullcontext("gpu_model_runner: sample"):
        sampler_output = self._sample(logits, spec_decode_metadata)
    self._update_states_after_model_execute(
        sampler_output.sampled_token_ids, scheduler_output
    )
    if self.use_async_scheduling:
        pp = _native.get_pp_group()
        if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
            self._pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids)
    self._draft_token_ids = None
    self._draft_probs = None
    self._draft_prob_req_ids = None
    self._draft_token_req_ids = None
    self.valid_sampled_token_count_gpu = None
    self.input_batch.prev_sampled_token_ids = None

    def propose_draft_token_ids(sampled_token_ids):
        assert spec_decode_common_attn_metadata is not None
        with _native.record_function_or_nullcontext("gpu_model_runner: draft"):
            self._draft_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                slot_mappings,
            )
            self._copy_draft_token_ids_to_cpu(scheduler_output)

    spec_config = self.speculative_config
    draft_after_bookkeeping = False
    if spec_config is not None:
        input_fits_in_drafter = self._input_fits_in_drafter(
            spec_decode_common_attn_metadata
        )
        drafter_runs_model_forward = (
            spec_config.use_eagle()
            or spec_config.uses_draft_model()
            or spec_config.uses_extract_hidden_states()
        )
        use_gpu_toks = drafter_runs_model_forward and (
            not spec_config.disable_padded_drafter_batch
        )
        if use_gpu_toks:
            assert isinstance(
                self.drafter,
                _native.EagleProposer
                | _native.DFlashProposer
                | _native.DraftModelProposer
                | _native.ExtractHiddenStatesProposer
                | _native.Gemma4Proposer,
            )
            sampled_token_ids = sampler_output.sampled_token_ids
            if input_fits_in_drafter:
                propose_draft_token_ids(sampled_token_ids)
            else:
                if self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count = (
                        self.drafter.prepare_next_token_ids_padded(
                            sampled_token_ids,
                            self.requests,
                            self.input_batch,
                            self.discard_request_mask.gpu,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count
                    )
                if self.parallel_config.data_parallel_size > 1:
                    self.drafter.dummy_run(num_tokens=1)
        elif spec_config.use_ngram_gpu() and (
            not spec_config.disable_padded_drafter_batch
        ):
            assert isinstance(self.drafter, _native.NgramProposerGPU)
            sampled_token_ids = sampler_output.sampled_token_ids
            if input_fits_in_drafter:
                propose_draft_token_ids(sampled_token_ids)
            elif self.valid_sampled_token_count_event is not None:
                assert spec_decode_common_attn_metadata is not None
                next_token_ids, valid_sampled_tokens_count, _ = (
                    self.drafter.update_token_ids_ngram(
                        sampled_token_ids,
                        self.input_batch,
                        self.token_ids_gpu_tensor,
                        self.num_tokens_no_spec_gpu,
                        self.discard_request_mask.gpu,
                    )
                )
                self._copy_valid_sampled_token_count(
                    next_token_ids, valid_sampled_tokens_count
                )
        else:
            draft_after_bookkeeping = True
        if not input_fits_in_drafter:
            self._draft_token_ids = _native.torch.zeros(
                1, device=self.device, dtype=_native.torch.int32
            ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
            self._draft_probs = None
            self._draft_prob_req_ids = None
            self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
    with _native.record_function_or_nullcontext("gpu_model_runner: bookkeep"):
        (
            num_nans_in_logits,
            num_nans_device,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            prompt_logprobs_hidden_states=prompt_logprobs_hidden_states,
        )
    if draft_after_bookkeeping:
        if input_fits_in_drafter:
            propose_draft_token_ids(valid_sampled_token_ids)
        elif drafter_runs_model_forward and self.parallel_config.data_parallel_size > 1:
            assert isinstance(
                self.drafter,
                _native.EagleProposer
                | _native.DFlashProposer
                | _native.DraftModelProposer
                | _native.ExtractHiddenStatesProposer
                | _native.Gemma4Proposer,
            )
            self.drafter.dummy_run(num_tokens=1)
    if spec_config is not None:
        self.finalize_kv_connector()
    with _native.record_function_or_nullcontext("gpu_model_runner: eplb"):
        self.eplb_step()
    kv_connector_output = self.kv_connector_output
    self.kv_connector_output = None
    with _native.record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
        output = _native.ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            kv_connector_output=kv_connector_output,
            ec_connector_output=(
                ec_connector_output if self.supports_mm_inputs else None
            ),
            num_nans_in_logits=num_nans_in_logits,
            cudagraph_stats=cudagraph_stats,
            routed_experts=None,
        )
    if not self.use_async_scheduling:
        if self.routed_experts_initialized:
            total = scheduler_output.total_num_scheduled_tokens
            output.routed_experts = _native.RoutedExpertsLists(
                routing_data=self.routed_experts_cpu[:total].numpy(),
                slot_mapping=self.routed_experts_slot_mapping_cpu[:total].numpy(),
            )
        return output
    with _native.record_function_or_nullcontext(
        "gpu_model_runner: AsyncGPUModelRunnerOutput"
    ):
        routed_experts_snapshot = self.get_routed_experts(
            scheduler_output.total_num_scheduled_tokens
        )
        async_output = _native.AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
            vocab_size=self.input_batch.vocab_size,
            routed_experts=routed_experts_snapshot,
            check_ep_fault=self.check_ep_fault,
            num_nans=num_nans_device,
        )
    with _native.record_function_or_nullcontext(
        "gpu_model_runner: set_async_sampled_token_ids"
    ):
        self.input_batch.set_async_sampled_token_ids(
            async_output.sampled_token_ids_cpu, async_output.async_copy_ready_event
        )
    return async_output


@instrument(span_name="Loading (GPU)")
def runnerload_model(self, load_dummy_weights=False):
    """
    Args:
        load_dummy_weights: load dummy weights instead of real weights.
    """
    _native = self._hy4_runtime_module
    _native.logger.info_once(
        "Starting to load model %s...", self.model_config.model, scope="global"
    )
    if self.parallel_config.enable_eplb:
        self.eplb_state = _native.EplbState(self.parallel_config, self.device)
        eplb_models = 0
    try:
        with _native.DeviceMemoryProfiler() as m:
            time_before_load = _native.time.perf_counter()
            if load_dummy_weights:
                self.load_config.load_format = "dummy"
            model_loader = _native.get_model_loader(self.load_config)
            self.model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.model_config
            )
            if self.lora_config:
                self.model = self.load_lora_model(
                    self.model, self.vllm_config, self.device
                )
            _native.logger.info_once("Loading drafter model...")
            self.drafter.load_model(self.model)
            if self.parallel_config.enable_eplb and (
                (
                    drafter_moe_model := _native.get_mixture_of_experts_model(
                        self.drafter.model
                    )
                )
                is not None
            ):
                assert not self.parallel_config.enable_elastic_ep, (
                    "Elastic EP is not supported with drafter model."
                )
                spec_config = self.vllm_config.speculative_config
                assert spec_config is not None
                assert spec_config.draft_model_config is not None
                _native.logger.info_once(
                    "EPLB is enabled for MoE part of drafter model %s.",
                    spec_config.draft_model_config.model,
                )
                if self.eplb_state is None:
                    self.eplb_state = _native.EplbState(
                        self.parallel_config, self.device
                    )
                self.eplb_state.add_model(
                    drafter_moe_model, spec_config.draft_model_config
                )
                self.drafter.set_eplb_state(self.eplb_state)
                eplb_models += 1
            self._setup_eagle3_aux_hidden_state_outputs()
            self._moe_model = _native.get_mixture_of_experts_model(self.model)
            if (
                self.parallel_config.enable_eplb
                and (not load_dummy_weights)
                and (self._moe_model is not None)
            ):
                _native.logger.info_once(
                    "EPLB is enabled for MoE part of model %s.", self.model_config.model
                )
                assert self.eplb_state is not None
                self.eplb_state.add_model(self._moe_model, self.model_config)
                eplb_models += 1
            time_after_load = _native.time.perf_counter()
        self.model_memory_usage = m.consumed_memory
    except _native.torch.cuda.OutOfMemoryError as e:
        msg = "Failed to load model - not enough GPU memory. Try lowering --gpu-memory-utilization to free memory for weights, increasing --tensor-parallel-size, or using --quantization. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more tips."
        combined_msg = f"{msg} (original error: {e})"
        _native.logger.error(combined_msg)
        raise e
    _native.logger.info_once(
        "Model loading took %s GiB memory and %.6f seconds",
        _native.format_gib(self.model_memory_usage),
        time_after_load - time_before_load,
    )
    mm_config = self.model_config.multimodal_config
    self.is_multimodal_pruning_enabled = (
        _native.supports_multimodal_pruning(self.get_model())
        and mm_config is not None
        and mm_config.is_multimodal_pruning_enabled()
    )
    self.requires_sequential_video_encoding = hasattr(
        self.get_model(), "requires_sequential_video_encoding"
    )
    if (
        self.parallel_config.enable_eplb
        and self._moe_model is not None
        and (not load_dummy_weights)
        and (self.eplb_state is not None)
        and self.eplb_state.is_async
    ):
        self.eplb_state.start_async_loop()
    if (
        self.vllm_config.compilation_config.mode
        == _native.CompilationMode.STOCK_TORCH_COMPILE
    ):
        from vllm.env_override import _apply_constrain_to_fx_strides_patch

        _apply_constrain_to_fx_strides_patch()
        backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
        _native.compilation_counter.stock_torch_compile_count += 1
        self.model.compile(fullgraph=True, backend=backend)
        return
    cudagraph_mode = self.compilation_config.cudagraph_mode
    assert cudagraph_mode is not None
    if (
        _native.is_breakable_cudagraph_enabled()
        and cudagraph_mode != _native.CUDAGraphMode.NONE
        and (not self.parallel_config.use_ubatching)
    ):
        if self._model_runtime is not None and self._model_runtime.mixed_graphs:
            self.model = self._model_runtime.wrap_breakable(
                self.model, self.vllm_config
            )
        else:
            self.model = _native.BreakableCUDAGraphWrapper(self.model, self.vllm_config)
        drafter = getattr(self, "drafter", None)
        if drafter is not None and hasattr(drafter, "model"):
            if self._model_runtime is not None and self._model_runtime.mixed_graphs:
                drafter.model = self._model_runtime.wrap_breakable(
                    drafter.model, self.vllm_config, draft=True
                )
            else:
                drafter.model = _native.BreakableCUDAGraphWrapper(
                    drafter.model, self.vllm_config
                )
    elif cudagraph_mode.has_full_cudagraphs() and (
        not self.parallel_config.use_ubatching
    ):
        self.model = _native.CUDAGraphWrapper(
            self.model, self.vllm_config, runtime_mode=_native.CUDAGraphMode.FULL
        )
    elif self.parallel_config.use_ubatching:
        if cudagraph_mode.has_full_cudagraphs():
            self.model = _native.UBatchWrapper(
                self.model, self.vllm_config, _native.CUDAGraphMode.FULL, self.device
            )
        else:
            self.model = _native.UBatchWrapper(
                self.model, self.vllm_config, _native.CUDAGraphMode.NONE, self.device
            )
    _native.get_offloader().post_init()


def runner_retain_prompt_logprobs_hidden_states(
    self, hidden_states, num_scheduled_tokens
):
    _native = self._hy4_runtime_module
    if (
        self._model_runtime is None
        or not self.num_prompt_logprobs
        or self.is_pooling_model
        or (not _native.get_pp_group().is_last_rank)
    ):
        return None
    retained = {}
    for req_id in self.num_prompt_logprobs:
        num_tokens = num_scheduled_tokens.get(req_id)
        if num_tokens is None:
            continue
        request = self.requests[req_id]
        if request.prompt_token_ids is None:
            continue
        num_logits = min(
            num_tokens, len(request.prompt_token_ids) - request.num_computed_tokens - 1
        )
        if num_logits <= 0:
            continue
        req_idx = self.input_batch.req_id_to_index[req_id]
        offset = int(self.query_start_loc.np[req_idx])
        retained[req_id] = hidden_states[offset : offset + num_logits].clone()
    return retained


def runner_get_prompt_logprobs_dict(
    self, hidden_states, num_scheduled_tokens, retained_prompt_hidden_states=None
):
    _native = self._hy4_runtime_module
    num_prompt_logprobs_dict = self.num_prompt_logprobs
    if not num_prompt_logprobs_dict:
        return {}
    prompt_logprobs_dict = {}
    completed_prefill_reqs = []
    for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
        num_tokens = num_scheduled_tokens.get(req_id)
        if num_tokens is None:
            continue
        request = self.requests[req_id]
        if request.prompt_token_ids is None:
            continue
        num_prompt_tokens = len(request.prompt_token_ids)
        prompt_token_ids = _native.async_tensor_h2d(
            request.prompt_token_ids, device=self.device
        )
        logprobs_tensors = request.in_progress_prompt_logprobs_cpu
        if logprobs_tensors is None:
            logprobs_tensors = _native.LogprobsTensors.empty_cpu(
                num_prompt_tokens - 1, num_prompt_logprobs + 1
            )
            request.in_progress_prompt_logprobs_cpu = logprobs_tensors
        start_idx = request.num_computed_tokens
        start_tok = start_idx + 1
        num_remaining_tokens = num_prompt_tokens - start_tok
        if num_tokens <= num_remaining_tokens:
            num_logits = num_tokens
        else:
            num_logits = num_remaining_tokens
            completed_prefill_reqs.append(req_id)
            prompt_logprobs_dict[req_id] = logprobs_tensors
        if num_logits <= 0:
            continue
        if retained_prompt_hidden_states is None:
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
        else:
            prompt_hidden_states = retained_prompt_hidden_states[req_id]
        logits = self.model.compute_logits(prompt_hidden_states)
        tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]
        if self.model_config.logprobs_mode in ("raw_logits", "processed_logits"):
            scores = logits.to(_native.torch.float32)
        else:
            scores = self.sampler.compute_logprobs(logits)
        token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
            scores, num_prompt_logprobs, tgt_token_ids
        )
        chunk_slice = slice(start_idx, start_idx + num_logits)
        logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
            token_ids, non_blocking=True
        )
        logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
        logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
            ranks, non_blocking=True
        )
    for req_id in completed_prefill_reqs:
        del num_prompt_logprobs_dict[req_id]
        self.requests[req_id].in_progress_prompt_logprobs_cpu = None
    if prompt_logprobs_dict:
        self._sync_device()
    return prompt_logprobs_dict


@torch.inference_mode()
def runner_dummy_run(
    self,
    num_tokens,
    cudagraph_runtime_mode=None,
    force_attention=False,
    uniform_decode=False,
    allow_microbatching=True,
    skip_eplb=False,
    is_profile=False,
    create_mixed_batch=False,
    remove_lora=True,
    is_graph_capturing=False,
    num_active_loras=0,
    profile_seq_lens=None,
    randomize_inputs=False,
):
    """
    Run a dummy forward pass to warm up/profile run or capture the
    CUDA graph for the model.

    Args:
        num_tokens: Number of tokens to run the dummy forward pass.
        cudagraph_runtime_mode: used to control the behavior.
            - if not set will determine the cudagraph mode based on using
                the self.cudagraph_dispatcher.
            - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
            - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
            - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                needed.
        force_attention: If True, always create attention metadata. Used to
            warm up attention backend when mode is NONE.
        uniform_decode: If True, the batch is a uniform decode batch.
        skip_eplb: If True, skip EPLB state update.
        is_profile: If True, this is a profile run.
        create_mixed_batch: If True, create a mixed batch with both decode
            (1 token) and prefill (multiple tokens) requests.
        remove_lora: If False, dummy LoRAs are not destroyed after the run
        num_active_loras: Number of distinct active LoRAs to capture for.
            LoRA is activated when num_active_loras > 0.
        profile_seq_lens: If provided, use this value for seq_lens instead
            of max_query_len. Used to profile attention workspace that
            scales with context length.
    """
    _native = self._hy4_runtime_module
    mm_config = self.vllm_config.model_config.multimodal_config
    if mm_config and mm_config.mm_encoder_only:
        return (_native.torch.tensor([]), _native.torch.tensor([]))
    assert (
        cudagraph_runtime_mode is None or cudagraph_runtime_mode.is_valid_runtime_mode()
    )
    max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens
    assert num_tokens <= self.max_num_tokens
    max_num_reqs = self.scheduler_config.max_num_seqs
    if create_mixed_batch:
        assert not uniform_decode
        num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
        num_prefill_tokens = num_tokens - num_decode_tokens
        num_reqs = num_decode_tokens + 1
        num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
        max_query_len = num_prefill_tokens
    elif uniform_decode:
        assert not create_mixed_batch
        num_reqs = min(max_num_reqs, _native.cdiv(num_tokens, max_query_len))
        num_scheduled_tokens_list = [max_query_len] * num_reqs
        if num_tokens % max_query_len != 0:
            num_scheduled_tokens_list[-1] = num_tokens % max_query_len
    else:
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
    assert sum(num_scheduled_tokens_list) == num_tokens
    assert len(num_scheduled_tokens_list) == num_reqs
    num_scheduled_tokens = _native.np.array(
        num_scheduled_tokens_list, dtype=_native.np.int32
    )
    num_tokens_unpadded = int(num_scheduled_tokens.sum())
    num_sampled_tokens = _native.np.ones(num_reqs, dtype=_native.np.int32)
    _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
        self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
            allow_microbatching=allow_microbatching,
            force_eager=is_profile
            or cudagraph_runtime_mode == _native.CUDAGraphMode.NONE,
            force_uniform_decode=uniform_decode,
            force_has_lora=num_active_loras > 0,
            force_num_active_loras=num_active_loras,
        )
    )
    if cudagraph_runtime_mode is None:
        cudagraph_runtime_mode = _cudagraph_mode
    else:
        assert cudagraph_runtime_mode == _cudagraph_mode, (
            f"Cudagraph runtime mode mismatch in dummy_run. Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
        )
    num_tokens_padded = batch_desc.num_tokens
    num_reqs_padded = (
        batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
    )
    dcp_dummy_context_len = _native.get_dcp_dummy_context_len(
        self.dcp_world_size,
        self.parallel_config.cp_kv_cache_interleave_size,
        hasattr(self, "kv_cache_config"),
        create_mixed_batch,
        is_graph_capturing,
        uniform_decode,
    )
    ubatch_slices, ubatch_slices_padded = _native.maybe_create_ubatch_slices(
        should_ubatch,
        num_scheduled_tokens,
        num_tokens_padded,
        num_reqs_padded,
        self.vllm_config.parallel_config.num_ubatches,
    )
    _native.logger.debug(
        "ubatch_slices: %s, ubatch_slices_padded: %s",
        ubatch_slices,
        ubatch_slices_padded,
    )
    attn_metadata = None
    slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
        num_tokens_padded=num_tokens_padded,
        num_reqs_padded=num_reqs_padded,
        num_tokens_unpadded=num_tokens_unpadded,
        ubatch_slices=ubatch_slices_padded,
        slot_mapping_is_current=self.common_attention_metadata_graph is not None,
    )
    with self.synchronize_input_prep():
        build_attention = (
            force_attention or cudagraph_runtime_mode == _native.CUDAGraphMode.FULL
        )
        if build_attention or self.common_attention_metadata_graph is not None:
            if profile_seq_lens is not None:
                seq_lens = profile_seq_lens
            elif create_mixed_batch:
                if dcp_dummy_context_len > 0:
                    seq_lens = _native.torch.tensor(
                        [1 + dcp_dummy_context_len] * num_decode_tokens
                        + [num_prefill_tokens + dcp_dummy_context_len],
                        dtype=_native.torch.int,
                    )
                else:
                    seq_lens = _native.torch.tensor(
                        [1] * num_decode_tokens + [num_prefill_tokens + 1],
                        dtype=_native.torch.int,
                    )
            elif dcp_dummy_context_len > 0:
                seq_lens = max_query_len + dcp_dummy_context_len
            else:
                seq_lens = max_query_len
            self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
            self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
            self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)
            cum_num_tokens = self._get_cumsum_and_arange(
                num_scheduled_tokens, self.query_pos.np
            )
            self.query_start_loc.np[0] = 0
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.np[num_reqs + 1 :].fill(cum_num_tokens[-1])
            self.query_start_loc.copy_to_gpu()
            _native.prepare_dcp_dummy_context_metadata(
                input_batch=self.input_batch,
                kv_cache_config=getattr(self, "kv_cache_config", None),
                query_pos=self.query_pos,
                positions=self.positions,
                query_start_loc=self.query_start_loc,
                num_reqs=num_reqs,
                num_tokens_unpadded=num_tokens_unpadded,
                dcp_dummy_context_len=dcp_dummy_context_len,
            )
            self.input_batch.block_table.commit_block_table(num_reqs_padded)
            if self.common_attention_metadata_graph is not None:
                self.num_computed_tokens.copy_(
                    self.seq_lens
                    - (self.query_start_loc.gpu[1:] - self.query_start_loc.gpu[:-1])
                )
            if self._mrope_decode_state is not None:
                self._mrope_decode_state.gpu.zero_()
                self._mrope_decode_active = False
            self._run_common_attention_metadata(
                num_reqs_padded, cudagraph_runtime_mode, capture=is_graph_capturing
            )
        if build_attention:
            pad_attn = cudagraph_runtime_mode == _native.CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                for_cudagraph_capture=is_graph_capturing
                or cudagraph_runtime_mode == _native.CUDAGraphMode.FULL,
                slot_mappings=slot_mappings_by_group,
                use_spec_decode=self.speculative_config is not None,
                block_table_rows_are_current=self.common_attention_metadata_graph
                is not None,
            )
        if slot_mappings_by_group is not None:
            for slot_mapping in slot_mappings_by_group.values():
                slot_mapping.fill_(-1)
    # Compile every metadata launch before capture. A descriptor becomes
    # available to the indexer only after warmup synchronizes successfully.
    if (
        is_graph_capturing
        and cudagraph_runtime_mode == _native.CUDAGraphMode.FULL
        and self._model_runtime is not None
    ):
        self._model_runtime.warm_full_decode_metadata(attn_metadata, self.device)
    with self.maybe_dummy_run_with_lora(
        self.lora_config,
        num_scheduled_tokens,
        num_sampled_tokens,
        remove_lora,
        num_active_loras,
    ):
        assert num_tokens_padded <= self.max_num_tokens
        model_kwargs = self._init_model_kwargs()
        if self.supports_mm_inputs and (not self.model_config.is_encoder_decoder):
            input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)
            model_kwargs = {**model_kwargs, **self._dummy_mm_kwargs(num_reqs)}
        elif self.enable_prompt_embeds:
            input_ids = None
            inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            model_kwargs = self._init_model_kwargs()
        else:
            input_ids = self.input_ids.gpu[:num_tokens_padded]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_tokens_padded]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
        else:
            positions = self.positions[:num_tokens_padded]
        if _native.get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            if self.intermediate_tensors is None:
                self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                    batch_size=self.max_num_tokens,
                    dtype=self.model_config.dtype,
                    device=self.device,
                )
            intermediate_tensors = self.sync_and_gather_intermediate_tensors(
                num_tokens_padded, None, False
            )
        if ubatch_slices_padded is not None:
            num_tokens_padded = ubatch_slices_padded[0].num_tokens
            if num_tokens_across_dp is not None:
                num_tokens_across_dp[:] = num_tokens_padded
        with (
            self.maybe_randomize_inputs(
                input_ids, inputs_embeds, randomize_inputs=randomize_inputs
            ),
            _native.set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,
            ),
        ):
            outputs = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
        if self.use_aux_hidden_state_outputs:
            hidden_states, _ = outputs
        else:
            hidden_states = outputs
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(
                self.drafter,
                _native.EagleProposer
                | _native.DFlashProposer
                | _native.DraftModelProposer
                | _native.ExtractHiddenStatesProposer
                | _native.Gemma4Proposer,
            )
            assert self.speculative_config is not None
            use_cudagraphs = (
                is_graph_capturing
                and cudagraph_runtime_mode == _native.CUDAGraphMode.PIECEWISE
                or (
                    not is_graph_capturing
                    and cudagraph_runtime_mode != _native.CUDAGraphMode.NONE
                )
            ) and (not self.speculative_config.enforce_eager)
            if (
                self.compilation_config.cudagraph_specialize_lora
                and num_active_loras > 0
            ):
                use_cudagraphs = False
            self.drafter.dummy_run(
                num_tokens,
                use_cudagraphs=use_cudagraphs,
                is_graph_capturing=is_graph_capturing,
                slot_mappings=slot_mappings,
            )
    self._register_layerwise_nvtx_hooks()
    if not skip_eplb:
        self.eplb_step(is_dummy=True, is_profile=is_profile)
    logit_indices = _native.np.cumsum(num_scheduled_tokens) - 1
    logit_indices_device = _native.torch.from_numpy(logit_indices).to(
        self.device, non_blocking=True
    )
    return (hidden_states, hidden_states[logit_indices_device])


def runner_cleanup_profiling_kv_cache(self):
    _native = self._hy4_runtime_module
    _native.torch.accelerator.synchronize()
    draft_graph = self.drafter._model_draft_graph
    if draft_graph is not None:
        draft_graph.clear()
    if self.common_attention_metadata_graph is not None:
        self.common_attention_metadata_graph.clear()
    if hasattr(self, "kv_caches") and self.kv_caches:
        for i in range(len(self.kv_caches)):
            self.kv_caches[i] = None
        self.kv_caches.clear()
    if hasattr(self, "cross_layers_kv_cache"):
        self.cross_layers_kv_cache = None
        self.cross_layers_attn_backend = None
    if hasattr(self, "attn_groups"):
        self.attn_groups.clear()
    if hasattr(self, "kv_cache_config"):
        delattr(self, "kv_cache_config")
    self.cache_config.num_gpu_blocks = None
    for layer in self.compilation_config.static_forward_context.values():
        if hasattr(layer, "kv_cache"):
            kv_cache = layer.kv_cache
            layer.kv_cache = (
                _native.torch.tensor([])
                if isinstance(kv_cache, _native.torch.Tensor)
                else []
            )
        if hasattr(layer, "impl"):
            if hasattr(layer.impl, "_k_scale_cache"):
                layer.impl._k_scale_cache = None
            if hasattr(layer.impl, "_v_scale_cache"):
                layer.impl._v_scale_cache = None
    _native.gc.collect()
    _native.torch.accelerator.empty_cache()
    _native.logger.debug("Cleaned up profiling KV cache and CUDA graphs")


@torch.inference_mode()
def runnerprofile_cudagraph_memory(self):
    _native = self._hy4_runtime_module
    with _native.set_current_vllm_config(self.vllm_config):
        self._init_minimal_kv_cache_for_profiling()
    saved_num_cudagraph_captured = _native.compilation_counter.num_cudagraph_captured
    capture_descs = self.cudagraph_dispatcher.get_capture_descs()
    encoder_cudagraph_manager = self._create_encoder_cudagraph_manager()
    decoder_graphs = sum((len(descs) for _, descs in capture_descs))
    encoder_graphs = (
        encoder_cudagraph_manager.get_num_graphs_to_capture()
        if encoder_cudagraph_manager is not None
        else 0
    )
    total_graphs = decoder_graphs + encoder_graphs
    if total_graphs == 0:
        _native.logger.debug("No CUDA graphs will be captured, skipping profiling")
        self._cleanup_profiling_kv_cache()
        return 0
    graph_groups = [
        *(
            f"{mode.name}={len(descs)} (largest={descs[0].num_tokens})"
            for mode, descs in capture_descs
            if descs
        )
    ]
    if encoder_graphs > 0:
        graph_groups.append(
            f"ENCODER={encoder_graphs} (largest={encoder_cudagraph_manager.token_budgets[-1]})"
        )
    _native.logger.info("Profiling CUDA graph memory: %s", ", ".join(graph_groups))
    profiling_pool = _native.current_platform.graph_pool_handle()
    encoder_profiling_pool = _native.current_platform.graph_pool_handle()
    original_pools = {}
    all_wrappers = list(_native.CUDAGraphWrapper._all_instances) + list(
        _native.BreakableCUDAGraphWrapper._all_instances
    )
    for instance in all_wrappers:
        original_pools[id(instance)] = instance.graph_pool
        instance.graph_pool = profiling_pool
    shared_memory_estimate = {}
    per_graph_estimate = {}
    encoder_memory_estimate = 0
    draft_graph_memory_estimate = 0
    cap_ctx = (
        _native.GraphCaptureContext(_native.current_stream())
        if _native.current_platform.is_rocm()
        else None
    )
    try:
        _native.set_cudagraph_capturing_enabled(True)
        with (
            self._freeze_gc(),
            _native.graph_capture(device=self.device, graph_capture_context=cap_ctx),
        ):
            _native.torch.accelerator.synchronize()
            _native.torch.accelerator.empty_cache()
            for mode, descs in capture_descs:
                profile_descs = descs[:2]
                mem_samples = []
                for i, desc in enumerate(profile_descs):
                    mem_before = _native.torch.accelerator.get_memory_info()[0]
                    self._warmup_and_capture(
                        desc,
                        cudagraph_runtime_mode=mode,
                        profile_seq_lens=(
                            min(
                                self.max_model_len,
                                self.max_num_tokens // desc.num_tokens,
                            )
                            if mode == _native.CUDAGraphMode.FULL and i == 0
                            else None
                        ),
                    )
                    _native.torch.accelerator.synchronize()
                    free_after = _native.torch.accelerator.get_memory_info()[0]
                    mem_samples.append(mem_before - free_after)
                first_capture = mem_samples[0]
                per_graph = max(mem_samples[1] if len(mem_samples) > 1 else 0, 1 << 20)
                shared_memory_estimate[mode] = first_capture
                per_graph_estimate[mode] = per_graph * (len(descs) - 1)
                _native.logger.debug(
                    "Estimated %s CUDA graph memory: %.2f MiB first-capture + (%d-1) × %.2f MiB per-graph",
                    mode.name,
                    first_capture / (1 << 20),
                    len(descs),
                    per_graph / (1 << 20),
                )
            draft_graph = self.drafter._model_draft_graph
            if draft_graph is not None:
                mem_before = _native.torch.accelerator.get_memory_info()[0]
                draft_graph.capture()
                _native.torch.accelerator.synchronize()
                draft_graph_memory_estimate = max(
                    mem_before - _native.torch.accelerator.get_memory_info()[0],
                    draft_graph.graph_entry_count * (1 << 20),
                )
                _native.logger.info(
                    "Estimated HYV4 draft graph memory: %.2f MiB",
                    draft_graph_memory_estimate / (1 << 20),
                )
            if encoder_cudagraph_manager is not None:
                mem_before = _native.torch.accelerator.get_memory_info()[0]
                encoder_cudagraph_manager.capture(graph_pool=encoder_profiling_pool)
                _native.torch.accelerator.synchronize()
                free_after = _native.torch.accelerator.get_memory_info()[0]
                encoder_memory_estimate = max(mem_before - free_after, 0)
                _native.logger.debug(
                    "Estimated encoder CUDA graph memory: %.2f MiB for %d graphs",
                    encoder_memory_estimate / (1 << 20),
                    encoder_graphs,
                )
    finally:
        _native.set_cudagraph_capturing_enabled(False)
        _native.CUDAGraphWrapper.clear_all_graphs()
        _native.BreakableCUDAGraphWrapper.clear_all_graphs()
        if encoder_cudagraph_manager is not None:
            encoder_cudagraph_manager.clear()
        all_wrappers = list(_native.CUDAGraphWrapper._all_instances) + list(
            _native.BreakableCUDAGraphWrapper._all_instances
        )
        for instance in all_wrappers:
            if id(instance) in original_pools:
                instance.graph_pool = original_pools[id(instance)]
        for key_set in self.cudagraph_dispatcher.cudagraph_keys.values():
            key_set.clear()
        self.cudagraph_dispatcher.keys_initialized = False
        self.maybe_remove_all_loras(self.lora_config)
        self._cleanup_profiling_kv_cache()
        _native.compilation_counter.num_cudagraph_captured = (
            saved_num_cudagraph_captured
        )
    decoder_estimate = max(shared_memory_estimate.values(), default=0) + sum(
        per_graph_estimate.values()
    )
    total_estimate = (
        decoder_estimate + encoder_memory_estimate + draft_graph_memory_estimate
    )
    _native.logger.info(
        "Estimated CUDA graph memory: %.2f GiB total", total_estimate / (1 << 30)
    )
    return int(total_estimate)


@instrument(span_name="Capture model")
def runnercapture_model(self):
    _native = self._hy4_runtime_module
    if self.compilation_config.cudagraph_mode == _native.CUDAGraphMode.NONE:
        _native.logger.warning(
            "Skipping CUDA graph capture. To turn on CUDA graph capture, ensure `cudagraph_mode` was not manually set to `NONE`"
        )
        return 0
    self._maybe_init_encoder_cudagraph_manager()
    _native.compilation_counter.num_gpu_runner_capture_triggers += 1
    start_time = _native.time.perf_counter()
    _native.set_cudagraph_capturing_enabled(True)
    from vllm.distributed.parallel_state import get_world_group

    local_rank = get_world_group().local_rank
    enable_profiler = (
        local_rank == 0 and self.vllm_config.profiler_config.capture_torch_profiler
    )
    if enable_profiler:
        trace_dir = (
            self.vllm_config.profiler_config.torch_profiler_dir + "/capture_traces"
        )
        profiler = _native.torch.profiler.profile(
            activities=[
                _native.torch.profiler.ProfilerActivity.CPU,
                _native.torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=_native.torch.profiler.tensorboard_trace_handler(
                trace_dir, worker_name=f"graph_capture_rank_{local_rank}", use_gzip=True
            ),
        )
        _native.logger.info_once(
            "Rank %d: Torch profiler enabled for CUDA graph capture, traces will be saved to: %s",
            local_rank,
            trace_dir,
        )
    else:
        profiler = _native.nullcontext()
        _native.logger.info_once(
            "Rank %d: Torch profiler disabled for CUDA graph capture", local_rank
        )
    with self._freeze_gc(), _native.graph_capture(device=self.device):
        _native.torch.accelerator.synchronize()
        _native.torch.accelerator.empty_cache()
        start_free_gpu_memory = _native.torch.accelerator.get_memory_info()[0]
        for runtime_mode, batch_descs in self.cudagraph_dispatcher.get_capture_descs():
            self._capture_cudagraphs(
                batch_descriptors=batch_descs,
                cudagraph_runtime_mode=runtime_mode,
                profiler=profiler,
            )
            _native.torch.accelerator.synchronize()
        draft_graph = self.drafter._model_draft_graph
        if draft_graph is not None:
            draft_graph.capture()
        if self.encoder_cudagraph_manager is not None:
            encoder_graph_pool = _native.current_platform.graph_pool_handle()
            self.encoder_cudagraph_manager.capture(graph_pool=encoder_graph_pool)
        _native.torch.accelerator.synchronize()
        end_free_gpu_memory = _native.torch.accelerator.get_memory_info()[0]
    _native.set_cudagraph_capturing_enabled(False)
    _native.torch.accelerator.synchronize()
    _native.torch.accelerator.empty_cache()
    _native.lock_workspace()
    end_time = _native.time.perf_counter()
    elapsed_time = end_time - start_time
    cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
    _native.logger.info_once(
        "Graph capturing finished in %.0f secs, took %.2f GiB",
        elapsed_time,
        cuda_graph_size / (1 << 30),
    )
    return cuda_graph_size


def runner_warmup_and_capture(
    self,
    desc,
    cudagraph_runtime_mode,
    profile_seq_lens=None,
    allow_microbatching=False,
    num_warmups=None,
    profiler=None,
):
    _native = self._hy4_runtime_module
    if profiler is None:
        profiler = _native.nullcontext()
    if num_warmups is None:
        num_warmups = self.compilation_config.cudagraph_num_of_warmups
    force_attention = cudagraph_runtime_mode == _native.CUDAGraphMode.FULL or (
        self._model_runtime is not None
        and self._model_runtime.mixed_graphs
        and (cudagraph_runtime_mode == _native.CUDAGraphMode.PIECEWISE)
    )
    for _ in range(num_warmups):
        self._dummy_run(
            desc.num_tokens,
            cudagraph_runtime_mode=_native.CUDAGraphMode.NONE,
            force_attention=force_attention,
            uniform_decode=desc.uniform,
            allow_microbatching=allow_microbatching,
            skip_eplb=True,
            remove_lora=False,
            num_active_loras=desc.num_active_loras,
            profile_seq_lens=profile_seq_lens,
        )
    if num_warmups > 0:
        _native.torch.accelerator.synchronize()
    with (
        profiler,
        _native.torch.profiler.record_function(
            f"capture_{desc.num_tokens}_{cudagraph_runtime_mode.name}"
        ),
    ):
        self._dummy_run(
            desc.num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            uniform_decode=desc.uniform,
            allow_microbatching=allow_microbatching,
            skip_eplb=True,
            remove_lora=False,
            num_active_loras=desc.num_active_loras,
            is_graph_capturing=True,
            profile_seq_lens=profile_seq_lens,
        )


def proposerinitialize_cudagraph_keys(self, cudagraph_mode):
    """Initialize cudagraph dispatcher keys for the drafter.

    Only supports PIECEWISE cudagraphs (via mixed_mode).
    This should be called after adjust_cudagraph_sizes_for_spec_decode.
    """
    _native = self._hy4_runtime_module
    if not self.speculative_config.enforce_eager and cudagraph_mode.mixed_mode() in [
        _native.CUDAGraphMode.PIECEWISE,
        _native.CUDAGraphMode.FULL,
    ]:
        eagle_cudagraph_mode = _native.CUDAGraphMode.PIECEWISE
    else:
        eagle_cudagraph_mode = _native.CUDAGraphMode.NONE
    self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)
    if self._model_draft_graph is not None:
        self._model_draft_graph.initialize(cudagraph_mode)


def proposerpropose(
    self,
    num_speculative_tokens,
    target_token_ids,
    target_positions,
    target_hidden_states,
    next_token_ids,
    token_indices_to_sample,
    common_attn_metadata,
    sampling_metadata,
    mm_embed_inputs=None,
    num_rejected_tokens_gpu=None,
    slot_mappings=None,
):
    _native = self._hy4_runtime_module
    self.num_speculative_tokens = num_speculative_tokens
    self._last_draft_probs = None
    batch_size = common_attn_metadata.batch_size()
    if self.method in ("eagle3", "dflash"):
        model = self.model
        if isinstance(model, _native.BreakableCUDAGraphWrapper):
            model = model.unwrap()
        assert isinstance(
            model,
            (
                _native.Eagle3LlamaForCausalLM,
                _native.Eagle3DeepseekV2ForCausalLM,
                _native.DFlashQwen3ForCausalLM,
                _native.Eagle3Qwen3ForCausalLM,
                _native.DFlashLagunaForCausalLM,
            ),
        )
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
        assert target_hidden_states.shape[-1] == self.hidden_size
    num_tokens, token_indices_to_sample, common_attn_metadata = (
        self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
    )
    per_group_attn_metadata, per_layer_attn_metadata = (
        self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
    )
    cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
        self._determine_batch_execution_and_padding(num_tokens)
    )
    model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
        num_tokens, num_input_tokens, mm_embed_inputs
    )
    if self._share_mtp_indices:
        self.model.model.set_skip_topk(False)
    if self.eplb_state is not None:
        self.eplb_state.prepare_forward(self.draft_model_config, num_tokens)
    with _native.set_forward_context(
        per_layer_attn_metadata,
        self.vllm_config,
        num_tokens=num_input_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        slot_mapping=self._get_slot_mapping(
            slot_mapping_size, common_attn_metadata.slot_mapping
        ),
    ):
        full = (
            self._model_draft_graph.full_draft3
            if self._model_draft_graph is not None
            else None
        )
        if full is not None:
            merged = full.try_run(
                model_kwargs,
                common_attn_metadata,
                sampling_metadata,
                token_indices_to_sample=token_indices_to_sample,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                group_metadata=per_group_attn_metadata,
                mm_embed_inputs=mm_embed_inputs,
            )
            if merged is not None:
                return merged
        ret_hidden_states = self._call_model_with_graph(
            model_kwargs,
            common_attn_metadata,
            draft_index=0,
            verified_step=num_rejected_tokens_gpu is not None,
        )
        if not self.model_returns_tuple():
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
    if self._share_mtp_indices:
        self.model.model.set_skip_topk(True)
        self.model.model.compact_topk_indices(token_indices_to_sample)
    sample_hidden_states = last_hidden_states[token_indices_to_sample]
    if self.num_speculative_tokens == 0:
        return _native.torch.empty(
            batch_size, 0, device=sample_hidden_states.device, dtype=_native.torch.int64
        )
    if self.num_speculative_tokens == 1 or self.parallel_drafting:
        draft_token_ids, draft_probs = self._sample_draft_tokens(
            sample_hidden_states, sampling_metadata
        )
        if draft_probs is not None:
            self._last_draft_probs = draft_probs.view(
                -1, self.num_speculative_tokens, draft_probs.shape[-1]
            ).contiguous()
        return draft_token_ids.view(-1, self.num_speculative_tokens)
    if self.uses_mrope:
        positions = self.mrope_positions[:, token_indices_to_sample]
    else:
        positions = self.positions[token_indices_to_sample]
    hidden_states = hidden_states[token_indices_to_sample]
    if self.constant_draft_positions:
        self.positions[:batch_size] = positions
    draft_token_ids, draft_probs = self._sample_draft_tokens(
        sample_hidden_states, sampling_metadata
    )
    draft_probs_list = None if draft_probs is None else [draft_probs]
    if self.allowed_attn_types is not None:
        for group_md in per_group_attn_metadata:
            if not isinstance(group_md, self.allowed_attn_types):
                raise ValueError(
                    f"Unsupported attention metadata type for speculative decoding with num_speculative_tokens > 1: {type(group_md)}. Supported types are: {self.allowed_attn_types}"
                )
    draft_token_ids_list = [draft_token_ids]
    cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
        self._determine_batch_execution_and_padding(batch_size)
    )
    common_attn_metadata.num_actual_tokens = batch_size
    common_attn_metadata.max_query_len = 1
    common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
    common_attn_metadata.query_start_loc_cpu = _native.torch.from_numpy(
        self.token_arange_np[: batch_size + 1]
    ).clone()
    if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
        common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
        common_attn_metadata._seq_lens_cpu = None
        common_attn_metadata._num_computed_tokens_cpu = None
    block_size = self.block_size
    assert block_size > 0, "block_size has not been initialized."
    tail = self._model_draft_graph.tail if self._model_draft_graph is not None else None
    if tail is not None:
        merged = tail.try_run(
            draft_token_ids,
            hidden_states,
            positions,
            common_attn_metadata,
            sampling_metadata,
            batch_size=batch_size,
            input_batch_size=input_batch_size,
            verified_step=num_rejected_tokens_gpu is not None,
        )
        if merged is not None:
            return merged
    for token_index in range(self.num_speculative_tokens - 1):
        input_ids = draft_token_ids_list[-1].int()
        if self.use_heterogeneous_vocab:
            assert self.vocab_mapping is not None
            input_ids = self.vocab_mapping.map_target_to_draft_ids(input_ids)
        if not self.constant_draft_positions:
            positions = self._update_positions_dependent_metadata(
                positions,
                common_attn_metadata,
                batch_size,
                input_batch_size,
                block_size,
            )
        if not self.constant_draft_positions or token_index == 0:
            _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
                common_attn_metadata, draft_index=token_index + 1
            )
        self.input_ids[:batch_size] = input_ids
        self.hidden_states[:batch_size] = hidden_states
        if self.supports_mm_inputs:
            self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)
            input_ids = None
            inputs_embeds = self.inputs_embeds[:input_batch_size]
        else:
            input_ids = self.input_ids[:input_batch_size]
            inputs_embeds = None
        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(input_batch_size),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]
        if self.eplb_state is not None:
            self.eplb_state.prepare_forward(self.draft_model_config, batch_size)
        with _native.set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=input_batch_size,
            num_tokens_across_dp=batch_size_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(input_batch_size),
        ):
            ret_hidden_states = self._call_model_with_graph(
                model_kwargs,
                common_attn_metadata,
                draft_index=token_index + 1,
                verified_step=True,
            )
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
        hidden_states = hidden_states[:batch_size]
        draft_token_ids, draft_probs = self._sample_draft_tokens(
            last_hidden_states[:batch_size], sampling_metadata
        )
        if draft_probs is not None:
            assert draft_probs_list is not None
            draft_probs_list.append(draft_probs)
        draft_token_ids_list.append(draft_token_ids)
    draft_token_ids = _native.torch.stack(draft_token_ids_list, dim=1)
    if draft_probs_list is not None:
        self._last_draft_probs = _native.torch.stack(
            draft_probs_list, dim=1
        ).contiguous()
    return draft_token_ids


def proposer_call_model_with_graph(self, kwargs, common, *, draft_index, verified_step):
    manager = self._model_draft_graph
    if manager is None:
        return self.model(**kwargs)
    # Admission checks run before any model invocation. A failure after graph
    # execution starts propagates; the native forward is not a retry path.
    manager.validate_cache_binding()
    return manager.run(
        kwargs, common, draft_index=draft_index, verified_step=verified_step
    )


def install_runtime_patches():
    """Install once before constructing the selected NVIDIA V1 worker."""
    from vllm.v1.spec_decode import llm_base_proposer
    from vllm.v1.worker import gpu_model_runner

    return _install_runtime_patches(gpu_model_runner, llm_base_proposer)


def _install_runtime_patches(runner_module, proposer_module):
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return False
        runner = runner_module.GPUModelRunner
        proposer = proposer_module.SpecDecodeBaseProposer
        runner_methods = (
            ("__init__", runner.__init__, None),
            ("_prepare_input_ids", runner._prepare_input_ids, runner_prepare_input_ids),
            ("_prepare_inputs", runner._prepare_inputs, runner_prepare_inputs),
            (
                "_build_attention_metadata",
                runner._build_attention_metadata,
                runner_build_attention_metadata,
            ),
            ("_bookkeeping_sync", runner._bookkeeping_sync, runner_bookkeeping_sync),
            (
                "synchronize_input_prep",
                runner.synchronize_input_prep,
                runnersynchronize_input_prep,
            ),
            ("_prepare_decode_mrope", _ABSENT, runner_prepare_decode_mrope),
            (
                "_run_common_attention_metadata",
                _ABSENT,
                runner_run_common_attention_metadata,
            ),
            ("_get_slot_mappings", runner._get_slot_mappings, runner_get_slot_mappings),
            ("execute_model", runner.execute_model, runnerexecute_model),
            ("sample_tokens", runner.sample_tokens, runnersample_tokens),
            ("load_model", runner.load_model, runnerload_model),
            (
                "_retain_prompt_logprobs_hidden_states",
                _ABSENT,
                runner_retain_prompt_logprobs_hidden_states,
            ),
            (
                "_get_prompt_logprobs_dict",
                runner._get_prompt_logprobs_dict,
                runner_get_prompt_logprobs_dict,
            ),
            ("_dummy_run", runner._dummy_run, runner_dummy_run),
            (
                "_cleanup_profiling_kv_cache",
                runner._cleanup_profiling_kv_cache,
                runner_cleanup_profiling_kv_cache,
            ),
            (
                "profile_cudagraph_memory",
                runner.profile_cudagraph_memory,
                runnerprofile_cudagraph_memory,
            ),
            ("capture_model", runner.capture_model, runnercapture_model),
            (
                "_warmup_and_capture",
                runner._warmup_and_capture,
                runner_warmup_and_capture,
            ),
        )
        proposer_methods = (
            ("__init__", proposer.__init__, None),
            (
                "initialize_cudagraph_keys",
                proposer.initialize_cudagraph_keys,
                proposerinitialize_cudagraph_keys,
            ),
            ("propose", proposer.propose, proposerpropose),
            ("_call_model_with_graph", _ABSENT, proposer_call_model_with_graph),
        )
        plan = []
        for kind, owner, module, methods in (
            ("runner", runner, runner_module, runner_methods),
            ("proposer", proposer, proposer_module, proposer_methods),
        ):
            for name, original, implementation in methods:
                replacement = _scoped_method(
                    original, implementation, module, kind, name
                )
                plan.append((owner, name, original, replacement))
        apply_patch_plan(plan)
        _INSTALLED = True
        return True
