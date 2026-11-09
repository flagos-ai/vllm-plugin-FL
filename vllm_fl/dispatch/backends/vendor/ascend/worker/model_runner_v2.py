# Copyright (c) 2026 BAAI. All rights reserved.
# KV initialization adapted from vLLM v0.32.0:
# https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend execution adapter for the upstream vLLM V2 model runner.

The upstream V2 request state, input preparation and sampler remain in use.
Only native KV layouts and block operations differ from the GPU implementation.
"""

from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy

import numpy as np
import torch

from ..patches.ascend_v2_device import install_ascend_v2_device_shims

# Direct imports of this adapter must get the same shims as WorkerFL imports.
install_ascend_v2_device_shims()

from ..patches.ascend_v2_sampling import install_ascend_v2_sampling_shims

install_ascend_v2_sampling_shims()

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.aux_output_connector.worker import get_aux_output_connector
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
from vllm.v1.worker.gpu import pcp_manager as pcp
from vllm.v1.worker.gpu.attn_utils import (
    FastPrefillHelper,
    get_shared_kv_cache_layers,
    init_attn_backend,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    has_compiled_submodule,
)
from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR, get_kv_connector
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    maybe_create_adaptive_verification_manager,
    resolve_adaptive_cudagraph_mode,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import get_max_chunk_logits
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.ubatch_utils import maybe_build_ubatch_runner
from vllm.v1.worker.utils import bind_kv_cache_to_layers

from .ascend_v2_block_table import AscendBlockTables
from .ascend_v2_graph import (
    AscendModelGraphManager,
    install_graph_metadata_bridge,
)
from vllm_fl.kv_cache.ascend.hybrid import (
    allocate_ascend_kv_cache,
    copy_block_planes,
    uses_dense_hybrid_cache,
    zero_block_planes,
)

# vLLM configures INFO handlers for the vllm namespace, not vllm_fl.
logger = init_logger("vllm.ascend.model_runner_v2")


class _AscendBlockZeroer:
    def __init__(self, planes):
        self.planes = planes

    def zero_block_ids(self, block_ids):
        zero_block_planes(self.planes, block_ids)

    def warmup(self, num_kv_blocks):
        if num_kv_blocks:
            self.zero_block_ids([0])


class GPUModelRunnerFLV2(GPUModelRunner):
    """Use the V2 execution path with Ascend's native cache contract."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        assert vllm_config.use_v2_model_runner, "Ascend V2 requires use_v2_model_runner"
        if device.type != "npu":
            raise ValueError("GPUModelRunnerFLV2 is the Ascend V2 adapter")
        if vllm_config.compilation_config.cudagraph_mode not in (
            CUDAGraphMode.NONE,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ):
            raise NotImplementedError("Ascend V2 supports NONE/FULL_DECODE_ONLY graphs")
        self.use_ascend_hybrid_cache = uses_dense_hybrid_cache(vllm_config.model_config)
        if self.use_ascend_hybrid_cache and vllm_config.kv_transfer_config is not None:
            raise NotImplementedError(
                "Ascend dense hybrid KV transfer requires a connector that understands "
                "the separate convolution, SSM and attention block planes"
            )
        self._ascend_manager_views = {}
        self._ascend_block_planes = []
        super().__init__(vllm_config, device)
        logger.info(
            "Ascend model runner V2 active: %s.%s",
            type(self).__module__,
            type(self).__name__,
        )

    def _allocate_and_bind_kv_cache(self, kv_cache_allocation_context=None):
        if self.vllm_config.attention_config.hisparse_config is not None:
            raise NotImplementedError("Ascend V2 does not support HiSparse KV pools")
        with kv_cache_allocation_context or nullcontext():
            bind_views, managers, planes = allocate_ascend_kv_cache(
                self.kv_cache_config,
                self.device,
                self.cache_config.get_resolved_kv_cache_layout(),
                self.kernel_block_sizes,
                (group for groups in self.attn_groups for group in groups),
                self.use_ascend_hybrid_cache,
            )
        for name, target in get_shared_kv_cache_layers(self.vllm_config).items():
            bind_views[name] = bind_views[target]
            if target in managers:
                managers[name] = managers[target]
        forward_context = self.compilation_config.static_forward_context
        bind_kv_cache_to_layers(
            {
                name: view
                for name, view in bind_views.items()
                if name in forward_context
            },
            forward_context,
            num_attn_module=(
                2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
            ),
            kv_cache_groups=self.kv_cache_config.kv_cache_groups,
        )
        self._ascend_manager_views = managers
        self._ascend_block_planes = planes
        # Dense recurrent state has no single affine B-first manager view.
        # The connector is rejected above, and block operations use planes.
        return managers

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        # GPUWorker finalizes the PD interleave before KV cache initialization.
        self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size
        # V2 creates model_state in load_model(), after runner construction.
        install_graph_metadata_bridge(self.model_state, lambda: self.cudagraph_manager)
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        block_table_max_model_len = self.max_model_len
        if self.is_encoder_decoder:
            # Cross-attention block tables need to index encoder tokens, which
            # can exceed the decoder's max_model_len.
            block_table_max_model_len = max(
                block_table_max_model_len,
                self.scheduler_config.max_num_encoder_input_tokens,
                getattr(self.model_config.hf_config, "max_source_positions", 0),
            )

        block_sizes = []
        max_num_blocks_per_group = []
        slot_mapping_enabled = []
        dcp_sharded = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            layer_spec = (
                spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            )
            slot_mapping_enabled.append(layer_spec.uses_slot_mapping)
            dcp_sharded.append(spec.dcp_sharded)
            # Let each cache type account for CP. Attention KV is DCP-sharded,
            # while Mamba/GDN recurrent state is replicated across DCP ranks.
            max_num_blocks = spec.max_num_blocks_per_req(
                self.vllm_config, block_table_max_model_len
            )
            # Preserve each cache type's alignment requirements after applying
            # its topology-aware block-table width.
            max_num_blocks = get_block_table_width(
                max_num_blocks,
                spec.block_size,
                token_alignment=spec.block_table_token_alignment,
            )
            max_num_blocks_per_group.append(max_num_blocks)

        target_attn_layer_names = None
        if isinstance(self.speculator, DraftModelSpeculator):
            # Adaptive verification validates target attention separately.
            target_attn_layer_names = {
                layer_name
                for group in self.kv_cache_config.kv_cache_groups
                for layer_name in group.layer_names
            } - self.speculator.draft_attn_layer_names
        draft_attn_layer_names = None
        if isinstance(self.speculator, DraftModelSpeculator):
            draft_attn_layer_names = self.speculator.draft_attn_layer_names
        # Metadata builders select attention kernels that need JIT warmup.
        with self.jit_warmup_registry.activate():
            (
                self.attn_groups,
                attn_cg_support,
                self.kernel_block_sizes,
            ) = init_attn_backend(
                self.kv_cache_config,
                self.vllm_config,
                self.device,
                draft_layer_names=draft_attn_layer_names,
            )
        additional_attn_cg_support = self.model_state.get_additional_cg_support()
        attn_cg_support = attn_cg_support.narrow(*additional_attn_cg_support)
        # The speculator clears the flag at load time when the checkpoint has
        # no confidence head, so it holds the effective value.
        self.adaptive_verification = maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=getattr(
                self.speculator, "enable_adaptive_verification", False
            ),
            attn_groups=self.attn_groups,
            req_states=self.req_states,
            query_start_loc=self.input_buffers.query_start_loc,
            num_bonus_tokens=self.model_state.num_new_sampled_tokens_per_step,
            max_total_logits=get_max_chunk_logits(self.vocab_size),
            vllm_config=self.vllm_config,
            target_layer_names=target_attn_layer_names,
            additional_attn_cg_support=additional_attn_cg_support,
        )

        self.block_tables = AscendBlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            slot_mapping_enabled=slot_mapping_enabled,
            dcp_sharded=dcp_sharded,
            cp_size=self.dcp_size,
            cp_rank=self.dcp_rank,
            cp_interleave=self.cp_interleave,
        )
        self.pcp_manager = pcp.maybe_build_pcp_manager(
            self.vllm_config,
            self.device,
            self.supports_mm_inputs,
            self.block_tables,
            cls=self.pcp_manager_cls,
        )
        self.ubatch_runner = maybe_build_ubatch_runner(
            self.vllm_config,
            self.device,
            self.model_state,
            self.attn_groups,
            self.kv_cache_config,
            self.max_num_reqs,
        )
        if self.speculator is not None:
            self.speculator.pcp_manager = self.pcp_manager
        initialize_mamba_ssu_backend(
            self.vllm_config.mamba_config,
            self.kv_cache_config,
            use_replayssm=self.cache_config.use_replayssm,
        )
        piecewise_capture_available = bool(
            envs.VLLM_USE_BREAKABLE_CUDAGRAPH or has_compiled_submodule(self.model)
        )
        if self.adaptive_verification is not None:
            self.compilation_config.cudagraph_mode = resolve_adaptive_cudagraph_mode(
                self.compilation_config.cudagraph_mode,
                piecewise_capture_available=piecewise_capture_available,
            )
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            attn_cg_support.min_cg_support,
            attn_cg_support.min_cg_attn_backend,
            self.decode_query_len,
            use_v2_model_runner=True,
            kv_cache_config=self.kv_cache_config,
            max_num_reqs=self.max_num_reqs,
            is_profiling=is_profiling,
            piecewise_capture_available=piecewise_capture_available,
        )
        self.cudagraph_manager = AscendModelGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.decode_query_len,
            lora_capture_cases=self.lora_capture_cases,
            varlen_decode=self.adaptive_verification is not None,
            ubatch_runner=self.ubatch_runner,
        )
        if self.cache_config.kv_sharing_fast_prefill and self.pcp_manager is None:
            self.fast_prefill = FastPrefillHelper(
                self.cudagraph_manager, self.max_num_tokens
            )
        check_attention_cp_compatibility(self.vllm_config, target_attn_layer_names)
        if isinstance(self.speculator, DraftModelSpeculator):
            # HACK(woosuk)
            self.speculator.set_attn(
                self.model_state,
                self.kv_cache_config,
                self.block_tables,
                self.input_buffers,
                self.attn_groups,
            )
        if self.speculator is not None:
            # After set_attn, so the speculator can size its cudagraph mode
            # to its own attention support.
            self.speculator.init_cudagraph_manager(cudagraph_mode)

        # Capture warmup providers that depend on allocated KV-cache strides.
        with self.jit_warmup_registry.activate():
            kv_caches_dict = self._allocate_and_bind_kv_cache(
                kv_cache_allocation_context=kv_cache_allocation_context
            )
        self.kv_caches = [
            cache for cache in kv_caches_dict.values() if cache.device == self.device
        ]
        self.model_state.initialize_kv_cache(self.kv_cache_config, self.block_tables)
        if is_profiling:
            self.kv_connector = NO_OP_KV_CONNECTOR
        else:
            self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

            # AuxOutput connector requires resolved kv_cache_config.
            if self.vllm_config.aux_output_config.enabled:
                self.aux_output_connector = get_aux_output_connector(
                    self.model, self.vllm_config, kv_cache_config
                )

    def profile_cudagraph_memory(self) -> int:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            return super().profile_cudagraph_memory()
        try:
            return super().profile_cudagraph_memory()
        finally:
            # Upstream releases its temporary managers/layer caches. Release
            # Ascend's additional aliases and task registry before real init.
            from vllm_fl.compilation.graph_params import clear_graph_params

            self._ascend_manager_views.clear()
            self._ascend_block_planes.clear()
            self.kv_block_zeroer = None
            clear_graph_params()

    def _init_kv_zero_meta(self) -> None:
        self.kv_block_zeroer = _AscendBlockZeroer(self._ascend_block_planes)

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        # Keep the V2 request-state update; its generic copy path assumes the
        # cache bound to each layer is addressed by scheduler block IDs.
        reqs = scheduler_output.scheduled_cached_reqs
        for req_id, num_computed_tokens, new_block_ids in zip(
            reqs.req_ids, reqs.num_computed_tokens, reqs.new_block_ids
        ):
            req_index = self.req_states.req_id_to_index[req_id]
            self.req_states.num_computed_tokens_np[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.block_tables.append_block_ids(
                    req_index, new_block_ids, overwrite=False
                )
        np.minimum(
            self.req_states.num_computed_tokens_np,
            self.req_states.prefill_len.np,
            out=self.req_states.num_computed_prefill_tokens,
        )
        if scheduler_output.new_block_ids_to_zero:
            zero_block_planes(
                self._ascend_block_planes, scheduler_output.new_block_ids_to_zero
            )
        if scheduler_output.kv_cache_block_copies:
            copy_block_planes(
                self._ascend_block_planes, scheduler_output.kv_cache_block_copies
            )

    def shutdown(self) -> None:
        if isinstance(self.cudagraph_manager, AscendModelGraphManager):
            self.cudagraph_manager.release_graphs()
        self._ascend_block_planes.clear()
        self._ascend_manager_views.clear()
        self.kv_block_zeroer = None
        super().shutdown()
