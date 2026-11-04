# Copyright (c) 2026 BAAI. All rights reserved.
# Capture input/output preparation adapted from vLLM v0.32.0:
# https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu/cudagraph_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 uniform decode graphs using the existing Ascend attention task lifecycle.

The upstream V2 runner stages request inputs and samples outside the model
forward graph. Native FIA also carries host sequence lengths, so replay must
update its captured task parameters from this step's attention metadata.
"""

from collections.abc import Callable
from functools import wraps
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.compilation.counter import compilation_counter
from vllm.compilation.cudagraph_pool import capture_pool
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import is_global_first_rank
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CreateForwardFn,
    ModelCudaGraphManager,
    has_compiled_submodule,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_fl.compilation.graph_runtime import GraphRuntimeController

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

logger = init_logger("vllm.ascend.v2_graph")


def _validate_uniform_decode_batch(input_batch):
    # The upstream dispatcher permits one-token prompt extensions only when
    # they have existing context. Do not turn a new prompt or longer prefill
    # into a graph decode merely because a FULL descriptor was selected.
    if input_batch is None or not input_batch.decode_graph_eligible:
        raise ValueError("Ascend V2 decode graph cannot replay a prefill batch")
    scheduled = input_batch.num_scheduled_tokens[: input_batch.num_reqs]
    if len(scheduled) == 0:
        raise ValueError("Ascend V2 decode graph requires active requests")
    if any(int(length) != 1 for length in scheduled):
        raise ValueError("Ascend V2 decode graph requires one query per active request")


def install_graph_metadata_bridge(model_state, manager_getter):
    """Stage FULL metadata on this worker instance without changing V2 globals."""
    if hasattr(model_state, "_ascend_graph_prepare_attn_original"):
        return
    original = model_state.prepare_attn

    @wraps(original)
    def prepare_attn(*args, **kwargs):
        mode = kwargs.get("cudagraph_mode", args[1] if len(args) > 1 else None)
        if mode == CUDAGraphMode.FULL:
            input_batch = kwargs.get("input_batch", args[0] if args else None)
            _validate_uniform_decode_batch(input_batch)
        metadata = original(*args, **kwargs)
        if mode == CUDAGraphMode.FULL:
            manager = manager_getter()
            if not isinstance(manager, AscendModelGraphManager):
                raise RuntimeError("Ascend FULL metadata has no graph manager")
            manager.stage_replay_metadata(metadata)
        return metadata

    model_state._ascend_graph_prepare_attn_original = original
    model_state.prepare_attn = prepare_attn


class AscendModelGraphManager(ModelCudaGraphManager):
    """Keep V2 dispatch and persistent outputs; specialize NPU capture/replay."""

    def __init__(
        self, vllm_config: VllmConfig, device, cudagraph_mode, *args, **kwargs
    ):
        if cudagraph_mode not in (
            CUDAGraphMode.NONE,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ):
            raise NotImplementedError("Ascend V2 supports NONE/FULL_DECODE_ONLY graphs")
        if device.type != "npu":
            raise ValueError("Ascend graph manager requires an NPU")
        if cudagraph_mode != CUDAGraphMode.NONE:
            if vllm_config.speculative_config is not None:
                raise NotImplementedError("Ascend V2 graphs do not support speculation")
            if vllm_config.parallel_config.use_ubatching:
                raise NotImplementedError("Ascend V2 graphs do not support ubatching")
        super().__init__(vllm_config, device, cudagraph_mode, *args, **kwargs)
        self.runtime = GraphRuntimeController(vllm_config=vllm_config)
        self._capturing = False
        self._capture_has_lora = False
        self._capture_metadata = {}
        self._pending_metadata = None
        self._last_replay_metadata = None
        self._npu_capture_stream = None
        self.capture_count = 0
        self.replay_count = 0
        self.replay_counts = {}

    def stage_replay_metadata(self, metadata):
        self._pending_metadata = metadata

    def _capture_graphs(self, create_forward_fn: CreateForwardFn, progress_bar_desc):
        if any(mode != CUDAGraphMode.FULL for mode in self._capture_descs):
            raise NotImplementedError("Ascend V2 captures uniform FULL decode only")
        descs = self._capture_descs.get(CUDAGraphMode.FULL, [])
        if self._max_full_descs_to_capture is not None:
            descs = descs[: self._max_full_descs_to_capture]
        if not descs:
            self._graphs_captured = True
            return
        if self._npu_capture_stream is None:
            self._npu_capture_stream = torch.npu.Stream(device=self.device)
        capture_stream = self._npu_capture_stream
        original_stream = torch.npu.current_stream(self.device)
        capture_stream.wait_stream(original_stream)
        try:
            # Upstream graph_capture assumes Cuda/XpuCommunicator. FlagCX NPU
            # collectives use this native stream directly, as ordinary NPU ops.
            with (
                torch.npu.stream(capture_stream),
                set_current_vllm_config(self.vllm_config),
            ):
                iterable = (
                    tqdm(descs, desc=f"{progress_bar_desc} (Ascend FULL)")
                    if is_global_first_rank()
                    else descs
                )
                for desc in iterable:
                    create_forward_fn(desc, warmup=True)(CUDAGraphMode.NONE)
                    torch.npu.synchronize(self.device)
                    forward_fn = create_forward_fn(desc, warmup=False)
                    if desc in self.graphs:
                        raise RuntimeError(f"Graph already captured for {desc}")
                    graph = torch.npu.NPUGraph()
                    get_offloader().sync_prev_onload()
                    free_before = None
                    if self._capture_mem_samples is not None:
                        torch.npu.synchronize(self.device)
                        free_before = torch.npu.mem_get_info(self.device)[0]
                    context = SimpleNamespace(
                        batch_descriptor=BatchDescriptor(
                            num_tokens=desc.num_tokens,
                            has_lora=self._capture_has_lora,
                            num_active_loras=desc.num_active_loras,
                        )
                    )
                    with (
                        self.runtime.capture_scope(context),
                        capture_pool(self.pool, self.vllm_config) as pool,
                        torch.npu.graph(graph, pool=pool, stream=capture_stream),
                    ):
                        self._capturing = True
                        try:
                            forward_fn(CUDAGraphMode.NONE)
                            get_offloader().join_after_forward()
                        finally:
                            self._capturing = False
                    if self._capture_mem_samples is not None:
                        torch.npu.synchronize(self.device)
                        self._capture_mem_samples.append(
                            free_before - torch.npu.mem_get_info(self.device)[0]
                        )
                    self.graphs[desc] = graph
                    self.capture_count += 1
                    compilation_counter.num_cudagraph_captured += 1
                    logger.info("Ascend V2 graph captured: %s", desc)
        finally:
            original_stream.wait_stream(capture_stream)
        self._graphs_captured = True

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        pcp_manager: "PCPManager | None" = None,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        self.runtime.bind_attention_groups(attn_groups)
        self._capture_has_lora = has_lora
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs
        if self.use_breakable_cg:
            self.init_breakable_cg_runner(model)

        if self.cudagraph_mode.has_piecewise_cudagraphs() and not (
            self.use_breakable_cg or has_compiled_submodule(model)
        ):
            raise RuntimeError(
                f"{type(model).__name__}: piecewise CUDA graphs "
                f"(cudagraph_mode={self.cudagraph_mode.name}) unavailable, "
                "model is not torch-compiled and breakable CUDA graph is off. "
                "Set VLLM_USE_BREAKABLE_CUDAGRAPH=1 or cudagraph_mode=NONE/FULL."
            )

        def store_capture_output(num_tokens: int, model_output: Any) -> None:
            """Copy outputs to persistent buffers, allocating on first use."""
            if self.is_last_pp_rank:
                # Last PP rank (common case).
                if self.use_aux_hidden_state_outputs:
                    hidden_states, aux_hidden_states = model_output
                else:
                    hidden_states = model_output
                    aux_hidden_states = []
                if self.hidden_states is None:
                    self.hidden_states = torch.empty_like(hidden_states)
                self.hidden_states[:num_tokens] = hidden_states
                if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
                    self.aux_hidden_states = [
                        torch.empty_like(x) for x in aux_hidden_states
                    ]
                for i, aux in enumerate(aux_hidden_states):
                    self.aux_hidden_states[i][:num_tokens] = aux
            else:
                # Non-last PP rank.
                assert isinstance(model_output, IntermediateTensors)
                intermediate_tensors = model_output
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = IntermediateTensors.empty_like(
                        intermediate_tensors
                    )
                for k, v in intermediate_tensors.tensors.items():
                    self.intermediate_tensors[k][:num_tokens] = v

        def create_forward_fn(
            desc: BatchExecutionDescriptor, warmup: bool
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)

            # Set LoRA state before capture so kernels see correct adapters.
            if lora_capture_hook is not None:
                lora_capture_hook(desc.num_active_loras, num_reqs, num_tokens)

            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )

            model_inputs = {
                "input_ids": input_buffers.input_ids[:num_tokens],
                "positions": input_buffers.positions[:num_tokens],
                "intermediate_tensors": None,
                **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
            }
            if not self.is_first_pp_rank:
                # Update for non-first PP ranks.
                model_inputs["input_ids"] = None
                model_inputs["inputs_embeds"] = None
                assert intermediate_tensors is not None
                model_inputs["intermediate_tensors"] = intermediate_tensors[:num_tokens]

            if desc.num_ubatches > 1:
                # Prepare and park threads before capture; finish runs inside it.
                assert self.ubatch_runner is not None
                ubatch_state = self.ubatch_runner.prepare(
                    InputBatch.make_dummy(num_reqs, num_tokens, input_buffers),
                    block_tables.get_dummy_block_tables(num_reqs),
                    block_tables.get_dummy_slot_mappings(num_tokens),
                    cg_mode=CUDAGraphMode.FULL,
                    for_capture=True,
                )
                # Capture with dummy rows marked as padding.
                input_buffers.is_padding.fill_(True)
                finish = self.ubatch_runner.begin_capturable_run(
                    model, model_inputs, ubatch_state, for_capture=True
                )

                def ubatch_forward_fn(cg_mode: CUDAGraphMode) -> None:
                    assert cg_mode != CUDAGraphMode.PIECEWISE, (
                        "DBO does not support PIECEWISE cudagraphs"
                    )
                    store_capture_output(num_tokens, finish())

                return ubatch_forward_fn

            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                full_cudagraph=desc.cg_mode == CUDAGraphMode.FULL,
                max_query_len=desc.max_query_len or desc.uniform_token_count,
                pcp_manager=pcp_manager,
            )

            # Capture with dummy rows marked as padding.
            input_buffers.is_padding.fill_(True)

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                batch_descriptor = BatchDescriptor(
                    num_tokens=num_tokens,
                    has_lora=has_lora,
                    num_active_loras=desc.num_active_loras,
                )
                with set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=cg_mode,
                    num_tokens_across_dp=num_tokens_across_dp,
                    slot_mapping=slot_mappings,
                    batch_descriptor=batch_descriptor,
                    is_padding=input_buffers.is_padding[:num_tokens],
                ):
                    context = get_forward_context()
                    context.capturing = self._capturing
                    if self._capturing:
                        self._capture_metadata[desc] = attn_metadata
                    if cg_mode == CUDAGraphMode.PIECEWISE:
                        # PIECEWISE graph (compiled PW or breakable, chosen inside
                        # run_pw_graph).
                        model_output = self.run_pw_graph(model, model_inputs)
                    else:
                        model_output = model(**model_inputs)

                if cg_mode == CUDAGraphMode.PIECEWISE:
                    # PW CUDA graph (compiled or breakable) internally handles the
                    # model outputs. No need to keep track of the hidden states.
                    return None

                store_capture_output(num_tokens, model_output)

            return forward_fn

        self._capture_graphs(create_forward_fn, progress_bar_desc)

    def run_fullgraph(self, desc):
        if self._pending_metadata is None:
            raise RuntimeError("Ascend V2 replay requires this step's FULL metadata")
        metadata = self._pending_metadata
        # Fail before enqueueing an external-event graph if a future metadata
        # change ever routes real prefill to this uniform-decode path.
        for layer_name, layer_metadata in metadata.items():
            state = getattr(layer_metadata, "attn_state", None)
            if state is not None and state.name != "DecodeOnly":
                raise ValueError(
                    f"Ascend V2 decode graph cannot replay {layer_name}: {state.name}"
                )
        self._pending_metadata = None
        context = SimpleNamespace(
            batch_descriptor=BatchDescriptor(
                num_tokens=desc.num_tokens,
                has_lora=self._capture_has_lora,
                num_active_loras=desc.num_active_loras,
            ),
            attn_metadata=metadata,
        )
        with self.runtime.replay_scope(context):
            output = super().run_fullgraph(desc)
        # Keep cross-stream task-update inputs alive until the next replay's
        # synchronization, even if their metadata object was freshly allocated.
        self._last_replay_metadata = metadata
        self.replay_count += 1
        self.replay_counts[desc] = self.replay_counts.get(desc, 0) + 1
        logger.debug("Ascend V2 graph replayed: %s (count=%d)", desc, self.replay_count)
        return output

    def graph_statistics(self):
        """Return direct capture/replay evidence for worker collective RPCs."""
        return {
            "manager_type": type(self).__name__,
            "capture_count": self.capture_count,
            "replay_count": self.replay_count,
            "graphs": [
                {
                    "num_tokens": desc.num_tokens,
                    "num_reqs": desc.num_reqs,
                    "graph_type": f"{type(graph).__module__}.{type(graph).__name__}",
                    "replay_count": self.replay_counts.get(desc, 0),
                }
                for desc, graph in self.graphs.items()
            ],
        }

    def release_graphs(self):
        torch.npu.synchronize(self.device)
        super().release_graphs()
        self._capture_metadata.clear()
        self.replay_counts.clear()
        self._pending_metadata = None
        self._last_replay_metadata = None
        self.runtime.clear_decoder_graphs()
