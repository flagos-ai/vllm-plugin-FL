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

"""Coverage tests for ModelRunnerFL.__init__ buffer and config plumbing.

Construction is driven through the real class on CPU (the shared shim's
recipe); every test asserts derived values — buffer shapes/dtypes, config
propagation, spec-token counts — rather than merely surviving the call.
"""

import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import make_runner
from vllm_fl.worker.common_attention_metadata import (
    CommonAttentionMetadataGraphRunner,
)


def test_init_propagates_scheduler_and_model_config():
    """Scheduler and model config values propagate to runner fields."""
    runner, patches = make_runner()
    try:
        assert runner.max_num_tokens == 64
        assert runner.max_num_reqs == 4
        assert runner.max_model_len == 128
        assert runner.is_pooling_model is False
        assert runner.enable_prompt_embeds is False
        assert runner.pin_memory is False
        assert runner.dtype == torch.float16
        # cache_dtype="auto" + fp16 model dtype keeps the KV cache fp16.
        assert runner.kv_cache_dtype == torch.float16
        assert runner.calculate_kv_scales is False
        assert runner.dcp_world_size == 1
        assert runner.dcp_rank == 0
        assert runner.num_query_heads == 4
        assert runner.inputs_embeds_size == 16
        assert runner.num_spec_tokens == 0
        assert runner.uniform_decode_query_len == 1
        assert runner.use_async_scheduling is False
        assert runner.use_async_spec_decode is False
    finally:
        patches.__exit__(None, None, None)


def test_init_allocates_persistent_buffers_with_exact_shapes():
    """Persistent buffers are allocated with the derived shapes and dtypes."""
    runner, patches = make_runner()
    try:
        from vllm.v1.worker.gpu_input_batch import InputBatch

        assert isinstance(runner.input_batch, InputBatch)
        assert runner.input_ids.cpu.shape == (64,)
        assert runner.input_ids.cpu.dtype == torch.int32
        assert runner.positions.shape == (64,)
        assert runner.positions.dtype == torch.int64
        assert runner.query_start_loc.cpu.shape == (5,)
        assert runner.seq_lens.shape == (4,)
        assert runner.num_computed_tokens.shape == (4,)
        assert runner.req_indices.cpu.shape == (64,)
        assert runner.req_indices.cpu.dtype == torch.int64
        assert runner.prev_positions.cpu.shape == (4,)
        assert runner.num_scheduled_tokens.cpu.shape == (4,)
        assert runner.encoder_seq_lens.cpu.shape == (4,)
        assert runner.inputs_embeds.cpu.shape == (64, 16)
        assert runner.inputs_embeds.cpu.dtype == torch.float16
        assert runner.is_token_ids.cpu.shape == (64,)
        assert runner.is_token_ids.cpu.dtype == torch.bool
        assert runner.discard_request_mask.cpu.shape == (4,)
        assert runner.num_decode_draft_tokens.cpu.shape == (4,)
        assert runner.num_accepted_tokens.cpu.shape == (4,)
        # No mrope/xdrope buffers for a plain generate config.
        assert not hasattr(runner, "mrope_positions")
        assert not hasattr(runner, "xdrope_positions")
        # arange cache covers both max_num_reqs+1 and max_num_tokens.
        assert runner.arange_np.shape == (64,)
        assert runner.arange_np[0] == 0
        assert runner.arange_np[-1] == 63
        assert runner.query_pos.cpu.shape == (64,)
    finally:
        patches.__exit__(None, None, None)


def test_init_uses_cudagraph_off_configuration():
    """CUDAGraph-off config leaves graph sizes empty and no copy stream."""
    runner, patches = make_runner()
    try:
        assert runner.cudagraph_batch_sizes == []
        assert isinstance(
            runner.common_attention_metadata_graph,
            CommonAttentionMetadataGraphRunner,
        )
        # Device properties come from the pinned probe.
        assert runner.num_sms == 8
        # async scheduling off: no copy stream or event.
        assert runner.async_output_copy_stream is None
        assert runner.prepare_inputs_event is None
        assert runner.broadcast_pp_output is False
        assert runner.attn_groups == []
        assert runner.encoder_cache == {}
        assert runner.late_interaction_runner is not None
    finally:
        patches.__exit__(None, None, None)


def test_init_allocates_sampled_token_ids_pinned_cpu():
    """The sampled-id CPU buffer is int64 with one column per request slot."""
    runner, patches = make_runner()
    try:
        buf = runner.sampled_token_ids_pinned_cpu
        assert buf.shape == (4, 1)
        # CUDA uses int64 for sampled ids (device_type is not npu).
        assert buf.dtype == torch.int64
        assert buf.device.type == "cpu"
        # mamba "all" mode with no spec tokens: no prev-last buffer.
        assert runner.mamba_prev_last_scheduled_idx is None
    finally:
        patches.__exit__(None, None, None)


def test_init_encoder_decoder_config_uses_encoder_len():
    """Encoder-decoder models carry the max encoder length alongside text."""
    from ._imports import _model_config, _vllm_config

    config = _vllm_config(
        model_config=_model_config(is_encoder_decoder=True),
        scheduler_config=_NS(
            max_num_batched_tokens=64,
            max_num_seqs=4,
            max_num_encoder_input_tokens=32,
            async_scheduling=False,
            enable_chunked_prefill=True,
        ),
    )
    runner, patches = make_runner(config)
    try:
        assert runner.max_encoder_len == 32
        # max(input_len=128, encoder_len=32) stays 128 in the input batch.
        assert runner.input_batch.max_model_len == 128
    finally:
        patches.__exit__(None, None, None)


def test_init_async_scheduling_creates_copy_stream_and_event():
    """Async scheduling allocates the copy stream and prepare-inputs event."""
    from types import SimpleNamespace as NS

    from ._imports import _vllm_config

    config = _vllm_config(
        scheduler_config=NS(
            max_num_batched_tokens=64,
            max_num_seqs=4,
            max_num_encoder_input_tokens=0,
            async_scheduling=True,
            enable_chunked_prefill=True,
        )
    )
    runner, patches = make_runner(config)
    try:
        assert runner.use_async_scheduling is True
        assert runner.async_output_copy_stream is not None
        assert runner.prepare_inputs_event is not None
        assert runner.num_spec_tokens == 0
        assert runner.use_async_spec_decode is False
    finally:
        patches.__exit__(None, None, None)


def test_init_mrope_and_xdrope_buffers():
    """M-RoPE/XD-RoPE configs allocate their (dim, N) position buffers."""
    from ._imports import _model_config, _vllm_config

    mrope_runner, mpatches = make_runner(_vllm_config(_model_config(uses_mrope=True)))
    try:
        assert mrope_runner.mrope_positions.cpu.shape == (3, 65)
        assert mrope_runner.mrope_positions.cpu.dtype == torch.int64
    finally:
        mpatches.__exit__(None, None, None)

    xd_runner, xpatches = make_runner(_vllm_config(_model_config(uses_xdrope_dim=4)))
    try:
        assert xd_runner.xdrope_positions.cpu.shape == (4, 65)
        assert xd_runner.xdrope_positions.cpu.dtype == torch.int64
        assert not hasattr(xd_runner, "mrope_positions")
    finally:
        xpatches.__exit__(None, None, None)


def _NS(**kw):  # helper used above
    from types import SimpleNamespace

    return SimpleNamespace(**kw)
