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

"""Coverage tests for the remaining ModelRunnerFL._prepare_inputs branches.

The core gather/positions/logits layout is covered in
test_cov_prepare_inputs.py; this file drives the optional sub-paths:
prompt-embed row splicing (including the skip cases), M-RoPE/XD-RoPE
buffer upload, CPU accepted-token synchronization (plain and async
batch-reorder), the async spec-decode kernel dispatch, and the
async-scheduling draft scatter of _prepare_input_ids.

Buffers are the real CpuGpuBuffer instances from the CPU construction
recipe; device kernels that cannot run on a CPU tensor are boundaries
replaced with recorders (CONTRACT R2).
"""

from types import SimpleNamespace

import numpy as np
import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import (
    _model_config,
    _vllm_config,
    make_runner,
    make_scheduler_output,
    mr_module,
)


class _Patch:
    """Deterministic set/restore of a module attribute (input boundary)."""

    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


class _CpuEvent:
    """torch.Event stand-in; the cpu backend supports no real events."""

    def synchronize(self):
        pass

    def record(self):
        pass


def _arm_kv_cache(runner):
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

    runner.kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([], None)],  # type: ignore[arg-type]
    )
    runner.attn_groups = []


def _sched(**kw):
    from vllm.v1.core.sched.output import CachedRequestData

    return make_scheduler_output(
        scheduled_new_reqs=kw.get("new_reqs", []),
        scheduled_cached_reqs=kw.get("cached", CachedRequestData.make_empty()),
        num_scheduled_tokens=kw.get("num_scheduled_tokens", {}),
        total_num_scheduled_tokens=kw.get(
            "total", sum(kw.get("num_scheduled_tokens", {}).values())
        ),
        scheduled_spec_decode_tokens=kw.get("spec_tokens", {}),
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _new_req(req_id, prompt):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=SamplingParams(),
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=0,
        lora_request=None,
    )


def _prime(runner, prompts):
    out = _sched(
        new_reqs=[_new_req(rid, prompt) for rid, prompt in prompts.items()],
        num_scheduled_tokens={rid: 0 for rid in prompts},
        total=0,
    )
    runner._update_states(out)
    return out


def test_prepare_inputs_splices_prompt_embeds_for_scheduled_window():
    """Scheduled prompt-embed rows are copied into the inputs_embeds buffer."""
    runner, patches = make_runner(
        _vllm_config(_model_config(enable_prompt_embeds=True))
    )
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [1, 2, 3]})
        embeds = torch.arange(32, dtype=torch.float16).reshape(2, 16)
        runner.input_batch.req_prompt_embeds = {0: embeds}

        out = _sched(num_scheduled_tokens={"a": 3}, total=3)
        runner._prepare_inputs(out, np.array([3], dtype=np.int32))

        # start_pos 0, scheduled 3, only 2 embeds exist: rows 0..1 copied.
        torch.testing.assert_close(runner.inputs_embeds.cpu[:2], embeds)
        # The third slot (a pure token position) was not touched.
        assert runner.inputs_embeds.cpu[2].abs().sum().item() == 0
    finally:
        patches.__exit__(None, None, None)


def test_prepare_inputs_skips_prompt_embeds_past_their_extent():
    """A continued request past the embed extent copies nothing."""
    runner, patches = make_runner(
        _vllm_config(_model_config(enable_prompt_embeds=True))
    )
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [1, 2, 3]})
        embeds = torch.ones(2, 16, dtype=torch.float16)
        runner.input_batch.req_prompt_embeds = {0: embeds}
        runner.input_batch.num_computed_tokens_cpu[0] = 3
        runner.inputs_embeds.cpu[:1].fill_(5.0)

        out = _sched(num_scheduled_tokens={"a": 1}, total=1)
        runner._prepare_inputs(out, np.array([1], dtype=np.int32))

        # start_pos 3 >= embed rows 2: the pre-existing row is preserved.
        assert runner.inputs_embeds.cpu[0].eq(5).all()
    finally:
        patches.__exit__(None, None, None)


class _MRopeModel:
    """Fake model publishing deterministic M-RoPE prompt positions."""

    def __init__(self, rows):
        self.rows = rows

    def get_mrope_input_positions(self, input_tokens, mm_features):
        return self.rows, 0


def test_prepare_inputs_uploads_mrope_positions_to_device_buffer():
    """The computed M-RoPE positions are copied into the gpu view."""
    runner, patches = make_runner(_vllm_config(_model_config(uses_mrope=True)))
    try:
        _arm_kv_cache(runner)
        rows = torch.arange(9, dtype=torch.int64).reshape(3, 3)
        runner.model = _MRopeModel(rows)
        with _Patch(mr_module, "supports_mrope", lambda m: True):
            _prime(runner, {"a": [7, 8, 9]})
        runner.input_batch.num_computed_tokens_cpu[0] = 0

        out = _sched(num_scheduled_tokens={"a": 3}, total=3)
        runner._prepare_inputs(out, np.array([3], dtype=np.int32))

        # Prompt-only step: the request positions are copied verbatim into
        # the first three device columns.
        torch.testing.assert_close(runner.mrope_positions.gpu[:, :3], rows)
    finally:
        patches.__exit__(None, None, None)


class _XDRopeModel:
    """Fake model publishing deterministic XD-RoPE prompt positions."""

    def __init__(self, rows):
        self.rows = rows

    def get_xdrope_input_positions(self, input_tokens, mm_features):
        return self.rows


def test_prepare_inputs_uploads_xdrope_positions_to_device_buffer():
    """The computed XD-RoPE positions are copied into the gpu view."""
    runner, patches = make_runner(_vllm_config(_model_config(uses_xdrope_dim=4)))
    try:
        _arm_kv_cache(runner)
        rows = torch.arange(8, dtype=torch.int64).reshape(4, 2)
        runner.model = _XDRopeModel(rows)
        with _Patch(mr_module, "supports_xdrope", lambda m: True):
            _prime(runner, {"a": [4, 5]})
        runner.input_batch.num_computed_tokens_cpu[0] = 0

        out = _sched(num_scheduled_tokens={"a": 2}, total=2)
        runner._prepare_inputs(out, np.array([2], dtype=np.int32))

        torch.testing.assert_close(runner.xdrope_positions.gpu[:, :2], rows)
    finally:
        patches.__exit__(None, None, None)


def test_prepare_inputs_syncs_accepted_counts_from_cpu():
    """A completed accepted-token event copies the CPU counts to the device."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [1, 2]})
        runner.num_accepted_tokens_event = _CpuEvent()
        runner.input_batch.num_accepted_tokens_cpu[:] = [3, 2, 1, 0]

        out = _sched(num_scheduled_tokens={"a": 1}, total=1)
        runner._prepare_inputs(out, np.array([1], dtype=np.int32))

        # Live rows mirror the CPU values; padding rows default to one.
        assert runner.num_accepted_tokens.np.tolist() == [3, 1, 1, 1]
        assert runner.num_accepted_tokens.gpu.tolist() == [3, 1, 1, 1]
    finally:
        patches.__exit__(None, None, None)


def test_prepare_inputs_async_align_gathers_accepted_counts_by_prev_row():
    """Async scheduling with align mode reorders accepted counts via prev rows."""
    cache_config = SimpleNamespace(
        cache_dtype="auto",
        calculate_kv_scales=False,
        block_size=16,
        kv_sharing_fast_prefill=False,
        mamba_cache_mode="align",
    )
    scheduler_config = SimpleNamespace(
        max_num_batched_tokens=64,
        max_num_seqs=4,
        max_num_encoder_input_tokens=0,
        async_scheduling=True,
        enable_chunked_prefill=False,
    )
    runner, patches = make_runner(
        _vllm_config(cache_config=cache_config, scheduler_config=scheduler_config)
    )
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [1, 2], "b": [3, 4]})
        runner.num_accepted_tokens_event = _CpuEvent()
        runner.input_batch.num_accepted_tokens_cpu[:2] = [5, 6]
        # "a" was previous row 1; "b" is new (no previous row).
        runner.input_batch.prev_req_id_to_index = {"a": 1}

        out = _sched(num_scheduled_tokens={"a": 1, "b": 1}, total=2)
        runner._prepare_inputs(out, np.array([1, 1], dtype=np.int32))

        # Row 0 resolves through prev row 1 -> 6; row 1 is new -> 1.
        assert runner.num_accepted_tokens.np[:2].tolist() == [6, 1]
        assert runner.input_batch.num_accepted_tokens_cpu[:2].tolist() == [6, 1]
        # Padding rows are reset to one.
        assert runner.num_accepted_tokens.np[2:].tolist() == [1, 1]
    finally:
        patches.__exit__(None, None, None)


def test_prepare_inputs_async_spec_decode_dispatches_correction_kernel():
    """Async spec decode dispatches the GPU batch-change correction."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [1, 2]})
        runner.use_async_spec_decode = True
        valid = torch.tensor([2], dtype=torch.int32)
        runner.valid_sampled_token_count_gpu = valid
        runner.input_batch.prev_req_id_to_index = {"a": 0}
        runner.prev_num_draft_tokens.np[0] = 1
        runner.input_batch.num_computed_tokens_cpu[0] = 2
        calls = []

        def _kernel(*args):
            calls.append(args)

        with _Patch(
            mr_module,
            "update_num_computed_tokens_for_batch_change",
            _kernel,
        ):
            out = _sched(num_scheduled_tokens={"a": 1}, total=1)
            runner._prepare_inputs(out, np.array([1], dtype=np.int32))

        assert len(calls) == 1
        (
            num_computed,
            accepted_gpu,
            prev_positions_gpu,
            valid_gpu,
            prev_drafts_gpu,
            cpu_values,
        ) = calls[0]
        assert num_computed is runner.num_computed_tokens
        torch.testing.assert_close(accepted_gpu, runner.num_accepted_tokens.gpu[:1])
        assert prev_positions_gpu.tolist() == [0]
        assert valid_gpu is valid
        assert prev_drafts_gpu[:1].tolist() == [1]
        assert cpu_values.tolist() == [2]
        # The kernel is the only writer of the device counts in this path:
        # the plain CPU copy branch must not have run.
        assert runner.num_computed_tokens.tolist() == [0, 0, 0, 0]
    finally:
        patches.__exit__(None, None, None)


def test_prepare_input_ids_async_scatters_spec_draft_tokens():
    """Draft tokens are scattered after the sampled-token scatter."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [10], "b": [20]})
        out = _sched(
            num_scheduled_tokens={"a": 2, "b": 1},
            total=3,
            spec_tokens={"a": [99]},
        )
        runner._update_states(out)

        # Async state: both requests continue from previous rows 0 and 1.
        runner.input_batch.prev_sampled_token_ids = torch.tensor(
            [[7], [8]], dtype=torch.int32
        )
        runner.prev_num_spec_tokens = 1
        runner.prev_positions.np[:2] = [0, 1]
        runner.input_batch.prev_req_id_to_index = {"a": 0, "b": 1}
        # "a" carries one draft token from the previous step.
        runner._draft_token_ids = torch.tensor([[41], [42]], dtype=torch.int64)
        runner.input_ids.cpu[:3] = torch.tensor([0, 0, 0], dtype=torch.int32)

        cu = np.array([2, 3], dtype=np.int64)
        runner._prepare_input_ids(out, 2, 3, cu)

        # Sampled tokens land on each request's last scheduled slot; the
        # previous step's draft token lands on a's earlier slot.
        assert runner.input_ids.gpu[:3].tolist() == [7, 41, 8]
    finally:
        patches.__exit__(None, None, None)
