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

"""Coverage tests for the small ModelRunnerFL helpers of
vllm_fl/worker/model_runner.py: lifecycle caches (update_max_model_len,
reset_mm_cache, reset_encoder_cache, init_fp8_kv_scales fast path),
position selection (_get_positions), pooling model kwargs
(_init_model_kwargs), the async copy stream accessor, and encoder
sequence lens.
"""

from types import SimpleNamespace

import numpy as np
import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import make_runner, mr_module
from vllm_fl.worker.model_runner import ModelRunnerFL


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


def _spec():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )


def test_update_max_model_len_propagates_with_and_without_draft_config():
    """max_model_len moves with and without a drafter limit tracking it."""
    runner, patches = make_runner()
    try:
        # No speculative config: only the runner field moves.
        runner.speculative_config = None
        runner.update_max_model_len(256)
        assert runner.max_model_len == 256
        assert not hasattr(runner, "effective_drafter_max_model_len")

        # Speculative config with no draft max len: the drafter limit
        # tracks the new max.
        runner.speculative_config = SimpleNamespace(
            draft_model_config=SimpleNamespace(max_model_len=None)
        )
        runner.update_max_model_len(512)
        assert runner.max_model_len == 512
        assert runner.effective_drafter_max_model_len == 512

        # Draft config already pins its own limit: the drafter field is
        # left untouched.
        runner.speculative_config = SimpleNamespace(
            draft_model_config=SimpleNamespace(max_model_len=64)
        )
        runner.update_max_model_len(1024)
        assert runner.max_model_len == 1024
        assert runner.effective_drafter_max_model_len == 512
    finally:
        patches.__exit__(None, None, None)


def test_reset_mm_cache_and_reset_encoder_cache_clear_their_stores():
    """Cache resets clear the mm budget, late runner, and encoder cache."""
    runner, patches = make_runner()
    try:
        cleared = {"budget": 0, "late": 0}
        runner.mm_budget = SimpleNamespace(
            reset_cache=lambda: cleared.__setitem__("budget", cleared["budget"] + 1)
        )
        runner.late_interaction_runner = SimpleNamespace(
            clear=lambda: cleared.__setitem__("late", cleared["late"] + 1)
        )
        runner.encoder_cache["h1"] = torch.zeros(1)
        runner.encoder_cache["h2"] = torch.zeros(1)

        runner.reset_mm_cache()
        assert cleared == {"budget": 1, "late": 1}

        runner.reset_encoder_cache()
        # Encoder cache emptied and the late-interaction runner cleared
        # again.
        assert runner.encoder_cache == {}
        assert cleared == {"budget": 1, "late": 2}
    finally:
        patches.__exit__(None, None, None)


def test_init_fp8_kv_scales_skips_non_quantized_cache():
    """Auto (non-fp8) cache dtype: the method is a no-op and returns None."""
    runner, patches = make_runner()
    try:
        # cache_config.cache_dtype 'auto' is set by the construction recipe.
        assert runner.cache_config.cache_dtype == "auto"
        # __init__ creates kv_caches as an empty list: with a non-quantized
        # dtype the method returns before touching it, and no scale
        # attributes appear on the (absent) attention layers.
        assert runner.kv_caches == []
        assert runner.init_fp8_kv_scales() is None
    finally:
        patches.__exit__(None, None, None)


def test_get_positions_int_and_slice_selectors():
    """int and slice selectors read the right mrope columns."""
    from ._imports import (
        _model_config,
        _vllm_config,
    )

    config = _vllm_config(_model_config(uses_mrope=True))
    runner, patches = make_runner(config)
    try:
        runner.mrope_positions.cpu.zero_()
        runner.mrope_positions.cpu[:, 10:] = 7
        runner.mrope_positions.copy_to_gpu(runner.mrope_positions.cpu.shape[1])

        # int selector: the first num_tokens columns.
        got = runner._get_positions(5)
        torch.testing.assert_close(got, runner.mrope_positions.gpu[:, :5])
        assert got.shape == (3, 5)

        # slice selector: arbitrary column windows.
        got = runner._get_positions(slice(8, 12))
        torch.testing.assert_close(got, runner.mrope_positions.gpu[:, 8:12])
        assert got[:, :2].eq(0).all()
        assert got[:, 2:].eq(7).all()

        # A generate-only text model falls back to the flat positions
        # buffer for both selector kinds.
        runner.uses_mrope = False
        runner.uses_xdrope_dim = 0
        runner.positions[:4] = torch.tensor([4, 5, 6, 7])
        got = runner._get_positions(4)
        torch.testing.assert_close(got, torch.tensor([4, 5, 6, 7]))
        got = runner._get_positions(slice(1, 3))
        torch.testing.assert_close(got, torch.tensor([5, 6]))
    finally:
        patches.__exit__(None, None, None)


def test_get_positions_xdrope_dim_selects_four_dim_buffer():
    """XD-RoPE configs read positions from the four-dimension buffer."""
    from ._imports import (
        _model_config,
        _vllm_config,
    )

    config = _vllm_config(_model_config(uses_xdrope_dim=4))
    runner, patches = make_runner(config)
    try:
        runner.uses_mrope = False
        runner.uses_xdrope_dim = 4
        runner.xdrope_positions.cpu.zero_()
        runner.xdrope_positions.cpu[:, 3] = 11
        runner.xdrope_positions.copy_to_gpu(runner.xdrope_positions.cpu.shape[1])
        got = runner._get_positions(5)
        assert got.shape == (4, 5)
        assert got[:, 3].eq(11).all()
        got = runner._get_positions(slice(3, 4))
        assert got.shape == (4, 1)
        assert got[:, 0].eq(11).all()
    finally:
        patches.__exit__(None, None, None)


def test_init_model_kwargs_empty_for_non_pooling():
    """Generate models produce no pooling model kwargs."""
    runner, patches = make_runner()
    try:
        assert not runner.is_pooling_model
        assert runner._init_model_kwargs() == {}
    finally:
        patches.__exit__(None, None, None)


def test_init_model_kwargs_builds_compressed_token_type_ids():
    """Pooling models with compressed_token_type_ids get the CPU tensor."""
    from vllm.pooling_params import PoolingParams

    from ._imports import (
        _model_config,
        _vllm_config,
    )

    config = _vllm_config(_model_config(runner_type="pooling"))
    runner, patches = make_runner(config)
    try:
        assert runner.is_pooling_model

        def _state(req_id, prompt_len, pooling_params):
            return SimpleNamespace(
                req_id=req_id,
                prompt_token_ids=list(range(prompt_len)),
                prompt_embeds=None,
                prompt_is_token_ids=None,
                mm_features=[],
                sampling_params=None,
                pooling_params=pooling_params,
                pooling_states=SimpleNamespace(),
                generator=None,
                block_ids=([0],),
                num_computed_tokens=0,
                num_tokens=prompt_len,
                output_token_ids=[],
                lora_request=None,
            )

        # Two pooling requests: r0 wants compressed ids from position 1,
        # r1 has none.
        runner.input_batch.add_request(
            _state(
                "r0",
                2,
                PoolingParams(extra_kwargs={"compressed_token_type_ids": 1}),
            )
        )
        runner.input_batch.add_request(_state("r1", 3, PoolingParams()))
        assert runner.input_batch.num_reqs == 2
        assert runner.input_batch.req_ids == ["r0", "r1"]
        # Optimistic lengths 2 and 3 for the two requests.
        runner.optimistic_seq_lens_cpu[:2] = torch.tensor([2, 3])

        # PIN_MEMORY (pinned-host allocation capability probe) is True on
        # this CPU-only host's UnspecifiedPlatform but torch cannot pin
        # without an accelerator; force the unpinned path locally.
        with _Patch(mr_module, "PIN_MEMORY", False):
            model_kwargs = runner._init_model_kwargs()
        assert set(model_kwargs) == {"token_type_ids"}
        ids = model_kwargs["token_type_ids"]
        assert ids.dtype == torch.int32
        # Request 0: [0, 1] (positions >= 1 marked); request 1: [0, 0, 0].
        assert ids.tolist() == [0, 1, 0, 0, 0]
    finally:
        patches.__exit__(None, None, None)


def test_get_or_create_async_output_copy_stream_caches_one_stream():
    """The async output copy stream is created once and cached."""
    runner, patches = make_runner()
    try:
        assert runner.async_output_copy_stream is None
        with _Patch(
            mr_module.current_platform,
            "torch_device_fn",
            SimpleNamespace(
                Stream=staticmethod(lambda: "the-stream"),
            ),
        ):
            first = runner._get_or_create_async_output_copy_stream()
            assert first == "the-stream"
            # Second call reuses the cached stream (Stream not called
            # again would be unobservable, but the identity holds).
            assert runner._get_or_create_async_output_copy_stream() is first
        assert runner.async_output_copy_stream is first
    finally:
        patches.__exit__(None, None, None)


def test_get_encoder_seq_lens_non_cross_attention_short_circuits():
    """Non-cross-attention specs skip encoder seq-len computation."""
    runner = SimpleNamespace()
    spec = _spec()  # FullAttentionSpec, not CrossAttentionSpec
    out = ModelRunnerFL._get_encoder_seq_lens.__get__(runner, ModelRunnerFL)(
        {"r0": 2}, spec, 1
    )
    assert out == (None, None)


def test_get_encoder_seq_lens_sums_feature_lengths():
    """Cross-attention specs sum feature lengths into the buffer."""
    from vllm.v1.kv_cache_interface import CrossAttentionSpec

    runner, patches = make_runner()
    try:
        spec = CrossAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
        )
        # r0: two features (lengths 3 and 4); r1: no features at all.
        runner.requests["r0"] = SimpleNamespace(
            mm_features=[
                SimpleNamespace(mm_position=SimpleNamespace(length=3)),
                SimpleNamespace(mm_position=SimpleNamespace(length=4)),
            ]
        )
        runner.requests["r1"] = SimpleNamespace(mm_features=None)
        runner.input_batch.req_id_to_index = {"r0": 0, "r1": 1}

        lens, lens_cpu = ModelRunnerFL._get_encoder_seq_lens.__get__(
            runner, ModelRunnerFL
        )({"r0": 1, "r1": 1}, spec, 2)
        assert lens.tolist() == [7, 0]
        np.testing.assert_array_equal(lens_cpu, [7, 0])
        # The buffer itself was updated and copied to the gpu view.
        assert runner.encoder_seq_lens.np[:2].tolist() == [7, 0]
        assert runner.encoder_seq_lens.gpu[:2].tolist() == [7, 0]

        # Capture mode overwrites every row with the max encoder len.
        runner.model_config.hf_config = SimpleNamespace(max_source_positions=9)
        lens, _ = ModelRunnerFL._get_encoder_seq_lens.__get__(runner, ModelRunnerFL)(
            {"r0": 1}, spec, 2, for_cudagraph_capture=True
        )
        assert lens.tolist() == [9, 9]
    finally:
        patches.__exit__(None, None, None)
