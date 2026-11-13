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

"""Coverage tests for ModelRunnerFL._preprocess.

``_preprocess`` runs against real CPU-constructed runners (the area
shim's recipe) across its three input branches: text-only token ids,
prompt-embeds token/embed splicing, and the encoder-decoder input path
(the encoder itself is a recorded input, CONTRACT R2).
"""

from types import SimpleNamespace

import numpy as np
import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import make_runner, make_scheduler_output
from vllm_fl.worker.model_runner import ModelRunnerFL


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
        scheduled_encoder_inputs=kw.get("encoder_inputs", {}),
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _prime(runner, prompts):
    """Add requests through the state machine; return a SchedulerOutput."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    new_reqs = []
    for rid, prompt in prompts.items():
        new_reqs.append(
            NewRequestData(
                req_id=rid,
                prompt_token_ids=prompt,
                mm_features=[],
                sampling_params=SamplingParams(),
                pooling_params=None,
                block_ids=([0],),
                num_computed_tokens=0,
                lora_request=None,
            )
        )
    out = _sched(
        new_reqs=new_reqs,
        num_scheduled_tokens={rid: 0 for rid in prompts},
        total=0,
    )
    runner._update_states(out)
    return out


class _StateModel:
    """Stand-in for the loaded model's embed_input_ids."""

    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.calls = []

    def embed_input_ids(self, input_ids, **kwargs):
        self.calls.append(input_ids.tolist())
        # The runner's inputs_embeds buffer is float16 (self.dtype).
        return torch.full(
            (input_ids.shape[0], self.embed_dim), 2.0, dtype=torch.float16
        )


def test_preprocess_text_only_returns_token_ids_and_kwargs():
    """Text-only prefill returns gathered ids, positions, and kwargs."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [11, 12, 13]})
        out = _sched(num_scheduled_tokens={"a": 3}, total=3)
        runner._prepare_inputs(out, np.array([3], dtype=np.int32))
        # Prime the state the preprocess contract expects.
        runner.supports_mm_inputs = False
        runner.enable_prompt_embeds = False
        runner.model_state = SimpleNamespace(
            prepare_inputs=lambda num_reqs, padded, ids: {"n_reqs": num_reqs}
        )

        (
            input_ids,
            inputs_embeds,
            positions,
            inter,
            model_kwargs,
            ec_out,
        ) = ModelRunnerFL._preprocess(runner, out, 3, 1, 1)
        assert input_ids is not None
        assert input_ids.tolist() == [11, 12, 13]
        assert inputs_embeds is None
        assert positions.tolist() == [0, 1, 2]
        assert inter is None
        assert ec_out is None
        # Model-state kwargs merged into the base dict.
        assert model_kwargs["n_reqs"] == 1
    finally:
        patches.__exit__(None, None, None)


def test_preprocess_prompt_embeds_splices_token_embeds():
    """Prompt-embeds requests splice model embeddings over precomputed rows."""
    from ._imports import (
        _model_config,
        _vllm_config,
    )

    config = _vllm_config(_model_config(enable_prompt_embeds=True))
    runner, patches = make_runner(config)
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [7, 8]})
        out = _sched(num_scheduled_tokens={"a": 2}, total=2)
        runner._prepare_inputs(out, np.array([2], dtype=np.int32))
        runner.supports_mm_inputs = False
        runner.enable_prompt_embeds = True
        model = _StateModel(16)
        runner.model = model
        runner.model_state = SimpleNamespace(
            prepare_inputs=lambda num_reqs, padded, ids: {}
        )

        # Token 0 is a real id; token 1 carries a precomputed embed.
        runner.is_token_ids.cpu[:2] = torch.tensor([True, False])
        runner.inputs_embeds.cpu[1] = torch.arange(16, dtype=torch.float16)
        runner.inputs_embeds.copy_to_gpu(2)
        runner.is_token_ids.copy_to_gpu(2)

        _, inputs_embeds, positions, _, _, _ = ModelRunnerFL._preprocess(
            runner, out, 2, 1, 1
        )
        # The token-id position was re-embedded with the model's output.
        torch.testing.assert_close(
            inputs_embeds[0], torch.full((16,), 2.0, dtype=torch.float16)
        )
        # The embed position keeps the precomputed row.
        torch.testing.assert_close(
            inputs_embeds[1], torch.arange(16, dtype=torch.float16)
        )
        # Only the token id reached the embedding call.
        assert model.calls == [[7]]
        assert positions.tolist() == [0, 1]
    finally:
        patches.__exit__(None, None, None)


def test_preprocess_encoder_decoder_runs_encoder_and_merges_kwargs():
    """Encoder-decoder prefill runs the encoder and merges its outputs."""
    from ._imports import (
        _model_config,
        _vllm_config,
    )

    config = _vllm_config(_model_config(is_encoder_decoder=True))
    runner, patches = make_runner(config)
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [3, 4]})
        out = _sched(
            num_scheduled_tokens={"a": 2},
            total=2,
            encoder_inputs={"a": [0]},
        )
        runner._prepare_inputs(out, np.array([2], dtype=np.int32))
        runner.supports_mm_inputs = False
        runner.enable_prompt_embeds = False
        runner.model_state = SimpleNamespace(
            prepare_inputs=lambda num_reqs, padded, ids: {}
        )
        # The encoder is an input: a recorder returning one output.
        encoder_outputs = [torch.full((2, 4), 9.0)]
        calls = {}

        def _execute_mm_encoder(sched):
            calls["sched"] = sched
            return encoder_outputs

        runner._execute_mm_encoder = _execute_mm_encoder

        (
            input_ids,
            inputs_embeds,
            positions,
            _,
            model_kwargs,
            _,
        ) = ModelRunnerFL._preprocess(runner, out, 2, 1, 1)
        assert input_ids.tolist() == [3, 4]
        # The encoder ran on the same scheduler output.
        assert calls["sched"] is out
        assert model_kwargs["encoder_outputs"] is encoder_outputs
        assert positions.tolist() == [0, 1]
    finally:
        patches.__exit__(None, None, None)
