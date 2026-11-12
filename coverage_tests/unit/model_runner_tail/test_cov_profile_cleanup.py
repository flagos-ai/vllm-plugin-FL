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

"""Coverage tests for ModelRunnerFL.profile_run and
_cleanup_profiling_kv_cache (vllm_fl/worker/model_runner.py).

profile_run branches covered on CPU: the multimodal skip branches
(``skip_mm_profiling`` and the embedding-only budget with no encoder), the
``is_profile`` / last-rank pooling-vs-sampler dispatch, device sync,
encoder-cache clearing and gc. The real encoder run
(``_get_mm_dummy_batch`` + ``embed_multimodal`` + sanity check) needs a real
multimodal model and is left to the A100 image; the full CUDA-graph
profiling pipeline around it is unreachable on CPU.

_cleanup_profiling_kv_cache runs its whole state-reset body with a recorder
runner; ``_accelerator_synchronize`` / ``torch.accelerator.empty_cache``
are device probes patched at module boundaries (both raise "Cannot access
accelerator device" on this CPU-only host).
"""

from types import SimpleNamespace

import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._loader_helpers import Patch, module_under_test
from vllm_fl.worker.model_runner import ModelRunnerFL

model_runner_module = module_under_test()


def _profile_runner(**extra):
    """Runner namespace exercising the non-encoder profile_run branches."""
    calls = SimpleNamespace(dummy=[], pooler=[], sampler=[], sync=0)
    runner = SimpleNamespace(
        supports_mm_inputs=False,
        is_pooling_model=False,
        max_num_tokens=8,
        calls=calls,
        encoder_cache={"tmp_0": object()},
    )
    runner._dummy_run = lambda num_tokens, **kw: (
        calls.dummy.append((num_tokens, kw)) or (torch.zeros(4, 2), torch.zeros(2, 2))
    )
    runner._dummy_pooler_run = lambda hidden: (
        calls.pooler.append(hidden) or "pooler-out"
    )
    runner._dummy_sampler_run = lambda last: calls.sampler.append(last) or "sampler-out"
    runner._sync_device = lambda: calls.sync.__class__.__setattr__(
        calls, "sync", calls.sync + 1
    )
    for key, value in extra.items():
        setattr(runner, key, value)
    return runner


def test_profile_run_without_mm_runs_dummy_and_sampler():
    """A non-multimodal runner profiles the dummy run and the sampler warm-up
    on the last pipeline rank."""
    runner = _profile_runner()

    with Patch(
        model_runner_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    ):
        ModelRunnerFL.profile_run(runner)

    # The dummy run saw is_profile=True with the max token count.
    assert runner.calls.dummy == [(8, {"is_profile": True})]
    # Last rank, non-pooling: the sampler warm-up ran on the last hidden
    # states (not the full hidden states).
    assert runner.calls.pooler == []
    assert len(runner.calls.sampler) == 1
    assert runner.calls.sampler[0].shape == (2, 2)
    assert runner.calls.sync == 1
    # The encoder cache was cleared after profiling.
    assert runner.encoder_cache == {}


def test_profile_run_pooling_model_runs_pooler():
    """A pooling model warms the pooler instead of the sampler."""
    runner = _profile_runner(is_pooling_model=True)

    with Patch(
        model_runner_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    ):
        ModelRunnerFL.profile_run(runner)

    assert len(runner.calls.pooler) == 1
    assert runner.calls.pooler[0].shape == (4, 2)
    assert runner.calls.sampler == []


def test_profile_run_not_last_rank_skips_output_warmup():
    """A non-last pipeline rank profiles the forward only; no sampler or
    pooler warm-up runs."""
    runner = _profile_runner()

    with Patch(
        model_runner_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False),
    ):
        ModelRunnerFL.profile_run(runner)

    assert runner.calls.pooler == []
    assert runner.calls.sampler == []
    assert runner.calls.sync == 1


def test_profile_run_mm_skip_profiling_flag_logs_and_skips_encoder():
    """A multimodal config with skip_mm_profiling logs and skips the encoder
    profiling branch entirely."""
    runner = _profile_runner(
        supports_mm_inputs=True,
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(skip_mm_profiling=True),
        ),
    )

    with Patch(
        model_runner_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    ):
        ModelRunnerFL.profile_run(runner)

    assert not hasattr(runner, "mm_budget")


def test_profile_run_mm_embedding_only_budget_skips_encoder():
    """A non-zero encoder budget with all modality limits at 0 takes the
    embedding-only early exit: no encoder run."""
    runner = _profile_runner(
        supports_mm_inputs=True,
        model_config=SimpleNamespace(multimodal_config=None),
        mm_budget=SimpleNamespace(
            get_encoder_budget=lambda: 1024,
            mm_max_toks_per_item={},
        ),
    )

    with Patch(
        model_runner_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    ):
        ModelRunnerFL.profile_run(runner)

    # No dummy encoder batch was requested (mm_max_toks_per_item falsy).
    assert not hasattr(runner, "_get_mm_dummy_batch")


def _cleanup_runner():
    """Runner with every cleanup target attribute populated."""
    layer_with_cache = SimpleNamespace(kv_cache=torch.zeros(2, 2))
    plain_layer = SimpleNamespace()
    quant_layer = SimpleNamespace(
        kv_cache=[],
        impl=SimpleNamespace(_k_scale_cache=1.0, _v_scale_cache=2.0),
    )
    static_forward_context = {
        "model.layers.0": layer_with_cache,
        "model.layers.1": plain_layer,
        "model.layers.2": quant_layer,
    }
    runner = SimpleNamespace(
        common_attention_metadata_graph=["camd-entry"],
        kv_caches=[torch.zeros(1), torch.zeros(1)],
        cross_layers_kv_cache=torch.zeros(1),
        cross_layers_attn_backend=object(),
        attn_groups=[[object()]],
        kv_cache_config=SimpleNamespace(num_blocks=4),
        cache_config=SimpleNamespace(num_gpu_blocks=16),
        compilation_config=SimpleNamespace(
            static_forward_context=static_forward_context,
        ),
    )
    return runner, static_forward_context


def test_cleanup_profiling_kv_cache_resets_all_state():
    """The cleanup drops graph metadata, kv caches, group state and the
    per-layer bindings, and clears the accelerator cache."""
    runner, ctx = _cleanup_runner()

    with (
        Patch(
            model_runner_module,
            "_accelerator_synchronize",
            lambda: None,
        ),
        Patch(torch.accelerator, "empty_cache", lambda: None),
    ):
        ModelRunnerFL._cleanup_profiling_kv_cache(runner)

    # Graph metadata cleared, kv cache list emptied and Nones dropped.
    assert runner.common_attention_metadata_graph == []
    assert runner.kv_caches == []
    # Cross-layer state dropped, attention groups emptied.
    assert runner.cross_layers_kv_cache is None
    assert runner.cross_layers_attn_backend is None
    assert runner.attn_groups == []
    # The config attribute was deleted and the block count reset.
    assert not hasattr(runner, "kv_cache_config")
    assert runner.cache_config.num_gpu_blocks is None
    # Tensor caches become empty tensors; non-tensor caches empty lists;
    # scale views reset to None.
    layer0, layer1, layer2 = (
        ctx["model.layers.0"],
        ctx["model.layers.1"],
        ctx["model.layers.2"],
    )
    assert isinstance(layer0.kv_cache, torch.Tensor)
    assert layer0.kv_cache.numel() == 0
    assert layer2.kv_cache == []
    assert layer2.impl._k_scale_cache is None
    assert layer2.impl._v_scale_cache is None
    # The plain layer (no kv_cache attr before) was never assigned one.
    assert not hasattr(layer1, "kv_cache")


def test_cleanup_profiling_kv_cache_without_kv_caches_is_noop_state():
    """A runner that never initialized KV caches clears nothing but still
    resets the attention-group state it has."""
    runner = SimpleNamespace(
        common_attention_metadata_graph=None,
        attn_groups=[[object()], [object()]],
        cache_config=SimpleNamespace(num_gpu_blocks=3),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )

    with (
        Patch(
            model_runner_module,
            "_accelerator_synchronize",
            lambda: None,
        ),
        Patch(torch.accelerator, "empty_cache", lambda: None),
    ):
        ModelRunnerFL._cleanup_profiling_kv_cache(runner)

    assert runner.attn_groups == []
    assert not hasattr(runner, "kv_caches")
    assert runner.cache_config.num_gpu_blocks is None
