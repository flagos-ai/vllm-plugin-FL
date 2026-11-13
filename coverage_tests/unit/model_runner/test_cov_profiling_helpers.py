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

"""Coverage tests for the profiling and teardown helpers.

``profile_run`` drives the multimodal encoder profile then the dummy
language-model/sampler (or pooler) run; ``_dummy_sampler_run`` builds the
synthetic sampling metadata and warms the sampler (and rejection sampler);
``_dummy_pooler_run*`` pick the largest pooling task; the minimal-KV-cache
and cleanup/shutdown helpers release profiling state. Collaborators are
recorders, and the multimodal registry/encoder is a boundary.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import mr_module
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


class _Recorder:
    def __init__(self, result=None, result_factory=None):
        self.calls = []
        self.result = result
        self.result_factory = result_factory

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.result_factory is not None:
            return self.result_factory(*args, **kwargs)
        return self.result


class _Sampler:
    """Sampler stand-in: records calls and optionally raises."""

    def __init__(self, logprobs_mode="raw_logprobs", error=None):
        self.logprobs_mode = logprobs_mode
        self.calls = []
        self.error = error

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(kind="sampler-out", call=len(self.calls))


class _FakeModel:
    def __init__(self, logits):
        self.compute_logits = _Recorder(result=logits)


def _sampler_runner(logits=None, *, mode="raw_logprobs", error=None, **overrides):
    logits = logits if logits is not None else torch.zeros(2, 4)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None)
        ),
        model=_FakeModel(logits),
        sampler=_Sampler(mode, error),
        device=torch.device("cpu"),
        speculative_config=None,
        rejection_sampler=_Recorder(result="rejection-out"),
    )
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner


def test_dummy_sampler_run_builds_metadata_and_warms_both_paths():
    """The dummy sampler uses vocab-1 top-k and warms forward_native too."""
    runner = _sampler_runner()
    metadata_cls = _Recorder(result="meta")
    logits_procs = _Recorder(result="procs")
    replace_rec = _Recorder(result="warm-meta")
    platform = __import__("vllm_fl.platform", fromlist=["PlatformFL"]).PlatformFL
    with (
        _Patch(mr_module, "SamplingMetadata", metadata_cls),
        _Patch(mr_module, "LogitsProcessors", logits_procs),
        _Patch(mr_module, "replace", replace_rec),
        _Patch(platform, "vendor_name", "nvidia"),
    ):
        out = ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(2, 3))

    assert out.kind == "sampler-out"
    meta_kwargs = metadata_cls.calls[0][1]
    assert meta_kwargs["all_greedy"] is False
    assert meta_kwargs["top_k"].tolist() == [3, 3]
    assert meta_kwargs["temperature"].tolist() == [0.5, 0.5]
    assert meta_kwargs["top_p"].tolist() == pytest.approx([0.9, 0.9])
    assert meta_kwargs["output_token_ids"] == [[], []]
    assert meta_kwargs["spec_token_ids"] == [[], []]
    assert meta_kwargs["no_penalties"] is True
    assert meta_kwargs["logitsprocs"] == "procs"
    # First call: plain metadata; second warmup: a seeded generator is added.
    assert len(runner.sampler.calls) == 2
    assert runner.sampler.calls[0]["sampling_metadata"] == "meta"
    assert runner.sampler.calls[1]["sampling_metadata"] == "warm-meta"
    replaced = replace_rec.calls[0]
    assert replaced[0][0] == "meta"
    warm_generators = replaced[1]["generators"]
    assert list(warm_generators) == [0]
    assert warm_generators[0].device == torch.device("cpu")
    # compute_logits received the randomized hidden states of the same shape.
    assert runner.model.compute_logits.calls[0][0][0].shape == (2, 3)


def test_dummy_sampler_run_skips_native_warmup_in_processed_mode():
    """processed_logits binds forward=forward_native, so no extra call is made."""
    runner = _sampler_runner(mode="processed_logits")
    with (
        _Patch(mr_module, "SamplingMetadata", _Recorder(result="meta")),
        _Patch(mr_module, "replace", _Recorder(result="warm-meta")),
    ):
        ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(1, 3))
    assert len(runner.sampler.calls) == 1


def test_dummy_sampler_run_mm_encoder_only_returns_empty():
    """MM-encoder-only models have no sampler to warm."""
    runner = _sampler_runner()
    runner.vllm_config.model_config.multimodal_config = SimpleNamespace(
        mm_encoder_only=True
    )
    with _Patch(mr_module, "SamplingMetadata", _Recorder()):
        out = ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(1, 3))
    assert out.numel() == 0
    assert runner.sampler.calls == []


def test_dummy_sampler_run_rewrites_oom_error():
    """Only an out-of-memory RuntimeError is rewritten with guidance."""
    runner = _sampler_runner(error=RuntimeError("CUDA out of memory while sampling"))
    with (
        _Patch(mr_module, "SamplingMetadata", _Recorder()),
        pytest.raises(RuntimeError, match="lowering `max_num_seqs`"),
    ):
        ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(1, 3))

    runner = _sampler_runner(error=RuntimeError("unrelated failure"))
    with (
        _Patch(mr_module, "SamplingMetadata", _Recorder()),
        pytest.raises(RuntimeError, match="unrelated failure"),
    ):
        ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(1, 3))


def test_dummy_sampler_run_runs_rejection_sampler_for_spec_config():
    """Speculative configs warm the rejection sampler with softmax draft probs."""
    runner = _sampler_runner()
    runner.speculative_config = SimpleNamespace(
        rejection_sample_method="standard",
        draft_sample_method="probabilistic",
    )
    meta_cls = _Recorder(result="meta")
    spec_meta = _Recorder()
    spec_meta.make_dummy = lambda draft_ids, device: ("dummy-meta", draft_ids)
    platform = __import__("vllm_fl.platform", fromlist=["PlatformFL"]).PlatformFL
    with (
        _Patch(mr_module, "SamplingMetadata", meta_cls),
        _Patch(mr_module, "LogitsProcessors", _Recorder()),
        _Patch(mr_module, "replace", _Recorder(result="warm-meta")),
        _Patch(mr_module, "SpecDecodeMetadata", spec_meta),
        _Patch(platform, "vendor_name", "nvidia"),
    ):
        out = ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(2, 3))

    assert out.kind == "sampler-out"
    rej_args = runner.rejection_sampler.calls[0][0]
    assert rej_args[0] == ("dummy-meta", [[0], [0]])
    draft_probs = rej_args[1]
    torch.testing.assert_close(
        draft_probs.sum(dim=-1), torch.ones(2), rtol=1e-6, atol=1e-6
    )
    # Two requests, one draft token each: 2 + 2 logit rows.
    assert rej_args[2].shape == (4, 4)
    assert rej_args[3] == "meta"


def test_dummy_sampler_run_rejection_without_probabilistic_drafts():
    """Non-probabilistic draft sampling passes draft_probs=None."""
    runner = _sampler_runner()
    runner.speculative_config = SimpleNamespace(
        rejection_sample_method="standard",
        draft_sample_method="greedy",
    )
    spec_meta = _Recorder()
    spec_meta.make_dummy = lambda draft_ids, device: "dummy-meta"
    platform = __import__("vllm_fl.platform", fromlist=["PlatformFL"]).PlatformFL
    with (
        _Patch(mr_module, "SamplingMetadata", _Recorder(result="meta")),
        _Patch(mr_module, "LogitsProcessors", _Recorder()),
        _Patch(mr_module, "replace", _Recorder(result="warm-meta")),
        _Patch(mr_module, "SpecDecodeMetadata", spec_meta),
        _Patch(platform, "vendor_name", "nvidia"),
    ):
        ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(2, 3))

    assert runner.rejection_sampler.calls[0][0][1] is None


def test_profile_run_runs_dummy_model_and_sampler():
    """The text-only profile runs the max-size dummy batch and samples it."""
    runner = SimpleNamespace(
        supports_mm_inputs=False,
        max_num_tokens=32,
        _dummy_run=_Recorder(result=("hidden", "last-hidden")),
        is_pooling_model=False,
        _dummy_sampler_run=_Recorder(result="sampled"),
        _dummy_pooler_run=_Recorder(result="pooled"),
        _sync_device=_Recorder(),
        encoder_cache={"stale": 1},
    )
    with _Patch(mr_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)):
        ModelRunnerFL.profile_run(runner)

    assert runner._dummy_run.calls[0][0] == (32,)
    assert runner._dummy_run.calls[0][1] == {"is_profile": True}
    assert runner._dummy_sampler_run.calls[0][0][0] == "last-hidden"
    assert runner._dummy_pooler_run.calls == []
    assert runner.encoder_cache == {}
    assert len(runner._sync_device.calls) == 1


def test_profile_run_pooling_model_uses_pooler():
    """Pooling runners profile the pooler instead of the sampler."""
    runner = SimpleNamespace(
        supports_mm_inputs=False,
        max_num_tokens=16,
        _dummy_run=_Recorder(result=("hidden", "last-hidden")),
        is_pooling_model=True,
        _dummy_sampler_run=_Recorder(result="sampled"),
        _dummy_pooler_run=_Recorder(result="pooled"),
        _sync_device=_Recorder(),
        encoder_cache={},
    )
    with _Patch(mr_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)):
        ModelRunnerFL.profile_run(runner)
    assert runner._dummy_pooler_run.calls[0][0][0] == "hidden"
    assert runner._dummy_sampler_run.calls == []


def test_profile_run_non_last_rank_skips_output_work():
    """A non-last PP rank profiles the forward but produces no output."""
    runner = SimpleNamespace(
        supports_mm_inputs=False,
        max_num_tokens=16,
        _dummy_run=_Recorder(result=("hidden", "last-hidden")),
        is_pooling_model=False,
        _dummy_sampler_run=_Recorder(),
        _dummy_pooler_run=_Recorder(),
        _sync_device=_Recorder(),
        encoder_cache={},
    )
    with _Patch(mr_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)):
        ModelRunnerFL.profile_run(runner)
    assert runner._dummy_sampler_run.calls == []
    assert runner._dummy_pooler_run.calls == []


def _mm_runner(**overrides):
    mm_config = SimpleNamespace(skip_mm_profiling=False)
    runner = SimpleNamespace(
        supports_mm_inputs=True,
        model_config=SimpleNamespace(multimodal_config=mm_config),
        mm_budget=None,
        max_num_tokens=16,
        _dummy_run=_Recorder(result=("hidden", "last-hidden")),
        is_pooling_model=False,
        _dummy_sampler_run=_Recorder(result="sampled"),
        _dummy_pooler_run=_Recorder(result="pooled"),
        _sync_device=_Recorder(),
        encoder_cache={},
        _get_mm_dummy_batch=_Recorder(result={"pixel_values": torch.zeros(2, 4)}),
        model=SimpleNamespace(
            embed_multimodal=_Recorder(result=[torch.zeros(1, 4), torch.zeros(1, 4)])
        ),
    )
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner


def test_profile_run_encoder_budget_skips_embedding_only_models():
    """A zero modality limit (embedding-only) skips encoder profiling."""
    runner = _mm_runner()
    runner.mm_budget = SimpleNamespace(
        get_encoder_budget=lambda: 100,
        mm_max_toks_per_item=0,
        mm_max_items_per_batch={},
    )
    with _Patch(mr_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)):
        ModelRunnerFL.profile_run(runner)
    assert runner._get_mm_dummy_batch.calls == []
    assert runner.model.embed_multimodal.calls == []


def test_profile_run_profiles_max_modality_and_caches_outputs():
    """The encoder profile uses the largest modality and a sanity check."""
    runner = _mm_runner()
    runner.mm_budget = SimpleNamespace(
        get_encoder_budget=lambda: 100,
        mm_max_toks_per_item=1,
        get_modality_with_max_tokens=lambda: "image",
        mm_max_items_per_batch={"image": 2},
    )
    sanity = _Recorder()
    with (
        _Patch(mr_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)),
        _Patch(mr_module, "sanity_check_mm_encoder_outputs", sanity),
    ):
        ModelRunnerFL.profile_run(runner)

    assert runner._get_mm_dummy_batch.calls[0][0] == ("image", 2)
    embed_kwargs = runner.model.embed_multimodal.calls[0][1]
    assert (
        embed_kwargs["pixel_values"]
        is runner._get_mm_dummy_batch.result["pixel_values"]
    )
    assert sanity.calls[0][1] == {"expected_num_items": 2}
    # Outputs were cached under temporary keys and cleared at the end.
    assert runner.encoder_cache == {}


def test_profile_run_skip_mm_profiling_flag_bypasses_encoder():
    """skip_mm_profiling avoids both the encoder budget and the encoder call."""
    runner = _mm_runner(
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(skip_mm_profiling=True)
        )
    )
    with _Patch(mr_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)):
        ModelRunnerFL.profile_run(runner)
    assert runner.model.embed_multimodal.calls == []


class _PoolingParams:
    def __init__(self, task):
        self.task = task
        self.applied = False
        self.verified = 0

    def verify(self, model_config):
        self.verified += 1


class _Pooler:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []
        self.update_tasks = []

    def get_pooling_updates(self, task):
        self.update_tasks.append(task)
        return SimpleNamespace(apply=lambda params: setattr(params, "applied", True))

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class _PoolingMetadata:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cursor = None

    def build_pooling_cursor(self, num_scheduled_tokens_np, **kwargs):
        self.cursor = (num_scheduled_tokens_np, kwargs)


def _pool_task_runner(pooler, num_seqs=2):
    model = SimpleNamespace(pooler=pooler)
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=num_seqs),
        device=torch.device("cpu"),
        model_config=object(),
        get_model=lambda: model,
    )


def test_dummy_pooler_run_task_builds_split_metadata():
    """The pooling dummy splits tokens across requests and builds a cursor."""
    pooler = _Pooler(result="pooled-out")
    runner = _pool_task_runner(pooler)
    params_cls = _Recorder(result=_PoolingParams(task="embed"))
    with (
        _Patch(mr_module, "PoolingParams", params_cls),
        _Patch(mr_module, "PoolingMetadata", _PoolingMetadata),
        _Patch(mr_module, "PoolingStates", _Recorder(result="state")),
    ):
        out = ModelRunnerFL._dummy_pooler_run_task(
            runner, torch.zeros(5, 3), task="embed"
        )

    assert out == "pooled-out"
    args = pooler.calls[0]
    metadata = args["pooling_metadata"]
    np.testing.assert_array_equal(
        metadata.kwargs["prompt_lens"].numpy(), np.array([2, 3])
    )
    assert metadata.kwargs["prompt_token_ids"].shape == (2, 2)
    assert len(metadata.kwargs["pooling_params"]) == 2
    assert metadata.kwargs["pooling_params"][0].applied is True
    schedule, cursor_kwargs = metadata.cursor
    np.testing.assert_array_equal(schedule, np.array([2, 3]))
    assert cursor_kwargs["device"] == torch.device("cpu")
    torch.testing.assert_close(
        cursor_kwargs["seq_lens_cpu"], metadata.kwargs["prompt_lens"]
    )
    assert params_cls.calls[0][1] == {"task": "embed"}
    assert pooler.update_tasks == ["embed"]


def test_dummy_pooler_run_task_rewrites_oom_only():
    """A pooler OOM is rewritten with the task name; others propagate."""
    pooler = _Pooler(error=RuntimeError("CUDA out of memory in pooler"))
    runner = _pool_task_runner(pooler)
    with (
        _Patch(mr_module, "PoolingParams", _Recorder(result=_PoolingParams("embed"))),
        _Patch(mr_module, "PoolingMetadata", _PoolingMetadata),
        _Patch(mr_module, "PoolingStates", _Recorder()),
        pytest.raises(RuntimeError, match="warming up pooler"),
    ):
        ModelRunnerFL._dummy_pooler_run_task(runner, torch.zeros(2, 3), task="embed")


def _raise_oom():
    raise RuntimeError("CUDA out of memory in pooler")


def test_dummy_pooler_run_picks_the_largest_output_task():
    """Each supported task runs once and the largest is repeated."""
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None)
        ),
        get_supported_pooling_tasks=lambda: ["a", "b"],
        _dummy_pooler_run_task=_Recorder(
            result_factory=lambda hidden, task: (
                [SimpleNamespace(nbytes=10)]
                if task == "a"
                else [SimpleNamespace(nbytes=20), None]
            )
        ),
    )
    out = ModelRunnerFL._dummy_pooler_run(runner, torch.zeros(2, 3))
    assert out[0].nbytes == 20
    assert [call[0][1] for call in runner._dummy_pooler_run_task.calls] == [
        "a",
        "b",
        "b",
    ]


def test_dummy_pooler_run_errors_without_supported_tasks():
    """A non-pooling model reports the documentation pointer."""
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None)
        ),
        get_supported_pooling_tasks=lambda: [],
        model_config=SimpleNamespace(model="m"),
    )
    with pytest.raises(RuntimeError, match="does not support any pooling tasks"):
        ModelRunnerFL._dummy_pooler_run(runner, torch.zeros(1, 3))


def test_dummy_pooler_run_mm_encoder_only_returns_empty():
    """MM-encoder-only models skip pooling entirely."""
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                multimodal_config=SimpleNamespace(mm_encoder_only=True)
            )
        )
    )
    out = ModelRunnerFL._dummy_pooler_run(runner, torch.zeros(1, 3))
    assert out.numel() == 0


def test_init_minimal_kv_cache_overrides_and_restores_block_count():
    """Profiling allocates the capture-size block budget then restores it."""
    kv_utils = __import__(
        "vllm.v1.core.kv_cache_utils",
        fromlist=["get_kv_cache_groups", "get_kv_cache_config_from_groups"],
    )
    spec = {"layer": "spec"}
    groups = ["group"]
    minimal = SimpleNamespace(num_blocks=5)
    runner = SimpleNamespace(
        get_kv_cache_spec=_Recorder(result=spec),
        vllm_config=object(),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=32),
        cache_config=SimpleNamespace(num_gpu_blocks_override=7, num_gpu_blocks=None),
        initialize_kv_cache=_Recorder(),
    )
    registry = SimpleNamespace(check_kv_cache_spec_registry=_Recorder())
    seen_override = []

    def _config_from_groups(cfg, kv_groups, available_memory):
        seen_override.append(runner.cache_config.num_gpu_blocks_override)
        assert kv_groups is groups
        assert available_memory == 0
        return minimal

    with (
        _Patch(mr_module, "KVCacheSpecRegistry", registry),
        _Patch(kv_utils, "get_kv_cache_groups", _Recorder(result=groups)),
        _Patch(kv_utils, "get_kv_cache_config_from_groups", _config_from_groups),
    ):
        ModelRunnerFL._init_minimal_kv_cache_for_profiling(runner)

    assert registry.check_kv_cache_spec_registry.calls[0][0][0] is spec
    assert seen_override == [32]
    assert runner.cache_config.num_gpu_blocks_override == 7
    assert runner.cache_config.num_gpu_blocks == 5
    assert runner.initialize_kv_cache.calls[0][1] == {"is_profiling": True}


def test_cleanup_profiling_kv_cache_drops_graphs_and_scales():
    """Cleanup clears graphs, KV caches, scale views, and the config attr."""
    metadata_graph = SimpleNamespace(clear=_Recorder())
    first = torch.zeros(2)
    second = torch.zeros(3)
    layer = SimpleNamespace(
        kv_cache=torch.zeros(4, 2),
        impl=SimpleNamespace(
            _k_scale_cache=torch.ones(1), _v_scale_cache=torch.ones(1)
        ),
    )
    runner = SimpleNamespace(
        common_attention_metadata_graph=metadata_graph,
        kv_caches=[first, second],
        attn_groups={"layer": "group"},
        kv_cache_config=object(),
        cache_config=SimpleNamespace(num_gpu_blocks=11),
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer}),
    )
    synced = []
    with (
        _Patch(mr_module, "_accelerator_synchronize", lambda: synced.append(True)),
        _Patch(
            mr_module.torch,
            "accelerator",
            SimpleNamespace(empty_cache=lambda: None),
        ),
    ):
        ModelRunnerFL._cleanup_profiling_kv_cache(runner)

    assert synced == [True]
    assert len(metadata_graph.clear.calls) == 1
    assert runner.kv_caches == []
    assert runner.attn_groups == {}
    assert not hasattr(runner, "kv_cache_config")
    assert runner.cache_config.num_gpu_blocks is None
    assert layer.kv_cache.numel() == 0
    assert isinstance(layer.kv_cache, torch.Tensor)
    assert layer.impl._k_scale_cache is None
    assert layer.impl._v_scale_cache is None


def test_cleanup_profiling_kv_cache_clears_cross_layer_state():
    """Cross-layer caches and backends are reset when present."""
    runner = SimpleNamespace(
        common_attention_metadata_graph=None,
        kv_caches=[],
        cross_layers_kv_cache=torch.zeros(2),
        cross_layers_attn_backend="backend",
        cache_config=SimpleNamespace(num_gpu_blocks=3),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with (
        _Patch(mr_module, "_accelerator_synchronize", lambda: None),
        _Patch(
            mr_module.torch,
            "accelerator",
            SimpleNamespace(empty_cache=lambda: None),
        ),
    ):
        ModelRunnerFL._cleanup_profiling_kv_cache(runner)

    assert runner.cross_layers_kv_cache is None
    assert runner.cross_layers_attn_backend is None


def test_shutdown_releases_model_workspace_and_rope():
    """The CPU/non-ROCm shutdown path clears runtime state and weights."""
    rope_module = __import__(
        "vllm.model_executor.layers.rotary_embedding",
        fromlist=["_ROPE_DICT"],
    )
    workspace = __import__(
        "vllm.v1.worker.workspace", fromlist=["reset_workspace_manager"]
    )
    rope = {"layer": object()}
    cleared = _Recorder()
    runner = SimpleNamespace(
        graph_runtime=SimpleNamespace(clear_decoder_graphs=cleared),
        _cleanup_profiling_kv_cache=_Recorder(),
        compilation_config=SimpleNamespace(static_forward_context={"layer": object()}),
        model=object(),
    )
    reset = _Recorder()
    with (
        _Patch(mr_module, "current_platform", SimpleNamespace(is_rocm=lambda: False)),
        _Patch(rope_module, "_ROPE_DICT", rope),
        _Patch(workspace, "reset_workspace_manager", reset),
    ):
        ModelRunnerFL.shutdown(runner)

    assert len(cleared.calls) == 1
    assert len(runner._cleanup_profiling_kv_cache.calls) == 1
    assert runner.compilation_config.static_forward_context == {}
    assert runner.model is None
    assert rope == {}
    assert len(reset.calls) == 1


def test_shutdown_rocm_drops_graphs_and_empties_cache():
    """The ROCm teardown drops graph wrappers and empties the accelerator."""
    rope_module = __import__(
        "vllm.model_executor.layers.rotary_embedding",
        fromlist=["_ROPE_DICT"],
    )
    workspace = __import__(
        "vllm.v1.worker.workspace", fromlist=["reset_workspace_manager"]
    )
    graph_clears = _Recorder()
    breakable_clears = _Recorder()
    runner = SimpleNamespace(
        graph_runtime=None,
        _cleanup_profiling_kv_cache=_Recorder(),
        compilation_config=SimpleNamespace(static_forward_context={}),
        model=object(),
        encoder_cudagraph_manager="manager",
    )
    empty_cache = _Recorder()
    collected = _Recorder()
    synced = _Recorder()
    with (
        _Patch(mr_module, "current_platform", SimpleNamespace(is_rocm=lambda: True)),
        _Patch(
            mr_module,
            "GraphWrapper",
            SimpleNamespace(clear_all_graphs=lambda: graph_clears()),
        ),
        _Patch(
            mr_module,
            "BreakableCUDAGraphWrapper",
            SimpleNamespace(clear_all_graphs=lambda: breakable_clears()),
        ),
        _Patch(rope_module, "_ROPE_DICT", {}),
        _Patch(workspace, "reset_workspace_manager", _Recorder()),
        _Patch(
            mr_module.torch,
            "accelerator",
            SimpleNamespace(empty_cache=lambda: empty_cache()),
        ),
        _Patch(mr_module, "_accelerator_synchronize", lambda: synced()),
        _Patch(mr_module, "gc", SimpleNamespace(collect=lambda: collected())),
    ):
        ModelRunnerFL.shutdown(runner)

    assert len(graph_clears.calls) == 1
    assert len(breakable_clears.calls) == 1
    assert runner.encoder_cudagraph_manager is None
    assert len(empty_cache.calls) == 1
    assert len(collected.calls) == 1
    assert len(synced.calls) == 1
