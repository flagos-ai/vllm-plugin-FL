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

"""G7-E (batch 2) coverage: lifecycle / KV-cache config / MM dummy / sampler.

Targets (814b): _init_minimal_kv_cache_for_profiling 6853-6879, _freeze_gc
6883-6903, shutdown 6905-6930, _cleanup_profiling_kv_cache 6932-6964,
may_reinitialize_input_batch 7803-7875,
may_add_encoder_only_layers_to_kv_cache_config 8077-8101,
get_kv_cache_spec 8103-8150, init_routed_experts_capturer 8035-8075,
_get_mm_dummy_batch 6072-6106, _dummy_sampler_run 6558-6689.

All collaborators are patched at the ``model_runner`` / vLLM module boundary;
the runner itself is a SimpleNamespace. CPU only.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.worker import model_runner as mr


_MISSING = object()


class _Patch:
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name, _MISSING)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        if self.saved is _MISSING:
            try:
                delattr(self.module, self.name)
            except AttributeError:
                pass
        else:
            setattr(self.module, self.name, self.saved)
        return False


def _stack(*pairs):
    stack = contextlib.ExitStack()
    for module, name, value in pairs:
        stack.enter_context(_Patch(module, name, value))
    return stack


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


# ---------------------------------------------------------------------------
# _freeze_gc
# ---------------------------------------------------------------------------
class _FakeGC:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.calls = []

    def isenabled(self):
        self.calls.append("isenabled")
        return self.enabled

    def collect(self):
        self.calls.append("collect")

    def freeze(self):
        self.calls.append("freeze")

    def disable(self):
        self.calls.append("disable")
        self.enabled = False

    def unfreeze(self):
        self.calls.append("unfreeze")

    def enable(self):
        self.calls.append("enable")
        self.enabled = True


def test_freeze_gc_freezes_and_restores_when_disabled_env():
    gc = _FakeGC(enabled=True)
    with _stack(
        (mr, "gc", gc),
        (mr.envs, "VLLM_ENABLE_CUDAGRAPH_GC", False),
    ):
        with mr.ModelRunnerFL._freeze_gc():
            assert "freeze" in gc.calls
    assert gc.calls == [
        "isenabled",
        "collect",
        "freeze",
        "disable",
        "unfreeze",
        "collect",
        "enable",
    ]


def test_freeze_gc_does_not_freeze_when_env_enables_gc():
    gc = _FakeGC(enabled=True)
    with _stack(
        (mr, "gc", gc),
        (mr.envs, "VLLM_ENABLE_CUDAGRAPH_GC", True),
    ):
        with mr.ModelRunnerFL._freeze_gc():
            pass
    assert "freeze" not in gc.calls
    assert "unfreeze" not in gc.calls


# ---------------------------------------------------------------------------
# shutdown / cleanup
# ---------------------------------------------------------------------------
def test_shutdown_releases_model_and_kv_cache():
    pytest.importorskip("vllm.model_executor.layers.rotary_embedding")
    pytest.importorskip("vllm.v1.worker.workspace")
    cleanup = _Recorder()
    static_ctx = {"l": object()}
    runner = SimpleNamespace(
        graph_runtime=None,
        _cleanup_profiling_kv_cache=lambda: cleanup("cleanup"),
        compilation_config=SimpleNamespace(static_forward_context=static_ctx),
        model=object(),
        encoder_cudagraph_manager=object(),
    )
    platform = SimpleNamespace(is_rocm=lambda: False, is_xpu=lambda: False)
    with _stack((mr, "current_platform", platform)):
        mr.ModelRunnerFL.shutdown(runner)
    assert cleanup.calls == [(("cleanup",), {})]
    assert runner.model is None
    assert static_ctx == {}


def test_cleanup_profiling_kv_cache_clears_state_and_layers():
    layer = SimpleNamespace(
        kv_cache=torch.zeros(2),
        impl=SimpleNamespace(_k_scale_cache=1, _v_scale_cache=2),
    )
    static_ctx = {"l": layer}
    runner = SimpleNamespace(
        kv_caches=[torch.zeros(1)],
        cross_layers_kv_cache=torch.zeros(1),
        cross_layers_attn_backend="backend",
        attn_groups={"g": 1},
        kv_cache_config="cfg",
        cache_config=SimpleNamespace(num_gpu_blocks=5),
        compilation_config=SimpleNamespace(static_forward_context=static_ctx),
    )
    fake_accel = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)
    with _stack((mr.torch, "accelerator", fake_accel)):
        mr.ModelRunnerFL._cleanup_profiling_kv_cache(runner)
    assert runner.kv_caches == []
    assert runner.cross_layers_kv_cache is None
    assert runner.cross_layers_attn_backend is None
    assert runner.attn_groups == {}
    assert not hasattr(runner, "kv_cache_config")
    assert runner.cache_config.num_gpu_blocks is None
    assert torch.is_tensor(layer.kv_cache) and layer.kv_cache.numel() == 0
    assert layer.impl._k_scale_cache is None
    assert layer.impl._v_scale_cache is None


def test_init_minimal_kv_cache_for_profiling_uses_min_blocks():
    kcu = pytest.importorskip("vllm.v1.core.kv_cache_utils")
    minimal_cfg = SimpleNamespace(num_blocks=3)
    init_calls = _Recorder()
    runner = SimpleNamespace(
        get_kv_cache_spec=lambda: {"l": "spec"},
        max_num_reqs=4,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=8),
        cache_config=SimpleNamespace(num_gpu_blocks_override=7, num_gpu_blocks=None),
        vllm_config=object(),
        initialize_kv_cache=lambda cfg, is_profiling: init_calls(cfg, is_profiling),
    )
    with _stack(
        (mr.KVCacheSpecRegistry, "check_kv_cache_spec_registry", lambda spec: None),
        (kcu, "get_kv_cache_groups", lambda vc, spec: ["group"]),
        (
            kcu,
            "get_kv_cache_config_from_groups",
            lambda vc, groups, available_memory: minimal_cfg,
        ),
    ):
        mr.ModelRunnerFL._init_minimal_kv_cache_for_profiling(runner)
    assert init_calls.calls == [((minimal_cfg, True), {})]
    assert runner.cache_config.num_gpu_blocks == 3
    assert runner.cache_config.num_gpu_blocks_override == 7


# ---------------------------------------------------------------------------
# may_reinitialize_input_batch
# ---------------------------------------------------------------------------
def _reinit_runner(init_blocks, init_modes):
    spec = SimpleNamespace(
        block_size=16,
        kind="attn",
        max_num_blocks_per_req=lambda vc, ml: 4,
    )
    return SimpleNamespace(
        max_model_len=100,
        max_encoder_len=200,
        vllm_config=SimpleNamespace(reasoning_config=None),
        _init_block_sizes=list(init_blocks),
        _init_kernel_block_sizes=[16],
        _init_max_num_blocks=[4],
        _init_slot_mapping_modes=list(init_modes),
        max_num_reqs=2,
        max_num_tokens=32,
        device=torch.device("cpu"),
        model_config=SimpleNamespace(get_vocab_size=lambda: 100),
        num_spec_tokens=0,
        is_pooling_model=False,
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        cache_config=SimpleNamespace(use_replayssm=False),
        input_batch=SimpleNamespace(
            logitsprocs="lp", logitsprocs_need_output_token_ids=False
        ),
        jit_warmup_registry=SimpleNamespace(activate=lambda: contextlib.nullcontext()),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
        ),
    )


def test_may_reinitialize_input_batch_no_change_keeps_batch():
    runner = _reinit_runner([16], ["tok"])
    with _stack(
        (mr, "get_kv_cache_spec_kind", lambda spec: spec.kind),
        (mr, "KVCacheSpecKind", SimpleNamespace(ENCODER_ONLY_ATTENTION="enc", MAMBA="mamba")),
        (mr, "SlotMappingMode", SimpleNamespace(TOKEN_TO_KV_SLOT="tok", NONE="none")),
        (mr, "InputBatch", _Recorder()),
    ):
        original = runner.input_batch
        mr.ModelRunnerFL.may_reinitialize_input_batch(runner, runner.kv_cache_config, [16])
    assert runner.input_batch is original


def test_may_reinitialize_input_batch_rebuilds_on_change():
    runner = _reinit_runner([], [])
    new_batch = object()
    with _stack(
        (mr, "get_kv_cache_spec_kind", lambda spec: spec.kind),
        (mr, "KVCacheSpecKind", SimpleNamespace(ENCODER_ONLY_ATTENTION="enc", MAMBA="mamba")),
        (mr, "SlotMappingMode", SimpleNamespace(TOKEN_TO_KV_SLOT="tok", NONE="none")),
        (mr, "InputBatch", lambda **kw: new_batch),
    ):
        mr.ModelRunnerFL.may_reinitialize_input_batch(runner, runner.kv_cache_config, [16])
    assert runner.input_batch is new_batch
    assert runner._init_block_sizes == [16]
    assert runner._init_slot_mapping_modes == ["tok"]


# ---------------------------------------------------------------------------
# may_add_encoder_only_layers_to_kv_cache_config
# ---------------------------------------------------------------------------
def test_may_add_encoder_only_layers_appends_group():
    attn = SimpleNamespace(attn_type="enc", num_kv_heads=2, head_size=8)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(block_size=16)),
        kv_cache_dtype="auto",
        runner_only_attn_layers=set(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )

    class _Spec:
        def __init__(self, **kw):
            self.kw = kw

    class _Group:
        def __init__(self, layer_names, kv_cache_spec):
            self.layer_names = layer_names
            self.kv_cache_spec = kv_cache_spec

    with _stack(
        (mr, "get_layers_from_vllm_config", lambda cfg, cls: {"l0": attn}),
        (mr, "AttentionType", SimpleNamespace(ENCODER_ONLY="enc")),
        (mr, "EncoderOnlyAttentionSpec", _Spec),
        (mr, "KVCacheGroupSpec", _Group),
    ):
        mr.ModelRunnerFL.may_add_encoder_only_layers_to_kv_cache_config(runner)
    assert runner.runner_only_attn_layers == {"l0"}
    assert len(runner.kv_cache_config.kv_cache_groups) == 1
    group = runner.kv_cache_config.kv_cache_groups[0]
    assert group.layer_names == ["l0"]
    assert isinstance(group.kv_cache_spec, _Spec)


def test_may_add_encoder_only_layers_noop_without_encoder_layers():
    attn = SimpleNamespace(attn_type="decoder")
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(block_size=16)),
        kv_cache_dtype="auto",
        runner_only_attn_layers=set(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )
    with _stack(
        (mr, "get_layers_from_vllm_config", lambda cfg, cls: {"l0": attn}),
        (mr, "AttentionType", SimpleNamespace(ENCODER_ONLY="enc")),
    ):
        mr.ModelRunnerFL.may_add_encoder_only_layers_to_kv_cache_config(runner)
    assert runner.kv_cache_config.kv_cache_groups == []


# ---------------------------------------------------------------------------
# get_kv_cache_spec
# ---------------------------------------------------------------------------
def test_get_kv_cache_spec_ec_producer_returns_empty():
    runner = SimpleNamespace(vllm_config=object())
    with _stack(
        (mr, "has_ec_transfer", lambda: True),
        (mr, "get_ec_transfer", lambda: SimpleNamespace(is_consumer=False)),
    ):
        assert mr.ModelRunnerFL.get_kv_cache_spec(runner) == {}


def test_get_kv_cache_spec_records_sharing_and_skips_none():
    class _FakeAttention:
        pass

    sharing = _FakeAttention()
    sharing.kv_sharing_target_layer_name = "target"
    normal = SimpleNamespace(
        get_kv_cache_spec=lambda cfg: "spec",
    )
    skipped = SimpleNamespace(get_kv_cache_spec=lambda cfg: None)
    runner = SimpleNamespace(
        vllm_config=object(),
        shared_kv_cache_layers={},
        use_ascend_hybrid_cache=False,
    )
    with _stack(
        (mr, "has_ec_transfer", lambda: False),
        (mr, "get_layers_from_vllm_config", lambda cfg, cls: {
            "share": sharing,
            "normal": normal,
            "skip": skipped,
        }),
        (mr, "Attention", _FakeAttention),
    ):
        out = mr.ModelRunnerFL.get_kv_cache_spec(runner)
    assert runner.shared_kv_cache_layers == {"share": "target"}
    assert out == {"normal": "spec"}


def test_get_kv_cache_spec_enriches_attention_spec_via_backend():
    class _FakeAttention:
        pass

    class _FakeSpec:
        pass

    spec = _FakeSpec()
    backend = SimpleNamespace(
        indexes_kv_by_block_stride=lambda: "idx",
        customize_spec=lambda s: ("customized", s),
    )
    layer = _FakeAttention()
    layer.kv_sharing_target_layer_name = None
    layer.get_kv_cache_spec = lambda cfg: spec
    layer.get_attn_backend = lambda: backend
    runner = SimpleNamespace(
        vllm_config=object(),
        shared_kv_cache_layers={},
        use_ascend_hybrid_cache=False,
    )
    with _stack(
        (mr, "has_ec_transfer", lambda: False),
        (mr, "get_layers_from_vllm_config", lambda cfg, cls: {"l": layer}),
        (mr, "Attention", _FakeAttention),
        (mr, "AttentionSpec", _FakeSpec),
        (mr, "set_current_vllm_config", lambda cfg: contextlib.nullcontext()),
        (mr, "replace", lambda obj, **kw: ("replaced", kw)),
    ):
        out = mr.ModelRunnerFL.get_kv_cache_spec(runner)
    assert out["l"][0] == "customized"
    assert out["l"][1][0] == "replaced"
    assert out["l"][1][1] == {"indexes_kv_by_block_stride": "idx"}


# ---------------------------------------------------------------------------
# init_routed_experts_capturer
# ---------------------------------------------------------------------------
def test_init_routed_experts_capturer_builds_buffers():
    bind = _Recorder()

    class _Cap:
        def __init__(self, **kw):
            self.kw = kw
            self.device_buffer = torch.zeros((2, 3), dtype=torch.int32)

    runner = SimpleNamespace(
        model_config=SimpleNamespace(enable_return_routed_experts=True),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        vllm_config="vc",
        kv_cache_config="kc",
        model="model",
        device=torch.device("cpu"),
    )
    with _stack(
        (mr, "RoutedExpertsCapturer", _Cap),
        (mr, "bind_routed_experts_capturer", bind),
        (mr, "PIN_MEMORY", False),
    ):
        mr.ModelRunnerFL.init_routed_experts_capturer(runner)
    assert runner.routed_experts_initialized is True
    assert runner.routed_experts_cpu.shape == (2, 3)
    assert runner.routed_experts_slot_mapping_cpu.shape == (8,)
    assert runner.routed_experts_slot_mapping_device.shape == (8,)
    assert bind.calls and bind.calls[0][0] == ("model", runner.routed_experts_capturer)


# ---------------------------------------------------------------------------
# _get_mm_dummy_batch
# ---------------------------------------------------------------------------
def test_get_mm_dummy_batch_new_budget_api():
    captured = {}

    def _group(dummy, device, pin_memory):
        captured["dummy"] = dummy
        return iter([("image", 0, "BATCH")])

    runner = SimpleNamespace(
        mm_budget=SimpleNamespace(
            get_dummy_encoder_profile_inputs=lambda modality, maxi: ["dm"],
            cache=object(),
        ),
        mm_registry=SimpleNamespace(),
        device=torch.device("cpu"),
    )
    with _stack((mr, "group_and_batch_mm_kwargs", _group)):
        out = mr.ModelRunnerFL._get_mm_dummy_batch(runner, "image", 2)
    assert out == "BATCH"
    assert captured["dummy"] == ["dm"]


def test_get_mm_dummy_batch_legacy_registry_api():
    captured = {}
    item = {"v": 1}

    def _group(dummy, device, pin_memory):
        captured["dummy"] = dummy
        return iter([("image", 0, "BATCH")])

    runner = SimpleNamespace(
        mm_budget=SimpleNamespace(cache="cache"),
        mm_registry=SimpleNamespace(
            get_dummy_mm_inputs=lambda mc, mm_counts, cache: {
                "mm_kwargs": {"image": [item]}
            }
        ),
        model_config=object(),
        device=torch.device("cpu"),
    )
    with _stack((mr, "group_and_batch_mm_kwargs", _group)):
        out = mr.ModelRunnerFL._get_mm_dummy_batch(runner, "image", 2)
    assert out == "BATCH"
    assert captured["dummy"] == [("image", item), ("image", item)]


# ---------------------------------------------------------------------------
# _dummy_sampler_run
# ---------------------------------------------------------------------------
class _Sampler:
    def __init__(self, mode):
        self.logprobs_mode = mode
        self.calls = 0

    def __call__(self, logits, sampling_metadata):
        self.calls += 1
        return "SAMPLER_OUT"


def _sampler_runner(mode):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None)
        ),
        model=SimpleNamespace(
            compute_logits=lambda h: torch.zeros(h.shape[0], 5)
        ),
        device=torch.device("cpu"),
        sampler=_Sampler(mode),
        speculative_config=None,
    )


def test_dummy_sampler_run_mm_encoder_only_returns_empty():
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                multimodal_config=SimpleNamespace(mm_encoder_only=True)
            )
        )
    )
    out = mr.ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(3, 4))
    assert torch.is_tensor(out) and out.numel() == 0






