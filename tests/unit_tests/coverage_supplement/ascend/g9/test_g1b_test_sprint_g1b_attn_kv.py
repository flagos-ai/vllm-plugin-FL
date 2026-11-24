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

"""G1b sprint coverage: attention-backend setup, KV-cache lifecycle and
routed-experts capturer of ModelRunnerFL (814b line).

Targets: ``capture_model`` (NONE early return), ``initialize_attn_backend``
(intra-group dedupe + num_heads split), ``_check_and_update_cudagraph_mode``,
``initialize_kv_cache`` wiring, ``init_routed_experts_capturer``,
``may_add_encoder_only_layers_to_kv_cache_config``, ``get_kv_cache_spec``
(empty), ``init_fp8_kv_scales`` via ``post_kv_cache_wake_up``,
``timed_encoder_operation`` and ``get_encoder_timing_stats``.

Attention groups are fake recorders; the KV cache specs are real vllm
dataclasses. Module-level collaborators (``get_layers_from_vllm_config``,
``prepare_kernel_block_sizes``, ``initialize_mamba_ssu_backend``,
``has_kv_transfer_group``, ``RoutedExpertsCapturer``,
``bind_routed_experts_capturer``, ``PIN_MEMORY``) are patched at the module
boundary with ExitStack.
"""

from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace

import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
)

import vllm_fl.worker.model_runner as model_runner_module
from vllm_fl.worker.model_runner import ModelRunnerFL


class _Patch:
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _stack(*pairs):
    stack = contextlib.ExitStack()
    for module, name, value in pairs:
        stack.enter_context(_Patch(module, name, value))
    return stack


class _AttentionGroup:
    def __init__(self, backend, layer_names, kv_cache_spec, kv_cache_group_id):
        self.backend = backend
        self.layer_names = list(layer_names)
        self.kv_cache_spec = kv_cache_spec
        self.kv_cache_group_id = kv_cache_group_id


class _Backend:
    @classmethod
    def full_cls_name(cls):
        return "test.Backend"


def _attn_spec():
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float32,
    )


def _layer(num_heads=8, backend=_Backend):
    return SimpleNamespace(
        num_heads=num_heads,
        get_attn_backend=lambda: backend,
    )


def _attn_runner(layers, groups, **extra):
    calls = SimpleNamespace(attn=[], cp=[])
    runner = SimpleNamespace(
        attn_groups=[],
        vllm_config=object(),
        kv_sharing_fast_prefill_eligible_layers=set(),
        calls=calls,
    )
    runner._check_and_update_cudagraph_mode = (
        lambda backends, kvgroups, is_profiling=False: calls.attn.append(
            (backends, kvgroups, is_profiling)
        )
    )
    for key, value in extra.items():
        setattr(runner, key, value)
    runner._layers = layers
    return runner


def test_capture_model_none_mode_warns_and_returns_zero(caplog):
    import logging

    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    )
    with caplog.at_level(
        logging.WARNING, logger="vllm_fl.worker.model_runner"
    ):
        captured = ModelRunnerFL.capture_model(runner)

    assert captured == 0
    skipped = [
        r for r in caplog.records
        if "Skipping CUDA graph capture" in r.getMessage()
    ]
    assert len(skipped) == 1


def test_initialize_attn_backend_dedupes_within_group():
    spec = _attn_spec()
    group = KVCacheGroupSpec(
        layer_names=["model.layers.0", "model.layers.1"],
        kv_cache_spec=spec,
    )
    layers = {
        "model.layers.0": _layer(),
        "model.layers.1": _layer(),
    }
    runner = _attn_runner(layers, [group])

    with _stack(
        (model_runner_module, "AttentionGroup", _AttentionGroup),
        (model_runner_module, "get_layers_from_vllm_config",
         lambda vc, layer_type, layer_names: {
             n: layers[n] for n in layer_names
         }),
        (model_runner_module, "check_attention_cp_compatibility",
         lambda vc: runner.calls.cp.append(vc)),
    ):
        ModelRunnerFL.initialize_attn_backend(
            runner, SimpleNamespace(kv_cache_groups=[group])
        )

    assert len(runner.attn_groups) == 1
    subgroups = runner.attn_groups[0]
    assert len(subgroups) == 1
    assert subgroups[0].backend is _Backend
    assert subgroups[0].layer_names == ["model.layers.0", "model.layers.1"]
    assert subgroups[0].kv_cache_spec is spec
    assert subgroups[0].kv_cache_group_id == 0
    assert runner.calls.cp == [runner.vllm_config]
    # The resolved mode was requested once for the single group.
    assert len(runner.calls.attn) == 1
    assert runner.calls.attn[0][2] is False


def test_initialize_attn_backend_splits_on_num_heads():
    spec = _attn_spec()
    group = KVCacheGroupSpec(
        layer_names=["model.layers.0", "model.layers.2"],
        kv_cache_spec=spec,
    )
    layers = {
        "model.layers.0": _layer(num_heads=8),
        "model.layers.2": _layer(num_heads=4),
    }
    runner = _attn_runner(layers, [group])

    with _stack(
        (model_runner_module, "AttentionGroup", _AttentionGroup),
        (model_runner_module, "get_layers_from_vllm_config",
         lambda vc, layer_type, layer_names: {
             n: layers[n] for n in layer_names
         }),
        (model_runner_module, "check_attention_cp_compatibility",
         lambda vc: None),
    ):
        ModelRunnerFL.initialize_attn_backend(
            runner, SimpleNamespace(kv_cache_groups=[group])
        )

    subgroups = runner.attn_groups[0]
    assert len(subgroups) == 2
    by_layer = {g.layer_names[0]: g for g in subgroups}
    assert by_layer["model.layers.0"].layer_names == ["model.layers.0"]
    assert by_layer["model.layers.2"].layer_names == ["model.layers.2"]


def test_check_and_update_cudagraph_mode_resolves_and_initializes():
    spec = _attn_spec()

    class _Builder:
        @classmethod
        def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
            return AttentionCGSupport.ALWAYS

    class _BackendX:
        @classmethod
        def get_builder_cls(cls):
            return _Builder

    resolve_calls = []
    key_calls = []
    runner = SimpleNamespace(
        vllm_config=object(),
        uniform_decode_query_len=3,
        kv_cache_config=object(),
        max_num_reqs=7,
        compilation_config=SimpleNamespace(
            resolve_cudagraph_mode_and_sizes=(
                lambda *a, **k: resolve_calls.append((a, k))
                or CUDAGraphMode.NONE
            ),
        ),
        cudagraph_dispatcher=SimpleNamespace(
            initialize_cudagraph_keys=(
                lambda mode, q: key_calls.append((mode, q))
            ),
        ),
        speculative_config=None,
    )
    kv_groups = [SimpleNamespace(kv_cache_spec=spec)]

    ModelRunnerFL._check_and_update_cudagraph_mode(
        runner,
        [{_BackendX}],
        kv_groups,
        is_profiling=True,
    )

    args, kwargs = resolve_calls[0]
    assert args == (AttentionCGSupport.ALWAYS, None, 3)
    assert kwargs == {
        "use_v2_model_runner": False,
        "kv_cache_config": runner.kv_cache_config,
        "max_num_reqs": 7,
        "is_profiling": True,
    }
    assert key_calls == [(CUDAGraphMode.NONE, 3)]


def test_initialize_kv_cache_runs_lifecycle_wiring():
    order = []
    spec = _attn_spec()
    group = KVCacheGroupSpec(layer_names=["model.layers.0"], kv_cache_spec=spec)
    config = SimpleNamespace(
        kv_cache_groups=[group],
        kv_cache_tensors=[],
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(mamba_config=object()),
        attn_groups=[[object()]],
        graph_runtime=None,
        speculative_config=None,
        _mamba_bufs="stale",
        jit_warmup_registry=SimpleNamespace(
            activate=lambda: contextlib.nullcontext()
        ),
    )
    runner.may_add_encoder_only_layers_to_kv_cache_config = (
        lambda: order.append("encoder")
    )
    runner.maybe_add_kv_sharing_layers_to_kv_cache_groups = (
        lambda cfg: order.append("sharing")
    )
    runner.initialize_attn_backend = (
        lambda cfg, is_profiling=False: order.append(("attn", is_profiling))
    )
    runner.initialize_metadata_builders = (
        lambda cfg, kbs: order.append("meta")
    )
    runner.may_reinitialize_input_batch = (
        lambda cfg, kbs: order.append("reinit")
    )
    runner.initialize_kv_cache_tensors = (
        lambda cfg, kbs: order.append("tensors") or {"model.layers.0": "t"}
    )

    with _stack(
        (model_runner_module, "prepare_kernel_block_sizes",
         lambda cfg, groups: [16]),
        (model_runner_module, "initialize_mamba_ssu_backend",
         lambda mamba_cfg, cfg: order.append("mamba")),
        (model_runner_module, "has_kv_transfer_group", lambda: False),
    ):
        ModelRunnerFL.initialize_kv_cache(runner, config, is_profiling=True)

    assert runner.kv_cache_config is not config
    assert runner.kv_cache_config.kv_cache_groups[0].layer_names == [
        "model.layers.0"
    ]
    assert runner._mamba_bufs is None
    assert runner._kernel_block_sizes == [16]
    assert order == [
        "encoder",
        "sharing",
        ("attn", True),
        "mamba",
        "meta",
        "reinit",
        "tensors",
    ]


def test_init_routed_experts_capturer_allocates_buffers():
    bound = []

    class _Capturer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.device_buffer = torch.zeros(2, 3, dtype=torch.float32)
            self.device = torch.device("cpu")

    model = object()
    runner = SimpleNamespace(
        model_config=SimpleNamespace(enable_return_routed_experts=True),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        vllm_config=object(),
        kv_cache_config=object(),
        model=model,
        device=torch.device("cpu"),
    )

    with _stack(
        (model_runner_module, "RoutedExpertsCapturer", _Capturer),
        (model_runner_module, "bind_routed_experts_capturer",
         lambda m, c: bound.append((m, c))),
        (model_runner_module, "PIN_MEMORY", False),
    ):
        ModelRunnerFL.init_routed_experts_capturer(runner)

    assert runner.routed_experts_initialized is True
    assert runner.routed_experts_cpu.shape == (2, 3)
    assert runner.routed_experts_cpu.dtype is torch.float32
    assert runner.routed_experts_slot_mapping_cpu.shape == (16,)
    assert runner.routed_experts_slot_mapping_device.shape == (16,)
    assert bound == [(model, runner.routed_experts_capturer)]


def test_may_add_encoder_only_layers_appends_group():
    enc = SimpleNamespace(
        attn_type=AttentionType.ENCODER_ONLY,
        num_kv_heads=2,
        head_size=8,
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(block_size=16)),
        kv_cache_dtype=torch.float16,
        runner_only_attn_layers=set(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )

    with _stack(
        (model_runner_module, "get_layers_from_vllm_config",
         lambda vc, layer_type: {"enc.0": enc}),
    ):
        ModelRunnerFL.may_add_encoder_only_layers_to_kv_cache_config(runner)

    assert runner.runner_only_attn_layers == {"enc.0"}
    assert len(runner.kv_cache_config.kv_cache_groups) == 1
    group = runner.kv_cache_config.kv_cache_groups[0]
    assert group.layer_names == ["enc.0"]
    assert isinstance(group.kv_cache_spec, EncoderOnlyAttentionSpec)
    assert group.kv_cache_spec.block_size == 16


def test_get_kv_cache_spec_returns_empty_without_layers():
    runner = SimpleNamespace(
        vllm_config=object(),
        use_ascend_hybrid_cache=False,
    )
    with _stack(
        (model_runner_module, "has_ec_transfer", lambda: False),
        (model_runner_module, "get_layers_from_vllm_config",
         lambda vc, layer_type: {}),
    ):
        assert ModelRunnerFL.get_kv_cache_spec(runner) == {}


def test_post_kv_cache_wake_up_non_quantized_returns_early():
    calls = []
    runner = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto"),
        init_fp8_kv_scales=lambda: calls.append("init"),
    )
    with _stack(
        (model_runner_module, "is_quantized_kv_cache", lambda dtype: False),
    ):
        ModelRunnerFL.post_kv_cache_wake_up(runner)

    assert calls == ["init"]


def test_timed_encoder_operation_noop_when_not_timed():
    with ModelRunnerFL.timed_encoder_operation(
        SimpleNamespace(),
        should_time=False,
        group_lora_refs=[],
        current_item_idx=0,
        num_items=0,
    ):
        pass


def test_get_encoder_timing_stats_returns_and_clears():
    stats = SimpleNamespace(to_dict=lambda: {"secs": 1.0})
    runner = SimpleNamespace(
        _encoder_timing_lock=threading.Lock(),
        encoder_timing_registry={"r0": stats},
    )

    result = ModelRunnerFL.get_encoder_timing_stats(runner)

    assert result == {"r0": {"secs": 1.0}}
    assert runner.encoder_timing_registry == {}
