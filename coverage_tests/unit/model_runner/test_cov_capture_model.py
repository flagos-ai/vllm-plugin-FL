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

"""Coverage tests for CUDA-graph capture and graph-memory profiling.

``capture_model`` walks the dispatcher's capture descriptors in the order
the dispatcher produced them (largest first), captures the encoder manager
last, and reports the free-memory delta. ``profile_cudagraph_memory``
measures one or two descriptors per shared-pool mode (all descriptors for
private breakable pools), extrapolates per-graph cost, adds encoder memory,
and restores every wrapper pool and counter in ``finally``. The musa size
limiter and the warmup/capture helper are exercised directly. Device
memory sampling and the graph-capture context are module boundaries.
"""

import sys  # noqa: I001  (shim import must precede vllm)
import types
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import mr_module
from vllm.config import CUDAGraphMode

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


class _MemPlatform:
    """Platform stand-in yielding a scripted free-memory sequence."""

    def __init__(self, values):
        self.values = list(values)
        self.pools = 0
        self.torch_device_fn = SimpleNamespace(mem_get_info=self._mem)

    def _mem(self):
        return (self.values.pop(0), 0)

    def graph_pool_handle(self):
        self.pools += 1
        return f"pool{self.pools}"


def _capture_ctx(calls):
    @contextmanager
    def _graph_capture(*args, **kwargs):
        calls.append((args, kwargs))
        yield

    return _graph_capture


def _set_config_calls(calls):
    @contextmanager
    def _set_current_vllm_config(cfg):
        calls.append(cfg)
        yield

    return _set_current_vllm_config


def _desc(num_tokens=8, uniform=False):
    return SimpleNamespace(
        num_tokens=num_tokens, uniform=uniform, num_active_loras=0, num_reqs=None
    )


def _fake_wrapper_cls(instances):
    return types.SimpleNamespace(
        _all_instances=instances,
        clear_all_graphs=staticmethod(lambda: None),
    )


def test_capture_model_skips_when_cudagraph_mode_is_none():
    """CUDAGraphMode.NONE returns 0 without touching the dispatcher."""
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        _limit_musa_piecewise_capture_sizes=_Recorder(),
        _maybe_init_encoder_cudagraph_manager=_Recorder(),
        cudagraph_dispatcher=SimpleNamespace(get_capture_descs=_Recorder()),
    )
    assert ModelRunnerFL.capture_model(runner) == 0
    assert runner._limit_musa_piecewise_capture_sizes.calls == []
    assert runner._maybe_init_encoder_cudagraph_manager.calls == []
    assert runner.cudagraph_dispatcher.get_capture_descs.calls == []


def test_capture_model_captures_descriptors_then_encoder_graphs():
    """Capture walks the descriptor groups, then the encoder manager."""
    descs = [
        (CUDAGraphMode.PIECEWISE, [_desc(16), _desc(8)]),
        (CUDAGraphMode.FULL, [_desc(4)]),
    ]
    captured = _Recorder()
    encoder_capture = _Recorder()
    encoder_manager = SimpleNamespace(capture=encoder_capture)
    counter = SimpleNamespace(
        num_gpu_runner_capture_triggers=0, num_cudagraph_captured=0
    )
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
        _limit_musa_piecewise_capture_sizes=_Recorder(),
        _maybe_init_encoder_cudagraph_manager=_Recorder(),
        cudagraph_dispatcher=SimpleNamespace(get_capture_descs=lambda: list(descs)),
        _capture_cudagraphs=captured,
        encoder_cudagraph_manager=encoder_manager,
        device="cpu",
        _freeze_gc=lambda: nullcontext(),
    )
    platform = _MemPlatform([1000, 400])
    capture_ctx = []
    enabled = []
    synced = []
    locked = []
    with (
        _Patch(mr_module, "current_platform", platform),
        _Patch(mr_module, "compilation_counter", counter),
        _Patch(
            mr_module,
            "set_cudagraph_capturing_enabled",
            lambda value: enabled.append(value),
        ),
        _Patch(mr_module, "graph_capture", _capture_ctx(capture_ctx)),
        _Patch(mr_module, "_accelerator_synchronize", lambda: synced.append(True)),
        _Patch(mr_module, "lock_workspace", lambda: locked.append(True)),
        _Patch(
            mr_module.torch,
            "accelerator",
            SimpleNamespace(empty_cache=lambda: None),
        ),
    ):
        assert ModelRunnerFL.capture_model(runner) == 600

    assert counter.num_gpu_runner_capture_triggers == 1
    assert [call[1]["cudagraph_runtime_mode"] for call in captured.calls] == [
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL,
    ]
    assert captured.calls[0][1]["batch_descriptors"] == descs[0][1]
    assert encoder_capture.calls[0][1] == {"graph_pool": "pool1"}
    assert enabled == [True, False]
    assert capture_ctx[0][1] == {"device": "cpu"}
    assert locked == [True]
    assert len(synced) >= 3


def test_limit_musa_piecewise_reduces_capture_sizes():
    """The musa limiter truncates sizes to the live-graph budget."""
    keys = {"pi": _ClearableSet(), "pl": _ClearableSet()}
    dispatcher = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        cudagraph_keys=keys,
        keys_initialized=True,
        initialize_cudagraph_keys=_Recorder(),
        get_capture_descs=lambda: [
            (CUDAGraphMode.PIECEWISE, [_desc(4)] * 4),
            (CUDAGraphMode.NONE, []),
        ],
    )
    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        cudagraph_capture_sizes=[3, 1, 2, 4],
        max_cudagraph_capture_size=4,
    )
    runner = SimpleNamespace(
        compilation_config=compilation_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        cudagraph_dispatcher=dispatcher,
        uniform_decode_query_len=1,
    )
    wrappers = [
        SimpleNamespace(runtime_mode=CUDAGraphMode.PIECEWISE),
        SimpleNamespace(runtime_mode=CUDAGraphMode.PIECEWISE),
    ]
    platform = SimpleNamespace(device_type="musa")
    with (
        _Patch(mr_module, "current_platform", platform),
        _Patch(mr_module, "GraphWrapper", _fake_wrapper_cls(wrappers)),
        _Patch(mr_module, "_MUSA_MAX_LIVE_GRAPHS", 2),
    ):
        ModelRunnerFL._limit_musa_piecewise_capture_sizes(runner)

    assert compilation_config.cudagraph_capture_sizes == [1]
    assert compilation_config.max_cudagraph_capture_size == 1
    assert dispatcher.keys_initialized is False
    assert all(s.cleared == 1 for s in keys.values())
    init_args = dispatcher.initialize_cudagraph_keys.calls[0][0]
    assert init_args == (CUDAGraphMode.PIECEWISE, 1)


def test_limit_musa_piecewise_ignores_other_devices_and_modes():
    """Non-musa devices and non-piecewise modes leave capture sizes alone."""
    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.NONE,
        cudagraph_capture_sizes=[1, 2, 3],
        max_cudagraph_capture_size=3,
    )
    runner = SimpleNamespace(
        compilation_config=compilation_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        cudagraph_dispatcher=SimpleNamespace(get_capture_descs=_Recorder()),
    )
    with _Patch(mr_module, "current_platform", SimpleNamespace(device_type="musa")):
        ModelRunnerFL._limit_musa_piecewise_capture_sizes(runner)
    assert compilation_config.cudagraph_capture_sizes == [1, 2, 3]

    with _Patch(mr_module, "current_platform", SimpleNamespace(device_type="cuda")):
        ModelRunnerFL._limit_musa_piecewise_capture_sizes(runner)
    assert compilation_config.cudagraph_capture_sizes == [1, 2, 3]


def test_warmup_and_capture_runs_warmups_then_capture():
    """Warmups run eager with the descriptor's uniform/LoRA state."""
    dummy = _Recorder()
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_num_of_warmups=2),
        _dummy_run=dummy,
    )
    desc = SimpleNamespace(num_tokens=8, uniform=True, num_active_loras=1)

    ModelRunnerFL._warmup_and_capture(
        runner,
        desc,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        profile_seq_lens=3,
    )

    assert len(dummy.calls) == 3
    for args, kwargs in dummy.calls[:2]:
        assert args[0] == 8
        assert kwargs["cudagraph_runtime_mode"] == CUDAGraphMode.NONE
        assert kwargs["force_attention"] is False
        assert kwargs["uniform_decode"] is True
        assert kwargs["skip_eplb"] is True
        assert kwargs["remove_lora"] is False
        assert kwargs["num_active_loras"] == 1
        assert kwargs["profile_seq_lens"] == 3
        assert "is_graph_capturing" not in kwargs
    _, capture_kwargs = dummy.calls[2]
    assert capture_kwargs["cudagraph_runtime_mode"] == CUDAGraphMode.PIECEWISE
    assert capture_kwargs["is_graph_capturing"] is True


def test_warmup_and_capture_full_mode_forces_attention():
    """FULL capture runs its warmups with attention metadata enabled."""
    dummy = _Recorder()
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_num_of_warmups=1),
        _dummy_run=dummy,
    )
    desc = SimpleNamespace(num_tokens=2, uniform=False, num_active_loras=0)

    ModelRunnerFL._warmup_and_capture(
        runner, desc, cudagraph_runtime_mode=CUDAGraphMode.FULL, num_warmups=1
    )

    assert dummy.calls[0][1]["force_attention"] is True
    assert dummy.calls[1][1]["is_graph_capturing"] is True
    assert dummy.calls[1][1]["uniform_decode"] is False


def test_capture_cudagraphs_walks_each_descriptor():
    """Each descriptor is warmed up and captured, then LoRAs are removed."""
    warmup = _Recorder()
    remove = _Recorder()
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=False),
        _warmup_and_capture=warmup,
        maybe_remove_all_loras=remove,
        lora_config=None,
    )
    descs = [_desc(8, uniform=True), _desc(4, uniform=True)]
    synced = []
    with (
        _Patch(mr_module, "is_global_first_rank", lambda: False),
        _Patch(mr_module, "_accelerator_synchronize", lambda: synced.append(True)),
    ):
        ModelRunnerFL._capture_cudagraphs(
            runner, batch_descriptors=descs, cudagraph_runtime_mode=CUDAGraphMode.FULL
        )

    assert [c[0][0].num_tokens for c in warmup.calls] == [8, 4]
    assert all(c[1]["allow_microbatching"] is False for c in warmup.calls)
    assert len(synced) == 2
    assert len(remove.calls) == 1


def test_capture_cudagraphs_empty_list_is_a_noop():
    """No descriptors means no warmups and no LoRA removal."""
    warmup = _Recorder()
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=False),
        _warmup_and_capture=warmup,
        maybe_remove_all_loras=_Recorder(),
    )
    ModelRunnerFL._capture_cudagraphs(
        runner, batch_descriptors=[], cudagraph_runtime_mode=CUDAGraphMode.FULL
    )
    assert warmup.calls == []


def test_capture_cudagraphs_rejects_invalid_mode():
    """Capture refuses CUDAGraphMode.NONE."""
    runner = SimpleNamespace(parallel_config=SimpleNamespace(use_ubatching=False))
    with pytest.raises(AssertionError, match="Invalid cudagraph runtime mode"):
        ModelRunnerFL._capture_cudagraphs(
            runner,
            batch_descriptors=[_desc(2)],
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )


def test_profile_cudagraph_memory_without_graphs_cleans_up():
    """No capture descriptors means cleanup runs and zero bytes are reported."""
    runner = SimpleNamespace(
        vllm_config=object(),
        _init_minimal_kv_cache_for_profiling=_Recorder(),
        _create_encoder_cudagraph_manager=_Recorder(result=None),
        cudagraph_dispatcher=SimpleNamespace(get_capture_descs=_Recorder(result=[])),
        _cleanup_profiling_kv_cache=_Recorder(),
    )
    counter = SimpleNamespace(num_cudagraph_captured=7)
    cfg_calls = []
    with (
        _Patch(mr_module, "set_current_vllm_config", _set_config_calls(cfg_calls)),
        _Patch(mr_module, "compilation_counter", counter),
    ):
        assert ModelRunnerFL.profile_cudagraph_memory(runner) == 0

    assert cfg_calls == [runner.vllm_config]
    assert len(runner._init_minimal_kv_cache_for_profiling.calls) == 1
    assert len(runner._cleanup_profiling_kv_cache.calls) == 1


def test_profile_cudagraph_memory_estimates_shared_and_encoder_pools():
    """Shared-pool modes measure two descriptors and add the encoder estimate."""
    descs3 = [_desc(16), _desc(8), _desc(4)]
    warmup = _Recorder()
    wrapper = SimpleNamespace(graph_pool="orig-pool")
    keys = {"pi": _ClearableSet()}
    encoder_capture = _Recorder()
    encoder_manager = SimpleNamespace(
        get_num_graphs_to_capture=lambda: 2,
        token_budgets=[4, 8],
        capture=encoder_capture,
        clear=_Recorder(),
    )
    dispatcher = SimpleNamespace(
        get_capture_descs=lambda: [(CUDAGraphMode.PIECEWISE, descs3)],
        cudagraph_keys=keys,
        keys_initialized=True,
    )
    runner = SimpleNamespace(
        vllm_config=object(),
        _init_minimal_kv_cache_for_profiling=_Recorder(),
        _create_encoder_cudagraph_manager=_Recorder(result=encoder_manager),
        cudagraph_dispatcher=dispatcher,
        _warmup_and_capture=warmup,
        _freeze_gc=lambda: nullcontext(),
        common_attention_metadata_graph=None,
        maybe_remove_all_loras=_Recorder(),
        _cleanup_profiling_kv_cache=_Recorder(),
        lora_config=None,
        device="cpu",
    )
    platform = _MemPlatform([1000, 400, 1000, 600, 2000, 1500])
    counter = SimpleNamespace(num_cudagraph_captured=7)
    enabled, cleared = [], []
    with (
        _Patch(mr_module, "current_platform", platform),
        _Patch(mr_module, "compilation_counter", counter),
        _Patch(mr_module, "GraphWrapper", _fake_wrapper_cls([wrapper])),
        _Patch(mr_module, "BreakableCUDAGraphWrapper", _fake_wrapper_cls([])),
        _Patch(
            mr_module,
            "set_cudagraph_capturing_enabled",
            lambda value: enabled.append(value),
        ),
        _Patch(mr_module, "graph_capture", _capture_ctx([])),
        _Patch(mr_module, "_accelerator_synchronize", lambda: None),
        _Patch(mr_module, "_clear_decoder_graphs", lambda rt: cleared.append(rt)),
        _Patch(mr_module, "set_current_vllm_config", _set_config_calls([])),
        _Patch(
            mr_module.torch,
            "accelerator",
            SimpleNamespace(empty_cache=lambda: None),
        ),
    ):
        estimate = ModelRunnerFL.profile_cudagraph_memory(runner)

    # first capture = 600, second sample 400 -> 1 MiB floor, 2 extra graphs.
    expected = 600 + (1 << 20) * 2 + 500
    assert estimate == expected
    assert len(warmup.calls) == 2
    assert encoder_capture.calls[0][1] == {"graph_pool": "pool2"}
    # Cleanup restores wrapper pools and drops the dispatcher keys.
    assert wrapper.graph_pool == "orig-pool"
    assert dispatcher.keys_initialized is False
    assert keys["pi"].cleared == 1
    assert enabled == [True, False]
    assert cleared == [None]  # no graph_runtime attribute -> None
    assert counter.num_cudagraph_captured == 7
    assert len(runner.maybe_remove_all_loras.calls) == 1
    assert len(runner._cleanup_profiling_kv_cache.calls) == 1


def test_profile_cudagraph_memory_measures_all_private_pool_descriptors():
    """Private breakable pools profile every descriptor and add full costs."""
    descs3 = [_desc(16), _desc(8), _desc(4)]
    wrapper = SimpleNamespace(graph_pool=None)
    dispatcher = SimpleNamespace(
        get_capture_descs=lambda: [(CUDAGraphMode.PIECEWISE, descs3)],
        cudagraph_keys={},
        keys_initialized=True,
    )
    runner = SimpleNamespace(
        vllm_config=object(),
        _init_minimal_kv_cache_for_profiling=_Recorder(),
        _create_encoder_cudagraph_manager=_Recorder(result=None),
        cudagraph_dispatcher=dispatcher,
        _warmup_and_capture=_Recorder(),
        _freeze_gc=lambda: nullcontext(),
        common_attention_metadata_graph=None,
        maybe_remove_all_loras=_Recorder(),
        _cleanup_profiling_kv_cache=_Recorder(),
        lora_config=None,
        device="cpu",
    )
    platform = _MemPlatform([1000, 500, 1000, 900, 1000, 100])
    with (
        _Patch(mr_module, "current_platform", platform),
        _Patch(
            mr_module,
            "compilation_counter",
            SimpleNamespace(num_cudagraph_captured=0),
        ),
        _Patch(mr_module, "GraphWrapper", _fake_wrapper_cls([])),
        _Patch(mr_module, "BreakableCUDAGraphWrapper", _fake_wrapper_cls([wrapper])),
        _Patch(mr_module, "set_cudagraph_capturing_enabled", lambda v: None),
        _Patch(mr_module, "graph_capture", _capture_ctx([])),
        _Patch(mr_module, "_accelerator_synchronize", lambda: None),
        _Patch(mr_module, "_clear_decoder_graphs", lambda rt: None),
        _Patch(mr_module, "set_current_vllm_config", _set_config_calls([])),
        _Patch(
            mr_module.torch,
            "accelerator",
            SimpleNamespace(empty_cache=lambda: None),
        ),
    ):
        estimate = ModelRunnerFL.profile_cudagraph_memory(runner)

    assert len(runner._warmup_and_capture.calls) == 3
    assert estimate == (1 << 20) * 3


def test_create_encoder_manager_short_circuits_without_config():
    """The encoder manager is None unless mm-encoder graphs are enabled."""
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=False),
        supports_mm_inputs=True,
    )
    assert ModelRunnerFL._create_encoder_cudagraph_manager(runner) is None

    runner.compilation_config.cudagraph_mm_encoder = True
    runner.supports_mm_inputs = False
    assert ModelRunnerFL._create_encoder_cudagraph_manager(runner) is None


def test_create_encoder_manager_requires_model_support():
    """A model without the encoder-cudagraph protocol yields None."""
    interfaces = __import__(
        "vllm.model_executor.models.interfaces", fromlist=["supports_encoder_cudagraph"]
    )
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=True),
        supports_mm_inputs=True,
        get_model=lambda: "raw-model",
    )
    with _Patch(interfaces, "supports_encoder_cudagraph", lambda model: False):
        assert ModelRunnerFL._create_encoder_cudagraph_manager(runner) is None


def test_create_encoder_manager_builds_manager_for_supported_model():
    """A supported model is wrapped in the real manager with runner settings."""
    interfaces = __import__(
        "vllm.model_executor.models.interfaces", fromlist=["supports_encoder_cudagraph"]
    )
    fake_module = types.ModuleType("vllm.v1.worker.encoder_cudagraph")
    manager_calls = []

    class _Manager:
        def __init__(self, **kwargs):
            manager_calls.append(kwargs)

    fake_module.EncoderCudaGraphManager = _Manager
    raw = object()
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=True),
        supports_mm_inputs=True,
        vllm_config="config",
        device="cpu",
        dtype="float16",
        get_model=lambda: raw,
    )
    with (
        _Patch(interfaces, "supports_encoder_cudagraph", lambda model: model is raw),
        patch.dict(sys.modules, {"vllm.v1.worker.encoder_cudagraph": fake_module}),
    ):
        manager = ModelRunnerFL._create_encoder_cudagraph_manager(runner)

    assert isinstance(manager, _Manager)
    assert manager_calls[0] == {
        "vllm_config": "config",
        "device": "cpu",
        "dtype": "float16",
        "model": raw,
    }


def test_maybe_init_encoder_manager_only_creates_once():
    """An existing manager is kept; otherwise it is created exactly once."""
    runner = SimpleNamespace(encoder_cudagraph_manager=None)
    created = SimpleNamespace(kind="manager")
    runner._create_encoder_cudagraph_manager = _Recorder(result=created)

    ModelRunnerFL._maybe_init_encoder_cudagraph_manager(runner)
    assert runner.encoder_cudagraph_manager is created

    runner._create_encoder_cudagraph_manager = _Recorder(result=None)
    ModelRunnerFL._maybe_init_encoder_cudagraph_manager(runner)
    assert runner.encoder_cudagraph_manager is created
    assert runner._create_encoder_cudagraph_manager.calls == []


class _ClearableSet:
    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1
