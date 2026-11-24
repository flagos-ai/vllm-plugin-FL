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

"""G10-D coverage: module helpers, AsyncIntermediateTensors and the
device-lifecycle methods of WorkerFL (814b).

Targets: _num_workspace_lanes 110-116, _normalize_process_group_backend
119-131, _is_hybrid_kv_model 140-160, _apply_worker_vendor_patches 163-176,
AsyncIntermediateTensors 179-208, _maybe_get_memory_pool_context 374-409,
_scoped_allocator_max_split 411-433, sleep 310-343, wake_up 345-366,
checkpoint_prepare/restore 368-372, handle_ft_command 602-604,
load_model 608-627, update_config 629-630, reload_weights 632-634,
get_kv_connector_handshake_metadata 879-898, get_kv_cache_spec 900-901,
update_max_model_len 909-919, initialize_from_config 922-951.
"""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.worker import worker as wmod


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


class _Ctx:
    def __init__(self, value=None):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


def _platform(**over):
    base = dict(
        device_type="cuda",
        vendor_name="nvidia",
        is_cuda_alike=lambda: True,
        is_cuda=lambda: True,
        is_xpu=lambda: False,
        is_cpu=lambda: False,
        is_rocm=lambda: False,
        is_sleep_mode_available=lambda: True,
    )
    base.update(over)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------
# module helpers
# --------------------------------------------------------------------------
def test_num_workspace_lanes_defaults_to_one():
    cfg = SimpleNamespace(speculative_config=None)
    assert wmod._num_workspace_lanes(cfg, True) == 1
    assert wmod._num_workspace_lanes(cfg, False) == 1


def test_num_workspace_lanes_two_for_dspark_v2():
    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(use_dspark=lambda: True)
    )
    assert wmod._num_workspace_lanes(cfg, True) == 2
    assert wmod._num_workspace_lanes(cfg, False) == 1


@pytest.mark.parametrize(
    "device_type,vendor,backend,expected",
    [
        ("ptpu", "x", "flagcx", "pccl"),
        ("ptpu", "x", "nccl", "pccl"),
        ("npu", "x", "flagcx", "hccl"),
        ("cuda", "amd", "flagcx", "nccl"),
        ("cuda", "nvidia", "flagcx", "flagcx"),
        ("cuda", "nvidia", "nccl", "nccl"),
    ],
)
def test_normalize_process_group_backend(device_type, vendor, backend, expected):
    platform = _platform(device_type=device_type, vendor_name=vendor)
    with _stack((wmod, "current_platform", platform)):
        assert wmod._normalize_process_group_backend(backend) == expected


def test_is_hybrid_kv_model_mamba_align():
    cfg = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    assert wmod._is_hybrid_kv_model(cfg) is True


def test_is_hybrid_kv_model_flag_true():
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        model_config=SimpleNamespace(is_hybrid=True),
    )
    assert wmod._is_hybrid_kv_model(cfg) is True


def test_is_hybrid_kv_model_layer_types():
    cfg = SimpleNamespace(
        cache_config=None,
        model_config=SimpleNamespace(
            is_hybrid=False,
            hf_text_config=SimpleNamespace(
                layer_types=["full_attention", "linear_attention"]
            ),
        ),
    )
    assert wmod._is_hybrid_kv_model(cfg) is True


def test_is_hybrid_kv_model_false_for_full_attention():
    cfg = SimpleNamespace(
        cache_config=None,
        model_config=SimpleNamespace(
            is_hybrid=False,
            hf_text_config=SimpleNamespace(layer_types=["full_attention"]),
        ),
    )
    assert wmod._is_hybrid_kv_model(cfg) is False


def test_apply_worker_vendor_patches_non_iluvatar_is_noop():
    with _stack((wmod, "current_platform", _platform(vendor_name="nvidia"))):
        assert wmod._apply_worker_vendor_patches() is None


# --------------------------------------------------------------------------
# AsyncIntermediateTensors
# --------------------------------------------------------------------------
class _Handle:
    def __init__(self):
        self.waits = 0

    def wait(self):
        self.waits += 1


def test_async_intermediate_tensors_waits_for_comm_once():
    handle = _Handle()
    post = []
    tensors = {"h": torch.zeros(2)}
    ait = wmod.AsyncIntermediateTensors(
        tensors, comm_handles=[handle], comm_postprocess=[lambda: post.append(1)]
    )
    ait.wait_for_comm()
    assert handle.waits == 1
    assert post == [1]
    ait.wait_for_comm()
    assert handle.waits == 1


def test_async_intermediate_tensors_tensors_access_triggers_wait():
    handle = _Handle()
    ait = wmod.AsyncIntermediateTensors({"h": torch.zeros(1)}, [handle], None)
    _ = ait.tensors
    assert handle.waits == 1


def test_async_intermediate_tensors_without_comm_is_noop():
    ait = wmod.AsyncIntermediateTensors({"h": torch.zeros(1)})
    ait.wait_for_comm()
    assert ait._comm_waited is True


# --------------------------------------------------------------------------
# _maybe_get_memory_pool_context
# --------------------------------------------------------------------------
def test_maybe_memory_pool_npu_nullcontext():
    worker = SimpleNamespace()
    with _stack((wmod, "current_platform", _platform(device_type="npu"))):
        with wmod.WorkerFL._maybe_get_memory_pool_context(worker, "weights") as v:
            assert v is None


def test_maybe_memory_pool_cuda_without_cumem_is_null():
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_cumem_allocator=False)
        )
    )
    with _stack((wmod, "current_platform", _platform())):
        with wmod.WorkerFL._maybe_get_memory_pool_context(worker, "kv_cache") as v:
            assert v is None


def test_maybe_memory_pool_xpu_without_sleep_is_null():
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_sleep_mode=False)
        )
    )
    platform = _platform(
        is_cuda_alike=lambda: False, is_xpu=lambda: True
    )
    with _stack((wmod, "current_platform", platform)):
        with wmod.WorkerFL._maybe_get_memory_pool_context(worker, "weights") as v:
            assert v is None


def test_maybe_memory_pool_cpu_is_null():
    worker = SimpleNamespace(vllm_config=SimpleNamespace(model_config=None))
    platform = _platform(
        is_cuda_alike=lambda: False, is_xpu=lambda: False, is_cpu=lambda: True
    )
    with _stack((wmod, "current_platform", platform)):
        with wmod.WorkerFL._maybe_get_memory_pool_context(worker, "weights") as v:
            assert v is None


def test_maybe_memory_pool_allocator_path_weights_guard():
    calls = []

    class _Alloc:
        def get_current_usage(self):
            return 0

        def use_memory_pool(self, tag):
            calls.append(tag)
            return _Ctx("pool")

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_sleep_mode=True)
        )
    )
    platform = _platform(
        is_cuda_alike=lambda: False,
        is_xpu=lambda: False,
        is_cpu=lambda: False,
        is_sleep_mode_available=lambda: True,
    )
    with _stack(
        (wmod, "current_platform", platform),
        (wmod, "get_mem_allocator_instance", lambda: _Alloc()),
    ):
        with wmod.WorkerFL._maybe_get_memory_pool_context(worker, "weights") as v:
            assert v == "pool"
    assert calls == ["weights"]


# --------------------------------------------------------------------------
# _scoped_allocator_max_split
# --------------------------------------------------------------------------
def test_scoped_allocator_max_split_non_cuda_is_passthrough():
    with _stack((wmod, "current_platform", _platform(is_cuda=lambda: False))):
        with wmod.WorkerFL._scoped_allocator_max_split(SimpleNamespace(), 20):
            pass


def test_scoped_allocator_max_split_sets_and_restores(monkeypatch):
    calls = []
    fake_torch = SimpleNamespace(
        _C=SimpleNamespace(
            _accelerator_setAllocatorSettings=lambda value: calls.append(value)
        )
    )
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    with _stack(
        (wmod, "current_platform", _platform(is_cuda=lambda: True)),
        (wmod, "torch", fake_torch),
    ):
        with wmod.WorkerFL._scoped_allocator_max_split(SimpleNamespace(), 20):
            pass
    assert calls[0] == "max_split_size_mb:20"
    assert calls[1] == "max_split_size_mb:64"


# --------------------------------------------------------------------------
# sleep / wake_up
# --------------------------------------------------------------------------
class _TorchAccel:
    def __init__(self, free=100, total=1000):
        self.free = free
        self.total = total
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1

    def get_memory_info(self):
        return self.free, self.total


def test_sleep_level_one_suspends_and_reports():
    accel = _TorchAccel()
    suspended = []
    worker = SimpleNamespace(
        _get_sleep_mode_backend=lambda: SimpleNamespace(
            suspend=lambda level: suspended.append(level)
        ),
    )
    with _stack(
        (wmod, "torch", SimpleNamespace(accelerator=accel)),
        (wmod, "current_platform", _platform(is_rocm=lambda: False)),
        (wmod, "logger", SimpleNamespace(info=lambda *a: None)),
    ):
        wmod.WorkerFL.sleep(worker, 1)
    assert suspended == [1]
    assert accel.syncs == 2


def test_sleep_level_two_saves_buffers():
    accel = _TorchAccel()
    saved = []
    model = SimpleNamespace(named_buffers=lambda: [("b", torch.ones(2))])
    worker = SimpleNamespace(
        _get_sleep_mode_backend=lambda: SimpleNamespace(suspend=lambda l: None),
        model_runner=SimpleNamespace(model=model),
        get_draft_model=lambda: None,
    )
    with _stack(
        (wmod, "torch", SimpleNamespace(accelerator=accel)),
        (wmod, "current_platform", _platform(is_rocm=lambda: False)),
        (wmod, "logger", SimpleNamespace(info=lambda *a: None)),
    ):
        wmod.WorkerFL.sleep(worker, 2)
    assert "b" in worker._sleep_saved_buffers
    assert worker._sleep_saved_buffers["b"].tolist() == [1.0, 1.0]


def test_wake_up_restores_buffers_and_kv_cache():
    resume = []
    kv_wake = []
    buffer = torch.zeros(2)
    model = SimpleNamespace(named_buffers=lambda: [("b", buffer)])
    worker = SimpleNamespace(
        _get_sleep_mode_backend=lambda: SimpleNamespace(
            resume=lambda tags: resume.append(tags)
        ),
        _sleep_saved_buffers={"b": torch.ones(2)},
        _sleep_saved_draft_buffers={},
        model_runner=SimpleNamespace(
            model=model, post_kv_cache_wake_up=lambda: kv_wake.append(1)
        ),
        get_draft_model=lambda: None,
    )
    wmod.WorkerFL.wake_up(worker, None)
    assert resume == [None]
    assert model.named_buffers()[0][1].tolist() == [1.0, 1.0]
    assert worker._sleep_saved_buffers == {}
    assert kv_wake == [1]


def test_wake_up_tags_weights_only_skips_kv_cache():
    resumed = []
    kv_wake = []
    worker = SimpleNamespace(
        _get_sleep_mode_backend=lambda: SimpleNamespace(
            resume=lambda tags: resumed.append(tags)
        ),
        _sleep_saved_buffers={},
        _sleep_saved_draft_buffers={},
        model_runner=SimpleNamespace(
            model=SimpleNamespace(named_buffers=lambda: []),
            post_kv_cache_wake_up=lambda: kv_wake.append(1),
        ),
        get_draft_model=lambda: None,
    )
    wmod.WorkerFL.wake_up(worker, ["weights"])
    assert resumed == [["weights"]]
    assert kv_wake == []


def test_checkpoint_prepare_and_restore():
    calls = []
    with _stack(
        (
            wmod,
            "checkpoint_prepare_distributed_state",
            lambda: calls.append("prepare"),
        ),
        (
            wmod,
            "checkpoint_restore_distributed_state",
            lambda: calls.append("restore"),
        ),
    ):
        wmod.WorkerFL.checkpoint_prepare(SimpleNamespace())
        wmod.WorkerFL.checkpoint_restore(SimpleNamespace())
    assert calls == ["prepare", "restore"]


# --------------------------------------------------------------------------
# handle_ft_command / load_model / update_config / reload_weights
# --------------------------------------------------------------------------
def test_handle_ft_command_delegates_to_sentinel():
    calls = []
    worker = SimpleNamespace(
        worker_sentinel=SimpleNamespace(
            handle_command=lambda req: calls.append(req) or "reply"
        )
    )
    assert wmod.WorkerFL.handle_ft_command(worker, "req") == "reply"
    assert calls == ["req"]


def test_load_model_without_weight_transfer_cuda():
    calls = []
    worker = SimpleNamespace(
        _maybe_get_memory_pool_context=lambda tag: _Ctx(),
        _scoped_allocator_max_split=lambda max_split_size_mb: _Ctx(),
        model_runner=SimpleNamespace(
            load_model=lambda load_dummy_weights=False: calls.append(
                load_dummy_weights
            )
        ),
        vllm_config=SimpleNamespace(weight_transfer_config=None),
        device=SimpleNamespace(type="cuda"),
    )
    with _stack((wmod, "set_current_vllm_config", lambda cfg: _Ctx())):
        wmod.WorkerFL.load_model(worker, load_dummy_weights=True)
    assert calls == [True]


def test_load_model_with_weight_transfer_and_npu():
    calls = []
    engine = SimpleNamespace()
    worker = SimpleNamespace(
        _maybe_get_memory_pool_context=lambda tag: _Ctx(),
        _scoped_allocator_max_split=lambda max_split_size_mb: _Ctx(),
        model_runner=SimpleNamespace(
            load_model=lambda load_dummy_weights=False: None,
            get_model=lambda: "model",
        ),
        vllm_config=SimpleNamespace(weight_transfer_config="wt"),
        device=SimpleNamespace(type="npu"),
    )
    factory = SimpleNamespace(
        create_engine=lambda *a: calls.append(a) or engine
    )
    with _stack(
        (wmod, "set_current_vllm_config", lambda cfg: _Ctx()),
        (wmod, "WeightTransferEngineFactory", factory),
        (wmod, "set_torch_threads_for_runtime", lambda: calls.append("threads")),
    ):
        wmod.WorkerFL.load_model(worker)
    assert worker.weight_transfer_engine is engine
    assert calls[0][0] == "wt"
    assert calls[-1] == "threads"


def test_update_config_and_reload_weights_delegate():
    calls = []
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            update_config=lambda ov: calls.append(("update", ov)),
            reload_weights=lambda *a, **k: calls.append(("reload", a, k)),
        ),
        vllm_config="cfg",
    )
    with _stack((wmod, "set_current_vllm_config", lambda cfg: _Ctx())):
        wmod.WorkerFL.update_config(worker, {"x": 1})
        wmod.WorkerFL.reload_weights(worker, 1, two=2)
    assert calls[0] == ("update", {"x": 1})
    assert calls[1][0] == "reload"


# --------------------------------------------------------------------------
# kv connector / kv spec / max len / initialize_from_config
# --------------------------------------------------------------------------
def test_kv_connector_handshake_none_without_transfer_group():
    with _stack((wmod, "has_kv_transfer_group", lambda: False)):
        assert (
            wmod.WorkerFL.get_kv_connector_handshake_metadata(SimpleNamespace())
            is None
        )


def test_kv_connector_handshake_none_without_metadata():
    with _stack(
        (wmod, "has_kv_transfer_group", lambda: True),
        (
            wmod,
            "get_kv_transfer_group",
            lambda: SimpleNamespace(get_handshake_metadata=lambda: None),
        ),
    ):
        assert (
            wmod.WorkerFL.get_kv_connector_handshake_metadata(SimpleNamespace())
            is None
        )


def test_kv_connector_handshake_returns_rank_keyed_metadata():
    with _stack(
        (wmod, "has_kv_transfer_group", lambda: True),
        (
            wmod,
            "get_kv_transfer_group",
            lambda: SimpleNamespace(get_handshake_metadata=lambda: "meta"),
        ),
        (
            wmod,
            "get_pp_group",
            lambda: SimpleNamespace(rank_in_group=1),
        ),
        (
            wmod,
            "get_tp_group",
            lambda: SimpleNamespace(rank_in_group=2),
        ),
    ):
        out = wmod.WorkerFL.get_kv_connector_handshake_metadata(SimpleNamespace())
    assert out == {(1, 2): "meta"}


def test_get_kv_cache_spec_delegates():
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(get_kv_cache_spec=lambda: {"h": "spec"})
    )
    assert wmod.WorkerFL.get_kv_cache_spec(worker) == {"h": "spec"}


def test_update_max_model_len_updates_config_and_runner():
    calls = []
    worker = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1),
        model_runner=SimpleNamespace(
            update_max_model_len=lambda n: calls.append(n)
        ),
    )
    with _stack((wmod, "logger", SimpleNamespace(debug=lambda *a: None))):
        wmod.WorkerFL.update_max_model_len(worker, 4096)
    assert worker.model_config.max_model_len == 4096
    assert calls == [4096]


def test_update_max_model_len_without_runner():
    worker = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1), model_runner=None
    )
    with _stack((wmod, "logger", SimpleNamespace(debug=lambda *a: None))):
        wmod.WorkerFL.update_max_model_len(worker, 8)
    assert worker.model_config.max_model_len == 8


def test_initialize_from_config_full_wiring():
    calls = []
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks=None),
        vllm_config="cfg",
        _maybe_get_memory_pool_context=lambda tag: _Ctx(),
        model_runner=SimpleNamespace(
            initialize_kv_cache=lambda cfg: calls.append("init-kv"),
            init_routed_experts_capturer=lambda: calls.append("experts"),
            _init_kv_zero_meta=lambda: calls.append("zero-meta"),
        ),
        model_config=SimpleNamespace(enable_return_routed_experts=True),
    )
    cfg = SimpleNamespace(num_blocks=10, needs_kv_cache_zeroing=True)
    with _stack(
        (
            wmod,
            "ensure_kv_transfer_initialized",
            lambda cfg_, kv: calls.append("connector"),
        )
    ):
        wmod.WorkerFL.initialize_from_config(worker, cfg)
    assert worker.cache_config.num_gpu_blocks == 10
    assert calls == ["connector", "init-kv", "experts", "zero-meta"]


def test_initialize_from_config_minimal():
    calls = []
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks=None),
        vllm_config="cfg",
        _maybe_get_memory_pool_context=lambda tag: _Ctx(),
        model_runner=SimpleNamespace(
            initialize_kv_cache=lambda cfg: calls.append("init-kv")
        ),
        model_config=SimpleNamespace(enable_return_routed_experts=False),
    )
    cfg = SimpleNamespace(num_blocks=4, needs_kv_cache_zeroing=False)
    with _stack(
        (wmod, "ensure_kv_transfer_initialized", lambda cfg_, kv: None)
    ):
        wmod.WorkerFL.initialize_from_config(worker, cfg)
    assert calls == ["init-kv"]
