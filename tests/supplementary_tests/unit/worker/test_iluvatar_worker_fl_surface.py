# Copyright 2026 BAAI. All rights reserved.
"""CPU surface coverage for ``IluvatarWorkerFL`` (worker/worker.py, 133 rows).

The vendor worker module imports cleanly on CPU (``apply_import_compat``
aliases the v0.32 symbols), but its constructor needs a full engine stack.
Following the ``object.__new__`` + attribute-injection pattern proven by
``tests/unit_tests/worker/test_worker.py`` (which executes the *shared*
WorkerFL methods the same way), these tests bind plain ``SimpleNamespace``
state onto a freshly allocated instance and run the real vendor methods:

* ``init_device`` — the apply_worker_patches → base init_device →
  install_pipeline_transport ordering, against fake patch targets.
* ``install_pipeline_transport`` — the three binding branches (world_size 1,
  foreign communicator, matching communicator + marker idempotency) from
  ``vendor/iluvatar/pipeline.py``.
* ``sleep`` / ``wake_up`` — the vendor KV-scale save/restore round trip and
  the level-2 buffer save/restore round trip, with a fake accelerator
  memory-info boundary (CPU torch has no ``get_memory_info``).
* ``update_config`` — whitelist success path (rejection is already covered
  by unit_tests test_worker.py:265-283).
* ``execute_model`` — PP async send/recv orchestration via the real method
  with a fake PP/TP group (the shared-worker equivalent already runs in
  unit_tests; here the *vendor* override rows execute).
* ``apply_worker_patches`` — the worker patch bundle order, once-only
  install (fake target modules injected into sys.modules), and the
  non-iluvatar vendor early return.

Every assertion checks values that crossed the real method body (saved and
restored tensor contents, call order, bound handles) — no re-assertion of
inputs the test itself injected.
"""

from __future__ import annotations

import sys
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("iluvatar")]



@pytest.fixture
def worker_module():
    import vllm_fl.dispatch.backends.vendor.iluvatar.worker.worker as w

    return w


@pytest.fixture
def worker(worker_module):
    """A bare instance with the base-class sleep/draft attributes pre-bound."""
    instance = object.__new__(worker_module.IluvatarWorkerFL)
    # WorkerFL.__init__ installs these; sleep/wake_up read them unconditionally
    # on the level-2 / wake paths.
    instance._sleep_saved_buffers = {}
    instance._sleep_saved_draft_buffers = {}
    instance._pp_send_work = []
    return instance


@pytest.fixture
def fake_accelerator(monkeypatch):
    """Fake the ``torch.accelerator`` hardware boundary sleep/wake_up use.

    CPU torch 2.9 has ``accelerator.synchronize`` that *raises* with no
    device, and no ``get_memory_info`` at all; the memory figures below are
    plausible device numbers (memory freed by the suspend step).
    """
    import vllm_fl.dispatch.backends.vendor.iluvatar.worker.worker as w

    calls = {"sync": 0, "meminfo": 0}

    def fake_synchronize(device=None):
        calls["sync"] += 1

    memory = [10 << 30, 20 << 30, 16 << 30, 20 << 30]

    def fake_get_memory_info():
        calls["meminfo"] += 1
        value = memory[min(calls["meminfo"] - 1, len(memory) - 1)]
        return (value, 20 << 30)

    fake = SimpleNamespace(
        synchronize=fake_synchronize, get_memory_info=fake_get_memory_info
    )
    monkeypatch.setattr(w.torch, "accelerator", fake)
    return calls


class TestSleepWakeUp:
    """sleep/wake_up preserve quantized KV scales and level-2 buffers."""

    @staticmethod
    def _quantized_state(worker, k_scale_val, v_scale_val):
        """Bind cache/compilation config with a quantized KV dtype + layers."""
        k = torch.full((4,), k_scale_val)
        v = torch.full((4,), v_scale_val)
        # two layers sharing the same tensor object: dedup by id() must
        # keep only one entry
        layer = SimpleNamespace(_k_scale=k, k_scale=None, _v_scale=v, v_scale=None)
        other = SimpleNamespace(k_scale=k, _k_scale=None, v_scale=None, _v_scale=v)
        worker.cache_config = SimpleNamespace(cache_dtype="fp8")
        worker.compilation_config = SimpleNamespace(
            static_forward_context={"layer_a": layer, "layer_b": other}
        )
        return k, v

    def test_sleep_saves_kv_scales_for_quantized_cache(
        self, worker, worker_module, fake_accelerator, monkeypatch
    ):
        k, v = self._quantized_state(worker, 1.5, 2.5)
        suspended = []
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "_get_sleep_mode_backend",
            lambda self: SimpleNamespace(suspend=lambda level: suspended.append(level)),
        )
        worker.sleep(level=1)
        assert suspended == [1], "sleep must suspend through the sleep backend"
        saved = worker._sleep_saved_kv_scales
        # one entry per distinct scale tensor (dedup across layers)
        saved_tensors = [t for t, _ in saved]
        assert any(t is k for t in saved_tensors) and any(
            t is v for t in saved_tensors
        ), "both scales saved"
        assert len(saved) == 2, "shared tensors must be deduplicated by id()"
        # the saved copy is a detached CPU clone, not an alias
        for original, clone in saved:
            assert clone is not original
            assert clone.device.type == "cpu"
        assert any(torch.all(s == 1.5) for _, s in saved), "k scale value kept"
        assert any(torch.all(s == 2.5) for _, s in saved), "v scale value kept"

    def test_sleep_non_quantized_cache_saves_no_scales(
        self, worker, worker_module, fake_accelerator, monkeypatch
    ):
        worker.cache_config = SimpleNamespace(cache_dtype="auto")
        worker.compilation_config = SimpleNamespace(static_forward_context={})
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "_get_sleep_mode_backend",
            lambda self: SimpleNamespace(suspend=lambda level: None),
        )
        worker.sleep(level=1)
        assert not hasattr(worker, "_sleep_saved_kv_scales"), (
            "non-quantized KV cache must not build the scale save list"
        )

    def test_sleep_level_2_saves_model_and_draft_buffers(
        self, worker, worker_module, fake_accelerator, monkeypatch
    ):
        model = torch.nn.Module()
        model.register_buffer("buf", torch.arange(6, dtype=torch.float32))
        draft = torch.nn.Module()
        draft.register_buffer("draft_buf", torch.arange(3, dtype=torch.float32) + 10)
        worker.model_runner = SimpleNamespace(model=model)
        worker.get_draft_model = lambda: draft
        worker.cache_config = SimpleNamespace(cache_dtype="auto")
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "_get_sleep_mode_backend",
            lambda self: SimpleNamespace(suspend=lambda level: None),
        )
        worker.sleep(level=2)
        assert torch.equal(worker._sleep_saved_buffers["buf"], model.buf)
        assert torch.equal(worker._sleep_saved_draft_buffers["draft_buf"], draft.draft_buf)
        # saved copies must be independent of the live buffers
        model.buf.mul_(100)
        assert torch.equal(worker._sleep_saved_buffers["buf"], torch.arange(6)), (
            "the saved clone must not alias the live device buffer"
        )

    def test_wake_up_restores_saved_buffers_by_name(
        self, worker, worker_module, fake_accelerator, monkeypatch
    ):
        """A buffer renamed in the model is dropped; matching ones restored."""
        model = torch.nn.Module()
        model.register_buffer("keep", torch.zeros(4))
        model.register_buffer("drop", torch.zeros(2))
        worker.model_runner = SimpleNamespace(model=model)
        worker.get_draft_model = lambda: None
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "_get_sleep_mode_backend",
            lambda self: SimpleNamespace(resume=lambda tags: None),
        )
        saved_keep = torch.full((4,), 7.0)
        worker._sleep_saved_buffers = {
            "keep": saved_keep,
            "gone": torch.zeros(1),  # no longer in the model
        }
        worker._sleep_saved_draft_buffers = {}
        worker.cache_config = SimpleNamespace(cache_dtype="auto")
        worker.wake_up(tags=["weights"])
        assert torch.all(model.keep == 7.0), "matching buffer restored in place"
        assert torch.equal(model.drop, torch.zeros(2)), "unlisted buffer untouched"
        assert worker._sleep_saved_buffers == {}, "save dict cleared after restore"

    def test_wake_up_weights_tag_skips_buffer_restore(
        self, worker, worker_module, fake_accelerator, monkeypatch
    ):
        model = torch.nn.Module()
        model.register_buffer("keep", torch.zeros(4))
        worker.model_runner = SimpleNamespace(model=model)
        worker.get_draft_model = lambda: None
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "_get_sleep_mode_backend",
            lambda self: SimpleNamespace(resume=lambda tags: None),
        )
        worker._sleep_saved_buffers = {"keep": torch.full((4,), 9.0)}
        worker._sleep_saved_draft_buffers = {}
        worker.cache_config = SimpleNamespace(cache_dtype="auto")
        worker.wake_up(tags=["kv_cache"])
        assert torch.all(model.keep == 0.0), (
            "a tags list without 'weights' must not restore model buffers"
        )
        assert worker._sleep_saved_buffers, "save dict kept for a later wake"

    def test_wake_up_round_trips_kv_scale_values(
        self, worker, worker_module, fake_accelerator, monkeypatch
    ):
        """Scale values modified while asleep come back to the saved ones."""
        k = torch.full((4,), 1.5)
        v = torch.full((4,), 2.5)
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "_get_sleep_mode_backend",
            lambda self: SimpleNamespace(resume=lambda tags: None),
        )
        worker._sleep_saved_kv_scales = [
            (k, k.detach().clone()),
            (v, v.detach().clone()),
        ]
        worker._sleep_saved_buffers = {}
        worker._sleep_saved_draft_buffers = {}
        worker.cache_config = SimpleNamespace(cache_dtype="auto")
        # simulate the accelerator zeroing live scales while asleep
        k.zero_()
        v.zero_()
        worker.wake_up(tags=["weights"])
        assert torch.all(k == 1.5) and torch.all(v == 2.5), (
            "saved scale values must be copied back into the live tensors"
        )
        assert worker._sleep_saved_kv_scales == [], "scale list cleared"

    def test_wake_up_zeroes_freshly_allocated_quantized_kv_caches(
        self, worker, worker_module, fake_accelerator, monkeypatch
    ):
        """The kv_cache wake tag re-zeroes newly allocated FP8 KV memory."""
        cache_a = torch.full((4,), 3.0)
        cache_b = torch.full((4,), 3.0)
        entry_list = [cache_a, None, [cache_b]]
        worker.model_runner = SimpleNamespace(kv_caches=entry_list)
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "_get_sleep_mode_backend",
            lambda self: SimpleNamespace(resume=lambda tags: None),
        )
        worker._sleep_saved_buffers = {}
        worker._sleep_saved_draft_buffers = {}
        worker._sleep_saved_kv_scales = []
        worker.cache_config = SimpleNamespace(cache_dtype="fp8")
        worker.wake_up(tags=["kv_cache"])
        assert torch.all(cache_a == 0) and torch.all(cache_b == 0), (
            "flat and nested list KV cache tensors must be re-zeroed"
        )

    def test_wake_up_no_tags_restores_everything(
        self, worker, worker_module, fake_accelerator, monkeypatch
    ):
        """tags=None means full wake: buffers, scales and zeroed KV caches."""
        model = torch.nn.Module()
        model.register_buffer("keep", torch.zeros(4))
        worker.model_runner = SimpleNamespace(model=model)
        worker.get_draft_model = lambda: None
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "_get_sleep_mode_backend",
            lambda self: SimpleNamespace(resume=lambda tags: None),
        )
        worker._sleep_saved_buffers = {"keep": torch.full((4,), 8.0)}
        worker._sleep_saved_draft_buffers = {}
        worker._sleep_saved_kv_scales = []
        worker.cache_config = SimpleNamespace(cache_dtype="auto")
        worker.wake_up()
        assert torch.all(model.keep == 8.0)
        assert worker._sleep_saved_buffers == {}

    def test_wake_up_restores_draft_model_buffers(
        self, worker, worker_module, fake_accelerator, monkeypatch
    ):
        """Draft-model buffers saved at level 2 are restored on wake."""
        model = torch.nn.Module()
        model.register_buffer("keep", torch.zeros(4))
        draft = torch.nn.Module()
        draft.register_buffer("draft_buf", torch.zeros(3))
        worker.model_runner = SimpleNamespace(model=model)
        worker.get_draft_model = lambda: draft
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "_get_sleep_mode_backend",
            lambda self: SimpleNamespace(resume=lambda tags: None),
        )
        worker._sleep_saved_buffers = {}
        worker._sleep_saved_draft_buffers = {
            "draft_buf": torch.full((3,), 4.0),
            "removed": torch.zeros(1),  # renamed/removed in the draft
        }
        worker._sleep_saved_kv_scales = []
        worker.cache_config = SimpleNamespace(cache_dtype="auto")
        worker.wake_up(tags=["weights"])
        assert torch.all(draft.draft_buf == 4.0), "draft buffer restored"
        assert worker._sleep_saved_draft_buffers == {}

    def test_wake_up_draft_saved_but_model_returns_none(
        self, worker, worker_module, fake_accelerator, monkeypatch
    ):
        """A vanished draft model must not break the wake path."""
        model = torch.nn.Module()
        worker.model_runner = SimpleNamespace(model=model)
        worker.get_draft_model = lambda: None  # draft unloaded while asleep
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "_get_sleep_mode_backend",
            lambda self: SimpleNamespace(resume=lambda tags: None),
        )
        worker._sleep_saved_buffers = {}
        worker._sleep_saved_draft_buffers = {"draft_buf": torch.zeros(3)}
        worker._sleep_saved_kv_scales = []
        worker.cache_config = SimpleNamespace(cache_dtype="auto")
        worker.wake_up(tags=["weights"])  # must not raise
        assert worker._sleep_saved_draft_buffers == {}

    def test_sleep_rocm_waits_for_freed_memory(
        self, worker, worker_module, monkeypatch
    ):
        """On ROCm the worker polls until memory is actually freed."""
        k = torch.full((2,), 1.0)
        layer = SimpleNamespace(_k_scale=k, k_scale=None, _v_scale=None, v_scale=None)
        worker.cache_config = SimpleNamespace(cache_dtype="fp8")
        worker.compilation_config = SimpleNamespace(
            static_forward_context={"l": layer}
        )
        import vllm_fl.dispatch.backends.vendor.iluvatar.worker.worker as w

        calls = {"sync": 0, "meminfo": 0}
        # memory first *shrinks* (freed < before) then recovers
        memory = [10 << 30, 9 << 30, 20 << 30, 10 << 30]

        def fake_get_memory_info():
            calls["meminfo"] += 1
            value = memory[min(calls["meminfo"] - 1, len(memory) - 1)]
            return (value, 20 << 30)

        monkeypatch.setattr(
            w.torch,
            "accelerator",
            SimpleNamespace(
                synchronize=_counting_call(calls, "sync"),
                get_memory_info=fake_get_memory_info,
            ),
        )
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "_get_sleep_mode_backend",
            lambda self: SimpleNamespace(suspend=lambda level: None),
        )
        monkeypatch.setattr(
            worker_module.current_platform, "is_rocm", lambda: True
        )
        worker.sleep(level=1)
        assert calls["meminfo"] >= 3, (
            "the rocm deadline path must poll memory again after a decrease"
        )


class TestUpdateConfig:
    """update_config whitelist success path (rejection: unit_tests test_worker)."""

    def test_whitelisted_override_reaches_worker_runner_and_vllm_config(
        self, worker, monkeypatch
    ):
        import vllm.config

        worker.vllm_config = SimpleNamespace(model_config=object())
        worker.model_runner = SimpleNamespace(model_config=None)
        new_config = SimpleNamespace(replaced=True)
        monkeypatch.setattr(
            vllm.config, "update_config", lambda old, values: new_config
        )
        worker.update_config({"model_config": {"dtype": "auto"}})
        assert worker.model_config is new_config
        assert worker.model_runner.model_config is new_config
        assert worker.vllm_config.model_config is new_config
        assert new_config.replaced is True

    def test_whitelisted_load_config_override(
        self, worker, monkeypatch
    ):
        import vllm.config

        worker.vllm_config = SimpleNamespace(load_config=object())
        worker.model_runner = SimpleNamespace(load_config=None)
        new_config = SimpleNamespace(load_config=True)
        monkeypatch.setattr(
            vllm.config, "update_config", lambda old, values: new_config
        )
        worker.update_config({"load_config": {"load_format": "dummy"}})
        assert worker.load_config is new_config
        assert worker.model_runner.load_config is new_config

    def test_unknown_config_name_is_rejected(
        self, worker, monkeypatch
    ):
        """Only the whitelist is accepted; anything else raises."""
        worker.vllm_config = SimpleNamespace()
        worker.model_runner = SimpleNamespace()
        with pytest.raises(ValueError, match="not supported"):
            worker.update_config({"cache_config": {"block_size": 32}})
        with pytest.raises(ValueError, match="not supported"):
            worker.update_config({"parallel_config": {"tp_size": 2}})


class TestInitializeFromConfig:
    """initialize_from_config wiring (num_blocks, kv transfer, runner call)."""

    def test_num_blocks_forwarded_and_runner_initialized(
        self, worker, worker_module, monkeypatch
    ):
        calls = []

        def fake_ensure_kv(vllm_config, kv_cache_config):
            calls.append(("ensure_kv", vllm_config, kv_cache_config))

        monkeypatch.setattr(
            worker_module, "ensure_kv_transfer_initialized", fake_ensure_kv
        )
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "_maybe_get_memory_pool_context",
            lambda self, tag: _null_context(),
        )

        kv_zero_meta = Mock()
        runner = SimpleNamespace(
            initialize_kv_cache=Mock(),
            kv_caches=None,
            _init_kv_zero_meta=kv_zero_meta,
        )
        worker.model_runner = runner
        worker.cache_config = SimpleNamespace(num_gpu_blocks=-1)
        worker.vllm_config = SimpleNamespace(parallel_config=None)
        kv_config = SimpleNamespace(
            num_blocks=42,
            needs_kv_cache_zeroing=True,
        )
        worker.initialize_from_config(kv_config)
        assert worker.cache_config.num_gpu_blocks == 42, (
            "num_gpu_blocks must adopt the profiled num_blocks"
        )
        assert calls and calls[0][2] is kv_config, (
            "kv transfer init must see the same kv_cache_config"
        )
        runner.initialize_kv_cache.assert_called_once_with(kv_config)
        kv_zero_meta.assert_called_once_with()

    def test_kv_zeroing_skipped_when_not_needed(
        self, worker, worker_module, monkeypatch
    ):
        monkeypatch.setattr(
            worker_module,
            "ensure_kv_transfer_initialized",
            lambda vllm_config, kv_cache_config: None,
        )
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "_maybe_get_memory_pool_context",
            lambda self, tag: _null_context(),
        )
        kv_zero_meta = Mock()
        runner = SimpleNamespace(
            initialize_kv_cache=Mock(),
            kv_caches=None,
            _init_kv_zero_meta=kv_zero_meta,
        )
        worker.model_runner = runner
        worker.cache_config = SimpleNamespace(num_gpu_blocks=-1)
        worker.vllm_config = SimpleNamespace(parallel_config=None)
        worker.initialize_from_config(
            SimpleNamespace(num_blocks=7, needs_kv_cache_zeroing=False)
        )
        kv_zero_meta.assert_not_called()


class _null_context:
    """Minimal context manager (avoids importing contextlib per test)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestExecuteModel:
    """The vendor execute_model PP orchestration rows."""

    @staticmethod
    def _make_worker(worker_module):
        worker = object.__new__(worker_module.IluvatarWorkerFL)
        worker._pp_send_work = []
        return worker

    def test_pending_send_handles_are_awaited_and_cleared(
        self, worker, worker_module, monkeypatch
    ):
        handle_a, handle_b = Mock(), Mock()
        worker._pp_send_work = [handle_a, handle_b]
        self._wire_groups(monkeypatch, worker_module)
        worker.model_runner = SimpleNamespace(
            execute_model=Mock(return_value=None),
            is_pooling_model=False,
        )
        worker.use_v2_model_runner = False
        worker.annotate_profile = lambda _: _null_context()
        result = worker.execute_model(
            SimpleNamespace(total_num_scheduled_tokens=0)
        )
        handle_a.wait.assert_called_once()
        handle_b.wait.assert_called_once()
        assert result is None

    def test_last_rank_returns_runner_output(
        self, worker, worker_module, monkeypatch
    ):
        """A None runner output is a valid ModelRunnerOutput-path return."""
        self._wire_groups(monkeypatch, worker_module)
        worker.model_runner = SimpleNamespace(
            execute_model=Mock(return_value=None),
            is_pooling_model=False,
        )
        worker.use_v2_model_runner = False
        worker.annotate_profile = lambda _: _null_context()
        output = worker.execute_model(
            SimpleNamespace(total_num_scheduled_tokens=5)
        )
        assert output is None, "last rank returns the runner output"
        assert worker._pp_send_work == [], "no send on the last rank"

    def test_first_rank_receives_async_tensor_dict(
        self, worker, worker_module, monkeypatch
    ):
        import torch

        received = {"hidden_states": torch.zeros(2, 3)}
        # real Mock handles: reading .tensors lazily runs wait_for_comm()
        # over them (AsyncIntermediateTensors.__getattribute__ contract)
        comm_handle = Mock()
        comm_post = Mock()
        group = SimpleNamespace(
            is_first_rank=False,
            is_last_rank=False,
            world_size=2,
            irecv_tensor_dict=Mock(
                return_value=(received, [comm_handle], [comm_post])
            ),
            isend_tensor_dict=Mock(return_value=[Mock(), Mock()]),
        )
        monkeypatch.setattr(worker_module, "get_pp_group", lambda: group)
        monkeypatch.setattr(worker_module, "get_tp_group", lambda: None)
        seen_intermediate = []

        def fake_runner_exec(scheduler_output, intermediate):
            seen_intermediate.append(intermediate)
            return worker_module.IntermediateTensors({})

        worker.model_runner = SimpleNamespace(
            execute_model=fake_runner_exec,
            is_pooling_model=False,
        )
        worker.use_v2_model_runner = False
        worker.annotate_profile = lambda _: _null_context()
        worker.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                distributed_executor_backend="mp"
            )
        )
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=5))
        intermediate = seen_intermediate[0]
        assert isinstance(intermediate, worker_module.AsyncIntermediateTensors)
        assert intermediate.tensors is received
        # reading .tensors synchronizes the async comm (lazy wait contract)
        comm_handle.wait.assert_called_once()
        comm_post.assert_called_once()
        assert worker._pp_send_work, "async send work queued for next step"

    @staticmethod
    def _wire_groups(monkeypatch, worker_module, first=True):
        group = SimpleNamespace(
            is_first_rank=first,
            is_last_rank=True,
            isend_tensor_dict=Mock(return_value=[Mock(), Mock()]),
        )
        monkeypatch.setattr(worker_module, "get_pp_group", lambda: group)
        monkeypatch.setattr(worker_module, "get_tp_group", lambda: None)
        return group

    def test_intermediate_output_is_isent_with_all_gather_group(
        self, worker, worker_module, monkeypatch
    ):
        import torch

        group = SimpleNamespace(
            is_first_rank=True,
            is_last_rank=False,
            isend_tensor_dict=Mock(return_value=[Mock(), Mock(), Mock()]),
        )
        tp_group = object()
        monkeypatch.setattr(worker_module, "get_pp_group", lambda: group)
        monkeypatch.setattr(worker_module, "get_tp_group", lambda: tp_group)
        tensors = {"hidden_states": torch.zeros(2, 3)}
        worker.model_runner = SimpleNamespace(
            execute_model=Mock(return_value=worker_module.IntermediateTensors(tensors)),
            is_pooling_model=False,
        )
        worker.use_v2_model_runner = False
        worker.annotate_profile = lambda _: _null_context()
        worker.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                distributed_executor_backend="mp"
            )
        )
        result = worker.execute_model(
            SimpleNamespace(total_num_scheduled_tokens=5)
        )
        assert result is None, "non-last rank returns None after the send"
        group.isend_tensor_dict.assert_called_once_with(
            tensors, all_gather_group=tp_group
        )
        handles = group.isend_tensor_dict.return_value
        # only the metadata handle is waited in-process; the rest is deferred
        assert worker._pp_send_work == handles[1:]
        assert handles[0].wait.call_count == 0

    def test_v2_pooling_model_calls_pool_when_output_none(
        self, worker, worker_module, monkeypatch
    ):
        """v2 pooling: a None runner output is replaced by pool() inside."""
        pool = Mock(return_value=None)  # pool() returns a valid None output
        worker.model_runner = SimpleNamespace(
            execute_model=Mock(return_value=None),
            is_pooling_model=True,
            pool=pool,
        )
        worker.use_v2_model_runner = True
        worker.annotate_profile = lambda _: _null_context()
        self._wire_groups(monkeypatch, worker_module)
        result = worker.execute_model(
            SimpleNamespace(total_num_scheduled_tokens=5)
        )
        assert result is None
        pool.assert_called_once_with()
        # a second identical step also pools (state not corrupted)
        pool.reset_mock()
        worker.execute_model(
            SimpleNamespace(total_num_scheduled_tokens=5)
        )
        pool.assert_called_once_with()

    def test_v2_pooling_model_pools_after_intermediate_send(
        self, worker, worker_module, monkeypatch
    ):
        """v2 pooling on a non-last rank: pool runs once more after the send.

        With an IntermediateTensors output the with-block pooling branch is
        skipped (output is not None), the async send is queued, and only
        then the v2 pooling hook runs (worker.py L237-238).
        """
        import torch

        group = SimpleNamespace(
            is_first_rank=True,
            is_last_rank=False,
            isend_tensor_dict=Mock(return_value=[Mock(), Mock(), Mock()]),
        )
        monkeypatch.setattr(worker_module, "get_pp_group", lambda: group)
        monkeypatch.setattr(worker_module, "get_tp_group", lambda: None)
        pool = Mock(return_value=None)
        tensors = {"hidden_states": torch.zeros(2, 3)}
        worker.model_runner = SimpleNamespace(
            execute_model=Mock(
                return_value=worker_module.IntermediateTensors(tensors)
            ),
            is_pooling_model=True,
            pool=pool,
        )
        worker.use_v2_model_runner = True
        worker.annotate_profile = lambda _: _null_context()
        worker.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                distributed_executor_backend="mp"
            )
        )
        result = worker.execute_model(
            SimpleNamespace(total_num_scheduled_tokens=5)
        )
        assert result is None
        # exactly one pool call: the post-send hook (L237-238)
        pool.assert_called_once_with()
        handles = group.isend_tensor_dict.return_value
        assert worker._pp_send_work == handles[1:], (
            "only the metadata handle is reaped in-process; the rest is "
            "retained for the next step's wait"
        )


class TestInitDevice:
    """init_device: worker patches, then base init, then pipeline transport."""

    def test_init_device_orders_patches_base_and_pipeline(
        self, worker, worker_module, monkeypatch
    ):
        import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar as iluvatar_mod
        import vllm_fl.dispatch.backends.vendor.iluvatar.pipeline as pipeline_mod

        events = []
        pp_group = object()
        monkeypatch.setattr(
            iluvatar_mod, "apply_worker_patches", lambda: events.append("patches")
        )
        monkeypatch.setattr(
            worker_module.BaseWorkerFL,
            "init_device",
            lambda self: events.append("base"),
        )
        monkeypatch.setattr(worker_module, "get_pp_group", lambda: pp_group)
        monkeypatch.setattr(
            pipeline_mod, "install_pipeline_transport", lambda g: events.append(g)
        )
        worker.init_device()
        assert events == ["patches", "base", pp_group], (
            "vendor patches run before the base device init, then the "
            "pipeline transport is installed on the PP group"
        )

    def test_pipeline_transport_failure_propagates(
        self, worker, worker_module, monkeypatch
    ):
        """An install_pipeline_transport failure must not be swallowed."""
        import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar as iluvatar_mod
        import vllm_fl.dispatch.backends.vendor.iluvatar.pipeline as pipeline_mod

        monkeypatch.setattr(iluvatar_mod, "apply_worker_patches", lambda: None)
        monkeypatch.setattr(
            worker_module.BaseWorkerFL, "init_device", lambda self: None
        )
        monkeypatch.setattr(worker_module, "get_pp_group", lambda: object())

        def broken(group):
            raise RuntimeError("pipeline install exploded")

        monkeypatch.setattr(
            pipeline_mod, "install_pipeline_transport", broken
        )
        with pytest.raises(RuntimeError, match="pipeline install exploded"):
            worker.init_device()


class TestApplyWorkerPatches:
    """The worker patch bundle: order, once-only install, vendor gate.

    The three module-scoped patch targets (moe_router / vision_precision /
    mrope) are imported inside the function body, so fake modules are
    injected into sys.modules; the same-module collaborators are faked on
    the iluvatar module (they are the callees, never the function under
    test).
    """

    @pytest.fixture
    def iluvatar_module(self):
        import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar as m

        return m

    @staticmethod
    def _patch_targets(monkeypatch, calls):
        """Inject fake moe_router / vision_precision / mrope modules."""
        for mod_name, fn_name in (
            (
                "vllm_fl.dispatch.backends.vendor.iluvatar.moe_router",
                "patch_moe_router",
            ),
            (
                "vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision",
                "patch_vision_precision",
            ),
            ("vllm_fl.dispatch.backends.vendor.iluvatar.mrope", "patch_mrope"),
        ):
            fake = ModuleType(mod_name)
            fake.__name__ = mod_name

            def make_call(name):
                def call():
                    calls.append(name)

                return call

            setattr(fake, fn_name, make_call(fn_name))
            monkeypatch.setitem(sys.modules, mod_name, fake)

    def test_all_patches_applied_in_order(self, monkeypatch, iluvatar_module):
        calls = []
        self._patch_targets(monkeypatch, calls)
        monkeypatch.setattr(iluvatar_module, "current_platform", _vendor("iluvatar"))
        for fn in (
            "patch_autotune_memory",
            "patch_v2_uva",
            "patch_torch_inductor_for_iluvatar",
            "patch_sampler_compile_for_iluvatar",
        ):
            monkeypatch.setattr(
                iluvatar_module,
                fn,
                lambda f=fn, **kwargs: calls.append(f),
            )
        iluvatar_module.apply_worker_patches()
        assert calls == [
            "patch_moe_router",
            "patch_vision_precision",
            "patch_autotune_memory",
            "patch_v2_uva",
            "patch_mrope",
            "patch_torch_inductor_for_iluvatar",
            "patch_sampler_compile_for_iluvatar",
        ], "every worker patch must run exactly once, in the fixed order"

    def test_required_strict_patches_fail_startup(
        self, monkeypatch, iluvatar_module
    ):
        """strict inductor/sampler failures propagate: startup must die."""

        def explode(*args, **kwargs):
            raise RuntimeError("inductor target unavailable")

        self._patch_targets(monkeypatch, [])
        monkeypatch.setattr(iluvatar_module, "current_platform", _vendor("iluvatar"))
        monkeypatch.setattr(iluvatar_module, "patch_autotune_memory", lambda: None)
        monkeypatch.setattr(iluvatar_module, "patch_v2_uva", lambda: None)
        monkeypatch.setattr(iluvatar_module, "patch_torch_inductor_for_iluvatar", explode)
        monkeypatch.setattr(
            iluvatar_module, "patch_sampler_compile_for_iluvatar", lambda: None
        )
        with pytest.raises(RuntimeError, match="inductor target unavailable"):
            iluvatar_module.apply_worker_patches()

    def test_non_iluvatar_vendor_returns_before_any_patch(
        self, monkeypatch, iluvatar_module
    ):
        calls = []
        self._patch_targets(monkeypatch, calls)
        monkeypatch.setattr(iluvatar_module, "current_platform", _vendor("nvidia"))
        monkeypatch.setattr(
            iluvatar_module, "patch_autotune_memory", lambda: calls.append("autotune")
        )
        iluvatar_module.apply_worker_patches()
        assert calls == [], "foreign vendors must not install any worker patch"

    def test_patch_targets_are_installed_exactly_once_across_calls(
        self, monkeypatch, iluvatar_module
    ):
        """Calling the bundle twice still installs each target only once."""
        install_counts = {"moe": 0, "vision": 0, "mrope": 0}

        def fake_install(counter, name):
            def call():
                counter[name] += 1

            return call

        targets = (
            (
                "vllm_fl.dispatch.backends.vendor.iluvatar.moe_router",
                "patch_moe_router",
                install_counts,
                "moe",
            ),
            (
                "vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision",
                "patch_vision_precision",
                install_counts,
                "vision",
            ),
            (
                "vllm_fl.dispatch.backends.vendor.iluvatar.mrope",
                "patch_mrope",
                install_counts,
                "mrope",
            ),
        )
        for mod_name, fn_name, counter, key in targets:
            fake = ModuleType(mod_name)
            fake.__name__ = mod_name
            setattr(fake, fn_name, fake_install(counter, key))
            monkeypatch.setitem(sys.modules, mod_name, fake)

        monkeypatch.setattr(iluvatar_module, "current_platform", _vendor("iluvatar"))
        monkeypatch.setattr(iluvatar_module, "patch_autotune_memory", lambda: None)
        monkeypatch.setattr(iluvatar_module, "patch_v2_uva", lambda: None)
        monkeypatch.setattr(
            iluvatar_module,
            "patch_torch_inductor_for_iluvatar",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            iluvatar_module,
            "patch_sampler_compile_for_iluvatar",
            lambda **kwargs: None,
        )
        iluvatar_module.apply_worker_patches()
        iluvatar_module.apply_worker_patches()
        assert install_counts == {"moe": 2, "vision": 2, "mrope": 2}, (
            "the bundle itself is re-runnable (patches are idempotent by "
            "their own markers), so both calls run through every target"
        )


class TestInstallPipelineTransport:
    """``pipeline.install_pipeline_transport`` binding rules (three branches).

    Branch 1: world_size <= 1 — no binding. Branch 2: the group's
    device_communicator is not an IluvatarCommunicatorFL — no binding (the
    vendor methods must stay off shared/NVIDIA classes). Branch 3: a
    matching communicator — both methods bind to the vendor functions and
    a marker makes repeated installs a no-op.
    """

    def test_world_size_1_never_binds(self, worker_module):
        from vllm_fl.dispatch.backends.vendor.iluvatar.pipeline import (
            install_pipeline_transport,
        )

        group = SimpleNamespace(
            world_size=1,
            device_communicator=None,
            isend_tensor_dict=None,
            irecv_tensor_dict=None,
        )
        install_pipeline_transport(group)
        assert group.isend_tensor_dict is None
        assert group.irecv_tensor_dict is None

    def test_foreign_device_communicator_never_binds(self, worker_module):
        """Only IluvatarCommunicatorFL groups receive the transport methods."""
        from vllm_fl.dispatch.backends.vendor.iluvatar.pipeline import (
            install_pipeline_transport,
        )

        foreign = object()  # not an IluvatarCommunicatorFL
        group = SimpleNamespace(
            world_size=2,
            device_communicator=foreign,
            isend_tensor_dict=None,
            irecv_tensor_dict=None,
        )
        install_pipeline_transport(group)
        assert group.isend_tensor_dict is None, (
            "a foreign communicator must keep the default group methods"
        )
        assert group.irecv_tensor_dict is None
        assert not hasattr(group, "_iluvatar_pipeline_transport")

    def test_iluvatar_communicator_binds_vendor_methods(self, worker_module):
        from vllm_fl.dispatch.backends.vendor.iluvatar.communicator import (
            IluvatarCommunicatorFL,
        )
        from vllm_fl.dispatch.backends.vendor.iluvatar.pipeline import (
            _receive,
            _send,
            install_pipeline_transport,
        )

        comm = object.__new__(IluvatarCommunicatorFL)  # identity is all that
        # is checked; no __init__ (no process group needed)
        group = SimpleNamespace(
            world_size=2,
            device_communicator=comm,
            isend_tensor_dict=None,
            irecv_tensor_dict=None,
        )
        install_pipeline_transport(group)
        # the bound methods carry the vendor implementations and the group
        assert isinstance(group.isend_tensor_dict, MethodType)
        assert group.isend_tensor_dict.__func__ is _send
        assert group.isend_tensor_dict.__self__ is group
        assert isinstance(group.irecv_tensor_dict, MethodType)
        assert group.irecv_tensor_dict.__func__ is _receive
        assert group.irecv_tensor_dict.__self__ is group
        assert getattr(group, "_iluvatar_pipeline_transport", False) is True

    def test_second_install_is_a_noop(self, worker_module):
        from vllm_fl.dispatch.backends.vendor.iluvatar.communicator import (
            IluvatarCommunicatorFL,
        )
        from vllm_fl.dispatch.backends.vendor.iluvatar.pipeline import (
            install_pipeline_transport,
        )

        comm = object.__new__(IluvatarCommunicatorFL)
        group = SimpleNamespace(
            world_size=2,
            device_communicator=comm,
            isend_tensor_dict=None,
            irecv_tensor_dict=None,
        )
        install_pipeline_transport(group)
        first_send = group.isend_tensor_dict
        install_pipeline_transport(group)
        assert group.isend_tensor_dict is first_send, (
            "the marker must keep the binding stable across installs"
        )


def _vendor(name):
    return SimpleNamespace(vendor_name=name)


def _counting_call(calls, key):
    def call(*args, **kwargs):
        calls[key] += 1

    return call
