# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ``vllm_fl/worker/worker.py`` helpers.

CPU-only checks (stub doubles; no accelerator or distributed init):

* module-level helpers: ``_num_workspace_lanes``,
  ``_normalize_process_group_backend``, ``_is_hybrid_kv_model``,
  ``_apply_worker_vendor_patches``.
* ``AsyncIntermediateTensors`` lazy comm synchronization.
* ``WorkerFL`` weight-transfer session state machine
  (``_check_weight_transfer_engine``, ``start_weight_update``,
  ``start_draft_weight_update``, ``update_weights``,
  ``finish_weight_update``, ``_set_draft_weight_update_target``).
* ``WorkerFL.supports_draft_weight_updates`` and thin delegations.
* ``WorkerFL.shutdown`` cleanup ordering.
* ``WorkerFL.execute_model`` PP plumbing (async intermediate receives,
  non-blocking sends).
* ``WorkerFL.profile`` wrapper creation.
* ``WorkerFL.determine_available_memory`` txda path.
* jit-monitor gate and kernel-warmup vendor skips in
  ``compile_or_warm_up_model`` (ast-extracted).
"""

from __future__ import annotations

import ast
import __future__
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.supplementary]

worker_mod = pytest.importorskip("vllm_fl.worker.worker")

SOURCE = Path(worker_mod.__file__).resolve().parent
MIB = 1 << 20
GIB = 1 << 30


def _method(class_name, method_name, namespace):
    """Extract and execute one method of WorkerFL into ``namespace``."""
    path = SOURCE / "worker.py"
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(n for n in module.body if getattr(n, "name", None) == class_name)
    method = next(n for n in cls.body if getattr(n, "name", None) == method_name)
    method.decorator_list = []  # drop decorators (inference/profiling/etc.)
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        namespace,
    )
    return namespace[method_name]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "v2,spec,expected",
    [
        (False, None, 1),
        (True, None, 1),
        (True, NS(use_dspark=lambda: False), 1),
        (True, NS(use_dspark=lambda: True), 2),
        (False, NS(use_dspark=lambda: True), 1),
    ],
)
def test_num_workspace_lanes(v2, spec, expected):
    cfg = NS(speculative_config=spec, use_v2_model_runner=v2)
    assert worker_mod._num_workspace_lanes(cfg, v2) == expected


@pytest.mark.parametrize(
    "device_type,vendor,backend,expected",
    [
        ("ptpu", None, "flagcx", "pccl"),
        ("ptpu", None, "nccl", "pccl"),
        ("ptpu", None, "gloo", "gloo"),
        ("npu", None, "flagcx", "hccl"),
        ("npu", None, "nccl", "nccl"),
        ("cuda", "amd", "flagcx", "nccl"),
        ("cuda", "nvidia", "flagcx", "flagcx"),
        ("cuda", None, "nccl", "nccl"),
        ("musa", None, "flagcx", "flagcx"),
    ],
)
def test_normalize_process_group_backend_mapping(
    monkeypatch, device_type, vendor, backend, expected
):
    platform = NS(device_type=device_type)
    if vendor is not None:
        platform.vendor_name = vendor
    monkeypatch.setattr(worker_mod, "current_platform", platform)
    assert worker_mod._normalize_process_group_backend(backend) == expected


class TestIsHybridKvModel:
    def test_mamba_cache_mode_align_or_all(self):
        cfg = NS(
            cache_config=NS(mamba_cache_mode="align"),
            model_config=NS(is_hybrid=False, hf_text_config=NS(layer_types=None)),
        )
        assert worker_mod._is_hybrid_kv_model(cfg) is True

        cfg = NS(
            cache_config=NS(mamba_cache_mode="all"),
            model_config=NS(is_hybrid=False, hf_text_config=NS(layer_types=None)),
        )
        assert worker_mod._is_hybrid_kv_model(cfg) is True

    def test_model_config_is_hybrid_flag(self):
        cfg = NS(
            cache_config=NS(mamba_cache_mode="none"),
            model_config=NS(is_hybrid=True, hf_text_config=NS(layer_types=None)),
        )
        assert worker_mod._is_hybrid_kv_model(cfg) is True

    def test_is_hybrid_raises_falls_back_to_layer_types(self):
        class RaisingIsHybrid:
            # Some model configs raise while resolving is_hybrid before the
            # model is initialized; production catches Exception and falls
            # back to the HF layer-type list.
            @property
            def is_hybrid(self):
                raise RuntimeError("cannot resolve before init")

            @property
            def hf_text_config(self):
                return None

        cfg = NS(
            cache_config=NS(mamba_cache_mode="none"),
            model_config=RaisingIsHybrid(),
        )
        # The exception is swallowed; no layer types -> not hybrid.
        assert worker_mod._is_hybrid_kv_model(cfg) is False

    def test_layer_types_with_linear_attention(self):
        cfg = NS(
            cache_config=NS(mamba_cache_mode="none"),
            model_config=NS(
                is_hybrid=False,
                hf_text_config=NS(layer_types=["full_attention", "linear_attention"]),
            ),
        )
        assert worker_mod._is_hybrid_kv_model(cfg) is True

    def test_all_full_attention_layers(self):
        cfg = NS(
            cache_config=NS(mamba_cache_mode="none"),
            model_config=NS(
                is_hybrid=False,
                hf_text_config=NS(layer_types=["full_attention", "full_attention"]),
            ),
        )
        assert worker_mod._is_hybrid_kv_model(cfg) is False

    def test_missing_config_sections(self):
        assert worker_mod._is_hybrid_kv_model(NS()) is False
        assert worker_mod._is_hybrid_kv_model(NS(cache_config=None)) is False


def test_apply_worker_vendor_patches_noop_outside_iluvatar(monkeypatch):
    monkeypatch.setattr(
        worker_mod.current_platform, "vendor_name", "kunlunxin"
    )
    # Must not import the iluvatar vendor module.
    import sys

    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar", None
    )
    worker_mod._apply_worker_vendor_patches()


def test_apply_worker_vendor_patches_applies_on_iluvatar(monkeypatch):
    import sys

    monkeypatch.setattr(worker_mod.current_platform, "vendor_name", "iluvatar")
    calls = []
    fake = ModuleType("fake_iluvatar")
    fake.patch_torch_inductor_for_iluvatar = lambda: calls.append("inductor")
    fake.patch_sampler_compile_for_iluvatar = lambda: calls.append("sampler")
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar", fake
    )
    worker_mod._apply_worker_vendor_patches()
    assert calls == ["inductor", "sampler"]


def test_apply_worker_vendor_patches_swallows_import_errors(monkeypatch):
    import sys

    monkeypatch.setattr(worker_mod.current_platform, "vendor_name", "iluvatar")

    def broken_import():
        raise ImportError("vendor stack missing")

    real_import = __import__
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar", None
    )
    # A failed import inside the try block logs a warning and continues.
    import builtins

    original = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if "iluvatar.iluvatar" in name:
            raise ImportError("vendor stack missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    worker_mod._apply_worker_vendor_patches()  # must not raise


# ---------------------------------------------------------------------------
# AsyncIntermediateTensors
# ---------------------------------------------------------------------------


def _ait(tensors, handles=None, postprocess=None):
    return worker_mod.AsyncIntermediateTensors(
        tensors, comm_handles=handles, comm_postprocess=postprocess
    )


def test_async_intermediate_tensors_waits_for_handles():
    handle_a, handle_b = Mock(), Mock()
    post = Mock()
    ait = _ait({"t": object()}, handles=[handle_a, handle_b], postprocess=[post])
    # First access to .tensors waits on both handles and runs postprocess.
    ait.tensors
    handle_a.wait.assert_called_once_with()
    handle_b.wait.assert_called_once_with()
    post.assert_called_once_with()
    # Second access: waits are not repeated.
    ait.tensors
    handle_a.wait.assert_called_once_with()
    post.assert_called_once_with()


def test_async_intermediate_tensors_waits_once_without_handles():
    ait = _ait({"t": 1})
    assert ait.tensors == {"t": 1}
    # Attribute reads besides .tensors never trigger the wait.
    assert ait._comm_waited is True


def test_async_intermediate_tensors_empty_hook_lists():
    ait = _ait({"t": 1}, handles=None, postprocess=None)
    assert ait.tensors == {"t": 1}


# ---------------------------------------------------------------------------
# WorkerFL weight-transfer session state machine
# ---------------------------------------------------------------------------


def _worker(**attrs):
    worker = object.__new__(worker_mod.WorkerFL)
    worker.vllm_config = NS(model_config=NS(dtype=torch_dtype()))
    defaults = dict(
        weight_transfer_engine=None,
        _weight_update_active=False,
        _weight_update_is_draft=False,
        model_runner=Mock(),
        speculative_config=None,
        rank=0,
        local_rank=0,
        profiler=None,
    )
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(worker, key, value)
    return worker


def torch_dtype():
    import torch

    return torch.float16


class TestWeightTransferSession:
    def test_check_engine_requires_configuration(self):
        worker = _worker()
        with pytest.raises(RuntimeError, match="Weight transfer not configured"):
            worker._check_weight_transfer_engine()

    def test_start_weight_update_without_engine_raises(self):
        worker = _worker()
        with pytest.raises(RuntimeError):
            worker.start_weight_update()

    def test_start_weight_update_happy_path(self):
        engine = Mock(supports_draft_weight_update=True)
        worker = _worker(weight_transfer_engine=engine)
        worker.start_weight_update()
        engine.start_weight_update.assert_called_once_with()
        assert worker._weight_update_active is True
        assert worker._weight_update_is_draft is False

    def test_start_draft_weight_update_unsupported_engine(self):
        engine = Mock(supports_draft_weight_update=False)
        worker = _worker(weight_transfer_engine=engine)
        with pytest.raises(RuntimeError, match="does not support"):
            worker.start_draft_weight_update()

    def test_start_draft_weight_update_requires_draft_model(self):
        engine = Mock(supports_draft_weight_update=True)
        worker = _worker(weight_transfer_engine=engine)
        worker.get_draft_model = lambda: None
        with pytest.raises(RuntimeError, match="no draft model is configured"):
            worker.start_draft_weight_update()

    def test_start_draft_weight_update_requires_draft_config(self):
        engine = Mock(supports_draft_weight_update=True)
        worker = _worker(weight_transfer_engine=engine)
        worker.get_draft_model = lambda: object()
        with pytest.raises(RuntimeError, match="no draft model"):
            worker.start_draft_weight_update()

    def test_start_draft_weight_update_sets_target(self):
        engine = Mock(supports_draft_weight_update=True)
        worker = _worker(weight_transfer_engine=engine)
        draft = object()
        worker.get_draft_model = lambda: draft
        worker.speculative_config = NS(draft_model_config=NS())
        worker.start_draft_weight_update()
        engine.set_weight_update_target.assert_called_once_with(
            draft, worker.speculative_config.draft_model_config
        )
        assert worker._weight_update_is_draft is True

    def test_double_start_rejected_and_target_reset_on_failure(self):
        engine = Mock(supports_draft_weight_update=True)
        worker = _worker(weight_transfer_engine=engine, _weight_update_active=True)
        with pytest.raises(RuntimeError, match="already"):
            worker.start_weight_update()

        # Engine failure mid-start resets the target and re-raises.
        engine2 = Mock(
            supports_draft_weight_update=True,
            start_weight_update=Mock(side_effect=RuntimeError("boom")),
        )
        worker2 = _worker(weight_transfer_engine=engine2)
        with pytest.raises(RuntimeError, match="boom"):
            worker2.start_weight_update()
        engine2.reset_weight_update_target.assert_called_once_with()
        assert worker2._weight_update_active is False

    def test_update_weights_requires_active_session(self):
        engine = Mock(supports_draft_weight_update=True)
        worker = _worker(weight_transfer_engine=engine)
        with pytest.raises(RuntimeError, match="start_weight_update must be called"):
            worker.update_weights({"chunk": 1})

    def test_update_weights_failure_deactivates_session(self):
        engine = Mock(
            supports_draft_weight_update=True,
            update_weights=Mock(side_effect=RuntimeError("oom")),
        )
        worker = _worker(
            weight_transfer_engine=engine, _weight_update_active=True
        )
        with pytest.raises(RuntimeError, match="oom"):
            worker.update_weights({"chunk": 1})
        assert worker._weight_update_active is False
        engine.reset_weight_update_target.assert_called_once_with()

    def test_finish_weight_update_requires_active_session(self):
        engine = Mock(supports_draft_weight_update=True)
        worker = _worker(weight_transfer_engine=engine)
        with pytest.raises(RuntimeError, match="without a matching start"):
            worker.finish_weight_update()

    def test_finish_weight_update_clears_session(self):
        engine = Mock(supports_draft_weight_update=True)
        runner = Mock()
        worker = _worker(
            weight_transfer_engine=engine,
            _weight_update_active=True,
            _weight_update_is_draft=False,
            model_runner=runner,
        )
        worker.finish_weight_update()
        engine.finish_weight_update.assert_called_once_with()
        engine.reset_weight_update_target.assert_called_once_with()
        assert worker._weight_update_active is False
        # Non-draft updates reset the LoRA state on the runner.
        runner.reset_lora_state.assert_called_once_with()

    def test_finish_draft_update_skips_lora_reset(self):
        engine = Mock(supports_draft_weight_update=True)
        runner = Mock()
        worker = _worker(
            weight_transfer_engine=engine,
            _weight_update_active=True,
            _weight_update_is_draft=True,
            model_runner=runner,
        )
        worker.finish_weight_update()
        runner.reset_lora_state.assert_not_called()


def test_supports_draft_weight_updates_matrix():
    runner = Mock(get_draft_model=lambda: object())
    # No engine at all.
    worker = _worker(model_runner=runner)
    assert worker.supports_draft_weight_updates() is False

    # Engine without draft support.
    engine = Mock(supports_draft_weight_update=False)
    worker = _worker(weight_transfer_engine=engine, model_runner=runner)
    assert worker.supports_draft_weight_updates() is False

    # Engine with draft support but no speculative config.
    engine = Mock(supports_draft_weight_update=True)
    worker = _worker(weight_transfer_engine=engine, model_runner=runner)
    assert worker.supports_draft_weight_updates() is False

    # Full combination.
    worker.speculative_config = NS(draft_model_config=NS())
    assert worker.supports_draft_weight_updates() is True


def test_set_draft_weight_update_target_assertions():
    engine = Mock(supports_draft_weight_update=True)
    worker = _worker(weight_transfer_engine=engine)
    worker.get_draft_model = lambda: None
    with pytest.raises(RuntimeError, match="no draft model is configured"):
        worker._set_draft_weight_update_target()

    worker.get_draft_model = lambda: object()
    with pytest.raises(RuntimeError, match="no draft model"):
        worker._set_draft_weight_update_target()

    draft = object()
    worker.get_draft_model = lambda: draft
    worker.speculative_config = NS(draft_model_config=NS(model="d"))
    worker._set_draft_weight_update_target()
    engine.set_weight_update_target.assert_called_once_with(
        draft, worker.speculative_config.draft_model_config
    )


# ---------------------------------------------------------------------------
# Thin delegations and misc methods
# ---------------------------------------------------------------------------


def test_init_weight_transfer_engine_delegates():
    engine = Mock()
    worker = _worker(weight_transfer_engine=engine)
    worker.init_weight_transfer_engine({"backend": "nccl"})
    engine.parse_init_info.assert_called_once_with({"backend": "nccl"})
    engine.init_transfer_engine.assert_called_once_with(
        engine.parse_init_info.return_value
    )


def test_elastic_ep_execute_delegates():
    executor = Mock()
    worker = _worker()
    worker.elastic_ep_executor = executor
    worker.elastic_ep_execute("method", 1, k=2)
    executor.execute.assert_called_once_with("method", 1, k=2)


def test_handle_ft_command_requires_sentinel():
    worker = _worker()
    worker.worker_sentinel = Mock()
    worker.handle_ft_command("req")
    worker.worker_sentinel.handle_command.assert_called_once_with("req")


def test_update_max_model_len_delegates_to_runner():
    runner = Mock()
    worker = _worker(model_runner=runner)
    worker.model_config = NS(max_model_len=8)
    worker.update_max_model_len(16)
    assert worker.model_config.max_model_len == 16
    runner.update_max_model_len.assert_called_once_with(16)

    # A missing runner still updates the config without failing.
    worker2 = _worker()
    worker2.model_config = NS(max_model_len=8)
    worker2.model_runner = None
    worker2.update_max_model_len(32)
    assert worker2.model_config.max_model_len == 32


def test_execute_dummy_batch_uses_uniform_query_len():
    runner = Mock(uniform_decode_query_len=3)
    worker = _worker(model_runner=runner)
    worker.execute_dummy_batch()
    runner._dummy_run.assert_called_once_with(3, uniform_decode=True)


def test_checkpoint_round_trip_delegates(monkeypatch):
    calls = []
    monkeypatch.setattr(
        worker_mod, "checkpoint_prepare_distributed_state", lambda: calls.append("p")
    )
    monkeypatch.setattr(
        worker_mod, "checkpoint_restore_distributed_state", lambda: calls.append("r")
    )
    worker = _worker()
    worker.checkpoint_prepare()
    worker.checkpoint_restore()
    assert calls == ["p", "r"]


def test_get_kv_connector_handshake_metadata_variants(monkeypatch):
    # No KV transfer group at all.
    monkeypatch.setattr(worker_mod, "has_kv_transfer_group", lambda: False)
    worker = _worker()
    assert worker.get_kv_connector_handshake_metadata() is None

    # Connector with no handshake metadata.
    monkeypatch.setattr(worker_mod, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(
        worker_mod,
        "get_kv_transfer_group",
        lambda: NS(get_handshake_metadata=lambda: None),
    )
    assert worker.get_kv_connector_handshake_metadata() is None

    # Full path: metadata keyed by (pp_rank, tp_rank).
    metadata = object()
    monkeypatch.setattr(
        worker_mod,
        "get_kv_transfer_group",
        lambda: NS(get_handshake_metadata=lambda: metadata),
    )
    monkeypatch.setattr(
        worker_mod, "get_pp_group", lambda: NS(rank_in_group=1)
    )
    monkeypatch.setattr(
        worker_mod, "get_tp_group", lambda: NS(rank_in_group=2)
    )
    assert worker.get_kv_connector_handshake_metadata() == {(1, 2): metadata}


def test_annotate_profile_without_profiler_is_nullcontext():
    worker = _worker()
    worker.profiler = None
    with worker.annotate_profile(object()):
        pass


def test_annotate_profile_stopped_profiler_is_nullcontext():
    profiler = Mock()
    profiler.is_running = False
    worker = _worker()
    worker.profiler = profiler
    with worker.annotate_profile(object()):
        pass
    profiler.step.assert_called_once_with()


# ---------------------------------------------------------------------------
# determine_available_memory — txda short-circuit
# ---------------------------------------------------------------------------


def test_determine_available_memory_txda_budget(monkeypatch):
    platform = NS(device_type="txda")
    monkeypatch.setattr(worker_mod, "current_platform", platform)

    profile = NS(
        total_consumed=0,
        transient_peak_headroom=0,
        non_kv_cache_memory=0,
        after_profile=NS(free_memory=0),
    )
    runner = NS(
        device="txda:0",
        cache_config=NS(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.8
        ),
        init_snapshot=NS(total_memory=0, free_memory=0),
        requested_memory=0,
        model_runner=NS(
            model_memory_usage=0, profile_run=Mock(),
        ),
        vllm_config=NS(compilation_config=NS(cudagraph_mode=object())),
        model_config=NS(multimodal_config=object()),
        parallel_config=NS(_api_process_count=1),
    )
    namespace = {
        "current_platform": platform,
        "maybe_apply_startup_plan": Mock(),
        "memory_profiling": lambda *a, **k: nullcontext(profile),
        "nullcontext": nullcontext,
        "CUDAGraphMode": NS(NONE=object()),
        "envs": NS(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=False),
        "logger": Mock(),
        "init_logger": Mock(return_value=Mock()),
        "format_gib": str,
        "reserve_mm_ipc_gpu_memory": lambda amount, *a: amount,
        "os": NS(environ={}),
        "GiB_bytes": GIB,
    }
    method = _method("WorkerFL", "determine_available_memory", namespace)
    out = method(runner)
    assert out == 20 * GIB  # default TXDA budget


def test_determine_available_memory_txda_env_override(monkeypatch):
    import os

    platform = NS(device_type="txda")
    monkeypatch.setattr(worker_mod, "current_platform", platform)
    monkeypatch.setenv("VLLM_TXDA_KV_CACHE_SIZE", str(5 * GIB))

    profile = NS(
        total_consumed=0,
        transient_peak_headroom=0,
        non_kv_cache_memory=0,
        after_profile=NS(free_memory=0),
    )
    runner = NS(
        device="txda:0",
        cache_config=NS(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.8
        ),
        init_snapshot=NS(total_memory=0, free_memory=0),
        requested_memory=0,
        model_runner=NS(model_memory_usage=0, profile_run=Mock()),
        vllm_config=NS(compilation_config=NS(cudagraph_mode=object())),
        model_config=NS(multimodal_config=object()),
        parallel_config=NS(_api_process_count=1),
    )
    namespace = {
        "current_platform": platform,
        "maybe_apply_startup_plan": Mock(),
        "memory_profiling": lambda *a, **k: nullcontext(profile),
        "nullcontext": nullcontext,
        "CUDAGraphMode": NS(NONE=object()),
        "envs": NS(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=False),
        "logger": Mock(),
        "init_logger": Mock(return_value=Mock()),
        "format_gib": str,
        "reserve_mm_ipc_gpu_memory": lambda amount, *a: amount,
        "os": os,
        "GiB_bytes": GIB,
    }
    method = _method("WorkerFL", "determine_available_memory", namespace)
    assert method(runner) == 5 * GIB


def test_determine_available_memory_txda_rejects_bad_env(monkeypatch):
    import os

    platform = NS(device_type="txda")
    monkeypatch.setattr(worker_mod, "current_platform", platform)
    monkeypatch.setenv("VLLM_TXDA_KV_CACHE_SIZE", "not-a-number")

    profile = NS(
        total_consumed=0, transient_peak_headroom=0, non_kv_cache_memory=0,
        after_profile=NS(free_memory=0),
    )
    runner = NS(
        device="txda:0",
        cache_config=NS(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.8
        ),
        init_snapshot=NS(total_memory=0, free_memory=0),
        requested_memory=0,
        model_runner=NS(model_memory_usage=0, profile_run=Mock()),
        vllm_config=NS(compilation_config=NS(cudagraph_mode=object())),
        model_config=NS(multimodal_config=object()),
        parallel_config=NS(_api_process_count=1),
    )
    namespace = {
        "current_platform": platform,
        "maybe_apply_startup_plan": Mock(),
        "memory_profiling": lambda *a, **k: nullcontext(profile),
        "nullcontext": nullcontext,
        "CUDAGraphMode": NS(NONE=object()),
        "envs": NS(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=False),
        "logger": Mock(),
        "init_logger": Mock(return_value=Mock()),
        "format_gib": str,
        "reserve_mm_ipc_gpu_memory": lambda amount, *a: amount,
        "os": os,
        "GiB_bytes": GIB,
    }
    method = _method("WorkerFL", "determine_available_memory", namespace)
    with pytest.raises(ValueError, match="positive integer"):
        method(runner)


def test_determine_available_memory_txda_rejects_nonpositive(monkeypatch):
    import os

    platform = NS(device_type="txda")
    monkeypatch.setattr(worker_mod, "current_platform", platform)
    monkeypatch.setenv("VLLM_TXDA_KV_CACHE_SIZE", "0")

    profile = NS(
        total_consumed=0, transient_peak_headroom=0, non_kv_cache_memory=0,
        after_profile=NS(free_memory=0),
    )
    runner = NS(
        device="txda:0",
        cache_config=NS(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.8
        ),
        init_snapshot=NS(total_memory=0, free_memory=0),
        requested_memory=0,
        model_runner=NS(model_memory_usage=0, profile_run=Mock()),
        vllm_config=NS(compilation_config=NS(cudagraph_mode=object())),
        model_config=NS(multimodal_config=object()),
        parallel_config=NS(_api_process_count=1),
    )
    namespace = {
        "current_platform": platform,
        "maybe_apply_startup_plan": Mock(),
        "memory_profiling": lambda *a, **k: nullcontext(profile),
        "nullcontext": nullcontext,
        "CUDAGraphMode": NS(NONE=object()),
        "envs": NS(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=False),
        "logger": Mock(),
        "init_logger": Mock(return_value=Mock()),
        "format_gib": str,
        "reserve_mm_ipc_gpu_memory": lambda amount, *a: amount,
        "os": os,
        "GiB_bytes": GIB,
    }
    method = _method("WorkerFL", "determine_available_memory", namespace)
    with pytest.raises(ValueError, match="must be positive"):
        method(runner)


# ---------------------------------------------------------------------------
# execute_model PP plumbing (undecorated method execution)
# ---------------------------------------------------------------------------


class TestWorkerExecuteModel:
    def _method(self):
        from types import NoneType

        from vllm.sequence import IntermediateTensors
        from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

        namespace = dict(worker_mod.__dict__)
        namespace.update(
            {
                "get_pp_group": worker_mod.get_pp_group,
                "get_tp_group": worker_mod.get_tp_group,
                "ModelRunnerOutput": ModelRunnerOutput,
                "AsyncModelRunnerOutput": AsyncModelRunnerOutput,
                "IntermediateTensors": IntermediateTensors,
                "NoneType": NoneType,
            }
        )
        return _method("WorkerFL", "execute_model", namespace)

    def _worker(self, monkeypatch, last_rank, first_rank, output):
        group = NS(
            is_first_rank=first_rank,
            is_last_rank=last_rank,
            irecv_tensor_dict=lambda **kw: ({}, None, None),
            isend_tensor_dict=Mock(return_value=[Mock()]),
        )
        monkeypatch.setattr(worker_mod, "get_pp_group", lambda: group)
        monkeypatch.setattr(worker_mod, "get_tp_group", lambda: None)
        runner = NS(is_pooling_model=False, calls=0)

        def execute(so, tensors):
            return output(runner)

        runner.execute_model = execute
        return NS(
            model_runner=runner,
            use_v2_model_runner=False,
            _pp_send_work=[],
            annotate_profile=lambda _: nullcontext(),
            vllm_config=NS(
                parallel_config=NS(distributed_executor_backend="mp")
            ),
        ), group

    def test_first_rank_empty_output_returns_output(self, monkeypatch):
        from vllm.v1.outputs import ModelRunnerOutput

        sentinel = ModelRunnerOutput(req_ids=[], req_id_to_index={})

        def output(runner):
            return sentinel

        worker, _ = self._worker(monkeypatch, False, True, output)
        method = self._method()
        so = NS(total_num_scheduled_tokens=1)
        assert method(worker, so) is sentinel

    def test_non_last_rank_returns_none_after_send(self, monkeypatch):
        def output(runner):
            from vllm.sequence import IntermediateTensors

            return IntermediateTensors({"residual": object()})

        worker, group = self._worker(monkeypatch, False, True, output)
        method = self._method()
        so = NS(total_num_scheduled_tokens=1)
        assert method(worker, so) is None
        group.isend_tensor_dict.assert_called_once()
        # The send handles are retained for the next iteration's wait.
        assert worker._pp_send_work == group.isend_tensor_dict.return_value


# ---------------------------------------------------------------------------
# profile wrapper creation
# ---------------------------------------------------------------------------


class TestWorkerProfile:
    def test_requires_profiler_config(self):
        worker = _worker()
        worker.profiler_config = None
        with pytest.raises(RuntimeError, match="Profiling is not enabled"):
            worker.profile()

    def test_requires_profiler_field(self):
        worker = _worker()
        worker.profiler_config = NS(profiler=None)
        with pytest.raises(RuntimeError, match="Profiling is not enabled"):
            worker.profile()

    def test_stop_without_start_warns_and_returns(self):
        worker = _worker()
        worker.profiler_config = NS(profiler="torch")
        worker.profiler = None
        worker.profile(is_start=False)  # no raise

    def test_invalid_profiler_type_rejected(self):
        worker = _worker()
        worker.profiler_config = NS(profiler="bogus")
        with pytest.raises(ValueError, match="Invalid profiler"):
            worker.profile(is_start=True, profile_prefix="trace")

    def test_torch_profiler_started_and_stopped(self, monkeypatch):
        from vllm.profiler.wrapper import TorchProfilerWrapper

        worker = _worker()
        worker.profiler_config = NS(profiler="torch")
        made = Mock()
        monkeypatch.setattr(worker_mod, "TorchProfilerWrapper", made)
        import vllm.distributed.utils as dist_utils

        monkeypatch.setattr(
            dist_utils, "get_worker_rank_suffix", lambda global_rank: "rank-0"
        )
        worker.profile(is_start=True, profile_prefix="pre")
        made.assert_called_once()
        worker.profiler.start.assert_called_once_with()

        worker.profile(is_start=False)
        worker.profiler.stop.assert_called_once_with()
        # torch profilers are retained for reuse.
        assert worker.profiler is not None

    def test_proton_profiler_reset_on_stop(self, monkeypatch):
        worker = _worker()
        worker.profiler_config = NS(profiler="proton")
        made = Mock()
        monkeypatch.setattr(worker_mod, "ProtonProfilerWrapper", made)
        import vllm.distributed.utils as dist_utils

        monkeypatch.setattr(
            dist_utils, "get_worker_rank_suffix", lambda global_rank: "rank-0"
        )
        worker.profile(is_start=True)
        made.assert_called_once()
        assert worker.profiler is made.return_value
        worker.profiler.stop.assert_not_called()
        worker.profile(is_start=False)
        made.return_value.stop.assert_called_once_with()
        # Proton wrappers are rebuilt on the next start (fixed output names).
        assert worker.profiler is None


# ---------------------------------------------------------------------------
# shutdown ordering
# ---------------------------------------------------------------------------


def test_shutdown_releases_every_layer(monkeypatch):
    events = []

    monkeypatch.setattr(
        worker_mod, "ensure_kv_transfer_shutdown", lambda: events.append("kv")
    )
    monkeypatch.setattr(
        worker_mod, "ensure_ec_transfer_shutdown", lambda: events.append("ec")
    )
    runner = Mock()
    runner.shutdown = lambda: events.append("runner")
    worker = _worker(model_runner=runner)
    worker.profiler = Mock(shutdown=lambda: events.append("profiler"))
    worker.weight_transfer_engine = Mock(
        shutdown=lambda: events.append("weights")
    )
    # CPU platform: no CuMem pool release branch.
    monkeypatch.setattr(
        worker_mod.current_platform, "is_cuda_alike", lambda: False
    )
    worker.shutdown()
    # Order: kv, ec, profiler, weights, runner.
    assert events == ["kv", "ec", "profiler", "weights", "runner"]


def test_shutdown_tolerates_missing_attributes(monkeypatch):
    monkeypatch.setattr(
        worker_mod, "ensure_kv_transfer_shutdown", lambda: None
    )
    monkeypatch.setattr(
        worker_mod, "ensure_ec_transfer_shutdown", lambda: None
    )
    worker = _worker()
    del worker.weight_transfer_engine
    del worker.model_runner
    monkeypatch.setattr(
        worker_mod.current_platform, "is_cuda_alike", lambda: False
    )
    worker.shutdown()  # must not raise


# ---------------------------------------------------------------------------
# compile_or_warm_up_model — jit-monitor gate + kernel warmup skips
# ---------------------------------------------------------------------------


def _warmup_method(monkeypatch, device_type="cuda", vendor="nvidia",
                   npu_device=False, enable_jit_warmup=True):
    """ast-extract compile_or_warm_up_model with stub dependencies.

    The method body resolves ``kernel_warmup`` / ``activate_jit_monitor`` /
    ``warmup_kernels`` from this module's globals at call time, so patching
    the symbols on ``worker_mod`` fully diverts the heavy machinery.
    """
    activated = []
    monkeypatch.setattr(
        worker_mod, "kernel_warmup", lambda w: activated.append("kernel")
    )
    monkeypatch.setattr(
        worker_mod,
        "warmup_kernels",
        Mock(side_effect=lambda *a: activated.append("v2")),
    )
    import vllm.utils.jit_monitor as jm

    real_activate = jm.activate

    def record_activate(mode=None, verbose=False):
        activated.append("jit")

    monkeypatch.setattr(jm, "activate", record_activate)

    from vllm.config import CUDAGraphMode, CompilationMode

    runner = NS(
        _dummy_run=Mock(return_value=("hidden", "last-hidden")),
        maybe_remove_all_loras=Mock(),
        lora_config=None,
        capture_model=Mock(return_value=0),
        is_pooling_model=False,
        _dummy_sampler_run=Mock(),
        _dummy_pooler_run=Mock(),
        uniform_decode_query_len=1,
    )
    worker = NS(
        device=NS(type="npu" if npu_device else "cuda"),
        use_v2_model_runner=False,
        model_config=NS(
            enforce_eager=True,  # skip capture_model
            seed=0,
        ),
        scheduler_config=NS(max_num_seqs=4, max_num_batched_tokens=8),
        cache_config=NS(kv_cache_memory_bytes=None),
        lora_config=None,
        compilation_config=NS(
            mode=CompilationMode.NONE,
            compilation_time=0.0,
            encoder_compilation_time=0.0,
        ),
        vllm_config=NS(
            compilation_config=NS(mode=CompilationMode.NONE),
            kernel_config=NS(enable_jit_warmup=enable_jit_warmup),
            observability_config=NS(
                jit_monitor_mode="off", jit_monitor_verbose=False
            ),
        ),
        parallel_config=NS(tensor_parallel_size=1),
        observability_config=NS(
            jit_monitor_mode="off", jit_monitor_verbose=False
        ),
        model_runner=runner,
        execute_model=Mock(),
        sample_tokens=Mock(),
    )
    platform = NS(device_type=device_type, vendor_name=vendor)
    monkeypatch.setattr(worker_mod, "current_platform", platform)

    monkeypatch.setattr(
        worker_mod, "get_pp_group", lambda: NS(is_last_rank=True, world_size=1)
    )
    monkeypatch.setattr(worker_mod, "freeze_gc_heap", lambda: activated.append("gc"))
    monkeypatch.setattr(
        worker_mod, "enable_gpu_sync_check", lambda: activated.append("sync")
    )
    import vllm.utils.torch_utils as torch_utils

    monkeypatch.setattr(
        torch_utils, "set_torch_threads_for_runtime", lambda: None
    )

    namespace = dict(worker_mod.__dict__)
    method = _method("WorkerFL", "compile_or_warm_up_model", namespace)
    return method, worker, activated


@pytest.mark.parametrize(
    "device_type,vendor,npu_device,enable_jit_warmup,expect_jit",
    [
        # Non-vendor platforms always activate the monitor.
        ("cuda", "nvidia", False, False, True),
        ("cuda", "nvidia", False, True, True),
        # Kunlunxin: runtime JIT allowed -> only with warmup enabled.
        ("cuda", "kunlunxin", False, True, True),
        ("cuda", "kunlunxin", False, False, False),
        # Ascend (npu device): only with warmup enabled.
        ("npu", "ascend", True, True, True),
        ("npu", "ascend", True, False, False),
    ],
)
def test_jit_monitor_gate(
    monkeypatch, device_type, vendor, npu_device, enable_jit_warmup, expect_jit
):
    method, worker, activated = _warmup_method(
        monkeypatch,
        device_type=device_type,
        vendor=vendor,
        npu_device=npu_device,
        enable_jit_warmup=enable_jit_warmup,
    )
    method(worker)
    assert ("jit" in activated) is expect_jit


@pytest.mark.parametrize(
    "device_type,vendor,expect_kernel_warmup",
    [
        # txda/kunlunxin skip the generic warmup entirely.
        ("txda", None, False),
        ("cuda", "kunlunxin", False),
        ("cuda", "nvidia", True),
    ],
)
def test_kernel_warmup_vendor_skips(
    monkeypatch, device_type, vendor, expect_kernel_warmup
):
    method, worker, activated = _warmup_method(
        monkeypatch, device_type=device_type, vendor=vendor
    )
    method(worker)
    assert ("kernel" in activated) is expect_kernel_warmup


def test_v1_warmup_runs_sampler_run(monkeypatch):
    method, worker, activated = _warmup_method(monkeypatch)
    method(worker)
    worker.model_runner._dummy_sampler_run.assert_called_once()
    worker.model_runner._dummy_pooler_run.assert_not_called()


def test_v1_warmup_runs_pooler_run_for_pooling(monkeypatch):
    method, worker, activated = _warmup_method(monkeypatch)
    worker.model_runner.is_pooling_model = True
    method(worker)
    worker.model_runner._dummy_pooler_run.assert_called_once()
    worker.model_runner._dummy_sampler_run.assert_not_called()
