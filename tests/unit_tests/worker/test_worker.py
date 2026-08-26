# Copyright (c) 2025 BAAI. All rights reserved.

"""Contract tests for the FL adaptation of vLLM v0.28.0 GPUWorker."""

import pytest


def has_vllm_worker() -> bool:
    try:
        from vllm_fl.worker.worker import WorkerFL  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


pytestmark = pytest.mark.skipif(
    not has_vllm_worker(), reason="vllm_fl.worker.worker is unavailable"
)


def test_worker_keeps_target_lifecycle_contract():
    from vllm_fl.worker.worker import WorkerFL

    required_methods = {
        "sleep",
        "wake_up",
        "checkpoint_prepare",
        "checkpoint_restore",
        "init_weight_transfer_engine",
        "start_weight_update",
        "start_draft_weight_update",
        "update_weights",
        "finish_weight_update",
        "elastic_ep_execute",
        "shutdown",
    }

    missing = sorted(name for name in required_methods if not hasattr(WorkerFL, name))
    assert not missing, f"WorkerFL is missing v0.28.0 methods: {missing}"


def test_worker_selects_v1_or_v2_model_runner():
    import inspect

    from vllm_fl.worker.worker import WorkerFL

    source = inspect.getsource(WorkerFL.init_device)
    assert "GPUModelRunnerV2" in source
    assert "vllm_fl.worker.model_runner" in source
    assert "ModelRunnerFL" in source


def test_platform_accepts_v2_model_runner():
    from types import SimpleNamespace

    from vllm.config import CUDAGraphMode
    from vllm_fl.platform import PlatformFL

    parallel_config = SimpleNamespace(
        worker_cls=None,
        all2all_backend=None,
        data_parallel_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=None,
        scheduler_config=SimpleNamespace(),
        cache_config=None,
        compilation_config=SimpleNamespace(
            compile_sizes=[],
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        attention_config=None,
        use_v2_model_runner=True,
    )

    PlatformFL.check_and_update_config(vllm_config)

    assert parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"
