# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for worker module.

Note: These tests require vllm >= 0.13.0 with profiler support.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def has_vllm_profiler():
    """Check if vllm profiler is available."""
    try:
        from vllm.profiler.wrapper import TorchProfilerWrapper  # noqa: F401

        return True
    except ImportError:
        return False


# Skip all tests if vllm profiler is not available
pytestmark = pytest.mark.skipif(
    not has_vllm_profiler(),
    reason="vllm.profiler.wrapper not available (requires vllm >= 0.13.0)",
)


@pytest.mark.parametrize("world_size", [2, 4])
def test_musa_workers_bind_config_before_patching_and_loading(monkeypatch, world_size):
    pytest.importorskip("torch_musa")

    import vllm_fl.worker.model_runner as runner_module
    import vllm_fl.worker.worker as worker_module
    from vllm_fl.patches import moe_sum

    events = []
    platform = SimpleNamespace(
        device_type="musa",
        dist_backend="mccl",
        logical_device_id_to_visible_device_id=lambda rank: rank,
        set_device=lambda device: events.append(("set_device", device)),
        check_if_supports_dtype=lambda dtype: None,
        empty_cache=lambda: None,
    )
    monkeypatch.setattr(worker_module, "current_platform", platform)
    monkeypatch.setattr(
        worker_module,
        "init_worker_distributed_environment",
        lambda *args: events.append(("dist", args[1])),
    )
    monkeypatch.setattr(worker_module, "set_random_seed", lambda seed: None)
    monkeypatch.setattr(worker_module.gc, "collect", lambda: None)
    monkeypatch.setattr(
        worker_module,
        "MemorySnapshot",
        lambda: SimpleNamespace(total_memory=100, free_memory=100),
    )
    monkeypatch.setattr(worker_module, "init_workspace_manager", lambda *args: None)
    monkeypatch.setattr(
        worker_module, "set_current_vllm_config", lambda config: nullcontext()
    )
    monkeypatch.setattr(worker_module, "report_usage_stats", lambda config: None)
    monkeypatch.setattr(
        moe_sum,
        "patch_vllm_moe_sum",
        lambda: events.append(("patch", None)),
    )

    for rank in range(world_size):
        events.clear()
        runner = MagicMock()
        expected_device = torch.device(f"musa:{rank}")
        runner.model.named_parameters.return_value = [
            ("weight", SimpleNamespace(device=expected_device))
        ]
        runner.load_model.side_effect = lambda **kwargs: events.append(("load", None))
        monkeypatch.setattr(
            runner_module, "ModelRunnerFL", lambda *args, runner=runner: runner
        )
        parallel = SimpleNamespace(
            distributed_executor_backend="ray",
            assigned_physical_gpu_ids=None,
            enable_dbo=False,
        )
        config = SimpleNamespace(
            parallel_config=parallel,
            device_config=SimpleNamespace(device=torch.device("musa:0")),
        )
        worker = SimpleNamespace(
            vllm_config=config,
            device_config=config.device_config,
            parallel_config=parallel,
            model_config=SimpleNamespace(dtype=torch.float16, seed=1),
            cache_config=SimpleNamespace(gpu_memory_utilization=0.5),
            local_rank=rank,
            rank=rank,
            distributed_init_method="test",
        )

        worker_module.WorkerFL.init_device(worker)
        worker_module.WorkerFL.load_model(worker)

        assert worker.device == expected_device
        assert worker.device_config.device == expected_device
        assert [name for name, _ in events] == ["set_device", "dist", "patch", "load"]
        runner.load_model.assert_called_once_with(load_dummy_weights=False)


def test_musa_worker_rejects_parameters_on_another_rank(monkeypatch):
    pytest.importorskip("torch_musa")

    import vllm_fl.worker.worker as worker_module
    from vllm_fl.patches import moe_sum

    runner = MagicMock()
    runner.model.named_parameters.return_value = [
        ("weight", SimpleNamespace(device=torch.device("musa:0")))
    ]
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(),
        model_runner=runner,
        device=torch.device("musa:1"),
        rank=1,
    )
    monkeypatch.setattr(
        worker_module, "current_platform", SimpleNamespace(device_type="musa")
    )
    monkeypatch.setattr(
        worker_module, "set_current_vllm_config", lambda config: nullcontext()
    )
    monkeypatch.setattr(moe_sum, "patch_vllm_moe_sum", lambda: None)

    with pytest.raises(RuntimeError, match="expected model parameters on musa:1"):
        worker_module.WorkerFL.load_model(worker)


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass behavior."""

    def test_default_values_without_auto_measure(self):
        """Test MemorySnapshot initializes with correct default values."""
        from vllm_fl.worker.worker import MemorySnapshot

        snapshot = MemorySnapshot(auto_measure=False)

        assert snapshot.torch_peak == 0
        assert snapshot.free_memory == 0
        assert snapshot.total_memory == 0
        assert snapshot.cuda_memory == 0
        assert snapshot.torch_memory == 0
        assert snapshot.non_torch_memory == 0

    def test_subtraction_computes_difference(self):
        """Test MemorySnapshot subtraction operator computes correct differences."""
        from vllm_fl.worker.worker import MemorySnapshot

        snapshot1 = MemorySnapshot(auto_measure=False)
        snapshot1.torch_peak = 1000
        snapshot1.free_memory = 5000
        snapshot1.total_memory = 10000
        snapshot1.cuda_memory = 5000
        snapshot1.torch_memory = 3000
        snapshot1.non_torch_memory = 2000
        snapshot1.timestamp = 10.0

        snapshot2 = MemorySnapshot(auto_measure=False)
        snapshot2.torch_peak = 500
        snapshot2.free_memory = 6000
        snapshot2.total_memory = 10000
        snapshot2.cuda_memory = 4000
        snapshot2.torch_memory = 2000
        snapshot2.non_torch_memory = 2000
        snapshot2.timestamp = 5.0

        diff = snapshot1 - snapshot2

        assert diff.torch_peak == 500
        assert diff.free_memory == -1000
        assert diff.cuda_memory == 1000
        assert diff.torch_memory == 1000
        assert diff.timestamp == 5.0


class TestMemoryProfilingResult:
    """Test MemoryProfilingResult dataclass behavior."""

    def test_default_values(self):
        """Test MemoryProfilingResult initializes with correct default values."""
        from vllm_fl.worker.worker import MemoryProfilingResult

        result = MemoryProfilingResult()

        assert result.weights_memory == 0
        assert result.torch_peak_increase == 0
        assert result.non_torch_increase == 0
        assert result.non_kv_cache_memory == 0
        assert result.profile_time == 0.0

    def test_creates_default_snapshots(self):
        """Test MemoryProfilingResult creates default snapshot objects."""
        from vllm_fl.worker.worker import MemoryProfilingResult

        result = MemoryProfilingResult()

        assert result.before_profile is not None
        assert result.after_profile is not None
