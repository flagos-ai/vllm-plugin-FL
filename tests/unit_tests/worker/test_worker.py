# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for worker module.

Note: These tests require vllm >= 0.13.0 with profiler support.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


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


class TestAscendCpuBindingSwitch:
    @patch("vllm_fl.cpu_binding.bind_cpus")
    @patch("vllm_fl.worker.worker.current_platform")
    def test_cpu_binding_defaults_to_enabled_on_npu(
        self, mock_platform, mock_bind_cpus
    ):
        from vllm_fl.worker.worker import _maybe_bind_ascend_cpus

        mock_platform.device_type = "npu"
        _maybe_bind_ascend_cpus(SimpleNamespace(additional_config={}), 1)

        mock_bind_cpus.assert_called_once_with(1)

    @patch("vllm_fl.cpu_binding.bind_cpus")
    @patch("vllm_fl.worker.worker.current_platform")
    def test_cpu_binding_honors_explicit_disable(self, mock_platform, mock_bind_cpus):
        from vllm_fl.worker.worker import _maybe_bind_ascend_cpus

        mock_platform.device_type = "npu"
        config = SimpleNamespace(additional_config={"enable_cpu_binding": False})
        _maybe_bind_ascend_cpus(config, 0)

        mock_bind_cpus.assert_not_called()

    @patch("vllm_fl.cpu_binding.bind_cpus")
    @patch("vllm_fl.worker.worker.current_platform")
    def test_cpu_binding_is_ascend_only(self, mock_platform, mock_bind_cpus):
        from vllm_fl.worker.worker import _maybe_bind_ascend_cpus

        mock_platform.device_type = "cuda"
        _maybe_bind_ascend_cpus(SimpleNamespace(additional_config={}), 0)

        mock_bind_cpus.assert_not_called()

    @patch("vllm_fl.cpu_binding.bind_cpus", side_effect=RuntimeError("topology"))
    @patch("vllm_fl.worker.worker.current_platform")
    def test_cpu_binding_failure_is_non_fatal(
        self, mock_platform, mock_bind_cpus, caplog
    ):
        from vllm_fl.worker.worker import _maybe_bind_ascend_cpus

        mock_platform.device_type = "npu"
        _maybe_bind_ascend_cpus(SimpleNamespace(additional_config={}), 0)

        mock_bind_cpus.assert_called_once_with(0)
        assert "Skipping CPU binding" in caplog.text


class TestAscendMemoryProfilingSwitch:
    @patch("vllm_fl.worker.worker.MemorySnapshot.measure", autospec=True)
    @patch("vllm_fl.worker.worker.current_platform")
    @patch("vllm_fl.worker.worker.gc.collect")
    def test_memory_profiling_cleans_before_after_snapshot_and_classifies(
        self, mock_gc_collect, mock_platform, mock_measure
    ):
        """Keep FL's cleanup lifecycle aligned with vLLM 0.20.2.

        The post-profile snapshot must be taken only after Python objects and
        reclaimable allocator blocks have been released.  Otherwise temporary
        activations would be mistaken for persistent non-torch memory and the
        KV cache budget would be understated.
        """
        from vllm_fl.worker.worker import MemorySnapshot, memory_profiling_fl

        events = []
        mock_gc_collect.side_effect = lambda: events.append("gc")
        mock_platform.torch_device_fn.empty_cache.side_effect = (
            lambda: events.append("empty_cache")
        )
        mock_platform.torch_device_fn.reset_peak_memory_stats.side_effect = (
            lambda: events.append("reset_peak")
        )

        snapshots = [
            # before_profile: weights are resident, activations are not.
            {
                "torch_peak": 100,
                "free_memory": 7_000,
                "total_memory": 10_000,
                "cuda_memory": 3_000,
                "torch_memory": 2_000,
                "non_torch_memory": 1_000,
                "timestamp": 2.0,
            },
            # after_profile: temporary activation objects/cache are gone, but
            # 500 bytes of persistent non-torch allocations remain.
            {
                "torch_peak": 400,
                "free_memory": 6_500,
                "total_memory": 10_000,
                "cuda_memory": 3_500,
                "torch_memory": 2_000,
                "non_torch_memory": 1_500,
                "timestamp": 5.0,
            },
        ]

        def measure(snapshot):
            values = snapshots.pop(0)
            events.append(
                "measure_before" if not events.count("profile") else "measure_after"
            )
            for name, value in values.items():
                setattr(snapshot, name, value)

        mock_measure.side_effect = measure
        baseline = MemorySnapshot(auto_measure=False)
        baseline.non_torch_memory = 1_000

        with memory_profiling_fl(baseline, weights_memory=500) as result:
            events.append("profile")

        assert events == [
            "gc",
            "empty_cache",
            "reset_peak",
            "measure_before",
            "profile",
            "gc",
            "empty_cache",
            "measure_after",
        ]
        assert result.torch_peak_increase == 300
        assert result.non_torch_increase == 500
        assert result.non_kv_cache_memory == 1_300
        assert result.profile_time == 3.0

    @pytest.mark.parametrize(
        ("additional_config", "expected"),
        [
            (None, True),
            ({}, True),
            ({"enable_npu_memory_profiling": False}, False),
            ({"enable_npu_memory_profiling": True}, True),
            ("invalid", True),
        ],
    )
    def test_npu_memory_profiling_is_native_aligned_by_default(
        self, additional_config, expected
    ):
        from vllm_fl.worker.worker import _npu_memory_profiling_enabled

        config = SimpleNamespace(additional_config=additional_config)

        assert _npu_memory_profiling_enabled(config) is expected

    @patch("vllm_fl.worker.worker.gc.collect")
    @patch("vllm_fl.worker.worker.memory_profiling_fl")
    @patch("vllm_fl.worker.worker.current_platform")
    def test_enabled_profiles_and_uses_measured_memory_categories(
        self, mock_platform, mock_memory_profiling, mock_gc_collect
    ):
        from vllm_fl.worker.worker import WorkerFL

        mock_platform.device_type = "npu"
        mock_platform.torch_device_fn.memory_stats.return_value = {
            "allocated_bytes.all.peak": 400
        }

        profile_result = SimpleNamespace(
            before_profile=SimpleNamespace(torch_peak=100),
            after_profile=SimpleNamespace(free_memory=2500),
            weights_memory=500,
            torch_peak_increase=0,
            non_torch_increase=50,
            non_kv_cache_memory=0,
        )
        profile_context = MagicMock()
        profile_context.__enter__.return_value = profile_result
        profile_context.__exit__.return_value = False
        mock_memory_profiling.return_value = profile_context

        profile_run = MagicMock()
        worker = SimpleNamespace(
            vllm_config=SimpleNamespace(
                additional_config={"enable_npu_memory_profiling": True}
            ),
            cache_config=SimpleNamespace(
                kv_cache_memory_bytes=None,
                gpu_memory_utilization=0.9,
            ),
            model_runner=SimpleNamespace(
                model_memory_usage=500,
                profile_run=profile_run,
            ),
            init_snapshot=SimpleNamespace(free_memory=3000),
            requested_memory=2000,
            device="cpu",
        )

        available = WorkerFL.determine_available_memory.__wrapped__(worker)

        profile_run.assert_called_once_with()
        mock_memory_profiling.assert_called_once_with(
            worker.init_snapshot,
            weights_memory=500,
        )
        mock_platform.torch_device_fn.memory_stats.assert_called_once_with("cpu")
        assert profile_result.torch_peak_increase == 300
        assert profile_result.non_kv_cache_memory == 850
        assert worker.peak_activation_memory == 300
        assert worker.non_torch_memory == 50
        assert worker.available_kv_cache_memory_bytes == 1150
        assert available == 1150

    @patch("vllm_fl.worker.worker.gc.collect")
    @patch("vllm_fl.worker.worker.memory_profiling_fl")
    @patch("vllm_fl.worker.worker.current_platform")
    def test_disabled_keeps_half_budget_fallback(
        self, mock_platform, mock_memory_profiling, mock_gc_collect
    ):
        from vllm_fl.worker.worker import WorkerFL

        mock_platform.device_type = "npu"
        mock_platform.torch_device_fn.mem_get_info.return_value = (7000, 10000)

        profile_run = MagicMock()
        worker = SimpleNamespace(
            vllm_config=SimpleNamespace(
                additional_config={"enable_npu_memory_profiling": False}
            ),
            cache_config=SimpleNamespace(
                kv_cache_memory_bytes=None,
                gpu_memory_utilization=0.9,
            ),
            model_runner=SimpleNamespace(
                model_memory_usage=500,
                profile_run=profile_run,
            ),
            init_snapshot=SimpleNamespace(free_memory=7000),
            requested_memory=9000,
            device="cpu",
        )

        available = WorkerFL.determine_available_memory.__wrapped__(worker)

        profile_run.assert_not_called()
        mock_memory_profiling.assert_not_called()
        assert worker.peak_activation_memory == 3000
        assert worker.non_torch_memory == 0
        assert worker.available_kv_cache_memory_bytes == 3000
        assert available == 3000


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
