"""Exercise the budget and graph profiler without loading accelerator libraries."""

import ast
import unittest
from contextlib import contextmanager, nullcontext
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]


class Mode(IntEnum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2


def method(file, class_name, name, namespace):
    tree = ast.parse((ROOT / "vllm_fl/worker" / file).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    node = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(file), "exec"), namespace
    )
    return namespace[name]


class TestPPUBudget(unittest.TestCase):
    def budget(
        self,
        vendor,
        device_type="cuda",
        mode=Mode.FULL,
        enabled=True,
        graph_error=False,
    ):
        peak = {"value": 40}
        device = SimpleNamespace(
            reset_peak_memory_stats=Mock(),
            memory_stats=lambda: {"allocated_bytes.all.peak": peak["value"]},
        )
        platform = SimpleNamespace(
            device_type=device_type,
            vendor_name=vendor,
            is_cuda=lambda: vendor == "nvidia",
            empty_cache=Mock(),
            torch_device_fn=device,
        )

        def estimate_graph():
            peak["value"] = 240
            return 100

        graph = Mock(side_effect=estimate_graph)
        if graph_error:
            graph.side_effect = RuntimeError("capture failed")
        runner = SimpleNamespace(
            model_memory_usage=160, profile_run=Mock(), profile_cudagraph_memory=graph
        )
        worker = SimpleNamespace(
            cache_config=SimpleNamespace(
                kv_cache_memory_bytes=0, gpu_memory_utilization=0.9
            ),
            init_snapshot=SimpleNamespace(free_memory=1000, total_memory=1000),
            requested_memory=900,
            model_runner=runner,
            vllm_config=SimpleNamespace(
                compilation_config=SimpleNamespace(cudagraph_mode=mode)
            ),
        )

        @contextmanager
        def profiling(snapshot, weights_memory):
            yield SimpleNamespace(
                before_profile=SimpleNamespace(torch_peak=0),
                after_profile=SimpleNamespace(free_memory=800),
                non_torch_increase=10,
                weights_memory=weights_memory,
            )

        function = method(
            "worker.py",
            "WorkerFL",
            "determine_available_memory",
            {
                "current_platform": platform,
                "GiB_bytes": 1 << 30,
                "memory_profiling_fl": profiling,
                "CUDAGraphMode": Mode,
                "envs": SimpleNamespace(
                    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=enabled
                ),
                "logger": Mock(),
                "gc": SimpleNamespace(collect=lambda: None),
            },
        )
        return function, worker, graph

    def test_ppu_graph_cost_is_deducted_from_kv_budget(self):
        function, worker, graph = self.budget("thead")
        self.assertEqual(function(worker), 590)
        graph.assert_called_once()
        self.assertEqual(worker.peak_activation_memory, 40)
        self.assertEqual(worker.cudagraph_memory_estimate, 100)

    def test_nvidia_budget_is_unchanged(self):
        function, worker, graph = self.budget("nvidia")
        self.assertEqual(function(worker), 590)
        graph.assert_called_once()

    def test_other_platforms_do_not_gain_an_unverified_graph_path(self):
        for vendor, device_type in [("metax", "cuda"), ("thead", "xpu")]:
            with self.subTest(vendor=vendor, device_type=device_type):
                function, worker, graph = self.budget(vendor, device_type)
                self.assertEqual(function(worker), 690)
                graph.assert_not_called()

    def test_eager_mode_and_explicit_estimator_disable_keep_budget(self):
        function, worker, graph = self.budget("thead", mode=Mode.NONE)
        self.assertEqual(function(worker), 690)
        graph.assert_not_called()
        function, worker, graph = self.budget("thead", enabled=False)
        self.assertEqual(function(worker), 690)
        self.assertEqual(worker.cudagraph_memory_estimate, 100)

    def test_failed_capture_is_not_hidden_by_budget_fallback(self):
        function, worker, _ = self.budget("thead", graph_error=True)
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            function(worker)
        self.assertFalse(hasattr(worker, "available_kv_cache_memory_bytes"))


class TestGraphMeasurements(unittest.TestCase):
    def profiler(self, fail=False):
        memory = {"live": 0, "cached": 0}
        events = []
        wrappers = [SimpleNamespace(graph_pool="original-pool")]
        counter = SimpleNamespace(num_cudagraph_captured=7)

        def empty_cache():
            memory["cached"] = 0

        def mem_get_info():
            events.append(("read", memory["cached"]))
            return (
                128 * (1 << 30) - memory["live"] - memory["cached"],
                128 * (1 << 30),
            )

        def clear_graphs():
            memory["live"] = 0

        captures = []

        def capture(desc, **kwargs):
            self.assertEqual(wrappers[0].graph_pool, "profiling-pool")
            captures.append(desc.num_tokens)
            counter.num_cudagraph_captured += 1
            memory["live"] += (8 if len(captures) == 1 else 2) * (1 << 20)
            # Mimic PPU's reusable allocator cache, which is not live graph memory.
            memory["cached"] = 90 * (1 << 30)
            if fail:
                raise RuntimeError("capture failed")

        dispatcher = SimpleNamespace(
            get_capture_descs=lambda: [
                (Mode.FULL, [SimpleNamespace(num_tokens=n) for n in (512, 256, 128)])
            ],
            cudagraph_keys={Mode.FULL: {1}},
            keys_initialized=True,
        )
        runner = SimpleNamespace(
            vllm_config=SimpleNamespace(),
            _init_minimal_kv_cache_for_profiling=lambda: events.append("init-kv"),
            _cleanup_profiling_kv_cache=lambda: events.append("cleanup-kv"),
            cudagraph_dispatcher=dispatcher,
            _create_encoder_cudagraph_manager=lambda: None,
            _freeze_gc=nullcontext,
            device="cuda:0",
            max_model_len=32768,
            max_num_tokens=8192,
            _warmup_and_capture=capture,
            maybe_remove_all_loras=lambda config: events.append("remove-loras"),
            lora_config=None,
        )
        function = method(
            "model_runner.py",
            "ModelRunnerFL",
            "profile_cudagraph_memory",
            {
                "set_current_vllm_config": lambda config: nullcontext(),
                "compilation_counter": counter,
                "Any": object,
                "current_platform": SimpleNamespace(
                    graph_pool_handle=lambda: "profiling-pool",
                    torch_device_fn=SimpleNamespace(mem_get_info=mem_get_info),
                ),
                "GraphWrapper": SimpleNamespace(
                    _all_instances=wrappers, clear_all_graphs=clear_graphs
                ),
                "set_cudagraph_capturing_enabled": lambda value: events.append(
                    ("capture-enabled", value)
                ),
                "graph_capture": lambda device: nullcontext(),
                "_accelerator_synchronize": lambda: None,
                "_accelerator_empty_cache": empty_cache,
                "CUDAGraphMode": Mode,
                "logger": Mock(),
            },
        )
        return function, runner, wrappers, events, memory, counter

    def test_graph_estimate_excludes_ordinary_allocator_cache(self):
        function, runner, wrappers, events, memory, counter = self.profiler()
        self.assertEqual(function(runner), 12 * (1 << 20))
        readings = [
            event[1]
            for event in events
            if isinstance(event, tuple) and event[0] == "read"
        ]
        self.assertTrue(readings)
        self.assertEqual(set(readings), {0})
        self.assertEqual(wrappers[0].graph_pool, "original-pool")
        self.assertIn("cleanup-kv", events)
        self.assertEqual(memory["live"], 0)
        self.assertEqual(counter.num_cudagraph_captured, 7)
        self.assertFalse(runner.cudagraph_dispatcher.keys_initialized)

    def test_failed_measurement_restores_original_pool_and_kv(self):
        function, runner, wrappers, events, memory, counter = self.profiler(fail=True)
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            function(runner)
        self.assertEqual(wrappers[0].graph_pool, "original-pool")
        self.assertIn("cleanup-kv", events)
        self.assertIn(("capture-enabled", False), events)
        self.assertEqual(memory["live"], 0)
        self.assertEqual(counter.num_cudagraph_captured, 7)

    def test_mixed_modes_include_sampled_pool_peak_and_unsampled_graphs(self):
        function, runner, wrappers, events, memory, counter = self.profiler()
        runner.cudagraph_dispatcher.get_capture_descs = lambda: [
            (Mode.FULL, [SimpleNamespace(num_tokens=n) for n in (512, 256, 128)]),
            (
                Mode.PIECEWISE,
                [SimpleNamespace(num_tokens=n) for n in (384, 192, 96)],
            ),
        ]
        runner.cudagraph_dispatcher.cudagraph_keys = {
            Mode.FULL: {1},
            Mode.PIECEWISE: {2},
        }
        captures = []
        observed_peak = {"value": 0}

        def capture(desc, **kwargs):
            self.assertEqual(wrappers[0].graph_pool, "profiling-pool")
            captures.append(desc.num_tokens)
            counter.num_cudagraph_captured += 1
            memory["live"] += {512: 8, 256: 2, 384: 12, 192: 2}[desc.num_tokens] * (
                1 << 20
            )
            memory["cached"] = 90 * (1 << 30)
            observed_peak["value"] = max(observed_peak["value"], memory["live"])

        runner._warmup_and_capture = capture
        self.assertEqual(function(runner), 28 * (1 << 20))
        self.assertEqual(observed_peak["value"], 24 * (1 << 20))
        self.assertEqual(captures, [512, 256, 384, 192])
        self.assertEqual(wrappers[0].graph_pool, "original-pool")
        self.assertTrue(
            all(
                not keys for keys in runner.cudagraph_dispatcher.cudagraph_keys.values()
            )
        )
        self.assertIn("cleanup-kv", events)
        self.assertEqual(counter.num_cudagraph_captured, 7)

    def test_profiled_second_graph_keeps_existing_driver_overhead_floor(self):
        function, runner, wrappers, events, memory, counter = self.profiler()
        captures = []

        def capture(desc, **kwargs):
            captures.append(desc.num_tokens)
            memory["live"] += (5 if len(captures) == 1 else 0) * (1 << 20)
            memory["cached"] = 90 * (1 << 30)

        runner._warmup_and_capture = capture
        self.assertEqual(function(runner), 7 * (1 << 20))
        self.assertEqual(captures, [512, 256])
        self.assertIn("cleanup-kv", events)

    def test_pool_peak_excludes_live_memory_present_before_graph_profiling(self):
        function, runner, wrappers, events, memory, counter = self.profiler()
        memory["live"] = 100 * (1 << 20)
        self.assertEqual(function(runner), 12 * (1 << 20))
        self.assertIn("cleanup-kv", events)

    def test_encoder_graph_measurement_excludes_allocator_cache(self):
        function, runner, wrappers, events, memory, counter = self.profiler()
        runner.cudagraph_dispatcher.get_capture_descs = lambda: []

        def capture(graph_pool):
            self.assertEqual(graph_pool, "profiling-pool")
            memory["live"] = 5 * (1 << 20)
            memory["cached"] = 90 * (1 << 30)

        encoder = SimpleNamespace(
            get_num_graphs_to_capture=lambda: 2,
            token_budgets=[512, 256],
            capture=capture,
        )
        runner._create_encoder_cudagraph_manager = lambda: encoder
        self.assertEqual(function(runner), 5 * (1 << 20))
        readings = [
            event[1]
            for event in events
            if isinstance(event, tuple) and event[0] == "read"
        ]
        self.assertTrue(readings)
        self.assertEqual(set(readings), {0})
        self.assertIn("cleanup-kv", events)
        self.assertEqual(wrappers[0].graph_pool, "original-pool")


if __name__ == "__main__":
    unittest.main()
