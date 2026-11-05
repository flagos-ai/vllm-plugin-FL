# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU accounting and WorkerFL wiring tests; no device execution is exercised."""

import ast
import importlib.util
import re
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "memory_policy", ROOT / "vllm_fl/worker/memory_policy.py"
)
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


def _worker_method(namespace, name="determine_available_memory"):
    tree = ast.parse((ROOT / "vllm_fl/worker/worker.py").read_text())
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerFL"
    )
    method = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    # Exercise the actual host-side method without importing Torch or its
    # inference-mode decorator. Runtime dependencies are supplied by each test.
    method.decorator_list = []
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "worker.py", "exec"),
        namespace,
    )
    return namespace[method.name]


class BudgetTest(unittest.TestCase):
    def test_initial_occupancy_is_subtracted_once(self):
        # Independently account for weights=200, activations=100, initial
        # non-Torch=80 and later non-Torch growth=20 inside an 800-byte budget.
        self.assertEqual(POLICY.physical_kv_cache_budget(800, 320, 80), 400)

    def test_boundary_values(self):
        for requested, non_kv, initial, expected in [
            (100, 30, 0, 70),
            (100, 30, -10, 70),
            (100, 30, 70, 0),
            (100, 30, 80, 0),
            (100, 130, 10, 0),
            (0, 0, 0, 0),
        ]:
            with self.subTest(values=(requested, non_kv, initial)):
                self.assertEqual(
                    POLICY.physical_kv_cache_budget(requested, non_kv, initial),
                    expected,
                )

    def test_float_requested_budget_returns_integer_bytes(self):
        self.assertEqual(POLICY.physical_kv_cache_budget(100.75, 30, 10), 60)
        self.assertIsInstance(POLICY.physical_kv_cache_budget(100.75, 30, 10), int)

    def test_large_byte_counts(self):
        self.assertEqual(
            POLICY.physical_kv_cache_budget(64 * 2**30, 30 * 2**30, 4 * 2**30),
            30 * 2**30,
        )


class WorkerBudgetTest(unittest.TestCase):
    def setUp(self):
        self.profile = SimpleNamespace(
            non_kv_cache_memory=300,
            non_torch_increase=50,
            torch_peak_increase=100,
            before_create=SimpleNamespace(non_torch_memory=100),
            after_profile=SimpleNamespace(free_memory=200),
        )

        @contextmanager
        def profiling(*_args, **_kwargs):
            yield self.profile

        self.profiling = Mock(side_effect=profiling)
        self.policy = Mock(wraps=POLICY.physical_kv_cache_budget)
        self.platform = SimpleNamespace(
            vendor_name="metax",
            device_type="cuda",
            empty_cache=Mock(),
            torch_device_fn=SimpleNamespace(
                reset_peak_memory_stats=Mock(),
                mem_get_info=Mock(return_value=(600, 1000)),
            ),
        )
        self.worker = SimpleNamespace(
            device="test-device",
            requested_memory=800,
            init_snapshot=SimpleNamespace(free_memory=900),
            cache_config=SimpleNamespace(
                kv_cache_memory_bytes=None,
                gpu_memory_utilization=0.9,
            ),
            model_runner=SimpleNamespace(model_memory_usage=200, profile_run=Mock()),
        )
        self.method = _worker_method(
            {
                "current_platform": self.platform,
                "memory_profiling_fl": self.profiling,
                "physical_kv_cache_budget": self.policy,
                "logger": Mock(),
                "GiB_bytes": 2**30,
                "gc": SimpleNamespace(collect=Mock()),
            }
        )

    def test_metax_auto_budget_uses_initial_snapshot(self):
        self.assertEqual(self.method(self.worker), 400)
        self.policy.assert_called_once_with(800, 300, 100)
        self.profiling.assert_called_once_with(
            self.worker.init_snapshot,
            weights_memory=200,
        )
        self.worker.model_runner.profile_run.assert_called_once_with()
        self.assertEqual(self.worker.non_torch_memory, 50)
        self.assertEqual(self.worker.peak_activation_memory, 100)

    def test_zero_or_negative_initial_occupancy_does_not_raise_budget(self):
        for initial in [0, -20]:
            with self.subTest(initial=initial):
                self.profile.before_create.non_torch_memory = initial
                self.assertEqual(self.method(self.worker), 500)

    def test_exhausted_budget_is_zero(self):
        self.profile.before_create.non_torch_memory = 600
        self.assertEqual(self.method(self.worker), 0)

    def test_other_vendors_keep_original_budget(self):
        for vendor in ["nvidia", "musa", "hygon", None]:
            with self.subTest(vendor=vendor):
                self.platform.vendor_name = vendor
                self.assertEqual(self.method(self.worker), 500)
        del self.platform.vendor_name
        self.assertEqual(self.method(self.worker), 500)
        self.policy.assert_not_called()

    def test_explicit_override_bypasses_automatic_budget(self):
        self.worker.cache_config.kv_cache_memory_bytes = 1200
        self.assertEqual(self.method(self.worker), 1200)
        self.worker.model_runner.profile_run.assert_called_once_with()
        self.profiling.assert_not_called()
        self.policy.assert_not_called()
        self.platform.empty_cache.assert_not_called()

    def test_profile_error_propagates_without_retry(self):
        self.worker.model_runner.profile_run.side_effect = RuntimeError(
            "profile failed"
        )
        with self.assertRaisesRegex(RuntimeError, "profile failed"):
            self.method(self.worker)
        self.worker.model_runner.profile_run.assert_called_once_with()
        self.profiling.assert_called_once()
        self.policy.assert_not_called()

    def test_npu_estimation_path_is_unchanged(self):
        self.platform.device_type = "npu"
        self.platform.vendor_name = "ascend"
        self.assertEqual(self.method(self.worker), 250)
        self.profiling.assert_not_called()
        self.policy.assert_not_called()
        self.worker.model_runner.profile_run.assert_not_called()


class WarmupBudgetHintTest(unittest.TestCase):
    def setUp(self):
        self.mib = 1 << 20
        self.events = []
        self.policy = Mock(wraps=POLICY.physical_kv_cache_budget)
        self.platform = SimpleNamespace(vendor_name="metax", device_type="cuda")
        self.logger = Mock()
        self.logger.debug.side_effect = lambda _message: self.events.append("hint")
        compilation = SimpleNamespace(
            mode="compile",
            compile_sizes=[4, 8],
            cudagraph_mode="full",
            cudagraph_capture_sizes=[8],
            get_compile_ranges=lambda: [],
        )

        def capture():
            self.events.append("capture")
            return 30 * self.mib

        runner = SimpleNamespace(
            model_memory_usage=200 * self.mib,
            lora_config=None,
            _dummy_run=Mock(
                side_effect=lambda *_a, **_kw: self.events.append("compile")
            ),
            maybe_remove_all_loras=Mock(),
            capture_model=Mock(side_effect=capture),
        )
        # Post-profiling state: initial non-Torch=100MiB, growth=50MiB,
        # weights=200MiB, activations=100MiB. Initial free memory already
        # excludes the 100MiB initial occupancy. Auto KV budget is 450MiB.
        self.worker = SimpleNamespace(
            vllm_config=SimpleNamespace(compilation_config=compilation),
            compilation_config=SimpleNamespace(
                compilation_time=0,
                encoder_compilation_time=0,
            ),
            cache_config=SimpleNamespace(
                kv_cache_memory_bytes=None,
                gpu_memory_utilization=0.9,
            ),
            init_snapshot=SimpleNamespace(
                total_memory=1000 * self.mib,
                free_memory=900 * self.mib,
                non_torch_memory=100 * self.mib,
            ),
            requested_memory=900 * self.mib,
            peak_activation_memory=100 * self.mib,
            non_torch_memory=50 * self.mib,
            available_kv_cache_memory_bytes=450 * self.mib,
            model_runner=runner,
            model_config=SimpleNamespace(seed=42),
        )
        self.kernel_warmup = Mock(
            side_effect=lambda _worker: self.events.append("kernel")
        )
        self.method = _worker_method(
            {
                "current_platform": self.platform,
                "physical_kv_cache_budget": self.policy,
                "CompilationMode": SimpleNamespace(VLLM_COMPILE="compile"),
                "CUDAGraphMode": SimpleNamespace(NONE="none"),
                "CompilationTimes": SimpleNamespace,
                "kernel_warmup": self.kernel_warmup,
                "get_pp_group": lambda: SimpleNamespace(is_last_rank=False),
                "set_random_seed": Mock(),
                "logger": self.logger,
                "GiB_bytes": 1 << 30,
            },
            name="compile_or_warm_up_model",
        )

    def _hints_after_warmup(self):
        self.method(self.worker)
        message = self.logger.debug.call_args.args[0]
        return [int(value) for value in re.findall(r"--kv-cache-memory=(\d+)", message)]

    def test_metax_requested_hint_after_warmup_counts_initial_once(self):
        # Requested: 900-200-100-50-30(graph)-150(buffer)-100(initial)=270.
        # Free-based: 900-200-100-50-30-150=370, without another initial debit.
        self.assertEqual(self._hints_after_warmup(), [270 * self.mib, 370 * self.mib])
        self.assertEqual(self.events, ["compile", "kernel", "capture", "hint"])
        self.policy.assert_called_once_with(
            900 * self.mib,
            530 * self.mib,
            100 * self.mib,
        )
        self.assertEqual(self.worker.non_torch_memory, 50 * self.mib)
        self.assertEqual(self.worker.available_kv_cache_memory_bytes, 450 * self.mib)

    def test_other_vendors_keep_both_original_hints(self):
        for vendor in ["nvidia", "musa", None]:
            with self.subTest(vendor=vendor):
                self.platform.vendor_name = vendor
                self.assertEqual(
                    self._hints_after_warmup(),
                    [370 * self.mib, 370 * self.mib],
                )
        del self.platform.vendor_name
        self.assertEqual(self._hints_after_warmup(), [370 * self.mib, 370 * self.mib])
        self.policy.assert_not_called()

    def test_explicit_override_emits_no_automatic_hints(self):
        self.worker.cache_config.kv_cache_memory_bytes = 1200 * self.mib
        self.method(self.worker)
        self.logger.debug.assert_not_called()
        self.policy.assert_not_called()
        self.worker.model_runner.capture_model.assert_called_once_with()

    def test_requested_hint_without_graph_capture(self):
        self.worker.vllm_config.compilation_config.cudagraph_mode = "none"
        self.assertEqual(self._hints_after_warmup(), [300 * self.mib, 400 * self.mib])
        self.worker.model_runner.capture_model.assert_not_called()
        self.kernel_warmup.assert_called_once_with(self.worker)
        self.assertEqual(self.worker.non_torch_memory, 50 * self.mib)


if __name__ == "__main__":
    unittest.main()
