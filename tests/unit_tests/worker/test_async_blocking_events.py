# Copyright (c) 2025 BAAI. All rights reserved.

"""Source-level checks for the platform-scoped async event policy."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

MODEL_RUNNER = Path(__file__).parents[3] / "vllm_fl" / "worker" / "model_runner.py"


def _model_runner_ast() -> ast.Module:
    return ast.parse(MODEL_RUNNER.read_text())


class AsyncReadyEventTest(TestCase):
    def test_thead_blocks_without_changing_other_platforms(self) -> None:
        tree = _model_runner_ast()
        factory = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_async_ready_event"
        )
        for device_name in ("thead", "cuda"):
            with self.subTest(device_name=device_name):
                generic_event = Mock(return_value=object())
                blocking_event = Mock(return_value=object())
                torch = SimpleNamespace(
                    Event=generic_event,
                    cuda=SimpleNamespace(Event=blocking_event),
                )
                namespace = {
                    "torch": torch,
                    "current_platform": SimpleNamespace(device_name=device_name),
                }
                module = ast.fix_missing_locations(
                    ast.Module(body=[factory], type_ignores=[])
                )
                exec(compile(module, str(MODEL_RUNNER), "exec"), namespace)
                result = namespace["_async_ready_event"]()
                if device_name == "thead":
                    self.assertIs(result, blocking_event.return_value)
                    blocking_event.assert_called_once_with(blocking=True)
                    generic_event.assert_not_called()
                else:
                    self.assertIs(result, generic_event.return_value)
                    generic_event.assert_called_once_with()
                    blocking_event.assert_not_called()

    def test_three_async_ready_events_use_factory(self) -> None:
        tree = _model_runner_ast()
        expected = {
            "AsyncGPUModelRunnerOutput": "async_copy_ready_event",
            "AsyncGPUPoolingModelRunnerOutput": "async_copy_ready_event",
            "ModelRunnerFL": "prepare_inputs_event",
        }
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            if cls.name not in expected:
                continue
            target_name = expected[cls.name]
            assignments = [
                node
                for node in ast.walk(cls)
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Attribute) and target.attr == target_name
                    for target in node.targets
                )
            ]
            self.assertEqual(len(assignments), 1, cls.name)
            call = assignments[0].value
            self.assertIsInstance(call, ast.Call)
            self.assertIsInstance(call.func, ast.Name)
            self.assertEqual(call.func.id, "_async_ready_event")
