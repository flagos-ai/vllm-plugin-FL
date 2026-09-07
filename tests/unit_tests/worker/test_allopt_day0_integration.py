# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU-only checks for the all-on/day0 merge boundaries, without vLLM imports."""

from __future__ import annotations

import ast
import os
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "vllm_fl/worker/model_runner.py"
POLICY = ROOT / "vllm_fl/patches/qwen3_8_flash_next.py"


class GraphMode(Enum):
    NONE = 0
    FULL = 1


def extract_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class AlloptDay0IntegrationTests(unittest.TestCase):
    def test_metadata_routes_to_matching_arena_only(self):
        common_compute, packed_compute = object(), object()
        run = extract_function(
            RUNNER,
            "_run_common_attention_metadata",
            {
                "CUDAGraphMode": GraphMode,
                "compute_common_attention_metadata": common_compute,
                "compute_common_slot_mapping": packed_compute,
            },
        )
        for arena_kind in ("none", "matching", "stale"):
            for mode in GraphMode:
                for ubatching in (False, True):
                    with self.subTest(arena=arena_kind, mode=mode, ubatching=ubatching):
                        table = object()
                        arena = (
                            None
                            if arena_kind == "none"
                            else SimpleNamespace(
                                block_table=table
                                if arena_kind == "matching"
                                else object()
                            )
                        )
                        common, packed = Mock(), Mock()
                        runner = SimpleNamespace(
                            input_batch=SimpleNamespace(block_table=table),
                            parallel_config=SimpleNamespace(use_ubatching=ubatching),
                            packed_block_table_arena=arena,
                            common_attention_metadata_graph=common,
                            common_slot_mapping_graph=packed,
                            query_start_loc=SimpleNamespace(gpu=[0, 1, 2]),
                            positions=[0, 0],
                            seq_lens=[1, 1],
                            num_computed_tokens=[0, 0],
                        )
                        selected = packed if arena_kind == "matching" else common
                        other = common if arena_kind == "matching" else packed
                        self.assertIs(
                            run(runner, 2, mode, capture=True),
                            selected.run.return_value,
                        )
                        selected.run.assert_called_once()
                        other.run.assert_not_called()
                        args, kwargs = selected.run.call_args
                        self.assertIs(args[0], table)
                        self.assertEqual(args[1], 2)
                        self.assertIs(
                            kwargs["compute"],
                            packed_compute
                            if arena_kind == "matching"
                            else common_compute,
                        )
                        self.assertEqual(
                            kwargs["use_graph"],
                            mode != GraphMode.NONE and not ubatching,
                        )
                        self.assertTrue(kwargs["capture"])

    def test_packed_arena_opt_in_and_required_gate(self):
        arena_factory, logger = Mock(), Mock()
        install = extract_function(
            RUNNER,
            "_install_packed_block_table_arena",
            {
                "os": os,
                "PackedBlockTableArena": arena_factory,
                "logger": logger,
            },
        )
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(block_table=object()),
            device="cuda",
            pin_memory=True,
        )
        with patch.dict(os.environ, {}, clear=True):
            install(runner)
            self.assertIsNone(runner.packed_block_table_arena)
            arena_factory.assert_not_called()
            os.environ["VLLM_FL_PACKED_BLOCK_TABLE_REQUIRE"] = "1"
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                install(runner)
            os.environ["VLLM_FL_PACKED_BLOCK_TABLE_ARENA"] = "1"
            install(runner)
            self.assertIs(runner.packed_block_table_arena, arena_factory.return_value)

    def test_plan_cache_request_retains_generic_flaggems(self):
        skip = extract_function(
            POLICY,
            "should_skip_generic_flaggems_aten",
            {
                "os": os,
                "needs_native_index_select": bool,
            },
        )
        for plugin_value in (None, "0", "1", "true"):
            for gems_value in ("0", "1"):
                with self.subTest(plugin=plugin_value, gems=gems_value):
                    values = {"FLAGGEMS_ATEN_PLAN_CACHE": gems_value}
                    if plugin_value is not None:
                        values["VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE"] = plugin_value
                    with patch.dict(os.environ, values, clear=True):
                        requested = (
                            plugin_value if plugin_value is not None else gems_value
                        ) != "0"
                        self.assertEqual(
                            skip(True, vendor_name="nvidia", whitelist=None),
                            not requested,
                        )
                        self.assertFalse(
                            skip(False, vendor_name="nvidia", whitelist=None)
                        )
                        self.assertFalse(skip(True, vendor_name="amd", whitelist=None))
                        self.assertFalse(
                            skip(True, vendor_name="nvidia", whitelist=["sigmoid"])
                        )

    def test_packed_path_retains_padded_row_cleanup(self):
        tree = ast.parse(RUNNER.read_text())
        values = [
            keyword.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "block_table_rows_are_current"
        ]
        self.assertEqual(len(values), 2)
        for value in values:
            self.assertEqual(
                ast.unparse(value), "self.packed_block_table_arena is None"
            )

    def test_qsa_workspace_warmup_keeps_attention_enabled(self):
        source = RUNNER.read_text()
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_warmup_and_capture"
        )
        text = ast.get_source_segment(source, method)
        self.assertIn(
            "force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL", text
        )
        self.assertIn("force_attention=force_attention", text)
        self.assertIn("cudagraph_runtime_mode=CUDAGraphMode.NONE", text)

    def test_vllm024_multimodal_pruning_compatibility_is_retained(self):
        model_path = ROOT / "vllm_fl/models/qwen3_8_flash_next/gpu/model.py"
        tree = ast.parse(model_path.read_text())
        model_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "Qwen3_8FlashNextForConditionalGeneration"
        )
        calls = [
            node.func.attr
            for node in ast.walk(model_class)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertNotIn("_init_video_pruning", calls)
        assignments = {
            target.attr: node.value.value
            for node in ast.walk(model_class)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            for target in node.targets
            if isinstance(target, ast.Attribute)
        }
        self.assertFalse(assignments["is_multimodal_pruning_enabled"])
        self.assertIsNone(assignments["video_pruning_method"])
        self.assertEqual(assignments["video_pruning_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
