# Copyright (c) 2026 BAAI. All rights reserved.
"""Worker policy behavior; GPU producer/capture coverage lives in functional tests."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm.config import CUDAGraphMode

from vllm_fl.patches.qwen3_8_flash_next import should_skip_generic_flaggems_aten
from vllm_fl.worker import model_runner as runner_module


class AlloptDay0IntegrationTests(unittest.TestCase):
    def test_packed_storage_uses_the_common_metadata_owner(self):
        table = object()
        owner = Mock()
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(block_table=table),
            parallel_config=SimpleNamespace(use_ubatching=False),
            common_metadata_policy=SimpleNamespace(mode="graph"),
            common_attention_metadata_graph=owner,
            packed_block_table_arena=SimpleNamespace(block_table=table),
            query_start_loc=SimpleNamespace(gpu=[0, 1, 2]),
            positions=[0, 0],
            seq_lens=[1, 1],
            num_computed_tokens=[0, 0],
        )
        result = runner_module.ModelRunnerFL._run_common_attention_metadata(
            runner, 2, CUDAGraphMode.FULL, capture=True
        )
        self.assertIs(result, owner.run.return_value)
        args, kwargs = owner.run.call_args
        self.assertIs(args[0], table)
        self.assertIs(
            kwargs["compute"], runner_module.compute_common_attention_metadata
        )
        self.assertTrue(kwargs["use_graph"])
        runner.common_attention_metadata_graph = None
        self.assertIsNone(
            runner_module.ModelRunnerFL._run_common_attention_metadata(
                runner, 2, CUDAGraphMode.FULL
            )
        )

    def test_packed_arena_opt_in_and_required_gate(self):
        arena_factory = Mock()
        install = runner_module.ModelRunnerFL._install_packed_block_table_arena
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(block_table=object()),
            device="cuda",
            pin_memory=True,
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(runner_module, "PackedBlockTableArena", arena_factory),
        ):
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
        skip = should_skip_generic_flaggems_aten

        def config(model_type):
            return SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(model_type=model_type)
                )
            )

        qwen = config("qwen3_8_flash_next_text")
        other = config("llama")
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
                            skip(qwen, vendor_name="nvidia", whitelist=None),
                            not requested,
                        )
                        self.assertFalse(
                            skip(other, vendor_name="nvidia", whitelist=None)
                        )
                        self.assertFalse(skip(qwen, vendor_name="amd", whitelist=None))
                        self.assertFalse(
                            skip(qwen, vendor_name="nvidia", whitelist=["sigmoid"])
                        )
