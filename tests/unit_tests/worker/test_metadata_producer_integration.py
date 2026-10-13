# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-isolated behavior checks for the merged graph lifecycle.

Execute the real wrapper and runner methods without importing device runtimes.
The callback/graph doubles test producer selection and lifecycle, not kernels.
"""

import ast
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).parents[3]


def _load_definition(path, name, namespace, *, owner=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if owner is not None:
        nodes = next(
            node
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == owner
        ).body
    node = next(
        node
        for node in nodes
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace
    )
    return namespace[name]


class Mode(Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2

    def is_valid_runtime_mode(self):
        return True


class MetadataProducerIntegrationTest(TestCase):
    def setUp(self):
        self.events = []
        self.capturing = None
        self.input = [1]
        self.output = [-99]
        self.generic_compute = Mock()
        self.dispatch_calls = 0
        self.graphs = []

        @contextmanager
        def capture(graph, **kwargs):
            self.capturing = graph
            try:
                yield
            finally:
                self.capturing = None

        def make_graph():
            graph = SimpleNamespace(action=None)

            def replay():
                self.events.append("replay")
                graph.action()

            graph.replay = replay
            self.graphs.append(graph)
            return graph

        def compute(*args):
            self.dispatch_calls += 1

            def write():
                self.output[:] = self.input

            if self.capturing is None:
                write()
            else:
                self.capturing.action = write
                self.output[:] = [-99]

        self.dispatch_compute = compute
        self.platform = SimpleNamespace(
            device_name="thead",
            device_type="cuda",
            torch_device_fn=SimpleNamespace(
                graph=capture,
                current_stream=lambda: SimpleNamespace(
                    synchronize=lambda: self.events.append("sync")
                ),
            ),
            get_global_graph_pool=lambda: None,
        )
        namespace = {
            "current_platform": self.platform,
            "supports_accelerator_graph": lambda: True,
            "Graph": SimpleNamespace(graph=make_graph),
            "logger": Mock(),
            "compute_common_attention_metadata": self.generic_compute,
        }
        base = _load_definition(
            "vllm_fl/worker/common_attention_metadata.py",
            "CommonAttentionMetadataGraphRunner",
            namespace,
        )
        self.slot_class = _load_definition(
            "vllm_fl/worker/common_slot_mapping.py",
            "CommonSlotMappingGraphRunner",
            {
                "CommonAttentionMetadataGraphRunner": base,
                "current_platform": self.platform,
                "compute_common_slot_mapping": compute,
            },
        )
        self.base_class = base
        self.table = object()
        self.args = (self.table, 1, self.input, self.input, self.input, self.output)

    def test_capture_and_changed_input_replays_keep_dispatch_and_ordering(self):
        runner = self.slot_class()
        self.assertTrue(runner.run(*self.args, use_graph=True, capture=True))
        self.assertEqual(self.dispatch_calls, 2)  # warmup then capture
        self.assertEqual(self.output, [1])  # initialized by first replay
        for capture, value in ((True, 7), (False, 11)):
            self.input[:] = [value]
            self.output[:] = [-99]
            self.assertTrue(runner.run(*self.args, use_graph=True, capture=capture))
            self.assertEqual(self.output, [value])
        self.assertEqual(self.dispatch_calls, 2)
        self.assertEqual(self.events, ["replay", "sync"] * 3)
        self.generic_compute.assert_not_called()
        runner.clear()
        self.assertFalse(runner.graphs)
        self.assertFalse(runner.run(*self.args, use_graph=True, capture=False))
        self.assertEqual(self.dispatch_calls, 3)

    def test_eager_and_unavailable_graph_use_dispatch_without_replay_sync(self):
        for graph_available, use_graph in ((True, False), (False, True)):
            with self.subTest(graph_available=graph_available, use_graph=use_graph):
                runner = self.slot_class()
                runner._graph_capture_supported = graph_available
                self.assertFalse(
                    runner.run(*self.args, use_graph=use_graph, capture=True)
                )
                self.assertEqual(self.output, self.input)
                self.assertFalse(runner.graphs)
        self.assertEqual(self.events, [])
        self.generic_compute.assert_not_called()

    def test_other_platform_retains_asynchronous_replay(self):
        self.platform.device_name = "cuda"
        runner = self.slot_class()
        self.assertTrue(runner.run(*self.args, use_graph=True, capture=True))
        self.assertEqual(self.events, ["replay"])

    def test_generic_runner_retains_its_original_producer(self):
        runner = self.base_class()
        self.assertFalse(runner.run(*self.args, use_graph=False, capture=False))
        self.generic_compute.assert_called_once_with(*self.args)
        self.assertEqual(self.dispatch_calls, 0)
        self.assertEqual(self.events, [])

    def test_model_runner_preserves_selected_default_and_piecewise_extent(self):
        run = _load_definition(
            "vllm_fl/worker/model_runner.py",
            "_run_common_attention_metadata",
            {
                "CUDAGraphMode": Mode,
                "compute_common_attention_metadata": self.generic_compute,
            },
            owner="ModelRunnerFL",
        )
        for graph_class in (self.slot_class, self.base_class):
            for mode in Mode:
                for ubatching in (False, True):
                    with self.subTest(
                        graph_class=graph_class.__name__, mode=mode, ubatching=ubatching
                    ):
                        graph = graph_class()
                        runner = SimpleNamespace(
                            common_attention_metadata_graph=graph,
                            parallel_config=SimpleNamespace(use_ubatching=ubatching),
                            max_num_reqs=4,
                            input_batch=SimpleNamespace(block_table=self.table),
                            query_start_loc=SimpleNamespace(gpu=[0, 1, 1, 1, 1]),
                            positions=[0] * 8,
                            seq_lens=[1, 0, 0, 0],
                            num_computed_tokens=[0] * 4,
                        )
                        # Exercise real eager/missing-key handling and check
                        # which producer actually ran, not the call spelling.
                        call = Mock(wraps=graph.run)
                        graph.run = call
                        self.generic_compute.reset_mock()
                        dispatch_before = self.dispatch_calls
                        run(runner, 1, mode)
                        args, kwargs = call.call_args
                        if graph_class is self.slot_class:
                            self.assertEqual(self.dispatch_calls, dispatch_before + 1)
                            self.generic_compute.assert_not_called()
                            self.assertEqual(self.output, self.input)
                        else:
                            self.generic_compute.assert_called_once()
                            self.assertEqual(self.dispatch_calls, dispatch_before)
                        self.assertEqual(args[1], 4 if mode == Mode.PIECEWISE else 1)
                        self.assertEqual(
                            kwargs["use_graph"], mode != Mode.NONE and not ubatching
                        )
        inactive = SimpleNamespace(common_attention_metadata_graph=None)
        self.assertFalse(run(inactive, 1, Mode.FULL))

    def test_dummy_capture_preserves_qwen_slots_and_clears_generic_slots(self):
        class HostTensor(np.ndarray):
            def fill_(self, value):
                self.fill(value)
                return self

            def copy_(self, value, **kwargs):
                self[:] = value
                return self

        class ModelBoundary(Exception):
            pass

        dummy = _load_definition(
            "vllm_fl/worker/model_runner.py",
            "_dummy_run",
            {
                "CUDAGraphMode": Mode,
                "np": np,
                "managed_inference_mode": lambda: lambda fn: fn,
                "maybe_create_ubatch_slices": lambda *args: (None, None),
                "logger": Mock(),
            },
            owner="ModelRunnerFL",
        )
        for ngram in (False, True):
            for mode in (Mode.FULL, Mode.PIECEWISE):
                with self.subTest(ngram=ngram, mode=mode):
                    slots = np.full(2, -99, dtype=np.int64).view(HostTensor)
                    query_start = np.full(5, -99, dtype=np.int32)
                    seq_lens = np.full(4, -99, dtype=np.int32).view(HostTensor)
                    graph = self.slot_class() if ngram else self.base_class()
                    runner = SimpleNamespace(
                        vllm_config=SimpleNamespace(
                            model_config=SimpleNamespace(multimodal_config=None),
                            parallel_config=SimpleNamespace(num_ubatches=1),
                        ),
                        scheduler_config=SimpleNamespace(max_num_seqs=4),
                        uniform_decode_query_len=1,
                        max_num_tokens=16,
                        lora_config=None,
                        speculative_config=None,
                        uses_ngram_embedding=ngram,
                        common_attention_metadata_graph=graph,
                        _determine_batch_execution_and_padding=Mock(
                            return_value=(
                                mode,
                                SimpleNamespace(num_tokens=2, num_reqs=2),
                                False,
                                None,
                                None,
                            )
                        ),
                        _get_slot_mappings=Mock(return_value=({0: slots}, None)),
                        synchronize_input_prep=nullcontext,
                        optimistic_seq_lens_cpu=seq_lens.copy(),
                        seq_lens=seq_lens,
                        query_pos=SimpleNamespace(np=np.zeros(16, dtype=np.int64)),
                        _get_cumsum_and_arange=lambda tokens, out: np.cumsum(tokens),
                        query_start_loc=SimpleNamespace(
                            np=query_start, copy_to_gpu=Mock()
                        ),
                        input_batch=SimpleNamespace(block_table=Mock()),
                        _run_common_attention_metadata=Mock(
                            side_effect=lambda *args, _slots=slots, **kwargs: (
                                _slots.fill(42)
                            )
                        ),
                        _build_attention_metadata=Mock(return_value=({}, None)),
                        maybe_dummy_run_with_lora=Mock(side_effect=ModelBoundary),
                    )
                    with self.assertRaises(ModelBoundary):
                        dummy(
                            runner,
                            2,
                            cudagraph_runtime_mode=mode,
                            is_graph_capturing=True,
                        )
                    np.testing.assert_array_equal(query_start, [0, 1, 2, 2, 2])
                    np.testing.assert_array_equal(seq_lens, [2, 2, 0, 0])
                    np.testing.assert_array_equal(
                        slots, [42, 42] if ngram else [-1, -1]
                    )
                    runner._run_common_attention_metadata.assert_called_once_with(
                        2, mode, capture=True
                    )
                    self.assertEqual(
                        runner._build_attention_metadata.call_count, mode == Mode.FULL
                    )
