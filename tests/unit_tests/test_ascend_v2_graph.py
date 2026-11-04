# SPDX-License-Identifier: Apache-2.0
"""CPU execution-contract checks for the NPU-specific V2 graph manager."""

import ast
import logging
import unittest
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch


class _Mode(Enum):
    NONE = 0
    FULL = 1
    FULL_DECODE_ONLY = 2
    PIECEWISE = 3

    def has_piecewise_cudagraphs(self):
        return self == self.PIECEWISE


@dataclass(frozen=True)
class _Descriptor:
    num_tokens: int
    num_reqs: int = 1
    cg_mode: _Mode = _Mode.FULL
    num_active_loras: int = 0
    num_ubatches: int = 1
    max_query_len: int | None = None
    uniform_token_count: int = 1


@dataclass(frozen=True)
class _BatchDescriptor:
    num_tokens: int
    has_lora: bool = False
    num_active_loras: int = 0


class _UpstreamManager:
    def __init__(self, config, device, mode, *args, **kwargs):
        self.vllm_config = config
        self.device = device
        self.cudagraph_mode = mode
        self.max_num_reqs = 4
        self.is_first_pp_rank = self.is_last_pp_rank = True
        self.dp_size = 1
        self.use_breakable_cg = False
        self.graphs = {}
        self.pool = (7, 8)
        self._capture_descs = {_Mode.FULL: config.descriptors}
        self._max_full_descs_to_capture = None
        self._capture_mem_samples = None
        self._graphs_captured = False
        self.hidden_states = None
        self.aux_hidden_states = []
        self.intermediate_tensors = None

    def run_fullgraph(self, descriptor):
        self.graphs[descriptor].replay()
        return self.hidden_states[: descriptor.num_tokens]

    def release_graphs(self):
        self.graphs.clear()
        self._graphs_captured = False


class _NpuSpy:
    def __init__(self, events):
        self.events = events
        self.current = self.Stream()
        self.graph_contexts = []

    class Stream:
        def __init__(self, device=None):
            self.waited = []

        def wait_stream(self, stream):
            self.waited.append(stream)

    def NPUGraph(self):
        events = self.events

        class Graph:
            def replay(self):
                events.append("replay")

        return Graph()

    def current_stream(self, device=None):
        return self.current

    @contextmanager
    def stream(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous

    @contextmanager
    def graph(self, graph, pool=None, stream=None):
        self.graph_contexts.append((graph, pool, stream))
        yield

    def synchronize(self, device=None):
        self.events.append("synchronize")


class _Runtime:
    def __init__(self, events):
        self.events = events
        self.capture_keys = []
        self.updates = []
        self.cleared = False

    def bind_attention_groups(self, groups):
        self.groups = groups

    @contextmanager
    def capture_scope(self, context):
        self.capture_keys.append(context.batch_descriptor)
        yield

    @contextmanager
    def replay_scope(self, context):
        self.events.append("synchronize")
        yield
        self.updates.append(context)
        self.events.append("task-update")

    def clear_decoder_graphs(self):
        self.cleared = True


class AscendV2GraphTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.npu = _NpuSpy(self.events)
        self.context = None
        self.metadata_serial = 0
        self.prepared_metadata = []
        self.forward_observations = []
        npu_patch = patch.object(torch, "npu", self.npu, create=True)
        npu_patch.start()
        self.addCleanup(npu_patch.stop)

        @contextmanager
        def set_forward_context(metadata, config, **kwargs):
            previous = self.context
            self.context = SimpleNamespace(attn_metadata=metadata, **kwargs)
            try:
                # Upstream set_forward_context deliberately yields None.
                yield
            finally:
                self.context = previous

        def prepare_inputs(*args, **kwargs):
            self.metadata_serial += 1
            metadata = {"serial": self.metadata_serial}
            self.prepared_metadata.append(metadata)
            return metadata, {}

        source = (
            Path(__file__).resolve().parents[2] / "vllm_fl/worker/ascend_v2_graph.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ]
        self.counter = SimpleNamespace(num_cudagraph_captured=0)
        self.namespace = {
            "__name__": "ascend_v2_graph_under_test",
            "torch": torch,
            "nn": SimpleNamespace(Module=object),
            "Callable": Callable,
            "Any": Any,
            "SimpleNamespace": SimpleNamespace,
            "wraps": wraps,
            "VllmConfig": object,
            "CUDAGraphMode": _Mode,
            "BatchDescriptor": _BatchDescriptor,
            "BatchExecutionDescriptor": _Descriptor,
            "CreateForwardFn": object,
            "InputBuffers": object,
            "InputBatch": object,
            "ModelState": object,
            "BlockTables": object,
            "KVCacheConfig": object,
            "AttentionGroup": object,
            "IntermediateTensors": object,
            "ModelCudaGraphManager": _UpstreamManager,
            "GraphRuntimeController": lambda **kwargs: _Runtime(self.events),
            "set_current_vllm_config": lambda config: nullcontext(),
            "set_forward_context": set_forward_context,
            "get_forward_context": lambda: self.context,
            "capture_pool": lambda pool, config: nullcontext(pool),
            "prepare_inputs_to_capture": prepare_inputs,
            "has_compiled_submodule": lambda model: False,
            "is_global_first_rank": lambda: False,
            "get_offloader": lambda: SimpleNamespace(
                sync_prev_onload=lambda: None,
                join_after_forward=lambda: None,
            ),
            "compilation_counter": self.counter,
            "logger": logging.getLogger(__name__),
        }
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"),
            self.namespace,
        )
        self.manager_type = self.namespace["AscendModelGraphManager"]
        self.bridge = self.namespace["install_graph_metadata_bridge"]

    def manager(self, descriptors=None, mode=_Mode.FULL_DECODE_ONLY):
        config = SimpleNamespace(
            speculative_config=None,
            parallel_config=SimpleNamespace(use_ubatching=False),
            descriptors=descriptors or [_Descriptor(4, 4), _Descriptor(2, 2)],
        )
        return self.manager_type(config, SimpleNamespace(type="npu"), mode)

    def capture(self, manager, *, fail=False, has_lora=False):
        test = self

        class Model:
            def __call__(self, **inputs):
                context = test.context
                test.forward_observations.append(
                    (context.capturing, context.batch_descriptor, context.attn_metadata)
                )
                if fail and context.capturing:
                    raise RuntimeError("capture failed")
                return torch.full(
                    (inputs["input_ids"].shape[0], 3),
                    float(context.attn_metadata["serial"]),
                )

        model_state = SimpleNamespace(prepare_dummy_inputs=lambda *args: {})
        buffers = SimpleNamespace(
            input_ids=torch.zeros(4, dtype=torch.int64),
            positions=torch.zeros(4, dtype=torch.int64),
            is_padding=torch.zeros(4, dtype=torch.bool),
        )
        manager.capture(
            Model(), model_state, buffers, None, None, [], None, has_lora=has_lora
        )

    def test_each_size_has_fresh_metadata_and_native_capture_context(self):
        manager = self.manager()
        original_stream = self.npu.current
        with patch.object(
            torch.cuda, "graph", side_effect=AssertionError("CUDA graph used")
        ):
            self.capture(manager)
        self.assertEqual(
            [item[0] for item in self.forward_observations], [False, True, False, True]
        )
        self.assertEqual(len(manager.graphs), 2)
        self.assertEqual(manager.capture_count, 2)
        self.assertEqual(self.counter.num_cudagraph_captured, 2)
        self.assertEqual(len(self.npu.graph_contexts), 2)
        self.assertIs(self.npu.current, original_stream)
        self.assertIs(original_stream.waited[-1], manager._npu_capture_stream)
        self.assertEqual(
            manager.runtime.capture_keys, [_BatchDescriptor(4), _BatchDescriptor(2)]
        )
        for index, descriptor in enumerate(manager.graphs):
            self.assertIs(
                manager._capture_metadata[descriptor],
                self.prepared_metadata[index * 2 + 1],
            )
        self.assertTrue(manager._graphs_captured)
        self.assertFalse(manager._capturing)

    def test_persistent_outputs_keep_storage_after_capture(self):
        manager = self.manager()
        self.capture(manager)
        self.assertEqual(manager.hidden_states.shape, (4, 3))
        pointer = manager.hidden_states.data_ptr()
        for descriptor in manager.graphs:
            manager.stage_replay_metadata({"actual_lengths": [5]})
            output = manager.run_fullgraph(descriptor)
            self.assertEqual(output.data_ptr(), pointer)
            self.assertEqual(output.shape[0], descriptor.num_tokens)
        self.assertEqual(manager.replay_count, 2)
        stats = manager.graph_statistics()
        self.assertEqual(stats["capture_count"], 2)
        self.assertEqual(stats["replay_count"], 2)
        self.assertEqual([graph["num_tokens"] for graph in stats["graphs"]], [4, 2])
        self.assertTrue(all(graph["replay_count"] == 1 for graph in stats["graphs"]))
        self.assertTrue(
            all(graph["graph_type"].endswith(".Graph") for graph in stats["graphs"])
        )

    def test_replay_requires_fresh_metadata_and_updates_after_enqueue(self):
        manager = self.manager()
        self.capture(manager)
        descriptor = next(iter(manager.graphs))
        metadata = {"actual_lengths": [17, 29]}
        manager.stage_replay_metadata(metadata)
        self.events.clear()
        manager.run_fullgraph(descriptor)
        self.assertEqual(self.events, ["synchronize", "replay", "task-update"])
        self.assertIs(manager.runtime.updates[-1].attn_metadata, metadata)
        self.assertIs(manager._last_replay_metadata, metadata)
        self.assertIsNone(manager._pending_metadata)
        with self.assertRaisesRegex(RuntimeError, "this step"):
            manager.run_fullgraph(descriptor)

    def test_instance_bridge_preserves_eager_and_tracks_replaced_manager(self):
        class ModelState:
            def prepare_attn(self, input_batch, cudagraph_mode):
                return {"mode": cudagraph_mode}

        first, second = ModelState(), ModelState()
        generic_method = ModelState.prepare_attn
        active = [self.manager()]
        self.bridge(first, lambda: active[0])
        wrapper = first.prepare_attn
        self.bridge(first, lambda: None)
        self.assertIs(first.prepare_attn, wrapper)
        eager = first.prepare_attn(None, _Mode.NONE)
        self.assertEqual(eager, {"mode": _Mode.NONE})
        self.assertIsNone(active[0]._pending_metadata)
        batch = SimpleNamespace(
            decode_graph_eligible=True,
            num_reqs=3,
            num_reqs_after_padding=4,
            num_tokens=3,
            num_tokens_after_padding=4,
            num_scheduled_tokens=[1, 1, 1],
        )
        original_batch = vars(batch).copy()
        full = first.prepare_attn(batch, cudagraph_mode=_Mode.FULL)
        self.assertIs(active[0]._pending_metadata, full)
        active[0] = self.manager()
        latest = first.prepare_attn(batch, _Mode.FULL)
        self.assertIs(active[0]._pending_metadata, latest)
        self.assertEqual(vars(batch), original_batch)
        self.assertIs(ModelState.prepare_attn, generic_method)
        self.assertIs(second.prepare_attn.__func__, generic_method)

    def test_full_bridge_rejects_real_prefill_before_building_metadata(self):
        calls = []
        state = SimpleNamespace(prepare_attn=lambda *args, **kwargs: calls.append(args))
        manager = self.manager()
        self.bridge(state, lambda: manager)
        for eligible, scheduled in ((False, [1]), (False, [4]), (True, [2, 1])):
            batch = SimpleNamespace(
                decode_graph_eligible=eligible,
                num_reqs=len(scheduled),
                num_scheduled_tokens=scheduled,
            )
            with self.assertRaisesRegex(ValueError, "prefill|one query"):
                state.prepare_attn(batch, _Mode.FULL)
        self.assertEqual(calls, [])
        self.assertIsNone(manager._pending_metadata)

    def test_prefill_metadata_cannot_enqueue_a_decode_graph(self):
        manager = self.manager()
        self.capture(manager)
        descriptor = next(iter(manager.graphs))
        metadata = {
            "layer": SimpleNamespace(attn_state=SimpleNamespace(name="PrefillNoCache"))
        }
        manager.stage_replay_metadata(metadata)
        self.events.clear()
        with self.assertRaisesRegex(ValueError, "PrefillNoCache"):
            manager.run_fullgraph(descriptor)
        self.assertEqual(self.events, [])
        self.assertEqual(manager.replay_count, 0)

    def test_lora_enabled_zero_adapter_key_is_stable_at_replay(self):
        manager = self.manager()
        self.capture(manager, has_lora=True)
        self.assertTrue(all(key.has_lora for key in manager.runtime.capture_keys))
        descriptor = next(iter(manager.graphs))
        manager.stage_replay_metadata({})
        manager.run_fullgraph(descriptor)
        self.assertEqual(
            manager.runtime.updates[-1].batch_descriptor,
            manager.runtime.capture_keys[0],
        )

    def test_failed_capture_resets_flag_and_orders_capture_stream(self):
        manager = self.manager()
        original_stream = self.npu.current
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            self.capture(manager, fail=True)
        self.assertFalse(manager._capturing)
        self.assertEqual(manager.graphs, {})
        self.assertIs(self.npu.current, original_stream)
        self.assertIs(original_stream.waited[-1], manager._npu_capture_stream)

    def test_release_drops_graph_and_metadata_references(self):
        manager = self.manager()
        self.capture(manager)
        manager.stage_replay_metadata({"pending": True})
        manager._last_replay_metadata = {"old": True}
        manager.release_graphs()
        self.assertEqual(manager.graphs, {})
        self.assertEqual(manager._capture_metadata, {})
        self.assertIsNone(manager._pending_metadata)
        self.assertIsNone(manager._last_replay_metadata)
        self.assertTrue(manager.runtime.cleared)
        self.assertFalse(manager._graphs_captured)

    def test_runner_construction_does_not_require_model_state_before_load(self):
        source = (
            Path(__file__).resolve().parents[2] / "vllm_fl/worker/model_runner_v2.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        runner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunnerFLV2"
        )
        constructor = next(
            node
            for node in runner.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        klass = ast.ClassDef(
            name="RunnerUnderTest",
            bases=[ast.Name(id="UpstreamRunner", ctx=ast.Load())],
            keywords=[],
            body=[constructor],
            decorator_list=[],
            type_params=[],
        )

        class UpstreamRunner:
            def __init__(self, config, device):
                # Mirrors V2: model_state is absent until load_model().
                self.base_initialized = True

        namespace = {
            "UpstreamRunner": UpstreamRunner,
            "VllmConfig": object,
            "torch": torch,
            "CUDAGraphMode": _Mode,
            "uses_dense_hybrid_cache": lambda config: False,
            "logger": logging.getLogger(__name__),
        }
        module = ast.fix_missing_locations(ast.Module(body=[klass], type_ignores=[]))
        exec(compile(module, str(source), "exec"), namespace)
        config = SimpleNamespace(
            use_v2_model_runner=True,
            compilation_config=SimpleNamespace(cudagraph_mode=_Mode.FULL_DECODE_ONLY),
            model_config=object(),
            kv_transfer_config=None,
        )
        actual = namespace["RunnerUnderTest"](config, SimpleNamespace(type="npu"))
        self.assertTrue(actual.base_initialized)
        self.assertFalse(hasattr(actual, "model_state"))

    def test_unsupported_modes_are_rejected_without_affecting_none(self):
        with self.assertRaisesRegex(NotImplementedError, "NONE/FULL_DECODE_ONLY"):
            self.manager(mode=_Mode.PIECEWISE)
        config = SimpleNamespace(
            speculative_config=object(),
            parallel_config=SimpleNamespace(use_ubatching=True),
            descriptors=[],
        )
        with self.assertRaisesRegex(NotImplementedError, "speculation"):
            self.manager_type(
                config, SimpleNamespace(type="npu"), _Mode.FULL_DECODE_ONLY
            )
        self.manager_type(config, SimpleNamespace(type="npu"), _Mode.NONE)


if __name__ == "__main__":
    unittest.main()
