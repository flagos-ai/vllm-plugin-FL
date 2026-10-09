"""Isolated actual wrapper bodies with Mock collaborators, never vendor/GPU.

AST only selects production FunctionDef/ClassDef bodies. Tests EXECUTE those
bodies and actual contextvars/weakrefs, without modifying real sys.modules or
loading target packages. Mock collaborator success is not native/Gate proof.
"""

import ast
import asyncio
import contextvars
from collections import namedtuple
import functools
import gc
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import traceback
from types import SimpleNamespace
import unittest
import uuid
import weakref

ROOT = Path(__file__).resolve().parents[1] / "musa_gate_observer"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


class Unsupported(RuntimeError):
    pass


class MockTrace:
    def __init__(self, config, role, rank=None):
        self.records, self.errors = [], []
        self.seq = 0
        self.closed = False

    def emit(self, kind, data, status="ok"):
        self.records.append((kind, data, status))
        self.seq += 1

    def activation(self, hooks, sources, status="ok", error=None, client_scope=None):
        self.emit("activation", {"hooks": hooks, "sources": sources,
                                 **(client_scope or {})}, status)

    def fail(self, error):
        self.errors.append(error)

    def close(self):
        self.closed = True


def load_definitions(name, extra=None):
    source = ROOT / name
    tree = ast.parse(source.read_bytes(), filename=str(source), feature_version=(3, 10))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef,
                ast.AsyncFunctionDef, ast.ClassDef, ast.Assign))]
    scope = {"functools": functools, "inspect": inspect, "os": os,
             "sys": SimpleNamespace(modules={}, stderr=io.StringIO()),
             "weakref": weakref, "traceback": traceback, "Path": Path,
             "hashlib": hashlib, "Trace": MockTrace, "Unsupported": Unsupported,
             "unique_id": lambda: uuid.uuid4().hex, "digest": digest,
             "strict_json": json.loads}
    for key in ("api_context", "execute_context", "graph_context", "logical_context", "scenario_context"):
        scope[key] = contextvars.ContextVar(key, default=None)
    if extra:
        scope.update(extra)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), scope)
    return scope


def worker_fixture():
    scope = load_definitions("server.py")

    class MockGraph:
        def replay(self):
            return None

    class MockContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return None

    class MockWrapper:
        def __init__(self):
            self.concrete_graph_entries = {}

        def __call__(self, *args, **kwargs):
            return self.result

    class MockRunner:
        def __init__(self):
            self.calls = []

        def execute_model(self, scheduler_output):
            self.calls.append(scheduler_output)
            return self.result

    MockRunner.__module__ = "vllm_fl.worker.model_runner"
    MockRunner.__name__ = "ModelRunnerFL"

    class MockWorker:
        def __init__(self):
            self.rank = 0
            self.model_runner = MockRunner()

        def shutdown(self):
            return None

    Batch = namedtuple("MockBatch", "num_tokens num_reqs uniform has_lora num_active_loras")
    forward = SimpleNamespace(batch_descriptor=Batch(8, 1, False, False, 0),
              cudagraph_runtime_mode="PIECEWISE")
    graph_module = SimpleNamespace(GraphWrapper=MockWrapper,
                    is_forward_context_available=lambda: True,
                    get_forward_context=lambda: forward)
    vendor = SimpleNamespace(graph=MockContext, MUSAGraph=MockGraph)
    scope["sys"].modules.update({"torch_musa.musa_graph.graphs": vendor,
           "torch": SimpleNamespace(musa=vendor), "vllm_fl.compilation.graph": graph_module})
    config = SimpleNamespace(data={"required_role_sources": {"worker": [
        "vllm.plugins", "vllm.v1.worker.worker_base", "vllm_fl.worker.worker",
        "vllm_fl.worker.model_runner", "vllm_fl.compilation.graph",
        "vllm.forward_context", "torch_musa.musa_graph.graphs"]},
        "required_role_hooks": {"worker": ["runner.execute_model", "GraphWrapper.__call__",
        "torch.musa.graph.__enter__", "torch.musa.graph.__exit__", "MUSAGraph.replay",
        "worker.shutdown"]}, "vendor_graph": {"module": "torch_musa.musa_graph.graphs"}},
        verify_role=lambda role: {}, verify_module=lambda name: {})
    worker = MockWorker()
    hooks = scope["WorkerHooks"](config, worker)
    return scope, hooks, worker, MockGraph, MockWrapper, forward


class WorkerBodyTests(unittest.TestCase):
    def test_actual_install_uses_self_worker_and_retains_original_once(self):
        scope, hooks, worker, graph, wrapper, forward = worker_fixture()
        marker = object()
        worker.model_runner.result = marker
        hooks.install()  # Executes the actual production method; old body NameError.
        result = worker.model_runner.execute_model(SimpleNamespace(num_scheduled_tokens={"real-id": 1}))
        self.assertIs(result, marker)
        self.assertEqual(len(worker.model_runner.calls), 1)
        self.assertEqual([row[0] for row in hooks.trace.records].count("scheduler_begin"), 1)
        self.assertEqual([row[0] for row in hooks.trace.records].count("scheduler_return"), 1)
        self.assertFalse(hooks.trace.errors)

    def test_execute_delegates_once_and_preserves_exception_identity(self):
        scope, hooks, worker, graph, wrapper, forward = worker_fixture()
        calls = []
        error = RuntimeError("original model failure")

        def original(runner, scheduler_output, **kwargs):
            calls.append((runner, scheduler_output, kwargs))
            raise error

        scheduler = SimpleNamespace(num_scheduled_tokens={"request": 1})
        with self.assertRaises(RuntimeError) as caught:
            hooks.execute_builder(original)(worker.model_runner, scheduler, untouched=7)
        self.assertIs(caught.exception, error)
        self.assertEqual(calls, [(worker.model_runner, scheduler, {"untouched": 7})])
        self.assertIsNone(scope["execute_context"].get())

    def test_context_enter_exit_delegates_once_with_identical_arguments(self):
        scope, hooks, worker, graph, wrapper, forward = worker_fixture()
        context = SimpleNamespace(musa_graph=graph())
        calls, marker = [], object()

        def enter(value):
            calls.append(("enter", value))
            return marker

        def exit_(value, type_, value_, traceback_):
            calls.append(("exit", value, type_, value_, traceback_))
            return marker

        error = ValueError("original body error")
        self.assertIs(hooks.enter_builder(enter)(context), marker)
        self.assertIs(hooks.exit_builder(exit_)(context, ValueError, error, None), marker)
        self.assertEqual(calls, [("enter", context), ("exit", context, ValueError, error, None)])

    def test_native_replay_exception_identity_and_once(self):
        scope, hooks, worker, graph, wrapper, forward = worker_fixture()
        calls, error, value = [], RuntimeError("native replay error"), graph()

        def original(graph_, *args, **kwargs):
            calls.append((graph_, args, kwargs))
            raise error

        with self.assertRaises(RuntimeError) as caught:
            hooks.replay_builder(original)(value, 4, untouched=True)
        self.assertIs(caught.exception, error)
        self.assertEqual(calls, [(value, (4,), {"untouched": True})])

    def test_dead_entry_binding_is_discarded_before_graph_call(self):
        scope, hooks, worker, graph, wrapper, forward = worker_fixture()
        hooks.graph_module = scope["sys"].modules["vllm_fl.compilation.graph"]
        value, marker = wrapper(), object()
        value.result = marker
        uid = "wrapper-uid"
        hooks.wrapper_ids[value] = uid
        entry, graph_ = SimpleNamespace(), graph()
        # SimpleNamespace is not weakrefable; a plain owned Python object is.
        class Entry:
            pass
        entry = Entry()
        key = (uid, tuple(scope["batch_fields"](forward.batch_descriptor).items()))
        hooks.entries[key] = {"entry_ref": weakref.ref(entry), "graph_ref": weakref.ref(graph_)}
        del entry
        gc.collect()
        self.assertIs(hooks.graph_builder(lambda value: marker)(value), marker)
        self.assertNotIn(key, hooks.entries)
        self.assertIsNone(scope["graph_context"].get())
        self.assertFalse(hooks.trace.errors)

    def test_reused_graph_label_cannot_replace_actual_weak_identity(self):
        scope, hooks, worker, graph, wrapper, forward = worker_fixture()
        old_graph, new_graph, marker, calls = graph(), graph(), object(), []
        execution = {"execute_call_id": "exec", "engine_request_ids": ["engine"],
                     "request_tokens": {"engine": 1}, "rank": 0}
        binding = {"graph_ref": weakref.ref(old_graph), "graph_id": str(id(new_graph)),
                   "capture_id": "capture", "graph_generation": 1}
        token = scope["execute_context"].set(execution)
        token2 = scope["graph_context"].set({"binding": binding})
        try:
            result = hooks.replay_builder(lambda value: calls.append(value) or marker)(new_graph)
        finally:
            scope["graph_context"].reset(token2)
            scope["execute_context"].reset(token)
        self.assertIs(result, marker)
        self.assertEqual(calls, [new_graph])
        self.assertIsInstance(hooks.trace.errors[0], Unsupported)
        self.assertFalse(any(row[0] == "replay_success" for row in hooks.trace.records))

    def test_capture_generation_changes_and_tracks_actual_graph(self):
        scope, hooks, worker, graph, wrapper, forward = worker_fixture()
        one, two = graph(), graph()
        context = {"pending": [], "wrapper_uid": "wrapper", "batch": {"num_tokens": 8}}
        token = scope["graph_context"].set(context)
        calls = []
        original = lambda value, *args: calls.append(value) or None
        try:
            hooks.exit_builder(original)(SimpleNamespace(musa_graph=one), None, None, None)
            hooks.exit_builder(original)(SimpleNamespace(musa_graph=two), None, None, None)
        finally:
            scope["graph_context"].reset(token)
        self.assertEqual(len(calls), 2)
        self.assertEqual([item["graph_generation"] for item in context["pending"]], [1, 2])
        self.assertIs(context["pending"][0]["graph_ref"](), one)
        self.assertIs(context["pending"][1]["graph_ref"](), two)
        self.assertNotEqual(context["pending"][0]["capture_id"], context["pending"][1]["capture_id"])

    def test_unassociated_warmup_is_only_diagnostic(self):
        scope, hooks, worker, graph, wrapper, forward = worker_fixture()
        marker = object()
        self.assertIs(hooks.replay_builder(lambda value: marker)(graph()), marker)
        self.assertEqual(hooks.trace.records[-1][0], "native_replay_unassociated")
        self.assertFalse(hooks.trace.errors)

    def test_actual_wrapper_capture_then_replay_keeps_lifetime_generation(self):
        scope, hooks, worker, graph, wrapper, forward = worker_fixture()
        hooks.graph_module = scope["sys"].modules["vllm_fl.compilation.graph"]
        value, native_graph, marker, calls = wrapper(), graph(), object(), []
        class Entry:
            def __init__(self, graph_):
                self.graph = graph_
        def capture_original(value_):
            calls.append("capture-original")
            hooks.exit_builder(lambda *args: None)(
                SimpleNamespace(musa_graph=native_graph), None, None, None)
            value_.concrete_graph_entries[forward.batch_descriptor] = Entry(native_graph)
            return marker
        self.assertIs(hooks.graph_builder(capture_original)(value), marker)
        capture = next(data for kind, data, status in hooks.trace.records if kind == "capture_success")
        execution = {"execute_call_id": "actual-fixture-exec", "engine_request_ids": ["engine"],
                     "request_tokens": {"engine": 1}, "rank": 0}
        token = scope["execute_context"].set(execution)
        def replay_original(value_):
            calls.append("replay-original")
            return hooks.replay_builder(lambda graph_: marker)(native_graph)
        try:
            self.assertIs(hooks.graph_builder(replay_original)(value), marker)
        finally:
            scope["execute_context"].reset(token)
        replay = next(data for kind, data, status in hooks.trace.records if kind == "replay_success")
        self.assertEqual(calls, ["capture-original", "replay-original"])
        for name in ("capture_id", "graph_generation", "graph_id", "wrapper_uid", "batch"):
            self.assertEqual(capture[name], replay[name])
        self.assertEqual(replay["execute_call_id"], execution["execute_call_id"])
        self.assertFalse(hooks.trace.errors)

    def test_after_original_observation_failure_cannot_clobber_result(self):
        scope = load_definitions("server.py")
        calls, marker = [], object()
        plugins = SimpleNamespace(__name__="vllm.plugins")
        plugins.load_general_plugins = lambda: calls.append("original") or marker
        config = SimpleNamespace(verify_module=lambda name: {})
        scope["sys"].modules["vllm.plugins"] = plugins
        scope["state"] = lambda: {"config": config}
        def broken():
            calls.append("observer")
            raise FileExistsError("exclusive trace already exists")
        scope["install_lifecycle"] = broken
        scope["install_api"] = broken
        scope["register"]()
        self.assertIs(plugins.load_general_plugins(), marker)
        self.assertIs(plugins.load_general_plugins(), marker)
        self.assertEqual(calls.count("original"), 2)
        self.assertIn("FileExistsError", scope["sys"].stderr.getvalue())

    def test_failed_api_install_attempts_trace_only_once(self):
        scope = load_definitions("server.py")
        current = {"api_installed": False, "api_attempted": False, "config": object()}
        scope["state"] = lambda: current
        scope["sys"].modules.update({
            "vllm.entrypoints.openai.chat_completion.serving": SimpleNamespace(),
            "vllm.v1.engine.async_llm": SimpleNamespace()})
        calls = []
        def broken_trace(*args):
            calls.append(args)
            raise FileExistsError("actual exclusive filename")
        scope["Trace"] = broken_trace
        scope["safe_install"](scope["install_api"])
        scope["safe_install"](scope["install_api"])
        self.assertEqual(len(calls), 1)
        self.assertFalse(current["api_installed"])


class ClientBodyTests(unittest.TestCase):
    def fixture(self, module_name, failure=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        request = {"request_id": "original-id", "messages": [{"role": "user", "content": "fixture"}]}
        original_result = {"request_id": "original-id", "kind": "text", "text": "fixture",
                 "checks": {name: True for name in (
                 "non_empty", "minimum_length", "expected_semantics", "expected_order",
                 "no_bang_triplet", "no_mojibake", "no_control_characters",
                 "no_long_character_run", "no_repeated_word_run", "no_repeated_phrase")},
                 "passed": True, "response": {"id": "fixture-response"}}
        scope = load_definitions("pytest_plugin.py")
        calls = []
        gate = SimpleNamespace(ROOT=root, safe_slug=lambda value: value.rstrip("/").split("/")[-1],
                               resolve_served_model_name=lambda: "fixture-model")
        scenario = {"test_text": "text_single", "test_image": "image_single",
                    "test_mix_text_image": "mixed_concurrent_8"}[module_name]
        expected = {"scenario": scenario, "original_request_id": "original-id",
                    "logical_id": "fixture-logical", "ordinal": 0,
                    "request_input_sha256": digest(request)}
        async def execute(client, request_, served):
            calls.append(("execute", client, request_, served))
            return original_result
        def run(scenario_):
            calls.append(("run", scenario_))
            result = asyncio.run(gate.execute_request(None, request, "fixture-model"))
            destination = root / "fixture-model" / "port-1" / (scenario_ + ".json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({"schema_version": 1,
                "case": {"scenario": scenario_, "port": "1", "model_path": None,
                         "served_model_name": "fixture-model", "concurrency": 1},
                "input": [request], "output": [result],
                "summary": {"request_count": 1, "passed_count": 1, "failed_count": 0, "passed": True}}))
            if failure is not None:
                raise failure
            return original_result
        class HttpClient:
            async def send(self, request, *args, **kwargs):
                calls.append(("send", request, args, kwargs))
                if getattr(self, "error", None) is not None:
                    raise self.error
                return self.reply
        gate.execute_request, gate.run_case = execute, run
        test = SimpleNamespace(__name__=module_name, run_case=run)
        scope["sys"].modules.update({"case_utils": gate, module_name: test,
                                     "httpx._client": SimpleNamespace(AsyncClient=HttpClient)})
        sources = {name: {"path": str(root / (name + ".py")), "bytes": 1, "sha256": "a"*64}
                   for name in ("case_utils", "test_text", "test_image", "test_mix_text_image")}
        config = SimpleNamespace(route={"requests": [expected]},
             data={"required_role_sources": {"client": list(sources) + ["httpx._client", "openai._base_client"]},
                   "required_role_hooks": {"client": scope["HOOKS"]}},
             verify_gate_kit=lambda: {name + ".py": fp for name, fp in sources.items()},
             verify_role=lambda *args, **kwargs: sources)
        scope["Config"] = lambda: config
        def fixture_read(path):
            raw = Path(path).read_bytes()
            return raw, {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        scope["read_original"] = fixture_read
        old_results, old_port = os.environ.get("RESULTS_DIR"), os.environ.get("PORT")
        old_model, old_served = os.environ.pop("MODEL_PATH", None), os.environ.get("SERVED_MODEL_NAME")
        os.environ.update(RESULTS_DIR=str(root), PORT="1", SERVED_MODEL_NAME="fixture-model")
        def restore():
            for name, value in (("RESULTS_DIR", old_results), ("PORT", old_port),
                                ("MODEL_PATH", old_model), ("SERVED_MODEL_NAME", old_served)):
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.addCleanup(restore)
        return scope, test, scenario, original_result, calls

    def test_each_original_single_file_scope_without_importing_other_tests(self):
        for name in ("test_text", "test_image", "test_mix_text_image"):
            with self.subTest(module=name):
                scope, module, scenario, original, calls = self.fixture(name)
                before = set(scope["sys"].modules)
                scope["pytest_collection_modifyitems"](None, None, [SimpleNamespace(module=module)])
                result = module.run_case(scenario)
                self.assertIs(result, original)
                self.assertEqual(set(scope["sys"].modules), before)
                self.assertEqual([call[0] for call in calls], ["run", "execute"])
                activation = scope["_state"]["trace"].records[0]
                self.assertEqual(activation[1]["loaded_test_modules"], [name])
                self.assertFalse(scope["_state"]["trace"].errors)
                self.assertIsNone(scope["scenario_context"].get())
                self.assertIsNone(scope["logical_context"].get())

    def test_original_case_exception_identity_preserved_with_original_report(self):
        error = AssertionError("original quality assertion fixture")
        scope, module, scenario, original, calls = self.fixture("test_text", error)
        scope["install"](["test_text"])
        with self.assertRaises(AssertionError) as caught:
            module.run_case(scenario)
        self.assertIs(caught.exception, error)
        self.assertEqual([call[0] for call in calls], ["run", "execute"])
        self.assertTrue(any(row[0] == "scenario_result" for row in scope["_state"]["trace"].records))

    def test_missing_config_does_not_change_original_collected_case(self):
        scope, module, scenario, original, calls = self.fixture("test_text")
        original_function = module.run_case
        def missing():
            raise Unsupported("UNBOUND explicit source configuration")
        scope["Config"] = missing
        scope["pytest_collection_modifyitems"](None, None, [SimpleNamespace(module=module)])
        self.assertIs(module.run_case, original_function)
        self.assertIs(module.run_case(scenario), original)
        self.assertIn("UNBOUND", scope["sys"].stderr.getvalue())

    def test_sdk_send_original_once_and_result_identity(self):
        scope, module, scenario, original, calls = self.fixture("test_text")
        scope["install"](["test_text"])
        client = scope["sys"].modules["httpx._client"].AsyncClient()
        client.reply = SimpleNamespace(status_code=200)
        request = SimpleNamespace(method="POST", url=SimpleNamespace(path="/v1/chat/completions"), content=b"{}")
        token = scope["logical_context"].set({"logical_id": "fixture-logical"})
        try:
            result = asyncio.run(client.send(request, "extra", unchanged=7))
        finally:
            scope["logical_context"].reset(token)
        self.assertIs(result, client.reply)
        self.assertEqual(calls, [("send", request, ("extra",), {"unchanged": 7})])
        self.assertEqual(scope["_state"]["trace"].records[-1][0], "observed_send_call")

    def test_sdk_send_exception_identity_once(self):
        scope, module, scenario, original, calls = self.fixture("test_text")
        scope["install"](["test_text"])
        client = scope["sys"].modules["httpx._client"].AsyncClient()
        client.error = RuntimeError("original SDK transport error")
        request = SimpleNamespace(method="POST", url=SimpleNamespace(path="/v1/chat/completions"), content=b"{}")
        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(client.send(request, unchanged=8))
        self.assertIs(caught.exception, client.error)
        self.assertEqual(calls, [("send", request, (), {"unchanged": 8})])


if __name__ == "__main__":
    unittest.main(verbosity=2)
