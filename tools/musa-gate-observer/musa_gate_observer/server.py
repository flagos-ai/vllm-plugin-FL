"""Deferred wrappers of real, source-bound Python methods. No eager GPU imports."""

import functools
import inspect
import os
import sys
import traceback
import weakref

from .trace import (
    Config,
    Trace,
    Unsupported,
    api_context,
    execute_context,
    graph_context,
    unique_id,
)

_state = {}
_workers = {}


def patch(owner, name, builder):
    current = getattr(owner, name)
    original = getattr(current, "_musa_gate_original", current)
    wrapped = builder(original)
    wrapped._musa_gate_original = original
    setattr(owner, name, wrapped)
    if getattr(owner, name) is not wrapped:
        raise Unsupported(f"Method replacement was not retained: {name}")


def state():
    if _state.get("pid") != os.getpid():
        _state.clear()
        _state.update(
            pid=os.getpid(),
            config=Config(),
            api_installed=False,
            api_attempted=False,
            lifecycle_installed=False,
            api_trace=None,
        )
        _workers.clear()
    return _state


def batch_fields(value):
    if value is None:
        return None
    names = ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras")
    result = {name: getattr(value, name) for name in names}
    if (
        type(result["num_tokens"]) is not int
        or result["num_tokens"] < 0
        or result["num_reqs"] is not None
        and type(result["num_reqs"]) is not int
        or type(result["uniform"]) is not bool
        or type(result["has_lora"]) is not bool
        or type(result["num_active_loras"]) is not int
    ):
        raise Unsupported("Unsupported batch descriptor types")
    return result


class WorkerHooks:
    def __init__(self, config, worker):
        self.config, self.worker = config, worker
        self.trace = Trace(config, "worker", worker.rank)
        self.wrapper_ids = weakref.WeakKeyDictionary()
        self.entries = {}
        self.generation = 0

    def install(self):
        minimum = {
            "vllm.plugins",
            "vllm.v1.worker.worker_base",
            "vllm_fl.worker.worker",
            "vllm_fl.worker.model_runner",
            "vllm_fl.compilation.graph",
            "vllm.forward_context",
            "torch_musa.musa_graph.graphs",
        }
        if not minimum <= set(self.config.data["required_role_sources"]["worker"]):
            raise Unsupported("Missing required worker/graph source profile")
        sources = self.config.verify_role("worker")
        vendor = self.config.data["vendor_graph"]
        name = vendor.get("module", "UNBOUND")
        if name != "torch_musa.musa_graph.graphs":
            raise Unsupported("Unbound vendor graph module")
        module = sys.modules.get(name)
        torch = sys.modules.get("torch")
        graph_module = sys.modules.get("vllm_fl.compilation.graph")
        if module is None or torch is None or graph_module is None:
            raise Unsupported("Normal worker has not loaded the graph modules")
        self.config.verify_module(name)
        context_class, native_class = module.graph, module.MUSAGraph
        if (
            torch.musa.graph is not context_class
            or torch.musa.MUSAGraph is not native_class
        ):
            raise Unsupported("Vendor graph aliases differ from bound module")
        for owner, attr, signature in (
            (context_class, "__enter__", "(self)"),
            (context_class, "__exit__", "(self, exc_type, exc_value, traceback)"),
            (native_class, "replay", "(self)"),
        ):
            original = getattr(
                getattr(owner, attr), "_musa_gate_original", getattr(owner, attr)
            )
            if (
                not inspect.isfunction(original)
                or str(inspect.signature(original)) != signature
            ):
                raise Unsupported(f"Unbound Python graph method: {attr}")
        runner = self.worker.model_runner
        if (
            type(runner).__module__ != "vllm_fl.worker.model_runner"
            or type(runner).__name__ != "ModelRunnerFL"
        ):
            raise Unsupported("This draft observes only the actual FL V1 runner")
        self.graph_module = graph_module
        patch(type(runner), "execute_model", self.execute_builder)
        patch(graph_module.GraphWrapper, "__call__", self.graph_builder)
        patch(context_class, "__enter__", self.enter_builder)
        patch(context_class, "__exit__", self.exit_builder)
        patch(native_class, "replay", self.replay_builder)
        patch(type(self.worker), "shutdown", self.shutdown_builder)
        hooks = [
            "runner.execute_model",
            "GraphWrapper.__call__",
            "torch.musa.graph.__enter__",
            "torch.musa.graph.__exit__",
            "MUSAGraph.replay",
            "worker.shutdown",
        ]
        if set(hooks) != set(self.config.data["required_role_hooks"]["worker"]):
            raise Unsupported("Worker hook contract differs")
        self.trace.activation(hooks, sources)

    def execute_builder(self, original):
        @functools.wraps(original)
        def wrapped(runner, scheduler_output, *args, **kwargs):
            token = None
            try:
                requests = dict(scheduler_output.num_scheduled_tokens)
                if not all(
                    type(key) is str and type(value) is int and value >= 0
                    for key, value in requests.items()
                ):
                    raise Unsupported("Unsupported scheduler ID/token types")
                context = {
                    "execute_call_id": unique_id(),
                    "request_tokens": requests,
                    "engine_request_ids": list(requests),
                    "rank": self.worker.rank,
                }
                token = execute_context.set(context)
                self.trace.emit("scheduler_begin", context)
            except Exception as error:
                self.trace.fail(error)
            try:
                result = original(runner, scheduler_output, *args, **kwargs)
                self.trace.emit("scheduler_return", execute_context.get() or {})
                return result
            except BaseException:
                self.trace.emit(
                    "scheduler_exception", execute_context.get() or {}, "error"
                )
                raise
            finally:
                if token is not None:
                    execute_context.reset(token)

        return wrapped

    def graph_builder(self, original):
        @functools.wraps(original)
        def wrapped(wrapper, *args, **kwargs):
            token, context = None, None
            try:
                if self.graph_module.is_forward_context_available():
                    forward = self.graph_module.get_forward_context()
                    descriptor = forward.batch_descriptor
                    batch = batch_fields(descriptor)
                    if batch is not None:
                        for key, item in list(self.entries.items()):
                            if (
                                item["entry_ref"]() is None
                                or item["graph_ref"]() is None
                            ):
                                self.entries.pop(key, None)
                        uid = self.wrapper_ids.get(wrapper)
                        if uid is None:
                            uid = unique_id()
                            self.wrapper_ids[wrapper] = uid
                        key = (uid, tuple(batch.items()))
                        entry = wrapper.concrete_graph_entries.get(descriptor)
                        binding = self.entries.get(key)
                        if binding is not None and (
                            entry is None
                            or binding["entry_ref"]() is not entry
                            or binding["graph_ref"]() is not entry.graph
                        ):
                            self.entries.pop(key, None)
                            binding = None
                        context = {
                            "wrapper_uid": uid,
                            "batch": batch,
                            "key": key,
                            "descriptor": descriptor,
                            "pending": [],
                            "binding": binding,
                            "runtime_mode": str(forward.cudagraph_runtime_mode),
                        }
                        token = graph_context.set(context)
                        self.trace.emit(
                            "graph_call",
                            {
                                "wrapper_uid": uid,
                                "batch": batch,
                                "runtime_mode": context["runtime_mode"],
                                **(execute_context.get() or {}),
                            },
                        )
            except Exception as error:
                self.trace.fail(error)
            try:
                result = original(wrapper, *args, **kwargs)
                if context is not None and context["pending"]:
                    try:
                        if len(context["pending"]) != 1:
                            raise Unsupported("Ambiguous nested graph captures")
                        pending = context["pending"][0]
                        entry = wrapper.concrete_graph_entries.get(
                            context["descriptor"]
                        )
                        if entry is None or pending["graph_ref"]() is not entry.graph:
                            raise Unsupported(
                                "Captured graph was not bound to actual GraphEntry"
                            )
                        binding = dict(pending, entry_ref=weakref.ref(entry))
                        if (
                            context["key"] not in self.entries
                            and len(self.entries) >= 4096
                        ):
                            raise Unsupported("Live graph binding cap exceeded")
                        self.entries[context["key"]] = binding
                        data = {
                            key: value
                            for key, value in pending.items()
                            if key != "graph_ref"
                        }
                        self.trace.emit("capture_success", data)
                    except Exception as error:
                        self.trace.fail(error)
                return result
            except BaseException:
                self.trace.emit(
                    "graph_call_exception",
                    {"wrapper_uid": context["wrapper_uid"] if context else None},
                    "error",
                )
                raise
            finally:
                if token is not None:
                    graph_context.reset(token)

        return wrapped

    def enter_builder(self, original):
        @functools.wraps(original)
        def wrapped(context_object):
            # Validate weakref on the actual production-created graph, never create one.
            try:
                weakref.ref(context_object.musa_graph)
            except Exception as error:
                self.trace.fail(error)
            try:
                return original(context_object)
            except BaseException:
                self.trace.emit("capture_enter_exception", {}, "error")
                raise

        return wrapped

    def exit_builder(self, original):
        @functools.wraps(original)
        def wrapped(context_object, exc_type, exc_value, traceback):
            try:
                result = original(context_object, exc_type, exc_value, traceback)
            except BaseException:
                self.trace.emit("capture_exit_exception", {}, "error")
                raise
            if exc_type is not None:
                self.trace.emit(
                    "capture_body_exception", {"type": exc_type.__name__}, "error"
                )
            else:
                try:
                    context = graph_context.get()
                    if context is None:
                        self.trace.emit("native_capture_unassociated", {}, "ok")
                    else:
                        self.generation += 1
                        pending = {
                            "capture_id": unique_id(),
                            "graph_generation": self.generation,
                            "graph_id": str(id(context_object.musa_graph)),
                            "graph_ref": weakref.ref(context_object.musa_graph),
                            "wrapper_uid": context["wrapper_uid"],
                            "batch": context["batch"],
                            "rank": self.worker.rank,
                            "engine_request_ids": [],
                            **(execute_context.get() or {}),
                        }
                        context["pending"].append(pending)
                except Exception as error:
                    self.trace.fail(error)
            return result

        return wrapped

    def replay_builder(self, original):
        @functools.wraps(original)
        def wrapped(graph, *args, **kwargs):
            try:
                result = original(graph, *args, **kwargs)
            except BaseException:
                self.trace.emit("replay_exception", {}, "error")
                raise
            try:
                context, execution = graph_context.get(), execute_context.get()
                binding = context["binding"] if context else None
                if context is None or execution is None:
                    self.trace.emit(
                        "native_replay_unassociated",
                        {
                            "graph_id": str(id(graph)),
                            "has_graph_wrapper_context": context is not None,
                            "has_scheduler_context": execution is not None,
                        },
                    )
                    return result
                if binding is None or binding["graph_ref"]() is not graph:
                    raise Unsupported(
                        "Replay lacks lifetime/capture/scheduler association"
                    )
                data = {
                    key: value
                    for key, value in binding.items()
                    if key
                    not in {
                        "graph_ref",
                        "entry_ref",
                        "engine_request_ids",
                        "execute_call_id",
                        "request_tokens",
                    }
                }
                self.trace.emit("replay_success", dict(data, **execution))
            except Exception as error:
                self.trace.fail(error)
            return result

        return wrapped

    def shutdown_builder(self, original):
        @functools.wraps(original)
        def wrapped(worker, *args, **kwargs):
            try:
                return original(worker, *args, **kwargs)
            except BaseException as error:
                self.trace.fail(error)
                raise
            finally:
                self.trace.close()

        return wrapped


def activate_worker(wrapper):
    current = state()
    worker = wrapper.worker
    key = (os.getpid(), id(worker))
    if key in _workers:
        return
    hooks = WorkerHooks(current["config"], worker)
    _workers[key] = hooks
    try:
        hooks.install()
    except Exception as error:
        if hooks.trace.seq == 0:
            hooks.trace.activation([], {}, "unsupported", str(error))
        hooks.trace.fail(error)


def install_lifecycle():
    current = state()
    if current["lifecycle_installed"]:
        return
    module = sys.modules.get("vllm.v1.worker.worker_base")
    if module is None:
        return
    current["config"].verify_module(module.__name__)

    def builder(original):
        @functools.wraps(original)
        def wrapped(wrapper, *args, **kwargs):
            result = original(wrapper, *args, **kwargs)
            try:
                activate_worker(wrapper)
            except Exception as error:
                sys.stderr.write(f"MUSA_GATE_OBSERVER_UNSUPPORTED: {error}\n")
            return result

        return wrapped

    patch(module.WorkerWrapperBase, "init_device", builder)
    current["lifecycle_installed"] = True


def install_api():
    current = state()
    if current["api_installed"] or current["api_attempted"]:
        return
    chat = sys.modules.get("vllm.entrypoints.openai.chat_completion.serving")
    engine = sys.modules.get("vllm.v1.engine.async_llm")
    if chat is None or engine is None:
        return
    current["api_attempted"] = True
    config = current["config"]
    trace = Trace(config, "api")
    current["api_trace"] = trace
    try:
        minimum = {
            "vllm.plugins",
            "vllm.entrypoints.openai.api_server",
            "vllm.entrypoints.openai.chat_completion.serving",
            "vllm.v1.engine.async_llm",
            "vllm.v1.engine.input_processor",
        }
        if not minimum <= set(config.data["required_role_sources"]["api"]):
            raise Unsupported("Missing required API/engine source profile")
        sources = config.verify_role("api")

        def chat_builder(original):
            @functools.wraps(original)
            async def wrapped(serving, *args, **kwargs):
                call_id = unique_id()
                token = api_context.set({"api_call_id": call_id})
                try:
                    result = await original(serving, *args, **kwargs)
                    trace.emit(
                        "api_response",
                        {
                            "api_call_id": call_id,
                            "response_id": getattr(result, "id", None),
                        },
                    )
                    return result
                except BaseException:
                    trace.emit("api_exception", {"api_call_id": call_id}, "error")
                    raise
                finally:
                    api_context.reset(token)

            return wrapped

        def generate_builder(original):
            @functools.wraps(original)
            def wrapped(engine_object, *args, **kwargs):
                trace.emit(
                    "generate_call",
                    dict(
                        api_context.get() or {},
                        external_request_id=kwargs.get(
                            "request_id", args[2] if len(args) > 2 else None
                        ),
                    ),
                )
                return original(
                    engine_object, *args, **kwargs
                )  # Exact original async generator.

            return wrapped

        def queued_builder(original):
            @functools.wraps(original)
            async def wrapped(engine_object, request, *args, **kwargs):
                result = await original(engine_object, request, *args, **kwargs)
                trace.emit(
                    "engine_queued",
                    dict(
                        api_context.get() or {},
                        external_req_id=request.external_req_id,
                        engine_request_id=request.request_id,
                    ),
                )
                return result

            return wrapped

        patch(chat.OpenAIServingChat, "create_chat_completion", chat_builder)
        patch(engine.AsyncLLM, "generate", generate_builder)
        patch(engine.AsyncLLM, "_add_request", queued_builder)
        hooks = [
            "OpenAIServingChat.create_chat_completion",
            "AsyncLLM.generate",
            "AsyncLLM._add_request",
        ]
        if set(hooks) != set(config.data["required_role_hooks"]["api"]):
            raise Unsupported("API hook contract differs")
        trace.activation(hooks, sources)
        current["api_installed"] = True
    except Exception as error:
        if trace.seq == 0:
            trace.activation([], {}, "unsupported", str(error))
        trace.fail(error)


def safe_install(callback):
    """Observation failure cannot replace an original model result/exception."""
    try:
        callback()
    except Exception:
        sys.stderr.write("MUSA_GATE_OBSERVER_UNSUPPORTED\n" + traceback.format_exc())


def register():
    """Called by the real general-plugin loader; never import GPU target modules."""
    try:
        current = state()
        plugins = sys.modules.get("vllm.plugins")
        if plugins is None:
            raise Unsupported("General plugin loaded outside vllm.plugins")
        current["config"].verify_module("vllm.plugins")
    except Exception:
        sys.stderr.write("MUSA_GATE_OBSERVER_UNSUPPORTED\n" + traceback.format_exc())
        return

    def load_builder(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            safe_install(install_lifecycle)
            safe_install(install_api)
            return result

        return wrapped

    patch(plugins, "load_general_plugins", load_builder)
    safe_install(install_lifecycle)
    safe_install(install_api)
    api_server = sys.modules.get("vllm.entrypoints.openai.api_server")
    if api_server is not None:
        try:
            current["config"].verify_module(api_server.__name__)
        except Exception:
            sys.stderr.write(
                "MUSA_GATE_OBSERVER_UNSUPPORTED\n" + traceback.format_exc()
            )
            return

        def build_builder(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                safe_install(install_api)
                return result

            return wrapped

        def init_builder(original):
            @functools.wraps(original)
            async def wrapped(*args, **kwargs):
                result = await original(*args, **kwargs)
                safe_install(install_api)
                return result

            return wrapped

        patch(api_server, "build_app", build_builder)
        patch(api_server, "init_app_state", init_builder)
