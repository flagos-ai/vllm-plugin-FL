# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for the vLLM 0.32 worker import and PP protocol."""

import ast
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/worker/worker.py"


@pytest.fixture(scope="module")
def source_tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def test_worker_logger_uses_the_configured_vllm_namespace(source_tree):
    assignment = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "logger"
            for target in node.targets
        )
    )
    initialize = Mock()
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), str(SOURCE), "exec"),
        {"init_logger": initialize, "__name__": "vllm_fl.worker.worker"},
    )
    initialize.assert_called_once_with("vllm.fl.worker")


@pytest.mark.parametrize(
    "device_type,module_name,class_name",
    [
        ("txda", "vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
        ("txda", "vllm_fl.worker.model_runner", "ModelRunnerFL"),
        ("cuda", "vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
    ],
)
def test_runner_receipt_records_the_actual_constructed_class(
    source_tree, caplog, device_type, module_name, class_name
):
    worker = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerFL"
    )
    initialize = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == "init_device"
    )
    receipt = next(
        node
        for node in ast.walk(initialize)
        if isinstance(node, ast.If)
        and any(
            isinstance(part, ast.Constant)
            and isinstance(part.value, str)
            and part.value.startswith("TXDA runner initialized:")
            for part in ast.walk(node)
        )
    )
    runner_type = type(class_name, (), {"__module__": module_name})
    instance = SimpleNamespace(
        rank=3, model_runner=runner_type(), use_v2_model_runner=True
    )
    with caplog.at_level(logging.INFO):
        exec(
            compile(ast.Module(body=[receipt], type_ignores=[]), str(SOURCE), "exec"),
            {
                "self": instance,
                "current_platform": SimpleNamespace(device_type=device_type),
                "logger": logging.getLogger("test.txda.runner.receipt"),
            },
        )
    if device_type == "txda":
        assert (
            f"TXDA runner initialized: rank=3 class={module_name}.{class_name} use_v2=True"
            in caplog.text
        )
    else:
        assert "TXDA runner initialized" not in caplog.text


def test_worker_utils_import_runs_with_only_the_vllm_032_exports(
    source_tree, monkeypatch
):
    utils = ModuleType("vllm.v1.worker.utils")
    utils.request_memory = Mock()
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    imports = [
        node
        for node in source_tree.body
        if isinstance(node, ast.ImportFrom) and node.module == utils.__name__
    ]
    namespace = {}
    exec(
        compile(ast.Module(body=imports, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    assert namespace["request_memory"] is utils.request_memory


def test_distributed_init_import_uses_the_032_determinism_package(
    source_tree, monkeypatch
):
    batch_invariant = ModuleType("vllm.model_executor.determinism.batch_invariant")
    batch_invariant.init_batch_invariance = Mock()
    monkeypatch.setitem(sys.modules, batch_invariant.__name__, batch_invariant)
    initialize = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "init_worker_distributed_environment"
    )
    imports = [
        node
        for node in initialize.body
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "init_batch_invariance" for alias in node.names)
    ]
    assert len(imports) == 1
    namespace = {}
    exec(
        compile(ast.Module(body=imports, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    assert namespace["init_batch_invariance"] is batch_invariant.init_batch_invariance


@pytest.fixture
def pp_worker(source_tree):
    worker = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerFL"
    )
    method = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_model"
    )
    # Preserve the production body, while avoiding GPU-only decorators here.
    method = ast.parse(ast.unparse(method)).body[0]
    method.decorator_list = []
    events = []
    tp_group = object()
    metadata_handle = Mock(name="metadata_send")
    device_handle = Mock(name="device_send")
    device_handle.wait.side_effect = lambda: events.append("device_wait")

    class Intermediate:
        def __init__(self, tensors, **kwargs):
            self.tensors = tensors
            self.comm = kwargs

    class Output:
        pass

    class AsyncOutput:
        pass

    class Group:
        is_first_rank = True
        is_last_rank = False

        def irecv_tensor_dict(self, *, all_gather_group):
            assert all_gather_group is tp_group
            events.append("recv")
            return {"residual": "received"}, ["recv_handle"], ["postprocess"]

        def isend_tensor_dict(self, tensors, *, all_gather_group):
            assert all_gather_group is tp_group
            assert tensors == {"hidden_states": "send"}
            events.append("send")
            return [metadata_handle, device_handle]

    group = Group()
    namespace = {
        "ModelRunnerOutput": Output,
        "AsyncModelRunnerOutput": AsyncOutput,
        "IntermediateTensors": Intermediate,
        "AsyncIntermediateTensors": Intermediate,
        "NoneType": type(None),
        "get_pp_group": lambda: group,
        "get_tp_group": lambda: tp_group,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(SOURCE), "exec"), namespace)
    runner = SimpleNamespace(
        is_pooling_model=False,
        execute_model=Mock(
            side_effect=lambda *args: events.append("model") or Output()
        ),
        pool=Mock(return_value=Output()),
    )
    instance = SimpleNamespace(
        _pp_send_work=[],
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=2, distributed_executor_backend="mp"
            )
        ),
        use_v2_model_runner=True,
        model_runner=runner,
        annotate_profile=lambda output: nullcontext(),
    )
    return SimpleNamespace(
        run=lambda tokens=1: namespace["execute_model"](
            instance, SimpleNamespace(total_num_scheduled_tokens=tokens)
        ),
        worker=instance,
        group=group,
        runner=runner,
        events=events,
        intermediate=Intermediate,
        output=Output,
        metadata_handle=metadata_handle,
        device_handle=device_handle,
    )


@pytest.mark.parametrize("use_v2", [False, True])
def test_receive_uses_the_032_protocol_without_the_old_sp_predicate(pp_worker, use_v2):
    pp_worker.group.is_first_rank = False
    pp_worker.worker.use_v2_model_runner = use_v2
    pp_worker.run()
    intermediate = pp_worker.runner.execute_model.call_args.args[1]
    assert intermediate.tensors == {"residual": "received"}
    assert intermediate.comm == {
        "comm_handles": ["recv_handle"],
        "comm_postprocess": ["postprocess"],
    }
    assert pp_worker.events == ["recv", "model"]


@pytest.mark.parametrize(
    "use_v2,pooling", [(False, False), (True, False), (True, True)]
)
def test_send_retains_only_device_handles_and_finishes_v2_pooling(
    pp_worker, use_v2, pooling
):
    pp_worker.worker.use_v2_model_runner = use_v2
    pp_worker.runner.is_pooling_model = pooling
    pp_worker.runner.execute_model.side_effect = None
    pp_worker.runner.execute_model.return_value = pp_worker.intermediate(
        {"hidden_states": "send"}
    )

    result = pp_worker.run()

    assert pp_worker.worker._pp_send_work == [pp_worker.device_handle]
    assert pp_worker.events == ["send"]
    if use_v2 and pooling:
        assert result is pp_worker.runner.pool.return_value
        pp_worker.runner.pool.assert_called_once_with()
    else:
        assert result is None
        pp_worker.runner.pool.assert_not_called()


def test_next_forward_waits_for_device_send_and_leaves_metadata_to_coordinator(
    pp_worker,
):
    pp_worker.runner.execute_model.side_effect = None
    pp_worker.runner.execute_model.return_value = pp_worker.intermediate(
        {"hidden_states": "send"}
    )
    pp_worker.run()
    pp_worker.runner.execute_model.return_value = pp_worker.output()
    pp_worker.run()
    assert pp_worker.events == ["send", "device_wait"]
    pp_worker.device_handle.wait.assert_called_once_with()
    pp_worker.metadata_handle.wait.assert_not_called()
    assert pp_worker.worker._pp_send_work == []


def test_empty_forward_does_not_receive_pipeline_tensors(pp_worker):
    pp_worker.group.is_first_rank = False
    pp_worker.run(tokens=0)
    assert pp_worker.events == ["model"]
