# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU fullgraph regression for the actual MUSA worker logging setup block.

The worker block is extracted unchanged from production source. Real CPU
Dynamo traces a tensor operator containing the same logging.Logger.debug call
as FlagGems; this does not validate native MUSA graphs or vendor kernels.
"""

import ast
import builtins
import logging
import math
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch._dynamo import config as dynamo_config

SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/worker/model_runner.py"
OP_LOGGER = logging.getLogger("musa_logging_contract.rms_norm")


class Mode(IntEnum):
    NONE = 0
    VLLM_COMPILE = 3


@pytest.fixture(autouse=True)
def isolated_dynamo_logging():
    with dynamo_config.patch(ignore_logger_methods=set()):
        torch._dynamo.reset()
        yield
        torch._dynamo.reset()


def worker_setup(device="musa", mode=Mode.VLLM_COMPILE):
    module = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    runner = next(
        n
        for n in module.body
        if isinstance(n, ast.ClassDef) and n.name == "ModelRunnerFL"
    )
    load_model = next(
        n
        for n in runner.body
        if isinstance(n, ast.FunctionDef) and n.name == "load_model"
    )
    blocks = [
        n
        for n in load_model.body
        if isinstance(n, ast.If)
        and any(
            isinstance(child, ast.ImportFrom)
            and child.module == "vllm_fl.dispatch"
            and any(alias.name == "warmup_cached_ops" for alias in child.names)
            for child in ast.walk(n)
        )
    ]
    assert len(blocks) == 1
    warmed_with = []
    warmup = Mock(
        side_effect=lambda: warmed_with.append(set(dynamo_config.ignore_logger_methods))
    )

    def import_module(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm_fl.dispatch":
            return SimpleNamespace(warmup_cached_ops=warmup)
        return builtins.__import__(name, globals, locals, fromlist, level)

    compilation = SimpleNamespace(
        mode=mode, cudagraph_mode="PIECEWISE", backend="eager"
    )
    namespace = {
        "__builtins__": {**vars(builtins), "__import__": import_module},
        "current_platform": SimpleNamespace(device_type=device),
        "CompilationMode": Mode,
        "self": SimpleNamespace(compilation_config=compilation),
    }
    exec(
        compile(ast.Module(body=blocks, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    assert vars(compilation) == {
        "mode": mode,
        "cudagraph_mode": "PIECEWISE",
        "backend": "eager",
    }
    return warmup, warmed_with


def logging_rms(x, weight):
    OP_LOGGER.debug("FlagGems RMSNorm tracing sentinel")
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight


def compiler(graphs):
    def compile_graph(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    return compile_graph


def test_cold_logging_fullgraph_reports_exact_logger_limitation():
    graphs = []
    compiled = torch.compile(logging_rms, backend=compiler(graphs), fullgraph=True)
    with pytest.raises(torch._dynamo.exc.Unsupported, match="[Ll]ogger"):
        compiled(torch.ones(2, 4), torch.ones(4))
    assert graphs == []


def test_musa_worker_setup_keeps_one_fullgraph_and_numerical_outputs(caplog):
    caplog.set_level(logging.DEBUG, logger=OP_LOGGER.name)
    warmup, warmed_with = worker_setup()
    warmup.assert_called_once()
    x = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
    weight = torch.tensor([0.5, 1.5])
    logging_rms(x, weight)
    assert sum(record.name == OP_LOGGER.name for record in caplog.records) == 1
    caplog.clear()
    graphs = []
    compiled = torch.compile(logging_rms, backend=compiler(graphs), fullgraph=True)
    for sample in (x, x * 2):
        expected = torch.tensor(
            [
                [
                    value * scale / math.sqrt(sum(v * v for v in row) / len(row) + 1e-6)
                    for value, scale in zip(row, weight.tolist())
                ]
                for row in sample.tolist()
            ]
        )
        torch.testing.assert_close(compiled(sample, weight), expected)
    assert len(graphs) == 1
    assert warmed_with == [{logging.Logger.debug}]
    assert not any(record.name == OP_LOGGER.name for record in caplog.records)


@pytest.mark.parametrize("device", ["cuda", "npu", "xpu"])
def test_other_backends_keep_logging_behavior(device):
    warmup, _ = worker_setup(device)
    assert dynamo_config.ignore_logger_methods == set()
    assert warmup.call_count == (1 if device == "npu" else 0)
    compiled = torch.compile(logging_rms, backend="eager", fullgraph=True)
    with pytest.raises(torch._dynamo.exc.Unsupported, match="[Ll]ogger"):
        compiled(torch.ones(2, 4), torch.ones(4))


def test_musa_compilation_none_does_not_change_logging():
    warmup, _ = worker_setup(mode=Mode.NONE)
    warmup.assert_not_called()
    assert dynamo_config.ignore_logger_methods == set()


def test_preserves_user_methods_and_idempotent_worker_setup():
    methods = dynamo_config.ignore_logger_methods
    methods.add(logging.Logger.info)
    for _ in range(2):
        warmup, warmed_with = worker_setup()
        warmup.assert_called_once()
        assert warmed_with == [{logging.Logger.info, logging.Logger.debug}]
        assert dynamo_config.ignore_logger_methods is methods
    assert methods == {logging.Logger.info, logging.Logger.debug}


@pytest.mark.parametrize("method", [logging.Logger.warning, logging.Logger.error])
def test_warning_and_error_calls_still_raise_under_fullgraph(method):
    worker_setup()
    assert method not in dynamo_config.ignore_logger_methods

    def op(x):
        method(OP_LOGGER, "visible diagnostic")
        return x + 1

    compiled = torch.compile(op, backend="eager", fullgraph=True)
    with pytest.raises(torch._dynamo.exc.Unsupported):
        compiled(torch.ones(2))


def test_fresh_worker_configuration_does_not_rely_on_parent_global_state():
    # A spawned worker starts with its own config set. Execute the production
    # per-worker block again after discarding the first worker's settings.
    worker_setup()
    with dynamo_config.patch(ignore_logger_methods=set()):
        assert not dynamo_config.ignore_logger_methods
        warmup, warmed_with = worker_setup()
        warmup.assert_called_once()
        assert warmed_with == [{logging.Logger.debug}]
