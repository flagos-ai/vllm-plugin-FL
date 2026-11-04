# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Contracts for early Sunrise native-V2 graph configuration defaults."""

import ast
import sys
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class CompileMode(IntEnum):
    NONE = 0
    STOCK_TORCH_COMPILE = 1
    DYNAMO_TRACE_ONCE = 2
    VLLM_COMPILE = 3


class GraphMode(IntEnum):
    NONE = 0
    FULL = 2
    PIECEWISE = 1
    FULL_DECODE_ONLY = 3
    FULL_AND_PIECEWISE = 4


@pytest.fixture
def configure(monkeypatch):
    # Load the real pure policy and its platform hook without initializing an
    # unrelated local accelerator or requiring a hardware vLLM wheel on the host.
    path = Path(__file__).resolve().parents[2] / "vllm_fl/platform.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helper = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_configure_sunrise_compilation_defaults"
    )
    platform = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PlatformFL"
    )
    hook = next(
        n
        for n in platform.body
        if isinstance(n, ast.FunctionDef) and n.name == "apply_config_platform_defaults"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[helper, hook], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    config = ModuleType("vllm.config")
    config.CompilationMode = CompileMode
    config.CUDAGraphMode = GraphMode
    envs = ModuleType("vllm.envs")
    envs.VLLM_USE_BREAKABLE_CUDAGRAPH = False
    monkeypatch.setitem(sys.modules, config.__name__, config)
    monkeypatch.setitem(sys.modules, envs.__name__, envs)
    hook_class = type(
        "TestPlatform", (), {"apply_config_platform_defaults": namespace[hook.name]}
    )
    return SimpleNamespace(apply=namespace[helper.name], envs=envs, platform=hook_class)


def make_config(
    *, mode=None, graph=None, optimization_level=2, enforce_eager=False, backend="eager"
):
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        optimization_level=optimization_level,
        parallel_config=SimpleNamespace(use_ubatching=False),
        compilation_config=SimpleNamespace(
            mode=mode,
            backend=backend,
            cudagraph_mode=graph,
            cudagraph_capture_sizes=[1, 2],
            max_cudagraph_capture_size=2,
            cudagraph_mm_encoder=False,
            use_inductor_graph_partition=False,
        ),
    )


def test_bare_sunrise_defaults_to_native_v2_decode_graphs(configure):
    config = make_config()
    configure.platform.device_type = "ptpu"
    configure.platform.vendor_name = "sunrise"
    configure.platform.apply_config_platform_defaults(config)
    assert config.compilation_config.mode == CompileMode.NONE
    assert config.compilation_config.backend == "eager"
    assert config.compilation_config.cudagraph_mode == GraphMode.FULL_DECODE_ONLY
    assert config.compilation_config.cudagraph_capture_sizes == [1, 2]
    assert config.compilation_config.max_cudagraph_capture_size == 2


@pytest.mark.parametrize("optimization_level", [0, 1, 2, 3])
def test_optimization_level_zero_keeps_graphs_disabled(configure, optimization_level):
    config = make_config(optimization_level=optimization_level)
    configure.apply(config)
    expected = GraphMode.NONE if optimization_level == 0 else GraphMode.FULL_DECODE_ONLY
    assert config.compilation_config.cudagraph_mode == expected
    assert config.compilation_config.mode == CompileMode.NONE


def test_enforce_eager_remains_authoritative(configure):
    config = make_config(
        enforce_eager=True, mode=CompileMode.VLLM_COMPILE, graph=GraphMode.FULL
    )
    configure.apply(config)
    assert config.compilation_config.mode == CompileMode.NONE
    assert config.compilation_config.cudagraph_mode == GraphMode.NONE


@pytest.mark.parametrize("graph", [GraphMode.NONE, GraphMode.FULL_DECODE_ONLY])
def test_explicit_native_execution_modes_and_capture_sizes_are_preserved(
    configure, graph
):
    config = make_config(mode=CompileMode.NONE, graph=graph)
    configure.apply(config)
    configure.apply(config)
    assert config.compilation_config.mode == CompileMode.NONE
    assert config.compilation_config.cudagraph_mode == graph
    assert config.compilation_config.cudagraph_capture_sizes == [1, 2]


@pytest.mark.parametrize(
    "mode",
    [
        CompileMode.STOCK_TORCH_COMPILE,
        CompileMode.DYNAMO_TRACE_ONCE,
        CompileMode.VLLM_COMPILE,
    ],
)
def test_unsupported_explicit_model_compilation_fails(configure, mode):
    config = make_config(mode=mode)
    with pytest.raises(NotImplementedError, match="compilation mode NONE"):
        configure.apply(config)


@pytest.mark.parametrize("backend", ["inductor", "custom.backend"])
def test_explicit_backend_request_is_not_silently_disabled(configure, backend):
    config = make_config(backend=backend)
    with pytest.raises(NotImplementedError, match="backend is unsupported"):
        configure.apply(config)


@pytest.mark.parametrize(
    "graph", [GraphMode.FULL, GraphMode.PIECEWISE, GraphMode.FULL_AND_PIECEWISE]
)
def test_unsupported_graph_modes_fail_instead_of_downgrading(configure, graph):
    config = make_config(graph=graph)
    with pytest.raises(NotImplementedError, match="NONE or FULL_DECODE_ONLY"):
        configure.apply(config)
    assert config.compilation_config.cudagraph_mode == graph


@pytest.mark.parametrize("feature", ["breakable", "ubatching", "encoder", "partition"])
def test_unsupported_graph_features_fail_explicitly(configure, feature):
    config = make_config(graph=GraphMode.FULL_DECODE_ONLY)
    if feature == "breakable":
        configure.envs.VLLM_USE_BREAKABLE_CUDAGRAPH = True
    elif feature == "ubatching":
        config.parallel_config.use_ubatching = True
    elif feature == "encoder":
        config.compilation_config.cudagraph_mm_encoder = True
    else:
        config.compilation_config.use_inductor_graph_partition = True
    with pytest.raises(NotImplementedError):
        configure.apply(config)


@pytest.mark.parametrize(
    "vendor,device_type",
    [
        ("nvidia", "cuda"),
        ("ascend", "npu"),
        ("mthreads", "musa"),
        ("tsingmicro", "txda"),
    ],
)
def test_other_vendor_platform_defaults_are_untouched(configure, vendor, device_type):
    configure.platform.vendor_name = vendor
    configure.platform.device_type = device_type
    config = make_config(
        mode=CompileMode.VLLM_COMPILE, graph=GraphMode.FULL_AND_PIECEWISE
    )
    configure.platform.apply_config_platform_defaults(config)
    assert config.compilation_config.mode == CompileMode.VLLM_COMPILE
    assert config.compilation_config.cudagraph_mode == GraphMode.FULL_AND_PIECEWISE
