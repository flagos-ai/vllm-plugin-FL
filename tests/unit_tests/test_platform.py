"""Focused tests for platform-level Ascend graph-mode policy."""

import ast
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


PLATFORM_PATH = Path(__file__).parents[2] / "vllm_fl" / "platform.py"


class CompilationMode(Enum):
    NONE = 0
    VLLM_COMPILE = 1


class CUDAGraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = 3

    def has_full_cudagraphs(self):
        return self in (self.FULL, self.FULL_DECODE_ONLY)


def _load_platform_policy(monkeypatch):
    module = ast.parse(PLATFORM_PATH.read_text())
    platform_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "PlatformFL"
    )
    method = next(
        node
        for node in platform_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "check_and_update_config"
    )
    method.decorator_list = []

    vllm_config = ModuleType("vllm.config")
    vllm_config.CompilationMode = CompilationMode
    vllm_config.CUDAGraphMode = CUDAGraphMode
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vllm_config)

    patch_module = ModuleType("vllm_fl.dispatch.backends.vendor.ascend.patch")
    patch_module.refresh_block_size = lambda config: None
    for name in (
        "vllm_fl",
        "vllm_fl.dispatch",
        "vllm_fl.dispatch.backends",
        "vllm_fl.dispatch.backends.vendor",
        "vllm_fl.dispatch.backends.vendor.ascend",
    ):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, patch_module.__name__, patch_module)

    namespace = {
        "logger": SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        )
    }
    exec(
        compile(
            ast.Module([method], type_ignores=[]),
            str(PLATFORM_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["check_and_update_config"]


def _graph_config(cudagraph_mode, *, enforce_eager=False):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(worker_cls=None, data_parallel_size=1),
        model_config=SimpleNamespace(
            use_mla=False,
            enforce_eager=enforce_eager,
        ),
        cache_config=None,
        scheduler_config=None,
        compilation_config=SimpleNamespace(
            compile_sizes=None,
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=cudagraph_mode,
            backend="inductor",
            use_inductor=True,
            splitting_ops=["vllm.unified_attention"],
            pass_config=SimpleNamespace(
                fuse_norm_quant=True,
                fuse_act_quant=True,
                fuse_attn_quant=True,
            ),
            cudagraph_num_of_warmups=0,
        ),
        attention_config=None,
    )


def test_npu_full_decode_uses_eager_graph_policy(monkeypatch):
    policy = _load_platform_policy(monkeypatch)
    config = _graph_config(CUDAGraphMode.FULL_DECODE_ONLY)

    policy(SimpleNamespace(device_type="npu", vendor_name="ascend"), config)

    compilation = config.compilation_config
    assert compilation.mode == CompilationMode.VLLM_COMPILE
    assert compilation.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    assert compilation.backend == "eager"
    assert compilation.use_inductor is False
    assert compilation.splitting_ops == []
    assert compilation.pass_config.fuse_norm_quant is False
    assert compilation.pass_config.fuse_act_quant is False
    assert compilation.pass_config.fuse_attn_quant is False
    assert compilation.cudagraph_num_of_warmups == 1


@pytest.mark.parametrize(
    ("cudagraph_mode", "enforce_eager"),
    [
        (CUDAGraphMode.NONE, False),
        (CUDAGraphMode.FULL_DECODE_ONLY, True),
        (CUDAGraphMode.FULL, False),
    ],
)
def test_npu_non_supported_graph_policy_falls_back_to_none(
    monkeypatch, cudagraph_mode, enforce_eager
):
    policy = _load_platform_policy(monkeypatch)
    config = _graph_config(cudagraph_mode, enforce_eager=enforce_eager)

    policy(SimpleNamespace(device_type="npu", vendor_name="ascend"), config)

    assert config.compilation_config.mode == CompilationMode.NONE
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize(
    ("device_type", "expected"),
    [("npu", "eager"), ("cuda", "inductor"), ("musa", "inductor")],
)
def test_simple_compile_backend_is_eager_only_on_npu(device_type, expected):
    module = ast.parse(PLATFORM_PATH.read_text())
    platform_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "PlatformFL"
    )
    assignment = next(
        node
        for node in platform_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "simple_compile_backend"
    )

    value = eval(
        compile(ast.Expression(assignment.value), str(PLATFORM_PATH), "eval"),
        {"device_type": device_type},
    )
    assert value == expected


def test_non_ascend_graph_capture_context_is_preserved():
    model_runner_path = (
        Path(__file__).parents[2] / "vllm_fl" / "worker" / "model_runner.py"
    )
    module = ast.parse(model_runner_path.read_text())
    graph_capture_if = next(
        node
        for node in module.body
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.FunctionDef) and child.name == "graph_capture"
            for child in node.body
        )
    )
    condition = compile(
        ast.Expression(graph_capture_if.test), str(model_runner_path), "eval"
    )

    def uses_local_capture(device_type, dist_backend="nccl"):
        platform = SimpleNamespace(
            device_type=device_type,
            dist_backend=dist_backend,
        )
        return eval(condition, {"current_platform": platform})

    assert uses_local_capture("npu", "hccl") is False
    assert uses_local_capture("cuda", "nccl") is False
    assert uses_local_capture("musa", "mccl") is True
    assert uses_local_capture("cuda", "flagcx") is True


def test_npu_pre_register_adds_ascend_quantization_choice(monkeypatch):
    module = ast.parse(PLATFORM_PATH.read_text())
    platform_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "PlatformFL"
    )
    method = next(
        node
        for node in platform_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "pre_register_and_update"
    )
    method.decorator_list = []

    for name in (
        "vllm_fl",
        "vllm_fl.dispatch",
        "vllm_fl.dispatch.backends",
        "vllm_fl.dispatch.backends.vendor",
        "vllm_fl.dispatch.backends.vendor.ascend",
        "vllm_fl.quantization",
    ):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    modelslim = ModuleType("vllm_fl.quantization.modelslim")
    modelslim.ASCEND_QUANTIZATION_METHOD = "ascend"
    modelslim.AscendModelSlimConfig = type("AscendModelSlimConfig", (), {})
    monkeypatch.setitem(sys.modules, modelslim.__name__, modelslim)

    namespace = {}
    exec(
        compile(
            ast.Module([method], type_ignores=[]),
            str(PLATFORM_PATH),
            "exec",
        ),
        namespace,
    )
    choices = ["compressed-tensors"]
    parser = SimpleNamespace(
        _option_string_actions={
            "--quantization": SimpleNamespace(choices=choices),
        }
    )

    namespace["pre_register_and_update"](
        SimpleNamespace(device_name="npu"),
        parser,
    )
    namespace["pre_register_and_update"](
        SimpleNamespace(device_name="npu"),
        parser,
    )

    assert choices == ["compressed-tensors", "ascend"]
