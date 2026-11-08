"""Version-specific runner routing without importing a device worker."""

import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("vendor", ["iluvatar", "nvidia", "ascend", "kunlunxin"])
def test_v1_runner_shared_entry(monkeypatch, vendor):
    # Execute the actual construction branch; neither runner is constructed on GPU.
    tree = ast.parse((ROOT / "vllm_fl/worker/worker.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WorkerFL"
    )
    init = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "init_device"
    )
    branch = next(
        n
        for n in init.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "self.use_v2_model_runner"
    )
    calls = []
    for name, symbol in [("model_runner", "ModelRunnerFL")]:
        module = ModuleType("vllm_fl.worker." + name)

        def constructor(config, device, selected=name):
            calls.append((selected, config, device))
            return selected

        setattr(module, symbol, constructor)
        monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    worker = SimpleNamespace(vllm_config=object(), device=object())
    module = ast.fix_missing_locations(ast.Module(body=branch.orelse, type_ignores=[]))
    exec(
        compile(module, "<worker v1 runner construction>", "exec"),
        {"self": worker, "current_platform": SimpleNamespace(vendor_name=vendor)},
    )
    expected = "model_runner"
    assert worker.model_runner == expected
    assert calls == [(expected, worker.vllm_config, worker.device)]


def test_target_runner_import():
    from vllm_fl.worker.model_runner import ModelRunnerFL

    assert callable(ModelRunnerFL.execute_model)
    assert callable(ModelRunnerFL.sample_tokens)


@pytest.mark.parametrize("configured", [False, True])
def test_runner_device_module_selection(configured):
    import torch

    tree = ast.parse((ROOT / "vllm_fl/worker/model_runner.py").read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_device_module"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "device-module", "exec"),
        namespace,
    )
    override = object()
    platform = SimpleNamespace(device_type="cuda")
    if configured:
        platform.torch_device_fn = override
    assert namespace["_get_device_module"](platform) is (
        override if configured else torch.cuda
    )


def test_native_cuda_runner_import_in_fresh_process():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import torch",
                    "import vllm.platforms",
                    "from vllm.platforms.interface import Platform",
                    "platform = Platform()",
                    "platform.device_type = 'cuda'",
                    "assert platform.torch_device_fn is None",
                    "vllm.platforms.current_platform = platform",
                    "from vllm_fl.worker import model_runner",
                    "assert model_runner.device_module is torch.cuda",
                    "assert callable(model_runner.ModelRunnerFL.execute_model)",
                    "assert model_runner.AsyncGPUModelRunnerOutput.__init__.__annotations__['async_output_copy_stream'] is torch.cuda.Stream",
                ]
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
