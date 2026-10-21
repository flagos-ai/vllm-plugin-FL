# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU loader checks; native ABI acceptance requires installed MUSA CI headers."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "vllm_fl/dispatch/backends/vendor/musa/compilation/musa_graph_exec.py"
)


@pytest.fixture
def loader(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("musa_graph_exec_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = tmp_path / "torch_musa"
    sdk = tmp_path / "sdk"
    root = tmp_path / "plugin"
    for file in (
        root / "musa_graph_exec.cpp",
        package / "csrc/aten/musa/MUSAGraph.h",
        package / "lib/libmusa_python.so",
        sdk / "include/musa_runtime_api.h",
        sdk / "lib/libmusart.so",
    ):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("mock prerequisite", encoding="utf-8")
    (package / "share/generated_cuda_compatible/include").mkdir(parents=True)
    module.__file__ = str(root / "musa_graph_exec.py")
    module.sys = NS(platform="linux")
    monkeypatch.setenv("MUSA_HOME", str(sdk))
    monkeypatch.setattr(module.shutil, "which", lambda _: "/bin/c++")
    native = NS(free_exec=Mock(), inst_exec=Mock(), has_exec=Mock())
    cpp = NS(
        load=Mock(return_value=native),
        get_cxx_compiler=lambda: "c++",
        verify_ninja_availability=Mock(),
    )
    torch = ModuleType("torch")
    torch.__version__ = "2.9.0+cpu"
    utils = ModuleType("torch.utils")
    utils.cpp_extension = cpp
    torch.utils = utils
    musa = ModuleType("torch_musa")
    musa.__file__ = str(package / "__init__.py")
    musa.__version__ = "2.9.0+git123"
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = NS(device_type="musa")
    for name, value in {
        "torch": torch,
        "torch.utils": utils,
        "torch.utils.cpp_extension": cpp,
        "torch_musa": musa,
        "vllm.platforms": platforms,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    return NS(
        module=module,
        cpp=cpp,
        native=native,
        torch=torch,
        musa=musa,
        platform=platforms.current_platform,
        package=package,
        sdk=sdk,
        root=root,
    )


def test_import_is_lazy_and_non_musa_never_builds(loader):
    assert loader.module._OPS is None
    assert not loader.cpp.load.called
    loader.platform.device_type = "cuda"
    with pytest.raises(RuntimeError, match="only available on MUSA"):
        loader.module.load_musa_graph_exec_ops()
    assert not loader.cpp.load.called


def test_host_only_build_paths_and_success_cache(loader):
    first = loader.module.load_musa_graph_exec_ops()
    assert first is loader.native
    assert loader.module.load_musa_graph_exec_ops() is first
    loader.cpp.load.assert_called_once()
    args = loader.cpp.load.call_args.kwargs
    assert args["with_cuda"] is False
    assert args["sources"] == [str(loader.root / "musa_graph_exec.cpp")]
    assert str(loader.package.parent) in args["extra_include_paths"]
    assert str(loader.package / "lib/libmusa_python.so") in args["extra_ldflags"]
    assert str(loader.sdk / "lib/libmusart.so") in args["extra_ldflags"]
    assert "build_directory" not in args  # respect PyTorch's configured cache


def test_build_failure_is_not_swallowed_or_cached(loader):
    failure = RuntimeError("actual compiler failure")
    loader.cpp.load.side_effect = [failure, loader.native]
    with pytest.raises(RuntimeError) as caught:
        loader.module.load_musa_graph_exec_ops()
    assert caught.value is failure and loader.module._OPS is None
    assert loader.module.load_musa_graph_exec_ops() is loader.native


@pytest.mark.parametrize("name", ["free_exec", "inst_exec", "has_exec"])
def test_missing_native_entry_is_not_cached(loader, name):
    delattr(loader.native, name)
    with pytest.raises(RuntimeError, match="missing " + name):
        loader.module.load_musa_graph_exec_ops()
    assert loader.module._OPS is None


@pytest.mark.parametrize("name", ["torch", "musa"])
@pytest.mark.parametrize("version", ["", "2.8.0", "2.10.0"])
def test_unknown_header_version_fails_before_build(loader, name, version):
    getattr(loader, name).__version__ = version
    with pytest.raises(RuntimeError, match="requires .* 2.9"):
        loader.module.load_musa_graph_exec_ops()
    assert not loader.cpp.load.called


@pytest.mark.parametrize(
    "base,relative",
    [
        ("root", "musa_graph_exec.cpp"),
        ("package", "csrc/aten/musa/MUSAGraph.h"),
        ("package", "lib/libmusa_python.so"),
        ("sdk", "include/musa_runtime_api.h"),
        ("sdk", "lib/libmusart.so"),
    ],
)
def test_missing_build_prerequisite_fails_closed(loader, base, relative):
    (getattr(loader, base) / relative).unlink()
    with pytest.raises(RuntimeError, match="requires"):
        loader.module.load_musa_graph_exec_ops()
    assert not loader.cpp.load.called


def test_missing_generated_headers_fails_before_build(loader):
    (loader.package / "share/generated_cuda_compatible/include").rmdir()
    with pytest.raises(RuntimeError, match="requires headers"):
        loader.module.load_musa_graph_exec_ops()
    assert not loader.cpp.load.called


def test_missing_compiler_fails_before_build(loader, monkeypatch):
    monkeypatch.setattr(loader.module.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match=r"requires C\+\+ compiler"):
        loader.module.load_musa_graph_exec_ops()
    assert not loader.cpp.load.called


def test_missing_ninja_preserves_original_failure(loader):
    failure = RuntimeError("ninja unavailable")
    loader.cpp.verify_ninja_availability.side_effect = failure
    with pytest.raises(RuntimeError) as caught:
        loader.module.load_musa_graph_exec_ops()
    assert caught.value is failure and not loader.cpp.load.called


def test_sdk_lib64_layout_is_supported(loader):
    source = loader.sdk / "lib/libmusart.so"
    target = loader.sdk / "lib64/libmusart.so"
    target.parent.mkdir()
    source.rename(target)
    loader.module.load_musa_graph_exec_ops()
    assert str(target) in loader.cpp.load.call_args.kwargs["extra_ldflags"]
