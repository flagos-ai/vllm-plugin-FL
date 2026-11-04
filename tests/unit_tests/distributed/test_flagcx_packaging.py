# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU coverage for source-checkout and wheel-packaged FlagCX discovery."""

import ctypes
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "vllm_fl/distributed/device_communicators/flagcx.py"


class Group:
    def __init__(self, world_size=2):
        self.rank = 0
        self.world_size = world_size


@pytest.fixture
def load_module(monkeypatch):
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.setitem(sys.modules, "_vllm_fl_flagcx_source_api", None)

    def package(name):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    for name in ("vllm", "vllm.distributed", "vllm.utils"):
        package(name)
    utils = ModuleType("vllm.distributed.utils")
    utils.StatelessProcessGroup = Group
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    logger = ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, logger.__name__, logger)
    torch_utils = ModuleType("vllm.utils.torch_utils")
    torch_utils.current_stream = Mock()
    monkeypatch.setitem(sys.modules, torch_utils.__name__, torch_utils)

    def wrapper(name):
        module = ModuleType(name)
        module.FLAGCXLibrary = Mock(side_effect=OSError("SDK load stopped"))
        module.buffer_type = ctypes.c_void_p
        module.flagcxComm_t = ctypes.c_void_p
        module.flagcxDataTypeEnum = object()
        module.flagcxUniqueId = object()
        module.flagcxRedOpTypeEnum = object()
        return module

    def load(*, legacy=True, wheel=True, incomplete_legacy=False):
        for name in ("plugin", "plugin.interservice", "flagcx"):
            package(name)
        legacy_module = wrapper("plugin.interservice.flagcx_wrapper")
        wheel_module = wrapper("flagcx.api")
        if incomplete_legacy:
            del legacy_module.flagcxComm_t
        monkeypatch.setitem(
            sys.modules, legacy_module.__name__, legacy_module if legacy else None
        )
        monkeypatch.setitem(
            sys.modules, wheel_module.__name__, wheel_module if wheel else None
        )
        # Keep optional imports deterministic even on a machine with FlagCX
        # installed. Execute the real module, not an AST copy of its logic.
        monkeypatch.setattr(sys, "path", list(sys.path))
        spec = importlib.util.spec_from_file_location("flagcx_packaging_test", MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return SimpleNamespace(module=module, legacy=legacy_module, wheel=wheel_module)

    return load


def test_source_wrapper_keeps_priority_over_installed_wheel(load_module):
    loaded = load_module()
    assert loaded.module._flagcx_available
    assert loaded.module.FLAGCXLibrary is loaded.legacy.FLAGCXLibrary
    assert loaded.module.flagcxUniqueId is loaded.legacy.flagcxUniqueId


def test_configured_new_source_api_has_priority_and_dataclass_module(
    load_module, monkeypatch, tmp_path
):
    source = tmp_path / "src/flagcx/api.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class FLAGCXLibrary:\n"
        "    origin: str = 'configured-source'\n"
        "buffer_type = flagcxComm_t = flagcxDataTypeEnum = object()\n"
        "flagcxUniqueId = flagcxRedOpTypeEnum = object()\n"
    )
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))

    loaded = load_module()

    assert loaded.module._flagcx_available
    assert loaded.module.FLAGCXLibrary().origin == "configured-source"
    assert loaded.module.FLAGCXLibrary is not loaded.legacy.FLAGCXLibrary
    assert loaded.module.FLAGCXLibrary is not loaded.wheel.FLAGCXLibrary
    assert (
        sys.modules["_vllm_fl_flagcx_source_api"].FLAGCXLibrary
        is loaded.module.FLAGCXLibrary
    )
    assert load_module().module.FLAGCXLibrary is loaded.module.FLAGCXLibrary


@pytest.mark.parametrize(
    "contents,error,message",
    [
        ("FLAGCXLibrary = object()\n", ImportError, "missing buffer_type"),
        ("raise RuntimeError('broken configured wrapper')\n", RuntimeError, "broken"),
    ],
)
def test_configured_source_failure_does_not_load_an_unmatched_wheel_wrapper(
    load_module, monkeypatch, tmp_path, contents, error, message
):
    source = tmp_path / "src/flagcx/api.py"
    source.parent.mkdir(parents=True)
    source.write_text(contents)
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))

    with pytest.raises(error, match=message):
        load_module()

    assert "_vllm_fl_flagcx_source_api" not in sys.modules


@pytest.mark.parametrize("legacy", [False, True])
def test_wheel_wrapper_loads_when_source_wrapper_is_absent_or_incomplete(
    load_module, legacy
):
    loaded = load_module(legacy=legacy, incomplete_legacy=legacy)
    assert loaded.module._flagcx_available
    for name in (
        "FLAGCXLibrary",
        "buffer_type",
        "flagcxComm_t",
        "flagcxDataTypeEnum",
        "flagcxUniqueId",
        "flagcxRedOpTypeEnum",
    ):
        assert getattr(loaded.module, name) is getattr(loaded.wheel, name)


def test_missing_wrappers_disable_communicator_with_explicit_error(load_module):
    loaded = load_module(legacy=False, wheel=False)
    assert not loaded.module._flagcx_available
    assert loaded.module.FLAGCXLibrary is None
    comm = loaded.module.PyFlagcxCommunicator(Group(), "txda:0", library_path="unused")
    assert comm.disabled and not comm.available
    assert isinstance(comm.init_error, RuntimeError)
    assert "Python bindings are not available" in str(comm.init_error)


@pytest.mark.parametrize("layout", ["build/lib", "lib"])
def test_configured_root_resolves_source_or_wheel_library(
    load_module, monkeypatch, tmp_path, layout
):
    library = tmp_path / layout / "libflagcx.so"
    library.parent.mkdir(parents=True)
    library.touch()
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    loaded = load_module(legacy=False)
    assert loaded.module._resolve_flagcx_library_path() == str(library)
    comm = loaded.module.PyFlagcxCommunicator(Group(), "txda:0")
    loaded.wheel.FLAGCXLibrary.assert_called_once_with(str(library))
    assert isinstance(comm.init_error, OSError)
    assert comm.disabled and not comm.available


def test_source_library_has_priority_and_load_failure_is_not_hidden(
    load_module, monkeypatch, tmp_path
):
    for layout in ("build/lib", "lib"):
        library = tmp_path / layout / "libflagcx.so"
        library.parent.mkdir(parents=True)
        library.touch()
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    loaded = load_module()
    expected = str(tmp_path / "build/lib/libflagcx.so")
    comm = loaded.module.PyFlagcxCommunicator(Group(), "txda:0")
    loaded.legacy.FLAGCXLibrary.assert_called_once_with(expected)
    loaded.wheel.FLAGCXLibrary.assert_not_called()
    assert isinstance(comm.init_error, OSError)


def test_installed_wheel_does_not_implicitly_enable_flagcx(load_module, monkeypatch):
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    loaded = load_module(legacy=False)
    assert loaded.module._flagcx_available
    comm = loaded.module.PyFlagcxCommunicator(Group(), "txda:0")
    assert comm.disabled and not comm.available
    assert isinstance(comm.init_error, RuntimeError)
    assert "FLAGCX_PATH must be set" in str(comm.init_error)
    loaded.wheel.FLAGCXLibrary.assert_not_called()


def test_missing_library_retains_a_clear_initialization_error(
    load_module, monkeypatch, tmp_path
):
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    loaded = load_module(legacy=False)
    comm = loaded.module.PyFlagcxCommunicator(Group(), "txda:0")
    assert comm.disabled and not comm.available
    assert isinstance(comm.init_error, FileNotFoundError)
    assert "lib/libflagcx.so" in str(comm.init_error)
    loaded.wheel.FLAGCXLibrary.assert_not_called()


@pytest.mark.parametrize("configured", [False, True])
def test_explicit_library_path_takes_priority_over_environment_discovery(
    load_module, monkeypatch, tmp_path, configured
):
    if configured:
        monkeypatch.setenv("FLAGCX_PATH", str(tmp_path / "missing-root"))
    else:
        monkeypatch.delenv("FLAGCX_PATH", raising=False)
    loaded = load_module(legacy=False)
    comm = loaded.module.PyFlagcxCommunicator(
        Group(), "txda:0", library_path="explicit-libflagcx.so"
    )
    loaded.wheel.FLAGCXLibrary.assert_called_once_with("explicit-libflagcx.so")
    assert isinstance(comm.init_error, OSError)


def test_single_device_does_not_require_a_wrapper_or_library(load_module, monkeypatch):
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    loaded = load_module(legacy=False, wheel=False)
    comm = loaded.module.PyFlagcxCommunicator(Group(world_size=1), "txda:0")
    assert comm.disabled and not comm.available
    assert comm.comm is None
