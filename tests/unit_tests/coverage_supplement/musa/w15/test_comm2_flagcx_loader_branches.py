# Copyright (c) 2025 BAAI. All rights reserved.
"""Residual loader / import-time branches of the FlagCX communicator module.

Each case re-executes the real source file under a private module name after
replacing the vLLM helper imports, so the environment-dependent top-level
branches are exercised without a FlagCX wheel.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_fl.distributed.device_communicators.flagcx as fx


SOURCE = Path(fx.__file__)
WHEEL_API = """\
from dataclasses import dataclass

@dataclass
class flagcxUniqueId:
    internal: bytes = b""

class FLAGCXLibrary:
    pass

buffer_type = flagcxComm_t = flagcxDataTypeEnum = flagcxRedOpTypeEnum = object()
"""


def _api_root(tmp_path, *, with_lib=False):
    root = tmp_path / "flagcx"
    (root / "src" / "flagcx").mkdir(parents=True)
    (root / "src" / "flagcx" / "api.py").write_text(WHEEL_API, encoding="utf-8")
    if with_lib:
        (root / "lib").mkdir()
        (root / "lib" / "libflagcx.so").touch()
        (root / "api.py").write_text(WHEEL_API, encoding="utf-8")
        (root / "__init__.py").write_text(
            'raise AssertionError("wheel backend must not be imported")\n',
            encoding="utf-8",
        )
    return root


def _load_fresh(monkeypatch, tag):
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.utils",
        SimpleNamespace(StatelessProcessGroup=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.logger",
        SimpleNamespace(init_logger=lambda name: logging.getLogger(name)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.torch_utils",
        SimpleNamespace(current_stream=Mock()),
    )
    monkeypatch.setattr(sys, "path", sys.path.copy())
    spec = importlib.util.spec_from_file_location(f"_w15_flagcx_{tag}", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# loader guards
# ---------------------------------------------------------------------------
def test_wheel_loader_without_spec_raises(monkeypatch):
    monkeypatch.setattr(
        fx.importlib.util, "spec_from_file_location", lambda *a, **k: None
    )
    with pytest.raises(ImportError, match="Cannot load FlagCX API"):
        fx._load_flagcx_wheel_bindings(Path("api.py"))


def test_amd_loader_without_spec_raises(monkeypatch):
    monkeypatch.setattr(
        fx.importlib.util, "spec_from_file_location", lambda *a, **k: None
    )
    with pytest.raises(ImportError, match="Cannot load FlagCX API"):
        fx._load_amd_flagcx_api(Path("api.py"))


def test_wheel_loader_restores_previous_module_on_failure(monkeypatch, tmp_path):
    path = tmp_path / "api.py"
    path.write_text("raise RuntimeError('broken')\n", encoding="utf-8")
    name = f"{fx.__name__}._wheel_api"
    previous = SimpleNamespace(marker=True)
    monkeypatch.setitem(sys.modules, name, previous)

    with pytest.raises(RuntimeError):
        fx._load_flagcx_wheel_bindings(path)

    assert sys.modules[name] is previous


def test_wheel_loader_replaces_previous_module_on_success(monkeypatch, tmp_path):
    path = tmp_path / "api.py"
    path.write_text(WHEEL_API, encoding="utf-8")
    name = f"{fx.__name__}._wheel_api"
    previous = SimpleNamespace(marker=True)
    monkeypatch.setitem(sys.modules, name, previous)

    loaded = fx._load_flagcx_wheel_bindings(path)

    assert len(loaded) == 6
    assert sys.modules[name] is not previous


# ---------------------------------------------------------------------------
# import-time layout selection
# ---------------------------------------------------------------------------
def test_amd_source_layout(monkeypatch, tmp_path):
    root = _api_root(tmp_path)
    monkeypatch.setenv("FLAGCX_PATH", str(root))
    monkeypatch.setattr(torch.version, "hip", "6.4", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(current_platform=SimpleNamespace(vendor_name="amd")),
    )

    module = _load_fresh(monkeypatch, "amd")

    assert module._flagcx_amd_source is True
    assert module._flagcx_available is True
    assert module.FLAGCXLibrary is not None
    assert module._flagcx_wheel_library is None


def test_wheel_layout_binds_paired_library(monkeypatch, tmp_path):
    root = _api_root(tmp_path, with_lib=True)
    monkeypatch.setenv("FLAGCX_PATH", str(root))
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    module = _load_fresh(monkeypatch, "wheel")

    assert module._flagcx_available is True
    assert module.FLAGCXLibrary is not None
    assert module._flagcx_wheel_library == str((root / "lib" / "libflagcx.so").absolute())
    assert str(root) not in sys.path


def test_source_layout_appends_search_path(monkeypatch, tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setenv("FLAGCX_PATH", str(root))
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "plugin.interservice.flagcx_wrapper",
        SimpleNamespace(
            FLAGCXLibrary=object,
            buffer_type=object,
            flagcxComm_t=object,
            flagcxDataTypeEnum=object,
            flagcxUniqueId=object,
            flagcxRedOpTypeEnum=object,
        ),
    )

    module = _load_fresh(monkeypatch, "source")

    assert module._flagcx_available is True
    assert str(root) in sys.path


def test_missing_source_dir_skips_search_path(monkeypatch, tmp_path):
    absent = tmp_path / "absent"
    monkeypatch.setenv("FLAGCX_PATH", str(absent))
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "plugin.interservice.flagcx_wrapper",
        SimpleNamespace(
            FLAGCXLibrary=object,
            buffer_type=object,
            flagcxComm_t=object,
            flagcxDataTypeEnum=object,
            flagcxUniqueId=object,
            flagcxRedOpTypeEnum=object,
        ),
    )

    module = _load_fresh(monkeypatch, "absent")

    assert module._flagcx_available is True
    assert str(absent) not in sys.path


def test_missing_wrapper_marks_bindings_unavailable(monkeypatch):
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setitem(sys.modules, "plugin.interservice.flagcx_wrapper", None)

    module = _load_fresh(monkeypatch, "missing")

    assert module._flagcx_available is False
    assert module.FLAGCXLibrary is None
    assert isinstance(module._flagcx_import_error, ImportError)


def test_non_import_error_from_wrapper_is_reraised(monkeypatch):
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    class BrokenWrapper:
        def __getattr__(self, name):
            raise OSError(f"native symbol {name} unavailable")

    monkeypatch.setitem(
        sys.modules, "plugin.interservice.flagcx_wrapper", BrokenWrapper()
    )

    with pytest.raises(OSError, match="native symbol"):
        _load_fresh(monkeypatch, "oserror")
