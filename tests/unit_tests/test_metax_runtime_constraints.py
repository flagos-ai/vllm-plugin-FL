# Copyright (c) 2026 BAAI. All rights reserved.

"""Dependency updates retain vendor binaries and install an ordinary empty wheel."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_helper():
    path = Path(__file__).resolve().parents[2] / "docker/metax/runtime_constraints.py"
    spec = importlib.util.spec_from_file_location(
        "metax_runtime_constraints_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_normalizes_vendor_names_and_keeps_local_versions(
    monkeypatch, tmp_path
):
    helper = _load_helper()
    distributions = [
        SimpleNamespace(metadata={"Name": name}, version=version)
        for name, version in (
            ("Torch", "2.10.0+metax3.8.1.0"),
            ("Triton", "3.6.0+metax"),
            ("Flag_Gems", "5.5.0.dev0"),
            ("flash_attn", "2.8.3+metax"),
            ("numpy", "1.26.4"),
            ("xgrammar", "0.2.3+flagos"),
            ("vllm", "0.24.0+flagos"),
        )
    ]
    monkeypatch.setattr(
        helper.importlib.metadata, "distributions", lambda: distributions
    )
    helper.capture(tmp_path)
    overrides = (tmp_path / "native-overrides.txt").read_text()
    assert "torch==2.10.0+metax3.8.1.0\n" in overrides
    assert "flag-gems==5.5.0.dev0\n" in overrides
    assert "flash-attn==2.8.3+metax\n" in overrides
    assert "numpy==1.26.4\n" in overrides
    assert "xgrammar==0.2.3+flagos\n" in overrides
    assert "vllm==" not in overrides
    # mcoplib is optional: the FlagOS base can supply only FlagGems kernels.
    assert "mcoplib" not in overrides
    helper.verify(tmp_path, None)


def test_verify_rejects_replacement_with_stock_torch(monkeypatch, tmp_path):
    helper = _load_helper()
    (tmp_path / "native-stack.json").write_text(
        json.dumps({"torch": "2.10.0+metax3.8.1.0"})
    )
    monkeypatch.setattr(helper, "native_versions", lambda: {"torch": "2.10.0"})
    with pytest.raises(RuntimeError, match="changed MACA packages"):
        helper.verify(tmp_path, None)


def test_flagtree_excludes_stock_triton_and_detects_namespace_replacement(
    monkeypatch, tmp_path
):
    helper = _load_helper()
    native = {"torch": "2.10.0+metax", "flagtree": "0.7.0+metax3.6"}
    monkeypatch.setattr(helper, "native_versions", lambda: native)
    helper.capture(tmp_path)
    assert (tmp_path / "uv.toml").read_text() == 'exclude-dependencies = ["triton"]\n'
    monkeypatch.setattr(
        helper, "native_versions", lambda: {**native, "triton": "3.6.0"}
    )
    with pytest.raises(RuntimeError, match="triton"):
        helper.verify(tmp_path, None)


@pytest.mark.parametrize(
    "version,editable,extension,error",
    [
        ("0.32.0+empty", False, None, None),
        ("0.32.0+empty", True, None, "ordinary wheel"),
        ("0.32.0+empty", False, "vllm/_C.abi3.so", "CUDA extensions"),
        ("0.32.0", False, None, "Expected the vLLM empty wheel"),
    ],
)
def test_verify_checks_empty_wheel_installation(
    monkeypatch, tmp_path, version, editable, extension, error
):
    helper = _load_helper()
    (tmp_path / "native-stack.json").write_text("{}")
    monkeypatch.setattr(helper, "native_versions", dict)
    distribution = SimpleNamespace(
        version=version,
        files=[Path(extension)] if extension else [],
        read_text=lambda name: json.dumps({"dir_info": {"editable": editable}}),
    )
    monkeypatch.setattr(
        helper.importlib.metadata, "distribution", lambda name: distribution
    )
    if error is None:
        helper.verify(tmp_path, "0.32.0")
    else:
        with pytest.raises(RuntimeError, match=error):
            helper.verify(tmp_path, "0.32.0")
