# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU contracts for Sunrise image dependency preservation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "sunrise_runtime_constraints", Path(__file__).with_name("runtime_constraints.py")
)
constraints = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(constraints)


def distribution(name, version, *, files=(), direct_url=None):
    return SimpleNamespace(
        metadata={"Name": name},
        version=version,
        files=[Path(filename) for filename in files],
        read_text=lambda filename: json.dumps(direct_url or {}),
    )


@pytest.fixture
def native(monkeypatch):
    packages = [
        distribution("torch", "2.11.0+cpu"),
        distribution("torch_ptpu", "0.2.3+torch2.11"),
        distribution("flagtree", "0.7.0+sunrise3.6"),
        distribution("flag_gems", "5.5.0"),
        distribution("flagcx", "0.14.0+tangrt1.2.0"),
    ]
    monkeypatch.setattr(
        constraints.importlib.metadata, "distributions", lambda: packages
    )
    return packages


def test_capture_keeps_vendor_builds_and_excludes_pypi_triton(tmp_path, native):
    constraints.capture(tmp_path)
    overrides = (tmp_path / "native-overrides.txt").read_text()
    assert "torch==2.11.0+cpu\n" in overrides
    assert "torch-ptpu==0.2.3+torch2.11\n" in overrides
    assert "flag-gems==5.5.0\n" in overrides
    assert (tmp_path / "uv.toml").read_text() == 'exclude-dependencies = ["triton"]\n'


def test_capture_rejects_missing_vendor_plugin(tmp_path, native):
    native[:] = [
        package for package in native if package.metadata["Name"] != "torch_ptpu"
    ]
    with pytest.raises(RuntimeError, match="torch_ptpu"):
        constraints.capture(tmp_path)


@pytest.mark.parametrize("replacement", ["replace_torch", "add_cuda"])
def test_verify_rejects_native_stack_changes(tmp_path, native, replacement):
    constraints.capture(tmp_path)
    if replacement == "replace_torch":
        native[0] = distribution("torch", "2.13.0")
    else:
        native.append(distribution("nvidia-cuda-runtime-cu13", "13.0"))
    with pytest.raises(RuntimeError, match="changed Sunrise packages"):
        constraints.verify(tmp_path)


@pytest.mark.parametrize(
    ("version", "files", "direct_url", "error"),
    [
        ("0.32.0+empty", [], {}, None),
        ("0.32.0", [], {}, "Expected vLLM"),
        ("0.32.0+empty", [], {"dir_info": {"editable": True}}, "ordinary empty wheel"),
        (
            "0.32.0+empty",
            ["vllm/_C_stable_libtorch.abi3.so"],
            {},
            "native vLLM extensions",
        ),
    ],
)
def test_verify_checks_empty_wheel(
    tmp_path, native, monkeypatch, version, files, direct_url, error
):
    constraints.capture(tmp_path)
    vllm = distribution("vllm", version, files=files, direct_url=direct_url)
    monkeypatch.setattr(
        constraints.importlib.metadata, "distribution", lambda name: vllm
    )
    if error:
        with pytest.raises(RuntimeError, match=error):
            constraints.verify(tmp_path, "0.32.0")
    else:
        constraints.verify(tmp_path, "0.32.0")


def test_ambiguous_native_distributions_fail(native):
    native.append(distribution("torch-ptpu", "0.2.4+torch2.11"))
    with pytest.raises(RuntimeError, match="Multiple active versions"):
        constraints.native_versions()


def test_pinned_provider_addition_does_not_hide_flaggems_replacement(tmp_path, native):
    constraints.capture(tmp_path)
    native.append(distribution("flaggems_vllm", "0.1.0"))
    constraints.verify(tmp_path)
    assert "flaggems-vllm" not in constraints.native_versions()
    # The provider is independently pinned; FlagGems itself remains protected.
    native[:] = [
        distribution("flag_gems", "5.6.0")
        if package.metadata["Name"] == "flag_gems"
        else package
        for package in native
    ]
    with pytest.raises(RuntimeError, match="flag-gems"):
        constraints.verify(tmp_path)
