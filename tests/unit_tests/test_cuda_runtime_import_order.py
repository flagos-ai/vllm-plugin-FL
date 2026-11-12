# SPDX-License-Identifier: Apache-2.0
"""CUDA runtime binding happens before optional model/compiler imports."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def initialize(monkeypatch):
    # Execute the real bootstrap function in isolation: importing FlagGems
    # itself needs an accelerator, but this hook only controls host imports.
    source = Path(__file__).resolve().parents[2] / "vllm_fl/__init__.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_init_vendor_device"
    )
    importer = Mock()
    namespace = {
        "importlib": SimpleNamespace(import_module=importer),
        "logger": SimpleNamespace(debug=Mock()),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), source, "exec"), namespace
    )

    def run(vendor):
        monkeypatch.setitem(
            sys.modules,
            "vllm_fl.utils",
            SimpleNamespace(DeviceInfo=lambda: SimpleNamespace(vendor_name=vendor)),
        )
        namespace["_init_vendor_device"]()

    return run, importer


def test_nvidia_binds_flashinfer_before_later_compiler_probes(initialize):
    run, importer = initialize
    run("nvidia")
    importer.assert_called_once_with("flashinfer.comm")


@pytest.mark.parametrize("vendor", ["hygon", "metax", "ascend", "mthreads"])
def test_other_vendors_do_not_load_nvidia_communication(initialize, vendor):
    run, importer = initialize
    run(vendor)
    importer.assert_not_called()


@pytest.mark.parametrize(
    "error", [ImportError("not installed"), OSError("missing ABI")]
)
def test_missing_optional_communication_keeps_bootstrap_available(initialize, error):
    run, importer = initialize
    importer.side_effect = error
    run("nvidia")
    importer.assert_called_once_with("flashinfer.comm")


def test_unexpected_import_failure_propagates(initialize):
    run, importer = initialize
    error = RuntimeError("initialization failed")
    importer.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        run("nvidia")
    assert caught.value is error
