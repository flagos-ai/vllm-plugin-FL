# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_fl.compilation import compiler_interface


def _vllm_config(**ascend_options):
    return SimpleNamespace(
        additional_config={
            "ascend_compilation_config": ascend_options,
        },
        compute_hash=MagicMock(return_value="config-hash"),
    )


def test_compute_hash_saves_the_vllm_config():
    compiler = compiler_interface.AscendCompiler()
    config = _vllm_config(enable_npugraph_ex=True)

    assert compiler.compute_hash(config) == "config-hash"
    assert compiler._vllm_config is config
    config.compute_hash.assert_called_once_with()


def test_compile_requires_compute_hash_first():
    compiler = compiler_interface.AscendCompiler()

    with pytest.raises(RuntimeError, match="compute_hash"):
        compiler.compile(MagicMock(), [], {}, MagicMock())


def test_compile_requires_explicit_npugraph_ex_opt_in():
    compiler = compiler_interface.AscendCompiler()
    compiler.compute_hash(_vllm_config(enable_npugraph_ex=False))

    with pytest.raises(RuntimeError, match="enable_npugraph_ex"):
        compiler.compile(MagicMock(), [], {}, MagicMock())


def test_compile_rejects_static_kernel_in_initial_migration():
    compiler = compiler_interface.AscendCompiler()
    compiler.compute_hash(
        _vllm_config(enable_npugraph_ex=True, enable_static_kernel=True)
    )

    with pytest.raises(NotImplementedError, match="enable_static_kernel"):
        compiler.compile(MagicMock(), [], {}, MagicMock())


def test_compile_copies_graph_before_calling_backend(monkeypatch):
    compiler = compiler_interface.AscendCompiler()
    compiler.compute_hash(_vllm_config(enable_npugraph_ex=True))
    graph = MagicMock(name="original_graph")
    compiled = MagicMock(name="compiled")
    observed = {}

    def fake_compile(copied_graph, example_inputs):
        observed["graph"] = copied_graph
        observed["inputs"] = example_inputs
        return compiled, None

    monkeypatch.setattr(
        compiler_interface, "_compile_with_npugraph_ex", fake_compile
    )
    monkeypatch.setattr(torch._guards, "detect_fake_mode", lambda: None)

    result, handle = compiler.compile(graph, ["input"], {}, MagicMock())

    assert result is compiled
    assert handle is None
    assert observed["graph"] is not graph
    assert observed["inputs"] == ["input"]


def test_npugraph_ex_backend_uses_native_compatible_options(monkeypatch):
    compiler_config = SimpleNamespace()
    backend = MagicMock(name="npugraph_ex_backend")
    options_seen = {}
    fake_npu = SimpleNamespace(set_compile_mode=MagicMock())

    nge = SimpleNamespace(
        CompilerConfig=lambda: compiler_config,
        get_npu_backend=MagicMock(return_value=backend),
    )
    config_module = SimpleNamespace(
        _process_kwargs_options=lambda config, kwargs: options_seen.update(
            config=config, kwargs=kwargs
        )
    )

    def fake_import(name):
        if name == "npugraph_ex":
            return nge
        if name == "npugraph_ex.configs.compiler_config":
            return config_module
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(compiler_interface.importlib, "import_module", fake_import)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    assert compiler_interface._load_npugraph_ex_backend() is backend
    assert options_seen == {
        "config": compiler_config,
        "kwargs": {
            "options": {
                "force_eager": True,
                "inplace_pass": False,
            }
        },
    }
    fake_npu.set_compile_mode.assert_called_once_with(jit_compile=False)
    nge.get_npu_backend.assert_called_once_with(compiler_config=compiler_config)


def test_npugraph_ex_backend_supports_legacy_config_module(monkeypatch):
    compiler_config = SimpleNamespace()
    backend = MagicMock(name="npugraph_ex_backend")
    options_seen = {}
    fake_npu = SimpleNamespace(set_compile_mode=MagicMock())
    nge = SimpleNamespace(
        CompilerConfig=lambda: compiler_config,
        get_npu_backend=MagicMock(return_value=backend),
    )
    new_config_module = SimpleNamespace()
    legacy_config_module = SimpleNamespace(
        _process_kwargs_options=lambda config, kwargs: options_seen.update(
            config=config, kwargs=kwargs
        )
    )

    def fake_import(name):
        if name == "npugraph_ex":
            return nge
        if name == "npugraph_ex.configs.compiler_config":
            return new_config_module
        if name == "npugraph_ex.configs.npugraphex_config":
            return legacy_config_module
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(compiler_interface.importlib, "import_module", fake_import)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    assert compiler_interface._load_npugraph_ex_backend() is backend
    assert options_seen["config"] is compiler_config
    assert options_seen["kwargs"]["options"] == {
        "force_eager": True,
        "inplace_pass": False,
    }


def test_npugraph_ex_backend_falls_back_to_torchair(monkeypatch):
    aclgraph = SimpleNamespace(disable_reinplace_inplaceable_ops_pass=False)
    debug = SimpleNamespace(run_eagerly=False, aclgraph=aclgraph)
    compiler_config = SimpleNamespace(mode=None, debug=debug)
    backend = MagicMock(name="torchair_backend")
    fake_npu = SimpleNamespace(set_compile_mode=MagicMock())
    torchair = SimpleNamespace(
        CompilerConfig=lambda: compiler_config,
        get_npu_backend=MagicMock(return_value=backend),
    )

    def fake_import(name):
        if name.startswith("npugraph_ex"):
            raise ImportError("npugraph_ex is unavailable")
        if name == "torchair":
            return torchair
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(compiler_interface.importlib, "import_module", fake_import)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    assert compiler_interface._load_npugraph_ex_backend() is backend
    assert compiler_config.mode == "reduce-overhead"
    assert compiler_config.debug.run_eagerly is True
    assert (
        compiler_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass
        is True
    )
    fake_npu.set_compile_mode.assert_called_once_with(jit_compile=False)
    torchair.get_npu_backend.assert_called_once_with(
        compiler_config=compiler_config
    )


def test_missing_npugraph_ex_and_torchair_has_clear_error(monkeypatch):
    def fake_import(name):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(compiler_interface.importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError) as exc_info:
        compiler_interface._load_npugraph_ex_backend()

    message = str(exc_info.value)
    assert "npugraph_ex" in message
    assert "torchair" in message


def test_compile_wraps_non_tuple_graph(monkeypatch):
    backend = MagicMock(name="backend")
    wrapped = MagicMock(name="wrapped")
    graph = MagicMock(name="graph")
    inputs = [MagicMock(name="input")]

    monkeypatch.setattr(
        compiler_interface, "_load_npugraph_ex_backend", lambda: backend
    )
    monkeypatch.setattr(
        compiler_interface, "graph_returns_tuple", lambda candidate: False
    )
    make_tuple = MagicMock(return_value=wrapped)
    monkeypatch.setattr(compiler_interface, "make_graph_return_tuple", make_tuple)

    result, handle = compiler_interface._compile_with_npugraph_ex(graph, inputs)

    assert result is wrapped
    assert handle is None
    make_tuple.assert_called_once_with(graph, inputs, backend)
    backend.assert_not_called()
