# SPDX-License-Identifier: Apache-2.0
"""Exercise the merged allocation method without initializing an accelerator."""

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("ascend", [False, True])
@pytest.mark.parametrize("explicit_context", [False, True])
def test_cache_allocation_preserves_backend_and_pool(ascend, explicit_context):
    path = Path(__file__).resolve().parents[3] / "vllm_fl/worker/model_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ModelRunnerFL"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "initialize_kv_cache_tensors"
    )
    events = []

    @contextmanager
    def pool():
        events.append("enter")
        yield
        events.append("exit")

    caches = {"layer": object()}

    def allocate(*args):
        if explicit_context:
            assert events == ["enter"]
        return caches

    standard = Mock(side_effect=allocate)
    reshape = Mock(side_effect=allocate)
    bind = Mock()
    namespace = {
        "nullcontext": nullcontext,
        "allocate_kv_cache": standard,
        "bind_kv_cache": bind,
        "map_kv_caches_to_kernel_blocks": Mock(return_value="mapped"),
        "logger": Mock(),
    }
    module = ast.Module(body=[method], type_ignores=[])
    exec(
        compile(
            module,
            str(path),
            "exec",
            flags=__import__("__future__").annotations.compiler_flag,
        ),
        namespace,
    )
    config = SimpleNamespace(kv_cache_groups=[])
    runner = SimpleNamespace(
        use_ascend_hybrid_cache=ascend,
        _allocate_kv_cache_tensors=Mock(return_value="raw"),
        _reshape_kv_cache_tensors=reshape,
        device="cpu",
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: "layout"),
        shared_kv_cache_layers={"shared": "layer"},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test")),
        attn_groups=[],
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
        kv_caches=[],
    )
    result = namespace[method.name](
        runner, config, [16], pool() if explicit_context else None
    )
    assert result["shared"] is result["layer"]
    assert events == (["enter", "exit"] if explicit_context else [])
    if ascend:
        standard.assert_not_called()
        runner._allocate_kv_cache_tensors.assert_called_once_with(config)
        reshape.assert_called_once_with("raw", [16])
    else:
        standard.assert_called_once_with(config, "cpu", "layout", [16])
        runner._allocate_kv_cache_tensors.assert_not_called()
        reshape.assert_not_called()
    bind.assert_called_once()
    assert bind.call_args.kwargs["layer_kv_caches"] == "mapped"
