# Copyright (c) 2025 BAAI. All rights reserved.

"""CPU behavior tests for WorkerFL's allocator selection.

Only the real method is compiled, so these tests run without importing vLLM or
initializing hardware. Full Worker import tests remain in test_worker.py.
"""

import ast
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture(scope="module")
def memory_pool_code():
    source_path = Path(__file__).resolve().parents[3] / "vllm_fl/worker/worker.py"
    module = ast.parse(
        source_path.read_text(encoding="utf-8"), filename=str(source_path)
    )
    (worker,) = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerFL"
    ]
    (method,) = [
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_maybe_get_memory_pool_context"
    ]
    assert not method.decorator_list, "Include method decorators in the CPU harness"
    return compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec")


@pytest.fixture
def memory_pool_worker(memory_pool_code):
    def make_worker(platform, *, enable_cumem_allocator, enable_sleep_mode):
        allocator = Mock()
        allocator.get_current_usage.return_value = 0
        allocator.use_memory_pool.return_value = nullcontext("allocator")
        get_allocator = Mock(return_value=allocator)
        namespace = {
            "AbstractContextManager": AbstractContextManager,
            "nullcontext": nullcontext,
            "current_platform": SimpleNamespace(
                is_cuda_alike=lambda: platform == "cuda",
                is_xpu=lambda: platform == "xpu",
                is_cpu=lambda: platform == "cpu",
            ),
            "get_mem_allocator_instance": get_allocator,
        }
        exec(memory_pool_code, namespace)
        worker = SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(
                    enable_cumem_allocator=enable_cumem_allocator,
                    enable_sleep_mode=enable_sleep_mode,
                )
            )
        )

        def memory_pool(tag):
            return namespace["_maybe_get_memory_pool_context"](worker, tag)

        return memory_pool, get_allocator, allocator

    return make_worker


@pytest.mark.parametrize(
    ("platform", "enable_cumem_allocator", "enable_sleep_mode"),
    [
        ("hygon", False, False),
        ("hygon", True, False),
        ("cuda", False, False),
        ("cuda", False, True),
        ("xpu", False, False),
        ("xpu", True, False),
        ("cpu", False, False),
        ("cpu", True, True),
    ],
)
@pytest.mark.parametrize("tag", ["weights", "kv_cache"])
def test_worker_uses_ordinary_memory_without_allocator(
    memory_pool_worker, platform, enable_cumem_allocator, enable_sleep_mode, tag
):
    memory_pool, get_allocator, allocator = memory_pool_worker(
        platform,
        enable_cumem_allocator=enable_cumem_allocator,
        enable_sleep_mode=enable_sleep_mode,
    )
    get_allocator.side_effect = AssertionError("Ordinary allocation reached allocator")

    with memory_pool(tag) as context:
        assert context is None

    get_allocator.assert_not_called()
    allocator.use_memory_pool.assert_not_called()


@pytest.mark.parametrize(
    ("platform", "enable_cumem_allocator", "enable_sleep_mode"),
    [
        ("cuda", True, False),
        ("cuda", True, True),
        ("xpu", False, True),
        ("xpu", True, True),
    ],
)
@pytest.mark.parametrize("tag", ["weights", "kv_cache"])
def test_worker_keeps_supported_allocator_selection(
    memory_pool_worker, platform, enable_cumem_allocator, enable_sleep_mode, tag
):
    memory_pool, get_allocator, allocator = memory_pool_worker(
        platform,
        enable_cumem_allocator=enable_cumem_allocator,
        enable_sleep_mode=enable_sleep_mode,
    )

    with memory_pool(tag) as context:
        assert context == "allocator"

    get_allocator.assert_called_once_with()
    allocator.use_memory_pool.assert_called_once_with(tag=tag)
    if tag == "weights":
        allocator.get_current_usage.assert_called_once_with()
    else:
        allocator.get_current_usage.assert_not_called()


@pytest.mark.parametrize("enable_cumem_allocator", [False, True])
@pytest.mark.parametrize("tag", ["weights", "kv_cache"])
def test_hygon_explicit_sleep_preserves_unsupported_allocator_error(
    memory_pool_worker, enable_cumem_allocator, tag
):
    memory_pool, get_allocator, allocator = memory_pool_worker(
        "hygon",
        enable_cumem_allocator=enable_cumem_allocator,
        enable_sleep_mode=True,
    )
    error = RuntimeError("Sleep mode allocator is not available")
    get_allocator.side_effect = error

    with pytest.raises(RuntimeError) as raised:
        memory_pool(tag)

    assert raised.value is error
    get_allocator.assert_called_once_with()
    allocator.use_memory_pool.assert_not_called()


@pytest.mark.parametrize("tag", ["weights", "kv_cache"])
def test_allocator_usage_guard_applies_only_to_weights(memory_pool_worker, tag):
    memory_pool, get_allocator, allocator = memory_pool_worker(
        "cuda", enable_cumem_allocator=True, enable_sleep_mode=False
    )
    allocator.get_current_usage.return_value = 1

    if tag == "weights":
        with pytest.raises(AssertionError, match="one instance per process"):
            memory_pool(tag)
        allocator.get_current_usage.assert_called_once_with()
        allocator.use_memory_pool.assert_not_called()
    else:
        with memory_pool(tag) as context:
            assert context == "allocator"
        allocator.get_current_usage.assert_not_called()
        allocator.use_memory_pool.assert_called_once_with(tag=tag)
    get_allocator.assert_called_once_with()


@pytest.mark.parametrize("stage", ["factory", "pool", "enter"])
@pytest.mark.parametrize("tag", ["weights", "kv_cache"])
def test_allocator_errors_propagate(memory_pool_worker, stage, tag):
    memory_pool, get_allocator, allocator = memory_pool_worker(
        "cuda", enable_cumem_allocator=True, enable_sleep_mode=False
    )
    error = RuntimeError(f"Allocator failure in {stage}")
    if stage == "factory":
        get_allocator.side_effect = error
    elif stage == "pool":
        allocator.use_memory_pool.side_effect = error
    else:
        context = MagicMock(spec=AbstractContextManager)
        context.__enter__.side_effect = error
        allocator.use_memory_pool.return_value = context

    with pytest.raises(RuntimeError) as raised, memory_pool(tag):
        pass

    assert raised.value is error
    get_allocator.assert_called_once_with()
    if stage == "factory":
        allocator.use_memory_pool.assert_not_called()
    else:
        allocator.use_memory_pool.assert_called_once_with(tag=tag)
