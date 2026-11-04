# Copyright (c) 2026 BAAI. All rights reserved.
"""Exercise KV initialization with the vLLM 0.32 V2 runner interface."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/worker/worker.py"


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("legacy_capture", [None, False, True])
@pytest.mark.parametrize("zeroing", [False, True])
def test_kv_initialization_does_not_require_legacy_capture_on_v2(
    use_v2, legacy_capture, zeroing
):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    worker = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WorkerFL"
    )
    method = next(
        n
        for n in worker.body
        if isinstance(n, ast.FunctionDef) and n.name == "initialize_from_config"
    )
    method.decorator_list = []
    events = []
    connector = Mock(side_effect=lambda *args: events.append("connector"))
    namespace = {"KVCacheConfig": object, "ensure_kv_transfer_initialized": connector}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    runner = SimpleNamespace(
        initialize_kv_cache=Mock(side_effect=lambda cfg: events.append("initialize")),
        _init_kv_zero_meta=Mock(side_effect=lambda: events.append("zero")),
    )
    # The real V2 runner has no legacy capturer method.
    if not use_v2:
        runner.init_routed_experts_capturer = Mock(
            side_effect=lambda: events.append("capture")
        )
    model = SimpleNamespace()
    if legacy_capture is not None:
        model.enable_return_routed_experts = legacy_capture
    config = SimpleNamespace(
        aux_output_config=SimpleNamespace(enable_return_routed_experts=True)
    )
    instance = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks=0),
        model_config=model,
        vllm_config=config,
        model_runner=runner,
        use_v2_model_runner=use_v2,
        _maybe_get_memory_pool_context=lambda **kw: nullcontext(),
    )
    kv_config = SimpleNamespace(num_blocks=17, needs_kv_cache_zeroing=zeroing)
    namespace["initialize_from_config"](instance, kv_config)
    assert instance.cache_config.num_gpu_blocks == 17
    connector.assert_called_once_with(config, kv_config)
    runner.initialize_kv_cache.assert_called_once_with(kv_config)
    expected = ["connector", "initialize"]
    if not use_v2 and legacy_capture:
        expected.append("capture")
    if zeroing:
        expected.append("zero")
    assert events == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_v2_respects_the_032_jit_warmup_switch(enabled):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    worker = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WorkerFL"
    )
    method = next(
        n
        for n in worker.body
        if isinstance(n, ast.FunctionDef) and n.name == "compile_or_warm_up_model"
    )
    branch = next(
        n
        for n in method.body
        if isinstance(n, ast.If)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "warmup_kernels"
            for call in ast.walk(n)
        )
    )
    warmup = Mock()
    instance = SimpleNamespace(
        use_v2_model_runner=True,
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_jit_warmup=enabled)
        ),
        model_runner=object(),
        execute_model=object(),
        sample_tokens=object(),
    )
    namespace = {"self": instance, "warmup_kernels": warmup}
    exec(
        compile(ast.Module(body=[branch], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    if enabled:
        warmup.assert_called_once_with(
            instance.model_runner, instance.execute_model, instance.sample_tokens
        )
    else:
        warmup.assert_not_called()
