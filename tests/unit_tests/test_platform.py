import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_fl.platform import PlatformFL


def test_npu_simple_compile_backend_is_eager():
    expected_backend = "eager" if PlatformFL.device_type == "npu" else "inductor"

    assert PlatformFL.simple_compile_backend == expected_backend


def test_ascend_entrypoint_stays_inside_fl(monkeypatch):
    if PlatformFL.device_type != "npu":
        pytest.skip("Ascend-only entrypoint behavior")

    import vllm_fl

    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)

    assert vllm_fl.register() == "vllm_fl.platform.PlatformFL"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_ascend_platform_hook_installs_hybrid_cache_patch_early(monkeypatch):
    if PlatformFL.device_type != "npu":
        pytest.skip("Ascend-only entrypoint behavior")

    from vllm_fl.dispatch.backends.vendor.ascend import patch as ascend_patch

    called = []
    monkeypatch.setattr(ascend_patch, "patch_mamba_config", lambda: called.append(True))

    PlatformFL.pre_register_and_update()

    assert called == [True]


def test_ascend_refresh_block_size_preserves_hybrid_page_alignment():
    from vllm_fl.dispatch.backends.vendor.ascend.patch import refresh_block_size

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=2048,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        model_config=SimpleNamespace(is_hybrid=True),
    )

    refresh_block_size(config)

    assert config.cache_config.block_size == 2048


def test_ascend_refresh_block_size_keeps_generic_chunked_prefill_default():
    from vllm_fl.dispatch.backends.vendor.ascend.patch import refresh_block_size

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=2048,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        model_config=SimpleNamespace(is_hybrid=False),
    )

    refresh_block_size(config)

    assert config.cache_config.block_size == 128


def test_non_ascend_graph_capture_context_is_preserved():
    model_runner_path = (
        Path(__file__).parents[2] / "vllm_fl" / "worker" / "model_runner.py"
    )
    module = ast.parse(model_runner_path.read_text())
    graph_capture_if = next(
        node
        for node in module.body
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.FunctionDef) and child.name == "graph_capture"
            for child in node.body
        )
    )
    condition = compile(
        ast.Expression(graph_capture_if.test), str(model_runner_path), "eval"
    )

    def uses_local_capture(device_type, dist_backend="nccl"):
        platform = SimpleNamespace(
            device_type=device_type,
            dist_backend=dist_backend,
        )
        return eval(condition, {"current_platform": platform})

    assert uses_local_capture("npu", "hccl") is False
    assert uses_local_capture("cuda", "nccl") is False
    assert uses_local_capture("musa", "mccl") is True
    assert uses_local_capture("cuda", "flagcx") is True
