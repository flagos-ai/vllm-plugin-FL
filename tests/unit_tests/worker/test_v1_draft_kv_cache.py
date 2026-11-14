# SPDX-License-Identifier: Apache-2.0
"""V1 KV cache initialization only reads the drafter on the last PP rank."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("is_last_rank", [False, True], ids=["non-last", "last"])
def test_draft_attention_groups_are_only_added_on_last_pp_rank(
    monkeypatch, is_last_rank
):
    # Load the method without importing accelerator-specific runner dependencies.
    source = Path(__file__).resolve().parents[3] / "vllm_fl/worker/model_runner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerFL"
    )
    method = next(
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_initialize_kv_cache_tensors_v032"
    )

    caches = {}
    bind_kv = Mock()
    map_cache = Mock(return_value={})
    gpu_runner = ModuleType("vllm.v1.worker.gpu_model_runner")
    gpu_runner.bind_kv_cache = bind_kv
    gpu_runner.map_kv_caches_to_kernel_blocks = map_cache
    worker_utils = ModuleType("vllm.v1.worker.utils")
    worker_utils.allocate_kv_cache = Mock(return_value=caches)
    monkeypatch.setitem(sys.modules, gpu_runner.__name__, gpu_runner)
    monkeypatch.setitem(sys.modules, worker_utils.__name__, worker_utils)
    namespace = {
        "get_pp_group": lambda: SimpleNamespace(is_last_rank=is_last_rank),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    model_group, draft_group = object(), object()
    runner = SimpleNamespace(
        attn_groups=[[model_group]],
        speculative_config=object(),
        device="cpu",
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: "LBNHC",
            cache_dtype="auto",
        ),
        shared_kv_cache_layers={},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test")),
        compilation_config=SimpleNamespace(static_forward_context={}),
        kv_caches=[],
    )
    if is_last_rank:
        runner.drafter = SimpleNamespace(draft_attn_groups=[draft_group])

    initialize = namespace["_initialize_kv_cache_tensors_v032"]
    result = initialize(runner, SimpleNamespace(kv_cache_groups=[]), [128])
    assert result is caches
    expected_groups = [model_group, draft_group] if is_last_rank else [model_group]
    map_cache.assert_called_once_with(caches, expected_groups)
    bind_kv.assert_called_once()
