# Copyright (c) 2026 BAAI. All rights reserved.

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_fl.dispatch.backends.flaggems import register_ops
from vllm_fl.dispatch.backends.flaggems.register_ops import _lazy_fn
from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind

ROOT = Path(__file__).resolve().parents[3]


MIGRATED_FLAGGEMS_OPS = {
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_kda_with_safe_gate",
    "fused_recurrent_kda",
    "fused_safe_kda_gate",
    "top_k_per_row_prefill",
    "top_k_per_row_decode",
    "sparse_indexer_gather_cache",
    "sparse_indexer_paged_mqa_logits",
    "sparse_indexer_pack_seq",
    "sparse_indexer_unpack_seq",
    "sparse_indexer_expand_pools_to_tokens",
    "sparse_indexer_append_tail_to_topk",
    "sparse_indexer_persist_prefill_tail",
    "sparse_indexer_kpool_compress_and_write_cache",
    "sparse_indexer_kpool_decode_update_and_maybe_write_cache_batched",
}


def test_lazy_fn_imports_once_without_prepare():
    module = SimpleNamespace(kernel=lambda value: value + 1)
    with patch(
        "vllm_fl.dispatch.backends.flaggems.register_ops.import_module",
        return_value=module,
    ) as import_module:
        lazy = _lazy_fn("test_backend", "kernel", lambda: True)
        assert lazy(1) == 2
        assert lazy(2) == 3

    import_module.assert_called_once_with("test_backend")


def test_lazy_fn_prepared_call_is_fullgraph_safe():
    module = SimpleNamespace(kernel=lambda value: value + 1)
    with patch(
        "vllm_fl.dispatch.backends.flaggems.register_ops.import_module",
        return_value=module,
    ) as import_module:
        lazy = _lazy_fn("test_backend", "kernel", lambda: True)
        lazy._prepare()

    import_module.assert_called_once_with("test_backend")
    value = torch.tensor(1)
    with patch(
        "vllm_fl.dispatch.backends.flaggems.register_ops.import_module",
        side_effect=AssertionError("prepared calls must not import"),
    ):
        compiled = torch.compile(lazy, backend="eager", fullgraph=True)
        assert compiled(value).item() == 2
        assert compiled(value + 1).item() == 3


def test_migrated_kernels_are_registered_as_flagos(monkeypatch):
    monkeypatch.setattr(register_ops, "use_flaggems_op", lambda _: True)
    registry = OpRegistry()
    register_ops.register_builtins(registry)
    entries = registry.snapshot().impls_by_op

    for op_name in MIGRATED_FLAGGEMS_OPS:
        (impl,) = entries[op_name]
        assert impl.kind == BackendImplKind.DEFAULT
        assert impl.impl_id == "default.flagos"
        assert impl.vendor is None
        assert callable(impl.fn._prepare)


def test_migrated_kernel_dependency_blacklist_removes_candidate(monkeypatch):
    monkeypatch.setattr(
        register_ops,
        "use_flaggems_op",
        lambda name: name != "cp_gather_indexer_k_bf16_cache",
    )
    registry = OpRegistry()
    register_ops.register_builtins(registry)

    assert "sparse_indexer_gather_cache" not in registry.snapshot().impls_by_op


def test_model_capabilities_use_generic_dispatch_modules():
    assert not (ROOT / "vllm_fl/kernels").exists()
    assert not (
        ROOT / "vllm_fl/dispatch/backends/flaggems/impl/bounded_activation.py"
    ).exists()

    flaggems_activation = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.impl.activation"
    )
    flaggems_topk = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.impl.top_k_per_row"
    )
    reference_activation = importlib.import_module(
        "vllm_fl.dispatch.backends.reference.impl.activation"
    )
    reference_topk = importlib.import_module(
        "vllm_fl.dispatch.backends.reference.impl.top_k_per_row"
    )
    assert callable(flaggems_activation.silu_and_mul_with_clamp)
    assert callable(reference_activation.silu_and_mul_with_clamp)
    assert callable(flaggems_topk.top_k_per_row_prefill)
    assert callable(flaggems_topk.top_k_per_row_decode)
    assert callable(reference_topk.top_k_per_row_prefill)
    assert callable(reference_topk.top_k_per_row_decode)


def test_legacy_model_scoped_topk_ops_are_not_registered(monkeypatch):
    monkeypatch.setattr(register_ops, "use_flaggems_op", lambda _: True)
    registry = OpRegistry()
    register_ops.register_builtins(registry)
    entries = registry.snapshot().impls_by_op
    assert "sparse_indexer_topk_prefill" not in entries
    assert "sparse_indexer_topk_decode" not in entries
