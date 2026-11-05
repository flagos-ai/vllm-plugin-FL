# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 integration and contract regressions."""

import ast
import unittest
import weakref
from contextlib import contextmanager
from functools import lru_cache
from inspect import signature
from types import SimpleNamespace as NS

import pytest
import torch
from _qwen38_cpu import ROOT, declarations

# Native contracts.

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


native_VENDOR = ROOT / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next"
native_COMMON = ROOT / "vllm_fl/models/qwen3_8_flash_next/common"
native_packing = declarations(
    native_COMMON / "qsa_packing.py",
    "qsa_contract_packing",
    namespace={"contextmanager": contextmanager},
)
native_workspace = declarations(
    native_COMMON / "qsa_workspace.py",
    "qsa_contract_workspace",
    namespace={
        "arange": torch.arange,
        "_QSA_PACK_ROWS_CACHE": native_packing._QSA_PACK_ROWS_CACHE,
    },
)


def native_load_capabilities(importer, get_device_name):
    return declarations(
        native_VENDOR / "qsa_native.py",
        "qsa_contract_native",
        namespace={
            "lru_cache": lru_cache,
            "signature": signature,
            "import_module": importer,
            "torch": NS(cuda=NS(get_device_name=get_device_name)),
        },
    )


def native_paged_api(q=None, *, block_table=None, softmax_scale=None, causal=False):
    pass


def test_native_capability_checks_actual_extension_and_device():
    calls = []
    modules = {
        "flash_attn": NS(flash_attn_varlen_func=native_paged_api),
        "flash_attn_2_cuda": NS(varlen_fwd=lambda: None),
    }

    def importer(name):
        calls.append(name)
        return modules[name]

    cap = native_load_capabilities(
        importer, lambda d: "C550" if d.index == 0 else "other"
    )
    assert cap.native_available(torch.device("cuda:0"))
    assert not cap.native_available(torch.device("cuda:1"))
    assert not cap.native_available(torch.device("cpu"))
    assert calls == ["flash_attn", "flash_attn_2_cuda"]


@pytest.mark.parametrize(
    "kind",
    ["package_missing", "symbol_missing", "wrong_signature", "extension_missing"],
)
def test_missing_native_capability_is_detected_before_execution(kind):
    def importer(name):
        if kind == "package_missing":
            raise ImportError("missing optional dependency")
        if name == "flash_attn":
            if kind == "symbol_missing":
                return NS()
            return NS(
                flash_attn_varlen_func=(lambda q: None)
                if kind == "wrong_signature"
                else native_paged_api
            )
        return NS() if kind == "extension_missing" else NS(varlen_fwd=lambda: None)

    cap = native_load_capabilities(importer, lambda device: "C550")
    assert not cap.native_available(torch.device("cuda:0"))


def test_workspace_shared_only_in_same_metadata_scope():
    with native_packing.qsa_metadata_build_scope():
        one = native_workspace.prepare_native_qsa_workspace(17, torch.device("cpu"))
        two = native_workspace.prepare_native_qsa_workspace(17, torch.device("cpu"))
        other_shape = native_workspace.prepare_native_qsa_workspace(
            18, torch.device("cpu")
        )
        assert one is two
        assert one is not other_shape
        assert one["table"].shape == (17, 513)
        assert one["query"].shape == (17, 3, 256)
        assert one["query"].dtype == torch.bfloat16
        assert one["cuq"].tolist() == list(range(18))
    with native_packing.qsa_metadata_build_scope():
        fresh = native_workspace.prepare_native_qsa_workspace(17, torch.device("cpu"))
        assert fresh["query"].data_ptr() != one["query"].data_ptr()


def test_workspace_rejects_first_allocation_inside_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        native_workspace,
        "allocate_qsa_workspace",
        lambda *args: pytest.fail("must reject before allocating"),
    )
    with pytest.raises(RuntimeError, match="before graph capture"):
        native_workspace.prepare_native_qsa_workspace(8, torch.device("cuda:0"))


def test_native_guard_preserves_scale_gate_and_output_contract():
    guard = declarations(
        native_VENDOR / "qsa_native.py",
        "qsa_native_eligibility",
        names={"native_eligible"},
    ).native_eligible
    device = torch.device("cuda:0")

    def tensor(shape, strides, dtype=torch.bfloat16):
        return NS(
            shape=shape,
            dtype=dtype,
            device=device,
            stride=lambda axis=None: strides if axis is None else strides[axis],
        )

    q = tensor((8, 3, 256), (768, 256, 1))
    k = tensor((10, 400, 1, 256), (102400, 256, 256, 1))
    idx = tensor((8, 2051), (2051, 1), torch.int32)
    good = guard(q, k, k, idx, 0.0625, q, q, {})
    assert good
    assert not guard(q, k, k, idx, 0.125, q, q, {})
    assert not guard(q, k, k, idx, 0.0625, q, None, {})
    fp32 = tensor(q.shape, q.stride(), torch.float32)
    assert not guard(q, k, k, idx, 0.0625, q, fp32, {})
    assert not guard(q, k, k, idx, 0.0625, fp32, q, {})
    assert not guard(q, k, k, idx, 0.0625, q, q, None)


def test_capture_workspace_lifetime_is_owned_by_model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: NS(cuda_stream=13))
    owner = {}
    query = NS(shape=(8, 3, 256), device=torch.device("cuda:0"), dtype=torch.bfloat16)
    first, second = {"buffer": object()}, {"buffer": object()}
    assert (
        native_workspace.retain_native_graph_workspace(owner, first, query, 2048)
        is first
    )
    assert (
        native_workspace.retain_native_graph_workspace(owner, second, query, 2048)
        is first
    )
    assert (
        native_workspace.retain_native_graph_workspace(owner, second, query, 8192)
        is second
    )
    assert len(owner) == 2
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    eager = {}
    assert (
        native_workspace.retain_native_graph_workspace(owner, eager, query, 2048)
        is eager
    )
    assert len(owner) == 2


# Groups contracts.

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


groups_packing = declarations(
    ROOT / "vllm_fl/models/qwen3_8_flash_next/common/qsa_packing.py",
    "qwen38_indexer_groups_cpu",
    namespace={"contextmanager": contextmanager},
)


def test_indexer_groups_preserve_request_boundaries_and_step_lifetime():
    starts = torch.tensor([0, 3, 3, 20], dtype=torch.int32)
    with groups_packing.qsa_metadata_build_scope():
        rows = groups_packing._cached_indexer_pack_rows(starts, 20, torch.device("cpu"))
        assert rows is groups_packing._cached_indexer_pack_rows(
            starts.clone(), 20, torch.device("cpu")
        )
        active = rows[rows >= 0]
        assert active.tolist() == list(range(20))
        assert rows.shape == (3, 16)
        for group in rows.tolist():
            requests = {0 if x < 3 else 2 for x in group if x >= 0}
            assert len(requests) <= 1
    with groups_packing.qsa_metadata_build_scope():
        fresh = groups_packing._cached_indexer_pack_rows(
            starts, 20, torch.device("cpu")
        )
        assert fresh is not rows
        assert torch.equal(fresh, rows)


def test_indexer_groups_include_padded_rows_without_gpu_readback():
    rows = groups_packing._cached_indexer_pack_rows(
        torch.tensor([0, 3, 3, 20]), 24, torch.device("cpu")
    )
    assert rows[rows >= 0].tolist() == list(range(24))
    with pytest.raises(ValueError, match="already be on CPU"):
        groups_packing._cached_indexer_pack_rows(
            torch.empty(3, device="meta"), 24, torch.device("cpu")
        )


@pytest.mark.parametrize(
    "starts,tokens,pack",
    [([], 0, 16), ([1, 2], 2, 16), ([0, 4, 3], 4, 16), ([0, 5], 4, 16), ([0, 4], 4, 4)],
)
def test_indexer_rejects_invalid_request_boundaries(starts, tokens, pack):
    with pytest.raises(ValueError):
        groups_packing._cached_indexer_pack_rows(
            torch.tensor(starts, dtype=torch.int32), tokens, torch.device("cpu"), pack
        )


def test_indexer_empty_batch_is_valid():
    assert (
        groups_packing._cached_indexer_pack_rows(
            torch.tensor([0]), 0, torch.device("cpu")
        )
        is None
    )


# Width contracts.

# SPDX-License-Identifier: Apache-2.0


width_width = declarations(
    ROOT / "vllm_fl/models/qwen3_8_flash_next/common/qsa_cache.py",
    "qwen38_width_cpu",
    namespace={"PreparedQSAForwardMetadata": NS},
    names={
        "qsa_column_buckets",
        "qsa_score_columns",
        "qsa_graph_columns",
        "qsa_cudagraph_key",
    },
)


@pytest.mark.parametrize(
    ("length", "columns"),
    [
        (0, 512),
        (1024, 512),
        (2051, 512),
        (2052, 2048),
        (8195, 2048),
        (8196, 8192),
        (32771, 8192),
        (32772, 17600),
        (69632, 17600),
    ],
)
def test_complete_group_boundaries(length, columns):
    assert width_width.qsa_score_columns(length, 4, 17600) == columns


def test_capacity_guard_and_page_alignment():
    assert width_width.qsa_column_buckets(17600) == (512, 2048, 8192, 17600)
    assert width_width.qsa_column_buckets(1600) == (512, 1600)
    with pytest.raises(ValueError, match="capacity"):
        width_width.qsa_score_columns(70404, 4, 17600)
    for length in range(0, 70401, 127):
        assert length // 4 <= width_width.qsa_score_columns(length, 4, 17600) <= 17600


def test_graph_key_separates_lengths_without_changing_token_descriptor():
    descriptor = (8, 4, True)
    one = NS(attn_metadata={"qsa": NS(compress_ratio=4, score_columns=512)})
    two = NS(attn_metadata={"qsa": NS(compress_ratio=4, score_columns=2048)})
    assert width_width.qsa_cudagraph_key(
        descriptor, one
    ) != width_width.qsa_cudagraph_key(descriptor, two)
    assert width_width.qsa_cudagraph_key(descriptor, NS(attn_metadata={})) == descriptor
    assert (
        width_width.qsa_cudagraph_key(descriptor, NS(attn_metadata=None)) == descriptor
    )


def test_graph_key_rejects_inconsistent_layer_widths():
    with pytest.raises(ValueError, match="disagree"):
        width_width.qsa_graph_columns(
            {
                "a": NS(compress_ratio=4, score_columns=512),
                "b": NS(compress_ratio=4, score_columns=2048),
            }
        )


# Binding contracts.

binding_SOURCE = ROOT / "vllm_fl/models/qwen3_8_flash_next/common/qsa_cache.py"


def binding_cache_type():
    tree = ast.parse(binding_SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "QSAKeyStateCache"
    )

    class StateCache:
        def __init__(self, head_size, **kwargs):
            self.head_size = head_size
            self.kv_cache = torch.tensor([])

        def bind_kv_cache(self, tensor):
            self.kv_cache = tensor

    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(code)
    scope = {"torch": torch, "_QSAStateCache": StateCache}
    exec(compile(code, str(binding_SOURCE), "exec"), scope)
    return scope["QSAKeyStateCache"]


class binding_QSACacheBindingTest(unittest.TestCase):
    def test_legacy_assignment_views_track_rebinding_and_padding(self):
        cache = binding_cache_type()(head_size=128, cache_rope_positions=True)
        storage = torch.zeros(3, 4, 1, 160, dtype=torch.bfloat16)
        bound = storage[..., :140]
        cache.kv_cache = bound  # Exact vLLM 0.24 binding ABI.
        self.assertEqual(cache.key_cache.shape, (3, 4, 1, 128))
        self.assertEqual(cache.rope_position_cache.shape, (3, 4, 1, 3))
        cache.key_cache[1, 2, 0, 7] = 5
        self.assertEqual(storage[1, 2, 0, 7].item(), 5)
        positions = torch.tensor([2**40 + 1, 123, -7], dtype=torch.int64)
        cache.rope_position_cache[1, 2, 0].copy_(positions)
        torch.testing.assert_close(
            storage[1, 2, 0, 128:140].view(torch.int64), positions
        )
        self.assertEqual(cache.key_cache.stride(0), storage.stride(0))
        previous = weakref.ref(bound)
        replacement = torch.zeros_like(storage)[..., :140]
        cache.kv_cache = replacement
        del bound
        self.assertIsNone(previous())
        self.assertEqual(cache.key_cache.data_ptr(), replacement.data_ptr())
        self.assertEqual(cache.key_cache[1, 2, 0, 7].item(), 0)
        cache.kv_cache = torch.tensor([])
        self.assertEqual(cache.key_cache.numel(), 0)
        self.assertEqual(cache.rope_position_cache.numel(), 0)

    def test_optional_hook_validates_and_text_cache_has_no_positions(self):
        cache = binding_cache_type()(head_size=128, cache_rope_positions=False)
        data = torch.zeros(2, 4, 1, 128, dtype=torch.bfloat16)
        cache.bind_kv_cache(data)
        self.assertIsNone(cache.rope_position_cache)
        self.assertEqual(cache.key_cache.data_ptr(), data.data_ptr())
        with self.assertRaisesRegex(ValueError, "packed BF16"):
            cache.bind_kv_cache(data.float())
        with self.assertRaisesRegex(ValueError, "QSA raw cache"):
            cache.bind_kv_cache(data.squeeze(2))
