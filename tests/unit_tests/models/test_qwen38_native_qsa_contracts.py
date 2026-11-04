# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU capability and ownership contracts; these do not validate GPU numerics."""

from contextlib import contextmanager
from functools import lru_cache
from inspect import signature
from types import SimpleNamespace as NS

import pytest
import torch
from _qwen38_cpu import ROOT, declarations

VENDOR = ROOT / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next"
COMMON = ROOT / "vllm_fl/models/qwen3_8_flash_next/common"
packing = declarations(
    COMMON / "qsa_packing.py",
    "qsa_contract_packing",
    namespace={"contextmanager": contextmanager},
)
workspace = declarations(
    COMMON / "qsa_workspace.py",
    "qsa_contract_workspace",
    namespace={
        "arange": torch.arange,
        "_QSA_PACK_ROWS_CACHE": packing._QSA_PACK_ROWS_CACHE,
    },
)


def load_capabilities(importer, get_device_name):
    return declarations(
        VENDOR / "qsa_native.py",
        "qsa_contract_native",
        namespace={
            "lru_cache": lru_cache,
            "signature": signature,
            "import_module": importer,
            "torch": NS(cuda=NS(get_device_name=get_device_name)),
        },
    )


def paged_api(q=None, *, block_table=None, softmax_scale=None, causal=False):
    pass


def test_native_capability_checks_actual_extension_and_device():
    calls = []
    modules = {
        "flash_attn": NS(flash_attn_varlen_func=paged_api),
        "flash_attn_2_cuda": NS(varlen_fwd=lambda: None),
    }

    def importer(name):
        calls.append(name)
        return modules[name]

    cap = load_capabilities(importer, lambda d: "C550" if d.index == 0 else "other")
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
                else paged_api
            )
        return NS() if kind == "extension_missing" else NS(varlen_fwd=lambda: None)

    cap = load_capabilities(importer, lambda device: "C550")
    assert not cap.native_available(torch.device("cuda:0"))


def test_workspace_shared_only_in_same_metadata_scope():
    with packing.qsa_metadata_build_scope():
        one = workspace.prepare_native_qsa_workspace(17, torch.device("cpu"))
        two = workspace.prepare_native_qsa_workspace(17, torch.device("cpu"))
        other_shape = workspace.prepare_native_qsa_workspace(18, torch.device("cpu"))
        assert one is two
        assert one is not other_shape
        assert one["table"].shape == (17, 513)
        assert one["query"].shape == (17, 3, 256)
        assert one["query"].dtype == torch.bfloat16
        assert one["cuq"].tolist() == list(range(18))
    with packing.qsa_metadata_build_scope():
        fresh = workspace.prepare_native_qsa_workspace(17, torch.device("cpu"))
        assert fresh["query"].data_ptr() != one["query"].data_ptr()


def test_workspace_rejects_first_allocation_inside_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        workspace,
        "allocate_qsa_workspace",
        lambda *args: pytest.fail("must reject before allocating"),
    )
    with pytest.raises(RuntimeError, match="before graph capture"):
        workspace.prepare_native_qsa_workspace(8, torch.device("cuda:0"))


def test_native_guard_preserves_scale_gate_and_output_contract():
    guard = declarations(
        VENDOR / "qsa_native.py", "qsa_native_eligibility", names={"native_eligible"}
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
    assert workspace.retain_native_graph_workspace(owner, first, query, 2048) is first
    assert workspace.retain_native_graph_workspace(owner, second, query, 2048) is first
    assert workspace.retain_native_graph_workspace(owner, second, query, 8192) is second
    assert len(owner) == 2
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    eager = {}
    assert workspace.retain_native_graph_workspace(owner, eager, query, 2048) is eager
    assert len(owner) == 2
