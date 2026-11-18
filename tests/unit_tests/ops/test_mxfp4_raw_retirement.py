# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.weak import WeakTensorKeyDictionary

spec = importlib.util.spec_from_file_location(
    "fl_retirement_contract_helper",
    Path(__file__).resolve().parents[3] / "vllm_fl/ops/fused_moe/raw_retirement.py",
)
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
from fl_retirement_contract_helper import (
    PARAMETER_NAMES,
    assert_retired_aliases,
    retire_raw_parameters,
)


def test_retirement_releases_raw_storage_preserves_parameter_contract():
    shapes = [(3, 256, 64), (3, 128, 64), (3, 256, 4), (3, 128, 4)]
    params = [
        torch.nn.Parameter(torch.zeros(s, dtype=torch.uint8), requires_grad=False)
        for s in shapes
    ]
    for p in params:
        p.weight_loader = object()
    layer = SimpleNamespace(**dict(zip(PARAMETER_NAMES, params)))
    layer.quant_method = object()
    layer._replace_quant_method = lambda replacement: None
    packed = (
        torch.ones((3, 16, 256), dtype=torch.int32),
        torch.ones((3, 16, 128), dtype=torch.int32),
        torch.ones((3, 4, 256), dtype=torch.int32),
        torch.ones((3, 4, 128), dtype=torch.int32),
    )
    packed_copies = [p.clone() for p in packed]
    source_refs = [p.untyped_storage()._weak_ref() for p in params]
    loaders = [p.weight_loader for p in params]
    ids = [id(p) for p in params]
    caches = {
        name: WeakTensorKeyDictionary()
        for name in ("_W_PACK_CACHE", "_SCALE_PACK_CACHE_E8M0", "_E8M0_FOLD_SAFE_CACHE")
    }
    for cache in caches.values():
        for p in params:
            cache[p] = object()
    gems = SimpleNamespace(**caches)
    try:
        state = retire_raw_parameters(layer, packed, gems)
        assert [id(p) for p in params] == ids
        assert [p.weight_loader for p in params] == loaders
        assert [tuple(p.shape) for p in params] == shapes
        assert all(torch.UntypedStorage._expired(r) for r in source_refs)
        assert all(len(c) == 0 for c in caches.values())
        assert assert_retired_aliases(layer, packed) is state
        assert retire_raw_parameters(layer, packed, gems) is state
        layer._replace_quant_method(layer.quant_method)
        with pytest.raises(RuntimeError, match="cannot switch"):
            layer._replace_quant_method(object())
        for actual, expected in zip(packed, packed_copies):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        params[0].data = params[0].clone()
        with pytest.raises(RuntimeError, match="moved/changed"):
            assert_retired_aliases(layer, packed)
    finally:
        for ref in source_refs:
            torch.UntypedStorage._free_weak_ref(ref)


def test_compaction_removes_only_zero_padding_and_keeps_gate_up_order():
    from fl_retirement_contract_helper import compact_raw_intermediate

    w1 = torch.full((1, 512, 64), 0x88, dtype=torch.uint8)  # Signed FP4 zeros.
    w1[:, :192] = 0x11
    w1[:, 256:448] = 0x22
    w2 = torch.full((1, 128, 128), 0x88, dtype=torch.uint8)
    w2[:, :, :96] = 0x33
    layer = SimpleNamespace(
        w13_weight=w1,
        w2_weight=w2,
        w13_weight_scale=torch.ones(1, 512, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(1, 128, 8, dtype=torch.uint8),
    )
    ids = [id(getattr(layer, n)) for n in PARAMETER_NAMES]
    compact_raw_intermediate(layer, 192)
    assert [id(getattr(layer, n)) for n in PARAMETER_NAMES] == ids
    assert w1.shape == (1, 384, 64)
    assert w2.shape == (1, 128, 96)
    assert torch.all(w1[:, :192] == 0x11)
    assert torch.all(w1[:, 192:] == 0x22)
    assert torch.all(w2 == 0x33)
    assert layer.w13_weight_scale.shape == (1, 384, 4)
    assert layer.w2_weight_scale.shape == (1, 128, 6)


def test_compaction_refuses_nonzero_tail_before_mutating_weights():
    from fl_retirement_contract_helper import compact_raw_intermediate

    w1 = torch.zeros((1, 512, 64), dtype=torch.uint8)
    w1[0, 200, 0] = 1
    layer = SimpleNamespace(
        w13_weight=w1,
        w2_weight=torch.zeros((1, 128, 128), dtype=torch.uint8),
        w13_weight_scale=torch.ones(1, 512, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(1, 128, 8, dtype=torch.uint8),
    )
    with pytest.raises(RuntimeError, match="nonzero"):
        compact_raw_intermediate(layer, 192)
    assert w1.shape == (1, 512, 64)
