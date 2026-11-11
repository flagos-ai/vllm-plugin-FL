"""The CoreX provider must never attempt the cooperative TLE entry point."""

import importlib.util
import sys
import types
from pathlib import Path


def test_no_tle_forwarding(monkeypatch):
    path = (
        Path(__file__).parents[2]
        / "vllm_fl/dispatch/backends/vendor/iluvatar/impl/fused_moe.py"
    )
    spec = importlib.util.spec_from_file_location("iluvatar_align_test", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    calls = []
    result = (object(), object(), object())

    def provider(*args):
        calls.append(args)
        return result

    module = types.ModuleType("flaggems_vllm.ops.moe_align_block_size")
    module.moe_align_block_size_no_tle = provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ids, mapping = object(), object()
    assert adapter.moe_align_block_size(ids, 16, 256, mapping, True, True) is result
    assert calls == [(ids, 16, 256, mapping, True)]
    assert adapter.moe_align_block_size(ids, 32, 8) is result
    assert calls[-1] == (ids, 32, 8, None, False)


def test_iluvatar_alignment_policy_avoids_generic_tle():
    import yaml

    root = Path(__file__).parents[2]
    policy = yaml.safe_load(
        (root / "vllm_fl/dispatch/config/iluvatar.yaml").read_text()
    )
    assert policy["op_backends"]["moe_align_block_size"] == ["vendor:iluvatar"]
    # Unrelated operations retain the existing FlagGems-first policy.
    assert policy["prefer"] == "flagos"


def test_vendor_fallbacks_reuse_reference_functions():
    from vllm_fl.dispatch.backends.reference.impl.activation import silu_and_mul_torch
    from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch
    from vllm_fl.dispatch.backends.reference.impl.rotary import rotary_embedding_torch
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.activation import silu_and_mul_iluvatar
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.normalization import rms_norm_iluvatar
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.rotary import rotary_embedding_iluvatar

    assert silu_and_mul_iluvatar is silu_and_mul_torch
    assert rms_norm_iluvatar is rms_norm_torch
    assert rotary_embedding_iluvatar is rotary_embedding_torch
