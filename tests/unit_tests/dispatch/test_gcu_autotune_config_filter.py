"""Unit tests for the GCU autotune config filter (#557 layer 4).

torch_gcu emits autotune candidates with kwargs the Inductor kernel
signatures do not define (SPLIT_K); the filter keeps only kwargs-free
candidates and synthesizes a plain fallback when every candidate carries
extras.
"""

from types import SimpleNamespace
from unittest.mock import patch

from vllm_fl.dispatch.backends.vendor.gcu.patches.autotune_config_filter import (
    _filter_configs,
    apply_autotune_config_filter_for_gcu,
)


def _config(extra=None, num_warps=4, num_stages=2):
    return SimpleNamespace(
        kwargs=dict(extra) if extra else {},
        num_warps=num_warps,
        num_stages=num_stages,
    )


def test_filter_keeps_plain_candidates_and_drops_split_k():
    plain = _config()
    split_k = _config({"SPLIT_K": 8})

    assert _filter_configs([plain, split_k]) == [plain]


def test_filter_synthesizes_fallback_when_all_carry_extras():
    split_k = _config({"SPLIT_K": 8}, num_warps=8, num_stages=3)

    (result,) = _filter_configs([split_k])
    assert result.kwargs == {}
    assert result.num_warps == 8
    assert result.num_stages == 3
    # the original candidate is untouched
    assert split_k.kwargs == {"SPLIT_K": 8}


def test_filter_passes_through_empty_and_odd_inputs():
    assert _filter_configs([]) == []
    sentinel = object()
    assert _filter_configs(sentinel) is sentinel


def test_apply_wraps_entry_points_idempotently():
    import torch._inductor.runtime.triton_heuristics as th

    def _vendor_configs(*args, **kwargs):
        return [_config({"SPLIT_K": 4})]

    with (
        patch.object(th, "_reduction_configs", _vendor_configs),
        patch.object(th, "_persistent_reduction_configs", _vendor_configs),
    ):
        delattr(th, "_gcu_config_filter_patched")
        apply_autotune_config_filter_for_gcu()
        first = th._reduction_configs
        assert getattr(first, "_gcu_config_filter", False)
        (result,) = first()
        assert result.kwargs == {}

        # second apply must not re-wrap
        apply_autotune_config_filter_for_gcu()
        assert th._reduction_configs is first
        th._gcu_config_filter_patched = False
