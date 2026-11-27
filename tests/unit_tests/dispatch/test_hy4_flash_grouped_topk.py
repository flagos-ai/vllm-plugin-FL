# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_fl.dispatch import grouped_topk as routing
from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import (
    grouped_topk_flaggems,
)
from vllm_fl.dispatch.backends.vendor.cuda import cuda
from vllm_fl.dispatch.backends.vendor.cuda.impl import fused_moe as cuda_moe
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.types import BackendImplKind, OpImpl

ARGS = (object(), 1, 1, 2, True, 1.0, object(), 1)


def test_legacy_provider_preserves_eight_arguments(monkeypatch):
    def legacy(scores, groups, top_group, topk, norm, scale, bias, scoring):
        return scores, topk

    monkeypatch.setitem(
        sys.modules, "flaggems_vllm", SimpleNamespace(grouped_topk=legacy)
    )
    assert grouped_topk_flaggems(*ARGS) == (ARGS[0], 2)
    assert grouped_topk_flaggems(*ARGS, is_padding=None) == (ARGS[0], 2)
    with pytest.raises(NotImplementedError, match="#1291"):
        grouped_topk_flaggems(*ARGS, is_padding=(False, True))


@pytest.mark.parametrize("mask", [(False, False), (False, True), (True, True)])
def test_mask_only_runs_capable_policy_backend(monkeypatch, mask):
    provider = Mock(return_value="flagos")

    def masked(
        scores, groups, top_group, topk, norm, scale, bias, scoring, is_padding=None
    ):
        return provider(
            scores,
            groups,
            top_group,
            topk,
            norm,
            scale,
            bias,
            scoring,
            is_padding=is_padding,
        )

    monkeypatch.setitem(
        sys.modules, "flaggems_vllm", SimpleNamespace(grouped_topk=masked)
    )
    native = Mock(side_effect=AssertionError("CUDA8 must not see a mask"))
    flagos = OpImpl(
        "grouped_topk", "default.flagos", BackendImplKind.DEFAULT, grouped_topk_flaggems
    )
    vendor = OpImpl(
        "grouped_topk", "vendor.cuda", BackendImplKind.VENDOR, native, vendor="cuda"
    )
    manager = OpManager()
    monkeypatch.setattr(
        manager, "resolve_candidates", Mock(return_value=[vendor, flagos])
    )
    monkeypatch.setattr(routing, "get_default_manager", lambda: manager)
    assert routing.grouped_topk(*ARGS, is_padding=mask) == "flagos"
    native.assert_not_called()
    provider.assert_called_once_with(*ARGS, is_padding=mask)

    # The candidate list reflects a policy that allows CUDA only.
    manager.resolve_candidates.return_value = [vendor]
    with pytest.raises(NotImplementedError, match="policy-eligible"):
        routing.grouped_topk(*ARGS, is_padding=mask)
    native.assert_not_called()


def test_unmasked_hy4_obeys_existing_policy_entry(monkeypatch):
    chosen = Mock(return_value="policy backend")
    monkeypatch.setattr(routing, "_grouped_topk", chosen)
    assert routing.grouped_topk(*ARGS, is_padding=None) == "policy backend"
    chosen.assert_called_once_with(*ARGS)


def test_cuda_backend_does_not_discard_mask(monkeypatch):
    dispatch = Mock(wraps=cuda_moe.grouped_topk_cuda)
    monkeypatch.setattr(cuda_moe, "grouped_topk_cuda", dispatch)
    with pytest.raises(NotImplementedError, match="cannot consume is_padding"):
        cuda.CudaBackend.grouped_topk(None, *ARGS, is_padding=(False, True))
    dispatch.assert_called_once_with(*ARGS, is_padding=(False, True))
