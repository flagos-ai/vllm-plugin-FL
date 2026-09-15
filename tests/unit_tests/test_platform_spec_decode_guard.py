# Copyright (c) 2026 BAAI. All rights reserved.

"""Kunlunxin must reject speculative decoding at configuration time.

The native xtorch_ops.causal_conv1d_update kernel has no num_accepted_tokens
support, so a spec-decode config that reaches the engine only fails on the
first inference request. Failing during config validation names the cause
while the user can still act on it.
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm_fl.platform import PlatformFL


def _config(speculative_config=None):
    """A VllmConfig stub exposing only what check_and_update_config reads.

    cache_config/model_config are None so the block-size and MLA branches are
    skipped -- those need a real config and are unrelated to this guard.
    """
    config = MagicMock()
    config.speculative_config = speculative_config
    config.cache_config = None
    config.model_config = None
    return config


def _as_platform(monkeypatch, vendor_name, device_type):
    monkeypatch.setattr(PlatformFL, "vendor_name", vendor_name)
    monkeypatch.setattr(PlatformFL, "device_type", device_type)


@pytest.fixture
def kunlunxin(monkeypatch):
    _as_platform(monkeypatch, "kunlunxin", "cuda")


def test_kunlunxin_rejects_speculative_decoding(kunlunxin):
    with pytest.raises(ValueError, match="Speculative decoding is not supported"):
        PlatformFL.check_and_update_config(_config(speculative_config=MagicMock()))


def test_kunlunxin_configures_without_speculative_decoding(kunlunxin):
    PlatformFL.check_and_update_config(_config())


@pytest.mark.parametrize("vendor_name", ["nvidia", "metax"])
def test_other_vendors_keep_speculative_decoding(monkeypatch, vendor_name):
    _as_platform(monkeypatch, vendor_name, "cuda")

    PlatformFL.check_and_update_config(_config(speculative_config=MagicMock()))


def _kunlunxin_available():
    try:
        import xtorch_ops  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(
    not _kunlunxin_available(),
    reason="Kunlunxin device or xtorch_ops not available",
)
def test_kernel_boundary_rejects_accepted_tokens():
    """Second layer: a spec-decode call that does reach the kernel is named, not crashed.

    The config guard is what users hit; this pins the failure surface behind it
    so the two cannot drift into a silent wrong answer.
    """
    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.causal_conv1d import (
        causal_conv1d_update_kunlunxin as causal_conv1d_update,
    )

    device = torch.device("cuda")
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        causal_conv1d_update(
            torch.randn(2, 8, dtype=torch.bfloat16, device=device),
            torch.randn(4, 8, 4, dtype=torch.bfloat16, device=device),
            torch.randn(8, 4, dtype=torch.bfloat16, device=device),
            None,
            "silu",
            conv_state_indices=torch.zeros(2, dtype=torch.long, device=device),
            num_accepted_tokens=torch.ones(2, dtype=torch.int32, device=device),
        )
