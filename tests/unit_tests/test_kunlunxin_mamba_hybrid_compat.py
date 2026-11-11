# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

"""Unit tests for mamba_hybrid_compat."""

import types
import sys

import pytest
import torch


@pytest.fixture
def fake_mamba_hybrid(monkeypatch):
    """Inject a fake mamba_hybrid module into sys.modules.

    Pre-seed the full parent-package path so the xpytorch import hook never
    attempts to import the real CUDA-dependent module chain.
    """
    # Build a minimal fake module with the two kernels we want to replace.
    mh = types.ModuleType("vllm.v1.worker.gpu.model_states.mamba_hybrid")

    class _FakeKernel:
        """Stand-in for a Triton kernel before the compat patch runs."""
        def __getitem__(self, _grid):
            raise RuntimeError(
                "original Triton kernel shim should not be called after patch"
            )

    mh._scatter_num_accepted_kernel = _FakeKernel()
    mh._fill_num_accepted_kernel = _FakeKernel()

    # Seed parent packages so nested-import resolution never touches real code.
    for pkg in (
        "vllm",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.model_states",
    ):
        if pkg not in sys.modules:
            monkeypatch.setitem(sys.modules, pkg, types.ModuleType(pkg))

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu.model_states.mamba_hybrid",
        mh,
    )
    return mh


def test_scatter_basic(fake_mamba_hybrid):
    # Reset the idempotency flag so the compat runs even if a prior test set it.
    mh = fake_mamba_hybrid
    if hasattr(mh, "_vllm_fl_kunlunxin_mamba_hybrid_scatter_torch"):
        delattr(mh, "_vllm_fl_kunlunxin_mamba_hybrid_scatter_torch")

    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.mamba_hybrid_compat import (
        apply_mamba_hybrid_compat,
    )
    apply_mamba_hybrid_compat()

    idx_mapping = torch.tensor([2, -1, 0, 1], dtype=torch.int64)
    num_sampled = torch.tensor([3, 5, 0, 2], dtype=torch.int32)
    num_accepted = torch.zeros(4, dtype=torch.int32)

    mh._scatter_num_accepted_kernel[(4,)](idx_mapping, num_sampled, num_accepted)

    assert num_accepted[2].item() == 3  # max(3, 1)
    assert num_accepted[0].item() == 1  # max(0, 1) → 1
    assert num_accepted[1].item() == 2  # max(2, 1)
    assert num_accepted[3].item() == 0  # never written (idx=-1)


def test_scatter_clamp_to_one(fake_mamba_hybrid):
    mh = fake_mamba_hybrid
    if hasattr(mh, "_vllm_fl_kunlunxin_mamba_hybrid_scatter_torch"):
        delattr(mh, "_vllm_fl_kunlunxin_mamba_hybrid_scatter_torch")

    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.mamba_hybrid_compat import (
        apply_mamba_hybrid_compat,
    )
    apply_mamba_hybrid_compat()

    idx_mapping = torch.tensor([0], dtype=torch.int64)
    num_sampled = torch.tensor([0], dtype=torch.int32)
    num_accepted = torch.zeros(1, dtype=torch.int32)

    mh._scatter_num_accepted_kernel[(1,)](idx_mapping, num_sampled, num_accepted)
    assert num_accepted[0].item() == 1


def test_scatter_all_negative(fake_mamba_hybrid):
    mh = fake_mamba_hybrid
    if hasattr(mh, "_vllm_fl_kunlunxin_mamba_hybrid_scatter_torch"):
        delattr(mh, "_vllm_fl_kunlunxin_mamba_hybrid_scatter_torch")

    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.mamba_hybrid_compat import (
        apply_mamba_hybrid_compat,
    )
    apply_mamba_hybrid_compat()

    idx_mapping = torch.tensor([-1, -1, -1], dtype=torch.int64)
    num_sampled = torch.tensor([2, 3, 4], dtype=torch.int32)
    num_accepted = torch.zeros(5, dtype=torch.int32)

    mh._scatter_num_accepted_kernel[(3,)](idx_mapping, num_sampled, num_accepted)
    assert num_accepted.sum().item() == 0


def test_fill_basic(fake_mamba_hybrid):
    mh = fake_mamba_hybrid
    if hasattr(mh, "_vllm_fl_kunlunxin_mamba_hybrid_scatter_torch"):
        delattr(mh, "_vllm_fl_kunlunxin_mamba_hybrid_scatter_torch")

    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.mamba_hybrid_compat import (
        apply_mamba_hybrid_compat,
    )
    apply_mamba_hybrid_compat()

    idx_mapping = torch.tensor([3, -1, 1], dtype=torch.int64)
    num_accepted = torch.zeros(4, dtype=torch.int32)

    mh._fill_num_accepted_kernel[(3,)](idx_mapping, num_accepted, 7)
    assert num_accepted[3].item() == 7
    assert num_accepted[1].item() == 7
    assert num_accepted[0].item() == 0  # never touched
    assert num_accepted[2].item() == 0  # never touched


def test_idempotent(fake_mamba_hybrid):
    mh = fake_mamba_hybrid
    if hasattr(mh, "_vllm_fl_kunlunxin_mamba_hybrid_scatter_torch"):
        delattr(mh, "_vllm_fl_kunlunxin_mamba_hybrid_scatter_torch")

    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.mamba_hybrid_compat import (
        apply_mamba_hybrid_compat,
        _PATCH_FLAG,
    )
    apply_mamba_hybrid_compat()
    apply_mamba_hybrid_compat()
    assert getattr(mh, _PATCH_FLAG, False)
