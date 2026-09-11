# Copyright (c) 2026 BAAI. All rights reserved.

import pytest
import torch

pytest.importorskip("torch_npu")

import triton
import triton.language as tl
from triton import knobs

from vllm_fl.dispatch.backends.vendor.ascend.patch import patch_topk_topp_sampler
from vllm_fl.dispatch.backends.vendor.ascend.patches.triton_compat import (
    patch_triton_compile_hooks,
)


@triton.jit
def _copy_with_compile_hook(source, destination, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(destination + offsets, tl.load(source + offsets))


@pytest.mark.gpu
def test_ascend_jit_post_compile_hook(monkeypatch):
    patch_triton_compile_hooks()
    events = []
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setattr(
        knobs.runtime, "jit_post_compile_hook", lambda **kw: events.append(kw)
    )
    source = torch.arange(128, dtype=torch.float32, device="npu")
    destination = torch.empty_like(source)

    _copy_with_compile_hook[(1,)](source, destination, BLOCK=128)

    torch.testing.assert_close(destination.cpu(), source.cpu())
    assert events
    assert events[0]["compile"]["launch_cooperative_grid"] is False


def test_ascend_uses_pytorch_topk_topp_for_large_batches(monkeypatch):
    import vllm.v1.sample.ops.topk_topp_sampler as sampler

    monkeypatch.setattr(sampler, "HAS_TRITON", True)
    patch_topk_topp_sampler()

    logits = torch.arange(16, dtype=torch.float32).repeat(8, 1)
    top_k = torch.full((8,), 4, dtype=torch.int32)
    output = sampler.apply_top_k_top_p(logits, top_k, None)

    assert sampler.HAS_TRITON is False
    assert torch.isfinite(output).sum(dim=-1).tolist() == [4] * 8
