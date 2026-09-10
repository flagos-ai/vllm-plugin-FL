# Copyright (c) 2026 BAAI. All rights reserved.

import pytest
import torch

pytest.importorskip("torch_npu")

import triton
import triton.language as tl
from triton import knobs

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
    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", lambda **kw: events.append(kw))
    source = torch.arange(128, dtype=torch.float32, device="npu")
    destination = torch.empty_like(source)

    _copy_with_compile_hook[(1,)](source, destination, BLOCK=128)

    torch.testing.assert_close(destination.cpu(), source.cpu())
    assert events
    assert events[0]["compile"]["launch_cooperative_grid"] is False
