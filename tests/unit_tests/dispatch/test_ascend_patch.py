# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import torch

from vllm_fl.dispatch.backends.vendor.ascend.patch import (
    patch_accelerator_empty_cache,
)


def test_patch_accelerator_empty_cache(monkeypatch):
    replacement = lambda: None
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(empty_cache=replacement), raising=False
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)

    patch_accelerator_empty_cache()

    assert torch.accelerator.empty_cache is replacement
