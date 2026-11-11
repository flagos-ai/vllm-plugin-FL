# Copyright (c) 2026 BAAI. All rights reserved.
"""GCU eager/compiled SiLU must obey the same split and rounding contract."""

import pytest
import torch
import torch.nn.functional as F

from vllm_fl.dispatch.backends.vendor.gcu.impl import activation


@pytest.mark.parametrize("shape", [(1, 16), (8, 64), (2, 3, 32)])
@pytest.mark.parametrize("compiled", [False, True])
def test_silu_paths(monkeypatch, shape, compiled):
    import flaggems_vllm

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiled)
    monkeypatch.setattr(flaggems_vllm, "silu_and_mul", lambda a, b: F.silu(a) * b)
    monkeypatch.setattr(activation, "_compiled_silu", lambda a, b: F.silu(a) * b)
    x = torch.linspace(-5, 5, torch.tensor(shape).prod().item()).reshape(shape)
    a, b = x.chunk(2, dim=-1)
    with torch.no_grad():
        torch.testing.assert_close(activation.silu_and_mul_gcu(None, x), F.silu(a) * b)


def test_missing_private_entry_uses_reference(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(activation, "_compiled_silu", None)
    x = torch.tensor([[-3.0, -1.0, 2.0, 4.0]])
    with torch.no_grad():
        torch.testing.assert_close(
            activation.silu_and_mul_gcu(None, x), F.silu(x[:, :2]) * x[:, 2:]
        )
