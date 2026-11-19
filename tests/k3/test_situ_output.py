# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from flag_gems import runtime
from flaggems_vllm.ops.situ_and_mul import situ_and_mul

from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts import (
    activations,
    tp8_experts,
)


@pytest.mark.skipif(
    runtime.device.vendor_name != "metax" or not torch.cuda.is_available(),
    reason="MetaX BF16 output contract",
)
def test_situ_public_provider_and_aliased_output(monkeypatch):
    assert tp8_experts.situ_and_mul is activations.situ_and_mul is situ_and_mul
    x = torch.linspace(-4, 4, 8 * 384, device="cuda").bfloat16().view(8, 384)
    initial = x.clone()
    gate, up = initial.float().chunk(2, dim=1)
    expected = (
        (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (25 * torch.tanh(up / 25))
    ).bfloat16()

    def unexpected(*args, **kwargs):
        pytest.fail("The FGV out API cannot consume overlapping storage")

    monkeypatch.setattr(activations, "situ_and_mul", unexpected)
    out = x.view(-1)[: expected.numel()].view_as(expected)
    activations.situ_out(out, x, beta=4.0, linear_beta=25.0)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert torch.equal(x.view(-1)[out.numel() :], initial.view(-1)[out.numel() :])
