# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("tokens", [1, 8, 64])
def test_npu_workspace_reserves_prefill_capacity(tokens, monkeypatch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_fl.ops.fused_moe import fused_moe_utils as module

    experts = object.__new__(module.TritonExpertsFL)
    experts.moe_config = SimpleNamespace(max_num_tokens=32)
    monkeypatch.setattr(module, "current_platform", SimpleNamespace(device_type="npu"))

    workspace13, workspace2, output = experts.workspace_shapes(
        tokens, 64, 32, 8, 16, 16, None, MoEActivation.SILU
    )
    assert workspace13 == (max(tokens, 32), 8, 32)
    assert workspace2 == (max(tokens, 32), 8, 64)
    assert output == (tokens, 32)
