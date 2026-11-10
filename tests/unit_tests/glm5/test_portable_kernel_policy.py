# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

import pytest

from vllm_fl.models.glm5 import runtime


@pytest.mark.parametrize("name", ["causal_conv1d_fn", "chunk_kda_with_safe_gate"])
def test_portable_model_selects_its_public_kda_contract(monkeypatch, name):
    monkeypatch.setattr(runtime, "use_nvidia_reference", lambda: False)
    assert runtime.glm5_portable_moe_defaults().order_for(name) == ("flagos",)
    monkeypatch.setattr(runtime, "use_nvidia_reference", lambda: True)
    assert runtime.glm5_portable_moe_defaults().order_for(name) is None
