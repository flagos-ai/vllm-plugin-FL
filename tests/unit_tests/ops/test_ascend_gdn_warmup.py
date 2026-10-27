# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest


def test_native_gdn_skips_only_qwen_triton_warmup(monkeypatch):
    pytest.importorskip("torch_npu")
    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot
    from vllm.model_executor.warmup import kernel_warmup

    from vllm_fl.dispatch.backends.vendor.ascend.patch import patch_gdn

    original_kernel_warmup = kernel_warmup.kernel_warmup
    original_qwen_warmup = Mock(side_effect=AssertionError("Triton is unused"))
    monkeypatch.setattr(kernel_warmup, "qwen_triton_warmup", original_qwen_warmup)
    register = Mock()
    monkeypatch.setattr(PluggableLayer, "register_oot", register)
    monkeypatch.delitem(op_registry_oot, "QwenGatedDeltaNetAttention", raising=False)
    patch_gdn()
    kernel_warmup.qwen_triton_warmup(None, None)
    original_qwen_warmup.assert_not_called()
    register.assert_called_once()
    assert kernel_warmup.kernel_warmup is original_kernel_warmup


def test_native_gdn_registration_is_idempotent(monkeypatch):
    from vllm.model_executor.custom_op import op_registry_oot

    from vllm_fl.dispatch.backends.vendor.ascend.impl.gdn import (
        AscendQwenGatedDeltaNetAttention,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.patch import patch_gdn

    monkeypatch.setitem(
        op_registry_oot, "QwenGatedDeltaNetAttention", AscendQwenGatedDeltaNetAttention
    )
    patch_gdn()
    patch_gdn()
    assert (
        op_registry_oot["QwenGatedDeltaNetAttention"]
        is AscendQwenGatedDeltaNetAttention
    )
