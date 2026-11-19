# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU call-routing checks; providers are substituted, no device numerics."""

import __future__

import ast
import enum
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def method(path, cls, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    owner = next(n for n in tree.body if getattr(n, "name", None) == cls)
    node = next(n for n in owner.body if getattr(n, "name", None) == name)
    exec(
        compile(
            ast.Module([node], type_ignores=[]),
            path,
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[name]


class Activation(enum.Enum):
    SILU = "silu"
    GELU = "gelu"


class FallbackSelected(Exception):
    pass


@pytest.mark.parametrize(
    "case",
    [
        "enabled",
        "disabled",
        "clamp",
        "quantized",
        "lora",
        "gelu",
        "empty",
        "backend",
        "error",
    ],
)
def test_thead_moe_provider_guards_and_error_propagation(monkeypatch, case):
    provider = Mock(return_value=torch.ones(2, 4))
    if case == "error":
        provider.side_effect = RuntimeError("provider failed")
    package = ModuleType("flaggems_vllm")
    package.fused_experts_impl = provider
    monkeypatch.setitem(sys.modules, package.__name__, package)
    apply = method(
        "vllm_fl/ops/fused_moe/fused_moe_utils.py",
        "TritonExpertsFL",
        "apply",
        {
            "torch": torch,
            "MoEActivation": Activation,
            "current_platform": SimpleNamespace(
                vendor_name="thead", is_cuda=lambda: False
            ),
            "fl_envs": SimpleNamespace(VLLM_FL_USE_FLAGGEMS_VLLM=case != "disabled"),
            "logger": Mock(),
        },
    )
    quant = SimpleNamespace(
        quant_dtype="fp8" if case == "quantized" else None,
        weight_quant_dtype=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
    )
    experts = SimpleNamespace(
        _lora_context=object() if case == "lora" else None,
        moe_config=SimpleNamespace(
            moe_backend="cutlass" if case == "backend" else "triton"
        ),
        activation_config=SimpleNamespace(clamp_limit=7.0 if case == "clamp" else None),
        quant_config=quant,
        per_act_token_quant=False,
        block_shape=None,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        moe_problem_size=Mock(side_effect=FallbackSelected),
    )
    hidden = torch.zeros(0 if case == "empty" else 2, 4)
    output = torch.empty_like(hidden)

    def run():
        apply(
            experts,
            output,
            hidden,
            torch.zeros(2, 8, 4),
            torch.zeros(2, 4, 4),
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int32),
            Activation.GELU if case == "gelu" else Activation.SILU,
            2,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
        )

    if case == "enabled":
        run()
        torch.testing.assert_close(output, torch.ones_like(output))
    elif case == "error":
        with pytest.raises(RuntimeError, match="provider failed"):
            run()
    else:
        with pytest.raises(FallbackSelected):
            run()
    assert provider.call_count == (1 if case in ("enabled", "error") else 0)


@pytest.mark.parametrize("tokens", [0, 2])
def test_qsa_uses_vendor_export_and_excludes_packed_count(monkeypatch, tokens):
    sparse, store = Mock(), Mock()
    package = ModuleType("flaggems_vllm")
    package.qsa_sparse_paged_attention = sparse
    ops = ModuleType("flaggems_vllm.ops.qwen4")
    ops.qsa_store_kv_cache_rows = store
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    forward = method(
        "vllm_fl/models/qwen4_exp/thead_qsa.py",
        "Qwen4ExpQSAFlashAttentionImpl",
        "forward_qsa",
        {"torch": torch},
    )
    query = torch.ones(2, 2, 4)
    output = torch.full_like(query, 9)
    indices = torch.tensor([[0, 1, 2], [1, -1, 1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=tokens,
        slot_mapping=torch.tensor([0, 1]),
        block_table=torch.tensor([[0]]),
    )
    forward(
        SimpleNamespace(head_size=4, scale=0.5),
        query,
        query,
        query,
        torch.zeros(1, 2, 2, 8),
        metadata,
        indices,
        torch.zeros(2, dtype=torch.int32),
        output,
        query,
    )
    assert not output.any()
    assert sparse.call_count == store.call_count == bool(tokens)
    if tokens:
        torch.testing.assert_close(sparse.call_args.args[3], indices[:, :-1])
        assert sparse.call_args.kwargs["softmax_scale"] == 0.5
