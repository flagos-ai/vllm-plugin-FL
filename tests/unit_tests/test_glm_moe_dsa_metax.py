# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from types import SimpleNamespace

import torch
from vllm import platforms

from vllm_fl import dispatch
from vllm_fl.ops.fused_moe import layer, router
from vllm_fl.ops.fused_moe.router import _metax_grouped_topk
from vllm_fl.patches import glm_moe_dsa_metax
from vllm_fl.platform import PlatformFL
from vllm_fl.patches.glm_moe_dsa_metax import (
    _load_int8_indexer_wk,
    _make_int8_moe_quant_config,
)


class _FusedWeight:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, tensor, shard_id):
        self.loaded = tensor, shard_id


def test_load_int8_indexer_weight_into_bf16_fused_projection():
    fused = _FusedWeight()
    params = {"model.layers.0.self_attn.indexer.wk_weights_proj.weight": fused}
    pending = {}
    loaded = set()

    weight = torch.tensor([[2, -3], [4, 1]], dtype=torch.int8)
    scale = torch.tensor([[0.5], [0.25]], dtype=torch.float32)

    assert _load_int8_indexer_wk(
        "model.layers.0.self_attn.indexer.wk.weight",
        weight,
        pending,
        params,
        loaded,
        set(),
    )
    assert _load_int8_indexer_wk(
        "model.layers.0.self_attn.indexer.wk.weight_scale",
        scale,
        pending,
        params,
        loaded,
        set(),
    )

    tensor, shard_id = fused.loaded
    assert shard_id == 0
    assert tensor.dtype == torch.bfloat16
    torch.testing.assert_close(tensor.float(), weight.float() * scale)
    assert loaded == {"model.layers.0.self_attn.indexer.wk_weights_proj.weight"}


def test_dynamic_activation_int8_moe_keeps_w8a8_without_static_scales():
    scale = torch.ones(2)
    config = _make_int8_moe_quant_config(
        scale,
        scale,
        per_act_token_quant=True,
    )

    assert config.use_int8_w8a8
    assert not config.use_int8_w8a16


def test_metax_grouped_topk_uses_bias_only_for_expert_selection():
    logits = torch.zeros(1, 4)
    correction_bias = torch.tensor([0.0, 0.0, 10.0, 9.0])

    weights, ids = _metax_grouped_topk(
        logits,
        topk=1,
        renormalize=False,
        num_expert_group=2,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        correction_bias=correction_bias,
    )

    assert ids.item() == 2
    assert weights.item() == 0.5


def test_non_metax_grouped_topk_keeps_existing_dispatch(monkeypatch):
    expected_weights = torch.tensor([[0.75]], dtype=torch.float32)
    expected_ids = torch.tensor([[3]], dtype=torch.int32)
    captured = {}

    def grouped_topk(*args):
        captured["args"] = args
        return expected_weights, expected_ids

    monkeypatch.setattr(
        router, "current_platform", SimpleNamespace(vendor_name="cuda")
    )
    monkeypatch.setattr(router, "_grouped_topk", grouped_topk)

    hidden_states = torch.zeros(1, 2)
    gating_output = torch.zeros(1, 4)
    correction_bias = torch.ones(4)
    weights, ids = router._fl_grouped_topk(
        hidden_states,
        gating_output,
        topk=1,
        renormalize=True,
        num_expert_group=2,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.0,
        e_score_correction_bias=correction_bias,
    )

    assert weights is expected_weights
    assert ids is expected_ids
    assert captured["args"] == (
        gating_output,
        2,
        1,
        1,
        True,
        2.0,
        correction_bias,
        1,
    )


def test_non_metax_model_entry_does_not_apply_metax_patches(monkeypatch):
    called = False

    def apply_model_patches():
        nonlocal called
        called = True

    importlib.import_module("vllm.model_executor.models.deepseek_v2")
    module_name = "vllm_fl.models.glm_moe_dsa"
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(vendor_name="cuda")
    )
    monkeypatch.setattr(
        glm_moe_dsa_metax, "apply_model_patches", apply_model_patches
    )
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)
    sys.modules.pop(module_name, None)

    assert not called


def test_non_metax_attention_backend_keeps_dispatch_path(monkeypatch):
    selected_backend = SimpleNamespace(
        get_path=lambda: "unexpected-selected-backend"
    )
    selector = SimpleNamespace(use_mla=True, use_sparse=True)
    monkeypatch.setattr(PlatformFL, "vendor_name", "cuda")
    monkeypatch.setattr(
        dispatch, "call_op", lambda *args, **kwargs: "existing-dispatch-backend"
    )

    backend = PlatformFL.get_attn_backend_cls(selected_backend, selector)

    assert backend == "existing-dispatch-backend"


def test_non_metax_quantized_moe_keeps_existing_fl_replacement(monkeypatch):
    class UnquantizedMethod:
        pass

    class Runner:
        moe_config = object()
        routed_experts = SimpleNamespace(quant_method=object())
        replacement = None

        def _replace_quant_method(self, quant_method):
            self.replacement = quant_method

    runner = Runner()
    replacement = object()
    monkeypatch.setattr(
        layer, "current_platform", SimpleNamespace(vendor_name="cuda")
    )
    monkeypatch.setattr(layer, "UnquantizedFusedMoEMethod", UnquantizedMethod)
    monkeypatch.setattr(
        layer, "UnquantizedFusedMoEMethodFL", lambda config: replacement
    )
    monkeypatch.setattr(layer, "_OrigFusedMoE", lambda *args, **kwargs: runner)
    monkeypatch.setattr(layer, "replace_router_with_fl", lambda: None)

    assert layer.FusedMoEFL() is runner
    assert runner.replacement is replacement
