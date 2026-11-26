"""CPU lifecycle coverage for single-resident gate/up weights."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.patches import single_weight_gateup as patch


def _gate(fail_scale_registration=False):
    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter(
                "weight", torch.nn.Parameter(torch.ones((8, 4), dtype=torch.bfloat16))
            )
            self.register_buffer(
                "fl_prefill_gateup_weight_scale", torch.ones(8), persistent=False
            )

        def register_buffer(self, name, tensor, persistent=True):
            if fail_scale_registration and name == patch.SCALE_BUFFER:
                raise RuntimeError("injected scale-buffer registration failure")
            return super().register_buffer(name, tensor, persistent=persistent)

    return Gate()


def _target(layer, fail_scale_registration=False):
    gate = _gate(fail_scale_registration)
    return {
        "module": SimpleNamespace(),
        "gate": gate,
        "weight": gate.weight,
        "scale": gate.fl_prefill_gateup_weight_scale,
        "layer": layer,
    }


def test_commit_replaces_one_bf16_parameter_and_reuses_scale_storage():
    target = _target(3)
    weight = target["weight"]
    scale = target["scale"]
    qweight = torch.ones_like(weight, dtype=torch.int8)

    q_bytes, scale_bytes, delta = patch._attach_and_remove_weight(
        target, qweight, scale
    )

    gate = target["gate"]
    assert "weight" not in gate._parameters
    assert not any(name.endswith("weight") for name, _ in gate.named_parameters())
    assert getattr(gate, patch.QWEIGHT_BUFFER) is qweight
    assert getattr(gate, patch.SCALE_BUFFER) is scale
    assert patch.QWEIGHT_BUFFER in gate._non_persistent_buffers_set
    assert patch.SCALE_BUFFER in gate._non_persistent_buffers_set
    assert "fl_prefill_gateup_weight_scale" not in gate._buffers
    assert target["weight"] is None
    assert q_bytes == qweight.numel()
    assert scale_bytes == scale.numel() * scale.element_size()
    assert delta == q_bytes - weight.numel() * weight.element_size()


def test_buffer_registration_failure_rolls_back_before_parameter_commit():
    target = _target(0, fail_scale_registration=True)
    original_weight = target["gate"].weight
    original_scale = target["scale"]
    qweight = torch.ones_like(original_weight, dtype=torch.int8)

    with pytest.raises(RuntimeError, match="injected scale-buffer"):
        patch._attach_and_remove_weight(target, qweight, original_scale)

    gate = target["gate"]
    assert gate.weight is original_weight
    assert gate.fl_prefill_gateup_weight_scale is original_scale
    assert patch.QWEIGHT_BUFFER not in gate._buffers
    assert patch.SCALE_BUFFER not in gate._buffers


def test_finalize_commits_selected_layers_but_leaves_failed_layers_bf16(monkeypatch):
    selected = _target(0)
    unchanged = _target(1)
    targets = [selected, unchanged]
    monkeypatch.setattr(patch, "_installed", True)
    monkeypatch.setattr(patch, "_converted_layers", 0)
    monkeypatch.setattr(patch, "_targets", lambda *_args: targets)
    monkeypatch.setattr(patch, "_qualify", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(patch, "_source_hash", lambda *_args: "test-hash")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = []

    def candidate(target):
        calls.append(target["layer"])
        if target is unchanged:
            raise ValueError("injected pre-commit numeric failure")
        return (
            torch.ones_like(target["weight"], dtype=torch.int8),
            target["scale"],
            None,
        )

    monkeypatch.setattr(patch, "_layer_candidate", candidate)
    config = SimpleNamespace(additional_config={})
    stats = patch.finalize(None, "cpu", config, model_memory_usage=4096)

    assert stats["layers"] == 1
    assert calls == [0, 1]
    assert "weight" not in selected["gate"]._parameters
    assert patch.QWEIGHT_BUFFER in selected["gate"]._buffers
    assert selected["weight"] is None
    assert unchanged["gate"].weight is unchanged["weight"]
    assert "fl_prefill_gateup_weight_scale" in unchanged["gate"]._buffers
    assert patch.QWEIGHT_BUFFER not in unchanged["gate"]._buffers
    assert config.additional_config["fl_single_weight_gateup_layers"] == 1
    assert config.additional_config["fl_single_weight_gateup_hash"] == "test-hash"
    assert patch.has_converted_layers()
