# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

from vllm_fl.ops.fused_moe import layer as layer_module


class _FakeRunner:
    def __init__(self, quant_method):
        self.routed_experts = SimpleNamespace(quant_method=quant_method)
        self.moe_config = SimpleNamespace()
        self.replacements = []

    def _replace_quant_method(self, quant_method):
        self.replacements.append(quant_method)
        self.routed_experts.quant_method = quant_method


def test_fused_moe_factory_preserves_quantized_method(monkeypatch):
    quantized_method = object()
    runner = _FakeRunner(quantized_method)
    monkeypatch.setattr(
        layer_module,
        "_OrigFusedMoEFactory",
        lambda *args, **kwargs: runner,
    )
    monkeypatch.setattr(layer_module, "replace_router_with_fl", lambda: None)

    result = layer_module.FusedMoEFactoryFL()

    assert result is runner
    assert runner.routed_experts.quant_method is quantized_method
    assert runner.replacements == []


def test_fused_moe_factory_still_replaces_unquantized_method(monkeypatch):
    upstream_method = object.__new__(layer_module.UnquantizedFusedMoEMethod)
    fl_method = object()
    runner = _FakeRunner(upstream_method)
    monkeypatch.setattr(
        layer_module,
        "_OrigFusedMoEFactory",
        lambda *args, **kwargs: runner,
    )
    monkeypatch.setattr(
        layer_module,
        "UnquantizedFusedMoEMethodFL",
        lambda moe_config: fl_method,
    )
    monkeypatch.setattr(layer_module, "replace_router_with_fl", lambda: None)

    layer_module.FusedMoEFactoryFL()

    assert runner.routed_experts.quant_method is fl_method
    assert runner.replacements == [fl_method]
