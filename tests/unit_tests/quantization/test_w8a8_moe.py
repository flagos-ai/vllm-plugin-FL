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

import sys
from types import SimpleNamespace

from vllm_fl.quantization.w8a8 import moe as moe_adapter


def test_w8a8_moe_selector_patches_oracle_and_scheme(monkeypatch):
    def upstream_selector(*args, **kwargs):
        return "upstream"

    def upstream_builder(*args, **kwargs):
        return "upstream-config"

    oracle = SimpleNamespace(
        select_int8_moe_backend=upstream_selector,
    )
    scheme = SimpleNamespace(
        select_int8_moe_backend=upstream_selector,
        make_int8_moe_quant_config=upstream_builder,
    )
    modules = {
        moe_adapter._ORACLE_MODULE: oracle,
        moe_adapter._SCHEME_MODULE: scheme,
    }
    monkeypatch.setattr(
        moe_adapter,
        "import_module",
        lambda name: modules[name],
    )

    assert moe_adapter.install_fl_w8a8_moe_selector() is True
    assert oracle.select_int8_moe_backend is scheme.select_int8_moe_backend
    assert getattr(
        oracle.select_int8_moe_backend,
        moe_adapter._ADAPTER_MARKER,
    )
    assert getattr(
        scheme.make_int8_moe_quant_config,
        moe_adapter._CONFIG_BUILDER_MARKER,
    )

    config_module = SimpleNamespace(
        int8_w8a8_moe_quant_config=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.config",
        config_module,
    )
    dynamic_config = scheme.make_int8_moe_quant_config(
        w1_scale="w1",
        w2_scale="w2",
        a1_scale=None,
        a2_scale=None,
        per_act_token_quant=True,
    )
    assert dynamic_config == {
        "w1_scale": "w1",
        "w2_scale": "w2",
        "a1_scale": None,
        "a2_scale": None,
        "per_act_token_quant": True,
    }
    assert (
        scheme.make_int8_moe_quant_config(
            w1_scale="w1",
            w2_scale="w2",
            per_act_token_quant=False,
        )
        == "upstream-config"
    )


def test_w8a8_moe_selector_install_is_idempotent(monkeypatch):
    def installed_selector(*args, **kwargs):
        return "installed"

    setattr(installed_selector, moe_adapter._ADAPTER_MARKER, True)

    def installed_builder(*args, **kwargs):
        return "installed-config"

    setattr(
        installed_builder,
        moe_adapter._CONFIG_BUILDER_MARKER,
        True,
    )
    oracle = SimpleNamespace(
        select_int8_moe_backend=installed_selector,
    )
    scheme = SimpleNamespace(
        select_int8_moe_backend=None,
        make_int8_moe_quant_config=installed_builder,
    )
    modules = {
        moe_adapter._ORACLE_MODULE: oracle,
        moe_adapter._SCHEME_MODULE: scheme,
    }
    monkeypatch.setattr(
        moe_adapter,
        "import_module",
        lambda name: modules[name],
    )

    assert moe_adapter.install_fl_w8a8_moe_selector() is True
    assert scheme.select_int8_moe_backend is installed_selector
