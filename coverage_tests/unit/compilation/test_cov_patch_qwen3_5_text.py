# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the Qwen3.5 text-only registry installer.

The three vllm registries and ``ModelRegistry.register_model`` are the
external boundary; each test installs writable fakes on the vllm package
attributes the installer reads at call time and asserts the exact keys,
values, and registration paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# The shim import is load-bearing: it must precede vllm_fl imports.
# isort: split
from . import _flaggems_shim  # noqa: F401

# isort: split
import vllm_fl.patches.qwen3_5_text as patch_module


class _Fakes:
    """Fake vllm registries plus the model-registration call log."""

    def __init__(self):
        self.register_calls = []
        self.config_registry = {}
        self.models_config_map = {}
        self.text_models = {}
        self.vllm_models = {}
        self.transformers_config = SimpleNamespace(
            _CONFIG_REGISTRY=self.config_registry
        )
        self.model_config = SimpleNamespace(MODELS_CONFIG_MAP=self.models_config_map)
        self.model_registry = SimpleNamespace(
            _TEXT_GENERATION_MODELS=self.text_models,
            _VLLM_MODELS=self.vllm_models,
            ModelRegistry=SimpleNamespace(
                register_model=lambda arch, path: self.register_calls.append(
                    (arch, path)
                )
            ),
        )


@pytest.fixture
def fakes(monkeypatch):
    """Install the fake registries on the vllm package attributes."""
    import vllm.model_executor.models as models_pkg
    import vllm.transformers_utils as transformers_utils_pkg

    state = _Fakes()
    monkeypatch.setattr(models_pkg, "config", state.model_config, raising=False)
    monkeypatch.setattr(models_pkg, "registry", state.model_registry, raising=False)
    monkeypatch.setattr(
        transformers_utils_pkg, "config", state.transformers_config, raising=False
    )
    return state


def test_install_registers_text_configs_models_and_verifiers(fakes):
    """All three registries receive the canonical text-only entries."""
    assert patch_module.apply_qwen3_5_text_patches() is True

    verifier = patch_module.Qwen3_5ForConditionalGenerationConfig
    assert fakes.config_registry == {
        "qwen3_5_text": "Qwen3_5TextConfig",
        "qwen3_5_moe_text": "Qwen3_5MoeTextConfig",
    }
    assert fakes.models_config_map == {
        "Qwen3_5ForCausalLM": verifier,
        "Qwen3_5MoeForCausalLM": verifier,
    }
    assert fakes.text_models == {
        "Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),
        "Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),
    }
    assert fakes.vllm_models == fakes.text_models
    assert fakes.register_calls == [
        ("Qwen3_5ForCausalLM", "vllm_fl.models.qwen3_5:Qwen3_5ForCausalLM"),
        ("Qwen3_5MoeForCausalLM", "vllm_fl.models.qwen3_5:Qwen3_5MoeForCausalLM"),
    ]


def test_install_is_repeatable_and_preserves_existing_entries(fakes):
    """setdefault keeps pre-registered values; lazy registration repeats."""
    fakes.config_registry["qwen3_5_text"] = "PRESEEDED"
    fakes.text_models["Qwen3_5ForCausalLM"] = ("other", "OtherCausalLM")
    fakes.vllm_models["Qwen3_5MoeForCausalLM"] = ("other", "OtherMoe")

    assert patch_module.apply_qwen3_5_text_patches() is True
    assert patch_module.apply_qwen3_5_text_patches() is True

    assert fakes.config_registry["qwen3_5_text"] == "PRESEEDED"
    assert fakes.config_registry["qwen3_5_moe_text"] == "Qwen3_5MoeTextConfig"
    assert fakes.text_models["Qwen3_5ForCausalLM"] == ("other", "OtherCausalLM")
    assert fakes.text_models["Qwen3_5MoeForCausalLM"] == (
        "qwen3_5",
        "Qwen3_5MoeForCausalLM",
    )
    assert fakes.vllm_models["Qwen3_5MoeForCausalLM"] == ("other", "OtherMoe")
    assert fakes.vllm_models["Qwen3_5ForCausalLM"] == (
        "qwen3_5",
        "Qwen3_5ForCausalLM",
    )
    # Lazy ModelRegistry registration is intentionally repeated per call.
    assert len(fakes.register_calls) == 4


def test_install_returns_true_on_every_call(fakes):
    """The installer reports success rather than tracking a module marker."""
    assert patch_module.apply_qwen3_5_text_patches() is True
    assert patch_module.apply_qwen3_5_text_patches() is True
