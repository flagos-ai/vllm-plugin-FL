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

"""Supplementary coverage for the ARM packed-W4A8 installer front half.

Exercises ``install_arm_cpu_packed_w4a8`` itself and the two closures that
run at scheme-construction time:

* first install patches all five vLLM entry points and records the captured
  originals, while a second install is a no-op returning ``False``;
* the partial-install guard raises ``RuntimeError`` when only the scheme or
  only the dynamic-4bit kernel carries an install marker, and leaves the
  classes untouched;
* ``scheme_init`` forwards the five stock arguments to the original
  constructor and stores the checkpoint-packed flag plus the 32 / num_bits
  packing factor;
* ``_get_scheme_from_parts`` returns a packed scheme only for the
  pack-quantized format with a dynamic token W4A8 input, and otherwise
  delegates to the captured original with the caller's arguments verbatim.
"""

from types import SimpleNamespace

import pytest

from ._vllm_fakes import build_environment, injected_modules, make_scheme
from vllm_fl.quantization.arm_cpu_w4a8 import install_arm_cpu_packed_w4a8


@pytest.fixture
def raw_env():
    """Fresh fake boundary without running the adapter installer."""
    env = build_environment()
    with injected_modules(env):
        yield env


@pytest.fixture
def fake_env(raw_env):
    """Fresh fake boundary with the adapter installed against it."""
    assert install_arm_cpu_packed_w4a8() is True
    return raw_env


class TestInstall:
    """Install bookkeeping, idempotency and the partial-install guard."""

    def test_first_install_patches_and_records_originals(self, raw_env):
        """Verifies the first install patches all five entry points and records the originals."""
        scheme_cls = raw_env.CompressedTensorsW4A8Int
        config_cls = raw_env.CompressedTensorsConfig
        kernel_cls = raw_env.Dynamic4bitLinearKernel
        original_init = scheme_cls.__init__
        original_create = scheme_cls.create_weights
        original_get_scheme = config_cls._get_scheme_from_parts
        original_process = kernel_cls.process_weights_after_loading
        original_apply = kernel_cls.apply_weights

        assert install_arm_cpu_packed_w4a8() is True

        assert scheme_cls.__init__ is not original_init
        assert scheme_cls.create_weights is not original_create
        assert config_cls._get_scheme_from_parts is not original_get_scheme
        assert kernel_cls.process_weights_after_loading is not original_process
        assert kernel_cls.apply_weights is not original_apply

        assert scheme_cls._vllm_fl_arm_packed_w4a8 is True
        assert kernel_cls._vllm_fl_arm_w4a8 is True
        assert scheme_cls._vllm_fl_arm_original_init is original_init
        assert scheme_cls._vllm_fl_arm_original_create_weights is original_create
        assert config_cls._vllm_fl_arm_original_get_scheme is original_get_scheme
        assert kernel_cls._vllm_fl_arm_original_process is original_process
        assert kernel_cls._vllm_fl_arm_original_apply is original_apply

    def test_second_install_returns_false_and_keeps_patches(self, raw_env):
        """Verifies a second install is a no-op that returns False and keeps patches."""
        scheme_cls = raw_env.CompressedTensorsW4A8Int
        assert install_arm_cpu_packed_w4a8() is True
        patched_init = scheme_cls.__init__
        patched_create = scheme_cls.create_weights

        assert install_arm_cpu_packed_w4a8() is False
        assert scheme_cls.__init__ is patched_init
        assert scheme_cls.create_weights is patched_create

    def test_scheme_marker_without_kernel_marker_raises(self, raw_env):
        """Verifies a scheme-only install marker raises RuntimeError and leaves classes untouched."""
        scheme_cls = raw_env.CompressedTensorsW4A8Int
        kernel_cls = raw_env.Dynamic4bitLinearKernel
        scheme_cls._vllm_fl_arm_packed_w4a8 = True
        original_init = scheme_cls.__init__

        with pytest.raises(
            RuntimeError, match="incomplete ARM W4A8 runtime installation"
        ):
            install_arm_cpu_packed_w4a8()

        assert scheme_cls.__init__ is original_init
        assert not hasattr(kernel_cls, "_vllm_fl_arm_w4a8")

    def test_kernel_marker_without_scheme_marker_raises(self, raw_env):
        """Verifies a kernel-only install marker raises RuntimeError and leaves classes untouched."""
        scheme_cls = raw_env.CompressedTensorsW4A8Int
        kernel_cls = raw_env.Dynamic4bitLinearKernel
        kernel_cls._vllm_fl_arm_w4a8 = True
        original_apply = kernel_cls.apply_weights

        with pytest.raises(
            RuntimeError, match="incomplete ARM W4A8 runtime installation"
        ):
            install_arm_cpu_packed_w4a8()

        assert kernel_cls.apply_weights is original_apply
        assert not hasattr(scheme_cls, "_vllm_fl_arm_packed_w4a8")


class TestSchemeInit:
    """The patched scheme constructor and the metadata it stores."""

    def test_packed_true_stores_pack_factor_for_four_bits(self, fake_env):
        """Verifies packed four-bit schemes store the checkpoint flag and pack factor 8."""
        scheme = make_scheme(fake_env, packed=True, num_bits=4)

        assert scheme._vllm_fl_checkpoint_packed is True
        assert scheme._vllm_fl_pack_factor == 8
        assert scheme.num_bits == 4
        assert scheme.group_size == 128
        assert fake_env.original_init_calls == [
            {
                "strategy": "group",
                "num_bits": 4,
                "group_size": 128,
                "is_static_input_scheme": False,
                "input_symmetric": True,
            }
        ]

    def test_eight_bits_halves_the_pack_factor(self, fake_env):
        """Verifies eight-bit schemes record a pack factor of 4."""
        scheme = make_scheme(fake_env, packed=False, num_bits=8)

        assert scheme._vllm_fl_checkpoint_packed is False
        assert scheme._vllm_fl_pack_factor == 4
        assert fake_env.original_init_calls[0]["num_bits"] == 8


def _quant_args(**overrides):
    values = {"num_bits": 4, "strategy": "group", "group_size": 128}
    values.update(overrides)
    return SimpleNamespace(**values)


class TestGetSchemeFromParts:
    """Routing between the packed adapter scheme and the stock scheme."""

    def _config(self, env, quant_format):
        config = env.CompressedTensorsConfig()
        config.quant_format = quant_format
        return config

    def test_pack_format_dynamic_token_returns_packed_scheme(self, fake_env):
        """Verifies pack-quantized dynamic-token input returns the packed adapter scheme."""
        fake_env.dynamic_result = True
        config = self._config(fake_env, "pack-quantized")
        weight_quant = _quant_args()
        input_quant = SimpleNamespace(symmetric=True)

        scheme = config._get_scheme_from_parts(
            weight_quant,
            input_quant,
            format="pack-quantized",
            layer_name="model.layers.0.in_proj",
        )

        assert isinstance(scheme, fake_env.CompressedTensorsW4A8Int)
        assert scheme._vllm_fl_checkpoint_packed is True
        assert scheme._vllm_fl_pack_factor == 8
        assert scheme.strategy == "group"
        assert scheme.group_size == 128
        assert scheme.is_static_input_scheme is False
        assert scheme.input_symmetric is True
        assert fake_env.dynamic_token_calls == [(weight_quant, input_quant)]
        assert fake_env.original_get_scheme_calls == []

    def test_none_format_falls_back_to_config_quant_format(self, fake_env):
        """Verifies a missing format falls back to the config's quant format."""
        fake_env.dynamic_result = True
        config = self._config(fake_env, "pack-quantized")

        scheme = config._get_scheme_from_parts(
            _quant_args(), SimpleNamespace(symmetric=False)
        )

        assert isinstance(scheme, fake_env.CompressedTensorsW4A8Int)
        assert scheme._vllm_fl_checkpoint_packed is True
        assert scheme.input_symmetric is False
        assert len(fake_env.dynamic_token_calls) == 1

    def test_non_pack_format_delegates_without_dynamic_probe(self, fake_env):
        """Verifies a non-pack format delegates without probing dynamic tokens."""
        fake_env.dynamic_result = True
        fake_env.original_get_scheme_result = "stock-scheme"
        config = self._config(fake_env, "int-quantized")
        weight_quant = _quant_args()
        input_quant = SimpleNamespace(symmetric=True)

        result = config._get_scheme_from_parts(
            weight_quant, input_quant, format=None, layer_name="layer"
        )

        assert result == "stock-scheme"
        assert fake_env.dynamic_token_calls == []
        assert fake_env.original_get_scheme_calls == [
            {
                "weight_quant": weight_quant,
                "input_quant": input_quant,
                "output_quant": None,
                "format": None,
                "layer_name": "layer",
            }
        ]

    def test_pack_format_without_dynamic_token_delegates(self, fake_env):
        """Verifies pack format without a dynamic token input delegates to the original."""
        fake_env.dynamic_result = False
        fake_env.original_get_scheme_result = "stock-scheme"
        config = self._config(fake_env, "pack-quantized")

        result = config._get_scheme_from_parts(
            _quant_args(), SimpleNamespace(symmetric=True), format="pack-quantized"
        )

        assert result == "stock-scheme"
        assert len(fake_env.dynamic_token_calls) == 1
        assert fake_env.original_get_scheme_calls[0]["format"] == "pack-quantized"

    def test_delegate_forwards_output_quant_and_format(self, fake_env):
        """Verifies delegation forwards output_quant, format and layer_name verbatim."""
        fake_env.original_get_scheme_result = "stock-scheme"
        config = self._config(fake_env, "int-quantized")
        output_quant = object()

        result = config._get_scheme_from_parts(
            _quant_args(),
            SimpleNamespace(symmetric=True),
            output_quant=output_quant,
            format="fp8",
            layer_name="linear",
        )

        assert result == "stock-scheme"
        call = fake_env.original_get_scheme_calls[0]
        assert call["output_quant"] is output_quant
        assert call["format"] == "fp8"
        assert call["layer_name"] == "linear"
