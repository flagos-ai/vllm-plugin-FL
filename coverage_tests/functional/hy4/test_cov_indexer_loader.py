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

"""Dequantization and load-time pairing of the HY4 indexer ``wk`` weights.

The FP8/MXFP8 upcast helpers are compared against a reference computed
with plain PyTorch loops, ``IndexerWKLoader`` pairs each quantized
``wk`` weight with its scale before writing the fused BF16 parameter
through the parameter's own weight loader, and ``HYV4Config``
validation plus the ``rope_theta`` bridge round-trip are exercised.
"""

import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.hy4.config import HYV4Config
from vllm_fl.models.hy4.indexer import (
    IndexerWKLoader,
    dequantize_fp8_wk,
    dequantize_mxfp8_wk,
    normalize_quant_format,
    quant_metadata_from_quant_config,
)

pytestmark = [pytest.mark.gpu]


def _tiny_config(**overrides):
    """A dimension-shrunk HYV4Config that still validates."""
    overrides.setdefault("vocab_size", 32)
    overrides.setdefault("hidden_size", 8)
    overrides.setdefault("num_hidden_layers", 2)
    overrides.setdefault("qk_nope_head_dim", 4)
    overrides.setdefault("qk_rope_head_dim", 4)
    overrides.setdefault("qk_head_dim", 8)
    overrides.setdefault("v_head_dim", 4)
    overrides.setdefault("q_lora_rank", 4)
    overrides.setdefault("kv_lora_rank", 6)
    overrides.setdefault("n_routed_experts", 2)
    overrides.setdefault("indexer_types", ["full", "full"])
    return HYV4Config(**overrides)


def _reference_fp8_block_dequant(weight, scale):
    """Plain-loop block dequantization matching vLLM's grouped semantics."""
    rows, cols = weight.shape
    block = cols // scale.shape[-1]
    out = torch.empty(rows, cols, dtype=torch.float32, device=weight.device)
    for r in range(scale.shape[-2]):
        for c in range(scale.shape[-1]):
            block_w = weight.float()[
                r * block : (r + 1) * block, c * block : (c + 1) * block
            ]
            out[r * block : (r + 1) * block, c * block : (c + 1) * block] = (
                block_w * scale[r, c].item()
            )
    return out


class TestIndexerWKDequant:
    """FP8/MXFP8 upcast of the indexer ``wk`` projection."""

    def _fp8_pair(self, device):
        torch.manual_seed(5)
        weight = (torch.randn(8, 128, device=device) * 20).to(torch.float8_e4m3fn)
        scale = torch.rand(2, 32, device=device) * 0.05 + 0.01
        return weight, scale

    def test_fp8_dequant_matches_block_reference(self, device):
        """FP8 dequantization matches block reference."""
        weight, scale = self._fp8_pair(device)
        out = dequantize_fp8_wk(weight, scale)
        reference = _reference_fp8_block_dequant(weight, scale)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out.float(), reference, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "weight,scale,match",
        [
            (torch.zeros(8), torch.zeros(2, 2), "2D"),
            (
                torch.zeros(8, 128, dtype=torch.float8_e4m3fn),
                torch.zeros(2, 33),
                "divide",
            ),
            (
                torch.zeros(8, 128, dtype=torch.float8_e4m3fn),
                torch.zeros(3, 32),
                "rows",
            ),
        ],
    )
    def test_fp8_shape_violations_are_rejected(self, device, weight, scale, match):
        """FP8 invalid shapes are rejected."""
        weight = weight.to(device)
        scale = scale.to(device)
        with pytest.raises(ValueError, match=match):
            dequantize_fp8_wk(weight, scale)

    def test_mxfp8_dequant_uses_e8m0_scales(self, device):
        # An e8m0 scale byte of 127 means descale 2**0 = 1.
        """MXFP8 dequantization uses E8M0 scales."""
        weight = (torch.randn(4, 64, device=device) * 3).to(torch.float8_e4m3fn)
        scale = torch.full((4, 2), 127, dtype=torch.uint8, device=device)
        out = dequantize_mxfp8_wk(weight, scale)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out.float(), weight.float(), rtol=1e-2, atol=1e-2)

    def test_mxfp8_shape_violations_are_rejected(self, device):
        """MXFP8 invalid shapes are rejected."""
        weight = torch.zeros(4, 64, dtype=torch.float8_e4m3fn, device=device)
        with pytest.raises(ValueError, match="group size"):
            dequantize_mxfp8_wk(
                weight, torch.zeros(4, 3, dtype=torch.uint8, device=device)
            )
        with pytest.raises(ValueError, match="MXFP8 shape mismatch"):
            dequantize_mxfp8_wk(
                weight, torch.zeros(5, 2, dtype=torch.uint8, device=device)
            )

    def test_quant_format_labels(self):
        """Map checkpoint dtypes to their supported quantization formats."""
        assert normalize_quant_format("ModelOptFP8") == "fp8"
        assert normalize_quant_format("checkpoint-mxfp8") == "mxfp8"
        assert normalize_quant_format(None) is None
        assert normalize_quant_format("other") is None

        class Quant:
            def get_name(self):
                return "fbgemm-mxfp8"

        assert quant_metadata_from_quant_config(Quant()) == "mxfp8"
        assert quant_metadata_from_quant_config(None) is None

        class NoName:
            pass

        assert quant_metadata_from_quant_config(NoName()) is None


class TestIndexerWKLoader:
    """Pairing of ``wk`` weights with their scales at load time."""

    FUSED = "model.layers.0.self_attn.indexer.wk_weights_proj.weight"

    def _loader(self, device, quant_metadata="fp8", pp_missing=()):
        written = []

        class Param:
            weight = torch.zeros(8, 128, dtype=torch.bfloat16, device=device)

            @staticmethod
            def weight_loader(param, tensor, shard_id):
                written.append((tensor.clone(), shard_id))

        params = {self.FUSED: Param()}
        loader = IndexerWKLoader(params, pp_missing, quant_metadata)
        return loader, written

    def test_fp8_pair_dequantizes_into_fused_parameter(self, device):
        """FP8 pair dequantizes into fused parameter."""
        loader, written = self._loader(device)
        weight, scale = TestIndexerWKDequant()._fp8_pair(device)
        assert (
            loader.consume(
                "model.layers.0.self_attn.indexer.wq_b.weight", torch.zeros(1)
            )
            is False
        )
        assert (
            loader.consume("model.layers.0.self_attn.indexer.wk.weight", weight) is True
        )
        # A weight alone is not enough to finalize.
        assert written == []
        assert (
            loader.consume(
                "model.layers.0.self_attn.indexer.wk.weight_scale_inv", scale
            )
            is True
        )
        assert len(written) == 1
        reference = _reference_fp8_block_dequant(weight, scale)
        torch.testing.assert_close(
            written[0][0].float(), reference, rtol=1e-2, atol=1e-2
        )
        assert written[0][1] == 0
        assert loader.finish() == {self.FUSED}

    def test_unquantized_weight_is_left_to_the_stacked_loader(self, device):
        """Unquantized weight is left to the stacked loader."""
        loader, written = self._loader(device)
        plain = torch.randn(8, 128, dtype=torch.bfloat16, device=device)
        assert (
            loader.consume("model.layers.0.self_attn.indexer.wk.weight", plain) is False
        )
        assert written == []
        assert loader.finish() == set()

    def test_format_conflict_against_checkpoint_metadata_is_rejected(self, device):
        """Format conflict against checkpoint metadata is rejected."""
        loader, _ = self._loader(device, quant_metadata="mxfp8")
        weight, scale = TestIndexerWKDequant()._fp8_pair(device)
        loader.consume("model.layers.0.self_attn.indexer.wk.weight", weight)
        with pytest.raises(ValueError, match="format conflict"):
            loader.consume(
                "model.layers.0.self_attn.indexer.wk.weight_scale_inv", scale
            )

    def test_incomplete_pair_fails_at_finish(self, device):
        """Incomplete pair fails at finish."""
        loader, _ = self._loader(device)
        weight, _scale = TestIndexerWKDequant()._fp8_pair(device)
        loader.consume("model.layers.0.self_attn.indexer.wk.weight", weight)
        with pytest.raises(ValueError, match="missing scale"):
            loader.finish()

    def test_pp_missing_layers_are_swallowed_without_pairing(self, device):
        # A pipeline rank owns neither the parameter nor the checkpoint
        # shard, so foreign-layer tensors are consumed without validation.
        """Ignore checkpoint weights owned by another pipeline stage."""
        loader, written = self._loader(
            device, pp_missing=("model.layers.9.self_attn.indexer",)
        )
        weight, _ = TestIndexerWKDequant()._fp8_pair(device)
        assert (
            loader.consume("model.layers.9.self_attn.indexer.wk.weight", weight) is True
        )
        assert written == []
        assert loader.finish() == set()

    def test_missing_fused_target_is_rejected(self, device):
        """Missing fused target is rejected."""
        loader = IndexerWKLoader({}, ())
        weight, scale = TestIndexerWKDequant()._fp8_pair(device)
        loader.consume("model.layers.0.self_attn.indexer.wk.weight", weight)
        with pytest.raises(ValueError, match="target parameter is missing"):
            loader.consume(
                "model.layers.0.self_attn.indexer.wk.weight_scale_inv", scale
            )

    def test_unsupported_weight_dtype_is_rejected(self, device):
        """Unsupported weight dtype is rejected."""
        loader, _ = self._loader(device)
        with pytest.raises(ValueError, match="unsupported dtype"):
            loader.consume(
                "model.layers.0.self_attn.indexer.wk.weight",
                torch.zeros(8, 128, dtype=torch.int8, device=device),
            )

    def test_scale_dtypes_are_validated(self, device):
        """Scale dtypes are validated."""
        loader, _ = self._loader(device)
        with pytest.raises(ValueError, match="FP8 scale must be a floating"):
            loader.consume(
                "model.layers.0.self_attn.indexer.wk.weight_scale_inv",
                torch.zeros(2, 32, dtype=torch.int32, device=device),
            )
        with pytest.raises(ValueError, match="MXFP8 scale must be"):
            loader.consume(
                "model.layers.0.self_attn.indexer.wk.weight_scale",
                torch.zeros(2, 32, dtype=torch.float32, device=device),
            )

    def test_plain_weight_after_quantized_input_is_rejected(self, device):
        """Plain weight after quantized input is rejected."""
        loader, _ = self._loader(device)
        weight, scale = TestIndexerWKDequant()._fp8_pair(device)
        loader.consume("model.layers.0.self_attn.indexer.wk.weight", weight)
        loader.consume("model.layers.0.self_attn.indexer.wk.weight_scale_inv", scale)
        plain = torch.randn(8, 128, dtype=torch.bfloat16, device=device)
        with pytest.raises(ValueError, match="already loaded"):
            loader.consume("model.layers.0.self_attn.indexer.wk.weight", plain)


class TestHyv4Config:
    """HYV4Config validation and the legacy RoPE bridge."""

    def test_defaults_validate_and_rope_theta_round_trips(self):
        """Defaults validate and rope theta round trips."""
        config = _tiny_config()
        assert config.qk_head_dim == 8
        assert config.rope_theta == 10_000_000.0
        config.rope_theta = 500_000.0
        assert config.rope_parameters["rope_theta"] == 500_000.0
        assert config.rope_theta == 500_000.0

    @pytest.mark.parametrize(
        "overrides,match",
        [
            (dict(num_attention_heads=0), "num_attention_heads"),
            (dict(n_routed_experts=0), "n_routed_experts"),
            (dict(qk_head_dim=9), "qk_head_dim"),
            (dict(hc_mult=0), "hc_mult"),
            (dict(layer_types=["sparse"]), "layer_types"),
            (dict(mlp_layer_types=["dense"] * 3), "mlp_layer_types"),
            (dict(indexer_types=["full"]), "indexer_types"),
        ],
    )
    def test_invalid_dimensions_are_rejected(self, overrides, match):
        """Invalid dimensions are rejected."""
        with pytest.raises(ValueError, match=match):
            _tiny_config(**overrides)

    def test_multimodal_keys_are_rejected(self):
        """Multimodal keys are rejected."""
        with pytest.raises(ValueError, match="text-only"):
            _tiny_config(vision_config={"width": 224})

    def test_model_type_and_capability_flags(self):
        """Model type and capability flags."""
        config = _tiny_config()
        assert config.model_type == "hy_v4"
        assert config.supports_multimodal is False
