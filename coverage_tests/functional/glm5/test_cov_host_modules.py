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

"""Single-GPU tests for the glm5 host-side startup modules (M1).

Covers ``config.py`` (transformers schema classes), ``provider.py``
(capability probes), ``loader.py`` (checkpoint shard audit with full
restore-on-raise verification) and ``runtime.py`` (startup validation
guards). No engine is started; every contract is exercised with plain
objects.
"""

from types import SimpleNamespace

import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5.config import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from vllm_fl.models.glm5.loader import (
    _required_shards,
    audit_packed_weights,
    audit_text_weights,
    unquantized_weights,
)
from vllm_fl.models.glm5.runtime import Glm5ConfigError, validate_glm5_config

pytestmark = [pytest.mark.gpu]


def _text_config(layer_types, **overrides):
    kwargs = dict(num_hidden_layers=len(layer_types), layer_types=layer_types)
    kwargs.update(overrides)
    return Glm5NextTextConfig(**kwargs)


class TestGlm5NextTextConfig:
    """Layer-type predicates and the sparse-alias validation view."""

    LAYERS = ["full_attention", "linear_attention", "full_attention"]

    def test_layer_predicates_follow_layer_types(self):
        """Layer predicates follow layer types."""
        config = _text_config(self.LAYERS)
        assert config.is_moe is True  # default n_routed_experts=288
        assert config.is_linear_attn is True
        assert config.is_kda_layer(0) is False
        assert config.is_kda_layer(1) is True
        # Only linear_attention keeps its name; attention kinds collapse.
        assert config.layers_block_type == [
            "attention",
            "linear_attention",
            "attention",
        ]

    def test_all_attention_config_is_not_linear(self):
        """All attention config is not linear."""
        config = _text_config(["full_attention"] * 3, n_routed_experts=None)
        assert config.is_moe is False
        assert config.is_linear_attn is False
        assert config.layers_block_type == ["attention"] * 3

    @pytest.mark.parametrize(
        ("allowed", "expected"),
        [
            (("attention", "sparse"), "sparse"),
            (("attention", "deepseek_sparse_attention"), "deepseek_sparse_attention"),
        ],
    )
    def test_sparse_alias_respects_upstream_allow_list(
        self, monkeypatch, allowed, expected
    ):
        """Sparse alias respects upstream allow list."""
        from transformers import configuration_utils

        monkeypatch.setattr(
            configuration_utils, "ALLOWED_ATTN_LAYER_TYPES", allowed, raising=False
        )
        kinds = ["deepseek_sparse_attention", "attention"]
        assert Glm5NextTextConfig._validation_layer_types(kinds) == [
            expected,
            "attention",
        ]


class TestGlm5NextConfig:
    """Composite config: nested construction and text delegation."""

    @staticmethod
    def _build():
        return Glm5NextConfig(
            text_config=dict(num_hidden_layers=2, layer_types=["full_attention"] * 2),
            vision_config=dict(depth=5),
        )

    def test_dict_sub_configs_are_built(self):
        """Dict subconfigurations are built."""
        config = self._build()
        assert isinstance(config.text_config, Glm5NextTextConfig)
        assert isinstance(config.vision_config, Glm5NextVisionConfig)
        assert config.vision_config.depth == 5
        assert config.image_token_id == 154854
        assert config.video_token_id == 154855

    def test_get_text_config_returns_nested_config(self):
        """Return the text subconfiguration from the composite config."""
        config = self._build()
        assert config.get_text_config() is config.text_config

    def test_unknown_attributes_delegate_then_raise(self):
        """Unknown attributes delegate then raise."""
        config = self._build()
        # Delegated text attribute:
        assert config.n_routed_experts == 288
        with pytest.raises(AttributeError):
            _ = config.not_a_real_attribute

    def test_vision_config_keeps_extra_kwargs(self):
        """Vision config keeps extra keyword arguments."""
        vision = Glm5NextVisionConfig(depth=3, hidden=7)
        assert vision.depth == 3
        assert vision.model_type == "glm5_next_vision"


class TestProviderProbes:
    """Capability probes return booleans without importing device code."""

    def test_probes_are_boolean_and_cached(self):
        """Probes are boolean and cached."""
        from vllm_fl.models.glm5.provider import (
            _has_deep_gemm,
            _has_nvidia_reference_kernels,
            _has_vllm_native_extension,
            use_nvidia_reference,
        )

        assert isinstance(use_nvidia_reference(), bool)
        assert isinstance(_has_deep_gemm(), bool)
        assert isinstance(_has_vllm_native_extension(), bool)
        first = _has_nvidia_reference_kernels()
        assert _has_nvidia_reference_kernels() is first


class TestUnquantizedWeights:
    """The dtype gate rejects undeclared quantized storage."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_unquantized_dtypes_stream_through(self, dtype):
        """Unquantized dtypes stream through."""
        weights = [("a.weight", torch.zeros(2, dtype=dtype))]
        assert [n for n, _ in unquantized_weights(weights)] == ["a.weight"]

    @pytest.mark.parametrize(
        "dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.int8]
    )
    def test_quantized_dtypes_are_rejected(self, dtype):
        """Quantized dtypes are rejected."""
        weights = [("a.weight", torch.zeros(2, dtype=dtype))]
        with pytest.raises(ValueError, match="unquantized checkpoint required"):
            list(unquantized_weights(weights))


class _Param:
    """Parameter double recording weight-loader calls."""

    def __init__(self, result=True):
        self.calls = []
        self.result = result
        self.original_loader = self._loader
        self.weight_loader = self._loader

    def _loader(self, param, weight, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _Model:
    """Model double routing mlp submodules to an expert owner."""

    def __init__(self, **params):
        self._params = params

    def named_parameters(self):
        return list(self._params.items())

    def get_submodule(self, name):
        if name.rsplit(".", 1)[-1] == "mlp":
            return SimpleNamespace(expert_map=None)
        return SimpleNamespace()


class TestRequiredShards:
    """Shard/expert requirements per parameter kind."""

    def test_expert_weights_cover_all_experts(self):
        """Expert weights cover all experts."""
        model = _Model()
        model.config = SimpleNamespace(n_routed_experts=3)
        got = _required_shards(model, "layers.0.mlp.w13_weight")
        assert got == {(shard, expert) for expert in range(3) for shard in ("w1", "w3")}
        assert _required_shards(model, "layers.0.mlp.w2_weight") == {
            ("w2", e) for e in range(3)
        }

    def test_local_expert_map_selects_owned_experts(self):
        """Local expert map selects owned experts."""
        model = _Model()
        owner = SimpleNamespace(expert_map=torch.tensor([-1, 0, -1, 2]))
        model.get_submodule = lambda name: owner
        got = _required_shards(model, "layers.0.mlp.w13_weight")
        assert got == {("w1", 1), ("w1", 3), ("w3", 1), ("w3", 3)}

    def test_fused_projections_need_two_shards(self):
        """Fused projections need two shards."""
        model = _Model()
        names = (
            "layers.0.mlp.gate_up_proj.weight",
            "layers.0.self_attn.fused_qkv_a_proj.weight",
            "layers.0.self_attn.wk_weights_proj.weight",
        )
        for name in names:
            assert _required_shards(model, name) == {(0, None), (1, None)}

    def test_ordinary_parameters_return_none(self):
        """Leave parameters outside the fused loader to the default loader."""
        model = _Model()
        assert _required_shards(model, "layers.0.self_attn.q_proj.weight") is None


class TestAuditPackedWeights:
    """The packed-shard audit tracks, validates and restores loaders."""

    @staticmethod
    def _load_experts(w13, experts=(0, 1)):
        for shard in ("w1", "w3"):
            for expert in experts:
                w13.weight_loader(
                    None, torch.zeros(1), shard_id=shard, expert_id=expert
                )

    def test_complete_load_passes_and_restores(self):
        """Complete load passes and restores."""
        gate, w13 = _Param(), _Param()
        model = _Model(
            **{
                "model.layers.0.mlp.gate_up_proj.weight": gate,
                "model.layers.0.mlp.w13_weight": w13,
            }
        )
        model.config = SimpleNamespace(n_routed_experts=2)
        with audit_packed_weights(model):
            gate.weight_loader(None, torch.zeros(1), shard_id=0)
            gate.weight_loader(None, torch.zeros(1), shard_id=1)
            self._load_experts(w13)
        assert gate.weight_loader == gate.original_loader
        assert w13.weight_loader == w13.original_loader
        assert len(gate.calls) == 2 and len(w13.calls) == 4

    def test_missing_shards_raise_and_restore(self):
        """Missing shards raise and restore."""
        gate = _Param()
        model = _Model(**{"model.layers.0.mlp.gate_up_proj.weight": gate})
        with (
            pytest.raises(RuntimeError, match="missing packed weight shards"),
            audit_packed_weights(model),
        ):
            gate.weight_loader(None, torch.zeros(1), shard_id=0)
        assert gate.weight_loader == gate.original_loader

    def test_duplicate_shard_raises(self):
        """Duplicate shard raises."""
        gate = _Param()
        model = _Model(**{"model.layers.0.mlp.gate_up_proj.weight": gate})
        with (
            pytest.raises(ValueError, match="Duplicate GLM5-Next packed weight"),
            audit_packed_weights(model),
        ):
            gate.weight_loader(None, torch.zeros(1), shard_id=0)
            gate.weight_loader(None, torch.zeros(1), shard_id=0)

    def test_unowned_expert_returns_do_not_count(self):
        """Expert loaders returning False are not recorded as loaded."""
        w13 = _Param(result=False)
        model = _Model(**{"model.layers.0.mlp.w13_weight": w13})
        model.config = SimpleNamespace(n_routed_experts=2)
        with (
            pytest.raises(RuntimeError, match="missing packed weight shards"),
            audit_packed_weights(model),
        ):
            self._load_experts(w13)

    def test_plain_parameters_are_not_wrapped(self):
        """Plain parameters are not wrapped."""
        plain = _Param()
        model = _Model(**{"model.layers.0.self_attn.q_proj.weight": plain})
        with audit_packed_weights(model):
            pass
        assert plain.weight_loader == plain.original_loader


class _TextInner:
    def named_parameters(self):
        return [("model.a", 1), ("model.b", 2)]

    def get_submodule(self, name):
        return SimpleNamespace()

    def finalize_mhc_broadcast_weights(self):
        self.finalized = True


class _TextOuter:
    def __init__(self, inner):
        self.model = inner

    def named_parameters(self):
        return self.model.named_parameters()


class TestAuditTextWeights:
    """The text-level audit finalizes only on the complete outer load."""

    def test_complete_load_finalizes_mhc(self):
        """Complete load finalizes mHC."""
        outer = _TextOuter(_TextInner())
        with audit_text_weights(outer) as loaded:
            loaded.update(n for n, _ in outer.model.named_parameters())
        assert outer.model.finalized is True
        assert not hasattr(outer, "_glm5_loading_names")

    def test_missing_names_raise_and_cleanup(self):
        """Missing names raise and cleanup."""
        outer = _TextOuter(_TextInner())
        with (
            pytest.raises(RuntimeError, match="did not initialize all text"),
            audit_text_weights(outer) as loaded,
        ):
            loaded.add("model.a")
        assert not hasattr(outer, "_glm5_loading_names")

    def test_nested_call_shares_the_outer_set(self):
        """Nested call shares the outer set."""
        outer = _TextOuter(_TextInner())
        existing = {"model.a"}
        outer._glm5_loading_names = existing
        with audit_text_weights(outer) as active:
            assert active is existing
        assert outer._glm5_loading_names is existing


def _vllm_config(
    hf=None,
    text=None,
    quantization=None,
    quant_config=None,
    pipeline_parallel_size=1,
    speculative=None,
    eplb=False,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf if hf is not None else SimpleNamespace(model_type="glm5_next"),
            hf_text_config=text,
            quantization=quantization,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size, enable_eplb=eplb
        ),
        speculative_config=speculative,
        quant_config=quant_config,
    )


class TestValidateGlm5Config:
    """Startup guards rejecting unsupported modes (runtime.py:415)."""

    def test_non_glm5_model_is_a_no_op(self):
        # A model_config whose hf_config is not a GLM5 schema is ignored
        # entirely, even when other fields request unsupported modes.
        """Leave non-GLM5 models outside GLM5 startup validation."""
        assert (
            validate_glm5_config(
                SimpleNamespace(
                    model_config=SimpleNamespace(hf_config=None),
                    parallel_config=SimpleNamespace(pipeline_parallel_size=2),
                )
            )
            is None
        )

    def test_pipeline_parallelism_rejected(self):
        """Pipeline parallelism rejected."""
        with pytest.raises(Glm5ConfigError, match="pipeline_parallel_size=1"):
            validate_glm5_config(_vllm_config(pipeline_parallel_size=2))

    def test_speculative_decoding_rejected(self):
        """Speculative decoding rejected."""
        with pytest.raises(Glm5ConfigError, match="speculative decoding"):
            validate_glm5_config(_vllm_config(speculative=SimpleNamespace()))

    def test_eplb_rejected(self):
        """EPLB rejected."""
        with pytest.raises(Glm5ConfigError, match="enable_eplb"):
            validate_glm5_config(_vllm_config(eplb=True))

    def test_quantized_checkpoints_rejected(self):
        """Quantized checkpoints rejected."""
        with pytest.raises(Glm5ConfigError, match="unquantized"):
            validate_glm5_config(_vllm_config(quantization="fp8"))
        with pytest.raises(Glm5ConfigError, match="unquantized"):
            validate_glm5_config(_vllm_config(quant_config=SimpleNamespace()))
        with pytest.raises(Glm5ConfigError, match="unquantized"):
            validate_glm5_config(
                _vllm_config(
                    hf=SimpleNamespace(
                        model_type="glm5_next", quantization_config={"fmt": "fp8"}
                    )
                )
            )

    def test_bf16_indexer_requires_zero_rope_and_compression(self):
        """A100 (sm_80) takes the bf16 path; its guards must hold there."""
        text = SimpleNamespace
        with pytest.raises(Glm5ConfigError, match="zero RoPE"):
            validate_glm5_config(_vllm_config(text=text(qk_rope_head_dim=64)))
        with pytest.raises(Glm5ConfigError, match="K-pool compression"):
            validate_glm5_config(
                _vllm_config(text=text(qk_rope_head_dim=0, index_kpool_compress=False))
            )
        # The valid bf16 configuration passes.
        validate_glm5_config(
            _vllm_config(text=text(qk_rope_head_dim=0, index_kpool_compress=True))
        )
