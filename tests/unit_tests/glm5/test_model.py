# SPDX-License-Identifier: Apache-2.0
"""Model contracts and regression coverage."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import nn
from transformers import PreTrainedTokenizerFast
from transformers.configuration_utils import PretrainedConfig

from vllm_fl.models.glm5.config import Glm5NextTextConfig
from vllm_fl.models.glm5.loader import audit_packed_weights
from vllm_fl.models.glm5.model import Glm5NextForCausalLM, Glm5NextModel
from vllm_fl.models.glm5.multimodal import Glm5NextForConditionalGeneration
from vllm_fl.models.glm5.processor import (
    _LEGACY_MM_MAX_PIXELS,
    _MAX_VIDEO_TOKENS,
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    Glm5NextVideoProcessor,
    _normalize_processor_config,
    glm_sample_frame_indices,
    glm_sample_frame_indices_legacy,
)
from vllm_fl.models.glm5.runtime import (
    Glm5NextModelArchConfigConvertor,
)


class _Fixture(nn.Module):
    load_weights = Glm5NextModel.load_weights

    def __init__(self, ep=False):
        super().__init__()
        self.config = SimpleNamespace(
            n_routed_experts=2,
            n_shared_experts=0,
            num_hidden_layers=1,
            num_nextn_predict_layers=0,
        )
        self.use_mha = False
        self.num_redundant_experts = 0
        layer = nn.Module()
        self.layers = nn.ModuleList([layer])
        layer.mlp = nn.Module()
        layer.mlp.gate_up_proj = nn.Module()
        param = nn.Parameter(torch.full((4, 2), -99.0), requires_grad=False)

        def packed_loader(param, value, shard_id=None):
            if shard_id is None:
                param.copy_(value)
            else:
                param[shard_id * 2 : (shard_id + 1) * 2].copy_(value)

        param.weight_loader = packed_loader
        layer.mlp.gate_up_proj.weight = param
        layer.mlp.experts = nn.Module()
        experts = nn.Module()
        layer.mlp.experts.routed_experts = experts
        experts.expert_map = torch.tensor([-1, 0]) if ep else None
        nlocal = 1 if ep else 2
        for name, width in (("w13_weight", 4), ("w2_weight", 2)):
            param = nn.Parameter(
                torch.full((nlocal, width, 2), -99.0), requires_grad=False
            )

            def expert_loader(
                param, value, name, *, shard_id, expert_id, return_success
            ):
                if ep and expert_id == 0:
                    return False
                local_id = 0 if ep else expert_id
                if shard_id == "w2":
                    param[local_id].copy_(value)
                else:
                    start = 0 if shard_id == "w1" else 2
                    param[local_id, start : start + 2].copy_(value)
                return True

            param.weight_loader = expert_loader
            setattr(experts, name, param)


def _weights():
    result = [
        (f"layers.0.mlp.{part}_proj.weight", torch.full((2, 2), float(i)))
        for i, part in enumerate(("gate", "up"), 1)
    ]
    result += [
        (
            f"layers.0.mlp.experts.{expert}.{part}_proj.weight",
            torch.full((2, 2), float(i + expert)),
        )
        for expert in range(2)
        for i, part in enumerate(("gate", "up", "down"), 3)
    ]
    return result


@pytest.mark.parametrize("ep", [False, True])
def test_complete_packed_and_expert_weights(ep):
    model = _Fixture(ep)
    originals = [p.weight_loader for p in model.parameters()]
    with audit_packed_weights(model):
        loaded = model.load_weights(iter(_weights()))
    assert set(dict(model.named_parameters())) <= loaded
    assert all(not (p == -99).any() for p in model.parameters())
    assert all(p.weight_loader is old for p, old in zip(model.parameters(), originals))


@pytest.mark.parametrize(
    "missing",
    [
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.experts.1.up_proj.weight",
        "layers.0.mlp.experts.1.down_proj.weight",
    ],
)
def test_missing_shard_fails_even_when_destination_name_is_loaded(missing):
    model = _Fixture()
    originals = [p.weight_loader for p in model.parameters()]
    with (
        pytest.raises(RuntimeError, match="missing packed weight shards"),
        audit_packed_weights(model),
    ):
        model.load_weights((n, w) for n, w in _weights() if n != missing)
    assert all(p.weight_loader is old for p, old in zip(model.parameters(), originals))


def test_undeclared_fp8_rejected_before_copy():
    model = _Fixture()
    name, value = _weights()[0]
    with pytest.raises(ValueError, match="unquantized checkpoint required"):
        model.load_weights([(name, value.to(torch.float8_e4m3fn))])
    assert (model.layers[0].mlp.gate_up_proj.weight == -99).all()


def _causal_model():
    model = Glm5NextForCausalLM.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(tie_word_embeddings=False)
    model.model = _Fixture()
    model.lm_head = nn.Linear(2, 2, bias=False)
    model.finalizations = 0

    def finalize():
        model.finalizations += 1

    model.model.finalize_mhc_broadcast_weights = finalize
    return model


def _interleaved_weights(multimodal):
    prefix = "model.language_model." if multimodal else "model."
    weights = [(prefix + n, w) for n, w in _weights()]
    # File 45 starts with lm_head, interrupting the text subtree.
    weights.insert(4, ("lm_head.weight", torch.ones(2, 2)))
    if multimodal:
        # A separate vision file can also interrupt the language-model prefix.
        weights.insert(2, ("model.visual.weight", torch.ones(2, 2)))
    return weights


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("missing_up", [False, True])
def test_checkpoint_audit_spans_interleaved_subtree_calls(multimodal, missing_up):
    causal = _causal_model()
    model = causal
    if multimodal:
        model = Glm5NextForConditionalGeneration.__new__(
            Glm5NextForConditionalGeneration
        )
        nn.Module.__init__(model)
        model.language_model = causal
        model.visual = nn.Linear(2, 2, bias=False)
    weights = _interleaved_weights(multimodal)
    if missing_up:
        weights = [
            (n, w) for n, w in weights if not n.endswith("experts.1.up_proj.weight")
        ]
    originals = [p.weight_loader for p in causal.model.parameters()]
    if missing_up:
        with pytest.raises(RuntimeError, match="missing packed weight shards"):
            model.load_weights(iter(weights))
        assert causal.finalizations == 0
    else:
        loaded = model.load_weights(iter(weights))
        assert set(dict(model.named_parameters())) <= loaded
        assert all(not (p == -99).any() for p in causal.model.parameters())
        assert causal.finalizations == 1
    assert not hasattr(causal, "_glm5_loading_names")
    assert all(
        p.weight_loader is old for p, old in zip(causal.model.parameters(), originals)
    )


def test_interleaved_duplicate_shard_rejected():
    model = _causal_model()
    weights = _interleaved_weights(False)
    weights.append(weights[0])
    with pytest.raises(ValueError, match="Duplicate GLM5-Next packed weight"):
        model.load_weights(iter(weights))
    assert model.finalizations == 0
    assert not hasattr(model, "_glm5_loading_names")


def test_sparse_checkpoint_alias_keeps_schema_and_hf_validation():
    kinds = ["linear_attention", "deepseek_sparse_attention"]
    config = Glm5NextTextConfig(num_hidden_layers=2, layer_types=kinds)
    config.validate_layer_type()
    assert config.layer_types == kinds
    assert config.to_dict()["layer_types"] == kinds
    assert config.layers_block_type == ["linear_attention", "attention"]
    if hasattr(PretrainedConfig, "validate_layer_type"):
        config.layer_types = ["invalid", "linear_attention"]
        with pytest.raises(ValueError):
            config.validate_layer_type()


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_video_rgb_conversion_without_new_torchvision_api(channels):
    video = torch.full((2, 3, channels, 8, 8), 64, dtype=torch.uint8)
    actual = Glm5NextVideoProcessor().convert_to_rgb(video)
    expected = video[..., :1, :, :].expand(2, 3, 3, 8, 8).float()
    if channels == 4:
        expected = (1 - 64 / 255) * 255 + (64 / 255) * expected
    torch.testing.assert_close(actual.float(), expected)


@pytest.mark.parametrize("fps", [None, 0.1])
def test_flat_video_metadata_survives_serving_budget_normalization(fps):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "<|video|>": 2}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
    )
    processor = Glm5NextProcessor(
        Glm5NextImageProcessor(),
        tokenizer,
        Glm5NextVideoProcessor(max_image_tokens=128),
    )
    processor.configure_serving({"max_frames": 8, "max_image_tokens": 128})
    options = {} if fps is None else {"fps": fps}
    result = processor(
        videos=[np.zeros((60, 112, 112, 3), dtype=np.uint8)],
        video_metadata=[dict(total_num_frames=60, fps=2.0, duration=30.0)],
        text="<|video|>",
        return_tensors="pt",
        **options,
    )
    grid = result["video_grid_thw"][0]
    assert 0 < int(grid[0]) <= 4
    assert int(grid.prod()) // 4 <= 128


class _TokenizerStub:
    init_kwargs = {}

    def __init__(self) -> None:
        self.last_text = None

    def __call__(self, text, **kwargs):
        self.last_text = text
        return {"input_ids": [[11, 12]]}


class _ImageProcessorStub:
    def __call__(self, images, **kwargs):
        return {"pixel_values": "pixels", "image_grid_thw": "grid"}


class _ProcessorCallHarness:
    """A method harness; real HF initialization is covered separately."""

    _call_with_resolved_kwargs = Glm5NextProcessor._call_with_resolved_kwargs

    def __init__(self) -> None:
        self.tokenizer = _TokenizerStub()
        self.image_processor = _ImageProcessorStub()

    def _merge_kwargs(self, *args, **kwargs):
        return {
            "text_kwargs": {
                "return_mm_token_type_ids": False,
            },
            "images_kwargs": {},
            "videos_kwargs": {},
        }


def test_legacy_size_config_is_normalized_without_changing_geometry_mode() -> None:
    config = _normalize_processor_config(
        {
            "size": {
                "shortest_edge": 112 * 112,
                "longest_edge": 100_000_000,
            },
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "patch_expand_factor": 2,
        },
        default_min_tokens=16,
        default_max_tokens=8000,
        is_video=False,
    )

    pixels_per_token = 2 * (14 * 2) ** 2
    assert config["min_image_tokens"] == 8
    assert config["max_image_tokens"] == _LEGACY_MM_MAX_PIXELS // pixels_per_token
    assert config["patch_expand_factor"] == 2
    assert config["resize_mode"] == "resize"
    assert config["sampling_policy"] == "legacy_dynamic"
    assert "size" not in config


def test_token_budget_image_config_keeps_checkpoint_budget() -> None:
    config = _normalize_processor_config(
        {
            "min_image_tokens": 16,
            "max_image_tokens": 8000,
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "patch_expand_factor": 1,
            "resize_mode": "pad",
        },
        default_min_tokens=16,
        default_max_tokens=8000,
        is_video=False,
    )

    assert config["min_image_tokens"] == 16
    assert config["max_image_tokens"] == 8000
    assert config["patch_expand_factor"] == 1
    assert config["resize_mode"] == "pad"
    assert config["sampling_policy"] == "fps_interval"


def test_token_budget_video_config_uses_reference_serving_cap() -> None:
    config = _normalize_processor_config(
        {
            "min_image_tokens": 16,
            "max_image_tokens": 240000,
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
        },
        default_min_tokens=16,
        default_max_tokens=240000,
        is_video=True,
    )

    assert config["min_image_tokens"] == 16
    assert config["max_image_tokens"] == _MAX_VIDEO_TOKENS


def test_explicit_zero_head_dim_keeps_nope_mla_layout() -> None:
    config = Glm5NextTextConfig(
        head_dim=0,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
    )
    convertor = Glm5NextModelArchConfigConvertor(config, config)

    assert config.head_dim == 0
    assert convertor.is_deepseek_mla()
    assert convertor.get_head_size() == 512


def test_feature_processor_leaves_prompt_placeholders_unchanged() -> None:
    processor = _ProcessorCallHarness()
    prompt = "before <|image|> after"

    output = Glm5NextProcessor.__call__(processor, images=object(), text=prompt)

    assert processor.tokenizer.last_text == [prompt]
    assert output["image_grid_thw"] == "grid"


def test_video_sampling_remains_config_selectable() -> None:
    new_indices = glm_sample_frame_indices(
        300,
        30.0,
        10.0,
        target_fps=2.0,
        max_frame_count=2048,
        temporal_patch_size=2,
    )
    legacy_indices = glm_sample_frame_indices_legacy(
        300,
        30.0,
        10.0,
        target_fps=3.0,
        max_frame_count=640,
        temporal_patch_size=2,
    )

    assert len(new_indices) == 20
    assert len(legacy_indices) == 60
    assert new_indices[-1] == 299
    # Legacy sampling walks at 6 Hz; unlike fps_interval's linspace fixup,
    # it does not force inclusion of the source's final frame.
    assert legacy_indices[0] == 0
    assert legacy_indices[-1] == 295
    assert all(a < b for a, b in zip(legacy_indices, legacy_indices[1:]))
