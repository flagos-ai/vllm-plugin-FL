"""HF/torchvision compatibility without a vLLM model or accelerator."""

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.configuration_utils import PretrainedConfig

from vllm_fl.configs.glm5_next import Glm5NextTextConfig
from vllm_fl.transformers_utils.processors.glm5_next import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    Glm5NextVideoProcessor,
)


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
