# SPDX-License-Identifier: Apache-2.0
"""Load real processors/tokenizers from local directories and HF snapshots.

Only transport and forwarding observations are mocked. Hugging Face cache
resolution, JSON parsing, tokenizer construction, and processor construction
use the installed dependencies and their normal public APIs.
"""

import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
import requests
from huggingface_hub import try_to_load_from_cache
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, BertConfig, PreTrainedTokenizerFast

from vllm_fl.transformers_utils.processors import glm5_next as processor_module
from vllm_fl.transformers_utils.processors.glm5_next import Glm5NextProcessor

REPO_ID = "processor-cache-tests/glm5-next"
REVISION_A = "a" * 40
REVISION_B = "b" * 40


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """An offline snapshot load must never reach the HTTP boundary."""

    def unexpected_request(*args, **kwargs):
        raise AssertionError("Processor cache regression attempted HTTP transport")

    async def unexpected_async_request(*args, **kwargs):
        raise AssertionError("Processor cache regression attempted HTTP transport")

    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setattr(requests.Session, "request", unexpected_request)
    monkeypatch.setattr(httpx.Client, "send", unexpected_request)
    monkeypatch.setattr(httpx.AsyncClient, "send", unexpected_async_request)


def write_checkpoint(path, *, image_tokens=512, video_tokens=1024, word_id=7):
    path.mkdir(parents=True, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {
                    "[UNK]": 0,
                    "<|image|>": 1,
                    "<|video|>": 2,
                    "revision-word": word_id,
                },
                unk_token="[UNK]",
            )
        ),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)
    BertConfig(
        vocab_size=word_id + 1,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
    ).save_pretrained(path)
    config = {
        "processor_class": "Glm5NextProcessor",
        "image_processor": {
            "min_image_tokens": 16,
            "max_image_tokens": image_tokens,
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "patch_expand_factor": 1,
            "resize_mode": "pad",
        },
        "video_processor": {
            "min_image_tokens": 16,
            "max_image_tokens": video_tokens,
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "patch_expand_factor": 1,
            "fps_interval": 1.5,
            "max_frame_count_dynamic": 48,
        },
    }
    (path / "processor_config.json").write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.fixture
def local_checkpoint(tmp_path):
    return write_checkpoint(tmp_path / "local-checkpoint")


@pytest.fixture
def hub_cache(tmp_path):
    cache_dir = tmp_path / "hub"
    repo_cache = cache_dir / ("models--" + REPO_ID.replace("/", "--"))
    snapshot_a = write_checkpoint(repo_cache / "snapshots" / REVISION_A)
    snapshot_b = write_checkpoint(
        repo_cache / "snapshots" / REVISION_B,
        image_tokens=768,
        video_tokens=2048,
        word_id=9,
    )
    refs = repo_cache / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(REVISION_A, encoding="utf-8")
    (refs / "review-target").write_text(REVISION_B, encoding="utf-8")
    return SimpleNamespace(
        cache_dir=cache_dir,
        repo_cache=repo_cache,
        snapshots={REVISION_A: snapshot_a, REVISION_B: snapshot_b},
    )


def assert_checkpoint(processor, *, image_tokens, video_tokens, word_id):
    assert processor.image_processor.max_image_tokens == image_tokens
    assert processor.video_processor.max_image_tokens == video_tokens
    assert processor.video_processor.fps_interval == 1.5
    assert processor.video_processor.max_frame_count_dynamic == 48
    assert isinstance(processor.tokenizer, PreTrainedTokenizerFast)
    assert processor.tokenizer.convert_tokens_to_ids("revision-word") == word_id


def test_local_directory_uses_nested_processor_config(local_checkpoint):
    assert not (local_checkpoint / "preprocessor_config.json").exists()
    processor = Glm5NextProcessor.from_pretrained(
        str(local_checkpoint), local_files_only=True
    )
    assert_checkpoint(processor, image_tokens=512, video_tokens=1024, word_id=7)


@pytest.mark.parametrize(
    "revision,image_tokens,video_tokens,word_id",
    [
        ("main", 512, 1024, 7),
        ("review-target", 768, 2048, 9),
        (REVISION_B, 768, 2048, 9),
    ],
)
def test_repo_id_resolves_offline_snapshot_revision(
    hub_cache, revision, image_tokens, video_tokens, word_id
):
    cached_path = try_to_load_from_cache(
        REPO_ID,
        "processor_config.json",
        cache_dir=hub_cache.cache_dir,
        revision=revision,
    )
    selected_sha = REVISION_A if revision == "main" else REVISION_B
    assert cached_path == str(
        hub_cache.snapshots[selected_sha] / "processor_config.json"
    )

    processor = Glm5NextProcessor.from_pretrained(
        REPO_ID,
        cache_dir=str(hub_cache.cache_dir),
        revision=revision,
        local_files_only=True,
    )
    assert_checkpoint(
        processor,
        image_tokens=image_tokens,
        video_tokens=video_tokens,
        word_id=word_id,
    )


def test_tokenizer_revision_does_not_select_processor_revision(hub_cache):
    processor = Glm5NextProcessor.from_pretrained(
        REPO_ID,
        cache_dir=str(hub_cache.cache_dir),
        revision="main",
        tokenizer_revision="review-target",
        local_files_only=True,
    )
    assert_checkpoint(processor, image_tokens=512, video_tokens=1024, word_id=9)


def test_repo_id_without_revision_uses_cached_main(hub_cache):
    processor = Glm5NextProcessor.from_pretrained(
        REPO_ID,
        cache_dir=str(hub_cache.cache_dir),
        local_files_only=True,
    )
    assert_checkpoint(processor, image_tokens=512, video_tokens=1024, word_id=7)


def test_hub_and_tokenizer_options_are_forwarded_without_mutation(
    hub_cache, monkeypatch
):
    bundle = hub_cache.repo_cache / "snapshots" / REVISION_B / "bundle"
    write_checkpoint(bundle, image_tokens=768, video_tokens=2048, word_id=9)
    original_cached_file = processor_module.cached_file
    original_tokenizer_loader = AutoTokenizer.from_pretrained
    config_calls = []
    tokenizer_calls = []

    def observe_config(model, filename, **kwargs):
        config_calls.append((model, filename, deepcopy(kwargs)))
        return original_cached_file(model, filename, **kwargs)

    def observe_tokenizer(model, *args, **kwargs):
        tokenizer_calls.append((model, deepcopy(kwargs)))
        return original_tokenizer_loader(model, *args, **kwargs)

    monkeypatch.setattr(processor_module, "cached_file", observe_config)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", observe_tokenizer)
    kwargs = dict(
        cache_dir=str(hub_cache.cache_dir),
        revision="review-target",
        token="hf_offline_test_token",
        local_files_only=True,
        force_download=False,
        subfolder="bundle",
        trust_remote_code=False,
        use_fast=True,
    )
    before = deepcopy(kwargs)
    processor = Glm5NextProcessor.from_pretrained(REPO_ID, **kwargs)

    assert kwargs == before
    assert len(config_calls) == len(tokenizer_calls) == 1
    model, filename, config_kwargs = config_calls[0]
    assert (model, filename) == (REPO_ID, "processor_config.json")
    for option in (
        "cache_dir",
        "revision",
        "token",
        "local_files_only",
        "force_download",
        "subfolder",
    ):
        assert config_kwargs[option] == kwargs[option]
        assert tokenizer_calls[0][1][option] == kwargs[option]
    assert tokenizer_calls[0][0] == REPO_ID
    assert tokenizer_calls[0][1]["trust_remote_code"] is False
    assert tokenizer_calls[0][1]["use_fast"] is True
    assert_checkpoint(processor, image_tokens=768, video_tokens=2048, word_id=9)


@pytest.mark.parametrize("location", ["local", "hub"])
def test_missing_processor_config_raises_without_defaults(
    local_checkpoint, hub_cache, location
):
    if location == "local":
        (local_checkpoint / "processor_config.json").unlink()
        model = str(local_checkpoint)
        kwargs = {}
        missing_config_error = "processor_config.json"
    else:
        (hub_cache.snapshots[REVISION_A] / "processor_config.json").unlink()
        model = REPO_ID
        kwargs = dict(cache_dir=str(hub_cache.cache_dir), revision="main")
        missing_config_error = (
            r"processor_config\.json|couldn't find them in (?:the )?cached files"
        )
    with pytest.raises(OSError, match=missing_config_error):
        Glm5NextProcessor.from_pretrained(model, local_files_only=True, **kwargs)


def make_processing_info(model_config, deployment):
    from vllm_fl.models.glm5_next_multimodal import Glm5NextProcessingInfo

    info = object.__new__(Glm5NextProcessingInfo)
    info.ctx = SimpleNamespace(
        model_config=model_config,
        get_merged_mm_kwargs=lambda kwargs: {**deepcopy(deployment), **kwargs},
    )
    return info


@pytest.mark.parametrize("explicit_overrides", [False, True])
def test_processing_info_forwards_model_defaults_and_caches_only_after_success(
    hub_cache, monkeypatch, explicit_overrides
):
    model_config = SimpleNamespace(
        model=REPO_ID,
        revision="review-target",
        tokenizer_revision="main",
        hf_token="hf_context_offline_test_token",
        trust_remote_code=False,
    )
    deployment = {
        "images_kwargs": {"max_image_tokens": 128},
        "videos_kwargs": {"max_image_tokens": 256, "max_frames": 8},
    }
    info = make_processing_info(model_config, deployment)
    original_loader = Glm5NextProcessor.from_pretrained
    calls = []

    def observe_loader(model, **kwargs):
        calls.append((model, deepcopy(kwargs)))
        return original_loader(model, **kwargs)

    monkeypatch.setattr(Glm5NextProcessor, "from_pretrained", observe_loader)
    kwargs = dict(cache_dir=str(hub_cache.cache_dir), local_files_only=True)
    if explicit_overrides:
        kwargs.update(
            revision="main",
            tokenizer_revision="review-target",
            token="hf_explicit_offline_test_token",
            trust_remote_code=True,
        )
    before = deepcopy(kwargs)
    processor = info.get_hf_processor(**kwargs)

    assert kwargs == before
    assert len(calls) == 1
    assert calls[0][0] == REPO_ID
    expected = {
        "revision": "main" if explicit_overrides else "review-target",
        "tokenizer_revision": "review-target" if explicit_overrides else "main",
        "token": kwargs.get("token", model_config.hf_token),
        "trust_remote_code": explicit_overrides,
        "cache_dir": str(hub_cache.cache_dir),
        "local_files_only": True,
    }
    for key, value in expected.items():
        assert calls[0][1][key] == value
    assert_checkpoint(
        processor,
        image_tokens=512 if explicit_overrides else 768,
        video_tokens=1024 if explicit_overrides else 2048,
        word_id=9 if explicit_overrides else 7,
    )
    assert processor.serving_budgets["image"].max_tokens == 128
    assert processor.serving_budgets["video"].max_tokens == 256
    assert deployment["images_kwargs"]["max_image_tokens"] == 128
    assert info.get_hf_processor(max_image_tokens=64) is processor
    assert len(calls) == 1


def test_failed_processing_info_load_is_retriable(local_checkpoint):
    config_file = local_checkpoint / "processor_config.json"
    config = config_file.read_text(encoding="utf-8")
    config_file.unlink()
    info = make_processing_info(
        SimpleNamespace(
            model=str(local_checkpoint),
            revision=None,
            tokenizer_revision=None,
            hf_token=None,
            trust_remote_code=False,
        ),
        {},
    )
    with pytest.raises(OSError, match="processor_config.json"):
        info.get_hf_processor(local_files_only=True)
    assert getattr(info, "_glm5_hf_processor", None) is None

    config_file.write_text(config, encoding="utf-8")
    processor = info.get_hf_processor(local_files_only=True)
    assert_checkpoint(processor, image_tokens=512, video_tokens=1024, word_id=7)
    assert info.get_hf_processor() is processor
