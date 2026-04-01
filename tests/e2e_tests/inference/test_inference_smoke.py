# Copyright (c) 2025 BAAI. All rights reserved.

"""
Unified inference smoke test — auto-generated from model YAML configs.

This test file is driven by two environment variables set by ``tests/run.py``:

- ``FL_TEST_MODEL``: Model family (e.g. ``qwen3``, ``minicpm``)
- ``FL_TEST_CASE``:  Case name within the family (e.g. ``06b_tp2``, ``o45_tp2``)

It loads ``tests/models/<model>/<case>.yaml``, constructs the LLM engine,
runs generation for each prompt (with optional parametrize combos), and
asserts that outputs are non-empty.

Supports both text-only and multimodal (audio/image/video) models via the
``generate.modality`` field in the YAML config.
"""

import os

import pytest

from tests.utils.model_config import ModelConfig
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

# ---------------------------------------------------------------------------
# Load config from environment (injected by run.py)
# ---------------------------------------------------------------------------

_MODEL = os.environ.get("FL_TEST_MODEL", "")
_CASE = os.environ.get("FL_TEST_CASE", "")

if not _MODEL or not _CASE:
    pytest.skip(
        "FL_TEST_MODEL and FL_TEST_CASE must be set (injected by run.py)",
        allow_module_level=True,
    )

_CFG = ModelConfig.load(_MODEL, _CASE)

if not os.path.exists(_CFG.model):
    pytest.fail(
        f"Model not found: {_CFG.model}",
        pytrace=False,
    )


# ---------------------------------------------------------------------------
# Multimodal helpers
# ---------------------------------------------------------------------------


def _load_assets(modality: str, asset_names: list[str], count: int) -> dict:
    """Load vllm built-in assets for multimodal input.

    Args:
        modality: One of ``audio``, ``image``, ``video``.
        asset_names: Pool of asset names from YAML config.
        count: Number of assets to include (0 returns empty dict).

    Returns:
        Dict suitable for ``multi_modal_data`` in vllm inputs.
    """
    if count <= 0:
        return {}

    selected = asset_names[:count]

    if modality == "audio":
        from vllm.assets.audio import AudioAsset

        return {"audio": [AudioAsset(name).audio_and_sample_rate for name in selected]}
    elif modality == "image":
        from vllm.assets.image import ImageAsset

        return {"image": [ImageAsset(name).pil_image for name in selected]}
    elif modality == "video":
        from vllm.assets.video import VideoAsset

        return {"video": [VideoAsset(name).np_ndarrays for name in selected]}
    else:
        raise ValueError(f"Unsupported modality: {modality}")


def _build_multimodal_prompt(
    tokenizer,
    question: str,
    modality: str,
    asset_count: int,
) -> str:
    """Build a chat prompt with multimodal placeholders.

    Uses the tokenizer's ``apply_chat_template`` to format the prompt.
    The placeholder format is determined by modality.
    """
    placeholder_map = {
        "audio": "(<audio>./</audio>)",
        "image": "(<image>./</image>)",
        "video": "(<video>./</video>)",
    }
    placeholder = placeholder_map.get(modality, "")
    content = (
        f"{placeholder * asset_count}\n{question}" if asset_count > 0 else question
    )

    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Test function
# ---------------------------------------------------------------------------


def _get_max_asset_count() -> int:
    """Return the maximum asset_count across all multimodal prompts."""
    gen = _CFG.generate
    if gen.modality == "text":
        return 0
    return max(
        (p.get("asset_count", 0) for p in gen.prompts if isinstance(p, dict)),
        default=0,
    )


def _run_text_test(llm: LLM, sampling_params: SamplingParams) -> None:
    """Run text-only generation test.

    Prompts can be plain strings or dicts with ``text`` and optional
    ``expected`` (substring that must appear in the output).
    """
    gen = _CFG.generate
    raw_prompts = gen.prompts
    assert len(raw_prompts) > 0, "No prompts defined in YAML config"

    # Normalize: str → {"text": str}, dict stays as-is
    prompt_cfgs = []
    for p in raw_prompts:
        if isinstance(p, str):
            prompt_cfgs.append({"text": p})
        else:
            prompt_cfgs.append(p)

    prompt_texts = [p["text"] for p in prompt_cfgs]
    outputs = llm.generate(prompt_texts, sampling_params)
    assert len(outputs) == len(prompt_cfgs), (
        f"Expected {len(prompt_cfgs)} outputs, got {len(outputs)}"
    )

    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        prompt = prompt_cfgs[i]["text"]
        expected = prompt_cfgs[i].get("expected")

        assert len(text) > 0, f"Empty output for prompt[{i}]: {prompt}"
        print(f"  prompt[{i}]: {prompt!r}")
        print(f"  output[{i}]: {text!r}")

        if expected:
            assert expected in text, (
                f"Expected '{expected}' in output for prompt[{i}], got: {text!r}"
            )


def _run_multimodal_test(llm: LLM, sampling_params: SamplingParams) -> None:
    """Run multimodal generation test."""
    from transformers import AutoTokenizer

    gen = _CFG.generate
    tokenizer = AutoTokenizer.from_pretrained(
        _CFG.model,
        trust_remote_code=True,
    )

    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_ids = []
    for t in stop_tokens:
        token_id = tokenizer.convert_tokens_to_ids(t)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
            stop_ids.append(token_id)

    if stop_ids:
        sampling_params = SamplingParams(
            **{**_CFG.generate.sampling, "stop_token_ids": stop_ids},
        )

    for i, prompt_cfg in enumerate(gen.prompts):
        assert isinstance(prompt_cfg, dict), (
            f"Multimodal prompts must be dicts, got: {type(prompt_cfg)}"
        )
        question = prompt_cfg["question"]
        asset_count = prompt_cfg.get("asset_count", 0)

        prompt_text = _build_multimodal_prompt(
            tokenizer,
            question,
            gen.modality,
            asset_count,
        )
        mm_data = _load_assets(gen.modality, gen.assets, asset_count)

        inputs = {"prompt": prompt_text, "multi_modal_data": mm_data}
        outputs = llm.generate(inputs, sampling_params=sampling_params)

        assert len(outputs) > 0, f"No output for prompt[{i}]"
        text = outputs[0].outputs[0].text
        assert isinstance(text, str), f"Output is not str for prompt[{i}]"

        print(f"  [{gen.modality} count={asset_count}] Q: {question}")
        print(f"  Output: {text!r}")


# ---------------------------------------------------------------------------
# Parametrized test entry point
# ---------------------------------------------------------------------------


_COMBOS = _CFG.generate.get_parametrize_combos()
_COMBO_IDS = [
    "-".join(f"{k}={v}" for k, v in combo.items()) or "default" for combo in _COMBOS
]


@pytest.mark.e2e
@pytest.mark.parametrize("combo", _COMBOS, ids=_COMBO_IDS)
def test_inference(combo: dict) -> None:
    """Smoke test: load model, generate, assert non-empty output.

    Each parametrize combo overrides engine params from the YAML config.
    """
    gen = _CFG.generate

    # Build LLM kwargs with parametrize overrides
    llm_kwargs = _CFG.engine_kwargs(**combo)

    # For multimodal: inject limit_mm_per_prompt
    max_assets = _get_max_asset_count()
    if gen.modality != "text" and max_assets > 0:
        llm_kwargs.setdefault(
            "limit_mm_per_prompt",
            {
                "image": 0,
                "video": 0,
                "audio": 0,
                gen.modality: max_assets,
            },
        )

    combo_desc = ", ".join(f"{k}={v}" for k, v in combo.items()) or "default"
    print(f"\n[{_MODEL}/{_CASE}] combo: {combo_desc}")
    print(f"[{_MODEL}/{_CASE}] model: {_CFG.model}")

    llm = LLM(**llm_kwargs)
    try:
        sampling_params = SamplingParams(**gen.sampling)

        if gen.modality == "text":
            _run_text_test(llm, sampling_params)
        else:
            _run_multimodal_test(llm, sampling_params)
    finally:
        del llm
        cleanup_dist_env_and_memory()
