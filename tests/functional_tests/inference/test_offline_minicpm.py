# Copyright (c) 2025 BAAI. All rights reserved.

"""
Offline inference tests for MiniCPM multimodal model.
Tests audio input processing with various audio counts.

Need to install vllm audio extension:
pip install vllm[audio]
"""

import os

import pytest
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset

MODEL_PATH = "/data/models/MiniCPM"
pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason=f"Model not found: {MODEL_PATH}"
)

AUDIO_ASSETS = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]
QUESTION_PER_AUDIO_COUNT = {
    0: "What is 1+1?",
    1: "What is recited in the audio?",
    2: "What sport and what nursery rhyme are referenced?",
}
AUDIO_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>' }}"
    "{% endif %}"
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def _build_prompt(tokenizer, question, audio_count):
    audio_placeholder = "(<audio>./</audio>)" * audio_count
    messages = [{"role": "user", "content": f"{audio_placeholder}\n{question}"}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=AUDIO_CHAT_TEMPLATE,
    )


def _build_stop_token_ids(tokenizer):
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    return [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens]


class TestMiniCPMO:
    @pytest.mark.parametrize("audio_count", [0, 1, 2])
    def test_inference(self, tokenizer, audio_count):
        question = QUESTION_PER_AUDIO_COUNT[audio_count]
        prompt = _build_prompt(tokenizer, question, audio_count)
        stop_token_ids = _build_stop_token_ids(tokenizer)

        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            max_model_len=4096,
            enforce_eager=True,
            max_num_seqs=2,
            seed=0,
            limit_mm_per_prompt={"image": 0, "video": 0, "audio": audio_count},
            load_format="dummy",
            tensor_parallel_size=2,
        )

        sampling_params = SamplingParams(
            temperature=0.2, max_tokens=128, stop_token_ids=stop_token_ids
        )

        mm_data = {}
        if audio_count > 0:
            mm_data = {
                "audio": [
                    asset.audio_and_sample_rate
                    for asset in AUDIO_ASSETS[:audio_count]
                ]
            }

        inputs = {"prompt": prompt, "multi_modal_data": mm_data}
        outputs = llm.generate(inputs, sampling_params=sampling_params)

        assert len(outputs) > 0
        generated_text = outputs[0].outputs[0].text

        print(f"\n[Audio Count: {audio_count}]")
        print(f"Question: {question}")
        print(f"Generated Output: {generated_text}")

        assert isinstance(generated_text, str)
