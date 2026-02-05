import pytest
from dataclasses import asdict
from typing import Any, NamedTuple

from transformers import AutoTokenizer
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.audio import AudioAsset

MODEL_PATH = "/models/MiniCMP"

audio_assets = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]
question_per_audio_count = {
    0: "What is 1+1?",
    1: "What is recited in the audio?",
    2: "What sport and what nursery rhyme are referenced?",
}


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str | None = None
    prompt_token_ids: dict[str, list[int]] | None = None
    multi_modal_data: dict[str, Any] | None = None
    stop_token_ids: list[int] | None = None
    lora_requests: list[Any] | None = None


def get_minicpmo_request(question: str, audio_count: int) -> ModelRequestData:
    model_name = MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        enforce_eager=True,
        max_num_seqs=2,
        limit_mm_per_prompt={"audio": audio_count},
        load_format="dummy",
    )

    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    audio_placeholder = "(<audio>./</audio>)" * audio_count
    audio_chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>' }}{% endif %}"

    messages = [{"role": "user", "content": f"{audio_placeholder}\n{question}"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=audio_chat_template,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
    )


class TestMiniCPMO:
    @pytest.mark.parametrize("audio_count", [0, 1, 2])
    def test_inference(self, audio_count):
        question = question_per_audio_count[audio_count]
        req_data = get_minicpmo_request(question, audio_count)

        default_limits = {"image": 0, "video": 0, "audio": 0}
        req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
            req_data.engine_args.limit_mm_per_prompt or {}
        )

        engine_args_dict = asdict(req_data.engine_args)
        engine_args_dict["seed"] = 0

        llm = LLM(**engine_args_dict)

        sampling_params = SamplingParams(
            temperature=0.2, max_tokens=128, stop_token_ids=req_data.stop_token_ids
        )

        mm_data = {}
        if audio_count > 0:
            mm_data = {
                "audio": [
                    asset.audio_and_sample_rate for asset in audio_assets[:audio_count]
                ]
            }

        inputs = {"prompt": req_data.prompt, "multi_modal_data": mm_data}

        outputs = llm.generate(inputs, sampling_params=sampling_params)

        assert len(outputs) > 0
        generated_text = outputs[0].outputs[0].text

        print(f"\n[Audio Count: {audio_count}]")
        print(f"Question: {question}")
        print(f"Generated Output: {generated_text}")

        assert isinstance(generated_text, str)
