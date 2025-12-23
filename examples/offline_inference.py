import torch
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig

if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
    # Create an LLM.
    llm = LLM(model="Qwen/Qwen3-4B", max_num_batched_tokens=16384, max_num_seqs=2048)

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
