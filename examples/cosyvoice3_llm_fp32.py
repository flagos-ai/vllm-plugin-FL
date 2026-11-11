# SPDX-License-Identifier: Apache-2.0
"""Offline speech-token smoke test for an exported original-FP32 CosyVoice3 LLM.

Install vLLM 0.24.0, the matching FL checkout, and Ascend torch-npu/CANN.
Set VLLM_PLUGINS=fl, USE_FLAGGEMS=0 and ASCEND_RT_VISIBLE_DEVICES=0.
Provide --model with a Qwen2ForCausalLM-format export of the original FP32
speech-LLM tensors from FunAudioLLM/Fun-CosyVoice3-0.5B-2512. Casting a BF16
export to FP32 is insufficient.

Pass --prompt-embeds with a torch.save-serialized FP32 tensor of shape [tokens,
896] containing speech-LLM conditioning embeddings. Without it, random embeddings
provide a backend smoke test. Output is speech token IDs, not synthesized audio.

Usage: python examples/cosyvoice3_llm_fp32.py --model MODEL --prompt-embeds EMBEDS.pt
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Original FP32 Qwen2-format export"
    )
    parser.add_argument("--eager", action="store_true", help="Use eager execution")
    parser.add_argument(
        "--prompt-embeds",
        help="torch.save file containing FP32 [tokens, 896] embeddings",
    )
    args = parser.parse_args()

    import torch

    from vllm import LLM, SamplingParams

    if args.prompt_embeds:
        embeddings = torch.load(
            args.prompt_embeds, map_location="cpu", weights_only=True
        )
        if not isinstance(embeddings, torch.Tensor):
            parser.error("--prompt-embeds must contain a tensor")
        if embeddings.ndim != 2 or embeddings.shape[1] != 896:
            parser.error("--prompt-embeds must have shape [tokens, 896]")
        if not 1 <= embeddings.shape[0] <= 4080:
            parser.error("--prompt-embeds must contain 1 to 4080 tokens")
        if embeddings.dtype != torch.float32 or not torch.isfinite(embeddings).all():
            parser.error("--prompt-embeds must contain finite FP32 values")
        prompts = [{"prompt_embeds": embeddings.contiguous()}]
    else:
        torch.manual_seed(1105)
        prompts = [{"prompt_embeds": torch.randn(n, 896) * 0.02} for n in (4, 9)]
    execution = (
        {"enforce_eager": True}
        if args.eager
        else {
            "compilation_config": {
                "mode": 3,
                "cudagraph_mode": "FULL_DECODE_ONLY",
            }
        }
    )
    llm = LLM(
        model=args.model,
        dtype="float32",
        tensor_parallel_size=1,
        skip_tokenizer_init=True,
        enable_prompt_embeds=True,
        enable_prefix_caching=False,
        max_model_len=4096,
        max_num_seqs=2,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.18,
        **execution,
    )
    params = SamplingParams(
        temperature=0,
        min_tokens=16,
        max_tokens=16,
        ignore_eos=True,
        detokenize=False,
        logprobs=5,
    )
    for result in llm.generate(prompts, params, use_tqdm=False):
        print(result.outputs[0].token_ids)


if __name__ == "__main__":
    main()
