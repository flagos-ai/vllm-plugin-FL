# SPDX-License-Identifier: Apache-2.0
"""Offline speech-token smoke test for an exported BF16 CosyVoice3 LLM.

Install vLLM 0.24.0, the matching FL checkout, and Ascend torch-npu/CANN.
Set VLLM_PLUGINS=fl, USE_FLAGGEMS=0 and ASCEND_RT_VISIBLE_DEVICES=0.
Provide --model with a Qwen2ForCausalLM-format export of the BF16
speech-LLM tensors from FunAudioLLM/Fun-CosyVoice3-0.5B-2512.

Pass --prompt-embeds with a torch.save-serialized floating-point tensor of shape [tokens,
896] containing speech-LLM conditioning embeddings. Without it, random embeddings
provide a backend smoke test. Output is speech token IDs, not synthesized audio.

Download the complete official model repository first, then export its speech LLM:
python examples/cosyvoice3_llm.py --export-from ORIGINAL_MODEL --model BF16_MODEL
Run: python examples/cosyvoice3_llm.py --model BF16_MODEL --prompt-embeds EMBEDS.pt
"""

import argparse


def export_bf16(source, destination):
    """Convert the official 2512 speech-LLM state dict to a Qwen2 checkpoint."""
    import json
    from pathlib import Path

    import torch
    from safetensors.torch import save_file
    from transformers import Qwen2Config

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Export destination already exists: {destination}")
    config = Qwen2Config.from_pretrained(
        source / "CosyVoice-BlankEN", local_files_only=True
    )
    if config.hidden_size != 896 or config.num_hidden_layers != 24:
        raise ValueError("Expected the Fun-CosyVoice3-0.5B-2512 Qwen2 configuration")
    state = torch.load(
        source / "llm.pt", map_location="cpu", weights_only=True, mmap=True
    )
    weights = {
        name.removeprefix("llm.model."): tensor.to(torch.bfloat16).contiguous()
        for name, tensor in state.items()
        if name.startswith(("llm.model.model.layers.", "llm.model.model.norm."))
    }
    weights["model.embed_tokens.weight"] = (
        state["speech_embedding.weight"].to(torch.bfloat16).contiguous()
    )
    weights["lm_head.weight"] = (
        state["llm_decoder.weight"].to(torch.bfloat16).contiguous()
    )
    if len(weights) != 291 or any(
        weights[name].shape != (6761, 896)
        for name in ("model.embed_tokens.weight", "lm_head.weight")
    ):
        raise ValueError("Unexpected Fun-CosyVoice3-0.5B-2512 speech-LLM weights")
    config.architectures = ["Qwen2ForCausalLM"]
    config.vocab_size = 6761
    config.tie_word_embeddings = False
    config.dtype = torch.bfloat16
    config.bos_token_id = config.pad_token_id = None
    config.eos_token_id = 6562
    destination.mkdir(parents=True)
    config.save_pretrained(destination)
    (destination / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 6562})
    )
    save_file(weights, destination / "model.safetensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="BF16 Qwen2-format speech-LLM export"
    )
    parser.add_argument(
        "--export-from", help="Official 2512 model directory; export BF16 and exit"
    )
    parser.add_argument("--eager", action="store_true", help="Use eager execution")
    parser.add_argument(
        "--prompt-embeds",
        help="torch.save file containing BF16/FP32 [tokens, 896] embeddings",
    )
    args = parser.parse_args()

    if args.export_from:
        if args.eager or args.prompt_embeds:
            parser.error("--export-from cannot be combined with inference options")
        export_bf16(args.export_from, args.model)
        return

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
        if (
            embeddings.dtype not in (torch.bfloat16, torch.float32)
            or not torch.isfinite(embeddings).all()
        ):
            parser.error("--prompt-embeds must contain finite BF16 or FP32 values")
        embeddings = embeddings.to(torch.bfloat16).contiguous()
        if not torch.isfinite(embeddings).all():
            parser.error("--prompt-embeds values must remain finite in BF16")
        prompts = [{"prompt_embeds": embeddings}]
    else:
        torch.manual_seed(1105)
        prompts = [
            {"prompt_embeds": torch.randn(n, 896, dtype=torch.bfloat16) * 0.02}
            for n in (4, 9)
        ]
    execution = (
        {"enforce_eager": True}
        if args.eager
        else {
            "compilation_config": {
                "cudagraph_mode": "FULL_DECODE_ONLY",
            }
        }
    )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
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
