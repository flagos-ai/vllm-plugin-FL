# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Exercise upstream CosyVoice3 zero-shot synthesis through the FL platform."""

import argparse
from importlib.metadata import version
from pathlib import Path


def validate_audio(audio, sample_rate):
    import torch

    if sample_rate != 24000 or audio.ndim != 2 or audio.shape[0] != 1:
        raise RuntimeError("Expected 24 kHz mono speech")
    signal = audio.float()
    if not signal.numel() or not torch.isfinite(signal).all().item():
        raise RuntimeError("Synthesis returned empty or non-finite audio")
    if signal.std(unbiased=False).item() < 1e-3 or signal.mean().abs().item() > 0.5:
        raise RuntimeError("Synthesis returned silent or DC-biased audio")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-wav", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.model_dir / "cosyvoice3.yaml").is_file():
        parser.error("--model-dir must contain the local CosyVoice3 checkpoint")
    if not args.reference_wav.is_file():
        parser.error("--reference-wav must be an existing file")
    if not args.text.strip() or not args.reference_text.strip():
        parser.error("--text and --reference-text must be nonempty")
    if args.output.exists() or args.output.suffix.lower() != ".wav":
        parser.error("--output must be a new WAV file")
    if version("vllm").split("+", 1)[0] != "0.24.0":
        parser.error("This FL example targets vLLM 0.24.0")

    import soundfile
    import torch
    from cosyvoice.cli.cosyvoice import AutoModel

    from vllm.logger import init_logger
    from vllm.platforms import current_platform

    from vllm_fl.platform import PlatformFL

    if (
        not isinstance(current_platform, PlatformFL)
        or current_platform.vendor_name != "metax"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Select the FL platform on MetaX with VLLM_PLUGINS=fl")

    model = AutoModel(
        model_dir=str(args.model_dir), load_vllm=True, load_trt=False, fp16=True
    )
    engine = model.model.llm.vllm
    if engine.model_config.hf_config.architectures != ["Qwen2ForCausalLM"]:
        raise RuntimeError("Expected CosyVoice3 to use vLLM's Qwen2 speech-token model")

    chunks = [
        result["tts_speech"].detach().cpu()
        for result in model.inference_zero_shot(
            args.text,
            f"You are a helpful assistant.<|endofprompt|>{args.reference_text}",
            str(args.reference_wav),
            stream=False,
        )
    ]
    if not chunks:
        raise RuntimeError("CosyVoice3 produced no speech chunks")
    audio = torch.cat(chunks, dim=-1)
    validate_audio(audio, model.sample_rate)
    soundfile.write(args.output, audio.float().squeeze(0).numpy(), model.sample_rate)
    init_logger(__name__).info(
        "Saved %.2f seconds of CosyVoice3 audio to %s",
        audio.shape[-1] / model.sample_rate,
        args.output,
    )


if __name__ == "__main__":
    main()
