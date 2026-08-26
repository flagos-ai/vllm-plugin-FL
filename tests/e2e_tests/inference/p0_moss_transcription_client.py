# Copyright (c) 2025 BAAI. All rights reserved.

"""Exercise the OpenAI transcription endpoint for MOSS-Transcribe-Diarize."""

from __future__ import annotations

import argparse

import requests

from vllm.assets.audio import AudioAsset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    audio_path = AudioAsset("mary_had_lamb").get_local_path()
    with open(audio_path, "rb") as audio_file:
        response = requests.post(
            f"{args.base_url}/v1/audio/transcriptions",
            data={
                "model": args.model,
                "response_format": "json",
                "temperature": "0",
            },
            files={"file": (audio_path.name, audio_file, "audio/wav")},
            timeout=600,
        )
    response.raise_for_status()
    payload = response.json()
    text = payload.get("text", "").strip()
    print(f"P0_TRANSCRIPTION_OUTPUT={text!r}")
    assert text, f"Empty transcription response: {payload!r}"
    assert "mary" in text.casefold() or "lamb" in text.casefold(), text


if __name__ == "__main__":
    main()
