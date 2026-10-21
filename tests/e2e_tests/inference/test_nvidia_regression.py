# Copyright 2026 FlagOS Contributors
"""Explicit NVIDIA new-model regression; not part of the automatic device matrix."""

import contextlib
import os
from pathlib import Path

import pytest

from tests.utils.nvidia_validation import (
    InferenceEvidence,
    checkpoint_facts,
    load_manual_case,
    validate_resources,
)

if not os.environ.get("FL_NVIDIA_MODEL_PATH"):
    pytest.skip("Selected through cuda/run_regression.py only", allow_module_level=True)


def multimodal_cases(assets, paired):
    """Real local media passed through the upstream chat processor."""
    from PIL import Image

    assets.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (448, 448))
    for color, box in zip(
        ("red", "green", "blue", "yellow"),
        (
            (0, 0, 224, 224),
            (224, 0, 448, 224),
            (0, 224, 224, 448),
            (224, 224, 448, 448),
        ),
        strict=True,
    ):
        # Use full-intensity RGB green, as in the validated video frames.
        image.paste((0, 255, 0) if color == "green" else color, box)
    image_path = assets / "quadrants.png"
    image.save(image_path)
    image_part = {"type": "image_url", "image_url": {"url": image_path.as_uri()}}
    colors = ["red", "green", "blue", "yellow"]
    if paired:
        green = assets / "green.png"
        Image.new("RGB", (224, 224), (0, 255, 0)).save(green)
        return [
            (
                "paired_images",
                [
                    image_part,
                    {"type": "image_url", "image_url": {"url": green.as_uri()}},
                    {
                        "type": "text",
                        "text": "For the FIRST image, name the four quadrant colors in order: top-left, top-right, bottom-left, bottom-right. Then name the dominant color of the SECOND image. Reply in English.",
                    },
                ],
                colors + ["green"],
            )
        ]
    import av
    import numpy as np

    video = assets / "colors.mp4"
    with av.open(str(video), mode="w") as encoder:
        stream = encoder.add_stream("libx264", rate=1)
        stream.width = stream.height = 224
        stream.pix_fmt = "yuv420p"
        for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)):
            pixels = np.empty((224, 224, 3), dtype=np.uint8)
            pixels[:] = color
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(pixels, format="rgb24")
            ):
                encoder.mux(packet)
        for packet in stream.encode():
            encoder.mux(packet)
    return [
        ("text", "What is 2 + 2? Give only the number.", ["4"]),
        (
            "image",
            [
                image_part,
                {
                    "type": "text",
                    "text": "Name the four quadrant colors in order: top-left, top-right, bottom-left, bottom-right. Reply in English.",
                },
            ],
            colors,
        ),
        (
            "video",
            [
                {"type": "video_url", "video_url": {"url": video.as_uri()}},
                {
                    "type": "text",
                    "text": "List the screen colors in this video in temporal order. Reply in English.",
                },
            ],
            colors,
        ),
    ]


@pytest.mark.e2e
def test_nvidia_regression():
    import torch

    from vllm import LLM, SamplingParams

    cfg, checks = load_manual_case(
        os.environ["FL_TEST_MODEL"],
        os.environ["FL_TEST_CASE"],
        os.environ["FL_NVIDIA_MODEL_PATH"],
    )
    evidence = InferenceEvidence(
        cfg, checks["runner"], os.environ["FL_NVIDIA_RESULT_PATH"]
    )
    llm = None
    try:
        evidence.data["checkpoint"] = checkpoint_facts(
            cfg.model, checks["expected_architectures"]
        )
        devices = [
            {
                "free_bytes": torch.cuda.mem_get_info(i)[0],
                "total_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ]
        validate_resources(devices, cfg, checks["minimum_memory_gib"])
        assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == str(
            int(checks["runner"] == "v2")
        )
        kwargs = cfg.engine_kwargs(
            worker_extension_cls="tests.utils.nvidia_validation.NvidiaWorkerEvidence"
        )
        if checks["kind"] != "text":
            assets = (
                Path(os.environ["FL_NVIDIA_RESULT_PATH"]).resolve().parent / "media"
            )
            cases = multimodal_cases(assets, checks["kind"] == "paired_images")
            kwargs["allowed_local_media_path"] = str(assets)
        else:
            cases = [
                (f"text_{i}", p["text"], p["accepted_exact"])
                for i, p in enumerate(cfg.generate.prompts)
            ]
        llm = LLM(**kwargs)
        evidence.workers(llm, checks["expected_model_class"])
        sampling = SamplingParams(**cfg.sampling_kwargs())
        for name, content, expected in cases:
            messages = [{"role": "user", "content": content}]
            first = llm.chat(
                messages,
                sampling,
                chat_template_kwargs=checks["chat_template_kwargs"],
                use_tqdm=False,
            )
            repeat = llm.chat(
                messages,
                sampling,
                chat_template_kwargs=checks["chat_template_kwargs"],
                use_tqdm=False,
            )
            mode = (
                "exact"
                if checks["kind"] == "text" or name == "text"
                else "ordered_terms"
            )
            evidence.outputs(name, first, repeat, [expected], match=mode)
        evidence.finish()
    except Exception as exc:
        evidence.finish(exc)
        raise
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.llm_engine.engine_core.shutdown()
