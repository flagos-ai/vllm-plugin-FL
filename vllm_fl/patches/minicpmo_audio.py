# Copyright (c) 2026 BAAI. All rights reserved.

from collections.abc import Mapping


def patch_minicpmo_audio_budget() -> bool:
    """Include audio in vLLM 0.32's MiniCPM-O encoder token budget."""
    from vllm.model_executor.models.minicpmo import MiniCPMOProcessingInfo

    # Upstream inherits MiniCPM-V's image/video-only estimate, so audio-only
    # requests receive a zero encoder cache. Preserve a native override (or
    # this patch on repeated registration) when one is already present.
    if "get_mm_max_tokens_per_item" in MiniCPMOProcessingInfo.__dict__:
        return False

    inherited = MiniCPMOProcessingInfo.get_mm_max_tokens_per_item

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        return {
            **inherited(self, seq_len, mm_counts),
            "audio": self.get_max_audio_tokens(),
        }

    MiniCPMOProcessingInfo.get_mm_max_tokens_per_item = get_mm_max_tokens_per_item
    return True
