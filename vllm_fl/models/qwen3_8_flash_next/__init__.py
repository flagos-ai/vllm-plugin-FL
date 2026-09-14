# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next model package."""

from typing import TYPE_CHECKING, Any

from .common.hyperconnection import (
    GatedResidualSimple,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
)

if TYPE_CHECKING:
    from .gpu.model import (
        Qwen3_8FlashNextForCausalLM,
        Qwen3_8FlashNextForConditionalGeneration,
    )
    from .gpu.mtp import Qwen3_8FlashNextMTP


def __getattr__(name: str) -> Any:
    if name in {
        "Qwen3_8FlashNextForCausalLM",
        "Qwen3_8FlashNextForConditionalGeneration",
        "Qwen3_8FlashNextMTP",
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpMTP",
    }:
        if name in {"Qwen3_8FlashNextMTP", "Qwen4ExpMTP"}:
            from .gpu.mtp import Qwen3_8FlashNextMTP

            return Qwen3_8FlashNextMTP

        from .gpu.model import (
            Qwen3_8FlashNextForCausalLM,
            Qwen3_8FlashNextForConditionalGeneration,
        )

        return {
            "Qwen3_8FlashNextForCausalLM": Qwen3_8FlashNextForCausalLM,
            "Qwen3_8FlashNextForConditionalGeneration": (
                Qwen3_8FlashNextForConditionalGeneration
            ),
            "Qwen4ExpForCausalLM": Qwen3_8FlashNextForCausalLM,
            "Qwen4ExpForConditionalGeneration": (
                Qwen3_8FlashNextForConditionalGeneration
            ),
        }[name]
    raise AttributeError(name)


__all__ = [
    "GatedResidualSimple",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "Qwen3_8FlashNextForCausalLM",
    "Qwen3_8FlashNextForConditionalGeneration",
    "Qwen3_8FlashNextMTP",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMTP",
]
