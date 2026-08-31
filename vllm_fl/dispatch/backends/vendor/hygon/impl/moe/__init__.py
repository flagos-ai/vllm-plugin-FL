"""Hygon modular MoE experts, kernels, and tuning helpers.

Keep this initializer lightweight: native-op fallbacks also import individual
MoE helpers and must not eagerly load vLLM's full modular-MoE stack.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "HygonTritonExpertsFL":
        from .triton_experts import HygonTritonExpertsFL

        return HygonTritonExpertsFL
    raise AttributeError(name)


__all__ = ["HygonTritonExpertsFL"]
