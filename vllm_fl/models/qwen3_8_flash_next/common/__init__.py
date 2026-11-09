# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common Qwen3.8-Flash-Next model components."""


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hyperconnection import (
        GatedResidualSimple,
        GroupedGemmaRMSNorm,
        HyperConnectionBase,
        HyperConnectionConfig,
    )


def __getattr__(name):
    if name in __all__:
        from . import hyperconnection

        return getattr(hyperconnection, name)
    raise AttributeError(name)


__all__ = [
    "GatedResidualSimple",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
]
