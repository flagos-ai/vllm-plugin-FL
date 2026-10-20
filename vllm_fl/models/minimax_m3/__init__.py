# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors

"""Opt-in MiniMax classes; unchanged model code is supplied by vLLM."""

__all__ = [
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
    "MiniMaxM3MTP",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from .bindings import model_class

    return model_class(name)
