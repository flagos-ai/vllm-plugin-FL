# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors

"""Explicit, idempotent bindings limited to vLLM's MiniMax M3 modules.

This is a runtime hook, not a source rewrite. Install before constructing any
M3 model. Keep the vLLM implementation of loading, metadata and preprocessing.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

_INSTALLED = False
_BINDINGS = []


def validate_upstream():
    spec = importlib.util.find_spec("vllm")
    root = Path(next(iter(spec.submodule_search_locations))) / "models/minimax_m3"
    expected = json.loads(Path(__file__).with_name("upstream.json").read_text())
    for relative, digest in expected["files"].items():
        path = root / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError(
                "M3 requires clean vLLM v0.24.0 model sources: " + relative
            )


def install():
    global _INSTALLED
    if _INSTALLED:
        for module, name, value in _BINDINGS:
            if getattr(module, name) is not value:
                raise RuntimeError(
                    "M3 binding changed: " + module.__name__ + "." + name
                )
        return
    from vllm_fl.minimax_m3_config import load

    profile = load()
    if not profile.framework.flagos_msa:
        raise RuntimeError("This profile requires the accepted MSA path")
    validate_upstream()
    from vllm.models.minimax_m3.common import indexer, sparse_attention, vision_tower
    from vllm.models.minimax_m3.nvidia import model as native, mtp

    from . import model, vision
    from vllm_fl.attention.minimax_m3 import bridge as attention
    from vllm_fl.ops.minimax_m3 import layers, msa
    from vllm_fl.ops.minimax_m3.vision import M3VisionAttentionFL

    updates = []

    def bind(module, name, value):
        if not hasattr(module, name):
            raise RuntimeError("Missing M3 binding: " + module.__name__ + "." + name)
        updates.append((module, name, value))

    for name in (
        "MiniMAXGemmaRMSNorm",
        "MiniMaxM3MoE",
        "MiniMaxM3Attention",
        "MiniMaxM3SparseAttention",
    ):
        replacement = getattr(model, name)
        bind(native, name, replacement)
        if hasattr(mtp, name):
            bind(mtp, name, replacement)
    if profile.framework.flagos_swiglu:
        bind(native, "SiluAndMulWithClamp", layers.SiluAndMulWithClampFL)
    for name in ("MiniMaxVLPatchEmbed", "MiniMaxVLVisionModel"):
        bind(vision_tower, name, getattr(vision, name))
    bind(vision_tower, "MMEncoderAttention", M3VisionAttentionFL)
    bind(native, "MiniMaxVLVisionModel", vision.MiniMaxVLVisionModel)
    for name in (
        "minimax_m3_index_decode",
        "minimax_m3_index_score",
        "minimax_m3_index_topk",
    ):
        bind(indexer, name, getattr(msa, name))
    bind(sparse_attention, "minimax_m3_sparse_attn", attention.sparse_attn)
    bind(sparse_attention, "minimax_m3_sparse_attn_decode", attention.sparse_decode)
    previous = [(module, name, getattr(module, name)) for module, name, _ in updates]
    try:
        for module, name, value in updates:
            setattr(module, name, value)
    except Exception:
        for module, name, value in reversed(previous):
            setattr(module, name, value)
        raise
    _BINDINGS[:] = updates
    _INSTALLED = True
