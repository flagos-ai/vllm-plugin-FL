"""T-Head compiler launch compatibility, isolated from model semantics."""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


class _PrefillHeadTile:
    """Pad the affected BF16 GQA tile while retaining the upstream kernel."""

    def __init__(self, original, jit):
        self._original = original
        self._jit = jit
        self._reported = False

    def __getitem__(self, grid):
        original_launch = self._original[grid]
        padded_launch = self._jit[grid]

        def launch(*args, **kwargs):
            def argument(index, name):
                return args[index] if len(args) > index else kwargs.get(name)

            query = argument(0, "q_ptr")
            supported = (
                isinstance(query, torch.Tensor)
                and query.dtype == torch.bfloat16
                and argument(10, "gqa_group_size") == 8
                and argument(11, "head_dim") == 128
                and argument(12, "max_topk") == 16
                and kwargs.get("BLOCK_SIZE_Q") == 1
                and kwargs.get("BLOCK_SIZE_K") == 128
                and kwargs.get("USE_FP8") is False
                and not {
                    "BLOCK_SIZE_H",
                    "BLOCK_SIZE_QH",
                    "BLOCK_SIZE_D",
                    "BLOCK_SIZE_T",
                }.intersection(kwargs)
            )
            if not supported:
                return original_launch(*args, **kwargs)
            if not self._reported:
                logger.warning(
                    "Using PPU MiniMax-M3 Prefill head tile 16 for BF16 GQA8/D128/topk16"
                )
                self._reported = True
            return padded_launch(
                *args,
                BLOCK_SIZE_H=16,
                BLOCK_SIZE_QH=16,
                BLOCK_SIZE_D=128,
                BLOCK_SIZE_T=16,
                **kwargs,
            )

        return launch

    def __getattr__(self, name):
        return getattr(self._original, name)


def _patch_prefill_head_tile(sparse_attn) -> bool:
    """Install once on PPU; other platforms and shapes retain upstream launch."""
    current = sparse_attn._gqa_sparse_fwd_kernel
    if isinstance(current, _PrefillHeadTile):
        return False
    jit = getattr(current, "fn", None)
    expected = {0: "q_ptr", 10: "gqa_group_size", 11: "head_dim", 12: "max_topk"}
    names = getattr(jit, "arg_names", ())
    if (
        getattr(jit, "__name__", None) != "_gqa_sparse_fwd_kernel"
        or len(names) < 13
        or any(names[i] != name for i, name in expected.items())
    ):
        raise RuntimeError(
            "Unsupported MiniMax-M3 Prefill kernel signature for PPU head-tile compatibility"
        )
    sparse_attn._gqa_sparse_fwd_kernel = _PrefillHeadTile(current, jit)
    return True


def apply_thead_patches(model_config) -> bool:
    """Worker model-load hook; other vendors never import these launch overrides."""
    architectures = getattr(
        getattr(model_config, "hf_config", None), "architectures", ()
    )
    if not set(architectures or ()).intersection(
        {
            "MiniMaxM3SparseForCausalLM",
            "MiniMaxM3SparseForConditionalGeneration",
        }
    ):
        return False
    try:
        from vllm.models.minimax_m3.common.ops import sparse_attn
    except ModuleNotFoundError as error:
        if error.name not in {
            "vllm.models",
            "vllm.models.minimax_m3",
            "vllm.models.minimax_m3.common",
            "vllm.models.minimax_m3.common.ops",
        }:
            raise
        return False
    return _patch_prefill_head_tile(sparse_attn)
