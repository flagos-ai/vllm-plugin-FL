# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MetaX entrypoint; preserve vLLM's model and checkpoint layout."""

from vllm_fl.ops.minimax_m3.integration import install

install()

from vllm.models.minimax_m3.nvidia.model import (  # noqa: E402,F401
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)
