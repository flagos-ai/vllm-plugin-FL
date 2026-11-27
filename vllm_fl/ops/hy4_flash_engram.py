# SPDX-License-Identifier: Apache-2.0
"""Hy4 Flash Engram operations under the shared backend dispatch policy."""

from vllm_fl.dispatch import CachedOp

lookup_norm = CachedOp("hy4_flash_engram_lookup_norm")
gate = CachedOp("hy4_flash_engram_gate")
residual = CachedOp("hy4_flash_engram_residual")
