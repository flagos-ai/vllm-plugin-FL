# SPDX-License-Identifier: Apache-2.0
"""Opt-in collocated prefill CP, independent of the FL platform backend."""

import os


def register():
    """Called in every vLLM process by the general-plugin entry point."""
    value = os.environ.get("VLLM_FL_QWEN_CP", "0")
    if value not in {"0", "1"}:
        raise ValueError("VLLM_FL_QWEN_CP must be 0 or 1")
    # Register the connector lazily on both P and D, including CP-off D. This
    # imports neither the FlagCX transport wrapper nor the optional FL platform.
    from vllm_fl import _register_flagcx_connector
    _register_flagcx_connector()
    # Independent of CP so the CP-off D can match P's GDN state layout.
    from vllm_fl.cp import state_dtype
    if state_dtype.enabled():
        state_dtype.apply_gdn_state_bf16()
    if value == "0":
        return

    # The standalone copy is intentionally PR-style: sequence ownership is
    # patched at the layer/model boundary and attention/GDN keep native weights.
    from vllm_fl.cp.prstyle import patch_model

    patch_model()
