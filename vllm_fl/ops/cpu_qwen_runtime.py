"""Thin vLLM-to-FlagGems integration for ARM CPU Qwen models.

Machine policy belongs to the launcher profile.  This module deliberately
does not set Triton, OpenMP, scheduler, or kernel-selection environment
variables; it only installs the version-locked vLLM compatibility hooks and
registers the FlagGems runtime lazily in the model-loading process.
"""

from __future__ import annotations

_ACTIVE = False


def _print_runtime_banner() -> None:
    print(
        "[vllm_fl] ARM Qwen runtime active "
        "(quant=FlagGems/libtriton_jit, gdn=FlagGems/native)",
        flush=True,
    )


def enable_qwen_runtime(*, verbose: bool = True) -> bool:
    """Install the stock-vLLM hooks and the FlagGems ARM runtime once."""
    global _ACTIVE
    if _ACTIVE:
        return False

    from vllm_fl.patches.arm_cpu_vllm_0240 import (
        install_arm_cpu_vllm_0240_compat,
    )

    install_arm_cpu_vllm_0240_compat()
    from vllm_fl.ops.cpu_qwen_gdn import install_vllm_gdn_bridge

    install_vllm_gdn_bridge()
    from flag_gems.runtime.backend._arm.q4 import enable_vllm_q4_codegen

    enable_vllm_q4_codegen(verbose=verbose, runtime="libtriton_jit")
    from vllm_fl.ops.cpu_w8_greedy import install_w8_cached_greedy

    install_w8_cached_greedy()
    from flag_gems.integrations.vllm import maybe_install_kernel_coverage

    maybe_install_kernel_coverage()
    _ACTIVE = True
    if verbose:
        _print_runtime_banner()
    return True


__all__ = [
    "enable_qwen_runtime",
]
