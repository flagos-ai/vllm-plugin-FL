# SPDX-License-Identifier: Apache-2.0
"""Vendor-neutral worker-runtime optimizations for vLLM V1 execution.

Exposes HBM locality-preserving KV block allocation (`LocalityAwareFreeBlockQueue`),
chunked low-VRAM / sort-free Top-P sampling hooks (`install_sort_free_sampler_hooks`),
and fused GPU step-preparation + async output ring hooks (`StepPrepConfig`,
`install_step_prep`).
"""

from vllm_fl.worker.locality_block_queue import (
    LocalityAwareFreeBlockQueue,
    install_locality_free_block_queue,
)
from vllm_fl.worker.sampler import install_sort_free_sampler_hooks
from vllm_fl.worker.step_prep import (
    StepPrepConfig,
    install_step_prep,
    install_step_prep_on_runner_module,
)

__all__ = [
    "LocalityAwareFreeBlockQueue",
    "StepPrepConfig",
    "install_locality_free_block_queue",
    "install_sort_free_sampler_hooks",
    "install_step_prep",
    "install_step_prep_on_runner_module",
]
