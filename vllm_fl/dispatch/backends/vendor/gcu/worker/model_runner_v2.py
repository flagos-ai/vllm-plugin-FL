# Copyright (c) 2026 BAAI. All rights reserved.
"""GCU adaptation of the upstream vLLM 0.32 V2 runner."""

from ..v2_buffers import install_gcu_v2_buffers
from ..v2_runtime import install_gcu_v2_runtime

# Install before upstream imports bind buffer classes and graph helpers.
install_gcu_v2_buffers()
install_gcu_v2_runtime()

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class GPUModelRunnerFLV2(GPUModelRunner):
    """Retain upstream execution, specializing GCU setup and dispatch warmup."""

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        # Resolve FL dispatch before Dynamo traces the first profile/capture.
        from vllm_fl.dispatch import warmup_cached_ops

        warmup_cached_ops()
