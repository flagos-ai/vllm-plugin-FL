# Copyright (c) 2026 BAAI. All rights reserved.
"""vLLM V2 runner with Iluvatar device-buffer initialization."""

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class IluvatarGPUModelRunnerV2(GPUModelRunner):
    def __init__(self, vllm_config, device):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import patch_v2_uva

        # CoreX cannot use CUDA UVA for these scheduler buffers. Configure this
        # before upstream allocates its persistent host/device input buffers.
        patch_v2_uva()
        super().__init__(vllm_config, device)
