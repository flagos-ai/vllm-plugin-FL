# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""M3 execution boundary and audit, scoped to this worker/runner pair."""

import torch

from .model_runner import ModelRunnerFL
from .worker import WorkerFL


class MiniMaxModelRunnerFL(ModelRunnerFL):
    def _model_forward(self, *args, **kwargs):
        output = super()._model_forward(*args, **kwargs)
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize(self.device)
        return output

    def capture_model(self, *args, **kwargs):
        result = super().capture_model(*args, **kwargs)
        from vllm_fl.patches.minimax_m3_metax.bootstrap import audit_effective

        audit_effective("after_capture", self)
        return result


class MiniMaxWorkerFL(WorkerFL):
    def oot_registration_kwargs(self):
        # The M3 model supplies its router and local RoutedExperts factory.
        return {"install_moe_hooks": False}

    def flaggems_registration_kwargs(self):
        from vllm_fl.ops.minimax_m3.aten import MiniMaxRegistrar

        return {"registrar": MiniMaxRegistrar}

    def determine_available_memory(self):
        result = super().determine_available_memory()
        from vllm_fl.patches.minimax_m3_metax.bootstrap import audit_memory_budget

        audit_memory_budget(self, result)
        return result

    def warmup_kernels(self):
        from vllm.model_executor.warmup.kernel_warmup import kernel_warmup

        kernel_warmup(self)
        print("M3_KERNEL_WARMUP_COMPLETE", flush=True)

    def model_runner_type(self):
        return MiniMaxModelRunnerFL

    def init_device(self):
        from vllm_fl.patches.minimax_m3_metax.bootstrap import initialize

        initialize()
        return super().init_device()
