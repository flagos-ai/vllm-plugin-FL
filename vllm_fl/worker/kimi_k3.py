"""K3 lifecycle integration; arithmetic remains in the operator modules."""

from vllm_fl.distributed.kimi_k3_pp import HostPackedPPHandler
from vllm_fl.worker.kimi_k3_graph import MetaXModelCudaGraphManager
from vllm_fl.models.kimi_k3_context import execution_config
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm_fl.worker.pipeline_input import AsyncIntermediateTensors

from vllm_fl.quantization.mxfp4.kimi_k3 import K3Bf16MoEMethod
from vllm_fl.worker.kimi_k3_workspace import MIN_PREFILL_TOKENS, TP8PrefillWorkspace


class K3ModelRunner(GPUModelRunner):
    def __init__(self, vllm_config, device, *, diagnostics):
        self.k3_execution = execution_config(vllm_config)
        self.pp_handler_cls = (
            HostPackedPPHandler
            if self.k3_execution.host_pp_feedback
            else GPUModelRunner.pp_handler_cls
        )
        self.cudagraph_manager_cls = (
            MetaXModelCudaGraphManager
            if self.k3_execution.enabled
            else GPUModelRunner.cudagraph_manager_cls
        )
        if not callable(getattr(GPUModelRunner, "_prepare_intermediate_tensors", None)):
            raise RuntimeError("K3 runner requires the explicit PP input-copy hook")
        super().__init__(vllm_config, device)
        self._runtime_diagnostics = diagnostics
        self._tp8_prefill_workspace = None

    def profile_run(self):
        # Worker memory_profiling encloses this call. Reserve before both vision
        # and text dummy execution so the persistent arena is included in KV sizing.
        if (
            self.vllm_config.scheduler_config.max_num_batched_tokens
            >= MIN_PREFILL_TOKENS
        ):
            methods = [
                module.quant_method
                for module in self.model.modules()
                if isinstance(getattr(module, "quant_method", None), K3Bf16MoEMethod)
                and module.quant_method.tp8_prepared is not None
            ]
            if methods:
                if self._tp8_prefill_workspace is None:
                    self._tp8_prefill_workspace = TP8PrefillWorkspace(self.device)
                self._tp8_prefill_workspace.reserve()
                for method in methods:
                    method.prefill_workspace = self._tp8_prefill_workspace
                if self._runtime_diagnostics.enabled:
                    self._runtime_diagnostics.record_workspace(
                        self._tp8_prefill_workspace.describe()
                    )
        return super().profile_run()

    def _prepare_intermediate_tensors(
        self, intermediate_tensors, num_tokens, dummy_run
    ):
        result = super()._prepare_intermediate_tensors(
            intermediate_tensors, num_tokens, dummy_run
        )
        if not dummy_run and isinstance(intermediate_tensors, AsyncIntermediateTensors):
            intermediate_tensors.release_after_copy(result.tensors.keys())
        return result
