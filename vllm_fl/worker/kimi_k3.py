"""K3 lifecycle integration; arithmetic remains in the operator modules."""

from contextlib import ExitStack

import torch
import triton

from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_fl.distributed.kimi_k3_pp import HostPackedPPHandler
from vllm_fl.models.kimi_k3_context import execution_config
from vllm_fl.ops.kimi_k3.workspace_contract import (
    ACTIVATION_COLUMNS,
    MAX_PREFILL_TOKENS,
    MIN_PREFILL_TOKENS,
    ROUTE_COLUMNS,
    TOP_K,
)
from vllm_fl.quantization.mxfp4.kimi_k3 import K3Bf16MoEMethod
from vllm_fl.worker.pipeline_input import AsyncIntermediateTensors


class MetaXModelCudaGraphManager(ModelCudaGraphManager):
    def run_fullgraph(self, desc):
        result = super().run_fullgraph(desc)
        # MCCL requires completion before following eager collectives.
        # Identical stream boundary to the qualified runtime workaround.
        torch.cuda.current_stream().synchronize()
        return result


class TP8PrefillWorkspace:
    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.index is None:
            raise ValueError("Workspace requires an explicit device index")
        self._buffers = {}
        self._closed = False

    def _stream_key(self):
        return int(triton.runtime.driver.active.get_current_stream(self.device.index))

    def reserve(self):
        """Reserve the current stream before KV sizing; never during capture."""
        if self._closed:
            raise RuntimeError("The prefill workspace has been closed")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Reserve the prefill workspace before Graph capture")
        stream = self._stream_key()
        if stream not in self._buffers:
            scratch = torch.empty(
                MAX_PREFILL_TOKENS * TOP_K * ROUTE_COLUMNS,
                dtype=torch.bfloat16,
                device=self.device,
            )
            act = torch.empty(
                (MAX_PREFILL_TOKENS * TOP_K, ACTIVATION_COLUMNS),
                dtype=torch.bfloat16,
                device=self.device,
            )
            self._buffers[stream] = (scratch, act)

    def get(self, device):
        """Borrow an already reserved stream's buffers; never allocate here."""
        if self._closed:
            raise RuntimeError("The prefill workspace has been closed")
        if torch.device(device) != self.device:
            raise ValueError("Workspace belongs to another device")
        stream = self._stream_key()
        if stream not in self._buffers:
            raise RuntimeError(
                "Reserve this prefill stream before setting the KV budget"
            )
        return self._buffers[stream]

    def close(self):
        """Release registered arenas only after all device work has completed."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._buffers:
                torch.cuda.synchronize(self.device)
        finally:
            self._buffers.clear()

    def describe(self):
        """Return CPU metadata for optional startup diagnostics."""
        return [
            {
                "device": self.device.index,
                "stream": stream,
                "bytes": scratch.numel() * scratch.element_size()
                + act.numel() * act.element_size(),
                "arenas": len(self._buffers),
                "scratch_ptr": scratch.data_ptr(),
                "act_ptr": act.data_ptr(),
            }
            for stream, (scratch, act) in sorted(self._buffers.items())
        ]


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
        if getattr(vllm_config, "weight_transfer_config", None) is not None:
            raise NotImplementedError(
                "K3 MXFP4 weight updates are unsupported; restart the worker "
                "to load a new checkpoint"
            )
        self._runtime_diagnostics = diagnostics
        self._tp8_prefill_workspace = None
        try:
            super().__init__(vllm_config, device)
        except BaseException:
            self._release_k3_resources()
            raise

    def reload_weights(self, *args, **kwargs):
        """Reject reload before vLLM restores raw checkpoint parameter layouts."""
        raise NotImplementedError(
            "K3 MXFP4 checkpoint reload is unsupported; restart the worker "
            "to load a new checkpoint"
        )

    def _release_k3_resources(self):
        workspace = getattr(self, "_tp8_prefill_workspace", None)
        self._tp8_prefill_workspace = None
        handler = getattr(self, "pp_handler", None)
        with ExitStack() as resources:
            if workspace is not None:
                resources.callback(workspace.close)
            if isinstance(handler, HostPackedPPHandler):
                self.pp_handler = None
                resources.callback(handler.close)

    def shutdown(self):
        try:
            self._release_k3_resources()
        finally:
            super().shutdown()

    def profile_run(self):
        try:
            # Worker memory_profiling encloses this call. Reserve before both vision
            # and text dummy execution so the persistent arena is included in KV sizing.
            if (
                self.vllm_config.scheduler_config.max_num_batched_tokens
                >= MIN_PREFILL_TOKENS
            ):
                methods = [
                    module.quant_method
                    for module in self.model.modules()
                    if isinstance(
                        getattr(module, "quant_method", None), K3Bf16MoEMethod
                    )
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
        except BaseException:
            self._release_k3_resources()
            raise

    def _prepare_intermediate_tensors(
        self, intermediate_tensors, num_tokens, dummy_run
    ):
        result = super()._prepare_intermediate_tensors(
            intermediate_tensors, num_tokens, dummy_run
        )
        if not dummy_run and isinstance(intermediate_tensors, AsyncIntermediateTensors):
            intermediate_tensors.release_after_copy(result.tensors.keys())
        return result
