# Copyright (c) 2026 BAAI. All rights reserved.

"""FL runtime hooks around the native vLLM v0.32 NVIDIA worker.

Keep this module independent of the copied multi-vendor worker so NVIDIA
inherits upstream worker and model-runner changes without legacy imports.
"""

from collections.abc import Callable
from types import MethodType
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker as NativeGPUWorker

from vllm_fl.dispatch.io_common import managed_inference_mode
from vllm_fl.worker.runtime import _initialize_fl_runtime

logger = init_logger(__name__)


_NATIVE_RUNNER_IO_METHODS = ("execute_model", "sample_tokens", "pool")


def _install_native_runner_io_methods(model_runner: Any) -> None:
    """Replace native runner inference-mode wrappers on one runner instance.

    vLLM's NVIDIA V1 and V2 runners decorate their execution entry points
    with ``torch.inference_mode``. IO dumping needs ``torch.no_grad`` while its
    TorchDispatchMode is active. Unwrapping only that outer decorator keeps the
    native runner implementation and any inner vLLM decorators unchanged.
    """
    for method_name in _NATIVE_RUNNER_IO_METHODS:
        bound_method = getattr(model_runner, method_name, None)
        if bound_method is None or getattr(bound_method, "_vllm_fl_managed_io", False):
            continue

        method = getattr(bound_method, "__func__", bound_method)
        native_impl = getattr(method, "__wrapped__", None)
        if native_impl is None:
            raise RuntimeError(
                f"Native model runner method {method_name!r} no longer exposes "
                "the vLLM inference-mode wrapper"
            )

        managed_impl = managed_inference_mode()(native_impl)
        managed_impl._vllm_fl_managed_io = True
        setattr(model_runner, method_name, MethodType(managed_impl, model_runner))


def _call_native_worker_with_managed_inference(
    method: Callable, worker: "NvidiaWorkerFL", *args, **kwargs
) -> Any:
    """Call a native worker method without its fixed inference-mode wrapper."""
    native_impl = getattr(method, "__wrapped__", None)
    if native_impl is None:
        raise RuntimeError(
            f"Native worker method {method.__name__!r} no longer exposes "
            "the vLLM inference-mode wrapper"
        )
    return managed_inference_mode()(native_impl)(worker, *args, **kwargs)


class NvidiaWorkerFL(NativeGPUWorker):
    """Thin FL wrapper around vLLM's target-version NVIDIA worker.

    NVIDIA inherits the upstream v0.32 GPU worker and V1/V2 model runners;
    the plugin only installs its dispatch hooks before model construction.
    Keeping this as a delegating wrapper prevents the plugin worker from
    drifting from upstream CUDA behavior in future vLLM releases.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fl_io_dump_initialized = False
        _initialize_fl_runtime(self.rank)

    def init_device(self):
        super().init_device()

        from vllm_fl.patches.qsa_cache_hint import patch_qsa_cache_hint

        if patch_qsa_cache_hint(self.model_config, self.device):
            logger.info("Applied the FlagTree 3.6 Hopper QSA K-cache hint workaround")

    def _ensure_io_dump_initialized(self) -> None:
        """Enable IO dumping on the first execution after model loading."""
        if self._fl_io_dump_initialized:
            return

        from vllm_fl.dispatch.io_dumper import (
            init_io_dump_from_env,
            is_dump_enabled,
            register_io_module_hooks,
        )

        init_io_dump_from_env(getattr(self.model_config, "enforce_eager", False))
        if is_dump_enabled():
            _install_native_runner_io_methods(self.model_runner)
            register_io_module_hooks(self.model_runner.get_model())
            logger.info("Enabled FL IO dumping for the native NVIDIA model runner")
        self._fl_io_dump_initialized = True

    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        self._ensure_io_dump_initialized()
        output = _call_native_worker_with_managed_inference(
            NativeGPUWorker.sample_tokens, self, grammar_output
        )
        if output is not None:
            from vllm_fl.dispatch.io_dumper import advance_io_step

            advance_io_step()
        return output

    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        self._ensure_io_dump_initialized()
        output = _call_native_worker_with_managed_inference(
            NativeGPUWorker.execute_model, self, scheduler_output
        )
        if output is not None:
            from vllm_fl.dispatch.io_dumper import advance_io_step

            advance_io_step()
        return output
