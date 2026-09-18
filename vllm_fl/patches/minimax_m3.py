# SPDX-License-Identifier: Apache-2.0
"""Lazy, MetaX-only registration for the vLLM 0.24 MiniMax-M3 model."""

_ARCHITECTURES = (
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
)


def register_metax_models() -> bool:
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "metax":
        return False
    from vllm.model_executor.models import ModelRegistry

    for architecture in _ARCHITECTURES:
        ModelRegistry.register_model(
            architecture, f"vllm_fl.models.minimax_m3:{architecture}"
        )
    return True


def is_m3_tp_model(runner) -> bool:
    """The graph/eager MCCL fence is needed only for the M3 TP runner."""
    config = runner.vllm_config
    architectures = getattr(config.model_config.hf_config, "architectures", ()) or ()
    return (
        any(name in _ARCHITECTURES for name in architectures)
        and config.parallel_config.tensor_parallel_size > 1
    )


def install_collective_boundary() -> None:
    import functools

    import torch

    from vllm_fl.worker.model_runner import ModelRunnerFL

    original = ModelRunnerFL._model_forward
    if getattr(original, "_fl_m3_collective_boundary", False):
        return

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        # Captured collectives must finish before compute_logits schedules
        # an eager all-gather on the same MCCL communicator. Never synchronize
        # while capturing the outer graph.
        if is_m3_tp_model(self) and not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize(self.device)
        return output

    forward._fl_m3_collective_boundary = True
    ModelRunnerFL._model_forward = forward
