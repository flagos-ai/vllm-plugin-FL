# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4 runtime provider for the pinned native runner extension contract."""

from __future__ import annotations


def create_hy4_runtime(config):
    """Decline unsupported configurations before creating a graph or buffer."""
    from vllm.config import CUDAGraphMode
    from vllm.platforms import current_platform

    from ..indexer_policy import use_int8_indexer

    if not current_platform.is_cuda():
        return None
    if current_platform.get_device_capability() != (9, 0):
        return None
    parallel = config.parallel_config
    spec = config.speculative_config
    compilation = config.compilation_config
    if (
        spec is None
        or spec.method != "mtp"
        or spec.num_speculative_tokens != 3
        or spec.enforce_eager
        or spec.parallel_drafting
        or spec.disable_padded_drafter_batch
        or parallel.tensor_parallel_size != 16
        or not parallel.enable_expert_parallel
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.use_ubatching
        or config.lora_config is not None
        or compilation.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
    ):
        return None
    if not use_int8_indexer():
        return None
    from .prefill_graph import supports_mixed_config

    if not supports_mixed_config(config):
        return None
    return Hy4RuntimeHooks()


class Hy4RuntimeHooks:
    """Framework resource factories; numerical providers retain their policy."""

    mixed_graphs = True

    @staticmethod
    def make_common_metadata_graph():
        from .runtime_metadata import CommonAttentionMetadataGraphRunner

        return CommonAttentionMetadataGraphRunner()

    @staticmethod
    def make_input_staging(runner):
        from .runtime_staging import make_async_input_staging

        return make_async_input_staging(runner)

    @staticmethod
    def make_draft_graph(proposer, runner):
        from .mtp_draft_graph import make_hyv4_draft_graph

        return make_hyv4_draft_graph(proposer, runner)

    @staticmethod
    def select_decode_deltas(*args, **kwargs):
        from .runtime_metadata import select_decode_deltas

        return select_decode_deltas(*args, **kwargs)

    @staticmethod
    def compute_step_common_attention_metadata(*args, **kwargs):
        from .runtime_metadata import compute_step_common_attention_metadata

        return compute_step_common_attention_metadata(*args, **kwargs)

    @staticmethod
    def decode_lens_from_common(*args, **kwargs):
        from .runtime_metadata import decode_lens_from_common

        return decode_lens_from_common(*args, **kwargs)

    @staticmethod
    def wrap_breakable(*args, **kwargs):
        from .prefill_graph import wrap_native_breakable

        return wrap_native_breakable(*args, **kwargs)

    @staticmethod
    def warm_full_decode_metadata(*args, **kwargs):
        from .decode_metadata import warm_full_decode_metadata_before_capture

        return warm_full_decode_metadata_before_capture(*args, **kwargs)


def register_runtime_hooks():
    """Register optional native hooks while preserving stock native imports.

    Only an absent extension module is optional. A broken extension or any
    missing transitive dependency is an installation error and propagates.
    """
    try:
        from vllm.v1.worker.model_runtime import register_model_runtime
    except ModuleNotFoundError as error:
        if error.name != "vllm.v1.worker.model_runtime":
            raise
        return False
    register_model_runtime(
        ("HYV4ForCausalLM", "HYV4VLForConditionalGeneration"), create_hy4_runtime
    )
    return True
