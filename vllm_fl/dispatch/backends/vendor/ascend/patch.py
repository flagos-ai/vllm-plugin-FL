# Copyright (c) 2026 BAAI. All rights reserved.

import logging
from contextlib import contextmanager
from types import SimpleNamespace

logger = logging.getLogger(__name__)
_patch_state = SimpleNamespace(applied=False, applying=False)


def apply_ascend_patches():
    """Apply all Ascend-specific patches without caching an incomplete install."""
    if _patch_state.applied or _patch_state.applying:
        return
    _patch_state.applying = True
    try:
        with _mtp_patch_transaction():
            patch_rejection_sampling()
            patch_speculative_metadata()
            patch_causal_conv1d()
            patch_fla_ops()
            patch_op_cls()
            patch_fused_moe()
            patch_gdn()
            patch_vision_position_interpolation()
            patch_sampler_filter()
            from .patches.patch_mamba_utils import apply_mamba_copy_patch

            apply_mamba_copy_patch()
        _patch_state.applied = True
    finally:
        _patch_state.applying = False


@contextmanager
def _mtp_patch_transaction():
    """Validate dependent hooks together and restore their exact prior objects."""
    import inspect

    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn
    import vllm.model_executor.layers.mamba.ops.causal_conv1d as conv
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
    from vllm.v1.sample import rejection_sampler
    from vllm.v1.sample.ops import topk_topp_sampler
    from vllm.v1.spec_decode import llm_base_proposer, utils

    from vllm_fl.worker.spec_decode import prepare_next_token_padded_kernel

    from .impl.gdn import AscendQwenGatedDeltaNetAttention

    targets = [
        (rejection_sampler, name)
        for name in (
            "sample_recovered_tokens_kernel", "rejection_greedy_sample_kernel",
            "rejection_random_sample_kernel", "apply_top_k_top_p",
        )
    ] + [(topk_topp_sampler, "apply_top_k_top_p")]
    metadata_name = "eagle_prepare_next_token_padded_kernel"
    targets += [(module, metadata_name) for module in (utils, llm_base_proposer)]
    targets += [
        (module, name) for module in (conv, gdn)
        for name in ("causal_conv1d_fn", "causal_conv1d_update")
    ]
    previous = [(module, name, getattr(module, name)) for module, name in targets]
    for _, name, value in previous:
        if name in _REJECTION_KERNEL_ARGUMENTS or name == metadata_name:
            if not callable(getattr(value, "__getitem__", None)):
                raise TypeError(f"MTP kernel is not launchable: {name}")
        elif not callable(value):
            raise TypeError(f"MTP patch target is not callable: {name}")
    for module in (utils, llm_base_proposer):
        if tuple(getattr(getattr(module, metadata_name), "arg_names", ())) != tuple(
            prepare_next_token_padded_kernel.arg_names
        ):
            raise RuntimeError("Incompatible padded metadata parameter order")
    expected_conv = (
        "x", "conv_state", "weight", "bias", "activation", "conv_state_indices",
        "num_accepted_tokens", "query_start_loc", "max_query_len", "null_block_id",
        "block_idx_last_scheduled_token", "initial_state_idx", "validate_data",
    )
    for module in (conv, gdn):
        signature = inspect.signature(module.causal_conv1d_update)
        if (
            tuple(signature.parameters) != expected_conv
            or signature.parameters["null_block_id"].default != NULL_BLOCK_ID
        ):
            raise RuntimeError("Incompatible upstream convolution update contract")
    # Kernel launch adaptation uses named parameters, including positional calls.
    for name, required in _REJECTION_KERNEL_ARGUMENTS.items():
        kernel = getattr(rejection_sampler, name)
        if not required.issubset(getattr(kernel, "arg_names", ())):
            raise RuntimeError(f"Incompatible speculative sampling API: {name}")
    key = "QwenGatedDeltaNetAttention"
    missing = object()
    registered = op_registry_oot.get(key, missing)
    if registered is not missing and registered is not AscendQwenGatedDeltaNetAttention:
        raise RuntimeError("Conflicting GDN out-of-tree registration")
    own_name = AscendQwenGatedDeltaNetAttention.__dict__.get("name", missing)
    try:
        yield
    except BaseException:
        for module, name, value in previous:
            setattr(module, name, value)
        if registered is missing:
            op_registry_oot.pop(key, None)
        else:
            op_registry_oot[key] = registered
        if own_name is missing:
            if "name" in AscendQwenGatedDeltaNetAttention.__dict__:
                delattr(AscendQwenGatedDeltaNetAttention, "name")
        else:
            AscendQwenGatedDeltaNetAttention.name = own_name
        raise


_REJECTION_KERNEL_ARGUMENTS = {
    "sample_recovered_tokens_kernel": {"BLOCK_SIZE", "NO_DRAFT_PROBS", "USE_FP64_GUMBEL"},
    "rejection_greedy_sample_kernel": {"uniform_probs_ptr", "SYNTHETIC_MODE"},
    "rejection_random_sample_kernel": {"uniform_probs_ptr", "SYNTHETIC_MODE"},
}


class _NpuUniformProbabilityKernel:
    """Retain upstream RNG while adapting unsupported device FP64 arithmetic."""

    _fl_uniform_fp32 = True

    def __init__(self, kernel):
        self._kernel = kernel
        self._uniform_index = kernel.arg_names.index("uniform_probs_ptr")

    def __getattr__(self, name):
        return getattr(self._kernel, name)

    def __getitem__(self, grid):
        launch = self._kernel[grid]

        def adapted_launch(*args, **kwargs):
            import torch

            def convert(probabilities):
                if probabilities is not None and probabilities.dtype == torch.float64:
                    return probabilities.to(torch.float32)
                return probabilities

            if self._uniform_index < len(args):
                args = list(args)
                args[self._uniform_index] = convert(args[self._uniform_index])
            elif "uniform_probs_ptr" in kwargs:
                kwargs["uniform_probs_ptr"] = convert(kwargs["uniform_probs_ptr"])
            return launch(*args, **kwargs)

        return adapted_launch


def patch_rejection_sampling():
    """Reuse upstream sampling math with bounded NPU launch configurations."""
    from vllm.v1.sample import rejection_sampler
    from vllm_fl.patches.triton_kernel import patch_kernel_launch_meta

    previous = []
    for name, arguments in _REJECTION_KERNEL_ARGUMENTS.items():
        kernel = getattr(rejection_sampler, name)
        if not arguments.issubset(getattr(kernel, "arg_names", ())):
            raise RuntimeError(f"Incompatible speculative sampling API: {name}")
        previous.append((name, kernel))
    # Keep RNG generation unchanged, then cast its FP64 uniform values at the
    # device boundary. NPU sampling consumes FP32 probabilities. The recovery
    # tile also needs to fit UB; neither adaptation replaces a numerical kernel.
    try:
        patch_kernel_launch_meta(
            rejection_sampler, "sample_recovered_tokens_kernel", {"BLOCK_SIZE": 1024}
        )
        for name in ("rejection_greedy_sample_kernel", "rejection_random_sample_kernel"):
            kernel = getattr(rejection_sampler, name)
            if not getattr(kernel, "_fl_uniform_fp32", False):
                setattr(rejection_sampler, name, _NpuUniformProbabilityKernel(kernel))
    except Exception:
        for name, kernel in previous:
            setattr(rejection_sampler, name, kernel)
        raise


def patch_speculative_metadata():
    """Bind the framework token bookkeeping kernel at both upstream aliases."""
    from vllm.v1.spec_decode import llm_base_proposer, utils
    from vllm_fl.worker.spec_decode import prepare_next_token_padded_kernel

    name = "eagle_prepare_next_token_padded_kernel"
    modules = (utils, llm_base_proposer)
    previous = [(module, getattr(module, name)) for module in modules]
    required = set(prepare_next_token_padded_kernel.arg_names)
    if not all(required.issubset(getattr(kernel, "arg_names", ()))
               for _, kernel in previous):
        raise RuntimeError("Incompatible padded next-token metadata API")
    if all(kernel is prepare_next_token_padded_kernel for _, kernel in previous):
        return
    try:
        for module in modules:
            setattr(module, name, prepare_next_token_padded_kernel)
    except Exception:
        for module, kernel in previous:
            setattr(module, name, kernel)
        raise


def patch_sampler_filter():
    """Use the packaged A2/A3 filter while retaining vLLM's sampler contract."""
    import torch_npu
    from vllm.v1.sample import rejection_sampler
    from vllm.v1.sample.ops import topk_topp_sampler

    modules = (topk_topp_sampler, rejection_sampler)
    previous = [(module, module.apply_top_k_top_p) for module in modules]
    if not all(callable(original) for _, original in previous):
        raise TypeError("Sampling filters require callable upstream APIs")
    if all(getattr(original, "_fl_npu_filter", False) for _, original in previous):
        return

    def apply_top_k_top_p(logits, k, p):
        if k is None and p is None:
            return logits
        # vLLM's metadata uses int64; CANN's filter accepts int32 only.
        if k is not None:
            import torch

            k = k.to(torch.int32)
        return torch_npu.npu_top_k_top_p(logits, k=k, p=p)

    # vLLM chooses its CUDA-oriented Triton filter for batches >=8 even on
    # OOT platforms. Match rc1's built-in A2/A3 filtering path instead.
    apply_top_k_top_p._fl_npu_filter = True
    try:
        for module in modules:
            module.apply_top_k_top_p = apply_top_k_top_p
    except Exception:
        for module, original in previous:
            module.apply_top_k_top_p = original
        raise


def patch_vision_position_interpolation():
    """Use vLLM's tensor reference instead of the CUDA-oriented launch grid."""
    from vllm.model_executor.models import qwen3_vl

    # The Triton implementation launches one block per spatial patch. Vision
    # profiling can exceed Ascend's 65535-block limit; the upstream tensor
    # implementation has the same interpolation/spatial-merge contract.
    qwen3_vl.triton_pos_embed_interpolate = qwen3_vl.pos_embed_interpolate_native


def patch_gdn():
    """Keep the current upstream GDN model, replacing its numerical core."""
    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot

    from .impl.gdn import AscendQwenGatedDeltaNetAttention

    existing = op_registry_oot.get("QwenGatedDeltaNetAttention")
    if existing is AscendQwenGatedDeltaNetAttention:
        return
    if existing is not None:
        raise RuntimeError("Conflicting GDN out-of-tree registration")
    PluggableLayer.register_oot(
        _decorated_layer_cls=AscendQwenGatedDeltaNetAttention,
        name="QwenGatedDeltaNetAttention",
    )

def patch_causal_conv1d():
    """Patch causal_conv1d ops with Ascend implementations."""
    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv1d_lib
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as _gdn_lib

        from .impl.causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_npu
        from functools import wraps
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
        from vllm_fl.dispatch import CachedOp

        update = CachedOp("causal_conv1d_update")

        @wraps(causal_conv1d_fn_npu)
        def prefill_with_contiguous_channels(*args, **kwargs):
            output = causal_conv1d_fn_npu(*args, **kwargs)
            # Upstream GDN preparation addresses channels with unit stride.
            # Preserve [channels, tokens] while making its transpose contiguous.
            if output.ndim == 2 and output.stride(0) != 1:
                output = output.transpose(0, 1).contiguous().transpose(0, 1)
            return output

        @wraps(_conv1d_lib.causal_conv1d_update)
        def dispatched_update(
            x, conv_state, weight, bias=None, activation=None,
            conv_state_indices=None, num_accepted_tokens=None,
            query_start_loc=None, max_query_len=-1,
            null_block_id=NULL_BLOCK_ID,
            block_idx_last_scheduled_token=None, initial_state_idx=None,
            validate_data=False,
        ):
            return update(
                x, conv_state, weight, bias=bias, activation=activation,
                conv_state_indices=conv_state_indices,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=query_start_loc, max_query_len=max_query_len,
                pad_slot_id=null_block_id,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token,
                initial_state_idx=initial_state_idx, validate_data=validate_data,
            )

        targets = (
            (_conv1d_lib, "causal_conv1d_fn", prefill_with_contiguous_channels),
            (_conv1d_lib, "causal_conv1d_update", dispatched_update),
            (_gdn_lib, "causal_conv1d_fn", prefill_with_contiguous_channels),
            (_gdn_lib, "causal_conv1d_update", dispatched_update),
        )
        previous = [(module, name, getattr(module, name))
                    for module, name, _ in targets]
        if not all(callable(original) for _, _, original in previous):
            raise TypeError("Causal convolution patch requires callable upstream APIs")
        if all(getattr(getattr(module, "causal_conv1d_update"),
                       "_fl_causal_dispatch", False)
               for module in (_conv1d_lib, _gdn_lib)):
            return
        dispatched_update._fl_causal_dispatch = True
        try:
            for module, name, implementation in targets:
                setattr(module, name, implementation)
        except Exception:
            for module, name, original in previous:
                setattr(module, name, original)
            raise
        logger.info("Patched causal_conv1d ops for Ascend")
    except Exception:
        logger.exception("Failed to patch causal_conv1d ops")
        raise

def patch_fused_moe():
    """Patch fused MoE ops with Ascend implementations."""
    # TODO ops' triton implementation is not ready yet
    from .impl.fused_moe import fused_experts_impl
    try:
        import vllm_fl.ops.fused_moe.fused_moe as fused_moe_lib

        fused_moe_lib.fused_experts_impl = fused_experts_impl

        logger.info("Patched fused_moe for Ascend")
    except Exception as e:
        logger.warning("Failed to patch fused_moe ops: %s", e)

def patch_fla_ops():
    """Patch FLA ops and fused_gdn_gating with Ascend implementations."""
    try:
        import vllm.model_executor.layers.fla.ops as _fla_ops_lib
        import vllm.model_executor.layers.fla.ops.chunk as _fla_chunk_lib
        import vllm.model_executor.layers.fla.ops.fused_recurrent as _fla_recurrent_lib
        import vllm.model_executor.layers.fla.ops.layernorm_guard as _fla_layernorm_lib
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib
        from flag_gems.runtime.backend._ascend.fla import (
            chunk_gated_delta_rule_fwd,
            fused_recurrent_gated_delta_rule_fwd,
        )
        from flag_gems.runtime.backend._ascend.fla.layernorm_guard import (
            LayerNormFn as ascend_LayerNormFn,
        )

        from .impl.fla import chunk_gated_delta_rule_npu

        _fla_ops_lib.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd
        _fla_chunk_lib.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd
        _fla_chunk_lib.chunk_gated_delta_rule = chunk_gated_delta_rule_npu
        _fla_recurrent_lib.fused_recurrent_gated_delta_rule_fwd = fused_recurrent_gated_delta_rule_fwd
        _fla_layernorm_lib.LayerNormFn = ascend_LayerNormFn
        _qwen3_next_lib.chunk_gated_delta_rule = chunk_gated_delta_rule_npu
        logger.info("Patched FLA ops for Ascend")
    except Exception as e:
        logger.warning("Failed to patch FLA ops: %s", e)

def patch_op_cls():
    """Patch MMEncoderAttention to use manual matmul attention on NPU.

    The NPU npu_fused_infer_attention_score kernel only supports head_dim
    in {64, 128, 192}. The vision encoder may have non-standard head_dim
    (e.g. 72 for Qwen3.5). F.scaled_dot_product_attention on NPU may also
    dispatch to the same problematic kernel. Use pure-PyTorch matmul
    attention instead.
    """
    try:
        from vllm.model_executor.custom_op import CustomOp

        from .impl.mm_encoder_attention import AscendMMEncoderAttention
        from .impl.vocab_parallel_embedding import AscendVocabParallelEmbedding
        REGISTERED_ASCEND_OPS = {
            "VocabParallelEmbedding": AscendVocabParallelEmbedding,
            "MMEncoderAttention": AscendMMEncoderAttention,
        }
        for name, op_cls in REGISTERED_ASCEND_OPS.items():
            CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)
        logger.info("Patched MMEncoderAttention for NPU (matmul attention)")
    except Exception as e:
        logger.warning("Failed to patch MMEncoderAttention: %s", e)

def refresh_block_size(vllm_config, block_size = 128):
    """
    Refresh the block size in cache config.
    """
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config

    if not cache_config:
        return

    if cache_config.block_size is None:
        cache_config.block_size = block_size

    if not scheduler_config or not model_config:
        return

    # Hybrid state pages are aligned by the Mamba configuration verifier.
    if not model_config.is_hybrid and cache_config.block_size != block_size:
        if cache_config.enable_prefix_caching or scheduler_config.enable_chunked_prefill:
            logger.info(f"Block size is set to {block_size} if prefix cache or chunked prefill is enabled.")
            cache_config.block_size = block_size
