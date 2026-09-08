# Copyright 2026 FlagOS Contributors

"""GLM5.2 IndexShare compatibility patch for vLLM 0.20.0"""

from __future__ import annotations

import inspect
import re
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any
from contextvars import ContextVar

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_GLM_MODEL_TYPE = "glm_moe_dsa"

# Construction-time bridge for vLLM 0.20.x.
#
# PR #45895 passes topk_indices_buffer explicitly through the MLA stack.
# vLLM 0.20.x does not yet pass that argument from
# MultiHeadLatentAttentionWrapper to MLAAttention.
#
# Keep the buffer only during construction of one GLM MLA layer.
_GLM_SHARED_TOPK_BUFFER: ContextVar[
    torch.Tensor | None
] = ContextVar(
    "fl_glm_shared_topk_buffer",
    default=None,
)


_LAYER_ID_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

# DeepseekV2MLAAttention.__init__ accesses Indexer through the module-global
# symbol in deepseek_v2.py. A reentrant lock makes the temporary replacement
# deterministic if model initialization is ever entered concurrently.
_INDEXER_OVERRIDE_LOCK = threading.RLock()


def _normalize_schedule_value(value: Any) -> str:
    """Normalize an IndexShare schedule marker."""

    return str(value).strip().lower()


def _is_shared_schedule_value(value: Any) -> bool:
    """Return whether a schedule marker represents a shared layer."""

    return _normalize_schedule_value(value) in {
        "s",
        "share",
        "shared",
        "skip",
    }


def _get_indexer_pattern(config: Any) -> Any:
    """Get an explicit per-layer IndexShare pattern when available.

    ``index_topk_pattern`` is the upstream field. ``indexer_types`` is
    accepted as a compatibility alias for some GLM checkpoint configs.
    """

    pattern = getattr(config, "index_topk_pattern", None)

    if pattern is None:
        pattern = getattr(config, "indexer_types", None)

    return pattern


def _is_glm_index_share_config(config: Any) -> bool:
    """Return whether the configuration requires GLM IndexShare handling."""

    if config is None:
        return False

    if getattr(config, "model_type", None) != _GLM_MODEL_TYPE:
        return False

    if not hasattr(config, "index_topk"):
        return False

    pattern = _get_indexer_pattern(config)

    if pattern is not None:
        try:
            return any(_is_shared_schedule_value(value) for value in pattern)
        except TypeError:
            logger.warning(
                "Ignoring invalid GLM IndexShare pattern of type %s",
                type(pattern).__name__,
            )
            return False

    try:
        frequency = int(getattr(config, "index_topk_freq", 1))
    except (TypeError, ValueError):
        return False

    # frequency == 1 means every layer owns a full Indexer and therefore
    # there is no IndexShare behavior to patch.
    return frequency > 1


def _extract_layer_id(prefix: str) -> int | None:
    """Extract the decoder layer index from an Indexer module prefix."""

    match = _LAYER_ID_PATTERN.search(prefix)

    if match is None:
        return None

    return int(match.group(1))


def _should_build_full_indexer(config: Any, layer_id: int) -> bool:
    """Return whether a layer must construct a physical vLLM Indexer.

    Backbone layers follow either:

    1. an explicit ``index_topk_pattern`` / ``indexer_types`` list; or
    2. ``index_topk_freq`` and ``index_skip_topk_offset``.

    MTP/next-n layers are outside the backbone schedule and always retain
    a full Indexer.
    """

    num_hidden_layers = getattr(config, "num_hidden_layers", None)

    # MTP layers are normally appended after the backbone layers. They must
    # keep their own Indexer instead of sharing a backbone layer's result.
    if (
        num_hidden_layers is not None
        and layer_id >= int(num_hidden_layers)
    ):
        return True

    pattern = _get_indexer_pattern(config)

    if pattern is not None:
        if 0 <= layer_id < len(pattern):
            return not _is_shared_schedule_value(pattern[layer_id])

        # If an explicit pattern does not cover this layer, preserve the
        # original vLLM behavior instead of incorrectly skipping an Indexer.
        return True

    try:
        frequency = int(getattr(config, "index_topk_freq", 1))
        offset = int(getattr(config, "index_skip_topk_offset", 2))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "GLM IndexShare fields index_topk_freq and "
            "index_skip_topk_offset must be integers."
        ) from error

    if frequency <= 0:
        raise ValueError(
            "GLM index_topk_freq must be greater than zero, "
            f"but got {frequency}."
        )

    # This is the same schedule formula used by the upstream IndexCache
    # implementation. A False result means this layer reuses indices
    # generated by the preceding full Indexer layer.
    skip_topk = (
        max(layer_id - offset + 1, 0) % frequency != 0
    )

    return not skip_topk


def _call_original_indexer(
    original_indexer: type[nn.Module],
    *,
    vllm_config: Any,
    config: Any,
    hidden_size: int,
    q_lora_rank: int,
    quant_config: Any,
    cache_config: Any,
    topk_indices_buffer: torch.Tensor | None,
    prefix: str,
    extra_args: tuple[Any, ...],
    extra_kwargs: dict[str, Any],
) -> nn.Module:
    """Construct the original vLLM Indexer without changing its behavior."""

    return original_indexer(
        vllm_config,
        config,
        hidden_size,
        q_lora_rank,
        quant_config,
        cache_config,
        topk_indices_buffer,
        prefix,
        *extra_args,
        **extra_kwargs,
    )


def _get_glm_sparse_attn_indexer_cls():
    """Select the sparse Indexer implementation used by physical GLM layers.

    Only GLM + Hygon uses the dedicated implementation. Other platforms
    preserve the existing SparseAttnIndexerFL behavior.
    """

    from vllm.platforms import current_platform

    vendor_name = getattr(
        current_platform,
        "vendor_name",
        None,
    )

    if vendor_name == "hygon":
        # Importing this module registers:
        #
        # torch.ops.vllm.glm_hygon_sparse_attn_indexer_fl
        from vllm_fl.ops.glm_hygon_sparse_attn_indexer import (
            GlmHygonSparseAttnIndexer,
        )

        return GlmHygonSparseAttnIndexer

    # Preserve the existing behavior for non-Hygon GLM platforms.
    from vllm_fl.ops.sparse_attn_indexer import SparseAttnIndexerFL

    return SparseAttnIndexerFL


def _make_glm_indexer_factory(
    original_indexer: type[nn.Module],
):
    """Create the GLM-5.2 IndexShare Indexer factory.    """

    def glm_indexer_factory(
        vllm_config,
        config,
        hidden_size,
        q_lora_rank,
        quant_config,
        cache_config,
        topk_indices_buffer,
        prefix="",
        *args,
        **kwargs,
    ):
        # Preserve original behavior outside GLM IndexShare.
        if not _is_glm_index_share_config(config):
            return _call_original_indexer(
                original_indexer,
                vllm_config=vllm_config,
                config=config,
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                prefix=prefix,
                extra_args=args,
                extra_kwargs=kwargs,
            )

        layer_id = _extract_layer_id(prefix)

        if layer_id is None:
            logger.warning(
                "Cannot extract GLM decoder layer id from Indexer "
                "prefix %r; constructing the original Indexer.",
                prefix,
            )

            return _call_original_indexer(
                original_indexer,
                vllm_config=vllm_config,
                config=config,
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                prefix=prefix,
                extra_args=args,
                extra_kwargs=kwargs,
            )

        if _should_build_full_indexer(
            config,
            layer_id,
        ):
            logger.debug(
                "GLM layer %d constructs a physical Indexer.",
                layer_id,
            )

            return _call_original_indexer(
                original_indexer,
                vllm_config=vllm_config,
                config=config,
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                prefix=prefix,
                extra_args=args,
                extra_kwargs=kwargs,
            )

        # PR #45895 semantics:
        #
        # shared backbone layers do NOT construct an Indexer.
        #
        # Save only the shared buffer reference temporarily so the
        # vLLM 0.20.x MLA construction bridge can pass it to the
        # sparse MLA implementation.
        if topk_indices_buffer is None:
            raise RuntimeError(
                "GLM IndexShare layer requires "
                "topk_indices_buffer."
            )

        _GLM_SHARED_TOPK_BUFFER.set(
            topk_indices_buffer
        )

        logger.debug(
            "GLM layer %d skips physical Indexer construction "
            "and reuses topk_indices_buffer.",
            layer_id,
        )

        return None

    return glm_indexer_factory


def _make_mla_attention_factory(
    original_mla_attention,
):
    """Backport PR #45895 top-k buffer propagation.

    vLLM 0.20.x MLAAttention already accepts arbitrary MLA
    implementation kwargs via **extra_impl_args, but
    MultiHeadLatentAttentionWrapper does not yet pass the shared
    topk_indices_buffer.

    Inject it only while constructing a GLM IndexShare shared layer.
    """

    def mla_attention_factory(
        *args,
        **kwargs,
    ):
        use_sparse = bool(
            kwargs.get(
                "use_sparse",
                False,
            )
        )

        indexer = kwargs.get(
            "indexer",
            None,
        )

        if use_sparse and indexer is None:
            topk_indices_buffer = (
                _GLM_SHARED_TOPK_BUFFER.get()
            )

            if topk_indices_buffer is None:
                raise RuntimeError(
                    "Sparse GLM MLA has indexer=None but no "
                    "shared topk_indices_buffer is available."
                )

            kwargs["topk_indices_buffer"] = (
                topk_indices_buffer
            )

        return original_mla_attention(
            *args,
            **kwargs,
        )

    return mla_attention_factory


@contextmanager
def _temporary_indexer_override(
    deepseek_v2_module,
):
    """Install GLM-5.2 construction-time overrides.

    1. Indexer
       Skip physical Indexer construction on IndexShare layers.

    2. SparseAttnIndexer
       Physical Indexers use the platform-specific FL implementation.

    3. MLAAttention
       Backport PR #45895 shared top-k buffer propagation without
       modifying vLLM source.
    """

    from vllm.model_executor.layers import (
        mla as mla_module,
    )

    sparse_attn_indexer_cls = (
        _get_glm_sparse_attn_indexer_cls()
    )

    with _INDEXER_OVERRIDE_LOCK:
        original_indexer = (
            deepseek_v2_module.Indexer
        )

        original_sparse_attn_indexer = (
            deepseek_v2_module.SparseAttnIndexer
        )

        original_mla_attention = (
            mla_module.MLAAttention
        )

        indexer_factory = (
            _make_glm_indexer_factory(
                original_indexer
            )
        )

        mla_attention_factory = (
            _make_mla_attention_factory(
                original_mla_attention
            )
        )

        deepseek_v2_module.Indexer = indexer_factory
        deepseek_v2_module.SparseAttnIndexer = sparse_attn_indexer_cls
        mla_module.MLAAttention = mla_attention_factory

        try:
            yield
        finally:
            mla_module.MLAAttention = original_mla_attention
            deepseek_v2_module.SparseAttnIndexer = original_sparse_attn_indexer
            deepseek_v2_module.Indexer = original_indexer


def _get_mla_config(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Read ``config`` from DeepseekV2MLAAttention.__init__ arguments."""

    if "config" in kwargs:
        return kwargs["config"]

    # The patched method receives arguments after ``self``.
    # Original positional order:
    #   vllm_config, config, hidden_size, ...
    if len(args) >= 2:
        return args[1]

    return None


def _validate_vllm_api() -> None:
    """Fail early when the installed vLLM API is incompatible.

    This patch targets the vLLM 0.20.x DeepSeek-v2 implementation. Explicit
    signature checks prevent a future vLLM upgrade from silently applying an
    invalid monkey patch.
    """

    from vllm.model_executor.models import deepseek_v2

    mla_parameters = set(
        inspect.signature(
            deepseek_v2.DeepseekV2MLAAttention.__init__
        ).parameters
    )

    required_mla_parameters = {
        "vllm_config",
        "config",
        "topk_indices_buffer",
        "prefix",
    }

    missing_mla_parameters = (
        required_mla_parameters - mla_parameters
    )

    if missing_mla_parameters:
        raise RuntimeError(
            "Unsupported vLLM DeepseekV2MLAAttention API. "
            "Missing constructor parameters: "
            f"{sorted(missing_mla_parameters)}"
        )

    indexer_parameters = set(
        inspect.signature(
            deepseek_v2.Indexer.__init__
        ).parameters
    )

    required_indexer_parameters = {
        "vllm_config",
        "config",
        "hidden_size",
        "q_lora_rank",
        "quant_config",
        "cache_config",
        "topk_indices_buffer",
        "prefix",
    }

    missing_indexer_parameters = (
        required_indexer_parameters - indexer_parameters
    )

    if missing_indexer_parameters:
        raise RuntimeError(
            "Unsupported vLLM Indexer API. "
            "Missing constructor parameters: "
            f"{sorted(missing_indexer_parameters)}"
        )

    if not hasattr(
        deepseek_v2,
        "GlmMoeDsaForCausalLM",
    ):
        raise RuntimeError(
            "The installed vLLM does not provide "
            "GlmMoeDsaForCausalLM."
        )

    if not hasattr(
        deepseek_v2,
        "SparseAttnIndexer",
    ):
        raise RuntimeError(
            "The installed vLLM deepseek_v2 module does not expose "
            "SparseAttnIndexer."
        )

    sparse_attn_indexer_cls = (
        _get_glm_sparse_attn_indexer_cls()
    )

    upstream_sparse_indexer = (
        deepseek_v2.SparseAttnIndexer
    )

    if not issubclass(
        sparse_attn_indexer_cls,
        upstream_sparse_indexer,
    ):
        raise RuntimeError(
            f"{sparse_attn_indexer_cls.__name__} "
            "is incompatible with the installed "
            "vLLM SparseAttnIndexer implementation."
        )


def _add_hygon_indexer_splitting_op(vllm_config: Any) -> None:
    """Keep the stateful Indexer write outside adjacent compiled regions."""

    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "hygon":
        return

    compilation_config = vllm_config.compilation_config
    splitting_op = "vllm::glm_hygon_sparse_attn_indexer_fl"
    splitting_ops = compilation_config.splitting_ops

    if splitting_ops is None:
        splitting_ops = list(compilation_config._attention_ops)
        compilation_config.splitting_ops = splitting_ops

    if splitting_op not in splitting_ops:
        splitting_ops.append(splitting_op)
        logger.info(
            "Added %s to GLM compilation splitting_ops.",
            splitting_op,
        )


def _patch_glm_indexer_construction() -> None:
    """Patch GLM MLA initialization to honor the IndexShare schedule."""

    from vllm.model_executor.models import deepseek_v2

    attention_class = deepseek_v2.DeepseekV2MLAAttention

    if getattr(
        attention_class,
        "_fl_glm_index_share_patched",
        False,
    ):
        return

    original_init = attention_class.__init__

    @wraps(original_init)
    def patched_init(
        self,
        *args,
        **kwargs,
    ):
        config = _get_mla_config(
            args,
            kwargs,
        )

        # Preserve the exact original behavior for:
        #
        # - non-GLM models
        # - GLM without IndexShare
        if not _is_glm_index_share_config(config):
            return original_init(
                self,
                *args,
                **kwargs,
            )

        vllm_config = kwargs.get(
            "vllm_config",
            args[0] if args else None,
        )
        if vllm_config is None:
            raise RuntimeError(
                "GLM IndexShare patch cannot locate vllm_config."
            )

        _add_hygon_indexer_splitting_op(vllm_config)

        # This ContextVar is only a construction-time bridge for
        # vLLM 0.20.x.
        token = _GLM_SHARED_TOPK_BUFFER.set(
            None
        )

        try:
            with _temporary_indexer_override(
                deepseek_v2
            ):
                return original_init(
                    self,
                    *args,
                    **kwargs,
                )
        finally:
            _GLM_SHARED_TOPK_BUFFER.reset(
                token
            )

    attention_class.__init__ = patched_init
    attention_class._fl_glm_index_share_patched = True
    attention_class._fl_original_init = original_init

    logger.info(
        "Applied GLM IndexShare construction patch."
    )


def apply_glm_index_share_patches() -> None:
    """Backport GLM-5.2 IndexShare semantics from vLLM PR #45895."""

    _validate_vllm_api()
    _patch_glm_indexer_construction()
