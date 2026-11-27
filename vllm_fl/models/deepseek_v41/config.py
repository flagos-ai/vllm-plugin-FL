# SPDX-License-Identifier: Apache-2.0
"""Workspace sizes for the V4.1 sparse indexer."""

INDEXER_LIMIT = "deepseek_v41_indexer_max_context"


def indexer_max_context(config):
    """Logical admission limit. An unset limit uses the full model context."""
    return (config.additional_config or {}).get(INDEXER_LIMIT)


def indexer_workspace_tokens(config):
    model_limit = config.model_config.max_model_len
    limit = indexer_max_context(config)
    if limit is None:
        return model_limit
    if type(limit) is not int or not 32768 <= limit <= model_limit or limit % 64:
        raise ValueError(
            f"{INDEXER_LIMIT} must be a multiple of 64 in [32768, {model_limit}]"
        )
    if config.model_config.hf_config.model_type != "deepseek_v41":
        raise ValueError(f"{INDEXER_LIMIT} requires DeepSeek V4.1")
    block = config.cache_config.block_size or 64
    lookahead = getattr(config.speculative_config, "num_speculative_tokens", 0)
    # A speculative verify step can write past the last logical token.
    return min(model_limit, ((limit + lookahead + block) // block) * block)
