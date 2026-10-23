# SPDX-License-Identifier: Apache-2.0
"""Framework adaptation policy, configured before dispatch is bound."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class FrameworkPolicy:
    decode_hq16: bool = True
    real_query_heads: int = 8
    prefill_headpack: bool = True
    indexer_graphsafe: bool = True
    indexer_paged_logits: bool = True
    indexer_row_sharding: bool = True
    indexer_query_shards: int = 4
    indexer_min_keys: int = 4096
    single_stream: bool = True


@dataclass(frozen=True)
class MoePolicy:
    use_mctlass: bool = True
    python_api: bool = True
    chunk_size: int = 16384
    hidden_size: int = 4096
    tuned_config_folder: str = ""


current = FrameworkPolicy()
moe = MoePolicy()
_configured = False


def is_enabled() -> bool:
    return _configured


def configure(framework: dict, moe_values: dict) -> None:
    global current, moe, _configured
    for values, cls in ((framework, FrameworkPolicy), (moe_values, MoePolicy)):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
        for field in fields(cls):
            if field.name in values and type(values[field.name]) is not field.type:
                raise TypeError(f"Invalid type for {cls.__name__}.{field.name}")
    selected, selected_moe = FrameworkPolicy(**framework), MoePolicy(**moe_values)
    if selected.real_query_heads != 8 or selected.indexer_query_shards != 4:
        raise ValueError("Only the validated TP8 / M4 profile is supported")
    if not selected.single_stream:
        raise ValueError("Multi-stream execution has not been validated")
    if _configured and (selected != current or selected_moe != moe):
        raise RuntimeError("Framework policy cannot change after initialization")
    current, moe = selected, selected_moe
    _configured = True
