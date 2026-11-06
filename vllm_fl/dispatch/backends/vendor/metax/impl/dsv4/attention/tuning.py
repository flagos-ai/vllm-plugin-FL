# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FL-owned TP8 attention launch policy. Configure once before tracing/inference.

This module has no vLLM/FL/process-environment dependency.
Input semantics (e.g. the number of real heads) remain explicit API arguments.
"""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class KernelPolicy:
    decode_splits: int = 4
    decode_block_k: int = 16
    decode_block_h: int = 16
    decode_warps: int = 4
    decode_stages: int = 1
    decode_max_batch: int = 64


current = KernelPolicy()
_configured = False


def configure(values: dict) -> None:
    global current, _configured
    unknown = set(values) - {f.name for f in fields(KernelPolicy)}
    if unknown:
        raise ValueError(f"Unknown kernel tuning fields: {sorted(unknown)}")
    for field in fields(KernelPolicy):
        if field.name in values and type(values[field.name]) is not field.type:
            raise TypeError(f"Invalid type for kernel field {field.name}")
    selected = KernelPolicy(**values)
    if selected.decode_block_h != 16 or selected.decode_splits != 4:
        raise ValueError("This validated profile requires decode BH16/S4")
    if _configured and selected != current:
        raise RuntimeError("Kernel policy cannot change after initialization")
    current = selected
    _configured = True
