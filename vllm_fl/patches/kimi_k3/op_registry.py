# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema-only declarations for empty vLLM builds; no model-kernel fallback.

The FL native extension owns the real weak-reference runtime wrapper. Missing
model kernels remain unsupported. A schema exists for compilation matching,
not as a claim that a CPU or accelerator implementation has been installed.
"""

import logging
import threading

import torch

from .op_schemas import SCHEMAS

logger = logging.getLogger(__name__)
_LIBRARY: torch.library.Library | None = None
_REGISTRATION_LOCK = threading.Lock()
_QUERY_OPS = (
    "cutlass_scaled_mm_supports_fp8",
    "cutlass_scaled_mm_supports_block_fp8",
    "cutlass_group_gemm_supported",
    "cutlass_scaled_mm_supports_fp4",
)


def _unavailable_cutlass(cuda_device_capability: int) -> bool:
    """Device capability alone cannot make an absent CUTLASS kernel available."""
    return False


def register_op_schemas() -> None:
    """Define missing schemas exactly once and preserve real implementations."""
    global _LIBRARY
    with _REGISTRATION_LOCK:
        if _LIBRARY is not None:
            return
        if not SCHEMAS:
            raise RuntimeError("The pinned vLLM _C schema list is empty")

        parsed = [torch._C.parse_schema("_C::" + schema) for schema in SCHEMAS]
        full_names = [
            schema.name + ("." + schema.overload_name if schema.overload_name else "")
            for schema in parsed
        ]
        if len(full_names) != len(set(full_names)):
            raise RuntimeError("Duplicate names in the pinned vLLM _C schema list")
        existing_ops = set(torch._C._dispatch_get_all_op_names())

        # Check every existing declaration before adding any new definition.
        for expected, full_name in zip(parsed, full_names):
            if full_name in existing_ops:
                actual = torch._C._dispatch_find_schema_or_throw(
                    expected.name, expected.overload_name
                ).schema()
                if str(actual) != str(expected):
                    raise RuntimeError(
                        f"Schema mismatch for {full_name}: "
                        f"loaded={actual}; pinned={expected}"
                    )

        lib = torch.library.Library("_C", "FRAGMENT")
        registered = 0
        for schema, full_name in zip(SCHEMAS, full_names):
            if full_name not in existing_ops:
                lib.define(schema)
                registered += 1

        for name in _QUERY_OPS:
            op_name = "_C::" + name
            has_native_query = any(
                torch._C._dispatch_has_kernel_for_dispatch_key(op_name, key)
                for key in ("CompositeExplicitAutograd", "CompositeImplicitAutograd",
                            "CPU", "CUDA")
            )
            if not has_native_query:
                lib.impl(name, _unavailable_cutlass, "CompositeExplicitAutograd")

        _LIBRARY = lib
        logger.info("Registered %d schema-only _C declarations", registered)
