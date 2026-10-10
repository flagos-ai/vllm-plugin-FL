# SPDX-License-Identifier: Apache-2.0
"""GLM index metadata compatibility exports."""
from .kpool_compress import (
    append_tail_to_topk,
    expand_pools_and_append_tail,
    expand_pools_to_tokens,
)

__all__ = [
    "append_tail_to_topk",
    "expand_pools_and_append_tail",
    "expand_pools_to_tokens",
]
