# SPDX-License-Identifier: Apache-2.0
"""Construction-time selection for the validated MetaX Qwen runtime."""


def is_metax() -> bool:
    from vllm.platforms import current_platform

    return getattr(current_platform, "vendor_name", None) == "metax"
