# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Select vLLM's native scaled-mm kernel for dynamic-token W8A8."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from vllm.model_executor.kernels.linear import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)

logger = logging.getLogger(__name__)


def _create_vllm_w8a8_linear_kernel(
    config: Int8ScaledMMLinearLayerConfig,
    layer_param_names: Sequence[str],
    module_name: str,
) -> Int8ScaledMMLinearKernel:
    """Select a native vLLM INT8 scaled-mm kernel."""
    import vllm.model_executor.kernels.linear as linear_module
    from vllm.platforms import current_platform

    candidates = linear_module._POSSIBLE_INT8_KERNELS.get(
        current_platform._enum,
        [],
    )
    if not candidates:
        raise RuntimeError(
            "vLLM has no native INT8 scaled-mm candidates for the current platform"
        )

    try:
        kernel_type = linear_module.choose_scaled_mm_linear_kernel(
            config,
            {current_platform._enum: candidates},
        )
    except ValueError as exc:
        raise RuntimeError(
            "no native vLLM INT8 scaled-mm kernel supports this platform"
        ) from exc

    logger.info(
        "Selected native vLLM W8A8 linear kernel %s for %s",
        kernel_type.__name__,
        module_name,
    )
    return kernel_type(config, layer_param_names)


def create_w8a8_linear_kernel(
    config: Int8ScaledMMLinearLayerConfig,
    layer_param_names: Sequence[str],
    module_name: str,
) -> Int8ScaledMMLinearKernel:
    """Select vLLM's native W8A8 implementation."""
    return _create_vllm_w8a8_linear_kernel(
        config,
        layer_param_names,
        module_name,
    )


__all__ = [
    "create_w8a8_linear_kernel",
]
