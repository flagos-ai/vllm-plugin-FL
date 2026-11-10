# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Restore worker helpers that vLLM 0.32 removed.

The FL worker and model runner are adapted from vLLM v0.28's GPU worker and
call a few helpers that the 0.32 release dropped during its worker refactor.
Rather than fork those call sites, re-publish the helpers on the upstream
modules where the worker already looks for them, so a single source tree runs
against both targets.

Every patch prefers whatever the installed vLLM provides and only installs a
v0.28-equivalent when the name is genuinely absent, so a build that still ships
the original keeps using it.

Patched names:

``vllm.v1.worker.utils.is_residual_scattered_for_sp``
    Sequence-parallel residual predicate; the logic moved into
    ``forward_context``/model code upstream. Every ``vllm_config`` field it
    reads still exists in 0.32.

``vllm.model_executor.layers.rotary_embedding.XDRotaryEmbedding``
``vllm.model_executor.models.interfaces.SupportsXDRoPE``
``vllm.model_executor.models.interfaces.supports_xdrope``
    The XDRoPE path was folded into MRoPE (``xdrope_section`` is now a legacy
    alias of ``mrope_section``). MRoPE is *not* a drop-in: XDRoPE's position
    helper takes no ``mrope_position_delta``. ``supports_xdrope`` becomes a real
    capability probe, so it correctly reports False for 0.32 models and the
    XDRoPE branch is simply never taken.

``vllm.v1.outputs.RoutedExpertsLists``
``vllm.v1.outputs.RoutedExpertsTensors``
    Carriers for the opt-in routed-experts output, redesigned upstream around
    ``vllm.distributed.aux_output_connector``. Only the gated
    ``enable_return_routed_experts`` path uses them.
"""

from __future__ import annotations

from functools import cache
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Sequence parallelism
# ---------------------------------------------------------------------------
def _is_residual_scattered_for_sp(vllm_config, num_input_tokens: int) -> bool:
    """Return True when the residual is scattered across TP ranks for SP.

    Port of vLLM v0.28's ``is_residual_scattered_for_sp``. The residual is
    scattered only when sequence parallelism is enabled, TP > 1, and full-graph
    compilation is in effect.
    """
    pass_config = vllm_config.compilation_config.pass_config
    if not getattr(pass_config, "enable_sp", False):
        return False

    tp = vllm_config.parallel_config.tensor_parallel_size
    if tp == 1:
        return False

    compilation_config = vllm_config.compilation_config
    assert (
        getattr(compilation_config, "use_inductor_graph_partition", False)
        or not compilation_config.splitting_ops
    ), "Sequence parallelism requires full-graph compilation"

    # SP always pads num_input_tokens to a multiple of tp earlier.
    assert num_input_tokens % tp == 0

    return True


# ---------------------------------------------------------------------------
# XDRoPE
# ---------------------------------------------------------------------------
@runtime_checkable
class _SupportsXDRoPE(Protocol):
    """Structural interface for models providing XDRoPE input positions."""

    def get_xdrope_input_positions(self, *args, **kwargs):
        pass


def _supports_xdrope(model: object) -> bool:
    """Return True when *model* actually implements the XDRoPE position hook."""
    return callable(getattr(model, "get_xdrope_input_positions", None))


class _XDRotaryEmbedding:
    """Static XDRoPE position helper ported from vLLM v0.28."""

    @staticmethod
    def get_next_input_positions_tensor(
        out: np.ndarray,
        out_offset: int,
        context_len: int,
        num_new_tokens: int,
    ) -> None:
        """Write the next XDRoPE input positions into ``out`` in place.

        XDRoPE positions advance with the context length directly, without the
        ``mrope_position_delta`` shift that MRoPE applies.
        """
        values = np.arange(
            context_len,
            context_len + num_new_tokens,
            dtype=out.dtype,
        )
        out[:, out_offset : out_offset + num_new_tokens] = values


# ---------------------------------------------------------------------------
# Routed experts carriers
# ---------------------------------------------------------------------------
class _RoutedExpertsLists(NamedTuple):
    """CPU-side routed experts, batched per scheduler step."""

    # (num_scheduled_tokens, num_layers, num_experts_per_tok)
    routing_data: np.ndarray
    # (num_scheduled_tokens,)
    slot_mapping: np.ndarray


class _RoutedExpertsTensors(NamedTuple):
    """Device-side snapshot of routed-experts data, pending async D2H."""

    # (num_scheduled_tokens, num_layers, num_experts_per_tok)
    routing_data: torch.Tensor
    # (num_scheduled_tokens,)
    slot_mapping: torch.Tensor

    def to_cpu_nonblocking(self) -> _RoutedExpertsTensors:
        """Issue a non-blocking D2H copy on the current stream."""
        if self.routing_data.device.type == "cpu":
            return self
        return _RoutedExpertsTensors(
            self.routing_data.to("cpu", non_blocking=True),
            self.slot_mapping.to("cpu", non_blocking=True),
        )

    def tolists(self) -> _RoutedExpertsLists:
        """Convert to the numpy-backed form consumed by the scheduler."""
        return _RoutedExpertsLists(
            self.routing_data.cpu().numpy(),
            self.slot_mapping.cpu().numpy(),
        )


def _set_missing(module, name: str, value) -> None:
    """Publish *value* as ``module.name`` only when the name is absent."""
    if not hasattr(module, name):
        setattr(module, name, value)


@cache
def apply_worker_symbols_032() -> bool:
    """Re-publish v0.28 worker helpers that vLLM 0.32 removed. Idempotent."""
    from vllm.model_executor.layers import rotary_embedding
    from vllm.model_executor.models import interfaces
    from vllm.v1 import outputs
    from vllm.v1.worker import utils as worker_utils

    _set_missing(
        worker_utils, "is_residual_scattered_for_sp", _is_residual_scattered_for_sp
    )
    _set_missing(rotary_embedding, "XDRotaryEmbedding", _XDRotaryEmbedding)
    _set_missing(interfaces, "SupportsXDRoPE", _SupportsXDRoPE)
    _set_missing(interfaces, "supports_xdrope", _supports_xdrope)
    _set_missing(outputs, "RoutedExpertsLists", _RoutedExpertsLists)
    _set_missing(outputs, "RoutedExpertsTensors", _RoutedExpertsTensors)

    return True
