# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606).

The two concrete variants are:
  - ``HyperConnectionBase``  - simple average pooling across hc_count parallel
    streams (equivalent to hyperconnection_average).
  - ``GatedResidualSimple``  - learnable low-rank gated mixing and injection
    (gated_residual_simple).

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout). The local torch
implementation consumes the hyper input viewed as ``[..., HC, HS]``.

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidualSimple(hc_config, role="attn")
    self.mlp_hc = GatedResidualSimple(hc_config, role="mlp")

    hidden_states, residual = self.attn_hc.mix(hidden_states)
    hidden_states = attention(hidden_states)
    hidden_states = self.attn_hc.combine(hidden_states, residual)

    hidden_states, residual = self.mlp_hc.mix(hidden_states)
    hidden_states = mlp(hidden_states)
    hidden_states = self.mlp_hc.combine(hidden_states, residual)
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

try:
    # Importing the leaf module registers opaque custom ops before Dynamo sees
    # the model.  The module itself is vendor-neutral Triton and keeps CPU or
    # unsupported accelerator execution on the torch formula below.
    from ..gpu.ops.hyperconnection import (
        can_use_hc_combine_norm_triton,
        can_use_hc_inject_triton,
        can_use_hc_triton,
    )
except ImportError:  # common/config-only imports do not require a GPU runtime
    can_use_hc_combine_norm_triton = None
    can_use_hc_inject_triton = None
    can_use_hc_triton = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class HyperConnectionConfig:
    """Configuration shared by all HyperConnection variants."""

    hc_count: int = 4
    hidden_size: int = 64
    params_dtype: torch.dtype = torch.bfloat16
    mtp_hc: bool = False
    hc_lowrank: int = 16
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = False


class GroupedGemmaRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        group_count = (
            hidden_states.shape[-1] // self.group_size
            if self.group_size is not None
            else 1
        )
        if can_use_hc_triton is not None and can_use_hc_triton(
            hidden_states, self.weight
        ):
            return torch.ops.vllm.qwen4_grouped_gemma_rmsnorm(
                hidden_states,
                self.weight,
                group_count,
                self.variance_epsilon,
            )
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (
                grouped * torch.rsqrt(variance + self.variance_epsilon)
            ).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


# ---------------------------------------------------------------------------
# Average-pooling variant
# ---------------------------------------------------------------------------
class HyperConnectionBase(nn.Module):
    """Average-pooling HyperConnection (``hyperconnection_average``).

    Splits the incoming ``[..., HC*HS]`` tensor (HC outer, HS inner) into
    ``HC`` parallel streams, averages them for the block input, and
    broadcasts the block output back to every stream.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        layer_idx: int | None = None,
        role: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.role = role

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    def mix(self, hyper_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Average the HC streams into a single block input."""
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        # [*, HC, HS] — mean over HC (dim=-2).
        unflat = hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = unflat.mean(dim=-2)
        return mixed_input, hyper_input

    def combine(
        self, block_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast the block output back to every stream."""
        assert residual.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        residual_reshaped = residual.unflatten(-1, (self.hc_count, self.hidden_size))
        combined = residual_reshaped + block_output.unsqueeze(-2)
        return combined.flatten(-2)


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
class GatedResidualSimple(HyperConnectionBase):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    This is not the matrix-valued mHC/Sinkhorn formulation. It produces a
    per-channel input gate through a low-rank MLP and one scalar injection
    gate per residual stream.

    ``mix()`` applies GemmaRMSNorm per HC stream and projects through a
    low-rank sigmoid gate to produce a single block input. ``combine()``
    injects the block output back into each stream through a learned
    per-stream injection weight.

    Accelerator paths may fuse the elementwise reductions with Triton;
    portable PyTorch fallbacks and tensor-parallel collectives are supplied
    independently.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        layer_idx: int | None = None,
        role: str | None = None,
        use_mix: bool = True,
        use_combine: bool = True,
    ) -> None:
        super().__init__(config, layer_idx, role)
        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- raw Linear names (checkpoint-compatible) ------------------------
        self.register_buffer(
            "_packed_down_inject_weight",
            None,
            persistent=False,
        )
        if use_mix and use_combine:
            packed = torch.empty(
                (config.hc_lowrank + self.hc_count, self.hyper_hidden_size),
                dtype=config.params_dtype,
            )
            nn.init.uniform_(
                packed,
                -(self.hyper_hidden_size**-0.5),
                self.hyper_hidden_size**-0.5,
            )
            # Meta construction avoids allocating two temporary real weights;
            # the checkpoint-visible parameters are contiguous views of the
            # single packed allocation from the start.
            self.input_mix_weight_down = nn.Linear(
                self.hyper_hidden_size,
                config.hc_lowrank,
                bias=False,
                dtype=config.params_dtype,
                device="meta",
            )
            self.block_inject_weight = nn.Linear(
                self.hyper_hidden_size,
                self.hc_count,
                bias=False,
                dtype=config.params_dtype,
                device="meta",
            )
            self.input_mix_weight_down.weight = nn.Parameter(
                packed[: config.hc_lowrank], requires_grad=False
            )
            self.block_inject_weight.weight = nn.Parameter(
                packed[config.hc_lowrank :], requires_grad=False
            )
            self._packed_down_inject_weight = packed
        elif use_mix:
            self.input_mix_weight_down = nn.Linear(
                self.hyper_hidden_size,
                config.hc_lowrank,
                bias=False,
                dtype=config.params_dtype,
            )
        elif use_combine:
            self.block_inject_weight = nn.Linear(
                self.hyper_hidden_size,
                self.hc_count,
                bias=False,
                dtype=config.params_dtype,
            )
        if use_mix:
            self.input_mix_weight_up = nn.Linear(
                config.hc_lowrank,
                self.hyper_hidden_size,
                bias=False,
                dtype=config.params_dtype,
            )

    def pack_down_inject_weights(self) -> bool:
        """Pack the two projections that share normalized HC input.

        The original named parameters become disjoint views of the packed
        storage.  This preserves checkpoint/reload names without retaining a
        second copy of roughly 6.6 MB per production HC module.
        """

        if not hasattr(self, "input_mix_weight_down") or not hasattr(
            self, "block_inject_weight"
        ):
            return False
        down_weight = self.input_mix_weight_down.weight
        inject_weight = self.block_inject_weight.weight
        if down_weight.device.type == "meta" or inject_weight.device.type == "meta":
            return False
        if self._packed_down_inject_weight is not None:
            packed = self._packed_down_inject_weight
            if (
                packed.device == down_weight.device == inject_weight.device
                and packed.untyped_storage().data_ptr()
                == down_weight.untyped_storage().data_ptr()
                == inject_weight.untyped_storage().data_ptr()
            ):
                return True
        packed = torch.empty(
            (down_weight.shape[0] + inject_weight.shape[0], down_weight.shape[1]),
            dtype=down_weight.dtype,
            device=down_weight.device,
        )
        with torch.no_grad():
            packed[: down_weight.shape[0]].copy_(down_weight)
            packed[down_weight.shape[0] :].copy_(inject_weight)
        self._packed_down_inject_weight = packed
        self.input_mix_weight_down.weight = nn.Parameter(
            packed[: down_weight.shape[0]], requires_grad=False
        )
        self.block_inject_weight.weight = nn.Parameter(
            packed[down_weight.shape[0] :], requires_grad=False
        )
        return True

    def _normalize(self, hyper_input: torch.Tensor) -> torch.Tensor:
        if self.config.hc_per_branch_norm:
            return self.hc_norm(hyper_input)
        return self.hc_norm(
            hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        ).flatten(-2)

    def _mix_from_normed(
        self, hyper_input_normed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the projection/gate half of ``mix`` on a normalized state."""

        injection_logits: torch.Tensor | None = None
        if self._packed_down_inject_weight is not None:
            down_and_inject = F.linear(
                hyper_input_normed, self._packed_down_inject_weight
            )
            down, injection_logits = down_and_inject.split(
                (self.config.hc_lowrank, self.hc_count), dim=-1
            )
        else:
            down = F.linear(hyper_input_normed, self.input_mix_weight_down.weight)
        gate = F.silu(down / self.hc_count)
        gate_logits = F.linear(gate, self.input_mix_weight_up.weight)
        if can_use_hc_triton is not None and can_use_hc_triton(
            gate_logits, hyper_input_normed
        ):
            mixed_input = torch.ops.vllm.qwen4_hc_gate_reduce(
                gate_logits,
                hyper_input_normed,
                self.hc_count,
            )
        else:
            gate = torch.sigmoid(gate_logits).unflatten(
                -1, (self.hc_count, self.hidden_size)
            )
            mixed_input = (
                gate
                * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))
            ).mean(dim=-2)
        return mixed_input.to(hyper_input_normed.dtype), injection_logits

    def mix(
        self, hyper_input: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    ]:
        """Mix: RMSNorm -> low-rank gate -> gated mean."""
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if not hasattr(self, "input_mix_weight_down"):
            raise RuntimeError("mix was disabled for this hyper-connection")
        hyper_input_normed = self._normalize(hyper_input)
        mixed_input, injection_logits = self._mix_from_normed(hyper_input_normed)
        residuals = (hyper_input, hyper_input_normed, injection_logits)
        return mixed_input.to(hyper_input.dtype), residuals

    def mix_delayed(
        self, hyper_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Prepare a block input while retaining the unmaterialized HC state.

        The official NVIDIA path returns the raw multi-stream state and the
        pending injection logits separately.  Keep the original ``mix`` API
        for callers/tests that need eager combine, and expose this explicit
        adapter for the delayed decoder pipeline.
        """

        block_input, residuals = self.mix(hyper_input)
        return hyper_input, block_input, residuals[2]

    def combine_pending(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize a delayed block output with its saved injection logits."""

        assert hidden_states.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        assert injection_logits.shape[-1] == self.hc_count
        if can_use_hc_inject_triton is not None and can_use_hc_inject_triton(
            injection_logits, block_output, hidden_states
        ):
            return torch.ops.vllm.qwen4_hc_inject_combine(
                injection_logits,
                block_output,
                hidden_states,
                self.hc_count,
            )
        residual = hidden_states.unflatten(-1, (self.hc_count, self.hidden_size))
        injection_weight = 2.0 * torch.sigmoid(injection_logits / self.hc_count)
        output = residual + block_output.unsqueeze(-2) * injection_weight.unsqueeze(-1)
        return output.flatten(-2).to(hidden_states.dtype)

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | None,
        prev_injection: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        This is the small API bridge needed to move the self-developed layer
        from eager ``combine`` to the official delayed-combine schedule.  On
        supported CUDA/Triton shapes it uses the fused combine+grouped-norm
        op; CPU, ROCm, and unusual layouts use the mathematically identical
        combine followed by the existing grouped RMSNorm fallback.
        """

        if (prev_block_output is None) != (prev_injection is None):
            raise ValueError("pending HC output and injection must be provided together")

        if prev_block_output is None:
            return self.mix_delayed(hidden_states)

        if (
            can_use_hc_combine_norm_triton is not None
            and can_use_hc_combine_norm_triton(
                prev_injection,
                prev_block_output,
                hidden_states,
                self.hc_norm.weight,
            )
        ):
            combined, normalized = torch.ops.vllm.qwen4_hc_combine_norm(
                hidden_states,
                prev_block_output,
                prev_injection,
                self.hc_norm.weight,
                self.config.rms_norm_eps,
                self.hc_count,
            )
        else:
            combined = self.combine_pending(
                hidden_states, prev_block_output, prev_injection
            )
            normalized = self._normalize(combined)
        block_input, injection = self._mix_from_normed(normalized)
        return combined, block_input, injection

    def combine(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        if not hasattr(self, "block_inject_weight"):
            raise RuntimeError("combine was disabled for this hyper-connection")
        hyper_input, hyper_input_normed, injection_logits = residuals
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        # The paired mix keeps its normalized hyper input so combine uses the
        # same HC module's injection weight.
        if injection_logits is None:
            injection_logits = F.linear(
                hyper_input_normed, self.block_inject_weight.weight
            )
        return self.combine_pending(hyper_input, block_output, injection_logits)


def pack_gated_hc_projection_weights(root: nn.Module) -> int:
    """Pack every eligible HC module after checkpoint loading."""

    packed = 0
    for module in root.modules():
        if isinstance(module, GatedResidualSimple):
            packed += int(module.pack_down_inject_weights())
    return packed


__all__ = [
    "GatedResidualSimple",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "pack_gated_hc_projection_weights",
]
