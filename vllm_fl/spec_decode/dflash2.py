# SPDX-License-Identifier: Apache-2.0
"""DFlash 2 proposer for the vLLM 0.24 CPU runner."""

import secrets

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer


def dflash_target_rope_is_neox_style(target_model: nn.Module) -> bool | None:
    """Read the target's RoPE layout before constructing the draft model."""
    language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    for module in language_model.modules():
        style = getattr(module, "is_neox_style", None)
        if isinstance(style, bool):
            return style
    return None


def compute_dflash2_anchor_positions(
    target_positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_rejected_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the absolute positions used to key the first selector sample."""
    position_row = (
        target_positions[0] if target_positions.ndim > 1 else target_positions
    )
    context_ends = query_start_loc[1:].to(torch.int64)
    if num_rejected_tokens is not None:
        context_ends = context_ends - num_rejected_tokens.to(torch.int64)
    last_context_indices = context_ends - 1
    if torch.any(last_context_indices < 0):
        raise ValueError("DFlash2 received an empty target context")
    return position_row.index_select(0, last_context_indices).add(1)


class DFlash2Proposer(DFlashProposer):
    """Parallel DFlash proposer with the DFlash 2 candidate path selector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner)
        self.runner = runner
        self._anchor_token_ids: torch.Tensor | None = None
        self._anchor_positions: torch.Tensor | None = None
        self._request_sampling_seeds: dict[str, int] = {}

    def load_model(self, target_model: nn.Module) -> None:
        # DFlash checkpoints do not record the target's rotary layout.  A
        # mismatch is silent but damages every draft attention layer.
        style = dflash_target_rope_is_neox_style(target_model)
        if style is not None:
            self.draft_model_config.hf_config.is_neox_style = style
        super().load_model(target_model)

    def propose(self, *args, **kwargs) -> torch.Tensor | list[list[int]]:
        next_token_ids = kwargs.get("next_token_ids")
        if next_token_ids is None and len(args) > 4:
            next_token_ids = args[4]
        if next_token_ids is None:
            raise ValueError("DFlash2 requires the verified anchor token IDs")

        target_positions = kwargs.get("target_positions")
        if target_positions is None and len(args) > 2:
            target_positions = args[2]
        common_attn_metadata = kwargs.get("common_attn_metadata")
        if common_attn_metadata is None and len(args) > 6:
            common_attn_metadata = args[6]
        num_rejected_tokens = kwargs.get("num_rejected_tokens_gpu")
        if num_rejected_tokens is None and len(args) > 9:
            num_rejected_tokens = args[9]
        if target_positions is None or common_attn_metadata is None:
            raise ValueError("DFlash2 requires target positions and attention metadata")

        # vLLM 0.24 already supplies CommonAttentionMetadata for the drafter's
        # KV-cache group.  Do not swap only the block table as the 0.20 bridge
        # did; doing so would separate it from its matching slot mapping.
        self._anchor_token_ids = next_token_ids
        # The first sampled draft follows next_token_ids, whose absolute
        # position is one past the final target context token.
        self._anchor_positions = compute_dflash2_anchor_positions(
            target_positions,
            common_attn_metadata.query_start_loc,
            num_rejected_tokens,
        )
        try:
            return super().propose(*args, **kwargs)
        finally:
            self._anchor_token_ids = None
            self._anchor_positions = None

    def _sampling_seeds(
        self,
        sampling_metadata: SamplingMetadata,
        batch_size: int,
    ) -> list[int]:
        """Get stable per-request seeds, matching upstream sampling states."""
        request_ids: list[str] = []
        runner = getattr(self, "runner", None)
        if runner is not None and hasattr(runner, "input_batch"):
            request_ids = runner.input_batch.req_ids[:batch_size]

        if request_ids:
            active = set(request_ids)
            cached_seeds = getattr(self, "_request_sampling_seeds", {})
            self._request_sampling_seeds = {
                req_id: seed
                for req_id, seed in cached_seeds.items()
                if req_id in active
            }

        seeds = []
        for row_index in range(batch_size):
            generator = sampling_metadata.generators.get(row_index)
            if generator is not None:
                seeds.append(generator.initial_seed())
                continue
            if row_index < len(request_ids):
                request_id = request_ids[row_index]
                cached_seeds = getattr(self, "_request_sampling_seeds", {})
                seed = cached_seeds.get(request_id)
                if seed is None:
                    seed = secrets.randbits(63)
                    cached_seeds[request_id] = seed
                    self._request_sampling_seeds = cached_seeds
                seeds.append(seed)
            else:
                seeds.append(secrets.randbits(63))
        return seeds

    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Select the trained DFlash2 path and return its exact proposal q."""
        if self._anchor_token_ids is None:
            raise RuntimeError("DFlash2 anchor tokens were not initialized")
        if self._anchor_positions is None:
            raise RuntimeError("DFlash2 anchor positions were not initialized")
        batch_size = self._anchor_token_ids.shape[0]
        steps = self.num_speculative_tokens
        hidden = hidden_states.view(batch_size, steps, -1)
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(batch_size, steps, -1)
        unary_logits = unary_logits.view_as(candidate_ids)
        if not sampling_metadata.all_greedy:
            if sampling_metadata.temperature is None:
                raise RuntimeError("random DFlash2 sampling requires temperatures")
            path, sparse_probs = self.model.model.candidate_selector.select_stochastic(
                candidate_ids,
                unary_logits,
                hidden,
                self._anchor_token_ids,
                sampling_metadata.temperature,
                self._sampling_seeds(sampling_metadata, batch_size),
                self._anchor_positions,
            )
            vocab_size = int(self.model.config.vocab_size)
            draft_probs = torch.zeros(
                (batch_size, steps, vocab_size),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            draft_probs.scatter_(2, candidate_ids, sparse_probs)
            return path.reshape(-1), draft_probs.flatten(0, 1)
        path = self.model.model.candidate_selector.select_greedy(
            candidate_ids,
            unary_logits,
            hidden,
            self._anchor_token_ids,
        )
        return path.reshape(-1), None
