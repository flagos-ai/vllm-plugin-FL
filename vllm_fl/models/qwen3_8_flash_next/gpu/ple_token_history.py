# Copyright (c) 2026 BAAI. All rights reserved.
"""GPU token history for the legacy runner's PLE n-gram inputs."""

from collections.abc import Iterable, Sequence

import numpy as np
import torch


class PLETokenHistory:
    """Keep consumed tokens independently of InputBatch's movable CPU rows.

    Async scheduling writes placeholders to CPU output history. The runner
    resolves those placeholders in ``input_ids`` on the device before calling
    ``prepare``. Retaining these real tokens makes both decode and rollback
    independent of CPU sampling completion. A full history, rather than just
    the last n-gram, also supports recomputation from an earlier position.

    CPU tokens are read only on admission/resume or a forward prefix jump.
    The caller must invalidate finished/replaced/resumed requests, and protect
    pinned host buffers with its input-preparation event before the next call.
    No sampling result or device tensor is copied back to the host here.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_tokens: int,
        context_len: int,
        eos_token_id: int,
        device: torch.device,
        pin_memory: bool,
    ) -> None:
        self.max_model_len = max_model_len
        self.context_len = context_len
        self.eos_token_id = eos_token_id
        self.sentinel = max_num_reqs * max_model_len
        self.tokens = torch.empty(self.sentinel + 1, dtype=torch.int32, device=device)
        self.tokens[self.sentinel].fill_(eos_token_id)
        self._slots: dict[str, int] = {}
        self._valid_lengths: dict[str, int] = {}
        self._free_slots = list(reversed(range(max_num_reqs)))
        self._offsets = np.arange(-context_len, 0, dtype=np.int64)
        self._read_cpu = torch.empty(
            (max_num_reqs, context_len), dtype=torch.int64, pin_memory=pin_memory
        )
        self._read_gpu = torch.empty_like(self._read_cpu, device=device)
        self._write_cpu = torch.empty(
            max_num_tokens, dtype=torch.int64, pin_memory=pin_memory
        )
        self._write_gpu = torch.empty_like(self._write_cpu, device=device)
        self._read_np = self._read_cpu.numpy()
        self._write_np = self._write_cpu.numpy()

    def forget(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            slot = self._slots.pop(req_id, None)
            if slot is not None:
                self._free_slots.append(slot)
                del self._valid_lengths[req_id]

    def prepare(
        self,
        *,
        req_ids: Sequence[str],
        num_computed_tokens: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        token_ids_cpu: torch.Tensor,
        is_token_ids: np.ndarray | None,
        req_indices: np.ndarray,
        query_positions: np.ndarray,
        input_ids: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Read the left context and save this step's actual input tokens.

        ``context`` includes graph padding rows; all other arrays contain only
        real requests/tokens. This runs before the model graph, on its stream.
        Dummy/capture warmups must not call it or advance the history.
        """
        num_reqs = len(req_ids)
        num_tokens = input_ids.numel()
        self.forget(self._slots.keys() - set(req_ids))
        slots = np.empty(num_reqs, dtype=np.int64)
        for row, req_id in enumerate(req_ids):
            end = int(num_computed_tokens[row])
            next_end = end + int(num_scheduled_tokens[row])
            if not 0 <= end <= next_end <= self.max_model_len:
                raise ValueError("PLE token history positions exceed model capacity")
            if req_id not in self._slots:
                self._slots[req_id] = self._free_slots.pop()
                self._valid_lengths[req_id] = 0
            slot = self._slots[req_id]
            slots[row] = slot
            valid_end = self._valid_lengths[req_id]
            if end > valid_end:
                # On resume, _update_states restores all_token_ids from the
                # scheduler before re-adding the request to InputBatch.
                prefix = token_ids_cpu[row, valid_end:end]
                if is_token_ids is not None:
                    mask = is_token_ids[row, valid_end:end]
                    if not mask.all():
                        prefix = prefix.clone()
                        prefix[~torch.from_numpy(mask)] = self.eos_token_id
                if (prefix.numpy() < 0).any():
                    raise RuntimeError(
                        "PLE cannot initialize token history from async placeholders"
                    )
                start = slot * self.max_model_len + valid_end
                self.tokens[start : start + prefix.numel()].copy_(
                    prefix, non_blocking=True
                )
            # A rollback invalidates the discarded suffix. Subsequent writes
            # overwrite it; a later jump must re-seed from committed CPU ids.
            self._valid_lengths[req_id] = next_end

        read = self._read_np[: context.shape[0]]
        read.fill(self.sentinel)
        positions = num_computed_tokens[:num_reqs, None] + self._offsets
        read[:num_reqs] = np.where(
            positions >= 0,
            slots[:, None] * self.max_model_len + positions,
            self.sentinel,
        )
        self._read_gpu[: context.shape[0]].copy_(
            self._read_cpu[: context.shape[0]], non_blocking=True
        )
        torch.index_select(
            self.tokens,
            0,
            self._read_gpu[: context.shape[0]].flatten(),
            out=context.flatten(),
        )

        write = self._write_np[:num_tokens]
        write[:] = (
            slots[req_indices] * self.max_model_len
            + num_computed_tokens[req_indices]
            + query_positions
        )
        self._write_gpu[:num_tokens].copy_(
            self._write_cpu[:num_tokens], non_blocking=True
        )
        # Qwen's runtime policy keeps indexed writes native. FlagGems'
        # index_copy_ performs a device-to-host bounds check on this path.
        self.tokens[self._write_gpu[:num_tokens]] = input_ids
        return context
