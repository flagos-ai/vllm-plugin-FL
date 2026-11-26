# Copyright (c) 2026 BAAI. All rights reserved.

"""Native replacements for the V2 "align" Mamba state kernels on TXDA.

The V2 runner (``vllm/v1/worker/gpu``) keeps mamba prefix-cache checkpoints in
step with the running state through three Triton kernels in
``vllm.v1.worker.mamba_utils``:

* ``get_aligned_state_indices_multi_group_kernel`` gathers each group's
  physical block ids at the block-aligned checkpoint position;
* ``precopy_mamba_align_fused_kernel`` copies the state of one block column
  into the next when a request crosses a block boundary;
* ``postprocess_mamba_fused_kernel`` (align variant) re-checkpoints after a
  step.

The copy kernels address memory through a *runtime-loaded* pointer: they
``tl.load`` a ``state_base_addrs`` / ``block_table_ptrs`` entry and cast the
resulting integer to a pointer. The TXDA Triton backend miscompiles that
pattern — the MLIR lowering aborts with ``MutableOperandRange::operator[]``
and, where it does compile, the load reads address 0 — so the kernels cannot
run on this device. The V1 ``mamba_memcpy`` patch works around the same
limitation by resolving copies against the owning tensors instead of raw
addresses.

The replacements below keep the copy semantics but express them as native
tensor indexing on the owning state tensors. Decisions stay on the device:
only gather/scatter index tensors are built, so no per-step host sync is
introduced.

Without speculative decoding (which TXDA rejects, see
``PlatformFL.check_and_update_config``) ``num_accepted_tokens`` is always 1, so
the accepted-token bias of a copy is always 0 and every copy moves a whole
block. The helpers assert that instead of silently producing wrong results if
the invariant ever changes.

Removal condition: delete this module once the TXDA Triton backend supports
casting loaded integers to pointers.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_installed = False

# Distinguishes "attribute was absent before patching" from "was None".
_MISSING = object()


def _iter_state_slots(kv_cache_config, forward_context, copy_funcs, group_ids):
    """Yield ``(state, group_local_idx)`` in metadata order.

    Mirrors ``MambaSpecDecodeGPUContext._populate_metadata`` so the recorded
    tensor list lines up index-for-index with the flattened metadata arrays
    the kernels build.
    """
    from vllm.v1.worker.mamba_utils import _get_mamba_spec_for_layer

    for group_local_idx, group_id in enumerate(group_ids):
        kv_cache_group = kv_cache_config.kv_cache_groups[group_id]
        for layer_name in kv_cache_group.layer_names:
            mamba_spec = _get_mamba_spec_for_layer(kv_cache_group, layer_name)
            state_copy_funcs = copy_funcs[mamba_spec.mamba_type]
            kv_caches = forward_context[layer_name].kv_cache
            for state_type_idx in range(len(state_copy_funcs)):
                yield kv_caches[state_type_idx], group_local_idx


def aligned_state_indices(
    out: torch.Tensor,
    aligned_block_tables: list[torch.Tensor],
    seq_lens: torch.Tensor,
    num_reqs: int,
    block_size: int,
) -> torch.Tensor:
    """Pure-torch ``get_aligned_state_indices_multi_group_kernel``."""
    if num_reqs == 0:
        return out[:, :0]
    rows = torch.arange(num_reqs, device=seq_lens.device)
    lens = seq_lens[:num_reqs].to(torch.int64)
    # ``tl.maximum((seq_lens - 1) // CACHE_BLOCK_SIZE, 0)``
    first_slot = torch.clamp((lens - 1) // block_size, min=0)
    slots = torch.arange(out.shape[2], device=seq_lens.device)
    table_width = aligned_block_tables[0].shape[1]
    columns = torch.clamp(first_slot[:, None] + slots[None, :], max=table_width - 1)
    for group_idx, table in enumerate(aligned_block_tables):
        out[group_idx, :num_reqs] = table[rows[:, None], columns]
    return out[:, :num_reqs]


def copy_mamba_state_blocks(
    states: list[torch.Tensor],
    state_groups: list[int],
    block_tables: list[torch.Tensor],
    rows: torch.Tensor,
    src_col: torch.Tensor,
    dst_col: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    """Copy state between block columns, per ``(row, column)`` pair.

    Every state tensor is ``[num_blocks, ...]``. The copy body mirrors
    ``_copy_mamba_state_block``: a conditional state (dim 3, ``state_len`` at
    1) shifts its window by ``bias`` tokens, a temporal state copies the whole
    block.
    """
    if bool((bias != 0).any()):
        raise RuntimeError(
            "TXDA native Mamba align copies require num_accepted_tokens=1 "
            f"(got a token bias of {int(bias.max())}); speculative decoding "
            "is not supported on TXDA."
        )
    for state, group_idx in zip(states, state_groups):
        table = block_tables[group_idx]
        src_blocks = table[rows, src_col].to(torch.int64)
        dst_blocks = table[rows, dst_col].to(torch.int64)
        if state.dim() == 3:
            # Conditional state: (num_blocks, state_len, dim).
            state_len = state.shape[1]
            state[dst_blocks, :state_len] = state[src_blocks, :state_len]
        else:
            state[dst_blocks] = state[src_blocks]


def install_mamba_align_native():
    """Replace the V2 align kernels that address memory via loaded pointers.

    Returns an ``undo`` callable that restores the replaced class attributes and
    the install flag, or ``None`` when nothing was changed (already installed, or
    already patched by a previous attempt). Returning the undo lets the caller
    roll a partially applied install back instead of leaving two of the four
    replaced methods behind while ``_fl_txda_native_align`` still claims success.
    """
    global _installed
    if _installed:
        return None

    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.v1.worker import mamba_utils

    ctx_cls = mamba_utils.MambaSpecDecodeGPUContext
    if getattr(ctx_cls, "_fl_txda_native_align", False):
        return

    if is_conv_state_dim_first():
        # The DS layout interleaves the slide axis across dim rows, so a
        # whole-block copy is not equivalent. vLLM defaults to SD; refuse
        # rather than produce wrong states if that ever changes.
        raise RuntimeError(
            "TXDA native Mamba align copies support the SD conv state layout "
            "only; unset VLLM_SSM_CONV_STATE_LAYOUT."
        )

    original_init = ctx_cls.initialize_from_forward_context

    def initialize_from_forward_context(
        self,
        kv_cache_config,
        forward_context,
        mamba_state_copy_funcs,
        block_tables,
        aligned_index_block_tables=None,
    ) -> None:
        original_init(
            self,
            kv_cache_config,
            forward_context,
            mamba_state_copy_funcs,
            block_tables,
            aligned_index_block_tables,
        )
        # Record the state tensors and block tables the kernels could only
        # reach through raw addresses.
        states, groups = [], []
        for state, group_local_idx in _iter_state_slots(
            kv_cache_config, forward_context, mamba_state_copy_funcs,
            self.mamba_group_ids,
        ):
            states.append(state)
            groups.append(group_local_idx)
        if len(states) != self.num_states:
            raise RuntimeError(
                f"TXDA Mamba align: {len(states)} state tensors for "
                f"{self.num_states} metadata slots"
            )
        self._fl_states = states
        self._fl_state_groups = groups
        self._fl_block_tables = list(block_tables)
        self._fl_aligned_block_tables = list(
            aligned_index_block_tables
            if aligned_index_block_tables is not None
            else block_tables
        )

    def _aligned_indices(self, seq_lens, num_reqs):
        return aligned_state_indices(
            self.aligned_state_indices,
            self._fl_aligned_block_tables,
            seq_lens,
            num_reqs,
            self.block_size,
        )

    def _precopy(
        self, num_reqs, state_idx_gpu, src_col_gpu, token_bias_gpu, idx_mapping
    ):
        if num_reqs == 0 or not self.is_initialized:
            return
        device = state_idx_gpu.device
        if idx_mapping is None:
            rows = torch.arange(num_reqs, device=device)
        else:
            rows = idx_mapping[:num_reqs].to(torch.int64)
            rows = rows[rows >= 0]
        if rows.numel() == 0:
            return
        dst_col = state_idx_gpu[rows].to(torch.int64)
        src_col = src_col_gpu[rows].to(torch.int64)
        bias = token_bias_gpu[rows].to(torch.int64)
        # ``src_col < 0`` = fresh state; ``src_col == dst_col`` = same block.
        active = (src_col >= 0) & (src_col != dst_col)
        if not bool(active.any()):
            return
        copy_mamba_state_blocks(
            self._fl_states,
            self._fl_state_groups,
            self._fl_block_tables,
            rows[active],
            src_col[active],
            dst_col[active],
            bias[active],
        )

    def _postprocess_align(
        self,
        num_reqs,
        num_accepted_tokens_gpu,
        state_idx_gpu,
        new_num_computed_tokens_gpu,
        idx_mapping,
    ):
        if num_reqs == 0 or not self.is_initialized:
            return
        rows = idx_mapping[:num_reqs].to(torch.int64)
        rows = rows[rows >= 0]
        if rows.numel() == 0:
            return
        accepted = num_accepted_tokens_gpu[rows].to(torch.int64)
        src_col = state_idx_gpu[rows].to(torch.int64)
        new_computed = new_num_computed_tokens_gpu[rows].to(torch.int64)

        running_state = new_computed - accepted + 1
        aligned = (new_computed // self.block_size) * self.block_size
        needs_copy = aligned >= running_state
        bias = aligned - running_state
        dest_col = aligned // self.block_size - 1

        # Mirrors the kernel: reset the accepted count when the checkpoint
        # block is already the running block and no token shift is needed.
        reset = needs_copy & (src_col == dest_col) & (bias == 0)
        if bool(reset.any()):
            num_accepted_tokens_gpu[rows[reset]] = 1

        move = needs_copy & (src_col != dest_col)
        if not bool(move.any()):
            return
        copy_mamba_state_blocks(
            self._fl_states,
            self._fl_state_groups,
            self._fl_block_tables,
            rows[move],
            src_col[move],
            dest_col[move],
            bias[move],
        )

    prev = {
        name: getattr(ctx_cls, name, _MISSING)
        for name in (
            "initialize_from_forward_context",
            "compute_aligned_state_indices",
            "run_fused_precopy",
            "run_fused_postprocess_align",
            "_fl_txda_native_align",
        )
    }
    installed = {
        "initialize_from_forward_context": initialize_from_forward_context,
        "compute_aligned_state_indices": _aligned_indices,
        "run_fused_precopy": _precopy,
        "run_fused_postprocess_align": _postprocess_align,
        "_fl_txda_native_align": True,
    }

    def _undo() -> None:
        # Best-effort: keep restoring the remaining attributes even if one
        # restore raises, otherwise a single failure leaves the class in the
        # half-patched state this undo exists to prevent.
        for name, old in prev.items():
            try:
                if old is _MISSING:
                    if hasattr(ctx_cls, name):
                        delattr(ctx_cls, name)
                else:
                    setattr(ctx_cls, name, old)
            except Exception:  # pragma: no cover - defensive
                logger.warning(
                    "could not restore %s.%s during install rollback",
                    ctx_cls.__name__,
                    name,
                    exc_info=True,
                )
        globals()["_installed"] = False

    try:
        for name, value in installed.items():
            setattr(ctx_cls, name, value)
    except Exception:
        _undo()
        raise

    logger.info("Installed native TXDA Mamba align state copies")
    _installed = True
    return _undo
