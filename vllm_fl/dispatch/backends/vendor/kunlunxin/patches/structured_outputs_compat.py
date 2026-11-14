# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

# --------------------------------------------------------------------------------
# Hotfix: FlagTree's XPU Triton backend cannot compile the V2 runner's grammar
# bitmask kernel, and -- worse -- miscompiles the bit unpacking it is built on.
#
# ``vllm.v1.worker.gpu.structured_outputs._apply_grammar_bitmask_kernel`` loads
# one packed int32 word per 32 vocabulary entries, unpacks it into a boolean
# mask and reshapes that back to a flat block::
#
#     bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
#     bitmask = bitmask.reshape(BLOCK_SIZE)
#
# Every structured-output request then aborts the engine with::
#
#     .../vllm/v1/worker/gpu/structured_outputs.py:162:0: error: Failures have
#     been detected while processing an MLIR pass pipeline
#     triton.runtime.errors.OutOfResources: out of resource: uni_sram
#
# Measured on P800, holding everything else fixed:
#
#     formulation                                 BLOCK=256  1024  8192
#     upstream (reshape to 1-D, then store)          FAIL    FAIL  FAIL
#     reshape removed, 2-D [word, bit] store         OK      OK    OK
#     reshape kept, scalar predicate removed         FAIL    FAIL  FAIL
#
# so ``.reshape()`` on the unpacked 2-D tensor is what fails to lower, not the
# predicate. But the 2-D store that does compile is not usable either: the
# unpack itself returns wrong bits. Dumping the kernel's own ``blocked`` tensor
# for one random word and comparing against torch shows isolated set bits read
# back as clear, e.g. for packed=-503177457 bits 11, 12, 29 and 31 all came out
# 1 where 0 is correct. A shift by a tensor of bit positions is therefore
# unreliable here, so no rearrangement of this kernel can be trusted.
#
# Two further FlagTree behaviours rule out the obvious repairs: a scalar bool
# AND-ed into a tensor mask is silently dropped (so the per-position predicate
# has to be folded into a bound), and an early return placed after tensor work
# cannot be lowered at all. Both are documented in input_batch_compat.
#
# Replace the launch with torch ops, as penalties_compat does for the penalty
# kernels. The unpacking becomes a shift against a [1, 32] row of bit positions
# and the write becomes a masked_fill_ on the selected logit rows. Rows are
# processed in chunks so the unpacked [rows, vocab] boolean never exceeds a
# bounded size. Semantics are preserved exactly, including resolving each
# request's logit offset from the device ``cu_num_logits`` (adaptive
# verification finalizes those on device) and skipping positions beyond a
# request's logit count.
#
# Verified against a torch reference across 48 configurations (vocab 8192 and
# 5000, mask strides 1 and 4, 1/3/5 masks, chunk sizes 1/4/64, mixing active
# and inactive positions and a vocabulary that is not a multiple of 32): the
# masked logits match exactly.
# --------------------------------------------------------------------------------

import logging

import torch

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_grammar_bitmask_torch"

# Unpacked rows held at once. Each row costs vocab_size bools, so 32 rows of a
# 256k vocabulary is ~8 MB -- bounded, while still vectorizing the common case
# where only a handful of requests carry a grammar.
_ROW_CHUNK = 32


def _apply_grammar_bitmask_torch(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    logits_indices: torch.Tensor,
    cu_num_logits: torch.Tensor,
    mask_stride: int,
) -> None:
    """Set disallowed logits to -inf, mirroring _apply_grammar_bitmask_kernel.

    ``bitmask`` is [num_masks, cdiv(vocab, 32)] packed int32, one row per
    (request, position) pair named by ``logits_indices`` as
    ``req_idx * mask_stride + position_idx``. A clear bit marks a token the
    grammar disallows.
    """
    num_masks = bitmask.shape[0]
    if num_masks == 0:
        return

    vocab_size = logits.shape[-1]
    device = logits.device

    mapping_idx = logits_indices[:num_masks].to(torch.int64)
    req_idx = mapping_idx // mask_stride
    position_idx = mapping_idx % mask_stride

    cu = cu_num_logits.to(torch.int64)
    row_start = cu[req_idx]
    num_req_logits = cu[req_idx + 1] - row_start
    logit_row = row_start + position_idx

    # Upstream masks out positions past a request's logit count instead of
    # returning, because the count is only known on device.
    active = position_idx < num_req_logits
    if not bool(active.any()):
        return

    active_rows = logit_row[active]
    active_masks = bitmask[: num_masks][active]

    bit_pos = torch.arange(32, device=device, dtype=torch.int32)
    for start in range(0, active_rows.shape[0], _ROW_CHUNK):
        rows = active_rows[start : start + _ROW_CHUNK]
        words = active_masks[start : start + _ROW_CHUNK]
        # [chunk, words, 32] -> [chunk, words * 32]; a clear bit means blocked.
        bits = (words.unsqueeze(-1) >> bit_pos) & 1
        blocked = (bits == 0).reshape(rows.shape[0], -1)
        blocked = blocked[:, :vocab_size]
        # Scatter -inf at the blocked positions rather than reading rows out,
        # masking and writing them back: two masks can name the same logit row
        # (the mapping is req_idx * mask_stride + position), and a
        # read-modify-write would keep only one of them. Writing -inf per
        # position is idempotent, which is what the upstream kernel relies on.
        selected = blocked.nonzero(as_tuple=False)
        if selected.numel() == 0:
            continue
        logits.index_put_(
            (rows[selected[:, 0]], selected[:, 1]),
            torch.tensor(float("-inf"), dtype=logits.dtype, device=device),
        )


def _build_apply(original_apply):
    """Wrap StructuredOutputsWorker.apply_grammar_bitmask with the torch path."""
    from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d
    from vllm.v1.worker.gpu.structured_outputs import _build_grammar_mapping

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch,
        grammar_req_ids: list[str],
        grammar_bitmask,
    ) -> None:
        if not grammar_req_ids:
            return

        # Asynchronously copy the bitmask to GPU.
        with torch.cuda.stream(self.copy_stream):
            bitmask = async_tensor_h2d(
                grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
            )

        mapping = _build_grammar_mapping(
            input_batch.req_ids,
            grammar_req_ids,
            input_batch.cu_num_logits_np,
            input_batch.num_draft_tokens_per_req,
            self.num_bonus_tokens,
            self.mask_stride,
        )

        with torch.cuda.stream(self.copy_stream):
            logits_indices = torch.tensor(
                mapping, dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
            )
            logits_indices = self.logits_indices[: len(mapping)].copy_(
                logits_indices, non_blocking=True
            )

        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.copy_stream)

        num_masks = bitmask.shape[0]
        assert num_masks == len(mapping)

        _apply_grammar_bitmask_torch(
            logits,
            bitmask[:num_masks],
            logits_indices,
            input_batch.cu_num_logits,
            self.mask_stride,
        )

        # Let the copy stream wait for the device tensors before reusing them.
        self.copy_stream.wait_stream(current_stream)

    return apply_grammar_bitmask


def apply_structured_outputs_compat() -> None:
    """Apply grammar bitmasks with torch instead of the FlagTree-hostile kernel."""
    try:
        import importlib

        structured_outputs = importlib.import_module(
            "vllm.v1.worker.gpu.structured_outputs"
        )

        worker_cls = getattr(structured_outputs, "StructuredOutputsWorker", None)
        if worker_cls is None:
            return

        current = getattr(worker_cls, "apply_grammar_bitmask", None)
        if current is None or getattr(current, _PATCH_FLAG, False):
            return

        replacement = _build_apply(current)
        setattr(replacement, _PATCH_FLAG, True)
        worker_cls.apply_grammar_bitmask = replacement
        logger.info(
            "kunlunxin structured_outputs_compat: applying grammar bitmasks with "
            "torch; FlagTree's XPU backend cannot lower the reshape in "
            "_apply_grammar_bitmask_kernel and miscompiles its bit unpacking"
        )
    except ImportError:
        # Builds without the V2 structured-output path have nothing to patch.
        return
