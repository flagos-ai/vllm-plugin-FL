# Copyright (c) 2026 BAAI. All rights reserved.
"""Close the two rope gaps the evas kernel cannot close itself.

Upstream ``RotaryEmbeddingFL.forward_oot`` hands the dispatch op an int64
position buffer (the kernel narrows with a 32-bit scalar load, so multi-token
calls rotate by the wrong angle) and a 3-D rope slice viewed out of a 2-D
buffer (the view keeps the parent's double-die descriptor, so it unzips one
rank short and the launch is rejected). Both are applied by wrapping the
module-global ``_rotary_embedding``; the upstream file stays byte-identical to
upstream/main.

"""

from __future__ import annotations

import torch

_PATCHED_ATTR = "_lpu_rope_operands_patched"


def _materialize(tensor: torch.Tensor) -> torch.Tensor:
    """A private ``split_dim=0`` copy with a descriptor matching its shape."""
    fresh = torch.ops.lpu.empty_with_format(
        list(tensor.shape), dtype=tensor.dtype, device=tensor.device, split_dim=0
    )
    fresh.copy_(tensor)
    return fresh


def _shard_parts(tensor: torch.Tensor):
    """The pair ``unzip`` builds from the tensor's descriptor, or ``None``.

    ``unzip`` sizes the parts from the tensor's own split dimension and
    rejects a split size of one, so a non-zero split dimension or a one-element
    leading dimension is "not this helper's business" rather than a failure.
    """
    try:
        if tensor.split_dim() != 0:
            return None
    except AttributeError:
        return None
    from ...local.double_die import unzip

    try:
        return unzip(tensor, 0)
    except RuntimeError:
        return None


def _repair_stale_shard_view(tensor: torch.Tensor) -> torch.Tensor:
    """Pass on a tensor whose shards carry the tensor's own rank.

    A tensor that has been flattened, split or reshaped can keep the double-die
    descriptor of the buffer it came from, and the unzip reads that descriptor
    rather than the logical shape.  Production hits it with a 3-D ``(T, 1, D)``
    rope slice whose parts come back 2-D, after which the launcher expands fewer
    positional arguments than the kernel declares and the Triton binder rejects
    the call.

    The offending tensor cannot be recognised by its attributes -- production
    reports no ``_base``, ``split_dim() == 0`` and contiguous -- so the test
    here is the physical part shape, the same comparison
    ``utils.double_die.unzip_logical`` makes for this class of bug.  A mismatch
    is repaired by allocating the shape fresh and copying into it, which
    restores the three-stride contract; a tensor whose parts already match is
    returned untouched, which is the ordinary case.
    """
    if tensor.dim() < 3 or tensor.shape[0] < 2:
        return tensor
    parts = _shard_parts(tensor)
    if parts is None:
        return tensor
    high = list(tensor.shape)
    high[0] = (tensor.shape[0] + 1) // 2
    low = list(tensor.shape)
    low[0] = tensor.shape[0] // 2
    if tuple(parts[0].shape) == tuple(high) and tuple(parts[1].shape) == tuple(low):
        return tensor
    return _materialize(tensor)


def _wrap_rotary(upstream):
    """Wrap the dispatch op with the position width and the operand repair.

    A repaired operand cannot be handed to the kernel with the caller's
    ``inplace`` flag, because the repair replaces it with a private copy and the
    result has to travel back into the caller's buffer by hand.  That rewrite
    forces a second rule: whenever the repair fires, the kernel is asked to
    write into its inputs, so an operand that needed no repair has to be made
    private too, or a caller that asked for fresh outputs would find one of its
    buffers overwritten.

    Nothing here closes a precondition beyond those two: if the kernel rejects
    the call, the exception propagates and the evas latch falls back to the
    pure twin, exactly as it would for a kernel that raised.
    """

    def rotary_embedding(
        obj,
        query,
        key,
        cos,
        sin,
        positions=None,
        rotary_interleaved=False,
        inplace=True,
    ):
        if positions is not None and positions.dtype != torch.int32:
            positions = positions.to(torch.int32)

        query_in = _repair_stale_shard_view(query)
        key_in = _repair_stale_shard_view(key)
        if query_in is query and key_in is key:
            return upstream(
                obj, query, key, cos, sin, positions, rotary_interleaved, inplace
            )

        if not inplace:
            # The call below forces inplace=True, so an operand that needed no
            # repair has to be private as well.
            if query_in is query:
                query_in = _materialize(query)
            if key_in is key:
                key_in = _materialize(key)
        query_out, key_out = upstream(
            obj, query_in, key_in, cos, sin, positions, rotary_interleaved, True
        )
        if inplace:
            # The kernel wrote into the private copies; the caller keeps using
            # its own views afterwards.
            query.copy_(query_out)
            key.copy_(key_out)
            return query, key
        return query_out, key_out

    return rotary_embedding


def apply_lpu_rotary_patch() -> None:
    """Wrap ``vllm_fl.ops.rotary_embedding._rotary_embedding`` on LPU."""
    import vllm_fl.ops.rotary_embedding as rope

    if getattr(rope, _PATCHED_ATTR, False):
        return
    rope._rotary_embedding = _wrap_rotary(rope._rotary_embedding)
    setattr(rope, _PATCHED_ATTR, True)
