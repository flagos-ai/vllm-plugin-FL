# SPDX-License-Identifier: Apache-2.0
"""Framework-owned decode metadata buffers and capture preparation."""

from __future__ import annotations

_FULL_WARMED_DESCRIPTORS = set()


def _full_decode_descriptor(counts, seq_lens, padded_tokens, device):
    if type(padded_tokens) is not int or not 0 <= padded_tokens <= 1024:
        return None
    if (
        not getattr(counts, "is_cuda", False)
        or not getattr(seq_lens, "is_cuda", False)
        or counts.device != device
        or seq_lens.device != device
    ):
        return None
    return (
        str(counts.device),
        tuple(counts.shape),
        tuple(counts.stride()),
        str(counts.dtype),
        tuple(seq_lens.shape),
        tuple(seq_lens.stride()),
        str(seq_lens.dtype),
        padded_tokens,
    )


def full_decode_metadata_available(counts, seq_lens, padded_tokens, device):
    """Whether native startup completed this exact metadata prewarm descriptor.

    This is a compile/launch capability, not a tensor-value or quality check.
    It does not retain input tensors or graph-owned output buffers. A stock
    native runner never publishes descriptors and retains its original chain.
    """
    key = _full_decode_descriptor(counts, seq_lens, padded_tokens, device)
    return key is not None and key in _FULL_WARMED_DESCRIPTORS


class DecodeMetadataWorkspace:
    def __init__(self, capacity, device):
        import torch

        from .runtime_decode_indexer_metadata import decode_indexer_metadata

        if type(capacity) is not int or capacity < 0:
            raise ValueError(
                "decode metadata workspace capacity must be a nonnegative integer"
            )
        self.torch, self.dispatch = torch, decode_indexer_metadata
        self.rows = torch.empty((capacity,), dtype=torch.int64, device=device)
        self.lengths_i64 = torch.empty((capacity,), dtype=torch.int64, device=device)
        self.lengths_i32 = torch.empty((capacity,), dtype=torch.int32, device=device)

    def prepare(self, counts, seq_lens, n):
        lengths = (
            self.lengths_i32
            if seq_lens.ndim == 2
            and seq_lens.shape[1] > 1
            and seq_lens.dtype == self.torch.int32
            else self.lengths_i64
        )
        return self.dispatch(counts, seq_lens, self.rows, lengths, n)


def make_decode_metadata_workspace(capacity, device):
    return DecodeMetadataWorkspace(capacity, device)


def prepare_full_decode_metadata(counts, seq_lens, padded_tokens, device):
    """Graph-local fixed-shape outputs; no state shared across graph instances.

    Warm this exact FULL branch outside capture first. CUDA graph allocation
    owns these intermediate buffers; replay reuses their captured addresses.
    The caller retains current GPU counts/seq inputs and all cache/slot masks.
    """
    workspace = DecodeMetadataWorkspace(padded_tokens, device)
    lengths = (
        workspace.lengths_i32
        if seq_lens.ndim == 2
        and seq_lens.shape[1] > 1
        and seq_lens.dtype == workspace.torch.int32
        else workspace.lengths_i64
    )
    return workspace.dispatch(
        counts, seq_lens, workspace.rows, lengths, padded_tokens, padded=True
    )


def warm_full_decode_metadata_before_capture(attn_metadata, device):
    """Warm exact FULL metadata descriptors before the native model capture.

    Native graph_capture's stream context is not an active CUDA capture yet.
    Use a separate stream, wait for metadata H2D/preparation, then finish every
    warm launch before returning to the unchanged native model wrapper.
    This function is startup-only; it is absent from model replay.
    """
    import torch

    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FULL metadata prewarm must precede CUDA capture")
    groups = [attn_metadata] if isinstance(attn_metadata, dict) else attn_metadata
    if not isinstance(groups, (list, tuple)) or any(
        not isinstance(group, dict) for group in groups
    ):
        raise TypeError("FULL metadata prewarm requires native per-layer metadata")
    unique, seen = [], set()
    for group in groups:
        for meta in group.values():
            if not isinstance(meta, DeepseekV32IndexerMetadata) or id(meta) in seen:
                continue
            seen.add(id(meta))
            if meta.num_prefills:
                raise ValueError("FULL INT8 metadata prewarm requires pure decode")
            if meta.num_decode_tokens:
                unique.append(meta)
    if not unique:
        raise ValueError("FULL INT8 metadata prewarm found no decode metadata")
    warmed = []
    prep_stream = torch.cuda.current_stream(device)
    warm_stream = torch.cuda.Stream(device=device)
    warm_stream.wait_stream(prep_stream)
    with torch.cuda.stream(warm_stream):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FULL metadata warm stream must not be capturing")
        try:
            for meta in unique:
                prepare_full_decode_metadata(
                    meta.decode.decode_lens,
                    meta.decode.seq_lens,
                    meta.num_decode_tokens,
                    device,
                )
                key = _full_decode_descriptor(
                    meta.decode.decode_lens,
                    meta.decode.seq_lens,
                    meta.num_decode_tokens,
                    device,
                )
                if key is None:
                    raise ValueError("invalid FULL metadata warm descriptor")
                warmed.append(key)
        finally:
            # Match native warmup ordering, including auxiliary/autotuner work.
            # Preserve failures; never enter model capture after warm failure.
            torch.cuda.synchronize(device)
    _FULL_WARMED_DESCRIPTORS.update(warmed)
    return len(unique)
