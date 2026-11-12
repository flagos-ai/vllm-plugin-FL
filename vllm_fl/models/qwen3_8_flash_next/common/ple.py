# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common Qwen3.8-Flash-Next PLE helpers."""

from dataclasses import dataclass
from functools import wraps

import torch


@dataclass(frozen=True)
class PLEShardOverlap:
    """Source and destination slices for one checkpoint embedding shard."""

    source_start: int
    destination_start: int
    row_count: int


def compute_ple_shard_overlap(
    *,
    checkpoint_start: int,
    checkpoint_rows: int,
    tp_start: int,
    tp_end: int,
) -> PLEShardOverlap | None:
    """Compute the overlap of a checkpoint shard and one TP vocabulary range."""

    if checkpoint_start < 0 or checkpoint_rows < 0:
        raise ValueError("checkpoint shard bounds must be non-negative")
    if tp_start < 0 or tp_end < tp_start:
        raise ValueError("invalid TP vocabulary range")
    checkpoint_end = checkpoint_start + checkpoint_rows
    overlap_start = max(checkpoint_start, tp_start)
    overlap_end = min(checkpoint_end, tp_end)
    if overlap_start >= overlap_end:
        return None
    return PLEShardOverlap(
        source_start=overlap_start - checkpoint_start,
        destination_start=overlap_start - tp_start,
        row_count=overlap_end - overlap_start,
    )


def validate_ple_embedding_shard(embedding, loaded_weight, shard_index, split_parts):
    """Validate one parsed checkpoint shard and return its global row start."""
    shard_size = (embedding.org_vocab_size + split_parts - 1) // split_parts
    checkpoint_start = shard_index * shard_size
    expected_rows = max(0, min(shard_size, embedding.org_vocab_size - checkpoint_start))
    expected_shape = (expected_rows, embedding.embedding_dim)
    if tuple(loaded_weight.shape) != expected_shape:
        raise ValueError(
            f"Shape mismatch for PLE embedding shard {shard_index}: "
            f"expected {expected_shape}, got {tuple(loaded_weight.shape)}"
        )
    return checkpoint_start


def copy_ple_embedding_shard_(
    destination: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    checkpoint_start: int,
    tp_start: int,
    tp_end: int,
) -> int:
    """Copy the overlapping rows of a PLE checkpoint shard into a TP table."""

    if destination.ndim == 0 or loaded_weight.ndim != destination.ndim:
        raise ValueError("destination and loaded weight must have matching ranks")
    if destination.shape[1:] != loaded_weight.shape[1:]:
        raise ValueError(
            "embedding shard dimensions do not match: "
            f"{tuple(destination.shape[1:])} != {tuple(loaded_weight.shape[1:])}"
        )
    if destination.shape[0] < tp_end - tp_start:
        raise ValueError("destination does not cover the requested TP range")
    overlap = compute_ple_shard_overlap(
        checkpoint_start=checkpoint_start,
        checkpoint_rows=loaded_weight.shape[0],
        tp_start=tp_start,
        tp_end=tp_end,
    )
    if overlap is None:
        return 0
    source = loaded_weight.narrow(0, overlap.source_start, overlap.row_count)
    target = destination.narrow(0, overlap.destination_start, overlap.row_count)
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))
    return overlap.row_count


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


_PLE_HASH_BUFFER_NAMES = (
    "layer_multipliers",
    "ngram_heads_vocab_sizes",
    "ngram_heads_offsets",
)


def initialize_ple_hash_buffers(module, config, layer_id, *, device=None):
    """Initialize the same checkpoint hash recipe for host and device tables."""
    bound = max(1, (((1 << 63) - 1) // module.unigram_vocab_size) // 2)
    seed = int(getattr(config, "seed", 1234)) + _PLE_LAYER_PRIME * layer_id
    multipliers = [
        2 * (_splitmix64(seed + _SPLITMIX_GAMMA * (i + 1)) % bound) + 1
        for i in range(module.ngram_size)
    ]
    sizes, offsets, total = [], [], 0
    for local_head in range(module.ngram_heads):
        global_head = layer_id * module.ngram_heads + local_head
        size = _nth_prime_after(int(config.ngram_vocab_size_base) - 1, global_head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    for name, values in zip(_PLE_HASH_BUFFER_NAMES, (multipliers, sizes, offsets)):
        module.register_buffer(
            name,
            torch.tensor(values, dtype=torch.int64, device=device),
            persistent=True,
        )
    divisor = int(config.make_ngram_vocab_size_divisible_by)
    return ((total + divisor - 1) // divisor) * divisor


def _release_uva_checkpoint_cache(model, owners):
    """Release this local checkpoint's file cache before MetaX UVA pinning."""
    import json
    import os
    from pathlib import Path

    import torch.distributed as dist

    lookup = next(
        (
            module._uva_lookup
            for module in owners
            if getattr(module, "_uva_lookup", None) is not None
        ),
        None,
    )
    source = getattr(getattr(model, "model_config", None), "model", None)
    if lookup is None or source is None:
        return
    root = Path(source).resolve()
    if not root.is_dir():
        return
    group = lookup.shared.group
    if group.world_size > 1:
        dist.barrier(group=group.cpu_group)
    status = [None]
    if group.rank_in_group == 0:
        try:
            index = root / "model.safetensors.index.json"
            if index.is_file():
                names = sorted(
                    set(json.loads(index.read_text())["weight_map"].values())
                )
            else:
                names = sorted(path.name for path in root.glob("*.safetensors"))
            if not names:
                raise ValueError("No safetensors checkpoint files found")
            for name in names:
                path = (root / name).resolve()
                if not path.is_relative_to(root) or path.suffix != ".safetensors":
                    raise ValueError(
                        "Checkpoint shard must remain inside the model directory"
                    )
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            from vllm.logger import init_logger

            init_logger(__name__).info(
                "Advised release of %d checkpoint shard caches before PLE UVA registration",
                len(names),
            )
        except Exception as error:  # noqa: BLE001
            # Broadcast initialization failures before any TP rank proceeds.
            status[0] = str(error)
    if group.world_size > 1:
        dist.broadcast_object_list(status, src=group.ranks[0], group=group.cpu_group)
    if status[0] is not None:
        raise RuntimeError("PLE checkpoint cache release failed: " + status[0])


def validate_ple_embedding_weights(load_weights):
    """Validate row coverage at the outermost model-loading boundary.

    AutoWeightsLoader groups *adjacent* module prefixes, so one PLE table can
    receive multiple calls as checkpoint files are streamed. Nested loaders
    participate in the same transaction; a direct embedding call is also a
    complete transaction. Only bookkeeping is retained, never checkpoint tensors.
    """

    @wraps(load_weights)
    def checked_load(model, weights, *args, **kwargs):
        owners = [
            module
            for module in model.modules()
            if (
                callable(getattr(type(module), "_required_embedding_shards", None))
                or getattr(module, "_uva_lookup", None) is not None
            )
            and getattr(module, "_ple_embedding_load_state", None) is None
        ]
        for module in owners:
            module._ple_embedding_load_state = {
                "shards": set(),
                "full": False,
                "buffers": set(),
            }
        try:
            loaded = load_weights(model, weights, *args, **kwargs)
            _release_uva_checkpoint_cache(model, owners)
            for module in owners:
                if getattr(module, "_uva_lookup", None) is not None:
                    module._finish_host_lookup_loading()
                    continue
                seen = module._ple_embedding_load_state["shards"]
                missing = sorted(module._required_embedding_shards() - seen)
                if seen and missing:
                    raise ValueError(
                        f"Missing PLE embedding shards for this TP rank: {missing}"
                    )
            return loaded
        finally:
            for module in owners:
                del module._ple_embedding_load_state

    return checked_load
