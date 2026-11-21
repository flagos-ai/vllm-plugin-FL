# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU-only Engram layout and tokenizer metadata for DeepSeek V4.1.

Adapted from vLLM e77daef89e18, common/engram.py; the corresponding CPU
algorithms are unchanged at 43067f6df60e. This module creates no device tensors.
"""

from dataclasses import dataclass

import numpy as np


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    odd = value - 1
    exponent = 0
    while odd % 2 == 0:
        odd //= 2
        exponent += 1
    for base in (2, 7, 61):
        witness = pow(base, odd, value)
        if witness in (1, value - 1):
            continue
        for _ in range(exponent - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def _next_prime(start: int, used: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in used:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    identifiers: dict[str, int] = {}
    lookup = []
    for token in range(len(tokenizer)):
        text = backend.decode([token], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token)
        else:
            key = normalizer.normalize_str(text) or text
        if key not in identifiers:
            identifiers[key] = len(identifiers)
        lookup.append(identifiers[key])
    return lookup, len(identifiers)


def compute_hash_multipliers(layer_ids, max_ngram_size, compressed_vocab_size):
    bound = max(1, ((2**63 - 1) // compressed_vocab_size) // 2)
    rows = []
    for layer in layer_ids:
        generator = np.random.default_rng(10007 * layer)
        values = generator.integers(0, bound, size=max_ngram_size, dtype=np.int64)
        rows.append(tuple(int(value) for value in values * 2 + 1))
    return tuple(rows)


@dataclass(frozen=True)
class EngramShard:
    head_start: int
    num_heads: int
    vocab_start: int
    vocab_end: int

    @property
    def rows(self) -> int:
        return self.vocab_end - self.vocab_start


class EngramLayout:
    """Immutable bucket constants; TP shards own whole consecutive hash heads."""

    def __init__(self, config):
        self.layer_ids = tuple(config.engram_layer_ids)
        self.num_embeddings = tuple(config.engram_num_embeddings)
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_heads
        self.head_dim = config.engram_head_dim
        self.compressed_vocab_size = config.engram_compressed_vocab_size
        self.pad_token_id = config.engram_pad_token_id
        if self.max_ngram_size != 4 or self.n_heads != 8 or self.head_dim != 256:
            raise NotImplementedError(
                "V4.1 Engram requires ngram=4, heads=8, head_dim=256"
            )
        if not self.layer_ids or len(self.layer_ids) != len(self.num_embeddings):
            raise ValueError(
                "Engram layer and table lists must be nonempty and aligned"
            )
        if len(set(self.layer_ids)) != len(self.layer_ids) or any(
            not isinstance(layer, int)
            or isinstance(layer, bool)
            or not 0 <= layer < config.num_hidden_layers
            for layer in self.layer_ids
        ):
            raise ValueError("Engram layer ids must be unique backbone layer indices")
        if not 1 < config.engram_vocab_size < 2**32 - 1024:
            raise ValueError("Engram prime buckets require a positive 32-bit base")
        if not 0 < self.compressed_vocab_size <= config.vocab_size:
            raise ValueError("Invalid Engram compressed vocabulary size")
        if not 0 <= self.pad_token_id < config.vocab_size:
            raise ValueError("Invalid Engram padding token")
        used: set[int] = set()
        primes, offsets = [], []
        for table_rows in self.num_embeddings:
            layer = []
            for _ in range(self.max_ngram_size - 1):
                sizes = []
                current = config.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = _next_prime(current, used)
                    used.add(current)
                    sizes.append(current)
                layer.append(tuple(sizes))
            heads = tuple(size for order in layer for size in order)
            if not isinstance(table_rows, int) or not sum(heads) <= table_rows < 2**31:
                raise ValueError(
                    "Engram table cannot contain its positive INT32 buckets"
                )
            prefix = [0]
            for size in heads[:-1]:
                prefix.append(prefix[-1] + size)
            primes.append(tuple(layer))
            offsets.append(tuple(prefix))
        self.primes = tuple(primes)
        self.offsets = tuple(offsets)
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads
        self.multipliers = compute_hash_multipliers(
            self.layer_ids, self.max_ngram_size, self.compressed_vocab_size
        )

    def shard(self, layer_index: int, tp_size: int, tp_rank: int) -> EngramShard:
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError("Invalid Engram TP rank or size")
        if tp_size > self.n_hash_cols:
            raise NotImplementedError(
                "Engram TP size cannot exceed the hash-head count"
            )
        if not 0 <= layer_index < len(self.layer_ids):
            raise ValueError("Invalid Engram layer index")
        sizes = tuple(size for order in self.primes[layer_index] for size in order)
        base, remainder = divmod(self.n_hash_cols, tp_size)
        count = base + int(tp_rank < remainder)
        first = tp_rank * base + min(tp_rank, remainder)
        return EngramShard(
            first, count, sum(sizes[:first]), sum(sizes[: first + count])
        )
