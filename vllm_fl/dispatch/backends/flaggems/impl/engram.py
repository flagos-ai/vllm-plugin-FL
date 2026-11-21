# Copyright (c) 2026 BAAI. All rights reserved.
"""Optional public Engram contracts for the existing FlagOS dispatcher.

These adapters allocate no buffers, retain no state and perform no numerical
work. FGV validates dtype/layout/device and returns fresh outputs. Exceptions
from execution propagate; availability is resolved before execution.
"""

import inspect

_REQUIRED = {
    "engram_ngram_hash": {
        "history_ids",
        "token_map",
        "multipliers",
        "primes",
        "offsets",
        "pad_id",
    },
    "engram_embedding_int8": {
        "weight",
        "weight_scale",
        "hash_ids",
        "head_start",
        "local_heads",
        "vocab_start",
    },
    "engram_gated_add": {
        "hidden_states",
        "kv",
        "q_weight",
        "k_weight",
        "eps",
        "clamp_value",
        "token_mask",
        "gate_input",
        "rotation_block",
    },
}


def is_engram_available(op_name):
    try:
        import flaggems_vllm
    except ModuleNotFoundError as error:
        if error.name != "flaggems_vllm":
            raise
        return False
    implementation = getattr(flaggems_vllm, op_name, None)
    if not callable(implementation):
        return False
    try:
        return _REQUIRED[op_name].issubset(inspect.signature(implementation).parameters)
    except (TypeError, ValueError):
        return False


def engram_ngram_hash(history_ids, token_map, multipliers, primes, offsets, pad_id):
    """Raw [N,4] INT32/64 ids -> fresh INT32 [N,L,24] compressed n-gram hashes.

    Metadata is fixed, validated model layout: INT32 map, INT64 multipliers,
    positive primes and offsets, nonnegative products fitting INT64. Missing
    raw ids terminate the n-gram with compressed pad_id; inputs are unchanged.
    """
    from flaggems_vllm import engram_ngram_hash as implementation

    return implementation(history_ids, token_map, multipliers, primes, offsets, pad_id)


def engram_embedding_int8(
    weight, weight_scale, hash_ids, head_start, local_heads, vocab_start
):
    """INT8 [rows,D] + FP32 [rows,1] -> fresh BF16 [N,local_heads,D].

    INT32/64 hash_ids [N,24] may be strided; each rank owns complete consecutive
    head buckets. Non-owned/invalid ids write zero. Inputs are unchanged;
    Plugin owns staging lifetime. Table addressing supports INT64 flat offsets.
    """
    from flaggems_vllm import engram_embedding_int8 as implementation

    return implementation(
        weight, weight_scale, hash_ids, head_start, local_heads, vocab_start
    )


def engram_gated_add(
    hidden_states,
    kv,
    q_weight,
    k_weight,
    eps=1e-20,
    clamp_value=1e-6,
    token_mask=None,
    gate_input=None,
    rotation_block=None,
):
    """BF16 hidden [N,4,H], KV [N,5H], q/k [4,H] -> fresh BF16 [N,4,H].

    Per-HC norms, dot and gate use FP32; residual materializes once in BF16.
    BOOL [N] mask optionally zeros the gate; IEEE nonfinite value propagation
    is retained. eps is finite/positive, clamp finite/nonnegative. No mutation.
    """
    from flaggems_vllm import engram_gated_add as implementation

    return implementation(
        hidden_states,
        kv,
        q_weight,
        k_weight,
        eps=eps,
        clamp_value=clamp_value,
        token_mask=token_mask,
        gate_input=gate_input,
        rotation_block=rotation_block,
    )
