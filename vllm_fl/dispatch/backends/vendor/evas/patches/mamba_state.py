# Copyright (c) 2026 BAAI. All rights reserved.
"""Give a small mamba state its own backing store instead of the packed page.

A mamba page packs ``conv_state`` and ``ssm_state`` into one allocation, so
both states are windows over the per-layer page pool; on LPU a copy of any
strict sub-interval of such a window is staged as the whole backing, and the
GDN cache-row helpers pay that price on every row. This patch re-homes the
small state into a tightly packed buffer of its own, wrapping
``ModelRunnerFL._reshape_kv_cache_tensors`` so the reshape loop's offset
bookkeeping is untouched.

"""

from __future__ import annotations

import functools
import os

from vllm.logger import init_logger

logger = init_logger(__name__)

# A/B escape hatch: 0 keeps the upstream packed mamba page layout.
_ENV_MAMBA_STATE_UNPACKED = "VLLM_FL_MAMBA_STATE_UNPACKED"

# Marks the class attribute holding the pre-wrap method, which doubles as the
# idempotency flag.
_ORIG_ATTR = "_evas_orig_reshape_kv_cache_tensors"

# A re-home only pays if the state is currently being charged for backing it does
# not use: a copy of a window costs its whole storage, so the saving scales with
# backing/payload.  States that already own their backing sit at 1x and are left
# alone.  In the packed-pool plans this ratio IS the page share below (both are
# page_bytes / block_bytes), so the two cuts agree instead of one silently
# overriding the other.
_STATE_REHOME_RATIO = 4

# A state block must fit in this fraction of a page to be re-homed.  The gate is
# per BLOCK, not per state: the cost being removed is charged per block touch, so
# the state's own size is irrelevant and its block count must not steer the
# decision.  What it bounds is the memory: re-homing costs at most one quarter of
# the layer pool, whether the pool holds one layer (default plan) or every layer
# (``VLLM_FL_KV_CONFIG_PACKED=1``).  The states either side of the line are a
# conv block at 1.9 % of a page and a temporal block at 98 %, so the cut-off is
# not load-bearing.
_STATE_PAGE_SHARE = 4


def _mamba_state_unpacked() -> bool:
    return os.environ.get(_ENV_MAMBA_STATE_UNPACKED, "1") != "0"


def _mamba_state_own_storage(state, page_bytes: int):
    """Move a small mamba state out of the packed page into its own buffer.

    Returns ``state`` unchanged when it has to keep the page: a state whose
    block fills most of a page IS the page, and detaching it would allocate a
    comparable amount of memory again instead of saving anything.  That
    distinction is what separates ``conv_state`` (1.9 % of a page) from
    ``ssm_state`` (98 %), and it is also what bounds the cost on the packed-plan
    A/B arm (``VLLM_FL_KV_CONFIG_PACKED=1``, every layer in one pool): there the
    whole pool is duplicated only in proportion to the page fraction taken, not
    once per layer.
    """
    import torch

    from ..local.packed_cache import cache_footprint_bytes

    if state.dim() < 2:
        return state
    own_bytes = state.numel() * state.element_size()
    inner_shape = tuple(state.shape[1:])
    inner_strides = tuple(state.stride()[1:])
    inner = state.shape[1:].numel()
    block_bytes = inner * state.element_size()
    if own_bytes == 0 or block_bytes * _STATE_PAGE_SHARE > page_bytes:
        return state
    if cache_footprint_bytes(state) < _STATE_REHOME_RATIO * own_bytes:
        return state
    # The block has to BE one packed run: stride[0] can shrink from the page to
    # the block only if the block's own extent fills exactly ``inner`` elements.
    # A padded block (holes between its elements) would be reinterpreted.
    span = 1 + sum((size - 1) * stride
                   for size, stride in zip(inner_shape, inner_strides))
    if span != inner:
        return state
    buf = torch.empty(state.shape[0] * inner, dtype=state.dtype, device=state.device)
    own = buf.as_strided(state.shape, (inner, *inner_strides), 0)
    # The span check above makes the block one packed run, so ``own`` covers
    # every element of ``buf`` and nothing is left uninitialised after the copy.
    own.copy_(state)
    return own


def _rehome_mamba_states(runner, kv_caches: dict) -> None:
    """Walk the reshaped caches and swap in an own buffer where the gates pass.

    Iterates the runner's own attention groups, so layers that got no cache
    (``runner_only_attn_layers``) or a non-mamba spec are simply not seen.
    """
    from vllm.v1.kv_cache_interface import MambaSpec

    rehomed_states, rehomed_bytes = 0, 0
    for group in runner._kv_cache_spec_attn_group_iterator():
        kv_cache_spec = group.kv_cache_spec
        if not isinstance(kv_cache_spec, MambaSpec):
            continue
        for layer_name in group.layer_names:
            states = kv_caches.get(layer_name)
            if not isinstance(states, list):
                continue
            for i, state in enumerate(states):
                moved = _mamba_state_own_storage(state, kv_cache_spec.page_size_bytes)
                if moved is state:
                    continue
                states[i] = moved
                rehomed_states += 1
                rehomed_bytes += moved.numel() * moved.element_size()
    if rehomed_states:
        logger.info(
            "[mamba-state] %d state tensor(s) re-homed out of the packed "
            "page into +%.1f MiB of own buffers; row copies no longer "
            "stage the page pool (VLLM_FL_MAMBA_STATE_UNPACKED=0 reverts)",
            rehomed_states,
            rehomed_bytes / 2**20,
        )


def apply_mamba_state_unpacked() -> bool | None:
    """Wrap ``ModelRunnerFL._reshape_kv_cache_tensors`` (idempotent).

    Returns False when the method is gone: the registry then logs a declined
    install (and raises under VLLM_FL_STRICT_UPSTREAM_REBIND=1) instead of
    reporting the no-op as applied.
    """
    from vllm_fl.worker.model_runner import ModelRunnerFL

    if getattr(ModelRunnerFL, _ORIG_ATTR, None) is not None:
        return None
    orig = getattr(ModelRunnerFL, "_reshape_kv_cache_tensors", None)
    if orig is None:
        logger.warning(
            "mamba_state patch did not install: "
            "ModelRunnerFL._reshape_kv_cache_tensors is gone (FL refactor?) -- "
            "the packed-page row-copy cost is back"
        )
        return False

    @functools.wraps(orig)
    def _reshape_and_rehome(self, kv_cache_raw_tensors, kernel_block_sizes):
        kv_caches = orig(self, kv_cache_raw_tensors, kernel_block_sizes)
        if _mamba_state_unpacked():
            _rehome_mamba_states(self, kv_caches)
        return kv_caches

    setattr(ModelRunnerFL, _ORIG_ATTR, orig)
    ModelRunnerFL._reshape_kv_cache_tensors = _reshape_and_rehome
