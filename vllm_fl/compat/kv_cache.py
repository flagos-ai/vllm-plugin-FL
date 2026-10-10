# SPDX-License-Identifier: Apache-2.0
"""Complete KV-cache owner binding when the upstream helper omits its hook."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, MutableMapping

import torch

__all__ = [
    "bind_kv_cache",
    "is_registered_owner",
    "register_kv_cache_owner",
    "upstream_binds_kv_cache_owners",
]


def _probe_upstream_hook() -> bool:
    """Observe whether the upstream helper calls a synthetic layer's hook."""
    from vllm.v1.worker.utils import bind_kv_cache as upstream_bind_kv_cache

    class _Probe:
        def __init__(self) -> None:
            self.bound = False

        def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
            self.bound = True

    probe = _Probe()
    layer_name = "model.layers.0.attn"
    upstream_bind_kv_cache(
        {layer_name: torch.zeros(1)},
        {layer_name: probe},
        [],
        1,
    )
    return probe.bound


@lru_cache(maxsize=1)
def upstream_binds_kv_cache_owners() -> bool:
    """Probe the installed helper once, without interpreting its version."""
    return _probe_upstream_hook()


_REGISTERED_OWNER_TYPES: set[type] = set()


def register_kv_cache_owner(cls: type) -> type:
    """Register a layer class whose instances need a post-bind hook.

    The class must expose ``bind_kv_cache(tensor)``.  Intended to be called at
    module import by the model that owns the cache, so the common runner never
    imports model-specific code.
    """
    if not callable(getattr(cls, "bind_kv_cache", None)):
        raise TypeError(
            f"{cls!r} cannot be a KV-cache owner: it has no callable "
            "bind_kv_cache method"
        )
    _REGISTERED_OWNER_TYPES.add(cls)
    return cls


def is_registered_owner(layer: Any) -> bool:
    return any(isinstance(layer, owner_type) for owner_type in _REGISTERED_OWNER_TYPES)


def _collect_owners(
    forward_context: MutableMapping[str, Any],
    kv_caches: MutableMapping[str, Any],
) -> list[tuple[str, Any, Any]]:
    owners = []
    for layer_name, kv_cache in kv_caches.items():
        layer = forward_context.get(layer_name)
        if layer is not None and is_registered_owner(layer):
            owners.append((layer_name, layer, kv_cache))
    return owners


def _preflight_owners(owners: list[tuple[str, Any, Any]]) -> None:
    """Validate every owner before anything is mutated.

    Raises before ``runner_kv_caches`` or any ``layer.kv_cache`` is touched, so
    an unsupported dtype/layout/shape cannot leave a half-bound cache.
    """
    for layer_name, layer, kv_cache in owners:
        validate = getattr(layer, "validate_kv_cache", None)
        if callable(validate):
            validate(kv_cache)


def bind_kv_cache(
    kv_caches: MutableMapping[str, Any],
    forward_context: MutableMapping[str, Any],
    runner_kv_caches: list,
    num_attn_module: int = 1,
) -> None:
    """Bind the allocated KV cache to the runner and to cache owners.

    Order: validate owners -> probe upstream behavior -> upstream bind ->
    missing owner hooks. Preflight failures leave state untouched.
    A failure during binding aborts runner initialization: the partially bound
    runner must be discarded, not retried. Hooks must not mutate cache contents.
    """
    from vllm.v1.worker.utils import bind_kv_cache as upstream_bind_kv_cache

    owners = _collect_owners(forward_context, kv_caches)
    # Validate before any mutation at all.
    _preflight_owners(owners)
    calls_owner_hook = upstream_binds_kv_cache_owners() if owners else True

    upstream_bind_kv_cache(
        kv_caches, forward_context, runner_kv_caches, num_attn_module
    )
    if not calls_owner_hook:
        for _layer_name, layer, kv_cache in owners:
            layer.bind_kv_cache(kv_cache)
