# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-scoped FlagGems dispatch policy registry.

The worker must not import a specific model integration to obtain its FlagGems
policy.  A model package registers a provider here; the worker resolves the
provider chain once, before the FlagGems runtime is reconfigured, and applies
the merged decision without mutating global policy.  Providers are responsible
for their own model scoping and validation, so resolving a policy for an
unrelated model is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class FlagGemsModelPolicy:
    """A validated, model-scoped FlagGems allow/deny decision.

    ``whitelist``/``blacklist`` replace the incoming values; ``None`` means the
    provider left that side unchanged.  ``skip_generic_aten`` requests that the
    generic ATen surface stay native while explicit FlagOS/OOT kernels remain
    enabled.  ``log_messages`` are emitted verbatim by the worker so all
    model-specific wording lives with the integration, not the worker.
    """

    whitelist: Optional[list[str]] = None
    blacklist: Optional[list[str]] = None
    skip_generic_aten: bool = False
    log_messages: tuple[str, ...] = field(default_factory=tuple)


# A provider receives the current (whitelist, blacklist) and returns either
# ``None`` (not applicable) or a policy to merge into the chain.
FlagGemsPolicyProvider = Callable[
    ..., Optional[FlagGemsModelPolicy]
]

_PROVIDERS: list[FlagGemsPolicyProvider] = []


def register_flag_gems_policy_provider(
    provider: FlagGemsPolicyProvider,
) -> FlagGemsPolicyProvider:
    """Register ``provider`` once; idempotent so plugin reloads are safe."""
    if provider not in _PROVIDERS:
        _PROVIDERS.append(provider)
    return provider


def unregister_flag_gems_policy_provider(
    provider: FlagGemsPolicyProvider,
) -> None:
    if provider in _PROVIDERS:
        _PROVIDERS.remove(provider)


def iter_flag_gems_policy_providers() -> tuple[FlagGemsPolicyProvider, ...]:
    return tuple(_PROVIDERS)


def resolve_flag_gems_policy(
    vllm_config,
    whitelist: Optional[list[str]],
    blacklist: Optional[list[str]],
    *,
    vendor_name: Optional[str] = None,
    providers: Optional[tuple[FlagGemsPolicyProvider, ...]] = None,
) -> FlagGemsModelPolicy:
    """Run the provider chain and return the merged model-scoped policy.

    Providers run in registration order and chain their allow/deny decisions;
    any provider may raise to reject an explicit user configuration that
    conflicts with the model contract.
    """
    chain = _PROVIDERS if providers is None else list(providers)
    skip_generic_aten = False
    messages: list[str] = []
    for provider in chain:
        result = provider(
            vllm_config,
            whitelist,
            blacklist,
            vendor_name=vendor_name,
        )
        if result is None:
            continue
        if result.whitelist is not None:
            whitelist = result.whitelist
        if result.blacklist is not None:
            blacklist = result.blacklist
        skip_generic_aten = skip_generic_aten or result.skip_generic_aten
        messages.extend(result.log_messages)
    return FlagGemsModelPolicy(
        whitelist=whitelist,
        blacklist=blacklist,
        skip_generic_aten=skip_generic_aten,
        log_messages=tuple(messages),
    )


__all__ = [
    "FlagGemsModelPolicy",
    "FlagGemsPolicyProvider",
    "register_flag_gems_policy_provider",
    "unregister_flag_gems_policy_provider",
    "iter_flag_gems_policy_providers",
    "resolve_flag_gems_policy",
]
