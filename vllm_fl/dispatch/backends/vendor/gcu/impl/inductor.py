# Copyright (c) 2026 BAAI. All rights reserved.

"""Keep TorchGCU reduction configuration compatible with Torch's metadata ABI."""

import inspect
from collections.abc import Mapping
from functools import wraps

_PARAMETERS = ("size_hints", "reduction_hint", "inductor_meta", "triton_meta")
_ADAPTER_MARKER = "_fl_gcu_persistent_reduction_adapter"
_REGISTRATION_MARKER = "_fl_gcu_persistent_reduction_registration"


def _configuration_arity(function) -> int:
    parameters = tuple(inspect.signature(function).parameters.values())
    if (
        len(parameters) not in (3, 4)
        or tuple(parameter.name for parameter in parameters)
        != _PARAMETERS[: len(parameters)]
        or any(
            parameter.kind
            not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for parameter in parameters
        )
        or parameters[0].default is not inspect.Parameter.empty
        or parameters[1].default is not False
        or parameters[2].default is not None
        or (len(parameters) == 4 and parameters[3].default is not None)
    ):
        raise RuntimeError("Unknown TorchGCU persistent reduction configuration ABI")
    return len(parameters)


def _adapt_registered_persistent_reduction(provider, heuristics) -> str:
    original = heuristics._persistent_reduction_configs
    if getattr(original, _ADAPTER_MARKER, False) is True:
        return "already_adapted"
    generator_type = provider.GCUTritonConfigGenerator
    if (
        not inspect.ismethod(original)
        or not isinstance(original.__self__, generator_type)
        or original.__func__ is not generator_type.persistent_reduction_configs
    ):
        # Before TorchGCU's first lazy registration this is Torch's own
        # function. Other providers are not ours to replace either.
        return "unchanged_other_provider"
    if _configuration_arity(original) == 4:
        return "unchanged_compatible_gcu"

    # Do not use wraps here: its __wrapped__ would advertise the old three-
    # argument signature to callers inspecting this new four-argument ABI.
    def persistent_reduction_configs(
        size_hints, reduction_hint=False, inductor_meta=None, triton_meta=None
    ):
        if triton_meta is not None:
            if not isinstance(triton_meta, Mapping):
                raise TypeError(
                    "TorchGCU persistent reduction triton_meta must be a mapping"
                )
            if triton_meta.get("native_matmul"):
                raise NotImplementedError(
                    "TorchGCU's three-argument persistent reduction provider "
                    "does not support native_matmul metadata"
                )
        # Ordinary reduction metadata has no other use in the observed Torch
        # implementation. Preserve all three original objects and vendor
        # configuration results; this adapter changes no kernel algorithm.
        return original(size_hints, reduction_hint, inductor_meta)

    setattr(persistent_reduction_configs, _ADAPTER_MARKER, True)
    persistent_reduction_configs._fl_gcu_original = original
    heuristics._persistent_reduction_configs = persistent_reduction_configs
    return "adapted"


def _install_gcu_inductor_hooks(provider, heuristics) -> None:
    patcher_type = provider.GCUInductorPatcher
    register = patcher_type.patch_triton_heuristics
    if getattr(register, _REGISTRATION_MARKER, False) is not True:

        @wraps(register)
        def register_compatible(self, *args, **kwargs):
            # TorchGCU registers heuristics lazily at its first torch.compile.
            # Installing only the current slot would be overwritten then.
            result = register(self, *args, **kwargs)
            _adapt_registered_persistent_reduction(provider, heuristics)
            return result

        setattr(register_compatible, _REGISTRATION_MARKER, True)
        patcher_type.patch_triton_heuristics = register_compatible
    # Also handle a process where TorchGCU has already registered its provider.
    _adapt_registered_persistent_reduction(provider, heuristics)


def apply_inductor_gcu_patch() -> None:
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "enflame":
        return
    from torch._inductor.runtime import triton_heuristics
    from torch_gcu.gcu.inductor import backend_register

    _install_gcu_inductor_hooks(backend_register, triton_heuristics)
