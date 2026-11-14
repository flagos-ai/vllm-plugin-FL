# Copyright (c) 2026 BAAI. All rights reserved.

"""Bind the FlagGems-vLLM Ascend ops to the upstream V2 sampler."""

import importlib

_PENALTIES_MODULE = "vllm.v1.worker.gpu.sample.penalties"
_GUMBEL_MODULE = "vllm.v1.worker.gpu.sample.gumbel"
_STATES_MODULE = "vllm.v1.worker.gpu.sample.states"
_GUMBEL_EXPORTS = ("murmur3_hash32", "murmur3_uniform32", "murmur3_uniform64")


def _provider_callable(module, name):
    export = getattr(module, name, None)
    if not callable(export):
        raise RuntimeError(
            f"Ascend V2 sampling requires a FlagGems-vLLM build exporting "
            f"{module.__name__}.{name}."
        )
    return export


def install_ascend_v2_sampling_shims() -> bool:
    """Install the supported Ascend bindings as one transaction."""
    from vllm.platforms import current_platform

    if current_platform.device_type != "npu":
        return False

    try:
        provider = importlib.import_module("flaggems_vllm")
    except ModuleNotFoundError as error:
        if error.name != "flaggems_vllm":
            raise
        raise RuntimeError("Ascend V2 sampling requires FlagGems-vLLM.") from error
    apply_penalties = _provider_callable(provider, "v2_apply_penalties")
    apply_temperature = _provider_callable(provider, "v2_apply_temperature")
    rng = getattr(provider, "v2_gumbel_rng", None)
    if rng is None:
        raise RuntimeError(
            "Ascend V2 sampling requires a FlagGems-vLLM build exporting "
            "flaggems_vllm.v2_gumbel_rng."
        )
    rng_updates = [(name, _provider_callable(rng, name)) for name in _GUMBEL_EXPORTS]

    penalties = importlib.import_module(_PENALTIES_MODULE)
    gumbel = importlib.import_module(_GUMBEL_MODULE)
    states = importlib.import_module(_STATES_MODULE)
    # SamplingStates imports apply_temperature directly. The upstream JIT
    # callers resolve the RNG helpers at compilation, so previously imported
    # argmax and sampling entry points keep using the upstream algorithms.
    updates = (
        (penalties, "apply_penalties", apply_penalties),
        (gumbel, "apply_temperature", apply_temperature),
        (states, "apply_temperature", apply_temperature),
        *((gumbel, name, function) for name, function in rng_updates),
    )
    bindings = [
        (module, name, getattr(module, name), replacement)
        for module, name, replacement in updates
    ]
    applied = []
    try:
        for module, name, original, replacement in bindings:
            if original is not replacement:
                applied.append((module, name, original))
                setattr(module, name, replacement)
    except Exception:
        for module, name, original in reversed(applied):
            if getattr(module, name) is not original:
                setattr(module, name, original)
        raise
    return True
