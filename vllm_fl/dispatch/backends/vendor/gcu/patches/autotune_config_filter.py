# Copyright (c) 2026 BAAI. All rights reserved.

import copy
import logging

logger = logging.getLogger(__name__)

# Entry points in torch._inductor.runtime.triton_heuristics whose return
# value (a list of Triton configs) feeds the runtime autotuner. torch_gcu
# rebinds these to its GCUTritonConfigGenerator.
_ENTRY_POINTS = (
    "_reduction_configs",
    "_persistent_reduction_configs",
)


def _filter_configs(configs):
    """Drop candidates carrying kwargs the kernel signature cannot accept.

    A parameter absent from the Inductor-generated kernel signature is
    inapplicable launch metadata for that kernel, not a semantic input, so
    dropping it cannot change results — worst case the chosen launch
    config is not the vendor's intended optimum.
    """
    try:
        plain = [c for c in configs if not getattr(c, "kwargs", None)]
        if plain:
            return plain
        if configs:
            # Every candidate carries extras (e.g. SPLIT_K): synthesize a
            # plain one from the first candidate via a shallow copy so the
            # constructor signature of the vendor's Config class does not
            # matter.
            fallback = copy.copy(configs[0])
            fallback.kwargs = {}
            return [fallback]
    except Exception as exc:  # fail-soft: never block compilation setup
        logger.warning("GCU autotune config filter failed: %s", exc)
    return configs


def apply_autotune_config_filter_for_gcu():
    """Filter vendor autotune candidates the kernels cannot accept (#557).

    torch_gcu's GCUTritonConfigGenerator emits autotune candidates with
    parameters the Inductor-generated kernel signatures do not define
    (observed: ``SPLIT_K``), so the runtime triton autotuner dies at the
    first kernel launch with::

        KeyError: 'Keyword argument SPLIT_K was specified but unrecognised'
        (triton/runtime/autotuner.py -> jit.py _pack_args)

    Wrap the ``*_configs`` entry points (whatever they are bound to at
    apply time — torch_gcu rebinds them to its generator at import) and
    keep only candidates without extra kwargs.

    TODO: remove once torch_gcu emits candidates compatible with this
    torch's kernel signatures (#557 layer 4).
    """
    try:
        import torch._inductor.runtime.triton_heuristics as _th

        if getattr(_th, "_gcu_config_filter_patched", False):
            return

        wrapped = 0
        for name in _ENTRY_POINTS:
            current = getattr(_th, name, None)
            if current is None or getattr(current, "_gcu_config_filter", False):
                continue

            def _filtered(*args, __current=current, **kwargs):
                return _filter_configs(__current(*args, **kwargs))

            _filtered._gcu_config_filter = True
            setattr(_th, name, _filtered)
            wrapped += 1

        _th._gcu_config_filter_patched = True
        logger.info(
            "Patched %d triton_heuristics config entry points for GCU "
            "(drop candidates with kernel-unknown kwargs, #557 layer 4)",
            wrapped,
        )
    except Exception as e:
        logger.warning("Failed to patch triton_heuristics for GCU: %s", e)
