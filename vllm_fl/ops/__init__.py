# Copyright (c) 2025 BAAI. All rights reserved.

"""FL operator submodules.

This package is imported from the plugin loading path (``vllm_fl.worker.worker``
imports ``vllm_fl.ops.custom_ops``), so submodules are resolved lazily: importing
``vllm_fl.ops`` must not eagerly pull in ``vllm.model_executor`` layers.  Attribute
access -- ``from vllm_fl.ops import activation`` or
``patch("vllm_fl.ops.activation._silu_and_mul")`` -- imports the submodule on demand.
"""

import importlib

__all__ = ["activation", "rotary_embedding"]


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
