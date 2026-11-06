# SPDX-License-Identifier: Apache-2.0
"""Common-policy causal convolution bindings."""
from .indexer_backend import INDEXER_BACKEND
from .provider import use_nvidia_reference


def _call(name, *args, **kwargs):
    native = None
    if use_nvidia_reference():
        try:
            from vllm.model_executor.layers.mamba.ops import causal_conv1d

            native = getattr(causal_conv1d, name)
        except (ImportError, AttributeError, OSError):
            # Missing optional ABI is resolved before any convolution/state write.
            native = None
    return INDEXER_BACKEND._call_flag(
        name, INDEXER_BACKEND._flag(name, name), None, *args, native=native, **kwargs
    )


def causal_conv1d_fn(*args, **kwargs):
    return _call("causal_conv1d_fn", *args, **kwargs)


def causal_conv1d_update(*args, **kwargs):
    return _call("causal_conv1d_update", *args, **kwargs)
