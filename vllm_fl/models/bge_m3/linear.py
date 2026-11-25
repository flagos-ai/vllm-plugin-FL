# Copyright (c) 2026 BAAI. All rights reserved.
"""Resolve model-owned Linear policy once; leave global dispatch intact."""

from functools import partial

import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_fl.dispatch import resolve_op


def _native_head(layer, x, bias=None):
    return torch.nn.Linear.forward(layer, x)


def _head_forward(provider, layer, x):
    return provider(_native_head, layer, x, layer.bias)


def bind_linear_policy(model):
    """Freeze this model's provider; retain native methods and parameter objects."""
    provider = resolve_op("bge_m3_linear")
    for layer in model.modules():
        if isinstance(layer, LinearBase):
            method = layer.quant_method
            if type(method) is UnquantizedLinearMethod and not (
                isinstance(method.apply, partial) and method.apply.func is provider
            ):
                method.apply = partial(provider, method.apply)
        elif type(layer) is torch.nn.Linear:
            layer.forward = partial(_head_forward, provider, layer)
