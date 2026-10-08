# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import torch

from vllm_fl.dispatch.backends.vendor.thead.register_ops import _lazy_attr


def test_lazy_attr_prepares_bound_method_before_fullgraph():
    owner = SimpleNamespace(prepare_query=lambda value: value * 2)
    owner_resolver = lambda: owner
    lazy = _lazy_attr(owner_resolver, "prepare_query", lambda: True)
    lazy._prepare()

    compiled = torch.compile(lazy, backend="eager", fullgraph=True)
    assert compiled(torch.tensor(3)).item() == 6
