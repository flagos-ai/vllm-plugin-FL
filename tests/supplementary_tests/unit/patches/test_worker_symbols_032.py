# Copyright 2025 BAAI. All rights reserved.
"""PR-#809 increments for ``vllm_fl.patches.worker_symbols_032``.

Coverage boundary (post-refactor): the common behaviour of the
``apply_worker_symbols_032`` installer (install-only-when-absent, caching),
the ``_is_residual_scattered_for_sp`` SP predicate, the
``_supports_xdrope`` / ``_SupportsXDRoPE`` probe, the
``_XDRotaryEmbedding`` position advance, the
``_reshape_attention_kv_cache`` layout matrix (contiguous / packed /
padded-pages / page-aligned-blocks / permuted) and the
``RoutedExperts*`` carriers is owned by PR #799's
``tests/supplementary_tests/unit/worker/test_worker_symbols_032.py``
(22 same-named tests). On merge, #799's copy moves to this
``unit/patches/`` location (matching the module-under-test domain); this
file then holds only the #809-specific increments that #799 does not
carry:

* ``apply_worker_symbols_032`` returning True (success report) on a
  compatible tree;
* the ``_dtype_size`` helper's element-size table;
* ``get_next_input_positions_tensor`` adapting its write to the caller's
  buffer dtype.

Every patched attribute is installed on module doubles under
``monkeypatch.setitem``-saved ``sys.modules`` entries, so the real vllm
import state is untouched. The ``apply_worker_symbols_032`` cache is
cleared before each installer assertion and restored afterwards.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.supplementary]


class _VllmTree:
    """Install/restore the fake ``vllm`` modules the patch targets."""

    def __init__(self, monkeypatch):
        self._monkeypatch = monkeypatch

        def put(name, **attrs):
            module = ModuleType(name)
            for key, value in attrs.items():
                setattr(module, key, value)
            monkeypatch.setitem(sys.modules, name, module)
            return module

        put("vllm")
        put("vllm.v1")
        self.worker_utils = put("vllm.v1.worker.utils")
        put("vllm.v1.worker")
        put("vllm.v1.worker.gpu")
        self.attn_utils = put("vllm.v1.worker.gpu.attn_utils")
        self.outputs = put("vllm.v1.outputs")
        put("vllm.model_executor")
        put("vllm.model_executor.layers")
        self.rotary_embedding = put("vllm.model_executor.layers.rotary_embedding")
        put("vllm.model_executor.models")
        self.interfaces = put("vllm.model_executor.models.interfaces")


@pytest.fixture
def vllm_tree(monkeypatch):
    """Install a fresh fake vllm tree; monkeypatch restores it on exit."""
    import vllm_fl.patches.worker_symbols_032 as ws

    ws.apply_worker_symbols_032.cache_clear()
    tree = _VllmTree(monkeypatch)
    yield SimpleNamespace(tree=tree, ws=ws)
    ws.apply_worker_symbols_032.cache_clear()


class TestApplyWorkerSymbolsIncrement:
    """#809-only: the installer's success return value."""

    def test_apply_returns_true(self, vllm_tree):
        """The installer reports success (True) on a compatible tree."""
        assert vllm_tree.ws.apply_worker_symbols_032() is True


class TestXDRoPEIncrement:
    """#809-only: dtype-adaptive position writes."""

    def test_next_input_positions_dtype_follows_output(self):
        """The write adapts to the caller's buffer dtype."""
        from vllm_fl.patches.worker_symbols_032 import _XDRotaryEmbedding

        buffer = np.zeros((1, 4), dtype=np.int32)
        _XDRotaryEmbedding.get_next_input_positions_tensor(
            buffer, out_offset=0, context_len=2, num_new_tokens=4
        )
        assert buffer[0].tolist() == [2, 3, 4, 5]
        assert buffer.dtype == np.int32


class TestReshapeAttentionKvCacheIncrement:
    """#809-only: the dtype element-size helper."""

    def test_dtype_size_helper(self):
        """_dtype_size reports the element size of a torch dtype."""
        from vllm_fl.patches.worker_symbols_032 import _dtype_size

        assert _dtype_size(torch.float32) == 4
        assert _dtype_size(torch.int32) == 4
        assert _dtype_size(torch.bfloat16) == 2
        assert _dtype_size(torch.uint8) == 1
