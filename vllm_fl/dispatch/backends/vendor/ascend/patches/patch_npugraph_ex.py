# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/patch/worker/patch_npugraph_ex_triton.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
npugraph_ex ValuePack patch for Triton scenarios.

This module patches npugraph_ex/torchair internals so that ValuePack inputs are
unpacked correctly when Triton kernels are present in the captured graph.  It is
applied at worker startup only when npugraph_ex is available.
"""

import importlib
import logging
import sys

import torch
from torch._subclasses.fake_tensor import FakeTensor

logger = logging.getLogger(__name__)

_PATCHED = False


def patch_npugraph_ex() -> None:
    """Apply npugraph_ex/torchair ValuePack patches."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    try:
        import npugraph_ex as nge
        from npugraph_ex.core._concrete_graph import _is_symlist
        from npugraph_ex.npu_fx_compiler import _unpack_meta_list
        _USE_NPUGRAPH_EX = True
    except ImportError:
        try:
            import torchair as nge
            from torchair.core._concrete_graph import _is_symlist
            from torchair.npu_fx_compiler import _unpack_meta_list
            _USE_NPUGRAPH_EX = False
        except ImportError:
            logger.info("npugraph_ex/torchair not available; skipping patch")
            return

    class ValuePack:
        def __init__(self, meta, npu_meta=None) -> None:
            self._meta = meta
            self._npu_meta = meta if npu_meta is None else npu_meta

        @property
        def meta(self):
            return self._meta

        @property
        def npu(self):
            return self._npu_meta

        def __getitem__(self, key):
            if isinstance(self._meta, dict):
                return self._meta.get(key)
            raise ValueError(
                f"Unsupported meta type for ValuePack __getitem__, "
                f"key:{key}, type: {type(self._meta)}")

        def __repr__(self) -> str:
            if isinstance(self._meta, FakeTensor):
                meta_str = f"FakeTensor(dtype={self._meta.dtype}, size={list(self._meta.size())}"
            elif isinstance(self._meta, torch.Tensor):
                meta_str = f"torch.Tensor(dtype={self._meta.dtype}, size={list(self._meta.size())}"
            elif isinstance(self._meta, torch.SymInt):
                meta_str = f"torch.SymInt({self._meta})"
            else:
                try:
                    meta_str = f"{type(self._meta)}({self._meta})"
                except Exception:
                    meta_str = f"{type(self._meta)}"
            return f"Pack(meta:{meta_str} npu:{self._npu_meta})"

    def _unpack_meta(args, kwargs):
        unpacked_args = []
        unpacked_kwargs = {}

        def _get_meta_part(arg):
            if isinstance(arg, (list, tuple)) and any(
                    isinstance(v, ValuePack) for v in arg):
                return _unpack_meta_list(arg)
            elif isinstance(arg, dict):
                return {
                    k: v.meta if isinstance(v, ValuePack) else v
                    for k, v in arg.items()
                }
            elif isinstance(arg, ValuePack):
                return arg.meta
            else:
                return arg

        for arg in args:
            unpacked_args.append(_get_meta_part(arg))

        for key, value in kwargs.items():
            unpacked_kwargs[key] = _get_meta_part(value)

        return list(unpacked_args), unpacked_kwargs

    def _unpack_npu(self, args, kwargs):
        unpacked = []
        unpacked_kwargs = {}

        def _get_npu_part(arg):
            if isinstance(arg, (list, tuple)) and len(arg):
                if _is_symlist(arg):
                    arg = self._graph.parse_symlist(arg)
                else:
                    arg = [(v.npu if isinstance(v, ValuePack) else v)
                           for v in arg]
                return arg
            elif isinstance(arg, dict):
                return {
                    k: v.npu if isinstance(v, ValuePack) else v
                    for k, v in arg.items()
                }
            elif isinstance(arg, ValuePack):
                return arg.npu
            else:
                return arg

        for arg in args:
            unpacked.append(_get_npu_part(arg))

        for key, value in kwargs.items():
            unpacked_kwargs[key] = _get_npu_part(value)

        return unpacked, unpacked_kwargs

    nge.core._concrete_graph.ValuePack = ValuePack
    # The ValuePack class is referenced in the npu_fx_compiler module (and
    # fx_summary for torchair), and after the patch these modules need to be
    # reloaded.
    if not _USE_NPUGRAPH_EX:
        importlib.reload(sys.modules["torchair.fx_summary"])
    pkg_prefix = "npugraph_ex" if _USE_NPUGRAPH_EX else "torchair"
    importlib.reload(sys.modules[f"{pkg_prefix}.npu_fx_compiler"])
    nge.npu_fx_compiler._unpack_meta = _unpack_meta
    nge.npu_fx_compiler._NpuGraphConverter._unpack_npu = _unpack_npu
    logger.info("Patched npugraph_ex/torchair ValuePack handling")
