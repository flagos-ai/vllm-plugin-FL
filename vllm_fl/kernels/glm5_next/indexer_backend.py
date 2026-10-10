# SPDX-License-Identifier: Apache-2.0
"""Platform dispatch for the GLM5-Next sparse indexer operator chain.

Native vLLM/DeepGEMM and public FlagGems-vllm entries are candidates when their
capabilities are available. The common policy selects before execution.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import cached_property

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

from . import kpool_compress as metadata
from .provider import use_nvidia_reference

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def _load_flaggems_op(module: str, name: str) -> Callable | None:
    """Load only the operator library's public, registered host API."""
    aliases = {"fp8_fp4_paged_mqa_logits": "paged_mqa_logits"}
    try:
        return getattr(
            importlib.import_module("flaggems_vllm"), aliases.get(name, name)
        )
    except (ImportError, AttributeError, OSError):
        return None


class Glm5NextIndexerBackend:
    """GLM implementations selected by the published common dispatch policy."""

    def __init__(self) -> None:
        from vllm_fl.dispatch.manager import OpManager

        self.is_nvidia = use_nvidia_reference()
        self._flag_ops = {}
        self._native_ops = {}
        self._bindings = {}
        self._manager = OpManager()
        self.name = "common-policy"

    def _native(self, module_name, name):
        key = (module_name, name)
        if key not in self._native_ops:
            fn = None
            if self.is_nvidia:
                try:
                    module = (
                        torch.ops._C
                        if module_name == "torch.ops._C"
                        else importlib.import_module(module_name)
                    )
                    candidate = getattr(module, name)
                    if callable(candidate):
                        fn = candidate
                except (ImportError, AttributeError, OSError):
                    pass
            self._native_ops[key] = fn
        return self._native_ops[key]

    def _flag(self, module, name):
        from vllm_fl.utils import use_flaggems_op

        if not use_flaggems_op(name):
            return None
        if name not in self._flag_ops:
            self._flag_ops[name] = _load_flaggems_op(module, name)
        return self._flag_ops[name]

    def _call_flag(
        self,
        name,
        fn,
        fallback,
        *args,
        native=None,
        native_available=None,
        graph=False,
        _preflight=False,
        **kwargs,
    ):
        from vllm_fl.dispatch.binding import OperatorBinding
        from vllm_fl.dispatch.types import BackendImplKind, OpImpl
        from vllm_fl.utils import use_flaggems_op

        if name not in self._bindings:
            implementations = []
            if fn is not None:

                def flag(*a, **k):
                    if name == "per_token_group_quant_fp8":
                        k["scale_ue8m0"] = k.pop("use_ue8m0", False)
                    return fn(*a, **k)

                def available():
                    capability = getattr(fn, "_is_available", None)
                    return use_flaggems_op(name) and (
                        capability is None or bool(capability())
                    )

                flag._is_available = available
                implementations.append(
                    OpImpl(name, "glm5.flaggems", BackendImplKind.DEFAULT, flag)
                )
            if native is not None:
                native._is_available = native_available or (lambda: self.is_nvidia)
                implementations.append(
                    OpImpl(
                        name, "glm5.cuda", BackendImplKind.VENDOR, native, vendor="cuda"
                    )
                )
            if fallback is not None:
                implementations.append(
                    OpImpl(name, "glm5.torch", BackendImplKind.REFERENCE, fallback)
                )
            self._manager.registry.register_many(implementations)
            self._manager.bump_policy_epoch()
            self._bindings[name] = OperatorBinding(
                self._manager,
                name,
                graph_capabilities={
                    "glm5.flaggems": graph,
                    "glm5.cuda": None,
                    "glm5.torch": False,
                },
            )
        if _preflight:
            return self._bindings[name].describe()
        return self._bindings[name](*args, **kwargs)

    def _run(self, name, module, fallback, native_module, native_name, *args, **kwargs):
        if name in self._bindings:
            if kwargs.pop("_preflight", False):
                return self._bindings[name].describe()
            return self._bindings[name](*args, **kwargs)
        fn = self._flag(module, name)
        native_fn = self._native(native_module, native_name)

        def native(*a, **k):
            if name == "pack_seq_triton":
                k.pop("max_length", None)
            if name == "unpack_seq_triton":
                k.pop("total_tokens", None)
            return native_fn(*a, **k)

        return self._call_flag(
            name,
            fn,
            fallback,
            *args,
            native=native if native_fn is not None else None,
            graph=(name == "fp8_fp4_paged_mqa_logits" and current_platform.is_cuda()),
            **kwargs,
        )

    def preflight(self):
        descriptions = []
        for method in (
            self.per_token_group_quant_fp8,
            self.indexer_k_quant_and_cache,
            self.cp_gather_indexer_k_quant_cache,
            self.mqa_logits,
            self.paged_mqa_logits,
        ):
            descriptions.append(method(_preflight=True))
        descriptions.append(self.topk_prefill(*([None] * 8), _preflight=True))
        descriptions.append(self.topk_decode(*([None] * 8), _preflight=True))
        descriptions.append(self.pack_seq(None, None, _preflight=True))
        descriptions.append(self.unpack_seq(None, None, _preflight=True))
        for method in (
            self.fwht128_quant_fp8,
            self.kpool_compress_and_write_cache,
            self.kpool_decode_update_and_maybe_write_cache_batched,
            self.kpool_seed_tail_cache,
            self.expand_pools_to_tokens,
            self.append_tail_to_topk,
            self.expand_pools_and_append_tail,
        ):
            descriptions.append(method(_preflight=True))
        # Resolve required public entries during model initialization, before
        # state/cache mutation or CUDA graph capture can begin.
        for name in (
            "safe_kda_gate",
            "fused_recurrent_kda",
            "chunk_kda_with_safe_gate",
            "gather_state_rows",
            "scatter_state_rows",
            "zero_state_rows",
            "copy_to",
            "gather_rows",
            "scatter_decode_tokens",
        ):
            descriptions.append(self._kpool(name, _preflight=True))
        from .causal_conv import causal_conv1d_fn, causal_conv1d_update

        for method in (causal_conv1d_fn, causal_conv1d_update):
            descriptions.append(method(_preflight=True))
        return descriptions

    @cached_property
    def has_nvidia_deep_gemm(self) -> bool:
        if not self.is_nvidia:
            return False
        from vllm.utils.deep_gemm import has_deep_gemm

        return has_deep_gemm()

    def per_token_group_quant_fp8(self, *args, **kwargs):
        return self._run(
            "per_token_group_quant_fp8",
            "per_token_group_quant_fp8",
            None,
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "per_token_group_quant_fp8",
            *args,
            **kwargs,
        )

    def indexer_k_quant_and_cache(self, *args, **kwargs):
        return self._run(
            "indexer_k_quant_and_cache",
            "indexer_k_quant_and_cache",
            None,
            "vllm._custom_ops",
            "indexer_k_quant_and_cache",
            *args,
            **kwargs,
        )

    def cp_gather_indexer_k_quant_cache(self, *args, **kwargs):
        return self._run(
            "cp_gather_indexer_k_quant_cache",
            "cp_gather_indexer_k_quant_cache",
            None,
            "vllm._custom_ops",
            "cp_gather_indexer_k_quant_cache",
            *args,
            **kwargs,
        )

    def mqa_logits(self, *args, **kwargs):
        return self._run(
            "fp8_fp4_mqa_logits",
            "fp8_fp4_mqa_logits",
            None,
            "vllm.utils.deep_gemm",
            "fp8_fp4_mqa_logits",
            *args,
            **kwargs,
        )

    def paged_mqa_logits(self, *args, **kwargs):
        return self._run(
            "fp8_fp4_paged_mqa_logits",
            "fp8_fp4_paged_mqa_logits",
            None,
            "vllm.utils.deep_gemm",
            "fp8_fp4_paged_mqa_logits",
            *args,
            **kwargs,
        )

    def topk_prefill(
        self,
        logits,
        row_starts,
        row_ends,
        indices,
        num_rows,
        stride0,
        stride1,
        top_k,
        *,
        _preflight=False,
    ):

        return self._run(
            "top_k_per_row_prefill",
            "top_k_per_row_prefill",
            None,
            "torch.ops._C",
            "top_k_per_row_prefill",
            logits,
            row_starts,
            row_ends,
            indices,
            num_rows,
            stride0,
            stride1,
            top_k,
            _preflight=_preflight,
        )

    def topk_decode(
        self,
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        stride0,
        stride1,
        top_k,
        *,
        max_seq_len=None,
        _preflight=False,
    ):

        flag_op = self._flag("top_k_per_row_decode", "top_k_per_row_decode")
        ordinary = self._native("torch.ops._C", "top_k_per_row_decode")
        persistent = self._native("torch.ops._C", "persistent_topk")
        workspace_manager = self._native(
            "vllm.v1.worker.workspace", "current_workspace_manager"
        )

        def flag(*args, max_seq_len=None):
            return flag_op(*args)

        def native(logits, next_n, lengths, output, rows, s0, s1, k, max_seq_len=None):
            if (
                current_platform.is_cuda()
                and persistent is not None
                and workspace_manager is not None
                and k in (512, 1024, 2048)
                and max_seq_len is not None
            ):
                (workspace,) = workspace_manager().get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                return persistent(
                    logits, lengths, output, workspace, k, max_seq_len
                )
            return ordinary(
                logits, next_n, lengths, output, rows, s0, s1, k
            )

        return self._call_flag(
            "top_k_per_row_decode",
            flag if flag_op else None,
            None,
            logits,
            next_n,
            seq_lens,
            indices,
            num_rows,
            stride0,
            stride1,
            top_k,
            native=native if ordinary is not None else None,
            max_seq_len=max_seq_len,
            _preflight=_preflight,
        )

    def pack_seq(
        self,
        tensor,
        lengths,
        pad_value=-float("inf"),
        *,
        max_length=None,
        _preflight=False,
    ):
        return self._run(
            "pack_seq_triton",
            "pack_seq",
            None,
            "vllm.v1.attention.ops.common",
            "pack_seq_triton",
            tensor,
            lengths,
            pad_value=pad_value,
            max_length=max_length,
            _preflight=_preflight,
        )

    def unpack_seq(self, tensor, lengths, *, total_tokens=None, _preflight=False):
        return self._run(
            "unpack_seq_triton",
            "unpack_seq",
            None,
            "vllm.v1.attention.ops.common",
            "unpack_seq_triton",
            tensor,
            lengths,
            total_tokens=total_tokens,
            _preflight=_preflight,
        )

    def _kpool(self, name, *args, **kwargs):
        if name in self._bindings:
            if kwargs.pop("_preflight", False):
                return self._bindings[name].describe()
            return self._bindings[name](*args, **kwargs)

        if name in {
            "expand_pools_to_tokens",
            "append_tail_to_topk",
            "expand_pools_and_append_tail",
        }:
            if kwargs.pop("_preflight", False):
                return {"op": name, "implementation": "framework.metadata"}
            return getattr(metadata, name)(*args, **kwargs)
        return self._call_flag(name, self._flag(name, name), None, *args, **kwargs)

    def fwht128_quant_fp8(self, *args, **kwargs):
        return self._kpool("fwht128_quant_fp8", *args, **kwargs)

    def kpool_compress_and_write_cache(self, *args, **kwargs):
        return self._kpool("kpool_compress_and_write_cache", *args, **kwargs)

    def kpool_decode_update_and_maybe_write_cache_batched(self, *args, **kwargs):
        return self._kpool(
            "kpool_decode_update_and_maybe_write_cache_batched", *args, **kwargs
        )

    def kpool_seed_tail_cache(self, *args, **kwargs):
        return self._kpool("kpool_seed_tail_cache", *args, **kwargs)

    def expand_pools_to_tokens(self, *args, **kwargs):
        return self._kpool("expand_pools_to_tokens", *args, **kwargs)

    def append_tail_to_topk(self, *args, **kwargs):
        return self._kpool("append_tail_to_topk", *args, **kwargs)

    def expand_pools_and_append_tail(self, *args, **kwargs):
        return self._kpool("expand_pools_and_append_tail", *args, **kwargs)

    def gather_rows(self, *args, **kwargs):
        return self._kpool("gather_rows", *args, **kwargs)

    def scatter_decode_tokens(self, *args, **kwargs):
        return self._kpool("scatter_decode_tokens", *args, **kwargs)


INDEXER_BACKEND = Glm5NextIndexerBackend()
