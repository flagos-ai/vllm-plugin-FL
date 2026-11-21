"""Explicit mixed-cache spec/registry/allocation for an isolated plugin copy.

Storage qualification is not service compute qualification. Compute startup
refuses until the request ownership binding exists; no format conversion.
"""
import math
from dataclasses import dataclass, replace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn import MacaFlashAttentionBackend
from .components import mixed_cache_views, MixedCacheZeroPlan

PAGE_BYTES = 12292


@dataclass(frozen=True, kw_only=True)
class C500MixedPageSpec(FullAttentionSpec):
    representation_version: int = 846

    def __post_init__(self):
        super().__post_init__()
        if (self.block_size, self.num_kv_heads, self.head_size, self.head_size_v) != (16, 2, 128, 128):
            raise ValueError('candidate storage contract is block16/KV2/D128')
        if self.dtype != torch.int8 or self.kv_quant_mode != KVQuantMode.NONE:
            raise ValueError('storage dtype is explicit INT8 backing; scaling is component-defined')
        if self.page_size_padded is not None or self.indexes_kv_by_block_stride:
            raise ValueError('slab layout does not support implicit padding or cross-layer packing')
        if self.sliding_window is not None or self.attention_chunk_size is not None or self.non_causal:
            raise ValueError('candidate is full causal decoder attention')
        if self.representation_version != 846:
            raise ValueError('unknown representation version')

    @property
    def real_page_size_bytes(self):
        return PAGE_BYTES

    @classmethod
    def merge(cls, specs):
        if not specs or any(type(s) is not cls or s != specs[0] for s in specs):
            raise AssertionError('mixed representation cannot merge with another contract')
        return replace(specs[0])


class C500MixedStorageBackend(MacaFlashAttentionBackend):
    @staticmethod
    def get_name():
        return 'C500_PAGE_INT8_K_BF16_V_V846'

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16]

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str='auto'):
        if (block_size, num_kv_heads, head_size) != (16, 2, 128) or cache_dtype_str != 'auto':
            raise ValueError('incompatible mixed-storage request')
        # Budget/container view, not a physical page-major layout.
        return (num_blocks, PAGE_BYTES)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return (0, 1, 2) if include_num_layers_dimension else (0, 1)

    @classmethod
    def indexes_kv_by_block_stride(cls):
        return False

    @classmethod
    def supports_kv_connector(cls):
        return False

    @staticmethod
    def get_impl_cls():
        raise RuntimeError('Mixed storage registered; compute requires explicit runner ownership binding before model startup')


def register_storage_contract():
    # Avoid preventing lazy built-in registration with a premature custom entry.
    KVCacheSpecRegistry._ensure_registered()
    KVCacheSpecRegistry.register(C500MixedPageSpec, FullAttentionManager,
                                uniform_type_base_spec=C500MixedPageSpec)
    register_backend(AttentionBackendEnum.CUSTOM, __name__+'.C500MixedStorageBackend')


@dataclass(frozen=True)
class MixedAuxBudget:
    layers: int
    max_requests: int
    max_query_tokens: int
    splits: int = 16

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (self.layers, self.max_requests, self.max_query_tokens)):
            raise ValueError('positive host dimensions required')
        if self.splits not in (1, 4, 8, 16, 32, 64):
            raise ValueError('unsupported reader split count')

    @property
    def tile_capacity(self):
        return math.ceil(self.max_query_tokens/4)+self.max_requests

    def tensor_specs(self):
        # Scratch is shared sequentially across layers on one runner stream.
        q = self.max_query_tokens
        return {
            'stages': ((self.layers, self.max_requests, 2, 16, 2, 128), torch.bfloat16),
            'banks': ((self.layers, self.max_requests), torch.int32),
            'request_slots': ((self.max_requests,), torch.int32),
            'request_generations': ((self.max_requests,), torch.int64),
            'q8': ((q, 16, 128), torch.int8), 'qs': ((q, 16), torch.bfloat16),
            'owners': ((self.tile_capacity,), torch.int32), 'bases': ((self.tile_capacity,), torch.int32),
            'counts': ((self.tile_capacity,), torch.int32), 'visible': ((self.tile_capacity,), torch.int32),
            'partial': (((q+3)*2, self.splits, 8, 128), torch.float32),
            'maxima': (((q+3)*2, self.splits, 8), torch.float32),
            'sums': (((q+3)*2, self.splits, 8), torch.float32),
            'zero_descriptors': ((self.layers*3, 3), torch.uint64),
            'zero_ids_gpu': ((8192,), torch.int64),
        }

    def byte_breakdown(self):
        return {name: math.prod(shape)*torch.empty((), dtype=dtype).element_size()
                for name, (shape, dtype) in self.tensor_specs().items()}

    @property
    def total_bytes(self):
        return sum(self.byte_breakdown().values())

    def deduct_from_unreserved_available(self, available_bytes):
        if type(available_bytes) is not int or available_bytes <= self.total_bytes:
            raise ValueError('available bytes must exceed auxiliary reservation')
        return available_bytes-self.total_bytes


class MixedCacheAllocation:
    """Bind framework-owned raw tensors with no duplicate backing or BF16 mirror."""
    def __init__(self, kv_cache_config, raw_tensors, *, device, aux_budget, existing_aux=None):
        self.views = {}
        self.budget = aux_budget
        if len(raw_tensors) != aux_budget.layers:
            raise ValueError('auxiliary layer count must match actual layers')
        names = []
        for group in kv_cache_config.kv_cache_groups:
            if type(group.kv_cache_spec) is not C500MixedPageSpec:
                raise ValueError('groups must declare the mixed spec explicitly')
            names.extend(group.layer_names)
        if len(names) != len(set(names)) or set(names) != set(raw_tensors):
            raise ValueError('raw cache coverage/ownership mismatch')
        if any(t.block_stride or len(t.shared_by) != 1 for t in kv_cache_config.kv_cache_tensors):
            raise ValueError('cross-layer packing/sharing is incompatible with current slab ABI')
        for name in names:
            raw = raw_tensors[name]
            if raw.device != torch.device(device):
                raise ValueError('cache device mismatch')
            self.views[name] = mixed_cache_views(raw, kv_cache_config.num_blocks)
        if len({v.raw.data_ptr() for v in self.views.values()}) != len(self.views):
            raise ValueError('layers must own distinct raw cache storage')
        self.aux = existing_aux if existing_aux is not None else {name: torch.empty(shape, dtype=dtype, device=device)
                    for name, (shape, dtype) in aux_budget.tensor_specs().items()}
        for name, (shape, dtype) in aux_budget.tensor_specs().items():
            if self.aux[name].shape != shape or self.aux[name].dtype != dtype or self.aux[name].device != torch.device(device):
                raise ValueError('pre-profiled auxiliary storage differs from declared contract')
        self.aux['stages'].zero_()
        self.aux['banks'].zero_()
        self.aux['request_slots'].fill_(-1)
        self.aux['request_generations'].zero_()
        self.aux['zero_ids_gpu'].fill_(-1)
        self.zero_plan = None
        self.actual_aux_bytes = sum(x.numel()*x.element_size() for x in self.aux.values())
        if self.actual_aux_bytes != aux_budget.total_bytes:
            raise AssertionError('declared and actual auxiliary bytes differ')

    def make_zero_plan(self):
        if self.zero_plan is not None:
            return self.zero_plan
        self.zero_plan = MixedCacheZeroPlan(list(self.views.values()))
        self.aux['zero_descriptors'].copy_(self.zero_plan.descriptors)
        self.zero_plan.descriptors = self.aux['zero_descriptors']
        return self.zero_plan

    def bound_container_tensors(self):
        return {name: view.raw.view(view.pages, PAGE_BYTES) for name, view in self.views.items()}
