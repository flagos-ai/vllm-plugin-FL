"""Explicit mixed-component views and lifecycle zeroing, isolated integration candidate.

No reader/writer math or cache format changes. A descriptor carries each
component's byte stride, so physical block IDs address K, V and scale correctly.
Metadata is prepared before capture; one launch performs the selected clears.
The caller supplies valid block IDs or -1 padding and must preserve ownership.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class MixedCacheViews:
    raw: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    scale: torch.Tensor

    @property
    def pages(self):
        return self.k.shape[0]


def mixed_cache_views(raw, pages):
    if type(pages) is not int or pages < 1:
        raise ValueError('pages must be positive host int')
    if raw.dtype != torch.int8 or raw.ndim != 1 or not raw.is_contiguous():
        raise ValueError('mixed backing must be contiguous one-dimensional INT8 bytes')
    if raw.numel() != pages*12292:
        raise ValueError('backing must account for K4096+V8192+scale4 bytes per page')
    k_end, v_end = pages*4096, pages*(4096+8192)
    return MixedCacheViews(raw, raw[:k_end].view(pages, 16, 2, 128),
                           raw[k_end:v_end].view(torch.bfloat16).view(pages, 16, 2, 128),
                           raw[v_end:].view(torch.bfloat16).view(pages, 2))


@triton.jit
def _zero_mixed_components(Descriptors, BlockIds, PAGES: tl.constexpr,
                           BLOCK: tl.constexpr):
    request = tl.program_id(0)
    component = tl.program_id(1)
    chunk = tl.program_id(2)
    block_id = tl.load(BlockIds+request)
    if block_id >= 0 and block_id < PAGES:
        base = tl.load(Descriptors+component*3)
        stride_words = tl.load(Descriptors+component*3+1)
        words = tl.load(Descriptors+component*3+2)
        offset = chunk*BLOCK+tl.arange(0, BLOCK)
        destination = base.to(tl.pointer_type(tl.int32))+block_id*stride_words+offset
        tl.store(destination, 0, mask=offset < words)


class MixedCacheZeroPlan:
    def __init__(self, layer_views):
        if not layer_views:
            raise ValueError('at least one layer is required')
        pages = layer_views[0].pages
        device = layer_views[0].raw.device
        if device.type != 'cuda':
            raise ValueError('device plan requires CUDA/MACA tensors')
        descriptors, seen = [], set()
        for views in layer_views:
            if views.pages != pages or views.raw.device != device:
                raise ValueError('all layers must have identical page counts and device')
            for tensor in [views.k, views.v, views.scale]:
                if tensor.data_ptr() in seen:
                    continue
                seen.add(tensor.data_ptr())
                stride_bytes = tensor.stride(0)*tensor.element_size()
                extent_bytes = tensor[0].numel()*tensor.element_size()
                if not tensor.is_contiguous() or stride_bytes != extent_bytes or stride_bytes % 4:
                    raise ValueError('component pages must be contiguous and word aligned')
                if tensor.data_ptr() % 4:
                    raise ValueError('component base must be word aligned')
                descriptors.append([tensor.data_ptr(), stride_bytes//4, extent_bytes//4])
        self.views = tuple(layer_views)
        self.descriptors_cpu = descriptors
        self.descriptors = torch.tensor(descriptors, dtype=torch.uint64, device=device)
        self.pages = pages
        self.max_words = max(x[2] for x in descriptors)
        self.device = device

    def zero(self, block_ids):
        if block_ids.dtype != torch.int64 or block_ids.ndim != 1 or not block_ids.is_contiguous():
            raise ValueError('block IDs must be a contiguous INT64 vector with -1 padding')
        if block_ids.device != self.device:
            raise ValueError('block IDs and cache must be on same device')
        if block_ids.numel() == 0:
            return
        _zero_mixed_components[(block_ids.numel(), len(self.descriptors_cpu), triton.cdiv(self.max_words, 256))](
            self.descriptors, block_ids, self.pages, 256, num_warps=4, num_stages=1)
