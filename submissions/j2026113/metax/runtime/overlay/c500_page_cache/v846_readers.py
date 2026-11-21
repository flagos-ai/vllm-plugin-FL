"""Use the validated readers without loading either native reader library."""
import hashlib
import importlib.util
from pathlib import Path

import torch
import triton

from .binding import _tile_metadata
from triton_native_prefill_m128_grouped_output_r1107 import m128_reader
from triton_native_reader_first_k_writeback_r1078 import launch_partial


class PreparedReader:
    V_CACHE_LAYOUT = 846

    def __init__(self, mode, *, single_query=False):
        if (mode, single_query) not in (('prefill', False), ('decode', True)):
            raise ValueError('unsupported reader path')
        self.mode = mode
        self.tile = 128 if mode == 'prefill' else 4
        if mode == 'decode':
            path = Path('/arch/evidence/unified_direct_build_r860/decode_only_merge_r860.py')
            if hashlib.sha256(path.read_bytes()).hexdigest() != '10ffb5a6a395b0c3009d05831b5f1777e5ffeb7efe2787f2015ff1e727358ab4':
                raise ValueError('decode merge source hash differs')
            spec = importlib.util.spec_from_file_location('vendor_mixed_merge_r860', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.merge = module.decode_only_merge

    def prepare(self, cu, ends, owners, bases, counts, visible):
        _tile_metadata[(triton.cdiv(owners.numel(), 128),)](
            cu, ends, owners, bases, counts, visible,
            owners.numel(), ends.numel(), self.tile, 128, num_warps=1, num_stages=1)

    def into(self, q, k, v, qs, ks, cu, owners, bases, counts, visible, table,
             part, maxima, sums, out, *, max_len, splits):
        if not out.is_contiguous() or out.dtype != torch.bfloat16 or out.data_ptr() % 8:
            raise ValueError('reader requires aligned contiguous BF16 output')
        if not all(t.is_contiguous() for t in (q,k,v,qs,ks,cu,owners,bases,counts,visible,table)):
            raise ValueError('reader requires contiguous input buffers')
        if table.shape[1]*16 < max_len:
            raise ValueError('sequence exceeds block table')
        if self.mode == 'prefill':
            if splits != 1:
                raise ValueError('prefill needs one split')
            m128_reader[(16*owners.numel(),1)](
                q.view(torch.int32), k.view(torch.int32), v.view(torch.int32), qs, ks,
                visible, table, owners, bases, counts, out.view(torch.int32),
                owners.numel(), table.shape[1], num_warps=4, num_stages=1, enable_fp_fusion=False)
            return
        if splits != 16:
            raise ValueError('decode needs the validated S16 path')
        launch_partial(q,k,v,qs,ks,visible,table,owners,bases,counts,part,maxima,sums,splits=splits)
        self.merge[(cu.numel()-1,16)](part,maxima,sums,cu,out,splits,splits,
                                     num_warps=1,num_stages=1)
