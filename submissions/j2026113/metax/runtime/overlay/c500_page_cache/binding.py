"""Query-tile metadata for ABI4; buffers and stream owned by the caller."""
import ctypes as C
import hashlib
from pathlib import Path
import torch
import triton
import triton.language as tl
from .combine import _query_pair_combine


@triton.jit
def _tile_metadata(Cu, Ends, Owners, Bases, Counts, Visible,
                   CAP: tl.constexpr, BATCH: tl.constexpr, TILE: tl.constexpr,
                   BLOCK: tl.constexpr):
    tile = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    owner = tl.full((BLOCK,), 0, tl.int32)
    base = tl.full((BLOCK,), 0, tl.int32)
    count = tl.full((BLOCK,), 0, tl.int32)
    visible = tl.full((BLOCK,), 0, tl.int32)
    prefix = 0
    for request in range(BATCH):
        begin = tl.load(Cu+request)
        finish = tl.load(Cu+request+1)
        qlen = finish-begin
        tiles = tl.cdiv(qlen, TILE)
        active = (tile >= prefix) & (tile < prefix+tiles) & (tile < CAP)
        qb = begin+(tile-prefix)*TILE
        owner = tl.where(active, request, owner)
        base = tl.where(active, qb, base)
        count = tl.where(active, tl.minimum(TILE, finish-qb), count)
        end = tl.load(Ends+request)
        visible = tl.where(active, end-qlen+qb-begin+1, visible)
        prefix += tiles
    tl.store(Owners+tile, owner, mask=tile < CAP)
    tl.store(Bases+tile, base, mask=tile < CAP)
    tl.store(Counts+tile, count, mask=tile < CAP)
    tl.store(Visible+tile, visible, mask=tile < CAP)


class NativeQueryReuse:
    def __init__(self, library, digest):
        p = Path(library)
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
        self.lib = C.CDLL(str(p), mode=C.RTLD_LOCAL)
        self.lib.c500_query_reuse_abi.restype = C.c_int
        assert self.lib.c500_query_reuse_abi() == 4
        self.lib.c500_query_reuse_page_scale.restype = C.c_int
        assert self.lib.c500_query_reuse_page_scale() == 1
        self.lib.c500_query_reuse_queries_per_cta.restype = C.c_int
        self.tile = self.lib.c500_query_reuse_queries_per_cta()
        assert self.tile in (4, 8, 16)
        self.fn = self.lib.c500_query_reuse_forward
        self.fn.argtypes = [C.c_void_p]*13+[C.c_int]*4+[C.c_void_p]
        self.fn.restype = C.c_int
        self.error = self.lib.c500_query_reuse_error
        self.error.argtypes = [C.c_int]
        self.error.restype = C.c_char_p

    def prepare(self, cu, ends, owners, bases, counts, visible):
        _tile_metadata[(triton.cdiv(owners.numel(), 128),)](
            cu, ends, owners, bases, counts, visible,
            owners.numel(), ends.numel(), self.tile, 128,
            num_warps=1, num_stages=1)

    def into(self, q, k, v, qs, ks, cu, owners, bases, counts, visible, table,
             part, maxima, sums, out, *, max_len, splits):
        if splits not in (1, 4, 8, 16, 32, 64):
            raise ValueError('unsupported splits')
        status = self.fn(*[t.data_ptr() for t in (
            q, k, v, qs, ks, visible, table, owners, bases, counts, part, maxima, sums)],
            owners.numel(), table.shape[1], max_len, splits,
            torch.cuda.current_stream(q.device).cuda_stream)
        if status:
            raise RuntimeError('query reuse launch: '+self.error(status).decode())
        _query_pair_combine[(q.shape[0]*2, 8)](
            part, maxima, sums, cu, out, table.shape[0], splits,
            triton.next_power_of_2(splits), num_warps=1, num_stages=1)
