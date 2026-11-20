"""Experimental fused MLA; currently slower than vendor. Not deployed."""
from .packed_acc import sparse_mla as _buffer
from .compact_q import sparse_mla as _wide

def sparse_mla(q, cache, indices, scale, *, splits=None):
    # buffer_load uses signed 32-bit byte offsets. Retain 64-bit addressing
    # for larger extents; Q is BF16 and indices are int32.
    use_buffer = all(x.numel() * x.element_size() < 2**31 for x in (q, cache, indices))
    implementation = _buffer if use_buffer else _wide
    return implementation(q, cache, indices, scale, splits=splits, block_n=16, num_warps=2)
