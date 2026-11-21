"""C500 16x32x256 native-schedule W8A8 decode experiment."""
import hashlib
from pathlib import Path
import re

import triton
import triton.language as tl
from triton.language import core

from triton_c500_storage_r704b import physical_thread_id, install_storage_lowering
from triton_c500_mma_r881 import IDENTITY as MMA_ID, lower_mma, mma_i8


def _extern(name, args, builder, scalar=core.int32):
    args = [core._to_tensor(x, builder) for x in args]
    dtype = core.block_type(scalar, args[0].type.shape)
    return core.tensor(builder.create_extern_elementwise(
        '', '', '__w8d_' + name, [x.handle for x in args], dtype.to_ir(builder), False), dtype)


@core.builtin
def publish(offset, source, pos, end, _builder=None):
    return _extern('publish', (offset, source, pos, end), _builder)


@core.builtin
def arrive(tid, outstanding, _builder=None):
    return _extern(f'arrive_{core._unwrap_if_constexpr(outstanding)}', (tid,), _builder)


@core.builtin
def shared_barrier(tid, _builder=None):
    return _extern('barrier', (tid,), _builder)


@core.builtin
def load4_word(byte, i, _builder=None):
    return _extern(f'load4_{core._unwrap_if_constexpr(i)}', (byte,), _builder)


@triton.jit
def load4(byte):
    return load4_word(byte, 0), load4_word(byte, 1), load4_word(byte, 2), load4_word(byte, 3)


@core.builtin
def store4(byte, a, b, c, d, _builder=None):
    return _extern('store4', (byte, a, b, c, d), _builder)


@core.builtin
def load2(byte, _builder=None):
    return _extern('load2', (byte,), _builder, core.uint64)


@core.builtin
def scale_pair(valid, source, _builder=None):
    return _extern('scale_pair', (valid, source), _builder, core.uint64)


@core.builtin
def scale_one(valid, source, _builder=None):
    return _extern('scale_one', (valid, source), _builder)


@core.builtin
def epilogue(out0, out1, sa, sb0, sb1, _builder=None):
    return _extern('epilogue', (out0, out1, sa, sb0, sb1), _builder)


@core.builtin
def publish_result(valid, output, packed, _builder=None):
    return _extern('publish_result', (valid, output, packed), _builder)


@triton.jit
def copy_stage(A, B, tid, by, bx, tile, m, n, k, head, STAGE: tl.constexpr, PRIME: tl.constexpr):
    row = tid >> 4
    p = (((tid & 15) - row + 16) & 15) * 16 + STAGE * 256
    dst = (tid >> 6) * 1024
    a = tl.minimum(row, m - 1)
    col0 = tl.minimum(row, n - bx - 1)
    col1 = tl.minimum(row + 16, n - bx - 1)
    if PRIME:
        pred, bound = p, head
    else:
        pred, bound = tid * 0, 1
    publish(dst + STAGE * 4096, A + (by + a) * k + tile * 512 + p, pred, bound)
    publish(dst + 8192 + STAGE * 8192, B + (bx + col0) * k + tile * 512 + p, pred, bound)
    publish(dst + 12288 + STAGE * 8192, B + (bx + col1) * k + tile * 512 + p, pred, bound)


@triton.jit
def dot4(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3):
    c0, c1, c2, c3 = mma_i8(a0, b0, c0, c1, c2, c3)
    c0, c1, c2, c3 = mma_i8(a1, b1, c0, c1, c2, c3)
    c0, c1, c2, c3 = mma_i8(a2, b2, c0, c1, c2, c3)
    c0, c1, c2, c3 = mma_i8(a3, b3, c0, c1, c2, c3)
    return c0, c1, c2, c3


@triton.jit
def low32(x):
    return x.to(tl.uint32).to(tl.int32, bitcast=True)


@triton.jit
def high32(x):
    return low32(x >> 32)


@triton.jit
def w8a8_decode(A, B, Sa, Sb, C, M, N, K):
    tid = physical_thread_id(tl.arange(0, 256))
    by = tl.program_id(0) * 16
    bx = tl.program_id(1) * 32
    tail_tile = (K - 1) // 512
    head = (K - 1) % 512 + 1
    copy_stage(A, B, tid, by, bx, tail_tile, M, N, K, head, 0, True)
    copy_stage(A, B, tid, by, bx, tail_tile, M, N, K, head, 1, True)
    arrive(tid, 67)
    row = tid >> 4
    lane16 = tid & 15
    lds = lane16 * 256 + ((row + lane16) & 15) * 16
    a0,a1,a2,a3 = load4(lds)
    b0,b1,b2,b3 = load4(lds + 8192)
    d0,d1,d2,d3 = load4(lds + 12288)
    zero = tl.full((256,), 0, tl.int32)
    c0,c1,c2,c3 = zero,zero,zero,zero
    e0,e1,e2,e3 = zero,zero,zero,zero
    for tile in range(tail_tile):
        for stage in tl.static_range(2):
            shared_barrier(tid)
            copy_stage(A, B, tid, by, bx, tile, M, N, K, head, stage, False)
            c0,c1,c2,c3 = dot4(a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3)
            arrive(tid, 67)
            n0,n1,n2,n3 = load4(lds + (1 - stage) * 4096)
            e0,e1,e2,e3 = dot4(a0,a1,a2,a3,d0,d1,d2,d3,e0,e1,e2,e3)
            a0,a1,a2,a3 = n0,n1,n2,n3
            b0,b1,b2,b3 = load4(lds + 8192 + (1 - stage) * 8192)
            d0,d1,d2,d3 = load4(lds + 12288 + (1 - stage) * 8192)
    arrive(tid, 64)
    n0,n1,n2,n3 = load4(lds + 4096)
    u0,u1,u2,u3 = load4(lds + 16384)
    v0,v1,v2,v3 = load4(lds + 20480)
    c0,c1,c2,c3 = dot4(a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3)
    e0,e1,e2,e3 = dot4(a0,a1,a2,a3,d0,d1,d2,d3,e0,e1,e2,e3)
    c0,c1,c2,c3 = dot4(n0,n1,n2,n3,u0,u1,u2,u3,c0,c1,c2,c3)
    e0,e1,e2,e3 = dot4(n0,n1,n2,n3,v0,v1,v2,v3,e0,e1,e2,e3)
    col = bx + 2 * lane16
    sb = scale_pair((col < N).to(tl.int32), Sb + col)
    sa = scale_one((by + row < M).to(tl.int32), Sa + by + row)
    shared_barrier(tid)
    write_base = (128 * lane16 + 32 * (tid >> 6)) * 4
    store4(write_base + 16 * ((row & 3) ^ (tid & 7)), c0,c1,c2,c3)
    store4(write_base + 16 * (((row & 3) + 4) ^ (tid & 7)), e0,e1,e2,e3)
    shared_barrier(tid)
    read_base = (128 * row + 4 * ((row & 7) ^ (lane16 >> 1)) + 2 * (tid & 1)) * 4
    s0, s1, s2, s3 = load2(read_base), load2(read_base + 128), load2(read_base + 256), load2(read_base + 384)
    x = ((low32(s0) + low32(s1)) + low32(s2)) + low32(s3)
    y = ((high32(s0) + high32(s1)) + high32(s2)) + high32(s3)
    packed = epilogue(x, y, sa, low32(sb), high32(sb))
    publish_result(((by + row < M) & (col < N)).to(tl.int32),
                   C + (by + row) * N + col, packed)


def lower_decode(module):
    # Keep multiword LDS as one vector read with four ordinary uses.
    groups = {}
    for i in range(4):
        pattern = rf'(?m)^( *)(%[-a-zA-Z$._0-9]+) = (?:tail )?call i32 @__w8d_load4_{i}\(([^\n]*)\)[^\n]*$'

        def bind(match):
            indent, result, args = match.groups()
            key = (args, indent)
            slots = groups.setdefault(key, [])
            if i == 0:
                slots.append((f'%w8d_lds_{sum(map(len, groups.values()))}', 0))
                vector, _ = slots[-1]
                prefix = f'{indent}{vector} = call <4 x i32> @__w8d_load4({args})\n'
            else:
                position = next(j for j, entry in enumerate(slots) if entry[1] == i - 1)
                vector, _ = slots[position]
                slots[position] = (vector, i)
                prefix = ''
            return prefix + f'{indent}{result} = extractelement <4 x i32> {vector}, i32 {i}'

        module, count = re.subn(pattern, bind, module)
        assert count > 0, i
    assert all(i == 3 for slots in groups.values() for _, i in slots)
    for name in ('publish', 'arrive_64', 'arrive_67', 'barrier', 'store4', 'load2', 'scale_pair',
                 'scale_one', 'epilogue', 'publish_result', *(f'load4_{i}' for i in range(4))):
        module, count = re.subn(rf'(?m)^declare[^\n]*@__w8d_{name}\([^\n]*\)[^\n]*$', '', module)
        assert count == 1, (name, count)
    return lower_mma(module) + (Path(__file__).with_name('w8a8_decode_primitives_r1137.ll').read_text())


PRIMITIVES_ID = hashlib.sha256(Path(__file__).with_name('w8a8_decode_primitives_r1137.ll').read_bytes()).hexdigest()
install_storage_lowering(w8a8_decode, extra_lowering=lower_decode, extra_identity=MMA_ID + PRIMITIVES_ID)


def launch(a, b, sa, sb, out):
    import torch

    m, k = a.shape
    n = b.shape[0]
    if (m not in (1, 4) or (n, k) not in ((2560, 2048), (2048, 2048), (2048, 6144))
            or b.shape != (n, k) or sa.shape != (m, 1) or sb.shape != (n, 1)
            or out.shape != (m, n)):
        raise ValueError('shape is outside the measured native dispatch domain')
    for t, dtype in zip((a, b, sa, sb, out),
                        (torch.int8, torch.int8, torch.float32, torch.float32, torch.bfloat16)):
        if t.device != a.device or not t.is_cuda or not t.is_contiguous() or t.dtype != dtype or t.data_ptr() % 16:
            raise ValueError('expected the native dense aligned operands')
    return w8a8_decode[(1, n // 32)](a, b, sa, sb, out, m, n, k,
                                      num_warps=4, num_stages=1, enable_fp_fusion=False)
