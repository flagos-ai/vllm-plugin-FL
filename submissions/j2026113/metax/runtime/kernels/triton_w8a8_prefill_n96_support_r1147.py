"""Lower the prefill-specific B128 scale loads, waits and B64 publication."""
from hashlib import sha256
from pathlib import Path
import re

import triton
from triton.language import core
from triton_c500_mma_r881 import IDENTITY as MMA_ID, lower_mma, mma_i8
from triton_c500_storage_r704b import install_storage_lowering, physical_thread_id
from triton_w8a8_decode_16x32_r1137 import (
    _extern, epilogue, load4, publish, scale_one, shared_barrier, store4,
)
from triton_w8a8_gate_support_r1145 import _vectorize


@core.builtin
def scale_word(valid, ptr, i, _builder=None):
    return _extern(f'n96_scale4_{core._unwrap_if_constexpr(i)}', (valid, ptr), _builder)


@triton.jit
def scale4(valid, ptr):
    return scale_word(valid, ptr, 0), scale_word(valid, ptr, 1), scale_word(valid, ptr, 2), scale_word(valid, ptr, 3)


@triton.jit
def pack64(a, b):
    return a.to(triton.language.uint32, bitcast=True).to(triton.language.uint64) | (b.to(triton.language.uint32, bitcast=True).to(triton.language.uint64) << 32)


@core.builtin
def publish64(valid, output, bits, _builder=None):
    return _extern('n96_publish64', (valid, output, bits), _builder)


@core.builtin
def wait72(tid, _builder=None):
    return _extern('n96_wait72', (tid,), _builder)


@core.builtin
def wait_only64(tid, _builder=None):
    return _extern('n96_wait64', (tid,), _builder)


@core.builtin
def inst_barrier(tid, _builder=None):
    return _extern('n96_inst_barrier', (tid,), _builder)


@core.builtin
def async_fence_marker(tid, _builder=None):
    return _extern('n96_async_marker', (tid,), _builder)


EXTRA = """
declare <4 x i32> @llvm.mxc.ldg.predicator.v4i32(ptr addrspace(1), i32, i64, i1, i1, i1, i1)
declare void @llvm.mxc.stg.predicator.i64(ptr addrspace(1), i32, i64, i64, i1, i1, i1)

define internal <4 x i32> @__w8d_n96_scale4(i32 %valid, ptr addrspace(1) %src) alwaysinline nounwind convergent {
  %p = call i64 @llvm.mxc.icmp.i64.i32(i32 %valid, i32 1, i32 32)
  %value = call <4 x i32> @llvm.mxc.ldg.predicator.v4i32(ptr addrspace(1) %src, i32 0, i64 %p, i1 true, i1 true, i1 false, i1 false)
  ret <4 x i32> %value
}
define internal i32 @__w8d_n96_publish64(i32 %valid, ptr addrspace(1) %dst, i64 %bits) alwaysinline nounwind convergent {
  %p = call i64 @llvm.mxc.icmp.i64.i32(i32 %valid, i32 1, i32 32)
  call void @llvm.mxc.stg.predicator.i64(ptr addrspace(1) %dst, i32 0, i64 %bits, i64 %p, i1 true, i1 false, i1 false)
  ret i32 %valid
}
define internal i32 @__w8d_n96_inst_barrier(i32 %tid) alwaysinline nounwind convergent {
  call void @llvm.mxc.barrier.inst()
  ret i32 %tid
}
define internal i32 @__w8d_n96_async_marker(i32 %tid) alwaysinline nounwind convergent {
  call void asm sideeffect ";--------------", ""()
  ret i32 %tid
}
"""
for count in (64, 72):
    EXTRA += f"""
define internal i32 @__w8d_n96_wait{count}(i32 %tid) alwaysinline nounwind convergent {{
  call void @llvm.mxc.arrive(i32 {count})
  ret i32 %tid
}}
"""


def bind_prefill(kernel):
    base = Path(__file__).with_name('w8a8_decode_primitives_r1137.ll').read_text()
    assert base.count('[24576 x i8]') == 1
    base = base.replace('[24576 x i8]', '[65536 x i8]')
    for op in ('load4', 'store4'):
        a = base.index(f'@__w8d_{op}(')
        z = base.index('\n}', a)
        old = base[a:z]
        new = old.replace(
            'ptr addrspace(3) @__w8d_shared, i32 %offset',
            'ptr addrspacecast (ptr addrspace(3) @__w8d_shared to ptr), i32 %offset')
        new = re.sub(
            r'(?m)^(  (?:%v = load|store) [^\n]+)$',
            r'  call void asm sideeffect ";--------------", ""()\n\1\n  call void asm sideeffect ";--------------", ""()',
            new.replace('ptr addrspace(3) %p, align 16','ptr %p, align 16'))
        assert old != new
        base=base[:a]+new+base[z:]

    def lower(module):
        for name in ('load4', 'n96_scale4'):
            module = _vectorize(module, name)
        declarations = r'(?m)^declare[^\n]*@__w8d_([a-zA-Z0-9_]+)\([^\n]*\)[^\n]*$'
        expected = {'publish', 'store4', 'scale_one', 'barrier', 'epilogue',
                    'n96_wait72', 'n96_wait64', 'n96_inst_barrier', 'n96_async_marker', 'n96_publish64'}
        expected.update(f'{name}_{i}' for name in ('load4', 'n96_scale4') for i in range(4))
        assert sorted(re.findall(declarations,module)) == sorted(expected)
        module = re.sub(declarations,'',module)
        return lower_mma(module) + base + EXTRA

    identity = sha256((base + EXTRA + MMA_ID).encode()).hexdigest()
    install_storage_lowering(kernel, extra_lowering=lower, extra_identity=identity)


def launch_prefill(a, b, sa, sb, out):
    import torch
    from triton_w8a8_prefill_n96_generated_r1147 import w8a8_prefill_n96

    m, k = a.shape
    n = b.shape[0]
    if ((m,n,k) not in ((128,2048,2048),(128,2048,6144)) or b.shape != (n,k)
            or sa.shape != (m,1) or sb.shape != (n,1) or out.shape != (m,n)):
        raise ValueError('outside the measured native 32x96 shapes')
    for t, dtype in zip((a,b,sa,sb,out), (torch.int8,torch.int8,torch.float32,torch.float32,torch.bfloat16)):
        if t.device != a.device or not t.is_cuda or not t.is_contiguous() or t.dtype != dtype or t.data_ptr()%16:
            raise ValueError('expected the native dense aligned operands')
    return w8a8_prefill_n96[(m//32,triton.cdiv(n,96))](a,b,sa,sb,out,m,n,k,
                                                      num_warps=4,num_stages=1,enable_fp_fusion=False)
