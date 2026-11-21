"""Bind the gate/up M512 schedule to existing MACA Triton primitives."""
from hashlib import sha256
from pathlib import Path
import re

from triton.language import core
from triton_c500_mma_r881 import IDENTITY as MMA_ID, lower_mma, mma_i8
from triton_c500_storage_r704b import install_storage_lowering, physical_thread_id
from triton_w8a8_decode_16x32_r1137 import (
    _extern, load4, shared_barrier, store4, scale_one, scale_pair, low32, high32, publish_result,
)
from triton_w8a8_gate_support_r1145 import _vectorize, publish_a_all, load_b_all, GATE_LLVM
from triton_w8a8_prefill_n96_support_r1147 import EXTRA, async_fence_marker, inst_barrier, publish64, scale4


@core.builtin
def gate_wait(tid, count, _builder=None):
    return _extern(f'gate512_wait_{core._unwrap_if_constexpr(count)}', (tid,), _builder)


@core.builtin
def gate_publish128(dst, src, valid, _builder=None):
    return _extern('gate512_publish128', (dst, src, valid), _builder)


@core.builtin
def gate_scale_bsm(dst, src, valid, _builder=None):
    return _extern('gate512_scale_bsm', (dst, src, valid), _builder)


@core.builtin
def gate_shared_float(dst, _builder=None):
    return _extern('gate512_shared_bits', (dst,), _builder).to(core.float32, bitcast=True, _builder=_builder)


@core.builtin
def gate_pack_bf16(a, b, _builder=None):
    return _extern('gate512_pack_bf16', (a, b), _builder)


@core.builtin
def gate_add_zero(a, _builder=None):
    bits = a.to(core.int32, bitcast=True, _builder=_builder)
    return _extern('gate512_add_zero_bits', (bits,), _builder).to(core.float32, bitcast=True, _builder=_builder)


WAITS = (64,65,66,67,68,69,70,72,76,4096,4352)
EXACT = '''
declare i32 @llvm.mxc.ldg.predicator.bsm.i32(ptr addrspace(3), ptr addrspace(1), i32, i64, i1, i1, i1, i1)

define internal i32 @__w8d_gate512_publish128(i32 %offset, ptr addrspace(1) %src, i32 %valid) alwaysinline nounwind convergent {
  %dst = getelementptr inbounds i8, ptr addrspace(3) @__w8d_shared, i32 %offset
  %pred = call i64 @llvm.mxc.icmp.i64.i32(i32 %valid, i32 1, i32 32)
  %copy = call <4 x i32> @llvm.mxc.ldg.predicator.bsm.v4i32(ptr addrspace(3) %dst, ptr addrspace(1) %src, i32 0, i64 %pred, i1 true, i1 true, i1 false, i1 true)
  ret i32 %offset
}
define internal i32 @__w8d_gate512_scale_bsm(i32 %offset, ptr addrspace(1) %src, i32 %valid) alwaysinline nounwind convergent {
  %dst = getelementptr inbounds i8, ptr addrspace(3) @__w8d_shared, i32 %offset
  %pred = call i64 @llvm.mxc.icmp.i64.i32(i32 %valid, i32 1, i32 32)
  %copy = call i32 @llvm.mxc.ldg.predicator.bsm.i32(ptr addrspace(3) %dst, ptr addrspace(1) %src, i32 0, i64 %pred, i1 true, i1 true, i1 false, i1 true)
  ret i32 %offset
}
define internal i32 @__w8d_gate512_shared_bits(i32 %offset) alwaysinline nounwind {
  %dst = getelementptr inbounds i8, ptr addrspace(3) @__w8d_shared, i32 %offset
  %value = load i32, ptr addrspace(3) %dst, align 4
  ret i32 %value
}
define internal i32 @__w8d_gate512_pack_bf16(i32 %a, i32 %b) alwaysinline nounwind {
  %bits = call i32 @llvm.mxc.byte.perm(i32 %a, i32 %b, i32 50464518)
  ret i32 %bits
}
define internal i32 @__w8d_gate512_add_zero_bits(i32 %x) alwaysinline nounwind {
  %value = bitcast i32 %x to float
  %plus = fadd contract float %value, 0.0
  %bits = bitcast float %plus to i32
  ret i32 %bits
}
'''
for count in WAITS:
    EXACT += f'''
define internal i32 @__w8d_gate512_wait_{count}(i32 %tid) alwaysinline nounwind convergent {{
  call void @llvm.mxc.arrive(i32 {count})
  ret i32 %tid
}}
'''


def bind_gate_m512(kernel, large=False, dual=None):
    base = Path(__file__).with_name('w8a8_decode_primitives_r1137.ll').read_text()
    assert base.count('[24576 x i8]') == 1
    base = base.replace('[24576 x i8]', '[32768 x i8]' if large or dual in ('o','qkv') else '[65536 x i8]')
    line = '  %v = load <4 x i32>, ptr addrspace(3) %p, align 16'
    marker = '  call void asm sideeffect ";--------------", ""()'
    assert base.count(line) == 1
    base = base.replace(line, '\n'.join((marker, line, marker)))
    if large or dual:
        for op in ('load4','store4'):
            x = base.index(f'@__w8d_{op}(')
            y = base.index('\n}',x)
            old = base[x:y]
            new = old.replace('ptr addrspace(3) @__w8d_shared, i32 %offset',
                              'ptr addrspacecast (ptr addrspace(3) @__w8d_shared to ptr), i32 %offset')
            new = new.replace('ptr addrspace(3) %p, align 16','ptr %p, align 16')
            base = base[:x] + new + base[y:]
    a = GATE_LLVM.index('define internal i32 @__w8d_gate_a_all(')
    tail = GATE_LLVM.index('\n}', a) + 2
    a0 = GATE_LLVM.index('define internal <4 x i32> @__w8d_gate_b_all(')
    helpers = base + '\n' + GATE_LLVM[a0 if large or dual else a:tail] + '\n' + EXTRA + EXACT

    def lower(module):
        names = ('load4','n96_scale4','gate_b_all') if large or dual else ('load4','n96_scale4')
        for name in names:
            module = _vectorize(module,name)
        pat = r'(?m)^declare[^\n]*@__w8d_([A-Za-z0-9_]+)\([^\n]*\)[^\n]*$'
        expected = {'barrier','gate_a_all','gate512_publish128','gate512_scale_bsm',
                    'gate512_shared_bits','gate512_pack_bf16','gate512_add_zero_bits',
                    'n96_inst_barrier','n96_async_marker','n96_publish64'}
        if large:
            expected -= {'gate_a_all','gate512_publish128','n96_inst_barrier'}
            expected.add('store4')
        expected.update(f'gate512_wait_{i}' for i in ((65,) if large else (64,65,66,67,68,69,4352)))
        expected.update(f'{name}_{i}' for name in names for i in range(4))
        actual = set(re.findall(pat,module))
        if dual:
            allowed = expected | {'store4','scale_one','scale_pair','publish_result'}
            allowed.update(f'gate512_wait_{i}' for i in WAITS)
            assert actual <= allowed and {'barrier','store4','scale_pair','publish_result'} <= actual, actual
        else:
            assert actual == expected, actual
        return lower_mma(re.sub(pat,'',module)) + helpers

    identity = sha256((helpers+MMA_ID).encode()).hexdigest()
    return install_storage_lowering(kernel, extra_lowering=lower, extra_identity=identity)
