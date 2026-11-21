"""Kernel-local MACA lowering for the gate/up schedule."""
from hashlib import sha256
from pathlib import Path
import re

import triton
from triton.language import core
from triton_c500_mma_r881 import IDENTITY as MMA_ID, lower_mma, mma_i8
from triton_c500_storage_r704b import install_storage_lowering, physical_thread_id
from triton_w8a8_decode_16x32_r1137 import (
    _extern, epilogue, high32, load4, low32, publish, publish_result,
    scale_one, scale_pair, shared_barrier, store4,
)


@core.builtin
def b_word_tail(pos, ptr, end, i, _builder=None):
    return _extern(f'gate_b_tail_{core._unwrap_if_constexpr(i)}', (pos, ptr, end), _builder)


@core.builtin
def b_word_all(ptr, i, _builder=None):
    return _extern(f'gate_b_all_{core._unwrap_if_constexpr(i)}', (ptr,), _builder)


@core.builtin
def wait_only(tid, count, _builder=None):
    return _extern(f'gate_wait_{core._unwrap_if_constexpr(count)}', (tid,), _builder)


@core.builtin
def publish_a_all(dst, ptr, _builder=None):
    return _extern('gate_a_all', (dst, ptr), _builder)


@triton.jit
def load_b_tail(pos, ptr, end):
    return b_word_tail(pos, ptr, end, 0), b_word_tail(pos, ptr, end, 1), b_word_tail(pos, ptr, end, 2), b_word_tail(pos, ptr, end, 3)


@triton.jit
def load_b_all(ptr):
    return b_word_all(ptr, 0), b_word_all(ptr, 1), b_word_all(ptr, 2), b_word_all(ptr, 3)


def _vectorize(module, name):
    slots = {}
    for i in range(4):
        pattern = rf'(?m)^( *)(%[-a-zA-Z$._0-9]+) = (?:tail )?call i32 @__w8d_{name}_{i}\(([^\n]*)\)[^\n]*$'

        def bind(match):
            indent, result, args = match.groups()
            sequence = slots.setdefault(args, [])
            if i == 0:
                vector = f'%w8g_{name}_{sum(map(len, slots.values()))}'
                sequence.append((vector, 0))
                prefix = f'{indent}{vector} = call <4 x i32> @__w8d_{name}({args})\n'
            else:
                j = next(j for j, (_, seen) in enumerate(sequence) if seen == i - 1)
                vector, _ = sequence[j]
                sequence[j] = (vector, i)
                prefix = ''
            return f'{prefix}{indent}{result} = extractelement <4 x i32> {vector}, i32 {i}'

        module, count = re.subn(pattern, bind, module)
        assert count > 0, (name, i)
    assert all(i == 3 for sequence in slots.values() for _, i in sequence)
    return module


GATE_LLVM = """
declare <4 x i32> @llvm.mxc.ldg.predicator.v4i32(ptr addrspace(1), i32, i64, i1, i1, i1, i1)

define internal <4 x i32> @__w8d_gate_b_tail(i32 %pos, ptr addrspace(1) %src, i32 %end) alwaysinline nounwind convergent {
  %p = call i64 @llvm.mxc.icmp.i64.i32(i32 %pos, i32 %end, i32 40)
  %value = call <4 x i32> @llvm.mxc.ldg.predicator.v4i32(ptr addrspace(1) %src, i32 0, i64 %p, i1 true, i1 true, i1 false, i1 false)
  ret <4 x i32> %value
}
define internal <4 x i32> @__w8d_gate_b_all(ptr addrspace(1) %src) alwaysinline nounwind convergent {
  %value = call <4 x i32> @llvm.mxc.ldg.predicator.v4i32(ptr addrspace(1) %src, i32 0, i64 -1, i1 true, i1 true, i1 false, i1 false)
  ret <4 x i32> %value
}
define internal i32 @__w8d_gate_a_all(i32 %offset, ptr addrspace(1) %src) alwaysinline nounwind convergent {
  %dst = getelementptr inbounds i8, ptr addrspace(3) @__w8d_shared, i32 %offset
  %copy = call <4 x i32> @llvm.mxc.ldg.predicator.bsm.v4i32(ptr addrspace(3) %dst, ptr addrspace(1) %src, i32 0, i64 -1, i1 true, i1 true, i1 false, i1 true)
  ret i32 %offset
}
"""
for count in (64, 68, 4096):
    GATE_LLVM += f"""
define internal i32 @__w8d_gate_wait_{count}(i32 %tid) alwaysinline nounwind convergent {{
  call void @llvm.mxc.arrive(i32 {count})
  ret i32 %tid
}}
"""


def bind_gate(kernel):
    base = Path(__file__).with_name('w8a8_decode_primitives_r1137.ll').read_text()
    assert base.count('[24576 x i8]') == 1
    base = base.replace('[24576 x i8]', '[20480 x i8]')

    def lower(module):
        for name in ('load4', 'gate_b_tail', 'gate_b_all'):
            module = _vectorize(module, name)
        module, count = re.subn(r'(?m)^declare[^\n]*@__w8d_[a-zA-Z0-9_]+\([^\n]*\)[^\n]*$', '', module)
        assert count == 23, count
        return lower_mma(module) + base + GATE_LLVM

    identity = sha256((base + GATE_LLVM + MMA_ID).encode()).hexdigest()
    install_storage_lowering(kernel, extra_lowering=lower, extra_identity=identity)


def launch_gate(a, b, sa, sb, out):
    import torch
    from triton_w8a8_gate_generated_r1145 import w8a8_gate

    m, k = a.shape
    n = b.shape[0]
    if (m not in (1, 4) or (n, k) != (12288, 2048) or b.shape != (n, k)
            or sa.shape != (m, 1) or sb.shape != (n, 1) or out.shape != (m, n)):
        raise ValueError('outside the measured native gate/up shapes')
    for t, dtype in zip((a, b, sa, sb, out),
                        (torch.int8, torch.int8, torch.float32, torch.float32, torch.bfloat16)):
        if t.device != a.device or not t.is_cuda or not t.is_contiguous() or t.dtype != dtype or t.data_ptr() % 16:
            raise ValueError('expected the native dense aligned operands')
    return w8a8_gate[(1, n // 128)](a, b, sa, sb, out, m, n, k,
                                    num_warps=4, num_stages=1, enable_fp_fusion=False)
