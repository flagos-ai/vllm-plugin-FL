"""Kernel-local lowering for the four-stage 128x128x256 down projection."""
from hashlib import sha256
from pathlib import Path
import re

from triton.language import core
from triton_c500_mma_r881 import IDENTITY as MMA_ID, lower_mma, mma_i8
from triton_c500_storage_r704b import install_storage_lowering, physical_thread_id
from triton_w8a8_decode_16x32_r1137 import _extern, epilogue, load4, publish
from triton_w8a8_gate_support_r1145 import _vectorize
from triton_w8a8_prefill_n96_support_r1147 import (
    EXTRA, async_fence_marker, inst_barrier, pack64, publish64, scale4,
)


@core.builtin
def down_wait(tid, count, _builder=None):
    return _extern(f'down_wait_{core._unwrap_if_constexpr(count)}', (tid,), _builder)


@core.builtin
def publish_scale(offset, source, valid, _builder=None):
    return _extern('down_publish_scale', (offset, source, valid), _builder)


@core.builtin
def shared_scale(offset, _builder=None):
    return _extern('down_shared_scale', (offset,), _builder)


WAITS = (64,65,68,70,72,76,4096)
LLVM = """
declare i32 @llvm.mxc.ldg.predicator.bsm.i32(ptr addrspace(3), ptr addrspace(1), i32, i64, i1, i1, i1, i1)

define internal i32 @__w8d_down_publish_scale(i32 %offset, ptr addrspace(1) %src, i32 %valid) alwaysinline nounwind convergent {
  %dst = getelementptr inbounds i8, ptr addrspace(3) @__w8d_shared, i32 %offset
  %pred = call i64 @llvm.mxc.icmp.i64.i32(i32 %valid, i32 1, i32 32)
  %value = call i32 @llvm.mxc.ldg.predicator.bsm.i32(ptr addrspace(3) %dst, ptr addrspace(1) %src, i32 0, i64 %pred, i1 true, i1 true, i1 false, i1 true)
  ret i32 %offset
}

define internal i32 @__w8d_down_shared_scale(i32 %offset) alwaysinline nounwind convergent {
  %p = getelementptr inbounds i8, ptr addrspacecast (ptr addrspace(3) @__w8d_shared to ptr), i32 %offset
  %v = load i32, ptr %p, align 4
  ret i32 %v
}
"""
for count in WAITS:
    LLVM += f"""
define internal i32 @__w8d_down_wait_{count}(i32 %tid) alwaysinline nounwind convergent {{
  call void @llvm.mxc.arrive(i32 {count})
  ret i32 %tid
}}
"""


def bind_down(kernel, *, schedule=None, schedule_identity=""):
    assert (schedule is None) == (not schedule_identity)
    base = Path(__file__).with_name('w8a8_decode_primitives_r1137.ll').read_text()
    assert base.count('[24576 x i8]') == 1
    base = base.replace('[24576 x i8]', '[65536 x i8]')
    start = base.index('define internal <4 x i32> @__w8d_load4(')
    end = base.index('\n}',start)
    helper = base[start:end]
    modified = helper.replace(
        'ptr addrspace(3) @__w8d_shared, i32 %offset',
        'ptr addrspacecast (ptr addrspace(3) @__w8d_shared to ptr), i32 %offset')
    modified = modified.replace(
        '  %v = load <4 x i32>, ptr addrspace(3) %p, align 16',
        '  call void asm sideeffect ";--------------", ""()\n'
        '  %v = load <4 x i32>, ptr %p, align 16\n'
        '  call void asm sideeffect ";--------------", ""()')
    assert modified != helper and modified.count('asm sideeffect')==2
    base = base[:start] + modified + base[end:]
    extra = EXTRA + LLVM

    def lower(module):
        for name in ('load4', 'n96_scale4'):
            module = _vectorize(module, name)
        declarations = r'(?m)^declare[^\n]*@__w8d_([A-Za-z0-9_]+)\([^\n]*\)[^\n]*$'
        expected = {'publish','epilogue','n96_publish64','n96_inst_barrier','n96_async_marker',
                    'down_publish_scale','down_shared_scale'}
        expected.update(f'down_wait_{i}' for i in WAITS)
        expected.update(f'{name}_{i}' for name in ('load4','n96_scale4') for i in range(4))
        assert sorted(re.findall(declarations,module)) == sorted(expected), re.findall(declarations,module)
        text = lower_mma(re.sub(declarations,'',module)) + base + extra
        return text if schedule is None else schedule(text)

    install_storage_lowering(kernel, extra_lowering=lower,
                             extra_identity=sha256((base+extra+MMA_ID+schedule_identity).encode()).hexdigest())
