"""Bind only the MACA intrinsics used by the pinned 64x128 path."""
from hashlib import sha256
from pathlib import Path
import re

from triton.language import core
from triton_c500_mma_r881 import IDENTITY as MMA_ID, lower_mma, mma_i8
from triton_c500_storage_r704b import install_storage_lowering, physical_thread_id
from triton_w8a8_decode_16x32_r1137 import (
    _extern, epilogue, high32, load4, low32, publish, publish_result,
    scale_one, scale_pair, shared_barrier, store4,
)
from triton_w8a8_gate_support_r1145 import (
    GATE_LLVM, _vectorize, load_b_all, load_b_tail, publish_a_all,
)


@core.builtin
def n128_wait(tid, count, _builder=None):
    return _extern(f'gate_wait_{core._unwrap_if_constexpr(count)}', (tid,), _builder)


def bind_n128(kernel, *, schedule=None, schedule_identity=""):
    assert (schedule is None) == (not schedule_identity)
    base = Path(__file__).with_name('w8a8_decode_primitives_r1137.ll').read_text()
    assert base.count('[24576 x i8]') == 1
    base = base.replace('[24576 x i8]', '[32768 x i8]')
    extra = GATE_LLVM + """
define internal i32 @__w8d_gate_wait_72(i32 %tid) alwaysinline nounwind convergent {
  call void @llvm.mxc.arrive(i32 72)
  ret i32 %tid
}
"""

    def lower(module):
        for name in ('load4','gate_b_tail','gate_b_all'):
            module = _vectorize(module,name)
        declarations = r'(?m)^declare[^\n]*@__w8d_([a-zA-Z0-9_]+)\([^\n]*\)[^\n]*$'
        expected = {'publish','barrier','store4','scale_pair','scale_one','epilogue','publish_result','gate_a_all'}
        expected.update(f'gate_wait_{i}' for i in (64,68,72,4096))
        expected.update(f'{name}_{i}' for name in ('load4','gate_b_tail','gate_b_all') for i in range(4))
        assert sorted(re.findall(declarations,module)) == sorted(expected), re.findall(declarations,module)
        result = lower_mma(re.sub(declarations,'',module)) + base + extra
        return result if schedule is None else schedule(result)

    install_storage_lowering(kernel,extra_lowering=lower,
                             extra_identity=sha256((base+extra+MMA_ID+schedule_identity).encode()).hexdigest())
