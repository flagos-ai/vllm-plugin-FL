"""Keep loop-time BSM prefetches behind their native MMA positions."""
from hashlib import sha256
import inspect
import re

import triton

from triton_w8a8_down_m512_support_r1200 import bind_down
from triton_w8a8_down_m512_zext_r1209 import w8a8_down_m512 as original

SLOTS = (8,12,22,32,42,52,94,104,114,126,166,176,186,196,226,242)


def order_loop_loads(text):
    lines = text.splitlines(keepends=True)
    publish = re.compile(
        r"^  (%\d+) = call i32 @__w8d_publish\(i32 %\d+, ptr addrspace\(1\) "
        r"(%\d+), i32 (%\d+), i32 (%\d+)\)\n$")
    calls = [(idx, m) for idx, line in enumerate(lines) if (m := publish.fullmatch(line))]
    prime, steady = calls[:16], calls[16:]
    assert len(prime) == len(steady) == len(SLOTS)
    assert len({m.group(3, 4) for _,m in prime}) == 1
    assert len({m.group(3, 4) for _,m in steady}) == 1
    assert prime[0][1][3] != steady[0][1][3] and prime[0][1][4] == steady[0][1][4]
    definitions = {m[1]:idx for idx,line in enumerate(lines)
                   if (m := re.match(r"^  (%[\w.]+) = ",line))}
    removed, inserts, renamed = set(), {}, {}
    for i, ((call_index, call), slot) in enumerate(zip(steady, SLOTS, strict=True)):
        anchor_index = definitions[f"%c500_mma_i8_pack_{slot-1}"]
        address_index = definitions[call[2]]
        address = re.fullmatch(
            r"  %\d+ = getelementptr i8, ptr addrspace\(1\) %\d+, i64 (%[\w.]+)\n",
            lines[address_index])
        assert address, lines[address_index]
        offset_index = definitions[address[1]]
        offset = re.fullmatch(r"  %[\w.]+ = zext i32 (%\d+) to i64\n",lines[offset_index])
        assert offset, lines[offset_index]
        assert offset_index < address_index < call_index < anchor_index
        assert not any(re.match(r"^\d+:",line) or "call i32 @__w8d_n96_inst_barrier" in line
                       for line in lines[offset_index:anchor_index])
        # Tied output keeps the offset bits; the second input orders the load.
        prefix = (f"  %bsm_mma_{i} = extractelement <4 x i32> %c500_mma_i8_pack_{slot-1}, i32 0\n"
                  f'  %bsm_offset_{i} = call i32 asm sideeffect "", "=r,0,r"'
                  f"(i32 {offset[1]}, i32 %bsm_mma_{i})\n")
        moved = lines[offset_index].replace(f"zext i32 {offset[1]} to i64",
                                            f"zext i32 %bsm_offset_{i} to i64",1)
        assert anchor_index not in inserts
        inserts[anchor_index] = prefix + moved + lines[address_index] + lines[call_index]
        removed.update((offset_index,address_index,call_index))
        for value in (address[1],call[2],call[1]):
            if value[1:].isdigit():
                renamed[value] = f"%bsm_value_{value[1:]}"
    result = "".join(line + inserts.get(i,"") for i,line in enumerate(lines) if i not in removed)
    return re.sub(r"%\d+(?![\w.])",lambda m:renamed.get(m[0],m[0]),result)


w8a8_down_m512 = triton.jit(original.fn)
bind_down(w8a8_down_m512, schedule=order_loop_loads,
          schedule_identity=sha256((inspect.getsource(order_loop_loads)+repr(SLOTS)).encode()).hexdigest())
