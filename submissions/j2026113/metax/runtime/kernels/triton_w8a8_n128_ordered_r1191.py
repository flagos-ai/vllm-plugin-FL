"""Source lowering for the four native-position B prefetches."""
from hashlib import sha256
from pathlib import Path
import re

import triton

from triton_w8a8_n128_generated_r1157 import w8a8_n128 as original
from triton_w8a8_n128_support_r1157 import bind_n128


def order_first_b(text):
    offsets = set()
    for i, mma in enumerate((1, 5, 9, 14)):
        pattern = (
            r"(?m)^  %(\d+) = getelementptr i8, ptr addrspace\(1\) %\d+, i32 (%\d+)\n"
            rf"^  %w8g_gate_b_all_{i} = call <4 x i32> @__w8d_gate_b_all\(ptr addrspace\(1\) %\1\)\n"
            rf"(?:^  %\d+ = extractelement <4 x i32> %w8g_gate_b_all_{i}, i32 [0-3]\n){{4}}"
        )
        match, = re.finditer(pattern, text)
        offset = match[2]
        offsets.add(offset)
        assert len(offsets) == 1
        assert re.search(rf"(?m)^  {offset} = mul i32 %\d+, 256$", text)
        old = match[0]
        new = old.replace(f"i32 {offset}\n", f"i32 %ordered_offset_{i}\n", 1)
        anchor, = re.findall(rf"(?m)^  %c500_mma_i8_pack_{mma} = call .*?\n", text)
        assert match.start() < text.index(anchor)
        assert not re.search(r"(?m)^\S[^\n]*:\s", text[match.end():text.index(anchor)])
        # A tied output preserves the byte offset. The accumulator input
        # prevents the backend from issuing this prefetch ahead of its MMA.
        dep = (
            f"  %ordered_dep_{i} = extractelement <4 x i32> %c500_mma_i8_pack_{mma}, i32 0\n"
            f'  %ordered_offset_{i} = call i32 asm sideeffect "", "=r,0,r"'
            f"(i32 {offset}, i32 %ordered_dep_{i})\n"
        )
        text = text.replace(old, "")
        text = text.replace(anchor, anchor + dep + new)
        for value in re.findall(r"(?m)^  %(\d+) = ", old):
            text = re.sub(rf"%{value}(?![\w.])", f"%ordered_value_{value}", text)
    return text


w8a8_n128 = triton.jit(original.fn)
bind_n128(w8a8_n128, schedule=order_first_b,
          schedule_identity=sha256(Path(__file__).read_bytes()).hexdigest())
