#!/usr/bin/env python3
"""FlagOS Ascend 冒烟测试: FlagGems + Triton kernel + FlagCX 通信器验证"""
import os, sys, torch, torch_npu
import flag_gems
import triton
import triton.language as tl

flag_gems.enable()
assert flag_gems.__version__


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


x = torch.randn(1024, device="npu")
y = torch.randn(1024, device="npu")
out = torch.empty_like(x)
add_kernel[(1,)](x, y, out, 1024, BLOCK=1024)
torch.npu.synchronize()
assert torch.allclose(out, x + y), "add kernel mismatch"

a = torch.randn(16, 16, device="npu")
assert torch.allclose(torch.relu(a), torch.clamp(a, min=0), atol=1e-5), "flag_gems relu mismatch"
b = torch.randn(8, 16, device="npu")
assert torch.allclose(torch.pow(b, 2), b * b, atol=1e-4), "flag_gems pow mismatch"

flagcx_path = os.environ.get("FLAGCX_PATH", "")
if flagcx_path:
    sys.path.insert(0, flagcx_path)
    import flagcx
    import torch.distributed as dist
    assert dist.is_backend_available("flagcx"), "flagcx backend not available"
    from vllm.platforms import current_platform
    assert current_platform.dist_backend == "flagcx", \
        f"dist_backend should be 'flagcx', got '{current_platform.dist_backend}'"
    assert "CommunicatorFL" in current_platform.get_device_communicator_cls(), \
        f"communicator should be CommunicatorFL, got '{current_platform.get_device_communicator_cls()}'"
    print(f"[OK] FlagCX: dist_backend={current_platform.dist_backend}, "
          f"communicator={current_platform.get_device_communicator_cls()}")
else:
    print("[WARN] FLAGCX_PATH not set — multi-NPU (TP>1) will fail with NCCL error")

print("FLAGOS SMOKE OK")
