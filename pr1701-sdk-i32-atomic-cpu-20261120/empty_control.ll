target datalayout = "e-m:e-p:32:32-p1:64:64-p2:32:32-p4:32:32-p135:64:64-p134:64:64-p133:32:32-p131:64:64-i1:8:32-i8:8:32-i16:16:32-i64:64:64-v512:512-a:0:32-n32-S32"
target triple = "xpu3-baidu-none-gnu"

define xpu_kernel void @empty_control(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  ret void
}

declare void @llvm.xpu.gm2lm_v3(ptr, ptr addrspace(1), i32, i32)
declare void @llvm.xpu.lm2gm_v3(ptr addrspace(1), ptr, i32, i32)
declare void @llvm.xpu.mfence.v2(i32)

!xpu.annotations = !{!0}
!0 = !{ptr @empty_control, !"kernel", i32 1}
