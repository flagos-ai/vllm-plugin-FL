; ModuleID = '/study/actual.bc'
source_filename = "ld-temp.o"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "mxc-metax-macahca"

%struct.mcDevMallocInfo.0 = type { i32, i32, ptr }

@mcDeviceMemoryInfo = weak protected addrspace(1) externally_initialized global [1 x %struct.mcDevMallocInfo.0] zeroinitializer, align 8
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @mcDeviceMemoryInfo to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define metaxgpu_kernel void @scalar_input_phase(ptr addrspace(1) nocapture readonly %0, ptr addrspace(1) nocapture readonly %1, ptr addrspace(1) nocapture writeonly %2, ptr addrspace(1) nocapture writeonly %3, ptr addrspace(1) nocapture writeonly %4, i64 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10) local_unnamed_addr #0 {
  %12 = shl i32 %7, 11
  %13 = add i32 %12, %8
  %14 = icmp slt i32 %13, %6
  br i1 %14, label %15, label %27

15:                                               ; preds = %11
  %16 = sext i32 %13 to i64
  %17 = add i64 %16, %5
  %18 = getelementptr i32, ptr addrspace(1) %0, i64 %17
  %19 = load i32, ptr addrspace(1) %18, align 4
  %20 = getelementptr i32, ptr addrspace(1) %1, i64 %17
  %21 = load i32, ptr addrspace(1) %20, align 4
  %22 = icmp sgt i32 %19, -1
  %23 = select i1 %22, i32 -2147483648, i32 -1
  %24 = sub i32 0, %10
  %25 = xor i32 %23, %24
  %26 = xor i32 %25, %19
  br label %27

27:                                               ; preds = %15, %11
  %28 = phi i32 [ %19, %15 ], [ undef, %11 ]
  %29 = phi i32 [ %21, %15 ], [ undef, %11 ]
  %30 = phi i32 [ %26, %15 ], [ -1, %11 ]
  store i32 %28, ptr addrspace(1) %2, align 4
  store i32 %29, ptr addrspace(1) %3, align 4
  store i32 %30, ptr addrspace(1) %4, align 4
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "metaxgpu-flat-work-group-size"="1, 256" "metaxgpu-implicitarg-num-bytes"="56" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="512" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 0}
!3 = !{i32 1, !"LTOPostLink", i32 1}
