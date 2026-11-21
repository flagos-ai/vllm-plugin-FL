; ModuleID = '/study/actual.bc'
source_filename = "ld-temp.o"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "mxc-metax-macahca"

%struct.mcDevMallocInfo.0 = type { i32, i32, ptr }

@__c500_claimed_tile = local_unnamed_addr addrspace(3) externally_initialized global i32 undef, align 4
@__c500_rank_temporary = local_unnamed_addr addrspace(3) externally_initialized global [544 x i8] undef, align 16
@__c500_scatter_base = local_unnamed_addr addrspace(3) externally_initialized global [16 x i32] undef, align 4
@__c500_local_bin_prefix = local_unnamed_addr addrspace(3) externally_initialized global [17 x i32] undef, align 4
@mcDeviceMemoryInfo = weak protected addrspace(1) externally_initialized global [1 x %struct.mcDevMallocInfo.0] zeroinitializer, align 8
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @mcDeviceMemoryInfo to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none)
define metaxgpu_kernel void @scalar_scatter_phase(ptr addrspace(1) nocapture writeonly %0, ptr addrspace(1) nocapture writeonly %1, i64 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9) local_unnamed_addr #0 {
  %11 = shl i32 %4, 11
  %12 = add i32 %11, %5
  %13 = icmp slt i32 %12, %3
  br i1 %13, label %14, label %22

14:                                               ; preds = %10
  %15 = getelementptr i32, ptr addrspace(3) @__c500_scatter_base, i32 %8
  %16 = load i32, ptr addrspace(3) %15, align 4
  %17 = add i32 %16, %9
  %18 = zext i32 %17 to i64
  %19 = add i64 %18, %2
  %20 = getelementptr i32, ptr addrspace(1) %0, i64 %19
  store i32 %6, ptr addrspace(1) %20, align 4
  %21 = getelementptr i32, ptr addrspace(1) %1, i64 %19
  store i32 %7, ptr addrspace(1) %21, align 4
  br label %22

22:                                               ; preds = %14, %10
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none) "metaxgpu-flat-work-group-size"="1, 256" "metaxgpu-implicitarg-num-bytes"="56" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="512" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 0}
!3 = !{i32 1, !"LTOPostLink", i32 1}
