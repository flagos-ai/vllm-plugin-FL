; ModuleID = '/input/helper.bc'
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

; Function Attrs: nofree norecurse nounwind memory(readwrite, inaccessiblemem: none)
define metaxgpu_kernel void @scalar_publish_phase(ptr addrspace(1) nocapture %0, ptr addrspace(1) nocapture readonly %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8) local_unnamed_addr #0 {
  %10 = icmp slt i32 %3, 16
  br i1 %10, label %11, label %71

11:                                               ; preds = %9
  %12 = add i32 %4, -1
  %13 = sext i32 %2 to i64
  %14 = shl nsw i64 %13, 4
  %15 = sext i32 %3 to i64
  %16 = add nsw i64 %14, %15
  %17 = sext i32 %5 to i64
  %18 = mul i64 %16, %17
  %19 = icmp sgt i32 %12, -1
  br i1 %19, label %20, label %40

20:                                               ; preds = %11
  %21 = getelementptr i32, ptr addrspace(1) %0, i64 %18
  br label %22

22:                                               ; preds = %32, %20
  %23 = phi i32 [ %12, %20 ], [ %38, %32 ]
  %24 = phi i32 [ 0, %20 ], [ %35, %32 ]
  %25 = zext nneg i32 %23 to i64
  %26 = getelementptr i32, ptr addrspace(1) %21, i64 %25
  %27 = atomicrmw or ptr addrspace(1) %26, i32 0 syncscope("device-one-as") monotonic, align 4
  %28 = icmp ult i32 %27, 1073741824
  br i1 %28, label %29, label %32, !mma_unroll !4

29:                                               ; preds = %29, %22
  %30 = atomicrmw or ptr addrspace(1) %26, i32 0 syncscope("device-one-as") monotonic, align 4
  %31 = icmp ult i32 %30, 1073741824
  br i1 %31, label %29, label %32, !llvm.loop !5, !mma_unroll !4

32:                                               ; preds = %29, %22
  %33 = phi i32 [ %27, %22 ], [ %30, %29 ]
  %34 = and i32 %33, 1073741823
  %35 = add i32 %34, %24
  %36 = icmp sgt i32 %33, -1
  %37 = add nsw i32 %23, -1
  %38 = select i1 %36, i32 %37, i32 -1
  %39 = icmp sgt i32 %38, -1
  br i1 %39, label %22, label %40, !llvm.loop !7

40:                                               ; preds = %32, %11
  %41 = phi i32 [ 0, %11 ], [ %35, %32 ]
  %42 = getelementptr i32, ptr addrspace(3) @__c500_local_bin_prefix, i32 %3
  %43 = load i32, ptr addrspace(3) %42, align 4
  %44 = getelementptr i8, ptr addrspace(3) %42, i32 4
  %45 = load i32, ptr addrspace(3) %44, align 4
  %46 = shl i32 %4, 11
  %47 = add i32 %46, 2048
  %48 = sub i32 %47, %8
  %49 = tail call i32 @llvm.smax.i32(i32 %48, i32 0)
  %50 = icmp eq i32 %3, 15
  %51 = select i1 %50, i32 %49, i32 0
  %52 = sext i32 %4 to i64
  %53 = getelementptr i32, ptr addrspace(1) %0, i64 %18
  %54 = getelementptr i32, ptr addrspace(1) %53, i64 %52
  %55 = add i32 %51, %43
  %56 = sub i32 %41, %55
  %57 = add i32 %56, %45
  %58 = or i32 %57, -2147483648
  %59 = atomicrmw xchg ptr addrspace(1) %54, i32 %58 syncscope("device-one-as") monotonic, align 4
  %60 = sext i32 %6 to i64
  %61 = mul nsw i64 %60, %13
  %62 = sext i32 %7 to i64
  %63 = add nsw i64 %61, %62
  %64 = shl i64 %63, 6
  %65 = getelementptr i8, ptr addrspace(1) %1, i64 %64
  %66 = getelementptr i32, ptr addrspace(1) %65, i64 %15
  %67 = load i32, ptr addrspace(1) %66, align 4
  %68 = sub i32 %41, %43
  %69 = add i32 %68, %67
  %70 = getelementptr i32, ptr addrspace(3) @__c500_scatter_base, i32 %3
  store i32 %69, ptr addrspace(3) %70, align 4
  br label %71

71:                                               ; preds = %40, %9
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #1

attributes #0 = { nofree norecurse nounwind memory(readwrite, inaccessiblemem: none) "metaxgpu-flat-work-group-size"="1, 256" "metaxgpu-implicitarg-num-bytes"="56" "metaxgpu-inline-scope"="11" "metaxgpu-max-block-size"="512" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 0}
!3 = !{i32 1, !"LTOPostLink", i32 1}
!4 = !{!"already_mma_unroll"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.unroll.disable"}
!7 = distinct !{!7, !6}
