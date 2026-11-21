
@__w8d_shared = internal addrspace(3) global [24576 x i8] undef, align 16

declare i64 @llvm.mxc.icmp.i64.i32(i32, i32, i32)
declare <4 x i32> @llvm.mxc.ldg.predicator.bsm.v4i32(ptr addrspace(3), ptr addrspace(1), i32, i64, i1, i1, i1, i1)
declare <2 x i32> @llvm.mxc.ldg.predicator.v2i32(ptr addrspace(1), i32, i64, i1, i1, i1, i1)
declare i32 @llvm.mxc.ldg.predicator.i32(ptr addrspace(1), i32, i64, i1, i1, i1, i1)
declare void @llvm.mxc.stg.predicator.i32(ptr addrspace(1), i32, i32, i64, i1, i1, i1)
declare void @llvm.mxc.arrive(i32) convergent nounwind
declare void @llvm.mxc.barrier.inst() convergent nounwind
declare void @llvm.mxc.barrier.shared() convergent nounwind
declare i32 @llvm.mxc.byte.perm(i32, i32, i32)

define internal i32 @__w8d_publish(i32 %offset, ptr addrspace(1) %src, i32 %pos, i32 %bound) alwaysinline nounwind convergent {
  %dst = getelementptr inbounds i8, ptr addrspace(3) @__w8d_shared, i32 %offset
  %pred = call i64 @llvm.mxc.icmp.i64.i32(i32 %pos, i32 %bound, i32 40)
  %copy = call <4 x i32> @llvm.mxc.ldg.predicator.bsm.v4i32(ptr addrspace(3) %dst, ptr addrspace(1) %src, i32 0, i64 %pred, i1 true, i1 true, i1 false, i1 true)
  ret i32 %offset
}

define internal i32 @__w8d_arrive_64(i32 %tid) alwaysinline nounwind convergent {
  call void @llvm.mxc.arrive(i32 64)
  call void @llvm.mxc.barrier.inst()
  ret i32 %tid
}

define internal i32 @__w8d_arrive_67(i32 %tid) alwaysinline nounwind convergent {
  call void @llvm.mxc.arrive(i32 67)
  call void @llvm.mxc.barrier.inst()
  ret i32 %tid
}

define internal i32 @__w8d_barrier(i32 %tid) alwaysinline nounwind convergent {
  fence syncscope("block") release
  call void @llvm.mxc.barrier.shared()
  fence syncscope("block") acquire
  ret i32 %tid
}

define internal <4 x i32> @__w8d_load4(i32 %offset) alwaysinline nounwind convergent {
  %p = getelementptr inbounds i8, ptr addrspace(3) @__w8d_shared, i32 %offset
  %v = load <4 x i32>, ptr addrspace(3) %p, align 16
  ret <4 x i32> %v
}

define internal i32 @__w8d_store4(i32 %offset, i32 %a, i32 %b, i32 %c, i32 %d) alwaysinline nounwind convergent {
  %v0 = insertelement <4 x i32> poison, i32 %a, i32 0
  %v1 = insertelement <4 x i32> %v0, i32 %b, i32 1
  %v2 = insertelement <4 x i32> %v1, i32 %c, i32 2
  %v3 = insertelement <4 x i32> %v2, i32 %d, i32 3
  %p = getelementptr inbounds i8, ptr addrspace(3) @__w8d_shared, i32 %offset
  store <4 x i32> %v3, ptr addrspace(3) %p, align 16
  ret i32 %offset
}

define internal i64 @__w8d_load2(i32 %offset) alwaysinline nounwind convergent {
  %p = getelementptr inbounds i8, ptr addrspace(3) @__w8d_shared, i32 %offset
  %v = load <2 x i32>, ptr addrspace(3) %p, align 8
  %bits = bitcast <2 x i32> %v to i64
  ret i64 %bits
}

define internal i64 @__w8d_scale_pair(i32 %valid, ptr addrspace(1) %src) alwaysinline nounwind convergent {
  %pred = call i64 @llvm.mxc.icmp.i64.i32(i32 %valid, i32 1, i32 32)
  %v = call <2 x i32> @llvm.mxc.ldg.predicator.v2i32(ptr addrspace(1) %src, i32 0, i64 %pred, i1 true, i1 true, i1 false, i1 false)
  %bits = bitcast <2 x i32> %v to i64
  ret i64 %bits
}

define internal i32 @__w8d_scale_one(i32 %valid, ptr addrspace(1) %src) alwaysinline nounwind convergent {
  %pred = call i64 @llvm.mxc.icmp.i64.i32(i32 %valid, i32 1, i32 32)
  %v = call i32 @llvm.mxc.ldg.predicator.i32(ptr addrspace(1) %src, i32 0, i64 %pred, i1 true, i1 true, i1 false, i1 false)
  ret i32 %v
}

define internal i32 @__w8d_epilogue(i32 %x, i32 %y, i32 %sa, i32 %sb0, i32 %sb1) alwaysinline nounwind {
  %a = bitcast i32 %sa to float
  %b0 = bitcast i32 %sb0 to float
  %b1 = bitcast i32 %sb1 to float
  %af0 = sitofp i32 %x to float
  %af1 = sitofp i32 %y to float
  %mul0 = fmul contract float %b0, %af0
  %mul1 = fmul contract float %b1, %af1
  %prod0 = fmul contract float %mul0, %a
  %prod1 = fmul contract float %mul1, %a
  %z0 = fadd contract float %prod0, 0.0
  %z1 = fadd contract float %prod1, 0.0
  %i0 = bitcast float %z0 to i32
  %i1 = bitcast float %z1 to i32
  %hi0 = lshr i32 %i0, 16
  %hi1 = lshr i32 %i1, 16
  %odd0 = and i32 %hi0, 1
  %odd1 = and i32 %hi1, 1
  %r0 = add i32 %i0, 32767
  %r1 = add i32 %i1, 32767
  %rne0 = add i32 %r0, %odd0
  %rne1 = add i32 %r1, %odd1
  %out = call i32 @llvm.mxc.byte.perm(i32 %rne0, i32 %rne1, i32 50464518)
  ret i32 %out
}

define internal i32 @__w8d_publish_result(i32 %valid, ptr addrspace(1) %dst, i32 %packed) alwaysinline nounwind convergent {
  %pred = call i64 @llvm.mxc.icmp.i64.i32(i32 %valid, i32 1, i32 32)
  call void @llvm.mxc.stg.predicator.i32(ptr addrspace(1) %dst, i32 0, i32 %packed, i64 %pred, i1 true, i1 false, i1 false)
  ret i32 %packed
}
