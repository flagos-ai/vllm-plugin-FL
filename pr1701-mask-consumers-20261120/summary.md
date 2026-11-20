# XPU masked-load consumer failures

Two standalone XPU test kernels compile and execute but fail exact INT32 checks because `TritonXPUMask` moves a later **unmasked** store under the masked-load condition. The false lanes have a defined `other`, so their consumers must still execute.

This is separate from #1701's residual scalar `i1` DMA predicate: these two cases retain an INT32 length and/or defined `other`, and the non-atomic case has `atomicSim=false`.


See receipt.json, the unchanged eight-case test file, raw pytest log/XML, and the two pre-LLVM modules. They are control failures, not candidate/regression or Gate passes.
