# Isolated FlagTree1540 complete build and limited GPU validation

Exact signed CI merge6ed3ec13; no compiler source patch. See manifest for
source/tree/LLVM pins, tools and full build provenance. Build1 wheel is
407551897bytes, SHA256fe97e2d3931a583e27e9d907866c5a055d26989568ab73d6467d11d21db8049c.
Corelibtriton SHA9026d97c42075b0c0125bfbedf6e375a1fa16bc65fa37de26eaae03add51db78.

The first build passed CPU import but omitted CUDA runtime assets due to
setup external-tool path behavior. Early cuda.h/ptxas failures were packaging
failures, not numerical results. The complete build copies173 matching pinned
assets; RECORD, every asset hash, default4tool lookup with overrides cleared,
and CPU ABI are verified. Original incomplete build/prefix are preserved.

GPU: original Marlin public small BF16/INT8 test, unchanged assertions,
passes all11 original stage3 tile configurations in separate contexts.
The complete package plus production safe Hopper table and source11622d7
passes full279/176 existing skips on target0.32/Torch2.13. Its elapsed time
includes an intentional pause for independent packing timing, so is not a
performance measurement. A broad original-generic-tuning candidate and
frozen14 performance Gate remain pending. This does not qualify FP8 all
candidates, all shapes, MHC1692, or FlagCX Python collective ABI.

No claim that PR1540 alone caused the change: the controlled rebuilt runtime
has an explicit source+LLVM/toolchain provenance different from the opaque
packaged baseline. Production compiler/image are not replaced.
