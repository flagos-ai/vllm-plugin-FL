# P800 public SDK integer-atomic interface probes

These are bounded, CPU-only invocations of the unmodified public XPU LLVM19
compiler. No GPU kernel was executed and no production package or SDK was
modified. The same target/ABI accepted an empty kernel and the ordinary DMA
read/add/write sequence used by AtomicSim, but rejected standard INT32 global
atomic ADD (monotonic/acq_rel) and OR (monotonic) during instruction selection.

The result scopes an unsupported compiler interface. It does not establish that
the P800 hardware cannot implement integer atomics or that no other vendor API
exists. The corresponding device library exposes FP16/FP32 Add/Mul names but no
typed INT32 ADD/OR/CAS entry point was observed in its defined-symbol list.

The native 2048 count failure remains tracked in FlagTree #1701. A complete
repair needs a documented, correctly lowered device-scope integer RMW primitive
with returned old values and ordering, or a proven shared synchronization ABI
across logical programs, streams, and separately loaded JIT modules. Merely
serializing one kernel's grid cannot establish that complete contract.

Source/commands, both successful controls, failing diagnostics, exact llc and
device-library hashes, and the already disclosed SDK pin deviation are in the
included files. No performance or G6 pass is claimed.
