# XPU Python-module conceal option

`TRITON_XPU_CONCEAL_IR=OFF` disables concealment on the `TritonXPU` target but the Python `triton` target still receives `TRITON_CONCEAL_IR=1`. Consequently its public module string/dump methods remain empty and pass-manager debug output is disabled. Apply the existing option to that target too.

Validation on current main `bb8a53cc427fb426b65e38bfdd00d0f4217af686`: the complete real `FlagTreeOptions.cmake` module configured and compiled a tiny target for XPU default ON, explicit ON/OFF, NVIDIA ON/OFF, and Ascend ON. All six contracts passed. Original main with XPU OFF failed the same negative-control contract. This checks the real target macro, not a full current-main compiler build.

Full compiler validation used release source `e6a8799cc14819def1bdb8fe68b018fd774fe4c9` plus only the same one-line change (`49ce384960a4f716a82445ff49b42549f4d15b23`). Default ON actually compiled the XPU Python binding with `-DTRITON_CONCEAL_IR=1`; OFF compiled it without that definition or an undefine workaround. The OFF wheel built successfully, all 1,234 wheel RECORD entries verified, and standard native compilation produced a 980,018-byte MLIR pass dump. The native kernel's ELF and XPU binary stayed byte-identical to the official 0.7.1 wheel and the unmodified source control.

This is a diagnostic build fix. Native bincount still has 225/688 incorrect counts; no numerical or performance Gate is claimed by this change. The numerical issue is tracked in [FlagTree #1701](https://github.com/flagos-ai/FlagTree/issues/1701).

The isolated full builds share the nine dependency payloads recorded in the runtime receipt. The official LLVM19 download currently disagrees with the source MD5 prefix, so `pin_mismatch=true` is retained and an exact official release rebuild is not claimed; see [FlagTree #1703](https://github.com/flagos-ai/FlagTree/issues/1703). Existing production environments and dependencies were not replaced.
