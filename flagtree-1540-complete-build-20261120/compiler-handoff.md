# Complete isolated FlagTree1540 compiler artifact

Use `/data/codex/flagtree-1540-20261120/compiler-complete` read-only. Version is unchanged FlagTree0.7.0+git6ed3ec13 / Triton3.6; source merge6ed3ec13/tree11ad7f04 and libtriton SHA9026d97c42075b0c0125bfbedf6e375a1fa16bc65fa37de26eaae03add51db78 are unchanged.

The build1wheel is `/data/codex/flagtree-1540-20261120/wheels-complete/flagtree-0.7.0+git6ed3ec13-1-cp312-cp312-linux_x86_64.whl`, 407551897 bytes, SHA256fe97e2d3931a583e27e9d907866c5a055d26989568ab73d6467d11d21db8049c.

The initial349697223-byte wheel SHA302f4b5b... and original `compiler` prefix remain preserved. They passed CPU imports but omitted NVIDIA bin/include/lib/cupti and were insufficient for GPU compilation. The build used external TRITON_* paths, and setup.py's download_and_copy returned early instead of packaging those assets. The size difference from the original CI wheel cannot be attributed only to LLVM.

The complete artifact adds173 files/213086900 bytes from the unchanged baseline NVIDIA asset directories. Every asset hash matches baseline; all337 original Python and shared-library files match the first artifact. Four default tools match source-pinned versions: ptxas12.8.93, Blackwellptxas12.9.86, cuobjdump/nvdisasm12.8.55. Wheel RECORD hashes were verified after standard wheel pack --build-number1; no C++ rebuild or GPU task was run.

```bash
-v /data/codex/flagtree-1540-20261120/compiler-complete:/compiler1540:ro \
-v /data/codex/flaggems-vllm-h20-marlin-0d86ec1/source/.venv/lib/python3.12/site-packages/flagcx:/flagcx-dependency/flagcx:ro \
-e PYTHONPATH=/compiler1540:/flagcx-dependency:/candidate/src:/candidate:/deps
```

Keep the original target image/deps and CUDA13 LD_LIBRARY_PATH. Import torch before triton/TLE, use fresh caches, and assert the new triton path and external FlagCX SHA c768f5fa077c6d932eb23542131bc3a5ae436b0beebb96bac9581aed58279b48. No external CUDA include/tool knob or CPATH override is needed: CPU ABI verified all four default resolvers point within compiler-complete, cuda.h is present, and0 GPUs were visible. The external CUDA13 FlagCX requirement is unchanged; the bundled legacy CUDA12 fallback is not used.

This is a packaging/CPU validation result. GPU numerical and performance Gates, and causal attribution to PR1540, remain separate.
