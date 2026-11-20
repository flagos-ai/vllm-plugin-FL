The official FlagTree 0.7.1+xpu3.6 candidate gets past the earlier native bincount verifier failure, but the original case still fails exact integer numerics. This package records candidate execution evidence, not a compiler fix or completed native performance gate.

The native vLLM 0.32 source, original block 1024, synthetic input fixture, INT32 token IDs/counts, INT64 mapping and noncontiguous output views are unchanged. Plugin/provider discovery is disabled only to prove original native identity; no FlagGems-vllm or plugin provider is imported. The original image, Torch 2.9 and SDK remain unchanged, with the new compiler loaded from a separate prefix.

- Cold compilation and GPU execution complete; packed-mask backing output is exact.
- Counts differ in 225 of 688 backing elements, all in active visible columns. Inactive rows and padding remain exact.
- Active row 1 sums to 1024 rather than 50; row 3 sums to 1024 rather than 103. The original per-position output mask is not respected.
- Three independent original-fixture runs produce identical ELF and kernel-binary hashes, including the storage-only collection run. Successful public TTIR/TTXIR/LLVM string exports are empty; assembly is preserved and shows the missing output per-position guards.
- Warm acceptance was not run after the first cold failure. No performance timing, gate pass, production dependency update or issue closure is claimed. Unmeasured build/prototype plans are excluded from this packet.

Files retain the original synthetic data and source hashes. source-analysis.md separates observed output/assembly facts from the inferred source repair direction. provenance/official-wheel.json pins the actual full-wheel and loaded-core hashes, source tag and public delivery run. receipts/strict-result.json summarizes the three-run consistency; receipts/counts-snapshot.json contains full actual/expected integer backing tensors and input identities. Original/public file digests and any transformation are recorded in original-vs-public-hashes.json; SHA256SUMS covers every other file.

The original verifier issue remains [FlagTree #1701](https://github.com/flagos-ai/FlagTree/issues/1701). The separate grammar/penalties reshape issue [#1700](https://github.com/flagos-ai/FlagTree/issues/1700) is not resolved by these bincount results. Backend [FlagGems-vllm #1248](https://github.com/flagos-ai/FlagGems-vllm/pull/1248) still requires the full native-reference gate.
