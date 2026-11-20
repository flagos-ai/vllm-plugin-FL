# Marlin packing Gate, 2026-11-20

PR1347 e5c1a998ff233a0e5ab09f861fcb150ce4c11f7a; original packaged compiler.
Frozen source8332eb9ec26c1a3b8c06d23d409e50b101b8719ba8c7ad8d266037e9a3f3f43d.
Both target vLLM0.32/Torch2.13 and CI vLLM0.20.2/Torch2.11:
492 collected,37 added boundary tests pass; full316 pass/176 existing skips.
Original packing oracle and tolerances unchanged. Initial failed compile and
explicit uint32 shift repair are retained in verification-receipt.json.

All10 uncached packing timing rows are bitwise equal first, then faster than
the original Torch reference:1.73-7.18x. Three interleaved paired rounds,
JIT warm, actual allocation/computation each call, no pack-cache hits.
Large gateup weight12.64560 to6.18335ms (2.045x); peak2.02GiB.
GPU UUIDs are redacted in this public copy; receipt raw-file hashes identify
the original logs. These timings validate the packing migration only and do
not replace the frozen14 W8A16/native performance Gate.
