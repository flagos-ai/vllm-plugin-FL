# MHC actual-source Gate, 2026-11-20

FlagGems-vllm PR #1350 head aa3e24a894988f2631bd14f4f75107956c55347f.
Actual source SHA256 7d2413678bdc6ce4ad6c230e484c629a321d0fb8446fd92c18fd37a620b1f19b.
Original packaged FlagTree 0.7 / Triton 3.6, NVIDIA H20-3e.

Both target vLLM 0.32/Torch 2.13 and CI vLLM 0.20/Torch 2.11
pass 26 norm tests cold and warm. The full nearby collection is 214:
target181 passed/33 existing skips; CI191 passed/23 existing skips.
All original 15 safe configs x12 shapes x3 repeats pass coefficient checks.
No added register-cap configuration remains in this source.

Native target performance uses the unchanged 12 workloads, 0.9 gate,
2e-5/0.01 numerical contracts and three paired rounds with30ms warmup/80ms rep.
All12 pass; minimum0.947433866. A fresh reversed-order critical case passes0.947311121.
The libtuner-only ablation retaining original splits passes0/3 prior deficits.
Raw rounds, sources, numerical checks and ablation are retained here.

This qualifies normal version-tracked weights and single-GPU operators.
Versionless-weight repacking performance and FlagCX Python collective ABI
are outside this frozen performance set. The PR's automatic full CI is red
at shared KDA contract failures followed by Marlin illegal-instruction
cascades. Latest-head full integrated CI must still be rerun after shared repairs.
