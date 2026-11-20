# FlagTree #1706: local MM dependency control

The dependency-complete local control now passes all original five cases on
both compilers, using the exact first-attempt inputs/RNG and unchanged oracle.
Real GEMV/TMA kernels and compiler artifacts are recorded in
`official-extras-paired-five-pass.json`. The task ended naturally in53.944s;
source, owned-container removal, device idle/PID0 and both released locks were
verified. The original dependency-blocked attempt below remains a failure.

Three independent main-branch baseline runs (fc0ec83b,343d64c0,9a782d9f)
reproduce the same five MM regressions:576pass/6fail/18skip out of600, versus
the stored best581pass/1fail/18skip. See `main-baseline-reproductions.json`.
Those workflows report success while their comparison records REJECT; this is
not full-Gate acceptance. Their logs do not expose tracebacks or dependency
identity, so the local missing-extra cause is not proven to be the CI cause.

The NVIDIA CI comparison reported five new MM/MM-out failures against a stored
baseline (576 pass, six fail, 18 skip). Its log has no tracebacks or runtime core
identity. This local experiment does not establish that CI has the same cause.

On an H20-3e with Torch 2.13/CUDA 13, the original five FlagGems v5.3.5 cases were
run with complete local FlagTree builds fc0ec83b and 6ed3ec13, the versioned CI
conftest, unchanged assertions, and identical input bytes/initial RNG per case.
Both variants fail all five before MM numeric comparison: the FlagTune model
loader cannot import xgboost. No original MM kernel launches; the stride-zero
case alone launches its ordinary copy helper. Focused RNG does not reconstruct
the unknown preceding 600-case CI RNG stream. Neither build is the CI binary.

The original aggregate remains INCOMPLETE_OR_HARNESS_INVALID: it also rejected
different manifest.meta fetched_at timestamps. The payload manifest is identical;
the complete original metadata is retained here. No old result was rewritten.
Exact owned-container removal and target-device/lock release were verified.

The official four FlagTune extra specifications were resolved normally into a
new private directory. All ten official wheels match their PyPI SHA256; CPU
imports succeed, with 3,733 installed RECORD hashes verified. The sole normal
pip --target f2py entrypoint relocation is recorded without editing RECORD.
This preparation imports no Torch and has no GPU devices. It does not pass MM.

The required NumPy 1.26.4 conflicts with NumPy>=2 declarations of the unchanged
local-image CuPy 14.2, ml-dtypes 0.6 and OpenCV 5. Those packages were not changed;
image-wide dependency consistency is not claimed. A subsequent original-case
control must preserve and report any real import/ABI failure. NCCL-cu12 from the
official XGBoost requirement coexists with the original image's NCCL-cu13 and
must be fingerprinted if loaded. No test selection change for full acceptance,
relaxed tolerance, forced tuning winner, production-image modification, or
successful 600-case/performance Gate follows from these focused controls.
