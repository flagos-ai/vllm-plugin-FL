#!/bin/bash
set -euo pipefail
root=/nvme2/minimax/experiments/flagos_s2_recovery_20261113
arch=/nvme2/minimax/experiments/flagos_s2_cache_arch_gate_20261026
base="$root/w8a8-submission-package-r1241"
pkg="$base/C500_Triton_r1241"
test "$(findmnt -n -o TARGET /nvme2)" = /nvme2
mkdir "$base"
mkdir -p "$pkg"/{arch/evidence/unified_direct_build_r860,evidence}
cp -a "$root/w8a8-submission-stage-r1239/"{plugin,overlay} "$pkg/"
cp -a "$root/gems_route_stage_r862/gems" "$pkg/gems"
cp -a "$root/w8a8-submission-input-r1239" "$pkg/kernels"
cp -a "$root/vendor-mixed-input-r1118" "$pkg/diag"
cp "$arch/evidence/unified_direct_build_r860/decode_only_merge_r860.py" "$pkg/arch/evidence/unified_direct_build_r860/"
cp "$root/w8a8-submission-smoke-r1240/"{requests.json,runtime_121.jsonl,w8a8_dispatch.jsonl,container.id} "$pkg/evidence/"
cp "$root/vendor-model-check-r1132/requests.json" "$pkg/evidence/control_requests.json"
cp "$root/w8a8-m2048-dispatch-r1237/report.json" "$pkg/evidence/m2048_native_dispatch.json"
cp "$root/w8a8-gate-m512-triton-r1234/report.json" "$pkg/evidence/gate_m512.json"
cp "$root/w8a8-gate-m2048-triton-r1236/report.json" "$pkg/evidence/gate_m2048.json"
for name in o qkv down; do
    cp "$root/w8a8-m2048-triton-r1238/$name/report.json" "$pkg/evidence/${name}_m2048.json"
done
cp "$root/w8a8-submission-input-r1239/run_packaged_w8a8_r1241.sh" "$pkg/run_smoke.sh"
cd "$pkg"
test "$(sha256sum arch/evidence/unified_direct_build_r860/decode_only_merge_r860.py | cut -d' ' -f1)" = 10ffb5a6a395b0c3009d05831b5f1777e5ffeb7efe2787f2015ff1e727358ab4
while IFS= read -r -d '' link; do
    resolved=$(realpath -e "$link")
    [[ "$resolved" == "$pkg/"* ]]
done < <(find . -type l -print0)
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum --quiet -c SHA256SUMS
tar -czf "$base/C500_Triton_r1241.tar.gz" -C "$base" C500_Triton_r1241
sha256sum "$base/C500_Triton_r1241.tar.gz"
wc -l SHA256SUMS
