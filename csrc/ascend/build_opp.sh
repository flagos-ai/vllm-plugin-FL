#!/usr/bin/env bash
# Build the FL-local CANN custom operators used by Qwen3.6 on A2/A3.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "${script_dir}/../.." && pwd)
# A3 is the no-override delivery default. A2 uses the common family selector
# SOC_VERSION=ascend910b for B1/B2/B3/B4.
soc_version=${SOC_VERSION:-ascend910_93}

case "${soc_version}" in
  ascend910b*|910b)
    soc_family=ascend910b
    prebuilt_dir=ascend910b1
    ;;
  ascend910_93*|910c)
    soc_family=ascend910_93
    prebuilt_dir=ascend910_93
    ;;
  *)
    echo "Unsupported Qwen3.6 Ascend SOC_VERSION: ${soc_version}" >&2
    exit 2
    ;;
esac

catlass_root=${CATLASS_PATH:-${repo_dir}/.deps/catlass}
catlass_include=${catlass_root}/include
catlass_commit=41bf90da655bba3c66d0acd7e00abe33960ecfd6
if [[ ! -d "${catlass_include}/catlass" ]]; then
  mkdir -p "$(dirname "${catlass_root}")"
  if [[ ! -d "${catlass_root}/.git" ]]; then
    git clone --filter=blob:none https://gitcode.com/cann/catlass.git "${catlass_root}"
  fi
  git -C "${catlass_root}" fetch origin "${catlass_commit}"
  git -C "${catlass_root}" checkout --detach "${catlass_commit}"
fi
export CPATH="${catlass_include}${CPATH:+:${CPATH}}"

ops=(
  add_rms_norm_bias
  causal_conv1d
  recurrent_gated_delta_rule
  chunk_fwd_o
  chunk_gated_delta_rule_fwd_h
  moe_gating_top_k
  moe_init_routing_custom
)
ops_csv=$(IFS=,; echo "${ops[*]}")

cd "${script_dir}"
bash build.sh \
  --pkg \
  --soc="${soc_family}" \
  --vendor_name=custom \
  --ops="${ops_csv}"

installer_candidates=(build/cann-ops-transformer*.run)
if [[ ${#installer_candidates[@]} -ne 1 || ! -f "${installer_candidates[0]}" ]]; then
  echo "Expected one CANN OPP installer, found ${#installer_candidates[@]}" >&2
  exit 3
fi

install_root="${repo_dir}/vllm_fl/dispatch/backends/vendor/ascend/prebuilt/${prebuilt_dir}/opp"
mkdir -p "${install_root}"
chmod +x "${installer_candidates[0]}"
"${installer_candidates[0]}" --install-path="${install_root}"
