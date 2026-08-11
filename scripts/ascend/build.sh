#!/usr/bin/env bash
# ============================================================================
# build.sh — FlagOS Ascend 通用构建脚本 (clone + 构建 FlagCX/FlagTree/FlagGems + vllm)
#
# 本脚本不含任何本地环境假设, 可在任何 Ascend + CANN + conda 环境下运行。
# 本地 install.sh 负责 conda/CANN/系统依赖准备后调用此脚本。
#
# 用法 (由 install.sh 调用, 也可单独运行):
#   bash build.sh
#
# 前提: conda env 已激活, CANN set_env.sh 已 source, pip 可用
#
# 可选环境变量:
#   ROOT                  安装根目录 (默认: 脚本所在目录的父目录)
#   VLLM_PLUGIN_FL_REPO   vllm-plugin-FL 仓库地址
#   FLAGCX_REPO           FlagCX 仓库地址
#   FLAGTREE_REPO         FlagTree 仓库地址
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY=python3

# 仓库地址 (上游合并 PR 后可改为 flagos-ai/*)
VLLM_PLUGIN_FL_REPO="${VLLM_PLUGIN_FL_REPO:-https://github.com/Joiin0392/vllm-plugin-FL}"
VLLM_PLUGIN_FL_BRANCH="ascend-npu-support"
FLAGCX_REPO="${FLAGCX_REPO:-https://github.com/Joiin0392/FlagCX.git}"
FLAGCX_BRANCH="ascend-npu-support"
FLAGTREE_REPO="${FLAGTREE_REPO:-https://github.com/Joiin0392/FlagTree.git}"
FLAGTREE_BRANCH="cann851-compat"

echo "== FlagOS Ascend build (ROOT=$ROOT) =="

# ----------------------------------------------------------------------------
# 1. vllm 0.20.2 + torch_npu 2.11.0 版本对齐
# ----------------------------------------------------------------------------
pip uninstall -y vllm-ascend triton-ascend 2>/dev/null || true
pip install vllm==0.20.2
$PY - <<'PYEOF'
import subprocess, sys
from importlib.metadata import version, PackageNotFoundError
def get_ver(d):
    try: return version(d).split("+")[0]
    except PackageNotFoundError: return None
tv = get_ver("torch")
if tv != "2.11.0": raise SystemExit(f"[FAIL] torch={tv}, need 2.11.0")
nv = get_ver("torch-npu")
if nv and not nv.startswith("2.11.0"):
    print(f"[AUTO-FIX] torch_npu {nv} -> 2.11.0")
    subprocess.run([sys.executable,"-m","pip","install","-U","--no-deps","torch-npu==2.11.0"], check=True)
print(f"[OK] torch {tv} + torch_npu {get_ver('torch-npu') or 'none'}")
PYEOF

# ----------------------------------------------------------------------------
# 2. vllm-plugin-FL (fork: ascend-npu-support, 含 npu device context + head_dim>192 补丁)
# ----------------------------------------------------------------------------
if [ ! -d "$ROOT/vllm-plugin-FL" ]; then
  git clone -b "$VLLM_PLUGIN_FL_BRANCH" "$VLLM_PLUGIN_FL_REPO" "$ROOT/vllm-plugin-FL"
else
  (cd "$ROOT/vllm-plugin-FL" && git checkout "$VLLM_PLUGIN_FL_BRANCH" 2>/dev/null || true)
fi
(cd "$ROOT/vllm-plugin-FL" && pip install --no-build-isolation -e .)
echo "[OK] vllm-plugin-FL installed (branch: $VLLM_PLUGIN_FL_BRANCH)"

# ----------------------------------------------------------------------------
# 3. FlagCX (fork: ascend-npu-support, 含 ascend.mk + backend_flagcx.hpp + _build_config.py + flagcx_wrapper.py)
#    额外: torch_npu 捆绑 ACL 头文件替换 (pip 包级别, 非 git 仓)
# ----------------------------------------------------------------------------
if [ ! -d "$ROOT/FlagCX" ]; then
  git clone -b "$FLAGCX_BRANCH" "$FLAGCX_REPO" "$ROOT/FlagCX"
else
  (cd "$ROOT/FlagCX" && git checkout "$FLAGCX_BRANCH" 2>/dev/null || true)
fi
cd "$ROOT/FlagCX"
git submodule update --init --recursive 2>/dev/null || true

# 替换 torch_npu 捆绑 ACL 头文件为 CANN 版本 (torch_npu 2.11.0 的 ACL 头比 CANN 8.5.1 新)
ARCH=$(uname -m)
CANN_ACL_INC="$ASCEND_HOME_PATH/${ARCH}-linux/include/acl"
TNPU_INC=$($PY -c "import torch_npu,os;print(os.path.join(os.path.dirname(os.path.abspath(torch_npu.__file__)),'include','third_party','acl','inc','acl'))" 2>/dev/null)
if [ -n "$TNPU_INC" ] && [ -d "$TNPU_INC" ] && [ ! -L "$TNPU_INC" ]; then
    mv "$TNPU_INC" "${TNPU_INC}.bak"
    ln -sf "$CANN_ACL_INC" "$TNPU_INC"
    echo "[OK] torch_npu ACL headers -> $CANN_ACL_INC"
elif [ -L "$TNPU_INC" ]; then
    echo "[SKIP] torch_npu ACL headers already symlinked"
fi

# 构建 libflagcx.so + 安装 torch 插件
make USE_ASCEND=1 clean 2>/dev/null || true
make USE_ASCEND=1
cd "$ROOT/FlagCX/plugin/torch" && rm -rf build/
FLAGCX_ADAPTOR=ascend pip install --no-build-isolation .

# 把 libflagcx.so 拷到 flagcx Python 包目录, 让包自带 .so, 无需 FLAGCX_PATH
FLAGCX_PKG_DIR=$($PY -c "import flagcx, os; print(os.path.dirname(os.path.abspath(flagcx.__file__)))")
cp "$ROOT/FlagCX/build/lib/libflagcx.so" "$FLAGCX_PKG_DIR/libflagcx.so"
echo "[OK] libflagcx.so copied to $FLAGCX_PKG_DIR (auto-discovered, no FLAGCX_PATH needed)"

cd "$ROOT"
echo "[OK] FlagCX installed"

# ----------------------------------------------------------------------------
# 4. FlagGems (上游, 无补丁)
# ----------------------------------------------------------------------------
pip install -U scikit-build-core==0.11 pybind11 ninja cmake nanobind==2.4.0
if [ ! -d "$ROOT/FlagGems" ]; then
  git clone https://github.com/flagos-ai/FlagGems "$ROOT/FlagGems"
fi
(cd "$ROOT/FlagGems" && git checkout 3b2b55c8eda5de44ba3476d26566ecf134db0662 && pip install --no-build-isolation -e .)
echo "[OK] FlagGems installed"

# ----------------------------------------------------------------------------
# 5. FlagTree (fork: cann851-compat, 含 CANN 8.5.1 bishengir 兼容补丁)
#    x86_64 系统: 需下载 enflame x64 LLVM 22 (ascend 后端只有 aarch64)
# ----------------------------------------------------------------------------
if [ ! -d "$ROOT/FlagTree" ]; then
  git clone -b "$FLAGTREE_BRANCH" "$FLAGTREE_REPO" "$ROOT/FlagTree"
else
  (cd "$ROOT/FlagTree" && git checkout "$FLAGTREE_BRANCH" 2>/dev/null || true)
fi
cd "$ROOT/FlagTree"
git submodule update --init --recursive 2>/dev/null || true
rm -rf build dist *.egg-info /root/.flaggems /root/.triton/llvm /root/.triton/nvidia /root/.triton/json
rm -rf /root/.flagtree/ascend/llvm-7d5de303-ubuntu-aarch64-python311-compat

# x86_64 LLVM 22 预编译包
if [ "$(uname -m)" = "x86_64" ]; then
  LLVM_X64_DIR=/root/.flagtree/ascend/llvm-x64
  if [ ! -x "$LLVM_X64_DIR/bin/mlir-tblgen" ]; then
    echo "[INFO] Downloading x86_64 LLVM 22 ..."
    mkdir -p /root/.flagtree/ascend
    curl -L -o /tmp/llvm-x64.tar.gz "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm22-189e06b-gcc9-x64_v0.4.0.tar.gz"
    mkdir -p /tmp/llvm-x64-extract && tar xzf /tmp/llvm-x64.tar.gz -C /tmp/llvm-x64-extract
    cp -r /tmp/llvm-x64-extract/llvm-189e06b-gcc9-x64 "$LLVM_X64_DIR"
    rm -rf /tmp/llvm-x64.tar.gz /tmp/llvm-x64-extract
  fi
  export LLVM_SYSPATH="$LLVM_X64_DIR"
  export PYTHONPATH="$LLVM_X64_DIR/python_packages/mlir_core:${PYTHONPATH:-}"
  echo "[OK] LLVM 22 ready at $LLVM_X64_DIR"
fi

# clang 自检 (FlagTree 构建 _C.so 需要)
if ! command -v clang >/dev/null 2>&1; then
  vclang=$(ls /usr/bin/clang-[0-9]* /usr/local/bin/clang-[0-9]* 2>/dev/null | sort -V | tail -1)
  vclangxx=$(ls /usr/bin/clang++-[0-9]* /usr/local/bin/clang++-[0-9]* 2>/dev/null | sort -V | tail -1)
  if [ -n "$vclang" ] && [ -n "$vclangxx" ]; then
    ln -sf "$vclang" /usr/local/bin/clang; ln -sf "$vclangxx" /usr/local/bin/clang++
  elif command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
    apt-get install -y clang-17 2>/dev/null || apt-get install -y clang
    [ -f /usr/bin/clang-17 ] && { ln -sf /usr/bin/clang-17 /usr/local/bin/clang; ln -sf /usr/bin/clang++-17 /usr/local/bin/clang++; }
  else
    echo "[FAIL] clang not found. Install: apt install clang-17"; exit 1
  fi
fi
echo "[OK] clang: $(clang --version 2>/dev/null | head -1)"

# 构建 FlagTree
env FLAGTREE_BACKEND=ascend LLVM_SYSPATH="${LLVM_SYSPATH:-}" PYTHONPATH="${PYTHONPATH:-}" \
  $PY -m pip install . --no-build-isolation -v --root-user-action ignore
cd "$ROOT"
echo "[OK] FlagTree installed (branch: $FLAGTREE_BRANCH)"

# ----------------------------------------------------------------------------
# 6. 冒烟测试 (FlagGems + Triton kernel + FlagCX 通信器)
# ----------------------------------------------------------------------------
FLAGCX_PATH="${FLAGCX_PATH:-$ROOT/FlagCX}" $PY "$SCRIPT_DIR/smoke_test.py"

echo ""
echo "== FlagOS build complete =="
echo "FLAGCX_PATH=$ROOT/FlagCX"
