# vLLM-FL Ascend Custom Ops Tests

本目录包含 `vllm-plugin-FL` 在 Ascend NPU 上的自定义算子连接测试。

## 1. `csrc/ascend/` 下的两种算子接入方式

`csrc/ascend/` 目前承载两类 Ascend 自定义算子，它们的编译、安装和运行加载方式完全不同：

### 1.1 CANN Framework 算子（aclnn 路径）

- **源码位置**：`csrc/ascend/<category>/<op_name>/`
  - 例如 `csrc/ascend/moe/causal_conv1d/`、`csrc/ascend/attention/fused_gdn_gating/`。
- **构建工具链**：CANN `op_host` / `op_kernel` / `aclnn` 工具链。
- **产物**：自解压 `.run` 算子包，例如 `csrc/ascend/build/cann-ops-transformer-custom_linux-aarch64.run`。
- **安装位置**：默认隔离安装到项目目录 `vllm_fl/_cann_ops_custom/vendors/custom_transformer/`，不污染系统 CANN。
  - 该目录是构建产物，**不应提交到版本控制**。
- **运行时加载**：
  - C++ torch extension `vllm_fl._C_ascend` 注册 `torch.ops._C_ascend.*` schema；
  - `vllm_fl.utils.enable_custom_op()` 会自动发现已安装的 `_cann_ops_custom` 包，设置 `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH`；
  - 即使 `set_env.bash` 中写入了安装时的绝对路径，运行时也以 `vllm_fl` 包的实际位置为准，因此安装目录可以被移动。

### 1.2 PTO GDN 预编译算子（Bisheng 路径）

- **源码位置**：`csrc/ascend/pto_chunk_gdn/`。
- **构建工具链**：Bisheng C++ 编译器（`-xcce --cce-aicore-arch=dav-c220`），直接编译 `.cpp` 为 AI Core `.so`。
- **依赖头库**：`csrc/ascend/third_party/pto-isa/`。
- **产物**：多个 `mega_kernel_H*_Hg*_D*_C*.so`。
- **安装位置**：
  - 预编译模式：安装到 Python site-packages 下的 `vllm_fl/dispatch/backends/vendor/ascend/impl/pto_chunk_gdn/kernels/compiled_lib/`；
  - JIT 模式：首次调用时由 `vllm_fl/dispatch/backends/vendor/ascend/impl/pto_chunk_gdn/compile.py` 自动编译并缓存到同一目录。
- **运行时加载**：Python 代码通过 `ctypes.CDLL` / `torch.ops.load_library` 直接加载 `.so`，不经过 CANN `opp/vendors` 路径。

## 2. 环境准备

```bash
# 1. 激活 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 初始化 catlass 和 pto-isa 源码子模块（这两个第三方库不随主仓库一起拉取）
git submodule update --init --recursive csrc/ascend/third_party/catlass
git submodule update --init --recursive csrc/ascend/third_party/pto-isa

# 3. （可选）固定子模块版本，确保与上游来源一致
#    catlass 固定到 vllm-ascend 使用的 commit
#    pto-isa 固定到 PR #8872 使用的 commit（如有需要）
cd csrc/ascend/third_party/catlass
git checkout 41bf90da655bba3c66d0acd7e00abe33960ecfd6
cd ../../..

cd csrc/ascend/third_party/pto-isa
# 如需固定版本，替换为 PR #8872 对应的 commit
# git checkout <pto-isa-commit-hash>
cd ../../..

# 4. 确认环境变量（根据实际安装路径调整）
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export SOC_VERSION=ascend910b   # 根据实际芯片调整
```

## 3. 一键构建并运行（推荐）

项目提供了 `tests/custom_ops_tests/build_and_run.sh`，在激活 CANN 环境并初始化子模块后，可以一键完成扩展编译、CANN framework 算子检查以及全部测试：

```bash
bash tests/custom_ops_tests/build_and_run.sh
```

脚本会自动完成以下步骤：

1. 检查 `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` 是否已设置；
2. 检测 CANN 版本，要求 **CANN 9.0.0 及以上**，否则终止并提示安装；
3. 询问是否配置 GitHub / pip 镜像（分别对应 `ghfast.top` 和清华源）；
4. 检查并初始化 `catlass`、`pto-isa` 子模块；
5. 编译/安装 Python 包与 torch extension；
6. （可选）编译并安装 CANN framework 算子包；
7. 检查 CANN framework 算子包是否已安装；
8. source `set_env.bash` 并设置 `ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH`；
9. 检查 `FlagGems`，如果没有则从 `https://github.com/flagos-ai/FlagGems` clone 并以 editable 模式安装；
10. 依次运行 `tests/custom_ops_tests/test_*.py`。

### 常用选项

| 选项 | 说明 |
|---|---|
| `--build-ops` | 从源码编译并安装 CANN framework 算子包（执行 `csrc/ascend/build_aclnn.sh`）。 |
| `--soc <version>` | 指定芯片版本，如 `ascend910b`、`ascend910_93`；优先级高于环境变量 `SOC_VERSION`，默认由 `build_aclnn.sh` 决定。 |
| `--editable` / `-e` | 使用 `VLLM_VENDOR=ascend pip install --no-build-isolation -e .` 以 editable 模式安装（当前环境必须已有 `torch_npu`），而不执行 `build_ext --inplace`。 |
| `-h` / `--help` | 显示帮助信息。 |

示例：

```bash
# 默认：只编译 torch extension、检查已安装的算子包、跑测试
bash tests/custom_ops_tests/build_and_run.sh

# 一键完成所有事情：编译 extension + 编译安装 framework 算子 + 跑测试
bash tests/custom_ops_tests/build_and_run.sh --build-ops

# 指定 910C 芯片并启用 editable install
bash tests/custom_ops_tests/build_and_run.sh --build-ops --soc ascend910_93 --editable
```

> **注意**：脚本不会替你安装 CANN toolkit。如果第 7 步报错且你没有加 `--build-ops`，请先安装 CANN framework 算子包（见第 5 节），或者重新运行脚本并加上 `--build-ops`。

## 4. 编译并安装 torch extension `vllm_fl._C_ascend`

该 extension 把 C++ 算子实现注册到 `torch.ops._C_ascend`，同时包含 `camem_allocator` 等基础设施。

```bash
VLLM_VENDOR=ascend python setup.py build_ext --inplace
```

完成后会在项目根目录生成：

```text
vllm_fl/_C_ascend.cpython-311-aarch64-linux-gnu.so
vllm_fl/libvllm_fl_kernels.so
```

测试脚本中通过 `import vllm_fl._C_ascend` 加载 extension，随后即可调用 `torch.ops._C_ascend.*`。

## 5. 安装 CANN framework 算子包

如果已经存在构建好的 `.run` 包（例如 `csrc/ascend/build/cann-ops-transformer-custom_linux-aarch64.run`），可以直接安装：

```bash
bash csrc/ascend/build/cann-ops-transformer-custom_linux-aarch64.run \
    --install-path="$(pwd)/vllm_fl/_cann_ops_custom"
```

`.run` 包会把算子安装到指定的 `--install-path` 下，生成：

```text
vllm_fl/_cann_ops_custom/vendors/custom_transformer/
├── bin/set_env.bash          # 安装时生成的环境脚本
├── op_api/include/aclnn/     # aclnn 头文件
├── op_api/lib/libcust_opapi.so
├── op_proto/
└── op_impl/
```

> **注意**：`set_env.bash` 会记录安装时的绝对路径。测试脚本不依赖该脚本，而是调用 `vllm_fl.utils.enable_custom_op()` 根据 `vllm_fl` 包的实际位置自动设置环境变量，因此安装目录可以被移动。

如果需要从头编译 `.run` 包，执行：

```bash
bash csrc/ascend/build_aclnn.sh <soc_version>
# 例如：bash csrc/ascend/build_aclnn.sh ascend910b
```

构建完成后会生成 `csrc/ascend/build/cann-ops-transformer-custom_linux-aarch64.run` 并自动安装到 `vllm_fl/_cann_ops_custom/`。

如果希望清理构建过程中下载的第三方库（`abseil-cpp`、`ascend_protobuf`、`json`、`pkg` 缓存），保留源码子模块 `catlass` 和 `pto-isa`，可以加上 `--clean-third-party`：

```bash
bash csrc/ascend/build_aclnn.sh ascend910b --clean-third-party
```

## 6. PTO GDN 算子的两种使用方式

### 方式 A：预编译（推荐生产环境）

```bash
VLLM_VENDOR=ascend BUILD_PTO_CHUNK_GDN=ON python setup.py build_ext --inplace

# 显式编译 PTO megakernel
cmake --build build/temp.linux-aarch64-cpython-311 \
      --target pto_chunk_gdn_kernels -j$(nproc)
```

产物会安装到当前 Python 环境 site-packages 下的：

```text
vllm_fl/dispatch/backends/vendor/ascend/impl/pto_chunk_gdn/kernels/compiled_lib/
├── mega_kernel_H16_Hg8_D128_C128.so
├── mega_kernel_H16_Hg16_D128_C128.so
└── ...
```

> 在 editable install（`pip install -e .`）下，`_PACKAGE_ROOT` 等于仓库根目录，因此也会写到仓库内的 `vllm_fl/dispatch/backends/vendor/ascend/impl/pto_chunk_gdn/kernels/compiled_lib/`。

### 方式 B：JIT 首次编译（开发调试用）

不预编译，直接运行 `tests/custom_ops_tests/test_pto_chunk_gdn.py`。`vllm_fl/dispatch/backends/vendor/ascend/impl/pto_chunk_gdn/compile.py` 会：

1. 自动查找 `csrc/ascend/third_party/pto-isa`；
2. 调用系统 `bisheng` 编译对应配置的 `mega_kernel_*.so`；
3. 缓存到 `vllm_fl/dispatch/backends/vendor/ascend/impl/pto_chunk_gdn/kernels/compiled_lib/`；
4. 后续调用直接复用缓存。

## 7. 目录结构总览

```text
csrc/ascend/
├── CMakeLists.txt              # 构建 _C_ascend + PTO GDN 预编译
├── torch_binding.cpp           # torch.ops._C_ascend 注册
├── torch_binding_meta.cpp      # meta kernel 注册
├── camem_allocator.cpp         # NPU 显存分配器
├── build.sh                    # CANN framework 算子构建脚本
├── build_aclnn.sh              # 一键打包 + 安装 .run
├── build/                      # 构建产物（包含 .run 包）
│   └── cann-ops-transformer-custom_linux-aarch64.run
├── <category>/<op_name>/       # CANN framework 算子源码
│   └── op_host/op_kernel/...
├── pto_chunk_gdn/              # PTO GDN megakernel 源码
│   ├── CMakeLists.txt
│   ├── mega_kernel.cpp
│   └── include/
└── third_party/
    ├── catlass/                # CANN 算子依赖
    └── pto-isa/                # PTO 算子依赖
```

## 8. 如何执行测试

测试脚本会自动调用 `vllm_fl.utils.enable_custom_op()` 设置 CANN 自定义算子环境，**不需要手动 `source set_env.bash`**。

### 7.1 逐个运行

```bash
# CANN framework 算子
python tests/custom_ops_tests/test_causal_conv1d.py
python tests/custom_ops_tests/test_fused_gdn_gating.py
python tests/custom_ops_tests/test_gemma_rms_norm.py
python tests/custom_ops_tests/test_recurrent_gated_delta_rule.py
python tests/custom_ops_tests/test_chunk_gated_delta_rule_fwd_h.py

# PTO GDN 算子（首次运行会触发 Bisheng JIT 编译）
python tests/custom_ops_tests/test_pto_chunk_gdn.py
```

### 7.2 批量运行

```bash
for f in tests/custom_ops_tests/test_*.py; do
    echo "=== $f ==="
    python "$f" 2>&1 | tail -3
done
```

### 7.3 常见失败原因

| 现象 | 原因 | 解决 |
|---|---|---|
| `AttributeError: '_OpNamespace' '_C_ascend' object has no attribute 'xxx'` | `vllm_fl._C_ascend` 未编译或算子未注册 | 重新执行 `VLLM_VENDOR=ascend python setup.py build_ext --inplace` |
| `aclnnXxx ... not in libopapi.so` | `_cann_ops_custom` 未安装 | 执行 `bash csrc/ascend/build/cann-ops-transformer-custom_linux-aarch64.run --install-path="$(pwd)/vllm_fl/_cann_ops_custom"` |
| `ImportError: dynamic module does not define module export function (PyInit__C_ascend)` | `camem_allocator.cpp` 里的 PyInit 函数名与 extension 名不匹配 | 检查 `csrc/ascend/camem_allocator.cpp` 是否为 `PyInit__C_ascend` |
| PTO 测试提示找不到 `pto-isa` | 子模块未初始化或路径错误 | `git submodule update --init --recursive csrc/ascend/third_party/pto-isa` |

## 9. 测试脚本说明

| 测试脚本 | 对应算子 | 接入方式 |
|---|---|---|
| `test_causal_conv1d.py` | `npu_causal_conv1d_custom` | CANN framework |
| `test_fused_gdn_gating.py` | `npu_fused_gdn_gating` | CANN framework |
| `test_gemma_rms_norm.py` | `npu_gemma_rms_norm` | CANN framework |
| `test_recurrent_gated_delta_rule.py` | `npu_recurrent_gated_delta_rule` | CANN framework |
| `test_chunk_gated_delta_rule_fwd_h.py` | `chunk_gated_delta_rule_fwd_h` | CANN framework |
| `test_chunk_gated_delta_rule.py` | `npu_chunk_gated_delta_rule`（连接性） | CANN framework |
| `test_chunk_gated_delta_rule_accuracy.py` | `npu_chunk_gated_delta_rule`（数值精度，fp32 参考 + Triton 交叉校验） | CANN framework |
| `test_pto_chunk_gdn.py` | PTO GDN megakernel | Bisheng PTO |
