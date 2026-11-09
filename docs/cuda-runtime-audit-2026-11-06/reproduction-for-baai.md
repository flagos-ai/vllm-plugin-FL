# CUDA 13.3 Runtime 安装 vLLM 0.32.0+flagos 的复现步骤

测试日期：2026-11-06。本文供 Runtime / wheel 维护方核对，记录已执行的实验、版本和原始错误。文中不包含登录密码。

## 1. 问题摘要

在指定镜像 `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3:2.3.0` 中，安装甲方提供的 `vllm-0.32.0+flagos-cp312-cp312-linux_x86_64.whl` 及其声明的依赖，再安装 `vllm-plugin-FL` 的 `0.5.0-dev` 对应代码：

- wheel 和插件均安装成功，`import vllm`、`import vllm_fl` 通过。
- 保留 Runtime 的 `numpy==1.26.4` 后，定制 OpenCV 导入及 resize 数值检查通过，**没有再出现官方 OpenCV 的 NumPy>=2 解析冲突**。
- wheel 仍声明 `setuptools>=78.1.1,<81.0.0`，安装过程中 setuptools 从 Runtime 的 84.2.1 降为 80.10.2。
- 启用 FL NVIDIA 平台后，`from vllm import LLM` 报 `ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'`，推理未进入模型加载。
- UT 入口还会报缺少 pytest；不能据此声称已执行或通过整个 CI。

**请优先复现第 7 节的平台导入错误。** 它独立于 pytest 是否安装，且安装插件前直接导入上游 LLM 时也能出现。

## 2. 测试机器与基础环境

| 项目 | 实测值 |
|---|---|
| 机器入口 | `bastion.aiops.baai.ac.cn:2224`，目标实例 `341d2313-1f18-414c-92eb-0e114a9c966f` |
| 实际 shell hostname | `dsw-1151459-64c4bb5cc8-cqxts` |
| GPU | 8 × NVIDIA H20-3e，每卡 143771 MiB |
| 宿主 NVIDIA 驱动 | 580.95.05 |
| 容器架构 | Linux x86_64 |
| 基础镜像 | `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3:2.3.0` |
| Python | 3.12.13，`/flagos/bin/python` |
| 默认编译器 | FlagTree 0.7.0，活动 Triton 3.6.0，路径 `/opt/flagtree/triton` |
| 备用编译器 | Triton 3.7.1，位于 `/opt/triton`；本次没有切换 |

[Runtime 官方说明](https://flagos-ai.github.io/release-info/runtime/nvidia-cuda13.3/)。说明列出的宿主驱动为 610.43.02，与本机不同；本文记录实际驱动，不把两者混称。当前缺少 Python 扩展文件的错误发生在平台导入阶段。

实际使用的 SSH 命令：

```bash
ssh -p 2224 \
  'htma#root#341d2313-1f18-414c-92eb-0e114a9c966f@bastion.aiops.baai.ac.cn'
```

按已有账号权限交互输入密码。以下命令均在该 H20 机器上执行。

## 3. 仓库、分支与代码提交

| 项目 | 值 |
|---|---|
| 组内仓库 | https://github.com/BrianPei/vllm-plugin-FL |
| 个人开发分支 | `codex/cuda-runtime-2.3.0-0.5.0-dev` |
| 组内基线分支 | `0.5.0-dev` |
| 官方上游 | https://github.com/flagos-ai/vllm-plugin-FL/tree/0.5.0-dev |
| 实测固定提交 | `4b3a76a19de52af9981237e860c2177e73b9ee68` |

本次实测时个人分支、组内 `0.5.0-dev` 和官方 `0.5.0-dev` 均指向此提交；安装和启动实验未修改插件源代码、Dockerfile 或 CI 检查。

原始实验目录与容器：

```text
宿主证据目录：/data/codex-cuda-runtime-audit-20261106
插件 checkout：/data/codex-cuda-runtime-audit-20261106/repo
wheel 与日志：/data/codex-cuda-runtime-audit-20261106/modified-wheel
测试容器名：codex-cuda-flagos-wheel-20261106
容器内工作目录：/audit/repo
```

下文按实际执行顺序整理命令。为避免覆盖已有实验，复核示例使用新的目录和容器名；代码提交、镜像、安装包、版本保护约束和启动环境保持相同。镜像使用 `:2.3.0` 标签，不固定镜像 digest；若标签后续被重新上传，应记录拉取时间和包版本再比较结果。

## 4. 准备源码、wheel 和干净容器

```bash
# 在 H20 宿主 shell 设置；保持在同一个 shell 中执行后续命令。
REPRO_DIR=/data/vllm032-flagos-repro-20261106
REPRO_CONTAINER=vllm032-flagos-repro-20261106
RUNTIME_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3:2.3.0

mkdir -p "$REPRO_DIR/modified-wheel"
git clone --branch codex/cuda-runtime-2.3.0-0.5.0-dev \
  https://github.com/BrianPei/vllm-plugin-FL.git "$REPRO_DIR/repo"
git -C "$REPRO_DIR/repo" checkout --detach \
  4b3a76a19de52af9981237e860c2177e73b9ee68
git -C "$REPRO_DIR/repo" rev-parse HEAD

# 使用已有 Harbor 账号，按提示交互输入密码。
docker login harbor.baai.ac.cn
docker pull "$RUNTIME_IMAGE"

curl -fSL --retry 2 \
  'https://resource.flagos.net/repository/flagos-pypi-nvidia/packages/vllm/0.32.0%2Bflagos/vllm-0.32.0%2Bflagos-cp312-cp312-linux_x86_64.whl' \
  -o "$REPRO_DIR/modified-wheel/vllm-0.32.0+flagos-cp312-cp312-linux_x86_64.whl"

docker run -d --name "$REPRO_CONTAINER" \
  --gpus all --shm-size 32g \
  -v "$REPRO_DIR:/audit" \
  -v /data:/data \
  -w /audit/repo \
  "$RUNTIME_IMAGE" bash -lc 'sleep infinity'

docker inspect "$REPRO_CONTAINER" \
  --format 'BASE_IMAGE={{.Config.Image}} STATUS={{.State.Status}}'
docker exec "$REPRO_CONTAINER" bash -lc \
  'python --version; command -v python; nvidia-smi'
```

实际 wheel 的 [Nexus 浏览页](https://resource.flagos.net/#browse/browse:flagos-pypi-nvidia:vllm%2F0.32.0%2Bflagos%2Fvllm-0.32.0%2Bflagos-cp312-cp312-linux_x86_64.whl)。实际下载地址比浏览页多 `/repository/.../packages/` 路径。

## 5. 检查 wheel 内容和声明的依赖

```bash
docker exec -i "$REPRO_CONTAINER" bash -lc 'python -' <<'PY'
import email
import hashlib
import zipfile
from pathlib import Path

p = Path('/audit/modified-wheel/vllm-0.32.0+flagos-cp312-cp312-linux_x86_64.whl')
with zipfile.ZipFile(p) as z:
    names = z.namelist()
    metadata = email.message_from_bytes(z.read(next(
        n for n in names if n.endswith('.dist-info/METADATA')
    )))
    print('bytes:', p.stat().st_size)
    print('wheel sha256:', hashlib.sha256(p.read_bytes()).hexdigest())
    print('ZIP CRC error:', z.testzip())
    print('Name:', metadata['Name'])
    print('Version:', metadata['Version'])
    print('file count:', len(names))
    print('native files:', [
        n for n in names if '.so' in n or n.endswith(('.pyd', '.dll'))
    ])
    print('Requires-Dist:')
    print('\n'.join(metadata.get_all('Requires-Dist', [])))
PY
```

实测关键输出：

```text
bytes: 11446273
wheel sha256: 4b8f26d6c8e75aa97362948af212dfc32d77e7e58a245a2ee032279c7051235e
ZIP CRC error: None
Name: vllm
Version: 0.32.0+flagos
file count: 3301
native files: []
```

这里的哈希只用于比对 wheel 文件是否为同一个文件，不是镜像引用。

与官方 vLLM 0.32.0 相比，该 wheel 改为声明以下定制依赖：

```text
opencv-python-headless==5.0.0.93+flagos
xgrammar==0.2.8+flagos
compressed-tensors==0.17.0+flagos
```

它仍声明 `setuptools>=78.1.1,<81.0.0`，但去除了官方 CUDA requirements 对应的多项 Requires-Dist，包括 Torch、TorchVision、TorchAudio、FlashInfer、TorchCodec 等。**去掉依赖声明不等于把这些组件或原生 CUDA 扩展打进 wheel。**

## 6. 安装修改版 wheel 和插件

为尽量保留 Runtime 原有核心包，本次实际使用以下约束：

```bash
cat > "$REPRO_DIR/protected.txt" <<'EOF'
torch==2.13.0+cu130
torchvision==0.28.0+cu130
torchaudio==2.11.0+cu130
numpy==1.26.4
flag-gems==5.4.0
flagtree==0.7.0
flagcx==0.14.1+cuda13.3
EOF

# 先只解析，不安装。
docker exec "$REPRO_CONTAINER" bash -lc \
  'uv pip install --dry-run --python /flagos/bin/python \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    --extra-index-url https://resource.flagos.net/repository/flagos-pypi-nvidia/simple \
    --index-strategy unsafe-best-match \
    --constraint /audit/protected.txt \
    /audit/modified-wheel/vllm-0.32.0+flagos-cp312-cp312-linux_x86_64.whl' \
  > "$REPRO_DIR/modified-wheel/dry-run.log" 2>&1

# 实际安装，安装包版本及传递依赖以当时索引解析结果为准。
docker exec "$REPRO_CONTAINER" bash -lc \
  'uv pip install --python /flagos/bin/python \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    --extra-index-url https://resource.flagos.net/repository/flagos-pypi-nvidia/simple \
    --index-strategy unsafe-best-match \
    --constraint /audit/protected.txt \
    /audit/modified-wheel/vllm-0.32.0+flagos-cp312-cp312-linux_x86_64.whl' \
  > "$REPRO_DIR/modified-wheel/install.log" 2>&1
printf 'wheel_install_exit=%s\n' "$?"

# 按当前 NVIDIA setup 的方式，仅安装插件 checkout，不自动解析其依赖。
docker exec "$REPRO_CONTAINER" bash -lc \
  'env -u VLLM_VENDOR python -m pip install \
    --no-deps --no-build-isolation -e /audit/repo' \
  > "$REPRO_DIR/modified-wheel/plugin-install.log" 2>&1
printf 'plugin_install_exit=%s\n' "$?"
```

本次两次实际安装退出码均为 0。未执行整个 `.github/scripts/nvidia/setup.sh`，因为它还包含旧镜像来源检查和其他包安装；上述方式是为单独验证甲方 wheel 与当前插件的兼容性。

安装后核对包版本：

```bash
docker exec -i "$REPRO_CONTAINER" bash -lc 'python -' <<'PY'
from importlib.metadata import version
for name in (
    'vllm', 'numpy', 'setuptools', 'torch', 'torchvision', 'torchaudio',
    'flag-gems', 'flagtree', 'flagcx', 'opencv-python-headless',
    'xgrammar', 'compressed-tensors',
):
    print(f'{name}=={version(name)}')
PY
```

实测结果：

```text
vllm==0.32.0+flagos
numpy==1.26.4
setuptools==80.10.2
torch==2.13.0+cu130
torchvision==0.28.0+cu130
torchaudio==2.11.0+cu130
flag-gems==5.4.0
flagtree==0.7.0
flagcx==0.14.1+cuda13.3
opencv-python-headless==5.0.0.93+flagos
xgrammar==0.2.8+flagos
compressed-tensors==0.17.0+flagos
```

对初始 Runtime 的全部 distribution 比较后，原有包只有 setuptools 从 84.2.1 变为 80.10.2。其他变化为新增应用依赖。由于本次没有生成完整 lockfile，未来复现时传递依赖的非固定版本可能变化；原始完整包状态保留在随附证据中。

## 7. 复现当前 NVIDIA 插件的平台启动错误

这一步不需要下载模型，也不需要 pytest：

```bash
docker exec \
  -e VLLM_PLUGINS=fl \
  -e USE_FLAGGEMS=1 \
  -e GEMS_VENDOR=nvidia \
  -e USE_FLAGTUNE=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  "$REPRO_CONTAINER" bash -lc \
  'python -c "from vllm import LLM; print(LLM)"' \
  > "$REPRO_DIR/modified-wheel/plugin-import.log" 2>&1
printf 'import_exit=%s\n' "$?"
cat "$REPRO_DIR/modified-wheel/plugin-import.log"
```

实测退出码为 1，关键异常链为：

```text
  File "/audit/repo/vllm_fl/nvidia_platform.py", line 10, in <module>
    from vllm.platforms.cuda import CudaPlatform
  File "/flagos/lib/python3.12/site-packages/vllm/platforms/cuda.py", line 23, in <module>
    import vllm._C_stable_libtorch  # noqa
ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'
```

也执行过先 `import vllm_fl`，再获取 `vllm.platforms.current_platform` 的检查。前者输出 `PLUGIN_IMPORT_OK`，后者报相同错误。设置了群里提到的 `VLLM_USE_V2_MODEL_RUNNER=1` 后仍然复现。

对应的插件代码在固定提交中：

- [vllm_fl/nvidia_platform.py](https://github.com/flagos-ai/vllm-plugin-FL/blob/4b3a76a19de52af9981237e860c2177e73b9ee68/vllm_fl/nvidia_platform.py#L10)：直接继承上游 CudaPlatform。
- [vllm_fl/__init__.py](https://github.com/flagos-ai/vllm-plugin-FL/blob/4b3a76a19de52af9981237e860c2177e73b9ee68/vllm_fl/__init__.py#L215)：NVIDIA 选择 NvidiaPlatformFL，不走其他后端的 empty-wheel fallback。

## 8. 实际尝试的模型推理

H20 机器已有 `/data/models/Qwen/Qwen3-0.6B`。在相同容器、相同环境变量下尝试 TP1 eager 推理：

```bash
docker exec -i \
  -e VLLM_PLUGINS=fl \
  -e USE_FLAGGEMS=1 \
  -e GEMS_VENDOR=nvidia \
  -e USE_FLAGTUNE=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  "$REPRO_CONTAINER" bash -lc 'python -' \
  > "$REPRO_DIR/modified-wheel/inference-attempt.log" 2>&1 <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(
    model='/data/models/Qwen/Qwen3-0.6B',
    tensor_parallel_size=1,
    max_model_len=1024,
    enforce_eager=True,
    gpu_memory_utilization=0.1,
)
print(llm.generate(
    ['The capital of France is'],
    SamplingParams(temperature=0, max_tokens=8),
))
PY
printf 'inference_exit=%s\n' "$?"
cat "$REPRO_DIR/modified-wheel/inference-attempt.log"
```

实际在第一行导入 LLM 时就报第 7 节的缺模块错误，未进入模型初始化、权重加载或 token 生成。因此没有把此实验记为推理通过，也未继续声称 graph / Serving / E2E 通过。

## 9. 如何确认 NumPy 问题已被定制依赖避开

```bash
docker exec -i "$REPRO_CONTAINER" bash -lc 'python -' <<'PY'
import cv2
import numpy as np
from importlib.metadata import requires

x = np.arange(12, dtype=np.uint8).reshape(3, 4)
y = cv2.resize(x, (8, 6), interpolation=cv2.INTER_NEAREST)
np.testing.assert_array_equal(y[::2, ::2], x)
print('CV_NUMPY_TEST_PASS', cv2.__version__, np.__version__, y.shape)
print('OpenCV Requires-Dist:', requires('opencv-python-headless'))
PY
```

实际结果：

```text
CV_NUMPY_TEST_PASS 5.0.0 1.26.4 (6, 8)
OpenCV Requires-Dist: None
```

这证明本次定制 OpenCV + NumPy 1.26.4 的安装、导入和该 resize 用例可用，不表示所有多模态功能或算子均已验证。

## 10. CI 入口和依赖完整性检查

```bash
docker exec "$REPRO_CONTAINER" bash -lc \
  'python tests/run.py --platform nvidia --device h20 --scope unit' \
  > "$REPRO_DIR/modified-wheel/unit.log" 2>&1
printf 'unit_exit=%s\n' "$?"
cat "$REPRO_DIR/modified-wheel/unit.log"

docker exec "$REPRO_CONTAINER" bash -lc 'python -m pip check'
```

实际 UT 入口错误：

```text
  File "/audit/repo/tests/utils/device_utils.py", line 16, in <module>
    import pytest
ModuleNotFoundError: No module named 'pytest'
```

安装插件后，实际 pip check 输出：

```text
torch 2.13.0+cu130 requires triton, which is not installed.
vllm-plugin-fl 0.0.0+g4b3a76a19 requires modelscope, which is not installed.
```

第一条也存在于未安装 vLLM 的原始 Runtime：默认从 `/opt/flagtree` 导入 Triton，但当前环境没有独立 Triton distribution 元数据。本文没有把它等同于已复现的 GPU 编译失败；此前基础 Runtime 中 Torch GPU 运算和 FlagGems GPU 加法均已通过。第二条是插件 `--no-deps` 安装后未补齐 modelscope 的结果，不是修改版 wheel 声称安装 modelscope 却安装失败。

包清单核对还确认 pytest-xdist、pytest-cov、pytest-json-report、pytest-timeout、FlagGems-vllm 等没有随 wheel 默认依赖安装。它们需要按插件及 CI 的声明另行协调，并非单凭这个 wheel 就会具备全套 CI 环境。

## 11. 最初官方 wheel 路径的对照实验

**本节是较早的独立对照，不能与第 6 节的甲方 wheel 安装混为一谈。** 当时在原始 Runtime 的另一个独立容器中，尝试保留相同 protected.txt，解析官方 `vllm[audio]==0.32.0`：

```bash
# 仅作对照；应在另一个全新 Runtime 容器中运行，不要覆盖前述实验容器。
uv pip install --dry-run --python /flagos/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  --constraint /audit/protected.txt 'vllm[audio]==0.32.0'
```

实际解析失败，因果链是：

```text
Runtime 保护约束：numpy==1.26.4
官方 vLLM 0.32.0：opencv-python-headless>=4.13.0
官方 OpenCV 在 Python 3.12：numpy>=2
=> 无法同时满足
```

同时发现 Runtime setuptools 84.2.1 超出官方和修改版 vLLM 声明的 <81 上限。随后按照甲方补充的修改版 wheel 重新实验，NumPy 问题已按第 9 节避开，setuptools 则实际降为 80.10.2。**当前提交给维护方的主要运行阻塞是缺少原生扩展，并非仍把官方 OpenCV 的冲突算到定制 OpenCV 上。**

## 12. 希望维护方确认的接口契约

1. 指定的 `0.32.0+flagos` wheel 不含任何原生 `.so`，这是否是预期的 empty 构建？
2. 当前 `0.5.0-dev` 的 NVIDIA 平台需要 `vllm._C_stable_libtorch` 等原生扩展，配套扩展应由哪个包提供，或应改用哪个插件适配提交？
3. setuptools 统一使用 80.10.2 是否符合预期？是否由 Runtime 层或应用镜像层调整？
4. FlagGems-vllm、modelscope 和 pytest 系列依赖是否由应用/CI 镜像补齐？

当前没有绕过缺扩展错误、注入占位模块或修改测试结果；完整 UT / FT / E2E / benchmark 尚未通过。

## 13. 原始证据

原始机器上证据目录：`/data/codex-cuda-runtime-audit-20261106/modified-wheel/`。

随本地报告保留的 `modified-wheel-evidence/` 包含：

- `METADATA.txt`、`WHEEL.txt`、`file-list.json`、`wheel-summary.json`：包内容、依赖与完整性记录。
- `dry-run.log`、`install.log`、`plugin-install.log`：实际安装与退出结果。
- `checks.json`、`plugin-import.log`：导入检查及原生扩展缺失错误。
- `inference-attempt.log`：0.6B 模型推理启动尝试。
- `unit-after-wheel.log`：UT 入口及安装插件后的 pip check 输出。
- `final-state.json`、`final-state.log`：安装后包版本、原始包变更及 OpenCV 数值检查。

本文已嵌入关键命令、版本和错误，单独转发本 Markdown 也可阅读；原始证据用于进一步定位和核对。测试容器保留用于复测。

## 14. 实验记录补充：问题上报与后续跟踪（2026-11-09）

### 可粘贴到 Runtime 状态表的记录

2026-11-06 在 H20 上，基于 `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3:2.3.0` 安装甲方指定的 `vllm==0.32.0+flagos` wheel，配合 `vllm-plugin-FL 0.5.0-dev`（提交 `4b3a76a`）验证。wheel 和插件安装成功；Torch GPU 运算、FlagGems GPU 加法通过；定制 OpenCV 与 NumPy 1.26.4 的 resize 检查通过，setuptools 安装时由 84.2.1 调整为 80.10.2。但插件 NVIDIA 平台初始化失败：`ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'`，尚未进入模型加载。wheel 内无原生 `.so`，当前插件 NVIDIA 路径依赖原生 CUDA vLLM，二者接口约定不一致。该问题已于 2026-11-09 由 AlexMa616 上报 [build-infra Issue #1692](https://github.com/flagos-ai/build-infra/issues/1692)。截至本次核查 Issue 仍为 Open，未确认对应修复 PR；等待明确 native / empty 构建方案、提供匹配产物或适配代码后复测。完整 UT / FT / E2E / benchmark 尚未通过，建议该项标记为“适配阻塞，待维护方确认”，不标记“整套 CI 验证完成”。

### Issue 与相关 PR 的核查说明

- [Issue #1692](https://github.com/flagos-ai/build-infra/issues/1692) 是缺陷报告，不是 PR。作者为 AlexMa616，创建时间为 2026-11-09 11:23（北京时间）；本次读取时状态为 Open、无评论。
- Issue 的判断与本次实测一致：问题在于 empty wheel 与 native NVIDIA 平台的接口约定不匹配，不应仅凭缺失扩展就认定为一次编译异常。Issue 进一步报告旧版 `+flagos` wheel 同样采用 empty 构建；这部分是 Issue 作者的补充核查，不是本记录新增的本地实测。
- 已查看 Issue 引用的 [vllm-plugin-FL PR #615](https://github.com/flagos-ai/vllm-plugin-FL/pull/615)，标题为 `feat(nvidia): adapt native CUDA vLLM 0.32 runtime and CI`，状态为已合并。其变更保留上游原生 CUDA wheel 与 V1/V2 runner，移除 NVIDIA 的 EMPTY 适配路径，解释了当前插件为什么要求原生扩展。**它是此前的接口变更，不是针对 Issue #1692 新提交的修复 PR。**
- PR #615 的 CI 环境使用原生 `vllm==0.32.0+cu129`、`torch==2.13.0+cu129`，与本次 Runtime 的 `torch==2.13.0+cu130` 加 empty `vllm==0.32.0+flagos` 不同。该 PR 所记录的测试结果不能替代指定 Runtime + 指定 wheel 的验收。
- 本次检索尚未确认与 Issue #1692 对应的修复 PR，不能把“已提交 Issue”或“历史 PR 已合并”记为“本问题已修复”。
- Issue 中另行复现的容器限制为 1 张 H20；本记录 2026-11-06 的容器可见 8 张 H20。两者均在平台导入阶段失败，此处分别保留各自环境描述。

### 待确认与复测条件

1. 维护方明确 NVIDIA 2.3.0 发布栈采用 native 还是 empty vLLM：若采用 native，需要提供与 Runtime Torch/CUDA 匹配的原生扩展产物；若继续采用 empty，需要提供相应插件适配提交。
2. 获得新产物或适配代码后，记录 wheel 文件版本/校验、镜像拉取时间和插件提交，先复测 `VLLM_PLUGINS=fl` 下的 `from vllm import LLM` 及 NVIDIA 平台初始化。
3. 平台启动通过后，再按确认的包清单补齐 pytest 系列、modelscope、FlagGems-vllm 等，执行 UT、FT、H20 E2E eager/graph、Serving 和 benchmark，依据实际结果更新状态。

本次仅补充 2026-11-09 的 Issue/PR 跟踪记录，未宣称新版本已发布或已完成修复验证；未改动插件代码与 Dockerfile。
