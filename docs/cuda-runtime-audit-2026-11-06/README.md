> 最新进展：用户已授权补依赖、调整 CI 并构建镜像。请参阅 [2026-11-09/10 CI 镜像构建与实测记录](ci-image-2026-11-09.md)。下文保留最初 Runtime 核对的历史结果。

# CUDA 13.3 Runtime 与 vLLM 0.32.0 依赖核对

**后续更新（2026-11-06）：甲方修改版 wheel 已安装实测。** [修改版 wheel 专项报告](modified-wheel.md) 记录了新结论：配套定制 OpenCV 可以保留 NumPy 1.26.4，下面“NumPy 必须升级”的结论仅适用于官方 wheel 依赖路径，不再适用于这个定制组合。setuptools 仍需降至兼容范围，本次独立容器实装为 80.10.2。修改版 wheel 不含原生 CUDA 扩展，当前 NVIDIA 插件启动报缺少 `vllm._C_stable_libtorch`；完整 CI 仍未通过。以下内容保留为首次官方依赖路径的核对记录。

核对日期：2026-11-06（北京时间）。状态：基础 GPU 验证通过；依赖存在确定冲突，尚未进入完整 UT / FT / E2E / benchmark 验证。

## 结论

`harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3:2.3.0` 的升级有效：Torch 2.13、TorchVision、TorchAudio、FlagCX 能正常导入，Torch 能识别 8 张 H20 并执行 GPU 张量运算，FlagGems 默认编译器路径也通过了一个 GPU 加法编译与正确性验证。

这只能说明底层环境可用，不能说明已满足 vllm-plugin-FL 的完整依赖与 CI。本次未向 Runtime 安装任何包；核对结束前再次比较包清单，确认未改变。没有修改 Dockerfile 或 CI 配置。

当前有三个必须先协调的事项：

1. **NumPy 必须协调升级，或由甲方提供兼容构建。** Runtime 固定 `numpy==1.26.4`，但 vLLM 0.32.0 要求 `opencv-python-headless>=4.13.0`，该包在 Python 3.12 下要求 `numpy>=2`。保留 NumPy 1.26.4 的 dry-run 已明确解析失败。仅放开 NumPy 后，解析器选择了 `numpy==2.3.5`；这只是可解析候选，尚未安装或验证。
2. **setuptools 版本必须降低到双方兼容区间。** Runtime 是 `84.2.1`，vLLM 0.32.0 要求 `>=78.1.1,<81`；插件 build-system 要求 `>=77.0.3,<81`。候选 `80.10.2` 同时满足这些范围，亦为仓库现有 CUDA CI 工具清单版本。
3. **CI 环境检查要支持新的 Runtime 来源。** 本分支 `.github/scripts/nvidia/stack.py` 目前绑定旧 cu129 镜像、特定 FlagTree/FlagGems 源码构建及来源记录，不能原样用于此 cu130 Runtime。不能通过切回 cu129、更换甲方 Runtime 组件或跳过检查来声称新镜像验证通过，应另行明确新环境的检查契约。

## 核对对象和方法

- 本地工作目录：`/Users/m/BrianPei/vllm-plugin-FL`。
- 开发分支：`codex/cuda-runtime-2.3.0-0.5.0-dev`。
- 插件代码基线：`4b3a76a19de52af9981237e860c2177e73b9ee68`，与本次获取的组内、官方 `0.5.0-dev` 一致。
- 远端使用独立目录 `/data/codex-cuda-runtime-audit-20261106` 及独立测试容器，未更改其他任务容器。
- 从 Harbor 按 `:2.3.0` 拉取，没有在拟议镜像引用中使用 digest。
- 使用目标 `v0.32.0` 的 common/cuda 文件及 PyPI vLLM 0.32.0 元数据核对；另外比较用户指定的 `main` 文件。
- 数量按 Linux x86_64、Python 3.12 环境标记过滤，不包括被其他平台条件排除的要求；requirements 中 extras 的传递依赖另由解析器处理。

### 来源

- [Runtime 官方说明](https://flagos-ai.github.io/release-info/runtime/nvidia-cuda13.3/)
- [vLLM v0.32.0 common.txt](https://github.com/vllm-project/vllm/blob/v0.32.0/requirements/common.txt)
- [vLLM v0.32.0 cuda.txt](https://github.com/vllm-project/vllm/blob/v0.32.0/requirements/cuda.txt)
- [用户指定的 main common.txt](https://github.com/vllm-project/vllm/blob/main/requirements/common.txt)
- [用户指定的 main cuda.txt](https://github.com/vllm-project/vllm/blob/main/requirements/cuda.txt)
- [vLLM 0.32.0 发布元数据](https://pypi.org/pypi/vllm/0.32.0/json)
- [OpenCV 4.13.0.92 发布元数据](https://pypi.org/pypi/opencv-python-headless/4.13.0.92/json)

`main` 会继续变化；本次 `main/cuda.txt` 已要求 Torch 2.14.0、TorchVision 0.29.0，与目标 0.32.0 不一致。因此不要直接把 main 依赖安装到 0.32.0 环境。完整逐项比较见 `dependency-comparison.csv`。

## Runtime 实测包与行为

| 项目 | 实测 | 验证范围 |
|---|---|---|
| Python | 3.12.13，`/flagos/bin/python` | 解释器可用 |
| Torch | 2.13.0+cu130，`torch.version.cuda=13.0` | 导入、8 GPU 可见、GPU 张量运算通过 |
| TorchVision | 0.28.0+cu130 | 导入通过 |
| TorchAudio | 2.11.0+cu130 | 导入通过；未测完整音频模型 |
| FlagGems | 5.4.0 | 导入、默认 FlagTree 路径 GPU add 数值对齐通过 |
| FlagCX | 0.14.1+cuda13.3 | 导入通过，未做多卡通信验证 |
| FlagTree | 0.7.0 | 默认启用，活动 Triton 从 `/opt/flagtree/triton` 加载，版本 3.6.0 |
| 备用 Triton | 3.7.1，位于 `/opt/triton` | 已确认文件和 compiler 清单；未切换执行完整测试 |
| NumPy | 1.26.4 | 已安装；与 OpenCV 传递要求冲突 |
| setuptools | 84.2.1 | 已安装；不满足 vLLM 与插件声明的上限 |
| PyYAML / SQLAlchemy / pybind11 | 6.0.1 / 2.0.57 / 3.0.3 | 已安装，不应重复当作缺包 |
| git / gcc / g++ / cmake / ninja / uv | 工具已存在 | 未编译插件 CUDA 扩展；uv 是独立可执行文件，无 Python distribution 元数据 |
| nvcc | 未安装，`nvcc --version` 退出 127 | CUDA 本地扩展/JIT 是否必须补装需后续实测 |
| FFmpeg / 系统 libavcodec / libsndfile | 本次路径检查未发现 | 媒体依赖候选；尚不能宣称所有推理都需要 |

Runtime 的 `PYTHONPATH=/opt/flagtree` 保留了 FlagTree 的默认优先级。文档中的 Triton 3.7.1 指备用路径；它不等于默认活动编译器版本。不要直接安装独立 Triton 覆盖 FlagTree 后宣称沿用原环境。

`pip check` 报 `torch 2.13.0+cu130 requires triton, which is not installed`：当前解释器看不到独立 Triton 的 distribution，但能通过 FlagTree 导入 Triton。需要区分包元数据依赖与编译器切换设计；这不等于已证明 GPU 编译失败。

宿主驱动实际为 580.95.05，Runtime 文档列出的宿主驱动是 610.43.02。本机基础 GPU 与 FlagGems 测试已通过，但这些测试不能替代 CUDA 13.3 全栈兼容性和长时间多卡验证。

## 官方 vLLM 依赖对比

| 文件 | 满足 | 缺少 | 直接版本冲突 |
|---|---:|---:|---:|
| v0.32.0 common.txt | 6 | 51 | 1：setuptools |
| v0.32.0 cuda.txt（不重复 common） | 3 | 14 | 0 |
| main common.txt | 6 | 54 | 1：setuptools |
| main cuda.txt（不重复 common） | 1 | 14 | 2：Torch / TorchVision |

v0.32.0 的 CUDA 直接依赖里，Torch、TorchVision、TorchAudio 可保留。`==2.13.0` 这样的要求接受 `2.13.0+cu130` 本地版本标识，不构成版本冲突。

Runtime 未提供 vLLM 自身。缺少的 65 项官方直接要求见 CSV，覆盖 transformers/tokenizers、服务 API、结构化输出、FlashInfer、TorchCodec、CUDA 算子等；这些是正常的应用层依赖，不能仅凭 Runtime 未包含就认为基准镜像有缺陷。

`vllm[audio]==0.32.0` 的官方 audio extras 另外包含 av、scipy、soundfile、soxr 及 mistral_common 的音频 extras。它们不全出现在 common/cuda 两个文件里，但属于 vLLM 官方可选依赖；插件的 test extras 和 NVIDIA setup 明确使用它们。

### 解析结果（未执行安装）

命令模板：

```bash
uv pip install --dry-run --python /flagos/bin/python \
  --constraint /audit/protected.txt 'vllm[audio]==0.32.0'
```

完整命令中的镜像站、受保护约束和输出见 evidence。保护 Torch / TorchVision / TorchAudio / FlagGems / FlagTree / FlagCX / NumPy 时，因 NumPy 与 OpenCV 冲突解析失败。

仅从约束中移除 NumPy 后，解析器成功选择 202 个依赖图节点，计划安装 169 个包并替换 2 个已安装包：NumPy 1.26.4 → 2.3.5、setuptools 84.2.1 → 80.10.2。计划还会安装独立 Triton 3.7.1 以满足 Torch 元数据；实际活动编译器仍需按 Runtime 的切换规则验证。这不是完整 lockfile，不代表下载、二进制加载、推理或 CI 已通过。

Runtime 自带 `/etc/flagos/flaggems-constraints.txt` 也固定 NumPy 1.26.4。若甲方同意升级，约束文件及后续启动时 FlagGems 安装流程必须同时协调，避免后续操作再次回退 NumPy。

## 超出 common/cuda 的插件与 CI 必要依赖

下列版本来自当前插件代码/工具清单，不是本次实际安装结果，也不是甲方已批准的版本。

| 包或组件 | Runtime 状态 | 版本依据与用途 | 确认事项 |
|---|---|---|---|
| vllm-plugin-fl | 未安装 | 本开发分支 checkout；CI 以 `--no-deps -e .` 安装 | 应用本身 |
| FlagGems-vllm | 未安装 | 共享 installer 要求 v0.2.0、commit `b5c2c2249319277ccda4d41aa0f65164bd9b2a19`；其包元数据仍为 0.1.0，不能只按版本号判定 | 不在 vLLM common/cuda 中，但插件算子后端直接导入 |
| modelscope | 未安装 | 插件正式依赖固定 1.35.3；旧 CUDA CI 工具清单却为 1.37.1 | 两处声明应统一；建议先按正式依赖 1.35.3，待甲方确认 |
| pytest | 未安装 | 当前 CI 工具清单 9.1.1 | UT / FT / E2E 测试框架 |
| pytest-xdist | 未安装 | NVIDIA setup 要求 >=3.0.0，runner 使用 `-n` | 正常 CI 并行参数需要 |
| pytest-cov | 未安装 | 当前 CI 工具清单 7.1.0，runner 使用 `--cov` | 覆盖率采集 |
| pytest-json-report | 未安装 | 当前 CI 工具清单 1.5.0，runner 使用 `--json-report` | JSON 报告 |
| pytest-timeout | 未安装 | test extras；当前工具清单 2.4.0 | 仓库测试超时控制 |
| decorator | 未安装 | 插件 test extras 无固定版本 | 开发/测试依赖 |
| wheel | 未安装 | 插件 build-system 要求；当前工具清单 0.48.0 | 构建工具，版本可协商 |
| scikit-build-core | 未安装 | 插件 build-system 固定 0.11，即 0.11.0 | 构建环境声明 |
| xgboost | 未安装 | 当前 NVIDIA setup 无条件创建 XGBRanker；工具清单 3.3.0 | 当前 setup 的需要，不等于全部推理都必需 |
| scikit-learn | 未安装 | 当前 CI 工具清单 1.9.1；供 XGBRanker 的 sklearn 接口使用 | 与 xgboost 一并确认，可先评估 setup 是否应按功能按需检查 |
| uv | 独立工具已存在 | setup 会强制安装 Python 包 uv==0.13.4 | 这是工具版本要求，不应误报“机器没有 uv” |

PyYAML、packaging、SQLAlchemy、pybind11 已存在，FlagGems-vllm installer 的直接辅助依赖无需重复补装。setuptools-scm、ninja、cmake 也已存在；是否升级到旧 CI 工具清单的精确版本不是当前已证实的运行阻塞。

`extra-dependencies.csv` 逐项记录了 Runtime 状态、仓库版本依据和适用范围。FFmpeg、CUDA 编译器/开发头文件先作为待验证系统依赖，不把它们与已确认缺失的测试包混为一谈。

## CI 与实测边界

| 验证项 | 结果 |
|---|---|
| 拉取指定 Runtime 标签 | 成功 |
| 关键包导入 | 通过，详见 facts.json |
| Torch GPU 运算 | 通过 |
| FlagGems GPU add 编译及数值对齐 | 通过，详见 flaggems-kernel.log |
| 保留所有 Runtime 核心版本的 vLLM 依赖解析 | 失败：NumPy / OpenCV 确定冲突 |
| 放开 NumPy 的 vLLM 依赖解析 | 成功；未安装 |
| 原始 Runtime 的 UT 入口 | 失败：ModuleNotFoundError: pytest |
| UT 全量 | 未运行到测试收集 |
| FT 全量 | 未运行，受依赖与 CI 环境检查阻塞 |
| H20 E2E 推理 / Serving | 未运行，受相同阻塞 |
| benchmark 冒烟 | 未运行，受相同阻塞 |
| 完整 vLLM 上游测试套件 | 未执行；本任务 CI 范围按 vllm-plugin-FL 的 NVIDIA 配置，不能与整个 vLLM 上游测试套件混称 |

H20 上已发现 Qwen3.6-27B、Qwen3.6-35B-A3B、Qwen3-0.6B 等模型目录，后续仍应检查模型完整性。本分支 H20 E2E PR 配置覆盖 27B TP2 eager / graph 及 35B-A3B TP2 eager / graph 的推理或服务任务，不能用单个加法或 0.6B 冒烟替代。

### 旧 CI 检查与新 Runtime 的差异

`.github/scripts/nvidia/stack.py` 要求：

- vLLM distribution 为 `0.32.0+cu129`，Torch 为 `2.13.0+cu129`，TorchCodec 为 `0.16.0+cu129`，`torch.version.cuda` 为 12.9。
- FlagTree / FlagGems 为旧 CI 的特定 Git 派生版本，并在 `/opt/fl-ci/provenance.json` 中记录来源。
- 不允许独立 Triton distribution，与 Runtime 的双编译器切换机制需重新协调。

这些是来源和环境契约差异，不是补 pytest 就能解决的缺包问题。尚未修改此检查或绕过它执行“伪全绿”。

## 发给甲方的建议确认文本

> 已在 H20 实测新版 `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3:2.3.0`：Torch 2.13、TorchVision、TorchAudio、FlagCX 导入正常，Torch GPU 运算和 FlagGems 默认编译器 GPU 加法通过。当前尚不能确认整个插件 CI 通过。对齐 vLLM 0.32.0 时，Runtime 的 NumPy 1.26.4 与官方 OpenCV>=4.13 的 NumPy>=2 要求冲突；setuptools 84.2.1 也超出官方要求的 <81 上限。请确认允许应用层调整为 NumPy 2.3.5、setuptools 80.10.2，或由 Runtime 层统一修正，并同步 Runtime 内 NumPy 约束。另请确认插件所需 FlagGems-vllm v0.2.0、modelscope 1.35.3 及 pytest 系列/构建工具可作为应用及 CI 层依赖安装。现有 CI 的 cu129 和旧源码来源检查需随新 cu130 Runtime 更新。获得版本确认后再安装并执行 UT、FT、H20 E2E eager/graph、Serving 和 benchmark，系统编译/媒体缺包若在实测中出现，会另附错误证据报备。

后续 Dockerfile 按用户要求命名 `docker/cuda/Dockerfile.v0.28.0`，但内容实际面向 **vLLM 0.32.0**，需要在文件注释与构建说明中写清，避免文件名误导。本次未创建该 Dockerfile，等待上述依赖确认。

## 附件

- `installed-packages.csv`：默认活动解释器可见的 43 个 distribution；备用 Triton 另见说明。
- `dependency-comparison.csv`：v0.32.0 与 main 的逐项 direct requirement 比较。
- `extra-dependencies.csv`：插件/CI 额外包及来源范围。
- `evidence/packages.json`：完整版本、Requires-Dist、安装位置。
- `evidence/facts.json`：基础导入、GPU、工具、pip check 原始结果。
- `evidence/uv-dry-run.log`：保留 NumPy 1.26.4 的失败解析。
- `evidence/uv-dry-run-allow-numpy.log`：仅放开 NumPy 的候选解析计划。
- `evidence/baseline-unit.log`、`evidence/flaggems-kernel.log`：UT 入口失败与算子验证输出。
- `evidence/commands.md`：复核命令及清理状态。trial 容器的 stop 操作被堡垒机风险确认拦住，目前保留运行，未绕过该限制。

本地工具操作附记：一次未指定 `--no-project` 的 uv 调用自动重建了仓库内 Git 忽略的 `.venv`，调用随后已终止。后续本地核对在 `/tmp` 独立环境执行；没有尝试向该本地虚拟环境安装项目依赖。若 PyCharm 使用此 `.venv`，需重建开发依赖。
