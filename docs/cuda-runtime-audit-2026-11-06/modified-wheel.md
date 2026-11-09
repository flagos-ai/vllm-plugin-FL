# 甲方修改版 vLLM 0.32.0 wheel 实测

日期：2026-11-06（北京时间）。插件基线：`4b3a76a19de52af9981237e860c2177e73b9ee68`；本次再次查询官方 `0.5.0-dev`，仍为该提交。所有实验使用 H20 上独立容器 `codex-cuda-flagos-wheel-20261106`，未修改本地插件代码、Dockerfile 或 CI 检查。

## 结论与对上一份报告的修正

**这个 wheel 能安装，也解决了先前官方 OpenCV 与 NumPy 1.26.4 的安装冲突；但不能直接让当前分支的 NVIDIA 推理或完整 CI 跑通。**

| 先前问题 | 修改版 wheel 的实测结果 |
|---|---|
| NumPy 1.26.4 与官方 OpenCV>=4.13 的约束冲突 | 已避开：改用定制 `opencv-python-headless==5.0.0.93+flagos`。保留 NumPy 1.26.4，OpenCV 导入及 resize 数值检查通过。此前“必须升级 NumPy”的结论只适用于官方依赖路径。 |
| setuptools 84.2.1 超出 <81 上限 | 未改变该要求；安装器仍降级为 80.10.2。wheel 由 setuptools 84.2.1 构建，不表示它的运行依赖允许 84.2.1。 |
| 原始 Runtime 缺少 vLLM | 已安装 `vllm==0.32.0+flagos`，基础 `import vllm` 通过。 |
| 插件与 CI 的额外依赖缺失 | 未解决：pytest 系列、FlagGems-vllm、modelscope 等不属于该 wheel 的默认依赖。UT 入口仍报缺少 pytest，安装插件后 pip check 还报缺少 modelscope。 |
| 旧 CI 绑定官方 cu129 和原生扩展 | 未解决；该修改版还缺少当前 NVIDIA 平台直接导入的原生模块，已复现启动失败。 |

## 下载来源与内容

- [用户提供的 Nexus 浏览页](https://resource.flagos.net/#browse/browse:flagos-pypi-nvidia:vllm%2F0.32.0%2Bflagos%2Fvllm-0.32.0%2Bflagos-cp312-cp312-linux_x86_64.whl)
- [实际 wheel 下载地址](https://resource.flagos.net/repository/flagos-pypi-nvidia/packages/vllm/0.32.0%2Bflagos/vllm-0.32.0%2Bflagos-cp312-cp312-linux_x86_64.whl)

通过 Nexus 资产接口找到实际 `/packages/` 路径。下载大小为 11,446,273 字节，下载校验值与资产接口一致，ZIP CRC 检查通过。校验值记录在 `modified-wheel-evidence/wheel-summary.json`，只是 wheel 文件完整性记录；镜像仍按用户要求使用 `:2.3.0` 标签，不引用镜像 digest。

包内容实测：

- distribution 版本：`0.32.0+flagos`；模块的 `vllm.__version__` 为 `0.32.0`。
- wheel 标签：`cp312-cp312-linux_x86_64`，对应 CPython 3.12、Linux x86_64。
- 共 3301 个 ZIP 条目，其中 2642 个 `.py` 文件，包含 vLLM 的 Python 代码及 distribution 元数据。
- 没有 `.so`、`.pyd` 或 `.dll` 原生扩展，没有 `.pth` 启动注入文件。
- 没有 `vllm._C_stable_libtorch` 的 Python 替代模块。
- 文件清单见 `modified-wheel-evidence/file-list.json`，依赖声明见 `METADATA.txt`，wheel 标签见 `WHEEL.txt`。

这种 payload 表现与不携带原生 CUDA 扩展的构建一致，但仅凭文件不能断言甲方的构建命令或设计意图；需甲方确认它是否就是 intended empty 构建。

## 与官方 0.32.0 依赖声明的差别

对比原始 wheel METADATA 与官方发布的 vLLM 0.32.0 Requires-Dist，改变了三项 common 依赖：

| 依赖 | 官方声明 | 甲方声明 |
|---|---|---|
| xgrammar | 0.2.8 | 0.2.8+flagos |
| opencv-python-headless | >=4.13.0 | 5.0.0.93+flagos |
| compressed-tensors | 0.17.0 | 0.17.0+flagos |

另外，移除了 CUDA 依赖声明：numba、torch、torchaudio、torchvision、torchcodec、PyNvVideoCodec、flashinfer-python、apache-tvm-ffi、tilelang、nvidia-cudnn-frontend、nvtx、fastsafetensors、instanttensor、nvidia-cutlass-dsl、quack-kernels、tokenspeed-mla、humming-kernels。详见 wheel-summary.json 中的逐条差异。

**移除 Requires-Dist 不等于已经把这些包的实现打入 wheel，也不等于每条执行路径都不需要它们。** 例如当前包没有 native CUDA 扩展，而它的 `vllm/platforms/cuda.py` 第 23 行仍直接导入 `vllm._C_stable_libtorch`。

定制 OpenCV 的安装元数据没有 Requires-Dist；这使解析器不会因其 NumPy>=2 声明升级 NumPy。实际检查了 NumPy 1.26.4 下的导入及最近邻 resize，并与 NumPy 数组结果对齐，均通过。但没有据此推断所有 OpenCV 或多模态功能都已验证。

## 安装结果

独立容器基于：

```text
harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3:2.3.0
```

使用上一轮的 `/audit/protected.txt`，保护 Torch、TorchVision、TorchAudio、NumPy、FlagGems、FlagTree、FlagCX 的原始版本；同时配置普通镜像站和甲方 NVIDIA 包索引。

```bash
uv pip install --python /flagos/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  --extra-index-url https://resource.flagos.net/repository/flagos-pypi-nvidia/simple \
  --index-strategy unsafe-best-match \
  --constraint /audit/protected.txt \
  /audit/modified-wheel/vllm-0.32.0+flagos-cp312-cp312-linux_x86_64.whl
```

该命令既做了 dry-run，也实际执行了安装；安装退出码为 0。新增的应用依赖来自 wheel 声明的依赖及其传递依赖，没有执行全套旧 CI requirements 安装。

| 项目 | 安装后版本 | 相对 Runtime |
|---|---|---|
| vllm | 0.32.0+flagos | 新安装 |
| opencv-python-headless | 5.0.0.93+flagos | 新安装 |
| xgrammar | 0.2.8+flagos | 新安装 |
| compressed-tensors | 0.17.0+flagos | 新安装 |
| numpy | 1.26.4 | 保留 |
| torch | 2.13.0+cu130 | 保留 |
| torchvision / torchaudio | 0.28.0+cu130 / 2.11.0+cu130 | 保留 |
| flag-gems / flagtree / flagcx | 5.4.0 / 0.7.0 / 0.14.1+cuda13.3 | 保留 |
| setuptools | 80.10.2 | 由 84.2.1 降级 |

安装后对初始 Runtime 的全部 distribution 做了比较：原有包里只有 setuptools 版本改变。完整安装状态及缺包检查见 `final-state.json`。

随后按当前分支 NVIDIA CI 的安装方式，额外以 `--no-deps --no-build-isolation -e /audit/repo` 安装插件 checkout，构建和安装成功。此步骤不会自动安装 modelscope 或 pytest，也未安装 FlagGems-vllm。

## 运行验证与根因

| 验证 | 结果 |
|---|---|
| import vllm 与版本/路径核对 | 通过 |
| 定制 OpenCV 导入与 NumPy 1.26.4 resize 校验 | 通过 |
| 定制 xgrammar、compressed-tensors 导入 | 通过；未做所有内部功能测试 |
| from vllm import LLM（安装插件前） | 失败：缺少 vllm._C_stable_libtorch |
| import vllm_fl（安装插件后） | 通过 |
| 启用 FL 后解析 NVIDIA current_platform | 失败：相同 native 扩展缺失 |
| Qwen3-0.6B TP1 eager 推理启动尝试 | 失败于同样的平台导入，尚未加载模型或生成 token |
| UT 入口 | 失败：No module named pytest |
| 完整 FT / E2E eager/graph / Serving / benchmark | 未执行通过，不能报告全绿 |

推理尝试显式设置了：

```text
VLLM_PLUGINS=fl
USE_FLAGGEMS=1
GEMS_VENDOR=nvidia
USE_FLAGTUNE=0
VLLM_USE_V2_MODEL_RUNNER=1
```

所以截图提到的 V2 环境变量已在实验中设置，不能解决当前这个缺模块问题。

真实异常链为：

```text
vllm.config.compilation
  -> vllm.platforms.current_platform
  -> /audit/repo/vllm_fl/nvidia_platform.py:10
       from vllm.platforms.cuda import CudaPlatform
  -> site-packages/vllm/platforms/cuda.py:23
       import vllm._C_stable_libtorch
ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'
```

当前分支 `vllm_fl/__init__.py` 对 NVIDIA 明确选择 `vllm_fl.nvidia_platform.NvidiaPlatformFL`，该类继承上游原生 CudaPlatform；给其他后端的 empty-wheel fallback 不会用于 NVIDIA。旧 CI stack.py 还要求原生 `.so`、`0.32.0+cu129` distribution、指定源码来源。

因此这里既有真实运行时扩展缺失，又有旧 CI 来源检查不匹配，不能只放宽检查或把导入错误替换成占位模块就认定适配完成。

## 尚未解决的额外依赖

安装该 wheel 后仍缺少 pytest、pytest-xdist、pytest-cov、pytest-json-report、pytest-timeout、FlagGems-vllm、modelscope、decorator、wheel、scikit-build-core。它们分别属于测试框架、插件运行依赖或声明的构建工具；并非全部都是每次推理的运行时硬依赖。

pip check 在安装插件后报告缺少 modelscope；同时仍保留原 Runtime 的 Torch→独立 Triton metadata 提示。默认 FlagTree 编译器路径未切换，未为消除提示而安装/覆盖独立 Triton。

没有为了推进测试而擅自安装这些额外包，也没有改动 NVIDIA 平台实现。完整 CI 尚需上述依赖协调以及 native/empty 构建方案确认。

## 可转发甲方的说明

> 已在新版 CUDA 13.3 Runtime + H20 上实际安装指定的 vllm-0.32.0+flagos wheel。其配套 opencv-python-headless 5.0.0.93+flagos 能与 NumPy 1.26.4 导入并完成 resize 检查，先前官方 OpenCV 的 NumPy>=2 冲突可以避开；setuptools 仍按 wheel 声明需要降至 <81，实装为 80.10.2。当前 wheel 没有任何原生 .so，而官方/组内 0.5.0-dev 当前提交 4b3a76a 的 NVIDIA 平台继承上游 CudaPlatform，启动即报缺少 vllm._C_stable_libtorch，设置 VLLM_USE_V2_MODEL_RUNNER=1 仍然失败。请确认此 wheel 是否 intended empty 构建、NVIDIA 所需原生 CUDA 扩展应由哪个包提供，以及该 wheel 对应哪个插件适配提交。另需确认 FlagGems-vllm、modelscope 及 pytest 系列测试依赖；该 wheel 默认依赖没有包含它们。当前尚不能报告整套 CI 通过。

后续应在“甲方提供适用于当前 NVIDIA 原生执行路径的完整 wheel/扩展”和“插件调整为甲方指定的无原生扩展适配路径”之间明确接口契约，再继续实现；本次只检查和复现，不选择或实现其中一种方案。

## 实验保留与证据

远端 `/data/codex-cuda-runtime-audit-20261106/modified-wheel` 保存下载文件、元数据及日志。独立测试容器保留用于后续复核，没有改动原 Runtime 镜像或其他容器。没有提交或推送这些报告。

本地 `modified-wheel-evidence/` 中保存了原始元数据、文件清单、安装日志、导入检查、推理尝试与 UT 失败日志、最终包状态。wheel 本体不放入 Git；本地下载副本位于 `/tmp/cuda-runtime-audit-20261106/vllm-0.32.0+flagos-cp312-cp312-linux_x86_64.whl`。
