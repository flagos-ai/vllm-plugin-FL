# vLLM 0.28 Model Runner V1/V2 重大变化与 vllm-plugin-FL 迁移说明

> 适用范围：vLLM `0.28.0`（上游提交 `2cf0a6915ce544dc493a0990f2ea38d81601128a`）与基于该版本的 `vllm-plugin-FL`。
>
> 本文中的 V1/V2 指 **Model Runner**，不是 vLLM V1 Engine。vLLM 0.28 的两种 Model Runner 都位于 `vllm/v1/` 目录下，因此日志中的 `Initializing a V1 LLM engine` 不能用于判断 Model Runner 版本。

## 1. 结论

vLLM 0.28 将 V2 Model Runner 作为大多数纯 Attention 稠密生成模型的默认路径。V2 不是在 V1 上增加少量分支，而是对请求状态、GPU 输入准备、采样、异步流水和 CUDA Graph 管理进行模块化重构。

`vllm-plugin-FL` 的通用跨硬件 `WorkerFL` 已同步 0.28 的双 runner 分支：

- V1：实例化插件维护的 `vllm_fl.worker.model_runner.ModelRunnerFL`。
- V2：实例化上游 `vllm.v1.worker.gpu.model_runner.GPUModelRunner`。

迁移前仍有两道人工限制：插件注册时默认写入 `VLLM_USE_V2_MODEL_RUNNER=0`，平台配置又拒绝 `use_v2_model_runner=True`。本次迁移移除这两道限制，恢复上游 0.28 的自动选择逻辑，同时保留显式 `VLLM_USE_V2_MODEL_RUNNER=0` 的 V1 回退能力。

NVIDIA 最终不再注册为通用 OOT 平台。`NvidiaPlatformFL` 继承上游 `CudaPlatform` 并保持 `PlatformEnum.CUDA`，只把 worker 指向继承上游 GPU Worker 的薄封装 `NvidiaWorkerFL`；该封装仅安装 FL runtime/dispatch hooks，runner 生命周期和 V1/V2 自动选择完全交给 vLLM 0.28。实测表明，把 NVIDIA 标成 `PlatformEnum.OOT` 会让 MLA 与 MoE 进入通用加速器分支，并造成 DeepSeek-V2 decode 结果损坏。

## 2. 运行时如何选择 V1/V2

### 2.1 环境变量优先级

| `VLLM_USE_V2_MODEL_RUNNER` | 行为 |
|---|---|
| 未设置 | 使用 vLLM 0.28 的模型与功能自动选择逻辑 |
| `1` | 显式请求 V2；不兼容功能会在配置检查阶段报错 |
| `0` | 显式使用 V1，用于回归、差异定位和暂未支持的模型/功能 |

建议生产默认不设置该变量。只有在 P0 验证、A/B 对比或定位 runner 差异时才显式设置。

### 2.2 vLLM 0.28 的自动选择规则

- 纯 Attention、非 MoE 的生成模型默认使用 V2。
- 混合 Attention 模型默认使用 V1，除非架构已进入 V2 白名单。
- Attention-free 模型默认使用 V1。
- MoE 模型默认使用 V1，除非架构已进入 V2 白名单。
- Diffusion、DFlash、DSpark 等部分专用路径会强制使用 V2。
- V2 依赖 Triton；环境缺少可用 Triton 时不能启用。

0.28 的 V2 runner 架构允许列表（与 FlagGems 算子黑名单无关）包括：

- `DeepseekV2ForCausalLM`
- `DeepseekV4ForCausalLM`
- `GraniteMoeForCausalLM`
- `InklingForCausalLM`
- `InklingForConditionalGeneration`
- `KimiK3ForConditionalGeneration`
- `LongcatFlashNgramForCausalLM`
- `Qwen2MoeForCausalLM`

实际日志必须出现 `Using V2 Model Runner` 才能确认启用了 V2。只有 `Initializing a V1 LLM engine` 不足以判断。

## 3. V1 与 V2 的重大变化

| 维度 | V1 Model Runner | V2 Model Runner | 影响 |
|---|---|---|---|
| 代码组织 | 主要逻辑集中在约 8k 行的 `gpu_model_runner.py` | 顶层协调器约 2k 行，拆分为 `model_states/`、encoder、sampler、speculator、buffer 等模块 | 更容易为不同模型类型组合能力，插件不能再只维护一份单体 runner 副本 |
| 请求状态 | 每步围绕当前 batch 重建和搬运较多 Python/CPU 状态 | 持久请求状态与单步输入解耦，请求在 batch 中保持稳定 row | 减少调度抖动和 CPU 侧 bookkeeping |
| 输入更新 | CPU 组装完成后同步写入 GPU 输入张量 | `StagedWriteTensor` 分阶段写入，配合 GPU/Triton metadata 和 UVA | 降低 CPU→GPU 更新开销，便于异步流水 |
| 执行模型 | prepare、forward、sample 等步骤耦合在单体 runner 中 | `ModelState`、`EncoderRunner`、`EncoderCache`、speculator/pool runner 等职责分离 | 文本、多模态、pooling、投机解码路径更清晰，但插件 hook 位置发生变化 |
| 同步模型 | 多处同步和 barrier，异步能力是在同步路径上叠加 | async-first，尽量消除每步 barrier | 高并发和短输出场景更容易获得收益 |
| 采样 | Python 控制和通用 PyTorch sampler 占比较高 | Triton sampler、Gumbel 采样及更高效的 logprobs 路径 | 降低 decode 阶段 CPU 与 kernel launch 开销 |
| CUDA Graph | V1 runner 内维护 capture/replay 与 dummy run 逻辑 | 独立 graph manager，显式管理 graph、buffer 和多步投机解码 | 图模式职责更清晰；插件 graph 适配需通过平台接口和运行测试验证 |
| Warmup | `dummy_run` 同时承担 profile、编译和多种预热职责 | 对执行、采样、speculator 等分别提供更明确的预热入口 | 避免扩展功能继续堆叠在 dummy run 上 |
| 投机解码 | 逻辑与主 runner 紧耦合 | speculator 模块化，并支持 fused multi-step 流水 | 性能潜力更大，但 0.28 只支持部分 speculative method |
| 扩展方式 | 复制并修改单体 runner 较直接，但升级 diff 很大 | 更依赖 Platform、Worker、模型状态和注册接口 | FL 后续应减少复制上游 runner，优先在稳定接口与 dispatch 层扩展 |

## 4. 对 vllm-plugin-FL 的直接影响

### 4.1 NVIDIA V1/V2 都经过的 FL 层

- `NvidiaPlatformFL`：保留上游 CUDA 的平台枚举、attention/MoE 分支和配置检查，只追加 FL worker 选择。
- `NvidiaWorkerFL`：继承上游 `vllm.v1.worker.gpu_worker.Worker`，在构造阶段安装 FlagGems、FL OOT op 和 dispatch runtime；其余设备初始化、内存探测、执行、采样和 V1/V2 runner 构造继续使用上游实现。
- dispatch/custom ops：上游 V1 或 V2 调用已被 FL 注册或替换的 op 时，仍按 NVIDIA blacklist 与 backend policy 路由；未替换的算子保持原生 CUDA 实现。
- 非 NVIDIA 后端仍使用通用 `PlatformFL` / `WorkerFL` 路径，本轮不改变其平台语义。

### 4.2 NVIDIA 不再经过的插件 runner 副本

NVIDIA 的原生 worker 薄封装无论自动选择 V1 还是 V2，都不实例化插件维护的 `ModelRunnerFL`。因此所有只写在该类内部、又没有提升到 Platform/Worker/dispatch 的定制都不会自动生效，重点包括：

- runner 内部的 IO dumper hook；
- 直接写在 V1 prepare/forward/sample 路径中的诊断逻辑；
- 只在 `ModelRunnerFL` 中完成的 buffer、warmup 或模型特判；
- 通过复制 V1 runner 获得、但 V2 模块没有对应接入点的行为。

这也是 P0 不能只测 Qwen3.8 的原因：必须覆盖稠密文本、MoE+MLA、音频、多模态、混合 Attention、OCR 和大模型 TP。

### 4.3 建议的后续代码方向

1. 保留 V1 `ModelRunnerFL`，作为非 NVIDIA 跨硬件后端的兼容路径；NVIDIA 优先复用上游 runner。
2. 新增 FL 能力时优先放在 Platform、Worker、dispatch 或上游提供的模块化接口中。
3. 将必须覆盖 V2 内部状态的功能做成小型 adapter/hook，不复制整个 V2 目录。
4. 为 runner 选择保留契约测试：默认不改写环境变量，显式 `0/1` 必须被尊重。

## 5. V2 在 0.28 的主要限制

以下功能不能假设已经与 V2 组合兼容，启用时应以 0.28 配置校验和实测为准：

- 非 MLA 的 Prefill Context Parallel；
- stock `torch.compile` 模式；
- Tensor Parallel sequence parallelism；
- `external_launcher` 与 Pipeline Parallel 的组合；
- Dual Batch Overlap（DBO）；
- elastic Expert Parallel；
- custom logits processors；
- prompt embeds；
- KV sharing fast prefill；
- `ngram`、`ngram_gpu` 等未进入 V2 支持集合的 speculative method。

V2 在 0.28 支持的 speculative method 主要是 `eagle`、`eagle3`、`mtp`、`dflash`、`dspark`，具体组合仍有模型和并行配置约束。

## 6. P0 串行测试矩阵

成功标准统一为：模型完成真实权重加载和至少一次有效推理，输出非空且语义/模态结果合理；日志确认实际 runner；无 OOM、CUDA error、worker crash 或未捕获异常。

| 顺序 | 模型 | 自动 runner 预期 | 核心覆盖 | GPU 建议 | 状态 |
|---:|---|---|---|---:|---|
| P0-1 | `Qwen/Qwen3-8B` | V2 | 稠密文本、GQA；依次验证 eager 与 CUDA Graph | 1×A800 | 最终回归已通过 eager + CUDA Graph（2/2） |
| P0-2 | `deepseek-ai/DeepSeek-V2-Lite-Chat` | V2（架构允许列表） | MoE、MLA；单卡后可补 TP2 | 1×A800 | 已通过 eager + CUDA Graph（2/2） |
| P0-3 | `openmoss/MOSS-Transcribe-Diarize` | V2 | 0.28 新增音频、转写/说话人区分 | 1×A800 | 已通过真实音频转写（2/2） |
| P0-4 | `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` | V2 | 0.28 新增图像、多图/视频输入 | 1×A800 | 已通过 eager + CUDA Graph，图像输出 `stop` |
| P0-5a | `Qwen/Qwen3.5-0.8B` | V1 | 混合 GDN + full attention 的自动回退 | 1×A800 | 已通过自动 V1 文本推理，输出以 `Paris` 开头 |
| P0-5b | `Qwen/Qwen3.5-0.8B` | 强制 V2 | 混合 Attention、视觉、MTP 的实验兼容性 | 1×A800 | 已通过 V2 + 1-token MTP 图像推理，输出 `STOP` |
| P0-6a | `PaddlePaddle/Unlimited-OCR`（ModelScope） | V1 | 0.28 新增 MoE/R-SWA、OCR、本地图片资产 | 1×A800 | 已通过，官方百度图片识别出 `Bai du 百度` |
| P0-6b | `PaddlePaddle/Unlimited-OCR`（ModelScope） | 强制 V2 | OCR 模型 V2 实验兼容性 | 1×A800 | 已通过，日志确认 V2，识别结果与 V1 一致 |
| P0-7a | `Qwen/Qwen3.8-27B` | 强制 V2 | 混合 GDN/full attention、文本、TP2 | 2×A800 | 已通过，语义断言命中 `Paris` |
| P0-7b | `Qwen/Qwen3.8-27B` | 强制 V2 | 原生视觉塔、图片输入、TP2 | 2×A800 | 已通过，stop-sign 图片输出包含 `STOP` |
| P0-8 | `meituan-longcat/LongCat-Flash-Lite` | V2（架构允许列表） | 约 69B 大模型、MoE/长上下文、TP4 | 4×A800 | 已通过，129GB BF16 权重、原生 TritonExperts、CUDA Graph、2 条生成均成功 |

测试必须按表中顺序执行。每个模型失败时先保存完整日志并定位失败阶段，再继续下一个；不能用后续模型的成功掩盖前一项失败。

### 6.1 验证工具与提交范围

P0 验证期间曾临时扩展 YAML 驱动的 inference/serving smoke harness，用来覆盖 runner 断言、文本语义断言、多模态 placeholder、本地图片以及音频 multipart 请求。这些临时 harness 和模型 YAML 已在 A800 上执行完毕，但按本次 PR 的范围要求不提交。

本 PR 在 `tests/` 下只保留 `tests/unit_tests/**` 的必要改动；`tests/e2e_tests/**`、`tests/models/**`、`tests/platforms/**` 和 `tests/utils/**` 均保持与 `upstream/main` 一致。P0 实测结果与日志位置仍记录在第 9 节，作为 NVIDIA 适配的验证证据。

## 7. 建议记录项

每项至少记录：

- 模型仓库与本地权重路径；
- vLLM、插件 commit、容器镜像；
- `VLLM_USE_V2_MODEL_RUNNER` 值和日志中的实际 runner；
- TP、dtype、`max_model_len`、`gpu_memory_utilization`、eager/graph；
- 权重加载、engine init、首轮推理耗时；
- 输入摘要、输出摘要；
- 峰值显存及错误日志路径。

## 8. 上游参考

- [vLLM 0.28 runner 选择与配置](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/vllm.py)
- [Model Runner V2 设计文档](https://github.com/vllm-project/vllm/blob/v0.28.0/docs/design/model_runner_v2.md)
- [vLLM 官方 V2 Model Runner 介绍](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-03-24-mrv2.md)
- [vLLM 0.28 模型注册与测试清单](https://github.com/vllm-project/vllm/blob/v0.28.0/tests/models/registry.py)

## 9. 本次 A800 实测记录

- 机器：`bm-baai-dx-zone1-lc-a800-80g-15-171`，8×NVIDIA A800-SXM4-80GB。
- 镜像：`vllm-fl-test-env:0.28.0`（本地镜像 ID `878fccdd...`）。
- vLLM：`0.28.0`。
- 完整单测：`372 passed, 19 skipped`；NVIDIA 平台/blacklist 聚焦测试：`18 passed`。
- 静态验证：改动 Python 文件通过 Ruff（忽略仓库既有的 `UP007`/`UP045` 注解风格），`git diff --check` 通过；验证期间使用的 CUDA e2e 配置均成功解析，但未纳入本 PR。
- 详细 P0 推理日志：`/nfs/wlx/adapt/nvidia-vllm-0.28.0/logs/p0-*`；最终 Qwen3/DeepSeek-V2/LLaVA 回归：`nvidia-final-regressions.log`；最终单测：`nvidia-final-validation.log`。

已完成结果：

- Qwen3-8B：最终代码状态的 V2 eager 与 CUDA Graph 回归均完成有效文本生成（2/2）。
- DeepSeek-V2-Lite-Chat：V2 eager 与 CUDA Graph 均通过，`The capital of France is` 输出以 `Paris.` 开头；这项严格断言可捕获此前 OOT 平台语义导致的重复词损坏。
- MOSS-Transcribe-Diarize：V2 CUDA Graph 服务启动和真实 `/v1/audio/transcriptions` 请求通过，输出包含 `Mary had a little lamb`；实测新增 `gelu`、`clamp`、`le_scalar` 三个 NVIDIA FlagGems blacklist 项。
- LLaVA-OneVision-2-8B-Instruct：V2 eager 与 CUDA Graph 均完成真实 stop-sign 图像推理，输出包含 `stop`；首次 graph 失败由共享机器外部任务抢占显存引起，换至显存充足设备后通过。
- Qwen3.5-0.8B 自动模式：实际选择 V1，混合 GDN/full-attention 模型完成文本生成，输出以 `Paris` 开头；按真实错误补充了精确的 FlagGems 重载黑名单。
- Qwen3.5-0.8B 强制模式：日志确认 `Using V2 Model Runner`，1-token MTP 草稿层加载成功，真实 stop-sign 图像推理输出 `STOP`。
- Unlimited-OCR 自动模式：实际选择 V1；官方百度图片完成 OCR，输出 `title [0, 0, 999, 999]Bai du 百度`。
- Unlimited-OCR 强制模式：日志确认 `Using V2 Model Runner`；同一图片得到与 V1 一致的语义结果。
- Qwen3.8-27B 文本：TP2、强制 V2，日志确认 GDN prefill/decode 路径，`The capital of France is` 输出包含 `Paris`。
- Qwen3.8-27B 视觉：TP2、强制 V2，真实 stop-sign 图片输出包含 `STOP`。
- LongCat-Flash-Lite：TP4、V2，加载 128.69GiB BF16 权重并使用上游 `TritonExperts`；CUDA Graph 和 2 条真实生成通过，法国首都输出包含 `Paris`。

NVIDIA 默认配置继续采用黑名单策略。新增项均来自 A800 真实推理失败，按首次触发模型归类如下；未在表内且未复现失败的 FlagGems 算子仍保持启用：

| 触发模型 | 失败算子/路径 |
|---|---|
| Qwen3-8B、DeepSeek-V2-Lite | 基础 pointwise、索引、padding、采样，以及 FL fused-MoE 内部激活路径 |
| MOSS-Transcribe-Diarize | `gelu`、`clamp`、`le_scalar` |
| LLaVA-OneVision-2 | `cos`、`sin`、`floor_divide`、标量幂路径 |
| Qwen3.5-0.8B | `rsqrt`、`sigmoid`、比较/where/除法重载、`bitwise_not`、`clamp_` |
| Unlimited-OCR | `sqrt`、`bitwise_or_tensor`、`sum` |
| LongCat-Flash-Lite | `mul_`、`ge_scalar`、`eq_scalar`、`repeat_interleave_self_tensor`、`bitwise_and_tensor`；FL `fused_moe` 回退上游 CUDA 实现 |

Hopper 专用配置是显式 opt-in 路径，本轮没有 H100 实测，因此没有把 A800 新增项复制到该文件；默认 NVIDIA 配置仍适用于未启用 Hopper 专用优化的 H100。

P0-1 至 P0-8 均已按顺序完成；最终代码状态的关键模型回归与完整单测结果也记录在本节。
