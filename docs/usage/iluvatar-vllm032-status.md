# BI-V150：vLLM 0.32.0 实机适配记录（进行中）

## 环境及来源

- 原始 `/usr/local` vLLM 0.28.0 环境保留不变。
- 隔离解释器：`/workspace/envs/iluvatar-vllm032/bin/python`。
- 隔离环境实际导入 vLLM **0.32.0**，EMPTY 构建，无依赖覆盖安装，复用厂商 PyTorch 2.10.0。
- 源码：`/workspace/sources/vllm-0.32.0`，来自精确 tag 归档。
- 归档 SHA256：`bf592c132f27bbb5f65cff0bbd2ee873eae722b79c8286934fd5b5cf88b43d61`。
- 插件源路径：`/workspace/vllm-plugin-FL`。执行时显式设置 `PYTHONPATH` 为此路径，避免继承环境中的全局 site-packages 覆盖隔离 vLLM。
- FlagTree：0.6.2a3+iluvatar3.6；FlagGems：5.3.5。
- FlagGems-vllm 安装源 pin：`b5c2c2249319277ccda4d41aa0f65164bd9b2a19`（安装 helper 的 v0.2.0 源码；包元数据为 0.1.0）。
- 本地 FlagCX：`/mnt/share/user_homes/iluv/FlagCX`，commit `6143384`，已有 CoreX 动态库；运行验证中。

## 已观察通过

1. 实际 0.32 解释器插件导入；原始全局 0.28 未改动。
2. `regression032-v7.log`：99 passed，包括 runner 选择、非目标厂商隔离、PP 句柄、新 runner 导入、大块清零配置及 FlagCX 初始化失败关闭回归。
3. `worker-aux-regression.log`：5 passed，包括删除旧配置字段后的 KV cache 初始化回归。
4. `smoke032-auxfix.log`：Qwen3-4B、BF16、TP1、eager，真实 BI-V150 完成 16 token 生成。输入 `The capital of France is`，输出 `Paris. The capital of Germany is Berlin. The capital of Italy is Rome.`。

以上日志位于 `/workspace/adapt-logs/`。冒烟成功不等于 NVIDIA 精度对齐，也不等于完整硬件适配验收通过。

## 代码适配范围

- Iluvatar 使用基于上游 0.32 源码迁移的独立 `IluvatarModelRunner`，其他厂商仍使用已有 runner。
- EMPTY 环境 IR 默认优先级不再引用未注册的 `vllm_c`。
- worker 对齐上游 batch-invariant 导入、PP metadata 句柄约定及已移除的旧 routed-expert 配置访问。

## 最新完整模型 gate 进展

- `gate-27b-eager-v3.log`：文本 2 passed、图像 2 passed、混合 1 passed；TP4，BF16，max-model-len=4096。
- `gate-35b-eager-v3.log`：文本 2 passed、图像 2 passed、混合 1 passed；TP4，BF16，max-model-len=4096。
- 请求级 JSON 原样保存在 `/workspace/adapt-logs/gate-{27b,35b}-eager-v3/` 对应模型目录。
- 图像恢复正确输出的受控修复：排除 FlagGems `_scaled_dot_product_flash_attention` 与 `_flash_attention_forward`；仅排除前者仍会重入内层 flash，产生 NaN。其余 attention 注册保留。
- MoE 路由补齐 0.32 padding/bias 语义，GPU n=1/17/256 的选中 score、唯一 expert ID 与权重验证通过；BF16 并列值不要求任意索引顺序一致。
- `probe-tp-comm032-v2.log` 的实际 TP 组使用 CommunicatorFL，双卡 AllReduce 验证通过（独立组件证据，不冒充模型进程内追踪）。
- `regression032-v8.log`：99 passed，包括新增 CI 脚本语法、安装顺序和损坏归档拒绝测试。
- `ci-empty032-install-v2.log`：独立 venv 从精确归档真实构建安装 EMPTY vLLM，导入 0.32.0 且路径来自该 venv；CI setup 已调用安装 helper，清除继承的 PYTHONPATH 并校验精确版本。
- 两模型 graph v1 在实际 PIECEWISE 捕获阶段显存不足，失败日志 `qwen36-{27b,35b}-graph-v1.log` 保留；不能计为通过。
- graph v2 仅将 `GPU_MEMORY_UTILIZATION` 从上游默认 0.92 调整为 0.70，为捕获预留显存；TP4、max-model-len=4096、51 个捕获尺寸及测试集合不变。日志 `qwen36-{27b,35b}-graph-v2.log` 与 `gate-{27b,35b}-graph-v2.log`。27B 再次在 graph warmup 的 FlagGems/FlagTree autotune 路径 OOM；35B 完成捕获但 gate 为 3 passed / 2 failed（并发图像与混合失败），均不得计为通过。
- 35B 的受控诊断：新 image UUID 的串行请求正确，仅换文本仍错；新 UUID 的 8 路并发再次出错，随后同 UUID 串行仍错，表明错误视觉特征进入 encoder cache。视觉编码器未启用 compile。分块 SDPA（含 storage offset）及 BHSD/BSHD 拼接的独立 GPU 探针均与 native 结果一致（max abs error=0），该最小探针尚不能排除完整视觉网络中的其他批量算子。
- 进一步真实 TP4 视觉投影诊断：各 rank 归约前串行/批量输入一致，BF16 FlagCX 归约的最大差为 0.0625，FP32 归约为 2.38e-7。插件新增限定 Iluvatar Qwen 视觉 forward 的 FP32 归约，返回原 dtype，不改变 decoder 归约；21 项相关回归通过，真实编码器最大串行/批量差降至 0–0.0234375。新服务 `qwen36-35b-graph-v4.log` 已完成全部捕获及 5 场景/26 请求，但只有 text_single 场景通过，另外 4 个场景失败（20/26 请求通过）。JSON 实读发现 text_concurrent 的算术也错误，说明视觉归约补丁并未解决 graph 整体正确性；同补丁 eager v4 对照已完整通过 5 场景/26 请求（`gate-35b-eager-v4.log`），目前错误限定在 graph 路径；PIECEWISE 单独模式已执行全部 5 场景/26 请求，3 场景通过、2 场景失败，23/26 请求通过（`gate-35b-piecewise-v1.log`）；图像 9 请求全通过，文本并发与混合仍有错误，尚不能计为 graph 验收通过。v3 曾因旧服务占用端口而启动失败，其 gate 结果无效。
- 27B 同预算诊断运行 `qwen36-27b-graph-memory.log` 再次在 autotune 的 256 MiB cache 分配处 OOM：PyTorch 已分配 29.37 GiB，其中 graph private pool 仅 49.31 MiB；设备剩余 166 MiB。后续显式逐档诊断观察到捕获形状从 512 降至 88 时 allocated 从约 19.4 GB 增至 31.6 GB。独立 GPU 探针确认 FlagGems autotuner 保留诊断参数；限定 Iluvatar 的释放补丁及 16 项回归通过，最小探针保留量从 17.7 MB 降为 0。但完整 memory-v3 仍在 29.99 GiB allocated 时 OOM，证明该修复不足以解决整体问题；GPU 探针确认 EMPTY wheel 缺失 weak_ref_tensor 原语，插件回退持有强引用。现限定 Iluvatar 使用 PyTorch cudagraph_trees 同样的借用 storage 重建；新增 5 项测试通过，涵盖视图元数据、非持有生命周期、replay 和三个形状的共享池复用（后续 allocated 增量低于 1 MiB），既有 graph 测试为 20 passed / 5 skipped。受控重跑使用同一 TP4 / 0.70 / 4096 参数：27B 从 512 到 2 tokens 的 PIECEWISE allocated 保持约 20.12–20.20 GB，35B 已完成 51 个 PIECEWISE 与 35 个 FULL 形状捕获，allocated 约 22.09 GB。27B 也已完成 51 个 PIECEWISE 与 35 个 FULL 捕获，受控重跑不再 OOM。两个完整 gate 的 JSON 已读取：27B 为 4/5 场景、25/26 请求通过，长文本单请求约 30 tokens 后退化为重复感叹号；35B 为 2/5 场景、21/26 请求通过，并发文本、图像和混合仍有错误（`summary-27b-graph-borrowed-v2.txt`、`summary-35b-graph-borrowed-v1.txt`）。因此显存持有已修复，但 graph 生成正确性仍未验收通过。独立 TP4 FlagCX graph 探针对 FP32/BF16、3 个大小各 10 次更新输入后 replay 均精确通过，尚不能据此排除完整模型的通信与状态交互。最新汇总回归 `regression032-v10.log` 为 136 passed / 5 skipped；删除临时打印后 runner 回归 8 passed，补充缺失原语报错与 NVIDIA 隔离回归后 graph 组为 23 passed / 5 skipped（`graph-refs032-final-tests.log`）。`stack032-v10.log` 的实际依赖栈断言通过。
- CI 新增 `_iluvatar_gate.yml` 与 `run_gates.sh`，串行执行两模型 × eager/graph，上传失败时的原始日志；聚合通知在 Iluvatar gate 被跳过时不得成功。平台 registry 继续保持原 opt-in 策略。CI 需将 `ILUVATAR_FLAGCX_PATH` 仓库变量指向容器已有挂载内的实际 FlagCX 构建根目录，缺失即失败，不用空库代替。
- 新增 graph 参数传递与 CI wiring / 缺失模型拒绝回归，8 passed；`bash -n` 和 `git diff --check` 通过。这些检查不等价于远端硬件 CI 已运行。

## CI FlagCX 前置条件

Iluvatar 模型门禁读取上游仓库 Actions 变量 `ILUVATAR_FLAGCX_PATH`，并将其作为
`FLAGCX_PATH` 传入测试容器。它必须指向 **CI 容器内可见**、已为 CoreX 构建的
FlagCX 根目录，而不是开发容器中的历史路径。该目录必须同时包含
`build/lib/libflagcx.so` 和 `plugin/interservice/flagcx_wrapper.py`。

维护者需要先检查 runner 的实际安装位置：如果构建目录位于现有 `/data` 挂载中，
可直接使用对应容器路径；否则先在 `.github/configs/iluvatar.yml` 的
`container_volumes` 中显式挂载真实构建目录，再设置容器路径。仅设置变量不能让
未挂载的宿主机文件出现在容器中。

维护者可在仓库 Settings → Secrets and variables → Actions → Variables 设置该变量，
也可使用有上游仓库管理权限的 GitHub CLI：

```bash
gh variable set ILUVATAR_FLAGCX_PATH \
  --repo flagos-ai/vllm-plugin-FL \
  --body /actual/container/path/FlagCX
```

配置前，在相同 CI 容器中验证真实目录及动态库依赖：

```bash
export FLAGCX_PATH=/actual/container/path/FlagCX
test -f "$FLAGCX_PATH/plugin/interservice/flagcx_wrapper.py"
python -c 'import ctypes, os; ctypes.CDLL(os.path.join(os.environ["FLAGCX_PATH"], "build/lib/libflagcx.so"))'
```

路径是必须由维护者替换的占位符。确认配置后重新运行失败的模型门禁任务。
变量为空或所需文件缺失时，门禁在启动模型前失败；不能通过空库、跳过门禁或
将未执行的模型测试记为通过来解决。`a34eded` 的 CI 已确认因变量为空而退出，
尚未执行模型请求；这与同一轮 MUSA 单测 fixture 字段类型错误是两个独立问题。

## 编译路径进一步对照

- 当前代码的 27B eager 复测已完成，5/5 场景、26/26 请求通过，结果位于 `/workspace/adapt-logs/gate-27b-eager-final/`。
- 35B 保留 Inductor、仅关闭 cudagraph 的对照完成全部请求，2/5 场景、22/26 请求通过（`gate-35b-compile-only-v1/`）。关闭 graph 仍有错误，故不能将原因只归于捕获或 replay。
- 两模型将 compilation backend 改为 `eager`、保留默认混合 graph 的实验在加载权重前失败：平台启用了 Inductor graph partition，而上游 dispatcher 要求 piecewise attention 配置满足相应后端约束。这两次启动不计入生成测试结果。
- 随后保留 `backend=inductor` 和 `FULL_AND_PIECEWISE`，显式设定 `custom_ops=["all"]`、`ir_enable_torch_wrap=false`。双模型 EngineArgs 与 CudagraphDispatcher 配置预检通过，但完整模型均在生成前退出：27B 的 Dynamo 追踪遇到 CachedOp 冷解析的 RLock，35B 的 post-grad pass manager 引用了未导入的 RMSNormQuantFusionPass。已扩展 Iluvatar 编译前 dispatch 预热，并关闭 OOT 平台未导入的三项量化融合；增加实际 pass manager configure 的双模型预检后均通过。完整生成结果见下文；该 Inductor 配置未通过验收，不据此修改全平台 custom ops 或 IR wrapping 默认值。

- 27B 后续编译追踪在 FlagGems SiLU 的日志及类型推导处遇到 graph break。仅忽略日志的真实算子探针仍失败，因此为 Iluvatar 的 FlagGems SiLU 增加 `torch.library.custom_op` 边界，保留原 FlagGems 计算实现，并提供 fake shape 推导。真实设备上三种形状的 fullgraph 动态编译、逐位输出比较及 `opcheck` 均通过。包含该修复和启动脚本配置传递的最新相关回归为 **147 passed**（最近复跑：`iluvatar-regression032-v9.log`，16.18 秒）；启动脚本及 CI 默认/覆盖配置另有 **8 passed**。

- 35B 的上述 `custom_ops=all` / IR wrapping 关闭对照已执行完全部 5 场景、26 请求：**2/5 场景、22/26 请求通过**。失败为并发文本 `text-03`、并发图像 `image-01` / `image-06`、混合 `text-01`；原始 JSON 位于 `/workspace/adapt-logs/gate-35b-graph-customops-v2/`。该配置未解决正确性问题，不作为生产默认值。27B 同配置另加 SiLU 编译边界后也已完成，**4/5 场景、25/26 请求通过**；唯一失败为单条长文本生成中的连续感叹号退化（`gate-27b-graph-customops-v3/`），不能计为 graph 验收通过。
- 两模型采用 `compilation mode=NONE`、`cudagraph_mode=FULL`、`custom_ops=["all"]`、`ir_enable_torch_wrap=false`，实际完成 35 个 FULL graph 捕获尺寸（1–256），各自 **5/5 场景、26/26 请求通过**。原始结果在 `gate-27b-full-nocompile-v1/` 和 `gate-35b-full-nocompile-v1/`。该配置保留 graph replay，不是 `--enforce-eager`；Iluvatar CI 通过 `GRAPH_COMPILATION_CONFIG` 采用此配置，其他平台默认值不变，允许显式覆盖。该对照将错误定位到编译相关路径，但尚未定位具体错误算子。
- 最新代码的 35B `--enforce-eager` 回归为 **4/5 场景、25/26 请求通过**（`gate-35b-eager-final/`）；OCR `image-05` 输出 `HELLO VLM`，缺少一个 `L`，独立重复仍失败，未放宽判据。上游 `--enforce-eager` 同时关闭 JIT warmup；编译和 graph 均为 NONE、保留 JIT warmup 的受控对照也为 **25/26**，同一 OCR 失败，不能归因于缺少预热。逐 token logprobs 对照显示分歧前缀相同：FULL graph 的 `LL` / `LM` 为并列最高，而无 graph 对照偏向 `LM`；这仅说明该处候选接近，尚不能证明误差源头。上游支持的 FP32 输出投影对照也已读完全部 5 场景、26 请求，结果仍为 **25/26**，同一 OCR 输出 `HELLO VLM`（`gate-35b-eager-fp32head/`）。仅提高最终输出投影精度不足以解决该问题，仍需定位更早的数值差异；未改变 OCR 判据。

## OCR 稳定性补充验证

- 相同代码重新启动的 35B eager 服务完成原始门禁，**5/5 场景、26/26 请求通过**，结果在 `gate-35b-eager-reproduce-v2/`。但随后在同一服务上交替发送相同 OCR 请求，未启用 logprobs 的请求也分别得到 `HELLO VLLM` 和 `HELLO VLM`（`ocr-options032.json`）。因此单次通过不能视为 OCR 回归已修复，先前的 25/26 失败仍有效。
- 新的逐 token 对照在第 5 个生成 token 首次分歧：graph 的 `LL` 比 `LM` 高 0.125 log probability，eager 则由 `LM` 比 `LL` 高 0.0625（`serving-ocr-logprobs-v1.json`）。这些分数只描述该次请求的候选差距，未证明误差算子。
- 上游离线 LLM 与 API 服务的默认 token budget 分别为 8192 和 2048，默认最大序列数均为 256。显式匹配 2048/256 后，离线 eager 的首次和四次重复均输出 `HELLO VLLM`；匹配前 eager 和 graph 均输出 `HELLO VLIM`。离线输出不能直接替代服务验收。
- 独立 TP4 FlagCX 确定性探针使用固定有符号输入，在 BF16/FP32、2048/4096/32768 元素上各重复 200 次，四个 rank 均逐位一致（`flagcx-determinism032-rank*.json`）。该探针未复现集合通信波动，不构成完整模型根因排除。

## 最新复测（同一服务，未重启）

- `gate-27b-graph-recheck-v10/`：**5/5 场景，26/26 请求通过**。
- `gate-35b-eager-reproduce-v3/`：**4/5 场景，25/26 请求通过**，单图 OCR 为 `HELLO VLM`。
- `gate-35b-graph-recheck-v10/`：**4/5 场景，25/26 请求通过**，同一 OCR 失败。因此该稳定性问题覆盖 35B 的 eager 和 graph，不能继续将 graph 的历史 26/26 当作稳定修复证据。
- 三组共执行 **15 个场景、78 个请求，13 个场景通过、76 个请求通过，2 个场景/请求失败**；27B eager 本轮未重跑，保留前表的历史结果，不合并为本轮完整 104 请求验收。各组均保存全部五个 JSON；日志最后一项通过不代表整组通过。
- 内存预算修复后，相关回归 **150 passed、1 warning**（`iluvatar-regression032-v10.log`）；三组件依赖断言通过（`verify-stack032-v10.log`）。eager 启动脚本现与 graph 一样透传 `GPU_MEMORY_UTILIZATION`，默认 0.92；这项修复不构成 OCR 数值问题的修复。
- MoE router 固定输入重复 200 次索引/权重一致；GDN prefill 在长度 6/1062、固定初始状态并扰动分配历史时各重复 30 次，输出/最终状态逐位一致。独立算子探针只约束已测条件，不能代替模型通过。
- 上游 0.32 文档的在线可复现性依赖 batch invariance，其已列出的硬件支持不含 Iluvatar；不能仅设置 `VLLM_BATCH_INVARIANT=1` 就声称此后端获得保证。

## 已知问题：35B OCR 归因未定（停止排查）

按用户要求，35B OCR 数值排查到此停止，不再追加 OCR 实验。此项保持为未解决的已知问题；既有 eager/graph OCR 失败仍计为失败，不宣称完整 104 请求矩阵通过。继续进行 CI 配置、依赖核查和相关单元回归。

原始门禁失败保留于 `/workspace/adapt-logs/gate-35b-graph-recheck-v10/` 与 `/workspace/adapt-logs/gate-35b-eager-reproduce-v3/`；复现及采集证据保留于 `/workspace/adapt-logs/ocr-options032.json`、`component-repeat032.json`、`sync-trace032-v2.json` 及对应日志和 `.pt` 文件，均不删除或覆盖。最后一轮同步诊断磁盘记录完整 48 次请求，关闭/普通采集/同步采集/关闭各 12 次，失败数为 4/2/2/2；同步未消除问题。此前监控报告终止，但客户端最终写出了 48 条结果，计数以保存结果为准。

### 已有数值证据

- 服务日志确认 35B 使用 `TritonExpertsFL`；Iluvatar 的 `is_cuda()` 为 false，实际路径为两次 GEMM 加独立激活。此前独立 provider `fused_experts_impl` 的融合 SiLU 批量差异不在该服务调用路径上，不能据此修改模型实现。直接调用实际 `TritonExpertsFL.apply()`，固定首 token 和路由、比较 M=1/6/8/17/128，各重复三次，15 次首 token 输出均逐位一致（`moe-actual032.json`）；该探针仍不覆盖全部真实权重和路由。
- 保持 35B eager 的 TP4、4096 上下文及 0.70 显存预算，增加默认关闭的诊断扩展。每次请求先确认前缀缓存重置成功，在“关闭采集/开启采集/关闭采集”三阶段各执行六次相同 OCR；三阶段分别出现 2/1/2 次 `HELLO VLM`，其余为 `HELLO VLLM`（`serving-trace032.json`）。这说明在该受控条件下，波动可在两种采集状态中复现。
- 六份采集的 prefill 最后一个 token 在所有层均逐位一致，但成功请求 6 与失败请求 8 在首个 decode 步的首层输出已出现差异，最大绝对差为 0.0001220703125，1292 个元素不同（`serving-trace032-comparison-v2.log`）。这仅定位到较早的解码差异；prefill 中间位置、递归状态及首层内部算子尚未排除，不能据此宣称根因或修复完成。

- 首层子模块采集又执行了 36 次冷缓存 OCR，三阶段各 12 次，关闭/开启/关闭采集分别出现 2/2/1 次错误。成功请求 12 与失败请求 15 的首层输入归一化和输入投影逐位一致，而注意力输出投影已有差异；另一失败请求 22 在该位置仍与请求 12 相同。仅采集 rank 0 的部分张量不足以确定根因（`component-repeat032.json`、`inspect-repeat032-12-15.log`、`inspect-repeat032-12-22.log`）。
- 使用真实采集的首层输入和模型 checkpoint 中的 TP0 投影权重，原生 PyTorch 与按实际黑名单启用 FlagGems 的两组复算各执行 200 次，均逐位等于 CPU FP32 参考转 BF16 的结果。某份服务采集的投影输出仍有 14 个元素不符，最大差 0.04296875（`real-linear032.json`、`real-linear-gems032.json`）；尚未在隔离复算中复现服务上下文的差异。FlagGems 采集用行拷贝另测 18 个 stream、各 100 行，全部一致（`copy-record032.json`）。
- 首次全 rank 同步诊断因采集器误判 MoE 工作区轴布局中断，未取得同步阶段结果；布局处理已修正，并通过 prefill/decode 不同尺寸回归测试（`test-trace-layout032.log`）。该中断属于诊断代码问题，不计入模型通过或失败样本。

## 待验证（不得计为通过）

- TP2 CUDA Graph 捕获及 replay（验证中）。TP1 graph 已完成两条 16-token 生成，首条 IDs 与 eager 一致，见 `tp1-graph.json` / `.log`。
- 已通过的补充证据：`probe-flagcx032.log` 双卡 AllReduce 的 1024 个元素均为预期值 3；`tp2-flagcx-zerosfix.json` / `.log` 在设置本地 FLAGCX_PATH 后完成 TP2 eager 两条 16-token 生成。
- FlagGems 的 16 GiB 清零失败通过同时排除 `zeros` 和 `zero_` 修复（原生 `zeros` 会再次调用 `zero_`）；其余 FlagGems 算子保留，独立分配探针通过。
- Qwen3.6-27B 与 Qwen3.6-35B-A3B 完整场景矩阵，含多模态、并发与流式。
- NVIDIA + 原生上游 vLLM 0.32 的逐 token 基准对齐：小模型文本已完成对比但严格精度未通过，详见下节；大模型及多模态对齐未完成。
- 全部厂商 CI 配置、依赖 pin 和完整回归。

`setup.sh` 已安装并断言精确 0.32.0，但完整 CI 工作流尚未在远端 runner 执行；本记录不是发布说明或全量部署成功声明。

## 停止 OCR 排查后的收尾核验

- 相关单元回归在隔离 0.32 解释器完成：**157 passed，1 warning，16.53 秒**，日志 `/workspace/adapt-logs/iluvatar-regression032-final.log`。范围包括 Iluvatar 适配、启动显存参数、runner 选择、PP、dispatch 及 graph 引用生命周期，不等同于全仓库或其他厂商回归。
- `.github/scripts/iluvatar/verify_stack.py` 实际执行通过，日志 `/workspace/adapt-logs/verify-stack032-final.log`；断言覆盖 vLLM 0.32.0、厂商 PyTorch 2.10.0、FlagGems 5.3.5、FlagTree 0.6.2a3+iluvatar3.6、导入来源与可加载的真实 FlagCX 库。
- 只读检查 `install_flaggems_vllm.py` 的 `release_is_installed()` 返回 True，确认当前 FlagGems-vllm 来自 pin 的 `b5c2c2249319277ccda4d41aa0f65164bd9b2a19`，未触发重新安装。
- `setup.sh`、`install_vllm.sh`、`run_gates.sh`、eager/graph 启动脚本的 `bash -n` 通过；CI 安装顺序、精确归档校验、参数透传和门禁 wiring 均由相关单元测试覆盖。CI 保留原始完整门禁及失败状态，没有加入 OCR 过滤、xfail 或成功豁免。
- 按停止排查指令，本次收尾未运行含 OCR 的 `run_gates.sh`，也未触发远端硬件 CI。远端 runner 验证、其他厂商完整回归、完整 104 请求通过及 NVIDIA 逐 token 对齐仍未完成。

## Qwen3-4B 小模型文本精度验证

- 使用真实 NVIDIA A800 原生 vLLM 0.32.0（无 FL 插件）与 BI-V150 FL 进行比较；13 个模型文件的完整 SHA256 一致，包含全部三个权重分片。
- 双端均为 TP1、BF16、eager、seed=0、temperature=0、top_p=1、max_tokens=32；输入 chat template、prompt token IDs、engine 和 sampling 配置由比较器断言一致。
- 严格 token 对齐**未通过**：7 条请求中 4 条完全一致，144 个比较位置中 56 个一致（38.8889%），`all_exact=false`。请求 1、3、4 的首次分歧位置分别为 0、0、8（均从 0 计数）。自由生成分歧后的上下文不同，位置一致率不是 teacher-forced 算子精度指标。
- NVIDIA 使用 PyTorch 2.13.0+cu130 / Transformers 5.19.0，目标端为 PyTorch 2.10.0 / Transformers 5.14.1。依赖、硬件和后端仍有混杂，尚不能归因到具体算子；未进行 teacher-forced 或逐层定位。
- 原始双端结果、比较 JSON、模型哈希及详细报告保留于本地 `adapt-logs/precision-small-a800/`，未将模型权重或运行日志提交到仓库。该验证仅覆盖小模型文本，不认证大模型、多模态、TP>1、graph 或在线服务。
- 本次提交前在隔离 0.32 环境重跑相关单元集：**137 passed，1 warning，16.18 秒**；这是本次所选测试集，不替代此前 157 项历史回归或硬件门禁。
