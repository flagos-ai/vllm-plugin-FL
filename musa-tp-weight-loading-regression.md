# MUSA 张量并行权重加载失败问题说明

## 背景

vllm-plugin-FL 的通用插件注册函数 `register_model()` 会在 frontend、EngineCore 和各个 TP worker 中执行。PR #383（提交 `f4319bd`）在这个入口加入了 `patch_vllm_moe_sum()`，用于把 vLLM 的 `moe_sum` 入口接入插件算子分发。

与此同时，MUSA 为兼容 vLLM 0.24 的 `DeviceConfig`，曾把字符串 `"musa"` 转成带固定编号的 `torch.device("musa:0")`。这个实现基于旧版本 torch_musa 中无编号设备不可用的假设；当前软件栈已经支持 `torch.device("musa")`。

这两处代码单独看都不像权重加载逻辑，但它们会影响 TP worker 初始化设备的时机和模型参数实际分配到哪张卡。

## 问题现象

Qwen3.6-27B 和 Qwen3.6-35B-A3B 使用 TP=4 启动时，权重加载不能正常完成。排查中观察到过两类表现：

- rank 0 可以继续加载，rank 1～3 在 `param.data.copy_(loaded_weight)` 附近报 `MUSA error: unknown error`；
- 服务停在 mamba/attention page 对齐之后，非零 rank 高 CPU 自旋并生成设备 dump，GPU 利用率为 0。

典型调用栈如下：

```text
flag_gems/ops/copy.py: copy_
vllm/model_executor/model_loader/weight_utils.py: default_weight_loader
vllm/model_executor/models/qwen3_5.py: load_weights
vllm_fl/worker/model_runner.py: load_model
vllm_fl/worker/worker.py: load_model
RuntimeError: MUSA error: unknown error
```

错误虽然在 `copy_` 中暴露，但 `copy_` 不是根因。设备错误可能延迟到后续算子才被报告；实际异常发生在更早的 worker 初始化和参数放置阶段。

## 定位过程

### 1. 将回归缩小到通用插件注册阶段

保持模型、基础镜像和启动参数不变，仅替换插件代码，得到以下对照结果：

| 实验 | 结果 | 结论 |
| --- | --- | --- |
| 使用当前插件 | 失败 | 回归可以稳定复现 |
| 只换回旧版 `platform.py` | 失败 | 平台类本身不是唯一原因 |
| 只换回旧版 `vllm_fl/__init__.py` | 通过 | 问题受通用注册入口控制 |
| 只禁用 Qwen3.5 注册 hook | 失败 | 与模型注册本身无关 |
| 导入 `moe_sum` patch 模块但不执行 | 通过 | 模块定义没有副作用 |
| 执行 `patch_vllm_moe_sum()` | 失败 | 执行 patch 会触发问题 |

`patch_vllm_moe_sum()` 为替换上游入口会导入 `vllm._custom_ops`。原调用发生在 `register_model()` 中，早于 TP worker 的 `set_device(local_rank)` 和分布式初始化。这个导入会提前触发设备内核初始化，使非零 rank 在错误的设备上下文中继续运行。

### 2. 确认 DeviceConfig 将所有 rank 指向设备 0

Issue #536 提供了另一组直接证据：TP=4 时，四个 rank 的权重集中在同一张物理卡，而其余三张卡只有少量上下文显存。vLLM 0.24 的模型加载器会把 `device_config.device` 作为参数创建和加载的设备上下文；当该值固定为 `musa:0` 时，每个 worker 即使已经执行 `set_device(local_rank)`，参数仍会在设备 0 上创建。

最小实验也能复现二者差别：

```python
torch.musa.set_device(1)
with torch.device("musa:0"):
    fixed = torch.zeros(4)
print(fixed.device)  # musa:0

torch.musa.set_device(1)
with torch.device("musa"):
    local = torch.zeros(4)
print(local.device)  # musa:1
```

这说明固定 index 的 `DeviceConfig` 会覆盖 worker 的本地设备选择。跨设备参数复制随后可能导致设备 fault、非零 rank 自旋，或让集合通信永久等待。

## 根因

TP 权重加载失败由两个同属设备初始化边界的问题共同造成：

1. `patch_vllm_moe_sum()` 在通用插件注册阶段执行，导入 `vllm._custom_ops` 时 TP worker 尚未绑定 rank-local device。
2. MUSA `DeviceConfig` 被固定为 `musa:0`。vLLM 0.24 的 loader 直接使用该字段创建模型，导致非零 rank 的参数也落到设备 0。

第一项解释了 PR #383 后出现的明确回归；第二项解释了 #536 中四个 rank 权重集中到一张卡的现象。只在 MUSA 上跳过 `moe_sum` adapter 可以暂时避开第一项，但没有修正调用时序，也不能解决参数设备放置问题，因此不是完整修复。

## 解决方案

最终修改不再增加 MUSA 平台例外，而是把初始化顺序修正为：

```text
set_device(rank-local)
    -> 初始化分布式环境
    -> 安装 moe_sum runtime patch
    -> 创建模型并加载权重
    -> 校验模型参数设备
```

具体改动如下：

1. `register_model()` 不再执行 `patch_vllm_moe_sum()`，因此 frontend、模型检查进程和未绑定设备的 worker 都不会在通用注册阶段触发设备内核初始化。
2. `WorkerFL.load_model()` 在 `init_device()` 已完成设备绑定和分布式初始化之后、实际模型加载之前安装 adapter。所有平台继续使用同一适配器，不再通过 `device_type != "musa"` 固化平台例外。
3. MUSA `DeviceConfig` 初始值改为无 index 的 `torch.device("musa")`；worker 调用 `set_device()` 后，再把 `device_config.device` 回写为明确的 `musa:<local_rank>`，保证 loader 和后续消费者使用当前 rank 的设备。
4. 模型加载结束后检查所有 MUSA 参数是否位于当前 worker 设备。如果再次出现跨 rank 放置，服务会在初始化阶段给出明确错误，而不是在后续 `copy_` 或 collective 中以设备 fault 的形式暴露。

## 影响范围

- 通用插件注册不再安装需要设备上下文的 runtime adapter；该动作只在实际加载模型的 worker 中发生。
- CUDA 及其他平台仍会安装并使用 `moe_sum` adapter，功能没有被移除。
- MUSA loader 获得明确的 rank-local device，修复范围覆盖 dense、MoE 和 hybrid 模型，而不是只针对 Qwen3.6。
- TP=1 的设备仍为 `musa:0`，行为不变。
- eager 和 graph 模式共用同一套 worker 初始化及权重加载流程，因此都受此修复保护；graph replay 的兼容问题另行处理。

## 验证

单元测试覆盖以下契约：

- general plugin 注册不触发 `moe_sum` patch；
- TP=2 和 TP=4 的每个 worker 都遵守 `set_device -> dist init -> patch -> load_model`；
- `device_config.device` 与模型参数均位于各自的 `musa:0...N-1`；
- 参数出现在其他 rank 的设备上时会立即报错；
- 上游 `vllm._custom_ops.moe_sum` 入口实际命中 worker 安装的 adapter；
- MUSA `DeviceConfig` 在 frontend 阶段保持无 index。

最终三提交分支上的相关测试结果：

```text
50 passed in 7.58s
```

实机输出将在最终 TP=4 eager/graph 验证完成后补充。

## 关联信息

- 回归来源：PR #383，提交 `f4319bd`
- 设备放置复现与分析：Issue #536
- 修复提交：`9ba1eda` (`fix(musa): bind TP workers before runtime patches`)
