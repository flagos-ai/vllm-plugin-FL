# MUSA TP PIECEWISE graph 问题分析与修复

## 结论

Qwen3.6-27B 在 MUSA、TP=4、PIECEWISE graph 模式下启动失败，并不是单一原因造成的。排查中确认了三个相互独立、但会依次暴露的问题：

1. MCCL collective 不能在 MUSA stream capture 中执行，必须留在 eager 区间。
2. torch_musa 单个 context 同时存活的 graph 资源约有 2048 个上限，默认 51 个 capture size 会超过该上限。
3. 为修复 eager collective 输出地址变化而全局开启 `cudagraph_copy_inputs`，会在 vLLM 的全模型编译包装层中 clone 动态输入。Qwen3.6 的 M-RoPE `positions` 因此从非连续 stride `(8193, 1)` 变成连续 stride `(8192, 1)`，触发 MUSA Inductor/Triton kernel 非法访存。

最终方案是把输入复制限制在真正需要它的 MUSA TP PIECEWISE 子图中：全模型输入不再复制，collective 后的 graph segment 仍在 replay 前把运行时值复制回 capture 时的固定缓冲。配合 collective 切分和 graph 数量限制后，Qwen3.6-27B TP=4 已完成启动、graph capture，并连续通过两次真实请求。

## 背景

vLLM 的 PIECEWISE graph 模式会把 attention、collective 等边界留在 eager 模式，再分别捕获边界之间的计算子图。TP=1 没有张量并行通信；TP>1 会在模型层之间执行 all-reduce 等 collective，因此会产生更多 eager/graph 边界和更多待捕获子图。

MUSA 当前有两个已确认的运行时约束：

- MCCL collective 不能进入 stream capture；
- 单个 MUSA context 同时存活的 native graph 资源数量约为 2048。

把 collective 切出 graph 后，每次 eager collective 又会返回地址不同的新 tensor。graph replay 仍要求输入地址与 capture 时一致，因此 collective 后的计算子图需要保留 capture-time buffer，并在 replay 前复制最新值。

问题出在最初把这个“子图边界需要复制”的要求扩大成了全局配置：

```python
compilation_config.cudagraph_copy_inputs = True
```

该配置不仅影响 PIECEWISE 子图，也会影响 vLLM 上游包裹完整编译模型的 `make_copy_and_call()`。

## 问题现象

### MCCL capture 失败

collective 未切出 graph 时，TP 模式会报：

```text
MUSA error: operation not permitted when stream is capturing
```

### graph 资源耗尽

collective 切出后，Qwen3-0.6B TP=2 使用默认 51 个 capture size 时稳定停在 `23/51`，随后在 `capture_end()` 报非法访存：

```text
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 45% ... 23/51
RuntimeError: MUSA error: an illegal memory access was encountered
```

### 首次 replay 卡在 collective 后的子图

限制 capture size 后服务能够启动，但第一个生成请求会卡住。诊断显示，eager `wait_tensor` 的输出地址与 capture 时不同，而后续 graph 仍使用旧地址。

### Qwen3.6-27B TP=4 在 capture 前非法访存

加入全局输入复制后，大模型在 profile/compile 阶段、真正开始 graph capture 之前失败。问题内核为：

```text
triton_poi_fused_clone_copy_index_select_slice_split_0
```

错误最初在 pointwise autotune benchmark 的同步点暴露：

```text
autotune_to_one_config
  -> benchmark_all_configs
  -> triton.testing.do_bench
  -> musa.synchronize()
  -> MUSA error: an illegal memory access was encountered
```

关闭 autotune 只能改变错误暴露的位置，不能修正输入布局或 kernel，因此不是根因修复。

## 如何定位到全局输入复制

### 1. 先把 collective 和 graph 资源问题独立出来

将 vLLM collective、`_c10d_functional` collective 和 `wait_tensor` 设为 splitting op 后，确认 MCCL 不再进入 capture。

单独的 MUSAGraph 最小复现进一步确认了资源边界：

```text
保留全部 MUSAGraph：
  前 2043 次成功，第 2044 次 capture_end() 失败

每次捕获后 reset：
  5000 次全部成功

每次捕获后删除对象并回收：
  5000 次全部成功
```

因此这是“同时存活 graph 数量”上限，不是累计句柄 ID 永久耗尽。插件据实际 `GraphWrapper 数量 × 每个 size 的 descriptor 数量` 动态限制 capture size，把预算控制在 2000 以内。

### 2. 确认大模型失败发生在 graph capture 之前

Qwen3.6-27B TP=4 已完成权重加载，但在 Inductor profile run 中失败，日志尚未进入 graph capture。这排除了“graph 数量仍然过多”对该次非法访存的解释。

### 3. 核对四个 rank 的输入

在 Triton benchmark 前同步并打印问题 kernel 的输入，确认：

- 四个 rank 都绑定在各自的 rank-local MUSA device；
- `positions` 的 shape、值域和 device 正确；
- 输入值已初始化，没有越界索引；
- 开启同步执行后仍在同一 kernel 失败，不是更早异步错误延迟上报。

因此 rank binding、未初始化数据、索引越界和异步错误都不是根因。

### 4. 找到唯一稳定变化：stride 被 clone 改写

原始 M-RoPE `positions` 是一个带 padding 的非连续 view：

```text
shape=(3, 8192), stride=(8193, 1)
```

当全局 `cudagraph_copy_inputs=True` 时，上游 `make_copy_and_call()` 会执行：

```python
input_buffers[i] = runtime_tensor.clone()
```

clone 后输入变为连续布局：

```text
shape=(3, 8192), stride=(8192, 1)
```

A/B 结果稳定一致：

- stride `(8193, 1)`：候选 launch config 可以执行；
- stride `(8192, 1)`：首次执行问题 fused kernel 即非法访存；
- 四个 rank 在同一条件下同时失败。

这说明全局输入复制改变了模型动态输入的内存布局，触发了 MUSA backend 对该连续布局处理不正确的问题。

### 5. 用严格配置验证修复点

关闭全局 copy、只给 TP PIECEWISE segment 开启 copy 后，在显式保持 `triton.autotune_pointwise=True` 和 `MUSA_LAUNCH_BLOCKING=1` 的条件下，原失败点成功通过：

```text
Compiling a graph for compile range (1, 8192) takes 3.77 s
torch.compile and initial profiling/warmup run together took 24.87 s
```

这个实验只用于证明原编译期非法访存来自全局 copy；`MUSA_LAUNCH_BLOCKING=1` 会显著放慢后续 capture，因此最终功能验收使用正常运行配置。

## 修复方案

### collective 留在 eager 区间

以下操作作为 splitting op，不进入 MUSA graph capture：

```text
vllm::all_reduce
vllm::all_gather
vllm::reduce_scatter
vllm::patched_fused_scaled_matmul_reduce_scatter
_c10d_functional::all_reduce
_c10d_functional::all_gather_into_tensor
_c10d_functional::reduce_scatter_tensor
_c10d_functional::wait_tensor
```

同时在 process group 创建前设置 `TORCH_MCCL_BLOCKING_WAIT=1`，避免 watchdog 线程在 graph capture 期间调用 MUSA API。

### 动态限制同时存活的 graph 数量

根据实际 wrapper 和 descriptor 数量计算可保留的 capture size 数量，将单个 context 的 live graph 预算保持在 2000 以下。这个计算不依赖具体模型，也没有写死实验中的 `23` 或 `10`。

### 只在 segment 层复制输入

平台配置不再把输入复制强制改为 `True`，默认配置保持：

```python
cudagraph_copy_inputs = False
```

`GraphWrapper` 仅在以下条件同时满足时启用 segment-local copy：

```text
device_type == musa
tensor_parallel_size > 1
runtime_mode == PIECEWISE
```

capture 时保存该 segment 的 tensor 输入；replay 时如果地址变化，则先校验 shape、stride、dtype 和 device，再复制到 capture-time buffer。这样同时满足两项要求：

- 全模型动态输入保持原始 stride，不再被全局 clone；
- eager MCCL 后的新 tensor 仍能写回 graph 记录的固定地址。

如果用户显式启用了 `cudagraph_copy_inputs`，`GraphWrapper` 仍保留原有全局语义。

## 影响范围

自动启用 segment-local copy 的范围严格限制为 MUSA、TP>1、PIECEWISE graph：

- 不影响 MUSA TP=1；
- 不影响 eager 模式；
- 不影响 FULL graph；
- 不改变 CUDA、NPU 等其他平台的默认行为；
- 不改变用户显式设置 `cudagraph_copy_inputs=True` 时的行为。

修复没有修改模型实现或 M-RoPE 算法，只调整 graph 边界、资源预算和输入缓冲管理。

## 实机验证

### Qwen3.6-27B TP=4 PIECEWISE

启动命令：

```bash
vllm serve /data/models/Qwen/Qwen3.6-27B \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 50553 \
  --served-model-name qwen \
  --pipeline-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.4 \
  --disable-custom-all-reduce
```

使用正常运行配置，没有设置 `MUSA_LAUNCH_BLOCKING`，也没有手工覆盖 compilation config。关键启动日志：

```text
cudagraph_mode: PIECEWISE
cudagraph_copy_inputs: False
triton.autotune_pointwise: False

(Worker_TP0) INFO Compiling a graph for compile range (1, 8192) takes 19.76 s
(Worker_TP0) INFO torch.compile and initial profiling/warmup run together took 131.29 s in total
(Worker_TP0) MUSA: Reduced PIECEWISE graph capture sizes from 51 to 10
  (193 wrappers, 1 descriptors per size) to stay below the
  2000-live-graph runtime limit.
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|...| 10/10 [01:49<00:00, 10.99s/it]
(APIServer pid=1) INFO: Application startup complete.
```

连续发送两次 `/v1/chat/completions` 请求，两次均返回 HTTP 200：

```text
(APIServer pid=1) INFO: 127.0.0.1:33168 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1) INFO: 127.0.0.1:42974 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

两次响应的核心字段：

```json
{"object":"chat.completion","model":"qwen","choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":15,"completion_tokens":8,"total_tokens":23}}
{"object":"chat.completion","model":"qwen","choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":15,"completion_tokens":8,"total_tokens":23}}
```

因此最终验收的三项条件均已满足：服务启动、PIECEWISE graph capture 完成、连续两次真实请求成功。

### Qwen3-0.6B 非回归

此前验证结果保持通过：

| 配置 | 结果 |
| --- | --- |
| TP=1 PIECEWISE | 51/51 capture size 完成，请求 HTTP 200 |
| TP=2 PIECEWISE | 动态限制 capture size；64-token、5 次连续、3 路并发和长输入请求均 HTTP 200 |

### 单元测试

真实 MUSA 容器中与本次修复相关的测试结果：

```text
13 passed in 6.28s
```

单元测试只覆盖配置和输入复制条件；Qwen3.6-27B TP=4 的完整实机结果才是本问题的最终验收依据。
