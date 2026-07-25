# Qwen3.6 Ascend 特性开关说明（slot_mapping / conv1d_prepack / mm_ar_rmsnorm）

本文说明 vllm-plugin-FL 在 Ascend NPU 上为 Qwen3.6（27B / 35B-A3B）提供的三个
可选特性的含义、收益区间、收益减弱条件，以及启用方式。

测试平台：910B × 4（TP4），vLLM 0.13 + FULL graph 模式，并发 64，
`gmem 0.9`，`max_num_batched_tokens 2048`（chunked prefill）。

---

## 1. NPU slot_mapping（Triton kernel）

### 含义

vLLM 每个调度步都要把每个请求的 logical token 映射到 KV cache 的物理槽位
（slot mapping）。上游实现是 host 侧 NumPy 计算 + H2D 拷贝。本特性将其替换为
Ascend 上的 Triton kernel（backport 自 vllm-ascend PR #12096），直接在 device
侧完成映射，消除每步的 NumPy 计算和 H2D 小拷贝。

### 启用方式

```bash
# 默认开启（kernel 可用时）；显式关闭：
export VLLM_FL_DISABLE_NPU_SLOT_MAPPING=1   # 回退上游 NumPy + H2D 路径
export VLLM_FL_DISABLE_NPU_SLOT_MAPPING=0   # 开启（默认）
```

### 优势区间

- **decode 主导的场景**（短输入、高 decode 占比）：收益最大
- 实测（35B-A3B，Total token throughput vs 全关基线）：1k 输入 **+1.5%**

### 收益减弱条件

- **长输入 / prefill 主导**：它是 per-step 的 host 侧优化，prefill 计算量
  上来后占比被稀释。实测 4k 基本持平，16k/64k +0.2~0.5%（噪声内）
- 低并发（调度步数少）时绝对收益也变小

---

## 2. conv1d prepack（GDN 卷积权重预打包）

### 含义

GDN（Gated Delta Net）层的 causal conv1d，AscendC kernel 要求权重为
`(width, dim)` 连续布局，而 checkpoint 中的布局是 `(dim, 1, width)`。
本特性在**首次 forward 时一次性**把权重物理重排为 `(width, 1, dim)`
（backport 自 vllm-ascend PR #7555 的 post-load packing，因 vLLM 0.13
loader 没有 GDN post-load hook 而采用懒执行），避免每次 forward 都
materialize 一个 transpose view。

### 启用方式

```bash
# 默认开启；显式关闭：
export VLLM_FL_DISABLE_CONV1D_PREPACK=1   # 走上游缓存转置路径
export VLLM_FL_DISABLE_CONV1D_PREPACK=0   # 开启（默认）
```

注意：关闭时走的也是"加载后预转置 + buffer 缓存"的上游路径（性能与开启
基本等价），差异仅在于权重存储所有权（prepack 原地改写 `conv1d.weight`，
上游方案额外缓存一份 buffer）。因此该特性的 ON/OFF 对比主要验证
**正确性与稳健性**，性能差在噪声级。

### 优势区间

- 短输入、decode 主导场景：实测（slot+conv1d vs slot_only）1k **+3.3%**、
  4k +1.9%；TPOT 1k 降 3.7%

### 收益减弱条件

- **长输入（16k/64k）**：prefill GEMM 计算量占绝对主导，权重转置节省的
  开销被淹没，实测收益 ~0%
- 与上游缓存转置路径对比时（两者 forward 期都是零拷贝），无差异

---

## 3. mm_allreduce_add_rmsnorm 融合（MC2 算子）

### 含义

把 attention 输出投影的 `x @ weight.T → TP all-reduce（ReduceScatter +
AllGather 流水）→ + residual → RMSNorm` 融合为单个 AscendC MC2 算子，
替代 `RowParallelLinear` 内部的独立 all-reduce 加后续单独的
`npu_add_rms_norm_bias`。消除 eager 路径中未重叠的 HCCL all-reduce
（在 35B-A3B TP4 eager decode 中约占步长 18.5%）。

### 启用方式

```bash
# 默认关闭，显式开启：
export VLLM_FL_ENABLE_MM_AR_RMSNORM=1
# 触发阈值（token 数 M），默认 512：
export VLLM_FL_MM_AR_RMSNORM_MIN_TOKENS=512
```

**阈值斟酌**：decode 步的 M = 并发数（通常 ≤ 64），此时融合算子额外的
add_out all-gather 会抵消收益，因此阈值必须明显高于并发度；prefill chunk
（最大 2048）只有 M > 512 才融合。512 沿用 npugraph_ex 融合 pass 的调优值，
是保守合理的默认。TP = 1 时自动禁用（无 all-reduce 可省）。

### 优势区间

- **TTFT（prefill 延迟）**：融合省掉 prefill chunk 中的独立 all-reduce +
  norm。实测 4k 输入 TTFT **-6%**，1k -2.5%
- 总吞吐：1k/4k 约 **+2%**

### 收益减弱条件

- **长输入（16k/64k）**：TTFT 高达数十~数百秒，融合节省的通信开销占比
  太小，实测中性
- **decode 步（M ≤ 并发度）**：低于阈值不触发（设计如此）
- 小 batch / 低并发 prefill（M < 512）不触发

---

## 实测数据汇总

35B-A3B，FULL graph，并发 64，4 个 case（输入/输出 = 1k/4k/16k/64k，输出均
1k，256 条请求），两组并发互测，Total token throughput (tok/s)：

| 配置 | 1k | 4k | 16k | 64k |
|---|---|---|---|---|
| 全关（基线） | 1756.53 | 2727.91 | 3187.24 | 3659.41 |
| slot_mapping | 1783.02 (+1.5%) | 2722.10 (-0.2%) | 3202.15 (+0.5%) | 3666.73 (+0.2%) |
| slot + conv1d | 1842.57 (+4.9%) | 2774.05 (+1.7%) | 3187.11 (~0%) | 3664.24 (+0.1%) |
| slot + mm融合 | 1823.43 (+3.8%) | 2776.03 (+1.8%) | 3186.09 (~0%) | 3657.98 (~0%) |

规律：

1. **收益集中在短输入（1k/4k）**，长输入（16k/64k）全部趋于中性——三个
   特性优化的是 per-step 固定开销或通信开销，prefill 计算量上来后均被稀释
2. 三个特性在所有 case 上**均无回退**，可以常开
3. ±1% 以内为共享机时间窗噪声；conv1d 1k（+4.9%）与 mm 融合的 TTFT
   收益（4k -6%）超出噪声范围，为真实收益

## 快速启用（推荐配置）

```bash
export VLLM_FL_DISABLE_NPU_SLOT_MAPPING=0    # slot_mapping 开（默认）
export VLLM_FL_DISABLE_CONV1D_PREPACK=0      # conv1d_prepack 开（默认）
export VLLM_FL_ENABLE_MM_AR_RMSNORM=1        # mm 融合开（默认关，需显式开启）
export VLLM_FL_MM_AR_RMSNORM_MIN_TOKENS=512  # mm 融合阈值（默认 512）
```
