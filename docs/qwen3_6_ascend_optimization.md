# Qwen3.6 Ascend 推理优化技术文档

> 范围：vllm-plugin-FL 上 Qwen3.6-27B（model_type `qwen3_5`）在 Ascend 910B 上的 GDN（Gated Delta Net）推理优化。
> 涵盖：AscendC 融合算子接入、CANN 自定义算子环境自举、PTO megakernel prefill 加速、batch64 回退修复、Slice 热点修复。
> 分支演进：`add-qwen3_6_ascendc_gdn_ops` → `add-qwen3_6_pto_chunk_gdn`（均基于 `add-qwen3_6_ascend_split_qkv_rmsnorm_mrope`）。
> 环境：vllm 0.13.0（`/vllm-workspace/vllm` editable）、torch_npu、CANN 9.0.0、TP=4。

---

## 1. 总览

| 优化项 | 状态 | 主要收益 |
|---|---|---|
| AscendC GDN 融合算子接入（causal_conv1d / fused_gdn_gating / recurrent_gated_delta_rule / GemmaRMSNorm） | ✅ 完成 | eager TPOT ~155ms；graph TPOT 较 Triton +2%~19%，TTFT -25%~45% |
| CANN 自定义算子环境自动装配 | ✅ 完成 | 免手动 source `set_env.bash`，patch 不再静默回退 |
| PTO megakernel `chunk_gated_delta_rule`（prefill 加速） | ✅ 完成 | eager TTFT +15%~95%；aclgraph TTFT +18%~84% |
| batch64 回退修复（去同步点） | ✅ 完成 | 消除 ~5-10% 回退；1024 case 348.65 ≥ 基线 346.24 tok/s |
| Slice 热点修复（KV 缓存连续布局） | ✅ 完成 | 设备 kernel 总时间 **-66%**（慢 Slice 1.1ms → 2.3us） |

关键结论（TL;DR）：

- 图模式"接入无收益"实际是 **patch 因环境守卫静默回退 Triton**；修复为运行时自动 bootstrap。
- PTO 回退的根因不是 kernel（kernel 比 Triton 快 1.7x），而是 wrapper 在**每层每 prefill 步做 2 次 device→host 同步**；改为 CPU 元数据决策后同步点为零。
- Slice 热点的真凶是 **KV 缓存交织布局 + 恒真比较 bug**，导致每个 full-attention 层每步对整份 KV cache 各做一次 `.contiguous()`（~1ms/次）。

---

## 2. AscendC GDN 融合算子接入

### 2.1 算子与注册

FL 仓库已在 `csrc/ascend` 编译注册到 `torch.ops._C_ascend`（`TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)`，`csrc/ascend/torch_binding.cpp`），单测在 `tests/custom_ops_tests/`：

| 算子 | schema 要点 | 用途 |
|---|---|---|
| `npu_causal_conv1d_custom` | `(output, x, weight, conv_state, bias?, query_start_loc?, cache_indices?, initial_state_mode?, num_accepted_tokens?, activation_mode, pad_slot_id, run_mode) -> Tensor` | GDN 卷积；`run_mode=0` varlen prefill / `1` decode·spec update；weight 需 `(width, dim)` |
| `npu_fused_gdn_gating` | `(A_log, a, b, dt_bias, beta=1.0, threshold=20.0) -> (g, beta_output)` | `g = -exp(A_log)*softplus(a+dt_bias)`（fp32），`beta=sigmoid(b)` |
| `npu_recurrent_gated_delta_rule` | `(q, k, v, state(a!), *, beta?, scale?, actual_seq_lengths?, ssm_state_indices?, num_accepted_tokens?, g?, gk?) -> Tensor` | decode/spec 递归；q/k 需先 L2Norm；state `(B, Hv, Dv, Dk)` 原地更新；输出固定 bf16；`actual_seq_lengths` 为 cu_seqlens 风格 `[0, l1, ..., lB]`（batch = numel-1）；`MAX_MTP` 限制单序列 token 数 |
| `npu_gemma_rms_norm` | `(x, gamma, epsilon=1e-6) -> (y, rstd)` | 内部实现 Gemma `(1+w)` 语义（已数值验证） |
| `npu_add_rms_norm_bias` | `(x1, x2, gamma, beta?, epsilon) -> (y, rstd, x)` | residual 路径 |

### 2.2 patch 实现

新增 `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py`，注册进 `patch.py::apply_ascend_patches()`，参照 vllm-ascend `vllm_ascend/ops/gdn.py`（`AscendGatedDeltaNetAttention`）与 `ops/layernorm.py` 移植到 vllm 0.13 的 `GDNAttentionMetadata`：

- 替换 `Qwen3NextGatedDeltaNet._forward_core`（上游类；FL 的 `Qwen3_5GatedDeltaNet` 继承它，经 `torch.ops.vllm.gdn_attention_core` 按 prefix 分发）：
  - spec：`npu_causal_conv1d_custom(run_mode=1, num_accepted_tokens=...)` + rgdr（`ssm_state_indices=spec_state_indices_tensor.flatten()`）
  - prefill：`npu_causal_conv1d_custom(run_mode=0, initial_state_mode=has_initial_state)` + Triton `chunk_gated_delta_rule`（保留）
  - decode：`npu_causal_conv1d_custom(run_mode=1)` + rgdr（q/k 先 `l2norm_fwd`，`actual_seq_lengths=[0,1,...]`）
  - gating：`npu_fused_gdn_gating(A_log, a, b, dt_bias.to(A_log.dtype))`
- 替换 `get_state_shape`：ssm state 由 0.13 的 `(Hv, Dk, Dv)` 换成 kernel 原生 `(Hv, Dv, Dk)`（从 `csrc/ascend/attention/recurrent_gated_delta_rule` tiling 源码确认：`state.DIM_2==Dv, DIM_3==Dk`）；prefill chunk 边界 transpose。
- `GemmaRMSNorm.forward_oot` → `npu_gemma_rms_norm`（无 residual）/ `npu_add_rms_norm_bias`（有 residual）。
- conv cache 布局：0.13 的 `MambaStateShapeCalculator` 本来就返回 `(state_len, dim)`（`conv_state_shape` 交换后的结果），直接传入，无需 transpose。
- rgdr spec 语义（从 kernel 源码确认）：初始状态从 `indices[seq0 + accepted - 1]` 读取，逐 token 状态写回 `indices[seq_i]`。

### 2.3 关键 bug：mamba cache 布局与算子 dense 假设冲突（端到端乱码根因）

0.13 的 `_reshape_kv_cache_tensors` 把 conv+ssm 两种 state 交织在一个（带 padding 的）page 里，state 视图首维 stride ≠ dense block 大小。AscendC 算子按 dense 假设寻址：

- conv op 输出正确但 **state 写回偏移错误**（验证：paged 布局下 state diff 5.7）
- rgdr 读写全错

**修复**：`patch_qwen3_6_gdn.py::_patch_mamba_cache_dense_layout()` 包装 `ModelRunnerFL._reshape_kv_cache_tensors`，把 mamba state 视图重组为 dense 逐 state 视图（同一 raw 存储内按 state 分组；Triton 路径用显式 strides，布局无关，不受影响）。

### 2.4 数值与端到端验证

- 四算子全部与 PyTorch/FLA 参考实现一致（含 conv/ssm state 写回、spec `num_accepted_tokens` 槽位语义）。
- 组合流水线（prefill→多步 decode）与 Triton 生产路径逐步一致。
- **A/B（贪心解码）AscendC vs Triton 基线输出逐 token 全等**。
- eager 自测 6 case 全过；aclgraph TPOT +0.4%~19%，TTFT -25%~45%。
- `VLLM_FL_DISABLE_ASCENDC_GDN=1` 强制回退 Triton。

---

## 3. CANN 自定义算子环境自动装配

### 3.1 问题

`npu_*` 算子依赖 `_cann_ops_custom/vendors/custom_transformer` 包。原守卫要求 `ASCEND_CUSTOM_OPP_PATH`/`LD_LIBRARY_PATH` 预置，未置则**静默回退 Triton**——用户图模式运行日志里正是 `keep Triton GDN path`，造成"接入无收益"的假象。

### 3.2 两个硬性约束（逐一验证）

1. `ASCEND_CUSTOM_OPP_PATH` 必须在 **aclInit 之前**设置（算子 infershape/注册扫描发生在 CANN 初始化时，晚了报 `Op has no infershape func`）。
2. aclnn 适配层用**裸名** `dlopen("libcust_opapi.so")` 解析符号，只查进程**启动时**的 `LD_LIBRARY_PATH`（glibc 缓存），Python 侧后设无效。

### 3.3 修复

`vllm_fl/__init__.py` 增加 `_bootstrap_cann_custom_op_env()`，在包导入时（早于 CANN 初始化）：

- 将打包的 vendor 目录前置进 `ASCEND_CUSTOM_OPP_PATH`；
- `ctypes.CDLL(<abs>/libcust_opapi.so, mode=RTLD_LOCAL)` 预载（SONAME 匹配后，后续裸名 dlopen 命中已加载句柄；**RTLD_GLOBAL 会在进程退出时 double-free，必须用 RTLD_LOCAL**）；
- patch 内保留幂等二次引导（`LD_LIBRARY_PATH` 缺失才触发 CDLL）。

`libcust_opapi.so` 的依赖（libnnopbase/libopapi_math 等）均在 CANN 标准库路径内，无需额外处理。

---

## 4. PTO megakernel `chunk_gated_delta_rule`（prefill 加速）

### 4.1 接入

参照 vllm-ascend PR #8872（`/workspace/patch/pto_gdn.patch`）。FL 已内置 `vllm_fl/dispatch/backends/vendor/ascend/impl/pto_chunk_gdn/`（6 阶段融合：cumsum → scaled_dot_kkt → solve_tril → wy_fast → chunk_h → chunk_o，单次 launch）。

适配（`chunk_gated_delta_wrapper.py`）：l2norm/run_mega_kernel 导入改 FL 路径；`_triton()` 回退调用去掉 FL chunk 不支持的 `prebuilt_meta`。`compile.py` 增加 `.so` 已存在则跳过（避免 4 个 TP worker 每次启动并发重编）。

回退语义：仅零 `initial_state` 的 prefill 走 PTO；decode（非零 initial state）、PCP>1、头维不匹配回退 Triton。

### 4.2 kernel 修复 1：头维 32B 对齐（支持 H=12）

Qwen3.6-27B TP4 → 每 rank `H=NV/TP=12, Hg=NK/TP=4`，而 `mega_transpose_TH_to_HT` 的 `pto::Tile` 静态形状要求 `H*sizeof(T) % 32 == 0`（fp32 需 8 的倍数，fp16 需 16 的倍数），H=12 直接编译失败。

**修复**（`csrc/ascend/pto_chunk_gdn/mega_kernel.cpp`）：tile 静态形状用补齐头维 `HA=ceil32(H*ES)/ES`，GM 侧 stride/valid 仍用 H，加载后 `TFILLPAD_INPLACE` 补零（仅有效行写回 GM）。注意：不能靠 wrapper 外层补头——GQA 比例 `H/Hg=3` 会被破坏。

### 4.3 kernel 修复 2：final_state UB 异步写竞争（多序列状态损坏根因）

现象：单序列 prefill 全对，**多序列时除末尾序列外所有序列的 `final_state` 全错**。定位：错误 work item = 每个 core 上除最后一个外的所有 item（3 序列时第 1 序列 12 头全错、第 2 序列前 4 头错、第 3 序列全对）。

根因：`chunk_h.cpp` 的 fs `TSTORE`（MTE3 异步）未排干，下一 work item 的清零/TCVT 覆写同一 UB，导致写回 GM 的 final state 被污染。

**修复**：fs TSTORE 后补 `set_flag/wait_flag(PIPE_MTE3, PIPE_V)` 排干。修复后 1/2/3/4 序列、变长（56~512）o 与 final_state 全部与 Triton chunk 一致。

### 4.4 验证

- 数值：PTO vs Triton chunk，输出 o 与 final_state 一致（fp16 计算、bf16 往返）。
- eager 自测：**TTFT +15%~95%**；A/B 输出逐 token 全等。
- aclgraph 自测：**TTFT +18%~84%**，6 case 全过。
- kernel `.so` 按 (H,Hg,D,C) 缓存于 `vllm_fl/dispatch/backends/vendor/ascend/impl/pto_chunk_gdn/kernels/compiled_lib/`，新配置首用编译 ~9s。

---

## 5. batch64 性能回退分析与修复

### 5.1 现象与数据

batch64（1024~16384 输入 × 1024 输出 × 64 并发）下 PTO 比无 PTO eager 慢 5~10%（309 vs 346 tok/s @1024）。

### 5.2 根因（微基准定量）

| T=131072 tokens | Triton chunk | PTO wrapper | PTO kernel |
|---|---|---|---|
| 单层耗时 | 46.8~47.8ms | 26.9~27.8ms | **~1.7x 更快** |

kernel 本身更快。回退来自 wrapper 在**每个 GDN 层每个 prefill 步**引入的 2 次 device→host 同步：

1. `torch.any(initial_state != 0)`（回退判断）
2. `_total_chunks` 的 `cu_seqlens.cpu().tolist()`（chunk 计数）

48 层 × 2 = 96 次同步/步 ≈ 5~10% 开销。且稳态 batch（decode 行混入）下 initial_state 非零 → 必然回退 Triton，白付同步。

### 5.3 修复（免同步决策）

- `patch_qwen3_6_gdn.py::_patch_gdn_metadata_host_flags()`：包装 `GDNAttentionMetadataBuilder.build`，在 CPU 侧给 metadata 附加：
  - `any_initial_state_cpu`（由 `num_computed_tokens_cpu > 0` 计算，**零同步**）
  - `cu_seqlens_host`（由 `query_start_loc_cpu` 直接转换）
- `_forward_core` prefill 分支：`any_initial_state_cpu is False`（全新 prefill 批）→ 直接调 `_chunk_gdn_pto`（内置 l2norm+fp16 cast+`total_chunks` 透传），否则走 Triton chunk；**不再经过任何带同步的 wrapper**。
- `mega_kernel.py` 增加 `total_chunks` 可选参数，跳过 `.cpu()` 同步。
- 移除旧 `patch_pto_chunk_gdn` wrapper 注册。

### 5.4 验证

batch64 1024 case：**348.65 tok/s ≥ 无 PTO 基线 346.24**，回退消除（此前 wrapper 版 309.17）。

---

## 6. Slice 热点修复（KV 缓存连续布局）

### 6.1 现象

profile 算子统计中 `Slice` 类 kernel 合计 1.1ms，占设备时间 62%。

### 6.2 定位（排除法）

- 微基准：各类拷贝在真实 shape 下仅 20~38us —— decoder 层切片写不是热点。
- `kernel_details.csv` 呈**双峰**：848 次 × 2.7us（正常）+ **1024 次 × ~1068us**。
- 时间线：每个 full-attention 层每步恰好 2 次慢 Slice（16 层 × 32 步 × 2 = 1024 ✓），位置在 `split_qkv_rmsnorm_mrope` 之后、`ReshapeAndCache/FIAS` 之前。
- 代码定位：`impl/attention.py::forward()` 中 `kv_cache = [i.contiguous() for i in kv_cache]`。

### 6.3 根因（两个叠加问题）

1. `_update_hybrid_attention_mamba_layout`（混合模型缓存布局调整）把 `(2, num_blocks, ...)` 连续 KV 缓存重排为 k/v 逐块交织的 strided 视图 → `kv_cache[0]/[1]` 非连续。
2. 比较 bug：`if attn_metadata != AscendAttentionState.DecodeOnly:` 拿元数据**对象**与枚举比较，**恒为 True** → 每次调用都对整份 k/v cache 各做一次 `.contiguous()`（~340MB/份，实测 ~870us，profile 中 ~1ms）。16 层 ≈ 32ms/步（约占 TPOT 16%）。

### 6.4 修复

- `vllm_fl/worker/model_runner.py::_update_hybrid_attention_mamba_layout`：跳过 re-stride，缓存保持连续 `(2, num_blocks, ...)` 布局；所有访问经同一组视图，语义透明，`.contiguous()` 变 no-op（870us → 2us）。
- `impl/attention.py`：改为 `attn_metadata.attn_state != AscendAttentionState.DecodeOnly`。

### 6.5 验证（i32_o32 profile）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| Slice 总耗时 | 1,096,041us（1872 次，均值 585us） | **2,259us**（848 次，均值 2.7us） |
| 慢 Slice (>800us) | 1024 次 | **0** |
| 设备总 kernel 时间 | 1,769,371us | 603,710us（**-66%**） |
| FusedInferAttentionScore | 432us | 394us |
| MatMulV2 | 475,920us | 409,561us（-14%） |

正确性：贪心解码 2/3 prompt 与基线逐 token 全等，1/3 仅末尾 token 级分叉。

注：eager 单并发 TPOT 受 host 侧 launch 开销主导，设备侧收益在 aclgraph 图模式/大并发下体现更直接。

---

## 7. 性能数据汇总

### 7.1 eager（TP4, c1）

| case | TTFT 基线 | TTFT +AscendC | TTFT +PTO | TPOT 基线 | TPOT +AscendC |
|---|---|---|---|---|---|
| i32~i512 | 493~597ms | 411~589ms | **255~390ms** | ~153~190ms（机器噪声区间） | ~152~159ms |

A/B 正确性：AscendC vs Triton 输出逐 token 全等；PTO vs 无 PTO 输出逐 token 全等。

### 7.2 aclgraph FULL（TP4, c1）

| case | TTFT Triton | TTFT AscendC | TTFT +PTO | TPOT Triton | TPOT AscendC |
|---|---|---|---|---|---|
| i32~i512 | 682~789ms | 417~515ms | **243~436ms** | 73~79ms | 61~77ms（+0.4%~19%） |

### 7.3 batch64（64 并发，eager）

| case | 无 PTO | PTO(wrapper,回退) | PTO(免同步修复) |
|---|---|---|---|
| 1024_1024_c64 | 346.24 tok/s | 309.17 tok/s | **348.65 tok/s** |

---

## 8. 环境问题与运行注意事项

1. **nnal/atb 版本**：系统侧 `nnal/atb/latest` 已切到 9.0.0，与 torch_npu 不兼容（`libtbe_adapter.so` 符号未定义）。运行前置：
   `export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_1/lib:$LD_LIBRARY_PATH`
2. **fl 插件加载**：`import vllm_fl` 只有以 `/workspace/vllm-plugin-FL` 为 cwd 启动时才解析到仓库包（site-packages 下 namespace 目录遮蔽 editable 安装）。自测脚本须从该目录启动。
3. **共享机器噪声**：8 卡被其他租户频繁抢占，TPOT 类指标 ±15% 漂移常见；结论以算子级 profile 与中位步长为准。
4. 环境开关：
   - `VLLM_FL_DISABLE_ASCENDC_GDN=1`：回退全部 AscendC GDN 改动
   - `VLLM_FL_DISABLE_PTO_GDN=1`：仅关 PTO megakernel

## 9. 后续方向

- MTP(spec decode) + aclgraph 联调（spec 路径已移植，未覆盖验证；rgdr `MAX_MTP` 约束单序列 token 数）。
- eager host 侧开销治理（kernel 数量/调度），或优先 aclgraph 图模式。
- 慢 Slice 剩余 848 次（均值 2.7us，含 decoder 层切片写与 `cos_sin_cache[positions]` gather）：decoder 层改函数式直返、`cos_sin` gather 每步跨层共享（16 次 → 1 次）。
- PTO megakernel bf16 原生支持（当前 fp16 往返 cast）。

## 附：变更文件清单

| 文件 | 内容 |
|---|---|
| `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py` | 新增：AscendC GDN `_forward_core`/`get_state_shape`、GemmaRMSNorm、mamba dense 布局、PTO 免同步接入、环境 bootstrap |
| `vllm_fl/dispatch/backends/vendor/ascend/patch.py` | 注册 `patch_qwen3_6_gdn` |
| `vllm_fl/dispatch/backends/vendor/ascend/patches/README.md` | patch 条目 |
| `vllm_fl/__init__.py` | `_bootstrap_cann_custom_op_env`（OPP 路径 + RTLD_LOCAL 预载） |
| `vllm_fl/dispatch/backends/vendor/ascend/impl/pto_chunk_gdn/chunk_gated_delta_wrapper.py` | FL 导入适配、`_triton` 参数适配 |
| `vllm_fl/dispatch/backends/vendor/ascend/impl/pto_chunk_gdn/compile.py` | `.so` 存在则跳过编译 |
| `vllm_fl/dispatch/backends/vendor/ascend/impl/pto_chunk_gdn/mega_kernel.py` | `total_chunks` 可选参数 |
| `csrc/ascend/pto_chunk_gdn/mega_kernel.cpp` | 头维 32B 补齐 |
| `csrc/ascend/pto_chunk_gdn/chunk_h.cpp` | fs UB 写竞争排干 |
| `vllm_fl/worker/model_runner.py` | `_update_hybrid_attention_mamba_layout` 跳过 re-stride |
| `vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py` | `attn_state` 比较修正 |
