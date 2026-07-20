# vllm-plugin-FL 昇腾 ACL Graph 模式设计说明

> 分支：`add-qwen3_6_ascend_split_qkv_rmsnorm_mrope`  
> 相关 commit：`ab0798f`、`163ddfc`  
> 相关 patch：`qwen3_6_ascend_graph_moe.patch`

---

## 1. 背景与目标

vLLM v1 默认在 graph 模式下使用 `torch.compile(..., fullgraph=True)` + `CUDAGraph`/`NPUGraph` 来提升 decode 阶段吞吐。`vllm-plugin-FL` 原本只关注算子分发（FlagGems / vendor / reference），其 `OpManager` 内部使用 Python `RLock` 做延迟初始化，这会被 Dynamo 直接报错：

```text
torch._dynamo.exc.Unsupported: Unsupported context manager (RLock)
```

因此本阶段目标：

1. 在 **不破坏 FL 原有 eager 分发能力** 的前提下，让昇腾 NPU graph 模式服务化可运行。
2. 将 `vllm-ascend` 中的 ACL graph 能力以 **插件化、非侵入** 方式引入 `vllm-plugin-FL`。
3. 补齐 `npugraph_ex / torchair` 编译通路所需的 pass manager 与 patch。
4. 对 Qwen3.5/3.6 系列模型提供 attention / MLA graph-param 升级支持。

---

## 2. 总体架构

```text
vllm-plugin-FL
├── vllm_fl/compilation/graph.py              # 通用 GraphWrapper + 后端 mixin 注册表
├── vllm_fl/dispatch/backends/vendor/ascend/
│   ├── compilation/
│   │   ├── compiler_interface.py             # AscendCompiler (npugraph_ex 入口)
│   │   ├── graph_fusion_pass_manager.py      # 昇腾专属图融合 pass manager
│   │   └── passes/                           # 具体融合 pass
│   ├── impl/
│   │   ├── attention.py                      # AscendAttentionBackend + graph params
│   │   ├── activation.py                     # silu_and_mul_ascend 等
│   │   ├── normalization.py                  # rms_norm_ascend
│   │   └── rotary.py                         # rotary_embedding_ascend
│   └── patches/
│       ├── patch_graph.py                    # ACLGraphBackendMixin
│       └── patch_npugraph_ex.py              # npugraph_ex ValuePack patch
├── vllm_fl/ops/                              # FL OOT 算子层
├── vllm_fl/platform.py                       # NPU 平台配置
└── vllm_fl/dispatch/backends/vendor/ascend/patch.py
    └── patch_dynamo_safe_ops()               # Dynamo 安全化 OOT 算子
```

设计原则：

- **通用层保持硬件无关**：`GraphWrapper` 只暴露 hook 注册表，硬件细节下沉到 mixin。
- **按需启用**：默认 NPU 走 `backend=eager` + `NPUGraph`；只有显式开启 `ascend_compilation_config.enable_npugraph_ex` 时才启用 `AscendCompiler`。
- **Dynamo 兼容与分发兼容并存**：graph 模式下绕过 `OpManager/RLock`，eager 模式仍走完整 dispatch。

---

## 3. 通用 GraphWrapper 后端钩子机制

文件：`vllm_fl/compilation/graph.py`

### 3.1 注册表

```python
_graph_wrapper_backend_registry: dict[str, type] = {}

def register_graph_wrapper_backend(device_type: str, backend_cls: type) -> None
```

按 `device_type`（`npu`/`cuda`/...）注册一个 mixin 类。`GraphWrapper.__init__` 会根据当前平台自动实例化对应的 mixin。

### 3.2 Mixin 可实现的 hook

| Hook | 触发时机 | 用途 |
|------|----------|------|
| `before_capture(entry, args, kwargs)` | 进入 `torch.npu.graph` 捕获前 | stream 同步、offloader 同步 |
| `wrap_capture_context(entry, stack)` | `ExitStack` 中 | 禁用 `gc.collect` / `torch.npu.empty_cache` 等 |
| `after_capture(entry, output, args, kwargs)` | 捕获退出后 | weak-ref workspace、offloader join |
| `before_replay(entry, args, kwargs)` | replay 前 | host-device 同步 |
| `capture_error_handler(exc)` | 捕获异常时 | 翻译 CANN 错误码 |
| `weak_ref_tensors(tensor)` | 需要弱引用输出/ workspace 时 | 硬件特定实现 |

### 3.3 Graph 类抽象

```python
class Graph:
    if current_platform.device_type == "cuda":
        graph = torch.cuda.CUDAGraph
    elif current_platform.device_type == "npu":
        graph = torch.npu.NPUGraph
    elif current_platform.device_type == "musa":
        graph = torch.musa.MUSAGraph
```

这样 `GraphWrapper` 对 CUDA/NPU/MUSA 使用同一套捕获/回放逻辑。

---

## 4. 昇腾 ACLGraph 后端 Mixin

文件：`vllm_fl/dispatch/backends/vendor/ascend/patches/patch_graph.py`

### 4.1 `ACLGraphBackendMixin`

该类是 `vllm-ascend` 中 `ACLGraphWrapper` 的插件化重构版本，通过 hook 注入昇腾特有的行为：

- **捕获前 (`before_capture`)**
  - 调用 `_sync_offloader_before_capture()`，确保 host 侧 offload 数据已就位。
- **捕获上下文 (`wrap_capture_context`)**
  - 当 `graph_options.gc_disable=True` 时，把 `torch.npu.empty_cache` patch 掉，避免层间反复 GC。
- **捕获后 (`after_capture`)**
  - `weak_ref_workspaces()`：把 attention / MLA / causal_conv1d 的 workspace 转为弱引用，降低显存峰值。
  - `update_full_graph_params()` / `update_draft_graph_params()`：把当前 batch size 对应的 graph params 同步给 attention backend。
- **回放前 (`before_replay`)**
  - 在 `FULL` 模式下（非 EAGLE draft 路径、非 `enable_enpu` 模式）调用 `torch.npu.current_stream().synchronize()`，保证 host 侧 attention 参数更新与 graph 执行顺序。
- **异常处理 (`capture_error_handler`)**
  - 识别 CANN 错误码 `207008` / `insufficient_stream_resources`，给出可读的错误提示。

### 4.2 Graph Params 管理

```python
@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[Any]]
    attn_params: dict[int, list[tuple]]
    conv1d_params: ...
```

- `set_graph_params(capture_sizes)` 在模型加载/ warmup 时按 capture size 初始化空桶。
- `update_graph_params_workspaces(num_tokens, workspace)` 在每次 capture/replay 前把 backend 需要的 workspace 写入。
- `weak_ref_workspaces()` 在 capture 结束后把 workspace 转弱引用，复用同一块显存。

### 4.3 注册

在 `patch_graph()` 中：

```python
if current_platform.device_type != "npu":
    return
register_graph_wrapper_backend("npu", ACLGraphBackendMixin)
```

---

## 5. Attention / MLA Graph 参数升级

文件：`vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py`

### 5.1 `AscendAttentionBackendImpl`

- 使用 `torch_npu.npu_fused_infer_attention_score` 做 prefill / chunked-prefill。
- 使用 `torch_npu._npu_paged_attention` 做 decode。
- 使用 `torch_npu._npu_reshape_and_cache` 更新 KV cache。

### 5.2 `update_graph_params`

```python
@classmethod
def update_graph_params(cls, update_stream, forward_context, num_tokens, vllm_config, ...):
    params = get_graph_params()
    if params is not None and num_tokens in params.workspaces:
        if params.workspaces[num_tokens] is None:
            params.workspaces[num_tokens] = True  # marker
```

在 native torch_npu 路径下，workspace 是 kernel 内部惰性分配的，因此这里只记录一个占位 marker；对需要显式 workspace 的 kernel，可后续扩展为真实 tensor。

### 5.3 `AscendMLABackendImpl`

- 提供 MLA 的占位 backend 与 `update_graph_params`。
- 当前 `forward` 尚未接入原生 Ascend MLA kernel，但 graph params 的接口已经预留。

---

## 6. npugraph_ex 编译框架

### 6.1 `AscendCompiler`

文件：`vllm_fl/dispatch/backends/vendor/ascend/compilation/compiler_interface.py`

```python
class AscendCompiler(CompilerInterface):
    def __init__(self):
        try:
            import npugraph_ex as nge
        except ImportError:
            import torchair as nge
```

- 当 `ascend_compilation_config.enable_npugraph_ex=True` 时，vLLM 的 `VllmBackend` 会使用该 CompilerInterface。
- 当前 `compile()` / `load()` 尚未完整实现，仅完成接口注册与 hash/cache 初始化。

### 6.2 `GraphFusionPassManager`

文件：`vllm_fl/dispatch/backends/vendor/ascend/compilation/graph_fusion_pass_manager.py`

对应 vLLM 的 `PostGradPassManager`，但针对昇腾实现：

| Pass | 功能 |
|------|------|
| `AddRMSNormQuantFusionPass` | RMSNorm + 量化融合 |
| `QKNormRopeFusionPass` | QK norm + RoPE 融合 |
| `MatmulAllReduceAddRMSNormPass` | AllReduce + RMSNorm 融合 |
| `MulsAddFusionPass` | mul + add 常量折叠 |
| `SequenceParallelismPass` / `SequenceParallelismMoePass` | 序列并行相关优化 |
| `NoOpEliminationPass` | 无用节点消除 |

配置项通过 `config.additional_config["ascend_compilation_config"]` 控制开关。

### 6.3 `patch_npugraph_ex` ValuePack 补丁

文件：`vllm_fl/dispatch/backends/vendor/ascend/patches/patch_npugraph_ex.py`

`npugraph_ex`/`torchair` 在把 Triton kernel 接入 FX graph 时，输入可能被打包成 `ValuePack`（同时携带 meta tensor 与 npu tensor）。原实现 unpack 逻辑在 list/tuple 场景下会丢失 npu 部分，导致 graph 构建失败。

本补丁：

1. 重新定义 `ValuePack` 类，同时保存 `meta` 与 `npu`。
2. 替换 `npu_fx_compiler._unpack_meta` 与 `_NpuGraphConverter._unpack_npu`，确保 list/dict/单个 `ValuePack` 都能正确拆包。
3. 对 `torchair` 额外 reload `torchair.fx_summary`。

---

## 7. 平台配置

文件：`vllm_fl/platform.py`

NPU 分支在 `check_and_update_config` 中完成：

```python
if cls.device_type == "npu":
    ascend_compilation_config = (
        vllm_config.additional_config or {}
    ).get("ascend_compilation_config", {})
    enable_npugraph_ex = ascend_compilation_config.get("enable_npugraph_ex", False)

    # 非 CUDA 平台没有这些 pass，禁用避免 NameError
    compilation_config.pass_config.fuse_norm_quant = False
    compilation_config.pass_config.fuse_act_quant = False
    compilation_config.pass_config.fuse_attn_quant = False
    compilation_config.pass_config.fuse_allreduce_rms = False

    # 默认 backend 切到 eager
    if backend in ("", "inductor") and not enable_npugraph_ex:
        compilation_config.backend = "eager"
```

- 默认只使用 `NPUGraph`（即 `CUDAGraphMode.FULL`），不启用 inductor。
- 用户可通过 `ascend_compilation_config.enable_npugraph_ex=true` 开启 `AscendCompiler` 路径。

---

## 8. Dynamo 安全化 OOT 算子

文件：`vllm_fl/dispatch/backends/vendor/ascend/patch.py` 中的 `patch_dynamo_safe_ops()`

### 8.1 问题

`vllm-plugin-FL` 的 OOT 算子（`RMSNormFL`、`SiluAndMulFL`、`GeluAndMulFL`、`RotaryEmbeddingFL`）在 `forward_oot` 中调用 `call_op("...", self, ...)`，最终进入 `OpManager.ensure_initialized()` 的 `RLock`。在 `torch.compile(fullgraph=True)` 下直接报错。

### 8.2 方案

在 NPU 平台上，把这些 `forward_oot` 替换为 **直接调用 Ascend / reference 实现**：

| 算子 | graph 模式直接调用 | eager 模式仍走 |
|------|--------------------|----------------|
| `SiluAndMulFL` | `silu_and_mul_ascend` (torch_npu.npu_swiglu) | `call_op` |
| `GeluAndMulFL` | `gelu_and_mul_torch` (F.gelu * mul) | `call_op` |
| `RMSNormFL` | `rms_norm_ascend` (npu_rms_norm / npu_add_rms_norm) | `call_op` |
| `RotaryEmbeddingFL` | `rotary_embedding_ascend` (torch_npu._npu_rotary_embedding) | `call_op` |

注意点：

- `RotaryEmbeddingFL` 原实现会把 `self.cos_sin_cache = self.cos_sin_cache.to(positions.device)` 写回 buffer，这在 `torch.compile + cudagraph` 中非法。补丁版改为局部变量 `cos_sin_cache = self.cos_sin_cache.to(positions.device)`。
- 该补丁只在 NPU 平台生效，不影响 CUDA 或其他后端。
- 该补丁不影响 eager 路径：eager 下 vLLM 也会选择 `forward_oot`，但此时已被 patch 成直接实现，性能反而更优。

---

## 8.5 Fused MoE 图安全重写（Qwen3.6-35B-A3B）

文件：`vllm_fl/dispatch/backends/vendor/ascend/impl/fused_moe.py`

### 8.5.1 问题现象

Qwen3.6-35B-A3B 为 MoE 模型，在 `ACLGraph` capture 阶段 fused MoE 内部出现类似错误：

```text
torch_npu.npu.synchronize / npu graph capture 期间 stream synchronize 报错
```

根因是原 fused MoE 实现中存在 **`mask.any()` / `nonzero()` / `index_select` 等 CPU 同步或动态形状操作**，这些操作在 `torch.npu.NPUGraph` capture 中不被允许，会强制 synchronize 并破坏 graph。

### 8.5.2 方案

改用昇腾原生图安全算子实现 `fused_experts`：

```python
def _torch_fused_experts_impl(
    hidden_states, w1, w2, topk_weights, topk_ids,
    inplace=False, activation="silu",
    apply_router_weight_on_input=False,
    global_num_experts=-1, expert_map=None,
):
    ...
    expanded_x, row_idx, expert_token_count, _ = torch_npu.npu_moe_init_routing_v2(
        hidden_states, local_topk_ids.to(torch.int32),
        active_num=num_tokens * top_k, expert_num=global_num_experts,
        expert_tokens_num_type=1, expert_tokens_num_flag=True,
        quant_mode=-1, active_expert_range=[0, E], row_idx_type=1,
    )
    ...
    gate_up = torch_npu.npu_grouped_matmul(
        [expanded_x], [w1_t],
        group_list=expert_token_count, split_item=2,
        group_type=0, group_list_type=1,
    )[0]
    ...
    down = torch_npu.npu_grouped_matmul(
        [gate_up], [w2_t],
        group_list=expert_token_count, split_item=2,
        group_type=0, group_list_type=1,
    )[0]
    out = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=down, sorted_indices=sorted_indices, probs=probs,
    )
    return out
```

关键变化：

- 使用 `npu_moe_init_routing_v2` 替代 permute / sort / 动态索引。
- 使用 `npu_grouped_matmul` 完成 expert-wise MLP 计算，避免 `for` 循环专家逐个 matmul。
- 使用 `npu_moe_token_unpermute` 完成 token re-order 与 router weight 乘加。
- `activation="silu"` 通过 split + `npu_swiglu` 完成；当前版本先支持 unquantized MoE。
- 所有 tensor 操作均在 NPU stream 上完成，**无 host sync、无数据依赖 CPU**。

### 8.5.3 AscendC 自定义算子替换（当前版本）

在 8.5.2 的基础上，MoE 链路进一步替换为 `csrc/ascend/moe` 的 AscendC 自定义算子
（`patch_fused_moe()` 在 `ascendc_moe_available()` 为真时启用，`VLLM_FL_DISABLE_ASCENDC_MOE=1` 可回退）：

- **路由**：`fused_topk` 替换为 `torch.ops._C_ascend.moe_gating_top_k`（`fused_topk_ascend`），
  softmax + top-k + L1 renorm 单 kernel 完成，取代 FlagGems Triton `topk_softmax`。
  输入先转 fp32 以保持与 FlagGems 路径一致的精度（topk_weights 仍为 fp32）。
- **token 扩展/排序**：`npu_moe_init_routing_v2` 替换为 `torch.ops._C_ascend.npu_moe_init_routing_custom`
  （`_ascendc_fused_experts_impl`，count 模式）。`row_idx_type=0` 返回的 `expanded_row_idx`
  取 `abs()` 后即为 `npu_moe_token_unpermute` 需要的 gather 索引，省去旧路径的 scatter 求逆。
- **权重预转置**：`UnquantizedFusedMoEMethod.process_weights_after_loading` 被包装，
  `w13_weight`/`w2_weight` 在加载后一次性转置为 `[E, hidden, 2I]` / `[E, I, hidden]`，
  消除旧路径每次前向两次 `transpose(1, 2).contiguous()` 大拷贝（35B TP=4 每步每层约 2×128MB）。
  `fused_experts_impl` 按权重形状（`hidden == w1.size(1)`）在运行时选择 AscendC 路径或旧路径。
- **grouped matmul 保留 `torch_npu.npu_grouped_matmul`**：自定义 `moe_grouped_matmul`
  （NZ 权重版）kernel 在 ascend910b 上对平凡单专家输入也会触发 aicore 崩溃
  （CCU instruction address check error），暂不可用。

### 8.5.4 限制

- 当前只覆盖 **unquantized + silu** 的 fused MoE；量化 MoE（FP8/INT8）需要在 `quant_mode` 与 `npu_grouped_matmul` scale 参数上进一步扩展。
- `global_num_experts` 必须正确传递；若使用 expert parallelism 的 `expert_map`，需要在外部做 expert id 映射后再传入。

---

## 8.6 Multi-Token Prediction（MTP）昇腾适配

文件：`vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_mtp.py`

### 8.6.1 问题背景

Qwen3.5/Qwen3.6 的 MTP 在昇腾 NPU 上初始化时会遇到两类错误：

1. **`bind_kv_cache` 触发 `NotImplementedError`**：MTP drafter 模型与目标模型都包含同层索引的 attention 模块，upstream `vllm.v1.worker.utils.bind_kv_cache` 在非 CUDA/XPU/CPU 平台下直接抛错。
2. **MTP drafter 错误消费 PP intermediate tensors**：upstream `Qwen3NextMultiTokenPredictor.forward` 只在第一个 PP rank 上拼接 token embedding 与 target hidden states，而在昇腾部署中 drafter 与目标模型通常位于最后一个 PP rank，需要从本层直接拿到 target hidden states 和 input_ids。
3. **`qwen3_5` / `qwen3_5_moe` 不会自动识别为 MTP**：upstream `SpeculativeConfig.hf_config_override` 只把 `qwen3_next` 映射为 `qwen3_next_mtp`，Qwen3.5/Qwen3.6 的 `qwen3_5` / `qwen3_5_moe` 需要同样改写。

### 8.6.2 方案

- **重写 `bind_kv_cache`**：移除平台 guard，当同一 layer index 对应多个 attention 模块时，仅把第一个加入 `runner_kv_caches`，并把所有模块的 KV cache 绑定到 `forward_context`。
- **重写 `Qwen3NextMultiTokenPredictor.forward`**：在最后一个 PP rank 上始终使用本地 `embed_input_ids` 与传入的 `hidden_states` 拼接，忽略 `intermediate_tensors`；非最后一个 rank 仍返回 `IntermediateTensors`。
- **扩展 `SpeculativeConfig.hf_config_override`**：把 `qwen3_5` / `qwen3_5_moe` 的 draft config 映射为 `qwen3_next_mtp`，从而复用 upstream `Qwen3NextMTP` / `Qwen3NextMultiTokenPredictor`。

### 8.6.3 使用方式

```bash
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-27B \
  --model-name qwen3.6 \
  --mode graph \
  --mtp 1 \
  --cases "128,128,1" \
  --run-label mtp1 \
  --package none
```

等价的显式 `--speculative-config`：

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1,"model":"/models/Qwen3.6-27B"}'
```

> 注意：当前环境中没有带 `num_nextn_predict_layers` 的本地 checkpoint，因此功能验证以单元测试和启动参数检查为主；完整的端到端 MTP 性能验证需要准备对应的模型。

---

## 9. 验证结果

### 9.1 NPU graph 冒烟测试

脚本：`/tmp/verify_graph.py`

验证项：

- `ACLGraphBackendMixin` 已注册到 `device_type=npu`。
- `AscendCompiler` / `GraphFusionPassManager` 可导入。
- `GraphWrapper` 可成功 capture / replay一个 `torch.npu.NPUGraph`。
- `before_replay` hook 可正常执行。

### 9.2 Qwen3.6-27B 服务化压测

命令（与用户提供的命令一致）：

```bash
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-27B \
  --model-name qwen3.6 \
  --model-tag qwen3.6-27b \
  --mode graph \
  --cudagraph-mode FULL \
  --chunked true \
  --bench-profile false \
  --cases "32,32,1;32,128,1;128,32,1;128,128,1;512,32,1;512,128,1" \
  --run-label i32-128-512_o32-128_c_f_noprof \
  --package none
```

全部 6 个 case 成功跑完：

- `i32_o32`, `i32_o128`, `i128_o32`, `i128_o128`, `i512_o32`, `i512_o128`
- 结果目录：`/workspace/results/atp_qwen3.6-27b_fl_graph_chunked_i32-128-512_o32-128_c_f_noprof_tp4_gmem0.6_20260625_113610/`

### 9.3 Qwen3.6-35B-A3B（MoE）graph 服务化验证

命令：

```bash
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-35B-A3B \
  --model-name qwen3.6 \
  --model-tag qwen3.6-35b-a3b \
  --mode graph \
  --cudagraph-mode FULL_DECODE_ONLY \
  --chunked true \
  --bench-profile false \
  --devices 4,5,6,7 \
  --cases "32,32,1" \
  --run-label i32_o32_c_fdo_noprof \
  --package none
```

验证结果：

- 在 fused MoE 图安全重写后，`ACLGraph` capture / replay 成功完成，未再触发 stream synchronize 错误。
- 单 case `i32_o32` 已跑通，结果目录：`/workspace/results/atp_qwen3.6-35b-a3b_fl_graph_chunked_i32_o32_c_fdo_noprof_tp4_gmem0.6_20260629_152622/`
- 继续扩展更多 `input_len/output_len` case 前，需先清理残留 `VLLM::Worker_TP*` 进程并确认 NPU 显存充足。
- 在接入 MTP patch 后，使用相同命令重新验证 Qwen3.6-35B-A3B（非 MTP 模型）未出现回归，结果目录：`/workspace/results/atp_qwen3.6-35b-a3b_fl_graph_chunked_i32_o32_c_fdo_noprof_mtpreg2_tp4_gmem0.6_20260630_152852/`

---

## 10. 关键文件清单

| 文件 | 说明 |
|------|------|
| `vllm_fl/compilation/graph.py` | 通用 GraphWrapper + backend mixin 注册表 |
| `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_graph.py` | `ACLGraphBackendMixin`、GraphParams、stream 错误诊断 |
| `vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py` | Ascend attention backend + `update_graph_params` |
| `vllm_fl/dispatch/backends/vendor/ascend/compilation/compiler_interface.py` | `AscendCompiler` |
| `vllm_fl/dispatch/backends/vendor/ascend/compilation/graph_fusion_pass_manager.py` | 昇腾图融合 pass manager |
| `vllm_fl/dispatch/backends/vendor/ascend/compilation/passes/*.py` | 各融合 pass |
| `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_npugraph_ex.py` | `npugraph_ex`/`torchair` ValuePack patch |
| `vllm_fl/dispatch/backends/vendor/ascend/patch.py` | `patch_dynamo_safe_ops()` 等 Ascend patch 集合 |
| `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_mtp.py` | 昇腾 MTP 适配（bind_kv_cache / MTP forward / speculative config） |
| `vllm_fl/dispatch/backends/vendor/ascend/impl/fused_moe.py` | 昇腾图安全 fused MoE 实现 |
| `vllm_fl/platform.py` | NPU 平台配置（backend、eager、pass 开关） |
| `scripts/run_vllm_fl_profile_unified.sh` | 服务化压测脚本（新增 `--mtp` 快捷参数） |
| `qwen3_6_ascend_graph_moe.patch` | 完整补丁文件（含 MoE + MTP 修复） |

---

## 11. 使用方式

### 11.1 默认 graph 模式（已验证）

```bash
cd /workspace/vllm-plugin-FL
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_PLUGINS=fl
export VLLM_ENABLE_GRAPH_MODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 当前 FlagGems 部分算子在 vLLM v1 graph 模式下存在 bug，建议关闭
export USE_FLAGGEMS=0

# 若之前运行异常退出，先清理残留 worker 进程
pkill -f "VLLM::Worker_TP"

python -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3.6-27B \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --trust-remote-code
```

### 11.2 开启 npugraph_ex（实验性）

```bash
# 通过 additional_config 传入
--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true}}'
```

当前 `AscendCompiler.compile()` 尚未完整实现，仅完成接口与 cache 框架。

### 11.3 开启 MTP（Multi-Token Prediction）

使用压测脚本快捷参数：

```bash
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-27B \
  --mode graph \
  --mtp 1 \
  --cases "128,128,1" \
  --run-label mtp1 \
  --package none
```

或直接通过 vLLM CLI：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3.6-27B \
  --tensor-parallel-size 4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1,"model":"/models/Qwen3.6-27B"}' \
  --enable-chunked-prefill \
  --trust-remote-code
```

---

## 12. 后续可扩展点

1. **MLA kernel 接入**：`AscendMLABackendImpl.forward` 目前抛 `NotImplementedError`，需要接入原生 Ascend MLA 算子。
2. **`AscendCompiler.compile()` 完整实现**：真正把 FX graph 交给 `npugraph_ex` / `torchair` 编译并缓存。
3. **量化 MoE 支持**：当前 fused MoE 仅支持 unquantized + silu，FP8/INT8 MoE 需要在 `npu_moe_init_routing_v2` / `npu_grouped_matmul` 中传入 quant_mode 与 scale。
4. **更多 OOT 算子 Dynamo 化**：当前只 patch 了最常见的 4 个 OOT 算子；若后续出现新的 `call_op` 在 graph 路径下被 trace，可同样替换为直接实现。
5. **Dynamo-safe dispatch manager**：长期看可以重构 `OpManager`，使其不使用 `RLock` 或使用 `torch._dynamo.allow_in_graph` 注册为不透明节点，从而不需要 per-op patch。
