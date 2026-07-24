# npu_chunk_gated_delta_rule Integration Guide

## 概述

本次集成将 vllm-ascend PR #12607 中的 `npu_chunk_gated_delta_rule` 算子引入到 vllm-plugin-FL 中，用于 Qwen3.6 GDN (Gated Delta Net) 模型的 prefill 阶段。

## 算子对比

### 当前实现（默认）
| 阶段 | 算子 | 说明 |
|------|------|------|
| Prefill (fresh) | PTO megakernel | 仅支持 all-zero initial_state |
| Prefill (non-fresh) | Triton chunk_gated_delta_rule | 支持带 initial_state 的续写 |
| Speculative Decode | npu_recurrent_gated_delta_rule | 已集成的 aclnn 算子 |
| Decode | npu_recurrent_gated_delta_rule | 已集成的 aclnn 算子 |

### 新实现（VLLM_FL_USE_ACLNN_CHUNK_GDN=1）
| 阶段 | 算子 | 说明 |
|------|------|------|
| **Prefill (all)** | **npu_chunk_gated_delta_rule** | **统一处理 fresh 和 non-fresh** |
| Speculative Decode | npu_recurrent_gated_delta_rule | 保持不变 |
| Decode | npu_recurrent_gated_delta_rule | 保持不变 |

## 环境变量控制

### VLLM_FL_USE_ACLNN_CHUNK_GDN

```bash
# Level 0 (默认): 使用 PTO + Triton（当前实现）
export VLLM_FL_USE_ACLNN_CHUNK_GDN=0

# Level 1: 使用 aclnn 替换所有 prefill（新算子）
export VLLM_FL_USE_ACLNN_CHUNK_GDN=1
```

## 已修改的文件

### 1. C++ 算子层
- ✅ `csrc/ascend/attention/chunk_gated_delta_rule/` - 27 个新文件
- ✅ `csrc/ascend/build_aclnn.sh` - 添加到编译列表
- ✅ `csrc/ascend/torch_binding.cpp` - 注册算子
- ✅ `csrc/ascend/torch_binding_meta.cpp` - 添加 meta 实现

### 2. Python 集成层
- ✅ `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py`
  - 添加 `npu_chunk_gated_delta_rule` 到 `_REQUIRED_OPS`
  - 实现 `_chunk_gated_delta_rule_aclnn()` 函数
  - 修改 `_forward_core()` 添加环境变量控制

### 3. 测试文件
- ✅ `tests/custom_ops_tests/test_chunk_gated_delta_rule.py` - 单元测试
- ✅ `test_aclnn_chunk_gdn.sh` - 性能对比测试脚本

## 使用方法

### 1. 编译安装算子

```bash
# 方法 1: 使用完整构建脚本
bash tests/custom_ops_tests/build_and_run.sh --build-ops --editable

# 方法 2: 仅编译算子
bash csrc/ascend/build_aclnn.sh ascend910b
```

### 2. 单元测试

```bash
# 设置环境变量（如果算子未自动发现）
source /workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

# 运行单元测试
python tests/custom_ops_tests/test_chunk_gated_delta_rule.py
```

预期输出：
```
Testing npu_chunk_gated_delta_rule operator...
------------------------------------------------------------
✓ Output shape: torch.Size([16, 4, 64]), dtype: torch.bfloat16
✓ Final state shape: torch.Size([2, 4, 64, 64]), dtype: torch.bfloat16
✓ Basic test passed

✓ Test without g passed: out shape torch.Size([16, 2, 32])
✓ Test without g passed

------------------------------------------------------------
All tests passed!
```

### 3. 性能对比测试

```bash
# 运行性能对比脚本（会依次测试两种实现）
bash test_aclnn_chunk_gdn.sh
```

或者手动测试：

```bash
# 测试 baseline (PTO + Triton)
export VLLM_FL_USE_ACLNN_CHUNK_GDN=0
bash start_qwen3.6-27b.sh

# 测试 aclnn
export VLLM_FL_USE_ACLNN_CHUNK_GDN=1
bash start_qwen3.6-27b.sh
```

## 技术细节

### 算子签名

```python
torch.ops._C_ascend.npu_chunk_gated_delta_rule(
    query,              # (T, Nk, Dk) bfloat16, L2-normalized
    key,                # (T, Nk, Dk) bfloat16, L2-normalized
    value,              # (T, Nv, Dv) bfloat16
    beta,               # (T, Nv) bfloat16
    initial_state,      # (B, Nv, Dv, Dk) bfloat16/fp32
    actual_seq_lengths, # (B,) int32
    g,                  # (T, Nv) float32, optional
    scale_value         # float
) -> (out, final_state)
# 返回:
#   out: (T, Nv, Dv) bfloat16
#   final_state: (B, Nv, Dv, Dk) bfloat16/fp32
```

### 状态布局转换

```python
# vLLM ssm_state: (B, Nv, Dk, Dv)
# 算子 initial_state: (B, Nv, Dv, Dk)  <- 需要 transpose
# 算子 final_state: (B, Nv, Dv, Dk)    -> 需要 transpose 回去

initial_state = ssm_state[indices].transpose(-1, -2).contiguous()
# ... 调用算子 ...
ssm_state[indices] = final_state.transpose(-1, -2).contiguous()
```

### Fresh vs Non-Fresh 处理

```python
# Fresh sequences: 清零 initial_state
initial_state[~has_initial_state, ...] = 0

# 算子统一处理两种情况
out, final_state = npu_chunk_gated_delta_rule(...)
```

## 已知限制

1. **Python 层未完全集成**：当前仅在 `patch_qwen3_6_gdn.py` 中集成，未创建独立的 `gdn.py` 文件
2. **测试场景单一**：当前测试主要覆盖 fresh prefill，non-fresh prefill 场景需要多轮对话测试
3. **PTO 禁用**：设置 `VLLM_FL_USE_ACLNN_CHUNK_GDN=1` 会完全禁用 PTO megakernel

## 性能预期

### Fresh Prefill（主要测试场景）
- **对比对象**：PTO megakernel vs aclnn chunk_gated_delta_rule
- **测试用例**：input_len=512, output_len=128
- **关注指标**：
  - Prefill throughput (tokens/s)
  - Time to First Token (TTFT)
  - Memory usage

### Non-Fresh Prefill（需要额外测试）
- **对比对象**：Triton chunk vs aclnn chunk_gated_delta_rule
- **测试场景**：多轮对话、长文本分块
- **当前状态**：测试脚本未覆盖此场景

## 故障排查

### 算子未找到
```bash
# 检查算子是否编译安装
ls vllm_fl/_cann_ops_custom/vendors/custom_transformer/

# 设置环境变量
source vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

# 检查算子是否可用
python -c "import torch; import torch_npu; import vllm_fl._C_ascend; print(torch.ops._C_ascend.npu_chunk_gated_delta_rule)"
```

### 数据类型错误
算子要求输入为 bfloat16，如果遇到类型错误，检查：
```python
# 确保输入是 bfloat16
query = query.to(torch.bfloat16)
key = key.to(torch.bfloat16)
value = value.to(torch.bfloat16)
beta = beta.to(torch.bfloat16)
```

### 形状不匹配
检查 TND 布局转换：
```python
# 输入应该是 (1, T, N, D)，需要 squeeze(0) 得到 (T, N, D)
q_tnd = query.squeeze(0)
```

## 参考

- vllm-ascend PR #12607: https://github.com/vllm-ascend/vllm/pull/12607
- Qwen3.6 GDN 模型文档
- PTO megakernel: `vllm_fl/ops/pto_chunk_gdn/`
- Triton chunk kernel: FLA 库实现

## 联系

如有问题，请查看：
- 单元测试：`tests/custom_ops_tests/test_chunk_gated_delta_rule.py`
- 集成代码：`vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py`
- C++ 实现：`csrc/ascend/attention/chunk_gated_delta_rule/`
