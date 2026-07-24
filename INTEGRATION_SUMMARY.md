# npu_chunk_gated_delta_rule 集成完成报告

## 📋 任务概述

将 vllm-ascend PR #12607 中的 `npu_chunk_gated_delta_rule` 算子集成到 vllm-plugin-FL，用于优化 Qwen3.6-27B GDN 模型的 prefill 性能。

## ✅ 完成的工作

### 阶段一：C++ 算子层（已完成）

#### 1. 算子源码（27 个文件）
```
csrc/ascend/attention/chunk_gated_delta_rule/
├── CMakeLists.txt
├── chunk_gated_delta_rule_torch_adpt.h  ✅ 修复命名空间 vllm_ascend -> vllm_fl
├── op_graph/
│   ├── CMakeLists.txt
│   └── chunk_gated_delta_rule_proto.h
├── op_host/
│   ├── CMakeLists.txt
│   ├── chunk_gated_delta_rule_def.cpp
│   ├── chunk_gated_delta_rule_infershape.cpp
│   ├── chunk_gated_delta_rule_tiling.cpp
│   ├── chunk_gated_delta_rule_tiling.h
│   └── op_api/ (4 个文件)
└── op_kernel/
    ├── arch22/ (5 个文件 - A2/910b 支持)
    ├── arch35/ (5 个文件 - A3/910_93 支持)
    └── 4 个通用文件
```

#### 2. 构建配置
- ✅ `csrc/ascend/build_aclnn.sh` - 添加 `chunk_gated_delta_rule` 到 ascend910b 算子列表

#### 3. Torch 绑定
- ✅ `csrc/ascend/torch_binding.cpp`
  - 添加头文件引用
  - 注册算子定义和实现
  
- ✅ `csrc/ascend/torch_binding_meta.cpp`
  - 实现 `npu_chunk_gated_delta_rule_meta` 函数
  - 注册 meta 实现用于图捕获

#### 4. 单元测试
- ✅ `tests/custom_ops_tests/test_chunk_gated_delta_rule.py`
  - 基础功能测试（带/不带 g 参数）
  - 输入/输出形状验证
  - 数据类型验证（bfloat16）
  - **测试状态：通过** ✅

### 阶段二：Python 集成层（已完成）

#### 1. 集成到 GDN Patch
- ✅ `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py`

**修改内容**：
1. 添加 `npu_chunk_gated_delta_rule` 到 `_REQUIRED_OPS`
2. 实现 `_chunk_gated_delta_rule_aclnn()` 封装函数：
   - 状态布局转换：`(B, Nv, Dk, Dv)` ↔ `(B, Nv, Dv, Dk)`
   - TND 布局转换：`(1, T, N, D)` → `(T, N, D)`
   - L2 归一化：`l2norm_fwd(query/key)`
   - 状态清零：`initial_state[~has_initial_state, ...] = 0`
   - 序列长度转换：`cu_seqlens` → `actual_seq_lengths`

3. 修改 `_forward_core()` 方法的 prefill 路径：
   - 添加环境变量控制 `VLLM_FL_USE_ACLNN_CHUNK_GDN`
   - Level 0（默认）：PTO + Triton（保持现有行为）
   - Level 1：aclnn（完全替换 prefill 路径）

#### 2. 代码逻辑
```python
if attn_metadata.num_prefills > 0:
    use_aclnn = int(os.environ.get("VLLM_FL_USE_ACLNN_CHUNK_GDN", "0"))
    
    if use_aclnn == 1:
        # 新路径：使用 aclnn 统一处理 fresh + non-fresh
        core_attn_out_non_spec, last_recurrent_state = _chunk_gated_delta_rule_aclnn(...)
    elif _pto_prefill_usable(attn_metadata):
        # 原路径 1：PTO megakernel (fresh only)
        core_attn_out_non_spec, last_recurrent_state = _chunk_gdn_pto(...)
    else:
        # 原路径 2：Triton chunk (non-fresh fallback)
        core_attn_out_non_spec, last_recurrent_state = _qwen3_next_lib.chunk_gated_delta_rule(...)
```

### 阶段三：测试和文档（已完成）

#### 1. 性能测试脚本
- ✅ `test_aclnn_chunk_gdn.sh`
  - 自动运行 baseline（PTO + Triton）
  - 自动运行 aclnn 测试
  - 结果对比输出

#### 2. 文档
- ✅ `ACLNN_CHUNK_GDN_INTEGRATION.md` - 完整集成文档
  - 概述和算子对比
  - 环境变量说明
  - 使用方法和示例
  - 技术细节和故障排查

## 📊 算子使用分布

| 阶段 | Level 0（默认） | Level 1（新） |
|------|----------------|--------------|
| Speculative Decode | `npu_recurrent_gated_delta_rule` | `npu_recurrent_gated_delta_rule` |
| **Prefill (fresh)** | **PTO megakernel** | **npu_chunk_gated_delta_rule** |
| **Prefill (non-fresh)** | **Triton chunk** | **npu_chunk_gated_delta_rule** |
| Decode | `npu_recurrent_gated_delta_rule` | `npu_recurrent_gated_delta_rule` |

## 🎯 关键技术点

### 1. 命名空间修复
- PR 源码：`namespace vllm_ascend`
- 当前仓库：`namespace vllm_fl`
- ✅ 已修复所有引用

### 2. 状态布局转换
```python
# vLLM cache: (B, Nv, Dk, Dv)
# 算子需要:   (B, Nv, Dv, Dk)
initial_state = ssm_state[indices].transpose(-1, -2).contiguous()
final_state = result.transpose(-1, -2).contiguous()
```

### 3. 统一状态清零
```python
# 替代 PTO 的 "只支持 fresh" 和 Triton 的 "手动清零"
initial_state[~has_initial_state, ...] = 0  # 统一处理
```

### 4. 数据类型要求
- 输入：`bfloat16`（query, key, value, beta）
- g：`float32`（log-gate 值）
- actual_seq_lengths：`int32`
- 输出：`bfloat16`

## 🧪 测试结果

### 单元测试
```bash
$ python tests/custom_ops_tests/test_chunk_gated_delta_rule.py
✓ Output shape: torch.Size([16, 4, 64]), dtype: torch.bfloat16
✓ Final state shape: torch.Size([2, 4, 64, 64]), dtype: torch.bfloat16
✓ Basic test passed
✓ Test without g passed
------------------------------------------------------------
All tests passed!
```

### 集成测试
- ⏳ 等待运行性能对比测试

## 📝 使用指南

### 快速开始
```bash
# 1. 编译安装算子
bash tests/custom_ops_tests/build_and_run.sh --build-ops --editable

# 2. 测试单元功能
python tests/custom_ops_tests/test_chunk_gated_delta_rule.py

# 3. 运行性能对比
bash test_aclnn_chunk_gdn.sh
```

### 环境变量控制
```bash
# 使用默认实现（PTO + Triton）
export VLLM_FL_USE_ACLNN_CHUNK_GDN=0

# 使用新的 aclnn 算子
export VLLM_FL_USE_ACLNN_CHUNK_GDN=1
```

## 📈 性能预期

### 测试场景
- **输入**：input_len=512, output_len=128
- **模型**：Qwen3.6-27B
- **硬件**：Ascend 910B (A2)
- **批大小**：单请求

### 对比维度
1. **Prefill 阶段**：
   - Baseline：PTO megakernel
   - New：npu_chunk_gated_delta_rule
   
2. **Decode 阶段**：
   - 两者相同：`npu_recurrent_gated_delta_rule`

3. **关键指标**：
   - Prefill throughput (tokens/s)
   - Time to First Token (TTFT)
   - Total latency
   - Memory usage

## 🔍 已知限制

1. **测试覆盖**：主要覆盖 fresh prefill，non-fresh 场景需要多轮对话测试
2. **独立 Python 模块**：未创建独立的 `gdn.py`，集成在 patch 文件中
3. **开关粒度**：当前是全局开关，未实现细粒度控制

## 📂 修改文件清单

### C++ 层
- `csrc/ascend/build_aclnn.sh` (1 行修改)
- `csrc/ascend/torch_binding.cpp` (12 行新增)
- `csrc/ascend/torch_binding_meta.cpp` (21 行新增)
- `csrc/ascend/attention/chunk_gated_delta_rule/` (27 个新文件)

### Python 层
- `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py` (约 80 行修改/新增)

### 测试和文档
- `tests/custom_ops_tests/test_chunk_gated_delta_rule.py` (新文件)
- `test_aclnn_chunk_gdn.sh` (新文件)
- `ACLNN_CHUNK_GDN_INTEGRATION.md` (新文件)

## 🚀 下一步

1. **运行性能测试**：
   ```bash
   bash test_aclnn_chunk_gdn.sh
   ```

2. **对比结果分析**：
   - 查看 `/workspace/results/` 下的测试报告
   - 对比 baseline 和 aclnn 的性能指标

3. **扩展测试场景**（可选）：
   - 多轮对话测试（non-fresh prefill）
   - 不同 batch size
   - 不同 input/output 长度组合

4. **生产环境部署**（如果性能满意）：
   - 更新部署脚本添加环境变量
   - 制定回滚方案（设置 `VLLM_FL_USE_ACLNN_CHUNK_GDN=0`）

## ✨ 总结

✅ **C++ 算子层**：完整移植，支持 A2/A3 架构  
✅ **Python 集成层**：开关控制，向后兼容  
✅ **单元测试**：通过验证  
✅ **文档完善**：使用指南和技术细节  
⏳ **性能测试**：待运行对比  

**集成状态**：准备就绪，可以开始性能测试！
