# GLM-5.2：jia 最新基线上的优化集成

后续 GLM 开发入口是工作区相对路径
`projects/flagos_bw1000/repos/vllm-plugin-fl-jia741732680`，
本地分支 `agent/glm52-optimized-20260920`，上游基线 `68f3340`。
未推送远端。旧集成工作树和已测快照保留作来源证据。

## 已迁入的代码

- INT8 indexer loader、FP8 group quant、block1 indexer、CustomOp/Graph 边界。
- BF16 厂商 FlashMLA 路径、Graph 正确释放、异步 indexer gather。
- 可选的 Graph/eager 双 RCCL 通信器，保留异步调度。
- 656-byte FP8 MLA KV writer、metadata 复用、slot padding 声明修复。
- 自研 Triton/Gluon grouped_fullvalid MLA 及完整源码依赖。

同事最新 DeepSeek-V4 改动保留；动态 W8A8 的三个关键模块与此前验证
集成版一致，无需覆盖。MoE 默认 Triton，LightOp/Marlin 仍为可选候选。
自研仅限 gfx936、4 heads、Q576/V512、topk2048、656-byte packed KV，
scale 遵循当前 writer 的 2 次幂约定。切换 route 必须重启并重新捕获 Graph。

## 依赖与源码选择

源码集成不会自动替换系统 editable 安装。启动入口从自身位置选择新插件，
FlagGems 与 Compressor 通过 `--flaggems-root` / `--compressor-root` 明确指定；
也支持 `GLM52_FLAGGEMS_ROOT` / `GLM52_COMPRESSOR_ROOT`。
两节点必须各自检查真实 import、原生扩展和运行库。

当前已验证依赖均以工作区根为逻辑基准：

- vLLM：`projects/flagos_bw1000/repos/vllm-v0.24.0`，
  `ee0da84ab9e04ac7610e28580af62c365e898389`，empty venv。
- FlagGems：原 GLM 会话父目录内 `FlagGems--glm52-w8a8-integration`，
  `a4a5d86d73eb3c5287ad5b85714cb0ee1a47ff02`；
  `flaggems.patch` 保存相对 `ed2508b` 的必要修复，不能遗漏 indexer 完整 head 分块。
- Compressor：同父目录 `FlagOS-Compressor--glm52-w8a8-integration`，
  `b0bc4980d83fb9ab1ff50f3fcd826578d90ecd6d`。

依赖父目录为
`projects/flagos_bw1000/worktrees/10.232.2.18--yxa-plugin-glm5_2--flagos_bw1000--glm-5.2`。
它们仍作为只读依赖使用；GLM 插件后续修改在 jia 仓库完成。
本仓现有 `_C.so` 原样保留，属于未跟踪的机器本地构建产物。
shca/GDR 库仍用 `--network-backend shca --rdma-library-path` 显式指定，
不修改系统库。不同机器不得直接照搬本地运行库路径。

## 启动与回退

`profile_dual_node.py --help` 列出完整参数；先使用 `--dry-run`。
`--project`、`--model`、`--run-root` 和以上依赖根按当前机器显式指定。
既有 BF16 行为保持默认。要复用 FP8 组合，在当前参数上增加：

```text
--kv-cache-dtype fp8_ds_mla
--fp8-mla-impl triton_grouped
--fp8-reuse-metadata
--mla-kernel native_flashmla
--split-graph-comm
```

回退 FP8 厂商实现：`--fp8-mla-impl vendor`（默认）。
直接用 vLLM CLI 时对应 `--kv-cache-dtype fp8_ds_mla`，环境变量：
`VLLM_FL_HYGON_GLM_SPARSE_BACKEND=flaggems`、
`VLLM_FL_HYGON_GLM_MLA_KERNEL=native_flashmla`、
`VLLM_FL_HYGON_SPLIT_GRAPH_COMM=1`、
`VLLM_FL_GLM_FP8_REUSE_METADATA=1`、
`VLLM_FL_GLM_FP8_MLA_IMPL=triton_grouped`（回退 `vendor`）。

- `--benchmark-only`：同负载预热后正式测量，并精确停止本轮进程。
- `--no-async-scheduling`：可选串行调度；双通信器路线保留 async。
- `--profile-seconds`：自然退出后离线导出窗口，避免当前工具链在线 stop 问题。
- census 仅用于采样，不作为性能成绩。

## 证据与边界

FP8/MLA 八个实现文件与 2026-09-18 已测快照逐字一致。历史单轮
TP16、4K→1K、C64/N256、159056 KV tokens 下，厂商/自研分别为
869.46/897.22 total tok/s，耗时下降 3.09%。这是旧快照的证据，
不冒充本次新上游集成后的整模型复测；答案质量与多轮稳定性未随本次迁移验收。

后续先从这个分支生成独立测试快照，再进行优化和 A/B；
不要修改正在运行的快照，不把仅生成 token 的吞吐标成 total tok/s。

本次新基线集成验证：62 项 CPU/合约/Graph 接线/dry-run 回归通过；
未在新组合上启动完整模型或 GPU kernel 性能测试。
