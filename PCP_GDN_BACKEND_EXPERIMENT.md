# 64K×1 GDN 后端实验

日期：2026-10-13。

完整原始记录：`/vllm-workspace/validation/qwen36_pcp_gdn_backend_64k_20261013`。

状态：性能目标达到；数值不等价，保持显式实验选项，未修改生产默认。

## 实测结果

| 条件 | 中位数 ms | P10–P90 ms | 两轮中位数 ms |
|---|---:|---:|---:|
| TP / 原生 Norm / FlashInfer GDN | 889.248 | 887.367–891.267 | 887.791 / 889.808 |
| TP / 融合 Norm / FlashInfer GDN | 752.812 | 751.169–754.337 | 751.912 / 753.130 |
| TP / 原生 Norm / Triton GDN | 860.233 | 858.599–863.510 | 860.165 / 860.234 |
| TP / 融合 Norm / Triton GDN | 724.367 | 723.112–727.499 | 723.826 / 725.614 |
| 当前 PCP / 融合 Norm / FlashInfer GDN | 680.436 | 679.004–682.241 | 680.436 / 680.403 |
| 实验 PCP / 融合 Norm / Triton GDN | 651.559 | 650.002–654.111 | 651.288 / 652.911 |

实验 PCP 比同轮原生 TP 延迟降低 **26.729%**；比当前 PCP 再降低 **4.244%**（28.877 ms）。
原生 TP 的 25% 降延迟门槛为 666.936 ms，实验值与门槛的裕量为 15.378 ms（正数代表达标）。
TP 也应用相同 Norm 和 GDN 优化后，PCP 的优势为 **10.051%**。后端切换属于两者均可使用的通用优化，不能把相对未经这些优化的原生 TP 的全部差距解释成 PCP 独有收益。

指标为一次 LLM.generate 的完整 65536-token Prefill 加首个真实 token 的墙钟时间；不是 HTTP TTFT、长输出 Decode 吞吐或混合并发服务。全部调用均为单请求。

## 数值与有限任务检查

仅性能达标不等于无损优化。以下测试发生在独立的带诊断进程；计时进程不包含这些比较、logit hook 或 profiler。
- 120 次真实 GDN 层/rank 同输入比较，输出和最终 recurrent state 均有限，输入 initial state 均未被改写。输出最大绝对差 0.01562500，状态最大绝对差 0.13010120；两者均非逐位一致。
- 输出/状态的最大相对 RMS 差分别为 0.6250% / 0.9741%。这是算子级误差，不是任务准确率。
- 同后端重复对照：128/128 步完整 logits 逐位一致。换后端后，同一 token 历史下 0/128 步 logits 逐位一致，128/128 步 argmax 一致。
- 换后端 logits 最大绝对差 2.562500，平均 RMS 差 0.117302，平均/最大 KL(reference‖candidate) 为 0.002978 / 0.045706。
- 首 token logits 最大绝对差 0.656250；相对 RMS 差 5.5234%。
- 另行不强制历史的 128-token 自由生成：tokens 一致=True；保存的 top-5 logprobs 一致=False。
- 2 个 seed × 3 个深度的 64K 六位口令检索：两后端均通过=True（每后端 6/6）。
完整 logits 在四个 rank 重复检查；上述 128 步按 rank 0 计数，不把重复 rank 当成独立任务样本。同历史比较会返回参考 logits 强制保持输入一致，因此那一阶段的 token 一致不作为自由生成一致的证据。
这是一条长提示词和六个检索样本的有限筛查，不能证明广泛质量等价；尚未覆盖非零历史的实际模型 chunked prefill、长 Decode、混合请求或不同形状。本分支仅保留显式实验选择，不改默认派发。

## 实验控制

GPU 0–3，4×H100 80GB；/data/model Qwen3.6-35B-A3B（Qwen3.5 MoE 架构），BF16 / TP4 / eager / FA3 / Triton MoE；max_model_len=66560、max_num_batched_tokens=65536、max_num_seqs=64、KV cache=4 GiB/GPU，关闭 prefix cache，seed=42。max_num_seqs 是容量上限，实际始终只有 1 个请求。
TP → PCP → PCP → TP 四个进程；第二轮反转内部 Norm/后端顺序。每段每条件 3 次预热、10 次正式样本，六条件各 20 次，共 120 个正式样本，保留所有结果，不裁异常值。
每条件计时前独立诊断确认全部 30 层 GDN 每 rank 恰好各调用一次，输入为 65536 tokens、1 个序列且 chunk metadata 有效；随后恢复原生 bound method，再开始预热/计时。Full Attention 确认 10 层 FA3，PCP prefill 次数与实际调用数一致。
两边 Norm 对齐使用先前 Norm 消融的诊断 helper（101 个 Gemma Norm），仅修改导入路径；fused_norm kernel 与该已验证实现相同。GDN gated Norm 不属于此切换。普通 PCP 默认仍使用既有显式 Norm 路径。
每进程启动前持共享 GPU lock 并检查所有 GPU 进程、显存、利用率；有占用即拒绝启动。launch.json、source/ 和每段脚本保留环境、源代码哈希、GPU 空闲快照，报告重新校验全部源文件。
## 代码与复现

工作树：/vllm-workspace/vllm-plugin-FL-pcp-gdn-backend；分支 experiment/pcp-gdn-backend-64k-20261013，基于 d5798be。之前三项优化和已安装 vLLM 均未修改。
vllm_fl/cp/gdn_backend_probe.py 仅按显式调用在当前 worker 的原生 ChunkGatedDeltaRule.forward_cuda（FlashInfer）与 forward_native（Triton/FLA）之间选择，支持恢复原派发，没有注册或自动启用。原生 vLLM 同样提供 additional_config={"gdn_prefill_backend":"triton"} 启动选项。Decode 的 recurrent kernel 未被切换。
正确性脚本：correctness.py、check_worker.py、gpu_guard.py。性能脚本：run_timing.py、timing.py、timing_worker.py、benchmark_norm_ablation.py、benchmark_fused_norm.py。report.py 汇总并校验结果。
复跑时将这些脚本放入 /vllm-workspace/validation 下新的同级目录，保留 qwen36_64k_20260928/prompt_ids.json 与公共 GPU lock 路径；run_timing.py 仅在空闲时启动。脚本拒绝覆盖已有结果。
原始结果：correctness.json、r1_tp/result.json、r1_pcp/result.json、r2_pcp/result.json、r2_tp/result.json；汇总：summary.json、samples.csv。

## 后续 Nsight 归因

2026-10-13 的实际模型 trace 已确认 GDN core GPU 时间 98.410 → 70.019 ms；FlashInfer 核心 grid=8 blocks，FLA 状态递推 grid=32 blocks，块内阶段 grid=8192/16384 blocks。详见 [PCP_GDN_TRACE_ANALYSIS.md](PCP_GDN_TRACE_ANALYSIS.md)。这是 profiler 归因数据，正式端到端延迟仍采用本文的独立基准。
