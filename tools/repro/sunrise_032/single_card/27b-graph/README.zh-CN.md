# 27B：物理4上的TP1 graph补充gate

本目录归档当前插件生产代码7e提交在单卡上的已完成实测。Qwen3.6-27B在物理4、TP1、Runner V2、FULL_DECODE_ONLY下完成原5共享场景的26请求/260项原检查，独立离线重算全部通过。长prefill测得4118 prompt tokens，输出`violet maple 42`，10项检查通过。

| 共享报告 | 请求 | 原检查 | 结果 |
|---|---:|---:|---|
| [单条文本](reports/text_single.json) | 1 | 10 | PASS |
| [8并发文本](reports/text_concurrent_8.json) | 8 | 80 | PASS |
| [单图](reports/image_single.json) | 1 | 10 | PASS |
| [8并发图像](reports/image_concurrent_8.json) | 8 | 80 | PASS |
| [8并发混合](reports/mixed_concurrent_8.json) | 8 | 80 | PASS |

[长prefill](long-prefill.json)及五份共享report保留全部检查、文本、响应和usage；仅明确规范化绝对模型/图片/运行路径、loopback与临时服务别名。输入与原例匹配、17份用例/图片source bytes与Git7e一致的独立复核见[independent-verification](independent-verification.json)。17-source匹配证明用例来源，**不冒充安装wheel attestation**。

[实际配置](effective-configuration.json)：TP1、context8192、seq8、KV4GiB、V2、enforce_eager=false、compilation0/eager、FDO capture sizes`[1,2,4,8]`；KV容量39,996 tokens。[startup摘录](startup-excerpt.log)同时保留原生V2 worker、capture完成、KV容量和application ready行。

**replay证明不是capture证明。** [字节偏移证明](post-ready-byte-proof.json)对原始server.log从0-based字节偏移`277752`后独立解析：tail为115,188 bytes，22条原生FULL metric表格记录的`Count`求和为**295次**，与summary及独立verification精确一致。capture完成和application ready的结束字节均小于此边界。全部post-ready FULL行见[原生metrics摘录](post-ready-full-metrics.log)，保留原始1-based splitlines行号与每行原始byte区间及源/尾段SHA256；不能把该偏移用于归一后的文本。

这里使用**vLLM原生运行时FULL metrics**，没有额外的27B PTPUGraph RPC包装计数。4B smoke的ready后wrapper计数是另一种独立证据；两者不混称。表格22行不等于22次请求或295条日志，295是其Count总和。capture进度、配置表或“Graph capturing finished”均不替代上述post-ready执行指标。

[summary](summary.json)记录gate/outer0与owned cleanup完成；[前后状态](device-state.json)中物理4均0B/0%、ECC0、无列出GPU进程。该状态表不证明全机或peer健康。本包没有reset/reboot结论，也不追加GPU试验。

这是TP1补充结果，**不能计入要求TP2的四场景矩阵**；独立复核明确`tp2_matrix_eligible=false`。归档2489/ed99的TP2异步watchdog仍有效、未解决，不能因本次TP1成功宣布旧TP2修复。本轮更新前TP1 graph-r2缺字段错误来自still4f安装，与归档TP2 watchdog不是同一问题。

[provenance](provenance.json)记录每个明确源文件的SHA256/bytes/原始行范围、RFC6901 JSON pointers和逐字段规范化；[SHA256SUMS](SHA256SUMS)覆盖本目录。原始本地证据不改写，原始SHA与脱敏后SHA分别记录。全部检查/输出文本/响应usage保持不变，路径替换不冒充原始字节。
