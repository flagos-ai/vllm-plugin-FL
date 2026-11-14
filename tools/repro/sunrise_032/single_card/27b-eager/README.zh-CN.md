# 27B：物理4上的TP1 eager补充gate

本目录归档当前插件生产代码7e提交在单卡上的已完成实测。Qwen3.6-27B在物理4、TP1、eager、Runner V2下完成原5个共享场景的26请求，保留260项原质量检查；独立离线按原例重算全部通过。另一次长prefill测得4118 prompt tokens，输出`violet maple 42`，10项检查通过。

| 证据 | 请求 | 原检查 | 结果 |
|---|---:|---:|---|
| [单条文本](reports/text_single.json) | 1 | 10 | PASS |
| [8并发文本](reports/text_concurrent_8.json) | 8 | 80 | PASS |
| [单图](reports/image_single.json) | 1 | 10 | PASS |
| [8并发图像](reports/image_concurrent_8.json) | 8 | 80 | PASS |
| [8并发混合](reports/mixed_concurrent_8.json) | 8 | 80 | PASS |
| [长prefill](long-prefill.json) | 1 | 10 | PASS，4118 tokens |

五份共享report保留全部input、checks、输出文本、原响应和usage；长prefill也保留完整文本/响应/usage/检查。只规范化模型/图片/运行目录、loopback地址和临时服务别名，明确变更的JSON pointer记录在[provenance](provenance.json)中。图片请求仍对应原文件basename及17份源文件/图片SHA256；没有改prompt、放宽检查、替换图片或把失败请求删去。

[实际配置](effective-configuration.json)来自本次启动日志：TP1、max_model_len8192、max_num_seqs8、KV4GiB、enforce_eager=true、use_v2_model_runner=true、compilation0/eager、cudagraph NONE；可用KV容量39,996 tokens。关键原生启动行见[startup摘录](startup-excerpt.log)。其中“V1 LLM engine”是上游engine标签，独立的worker日志明确使用Sunrise Model Runner V2。

[summary](summary.json)记录outer/gate return0及owned cleanup完成；[设备状态](device-state.json)表明物理4运行前后0B/0%、ECC0、无列出GPU进程。只读状态不证明全机或peer健康。本补充验证没有reset/reboot，也没有graph验收结论。

[独立复核](independent-verification.json)核实26个输入与原共享用例精确对应，重算260项检查和长prefill10项检查；17份用例/图片source bytes与Git `7e0a83b5abe39bd6e7796ec9ea957003a5ae0dc4`一致。这个17-file匹配证明用例来源，**不是已安装wheel的完整attestation**；插件安装证明另见4B current提案，不能用Git路径或response fingerprint代替它。

这是TP1补充结果，**不得进入要求TP2的四场景矩阵**。独立复核明确`tp2_matrix_eligible=false`，严格TP2配置检查因tensor_parallel_size不符而拒绝。归档2489/ed99的TP2异步watchdog仍有效、尚未解决；本轮更新前TP1缺字段错误来自4f安装失配，两者不能混同。27B graph当时仍在独立运行，此包不推测其结果，也不宣称完整适配gate已完成。

[SHA256SUMS](SHA256SUMS)覆盖本目录提案；所有原始本地文件未变。JSON pointer使用RFC6901的`~0`/`~1`转义；日志行号为UTF-8解码后`splitlines()`的1-based范围（含CR刷新行）。绝对路径的占位归一不会冒充未经修改的原始bytes，原始SHA与归一后SHA分别记录。
