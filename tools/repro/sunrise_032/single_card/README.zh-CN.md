# 7e插件：物理4上的4B TP1当前验证

本目录归档2026-11-05在可用单卡上完成的当前提交补充验证。使用已有任务容器，在其上通过`--no-deps --no-build-isolation`安装插件源码`7e0a83b5abe39bd6e7796ec9ea957003a5ae0dc4`；不是重建了一个7e不可变镜像。

| 模式 | 实际runner/调度 | 请求 | 外层状态 | graph证明 |
|---|---|---|---:|---|
| eager | SunriseV2ModelRunner / async=true | 5/5符合检查 | 0 | enforce_eager=true，NONE，无捕获graph |
| graph | SunriseV2ModelRunner / async=true | 5/5符合检查 | 0 | FULL_DECODE_ONLY，原生PTPUGraph；batch2 replay3次、batch1 replay5次 |

[eager收据](eager/receipt.json)和[graph收据](graph/receipt.json)保留原始问题、预期及答案、运行前后实际V2/async/graph字段。请求检查采用预期字符串包含于答案的小写文本；五个答案语义为Paris、2、Beijing、12、Paris，标点差异按原检查保留。

计数器在`LLM(...)`完成并确认模型ready后安装，只包装已捕获的原生graph；每次先调用原生`graph.replay()`再加1。请求前计数均为0，请求后两graph分别为3、5，因而不把capture warmup算入。这里3、5表示原生replay调用次数，不是请求数量。原生类型与ready日志见[eager](eager/ready-proof.log)、[graph](graph/ready-proof.log)。

[安装前](installation/packages-before-r2.json)与[安装后](installation/packages-after-r2.json)的Python和所列版本表完全一致；安装后收据记录三个已安装文件与7e源码文件逐字核对的SHA256。[命令摘录](installation/command-excerpt.txt)保留无依赖安装选项。`triton:null`仅表示这份分发元数据查询未找到名为triton的distribution，不能推断FlagTree提供的Triton不可导入。未列出的包不在本版本表比较范围内；脚本摘录也不替代缺失的完整pip退出日志。

[前后设备快照](device-state.json)中，物理4在两次运行前后均为0B/0%、ECC0，进程表无列出GPU进程；这只是表面状态，不是全机或peer健康证明。本轮未执行设备reset、整机重启或驱动变更。

本轮更新前的TP1 smoke曾在早期4f安装中因缺少编译器元数据字段失败；切换为7e已有修复后，本轮TP1验证通过。此前归档的TP2 collective超时是在已有该修复的2489/ed99代码上发生，仍是有效失败证据；本次TP1成功不能宣布TP2问题已修复。27B的完整TP1补充gate另行验证，35B BF16权重超出单卡容量，未尝试加载；标准TP2矩阵仍未通过。

[provenance.json](provenance.json)记录各明确选定源文件的SHA256、bytes、原始UTF-8 splitlines 1-based行范围、脱敏和转换；[SHA256SUMS](SHA256SUMS)覆盖本目录交付文件。仅替换绝对模型/任务路径及进程PID，不保留SSH/IP、主机名、凭据或完整环境。


## 复现单卡 smoke

在已分配物理4、基础完整性检查通过的厂商环境中，进入独立tmux会话，保留厂商PYTHONPATH。每次输出目录必须尚不存在：

```bash
tmux new-session -s sunrise-tp1-UNIQUE
source /opt/sunrise-build/environment.sh
: "${OUT_DIR:?填入尚不存在的输出目录}"
TANG_VISIBLE_DEVICES=4 TA_CLOSE_GRAPH_MEMORY=1 \
VLLM_USE_V2_MODEL_RUNNER=1 GEMS_VENDOR=sunrise VLLM_PLUGINS=fl \
timeout --kill-after=5s 480s \
  python -u tools/repro/sunrise_032/single_card/smoke_single_card_v2.py \
    --mode graph --output-dir "$OUT_DIR"
```

`--mode eager`使用另一个新目录。脚本与本轮执行文件逐字节一致；它在请求前记录counter0，并在请求后要求每个真实PTPUGraph回放>0。异常后停止后续GPU阶段，保留状态；不自动retry/reset。这里的5题smoke不能替代两个大模型的26请求及长上下文Gate。
