# 曦望 vLLM 0.32：27B 图片异常与 TP2 graph 排查包

> 同日单卡补充：在已安装PR固定提交的物理4上，4B Runner V2 eager/graph各5/5通过，实际请求graph回放3/5次；见[单卡验证](single_card/README.zh-CN.md)。这不替代标准TP2矩阵。

> 2026-11-05 更新：按用户要求在不 reset 的当前环境做了有界基础检查；物理 0、2 的纯 Torch 大张量往返拷贝失败，4 通过。完整 TP2 gate 仍未完成，GPU 测试已再次停止。新增[逐卡全量复现、运行命令及证据](current_environment/README.zh-CN.md)。以下三类模型/graph 入口和 53 项验证是先前 81aea8d 的离线交付记录。

本目录供曦望侧离线审阅和后续在**已获授权的独占环境**手动复现。
本次只整理了已有脚本与历史证据，并做 CPU/语法检查，**没有重新执行 GPU 测试或 reset**。
整理版脚本尚未 GPU 重测；下列历史结果不能当作整理版的新运行结果。
完整四组合 Gate 仍未通过。

## 环境与测试版本

基础镜像：`harbor.baai.ac.cn/flagos-app/vllm0.24.0-sunrise-tangrt1.2.0:2.3.0-0.3.0`，
按 [Sunrise 构建说明](../../../docker/sunrise/README.md) 安装本 PR 的 `0.32.0+empty`。
本目录不安装包、不修改驱动，也不自动下载模型。

| 项目 | 既有证据中的值 |
| --- | --- |
| 设备 / 驱动 | SR-SUN-S2-X1 / 0.24.0 |
| Python / torch / torch_ptpu | 3.10.20 / 2.11.0+cpu / 0.2.3+torch2.11 |
| vLLM / 插件 | 0.32.0+empty / 0.5.0.dev0 |
| FlagTree / Triton / FlagGems / FlagCX | 0.7.0+sunrise3.6 / 3.6.0 / 5.4.0 / 0.14.0+tangrt1.2.0 |
| FlagGems-vllm | 源码 `b5c2c2249319277ccda4d41aa0f65164bd9b2a19`；包 metadata 仍为 0.1.0 |
| 27B 原始 eager Gate | 插件 `4f15be0`，Qwen3.6-27B，TP2、BF16、Runner V2 |
| 原始 4B TP2 graph / collective 探针 | 插件 `2489ee1`；模型入口为 Runner V2，独立 collective 探针不加载模型 |
| 35B eager 成功对照 | 插件 `ed99a0e`；26/26 共享请求及 4118-token 长输入通过 |

版本摘录见 [environment.json](logs/environment.json)。Python 3.10 是用户确认采用的下游兼容构建，
上游 vLLM 0.32 的 Python 要求及补丁记录见镜像说明；这里没有更换厂商二进制栈。

## 运行前的资源边界

以下命令仅供厂商后续手动执行，**本次整理没有执行这些命令**。
先通过资源管理员/调度平台确认独占卡对、直连拓扑、设备健康及服务端口。
`TANG_VISIBLE_DEVICES` 只限制本进程可见设备，不提供设备预约或硬件隔离；
进程表为空也不能证明设备干净或没有其他人预订。
当前保留的异常设备状态不能直接用于重跑。

在自己的隔离容器、仓库根目录和独立 tmux 会话中运行；不要在其他人的服务容器中操作。
`RESERVED_PAIR`、模型路径和端口需由操作者显式填写，没有自动选卡。
容器启动时就应限定获授权的可见卡；每个新 tmux 会话内重新设置并确认变量，不依赖 tmux 继承外层 shell。
下面每个 GPU 命令也会通过 `${RESERVED_PAIR:?}` 拒绝空设备变量。
脚本不会 reset、停止容器、杀其他进程、卸载驱动或自动重试。

```bash
# 由管理员确认后，在自己的容器/终端填写；本包不提供默认物理卡号。
: "${RESERVED_PAIR:?请设置已获授权的健康直连卡对}"
: "${MODEL_27B:?请设置本地 Qwen3.6-27B 路径}"
: "${REPRO_PORT:?请设置自己的空闲本地端口}"
source /opt/sunrise-build/environment.sh
```

## A. 27B 图片异常：已有服务上的单个请求

[repro_27b_image.py](repro_27b_image.py) 只使用 Python 标准库，不启动或停止服务。
它先检查 `/v1/models` 中的服务名及模型 root，随后发一个图像请求；
`--text-control` 才会额外发送一个短文本对照。HTTP 请求/响应原始字节、退出原因和
图片 SHA256 保存在新输出目录中，没有 logprobs 请求。

### 服务参数（原始 eager 配置）

若厂商已有参数一致的自有服务，直接使用它。需要单独启动时，在自己的 tmux 会话中运行：

```bash
tmux new-session -s "sunrise-27b-$(date +%s)"
# 在该会话内，仓库根目录，且已设置上面的三个变量：
source /opt/sunrise-build/environment.sh
SERVER_LOG_DIR=$(mktemp -d /tmp/sunrise032-27b.XXXXXXXX)
TANG_VISIBLE_DEVICES="${RESERVED_PAIR:?请设置已获授权的健康直连卡对}" \
GEMS_VENDOR=sunrise VLLM_PLUGINS=fl USE_FLAGGEMS=1 \
VLLM_USE_V2_MODEL_RUNNER=1 PCCL_PROTO=Simple \
env -u TA_CLOSE_GRAPH_MEMORY \
vllm serve "${MODEL_27B:?请设置本地模型路径}" \
  --served-model-name sunrise-repro-27b \
  --host 127.0.0.1 --port "${REPRO_PORT:?请设置自己的空闲端口}" \
  --tensor-parallel-size 2 --disable-custom-all-reduce \
  --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 8 \
  --gpu-memory-utilization 0.92 --kv-cache-memory-bytes 4294967296 \
  --skip-mm-profiling --trust-remote-code --cudagraph-metrics --enforce-eager \
  --allowed-local-media-path "$PWD/tools/adaptation-gate-cases/images" \
  > "$SERVER_LOG_DIR/server.log" 2>&1
```

这里 process-local 的 `env -u TA_CLOSE_GRAPH_MEMORY` 对齐原始 R1 服务的 unset 设置；
正式 Gate 清单中的 graph 设置为 `1`，二者在证据中分别标注。
`--skip-mm-profiling` 仅跳过启动画像，图像请求仍执行真实视觉模型。
本脚本的模型身份检查不替代 [完整 Gate](../../../.github/scripts/sunrise/README.md) 的实际配置校验。

### 原始 R1 请求参数

在同一自有容器的另一个 tmux 会话运行客户端。客户端和服务端必须看到相同的仓库图片路径：

```bash
python tools/repro/sunrise_032/repro_27b_image.py \
  --endpoint "http://127.0.0.1:$REPRO_PORT/v1" \
  --model "$MODEL_27B" --served-name sunrise-repro-27b \
  --image tools/adaptation-gate-cases/images/image_05.png \
  --output-dir "27b-image-r1-$(date +%s)"
```

默认参数与 R1 单图案例一致：file URI、原始 OCR prompt、temperature=0、
`enable_thinking=false`、max_tokens=128，**不添加** token-ID 或 skip-special-token 覆盖。
图片 SHA256：`197185d21a8fd09bc501381a5f0fac34be6799a4239b068435f8b039a1e15e5b`。
正常结果应包含按顺序出现的 `Hello` 和 `VLLM`。
R1 历史观测为 HTTP 成功、content 空、finish_reason=length、completion_tokens=128。
原始完整 Gate 为 21/26：5 个图像相关请求失败，文本 9/9 通过。

### 独立的 R2 token 诊断

```bash
python tools/repro/sunrise_032/repro_27b_image.py \
  --endpoint "http://127.0.0.1:$REPRO_PORT/v1" \
  --model "$MODEL_27B" --served-name sunrise-repro-27b \
  --image tools/adaptation-gate-cases/images/image_05.png \
  --diagnostic-token-ids --text-control --vocab-size 248320 --timeout 600 \
  --output-dir "27b-image-r2-$(date +%s)"
```

该模式改为 max_tokens=8、`return_token_ids=true`、`skip_special_tokens=false`。
R2 历史观测是同图返回 8 个 `248320`，等于 vocab_size，超出合法范围 `[0, 248319]`。
它与 R1 的 empty/128 是不同请求，不能合并成一次运行。
历史 1063/4103-token 纯文本控制正确；本最小客户端只可选发送一个短文本控制。
`--image-transport data-url` 可传输相同 PNG 字节，但它是另一个未 GPU 重测的便携变体。

客户端 exit0 表示没有检测到本脚本定义的图像症状，**不代表完整 Gate 通过**；
exit1 表示空内容、越界 token 或预期文字缺失；exit2 表示身份检查/采集失败。

## B. TP2 graph 超时：不依赖模型权重的 collective 探针

[repro_tp2_graph.py](repro_tp2_graph.py) 保留已执行探针的核心顺序：
BF16 `[N, 2560]`、32 层真实 FlagCX allreduce、每层乘 0.5，
在同一个 capture stream/shared pool 捕获 N=2 和 N=1 的真实 PTPUGraph，
然后依次提交 `[2,2,1,2,1,1,2,1]` 的 replay→eager allgather，最后一次 synchronize。
循环内没有逐步 barrier、device synchronize 或 stream-fence。
为保留已触发失败的条件，维持 32 层/8 步；尚未在 GPU 上继续缩减规模。
两 rank 共用父进程生成的 UUID FileStore，避免重启后的 PID 重用命中旧 rendezvous。

**这个探针不实例化模型或 Runner V2。** 它隔离 collective/shared-pool/异步顺序，
不能单凭相似超时证明与完整模型同根因。下面是显式 TA=1 的历史对照参数：

```bash
tmux new-session -s "sunrise-tp2-$(date +%s)"
# 在自己的会话/容器中，已设置 RESERVED_PAIR：
source /opt/sunrise-build/environment.sh
LOGDIR=$(mktemp -d /tmp/sunrise032-graph.XXXXXXXX)
set +e
TANG_VISIBLE_DEVICES="${RESERVED_PAIR:?请设置已获授权的健康直连卡对}" TA_CLOSE_GRAPH_MEMORY=1 \
GEMS_VENDOR=sunrise VLLM_PLUGINS=fl VLLM_USE_V2_MODEL_RUNNER=1 PCCL_PROTO=Simple \
timeout --kill-after=5s 120s \
  python -u tools/repro/sunrise_032/repro_tp2_graph.py \
    --visible-devices "$RESERVED_PAIR" --ta-close-graph-memory 1 \
  > "$LOGDIR/graph.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$LOGDIR/exit.status"
```

不要自动循环重试。外层 timeout 只能约束主机进程运行时间，不能保证设备上下文已回收。
若复现 unset 对照，需在该进程上 `env -u TA_CLOSE_GRAPH_MEMORY` 并传
`--ta-close-graph-memory unset`；不要把两种设置的日志混用。

历史 TA=1 日志显示两 rank 完成捕获和全部 host enqueue，停在
`final_synchronize_begin`，没有 `final_synchronize_end`，外层 120 秒 timeout 返回 124。
该日志没有打印具体 kernel watchdog 名，不能补称 AllReduce/AllGather 内核 watchdog。
unset 的扩展/fence 日志分别保留了实际 AllReduce/AllGather watchdog 字样。

## C. 对照完整原生 V2 模型路径

[repro_v2_model_graph.py](repro_v2_model_graph.py) 使用本地 Qwen3-4B，TP2、V2、原生异步调度、
compile mode=NONE/backend=eager、FULL_DECODE_ONLY、capture sizes `[1,2]`，
context/batch512、两序列、每 worker 1 GiB KV，按 `[2,1,2]` 三批执行 5 个语义请求。
模型不存在时在导入 GPU 前失败，禁止自动下载。

```bash
: "${MODEL_4B:?请设置本地 Qwen3-4B 路径}"
LOGDIR=$(mktemp -d /tmp/sunrise032-v2.XXXXXXXX)
set +e
TANG_VISIBLE_DEVICES="${RESERVED_PAIR:?请设置已获授权的健康直连卡对}" \
GEMS_VENDOR=sunrise VLLM_PLUGINS=fl VLLM_USE_V2_MODEL_RUNNER=1 PCCL_PROTO=Simple \
env -u TA_CLOSE_GRAPH_MEMORY \
timeout --kill-after=5s 600s \
  python -u tools/repro/sunrise_032/repro_v2_model_graph.py \
    --model "$MODEL_4B" --visible-devices "$RESERVED_PAIR" \
    --ta-close-graph-memory unset --output-dir "$LOGDIR" \
  > "$LOGDIR/model.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$LOGDIR/exit.status"
```

也必须在自己的 tmux 会话中运行。这个整理版移除了历史脚本的 replay 计数代理，
只读取实际 runner、capture mode/type/count；`request_replay_counted=false`。
语义输出正确和捕获成功不能单独当作请求期 replay 证明，更不能替代两个大模型的完整 graph Gate。
历史模型日志保留两 rank 的 V2 标记、FULL_DECODE_ONLY/PTPUGraph 捕获信息，
随后出现 `pcclAllGatherRingKernel_copy_i8` timeout 和 EngineDeadError。

## 关键证据与边界

| 文件 | 原始观测 |
| --- | --- |
| [27b-image-r1.json](logs/27b-image-r1.json) | 原始 26 请求及 5 个 empty/length128 失败摘录 |
| [27b-controls-r2.json](logs/27b-controls-r2.json) | 短/长文本控制与 8 个越界 ID，独立于 R1 |
| [tp2-model-graph-allgather.log](logs/tp2-model-graph-allgather.log) | V2、捕获和实际模型 AllGather watchdog |
| [tp2-extended-allreduce-unset.log](logs/tp2-extended-allreduce-unset.log) | 无逐步同步的多层 collective 探针 AllReduce watchdog |
| [tp2-fresh-uuid-fence-allgather-unset.log](logs/tp2-fresh-uuid-fence-allgather-unset.log) | 唯一 rendezvous、stream-fence 变体仍出现 AllGather watchdog |
| [tp2-ta1-final-sync-timeout.log](logs/tp2-ta1-final-sync-timeout.log) | 显式 TA=1，停在最终同步，host exit124 |
| [device-snapshot-1907.json](logs/device-snapshot-1907.json) | 19:07 历史设备快照，不是当前占用证明 |
| [35b-eager-control.json](logs/35b-eager-control.json) | 35B eager 26/26+4118-token 通过对照 |

[provenance.json](provenance.json) 记录原始文件 basename/SHA256、1-based 行号、JSON 字段和脱敏项。
行号按 UTF-8 解码后的 `splitlines()` 计算，包含 CR 进度刷新行；可能与只计 LF 的工具不同。
摘录已去除凭据、SSH 主机/IP、私有绝对路径及无关环境，原始本地证据保持原样。
包内文件校验见 [SHA256SUMS](SHA256SUMS)。

R3/R4 后续诊断曾见视觉特征 NaN/巨大值及 rank 差异，但当时已有未回收的互联卡状态。
peer DMA 污染只是假设；在清洁、独占的硬件上复验之前，不把这些观测认定为 GEMM、
SDPA、布局或通信实现的生产根因。当前材料也没有证明两个问题一定同因。

## 为什么曾建议 reset，以及如何避免影响别人

建议 reset 的目的是为后续测试取得可确认的清洁设备状态：图超时后测试进程已退出，
1/3/5/6 在历史快照中仍各显示约 1.535 GiB、8% 利用率，进程表没有对应 PID。
这只说明资源/设备状态未能确认回收，**不能证明 reset 是必要或足以修复问题**。
它与“修复模型/插件错误”是不同待验证事项。

[已采集的厂商工具帮助](logs/reset-tool-help.txt)只描述硬件状态 reset，并未保证互联 peer 的隔离边界；
因此当前用户选择“不 reset”，已停止 GPU 验证，没有执行 reset、重启节点或卸载驱动。
复现包不包含硬件恢复命令或自动恢复脚本。

曦望侧可先离线审阅，确认是否有只释放已退出测试上下文的办法。确需硬件恢复时，
由资源管理员协调受影响的互联范围和其他作业，选择维护窗口后单独处理；
不能因为某张卡进程表为空就直接重置，也不要停止不属于自己的服务。

## 原始整理包验证（81aea8d，CPU only）

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python -m pytest --noconftest -p no:cacheprovider \
  tools/repro/sunrise_032/test_repro_27b_image.py \
  tools/repro/sunrise_032/test_repro_tp2_graph.py
python -S tools/repro/sunrise_032/repro_27b_image.py --help
python -S tools/repro/sunrise_032/repro_tp2_graph.py --help
python -S tools/repro/sunrise_032/repro_v2_model_graph.py --help
```

本次本地 CPU 测试为 [53 项通过](validation.json)；Python 3.10 仅做语法兼容检查，未在 GPU 环境重跑整理版。
这些检查覆盖参数、原请求构造、输出保存、显式设备选择和 GPU 导入之前的失败边界。
上述 53 项仅描述原始三类入口的离线整理，没有为它们新增 HTTP 请求、模型推理或 GPU 复测。新增纯 Torch 拷贝检查有独立实测与来源记录，见[当前环境复现](current_environment/README.zh-CN.md)。