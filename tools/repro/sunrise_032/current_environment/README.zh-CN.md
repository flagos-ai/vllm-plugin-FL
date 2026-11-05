# 当前环境的纯copy复现与状态

这是继此前模型/TP2 graph失败后，在**不reset**条件下执行的新一轮环境检查。纯copy结果已阻止继续把0、2号卡视为健康卡；完整TP2 eager/graph模型gate仍然**blocked / 未完成**，不宣称适配问题已修复。此前旧证据保持冻结。

本目录的[脚本](repro_tensor_roundtrip.py)与实际执行源逐字一致，SHA256为`c9c96bf53575811c0bdadaf8dc64e9dfe1ed602c228bd558f43af23320738d0f`，三张卡的summary均记录同一值。执行命令见[已执行命令记录](logs/executed-commands.txt)，其中仅宿主任务路径和任务容器名作明确占位替换；实际使用物理mask分别为0、2、4，各启动一个fresh child，逻辑设备均为`ptpu:0`。这些是历史调用记录，不代表现在建议再次使用这些卡。

## 从此PR运行

以下供已确认资源窗口的排查人员手动使用。当前失败卡不应直接继续模型/graph测试；脚本不会自动选卡、重试或reset。保留原执行版本的参数边界，只接受单卡0、2、4，且环境mask必须与参数一致。

在目标厂商环境和仓库中，先进入独立tmux会话，再设置获分配的单卡与一个**不存在**的输出目录：

```bash
tmux new-session -s sunrise-roundtrip-UNIQUE
source /opt/sunrise-build/environment.sh
# 在此tmux中设置 RESERVED_CARD 和 OUT_DIR；不要将历史失败卡当作默认值。
: "${RESERVED_CARD:?填入已分配的单卡编号}"
: "${OUT_DIR:?填入父目录存在、当前尚不存在的结果目录}"
TANG_VISIBLE_DEVICES="$RESERVED_CARD" \
timeout --signal=TERM --kill-after=5s 65s \
  python -u tools/repro/sunrise_032/current_environment/repro_tensor_roundtrip.py \
    --physical-device "$RESERVED_CARD" --output-dir "$OUT_DIR" \
    --timeout-seconds 45
result=$?
printf 'exit_status=%s\n' "$result"
```

该调用只生成`roundtrip.jsonl`与`summary.json`。CLI0才代表完整性检查通过；CLI1可能表示数值失败，应读summary区分，不能把worker exit0当数值通过。host timeout只约束本任务进程树，不保证设备上下文回收；发生timeout/native crash后停止后续检查并保留日志。

## 已执行结果

| 物理卡 | 全量比较 | input比较 | weight比较 | 子进程/超时 | CPU源 |
|---|---:|---|---|---|---|
| 0 | 4精确 / 5失败 | 5次失败 | 4次精确 | worker exit0 / CLI1 / 无超时 | input、weight均未变 |
| 2 | 4精确 / 5失败 | 5次失败 | 4次精确 | worker exit0 / CLI1 / 无超时 | input、weight均未变 |
| 4 | 9精确 / 0失败 | 5次精确 | 4次精确 | worker exit0 / CLI0 / 无超时 | input、weight均未变 |

每次input比较覆盖`4096×2152 = 8,814,592`个BF16值，每次weight比较覆盖`1152×2152 = 2,479,104`个值。先比较input，再分配/copy weight并比较两者，随后3次hold各等待0.1秒并再次比较两者。hold循环未请求新的设备写入/分配，但D2H读回、同步和底层运行时内部行为仍存在。本检查没有GEMM、GPU转置、通信、graph或reset；未调用FlagGems enable。

- 0卡：weight分配前input错5,136个；分配后及3次hold各错10,256个。
- 2卡：weight分配前、分配后及前2次hold的input各错10,256个，最后一次hold错10,246个。
- 两卡input失败记录的首坏地址均为`0xfff3bd3c8`。这是由Tensor `data_ptr` 加行列 stride × element_size 计算的（`data_ptr` 已包含 `storage_offset`，不重复添加）**GPU/runtime虚拟地址**，不是physical DRAM地址或已证实的坏内存位置。4卡相同分配布局的这9次比较精确，只是这一轮单卡对照通过，不能外推全卡、通信或模型gate健康。

原始逐行receipt和summary见[卡0](logs/copy-card0-r1/roundtrip.jsonl)、[卡2](logs/copy-card2-r1/roundtrip.jsonl)、[卡4](logs/copy-card4-r1/roundtrip.jsonl)。`summary.exitcode=0`仅代表子进程正常退出；数值通过还需要`complete=true`、`timed_out=false`、`worker_alive=false`、9条comparison全`exact_values=true`。0/2两卡并未数值通过。独立下载的外层`timeout/docker exec`状态记录为0卡1、2卡1、4卡0，与脚本对数值失败返回1的合同一致；三个worker自身均正常exit0，无timeout。

[设备状态](logs/device-state.json)来自每卡紧邻运行前后共6份原始快照，以及最新`2026-11-04T23:59:35+08:00`全卡快照。三卡before/after的全8卡必要字段与最新表一致：0、2、4仍显示0B/0%，1、3、5、6仍各1.535GiB/8%，7保留旧12KiB/8%，ECC均0，无列出GPU进程；这些状态表中未见新增残留。为避免重复公开7份宽日志，仅保存逐卡前后必要字段和最新全卡精简JSON，原始SHA及行范围仍逐份记录。0B、ECC0及无PID都不能证明独占、无peer污染或设备健康，尤其0/2已经出现本轮纯copy错误。

此前本轮小常量copy和原生PG AllReduce的FP32/BF16观测通过，而一次PyFlagCX AR/AG尝试失败；随后独立的原位先行AR控制中，nativePG、FlagCX原位、FlagCX非原位在两dtype、两rank共12条检查均通过。后一个控制没有重测AG，不能据早先一次失败把根因限定为FlagCX、BF16或通信；这些矛盾的具体路径观测也不能抵消本次大Tensor全量copy错误。另有16条Tang DEVICE_POINTER查询均成功且raw=resolved，偏移视图也保留预期偏移；没有支持转换指针的证据。通信原始宽日志不重复发布，来源SHA及数值记录行范围仅在provenance中保留。更早`numerical-r1`把未被case使用的`wtc`回拷结果AND进`passed`，其失败计数不能直接引用；部分后续linear诊断仅比较选定行，不能与这里的全量exact比较混同。

当前根因仍未确定：地址相关性、peer写入、映射/allocator、设备/驱动/运行时问题均未获因果证实。保留失败证据，不执行reset，也不把“可能需要恢复硬件状态”写成已证明的修复办法。

[provenance.json](provenance.json)记录明确选择的来源、原始SHA256、UTF-8 `splitlines()`的1-based行范围、转换及脱敏；[SHA256SUMS](SHA256SUMS)覆盖本证据包文件；[离线验证记录](offline-validation.json)记录原有53项与新增9项共62项通过、Ruff及Python 3.10语法检查。保留GPU虚拟地址；CPUhost指针、宿主私有路径、SSH/IP、无关环境不进入发布内容。原始本地文件未改写。
