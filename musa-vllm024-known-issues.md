# MUSA (S5000) vLLM 0.24 适配问题调查报告

- 日期：2026-09-09 ～ 2026-09-10
- 环境：MTT S5000 ×8（80GB），mthreads-gmi 2.3.2 / driver 3.3.5-server，Ubuntu 22.04；
  容器 `harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.24.0-musa-ci`
  （vLLM 0.24.0+empty / torch 2.9.0 / torch_musa 2.9.0 / FlagGems 5.3.2.post1.dev22+gb1f939eb5 / flagtree 0.6.0+mthreads3.6）
- 插件侧：flagos-ai/vllm-plugin-FL，PR #457（本报告随该 PR 交付，报告本身不含任何运行时 py 改动）
- 结论速览：**五个故障模式，根因均不在插件层；①③④⑤归属摩尔（torch_musa/驱动），②一半归属 vLLM 上游、一半归属摩尔**。插件层无法根修，只能按 §0 的映射显式收缩测试面并等待根修。

---

## 0. 因问题被跳过/收缩的用例清单（与问题编号一一对应）

| 被跳过/收缩的用例或能力 | 原始形态 | 阻塞问题 | 恢复条件 |
|---|---|---|---|
| e2e inference `qwen3_6/27b_tp4_eager`（vendor 原有用例） | TP4 多卡推理 | ① | 摩尔修复 TP≥2 路径后恢复 |
| e2e serving `qwen3_6/35b_a3b_tp4_eager`（vendor 原有用例） | TP4 serving | ①＋② | ②修复可先恢复 serving 框架，①修复后恢复 TP4 |
| e2e serving 任务整体 | `tests/platforms/musa.yaml` serving 段 | ②（`vllm serve` 无法启动） | vLLM 上游或 torch_musa 任一方修复 fork 冲突 |
| benchmark（serve smoke 三件套） | `benchmark.enabled: true` | ②（`vllm bench serve` 同样要起服务） | 同上 |
| `enforce_eager=False` 变体 | `06b_tp1.yaml` parametrize `[true, false]` | ③（图模式捕获挂死） | 摩尔修复捕获路径后恢复 `[true, false]` |
| 多卡矩阵（现仅保留 `qwen3/06b_tp1` 单卡） | 原 TP4 矩阵 | ①（⑤在诊断中亦拦路，可用 `disable_custom_all_reduce=true` 绕） | 同① |

> 设计说明：以上收缩全部是**显式**的——`tests/platforms/musa.yaml` 内注释逐条标注问题编号，PR 描述与本报告互相引用；没有任何无痕跳过。当前保留的测试面（unit 425 例 + functional 25 例 + TP1 eager 推理 e2e）全部真实执行且通过。

## 1. 问题清单与时间线

| # | 故障 | 首次观测 | 复现性 |
|---|---|---|---|
| ① | TP≥2 权重加载挂死（rank≥1 用户态自旋） | 09-09 | 09-09 必现；09-10 表现漂移为④⑤（见 §6） |
| ② | `vllm serve` 启动失败：fork 冲突 | 09-09 | 稳定复现 |
| ③ | 图模式（torch.compile + PIECEWISE cudagraph）捕获挂死 | 09-10 CI | 干净 runner 必现（58 分钟零输出） |
| ④ | TP2 `profile_run` dummy 前向 **illegal memory access** | 09-10 | 当日两次复现，每次崩溃后泄漏 ~70GB 显存 |
| ⑤ | init 期 `can_device_access_peer` 断言（custom all-reduce p2p 探测） | 09-10 | 状态依赖 |

共通点：**TP1（单卡、in-process、eager）始终健康**——同一容器、同一模型、同一天内 TP1-OK 与 TP2 故障并存。

## 2. 问题①：TP≥2 权重加载挂死

- **现象**：`tensor_parallel_size>=2` 时 rank0 数秒完成加载，rank≥1 在权重 H2D 拷贝处永久自旋（用户态 R 状态、非锁等待），显存停在 ~293MB，无任何日志推进。
- **原理（根因假设）**：非纯拷贝 bug——§6 最小复现矩阵证明纯 torch_musa 多进程 H2D 拷贝（含 mccl、flag_gems、mmap 源）全部通过；结合 §5 的"崩溃后驱动状态污染"，假设为 **TP 上下文中某次内核异常污染驱动级设备状态，使后续拷贝在驱动内自旋**。待摩尔在复位后的干净环境复现定位。
- **证据**：py-spy 原生栈（自旋点在原生 `aten::copy_ → _copy_from`）；dispatch dump 证明此栈 flag_gems 未注册任何 `copy_` kernel（`PrivateUse1=False`，CompositeExplicitAutograd 来源为 PyTorch 原生 `RegisterCompositeExplicitAutograd_0.cpp:3332`），故权重加载全程走原生实现。
- **Owner**：摩尔（torch_musa/驱动）。
- **潜在解法**：临时绕过 = 只跑 TP1（已采用）；算子层绕过（为 musa 注册替代 `copy_`）理论上可行但见 §8 的否决理由；根修 = 摩尔按报告附件复现定位驱动路径。

## 3. 问题②：`vllm serve` 启动失败（fork 冲突）

- **现象**：`vllm serve`（任意 TP）启动即死，异常 `RuntimeError: Cannot re-initialize MUSA in forked subprocess`。
- **原理**：vLLM 0.24.0 上游 `vllm/utils/platform_utils.py:37-39` 的设备属性探测**硬编码 `multiprocessing.get_context("fork")`** 子进程；torch_musa `core/_lazy_init.py:107` 在已初始化 MUSA 的进程 fork 出的子进程中禁止再初始化。两个事实均为可直接核对的公开代码/异常栈。
- **非致命形态**：in-process LLM 只在 usage 上报线程触发同一崩溃（CI 日志 08:29:04/08:30:12），仅杀死上报线程、不影响推理。
- **Owner**：vLLM 上游（fork 探测应可配置/可降级）＋ 摩尔（fork 禁令是厂商自我声明约束）。
- **潜在解法**：根修（推荐）= 上游对禁 fork 后端退化为进程内直查（小改动，一个 try/except + 平台开关）；临时 = 插件 vendor patch 该函数（有先例形态，未做）；或 torch_musa 放宽 fork 限制。

## 4. 问题③：图模式捕获挂死

- **现象**：`enforce_eager=False`（torch.compile + PIECEWISE，capture sizes 至 512）引擎初始化完成后**零日志静默**，直到 job 超时。CI run 34453794059：08:30:12 → 09:27:46 无任何输出；同用例 eager 变体 15 秒出正确结果。
- **原理**：冷缓存下 torch.compile/inductor + PIECEWISE cudagraph 捕获在 torch_musa 上挂死（慢编译不可能 58 分钟一行不吐）。所有摩尔官方 musa 用例均带 `_eager` 后缀，与此一致。
- **Owner**：摩尔（torch_musa 编译/捕获路径）。
- **潜在解法**：临时 = eager-only（已采用；属 vendor 官方形态，非吞问题）；根修 = 摩尔排查 capture 路径。

## 5. 问题④/⑤：IMA 崩溃与 p2p 探测断言（09-10，宿主机状态劣化背景下）

**④ TP2 dummy 前向 IMA**
- **现象**：加 `disable_custom_all_reduce=true` 后 TP2 推进到 `profile_run`，`Worker_TP1` 的 `_dummy_run` 前向抛 `InductorError: RuntimeError: MUSA error: an illegal memory access was encountered`（两次复现：`tp2_nocar.log`、`tp2_tiny.log`）。
- **原理**：内核非法访存；且**每次崩溃后 device 0 泄漏 ~70GB 显存并残留 36-40% 幽灵算力，`kill -9` 全部持有进程后不回收**（gmi：`40% | 71228MiB` → 杀进程后 `36% | 71081MiB`），仅驱动复位/重启可清。
- **Owner**：摩尔（驱动/内核）。
- **潜在解法**：临时 = 重启宿主机回收（上报前必须做，见 §6）；根修 = 摩尔定位 IMA 内核与泄漏回收路径。

**⑤ p2p 探测断言**
- **现象**：TP2 init 在 `custom_all_reduce.py:_can_p2p → torch.cuda.can_device_access_peer` 抛 `AssertionError: Invalid device id`。
- **原理**：状态依赖（09-09 同环境可通过此探测）；与摩尔官方用例一律 `disable_custom_all_reduce: true` 相互印证——vendor 已知该路径不可靠。
- **Owner**：摩尔。
- **潜在解法**：临时 = `disable_custom_all_reduce=true`（vendor 约定）；根修 = 摩尔修正 peer 探测。

## 6. 最小复现矩阵（关键阴性结果）与状态漂移

无 vllm 的多进程复现矩阵（`scripts_debug/repro_musa_copy_hang*.py`，容器内 `/root/scripts_debug/`）：

| 配置 | 内容 | 结果 |
|---|---|---|
| A | 双进程 spawn，各自 pinned + non_blocking H2D 到自己的卡 | PASS 7s |
| B | A + mccl init（world=2） | PASS 66s |
| C | B 但双 rank 同卡（对照） | PASS 9s |
| D | mccl + 无 pin + 阻塞拷贝 | PASS 8s |
| E | B + `import flag_gems`（kernel 注册 + backend 激活） | PASS 12s |
| F | mccl + safetensors mmap 源、256 个 4MB 张量循环拷贝（模拟权重加载） | PASS 13s |

**全部通过** → 纯 torch_musa 多进程拷贝本身不挂，故障需要完整 vLLM TP 上下文。

**状态漂移记录**：宿主机横幅自 09-09 起提示 `*** System restart required ***`（未重启）。09-09 TP≥2 表现为①挂死；09-10 同容器同代码漂移为⑤断言→④IMA，且④的泄漏累积后使同宿主机的 CI unit 阶段从健康基线 77 秒劣化到 1.5 小时+。**上报前建议：重启宿主机、复位驱动，在干净状态按 A–F 矩阵 + vLLM TP2 顺序复测**。

## 7. 为什么"算子黑名单/注册替代算子"对本报告的问题不适用

本仓库存在 sglang 式算子黑名单机制且被广泛使用（`dispatch/config/*.yaml` 的 `flagos_blacklist`/`oot_blacklist`，env `VLLM_FL_FLAGOS_BLACKLIST_APPEND`；nvidia.yaml 甚至拉黑过 `copy_`/`to_copy`）。但：

- dispatch 系统只管理 5 类算子（attention/rms_norm/rotary/silu_and_mul/topk_softmax），**权重加载的 `copy_` 不在其中**，且此栈 flag_gems 根本没注册它——没有可拉黑的对象；②是进程模型问题、③是编译捕获问题、④⑤在内核/驱动层——均不可通过算子路由绕过。
- 理论方案"为 musa 注册替代 `copy_` kernel"（`torch.library` PrivateUse1）需覆盖 dtype cast/broadcast/non-contiguous/async pinned H2D 全部语义，且若根因是驱动状态污染（§6 假设）则注册什么都会被污染。**不建议在摩尔定性前做**。

## 8. CI 现状

CI（PR #457 最新 run，head bc2125f）：**全绿**——Setup / unit / functional / e2e（`qwen3/06b_tp1`，eager，2m34s）/ benchmark（配置为空、空跑通过）/ build / lint / CodeQL。该绿色代表"收缩后测试面无问题"，五个故障仍然开放（见 §0 恢复条件）。

## 9. 附件索引

- 容器 `musa-024-e2e`（已 stop，证据已备份宿主机 `/root/fl-musa-evidence/`）：`tp1_inf.log`、`tp1_srv.log`（09-09 TP1 通过记录）；`tp2_forensic*.log`、`tp2_nocar.log`、`tp2_tiny.log`（09-10 IMA/p2p 完整栈）；`pyspy_tp0.txt`、`pyspy_tp1.txt`；复现脚本 `/root/scripts_debug/`
- CI：run 34453794059（图模式挂死实录，含 58 分钟静默段）；run 34477790128（unit 劣化至 1.5h+ 的对照记录）
- 复现矩阵：仓库 `scripts_debug/repro_musa_copy_hang.py`（A–D）、`repro_musa_copy_hang_v2.py`（E–F）
