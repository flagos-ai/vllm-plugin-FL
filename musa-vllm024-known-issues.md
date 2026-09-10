# MUSA (S5000) vLLM 0.24 适配问题调查报告

- 日期：2026-09-09 ～ 2026-09-10
- 环境：MTT S5000 ×8（80GB），mthreads-gmi 2.3.2 / driver 3.3.5-server，Ubuntu 22.04；
  容器 `harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.24.0-musa-ci`
  （vLLM 0.24.0+empty / torch 2.9.0 / torch_musa 2.9.0 / FlagGems 5.3.2.post1.dev22+gb1f939eb5 / flagtree 0.6.0+mthreads3.6）
- 插件侧：flagos-ai/vllm-plugin-FL，PR #457（本报告随该 PR 交付，报告本身不含任何运行时 py 改动）
- 结论速览：**五个故障模式，全部指向 torch_musa / 驱动层不稳定；vLLM 上游 1 处（fork 探测）放大了后果；插件层无法修复，只能绕过**。建议摩尔在重启宿主机、复位驱动后按 §5 的复现矩阵逐项定位。

---

## 1. 问题清单与时间线

| # | 故障 | 首次观测 | 复现性 |
|---|---|---|---|
| ① | TP≥2 权重加载挂死（rank≥1 用户态自旋） | 09-09 | 09-09 必现；09-10 不再复现（见 §6 状态漂移） |
| ② | `vllm serve` 启动失败：fork 冲突 | 09-09 | 稳定复现 |
| ③ | 图模式（torch.compile + PIECEWISE cudagraph）挂死 | 09-10 CI | 干净 runner 必现（58 分钟零输出） |
| ④ | TP2 `profile_run` dummy 前向 **illegal memory access** | 09-10 | 当日两次复现，且每次崩溃后泄漏 ~70GB 显存 |
| ⑤ | init 期 `can_device_access_peer` 断言（custom all-reduce p2p 探测） | 09-10 | 状态依赖；vendor 用例 yaml 一律 `disable_custom_all_reduce: true` 可绕 |

共通点：**TP1（单卡、in-process、eager）始终健康**——同一容器、同一模型、同一天内 TP1-OK 与 TP2 崩溃并存。

## 2. 问题①：TP≥2 权重加载挂死

**现象**：`tensor_parallel_size>=2` 时 rank0 正常加载（秒级），rank≥1 在权重 H2D 拷贝处永久自旋（用户态 R 状态，非锁等待）。显存停在 ~293MB。

**证据**：
- py-spy：rank≥1 主线程自旋在原生 `aten::copy_ → _copy_from` 路径；
- dispatch dump（引擎进程内）：`aten::copy_` 在 `PrivateUse1=False`、`CompositeExplicitAutograd=True` 且注册来源为 PyTorch 原生 `RegisterCompositeExplicitAutograd_0.cpp:3332` —— **此栈上 flag_gems 未注册任何 copy_ kernel，权重加载全程走原生实现**；
- 单进程复现（set_device + pin_memory + non_blocking + mmap 源 + 大/小张量）：全部通过。

**已排除**：triton 缓存并发锁（per-rank `TRITON_CACHE_DIR` 无效）、flag_gems copy 拦截（未注册，无从绕过）、`_FALLBACK_KEYSET` 路由（前提不成立）、fork（spawn 模式下发生）。

## 3. 问题②：`vllm serve` 启动失败（fork 冲突）

**因果链（两端代码均可公开核对）**：
1. vLLM 0.24.0 上游 `vllm/utils/platform_utils.py:37-39` 的设备属性探测**硬编码 `multiprocessing.get_context("fork")`** 子进程；
2. torch_musa `torch_musa/core/_lazy_init.py:107` 在已初始化 MUSA 的进程 fork 出的子进程中直接抛出
   `RuntimeError: Cannot re-initialize MUSA in forked subprocess`；
3. `vllm serve`（API server → EngineCore）路径必经该探测 → 启动即死。in-process LLM 不受影响（MUSA 已初始化走直查分支）。

**非致命形态**：EngineCore 的 usage 上报线程也会触发同一 fork 子进程崩溃（CI 日志 2026-09-10 08:29:04、08:30:12 两次），但只杀死上报线程，不影响推理——eager 变体带着它 PASSED。

**归属**：上游 vLLM 一半（fork 探测应可配置/可降级），torch_musa 一半（fork 禁令是厂商自我声明的约束）。

## 4. 问题③：图模式捕获挂死（CI 实录）

CI（run 34453794059，干净 runner、冷缓存）：`enforce_eager=True` 变体 15 秒出正确结果；`enforce_eager=False`（torch.compile + PIECEWISE，capture sizes 至 512）引擎初始化完成后 **08:30:12 → 09:27:46 零日志输出**，被 60 分钟 job 超时杀掉。慢编译不可能 58 分钟一行不吐，定性挂死。所有摩尔官方 musa 用例均带 `_eager` 后缀与此一致。

## 5. 问题④/⑤：当日新捕获（宿主机状态劣化背景下）

09-10 下午起，宿主机（横幅提示 `*** System restart required ***`）出现行为漂移：

- TP2 init 在 custom all-reduce p2p 探测处断言 `can_device_access_peer → AssertionError: Invalid device id`（问题⑤）；加 `disable_custom_all_reduce=True` 后推进到 `profile_run`；
- **TP2 在 `Worker_TP1` 的 `_dummy_run` 前向中触发 `InductorError: RuntimeError: MUSA error: an illegal memory access was encountered`**（问题④，两次复现：日志 `tp2_nocar.log`、`tp2_tiny.log`）；
- **每次 IMA 崩溃后 device 0 泄漏 ~70GB 显存并残留 ~36-40% 幽灵算力占用；`kill -9` 全部持有进程后不回收**（gmi：`0 | 40% | 71228MiB` → 杀进程后 `36% | 71081MiB`），仅驱动复位/重启可清；
- TP1 全程健康（`TP1-OK`，含低显存配额复验）。

## 6. 最小复现矩阵（关键阴性结果）

为剥离 vllm 因素，编写了无 vllm 的多进程复现矩阵（`scripts_debug/repro_musa_copy_hang*.py`，容器内 `/root/scripts_debug/`）：

| 配置 | 内容 | 结果 |
|---|---|---|
| A | 双进程 spawn，各自 pinned + non_blocking H2D 到自己的卡 | PASS 7s |
| B | A + mccl init（world=2） | PASS 66s |
| C | B 但双 rank 同卡（对照） | PASS 9s |
| D | mccl + 无 pin + 阻塞拷贝 | PASS 8s |
| E | B + `import flag_gems`（kernel 注册 + backend 激活） | PASS 12s |
| F | mccl + safetensors mmap 源、256 个 4MB 张量循环拷贝（模拟权重加载） | PASS 13s |

**全部通过** → 挂死/IMA 需要完整 vLLM TP 上下文才触发，纯 torch_musa 多进程拷贝本身不挂。结合 §5 的驱动泄漏与幽灵占用，提出工作假设（供摩尔验证）：

> **假设**：TP≥2 路径中的某次内核崩溃/非法访问会污染驱动级设备状态；被污染的设备让后续进程表现为各种不一致的故障（权重拷贝自旋、p2p 断言、IMA），且崩溃本身又扩大泄漏——这能解释 09-09（挂死）与 09-10（IMA/p2p）的表现漂移。宿主机自 09-09 起持续提示需要重启，未重启。

## 7. 关于"让 vLLM 侧加算子绕过"的可行性（回应评审问题）

本仓库存在 sglang 式的算子黑名单机制且被各平台广泛使用（`dispatch/config/*.yaml` 的 `flagos_blacklist`/`oot_blacklist`，env `VLLM_FL_FLAGOS_BLACKLIST_APPEND`；nvidia.yaml 甚至拉黑了 `copy_`/`to_copy`）。但对本报告的问题：

- **①权重加载挂死**：`copy_` 不在 dispatch 系统管理的 5 类算子内，且此栈 flag_gems 根本没注册它——**没有可拉黑的对象**。理论方案是在插件里为 musa 注册一个替代 `copy_` kernel（`torch.library` PrivateUse1 或 vendor patch），绕开原生路径；工程上要覆盖 dtype cast/broadcast/non-contiguous/async pinned H2D 全部语义，且 §5 表明挂死可能与驱动状态污染相关——那种情况下注册什么算子都绕不开。**不建议在摩尔定性前做**。
- **②fork 冲突**：**可以且建议由 vLLM 上游修**——`platform_utils` 的 fork 探测对"禁止 fork 的后端"应退化为进程内直查（一个 try/except + 配置位）。插件层也可临时 patch 该函数（vendor patch 有先例），待上游接受。
- **③图模式**：绕过即 `enforce_eager=True`（已采用，属正当配置而非吞问题——eager 是 vendor 官方形态）。
- **④IMA / ⑤p2p**：内核访存与驱动探测层，算子路由不可达。⑤可用 `disable_custom_all_reduce=true`（vendor 自身约定）。

## 8. CI 现状与恢复清单

CI（PR #457，run 34453794059）：Setup/unit/functional 全绿；e2e eager 变体绿。当前 e2e 形态 = `qwen3/06b_tp1`（TP1、eager），serving 与 benchmark 暂关（②）。**每项收缩都在 `tests/platforms/musa.yaml` 注释与本报告中有对应故障编号**，不是无痕跳过。恢复条件：

1. ②修复（上游或 torch_musa）→ serving e2e + benchmark 恢复；
2. ①④⑤定性修复 → 恢复 27b/35b TP4 用例（vendor 用例本就存在）；
3. ③修复 → 用例 `parametrize` 恢复 `[true, false]`。

## 9. 附件索引

- 容器 `musa-024-e2e` 内：`/root/tp1_inf.log`、`/root/tp1_srv.log`（09-09 TP1 通过记录）；`/root/tp2_forensic*.log`、`/root/tp2_nocar.log`、`/root/tp2_tiny.log`（09-10 IMA/p2p 完整栈）；`/root/pyspy_tp0.txt`、`/root/pyspy_tp1.txt`；复现脚本 `/root/scripts_debug/`
- CI：run 34453794059（e2e 60 分钟超时实录，含 58 分钟静默段）
- 复现矩阵：仓库 `scripts_debug/repro_musa_copy_hang.py`（v1，A–D）、`repro_musa_copy_hang_v2.py`（E–F）
