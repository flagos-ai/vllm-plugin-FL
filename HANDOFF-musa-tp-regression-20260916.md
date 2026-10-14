# 交接文档：MUSA TP 回归定位（bisect 进行中）— 2026-09-16

> 未跟踪文件，勿提交。配套背景：`HANDOFF-enflame-slowness-20260908.md`（燧原线）。

## 0. 核心结论（本会话最重要）

1. **TP≥2 失败的根因在我们自己的插件代码里**（R22/R23 受控对照：同一镜像同一命令，只把插件从邓伟镜像内置的 0.3.0 换成本分支代码，27B TP4 从 PASS 翻 FAIL）。用户的 owner 也提示过此方向。
2. **报告（musa-vllm024-known-issues.md，PR #457，已推到 v3.5@618099e）不要再为这件事改**——用户明确：报告只用于"我们解决不了、向上反馈"；TP 回归是我们职责内要修的。报告摘要表 Owner 列仍写着"摩尔"，接受现状不改。
3. ③（图模式停滞）确实是环境层：FlagTree main 未修（当日 main@1b3a7b850 实测停滞），**邓伟镜像（08281629，flagtree 0.6.0+gitb97f8214）已修但未上游化**。性能问题定位：decode 快 4 倍（56.28 vs 14.33 tok/s），慢在 musa 图 capture ~26 分钟（py-spy：torch_musa/musa_graph/graphs.py:201）。
4. **已验证可用组合**（对外交付/演示可直接用）：`harbor.baai.ac.cn/flagrelease-public/flagrelease_mthreads-gmi_vllm024plugin_base:08281629` 原样使用——0.6B eager/graph/serve、27B TP4、35B-A3B TP4 全 PASS。硬条件：不要换成我们的插件。

## 1. bisect 现状与未解的混杂变量

已排除（正向单独替换均不能救回，R24-R26）：`worker.py`（9e6be44 per-rank TRITON_CACHE）、`dispatch/config/musa.yaml`、`worker/model_runner.py`。

**⚠️ 但 R29（把 vllm_fl 整体换成 0.3.0 副本经 bind mount 加载）也失败了**——这与 R22（镜像自带 0.3.0 = PASS）矛盾，说明二选一：
- (a) **bind mount 实验方式本身有问题**（残留 pyc / 挂载加载路径差异），R23-R26 的结论全部存疑；或
- (b) **R23 的设备级崩溃污染了宿主机状态**（"MUSA error: unknown error" 是设备错误；①族本就状态依赖），之后的失败测的是坏状态不是代码。

**消歧实验 R30 已发出未收结果**：在未动过的 dw-0828 容器里重跑 R22 同款（detached，timeout 900，日志 `/root/ctrl_r30.log`）。判读：
- R30 **PASS** → 状态没坏 → (a) 成立 → 弃用 bind mount，改用 **docker cp 覆盖容器内 site-packages/vllm_fl** 的方式重做 R23 确认 FAIL，再开始 bisect；
- R30 **FAIL** → 状态污染 → 清理残留（pkill -9 -f 'VLLM::'，查 R 态存活 worker 逐一补刀，gmi 确认 0MiB；必要时需用户同意重启宿主机）→ 重跑 R22 对照确认恢复 → 再重做 R23。

## 2. 用户指定的 bisect 方案

- good 端点：tag **`v0.3.0-rc2.post1`**（已 fetch 到宿主机仓库，共 14 个 tag）；bad 端点：`9e6be44`（verify24 头）。
- 先验证 `git merge-base --is-ancestor v0.3.0-rc2.post1 9e6be44`（上次的检查被取消未出结果）；再用 `git bisect run` 自动化：步进脚本 = checkout 提交 →（按 §1 结论选 docker cp 或 mount 方式）同步代码进 dw-ourplug → `timeout 900 python3 /root/mat_test.py 4 /data/models/Qwen/Qwen3.6-27B 1` → exit 0=good。
- 注意 zsh/引号坑：模型路径是 `/data/models/Qwen/Qwen3.6-27B`（**两次**打漏 `Qwen/` 白跑）。

## 3. 机器/容器状态（摩尔机）

- 访问：本机 tmux `musa` 会话（跳板交互认证；断线需用户重连输密码）。
- 容器：`musa-024-e2e`（我们的，flagtree 已装 0.7.0，快照 `musa-024-e2e:pre-flagtree-main` 可回滚）；`dw-0828`（邓伟镜像原样）；`dw-ourplug`（邓伟镜像 + `-v /root/vllm-plugin-FL/vllm_fl:/usr/local/lib/python3.10/dist-packages/vllm_fl`）。
- **⚠️ 宿主机 `/root/vllm-plugin-FL` 工作区当前被改坏**：`vllm_fl/` 目录内容是 0.3.0 副本（我 rm+cp 的）。恢复我们的版本：`rm -rf vllm_fl && cp -r /root/vllm_fl_ours vllm_fl`（备份就在那）。另原有的 tests/platforms/musa.yaml 本地收缩改动和 .mudmp 未跟踪文件是 09-09 前就有的。
- 参考副本：`/root/dw_vllm_fl`（0.3.0 树）、`/root/vllm_fl_ours`（verify24 树）。
- 测试日志：dw-ourplug `/root/mat_tp4*.log /root/bisect_*.log`；dw-0828 `/root/mat_tp4.log /root/serve06b_r27.log /root/mat35b_r28.log /root/ctrl_r30.log`。
- 脚本：容器内 `/root/mat_test.py`（tp, model, eager, nocar 四参）；本地 /tmp 有全套（mat_test/graph_test/perf_test/serve_test/assert_ab）。

## 4. 报告与 PR 状态

- musa 报告 v3.5 已推（`618099e`）；R24-R28 已入矩阵；**本地无未推送改动**（一次 v3.6 编辑尝试因断言失败未写入）。
- PR #457 代码头 c744c04 未变；**TP 回归修复前不宜合并**（会话内已向用户言明）。
- PR #464（enflame）CI 全绿等 owner review；分支多了 `e7c7fdc detour GCU top-p filtering through CPU`（他人对慢速问题的修复）——燧原黑名单实验是否还做待定。

## 5. 下一步清单（按序）

1. 收 R30 结果 → 按 §1 分支消歧。
2. 确认 v0.3.0-rc2.post1 为 good（用它跑一次 TP4 对照）。
3. `git bisect run` 自动化定位元凶提交。
4. 修/回退元凶 → 我们代码 TP4 过 → 恢复 e2e TP 用例（27b_tp4_eager 等）与 `06b_tp1` 的 `[true,false]`（③若用邓伟镜像当 CI 基础镜像才可恢复图模式——基础镜像决策悬而未决，倾向先修代码再评估）。
5. 收尾：恢复宿主机仓库工作区；容器清理（dw-ourplug 可删）。
