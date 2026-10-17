# 交接文档：ci/workflow-opt → PR #581 及周边（2026-10-14）

> **2026-10-14 深夜摘门验证战役终态（用户令"去掉label门控完整跑+12h内修掉"）：**
> 两轮全量验证（首轮 8+3 平台、复验轮定向），产出 4 个问题、3 修 1 裁定，全部闭环：
> ①thead 27b_tp4_graph 足迹 OOM（inference+serving 同根因）→ 撤用例定格 6 场景，复验 thead 全链绿；
> ②ascend HCCL 16666 端口竞争（并行调度器首次遇 NPU）→ 按槽位注入独立端口段，复验 ascend 全链绿；
> ③040 线 nvidia 0.28 镜像缺 pytest-xdist（-n 16 报错）→ setup.sh 补装，复验 functional 绿；
> ④h20 e2e 团灭（#592/0.28 镜像）→ **裁定为 0.28 运行时×4 路并发引擎问题**（证据链：失败窗口 nvidia-smi
> 全卡 0MiB 干净节点击杀残留假设；v0.24 同并发+同注入在 h20 双绿）——**待用户三选一：040 降并发/
> 报 0.28 owner（诊断材料齐）/已知红标注**。注入对 CUDA 栈惰性经 h20+a100 双点证明。
> 另：enflame serving 两轮均被外部取消（10:18/10:33 首轮、复验轮再次，无失败签名、非新 run 触发），
> **该 job 无本轮实证**，是否第三轮补跑待用户示下。
> **两侧终态：#581 @ c4e6ddd、#592 @ f159670（全部修复已 amend 进交付提交），门禁已恢复，
> 恢复后 CI 双双 completed/success（38598581398 / 38591577942）。TEMP 提交已全部 reset 掉。**

> 2026-10-14 晚新增三大块：①8 场景矩阵+并发任务；②0.4.0-dev 整套迁移；③**test-cuda→test-nvidia 改名**
> （用户指出 cuda 是六家共享的 device_type——VENDOR_DEVICE_MAP 里 iluvatar/hygon/metax/kunlunxin/thead/biren
> 全是 device_type cuda——CI 按厂商名区分，NVIDIA 流水线改用厂商名；波及 registry 键、三 workflow job+gate、
> tests/platforms+configs+scripts 目录改名、cleanup.py 平台键映射；vendor/cuda 后端目录与 device_type 语义不动；
> 顺带修掉主线 a100 nightly 引用不存在的 next_tp4→next_tp8 悬空引用）。注意：改名后已发评论里的"仅 cuda"
> 路由表述已过时（现为"仅 nvidia"），推送时已同步更新。

> 2026-10-14 晚新增两大块（用户 advice：pr 档全覆盖 qwen3.6 8 场景 + nightly 并发/benchmark + 整套迁移 0.4.0-dev）：
> ①**主分支侧（未推送，本地 fd600e1）**：9 平台 yaml 铺 8 场景矩阵（双模式平台 pr/nightly 各 inference+serving×{27b,35b}×{eager,graph}=8；enflame/iluvatar eager 4；nightly 增 concurrent 任务 2 例）；新增 concurrent e2e 任务全链路（tests/e2e_tests/concurrent/test_concurrent_smoke.py 移植自 sglang、ModelConfig.concurrent 块带默认、run.py/generate_matrix 任务映射+超时）；验证全绿（9 平台×2 档矩阵+用例文件存在性+ModelConfig 单测+ruff）。
> ②**0.4.0-dev 移植（本地分支 ci/workflow-opt-040 @ 4a25332，未推送）**：worktree ../vllm-plugin-FL-040，cherry-pick 重放全部 17 提交+适配提交：registry 启用按本线（cuda/ascend/iluvatar 开，余关，thead 无 runner）；a100 用本线 next_tp8；musa qwen3 0.6b 缺件删除；**iluvatar 本线铺 qwen2_5 7b（非 qwen3_6）**；**ascend 本线 eager-only→4 场景**；**run.py unit-multi-gpu 串行趟容忍 pytest exit 5**（本线无 multi_gpu 标记测试，不修必全线假红）。验证全绿（38 YAML+矩阵+11 路由+ruff）。
> ③两侧均未推送，待用户授权：主分支 fd600e1（force-with-lease 上 PR #581）；040 分支推 fork + 是否开 PR 到 0.4.0-dev 基。

> 交接原因：执行者（AI）多次违反"先证后说/受众优先/发布授权"纪律，用户终止其工作。
> 本文档按"接手者需要什么"组织，不按事故时间线。所有未标注"待确认"的事实都有执行证据。

## 1. 主线：PR #581（可合并状态）

- **地址**：flagos-ai/vllm-plugin-FL#581（fork `lentil1016:ci/workflow-opt` → main）
- **提交结构**：xmhubj 的 16 个提交（#545/#551+12 修复，已 rebase 至 main）+ 1 个修复提交
  `3920669`（路由豁免 + #559 移植 + multi_gpu 单测串行）。用户要求：后续修复一律 amend 进这个单提交。
- **状态**：lint/build/discover/ci-gate 绿；平台 job 按 label 门设计 skip；MERGEABLE，
  仅剩 review approval。合并方式未定（merge/squash 会破坏 17 提交结构，需与 owner 确认）。
- **验证证据**：删门全平台 50 pass/0 fail/0 pending（run 38360115823）；h20 e2e 第 4 次
  尝试真实执行通过（job 115370432571，日志含用例自身 GPU 分配与 PASSED）。PR 内有证据评论。
- **合并后动作**（PR desc 已列）：ci-gate 设唯一 required check 并移除旧 check；
  label 打入流程确认；h20/metax 两条已知风险见 desc。

### 1.1 最新状态（2026-10-15 交接 v2——第二个执行者被终止，接手者从这里读起）

**验收标准（用户原话级）**："我需要一个全平台通过的run" —— 一个全平台（8 启用平台，ascend=enabled:false 设计内 skip）
通过的完整摘门 run + 分支 head 定格回交付头。用户时限压力："如果你今天不解决，明天还会有新合并，
你就永远别想合并了"。

**分支/头状态**：
- fork `lentil1016:ci/workflow-opt`（跨仓库 PR，head 在 fork；**推 fork 不推 origin**，origin 是
  owner 自己的同名分支 cf2dd99，别碰）当前 = `a3e2201`（TEMP8=kunlunxin-only 探针，无任何修复）。
- **交付头 = `6eab46f`** = 修复 commit `4a5a2ed`（c4e6ddd 全部内容 + item15：enflame timeouts
  inference 90/serving 120/concurrent 120 + api-key 脱敏 v2 构造期隔离）+ merge origin/main
  （c55b3bd #583 Flaggems-vllm，唯一冲突点=check_flaggems.py 迁 nvidia/ + setup.sh 两处调用路径）。
- worktree `../vllm-plugin-FL-wopt` 干净，HEAD=a3e2201。

**正在跑的（交接时刻）**：
- 全量 run **38687287556**（head 828e037=交付内容+TEMP6 全量摘门，`-tempval-` 改名并发组）：
  kunlunxin functional 红（见 1.2 专案）+ e2e 级联 skip；其余已绿=nvidia functional、thead 全链、
  iluvatar 全链、musa e2e inference；在跑=h20 e2e×2、musa serving、metax inference；排队=a100 e2e、
  enflame e2e×2（**enflame serving=60min 超时修复的验证考题，预算 120min**）、metax serving、hygon functional。
  本轮注定非全绿（kunlunxin 级联），价值=其余平台证据。监视器 exec_7cb2651f 终态自动出完整台账。
- 探针 run **38692095775**（a3e2201，kunlunxin-only 复跑，无修复）已终态=**红，同签名复现**。

**PR head 注意**：head 现在是探针 TEMP。用户已立的纪律：**每次推头前必须先说明目的+预期面板表现**；
恢复交付头/铺新轮须按既定流程预告；绝不允许"预告+同回合就动手"。

### 1.2 kunlunxin functional 红专案（接手者第一优先，修测试侧）

**症状**：#583（Flaggems-vllm，10-15 03:29 UTC 进 main）新增
`tests/unit_tests/ops/test_clamped_activation_controls.py::test_upstream_clamp_controls`
参数组 controls3-True-1 / controls5-True-1（都是 ["none","+op"] 选入路径，期望 flaggems 路由）
AssertionError: bool([]) == True。**xdist-16 下 2/2 次复现同签名（系统性）；#583 自己的 PR CI
串行跑同平台 15/15 绿**（老管线 `pytest tests/unit_tests/ --platform=kunlunxin -q` 无 -n，
709 passed in 122s）。xdist 134s vs 串行 122s——并行在这个平台甚至不快。

**证据链（两次红 run 日志一致）**：失败 worker（gw6/gw8）captured log 序列：
`FlagGems-vllm integration activated（fake import 成功）` → `Registering vendor backends for
current platform: kunlunxin`（**fixture 明明 mock 成 cuda/nvidia！**）→ `Using custom config from
vllm_fl/dispatch/config/kunlunxin.yaml` → `Op 'silu_and_mul_with_clamp' switched from
'default.flagos' to 'reference.torch'` → forward 走原生 → `registered_clamp` 空。
"switched to reference.torch" 在探针日志恰好只出现 2 次=只伴随两个失败用例（controls0-True-1
期望相同的注册却过了——不同 worker 不同结果）。

**代码追踪已到位**：fixture 的平台 mock 用 `monkeypatch.context()` 只包住
`custom_ops.register_oot_ops(...)`；随后 **mock 作用域外**调 `reset_global_policy()` +
`reset_default_manager()` → `builtin_ops._register_vendor_backends`（builtin_ops.py:55 在调用时
`from vllm.platforms import current_platform`）读到**真实** kunlunxin → 注册 kunlunxin vendor +
加载 kunlunxin.yaml 策略 → op 被钉到 reference.torch。

**未解之问（下一步）**：为什么串行同序列不挂？#583 串行日志 grep 'Registering vendor backends'
=0 命中——**但先排除日志捕获差异**（确认串行日志里到底有没有任何 vllm_fl INFO 行）再下结论。
若确认串行从未触发 vendor 注册 → 分叉点在进程级缓存/幂等（serial 进程里 manager 已被更早测试或
conftest 初始化），追 `reset_default_manager`（manager.py:632）、policy 缓存（policy.py:515/210）、
以及 kunlunxin.yaml 是否显式钉 silu_and_mul_with_clamp。

**已排除**：flag_gems 包缺失论（#583 串行分发成功证明包装了）；FlagGemsBackend._available
缓存毒化论（whitelist 测试不碰 sys.modules/import）。

**修复方向（用户已定调）**：正修=**修 #583 测试 fixture 的隔离性**（平台 mock 覆盖 reset 调用，
或 fixture 强制清进程级缓存），修完 kunlunxin-only 探针验证。
**串行化（-n 0）已被用户否决**——理由：全平台为单测试买单、#583 修好后无人记得撤、掩盖套件
未来真正的并行安全问题；multi_gpu 先例属设备资源硬冲突与此不同类。排除测试=改考卷同样不可。
诊断包应经用户转 #583 作者（cyber-pioneer）——该测试的隔离缺陷在 main 上对任何并行 runner 都是雷。

**修好后的既定流程**：kunlunxin-only 探针验证绿 → 全平台全量摘门 run（推前预告）→ 全绿 →
head reset 交付头定格（gated 面板）→ 终报。

### 1.3 本轮（10-15）已裁决的历史红账（全量 run 尾部若复现按此对号）

- **enflame serving"cancelled×3"=误诊已翻案**：真因 60min 缺省超时（`_platform_test.yml:146`
  `matrix.test.timeout || 60`），第四轮 `Overall: PASSED` 后 21s、artifact 上传中被杀；
  已修=交付头里 `timeouts.e2e: inference 90/serving 120/concurrent 120`。
  本轮 enflame serving 带 120min 预算=该修复的验证考题。
- **hygon e2e inference 红（第四轮）**=FlagGems JIT 共享缓存 SQLite 锁竞争
  （`sqlite3.OperationalError: database is locked`，`/github/home/.flaggems/code_cache`，
  35b_a3b_tp4_graph graph capture 期 Worker_TP1 死）——未修只重试过；本轮 hygon e2e 若再红
  先比对同签名；修的方向=run.py 按用例注入隔离的缓存目录（与 HCCL 端口注入同机制的先例），
  但属配置级变更需用户点头。与 #592 h20/0.28 悬案同族（那边是 NCCL init 速死，证据链独立）。
- **nvidia benchmark**：第四轮排队 20h=runner 缺位（标签 `nvidia/gpu-8/cuda130`，绿证据在
  `bm-baai-dx-zone1-d-a100-40g-4-38` 上，run 38562228898）；机器不在只有用户能拉。
- **僵尸 run 死锁并发组**：run 级永久 queued（等不存在的 runner）会让同名并发组的新 run
  pending 不派 job、cancel-in-progress 无效；fork 身份 cancel/rerun-job API=403（admin only）；
  绕法=TEMP 层把 `concurrency.group` 改名（本轮已用 `-tempval-` 两次）。`/rerun-failed-ci`
  命令只认 PR head 的 run，对非 head 的验证 run 无效。
- **#583 落地即 PR dirty 的机制**：pull_request 类 workflow 从 merge ref 运行，merge ref 生成
  失败（冲突）则 CI/Code Scan **静默不建 run**（labeler 走 base 照跑）——症状=checks 面板只剩
  label。处置=merge main 解冲突后 CI 自行恢复。**明天再有合并进 main，同一套处置**。

## 2. 最重要未竟事项：Phase 2（reviewer 建议的本体）→ **已本地完成，待推送授权**

cyber-pioneer 行内建议（PR #581，platforms.yml:28）：`global_trigger_paths = whole_path - vendor_path`。
**vendor_path 指字面的 `vllm_fl/dispatch/backends/vendor/` 目录**（用户已确认；此前执行者误读为
config yaml 并按误读实现——误读部分的豁免效果无害，已保留在 3920669，无需回滚）。

**2026-10-13 接手者已落地**（amend 进原 3920669，本地现为 397de0b，未推送）：
7 个启用平台（cuda/musa/metax/iluvatar/thead/kunlunxin/enflame←gcu）在 trigger_paths 与
exempt_from_global 同加 vendor 目录条目；exempt_from_global 键注释已补；提交信息已更新。
本地验证全绿：拒绝重复键 YAML 加载 + detect_platforms.py 真实调用方式 21 用例路由矩阵
（vendor 单平台命中×7、双 vendor 双平台、无主目录 sunrise/supa/txda/template/ascend 留全局、
vendor+共享代码混合→全量、既有行为回归×5、no-diff 模式）+ ruff。零防漂移告警。
推送需 force-with-lease（amend 重写了已在 PR 里的 3920669）。回复策略（用户 2026-10-13 指示）：
**不跟帖**，把原"已实现"回复（评论 id 4266288294，cyber-pioneer 建议 id 4266110602 下）
原地重写为一条干净统一的终稿——只讲最终语义/边界/防漂移，删 commit 哈希、方案辩护、
实现顺序等过程信息；回复必须在推送之后发（否则"已实现"不实）。推送+改评论待用户授权。

映射来源 `vllm_fl/utils.py` 的 `VENDOR_DEVICE_MAP`；`vendor/hygon` 目录不存在（map 注释过时）；
thead 目录同时被 tsingmicro 注释引用但该平台无条目。

## 3.5 subagent 三路对抗审查结论（2026-10-14，全部主张已由接手者亲手复核）

**通过面**：gate 320 结果组合零偏差；fromJson 新用法位于不可达路径；heredoc 缩进/注入安全；
branch 传播失败响亮、空串=事件 ref；vendor 公式 B′ 无欠触发反例（未认领 vendor 目录=over-trigger
保守方向，reviewer 字面公式反而黑洞）；label 档位切换正确；#559 移植忠实；提交信息与 diff 相符。

**已坐实待修（我方单提交管辖面）→ **2026-10-14 已全部修复**（649296c，amend 进同一提交，
本地全绿含失败注入测试+CI run 38519592461 绿，评论哈希引用已同步更新）：①hygon 改指
hygon.yaml（ptg.yaml 死文件留全局=安全方向）；②nightly/weekly 过滤分支查 returncode+stderr
非零退出（stub exit2 注入实测步骤响亮失败且零输出写入）；③paths-ignore 删 docker/**；④globals
补 requirements/tools/tests-benchmarks（实测三路径全平台路由）；⑤文档四处收尾（docstring/
Rules 注释/头注 workflow_dispatch/"Stage design"改名）。gate 对齐差一列系 agent 误报未改。
**owner 决策项**：⑥ci-functional/ci-e2e/ci-full 三 label 仓库不存在（gh label list 证实），
合并前必须创建否则三档门+全部 PR 硬件测试是死的；⑦vllm_fl/ 未来新增顶层目录零路由
（globals 枚举式，reviewer 给过"保守回退"选项，语义大改需 owner 定）。
**预先存在（xmhubj 16 提交，reviewer 未 flag）**：⑧pr_nightly/weekly 触发步骤 ${{ }} 内插
bash=命令注入面（triage+ 可利用）；⑨multi_gpu 串行趟 exit-5 地雷；⑩unlabeled 不重跑陈旧绿、
draft gate 恒红、/nightly 空格分词、label 大小写敏感。
**CodeQL 边界**：告警 API 403 不可直读；代理证据=新 head 无新告警评论+Code Scan run 绿
（块内注释应已压制 agent A 的"注释无效"质疑）。

## 3. 其他待决/待办

0. **cyber-pioneer 7 条 merge-before 总结 + CodeQL ×2 + 提问：2026-10-14 已全部修复并推送**
   （d368c8a，amend 进同一提交；本地全量验证：gate 脚本实测 DISCOVER=fail→FAILED、
   #3 复现法修复前后对照、路由矩阵含 csrc/docker/workflows、YAML 重复键×11 文件、ruff；
   CI run 38516850124 绿）。已发两条 PR 评论：7 条落地情况（含 #7 收窄 d58f3ac 语义的
   可否决声明）+ PR/nightly 用例分布回答。**开口**：①#2 dispatcher 的端到端验证结构性
   绑定 merge 后（issue_comment workflow 只从默认分支执行，main 上无此体系，首个真实
   命令即验证）；②#7 若 reviewer 要保留 PR 侧 benchmark 常跑，改回一行；③并行会话在
   主工作目录切了 ci/metax-airs-model-mount（PR#588 线），本线工作在独立 worktree
   ../vllm-plugin-FL-wopt。
1. `platforms.yml` 的 `exempt_from_global` 键缺文件内说明注释（设计稿有、落地漏）。**已补（397de0b）**
2. 工具分发位置：`~/.local/bin/flagrun`（守护+缓存+TUI，--selftest 15/15）未入库，位置待用户定。
3. 改进 backlog（用户说"后面再说"）：label 三档等价、ci.yml dispatch 过滤输入、concurrency 组键
   防楔死、merge queue 兼容、PR 推理用例 smoke 化（前置=各节点 0.6B/4B 落盘）。
4. #587 已关闭（slash 命令尾空格静默失效的 trim 修复），等 #581 合并自然根治；分支
   `ci/slash-command-trim` 留存含完整事故证据。

## 4. 环境事实与坑（全部实测）

- 凭证：仅 fork 推送权。加 label / workflow dispatch / 删他人或机器人评论 = 403。
  `/rerun` `/cancel` 命令评论 PR 作者可用；**命令体严格匹配，尾空格静默失效**（#579 事故）。
- GitHub 怪癖：reusable workflow 的 run 级 status 可长期卡 queued 而 job 实际在跑/已完成——
  判断状态一律看 job 级；零 job 的 pending run 是楔死态（自愈或重触发，组织内已发现 8 例）。
- runner 池：h20 = baai-sailing-h20-1（单台）；同仓库常驻 ~8 条分支抢同池；h20 节点有**动态**
  显存残留（其他分支 27B/35B 用例占用未释放），0.95 gpu_memory_utilization 的用例在残留窗口
  等待 1800s 超时——干净窗口存在，重试可过，红了先查节点不是用例回归。
- 本地验证纪律：ruff 0.16.10（pip --break-system-packages）；YAML 用拒绝重复键的 Loader；
  detect_platforms.py 可本地直调验证路由矩阵；推送前跑全量。

## 5. 必须传承的工作规矩（用户明确立下）

1. **发布授权制**：任何离开本机去仓库的内容（评论/desc/PR/推送），逐条获得用户明确授权；
   模糊反问不构成授权。
2. **受众优先**：PR 交付物=决策输入（行为变化/证据/风险/归属），不是过程叙事；"内部 review/
   迭代史/自造术语"禁止出现在仓库侧；发言前先答"读者是谁、他知道什么、需要什么"。
3. **执行证据先于宣布**：没有执行证据不得宣布完成；机器故障不证明 workflow 没问题；
   主张边界=证据边界。
4. **无人值守**：拿到执行权的第一动作是执行（不是造观察工具）；每个终态转换必须带通知
   （run_in_background，不是 nohup）。
5. **改被测参数=改考卷**：gpu_memory_utilization 等测试内容变更需用户逐项点头。
6. 我的公开发言消耗的是用户的信用（"你在消耗别人对我的耐心"）。

## 6. 遗留进程

- 无（交接时已清空：run 监视器 exec_7cb2651f 已停；30 分钟汇报 automation 已删；
  org-runner-activity 守护已 kill）。全量 run 38687287556 在 GitHub 侧自然继续跑完，
  终态台账用 `gh api .../actions/runs/38687287556/jobs --paginate` 自取；
  /tmp/h20_retry_controller.py 是重试控制器模板（含显存快照可行性判定逻辑）。

## 7. 第二个执行者的追加教训（2026-10-15 被终止）

1. **"没有异议我就执行了"=同意剧场**。预告方案后同回合就动手=零异议窗口，等于没给授权机会。
   规则：预告后**停**，等真实回应；沉默≠同意（与 5.1 同根）。
2. **类比引用必须同类**：multi_gpu 串行化是设备资源硬冲突（结构性必需），不能给"绕过可修 bug
   的平台级串行化"背书。引先例前先问：先例成立的前提条件在这里满足吗？
3. **PR 面板是陈述的一部分**：验证未收口时不给 PR 换全绿脸（gated 绿会被读者当"完成"）；
   TEMP 探针/顶替 commit 的面板副作用要预告。每个头动之前三句话：推什么/为什么/面板显示什么。
4. **改法被质问"对么/有先例么"时**：先重新审计自己的主张（本轮串行化方案确实经不起"正修"
   追问），不要防御性坚持。
5. 环境补充：`url.git@github.com:.insteadof https://github.com/` 全局重写（https 推送其实走
   ssh）；gh/SSH 身份=lentil1016，对 flagos-ai 上游零管理权（cancel/rerun-job/label=403）。

## 8. 只读分析轮成果（2026-10-15 晚，两个 subagent，未动仓库）

### 8.1 kunlunxin clamp 失败机制【已钉死】
- **串行日志 0 命中"Registering vendor backends"=捕获差异**（pytest -q 只在失败时打 captured 段）；两份 coverage 表逐字节对比证明串行执行了完全相同的 dispatch 路径（含 kunlunxin vendor 注册）。
- **唯一行级差异**：`vllm_fl/dispatch/__init__.py:237-244` = CachedOp 的**异常静默回退分支**，仅 xdist 覆盖、串行从未进入。
- **完整因果链**：obj(x) → reset 后懒重建 manager（此时平台 mock 已退出→真实 kunlunxin vendor+kunlunxin.yaml）→ 解析选 default.flagos（日志"using 'default.flagos'"）→ **impl.fn 抛异常**（钉到 flaggems/impl/activation.py:62 的 `from flaggems_vllm import ...`，在 register 期 import 成功后的窗口内 fake 导入态被破坏）→ CachedOp except Exception **无日志吞掉** → 标记失败→回退 reference.torch（日志"switched"）→ calls 空 → 断言挂。
- **为何只有部分变体挂**：12 变体中只有 controls0/3/5-True-1 具备失败能力（enabled∧switch=1 才会真调 fake）；挂哪几个取决于 xdist 轮转落在哪个 worker（2 次 run 宿主 worker 不同、变体相同）。串行=单进程全序，毒化被稀释。
- **修复草案（测试侧，未落地）**：fixture 全测试期钉死 vllm_fl 两个平台入口（`vllm_fl.dispatch.config.utils.get_platform_name`→"cuda"、`builtin_ops._get_current_vendor_backend_dirs`→None）+ setup 内清 CachedOp 粘性状态（_impl/_frozen/_use_manager_call）并预热 dispatch 世界；可选加固断言（failed impls 非空即报错，把静默回退变响亮）。
- **生产侧独立建议**：dispatch/__init__.py:237-244 的静默回退至少加 logger.warning——本次根因不可见的直接原因。验证第一步=临时加 warning 跑一次拿异常原文。
- **波及面（review 侧交叉证实）**：clamp 测试的失败机制适用于**所有非 cuda 平台**（hygon/enflame/musa/thead/iluvatar 预期同挂，metax 可能过）；修复必须按"全平台"设计而非仅 kunlunxin。

### 8.2 增量 review 发现【两条 P1 静默缺陷，同根因=#583 按 "cuda" 写死 × 本 PR cuda→nvidia 改名】
- **F1（P1）**：#583 的 flaggems 双模 e2e 门控三处死代码——`generate_matrix.py:398` `if args.platform == "cuda"`（CI 恒传 nvidia→永假，flaggems_vllm 0/1 双模矩阵永不展开）+ `_e2e_test.yml:113` 同款条件→fused=0 回归门静默丢失。
- **F2（P1）**：`tests/functional_tests/flaggems/conftest.py:17-18` `platform != "cuda"`→skip——#583 的 nvidia release 检查测试（test_vllm_package/test_vllm_switch，13 个 fused 算子导出/分发/图捕获校验）**全平台静默零覆盖**。
- 两条都是"CI 绿但功能丢了"的静默类，不修则 #583 的核心保障在本 PR 里名存实亡。修法=三处守卫从 "cuda" 改 "nvidia"（或平台别名）。
- **P2×4**：脱敏建议在 serve_args/to_cli_args 层显式拒 api_key（当前树密闭但靠调用约定）；enflame concurrent 120 无实测依据；enflame 镜像 pip 私服可能缺 flaggems-vllm 的 sqlalchemy 依赖（install_flaggems_vllm.py 不继承 --extra-index-url，观察项）；platforms.yml:18/README/model_config/_functional_test.yml 四处旧名 "cuda" 文档串。
- **干净项**：enflame timeout 键位/链路/数值✓、脱敏 v2 当前树密闭✓、check_flaggems 迁移零差异✓、merge 零内容丢失✓、路由无零平台洞✓、#583 其余新测试 fixture 全密闭✓。

### 8.3 接手者注意
分支头已被后续工作推进（分析时观察到 TEMP10/8d81807、TEMP11/824e141 在修 clamp，方向=pin control tests to CPU）——8.1 的机制/修复草案与 8.2 的 F1/F2/F3 波及面结论应与其对表，避免只修 kunlunxin 单平台或漏掉两条静默 P1。

### 8.4 补遗：F1/F2 的成因定性（回答"昨天不是让改过名么"）
- test-cuda→test-nvidia 改名（10-14，交付 commit item 13）在当时的树上是完整的。
- F1/F2 三处（generate_matrix.py:398 / _e2e_test.yml:113 / flaggems/conftest.py:17-18）全部是 #583 的新代码，10-15 03:29 进 main、04:56 经 merge 进入本分支——#583 按 main 的旧名 "cuda" 写死，与本分支改名相撞。
- **本分支的漏=merge 后未做"改名 × 新合入内容"语义对表**（一个 `grep '"cuda"'` 级别的审计）；git 只报文本冲突，字符串比较级冲突不可见，导致两条静默缺陷穿透两轮全量绿 run。
- **规则（每次 main 新合流后必做）**：merge 解冲突后，对本 PR 改过的名字/键/路径（platform 键、目录、环境变量）在新合入 diff 里全量 grep 对表；静默类缺陷（矩阵不展开/测试 skip）不会让 CI 变红，只能靠审计抓。

### 8.5 平台速度画像（回答"谁慢、为什么"，供全量 run 排期预期）
- 结构性慢仅 enflame 一家：`TOPS_VISIBLE_DEVICES: "2,3"` 只可见 2 卡，tp2 用例被迫串行 + S60 单用例本身慢（~28min/个）+ eager-only。e2e serving 是每轮全量的关键路径（~61-70min）。提速杠杆=多可见卡让两用例并行（环境侧变更，需用户定）。
- hygon 非结构性慢：8 卡全开、4 用例两波并行；上轮 52.2min 灌了 ~30min 失败水（sqlite 慢死级联），干净基线 ~30-40min（kunlunxin/metax 档）；graph capture + serving 慢启动是正常开销。
- **sqlite 竞态修法注意**：hygon 已设 `TRITON_CACHE_DIR`（每 job）但不管用——撞的是 FlagGems 自有 `~/.flaggems/code_cache`。根治=给 flaggems code_cache 也做按用例隔离（HCCL 端口注入同机制先例）。
