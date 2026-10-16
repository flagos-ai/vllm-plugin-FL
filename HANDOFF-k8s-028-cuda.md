# 交接文档：vLLM 0.28 CUDA 线 / k8s 池适配（PR #562）（2026-10-14）

> 与 `HANDOFF-ci-workflow-opt.md` 配套（工作规矩两文档通用，见彼文档第 5 节）。
> 本线 10-10 达到可合并终态后被用户搁置，10-12 转入 main 线 ci/workflow-opt 工作。
> 所有事实均有执行证据支撑；未证实的明确标注"未证"。

## 1. 主线：PR #562（可合并状态，自 10-10）

- **地址**：flagos-ai/vllm-plugin-FL#562（fork `lentil1016:ci/cuda-nv-k8s-runner` → **0.4.0-dev**，
  注意不是 main）。0.28 工作全部以 0.4.0-dev 为 base（nvidia #484 已合、ascend #487、
  iluvatar #527、metax #554 均如此）。
- **终态**：单 commit `e8a1f77`（26 个调试提交压缩而成），run **38021001781 全线真绿**
  （22 checks 全过，含 a100 五 scope bypass 实跑），mergeStateStatus CLEAN、CLA 已签，
  判定 ready to merge。合并动作留给用户（执行者无基仓库写权限）。
- **终态架构**：
  - **cuda-a100 平台**（`.github/configs/cuda-a100.yml`：k8s 池 `vllm-plugin-nv-k8s`、
    `runner_mode: k8s-bypass`、`script_platform: cuda`、`model_nodes: [prod-172, prod-174]`）
    与 **cuda 标准线**（旧 gpu-8 runner、container、NVLS、h20）并列于 ci.yml；
  - bypass 四孪生（unit/functional/e2e/benchmark），静态命名 + runner_mode 门控，
    notify 公式接受任一孪生；benchmark-bypass 门 = unit-bypass success **且**
    (e2e-bypass success || skipped)——`|| skipped` 单用会被失败级联的 skipped 空开放行（实测吃过）；
  - e2e/benchmark `--pin-nodes` 钉健康节点 172/174；unit/functional pod `--model-hostpath ''`
    解挂载任意调度；pod 带 activeDeadlineSeconds（runner 被杀不漏占卡 pod）；
  - 模型：我方已落盘 `/flagcicd/model/ci-cache/{Qwen3-4B, Qwen3-Next-80B-A3B-Instruct,
    Qwen3.5-35B-A3B, MiniCPM}`（sharefs，172/174 可见），cuda-a100.yaml 以 per-case
    device_overrides 指过去；benchmark smoke 同法 override 到 staged 4B（dummy 权重只需
    config.json 存在，原始 /data 路径池上没有会炸 HFValidationError，两轮实证）。
    **全链无运行时下载**（用户裁决；仓库零先例）。下载链脚本已删（git 历史在）。
- **NVLS 修复闭环**：NCCL_NVLS_ENABLE=0 在 h20 节点 e2e 双绿首次实跑验证（10-09，清盘后）。

## 2. 池侧问题（运维渠道，报告=`k8s-pool-container-job-调查报告.md` 工作区根目录）

1. **容器 hook 坏（核心）**：`k8s-novolume`（ARC runner-container-hooks 魔改版）在
   "等 pod 就绪"成功后、第一个执行请求前**异步链静默断裂**，进程自然退出（码 0）、
   回执不落盘 → CI 报笼统的 "Executing the custom container implementation failed"。
   17MB 打包单行文件，断点定位需未压缩源码。四轮隔离实验（run 37743585700/37744570431/
   37746316638/37746787657）证明与 workflow 声明无关。给运维的选项 A（拿源码修）/
   B（换上游原版）/ C（确认 bypass 不违规）在报告第六节。**container_volumes 在该池是
   死配置**（hook 无 volumes 语义、节点无 /data）→"hook 修好自然绿"只对 unit/functional 成立。
2. **node-009 同构问题（根因未证）**：pod 20 秒 phase=Failed、containerStatuses 为空。
   曾过度归因为"缺 sharefs 挂载"——**不成立**（FailedMount 滞留 Pending 不转 Failed）。
   失败路径已加 k8s events 抓取（FailedMount 只在 events 里），下次复发自证。
   应对 = model_nodes 钉 172/174，已生效。
3. 其他：codeload 代理间歇超时（两池均中过）；pod 出网仅 modelscope（hf-mirror/hf 不通）；
   runner 扩容延迟（unit 曾排队 25-28 分钟）。
4. 调查工具（透明记录代理/出口追踪器/逐帧协议探针/权限自查/模拟 API 服务器）——
   **本轮工作区未找到实物**（报告声称本地存档）。接手时向用户确认位置，或按报告
   第三~五节的设计复刻（成本不高）。

## 3. 待办 / 待决

1. **PR #562 合并**（按钮在用户；方式 merge/squash 需确认——squash 会毁单 commit 结构）。
2. **Dockerfile.v0.28.0 缺失**：现镜像 v0.28.0-cuda-ci 是仓库外手工构建，需补 Dockerfile。
3. main → 0.4.0-dev 同步债：merge-tree 7 文件冲突。
4. CodeQL 4 项待修（基线上）。
5. #559（用户 debug 分支，base main）：版本 pin 建议已给，待用户定夺。**debug 分支勿动**。

## 4. 环境事实与操作知识（实测）

- **镜像**：`plugin/vllm-plugin-fl:v0.28.0-cuda-ci`（digest ad7a8dfc，12.8GB；冷拉 128-320s
  横跨扁平 300s 超时线）。基底 vllm-openai v0.28.0-cu129/torch 2.13.0+cu129；
  FlagTree 0.6.2a1（Triton 3.6）；FlagGems 5.3.5。
- **k8s_bypass.py 要点**：SA 建 pod（hostIPC:true）；tar 灌 workspace；读前不发 WS close 帧；
  退出码从 channel-3 解析；--copy-back；wait_ready 状态机 = Unschedulable 排队等满预算
  （--ready-timeout 1800s 与 --read-budget 是两笔预算）、容器级 TERMINAL_WAIT_REASONS 速败、
  **pod 级 phase Failed/Succeeded 也速败**（node-009 教训：容器没建起来时容器级检测全漏，
  曾对死 pod 空转 30 分钟）；失败瞬间抓容器详情 + events（超时路径也要 dump）。
  cleanup 注册表按 `platform.split('-')[0]` 归一（cuda-a100→cuda，否则清理静默 no-op）。
- **workflow 纪律（违者已实证翻车）**：对执行中 run force-push 会把它打进**不可取消的
  queued 态压住全队**（三次实证）——规则=一次推送盯到终态再动；concurrency 组 key 用
  `workflow+PR号`；YAML 用拒绝重复键的严格 Loader（safe_load 会静默吞重复键）；lint 前
  本地 ruff check+format 完整镜像；仓库自带 `/cancel-ci` `/rerun-failed-ci`（精确匹配、
  PR author 可用、只作用于当前 head SHA 的 run；**注意尾空格静默失效**，#579 有完整事故）。
- 凭证边界与通用规矩见 `HANDOFF-ci-workflow-opt.md` 第 4-5 节。
- 本机相关分支：`ci/cuda-nv-k8s-runner`（本地+fork）、`debug`（用户，勿动）、
  `origin/0.4.0-dev`。兄弟目录 `vllm-plugin-FL-040`/`vllm-plugin-FL-wopt` 疑似另两条线的
  独立 checkout（未验证内容，接手时确认）。

## 5. 与 main 线（ci/workflow-opt，PR #581）的关系

- 两条平行线：#562 在 0.4.0-dev（0.28），#581 在 main（0.24），共用 h20 节点池
  （baai-sailing-h20-1）与部分 runner；h20 节点动态显存残留问题两线共害。
- #581 若合并，平台 YAML 的 stage 结构与 0.28 线将来对齐时需重构（详见彼交接文档）。
