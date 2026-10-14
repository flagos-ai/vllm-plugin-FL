# 交接文档：燧原(GCU) 0.24 decode 慢速问题调查 + 0.24 使用文档

> 本文件**不要提交进仓库**（未跟踪的临时交接文档）。调查结论最终应整理进 PR #464 的评论或交付给燧原的问题报告。

## 0. 三条 PR 状态（背景）

| PR | 分支/head | 状态 |
|---|---|---|
| #464 enflame 0.24 | feat/enflame-vllm0.24 @ cf47a04，3 commits（b0b2e5c py迁移 / f837f5d build+ci / cf47a04 lint continue-on-error 临时措施） | 等 owner review；lint 问题已在 PR 评论解释（#436 使 lint 变 diff-scoped 绕过 exclude，主仓修法=--force-exclude+pin ruff，我们不负责修）；CodeQL import-sys 告警等 owner 意见，**勿自行推修复** |
| #457 musa 0.24 | feat/musa-vllm0.24 @ c6167fa 单 commit | CI 全绿；musa-vllm024-known-issues.md 五问题报告在仓库内，待用户上报摩尔 |
| #480 slash commands | （gh pr view 480 看） | rerun-failed-ci.yml + cancel-ci.yml + COMMANDS.md，已完成等合并 |

后台 CI 监控：本地 `/tmp/ci_monitor.log`（task exec_a1f9866d，watch run 34817591951）。最终结论没确认过，接手后 `gh run view 34817591951 -R flagos-ai/vllm-plugin-FL` 补看。

## 1. 核心发现（已实锤，有证据）

**燧原 0.24 decode 慢的根因定位到了算子级**：

- py-spy 两次 dump（间隔 15s，引擎已连续服务 31 分钟，均在 decode 活跃期）主线程都在同一条链上（S60 宿主机 `/root/pyspy1.txt`、`/root/pyspy2.txt`）：

```
vllm/v1/sample/ops/topk_topp_sampler.py  apply_top_k_top_p   ← 客户端 top_k=20 + top_p=0.95 触发
→ torch.sort
→ flag_gems ops/sort.py sort → radix_sort → ops/cumsum.py cumsum_wrapper
→ flag_gems/utils/libentry.py  policy/default_policy/bench/do_bench   ← 自动调优
→ triton_gcu compile → make_fatbin → gcc 子进程(subprocess.communicate)
```

- 即：**采样路径的 torch.sort 被 flag_gems（enflame 厂商后端 `_enflame/gcu300`）接管，其 libentry autotuner 在 decode 期间反复 do_bench + 编译**。
- 佐证：serve 日志反复刷 `flag_gems/runtime/backend/_enflame/gcu300/ops/cumsum.py:184 UserWarning`（= 反复执行/重编译该 kernel）；efsmi dev0 100% util 但只 158W/300W（设备在跑 benchmark kernel，不是有效计算）。
- CI 同样存在：27B TP2 e2e inference，`1/1 [10:10<00:00, 610.26s/it, output: 0.13 toks/s]`（本地证据 `/tmp/ci_infer.log`）。CI 仍绿是因为 smoke 用例 token 少、60min 超时内能跑完。
- 本地基线：0.6B TP1 enforce-eager，0.4-0.7 tok/s（serve 日志 Avg generation throughput）。
- **注意**：慢与采样参数绑定。top_k/top_p 采样才走 torch.sort；greedy 客户端理论上不受影响（未验证，见 §3 步骤 2）。

## 2. 黑名单机制（已查明）

- 官方开关（无需改 py）：`VLLM_FL_FLAGOS_BLACKLIST`（整体替换）/ `VLLM_FL_FLAGOS_BLACKLIST_APPEND`（追加）/ `VLLM_FL_FLAGOS_WHITELIST`。
- 生效点：`vllm_fl/worker/worker.py:258-289` → `flag_gems.enable(unused=blacklist)`（serve 启动时，改 env 后需重启 serve）。
- 算子名词表 = flag_gems `_FULL_CONFIG` 名：已确认存在 `sort`、`sort.stable`、`cumsum`、`cumsum.out`（还有 topk/argsort 等）。
- **enflame 厂商后端还有自己的黑名单**：容器内 `site-packages/flag_gems/runtime/backend/_enflame/__init__.py` 的 `CUSTOMIZED_UNUSED_OPS`（gcu300 一大串、gcu400 极简），sort/cumsum **不在** gcu300 列表里。改它=容器内临时 debug 措施（不进仓库）。
- 相关 flag_gems 旋钮：`FLAGGEMS_DB_URL`（flagtune 调优库，没有就在本地 do_bench）、`TRITON_PRINT_AUTOTUNING=1`（打印每次 autotune 及缓存命中情况）。

## 3. 已做/未做完的实验（接手从这继续）

### 2026-09-14 续查结论

1. **sglang 的 `nonzero,nonzero_numpy` 黑名单对 vLLM 无效（已 A/B）**。
   - 参考来源：sglang-plugin-FL commit `933223c`。sglang 的慢链是
     `index_put_ -> nonzero -> cumsum`，黑名单后回退 torch_gcu native nonzero，
     4B serving 从首请求 15-20 分钟降到整组 261 秒。
   - vLLM 0.24 的实际代码不经过 nonzero：
     `apply_top_k_top_p_pytorch -> logits.sort -> flag_gems radix_sort -> torch.cumsum`。
   - 相同 0.6B TP1 服务、相同 16-token sampled 请求实测：
     `VLLM_FL_FLAGOS_BLACKLIST_APPEND=nonzero,nonzero_numpy` 为 **16.287s**，
     不加黑名单为 **16.583s**；可视为无差异。greedy 对照为 **1.544s**。
   - 因此这两个算子名不能照搬，但它们强力佐证了同一版 Enflame
     FlagGems/Triton GCU 的 sampler 冷编译/算子实现问题。

2. **blacklist `sort` 也不是可用缓解**。
   - dev0 单卡 reset 后、不启用 FlagGems，原生 torch_gcu `torch.sort` 对
     `gcu float32 [2,32]` 直接 `SIGABRT`：`topsatenSort` 返回
     `TOPSATEN_STATUS_BAD_PARAM`。
   - 此前给 `sort,sort.stable,cumsum,cumsum.out` 加 blacklist 即使成功回退，
     也只会选到这条会中止进程的 native 实现。因此不应把这些项加入
     `enflame.yaml`。

3. **sglang 的真正可复用方案是 sampler CPU detour，vLLM 原型已实测成立**。
   - 参考：sglang-plugin-FL PR #88 / commit `7d2c05e` 的 Enflame sampler patch。
   - 容器内临时 patch vLLM 的 `apply_top_k_top_p_pytorch`：GCU logits/k/p 搬到
     CPU，以 vLLM 原函数完成 sort/cumsum/mask，再把 filtered logits 搬回 GCU；
     后续 softmax/exponential/argmax 仍在设备上。
   - 相同 sampled 16 token：**16.583s -> 1.859s**；同服务 greedy 为
     **1.275s**。sampled 128 token：**11.189s / 128 tokens**，服务日志观测到
     generation throughput **8.3 tok/s**，无 Sip/device/error 日志。
   - 原型补丁保存在本机 `/tmp/gcu_sampler.patch`，未改本仓库；正式实现前需补
     单元测试（CPU 参考语义、top-k only/top-p only/k+p、多 batch、dtype）和
     Enflame serving 回归。高并发下每 decode step 的 D2H/H2D 成本仍需压测。

当前建议：向 Enflame owner 报告 native `topsatenSort` 崩溃和 FlagGems sort
性能/调优问题；插件侧若需在 0.24 立即可用，采用 **Enflame-only CPU sampler
detour**，不要采用 `nonzero/nonzero_numpy` 或 `sort/cumsum` 黑名单。

旧 R1 曾用 `VLLM_FL_FLAGOS_BLACKLIST_APPEND=sort,sort.stable,cumsum,cumsum.out`
重启服务，但日志仍进入 Enflame cumsum。无论当时是 env 传递还是注册顺序问题，
本轮已证明 native `topsatenSort` 会 `SIGABRT`，所以不再继续把 sort blacklist
作为缓解方向。

**剩余工作（按序）：**

1. 将 `/tmp/gcu_sampler.patch` 整理成正式 Enflame-only 插件补丁并补单元测试。
2. 在 S60 跑 top-k only、top-p only、k+p、不同 batch/dtype，以及原有文本、图像、
   并发 e2e；重点记录高并发下 CPU 排序和每 token 往返的开销。
3. 若正式采用，再更新 0.24 使用文档：说明 sampled 的临时 CPU detour、性能数据和
   移除条件；greedy 无需绕行。
4. 给 Enflame owner 的报告同时附两条独立问题：torch_gcu native
   `topsatenSort` 的 `TOPSATEN_STATUS_BAD_PARAM`，以及 FlagGems radix sort/cumsum
   的异常慢速与 autotune/编译不收敛。

## 4. 机器/会话操作备忘（S60）

- 访问：**只能**走本机 tmux 会话 `sy`（`tmux attach -t sy`），它是 `ssh JMS-701152d2-5988-4d33-8942-f29f213668b7@bastion.aiops.baai.ac.cn -p 2224` 的交互式跳板会话。BatchMode 非交互复制连接已试过：Permission denied，**开不了并行连接**。
- 命令往返模式：`tmux send-keys -t sy '<cmd>' Enter` → sleep → `tmux capture-pane -t sy -p | tail -N`。发前先 capture 确认 pane 空闲。
- 教训：C-c 会把 tty 输入缓冲里排队的命令一并 flush 掉；有 [Y/N] guard 时 Y 和回车要分开发；zsh 无分词，变量当单 arg；**base64 注入必须在同一条 Bash 调用里生成变量再拼进 send-keys**（shell 状态不跨调用保持，踩过一次空注入）。
- 容器：`ef-doc`（--privileged --ipc=host --network host -v /dev:/dev -v /data:/data），镜像 `harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.24.0-enflame-ci`，插件源码 `/root/fl` @ cf47a04，模型 `/data/models/Qwen/Qwen3-0.6B`，8×ZIXIAOC200。用完记得 `docker rm -f ef-doc`（S60 是 CI runner 宿主机，占着 dev0 ~42GB）。
- 容器内已有客户端：`/root/client_text.py`（top_k=20/top_p=0.95/max_tokens=256，非流式）、`client_image.py`、`client_concurrent.py`。

## 5. serve 启动命令（已验证可用的原样）

```bash
docker exec -d ef-doc bash -c "cd /root && \
  ENABLE_I64_CHECK=0 TORCH_GCU_ENABLE_INT64_AND_UINT64=1 TORCHGCU_INDUCTOR_ENABLE=1 \
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=99999 VLLM_PLUGINS=fl \
  [VLLM_FL_FLAGOS_BLACKLIST_APPEND=...] \
  vllm serve /data/models/Qwen/Qwen3-0.6B --served-model-name qwen \
  --host 0.0.0.0 --port 8010 --tensor-parallel-size 1 --enforce-eager \
  > /root/serve_<tag>.log 2>&1"
```
就绪判据：日志出现 `Application startup complete`。吞吐：日志 grep `Avg generation`。EngineCore pid：`ps -eo pid,args | grep "[V]LLM::"`。py-spy：`docker exec ef-doc py-spy dump --pid <pid>`。
⚠️ 已知疑点：命令行 env 前缀是否真到 EngineCore 未验证（/proc 被 setproctitle 毁掉）；若需保 env，用 `python -c` 包一层（import 前先 os.environ 赋值再调 vllm CLI main）最稳。

## 6. 文档任务（0.24 使用文档，写到一半）

- 用户要求：**每条命令先验证可用再写**；**文档里附真实示例输出**供读者比对。
- 0.20 模板已提取：本地 `/tmp/enflame020_doc.txt`（结构：标题/日期/平台/模型 → 模型下载 → 1.1 硬件环境表 → 1.2 镜像与 docker → 1.3 软件环境表含 commits → 任务执行：FlagOS 软件栈 git log → 更新安装(可选) → 执行命令：启动服务+文本/图像/并发客户端，35B-A3B 与 27B 两组）。
- 客户端脚本本地 `/tmp/docclients/`（同容器内 /root 三件套）。
- 环境数据已采集（硬件、镜像 tag、serve 命令已验证、/v1/models 冒烟通过）。**性能小节等慢速问题定性后写**，要如实写 0.4-0.7 tok/s 现状 + 根因 + 黑名单缓解（若实验成立），文档建议读者暂避 top_k/top_p 采样（若 greedy 验证快）。
- 文件名建议：`燧原ZIXIAOC200+empty+vllm0.24.0+vllm-plugin-fl.md`。
- 软件版本表：容器内 `pip show torch torch_gcu triton triton_gcu flag_gems vllm vllm-plugin-fl 2>/dev/null | grep -E "Name|Version"` + `/root/fl` 的 `git log -1`（cf47a04）。

## 7. 关键文件路径速查

| 位置 | 文件 |
|---|---|
| 本机 | `/tmp/enflame020_doc.txt`（模板）、`/tmp/docclients/*`、`/tmp/bltest.py`、`/tmp/ci_infer.log`、`/tmp/ci_serving.log`、`/tmp/ci_monitor.log` |
| S60 宿主机 | `/root/pyspy1.txt`、`/root/pyspy2.txt` |
| ef-doc 容器 | `/root/serve06b.log`（基线 serve）、`/root/serve_bl.log`（黑名单 R1）、`/root/bltest.py`、`/root/client_*.py`、`/root/fl`（插件源码）、`/tmp/flaggems_enable_oplist.txt`（worker enable 证据） |
| 仓库 | `vllm_fl/worker/worker.py`（enable+黑名单消费点）、`vllm_fl/utils.py`（黑名单解析）、`vllm_fl/dispatch/config/enflame.yaml`（flagos_blacklist: scaled_dot_product_attention）、`musa-vllm024-known-issues.md` |
