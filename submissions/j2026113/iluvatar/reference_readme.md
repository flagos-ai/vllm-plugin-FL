# FlagOS S2：MiniCPM5-2B 天数 BI-V150 单卡复现

本包对应 `source_route_r1224`。使用原模型、BF16、原 `TRITON_ATTN` provider 和官方 serve 参数。参赛路径没有开启投机采样或量化。`FlagGems/` 和 `vllm-plugin-FL/` 是指定环境的源码树及本方案改动；额外的四个 vLLM 源码补丁及其安装版本约束见 `reproduction/`。正式技术报告为赛事包根目录 report.pdf 的第二部分。

## 环境

- 官方天数实例，单张 Iluvatar BI-V150（32 GiB），CoreX 4.5.0、IX-ML 4.4.0、Python 3.12.11、PyTorch 2.10.0+corex.4.5.0、vLLM 0.24.0+empty，以及官方镜像预装的 Triton/FlagTree 工具链。
- `vllm-plugin-FL` 基于官方镜像 `flagos-2026-s2` 的 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`；FlagGems 基于镜像 `v5.3.5` 的 `4df52d9168ac55c1c1061cefb4638c570292d896`。请以镜像文件哈希为准，不要更新到同名线上分支或标签的其他提交。
- 复用 `/workspace/MiniCPM5-2B` 与 `/workspace/evalscope-datasets/math_500`。压缩包不附权重、镜像或已编译的设备缓存。启动器会检查已安装源码的基线哈希。

## 从提交包恢复并启动

在作品根目录执行。恢复脚本只写入新建的独立运行目录，预编译缓存是空的；随包提供原实验使用的 FlagGems 调优配置种子，编译仍由官方镜像工具链执行。

```bash
sha256sum -c SHA256SUMS
mkdir /dev/shm/ts_review
tar -xzf reproduction/MiniCPM5_Tianshu_source_r1224_portable_r1232.tar.gz -C /dev/shm/ts_review
python3 /dev/shm/ts_review/portable_source_r1232/runtime.py prepare --runtime /dev/shm/ts_review/run
python3 /dev/shm/ts_review/portable_source_r1232/runtime.py verify --runtime /dev/shm/ts_review/run
python3 /dev/shm/ts_review/portable_source_r1232/runtime.py serve --runtime /dev/shm/ts_review/run --model /workspace/MiniCPM5-2B
```

`prepare` 要求 `--runtime` 不存在；如已有目录，请另选较短路径。`verify` 成功时输出 `all_source_hashes_exact: true`。`serve` 启动后，运行目录 `launch.json` 记录与赛题一致的实际命令：

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
```

在另一终端运行 `curl -fsS http://127.0.0.1:9031/health` 检查服务。调优与图缓存留在独立目录。评测时只运行一个服务。

## 官方吞吐与首 token 延迟

在一个新的结果目录运行原脚本；工作目录只决定它写入 `benchmark_results/` 的位置。脚本 SHA256：`0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565`。

```bash
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

两种场景应各产生 4 条成功原始记录，汇总取后三轮。赛事指标是 **Total Token Throughput (tok/s)**，同时比较平均 TTFT。输出吞吐另列。保留首轮、失败状态和汇总文件，不能用历史单轮或诊断成绩代替。

## 官方质量

```bash
evalscope eval --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY --eval-type openai_api --datasets math_500 \
  --dataset-args '{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}' \
  --eval-batch-size 8 --timeout 3600 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":32768}' \
  --work-dir /dev/shm/ts_review/eval --ignore-errors
```

检查 105 个 Level 3 题目、生成及评分错误、截断和最终准确率。已审计的单次同版本结果为 103/105（98.10%）、零生成或评分错误、零截断。随机采样的本次成绩不保证逐枚复现。

本地正式验收的原始 CSV、官方汇总、运行凭据及独立复算见 `evidence/standard/`。源码绑定与编译记录见 `evidence/runtime/`，质量原始汇总与逐题审计见 `evidence/quality/`。
技术报告引用的研发与迁移证据见 `evidence/engineering/`。
