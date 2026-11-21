# FlagOS S2 赛道二：MiniCPM5-2B / 沐曦 C500

候选：r1241，单张 C500 64 GB。当前材料用于本地复现和审阅。最终排名及代码合规以组委会复现和审核为准。

## 环境与文件

- Linux、单张沐曦 C500、与比赛匹配的 MACA 驱动和镜像；实测镜像 ID 为 `sha256:86c334f18ab914fa3de5e0ddbf313311d5bfe4b52b4fc6011c91476987988171`，vLLM 0.24.0，质量使用 EvalScope 1.11.1。
- `vllm-plugin-FL/` 包含推理框架、未修改的官方 benchmark 及量化适配器，比较起点为 `flagos-2026-s2` 归档提交 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`。
- `FlagGems/` 包含实际使用的库代码和算子改动。`runtime/` 包含 KV 路径、Triton kernels、项目内 lowering 及必要依赖源码。工作目录不得覆盖已有实验。源码包不包含模型权重、镜像、预编译设备二进制或历史编译缓存。
- 原始 BF16 权重文件的 SHA256 由 `runtime/convert_w8a8_candidate.py` 校验。用镜像内 Python 和单张可见 C500 执行 `python runtime/convert_w8a8_candidate.py --source <原始模型目录> --destination <新W8A8目录>`。转换不使用校准或评测数据，生成后按 `evidence/model_file_hashes.json` 复核实测权重和 tokenizer；转换留痕见 `evidence/weight_conversion.json`。镜像分发映射须在提交审核中核对。

## 性能复现

在同一 Linux 主机上解包后，先运行 `sha256sum -c SOURCE_SHA256SUMS`。启动器使用主机 Python 3 校验模型文件哈希，并接收 Docker socket 文件、镜像 ID、模型目录、新 NVMe 工作目录、GPU 编号：

```bash
bash reproduce_c500.sh "$DOCKER_SOCKET" "$IMAGE_ID" "$MODEL_DIR" "$NEW_NVME_STUDY" "$GPU_INDEX"
```

脚本使用原赛题的 `vllm serve` 参数：端口 9031、模型名 `minicpm`、显存利用率 0.85、模型上限 131072。`VLLM_PLUGINS=fl`；不显式设置批处理覆盖参数。新服务在空目录编译，不向模型发送额外预热请求。正式客户端在同一容器的 `/study` 运行原脚本，保存 `benchmark_results/raw_runs_*.csv` 和 `summary_*.csv`。4K 为 `[4096,1024,64,256]`，16K 为 `[16384,1024,64,128]`，各四轮，汇总后三轮。比较总吞吐和平均 TTFT；输出吞吐作为辅助指标。首轮和失败记录均保留。

本地按官方命令完成八轮，全部请求成功。4K 后三轮平均总吞吐 15819.11 tok/s、平均 TTFT 580.78 ms；16K 为 11162.07 tok/s、19400.58 ms。两项均满足公布的吞吐和 TTFT 门槛。原脚本在同一服务内运行多轮，保留框架默认缓存行为；这些均值不代表独立新服务的首轮表现。

```bash
python3 audit_official_results.py evidence/raw_runs_20261120_030755.csv \
  evidence/summary_20261120_030755.csv evidence/official_run_receipt.json
```

## 质量复现

使用镜像自带的 MATH-500 Level 3 数据和 EvalScope，仅在新工作目录执行。给启动器加最后一个参数 `quality`，它会启动独立的同配置服务，并在同一容器执行以下命令：

```bash
evalscope eval --model minicpm --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY --eval-type openai_api --datasets math_500 \
  --dataset-args '{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}' \
  --eval-batch-size 8 --timeout 3600 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":32768}' \
  --work-dir /study/eval --ignore-errors
```

核对 105 条预测和 105 条评分，至少 100 题正确；错误与达到 token 上限的样本均计入原分母并单独报告。现有质量实测为 101/105，96.19%，生成和评分无报错，2 题达到上限。该轮显式设置了 256 个请求与 2048 个批处理 token，两项均与正式性能命令的运行时解析值一致。

## 证据边界

源码中含项目内的 MetaX Triton lowering 和设备原语绑定。原生 SDK 只参与研发阶段的独立对照；最终路径由实际启动与设备映像证据核对。原算子入口、select 条件以及框架内部 KV 挂接的差异应逐项审阅；不能仅凭 Triton 语言或启动参数一致断言改动通过代码审核。组委会页面：[赛题、基线与评审标准](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn)。
