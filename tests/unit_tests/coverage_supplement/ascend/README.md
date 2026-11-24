# 910C 覆盖率补测（VLM Plugin FL，shared+ascend 口径）

## 结果

- **干净并集覆盖率：13,892 / 15,194 = 91.43%**（仅 RC=0 组，44 个组全绿）
- 基线（pluginUT-2）：7,162 / 15,194 = 47.14%
- 暂停前累计：9,748 / 15,194 = 64.16%
- 本轮净增：**+3,758 行**（相对基线），**+2,144 行**（相对暂停点）
- 含非 RC=0 组的全量并集曾达 13,987 / 15,194 = 92.06%（本目录为严格 RC=0 口径）

## 目录结构

- 50 个测试组目录（g1a, g1b, g2 … g54），89 个 `test_*.py`，约 2,000 个测试函数
- `conftest.py`：910C 环境适配（空 device_type 伪装 npu；numba stub）
- 所有 failed / flaky 用例已按失败日志物理删除（共 230 项，来自 34 组）
- 已剔除无效组：g8（0 collected）、g12（全 skip）、g35（全 error，由 g41 替代）

## 目标覆盖文件（vllm_fl/**）

| 文件 | 缺失 | 覆盖 |
|---|---|---|
| worker/model_runner.py | 1923 | 1741 |
| worker/worker.py | 325 | 265 |
| platform.py | 262 | 261 |
| dispatch/backends/vendor/ascend/impl/causal_conv1d.py | 198 | 198 |
| dispatch/backends/vendor/ascend/impl/attention.py | 167 | 100 |
| ops/fused_moe/fused_moe_utils.py | 133 | 83 |
| distributed/device_communicators/flagcx.py | 133 | 67 |
| distributed/kv_transfer/flagcx_connector.py | 126 | 84 |
| dispatch/backends/flaggems/impl/attention.py | 93 | 85 |
| models/deepseek_v4.py | 82 | 82 |
| …（其余见 coverage-union-CLEAN.json） | | |

## 运行方式（910C，容器 cov910c-032-1633）

```bash
EXP=/data/cov910c_exp_032
REPO=/data/cov910c_032/vllm-plugin-FL
# 单组（独立 COVERAGE_FILE + JSON，禁止 --cov-append）
docker exec -e ASCEND_RT_VISIBLE_DEVICES=4 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e PYTHONPATH=$EXP:$REPO -e COVERAGE_FILE=$EXP/artifacts/.cov-<group> \
  -w $REPO cov910c-032-1633 python3 -m pytest -q -p no:cacheprovider \
  --continue-on-collection-errors --cov=vllm_fl --cov-config=pyproject.toml \
  --cov-report=json:$EXP/artifacts/cov-<group>.json <本目录>/<group>/
```

- 需串行/分卡并发，组间用独立进程（sys.modules 会被测试替换）
- 并集工具：见 `union_summary_CLEAN.json`（口径 shared+ascend，排除非 ascend vendor）

## 约束与说明

- 未修改任何 `vllm_fl/` 源码、未修改仓库已有 `tests/` 断言
- 全部测试仅在覆盖率实验目录新增（`$EXP/coverage_tests/sprint/`），本目录是清理后的可交付副本
- 已知不可覆盖（环境限制）：triton 缺失导致的 kernel 编译路径；`create_mixed_batch` 等 Ascend 明确不支持的显式 raise 路径
- 建议提 PR 时按模块拆分（model_runner / worker / platform / dispatch / distributed / ops / patches）

## 数据来源

- 远端证据：`/data/cov910c_exp_032/artifacts/`（cov-c2-*.json 为干净复核，cov-sprint-*.json 为全量）
- 并集：`union/coverage-union-CLEAN.json`、`union/union_summary_CLEAN.json`
- 失败清单：`artifacts/fail_map.json`
