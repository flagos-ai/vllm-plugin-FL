# Iluvatar blacklist 单算子复现

每个算子一个独立 Python 文件，只依赖 PyTorch 和 FlagGems；无需模型、服务或其他复现文件。先执行原生对照，再直接调用 Gems 实现，避免黑名单导致实际仍执行原生。`slice` 使用仅启用该算子的 dispatcher。所有文件打印环境和种子。

实测环境：Iluvatar BI-V150，PyTorch 2.10.0，FlagGems 5.5.0.dev0（现有 gate 环境）。结论只适用于该环境，不代表其他版本。

```bash
# 在已安装依赖的新容器内执行；按实际情况调整仓库路径和空闲卡号。
cd /workspace/gate-src/vllm-plugin-FL/tools/reproducers/iluvatar_blacklist
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=4 COREX_VISIBLE_DEVICES=4 \
  python repro_linear.py
```

其他机器使用相应环境的 Python。这里必须清除继承的 `PYTHONPATH`，否则系统 site-packages 会覆盖虚拟环境内的 FlagGems。检查脚本打印的 `source`。可加 `--seed 43` 改变数据；`--small` 仅为两个清零文件提供 1 MiB 对照，默认清零输入为 16 GiB int8。每个清零文件先释放原生输入，再分配 Gems 输入；至少需要约 17 GiB 空闲显存。

退出码 1 表示脚本遇到异常或断言失败；结合 `NATIVE_PASS` 和 traceback 判断是否复现目标问题。环境导入失败也会非零，不能算复现。退出码 0 和 `NO_FAILURE_REPRODUCED` 表示当前输入未复现；修复后测试应正常退出。

| 文件 | 输入/检查 | 已观察现象 |
|---|---|---|
| `repro_zeros.py` | 16 GiB int8 | Gems kernel launch `IX_ERROR_INVALID_VALUE` |
| `repro_zero_.py` | 16 GiB int8 原地清零 | 同上 |
| `repro_linear.py` | BF16 `[17,1152]`、权重 `[3456,1152]`、bias | 缺少 `BLOCK_SIZE_M/N/K` |
| `repro_mm.py` | BF16 `[17,5120] @ [5120,4096]`，右矩阵转置视图 | 不识别 `SPLIT_K` |
| `repro_mm_out.py` | 同上，显式 out | 不识别 `SPLIT_K` |
| `repro_slice.py` | 8 元素 bool 的 `[1:5]` view 写入 | `unsupported dtype torch.bool` |
| `repro_native_layer_norm.py` | BF16 `[256,1152]`；比较完整三元组 | mean/rstd 应为 FP32，实际 BF16 |
| `repro_addmm.py` | BF16 `[256,1152] @ [1152,4608]` 加 bias | 见下方精度说明 |
| `repro__scaled_dot_product_flash_attention.py` | BF16 `[B,H,N,D]=[1,4,256,72]` | 与 math SDPA 明显不一致，max abs 4.5625 |
| `repro__flash_attention_forward.py` | 同上，内部入口转为 `[B,N,H,D]` | 同上 |

`native_layer_norm` 的用例证明完整 ATen 返回契约不兼容；它没有证明这一次的主输出存在严重误差，也不能单凭统计量 dtype 解释历史 ViT 前向误差。

`addmm` 使用 PyTorch 2.10 默认 BF16 容差（rtol=0.016、atol=1e-5），1947/1179648 个元素不一致；全张量 max abs 为 1.0。脚本同时计算 CPU FP64 对照：原生 max abs=1.0，Gems max abs=0.25。因而本用例证明与原生的精度一致性问题，不能断言 Gems 数学结果更差。先前宽松容差 atol=0.05、rtol=0.02 下此输入通过；不应把默认容差下的不一致夸大为严重算术错误。CPU FP64 对照只作诊断，避免设备 FP64 路径不可靠。

两个 flash 用例使用原生 math SDPA 作为参考，内部入口显式转换布局，不受 native flash 重新分派影响；采用较宽松的 atol=0.05、rtol=0.02 仍失败。

实际运行日志和逐项退出码存放在 `/workspace/iluvatar-blacklist-repros-results/`（`results.json` 与每文件同名 `.log`），不纳入源码提交。所有十项均先通过原生运行，随后目标实现报错或一致性断言失败。脚本没有预期失败装饰器，便于修复后直接重跑。

容差依据：https://docs.pytorch.org/docs/2.10/testing.html#torch.testing.assert_close
