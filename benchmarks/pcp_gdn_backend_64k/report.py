"""Validate all six timing cells and separately summarize numerical differences."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TAGS = ("r1_tp", "r1_pcp", "r2_pcp", "r2_tp")


def percentile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lower = int(pos)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (
        pos - lower
    )


def numerical_summary():
    data = json.loads((ROOT / "correctness.json").read_text())
    assert data["completed"]
    core = [
        row
        for rank in data["phases"]["core_audit"]["ranks"]
        for row in rank["core_checks"]
    ]
    assert len(core) == 120
    assert all(
        r["tokens"] == 65536 and r["initial_unchanged"] and r["chunk_metadata_present"]
        for r in core
    )
    result = {
        "core_comparisons": len(core),
        "core": {},
        "logits": {},
        "free_tokens_equal": data["free_tokens_equal"],
        "free_logprobs_equal": data["free_logprobs_equal"],
        "limited_retrieval_pass": data["limited_retrieval_pass"],
        "retrieval_cases": len(data["needles"]),
        "prompt_sha256": data["prompt_sha256"],
    }
    for key in ("output", "state"):
        assert all(r[key]["finite"] for r in core)
        result["core"][key] = {
            "exact_count": sum(r[key]["equal"] for r in core),
            "max_abs": max(r[key]["max_abs"] for r in core),
            "max_relative_rms": max(r[key]["relative_rms"] for r in core),
        }
    for phase in ("repeat", "teacher"):
        ranks = data["phases"][phase]["ranks"]
        assert len(ranks) == 4 and all(len(r["logit_checks"]) == 128 for r in ranks)
        # TP all-gather duplicates full logits across ranks. Report unique
        # rank-0 generation steps, while verifying finiteness on every rank.
        assert all(r["finite"] for rank in ranks for r in rank["logit_checks"])
        rows = ranks[0]["logit_checks"]
        result["logits"][phase] = {
            "steps": len(rows),
            "exact_steps": sum(r["equal"] for r in rows),
            "argmax_equal_steps": sum(r["argmax_equal"] for r in rows),
            "max_abs": max(r["max_abs"] for r in rows),
            "mean_rms": statistics.mean(r["rms"] for r in rows),
            "max_relative_rms": max(r["relative_rms"] for r in rows),
            "mean_kl": statistics.mean(r["kl_reference_to_candidate"] for r in rows),
            "max_kl": max(r["kl_reference_to_candidate"] for r in rows),
            "first_token": rows[0],
        }
    return result


def main():
    numerical = numerical_summary()
    results = {
        tag: json.loads((ROOT / tag / "result.json").read_text()) for tag in TAGS
    }
    manifests = {
        tag: json.loads((ROOT / tag / "launch.json").read_text()) for tag in TAGS
    }
    assert all(r["completed"] for r in results.values())
    assert all(r["settings"] == results[TAGS[0]]["settings"] for r in results.values())
    assert all(
        r["prompt_sha256"] == numerical["prompt_sha256"] for r in results.values()
    )
    first_tokens = set()
    cells, sample_rows = {}, []
    for tag, result in results.items():
        manifest = manifests[tag]
        assert not manifest["gpu_before"]["processes"]
        for name, digest in manifest["hashes"].items():
            assert (
                hashlib.sha256((ROOT / tag / "source" / name).read_bytes()).hexdigest()
                == digest
            )
            assert (
                hashlib.sha256(
                    (Path(manifest["source"]) / "vllm_fl/cp" / name).read_bytes()
                ).hexdigest()
                == digest
            )
        for name in (
            "timing.py",
            "timing_worker.py",
            "run_timing.py",
            "benchmark_fused_norm.py",
            "benchmark_norm_ablation.py",
            "gpu_guard.py",
        ):
            assert (ROOT / tag / name).read_bytes() == (ROOT / name).read_bytes()
        for variant, data in result["cells"].items():
            assert len(data["warmups"]) == 3 and len(data["samples"]) == 10
            assert len(data["core_audit"]) == 4
            assert all(len(rank) == 30 for rank in data["core_audit"])
            assert all(
                r["tokens"] == 65536 and r["sequences"] == 1 and r["chunk_metadata"]
                for rank in data["core_audit"]
                for r in rank
            )
            key = result["mode"] + "_" + variant
            cell = cells.setdefault(key, {"values": [], "round_medians_ms": {}})
            values = [r["wall_ms"] for r in data["samples"]]
            cell["values"].extend(values)
            cell["round_medians_ms"][tag] = statistics.median(values)
            for row in data["warmups"] + data["samples"]:
                assert len(row["tokens"]) == 1 and not row["corrupted"]
                first_tokens.add(tuple(row["tokens"]))
            sample_rows.extend(
                {"tag": tag, "condition": key, "sample": i, "wall_ms": value}
                for i, value in enumerate(values)
            )
    assert len(first_tokens) == 1
    for cell in cells.values():
        values = cell.pop("values")
        assert len(values) == 20
        cell.update(
            samples=len(values),
            median_ms=statistics.median(values),
            min_ms=min(values),
            max_ms=max(values),
            p10_ms=percentile(values, 0.1),
            p90_ms=percentile(values, 0.9),
        )
    candidate = cells["pcp_fused_triton"]["median_ms"]
    comparisons = {}
    for baseline in (
        "pcp_fused_flashinfer",
        "tp_native_flashinfer",
        "tp_fused_flashinfer",
        "tp_native_triton",
        "tp_fused_triton",
    ):
        reference = cells[baseline]["median_ms"]
        comparisons[baseline] = {
            "saved_ms": reference - candidate,
            "latency_reduction_pct": (1 - candidate / reference) * 100,
            "speedup": reference / candidate,
        }
    threshold = 0.75 * cells["tp_native_flashinfer"]["median_ms"]
    summary = {
        "metric": "LLM.generate wall time: 65536-token prefill plus first real token, one request",
        "date": "2026-10-13",
        "cells": cells,
        "candidate_vs": comparisons,
        "native_tp_25pct_threshold_ms": threshold,
        "candidate_below_25pct_threshold_by_ms": threshold - candidate,
        "target_met": candidate <= threshold,
        "first_token_equal_across_timing": True,
        "source_hashes_verified": True,
        "numerical": numerical,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (ROOT / "samples.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("tag", "condition", "sample", "wall_ms")
        )
        writer.writeheader()
        writer.writerows(sample_rows)
    write_report(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "numerical"}, indent=2))


def write_report(summary):
    c, n = summary["cells"], summary["numerical"]
    comparisons = summary["candidate_vs"]
    labels = {
        "tp_native_flashinfer": "TP / 原生 Norm / FlashInfer GDN",
        "tp_fused_flashinfer": "TP / 融合 Norm / FlashInfer GDN",
        "tp_native_triton": "TP / 原生 Norm / Triton GDN",
        "tp_fused_triton": "TP / 融合 Norm / Triton GDN",
        "pcp_fused_flashinfer": "当前 PCP / 融合 Norm / FlashInfer GDN",
        "pcp_fused_triton": "实验 PCP / 融合 Norm / Triton GDN",
    }
    lines = [
        "# 64K×1 GDN 后端实验",
        "",
        "日期：2026-10-13。",
        "",
        "## 实测结果",
        "",
        "| 条件 | 中位数 ms | P10–P90 ms | 两轮中位数 ms |",
        "|---|---:|---:|---:|",
    ]
    for key, label in labels.items():
        row = c[key]
        rounds = " / ".join(
            f"{value:.3f}" for value in row["round_medians_ms"].values()
        )
        lines.append(
            f"| {label} | {row['median_ms']:.3f} | {row['p10_ms']:.3f}–{row['p90_ms']:.3f} | {rounds} |"
        )
    lines += [
        "",
        f"实验 PCP 比同轮原生 TP 延迟降低 **{comparisons['tp_native_flashinfer']['latency_reduction_pct']:.3f}%**；比当前 PCP 再降低 **{comparisons['pcp_fused_flashinfer']['latency_reduction_pct']:.3f}%**（{comparisons['pcp_fused_flashinfer']['saved_ms']:.3f} ms）。",
        f"原生 TP 的 25% 降延迟门槛为 {summary['native_tp_25pct_threshold_ms']:.3f} ms，实验值与门槛的裕量为 {summary['candidate_below_25pct_threshold_by_ms']:.3f} ms（正数代表达标）。",
        f"TP 也应用相同 Norm 和 GDN 优化后，PCP 的优势为 **{comparisons['tp_fused_triton']['latency_reduction_pct']:.3f}%**。后端切换属于两者均可使用的通用优化，不能把相对未经这些优化的原生 TP 的全部差距解释成 PCP 独有收益。",
        "",
        "指标为一次 LLM.generate 的完整 65536-token Prefill 加首个真实 token 的墙钟时间；不是 HTTP TTFT、长输出 Decode 吞吐或混合并发服务。全部调用均为单请求。",
        "",
        "## 数值与有限任务检查",
        "",
        "仅性能达标不等于无损优化。以下测试发生在独立的带诊断进程；计时进程不包含这些比较、logit hook 或 profiler。",
        f"- 120 次真实 GDN 层/rank 同输入比较，输出和最终 recurrent state 均有限，输入 initial state 均未被改写。输出最大绝对差 {n['core']['output']['max_abs']:.8f}，状态最大绝对差 {n['core']['state']['max_abs']:.8f}；两者均非逐位一致。",
        f"- 输出/状态的最大相对 RMS 差分别为 {100 * n['core']['output']['max_relative_rms']:.4f}% / {100 * n['core']['state']['max_relative_rms']:.4f}%。这是算子级误差，不是任务准确率。",
        f"- 同后端重复对照：{n['logits']['repeat']['exact_steps']}/128 步完整 logits 逐位一致。换后端后，同一 token 历史下 {n['logits']['teacher']['exact_steps']}/128 步 logits 逐位一致，{n['logits']['teacher']['argmax_equal_steps']}/128 步 argmax 一致。",
        f"- 换后端 logits 最大绝对差 {n['logits']['teacher']['max_abs']:.6f}，平均 RMS 差 {n['logits']['teacher']['mean_rms']:.6f}，平均/最大 KL(reference‖candidate) 为 {n['logits']['teacher']['mean_kl']:.6f} / {n['logits']['teacher']['max_kl']:.6f}。",
        f"- 首 token logits 最大绝对差 {n['logits']['teacher']['first_token']['max_abs']:.6f}；相对 RMS 差 {100 * n['logits']['teacher']['first_token']['relative_rms']:.4f}%。",
        f"- 另行不强制历史的 128-token 自由生成：tokens 一致={n['free_tokens_equal']}；保存的 top-5 logprobs 一致={n['free_logprobs_equal']}。",
        f"- 2 个 seed × 3 个深度的 64K 六位口令检索：两后端均通过={n['limited_retrieval_pass']}（每后端 6/6）。",
        "完整 logits 在四个 rank 重复检查；上述 128 步按 rank 0 计数，不把重复 rank 当成独立任务样本。同历史比较会返回参考 logits 强制保持输入一致，因此那一阶段的 token 一致不作为自由生成一致的证据。",
        "这是一条长提示词和六个检索样本的有限筛查，不能证明广泛质量等价；尚未覆盖非零历史的实际模型 chunked prefill、长 Decode、混合请求或不同形状。本分支仅保留显式实验选择，不改默认派发。",
        "",
        "## 实验控制",
        "",
        "GPU 0–3，4×H100 80GB；/data/model Qwen3.6-35B-A3B（Qwen3.5 MoE 架构），BF16 / TP4 / eager / FA3 / Triton MoE；max_model_len=66560、max_num_batched_tokens=65536、max_num_seqs=64、KV cache=4 GiB/GPU，关闭 prefix cache，seed=42。max_num_seqs 是容量上限，实际始终只有 1 个请求。",
        "TP → PCP → PCP → TP 四个进程；第二轮反转内部 Norm/后端顺序。每段每条件 3 次预热、10 次正式样本，六条件各 20 次，共 120 个正式样本，保留所有结果，不裁异常值。",
        "每条件计时前独立诊断确认全部 30 层 GDN 每 rank 恰好各调用一次，输入为 65536 tokens、1 个序列且 chunk metadata 有效；随后恢复原生 bound method，再开始预热/计时。Full Attention 确认 10 层 FA3，PCP prefill 次数与实际调用数一致。",
        "两边 Norm 对齐使用先前 Norm 消融的诊断 helper（101 个 Gemma Norm），仅修改导入路径；fused_norm kernel 与该已验证实现相同。GDN gated Norm 不属于此切换。普通 PCP 默认仍使用既有显式 Norm 路径。",
        "每进程启动前持共享 GPU lock 并检查所有 GPU 进程、显存、利用率；有占用即拒绝启动。launch.json、source/ 和每段脚本保留环境、源代码哈希、GPU 空闲快照，报告重新校验全部源文件。",
        "## 代码与复现",
        "",
        "工作树：/vllm-workspace/vllm-plugin-FL-pcp-gdn-backend；分支 experiment/pcp-gdn-backend-64k-20261013，基于 d5798be。之前三项优化和已安装 vLLM 均未修改。",
        'vllm_fl/cp/gdn_backend_probe.py 仅按显式调用在当前 worker 的原生 ChunkGatedDeltaRule.forward_cuda（FlashInfer）与 forward_native（Triton/FLA）之间选择，支持恢复原派发，没有注册或自动启用。原生 vLLM 同样提供 additional_config={"gdn_prefill_backend":"triton"} 启动选项。Decode 的 recurrent kernel 未被切换。',
        "正确性脚本：correctness.py、check_worker.py、gpu_guard.py。性能脚本：run_timing.py、timing.py、timing_worker.py、benchmark_norm_ablation.py、benchmark_fused_norm.py。report.py 汇总并校验结果。",
        "复跑时将这些脚本放入 /vllm-workspace/validation 下新的同级目录，保留 qwen36_64k_20260928/prompt_ids.json 与公共 GPU lock 路径；run_timing.py 仅在空闲时启动。脚本拒绝覆盖已有结果。",
        "原始结果：correctness.json、r1_tp/result.json、r1_pcp/result.json、r2_pcp/result.json、r2_tp/result.json；汇总：summary.json、samples.csv。",
        "",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
