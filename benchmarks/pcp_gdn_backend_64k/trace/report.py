"""Produce measured trace attribution and a standalone stage comparison."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    result = json.loads((ROOT / "analysis.json").read_text())
    fi, fla = result["flashinfer"], result["triton"]
    previous = json.loads(
        (ROOT.parent / "qwen36_pcp_gdn_backend_64k_20261013/summary.json").read_text()
    )
    measured = {
        name: json.loads((ROOT / name / "result.json").read_text())
        for name in ("flashinfer", "triton")
    }
    assert measured["flashinfer"]["settings"] == measured["triton"]["settings"]
    assert all(
        x["prompt_sha256"] == previous["numerical"]["prompt_sha256"]
        for x in measured.values()
    )
    assert {tuple(r["tokens"]) for x in measured.values() for r in x["captures"]} == {
        (27775,)
    }
    fi_main = next(k for k in fi["kernels"] if k["name"] == "device_kernel")
    fla_state = next(
        k for k in fla["kernels"] if k["stage"] == "chunk_gated_delta_rule_fwd_h"
    )
    assert fi_main["grid"] == [8, 1, 1] and fi_main["calls"] == 360
    assert fla_state["grid"] == [4, 8, 1] and fla_state["calls"] == 360
    assert fi["sm_counts"] == fla["sm_counts"] == [132] * 4
    for mode in (fi, fla):
        for row in mode["per_request_rank"]:
            # No overlapping core CUDA activities in these captures: sums and
            # unions agree, so summing the 30 layer durations is unambiguous.
            assert abs(row["core_gpu_sum_ms"] - row["core_gpu_union_ms"]) < 1e-6

    state = fla["stage_mean_ms"]["chunk_gated_delta_rule_fwd_h"]
    saved = result["core_gpu_ms_saved"]
    ratio = saved / fi["mean_core_gpu_ms"] * 100
    compact = {
        "date": "2026-10-13",
        "scope": "PCP 65536-token single-request prefill plus first token",
        "captures_per_backend": 3,
        "ranks": 4,
        "gdn_layers": 30,
        "core_gpu_ms_flashinfer": fi["mean_core_gpu_ms"],
        "core_gpu_ms_triton": fla["mean_core_gpu_ms"],
        "core_gpu_ms_saved": saved,
        "core_gpu_reduction_pct": ratio,
        "core_span_ms_saved": result["core_gpu_span_ms_saved"],
        "prior_unprofiled_wall_ms_saved": previous["candidate_vs"][
            "pcp_fused_flashinfer"
        ]["saved_ms"],
        "flashinfer_fused_kernel_grid": fi_main["grid"],
        "triton_state_grid": fla_state["grid"],
        "triton_state_ms": state,
        "triton_state_fraction": state / fla["mean_core_gpu_ms"],
        "triton_stages_ms": fla["stage_mean_ms"],
        "flashinfer_adapter_ms": fi["mean_core_gpu_ms"]
        - fi_main["mean_per_request_rank_ms"],
        "interpretation": "Observed grids and timings support insufficient sequence/head parallelism in this FlashInfer SM90 path. Hardware counter occupancy was not measured.",
    }
    (ROOT / "trace_summary.json").write_text(json.dumps(compact, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.8, 4.3))
    components = [
        ("FlashInfer fused core", "#3264A3", fi_main["mean_per_request_rank_ms"], 0),
        ("Adapter", "#BEC5CE", compact["flashinfer_adapter_ms"], 0),
        ("FLA recurrent state", "#72549E", 0, state),
        ("FLA triangular solve", "#E29A38", 0, fla["stage_mean_ms"]["solve_tril"]),
        ("FLA output", "#399678", 0, fla["stage_mean_ms"]["chunk_fwd_o"]),
        (
            "FLA other chunk work",
            "#759CAC",
            0,
            sum(
                fla["stage_mean_ms"][k]
                for k in (
                    "chunk_local_cumsum",
                    "chunk_scaled_dot_kkt_fwd",
                    "recompute_w_u_fwd",
                )
            ),
        ),
    ]
    offsets = [0.0, 0.0]
    for label, color, a, b in components:
        for index, value in enumerate((a, b)):
            if value:
                ax.barh(
                    1 - index,
                    value,
                    left=offsets[index],
                    height=0.52,
                    color=color,
                    label=label,
                    edgecolor="white",
                    linewidth=0.6,
                )
                if value > 5:
                    ax.text(
                        offsets[index] + value / 2,
                        1 - index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=10,
                        fontweight="bold",
                    )
                offsets[index] += value
    for index, total in enumerate(offsets):
        ax.text(
            total + 1.0, 1 - index, f"{total:.2f} ms", va="center", fontweight="bold"
        )
    ax.set_yticks([1, 0], ["FlashInfer", "FLA / Triton"])
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.55, 1.6)
    ax.set_xlabel("GPU execution time (ms), sum over 30 GDN layers per request / rank")
    ax.set_title(
        f"64K x 1 GDN trace: {saved:.2f} ms less GPU time ({ratio:.2f}%)",
        loc="left",
        pad=12,
    )
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.12,
        0.025,
        "3 captured requests x 4 ranks per backend; profiling attribution, not an unprofiled latency benchmark.",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.83, bottom=0.35)
    fig.savefig(ROOT / "gdn_breakdown.svg", bbox_inches="tight")
    svg = ROOT / "gdn_breakdown.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(ROOT / "gdn_breakdown.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# 64K×1 GDN Nsight Trace 分析",
        "",
        "日期：2026-10-13。",
        "",
        "## 结论",
        "",
        f"本轮 trace 中，30 层 GDN 核心算子的 GPU 执行时间从 **{fi['mean_core_gpu_ms']:.3f} ms** 降至 **{fla['mean_core_gpu_ms']:.3f} ms**，减少 **{saved:.3f} ms（{ratio:.2f}%）**。",
        f"这个差值与之前独立、无 profiler 的整模型延迟收益 **{compact['prior_unprofiled_wall_ms_saved']:.3f} ms** 接近。收益主要落在 GDN 计算本身；两个指标分别是 GPU 算子时间和端到端墙钟时间，不作为严格等式或同轮速度比。",
        "实际 kernel grid 证实：FlashInfer 的这条 SM90 GDN 路径只启动 8 个 block；FLA 将状态更新切成 32 个 block，并将可按时间块独立计算的阶段展开为 8192 或 16384 个 block。这支持此前提出的单长序列并行度解释。",
        "",
        "![GDN stage comparison](gdn_breakdown.svg)",
        "",
        "## 实测工作划分",
        "",
        "以下 GPU 时间为每请求、每 rank 的 30 层合计，在 3 个请求 × 4 个 rank 上取均值；没有把四张卡的时间相加当成延迟。block 数是一次 kernel launch 的网格规模，不代表全部同时驻留。",
        "",
        "| 后端 / 阶段 | GPU 时间 ms | 实测 grid | block 数 |",
        "|---|---:|---|---:|",
        f"| FlashInfer 融合核心 | {fi_main['mean_per_request_rank_ms']:.3f} | 8×1×1 | 8 |",
        f"| FlashInfer 外部 exp / 序列长度类型转换 | {compact['flashinfer_adapter_ms']:.3f} | 512×1×1 / 1×1×1 | 512 / 1 |",
        f"| FLA 状态递推 | {state:.3f} | 4×8×1 | 32 |",
        f"| FLA 三角求解（含结果类型转换） | {fla['stage_mean_ms']['solve_tril']:.3f} | 主 kernel 1024×8×1 | 8192 |",
        f"| FLA 输出计算 | {fla['stage_mean_ms']['chunk_fwd_o']:.3f} | 2×1024×8 | 16384 |",
        f"| FLA W/U 重建 | {fla['stage_mean_ms']['recompute_w_u_fwd']:.3f} | 1024×8×1 | 8192 |",
        f"| FLA 块内 KKT | {fla['stage_mean_ms']['chunk_scaled_dot_kkt_fwd']:.3f} | 1024×8×1 | 8192 |",
        f"| FLA 门控累计和 | {fla['stage_mean_ms']['chunk_local_cumsum']:.3f} | 1024×8×1 | 8192 |",
        "",
        "真实输入每 rank：Q/K=[1,65536,4,128]，V=[1,65536,8,128]，BF16；g/beta/initial_state 为 FP32。分块 metadata 包含 1024 个时间块；两后端输入形状、布局和模型配置一致。",
        "",
        "## 为什么单条长请求受益",
        "",
        "FlashInfer 的融合 GDN 核心使用 grid=(8,1,1)、block=(512,1,1)，每 block 128 registers/thread 和 186368 bytes shared memory。机器每卡有 132 个 SM，但这个具体 kernel 只有 8 个 block，所以同时最多可覆盖 8 个 SM。这是由网格得到的上界，不是实际硬件计数器利用率，更不是整模型 GPU 利用率。",
        "FLA 将能按时间块独立做的工作分成单独阶段，向 GPU 提交大量任务；有递归依赖的状态更新仍沿时间推进，但在 value 维继续分成 4 份，实际 grid=(4,8,1)。其 block=(128,1,1)，150 registers/thread、111112 bytes shared memory。",
        "两个实现都有分块，并非只有 FLA 才把 64K 切成 64-token 块；差别是哪些工作在 block 内顺序推进，哪些作为独立 block 调度。FLA 仍有顺序依赖、额外 kernel launch 和中间张量读写，不能将 8192/8 解读为速度比。",
        "本次 GDN core 范围内未记录 CUDA memcpy 活动；输入 Q/K/V 已连续，g/beta/state 已是 FP32，FlashInfer adapter 的额外 exp 和 cu_seqlens 转换合计仅约 0.134 ms/30 层。大块数据复制不是解释这次约 28 ms 差值的主要因素；CUDA 类型转换本身由 kernel 执行，已计入时间。",
        "",
        "## 后续瓶颈",
        "",
        f"FLA 的状态递推仍占 **{state:.3f} ms，约 {state / fla['mean_core_gpu_ms'] * 100:.2f}%** 的 GDN core GPU 时间，是当前最值得进一步分析的阶段；三角求解约 11.13 ms，其次是输出计算约 8.79 ms。",
        "这次未修改这些 kernel。进一步调整 value 分块、寄存器使用或 pipeline，需要重新评估数值与性能；不能从更大的 grid 直接推断一定更快。",
        "",
        "## 采集与归因方法",
        "",
        "保持前一轮 PCP：TP4、4×H100 80GB、BF16、eager、FA3、Triton MoE、融合 Norm，固定原 65536-token prompt，每次只生成 1 token。两后端分别启动进程，5 次预热后安装轻量 NVTX 标记，各抓 3 次请求。共 720 个 GDN layer/rank/request 范围全部关联到 GPU 活动；首 token 均为 27775。",
        "Nsight Systems 2025.3.2，capture-range=cudaProfilerApi，CPU sampling 关闭。每个 GDN core、FLA 子阶段、FlashInfer API、请求及部分整模型阶段均有 NVTX；按进程和 CUDA correlation ID 将 GPU 事件关联到提交它的 CPU API，再归入包含该 API 的 NVTX。没有用 CPU NVTX 长度冒充 GPU 时间。",
        "解析 Nsight globalPid 时保留 VM 标识进行 CUDA correlation 匹配，再提取低 24-bit process ID 与 worker rank 对应。每后端 360 个 core 范围、每 rank 每请求 30 层均完整；core GPU 时间之和与区间并集一致，没有重复计算重叠活动。",
        f"把每层首个至最后 GPU 活动间的间隙也算入，core 跨度合计为 FlashInfer {fi['mean_core_gpu_span_sum_ms']:.3f} ms / FLA {fla['mean_core_gpu_span_sum_ms']:.3f} ms，减少 {result['core_gpu_span_ms_saved']:.3f} ms。",
        "采集顺序是 FlashInfer → FLA，trace 样本量用于解释执行路径；正式速度结论仍引用此前反向复测、每条件 20 次无 profiler 的基准。trace 不具备 Nsight Compute 的 achieved occupancy、SM active 或内存带宽硬件计数，无法把全部时间差精确归因于单一微架构因素。",
        "两套后端的 logits/recurrent state 已确认存在数值差异；此次首 token 相同不扩大之前的质量验证结论。实验仍未设为生产默认。",
        "",
        "## GPU 与复现",
        "",
        "启动前持公共 .gpu.lock，并逐次检查所有 GPU 的进程、显存及利用率；忙则拒绝启动。所有任务正常退出并释放 GPU，之后仅在 CPU 上导出和分析。",
        "原始目录：`/vllm-workspace/validation/qwen36_gdn_trace_64k_20261013`。每后端保存 prefill.nsys-rep、prefill.sqlite、result.json、launch.json、源码快照、analysis.json、layers.csv 与 kernels.csv。",
        "测量实现：experiment/pcp-gdn-backend-64k-20261013，c4a888c。抓取脚本仅给 worker 添加临时 NVTX 包装，不修改模型参数或已安装代码。capture.py / trace_case.py / trace_worker.py 采集；analyze.py 归因；report.py 生成本文和图。依赖前一实验的 gpu_guard.py、timing_worker.py 与两个 benchmark Norm helper。",
        "",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
