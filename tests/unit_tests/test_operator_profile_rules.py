import csv
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tools.operator_profile.extract_operator_shapes import (
    format_operator_time_percent,
    operator_list_rows,
)
from tools.operator_profile.rule.rule_coverage import (
    FlagOSEvidence,
    classify_coverage,
    read_flagos_evidence,
)
from tools.operator_profile.rule.rule_map import (
    kernel_callable_identity_name,
    kernel_signature,
    operator_descriptor,
    staged_kernel_family_name,
)


def test_operator_time_percent_uses_hundredth_percent_resolution():
    assert format_operator_time_percent(123, 10_000) == "1.23"
    assert format_operator_time_percent(1, 100_000) == "<0.01"
    assert format_operator_time_percent(1, 0) == "<0.01"


def test_kernel_callable_identity_ignores_launch_specialization():
    kernel = "void backend::kernel_rank_3<float, 128>(float*, int)"

    assert kernel_callable_identity_name(kernel) == "backend::kernel<float, 128>"


def test_kernel_identity_preserves_semantic_functor_types():
    fill = "void backend::vectorized_elementwise_kernel<FillFunctor<int>, 128>(int*)"
    sigmoid = "void backend::vectorized_elementwise_kernel<SigmoidFunctor, 256>(int*)"

    assert kernel_callable_identity_name(fill) == (
        "backend::vectorized_elementwise_kernel<FillFunctor<int>, 128>"
    )
    assert kernel_callable_identity_name(sigmoid) == (
        "backend::vectorized_elementwise_kernel<SigmoidFunctor, 256>"
    )
    nested_rank = "void backend::kernel<FillFunctor_rank_3<int>, 128>(int*)"
    assert kernel_callable_identity_name(nested_rank) == (
        "backend::kernel<FillFunctor_rank_3<int>, 128>"
    )


def test_kernel_signature_splits_only_top_level_template_arguments():
    signature = kernel_signature(
        "void backend::kernel<4, Functor<std::array<char*, 3ul>>, false>(int*)"
    )

    assert signature.symbol == "backend::kernel"
    assert signature.template_args == (
        "4",
        "Functor<std::array<char*, 3ul>>",
        "false",
    )


def test_custom_api_groups_template_specializations_but_not_other_apis():
    first = operator_descriptor(
        "extension::metadata", "void backend::prepare_kernel<1, false>(int*)"
    )
    second = operator_descriptor(
        "extension::metadata", "void backend::prepare_kernel<4, false>(int*)"
    )
    different_api = operator_descriptor(
        "extension::other", "void backend::prepare_kernel<4, false>(int*)"
    )

    assert first[2] == second[2]
    assert first[2] != different_api[2]


def test_generic_launch_wrapper_keeps_distinct_implementations():
    first = operator_descriptor(
        "extension::forward", "void cutlass::device_kernel<ForwardKernel<1>>(int*)"
    )
    second = operator_descriptor(
        "extension::forward", "void cutlass::device_kernel<CombineKernel<1>>(int*)"
    )

    assert first[2] != second[2]


def test_unattributed_kernel_without_a_rule_keeps_its_full_template():
    first = operator_descriptor("null", "void backend::kernel<1>(int*)")
    second = operator_descriptor("null", "void backend::kernel<2>(int*)")

    assert first[2] != second[2]


def test_paged_mqa_metadata_capacity_specializations_share_an_operator():
    first = operator_descriptor(
        "null",
        "void deep_gemm::sched::smxx_paged_mqa_logits_metadata<"
        "32u, 256u, 132u, false>(int*)",
    )
    second = operator_descriptor(
        "null",
        "void deep_gemm::sched::smxx_paged_mqa_logits_metadata<"
        "64u, 256u, 132u, false>(int*)",
    )

    assert first[2] == second[2]


def test_fp8_gemm_groups_all_template_specializations():
    template_args = [
        "(cute::UMMA::Major)0",
        "0u",
        "512u",
        "4096u",
        "1u",
        "128u",
        "128u",
        "128u",
        "128u",
        "128u",
        "128u",
        "5u",
        "128u",
        "256u",
        "1u",
        "false",
        "132u",
        "(deep_gemm::GemmType)0",
        "deep_gemm::epilogue::EpilogueIdentity",
    ]

    def identity(args: list[str]) -> tuple[str, ...]:
        kernel = (
            "void deep_gemm::sm90_fp8_gemm_1d2d_impl<" + ", ".join(args) + ">(float*)"
        )
        return operator_descriptor("null", kernel)[2]

    specialized = template_args.copy()
    specialized[2] = "1024u"
    specialized[15] = "true"
    batched = template_args.copy()
    batched[17] = "(deep_gemm::GemmType)4"

    assert identity(template_args) == identity(specialized)
    assert identity(template_args) == identity(batched)


def test_deep_gemm_function_name_groups_across_api_attribution():
    attributed = operator_descriptor(
        "vllm::gemm", "void deep_gemm::sm90_fp8_gemm_1d2d_impl<1>(float*)"
    )
    unattributed = operator_descriptor(
        "null", "void deep_gemm::sm90_fp8_gemm_1d2d_impl<4>(float*)"
    )
    another_function = operator_descriptor(
        "null", "void deep_gemm::sm90_fp8_paged_mqa_logits<1>(float*)"
    )

    assert attributed[2] == unattributed[2]
    assert attributed[2] != another_function[2]
    assert attributed[2] == (
        "kernel_family",
        "deep_gemm::sm90_fp8_gemm_1d2d_impl",
    )


def test_direct_kernel_families_group_template_specializations():
    for symbol in (
        "marlin_moe_wna16::Marlin",
        "cublasLt::splitKreduce_kernel",
    ):
        first = operator_descriptor("null", f"void {symbol}<1, float>(int*)")
        second = operator_descriptor("null", f"void {symbol}<2, float>(int*)")
        assert first[2] == second[2]


def test_triton_kernel_is_classified_without_an_operator_mapping():
    kernel = "triton_poi_fused_add_mul_0"

    name, kind, identity = operator_descriptor("null", kernel)

    assert name == "null"
    assert kind == "triton_compiled"
    assert identity == ("compile_kernel", kernel)


def test_operator_list_merges_attributed_and_unattributed_kernel_rows():
    kernel = "void extension::kernel<float>(float*)"
    rows = operator_list_rows(
        [
            {
                "operator_name": "extension::operator",
                "kernel_name": kernel,
                "kernel_time_us": 2.0,
            },
            {
                "operator_name": "null",
                "kernel_name": kernel,
                "kernel_time_us": 3.0,
            },
        ]
    )

    assert rows == [
        {
            "operator_id": 1,
            "operator_name": "extension::operator",
            "operator_kind": "custom",
            "kernel_name": kernel,
            "kernel_time_percent(%)": "100.00",
        }
    ]


def test_operator_list_keeps_ambiguous_same_kernel_apis_separate():
    rows = operator_list_rows(
        [
            {
                "operator_name": "extension::first",
                "kernel_name": "shared_kernel",
                "kernel_time_us": 1.0,
            },
            {
                "operator_name": "extension::second",
                "kernel_name": "shared_kernel",
                "kernel_time_us": 2.0,
            },
            {
                "operator_name": "null",
                "kernel_name": "shared_kernel",
                "kernel_time_us": 3.0,
            },
        ]
    )

    assert len({row["operator_id"] for row in rows}) == 3
    assert {row["operator_name"] for row in rows} == {
        "extension::first",
        "extension::second",
        "null",
    }


def test_kernel_family_rule_rejects_multiple_known_apis():
    with pytest.raises(ValueError, match="kernel family matches multiple APIs"):
        operator_list_rows(
            [
                {
                    "operator_name": "extension::first",
                    "kernel_name": "void marlin_moe_wna16::Marlin<1>(int*)",
                    "kernel_time_us": 1.0,
                },
                {
                    "operator_name": "extension::second",
                    "kernel_name": "void marlin_moe_wna16::Marlin<2>(int*)",
                    "kernel_time_us": 2.0,
                },
            ]
        )


def test_nvjet_kernels_share_the_aten_mm_operator():
    attributed = operator_descriptor("aten::mm", "nvjet_tst_128x64_TNT")
    unattributed = operator_descriptor("null", "nvjet_tst_64x8_TNN")
    tss_attributed = operator_descriptor("aten::mm", "nvjet_tss_128x64_TNT")
    tss_unattributed = operator_descriptor("null", "nvjet_tss_64x8_TNN")

    assert attributed == (
        "aten::mm",
        "aten",
        ("operator", "aten", "aten::mm"),
    )
    assert unattributed == attributed
    assert tss_attributed == attributed
    assert tss_unattributed == attributed


def test_moe_align_stages_are_distinct_custom_operators():
    first = operator_descriptor(
        "vllm::moe_forward_shared", "moe_align_block_size_stage1"
    )
    second = operator_descriptor("null", "moe_align_block_size_stage2_vec")

    assert first == (
        "moe_align_block_size_stage1",
        "custom",
        ("moe_align_block_size_stage", "moe_align_block_size_stage1"),
    )
    assert second == (
        "moe_align_block_size_stage2_vec",
        "custom",
        ("moe_align_block_size_stage", "moe_align_block_size_stage2_vec"),
    )
    assert first[2] != second[2]
    assert staged_kernel_family_name("moe_align_block_size_stage1") is None
    assert staged_kernel_family_name("moe_align_block_size_stage2_vec") is None


def test_numbered_stages_share_a_generic_kernel_family():
    first = "backend_pipeline_stage1"
    second = "backend_pipeline_stage2_vec"

    assert staged_kernel_family_name(first) == "backend_pipeline"
    assert staged_kernel_family_name(second) == "backend_pipeline"
    assert operator_descriptor("extension::pipeline", first)[2] == (
        "staged_kernel_family",
        "backend_pipeline",
    )
    assert operator_descriptor("extension::pipeline", second)[2] == (
        "staged_kernel_family",
        "backend_pipeline",
    )


def test_collective_and_fused_collective_are_distinguished_by_semantics():
    pure = operator_descriptor(
        "extension::all_reduce",
        "void backend::all_reduce_kernel(float*)",
    )
    fused = operator_descriptor(
        "extension::fused_allreduce_norm",
        "void backend::allreduce_norm_kernel(float*)",
    )

    assert pure[1] == "communication"
    assert fused[1] == "fused_communication_compute"


def test_elementwise_broadcast_is_not_classified_as_communication():
    descriptor = operator_descriptor("aten::mul", "mul_broadcast_2d_kernel")

    assert descriptor[1] == "aten"


def test_aten_coverage_uses_flaggems_enable_oplist():
    evidence = FlagOSEvidence(
        aten_apis=frozenset({"aten::add"}),
        fused_apis=frozenset(),
        lines=(),
    )

    decision = classify_coverage(
        operator_names={"aten::add"},
        operator_kinds={"aten"},
        kernel_names={"void at::native::add_kernel(float*)"},
        evidence=evidence,
    )

    assert decision.covered is True
    assert decision.flagos_type == "flaggems"
    assert decision.evidence.startswith("flaggems_enable_oplist")


def test_void_communication_kernel_uses_communication_policy():
    decision = classify_coverage(
        operator_names={"c10d::all_reduce"},
        operator_kinds={"communication"},
        kernel_names={"void nccl::all_reduce_kernel(float*)"},
        evidence=FlagOSEvidence(frozenset(), frozenset(), ()),
    )

    assert decision.covered is False
    assert decision.flagos_type == "none"
    assert decision.evidence == (
        "communication is in the denominator; no FlagCX evidence"
    )


def test_triton_communication_does_not_bypass_flagcx_evidence():
    decision = classify_coverage(
        operator_names={"c10d::all_reduce"},
        operator_kinds={"communication"},
        kernel_names={"triton_all_reduce"},
        evidence=FlagOSEvidence(frozenset(), frozenset(), ()),
    )

    assert decision.covered is False
    assert decision.flagos_type == "none"


def test_flaggems_registry_maps_callable_without_module_false_positive(
    tmp_path, monkeypatch
):
    def zeros():
        pass

    def zero_():
        pass

    def softmax_out():
        pass

    def true_divide_():
        pass

    zeros.__module__ = zero_.__module__ = "flag_gems.ops.zeros"
    softmax_out.__module__ = "flag_gems.ops.softmax"
    true_divide_.__module__ = "flag_gems.ops.div"
    flag_gems = ModuleType("flag_gems")
    flag_gems.FULL_CONFIG_BY_FUNC = {
        "zeros": [("zeros", zeros)],
        "zero_": [("zero_", zero_)],
        "softmax": [("_softmax", softmax_out)],
        "true_divide_": [("div_.Tensor", true_divide_)],
    }
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    evidence_path = tmp_path / "flaggems_enable_oplist.txt"
    evidence_path.write_text(
        "[DEBUG] flag_gems.ops.zeros.zero_: GEMS ZERO_\n"
        "[DEBUG] flag_gems.ops.softmax.softmax: GEMS SOFTMAX\n"
        "[DEBUG] flag_gems.ops.div.true_divide_: GEMS TRUE_DIVIDE_\n",
        encoding="utf-8",
    )

    evidence = read_flagos_evidence(evidence_path)

    assert evidence.aten_apis == {"aten::zero_", "aten::_softmax", "aten::div_"}
    assert "aten::zeros" not in evidence.aten_apis


def test_missing_flaggems_evidence_leaves_aten_undetermined():
    decision = classify_coverage(
        operator_names={"aten::sum"},
        operator_kinds={"aten"},
        kernel_names={"sum_kernel"},
        evidence=read_flagos_evidence(None),
    )

    assert decision.covered is None
    assert decision.flagos_type == ""


def test_null_operator_names_do_not_create_plugin_associations(tmp_path):
    baseline = tmp_path / "baseline.csv"
    plugin = tmp_path / "plugin.csv"
    for path, kernel in ((baseline, "baseline_kernel"), (plugin, "plugin_kernel")):
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "operator_id",
                    "operator_name",
                    "operator_kind",
                    "kernel_name",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "operator_id": 1,
                    "operator_name": "null",
                    "operator_kind": "unattributed",
                    "kernel_name": kernel,
                }
            )
    report = tmp_path / "report.csv"
    script = (
        Path(__file__).resolve().parents[2]
        / "tools/operator_profile/generate_flagos_coverage.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            str(baseline),
            "--plugin",
            str(plugin),
            "--output",
            str(report),
        ],
        check=True,
    )

    with report.open(encoding="utf-8", newline="") as source:
        row = next(csv.DictReader(source))
    assert row["plugin_operator_id"] == "[]"


def test_aten_without_enable_op_evidence_is_not_covered():
    decision = classify_coverage(
        operator_names={"aten::sum"},
        operator_kinds={"aten"},
        kernel_names={"sum_kernel"},
        evidence=FlagOSEvidence(frozenset(), frozenset(), ()),
    )

    assert decision.covered is False
    assert decision.flagos_type == "none"
    assert decision.evidence == "ATen API is absent from flaggems_enable_oplist"


def test_operator_without_decisive_evidence_has_empty_coverage_fields():
    decision = classify_coverage(
        operator_names={"extension::operator"},
        operator_kinds={"custom"},
        kernel_names={"extension_kernel"},
        evidence=FlagOSEvidence(frozenset(), frozenset(), ()),
    )

    assert decision.covered is None
    assert decision.flagos_type == ""
    assert decision.evidence == "insufficient evidence to determine FlagOS coverage"
