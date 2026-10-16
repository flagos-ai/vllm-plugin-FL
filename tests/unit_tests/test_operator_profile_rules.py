import csv
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tools.operator_profile.eager_reference import (
    attach_eager_reference,
    load_eager_reference,
    reference_operator_name,
    reference_operator_rows,
)
from tools.operator_profile.extract_operator_shapes import (
    collect_runtime,
    format_operator_time_percent,
    operator_list_rows,
    summary_csv_rows,
    validate_csv_outputs,
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


def eager_detail(kernel, operator, shapes, dtypes):
    return {
        "operator_name": operator,
        "kernel_name": kernel,
        "mapping_status": "operator_shape_matched",
        "input_shapes": json.dumps(shapes),
        "input_dtypes": json.dumps(dtypes),
        "kernel_event_count": 1,
        "kernel_time_us": 2.0,
    }


def write_eager_results(path, rows, rank=0):
    path.mkdir()
    (path / "summary.json").write_text(
        json.dumps(
            {"scope": {"rank": rank, "phase": "runtime_only"}, "conservation": {}}
        ),
        encoding="utf-8",
    )
    with (path / "kernel_shape_dtype.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_eager_reference_keeps_ambiguous_api_and_shape_candidates(tmp_path):
    results = tmp_path / "eager"
    write_eager_results(
        results,
        [
            eager_detail("shared_kernel", "aten::zeros", [[4]], ["float"]),
            eager_detail("shared_kernel", "aten::zero_", [[8]], ["bf16"]),
        ],
    )
    index, provenance = load_eager_reference(results, 0)
    original = eager_detail("shared_kernel", "null", None, None)
    original["mapping_status"] = "missing_external_id"
    operators, details, stats = attach_eager_reference(
        [
            {
                "operator_name": "null",
                "operator_kind": "unattributed",
                "kernel_name": "shared_kernel",
                "operator_id": 7,
            }
        ],
        [original],
        index,
    )
    assert operators[0]["operator_id"] == 7
    assert operators[0]["operator_name"] == "aten::zero_,aten::zeros"
    assert operators[0]["operator_kind"] == "eager-refer"
    assert set(operators[0]) == {
        "operator_id",
        "operator_name",
        "operator_kind",
        "kernel_name",
    }
    assert details[0]["eager_input_shapes"] == "null"
    assert details[0]["eager_input_dtypes"] == "null"
    candidates = json.loads(details[0]["eager_candidates"])
    assert {
        (row["operator_name"], tuple(row["input_shapes"][0]), row["input_dtypes"][0])
        for row in candidates
    } == {("aten::zeros", (4,), "float"), ("aten::zero_", (8,), "bf16")}
    assert all(details[0][key] == value for key, value in original.items())
    assert details[0]["mapping_source"] == provenance["source"] == "eager-refer"
    assert stats["events_with_reference_candidates"] == 1


def test_eager_reference_matches_full_name_and_preserves_known_api(tmp_path):
    results = tmp_path / "eager"
    write_eager_results(
        results,
        [
            eager_detail(
                "void backend::kernel<1>(float*)", "aten::add", [[4]], ["float"]
            ),
            eager_detail("shared_kernel", "aten::zeros", [[4]], ["float"]),
            eager_detail("shared_kernel", "aten::zero_", [[8]], ["float"]),
            eager_detail("triton_poi_fused_add_0", "aten::add", [[4]], ["float"]),
            eager_detail("triton_handwritten_add", "aten::add", [[4]], ["float"]),
        ],
    )
    index, _ = load_eager_reference(results, 0)
    details = [
        eager_detail("void backend::kernel<2>(float*)", "null", None, None),
        eager_detail("shared_kernel", "aten::zeros", None, None),
        eager_detail("shared_kernel", "aten::mul", None, None),
        eager_detail("triton_poi_fused_add_0", "null", None, None),
        eager_detail("triton_handwritten_add", "null", None, None),
    ]
    _, actual, _ = attach_eager_reference([], details, index)
    assert [row["mapping_source"] for row in actual] == [
        "null",
        "eager-refer",
        "null",
        "null",
        "eager-refer",
    ]
    assert json.loads(actual[1]["eager_input_shapes"]) == [[4]]
    assert actual[1]["eager_operator_name"] == "aten::zeros"


def test_eager_reference_rejects_other_rank(tmp_path):
    results = tmp_path / "eager"
    write_eager_results(
        results, [eager_detail("kernel", "aten::add", [[4]], ["float"])], rank=1
    )
    with pytest.raises(ValueError, match="same rank"):
        load_eager_reference(results, 0)


def test_eager_reference_does_not_chain_a_filled_ambiguous_api(tmp_path):
    results = tmp_path / "eager"
    row = eager_detail("kernel", "aten::mul", [[4]], ["float"])
    row["mapping_status"] = "operator_ambiguous"
    row["mapping_source"] = "eager-refer"
    write_eager_results(results, [row])
    index, _ = load_eager_reference(results, 0)
    assert index == {}


@pytest.mark.parametrize(
    ("operator", "kernel", "apis", "expected"),
    [
        ("null", "kernel", ["aten::mul", "aten::mul"], "aten::mul"),
        ("aten::add", "kernel", ["aten::mul"], "aten::add"),
        ("null", "kernel", ["aten::mul", "aten::add"], "null"),
        ("null", "kernel", [], "null"),
        ("null", "triton_poi_fused_mul_0", ["aten::mul"], "null"),
    ],
)
def test_reference_api_fill_requires_a_missing_name_and_unique_candidate(
    operator, kernel, apis, expected
):
    index = {kernel: [{"operator_name": api} for api in apis]}
    assert reference_operator_name(operator, kernel, index) == expected
    assert reference_operator_name(operator, kernel, None) == operator


def test_reference_fill_does_not_override_conflicting_runtime_candidates():
    index = {
        "kernel": [
            {
                "operator_name": "aten::mul",
                "input_shapes": [[4]],
                "input_dtypes": ["float"],
            }
        ]
    }
    observed = [{"operator": "aten::add"}, {"operator": "aten::sub"}]
    assert reference_operator_name("null", "kernel", index, observed) == "null"
    detail = eager_detail("kernel", "null", None, None)
    detail["mapping_status"] = "operator_ambiguous"
    detail["candidate_operators"] = json.dumps(observed)
    _, details, _ = attach_eager_reference([], [detail], index)
    assert details[0]["operator_name"] == "null"
    assert details[0]["mapping_source"] == "eager-refer"
    assert details[0]["candidate_operators"] == detail["candidate_operators"]
    report = {
        "kernel": {
            "operator_variants": {},
            "unattributed_variants": [
                {
                    "kernel_event_count": 1,
                    "kernel_time_us": 1,
                    "candidate_operators": observed,
                }
            ],
        }
    }
    row = summary_csv_rows(report, 1000, index)[0]
    assert row["operator_name"] == row["refer"] == "null"


def test_detail_fills_unique_api_without_selecting_an_ambiguous_shape():
    index = {
        "kernel": [
            {
                "operator_name": "aten::mul",
                "input_shapes": [[4]],
                "input_dtypes": ["float"],
            },
            {
                "operator_name": "aten::mul",
                "input_shapes": [[8]],
                "input_dtypes": ["float"],
            },
        ]
    }
    original = eager_detail("kernel", "null", None, None)
    original["mapping_status"] = "missing_external_id"
    _, details, _ = attach_eager_reference([], [original], index)
    assert details[0]["operator_name"] == "aten::mul"
    assert details[0]["eager_operator_name"] == "aten::mul"
    assert details[0]["eager_input_shapes"] == "null"
    assert len(json.loads(details[0]["eager_candidates"])) == 2
    assert {
        key: value
        for key, value in details[0].items()
        if key in original and key != "operator_name"
    } == {key: value for key, value in original.items() if key != "operator_name"}


def test_kernel_time_merges_references_before_percent_rounding():
    report = {
        "mul_kernel": {
            "operator_variants": {
                "aten::mul": [{"kernel_event_count": 1, "kernel_time_us": 0.004}]
            },
            "unattributed_variants": [
                {"kernel_event_count": 2, "kernel_time_us": 0.004}
            ],
        },
        "other_kernel": {
            "operator_variants": {
                "aten::add": [{"kernel_event_count": 1, "kernel_time_us": 99.992}]
            },
            "unattributed_variants": [],
        },
    }
    original = summary_csv_rows(report, 100_000)
    index = {"mul_kernel": [{"operator_name": "aten::mul"}]}
    actual = summary_csv_rows(report, 100_000, index)
    mul = [row for row in actual if row["kernel_name"] == "mul_kernel"]
    assert mul == [
        {
            "operator_name": "aten::mul",
            "refer": "eager-refer",
            "kernel_name": "mul_kernel",
            "kernel_call_count": 3,
            "kernel_time_us": 0.008,
            "percent": "0.008%",
        }
    ]
    assert all(row["refer"] == "null" for row in original)
    assert (
        next(row for row in actual if row["kernel_name"] == "other_kernel")["refer"]
        == "null"
    )
    assert sum(row["kernel_call_count"] for row in actual) == sum(
        row["kernel_call_count"] for row in original
    )


def test_kernel_time_keeps_distinct_apis_for_the_same_kernel():
    report = {
        "kernel": {
            "operator_variants": {
                "aten::add": [{"kernel_event_count": 1, "kernel_time_us": 1}]
            },
            "unattributed_variants": [{"kernel_event_count": 2, "kernel_time_us": 2}],
        }
    }
    rows = summary_csv_rows(report, 3_000, {"kernel": [{"operator_name": "aten::mul"}]})
    assert [(r["operator_name"], r["refer"], r["kernel_call_count"]) for r in rows] == [
        ("aten::add", "null", 1),
        ("aten::mul", "eager-refer", 2),
    ]


def test_eager_inventory_labels_only_unattributed_rows():
    rows = [
        {
            "operator_id": i,
            "operator_name": "aten::add",
            "operator_kind": kind,
            "kernel_name": "shared_kernel",
            "kernel_time_percent(%)": "1.23",
        }
        for i, kind in enumerate(
            [
                "aten",
                "custom",
                "communication",
                "fused_communication_compute",
                "triton_compiled",
                "unattributed",
            ],
            start=1,
        )
    ]
    rows[-1]["operator_name"] = "null"
    rows.append({**rows[-1], "operator_id": 7, "kernel_name": "missing_kernel"})
    index = {
        "shared_kernel": [
            {
                "operator_name": "aten::add",
                "input_shapes": [[4]],
                "input_dtypes": ["float"],
            },
            {
                "operator_name": "aten::add",
                "input_shapes": [[8]],
                "input_dtypes": ["float"],
            },
        ]
    }
    actual = reference_operator_rows(rows, index)
    assert actual[:5] == rows[:5]
    assert actual[-1] == rows[-1]
    assert actual[5] == {
        **rows[5],
        "operator_name": "aten::add",
        "operator_kind": "eager-refer",
    }
    assert rows[5]["operator_kind"] == "unattributed"


def test_operator_list_merges_direct_and_unique_eager_aten_before_rounding():
    index = {
        kernel: [{"operator_name": "aten::mul"}]
        for kernel in ["mul_kernel", "other_mul_kernel"]
    }
    rows = operator_list_rows(
        [
            {"operator_name": api, "kernel_name": kernel, "kernel_time_us": time}
            for api, kernel, time in [
                ("aten::mul", "mul_kernel", 0.006),
                ("null", "mul_kernel", 0.006),
                ("null", "other_mul_kernel", 0.002),
                ("aten::add", "add_kernel", 99.986),
            ]
        ],
        index,
    )
    mul_rows = [row for row in rows if row["operator_name"] == "aten::mul"]
    assert len(mul_rows) == 2
    assert {row["operator_id"] for row in mul_rows} == {2}
    assert mul_rows[0]["operator_kind"] == "aten"
    assert mul_rows[0]["kernel_time_percent(%)"] == "0.01"
    assert mul_rows[1]["operator_kind"] == "eager-refer"
    assert {row["operator_id"] for row in rows} == {1, 2}


def test_operator_list_preserves_ambiguous_eager_and_custom_identities():
    index = {
        "ambiguous_kernel": [
            {"operator_name": "aten::mul"},
            {"operator_name": "aten::add"},
        ],
        "custom_first": [{"operator_name": "extension::wrapper"}],
        "custom_second": [{"operator_name": "extension::wrapper"}],
    }
    rows = operator_list_rows(
        [
            {"operator_name": api, "kernel_name": kernel, "kernel_time_us": 1}
            for api, kernel in [
                ("aten::mul", "mul_kernel"),
                ("null", "ambiguous_kernel"),
                ("null", "custom_first"),
                ("null", "custom_second"),
            ]
        ],
        index,
    )
    assert len({row["operator_id"] for row in rows}) == 4
    assert rows[0]["operator_name"] == "aten::mul"
    assert {row["operator_kind"] for row in rows[1:]} == {"eager-refer"}


def test_extractor_merges_unique_eager_aten_and_preserves_runtime_details(tmp_path):
    eager = tmp_path / "eager"
    write_eager_results(
        eager, [eager_detail("mul_kernel", "aten::mul", [[4]], ["float"])]
    )
    trace = tmp_path / "rank0.pt.trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "cpu_op", "name": "aten::mul", "args": {"External id": 1}},
                    {
                        "cat": "kernel",
                        "name": "mul_kernel",
                        "dur": 1,
                        "args": {"External id": 1},
                    },
                    {"cat": "kernel", "name": "mul_kernel", "dur": 2},
                ]
            }
        ),
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[2]
        / "tools/operator_profile/extract_operator_shapes.py"
    )
    output = tmp_path / "graph"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--trace-dir",
            str(trace),
            "--output-dir",
            str(output),
            "--eager-reference",
            str(eager),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with (output / "operator_list.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert rows == [
        {
            "operator_id": "1",
            "operator_name": "aten::mul",
            "operator_kind": "aten",
            "kernel_name": "mul_kernel",
            "kernel_time_percent(%)": "100.00",
        }
    ]
    with (output / "kernel_shape_dtype.csv").open() as source:
        details = list(csv.DictReader(source))
    assert len(details) == 2
    reference_row = next(
        row for row in details if row["mapping_status"] == "missing_external_id"
    )
    assert reference_row["operator_name"] == "aten::mul"
    assert reference_row["mapping_source"] == "eager-refer"
    assert reference_row["input_shapes"] == reference_row["input_dtypes"] == "null"
    assert reference_row["eager_operator_name"] == "aten::mul"
    assert json.loads(reference_row["eager_input_shapes"]) == [[4]]
    assert len(reference_row) == 14
    with (output / "kernel_time.csv").open() as source:
        time_rows = list(csv.DictReader(source))
    assert time_rows == [
        {
            "operator_name": "aten::mul",
            "refer": "eager-refer",
            "kernel_name": "mul_kernel",
            "kernel_call_count": "2",
            "kernel_time_us": "3.0",
            "percent": "100.000%",
        }
    ]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["kernel_event_count"] == 2
    assert summary["kernel_time_total_us"] == 3
    assert all(
        value for value in summary["conservation"].values() if isinstance(value, bool)
    )
    kernel_summary, kernel_report, _ = collect_runtime([trace], 0)
    index, _ = load_eager_reference(eager, 0)
    time_rows[0]["refer"] = "null"
    with (output / "kernel_time.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(time_rows[0]))
        writer.writeheader()
        writer.writerows(time_rows)
    checks = validate_csv_outputs(
        output / "kernel_time.csv",
        output / "kernel_shape_dtype.csv",
        output / "operator_list.csv",
        kernel_summary,
        kernel_report,
        index,
    )
    assert checks["csv_kernel_event_count_matches"]
    assert checks["csv_kernel_time_matches"]
    assert not checks["csv_summary_rows_match_expected"]


def test_extractor_eager_reference_preserves_graph_outputs(tmp_path):
    eager = tmp_path / "eager"
    write_eager_results(
        eager,
        [
            eager_detail("copy_kernel", "aten::copy_", [[4]], ["float"]),
            eager_detail("copy_kernel", "aten::add", [[8]], ["float"]),
        ],
    )
    trace = tmp_path / "rank0.pt.trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "cat": "cpu_op",
                        "name": "aten::copy_",
                        "args": {
                            "External id": 1,
                            "Input Dims": [[2]],
                            "Input type": ["float"],
                        },
                    },
                    {
                        "cat": "kernel",
                        "name": "copy_kernel",
                        "dur": 1,
                        "args": {"External id": 1},
                    },
                    {"cat": "kernel", "name": "copy_kernel", "dur": 2},
                    {"cat": "kernel", "name": "unknown_kernel", "dur": 3},
                    {"cat": "kernel", "name": "triton_poi_fused_add_0", "dur": 4},
                ]
            }
        ),
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[2]
        / "tools/operator_profile/extract_operator_shapes.py"
    )
    plain = tmp_path / "plain"
    enriched = tmp_path / "enriched"
    for output in [plain, enriched]:
        command = [
            sys.executable,
            str(script),
            "--trace-dir",
            str(trace),
            "--output-dir",
            str(output),
        ]
        if output == enriched:
            command += ["--eager-reference", str(eager)]
        subprocess.run(command, check=True, capture_output=True, text=True)
    assert (plain / "kernel_time.csv").read_bytes() == (
        enriched / "kernel_time.csv"
    ).read_bytes()
    for name in ["operator_list.csv", "kernel_shape_dtype.csv"]:
        with (plain / name).open() as source:
            original = list(csv.DictReader(source))
        with (enriched / name).open() as source:
            actual = list(csv.DictReader(source))
        assert len(original) == len(actual)
        for old, new in zip(original, actual):
            if name == "operator_list.csv":
                assert set(new) == set(old)
                for key in ["operator_id", "kernel_name", "kernel_time_percent(%)"]:
                    assert new[key] == old[key]
                if (
                    old["operator_kind"] == "unattributed"
                    and old["kernel_name"] == "copy_kernel"
                ):
                    assert new["operator_kind"] == "eager-refer"
                    assert new["operator_name"] == "aten::add,aten::copy_"
                else:
                    assert new == old
            else:
                assert {key: new[key] for key in old} == old
                if new["kernel_name"] != "copy_kernel":
                    assert new["mapping_source"] == "null"
    summary = json.loads((enriched / "summary.json").read_text())
    assert summary["kernel_event_count"] == 4
    assert summary["kernel_time_total_us"] == 10
    assert all(
        value for value in summary["conservation"].values() if isinstance(value, bool)
    )
    assert summary["eager_reference"]["referenced_kernel_count"] == 1
    assert summary["eager_reference"]["events_with_reference_candidates"] == 1
