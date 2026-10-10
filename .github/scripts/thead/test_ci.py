# Copyright 2026 FlagOS Contributors
"""CPU-only regressions for CI acceptance boundaries; no vendor imports."""

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
# These are the actual standalone CI modules, not substitutes for runtime code.
sys.path.insert(0, str(HERE))
import run_tests
import stack
from common import session_members, validate_junit
from config import resolve_config
from run_gate import (
    CHECKS,
    QUALITY_ORACLE_VERSION,
    SCENARIOS,
    parse_graph_observations,
    validate_documents,
)


class IsolatedEntryTests(unittest.TestCase):
    def test_actual_cli_entrypoints_under_isolated_python(self):
        for filename in ("run_tests.py", "run_gate.py", "stack.py"):
            with self.subTest(filename=filename):
                result = subprocess.run(
                    [sys.executable, "-I", "-B", str(HERE / filename), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_stack_cli_keeps_import_info_off_json_stdout(self):
        facts = {"provider": "ordinary-wheel", "version": "0.28.0+empty"}

        def check():
            print("INFO normal import check")
            return facts

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(sys, "argv", ["stack.py", "check"]),
            patch.object(stack, "check_stack", side_effect=check),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            stack.main()
        self.assertEqual(json.loads(stdout.getvalue()), facts)
        self.assertIn("INFO normal import check", stderr.getvalue())
        self.assertNotIn("INFO", stdout.getvalue())


class VllmProviderTests(unittest.TestCase):
    def setUp(self):
        self.origin = str((HERE / "image-provider/vllm/__init__.py").resolve())

    def assert_provider_rejected(self, module_version, distribution_version, origin):
        with self.assertRaises(RuntimeError) as error:
            stack.validate_vllm_provider(
                module_version, distribution_version, self.origin, origin
            )
        for value in (module_version, distribution_version, self.origin, origin):
            self.assertIn(repr(value), str(error.exception))

    def test_actual_split_versions_and_resolved_origin_pass(self):
        distribution_origin = str(HERE / "image-provider/vllm/../vllm/__init__.py")
        self.assertEqual(
            stack.validate_vllm_provider(
                "0.28.0", "0.28.0+empty", self.origin, distribution_origin
            ),
            {
                "vllm_module_version": "0.28.0",
                "vllm_distribution_version": "0.28.0+empty",
                "vllm_origin": self.origin,
                "vllm_distribution_origin": distribution_origin,
            },
        )

    def test_wrong_module_version_rejected_with_actual_values(self):
        self.assert_provider_rejected("0.23.0", "0.28.0+empty", self.origin)

    def test_distribution_without_empty_rejected_with_actual_values(self):
        self.assert_provider_rejected("0.28.0", "0.28.0", self.origin)

    def test_wrong_provider_origin_rejected_with_actual_values(self):
        wrong_origin = str((HERE / "system-provider/vllm/__init__.py").resolve())
        self.assert_provider_rejected("0.28.0", "0.28.0+empty", wrong_origin)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.defaults = {"container_options": "--device /dev/alixpu"}
        self.env = {
            "THEAD_CI_IMAGE": "registry.example/ppu@sha256:" + "a" * 64,
            "THEAD_CI_VISIBLE_DEVICES": "0,1,2,3",
            "THEAD_CI_MODEL_27B": "/models/dense",
            "THEAD_CI_MODEL_35B": "/models/moe",
        }

    def versioned_resources(self):
        return {
            "ci_image": "registry.example/pinned@sha256:" + "b" * 64,
            "runner_labels": ["reserved-ppu"],
            "container_volumes": ["/model-store:/models:ro"],
            "container_options": "--device /dev/alixpu",
            "visible_devices": "12,13,14,15",
            "model_27b": "/models/dense",
            "model_35b": "/models/moe",
            "base_python": "/opt/thead/venv/bin/python",
        }

    def test_empty_vars_use_versioned_resources(self):
        defaults = self.versioned_resources()
        empty = {
            key: ""
            for key in (
                *self.env,
                "THEAD_CI_RUNNER_LABELS",
                "THEAD_CI_CONTAINER_OPTIONS",
                "THEAD_CI_CONTAINER_VOLUMES",
                "THEAD_CI_BASE_PYTHON",
            )
        }
        self.assertEqual(resolve_config(defaults, empty), defaults)

    def test_invalid_versioned_resources_fail(self):
        for key, value in (
            ("ci_image", "registry.example/ppu:latest"),
            ("visible_devices", "12,12,14,15"),
            ("runner_labels", "reserved-ppu"),
            ("container_volumes", {"host": "/models"}),
            ("base_python", "python"),
        ):
            defaults = dict(self.versioned_resources(), **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                resolve_config(defaults, {})

    def test_invalid_override_does_not_fall_back(self):
        with self.assertRaises(ValueError):
            resolve_config(
                self.versioned_resources(),
                {"THEAD_CI_IMAGE": "registry.example/ppu:latest"},
            )
        with self.assertRaises(ValueError):
            resolve_config(
                self.versioned_resources(),
                {"THEAD_CI_CONTAINER_VOLUMES": "not-json"},
            )

    def test_valid_overrides_replace_versioned_resources(self):
        config = resolve_config(
            self.versioned_resources(),
            dict(
                self.env,
                THEAD_CI_CONTAINER_VOLUMES='["/new-model-store:/models:ro"]',
                THEAD_CI_BASE_PYTHON="/custom/runtime/bin/python",
            ),
        )
        self.assertEqual(config["ci_image"], self.env["THEAD_CI_IMAGE"])
        self.assertEqual(config["visible_devices"], "0,1,2,3")
        self.assertEqual(config["runner_labels"], ["reserved-ppu"])
        self.assertEqual(config["container_volumes"], ["/new-model-store:/models:ro"])
        self.assertEqual(config["base_python"], "/custom/runtime/bin/python")

    def test_main_runner_default_and_overrides(self):
        config = resolve_config(self.defaults, self.env)
        self.assertEqual(config["runner_labels"], ["flagcicd-810e"])
        self.env["THEAD_CI_RUNNER_LABELS"] = '["self-hosted","ppu-dedicated"]'
        self.assertEqual(
            resolve_config(self.defaults, self.env)["runner_labels"][-1],
            "ppu-dedicated",
        )

    def test_unconfigured_image_fails(self):
        self.env.pop("THEAD_CI_IMAGE")
        with self.assertRaisesRegex(ValueError, "THEAD_CI_IMAGE"):
            resolve_config(self.defaults, self.env)

    def test_unpinned_image_fails(self):
        self.env["THEAD_CI_IMAGE"] = "registry.example/ppu:latest"
        with self.assertRaises(ValueError):
            resolve_config(self.defaults, self.env)

    def test_duplicate_devices_and_privileged_fail(self):
        self.env["THEAD_CI_VISIBLE_DEVICES"] = "0,1,1,3"
        with self.assertRaises(ValueError):
            resolve_config(self.defaults, self.env)
        self.env["THEAD_CI_VISIBLE_DEVICES"] = "0,1,2,3"
        self.env["THEAD_CI_CONTAINER_OPTIONS"] = "--privileged"
        with self.assertRaises(ValueError):
            resolve_config(self.defaults, self.env)

    def test_invalid_json_shapes_fail(self):
        for value in ('"ppu"', "[]", "[1]"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_config(
                    self.defaults, dict(self.env, THEAD_CI_RUNNER_LABELS=value)
                )


class LlmQualityTests(unittest.TestCase):
    # Actual text_single response; source SHA256 6e934ae3ded82db2e8dd8b113edff143247b55a30c9aae17a039e578a2f78964.
    V51_TEXT = (
        "Large language models (LLMs) are advanced artificial intelligence systems de"
        "signed to understand, generate, and manipulate human language with remarkabl"
        "e fluency. These models are built upon deep learning architectures, primaril"
        "y the transformer framework, which utilizes self-attention mechanisms to pro"
        "cess sequential data efficiently. Training involves exposing the model to va"
        "st datasets comprising text from books, websites, and articles, allowing it "
        "to learn complex patterns, grammar, and factual knowledge through statistica"
        "l prediction. During inference, the model generates text token by token, cal"
        "culating probabilities for the next word based on the preceding context. Cap"
        "abilities include answering questions, writing code, translating languages, "
        "and summarizing documents. However, limitations such as hallucinations, bias"
        ", and lack of true reasoning persist. Responsible use requires careful overs"
        "ight, fact-checking, and ethical guidelines to mitigate risks and ensure ben"
        "eficial outcomes for society."
    )

    # Actual text_single response; source SHA256 1357e9390c957666a1598ff2336516dcc4838dd50c4258027f5537f61c80e5a5.
    CI_GRAPH_TEXT = (
        "Large language models (LLMs) are sophisticated artificial intelligence syste"
        "ms designed to understand, generate, and manipulate human language with rema"
        "rkable fluency. These models are built upon deep learning architectures, pri"
        "marily the transformer framework, which utilizes self-attention mechanisms t"
        "o process vast sequences of text data efficiently. Training involves exposin"
        "g the model to massive datasets comprising books, articles, code, and web co"
        "ntent, allowing it to learn complex linguistic patterns, factual knowledge, "
        "and reasoning structures through predictive tasks. During inference, the mod"
        "el generates text token by token, calculating probabilities for the next wor"
        "d based on the context provided by the user’s prompt. This process enables a"
        " wide array of capabilities, including natural conversation, creative writin"
        "g, code generation, translation, and summarization. However, LLMs are not in"
        "fallible; they can suffer from hallucinations, where they confidently presen"
        "t false information as fact, and may inadvertently reflect biases present in"
        " their training data. Furthermore, they lack true understanding or conscious"
        "ness, operating instead on statistical correlations. Responsible use require"
        "s careful oversight, including human-in-the-loop verification, rigorous bias"
        " mitigation strategies, and transparent disclosure of AI involvement. Users "
        "should treat LLM outputs as suggestions rather than absolute truths, ensurin"
        "g ethical deployment in sensitive domains like healthcare, law, and educatio"
        "n to maintain trust and safety."
    )

    # Actual text_single response; source SHA256 4d2d4d1bd600735eb17ba40eab03c2712cc6e64aa6e347be014cff67b59ffecd.
    CI_EAGER_TEXT = (
        "Large language models (LLMs) are sophisticated artificial intelligence syste"
        "ms designed to understand, generate, and manipulate human language with rema"
        "rkable fluency. These models are built upon deep learning architectures, pri"
        "marily the transformer framework, which utilizes self-attention mechanisms t"
        "o process vast sequences of text data efficiently. Training involves exposin"
        "g the model to massive datasets comprising books, articles, code, and web co"
        "ntent, allowing it to learn complex linguistic patterns, factual knowledge, "
        "and reasoning structures through predictive tasks. During inference, the mod"
        "el generates text token by token, calculating probabilities for the next wor"
        "d based on the context provided by the user’s prompt. This process enables a"
        " wide array of capabilities, including natural conversation, creative writin"
        "g, code generation, translation, and summarization. However, LLMs are not in"
        "fallible; they can suffer from hallucinations, where they confidently presen"
        "t false information as fact, and may inadvertently reflect biases present in"
        " their training data. Furthermore, they lack true understanding or conscious"
        "ness, operating instead on statistical correlations. Responsible use require"
        "s careful oversight, including human-in-the-loop validation, rigorous bias m"
        "itigation strategies, and transparent disclosure of the model’s limitations "
        "to ensure ethical deployment in sensitive applications such as healthcare, l"
        "aw, and education."
    )

    REPHRASED_TEXT = (
        "Large language models learn patterns from examples. Training adjusts their "
        "parameters on extensive text. During inference they predict likely next "
        "tokens to answer questions and summarize material. They can confidently "
        "invent incorrect facts and may reproduce stereotypes learned from data. "
        "People should check claims and guard sensitive information when applying "
        "the system."
    )
    PREFIX = (
        "Large language models learn patterns from examples. Training adjusts their "
        "parameters on extensive text. During inference they predict likely next "
        "tokens to answer questions and summarize material. "
    )
    REQUIRED = ["large language model", "training", "inference", "limitations"]

    @classmethod
    def setUpClass(cls):
        source = HERE.parents[2] / "tools/adaptation-gate-cases/gate_quality.py"
        spec = importlib.util.spec_from_file_location(
            "_actual_gate_quality_tests", source
        )
        cls.quality = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.quality)
        if Path(cls.quality.__file__).resolve() != source.resolve():
            raise AssertionError("Quality tests must load the actual Gate module")

    def checks(self, text):
        return self.quality.quality_checks(
            text, self.REQUIRED, min_length=256, semantic_profile="llm-explanation-v1"
        )

    def test_versioned_oracle_accepts_three_real_responses_with_reviewable_spans(self):
        self.assertEqual(self.quality.ORACLE_VERSION, "2-llm-concepts")
        for text in (self.V51_TEXT, self.CI_GRAPH_TEXT, self.CI_EAGER_TEXT):
            with self.subTest(text=text[:70]):
                checks = self.checks(text)
                self.assertEqual(set(checks), set(CHECKS))
                self.assertTrue(all(checks.values()), checks)
                evidence = self.quality.llm_explanation_evidence(text)
                self.assertTrue(evidence["semantics"])
                self.assertTrue(evidence["ordered"])
                self.assertGreaterEqual(len(evidence["distinct_categories"]), 2)
                self.assertGreaterEqual(len(evidence["ordered_categories"]), 2)
                self.assertEqual(evidence["offsets"], "normalized_text")
                normalized = self.quality.normalized_text(text)
                for row in evidence["limitations"]:
                    self.assertEqual(normalized[row["start"] : row["end"]], row["text"])
                    self.assertIn(row["category"], evidence["distinct_categories"])

    def test_independently_rephrased_affirmative_limitations_pass(self):
        self.assertNotIn("limitations", self.REPHRASED_TEXT)
        self.assertTrue(all(self.checks(self.REPHRASED_TEXT).values()))
        evidence = self.quality.llm_explanation_evidence(self.REPHRASED_TEXT)
        self.assertEqual(
            set(evidence["distinct_categories"]), {"factual_reliability", "bias"}
        )

        for ending, categories in (
            (
                "Their outputs can contain fabricated information and biases "
                "inherited from training data.",
                {"factual_reliability", "bias"},
            ),
            (
                "Their reasoning is unreliable, and their knowledge is outdated.",
                {"understanding_reasoning", "knowledge_freshness"},
            ),
            (
                "They not only can hallucinate facts but may also reproduce biases.",
                {"factual_reliability", "bias"},
            ),
        ):
            text = (
                self.PREFIX + ending + " Responsible use requires careful oversight "
                "and independent verification before applying generated answers."
            )
            with self.subTest(ending=ending):
                self.assertTrue(all(self.checks(text).values()))
                self.assertEqual(
                    set(
                        self.quality.llm_explanation_evidence(text)[
                            "distinct_categories"
                        ]
                    ),
                    categories,
                )

    def test_missing_or_only_one_limitation_category_fails(self):
        for ending, categories in (
            (
                "These systems have limitations. Responsible use requires careful "
                "oversight and experienced human review before deployment.",
                0,
            ),
            (
                "They may generate false information while sounding authoritative. "
                "Responsible use requires careful oversight and human review.",
                1,
            ),
        ):
            text = self.PREFIX + ending
            with self.subTest(categories=categories):
                checks = self.checks(text)
                self.assertTrue(checks["minimum_length"])
                self.assertFalse(checks["expected_semantics"])
                self.assertFalse(checks["expected_order"])
                self.assertEqual(
                    len(
                        self.quality.llm_explanation_evidence(text)[
                            "distinct_categories"
                        ]
                    ),
                    categories,
                )

    def test_limitation_evidence_before_training_and_inference_is_unordered(self):
        text = (
            "Large language models can hallucinate and may exhibit bias. Training "
            "adjusts their parameters on extensive text. During inference they "
            "predict likely next tokens to answer questions and summarize material. "
            "People should check claims carefully before applying generated answers "
            "in consequential settings."
        )
        checks = self.checks(text)
        self.assertTrue(checks["expected_semantics"])
        self.assertFalse(checks["expected_order"])
        evidence = self.quality.llm_explanation_evidence(text)
        self.assertEqual(len(evidence["distinct_categories"]), 2)
        self.assertEqual(evidence["ordered_categories"], [])

    def test_ordered_body_limitations_pass_even_when_introduction_mentions_them(self):
        text = (
            "Large language models may generate false information and may reproduce "
            "biases. Training adjusts their parameters on extensive text. During "
            "inference they predict likely next tokens to answer questions and "
            "summarize material. They can hallucinate and may reflect stereotypes. "
            "People should verify generated claims and consider unfair assumptions "
            "when applying the system."
        )
        self.assertTrue(all(self.checks(text).values()))
        evidence = self.quality.llm_explanation_evidence(text)
        self.assertEqual(
            set(evidence["ordered_categories"]), {"factual_reliability", "bias"}
        )
        inference_end = evidence["anchor_positions"][-1] + len("inference")
        for category in evidence["ordered_categories"]:
            self.assertTrue(
                any(
                    row["category"] == category and row["start"] >= inference_end
                    for row in evidence["limitations"]
                )
            )

    def test_bare_keyword_headings_are_not_limitation_evidence(self):
        text = (
            self.PREFIX + "Limitations: hallucinations, bias, understanding, "
            "knowledge cutoff. Responsible use: oversight, review, verification, "
            "human judgment, policy, ethics, transparency, risk management."
        )
        checks = self.checks(text)
        self.assertTrue(checks["minimum_length"])
        self.assertFalse(checks["expected_semantics"])
        self.assertFalse(checks["expected_order"])
        self.assertEqual(self.quality.llm_explanation_evidence(text)["limitations"], [])

    def test_denied_limitations_are_not_affirmative_evidence(self):
        for ending, categories in (
            (
                "They never hallucinate and never exhibit bias. They do not generate "
                "false information or reinforce stereotypes.",
                [],
            ),
            (
                "They cannot hallucinate and cannot exhibit bias. Their answers are "
                "always correct and fair in every context.",
                [],
            ),
            (
                "They never suffer from hallucinations and do not lack true "
                "understanding. They always reason accurately about every topic.",
                [],
            ),
            (
                "Limitations include no hallucinations and no bias. Responsible use "
                "requires no fact checking or further review.",
                [],
            ),
            (
                "It is not true that they can hallucinate facts. It is not true "
                "that they may reproduce biases.",
                [],
            ),
            (
                "There is no evidence that they can hallucinate facts or may "
                "reproduce biases.",
                [],
            ),
            (
                "Limitations include hallucinations and bias, but neither is a "
                "problem for them.",
                [],
            ),
            (
                "They lack any problems with reasoning and may reproduce biases.",
                ["bias"],
            ),
        ):
            text = self.PREFIX + ending
            with self.subTest(ending=ending):
                checks = self.checks(text)
                self.assertTrue(checks["minimum_length"])
                self.assertFalse(checks["expected_semantics"])
                self.assertFalse(checks["expected_order"])
                self.assertEqual(
                    self.quality.llm_explanation_evidence(text)["distinct_categories"],
                    categories,
                )

    def test_affirmative_persist_and_realtime_knowledge_limitations_pass(self):
        # The affirmative clause is from the earlier 27B Gate response.
        for term in ("knowledge", "data", "awareness"):
            text = (
                self.PREFIX + "However, limitations persist, such as potential "
                "hallucinations and lack of real-time " + term + ". Responsible use "
                "requires careful review of claims before using generated answers."
            )
            with self.subTest(term=term):
                self.assertTrue(all(self.checks(text).values()))
                self.assertEqual(
                    set(
                        self.quality.llm_explanation_evidence(text)[
                            "distinct_categories"
                        ]
                    ),
                    {"factual_reliability", "knowledge_freshness"},
                )

    def test_corruption_and_repetition_checks_remain_mandatory(self):
        for suffix, failed_check in (
            (" warning warning warning warning", "no_repeated_word_run"),
            (" review facts review facts review facts", "no_repeated_phrase"),
            (" \ufffd", "no_mojibake"),
            ("\x01", "no_control_characters"),
            ("!!!", "no_bang_triplet"),
        ):
            with self.subTest(failed_check=failed_check):
                checks = self.checks(self.REPHRASED_TEXT + suffix)
                self.assertTrue(checks["expected_semantics"])
                self.assertTrue(checks["expected_order"])
                self.assertFalse(checks[failed_check])
                self.assertFalse(all(checks.values()))

    def test_default_profile_and_other_tasks_keep_exact_term_matching(self):
        checks = self.quality.quality_checks(self.CI_GRAPH_TEXT, self.REQUIRED, 256)
        self.assertFalse(checks["expected_semantics"])
        self.assertFalse(checks["expected_order"])
        self.assertTrue(
            all(
                self.quality.quality_checks(
                    "two red squares", ["2", "red", "squares"]
                ).values()
            )
        )
        self.assertFalse(
            self.quality.quality_checks("two crimson squares", ["2", "red", "squares"])[
                "expected_semantics"
            ]
        )

    def test_unknown_or_mismatched_profile_is_rejected(self):
        for required, profile in (
            (self.REQUIRED, "unknown-profile"),
            (["red", "square"], "llm-explanation-v1"),
        ):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                self.quality.quality_checks(
                    self.REPHRASED_TEXT, required, semantic_profile=profile
                )


class AcceptanceTests(unittest.TestCase):
    def test_cleanup_rejects_reused_pid_and_ignores_unrelated_or_zombie(self):
        identity = {"pid": 100, "pgid": 100, "sid": 100, "start_ticks": 200}
        own = dict(identity, state="S")
        child = dict(identity, pid=101, start_ticks=201, state="S")
        unrelated = dict(identity, pid=102, pgid=102, sid=102, state="S")
        zombie = dict(identity, pid=103, state="Z")
        self.assertEqual(
            session_members(identity, [own, child, unrelated, zombie]), [own, child]
        )
        self.assertEqual(session_members(identity, [child]), [child])
        self.assertEqual(session_members(identity, [zombie]), [])
        with self.assertRaisesRegex(RuntimeError, "reused"):
            session_members(identity, [dict(own, start_ticks=300), child])
        with self.assertRaisesRegex(RuntimeError, "cleanup incomplete"):
            session_members(identity, [own, dict(child, pgid=101)])

    def test_unit_cleanup_failure_stops_following_suites(self):
        for cleaned, expected_attempts in ((False, 1), (True, 3)):
            with self.subTest(cleaned=cleaned), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "units"
                receipt = {"clean": False, "cleanup_complete": cleaned}
                with (
                    patch.object(
                        sys, "argv", ["run_tests.py", "--output-dir", str(out)]
                    ),
                    patch.object(run_tests, "run_command", return_value=receipt) as run,
                    patch.object(run_tests, "validate_junit", return_value={}),
                ):
                    self.assertEqual(run_tests.main(), 1)
                facts = json.loads((out / "summary.json").read_text())
                self.assertEqual(run.call_count, expected_attempts)
                self.assertFalse(facts["passed"])
                self.assertEqual(facts["only_owned_cleanup_complete"], cleaned)

    def test_zero_skip_unique_junit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.xml"
            path.write_text(
                '<testsuites><testsuite><testcase classname="c" name="a"/><testcase classname="c" name="b"/></testsuite></testsuites>'
            )
            self.assertEqual(validate_junit(path, 2)["passed"], 2)
            for xml in (
                "<testsuites/>",
                '<testsuite><testcase name="a"><skipped/></testcase></testsuite>',
                '<testsuite><testcase name="a"/><testcase name="a"/></testsuite>',
                '<testsuite><testcase name="a"><error/></testcase></testsuite>',
            ):
                path.write_text(xml)
                with self.subTest(xml=xml), self.assertRaises(RuntimeError):
                    validate_junit(path, 2)

    def documents(self, root):
        for scenario, count in SCENARIOS.items():
            doc = {
                "quality_oracle_version": QUALITY_ORACLE_VERSION,
                "case": {"scenario": scenario},
                "input": [{}] * count,
                "output": [
                    {"passed": True, "checks": dict.fromkeys(CHECKS, True)}
                    for _ in range(count)
                ],
            }
            (root / (scenario + ".json")).write_text(json.dumps(doc))

    def test_original_five_scenarios_26_requests_260_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.documents(root)
            facts = validate_documents(root)
            self.assertEqual((facts["logical_requests"], facts["checks"]), (26, 260))
            self.assertTrue(facts["passed"])
            p = root / "text_single.json"
            doc = json.loads(p.read_text())
            doc["output"][0]["checks"]["expected_semantics"] = False
            p.write_text(json.dumps(doc))
            self.assertFalse(validate_documents(root)["passed"])

    def test_missing_check_or_scenario_is_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.documents(root)
            p = root / "text_single.json"
            doc = json.loads(p.read_text())
            doc["output"][0]["checks"].pop("expected_semantics")
            p.write_text(json.dumps(doc))
            with self.assertRaises(RuntimeError):
                validate_documents(root)
            p.unlink()
            with self.assertRaises(RuntimeError):
                validate_documents(root)

    def test_missing_or_mismatched_quality_oracle_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.documents(root)
            path = root / "text_single.json"
            original = json.loads(path.read_text())
            for version in (None, "1-exact-keywords"):
                document = dict(original)
                if version is None:
                    document.pop("quality_oracle_version")
                else:
                    document["quality_oracle_version"] = version
                path.write_text(json.dumps(document))
                with self.subTest(version=version), self.assertRaises(RuntimeError):
                    validate_documents(root)

    def test_actual_info_graph_evidence_requires_capture_and_dispatch(self):
        # Sanitized INFO lines from the verified 0.28 graph runs.
        capture = "Capturing CUDA graphs (decode, FULL): 100%|####| 4/4 [00:01]"
        stats = "**CUDAGraph Stats:**\\n| Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count |\\n| 1 | 1 | 0 | FULL | 39 |"
        observed = parse_graph_observations(stats, capture)
        self.assertTrue(observed["graph_capture_observed"])
        self.assertTrue(observed["graph_runtime_dispatch_observed"])
        for output, error in (
            ("cudagraph_mode=FULL", ""),
            (stats.replace("39", "0"), capture.replace("4/4", "3/4")),
            ("| 1 | 1 | 0 | FULL | 39 |", "Capturing CUDA graphs 3/4"),
        ):
            with self.subTest(output=output):
                facts = parse_graph_observations(output, error)
                self.assertFalse(
                    facts["graph_capture_observed"]
                    and facts["graph_runtime_dispatch_observed"]
                )

    def test_three_file_fix_required_not_version_only(self):
        self.assertEqual(len(stack.STACK["files"]), 3)
        self.assertEqual(
            hashlib.sha256((HERE / "flaggems-6894.patch").read_bytes()).hexdigest(),
            stack.STACK["patch_sha256"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for row in stack.STACK["files"]:
                p = root / row["path"]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"unpatched or changed body")
            with self.assertRaisesRegex(RuntimeError, "binding failed"):
                stack.verify_files(root, "after")


if __name__ == "__main__":
    unittest.main()
