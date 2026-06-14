"""Audit the current MUAT small-scale evidence package.

The audit is deliberately conservative: it verifies only the narrow claims that
are supported by current artifacts, and it records major claims that remain out
of scope. It does not rerun training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "runs"
LOG_PATH = REPO_ROOT / "research-log.md"
REPORT_PATH = REPO_ROOT / "docs" / "muat_small_scale_evidence_report.md"
REP_STEERING_V1_DIAGNOSIS_PATH = (
    REPO_ROOT / "docs" / "representation_steering_v1_failure_diagnosis.md"
)
OUTPUT_DIR = RUNS / "muat_evidence_package_audit"


def main() -> None:
    audit = {
        "claims": {
            "evidence_supported": [],
            "not_proven": [
                "larger models",
                "functional decode or behavioral steering beyond sorted_ascending/sorted_descending",
                "broad MUAT generality",
                "non-aggressive steering-vector norm requirement",
            ],
        },
        "checks": {},
        "failures": [],
    }

    add_check(audit, "interpretability", audit_interpretability())
    add_check(audit, "decode_final", audit_decode_final())
    add_check(audit, "v1_robust_negative", audit_v1_robust_negative())
    add_check(audit, "v2_development", audit_v2_development())
    add_check(audit, "v2_final_robust_steering", audit_v2_final_robust_steering())
    add_check(
        audit,
        "additional_behavior_decode_feasibility_negative",
        audit_additional_behavior_decode_feasibility_negative(),
    )
    add_check(
        audit,
        "four_behavior_source_generation_feasibility_negative",
        audit_four_behavior_source_generation_feasibility_negative(),
    )
    add_check(
        audit,
        "four_behavior_source_generation_v2_expanded_support_negative",
        audit_four_behavior_source_generation_v2_expanded_support_negative(),
    )
    add_check(
        audit,
        "four_behavior_source_generation_v3_full_pool_negative",
        audit_four_behavior_source_generation_v3_full_pool_negative(),
    )
    add_check(
        audit,
        "four_behavior_source_generation_v4_accept_reject_positive",
        audit_four_behavior_source_generation_v4_accept_reject_positive(),
    )
    add_check(
        audit,
        "four_behavior_decoder_source_pools_v1_negative",
        audit_four_behavior_decoder_source_pools_v1_negative(),
    )
    add_check(
        audit,
        "four_behavior_decoder_source_pools_v2_positive",
        audit_four_behavior_decoder_source_pools_v2_positive(),
    )
    add_check(
        audit,
        "four_behavior_decoder_development_v1_negative",
        audit_four_behavior_decoder_development_v1_negative(),
    )
    add_check(
        audit,
        "four_behavior_decoder_development_v2_negative",
        audit_four_behavior_decoder_development_v2_negative(),
    )
    add_check(
        audit,
        "four_behavior_decoder_development_v3_signature_inversion_negative",
        audit_four_behavior_decoder_development_v3_signature_inversion_negative(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v1_source_pools_positive",
        audit_four_behavior_representation_steering_v1_source_pools_positive(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v1_development_negative",
        audit_four_behavior_representation_steering_v1_development_negative(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v1_failure_diagnosis",
        audit_four_behavior_representation_steering_v1_failure_diagnosis(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v2_source_pools_positive",
        audit_four_behavior_representation_steering_v2_source_pools_positive(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v2_development_negative",
        audit_four_behavior_representation_steering_v2_development_negative(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v2_failure_diagnosis",
        audit_four_behavior_representation_steering_v2_failure_diagnosis(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v3_source_pools_positive",
        audit_four_behavior_representation_steering_v3_source_pools_positive(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v3_development_negative",
        audit_four_behavior_representation_steering_v3_development_negative(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v4_source_pools_positive",
        audit_four_behavior_representation_steering_v4_source_pools_positive(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v4_development_negative",
        audit_four_behavior_representation_steering_v4_development_negative(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v5_source_pools_positive",
        audit_four_behavior_representation_steering_v5_source_pools_positive(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v5_development_negative",
        audit_four_behavior_representation_steering_v5_development_negative(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v6_source_pools_positive",
        audit_four_behavior_representation_steering_v6_source_pools_positive(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v6_development_negative",
        audit_four_behavior_representation_steering_v6_development_negative(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v6_posthoc_pareto_diagnosis",
        audit_four_behavior_representation_steering_v6_posthoc_pareto_diagnosis(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v7_source_pools_positive",
        audit_four_behavior_representation_steering_v7_source_pools_positive(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v7_development_negative",
        audit_four_behavior_representation_steering_v7_development_negative(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v8_source_pools_positive",
        audit_four_behavior_representation_steering_v8_source_pools_positive(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v8_development_negative",
        audit_four_behavior_representation_steering_v8_development_negative(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v9_source_pools_positive",
        audit_four_behavior_representation_steering_v9_source_pools_positive(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v9_development_positive",
        audit_four_behavior_representation_steering_v9_development_positive(),
    )
    add_check(
        audit,
        "four_behavior_representation_steering_v9_final_positive",
        audit_four_behavior_representation_steering_v9_final_positive(),
    )
    add_check(
        audit,
        "four_behavior_functional_weight_editing_v10_source_pools_positive",
        audit_four_behavior_functional_weight_editing_v10_source_pools_positive(),
    )
    add_check(
        audit,
        "four_behavior_functional_weight_editing_v10_development_negative",
        audit_four_behavior_functional_weight_editing_v10_development_negative(),
    )
    add_check(audit, "subject_pool_separation", audit_subject_pool_separation())
    add_check(audit, "research_log_review_status", audit_research_log())
    add_check(audit, "evidence_report_scope", audit_evidence_report())

    audit["passed"] = not audit["failures"]
    if audit["passed"]:
        audit["claims"]["evidence_supported"] = [
            (
                "stored-probe signatures contain heldout-decodable behavior information "
                "for four clean behaviors under logistic/RF classifiers with "
                "shuffled-label controls"
            ),
            "restricted two-behavior functional decoding on fresh final holdout",
            "restricted two-behavior robust steering on fresh final holdout",
            (
                "narrow four-behavior representation-space source-invariant "
                "target-attractor result on fresh final holdout"
            ),
        ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "results.json"
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({
        "passed": audit["passed"],
        "failure_count": len(audit["failures"]),
        "failures": audit["failures"],
        "results_path": str(output_path),
        "evidence_supported_claims": audit["claims"]["evidence_supported"],
        "not_proven": audit["claims"]["not_proven"],
    }, indent=2, sort_keys=True))


def add_check(audit: Dict[str, Any], name: str, result: Dict[str, Any]) -> None:
    audit["checks"][name] = result
    if not result.get("passed", False):
        for failure in result.get("failures", [f"{name} failed"]):
            audit["failures"].append(f"{name}: {failure}")


def audit_interpretability() -> Dict[str, Any]:
    path = RUNS / "stored_probe_interpret_v1" / "results.json"
    result = load_json(path)
    failures = []
    logistic = result["models"]["logistic_regression"]
    forest = result["models"]["random_forest"]
    require(
        failures,
        result["dataset_provenance_comparison"]["matches"],
        "dataset provenance mismatch",
    )
    require(failures, result["signature_dim"] == 560, "signature_dim is not 560")
    require(
        failures,
        logistic["balanced_accuracy"] >= 0.90,
        "logistic balanced accuracy below 0.90",
    )
    require(
        failures,
        forest["balanced_accuracy"] >= 0.95,
        "random forest balanced accuracy below 0.95",
    )
    require(
        failures,
        logistic["shuffled_train_label_control"]["balanced_accuracy"] <= 0.30,
        "logistic shuffled-label control too high",
    )
    require(
        failures,
        forest["shuffled_train_label_control"]["balanced_accuracy"] <= 0.30,
        "random forest shuffled-label control too high",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "metrics": {
            "logistic_balanced_accuracy": logistic["balanced_accuracy"],
            "random_forest_balanced_accuracy": forest["balanced_accuracy"],
            "majority_balanced_accuracy": result["majority_baseline"]["balanced_accuracy"],
        },
    }


def audit_decode_final() -> Dict[str, Any]:
    path = RUNS / "stored_probe_functional_decoder_v2_final_eval" / "results.json"
    result = load_json(path)
    failures = []
    aggregate = result["paired_evaluation"]["metrics"]["aggregate"]
    require(failures, result["passed"], "decode final result did not pass")
    require(failures, not result["failures"], "decode final has failures")
    require(
        failures,
        result["development_status"] == "one_shot_final_artifact_evaluation_no_tuning_after_result",
        "decode final development_status is not one-shot final",
    )
    leakage = result["leakage_audit"]
    require(failures, leakage["final_overlap_train_count"] == 0, "final overlaps train")
    require(
        failures,
        leakage["final_overlap_development_artifact_count"] == 0,
        "final overlaps development artifact",
    )
    require(failures, aggregate["n"] == 54, "decode final aggregate n is not 54")
    require(
        failures,
        aggregate["mean_matched_minus_control_behavior_margin"] >= 0.20,
        "decode aggregate behavior delta below 0.20",
    )
    require(
        failures,
        aggregate["mean_control_minus_matched_subject_output_mse"] >= 0.05,
        "decode aggregate subject-MSE delta below 0.05",
    )
    by_behavior = result["paired_evaluation"]["metrics"]["by_split"]["validation"]["by_behavior"]
    expected_thresholds = {
        "noise_signature": {
            "min_mean_matched_minus_control_behavior_margin": 0.20,
            "min_mean_control_minus_matched_subject_output_mse": 0.05,
        },
        "opposite_direction": {
            "min_mean_matched_minus_control_behavior_margin": 0.20,
            "min_mean_control_minus_matched_subject_output_mse": 0.05,
        },
        "same_label_other_subject": {
            "min_mean_control_minus_matched_subject_output_mse": 0.02,
        },
    }
    thresholds = result["thresholds"]["by_control_type"]
    require(
        failures,
        thresholds == expected_thresholds,
        "decode control-specific thresholds drifted from expected values",
    )
    for behavior, controls in by_behavior.items():
        for control_type, metrics in controls.items():
            control_thresholds = expected_thresholds[control_type]
            for threshold_name, threshold in control_thresholds.items():
                metric_name = decode_threshold_to_metric(threshold_name)
                require(
                    failures,
                    metrics[metric_name] >= threshold,
                    f"{behavior}/{control_type} {metric_name} below {threshold}",
                )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "metrics": aggregate,
        "leakage_audit": leakage,
    }


def audit_v1_robust_negative() -> Dict[str, Any]:
    path = RUNS / "stored_probe_signature_edit_vectors_v1_robust_external_eval" / "results.json"
    result = load_json(path)
    failures = []
    audit = result["individual_gate_audit"]
    require(failures, result["passed"] is False, "V1 robustness unexpectedly passed")
    require(
        failures,
        audit["all_gate_pass_count"] == 43 and audit["n"] == 48,
        "V1 robust negative pass count is not 43/48",
    )
    require(
        failures,
        audit["by_target"]["sorted_ascending"]["all_gate_pass_count"] == 19,
        "V1 sorted_ascending target pass count is not 19/24",
    )
    require(
        failures,
        all(
            record["source_pattern"] == "sorted_descending"
            and record["target_pattern"] == "sorted_ascending"
            for record in audit["failed_records"]
        ),
        "V1 failed records are not all descending-to-ascending",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "holdout_subjects_sha256": result["holdout_subjects_sha256"],
        "individual_gate_audit": {
            "all_gate_pass_count": audit["all_gate_pass_count"],
            "n": audit["n"],
            "by_target": audit["by_target"],
            "failed_record_count": len(audit["failed_records"]),
        },
    }


def audit_v2_development() -> Dict[str, Any]:
    path = RUNS / "stored_probe_signature_edit_vectors_v2_robust_development" / "results.json"
    result = load_json(path)
    failures = []
    require(failures, result["passed"], "V2 development did not pass")
    require(failures, result["train_pool_passed"], "V2 train pool did not pass")
    require(failures, result["validation_pool_passed"], "V2 validation pool did not pass")
    require(
        failures,
        result["train_pool_summary"]["individual_gate_audit"]["all_gate_pass_count"] == 64,
        "V2 train pool did not pass 64/64 individual gates",
    )
    require(
        failures,
        result["validation_pool_summary"]["individual_gate_audit"]["all_gate_pass_count"] == 16,
        "V2 validation pool did not pass 16/16 individual gates",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "training_pool_subjects_sha256": result["training_pool_subjects_sha256"],
        "best_epoch": result["best_epoch"],
        "best_train_loss": result["best_train_loss"],
    }


def audit_v2_final_robust_steering() -> Dict[str, Any]:
    path = RUNS / "stored_probe_signature_edit_vectors_v2_robust_final_eval" / "results.json"
    result = load_json(path)
    failures = []
    aggregate = result["aggregate"]
    individual = result["individual_gate_audit"]
    expected_thresholds = {
        "min_mean_steered_minus_no_edit_target_margin": 0.20,
        "min_mean_steered_minus_reverse_direction_target_margin": 0.20,
        "min_mean_steered_minus_noise_target_margin": 0.20,
        "min_mean_steered_minus_worst_random_norm_matched_target_margin": 0.20,
        "min_mean_steered_target_margin": 0.20,
        "max_mean_steered_source_margin_change": -0.05,
        "min_individual_all_gate_pass_rate": 0.95,
        "min_per_target_individual_all_gate_pass_rate": 0.90,
    }
    thresholds = result["thresholds"]
    require(failures, result["passed"], "V2 final did not pass")
    require(failures, not result["failures"], "V2 final has failures")
    require(failures, aggregate["n"] == 48, "V2 final aggregate n is not 48")
    require(
        failures,
        result["random_control_count"] == 32,
        "V2 final random control count is not 32",
    )
    require(
        failures,
        thresholds == expected_thresholds,
        "V2 final steering thresholds drifted from expected values",
    )
    mean_checks = [
        (
            "mean_steered_minus_no_edit_target_margin",
            expected_thresholds["min_mean_steered_minus_no_edit_target_margin"],
            ">=",
        ),
        (
            "mean_steered_minus_reverse_direction_target_margin",
            expected_thresholds["min_mean_steered_minus_reverse_direction_target_margin"],
            ">=",
        ),
        (
            "mean_steered_minus_noise_target_margin",
            expected_thresholds["min_mean_steered_minus_noise_target_margin"],
            ">=",
        ),
        (
            "mean_steered_minus_worst_random_norm_matched_target_margin",
            expected_thresholds[
                "min_mean_steered_minus_worst_random_norm_matched_target_margin"
            ],
            ">=",
        ),
        (
            "mean_steered_target_margin",
            expected_thresholds["min_mean_steered_target_margin"],
            ">=",
        ),
        (
            "mean_steered_source_margin_change",
            expected_thresholds["max_mean_steered_source_margin_change"],
            "<=",
        ),
    ]
    for metric_name, threshold, operator in mean_checks:
        require(
            failures,
            compare(aggregate[metric_name], threshold, operator),
            f"V2 final aggregate {metric_name} failed expected threshold",
        )
        for target, summary in result["by_target"].items():
            require(
                failures,
                compare(summary[metric_name], threshold, operator),
                f"V2 final {target} {metric_name} failed expected threshold",
            )
    require(
        failures,
        aggregate["mean_steered_minus_worst_random_norm_matched_target_margin"]
        >= expected_thresholds[
            "min_mean_steered_minus_worst_random_norm_matched_target_margin"
        ],
        "V2 final worst-random mean delta below threshold",
    )
    require(
        failures,
        individual["all_gate_pass_count"] == 48 and individual["n"] == 48,
        "V2 final individual all-gate pass is not 48/48",
    )
    require(
        failures,
        individual["all_gate_pass_rate"]
        >= expected_thresholds["min_individual_all_gate_pass_rate"],
        "V2 final individual all-gate pass rate below expected threshold",
    )
    for target, target_audit in individual["by_target"].items():
        require(
            failures,
            target_audit["all_gate_pass_count"] == 24 and target_audit["n"] == 24,
            f"V2 final {target} pass is not 24/24",
        )
        require(
            failures,
            target_audit["all_gate_pass_rate"]
            >= expected_thresholds["min_per_target_individual_all_gate_pass_rate"],
            f"V2 final {target} pass rate below expected threshold",
        )
    require(
        failures,
        not individual["failed_records"],
        "V2 final has failed individual records",
    )
    vector_norms = load_vector_norms(
        RUNS / "stored_probe_signature_edit_vectors_v2_robust_development" / "edit_vectors.pt"
    )
    require(
        failures,
        44.0 <= vector_norms["sorted_descending_to_sorted_ascending"] <= 45.0,
        "V2 descending-to-ascending vector norm not in expected caveat range",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "holdout_subjects_sha256": result["holdout_subjects_sha256"],
        "aggregate": aggregate,
        "individual_gate_audit": individual,
        "expected_thresholds": expected_thresholds,
        "vector_norms": vector_norms,
    }


def audit_additional_behavior_decode_feasibility_negative() -> Dict[str, Any]:
    path = RUNS / "stored_probe_additional_behavior_decode_feasibility_v1" / "results.json"
    result = load_json(path)
    failures = []
    aggregate = result["aggregate"]
    individual = result["individual_gate_audit"]
    expected_thresholds = {
        "min_mean_matched_target_margin": 0.20,
        "min_mean_matched_minus_noise_target_margin": 0.20,
        "min_individual_pass_rate": 0.90,
        "min_per_behavior_individual_pass_rate": 0.80,
    }
    require(
        failures,
        result["claim_scope"] == "fresh_subject_additional_behavior_decode_feasibility_not_proof",
        "additional-behavior claim_scope drifted",
    )
    require(
        failures,
        result["development_status"] == "feasibility_additional_behavior_no_final_claim",
        "additional-behavior development_status drifted",
    )
    require(failures, result["passed"] is False, "additional-behavior feasibility unexpectedly passed")
    require(
        failures,
        result["thresholds"] == expected_thresholds,
        "additional-behavior thresholds drifted",
    )
    require(failures, result["noise_control_count"] == 8, "noise control count is not 8")
    require(failures, aggregate["n"] == 16, "additional-behavior aggregate n is not 16")
    require(
        failures,
        result["holdout_subjects_sha256"]
        == "03b72098c773690011fa330487e51d08f69c2f3b4558e7ab1ae31ae82f5aeb6b",
        "additional-behavior holdout SHA mismatch",
    )
    require(
        failures,
        individual["all_gate_pass_count"] == 0 and individual["n"] == 16,
        "additional-behavior individual pass is not 0/16",
    )
    for behavior in ("has_majority", "mountain_pattern"):
        behavior_audit = individual["by_behavior"][behavior]
        require(
            failures,
            behavior_audit["all_gate_pass_count"] == 0
            and behavior_audit["n"] == 8,
            f"{behavior} individual pass is not 0/8",
        )
        require(
            failures,
            result["by_behavior"][behavior]["mean_matched_target_margin"]
            < expected_thresholds["min_mean_matched_target_margin"],
            f"{behavior} matched target margin unexpectedly cleared threshold",
        )
        require(
            failures,
            result["by_behavior"][behavior]["mean_matched_minus_worst_noise_target_margin"]
            < expected_thresholds["min_mean_matched_minus_noise_target_margin"],
            f"{behavior} matched-minus-noise margin unexpectedly cleared threshold",
        )
    source_summary = result["source_margin_summary"]
    require(
        failures,
        0.20 <= source_summary["has_majority"]["mean"] <= 0.24,
        "has_majority source margin caveat not in expected weak range",
    )
    require(
        failures,
        source_summary["mountain_pattern"]["mean"] >= 0.80,
        "mountain_pattern source margin not strong",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "holdout_subjects_sha256": result["holdout_subjects_sha256"],
        "aggregate": aggregate,
        "by_behavior": result["by_behavior"],
        "individual_gate_audit": {
            "all_gate_pass_count": individual["all_gate_pass_count"],
            "n": individual["n"],
            "by_behavior": individual["by_behavior"],
        },
        "source_margin_summary": source_summary,
        "expected_thresholds": expected_thresholds,
    }


def audit_four_behavior_source_generation_feasibility_negative() -> Dict[str, Any]:
    path = RUNS / "four_behavior_source_generation_feasibility_v1" / "results.json"
    result = load_json(path)
    failures = []
    config = result["config"]
    aggregate = result["aggregate"]
    by_behavior = result["by_behavior"]
    require(
        failures,
        result["claim_scope"] == "source_generation_feasibility_only_not_decoder_evidence",
        "source-generation feasibility claim_scope drifted",
    )
    require(
        failures,
        result["development_status"] == "feasibility_check_before_decoder_training",
        "source-generation feasibility development_status drifted",
    )
    require(
        failures,
        result["passed"] is False,
        "source-generation feasibility unexpectedly passed",
    )
    require(
        failures,
        config["n_per_behavior"] == 8,
        "source-generation feasibility n_per_behavior is not 8",
    )
    require(
        failures,
        config["source_margin_gate"] == 0.40,
        "source-generation source_margin_gate is not 0.40",
    )
    require(
        failures,
        result["behavior_suite_metadata"]["support_heldout_overlap_count"] == 0,
        "source-generation support/heldout overlap is not zero",
    )
    require(
        failures,
        aggregate["n"] == 32 and aggregate["pass_count"] == 24,
        "source-generation aggregate pass count is not 24/32",
    )
    expected_pass_counts = {
        "sorted_ascending": 8,
        "sorted_descending": 8,
        "mountain_pattern": 8,
        "has_majority": 0,
    }
    for behavior, expected_pass_count in expected_pass_counts.items():
        summary = by_behavior[behavior]
        require(
            failures,
            summary["n"] == 8 and summary["pass_count"] == expected_pass_count,
            f"{behavior} source-generation pass count drifted",
        )
    require(
        failures,
        by_behavior["has_majority"]["heldout_margin_max"] < 0.40,
        "has_majority unexpectedly has a source subject clearing the gate",
    )
    require(
        failures,
        "Pilot feasibility sample; n=8 per behavior is not an impossibility result."
        in result["caveats"],
        "source-generation pilot caveat missing",
    )
    require(
        failures,
        "Tests support-only source generation, not stored-probe decoding."
        in result["caveats"],
        "source-generation decoder-scope caveat missing",
    )
    require(
        failures,
        bool(result.get("result_payload_sha256")),
        "source-generation result payload hash missing",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "by_behavior": by_behavior,
        "result_payload_sha256": result.get("result_payload_sha256"),
    }


def audit_four_behavior_source_generation_v2_expanded_support_negative() -> Dict[str, Any]:
    path = RUNS / "four_behavior_source_generation_v2_expanded_support" / "results.json"
    result = load_json(path)
    failures = []
    config = result["config"]
    aggregate = result["aggregate"]
    by_behavior = result["by_behavior"]
    require(
        failures,
        result["claim_scope"]
        == "source_generation_v2_expanded_support_feasibility_only_not_decoder_evidence",
        "source-generation V2 claim_scope drifted",
    )
    require(
        failures,
        result["development_status"]
        == "preregistered_v2_expanded_support_feasibility_before_decoder_training",
        "source-generation V2 development_status drifted",
    )
    require(
        failures,
        result["passed"] is False,
        "source-generation V2 unexpectedly passed",
    )
    require(
        failures,
        config["support_per_class"] == 160 and config["heldout_per_class"] == 64,
        "source-generation V2 support/heldout counts drifted",
    )
    require(
        failures,
        config["base_seed"] == 20261110,
        "source-generation V2 base seed drifted",
    )
    require(
        failures,
        config["n_per_behavior"] == 8,
        "source-generation V2 n_per_behavior is not 8",
    )
    require(
        failures,
        config["source_margin_gate"] == 0.40,
        "source-generation V2 source_margin_gate is not 0.40",
    )
    require(
        failures,
        result["behavior_suite_metadata"]["support_heldout_overlap_count"] == 0,
        "source-generation V2 support/heldout overlap is not zero",
    )
    require(
        failures,
        aggregate["n"] == 32 and aggregate["pass_count"] == 28,
        "source-generation V2 aggregate pass count is not 28/32",
    )
    expected_pass_counts = {
        "sorted_ascending": 8,
        "sorted_descending": 8,
        "mountain_pattern": 8,
        "has_majority": 4,
    }
    for behavior, expected_pass_count in expected_pass_counts.items():
        summary = by_behavior[behavior]
        require(
            failures,
            summary["n"] == 8 and summary["pass_count"] == expected_pass_count,
            f"{behavior} source-generation V2 pass count drifted",
        )
    require(
        failures,
        by_behavior["has_majority"]["heldout_margin_min"] < 0.40,
        "has_majority V2 unexpectedly clears the min-margin gate",
    )
    require(
        failures,
        by_behavior["has_majority"]["heldout_margin_mean"]
        > by_behavior["has_majority"]["source_margin_gate"] - 0.06,
        "has_majority V2 mean margin no longer reflects partial improvement",
    )
    require(
        failures,
        bool(result.get("result_payload_sha256")),
        "source-generation V2 result payload hash missing",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "by_behavior": by_behavior,
        "result_payload_sha256": result.get("result_payload_sha256"),
    }


def audit_four_behavior_source_generation_v3_full_pool_negative() -> Dict[str, Any]:
    path = RUNS / "four_behavior_source_generation_v3_full_pool" / "results.json"
    result = load_json(path)
    failures = []
    config = result["config"]
    aggregate = result["aggregate"]
    by_behavior = result["by_behavior"]
    require(
        failures,
        result["claim_scope"]
        == "source_generation_v3_full_pool_feasibility_only_not_decoder_evidence",
        "source-generation V3 claim_scope drifted",
    )
    require(
        failures,
        result["development_status"]
        == "preregistered_v3_full_pool_feasibility_before_decoder_training",
        "source-generation V3 development_status drifted",
    )
    require(failures, result["passed"] is False, "source-generation V3 unexpectedly passed")
    require(
        failures,
        config["training_mode"] == "heldout_excluded_full_pool",
        "source-generation V3 training mode drifted",
    )
    require(
        failures,
        config["positive_cap"] == 2048
        and config["hard_negative_cap"] == 1024
        and config["generic_negative_cap"] == 1024,
        "source-generation V3 sampling caps drifted",
    )
    require(
        failures,
        config["base_seed"] == 20261210 and config["n_per_behavior"] == 8,
        "source-generation V3 seed or n_per_behavior drifted",
    )
    require(
        failures,
        result["max_selected_train_vs_heldout_overlap_count"] == 0,
        "source-generation V3 selected train/heldout overlap is nonzero",
    )
    require(
        failures,
        aggregate["n"] == 32 and aggregate["pass_count"] == 31,
        "source-generation V3 aggregate pass count is not 31/32",
    )
    expected_pass_counts = {
        "sorted_ascending": 8,
        "sorted_descending": 8,
        "mountain_pattern": 8,
        "has_majority": 7,
    }
    for behavior, expected_pass_count in expected_pass_counts.items():
        summary = by_behavior[behavior]
        require(
            failures,
            summary["n"] == 8 and summary["pass_count"] == expected_pass_count,
            f"{behavior} source-generation V3 pass count drifted",
        )
    require(
        failures,
        by_behavior["has_majority"]["heldout_margin_min"] < 0.40,
        "has_majority V3 unexpectedly clears the min-margin gate",
    )
    require(
        failures,
        by_behavior["has_majority"]["heldout_margin_mean"] > 0.45,
        "has_majority V3 mean margin no longer reflects near-pass improvement",
    )
    require(
        failures,
        all(
            record["train_info"]["selected_train_vs_heldout_overlap_count"] == 0
            for record in result["records"]
        ),
        "source-generation V3 per-subject overlap is nonzero",
    )
    require(
        failures,
        bool(result.get("result_payload_sha256")),
        "source-generation V3 result payload hash missing",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "by_behavior": by_behavior,
        "max_selected_train_vs_heldout_overlap_count": result[
            "max_selected_train_vs_heldout_overlap_count"
        ],
        "result_payload_sha256": result.get("result_payload_sha256"),
    }


def audit_four_behavior_source_generation_v4_accept_reject_positive() -> Dict[str, Any]:
    path = RUNS / "four_behavior_source_generation_v4_accept_reject" / "results.json"
    result = load_json(path)
    failures = []
    config = result["config"]
    aggregate = result["aggregate"]
    by_behavior = result["by_behavior"]
    acceptance = result["acceptance_summary"]
    require(
        failures,
        result["claim_scope"]
        == "source_generation_v4_accept_reject_feasibility_only_not_decoder_evidence",
        "source-generation V4 claim_scope drifted",
    )
    require(
        failures,
        result["development_status"]
        == "preregistered_v4_accept_reject_feasibility_before_decoder_training",
        "source-generation V4 development_status drifted",
    )
    require(failures, result["passed"] is True, "source-generation V4 did not pass")
    require(
        failures,
        config["collection_mode"] == "accept_reject"
        and config["training_mode"] == "heldout_excluded_full_pool",
        "source-generation V4 collection/training mode drifted",
    )
    require(
        failures,
        config["target_accepted_per_behavior"] == 8
        and config["max_attempts_per_behavior"] == 32,
        "source-generation V4 accept-reject bounds drifted",
    )
    require(
        failures,
        config["base_seed"] == 20261310,
        "source-generation V4 base seed drifted",
    )
    require(
        failures,
        result["max_selected_train_vs_heldout_overlap_count"] == 0,
        "source-generation V4 selected train/heldout overlap is nonzero",
    )
    require(
        failures,
        aggregate["n"] == 32 and aggregate["pass_count"] == 32,
        "source-generation V4 aggregate pass count is not 32/32",
    )
    for behavior, summary in by_behavior.items():
        require(
            failures,
            summary["n"] == 8
            and summary["pass_count"] == 8
            and summary["heldout_margin_min"] >= 0.40,
            f"{behavior} source-generation V4 gate failed",
        )
        behavior_acceptance = acceptance[behavior]
        require(
            failures,
            behavior_acceptance["accepted_count"] == 8
            and behavior_acceptance["attempts_used"] <= 32,
            f"{behavior} source-generation V4 acceptance count failed",
        )
    require(
        failures,
        all(
            record["accepted"]
            and record["train_info"]["selected_train_vs_heldout_overlap_count"] == 0
            for record in result["records"]
        ),
        "source-generation V4 record acceptance/overlap invariant failed",
    )
    require(
        failures,
        "No rejection was required under this seed schedule"
        in result["interpretation"],
        "source-generation V4 no-rejection caveat missing",
    )
    require(
        failures,
        bool(result.get("result_payload_sha256")),
        "source-generation V4 result payload hash missing",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "acceptance_summary": acceptance,
        "by_behavior": by_behavior,
        "max_selected_train_vs_heldout_overlap_count": result[
            "max_selected_train_vs_heldout_overlap_count"
        ],
        "result_payload_sha256": result.get("result_payload_sha256"),
    }


def audit_four_behavior_decoder_source_pools_v1_negative() -> Dict[str, Any]:
    path = RUNS / "four_behavior_decoder_source_pools_v1" / "combined_audit.json"
    final_redacted_path = (
        RUNS / "four_behavior_decoder_source_pools_v1" / "final_redacted_audit.json"
    )
    result = load_json(path)
    final_redacted = load_json(final_redacted_path)
    failures = []
    expected_seed_overlaps = {
        "train__development__seed": 71,
        "train__final__seed": 47,
        "development__final__seed": 67,
    }
    require(failures, result["passed"] is False, "V1 source pools unexpectedly passed")
    for key, expected in expected_seed_overlaps.items():
        require(
            failures,
            result["overlap_counts"][key] == expected,
            f"V1 {key} overlap count drifted",
        )
    for key, value in result["overlap_counts"].items():
        if not key.endswith("__seed"):
            require(failures, value == 0, f"V1 non-seed overlap {key} is nonzero")
    for pool, expected_count in {"train": 64, "development": 24, "final": 24}.items():
        summary = result["pool_summaries"][pool]
        require(
            failures,
            all(
                count == expected_count
                for count in summary["accepted_counts_by_behavior"].values()
            ),
            f"V1 {pool} accepted counts drifted",
        )
        require(
            failures,
            summary["max_selected_train_vs_heldout_overlap_count"] == 0,
            f"V1 {pool} selected train/heldout overlap is nonzero",
        )
    require(
        failures,
        final_redacted["claim_scope"] == "redacted_final_source_pool_audit_surface_only",
        "V1 final redacted claim_scope drifted",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "final_redacted_artifact": rel(final_redacted_path),
        "seed_overlap_counts": {
            key: result["overlap_counts"][key]
            for key in expected_seed_overlaps
        },
        "interpretation": "negative_source_pool_construction_seed_overlap",
    }


def audit_four_behavior_decoder_source_pools_v2_positive() -> Dict[str, Any]:
    path = RUNS / "four_behavior_decoder_source_pools_v2" / "combined_audit.json"
    final_redacted_path = (
        RUNS / "four_behavior_decoder_source_pools_v2" / "final_redacted_audit.json"
    )
    result = load_json(path)
    final_redacted = load_json(final_redacted_path)
    failures = []
    require(
        failures,
        result["claim_scope"] == "source_pool_construction_not_decoder_evidence",
        "V2 source-pool claim_scope drifted",
    )
    require(failures, result["passed"] is True, "V2 source pools did not pass")
    require(failures, not result["failures"], "V2 source pools have failures")
    require(
        failures,
        result["seed_preflight"]["passed"] is True
        and not result["seed_preflight"]["failures"],
        "V2 seed preflight did not pass",
    )
    require(
        failures,
        result["seed_preflight"]["seed_behavior_stride"] == 100000,
        "V2 seed behavior stride drifted",
    )
    require(
        failures,
        len(result["seed_preflight"]["seed_ranges"]) == 12,
        "V2 seed preflight range count is not 12",
    )
    for key, value in result["overlap_counts"].items():
        require(failures, value == 0, f"V2 overlap {key} is nonzero")
    expected_counts = {"train": 64, "development": 24, "final": 24}
    for pool, expected_count in expected_counts.items():
        summary = result["pool_summaries"][pool]
        require(
            failures,
            all(
                count == expected_count
                for count in summary["accepted_counts_by_behavior"].values()
            ),
            f"V2 {pool} accepted counts drifted",
        )
        require(
            failures,
            summary["max_selected_train_vs_heldout_overlap_count"] == 0,
            f"V2 {pool} selected train/heldout overlap is nonzero",
        )
    final_public_summary = result["pool_summaries"]["final"]
    forbidden_final_keys = {
        "attempt_counts_by_behavior",
        "by_behavior",
        "record_count",
    }
    require(
        failures,
        not (forbidden_final_keys & set(final_public_summary)),
        "V2 combined audit exposes forbidden final summary keys",
    )
    forbidden_tokens = [
        "records",
        "weights",
        "signature",
        "subject_id",
        "attempt_count",
        "rejected_count",
        "acceptance_rate",
        "heldout_margin",
        "support_margin",
    ]
    final_public_text = json.dumps(final_public_summary, sort_keys=True)
    final_redacted_text = json.dumps(final_redacted, sort_keys=True)
    for token in forbidden_tokens:
        require(
            failures,
            token not in final_public_text,
            f"V2 combined final summary leaks token {token}",
        )
        require(
            failures,
            token not in final_redacted_text,
            f"V2 final redacted audit leaks token {token}",
        )
    require(
        failures,
        final_redacted["claim_scope"] == "redacted_final_source_pool_audit_surface_only",
        "V2 final redacted claim_scope drifted",
    )
    require(
        failures,
        final_redacted["summary"]["accepted_counts_by_behavior"]
        == final_public_summary["accepted_counts_by_behavior"],
        "V2 final redacted accepted counts mismatch combined audit",
    )
    require(
        failures,
        final_redacted["pool_file_sha256"] == result["pool_file_sha256"]["final"],
        "V2 final file hash mismatch between redacted and combined audit",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "final_redacted_artifact": rel(final_redacted_path),
        "accepted_counts": {
            pool: result["pool_summaries"][pool]["accepted_counts_by_behavior"]
            for pool in expected_counts
        },
        "overlap_counts": result["overlap_counts"],
        "pool_file_sha256": result["pool_file_sha256"],
        "seed_preflight": result["seed_preflight"],
        "interpretation": "positive_source_pool_construction_not_decoder_evidence",
    }


def audit_four_behavior_decoder_development_v1_negative() -> Dict[str, Any]:
    path = RUNS / "four_behavior_decoder_development_v1" / "results.json"
    result = load_json(path)
    failures = []
    aggregate = result["aggregate"]
    require(
        failures,
        result["claim_scope"] == "four_behavior_decoder_development_not_final_proof",
        "decoder development claim_scope drifted",
    )
    require(
        failures,
        result["development_status"] == "train_development_only_final_pool_sealed",
        "decoder development status drifted",
    )
    require(failures, result["passed"] is False, "decoder development unexpectedly passed")
    require(failures, result["best_epoch"] == 25, "decoder development best epoch drifted")
    require(failures, aggregate["n"] == 96, "decoder development n is not 96")
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == 0,
        "decoder development unexpectedly has individual passes",
    )
    require(
        failures,
        aggregate["mean_matched_target_margin"] < 0.01,
        "decoder development matched target margin no longer reflects failure",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_target_margin"] < 0.0,
        "decoder development best-control margin unexpectedly positive",
    )
    require(
        failures,
        aggregate["mean_best_control_minus_matched_subject_output_mse"] < 0.0,
        "decoder development subject-output specificity unexpectedly positive",
    )
    for pattern, summary in result["by_behavior"].items():
        require(
            failures,
            summary["n"] == 24 and summary["individual_all_gate_pass_count"] == 0,
            f"{pattern} decoder development pass count drifted",
        )
    input_audit = result["input_path_audit"]
    require(
        failures,
        input_audit["no_opened_path_endswith_final_subjects_json"],
        "decoder development opened final_subjects.json",
    )
    require(
        failures,
        input_audit["sealed_final_raw_path_not_opened"],
        "decoder development opened sealed final raw path",
    )
    result_text = json.dumps(result, sort_keys=True)
    require(
        failures,
        "final_subjects.json" not in result_text,
        "decoder development result names final raw artifact",
    )
    require(
        failures,
        all(value == 0 for value in result["overlap_counts"].values()),
        "decoder development train/development overlaps are nonzero",
    )
    require(
        failures,
        len(result["training_history"]) == 48,
        "decoder development checkpoint history length drifted",
    )
    records = result["development_records"]
    require(
        failures,
        len(records) == 96,
        "decoder development record count is not 96",
    )
    control_counts = [len(record["controls"]) for record in records]
    require(
        failures,
        all(count == 42 for count in control_counts),
        "decoder development controls are not 42 per subject",
    )
    noise_counts = [
        sum(1 for control in record["controls"] if control["control_type"].startswith("noise_signature:"))
        for record in records
    ]
    require(
        failures,
        all(count == 32 for count in noise_counts),
        "decoder development noise controls are not 32 per subject",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "by_behavior": result["by_behavior"],
        "input_path_audit": input_audit,
        "overlap_counts": result["overlap_counts"],
        "interpretation": "negative_decoder_development_not_final_proof",
    }


def audit_four_behavior_decoder_development_v2_negative() -> Dict[str, Any]:
    path = RUNS / "four_behavior_decoder_development_v2" / "results.json"
    result = load_json(path)
    failures = []
    aggregate = result["aggregate"]
    require(
        failures,
        result["claim_scope"] == "four_behavior_decoder_development_v2_not_final_proof",
        "decoder development V2 claim_scope drifted",
    )
    require(
        failures,
        result["development_status"]
        == "adaptive_v2_train_development_only_final_pool_sealed",
        "decoder development V2 status drifted",
    )
    require(failures, result["passed"] is False, "decoder development V2 unexpectedly passed")
    require(failures, result["best_epoch"] == 50, "decoder development V2 best epoch drifted")
    require(failures, aggregate["n"] == 96, "decoder development V2 n is not 96")
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == 0,
        "decoder development V2 unexpectedly has individual passes",
    )
    require(
        failures,
        aggregate["mean_matched_target_margin"] > 0.40,
        "decoder development V2 no longer reflects improved target-margin learning",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_target_margin"] < 0.0,
        "decoder development V2 best-control margin unexpectedly positive",
    )
    require(
        failures,
        aggregate["mean_best_control_minus_matched_subject_output_mse"] < 0.0,
        "decoder development V2 subject-output specificity unexpectedly positive",
    )
    for pattern, summary in result["by_behavior"].items():
        require(
            failures,
            summary["n"] == 24 and summary["individual_all_gate_pass_count"] == 0,
            f"{pattern} decoder development V2 pass count drifted",
        )
    require(
        failures,
        result["by_behavior"]["has_majority"]["mean_matched_target_margin"] < 0.20,
        "decoder development V2 has_majority unexpectedly clears target gate",
    )
    input_audit = result["input_path_audit"]
    require(
        failures,
        input_audit["no_opened_path_endswith_final_subjects_json"],
        "decoder development V2 opened final_subjects.json",
    )
    require(
        failures,
        input_audit["sealed_final_raw_path_not_opened"],
        "decoder development V2 opened sealed final raw path",
    )
    result_text = json.dumps(result, sort_keys=True)
    require(
        failures,
        "final_subjects.json" not in result_text,
        "decoder development V2 result names final raw artifact",
    )
    require(
        failures,
        result["distillation_case_count"] == 4096,
        "decoder development V2 distillation case count drifted",
    )
    require(
        failures,
        result["result_text_excludes_final_subjects_json"],
        "decoder development V2 final-subject text guard failed",
    )
    require(
        failures,
        all(value == 0 for value in result["overlap_counts"].values()),
        "decoder development V2 train/development overlaps are nonzero",
    )
    require(
        failures,
        len(result["training_history"]) == 24,
        "decoder development V2 checkpoint history length drifted",
    )
    records = result["development_records"]
    require(
        failures,
        len(records) == 96,
        "decoder development V2 record count is not 96",
    )
    control_counts = [len(record["controls"]) for record in records]
    require(
        failures,
        all(count == 42 for count in control_counts),
        "decoder development V2 controls are not 42 per subject",
    )
    noise_counts = [
        sum(
            1
            for control in record["controls"]
            if control["control_type"].startswith("noise_signature:")
        )
        for record in records
    ]
    require(
        failures,
        all(count == 32 for count in noise_counts),
        "decoder development V2 noise controls are not 32 per subject",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "by_behavior": result["by_behavior"],
        "distillation_case_hash": result["distillation_case_hash"],
        "input_path_audit": input_audit,
        "overlap_counts": result["overlap_counts"],
        "interpretation": "negative_adaptive_v2_decoder_development_not_final_proof",
    }


def audit_four_behavior_decoder_development_v3_signature_inversion_negative() -> Dict[str, Any]:
    path = RUNS / "four_behavior_decoder_development_v3_signature_inversion" / "results.json"
    result = load_json(path)
    failures = []
    aggregate = result["aggregate"]
    require(
        failures,
        result["claim_scope"]
        == "four_behavior_decoder_development_v3_signature_inversion_not_final_proof",
        "decoder development V3 claim_scope drifted",
    )
    require(
        failures,
        result["development_status"]
        == "adaptive_v3_signature_inversion_train_development_only_final_pool_sealed",
        "decoder development V3 status drifted",
    )
    require(failures, result["passed"] is False, "decoder development V3 unexpectedly passed")
    require(failures, aggregate["n"] == 96, "decoder development V3 n is not 96")
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == 0,
        "decoder development V3 unexpectedly has individual passes",
    )
    require(
        failures,
        aggregate["inferred_behavior_accuracy"] < 0.90,
        "decoder development V3 unexpectedly clears inferred-behavior gate",
    )
    require(
        failures,
        aggregate["mean_matched_target_margin"] > 0.20,
        "decoder development V3 no longer reflects nonzero target behavior",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_target_margin"] < 0.0,
        "decoder development V3 best-control margin unexpectedly positive",
    )
    require(
        failures,
        aggregate["mean_best_control_minus_matched_subject_output_mse"] < 0.0,
        "decoder development V3 subject-output specificity unexpectedly positive",
    )
    for pattern, summary in result["by_behavior"].items():
        require(
            failures,
            summary["n"] == 24 and summary["individual_all_gate_pass_count"] == 0,
            f"{pattern} decoder development V3 pass count drifted",
        )
    input_audit = result["input_path_audit"]
    require(
        failures,
        input_audit["no_opened_path_endswith_final_subjects_json"],
        "decoder development V3 opened final_subjects.json",
    )
    require(
        failures,
        input_audit["sealed_final_raw_path_not_opened"],
        "decoder development V3 opened sealed final raw path",
    )
    result_text = json.dumps(result, sort_keys=True)
    require(
        failures,
        "final_subjects.json" not in result_text,
        "decoder development V3 result names final raw artifact",
    )
    require(
        failures,
        result["query_count"] == 3744,
        "decoder development V3 query count drifted",
    )
    require(
        failures,
        all(value == 0 for value in result["overlap_counts"].values()),
        "decoder development V3 train/development overlaps are nonzero",
    )
    records = result["development_records"]
    require(
        failures,
        len(records) == 96,
        "decoder development V3 record count is not 96",
    )
    control_counts = [len(record["controls"]) for record in records]
    require(
        failures,
        all(count == 43 for count in control_counts),
        "decoder development V3 controls are not 43 per subject",
    )
    noise_counts = [
        sum(
            1
            for control in record["controls"]
            if control["control_type"].startswith("v3_inversion:noise_signature:")
        )
        for record in records
    ]
    require(
        failures,
        all(count == 32 for count in noise_counts),
        "decoder development V3 noise controls are not 32 per subject",
    )
    missing_delta_fields = []
    for record_index, record in enumerate(records):
        for control_index, control in enumerate(record["controls"]):
            for field in (
                "matched_minus_control_target_margin",
                "control_minus_matched_subject_output_mse",
            ):
                if field not in control:
                    missing_delta_fields.append((record_index, control_index, field))
    require(
        failures,
        not missing_delta_fields,
        "decoder development V3 controls are missing per-control delta fields",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "by_behavior": result["by_behavior"],
        "input_path_audit": input_audit,
        "inferred_behavior_confusion": result["inferred_behavior_confusion"],
        "overlap_counts": result["overlap_counts"],
        "interpretation": (
            "negative_adaptive_v3_signature_inversion_development_not_final_proof"
        ),
    }


def audit_four_behavior_representation_steering_v1_source_pools_positive() -> Dict[str, Any]:
    base = RUNS / "four_behavior_representation_steering_v1_pools"
    combined_path = base / "combined_audit.json"
    redacted_path = base / "final_redacted_audit.json"
    combined = load_json(combined_path)
    redacted = load_json(redacted_path)
    failures = []
    require(failures, combined["passed"], "representation steering source-pool audit failed")
    require(
        failures,
        combined["claim_scope"] == "four_behavior_representation_steering_source_pool_construction",
        "representation steering source-pool claim scope drifted",
    )
    required = {"train": 64, "development": 24, "final": 24}
    require(
        failures,
        combined["required_counts"] == required,
        "representation steering source-pool required counts drifted",
    )
    for pool_name, required_count in required.items():
        counts = combined["pool_summaries"][pool_name]["accepted_counts_by_behavior"]
        for pattern, count in counts.items():
            require(
                failures,
                count == required_count,
                f"representation steering {pool_name}/{pattern} count drifted",
            )
    require(
        failures,
        all(value == 0 for value in combined["overlap_counts"].values()),
        "representation steering source-pool overlaps are nonzero",
    )
    final_summary = redacted["summary"]
    require(
        failures,
        final_summary["accepted_counts_by_behavior"]
        == combined["pool_summaries"]["final"]["accepted_counts_by_behavior"],
        "representation steering final redacted counts drifted",
    )
    require(
        failures,
        final_summary["max_selected_train_vs_heldout_overlap_count"] == 0,
        "representation steering final selected-train overlap is nonzero",
    )
    redacted_text = json.dumps(redacted, sort_keys=True)
    forbidden_terms = [
        "accepted_subject_ids",
        "acceptance_rate",
        "attempt_count",
        "heldout_margin",
        "rejected_count",
        "rejected_subject_ids",
        "signature",
        "subject_id",
        "weights",
    ]
    leaked_terms = [term for term in forbidden_terms if term in redacted_text]
    require(
        failures,
        not leaked_terms,
        f"representation steering final redacted audit leaks terms: {leaked_terms}",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "combined_artifact": rel(combined_path),
        "final_redacted_artifact": rel(redacted_path),
        "overlap_counts": combined["overlap_counts"],
        "pool_file_sha256": combined["pool_file_sha256"],
        "interpretation": "positive_representation_steering_source_pool_construction_only",
    }


def audit_four_behavior_representation_steering_v1_development_negative() -> Dict[str, Any]:
    path = RUNS / "four_behavior_representation_steering_v1" / "development_results.json"
    result = load_json(path)
    failures = []
    aggregate = result["aggregate"]
    require(
        failures,
        result["claim_scope"] == "four_behavior_representation_steering_development",
        "representation steering development claim scope drifted",
    )
    require(failures, result["passed"] is False, "representation steering development unexpectedly passed")
    require(failures, aggregate["n"] == 288, "representation steering development n is not 288")
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == 16,
        "representation steering development pass count drifted",
    )
    require(
        failures,
        aggregate["individual_all_gate_pass_rate"] < 0.90,
        "representation steering development unexpectedly clears pass-rate gate",
    )
    require(
        failures,
        abs(
            aggregate["mean_matched_minus_best_control_centroid_improvement"]
            - (-1.053592734866672)
        )
        < 1e-9,
        "representation steering development corrected centroid specificity drifted",
    )
    require(
        failures,
        result["vector_training_summary"]["best_epoch"] == 140,
        "representation steering development corrected best epoch drifted",
    )
    require(
        failures,
        result.get("objective_correction_status")
        == "corrected_no_edit_relative_centroid_improvement",
        "representation steering development missing objective correction status",
    )
    require(
        failures,
        result.get("supersedes_development_artifact")
        == "flawed_v1_centroid_objective_55_of_288_passes",
        "representation steering development missing superseded artifact note",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_centroid_improvement"] < 0.0,
        "representation steering development centroid specificity unexpectedly positive",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_primary_target_margin"] > 0.0,
        "representation steering development no longer shows primary-margin movement",
    )
    require(
        failures,
        aggregate["mean_source_primary_margin_change"] < -0.05,
        "representation steering development source suppression drifted",
    )
    require(
        failures,
        result["next_action"] == "log_negative_development_result_do_not_open_final_raw",
        "representation steering development next action drifted",
    )
    require(
        failures,
        result["forbidden_decoder_final_raw_opened"] is False,
        "representation steering development opened decoder final raw",
    )
    result_text = json.dumps(result, sort_keys=True)
    require(
        failures,
        "runs/four_behavior_decoder_source_pools_v2/final_subjects.json" not in result_text,
        "representation steering development result names decoder final raw",
    )
    require(
        failures,
        "runs/four_behavior_representation_steering_v1_pools/final_subjects.json" not in result_text,
        "representation steering development result names steering final raw",
    )
    require(
        failures,
        "observed " in " ".join(result["failures"]),
        "representation steering development failures lack observed/required wording",
    )
    require(
        failures,
        len(result["records"]) == 288,
        "representation steering development record count drifted",
    )
    require(
        failures,
        all(
            sum(
                1
                for control in record["controls"]
                if control["control_type"] == "random_norm_matched_vector"
            )
            == 32
            for record in result["records"]
        ),
        "representation steering development random controls are not 32 per record",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "by_target": result["by_target"],
        "failure_count": len(result["failures"]),
        "interpretation": "negative_representation_steering_v1_development_blocks_final_eval",
    }


def audit_four_behavior_representation_steering_v1_failure_diagnosis() -> Dict[str, Any]:
    diagnosis_text = REP_STEERING_V1_DIAGNOSIS_PATH.read_text()
    result_path = RUNS / "four_behavior_representation_steering_v1" / "development_results.json"
    result = load_json(result_path)
    failures = []

    required_snippets = [
        "It is not a new proof result.",
        "does not inspect or use either raw final pool",
        "individual all-gate pass count: `16/288`",
        "mean matched-minus-best-control centroid improvement: `-1.0535927349`",
        "selected edit-vector epoch: `140`",
        "best centroid-improvement control type was `target_source_centroid_delta` for\n  `244/288` records",
        "matched steering beat the centroid-delta control on centroid improvement for\n  only `21/288` records",
        "mean matched-minus-centroid-delta centroid improvement was `-0.9224472046`",
        "negative for all\ntwelve directions",
        "valid negative for the frozen V1 protocol",
        "does not show\nthat fixed-probe representation steering is impossible",
        "requires a new\npreregistration and reviewer acceptance",
    ]
    for snippet in required_snippets:
        require(
            failures,
            snippet in diagnosis_text,
            f"missing diagnosis snippet: {snippet}",
        )

    aggregate = result["aggregate"]
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == 16,
        "diagnosis backing result pass count drifted",
    )
    require(
        failures,
        abs(
            aggregate["mean_matched_minus_best_control_centroid_improvement"]
            - (-1.053592734866672)
        )
        < 1e-9,
        "diagnosis backing result centroid specificity drifted",
    )
    require(
        failures,
        result["vector_training_summary"]["best_epoch"] == 140,
        "diagnosis backing result best epoch drifted",
    )

    best_primary_counts = count_record_summary_values(result, "best_primary_control_type")
    best_centroid_counts = count_record_summary_values(result, "best_centroid_control_type")
    require(
        failures,
        best_primary_counts == {
            "same_source_other_target_edit_vector": 3,
            "target_source_centroid_delta": 285,
        },
        "diagnosis backing result primary-control distribution drifted",
    )
    require(
        failures,
        best_centroid_counts == {
            "no_edit": 41,
            "same_source_other_target_edit_vector": 3,
            "target_source_centroid_delta": 244,
        },
        "diagnosis backing result centroid-control distribution drifted",
    )

    centroid_delta_diffs = [
        record["summary"]["matched_centroid_improvement"]
        - next(
            control["centroid_improvement"]
            for control in record["controls"]
            if control["control_type"] == "target_source_centroid_delta"
        )
        for record in result["records"]
    ]
    require(
        failures,
        sum(1 for value in centroid_delta_diffs if value > 0.0) == 21,
        "diagnosis backing result centroid-delta win count drifted",
    )
    require(
        failures,
        abs((sum(centroid_delta_diffs) / len(centroid_delta_diffs)) - (-0.9224472045898438))
        < 1e-9,
        "diagnosis backing result mean matched-minus-centroid-delta drifted",
    )

    direction_means = {}
    for record in result["records"]:
        direction_means.setdefault(record["vector_key"], []).append(
            record["summary"]["matched_centroid_improvement"]
            - next(
                control["centroid_improvement"]
                for control in record["controls"]
                if control["control_type"] == "target_source_centroid_delta"
            )
        )
    require(
        failures,
        all((sum(values) / len(values)) < 0.0 for values in direction_means.values()),
        "diagnosis backing result no longer has all negative direction means",
    )

    require(
        failures,
        "runs/four_behavior_representation_steering_v1_pools/final_subjects.json"
        in diagnosis_text,
        "diagnosis no longer names the sealed steering final raw pool as excluded",
    )
    require(
        failures,
        "runs/four_behavior_decoder_source_pools_v2/final_subjects.json"
        in diagnosis_text,
        "diagnosis no longer names the sealed decoder final raw pool as excluded",
    )

    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(REP_STEERING_V1_DIAGNOSIS_PATH),
        "backing_result": rel(result_path),
        "best_centroid_control_counts": best_centroid_counts,
        "best_primary_control_counts": best_primary_counts,
        "centroid_delta_win_count": sum(1 for value in centroid_delta_diffs if value > 0.0),
        "interpretation": "accepted_failure_diagnosis_not_steering_evidence",
    }


def audit_four_behavior_representation_steering_v2_source_pools_positive() -> Dict[str, Any]:
    base = RUNS / "four_behavior_representation_steering_v2_pools"
    combined_path = base / "combined_audit.json"
    redacted_path = base / "final_redacted_audit.json"
    combined = load_json(combined_path)
    redacted = load_json(redacted_path)
    failures = []
    require(
        failures,
        combined["claim_scope"] == "four_behavior_representation_steering_v2_source_pool_construction",
        "V2 steering source-pool claim scope drifted",
    )
    require(failures, combined["passed"] is True, "V2 steering source-pool audit did not pass")
    require(
        failures,
        redacted["claim_scope"] == "redacted_final_steering_v2_source_pool_audit_surface_only",
        "V2 steering final redacted claim scope drifted",
    )
    expected_counts = {"train": 64, "development": 24, "final": 24}
    for pool_name, expected_count in expected_counts.items():
        if pool_name == "final":
            counts = redacted["summary"]["accepted_counts_by_behavior"]
        else:
            counts = combined["pool_summaries"][pool_name]["accepted_counts_by_behavior"]
        for pattern in ("sorted_ascending", "sorted_descending", "has_majority", "mountain_pattern"):
            require(
                failures,
                counts[pattern] == expected_count,
                f"V2 {pool_name}/{pattern} accepted count drifted",
            )
    for key, value in combined["overlap_counts"].items():
        require(failures, int(value) == 0, f"V2 source-pool overlap drifted: {key}")
    require(
        failures,
        combined["seed_preflight"]["passed"] is True,
        "V2 source-pool seed preflight did not pass",
    )
    final_summary_text = json.dumps(combined["pool_summaries"]["final"], sort_keys=True)
    forbidden_final_terms = [
        "accepted_subject_ids",
        "rejected_subject_ids",
        "attempt_count",
        "acceptance_rate",
        "heldout_margin",
        "records",
    ]
    for term in forbidden_final_terms:
        require(
            failures,
            term not in final_summary_text,
            f"V2 combined audit exposes forbidden final detail: {term}",
        )
        require(
            failures,
            term not in json.dumps(redacted, sort_keys=True),
            f"V2 final redacted audit exposes forbidden final detail: {term}",
        )
    return {
        "passed": not failures,
        "failures": failures,
        "combined_artifact": rel(combined_path),
        "final_redacted_artifact": rel(redacted_path),
        "overlap_counts": combined["overlap_counts"],
        "interpretation": "positive_v2_source_pool_construction_only",
    }


def audit_four_behavior_representation_steering_v2_development_negative() -> Dict[str, Any]:
    path = RUNS / "four_behavior_representation_steering_v2_centroid_delta" / "development_results.json"
    result = load_json(path)
    failures = []
    aggregate = result["aggregate"]
    require(
        failures,
        result["claim_scope"] == "four_behavior_representation_steering_v2_centroid_delta_development",
        "V2 steering development claim scope drifted",
    )
    require(failures, result["passed"] is False, "V2 steering development unexpectedly passed")
    require(failures, aggregate["n"] == 288, "V2 steering development n drifted")
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == 142,
        "V2 steering development pass count drifted",
    )
    require(
        failures,
        abs(aggregate["individual_all_gate_pass_rate"] - 0.4930555555555556) < 1e-12,
        "V2 steering development pass rate drifted",
    )
    require(
        failures,
        aggregate["mean_matched_centroid_improvement"] > 0.0,
        "V2 steering development no longer shows centroid movement",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_centroid_improvement"] > 0.0,
        "V2 steering development no longer beats controls on mean centroid movement",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_primary_target_margin"] > 0.0,
        "V2 steering development no longer beats controls on mean primary margin",
    )
    require(
        failures,
        aggregate["mean_source_primary_margin_change"] < -0.05,
        "V2 steering development source suppression drifted",
    )
    require(
        failures,
        result["next_action"] == "log_negative_development_result_do_not_open_final_raw",
        "V2 steering development next action drifted",
    )
    require(
        failures,
        result["edit_vector_method"] == "train_centroid_delta_no_optimizer",
        "V2 steering development edit-vector method drifted",
    )
    require(
        failures,
        result["vector_training_summary"]["best_epoch"] is None,
        "V2 steering development unexpectedly has a learned-vector epoch",
    )
    require(
        failures,
        all(
            summary["individual_all_gate_pass_rate"] < 0.80
            for summary in result["by_target"].values()
        ),
        "V2 steering development target pass-rate failure drifted",
    )
    require(
        failures,
        all(
            summary["individual_all_gate_pass_rate"] < 0.90
            for summary in result["by_direction"].values()
        ),
        "V2 steering development direction pass-rate failure drifted",
    )
    result_text = json.dumps(result, sort_keys=True)
    require(
        failures,
        "final_subjects.json" not in result_text,
        "V2 steering development result names a raw final pool",
    )
    require(
        failures,
        "combined audit exposes forbidden" not in result_text,
        "V2 steering development includes stale redaction false-positive failure",
    )
    require(
        failures,
        "final redacted audit exposes forbidden" not in result_text,
        "V2 steering development includes stale final-redaction false-positive failure",
    )
    require(
        failures,
        len(result["records"]) == 288,
        "V2 steering development record count drifted",
    )
    require(
        failures,
        all(
            sum(
                1
                for control in record["controls"]
                if control["control_type"] == "random_norm_matched_vector"
            )
            == 32
            for record in result["records"]
        ),
        "V2 steering development random controls are not 32 per record",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "by_target": result["by_target"],
        "failure_count": len(result["failures"]),
        "interpretation": "negative_v2_centroid_delta_development_blocks_final_eval",
    }


def audit_four_behavior_representation_steering_v2_failure_diagnosis() -> Dict[str, Any]:
    diagnosis_path = REPO_ROOT / "docs" / "representation_steering_v2_failure_diagnosis.md"
    diagnosis_text = diagnosis_path.read_text()
    result_path = RUNS / "four_behavior_representation_steering_v2_centroid_delta" / "development_results.json"
    result = load_json(result_path)
    failures = []

    required_snippets = [
        "It is not a new proof result.",
        "individual all-gate pass count: `142/288`",
        "individual all-gate pass rate: `0.4930555556`",
        "mean matched-minus-best-control centroid improvement: `0.4016633564`",
        "Final evaluation is blocked under the V2 preregistration.",
        "other source-to-same-target deltas",
        "source-specificity",
        "does not support proof-grade four-behavior representation steering",
        "requires a new preregistration and reviewer acceptance",
    ]
    for snippet in required_snippets:
        require(
            failures,
            snippet in diagnosis_text,
            f"missing V2 diagnosis snippet: {snippet}",
        )

    aggregate = result["aggregate"]
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == 142,
        "V2 diagnosis backing pass count drifted",
    )
    require(
        failures,
        abs(aggregate["individual_all_gate_pass_rate"] - 0.4930555555555556) < 1e-12,
        "V2 diagnosis backing pass rate drifted",
    )
    require(
        failures,
        result["next_action"] == "log_negative_development_result_do_not_open_final_raw",
        "V2 diagnosis backing result no longer blocks final",
    )
    best_primary_counts = count_record_summary_values(result, "best_primary_control_type")
    best_centroid_counts = count_record_summary_values(result, "best_centroid_control_type")
    require(
        failures,
        best_primary_counts == {
            "same_source_other_target_centroid_delta": 10,
            "same_target_other_source_centroid_delta": 278,
        },
        "V2 diagnosis backing primary-control distribution drifted",
    )
    require(
        failures,
        best_centroid_counts == {
            "no_edit": 46,
            "same_source_other_target_centroid_delta": 102,
            "same_target_other_source_centroid_delta": 140,
        },
        "V2 diagnosis backing centroid-control distribution drifted",
    )
    require(
        failures,
        "final_subjects.json" not in json.dumps(result, sort_keys=True),
        "V2 diagnosis backing result names a raw final pool",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(diagnosis_path),
        "backing_result": rel(result_path),
        "best_centroid_control_counts": best_centroid_counts,
        "best_primary_control_counts": best_primary_counts,
        "interpretation": "accepted_v2_failure_diagnosis_not_steering_evidence",
    }


def audit_four_behavior_representation_steering_v3_source_pools_positive() -> Dict[str, Any]:
    base = RUNS / "four_behavior_representation_steering_v3_pools"
    combined_path = base / "combined_audit.json"
    redacted_path = base / "final_redacted_audit.json"
    combined = load_json(combined_path)
    redacted = load_json(redacted_path)
    failures = []
    require(
        failures,
        combined["claim_scope"] == "four_behavior_representation_steering_v3_source_pool_construction",
        "V3 steering source-pool claim scope drifted",
    )
    require(failures, combined["passed"] is True, "V3 steering source-pool audit did not pass")
    require(
        failures,
        redacted["claim_scope"] == "redacted_final_steering_v3_source_pool_audit_surface_only",
        "V3 steering final redacted claim scope drifted",
    )
    expected_counts = {"train": 64, "development": 24, "final": 24}
    require(
        failures,
        combined["required_counts"] == expected_counts,
        "V3 steering source-pool required counts drifted",
    )
    for pool_name, expected_count in expected_counts.items():
        if pool_name == "final":
            counts = redacted["summary"]["accepted_counts_by_behavior"]
        else:
            counts = combined["pool_summaries"][pool_name]["accepted_counts_by_behavior"]
        for pattern in ("sorted_ascending", "sorted_descending", "has_majority", "mountain_pattern"):
            require(
                failures,
                counts[pattern] == expected_count,
                f"V3 {pool_name}/{pattern} accepted count drifted",
            )
    for key, value in combined["overlap_counts"].items():
        require(failures, int(value) == 0, f"V3 source-pool overlap drifted: {key}")
    require(
        failures,
        combined["seed_preflight"]["passed"] is True,
        "V3 source-pool seed preflight did not pass",
    )
    final_summary_text = json.dumps(combined["pool_summaries"]["final"], sort_keys=True)
    final_redacted_text = json.dumps(redacted, sort_keys=True)
    forbidden_final_terms = [
        "accepted_subject_ids",
        "rejected_subject_ids",
        "attempt_count",
        "acceptance_rate",
        "heldout_margin",
        "records",
        "signature",
        "subject_id",
        "weights",
    ]
    for term in forbidden_final_terms:
        require(
            failures,
            term not in final_summary_text,
            f"V3 combined audit exposes forbidden final detail: {term}",
        )
        require(
            failures,
            term not in final_redacted_text,
            f"V3 final redacted audit exposes forbidden final detail: {term}",
        )
    require(
        failures,
        redacted["summary"]["max_selected_train_vs_heldout_overlap_count"] == 0,
        "V3 final redacted selected-train overlap is nonzero",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "combined_artifact": rel(combined_path),
        "final_redacted_artifact": rel(redacted_path),
        "overlap_counts": combined["overlap_counts"],
        "interpretation": "positive_v3_source_pool_construction_only",
    }


def audit_four_behavior_representation_steering_v3_development_negative() -> Dict[str, Any]:
    path = (
        RUNS
        / "four_behavior_representation_steering_v3_diagonal_transport"
        / "development_results.json"
    )
    result = load_json(path)
    failures = []
    aggregate = result["aggregate"]
    require(
        failures,
        result["claim_scope"] == "four_behavior_representation_steering_v3_diagonal_transport_development",
        "V3 steering development claim scope drifted",
    )
    require(failures, result["passed"] is False, "V3 steering development unexpectedly passed")
    require(
        failures,
        result["next_action"] == "log_negative_development_result_do_not_open_final_raw",
        "V3 steering development next action drifted",
    )
    require(failures, result["source_pool_audit_passed"] is True, "V3 source-pool audit failed")
    require(failures, aggregate["n"] == 288, "V3 steering development n drifted")
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == 30,
        "V3 steering development pass count drifted",
    )
    require(
        failures,
        abs(aggregate["individual_all_gate_pass_rate"] - 0.10416666666666667) < 1e-12,
        "V3 steering development pass rate drifted",
    )
    require(
        failures,
        aggregate["mean_matched_primary_target_margin"] > 0.2,
        "V3 steering development no longer shows primary target movement",
    )
    require(
        failures,
        aggregate["mean_matched_centroid_improvement"] > 0.15,
        "V3 steering development no longer shows centroid movement",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v2_centroid_delta_primary_target_margin"] > 0.1,
        "V3 no longer beats V2 centroid delta on mean primary target margin",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v2_centroid_delta_centroid_improvement"] > 0.1,
        "V3 no longer beats V2 centroid delta on mean centroid improvement",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_primary_target_margin"] < 0.0,
        "V3 primary best-control specificity unexpectedly positive",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_centroid_improvement"] < 0.0,
        "V3 centroid best-control specificity unexpectedly positive",
    )
    require(
        failures,
        aggregate["mean_source_primary_margin_change"] < -0.05,
        "V3 steering development source suppression drifted",
    )
    require(
        failures,
        result["transport_method"] == "train_diagonal_covariance_transport_no_optimizer",
        "V3 steering development transport method drifted",
    )
    require(
        failures,
        result["vector_training_summary"]["best_epoch"] is None,
        "V3 steering development unexpectedly has a learned-vector epoch",
    )
    require(
        failures,
        all(
            summary["individual_all_gate_pass_rate"] < 0.80
            for summary in result["by_target"].values()
        ),
        "V3 steering development target pass-rate failure drifted",
    )
    require(
        failures,
        all(
            summary["individual_all_gate_pass_rate"] < 0.90
            for summary in result["by_direction"].values()
        ),
        "V3 steering development direction pass-rate failure drifted",
    )
    require(
        failures,
        len(result["records"]) == 288,
        "V3 steering development record count drifted",
    )
    require(
        failures,
        all(len(record["controls"]) == 41 for record in result["records"]),
        "V3 steering development controls are not 41 per record",
    )
    require(
        failures,
        all(
            sum(
                1
                for control in record["controls"]
                if control["control_type"] == "random_norm_matched_vector"
            )
            == 32
            for record in result["records"]
        ),
        "V3 steering development random controls are not 32 per record",
    )
    best_primary_counts = count_record_summary_values(result, "best_primary_control_type")
    best_centroid_counts = count_record_summary_values(result, "best_centroid_control_type")
    require(
        failures,
        best_primary_counts == {
            "same_target_other_source_diagonal_transport": 63,
            "v2_centroid_delta": 225,
        },
        "V3 steering development primary-control distribution drifted",
    )
    require(
        failures,
        best_centroid_counts == {
            "no_edit": 13,
            "reverse_diagonal_transport": 1,
            "same_source_other_target_diagonal_transport": 66,
            "same_target_other_source_diagonal_transport": 71,
            "shuffled_diagonal_transport": 2,
            "v2_centroid_delta": 135,
        },
        "V3 steering development centroid-control distribution drifted",
    )
    result_text = json.dumps(result, sort_keys=True)
    require(
        failures,
        "final_subjects.json" not in result_text,
        "V3 steering development result names a raw final pool",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "by_target": result["by_target"],
        "best_centroid_control_counts": best_centroid_counts,
        "best_primary_control_counts": best_primary_counts,
        "failure_count": len(result["failures"]),
        "interpretation": "negative_v3_diagonal_transport_development_blocks_final_eval",
    }


def audit_four_behavior_representation_steering_v4_source_pools_positive() -> Dict[str, Any]:
    base = RUNS / "four_behavior_representation_steering_v4_pools"
    combined_path = base / "combined_audit.json"
    redacted_path = base / "final_redacted_audit.json"
    combined = load_json(combined_path)
    redacted = load_json(redacted_path)
    failures = []
    require(
        failures,
        combined["claim_scope"] == "four_behavior_representation_steering_v4_source_pool_construction",
        "V4 steering source-pool claim scope drifted",
    )
    require(failures, combined["passed"] is True, "V4 steering source-pool audit did not pass")
    require(
        failures,
        redacted["claim_scope"] == "redacted_final_steering_v4_source_pool_audit_surface_only",
        "V4 steering final redacted claim scope drifted",
    )
    expected_counts = {"train": 64, "development": 24, "final": 24}
    require(
        failures,
        combined["required_counts"] == expected_counts,
        "V4 steering source-pool required counts drifted",
    )
    for pool_name, expected_count in expected_counts.items():
        if pool_name == "final":
            counts = redacted["summary"]["accepted_counts_by_behavior"]
        else:
            counts = combined["pool_summaries"][pool_name]["accepted_counts_by_behavior"]
        for pattern in ("sorted_ascending", "sorted_descending", "has_majority", "mountain_pattern"):
            require(
                failures,
                counts[pattern] == expected_count,
                f"V4 {pool_name}/{pattern} accepted count drifted",
            )
    for key, value in combined["overlap_counts"].items():
        require(failures, int(value) == 0, f"V4 source-pool overlap drifted: {key}")
    require(
        failures,
        combined["seed_preflight"]["passed"] is True,
        "V4 source-pool seed preflight did not pass",
    )
    require(
        failures,
        combined["seed_preflight"]["seed_ranges"][0]["start_seed"] == 39300000,
        "V4 source-pool train seed base drifted",
    )
    expected_pool_hashes = {
        "train": "83484f74c43b02ab8aa5ae1b91ffa88066b5c438d0d47aa502519075a37ead8b",
        "development": "f08f8d5aecf30ddfa301baff889b834ca70ec60e8180ec6a6c995d6a0fa993db",
        "final": "7df5cc1ebb826e6e00a3679af7cd443c9ea84980a0e6a546fd9533ea9944bd93",
    }
    require(
        failures,
        combined["pool_file_sha256"] == expected_pool_hashes,
        "V4 source-pool raw pool file hashes drifted",
    )
    require(
        failures,
        redacted["pool_file_sha256"] == expected_pool_hashes["final"],
        "V4 final redacted audit file hash drifted",
    )
    final_summary_text = json.dumps(combined["pool_summaries"]["final"], sort_keys=True)
    final_redacted_text = json.dumps(redacted, sort_keys=True)
    forbidden_final_terms = [
        "accepted_subject_ids",
        "candidate_records",
        "heldout_margin",
        "primary_source_margin",
        "records",
        "rejected_subject_ids",
        "signature",
        "signature_hash",
        "source_margin",
        "subject_id",
        "support_margin",
        "weights",
        "weights_hash",
    ]
    for term in forbidden_final_terms:
        require(
            failures,
            term not in final_summary_text,
            f"V4 combined audit exposes forbidden final detail: {term}",
        )
        require(
            failures,
            term not in final_redacted_text,
            f"V4 final redacted audit exposes forbidden final detail: {term}",
        )
    require(
        failures,
        redacted["summary"]["max_selected_train_vs_heldout_overlap_count"] == 0,
        "V4 final redacted selected-train overlap is nonzero",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "combined_artifact": rel(combined_path),
        "final_redacted_artifact": rel(redacted_path),
        "overlap_counts": combined["overlap_counts"],
        "pool_file_sha256": combined["pool_file_sha256"],
        "interpretation": "positive_v4_source_pool_construction_only",
    }


def audit_four_behavior_representation_steering_v4_development_negative() -> Dict[str, Any]:
    path = (
        RUNS
        / "four_behavior_representation_steering_v4_low_rank_residual_transport"
        / "development_results.json"
    )
    result = load_json(path)
    failures = []
    aggregate = result["aggregate"]
    require(
        failures,
        (
            result["claim_scope"]
            == "four_behavior_representation_steering_v4_low_rank_residual_transport_development"
        ),
        "V4 steering development claim scope drifted",
    )
    require(failures, result["passed"] is False, "V4 steering development unexpectedly passed")
    require(
        failures,
        result["next_action"] == "log_negative_development_result_do_not_open_final_raw",
        "V4 steering development next action drifted",
    )
    require(failures, result["source_pool_audit_passed"] is True, "V4 source-pool audit failed")
    require(failures, aggregate["n"] == 288, "V4 steering development n drifted")
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == 42,
        "V4 steering development pass count drifted",
    )
    require(
        failures,
        abs(aggregate["individual_all_gate_pass_rate"] - 0.14583333333333334) < 1e-12,
        "V4 steering development pass rate drifted",
    )
    require(
        failures,
        aggregate["mean_matched_primary_target_margin"] > 0.2,
        "V4 steering development no longer shows primary target movement",
    )
    require(
        failures,
        aggregate["mean_matched_centroid_improvement"] > 0.15,
        "V4 steering development no longer shows centroid movement",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v2_centroid_delta_centroid_improvement"] > 0.1,
        "V4 no longer beats V2 centroid delta on mean centroid improvement",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v3_diagonal_transport_centroid_improvement"] > 0.1,
        "V4 no longer beats V3 diagonal transport on mean centroid improvement",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v2_centroid_delta_primary_target_margin"] < 0.0,
        "V4 unexpectedly beats V2 centroid delta on mean primary target margin",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v3_diagonal_transport_primary_target_margin"] < 0.0,
        "V4 unexpectedly beats V3 diagonal transport on mean primary target margin",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_primary_target_margin"] < 0.0,
        "V4 primary best-control specificity unexpectedly positive",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_centroid_improvement"] < 0.0,
        "V4 centroid best-control specificity unexpectedly positive",
    )
    require(
        failures,
        aggregate["mean_source_primary_margin_change"] < -0.05,
        "V4 steering development source suppression drifted",
    )
    require(
        failures,
        result["transport_method"] == "train_low_rank_residual_covariance_transport_no_optimizer",
        "V4 steering development transport method drifted",
    )
    config = result["training_config"]
    require(failures, config["pca_rank"] == 48, "V4 PCA rank drifted")
    require(
        failures,
        abs(config["covariance_shrinkage_behavior_weight"] - 0.75) < 1e-12,
        "V4 behavior covariance shrinkage weight drifted",
    )
    require(
        failures,
        abs(config["covariance_shrinkage_global_weight"] - 0.25) < 1e-12,
        "V4 global covariance shrinkage weight drifted",
    )
    require(
        failures,
        abs(config["orthogonal_residual_carry_weight"] - 0.0) < 1e-12,
        "V4 orthogonal residual carry weight drifted",
    )
    require(
        failures,
        abs(config["displacement_norm_cap"] - 200.0) < 1e-12,
        "V4 displacement norm cap drifted",
    )
    require(
        failures,
        result["vector_training_summary"]["best_epoch"] is None,
        "V4 steering development unexpectedly has a learned-vector epoch",
    )
    require(
        failures,
        result["train_pool_sha256"]
        == "83484f74c43b02ab8aa5ae1b91ffa88066b5c438d0d47aa502519075a37ead8b",
        "V4 train pool hash drifted",
    )
    require(
        failures,
        result["eval_pool_sha256"]
        == "f08f8d5aecf30ddfa301baff889b834ca70ec60e8180ec6a6c995d6a0fa993db",
        "V4 development pool hash drifted",
    )
    require(
        failures,
        result["combined_audit_sha256"] == sha256_file(
            RUNS / "four_behavior_representation_steering_v4_pools" / "combined_audit.json"
        ),
        "V4 combined audit hash does not match artifact",
    )
    require(
        failures,
        result["final_redacted_audit_sha256"] == sha256_file(
            RUNS / "four_behavior_representation_steering_v4_pools" / "final_redacted_audit.json"
        ),
        "V4 final redacted audit hash does not match artifact",
    )
    require(
        failures,
        result["train_only_statistics_hash"]
        == "5db35e5a4faf01e83e3947bc4b1e324f0cce04b0e47de144a878fea7aaca8e9b",
        "V4 train-only statistics hash drifted",
    )
    require(
        failures,
        all(
            summary["individual_all_gate_pass_rate"] < 0.80
            for summary in result["by_target"].values()
        ),
        "V4 steering development target pass-rate failure drifted",
    )
    require(
        failures,
        all(
            summary["individual_all_gate_pass_rate"] < 0.90
            for summary in result["by_direction"].values()
        ),
        "V4 steering development direction pass-rate failure drifted",
    )
    require(
        failures,
        len(result["records"]) == 288,
        "V4 steering development record count drifted",
    )
    require(
        failures,
        all(len(record["controls"]) == 42 for record in result["records"]),
        "V4 steering development controls are not 42 per record",
    )
    require(
        failures,
        all(
            sum(
                1
                for control in record["controls"]
                if control["control_type"] == "random_norm_matched_vector"
            )
            == 32
            for record in result["records"]
        ),
        "V4 steering development random controls are not 32 per record",
    )
    best_primary_counts = count_record_summary_values(result, "best_primary_control_type")
    best_centroid_counts = count_record_summary_values(result, "best_centroid_control_type")
    require(
        failures,
        best_primary_counts == {
            "same_target_other_source_low_rank_residual_transport": 92,
            "v2_centroid_delta": 104,
            "v3_diagonal_transport": 92,
        },
        "V4 steering development primary-control distribution drifted",
    )
    require(
        failures,
        best_centroid_counts == {
            "reverse_low_rank_residual_transport": 6,
            "same_source_other_target_low_rank_residual_transport": 160,
            "same_target_other_source_low_rank_residual_transport": 104,
            "shuffled_low_rank_residual_transport": 7,
            "v2_centroid_delta": 2,
            "v3_diagonal_transport": 9,
        },
        "V4 steering development centroid-control distribution drifted",
    )
    result_text = json.dumps(result, sort_keys=True)
    require(
        failures,
        "final_subjects.json" not in result_text,
        "V4 steering development result names a raw final pool",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "by_target": result["by_target"],
        "best_centroid_control_counts": best_centroid_counts,
        "best_primary_control_counts": best_primary_counts,
        "failure_count": len(result["failures"]),
        "interpretation": "negative_v4_low_rank_residual_transport_development_blocks_final_eval",
    }


def audit_four_behavior_representation_steering_v5_source_pools_positive() -> Dict[str, Any]:
    base = RUNS / "four_behavior_representation_steering_v5_pools"
    combined_path = base / "combined_audit.json"
    redacted_path = base / "final_redacted_audit.json"
    combined = load_json(combined_path)
    redacted = load_json(redacted_path)
    failures = []
    require(
        failures,
        combined["claim_scope"] == "four_behavior_representation_steering_v5_source_pool_construction",
        "V5 steering source-pool claim scope drifted",
    )
    require(failures, combined["passed"] is True, "V5 steering source-pool audit did not pass")
    require(
        failures,
        redacted["claim_scope"] == "redacted_final_steering_v5_source_pool_audit_surface_only",
        "V5 steering final redacted claim scope drifted",
    )
    expected_counts = {"train": 64, "development": 24, "final": 24}
    require(
        failures,
        combined["required_counts"] == expected_counts,
        "V5 steering source-pool required counts drifted",
    )
    for pool_name, expected_count in expected_counts.items():
        if pool_name == "final":
            counts = redacted["summary"]["accepted_counts_by_behavior"]
        else:
            counts = combined["pool_summaries"][pool_name]["accepted_counts_by_behavior"]
        for pattern in ("sorted_ascending", "sorted_descending", "has_majority", "mountain_pattern"):
            require(
                failures,
                counts[pattern] == expected_count,
                f"V5 {pool_name}/{pattern} accepted count drifted",
            )
    for key, value in combined["overlap_counts"].items():
        require(failures, int(value) == 0, f"V5 source-pool overlap drifted: {key}")
    require(
        failures,
        combined["seed_preflight"]["passed"] is True,
        "V5 source-pool seed preflight did not pass",
    )
    require(
        failures,
        combined["seed_preflight"]["seed_ranges"][0]["start_seed"] == 42300000,
        "V5 source-pool train seed base drifted",
    )
    expected_pool_hashes = {
        "train": "49923d35cedee5bf5359f91f47e7a5873c8ac99b498cb855b5d2c20037413365",
        "development": "31f4c31a435d87330afe97aa979b679c5a85f9d8a369922c2854daec322d94b0",
        "final": "0c93ef6a2977e0a810d0dac9624dfcee2da6c17dea77e3cc2e5e81f12b5c355d",
    }
    require(
        failures,
        combined["pool_file_sha256"] == expected_pool_hashes,
        "V5 source-pool raw pool file hashes drifted",
    )
    require(
        failures,
        redacted["pool_file_sha256"] == expected_pool_hashes["final"],
        "V5 final redacted audit file hash drifted",
    )
    final_summary_text = json.dumps(combined["pool_summaries"]["final"], sort_keys=True)
    final_redacted_text = json.dumps(redacted, sort_keys=True)
    forbidden_final_terms = [
        "accepted_subject_ids",
        "candidate_records",
        "heldout_margin",
        "primary_source_margin",
        "records",
        "rejected_subject_ids",
        "signature",
        "signature_hash",
        "source_margin",
        "subject_id",
        "support_margin",
        "weights",
        "weights_hash",
    ]
    for term in forbidden_final_terms:
        require(
            failures,
            term not in final_summary_text,
            f"V5 combined audit exposes forbidden final detail: {term}",
        )
        require(
            failures,
            term not in final_redacted_text,
            f"V5 final redacted audit exposes forbidden final detail: {term}",
        )
    require(
        failures,
        redacted["summary"]["max_selected_train_vs_heldout_overlap_count"] == 0,
        "V5 final redacted selected-train overlap is nonzero",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "combined_artifact": rel(combined_path),
        "final_redacted_artifact": rel(redacted_path),
        "overlap_counts": combined["overlap_counts"],
        "pool_file_sha256": combined["pool_file_sha256"],
        "interpretation": "positive_v5_source_pool_construction_only",
    }


def audit_four_behavior_representation_steering_v5_development_negative() -> Dict[str, Any]:
    path = (
        RUNS
        / "four_behavior_representation_steering_v5_contrastive_residual_calibration"
        / "development_results.json"
    )
    result = load_json(path)
    failures = []
    aggregate = result["aggregate"]
    require(
        failures,
        (
            result["claim_scope"]
            == "four_behavior_representation_steering_v5_contrastive_residual_calibration_development"
        ),
        "V5 steering development claim scope drifted",
    )
    require(failures, result["passed"] is False, "V5 steering development unexpectedly passed")
    require(
        failures,
        result["next_action"] == "log_negative_development_result_do_not_open_final_raw",
        "V5 steering development next action drifted",
    )
    require(failures, result["source_pool_audit_passed"] is True, "V5 source-pool audit failed")
    require(failures, aggregate["n"] == 288, "V5 steering development n drifted")
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == 20,
        "V5 steering development pass count drifted",
    )
    require(
        failures,
        abs(aggregate["individual_all_gate_pass_rate"] - 0.06944444444444445) < 1e-12,
        "V5 steering development pass rate drifted",
    )
    require(
        failures,
        aggregate["mean_matched_primary_target_margin"] > 100.0,
        "V5 steering development no longer shows strong primary target movement",
    )
    require(
        failures,
        aggregate["mean_source_primary_margin_change"] < -100.0,
        "V5 steering development source suppression drifted",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_primary_target_margin"] > 0.0,
        "V5 no longer beats best control on mean primary target margin",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_centroid_improvement"] < 0.0,
        "V5 centroid best-control specificity unexpectedly positive",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v2_centroid_delta_primary_target_margin"] > 0.1,
        "V5 no longer beats V2 on mean primary target margin",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v3_diagonal_transport_primary_target_margin"] > 0.1,
        "V5 no longer beats V3 on mean primary target margin",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v4_low_rank_primary_target_margin"] > 0.1,
        "V5 no longer beats V4 on mean primary target margin",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v4_low_rank_centroid_improvement"] < 0.0,
        "V5 unexpectedly beats V4 on mean centroid improvement",
    )
    require(
        failures,
        result["transport_method"] == "train_contrastive_residual_calibration_on_v4_low_rank_transport",
        "V5 steering development transport method drifted",
    )
    config = result["training_config"]
    require(failures, config["pca_rank"] == 48, "V5 PCA rank drifted")
    require(failures, config["calibration_epochs"] == 500, "V5 calibration epochs drifted")
    require(
        failures,
        abs(config["calibration_lr"] - 0.03) < 1e-12,
        "V5 calibration lr drifted",
    )
    require(
        failures,
        result["vector_training_summary"]["best_epoch"] is None,
        "V5 steering development unexpectedly selected a best epoch",
    )
    require(
        failures,
        result["vector_training_summary"]["final_epoch"] == 500,
        "V5 steering development final epoch drifted",
    )
    require(
        failures,
        result["train_pool_sha256"]
        == "49923d35cedee5bf5359f91f47e7a5873c8ac99b498cb855b5d2c20037413365",
        "V5 train pool hash drifted",
    )
    require(
        failures,
        result["eval_pool_sha256"]
        == "31f4c31a435d87330afe97aa979b679c5a85f9d8a369922c2854daec322d94b0",
        "V5 development pool hash drifted",
    )
    require(
        failures,
        result["combined_audit_sha256"] == sha256_file(
            RUNS / "four_behavior_representation_steering_v5_pools" / "combined_audit.json"
        ),
        "V5 combined audit hash does not match artifact",
    )
    require(
        failures,
        result["final_redacted_audit_sha256"] == sha256_file(
            RUNS / "four_behavior_representation_steering_v5_pools" / "final_redacted_audit.json"
        ),
        "V5 final redacted audit hash does not match artifact",
    )
    require(
        failures,
        result["train_only_statistics_hash"]
        == "371afbcc3aa87e068d43b10f7afbeabedeed5332dd46b7fd226851c495c9d147",
        "V5 train-only statistics hash drifted",
    )
    require(
        failures,
        result["calibration_coefficient_hash"]
        == "18288f4ecf046002b1640472eb09d0c2d00157de39fb350105449c626c82a5f2",
        "V5 calibration coefficient hash drifted",
    )
    require(
        failures,
        all(
            summary["individual_all_gate_pass_rate"] < 0.80
            for summary in result["by_target"].values()
        ),
        "V5 steering development target pass-rate failure drifted",
    )
    require(
        failures,
        all(
            summary["individual_all_gate_pass_rate"] < 0.90
            for summary in result["by_direction"].values()
        ),
        "V5 steering development direction pass-rate failure drifted",
    )
    require(
        failures,
        len(result["records"]) == 288,
        "V5 steering development record count drifted",
    )
    require(
        failures,
        all(len(record["controls"]) == 43 for record in result["records"]),
        "V5 steering development controls are not 43 per record",
    )
    require(
        failures,
        all(
            sum(
                1
                for control in record["controls"]
                if control["control_type"] == "random_norm_matched_vector"
            )
            == 32
            for record in result["records"]
        ),
        "V5 steering development random controls are not 32 per record",
    )
    best_primary_counts = count_record_summary_values(result, "best_primary_control_type")
    best_centroid_counts = count_record_summary_values(result, "best_centroid_control_type")
    require(
        failures,
        best_primary_counts == {
            "same_target_other_source_v5_calibrated_transport": 205,
            "v2_centroid_delta": 24,
            "v3_diagonal_transport": 31,
            "v4_low_rank_residual_transport": 28,
        },
        "V5 steering development primary-control distribution drifted",
    )
    require(
        failures,
        best_centroid_counts == {
            "same_source_other_target_v5_calibrated_transport": 68,
            "same_target_other_source_v5_calibrated_transport": 20,
            "shuffled_v5_calibrated_transport": 4,
            "v2_centroid_delta": 3,
            "v4_low_rank_residual_transport": 193,
        },
        "V5 steering development centroid-control distribution drifted",
    )
    result_text = json.dumps(result, sort_keys=True)
    require(
        failures,
        "final_subjects.json" not in result_text,
        "V5 steering development result names a raw final pool",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "by_target": result["by_target"],
        "best_centroid_control_counts": best_centroid_counts,
        "best_primary_control_counts": best_primary_counts,
        "failure_count": len(result["failures"]),
        "interpretation": "negative_v5_contrastive_residual_calibration_development_blocks_final_eval",
    }


def audit_four_behavior_representation_steering_v6_source_pools_positive() -> Dict[str, Any]:
    base = RUNS / "four_behavior_representation_steering_v6_pools"
    combined_path = base / "combined_audit.json"
    redacted_path = base / "final_redacted_audit.json"
    combined = load_json(combined_path)
    redacted = load_json(redacted_path)
    failures = []
    require(
        failures,
        combined["claim_scope"] == "four_behavior_representation_steering_v6_source_pool_construction",
        "V6 steering source-pool claim scope drifted",
    )
    require(failures, combined["passed"] is True, "V6 steering source-pool audit did not pass")
    require(
        failures,
        redacted["claim_scope"] == "redacted_final_steering_v6_source_pool_audit_surface_only",
        "V6 steering final redacted claim scope drifted",
    )
    expected_counts = {"train": 64, "development": 24, "final": 24}
    require(
        failures,
        combined["required_counts"] == expected_counts,
        "V6 steering source-pool required counts drifted",
    )
    for pool_name, expected_count in expected_counts.items():
        if pool_name == "final":
            counts = redacted["summary"]["accepted_counts_by_behavior"]
        else:
            counts = combined["pool_summaries"][pool_name]["accepted_counts_by_behavior"]
        for pattern in ("sorted_ascending", "sorted_descending", "has_majority", "mountain_pattern"):
            require(
                failures,
                counts[pattern] == expected_count,
                f"V6 {pool_name}/{pattern} accepted count drifted",
            )
    for key, value in combined["overlap_counts"].items():
        require(failures, int(value) == 0, f"V6 source-pool overlap drifted: {key}")
    require(
        failures,
        combined["seed_preflight"]["passed"] is True,
        "V6 source-pool seed preflight did not pass",
    )
    require(
        failures,
        combined["seed_preflight"]["seed_ranges"][0]["start_seed"] == 45300000,
        "V6 source-pool train seed base drifted",
    )
    expected_pool_hashes = {
        "train": "fa88913363f7e0b25245c32bb065be12ccba2c62f5f07f7eb067b21668120f96",
        "development": "fabe93ad16ec3047e914372cc831c1bf147443aaa37614d52fa57242041bc974",
        "final": "ad47b922b49153e03aa677ae610c954336b79de61b64fb73bc9c9648ff49bde8",
    }
    require(
        failures,
        combined["pool_file_sha256"] == expected_pool_hashes,
        "V6 source-pool raw pool file hashes drifted",
    )
    require(
        failures,
        redacted["pool_file_sha256"] == expected_pool_hashes["final"],
        "V6 final redacted audit file hash drifted",
    )
    final_summary_text = json.dumps(combined["pool_summaries"]["final"], sort_keys=True)
    final_redacted_text = json.dumps(redacted, sort_keys=True)
    forbidden_final_terms = [
        "accepted_subject_ids",
        "candidate_records",
        "heldout_margin",
        "primary_source_margin",
        "records",
        "rejected_subject_ids",
        "signature",
        "signature_hash",
        "source_margin",
        "subject_id",
        "support_margin",
        "weights",
        "weights_hash",
    ]
    for term in forbidden_final_terms:
        require(
            failures,
            term not in final_summary_text,
            f"V6 combined audit exposes forbidden final detail: {term}",
        )
        require(
            failures,
            term not in final_redacted_text,
            f"V6 final redacted audit exposes forbidden final detail: {term}",
        )
    require(
        failures,
        redacted["summary"]["max_selected_train_vs_heldout_overlap_count"] == 0,
        "V6 final redacted selected-train overlap is nonzero",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "combined_artifact": rel(combined_path),
        "final_redacted_artifact": rel(redacted_path),
        "overlap_counts": combined["overlap_counts"],
        "pool_file_sha256": combined["pool_file_sha256"],
        "interpretation": "positive_v6_source_pool_construction_only",
    }


def audit_four_behavior_representation_steering_v6_development_negative() -> Dict[str, Any]:
    path = (
        RUNS
        / "four_behavior_representation_steering_v6_centroid_constrained_primary_correction"
        / "development_results.json"
    )
    result = load_json(path)
    failures = []
    aggregate = result["aggregate"]
    require(
        failures,
        (
            result["claim_scope"]
            == "four_behavior_representation_steering_v6_centroid_constrained_primary_correction_development"
        ),
        "V6 steering development claim scope drifted",
    )
    require(failures, result["passed"] is False, "V6 steering development unexpectedly passed")
    require(
        failures,
        result["next_action"] == "log_negative_development_result_do_not_open_final_raw",
        "V6 steering development next action drifted",
    )
    require(failures, result["source_pool_audit_passed"] is True, "V6 source-pool audit failed")
    require(failures, aggregate["n"] == 288, "V6 steering development n drifted")
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == 2,
        "V6 steering development pass count drifted",
    )
    require(
        failures,
        abs(aggregate["individual_all_gate_pass_rate"] - 0.006944444444444444) < 1e-12,
        "V6 steering development pass rate drifted",
    )
    require(
        failures,
        aggregate["mean_matched_primary_target_margin"] > 40.0,
        "V6 steering development no longer shows primary target movement",
    )
    require(
        failures,
        aggregate["mean_matched_centroid_improvement"] > 5.0,
        "V6 steering development no longer shows centroid movement",
    )
    require(
        failures,
        aggregate["mean_source_primary_margin_change"] < -100.0,
        "V6 steering development source suppression drifted",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_primary_target_margin"] < 0.0,
        "V6 primary best-control specificity unexpectedly positive",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_centroid_improvement"] < 0.0,
        "V6 centroid best-control specificity unexpectedly positive",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v5_calibrated_primary_target_margin"] < 0.0,
        "V6 unexpectedly beats V5 on mean primary target margin",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v5_calibrated_centroid_improvement"] > 0.05,
        "V6 no longer beats V5 on mean centroid improvement",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v4_low_rank_centroid_improvement"] > 0.05,
        "V6 no longer beats V4 on mean centroid improvement",
    )
    require(
        failures,
        aggregate["mean_matched_minus_v4_low_rank_primary_target_margin"] > 0.1,
        "V6 no longer beats V4 on mean primary target margin",
    )
    require(
        failures,
        result["transport_method"] == "train_centroid_constrained_primary_correction",
        "V6 steering development transport method drifted",
    )
    config = result["training_config"]
    require(failures, config["pca_rank"] == 48, "V6 PCA rank drifted")
    require(failures, config["correction_steps"] == 80, "V6 correction steps drifted")
    require(
        failures,
        abs(config["correction_lr"] - 0.05) < 1e-12,
        "V6 correction lr drifted",
    )
    require(
        failures,
        result["v5_baseline_training_summary"]["final_epoch"] == 500,
        "V6 V5-baseline final epoch drifted",
    )
    require(
        failures,
        result["train_pool_sha256"]
        == "fa88913363f7e0b25245c32bb065be12ccba2c62f5f07f7eb067b21668120f96",
        "V6 train pool hash drifted",
    )
    require(
        failures,
        result["eval_pool_sha256"]
        == "fabe93ad16ec3047e914372cc831c1bf147443aaa37614d52fa57242041bc974",
        "V6 development pool hash drifted",
    )
    require(
        failures,
        result["combined_audit_sha256"] == sha256_file(
            RUNS / "four_behavior_representation_steering_v6_pools" / "combined_audit.json"
        ),
        "V6 combined audit hash does not match artifact",
    )
    require(
        failures,
        result["final_redacted_audit_sha256"] == sha256_file(
            RUNS / "four_behavior_representation_steering_v6_pools" / "final_redacted_audit.json"
        ),
        "V6 final redacted audit hash does not match artifact",
    )
    require(
        failures,
        result["train_only_statistics_hash"]
        == "fe8ea63ad042ca89a51823b5f49e34e1287805116ad5162a207d17043b95b1b2",
        "V6 train-only statistics hash drifted",
    )
    require(
        failures,
        result["v5_baseline_calibration_hash"]
        == "cc0d07d08e099b27cc291c1571b0507a6099d0fdb2fa555e005adc423230e5db",
        "V6 V5-baseline calibration hash drifted",
    )
    require(
        failures,
        all(
            summary["individual_all_gate_pass_rate"] < 0.80
            for summary in result["by_target"].values()
        ),
        "V6 steering development target pass-rate failure drifted",
    )
    require(
        failures,
        all(
            summary["individual_all_gate_pass_rate"] < 0.90
            for summary in result["by_direction"].values()
        ),
        "V6 steering development direction pass-rate failure drifted",
    )
    require(
        failures,
        len(result["records"]) == 288,
        "V6 steering development record count drifted",
    )
    require(
        failures,
        all(len(record["controls"]) == 44 for record in result["records"]),
        "V6 steering development controls are not 44 per record",
    )
    require(
        failures,
        all(
            sum(
                1
                for control in record["controls"]
                if control["control_type"] == "random_norm_matched_vector"
            )
            == 32
            for record in result["records"]
        ),
        "V6 steering development random controls are not 32 per record",
    )
    best_primary_counts = count_record_summary_values(result, "best_primary_control_type")
    best_centroid_counts = count_record_summary_values(result, "best_centroid_control_type")
    require(
        failures,
        best_primary_counts == {
            "same_target_other_source_v6_centroid_constrained_primary_correction": 8,
            "v2_centroid_delta": 5,
            "v3_diagonal_transport": 2,
            "v5_contrastive_residual_calibration": 273,
        },
        "V6 steering development primary-control distribution drifted",
    )
    require(
        failures,
        best_centroid_counts == {
            "reverse_v6_centroid_constrained_primary_correction": 9,
            "same_source_other_target_v6_centroid_constrained_primary_correction": 102,
            "same_target_other_source_v6_centroid_constrained_primary_correction": 59,
            "shuffled_v6_centroid_constrained_primary_correction": 3,
            "v4_low_rank_residual_transport": 94,
            "v5_contrastive_residual_calibration": 21,
        },
        "V6 steering development centroid-control distribution drifted",
    )
    result_text = json.dumps(result, sort_keys=True)
    require(
        failures,
        "final_subjects.json" not in result_text,
        "V6 steering development result names a raw final pool",
    )
    for flag_name in (
        "forbidden_decoder_final_raw_opened",
        "forbidden_v1_steering_final_raw_opened",
        "forbidden_v2_steering_final_raw_opened",
        "forbidden_v3_steering_final_raw_opened",
        "forbidden_v4_steering_final_raw_opened",
        "forbidden_v5_steering_final_raw_opened",
        "forbidden_v6_steering_final_raw_opened",
    ):
        require(failures, result[flag_name] is False, f"V6 final-open flag set: {flag_name}")
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "by_target": result["by_target"],
        "best_centroid_control_counts": best_centroid_counts,
        "best_primary_control_counts": best_primary_counts,
        "failure_count": len(result["failures"]),
        "interpretation": "negative_v6_centroid_constrained_primary_correction_development_blocks_final_eval",
    }


def audit_four_behavior_representation_steering_v6_posthoc_pareto_diagnosis() -> Dict[str, Any]:
    path = (
        RUNS
        / "four_behavior_representation_steering_v6_centroid_constrained_primary_correction"
        / "posthoc_pareto_diagnosis.json"
    )
    result = load_json(path)
    failures = []
    require(
        failures,
        result["claim_scope"] == "v6_development_posthoc_diagnosis_not_proof",
        "V6 posthoc diagnosis claim scope drifted",
    )
    require(
        failures,
        result["development_status"]
        == "posthoc_development_only_diagnosis_after_preregistered_v6_failure",
        "V6 posthoc diagnosis development status drifted",
    )
    require(
        failures,
        result["final_access_status"] == "does_not_authorize_opening_or_evaluating_v6_final_raw",
        "V6 posthoc diagnosis final access status drifted",
    )
    require(
        failures,
        result["next_action"]
        == "use_only_to_motivate_fresh_preregistered_v7_design_do_not_open_v6_final_raw",
        "V6 posthoc diagnosis next action drifted",
    )
    require(
        failures,
        result["source_artifact_sha256"]
        == "4dcb37d961d00dcdb4535208e1caed32961c4719e49f79580719c2960f3c35a3",
        "V6 posthoc diagnosis source artifact hash drifted",
    )
    require(
        failures,
        result["pareto_undominated_count"] == 226 and result["n"] == 288,
        "V6 posthoc Pareto undominated count drifted",
    )
    require(
        failures,
        abs(result["pareto_undominated_rate"] - 0.7847222222222222) < 1e-12,
        "V6 posthoc Pareto undominated rate drifted",
    )
    require(
        failures,
        result["dominator_type_counts"] == {
            "same_target_other_source_v6_centroid_constrained_primary_correction": 33,
            "shuffled_v6_centroid_constrained_primary_correction": 5,
            "v2_centroid_delta": 12,
            "v3_diagonal_transport": 12,
            "v5_contrastive_residual_calibration": 27,
        },
        "V6 posthoc dominator type counts drifted",
    )
    worst_undominated = min(
        item["undominated"]
        for item in result["dominance_by_direction"].values()
    )
    require(
        failures,
        worst_undominated == 16,
        "V6 posthoc worst-direction undominated count drifted",
    )
    limitations_text = "\n".join(result.get("limitations", []))
    for snippet in (
        "Posthoc diagnosis on V6 development data only.",
        "Not proof-grade evidence and not final evidence.",
        "The preregistered V6 development gates failed",
        "Any V7 proof attempt must use a new preregistration and fresh pools",
        "Worst-direction Pareto-undominated rate is only 16/24",
    ):
        require(
            failures,
            snippet in limitations_text,
            f"V6 posthoc diagnosis missing limitation: {snippet}",
        )
    result_text = json.dumps(result, sort_keys=True)
    require(
        failures,
        "final_subjects.json" not in result_text,
        "V6 posthoc diagnosis names a raw final pool",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "pareto_undominated_count": result["pareto_undominated_count"],
        "pareto_undominated_rate": result["pareto_undominated_rate"],
        "worst_direction_undominated_count": worst_undominated,
        "interpretation": "posthoc_v6_development_diagnosis_not_proof_not_final_evidence",
    }


def audit_four_behavior_representation_steering_v7_source_pools_positive() -> Dict[str, Any]:
    return audit_recent_steering_source_pools_positive(
        version="V7",
        base=RUNS / "four_behavior_representation_steering_v7_pools",
        combined_scope="four_behavior_representation_steering_v7_source_pool_construction",
        redacted_scope="redacted_final_steering_v7_source_pool_audit_surface_only",
        train_seed_base=51300000,
        expected_pool_hashes={
            "train": "051409a6a91ad0d5bc8549b21ee8892224099d7ff8fdb872a2b9ec8956b3db7a",
            "development": "815df92248f7bad4be8e716e754735553c15720f3b416493f53d5a3b9f127414",
            "final": "e3e8ea37ea9fd5796227c2849f293db658da0e9ba93e4eb8154a7f21bea22c1b",
        },
    )


def audit_four_behavior_representation_steering_v7_development_negative() -> Dict[str, Any]:
    return audit_recent_steering_development_negative(
        version="V7",
        path=(
            RUNS
            / "four_behavior_representation_steering_v7_pareto_frontier_correction"
            / "development_results.json"
        ),
        claim_scope="four_behavior_representation_steering_v7_pareto_frontier_correction_development",
        transport_method="train_pareto_frontier_correction",
        expected_train_hash="051409a6a91ad0d5bc8549b21ee8892224099d7ff8fdb872a2b9ec8956b3db7a",
        expected_development_hash="815df92248f7bad4be8e716e754735553c15720f3b416493f53d5a3b9f127414",
        expected_train_stats_hash="ffffe5e58ed6263d3a0ea251f8bcd92d6dc45bf220199b419b417bb2da724f03",
        expected_calibration_hash="f774f17cb3fb084cf088dd03d5e52ee1b41388c775cc232f7727cda6790538d9",
        expected_pass_count=245,
        expected_pareto_count=257,
        expected_target_prediction_count=278,
        expected_controls_per_record=69,
        source_pool_dir=RUNS / "four_behavior_representation_steering_v7_pools",
        required_failed_substrings=[
            "aggregate individual pass rate failed",
            "aggregate Pareto-undominated record rate failed",
            "aggregate mean selected-minus-V6-correction centroid improvement failed",
            "direction mountain_pattern_to_has_majority individual pass rate failed",
        ],
    )


def audit_four_behavior_representation_steering_v8_source_pools_positive() -> Dict[str, Any]:
    return audit_recent_steering_source_pools_positive(
        version="V8",
        base=RUNS / "four_behavior_representation_steering_v8_pools",
        combined_scope="four_behavior_representation_steering_v8_source_pool_construction",
        redacted_scope="redacted_final_steering_v8_source_pool_audit_surface_only",
        train_seed_base=54300000,
        expected_pool_hashes={
            "train": "38c63e3655dcc25def775d2d89d75439947a02e186e46586ee32942bd9ac4084",
            "development": "37b5cedbed50df65b827c1597ce7cc6af3a320ccd13c77b03134b1273f0ff784",
            "final": "61810eeb45b25b6074ae20e316a56bf2d80be9d1ccdd3baa16cf1fbc0b39086c",
        },
    )


def audit_four_behavior_representation_steering_v8_development_negative() -> Dict[str, Any]:
    return audit_recent_steering_development_negative(
        version="V8",
        path=(
            RUNS
            / "four_behavior_representation_steering_v8_source_conditional_tournament_correction"
            / "development_results.json"
        ),
        claim_scope="four_behavior_representation_steering_v8_source_conditional_tournament_correction_development",
        transport_method="train_source_conditional_tournament_correction",
        expected_train_hash="38c63e3655dcc25def775d2d89d75439947a02e186e46586ee32942bd9ac4084",
        expected_development_hash="37b5cedbed50df65b827c1597ce7cc6af3a320ccd13c77b03134b1273f0ff784",
        expected_train_stats_hash="1199f2c176b386db8f9f242b4c89c7fb22ea71c2d8470525378e851bb08bc46e",
        expected_calibration_hash="323dd0ade6f1ea302631bf9aafd8691bcdf69c0db4e3337dee3a05ee73de39bf",
        expected_pass_count=237,
        expected_pareto_count=243,
        expected_target_prediction_count=281,
        expected_controls_per_record=74,
        source_pool_dir=RUNS / "four_behavior_representation_steering_v8_pools",
        required_failed_substrings=[
            "aggregate individual pass rate failed",
            "aggregate Pareto-undominated record rate failed",
            "direction mountain_pattern_to_has_majority individual pass rate failed",
            "direction sorted_descending_to_mountain_pattern Pareto-undominated record rate failed",
        ],
    )


def audit_four_behavior_representation_steering_v9_source_pools_positive() -> Dict[str, Any]:
    return audit_recent_steering_source_pools_positive(
        version="V9",
        base=RUNS / "four_behavior_representation_steering_v9_pools",
        combined_scope="four_behavior_representation_steering_v9_source_pool_construction",
        redacted_scope="redacted_final_steering_v9_source_pool_audit_surface_only",
        train_seed_base=57300000,
        expected_pool_hashes={
            "train": "1522c9efa4e7e1fb3d7cb82e523141fbe9c7e84f4a28ae4caf8bd2b721d57b2c",
            "development": "2a574f2aaf3c5499d3084aefac3b4dddcf6aef52eeb807709535284b26633f15",
            "final": "130b50e7ecb95e27e5e60109cc02c86f5b340df67fd87698b8cc450a58c4d1d0",
        },
    )


def audit_four_behavior_representation_steering_v9_development_positive() -> Dict[str, Any]:
    return audit_v9_target_attractor_result(
        phase="development",
        path=(
            RUNS
            / "four_behavior_representation_steering_v9_source_invariant_target_attractor"
            / "development_results.json"
        ),
        claim_scope=(
            "four_behavior_representation_steering_v9_source_invariant_target_attractor_development"
        ),
        expected_eval_hash="2a574f2aaf3c5499d3084aefac3b4dddcf6aef52eeb807709535284b26633f15",
        expected_pass_count=277,
        expected_pareto_count=279,
        expected_target_prediction_count=286,
        expected_transfer_count=2735,
    )


def audit_four_behavior_representation_steering_v9_final_positive() -> Dict[str, Any]:
    return audit_v9_target_attractor_result(
        phase="final",
        path=(
            RUNS
            / "four_behavior_representation_steering_v9_source_invariant_target_attractor"
            / "final_results.json"
        ),
        claim_scope="four_behavior_representation_steering_v9_source_invariant_target_attractor_final",
        expected_eval_hash="130b50e7ecb95e27e5e60109cc02c86f5b340df67fd87698b8cc450a58c4d1d0",
        expected_pass_count=278,
        expected_pareto_count=285,
        expected_target_prediction_count=281,
        expected_transfer_count=2664,
    )


def audit_four_behavior_functional_weight_editing_v10_source_pools_positive() -> Dict[str, Any]:
    return audit_recent_steering_source_pools_positive(
        version="V10",
        base=RUNS / "four_behavior_functional_weight_editing_v10_pools",
        combined_scope="four_behavior_functional_weight_editing_v10_source_pool_construction",
        redacted_scope="redacted_final_functional_weight_editing_v10_source_pool_audit_surface_only",
        train_seed_base=60300000,
        expected_pool_hashes={
            "train": "bd47ad73128bc686c46dc7b68753c4f86d0b9e3cbcc478ec1e5282bc38f0016c",
            "development": "08bb89db889fcc7d8994f49d71e0a649c22e07e6423ffec972c686d10fa18d36",
            "final": "2288f3403f8c0951e24d347aa195dc944478bdbb1f22b692cb99bea6467f5854",
        },
    )


def audit_four_behavior_functional_weight_editing_v10_development_negative() -> Dict[str, Any]:
    path = (
        RUNS
        / "four_behavior_functional_weight_editing_v10_v9_conditioned_delta"
        / "development_results.json"
    )
    result = load_json(path)
    aggregate = result["aggregate"]
    pool_dir = RUNS / "four_behavior_functional_weight_editing_v10_pools"
    failures = []
    require(
        failures,
        result["claim_scope"]
        == "four_behavior_functional_weight_editing_v10_v9_conditioned_delta_development",
        "V10 development scope drifted",
    )
    require(failures, result["phase"] == "development", "V10 phase drifted")
    require(failures, result["passed"] is False, "V10 development unexpectedly passed")
    require(
        failures,
        result["next_action"] == "log_negative_development_result_do_not_open_final_raw",
        "V10 next action drifted",
    )
    require(
        failures,
        result["editor_method"] == "v9_conditioned_train_only_ridge_weight_delta",
        "V10 editor method drifted",
    )
    require(failures, result["source_pool_audit_passed"] is True, "V10 source-pool audit failed")
    require(failures, aggregate["n"] == 288, "V10 development n drifted")
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == 4,
        "V10 pass count drifted",
    )
    require(
        failures,
        aggregate["target_prediction_count"] == 73,
        "V10 target-prediction count drifted",
    )
    require(
        failures,
        aggregate["pareto_undominated_count"] == 32,
        "V10 Pareto-undominated count drifted",
    )
    require(
        failures,
        aggregate["individual_all_gate_pass_rate"] < 0.85,
        "V10 pass-rate failure drifted",
    )
    require(
        failures,
        aggregate["target_prediction_rate"] < 0.90,
        "V10 target-prediction failure drifted",
    )
    require(
        failures,
        aggregate["pareto_undominated_rate"] < 0.85,
        "V10 Pareto failure drifted",
    )
    require(
        failures,
        aggregate["mean_matched_target_margin"] < 0.20,
        "V10 target-margin failure drifted",
    )
    require(
        failures,
        aggregate["mean_matched_minus_best_control_target_margin"] < 0.0,
        "V10 best-control failure drifted",
    )
    require(
        failures,
        aggregate["mean_nearest_train_minus_matched_source_output_mse"] < 0.0,
        "V10 nearest-train source-output failure drifted",
    )
    require(
        failures,
        result["train_pool_sha256"]
        == "bd47ad73128bc686c46dc7b68753c4f86d0b9e3cbcc478ec1e5282bc38f0016c",
        "V10 train pool hash drifted",
    )
    require(
        failures,
        result["eval_pool_sha256"]
        == "08bb89db889fcc7d8994f49d71e0a649c22e07e6423ffec972c686d10fa18d36",
        "V10 development pool hash drifted",
    )
    require(
        failures,
        result["combined_audit_sha256"] == sha256_file(pool_dir / "combined_audit.json"),
        "V10 combined audit hash does not match artifact",
    )
    require(
        failures,
        result["final_redacted_audit_sha256"] == sha256_file(pool_dir / "final_redacted_audit.json"),
        "V10 final redacted audit hash does not match artifact",
    )
    require(
        failures,
        result["final_redacted_summary"]["claim_scope"]
        == "redacted_final_functional_weight_editing_v10_source_pool_audit_surface_only",
        "V10 final-redacted summary scope drifted",
    )
    require(
        failures,
        sorted(result["final_redacted_summary"]["summary"].keys())
        == ["accepted_counts_by_behavior", "max_selected_train_vs_heldout_overlap_count"],
        "V10 final-redacted summary exposes unexpected keys",
    )
    require(failures, len(result["records"]) == 288, "V10 record count drifted")
    require(
        failures,
        all(record["random_control_count"] == 32 for record in result["records"]),
        "V10 random controls are not 32 per record",
    )
    require(
        failures,
        all(
            sum(
                1
                for control in record["controls"]
                if control["control_type"] == "nearest_train_target_retrieval"
            )
            == 1
            for record in result["records"]
        ),
        "V10 nearest-train retrieval control missing",
    )
    require(
        failures,
        all(len(record["controls"]) == 41 for record in result["records"]),
        "V10 controls are not 41 per record",
    )
    for snippet in [
        "aggregate individual pass rate",
        "aggregate target prediction rate",
        "aggregate Pareto-undominated rate",
        "mean matched target margin",
        "mean_matched_minus_best_control_target_margin",
    ]:
        require(
            failures,
            any(snippet in failure for failure in result["failures"]),
            f"V10 missing expected failed gate: {snippet}",
        )
    result_text = json.dumps(result, sort_keys=True)
    require(
        failures,
        "final_subjects.json" not in result_text,
        "V10 development result names a raw final pool",
    )
    require(
        failures,
        result["forbidden_prior_final_raw_opened"] is False,
        "V10 prior final-open flag set",
    )
    require(
        failures,
        result["forbidden_v10_final_raw_opened_before_authorization"] is False,
        "V10 final-open-before-authorization flag set",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "failure_count": len(result["failures"]),
        "interpretation": "negative_v10_functional_weight_editing_development_blocks_final_eval",
    }


def audit_v9_target_attractor_result(
    *,
    phase: str,
    path: Path,
    claim_scope: str,
    expected_eval_hash: str,
    expected_pass_count: int,
    expected_pareto_count: int,
    expected_target_prediction_count: int,
    expected_transfer_count: int,
) -> Dict[str, Any]:
    result = load_json(path)
    aggregate = result["aggregate"]
    source_pool_dir = RUNS / "four_behavior_representation_steering_v9_pools"
    failures = []
    require(failures, result["claim_scope"] == claim_scope, f"V9 {phase} scope drifted")
    require(failures, result["phase"] == phase, f"V9 {phase} phase drifted")
    require(failures, result["passed"] is True, f"V9 {phase} did not pass")
    require(failures, not result["failures"], f"V9 {phase} has failures")
    require(
        failures,
        result["transport_method"] == "train_source_invariant_target_attractor",
        f"V9 {phase} transport method drifted",
    )
    require(failures, result["source_pool_audit_passed"] is True, f"V9 {phase} source-pool audit failed")
    require(failures, aggregate["n"] == 288, f"V9 {phase} n drifted")
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == expected_pass_count,
        f"V9 {phase} pass count drifted",
    )
    require(
        failures,
        aggregate["pareto_undominated_count"] == expected_pareto_count,
        f"V9 {phase} Pareto-undominated count drifted",
    )
    require(
        failures,
        aggregate["target_prediction_pass_count"] == expected_target_prediction_count,
        f"V9 {phase} target-prediction count drifted",
    )
    require(
        failures,
        aggregate["same_target_transfer_probe_count"] == 2880,
        f"V9 {phase} transfer-probe count is not 2880",
    )
    require(
        failures,
        aggregate["same_target_transfer_gate_pass_count"] == expected_transfer_count,
        f"V9 {phase} transfer gate-pass count drifted",
    )
    require(
        failures,
        aggregate["same_target_transfer_target_prediction_count"] == expected_transfer_count,
        f"V9 {phase} transfer prediction count drifted",
    )
    require(
        failures,
        aggregate["individual_all_gate_pass_rate"] >= 0.90,
        f"V9 {phase} pass-rate gate failed",
    )
    require(
        failures,
        aggregate["pareto_undominated_rate"] >= 0.90,
        f"V9 {phase} Pareto-rate gate failed",
    )
    require(
        failures,
        aggregate["same_target_transfer_gate_pass_rate"] >= 0.80,
        f"V9 {phase} transfer gate-pass rate failed",
    )
    require(
        failures,
        aggregate["same_target_transfer_target_prediction_rate"] >= 0.90,
        f"V9 {phase} transfer prediction rate failed",
    )
    require(
        failures,
        aggregate["mean_selected_minus_best_control_primary_target_margin"] > 0.0,
        f"V9 {phase} primary best-control margin is not positive",
    )
    require(
        failures,
        aggregate["mean_selected_minus_best_control_centroid_improvement"] < 0.0,
        f"V9 {phase} centroid caveat drifted unexpectedly",
    )
    require(
        failures,
        aggregate["mean_selected_minus_v6_correction_centroid_improvement"] > 0.05,
        f"V9 {phase} V6 centroid margin failed",
    )
    require(
        failures,
        aggregate["mean_source_primary_margin_change"] < -100.0,
        f"V9 {phase} source suppression drifted",
    )
    require(
        failures,
        result["train_pool_sha256"]
        == "1522c9efa4e7e1fb3d7cb82e523141fbe9c7e84f4a28ae4caf8bd2b721d57b2c",
        f"V9 {phase} train pool hash drifted",
    )
    require(failures, result["eval_pool_sha256"] == expected_eval_hash, f"V9 {phase} eval pool hash drifted")
    require(
        failures,
        result["combined_audit_sha256"] == sha256_file(source_pool_dir / "combined_audit.json"),
        f"V9 {phase} combined audit hash does not match artifact",
    )
    require(
        failures,
        result["final_redacted_audit_sha256"] == sha256_file(source_pool_dir / "final_redacted_audit.json"),
        f"V9 {phase} final redacted audit hash does not match artifact",
    )
    require(
        failures,
        result["train_only_statistics_hash"]
        == "f1d815e22690bfb81af28feb9374f4d472ad15a477eb20e104d9039afd0b5099",
        f"V9 {phase} train-only statistics hash drifted",
    )
    require(
        failures,
        result["v5_baseline_calibration_hash"]
        == "6f8acd3b03903731def23f452e1e58c2009a99bcc853254ed23010ce09c0afa0",
        f"V9 {phase} V5-baseline calibration hash drifted",
    )
    require(failures, len(result["records"]) == 288, f"V9 {phase} record count drifted")
    require(
        failures,
        all(len(record["matched_candidates"]) == 5 for record in result["records"]),
        f"V9 {phase} matched candidate count is not 5 per record",
    )
    require(
        failures,
        all(len(record["controls"]) == 59 for record in result["records"]),
        f"V9 {phase} negative controls are not 59 per record",
    )
    require(
        failures,
        all(len(record["transfer_probes"]) == 10 for record in result["records"]),
        f"V9 {phase} transfer probes are not 10 per record",
    )
    require(
        failures,
        all(
            sum(
                1
                for control in record["controls"]
                if control["control_type"] == "random_norm_matched_vector"
            )
            == 32
            for record in result["records"]
        ),
        f"V9 {phase} random controls are not 32 per record",
    )
    require(
        failures,
        all(
            all(
                probe["control_type"]
                == "same_target_other_source_v9_source_invariant_target_attractor"
                for probe in record["transfer_probes"]
            )
            for record in result["records"]
        ),
        f"V9 {phase} transfer probes include a non-transfer control type",
    )
    if phase == "development":
        require(
            failures,
            result["next_action"] == "eligible_for_one_shot_final_eval_without_method_changes",
            "V9 development next action drifted",
        )
        require(
            failures,
            "final_subjects.json" not in json.dumps(result, sort_keys=True),
            "V9 development result names a raw final pool",
        )
    else:
        require(
            failures,
            result["eval_pool_path"] == "runs/four_behavior_representation_steering_v9_pools/final_subjects.json",
            "V9 final eval pool path drifted",
        )
    for flag_name, value in result.items():
        if flag_name.startswith("forbidden_") and flag_name.endswith("_final_raw_opened"):
            require(failures, value is False, f"V9 {phase} final-open flag set: {flag_name}")
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "interpretation": (
            "positive_v9_source_invariant_target_attractor_final"
            if phase == "final"
            else "positive_v9_development_authorized_one_shot_final"
        ),
        "limitations": [
            "not scalar centroid dominance over all controls",
            "not functional decoding or behavioral model editing evidence",
            "aggregate transfer gate only; weak per-direction transfer remains",
        ],
    }


def audit_recent_steering_source_pools_positive(
    *,
    version: str,
    base: Path,
    combined_scope: str,
    redacted_scope: str,
    train_seed_base: int,
    expected_pool_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    combined_path = base / "combined_audit.json"
    redacted_path = base / "final_redacted_audit.json"
    combined = load_json(combined_path)
    redacted = load_json(redacted_path)
    failures = []
    require(failures, combined["claim_scope"] == combined_scope, f"{version} source-pool scope drifted")
    require(failures, combined["passed"] is True, f"{version} source-pool audit did not pass")
    require(failures, redacted["claim_scope"] == redacted_scope, f"{version} final-redacted scope drifted")
    expected_counts = {"train": 64, "development": 24, "final": 24}
    require(failures, combined["required_counts"] == expected_counts, f"{version} required counts drifted")
    for pool_name, expected_count in expected_counts.items():
        if pool_name == "final":
            counts = redacted["summary"]["accepted_counts_by_behavior"]
        else:
            counts = combined["pool_summaries"][pool_name]["accepted_counts_by_behavior"]
        for pattern in ("sorted_ascending", "sorted_descending", "has_majority", "mountain_pattern"):
            require(
                failures,
                counts[pattern] == expected_count,
                f"{version} {pool_name}/{pattern} accepted count drifted",
            )
    for key, value in combined["overlap_counts"].items():
        require(failures, int(value) == 0, f"{version} source-pool overlap drifted: {key}")
    require(failures, combined["seed_preflight"]["passed"] is True, f"{version} seed preflight failed")
    require(
        failures,
        combined["seed_preflight"]["seed_ranges"][0]["start_seed"] == train_seed_base,
        f"{version} train seed base drifted",
    )
    require(
        failures,
        combined["pool_file_sha256"] == expected_pool_hashes,
        f"{version} raw pool file hashes drifted",
    )
    require(
        failures,
        redacted["pool_file_sha256"] == expected_pool_hashes["final"],
        f"{version} final redacted file hash drifted",
    )
    final_summary_text = json.dumps(combined["pool_summaries"]["final"], sort_keys=True)
    final_redacted_text = json.dumps(redacted, sort_keys=True)
    forbidden_final_terms = [
        "accepted_subject_ids",
        "attempt_count",
        "attempt_counts_by_behavior",
        "heldout_margin",
        "records",
        "rejected_subject_ids",
        "signature",
        "signature_hash",
        "subject_id",
        "weights",
        "weights_hash",
    ]
    for term in forbidden_final_terms:
        require(
            failures,
            term not in final_summary_text,
            f"{version} combined audit exposes forbidden final detail: {term}",
        )
        require(
            failures,
            term not in final_redacted_text,
            f"{version} final redacted audit exposes forbidden final detail: {term}",
        )
    require(
        failures,
        redacted["summary"]["max_selected_train_vs_heldout_overlap_count"] == 0,
        f"{version} final redacted selected-train overlap is nonzero",
    )
    return {
        "passed": not failures,
        "failures": failures,
        "combined_artifact": rel(combined_path),
        "final_redacted_artifact": rel(redacted_path),
        "overlap_counts": combined["overlap_counts"],
        "pool_file_sha256": combined["pool_file_sha256"],
        "interpretation": f"positive_{version.lower()}_source_pool_construction_only",
    }


def audit_recent_steering_development_negative(
    *,
    version: str,
    path: Path,
    claim_scope: str,
    transport_method: str,
    expected_train_hash: str,
    expected_development_hash: str,
    expected_train_stats_hash: str,
    expected_calibration_hash: str,
    expected_pass_count: int,
    expected_pareto_count: int,
    expected_target_prediction_count: int,
    expected_controls_per_record: int,
    source_pool_dir: Path,
    required_failed_substrings: Sequence[str],
) -> Dict[str, Any]:
    result = load_json(path)
    aggregate = result["aggregate"]
    failures = []
    require(failures, result["claim_scope"] == claim_scope, f"{version} development scope drifted")
    require(failures, result["passed"] is False, f"{version} development unexpectedly passed")
    require(
        failures,
        result["next_action"] == "log_negative_development_result_do_not_open_final_raw",
        f"{version} development next action drifted",
    )
    require(failures, result["source_pool_audit_passed"] is True, f"{version} source-pool audit failed")
    require(failures, result["transport_method"] == transport_method, f"{version} transport method drifted")
    require(failures, aggregate["n"] == 288, f"{version} development n drifted")
    require(
        failures,
        aggregate["individual_all_gate_pass_count"] == expected_pass_count,
        f"{version} pass count drifted",
    )
    require(
        failures,
        aggregate["pareto_undominated_count"] == expected_pareto_count,
        f"{version} Pareto-undominated count drifted",
    )
    require(
        failures,
        aggregate["target_prediction_pass_count"] == expected_target_prediction_count,
        f"{version} target-prediction count drifted",
    )
    require(failures, aggregate["individual_all_gate_pass_rate"] < 0.90, f"{version} pass-rate failure drifted")
    require(failures, aggregate["pareto_undominated_rate"] < 0.90, f"{version} Pareto-rate failure drifted")
    require(failures, aggregate["mean_selected_primary_target_margin"] > 100.0, f"{version} primary movement drifted")
    require(failures, aggregate["mean_selected_centroid_improvement"] > 5.0, f"{version} centroid movement drifted")
    require(failures, aggregate["mean_source_primary_margin_change"] < -100.0, f"{version} source suppression drifted")
    require(failures, result["train_pool_sha256"] == expected_train_hash, f"{version} train pool hash drifted")
    require(failures, result["eval_pool_sha256"] == expected_development_hash, f"{version} development pool hash drifted")
    require(
        failures,
        result["combined_audit_sha256"] == sha256_file(source_pool_dir / "combined_audit.json"),
        f"{version} combined audit hash does not match artifact",
    )
    require(
        failures,
        result["final_redacted_audit_sha256"] == sha256_file(source_pool_dir / "final_redacted_audit.json"),
        f"{version} final redacted audit hash does not match artifact",
    )
    require(
        failures,
        result["train_only_statistics_hash"] == expected_train_stats_hash,
        f"{version} train-only statistics hash drifted",
    )
    require(
        failures,
        result["v5_baseline_calibration_hash"] == expected_calibration_hash,
        f"{version} V5-baseline calibration hash drifted",
    )
    require(failures, len(result["records"]) == 288, f"{version} record count drifted")
    require(
        failures,
        all(len(record["matched_candidates"]) == 5 for record in result["records"]),
        f"{version} matched candidate count is not 5 per record",
    )
    require(
        failures,
        all(len(record["controls"]) == expected_controls_per_record for record in result["records"]),
        f"{version} controls are not {expected_controls_per_record} per record",
    )
    require(
        failures,
        all(
            sum(
                1
                for control in record["controls"]
                if control["control_type"] == "random_norm_matched_vector"
            )
            == 32
            for record in result["records"]
        ),
        f"{version} random controls are not 32 per record",
    )
    for snippet in required_failed_substrings:
        require(
            failures,
            any(snippet in failure for failure in result["failures"]),
            f"{version} missing expected failed gate: {snippet}",
        )
    result_text = json.dumps(result, sort_keys=True)
    require(
        failures,
        "final_subjects.json" not in result_text,
        f"{version} development result names a raw final pool",
    )
    for flag_name, value in result.items():
        if flag_name.startswith("forbidden_") and flag_name.endswith("_final_raw_opened"):
            require(failures, value is False, f"{version} final-open flag set: {flag_name}")
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(path),
        "aggregate": aggregate,
        "failure_count": len(result["failures"]),
        "interpretation": f"negative_{version.lower()}_development_blocks_final_eval",
    }


def audit_subject_pool_separation() -> Dict[str, Any]:
    pools = {
        "accepted_external_v1": RUNS / "fresh_external_steering_holdout_v1" / "subjects.json",
        "failed_robust_v1": RUNS / "fresh_external_steering_holdout_v2_robust" / "subjects.json",
        "v2_train_dev": RUNS / "fresh_robust_edit_v2_train_pool" / "subjects.json",
        "v2_final": RUNS / "fresh_external_steering_holdout_v3_robust_final" / "subjects.json",
    }
    failures = []
    subject_ids = {name: load_subject_ids(path) for name, path in pools.items()}
    overlaps = {}
    names = list(subject_ids)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = sorted(subject_ids[left] & subject_ids[right])
            overlaps[f"{left}__{right}"] = overlap
            require(failures, not overlap, f"{left} overlaps {right}")
    hashes = {name: sha256_file(path) for name, path in pools.items()}
    expected_hashes = {
        "accepted_external_v1": "6116ad4af8e10fbd515e41e09ed7ed738c28ff2611917d7561fad0cf74825754",
        "failed_robust_v1": "a0f1727294b7bb188a461b0222592d890245408394e7a87cb806000d9ad53e9f",
        "v2_train_dev": "ce49fc086eaab211c48e45c59f92d22c7288d6cb7da980c8cba8e65b0004e8dd",
        "v2_final": "8c9f2cc2ddf1f407c52155f6b483dbed96c00c02a9ad846b1f64ac9f5c1e1124",
    }
    for name, expected in expected_hashes.items():
        require(failures, hashes[name] == expected, f"{name} SHA mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "subject_counts": {name: len(ids) for name, ids in subject_ids.items()},
        "sha256": hashes,
        "overlap_counts": {key: len(value) for key, value in overlaps.items()},
    }


def audit_research_log() -> Dict[str, Any]:
    text = LOG_PATH.read_text()
    required_snippets = [
        "2026-06-10 - Locked V2 Final Holdout Decode Evaluation",
        "2026-06-10 - Strict External Steering Robustness Evaluation",
        "2026-06-10 - Robust Signature Edit Vectors V2 Final Holdout",
        "2026-06-10 - Additional-Behavior Decode Feasibility",
        "2026-06-10 - Four-Behavior Source-Generation Feasibility",
        "2026-06-10 - Four-Behavior Source-Generation V2 Expanded Support",
        "2026-06-10 - Four-Behavior Source-Generation V3 Full Pool",
        "2026-06-10 - Four-Behavior Source-Generation V4 Accept-Reject",
        "2026-06-10 - Four-Behavior Decoder Source Pools V1 Failed Attempt",
        "2026-06-10 - Four-Behavior Decoder Source Pools V2",
        "2026-06-10 - Four-Behavior Decoder Development V1",
        "2026-06-11 - Four-Behavior Decoder Development V2",
        "2026-06-11 - Four-Behavior Decoder Development V3 Signature Inversion",
        "2026-06-11 - Four-Behavior Representation Steering V1",
        "2026-06-11 - Four-Behavior Representation Steering V2 Centroid Delta",
        "2026-06-11 - Four-Behavior Representation Steering V3 Diagonal Transport",
        "2026-06-11 - Four-Behavior Representation Steering V4 Low-Rank Residual Transport",
        "2026-06-11 - Four-Behavior Representation Steering V5 Contrastive Residual Calibration",
        "2026-06-11 - Four-Behavior Representation Steering V6 Centroid-Constrained Primary Correction",
        "Reviewer confidence: `5/5`",
        "Final reviewer confidence after correction: `5/5`",
        "It does not prove larger models, additional behaviors, or broad MUAT generality.",
        "Final reviewer confidence: `5/5`.",
        "negative source-generation feasibility result for this support-only protocol",
        "expanded support helps but does not clear the preregistered gate",
        "full-pool training improves `has_majority` but does not clear the",
        "No rejections occurred under this preregistered seed schedule.",
        "Accepted seed disjointness failed",
        "positive source-pool construction checkpoint only",
        "negative train/development decoder result for this direct MLP",
        "It blocks final evaluation under the preregistration.",
        "negative adaptive train/development decoder result",
        "fails the control-specificity and subject-specificity gates",
        "Every control explicitly reports matched-minus-control target margin",
        "negative adaptive V3 signature-inversion decoder checkpoint only",
        "negative four-behavior representation-steering V1 development checkpoint",
        "superseded by the corrected no-edit-relative rerun",
        "It blocks final steering evaluation under the preregistration.",
        "exact centroid-delta vectors",
        "negative four-behavior representation-steering V2 development checkpoint",
        "It blocks final steering evaluation under the V2 preregistration.",
        "source-specificity against\n    same-target other-source centroid controls",
        "closed-form train-only diagonal covariance transport",
        "negative V3 diagonal-transport development checkpoint",
        "It blocks final steering evaluation under the V3 preregistration.",
        "source-specificity against the full best-control set",
        "closed-form train-only low-rank residual covariance transport",
        "negative V4 low-rank residual-transport development checkpoint",
        "It blocks final steering evaluation under the V4 preregistration.",
        "improves over V2 and V3 on centroid metrics but not on primary\n    target-margin specificity",
        "train-only V4 low-rank transport recomputed from V5 train subjects",
        "negative V5 contrastive residual-calibration development checkpoint",
        "It blocks final steering evaluation under the V5 preregistration.",
        "fails proof-grade reliability and centroid best-control specificity",
        "per-example train-only centroid-constrained primary\n  correction",
        "negative V6 centroid-constrained primary-correction development",
        "It blocks final steering evaluation under the V6 preregistration.",
        "V5 remains much stronger on primary target-margin controls",
        "2026-06-11 - V6 Posthoc Pareto Diagnosis",
        "v6_development_posthoc_diagnosis_not_proof",
        "use_only_to_motivate_fresh_preregistered_v7_design_do_not_open_v6_final_raw",
        "development-only posthoc diagnosis that motivates future design",
        "2026-06-11 - V7 Pareto-Frontier Correction",
        "2026-06-11 - V8 Source-Conditional Tournament Correction",
        "2026-06-11 - V9 Source-Invariant Target-Attractor",
        "test_v9_shuffled_same_target_direction_remains_negative_control",
        "Same-target transfer probe count: `2880`.",
        "Same-target transfer target-prediction count/rate: `2664/2880`, `0.925`.",
        "Mean selected-minus-best-control centroid improvement: `-0.9368553758`.",
        "has_majority_to_mountain_pattern` had transfer rate\n    `0.6625`",
        "positive V9 source-invariant target-attractor final result",
        "2026-06-12 - V10 Functional Weight Editing V9-Conditioned Delta",
        "source-label-known, target-label-requested functional weight editing",
        "Initial reviewer confidence: `3/5`.",
        "Revised preregistration reviewer confidence: `5/5`.",
        "Individual all-gate pass count/rate: `4/288`, `0.0138888889`.",
        "Mean matched-minus-best-control target margin: `-0.8004080978`.",
        "clean negative V10 development checkpoint only",
    ]
    failures = []
    for snippet in required_snippets:
        require(failures, snippet in text, f"missing log snippet: {snippet}")
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(LOG_PATH),
    }


def audit_evidence_report() -> Dict[str, Any]:
    text = REPORT_PATH.read_text()
    required_snippets = [
        "Four-Behavior Decoder Development Failures",
        "V1 direct MLP decoder",
        "V2 functional-distillation decoder",
        "V3 signature-inversion decoder",
        "0/96",
        "V3 inferred behavior accuracy: `0.78125`",
        "negative adaptive train/development signature-inversion checkpoint",
        "No four-behavior decoder should be run on the sealed final raw pool",
        "Four-Behavior Representation Steering V1 Development Failure",
        "A first development artifact reported `55/288` individual passes",
        "individual pass count: `16/288`",
        "mean matched-minus-best-control centroid improvement: `-1.0535927349`",
        "blocks final steering evaluation",
        "Four-Behavior Representation Steering V2 Centroid-Delta Development Failure",
        "individual pass count: `142/288`",
        "individual pass rate: `0.4930555556`",
        "mean matched-minus-best-control centroid improvement: `0.4016633564`",
        "does not justify\nopening the V2 final raw pool",
        "source-specificity against same-target other-source centroid controls",
        "Four-Behavior Representation Steering V3 Diagonal-Transport Development Failure",
        "individual pass count: `30/288`",
        "individual pass rate: `0.1041666667`",
        "mean matched-minus-V2-centroid-delta primary target margin: `1.9002874485`",
        "mean matched-minus-best-control centroid improvement: `-0.9124838478`",
        "does not justify opening the V3 final raw pool",
        "Four-Behavior Representation Steering V4 Low-Rank Residual-Transport Development Failure",
        "individual pass count: `42/288`",
        "individual pass rate: `0.1458333333`",
        "mean matched-minus-V3-diagonal-transport centroid improvement: `3.0668179062`",
        "mean matched-minus-best-control primary target margin: `-14.2722247789`",
        "does not justify opening the V4 final\nraw pool",
        "Four-Behavior Representation Steering V5 Contrastive Residual-Calibration Development Failure",
        "individual pass count: `20/288`",
        "individual pass rate: `0.0694444444`",
        "mean matched-minus-V4-low-rank primary target margin: `79.1476116118`",
        "mean matched-minus-best-control centroid improvement: `-2.5652369327`",
        "does not justify opening the V5 final raw pool",
        "Four-Behavior Representation Steering V6 Centroid-Constrained Primary-Correction Development Failure",
        "individual pass count: `2/288`",
        "individual pass rate: `0.0069444444`",
        "mean matched-minus-V5-calibrated primary target margin: `-62.6472493344`",
        "mean matched-minus-best-control centroid improvement: `-1.9438294139`",
        "does not justify opening the V6 final\nraw pool",
        "Posthoc development-only diagnosis",
        "Pareto-undominated records: `226/288`",
        "claim scope: `v6_development_posthoc_diagnosis_not_proof`",
        "does not change the V6 development failure",
        "Four-Behavior Representation Steering V7/V8 Fresh-Pool Development Failures",
        "individual pass count/rate: `245/288`, `0.8506944444`",
        "Pareto-undominated count/rate: `257/288`, `0.8923611111`",
        "individual pass count/rate: `237/288`, `0.8229166667`",
        "Pareto-undominated count/rate: `243/288`, `0.84375`",
        "These results are useful negative evidence",
        "do not authorize opening the V7 or\nV8 final raw pools",
        "Four-Behavior Representation-Space V9 Source-Invariant Target-Attractor",
        "target-attracting edits rather than\nreliable bad controls",
        "invalid run had `3125` transfer probes instead of the\npreregistered `2880`",
        "individual pass count/rate: `278/288`, `0.9652777778`",
        "Pareto-undominated count/rate: `285/288`, `0.9895833333`",
        "same-target transfer target-prediction count/rate: `2664/2880`, `0.925`",
        "mean selected-minus-best-control centroid improvement: `-0.9368553758`",
        "does not show scalar\ncentroid dominance over all controls",
        "has_majority_to_mountain_pattern` had final same-target transfer\nrate `0.6625`",
        "Four-Behavior Functional Weight Editing V10 Development Failure",
        "source-label-known and target-label-requested",
        "final public surfaces exposed no per-subject final details",
        "individual pass count/rate: `4/288`, `0.0138888889`",
        "target-prediction count/rate: `73/288`, `0.2534722222`",
        "mean matched-minus-best-control target margin: `-0.8004080978`",
        "This does not support a four-behavior\nfunctional weight-editing claim",
        "The V10 final raw pool remains sealed",
    ]
    failures = []
    for snippet in required_snippets:
        require(failures, snippet in text, f"missing report snippet: {snippet}")
    return {
        "passed": not failures,
        "failures": failures,
        "artifact": rel(REPORT_PATH),
    }


def decode_threshold_to_metric(threshold_name: str) -> str:
    mapping = {
        "min_mean_matched_minus_control_behavior_margin": (
            "mean_matched_minus_control_behavior_margin"
        ),
        "min_mean_control_minus_matched_subject_output_mse": (
            "mean_control_minus_matched_subject_output_mse"
        ),
    }
    return mapping[threshold_name]


def load_subject_ids(path: Path) -> set[str]:
    payload = load_json(path)
    return {subject["subject_id"] for subject in payload["subjects"]}


def count_record_summary_values(result: Mapping[str, Any], summary_key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in result["records"]:
        value = record["summary"][summary_key]
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def require(failures: List[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def compare(value: float, threshold: float, operator: str) -> bool:
    return value >= threshold if operator == ">=" else value <= threshold


def load_vector_norms(path: Path) -> Dict[str, float]:
    import torch

    payload = torch.load(path, map_location="cpu")
    return {
        key: float(value.norm().item())
        for key, value in payload["edit_vectors"].items()
    }


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
