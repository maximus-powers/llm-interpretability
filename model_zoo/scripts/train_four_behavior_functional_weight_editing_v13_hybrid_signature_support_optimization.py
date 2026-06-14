"""V13 hybrid functional editing via signature-support optimization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ZOO_ROOT = REPO_ROOT / "model_zoo"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_ZOO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import generate_four_behavior_decoder_source_pools as poolgen  # noqa: E402
import train_four_behavior_functional_weight_editing_v10_v9_conditioned_delta as v10  # noqa: E402
import train_four_behavior_representation_steering as v1  # noqa: E402
import train_four_behavior_representation_steering_v9_source_invariant_target_attractor as v9  # noqa: E402
from evaluate_four_behavior_source_generation_feasibility import (  # noqa: E402
    PATTERNS,
    build_candidate_pools,
    build_heldout_sequences,
    build_suite,
    summarize_candidate_pools,
)
from hypernet.dataset_provenance import stable_hash_json  # noqa: E402
from hypernet.behavior_suite import PREDICATES  # noqa: E402
from hypernet.paired_contrast import build_digit_probe_examples  # noqa: E402


SEED_BEHAVIOR_STRIDE = 100000
POOL_CONFIGS = {
    "train": {
        "base_seed": 69300000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 70300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 71300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}
DEFAULT_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v13_pools"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v13_hybrid_signature_support_optimization"
)
PREREG_PATH = (
    REPO_ROOT
    / "docs"
    / "preregistrations"
    / "four_behavior_functional_weight_editing_v13_hybrid_signature_support_optimization.md"
)
SCRIPT_PATH = Path(__file__).resolve()
SOURCE_POOL_SCOPE = "four_behavior_functional_weight_editing_v13_source_pool"
SOURCE_AUDIT_SCOPE = "four_behavior_functional_weight_editing_v13_source_pool_construction"
FINAL_REDACTED_SCOPE = (
    "redacted_final_functional_weight_editing_v13_source_pool_audit_surface_only"
)
DEVELOPMENT_SCOPE = (
    "four_behavior_functional_weight_editing_v13_hybrid_signature_support_optimization_development"
)
FINAL_SCOPE = "four_behavior_functional_weight_editing_v13_hybrid_signature_support_optimization_final"
EDITOR_METHOD = (
    "v9_selected_target_signature_source_initialized_support_optimization_v13"
)
MATCHED_ALPHA = 0.975
OPTIMIZATION_STEPS = 130
OPTIMIZATION_LR = 0.03
OPTIMIZER_BETAS = (0.9, 0.999)
OPTIMIZER_EPS = 1e-8
GRAD_CLIP_NORM = 10.0
TARGET_SUPPORT_BCE_WEIGHT = 4.0
CONFLICT_SUPPORT_BCE_WEIGHT = 2.0
COMPATIBLE_SUPPORT_MSE_WEIGHT = 0.01
SIGNATURE_MSE_WEIGHT = 0.01
SOURCE_WEIGHT_L2_WEIGHT = 0.0005
PREREGISTERED_ARG_VALUES = {
    "generic_negative_cap": 1024,
    "hard_negative_cap": 1024,
    "heldout_per_class": 64,
    "lr": 0.003,
    "positive_cap": 2048,
    "random_controls": 16,
    "source_margin_gate": 0.40,
    "support_per_class": 160,
    "train_epochs": 350,
}
THRESHOLDS = {
    "expected_controls_per_record": 24,
    "expected_record_count": 288,
    "expected_per_direction_count": 24,
    "min_aggregate_individual_pass_rate": 0.85,
    "min_aggregate_pareto_undominated_rate": 0.85,
    "min_aggregate_target_prediction_rate": 0.95,
    "min_direction_individual_pass_rate": 0.70,
    "min_direction_conflict_target_accuracy": 0.70,
    "min_direction_conflict_target_accuracy_improvement": 0.20,
    "min_direction_mean_full_retrieval_minus_matched_compatible_source_output_mse": 0.0,
    "min_direction_mean_target_margin": 0.20,
    "min_direction_pareto_undominated_rate": 0.70,
    "min_direction_target_prediction_rate": 0.90,
    "min_mean_full_retrieval_minus_matched_compatible_source_output_mse": 10.0,
    "min_mean_matched_minus_full_retrieval_target_margin": -0.25,
    "min_mean_matched_target_margin": 0.50,
    "min_aggregate_conflict_target_accuracy": 0.85,
    "min_aggregate_conflict_target_accuracy_improvement": 0.50,
    "min_aggregate_signature_target_margin_advantage": 0.0,
    "min_aggregate_signature_compatible_mse_advantage": 2.0,
    "min_aggregate_shuffled_target_margin_advantage": 0.05,
    "min_aggregate_shuffled_compatible_mse_advantage": 2.0,
    "min_per_record_no_signature_target_margin_advantage": 0.02,
    "min_per_record_no_signature_compatible_mse_advantage": 5.0,
    "min_per_record_shuffled_target_margin_advantage": 0.05,
    "min_per_record_shuffled_compatible_mse_advantage": 5.0,
    "min_per_record_matched_target_margin": 0.20,
    "min_per_record_conflict_target_accuracy": 0.70,
    "min_per_record_conflict_target_accuracy_improvement": 0.20,
    "random_controls_per_record": 16,
}
FORBIDDEN_FINAL_DETAIL_KEYS = v10.FORBIDDEN_FINAL_DETAIL_KEYS
PRIOR_FINAL_RAW_PATHS = {
    REPO_ROOT / "runs" / "four_behavior_decoder_source_pools_v2" / "final_subjects.json",
    *[
        REPO_ROOT
        / "runs"
        / f"four_behavior_representation_steering_v{i}_pools"
        / "final_subjects.json"
        for i in range(1, 10)
    ],
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v10_pools" / "final_subjects.json",
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v11_pools" / "final_subjects.json",
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v12_pools" / "final_subjects.json",
}
V13_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"
EXPECTED_SPLIT_COUNTS = {
    "has_majority_to_mountain_pattern": {"compatible": 33, "conflict": 95},
    "has_majority_to_sorted_ascending": {"compatible": 63, "conflict": 65},
    "has_majority_to_sorted_descending": {"compatible": 64, "conflict": 64},
    "mountain_pattern_to_has_majority": {"compatible": 32, "conflict": 96},
    "mountain_pattern_to_sorted_ascending": {"compatible": 64, "conflict": 64},
    "mountain_pattern_to_sorted_descending": {"compatible": 64, "conflict": 64},
    "sorted_ascending_to_has_majority": {"compatible": 38, "conflict": 90},
    "sorted_ascending_to_mountain_pattern": {"compatible": 59, "conflict": 69},
    "sorted_ascending_to_sorted_descending": {"compatible": 63, "conflict": 65},
    "sorted_descending_to_has_majority": {"compatible": 50, "conflict": 78},
    "sorted_descending_to_mountain_pattern": {"compatible": 46, "conflict": 82},
    "sorted_descending_to_sorted_ascending": {"compatible": 64, "conflict": 64},
}
EXPECTED_SUPPORT_SPLIT_COUNTS = {
    "has_majority_to_mountain_pattern": {"compatible": 80, "conflict": 240},
    "has_majority_to_sorted_ascending": {"compatible": 160, "conflict": 160},
    "has_majority_to_sorted_descending": {"compatible": 160, "conflict": 160},
    "mountain_pattern_to_has_majority": {"compatible": 82, "conflict": 238},
    "mountain_pattern_to_sorted_ascending": {"compatible": 160, "conflict": 160},
    "mountain_pattern_to_sorted_descending": {"compatible": 158, "conflict": 162},
    "sorted_ascending_to_has_majority": {"compatible": 105, "conflict": 215},
    "sorted_ascending_to_mountain_pattern": {"compatible": 137, "conflict": 183},
    "sorted_ascending_to_sorted_descending": {"compatible": 158, "conflict": 162},
    "sorted_descending_to_has_majority": {"compatible": 101, "conflict": 219},
    "sorted_descending_to_mountain_pattern": {"compatible": 139, "conflict": 181},
    "sorted_descending_to_sorted_ascending": {"compatible": 160, "conflict": 160},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["generate-pools", "development", "final"],
        required=True,
    )
    parser.add_argument(
        "--pool-dir",
        default=str(DEFAULT_POOL_DIR.relative_to(REPO_ROOT)),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)),
    )
    parser.add_argument("--train-epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--source-margin-gate", type=float, default=0.40)
    parser.add_argument("--support-per-class", type=int, default=160)
    parser.add_argument("--heldout-per-class", type=int, default=64)
    parser.add_argument("--positive-cap", type=int, default=2048)
    parser.add_argument("--hard-negative-cap", type=int, default=1024)
    parser.add_argument("--generic-negative-cap", type=int, default=1024)
    parser.add_argument("--random-controls", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_preregistered_args(args)
    pool_dir = REPO_ROOT / args.pool_dir
    output_dir = REPO_ROOT / args.output_dir
    assert_preregistered_pool_dir(pool_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "generate-pools":
        result = generate_pools(args, pool_dir)
    elif args.phase == "development":
        result = run_development(args, pool_dir, output_dir)
    else:
        result = run_final(args, pool_dir, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("passed", False):
        raise SystemExit(1)


def validate_preregistered_args(args: argparse.Namespace) -> None:
    failures = []
    for name, expected in PREREGISTERED_ARG_VALUES.items():
        actual = getattr(args, name)
        if actual != expected:
            failures.append(f"{name}={actual!r} does not match preregistered {expected!r}")
    if failures:
        raise ValueError("Non-preregistered V13 parameter override: " + "; ".join(failures))


def assert_preregistered_pool_dir(pool_dir: Path) -> None:
    resolved = pool_dir.resolve()
    if resolved != DEFAULT_POOL_DIR.resolve():
        raise ValueError(f"V13 requires preregistered pool directory: {DEFAULT_POOL_DIR}")


def assert_no_forbidden_final_raw_paths(
    paths: Iterable[Path | str],
    *,
    allow_v13_final: bool,
) -> None:
    forbidden = {path.resolve() for path in PRIOR_FINAL_RAW_PATHS}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in forbidden:
            raise ValueError(f"prior sealed final raw path is forbidden: {path}")
        if path.name != "final_subjects.json":
            continue
        if allow_v13_final and path == V13_FINAL_RAW.resolve():
            continue
        raise ValueError(f"V13 final raw path is forbidden before final eval: {path}")


def generate_pools(args: argparse.Namespace, pool_dir: Path) -> Dict[str, Any]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    seed_preflight = v9.build_seed_preflight(
        POOL_CONFIGS,
        behavior_stride=SEED_BEHAVIOR_STRIDE,
    )
    if seed_preflight["failures"]:
        result = {"failures": seed_preflight["failures"], "passed": False}
        (pool_dir / "combined_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        return result

    suite = build_suite(args.support_per_class, args.heldout_per_class)
    heldout_sequences = build_heldout_sequences(suite)
    candidate_pools = build_candidate_pools(heldout_sequences)
    candidate_pool_summary = summarize_candidate_pools(candidate_pools)
    probe_examples = build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    source_args = SimpleNamespace(
        generic_negative_cap=args.generic_negative_cap,
        hard_negative_cap=args.hard_negative_cap,
        heldout_per_class=args.heldout_per_class,
        lr=args.lr,
        positive_cap=args.positive_cap,
        source_margin_gate=args.source_margin_gate,
        support_per_class=args.support_per_class,
        train_epochs=args.train_epochs,
    )
    pool_payloads = {}
    pool_summaries = {}
    for pool_name, pool_config in POOL_CONFIGS.items():
        payload = poolgen.generate_pool(
            args=source_args,
            pool_name=pool_name,
            pool_config=pool_config,
            suite=suite,
            heldout_sequences=heldout_sequences,
            candidate_pools=candidate_pools,
            candidate_pool_summary=candidate_pool_summary,
            probe_examples=probe_examples,
        )
        payload["claim_scope"] = SOURCE_POOL_SCOPE
        payload["config"]["base_seed"] = int(pool_config["base_seed"])
        payload["config"]["seed_behavior_stride"] = int(SEED_BEHAVIOR_STRIDE)
        payload["pool_redacted_payload_sha256"] = stable_hash_json(
            poolgen.redact_weights_and_signatures(payload)
        )
        pool_payloads[pool_name] = payload
        pool_path = pool_dir / f"{pool_name}_subjects.json"
        pool_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        summary = poolgen.summarize_pool(payload)
        summary["pool_file_sha256"] = v1.sha256_file(pool_path)
        summary["pool_redacted_payload_sha256"] = payload["pool_redacted_payload_sha256"]
        pool_summaries[pool_name] = summary

    final_redacted = poolgen.build_final_redacted_summary(pool_payloads["final"])
    final_redacted["claim_scope"] = FINAL_REDACTED_SCOPE
    final_redacted["pool_file_sha256"] = pool_summaries["final"]["pool_file_sha256"]
    final_redacted["summary_payload_sha256"] = stable_hash_json(final_redacted)
    (pool_dir / "final_redacted_audit.json").write_text(
        json.dumps(final_redacted, indent=2, sort_keys=True)
    )

    audit = poolgen.build_combined_audit(
        pool_payloads=pool_payloads,
        pool_summaries=pool_summaries,
        seed_preflight=seed_preflight,
        suite=suite,
        probe_examples=probe_examples,
    )
    audit["claim_scope"] = SOURCE_AUDIT_SCOPE
    audit = v10.redact_combined_audit(audit)
    audit["audit_payload_sha256"] = stable_hash_json(audit)
    (pool_dir / "combined_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    return {
        "combined_audit_path": v1.rel(pool_dir / "combined_audit.json"),
        "final_redacted_audit_path": v1.rel(pool_dir / "final_redacted_audit.json"),
        "passed": bool(audit["passed"]),
        "pool_summaries": audit["pool_summaries"],
        "seed_preflight": seed_preflight,
    }


def development_input_paths(pool_dir: Path) -> Dict[str, Path]:
    return {
        "combined_audit": pool_dir / "combined_audit.json",
        "development": pool_dir / "development_subjects.json",
        "final_redacted_audit": pool_dir / "final_redacted_audit.json",
        "train": pool_dir / "train_subjects.json",
    }


def run_development(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> Dict[str, Any]:
    paths = development_input_paths(pool_dir)
    assert_no_forbidden_final_raw_paths(paths.values(), allow_v13_final=False)
    payload = train_and_evaluate(
        train_path=paths["train"],
        eval_path=paths["development"],
        combined_audit_path=paths["combined_audit"],
        final_redacted_path=paths["final_redacted_audit"],
        output_dir=output_dir,
        phase="development",
        random_controls=args.random_controls,
        allow_eval_final_raw=False,
    )
    if payload["passed"]:
        payload["next_action"] = "eligible_for_one_shot_final_eval_without_method_changes"
    else:
        payload["next_action"] = "log_negative_development_result_do_not_open_final_raw"
    result_path = output_dir / "development_results.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def run_final(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> Dict[str, Any]:
    development_path = output_dir / "development_results.json"
    if not development_path.exists():
        raise FileNotFoundError("development_results.json is required before final eval")
    development = json.loads(development_path.read_text())
    authorization_failures = validate_development_authorizes_final(
        development=development,
        pool_dir=pool_dir,
    )
    if authorization_failures:
        raise RuntimeError(
            "development artifact does not authorize final raw evaluation: "
            + "; ".join(authorization_failures)
        )
    eval_path = pool_dir / "final_subjects.json"
    assert_no_forbidden_final_raw_paths([eval_path], allow_v13_final=True)
    payload = train_and_evaluate(
        train_path=pool_dir / "train_subjects.json",
        eval_path=eval_path,
        combined_audit_path=pool_dir / "combined_audit.json",
        final_redacted_path=pool_dir / "final_redacted_audit.json",
        output_dir=output_dir,
        phase="final",
        random_controls=args.random_controls,
        allow_eval_final_raw=True,
    )
    result_path = output_dir / "final_results.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def validate_development_authorizes_final(
    *,
    development: Mapping[str, Any],
    pool_dir: Path,
) -> list[str]:
    failures = []
    expected = {
        "claim_scope": DEVELOPMENT_SCOPE,
        "combined_audit_sha256": v1.sha256_file(pool_dir / "combined_audit.json"),
        "editor_method": EDITOR_METHOD,
        "eval_pool_sha256": v1.sha256_file(pool_dir / "development_subjects.json"),
        "final_redacted_audit_sha256": v1.sha256_file(pool_dir / "final_redacted_audit.json"),
        "implementation_sha256": v1.sha256_file(SCRIPT_PATH),
        "next_action": "eligible_for_one_shot_final_eval_without_method_changes",
        "phase": "development",
        "preregistration_sha256": v1.sha256_file(PREREG_PATH),
        "train_pool_sha256": v1.sha256_file(pool_dir / "train_subjects.json"),
    }
    if development.get("passed") is not True:
        failures.append("development did not pass")
    for key, expected_value in expected.items():
        if development.get(key) != expected_value:
            failures.append(
                f"development {key}={development.get(key)!r} does not match {expected_value!r}"
            )
    return failures


def train_and_evaluate(
    *,
    train_path: Path,
    eval_path: Path,
    combined_audit_path: Path,
    final_redacted_path: Path,
    output_dir: Path,
    phase: str,
    random_controls: int,
    allow_eval_final_raw: bool,
) -> Dict[str, Any]:
    assert_no_forbidden_final_raw_paths(
        [train_path, eval_path, combined_audit_path, final_redacted_path],
        allow_v13_final=allow_eval_final_raw,
    )
    train_payload = v1.load_json(train_path)
    eval_payload = v1.load_json(eval_path)
    combined_audit = v1.load_json(combined_audit_path)
    final_redacted = v1.load_json(final_redacted_path)
    contract_failures = validate_source_pool_contract(
        train_path=train_path,
        eval_path=eval_path,
        combined_audit_path=combined_audit_path,
        final_redacted_path=final_redacted_path,
        train_payload=train_payload,
        eval_payload=eval_payload,
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase=phase,
    )
    if contract_failures:
        raise ValueError("V13 source-pool contract validation failed: " + "; ".join(contract_failures))

    train_subjects = v1.accepted_records(train_payload)
    eval_subjects = v1.accepted_records(eval_payload)
    train_stats = fit_v13_train_statistics(train_subjects)
    probe_examples = build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    train_stats["probe_examples"] = probe_examples
    train_stats["probe_examples_hash"] = stable_hash_json(probe_examples)
    classifier, classifier_summary = v9.fit_primary_classifier(
        train_stats["z_train"],
        train_stats["y_train"],
    )
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    calibration_coefficients, calibration_summary = v9.fit_contrastive_calibration(
        subjects=train_subjects,
        train_stats=train_stats,
        classifier=classifier,
    )
    train_stats["calibration_coefficients"] = calibration_coefficients
    stats_path = output_dir / "v13_hybrid_signature_support_optimization_stats.pt"
    torch.save(
        {
            "calibration_coefficients": train_stats["calibration_coefficients"],
            "centroids": train_stats["centroids"],
            "method": EDITOR_METHOD,
            "probe_examples_hash": train_stats["probe_examples_hash"],
            "train_statistics_hash": train_only_statistics_hash(train_stats),
        },
        stats_path,
    )
    eval_result = evaluate_subjects(
        subjects=eval_subjects,
        train_stats=train_stats,
        classifier=classifier,
        random_controls=random_controls,
    )
    failures = [*contract_failures, *eval_result["failures"]]
    result = {
        **eval_result,
        "calibration_summary": calibration_summary,
        "claim_scope": DEVELOPMENT_SCOPE if phase == "development" else FINAL_SCOPE,
        "classifier_summary": classifier_summary,
        "combined_audit_path": v1.rel(combined_audit_path),
        "combined_audit_sha256": v1.sha256_file(combined_audit_path),
        "dirty_worktree_caveat": True,
        "editor_method": EDITOR_METHOD,
        "eval_pool_path": v1.rel(eval_path),
        "eval_pool_sha256": v1.sha256_file(eval_path),
        "final_redacted_audit_path": v1.rel(final_redacted_path),
        "final_redacted_audit_sha256": v1.sha256_file(final_redacted_path),
        "final_redacted_summary": final_redacted,
        "forbidden_prior_final_raw_opened": False,
        "forbidden_v13_final_raw_opened_before_authorization": False,
        "implementation_sha256": v1.sha256_file(SCRIPT_PATH),
        "limitations": (
            "Small-subject source-label-known, target-label-requested hybrid "
            "signature-support optimization evidence only; not pure signature-only "
            "editing, source-label inference, source-free decoding, larger-model "
            "evidence, broad MUAT proof, or arbitrary capability preservation."
        ),
        "optimizer_config": {
            "amsgrad": False,
            "betas": list(OPTIMIZER_BETAS),
            "eps": OPTIMIZER_EPS,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "lr": OPTIMIZATION_LR,
            "steps": OPTIMIZATION_STEPS,
            "weight_decay": 0.0,
        },
        "phase": phase,
        "pool_overlap_counts": combined_audit.get("overlap_counts", {}),
        "probe_examples_hash": train_stats["probe_examples_hash"],
        "preregistration_sha256": v1.sha256_file(PREREG_PATH),
        "source_pool_audit_passed": bool(combined_audit.get("passed", False)),
        "stats_path": v1.rel(stats_path),
        "stats_sha256": v1.sha256_file(stats_path),
        "thresholds": THRESHOLDS,
        "train_only_statistics_hash": train_only_statistics_hash(train_stats),
        "train_pool_path": v1.rel(train_path),
        "train_pool_sha256": v1.sha256_file(train_path),
    }
    result["failures"] = failures
    result["passed"] = not failures
    return result


def validate_source_pool_contract(
    *,
    train_path: Path,
    eval_path: Path,
    combined_audit_path: Path,
    final_redacted_path: Path,
    train_payload: Mapping[str, Any],
    eval_payload: Mapping[str, Any],
    combined_audit: Mapping[str, Any],
    final_redacted: Mapping[str, Any],
    phase: str,
) -> list[str]:
    failures = []
    if train_payload.get("claim_scope") != SOURCE_POOL_SCOPE:
        failures.append("train pool claim_scope mismatch")
    if eval_payload.get("claim_scope") != SOURCE_POOL_SCOPE:
        failures.append("eval pool claim_scope mismatch")
    if combined_audit.get("claim_scope") != SOURCE_AUDIT_SCOPE:
        failures.append("combined audit claim_scope mismatch")
    if final_redacted.get("claim_scope") != FINAL_REDACTED_SCOPE:
        failures.append("final redacted audit claim_scope mismatch")
    forbidden_surfaces = {
        "combined_audit.pool_summaries.final": combined_audit.get("pool_summaries", {}).get(
            "final",
            {},
        ),
        "final_redacted": final_redacted,
    }
    for surface_name, surface in forbidden_surfaces.items():
        forbidden_paths = v10.find_forbidden_final_detail_paths(surface, prefix=surface_name)
        if forbidden_paths:
            failures.append(
                f"{surface_name} exposes forbidden final detail keys: "
                + ", ".join(forbidden_paths[:12])
            )
    if not combined_audit.get("passed", False):
        failures.append("combined source-pool audit did not pass")
    expected_eval_pool = "development" if phase == "development" else "final"
    pool_summaries = combined_audit.get("pool_summaries", {})
    expected_hashes = {
        "train": v1.sha256_file(train_path),
        expected_eval_pool: v1.sha256_file(eval_path),
    }
    for pool_name, actual_hash in expected_hashes.items():
        audit_hash = pool_summaries.get(pool_name, {}).get("pool_file_sha256")
        if audit_hash != actual_hash:
            failures.append(f"{pool_name} pool hash mismatch")
    if final_redacted.get("pool_file_sha256") != pool_summaries.get("final", {}).get(
        "pool_file_sha256"
    ):
        failures.append("final redacted pool hash does not match combined audit")
    if phase == "final":
        final_raw_hash = v1.sha256_file(DEFAULT_POOL_DIR / "final_subjects.json")
        if final_redacted.get("pool_file_sha256") != final_raw_hash:
            failures.append("final redacted pool hash does not match final raw file")
    for pool_name, expected_count in (("train", 64), (expected_eval_pool, 24)):
        counts = pool_summaries.get(pool_name, {}).get("accepted_counts_by_behavior", {})
        for pattern in PATTERNS:
            if counts.get(pattern) != expected_count:
                failures.append(f"{pool_name} accepted count for {pattern} is not {expected_count}")
    for name, value in combined_audit.get("overlap_counts", {}).items():
        if isinstance(value, (int, float)) and value != 0:
            failures.append(f"cross-pool overlap {name} is nonzero")
    if phase == "development":
        assert_no_forbidden_final_raw_paths(
            [train_path, eval_path, combined_audit_path, final_redacted_path],
            allow_v13_final=False,
        )
    return failures


def fit_v13_train_statistics(subjects: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    stats = v9.fit_train_statistics(subjects)
    train_weights = torch.tensor([record["weights"] for record in subjects], dtype=torch.float32)
    stats["global_weight_centroid"] = train_weights.mean(dim=0)
    stats["train_subjects"] = list(subjects)
    stats["train_weights"] = train_weights
    stats["weight_centroids"] = {
        pattern: train_weights[
            torch.tensor([v1.behavior_of(record) == pattern for record in subjects], dtype=torch.bool)
        ].mean(dim=0)
        for pattern in PATTERNS
    }
    return stats


def evaluate_subjects(
    *,
    subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    random_controls: int,
) -> Dict[str, Any]:
    records = []
    for subject in subjects:
        source = v1.behavior_of(subject)
        source_signature = torch.tensor(subject["signature"], dtype=torch.float32)
        z = (source_signature - train_stats["sig_mean"]) / train_stats["sig_std"]
        source_weights = torch.tensor(subject["weights"], dtype=torch.float32)
        for target in PATTERNS:
            if target == source:
                continue
            records.append(evaluate_record(
                subject=subject,
                z=z,
                source=source,
                target=target,
                source_weights=source_weights,
                train_stats=train_stats,
                classifier=classifier,
                random_controls=random_controls,
            ))
    aggregate = summarize_records(records)
    by_direction = {
        v1.vector_key(source, target): summarize_records([
            record for record in records
            if record["source_behavior"] == source and record["target_behavior"] == target
        ])
        for source in PATTERNS
        for target in PATTERNS
        if source != target
    }
    failures = gate_failures(aggregate=aggregate, by_direction=by_direction, records=records)
    return {
        "aggregate": aggregate,
        "by_direction": by_direction,
        "failures": failures,
        "passed": not failures,
        "records": records,
    }


def evaluate_record(
    *,
    subject: Mapping[str, Any],
    z: torch.Tensor,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    random_controls: int,
) -> Dict[str, Any]:
    selected = v10.selected_v9_conditioning(
        z=z,
        source=source,
        target=target,
        train_stats=train_stats,
        classifier=classifier,
    )
    retrieval = v10.nearest_train_target_retrieval(
        selected_signature_norm=selected["candidate_z"],
        target=target,
        train_stats=train_stats,
    )
    unaligned_retrieved_weights = retrieval["weights"]
    retrieved_weights = align_target_to_source(
        source_weights=source_weights,
        target_weights=unaligned_retrieved_weights,
    )
    matched_weights = optimize_editor_weights(
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats,
        signature_target_norm=selected["candidate_z"],
        loss_weights=matched_loss_weights(),
    )
    matched_delta_norm = (matched_weights - source_weights).norm()
    matched = {
        **functional_metrics(matched_weights, source, target, source_weights),
        "delta_norm": float(matched_delta_norm.item()),
        "optimization_steps": OPTIMIZATION_STEPS,
        "optimizer_lr": OPTIMIZATION_LR,
        "retrieved_subject_id_for_v12_style_controls": retrieval["subject_id"],
        "selected_candidate_index": int(selected["candidate_index"]),
        "selected_centroid_improvement": selected["selected_centroid_improvement"],
        "selected_displacement_norm": selected["selected_displacement_norm"],
        "selected_primary_target_margin": selected["selected_primary_target_margin"],
    }
    controls = build_controls(
        subject=subject,
        z=z,
        source=source,
        target=target,
        source_weights=source_weights,
        retrieved_weights=retrieved_weights,
        retrieved_subject_id=retrieval["subject_id"],
        selected_signature_norm=selected["candidate_z"],
        matched_delta_norm=matched_delta_norm,
        train_stats=train_stats,
        classifier=classifier,
        random_controls=random_controls,
    )
    for control in controls:
        control["matched_minus_control_target_margin"] = matched["target_margin"] - control[
            "target_margin"
        ]
        control["control_minus_matched_compatible_source_output_mse"] = (
            control["compatible_source_output_mse"]
            - matched["compatible_source_output_mse"]
        )
    full_retrieval = next(
        control for control in controls
        if control["control_type"] == "aligned_full_nearest_target_retrieval"
    )
    best_target = max(controls, key=lambda item: item["target_margin"])
    pareto_dominators = [
        control for control in controls if pareto_dominates_functional(control, matched)
    ]
    matched["pareto_dominator_count"] = len(pareto_dominators)
    matched["pareto_dominator_types"] = sorted({
        control["control_type"] for control in pareto_dominators
    })
    matched["pareto_undominated"] = not pareto_dominators
    matched["target_prediction_pass"] = matched["predicted_behavior"] == target
    no_signature = single_control(controls, "no_signature_optimizer")
    shuffled_signature = next(
        control for control in controls
        if control["control_type"].startswith("shuffled_signature_optimizer:")
    )
    matched["matched_minus_no_signature_target_margin"] = (
        matched["target_margin"] - no_signature["target_margin"]
    )
    matched["no_signature_minus_matched_compatible_source_output_mse"] = (
        no_signature["compatible_source_output_mse"]
        - matched["compatible_source_output_mse"]
    )
    matched["matched_minus_shuffled_signature_target_margin"] = (
        matched["target_margin"] - shuffled_signature["target_margin"]
    )
    matched["shuffled_signature_minus_matched_compatible_source_output_mse"] = (
        shuffled_signature["compatible_source_output_mse"]
        - matched["compatible_source_output_mse"]
    )
    matched["individual_all_gates_passed"] = individual_passed(
        matched=matched,
        full_retrieval=full_retrieval,
    )
    return {
        "controls": v10.strip_control_weights(controls),
        "individual_all_gates_passed": matched["individual_all_gates_passed"],
        "matched": v10.strip_weight(matched),
        "random_control_count": sum(
            1
            for control in controls
            if control["control_type"].startswith("random_norm_matched_weight_delta")
        ),
        "source_behavior": source,
        "subject_id": str(subject["subject_id"]),
        "summary": {
            "best_control_target_margin": best_target["target_margin"],
            "best_control_type": best_target["control_type"],
            "full_retrieval_minus_matched_compatible_source_output_mse": (
                full_retrieval["compatible_source_output_mse"]
                - matched["compatible_source_output_mse"]
            ),
            "matched_minus_best_control_target_margin": (
                matched["target_margin"] - best_target["target_margin"]
            ),
            "matched_minus_full_retrieval_target_margin": matched["target_margin"]
            - full_retrieval["target_margin"],
            "matched_minus_no_signature_target_margin": matched[
                "matched_minus_no_signature_target_margin"
            ],
            "matched_minus_shuffled_signature_target_margin": matched[
                "matched_minus_shuffled_signature_target_margin"
            ],
            "no_signature_minus_matched_compatible_source_output_mse": matched[
                "no_signature_minus_matched_compatible_source_output_mse"
            ],
            "pareto_undominated": matched["pareto_undominated"],
            "shuffled_signature_minus_matched_compatible_source_output_mse": matched[
                "shuffled_signature_minus_matched_compatible_source_output_mse"
            ],
            "target_prediction_pass": matched["target_prediction_pass"],
        },
        "target_behavior": target,
    }


def build_controls(
    *,
    subject: Mapping[str, Any],
    z: torch.Tensor,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    retrieved_weights: torch.Tensor,
    retrieved_subject_id: str,
    selected_signature_norm: torch.Tensor,
    matched_delta_norm: torch.Tensor,
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    random_controls: int,
) -> list[Dict[str, Any]]:
    controls = []
    controls.append(control_record("no_edit", source_weights, source, target, source_weights))
    controls.append(control_record(
        "aligned_full_nearest_target_retrieval",
        retrieved_weights,
        source,
        target,
        source_weights,
        {"alpha": 1.0, "retrieved_subject_id": retrieved_subject_id},
    ))
    controls.append(control_record(
        "aligned_interpolation_alpha_0.975",
        interpolate_weights(
            source_weights=source_weights,
            target_weights=retrieved_weights,
            alpha=MATCHED_ALPHA,
        ),
        source,
        target,
        source_weights,
        {"alpha": MATCHED_ALPHA, "retrieved_subject_id": retrieved_subject_id},
    ))
    controls.append(optimizer_control_record(
        control_type="no_signature_optimizer",
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats,
        signature_target_norm=selected_signature_norm,
        loss_weights={**matched_loss_weights(), "signature": 0.0},
    ))
    controls.append(optimizer_control_record(
        control_type="source_signature_optimizer",
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats,
        signature_target_norm=z,
        loss_weights=matched_loss_weights(),
    ))
    shuffled = select_shuffled_target(
        subject_id=str(subject["subject_id"]),
        source=source,
        target=target,
    )
    shuffled_selected = v10.selected_v9_conditioning(
        z=z,
        source=source,
        target=shuffled,
        train_stats=train_stats,
        classifier=classifier,
    )
    controls.append(optimizer_control_record(
        control_type=f"shuffled_signature_optimizer:{shuffled}",
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats,
        signature_target_norm=shuffled_selected["candidate_z"],
        loss_weights=matched_loss_weights(),
        metadata={"shuffled_target_behavior": shuffled},
    ))
    controls.append(optimizer_control_record(
        control_type="signature_only_optimizer",
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats,
        signature_target_norm=selected_signature_norm,
        loss_weights={
            "target": 0.0,
            "conflict": 0.0,
            "compatible": 0.0,
            "signature": SIGNATURE_MSE_WEIGHT,
            "l2": SOURCE_WEIGHT_L2_WEIGHT,
        },
    ))
    controls.append(optimizer_control_record(
        control_type="target_only_support_optimizer",
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats,
        signature_target_norm=selected_signature_norm,
        loss_weights={
            "target": TARGET_SUPPORT_BCE_WEIGHT,
            "conflict": 0.0,
            "compatible": 0.0,
            "signature": 0.0,
            "l2": SOURCE_WEIGHT_L2_WEIGHT,
        },
    ))
    controls.extend(random_weight_delta_controls(
        subject_id=str(subject["subject_id"]),
        source=source,
        target=target,
        source_weights=source_weights,
        matched_delta_norm=matched_delta_norm,
        random_controls=random_controls,
    ))
    return controls


def matched_loss_weights() -> Dict[str, float]:
    return {
        "target": TARGET_SUPPORT_BCE_WEIGHT,
        "conflict": CONFLICT_SUPPORT_BCE_WEIGHT,
        "compatible": COMPATIBLE_SUPPORT_MSE_WEIGHT,
        "signature": SIGNATURE_MSE_WEIGHT,
        "l2": SOURCE_WEIGHT_L2_WEIGHT,
    }


def optimizer_control_record(
    *,
    control_type: str,
    source_weights: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    signature_target_norm: torch.Tensor,
    loss_weights: Mapping[str, float],
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    weights = optimize_editor_weights(
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats,
        signature_target_norm=signature_target_norm,
        loss_weights=loss_weights,
    )
    return control_record(
        control_type,
        weights,
        source,
        target,
        source_weights,
        {
            "delta_norm": float((weights - source_weights).norm().item()),
            "optimization_steps": OPTIMIZATION_STEPS,
            **dict(metadata or {}),
        },
    )


def optimize_editor_weights(
    *,
    source_weights: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    signature_target_norm: torch.Tensor,
    loss_weights: Mapping[str, float],
) -> torch.Tensor:
    weights = source_weights.detach().clone().requires_grad_(True)
    source_detached = source_weights.detach().clone()
    signature_target = signature_target_norm.detach().to(dtype=torch.float32)
    support = prepare_support_tensors_with_source_logits(
        source_weights=source_detached,
        source=source,
        target=target,
    )
    probe_examples = train_stats["probe_examples"]
    sig_mean = train_stats["sig_mean"].detach()
    sig_std = train_stats["sig_std"].detach()
    optimizer = torch.optim.AdamW(
        [weights],
        lr=OPTIMIZATION_LR,
        betas=OPTIMIZER_BETAS,
        eps=OPTIMIZER_EPS,
        weight_decay=0.0,
        amsgrad=False,
    )
    for _ in range(OPTIMIZATION_STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), dtype=weights.dtype)
        if loss_weights.get("target", 0.0):
            logits = v10.decoder_v1.subject_forward_flat_batch(
                weights.unsqueeze(0),
                support["target_inputs"],
            )[0]
            target_loss = F.binary_cross_entropy_with_logits(
                logits,
                support["target_labels"],
            )
            loss = loss + float(loss_weights["target"]) * target_loss
        if loss_weights.get("conflict", 0.0):
            logits = v10.decoder_v1.subject_forward_flat_batch(
                weights.unsqueeze(0),
                support["conflict_inputs"],
            )[0]
            conflict_loss = F.binary_cross_entropy_with_logits(
                logits,
                support["conflict_target_labels"],
            )
            loss = loss + float(loss_weights["conflict"]) * conflict_loss
        if loss_weights.get("compatible", 0.0):
            edited = v10.decoder_v1.subject_forward_flat_batch(
                weights.unsqueeze(0),
                support["compatible_inputs"],
            )[0]
            compatible_loss = F.mse_loss(edited, support["compatible_source_logits"])
            loss = loss + float(loss_weights["compatible"]) * compatible_loss
        if loss_weights.get("signature", 0.0):
            signature = differentiable_signature(weights, probe_examples)
            signature_norm = (signature - sig_mean) / sig_std
            signature_loss = F.mse_loss(signature_norm, signature_target)
            loss = loss + float(loss_weights["signature"]) * signature_loss
        if loss_weights.get("l2", 0.0):
            loss = loss + float(loss_weights["l2"]) * F.mse_loss(weights, source_detached)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([weights], GRAD_CLIP_NORM)
        optimizer.step()
    return weights.detach().cpu()


def support_loss_tensors(*, source: str, target: str) -> Dict[str, torch.Tensor]:
    suite = v10.evaluation_suite()
    target_positive = v10.decoder_v1.sequence_tensor(suite["support"][target]["positive"])
    target_negative = v10.decoder_v1.sequence_tensor(suite["support"][target]["negative"])
    target_inputs = torch.cat([target_positive, target_negative], dim=0)
    target_labels = torch.cat([
        torch.ones(len(target_positive), dtype=torch.float32),
        torch.zeros(len(target_negative), dtype=torch.float32),
    ])

    return {
        "target_inputs": target_inputs,
        "target_labels": target_labels,
    }


def source_target_support_split(*, source: str, target: str) -> Dict[str, Any]:
    suite = v10.evaluation_suite()
    sequences = [
        tuple(sequence)
        for sequence in (
            suite["support"][source]["positive"]
            + suite["support"][source]["negative"]
        )
    ]
    source_labels = [int(PREDICATES[source](sequence)) for sequence in sequences]
    target_labels = [int(PREDICATES[target](sequence)) for sequence in sequences]
    compatible = [
        sequence for sequence, source_label, target_label
        in zip(sequences, source_labels, target_labels)
        if source_label == target_label
    ]
    conflict_pairs = [
        (sequence, target_label)
        for sequence, source_label, target_label
        in zip(sequences, source_labels, target_labels)
        if source_label != target_label
    ]
    direction = v1.vector_key(source, target)
    expected = EXPECTED_SUPPORT_SPLIT_COUNTS[direction]
    if len(compatible) != expected["compatible"] or len(conflict_pairs) != expected["conflict"]:
        raise ValueError(
            f"{direction} support split count mismatch: compatible={len(compatible)} "
            f"conflict={len(conflict_pairs)} expected={expected}"
        )
    return {
        "compatible_count": len(compatible),
        "compatible_inputs": v10.decoder_v1.sequence_tensor(compatible),
        "conflict_count": len(conflict_pairs),
        "conflict_inputs": v10.decoder_v1.sequence_tensor([pair[0] for pair in conflict_pairs]),
        "conflict_target_labels": torch.tensor(
            [pair[1] for pair in conflict_pairs],
            dtype=torch.float32,
        ),
    }


def prepare_support_tensors_with_source_logits(
    *,
    source_weights: torch.Tensor,
    source: str,
    target: str,
) -> Dict[str, torch.Tensor]:
    support = support_loss_tensors(source=source, target=target)
    split = source_target_support_split(source=source, target=target)
    support.update({
        "compatible_inputs": split["compatible_inputs"],
        "conflict_inputs": split["conflict_inputs"],
        "conflict_target_labels": split["conflict_target_labels"],
    })
    with torch.no_grad():
        source_logits = v10.decoder_v1.subject_forward_flat_batch(
            source_weights.detach().unsqueeze(0),
            support["compatible_inputs"],
        )[0]
    support["compatible_source_logits"] = source_logits.detach()
    return support


def differentiable_signature(
    flat_weights: torch.Tensor,
    probe_examples: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    probe_inputs = torch.tensor(
        [example["sequence"] for example in probe_examples],
        dtype=torch.float32,
        device=flat_weights.device,
    )
    activations = hidden_activations_flat(flat_weights, probe_inputs)
    features: List[torch.Tensor] = []
    n_samples = int(probe_inputs.shape[0])
    for layer_activations in activations:
        for neuron_idx in range(layer_activations.shape[1]):
            neuron_acts = layer_activations[:, neuron_idx]
            mean = neuron_acts.mean()
            std = neuron_acts.std(unbiased=False)
            features.append(mean)
            features.append(std)
            fft_mag = torch.abs(torch.fft.fft(neuron_acts))[: max(1, n_samples // 2)]
            for value_idx in range(5):
                features.append(fft_mag[value_idx])
            for input_idx in range(5):
                features.append(safe_corrcoef(neuron_acts, probe_inputs[:, input_idx]))
            features.append(mean)
            features.append(std)
    return torch.stack(features)


def hidden_activations_flat(
    flat_weights: torch.Tensor,
    inputs: torch.Tensor,
) -> List[torch.Tensor]:
    x = inputs
    offset = 0
    activations = []
    for out_dim, in_dim in [(8, 5), (8, 8), (8, 8), (8, 8), (8, 8)]:
        size = out_dim * in_dim
        weight = flat_weights[offset:offset + size].view(out_dim, in_dim)
        offset += size
        bias = flat_weights[offset:offset + out_dim]
        offset += out_dim
        x = F.gelu(torch.matmul(x, weight.t()) + bias)
        activations.append(x)
    return activations


def safe_corrcoef(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = left_centered.norm() * right_centered.norm()
    return torch.where(
        denom > 0,
        torch.dot(left_centered, right_centered) / denom.clamp_min(1e-12),
        torch.zeros((), dtype=left.dtype, device=left.device),
    )


def single_control(controls: Sequence[Mapping[str, Any]], control_type: str) -> Mapping[str, Any]:
    matches = [control for control in controls if control["control_type"] == control_type]
    if len(matches) != 1:
        raise ValueError(f"expected one {control_type} control, found {len(matches)}")
    return matches[0]


def interpolate_weights(
    *,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    return (1.0 - float(alpha)) * source_weights + float(alpha) * target_weights


def control_record(
    control_type: str,
    weights: torch.Tensor,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "control_type": control_type,
        "weights": weights,
        **functional_metrics(weights, source, target, source_weights),
        **dict(metadata or {}),
    }


def functional_metrics(
    weights: torch.Tensor,
    source: str,
    target: str,
    source_weights: torch.Tensor,
) -> Dict[str, Any]:
    margins = {pattern: v10.behavior_margin(weights, pattern) for pattern in PATTERNS}
    split = source_target_split(source=source, target=target)
    compatible_inputs = split["compatible_inputs"]
    conflict_inputs = split["conflict_inputs"]
    conflict_labels = split["conflict_target_labels"]
    with torch.no_grad():
        edited_compatible = v10.decoder_v1.subject_forward_flat_batch(
            weights.unsqueeze(0),
            compatible_inputs,
        )[0]
        source_compatible = v10.decoder_v1.subject_forward_flat_batch(
            source_weights.unsqueeze(0),
            compatible_inputs,
        )[0]
        edited_conflict = v10.decoder_v1.subject_forward_flat_batch(
            weights.unsqueeze(0),
            conflict_inputs,
        )[0]
        source_conflict = v10.decoder_v1.subject_forward_flat_batch(
            source_weights.unsqueeze(0),
            conflict_inputs,
        )[0]
    edited_conflict_labels = (torch.sigmoid(edited_conflict) >= 0.5).float()
    source_conflict_labels = (torch.sigmoid(source_conflict) >= 0.5).float()
    conflict_target_accuracy = float((edited_conflict_labels == conflict_labels).float().mean().item())
    source_conflict_accuracy = float((source_conflict_labels == conflict_labels).float().mean().item())
    return {
        "behavior_margins": margins,
        "compatible_count": int(split["compatible_count"]),
        "compatible_source_output_mse": float(
            F.mse_loss(edited_compatible, source_compatible).item()
        ),
        "conflict_count": int(split["conflict_count"]),
        "conflict_source_accuracy": source_conflict_accuracy,
        "conflict_target_accuracy": conflict_target_accuracy,
        "conflict_target_accuracy_improvement": (
            conflict_target_accuracy - source_conflict_accuracy
        ),
        "predicted_behavior": max(PATTERNS, key=lambda pattern: margins[pattern]),
        "source_margin": margins[source],
        "target_margin": margins[target],
    }


def source_target_split(*, source: str, target: str) -> Dict[str, Any]:
    suite = v10.evaluation_suite()
    sequences = [
        tuple(sequence)
        for sequence in (
            suite["heldout"][source]["positive"]
            + suite["heldout"][source]["negative"]
        )
    ]
    source_labels = [int(PREDICATES[source](sequence)) for sequence in sequences]
    target_labels = [int(PREDICATES[target](sequence)) for sequence in sequences]
    compatible = [
        sequence for sequence, source_label, target_label
        in zip(sequences, source_labels, target_labels)
        if source_label == target_label
    ]
    conflict_pairs = [
        (sequence, target_label)
        for sequence, source_label, target_label
        in zip(sequences, source_labels, target_labels)
        if source_label != target_label
    ]
    direction = v1.vector_key(source, target)
    expected = EXPECTED_SPLIT_COUNTS[direction]
    if len(compatible) != expected["compatible"] or len(conflict_pairs) != expected["conflict"]:
        raise ValueError(
            f"{direction} split count mismatch: compatible={len(compatible)} "
            f"conflict={len(conflict_pairs)} expected={expected}"
        )
    return {
        "compatible_count": len(compatible),
        "compatible_inputs": v10.decoder_v1.sequence_tensor(compatible),
        "conflict_count": len(conflict_pairs),
        "conflict_inputs": v10.decoder_v1.sequence_tensor([pair[0] for pair in conflict_pairs]),
        "conflict_target_labels": torch.tensor(
            [pair[1] for pair in conflict_pairs],
            dtype=torch.float32,
        ),
    }


def align_target_to_source(
    *,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    source_layers = unpack_subject_weights(source_weights)
    target_layers = unpack_subject_weights(target_weights)
    target_weight_layers = target_layers["weights"]
    target_bias_layers = target_layers["biases"]
    for layer_index in range(5):
        source_features = torch.cat([
            source_layers["weights"][layer_index],
            source_layers["biases"][layer_index].unsqueeze(1),
        ], dim=1)
        target_features = torch.cat([
            target_weight_layers[layer_index],
            target_bias_layers[layer_index].unsqueeze(1),
        ], dim=1)
        cost = torch.cdist(source_features, target_features, p=2).pow(2)
        tie_break = torch.tensor(
            [
                [
                    1e-9 * float((source_index + 1) * (target_index + 1))
                    for target_index in range(cost.shape[1])
                ]
                for source_index in range(cost.shape[0])
            ],
            dtype=cost.dtype,
        )
        row_indices, target_indices = linear_sum_assignment((cost + tie_break).cpu().numpy())
        if list(row_indices) != list(range(cost.shape[0])):
            raise ValueError("unexpected Hungarian row ordering")
        permutation = torch.tensor(target_indices, dtype=torch.long)
        target_weight_layers[layer_index] = target_weight_layers[layer_index][permutation]
        target_bias_layers[layer_index] = target_bias_layers[layer_index][permutation]
        target_weight_layers[layer_index + 1] = target_weight_layers[layer_index + 1][:, permutation]
    return pack_subject_weights(target_weight_layers, target_bias_layers)


def unpack_subject_weights(flat_weights: torch.Tensor) -> Dict[str, list[torch.Tensor]]:
    offset = 0
    weights = []
    biases = []
    for layer_index in range(5):
        in_dim = 5 if layer_index == 0 else 8
        out_dim = 8
        weight_count = out_dim * in_dim
        weights.append(flat_weights[offset:offset + weight_count].view(out_dim, in_dim).clone())
        offset += weight_count
        biases.append(flat_weights[offset:offset + out_dim].clone())
        offset += out_dim
    weights.append(flat_weights[offset:offset + 8].view(1, 8).clone())
    offset += 8
    biases.append(flat_weights[offset:offset + 1].clone())
    offset += 1
    if offset != flat_weights.numel():
        raise ValueError(f"Expected {offset} flat weights, got {flat_weights.numel()}")
    return {"biases": biases, "weights": weights}


def pack_subject_weights(
    weights: Sequence[torch.Tensor],
    biases: Sequence[torch.Tensor],
) -> torch.Tensor:
    parts = []
    for layer_index in range(5):
        parts.append(weights[layer_index].reshape(-1))
        parts.append(biases[layer_index].reshape(-1))
    parts.append(weights[5].reshape(-1))
    parts.append(biases[5].reshape(-1))
    return torch.cat(parts)


def pareto_dominates_functional(
    control: Mapping[str, float],
    matched: Mapping[str, float],
    *,
    epsilon: float = 1e-8,
) -> bool:
    weakly_better = (
        float(control["target_margin"]) >= float(matched["target_margin"])
        and float(control["compatible_source_output_mse"])
        <= float(matched["compatible_source_output_mse"])
    )
    strictly_better = (
        float(control["target_margin"]) > float(matched["target_margin"]) + epsilon
        or float(control["compatible_source_output_mse"])
        < float(matched["compatible_source_output_mse"]) - epsilon
    )
    return weakly_better and strictly_better


def select_shuffled_target(*, subject_id: str, source: str, target: str) -> str:
    candidates = sorted(pattern for pattern in PATTERNS if pattern not in {source, target})
    key = stable_hash_json([
        subject_id,
        source,
        target,
        "functional_weight_editing_v13_shuffled_signature_target",
    ])
    return candidates[int(key[:16], 16) % len(candidates)]


def random_weight_delta_controls(
    *,
    subject_id: str,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    matched_delta_norm: torch.Tensor,
    random_controls: int,
) -> list[Dict[str, Any]]:
    key = stable_hash_json([
        subject_id,
        source,
        target,
        "functional_weight_editing_v13_random_weight_delta",
    ])
    seed = int(key[:16], 16) % (2**31)
    generator = torch.Generator().manual_seed(seed)
    controls = []
    scale = matched_delta_norm.clamp_min(1e-12)
    for index in range(random_controls):
        delta = torch.randn(
            source_weights.shape,
            generator=generator,
            dtype=source_weights.dtype,
        )
        delta = scale * delta / delta.norm().clamp_min(1e-12)
        weights = source_weights + delta
        controls.append(control_record(
            f"random_norm_matched_weight_delta:{index:02d}",
            weights,
            source,
            target,
            source_weights,
            {
                "delta_norm": float(delta.norm().item()),
                "random_index": int(index),
                "random_seed": int(seed),
            },
        ))
    return controls


def individual_passed(
    *,
    matched: Mapping[str, Any],
    full_retrieval: Mapping[str, Any],
) -> bool:
    return (
        matched["target_prediction_pass"]
        and matched["target_margin"] > THRESHOLDS["min_per_record_matched_target_margin"]
        and matched["compatible_source_output_mse"]
        < full_retrieval["compatible_source_output_mse"]
        and matched["conflict_target_accuracy"]
        >= THRESHOLDS["min_per_record_conflict_target_accuracy"]
        and matched["conflict_target_accuracy_improvement"]
        >= THRESHOLDS["min_per_record_conflict_target_accuracy_improvement"]
        and matched["pareto_undominated"]
        and (
            matched["matched_minus_no_signature_target_margin"]
            >= THRESHOLDS["min_per_record_no_signature_target_margin_advantage"]
            or matched["no_signature_minus_matched_compatible_source_output_mse"]
            >= THRESHOLDS["min_per_record_no_signature_compatible_mse_advantage"]
        )
        and (
            matched["matched_minus_shuffled_signature_target_margin"]
            >= THRESHOLDS["min_per_record_shuffled_target_margin_advantage"]
            or matched["shuffled_signature_minus_matched_compatible_source_output_mse"]
            >= THRESHOLDS["min_per_record_shuffled_compatible_mse_advantage"]
        )
    )


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {
            "individual_all_gate_pass_count": 0,
            "individual_all_gate_pass_rate": 0.0,
            "n": 0,
            "pareto_undominated_count": 0,
            "pareto_undominated_rate": 0.0,
            "target_prediction_count": 0,
            "target_prediction_rate": 0.0,
        }
    passed = sum(1 for record in records if record["individual_all_gates_passed"])
    target_pred = sum(1 for record in records if record["summary"]["target_prediction_pass"])
    pareto = sum(1 for record in records if record["summary"]["pareto_undominated"])
    return {
        "individual_all_gate_pass_count": int(passed),
        "individual_all_gate_pass_rate": float(passed / n),
        "mean_conflict_target_accuracy": v10.mean(
            record["matched"]["conflict_target_accuracy"] for record in records
        ),
        "mean_conflict_target_accuracy_improvement": v10.mean(
            record["matched"]["conflict_target_accuracy_improvement"] for record in records
        ),
        "mean_full_retrieval_minus_matched_compatible_source_output_mse": v10.mean(
            record["summary"]["full_retrieval_minus_matched_compatible_source_output_mse"]
            for record in records
        ),
        "mean_matched_minus_best_control_target_margin": v10.mean(
            record["summary"]["matched_minus_best_control_target_margin"] for record in records
        ),
        "mean_matched_minus_full_retrieval_target_margin": v10.mean(
            record["summary"]["matched_minus_full_retrieval_target_margin"] for record in records
        ),
        "mean_matched_minus_no_signature_target_margin": v10.mean(
            record["summary"]["matched_minus_no_signature_target_margin"] for record in records
        ),
        "mean_matched_minus_shuffled_signature_target_margin": v10.mean(
            record["summary"]["matched_minus_shuffled_signature_target_margin"]
            for record in records
        ),
        "mean_matched_target_margin": v10.mean(
            record["matched"]["target_margin"] for record in records
        ),
        "mean_no_signature_minus_matched_compatible_source_output_mse": v10.mean(
            record["summary"]["no_signature_minus_matched_compatible_source_output_mse"]
            for record in records
        ),
        "mean_shuffled_signature_minus_matched_compatible_source_output_mse": v10.mean(
            record["summary"]["shuffled_signature_minus_matched_compatible_source_output_mse"]
            for record in records
        ),
        "n": int(n),
        "pareto_undominated_count": int(pareto),
        "pareto_undominated_rate": float(pareto / n),
        "target_prediction_count": int(target_pred),
        "target_prediction_rate": float(target_pred / n),
    }


def gate_failures(
    *,
    aggregate: Mapping[str, Any],
    by_direction: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    failures = []
    v10.require_equal(failures, aggregate["n"], THRESHOLDS["expected_record_count"], "aggregate n")
    v10.require_at_least(
        failures,
        aggregate["individual_all_gate_pass_rate"],
        THRESHOLDS["min_aggregate_individual_pass_rate"],
        "aggregate individual pass rate",
    )
    v10.require_at_least(
        failures,
        aggregate["target_prediction_rate"],
        THRESHOLDS["min_aggregate_target_prediction_rate"],
        "aggregate target prediction rate",
    )
    v10.require_at_least(
        failures,
        aggregate["pareto_undominated_rate"],
        THRESHOLDS["min_aggregate_pareto_undominated_rate"],
        "aggregate Pareto-undominated rate",
    )
    v10.require_greater(
        failures,
        aggregate["mean_matched_target_margin"],
        THRESHOLDS["min_mean_matched_target_margin"],
        "mean matched target margin",
    )
    v10.require_greater(
        failures,
        aggregate["mean_full_retrieval_minus_matched_compatible_source_output_mse"],
        THRESHOLDS["min_mean_full_retrieval_minus_matched_compatible_source_output_mse"],
        "mean full-retrieval-minus-matched compatible source-output MSE",
    )
    v10.require_at_least(
        failures,
        aggregate["mean_conflict_target_accuracy"],
        THRESHOLDS["min_aggregate_conflict_target_accuracy"],
        "aggregate conflict target accuracy",
    )
    v10.require_at_least(
        failures,
        aggregate["mean_conflict_target_accuracy_improvement"],
        THRESHOLDS["min_aggregate_conflict_target_accuracy_improvement"],
        "aggregate conflict target accuracy improvement",
    )
    if not (
        aggregate["mean_matched_minus_no_signature_target_margin"]
        > THRESHOLDS["min_aggregate_signature_target_margin_advantage"]
        or aggregate["mean_no_signature_minus_matched_compatible_source_output_mse"]
        > THRESHOLDS["min_aggregate_signature_compatible_mse_advantage"]
    ):
        failures.append("aggregate no-signature advantage gate failed")
    if not (
        aggregate["mean_matched_minus_shuffled_signature_target_margin"]
        > THRESHOLDS["min_aggregate_shuffled_target_margin_advantage"]
        or aggregate["mean_shuffled_signature_minus_matched_compatible_source_output_mse"]
        > THRESHOLDS["min_aggregate_shuffled_compatible_mse_advantage"]
    ):
        failures.append("aggregate shuffled-signature advantage gate failed")
    for direction, summary in by_direction.items():
        v10.require_equal(
            failures,
            summary["n"],
            THRESHOLDS["expected_per_direction_count"],
            f"{direction} n",
        )
        v10.require_at_least(
            failures,
            summary["individual_all_gate_pass_rate"],
            THRESHOLDS["min_direction_individual_pass_rate"],
            f"{direction} individual pass rate",
        )
        v10.require_at_least(
            failures,
            summary["pareto_undominated_rate"],
            THRESHOLDS["min_direction_pareto_undominated_rate"],
            f"{direction} Pareto-undominated rate",
        )
        v10.require_greater(
            failures,
            summary["mean_full_retrieval_minus_matched_compatible_source_output_mse"],
            THRESHOLDS["min_direction_mean_full_retrieval_minus_matched_compatible_source_output_mse"],
            f"{direction} full-retrieval-minus-matched compatible source-output MSE",
        )
        v10.require_at_least(
            failures,
            summary["target_prediction_rate"],
            THRESHOLDS["min_direction_target_prediction_rate"],
            f"{direction} target prediction rate",
        )
        v10.require_greater(
            failures,
            summary["mean_matched_target_margin"],
            THRESHOLDS["min_direction_mean_target_margin"],
            f"{direction} target margin",
        )
        v10.require_at_least(
            failures,
            summary["mean_conflict_target_accuracy"],
            THRESHOLDS["min_direction_conflict_target_accuracy"],
            f"{direction} conflict target accuracy",
        )
        v10.require_at_least(
            failures,
            summary["mean_conflict_target_accuracy_improvement"],
            THRESHOLDS["min_direction_conflict_target_accuracy_improvement"],
            f"{direction} conflict target accuracy improvement",
        )
    for record in records:
        direction = v1.vector_key(record["source_behavior"], record["target_behavior"])
        expected_split = EXPECTED_SPLIT_COUNTS[direction]
        if record["matched"]["compatible_count"] != expected_split["compatible"]:
            failures.append(f"{record['subject_id']} compatible count mismatch")
        if record["matched"]["conflict_count"] != expected_split["conflict"]:
            failures.append(f"{record['subject_id']} conflict count mismatch")
        expected_support_split = EXPECTED_SUPPORT_SPLIT_COUNTS[direction]
        support_split = source_target_support_split(
            source=record["source_behavior"],
            target=record["target_behavior"],
        )
        if support_split["compatible_count"] != expected_support_split["compatible"]:
            failures.append(f"{record['subject_id']} support compatible count mismatch")
        if support_split["conflict_count"] != expected_support_split["conflict"]:
            failures.append(f"{record['subject_id']} support conflict count mismatch")
        if len(record["controls"]) != THRESHOLDS["expected_controls_per_record"]:
            failures.append(f"{record['subject_id']} control count mismatch")
        if record["random_control_count"] != THRESHOLDS["random_controls_per_record"]:
            failures.append(f"{record['subject_id']} random control count mismatch")
        required_control_types = {
            "aligned_full_nearest_target_retrieval",
            "aligned_interpolation_alpha_0.975",
            "no_edit",
            "no_signature_optimizer",
            "source_signature_optimizer",
            "signature_only_optimizer",
            "target_only_support_optimizer",
        }
        control_types = {control["control_type"] for control in record["controls"]}
        for control_type in required_control_types:
            if control_type not in control_types:
                failures.append(f"{record['subject_id']} missing {control_type} control")
        if not any(
            control["control_type"].startswith("shuffled_signature_optimizer:")
            for control in record["controls"]
        ):
            failures.append(f"{record['subject_id']} missing shuffled-signature control")
    return failures


def train_only_statistics_hash(stats: Mapping[str, Any]) -> str:
    return stable_hash_json({
        "centroids": {
            pattern: v10.tensor_to_hashable(stats["centroids"][pattern])
            for pattern in PATTERNS
        },
        "global_weight_centroid": v10.tensor_to_hashable(stats["global_weight_centroid"]),
        "probe_examples_hash": stats.get("probe_examples_hash"),
        "sig_mean": v10.tensor_to_hashable(stats["sig_mean"]),
        "sig_std": v10.tensor_to_hashable(stats["sig_std"]),
    })


if __name__ == "__main__":
    main()
