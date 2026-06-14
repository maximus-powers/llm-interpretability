"""Four-behavior contrastive residual representation steering under V5 preregistration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ZOO_ROOT = REPO_ROOT / "model_zoo"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_ZOO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import generate_four_behavior_decoder_source_pools as poolgen  # noqa: E402
import train_four_behavior_representation_steering as v1  # noqa: E402
from evaluate_four_behavior_source_generation_feasibility import (  # noqa: E402
    PATTERNS,
    build_candidate_pools,
    build_heldout_sequences,
    build_suite,
    summarize_candidate_pools,
)
from hypernet.dataset_provenance import stable_hash_json  # noqa: E402
from hypernet.paired_contrast import build_digit_probe_examples  # noqa: E402


SEED_BEHAVIOR_STRIDE = 100000
POOL_CONFIGS = {
    "train": {
        "base_seed": 42300000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 43300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 44300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}
DEFAULT_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_representation_steering_v5_pools"
TRAINING_CONFIG = {
    "calibration_epochs": 500,
    "calibration_lr": 0.03,
    "calibration_seed": 20260801,
    "calibration_weight_decay": 0.001,
    "centroid_distance_loss_weight": 0.02,
    "classifier_epochs": 1000,
    "classifier_lr": 0.10,
    "classifier_seed": 20260731,
    "classifier_weight_decay": 0.0001,
    "coefficient_norm_loss_weight": 0.001,
    "covariance_eigenvalue_floor": 1e-4,
    "covariance_shrinkage_behavior_weight": 0.75,
    "covariance_shrinkage_global_weight": 0.25,
    "diagonal_ratio_clip_max": 4.0,
    "diagonal_ratio_clip_min": 0.25,
    "displacement_norm_cap": 200.0,
    "orthogonal_residual_carry_weight": 0.0,
    "pca_rank": 48,
    "primary_rank_margin": 0.50,
    "primary_rank_loss_weight": 1.0,
    "source_suppression_loss_margin": 0.05,
    "source_suppression_loss_weight": 1.0,
    "target_centroid_rank_margin": 0.05,
    "target_centroid_rank_loss_weight": 1.0,
    "transport_method": "train_contrastive_residual_calibration_on_v4_low_rank_transport",
    "variance_clamp_min": 1e-4,
}
PREREGISTERED_ARG_VALUES = {
    "generic_negative_cap": 1024,
    "hard_negative_cap": 1024,
    "heldout_per_class": 64,
    "lr": 0.003,
    "positive_cap": 2048,
    "random_controls": 32,
    "source_margin_gate": 0.40,
    "support_per_class": 160,
    "train_epochs": 350,
}
THRESHOLDS = {
    "expected_record_count": 288,
    "expected_per_target_count": 72,
    "expected_per_direction_count": 24,
    "max_mean_source_primary_margin_change": -0.05,
    "min_aggregate_individual_pass_rate": 0.90,
    "min_direction_individual_pass_rate": 0.90,
    "min_mean_matched_centroid_improvement": 0.15,
    "min_mean_matched_minus_best_control_centroid_improvement": 0.10,
    "min_mean_matched_minus_best_control_primary_target_margin": 0.10,
    "min_mean_matched_minus_v2_centroid_delta_centroid_improvement": 0.10,
    "min_mean_matched_minus_v2_centroid_delta_primary_target_margin": 0.10,
    "min_mean_matched_minus_v3_diagonal_transport_centroid_improvement": 0.10,
    "min_mean_matched_minus_v3_diagonal_transport_primary_target_margin": 0.10,
    "min_mean_matched_minus_v4_low_rank_centroid_improvement": 0.05,
    "min_mean_matched_minus_v4_low_rank_primary_target_margin": 0.10,
    "min_mean_matched_primary_target_margin": 0.25,
    "min_per_record_centroid_improvement": 0.15,
    "min_per_record_matched_minus_best_control_centroid_improvement": 0.0,
    "min_per_record_matched_minus_best_control_primary_target_margin": 0.0,
    "min_per_record_matched_minus_v2_centroid_delta_centroid_improvement": 0.0,
    "min_per_record_matched_minus_v2_centroid_delta_primary_target_margin": 0.0,
    "min_per_record_matched_minus_v3_diagonal_transport_centroid_improvement": 0.0,
    "min_per_record_matched_minus_v3_diagonal_transport_primary_target_margin": 0.0,
    "min_per_record_matched_minus_v4_low_rank_centroid_improvement": 0.0,
    "min_per_record_matched_minus_v4_low_rank_primary_target_margin": 0.0,
    "min_per_record_primary_target_margin": 0.25,
    "min_per_target_individual_pass_rate": 0.80,
    "per_record_source_primary_margin_change_must_be_below": -0.05,
}
DECODER_FINAL_RAW = (
    REPO_ROOT / "runs" / "four_behavior_decoder_source_pools_v2" / "final_subjects.json"
)
V1_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_representation_steering_v1_pools"
V2_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_representation_steering_v2_pools"
V3_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_representation_steering_v3_pools"
V4_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_representation_steering_v4_pools"
V5_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"


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
        default="runs/four_behavior_representation_steering_v5_contrastive_residual_calibration",
    )
    parser.add_argument("--train-epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--source-margin-gate", type=float, default=0.40)
    parser.add_argument("--support-per-class", type=int, default=160)
    parser.add_argument("--heldout-per-class", type=int, default=64)
    parser.add_argument("--positive-cap", type=int, default=2048)
    parser.add_argument("--hard-negative-cap", type=int, default=1024)
    parser.add_argument("--generic-negative-cap", type=int, default=1024)
    parser.add_argument("--random-controls", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_preregistered_args(args)
    pool_dir = REPO_ROOT / args.pool_dir
    assert_preregistered_pool_dir(pool_dir)
    output_dir = REPO_ROOT / args.output_dir
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
        raise ValueError("Non-preregistered V5 parameter override: " + "; ".join(failures))


def assert_preregistered_pool_dir(pool_dir: Path) -> None:
    resolved = pool_dir.resolve()
    if resolved != DEFAULT_POOL_DIR.resolve():
        raise ValueError(f"V5 requires preregistered pool directory: {DEFAULT_POOL_DIR}")
    for forbidden in (
        V1_POOL_DIR.resolve(),
        V2_POOL_DIR.resolve(),
        V3_POOL_DIR.resolve(),
        V4_POOL_DIR.resolve(),
    ):
        if resolved == forbidden or forbidden in resolved.parents:
            raise ValueError(f"V5 may not use prior steering source pools: {resolved}")


def assert_no_forbidden_final_raw_paths(
    paths: Iterable[Path | str],
    *,
    allow_v5_final: bool,
) -> None:
    forbidden = {
        DECODER_FINAL_RAW.resolve(),
        (V1_POOL_DIR / "final_subjects.json").resolve(),
        (V2_POOL_DIR / "final_subjects.json").resolve(),
        (V3_POOL_DIR / "final_subjects.json").resolve(),
        (V4_POOL_DIR / "final_subjects.json").resolve(),
    }
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in forbidden:
            raise ValueError(f"sealed final raw path is forbidden: {path}")
        if path.name != "final_subjects.json":
            continue
        if allow_v5_final and path == V5_FINAL_RAW.resolve():
            continue
        raise ValueError(f"V5 steering final raw path is forbidden before final eval: {path}")


def build_seed_preflight(
    configs: Mapping[str, Mapping[str, int]],
    *,
    behavior_stride: int,
) -> Dict[str, Any]:
    return v1.build_seed_preflight(configs, behavior_stride=behavior_stride)


def generate_pools(args: argparse.Namespace, pool_dir: Path) -> Dict[str, Any]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    seed_preflight = build_seed_preflight(
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
    pool_payloads = {}
    pool_summaries = {}
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
        payload["claim_scope"] = "four_behavior_representation_steering_v5_source_pool"
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
    final_redacted["claim_scope"] = "redacted_final_steering_v5_source_pool_audit_surface_only"
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
    audit["claim_scope"] = "four_behavior_representation_steering_v5_source_pool_construction"
    audit = redact_v5_combined_audit(audit)
    audit["audit_payload_sha256"] = stable_hash_json(audit)
    (pool_dir / "combined_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    return {
        "combined_audit_path": v1.rel(pool_dir / "combined_audit.json"),
        "final_redacted_audit_path": v1.rel(pool_dir / "final_redacted_audit.json"),
        "passed": bool(audit["passed"]),
        "pool_summaries": audit["pool_summaries"],
        "seed_preflight": seed_preflight,
    }


def redact_v5_combined_audit(audit: Mapping[str, Any]) -> Dict[str, Any]:
    redacted = json.loads(json.dumps(audit))
    final_summary = redacted["pool_summaries"]["final"]
    redacted["pool_summaries"]["final"] = {
        "accepted_counts_by_behavior": final_summary["accepted_counts_by_behavior"],
        "pool_file_sha256": final_summary["pool_file_sha256"],
        "pool_redacted_payload_sha256": final_summary["pool_redacted_payload_sha256"],
    }
    return redacted


def development_input_paths(pool_dir: Path) -> Dict[str, Path]:
    return {
        "combined_audit": pool_dir / "combined_audit.json",
        "development": pool_dir / "development_subjects.json",
        "final_redacted_audit": pool_dir / "final_redacted_audit.json",
        "train": pool_dir / "train_subjects.json",
    }


def run_development(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> Dict[str, Any]:
    paths = development_input_paths(pool_dir)
    assert_no_forbidden_final_raw_paths(paths.values(), allow_v5_final=False)
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
    final_authorization_failures = validate_development_authorizes_final(
        development=development,
        pool_dir=pool_dir,
    )
    if final_authorization_failures:
        raise RuntimeError(
            "development artifact does not authorize final raw evaluation: "
            + "; ".join(final_authorization_failures)
        )
    eval_path = pool_dir / "final_subjects.json"
    assert_no_forbidden_final_raw_paths([eval_path], allow_v5_final=True)
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
        "claim_scope": "four_behavior_representation_steering_v5_contrastive_residual_calibration_development",
        "combined_audit_sha256": v1.sha256_file(pool_dir / "combined_audit.json"),
        "eval_pool_sha256": v1.sha256_file(pool_dir / "development_subjects.json"),
        "final_redacted_audit_sha256": v1.sha256_file(pool_dir / "final_redacted_audit.json"),
        "next_action": "eligible_for_one_shot_final_eval_without_method_changes",
        "phase": "development",
        "train_pool_sha256": v1.sha256_file(pool_dir / "train_subjects.json"),
        "transport_method": "train_contrastive_residual_calibration_on_v4_low_rank_transport",
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
        allow_v5_final=allow_eval_final_raw,
    )
    train_payload = v1.load_json(train_path)
    eval_payload = v1.load_json(eval_path)
    combined_audit = v1.load_json(combined_audit_path)
    final_redacted = v1.load_json(final_redacted_path)
    hash_failures = validate_source_pool_file_hashes(
        train_path=train_path,
        eval_path=eval_path,
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase=phase,
    )
    if hash_failures:
        raise ValueError("V5 source-pool file hash validation failed: " + "; ".join(hash_failures))
    contract_failures = validate_source_pool_contract(
        train_payload=train_payload,
        eval_payload=eval_payload,
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase=phase,
    )
    if contract_failures:
        raise ValueError("V5 source-pool contract validation failed: " + "; ".join(contract_failures))
    train_subjects = v1.accepted_records(train_payload)
    eval_subjects = v1.accepted_records(eval_payload)
    train_stats = fit_train_statistics(train_subjects)
    classifier, classifier_summary = fit_primary_classifier(
        train_stats["z_train"],
        train_stats["y_train"],
    )
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    calibration_coefficients, calibration_summary = fit_contrastive_calibration(
        subjects=train_subjects,
        train_stats=train_stats,
        classifier=classifier,
    )
    train_stats["calibration_coefficients"] = calibration_coefficients
    transport_path = output_dir / "contrastive_residual_calibration_stats.pt"
    torch.save(
        {
            "calibration_coefficients": train_stats["calibration_coefficients"],
            "centroids": train_stats["centroids"],
            "inv_sqrt_cov": train_stats["inv_sqrt_cov"],
            "method": "train_contrastive_residual_calibration_on_v4_low_rank_transport",
            "pca_components": train_stats["pca_components"],
            "sqrt_cov": train_stats["sqrt_cov"],
            "summary": calibration_summary,
            "transport_stds": train_stats["transport_stds"],
        },
        transport_path,
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
        "calibration_coefficient_hash": calibration_coefficient_hash(
            train_stats["calibration_coefficients"]
        ),
        "classifier_summary": classifier_summary,
        "claim_scope": (
            "four_behavior_representation_steering_v5_contrastive_residual_calibration_development"
            if phase == "development"
            else "four_behavior_representation_steering_v5_contrastive_residual_calibration_final"
        ),
        "combined_audit_path": v1.rel(combined_audit_path),
        "combined_audit_sha256": v1.sha256_file(combined_audit_path),
        "dirty_worktree_caveat": True,
        "eval_pool_path": v1.rel(eval_path),
        "eval_pool_sha256": v1.sha256_file(eval_path),
        "final_redacted_audit_path": v1.rel(final_redacted_path),
        "final_redacted_audit_sha256": v1.sha256_file(final_redacted_path),
        "final_redacted_summary": final_redacted,
        "forbidden_decoder_final_raw_opened": False,
        "forbidden_v1_steering_final_raw_opened": False,
        "forbidden_v2_steering_final_raw_opened": False,
        "forbidden_v3_steering_final_raw_opened": False,
        "forbidden_v4_steering_final_raw_opened": False,
        "forbidden_v5_steering_final_raw_opened": False,
        "phase": phase,
        "pool_overlap_counts": combined_audit.get("overlap_counts", {}),
        "probe_examples_hash": combined_audit.get("probe_examples_hash"),
        "source_pool_audit_passed": bool(combined_audit.get("passed", False)),
        "thresholds": THRESHOLDS,
        "train_only_statistics_hash": train_only_statistics_hash(train_stats),
        "train_pool_path": v1.rel(train_path),
        "train_pool_sha256": v1.sha256_file(train_path),
        "training_config": TRAINING_CONFIG,
        "transport_method": "train_contrastive_residual_calibration_on_v4_low_rank_transport",
        "transport_stats_path": v1.rel(transport_path),
        "vector_training_summary": calibration_summary,
    }
    result["failures"] = failures
    result["passed"] = not failures
    return result


def fit_train_statistics(subjects: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    stats = v1.fit_train_statistics(subjects)
    z_train = stats["z_train"]
    y_train = stats["y_train"]
    rank = int(TRAINING_CONFIG["pca_rank"])
    centered = z_train - z_train.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    if vh.size(0) < rank:
        raise ValueError(f"V5 requires at least {rank} principal components")
    pca_components = vh[:rank].T.contiguous()

    residuals = []
    projected_by_behavior = {}
    for pattern_index, pattern in enumerate(PATTERNS):
        behavior_z = z_train[y_train == pattern_index]
        behavior_residual = behavior_z - stats["centroids"][pattern]
        projected = behavior_residual @ pca_components
        projected_by_behavior[pattern] = projected
        residuals.append(projected)
    global_projected = torch.cat(residuals, dim=0)
    global_cov = covariance(global_projected)

    sqrt_cov = {}
    inv_sqrt_cov = {}
    for pattern in PATTERNS:
        behavior_cov = covariance(projected_by_behavior[pattern])
        shrunk = (
            TRAINING_CONFIG["covariance_shrinkage_behavior_weight"] * behavior_cov
            + TRAINING_CONFIG["covariance_shrinkage_global_weight"] * global_cov
        )
        sqrt_cov[pattern], inv_sqrt_cov[pattern] = covariance_factors(shrunk)

    global_var = z_train.var(dim=0, unbiased=False)
    transport_stds = {}
    for pattern_index, pattern in enumerate(PATTERNS):
        behavior_z = z_train[y_train == pattern_index]
        behavior_var = behavior_z.var(dim=0, unbiased=False)
        shrunk_var = (
            TRAINING_CONFIG["covariance_shrinkage_behavior_weight"] * behavior_var
            + TRAINING_CONFIG["covariance_shrinkage_global_weight"] * global_var
        )
        shrunk_var = shrunk_var.clamp_min(TRAINING_CONFIG["variance_clamp_min"])
        transport_stds[pattern] = torch.sqrt(shrunk_var)

    stats["global_residual_covariance"] = global_cov
    stats["inv_sqrt_cov"] = inv_sqrt_cov
    stats["pca_components"] = pca_components
    stats["sqrt_cov"] = sqrt_cov
    stats["transport_stds"] = transport_stds
    return stats


def covariance(projected: torch.Tensor) -> torch.Tensor:
    return projected.T @ projected / projected.size(0)


def covariance_factors(covariance_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
    eigenvalues = eigenvalues.clamp_min(TRAINING_CONFIG["covariance_eigenvalue_floor"])
    sqrt_diag = torch.diag(torch.sqrt(eigenvalues))
    inv_sqrt_diag = torch.diag(torch.rsqrt(eigenvalues))
    return eigenvectors @ sqrt_diag @ eigenvectors.T, eigenvectors @ inv_sqrt_diag @ eigenvectors.T


def fit_primary_classifier(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
) -> tuple[v1.LinearSignatureEvaluator, Dict[str, Any]]:
    torch.manual_seed(TRAINING_CONFIG["classifier_seed"])
    model = v1.LinearSignatureEvaluator(z_train.size(1), len(PATTERNS))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["classifier_lr"],
        weight_decay=TRAINING_CONFIG["classifier_weight_decay"],
    )
    history = []
    for epoch in range(1, TRAINING_CONFIG["classifier_epochs"] + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(z_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 100 == 0 or epoch == TRAINING_CONFIG["classifier_epochs"]:
            pred = logits.argmax(dim=1)
            history.append({
                "accuracy": float((pred == y_train).float().mean().item()),
                "epoch": epoch,
                "loss": float(loss.item()),
            })
    model.eval()
    with torch.no_grad():
        final_logits = model(z_train)
        final_pred = final_logits.argmax(dim=1)
    return model, {
        "final_train_accuracy": float((final_pred == y_train).float().mean().item()),
        "history": history,
    }


def fit_contrastive_calibration(
    *,
    subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    classifier: v1.LinearSignatureEvaluator,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    torch.manual_seed(TRAINING_CONFIG["calibration_seed"])
    parameters = {
        v1.vector_key(source, target): torch.nn.Parameter(torch.zeros(
            int(TRAINING_CONFIG["pca_rank"]),
            dtype=torch.float32,
        ))
        for source in PATTERNS
        for target in PATTERNS
        if source != target
    }
    optimizer = torch.optim.AdamW(
        list(parameters.values()),
        lr=TRAINING_CONFIG["calibration_lr"],
        weight_decay=TRAINING_CONFIG["calibration_weight_decay"],
    )
    z_by_subject = {}
    for subject in subjects:
        signature = torch.tensor(subject["signature"], dtype=torch.float32)
        z_by_subject[str(subject["subject_id"])] = (
            signature - train_stats["sig_mean"]
        ) / train_stats["sig_std"]
    full_stats = dict(train_stats)
    full_stats["calibration_coefficients"] = parameters
    history = []
    final_loss = None
    for epoch in range(1, int(TRAINING_CONFIG["calibration_epochs"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for subject in subjects:
            source = v1.behavior_of(subject)
            subject_id = str(subject["subject_id"])
            z = z_by_subject[subject_id]
            for target in PATTERNS:
                if target == source:
                    continue
                losses.append(compute_train_example_loss(
                    classifier=classifier,
                    source=source,
                    stats=full_stats,
                    subject_id=subject_id,
                    target=target,
                    z=z,
                ))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if epoch == 1 or epoch % 50 == 0 or epoch == TRAINING_CONFIG["calibration_epochs"]:
            history.append({"epoch": epoch, "loss": final_loss})
    coefficients = {
        key: parameter.detach().clone()
        for key, parameter in parameters.items()
    }
    return coefficients, {
        "best_epoch": None,
        "best_train_objective": final_loss,
        "final_epoch": TRAINING_CONFIG["calibration_epochs"],
        "history_tail": history[-10:],
    }


def compute_train_example_loss(
    *,
    classifier: torch.nn.Module,
    source: str,
    stats: Mapping[str, Any],
    subject_id: str,
    target: str,
    z: torch.Tensor,
) -> torch.Tensor:
    matched = apply_v5_transport(z=z, source=source, target=target, stats=stats)
    logits = classifier(matched)
    target_index = torch.tensor(PATTERNS.index(target), dtype=torch.long, device=logits.device)
    target_ce = F.cross_entropy(logits.unsqueeze(0), target_index.unsqueeze(0))
    source_loss = TRAINING_CONFIG["source_suppression_loss_weight"] * F.relu(
        primary_margin(logits, PATTERNS.index(source))
        + TRAINING_CONFIG["source_suppression_loss_margin"]
    )
    centroid_loss = (
        TRAINING_CONFIG["centroid_distance_loss_weight"]
        * (matched - stats["centroids"][target]).pow(2).sum()
    )
    coefficient = stats["calibration_coefficients"][v1.vector_key(source, target)]
    coeff_loss = TRAINING_CONFIG["coefficient_norm_loss_weight"] * coefficient.pow(2).sum()
    matched_primary = primary_margin(logits, PATTERNS.index(target))
    matched_centroid = centroid_improvement(z=z, candidate_z=matched, target=target, stats=stats)
    primary_rank_losses = []
    centroid_rank_losses = []
    for control in build_train_control_candidate_specs(
        subject_id=subject_id,
        z=z,
        source=source,
        target=target,
        stats=stats,
    ):
        control_z = control["candidate_z"].detach()
        control_logits = classifier(control_z).detach()
        control_primary = primary_margin(control_logits, PATTERNS.index(target)).detach()
        control_centroid = centroid_improvement(
            z=z,
            candidate_z=control_z,
            target=target,
            stats=stats,
        ).detach()
        primary_rank_losses.append(F.relu(
            TRAINING_CONFIG["primary_rank_margin"] - (matched_primary - control_primary)
        ))
        centroid_rank_losses.append(F.relu(
            TRAINING_CONFIG["target_centroid_rank_margin"]
            - (matched_centroid - control_centroid)
        ))
    primary_rank_loss = (
        TRAINING_CONFIG["primary_rank_loss_weight"] * torch.stack(primary_rank_losses).mean()
    )
    centroid_rank_loss = (
        TRAINING_CONFIG["target_centroid_rank_loss_weight"]
        * torch.stack(centroid_rank_losses).mean()
    )
    return target_ce + source_loss + centroid_loss + primary_rank_loss + centroid_rank_loss + coeff_loss


def primary_margin(logits: torch.Tensor, index: int) -> torch.Tensor:
    mask = torch.ones(logits.numel(), dtype=torch.bool, device=logits.device)
    mask[index] = False
    return logits[index] - logits[mask].max()


def centroid_improvement(
    *,
    z: torch.Tensor,
    candidate_z: torch.Tensor,
    target: str,
    stats: Mapping[str, Any],
) -> torch.Tensor:
    target_centroid = stats["centroids"][target]
    return (z - target_centroid).norm() - (candidate_z - target_centroid).norm()


def apply_low_rank_residual_transport(
    *,
    z: torch.Tensor,
    source: str,
    target: str,
    stats: Mapping[str, Any],
) -> torch.Tensor:
    transported = apply_uncapped_low_rank_residual_transport(
        z=z,
        source=source,
        target=target,
        stats=stats,
    )
    return cap_total_displacement(z, transported)


def apply_uncapped_low_rank_residual_transport(
    *,
    z: torch.Tensor,
    source: str,
    target: str,
    stats: Mapping[str, Any],
) -> torch.Tensor:
    centroids = stats["centroids"]
    components = stats["pca_components"]
    residual = z - centroids[source]
    projected = components.T @ residual
    transported_projected = stats["sqrt_cov"][target] @ stats["inv_sqrt_cov"][source] @ projected
    transported = centroids[target] + components @ transported_projected
    if TRAINING_CONFIG["orthogonal_residual_carry_weight"] != 0.0:
        in_subspace = components @ projected
        transported = (
            transported
            + TRAINING_CONFIG["orthogonal_residual_carry_weight"] * (residual - in_subspace)
        )
    return transported


def cap_total_displacement(z: torch.Tensor, transported: torch.Tensor) -> torch.Tensor:
    displacement = transported - z
    norm = displacement.norm()
    cap = torch.tensor(
        float(TRAINING_CONFIG["displacement_norm_cap"]),
        dtype=z.dtype,
        device=z.device,
    )
    if bool(norm > cap):
        transported = z + cap * displacement / norm.clamp_min(1e-12)
    return transported


def apply_v5_transport(
    *,
    z: torch.Tensor,
    source: str,
    target: str,
    stats: Mapping[str, Any],
) -> torch.Tensor:
    base = apply_uncapped_low_rank_residual_transport(
        z=z,
        source=source,
        target=target,
        stats=stats,
    )
    coefficient = stats["calibration_coefficients"][v1.vector_key(source, target)]
    calibrated = base + stats["pca_components"] @ coefficient
    return cap_total_displacement(z, calibrated)


def apply_v2_centroid_delta(
    *,
    z: torch.Tensor,
    source: str,
    target: str,
    centroids: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return z + centroids[target] - centroids[source]


def apply_diagonal_transport(
    *,
    z: torch.Tensor,
    source: str,
    target: str,
    centroids: Mapping[str, torch.Tensor],
    stds: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    ratio = (stds[target] / stds[source]).clamp(
        min=TRAINING_CONFIG["diagonal_ratio_clip_min"],
        max=TRAINING_CONFIG["diagonal_ratio_clip_max"],
    )
    return centroids[target] + ratio * (z - centroids[source])


def evaluate_subjects(
    *,
    subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    classifier: v1.LinearSignatureEvaluator,
    random_controls: int,
) -> Dict[str, Any]:
    records = []
    for subject in subjects:
        source = v1.behavior_of(subject)
        signature = torch.tensor(subject["signature"], dtype=torch.float32)
        z = (signature - train_stats["sig_mean"]) / train_stats["sig_std"]
        for target in PATTERNS:
            if target == source:
                continue
            records.append(evaluate_record(
                subject=subject,
                z=z,
                source=source,
                target=target,
                classifier=classifier,
                train_stats=train_stats,
                random_controls=random_controls,
            ))
    aggregate = summarize_records(records)
    by_target = {
        target: summarize_records([record for record in records if record["target_behavior"] == target])
        for target in PATTERNS
    }
    by_direction = {
        v1.vector_key(source, target): summarize_records([
            record for record in records
            if record["source_behavior"] == source and record["target_behavior"] == target
        ])
        for source in PATTERNS
        for target in PATTERNS
        if source != target
    }
    individual_audit = individual_gate_audit(records)
    failures = gate_failures(
        aggregate=aggregate,
        by_target=by_target,
        by_direction=by_direction,
        individual_audit=individual_audit,
    )
    return {
        "aggregate": aggregate,
        "by_direction": by_direction,
        "by_target": by_target,
        "failures": failures,
        "individual_gate_audit": individual_audit,
        "passed": not failures,
        "records": records,
    }


def evaluate_record(
    *,
    subject: Mapping[str, Any],
    z: torch.Tensor,
    source: str,
    target: str,
    classifier: v1.LinearSignatureEvaluator,
    train_stats: Mapping[str, Any],
    random_controls: int,
) -> Dict[str, Any]:
    centroids = train_stats["centroids"]
    no_edit = v1.score_candidate(
        z=z,
        candidate_z=z,
        source=source,
        target=target,
        classifier=classifier,
        centroids=centroids,
    )
    matched_z = apply_v5_transport(
        z=z,
        source=source,
        target=target,
        stats=train_stats,
    )
    matched = v1.score_candidate(
        z=z,
        candidate_z=matched_z,
        source=source,
        target=target,
        classifier=classifier,
        centroids=centroids,
    )
    controls = build_controls(
        subject_id=str(subject["subject_id"]),
        z=z,
        source=source,
        target=target,
        classifier=classifier,
        train_stats=train_stats,
        random_controls=random_controls,
    )
    for control in controls:
        control["matched_minus_control_primary_target_margin"] = (
            matched["primary_target_margin"] - control["primary_target_margin"]
        )
        control["matched_minus_control_centroid_improvement"] = (
            matched["centroid_improvement"] - control["centroid_improvement"]
        )
    best_primary = max(controls, key=lambda item: item["primary_target_margin"])
    best_centroid = max(controls, key=lambda item: item["centroid_improvement"])
    v2_control = next(control for control in controls if control["control_type"] == "v2_centroid_delta")
    v3_control = next(control for control in controls if control["control_type"] == "v3_diagonal_transport")
    v4_control = next(control for control in controls if control["control_type"] == "v4_low_rank_residual_transport")
    summary = {
        "best_centroid_control_type": best_centroid["control_type"],
        "best_centroid_control_vector_key": best_centroid["control_transport_key"],
        "best_control_centroid_improvement": best_centroid["centroid_improvement"],
        "best_control_primary_target_margin": best_primary["primary_target_margin"],
        "best_primary_control_type": best_primary["control_type"],
        "best_primary_control_vector_key": best_primary["control_transport_key"],
        "matched_centroid_improvement": matched["centroid_improvement"],
        "matched_minus_best_control_centroid_improvement": (
            matched["centroid_improvement"] - best_centroid["centroid_improvement"]
        ),
        "matched_minus_best_control_primary_target_margin": (
            matched["primary_target_margin"] - best_primary["primary_target_margin"]
        ),
        "matched_minus_v2_centroid_delta_centroid_improvement": (
            matched["centroid_improvement"] - v2_control["centroid_improvement"]
        ),
        "matched_minus_v2_centroid_delta_primary_target_margin": (
            matched["primary_target_margin"] - v2_control["primary_target_margin"]
        ),
        "matched_minus_v3_diagonal_transport_centroid_improvement": (
            matched["centroid_improvement"] - v3_control["centroid_improvement"]
        ),
        "matched_minus_v3_diagonal_transport_primary_target_margin": (
            matched["primary_target_margin"] - v3_control["primary_target_margin"]
        ),
        "matched_minus_v4_low_rank_centroid_improvement": (
            matched["centroid_improvement"] - v4_control["centroid_improvement"]
        ),
        "matched_minus_v4_low_rank_primary_target_margin": (
            matched["primary_target_margin"] - v4_control["primary_target_margin"]
        ),
        "matched_primary_target_margin": matched["primary_target_margin"],
        "source_primary_margin_change": (
            matched["primary_source_margin"] - no_edit["primary_source_margin"]
        ),
    }
    passed = record_passes(summary, matched, target=target)
    return {
        "controls": controls,
        "individual_all_gates_passed": passed,
        "matched": matched,
        "random_control_count": int(random_controls),
        "source_behavior": source,
        "subject_id": str(subject["subject_id"]),
        "summary": summary,
        "target_behavior": target,
        "transport_displacement_norm": float((matched_z - z).norm().item()),
        "transport_key": v1.vector_key(source, target),
    }


def build_controls(
    *,
    subject_id: str,
    z: torch.Tensor,
    source: str,
    target: str,
    classifier: v1.LinearSignatureEvaluator,
    train_stats: Mapping[str, Any],
    random_controls: int,
) -> list[Dict[str, Any]]:
    controls = []
    for spec in build_control_candidate_specs(
        subject_id=subject_id,
        z=z,
        source=source,
        target=target,
        train_stats=train_stats,
        random_controls=random_controls,
    ):
        controls.append({
            "control_type": spec["control_type"],
            "control_transport_key": spec["control_transport_key"],
            **spec["metadata"],
            **v1.score_candidate(
                z=z,
                candidate_z=spec["candidate_z"],
                source=source,
                target=target,
                classifier=classifier,
                centroids=train_stats["centroids"],
            ),
        })
    return controls


def build_control_candidate_specs(
    *,
    subject_id: str,
    z: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    random_controls: int,
) -> list[Dict[str, Any]]:
    centroids = train_stats["centroids"]
    matched_z = apply_v5_transport(
        z=z,
        source=source,
        target=target,
        stats=train_stats,
    )
    specs: list[Dict[str, Any]] = [
        candidate_spec("no_edit", "no_edit", z),
        candidate_spec("null_vector", "null_vector", z),
        candidate_spec(
            "v2_centroid_delta",
            v1.vector_key(source, target),
            apply_v2_centroid_delta(
                z=z,
                source=source,
                target=target,
                centroids=centroids,
            ),
            {"control_source_behavior": source, "control_target_behavior": target},
        ),
        candidate_spec(
            "v3_diagonal_transport",
            v1.vector_key(source, target),
            apply_diagonal_transport(
                z=z,
                source=source,
                target=target,
                centroids=centroids,
                stds=train_stats["transport_stds"],
            ),
            {"control_source_behavior": source, "control_target_behavior": target},
        ),
        candidate_spec(
            "v4_low_rank_residual_transport",
            v1.vector_key(source, target),
            apply_low_rank_residual_transport(
                z=z,
                source=source,
                target=target,
                stats=train_stats,
            ),
            {"control_source_behavior": source, "control_target_behavior": target},
        ),
        candidate_spec(
            "reverse_v5_calibrated_transport",
            v1.vector_key(target, source),
            apply_v5_transport(
                z=z,
                source=target,
                target=source,
                stats=train_stats,
            ),
            {"control_source_behavior": target, "control_target_behavior": source},
        ),
    ]
    for other_target in PATTERNS:
        if other_target in (source, target):
            continue
        specs.append(candidate_spec(
            "same_source_other_target_v5_calibrated_transport",
            v1.vector_key(source, other_target),
            apply_v5_transport(
                z=z,
                source=source,
                target=other_target,
                stats=train_stats,
            ),
            {"control_source_behavior": source, "control_target_behavior": other_target},
        ))
    for other_source in PATTERNS:
        if other_source in (source, target):
            continue
        specs.append(candidate_spec(
            "same_target_other_source_v5_calibrated_transport",
            v1.vector_key(other_source, target),
            apply_v5_transport(
                z=z,
                source=other_source,
                target=target,
                stats=train_stats,
            ),
            {"control_source_behavior": other_source, "control_target_behavior": target},
        ))
    shuffled_source, shuffled_target = select_shuffled_direction(subject_id, source, target)
    specs.append(candidate_spec(
        "shuffled_v5_calibrated_transport",
        v1.vector_key(shuffled_source, shuffled_target),
        apply_v5_transport(
            z=z,
            source=shuffled_source,
            target=shuffled_target,
            stats=train_stats,
        ),
        {
            "control_source_behavior": shuffled_source,
            "control_target_behavior": shuffled_target,
        },
    ))
    matched_norm = float((matched_z - z).norm().clamp_min(1e-12).item())
    for index, random_vector in enumerate(random_norm_matched_vectors(
        subject_id=subject_id,
        source=source,
        target=target,
        z=z,
        matched_displacement_norm=matched_norm,
        random_controls=random_controls,
    )):
        specs.append(candidate_spec(
            "random_norm_matched_vector",
            f"random_norm_matched_vector:{index}",
            z + random_vector,
            {"random_index": index},
        ))
    return specs


def build_train_control_candidate_specs(
    *,
    subject_id: str,
    z: torch.Tensor,
    source: str,
    target: str,
    stats: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    return build_control_candidate_specs(
        subject_id=subject_id,
        z=z,
        source=source,
        target=target,
        train_stats=stats,
        random_controls=0,
    )


def candidate_spec(
    control_type: str,
    control_transport_key: str,
    candidate_z: torch.Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "candidate_z": candidate_z,
        "control_transport_key": control_transport_key,
        "control_type": control_type,
        "metadata": dict(metadata or {}),
    }


def select_shuffled_direction(
    subject_id: str,
    source: str,
    target: str,
) -> tuple[str, str]:
    candidates = sorted(
        (candidate_source, candidate_target)
        for candidate_source in PATTERNS
        for candidate_target in PATTERNS
        if candidate_source != candidate_target
        and (candidate_source, candidate_target) != (source, target)
        and (candidate_source, candidate_target) != (target, source)
    )
    digest = stable_hash_json([
        subject_id,
        source,
        target,
        "representation_steering_v5_contrastive_residual_calibration_shuffled_direction",
    ])
    index = int(digest[:16], 16) % len(candidates)
    return candidates[index]


def random_norm_matched_vectors(
    *,
    subject_id: str,
    source: str,
    target: str,
    z: torch.Tensor,
    matched_displacement_norm: float,
    random_controls: int,
) -> list[torch.Tensor]:
    seed_digest = stable_hash_json([
        subject_id,
        source,
        target,
        "representation_steering_v5_contrastive_residual_calibration_random",
    ])
    generator = torch.Generator().manual_seed(int(seed_digest[:16], 16) % (2 ** 31))
    matched_norm = max(float(matched_displacement_norm), 1e-12)
    vectors = []
    for _ in range(random_controls):
        random_vector = torch.randn(z.shape, dtype=z.dtype, generator=generator)
        random_vector = random_vector / random_vector.norm().clamp_min(1e-12)
        vectors.append(random_vector * matched_norm)
    return vectors


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summaries = [record["summary"] for record in records]
    return {
        "individual_all_gate_pass_count": sum(
            1 for record in records if record["individual_all_gates_passed"]
        ),
        "individual_all_gate_pass_rate": (
            sum(1 for record in records if record["individual_all_gates_passed"]) / len(records)
            if records else 0.0
        ),
        "mean_matched_centroid_improvement": v1.mean([
            item["matched_centroid_improvement"] for item in summaries
        ]),
        "mean_matched_minus_best_control_centroid_improvement": v1.mean([
            item["matched_minus_best_control_centroid_improvement"] for item in summaries
        ]),
        "mean_matched_minus_best_control_primary_target_margin": v1.mean([
            item["matched_minus_best_control_primary_target_margin"] for item in summaries
        ]),
        "mean_matched_minus_v2_centroid_delta_centroid_improvement": v1.mean([
            item["matched_minus_v2_centroid_delta_centroid_improvement"]
            for item in summaries
        ]),
        "mean_matched_minus_v2_centroid_delta_primary_target_margin": v1.mean([
            item["matched_minus_v2_centroid_delta_primary_target_margin"]
            for item in summaries
        ]),
        "mean_matched_minus_v3_diagonal_transport_centroid_improvement": v1.mean([
            item["matched_minus_v3_diagonal_transport_centroid_improvement"]
            for item in summaries
        ]),
        "mean_matched_minus_v3_diagonal_transport_primary_target_margin": v1.mean([
            item["matched_minus_v3_diagonal_transport_primary_target_margin"]
            for item in summaries
        ]),
        "mean_matched_minus_v4_low_rank_centroid_improvement": v1.mean([
            item["matched_minus_v4_low_rank_centroid_improvement"]
            for item in summaries
        ]),
        "mean_matched_minus_v4_low_rank_primary_target_margin": v1.mean([
            item["matched_minus_v4_low_rank_primary_target_margin"]
            for item in summaries
        ]),
        "mean_matched_primary_target_margin": v1.mean([
            item["matched_primary_target_margin"] for item in summaries
        ]),
        "mean_source_primary_margin_change": v1.mean([
            item["source_primary_margin_change"] for item in summaries
        ]),
        "n": len(records),
    }


def individual_gate_audit(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failures = []
    for record in records:
        summary = record["summary"]
        matched = record["matched"]
        failed = []
        checks = [
            ("primary_predicted_behavior", matched["primary_predicted_behavior"] == record["target_behavior"], "== target", matched["primary_predicted_behavior"]),
            ("centroid_predicted_behavior", matched["centroid_predicted_behavior"] == record["target_behavior"], "== target", matched["centroid_predicted_behavior"]),
            (
                "matched_primary_target_margin",
                v1.passes(summary["matched_primary_target_margin"], THRESHOLDS["min_per_record_primary_target_margin"], ">"),
                f"> {THRESHOLDS['min_per_record_primary_target_margin']}",
                summary["matched_primary_target_margin"],
            ),
            (
                "matched_minus_best_control_primary_target_margin",
                v1.passes(summary["matched_minus_best_control_primary_target_margin"], THRESHOLDS["min_per_record_matched_minus_best_control_primary_target_margin"], ">"),
                f"> {THRESHOLDS['min_per_record_matched_minus_best_control_primary_target_margin']}",
                summary["matched_minus_best_control_primary_target_margin"],
            ),
            (
                "matched_centroid_improvement",
                v1.passes(summary["matched_centroid_improvement"], THRESHOLDS["min_per_record_centroid_improvement"], ">"),
                f"> {THRESHOLDS['min_per_record_centroid_improvement']}",
                summary["matched_centroid_improvement"],
            ),
            (
                "matched_minus_best_control_centroid_improvement",
                v1.passes(summary["matched_minus_best_control_centroid_improvement"], THRESHOLDS["min_per_record_matched_minus_best_control_centroid_improvement"], ">"),
                f"> {THRESHOLDS['min_per_record_matched_minus_best_control_centroid_improvement']}",
                summary["matched_minus_best_control_centroid_improvement"],
            ),
            (
                "matched_minus_v2_centroid_delta_primary_target_margin",
                v1.passes(summary["matched_minus_v2_centroid_delta_primary_target_margin"], THRESHOLDS["min_per_record_matched_minus_v2_centroid_delta_primary_target_margin"], ">"),
                f"> {THRESHOLDS['min_per_record_matched_minus_v2_centroid_delta_primary_target_margin']}",
                summary["matched_minus_v2_centroid_delta_primary_target_margin"],
            ),
            (
                "matched_minus_v2_centroid_delta_centroid_improvement",
                v1.passes(summary["matched_minus_v2_centroid_delta_centroid_improvement"], THRESHOLDS["min_per_record_matched_minus_v2_centroid_delta_centroid_improvement"], ">"),
                f"> {THRESHOLDS['min_per_record_matched_minus_v2_centroid_delta_centroid_improvement']}",
                summary["matched_minus_v2_centroid_delta_centroid_improvement"],
            ),
            (
                "matched_minus_v3_diagonal_transport_primary_target_margin",
                v1.passes(summary["matched_minus_v3_diagonal_transport_primary_target_margin"], THRESHOLDS["min_per_record_matched_minus_v3_diagonal_transport_primary_target_margin"], ">"),
                f"> {THRESHOLDS['min_per_record_matched_minus_v3_diagonal_transport_primary_target_margin']}",
                summary["matched_minus_v3_diagonal_transport_primary_target_margin"],
            ),
            (
                "matched_minus_v3_diagonal_transport_centroid_improvement",
                v1.passes(summary["matched_minus_v3_diagonal_transport_centroid_improvement"], THRESHOLDS["min_per_record_matched_minus_v3_diagonal_transport_centroid_improvement"], ">"),
                f"> {THRESHOLDS['min_per_record_matched_minus_v3_diagonal_transport_centroid_improvement']}",
                summary["matched_minus_v3_diagonal_transport_centroid_improvement"],
            ),
            (
                "matched_minus_v4_low_rank_primary_target_margin",
                v1.passes(summary["matched_minus_v4_low_rank_primary_target_margin"], THRESHOLDS["min_per_record_matched_minus_v4_low_rank_primary_target_margin"], ">"),
                f"> {THRESHOLDS['min_per_record_matched_minus_v4_low_rank_primary_target_margin']}",
                summary["matched_minus_v4_low_rank_primary_target_margin"],
            ),
            (
                "matched_minus_v4_low_rank_centroid_improvement",
                v1.passes(summary["matched_minus_v4_low_rank_centroid_improvement"], THRESHOLDS["min_per_record_matched_minus_v4_low_rank_centroid_improvement"], ">"),
                f"> {THRESHOLDS['min_per_record_matched_minus_v4_low_rank_centroid_improvement']}",
                summary["matched_minus_v4_low_rank_centroid_improvement"],
            ),
            (
                "source_primary_margin_change",
                v1.passes(summary["source_primary_margin_change"], THRESHOLDS["per_record_source_primary_margin_change_must_be_below"], "<"),
                f"< {THRESHOLDS['per_record_source_primary_margin_change_must_be_below']}",
                summary["source_primary_margin_change"],
            ),
        ]
        for name, passed, required, observed in checks:
            if not passed:
                failed.append({"check": name, "observed": observed, "required": required})
        if failed:
            failures.append({
                "failed": failed,
                "source_behavior": record["source_behavior"],
                "subject_id": record["subject_id"],
                "target_behavior": record["target_behavior"],
            })
    return {
        "all_gate_pass_count": len(records) - len(failures),
        "all_gate_pass_rate": (
            (len(records) - len(failures)) / len(records) if records else 0.0
        ),
        "failed_records": failures,
        "n": len(records),
    }


def gate_failures(
    *,
    aggregate: Mapping[str, Any],
    by_target: Mapping[str, Mapping[str, Any]],
    by_direction: Mapping[str, Mapping[str, Any]],
    individual_audit: Mapping[str, Any],
) -> list[str]:
    failures = []
    aggregate_checks = [
        ("n", aggregate["n"], THRESHOLDS["expected_record_count"], "=="),
        ("mean matched primary target margin", aggregate["mean_matched_primary_target_margin"], THRESHOLDS["min_mean_matched_primary_target_margin"], ">"),
        ("mean matched-minus-best-control primary target margin", aggregate["mean_matched_minus_best_control_primary_target_margin"], THRESHOLDS["min_mean_matched_minus_best_control_primary_target_margin"], ">"),
        ("mean matched centroid improvement", aggregate["mean_matched_centroid_improvement"], THRESHOLDS["min_mean_matched_centroid_improvement"], ">"),
        ("mean matched-minus-best-control centroid improvement", aggregate["mean_matched_minus_best_control_centroid_improvement"], THRESHOLDS["min_mean_matched_minus_best_control_centroid_improvement"], ">"),
        ("mean matched-minus-V2-centroid-delta primary target margin", aggregate["mean_matched_minus_v2_centroid_delta_primary_target_margin"], THRESHOLDS["min_mean_matched_minus_v2_centroid_delta_primary_target_margin"], ">"),
        ("mean matched-minus-V2-centroid-delta centroid improvement", aggregate["mean_matched_minus_v2_centroid_delta_centroid_improvement"], THRESHOLDS["min_mean_matched_minus_v2_centroid_delta_centroid_improvement"], ">"),
        ("mean matched-minus-V3-diagonal-transport primary target margin", aggregate["mean_matched_minus_v3_diagonal_transport_primary_target_margin"], THRESHOLDS["min_mean_matched_minus_v3_diagonal_transport_primary_target_margin"], ">"),
        ("mean matched-minus-V3-diagonal-transport centroid improvement", aggregate["mean_matched_minus_v3_diagonal_transport_centroid_improvement"], THRESHOLDS["min_mean_matched_minus_v3_diagonal_transport_centroid_improvement"], ">"),
        ("mean matched-minus-V4-low-rank primary target margin", aggregate["mean_matched_minus_v4_low_rank_primary_target_margin"], THRESHOLDS["min_mean_matched_minus_v4_low_rank_primary_target_margin"], ">"),
        ("mean matched-minus-V4-low-rank centroid improvement", aggregate["mean_matched_minus_v4_low_rank_centroid_improvement"], THRESHOLDS["min_mean_matched_minus_v4_low_rank_centroid_improvement"], ">"),
        ("mean source primary margin change", aggregate["mean_source_primary_margin_change"], THRESHOLDS["max_mean_source_primary_margin_change"], "<"),
    ]
    for name, value, threshold, operator in aggregate_checks:
        if not v1.passes(float(value), float(threshold), operator):
            failures.append(v1.format_failure("aggregate", name, value, threshold, operator))
    if individual_audit["all_gate_pass_rate"] < THRESHOLDS["min_aggregate_individual_pass_rate"]:
        failures.append(v1.format_failure(
            "aggregate",
            "individual pass rate",
            individual_audit["all_gate_pass_rate"],
            THRESHOLDS["min_aggregate_individual_pass_rate"],
            ">=",
        ))
    for target, summary in by_target.items():
        target_checks = [
            ("n", summary["n"], THRESHOLDS["expected_per_target_count"], "=="),
            ("individual pass rate", summary["individual_all_gate_pass_rate"], THRESHOLDS["min_per_target_individual_pass_rate"], ">="),
        ]
        for name, value, threshold, operator in target_checks:
            if not v1.passes(float(value), float(threshold), operator):
                failures.append(v1.format_failure(f"target {target}", name, value, threshold, operator))
    for direction, summary in by_direction.items():
        direction_checks = [
            ("n", summary["n"], THRESHOLDS["expected_per_direction_count"], "=="),
            ("individual pass rate", summary["individual_all_gate_pass_rate"], THRESHOLDS["min_direction_individual_pass_rate"], ">="),
        ]
        for name, value, threshold, operator in direction_checks:
            if not v1.passes(float(value), float(threshold), operator):
                failures.append(v1.format_failure(f"direction {direction}", name, value, threshold, operator))
    return failures


def record_passes(
    summary: Mapping[str, float],
    matched: Mapping[str, Any],
    *,
    target: str,
) -> bool:
    return (
        matched["primary_predicted_behavior"] == target
        and matched["centroid_predicted_behavior"] == target
        and summary["matched_primary_target_margin"]
        > THRESHOLDS["min_per_record_primary_target_margin"]
        and summary["matched_minus_best_control_primary_target_margin"]
        > THRESHOLDS["min_per_record_matched_minus_best_control_primary_target_margin"]
        and summary["matched_centroid_improvement"]
        > THRESHOLDS["min_per_record_centroid_improvement"]
        and summary["matched_minus_best_control_centroid_improvement"]
        > THRESHOLDS["min_per_record_matched_minus_best_control_centroid_improvement"]
        and summary["matched_minus_v2_centroid_delta_primary_target_margin"]
        > THRESHOLDS["min_per_record_matched_minus_v2_centroid_delta_primary_target_margin"]
        and summary["matched_minus_v2_centroid_delta_centroid_improvement"]
        > THRESHOLDS["min_per_record_matched_minus_v2_centroid_delta_centroid_improvement"]
        and summary["matched_minus_v3_diagonal_transport_primary_target_margin"]
        > THRESHOLDS["min_per_record_matched_minus_v3_diagonal_transport_primary_target_margin"]
        and summary["matched_minus_v3_diagonal_transport_centroid_improvement"]
        > THRESHOLDS["min_per_record_matched_minus_v3_diagonal_transport_centroid_improvement"]
        and summary["matched_minus_v4_low_rank_primary_target_margin"]
        > THRESHOLDS["min_per_record_matched_minus_v4_low_rank_primary_target_margin"]
        and summary["matched_minus_v4_low_rank_centroid_improvement"]
        > THRESHOLDS["min_per_record_matched_minus_v4_low_rank_centroid_improvement"]
        and summary["source_primary_margin_change"]
        < THRESHOLDS["per_record_source_primary_margin_change_must_be_below"]
    )


def validate_source_pool_contract(
    *,
    train_payload: Mapping[str, Any],
    eval_payload: Mapping[str, Any],
    combined_audit: Mapping[str, Any],
    final_redacted: Mapping[str, Any],
    phase: str,
) -> list[str]:
    failures = []
    if combined_audit.get("claim_scope") != "four_behavior_representation_steering_v5_source_pool_construction":
        failures.append("combined source-pool audit is not V5-specific")
    if final_redacted.get("claim_scope") != "redacted_final_steering_v5_source_pool_audit_surface_only":
        failures.append("final redacted audit is not V5-specific")
    if train_payload.get("claim_scope") != "four_behavior_representation_steering_v5_source_pool":
        failures.append("train source pool is not V5-specific")
    if eval_payload.get("claim_scope") != "four_behavior_representation_steering_v5_source_pool":
        failures.append(f"{phase} eval source pool is not V5-specific")
    if train_payload.get("pool") != "train":
        failures.append("train source pool payload has wrong pool name")
    expected_eval_pool = "development" if phase == "development" else "final"
    if eval_payload.get("pool") != expected_eval_pool:
        failures.append(f"{phase} eval source pool payload has wrong pool name")
    if not combined_audit.get("passed", False):
        failures.append("combined source-pool audit did not pass")
    if not combined_audit.get("seed_preflight", {}).get("passed", False):
        failures.append("V5 seed-range preflight did not pass")
    for key, value in combined_audit.get("overlap_counts", {}).items():
        if int(value) != 0:
            failures.append(f"cross-pool accepted {key} overlap count {value}")
    pool_summaries = combined_audit.get("pool_summaries", {})
    for pool_name, pool_config in POOL_CONFIGS.items():
        expected_count = int(pool_config["target_accepted_per_behavior"])
        if pool_name == "final":
            summary = final_redacted.get("summary", {})
        else:
            summary = pool_summaries.get(pool_name, {})
        counts = summary.get("accepted_counts_by_behavior", {})
        for pattern in PATTERNS:
            if int(counts.get(pattern, -1)) != expected_count:
                failures.append(
                    f"{pool_name}/{pattern} accepted count "
                    f"{counts.get(pattern)} != preregistered {expected_count}"
                )
    forbidden_final_keys = {
        "accepted_subject_ids",
        "acceptance_rate",
        "attempt_count",
        "attempt_counts_by_behavior",
        "by_behavior",
        "heldout_margin",
        "heldout_margin_mean",
        "heldout_margin_min",
        "margin",
        "primary_source_margin",
        "records",
        "rejected_count",
        "rejected_subject_ids",
        "signature",
        "signature_hash",
        "source_heldout_margin",
        "source_margin",
        "subject_id",
        "support_margin",
        "weights",
        "weights_hash",
    }
    final_summary_keys = nested_keys(pool_summaries.get("final", {}))
    final_redacted_keys = nested_keys(final_redacted)
    for forbidden_key in forbidden_final_keys:
        if forbidden_key in final_summary_keys:
            failures.append(f"combined audit exposes forbidden final detail: {forbidden_key}")
        if forbidden_key in final_redacted_keys:
            failures.append(f"final redacted audit exposes forbidden final detail: {forbidden_key}")
    return failures


def validate_source_pool_file_hashes(
    *,
    train_path: Path,
    eval_path: Path,
    combined_audit: Mapping[str, Any],
    final_redacted: Mapping[str, Any],
    phase: str,
) -> list[str]:
    failures = []
    expected_eval_pool = "development" if phase == "development" else "final"
    pool_hashes = combined_audit.get("pool_file_sha256", {})
    pool_summaries = combined_audit.get("pool_summaries", {})
    expected_train_hash = pool_hashes.get("train") or pool_summaries.get("train", {}).get(
        "pool_file_sha256"
    )
    expected_eval_hash = pool_hashes.get(expected_eval_pool) or pool_summaries.get(
        expected_eval_pool, {}
    ).get("pool_file_sha256")
    actual_train_hash = v1.sha256_file(train_path)
    actual_eval_hash = v1.sha256_file(eval_path)
    if actual_train_hash != expected_train_hash:
        failures.append(
            f"train pool sha256 {actual_train_hash} does not match audit {expected_train_hash}"
        )
    if actual_eval_hash != expected_eval_hash:
        failures.append(
            f"{expected_eval_pool} pool sha256 {actual_eval_hash} "
            f"does not match audit {expected_eval_hash}"
        )
    if phase == "final":
        redacted_final_hash = final_redacted.get("pool_file_sha256")
        if actual_eval_hash != redacted_final_hash:
            failures.append(
                f"final pool sha256 {actual_eval_hash} "
                f"does not match final redacted audit {redacted_final_hash}"
            )
    return failures


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(nested_keys(child))
    return keys


def train_only_statistics_hash(train_stats: Mapping[str, Any]) -> str:
    return stable_hash_json({
        "centroids": {
            key: tensor_to_jsonable(value)
            for key, value in train_stats["centroids"].items()
        },
        "pca_components": tensor_to_jsonable(train_stats["pca_components"]),
        "signature_mean": tensor_to_jsonable(train_stats["sig_mean"]),
        "signature_std": tensor_to_jsonable(train_stats["sig_std"]),
        "transport_stds": {
            key: tensor_to_jsonable(value)
            for key, value in train_stats["transport_stds"].items()
        },
    })


def calibration_coefficient_hash(coefficients: Mapping[str, torch.Tensor]) -> str:
    return stable_hash_json({
        key: tensor_to_jsonable(value)
        for key, value in sorted(coefficients.items())
    })


def tensor_to_jsonable(tensor: torch.Tensor) -> Any:
    def convert(value: Any) -> Any:
        if isinstance(value, list):
            return [convert(item) for item in value]
        return float(value)

    return convert(tensor.detach().cpu().tolist())


if __name__ == "__main__":
    main()
