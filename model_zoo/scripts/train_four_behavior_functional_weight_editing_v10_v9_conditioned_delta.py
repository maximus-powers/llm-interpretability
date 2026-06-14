"""V10 source-label-known functional weight editing with V9-conditioned deltas."""

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
import train_four_behavior_decoder_development as decoder_v1  # noqa: E402
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
from hypernet.paired_contrast import build_digit_probe_examples  # noqa: E402


SEED_BEHAVIOR_STRIDE = 100000
POOL_CONFIGS = {
    "train": {
        "base_seed": 60300000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 61300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 62300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}
DEFAULT_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v10_pools"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runs"
    / "four_behavior_functional_weight_editing_v10_v9_conditioned_delta"
)
PREREG_PATH = (
    REPO_ROOT
    / "docs"
    / "preregistrations"
    / "four_behavior_functional_weight_editing_v10_v9_conditioned_delta.md"
)
SCRIPT_PATH = Path(__file__).resolve()
SOURCE_POOL_SCOPE = "four_behavior_functional_weight_editing_v10_source_pool"
SOURCE_AUDIT_SCOPE = "four_behavior_functional_weight_editing_v10_source_pool_construction"
FINAL_REDACTED_SCOPE = (
    "redacted_final_functional_weight_editing_v10_source_pool_audit_surface_only"
)
DEVELOPMENT_SCOPE = (
    "four_behavior_functional_weight_editing_v10_v9_conditioned_delta_development"
)
FINAL_SCOPE = "four_behavior_functional_weight_editing_v10_v9_conditioned_delta_final"
EDITOR_METHOD = "v9_conditioned_train_only_ridge_weight_delta"
RIDGE_LAMBDA = 10.0
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
    "expected_per_direction_count": 24,
    "max_mean_source_margin_change": -0.05,
    "min_aggregate_individual_pass_rate": 0.85,
    "min_aggregate_pareto_undominated_rate": 0.85,
    "min_aggregate_target_prediction_rate": 0.90,
    "min_direction_individual_pass_rate": 0.70,
    "min_direction_target_prediction_rate": 0.80,
    "min_mean_matched_minus_best_control_target_margin": 0.00,
    "min_mean_matched_minus_nearest_train_target_margin": -0.05,
    "min_mean_matched_minus_no_edit_target_margin": 0.20,
    "min_mean_matched_target_margin": 0.20,
    "min_mean_matched_target_vs_source_margin": 0.20,
    "min_mean_nearest_train_minus_matched_source_output_mse": 0.00,
    "min_per_record_matched_target_margin": 0.20,
    "min_per_record_target_vs_source_margin": 0.20,
    "per_record_source_margin_change_must_be_below": -0.05,
    "random_controls_per_record": 32,
}
FORBIDDEN_FINAL_DETAIL_KEYS = {
    "accepted_attempt_indices",
    "accepted_subject_ids",
    "acceptance_rate",
    "acceptance_rates",
    "attempt_count",
    "attempt_counts",
    "attempt_index",
    "attempt_indices",
    "behavior",
    "behavior_label",
    "heldout_margin",
    "heldout_margins",
    "margin",
    "margins",
    "per_subject_metrics",
    "record",
    "records",
    "rejected_attempt_indices",
    "rejected_subject_ids",
    "rejection_count",
    "rejection_counts",
    "seed",
    "seeds",
    "signature",
    "signature_hash",
    "signature_hashes",
    "source_margin",
    "source_margins",
    "subject_id",
    "subject_ids",
    "support_margin",
    "support_margins",
    "weight_hash",
    "weight_hashes",
    "weights",
    "weights_hash",
    "weights_hashes",
}
PRIOR_FINAL_RAW_PATHS = {
    REPO_ROOT / "runs" / "four_behavior_decoder_source_pools_v2" / "final_subjects.json",
    *[
        REPO_ROOT / "runs" / f"four_behavior_representation_steering_v{i}_pools" / "final_subjects.json"
        for i in range(1, 9)
    ],
    REPO_ROOT / "runs" / "four_behavior_representation_steering_v9_pools" / "final_subjects.json",
}
V10_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"
_EVAL_SUITE: Dict[str, Any] | None = None


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
    parser.add_argument("--random-controls", type=int, default=32)
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
        raise ValueError("Non-preregistered V10 parameter override: " + "; ".join(failures))


def assert_preregistered_pool_dir(pool_dir: Path) -> None:
    resolved = pool_dir.resolve()
    if resolved != DEFAULT_POOL_DIR.resolve():
        raise ValueError(f"V10 requires preregistered pool directory: {DEFAULT_POOL_DIR}")


def assert_no_forbidden_final_raw_paths(
    paths: Iterable[Path | str],
    *,
    allow_v10_final: bool,
) -> None:
    forbidden = {path.resolve() for path in PRIOR_FINAL_RAW_PATHS}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in forbidden:
            raise ValueError(f"prior sealed final raw path is forbidden: {path}")
        if path.name != "final_subjects.json":
            continue
        if allow_v10_final and path == V10_FINAL_RAW.resolve():
            continue
        raise ValueError(f"V10 final raw path is forbidden before final eval: {path}")


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
    audit = redact_combined_audit(audit)
    audit["audit_payload_sha256"] = stable_hash_json(audit)
    (pool_dir / "combined_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    return {
        "combined_audit_path": v1.rel(pool_dir / "combined_audit.json"),
        "final_redacted_audit_path": v1.rel(pool_dir / "final_redacted_audit.json"),
        "passed": bool(audit["passed"]),
        "pool_summaries": audit["pool_summaries"],
        "seed_preflight": seed_preflight,
    }


def redact_combined_audit(audit: Mapping[str, Any]) -> Dict[str, Any]:
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
    assert_no_forbidden_final_raw_paths(paths.values(), allow_v10_final=False)
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
    assert_no_forbidden_final_raw_paths([eval_path], allow_v10_final=True)
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
        allow_v10_final=allow_eval_final_raw,
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
        raise ValueError("V10 source-pool contract validation failed: " + "; ".join(contract_failures))

    train_subjects = v1.accepted_records(train_payload)
    eval_subjects = v1.accepted_records(eval_payload)
    train_stats = fit_v10_train_statistics(train_subjects)
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
    editor = fit_ridge_editor(
        subjects=train_subjects,
        train_stats=train_stats,
        classifier=classifier,
    )
    editor_path = output_dir / "v10_ridge_editor.pt"
    torch.save(
        {
            "coef": editor["coef"],
            "method": EDITOR_METHOD,
            "ridge_lambda": RIDGE_LAMBDA,
            "train_statistics_hash": train_only_statistics_hash(train_stats),
        },
        editor_path,
    )
    eval_result = evaluate_subjects(
        subjects=eval_subjects,
        train_stats=train_stats,
        classifier=classifier,
        editor=editor,
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
        "editor_path": v1.rel(editor_path),
        "editor_sha256": v1.sha256_file(editor_path),
        "eval_pool_path": v1.rel(eval_path),
        "eval_pool_sha256": v1.sha256_file(eval_path),
        "final_redacted_audit_path": v1.rel(final_redacted_path),
        "final_redacted_audit_sha256": v1.sha256_file(final_redacted_path),
        "final_redacted_summary": final_redacted,
        "forbidden_prior_final_raw_opened": False,
        "forbidden_v10_final_raw_opened_before_authorization": False,
        "implementation_sha256": v1.sha256_file(SCRIPT_PATH),
        "limitations": (
            "Small-subject source-label-known functional weight editing only; "
            "not source-label inference, larger-model evidence, broad MUAT proof, "
            "or non-target capability preservation evidence."
        ),
        "phase": phase,
        "pool_overlap_counts": combined_audit.get("overlap_counts", {}),
        "preregistration_sha256": v1.sha256_file(PREREG_PATH),
        "ridge_lambda": RIDGE_LAMBDA,
        "source_pool_audit_passed": bool(combined_audit.get("passed", False)),
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
        forbidden_paths = find_forbidden_final_detail_paths(surface, prefix=surface_name)
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
            allow_v10_final=False,
        )
    return failures


def find_forbidden_final_detail_paths(obj: Any, *, prefix: str) -> list[str]:
    found = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}"
            if key_text.lower() in FORBIDDEN_FINAL_DETAIL_KEYS:
                found.append(path)
            found.extend(find_forbidden_final_detail_paths(value, prefix=path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(find_forbidden_final_detail_paths(value, prefix=f"{prefix}[{index}]"))
    return found


def fit_v10_train_statistics(subjects: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    stats = v9.fit_train_statistics(subjects)
    train_weights = torch.tensor([record["weights"] for record in subjects], dtype=torch.float32)
    weight_mean = train_weights.mean(dim=0)
    weight_std = clamp_std(train_weights.std(dim=0, unbiased=False))
    train_weights_norm = (train_weights - weight_mean) / weight_std
    stats["train_subjects"] = list(subjects)
    stats["train_weights"] = train_weights
    stats["train_weights_norm"] = train_weights_norm
    stats["weight_mean"] = weight_mean
    stats["weight_std"] = weight_std
    stats["weight_centroids_norm"] = {
        pattern: train_weights_norm[
            torch.tensor([v1.behavior_of(record) == pattern for record in subjects], dtype=torch.bool)
        ].mean(dim=0)
        for pattern in PATTERNS
    }
    stats["global_weight_centroid_norm"] = train_weights_norm.mean(dim=0)
    return stats


def clamp_std(std: torch.Tensor) -> torch.Tensor:
    return torch.where(std < 1e-6, torch.ones_like(std), std)


def selected_v9_conditioning(
    *,
    z: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
) -> Dict[str, Any]:
    frontier = v9.apply_v8_frontier(
        z=z,
        source=source,
        target=target,
        stats=train_stats,
        classifier=classifier,
    )
    candidates = []
    for index, candidate_z in enumerate(frontier):
        score = v1.score_candidate(
            z=z,
            candidate_z=candidate_z,
            source=source,
            target=target,
            classifier=classifier,
            centroids=train_stats["centroids"],
        )
        candidates.append({
            "candidate_index": index,
            "candidate_z": candidate_z,
            "selected_centroid_improvement": score["centroid_improvement"],
            "selected_displacement_norm": float((candidate_z - z).norm().item()),
            "selected_primary_target_margin": score["primary_target_margin"],
        })
    return max(
        candidates,
        key=lambda item: (
            item["selected_primary_target_margin"],
            item["selected_centroid_improvement"],
            -item["candidate_index"],
        ),
    )


def editor_feature(
    *,
    z: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
) -> Dict[str, Any]:
    selected = selected_v9_conditioning(
        z=z,
        source=source,
        target=target,
        train_stats=train_stats,
        classifier=classifier,
    )
    selected_z = selected["candidate_z"]
    source_one_hot = one_hot_behavior(source)
    target_one_hot = one_hot_behavior(target)
    scalars = torch.tensor(
        [
            selected["selected_primary_target_margin"],
            selected["selected_centroid_improvement"],
            selected["selected_displacement_norm"],
        ],
        dtype=torch.float32,
    )
    feature = torch.cat([
        z,
        selected_z,
        selected_z - z,
        source_one_hot,
        target_one_hot,
        scalars,
    ])
    return {**selected, "feature": feature}


def one_hot_behavior(pattern: str) -> torch.Tensor:
    values = torch.zeros(len(PATTERNS), dtype=torch.float32)
    values[PATTERNS.index(pattern)] = 1.0
    return values


def fit_ridge_editor(
    *,
    subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    feature_dim = 560 * 3 + len(PATTERNS) * 2 + 3
    weight_dim = int(train_stats["train_weights_norm"].shape[1])
    xtx = torch.zeros((feature_dim + 1, feature_dim + 1), dtype=torch.float64)
    xty = torch.zeros((feature_dim + 1, weight_dim), dtype=torch.float64)
    by_behavior = {
        pattern: [
            index
            for index, record in enumerate(subjects)
            if v1.behavior_of(record) == pattern
        ]
        for pattern in PATTERNS
    }
    target_sums = {
        pattern: train_stats["train_weights_norm"][indices].sum(dim=0).to(torch.float64)
        for pattern, indices in by_behavior.items()
    }
    target_counts = {pattern: len(indices) for pattern, indices in by_behavior.items()}
    for source_index, subject in enumerate(subjects):
        source = v1.behavior_of(subject)
        z = train_stats["z_train"][source_index]
        source_weight_norm = train_stats["train_weights_norm"][source_index].to(torch.float64)
        for target in PATTERNS:
            if target == source:
                continue
            feature = editor_feature(
                z=z,
                source=source,
                target=target,
                train_stats=train_stats,
                classifier=classifier,
            )["feature"].to(torch.float64)
            x_aug = torch.cat([feature, torch.ones(1, dtype=torch.float64)])
            count = target_counts[target]
            y_sum = target_sums[target] - count * source_weight_norm
            xtx += count * torch.outer(x_aug, x_aug)
            xty += torch.outer(x_aug, y_sum)
    penalty = torch.eye(feature_dim + 1, dtype=torch.float64) * RIDGE_LAMBDA
    penalty[-1, -1] = 0.0
    coef = torch.linalg.solve(xtx + penalty, xty).to(torch.float32)
    return {"coef": coef}


def predict_delta_norm(feature: torch.Tensor, editor: Mapping[str, torch.Tensor]) -> torch.Tensor:
    x_aug = torch.cat([feature.to(torch.float32), torch.ones(1, dtype=torch.float32)])
    return x_aug @ editor["coef"]


def denormalize_weight_norm(weight_norm: torch.Tensor, train_stats: Mapping[str, Any]) -> torch.Tensor:
    return weight_norm * train_stats["weight_std"] + train_stats["weight_mean"]


def evaluate_subjects(
    *,
    subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    editor: Mapping[str, torch.Tensor],
    random_controls: int,
) -> Dict[str, Any]:
    records = []
    for subject in subjects:
        source = v1.behavior_of(subject)
        source_signature = torch.tensor(subject["signature"], dtype=torch.float32)
        z = (source_signature - train_stats["sig_mean"]) / train_stats["sig_std"]
        source_weights = torch.tensor(subject["weights"], dtype=torch.float32)
        source_weight_norm = (source_weights - train_stats["weight_mean"]) / train_stats["weight_std"]
        for target in PATTERNS:
            if target == source:
                continue
            records.append(evaluate_record(
                subject=subject,
                z=z,
                source=source,
                target=target,
                source_weights=source_weights,
                source_weight_norm=source_weight_norm,
                train_stats=train_stats,
                classifier=classifier,
                editor=editor,
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
    source_weight_norm: torch.Tensor,
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    editor: Mapping[str, torch.Tensor],
    random_controls: int,
) -> Dict[str, Any]:
    matched_feature = editor_feature(
        z=z,
        source=source,
        target=target,
        train_stats=train_stats,
        classifier=classifier,
    )
    matched_delta = predict_delta_norm(matched_feature["feature"], editor)
    matched_weight_norm = source_weight_norm + matched_delta
    matched_weights = denormalize_weight_norm(matched_weight_norm, train_stats)
    no_edit_metrics = functional_metrics(source_weights, source, target, source_weights)
    matched = {
        **functional_metrics(matched_weights, source, target, source_weights),
        "delta_norm": float(matched_delta.norm().item()),
        "selected_candidate_index": int(matched_feature["candidate_index"]),
        "selected_centroid_improvement": matched_feature["selected_centroid_improvement"],
        "selected_displacement_norm": matched_feature["selected_displacement_norm"],
        "selected_primary_target_margin": matched_feature["selected_primary_target_margin"],
    }
    controls = build_controls(
        subject=subject,
        z=z,
        source=source,
        target=target,
        source_weights=source_weights,
        source_weight_norm=source_weight_norm,
        matched_delta_norm=matched_delta.norm(),
        selected_signature_norm=matched_feature["candidate_z"],
        train_stats=train_stats,
        classifier=classifier,
        editor=editor,
        random_controls=random_controls,
    )
    for control in controls:
        control["matched_minus_control_target_margin"] = (
            matched["target_margin"] - control["target_margin"]
        )
        control["control_minus_matched_source_output_mse"] = (
            control["source_output_mse"] - matched["source_output_mse"]
        )
    best_target = max(controls, key=lambda item: item["target_margin"])
    nearest = next(control for control in controls if control["control_type"] == "nearest_train_target_retrieval")
    no_edit = next(control for control in controls if control["control_type"] == "no_edit")
    pareto_dominators = [
        control for control in controls if pareto_dominates_functional(control, matched)
    ]
    matched["target_vs_source_margin"] = matched["target_margin"] - matched["source_margin"]
    matched["source_margin_change"] = (
        matched["source_margin"] - no_edit_metrics["source_margin"]
    )
    matched["pareto_dominator_count"] = len(pareto_dominators)
    matched["pareto_dominator_types"] = sorted({
        control["control_type"] for control in pareto_dominators
    })
    matched["pareto_undominated"] = not pareto_dominators
    matched["target_prediction_pass"] = matched["predicted_behavior"] == target
    matched["individual_all_gates_passed"] = individual_passed(
        matched=matched,
        nearest=nearest,
    )
    target_inputs = heldout_inputs_for(target)
    matched["target_exemplar_output_mse"] = subject_output_mse(
        matched_weights,
        nearest["weights"],
        target_inputs,
    )
    return {
        "controls": strip_control_weights(controls),
        "individual_all_gates_passed": matched["individual_all_gates_passed"],
        "matched": strip_weight(matched),
        "nearest_train_minus_matched_source_output_mse": (
            nearest["source_output_mse"] - matched["source_output_mse"]
        ),
        "random_control_count": sum(
            1 for control in controls if control["control_type"].startswith("random_norm_matched_weight_delta")
        ),
        "source_behavior": source,
        "subject_id": str(subject["subject_id"]),
        "summary": {
            "best_control_target_margin": best_target["target_margin"],
            "best_control_type": best_target["control_type"],
            "matched_minus_best_control_target_margin": (
                matched["target_margin"] - best_target["target_margin"]
            ),
            "matched_minus_nearest_train_target_margin": (
                matched["target_margin"] - nearest["target_margin"]
            ),
            "matched_minus_no_edit_target_margin": (
                matched["target_margin"] - no_edit["target_margin"]
            ),
            "nearest_train_minus_matched_source_output_mse": (
                nearest["source_output_mse"] - matched["source_output_mse"]
            ),
            "pareto_undominated": matched["pareto_undominated"],
            "source_margin_change": matched["source_margin_change"],
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
    source_weight_norm: torch.Tensor,
    matched_delta_norm: torch.Tensor,
    selected_signature_norm: torch.Tensor,
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    editor: Mapping[str, torch.Tensor],
    random_controls: int,
) -> list[Dict[str, Any]]:
    controls = []
    controls.append(control_record("no_edit", source_weights, source, target, source_weights))
    controls.append(control_record("null_delta", source_weights, source, target, source_weights))
    controls.append(editor_control(
        control_type="reverse_behavior_pair",
        z=z,
        source_label=target,
        target_label=source,
        source=source,
        target=target,
        source_weights=source_weights,
        source_weight_norm=source_weight_norm,
        train_stats=train_stats,
        classifier=classifier,
        editor=editor,
    ))
    for other_target in PATTERNS:
        if other_target in {source, target}:
            continue
        controls.append(editor_control(
            control_type=f"same_source_other_target:{other_target}",
            z=z,
            source_label=source,
            target_label=other_target,
            source=source,
            target=target,
            source_weights=source_weights,
            source_weight_norm=source_weight_norm,
            train_stats=train_stats,
            classifier=classifier,
            editor=editor,
        ))
    shuffled = select_shuffled_target(
        subject_id=str(subject["subject_id"]),
        source=source,
        target=target,
    )
    controls.append(editor_control(
        control_type=f"shuffled_target:{shuffled}",
        z=z,
        source_label=source,
        target_label=shuffled,
        source=source,
        target=target,
        source_weights=source_weights,
        source_weight_norm=source_weight_norm,
        train_stats=train_stats,
        classifier=classifier,
        editor=editor,
    ))
    nearest = nearest_train_target_retrieval(
        selected_signature_norm=selected_signature_norm,
        target=target,
        train_stats=train_stats,
    )
    controls.append(control_record(
        "nearest_train_target_retrieval",
        nearest["weights"],
        source,
        target,
        source_weights,
        {"retrieved_subject_id": nearest["subject_id"]},
    ))
    target_centroid = denormalize_weight_norm(train_stats["weight_centroids_norm"][target], train_stats)
    controls.append(control_record(
        "train_target_behavior_centroid",
        target_centroid,
        source,
        target,
        source_weights,
    ))
    global_centroid = denormalize_weight_norm(train_stats["global_weight_centroid_norm"], train_stats)
    controls.append(control_record(
        "train_global_weight_centroid",
        global_centroid,
        source,
        target,
        source_weights,
    ))
    controls.extend(random_weight_delta_controls(
        subject_id=str(subject["subject_id"]),
        source=source,
        target=target,
        source_weights=source_weights,
        source_weight_norm=source_weight_norm,
        matched_delta_norm=matched_delta_norm,
        train_stats=train_stats,
        random_controls=random_controls,
    ))
    return controls


def editor_control(
    *,
    control_type: str,
    z: torch.Tensor,
    source_label: str,
    target_label: str,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    source_weight_norm: torch.Tensor,
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    editor: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    feature = editor_feature(
        z=z,
        source=source_label,
        target=target_label,
        train_stats=train_stats,
        classifier=classifier,
    )["feature"]
    delta = predict_delta_norm(feature, editor)
    weights = denormalize_weight_norm(source_weight_norm + delta, train_stats)
    return control_record(
        control_type,
        weights,
        source,
        target,
        source_weights,
        {
            "control_source_label": source_label,
            "control_target_label": target_label,
            "delta_norm": float(delta.norm().item()),
        },
    )


def nearest_train_target_retrieval(
    *,
    selected_signature_norm: torch.Tensor,
    target: str,
    train_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    candidates = []
    for index, subject in enumerate(train_stats["train_subjects"]):
        if v1.behavior_of(subject) != target:
            continue
        distance = float(
            (selected_signature_norm - train_stats["z_train"][index]).pow(2).sum().item()
        )
        candidates.append({
            "distance": distance,
            "signature_hash": subject["signature_hash"],
            "subject_id": str(subject["subject_id"]),
            "weights": train_stats["train_weights"][index],
            "weights_hash": subject["weights_hash"],
        })
    if not candidates:
        raise ValueError(f"no nearest-train candidates for {target}")
    return sorted(
        candidates,
        key=lambda item: (
            item["distance"],
            item["weights_hash"],
            item["signature_hash"],
            item["subject_id"],
        ),
    )[0]


def select_shuffled_target(*, subject_id: str, source: str, target: str) -> str:
    candidates = [pattern for pattern in PATTERNS if pattern not in {source, target}]
    key = stable_hash_json([
        subject_id,
        source,
        target,
        "functional_weight_editing_v10_shuffled_target",
    ])
    return candidates[int(key[:16], 16) % len(candidates)]


def random_weight_delta_controls(
    *,
    subject_id: str,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    source_weight_norm: torch.Tensor,
    matched_delta_norm: torch.Tensor,
    train_stats: Mapping[str, Any],
    random_controls: int,
) -> list[Dict[str, Any]]:
    key = stable_hash_json([
        subject_id,
        source,
        target,
        "functional_weight_editing_v10_random_weight_delta",
    ])
    seed = int(key[:16], 16) % (2**31)
    generator = torch.Generator().manual_seed(seed)
    controls = []
    scale = matched_delta_norm.clamp_min(1e-12)
    for index in range(random_controls):
        delta = torch.randn(source_weight_norm.shape, generator=generator)
        delta = scale * delta / delta.norm().clamp_min(1e-12)
        weights = denormalize_weight_norm(source_weight_norm + delta, train_stats)
        controls.append(control_record(
            f"random_norm_matched_weight_delta:{index:02d}",
            weights,
            source,
            target,
            source_weights,
            {"delta_norm": float(delta.norm().item()), "random_seed": int(seed)},
        ))
    return controls


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
    margins = {pattern: behavior_margin(weights, pattern) for pattern in PATTERNS}
    source_inputs = heldout_inputs_for(source)
    return {
        "behavior_margins": margins,
        "predicted_behavior": max(PATTERNS, key=lambda pattern: margins[pattern]),
        "source_margin": margins[source],
        "source_output_mse": subject_output_mse(weights, source_weights, source_inputs),
        "target_margin": margins[target],
    }


def behavior_margin(weights: torch.Tensor, pattern: str) -> float:
    suite = evaluation_suite()
    positive = decoder_v1.sequence_tensor(suite["heldout"][pattern]["positive"])
    negative = decoder_v1.sequence_tensor(suite["heldout"][pattern]["negative"])
    with torch.no_grad():
        pos = torch.sigmoid(decoder_v1.subject_forward_flat_batch(weights.unsqueeze(0), positive)[0]).mean()
        neg = torch.sigmoid(decoder_v1.subject_forward_flat_batch(weights.unsqueeze(0), negative)[0]).mean()
    return float((pos - neg).item())


def heldout_inputs_for(pattern: str) -> torch.Tensor:
    suite = evaluation_suite()
    return torch.cat([
        decoder_v1.sequence_tensor(suite["heldout"][pattern]["positive"]),
        decoder_v1.sequence_tensor(suite["heldout"][pattern]["negative"]),
    ])


def evaluation_suite() -> Dict[str, Any]:
    global _EVAL_SUITE
    if _EVAL_SUITE is None:
        _EVAL_SUITE = build_suite(160, 64)
    return _EVAL_SUITE


def subject_output_mse(
    weights: torch.Tensor,
    source_weights: torch.Tensor,
    inputs: torch.Tensor,
) -> float:
    with torch.no_grad():
        edited = decoder_v1.subject_forward_flat_batch(weights.unsqueeze(0), inputs)[0]
        source = decoder_v1.subject_forward_flat_batch(source_weights.unsqueeze(0), inputs)[0]
    return float(F.mse_loss(edited, source).item())


def pareto_dominates_functional(
    control: Mapping[str, float],
    matched: Mapping[str, float],
    *,
    epsilon: float = 1e-8,
) -> bool:
    weakly_better = (
        float(control["target_margin"]) >= float(matched["target_margin"])
        and float(control["source_output_mse"]) <= float(matched["source_output_mse"])
    )
    strictly_better = (
        float(control["target_margin"]) > float(matched["target_margin"]) + epsilon
        or float(control["source_output_mse"]) < float(matched["source_output_mse"]) - epsilon
    )
    return weakly_better and strictly_better


def individual_passed(
    *,
    matched: Mapping[str, Any],
    nearest: Mapping[str, Any],
) -> bool:
    return (
        matched["target_prediction_pass"]
        and matched["target_margin"] > THRESHOLDS["min_per_record_matched_target_margin"]
        and matched["target_vs_source_margin"]
        > THRESHOLDS["min_per_record_target_vs_source_margin"]
        and matched["source_margin_change"]
        < THRESHOLDS["per_record_source_margin_change_must_be_below"]
        and matched["source_output_mse"] < nearest["source_output_mse"]
        and matched["pareto_undominated"]
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
        "mean_matched_minus_best_control_target_margin": mean(
            record["summary"]["matched_minus_best_control_target_margin"] for record in records
        ),
        "mean_matched_minus_nearest_train_target_margin": mean(
            record["summary"]["matched_minus_nearest_train_target_margin"] for record in records
        ),
        "mean_matched_minus_no_edit_target_margin": mean(
            record["summary"]["matched_minus_no_edit_target_margin"] for record in records
        ),
        "mean_matched_target_margin": mean(
            record["matched"]["target_margin"] for record in records
        ),
        "mean_matched_target_vs_source_margin": mean(
            record["matched"]["target_vs_source_margin"] for record in records
        ),
        "mean_nearest_train_minus_matched_source_output_mse": mean(
            record["summary"]["nearest_train_minus_matched_source_output_mse"]
            for record in records
        ),
        "mean_source_margin_change": mean(
            record["summary"]["source_margin_change"] for record in records
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
    require_equal(failures, aggregate["n"], THRESHOLDS["expected_record_count"], "aggregate n")
    require_at_least(
        failures,
        aggregate["individual_all_gate_pass_rate"],
        THRESHOLDS["min_aggregate_individual_pass_rate"],
        "aggregate individual pass rate",
    )
    require_at_least(
        failures,
        aggregate["target_prediction_rate"],
        THRESHOLDS["min_aggregate_target_prediction_rate"],
        "aggregate target prediction rate",
    )
    require_at_least(
        failures,
        aggregate["pareto_undominated_rate"],
        THRESHOLDS["min_aggregate_pareto_undominated_rate"],
        "aggregate Pareto-undominated rate",
    )
    require_greater(
        failures,
        aggregate["mean_matched_target_margin"],
        THRESHOLDS["min_mean_matched_target_margin"],
        "mean matched target margin",
    )
    require_greater(
        failures,
        aggregate["mean_matched_target_vs_source_margin"],
        THRESHOLDS["min_mean_matched_target_vs_source_margin"],
        "mean matched target-vs-source margin",
    )
    require_less(
        failures,
        aggregate["mean_source_margin_change"],
        THRESHOLDS["max_mean_source_margin_change"],
        "mean source margin change",
    )
    for key in (
        "mean_matched_minus_no_edit_target_margin",
        "mean_matched_minus_nearest_train_target_margin",
        "mean_matched_minus_best_control_target_margin",
        "mean_nearest_train_minus_matched_source_output_mse",
    ):
        require_greater(failures, aggregate[key], THRESHOLDS[f"min_{key}"], key)
    for direction, summary in by_direction.items():
        require_equal(
            failures,
            summary["n"],
            THRESHOLDS["expected_per_direction_count"],
            f"{direction} n",
        )
        require_at_least(
            failures,
            summary["individual_all_gate_pass_rate"],
            THRESHOLDS["min_direction_individual_pass_rate"],
            f"{direction} individual pass rate",
        )
        require_at_least(
            failures,
            summary["target_prediction_rate"],
            THRESHOLDS["min_direction_target_prediction_rate"],
            f"{direction} target prediction rate",
        )
    for record in records:
        if record["random_control_count"] != THRESHOLDS["random_controls_per_record"]:
            failures.append(f"{record['subject_id']} random control count mismatch")
    return failures


def require_equal(failures: list[str], value: int, expected: int, label: str) -> None:
    if value != expected:
        failures.append(f"{label} {value} != {expected}")


def require_at_least(failures: list[str], value: float, threshold: float, label: str) -> None:
    if value < threshold:
        failures.append(f"{label} {value:.6f} < {threshold:.6f}")


def require_greater(failures: list[str], value: float, threshold: float, label: str) -> None:
    if value <= threshold:
        failures.append(f"{label} {value:.6f} <= {threshold:.6f}")


def require_less(failures: list[str], value: float, threshold: float, label: str) -> None:
    if value >= threshold:
        failures.append(f"{label} {value:.6f} >= {threshold:.6f}")


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


def strip_control_weights(controls: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    return [strip_weight(control) for control in controls]


def strip_weight(record: Mapping[str, Any]) -> Dict[str, Any]:
    stripped = dict(record)
    stripped.pop("weights", None)
    return stripped


def train_only_statistics_hash(stats: Mapping[str, Any]) -> str:
    return stable_hash_json({
        "centroids": {
            pattern: tensor_to_hashable(stats["centroids"][pattern])
            for pattern in PATTERNS
        },
        "global_weight_centroid_norm": tensor_to_hashable(stats["global_weight_centroid_norm"]),
        "sig_mean": tensor_to_hashable(stats["sig_mean"]),
        "sig_std": tensor_to_hashable(stats["sig_std"]),
        "weight_mean": tensor_to_hashable(stats["weight_mean"]),
        "weight_std": tensor_to_hashable(stats["weight_std"]),
    })


def tensor_to_hashable(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().view(-1).tolist()]


if __name__ == "__main__":
    main()
