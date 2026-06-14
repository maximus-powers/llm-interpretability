"""V23 functional editing via probe-routed sparse subspace editors."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ZOO_ROOT = REPO_ROOT / "model_zoo"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_ZOO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import train_four_behavior_functional_weight_editing_v17_layerwise_rank1_tsv as v17  # noqa: E402
import train_four_behavior_functional_weight_editing_v21_behavioral_probe_residual_output_editor as v21  # noqa: E402
import train_four_behavior_functional_weight_editing_v22_component_activation_rank1_editor as v22  # noqa: E402


v16 = v17.v16
PATTERNS = v17.PATTERNS

SEED_BEHAVIOR_STRIDE = 100000
POOL_CONFIGS = {
    "train": {
        "base_seed": 120400000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 121400000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 122400000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}
DEFAULT_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v23_pools"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runs"
    / "four_behavior_functional_weight_editing_v23_probe_routed_sparse_subspace_editor"
)
INNER_VALIDATION_CHECKPOINT_FILENAME = "inner_validation_progress.json"
INNER_VALIDATION_PROGRESS_LOG_FILENAME = "inner_validation_progress.jsonl"
INNER_VALIDATION_PROGRESS_SCOPE = (
    "four_behavior_functional_weight_editing_v23_inner_validation_progress"
)
DEVELOPMENT_PROGRESS_LOG_FILENAME = "development_progress.jsonl"
INNER_VALIDATION_PROGRESS_ONLY_COMPATIBLE_IMPLEMENTATION_SHA256 = {
    "00825023d9cacf886c87ef8f2f5330484b17ce0e37ba1aa291b184a5e8f334f9": (
        "development evaluation progress logging only; inner-validation selection "
        "algorithm and constants unchanged"
    ),
}
PREREG_PATH = (
    REPO_ROOT
    / "docs"
    / "preregistrations"
    / "four_behavior_functional_weight_editing_v23_probe_routed_sparse_subspace_editor.md"
)
PLAN_PATH = (
    REPO_ROOT
    / "docs"
    / "preregistrations"
    / "four_behavior_functional_weight_editing_v23_probe_routed_sparse_subspace_editor_plan.md"
)
INNER_VALIDATION_AMENDMENT_PATH = (
    REPO_ROOT
    / "docs"
    / "preregistrations"
    / "four_behavior_functional_weight_editing_v23_inner_validation_compute_amendment.md"
)
SCRIPT_PATH = Path(__file__).resolve()
HELPER_TEST_PATH = (
    REPO_ROOT
    / "model_zoo"
    / "scripts"
    / "test_four_behavior_functional_weight_editing_v23_helpers.py"
)
SOURCE_POOL_SCOPE = "four_behavior_functional_weight_editing_v23_source_pool"
SOURCE_AUDIT_SCOPE = "four_behavior_functional_weight_editing_v23_source_pool_construction"
FINAL_REDACTED_SCOPE = (
    "redacted_final_functional_weight_editing_v23_source_pool_audit_surface_only"
)
DEVELOPMENT_SCOPE = (
    "four_behavior_functional_weight_editing_v23_probe_routed_sparse_subspace_editor_development"
)
FINAL_SCOPE = (
    "four_behavior_functional_weight_editing_v23_probe_routed_sparse_subspace_editor_final"
)
EDITOR_METHOD = "probe_routed_sparse_subspace_editor_v23"
PLAN_SHA256 = "50a26e376b39b3f3d93d8bdf894534f1571ab444491c044de324e16ae6c671e8"
INNER_VALIDATION_AMENDMENT_SHA256 = (
    "41941156935739d0a66763e1e9bff51940256b96f6064b788284163c5d6f437e"
)
PASSING_DEVELOPMENT_NEXT_ACTION = "run_hash_bound_final_after_reviewer_authorization"
FAILING_DEVELOPMENT_NEXT_ACTION = "log_negative_development_result_do_not_open_final_raw"
V20_TANGENT_CONTROL_METHOD = "signature_conditioned_tangent_nullspace_editor_v20"
V23_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"
PRIOR_FINAL_RAW_PATHS = {
    v17.v16.v15.V15_FINAL_RAW,
    v17.v16.V16_FINAL_RAW,
    v17.V17_FINAL_RAW,
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v18_pools" / "final_subjects.json",
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v19_pools" / "final_subjects.json",
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v20_pools" / "final_subjects.json",
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v21_pools" / "final_subjects.json",
    v22.V22_FINAL_RAW,
}

SOURCE_WEIGHT_DIM = 345
SIGNATURE_DIM = 560
SIGNATURE_TOP_K = 8
SIGNATURE_TEMPERATURE = 1.0
DESIRED_LOGIT_MAGNITUDE = 2.0
COMPONENT_RANK1_LAMBDAS = [0.01, 0.1, 1.0, 10.0]
COMPONENT_RANK1_SCALES = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]
HIDDEN_LAYERS = [0, 1, 2, 3, 4]
HIDDEN_NORM_CAP_MULTIPLIER = 0.5
ROW_GROUP_ORDER = [
    "target_support",
    "conflict_support",
    "compatible_support",
    "probe_target",
    "probe_compatible",
]
SPARSE_K_VALUES = [1, 2, 3]
LAMBDA_RANK1_GRID = [0.01, 0.1, 1.0, 10.0]
LAMBDA_SOLVE_GRID = [0.01, 0.1, 1.0, 10.0]
COMPATIBLE_WEIGHT_GRID = [0.5, 1.0, 2.0, 4.0]
PROBE_CENTROID_WEIGHT_GRID = [0.0, 0.25, 0.5, 1.0]
CONTROL_PENALTY_WEIGHT_GRID = [0.0, 0.5, 1.0, 2.0]
CAP_MULTIPLIER_GRID = [0.25, 0.5, 1.0]
SPARSE_POST_SCALE_GRID = [0.25, 0.5, 0.75, 1.0]
INNER_VALIDATION_RUNG_RECORD_BUDGETS = [12, 48, 156]
INNER_VALIDATION_RUNG_SUBJECTS_PER_BEHAVIOR = [
    budget // (len(PATTERNS) * (len(PATTERNS) - 1))
    for budget in INNER_VALIDATION_RUNG_RECORD_BUDGETS
]
INNER_VALIDATION_RUNG_SURVIVORS = [32, 8, 1]
INNER_VALIDATION_EVALUATED_CONFIG_COUNT = 128
INNER_VALIDATION_TOTAL_CONFIG_COUNT = (
    len(SPARSE_K_VALUES)
    * len(LAMBDA_RANK1_GRID)
    * len(LAMBDA_SOLVE_GRID)
    * len(COMPATIBLE_WEIGHT_GRID)
    * len(PROBE_CENTROID_WEIGHT_GRID)
    * len(CONTROL_PENALTY_WEIGHT_GRID)
    * len(CAP_MULTIPLIER_GRID)
)
NULLSPACE_RELATIVE_CUTOFF = 1e-4
NULLSPACE_ABSOLUTE_CUTOFF = 1e-6
MASK_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
RIDGE_LAMBDAS = [0.01, 0.1, 1.0, 10.0]
PRIOR_LAMBDAS = [0.0, 0.01, 0.1, 1.0]
ACTIVATION_SCALE_GRID = [0.0, 0.5, 1.0]
POST_SCALE_GRID = [0.5, 0.75, 1.0, 1.25]
MAX_TANGENT_DELTA_NORM = 8.0
RANDOM_CONTROLS_PER_RECORD = 20
EXPECTED_CONTROLS_PER_RECORD = 32
RANDOM_CONTROL_EPS = 1e-12
PARETO_EPSILON = 1e-9

LOSS_WEIGHTS = {
    "compatible_source_mse": 1.0,
    "conflict_bce": 1.0,
    "target_bce": 1.0,
}
THRESHOLDS = {
    "expected_record_count": 288,
    "expected_controls_per_record": 32,
    "expected_random_controls_per_record": 20,
    "min_aggregate_target_prediction_rate": 0.85,
    "min_aggregate_individual_pass_rate": 0.85,
    "min_aggregate_pareto_rate": 0.85,
    "min_aggregate_target_margin": 0.25,
    "min_aggregate_conflict_target_accuracy": 0.75,
    "min_aggregate_conflict_target_accuracy_improvement": 0.25,
    "min_aggregate_best_control_target_margin_advantage": 0.02,
    "min_aggregate_target_label_target_margin_advantage": 0.02,
    "min_aggregate_shuffled_signature_target_margin_advantage": 0.05,
    "min_aggregate_output_layer_no_signature_target_margin_advantage": 0.02,
    "min_aggregate_v16_target_margin_advantage": 0.02,
    "min_aggregate_v17_target_margin_advantage": 0.02,
    "min_aggregate_v20_target_margin_advantage": 0.02,
    "min_aggregate_v21_target_margin_advantage": 0.02,
    "min_aggregate_v22_target_margin_advantage": 0.02,
    "min_direction_target_prediction_rate": 0.65,
    "min_direction_individual_pass_rate": 0.65,
    "min_direction_pareto_rate": 0.75,
    "min_direction_target_margin": 0.15,
    "min_direction_output_layer_no_signature_target_margin_advantage": 0.01,
    "min_direction_v16_target_margin_advantage": 0.01,
    "min_direction_v17_target_margin_advantage": 0.01,
    "min_direction_v20_target_margin_advantage": 0.01,
    "min_direction_v21_target_margin_advantage": 0.01,
    "min_direction_v22_target_margin_advantage": 0.01,
    "min_per_record_target_margin": 0.25,
    "min_per_record_conflict_target_accuracy": 0.75,
    "min_per_record_conflict_target_accuracy_improvement": 0.25,
    "min_per_record_control_target_margin_advantage": 0.02,
    "min_per_record_control_compatible_mse_advantage": -0.02,
}

PROOF_CRITICAL_CONTROL_TYPES = [
    "no_edit",
    "output_layer_no_signature_support_optimizer",
    "v17_layerwise_rank1_tsv",
    "v16_output_layer_conceptor",
    "v20_tangent_nullspace_editor_recomputed",
    "v21_behavioral_probe_residual_output_editor_recomputed",
    "v22_component_activation_rank1_editor_recomputed",
    "no_probe_sparse_subspace_editor",
    "source_probe_sparse_subspace_editor",
    "shuffled_probe_sparse_subspace_editor",
    "target_label_only_sparse_subspace_editor",
    "nearest_target_sparse_subspace_editor",
]
ADVANTAGE_CONTROL_TYPES = {
    "no_signature": "no_probe_sparse_subspace_editor",
    "target_label": "target_label_only_sparse_subspace_editor",
    "source_signature": "source_probe_sparse_subspace_editor",
    "shuffled_signature": "shuffled_probe_sparse_subspace_editor",
    "v17": "v17_layerwise_rank1_tsv",
    "v16": "v16_output_layer_conceptor",
    "v20": "v20_tangent_nullspace_editor_recomputed",
    "v21": "v21_behavioral_probe_residual_output_editor_recomputed",
    "v22": "v22_component_activation_rank1_editor_recomputed",
    "output_layer_no_signature": "output_layer_no_signature_support_optimizer",
    "nearest_target": "nearest_target_sparse_subspace_editor",
}

FINAL_COMBINED_SUMMARY_ALLOWED_KEYS = {
    "accepted_counts_by_behavior",
    "pool_file_sha256",
    "pool_redacted_payload_sha256",
}
FINAL_REDACTED_ALLOWED_TOP_LEVEL_KEYS = {
    "behavior_suite_hashes",
    "candidate_pool_summary_hash",
    "claim_scope",
    "config_hash",
    "pool",
    "pool_file_sha256",
    "pool_redacted_payload_sha256",
    "summary",
    "summary_payload_sha256",
}
FINAL_REDACTED_ALLOWED_SUMMARY_KEYS = {
    "accepted_counts_by_behavior",
    "max_selected_train_vs_heldout_overlap_count",
}


stable_hash_json = v17.stable_hash_json
tensor_to_hashable = v17.tensor_to_hashable


def constants_payload() -> dict[str, Any]:
    return {
        "cap_multiplier_grid": CAP_MULTIPLIER_GRID,
        "compatible_weight_grid": COMPATIBLE_WEIGHT_GRID,
        "control_penalty_weight_grid": CONTROL_PENALTY_WEIGHT_GRID,
        "component_rank1_hidden_layers": HIDDEN_LAYERS,
        "component_rank1_lambdas": LAMBDA_RANK1_GRID,
        "component_rank1_norm_cap_multiplier": HIDDEN_NORM_CAP_MULTIPLIER,
        "desired_logit_magnitude": DESIRED_LOGIT_MAGNITUDE,
        "editor_method": EDITOR_METHOD,
        "expected_controls_per_record": EXPECTED_CONTROLS_PER_RECORD,
        "inner_validation_amendment_sha256": INNER_VALIDATION_AMENDMENT_SHA256,
        "inner_validation_rung_record_budgets": INNER_VALIDATION_RUNG_RECORD_BUDGETS,
        "inner_validation_rung_subjects_per_behavior": INNER_VALIDATION_RUNG_SUBJECTS_PER_BEHAVIOR,
        "inner_validation_rung_survivors": INNER_VALIDATION_RUNG_SURVIVORS,
        "inner_validation_evaluated_config_count": INNER_VALIDATION_EVALUATED_CONFIG_COUNT,
        "inner_validation_total_config_count": INNER_VALIDATION_TOTAL_CONFIG_COUNT,
        "lambda_solve_grid": LAMBDA_SOLVE_GRID,
        "loss_weights": LOSS_WEIGHTS,
        "output_layer_theta_dim": 9,
        "plan_sha256": PLAN_SHA256,
        "probe_centroid_weight_grid": PROBE_CENTROID_WEIGHT_GRID,
        "random_controls_per_record": RANDOM_CONTROLS_PER_RECORD,
        "sparse_k_values": SPARSE_K_VALUES,
        "sparse_post_scale_grid": SPARSE_POST_SCALE_GRID,
        "source_weight_dim": SOURCE_WEIGHT_DIM,
        "thresholds": THRESHOLDS,
        "v20_tangent_control_method": V20_TANGENT_CONTROL_METHOD,
    }


def behavior_centroids(records: Sequence[Mapping[str, Any]], train_stats: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    grouped = v17.records_by_behavior(records)
    centroids = {}
    for behavior, items in grouped.items():
        signatures = [v17.normalized_signature(item, train_stats) for item in items]
        centroids[behavior] = torch.stack(signatures).mean(dim=0)
    return centroids


def full_train_statistics_hash_payload(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sparse_subspace_config_hash": stats.get("selected_sparse_config_hash"),
        "constants": constants_payload(),
        "global_hidden_centroids": [
            tensor_to_hashable(value) for value in stats.get("global_hidden_centroids", [])
        ],
        "hidden_descriptor_hashes": stats.get("hidden_descriptor_hashes"),
        "hidden_target_centroids": {
            behavior: [tensor_to_hashable(item) for item in values]
            for behavior, values in sorted(stats.get("hidden_target_centroids", {}).items())
        },
        "probe_examples_hash": stats.get("probe_examples_hash", "missing"),
        "sig_mean": tensor_to_hashable(stats["sig_mean"]),
        "sig_std": tensor_to_hashable(stats["sig_std"]),
        "signature_centroids": {
            key: tensor_to_hashable(value)
            for key, value in sorted(stats.get("signature_centroids", {}).items())
        },
        "target_probe_logit_centroids": {
            key: tensor_to_hashable(value)
            for key, value in sorted(stats.get("target_probe_logit_centroids", {}).items())
        },
        "target_probe_logit_centroid_hashes": stats.get("target_probe_logit_centroid_hashes"),
        "thresholds": THRESHOLDS,
        "train_subject_ids": [
            str(item["subject_id"]) for item in stats.get("train_subjects", [])
        ],
        "v16_baseline_train_statistics_hash": stats.get("v16_baseline_train_statistics_hash"),
        "v17_baseline_train_statistics_hash": stats.get("v17_baseline_train_statistics_hash"),
        "v21_baseline_train_statistics_hash": stats.get("v21_baseline_train_statistics_hash"),
        "v22_baseline_train_statistics_hash": stats.get("v22_baseline_train_statistics_hash"),
        "selected_sparse_config_hash": stats.get("selected_sparse_config_hash"),
    }


def sorted_train_subjects(subjects: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(subjects, key=lambda item: (v16.subject_behavior(item), str(item["subject_id"])))


def behavior_for_record(record: Mapping[str, Any]) -> str:
    if "behavior" in record:
        return str(record["behavior"])
    return str(v16.subject_behavior(record))


def inner_train_validation_split(subjects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {behavior: [] for behavior in PATTERNS}
    for subject in subjects:
        behavior = behavior_for_record(subject)
        if behavior in grouped:
            grouped[behavior].append(subject)
    inner_train_by_behavior = {}
    inner_validation_by_behavior = {}
    for behavior in PATTERNS:
        sorted_records = sorted(
            grouped[behavior],
            key=lambda item: (
                stable_hash_json({
                    "scope": "four_behavior_functional_weight_editing_v23_inner_split",
                    "behavior": behavior,
                    "subject_id": str(item["subject_id"]),
                }),
                str(item["subject_id"]),
            ),
        )
        inner_train_by_behavior[behavior] = sorted_records[:51]
        inner_validation_by_behavior[behavior] = sorted_records[51:]
    inner_train_subjects = [
        record
        for behavior in PATTERNS
        for record in inner_train_by_behavior[behavior]
    ]
    inner_validation_subjects = [
        record
        for behavior in PATTERNS
        for record in inner_validation_by_behavior[behavior]
    ]
    train_ids = {str(item["subject_id"]) for item in inner_train_subjects}
    validation_ids = {str(item["subject_id"]) for item in inner_validation_subjects}
    if train_ids & validation_ids:
        raise ValueError("inner train and validation subjects overlap")
    return {
        "inner_train_by_behavior": inner_train_by_behavior,
        "inner_validation_by_behavior": inner_validation_by_behavior,
        "inner_train_subjects": inner_train_subjects,
        "inner_validation_subjects": inner_validation_subjects,
    }


def sort_inner_validation_configs(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            bool(item.get("invalid", False)),
            -float(item.get("target_prediction_rate", float("-inf"))),
            -float(item.get("pareto_undominated_rate", float("-inf"))),
            -float(item.get("mean_matched_minus_best_control_target_margin", float("-inf"))),
            -float(item.get("mean_matched_minus_shuffled_signature_target_margin", float("-inf"))),
            -float(item.get("mean_target_margin", float("-inf"))),
            float(item.get("mean_compatible_source_mse", float("inf"))),
            float(item.get("effective_zero_coefficient_rate", float("inf"))),
            float(item.get("mean_hidden_delta_norm", float("inf"))),
            int(item.get("config_index", 0)),
        ),
    )


def select_inner_validation_config(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("no inner-validation config candidates")
    return sort_inner_validation_configs(candidates)[0]


def inner_validation_candidate_from_result(
    *,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
    expected_record_count: int = 156,
) -> dict[str, Any]:
    invalid_reasons = []
    records = list(result.get("records", []))
    if int(result.get("record_count", -1)) != int(expected_record_count):
        invalid_reasons.append(f"record_count_not_{int(expected_record_count)}")
    expected_controls = set(PROOF_CRITICAL_CONTROL_TYPES)
    for index, record in enumerate(records):
        control_types = [str(control.get("control_type")) for control in record.get("controls", [])]
        if set(control_types) != expected_controls or len(control_types) != len(expected_controls):
            invalid_reasons.append(f"proof_control_mismatch:{index}")
            break
    aggregate = result.get("aggregate", {})
    metric_values = {
        "target_prediction_rate": float(aggregate.get("target_prediction_rate", float("nan"))),
        "pareto_undominated_rate": float(aggregate.get("pareto_undominated_rate", float("nan"))),
        "mean_matched_minus_best_control_target_margin": float(
            aggregate.get("mean_matched_minus_best_control_target_margin", float("nan"))
        ),
        "mean_matched_minus_shuffled_signature_target_margin": float(
            aggregate.get("mean_matched_minus_shuffled_signature_target_margin", float("nan"))
        ),
        "mean_target_margin": float(aggregate.get("mean_target_margin", float("nan"))),
        "mean_compatible_source_mse": mean([
            float(record.get("matched", {}).get("compatible_source_output_mse", float("nan")))
            for record in records
        ]) if records else float("nan"),
        "effective_zero_coefficient_rate": mean([
            1.0 if record.get("matched", {}).get("editor", {}).get("scale_0_selected") else 0.0
            for record in records
        ]) if records else float("nan"),
        "mean_hidden_delta_norm": mean([
            float(record.get("matched", {}).get("editor", {}).get(
                "hidden_delta_norm",
                record.get("matched", {}).get("delta_norm", float("nan")),
            ))
            for record in records
        ]) if records else float("nan"),
    }
    for key, value in metric_values.items():
        if not math.isfinite(value):
            invalid_reasons.append(f"nonfinite:{key}")
    return {
        **dict(config),
        **metric_values,
        "inner_validation_record_count": int(result.get("record_count", -1)),
        "invalid": bool(invalid_reasons),
        "invalid_reasons": invalid_reasons,
    }


def train_stats_with_selected_sparse_config(
    base_stats: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    stats = dict(base_stats)
    selected_config = dict(config)
    stats["selected_sparse_config"] = selected_config
    stats["selected_sparse_config_hash"] = selected_config["config_hash"]
    try:
        stats["train_statistics_hash"] = stable_hash_json(full_train_statistics_hash_payload(stats))
    except KeyError:
        stats["train_statistics_hash"] = stable_hash_json({
            "scope": "four_behavior_functional_weight_editing_v23_train_stats_config_binding",
            "base_train_statistics_hash": base_stats.get("train_statistics_hash"),
            "selected_sparse_config_hash": selected_config["config_hash"],
        })
    return stats


def ordered_inner_validation_subjects_for_budget(
    subjects: Sequence[Mapping[str, Any]],
    *,
    record_budget: int,
) -> list[Mapping[str, Any]]:
    records_per_balanced_round = len(PATTERNS) * (len(PATTERNS) - 1)
    if int(record_budget) % records_per_balanced_round != 0:
        raise ValueError(
            "inner-validation record budget must be divisible by "
            f"{records_per_balanced_round}"
        )
    subjects_per_behavior = int(record_budget) // records_per_balanced_round
    selected = []
    for behavior in PATTERNS:
        ordered = sorted(
            [
                subject
                for subject in subjects
                if behavior_for_record(subject) == behavior
            ],
            key=lambda item: str(item["subject_id"]),
        )
        if len(ordered) < subjects_per_behavior:
            raise ValueError(
                f"inner-validation budget requires {subjects_per_behavior} "
                f"{behavior} subjects, got {len(ordered)}"
            )
        selected.extend(ordered[:subjects_per_behavior])
    return selected


def inner_validation_config_hashes_hash(configs: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash_json([str(config["config_hash"]) for config in configs])


def inner_validation_implementation_sha256() -> str:
    return v16.v15.v1.sha256_file(SCRIPT_PATH)


def inner_validation_constants_sha256() -> str:
    return stable_hash_json(constants_payload())


def is_inner_validation_checkpoint_implementation_compatible(
    implementation_sha256: Any,
) -> bool:
    implementation_sha256 = str(implementation_sha256)
    return (
        implementation_sha256 == inner_validation_implementation_sha256()
        or implementation_sha256 in
        INNER_VALIDATION_PROGRESS_ONLY_COMPATIBLE_IMPLEMENTATION_SHA256
    )


def new_inner_validation_progress_checkpoint(
    configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "scope": INNER_VALIDATION_PROGRESS_SCOPE,
        "amendment_sha256": INNER_VALIDATION_AMENDMENT_SHA256,
        "plan_sha256": PLAN_SHA256,
        "implementation_sha256": inner_validation_implementation_sha256(),
        "constants_sha256": inner_validation_constants_sha256(),
        "total_config_count": len(configs),
        "all_config_hashes_hash": inner_validation_config_hashes_hash(configs),
        "rung_record_budgets": INNER_VALIDATION_RUNG_RECORD_BUDGETS,
        "rung_survivors": INNER_VALIDATION_RUNG_SURVIVORS,
        "rungs": [],
        "status": "running",
        "updated_at_unix": time.time(),
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def record_development_progress_event(
    progress_log_path: Path,
    *,
    event: str,
    started_at_monotonic: float,
    extra: Mapping[str, Any] | None = None,
    now_monotonic: Any | None = None,
) -> None:
    monotonic = time.monotonic if now_monotonic is None else now_monotonic
    now = monotonic()
    payload = {
        "event": event,
        "elapsed_seconds": now - float(started_at_monotonic),
        "updated_at_unix": time.time(),
    }
    if extra:
        payload.update(dict(extra))
    append_jsonl(progress_log_path, payload)


def load_inner_validation_progress_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if payload.get("scope") != INNER_VALIDATION_PROGRESS_SCOPE:
        return None
    if payload.get("amendment_sha256") != INNER_VALIDATION_AMENDMENT_SHA256:
        return None
    if payload.get("plan_sha256") != PLAN_SHA256:
        return None
    if not is_inner_validation_checkpoint_implementation_compatible(
        payload.get("implementation_sha256")
    ):
        return None
    if payload.get("constants_sha256") != inner_validation_constants_sha256():
        return None
    if payload.get("rung_record_budgets") != INNER_VALIDATION_RUNG_RECORD_BUDGETS:
        return None
    if payload.get("rung_survivors") != INNER_VALIDATION_RUNG_SURVIVORS:
        return None
    return payload


def get_or_create_inner_validation_rung_progress(
    *,
    checkpoint: dict[str, Any],
    rung_index: int,
    record_budget: int,
    survivor_count: int,
    active_configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    active_config_hashes_hash = inner_validation_config_hashes_hash(active_configs)
    for rung in checkpoint.setdefault("rungs", []):
        if (
            rung.get("rung_index") == int(rung_index)
            and rung.get("record_budget") == int(record_budget)
            and rung.get("active_config_hashes_hash") == active_config_hashes_hash
        ):
            return rung
    rung = {
        "rung_index": int(rung_index),
        "record_budget": int(record_budget),
        "survivor_count": int(survivor_count),
        "active_config_count": len(active_configs),
        "active_config_hashes_hash": active_config_hashes_hash,
        "candidates": [],
        "completed_count": 0,
        "invalid_count": 0,
        "status": "running",
        "updated_at_unix": time.time(),
    }
    checkpoint["rungs"].append(rung)
    return rung


def inner_validation_completed_candidates_by_hash(
    *,
    checkpoint: Mapping[str, Any] | None,
    rung_index: int,
    record_budget: int,
    active_configs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not checkpoint:
        return {}
    if checkpoint.get("scope") != INNER_VALIDATION_PROGRESS_SCOPE:
        return {}
    if checkpoint.get("amendment_sha256") != INNER_VALIDATION_AMENDMENT_SHA256:
        return {}
    if checkpoint.get("plan_sha256") != PLAN_SHA256:
        return {}
    if not is_inner_validation_checkpoint_implementation_compatible(
        checkpoint.get("implementation_sha256")
    ):
        return {}
    if checkpoint.get("constants_sha256") != inner_validation_constants_sha256():
        return {}
    active_config_hashes_hash = inner_validation_config_hashes_hash(active_configs)
    active_hashes = {str(config["config_hash"]) for config in active_configs}
    for rung in checkpoint.get("rungs", []):
        if (
            rung.get("rung_index") != int(rung_index)
            or rung.get("record_budget") != int(record_budget)
            or rung.get("active_config_hashes_hash") != active_config_hashes_hash
        ):
            continue
        completed: dict[str, dict[str, Any]] = {}
        for candidate in rung.get("candidates", []):
            config_hash = str(candidate.get("config_hash", ""))
            if config_hash in active_hashes:
                completed[config_hash] = dict(candidate)
        return completed
    return {}


def record_inner_validation_candidate_progress(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    progress_log_path: Path,
    rung: dict[str, Any],
    candidate: Mapping[str, Any],
    event: str,
    elapsed_seconds: float | None = None,
) -> None:
    config_hash = str(candidate["config_hash"])
    existing = {
        str(item.get("config_hash")): index
        for index, item in enumerate(rung.setdefault("candidates", []))
    }
    if config_hash in existing:
        rung["candidates"][existing[config_hash]] = dict(candidate)
    else:
        rung["candidates"].append(dict(candidate))
    rung["completed_count"] = len(rung["candidates"])
    rung["invalid_count"] = sum(1 for item in rung["candidates"] if item.get("invalid"))
    rung["updated_at_unix"] = time.time()
    checkpoint["updated_at_unix"] = rung["updated_at_unix"]
    write_json_atomic(checkpoint_path, checkpoint)
    append_jsonl(progress_log_path, {
        "event": event,
        "rung_index": rung["rung_index"],
        "record_budget": rung["record_budget"],
        "completed_count": rung["completed_count"],
        "active_config_count": rung["active_config_count"],
        "config_hash": config_hash,
        "invalid": bool(candidate.get("invalid", False)),
        "elapsed_seconds": elapsed_seconds,
        "updated_at_unix": rung["updated_at_unix"],
    })


_INNER_WORKER_BASE_TRAIN_STATS: Mapping[str, Any] | None = None
_INNER_WORKER_RUNG_SUBJECTS: Sequence[Mapping[str, Any]] | None = None
_INNER_WORKER_RECORD_BUDGET = 0


def inner_validation_candidate_for_config(
    *,
    config: Mapping[str, Any],
    base_train_stats: Mapping[str, Any],
    rung_subjects: Sequence[Mapping[str, Any]],
    record_budget: int,
) -> dict[str, Any]:
    try:
        train_stats = train_stats_with_selected_sparse_config(base_train_stats, config)
        result = evaluate_subjects(
            subjects=rung_subjects,
            train_stats=train_stats,
            random_controls=0,
            parallel=False,
            max_workers=None,
        )
        return inner_validation_candidate_from_result(
            config=config,
            result=result,
            expected_record_count=int(record_budget),
        )
    except Exception as exc:  # fail closed for this config, continue sweep
        return {
            **dict(config),
            "inner_validation_record_count": -1,
            "invalid": True,
            "invalid_reasons": [f"exception:{type(exc).__name__}:{exc}"],
        }


def _init_inner_validation_config_worker(
    base_train_stats: Mapping[str, Any],
    rung_subjects: Sequence[Mapping[str, Any]],
    record_budget: int,
) -> None:
    global _INNER_WORKER_BASE_TRAIN_STATS, _INNER_WORKER_RUNG_SUBJECTS, _INNER_WORKER_RECORD_BUDGET
    torch.set_num_threads(1)
    _INNER_WORKER_BASE_TRAIN_STATS = base_train_stats
    _INNER_WORKER_RUNG_SUBJECTS = rung_subjects
    _INNER_WORKER_RECORD_BUDGET = int(record_budget)


def _evaluate_inner_validation_config_worker(config: Mapping[str, Any]) -> dict[str, Any]:
    if _INNER_WORKER_BASE_TRAIN_STATS is None or _INNER_WORKER_RUNG_SUBJECTS is None:
        raise RuntimeError("inner-validation config worker not initialized")
    return inner_validation_candidate_for_config(
        config=config,
        base_train_stats=_INNER_WORKER_BASE_TRAIN_STATS,
        rung_subjects=_INNER_WORKER_RUNG_SUBJECTS,
        record_budget=_INNER_WORKER_RECORD_BUDGET,
    )


def select_sparse_config_with_inner_validation(
    subjects: Sequence[Mapping[str, Any]],
    *,
    max_workers: int | None = None,
    progress_dir: Path | None = None,
) -> dict[str, Any]:
    split = inner_train_validation_split(subjects)
    inner_train_subjects = split["inner_train_subjects"]
    inner_validation_subjects = split["inner_validation_subjects"]
    full_configs = iter_sparse_subspace_configs()
    configs = inner_validation_evaluated_config_subset(full_configs)
    evaluated_config_subset_hash = inner_validation_evaluated_config_subset_hash(configs)
    base_train_stats = fit_v23_train_statistics(
        inner_train_subjects,
        include_models=True,
        include_baseline_stats=True,
        allow_default_small_pool=True,
        selected_sparse_config=configs[0],
        run_inner_validation=False,
    )
    active_configs = list(configs)
    checkpoint_path: Path | None = None
    progress_log_path: Path | None = None
    checkpoint = new_inner_validation_progress_checkpoint(configs)
    loaded_checkpoint: dict[str, Any] | None = None
    if progress_dir is not None:
        progress_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = progress_dir / INNER_VALIDATION_CHECKPOINT_FILENAME
        progress_log_path = progress_dir / INNER_VALIDATION_PROGRESS_LOG_FILENAME
        loaded_checkpoint = load_inner_validation_progress_checkpoint(checkpoint_path)
        if (
            loaded_checkpoint
            and loaded_checkpoint.get("total_config_count") == len(configs)
            and loaded_checkpoint.get("all_config_hashes_hash")
            == inner_validation_config_hashes_hash(configs)
        ):
            checkpoint = copy.deepcopy(loaded_checkpoint)
            append_jsonl(progress_log_path, {
                "event": "inner_validation_resume",
                "completed_rung_count": sum(
                    1 for rung in checkpoint.get("rungs", []) if rung.get("status") == "completed"
                ),
                "updated_at_unix": time.time(),
            })
        else:
            write_json_atomic(checkpoint_path, checkpoint)
            append_jsonl(progress_log_path, {
                "event": "inner_validation_start",
                "total_config_count": len(configs),
                "rung_record_budgets": INNER_VALIDATION_RUNG_RECORD_BUDGETS,
                "rung_survivors": INNER_VALIDATION_RUNG_SURVIVORS,
                "updated_at_unix": time.time(),
            })
    rung_summaries = []
    selected: dict[str, Any] | None = None
    for rung_index, (record_budget, survivor_count) in enumerate(zip(
        INNER_VALIDATION_RUNG_RECORD_BUDGETS,
        INNER_VALIDATION_RUNG_SURVIVORS,
    )):
        rung_subjects = ordered_inner_validation_subjects_for_budget(
            inner_validation_subjects,
            record_budget=int(record_budget),
        )
        candidates = []
        rung_progress: dict[str, Any] | None = None
        completed_by_hash: dict[str, dict[str, Any]] = {}
        if checkpoint_path is not None and progress_log_path is not None:
            rung_progress = get_or_create_inner_validation_rung_progress(
                checkpoint=checkpoint,
                rung_index=int(rung_index),
                record_budget=int(record_budget),
                survivor_count=int(survivor_count),
                active_configs=active_configs,
            )
            write_json_atomic(checkpoint_path, checkpoint)
            completed_by_hash = inner_validation_completed_candidates_by_hash(
                checkpoint=checkpoint,
                rung_index=int(rung_index),
                record_budget=int(record_budget),
                active_configs=active_configs,
            )
            append_jsonl(progress_log_path, {
                "event": "rung_start",
                "rung_index": int(rung_index),
                "record_budget": int(record_budget),
                "active_config_count": len(active_configs),
                "resumed_candidate_count": len(completed_by_hash),
                "updated_at_unix": time.time(),
            })
        pending_configs = []
        for config in active_configs:
            config_hash = str(config["config_hash"])
            if config_hash in completed_by_hash:
                candidates.append(copy.deepcopy(completed_by_hash[config_hash]))
                continue
            pending_configs.append(config)
        if pending_configs:
            contract = multiprocessing_contract(max_workers=max_workers)
            if int(contract["max_workers"]) == 1:
                for config in pending_configs:
                    config_hash = str(config["config_hash"])
                    started_at = time.monotonic()
                    if progress_log_path is not None:
                        append_jsonl(progress_log_path, {
                            "event": "candidate_start",
                            "rung_index": int(rung_index),
                            "record_budget": int(record_budget),
                            "config_hash": config_hash,
                            "updated_at_unix": time.time(),
                        })
                    candidate = inner_validation_candidate_for_config(
                        config=config,
                        base_train_stats=base_train_stats,
                        rung_subjects=rung_subjects,
                        record_budget=int(record_budget),
                    )
                    candidates.append(candidate)
                    if (
                        checkpoint_path is not None
                        and progress_log_path is not None
                        and rung_progress is not None
                    ):
                        record_inner_validation_candidate_progress(
                            checkpoint=checkpoint,
                            checkpoint_path=checkpoint_path,
                            progress_log_path=progress_log_path,
                            rung=rung_progress,
                            candidate=candidate,
                            event="candidate_completed",
                            elapsed_seconds=time.monotonic() - started_at,
                        )
            else:
                context = mp.get_context(contract["start_method"])
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=int(contract["max_workers"]),
                    mp_context=context,
                    initializer=_init_inner_validation_config_worker,
                    initargs=(base_train_stats, rung_subjects, int(record_budget)),
                ) as executor:
                    future_to_config = {}
                    future_started_at = {}
                    for config in pending_configs:
                        config_hash = str(config["config_hash"])
                        if progress_log_path is not None:
                            append_jsonl(progress_log_path, {
                                "event": "candidate_start",
                                "rung_index": int(rung_index),
                                "record_budget": int(record_budget),
                                "config_hash": config_hash,
                                "updated_at_unix": time.time(),
                            })
                        future = executor.submit(_evaluate_inner_validation_config_worker, config)
                        future_to_config[future] = config
                        future_started_at[future] = time.monotonic()
                    for future in concurrent.futures.as_completed(future_to_config):
                        config = future_to_config[future]
                        try:
                            candidate = future.result()
                        except Exception as exc:
                            candidate = {
                                **dict(config),
                                "inner_validation_record_count": -1,
                                "invalid": True,
                                "invalid_reasons": [f"exception:{type(exc).__name__}:{exc}"],
                            }
                        candidates.append(candidate)
                        if (
                            checkpoint_path is not None
                            and progress_log_path is not None
                            and rung_progress is not None
                        ):
                            record_inner_validation_candidate_progress(
                                checkpoint=checkpoint,
                                checkpoint_path=checkpoint_path,
                                progress_log_path=progress_log_path,
                                rung=rung_progress,
                                candidate=candidate,
                                event="candidate_completed",
                                elapsed_seconds=time.monotonic() - future_started_at[future],
                            )
        ranked = sort_inner_validation_configs(candidates)
        kept = ranked[: min(int(survivor_count), len(ranked))]
        if (
            checkpoint_path is not None
            and progress_log_path is not None
            and rung_progress is not None
        ):
            rung_progress["status"] = "completed"
            rung_progress["candidate_count"] = len(candidates)
            rung_progress["invalid_count"] = sum(
                1 for candidate in candidates if candidate.get("invalid")
            )
            rung_progress["kept_config_hashes"] = [
                str(candidate["config_hash"]) for candidate in kept
            ]
            rung_progress["updated_at_unix"] = time.time()
            checkpoint["updated_at_unix"] = rung_progress["updated_at_unix"]
            write_json_atomic(checkpoint_path, checkpoint)
            append_jsonl(progress_log_path, {
                "event": "rung_completed",
                "rung_index": int(rung_index),
                "record_budget": int(record_budget),
                "candidate_count": len(candidates),
                "invalid_count": sum(1 for candidate in candidates if candidate.get("invalid")),
                "survivor_count": len(kept),
                "updated_at_unix": time.time(),
            })
        rung_summaries.append({
            "candidate_count": len(candidates),
            "invalid_count": sum(1 for candidate in candidates if candidate.get("invalid")),
            "record_budget": int(record_budget),
            "rung_index": int(rung_index),
            "survivor_count": len(kept),
        })
        active_configs = [dict(config) for config in kept]
        selected = dict(kept[0]) if kept else None
    if selected is None:
        raise ValueError("no V23 inner-validation candidates evaluated")
    if selected.get("invalid"):
        raise ValueError(
            "all V23 inner-validation configs invalid; selected invalid reasons: "
            + ", ".join(selected.get("invalid_reasons", []))
        )
    selected["inner_validation_selection_hash"] = stable_hash_json({
        "scope": "four_behavior_functional_weight_editing_v23_inner_validation_selection",
        "amendment_sha256": INNER_VALIDATION_AMENDMENT_SHA256,
        "total_config_count": len(configs),
        "selected_config_hash": selected.get("config_hash"),
        "rung_summaries": rung_summaries,
        "metrics": {
            key: selected[key]
            for key in [
                "target_prediction_rate",
                "pareto_undominated_rate",
                "mean_matched_minus_best_control_target_margin",
                "mean_matched_minus_shuffled_signature_target_margin",
                "mean_target_margin",
                "mean_compatible_source_mse",
                "effective_zero_coefficient_rate",
                "mean_hidden_delta_norm",
            ]
        },
    })
    selected["inner_validation_amendment_path"] = str(INNER_VALIDATION_AMENDMENT_PATH.relative_to(REPO_ROOT))
    selected["inner_validation_amendment_sha256"] = INNER_VALIDATION_AMENDMENT_SHA256
    selected["inner_validation_rung_record_budgets"] = INNER_VALIDATION_RUNG_RECORD_BUDGETS
    selected["inner_validation_rung_subjects_per_behavior"] = (
        INNER_VALIDATION_RUNG_SUBJECTS_PER_BEHAVIOR
    )
    selected["inner_validation_rung_survivors"] = INNER_VALIDATION_RUNG_SURVIVORS
    selected["inner_validation_rung_summaries"] = rung_summaries
    selected["inner_validation_total_config_count"] = len(full_configs)
    selected["inner_validation_evaluated_config_count"] = len(configs)
    selected["inner_validation_evaluated_config_subset_hash"] = evaluated_config_subset_hash
    if checkpoint_path is not None and progress_log_path is not None:
        checkpoint["status"] = "completed"
        checkpoint["selected_config_hash"] = selected.get("config_hash")
        checkpoint["inner_validation_selection_hash"] = selected["inner_validation_selection_hash"]
        checkpoint["updated_at_unix"] = time.time()
        write_json_atomic(checkpoint_path, checkpoint)
        append_jsonl(progress_log_path, {
            "event": "inner_validation_completed",
            "selected_config_hash": selected.get("config_hash"),
            "inner_validation_selection_hash": selected["inner_validation_selection_hash"],
            "updated_at_unix": time.time(),
        })
    return selected


def sparse_config_hash(config: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in sorted(config.items())
        if key != "config_hash" and value is not None
    }
    return stable_hash_json({
        "scope": "four_behavior_functional_weight_editing_v23_sparse_config",
        "config": payload,
        "plan_sha256": PLAN_SHA256,
    })


def iter_sparse_subspace_configs() -> list[dict[str, Any]]:
    configs = []
    config_index = 0
    for k in SPARSE_K_VALUES:
        for lambda_rank1 in LAMBDA_RANK1_GRID:
            for lambda_solve in LAMBDA_SOLVE_GRID:
                for compatible_weight in COMPATIBLE_WEIGHT_GRID:
                    for probe_centroid_weight in PROBE_CENTROID_WEIGHT_GRID:
                        for control_penalty_weight in CONTROL_PENALTY_WEIGHT_GRID:
                            for cap_multiplier in CAP_MULTIPLIER_GRID:
                                config = {
                                    "cap_multiplier": float(cap_multiplier),
                                    "compatible_weight": float(compatible_weight),
                                    "config_index": int(config_index),
                                    "control_penalty_weight": float(control_penalty_weight),
                                    "k": int(k),
                                    "lambda_rank1": float(lambda_rank1),
                                    "lambda_solve": float(lambda_solve),
                                    "probe_centroid_weight": float(probe_centroid_weight),
                                }
                                config["config_hash"] = sparse_config_hash(config)
                                configs.append(config)
                                config_index += 1
    return configs


def inner_validation_evaluated_config_subset(
    configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(configs) <= INNER_VALIDATION_EVALUATED_CONFIG_COUNT:
        return [dict(config) for config in configs]
    strata: dict[tuple[int, float], list[Mapping[str, Any]]] = {}
    for config in configs:
        key = (int(config["k"]), float(config["cap_multiplier"]))
        strata.setdefault(key, []).append(config)
    stratum_keys = sorted(strata)
    base_quota = INNER_VALIDATION_EVALUATED_CONFIG_COUNT // len(stratum_keys)
    remainder = INNER_VALIDATION_EVALUATED_CONFIG_COUNT % len(stratum_keys)
    selected: list[dict[str, Any]] = []
    for index, key in enumerate(stratum_keys):
        quota = base_quota + (1 if index < remainder else 0)
        ordered = sorted(
            strata[key],
            key=lambda config: (
                stable_hash_json({
                    "scope": "four_behavior_functional_weight_editing_v23_config_subset",
                    "amendment_sha256": INNER_VALIDATION_AMENDMENT_SHA256,
                    "config_hash": str(config["config_hash"]),
                }),
                int(config["config_index"]),
            ),
        )
        if len(ordered) < quota:
            raise ValueError(f"config stratum {key} has {len(ordered)} configs, need {quota}")
        selected.extend(dict(config) for config in ordered[:quota])
    return sorted(selected, key=lambda config: int(config["config_index"]))


def inner_validation_evaluated_config_subset_hash(
    configs: Sequence[Mapping[str, Any]],
) -> str:
    return stable_hash_json({
        "scope": "four_behavior_functional_weight_editing_v23_evaluated_config_subset",
        "amendment_sha256": INNER_VALIDATION_AMENDMENT_SHA256,
        "config_hashes": [str(config["config_hash"]) for config in configs],
    })


def random_sparse_subspace_delta(
    *,
    basis: torch.Tensor,
    matched_hidden_delta: torch.Tensor,
    selected_layers: Sequence[int],
    seed_payload: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    selected_layers = [int(item) for item in selected_layers]
    basis = basis.detach().to(device="cpu", dtype=torch.float32)
    if basis.ndim != 2:
        raise ValueError("random sparse basis must be rank-2")
    for column_index in range(int(basis.shape[1])):
        assert_sparse_hidden_delta_scope(basis[:, column_index], selected_layers=selected_layers)
    matched_hidden_delta = matched_hidden_delta.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    assert_sparse_hidden_delta_scope(matched_hidden_delta, selected_layers=selected_layers)
    payload = {
        "scope": "four_behavior_functional_weight_editing_v23_random_sparse_control",
        **dict(seed_payload),
        "selected_layers": selected_layers,
    }
    seed_hash = stable_hash_json(payload)
    seed = int(seed_hash[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(int(basis.shape[1]), dtype=torch.float32, generator=generator)
    raw_delta = basis @ z
    assert_sparse_hidden_delta_scope(raw_delta, selected_layers=selected_layers)
    matched_norm = sparse_hidden_delta_norm(matched_hidden_delta, selected_layers=selected_layers)
    raw_norm = sparse_hidden_delta_norm(raw_delta, selected_layers=selected_layers)
    zero_norm = matched_norm < RANDOM_CONTROL_EPS or raw_norm < RANDOM_CONTROL_EPS
    if zero_norm:
        random_delta = torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.float32)
    else:
        random_delta = raw_delta / raw_norm * matched_norm
    random_norm = sparse_hidden_delta_norm(random_delta, selected_layers=selected_layers)
    return random_delta, {
        "matched_hidden_delta_norm": float(matched_norm),
        "random_hidden_delta_norm": float(random_norm),
        "random_seed": int(seed),
        "seed_hash": seed_hash,
        "zero_norm_fallback": bool(zero_norm),
    }


def record_weights_tensor(record: Mapping[str, Any]) -> torch.Tensor:
    return torch.tensor(record["weights"], dtype=torch.float32)


def probe_logits_for_weights(weights: torch.Tensor, probe_examples: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    inputs = v16.probe_inputs_tensor(probe_examples)
    logits = v16.v15.v10.decoder_v1.subject_forward_flat_batch(
        weights.detach().to(dtype=torch.float32).reshape(1, -1),
        inputs,
    )[0]
    return logits.reshape(-1).to(dtype=torch.float32)


def logits_and_jacobian_for_inputs(
    *,
    weights: torch.Tensor,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = weights.detach().clone().to(dtype=torch.float32).reshape(-1).requires_grad_(True)
    logits = v16.v15.v10.decoder_v1.subject_forward_flat_batch(
        flat.unsqueeze(0),
        inputs.to(dtype=torch.float32),
    )[0].reshape(-1)
    rows = []
    for logit in logits:
        grad = torch.autograd.grad(logit, flat, retain_graph=True, allow_unused=False)[0]
        if grad is None:
            raise RuntimeError("autograd returned None for V23 sparse row logit")
        rows.append(grad.detach().clone().to(dtype=torch.float32))
    return logits.detach().clone().to(dtype=torch.float32), torch.stack(rows).to(dtype=torch.float32)


def safe_mean_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    zero_mask = std < 1e-6
    std = torch.where(zero_mask, torch.ones_like(std), std)
    return mean.to(dtype=torch.float32), std.to(dtype=torch.float32), int(zero_mask.sum().item())


def probe_centroid_loss_for_weights(
    *,
    weights: torch.Tensor,
    target: str,
    train_stats: Mapping[str, Any],
) -> float:
    centroids = train_stats.get("target_probe_logit_centroids", {})
    centroid = centroids.get(target)
    if centroid is None or "probe_examples" not in train_stats:
        return 0.0
    logits = probe_logits_for_weights(weights, train_stats["probe_examples"])
    return float(F.mse_loss(logits, centroid.to(dtype=torch.float32)).item())


def hidden_rank1_descriptor_for_weights(
    *,
    weights: torch.Tensor,
    probe_examples: Sequence[Mapping[str, Any]],
) -> dict[str, list[torch.Tensor]]:
    probe_inputs = v16.probe_inputs_tensor(probe_examples)
    layer_inputs, layer_outputs = v17.hidden_inputs_and_outputs_flat_batch(
        weights.reshape(1, -1).to(dtype=torch.float32),
        probe_inputs,
    )
    return {
        "hbar": [item[0].mean(dim=0).to(dtype=torch.float32) for item in layer_outputs],
        "xbar": [item[0].mean(dim=0).to(dtype=torch.float32) for item in layer_inputs],
    }


def fit_hidden_rank1_train_statistics(
    subjects: Sequence[Mapping[str, Any]],
    *,
    probe_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    descriptor_by_subject = {}
    for record in subjects:
        descriptor_by_subject[str(record["subject_id"])] = hidden_rank1_descriptor_for_weights(
            weights=record_weights_tensor(record),
            probe_examples=probe_examples,
        )
    grouped = v17.records_by_behavior(subjects)
    hidden_centroids: dict[str, list[torch.Tensor]] = {}
    for behavior, records in grouped.items():
        hidden_centroids[behavior] = []
        for layer_index in HIDDEN_LAYERS:
            hidden_centroids[behavior].append(torch.stack([
                descriptor_by_subject[str(record["subject_id"])]["hbar"][layer_index]
                for record in records
            ]).mean(dim=0).to(dtype=torch.float32))
    global_hidden_centroids = []
    for layer_index in HIDDEN_LAYERS:
        global_hidden_centroids.append(torch.stack([
            descriptor_by_subject[str(record["subject_id"])]["hbar"][layer_index]
            for record in subjects
        ]).mean(dim=0).to(dtype=torch.float32))
    descriptor_hashes = {
        subject_id: stable_hash_json({
            "hbar": [tensor_to_hashable(item) for item in descriptor["hbar"]],
            "xbar": [tensor_to_hashable(item) for item in descriptor["xbar"]],
        })
        for subject_id, descriptor in descriptor_by_subject.items()
    }
    return {
        "global_hidden_centroids": global_hidden_centroids,
        "hidden_descriptor_by_subject": descriptor_by_subject,
        "hidden_descriptor_hashes": descriptor_hashes,
        "hidden_target_centroids": hidden_centroids,
    }


def hidden_descriptor_to_hashable(descriptor: Mapping[str, list[torch.Tensor]]) -> dict[str, Any]:
    return {
        "hbar": [tensor_to_hashable(item) for item in descriptor["hbar"]],
        "xbar": [tensor_to_hashable(item) for item in descriptor["xbar"]],
    }


def hidden_descriptor_from_hashable(payload: Mapping[str, Any]) -> dict[str, list[torch.Tensor]]:
    return {
        "hbar": [torch.tensor(item, dtype=torch.float32) for item in payload["hbar"]],
        "xbar": [torch.tensor(item, dtype=torch.float32) for item in payload["xbar"]],
    }


def fit_v23_train_statistics(
    subjects: Sequence[Mapping[str, Any]],
    *,
    include_models: bool = True,
    include_baseline_stats: bool = True,
    allow_default_small_pool: bool = False,
    selected_sparse_config: Mapping[str, Any] | None = None,
    run_inner_validation: bool = True,
    max_inner_workers: int | None = None,
    inner_validation_progress_dir: Path | None = None,
) -> dict[str, Any]:
    del include_models
    ordered_subjects = sorted_train_subjects(subjects)
    signatures = torch.tensor([record["signature"] for record in ordered_subjects], dtype=torch.float32)
    sig_mean, sig_std, _zero_sig = safe_mean_std(signatures)
    probe_examples = build_probe_examples()
    train_by_behavior = v17.records_by_behavior(ordered_subjects)
    hidden_stats = fit_hidden_rank1_train_statistics(
        ordered_subjects,
        probe_examples=probe_examples,
    )
    target_probe_logit_centroids = {
        behavior: torch.stack([
            probe_logits_for_weights(record_weights_tensor(record), probe_examples)
            for record in records
        ]).mean(dim=0)
        for behavior, records in train_by_behavior.items()
    }
    stats: dict[str, Any] = {
        **hidden_stats,
        "probe_examples": probe_examples,
        "probe_examples_hash": stable_hash_json(probe_examples),
        "sig_mean": sig_mean,
        "sig_std": sig_std,
        "target_probe_logit_centroid_hashes": {
            behavior: stable_hash_json(tensor_to_hashable(value))
            for behavior, value in target_probe_logit_centroids.items()
        },
        "target_probe_logit_centroids": target_probe_logit_centroids,
        "train_by_behavior": train_by_behavior,
        "train_subjects": ordered_subjects,
    }
    stats["signature_centroids"] = behavior_centroids(ordered_subjects, stats)
    if include_baseline_stats:
        stats["v16_baseline_train_stats"] = v16.fit_v16_train_statistics(
            ordered_subjects,
            probe_examples=probe_examples,
        )
        stats["v16_baseline_train_statistics_hash"] = stats["v16_baseline_train_stats"][
            "train_statistics_hash"
        ]
        v17_stats = v17.fit_v17_train_statistics(
            ordered_subjects,
            probe_examples=probe_examples,
            include_baseline_stats=False,
        )
        stats["v17_baseline_train_stats"] = v17_stats
        stats["v17_baseline_train_statistics_hash"] = v17_stats["train_statistics_hash"]
        v21_stats = v21.fit_v21_train_statistics(
            ordered_subjects,
            include_models=True,
            include_baseline_stats=False,
            allow_default_small_pool=allow_default_small_pool,
        )
        stats["v21_baseline_train_stats"] = v21_stats
        stats["v21_baseline_train_statistics_hash"] = v21_stats["train_statistics_hash"]
        v22_stats = v22.fit_v22_train_statistics(
            ordered_subjects,
            include_models=True,
            include_baseline_stats=False,
            allow_default_small_pool=allow_default_small_pool,
        )
        stats["v22_baseline_train_stats"] = v22_stats
        stats["v22_baseline_train_statistics_hash"] = v22_stats["train_statistics_hash"]
    else:
        stats["v16_baseline_train_statistics_hash"] = "not_computed"
        stats["v17_baseline_train_statistics_hash"] = "not_computed"
        stats["v21_baseline_train_statistics_hash"] = "not_computed"
        stats["v22_baseline_train_statistics_hash"] = "not_computed"
    if selected_sparse_config is not None:
        selected_config = dict(selected_sparse_config)
    elif include_baseline_stats and run_inner_validation:
        selected_config = select_sparse_config_with_inner_validation(
            ordered_subjects,
            max_workers=max_inner_workers,
            progress_dir=inner_validation_progress_dir,
        )
    else:
        selected_config = iter_sparse_subspace_configs()[0]
    stats["selected_sparse_config"] = selected_config
    stats["selected_sparse_config_hash"] = selected_config["config_hash"]
    stats["train_statistics_hash"] = stable_hash_json(full_train_statistics_hash_payload(stats))
    return stats


def build_probe_examples() -> list[dict[str, Any]]:
    return v16.v15.build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )


def output_layer_theta(weights: torch.Tensor) -> torch.Tensor:
    flat = weights.detach().to(dtype=torch.float32).reshape(-1)
    if int(flat.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError(f"expected {SOURCE_WEIGHT_DIM} weights, got {int(flat.numel())}")
    return torch.cat([flat[336:344], flat[344:345]]).clone()


def replace_output_layer_theta(weights: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    flat = weights.detach().to(dtype=torch.float32).reshape(-1).clone()
    replacement = theta.detach().to(dtype=torch.float32).reshape(-1)
    if int(flat.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError(f"expected {SOURCE_WEIGHT_DIM} weights, got {int(flat.numel())}")
    if int(replacement.numel()) != 9:
        raise ValueError(f"expected output theta dim 9, got {int(replacement.numel())}")
    flat[336:344] = replacement[:8]
    flat[344] = replacement[8]
    return flat


def hidden_layer_specs(layer_index: int) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    weight_spec = next(
        spec for spec in v17.LAYER_COMPONENT_SPECS
        if spec["name"] == f"weight_{int(layer_index)}"
    )
    bias_spec = next(
        spec for spec in v17.LAYER_COMPONENT_SPECS
        if spec["name"] == f"bias_{int(layer_index)}"
    )
    return weight_spec, bias_spec


def apply_hidden_rank1_edit(
    *,
    base_weights: torch.Tensor,
    layer_index: int,
    direction: torch.Tensor,
    xbar: torch.Tensor,
    ridge_lambda: float,
    scale: float,
    norm_cap: float,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    direction = direction.to(dtype=torch.float32).reshape(-1)
    xbar = xbar.to(dtype=torch.float32).reshape(-1)
    denom = float(xbar.pow(2).sum().item()) + float(ridge_lambda)
    if denom <= RANDOM_CONTROL_EPS and float(scale) != 0.0:
        return None, {"invalid_zero_denom": True}
    if float(scale) == 0.0:
        delta_w = torch.zeros((int(direction.numel()), int(xbar.numel())), dtype=torch.float32)
        delta_b = torch.zeros_like(direction)
    else:
        delta_w = float(scale) * torch.outer(direction, xbar) / max(denom, RANDOM_CONTROL_EPS)
        delta_b = float(scale) * direction
    raw_norm = torch.sqrt(delta_w.pow(2).sum() + delta_b.pow(2).sum())
    clipped = False
    if float(raw_norm.item()) > float(norm_cap) > 0.0:
        factor = float(norm_cap) / float(raw_norm.item())
        delta_w = delta_w * factor
        delta_b = delta_b * factor
        clipped = True
    elif float(norm_cap) <= 0.0 and float(raw_norm.item()) > RANDOM_CONTROL_EPS:
        delta_w = torch.zeros_like(delta_w)
        delta_b = torch.zeros_like(delta_b)
        clipped = True
    final_norm = torch.sqrt(delta_w.pow(2).sum() + delta_b.pow(2).sum())
    edited = base_weights.detach().clone().to(dtype=torch.float32)
    weight_spec, bias_spec = hidden_layer_specs(int(layer_index))
    current_w = v17.component_from_flat(edited, weight_spec)
    current_b = v17.component_from_flat(edited, bias_spec)
    v17.set_component(edited, weight_spec, current_w + delta_w.reshape_as(current_w))
    v17.set_component(edited, bias_spec, current_b + delta_b.reshape_as(current_b))
    return edited, {
        "hidden_delta_clipped": bool(clipped),
        "hidden_delta_norm": float(final_norm.item()),
        "raw_hidden_delta_norm": float(raw_norm.item()),
        "_delta_w": delta_w,
        "_delta_b": delta_b,
    }


def rank1_component_basis_vector(
    *,
    layer_index: int,
    direction: torch.Tensor,
    xbar: torch.Tensor,
    lambda_rank1: float,
) -> torch.Tensor:
    direction = direction.detach().to(dtype=torch.float32).reshape(-1)
    xbar = xbar.detach().to(dtype=torch.float32).reshape(-1)
    denom = float(torch.dot(xbar, xbar).item()) + float(lambda_rank1)
    if denom <= RANDOM_CONTROL_EPS:
        raise ValueError("rank1 component basis has non-positive denominator")
    delta_w = torch.outer(direction, xbar) / denom
    delta_b = direction
    basis = torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.float32)
    weight_spec, bias_spec = hidden_layer_specs(int(layer_index))
    current_w = v17.component_from_flat(basis, weight_spec)
    current_b = v17.component_from_flat(basis, bias_spec)
    v17.set_component(basis, weight_spec, delta_w.reshape_as(current_w))
    v17.set_component(basis, bias_spec, delta_b.reshape_as(current_b))
    return basis


def cosine_similarity_squared(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    eps: float = RANDOM_CONTROL_EPS,
) -> float:
    left = left.detach().to(dtype=torch.float32).reshape(-1)
    right = right.detach().to(dtype=torch.float32).reshape(-1)
    left_norm_sq = float(torch.dot(left, left).item())
    right_norm_sq = float(torch.dot(right, right).item())
    if left_norm_sq <= float(eps) or right_norm_sq <= float(eps):
        return 0.0
    numerator = float(torch.dot(left, right).item()) ** 2
    return numerator / max(left_norm_sq * right_norm_sq, float(eps))


def layer_relevance_score(
    *,
    edit_projection: torch.Tensor,
    edit_target: torch.Tensor,
    preserve_projection: torch.Tensor,
    matched_direction: torch.Tensor,
    control_directions: Sequence[torch.Tensor],
    compatible_weight: float,
    control_penalty_weight: float,
) -> dict[str, float]:
    target_gain = cosine_similarity_squared(edit_projection, edit_target)
    preserve = preserve_projection.detach().to(dtype=torch.float32).reshape(-1)
    preserve_cost = float(preserve.pow(2).mean().item()) if int(preserve.numel()) else float("inf")
    control_penalty = 0.0
    if control_directions:
        control_penalty = max(
            cosine_similarity_squared(matched_direction, control_direction)
            for control_direction in control_directions
        )
    relevance = (
        target_gain
        - float(compatible_weight) * preserve_cost
        - float(control_penalty_weight) * control_penalty
    )
    return {
        "target_gain": float(target_gain),
        "preserve_cost": float(preserve_cost),
        "control_similarity_penalty": float(control_penalty),
        "relevance": float(relevance),
    }


def select_sparse_layers_by_relevance(
    scores: Sequence[Mapping[str, Any]],
    *,
    k: int,
) -> list[Mapping[str, Any]]:
    return sorted(
        scores,
        key=lambda item: (
            -float(item["relevance"]),
            -float(item["target_gain"]),
            float(item["preserve_cost"]),
            float(item["control_similarity_penalty"]),
            int(item["layer_index"]),
        ),
    )[: int(k)]


def select_sparse_post_scale_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("no V23 sparse post-scale candidates")
    return sorted(
        candidates,
        key=lambda item: (
            float(item["support_objective"]),
            float(item["target_probe_centroid_loss"]),
            float(item["compatible_probe_utility_loss"]),
            float(item["hidden_delta_norm"]),
            float(item["post_scale"]),
            int(item["candidate_index"]),
        ),
    )[0]


def solve_sparse_ridge_coefficients(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    lambda_solve: float,
) -> dict[str, Any]:
    x = x.detach().to(device="cpu", dtype=torch.float32)
    y = y.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("sparse ridge design matrix must be rank-2")
    if int(x.shape[0]) == 0:
        return {
            "alpha": torch.zeros(int(x.shape[1]), dtype=torch.float32),
            "invalid": True,
            "jitter_retry": False,
            "reason": "zero_rows",
        }
    if int(y.numel()) != int(x.shape[0]):
        raise ValueError("sparse ridge target length must match row count")
    gram = x.T @ x
    rhs = x.T @ y
    identity = torch.eye(int(x.shape[1]), dtype=torch.float32)
    try:
        alpha = torch.linalg.solve(gram + float(lambda_solve) * identity, rhs)
        return {"alpha": alpha, "invalid": False, "jitter_retry": False}
    except RuntimeError as first_error:
        try:
            alpha = torch.linalg.solve(
                gram + (float(lambda_solve) + 1e-6) * identity,
                rhs,
            )
            return {"alpha": alpha, "invalid": False, "jitter_retry": True}
        except RuntimeError as second_error:
            return {
                "alpha": torch.zeros(int(x.shape[1]), dtype=torch.float32),
                "invalid": True,
                "jitter_retry": True,
                "reason": "solve_failed",
                "first_error": str(first_error),
                "second_error": str(second_error),
            }


def sparse_hidden_delta_mask(selected_layers: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.bool)
    for layer_index in selected_layers:
        weight_spec, bias_spec = hidden_layer_specs(int(layer_index))
        mask[weight_spec["start"] : weight_spec["end"]] = True
        mask[bias_spec["start"] : bias_spec["end"]] = True
    return mask


def assert_sparse_hidden_delta_scope(
    delta: torch.Tensor,
    *,
    selected_layers: Sequence[int],
) -> None:
    delta = delta.detach().to(dtype=torch.float32).reshape(-1)
    if int(delta.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError(f"expected {SOURCE_WEIGHT_DIM} delta values, got {int(delta.numel())}")
    mask = sparse_hidden_delta_mask(selected_layers)
    outside = delta[~mask]
    if bool(torch.any(outside.abs() > RANDOM_CONTROL_EPS).item()):
        raise ValueError("sparse hidden delta has nonzero entries outside selected hidden components")


def sparse_hidden_delta_norm(
    delta: torch.Tensor,
    *,
    selected_layers: Sequence[int],
) -> float:
    delta = delta.detach().to(dtype=torch.float32).reshape(-1)
    mask = sparse_hidden_delta_mask(selected_layers)
    return float(torch.norm(delta[mask]).item())


def clip_sparse_hidden_delta(
    delta: torch.Tensor,
    *,
    selected_layers: Sequence[int],
    norm_cap: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    delta = delta.detach().to(dtype=torch.float32).reshape(-1).clone()
    assert_sparse_hidden_delta_scope(delta, selected_layers=selected_layers)
    raw_norm = sparse_hidden_delta_norm(delta, selected_layers=selected_layers)
    clipped = False
    if raw_norm > float(norm_cap) > 0.0:
        delta = delta * (float(norm_cap) / raw_norm)
        clipped = True
    elif float(norm_cap) <= 0.0 and raw_norm > RANDOM_CONTROL_EPS:
        delta = torch.zeros_like(delta)
        clipped = True
    final_norm = sparse_hidden_delta_norm(delta, selected_layers=selected_layers)
    return delta, {
        "hidden_delta_clipped": bool(clipped),
        "hidden_delta_norm": float(final_norm),
        "raw_hidden_delta_norm": float(raw_norm),
    }


def weighted_design_from_row_groups(
    row_groups: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    basis: torch.Tensor,
    block_weights: Mapping[str, float],
) -> dict[str, Any]:
    basis = basis.detach().to(device="cpu", dtype=torch.float32)
    if basis.ndim != 2:
        raise ValueError("basis must be rank-2")
    x_blocks = []
    y_blocks = []
    row_group_names = []
    row_counts = {}
    for group_name in ROW_GROUP_ORDER:
        if group_name not in row_groups:
            continue
        block_weight = float(block_weights.get(group_name, 0.0))
        jacobian = row_groups[group_name]["jacobian"].detach().to(device="cpu", dtype=torch.float32)
        target = row_groups[group_name]["target"].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if jacobian.ndim != 2:
            raise ValueError(f"{group_name} jacobian must be rank-2")
        row_count = int(jacobian.shape[0])
        row_counts[group_name] = row_count
        if row_count == 0 or block_weight == 0.0:
            continue
        if int(target.numel()) != row_count:
            raise ValueError(f"{group_name} target length must match row count")
        row_scale = math.sqrt(block_weight / row_count)
        x_blocks.append((jacobian @ basis) * row_scale)
        y_blocks.append(target * row_scale)
        row_group_names.extend([group_name] * row_count)
    if x_blocks:
        x = torch.cat(x_blocks, dim=0).to(dtype=torch.float32)
        y = torch.cat(y_blocks, dim=0).to(dtype=torch.float32)
    else:
        x = torch.zeros((0, int(basis.shape[1])), dtype=torch.float32)
        y = torch.zeros(0, dtype=torch.float32)
    return {
        "x": x,
        "y": y,
        "row_counts": row_counts,
        "row_group_names": row_group_names,
    }


def compatible_probe_mask(
    *,
    probe_examples: Sequence[Mapping[str, Any]],
    source: str,
    target: str,
) -> torch.Tensor:
    predicates = v16.v15.v14.PREDICATES
    values = []
    for example in probe_examples:
        sequence = example.get("sequence", example)
        values.append(bool(predicates[source](sequence)) == bool(predicates[target](sequence)))
    mask = torch.tensor(values, dtype=torch.bool)
    if int(mask.sum().item()) == 0:
        raise ValueError(f"no compatible fixed-probe examples for {source}->{target}")
    return mask


def target_direction_for_mode(
    *,
    subject: Mapping[str, Any],
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    descriptor: Mapping[str, list[torch.Tensor]],
    descriptor_mode: str,
    shuffled_hidden_descriptor: Mapping[str, list[torch.Tensor]] | None = None,
) -> list[torch.Tensor]:
    if descriptor_mode == "matched_probe":
        target_values = train_stats["hidden_target_centroids"][target]
        source_values = descriptor["hbar"]
    elif descriptor_mode == "no_probe":
        return [torch.zeros(8, dtype=torch.float32) for _ in HIDDEN_LAYERS]
    elif descriptor_mode == "source_probe":
        target_values = train_stats["hidden_target_centroids"][source]
        source_values = descriptor["hbar"]
    elif descriptor_mode == "shuffled_probe":
        if shuffled_hidden_descriptor is None:
            return [torch.zeros(8, dtype=torch.float32) for _ in HIDDEN_LAYERS]
        target_values = train_stats["hidden_target_centroids"][target]
        source_values = shuffled_hidden_descriptor["hbar"]
    elif descriptor_mode == "target_label_only":
        target_values = train_stats["hidden_target_centroids"][target]
        source_values = train_stats["global_hidden_centroids"]
    elif descriptor_mode == "nearest_target_probe":
        nearest = nearest_target_hidden_descriptor(
            descriptor=descriptor,
            target=target,
            train_stats=train_stats,
        )
        target_values = nearest["hbar"]
        source_values = descriptor["hbar"]
    else:
        raise ValueError(f"unknown V22 descriptor mode: {descriptor_mode}")
    return [
        target_values[layer_index].to(dtype=torch.float32) - source_values[layer_index].to(dtype=torch.float32)
        for layer_index in HIDDEN_LAYERS
    ]


def nearest_target_hidden_descriptor(
    *,
    descriptor: Mapping[str, list[torch.Tensor]],
    target: str,
    train_stats: Mapping[str, Any],
) -> Mapping[str, list[torch.Tensor]]:
    candidates = []
    for record in train_stats["train_by_behavior"][target]:
        candidate = train_stats["hidden_descriptor_by_subject"][str(record["subject_id"])]
        distance = sum(
            float(torch.norm(candidate["hbar"][layer] - descriptor["hbar"][layer]).item()) ** 2
            for layer in HIDDEN_LAYERS
        )
        candidates.append((distance, str(record["subject_id"]), candidate))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def hidden_off_manifold_loss(
    *,
    weights: torch.Tensor,
    layer_index: int,
    train_stats: Mapping[str, Any],
) -> float:
    descriptor = hidden_rank1_descriptor_for_weights(
        weights=weights,
        probe_examples=train_stats["probe_examples"],
    )
    hbar = descriptor["hbar"][int(layer_index)]
    losses = [
        float(F.mse_loss(hbar, centroids[int(layer_index)].to(dtype=torch.float32)).item())
        for centroids in train_stats["hidden_target_centroids"].values()
    ]
    return min(losses)


def component_candidate_losses(
    *,
    weights: torch.Tensor,
    source_weights: torch.Tensor,
    source: str,
    target: str,
    layer_index: int,
    train_stats: Mapping[str, Any],
) -> dict[str, float]:
    support = v17.support_objective_for_weights(
        weights=weights,
        source_weights=source_weights,
        source=source,
        target=target,
    )
    probe_examples = train_stats["probe_examples"]
    candidate_logits = probe_logits_for_weights(weights, probe_examples)
    source_logits = probe_logits_for_weights(source_weights, probe_examples)
    target_centroid = train_stats["target_probe_logit_centroids"][target].to(dtype=torch.float32)
    mask = compatible_probe_mask(probe_examples=probe_examples, source=source, target=target)
    return {
        "compatible_probe_utility_loss": float(F.mse_loss(
            candidate_logits[mask],
            source_logits[mask],
        ).item()),
        "hidden_off_manifold_loss": hidden_off_manifold_loss(
            weights=weights,
            layer_index=layer_index,
            train_stats=train_stats,
        ),
        "support_objective": float(support["objective"]),
        "target_probe_centroid_loss": float(F.mse_loss(candidate_logits, target_centroid).item()),
    }


def sparse_subspace_row_groups(
    *,
    base_weights: torch.Tensor,
    source_weights: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
) -> dict[str, Mapping[str, torch.Tensor]]:
    support = v16.v15.v14.prepare_support_tensors_with_source_logits(
        source_weights=source_weights.detach().to(dtype=torch.float32),
        source=source,
        target=target,
    )
    target_logits, j_target = logits_and_jacobian_for_inputs(
        weights=base_weights,
        inputs=support["target_inputs"],
    )
    conflict_logits, j_conflict = logits_and_jacobian_for_inputs(
        weights=base_weights,
        inputs=support["conflict_inputs"],
    )
    compatible_logits, j_compatible = logits_and_jacobian_for_inputs(
        weights=base_weights,
        inputs=support["compatible_inputs"],
    )
    target_desired = torch.where(
        support["target_labels"].reshape(-1).to(dtype=torch.float32) > 0.5,
        torch.tensor(DESIRED_LOGIT_MAGNITUDE, dtype=torch.float32),
        torch.tensor(-DESIRED_LOGIT_MAGNITUDE, dtype=torch.float32),
    )
    conflict_desired = torch.where(
        support["conflict_target_labels"].reshape(-1).to(dtype=torch.float32) > 0.5,
        torch.tensor(DESIRED_LOGIT_MAGNITUDE, dtype=torch.float32),
        torch.tensor(-DESIRED_LOGIT_MAGNITUDE, dtype=torch.float32),
    )
    probe_examples = train_stats["probe_examples"]
    probe_inputs = v16.probe_inputs_tensor(probe_examples)
    base_probe_logits, j_probe = logits_and_jacobian_for_inputs(
        weights=base_weights,
        inputs=probe_inputs,
    )
    source_probe_logits = probe_logits_for_weights(source_weights, probe_examples)
    target_centroid = train_stats["target_probe_logit_centroids"][target].to(dtype=torch.float32)
    compatible_mask = compatible_probe_mask(
        probe_examples=probe_examples,
        source=source,
        target=target,
    )
    target_mask = ~compatible_mask
    return {
        "target_support": {
            "jacobian": j_target,
            "target": target_desired - target_logits,
        },
        "conflict_support": {
            "jacobian": j_conflict,
            "target": conflict_desired - conflict_logits,
        },
        "compatible_support": {
            "jacobian": j_compatible,
            "target": support["compatible_source_logits"].reshape(-1).to(dtype=torch.float32)
            - compatible_logits,
        },
        "probe_target": {
            "jacobian": j_probe[target_mask],
            "target": target_centroid[target_mask] - base_probe_logits[target_mask],
        },
        "probe_compatible": {
            "jacobian": j_probe[compatible_mask],
            "target": source_probe_logits[compatible_mask] - base_probe_logits[compatible_mask],
        },
    }


def sparse_block_weights(config: Mapping[str, Any]) -> dict[str, float]:
    compatible_weight = float(config["compatible_weight"])
    return {
        "target_support": 1.0,
        "conflict_support": 1.0,
        "compatible_support": compatible_weight,
        "probe_target": float(config["probe_centroid_weight"]),
        "probe_compatible": compatible_weight,
    }


def split_design_for_relevance(
    row_groups: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    basis: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    block_weights = sparse_block_weights(config)
    edit_design = weighted_design_from_row_groups(
        {
            "target_support": row_groups["target_support"],
            "conflict_support": row_groups["conflict_support"],
            "probe_target": row_groups["probe_target"],
        },
        basis=basis,
        block_weights=block_weights,
    )
    preserve_design = weighted_design_from_row_groups(
        {
            "compatible_support": row_groups["compatible_support"],
            "probe_compatible": row_groups["probe_compatible"],
        },
        basis=basis,
        block_weights=block_weights,
    )
    return {"edit": edit_design, "preserve": preserve_design}


def sparse_post_scale_candidate_losses(
    *,
    weights: torch.Tensor,
    source_weights: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
) -> dict[str, float]:
    support = v17.support_objective_for_weights(
        weights=weights,
        source_weights=source_weights,
        source=source,
        target=target,
    )
    probe_examples = train_stats["probe_examples"]
    candidate_logits = probe_logits_for_weights(weights, probe_examples)
    source_logits = probe_logits_for_weights(source_weights, probe_examples)
    target_centroid = train_stats["target_probe_logit_centroids"][target].to(dtype=torch.float32)
    compatible_mask = compatible_probe_mask(
        probe_examples=probe_examples,
        source=source,
        target=target,
    )
    target_mask = ~compatible_mask
    target_probe_centroid_loss = 0.0
    if bool(torch.any(target_mask).item()):
        target_probe_centroid_loss = float(F.mse_loss(
            candidate_logits[target_mask],
            target_centroid[target_mask],
        ).item())
    return {
        "compatible_probe_utility_loss": float(F.mse_loss(
            candidate_logits[compatible_mask],
            source_logits[compatible_mask],
        ).item()),
        "support_objective": float(support["objective"]),
        "target_probe_centroid_loss": float(target_probe_centroid_loss),
    }


def select_probe_routed_sparse_subspace_edit(
    *,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    source: str,
    target: str,
    subject: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    shuffled_hidden_descriptor: Mapping[str, list[torch.Tensor]] | None = None,
    descriptor_mode: str = "matched_probe",
    config: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    del source_signature_norm
    selected_config = dict(config or train_stats.get("selected_sparse_config") or iter_sparse_subspace_configs()[0])
    base_weights, base_metadata = v16.output_layer_no_signature_support_optimizer(
        source_weights=source_weights,
        source=source,
        target=target,
        subject_id=str(subject["subject_id"]),
    )
    descriptor = hidden_rank1_descriptor_for_weights(
        weights=source_weights,
        probe_examples=train_stats["probe_examples"],
    )
    matched_directions = target_direction_for_mode(
        subject=subject,
        source=source,
        target=target,
        train_stats=train_stats,
        descriptor=descriptor,
        descriptor_mode=descriptor_mode,
        shuffled_hidden_descriptor=shuffled_hidden_descriptor,
    )
    control_direction_sets = [
        target_direction_for_mode(
            subject=subject,
            source=source,
            target=target,
            train_stats=train_stats,
            descriptor=descriptor,
            descriptor_mode=mode,
            shuffled_hidden_descriptor=shuffled_hidden_descriptor,
        )
        for mode in ["source_probe", "shuffled_probe", "target_label_only", "nearest_target_probe"]
    ]
    row_groups = sparse_subspace_row_groups(
        base_weights=base_weights,
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats,
    )
    layer_scores = []
    basis_by_layer = {}
    for layer_index in HIDDEN_LAYERS:
        basis = rank1_component_basis_vector(
            layer_index=layer_index,
            direction=matched_directions[layer_index],
            xbar=descriptor["xbar"][layer_index],
            lambda_rank1=float(selected_config["lambda_rank1"]),
        )
        basis_by_layer[layer_index] = basis
        designs = split_design_for_relevance(
            row_groups,
            basis=basis.reshape(-1, 1),
            config=selected_config,
        )
        score = layer_relevance_score(
            edit_projection=designs["edit"]["x"].reshape(-1),
            edit_target=designs["edit"]["y"],
            preserve_projection=designs["preserve"]["x"].reshape(-1),
            matched_direction=matched_directions[layer_index],
            control_directions=[directions[layer_index] for directions in control_direction_sets],
            compatible_weight=float(selected_config["compatible_weight"]),
            control_penalty_weight=float(selected_config["control_penalty_weight"]),
        )
        layer_scores.append({**score, "layer_index": int(layer_index)})
    selected = select_sparse_layers_by_relevance(layer_scores, k=int(selected_config["k"]))
    if any(not math.isfinite(float(item["relevance"])) for item in selected):
        raise ValueError("nonfinite V23 sparse relevance in selected layer set")
    selected_layers = [int(item["layer_index"]) for item in selected]
    basis = torch.stack([basis_by_layer[layer_index] for layer_index in selected_layers], dim=1)
    design = weighted_design_from_row_groups(
        row_groups,
        basis=basis,
        block_weights=sparse_block_weights(selected_config),
    )
    solution = solve_sparse_ridge_coefficients(
        design["x"],
        design["y"],
        lambda_solve=float(selected_config["lambda_solve"]),
    )
    if solution["invalid"]:
        raise ValueError(f"invalid V23 sparse ridge solve: {solution.get('reason', 'unknown')}")
    base_delta_norm = (
        output_layer_theta(base_weights) - output_layer_theta(source_weights)
    ).norm()
    norm_cap = float(selected_config["cap_multiplier"]) * max(
        float(base_delta_norm.item()),
        RANDOM_CONTROL_EPS,
    )
    candidates = []
    solved_delta = basis @ solution["alpha"]
    for candidate_index, post_scale in enumerate(SPARSE_POST_SCALE_GRID):
        raw_delta = float(post_scale) * solved_delta
        delta, clip_metadata = clip_sparse_hidden_delta(
            raw_delta,
            selected_layers=selected_layers,
            norm_cap=norm_cap,
        )
        candidate_weights = base_weights.detach().clone().to(dtype=torch.float32) + delta
        candidates.append({
            **sparse_post_scale_candidate_losses(
                weights=candidate_weights,
                source_weights=source_weights,
                source=source,
                target=target,
                train_stats=train_stats,
            ),
            **clip_metadata,
            "candidate_index": int(candidate_index),
            "post_scale": float(post_scale),
            "weights": candidate_weights,
            "_delta": delta,
        })
    best = select_sparse_post_scale_candidate(candidates)
    edited = best["weights"]
    delta = best["_delta"]
    metadata = {
        "base_control_type": "output_layer_no_signature_support_optimizer",
        "base_metadata": base_metadata,
        "candidate_index": int(best["candidate_index"]),
        "compatible_probe_utility_loss": float(best["compatible_probe_utility_loss"]),
        "control_type": EDITOR_METHOD,
        "descriptor_mode": descriptor_mode,
        "hidden_delta_clipped": bool(best["hidden_delta_clipped"]),
        "hidden_delta_norm": float(best["hidden_delta_norm"]),
        "hidden_norm_cap": float(norm_cap),
        "jitter_retry": bool(solution["jitter_retry"]),
        "post_scale": float(best["post_scale"]),
        "raw_hidden_delta_norm": float(best["raw_hidden_delta_norm"]),
        "selected_config": {
            key: value for key, value in selected_config.items()
            if key != "config_hash"
        },
        "selected_config_hash": selected_config.get("config_hash") or sparse_config_hash(selected_config),
        "selected_layer_scores": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in selected
        ],
        "selected_layers": selected_layers,
        "scale_0_selected": bool(float(best["hidden_delta_norm"]) < RANDOM_CONTROL_EPS),
        "support_objective": float(best["support_objective"]),
        "target_probe_centroid_loss": float(best["target_probe_centroid_loss"]),
        "train_statistics_hash": train_stats.get("train_statistics_hash"),
        "_base_weights": base_weights.detach().clone().to(dtype=torch.float32),
        "_selected_basis": basis.detach().clone().to(dtype=torch.float32),
        "_selected_hidden_delta": delta.detach().clone().to(dtype=torch.float32),
    }
    return edited, metadata


def select_component_activation_rank1_edit(
    *,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    source: str,
    target: str,
    subject: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    shuffled_hidden_descriptor: Mapping[str, list[torch.Tensor]] | None = None,
    descriptor_mode: str = "matched_probe",
) -> tuple[torch.Tensor, dict[str, Any]]:
    return select_probe_routed_sparse_subspace_edit(
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        source=source,
        target=target,
        subject=subject,
        train_stats=train_stats,
        shuffled_hidden_descriptor=shuffled_hidden_descriptor,
        descriptor_mode=descriptor_mode,
    )


def signature_prior(
    *,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    subject_id: str,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    signature_pool_behavior: str | None = None,
    allowed_target_subject_ids: set[str] | None = None,
) -> dict[str, Any]:
    pool_behavior = signature_pool_behavior or target
    target_records = list(train_stats["train_by_behavior"][pool_behavior])
    if allowed_target_subject_ids is not None:
        allowed = {str(item) for item in allowed_target_subject_ids}
        target_records = [
            record for record in target_records
            if str(record["subject_id"]) in allowed
        ]
    if not target_records:
        raise ValueError(f"no train target records for signature pool {pool_behavior}")
    topk = v17.signature_topk_weights(
        target_records,
        source_signature_norm=source_signature_norm,
        train_stats=train_stats,
        top_k=SIGNATURE_TOP_K,
        temperature=SIGNATURE_TEMPERATURE,
    )
    weighted_delta = torch.zeros_like(source_weights, dtype=torch.float32)
    aligned_target_weights = []
    for index, (_distance, _target_subject_id, target_record) in enumerate(topk["selected"]):
        raw_delta = v16.v15.v14.target_delta_for_record(
            source_weights=source_weights,
            target_record=target_record,
            source=source,
            target=target,
            subject_id=subject_id,
            alignment_mode="hungarian",
        ).to(dtype=torch.float32)
        weighted_delta = weighted_delta + topk["weights"][index].to(dtype=torch.float32) * raw_delta
        aligned_target_weights.append(source_weights + raw_delta)
    if aligned_target_weights:
        activation_delta = v17.activation_rank1_delta(
            source_weights=source_weights,
            aligned_target_weights=torch.stack(aligned_target_weights),
            signature_weights=topk["weights"],
            probe_examples=train_stats["probe_examples"],
        )
    else:
        activation_delta = torch.zeros_like(source_weights, dtype=torch.float32)
    metadata = {
        "activation_delta_hash": stable_hash_json(tensor_to_hashable(activation_delta)),
        "selected_signature_targets": topk["metadata"],
        "selected_signature_targets_hash": stable_hash_json(topk["metadata"]),
        "signature_pool_behavior": pool_behavior,
        "weighted_delta_hash": stable_hash_json(tensor_to_hashable(weighted_delta)),
    }
    return {
        "activation_delta": activation_delta.to(dtype=torch.float32),
        "metadata": metadata,
        "weighted_delta": weighted_delta.to(dtype=torch.float32),
    }


def sparse_sensitivity_mask(sensitivity: torch.Tensor, *, fraction: float) -> torch.Tensor:
    values = sensitivity.to(dtype=torch.float32).reshape(-1)
    if int(values.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError(f"expected sensitivity dim {SOURCE_WEIGHT_DIM}, got {values.numel()}")
    mask = torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.bool)
    for spec in v17.LAYER_COMPONENT_SPECS:
        start = int(spec["start"])
        end = int(spec["end"])
        width = end - start
        keep = max(1, int(math.ceil(width * float(fraction))))
        scored = [
            (float(values[index].item()), int(index))
            for index in range(start, end)
        ]
        scored.sort(key=lambda item: (item[0], item[1]))
        for _score, index in scored[:keep]:
            mask[index] = True
    return mask


def _canonicalize_vh_rows(vh: torch.Tensor) -> torch.Tensor:
    rows = vh.detach().clone().to(dtype=torch.float32)
    for index in range(int(rows.shape[0])):
        row = rows[index]
        nonzero = torch.nonzero(torch.abs(row) > 1e-12, as_tuple=False)
        if int(nonzero.numel()) and float(row[int(nonzero[0].item())].item()) < 0.0:
            rows[index] = -row
    return rows


def compatible_nullspace_basis(
    *,
    j_preserve: torch.Tensor,
    mask: torch.Tensor,
    source_dim: int = SOURCE_WEIGHT_DIM,
) -> tuple[torch.Tensor, dict[str, Any]]:
    active = torch.nonzero(mask.to(dtype=torch.bool).reshape(-1), as_tuple=False).reshape(-1)
    if not int(active.numel()):
        raise ValueError("mask selects no parameters")
    e_mask = torch.eye(int(source_dim), dtype=torch.float32)[:, active]
    matrix = j_preserve.to(dtype=torch.float32) @ e_mask
    _u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=True)
    vh = _canonicalize_vh_rows(vh)
    max_s = float(singular_values.max().item()) if int(singular_values.numel()) else 0.0
    cutoff = max(max_s * NULLSPACE_RELATIVE_CUTOFF, NULLSPACE_ABSOLUTE_CUTOFF)
    rank = int(torch.sum(singular_values > cutoff).item()) if int(singular_values.numel()) else 0
    if rank >= int(active.numel()):
        basis = torch.zeros((int(source_dim), 0), dtype=torch.float32)
    else:
        n_mask = vh[rank:].T.contiguous()
        basis = e_mask @ n_mask
    if int(basis.shape[1]):
        gram = basis.T @ basis
        deviation = torch.max(torch.abs(gram - torch.eye(int(basis.shape[1]), dtype=torch.float32)))
        if float(deviation.item()) > 1e-4:
            raise ValueError("null-space basis failed orthonormality check")
    metadata = {
        "active_parameter_count": int(active.numel()),
        "basis_hash": stable_hash_json(tensor_to_hashable(basis)),
        "cutoff": float(cutoff),
        "empty_null_basis": int(basis.shape[1]) == 0,
        "null_dim": int(basis.shape[1]),
        "rank": int(rank),
        "singular_values": tensor_to_hashable(singular_values),
    }
    return basis.to(dtype=torch.float32), metadata


def iter_candidate_grid() -> list[dict[str, Any]]:
    grid = []
    candidate_index = 0
    for mask_fraction in MASK_FRACTIONS:
        for ridge_lambda in RIDGE_LAMBDAS:
            for prior_lambda in PRIOR_LAMBDAS:
                for activation_scale in ACTIVATION_SCALE_GRID:
                    for post_scale in POST_SCALE_GRID:
                        grid.append({
                            "activation_scale": float(activation_scale),
                            "candidate_index": int(candidate_index),
                            "mask_fraction": float(mask_fraction),
                            "post_scale": float(post_scale),
                            "prior_lambda": float(prior_lambda),
                            "ridge_lambda": float(ridge_lambda),
                        })
                        candidate_index += 1
    return grid


def select_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(
        candidates,
        key=lambda item: (
            float(item["support_objective"]),
            float(item["delta_norm"]),
            float(item["mask_fraction"]),
            float(item["ridge_lambda"]),
            float(item["prior_lambda"]),
            float(item["activation_scale"]),
            float(item["post_scale"]),
            int(item["candidate_index"]),
        ),
    )


def solve_tangent_ridge(
    *,
    basis: torch.Tensor,
    j_edit: torch.Tensor,
    b_edit: torch.Tensor,
    delta_signature: torch.Tensor,
    ridge_lambda: float,
    prior_lambda: float,
) -> dict[str, torch.Tensor] | None:
    b = basis.to(dtype=torch.float32)
    a = j_edit.to(dtype=torch.float32) @ b
    target = b_edit.to(dtype=torch.float32)
    z_prior = b.T @ delta_signature.to(dtype=torch.float32)
    eye = torch.eye(int(b.shape[1]), dtype=torch.float32)
    lhs = a.T @ a + (float(ridge_lambda) + float(prior_lambda)) * eye
    rhs = a.T @ target + float(prior_lambda) * z_prior
    try:
        z = torch.linalg.solve(lhs, rhs)
        jitter_added = False
    except RuntimeError:
        try:
            z = torch.linalg.solve(lhs + 1e-6 * eye, rhs)
            jitter_added = True
        except RuntimeError:
            return None
    delta = b @ z
    return {
        "coefficients": z.to(dtype=torch.float32),
        "delta": delta.to(dtype=torch.float32),
        "jitter_added": torch.tensor(bool(jitter_added)),
    }


def random_tangent_delta(
    *,
    basis: torch.Tensor,
    matched_delta: torch.Tensor,
    subject_id: str,
    source: str,
    target: str,
    index: int,
    train_statistics_hash: str,
    selected_candidate_metadata: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    basis_hash = stable_hash_json(tensor_to_hashable(basis))
    matched_norm = matched_delta.detach().to(dtype=torch.float32).norm()
    seed_payload = {
        "basis_hash": basis_hash,
        "index": int(index),
        "matched_delta_norm": tensor_to_hashable(matched_norm.reshape(1)),
        "method": EDITOR_METHOD,
        "source": source,
        "subject_id": subject_id,
        "target": target,
        "train_statistics_hash": train_statistics_hash,
    }
    seed_hash = stable_hash_json(seed_payload)
    seed = int(seed_hash[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    coeff = torch.randn(int(basis.shape[1]), generator=generator, dtype=torch.float32)
    raw_delta = basis.to(dtype=torch.float32) @ coeff
    raw_norm = raw_delta.norm()
    if float(matched_norm.item()) < RANDOM_CONTROL_EPS or float(raw_norm.item()) < RANDOM_CONTROL_EPS:
        final_delta = torch.zeros_like(matched_delta, dtype=torch.float32)
        zero_norm = True
    else:
        final_delta = raw_delta / raw_norm * matched_norm
        zero_norm = False
    return final_delta, {
        "activation_scale": float(selected_candidate_metadata["activation_scale"]),
        "basis_hash": basis_hash,
        "coefficient_hash": stable_hash_json(tensor_to_hashable(coeff)),
        "final_norm": float(final_delta.norm().item()),
        "index": int(index),
        "mask_fraction": float(selected_candidate_metadata["mask_fraction"]),
        "matched_norm": float(matched_norm.item()),
        "post_scale": float(selected_candidate_metadata["post_scale"]),
        "prior_lambda": float(selected_candidate_metadata["prior_lambda"]),
        "raw_norm": float(raw_norm.item()),
        "ridge_lambda": float(selected_candidate_metadata["ridge_lambda"]),
        "seed_hash": seed_hash,
        "zero_norm_fallback": bool(zero_norm),
    }


def _standard_basis_from_mask(mask: torch.Tensor, *, source_dim: int = SOURCE_WEIGHT_DIM) -> torch.Tensor:
    active = torch.nonzero(mask.to(dtype=torch.bool).reshape(-1), as_tuple=False).reshape(-1)
    return torch.eye(int(source_dim), dtype=torch.float32)[:, active]


def support_jacobians(
    *,
    source_weights: torch.Tensor,
    source: str,
    target: str,
) -> dict[str, torch.Tensor]:
    flat = source_weights.detach().clone().to(dtype=torch.float32).requires_grad_(True)
    support = v16.v15.v14.prepare_support_tensors_with_source_logits(
        source_weights=source_weights.detach().to(dtype=torch.float32),
        source=source,
        target=target,
    )

    def logits_and_jacobian(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = v16.v15.v10.decoder_v1.subject_forward_flat_batch(flat.unsqueeze(0), inputs)[0].reshape(-1)
        rows = []
        for logit in logits:
            grad = torch.autograd.grad(logit, flat, retain_graph=True, allow_unused=False)[0]
            if grad is None:
                raise RuntimeError("autograd returned None for support logit")
            rows.append(grad.detach().clone().to(dtype=torch.float32))
        return logits.detach().clone().to(dtype=torch.float32), torch.stack(rows).to(dtype=torch.float32)

    target_logits, j_target = logits_and_jacobian(support["target_inputs"])
    conflict_logits, j_conflict = logits_and_jacobian(support["conflict_inputs"])
    _compatible_logits, j_preserve = logits_and_jacobian(support["compatible_inputs"])
    target_desired = torch.where(
        support["target_labels"].reshape(-1).to(dtype=torch.float32) > 0.5,
        torch.tensor(DESIRED_LOGIT_MAGNITUDE, dtype=torch.float32),
        torch.tensor(-DESIRED_LOGIT_MAGNITUDE, dtype=torch.float32),
    )
    conflict_desired = torch.where(
        support["conflict_target_labels"].reshape(-1).to(dtype=torch.float32) > 0.5,
        torch.tensor(DESIRED_LOGIT_MAGNITUDE, dtype=torch.float32),
        torch.tensor(-DESIRED_LOGIT_MAGNITUDE, dtype=torch.float32),
    )
    return {
        "b_edit": torch.cat([target_desired - target_logits, conflict_desired - conflict_logits]).to(dtype=torch.float32),
        "j_edit": torch.cat([j_target, j_conflict], dim=0).to(dtype=torch.float32),
        "j_preserve": j_preserve.to(dtype=torch.float32),
        "support_row_counts": {
            "compatible": int(j_preserve.shape[0]),
            "conflict": int(j_conflict.shape[0]),
            "target": int(j_target.shape[0]),
        },
    }


def tangent_prior_for_control(
    *,
    control_type: str,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    shuffled_signature_norm: torch.Tensor,
    subject_id: str,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
) -> dict[str, Any]:
    zero_delta = torch.zeros_like(source_weights, dtype=torch.float32)
    if control_type in {V20_TANGENT_CONTROL_METHOD, "no_nullspace_signature_tangent"}:
        prior = signature_prior(
            source_weights=source_weights,
            source_signature_norm=source_signature_norm,
            subject_id=subject_id,
            source=source,
            target=target,
            train_stats=train_stats,
        )
        prior["metadata"] = {**prior["metadata"], "initialization_source": "matched_signature"}
        return prior
    if control_type == "target_label_tangent_nullspace":
        direction = v17.direction_key(source, target)
        delta = train_stats.get("target_centroid_deltas", {}).get(direction, zero_delta).detach().clone()
        return {
            "activation_delta": zero_delta,
            "metadata": {
                "initialization_source": "target_behavior_centroid",
                "signature_pool_behavior": target,
                "target_centroid_delta_hash": train_stats.get("target_centroid_delta_hashes", {}).get(direction),
            },
            "weighted_delta": delta.to(dtype=torch.float32),
        }
    if control_type == "no_signature_zero_tangent_nullspace":
        return {
            "activation_delta": zero_delta,
            "metadata": {"initialization_source": "zero", "signature_pool_behavior": None},
            "weighted_delta": zero_delta,
        }
    if control_type == "shuffled_signature_tangent_nullspace":
        prior = signature_prior(
            source_weights=source_weights,
            source_signature_norm=shuffled_signature_norm,
            subject_id=subject_id,
            source=source,
            target=target,
            train_stats=train_stats,
        )
        prior["metadata"] = {**prior["metadata"], "initialization_source": "shuffled_signature"}
        return prior
    if control_type == "source_signature_tangent_nullspace":
        prior = signature_prior(
            source_weights=source_weights,
            source_signature_norm=source_signature_norm,
            subject_id=subject_id,
            source=source,
            target=target,
            train_stats=train_stats,
            signature_pool_behavior=source,
        )
        prior["metadata"] = {**prior["metadata"], "initialization_source": "source_behavior_signature"}
        return prior
    raise ValueError(f"unknown V20 control type: {control_type}")


def select_tangent_nullspace_edit(
    *,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    source: str,
    target: str,
    subject_id: str,
    train_stats: Mapping[str, Any],
    shuffled_signature_norm: torch.Tensor | None = None,
    control_type: str = EDITOR_METHOD,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prior = tangent_prior_for_control(
        control_type=control_type,
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        shuffled_signature_norm=source_signature_norm if shuffled_signature_norm is None else shuffled_signature_norm,
        subject_id=subject_id,
        source=source,
        target=target,
        train_stats=train_stats,
    )
    tangent = support_jacobians(source_weights=source_weights, source=source, target=target)
    sensitivity = torch.sqrt(torch.mean(tangent["j_preserve"] ** 2, dim=0))
    candidates = []
    solver_failure_skipped_count = 0
    basis_cache: dict[tuple[float, bool], tuple[torch.Tensor, dict[str, Any]]] = {}
    for grid in iter_candidate_grid():
        mask_fraction = float(grid["mask_fraction"])
        no_nullspace = control_type == "no_nullspace_signature_tangent"
        cache_key = (mask_fraction, no_nullspace)
        if cache_key not in basis_cache:
            mask = sparse_sensitivity_mask(sensitivity, fraction=mask_fraction)
            if no_nullspace:
                basis = _standard_basis_from_mask(mask)
                basis_meta = {
                    "basis_hash": stable_hash_json(tensor_to_hashable(basis)),
                    "empty_null_basis": False,
                    "null_dim": int(basis.shape[1]),
                    "rank": 0,
                }
            else:
                basis, basis_meta = compatible_nullspace_basis(
                    j_preserve=tangent["j_preserve"],
                    mask=mask,
                    source_dim=SOURCE_WEIGHT_DIM,
                )
            basis_cache[cache_key] = (basis, basis_meta)
        basis, basis_meta = basis_cache[cache_key]
        if basis_meta["empty_null_basis"]:
            continue
        solution = solve_tangent_ridge(
            basis=basis,
            j_edit=tangent["j_edit"],
            b_edit=tangent["b_edit"],
            delta_signature=prior["weighted_delta"],
            ridge_lambda=float(grid["ridge_lambda"]),
            prior_lambda=float(grid["prior_lambda"]),
        )
        if solution is None:
            solver_failure_skipped_count += 1
            continue
        delta = solution["delta"] + float(grid["activation_scale"]) * prior["activation_delta"]
        delta_norm = delta.norm()
        clipped = False
        if float(delta_norm.item()) > MAX_TANGENT_DELTA_NORM:
            delta = delta / delta_norm * MAX_TANGENT_DELTA_NORM
            clipped = True
        candidate_delta = float(grid["post_scale"]) * delta
        weights = source_weights + candidate_delta
        losses = v17.support_objective_for_weights(
            weights=weights,
            source_weights=source_weights,
            source=source,
            target=target,
        )
        delta = weights - source_weights
        candidates.append({
            **losses,
            **grid,
            "_basis": basis,
            "basis_hash": basis_meta["basis_hash"],
            "delta_norm": float(delta.norm().item()),
            "jitter_added": bool(solution["jitter_added"].item()),
            "support_objective": float(losses["objective"]),
            "trust_region_clipped": bool(clipped),
            "weights": weights.detach().clone(),
        })
    if not candidates:
        raise ValueError("no valid V20 tangent candidates")
    best = select_candidate(candidates)
    metadata = {
        **prior["metadata"],
        "_selected_basis": best["_basis"],
        "activation_scale": float(best["activation_scale"]),
        "basis_hash": str(best["basis_hash"]),
        "candidate_index": int(best["candidate_index"]),
        "control_type": control_type,
        "jitter_added": bool(best["jitter_added"]),
        "mask_fraction": float(best["mask_fraction"]),
        "post_scale": float(best["post_scale"]),
        "prior_lambda": float(best["prior_lambda"]),
        "ridge_lambda": float(best["ridge_lambda"]),
        "selected_delta_norm": float(best["delta_norm"]),
        "selected_objective": float(best["objective"]),
        "solver_failure_skipped_count": int(solver_failure_skipped_count),
        "support_row_counts": tangent["support_row_counts"],
        "train_statistics_hash": train_stats.get("train_statistics_hash"),
        "trust_region_clipped": bool(best["trust_region_clipped"]),
    }
    return best["weights"], metadata


def control_record_from_weights(
    control_type: str,
    weights: torch.Tensor,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    safe_metadata = {
        key: value for key, value in dict(metadata or {}).items()
        if key != "control_type"
    }
    return v17.control_record_from_weights(
        control_type,
        weights,
        source,
        target,
        source_weights,
        safe_metadata,
    )


def build_controls(
    *,
    subject: Mapping[str, Any],
    source: str,
    target: str,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    shuffled_signature_norm: torch.Tensor,
    shuffled_hidden_descriptor: Mapping[str, list[torch.Tensor]] | None,
    matched_weights: torch.Tensor,
    matched_metadata: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    random_controls: int = RANDOM_CONTROLS_PER_RECORD,
) -> list[dict[str, Any]]:
    controls = [control_record_from_weights("no_edit", source_weights, source, target, source_weights)]
    optimized_weights, optimized_meta = v16.output_layer_no_signature_support_optimizer(
        source_weights=source_weights,
        source=source,
        target=target,
        subject_id=str(subject["subject_id"]),
    )
    controls.append(control_record_from_weights(
        "output_layer_no_signature_support_optimizer",
        optimized_weights,
        source,
        target,
        source_weights,
        optimized_meta,
    ))
    v17_stats = train_stats["v17_baseline_train_stats"]
    v17_weights, v17_meta = v17.select_layerwise_rank1_tsv_edit(
        source_weights=source_weights,
        subject_id=str(subject["subject_id"]),
        source=source,
        target=target,
        source_signature_norm=source_signature_norm,
        train_stats=v17_stats,
    )
    controls.append(control_record_from_weights(
        "v17_layerwise_rank1_tsv",
        v17_weights,
        source,
        target,
        source_weights,
        {**v17_meta, "baseline_train_statistics_hash": train_stats["v17_baseline_train_statistics_hash"]},
    ))
    v16_source_stats = v16.source_activation_stats(
        source_weights=source_weights,
        probe_examples=train_stats["probe_examples"],
    )
    v16_grid = v16.target_operator_grid_from_signature(
        train_stats=train_stats["v16_baseline_train_stats"],
        target_behavior=target,
        target_signature_norm=source_signature_norm,
    )
    v16_weights, v16_meta = v16.select_compiled_conceptor_edit(
        source_weights=source_weights,
        source=source,
        target=target,
        source_stats=v16_source_stats,
        target_operator_by_aperture=v16_grid,
    )
    controls.append(control_record_from_weights(
        "v16_output_layer_conceptor",
        v16_weights,
        source,
        target,
        source_weights,
        {**v16_meta, "baseline_train_statistics_hash": train_stats["v16_baseline_train_statistics_hash"]},
    ))
    v20_weights, v20_meta = select_tangent_nullspace_edit(
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        source=source,
        target=target,
        subject_id=str(subject["subject_id"]),
        train_stats=train_stats,
        control_type=V20_TANGENT_CONTROL_METHOD,
    )
    controls.append(control_record_from_weights(
        "v20_tangent_nullspace_editor_recomputed",
        v20_weights,
        source,
        target,
        source_weights,
        {key: value for key, value in v20_meta.items() if not key.startswith("_")},
    ))
    v21_weights, v21_meta = v21.select_behavioral_probe_residual_output_edit(
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        shuffled_signature_norm=shuffled_signature_norm,
        source=source,
        target=target,
        subject=subject,
        train_stats=train_stats["v21_baseline_train_stats"],
    )
    controls.append(control_record_from_weights(
        "v21_behavioral_probe_residual_output_editor_recomputed",
        v21_weights,
        source,
        target,
        source_weights,
        {
            **{key: value for key, value in v21_meta.items() if not key.startswith("_")},
            "baseline_train_statistics_hash": train_stats["v21_baseline_train_statistics_hash"],
        },
    ))
    v22_weights, v22_meta = v22.select_component_activation_rank1_edit(
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        source=source,
        target=target,
        subject=subject,
        train_stats=train_stats["v22_baseline_train_stats"],
        shuffled_hidden_descriptor=shuffled_hidden_descriptor,
    )
    controls.append(control_record_from_weights(
        "v22_component_activation_rank1_editor_recomputed",
        v22_weights,
        source,
        target,
        source_weights,
        {
            **{key: value for key, value in v22_meta.items() if not key.startswith("_")},
            "baseline_train_statistics_hash": train_stats["v22_baseline_train_statistics_hash"],
        },
    ))
    for control_type, descriptor_mode in [
        ("no_probe_sparse_subspace_editor", "no_probe"),
        ("source_probe_sparse_subspace_editor", "source_probe"),
        ("shuffled_probe_sparse_subspace_editor", "shuffled_probe"),
        ("target_label_only_sparse_subspace_editor", "target_label_only"),
        ("nearest_target_sparse_subspace_editor", "nearest_target_probe"),
    ]:
        weights, metadata = select_component_activation_rank1_edit(
            source_weights=source_weights,
            source_signature_norm=source_signature_norm,
            source=source,
            target=target,
            subject=subject,
            train_stats=train_stats,
            shuffled_hidden_descriptor=shuffled_hidden_descriptor,
            descriptor_mode=descriptor_mode,
        )
        controls.append(control_record_from_weights(
            control_type,
            weights,
            source,
            target,
            source_weights,
            {key: value for key, value in metadata.items() if not key.startswith("_")},
        ))
    subject_hash = stable_hash_json({
        "method": EDITOR_METHOD,
        "subject_id": str(subject["subject_id"]),
    })
    base_weights = matched_metadata["_base_weights"]
    selected_basis = matched_metadata["_selected_basis"]
    matched_hidden_delta = matched_metadata["_selected_hidden_delta"]
    selected_layers = matched_metadata["selected_layers"]
    for index in range(int(random_controls)):
        random_delta, metadata = random_sparse_subspace_delta(
            basis=selected_basis,
            matched_hidden_delta=matched_hidden_delta,
            selected_layers=selected_layers,
            seed_payload={
                "subject_hash": subject_hash,
                "source": source,
                "target": target,
                "selected_config_hash": matched_metadata["selected_config_hash"],
                "train_statistics_hash": train_stats["train_statistics_hash"],
                "index": index,
                "selected_layers": selected_layers,
            },
        )
        weights = base_weights + random_delta
        control_type = f"random_norm_matched_sparse_subspace_{index:02d}"
        metadata = {
            **metadata,
            "control_type": control_type,
            "index": int(index),
            "selected_config_hash": matched_metadata["selected_config_hash"],
            "selected_layers": selected_layers,
            "train_statistics_hash": train_stats["train_statistics_hash"],
        }
        controls.append(control_record_from_weights(
            control_type,
            weights,
            source,
            target,
            source_weights,
            metadata,
        ))
    return controls


def single_control(controls: Sequence[Mapping[str, Any]], control_type: str) -> Mapping[str, Any]:
    matches = [control for control in controls if control["control_type"] == control_type]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {control_type} control, found {len(matches)}")
    return matches[0]


def pareto_dominates(control: Mapping[str, float], matched: Mapping[str, float]) -> bool:
    return v17.pareto_dominates(control, matched)


def pareto_controls_for_record(controls: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        control for control in controls
        if (
            control["control_type"] in PROOF_CRITICAL_CONTROL_TYPES
            or str(control["control_type"]).startswith("random_norm_matched_sparse_subspace_")
        )
    ]


def individual_passed(matched: Mapping[str, Any]) -> bool:
    margin_advantage_passes = [
        matched[f"matched_minus_{metric_name}_target_margin"]
        >= THRESHOLDS["min_per_record_control_target_margin_advantage"]
        for metric_name in ADVANTAGE_CONTROL_TYPES
    ]
    return bool(
        matched["target_prediction_pass"]
        and matched["target_margin"] >= THRESHOLDS["min_per_record_target_margin"]
        and matched["conflict_target_accuracy"] >= THRESHOLDS["min_per_record_conflict_target_accuracy"]
        and matched["conflict_target_accuracy_improvement"]
        >= THRESHOLDS["min_per_record_conflict_target_accuracy_improvement"]
        and matched["pareto_undominated"]
        and matched["min_proof_critical_compatible_mse_advantage"]
        >= THRESHOLDS["min_per_record_control_compatible_mse_advantage"]
        and all(margin_advantage_passes)
    )


def evaluate_record(
    *,
    subject: Mapping[str, Any],
    source: str,
    target: str,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    shuffled_signature_norm: torch.Tensor | None,
    shuffled_hidden_descriptor: Mapping[str, list[torch.Tensor]] | None = None,
    train_stats: Mapping[str, Any],
    random_controls: int = RANDOM_CONTROLS_PER_RECORD,
) -> dict[str, Any]:
    matched_weights, matched_metadata = select_component_activation_rank1_edit(
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        shuffled_hidden_descriptor=shuffled_hidden_descriptor,
        source=source,
        target=target,
        subject=subject,
        train_stats=train_stats,
    )
    matched = {
        **v16.v15.v14.functional_metrics(matched_weights, source, target, source_weights),
        "delta_norm": float((matched_weights - source_weights).norm().item()),
        "editor": {key: value for key, value in matched_metadata.items() if not key.startswith("_")},
    }
    controls = build_controls(
        subject=subject,
        source=source,
        target=target,
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        shuffled_signature_norm=source_signature_norm if shuffled_signature_norm is None else shuffled_signature_norm,
        shuffled_hidden_descriptor=shuffled_hidden_descriptor,
        matched_weights=matched_weights,
        matched_metadata=matched_metadata,
        train_stats=train_stats,
        random_controls=random_controls,
    )
    gating_controls = [
        control for control in controls
        if control["control_type"] in PROOF_CRITICAL_CONTROL_TYPES
    ]
    pareto_controls = pareto_controls_for_record(controls)
    pareto_dominators = [
        control for control in pareto_controls if pareto_dominates(control, matched)
    ]
    best_target = max(gating_controls, key=lambda item: item["target_margin"])
    mse_advantages = [
        control["compatible_source_output_mse"] - matched["compatible_source_output_mse"]
        for control in gating_controls
    ]
    matched["pareto_dominator_count"] = len(pareto_dominators)
    matched["pareto_dominator_types"] = sorted({control["control_type"] for control in pareto_dominators})
    matched["min_proof_critical_compatible_mse_advantage"] = min(mse_advantages) if mse_advantages else 0.0
    matched["pareto_undominated"] = not pareto_dominators
    matched["target_prediction_pass"] = matched["predicted_behavior"] == target
    for metric_name, control_type in ADVANTAGE_CONTROL_TYPES.items():
        control = single_control(controls, control_type)
        matched[f"matched_minus_{metric_name}_target_margin"] = (
            matched["target_margin"] - control["target_margin"]
        )
        matched[f"{metric_name}_minus_matched_compatible_source_output_mse"] = (
            control["compatible_source_output_mse"] - matched["compatible_source_output_mse"]
        )
    matched["individual_all_gates_passed"] = individual_passed(matched)
    summary = {
        "best_control_target_margin": best_target["target_margin"],
        "best_control_type": best_target["control_type"],
        "matched_minus_best_control_target_margin": matched["target_margin"] - best_target["target_margin"],
        "pareto_undominated": matched["pareto_undominated"],
        "target_prediction_pass": matched["target_prediction_pass"],
    }
    for metric_name in ADVANTAGE_CONTROL_TYPES:
        summary[f"matched_minus_{metric_name}_target_margin"] = matched[
            f"matched_minus_{metric_name}_target_margin"
        ]
        summary[f"{metric_name}_minus_matched_compatible_source_output_mse"] = matched[
            f"{metric_name}_minus_matched_compatible_source_output_mse"
        ]
    return {
        "controls": v16.v15.v10.strip_control_weights(controls),
        "individual_all_gates_passed": matched["individual_all_gates_passed"],
        "matched": v16.v15.v10.strip_weight(matched),
        "random_control_count": sum(
            1 for control in controls
            if control["control_type"].startswith("random_norm_matched_sparse_subspace_")
        ),
        "source_behavior": source,
        "subject_id": str(subject["subject_id"]),
        "summary": summary,
        "target_behavior": target,
    }


_WORKER_TRAIN_STATS: Mapping[str, Any] | None = None
_WORKER_RANDOM_CONTROLS = RANDOM_CONTROLS_PER_RECORD
_WORKER_RECORD_EVALUATOR: Any = None


def evaluate_record_from_job(job: Mapping[str, Any], *, train_stats: Mapping[str, Any], random_controls: int) -> dict[str, Any]:
    subject = job["subject"]
    source = str(job["source"])
    target = str(job["target"])
    shuffled_signature_norm = job.get("shuffled_signature_norm")
    if shuffled_signature_norm is not None:
        shuffled_signature_norm = torch.tensor(shuffled_signature_norm, dtype=torch.float32)
    shuffled_hidden_descriptor = job.get("shuffled_hidden_descriptor")
    if shuffled_hidden_descriptor is not None:
        shuffled_hidden_descriptor = hidden_descriptor_from_hashable(shuffled_hidden_descriptor)
    source_weights = torch.tensor(subject["weights"], dtype=torch.float32)
    source_signature_norm = v17.normalized_signature(subject, train_stats)
    return evaluate_record(
        subject=subject,
        source=source,
        target=target,
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        shuffled_signature_norm=shuffled_signature_norm,
        shuffled_hidden_descriptor=shuffled_hidden_descriptor,
        train_stats=train_stats,
        random_controls=random_controls,
    )


def _init_eval_worker(train_stats: Mapping[str, Any], random_controls: int, record_evaluator: Any) -> None:
    global _WORKER_TRAIN_STATS, _WORKER_RANDOM_CONTROLS, _WORKER_RECORD_EVALUATOR
    torch.set_num_threads(1)
    _WORKER_TRAIN_STATS = train_stats
    _WORKER_RANDOM_CONTROLS = int(random_controls)
    _WORKER_RECORD_EVALUATOR = record_evaluator


def _evaluate_record_worker(job: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_TRAIN_STATS is None:
        raise RuntimeError("worker train stats not initialized")
    evaluator = _WORKER_RECORD_EVALUATOR or evaluate_record_from_job
    return evaluator(
        job,
        train_stats=_WORKER_TRAIN_STATS,
        random_controls=_WORKER_RANDOM_CONTROLS,
    )


def sort_records_for_artifact(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return v17.sort_records_for_artifact(records)


def mean(values: Sequence[float] | Any) -> float:
    return v17.mean(values)


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "n": 0,
            "individual_all_gate_pass_count": 0,
            "individual_all_gate_pass_rate": 0.0,
            "pareto_undominated_count": 0,
            "pareto_undominated_rate": 0.0,
            "scale_0_selection_count": 0,
            "scale_0_selection_rate": 0.0,
            "selected_k_counts": {},
            "selected_layer_counts": {},
            "selected_layer_entropy": 0.0,
            "selected_layer_set_counts": {},
            "target_prediction_count": 0,
            "target_prediction_rate": 0.0,
        }
    matched = [record["matched"] for record in records]
    summary = {
        "individual_all_gate_pass_count": sum(
            1 for record in records if record["individual_all_gates_passed"]
        ),
        "n": len(records),
        "pareto_undominated_count": sum(1 for item in matched if item["pareto_undominated"]),
        "scale_0_selection_count": sum(
            1 for item in matched if item.get("editor", {}).get("scale_0_selected") is True
        ),
        "target_prediction_count": sum(1 for item in matched if item["target_prediction_pass"]),
    }
    summary["individual_all_gate_pass_rate"] = summary["individual_all_gate_pass_count"] / len(records)
    summary["pareto_undominated_rate"] = summary["pareto_undominated_count"] / len(records)
    summary["scale_0_selection_rate"] = summary["scale_0_selection_count"] / len(records)
    summary["target_prediction_rate"] = summary["target_prediction_count"] / len(records)
    layer_counts: dict[str, int] = {}
    layer_set_counts: dict[str, int] = {}
    selected_k_counts: dict[str, int] = {}
    for item in matched:
        editor = item.get("editor", {})
        selected_layers = editor.get("selected_layers")
        if selected_layers is None:
            layer = editor.get("layer_index")
            selected_layers = [] if layer is None else [layer]
        selected_layers = [int(layer) for layer in selected_layers]
        if not selected_layers:
            continue
        selected_k = str(len(selected_layers))
        selected_k_counts[selected_k] = selected_k_counts.get(selected_k, 0) + 1
        layer_set_key = ",".join(str(layer) for layer in selected_layers)
        layer_set_counts[layer_set_key] = layer_set_counts.get(layer_set_key, 0) + 1
        for layer in selected_layers:
            layer_key = str(int(layer))
            layer_counts[layer_key] = layer_counts.get(layer_key, 0) + 1
    layer_entropy = 0.0
    for count in layer_counts.values():
        probability = float(count) / float(len(records))
        if probability > 0.0:
            layer_entropy -= probability * math.log(probability)
    summary["selected_k_counts"] = dict(sorted(selected_k_counts.items()))
    summary["selected_layer_counts"] = dict(sorted(layer_counts.items()))
    summary["selected_layer_entropy"] = float(layer_entropy)
    summary["selected_layer_set_counts"] = dict(sorted(layer_set_counts.items()))
    for key in [
        "conflict_target_accuracy",
        "conflict_target_accuracy_improvement",
        "target_margin",
        "matched_minus_best_control_target_margin",
    ]:
        source_values = matched if key != "matched_minus_best_control_target_margin" else [
            record["summary"] for record in records
        ]
        summary[f"mean_{key}"] = mean([item[key] for item in source_values])
    for metric_name in ADVANTAGE_CONTROL_TYPES:
        summary[f"mean_matched_minus_{metric_name}_target_margin"] = mean([
            item[f"matched_minus_{metric_name}_target_margin"] for item in matched
        ])
        summary[f"mean_{metric_name}_minus_matched_compatible_source_output_mse"] = mean([
            item[f"{metric_name}_minus_matched_compatible_source_output_mse"] for item in matched
        ])
    return summary


def require_at_least(failures: list[str], observed: float, expected: float, label: str) -> None:
    if observed < expected:
        failures.append(f"{label} {observed:.6f} < {expected:.6f}")


def require_equal(failures: list[str], observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        failures.append(f"{label} {observed!r} != {expected!r}")


def gate_failures(
    *,
    aggregate: Mapping[str, Any],
    by_direction: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    failures: list[str] = []
    require_equal(failures, aggregate["n"], THRESHOLDS["expected_record_count"], "record count")
    bad_random = [
        record["subject_id"] for record in records
        if record["random_control_count"] != THRESHOLDS["expected_random_controls_per_record"]
    ]
    if bad_random:
        failures.append(f"records with wrong random control count: {bad_random[:5]}")
    bad_total_controls = [
        record["subject_id"] for record in records
        if len(record.get("controls", ())) != THRESHOLDS["expected_controls_per_record"]
    ]
    if bad_total_controls:
        failures.append(f"records with wrong total control count: {bad_total_controls[:5]}")
    require_at_least(
        failures,
        aggregate["target_prediction_rate"],
        THRESHOLDS["min_aggregate_target_prediction_rate"],
        "aggregate target prediction rate",
    )
    require_at_least(
        failures,
        aggregate["individual_all_gate_pass_rate"],
        THRESHOLDS["min_aggregate_individual_pass_rate"],
        "aggregate individual pass rate",
    )
    require_at_least(
        failures,
        aggregate["pareto_undominated_rate"],
        THRESHOLDS["min_aggregate_pareto_rate"],
        "aggregate Pareto-undominated rate",
    )
    require_at_least(
        failures,
        aggregate["mean_target_margin"],
        THRESHOLDS["min_aggregate_target_margin"],
        "mean matched target margin",
    )
    require_at_least(
        failures,
        aggregate["mean_conflict_target_accuracy"],
        THRESHOLDS["min_aggregate_conflict_target_accuracy"],
        "aggregate conflict target accuracy",
    )
    require_at_least(
        failures,
        aggregate["mean_conflict_target_accuracy_improvement"],
        THRESHOLDS["min_aggregate_conflict_target_accuracy_improvement"],
        "aggregate conflict target accuracy improvement",
    )
    require_at_least(
        failures,
        aggregate["mean_matched_minus_best_control_target_margin"],
        THRESHOLDS["min_aggregate_best_control_target_margin_advantage"],
        "aggregate best-control target margin advantage",
    )
    for metric_name, threshold_key in [
        ("target_label", "min_aggregate_target_label_target_margin_advantage"),
        ("shuffled_signature", "min_aggregate_shuffled_signature_target_margin_advantage"),
        ("output_layer_no_signature", "min_aggregate_output_layer_no_signature_target_margin_advantage"),
        ("v16", "min_aggregate_v16_target_margin_advantage"),
        ("v17", "min_aggregate_v17_target_margin_advantage"),
        ("v20", "min_aggregate_v20_target_margin_advantage"),
        ("v21", "min_aggregate_v21_target_margin_advantage"),
        ("v22", "min_aggregate_v22_target_margin_advantage"),
    ]:
        require_at_least(
            failures,
            aggregate[f"mean_matched_minus_{metric_name}_target_margin"],
            THRESHOLDS[threshold_key],
            f"aggregate {metric_name} target margin advantage",
        )
    for direction, summary in sorted(by_direction.items()):
        require_at_least(
            failures,
            summary["target_prediction_rate"],
            THRESHOLDS["min_direction_target_prediction_rate"],
            f"{direction} target prediction rate",
        )
        require_at_least(
            failures,
            summary["individual_all_gate_pass_rate"],
            THRESHOLDS["min_direction_individual_pass_rate"],
            f"{direction} individual pass rate",
        )
        require_at_least(
            failures,
            summary["pareto_undominated_rate"],
            THRESHOLDS["min_direction_pareto_rate"],
            f"{direction} Pareto-undominated rate",
        )
        require_at_least(
            failures,
            summary["mean_target_margin"],
            THRESHOLDS["min_direction_target_margin"],
            f"{direction} target margin",
        )
        require_at_least(
            failures,
            summary["mean_matched_minus_output_layer_no_signature_target_margin"],
            THRESHOLDS["min_direction_output_layer_no_signature_target_margin_advantage"],
            f"{direction} output-layer target margin advantage",
        )
        for metric_name, threshold_key in [
            ("v16", "min_direction_v16_target_margin_advantage"),
            ("v17", "min_direction_v17_target_margin_advantage"),
            ("v20", "min_direction_v20_target_margin_advantage"),
            ("v21", "min_direction_v21_target_margin_advantage"),
            ("v22", "min_direction_v22_target_margin_advantage"),
        ]:
            require_at_least(
                failures,
                summary[f"mean_matched_minus_{metric_name}_target_margin"],
                THRESHOLDS[threshold_key],
                f"{direction} {metric_name} target margin advantage",
            )
    return failures


def evaluate_subjects(
    *,
    subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    random_controls: int = RANDOM_CONTROLS_PER_RECORD,
    parallel: bool = True,
    max_workers: int | None = None,
    record_evaluator: Any = None,
    progress_log_path: Path | None = None,
    progress_started_at_monotonic: float | None = None,
    progress_event_prefix: str = "evaluation",
) -> dict[str, Any]:
    jobs = []
    for subject in subjects:
        source = v16.subject_behavior(subject)
        for target in PATTERNS:
            if target != source:
                jobs.append({"source": source, "subject": subject, "target": target})
    assign_shuffled_signatures(jobs, train_stats)
    if progress_log_path is not None and progress_started_at_monotonic is not None:
        record_development_progress_event(
            progress_log_path,
            event=f"{progress_event_prefix}_jobs_queued",
            started_at_monotonic=progress_started_at_monotonic,
            extra={"record_count": len(jobs)},
        )
    if parallel and len(jobs) > 1:
        contract = multiprocessing_contract(max_workers=max_workers)
        context = mp.get_context(contract["start_method"])
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(contract["max_workers"]),
            mp_context=context,
            initializer=_init_eval_worker,
            initargs=(train_stats, int(random_controls), record_evaluator),
        ) as executor:
            futures = [
                executor.submit(_evaluate_record_worker, job)
                for job in jobs
            ]
            records = []
            for future in concurrent.futures.as_completed(futures):
                records.append(future.result())
                if progress_log_path is not None and progress_started_at_monotonic is not None:
                    record_development_progress_event(
                        progress_log_path,
                        event=f"{progress_event_prefix}_record_completed",
                        started_at_monotonic=progress_started_at_monotonic,
                        extra={
                            "completed_count": len(records),
                            "record_count": len(jobs),
                        },
                    )
    else:
        evaluator = record_evaluator or evaluate_record_from_job
        records = []
        for job in jobs:
            records.append(
                evaluator(job, train_stats=train_stats, random_controls=random_controls)
            )
            if progress_log_path is not None and progress_started_at_monotonic is not None:
                record_development_progress_event(
                    progress_log_path,
                    event=f"{progress_event_prefix}_record_completed",
                    started_at_monotonic=progress_started_at_monotonic,
                    extra={
                        "completed_count": len(records),
                        "record_count": len(jobs),
                    },
                )
    records = sort_records_for_artifact(records)
    aggregate = summarize_records(records)
    by_direction = {
        v17.direction_key(source, target): summarize_records([
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
        "multiprocessing_contract": multiprocessing_contract(max_workers=max_workers),
        "record_count": len(records),
        "records": records,
    }


def assign_shuffled_signatures(jobs: list[dict[str, Any]], train_stats: Mapping[str, Any]) -> None:
    grouped_for_shuffle: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for job in jobs:
        grouped_for_shuffle.setdefault((str(job["source"]), str(job["target"])), []).append(job)
    shuffled_signatures = {}
    shuffled_descriptors = {}
    for (_source, _target), group in sorted(grouped_for_shuffle.items()):
        sorted_group = sorted(
            group,
            key=lambda job: (
                str(job["source"]),
                str(job["subject"]["subject_id"]),
                str(job["target"]),
            ),
        )
        for index, job in enumerate(sorted_group):
            next_job = sorted_group[(index + 1) % len(sorted_group)]
            key = (
                str(job["source"]),
                str(job["subject"]["subject_id"]),
                str(job["target"]),
            )
            shuffled_signatures[key] = tensor_to_hashable(
                v17.normalized_signature(next_job["subject"], train_stats)
            )
            next_subject_id = str(next_job["subject"]["subject_id"])
            descriptor = train_stats["hidden_descriptor_by_subject"].get(next_subject_id)
            if descriptor is None:
                descriptor = hidden_rank1_descriptor_for_weights(
                    weights=torch.tensor(next_job["subject"]["weights"], dtype=torch.float32),
                    probe_examples=train_stats["probe_examples"],
                )
            shuffled_descriptors[key] = hidden_descriptor_to_hashable(descriptor)
    for job in jobs:
        key = (
            str(job["source"]),
            str(job["subject"]["subject_id"]),
            str(job["target"]),
        )
        job["shuffled_signature_norm"] = shuffled_signatures[key]
        job["shuffled_hidden_descriptor"] = shuffled_descriptors[key]


def multiprocessing_contract(max_workers: int | None = None) -> dict[str, Any]:
    default_workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    worker_count = default_workers if max_workers is None else int(max_workers)
    return {
        "max_workers": max(1, worker_count),
        "stable_record_sort_key": ["source_behavior", "subject_id", "target_behavior"],
        "start_method": "spawn",
        "torch_threads_per_worker": 1,
        "worker_writes_result_files": False,
    }


def build_v23_seed_preflight() -> dict[str, Any]:
    seed_ranges = []
    failures = []
    seen_ranges = []
    for pool_name, pool_config in POOL_CONFIGS.items():
        base_seed = int(pool_config["base_seed"])
        max_attempts = int(pool_config["max_attempts_per_behavior"])
        for behavior_index, pattern in enumerate(PATTERNS):
            start_seed = base_seed + behavior_index * int(SEED_BEHAVIOR_STRIDE)
            end_seed = start_seed + max_attempts - 1
            current = {
                "end_seed": int(end_seed),
                "pattern": pattern,
                "pool": pool_name,
                "start_seed": int(start_seed),
            }
            for previous in seen_ranges:
                overlaps = not (
                    current["end_seed"] < previous["start_seed"]
                    or previous["end_seed"] < current["start_seed"]
                )
                if overlaps:
                    failures.append({
                        "current": current,
                        "previous": previous,
                        "type": "seed_range_overlap",
                    })
            seen_ranges.append(current)
            seed_ranges.append(current)
    return {
        "failures": failures,
        "passed": not failures,
        "seed_ranges": seed_ranges,
    }


def forbidden_final_redacted_keys(payload: Mapping[str, Any]) -> list[str]:
    failures = []
    top_keys = set(payload.keys())
    if top_keys != FINAL_REDACTED_ALLOWED_TOP_LEVEL_KEYS:
        for key in sorted(top_keys - FINAL_REDACTED_ALLOWED_TOP_LEVEL_KEYS):
            failures.append(f"top_level.{key}")
        for key in sorted(FINAL_REDACTED_ALLOWED_TOP_LEVEL_KEYS - top_keys):
            failures.append(f"top_level_missing.{key}")
    summary = payload.get("summary", {})
    if isinstance(summary, Mapping):
        summary_keys = set(summary.keys())
        if summary_keys != FINAL_REDACTED_ALLOWED_SUMMARY_KEYS:
            for key in sorted(summary_keys - FINAL_REDACTED_ALLOWED_SUMMARY_KEYS):
                failures.append(f"summary.{key}")
            for key in sorted(FINAL_REDACTED_ALLOWED_SUMMARY_KEYS - summary_keys):
                failures.append(f"summary_missing.{key}")
    else:
        failures.append("summary_not_mapping")
    failures.extend(v16.forbidden_final_redacted_keys(payload))
    return sorted(set(failures))


def forbidden_combined_final_summary_keys(summary: Mapping[str, Any]) -> list[str]:
    keys = set(summary)
    failures = []
    for key in sorted(keys - FINAL_COMBINED_SUMMARY_ALLOWED_KEYS):
        failures.append(key)
    for key in sorted(FINAL_COMBINED_SUMMARY_ALLOWED_KEYS - keys):
        failures.append(f"missing:{key}")
    return failures


def assert_no_forbidden_final_raw_paths(paths: Sequence[Path | str], *, allow_v23_final: bool = False) -> None:
    prior = {path.resolve() for path in PRIOR_FINAL_RAW_PATHS}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in prior:
            raise ValueError(f"prior sealed final raw path is forbidden: {path}")
        if path.name == "final_subjects.json" and "runs" in path.parts:
            if not (allow_v23_final and path == V23_FINAL_RAW.resolve()):
                raise ValueError(f"sealed final raw path is forbidden: {path}")
        if path == V23_FINAL_RAW.resolve() and not allow_v23_final:
            raise ValueError(f"V23 final raw path is forbidden before authorization: {path}")


def validate_source_pool_contract(
    *,
    train_path: Path,
    eval_path: Path,
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
    final_summary = combined_audit.get("pool_summaries", {}).get("final", {})
    final_keys = forbidden_combined_final_summary_keys(final_summary)
    if final_keys:
        failures.append(
            "combined_audit.pool_summaries.final key mismatch: "
            + ", ".join(final_keys)
        )
    redacted_keys = forbidden_final_redacted_keys(final_redacted)
    if redacted_keys:
        failures.append(
            "final_redacted_audit exposes forbidden keys: "
            + ", ".join(redacted_keys)
        )
    summaries = combined_audit.get("pool_summaries", {})
    expected_train_counts = {pattern: 64 for pattern in PATTERNS}
    expected_eval_counts = {pattern: 24 for pattern in PATTERNS}
    if summaries.get("train", {}).get("accepted_counts_by_behavior") != expected_train_counts:
        failures.append("train accepted counts mismatch")
    eval_name = "development" if phase == "development" else "final"
    if summaries.get(eval_name, {}).get("accepted_counts_by_behavior") != expected_eval_counts:
        failures.append(f"{eval_name} accepted counts mismatch")
    if summaries.get("train", {}).get("pool_file_sha256") != v16.v15.v1.sha256_file(train_path):
        failures.append("train pool hash mismatch")
    if phase == "development":
        if summaries.get("development", {}).get("pool_file_sha256") != v16.v15.v1.sha256_file(eval_path):
            failures.append("development pool hash mismatch")
    if combined_audit.get("passed") is not True:
        failures.append("combined audit did not pass")
    assert_no_forbidden_final_raw_paths([train_path, eval_path], allow_v23_final=(phase == "final"))
    return failures


def generate_pools(args: argparse.Namespace, pool_dir: Path) -> dict[str, Any]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    seed_preflight = build_v23_seed_preflight()
    if seed_preflight["failures"]:
        result = {"failures": seed_preflight["failures"], "passed": False}
        (pool_dir / "combined_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        return result
    suite = v16.v15.build_suite(args.support_per_class, args.heldout_per_class)
    heldout_sequences = v16.v15.build_heldout_sequences(suite)
    candidate_pools = v16.v15.build_candidate_pools(heldout_sequences)
    candidate_pool_summary = v16.v15.summarize_candidate_pools(candidate_pools)
    probe_examples = build_probe_examples()
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
        payload = v16.v15.poolgen.generate_pool(
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
            v16.v15.poolgen.redact_weights_and_signatures(payload)
        )
        pool_payloads[pool_name] = payload
        pool_path = pool_dir / f"{pool_name}_subjects.json"
        pool_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        summary = v16.v15.poolgen.summarize_pool(payload)
        summary["pool_file_sha256"] = v16.v15.v1.sha256_file(pool_path)
        summary["pool_redacted_payload_sha256"] = payload["pool_redacted_payload_sha256"]
        pool_summaries[pool_name] = summary

    final_redacted = v16.v15.poolgen.build_final_redacted_summary(pool_payloads["final"])
    final_redacted["claim_scope"] = FINAL_REDACTED_SCOPE
    final_redacted["pool_file_sha256"] = pool_summaries["final"]["pool_file_sha256"]
    final_redacted["summary_payload_sha256"] = stable_hash_json(final_redacted)
    forbidden_redacted = forbidden_final_redacted_keys(final_redacted)
    if forbidden_redacted:
        raise ValueError(
            "final_redacted_audit exposes forbidden keys: "
            + ", ".join(forbidden_redacted)
        )
    (pool_dir / "final_redacted_audit.json").write_text(
        json.dumps(final_redacted, indent=2, sort_keys=True)
    )

    audit = v16.v15.poolgen.build_combined_audit(
        pool_payloads=pool_payloads,
        pool_summaries=pool_summaries,
        seed_preflight=seed_preflight,
        suite=suite,
        probe_examples=probe_examples,
    )
    audit["claim_scope"] = SOURCE_AUDIT_SCOPE
    audit = v16.v15.v10.redact_combined_audit(audit)
    final_summary = audit.get("pool_summaries", {}).get("final", {})
    audit["pool_summaries"]["final"] = {
        key: final_summary[key]
        for key in sorted(FINAL_COMBINED_SUMMARY_ALLOWED_KEYS)
        if key in final_summary
    }
    forbidden_final_summary = forbidden_combined_final_summary_keys(
        audit["pool_summaries"]["final"]
    )
    if forbidden_final_summary:
        raise ValueError(
            "combined_audit.pool_summaries.final key mismatch: "
            + ", ".join(forbidden_final_summary)
        )
    audit["audit_payload_sha256"] = stable_hash_json(audit)
    (pool_dir / "combined_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    return {
        "combined_audit_path": v16.v15.v1.rel(pool_dir / "combined_audit.json"),
        "final_redacted_audit_path": v16.v15.v1.rel(pool_dir / "final_redacted_audit.json"),
        "passed": bool(audit["passed"]),
        "pool_summaries": audit["pool_summaries"],
        "seed_preflight": seed_preflight,
    }


def summary_for_stdout(result: Mapping[str, Any]) -> dict[str, Any]:
    pool_summaries = result.get("pool_summaries")
    redacted_pool_summaries = None
    if isinstance(pool_summaries, Mapping):
        redacted_pool_summaries = {}
        for pool_name, summary in pool_summaries.items():
            if not isinstance(summary, Mapping):
                continue
            redacted_pool_summaries[pool_name] = {
                key: summary[key]
                for key in (
                    "accepted_counts_by_behavior",
                    "pool_file_sha256",
                    "pool_redacted_payload_sha256",
                    "record_count",
                )
                if key in summary
            }
    keys = [
        "aggregate",
        "claim_scope",
        "combined_audit_path",
        "development_results_path",
        "failures",
        "final_redacted_audit_path",
        "next_action",
        "passed",
        "phase",
        "pool_summaries",
        "record_count",
        "seed_preflight",
    ]
    summary = {key: result[key] for key in keys if key in result and key != "pool_summaries"}
    if redacted_pool_summaries is not None:
        summary["pool_summaries"] = redacted_pool_summaries
    return summary


def serializable_stats_artifact(train_stats: Mapping[str, Any], max_workers: int | None) -> dict[str, Any]:
    return {
        "sparse_subspace_config_hash": train_stats.get("selected_sparse_config_hash"),
        "constants_hash": stable_hash_json(constants_payload()),
        "global_hidden_centroid_hashes": [
            stable_hash_json(tensor_to_hashable(value))
            for value in train_stats.get("global_hidden_centroids", [])
        ],
        "hidden_descriptor_count": len(train_stats.get("hidden_descriptor_hashes", {})),
        "hidden_descriptor_hashes_hash": stable_hash_json(
            train_stats.get("hidden_descriptor_hashes", {})
        ),
        "hidden_target_centroid_hashes": {
            behavior: [
                stable_hash_json(tensor_to_hashable(value))
                for value in centroids
            ]
            for behavior, centroids in sorted(
                train_stats.get("hidden_target_centroids", {}).items()
            )
        },
        "inner_validation_amendment_path": str(INNER_VALIDATION_AMENDMENT_PATH.relative_to(REPO_ROOT)),
        "inner_validation_amendment_sha256": INNER_VALIDATION_AMENDMENT_SHA256,
        "inner_validation_rung_record_budgets": INNER_VALIDATION_RUNG_RECORD_BUDGETS,
        "inner_validation_rung_subjects_per_behavior": INNER_VALIDATION_RUNG_SUBJECTS_PER_BEHAVIOR,
        "inner_validation_rung_survivors": INNER_VALIDATION_RUNG_SURVIVORS,
        "inner_validation_rung_summaries": train_stats.get("selected_sparse_config", {}).get(
            "inner_validation_rung_summaries"
        ),
        "inner_validation_selection_hash": train_stats.get("selected_sparse_config", {}).get(
            "inner_validation_selection_hash"
        ),
        "inner_validation_evaluated_config_count": train_stats.get("selected_sparse_config", {}).get(
            "inner_validation_evaluated_config_count",
            INNER_VALIDATION_EVALUATED_CONFIG_COUNT,
        ),
        "inner_validation_evaluated_config_subset_hash": train_stats.get(
            "selected_sparse_config", {}
        ).get("inner_validation_evaluated_config_subset_hash"),
        "inner_validation_total_config_count": train_stats.get("selected_sparse_config", {}).get(
            "inner_validation_total_config_count",
            INNER_VALIDATION_TOTAL_CONFIG_COUNT,
        ),
        "method": EDITOR_METHOD,
        "multiprocessing_contract": multiprocessing_contract(max_workers=max_workers),
        "probe_examples_hash": train_stats["probe_examples_hash"],
        "selected_sparse_config": train_stats.get("selected_sparse_config"),
        "selected_sparse_config_hash": train_stats.get("selected_sparse_config_hash"),
        "target_probe_logit_centroid_hashes": train_stats.get("target_probe_logit_centroid_hashes"),
        "thresholds_hash": stable_hash_json(THRESHOLDS),
        "train_statistics_hash": train_stats["train_statistics_hash"],
        "v16_baseline_train_statistics_hash": train_stats.get("v16_baseline_train_statistics_hash"),
        "v17_baseline_train_statistics_hash": train_stats.get("v17_baseline_train_statistics_hash"),
        "v21_baseline_train_statistics_hash": train_stats.get("v21_baseline_train_statistics_hash"),
        "v22_baseline_train_statistics_hash": train_stats.get("v22_baseline_train_statistics_hash"),
    }


def run_development(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> dict[str, Any]:
    development_started_at = time.monotonic()
    development_progress_log_path = output_dir / DEVELOPMENT_PROGRESS_LOG_FILENAME
    record_development_progress_event(
        development_progress_log_path,
        event="development_start",
        started_at_monotonic=development_started_at,
        extra={
            "max_workers": args.max_workers,
            "pool_dir": v16.v15.v1.rel(pool_dir),
            "output_dir": v16.v15.v1.rel(output_dir),
        },
    )
    train_path = pool_dir / "train_subjects.json"
    eval_path = pool_dir / "development_subjects.json"
    combined_audit_path = pool_dir / "combined_audit.json"
    final_redacted_path = pool_dir / "final_redacted_audit.json"
    assert_no_forbidden_final_raw_paths(
        [train_path, eval_path, combined_audit_path, final_redacted_path],
        allow_v23_final=False,
    )
    train_payload = v16.v15.v1.load_json(train_path)
    eval_payload = v16.v15.v1.load_json(eval_path)
    combined_audit = v16.v15.v1.load_json(combined_audit_path)
    final_redacted = v16.v15.v1.load_json(final_redacted_path)
    record_development_progress_event(
        development_progress_log_path,
        event="source_payloads_loaded",
        started_at_monotonic=development_started_at,
        extra={
            "train_pool_sha256": v16.v15.v1.sha256_file(train_path),
            "eval_pool_sha256": v16.v15.v1.sha256_file(eval_path),
        },
    )
    contract_failures = validate_source_pool_contract(
        train_path=train_path,
        eval_path=eval_path,
        train_payload=train_payload,
        eval_payload=eval_payload,
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )
    if contract_failures:
        raise ValueError("V23 source-pool contract validation failed: " + "; ".join(contract_failures))
    record_development_progress_event(
        development_progress_log_path,
        event="source_pool_contract_validated",
        started_at_monotonic=development_started_at,
        extra={"contract_failure_count": 0},
    )
    train_subjects = v16.v15.v1.accepted_records(train_payload)
    eval_subjects = v16.v15.v1.accepted_records(eval_payload)
    record_development_progress_event(
        development_progress_log_path,
        event="train_statistics_start",
        started_at_monotonic=development_started_at,
        extra={
            "train_subject_count": len(train_subjects),
            "eval_subject_count": len(eval_subjects),
        },
    )
    train_stats = fit_v23_train_statistics(
        train_subjects,
        include_models=True,
        include_baseline_stats=True,
        max_inner_workers=args.max_workers,
        inner_validation_progress_dir=output_dir,
    )
    record_development_progress_event(
        development_progress_log_path,
        event="train_statistics_completed",
        started_at_monotonic=development_started_at,
        extra={
            "selected_sparse_config_hash": train_stats.get("selected_sparse_config_hash"),
            "train_statistics_hash": train_stats.get("train_statistics_hash"),
        },
    )
    stats_path = output_dir / "v23_probe_routed_sparse_subspace_editor_stats.pt"
    torch.save(serializable_stats_artifact(train_stats, args.max_workers), stats_path)
    record_development_progress_event(
        development_progress_log_path,
        event="stats_artifact_written",
        started_at_monotonic=development_started_at,
        extra={"stats_path": v16.v15.v1.rel(stats_path)},
    )
    record_development_progress_event(
        development_progress_log_path,
        event="development_evaluation_start",
        started_at_monotonic=development_started_at,
        extra={"eval_subject_count": len(eval_subjects)},
    )
    eval_result = evaluate_subjects(
        subjects=eval_subjects,
        train_stats=train_stats,
        random_controls=RANDOM_CONTROLS_PER_RECORD,
        parallel=True,
        max_workers=args.max_workers,
        progress_log_path=development_progress_log_path,
        progress_started_at_monotonic=development_started_at,
        progress_event_prefix="development_evaluation",
    )
    record_development_progress_event(
        development_progress_log_path,
        event="development_evaluation_completed",
        started_at_monotonic=development_started_at,
        extra={
            "record_count": eval_result.get("record_count"),
            "failure_count": len(eval_result.get("failures", [])),
        },
    )
    failures = [*contract_failures, *eval_result["failures"]]
    result = {
        **eval_result,
        "claim_scope": DEVELOPMENT_SCOPE,
        "combined_audit_path": v16.v15.v1.rel(combined_audit_path),
        "combined_audit_sha256": v16.v15.v1.sha256_file(combined_audit_path),
        "constants_sha256": stable_hash_json(constants_payload()),
        "development_results_path": v16.v15.v1.rel(output_dir / "development_results.json"),
        "dirty_worktree_caveat": True,
        "editor_method": EDITOR_METHOD,
        "eval_pool_path": v16.v15.v1.rel(eval_path),
        "eval_pool_sha256": v16.v15.v1.sha256_file(eval_path),
        "final_redacted_audit_path": v16.v15.v1.rel(final_redacted_path),
        "final_redacted_audit_sha256": v16.v15.v1.sha256_file(final_redacted_path),
        "final_redacted_summary": final_redacted,
        "formal_prereg_sha256": v16.v15.v1.sha256_file(PREREG_PATH),
        "helper_tests_sha256": v16.v15.v1.sha256_file(HELPER_TEST_PATH),
        "inner_validation_amendment_path": str(INNER_VALIDATION_AMENDMENT_PATH.relative_to(REPO_ROOT)),
        "inner_validation_amendment_sha256": INNER_VALIDATION_AMENDMENT_SHA256,
        "inner_validation_checkpoint_path": v16.v15.v1.rel(
            output_dir / INNER_VALIDATION_CHECKPOINT_FILENAME
        ),
        "inner_validation_checkpoint_sha256": v16.v15.v1.sha256_file(
            output_dir / INNER_VALIDATION_CHECKPOINT_FILENAME
        ),
        "inner_validation_progress_log_path": v16.v15.v1.rel(
            output_dir / INNER_VALIDATION_PROGRESS_LOG_FILENAME
        ),
        "inner_validation_progress_log_sha256": v16.v15.v1.sha256_file(
            output_dir / INNER_VALIDATION_PROGRESS_LOG_FILENAME
        ),
        "inner_validation_rung_record_budgets": INNER_VALIDATION_RUNG_RECORD_BUDGETS,
        "inner_validation_rung_subjects_per_behavior": INNER_VALIDATION_RUNG_SUBJECTS_PER_BEHAVIOR,
        "inner_validation_rung_survivors": INNER_VALIDATION_RUNG_SURVIVORS,
        "inner_validation_rung_summaries": train_stats.get("selected_sparse_config", {}).get(
            "inner_validation_rung_summaries"
        ),
        "inner_validation_selection_hash": train_stats.get("selected_sparse_config", {}).get(
            "inner_validation_selection_hash"
        ),
        "inner_validation_evaluated_config_count": train_stats.get("selected_sparse_config", {}).get(
            "inner_validation_evaluated_config_count",
            INNER_VALIDATION_EVALUATED_CONFIG_COUNT,
        ),
        "inner_validation_evaluated_config_subset_hash": train_stats.get(
            "selected_sparse_config", {}
        ).get("inner_validation_evaluated_config_subset_hash"),
        "inner_validation_total_config_count": INNER_VALIDATION_TOTAL_CONFIG_COUNT,
        "implementation_sha256": v16.v15.v1.sha256_file(SCRIPT_PATH),
        "development_progress_log_path": v16.v15.v1.rel(development_progress_log_path),
        "development_progress_log_sha256": v16.v15.v1.sha256_file(
            development_progress_log_path
        ),
        "limitations": (
            "Small-subject source-label-known, target-label-requested component activation "
            "sparse-subspace functional editing evidence only; not larger-model proof."
        ),
        "phase": "development",
        "pool_overlap_counts": combined_audit.get("overlap_counts", {}),
        "probe_examples_hash": train_stats["probe_examples_hash"],
        "source_pool_audit_passed": bool(combined_audit.get("passed", False)),
        "stats_path": v16.v15.v1.rel(stats_path),
        "stats_sha256": v16.v15.v1.sha256_file(stats_path),
        "thresholds": THRESHOLDS,
        "thresholds_sha256": stable_hash_json(THRESHOLDS),
        "train_pool_path": v16.v15.v1.rel(train_path),
        "train_pool_sha256": v16.v15.v1.sha256_file(train_path),
        "train_statistics_hash": train_stats["train_statistics_hash"],
        "selected_sparse_config": train_stats.get("selected_sparse_config"),
        "selected_sparse_config_hash": train_stats.get("selected_sparse_config_hash"),
        "v16_baseline_train_statistics_hash": train_stats.get("v16_baseline_train_statistics_hash"),
        "v17_baseline_train_statistics_hash": train_stats.get("v17_baseline_train_statistics_hash"),
        "v21_baseline_train_statistics_hash": train_stats.get("v21_baseline_train_statistics_hash"),
        "v22_baseline_train_statistics_hash": train_stats.get("v22_baseline_train_statistics_hash"),
    }
    result["failures"] = failures
    result["passed"] = not failures
    result["next_action"] = (
        PASSING_DEVELOPMENT_NEXT_ACTION
        if result["passed"]
        else FAILING_DEVELOPMENT_NEXT_ACTION
    )
    output_path = output_dir / "development_results.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    result["development_results_sha256"] = v16.v15.v1.sha256_file(output_path)
    record_development_progress_event(
        development_progress_log_path,
        event="development_results_written",
        started_at_monotonic=development_started_at,
        extra={
            "development_results_path": v16.v15.v1.rel(output_path),
            "passed": result["passed"],
        },
    )
    result["development_progress_log_sha256"] = v16.v15.v1.sha256_file(
        development_progress_log_path
    )
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    result["development_results_sha256"] = v16.v15.v1.sha256_file(output_path)
    return result


def run_final(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> dict[str, Any]:
    raise NotImplementedError(
        "V23 final is not implemented until development passes and reviewer authorizes final"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["generate-pools", "development", "final"],
        required=True,
    )
    parser.add_argument("--pool-dir", default=str(DEFAULT_POOL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--train-epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--source-margin-gate", type=float, default=0.40)
    parser.add_argument("--support-per-class", type=int, default=160)
    parser.add_argument("--heldout-per-class", type=int, default=64)
    parser.add_argument("--positive-cap", type=int, default=2048)
    parser.add_argument("--hard-negative-cap", type=int, default=1024)
    parser.add_argument("--generic-negative-cap", type=int, default=1024)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--summary-only-stdout", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool_dir = REPO_ROOT / args.pool_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "generate-pools":
        result = generate_pools(args, pool_dir)
    elif args.phase == "development":
        result = run_development(args, pool_dir, output_dir)
    else:
        result = run_final(args, pool_dir, output_dir)
    print(json.dumps(summary_for_stdout(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
