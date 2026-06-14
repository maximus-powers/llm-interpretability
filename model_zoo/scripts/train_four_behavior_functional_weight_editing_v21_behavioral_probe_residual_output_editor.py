"""V21 functional editing via behavioral-probe residual output editors."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
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


v16 = v17.v16
PATTERNS = v17.PATTERNS

SEED_BEHAVIOR_STRIDE = 100000
POOL_CONFIGS = {
    "train": {
        "base_seed": 114400000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 115400000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 116400000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}
DEFAULT_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v21_pools"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runs"
    / "four_behavior_functional_weight_editing_v21_behavioral_probe_residual_output_editor"
)
PREREG_PATH = (
    REPO_ROOT
    / "docs"
    / "preregistrations"
    / "four_behavior_functional_weight_editing_v21_behavioral_probe_residual_output_editor.md"
)
SCRIPT_PATH = Path(__file__).resolve()
HELPER_TEST_PATH = (
    REPO_ROOT
    / "model_zoo"
    / "scripts"
    / "test_four_behavior_functional_weight_editing_v21_helpers.py"
)
SOURCE_POOL_SCOPE = "four_behavior_functional_weight_editing_v21_source_pool"
SOURCE_AUDIT_SCOPE = "four_behavior_functional_weight_editing_v21_source_pool_construction"
FINAL_REDACTED_SCOPE = (
    "redacted_final_functional_weight_editing_v21_source_pool_audit_surface_only"
)
DEVELOPMENT_SCOPE = (
    "four_behavior_functional_weight_editing_v21_behavioral_probe_residual_output_editor_development"
)
FINAL_SCOPE = (
    "four_behavior_functional_weight_editing_v21_behavioral_probe_residual_output_editor_final"
)
EDITOR_METHOD = "behavioral_probe_residual_output_editor_v21"
V20_TANGENT_CONTROL_METHOD = "signature_conditioned_tangent_nullspace_editor_v20"
V21_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"
PRIOR_FINAL_RAW_PATHS = {
    v17.v16.v15.V15_FINAL_RAW,
    v17.v16.V16_FINAL_RAW,
    v17.V17_FINAL_RAW,
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v18_pools" / "final_subjects.json",
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v19_pools" / "final_subjects.json",
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v20_pools" / "final_subjects.json",
}

SOURCE_WEIGHT_DIM = 345
SIGNATURE_DIM = 560
SIGNATURE_TOP_K = 8
SIGNATURE_TEMPERATURE = 1.0
DESIRED_LOGIT_MAGNITUDE = 2.0
DECODER_RIDGE_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]
RESIDUAL_TARGET_NORMALIZATIONS = ["none", "per_component_train_std"]
RESIDUAL_NORM_CAP_MULTIPLIERS = [0.25, 0.5, 1.0]
RESIDUAL_SCALE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
ORACLE_PROBE_WEIGHTS = [0.25, 0.5, 1.0]
ORACLE_BASE_LAMBDAS = [0.01, 0.1, 1.0, 10.0]
NULLSPACE_RELATIVE_CUTOFF = 1e-4
NULLSPACE_ABSOLUTE_CUTOFF = 1e-6
MASK_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
RIDGE_LAMBDAS = [0.01, 0.1, 1.0, 10.0]
PRIOR_LAMBDAS = [0.0, 0.01, 0.1, 1.0]
ACTIVATION_SCALE_GRID = [0.0, 0.5, 1.0]
POST_SCALE_GRID = [0.5, 0.75, 1.0, 1.25]
MAX_TANGENT_DELTA_NORM = 8.0
RANDOM_CONTROLS_PER_RECORD = 16
EXPECTED_CONTROLS_PER_RECORD = 27
RANDOM_CONTROL_EPS = 1e-12
PARETO_EPSILON = 1e-9

LOSS_WEIGHTS = {
    "compatible_source_mse": 1.0,
    "conflict_bce": 1.0,
    "target_bce": 1.0,
}
THRESHOLDS = {
    "expected_record_count": 288,
    "expected_controls_per_record": 27,
    "expected_random_controls_per_record": 16,
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
    "min_aggregate_v17_target_margin_advantage": 0.02,
    "min_direction_target_prediction_rate": 0.65,
    "min_direction_individual_pass_rate": 0.65,
    "min_direction_pareto_rate": 0.75,
    "min_direction_target_margin": 0.15,
    "min_direction_output_layer_no_signature_target_margin_advantage": 0.01,
    "min_direction_v17_target_margin_advantage": 0.01,
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
    "no_probe_residual_output_editor",
    "source_probe_residual_output_editor",
    "shuffled_probe_residual_output_editor",
    "target_label_only_residual_output_editor",
    "nearest_target_probe_residual_output_editor",
    "oracle_train_centroid_probe_residual_output_editor",
]
ADVANTAGE_CONTROL_TYPES = {
    "no_signature": "no_probe_residual_output_editor",
    "target_label": "target_label_only_residual_output_editor",
    "source_signature": "source_probe_residual_output_editor",
    "shuffled_signature": "shuffled_probe_residual_output_editor",
    "v17": "v17_layerwise_rank1_tsv",
    "v16": "v16_output_layer_conceptor",
    "v20": "v20_tangent_nullspace_editor_recomputed",
    "output_layer_no_signature": "output_layer_no_signature_support_optimizer",
    "nearest_target": "nearest_target_probe_residual_output_editor",
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
        "desired_logit_magnitude": DESIRED_LOGIT_MAGNITUDE,
        "descriptor_clamp": [-10.0, 10.0],
        "decoder_ridge_lambdas": DECODER_RIDGE_LAMBDAS,
        "editor_method": EDITOR_METHOD,
        "expected_controls_per_record": EXPECTED_CONTROLS_PER_RECORD,
        "loss_weights": LOSS_WEIGHTS,
        "oracle_lambda_base_grid": ORACLE_BASE_LAMBDAS,
        "oracle_probe_weight_grid": ORACLE_PROBE_WEIGHTS,
        "output_layer_theta_dim": 9,
        "random_controls_per_record": RANDOM_CONTROLS_PER_RECORD,
        "residual_norm_cap_multipliers": RESIDUAL_NORM_CAP_MULTIPLIERS,
        "residual_scale_grid": RESIDUAL_SCALE_GRID,
        "residual_target_normalizations": RESIDUAL_TARGET_NORMALIZATIONS,
        "source_weight_dim": SOURCE_WEIGHT_DIM,
        "thresholds": THRESHOLDS,
        "v20_tangent_control_method": V20_TANGENT_CONTROL_METHOD,
    }


def one_hot(index: int, size: int) -> torch.Tensor:
    value = torch.zeros(size, dtype=torch.float32)
    value[int(index)] = 1.0
    return value


def behavior_centroids(records: Sequence[Mapping[str, Any]], train_stats: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    grouped = v17.records_by_behavior(records)
    centroids = {}
    for behavior, items in grouped.items():
        signatures = [v17.normalized_signature(item, train_stats) for item in items]
        centroids[behavior] = torch.stack(signatures).mean(dim=0)
    return centroids


def full_train_statistics_hash_payload(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "constants": constants_payload(),
        "decoder_config_hash": stats.get("decoder_config_hash"),
        "decoder_input_mean": tensor_to_hashable(stats["decoder_input_mean"]),
        "decoder_input_std": tensor_to_hashable(stats["decoder_input_std"]),
        "decoder_input_zero_variance_count": stats.get("decoder_input_zero_variance_count"),
        "decoder_pair_rows_hash": stats.get("decoder_pair_rows_hash"),
        "decoder_weights": tensor_to_hashable(stats["decoder_weights"]),
        "inner_split_hash": stats.get("inner_split_hash"),
        "probe_descriptor_centroids": {
            key: tensor_to_hashable(value)
            for key, value in sorted(stats.get("probe_descriptor_centroids", {}).items())
        },
        "probe_descriptor_mean": tensor_to_hashable(stats["probe_descriptor_mean"]),
        "probe_descriptor_std": tensor_to_hashable(stats["probe_descriptor_std"]),
        "probe_descriptor_zero_variance_count": stats.get("probe_descriptor_zero_variance_count"),
        "probe_examples_hash": stats.get("probe_examples_hash", "missing"),
        "residual_norm_cap_multiplier": stats.get("residual_norm_cap_multiplier"),
        "residual_output_mean": tensor_to_hashable(stats["residual_output_mean"]),
        "residual_output_std": tensor_to_hashable(stats["residual_output_std"]),
        "residual_output_zero_variance_count": stats.get("residual_output_zero_variance_count"),
        "selected_decoder_config": stats.get("selected_decoder_config"),
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
    }


def sorted_train_subjects(subjects: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(subjects, key=lambda item: (v16.subject_behavior(item), str(item["subject_id"])))


def inner_split_by_behavior(subjects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = v17.records_by_behavior(subjects)
    inner_train = []
    inner_validation = []
    split_records = {}
    for behavior in PATTERNS:
        scored = []
        for record in grouped[behavior]:
            subject_id = str(record["subject_id"])
            split_hash = stable_hash_json({
                "behavior": behavior,
                "scope": "four_behavior_functional_weight_editing_v21_inner_split",
                "subject_id": subject_id,
            })
            scored.append((split_hash, subject_id, record))
        scored.sort(key=lambda item: (item[0], item[1]))
        train_part = [item[2] for item in scored[:51]]
        validation_part = [item[2] for item in scored[51:64]]
        if len(train_part) != 51 or len(validation_part) != 13:
            raise ValueError(f"unexpected inner split size for {behavior}")
        inner_train.extend(train_part)
        inner_validation.extend(validation_part)
        split_records[behavior] = {
            "inner_train_subject_id_hashes": [
                hashlib.sha256(str(item["subject_id"]).encode("utf-8")).hexdigest()
                for item in train_part
            ],
            "inner_validation_subject_id_hashes": [
                hashlib.sha256(str(item["subject_id"]).encode("utf-8")).hexdigest()
                for item in validation_part
            ],
        }
    train_ids = {str(item["subject_id"]) for item in inner_train}
    validation_ids = {str(item["subject_id"]) for item in inner_validation}
    if train_ids & validation_ids:
        raise ValueError("inner train/validation split is not disjoint")
    split_payload = {
        "per_behavior": split_records,
        "scope": "four_behavior_functional_weight_editing_v21_inner_split",
    }
    return {
        "inner_train": sorted_train_subjects(inner_train),
        "inner_validation": sorted_train_subjects(inner_validation),
        "split_hash": stable_hash_json(split_payload),
        "split_payload": split_payload,
    }


def record_weights_tensor(record: Mapping[str, Any]) -> torch.Tensor:
    return torch.tensor(record["weights"], dtype=torch.float32)


def penultimate_design_matrix(source_weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    flat = source_weights.detach().to(dtype=torch.float32).reshape(1, -1)
    hidden = v16.v15.hidden_activations_flat_batch(flat, inputs)[-1][0].to(dtype=torch.float32)
    return torch.cat([hidden, torch.ones(hidden.shape[0], 1, dtype=torch.float32)], dim=1)


def probe_logits_for_weights(weights: torch.Tensor, probe_examples: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    inputs = v16.probe_inputs_tensor(probe_examples)
    logits = v16.v15.v10.decoder_v1.subject_forward_flat_batch(
        weights.detach().to(dtype=torch.float32).reshape(1, -1),
        inputs,
    )[0]
    return logits.reshape(-1).to(dtype=torch.float32)


def safe_mean_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    zero_mask = std < 1e-6
    std = torch.where(zero_mask, torch.ones_like(std), std)
    return mean.to(dtype=torch.float32), std.to(dtype=torch.float32), int(zero_mask.sum().item())


def fit_probe_descriptor_stats(
    subjects: Sequence[Mapping[str, Any]],
    *,
    probe_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    by_subject = {}
    for record in subjects:
        weights = record_weights_tensor(record)
        raw = concatenate_descriptor_components(raw_probe_descriptor_components(
            weights=weights,
            probe_examples=probe_examples,
            output_signature=torch.tensor(record["signature"], dtype=torch.float32),
        ))
        rows.append(raw)
        by_subject[str(record["subject_id"])] = raw
    matrix = torch.stack(rows)
    mean, std, zero_count = safe_mean_std(matrix)
    normalized_by_subject = {
        subject_id: ((raw - mean) / std).clamp(-10.0, 10.0)
        for subject_id, raw in by_subject.items()
    }
    centroids = {}
    for behavior, records in v17.records_by_behavior(subjects).items():
        centroids[behavior] = torch.stack([
            normalized_by_subject[str(record["subject_id"])]
            for record in records
        ]).mean(dim=0)
    return {
        "probe_descriptor_mean": mean,
        "probe_descriptor_std": std,
        "probe_descriptor_zero_variance_count": zero_count,
        "probe_descriptor_by_subject": normalized_by_subject,
        "probe_descriptor_centroids": centroids,
    }


def solve_oracle_output_layer_theta(
    *,
    source_weights: torch.Tensor,
    base_weights: torch.Tensor,
    source: str,
    target: str,
    target_probe_logit_centroid: torch.Tensor,
    probe_examples: Sequence[Mapping[str, Any]],
    probe_weight: float,
    lambda_base: float,
) -> torch.Tensor | None:
    support = v16.v15.v14.prepare_support_tensors_with_source_logits(
        source_weights=source_weights,
        source=source,
        target=target,
    )
    probe_inputs = v16.probe_inputs_tensor(probe_examples)
    blocks = []
    targets = []
    for x, y, block_weight in [
        (support["target_inputs"], torch.where(
            support["target_labels"] > 0.5,
            torch.full_like(support["target_labels"], DESIRED_LOGIT_MAGNITUDE),
            torch.full_like(support["target_labels"], -DESIRED_LOGIT_MAGNITUDE),
        ), 1.0),
        (support["compatible_inputs"], support["compatible_source_logits"], 1.0),
        (probe_inputs, target_probe_logit_centroid.to(dtype=torch.float32), float(probe_weight)),
    ]:
        design = penultimate_design_matrix(source_weights, x)
        y_vec = y.reshape(-1, 1).to(dtype=torch.float32)
        row_weight = math.sqrt(float(block_weight) / max(1, int(design.shape[0])))
        blocks.append(design * row_weight)
        targets.append(y_vec * row_weight)
    x_all = torch.cat(blocks, dim=0)
    y_all = torch.cat(targets, dim=0)
    theta_base = output_layer_theta(base_weights).reshape(1, -1)
    x_aug = torch.cat([
        x_all,
        math.sqrt(float(lambda_base) / 9.0) * torch.eye(9, dtype=torch.float32),
    ], dim=0)
    y_aug = torch.cat([
        y_all,
        math.sqrt(float(lambda_base) / 9.0) * theta_base.T,
    ], dim=0)
    return solve_ridge_regression(
        x=x_aug,
        y=y_aug,
        ridge_lambda=0.0,
        penalize=torch.zeros(9, dtype=torch.bool),
    )


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


def oracle_output_layer_theta(
    *,
    source_weights: torch.Tensor,
    base_weights: torch.Tensor,
    source: str,
    target: str,
    target_probe_logit_centroid: torch.Tensor,
    probe_examples: Sequence[Mapping[str, Any]] | None = None,
) -> torch.Tensor | None:
    examples = list(probe_examples) if probe_examples is not None else build_probe_examples()
    candidates = []
    candidate_index = 0
    for probe_weight in ORACLE_PROBE_WEIGHTS:
        for lambda_base in ORACLE_BASE_LAMBDAS:
            theta = solve_oracle_output_layer_theta(
                source_weights=source_weights,
                base_weights=base_weights,
                source=source,
                target=target,
                target_probe_logit_centroid=target_probe_logit_centroid,
                probe_examples=examples,
                probe_weight=float(probe_weight),
                lambda_base=float(lambda_base),
            )
            if theta is None:
                candidate_index += 1
                continue
            flat_theta = theta.reshape(-1).to(dtype=torch.float32)
            weights = replace_output_layer_theta(base_weights, flat_theta)
            support = v17.support_objective_for_weights(
                weights=weights,
                source_weights=source_weights,
                source=source,
                target=target,
            )
            probe_loss = float(F.mse_loss(
                probe_logits_for_weights(weights, examples),
                target_probe_logit_centroid.to(dtype=torch.float32),
            ).item())
            candidates.append({
                "candidate_index": int(candidate_index),
                "lambda_base": float(lambda_base),
                "probe_centroid_loss": probe_loss,
                "probe_weight": float(probe_weight),
                "support_objective": float(support["objective"]),
                "theta": flat_theta,
                "theta_delta_norm": float((flat_theta - output_layer_theta(base_weights)).norm().item()),
            })
            candidate_index += 1
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda item: (
            item["support_objective"],
            item["probe_centroid_loss"],
            item["theta_delta_norm"],
            item["probe_weight"],
            item["lambda_base"],
            item["candidate_index"],
        ),
    )
    return best["theta"]


def build_v21_decoder_pairs(
    subjects: Sequence[Mapping[str, Any]],
    *,
    train_stats: Mapping[str, Any],
    include_models: bool,
) -> list[dict[str, Any]]:
    pairs = []
    train_by_behavior = train_stats["train_by_behavior"]
    target_probe_logit_centroids = train_stats["target_probe_logit_centroids"]
    for record in sorted_train_subjects(subjects):
        source = v16.subject_behavior(record)
        source_weights = record_weights_tensor(record)
        for target in PATTERNS:
            if target == source:
                continue
            if include_models:
                base_weights, _base_meta = v16.output_layer_no_signature_support_optimizer(
                    source_weights=source_weights,
                    source=source,
                    target=target,
                    subject_id=str(record["subject_id"]),
                )
                oracle_theta = oracle_output_layer_theta(
                    source_weights=source_weights,
                    base_weights=base_weights,
                    source=source,
                    target=target,
                    target_probe_logit_centroid=target_probe_logit_centroids[target],
                    probe_examples=train_stats["probe_examples"],
                )
                if oracle_theta is None:
                    continue
                oracle_theta = oracle_theta.reshape(-1)
            else:
                base_weights = source_weights
                target_thetas = torch.stack([
                    output_layer_theta(record_weights_tensor(target_record))
                    for target_record in train_by_behavior[target]
                ])
                oracle_theta = target_thetas.mean(dim=0)
            decoder_input = build_decoder_input_vector(
                subject=record,
                source_weights=source_weights,
                base_weights=base_weights,
                source=source,
                target=target,
                train_stats=train_stats,
            )
            pairs.append({
                "base_weights": base_weights,
                "decoder_input": decoder_input,
                "row": {
                    "source": source,
                    "subject_hash": stable_hash_json({"subject_id": str(record["subject_id"])}),
                    "target": target,
                },
                "source": source,
                "source_weights": source_weights,
                "subject": record,
                "subject_id": str(record["subject_id"]),
                "target": target,
                "target_residual": oracle_theta - output_layer_theta(base_weights),
            })
    return pairs


def fit_decoder_from_pairs(
    pairs: Sequence[Mapping[str, Any]],
    *,
    ridge_lambda: float,
    residual_target_normalization: str,
) -> dict[str, Any] | None:
    if not pairs:
        return None
    x_raw = torch.stack([item["decoder_input"].to(dtype=torch.float32) for item in pairs])
    y_raw = torch.stack([item["target_residual"].to(dtype=torch.float32) for item in pairs])
    x_mean, x_std, x_zero = safe_mean_std(x_raw)
    x_norm = (x_raw - x_mean) / x_std
    if residual_target_normalization == "per_component_train_std":
        y_mean, y_std, y_zero = safe_mean_std(y_raw)
        y_train = (y_raw - y_mean) / y_std
    elif residual_target_normalization == "none":
        y_mean = torch.zeros(9, dtype=torch.float32)
        y_std = torch.ones(9, dtype=torch.float32)
        y_zero = 0
        y_train = y_raw
    else:
        raise ValueError(f"unknown residual target normalization: {residual_target_normalization}")
    design = torch.cat([torch.ones(x_norm.shape[0], 1), x_norm], dim=1)
    penalize = torch.ones(design.shape[1], dtype=torch.bool)
    penalize[0] = False
    decoder_weights = solve_ridge_regression(
        x=design,
        y=y_train,
        ridge_lambda=float(ridge_lambda),
        penalize=penalize,
    )
    if decoder_weights is None:
        return None
    return {
        "decoder_input_mean": x_mean,
        "decoder_input_std": x_std,
        "decoder_input_zero_variance_count": x_zero,
        "decoder_weights": decoder_weights,
        "residual_output_mean": y_mean,
        "residual_output_std": y_std,
        "residual_output_zero_variance_count": y_zero,
    }


def evaluate_decoder_config_on_pairs(
    pairs: Sequence[Mapping[str, Any]],
    *,
    decoder_payload: Mapping[str, Any],
    residual_norm_cap_multiplier: float,
    train_stats: Mapping[str, Any],
) -> dict[str, float]:
    if not pairs:
        return {
            "mean_compatible_mse": float("inf"),
            "mean_objective": float("inf"),
            "mean_probe_centroid_loss": float("inf"),
            "mean_target_margin": float("-inf"),
            "scale_0_rate": 1.0,
        }
    local_stats = {
        **train_stats,
        **decoder_payload,
        "residual_norm_cap_multiplier": float(residual_norm_cap_multiplier),
    }
    objectives = []
    probe_losses = []
    compatible_mses = []
    residual_norms = []
    target_margins = []
    scale_0_count = 0
    for pair in pairs:
        predicted_residual = predict_residual_from_decoder(
            decoder_input=pair["decoder_input"],
            train_stats=local_stats,
        )
        base_theta = output_layer_theta(pair["base_weights"])
        source_theta = output_layer_theta(pair["source_weights"])
        cap = float(residual_norm_cap_multiplier) * max(
            float((base_theta - source_theta).norm().item()),
            RANDOM_CONTROL_EPS,
        )
        if cap <= 0.0:
            clipped_residual = torch.zeros(9, dtype=torch.float32)
        elif float(predicted_residual.norm().item()) > cap:
            clipped_residual = predicted_residual / predicted_residual.norm() * cap
        else:
            clipped_residual = predicted_residual
        candidates = []
        for candidate_index, scale in enumerate(RESIDUAL_SCALE_GRID):
            residual = float(scale) * clipped_residual
            weights = replace_output_layer_theta(pair["source_weights"], base_theta + residual)
            support = v17.support_objective_for_weights(
                weights=weights,
                source_weights=pair["source_weights"],
                source=str(pair["source"]),
                target=str(pair["target"]),
            )
            metrics = v16.v15.v14.functional_metrics(
                weights,
                str(pair["source"]),
                str(pair["target"]),
                pair["source_weights"],
            )
            candidates.append({
                "candidate_index": int(candidate_index),
                "compatible_mse": float(support["compatible_mse"]),
                "objective": float(support["objective"]),
                "probe_centroid_loss": probe_centroid_loss_for_weights(
                    weights=weights,
                    target=str(pair["target"]),
                    train_stats=train_stats,
                ),
                "residual_norm": float(residual.norm().item()),
                "scale": float(scale),
                "target_margin": float(metrics["target_margin"]),
            })
        best = min(
            candidates,
            key=lambda item: (
                item["objective"],
                item["probe_centroid_loss"],
                item["compatible_mse"],
                -item["target_margin"],
                item["residual_norm"],
                item["scale"],
                item["candidate_index"],
            ),
        )
        objectives.append(best["objective"])
        probe_losses.append(best["probe_centroid_loss"])
        compatible_mses.append(best["compatible_mse"])
        residual_norms.append(best["residual_norm"])
        target_margins.append(best["target_margin"])
        scale_0_count += int(best["scale"] == 0.0)
    n = float(len(pairs))
    return {
        "mean_compatible_mse": float(sum(compatible_mses) / n),
        "mean_objective": float(sum(objectives) / n),
        "mean_probe_centroid_loss": float(sum(probe_losses) / n),
        "mean_residual_norm": float(sum(residual_norms) / n),
        "mean_target_margin": float(sum(target_margins) / n),
        "scale_0_rate": float(scale_0_count / n),
    }


def select_decoder_config(
    pairs: Sequence[Mapping[str, Any]],
    *,
    train_stats: Mapping[str, Any],
    split: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if split is None:
        config = {
            "decoder": "ridge",
            "ridge_lambda": 1.0,
            "residual_norm_cap_multiplier": 1.0,
            "residual_target_normalization": "per_component_train_std",
            "selection_mode": "default_small_pool",
        }
        return config, {"candidate_count": 1, "selected_score": None}
    train_ids = {str(item["subject_id"]) for item in split["inner_train"]}
    validation_ids = {str(item["subject_id"]) for item in split["inner_validation"]}
    inner_train_pairs = [item for item in pairs if str(item["subject_id"]) in train_ids]
    inner_validation_pairs = [item for item in pairs if str(item["subject_id"]) in validation_ids]
    candidates = []
    candidate_index = 0
    for residual_target_normalization in RESIDUAL_TARGET_NORMALIZATIONS:
        for ridge_lambda in DECODER_RIDGE_LAMBDAS:
            decoder_payload = fit_decoder_from_pairs(
                inner_train_pairs,
                ridge_lambda=float(ridge_lambda),
                residual_target_normalization=residual_target_normalization,
            )
            if decoder_payload is None:
                candidate_index += 1
                continue
            for cap_multiplier in RESIDUAL_NORM_CAP_MULTIPLIERS:
                score = evaluate_decoder_config_on_pairs(
                    inner_validation_pairs,
                    decoder_payload=decoder_payload,
                    residual_norm_cap_multiplier=float(cap_multiplier),
                    train_stats=train_stats,
                )
                candidates.append({
                    **score,
                    "candidate_index": int(candidate_index),
                    "decoder": "ridge",
                    "ridge_lambda": float(ridge_lambda),
                    "residual_norm_cap_multiplier": float(cap_multiplier),
                    "residual_target_normalization": residual_target_normalization,
                    "selection_mode": "inner_validation",
                })
            candidate_index += 1
    if not candidates:
        raise ValueError("no valid V21 inner-validation decoder candidates")
    best = min(
        candidates,
        key=lambda item: (
            item["mean_objective"],
            item["mean_probe_centroid_loss"],
            item["mean_residual_norm"],
            item["ridge_lambda"],
            RESIDUAL_TARGET_NORMALIZATIONS.index(str(item["residual_target_normalization"])),
            item["residual_norm_cap_multiplier"],
            item["candidate_index"],
        ),
    )
    config = {
        "decoder": "ridge",
        "ridge_lambda": float(best["ridge_lambda"]),
        "residual_norm_cap_multiplier": float(best["residual_norm_cap_multiplier"]),
        "residual_target_normalization": str(best["residual_target_normalization"]),
        "selection_mode": "inner_validation",
    }
    diagnostics = {
        "candidate_count": len(candidates),
        "selected_candidate_index": int(best["candidate_index"]),
        "selected_score": {
            key: best[key]
            for key in [
                "mean_compatible_mse",
                "mean_objective",
                "mean_probe_centroid_loss",
                "mean_residual_norm",
                "mean_target_margin",
                "scale_0_rate",
            ]
        },
    }
    return config, diagnostics


def fit_v21_train_statistics(
    subjects: Sequence[Mapping[str, Any]],
    *,
    include_models: bool = True,
    include_baseline_stats: bool = True,
    allow_default_small_pool: bool = False,
) -> dict[str, Any]:
    ordered_subjects = sorted_train_subjects(subjects)
    signatures = torch.tensor([record["signature"] for record in ordered_subjects], dtype=torch.float32)
    sig_mean, sig_std, _zero_sig = safe_mean_std(signatures)
    probe_examples = build_probe_examples()
    descriptor_stats = fit_probe_descriptor_stats(ordered_subjects, probe_examples=probe_examples)
    train_by_behavior = v17.records_by_behavior(ordered_subjects)
    target_probe_logit_centroids = {
        behavior: torch.stack([
            probe_logits_for_weights(record_weights_tensor(record), probe_examples)
            for record in records
        ]).mean(dim=0)
        for behavior, records in train_by_behavior.items()
    }
    stats: dict[str, Any] = {
        **descriptor_stats,
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
    pairs = build_v21_decoder_pairs(
        ordered_subjects,
        train_stats=stats,
        include_models=include_models,
    )
    if not pairs:
        raise ValueError("no valid V21 decoder train pairs")
    split = None
    try:
        split = inner_split_by_behavior(ordered_subjects)
    except ValueError as exc:
        if not allow_default_small_pool:
            raise ValueError(
                "V21 decoder hyperparameter selection requires a 64-per-behavior "
                "train pool; pass allow_default_small_pool=True only for tests/smokes"
            ) from exc
    selected_config, selection_diagnostics = select_decoder_config(
        pairs,
        train_stats=stats,
        split=split,
    )
    decoder_payload = fit_decoder_from_pairs(
        pairs,
        ridge_lambda=float(selected_config["ridge_lambda"]),
        residual_target_normalization=str(selected_config["residual_target_normalization"]),
    )
    if decoder_payload is None:
        raise ValueError("V21 decoder ridge solve failed")
    pair_rows = [item["row"] for item in pairs]
    stats.update({
        **decoder_payload,
        "decoder_config_hash": stable_hash_json(selected_config),
        "decoder_pair_rows_hash": stable_hash_json(pair_rows),
        "inner_split_hash": split["split_hash"] if split is not None else "not_available_small_pool",
        "residual_norm_cap_multiplier": float(selected_config["residual_norm_cap_multiplier"]),
        "selected_decoder_config": selected_config,
        "selected_decoder_diagnostics": selection_diagnostics,
    })
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
    else:
        stats["v16_baseline_train_statistics_hash"] = "not_computed"
        stats["v17_baseline_train_statistics_hash"] = "not_computed"
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


def base_edit_summary_vector(
    *,
    base_weights: torch.Tensor,
    source_weights: torch.Tensor,
    source: str,
    target: str,
) -> torch.Tensor:
    support = v17.support_objective_for_weights(
        weights=base_weights,
        source_weights=source_weights,
        source=source,
        target=target,
    )
    metrics = v16.v15.v14.functional_metrics(base_weights, source, target, source_weights)
    output_delta_norm = (output_layer_theta(base_weights) - output_layer_theta(source_weights)).norm()
    return torch.tensor(
        [
            float(support["objective"]),
            float(support["target_bce"]),
            float(support["conflict_bce"]),
            float(support["compatible_mse"]),
            float(support["source_l2"]),
            float(metrics["target_margin"]),
            float(metrics["conflict_target_accuracy"]),
            float(output_delta_norm.item()),
        ],
        dtype=torch.float32,
    )


def raw_probe_descriptor_components(
    *,
    weights: torch.Tensor,
    probe_examples: Sequence[Mapping[str, Any]],
    output_signature: torch.Tensor,
) -> dict[str, torch.Tensor]:
    flat = weights.detach().to(dtype=torch.float32).reshape(1, -1)
    if int(flat.shape[1]) != SOURCE_WEIGHT_DIM:
        raise ValueError(f"expected {SOURCE_WEIGHT_DIM} weights, got {int(flat.shape[1])}")
    probe_inputs = v16.probe_inputs_tensor(probe_examples)
    hidden_layers = v16.v15.hidden_activations_flat_batch(flat, probe_inputs)
    penultimate = hidden_layers[-1][0].detach().to(dtype=torch.float32)
    logits = v16.v15.v10.decoder_v1.subject_forward_flat_batch(flat, probe_inputs)[0].reshape(-1)
    gram = penultimate.T @ penultimate / max(1, int(penultimate.shape[0]))
    upper = torch.triu_indices(8, 8)
    return {
        "output_signature": output_signature.detach().to(dtype=torch.float32).reshape(-1).clone(),
        "penultimate_mean": penultimate.mean(dim=0).to(dtype=torch.float32),
        "penultimate_std": penultimate.std(dim=0, unbiased=False).to(dtype=torch.float32),
        "penultimate_gram_upper": gram[upper[0], upper[1]].to(dtype=torch.float32),
        "output_logit_stats": torch.tensor(
            [
                float(logits.mean().item()),
                float(logits.std(unbiased=False).item()),
                float(logits.min().item()),
                float(logits.max().item()),
            ],
            dtype=torch.float32,
        ),
        "output_logit_gram": torch.tensor(
            [float(torch.mean(logits ** 2).item())],
            dtype=torch.float32,
        ),
    }


def solve_ridge_regression(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    ridge_lambda: float,
    penalize: torch.Tensor,
) -> torch.Tensor | None:
    features = x.detach().to(dtype=torch.float32)
    targets = y.detach().to(dtype=torch.float32)
    penalty_mask = penalize.detach().to(dtype=torch.float32).reshape(-1)
    if features.ndim != 2:
        raise ValueError("x must be rank 2")
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    if targets.ndim != 2:
        raise ValueError("y must be rank 1 or 2")
    if int(features.shape[0]) != int(targets.shape[0]):
        raise ValueError("x and y row counts differ")
    if int(features.shape[1]) != int(penalty_mask.numel()):
        raise ValueError("penalize length must match x feature dim")
    penalty = torch.diag(penalty_mask) * float(ridge_lambda)
    lhs = features.T @ features + penalty
    rhs = features.T @ targets
    try:
        return torch.linalg.solve(lhs, rhs).to(dtype=torch.float32)
    except RuntimeError:
        try:
            eye = torch.eye(int(lhs.shape[0]), dtype=torch.float32)
            return torch.linalg.solve(lhs + 1e-6 * eye, rhs).to(dtype=torch.float32)
        except RuntimeError:
            return None


def concatenate_descriptor_components(components: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([
        components["output_signature"].reshape(-1),
        components["penultimate_mean"].reshape(-1),
        components["penultimate_std"].reshape(-1),
        components["penultimate_gram_upper"].reshape(-1),
        components["output_logit_stats"].reshape(-1),
        components["output_logit_gram"].reshape(-1),
    ]).to(dtype=torch.float32)


def normalized_probe_descriptor(
    *,
    subject: Mapping[str, Any],
    weights: torch.Tensor,
    train_stats: Mapping[str, Any],
    output_signature: torch.Tensor | None = None,
) -> torch.Tensor:
    signature = (
        output_signature.detach().to(dtype=torch.float32)
        if output_signature is not None
        else torch.tensor(subject["signature"], dtype=torch.float32)
    )
    components = raw_probe_descriptor_components(
        weights=weights,
        probe_examples=train_stats["probe_examples"],
        output_signature=signature,
    )
    raw = concatenate_descriptor_components(components)
    mean = train_stats.get("probe_descriptor_mean")
    std = train_stats.get("probe_descriptor_std")
    if mean is None or std is None:
        return raw
    return ((raw - mean.to(dtype=torch.float32)) / std.to(dtype=torch.float32)).clamp(-10.0, 10.0)


def nearest_target_probe_descriptor(
    *,
    descriptor: torch.Tensor,
    target: str,
    train_stats: Mapping[str, Any],
) -> torch.Tensor:
    descriptors = train_stats.get("probe_descriptor_by_subject", {})
    target_records = train_stats.get("train_by_behavior", {}).get(target, [])
    candidates = []
    for record in target_records:
        candidate = descriptors.get(str(record["subject_id"]))
        if candidate is None:
            continue
        candidate_tensor = candidate.to(dtype=torch.float32)
        candidates.append((
            float(torch.norm(candidate_tensor - descriptor).item()),
            str(record["subject_id"]),
            candidate_tensor,
        ))
    if not candidates:
        centroid = train_stats.get("probe_descriptor_centroids", {}).get(target)
        if centroid is None:
            return torch.zeros_like(descriptor)
        return centroid.to(dtype=torch.float32)
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def descriptor_components_for_mode(
    *,
    subject: Mapping[str, Any],
    source_weights: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    descriptor_mode: str,
    shuffled_signature_norm: torch.Tensor | None = None,
    shuffled_probe_descriptor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    matched_descriptor = normalized_probe_descriptor(
        subject=subject,
        weights=source_weights,
        train_stats=train_stats,
    )
    source_centroid = train_stats.get("probe_descriptor_centroids", {}).get(source)
    target_centroid = train_stats.get("probe_descriptor_centroids", {}).get(target)
    zero_descriptor = torch.zeros_like(matched_descriptor)
    if source_centroid is None or target_centroid is None:
        matched_delta = zero_descriptor
    else:
        matched_delta = target_centroid.to(dtype=torch.float32) - source_centroid.to(dtype=torch.float32)

    if descriptor_mode == "matched_probe":
        return matched_descriptor, matched_delta
    if descriptor_mode == "no_probe":
        return zero_descriptor, zero_descriptor
    if descriptor_mode == "source_probe":
        return matched_descriptor, zero_descriptor
    if descriptor_mode == "target_label_only":
        return zero_descriptor, zero_descriptor
    if descriptor_mode == "oracle_train_centroid_probe":
        if target_centroid is None:
            return zero_descriptor, zero_descriptor
        return target_centroid.to(dtype=torch.float32), zero_descriptor
    if descriptor_mode == "nearest_target_probe":
        return nearest_target_probe_descriptor(
            descriptor=matched_descriptor,
            target=target,
            train_stats=train_stats,
        ), zero_descriptor
    if descriptor_mode == "shuffled_probe":
        del shuffled_signature_norm
        if shuffled_probe_descriptor is None:
            return zero_descriptor, zero_descriptor
        return shuffled_probe_descriptor.to(dtype=torch.float32).reshape_as(matched_descriptor), zero_descriptor
    raise ValueError(f"unknown V21 descriptor mode: {descriptor_mode}")


def build_decoder_input_vector(
    *,
    subject: Mapping[str, Any],
    source_weights: torch.Tensor,
    base_weights: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    descriptor_mode: str = "matched_probe",
    shuffled_signature_norm: torch.Tensor | None = None,
    shuffled_probe_descriptor: torch.Tensor | None = None,
) -> torch.Tensor:
    descriptor, centroid_delta = descriptor_components_for_mode(
        subject=subject,
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats,
        descriptor_mode=descriptor_mode,
        shuffled_signature_norm=shuffled_signature_norm,
        shuffled_probe_descriptor=shuffled_probe_descriptor,
    )
    target_hot = one_hot(PATTERNS.index(target), len(PATTERNS))
    base_summary = base_edit_summary_vector(
        base_weights=base_weights,
        source_weights=source_weights,
        source=source,
        target=target,
    )
    return torch.cat([descriptor, centroid_delta, target_hot, base_summary]).to(dtype=torch.float32)


def predict_residual_from_decoder(
    *,
    decoder_input: torch.Tensor,
    train_stats: Mapping[str, Any],
) -> torch.Tensor:
    weights = train_stats.get("decoder_weights")
    if weights is None:
        return torch.zeros(9, dtype=torch.float32)
    x = decoder_input.detach().to(dtype=torch.float32).reshape(-1)
    mean = train_stats.get("decoder_input_mean", torch.zeros_like(x)).to(dtype=torch.float32)
    std = train_stats.get("decoder_input_std", torch.ones_like(x)).to(dtype=torch.float32)
    x_norm = (x - mean) / std.clamp_min(1e-6)
    design = torch.cat([torch.ones(1, dtype=torch.float32), x_norm])
    prediction = design @ weights.to(dtype=torch.float32)
    output_mean = train_stats.get("residual_output_mean", torch.zeros(9, dtype=torch.float32)).to(dtype=torch.float32)
    output_std = train_stats.get("residual_output_std", torch.ones(9, dtype=torch.float32)).to(dtype=torch.float32)
    return (prediction.reshape(-1) * output_std + output_mean).to(dtype=torch.float32)


def random_norm_matched_residual(
    *,
    selected_residual: torch.Tensor,
    subject_hash: str,
    source: str,
    target: str,
    index: int,
    train_statistics_hash: str,
    decoder_config_hash: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    residual = selected_residual.detach().to(dtype=torch.float32).reshape(-1)
    if int(residual.numel()) != 9:
        raise ValueError(f"expected selected residual dim 9, got {int(residual.numel())}")
    norm = residual.norm()
    norm_hash = stable_hash_json(tensor_to_hashable(norm.reshape(1)))
    seed_payload = {
        "decoder_config_hash": str(decoder_config_hash),
        "index": int(index),
        "matched_residual_norm_hash": norm_hash,
        "method": EDITOR_METHOD,
        "source": str(source),
        "subject_hash": str(subject_hash),
        "target": str(target),
        "train_statistics_hash": str(train_statistics_hash),
    }
    seed_hash = stable_hash_json(seed_payload)
    seed = int(seed_hash[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(9, generator=generator, dtype=torch.float32)
    raw_norm = raw.norm()
    if float(raw_norm.item()) < RANDOM_CONTROL_EPS:
        raw = torch.zeros(9, dtype=torch.float32)
        raw[0] = 1.0
        raw_norm = raw.norm()
    if float(norm.item()) < RANDOM_CONTROL_EPS:
        random_residual = torch.zeros(9, dtype=torch.float32)
    else:
        random_residual = raw / raw_norm * norm
    metadata = {
        "control_type": f"random_norm_matched_probe_residual_{int(index):02d}",
        "decoder_config_hash": str(decoder_config_hash),
        "index": int(index),
        "matched_residual_norm": float(norm.item()),
        "random_residual_norm": float(random_residual.norm().item()),
        "random_seed": int(seed),
        "seed_hash": seed_hash,
        "train_statistics_hash": str(train_statistics_hash),
    }
    return random_residual, metadata


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
        "activation_residual_hash": stable_hash_json(tensor_to_hashable(activation_delta)),
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


def select_behavioral_probe_residual_output_edit(
    *,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    source: str,
    target: str,
    subject: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    shuffled_signature_norm: torch.Tensor | None = None,
    shuffled_probe_descriptor: torch.Tensor | None = None,
    descriptor_mode: str = "matched_probe",
) -> tuple[torch.Tensor, dict[str, Any]]:
    del source_signature_norm
    base_weights, base_metadata = v16.output_layer_no_signature_support_optimizer(
        source_weights=source_weights,
        source=source,
        target=target,
        subject_id=str(subject["subject_id"]),
    )
    base_theta = output_layer_theta(base_weights)
    if train_stats.get("decoder_weights") is None:
        predicted_residual = torch.zeros(9, dtype=torch.float32)
    else:
        decoder_input = build_decoder_input_vector(
            subject=subject,
            source_weights=source_weights,
            base_weights=base_weights,
            source=source,
            target=target,
            train_stats=train_stats,
            descriptor_mode=descriptor_mode,
            shuffled_signature_norm=shuffled_signature_norm,
            shuffled_probe_descriptor=shuffled_probe_descriptor,
        )
        predicted_residual = predict_residual_from_decoder(
            decoder_input=decoder_input,
            train_stats=train_stats,
        )
    base_delta_norm = (base_theta - output_layer_theta(source_weights)).norm()
    cap_multiplier = float(train_stats.get("residual_norm_cap_multiplier", 0.0))
    residual_cap = cap_multiplier * max(float(base_delta_norm.item()), RANDOM_CONTROL_EPS)
    clipped = False
    if residual_cap <= 0.0:
        clipped_residual = torch.zeros(9, dtype=torch.float32)
    elif float(predicted_residual.norm().item()) > residual_cap:
        clipped_residual = predicted_residual / predicted_residual.norm() * residual_cap
        clipped = True
    else:
        clipped_residual = predicted_residual
    candidates = []
    for candidate_index, scale in enumerate(RESIDUAL_SCALE_GRID):
        residual = float(scale) * clipped_residual
        weights = replace_output_layer_theta(source_weights, base_theta + residual)
        support = v17.support_objective_for_weights(
            weights=weights,
            source_weights=source_weights,
            source=source,
            target=target,
        )
        candidates.append({
            "candidate_index": int(candidate_index),
            "compatible_mse": float(support["compatible_mse"]),
            "objective": float(support["objective"]),
            "probe_centroid_loss": probe_centroid_loss_for_weights(
                weights=weights,
                target=target,
                train_stats=train_stats,
            ),
            "residual": residual,
            "residual_norm": float(residual.norm().item()),
            "scale": float(scale),
            "weights": weights,
        })
    best = min(
        candidates,
        key=lambda item: (
            item["objective"],
            item["probe_centroid_loss"],
            item["compatible_mse"],
            -float(v16.v15.v14.functional_metrics(
                item["weights"], source, target, source_weights
            )["target_margin"]),
            item["residual_norm"],
            item["scale"],
            item["candidate_index"],
        ),
    )
    edited = best["weights"]
    metadata = {
        "base_control_type": "output_layer_no_signature_support_optimizer",
        "base_metadata": base_metadata,
        "control_type": EDITOR_METHOD,
        "decoder_config_hash": train_stats.get("decoder_config_hash", "not_fit"),
        "descriptor_mode": descriptor_mode,
        "predicted_residual_norm": float(predicted_residual.norm().item()),
        "residual_clipped": bool(clipped),
        "residual_norm_cap": float(residual_cap),
        "residual_scale": float(best["scale"]),
        "scale_0_selected": bool(best["scale"] == 0.0),
        "selected_candidate_index": int(best["candidate_index"]),
        "selected_objective": float(best["objective"]),
        "selected_residual_norm": float(best["residual_norm"]),
        "_selected_residual": best["residual"],
        "train_statistics_hash": train_stats.get("train_statistics_hash"),
    }
    return edited, metadata


def build_controls(
    *,
    subject: Mapping[str, Any],
    source: str,
    target: str,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    shuffled_signature_norm: torch.Tensor,
    shuffled_probe_descriptor: torch.Tensor | None,
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
    for control_type, descriptor_mode in [
        ("no_probe_residual_output_editor", "no_probe"),
        ("source_probe_residual_output_editor", "source_probe"),
        ("shuffled_probe_residual_output_editor", "shuffled_probe"),
        ("target_label_only_residual_output_editor", "target_label_only"),
        ("nearest_target_probe_residual_output_editor", "nearest_target_probe"),
        ("oracle_train_centroid_probe_residual_output_editor", "oracle_train_centroid_probe"),
    ]:
        weights, metadata = select_behavioral_probe_residual_output_edit(
            source_weights=source_weights,
            source_signature_norm=source_signature_norm,
            shuffled_signature_norm=shuffled_signature_norm,
            source=source,
            target=target,
            subject=subject,
            train_stats=train_stats,
            shuffled_probe_descriptor=shuffled_probe_descriptor,
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
    selected_residual = matched_metadata["_selected_residual"].detach().to(dtype=torch.float32)
    subject_hash = stable_hash_json({
        "method": EDITOR_METHOD,
        "subject_id": str(subject["subject_id"]),
    })
    for index in range(int(random_controls)):
        residual, metadata = random_norm_matched_residual(
            selected_residual=selected_residual,
            subject_hash=subject_hash,
            source=source,
            target=target,
            index=index,
            train_statistics_hash=train_stats["train_statistics_hash"],
            decoder_config_hash=str(matched_metadata.get("decoder_config_hash", "not_fit")),
        )
        theta = output_layer_theta(optimized_weights) + residual
        controls.append(control_record_from_weights(
            f"random_norm_matched_probe_residual_{index:02d}",
            replace_output_layer_theta(source_weights, theta),
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
            or str(control["control_type"]).startswith("random_norm_matched_probe_residual_")
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
    shuffled_probe_descriptor: torch.Tensor | None = None,
    train_stats: Mapping[str, Any],
    random_controls: int = RANDOM_CONTROLS_PER_RECORD,
) -> dict[str, Any]:
    matched_weights, matched_metadata = select_behavioral_probe_residual_output_edit(
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        shuffled_signature_norm=source_signature_norm if shuffled_signature_norm is None else shuffled_signature_norm,
        shuffled_probe_descriptor=shuffled_probe_descriptor,
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
        shuffled_probe_descriptor=shuffled_probe_descriptor,
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
            if control["control_type"].startswith("random_norm_matched_probe_residual_")
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
    shuffled_probe_descriptor = job.get("shuffled_probe_descriptor")
    if shuffled_probe_descriptor is not None:
        shuffled_probe_descriptor = torch.tensor(shuffled_probe_descriptor, dtype=torch.float32)
    source_weights = torch.tensor(subject["weights"], dtype=torch.float32)
    source_signature_norm = v17.normalized_signature(subject, train_stats)
    return evaluate_record(
        subject=subject,
        source=source,
        target=target,
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        shuffled_signature_norm=shuffled_signature_norm,
        shuffled_probe_descriptor=shuffled_probe_descriptor,
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
        ("v17", "min_aggregate_v17_target_margin_advantage"),
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
        require_at_least(
            failures,
            summary["mean_matched_minus_v17_target_margin"],
            THRESHOLDS["min_direction_v17_target_margin_advantage"],
            f"{direction} V17 target margin advantage",
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
) -> dict[str, Any]:
    jobs = []
    for subject in subjects:
        source = v16.subject_behavior(subject)
        for target in PATTERNS:
            if target != source:
                jobs.append({"source": source, "subject": subject, "target": target})
    assign_shuffled_signatures(jobs, train_stats)
    if parallel and len(jobs) > 1:
        contract = multiprocessing_contract(max_workers=max_workers)
        context = mp.get_context(contract["start_method"])
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(contract["max_workers"]),
            mp_context=context,
            initializer=_init_eval_worker,
            initargs=(train_stats, int(random_controls), record_evaluator),
        ) as executor:
            records = list(executor.map(_evaluate_record_worker, jobs))
    else:
        evaluator = record_evaluator or evaluate_record_from_job
        records = [
            evaluator(job, train_stats=train_stats, random_controls=random_controls)
            for job in jobs
        ]
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
            descriptor = train_stats.get("probe_descriptor_by_subject", {}).get(next_subject_id)
            if descriptor is None:
                descriptor = normalized_probe_descriptor(
                    subject=next_job["subject"],
                    weights=torch.tensor(next_job["subject"]["weights"], dtype=torch.float32),
                    train_stats=train_stats,
                )
            shuffled_descriptors[key] = tensor_to_hashable(descriptor.to(dtype=torch.float32))
    for job in jobs:
        key = (
            str(job["source"]),
            str(job["subject"]["subject_id"]),
            str(job["target"]),
        )
        job["shuffled_signature_norm"] = shuffled_signatures[key]
        job["shuffled_probe_descriptor"] = shuffled_descriptors[key]


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


def build_v21_seed_preflight() -> dict[str, Any]:
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


def assert_no_forbidden_final_raw_paths(paths: Sequence[Path | str], *, allow_v21_final: bool = False) -> None:
    prior = {path.resolve() for path in PRIOR_FINAL_RAW_PATHS}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in prior:
            raise ValueError(f"prior sealed final raw path is forbidden: {path}")
        if path.name == "final_subjects.json" and "runs" in path.parts:
            if not (allow_v21_final and path == V21_FINAL_RAW.resolve()):
                raise ValueError(f"sealed final raw path is forbidden: {path}")
        if path == V21_FINAL_RAW.resolve() and not allow_v21_final:
            raise ValueError(f"V21 final raw path is forbidden before authorization: {path}")


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
    assert_no_forbidden_final_raw_paths([train_path, eval_path], allow_v21_final=(phase == "final"))
    return failures


def generate_pools(args: argparse.Namespace, pool_dir: Path) -> dict[str, Any]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    seed_preflight = build_v21_seed_preflight()
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
        "constants_hash": stable_hash_json(constants_payload()),
        "decoder_config_hash": train_stats.get("decoder_config_hash"),
        "decoder_input_zero_variance_count": train_stats.get("decoder_input_zero_variance_count"),
        "decoder_pair_rows_hash": train_stats.get("decoder_pair_rows_hash"),
        "inner_split_hash": train_stats.get("inner_split_hash"),
        "method": EDITOR_METHOD,
        "multiprocessing_contract": multiprocessing_contract(max_workers=max_workers),
        "probe_descriptor_zero_variance_count": train_stats.get("probe_descriptor_zero_variance_count"),
        "probe_examples_hash": train_stats["probe_examples_hash"],
        "residual_norm_cap_multiplier": train_stats.get("residual_norm_cap_multiplier"),
        "residual_output_zero_variance_count": train_stats.get("residual_output_zero_variance_count"),
        "selected_decoder_config": train_stats.get("selected_decoder_config"),
        "selected_decoder_diagnostics": train_stats.get("selected_decoder_diagnostics"),
        "target_probe_logit_centroid_hashes": train_stats.get("target_probe_logit_centroid_hashes"),
        "thresholds_hash": stable_hash_json(THRESHOLDS),
        "train_statistics_hash": train_stats["train_statistics_hash"],
        "v16_baseline_train_statistics_hash": train_stats.get("v16_baseline_train_statistics_hash"),
        "v17_baseline_train_statistics_hash": train_stats.get("v17_baseline_train_statistics_hash"),
    }


def run_development(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> dict[str, Any]:
    train_path = pool_dir / "train_subjects.json"
    eval_path = pool_dir / "development_subjects.json"
    combined_audit_path = pool_dir / "combined_audit.json"
    final_redacted_path = pool_dir / "final_redacted_audit.json"
    assert_no_forbidden_final_raw_paths(
        [train_path, eval_path, combined_audit_path, final_redacted_path],
        allow_v21_final=False,
    )
    train_payload = v16.v15.v1.load_json(train_path)
    eval_payload = v16.v15.v1.load_json(eval_path)
    combined_audit = v16.v15.v1.load_json(combined_audit_path)
    final_redacted = v16.v15.v1.load_json(final_redacted_path)
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
        raise ValueError("V21 source-pool contract validation failed: " + "; ".join(contract_failures))
    train_subjects = v16.v15.v1.accepted_records(train_payload)
    eval_subjects = v16.v15.v1.accepted_records(eval_payload)
    train_stats = fit_v21_train_statistics(
        train_subjects,
        include_models=True,
        include_baseline_stats=True,
    )
    stats_path = output_dir / "v21_behavioral_probe_residual_output_editor_stats.pt"
    torch.save(serializable_stats_artifact(train_stats, args.max_workers), stats_path)
    eval_result = evaluate_subjects(
        subjects=eval_subjects,
        train_stats=train_stats,
        random_controls=RANDOM_CONTROLS_PER_RECORD,
        parallel=True,
        max_workers=args.max_workers,
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
        "implementation_sha256": v16.v15.v1.sha256_file(SCRIPT_PATH),
        "limitations": (
            "Small-subject source-label-known, target-label-requested behavioral-probe "
            "residual output-editor functional editing evidence only; not larger-model proof."
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
        "v16_baseline_train_statistics_hash": train_stats.get("v16_baseline_train_statistics_hash"),
        "v17_baseline_train_statistics_hash": train_stats.get("v17_baseline_train_statistics_hash"),
    }
    result["failures"] = failures
    result["passed"] = not failures
    result["next_action"] = (
        "eligible_for_one_shot_final_eval_without_method_changes"
        if result["passed"]
        else "log_negative_development_result_do_not_open_final_raw"
    )
    output_path = output_dir / "development_results.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    result["development_results_sha256"] = v16.v15.v1.sha256_file(output_path)
    return result


def run_final(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> dict[str, Any]:
    raise NotImplementedError(
        "V21 final is not implemented until development passes and reviewer authorizes final"
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
