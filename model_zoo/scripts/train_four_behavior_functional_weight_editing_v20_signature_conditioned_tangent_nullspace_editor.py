"""V20 functional editing via signature-conditioned tangent null-space editors."""

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
        "base_seed": 111400000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 112400000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 113400000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}
DEFAULT_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v20_pools"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runs"
    / "four_behavior_functional_weight_editing_v20_signature_conditioned_tangent_nullspace_editor"
)
PREREG_PATH = (
    REPO_ROOT
    / "docs"
    / "preregistrations"
    / "four_behavior_functional_weight_editing_v20_signature_conditioned_tangent_nullspace_editor.md"
)
SCRIPT_PATH = Path(__file__).resolve()
HELPER_TEST_PATH = (
    REPO_ROOT
    / "model_zoo"
    / "scripts"
    / "test_four_behavior_functional_weight_editing_v20_helpers.py"
)
SOURCE_POOL_SCOPE = "four_behavior_functional_weight_editing_v20_source_pool"
SOURCE_AUDIT_SCOPE = "four_behavior_functional_weight_editing_v20_source_pool_construction"
FINAL_REDACTED_SCOPE = (
    "redacted_final_functional_weight_editing_v20_source_pool_audit_surface_only"
)
DEVELOPMENT_SCOPE = (
    "four_behavior_functional_weight_editing_v20_signature_conditioned_tangent_nullspace_editor_development"
)
FINAL_SCOPE = (
    "four_behavior_functional_weight_editing_v20_signature_conditioned_tangent_nullspace_editor_final"
)
EDITOR_METHOD = "signature_conditioned_tangent_nullspace_editor_v20"
V20_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"
PRIOR_FINAL_RAW_PATHS = {
    v17.V17_FINAL_RAW,
    v17.v16.V16_FINAL_RAW,
    v17.v16.v15.V15_FINAL_RAW,
}

SOURCE_WEIGHT_DIM = 345
SIGNATURE_DIM = 560
DIRECTION_DIM = len(PATTERNS) * (len(PATTERNS) - 1)
COMPONENT_STAT_DIM = len(v17.LAYER_COMPONENT_SPECS) * 5
EDITOR_INPUT_DIM = (
    SIGNATURE_DIM
    + len(PATTERNS)
    + len(PATTERNS)
    + DIRECTION_DIM
    + SIGNATURE_DIM
    + SIGNATURE_DIM
    + SIGNATURE_DIM
    + COMPONENT_STAT_DIM
)
SVD_RANK = 8
SIGNATURE_TOP_K = 8
SIGNATURE_TEMPERATURE = 1.0
DESIRED_LOGIT_MAGNITUDE = 2.0
NULLSPACE_RELATIVE_CUTOFF = 1e-4
NULLSPACE_ABSOLUTE_CUTOFF = 1e-6
MASK_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
RIDGE_LAMBDAS = [0.01, 0.1, 1.0, 10.0]
PRIOR_LAMBDAS = [0.0, 0.01, 0.1, 1.0]
ACTIVATION_SCALE_GRID = [0.0, 0.5, 1.0]
POST_SCALE_GRID = [0.5, 0.75, 1.0, 1.25]
MAX_TANGENT_DELTA_NORM = 8.0
RANDOM_CONTROLS_PER_RECORD = 16
EXPECTED_CONTROLS_PER_RECORD = 26
RANDOM_CONTROL_EPS = 1e-12
PARETO_EPSILON = 1e-9

LOSS_WEIGHTS = {
    "compatible_source_mse": 1.0,
    "conflict_bce": 1.0,
    "target_bce": 1.0,
}
THRESHOLDS = {
    "expected_record_count": 288,
    "expected_controls_per_record": 26,
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
    "target_label_tangent_nullspace",
    "no_signature_zero_tangent_nullspace",
    "source_signature_tangent_nullspace",
    "shuffled_signature_tangent_nullspace",
    "no_nullspace_signature_tangent",
    "v17_layerwise_rank1_tsv",
    "v16_output_layer_conceptor",
    "output_layer_no_signature_support_optimizer",
    "nearest_target_layerwise_tsv",
]
ADVANTAGE_CONTROL_TYPES = {
    "no_signature": "no_signature_zero_tangent_nullspace",
    "target_label": "target_label_tangent_nullspace",
    "source_signature": "source_signature_tangent_nullspace",
    "shuffled_signature": "shuffled_signature_tangent_nullspace",
    "no_nullspace": "no_nullspace_signature_tangent",
    "v17": "v17_layerwise_rank1_tsv",
    "v16": "v16_output_layer_conceptor",
    "output_layer_no_signature": "output_layer_no_signature_support_optimizer",
    "nearest_target": "nearest_target_layerwise_tsv",
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
        "activation_scale_grid": ACTIVATION_SCALE_GRID,
        "desired_logit_magnitude": DESIRED_LOGIT_MAGNITUDE,
        "expected_controls_per_record": EXPECTED_CONTROLS_PER_RECORD,
        "loss_weights": LOSS_WEIGHTS,
        "mask_fractions": MASK_FRACTIONS,
        "max_tangent_delta_norm": MAX_TANGENT_DELTA_NORM,
        "nullspace_absolute_cutoff": NULLSPACE_ABSOLUTE_CUTOFF,
        "nullspace_relative_cutoff": NULLSPACE_RELATIVE_CUTOFF,
        "post_scale_grid": POST_SCALE_GRID,
        "prior_lambdas": PRIOR_LAMBDAS,
        "random_controls_per_record": RANDOM_CONTROLS_PER_RECORD,
        "ridge_lambdas": RIDGE_LAMBDAS,
        "signature_temperature": SIGNATURE_TEMPERATURE,
        "signature_top_k": SIGNATURE_TOP_K,
        "thresholds": THRESHOLDS,
    }


def direction_index(source: str, target: str) -> int:
    directions = [
        (src, dst)
        for src in PATTERNS
        for dst in PATTERNS
        if src != dst
    ]
    return directions.index((source, target))


def one_hot(index: int, size: int) -> torch.Tensor:
    value = torch.zeros(size, dtype=torch.float32)
    value[int(index)] = 1.0
    return value


def behavior_one_hot(behavior: str) -> torch.Tensor:
    return one_hot(PATTERNS.index(behavior), len(PATTERNS))


def direction_one_hot(source: str, target: str) -> torch.Tensor:
    return one_hot(direction_index(source, target), DIRECTION_DIM)


def component_stats(weights: torch.Tensor) -> torch.Tensor:
    values = []
    for spec in v17.LAYER_COMPONENT_SPECS:
        component = v17.component_from_flat(weights, spec).reshape(-1).to(dtype=torch.float32)
        values.extend([
            component.mean(),
            component.std(unbiased=False),
            component.norm(),
            component.min(),
            component.max(),
        ])
    return torch.stack(values).to(dtype=torch.float32)


def behavior_centroids(records: Sequence[Mapping[str, Any]], train_stats: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    grouped = v17.records_by_behavior(records)
    centroids = {}
    for behavior, items in grouped.items():
        signatures = [v17.normalized_signature(item, train_stats) for item in items]
        centroids[behavior] = torch.stack(signatures).mean(dim=0)
    return centroids


def build_editor_features(
    *,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    ablation: str = "matched",
) -> torch.Tensor:
    centroids = train_stats["signature_centroids"]
    source_centroid = centroids[source].to(dtype=torch.float32)
    target_centroid = centroids[target].to(dtype=torch.float32)
    source_sig = source_signature_norm.to(dtype=torch.float32)
    source_one = behavior_one_hot(source)
    target_one = behavior_one_hot(target)
    direction_one = direction_one_hot(source, target)

    if ablation == "target_label":
        source_sig = torch.zeros_like(source_sig)
        source_residual = torch.zeros_like(source_sig)
        target_minus_source = torch.zeros_like(source_sig)
    elif ablation == "source_signature":
        source_residual = source_sig - source_centroid
        target_one = behavior_one_hot(source)
        direction_one = torch.zeros(DIRECTION_DIM, dtype=torch.float32)
        target_centroid = source_centroid
        target_minus_source = torch.zeros_like(source_sig)
    else:
        source_residual = source_sig - source_centroid
        target_minus_source = target_centroid - source_sig
    return torch.cat([
        source_sig,
        source_one,
        target_one,
        direction_one,
        target_centroid,
        source_residual,
        target_minus_source,
        component_stats(source_weights),
    ]).to(dtype=torch.float32)


def component_ranks(direction_bases: Mapping[str, Any]) -> dict[str, int]:
    return {
        spec["name"]: int(direction_bases["components"][spec["name"]]["basis"].shape[0])
        for spec in v17.LAYER_COMPONENT_SPECS
    }


def coefficient_slices(direction_bases: Mapping[str, Any]) -> dict[str, slice]:
    offset = 0
    slices = {}
    ranks = component_ranks(direction_bases)
    for spec in v17.LAYER_COMPONENT_SPECS:
        rank = ranks[spec["name"]]
        slices[spec["name"]] = slice(offset, offset + rank)
        offset += rank
    return slices


def coefficient_dim_for_bases(direction_bases: Mapping[str, Any]) -> int:
    return sum(component_ranks(direction_bases).values())


def coefficients_for_delta(delta: torch.Tensor, direction_bases: Mapping[str, Any]) -> torch.Tensor:
    values = []
    for spec in v17.LAYER_COMPONENT_SPECS:
        info = direction_bases["components"][spec["name"]]
        component = v17.component_from_flat(delta, spec).reshape(-1).to(dtype=torch.float32)
        mean_delta = info["mean_delta"].to(dtype=torch.float32)
        basis = info["basis"].to(dtype=torch.float32)
        values.append((component - mean_delta) @ basis.t())
    return torch.cat(values).to(dtype=torch.float32)


def decode_coefficients(
    *,
    coefficients: torch.Tensor,
    component_gates: torch.Tensor,
    global_scale: torch.Tensor,
    source_weights: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    activation_scale: torch.Tensor | None = None,
    activation_delta: torch.Tensor | None = None,
    zero_means: bool = False,
) -> torch.Tensor:
    direction_bases = train_stats["layerwise_bases"][v17.direction_key(source, target)]
    slices = coefficient_slices(direction_bases)
    delta = torch.zeros_like(source_weights, dtype=torch.float32)
    for component_index, spec in enumerate(v17.LAYER_COMPONENT_SPECS):
        info = direction_bases["components"][spec["name"]]
        basis = info["basis"].to(dtype=torch.float32)
        coeff = coefficients[slices[spec["name"]]].to(dtype=torch.float32)
        gate = component_gates[component_index].to(dtype=torch.float32)
        mean_delta = torch.zeros_like(info["mean_delta"]) if zero_means else info["mean_delta"]
        component = mean_delta.to(dtype=torch.float32) + gate * (coeff @ basis)
        v17.set_component(delta, spec, component.reshape(tuple(spec["shape"])))
    if activation_delta is not None and activation_scale is not None:
        delta = delta + activation_scale.to(dtype=torch.float32) * activation_delta.to(dtype=torch.float32)
    return source_weights + global_scale.to(dtype=torch.float32) * delta


def full_train_statistics_hash_payload(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "component_order": [spec["name"] for spec in v17.LAYER_COMPONENT_SPECS],
        "constants": constants_payload(),
        "direction_hashes": {
            direction: {
                component_name: {
                    "basis": tensor_to_hashable(component["basis"]),
                    "mean_delta": tensor_to_hashable(component["mean_delta"]),
                    "rank": int(component["rank"]),
                    "singular_values": tensor_to_hashable(component["singular_values"]),
                }
                for component_name, component in sorted(bases["components"].items())
            }
            for direction, bases in sorted(stats.get("layerwise_bases", {}).items())
        },
        "direction_pair_hashes": stats.get("direction_pair_hashes", {}),
        "probe_examples_hash": stats.get("probe_examples_hash", "missing"),
        "sig_mean": tensor_to_hashable(stats["sig_mean"]),
        "sig_std": tensor_to_hashable(stats["sig_std"]),
        "signature_centroids": {
            key: tensor_to_hashable(value)
            for key, value in sorted(stats.get("signature_centroids", {}).items())
        },
        "target_centroid_coefficients": {
            key: tensor_to_hashable(value)
            for key, value in sorted(stats.get("target_centroid_coefficients", {}).items())
        },
        "target_centroid_deltas": {
            key: tensor_to_hashable(value)
            for key, value in sorted(stats.get("target_centroid_deltas", {}).items())
        },
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
                "scope": "four_behavior_functional_weight_editing_v20_inner_split",
                "subject_id": subject_id,
            })
            scored.append((split_hash, subject_id, record))
        scored.sort(key=lambda item: (item[0], item[1]))
        train_part = [item[2] for item in scored[:48]]
        validation_part = [item[2] for item in scored[48:64]]
        if len(train_part) != 48 or len(validation_part) != 16:
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
        "scope": "four_behavior_functional_weight_editing_v20_inner_split",
    }
    return {
        "inner_train": sorted_train_subjects(inner_train),
        "inner_validation": sorted_train_subjects(inner_validation),
        "split_hash": stable_hash_json(split_payload),
        "split_payload": split_payload,
    }


def fit_layerwise_bases(subjects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = v17.records_by_behavior(subjects)
    bases = {}
    for source in PATTERNS:
        for target in PATTERNS:
            if source == target:
                continue
            bases[v17.direction_key(source, target)] = v17.layerwise_component_bases_for_direction(
                source_records=grouped[source],
                target_records=grouped[target],
                source=source,
                target=target,
                rank=SVD_RANK,
            )
    return bases


def build_base_train_pair_dataset(
    subjects: Sequence[Mapping[str, Any]],
    *,
    train_stats: Mapping[str, Any],
) -> dict[str, Any]:
    grouped = v17.records_by_behavior(subjects)
    rows = []
    coefficients = []
    source_signatures = []
    source_weights_values = []
    target_deltas = []
    for source in PATTERNS:
        for target in PATTERNS:
            if source == target:
                continue
            direction_bases = train_stats["layerwise_bases"][v17.direction_key(source, target)]
            for source_record in sorted(grouped[source], key=lambda item: str(item["subject_id"])):
                source_weights = torch.tensor(source_record["weights"], dtype=torch.float32)
                source_signature_norm = v17.normalized_signature(source_record, train_stats)
                for target_record in sorted(grouped[target], key=lambda item: str(item["subject_id"])):
                    delta = v16.v15.v14.target_delta_for_record(
                        source_weights=source_weights,
                        target_record=target_record,
                        source=source,
                        target=target,
                        subject_id=str(source_record["subject_id"]),
                        alignment_mode="hungarian",
                    ).to(dtype=torch.float32)
                    coefficients.append(coefficients_for_delta(delta, direction_bases))
                    source_signatures.append(source_signature_norm)
                    source_weights_values.append(source_weights)
                    target_deltas.append(delta)
                    rows.append({
                        "source": source,
                        "source_subject_id": str(source_record["subject_id"]),
                        "target": target,
                        "target_subject_id": str(target_record["subject_id"]),
                    })
    return {
        "coefficients": torch.stack(coefficients).to(dtype=torch.float32),
        "rows": rows,
        "rows_hash": stable_hash_json(rows),
        "source_signatures": torch.stack(source_signatures).to(dtype=torch.float32),
        "source_weights": torch.stack(source_weights_values).to(dtype=torch.float32),
        "target_deltas": torch.stack(target_deltas).to(dtype=torch.float32),
    }


def differentiable_support_loss(
    *,
    weights: torch.Tensor,
    source_weights: torch.Tensor,
    source: str,
    target: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    support = v16.v15.v14.prepare_support_tensors_with_source_logits(
        source_weights=source_weights,
        source=source,
        target=target,
    )
    target_logits = v16.v15.v10.decoder_v1.subject_forward_flat_batch(
        weights.unsqueeze(0),
        support["target_inputs"],
    )[0]
    compatible_logits = v16.v15.v10.decoder_v1.subject_forward_flat_batch(
        weights.unsqueeze(0),
        support["compatible_inputs"],
    )[0]
    conflict_logits = v16.v15.v10.decoder_v1.subject_forward_flat_batch(
        weights.unsqueeze(0),
        support["conflict_inputs"],
    )[0]
    target_bce = F.binary_cross_entropy_with_logits(target_logits, support["target_labels"])
    conflict_bce = F.binary_cross_entropy_with_logits(
        conflict_logits,
        support["conflict_target_labels"],
    )
    compatible_mse = F.mse_loss(compatible_logits, support["compatible_source_logits"])
    return (
        LOSS_WEIGHTS["target_bce"] * target_bce
        + LOSS_WEIGHTS["conflict_bce"] * conflict_bce
        + LOSS_WEIGHTS["compatible_source_mse"] * compatible_mse,
        {
            "compatible_source_mse": compatible_mse,
            "conflict_bce": conflict_bce,
            "target_bce": target_bce,
        },
    )


def fit_v20_train_statistics(
    subjects: Sequence[Mapping[str, Any]],
    *,
    include_models: bool = True,
    include_baseline_stats: bool = True,
) -> dict[str, Any]:
    ordered_subjects = sorted_train_subjects(subjects)
    signatures = torch.tensor([record["signature"] for record in ordered_subjects], dtype=torch.float32)
    sig_mean = signatures.mean(dim=0)
    sig_std = signatures.std(dim=0, unbiased=False).clamp_min(1e-6)
    probe_examples = build_probe_examples()
    stats: dict[str, Any] = {
        "layerwise_bases": fit_layerwise_bases(ordered_subjects),
        "probe_examples": probe_examples,
        "probe_examples_hash": stable_hash_json(probe_examples),
        "sig_mean": sig_mean,
        "sig_std": sig_std,
        "train_by_behavior": v17.records_by_behavior(ordered_subjects),
        "train_subjects": ordered_subjects,
    }
    stats["signature_centroids"] = behavior_centroids(ordered_subjects, stats)
    full_pair_dataset = build_base_train_pair_dataset(ordered_subjects, train_stats=stats)
    stats["coefficient_dim"] = int(full_pair_dataset["coefficients"].shape[1])
    stats["direction_pair_hashes"] = {}
    stats["target_centroid_coefficients"] = {}
    stats["target_centroid_deltas"] = {}
    stats["target_centroid_delta_hashes"] = {}
    for source in PATTERNS:
        for target in PATTERNS:
            if source == target:
                continue
            direction = v17.direction_key(source, target)
            indices = [
                index for index, row in enumerate(full_pair_dataset["rows"])
                if row["source"] == source and row["target"] == target
            ]
            rows = [full_pair_dataset["rows"][index] for index in indices]
            stats["direction_pair_hashes"][direction] = stable_hash_json(rows)
            deltas = full_pair_dataset["target_deltas"][indices]
            centroid_delta = deltas.mean(dim=0)
            stats["target_centroid_deltas"][direction] = centroid_delta.detach().clone()
            stats["target_centroid_delta_hashes"][direction] = stable_hash_json(
                tensor_to_hashable(centroid_delta)
            )
            stats["target_centroid_coefficients"][direction] = coefficients_for_delta(
                centroid_delta,
                stats["layerwise_bases"][direction],
            )
    stats["train_pair_rows_hash"] = full_pair_dataset["rows_hash"]
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
    if control_type in {EDITOR_METHOD, "no_nullspace_signature_tangent"}:
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
    return v17.control_record_from_weights(
        control_type,
        weights,
        source,
        target,
        source_weights,
        metadata,
    )


def build_controls(
    *,
    subject: Mapping[str, Any],
    source: str,
    target: str,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    shuffled_signature_norm: torch.Tensor,
    matched_weights: torch.Tensor,
    matched_metadata: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    random_controls: int = RANDOM_CONTROLS_PER_RECORD,
) -> list[dict[str, Any]]:
    controls = [control_record_from_weights("no_edit", source_weights, source, target, source_weights)]
    for control_type in [
        "target_label_tangent_nullspace",
        "no_signature_zero_tangent_nullspace",
        "shuffled_signature_tangent_nullspace",
        "source_signature_tangent_nullspace",
        "no_nullspace_signature_tangent",
    ]:
        weights, metadata = select_tangent_nullspace_edit(
            source_weights=source_weights,
            source_signature_norm=source_signature_norm,
            shuffled_signature_norm=shuffled_signature_norm,
            source=source,
            target=target,
            subject_id=str(subject["subject_id"]),
            train_stats=train_stats,
            control_type=control_type,
        )
        controls.append(control_record_from_weights(
            control_type,
            weights,
            source,
            target,
            source_weights,
            {key: value for key, value in metadata.items() if not key.startswith("_")},
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
    nearest_weights, nearest_meta = v17.select_layerwise_rank1_tsv_edit(
        source_weights=source_weights,
        subject_id=str(subject["subject_id"]),
        source=source,
        target=target,
        source_signature_norm=source_signature_norm,
        train_stats=v17_stats,
        signature_top_k=1,
    )
    controls.append(control_record_from_weights(
        "nearest_target_layerwise_tsv",
        nearest_weights,
        source,
        target,
        source_weights,
        nearest_meta,
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
    matched_delta = matched_weights - source_weights
    random_basis = matched_metadata["_selected_basis"]
    public_candidate_metadata = {
        key: matched_metadata[key]
        for key in ("activation_scale", "mask_fraction", "post_scale", "prior_lambda", "ridge_lambda")
    }
    for index in range(int(random_controls)):
        delta, metadata = random_tangent_delta(
            basis=random_basis,
            matched_delta=matched_delta,
            subject_id=str(subject["subject_id"]),
            source=source,
            target=target,
            index=index,
            train_statistics_hash=train_stats["train_statistics_hash"],
            selected_candidate_metadata=public_candidate_metadata,
        )
        controls.append(control_record_from_weights(
            f"random_norm_matched_tangent_delta_{index:02d}",
            source_weights + delta,
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
            or str(control["control_type"]).startswith("random_norm_matched_tangent_delta_")
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
    train_stats: Mapping[str, Any],
    random_controls: int = RANDOM_CONTROLS_PER_RECORD,
) -> dict[str, Any]:
    matched_weights, matched_metadata = select_tangent_nullspace_edit(
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        shuffled_signature_norm=source_signature_norm if shuffled_signature_norm is None else shuffled_signature_norm,
        source=source,
        target=target,
        subject_id=str(subject["subject_id"]),
        train_stats=train_stats,
        control_type=EDITOR_METHOD,
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
            if control["control_type"].startswith("random_norm_matched_tangent_delta_")
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
    source_weights = torch.tensor(subject["weights"], dtype=torch.float32)
    source_signature_norm = v17.normalized_signature(subject, train_stats)
    return evaluate_record(
        subject=subject,
        source=source,
        target=target,
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        shuffled_signature_norm=shuffled_signature_norm,
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
        "target_prediction_count": sum(1 for item in matched if item["target_prediction_pass"]),
    }
    summary["individual_all_gate_pass_rate"] = summary["individual_all_gate_pass_count"] / len(records)
    summary["pareto_undominated_rate"] = summary["pareto_undominated_count"] / len(records)
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
    for job in jobs:
        key = (
            str(job["source"]),
            str(job["subject"]["subject_id"]),
            str(job["target"]),
        )
        job["shuffled_signature_norm"] = shuffled_signatures[key]


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


def build_v20_seed_preflight() -> dict[str, Any]:
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


def assert_no_forbidden_final_raw_paths(paths: Sequence[Path | str], *, allow_v20_final: bool = False) -> None:
    prior = {path.resolve() for path in PRIOR_FINAL_RAW_PATHS}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in prior:
            raise ValueError(f"prior sealed final raw path is forbidden: {path}")
        if path.name == "final_subjects.json" and "runs" in path.parts:
            if not (allow_v20_final and path == V20_FINAL_RAW.resolve()):
                raise ValueError(f"sealed final raw path is forbidden: {path}")
        if path == V20_FINAL_RAW.resolve() and not allow_v20_final:
            raise ValueError(f"V20 final raw path is forbidden before authorization: {path}")


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
    assert_no_forbidden_final_raw_paths([train_path, eval_path], allow_v20_final=(phase == "final"))
    return failures


def generate_pools(args: argparse.Namespace, pool_dir: Path) -> dict[str, Any]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    seed_preflight = build_v20_seed_preflight()
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
        "direction_pair_hashes": train_stats.get("direction_pair_hashes"),
        "method": EDITOR_METHOD,
        "multiprocessing_contract": multiprocessing_contract(max_workers=max_workers),
        "probe_examples_hash": train_stats["probe_examples_hash"],
        "target_centroid_delta_hashes": train_stats.get("target_centroid_delta_hashes"),
        "thresholds_hash": stable_hash_json(THRESHOLDS),
        "train_pair_rows_hash": train_stats["train_pair_rows_hash"],
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
        allow_v20_final=False,
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
        raise ValueError("V20 source-pool contract validation failed: " + "; ".join(contract_failures))
    train_subjects = v16.v15.v1.accepted_records(train_payload)
    eval_subjects = v16.v15.v1.accepted_records(eval_payload)
    train_stats = fit_v20_train_statistics(
        train_subjects,
        include_models=True,
        include_baseline_stats=True,
    )
    stats_path = output_dir / "v20_signature_conditioned_tangent_nullspace_editor_stats.pt"
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
            "Small-subject source-label-known, target-label-requested low-rank "
            "subspace optimizer functional editing evidence only; not larger-model proof."
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
        "V20 final is not implemented until development passes and reviewer authorizes final"
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
