"""V19 functional editing via signature-initialized subspace optimizers."""

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
        "base_seed": 101400000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 102400000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 103400000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}
DEFAULT_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v19_pools"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runs"
    / "four_behavior_functional_weight_editing_v19_signature_initialized_subspace_optimizer"
)
PREREG_PATH = (
    REPO_ROOT
    / "docs"
    / "preregistrations"
    / "four_behavior_functional_weight_editing_v19_signature_initialized_subspace_optimizer.md"
)
SCRIPT_PATH = Path(__file__).resolve()
HELPER_TEST_PATH = (
    REPO_ROOT
    / "model_zoo"
    / "scripts"
    / "test_four_behavior_functional_weight_editing_v19_helpers.py"
)
SOURCE_POOL_SCOPE = "four_behavior_functional_weight_editing_v19_source_pool"
SOURCE_AUDIT_SCOPE = "four_behavior_functional_weight_editing_v19_source_pool_construction"
FINAL_REDACTED_SCOPE = (
    "redacted_final_functional_weight_editing_v19_source_pool_audit_surface_only"
)
DEVELOPMENT_SCOPE = (
    "four_behavior_functional_weight_editing_v19_signature_initialized_subspace_optimizer_development"
)
FINAL_SCOPE = (
    "four_behavior_functional_weight_editing_v19_signature_initialized_subspace_optimizer_final"
)
EDITOR_METHOD = "signature_initialized_subspace_support_optimizer_v19"
V19_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"
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
OPTIMIZER_STEPS = 80
OPTIMIZER_LR = 0.08
OPTIMIZER_BETAS = (0.9, 0.999)
OPTIMIZER_EPS = 1e-8
OPTIMIZER_WEIGHT_DECAY = 0.0
OPTIMIZER_AMSGRAD = False
GRAD_CLIP_NORM = 5.0
COEFFICIENT_L2_WEIGHT = 0.01
SOURCE_WEIGHT_L2_WEIGHT = 0.0005
DELTA_L2_WEIGHT = 0.0005
GATE_L1_WEIGHT = 0.002
GLOBAL_SCALE_MAX = 1.5
SIGNATURE_TOP_K = 8
SIGNATURE_TEMPERATURE = 1.0
POST_SCALE_GRID = [0.5, 0.75, 1.0, 1.25]
RANDOM_CONTROLS_PER_RECORD = 16
RANDOM_CONTROL_EPS = 1e-12
PARETO_EPSILON = 1e-9

LOSS_WEIGHTS = {
    "compatible_source_mse": 1.0,
    "conflict_bce": 1.0,
    "target_bce": 1.0,
}
THRESHOLDS = {
    "expected_record_count": 288,
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
    "target_label_subspace_optimizer",
    "no_signature_zero_subspace_optimizer",
    "source_signature_subspace_optimizer",
    "shuffled_signature_subspace_optimizer",
    "v17_layerwise_rank1_tsv",
    "v16_output_layer_conceptor",
    "output_layer_no_signature_support_optimizer",
    "nearest_target_layerwise_tsv",
]
ADVANTAGE_CONTROL_TYPES = {
    "no_signature": "no_signature_zero_subspace_optimizer",
    "target_label": "target_label_subspace_optimizer",
    "source_signature": "source_signature_subspace_optimizer",
    "shuffled_signature": "shuffled_signature_subspace_optimizer",
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
        "coefficient_l2_weight": COEFFICIENT_L2_WEIGHT,
        "delta_l2_weight": DELTA_L2_WEIGHT,
        "gate_l1_weight": GATE_L1_WEIGHT,
        "global_scale_max": GLOBAL_SCALE_MAX,
        "loss_weights": LOSS_WEIGHTS,
        "optimizer_amsgrad": OPTIMIZER_AMSGRAD,
        "optimizer_betas": OPTIMIZER_BETAS,
        "optimizer_eps": OPTIMIZER_EPS,
        "optimizer_lr": OPTIMIZER_LR,
        "optimizer_steps": OPTIMIZER_STEPS,
        "optimizer_weight_decay": OPTIMIZER_WEIGHT_DECAY,
        "post_scale_grid": POST_SCALE_GRID,
        "random_controls_per_record": RANDOM_CONTROLS_PER_RECORD,
        "signature_temperature": SIGNATURE_TEMPERATURE,
        "signature_top_k": SIGNATURE_TOP_K,
        "source_weight_l2_weight": SOURCE_WEIGHT_L2_WEIGHT,
        "svd_rank": SVD_RANK,
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
            for direction, bases in sorted(stats["layerwise_bases"].items())
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
                "scope": "four_behavior_functional_weight_editing_v19_inner_split",
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
        "scope": "four_behavior_functional_weight_editing_v19_inner_split",
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


def fit_v19_train_statistics(
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


def signature_initialization(
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
    direction_bases = train_stats["layerwise_bases"][v17.direction_key(source, target)]
    coefficients = coefficients_for_delta(weighted_delta, direction_bases)
    metadata = {
        "activation_residual_hash": stable_hash_json(tensor_to_hashable(activation_delta)),
        "basis_hash": stable_hash_json({
            name: tensor_to_hashable(info["basis"])
            for name, info in sorted(direction_bases["components"].items())
        }),
        "coefficient_hash": stable_hash_json(tensor_to_hashable(coefficients)),
        "selected_signature_targets": topk["metadata"],
        "selected_signature_targets_hash": stable_hash_json(topk["metadata"]),
        "signature_pool_behavior": pool_behavior,
        "weighted_delta_hash": stable_hash_json(tensor_to_hashable(weighted_delta)),
    }
    return {
        "activation_delta": activation_delta.to(dtype=torch.float32),
        "coefficients": coefficients.to(dtype=torch.float32),
        "metadata": metadata,
        "weighted_delta": weighted_delta.to(dtype=torch.float32),
    }


def control_initialization_bundle(
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
    direction = v17.direction_key(source, target)
    coeff_dim = coefficient_dim_for_bases(train_stats["layerwise_bases"][direction])
    zeros = torch.zeros(coeff_dim, dtype=torch.float32)
    zero_delta = torch.zeros_like(source_weights, dtype=torch.float32)
    if control_type == EDITOR_METHOD:
        bundle = signature_initialization(
            source_weights=source_weights,
            source_signature_norm=source_signature_norm,
            subject_id=subject_id,
            source=source,
            target=target,
            train_stats=train_stats,
        )
        bundle["metadata"] = {**bundle["metadata"], "initialization_source": "matched_signature"}
        bundle["anchor"] = bundle["coefficients"].detach().clone()
        return bundle
    if control_type == "target_label_subspace_optimizer":
        coeff = train_stats.get("target_centroid_coefficients", {}).get(direction, zeros).detach().clone()
        return {
            "activation_delta": zero_delta,
            "anchor": coeff.detach().clone(),
            "coefficients": coeff,
            "metadata": {
                "initialization_source": "target_behavior_centroid",
                "signature_pool_behavior": target,
                "target_centroid_delta_hash": train_stats.get("target_centroid_delta_hashes", {}).get(direction),
            },
            "weighted_delta": zero_delta,
        }
    if control_type == "no_signature_zero_subspace_optimizer":
        return {
            "activation_delta": zero_delta,
            "anchor": zeros.detach().clone(),
            "coefficients": zeros,
            "metadata": {
                "initialization_source": "zero",
                "signature_pool_behavior": None,
            },
            "weighted_delta": zero_delta,
        }
    if control_type == "shuffled_signature_subspace_optimizer":
        bundle = signature_initialization(
            source_weights=source_weights,
            source_signature_norm=shuffled_signature_norm,
            subject_id=subject_id,
            source=source,
            target=target,
            train_stats=train_stats,
        )
        bundle["metadata"] = {**bundle["metadata"], "initialization_source": "shuffled_signature"}
        bundle["anchor"] = bundle["coefficients"].detach().clone()
        return bundle
    if control_type == "source_signature_subspace_optimizer":
        bundle = signature_initialization(
            source_weights=source_weights,
            source_signature_norm=source_signature_norm,
            subject_id=subject_id,
            source=source,
            target=target,
            train_stats=train_stats,
            signature_pool_behavior=source,
        )
        bundle["metadata"] = {**bundle["metadata"], "initialization_source": "source_behavior_signature"}
        bundle["anchor"] = bundle["coefficients"].detach().clone()
        return bundle
    raise ValueError(f"unknown V19 initialization control type: {control_type}")


def select_post_scale_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(
        candidates,
        key=lambda item: (
            float(item.get("support_objective", item["objective"])),
            float(item["delta_norm"]),
            float(item["post_scale"]),
            int(item["candidate_index"]),
        ),
    )


def _logit_scalar(value: float) -> torch.Tensor:
    tensor = torch.tensor(float(value), dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(tensor)


def optimize_subspace_coefficients(
    *,
    source_weights: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    c_init = initialization["coefficients"].detach().clone().to(dtype=torch.float32)
    c_anchor = initialization.get("anchor", c_init).detach().clone().to(dtype=torch.float32)
    activation_delta = initialization["activation_delta"].detach().clone().to(dtype=torch.float32)
    c_raw = c_init.detach().clone().requires_grad_(True)
    g_raw = torch.full(
        (len(v17.LAYER_COMPONENT_SPECS),),
        2.0,
        dtype=torch.float32,
        requires_grad=True,
    )
    a_raw = _logit_scalar(0.5).requires_grad_(True)
    s_raw = _logit_scalar(1.0 / GLOBAL_SCALE_MAX).requires_grad_(True)
    optimizer = torch.optim.Adam(
        [c_raw, g_raw, a_raw, s_raw],
        lr=OPTIMIZER_LR,
        betas=OPTIMIZER_BETAS,
        eps=OPTIMIZER_EPS,
        weight_decay=OPTIMIZER_WEIGHT_DECAY,
        amsgrad=OPTIMIZER_AMSGRAD,
    )
    last_terms: dict[str, float] = {}
    for _step in range(OPTIMIZER_STEPS):
        gates = torch.sigmoid(g_raw)
        activation_scale = torch.sigmoid(a_raw)
        global_scale = GLOBAL_SCALE_MAX * torch.sigmoid(s_raw)
        weights = decode_coefficients(
            coefficients=c_raw,
            component_gates=gates,
            global_scale=global_scale,
            source_weights=source_weights,
            source=source,
            target=target,
            train_stats=train_stats,
            activation_scale=activation_scale,
            activation_delta=activation_delta,
        )
        decoded_delta = weights - source_weights
        support_loss, support_terms = differentiable_support_loss(
            weights=weights,
            source_weights=source_weights,
            source=source,
            target=target,
        )
        source_weight_l2 = torch.mean((weights - source_weights) ** 2)
        coefficient_l2 = torch.mean((c_raw - c_anchor) ** 2)
        decoded_delta_l2 = torch.mean(decoded_delta ** 2)
        gate_l1 = torch.mean(torch.sigmoid(g_raw))
        loss = (
            support_loss
            + SOURCE_WEIGHT_L2_WEIGHT * source_weight_l2
            + COEFFICIENT_L2_WEIGHT * coefficient_l2
            + DELTA_L2_WEIGHT * decoded_delta_l2
            + GATE_L1_WEIGHT * gate_l1
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([c_raw, g_raw, a_raw, s_raw], GRAD_CLIP_NORM)
        optimizer.step()
        last_terms = {
            "coefficient_l2": float(coefficient_l2.detach().item()),
            "compatible_source_mse": float(support_terms["compatible_source_mse"].detach().item()),
            "conflict_bce": float(support_terms["conflict_bce"].detach().item()),
            "decoded_delta_l2": float(decoded_delta_l2.detach().item()),
            "gate_l1": float(gate_l1.detach().item()),
            "loss": float(loss.detach().item()),
            "source_weight_l2": float(source_weight_l2.detach().item()),
            "target_bce": float(support_terms["target_bce"].detach().item()),
        }
    with torch.no_grad():
        final_c = c_raw.detach().clone()
        final_gates = torch.sigmoid(g_raw.detach())
        final_activation_scale = torch.sigmoid(a_raw.detach())
        final_global_scale = GLOBAL_SCALE_MAX * torch.sigmoid(s_raw.detach())
    return {
        "activation_delta": activation_delta,
        "activation_scale": final_activation_scale,
        "coefficients": final_c,
        "component_gates": final_gates,
        "global_scale": final_global_scale,
        "last_terms": last_terms,
    }


def select_subspace_optimizer_edit(
    *,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    source: str,
    target: str,
    subject_id: str,
    train_stats: Mapping[str, Any],
    shuffled_signature_norm: torch.Tensor | None = None,
    control_type: str = EDITOR_METHOD,
    model: Any | None = None,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
    ablation: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    del model, feature_mean, feature_std, ablation
    init = control_initialization_bundle(
        control_type=control_type,
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        shuffled_signature_norm=source_signature_norm if shuffled_signature_norm is None else shuffled_signature_norm,
        subject_id=subject_id,
        source=source,
        target=target,
        train_stats=train_stats,
    )
    optimized = optimize_subspace_coefficients(
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats,
        initialization=init,
    )
    candidates = []
    for candidate_index, post_scale in enumerate(POST_SCALE_GRID):
        weights = decode_coefficients(
            coefficients=optimized["coefficients"],
            component_gates=optimized["component_gates"],
            global_scale=optimized["global_scale"] * float(post_scale),
            source_weights=source_weights,
            source=source,
            target=target,
            train_stats=train_stats,
            activation_scale=optimized["activation_scale"],
            activation_delta=optimized["activation_delta"],
        )
        losses = v17.support_objective_for_weights(
            weights=weights,
            source_weights=source_weights,
            source=source,
            target=target,
        )
        delta = weights - source_weights
        candidates.append({
            **losses,
            "candidate_index": int(candidate_index),
            "delta_norm": float(delta.norm().item()),
            "post_scale": float(post_scale),
            "support_objective": float(losses["objective"]),
            "weights": weights.detach().clone(),
        })
    best = select_post_scale_candidate(candidates)
    metadata = {
        **init["metadata"],
        "coefficient_anchor_hash": stable_hash_json(tensor_to_hashable(init.get("anchor", init["coefficients"]))),
        "coefficient_dim": int(optimized["coefficients"].numel()),
        "component_gate_mean": float(optimized["component_gates"].mean().item()),
        "control_type": control_type,
        "final_activation_scale": float(optimized["activation_scale"].item()),
        "global_scale": float(optimized["global_scale"].item()),
        "optimizer_last_terms": optimized["last_terms"],
        "optimizer_steps": OPTIMIZER_STEPS,
        "selected_candidate_index": int(best["candidate_index"]),
        "selected_delta_norm": float(best["delta_norm"]),
        "selected_objective": float(best["objective"]),
        "selected_post_scale": float(best["post_scale"]),
        "train_statistics_hash": train_stats.get("train_statistics_hash"),
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


def random_basis_constrained_delta(
    *,
    matched_delta: torch.Tensor,
    subject_id: str,
    source: str,
    target: str,
    index: int,
    train_stats: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    matched_norm = matched_delta.detach().to(dtype=torch.float32).norm()
    seed_payload = {
        "index": int(index),
        "matched_delta_norm": tensor_to_hashable(matched_norm.reshape(1)),
        "method": EDITOR_METHOD,
        "source": source,
        "subject_id": subject_id,
        "target": target,
        "train_statistics_hash": train_stats["train_statistics_hash"],
    }
    seed_hash = stable_hash_json(seed_payload)
    seed = int(seed_hash[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction_bases = train_stats["layerwise_bases"][v17.direction_key(source, target)]
    coeff_dim = coefficient_dim_for_bases(direction_bases)
    coeff = torch.randn(coeff_dim, generator=generator, dtype=torch.float32)
    gates = torch.ones(len(v17.LAYER_COMPONENT_SPECS), dtype=torch.float32)
    raw_weights = decode_coefficients(
        coefficients=coeff,
        component_gates=gates,
        global_scale=torch.tensor(1.0, dtype=torch.float32),
        source_weights=torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.float32),
        source=source,
        target=target,
        train_stats=train_stats,
        zero_means=True,
    )
    raw_delta = raw_weights
    raw_norm = raw_delta.norm()
    if float(matched_norm.item()) < RANDOM_CONTROL_EPS or float(raw_norm.item()) < RANDOM_CONTROL_EPS:
        final_delta = torch.zeros_like(matched_delta)
        zero_norm = True
    else:
        final_delta = raw_delta / raw_norm * matched_norm
        zero_norm = False
    return final_delta, {
        "basis_hash": stable_hash_json({
            name: tensor_to_hashable(info["basis"])
            for name, info in sorted(direction_bases["components"].items())
        }),
        "coefficient_hash": stable_hash_json(tensor_to_hashable(coeff)),
        "final_norm": float(final_delta.norm().item()),
        "index": int(index),
        "matched_norm": float(matched_norm.item()),
        "raw_norm": float(raw_norm.item()),
        "seed_hash": seed_hash,
        "zero_norm_fallback": bool(zero_norm),
    }


def build_controls(
    *,
    subject: Mapping[str, Any],
    source: str,
    target: str,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    shuffled_signature_norm: torch.Tensor,
    matched_weights: torch.Tensor,
    train_stats: Mapping[str, Any],
    random_controls: int = RANDOM_CONTROLS_PER_RECORD,
) -> list[dict[str, Any]]:
    controls = [control_record_from_weights("no_edit", source_weights, source, target, source_weights)]
    for control_type in [
        "target_label_subspace_optimizer",
        "no_signature_zero_subspace_optimizer",
        "shuffled_signature_subspace_optimizer",
        "source_signature_subspace_optimizer",
    ]:
        weights, metadata = select_subspace_optimizer_edit(
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
            metadata,
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
    for index in range(int(random_controls)):
        delta, metadata = random_basis_constrained_delta(
            matched_delta=matched_delta,
            subject_id=str(subject["subject_id"]),
            source=source,
            target=target,
            index=index,
            train_stats=train_stats,
        )
        controls.append(control_record_from_weights(
            f"random_norm_matched_lowrank_delta_{index:02d}",
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
            or str(control["control_type"]).startswith("random_norm_matched_lowrank_delta_")
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
    matched_weights, matched_metadata = select_subspace_optimizer_edit(
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
        "editor": matched_metadata,
    }
    controls = build_controls(
        subject=subject,
        source=source,
        target=target,
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        shuffled_signature_norm=source_signature_norm if shuffled_signature_norm is None else shuffled_signature_norm,
        matched_weights=matched_weights,
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
            if control["control_type"].startswith("random_norm_matched_lowrank_delta_")
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


def build_v19_seed_preflight() -> dict[str, Any]:
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


def assert_no_forbidden_final_raw_paths(paths: Sequence[Path | str], *, allow_v19_final: bool = False) -> None:
    prior = {path.resolve() for path in PRIOR_FINAL_RAW_PATHS}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in prior:
            raise ValueError(f"prior sealed final raw path is forbidden: {path}")
        if path.name == "final_subjects.json" and "runs" in path.parts:
            if not (allow_v19_final and path == V19_FINAL_RAW.resolve()):
                raise ValueError(f"sealed final raw path is forbidden: {path}")
        if path == V19_FINAL_RAW.resolve() and not allow_v19_final:
            raise ValueError(f"V19 final raw path is forbidden before authorization: {path}")


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
    assert_no_forbidden_final_raw_paths([train_path, eval_path], allow_v19_final=(phase == "final"))
    return failures


def generate_pools(args: argparse.Namespace, pool_dir: Path) -> dict[str, Any]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    seed_preflight = build_v19_seed_preflight()
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
        allow_v19_final=False,
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
        raise ValueError("V19 source-pool contract validation failed: " + "; ".join(contract_failures))
    train_subjects = v16.v15.v1.accepted_records(train_payload)
    eval_subjects = v16.v15.v1.accepted_records(eval_payload)
    train_stats = fit_v19_train_statistics(
        train_subjects,
        include_models=True,
        include_baseline_stats=True,
    )
    stats_path = output_dir / "v19_signature_initialized_subspace_optimizer_stats.pt"
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
        "V19 final is not implemented until development passes and reviewer authorizes final"
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
