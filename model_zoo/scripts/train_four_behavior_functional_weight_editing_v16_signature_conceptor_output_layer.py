"""V16 functional editing via signature-conditioned conceptor output-layer edits."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

import train_four_behavior_functional_weight_editing_v15_signature_conditioned_hypernetwork as v15


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v16_pools"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runs"
    / "four_behavior_functional_weight_editing_v16_signature_conceptor_output_layer"
)
PREREG_PATH = (
    REPO_ROOT
    / "docs"
    / "preregistrations"
    / "four_behavior_functional_weight_editing_v16_signature_conceptor_output_layer.md"
)

SOURCE_POOL_SCOPE = "four_behavior_functional_weight_editing_v16_source_pool"
SOURCE_AUDIT_SCOPE = "four_behavior_functional_weight_editing_v16_source_pool_construction"
FINAL_REDACTED_SCOPE = "redacted_final_functional_weight_editing_v16_source_pool_audit_surface_only"
DEVELOPMENT_SCOPE = (
    "four_behavior_functional_weight_editing_v16_signature_conceptor_output_layer_development"
)
FINAL_SCOPE = "four_behavior_functional_weight_editing_v16_signature_conceptor_output_layer_final"
EDITOR_METHOD = "signature_conditioned_conceptor_output_layer_v16"

PATTERNS = v15.PATTERNS
V16_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"
PRIOR_FINAL_RAW_PATHS = {
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v13_pools" / "final_subjects.json",
    v15.v14.V14_FINAL_RAW,
    v15.V15_FINAL_RAW,
}

POOL_CONFIGS = {
    "train": {
        "base_seed": 78300000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 79300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 80300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}
SEED_BEHAVIOR_STRIDE = v15.SEED_BEHAVIOR_STRIDE

APERTURE_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
BETA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
CONCEPTOR_RIDGE = 1e-4
V16_RANDOM_CONTROL_SEED = 20260630
OUTPUT_WEIGHT_START = 336
OUTPUT_WEIGHT_END = 344
OUTPUT_BIAS_INDEX = 344

FINAL_COMBINED_SUMMARY_ALLOWED_KEYS = {
    "accepted_counts_by_behavior",
    "pool_file_sha256",
    "pool_redacted_payload_sha256",
}
FINAL_REDACTED_ALLOWED_KEYS = {
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
FORBIDDEN_FINAL_DETAIL_KEYS = {
    "accepted_subject_ids",
    "per_attempt_details",
    "records",
    "rejected_subject_ids",
    "signature",
    "signature_hash",
    "subject_ids",
    "weights",
    "weights_hash",
}

REQUIRED_NON_RANDOM_CONTROL_TYPES = [
    "no_edit",
    "target_label_centroid_conceptor",
    "shuffled_signature_conceptor",
    "source_signature_conceptor",
    "nearest_train_target_conceptor",
    "activation_addition_mean_shift",
    "activation_conceptor_no_shift",
    "output_layer_no_signature_support_optimizer",
    "output_layer_random_conceptor",
    "v13_no_signature_support_optimizer",
]
RANDOM_OUTPUT_LAYER_CONTROL_TYPES = [
    f"random_norm_matched_output_layer_delta:{index:02d}"
    for index in range(16)
]
V16_GATING_CONTROL_TYPES = [
    control_type
    for control_type in REQUIRED_NON_RANDOM_CONTROL_TYPES
    if control_type != "v13_no_signature_support_optimizer"
] + RANDOM_OUTPUT_LAYER_CONTROL_TYPES

THRESHOLDS = {
    "expected_controls_per_record": 26,
    "expected_non_random_controls_per_record": 10,
    "random_controls_per_record": 16,
    "expected_record_count": 288,
    "expected_per_direction_count": 24,
    "min_aggregate_target_prediction_rate": 0.90,
    "min_aggregate_conflict_target_accuracy": 0.80,
    "min_aggregate_conflict_target_accuracy_improvement": 0.35,
    "min_aggregate_individual_pass_rate": 0.85,
    "min_aggregate_pareto_undominated_rate": 0.85,
    "min_mean_matched_target_margin": 0.35,
    "min_aggregate_target_label_target_margin_advantage": 0.02,
    "min_aggregate_target_label_compatible_mse_advantage": 2.0,
    "min_aggregate_shuffled_target_margin_advantage": 0.05,
    "min_aggregate_shuffled_compatible_mse_advantage": 2.0,
    "min_aggregate_no_signature_target_margin_advantage": 0.02,
    "min_aggregate_no_signature_compatible_mse_advantage": 2.0,
    "max_compile_logit_difference": 1e-5,
    "min_direction_target_prediction_rate": 0.85,
    "min_direction_conflict_target_accuracy": 0.65,
    "min_direction_mean_target_margin": 0.15,
    "min_direction_individual_pass_rate": 0.70,
    "min_direction_pareto_undominated_rate": 0.70,
    "min_per_record_target_margin": 0.15,
    "min_per_record_conflict_target_accuracy": 0.65,
    "min_per_record_conflict_target_accuracy_improvement": 0.15,
    "min_per_record_control_target_margin_advantage": 0.02,
    "min_per_record_control_compatible_mse_advantage": 2.0,
}

_WORKER_TRAIN_STATS: Mapping[str, Any] | None = None
_WORKER_CLASSIFIER: torch.nn.Module | None = None
_WORKER_RANDOM_CONTROLS = 16
_WORKER_RECORD_EVALUATOR: Any = None


def stable_hash_json(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def conceptor_from_activations(
    activations: torch.Tensor,
    *,
    aperture: float,
    ridge: float = CONCEPTOR_RIDGE,
) -> torch.Tensor:
    if activations.ndim != 2:
        raise ValueError("activations must have shape [n_examples, hidden_dim]")
    values = activations.to(dtype=torch.float32)
    covariance = values.T @ values / max(1, int(values.shape[0]))
    eye = torch.eye(int(covariance.shape[0]), dtype=torch.float32, device=values.device)
    regularized = covariance + (float(aperture) ** -2) * eye + float(ridge) * eye
    conceptor = covariance @ torch.linalg.inv(regularized)
    conceptor = 0.5 * (conceptor + conceptor.T)
    return conceptor.to(dtype=torch.float32)


def compile_hidden_steering_to_output_layer(
    *,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    operator: torch.Tensor,
    shift: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = output_weight.to(dtype=torch.float32)
    bias = output_bias.to(dtype=torch.float32)
    op = operator.to(dtype=torch.float32)
    vector = shift.to(dtype=torch.float32)
    edited_weight = weight @ op
    edited_bias = bias + (weight @ vector.reshape(-1, 1)).reshape_as(bias)
    return edited_weight, edited_bias


def max_compile_logit_difference(
    *,
    hidden: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    operator: torch.Tensor,
    shift: torch.Tensor,
) -> float:
    edited_weight, edited_bias = compile_hidden_steering_to_output_layer(
        output_weight=output_weight,
        output_bias=output_bias,
        operator=operator,
        shift=shift,
    )
    explicit = (hidden @ operator.T + shift) @ output_weight.T + output_bias
    compiled = hidden @ edited_weight.T + edited_bias
    return float((explicit - compiled).abs().max().item())


def compile_hidden_steering_to_flat_weights(
    *,
    source_weights: torch.Tensor,
    operator: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    flat = source_weights.detach().clone().to(dtype=torch.float32)
    output_weight = flat[OUTPUT_WEIGHT_START:OUTPUT_WEIGHT_END].reshape(1, 8)
    output_bias = flat[OUTPUT_BIAS_INDEX:OUTPUT_BIAS_INDEX + 1]
    edited_weight, edited_bias = compile_hidden_steering_to_output_layer(
        output_weight=output_weight,
        output_bias=output_bias,
        operator=operator,
        shift=shift,
    )
    flat[OUTPUT_WEIGHT_START:OUTPUT_WEIGHT_END] = edited_weight.reshape(-1)
    flat[OUTPUT_BIAS_INDEX] = edited_bias.reshape(-1)[0]
    return flat


def signature_weighted_target_operator(
    *,
    train_stats: Mapping[str, Any],
    target_behavior: str,
    target_signature_norm: torch.Tensor,
    aperture: float,
) -> dict[str, Any]:
    records = list(train_stats["train_by_behavior"][target_behavior])
    if not records:
        raise ValueError(f"no train records for behavior {target_behavior}")
    target = target_signature_norm.to(dtype=torch.float32)
    distances = torch.tensor([
        torch.mean((record["signature_norm"].to(dtype=torch.float32) - target) ** 2).item()
        for record in records
    ], dtype=torch.float32)
    weights = torch.softmax(-distances, dim=0)
    means = torch.stack([
        record["activation_mean"].to(dtype=torch.float32)
        for record in records
    ])
    conceptors = torch.stack([
        record["conceptors_by_aperture"][float(aperture)].to(dtype=torch.float32)
        for record in records
    ])
    target_mean = torch.sum(weights[:, None] * means, dim=0)
    target_conceptor = torch.sum(weights[:, None, None] * conceptors, dim=0)
    weighted_subjects = [
        {
            "subject_id_hash": stable_hash_json(str(record["subject_id"])),
            "weight": float(weight.item()),
            "signature_distance": float(distance.item()),
        }
        for record, weight, distance in zip(records, weights, distances)
    ]
    return {
        "aperture": float(aperture),
        "target_behavior": target_behavior,
        "target_conceptor": target_conceptor,
        "target_mean": target_mean,
        "weighted_subjects": weighted_subjects,
    }


def probe_inputs_tensor(probe_examples: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    values = []
    for example in probe_examples:
        value = example.get("sequence", example.get("input"))
        if value is None:
            raise KeyError("probe example is missing both 'sequence' and 'input'")
        values.append(value)
    return torch.tensor(values, dtype=torch.float32)


def subject_behavior(record: Mapping[str, Any]) -> str:
    """Return the source behavior key used by generated subject pool records."""
    value = record.get("pattern", record.get("behavior"))
    if value is None:
        raise KeyError("subject record is missing both 'pattern' and 'behavior'")
    return str(value)


def records_by_behavior(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped = {pattern: [] for pattern in PATTERNS}
    for record in records:
        grouped[subject_behavior(record)].append(record)
    for values in grouped.values():
        values.sort(key=lambda item: str(item["subject_id"]))
    return grouped


def fit_v16_train_statistics(
    subjects: Sequence[Mapping[str, Any]],
    *,
    probe_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered_subjects = sorted(subjects, key=lambda item: (subject_behavior(item), str(item["subject_id"])))
    signatures = torch.tensor([record["signature"] for record in ordered_subjects], dtype=torch.float32)
    sig_mean = signatures.mean(dim=0)
    sig_std = signatures.std(dim=0, unbiased=False).clamp_min(1e-6)
    weights = torch.tensor([record["weights"] for record in ordered_subjects], dtype=torch.float32)
    probe_inputs = probe_inputs_tensor(probe_examples)
    last_hidden = v15.hidden_activations_flat_batch(weights, probe_inputs)[-1]

    enriched = []
    for index, record in enumerate(ordered_subjects):
        activations = last_hidden[index]
        conceptors_by_aperture = {
            float(aperture): conceptor_from_activations(
                activations,
                aperture=float(aperture),
                ridge=CONCEPTOR_RIDGE,
            )
            for aperture in APERTURE_GRID
        }
        enriched.append({
            **record,
            "activation_mean": activations.mean(dim=0).detach().clone(),
            "conceptors_by_aperture": conceptors_by_aperture,
            "signature_norm": ((signatures[index] - sig_mean) / sig_std).detach().clone(),
        })

    grouped = records_by_behavior(enriched)
    centroids = {
        behavior: torch.stack([
            record["signature_norm"]
            for record in records
        ]).mean(dim=0)
        for behavior, records in grouped.items()
        if records
    }
    hash_payload = {
        "aperture_grid": [float(value) for value in APERTURE_GRID],
        "conceptor_ridge": float(CONCEPTOR_RIDGE),
        "probe_examples_hash": stable_hash_json(probe_examples),
        "subject_ids": [str(record["subject_id"]) for record in ordered_subjects],
    }
    stats = {
        "probe_examples": list(probe_examples),
        "probe_examples_hash": hash_payload["probe_examples_hash"],
        "sig_mean": sig_mean,
        "sig_std": sig_std,
        "signature_centroids": centroids,
        "train_by_behavior": grouped,
        "train_subjects": enriched,
        "train_statistics_hash": stable_hash_json(hash_payload),
    }
    return stats


def source_activation_stats(
    *,
    source_weights: torch.Tensor,
    probe_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    probe_inputs = probe_inputs_tensor(probe_examples)
    activations = v15.hidden_activations_flat_batch(
        source_weights.reshape(1, -1).to(dtype=torch.float32),
        probe_inputs,
    )[-1][0]
    return {
        "activation_mean": activations.mean(dim=0),
        "conceptors_by_aperture": {
            float(aperture): conceptor_from_activations(
                activations,
                aperture=float(aperture),
                ridge=CONCEPTOR_RIDGE,
            )
            for aperture in APERTURE_GRID
        },
        "last_hidden": activations,
    }


def support_objective_for_weights(
    *,
    edited_weights: torch.Tensor,
    source_weights: torch.Tensor,
    source: str,
    target: str,
) -> float:
    support = v15.v14.prepare_support_tensors_with_source_logits(
        source_weights=source_weights,
        source=source,
        target=target,
    )
    with torch.no_grad():
        target_logits = v15.v10.decoder_v1.subject_forward_flat_batch(
            edited_weights.reshape(1, -1),
            support["target_inputs"],
        )[0]
        compatible_logits = v15.v10.decoder_v1.subject_forward_flat_batch(
            edited_weights.reshape(1, -1),
            support["compatible_inputs"],
        )[0]
    target_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        target_logits,
        support["target_labels"],
    )
    compatible_mse = torch.nn.functional.mse_loss(
        compatible_logits,
        support["compatible_source_logits"],
    )
    output_delta = edited_weights[OUTPUT_WEIGHT_START:] - source_weights[OUTPUT_WEIGHT_START:]
    output_l2 = torch.mean(output_delta.pow(2))
    objective = 4.0 * target_bce + 0.05 * compatible_mse + 0.0005 * output_l2
    return float(objective.item())


def weighted_operator_from_records(
    *,
    records: Sequence[Mapping[str, Any]],
    weights: torch.Tensor,
    aperture: float,
    target_behavior: str,
) -> dict[str, Any]:
    normalized = weights.to(dtype=torch.float32)
    normalized = normalized / normalized.sum().clamp_min(1e-12)
    means = torch.stack([record["activation_mean"].to(dtype=torch.float32) for record in records])
    conceptors = torch.stack([
        record["conceptors_by_aperture"][float(aperture)].to(dtype=torch.float32)
        for record in records
    ])
    return {
        "aperture": float(aperture),
        "target_behavior": target_behavior,
        "target_conceptor": torch.sum(normalized[:, None, None] * conceptors, dim=0),
        "target_mean": torch.sum(normalized[:, None] * means, dim=0),
        "weighted_subjects": [
            {
                "subject_id_hash": stable_hash_json(str(record["subject_id"])),
                "weight": float(weight.item()),
            }
            for record, weight in zip(records, normalized)
        ],
    }


def select_compiled_conceptor_edit(
    *,
    source_weights: torch.Tensor,
    source: str,
    target: str,
    source_stats: Mapping[str, Any],
    target_operator_by_aperture: Mapping[float, Mapping[str, Any]],
    alpha_grid: Sequence[float] = ALPHA_GRID,
    beta_grid: Sequence[float] = BETA_GRID,
) -> tuple[torch.Tensor, dict[str, Any]]:
    best: tuple[
        float,
        float,
        float,
        float,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ] | None = None
    eye = torch.eye(8, dtype=torch.float32)
    for aperture in APERTURE_GRID:
        aperture = float(aperture)
        target_operator = target_operator_by_aperture[aperture]
        target_conceptor = target_operator["target_conceptor"].to(dtype=torch.float32)
        source_conceptor = source_stats["conceptors_by_aperture"][aperture].to(dtype=torch.float32)
        target_mean = target_operator["target_mean"].to(dtype=torch.float32)
        source_mean = source_stats["activation_mean"].to(dtype=torch.float32)
        for alpha in alpha_grid:
            operator = eye + float(alpha) * (target_conceptor - source_conceptor)
            for beta in beta_grid:
                shift = float(beta) * (target_mean - source_mean)
                edited = compile_hidden_steering_to_flat_weights(
                    source_weights=source_weights,
                    operator=operator,
                    shift=shift,
                )
                objective = support_objective_for_weights(
                    edited_weights=edited,
                    source_weights=source_weights,
                    source=source,
                    target=target,
                )
                metadata = {
                    "aperture": aperture,
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "support_objective": objective,
                    "target_operator": {
                        key: value
                        for key, value in target_operator.items()
                        if key not in {"target_conceptor", "target_mean"}
                    },
                }
                candidate = (
                    objective,
                    aperture,
                    float(alpha),
                    float(beta),
                    edited,
                    operator.detach().clone(),
                    shift.detach().clone(),
                    metadata,
                )
                if best is None or candidate[:4] < best[:4]:
                    best = candidate
    if best is None:
        raise ValueError("no conceptor edit candidates generated")
    _, _, _, _, edited_weights, operator, shift, metadata = best
    source_hidden = source_stats["last_hidden"].to(dtype=torch.float32)
    metadata["compile_max_abs_logit_diff"] = max_compile_logit_difference(
        hidden=source_hidden,
        output_weight=source_weights[OUTPUT_WEIGHT_START:OUTPUT_WEIGHT_END].reshape(1, 8),
        output_bias=source_weights[OUTPUT_BIAS_INDEX:OUTPUT_BIAS_INDEX + 1],
        operator=operator,
        shift=shift,
    )
    metadata["operator_fro_norm"] = float(operator.norm().item())
    metadata["shift_norm"] = float(shift.norm().item())
    return edited_weights, metadata


def output_layer_no_signature_support_optimizer(
    *,
    source_weights: torch.Tensor,
    source: str,
    target: str,
    subject_id: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    support = v15.v14.prepare_support_tensors_with_source_logits(
        source_weights=source_weights,
        source=source,
        target=target,
    )
    with torch.no_grad():
        target_hidden = v15.hidden_activations_flat_batch(
            source_weights.reshape(1, -1),
            support["target_inputs"],
        )[-1][0]
        compatible_hidden = v15.hidden_activations_flat_batch(
            source_weights.reshape(1, -1),
            support["compatible_inputs"],
        )[-1][0]
    output_weight = source_weights[OUTPUT_WEIGHT_START:OUTPUT_WEIGHT_END].reshape(1, 8).clone()
    output_bias = source_weights[OUTPUT_BIAS_INDEX:OUTPUT_BIAS_INDEX + 1].clone()
    output_weight = torch.nn.Parameter(output_weight)
    output_bias = torch.nn.Parameter(output_bias)
    seed = int(stable_hash_json([
        "v16_output_layer_no_signature_support_optimizer",
        subject_id,
        target,
        20260631,
    ])[:16], 16) % (2**31)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        [output_weight, output_bias],
        lr=0.05,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    source_output = source_weights[OUTPUT_WEIGHT_START:OUTPUT_BIAS_INDEX + 1].detach()
    for _ in range(250):
        optimizer.zero_grad(set_to_none=True)
        target_logits = (target_hidden @ output_weight.T + output_bias).reshape(-1)
        compatible_logits = (compatible_hidden @ output_weight.T + output_bias).reshape(-1)
        target_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            target_logits,
            support["target_labels"],
        )
        compatible_mse = torch.nn.functional.mse_loss(
            compatible_logits,
            support["compatible_source_logits"],
        )
        current_output = torch.cat([output_weight.reshape(-1), output_bias.reshape(-1)])
        output_l2 = torch.mean((current_output - source_output).pow(2))
        loss = 4.0 * target_bce + 0.05 * compatible_mse + 0.0005 * output_l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_([output_weight, output_bias], 10.0)
        optimizer.step()
    edited = source_weights.detach().clone().to(dtype=torch.float32)
    edited[OUTPUT_WEIGHT_START:OUTPUT_WEIGHT_END] = output_weight.detach().reshape(-1)
    edited[OUTPUT_BIAS_INDEX] = output_bias.detach().reshape(-1)[0]
    return edited, {
        "optimizer": "AdamW",
        "steps": 250,
        "lr": 0.05,
        "seed": int(seed),
    }


def random_output_layer_controls(
    *,
    source_weights: torch.Tensor,
    matched_weights: torch.Tensor,
    subject_id: str,
    target: str,
) -> list[tuple[str, torch.Tensor, dict[str, Any]]]:
    matched_delta = matched_weights[OUTPUT_WEIGHT_START:] - source_weights[OUTPUT_WEIGHT_START:]
    norm = matched_delta.norm().clamp_min(1e-12)
    controls = []
    for index, control_type in enumerate(RANDOM_OUTPUT_LAYER_CONTROL_TYPES):
        seed = int(stable_hash_json([
            subject_id,
            target,
            control_type,
            index,
            V16_RANDOM_CONTROL_SEED,
        ])[:16], 16) % (2**31)
        generator = torch.Generator().manual_seed(seed)
        random_delta = torch.randn(9, generator=generator, dtype=torch.float32)
        random_delta = random_delta / random_delta.norm().clamp_min(1e-12) * norm
        edited = source_weights.detach().clone().to(dtype=torch.float32)
        edited[OUTPUT_WEIGHT_START:] = edited[OUTPUT_WEIGHT_START:] + random_delta
        controls.append((control_type, edited, {"random_index": index, "random_seed": int(seed)}))
    return controls


def target_operator_grid_from_signature(
    *,
    train_stats: Mapping[str, Any],
    target_behavior: str,
    target_signature_norm: torch.Tensor,
) -> dict[float, Mapping[str, Any]]:
    return {
        float(aperture): signature_weighted_target_operator(
            train_stats=train_stats,
        target_behavior=target_behavior,
        target_signature_norm=target_signature_norm,
        aperture=float(aperture),
    )
        for aperture in APERTURE_GRID
    }


def target_label_centroid_operator_grid(
    *,
    train_stats: Mapping[str, Any],
    target_behavior: str,
) -> dict[float, Mapping[str, Any]]:
    records = list(train_stats["train_by_behavior"][target_behavior])
    weights = torch.ones(len(records), dtype=torch.float32)
    return {
        float(aperture): weighted_operator_from_records(
            records=records,
            weights=weights,
            aperture=float(aperture),
            target_behavior=target_behavior,
        )
        for aperture in APERTURE_GRID
    }


def nearest_target_operator_grid(
    *,
    train_stats: Mapping[str, Any],
    target_behavior: str,
    target_signature_norm: torch.Tensor,
) -> dict[float, Mapping[str, Any]]:
    records = list(train_stats["train_by_behavior"][target_behavior])
    distances = [
        float(torch.mean((record["signature_norm"] - target_signature_norm) ** 2).item())
        for record in records
    ]
    nearest_index = min(range(len(records)), key=lambda index: (distances[index], str(records[index]["subject_id"])))
    weights = torch.zeros(len(records), dtype=torch.float32)
    weights[nearest_index] = 1.0
    return {
        float(aperture): weighted_operator_from_records(
            records=records,
            weights=weights,
            aperture=float(aperture),
            target_behavior=target_behavior,
        )
        for aperture in APERTURE_GRID
    }


def shuffled_behavior_for(*, source: str, target: str, subject_id: str) -> str:
    candidates = [pattern for pattern in PATTERNS if pattern != target]
    key = stable_hash_json(["v16_shuffled_signature", source, target, subject_id])
    return candidates[int(key[:8], 16) % len(candidates)]


def random_conceptor_operator_grid(
    *,
    matched_operator_grid: Mapping[float, Mapping[str, Any]],
    subject_id: str,
    target: str,
) -> dict[float, Mapping[str, Any]]:
    result = {}
    for aperture, matched in matched_operator_grid.items():
        seed = int(stable_hash_json([
            "v16_output_layer_random_conceptor",
            subject_id,
            target,
            aperture,
        ])[:16], 16) % (2**31)
        generator = torch.Generator().manual_seed(seed)
        matrix = torch.randn(8, 8, generator=generator, dtype=torch.float32)
        psd = matrix @ matrix.T
        psd = psd / psd.norm().clamp_min(1e-12) * matched["target_conceptor"].norm().clamp_min(1e-12)
        result[float(aperture)] = {
            "aperture": float(aperture),
            "target_behavior": target,
            "target_conceptor": psd,
            "target_mean": matched["target_mean"],
            "weighted_subjects": [{"random_seed": int(seed)}],
        }
    return result


def control_record_from_weights(
    control_type: str,
    weights: torch.Tensor,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return v15.v14.control_record(
        control_type,
        weights,
        source,
        target,
        source_weights,
        metadata=metadata,
    )


def evaluate_record(
    *,
    subject: Mapping[str, Any],
    z: torch.Tensor,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    random_controls: int = 16,
) -> dict[str, Any]:
    selected = v15.v10.selected_v9_conditioning(
        z=z,
        source=source,
        target=target,
        train_stats=train_stats,
        classifier=classifier,
    )
    source_stats = source_activation_stats(
        source_weights=source_weights,
        probe_examples=train_stats["probe_examples"],
    )
    matched_grid = target_operator_grid_from_signature(
        train_stats=train_stats,
        target_behavior=target,
        target_signature_norm=selected["candidate_z"],
    )
    matched_weights, matched_meta = select_compiled_conceptor_edit(
        source_weights=source_weights,
        source=source,
        target=target,
        source_stats=source_stats,
        target_operator_by_aperture=matched_grid,
    )
    matched_delta_norm = (matched_weights - source_weights).norm()
    matched = {
        **v15.v14.functional_metrics(matched_weights, source, target, source_weights),
        "compile_max_abs_logit_diff": matched_meta["compile_max_abs_logit_diff"],
        "delta_norm": float(matched_delta_norm.item()),
        "editor": matched_meta,
        "selected_candidate_index": int(selected["candidate_index"]),
        "selected_centroid_improvement": selected["selected_centroid_improvement"],
        "selected_displacement_norm": selected["selected_displacement_norm"],
        "selected_primary_target_margin": selected["selected_primary_target_margin"],
    }

    controls = build_controls(
        subject=subject,
        z=z,
        selected_signature_norm=selected["candidate_z"],
        source=source,
        target=target,
        source_weights=source_weights,
        source_stats=source_stats,
        matched_weights=matched_weights,
        matched_grid=matched_grid,
        train_stats=train_stats,
        random_controls=random_controls,
    )
    for control in controls:
        control["matched_minus_control_target_margin"] = (
            matched["target_margin"] - control["target_margin"]
        )
        control["control_minus_matched_compatible_source_output_mse"] = (
            control["compatible_source_output_mse"]
            - matched["compatible_source_output_mse"]
        )
    gating_controls = [
        control for control in controls if control["control_type"] in set(V16_GATING_CONTROL_TYPES)
    ]
    pareto_dominators = [
        control
        for control in gating_controls
        if v15.v14.pareto_dominates_functional(control, matched)
    ]
    best_target = max(gating_controls, key=lambda item: item["target_margin"])
    matched["pareto_dominator_count"] = len(pareto_dominators)
    matched["pareto_dominator_types"] = sorted({control["control_type"] for control in pareto_dominators})
    matched["pareto_undominated"] = not pareto_dominators
    matched["target_prediction_pass"] = matched["predicted_behavior"] == target
    for metric_name, control_name in SIGNATURE_ADVANTAGE_CONTROLS.items():
        control = single_control(controls, control_name)
        matched[f"matched_minus_{metric_name}_target_margin"] = (
            matched["target_margin"] - control["target_margin"]
        )
        matched[f"{metric_name}_minus_matched_compatible_source_output_mse"] = (
            control["compatible_source_output_mse"]
            - matched["compatible_source_output_mse"]
        )
    matched["individual_all_gates_passed"] = individual_passed(matched)
    return {
        "controls": v15.v10.strip_control_weights(controls),
        "individual_all_gates_passed": matched["individual_all_gates_passed"],
        "matched": v15.v10.strip_weight(matched),
        "random_control_count": sum(
            1
            for control in controls
            if control["control_type"].startswith("random_norm_matched_output_layer_delta")
        ),
        "source_behavior": source,
        "subject_id": str(subject["subject_id"]),
        "summary": {
            "best_control_target_margin": best_target["target_margin"],
            "best_control_type": best_target["control_type"],
            "matched_minus_best_control_target_margin": (
                matched["target_margin"] - best_target["target_margin"]
            ),
            "pareto_undominated": matched["pareto_undominated"],
            "target_prediction_pass": matched["target_prediction_pass"],
            **signature_summary_fields(matched),
        },
        "target_behavior": target,
    }


SIGNATURE_ADVANTAGE_CONTROLS = {
    "shuffled_signature": "shuffled_signature_conceptor",
    "target_label": "target_label_centroid_conceptor",
    "output_layer_no_signature": "output_layer_no_signature_support_optimizer",
}


def build_controls(
    *,
    subject: Mapping[str, Any],
    z: torch.Tensor,
    selected_signature_norm: torch.Tensor,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    source_stats: Mapping[str, Any],
    matched_weights: torch.Tensor,
    matched_grid: Mapping[float, Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    random_controls: int,
) -> list[dict[str, Any]]:
    controls = [
        control_record_from_weights("no_edit", source_weights, source, target, source_weights)
    ]
    conceptor_controls = {
        "target_label_centroid_conceptor": target_label_centroid_operator_grid(
            train_stats=train_stats,
            target_behavior=target,
        ),
        "source_signature_conceptor": target_operator_grid_from_signature(
            train_stats=train_stats,
            target_behavior=target,
            target_signature_norm=z,
        ),
        "nearest_train_target_conceptor": nearest_target_operator_grid(
            train_stats=train_stats,
            target_behavior=target,
            target_signature_norm=selected_signature_norm,
        ),
        "activation_addition_mean_shift": matched_grid,
        "activation_conceptor_no_shift": matched_grid,
        "output_layer_random_conceptor": random_conceptor_operator_grid(
            matched_operator_grid=matched_grid,
            subject_id=str(subject["subject_id"]),
            target=target,
        ),
    }
    shuffled = shuffled_behavior_for(source=source, target=target, subject_id=str(subject["subject_id"]))
    conceptor_controls["shuffled_signature_conceptor"] = target_operator_grid_from_signature(
        train_stats=train_stats,
        target_behavior=target,
        target_signature_norm=train_stats["signature_centroids"][shuffled],
    )
    for control_type, grid in conceptor_controls.items():
        alpha_grid = ALPHA_GRID
        beta_grid = BETA_GRID
        if control_type == "activation_addition_mean_shift":
            alpha_grid = [0.0]
        if control_type == "activation_conceptor_no_shift":
            beta_grid = [0.0]
        weights, metadata = select_compiled_conceptor_edit(
            source_weights=source_weights,
            source=source,
            target=target,
            source_stats=source_stats,
            target_operator_by_aperture=grid,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
        )
        controls.append(control_record_from_weights(
            control_type,
            weights,
            source,
            target,
            source_weights,
            metadata=metadata,
        ))
    optimized_weights, optimized_meta = output_layer_no_signature_support_optimizer(
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
        metadata=optimized_meta,
    ))
    controls.append(v15.v14.optimizer_control_record(
        control_type="v13_no_signature_support_optimizer",
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats,
        signature_target_norm=selected_signature_norm,
        loss_weights={**v15.v14.matched_loss_weights(), "signature": 0.0},
    ))
    for control_type, weights, metadata in random_output_layer_controls(
        source_weights=source_weights,
        matched_weights=matched_weights,
        subject_id=str(subject["subject_id"]),
        target=target,
    )[:random_controls]:
        controls.append(control_record_from_weights(
            control_type,
            weights,
            source,
            target,
            source_weights,
            metadata=metadata,
        ))
    return controls


def single_control(controls: Sequence[Mapping[str, Any]], control_type: str) -> Mapping[str, Any]:
    matches = [control for control in controls if control["control_type"] == control_type]
    if len(matches) != 1:
        raise ValueError(f"expected one {control_type} control, found {len(matches)}")
    return matches[0]


def signature_summary_fields(matched: Mapping[str, Any]) -> dict[str, float]:
    fields = {}
    for metric_name in SIGNATURE_ADVANTAGE_CONTROLS:
        fields[f"matched_minus_{metric_name}_target_margin"] = matched[
            f"matched_minus_{metric_name}_target_margin"
        ]
        fields[f"{metric_name}_minus_matched_compatible_source_output_mse"] = matched[
            f"{metric_name}_minus_matched_compatible_source_output_mse"
        ]
    return fields


def individual_passed(matched: Mapping[str, Any]) -> bool:
    signature_advantages = []
    for metric_name in SIGNATURE_ADVANTAGE_CONTROLS:
        signature_advantages.append(
            matched[f"matched_minus_{metric_name}_target_margin"]
            >= THRESHOLDS["min_per_record_control_target_margin_advantage"]
            or matched[f"{metric_name}_minus_matched_compatible_source_output_mse"]
            >= THRESHOLDS["min_per_record_control_compatible_mse_advantage"]
        )
    return bool(
        matched["target_prediction_pass"]
        and matched["target_margin"] >= THRESHOLDS["min_per_record_target_margin"]
        and matched["conflict_target_accuracy"] >= THRESHOLDS["min_per_record_conflict_target_accuracy"]
        and matched["conflict_target_accuracy_improvement"]
        >= THRESHOLDS["min_per_record_conflict_target_accuracy_improvement"]
        and matched["pareto_undominated"]
        and matched["compile_max_abs_logit_diff"] <= THRESHOLDS["max_compile_logit_difference"]
        and all(signature_advantages)
    )


def evaluate_subjects(
    *,
    subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    random_controls: int = 16,
    parallel: bool = True,
    record_evaluator: Any = None,
) -> dict[str, Any]:
    if record_evaluator is None:
        record_evaluator = evaluate_record_from_job
    jobs = []
    for subject in subjects:
        source = v15.v1.behavior_of(subject)
        for target in PATTERNS:
            if target == source:
                continue
            jobs.append({
                "source": source,
                "subject": subject,
                "target": target,
            })
    if parallel and len(jobs) > 1:
        contract = multiprocessing_contract()
        context = mp.get_context(contract["start_method"])
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(contract["max_workers"]),
            mp_context=context,
            initializer=_init_eval_worker,
            initargs=(train_stats, classifier, int(random_controls), record_evaluator),
        ) as executor:
            records = list(executor.map(_evaluate_record_worker, jobs))
    else:
        records = [
            record_evaluator(
                job,
                train_stats=train_stats,
                classifier=classifier,
                random_controls=random_controls,
            )
            for job in jobs
        ]
    records = list(sort_records_for_artifact(records))
    aggregate = summarize_records(records)
    by_direction = {
        v15.v1.vector_key(source, target): summarize_records([
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


def _init_eval_worker(
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    random_controls: int,
    record_evaluator: Any,
) -> None:
    global _WORKER_CLASSIFIER
    global _WORKER_RANDOM_CONTROLS
    global _WORKER_RECORD_EVALUATOR
    global _WORKER_TRAIN_STATS
    torch.set_num_threads(1)
    _WORKER_TRAIN_STATS = train_stats
    _WORKER_CLASSIFIER = classifier
    _WORKER_RANDOM_CONTROLS = int(random_controls)
    _WORKER_RECORD_EVALUATOR = record_evaluator


def _evaluate_record_worker(job: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_TRAIN_STATS is None or _WORKER_CLASSIFIER is None:
        raise RuntimeError("V16 worker was not initialized")
    evaluator = _WORKER_RECORD_EVALUATOR or evaluate_record_from_job
    return evaluator(
        job,
        train_stats=_WORKER_TRAIN_STATS,
        classifier=_WORKER_CLASSIFIER,
        random_controls=_WORKER_RANDOM_CONTROLS,
    )


def lightweight_record_evaluator_for_tests(
    job: Mapping[str, Any],
    *,
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    random_controls: int,
) -> dict[str, Any]:
    source = str(job["source"])
    target = str(job["target"])
    subject_id = str(job["subject"]["subject_id"])
    margin = float((len(source) + len(target) + len(subject_id)) % 7) / 10.0
    matched = {
        "compile_max_abs_logit_diff": 0.0,
        "conflict_target_accuracy": 0.75,
        "conflict_target_accuracy_improvement": 0.25,
        "predicted_behavior": target,
        "target_margin": margin,
    }
    for metric_name in SIGNATURE_ADVANTAGE_CONTROLS:
        matched[f"matched_minus_{metric_name}_target_margin"] = 0.1
        matched[f"{metric_name}_minus_matched_compatible_source_output_mse"] = 3.0
    return {
        "controls": [],
        "individual_all_gates_passed": True,
        "matched": matched,
        "random_control_count": int(random_controls),
        "source_behavior": source,
        "subject_id": subject_id,
        "summary": {
            "best_control_target_margin": margin - 0.1,
            "best_control_type": "fixture",
            "matched_minus_best_control_target_margin": 0.1,
            "pareto_undominated": True,
            "target_prediction_pass": True,
            **signature_summary_fields(matched),
        },
        "target_behavior": target,
    }


def evaluate_record_from_job(
    job: Mapping[str, Any],
    *,
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    random_controls: int,
) -> dict[str, Any]:
    subject = job["subject"]
    source = str(job["source"])
    target = str(job["target"])
    source_signature = torch.tensor(subject["signature"], dtype=torch.float32)
    z = (source_signature - train_stats["sig_mean"]) / train_stats["sig_std"].clamp_min(1e-6)
    source_weights = torch.tensor(subject["weights"], dtype=torch.float32)
    return evaluate_record(
        subject=subject,
        z=z,
        source=source,
        target=target,
        source_weights=source_weights,
        train_stats=train_stats,
        classifier=classifier,
        random_controls=random_controls,
    )


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        summary = {
            "max_compile_max_abs_logit_difference": 0.0,
            "mean_compile_max_abs_logit_difference": 0.0,
            "mean_conflict_target_accuracy": 0.0,
            "mean_conflict_target_accuracy_improvement": 0.0,
            "mean_matched_minus_best_control_target_margin": 0.0,
            "mean_matched_target_margin": 0.0,
            "individual_all_gate_pass_count": 0,
            "individual_all_gate_pass_rate": 0.0,
            "n": 0,
            "pareto_undominated_count": 0,
            "pareto_undominated_rate": 0.0,
            "target_prediction_count": 0,
            "target_prediction_rate": 0.0,
        }
        for metric_name in SIGNATURE_ADVANTAGE_CONTROLS:
            summary[f"mean_matched_minus_{metric_name}_target_margin"] = 0.0
            summary[f"mean_{metric_name}_minus_matched_compatible_source_output_mse"] = 0.0
        return summary
    passed = sum(1 for record in records if record["individual_all_gates_passed"])
    target_pred = sum(1 for record in records if record["summary"]["target_prediction_pass"])
    pareto = sum(1 for record in records if record["summary"]["pareto_undominated"])
    summary = {
        "individual_all_gate_pass_count": int(passed),
        "individual_all_gate_pass_rate": float(passed / n),
        "mean_compile_max_abs_logit_difference": v15.v10.mean(
            record["matched"]["compile_max_abs_logit_diff"] for record in records
        ),
        "mean_conflict_target_accuracy": v15.v10.mean(
            record["matched"]["conflict_target_accuracy"] for record in records
        ),
        "mean_conflict_target_accuracy_improvement": v15.v10.mean(
            record["matched"]["conflict_target_accuracy_improvement"] for record in records
        ),
        "mean_matched_minus_best_control_target_margin": v15.v10.mean(
            record["summary"]["matched_minus_best_control_target_margin"] for record in records
        ),
        "mean_matched_target_margin": v15.v10.mean(
            record["matched"]["target_margin"] for record in records
        ),
        "max_compile_max_abs_logit_difference": max(
            record["matched"]["compile_max_abs_logit_diff"] for record in records
        ),
        "n": int(n),
        "pareto_undominated_count": int(pareto),
        "pareto_undominated_rate": float(pareto / n),
        "target_prediction_count": int(target_pred),
        "target_prediction_rate": float(target_pred / n),
    }
    for metric_name in SIGNATURE_ADVANTAGE_CONTROLS:
        summary[f"mean_matched_minus_{metric_name}_target_margin"] = v15.v10.mean(
            record["summary"][f"matched_minus_{metric_name}_target_margin"]
            for record in records
        )
        summary[f"mean_{metric_name}_minus_matched_compatible_source_output_mse"] = (
            v15.v10.mean(
                record["summary"][f"{metric_name}_minus_matched_compatible_source_output_mse"]
                for record in records
            )
        )
    return summary


def gate_failures(
    *,
    aggregate: Mapping[str, Any],
    by_direction: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    failures = []
    v15.v10.require_equal(
        failures,
        aggregate["n"],
        THRESHOLDS["expected_record_count"],
        "aggregate n",
    )
    v15.v10.require_at_least(
        failures,
        aggregate["target_prediction_rate"],
        THRESHOLDS["min_aggregate_target_prediction_rate"],
        "aggregate target prediction rate",
    )
    v15.v10.require_at_least(
        failures,
        aggregate["individual_all_gate_pass_rate"],
        THRESHOLDS["min_aggregate_individual_pass_rate"],
        "aggregate individual pass rate",
    )
    v15.v10.require_at_least(
        failures,
        aggregate["pareto_undominated_rate"],
        THRESHOLDS["min_aggregate_pareto_undominated_rate"],
        "aggregate Pareto-undominated rate",
    )
    v15.v10.require_at_least(
        failures,
        aggregate["mean_conflict_target_accuracy"],
        THRESHOLDS["min_aggregate_conflict_target_accuracy"],
        "aggregate conflict target accuracy",
    )
    v15.v10.require_at_least(
        failures,
        aggregate["mean_conflict_target_accuracy_improvement"],
        THRESHOLDS["min_aggregate_conflict_target_accuracy_improvement"],
        "aggregate conflict target accuracy improvement",
    )
    v15.v10.require_at_least(
        failures,
        aggregate["mean_matched_target_margin"],
        THRESHOLDS["min_mean_matched_target_margin"],
        "mean matched target margin",
    )
    if aggregate["max_compile_max_abs_logit_difference"] > THRESHOLDS["max_compile_logit_difference"]:
        failures.append(
            "max compile logit difference "
            f"{aggregate['max_compile_max_abs_logit_difference']:.6g} > "
            f"{THRESHOLDS['max_compile_logit_difference']:.6g}"
        )
    aggregate_advantage_specs = {
        "shuffled_signature": (
            "min_aggregate_shuffled_target_margin_advantage",
            "min_aggregate_shuffled_compatible_mse_advantage",
        ),
        "target_label": (
            "min_aggregate_target_label_target_margin_advantage",
            "min_aggregate_target_label_compatible_mse_advantage",
        ),
        "output_layer_no_signature": (
            "min_aggregate_no_signature_target_margin_advantage",
            "min_aggregate_no_signature_compatible_mse_advantage",
        ),
    }
    for metric_name, (margin_key, mse_key) in aggregate_advantage_specs.items():
        if not (
            aggregate[f"mean_matched_minus_{metric_name}_target_margin"]
            >= THRESHOLDS[margin_key]
            or aggregate[f"mean_{metric_name}_minus_matched_compatible_source_output_mse"]
            >= THRESHOLDS[mse_key]
        ):
            failures.append(f"aggregate {metric_name} advantage gate failed")
    for direction, summary in by_direction.items():
        v15.v10.require_equal(
            failures,
            summary["n"],
            THRESHOLDS["expected_per_direction_count"],
            f"{direction} n",
        )
        v15.v10.require_at_least(
            failures,
            summary["target_prediction_rate"],
            THRESHOLDS["min_direction_target_prediction_rate"],
            f"{direction} target prediction rate",
        )
        v15.v10.require_at_least(
            failures,
            summary["individual_all_gate_pass_rate"],
            THRESHOLDS["min_direction_individual_pass_rate"],
            f"{direction} individual pass rate",
        )
        v15.v10.require_at_least(
            failures,
            summary["pareto_undominated_rate"],
            THRESHOLDS["min_direction_pareto_undominated_rate"],
            f"{direction} Pareto-undominated rate",
        )
        v15.v10.require_at_least(
            failures,
            summary["mean_conflict_target_accuracy"],
            THRESHOLDS["min_direction_conflict_target_accuracy"],
            f"{direction} conflict target accuracy",
        )
        v15.v10.require_at_least(
            failures,
            summary["mean_matched_target_margin"],
            THRESHOLDS["min_direction_mean_target_margin"],
            f"{direction} target margin",
        )
    required_control_types = set(REQUIRED_NON_RANDOM_CONTROL_TYPES)
    for record in records:
        if len(record["controls"]) != THRESHOLDS["expected_controls_per_record"]:
            failures.append(f"{record['subject_id']} control count mismatch")
        if record["random_control_count"] != THRESHOLDS["random_controls_per_record"]:
            failures.append(f"{record['subject_id']} random control count mismatch")
        control_types = {control["control_type"] for control in record["controls"]}
        for control_type in required_control_types:
            if control_type not in control_types:
                failures.append(f"{record['subject_id']} missing {control_type}")
    return failures


def train_only_statistics_hash(stats: Mapping[str, Any]) -> str:
    return stable_hash_json({
        "aperture_grid": APERTURE_GRID,
        "alpha_grid": ALPHA_GRID,
        "beta_grid": BETA_GRID,
        "conceptor_ridge": CONCEPTOR_RIDGE,
        "multiprocessing_contract": multiprocessing_contract(),
        "probe_examples_hash": stats.get("probe_examples_hash"),
        "train_statistics_hash": stats.get("train_statistics_hash"),
    })


def sort_records_for_artifact(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            str(record["source_behavior"]),
            str(record["subject_id"]),
            str(record["target_behavior"]),
        ),
    )


def forbidden_combined_final_summary_keys(summary: Mapping[str, Any]) -> list[str]:
    return sorted(set(summary) - FINAL_COMBINED_SUMMARY_ALLOWED_KEYS)


def forbidden_final_redacted_keys(summary: Mapping[str, Any]) -> list[str]:
    forbidden = {f"top_level.{key}" for key in sorted(set(summary) - FINAL_REDACTED_ALLOWED_KEYS)}
    forbidden.update(recursive_forbidden_final_detail_keys(summary))
    return sorted(forbidden)


def recursive_forbidden_final_detail_keys(payload: Any, prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_str = str(key)
            path = f"{prefix}.{key_str}" if prefix else key_str
            if key_str in FORBIDDEN_FINAL_DETAIL_KEYS:
                found.add(path)
            found.update(recursive_forbidden_final_detail_keys(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.update(recursive_forbidden_final_detail_keys(value, f"{prefix}[{index}]"))
    return found


def build_v16_seed_preflight() -> dict[str, Any]:
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

    final_keys = forbidden_combined_final_summary_keys(
        combined_audit.get("pool_summaries", {}).get("final", {})
    )
    if final_keys:
        failures.append(
            "combined_audit.pool_summaries.final exposes forbidden keys: "
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
    if phase == "development":
        expected_hash = v15.v1.sha256_file(eval_path)
        observed_hash = summaries.get("development", {}).get("pool_file_sha256")
        if observed_hash != expected_hash:
            failures.append("development pool hash mismatch")
    train_hash = summaries.get("train", {}).get("pool_file_sha256")
    if train_hash != v15.v1.sha256_file(train_path):
        failures.append("train pool hash mismatch")
    if combined_audit.get("passed") is not True:
        failures.append("combined audit did not pass")
    return failures


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
    return parser.parse_args()


def generate_pools(args: argparse.Namespace, pool_dir: Path) -> dict[str, Any]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    seed_preflight = build_v16_seed_preflight()
    if seed_preflight["failures"]:
        result = {"failures": seed_preflight["failures"], "passed": False}
        (pool_dir / "combined_audit.json").write_text(
            json.dumps(result, indent=2, sort_keys=True)
        )
        return result

    suite = v15.build_suite(args.support_per_class, args.heldout_per_class)
    heldout_sequences = v15.build_heldout_sequences(suite)
    candidate_pools = v15.build_candidate_pools(heldout_sequences)
    candidate_pool_summary = v15.summarize_candidate_pools(candidate_pools)
    probe_examples = v15.build_digit_probe_examples(
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
        payload = v15.poolgen.generate_pool(
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
            v15.poolgen.redact_weights_and_signatures(payload)
        )
        pool_payloads[pool_name] = payload
        pool_path = pool_dir / f"{pool_name}_subjects.json"
        pool_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        summary = v15.poolgen.summarize_pool(payload)
        summary["pool_file_sha256"] = v15.v1.sha256_file(pool_path)
        summary["pool_redacted_payload_sha256"] = payload["pool_redacted_payload_sha256"]
        pool_summaries[pool_name] = summary

    final_redacted = v15.poolgen.build_final_redacted_summary(pool_payloads["final"])
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

    audit = v15.poolgen.build_combined_audit(
        pool_payloads=pool_payloads,
        pool_summaries=pool_summaries,
        seed_preflight=seed_preflight,
        suite=suite,
        probe_examples=probe_examples,
    )
    audit["claim_scope"] = SOURCE_AUDIT_SCOPE
    audit = v15.v10.redact_combined_audit(audit)
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
            "combined_audit.pool_summaries.final exposes forbidden keys: "
            + ", ".join(forbidden_final_summary)
        )
    audit["audit_payload_sha256"] = stable_hash_json(audit)
    (pool_dir / "combined_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    return {
        "combined_audit_path": v15.v1.rel(pool_dir / "combined_audit.json"),
        "final_redacted_audit_path": v15.v1.rel(pool_dir / "final_redacted_audit.json"),
        "passed": bool(audit["passed"]),
        "pool_summaries": audit["pool_summaries"],
        "seed_preflight": seed_preflight,
    }


def assert_no_forbidden_final_raw_paths(
    paths: Sequence[Path],
    *,
    allow_v16_final: bool,
) -> None:
    for path in paths:
        resolved = path.resolve()
        if resolved in {prior.resolve() for prior in PRIOR_FINAL_RAW_PATHS}:
            raise ValueError(f"prior sealed final raw path is forbidden: {path}")
        if resolved == V16_FINAL_RAW.resolve() and not allow_v16_final:
            raise ValueError(f"V16 final raw path is forbidden before authorization: {path}")


def train_and_evaluate(
    *,
    train_path: Path,
    eval_path: Path,
    combined_audit_path: Path,
    final_redacted_path: Path,
    output_dir: Path,
    phase: str,
    allow_eval_final_raw: bool,
) -> dict[str, Any]:
    assert_no_forbidden_final_raw_paths(
        [train_path, eval_path, combined_audit_path, final_redacted_path],
        allow_v16_final=allow_eval_final_raw,
    )
    train_payload = v15.v1.load_json(train_path)
    eval_payload = v15.v1.load_json(eval_path)
    combined_audit = v15.v1.load_json(combined_audit_path)
    final_redacted = v15.v1.load_json(final_redacted_path)
    contract_failures = validate_source_pool_contract(
        train_path=train_path,
        eval_path=eval_path,
        train_payload=train_payload,
        eval_payload=eval_payload,
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase=phase,
    )
    if contract_failures:
        raise ValueError("V16 source-pool contract validation failed: " + "; ".join(contract_failures))

    train_subjects = v15.v1.accepted_records(train_payload)
    eval_subjects = v15.v1.accepted_records(eval_payload)
    probe_examples = v15.build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    base_stats = v15.fit_v15_train_statistics(train_subjects)
    v16_stats = fit_v16_train_statistics(train_subjects, probe_examples=probe_examples)
    train_stats = {**base_stats, **v16_stats}
    train_stats["probe_examples"] = probe_examples
    train_stats["probe_examples_hash"] = stable_hash_json(probe_examples)
    classifier, classifier_summary = v15.v9.fit_primary_classifier(
        train_stats["z_train"],
        train_stats["y_train"],
    )
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    calibration_coefficients, calibration_summary = v15.v9.fit_contrastive_calibration(
        subjects=train_subjects,
        train_stats=train_stats,
        classifier=classifier,
    )
    train_stats["calibration_coefficients"] = calibration_coefficients
    stats_path = output_dir / "v16_signature_conceptor_output_layer_stats.pt"
    torch.save(
        {
            "method": EDITOR_METHOD,
            "multiprocessing_contract": multiprocessing_contract(),
            "probe_examples_hash": train_stats["probe_examples_hash"],
            "train_only_statistics_hash": train_only_statistics_hash(train_stats),
        },
        stats_path,
    )
    eval_result = evaluate_subjects(
        subjects=eval_subjects,
        train_stats=train_stats,
        classifier=classifier,
        random_controls=16,
    )
    failures = [*contract_failures, *eval_result["failures"]]
    result = {
        **eval_result,
        "calibration_summary": calibration_summary,
        "claim_scope": DEVELOPMENT_SCOPE if phase == "development" else FINAL_SCOPE,
        "classifier_summary": classifier_summary,
        "combined_audit_path": v15.v1.rel(combined_audit_path),
        "combined_audit_sha256": v15.v1.sha256_file(combined_audit_path),
        "dirty_worktree_caveat": True,
        "editor_method": EDITOR_METHOD,
        "eval_pool_path": v15.v1.rel(eval_path),
        "eval_pool_sha256": v15.v1.sha256_file(eval_path),
        "final_redacted_audit_path": v15.v1.rel(final_redacted_path),
        "final_redacted_audit_sha256": v15.v1.sha256_file(final_redacted_path),
        "final_redacted_summary": final_redacted,
        "forbidden_prior_final_raw_opened": False,
        "forbidden_v16_final_raw_opened_before_authorization": False,
        "implementation_sha256": v15.v1.sha256_file(SCRIPT_PATH),
        "limitations": (
            "Small-subject source-label-known, target-label-requested "
            "signature-conditioned conceptor output-layer evidence only; not "
            "full-network optimizer supremacy, source-label inference, "
            "source-free decoding, larger-model evidence, or broad MUAT proof."
        ),
        "multiprocessing_contract": multiprocessing_contract(),
        "phase": phase,
        "pool_overlap_counts": combined_audit.get("overlap_counts", {}),
        "preregistration_sha256": v15.v1.sha256_file(PREREG_PATH),
        "probe_examples_hash": train_stats["probe_examples_hash"],
        "source_pool_audit_passed": bool(combined_audit.get("passed", False)),
        "stats_path": v15.v1.rel(stats_path),
        "stats_sha256": v15.v1.sha256_file(stats_path),
        "thresholds": THRESHOLDS,
        "train_only_statistics_hash": train_only_statistics_hash(train_stats),
        "train_pool_path": v15.v1.rel(train_path),
        "train_pool_sha256": v15.v1.sha256_file(train_path),
    }
    result["failures"] = failures
    result["passed"] = not failures
    result["next_action"] = (
        "eligible_for_one_shot_final_eval_without_method_changes"
        if result["passed"]
        else "log_negative_development_result_do_not_open_final_raw"
    )
    return result


def run_development(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> dict[str, Any]:
    result = train_and_evaluate(
        train_path=pool_dir / "train_subjects.json",
        eval_path=pool_dir / "development_subjects.json",
        combined_audit_path=pool_dir / "combined_audit.json",
        final_redacted_path=pool_dir / "final_redacted_audit.json",
        output_dir=output_dir,
        phase="development",
        allow_eval_final_raw=False,
    )
    output_path = output_dir / "development_results.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_final(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> dict[str, Any]:
    raise NotImplementedError(
        "V16 final is not implemented until development passes and reviewer authorizes final"
    )


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
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("passed", False):
        sys.exit(1)


def multiprocessing_contract() -> dict[str, Any]:
    return {
        "max_workers": min(8, os.cpu_count() or 1),
        "start_method": "spawn",
        "torch_threads_per_worker": 1,
        "stable_record_sort_key": ["source_behavior", "subject_id", "target_behavior"],
        "worker_writes_result_files": False,
    }


if __name__ == "__main__":
    main()
