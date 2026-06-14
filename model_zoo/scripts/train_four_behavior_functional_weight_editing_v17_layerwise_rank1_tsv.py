"""V17 functional editing via signature-conditioned layerwise rank-1/TSV edits."""

from __future__ import annotations

import hashlib
import argparse
import concurrent.futures
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

import train_four_behavior_functional_weight_editing_v16_signature_conceptor_output_layer as v16


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v17_pools"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v17_layerwise_rank1_tsv"
)
PREREG_PATH = (
    REPO_ROOT
    / "docs"
    / "preregistrations"
    / "four_behavior_functional_weight_editing_v17_layerwise_rank1_tsv.md"
)

SOURCE_POOL_SCOPE = "four_behavior_functional_weight_editing_v17_source_pool"
SOURCE_AUDIT_SCOPE = "four_behavior_functional_weight_editing_v17_source_pool_construction"
FINAL_REDACTED_SCOPE = "redacted_final_functional_weight_editing_v17_source_pool_audit_surface_only"
DEVELOPMENT_SCOPE = "four_behavior_functional_weight_editing_v17_layerwise_rank1_tsv_development"
FINAL_SCOPE = "four_behavior_functional_weight_editing_v17_layerwise_rank1_tsv_final"
EDITOR_METHOD = "signature_conditioned_layerwise_rank1_tsv_v17"

PATTERNS = v16.PATTERNS
V17_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"
PRIOR_FINAL_RAW_PATHS = {*v16.PRIOR_FINAL_RAW_PATHS, v16.V16_FINAL_RAW}

POOL_CONFIGS = {
    "train": {
        "base_seed": 81300000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 82300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 83300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}
SEED_BEHAVIOR_STRIDE = v16.SEED_BEHAVIOR_STRIDE

SIGNATURE_TOP_K = 8
SIGNATURE_TEMPERATURE = 1.0
RANK_GRID = [1, 2, 4, 8]
TASK_SCALE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
ACTIVATION_SCALE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
LAYER_MASKS = ["hidden_only", "all_layers"]
RANK1_RIDGE = 1e-4
RANDOM_CONTROLS_PER_RECORD = 16
SVD_SIGN_EPS = 1e-8
PARETO_EPSILON = 1e-8

SUPPORT_OBJECTIVE_WEIGHTS = {
    "target_bce": 4.0,
    "conflict_bce": 2.0,
    "compatible_mse": 0.01,
    "source_l2": 0.0005,
}

THRESHOLDS = {
    "expected_record_count": 288,
    "expected_per_direction_count": 24,
    "expected_controls_per_record": 26,
    "random_controls_per_record": RANDOM_CONTROLS_PER_RECORD,
    "min_per_record_target_margin": 0.20,
    "min_per_record_conflict_target_accuracy": 0.65,
    "min_per_record_conflict_target_accuracy_improvement": 0.15,
    "min_per_record_control_target_margin_advantage": 0.02,
    "min_per_record_control_compatible_mse_advantage": 2.0,
    "min_aggregate_target_prediction_rate": 0.85,
    "min_aggregate_individual_pass_rate": 0.85,
    "min_aggregate_pareto_undominated_rate": 0.85,
    "min_mean_matched_target_margin": 0.25,
    "min_aggregate_conflict_target_accuracy": 0.75,
    "min_aggregate_conflict_target_accuracy_improvement": 0.30,
    "min_aggregate_shuffled_target_margin_advantage": 0.05,
    "min_aggregate_target_label_target_margin_advantage": 0.02,
    "min_aggregate_output_layer_no_signature_target_margin_advantage": 0.02,
    "min_direction_target_prediction_rate": 0.70,
    "min_direction_individual_pass_rate": 0.70,
    "min_direction_pareto_undominated_rate": 0.70,
    "min_direction_mean_target_margin": 0.15,
    "min_direction_conflict_target_accuracy": 0.60,
    "min_direction_shuffled_target_margin_advantage": 0.03,
    "min_direction_output_layer_no_signature_target_margin_advantage": 0.01,
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
FINAL_COMBINED_SUMMARY_ALLOWED_KEYS = {
    "accepted_counts_by_behavior",
    "pool_file_sha256",
    "pool_redacted_payload_sha256",
}

PROOF_CRITICAL_CONTROL_TYPES = {
    "target_label_layerwise_tsv_centroid",
    "source_signature_layerwise_tsv",
    "shuffled_signature_layerwise_tsv",
    "nearest_target_layerwise_tsv",
    "activation_rank1_only",
    "layerwise_tsv_only",
    "v14_flat_subspace_task_vector",
    "v16_output_layer_conceptor",
    "output_layer_no_signature_support_optimizer",
    *[
        f"random_norm_matched_layerwise_low_rank_delta:{index:02d}"
        for index in range(RANDOM_CONTROLS_PER_RECORD)
    ],
}


def stable_hash_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_to_hashable(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().reshape(-1).tolist()


def layer_component_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    offset = 0
    for layer_index in range(5):
        in_dim = 5 if layer_index == 0 else 8
        out_dim = 8
        weight_count = out_dim * in_dim
        specs.append({
            "active_in_hidden_only": True,
            "end": offset + weight_count,
            "kind": "weight",
            "layer": layer_index,
            "name": f"weight_{layer_index}",
            "shape": (out_dim, in_dim),
            "start": offset,
        })
        offset += weight_count
        specs.append({
            "active_in_hidden_only": True,
            "end": offset + out_dim,
            "kind": "bias",
            "layer": layer_index,
            "name": f"bias_{layer_index}",
            "shape": (out_dim,),
            "start": offset,
        })
        offset += out_dim
    specs.append({
        "active_in_hidden_only": False,
        "end": offset + 8,
        "kind": "weight",
        "layer": 5,
        "name": "weight_5",
        "shape": (1, 8),
        "start": offset,
    })
    offset += 8
    specs.append({
        "active_in_hidden_only": False,
        "end": offset + 1,
        "kind": "bias",
        "layer": 5,
        "name": "bias_5",
        "shape": (1,),
        "start": offset,
    })
    offset += 1
    if offset != 345:
        raise ValueError(f"unexpected flat subject size {offset}")
    return specs


LAYER_COMPONENT_SPECS = layer_component_specs()


def active_component_specs(layer_mask: str) -> list[dict[str, Any]]:
    if layer_mask == "all_layers":
        return list(LAYER_COMPONENT_SPECS)
    if layer_mask == "hidden_only":
        return [spec for spec in LAYER_COMPONENT_SPECS if spec["active_in_hidden_only"]]
    raise ValueError(f"unknown layer mask: {layer_mask}")


def component_from_flat(flat: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    return flat[int(spec["start"]):int(spec["end"])].reshape(tuple(spec["shape"]))


def set_component(flat: torch.Tensor, spec: Mapping[str, Any], value: torch.Tensor) -> None:
    flat[int(spec["start"]):int(spec["end"])] = value.reshape(-1)


def sign_canonicalize_basis_rows(basis: torch.Tensor) -> torch.Tensor:
    canonical = basis.detach().clone().to(dtype=torch.float32, device="cpu")
    for row_index in range(canonical.shape[0]):
        row = canonical[row_index]
        nonzero = torch.nonzero(row.abs() > SVD_SIGN_EPS, as_tuple=False)
        if nonzero.numel() and float(row[int(nonzero[0].item())].item()) < 0.0:
            canonical[row_index] = -row
    return canonical


def component_svd_basis(component_deltas: torch.Tensor, rank: int) -> dict[str, Any]:
    matrix = component_deltas.detach().to(dtype=torch.float32, device="cpu")
    mean_delta = matrix.mean(dim=0)
    centered = matrix - mean_delta
    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = sign_canonicalize_basis_rows(vh[: min(int(rank), vh.shape[0])])
    total_variance = singular_values.pow(2).sum().clamp_min(1e-12)
    explained = singular_values[: basis.shape[0]].pow(2).sum() / total_variance
    return {
        "basis": basis,
        "component_count": int(matrix.shape[0]),
        "explained_variance": float(explained.item()),
        "mean_delta": mean_delta,
        "rank": int(basis.shape[0]),
        "singular_values": singular_values.detach().cpu(),
    }


def project_component_delta(
    delta: torch.Tensor,
    basis_info: Mapping[str, Any],
    *,
    rank: int,
) -> torch.Tensor:
    mean_delta = basis_info["mean_delta"].to(dtype=delta.dtype)
    basis = basis_info["basis"][: int(rank)].to(dtype=delta.dtype)
    centered = delta.reshape(-1) - mean_delta
    return mean_delta + torch.matmul(torch.matmul(centered, basis.t()), basis)


def normalized_signature(record: Mapping[str, Any], train_stats: Mapping[str, Any]) -> torch.Tensor:
    signature = torch.tensor(record["signature"], dtype=torch.float32)
    return (signature - train_stats["sig_mean"]) / train_stats["sig_std"].clamp_min(1e-6)


def records_by_behavior(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped = {pattern: [] for pattern in PATTERNS}
    for record in records:
        grouped[v16.subject_behavior(record)].append(record)
    for values in grouped.values():
        values.sort(key=lambda item: str(item["subject_id"]))
    return grouped


def direction_key(source: str, target: str) -> str:
    return v16.v15.v1.vector_key(source, target)


def layerwise_component_bases_for_direction(
    *,
    source_records: Sequence[Mapping[str, Any]],
    target_records: Sequence[Mapping[str, Any]],
    source: str,
    target: str,
    rank: int,
) -> dict[str, Any]:
    component_deltas = {spec["name"]: [] for spec in LAYER_COMPONENT_SPECS}
    pair_ids = []
    for source_record in sorted(source_records, key=lambda item: str(item["subject_id"])):
        source_weights = torch.tensor(source_record["weights"], dtype=torch.float32)
        for target_record in sorted(target_records, key=lambda item: str(item["subject_id"])):
            raw_delta = v16.v15.v14.target_delta_for_record(
                source_weights=source_weights,
                target_record=target_record,
                source=source,
                target=target,
                subject_id=str(source_record["subject_id"]),
                alignment_mode="hungarian",
            )
            pair_ids.append([
                str(source_record["subject_id"]),
                str(target_record["subject_id"]),
            ])
            for spec in LAYER_COMPONENT_SPECS:
                component_deltas[spec["name"]].append(component_from_flat(raw_delta, spec).reshape(-1))
    pair_count = len(pair_ids)
    components = {}
    for spec in LAYER_COMPONENT_SPECS:
        matrix = torch.stack(component_deltas[spec["name"]]).to(dtype=torch.float32)
        max_rank = min(int(rank), matrix.shape[0], matrix.shape[1])
        components[spec["name"]] = {
            **component_svd_basis(matrix, rank=max_rank),
            "shape": tuple(spec["shape"]),
        }
    return {
        "components": components,
        "pair_count": int(pair_count),
        "pair_ids_hash": stable_hash_json(pair_ids),
        "rank": int(rank),
    }


def fit_v17_train_statistics(
    subjects: Sequence[Mapping[str, Any]],
    *,
    probe_examples: Sequence[Mapping[str, Any]],
    include_baseline_stats: bool = False,
) -> dict[str, Any]:
    ordered_subjects = sorted(
        subjects,
        key=lambda item: (v16.subject_behavior(item), str(item["subject_id"])),
    )
    signatures = torch.tensor([record["signature"] for record in ordered_subjects], dtype=torch.float32)
    sig_mean = signatures.mean(dim=0)
    sig_std = signatures.std(dim=0, unbiased=False).clamp_min(1e-6)
    grouped = records_by_behavior(ordered_subjects)
    layerwise_bases = {}
    for source in PATTERNS:
        for target in PATTERNS:
            if source == target:
                continue
            key = direction_key(source, target)
            layerwise_bases[key] = layerwise_component_bases_for_direction(
                source_records=grouped[source],
                target_records=grouped[target],
                source=source,
                target=target,
                rank=max(RANK_GRID),
            )
    hash_payload = {
        "layerwise_bases": {
            direction: {
                "components": {
                    name: {
                        "basis": tensor_to_hashable(component["basis"]),
                        "explained_variance": float(component["explained_variance"]),
                        "mean_delta": tensor_to_hashable(component["mean_delta"]),
                        "rank": int(component["rank"]),
                        "shape": list(component["shape"]),
                        "singular_values": tensor_to_hashable(component["singular_values"]),
                    }
                    for name, component in sorted(bases["components"].items())
                },
                "pair_count": int(bases["pair_count"]),
                "pair_ids_hash": bases["pair_ids_hash"],
                "rank": int(bases["rank"]),
            }
            for direction, bases in sorted(layerwise_bases.items())
        },
        "layer_component_names": [spec["name"] for spec in LAYER_COMPONENT_SPECS],
        "pair_counts": {
            key: value["pair_count"]
            for key, value in layerwise_bases.items()
        },
        "probe_examples_hash": stable_hash_json(probe_examples),
        "rank_grid": RANK_GRID,
        "sig_mean": tensor_to_hashable(sig_mean),
        "sig_std": tensor_to_hashable(sig_std),
        "signature_top_k": SIGNATURE_TOP_K,
        "signature_temperature": SIGNATURE_TEMPERATURE,
        "subject_ids": [str(record["subject_id"]) for record in ordered_subjects],
    }
    stats = {
        "layerwise_bases": layerwise_bases,
        "probe_examples": list(probe_examples),
        "probe_examples_hash": hash_payload["probe_examples_hash"],
        "sig_mean": sig_mean,
        "sig_std": sig_std,
        "train_by_behavior": grouped,
        "train_statistics_hash": stable_hash_json(hash_payload),
        "train_subjects": ordered_subjects,
    }
    if include_baseline_stats:
        v14_stats = v16.v15.v14.fit_v14_train_statistics(ordered_subjects)
        v16_stats = v16.fit_v16_train_statistics(ordered_subjects, probe_examples=probe_examples)
        stats["v14_baseline_train_stats"] = v14_stats
        stats["v16_baseline_train_stats"] = v16_stats
        stats["v14_baseline_train_statistics_hash"] = stable_hash_json({
            "edit_subspaces": {
                key: {
                    "basis": tensor_to_hashable(value["basis"]),
                    "explained_variance": float(value["explained_variance"]),
                    "mean_delta": tensor_to_hashable(value["mean_delta"]),
                    "mean_delta_norm": float(value["mean_delta_norm"]),
                    "rank": int(value["rank"]),
                    "singular_values": tensor_to_hashable(value["singular_values"]),
                }
                for key, value in sorted(v14_stats.get("edit_subspaces", {}).items())
            },
            "source": "v17_train_subjects_only",
            "sig_mean": tensor_to_hashable(v14_stats["sig_mean"]),
            "sig_std": tensor_to_hashable(v14_stats["sig_std"]),
            "subject_ids": [str(record["subject_id"]) for record in ordered_subjects],
        })
        stats["v16_baseline_train_statistics_hash"] = v16_stats["train_statistics_hash"]
    return stats


def project_full_delta_layerwise(
    delta: torch.Tensor,
    *,
    direction_bases: Mapping[str, Any],
    rank: int,
    layer_mask: str,
) -> torch.Tensor:
    projected = torch.zeros_like(delta, dtype=torch.float32)
    active_names = {spec["name"] for spec in active_component_specs(layer_mask)}
    for spec in LAYER_COMPONENT_SPECS:
        if spec["name"] not in active_names:
            continue
        component = component_from_flat(delta, spec).reshape(-1)
        basis_info = direction_bases["components"][spec["name"]]
        value = project_component_delta(component, basis_info, rank=min(int(rank), basis_info["rank"]))
        set_component(projected, spec, value.reshape(tuple(spec["shape"])))
    return projected


def train_target_records(train_stats: Mapping[str, Any], target: str) -> list[Mapping[str, Any]]:
    return sorted(
        train_stats["train_by_behavior"][target],
        key=lambda record: str(record["subject_id"]),
    )


def signature_weighted_aligned_delta(
    *,
    source_weights: torch.Tensor,
    subject_id: str,
    source: str,
    target: str,
    source_signature_norm: torch.Tensor,
    train_stats: Mapping[str, Any],
    top_k: int = SIGNATURE_TOP_K,
    signature_pool_behavior: str | None = None,
) -> dict[str, Any]:
    target_records = train_target_records(train_stats, signature_pool_behavior or target)
    weights_info = signature_topk_weights(
        target_records,
        source_signature_norm=source_signature_norm,
        train_stats=train_stats,
        top_k=top_k,
    )
    weighted_delta = torch.zeros_like(source_weights, dtype=torch.float32)
    raw_deltas = []
    aligned_target_weights = []
    for index, (_distance, _target_subject_id, target_record) in enumerate(weights_info["selected"]):
        raw_delta = v16.v15.v14.target_delta_for_record(
            source_weights=source_weights,
            target_record=target_record,
            source=source,
            target=target,
            subject_id=subject_id,
            alignment_mode="hungarian",
        ).to(dtype=torch.float32)
        weight = weights_info["weights"][index].to(dtype=torch.float32)
        weighted_delta = weighted_delta + weight * raw_delta
        raw_deltas.append(raw_delta)
        aligned_target_weights.append(source_weights + raw_delta)
    return {
        "aligned_target_weights": aligned_target_weights,
        "metadata": weights_info["metadata"],
        "raw_deltas": raw_deltas,
        "weights": weights_info["weights"],
        "weighted_delta": weighted_delta,
    }


def hidden_inputs_and_outputs_flat_batch(
    flat_weights: torch.Tensor,
    inputs: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    batch_size = int(flat_weights.shape[0])
    x = inputs.unsqueeze(0).expand(batch_size, -1, -1)
    layer_inputs = []
    layer_outputs = []
    offset = 0
    for out_dim, in_dim in [(8, 5), (8, 8), (8, 8), (8, 8), (8, 8)]:
        layer_inputs.append(x)
        size = out_dim * in_dim
        weight = flat_weights[:, offset:offset + size].view(batch_size, out_dim, in_dim)
        offset += size
        bias = flat_weights[:, offset:offset + out_dim]
        offset += out_dim
        x = torch.nn.functional.gelu(torch.bmm(x, weight.transpose(1, 2)) + bias.unsqueeze(1))
        layer_outputs.append(x)
    return layer_inputs, layer_outputs


def activation_rank1_delta(
    *,
    source_weights: torch.Tensor,
    aligned_target_weights: Sequence[torch.Tensor],
    signature_weights: torch.Tensor,
    probe_examples: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    probe_inputs = v16.probe_inputs_tensor(probe_examples)
    source_batch = source_weights.reshape(1, -1).to(dtype=torch.float32)
    source_inputs, source_outputs = hidden_inputs_and_outputs_flat_batch(source_batch, probe_inputs)
    target_batch = torch.stack([weights.to(dtype=torch.float32) for weights in aligned_target_weights])
    _target_inputs, target_outputs = hidden_inputs_and_outputs_flat_batch(target_batch, probe_inputs)
    delta = torch.zeros_like(source_weights, dtype=torch.float32)
    weights = signature_weights.to(dtype=torch.float32)
    for layer_index in range(5):
        source_output_mean = source_outputs[layer_index][0].mean(dim=0)
        target_output_means = target_outputs[layer_index].mean(dim=1)
        weighted_target_mean = torch.sum(weights.reshape(-1, 1) * target_output_means, dim=0)
        output_direction = weighted_target_mean - source_output_mean
        input_direction = source_inputs[layer_index][0].mean(dim=0)
        denom = input_direction.pow(2).sum().clamp_min(0.0) + float(RANK1_RIDGE)
        rank1 = torch.outer(output_direction, input_direction) / denom
        weight_spec = next(
            spec for spec in LAYER_COMPONENT_SPECS
            if spec["name"] == f"weight_{layer_index}"
        )
        bias_spec = next(
            spec for spec in LAYER_COMPONENT_SPECS
            if spec["name"] == f"bias_{layer_index}"
        )
        set_component(delta, weight_spec, rank1)
        set_component(delta, bias_spec, output_direction)
    return delta


def support_objective_for_weights(
    *,
    weights: torch.Tensor,
    source_weights: torch.Tensor,
    source: str,
    target: str,
) -> dict[str, float]:
    support = v16.v15.v14.prepare_support_tensors_with_source_logits(
        source_weights=source_weights,
        source=source,
        target=target,
    )
    with torch.no_grad():
        target_logits = v16.v15.v10.decoder_v1.subject_forward_flat_batch(
            weights.unsqueeze(0),
            support["target_inputs"],
        )[0]
        conflict_logits = v16.v15.v10.decoder_v1.subject_forward_flat_batch(
            weights.unsqueeze(0),
            support["conflict_inputs"],
        )[0]
        compatible_logits = v16.v15.v10.decoder_v1.subject_forward_flat_batch(
            weights.unsqueeze(0),
            support["compatible_inputs"],
        )[0]
        target_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            target_logits,
            support["target_labels"],
        )
        conflict_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            conflict_logits,
            support["conflict_target_labels"],
        )
        compatible_mse = torch.nn.functional.mse_loss(
            compatible_logits,
            support["compatible_source_logits"],
        )
        source_l2 = torch.nn.functional.mse_loss(weights, source_weights)
        objective = (
            SUPPORT_OBJECTIVE_WEIGHTS["target_bce"] * target_bce
            + SUPPORT_OBJECTIVE_WEIGHTS["conflict_bce"] * conflict_bce
            + SUPPORT_OBJECTIVE_WEIGHTS["compatible_mse"] * compatible_mse
            + SUPPORT_OBJECTIVE_WEIGHTS["source_l2"] * source_l2
        )
    return {
        "compatible_mse": float(compatible_mse.item()),
        "conflict_bce": float(conflict_bce.item()),
        "objective": float(objective.item()),
        "source_l2": float(source_l2.item()),
        "target_bce": float(target_bce.item()),
    }


def select_layerwise_rank1_tsv_edit(
    *,
    source_weights: torch.Tensor,
    subject_id: str,
    source: str,
    target: str,
    source_signature_norm: torch.Tensor,
    train_stats: Mapping[str, Any],
    task_scale_grid: Sequence[float] = TASK_SCALE_GRID,
    activation_scale_grid: Sequence[float] = ACTIVATION_SCALE_GRID,
    rank_grid: Sequence[int] = RANK_GRID,
    layer_masks: Sequence[str] = LAYER_MASKS,
    signature_top_k: int = SIGNATURE_TOP_K,
    signature_pool_behavior: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    weighted = signature_weighted_aligned_delta(
        source_weights=source_weights,
        subject_id=subject_id,
        source=source,
        target=target,
        source_signature_norm=source_signature_norm,
        train_stats=train_stats,
        top_k=signature_top_k,
        signature_pool_behavior=signature_pool_behavior,
    )
    direction_bases = train_stats["layerwise_bases"][direction_key(source, target)]
    activation_delta = activation_rank1_delta(
        source_weights=source_weights,
        aligned_target_weights=weighted["aligned_target_weights"],
        signature_weights=weighted["weights"],
        probe_examples=train_stats["probe_examples"],
    )
    candidates = []
    for layer_mask in layer_masks:
        for rank in rank_grid:
            projected_delta = project_full_delta_layerwise(
                weighted["weighted_delta"],
                direction_bases=direction_bases,
                rank=int(rank),
                layer_mask=layer_mask,
            )
            active_names = {spec["name"] for spec in active_component_specs(layer_mask)}
            masked_activation_delta = torch.zeros_like(activation_delta)
            for spec in LAYER_COMPONENT_SPECS:
                if spec["name"] in active_names and int(spec["layer"]) < 5:
                    set_component(
                        masked_activation_delta,
                        spec,
                        component_from_flat(activation_delta, spec),
                    )
            for task_scale in task_scale_grid:
                for activation_scale in activation_scale_grid:
                    candidate_delta = (
                        float(task_scale) * projected_delta
                        + float(activation_scale) * masked_activation_delta
                    )
                    candidate_weights = source_weights + candidate_delta
                    losses = support_objective_for_weights(
                        weights=candidate_weights,
                        source_weights=source_weights,
                        source=source,
                        target=target,
                    )
                    candidate_lexical_key = (
                        f"layer_mask={layer_mask}|rank={int(rank):02d}|"
                        f"task_scale={float(task_scale):.6f}|"
                        f"activation_scale={float(activation_scale):.6f}|"
                        f"candidate={len(candidates):06d}"
                    )
                    candidates.append({
                        **losses,
                        "activation_scale": float(activation_scale),
                        "candidate_lexical_key": candidate_lexical_key,
                        "delta_norm": float(candidate_delta.norm().item()),
                        "layer_mask": layer_mask,
                        "rank": int(rank),
                        "task_scale": float(task_scale),
                        "weights": candidate_weights,
                    })
    best = min(
        candidates,
        key=lambda item: (
            item["objective"],
            item["delta_norm"],
            item["rank"],
            item["task_scale"],
            item["activation_scale"],
            LAYER_MASKS.index(item["layer_mask"]),
            item["candidate_lexical_key"],
        ),
    )
    metadata = {
        "candidate_count": len(candidates),
        "selected_candidate_lexical_key": best["candidate_lexical_key"],
        "selected_activation_scale": float(best["activation_scale"]),
        "selected_delta_norm": float(best["delta_norm"]),
        "selected_layer_mask": best["layer_mask"],
        "selected_objective": float(best["objective"]),
        "selected_rank": int(best["rank"]),
        "selected_signature_targets": weighted["metadata"],
        "selected_signature_pool_behavior": signature_pool_behavior or target,
        "selected_signature_top_k": int(signature_top_k),
        "selected_task_scale": float(best["task_scale"]),
        "support_objective_weights": SUPPORT_OBJECTIVE_WEIGHTS,
    }
    return best["weights"].detach().clone(), metadata


ADVANTAGE_CONTROL_TYPES = {
    "shuffled_signature": "shuffled_signature_layerwise_tsv",
    "target_label": "target_label_layerwise_tsv_centroid",
    "output_layer_no_signature": "output_layer_no_signature_support_optimizer",
}


def control_record_from_weights(
    control_type: str,
    weights: torch.Tensor,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return v16.v15.v14.control_record(
        control_type,
        weights,
        source,
        target,
        source_weights,
        metadata=metadata,
    )


def single_control(controls: Sequence[Mapping[str, Any]], control_type: str) -> Mapping[str, Any]:
    matches = [control for control in controls if control["control_type"] == control_type]
    if len(matches) != 1:
        raise ValueError(f"expected one {control_type} control, found {len(matches)}")
    return matches[0]


def source_signature_for_control(
    train_stats: Mapping[str, Any],
    source: str,
) -> torch.Tensor:
    source_records = train_stats["train_by_behavior"][source]
    signatures = torch.stack([normalized_signature(record, train_stats) for record in source_records])
    return signatures.mean(dim=0)


def target_label_centroid_delta(
    *,
    source_weights: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    rank: int,
    layer_mask: str,
) -> torch.Tensor:
    direction_bases = train_stats["layerwise_bases"][direction_key(source, target)]
    full_delta = torch.zeros_like(source_weights, dtype=torch.float32)
    for spec in active_component_specs(layer_mask):
        basis_info = direction_bases["components"][spec["name"]]
        set_component(full_delta, spec, basis_info["mean_delta"].reshape(tuple(spec["shape"])))
    return project_full_delta_layerwise(
        full_delta,
        direction_bases=direction_bases,
        rank=rank,
        layer_mask=layer_mask,
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
    controls = [
        control_record_from_weights("no_edit", source_weights, source, target, source_weights)
    ]
    selected_rank = int(matched_metadata["selected_rank"])
    selected_layer_mask = str(matched_metadata["selected_layer_mask"])

    target_label_delta = target_label_centroid_delta(
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats,
        rank=selected_rank,
        layer_mask=selected_layer_mask,
    )
    controls.append(control_record_from_weights(
        "target_label_layerwise_tsv_centroid",
        source_weights + target_label_delta,
        source,
        target,
        source_weights,
        {"rank": selected_rank, "layer_mask": selected_layer_mask},
    ))

    source_signature = source_signature_for_control(train_stats, source)
    for control_type, signature, signature_pool_behavior in [
        ("source_signature_layerwise_tsv", source_signature, source),
        ("shuffled_signature_layerwise_tsv", shuffled_signature_norm, target),
    ]:
        weights, metadata = select_layerwise_rank1_tsv_edit(
            source_weights=source_weights,
            subject_id=str(subject["subject_id"]),
            source=source,
            target=target,
            source_signature_norm=signature,
            train_stats=train_stats,
            task_scale_grid=[matched_metadata["selected_task_scale"]],
            activation_scale_grid=[matched_metadata["selected_activation_scale"]],
            rank_grid=[selected_rank],
            layer_masks=[selected_layer_mask],
            signature_pool_behavior=signature_pool_behavior,
        )
        controls.append(control_record_from_weights(
            control_type,
            weights,
            source,
            target,
            source_weights,
            metadata,
        ))

    nearest_weights, nearest_meta = select_layerwise_rank1_tsv_edit(
        source_weights=source_weights,
        subject_id=str(subject["subject_id"]),
        source=source,
        target=target,
        source_signature_norm=source_signature_norm,
        train_stats=train_stats,
        task_scale_grid=[matched_metadata["selected_task_scale"]],
        activation_scale_grid=[matched_metadata["selected_activation_scale"]],
        rank_grid=[selected_rank],
        layer_masks=[selected_layer_mask],
        signature_top_k=1,
    )
    controls.append(control_record_from_weights(
        "nearest_target_layerwise_tsv",
        nearest_weights,
        source,
        target,
        source_weights,
        {**nearest_meta, "nearest_control_uses_top_k_selection": True},
    ))

    for control_type, task_grid, activation_grid in [
        ("activation_rank1_only", [0.0], [matched_metadata["selected_activation_scale"]]),
        ("layerwise_tsv_only", [matched_metadata["selected_task_scale"]], [0.0]),
    ]:
        weights, metadata = select_layerwise_rank1_tsv_edit(
            source_weights=source_weights,
            subject_id=str(subject["subject_id"]),
            source=source,
            target=target,
            source_signature_norm=source_signature_norm,
            train_stats=train_stats,
            task_scale_grid=task_grid,
            activation_scale_grid=activation_grid,
            rank_grid=[selected_rank],
            layer_masks=[selected_layer_mask],
        )
        controls.append(control_record_from_weights(control_type, weights, source, target, source_weights, metadata))

    if "v14_baseline_train_stats" not in train_stats:
        raise KeyError("v14_baseline_train_stats required for V17 proof-critical controls")
    if "v16_baseline_train_stats" not in train_stats:
        raise KeyError("v16_baseline_train_stats required for V17 proof-critical controls")
    v14_direction, _v14_unprojected, v14_meta = v16.v15.v14.signature_weighted_task_direction(
        subject_id=str(subject["subject_id"]),
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats["v14_baseline_train_stats"],
        signature_target_norm=source_signature_norm,
    )
    controls.append(v16.v15.v14.task_vector_control_record(
        control_type="v14_flat_subspace_task_vector",
        direction=v14_direction,
        source_weights=source_weights,
        source=source,
        target=target,
        metadata={
            **v14_meta,
            "baseline_train_statistics_hash": train_stats.get("v14_baseline_train_statistics_hash"),
            "source": "recomputed_from_v17_train_subjects_only",
        },
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
        {
            **v16_meta,
            "baseline_train_statistics_hash": train_stats.get("v16_baseline_train_statistics_hash"),
            "source": "recomputed_from_v17_train_subjects_only",
        },
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

    matched_delta_norm = (matched_weights - source_weights).norm()
    for index in range(int(random_controls)):
        delta, metadata = random_layerwise_low_rank_delta(
            source_weights=source_weights,
            matched_delta_norm=matched_delta_norm,
            subject_id=str(subject["subject_id"]),
            source=source,
            target=target,
            index=index,
            rank=selected_rank,
            layer_mask=selected_layer_mask,
        )
        control_type = f"random_norm_matched_layerwise_low_rank_delta:{index:02d}"
        controls.append(control_record_from_weights(
            control_type,
            source_weights + delta,
            source,
            target,
            source_weights,
            metadata,
        ))
    return controls


def individual_passed(matched: Mapping[str, Any]) -> bool:
    margin_advantage_passes = []
    for metric_name in ADVANTAGE_CONTROL_TYPES:
        margin_advantage_passes.append(
            matched[f"matched_minus_{metric_name}_target_margin"]
            >= THRESHOLDS["min_per_record_control_target_margin_advantage"]
        )
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
    shuffled_signature_norm: torch.Tensor | None = None,
    train_stats: Mapping[str, Any],
    random_controls: int = RANDOM_CONTROLS_PER_RECORD,
) -> dict[str, Any]:
    matched_weights, matched_metadata = select_layerwise_rank1_tsv_edit(
        source_weights=source_weights,
        subject_id=str(subject["subject_id"]),
        source=source,
        target=target,
        source_signature_norm=source_signature_norm,
        train_stats=train_stats,
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
        shuffled_signature_norm=(
            source_signature_norm if shuffled_signature_norm is None else shuffled_signature_norm
        ),
        matched_weights=matched_weights,
        matched_metadata=matched_metadata,
        train_stats=train_stats,
        random_controls=random_controls,
    )
    gating_controls = [
        control for control in controls if control["control_type"] in PROOF_CRITICAL_CONTROL_TYPES
    ]
    pareto_dominators = [
        control for control in gating_controls if pareto_dominates(control, matched)
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
            if control["control_type"].startswith("random_norm_matched_layerwise_low_rank_delta")
        ),
        "source_behavior": source,
        "subject_id": str(subject["subject_id"]),
        "summary": summary,
        "target_behavior": target,
    }


def sort_records_for_artifact(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            str(record["source_behavior"]),
            str(record["subject_id"]),
            str(record["target_behavior"]),
        ),
    )


def mean(values: Sequence[float] | Any) -> float:
    materialized = [float(value) for value in values]
    return float(sum(materialized) / len(materialized)) if materialized else 0.0


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        summary = {
            "individual_all_gate_pass_count": 0,
            "individual_all_gate_pass_rate": 0.0,
            "mean_conflict_target_accuracy": 0.0,
            "mean_conflict_target_accuracy_improvement": 0.0,
            "mean_matched_minus_best_control_target_margin": 0.0,
            "mean_matched_target_margin": 0.0,
            "n": 0,
            "pareto_undominated_count": 0,
            "pareto_undominated_rate": 0.0,
            "target_prediction_count": 0,
            "target_prediction_rate": 0.0,
        }
        for metric_name in ADVANTAGE_CONTROL_TYPES:
            summary[f"mean_matched_minus_{metric_name}_target_margin"] = 0.0
            summary[f"mean_{metric_name}_minus_matched_compatible_source_output_mse"] = 0.0
        return summary
    passed = sum(1 for record in records if record["individual_all_gates_passed"])
    target_pred = sum(1 for record in records if record["summary"]["target_prediction_pass"])
    pareto = sum(1 for record in records if record["summary"]["pareto_undominated"])
    summary = {
        "individual_all_gate_pass_count": int(passed),
        "individual_all_gate_pass_rate": float(passed / n),
        "mean_conflict_target_accuracy": mean(
            record["matched"]["conflict_target_accuracy"] for record in records
        ),
        "mean_conflict_target_accuracy_improvement": mean(
            record["matched"]["conflict_target_accuracy_improvement"] for record in records
        ),
        "mean_matched_minus_best_control_target_margin": mean(
            record["summary"]["matched_minus_best_control_target_margin"] for record in records
        ),
        "mean_matched_target_margin": mean(record["matched"]["target_margin"] for record in records),
        "n": int(n),
        "pareto_undominated_count": int(pareto),
        "pareto_undominated_rate": float(pareto / n),
        "target_prediction_count": int(target_pred),
        "target_prediction_rate": float(target_pred / n),
    }
    for metric_name in ADVANTAGE_CONTROL_TYPES:
        summary[f"mean_matched_minus_{metric_name}_target_margin"] = mean(
            record["summary"][f"matched_minus_{metric_name}_target_margin"]
            for record in records
        )
        summary[f"mean_{metric_name}_minus_matched_compatible_source_output_mse"] = mean(
            record["summary"][f"{metric_name}_minus_matched_compatible_source_output_mse"]
            for record in records
        )
    return summary


def require_at_least(failures: list[str], observed: float, expected: float, label: str) -> None:
    if float(observed) < float(expected):
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
    require_equal(failures, aggregate["n"], THRESHOLDS["expected_record_count"], "aggregate n")
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
        THRESHOLDS["min_aggregate_pareto_undominated_rate"],
        "aggregate Pareto-undominated rate",
    )
    require_at_least(
        failures,
        aggregate["mean_matched_target_margin"],
        THRESHOLDS["min_mean_matched_target_margin"],
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
    for metric_name, threshold_key in [
        ("shuffled_signature", "min_aggregate_shuffled_target_margin_advantage"),
        ("target_label", "min_aggregate_target_label_target_margin_advantage"),
        ("output_layer_no_signature", "min_aggregate_output_layer_no_signature_target_margin_advantage"),
    ]:
        require_at_least(
            failures,
            aggregate[f"mean_matched_minus_{metric_name}_target_margin"],
            THRESHOLDS[threshold_key],
            f"aggregate {metric_name} target margin advantage",
        )
    for direction, summary in by_direction.items():
        require_equal(
            failures,
            summary["n"],
            THRESHOLDS["expected_per_direction_count"],
            f"{direction} n",
        )
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
            THRESHOLDS["min_direction_pareto_undominated_rate"],
            f"{direction} Pareto-undominated rate",
        )
        require_at_least(
            failures,
            summary["mean_matched_target_margin"],
            THRESHOLDS["min_direction_mean_target_margin"],
            f"{direction} target margin",
        )
        require_at_least(
            failures,
            summary["mean_conflict_target_accuracy"],
            THRESHOLDS["min_direction_conflict_target_accuracy"],
            f"{direction} conflict target accuracy",
        )
        require_at_least(
            failures,
            summary["mean_matched_minus_shuffled_signature_target_margin"],
            THRESHOLDS["min_direction_shuffled_target_margin_advantage"],
            f"{direction} shuffled target margin advantage",
        )
        require_at_least(
            failures,
            summary["mean_matched_minus_output_layer_no_signature_target_margin"],
            THRESHOLDS["min_direction_output_layer_no_signature_target_margin_advantage"],
            f"{direction} output-layer no-signature target margin advantage",
        )
    for record in records:
        if len(record["controls"]) != THRESHOLDS["expected_controls_per_record"]:
            failures.append(f"{record['subject_id']} control count mismatch")
        if record["random_control_count"] != THRESHOLDS["random_controls_per_record"]:
            failures.append(f"{record['subject_id']} random control count mismatch")
        control_types = {control["control_type"] for control in record["controls"]}
        missing = sorted(PROOF_CRITICAL_CONTROL_TYPES - control_types)
        for control_type in missing:
            failures.append(f"{record['subject_id']} missing {control_type}")
    return failures


_WORKER_TRAIN_STATS: Mapping[str, Any] | None = None
_WORKER_RANDOM_CONTROLS = RANDOM_CONTROLS_PER_RECORD
_WORKER_RECORD_EVALUATOR: Any = None


def evaluate_record_from_job(
    job: Mapping[str, Any],
    *,
    train_stats: Mapping[str, Any],
    random_controls: int,
) -> dict[str, Any]:
    subject = job["subject"]
    source = str(job["source"])
    target = str(job["target"])
    source_weights = torch.tensor(subject["weights"], dtype=torch.float32)
    source_signature_norm = normalized_signature(subject, train_stats)
    shuffled_signature_norm = job.get("shuffled_signature_norm")
    if shuffled_signature_norm is not None:
        shuffled_signature_norm = torch.tensor(shuffled_signature_norm, dtype=torch.float32)
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


def _init_eval_worker(
    train_stats: Mapping[str, Any],
    random_controls: int,
    record_evaluator: Any,
) -> None:
    global _WORKER_TRAIN_STATS, _WORKER_RANDOM_CONTROLS, _WORKER_RECORD_EVALUATOR
    torch.set_num_threads(1)
    _WORKER_TRAIN_STATS = train_stats
    _WORKER_RANDOM_CONTROLS = int(random_controls)
    _WORKER_RECORD_EVALUATOR = record_evaluator


def _evaluate_record_worker(job: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_TRAIN_STATS is None or _WORKER_RECORD_EVALUATOR is None:
        raise RuntimeError("worker not initialized")
    return _WORKER_RECORD_EVALUATOR(
        job,
        train_stats=_WORKER_TRAIN_STATS,
        random_controls=_WORKER_RANDOM_CONTROLS,
    )


def evaluate_subjects(
    *,
    subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    random_controls: int = RANDOM_CONTROLS_PER_RECORD,
    parallel: bool = True,
    max_workers: int | None = None,
    record_evaluator: Any = None,
) -> dict[str, Any]:
    if record_evaluator is None:
        record_evaluator = evaluate_record_from_job
    jobs = []
    for subject in subjects:
        source = v16.subject_behavior(subject)
        for target in PATTERNS:
            if target == source:
                continue
            jobs.append({"source": source, "subject": subject, "target": target})
    grouped_for_shuffle: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for job in jobs:
        grouped_for_shuffle.setdefault((str(job["source"]), str(job["target"])), []).append(job)
    if grouped_for_shuffle:
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
                    normalized_signature(next_job["subject"], train_stats)
                )
        for job in jobs:
            key = (
                str(job["source"]),
                str(job["subject"]["subject_id"]),
                str(job["target"]),
            )
            job["shuffled_signature_norm"] = shuffled_signatures[key]
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
        records = [
            record_evaluator(job, train_stats=train_stats, random_controls=random_controls)
            for job in jobs
        ]
    records = sort_records_for_artifact(records)
    aggregate = summarize_records(records)
    by_direction = {
        direction_key(source, target): summarize_records([
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
        "records": records,
    }


def signature_topk_weights(
    target_records: Sequence[Mapping[str, Any]],
    *,
    source_signature_norm: torch.Tensor,
    train_stats: Mapping[str, Any],
    top_k: int = SIGNATURE_TOP_K,
    temperature: float = SIGNATURE_TEMPERATURE,
) -> dict[str, Any]:
    scored = []
    for record in target_records:
        z_target = normalized_signature(record, train_stats)
        distance = torch.mean((z_target - source_signature_norm) ** 2).item()
        scored.append((float(distance), str(record["subject_id"]), record))
    scored.sort(key=lambda item: (item[0], item[1]))
    selected = scored[: min(int(top_k), len(scored))]
    distances = torch.tensor([item[0] for item in selected], dtype=torch.float32)
    weights = torch.softmax(-distances / float(temperature), dim=0)
    return {
        "selected": selected,
        "weights": weights,
        "metadata": [
            {
                "rank_order": int(index),
                "signature_distance": float(distance),
                "subject_id_hash": hashlib.sha256(subject_id.encode("utf-8")).hexdigest(),
                "weight": float(weights[index].item()),
            }
            for index, (distance, subject_id, _record) in enumerate(selected)
        ],
    }


def pareto_dominates(control: Mapping[str, float], matched: Mapping[str, float]) -> bool:
    target_weak = float(control["target_margin"]) >= float(matched["target_margin"]) - PARETO_EPSILON
    mse_weak = (
        float(control["compatible_source_output_mse"])
        <= float(matched["compatible_source_output_mse"]) + PARETO_EPSILON
    )
    target_strict = float(control["target_margin"]) > float(matched["target_margin"]) + PARETO_EPSILON
    mse_strict = (
        float(control["compatible_source_output_mse"])
        < float(matched["compatible_source_output_mse"]) - PARETO_EPSILON
    )
    return bool(target_weak and mse_weak and (target_strict or mse_strict))


def random_layerwise_low_rank_delta(
    *,
    source_weights: torch.Tensor,
    matched_delta_norm: torch.Tensor,
    subject_id: str,
    source: str,
    target: str,
    index: int,
    rank: int,
    layer_mask: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    seed_hash = stable_hash_json([
        subject_id,
        source,
        target,
        int(index),
        "functional_weight_editing_v17_random_layerwise_low_rank",
    ])
    seed = int(seed_hash[:16], 16) % (2**31)
    generator = torch.Generator().manual_seed(seed)
    delta = torch.zeros_like(source_weights, dtype=torch.float32)
    active_names = {spec["name"] for spec in active_component_specs(layer_mask)}
    for spec in LAYER_COMPONENT_SPECS:
        if spec["name"] not in active_names:
            continue
        shape = tuple(spec["shape"])
        if spec["kind"] == "weight":
            out_dim, in_dim = int(shape[0]), int(shape[1])
            effective_rank = min(int(rank), out_dim, in_dim)
            left = torch.randn((out_dim, effective_rank), generator=generator, dtype=torch.float32)
            right = torch.randn((effective_rank, in_dim), generator=generator, dtype=torch.float32)
            value = left @ right / math.sqrt(max(1, effective_rank * in_dim))
        else:
            value = torch.randn(shape, generator=generator, dtype=torch.float32)
        set_component(delta, spec, value)
    matched_norm = matched_delta_norm.detach().to(dtype=torch.float32).clamp_min(0.0)
    raw_norm = delta.norm()
    if float(matched_norm.item()) < 1e-12:
        delta.zero_()
        zero_norm = True
    else:
        delta = delta / raw_norm.clamp_min(1e-12) * matched_norm
        zero_norm = False
    return delta, {
        "layer_mask": layer_mask,
        "matched_delta_norm": float(matched_norm.item()),
        "random_seed": int(seed),
        "rank": int(rank),
        "raw_delta_norm": float(raw_norm.item()),
        "zero_norm_matched_delta": bool(zero_norm),
    }


def forbidden_final_redacted_keys(payload: Mapping[str, Any]) -> list[str]:
    failures = []
    top_keys = set(payload.keys())
    for key in sorted(top_keys - FINAL_REDACTED_ALLOWED_TOP_LEVEL_KEYS):
        failures.append(f"top_level.{key}")
    summary = payload.get("summary", {})
    if isinstance(summary, Mapping):
        for key in sorted(set(summary.keys()) - FINAL_REDACTED_ALLOWED_SUMMARY_KEYS):
            failures.append(f"summary.{key}")
    failures.extend(v16.forbidden_final_redacted_keys(payload))
    return sorted(set(failures))


def multiprocessing_contract(max_workers: int | None = None) -> dict[str, Any]:
    worker_count = min(8, os.cpu_count() or 1) if max_workers is None else int(max_workers)
    return {
        "max_workers": max(1, worker_count),
        "start_method": "spawn",
        "torch_threads_per_worker": 1,
        "stable_record_sort_key": ["source_behavior", "subject_id", "target_behavior"],
        "worker_writes_result_files": False,
    }


def forbidden_combined_final_summary_keys(summary: Mapping[str, Any]) -> list[str]:
    return sorted(set(summary) - FINAL_COMBINED_SUMMARY_ALLOWED_KEYS)


def build_v17_seed_preflight() -> dict[str, Any]:
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


def assert_no_forbidden_final_raw_paths(
    paths: Sequence[Path | str],
    *,
    allow_v17_final: bool,
) -> None:
    prior = {path.resolve() for path in PRIOR_FINAL_RAW_PATHS}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in prior:
            raise ValueError(f"prior sealed final raw path is forbidden: {path}")
        if path == V17_FINAL_RAW.resolve() and not allow_v17_final:
            raise ValueError(f"V17 final raw path is forbidden before authorization: {path}")


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
    if summaries.get("train", {}).get("pool_file_sha256") != v16.v15.v1.sha256_file(train_path):
        failures.append("train pool hash mismatch")
    if phase == "development":
        if summaries.get("development", {}).get("pool_file_sha256") != v16.v15.v1.sha256_file(eval_path):
            failures.append("development pool hash mismatch")
    if combined_audit.get("passed") is not True:
        failures.append("combined audit did not pass")
    assert_no_forbidden_final_raw_paths(
        [train_path, eval_path],
        allow_v17_final=(phase == "final"),
    )
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
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--summary-only-stdout", action="store_true", default=True)
    return parser.parse_args()


def generate_pools(args: argparse.Namespace, pool_dir: Path) -> dict[str, Any]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    seed_preflight = build_v17_seed_preflight()
    if seed_preflight["failures"]:
        result = {"failures": seed_preflight["failures"], "passed": False}
        (pool_dir / "combined_audit.json").write_text(
            json.dumps(result, indent=2, sort_keys=True)
        )
        return result

    suite = v16.v15.build_suite(args.support_per_class, args.heldout_per_class)
    heldout_sequences = v16.v15.build_heldout_sequences(suite)
    candidate_pools = v16.v15.build_candidate_pools(heldout_sequences)
    candidate_pool_summary = v16.v15.summarize_candidate_pools(candidate_pools)
    probe_examples = v16.v15.build_digit_probe_examples(
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
            "combined_audit.pool_summaries.final exposes forbidden keys: "
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
        "seed_preflight",
    ]
    summary = {key: result[key] for key in keys if key in result and key != "pool_summaries"}
    if redacted_pool_summaries is not None:
        summary["pool_summaries"] = redacted_pool_summaries
    return summary


def run_development(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> dict[str, Any]:
    train_path = pool_dir / "train_subjects.json"
    eval_path = pool_dir / "development_subjects.json"
    combined_audit_path = pool_dir / "combined_audit.json"
    final_redacted_path = pool_dir / "final_redacted_audit.json"
    assert_no_forbidden_final_raw_paths(
        [train_path, eval_path, combined_audit_path, final_redacted_path],
        allow_v17_final=False,
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
        raise ValueError("V17 source-pool contract validation failed: " + "; ".join(contract_failures))

    train_subjects = v16.v15.v1.accepted_records(train_payload)
    eval_subjects = v16.v15.v1.accepted_records(eval_payload)
    probe_examples = v16.v15.build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    train_stats = fit_v17_train_statistics(
        train_subjects,
        probe_examples=probe_examples,
        include_baseline_stats=True,
    )
    stats_path = output_dir / "v17_layerwise_rank1_tsv_stats.pt"
    torch.save(
        {
            "method": EDITOR_METHOD,
            "multiprocessing_contract": multiprocessing_contract(max_workers=args.max_workers),
            "probe_examples_hash": train_stats["probe_examples_hash"],
            "train_statistics_hash": train_stats["train_statistics_hash"],
            "v14_baseline_train_statistics_hash": train_stats.get("v14_baseline_train_statistics_hash"),
            "v16_baseline_train_statistics_hash": train_stats.get("v16_baseline_train_statistics_hash"),
        },
        stats_path,
    )
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
        "development_results_path": v16.v15.v1.rel(output_dir / "development_results.json"),
        "dirty_worktree_caveat": True,
        "editor_method": EDITOR_METHOD,
        "eval_pool_path": v16.v15.v1.rel(eval_path),
        "eval_pool_sha256": v16.v15.v1.sha256_file(eval_path),
        "final_redacted_audit_path": v16.v15.v1.rel(final_redacted_path),
        "final_redacted_audit_sha256": v16.v15.v1.sha256_file(final_redacted_path),
        "final_redacted_summary": final_redacted,
        "formal_prereg_sha256": v16.v15.v1.sha256_file(PREREG_PATH),
        "implementation_sha256": v16.v15.v1.sha256_file(SCRIPT_PATH),
        "limitations": (
            "Small-subject source-label-known, target-label-requested layerwise "
            "rank1/TSV functional editing evidence only; not source-label inference, "
            "source-free decoding, larger-model evidence, or broad MUAT proof."
        ),
        "phase": "development",
        "pool_overlap_counts": combined_audit.get("overlap_counts", {}),
        "probe_examples_hash": train_stats["probe_examples_hash"],
        "source_pool_audit_passed": bool(combined_audit.get("passed", False)),
        "stats_path": v16.v15.v1.rel(stats_path),
        "stats_sha256": v16.v15.v1.sha256_file(stats_path),
        "thresholds": THRESHOLDS,
        "train_pool_path": v16.v15.v1.rel(train_path),
        "train_pool_sha256": v16.v15.v1.sha256_file(train_path),
        "train_statistics_hash": train_stats["train_statistics_hash"],
        "v14_baseline_train_statistics_hash": train_stats.get("v14_baseline_train_statistics_hash"),
        "v16_baseline_train_statistics_hash": train_stats.get("v16_baseline_train_statistics_hash"),
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
        "V17 final is not implemented until development passes and reviewer authorizes final"
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
    print(json.dumps(summary_for_stdout(result), indent=2, sort_keys=True))
    if not result.get("passed", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
