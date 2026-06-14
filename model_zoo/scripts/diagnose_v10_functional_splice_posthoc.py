"""Posthoc V10 development-only diagnosis for functional weight splicing.

This script is not proof. It reads only V10 train/development artifacts and the
negative V10 development result to diagnose whether layer splicing is a more
promising functional bridge than the failed V10 ridge editor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ZOO_ROOT = REPO_ROOT / "model_zoo"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_ZOO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import train_four_behavior_functional_weight_editing_v10_v9_conditioned_delta as v10  # noqa: E402
import train_four_behavior_representation_steering as v1  # noqa: E402
from evaluate_four_behavior_source_generation_feasibility import PATTERNS  # noqa: E402


POOL_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v10_pools"
V10_RESULT_DIR = (
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v10_v9_conditioned_delta"
)
OUTPUT_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v10_posthoc_splice_diagnosis"
TRAIN_PATH = POOL_DIR / "train_subjects.json"
DEVELOPMENT_PATH = POOL_DIR / "development_subjects.json"
DEVELOPMENT_RESULT_PATH = V10_RESULT_DIR / "development_results.json"


LAYER_SLICES = {
    "hidden_1": (0, 48),
    "hidden_2": (48, 120),
    "hidden_3": (120, 192),
    "hidden_4": (192, 264),
    "hidden_5": (264, 336),
    "output": (336, 345),
}
SPLICE_CONFIGS = {
    "no_edit": [],
    "output_only": ["output"],
    "hidden_5_output": ["hidden_5", "output"],
    "hidden_4_5_output": ["hidden_4", "hidden_5", "output"],
    "hidden_3_5_output": ["hidden_3", "hidden_4", "hidden_5", "output"],
    "hidden_2_5_output": ["hidden_2", "hidden_3", "hidden_4", "hidden_5", "output"],
    "all_target_retrieval": ["hidden_1", "hidden_2", "hidden_3", "hidden_4", "hidden_5", "output"],
}
INTERPOLATION_ALPHAS = [0.25, 0.50, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def main() -> None:
    v10.assert_no_forbidden_final_raw_paths(
        [TRAIN_PATH, DEVELOPMENT_PATH, DEVELOPMENT_RESULT_PATH],
        allow_v10_final=False,
    )
    train_payload = v1.load_json(TRAIN_PATH)
    development_payload = v1.load_json(DEVELOPMENT_PATH)
    development_result = v1.load_json(DEVELOPMENT_RESULT_PATH)
    if development_result["passed"]:
        raise ValueError("posthoc diagnosis expects the V10 development result to be negative")
    train_by_id = {str(record["subject_id"]): record for record in v1.accepted_records(train_payload)}
    dev_by_id = {str(record["subject_id"]): record for record in v1.accepted_records(development_payload)}

    records = []
    for result_record in development_result["records"]:
        source_record = dev_by_id[str(result_record["subject_id"])]
        source = result_record["source_behavior"]
        target = result_record["target_behavior"]
        source_weights = torch.tensor(source_record["weights"], dtype=torch.float32)
        nearest_control = next(
            control
            for control in result_record["controls"]
            if control["control_type"] == "nearest_train_target_retrieval"
        )
        target_record = train_by_id[str(nearest_control["retrieved_subject_id"])]
        target_weights = torch.tensor(target_record["weights"], dtype=torch.float32)
        candidates = []
        for config_name, target_layers in SPLICE_CONFIGS.items():
            candidate_weights = splice_weights(
                source_weights=source_weights,
                target_weights=target_weights,
                target_layers=target_layers,
            )
            candidates.append(candidate_record(
                config_name=config_name,
                weights=candidate_weights,
                source=source,
                target=target,
                source_weights=source_weights,
                target_subject_id=str(target_record["subject_id"]),
            ))
        for alpha in INTERPOLATION_ALPHAS:
            candidate_weights = (1.0 - alpha) * source_weights + alpha * target_weights
            candidates.append(candidate_record(
                config_name=f"interpolate_alpha_{alpha:.2f}",
                weights=candidate_weights,
                source=source,
                target=target,
                source_weights=source_weights,
                target_subject_id=str(target_record["subject_id"]),
            ))
        nearest = next(item for item in candidates if item["config_name"] == "all_target_retrieval")
        for candidate in candidates:
            candidate["nearest_minus_candidate_source_output_mse"] = (
                nearest["source_output_mse"] - candidate["source_output_mse"]
            )
            candidate["candidate_minus_nearest_target_margin"] = (
                candidate["target_margin"] - nearest["target_margin"]
            )
            candidate["diagnostic_pass"] = (
                candidate["predicted_behavior"] == target
                and candidate["target_margin"] > 0.20
                and candidate["source_output_mse"] < nearest["source_output_mse"]
            )
        records.append({
            "candidates": candidates,
            "nearest_train_target_subject_id": str(target_record["subject_id"]),
            "source_behavior": source,
            "subject_id": str(source_record["subject_id"]),
            "target_behavior": target,
        })

    aggregate_by_config = {
        config_name: summarize_candidates([
            candidate
            for record in records
            for candidate in record["candidates"]
            if candidate["config_name"] == config_name
        ])
        for config_name in sorted({candidate["config_name"] for record in records for candidate in record["candidates"]})
    }
    by_direction = {
        v1.vector_key(source, target): {
            config_name: summarize_candidates([
                candidate
                for record in records
                if record["source_behavior"] == source and record["target_behavior"] == target
                for candidate in record["candidates"]
                if candidate["config_name"] == config_name
            ])
            for config_name in aggregate_by_config
        }
        for source in PATTERNS
        for target in PATTERNS
        if source != target
    }
    best_config = max(
        aggregate_by_config.items(),
        key=lambda item: (
            item[1]["diagnostic_pass_rate"],
            item[1]["target_prediction_rate"],
            item[1]["mean_nearest_minus_candidate_source_output_mse"],
            item[1]["mean_target_margin"],
        ),
    )
    result = {
        "aggregate_by_config": aggregate_by_config,
        "best_config_by_diagnostic_sort": best_config[0],
        "by_direction": by_direction,
        "claim_scope": "v10_development_posthoc_splice_diagnosis_not_proof",
        "development_result_sha256": v1.sha256_file(DEVELOPMENT_RESULT_PATH),
        "development_pool_sha256": v1.sha256_file(DEVELOPMENT_PATH),
        "development_status": "posthoc_development_only_diagnosis_after_preregistered_v10_failure",
        "final_access_status": "does_not_authorize_opening_or_evaluating_v10_final_raw",
        "final_raw_opened": False,
        "interpretation": (
            "development-only posthoc diagnosis; use only to motivate a fresh "
            "preregistered V11 design"
        ),
        "limitations": [
            "not proof",
            "uses V10 development results after observing V10 failure",
            "does not authorize opening V10 final raw",
        ],
        "n": len(records),
        "next_action": "use_only_to_motivate_fresh_preregistered_v11_design_do_not_open_v10_final_raw",
        "records": records,
        "source_artifact_sha256": v1.sha256_file(DEVELOPMENT_RESULT_PATH),
        "train_pool_sha256": v1.sha256_file(TRAIN_PATH),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "results.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "best_config_by_diagnostic_sort": result["best_config_by_diagnostic_sort"],
        "claim_scope": result["claim_scope"],
        "n": result["n"],
        "output_path": v1.rel(output_path),
        "top_configs": dict(sorted(
            aggregate_by_config.items(),
            key=lambda item: item[1]["diagnostic_pass_rate"],
            reverse=True,
        )[:5]),
    }, indent=2, sort_keys=True))


def splice_weights(
    *,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    target_layers: Sequence[str],
) -> torch.Tensor:
    spliced = source_weights.clone()
    for layer in target_layers:
        start, end = LAYER_SLICES[layer]
        spliced[start:end] = target_weights[start:end]
    return spliced


def candidate_record(
    *,
    config_name: str,
    weights: torch.Tensor,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    target_subject_id: str,
) -> Dict[str, Any]:
    metrics = v10.functional_metrics(weights, source, target, source_weights)
    return {
        "config_name": config_name,
        "predicted_behavior": metrics["predicted_behavior"],
        "source_margin": metrics["source_margin"],
        "source_output_mse": metrics["source_output_mse"],
        "target_margin": metrics["target_margin"],
        "target_prediction_pass": metrics["predicted_behavior"] == target,
        "target_subject_id": target_subject_id,
        "target_vs_source_margin": metrics["target_margin"] - metrics["source_margin"],
    }


def summarize_candidates(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(candidates)
    if n == 0:
        return {
            "diagnostic_pass_count": 0,
            "diagnostic_pass_rate": 0.0,
            "n": 0,
            "target_prediction_count": 0,
            "target_prediction_rate": 0.0,
        }
    return {
        "diagnostic_pass_count": sum(1 for candidate in candidates if candidate["diagnostic_pass"]),
        "diagnostic_pass_rate": mean(1.0 if candidate["diagnostic_pass"] else 0.0 for candidate in candidates),
        "mean_candidate_minus_nearest_target_margin": mean(
            candidate["candidate_minus_nearest_target_margin"] for candidate in candidates
        ),
        "mean_nearest_minus_candidate_source_output_mse": mean(
            candidate["nearest_minus_candidate_source_output_mse"] for candidate in candidates
        ),
        "mean_source_output_mse": mean(candidate["source_output_mse"] for candidate in candidates),
        "mean_target_margin": mean(candidate["target_margin"] for candidate in candidates),
        "mean_target_vs_source_margin": mean(candidate["target_vs_source_margin"] for candidate in candidates),
        "n": n,
        "target_prediction_count": sum(
            1 for candidate in candidates if candidate["target_prediction_pass"]
        ),
        "target_prediction_rate": mean(
            1.0 if candidate["target_prediction_pass"] else 0.0
            for candidate in candidates
        ),
    }


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
