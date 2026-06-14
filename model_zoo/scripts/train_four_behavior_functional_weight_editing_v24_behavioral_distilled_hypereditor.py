"""V24 behavioral-distilled hypereditor for four-behavior functional editing."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ZOO_ROOT = REPO_ROOT / "model_zoo"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_ZOO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import train_four_behavior_functional_weight_editing_v23_probe_routed_sparse_subspace_editor as v23  # noqa: E402


PATTERNS = v23.PATTERNS
SOURCE_WEIGHT_DIM = v23.SOURCE_WEIGHT_DIM
SIGNATURE_DIM = v23.SIGNATURE_DIM
ACTIVATION_DESCRIPTOR_DIM = 77
PAIR_DIM = len(PATTERNS) * (len(PATTERNS) - 1)
HYPEREDITOR_INPUT_DIM = (
    SOURCE_WEIGHT_DIM
    + SIGNATURE_DIM
    + ACTIVATION_DESCRIPTOR_DIM
    + len(PATTERNS)
    + len(PATTERNS)
    + PAIR_DIM
)
HYPEREDITOR_HIDDEN_DIMS = [768, 768, 384]
MAX_HYPEREDITOR_SCALE = 2.0

SEED_BEHAVIOR_STRIDE = 100000
POOL_CONFIGS = {
    "train": {
        "base_seed": 123400000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 124400000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 125400000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}

DEFAULT_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v24_pools"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runs"
    / "four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor"
)
PREREG_PATH = (
    REPO_ROOT
    / "docs"
    / "preregistrations"
    / "four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor.md"
)
PLAN_PATH = (
    REPO_ROOT
    / "docs"
    / "preregistrations"
    / "four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor_plan.md"
)
SCRIPT_PATH = Path(__file__).resolve()
HELPER_TEST_PATH = (
    REPO_ROOT
    / "model_zoo"
    / "scripts"
    / "test_four_behavior_functional_weight_editing_v24_helpers.py"
)

SOURCE_POOL_SCOPE = "four_behavior_functional_weight_editing_v24_source_pool"
SOURCE_AUDIT_SCOPE = "four_behavior_functional_weight_editing_v24_source_pool_construction"
FINAL_REDACTED_SCOPE = (
    "redacted_final_functional_weight_editing_v24_source_pool_audit_surface_only"
)
DEVELOPMENT_SCOPE = (
    "four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor_development"
)
FINAL_SCOPE = "four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor_final"
EDITOR_METHOD = "behavioral_distilled_hypereditor_v24"
V24_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"
PRIOR_FINAL_RAW_PATHS = {
    v23.V23_FINAL_RAW,
    v23.v22.V22_FINAL_RAW,
    v23.v21.V21_FINAL_RAW,
    v23.v17.V17_FINAL_RAW,
    v23.v16.V16_FINAL_RAW,
    v23.v16.v15.V15_FINAL_RAW,
}

RANDOM_CONTROLS_PER_RECORD = 20
EXPECTED_CONTROLS_PER_RECORD = 30
TEACHER_STEPS_GRID = [40, 80]
TEACHER_LR_GRID = [0.03, 0.01]
TEACHER_SOURCE_COMPAT_WEIGHT_GRID = [0.25, 0.5, 1.0]
HYPEREDITOR_STEPS_GRID = [800, 1600]
DELTA_MSE_WEIGHT_GRID = [0.25, 1.0]
BEHAVIOR_WEIGHT_GRID = [0.5, 1.0, 2.0]
COMPAT_WEIGHT_GRID = [0.25, 0.5, 1.0]
INNER_VALIDATION_EVALUATED_CONFIG_COUNT = 48
INNER_VALIDATION_RUNG_RECORD_BUDGETS = [24, 72, 156]
INNER_VALIDATION_RUNG_SURVIVORS = [12, 3, 1]
PLAN_SHA256 = "88e86b5b63d6e76bee0dc5480121ca73c642a5a088dd745bab7009f79bb03890"
FAILING_DEVELOPMENT_NEXT_ACTION = "log_negative_development_result_do_not_open_final_raw"
PASSING_DEVELOPMENT_NEXT_ACTION = "run_hash_bound_final_after_reviewer_authorization"
SOURCE_POOL_PROGRESS_LOG_FILENAME = "source_pool_progress.jsonl"
DEVELOPMENT_PROGRESS_LOG_FILENAME = "development_progress.jsonl"
INNER_VALIDATION_PROGRESS_LOG_FILENAME = "inner_validation_progress.jsonl"
STATS_ARTIFACT_FILENAME = "v24_behavioral_distilled_hypereditor_stats.pt"
REQUIRED_NAMED_CONTROL_TYPES = [
    "no_edit",
    "no_signature_ablation_behavioral_hypereditor_v24",
    "no_signature_trained_behavioral_hypereditor_v24",
    "source_behavior_target_ablation_behavioral_hypereditor_v24",
    "shuffled_signature_behavioral_hypereditor_v24",
    "nearest_train_target_signature_behavioral_hypereditor_v24",
    "teacher_oracle_support_optimizer_train_protocol_v24",
    "v21_behavioral_probe_residual_output_editor_recomputed",
    "v22_component_activation_rank1_editor_recomputed",
    "v23_probe_routed_sparse_subspace_editor_recomputed",
]
PROOF_CRITICAL_CONTROL_TYPES = [
    "no_signature_ablation_behavioral_hypereditor_v24",
    "no_signature_trained_behavioral_hypereditor_v24",
    "source_behavior_target_ablation_behavioral_hypereditor_v24",
    "shuffled_signature_behavioral_hypereditor_v24",
    "v21_behavioral_probe_residual_output_editor_recomputed",
    "v22_component_activation_rank1_editor_recomputed",
    "v23_probe_routed_sparse_subspace_editor_recomputed",
]
ADVANTAGE_CONTROL_TYPES = {
    "no_signature": "no_signature_ablation_behavioral_hypereditor_v24",
    "no_signature_trained": "no_signature_trained_behavioral_hypereditor_v24",
    "source_behavior_target_ablation": (
        "source_behavior_target_ablation_behavioral_hypereditor_v24"
    ),
    "shuffled_signature": "shuffled_signature_behavioral_hypereditor_v24",
    "v21": "v21_behavioral_probe_residual_output_editor_recomputed",
    "v22": "v22_component_activation_rank1_editor_recomputed",
    "v23": "v23_probe_routed_sparse_subspace_editor_recomputed",
}
THRESHOLDS = {
    "expected_record_count": 288,
    "expected_controls_per_record": EXPECTED_CONTROLS_PER_RECORD,
    "expected_random_controls_per_record": RANDOM_CONTROLS_PER_RECORD,
    "min_aggregate_target_prediction_rate": 0.85,
    "min_aggregate_individual_pass_rate": 0.85,
    "min_aggregate_pareto_rate": 0.85,
    "min_aggregate_target_margin": 0.25,
    "min_aggregate_best_control_target_margin_advantage": 0.02,
    "min_aggregate_no_signature_target_margin_advantage": 0.02,
    "min_aggregate_no_signature_trained_target_margin_advantage": 0.02,
    "min_aggregate_source_behavior_target_ablation_target_margin_advantage": 0.02,
    "min_aggregate_shuffled_signature_target_margin_advantage": 0.05,
    "min_aggregate_v21_target_margin_advantage": 0.02,
    "min_aggregate_v22_target_margin_advantage": 0.02,
    "min_aggregate_v23_target_margin_advantage": 0.02,
    "min_direction_target_prediction_rate": 0.65,
    "min_direction_pareto_rate": 0.75,
    "min_direction_target_margin": 0.15,
    "min_direction_control_target_margin_advantage": 0.02,
    "min_per_record_target_margin": 0.15,
    "min_per_record_control_target_margin_advantage": 0.02,
    "min_per_record_control_compatible_mse_advantage": -0.05,
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
FORBIDDEN_FINAL_DETAIL_KEYS = {
    *v23.v16.v15.FORBIDDEN_FINAL_DETAIL_KEYS,
    "records",
    "weights",
    "signature",
    "subject_id",
    "seed",
    "train_info",
    "support_margin",
    "heldout_margin",
}
RECURSIVE_FORBIDDEN_FINAL_DETAIL_KEY_TERMS = {
    "records",
    "weights",
    "signature",
    "subject_id",
    "seed",
    "train_info",
    "support_margin",
    "heldout_margin",
}

stable_hash_json = v23.stable_hash_json


@dataclass(frozen=True)
class TeacherEditConfig:
    steps: int
    lr: float
    l2_weight: float
    source_compat_weight: float
    grad_clip_norm: float = 5.0


@dataclass(frozen=True)
class HypereditorTrainingConfig:
    steps: int
    batch_size: int
    lr: float
    seed: int
    delta_mse_weight: float
    behavior_weight: float
    compat_weight: float
    l2_weight: float
    log_every: int = 50


class BehavioralDistilledHypereditor(nn.Module):
    """MLP that emits a full source-weight delta and bounded diagnostic scale."""

    def __init__(self, *, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            layers: list[nn.Module] = []
            input_dim = HYPEREDITOR_INPUT_DIM
            for hidden_dim in HYPEREDITOR_HIDDEN_DIMS:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(hidden_dim))
                input_dim = hidden_dim
            self.trunk = nn.Sequential(*layers)
            self.delta_head = nn.Linear(input_dim, SOURCE_WEIGHT_DIM)
            self.scale_head = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features.to(dtype=torch.float32))
        delta = self.delta_head(hidden)
        scale = MAX_HYPEREDITOR_SCALE * torch.sigmoid(self.scale_head(hidden).squeeze(-1))
        return delta, scale


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return v23.v16.v15.v1.sha256_file(path)


def write_development_results_artifact(
    *,
    output_path: Path,
    result: Mapping[str, Any],
    progress_log_path: Path,
    started_at_monotonic: float,
) -> dict[str, Any]:
    written = dict(result)
    written.pop("development_results_sha256", None)
    written["development_results_payload_sha256"] = stable_hash_json(written)
    write_json_atomic(output_path, written)
    record_progress_event(
        progress_log_path,
        event="development_results_written",
        started_at_monotonic=started_at_monotonic,
        extra={
            "development_results_file_sha256": sha256_file(output_path),
            "development_results_payload_sha256": written[
                "development_results_payload_sha256"
            ],
            "passed": bool(written["passed"]),
        },
    )
    return written


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def constants_payload() -> dict[str, Any]:
    return {
        "activation_descriptor_dim": ACTIVATION_DESCRIPTOR_DIM,
        "compat_weight_grid": COMPAT_WEIGHT_GRID,
        "delta_mse_weight_grid": DELTA_MSE_WEIGHT_GRID,
        "editor_method": EDITOR_METHOD,
        "expected_controls_per_record": EXPECTED_CONTROLS_PER_RECORD,
        "hypereditor_hidden_dims": HYPEREDITOR_HIDDEN_DIMS,
        "hypereditor_input_dim": HYPEREDITOR_INPUT_DIM,
        "hypereditor_steps_grid": HYPEREDITOR_STEPS_GRID,
        "inner_validation_evaluated_config_count": INNER_VALIDATION_EVALUATED_CONFIG_COUNT,
        "inner_validation_rung_record_budgets": INNER_VALIDATION_RUNG_RECORD_BUDGETS,
        "inner_validation_rung_survivors": INNER_VALIDATION_RUNG_SURVIVORS,
        "max_hypereditor_scale": MAX_HYPEREDITOR_SCALE,
        "plan_sha256": PLAN_SHA256,
        "proof_critical_control_types": PROOF_CRITICAL_CONTROL_TYPES,
        "random_controls_per_record": RANDOM_CONTROLS_PER_RECORD,
        "required_named_control_types": REQUIRED_NAMED_CONTROL_TYPES,
        "teacher_lr_grid": TEACHER_LR_GRID,
        "teacher_source_compat_weight_grid": TEACHER_SOURCE_COMPAT_WEIGHT_GRID,
        "teacher_steps_grid": TEACHER_STEPS_GRID,
    }


def record_progress_event(
    progress_log_path: Path,
    *,
    event: str,
    started_at_monotonic: float,
    extra: Mapping[str, Any] | None = None,
    now_monotonic: Any | None = None,
) -> None:
    monotonic = time.monotonic if now_monotonic is None else now_monotonic
    payload = {
        "elapsed_seconds": float(monotonic()) - float(started_at_monotonic),
        "event": event,
        "updated_at_unix": time.time(),
    }
    if extra:
        payload.update(dict(extra))
    append_jsonl(progress_log_path, payload)


def sort_records_for_artifact(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(record) for record in records],
        key=lambda item: (
            str(item.get("source_behavior")),
            str(item.get("subject_id")),
            str(item.get("target_behavior")),
        ),
    )


def required_control_failures(records: Sequence[Mapping[str, Any]]) -> list[str]:
    failures = []
    expected_named = set(REQUIRED_NAMED_CONTROL_TYPES)
    expected_random = {
        f"random_matched_norm_{index:02d}"
        for index in range(RANDOM_CONTROLS_PER_RECORD)
    }
    for index, record in enumerate(records):
        control_types = [
            str(control.get("control_type"))
            for control in record.get("controls", [])
        ]
        random_controls = [
            control_type
            for control_type in control_types
            if control_type.startswith("random_matched_norm_")
        ]
        named_controls = set(control_types) - set(random_controls)
        if named_controls != expected_named:
            failures.append(f"record_{index}_named_control_mismatch")
        if set(random_controls) != expected_random:
            failures.append(f"record_{index}_random_control_set_mismatch")
        if len(random_controls) != RANDOM_CONTROLS_PER_RECORD:
            failures.append(f"record_{index}_random_control_count_mismatch")
        if len(control_types) != EXPECTED_CONTROLS_PER_RECORD:
            failures.append(f"record_{index}_control_count_mismatch")
    return failures


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        empty = {
            "individual_all_gate_pass_count": 0,
            "individual_all_gate_pass_rate": 0.0,
            "n": 0,
            "pareto_undominated_count": 0,
            "pareto_undominated_rate": 0.0,
            "target_prediction_count": 0,
            "target_prediction_rate": 0.0,
        }
        for key in [
            "target_margin",
            "matched_minus_best_control_target_margin",
            *[
                f"matched_minus_{metric_name}_target_margin"
                for metric_name in ADVANTAGE_CONTROL_TYPES
            ],
            *[
                f"{metric_name}_minus_matched_compatible_source_output_mse"
                for metric_name in ADVANTAGE_CONTROL_TYPES
            ],
        ]:
            empty[f"mean_{key}"] = 0.0
        return empty
    matched = [record["matched"] for record in records]
    summary = {
        "individual_all_gate_pass_count": sum(
            1 for record in records if record.get("individual_all_gates_passed")
        ),
        "n": len(records),
        "pareto_undominated_count": sum(
            1 for item in matched if item.get("pareto_undominated") is True
        ),
        "target_prediction_count": sum(
            1 for item in matched if item.get("target_prediction_pass") is True
        ),
    }
    summary["individual_all_gate_pass_rate"] = (
        summary["individual_all_gate_pass_count"] / len(records)
    )
    summary["pareto_undominated_rate"] = (
        summary["pareto_undominated_count"] / len(records)
    )
    summary["target_prediction_rate"] = summary["target_prediction_count"] / len(records)
    summary["mean_target_margin"] = v23.mean(
        [float(item["target_margin"]) for item in matched]
    )
    summary["mean_matched_minus_best_control_target_margin"] = v23.mean([
        float(record["summary"]["matched_minus_best_control_target_margin"])
        for record in records
    ])
    for metric_name in ADVANTAGE_CONTROL_TYPES:
        summary[f"mean_matched_minus_{metric_name}_target_margin"] = v23.mean([
            float(item[f"matched_minus_{metric_name}_target_margin"])
            for item in matched
        ])
        summary[
            f"mean_{metric_name}_minus_matched_compatible_source_output_mse"
        ] = v23.mean([
            float(item[f"{metric_name}_minus_matched_compatible_source_output_mse"])
            for item in matched
        ])
    return summary


def require_at_least(
    failures: list[str],
    observed: float,
    expected: float,
    label: str,
) -> None:
    if float(observed) < float(expected):
        failures.append(f"{label} {float(observed):.6f} < {float(expected):.6f}")


def gate_failures(
    *,
    aggregate: Mapping[str, Any],
    by_direction: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    expected_record_count: int | None = None,
) -> list[str]:
    failures: list[str] = []
    expected_n = (
        int(THRESHOLDS["expected_record_count"])
        if expected_record_count is None
        else int(expected_record_count)
    )
    if int(aggregate.get("n", -1)) != expected_n:
        failures.append(
            f"record count {aggregate.get('n')!r} != {expected_n!r}"
        )
    bad_random = [
        str(record["subject_id"])
        for record in records
        if int(record.get("random_control_count", -1))
        != int(THRESHOLDS["expected_random_controls_per_record"])
    ]
    if bad_random:
        failures.append(f"records with wrong random control count: {bad_random[:5]}")
    bad_total = [
        str(record["subject_id"])
        for record in records
        if len(record.get("controls", []))
        != int(THRESHOLDS["expected_controls_per_record"])
    ]
    if bad_total:
        failures.append(f"records with wrong total control count: {bad_total[:5]}")
    require_at_least(
        failures,
        float(aggregate.get("target_prediction_rate", 0.0)),
        THRESHOLDS["min_aggregate_target_prediction_rate"],
        "aggregate target prediction rate",
    )
    require_at_least(
        failures,
        float(aggregate.get("individual_all_gate_pass_rate", 0.0)),
        THRESHOLDS["min_aggregate_individual_pass_rate"],
        "aggregate individual pass rate",
    )
    require_at_least(
        failures,
        float(aggregate.get("pareto_undominated_rate", 0.0)),
        THRESHOLDS["min_aggregate_pareto_rate"],
        "aggregate Pareto-undominated rate",
    )
    require_at_least(
        failures,
        float(aggregate.get("mean_target_margin", 0.0)),
        THRESHOLDS["min_aggregate_target_margin"],
        "mean matched target margin",
    )
    require_at_least(
        failures,
        float(aggregate.get("mean_matched_minus_best_control_target_margin", 0.0)),
        THRESHOLDS["min_aggregate_best_control_target_margin_advantage"],
        "aggregate best-control target margin advantage",
    )
    for metric_name in ADVANTAGE_CONTROL_TYPES:
        require_at_least(
            failures,
            float(aggregate.get(f"mean_matched_minus_{metric_name}_target_margin", 0.0)),
            THRESHOLDS[f"min_aggregate_{metric_name}_target_margin_advantage"],
            f"aggregate {metric_name} target margin advantage",
        )
    for direction, summary in sorted(by_direction.items()):
        require_at_least(
            failures,
            float(summary.get("target_prediction_rate", 0.0)),
            THRESHOLDS["min_direction_target_prediction_rate"],
            f"{direction} target prediction rate",
        )
        require_at_least(
            failures,
            float(summary.get("pareto_undominated_rate", 0.0)),
            THRESHOLDS["min_direction_pareto_rate"],
            f"{direction} Pareto-undominated rate",
        )
        require_at_least(
            failures,
            float(summary.get("mean_target_margin", 0.0)),
            THRESHOLDS["min_direction_target_margin"],
            f"{direction} target margin",
        )
        for metric_name in ADVANTAGE_CONTROL_TYPES:
            require_at_least(
                failures,
                float(summary.get(f"mean_matched_minus_{metric_name}_target_margin", 0.0)),
                THRESHOLDS["min_direction_control_target_margin_advantage"],
                f"{direction} {metric_name} target margin advantage",
            )
    failures.extend(required_control_failures(records))
    return {
        "failures": failures,
    }["failures"]


INNER_VALIDATION_FINITE_AGGREGATE_METRICS = (
    "target_prediction_rate",
    "individual_all_gate_pass_rate",
    "pareto_undominated_rate",
    "mean_target_margin",
    "mean_matched_minus_best_control_target_margin",
    *(
        f"mean_matched_minus_{metric_name}_target_margin"
        for metric_name in ADVANTAGE_CONTROL_TYPES
    ),
    *(
        f"mean_{metric_name}_minus_matched_compatible_source_output_mse"
        for metric_name in ADVANTAGE_CONTROL_TYPES
    ),
)


def inner_validation_contract_failures(
    *,
    aggregate: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    expected_record_count: int,
) -> list[str]:
    failures: list[str] = []
    if int(aggregate.get("n", -1)) != int(expected_record_count):
        failures.append(
            f"record count {aggregate.get('n')!r} != {int(expected_record_count)!r}"
        )
    bad_random = [
        str(record["subject_id"])
        for record in records
        if int(record.get("random_control_count", -1))
        != int(THRESHOLDS["expected_random_controls_per_record"])
    ]
    if bad_random:
        failures.append(f"records with wrong random control count: {bad_random[:5]}")
    bad_total = [
        str(record["subject_id"])
        for record in records
        if len(record.get("controls", []))
        != int(THRESHOLDS["expected_controls_per_record"])
    ]
    if bad_total:
        failures.append(f"records with wrong total control count: {bad_total[:5]}")
    failures.extend(required_control_failures(records))
    for metric_name in INNER_VALIDATION_FINITE_AGGREGATE_METRICS:
        value = aggregate.get(metric_name)
        if value is None:
            failures.append(f"aggregate metric {metric_name} missing")
            continue
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            failures.append(f"aggregate metric {metric_name} nonfinite")
    return failures


def inner_validation_candidate_invalidity(
    *,
    result: Mapping[str, Any],
    expected_record_count: int,
) -> dict[str, Any]:
    contract_failures = inner_validation_contract_failures(
        aggregate=result["aggregate"],
        records=result["records"],
        expected_record_count=expected_record_count,
    )
    proof_gate_failures = [
        str(failure)
        for failure in result.get("failures", [])
        if str(failure) not in set(contract_failures)
    ]
    return {
        "invalid": bool(contract_failures),
        "invalid_reasons": contract_failures,
        "proof_gate_failure_count": len(proof_gate_failures),
        "proof_gate_failures": proof_gate_failures,
    }


def evaluate_subjects(
    *,
    subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    record_evaluator: Any,
    parallel: bool = True,
    max_workers: int | None = None,
    progress_log_path: Path | None = None,
    progress_started_at_monotonic: float | None = None,
    expected_record_count: int | None = None,
    now_monotonic: Any | None = None,
) -> dict[str, Any]:
    jobs = []
    for subject in subjects:
        source = subject_behavior(subject)
        for target in PATTERNS:
            if target != source:
                jobs.append({"source": source, "subject": subject, "target": target})
    if "signature_mean" in train_stats and "activation_descriptor_mean" in train_stats:
        grouped_jobs: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for job in jobs:
            grouped_jobs.setdefault((str(job["source"]), str(job["target"])), []).append(job)
        for (_source, _target), group in sorted(grouped_jobs.items()):
            ordered = sorted(
                group,
                key=lambda item: (
                    stable_hash_json({
                        "scope": "v24_shuffled_signature_order",
                        "source": str(item["source"]),
                        "subject_id": str(item["subject"]["subject_id"]),
                        "target": str(item["target"]),
                    }),
                    str(item["subject"]["subject_id"]),
                ),
            )
            for index, job in enumerate(ordered):
                next_subject = ordered[(index + 1) % len(ordered)]["subject"]
                next_tensors = subject_feature_tensors(next_subject, train_stats)
                job["shuffled_signature_norm"] = v23.tensor_to_hashable(
                    next_tensors["signature_norm"]
                )
                job["shuffled_activation_descriptor_norm"] = v23.tensor_to_hashable(
                    next_tensors["activation_descriptor_norm"]
                )
    if progress_log_path is not None and progress_started_at_monotonic is not None:
        record_progress_event(
            progress_log_path,
            event="development_evaluation_jobs_queued",
            started_at_monotonic=progress_started_at_monotonic,
            extra={"record_count": len(jobs)},
            now_monotonic=now_monotonic,
        )
    records = []
    if parallel and len(jobs) > 1:
        context = mp.get_context("spawn")
        default_workers = max(1, min(8, (os.cpu_count() or 2) - 2))
        worker_count = max(1, min(max_workers or default_workers, len(jobs)))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(record_evaluator, job, train_stats=train_stats)
                for job in jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                records.append(future.result())
                if progress_log_path is not None and progress_started_at_monotonic is not None:
                    record_progress_event(
                        progress_log_path,
                        event="development_evaluation_record_completed",
                        started_at_monotonic=progress_started_at_monotonic,
                        extra={
                            "completed_count": len(records),
                            "record_count": len(jobs),
                        },
                        now_monotonic=now_monotonic,
                    )
    else:
        for job in jobs:
            records.append(record_evaluator(job, train_stats=train_stats))
            if progress_log_path is not None and progress_started_at_monotonic is not None:
                record_progress_event(
                    progress_log_path,
                    event="development_evaluation_record_completed",
                    started_at_monotonic=progress_started_at_monotonic,
                    extra={
                        "completed_count": len(records),
                        "record_count": len(jobs),
                    },
                    now_monotonic=now_monotonic,
                )
    records = sort_records_for_artifact(records)
    aggregate = summarize_records(records)
    by_direction = {
        v23.v17.direction_key(source, target): summarize_records([
            record for record in records
            if record["source_behavior"] == source and record["target_behavior"] == target
        ])
        for source in PATTERNS
        for target in PATTERNS
        if source != target
    }
    failures = gate_failures(
        aggregate=aggregate,
        by_direction=by_direction,
        records=records,
        expected_record_count=expected_record_count,
    )
    return {
        "aggregate": aggregate,
        "by_direction": by_direction,
        "failures": failures,
        "record_count": len(records),
        "records": records,
    }


def record_weights_tensor(record: Mapping[str, Any]) -> torch.Tensor:
    return v23.record_weights_tensor(record)


def subject_behavior(record: Mapping[str, Any]) -> str:
    return v23.v16.subject_behavior(record)


def safe_mean_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    zero_mask = std < 1e-6
    std = torch.where(zero_mask, torch.ones_like(std), std)
    return mean.to(dtype=torch.float32), std.to(dtype=torch.float32), int(zero_mask.sum().item())


def activation_descriptor_for_weights(
    weights: torch.Tensor,
    *,
    probe_examples: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    descriptor = v23.hidden_rank1_descriptor_for_weights(
        weights=weights.reshape(-1).to(dtype=torch.float32),
        probe_examples=probe_examples,
    )
    flat_parts = [
        item.reshape(-1).to(dtype=torch.float32)
        for item in [*descriptor["hbar"], *descriptor["xbar"]]
    ]
    flat = torch.cat(flat_parts)
    if int(flat.numel()) != ACTIVATION_DESCRIPTOR_DIM:
        raise ValueError(
            f"expected activation descriptor dim {ACTIVATION_DESCRIPTOR_DIM}, got {int(flat.numel())}"
        )
    return flat


def records_by_behavior(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped = {behavior: [] for behavior in PATTERNS}
    for record in records:
        grouped[subject_behavior(record)].append(record)
    return grouped


def tensor_normalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value.to(dtype=torch.float32).reshape(-1) - mean.reshape(-1)) / std.reshape(-1)


def subject_feature_tensors(
    subject: Mapping[str, Any],
    train_stats: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    subject_id = str(subject["subject_id"])
    weights = record_weights_tensor(subject)
    signature = torch.tensor(subject["signature"], dtype=torch.float32)
    descriptor = train_stats["activation_descriptor_by_subject"].get(subject_id)
    if descriptor is None:
        descriptor = activation_descriptor_for_weights(
            weights,
            probe_examples=train_stats["probe_examples"],
        )
    return {
        "activation_descriptor": descriptor.to(dtype=torch.float32),
        "activation_descriptor_norm": tensor_normalize(
            descriptor,
            train_stats["activation_descriptor_mean"],
            train_stats["activation_descriptor_std"],
        ),
        "signature": signature,
        "signature_norm": tensor_normalize(
            signature,
            train_stats["signature_mean"],
            train_stats["signature_std"],
        ),
        "weights": weights,
        "weights_norm": tensor_normalize(
            weights,
            train_stats["weights_mean"],
            train_stats["weights_std"],
        ),
    }


def inner_train_validation_split(subjects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = {behavior: [] for behavior in PATTERNS}
    for subject in subjects:
        behavior = subject_behavior(subject)
        if behavior in grouped:
            grouped[behavior].append(subject)
    inner_train_by_behavior = {}
    inner_validation_by_behavior = {}
    for behavior in PATTERNS:
        sorted_records = sorted(
            grouped[behavior],
            key=lambda item: (
                stable_hash_json({
                    "behavior": behavior,
                    "scope": "v24_inner_split",
                    "subject_id": str(item["subject_id"]),
                }),
                str(item["subject_id"]),
            ),
        )
        if len(sorted_records) < 64:
            raise ValueError(f"expected at least 64 {behavior} subjects")
        inner_train_by_behavior[behavior] = sorted_records[:51]
        inner_validation_by_behavior[behavior] = sorted_records[51:64]
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
    train_ids = [str(item["subject_id"]) for item in inner_train_subjects]
    validation_ids = [str(item["subject_id"]) for item in inner_validation_subjects]
    if set(train_ids) & set(validation_ids):
        raise ValueError("inner train and validation subjects overlap")
    return {
        "inner_split_hash": stable_hash_json({
            "inner_train_ids": train_ids,
            "inner_validation_ids": validation_ids,
            "scope": "v24_inner_split_hash",
        }),
        "inner_train_by_behavior": inner_train_by_behavior,
        "inner_train_subjects": inner_train_subjects,
        "inner_validation_by_behavior": inner_validation_by_behavior,
        "inner_validation_subjects": inner_validation_subjects,
    }


def iter_v24_configs() -> list[dict[str, Any]]:
    configs = []
    index = 0
    for teacher_steps in TEACHER_STEPS_GRID:
        for teacher_lr in TEACHER_LR_GRID:
            for teacher_source_compat_weight in TEACHER_SOURCE_COMPAT_WEIGHT_GRID:
                for hypereditor_steps in HYPEREDITOR_STEPS_GRID:
                    for delta_mse_weight in DELTA_MSE_WEIGHT_GRID:
                        for behavior_weight in BEHAVIOR_WEIGHT_GRID:
                            for compat_weight in COMPAT_WEIGHT_GRID:
                                config = {
                                    "behavior_weight": float(behavior_weight),
                                    "compat_weight": float(compat_weight),
                                    "config_index": index,
                                    "delta_mse_weight": float(delta_mse_weight),
                                    "hypereditor_steps": int(hypereditor_steps),
                                    "teacher_lr": float(teacher_lr),
                                    "teacher_source_compat_weight": float(
                                        teacher_source_compat_weight
                                    ),
                                    "teacher_steps": int(teacher_steps),
                                }
                                config["config_hash"] = stable_hash_json({
                                    "scope": "v24_behavioral_distilled_hypereditor_config",
                                    **config,
                                })
                                configs.append(config)
                                index += 1
    return configs


def v24_evaluated_config_subset(
    configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for config in configs:
        key = (
            int(config["teacher_steps"]),
            float(config["teacher_lr"]),
            int(config["hypereditor_steps"]),
            float(config["delta_mse_weight"]),
        )
        grouped.setdefault(key, []).append(config)
    selected = []
    for key in sorted(grouped):
        ordered = sorted(
            grouped[key],
            key=lambda config: (
                stable_hash_json({
                    "config_hash": str(config["config_hash"]),
                    "plan_sha256": PLAN_SHA256,
                    "scope": "v24_evaluated_config_subset",
                }),
                int(config["config_index"]),
            ),
        )
        selected.extend(dict(item) for item in ordered[:3])
    if len(selected) != INNER_VALIDATION_EVALUATED_CONFIG_COUNT:
        raise ValueError(f"expected 48 evaluated configs, got {len(selected)}")
    return selected


def v24_evaluated_config_subset_hash(configs: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash_json({
        "config_hashes": [str(config["config_hash"]) for config in configs],
        "plan_sha256": PLAN_SHA256,
        "scope": "v24_evaluated_config_subset_hash",
    })


def v24_full_config_grid_hash(configs: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash_json({
        "config_count": len(configs),
        "config_hashes": [str(config["config_hash"]) for config in configs],
        "plan_sha256": PLAN_SHA256,
        "scope": "v24_full_config_grid_hash",
    })


def behavior_one_hot(behavior: str) -> torch.Tensor:
    values = torch.zeros(len(PATTERNS), dtype=torch.float32)
    values[PATTERNS.index(behavior)] = 1.0
    return values


def pair_one_hot(source_behavior: str, target_behavior: str) -> torch.Tensor:
    values = torch.zeros(PAIR_DIM, dtype=torch.float32)
    pairs = [
        (source, target)
        for source in PATTERNS
        for target in PATTERNS
        if source != target
    ]
    values[pairs.index((source_behavior, target_behavior))] = 1.0
    return values


def build_hypereditor_features(
    *,
    source_weights_norm: torch.Tensor,
    source_signature_norm: torch.Tensor,
    activation_descriptor_norm: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
) -> torch.Tensor:
    return torch.cat([
        source_weights_norm.reshape(-1).to(dtype=torch.float32),
        source_signature_norm.reshape(-1).to(dtype=torch.float32),
        activation_descriptor_norm.reshape(-1).to(dtype=torch.float32),
        behavior_one_hot(source_behavior),
        behavior_one_hot(target_behavior),
        pair_one_hot(source_behavior, target_behavior),
    ])


def tensor_hash(value: torch.Tensor) -> Any:
    return v23.tensor_to_hashable(value.detach().to(dtype=torch.float32))


def model_state_hash(model: nn.Module) -> str:
    return v23.stable_hash_json({
        name: tensor_hash(value)
        for name, value in sorted(model.state_dict().items())
    })


def teacher_record_features(record: Mapping[str, Any]) -> torch.Tensor:
    return build_hypereditor_features(
        source_weights_norm=record["source_weights_norm"],
        source_signature_norm=record["source_signature_norm"],
        activation_descriptor_norm=record["activation_descriptor_norm"],
        source_behavior=str(record["source_behavior"]),
        target_behavior=str(record["target_behavior"]),
    )


def hypereditor_behavior_compat_losses(
    *,
    pred_delta: torch.Tensor,
    batch_records: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    behavior_losses = []
    compat_losses = []
    for row_index, record in enumerate(batch_records):
        if "source_weights" not in record:
            raise ValueError(
                "source_weights are required when behavior or compat loss is enabled"
            )
        source_weights = record["source_weights"].reshape(-1).to(dtype=torch.float32)
        source_behavior = str(record["source_behavior"])
        target_behavior = str(record["target_behavior"])
        support = v23.v16.v15.v14.prepare_support_tensors_with_source_logits(
            source_weights=source_weights.detach(),
            source=source_behavior,
            target=target_behavior,
        )
        edited_weights = source_weights + pred_delta[row_index].reshape(-1)
        target_logits = subject_logits_for_inputs(edited_weights, support["target_inputs"])
        compatible_logits = subject_logits_for_inputs(
            edited_weights,
            support["compatible_inputs"],
        )
        behavior_losses.append(F.binary_cross_entropy_with_logits(
            target_logits,
            support["target_labels"].to(dtype=torch.float32),
        ))
        compat_losses.append(F.mse_loss(
            compatible_logits,
            support["compatible_source_logits"].to(dtype=torch.float32),
        ))
    return torch.stack(behavior_losses).mean(), torch.stack(compat_losses).mean()


def train_hypereditor_on_teacher_records(
    teacher_records: Sequence[Mapping[str, Any]],
    *,
    config: HypereditorTrainingConfig,
    progress_log_path: Path | None = None,
    started_at_monotonic: float | None = None,
    now_monotonic: Any | None = None,
) -> dict[str, Any]:
    if not teacher_records:
        raise ValueError("teacher_records must not be empty")
    model = BehavioralDistilledHypereditor(seed=int(config.seed))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    features = torch.stack([teacher_record_features(record) for record in teacher_records])
    deltas = torch.stack([
        record["delta"].detach().reshape(-1).to(dtype=torch.float32)
        for record in teacher_records
    ])
    generator = torch.Generator().manual_seed(int(config.seed))
    last_loss = float("nan")
    last_behavior_loss = 0.0
    last_compat_loss = 0.0
    for step in range(1, int(config.steps) + 1):
        indices = torch.randint(
            low=0,
            high=len(teacher_records),
            size=(int(config.batch_size),),
            generator=generator,
        )
        batch_records = [teacher_records[int(index)] for index in indices.tolist()]
        batch_features = features[indices]
        batch_deltas = deltas[indices]
        optimizer.zero_grad(set_to_none=True)
        pred_delta, _scale = model(batch_features)
        delta_mse = F.mse_loss(pred_delta, batch_deltas)
        pred_delta_mse = pred_delta.pow(2).mean()
        behavior_loss = pred_delta.sum() * 0.0
        compat_loss = pred_delta.sum() * 0.0
        if float(config.behavior_weight) != 0.0 or float(config.compat_weight) != 0.0:
            behavior_loss, compat_loss = hypereditor_behavior_compat_losses(
                pred_delta=pred_delta,
                batch_records=batch_records,
            )
        loss = (
            float(config.delta_mse_weight) * delta_mse
            + float(config.behavior_weight) * behavior_loss
            + float(config.compat_weight) * compat_loss
            + float(config.l2_weight) * pred_delta_mse
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        last_loss = float(loss.detach().item())
        last_behavior_loss = float(behavior_loss.detach().item())
        last_compat_loss = float(compat_loss.detach().item())
        if (
            progress_log_path is not None
            and started_at_monotonic is not None
            and step % int(config.log_every) == 0
        ):
            record_progress_event(
                progress_log_path,
                event="hypereditor_training_step",
                started_at_monotonic=started_at_monotonic,
                extra={
                    "delta_mse": float(delta_mse.detach().item()),
                    "behavior_loss": last_behavior_loss,
                    "compat_loss": last_compat_loss,
                    "loss": last_loss,
                    "step": step,
                    "step_count": int(config.steps),
                },
                now_monotonic=now_monotonic,
            )
    return {
        "behavior_loss": last_behavior_loss,
        "compat_loss": last_compat_loss,
        "final_loss": last_loss,
        "model": model,
        "model_hash": model_state_hash(model),
        "step_count": int(config.steps),
    }


def subject_logits_for_inputs(weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return v23.v16.v15.v10.decoder_v1.subject_forward_flat_batch(
        weights.reshape(1, -1).to(dtype=torch.float32),
        inputs.to(dtype=torch.float32),
    )[0].reshape(-1)


def optimize_teacher_edit(
    source_subject: Mapping[str, Any],
    *,
    target_behavior: str,
    config: TeacherEditConfig,
) -> dict[str, Any]:
    source_behavior = v23.v16.subject_behavior(source_subject)
    source_weights = record_weights_tensor(source_subject)
    delta = torch.zeros_like(source_weights, requires_grad=True)
    optimizer = torch.optim.Adam(
        [delta],
        lr=float(config.lr),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    support = v23.v16.v15.v14.prepare_support_tensors_with_source_logits(
        source_weights=source_weights.detach().to(dtype=torch.float32),
        source=source_behavior,
        target=target_behavior,
    )
    loss_value = float("nan")
    target_bce_value = float("nan")
    source_compat_mse_value = float("nan")
    for _step in range(int(config.steps)):
        optimizer.zero_grad(set_to_none=True)
        edited_weights = source_weights + delta
        target_logits = subject_logits_for_inputs(edited_weights, support["target_inputs"])
        compatible_logits = subject_logits_for_inputs(
            edited_weights,
            support["compatible_inputs"],
        )
        target_bce = F.binary_cross_entropy_with_logits(
            target_logits,
            support["target_labels"].to(dtype=torch.float32),
        )
        source_compat_mse = F.mse_loss(
            compatible_logits,
            support["compatible_source_logits"].to(dtype=torch.float32),
        )
        delta_mse = delta.pow(2).mean()
        loss = (
            target_bce
            + float(config.source_compat_weight) * source_compat_mse
            + float(config.l2_weight) * delta_mse
        )
        if not torch.isfinite(loss):
            return {
                "delta": delta.detach().to(dtype=torch.float32),
                "invalid": True,
                "invalid_reasons": ["nonfinite_loss"],
                "loss": float("nan"),
                "source_behavior": source_behavior,
                "step_count": int(config.steps),
                "target_behavior": target_behavior,
            }
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta], max_norm=float(config.grad_clip_norm))
        optimizer.step()
        loss_value = float(loss.detach().item())
        target_bce_value = float(target_bce.detach().item())
        source_compat_mse_value = float(source_compat_mse.detach().item())
    return {
        "delta": delta.detach().to(dtype=torch.float32),
        "invalid": False,
        "invalid_reasons": [],
        "loss": loss_value,
        "source_behavior": source_behavior,
        "source_compat_mse": source_compat_mse_value,
        "step_count": int(config.steps),
        "target_bce": target_bce_value,
        "target_behavior": target_behavior,
    }


def build_teacher_records(
    subjects: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    progress_log_path: Path | None = None,
    started_at_monotonic: float | None = None,
    progress_event_prefix: str = "teacher",
) -> list[dict[str, Any]]:
    teacher_config = TeacherEditConfig(
        steps=int(config["teacher_steps"]),
        lr=float(config["teacher_lr"]),
        l2_weight=0.001,
        source_compat_weight=float(config["teacher_source_compat_weight"]),
    )
    jobs = [
        (subject, target)
        for subject in subjects
        for target in PATTERNS
        if target != subject_behavior(subject)
    ]
    records = []
    for index, (subject, target) in enumerate(jobs, start=1):
        source = subject_behavior(subject)
        tensors = subject_feature_tensors(subject, train_stats)
        teacher = optimize_teacher_edit(
            subject,
            target_behavior=target,
            config=teacher_config,
        )
        if not teacher.get("invalid"):
            records.append({
                "activation_descriptor_norm": tensors["activation_descriptor_norm"],
                "delta": teacher["delta"],
                "source_behavior": source,
                "source_signature_norm": tensors["signature_norm"],
                "source_weights": tensors["weights"],
                "source_weights_norm": tensors["weights_norm"],
                "target_behavior": target,
                "teacher_loss": float(teacher.get("loss", float("nan"))),
            })
        if (
            progress_log_path is not None
            and started_at_monotonic is not None
            and (index == len(jobs) or index % 24 == 0)
        ):
            record_progress_event(
                progress_log_path,
                event=f"{progress_event_prefix}_teacher_records_completed",
                started_at_monotonic=started_at_monotonic,
                extra={
                    "completed_count": index,
                    "record_count": len(jobs),
                    "valid_count": len(records),
                    "config_hash": str(config["config_hash"]),
                },
            )
    return records


def teacher_config_cache_key(config: Mapping[str, Any]) -> str:
    return stable_hash_json({
        "scope": "v24_teacher_record_cache_key",
        "teacher_lr": float(config["teacher_lr"]),
        "teacher_source_compat_weight": float(config["teacher_source_compat_weight"]),
        "teacher_steps": int(config["teacher_steps"]),
    })


def build_teacher_record_cache(
    *,
    configs: Sequence[Mapping[str, Any]],
    subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    progress_log_path: Path | None = None,
    started_at_monotonic: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    cache: dict[str, list[dict[str, Any]]] = {}
    representative_by_key = {}
    for config in configs:
        representative_by_key.setdefault(teacher_config_cache_key(config), config)
    for index, (key, config) in enumerate(sorted(representative_by_key.items()), start=1):
        if progress_log_path is not None and started_at_monotonic is not None:
            record_progress_event(
                progress_log_path,
                event="teacher_record_cache_start",
                started_at_monotonic=started_at_monotonic,
                extra={
                    "cache_index": index,
                    "cache_key": key,
                    "cache_key_count": len(representative_by_key),
                },
            )
        cache[key] = build_teacher_records(
            subjects,
            config=config,
            train_stats=train_stats,
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            progress_event_prefix="teacher_cache",
        )
        if progress_log_path is not None and started_at_monotonic is not None:
            record_progress_event(
                progress_log_path,
                event="teacher_record_cache_completed",
                started_at_monotonic=started_at_monotonic,
                extra={
                    "cache_index": index,
                    "cache_key": key,
                    "cache_key_count": len(representative_by_key),
                    "teacher_record_count": len(cache[key]),
                },
            )
    return cache


def fit_hypereditor_for_config(
    *,
    subjects: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    teacher_records: Sequence[Mapping[str, Any]] | None = None,
    no_signature_training: bool = False,
    progress_log_path: Path | None = None,
    started_at_monotonic: float | None = None,
    progress_event_prefix: str = "hypereditor",
) -> dict[str, Any]:
    if teacher_records is None:
        teacher_records = build_teacher_records(
            subjects,
            config=config,
            train_stats=train_stats,
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            progress_event_prefix=progress_event_prefix,
        )
    else:
        teacher_records = [dict(record) for record in teacher_records]
    if not teacher_records:
        raise ValueError("no valid teacher records for V24 hypereditor training")
    if no_signature_training:
        teacher_records = [
            {
                **record,
                "activation_descriptor_norm": torch.zeros(ACTIVATION_DESCRIPTOR_DIM),
                "source_signature_norm": torch.zeros(SIGNATURE_DIM),
            }
            for record in teacher_records
        ]
    training_config = HypereditorTrainingConfig(
        steps=int(config["hypereditor_steps"]),
        batch_size=64,
        lr=1e-3,
        seed=20260724 + int(config["config_index"]) + (1000000 if no_signature_training else 0),
        delta_mse_weight=float(config["delta_mse_weight"]),
        behavior_weight=float(config["behavior_weight"]),
        compat_weight=float(config["compat_weight"]),
        l2_weight=0.0001,
        log_every=50,
    )
    return train_hypereditor_on_teacher_records(
        teacher_records,
        config=training_config,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
    )


def hypereditor_delta_for_subject(
    *,
    model: BehavioralDistilledHypereditor,
    subject: Mapping[str, Any],
    target: str,
    train_stats: Mapping[str, Any],
    signature_norm: torch.Tensor | None = None,
    activation_descriptor_norm: torch.Tensor | None = None,
    target_behavior_override: str | None = None,
    zero_pair: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source = subject_behavior(subject)
    tensors = subject_feature_tensors(subject, train_stats)
    sig = tensors["signature_norm"] if signature_norm is None else signature_norm
    desc = (
        tensors["activation_descriptor_norm"]
        if activation_descriptor_norm is None
        else activation_descriptor_norm
    )
    target_for_features = target if target_behavior_override is None else target_behavior_override
    if zero_pair:
        pair = torch.zeros(PAIR_DIM, dtype=torch.float32)
    else:
        pair = pair_one_hot(source, target_for_features)
    features = torch.cat([
        tensors["weights_norm"],
        sig.reshape(-1).to(dtype=torch.float32),
        desc.reshape(-1).to(dtype=torch.float32),
        behavior_one_hot(source),
        behavior_one_hot(target_for_features),
        pair,
    ])
    with torch.no_grad():
        delta, scale = model(features.unsqueeze(0))
    return delta[0].detach().to(dtype=torch.float32), {
        "scale": float(scale[0].detach().item()),
    }


def metrics_for_delta(
    *,
    control_type: str,
    delta: torch.Tensor,
    source: str,
    source_weights: torch.Tensor,
    target: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    edited_weights = source_weights + delta.reshape(-1).to(dtype=torch.float32)
    return {
        **v23.v16.v15.v14.functional_metrics(
            edited_weights,
            source,
            target,
            source_weights,
        ),
        "control_type": control_type,
        "delta_norm": float(delta.reshape(-1).norm().item()),
        "editor": dict(metadata or {}),
        "weights": edited_weights,
    }


def random_matched_norm_delta(
    *,
    matched_delta: torch.Tensor,
    subject_id: str,
    source: str,
    target: str,
    index: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    seed_hash = stable_hash_json({
        "index": int(index),
        "scope": "v24_random_control",
        "source": source,
        "subject_id": subject_id,
        "target": target,
    })
    seed = int(seed_hash[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(SOURCE_WEIGHT_DIM, dtype=torch.float32, generator=generator)
    raw_norm = float(raw.norm().item())
    matched_norm = float(matched_delta.reshape(-1).norm().item())
    if matched_norm <= 1e-12 or raw_norm <= 1e-12:
        return torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.float32), {
            "matched_norm_zero": True,
            "random_seed": int(seed),
            "seed_hash": seed_hash,
        }
    return raw / raw_norm * matched_norm, {
        "matched_norm_zero": False,
        "random_seed": int(seed),
        "seed_hash": seed_hash,
    }


def nearest_train_target_feature_tensors(
    *,
    source_descriptor_norm: torch.Tensor,
    target: str,
    train_stats: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    candidates = train_stats["train_by_behavior"][target]
    best = min(
        candidates,
        key=lambda item: float((
            subject_feature_tensors(item, train_stats)["activation_descriptor_norm"]
            - source_descriptor_norm
        ).pow(2).sum().item()),
    )
    return subject_feature_tensors(best, train_stats)


def prior_editor_control(
    *,
    control_type: str,
    selector: Any,
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    subject: Mapping[str, Any],
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    prior_stats_key: str,
) -> dict[str, Any]:
    edited, metadata = selector(
        source_weights=source_weights,
        source_signature_norm=source_signature_norm,
        source=source,
        target=target,
        subject=subject,
        train_stats=train_stats[prior_stats_key],
    )
    return {
        **v23.v16.v15.v14.functional_metrics(edited, source, target, source_weights),
        "control_type": control_type,
        "delta_norm": float((edited - source_weights).norm().item()),
        "editor": {key: value for key, value in metadata.items() if not key.startswith("_")},
        "weights": edited,
    }


def single_control(controls: Sequence[Mapping[str, Any]], control_type: str) -> Mapping[str, Any]:
    matches = [control for control in controls if control["control_type"] == control_type]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {control_type}, found {len(matches)}")
    return matches[0]


def pareto_controls_for_record(controls: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        control for control in controls
        if control["control_type"] in PROOF_CRITICAL_CONTROL_TYPES
        or str(control["control_type"]).startswith("random_matched_norm_")
    ]


def individual_passed(matched: Mapping[str, Any]) -> bool:
    margin_passes = [
        matched[f"matched_minus_{metric_name}_target_margin"]
        >= THRESHOLDS["min_per_record_control_target_margin_advantage"]
        for metric_name in ADVANTAGE_CONTROL_TYPES
    ]
    return bool(
        matched["target_prediction_pass"]
        and matched["target_margin"] >= THRESHOLDS["min_per_record_target_margin"]
        and matched["pareto_undominated"]
        and matched["min_proof_critical_compatible_mse_advantage"]
        >= THRESHOLDS["min_per_record_control_compatible_mse_advantage"]
        and all(margin_passes)
    )


def evaluate_v24_record_from_job(
    job: Mapping[str, Any],
    *,
    train_stats: Mapping[str, Any],
) -> dict[str, Any]:
    subject = job["subject"]
    source = str(job["source"])
    target = str(job["target"])
    source_weights = record_weights_tensor(subject)
    tensors = subject_feature_tensors(subject, train_stats)
    model = train_stats["model"]
    matched_delta, matched_meta = hypereditor_delta_for_subject(
        model=model,
        subject=subject,
        target=target,
        train_stats=train_stats,
    )
    matched = metrics_for_delta(
        control_type=EDITOR_METHOD,
        delta=matched_delta,
        source=source,
        source_weights=source_weights,
        target=target,
        metadata=matched_meta,
    )
    controls = [
        metrics_for_delta(
            control_type="no_edit",
            delta=torch.zeros_like(matched_delta),
            source=source,
            source_weights=source_weights,
            target=target,
        )
    ]
    zero_sig = torch.zeros(SIGNATURE_DIM, dtype=torch.float32)
    zero_desc = torch.zeros(ACTIVATION_DESCRIPTOR_DIM, dtype=torch.float32)
    no_sig_delta, no_sig_meta = hypereditor_delta_for_subject(
        model=model,
        subject=subject,
        target=target,
        train_stats=train_stats,
        signature_norm=zero_sig,
        activation_descriptor_norm=zero_desc,
    )
    controls.append(metrics_for_delta(
        control_type="no_signature_ablation_behavioral_hypereditor_v24",
        delta=no_sig_delta,
        source=source,
        source_weights=source_weights,
        target=target,
        metadata=no_sig_meta,
    ))
    no_sig_trained_delta, no_sig_trained_meta = hypereditor_delta_for_subject(
        model=train_stats["no_signature_model"],
        subject=subject,
        target=target,
        train_stats=train_stats,
        signature_norm=zero_sig,
        activation_descriptor_norm=zero_desc,
    )
    controls.append(metrics_for_delta(
        control_type="no_signature_trained_behavioral_hypereditor_v24",
        delta=no_sig_trained_delta,
        source=source,
        source_weights=source_weights,
        target=target,
        metadata=no_sig_trained_meta,
    ))
    source_target_delta, source_target_meta = hypereditor_delta_for_subject(
        model=model,
        subject=subject,
        target=target,
        train_stats=train_stats,
        target_behavior_override=source,
        zero_pair=True,
    )
    controls.append(metrics_for_delta(
        control_type="source_behavior_target_ablation_behavioral_hypereditor_v24",
        delta=source_target_delta,
        source=source,
        source_weights=source_weights,
        target=target,
        metadata=source_target_meta,
    ))
    shuffled_sig = torch.tensor(job["shuffled_signature_norm"], dtype=torch.float32)
    shuffled_desc = torch.tensor(
        job["shuffled_activation_descriptor_norm"],
        dtype=torch.float32,
    )
    shuffled_delta, shuffled_meta = hypereditor_delta_for_subject(
        model=model,
        subject=subject,
        target=target,
        train_stats=train_stats,
        signature_norm=shuffled_sig,
        activation_descriptor_norm=shuffled_desc,
    )
    controls.append(metrics_for_delta(
        control_type="shuffled_signature_behavioral_hypereditor_v24",
        delta=shuffled_delta,
        source=source,
        source_weights=source_weights,
        target=target,
        metadata=shuffled_meta,
    ))
    nearest = nearest_train_target_feature_tensors(
        source_descriptor_norm=tensors["activation_descriptor_norm"],
        target=target,
        train_stats=train_stats,
    )
    nearest_delta, nearest_meta = hypereditor_delta_for_subject(
        model=model,
        subject=subject,
        target=target,
        train_stats=train_stats,
        signature_norm=nearest["signature_norm"],
        activation_descriptor_norm=nearest["activation_descriptor_norm"],
    )
    controls.append(metrics_for_delta(
        control_type="nearest_train_target_signature_behavioral_hypereditor_v24",
        delta=nearest_delta,
        source=source,
        source_weights=source_weights,
        target=target,
        metadata=nearest_meta,
    ))
    teacher = optimize_teacher_edit(
        subject,
        target_behavior=target,
        config=TeacherEditConfig(
            steps=int(train_stats["selected_config"]["teacher_steps"]),
            lr=float(train_stats["selected_config"]["teacher_lr"]),
            l2_weight=0.001,
            source_compat_weight=float(
                train_stats["selected_config"]["teacher_source_compat_weight"]
            ),
        ),
    )
    controls.append(metrics_for_delta(
        control_type="teacher_oracle_support_optimizer_train_protocol_v24",
        delta=teacher["delta"],
        source=source,
        source_weights=source_weights,
        target=target,
        metadata={"teacher_loss": float(teacher.get("loss", float("nan")))},
    ))
    controls.append(prior_editor_control(
        control_type="v21_behavioral_probe_residual_output_editor_recomputed",
        selector=v23.v21.select_behavioral_probe_residual_output_edit,
        source_weights=source_weights,
        source_signature_norm=tensors["signature_norm"],
        subject=subject,
        source=source,
        target=target,
        train_stats=train_stats,
        prior_stats_key="v21_train_stats",
    ))
    controls.append(prior_editor_control(
        control_type="v22_component_activation_rank1_editor_recomputed",
        selector=v23.v22.select_component_activation_rank1_edit,
        source_weights=source_weights,
        source_signature_norm=tensors["signature_norm"],
        subject=subject,
        source=source,
        target=target,
        train_stats=train_stats,
        prior_stats_key="v22_train_stats",
    ))
    controls.append(prior_editor_control(
        control_type="v23_probe_routed_sparse_subspace_editor_recomputed",
        selector=v23.select_probe_routed_sparse_subspace_edit,
        source_weights=source_weights,
        source_signature_norm=tensors["signature_norm"],
        subject=subject,
        source=source,
        target=target,
        train_stats=train_stats,
        prior_stats_key="v23_train_stats",
    ))
    for index in range(RANDOM_CONTROLS_PER_RECORD):
        random_delta, random_meta = random_matched_norm_delta(
            matched_delta=matched_delta,
            subject_id=str(subject["subject_id"]),
            source=source,
            target=target,
            index=index,
        )
        controls.append(metrics_for_delta(
            control_type=f"random_matched_norm_{index:02d}",
            delta=random_delta,
            source=source,
            source_weights=source_weights,
            target=target,
            metadata=random_meta,
        ))
    gating_controls = [
        control for control in controls
        if control["control_type"] in PROOF_CRITICAL_CONTROL_TYPES
    ]
    pareto_dominators = [
        control for control in pareto_controls_for_record(controls)
        if v23.v17.pareto_dominates(control, matched)
    ]
    best_target = max(gating_controls, key=lambda item: item["target_margin"])
    matched["pareto_dominator_count"] = len(pareto_dominators)
    matched["pareto_dominator_types"] = sorted({
        str(control["control_type"]) for control in pareto_dominators
    })
    matched["pareto_undominated"] = not pareto_dominators
    matched["target_prediction_pass"] = matched["predicted_behavior"] == target
    mse_advantages = [
        float(control["compatible_source_output_mse"])
        - float(matched["compatible_source_output_mse"])
        for control in gating_controls
    ]
    matched["min_proof_critical_compatible_mse_advantage"] = (
        min(mse_advantages) if mse_advantages else 0.0
    )
    for metric_name, control_type in ADVANTAGE_CONTROL_TYPES.items():
        control = single_control(controls, control_type)
        matched[f"matched_minus_{metric_name}_target_margin"] = (
            float(matched["target_margin"]) - float(control["target_margin"])
        )
        matched[f"{metric_name}_minus_matched_compatible_source_output_mse"] = (
            float(control["compatible_source_output_mse"])
            - float(matched["compatible_source_output_mse"])
        )
    matched["individual_all_gates_passed"] = individual_passed(matched)
    summary = {
        "best_control_target_margin": float(best_target["target_margin"]),
        "best_control_type": str(best_target["control_type"]),
        "matched_minus_best_control_target_margin": (
            float(matched["target_margin"]) - float(best_target["target_margin"])
        ),
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
        "controls": v23.v16.v15.v10.strip_control_weights(controls),
        "individual_all_gates_passed": matched["individual_all_gates_passed"],
        "matched": v23.v16.v15.v10.strip_weight(matched),
        "random_control_count": sum(
            1 for control in controls
            if str(control["control_type"]).startswith("random_matched_norm_")
        ),
        "source_behavior": source,
        "subject_id": str(subject["subject_id"]),
        "summary": summary,
        "target_behavior": target,
    }


def fit_v24_normalization_stats(
    subjects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    probe_examples = build_probe_examples()
    weights = []
    signatures = []
    descriptors = []
    descriptor_by_subject = {}
    for subject in subjects:
        weight = record_weights_tensor(subject)
        signature = torch.tensor(subject["signature"], dtype=torch.float32)
        descriptor = activation_descriptor_for_weights(
            weight,
            probe_examples=probe_examples,
        )
        weights.append(weight)
        signatures.append(signature)
        descriptors.append(descriptor)
        descriptor_by_subject[str(subject["subject_id"])] = descriptor
    weights_mean, weights_std, weights_zero_std = safe_mean_std(torch.stack(weights))
    signature_mean, signature_std, signature_zero_std = safe_mean_std(torch.stack(signatures))
    descriptor_mean, descriptor_std, descriptor_zero_std = safe_mean_std(torch.stack(descriptors))
    return {
        "activation_descriptor_by_subject": descriptor_by_subject,
        "activation_descriptor_mean": descriptor_mean,
        "activation_descriptor_std": descriptor_std,
        "activation_descriptor_zero_std_count": descriptor_zero_std,
        "probe_examples": probe_examples,
        "probe_examples_hash": stable_hash_json(probe_examples),
        "signature_mean": signature_mean,
        "signature_std": signature_std,
        "signature_zero_std_count": signature_zero_std,
        "weights_mean": weights_mean,
        "weights_std": weights_std,
        "weights_zero_std_count": weights_zero_std,
    }


def rank_inner_validation_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return sorted(
        candidates,
        key=lambda candidate: (
            bool(candidate.get("invalid")),
            -float(candidate.get("target_prediction_rate", float("-inf"))),
            -float(candidate.get("pareto_undominated_rate", float("-inf"))),
            -float(candidate.get("mean_matched_minus_best_control_target_margin", float("-inf"))),
            -float(candidate.get(
                "mean_matched_minus_shuffled_signature_target_margin",
                float("-inf"),
            )),
            -float(candidate.get("mean_target_margin", float("-inf"))),
            float(candidate.get("mean_compatible_source_mse", float("inf"))),
            str(candidate.get("config_hash")),
        ),
    )


def inner_validation_subjects_for_budget(
    split: Mapping[str, Any],
    record_budget: int,
) -> list[Mapping[str, Any]]:
    subjects_per_behavior = int(record_budget) // (len(PATTERNS) * (len(PATTERNS) - 1))
    if subjects_per_behavior < 1:
        raise ValueError("inner-validation budget too small")
    return [
        record
        for behavior in PATTERNS
        for record in split["inner_validation_by_behavior"][behavior][:subjects_per_behavior]
    ]


def inner_validation_candidate_for_config(
    *,
    config: Mapping[str, Any],
    train_subjects: Sequence[Mapping[str, Any]],
    validation_subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    teacher_records: Sequence[Mapping[str, Any]] | None = None,
    progress_log_path: Path | None = None,
    started_at_monotonic: float | None = None,
) -> dict[str, Any]:
    try:
        fitted = fit_hypereditor_for_config(
            subjects=train_subjects,
            config=config,
            train_stats=train_stats,
            teacher_records=teacher_records,
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            progress_event_prefix="inner_validation",
        )
        no_sig = fit_hypereditor_for_config(
            subjects=train_subjects,
            config=config,
            train_stats=train_stats,
            teacher_records=teacher_records,
            no_signature_training=True,
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            progress_event_prefix="inner_validation_no_signature",
        )
        candidate_stats = {
            **train_stats,
            "model": fitted["model"],
            "model_hash": fitted["model_hash"],
            "no_signature_model": no_sig["model"],
            "no_signature_model_hash": no_sig["model_hash"],
            "selected_config": dict(config),
        }
        result = evaluate_subjects(
            subjects=validation_subjects,
            train_stats=candidate_stats,
            record_evaluator=evaluate_v24_record_from_job,
            parallel=False,
            expected_record_count=len(validation_subjects) * (len(PATTERNS) - 1),
        )
        aggregate = result["aggregate"]
        invalidity = inner_validation_candidate_invalidity(
            result=result,
            expected_record_count=len(validation_subjects) * (len(PATTERNS) - 1),
        )
        return {
            **dict(config),
            **invalidity,
            "inner_validation_record_count": int(result["record_count"]),
            "mean_compatible_source_mse": float(
                v23.mean([
                    float(record["matched"]["compatible_source_output_mse"])
                    for record in result["records"]
                ])
            ) if result["records"] else float("inf"),
            "mean_matched_minus_best_control_target_margin": float(
                aggregate.get("mean_matched_minus_best_control_target_margin", float("-inf"))
            ),
            "mean_matched_minus_shuffled_signature_target_margin": float(
                aggregate.get("mean_matched_minus_shuffled_signature_target_margin", float("-inf"))
            ),
            "mean_target_margin": float(aggregate.get("mean_target_margin", float("-inf"))),
            "model_hash": fitted["model_hash"],
            "pareto_undominated_rate": float(
                aggregate.get("pareto_undominated_rate", float("-inf"))
            ),
            "target_prediction_rate": float(
                aggregate.get("target_prediction_rate", float("-inf"))
            ),
        }
    except Exception as exc:
        return {
            **dict(config),
            "inner_validation_record_count": -1,
            "invalid": True,
            "invalid_reasons": [f"exception:{type(exc).__name__}:{exc}"],
        }


def safe_progress_invalid_reasons(
    candidate: Mapping[str, Any],
    *,
    max_reasons: int = 5,
    max_chars: int = 240,
) -> list[str]:
    """Return bounded, non-raw invalid reasons for progress logs."""
    sanitized = []
    forbidden_terms = ("weights", "signature")
    for reason in list(candidate.get("invalid_reasons", []))[:max_reasons]:
        text = " ".join(str(reason).split())
        if any(term in text.lower() for term in forbidden_terms):
            text = "[redacted_invalid_reason_contains_raw_term]"
        sanitized.append(text[:max_chars])
    return sanitized


def inner_validation_completion_progress_extra(
    *,
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    record_budget: int,
    rung_index: int,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "config_hash": str(config["config_hash"]),
        "invalid": bool(candidate.get("invalid")),
        "invalid_reason_count": len(candidate.get("invalid_reasons", [])),
        "record_budget": int(record_budget),
        "rung_index": int(rung_index),
    }
    if candidate.get("invalid"):
        extra["invalid_reasons"] = safe_progress_invalid_reasons(candidate)
    if "proof_gate_failure_count" in candidate:
        extra["proof_gate_failure_count"] = int(candidate["proof_gate_failure_count"])
    if candidate.get("proof_gate_failures"):
        extra["proof_gate_failures"] = safe_progress_invalid_reasons(
            {"invalid_reasons": candidate["proof_gate_failures"]}
        )
    for metric_name in (
        "inner_validation_record_count",
        "mean_matched_minus_best_control_target_margin",
        "mean_matched_minus_shuffled_signature_target_margin",
        "mean_target_margin",
        "pareto_undominated_rate",
        "target_prediction_rate",
    ):
        if metric_name in candidate:
            value = candidate[metric_name]
            if isinstance(value, (int, float, str, bool)) or value is None:
                extra[metric_name] = value
    return extra


def _inner_validation_candidate_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    return inner_validation_candidate_for_config(
        config=payload["config"],
        train_subjects=payload["train_subjects"],
        validation_subjects=payload["validation_subjects"],
        train_stats=payload["train_stats"],
        teacher_records=payload["teacher_records"],
        progress_log_path=payload.get("progress_log_path"),
        started_at_monotonic=payload.get("started_at_monotonic"),
    )


def select_config_with_inner_validation(
    *,
    split: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    max_workers: int | None,
    progress_log_path: Path,
    started_at_monotonic: float,
) -> dict[str, Any]:
    full_configs = iter_v24_configs()
    configs = v24_evaluated_config_subset(full_configs)
    full_grid_hash = v24_full_config_grid_hash(full_configs)
    subset_hash = v24_evaluated_config_subset_hash(configs)
    active_configs = [dict(config) for config in configs]
    rung_summaries = []
    selected: Mapping[str, Any] | None = None
    record_progress_event(
        progress_log_path,
        event="inner_validation_start",
        started_at_monotonic=started_at_monotonic,
        extra={
            "evaluated_config_count": len(configs),
            "evaluated_config_subset_hash": subset_hash,
            "full_config_grid_hash": full_grid_hash,
            "total_config_count": len(full_configs),
        },
    )
    for rung_index, (record_budget, survivor_count) in enumerate(zip(
        INNER_VALIDATION_RUNG_RECORD_BUDGETS,
        INNER_VALIDATION_RUNG_SURVIVORS,
    )):
        validation_subjects = inner_validation_subjects_for_budget(split, record_budget)
        record_progress_event(
            progress_log_path,
            event="inner_validation_rung_start",
            started_at_monotonic=started_at_monotonic,
            extra={
                "candidate_count": len(active_configs),
                "record_budget": int(record_budget),
                "rung_index": int(rung_index),
                "validation_subject_count": len(validation_subjects),
            },
        )
        teacher_cache = build_teacher_record_cache(
            configs=active_configs,
            subjects=split["inner_train_subjects"],
            train_stats=train_stats,
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
        )
        candidates = []
        for config in active_configs:
            record_progress_event(
                progress_log_path,
                event="inner_validation_candidate_start",
                started_at_monotonic=started_at_monotonic,
                extra={
                    "config_hash": str(config["config_hash"]),
                    "record_budget": int(record_budget),
                    "rung_index": int(rung_index),
                },
            )
        default_workers = max(1, min(4, (os.cpu_count() or 2) - 2))
        worker_count = max(1, min(max_workers or default_workers, len(active_configs)))
        if worker_count == 1:
            for config in active_configs:
                candidate = inner_validation_candidate_for_config(
                    config=config,
                    train_subjects=split["inner_train_subjects"],
                    validation_subjects=validation_subjects,
                    train_stats=train_stats,
                    teacher_records=teacher_cache[teacher_config_cache_key(config)],
                    progress_log_path=progress_log_path,
                    started_at_monotonic=started_at_monotonic,
                )
                candidates.append(candidate)
                record_progress_event(
                    progress_log_path,
                    event="inner_validation_candidate_completed",
                    started_at_monotonic=started_at_monotonic,
                    extra=inner_validation_completion_progress_extra(
                        config=config,
                        candidate=candidate,
                        record_budget=record_budget,
                        rung_index=rung_index,
                    ),
                )
        else:
            context = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
            ) as executor:
                future_to_config = {
                    executor.submit(_inner_validation_candidate_worker, {
                        "config": dict(config),
                        "train_stats": train_stats,
                        "train_subjects": list(split["inner_train_subjects"]),
                        "teacher_records": teacher_cache[teacher_config_cache_key(config)],
                        "validation_subjects": validation_subjects,
                        "progress_log_path": progress_log_path,
                        "started_at_monotonic": started_at_monotonic,
                    }): config
                    for config in active_configs
                }
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
                    record_progress_event(
                        progress_log_path,
                        event="inner_validation_candidate_completed",
                        started_at_monotonic=started_at_monotonic,
                        extra=inner_validation_completion_progress_extra(
                            config=config,
                            candidate=candidate,
                            record_budget=record_budget,
                            rung_index=rung_index,
                        ),
                    )
        ranked = rank_inner_validation_candidates(candidates)
        kept = ranked[: min(int(survivor_count), len(ranked))]
        rung_summaries.append({
            "candidate_count": len(candidates),
            "invalid_count": sum(1 for candidate in candidates if candidate.get("invalid")),
            "record_budget": int(record_budget),
            "rung_index": int(rung_index),
            "survivor_count": len(kept),
            "survivor_hashes": [str(candidate["config_hash"]) for candidate in kept],
        })
        record_progress_event(
            progress_log_path,
            event="inner_validation_rung_completed",
            started_at_monotonic=started_at_monotonic,
            extra={
                "invalid_count": rung_summaries[-1]["invalid_count"],
                "rung_index": int(rung_index),
                "survivor_count": len(kept),
            },
        )
        active_configs = [dict(config) for config in kept]
        selected = kept[0] if kept else None
    if selected is None or selected.get("invalid"):
        raise ValueError("no valid V24 inner-validation config selected")
    selected = dict(selected)
    selected["inner_validation_evaluated_config_subset_hash"] = subset_hash
    selected["inner_validation_full_config_grid_count"] = len(full_configs)
    selected["inner_validation_full_config_grid_hash"] = full_grid_hash
    selected["inner_validation_rung_record_budgets"] = INNER_VALIDATION_RUNG_RECORD_BUDGETS
    selected["inner_validation_rung_summaries"] = rung_summaries
    selected["inner_validation_rung_survivors"] = INNER_VALIDATION_RUNG_SURVIVORS
    selected["inner_validation_selection_hash"] = stable_hash_json({
        "metrics": {
            key: selected[key]
            for key in [
                "target_prediction_rate",
                "pareto_undominated_rate",
                "mean_matched_minus_best_control_target_margin",
                "mean_matched_minus_shuffled_signature_target_margin",
                "mean_target_margin",
                "mean_compatible_source_mse",
            ]
        },
        "rung_summaries": rung_summaries,
        "scope": "v24_inner_validation_selection",
        "selected_config_hash": selected["config_hash"],
        "evaluated_config_count": len(configs),
        "evaluated_config_subset_hash": subset_hash,
        "full_config_grid_count": len(full_configs),
        "full_config_grid_hash": full_grid_hash,
    })
    record_progress_event(
        progress_log_path,
        event="inner_validation_completed",
        started_at_monotonic=started_at_monotonic,
        extra={
            "inner_validation_selection_hash": selected["inner_validation_selection_hash"],
            "selected_config_hash": selected["config_hash"],
        },
    )
    return selected


def fit_v24_train_statistics(
    train_subjects: Sequence[Mapping[str, Any]],
    *,
    max_workers: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    started_at = time.monotonic()
    progress_log_path = output_dir / INNER_VALIDATION_PROGRESS_LOG_FILENAME
    split = inner_train_validation_split(train_subjects)
    train_stats = fit_v24_normalization_stats(split["inner_train_subjects"])
    all_descriptors = {}
    for subject in train_subjects:
        subject_id = str(subject["subject_id"])
        if subject_id not in train_stats["activation_descriptor_by_subject"]:
            all_descriptors[subject_id] = activation_descriptor_for_weights(
                record_weights_tensor(subject),
                probe_examples=train_stats["probe_examples"],
            )
    train_stats["activation_descriptor_by_subject"].update(all_descriptors)
    train_stats["inner_split_hash"] = split["inner_split_hash"]
    train_stats["train_by_behavior"] = records_by_behavior(train_subjects)
    record_progress_event(
        progress_log_path,
        event="baseline_train_statistics_start",
        started_at_monotonic=started_at,
        extra={"train_subject_count": len(train_subjects)},
    )
    train_stats["v21_train_stats"] = v23.v21.fit_v21_train_statistics(
        train_subjects,
        include_models=True,
        include_baseline_stats=False,
        allow_default_small_pool=True,
    )
    train_stats["v22_train_stats"] = v23.v22.fit_v22_train_statistics(
        train_subjects,
        include_models=True,
        include_baseline_stats=False,
        allow_default_small_pool=True,
    )
    train_stats["v23_train_stats"] = v23.fit_v23_train_statistics(
        train_subjects,
        include_models=True,
        include_baseline_stats=False,
        allow_default_small_pool=True,
        run_inner_validation=False,
    )
    record_progress_event(
        progress_log_path,
        event="baseline_train_statistics_completed",
        started_at_monotonic=started_at,
        extra={
            "v21_train_statistics_hash": train_stats["v21_train_stats"].get(
                "train_statistics_hash"
            ),
            "v22_train_statistics_hash": train_stats["v22_train_stats"].get(
                "train_statistics_hash"
            ),
            "v23_train_statistics_hash": train_stats["v23_train_stats"].get(
                "train_statistics_hash"
            ),
        },
    )
    selected = select_config_with_inner_validation(
        split=split,
        train_stats=train_stats,
        max_workers=max_workers,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at,
    )
    record_progress_event(
        progress_log_path,
        event="selected_model_training_start",
        started_at_monotonic=started_at,
        extra={"selected_config_hash": selected["config_hash"]},
    )
    selected_teacher_records = build_teacher_records(
        train_subjects,
        config=selected,
        train_stats=train_stats,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at,
        progress_event_prefix="selected_teacher_cache",
    )
    fitted = fit_hypereditor_for_config(
        subjects=train_subjects,
        config=selected,
        train_stats=train_stats,
        teacher_records=selected_teacher_records,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at,
        progress_event_prefix="selected",
    )
    no_sig = fit_hypereditor_for_config(
        subjects=train_subjects,
        config=selected,
        train_stats=train_stats,
        teacher_records=selected_teacher_records,
        no_signature_training=True,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at,
        progress_event_prefix="selected_no_signature",
    )
    train_stats["model"] = fitted["model"]
    train_stats["model_hash"] = fitted["model_hash"]
    train_stats["no_signature_model"] = no_sig["model"]
    train_stats["no_signature_model_hash"] = no_sig["model_hash"]
    train_stats["selected_config"] = selected
    train_stats["train_statistics_hash"] = stable_hash_json({
        "activation_descriptor_zero_std_count": train_stats[
            "activation_descriptor_zero_std_count"
        ],
        "inner_split_hash": train_stats["inner_split_hash"],
        "inner_validation_full_config_grid_count": selected[
            "inner_validation_full_config_grid_count"
        ],
        "inner_validation_full_config_grid_hash": selected[
            "inner_validation_full_config_grid_hash"
        ],
        "model_hash": train_stats["model_hash"],
        "no_signature_model_hash": train_stats["no_signature_model_hash"],
        "probe_examples_hash": train_stats["probe_examples_hash"],
        "scope": "v24_train_statistics",
        "selected_config_hash": selected["config_hash"],
        "signature_zero_std_count": train_stats["signature_zero_std_count"],
        "v21_train_statistics_hash": train_stats["v21_train_stats"].get(
            "train_statistics_hash"
        ),
        "v22_train_statistics_hash": train_stats["v22_train_stats"].get(
            "train_statistics_hash"
        ),
        "v23_train_statistics_hash": train_stats["v23_train_stats"].get(
            "train_statistics_hash"
        ),
        "weights_zero_std_count": train_stats["weights_zero_std_count"],
    })
    record_progress_event(
        progress_log_path,
        event="selected_model_training_completed",
        started_at_monotonic=started_at,
        extra={
            "model_hash": train_stats["model_hash"],
            "no_signature_model_hash": train_stats["no_signature_model_hash"],
            "train_statistics_hash": train_stats["train_statistics_hash"],
        },
    )
    return train_stats


def assert_no_forbidden_final_raw_paths(
    paths: Sequence[Path | str],
    *,
    allow_v24_final: bool = False,
) -> None:
    normalized = {Path(path).expanduser().resolve() for path in paths}
    if not allow_v24_final and V24_FINAL_RAW.resolve() in normalized:
        raise ValueError("V24 final raw path is forbidden before hash-bound authorization")
    prior_hits = sorted(str(path) for path in normalized & {p.resolve() for p in PRIOR_FINAL_RAW_PATHS})
    if prior_hits:
        raise ValueError(f"prior sealed final raw path is forbidden: {prior_hits[0]}")
    for path in normalized:
        if path.name == "final_subjects.json" and path != V24_FINAL_RAW.resolve():
            raise ValueError(f"unexpected final raw path is forbidden: {path}")


def forbidden_final_redacted_keys(payload: Mapping[str, Any]) -> list[str]:
    forbidden = []
    for key in payload:
        if key not in FINAL_REDACTED_ALLOWED_TOP_LEVEL_KEYS:
            forbidden.append(f"top_level.{key}")
    summary = payload.get("summary", {})
    if isinstance(summary, Mapping):
        for key in summary:
            if key not in FINAL_REDACTED_ALLOWED_SUMMARY_KEYS:
                forbidden.append(f"summary.{key}")
    else:
        forbidden.append("summary")
    forbidden.extend(recursive_forbidden_final_detail_keys(payload))
    return sorted(forbidden)


def recursive_forbidden_final_detail_keys(value: Any, *, prefix: str = "") -> list[str]:
    forbidden = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            key_lower = str(key).lower()
            if any(
                term in key_lower
                for term in RECURSIVE_FORBIDDEN_FINAL_DETAIL_KEY_TERMS
            ):
                forbidden.append(path)
            forbidden.extend(recursive_forbidden_final_detail_keys(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            forbidden.extend(
                recursive_forbidden_final_detail_keys(item, prefix=f"{prefix}[{index}]")
            )
    return forbidden


def forbidden_combined_final_summary_keys(payload: Mapping[str, Any]) -> list[str]:
    return sorted(
        key for key in payload
        if key not in FINAL_COMBINED_SUMMARY_ALLOWED_KEYS
    )


def build_v24_seed_preflight() -> dict[str, Any]:
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
    final_summary = summaries.get("final", {})
    expected_train_counts = {pattern: 64 for pattern in PATTERNS}
    expected_eval_counts = {pattern: 24 for pattern in PATTERNS}
    if final_redacted.get("pool_file_sha256") != final_summary.get("pool_file_sha256"):
        failures.append("final redacted pool_file_sha256 mismatch")
    if (
        final_redacted.get("pool_redacted_payload_sha256")
        != final_summary.get("pool_redacted_payload_sha256")
    ):
        failures.append("final redacted pool_redacted_payload_sha256 mismatch")
    final_redacted_summary = final_redacted.get("summary", {})
    if isinstance(final_redacted_summary, Mapping):
        if (
            final_redacted_summary.get("accepted_counts_by_behavior")
            != expected_eval_counts
        ):
            failures.append("final redacted accepted counts mismatch")
        if final_redacted_summary.get("max_selected_train_vs_heldout_overlap_count") != 0:
            failures.append("final redacted train/heldout overlap nonzero")
    if summaries.get("train", {}).get("accepted_counts_by_behavior") != expected_train_counts:
        failures.append("train accepted counts mismatch")
    eval_name = "development" if phase == "development" else "final"
    if summaries.get(eval_name, {}).get("accepted_counts_by_behavior") != expected_eval_counts:
        failures.append(f"{eval_name} accepted counts mismatch")
    if summaries.get("train", {}).get("pool_file_sha256") != sha256_file(train_path):
        failures.append("train pool hash mismatch")
    if phase == "development":
        if summaries.get("development", {}).get("pool_file_sha256") != sha256_file(eval_path):
            failures.append("development pool hash mismatch")
    if combined_audit.get("passed") is not True:
        failures.append("combined audit did not pass")
    assert_no_forbidden_final_raw_paths(
        [train_path, eval_path],
        allow_v24_final=(phase == "final"),
    )
    return failures


def build_probe_examples() -> list[dict[str, Any]]:
    return v23.build_probe_examples()


def generate_pools(args: argparse.Namespace, pool_dir: Path) -> dict[str, Any]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    progress_log_path = pool_dir / SOURCE_POOL_PROGRESS_LOG_FILENAME
    seed_preflight = build_v24_seed_preflight()
    record_progress_event(
        progress_log_path,
        event="seed_preflight_completed",
        started_at_monotonic=started_at,
        extra={
            "failure_count": len(seed_preflight["failures"]),
            "seed_range_count": len(seed_preflight["seed_ranges"]),
        },
    )
    if seed_preflight["failures"]:
        result = {"failures": seed_preflight["failures"], "passed": False}
        (pool_dir / "combined_audit.json").write_text(
            json.dumps(result, indent=2, sort_keys=True)
        )
        return result
    suite = v23.v16.v15.build_suite(args.support_per_class, args.heldout_per_class)
    heldout_sequences = v23.v16.v15.build_heldout_sequences(suite)
    candidate_pools = v23.v16.v15.build_candidate_pools(heldout_sequences)
    candidate_pool_summary = v23.v16.v15.summarize_candidate_pools(candidate_pools)
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
        record_progress_event(
            progress_log_path,
            event="pool_generation_start",
            started_at_monotonic=started_at,
            extra={"pool": pool_name},
        )
        payload = v23.v16.v15.poolgen.generate_pool(
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
        payload.setdefault("config", {})
        payload["config"]["base_seed"] = int(pool_config["base_seed"])
        payload["config"]["seed_behavior_stride"] = int(SEED_BEHAVIOR_STRIDE)
        payload["pool_redacted_payload_sha256"] = stable_hash_json(
            v23.v16.v15.poolgen.redact_weights_and_signatures(payload)
        )
        pool_payloads[pool_name] = payload
        pool_path = pool_dir / f"{pool_name}_subjects.json"
        pool_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        summary = v23.v16.v15.poolgen.summarize_pool(payload)
        summary["pool_file_sha256"] = sha256_file(pool_path)
        summary["pool_redacted_payload_sha256"] = payload["pool_redacted_payload_sha256"]
        pool_summaries[pool_name] = summary
        record_progress_event(
            progress_log_path,
            event="pool_generation_completed",
            started_at_monotonic=started_at,
            extra={
                "pool": pool_name,
                "pool_file_sha256": summary["pool_file_sha256"],
            },
        )

    final_redacted = v23.v16.v15.poolgen.build_final_redacted_summary(
        pool_payloads["final"]
    )
    final_redacted["claim_scope"] = FINAL_REDACTED_SCOPE
    final_redacted["pool_file_sha256"] = pool_summaries["final"]["pool_file_sha256"]
    final_redacted["pool_redacted_payload_sha256"] = pool_summaries["final"][
        "pool_redacted_payload_sha256"
    ]
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
    record_progress_event(
        progress_log_path,
        event="final_redacted_audit_written",
        started_at_monotonic=started_at,
        extra={"final_redacted_audit_sha256": sha256_file(pool_dir / "final_redacted_audit.json")},
    )

    audit = v23.v16.v15.poolgen.build_combined_audit(
        pool_payloads=pool_payloads,
        pool_summaries=pool_summaries,
        seed_preflight=seed_preflight,
        suite=suite,
        probe_examples=probe_examples,
    )
    audit["claim_scope"] = SOURCE_AUDIT_SCOPE
    audit = v23.v16.v15.v10.redact_combined_audit(audit)
    final_summary = audit.get("pool_summaries", {}).get("final", {})
    audit["pool_summaries"]["final"] = {
        key: final_summary[key]
        for key in sorted(FINAL_COMBINED_SUMMARY_ALLOWED_KEYS)
        if key in final_summary
    }
    final_summary_failures = forbidden_combined_final_summary_keys(
        audit["pool_summaries"]["final"]
    )
    if final_summary_failures:
        raise ValueError(
            "combined_audit.pool_summaries.final key mismatch: "
            + ", ".join(final_summary_failures)
        )
    audit["audit_payload_sha256"] = stable_hash_json(audit)
    (pool_dir / "combined_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True)
    )
    record_progress_event(
        progress_log_path,
        event="combined_audit_written",
        started_at_monotonic=started_at,
        extra={"combined_audit_sha256": sha256_file(pool_dir / "combined_audit.json")},
    )
    return {
        "combined_audit_path": str(pool_dir / "combined_audit.json"),
        "source_pool_progress_log_path": str(progress_log_path),
        "source_pool_progress_log_sha256": sha256_file(progress_log_path),
        "final_redacted_audit_path": str(pool_dir / "final_redacted_audit.json"),
        "passed": bool(audit.get("passed", False)),
        "pool_dir": str(pool_dir),
        "pool_summaries": audit.get("pool_summaries", {}),
        "seed_preflight": seed_preflight,
    }


def development_result_payload(
    *,
    eval_result: Mapping[str, Any],
    paths: Mapping[str, Path],
    selected_config_hash: str,
    selected_model_hash: str,
    train_statistics_hash: str,
    inner_validation_selection_hash: str,
    evaluated_config_subset_hash: str,
    full_config_grid_hash: str,
    full_config_grid_count: int,
) -> dict[str, Any]:
    failures = list(eval_result.get("failures", []))
    passed = not failures
    return {
        "aggregate": dict(eval_result.get("aggregate", {})),
        "by_direction": dict(eval_result.get("by_direction", {})),
        "claim_scope": DEVELOPMENT_SCOPE,
        "combined_audit_sha256": sha256_file(paths["combined_audit"]),
        "constants_sha256": stable_hash_json(constants_payload()),
        "development_pool_sha256": sha256_file(paths["development"]),
        "development_progress_log_pre_results_sha256": sha256_file(
            paths["development_progress"]
        ),
        "editor_method": EDITOR_METHOD,
        "evaluated_config_subset_hash": evaluated_config_subset_hash,
        "failures": failures,
        "final_redacted_audit_sha256": sha256_file(paths["final_redacted"]),
        "inner_validation_full_config_grid_count": int(full_config_grid_count),
        "inner_validation_full_config_grid_hash": full_config_grid_hash,
        "inner_validation_selection_hash": inner_validation_selection_hash,
        "inner_validation_progress_log_sha256": sha256_file(
            paths["inner_validation_progress"]
        ),
        "next_action": (
            PASSING_DEVELOPMENT_NEXT_ACTION
            if passed else FAILING_DEVELOPMENT_NEXT_ACTION
        ),
        "passed": passed,
        "phase": "development",
        "plan_sha256": PLAN_SHA256,
        "record_count": int(eval_result.get("record_count", 0)),
        "records": list(eval_result.get("records", [])),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "selected_config_hash": selected_config_hash,
        "selected_model_hash": selected_model_hash,
        "train_pool_sha256": sha256_file(paths["train"]),
        "train_statistics_hash": train_statistics_hash,
    }


def build_final_authorization_payload(
    *,
    development_result: Mapping[str, Any],
    development_results_sha256: str,
    formal_preregistration_sha256: str,
    helper_test_sha256: str,
    reviewer_authorization_sha256: str,
    reviewer_confidence: str,
    script_sha256: str,
) -> dict[str, Any]:
    return {
        "combined_audit_sha256": development_result["combined_audit_sha256"],
        "constants_sha256": stable_hash_json(constants_payload()),
        "development_claim_scope": development_result["claim_scope"],
        "development_next_action": development_result["next_action"],
        "development_passed": bool(development_result["passed"]),
        "development_phase": development_result["phase"],
        "development_pool_sha256": development_result["development_pool_sha256"],
        "development_results_sha256": development_results_sha256,
        "editor_method": development_result["editor_method"],
        "final_redacted_audit_sha256": development_result["final_redacted_audit_sha256"],
        "formal_preregistration_sha256": formal_preregistration_sha256,
        "helper_test_sha256": helper_test_sha256,
        "inner_validation_evaluated_config_subset_hash": development_result[
            "evaluated_config_subset_hash"
        ],
        "inner_validation_full_config_grid_count": development_result[
            "inner_validation_full_config_grid_count"
        ],
        "inner_validation_full_config_grid_hash": development_result[
            "inner_validation_full_config_grid_hash"
        ],
        "inner_validation_selection_hash": development_result[
            "inner_validation_selection_hash"
        ],
        "plan_sha256": PLAN_SHA256,
        "reviewer_authorization_sha256": reviewer_authorization_sha256,
        "reviewer_confidence": reviewer_confidence,
        "script_sha256": script_sha256,
        "selected_config_hash": development_result["selected_config_hash"],
        "selected_model_hash": development_result["selected_model_hash"],
        "train_pool_sha256": development_result["train_pool_sha256"],
        "train_statistics_hash": development_result["train_statistics_hash"],
    }


def validate_final_authorization_payload(
    authorization: Mapping[str, Any],
    *,
    development_result: Mapping[str, Any],
    development_results_sha256: str,
    formal_preregistration_sha256: str,
    helper_test_sha256: str,
    reviewer_authorization_sha256: str,
    reviewer_confidence: str,
    script_sha256: str,
) -> None:
    expected = build_final_authorization_payload(
        development_result=development_result,
        development_results_sha256=development_results_sha256,
        formal_preregistration_sha256=formal_preregistration_sha256,
        helper_test_sha256=helper_test_sha256,
        reviewer_authorization_sha256=reviewer_authorization_sha256,
        reviewer_confidence=reviewer_confidence,
        script_sha256=script_sha256,
    )
    for key, expected_value in expected.items():
        if authorization.get(key) != expected_value:
            raise ValueError(f"final authorization mismatch: {key}")
    extra_keys = set(authorization) - set(expected)
    if extra_keys:
        raise ValueError(f"final authorization has unexpected keys: {sorted(extra_keys)}")
    if authorization.get("development_claim_scope") != DEVELOPMENT_SCOPE:
        raise ValueError("final authorization development_claim_scope mismatch")
    if authorization.get("development_phase") != "development":
        raise ValueError("final authorization development_phase mismatch")
    if authorization.get("development_passed") is not True:
        raise ValueError("final authorization requires development_passed=True")
    if authorization.get("development_next_action") != PASSING_DEVELOPMENT_NEXT_ACTION:
        raise ValueError("final authorization next_action mismatch")
    if authorization.get("reviewer_confidence") != "5/5":
        raise ValueError("final authorization requires reviewer confidence 5/5")


def development_stdout_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in [
            "aggregate",
            "claim_scope",
            "development_results_path",
            "failures",
            "next_action",
            "passed",
            "phase",
            "record_count",
            "selected_config_hash",
            "selected_model_hash",
            "train_statistics_hash",
        ]
        if key in result
    }


def run_development(
    args: argparse.Namespace,
    pool_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started_at = time.monotonic()
    progress_log_path = output_dir / DEVELOPMENT_PROGRESS_LOG_FILENAME
    record_progress_event(
        progress_log_path,
        event="development_start",
        started_at_monotonic=started_at,
        extra={
            "max_workers": args.max_workers,
            "output_dir": rel(output_dir),
            "pool_dir": rel(pool_dir),
        },
    )
    train_path = pool_dir / "train_subjects.json"
    development_path = pool_dir / "development_subjects.json"
    combined_audit_path = pool_dir / "combined_audit.json"
    final_redacted_path = pool_dir / "final_redacted_audit.json"
    assert_no_forbidden_final_raw_paths([
        train_path,
        development_path,
        combined_audit_path,
        final_redacted_path,
    ])
    train_payload = v23.v16.v15.v1.load_json(train_path)
    development_payload = v23.v16.v15.v1.load_json(development_path)
    combined_audit = v23.v16.v15.v1.load_json(combined_audit_path)
    final_redacted = v23.v16.v15.v1.load_json(final_redacted_path)
    record_progress_event(
        progress_log_path,
        event="source_payloads_loaded",
        started_at_monotonic=started_at,
        extra={
            "development_pool_sha256": sha256_file(development_path),
            "train_pool_sha256": sha256_file(train_path),
        },
    )
    contract_failures = validate_source_pool_contract(
        train_path=train_path,
        eval_path=development_path,
        train_payload=train_payload,
        eval_payload=development_payload,
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )
    if contract_failures:
        raise ValueError(
            "V24 source-pool contract validation failed: "
            + "; ".join(contract_failures)
        )
    record_progress_event(
        progress_log_path,
        event="source_pool_contract_validated",
        started_at_monotonic=started_at,
        extra={"contract_failure_count": 0},
    )
    train_subjects = v23.v16.v15.v1.accepted_records(train_payload)
    development_subjects = v23.v16.v15.v1.accepted_records(development_payload)
    record_progress_event(
        progress_log_path,
        event="train_statistics_start",
        started_at_monotonic=started_at,
        extra={
            "development_subject_count": len(development_subjects),
            "train_subject_count": len(train_subjects),
        },
    )
    train_stats = fit_v24_train_statistics(
        train_subjects,
        max_workers=args.max_workers,
        output_dir=output_dir,
    )
    inner_progress_path = output_dir / INNER_VALIDATION_PROGRESS_LOG_FILENAME
    if not inner_progress_path.exists():
        record_progress_event(
            inner_progress_path,
            event="inner_validation_progress_unavailable",
            started_at_monotonic=started_at,
            extra={"reason": "not_written_by_train_statistics"},
        )
    record_progress_event(
        progress_log_path,
        event="train_statistics_completed",
        started_at_monotonic=started_at,
        extra={
            "selected_config_hash": train_stats["selected_config"]["config_hash"],
            "train_statistics_hash": train_stats["train_statistics_hash"],
        },
    )
    stats_path = output_dir / STATS_ARTIFACT_FILENAME
    no_signature_model = train_stats.get("no_signature_model", train_stats["model"])
    torch.save({
        "constants_sha256": stable_hash_json(constants_payload()),
        "inner_split_hash": train_stats.get("inner_split_hash"),
        "inner_validation_full_config_grid_count": train_stats["selected_config"].get(
            "inner_validation_full_config_grid_count"
        ),
        "inner_validation_full_config_grid_hash": train_stats["selected_config"].get(
            "inner_validation_full_config_grid_hash"
        ),
        "model_state_dict": train_stats["model"].state_dict(),
        "model_hash": train_stats["model_hash"],
        "no_signature_model_hash": train_stats.get("no_signature_model_hash"),
        "no_signature_model_state_dict": no_signature_model.state_dict(),
        "probe_examples_hash": train_stats.get("probe_examples_hash"),
        "selected_config": train_stats["selected_config"],
        "train_statistics_hash": train_stats["train_statistics_hash"],
    }, stats_path)
    record_progress_event(
        progress_log_path,
        event="stats_artifact_written",
        started_at_monotonic=started_at,
        extra={"stats_artifact_sha256": sha256_file(stats_path)},
    )
    record_progress_event(
        progress_log_path,
        event="development_evaluation_start",
        started_at_monotonic=started_at,
        extra={"development_subject_count": len(development_subjects)},
    )
    eval_result = evaluate_subjects(
        subjects=development_subjects,
        train_stats=train_stats,
        record_evaluator=evaluate_v24_record_from_job,
        parallel=True,
        max_workers=args.max_workers,
        progress_log_path=progress_log_path,
        progress_started_at_monotonic=started_at,
    )
    record_progress_event(
        progress_log_path,
        event="development_evaluation_completed",
        started_at_monotonic=started_at,
        extra={
            "failure_count": len(eval_result.get("failures", [])),
            "record_count": int(eval_result.get("record_count", 0)),
        },
    )
    selected_config = train_stats["selected_config"]
    result = development_result_payload(
        eval_result=eval_result,
        paths={
            "combined_audit": combined_audit_path,
            "development": development_path,
            "development_progress": progress_log_path,
            "final_redacted": final_redacted_path,
            "inner_validation_progress": inner_progress_path,
            "train": train_path,
        },
        selected_config_hash=str(selected_config["config_hash"]),
        selected_model_hash=str(train_stats["model_hash"]),
        train_statistics_hash=str(train_stats["train_statistics_hash"]),
        inner_validation_selection_hash=str(
            selected_config["inner_validation_selection_hash"]
        ),
        evaluated_config_subset_hash=str(
            selected_config["inner_validation_evaluated_config_subset_hash"]
        ),
        full_config_grid_hash=str(
            selected_config["inner_validation_full_config_grid_hash"]
        ),
        full_config_grid_count=int(
            selected_config["inner_validation_full_config_grid_count"]
        ),
    )
    result.update({
        "development_results_path": str(
            rel(output_dir / "development_results.json")
        ),
        "dirty_worktree_caveat": True,
        "final_redacted_audit_path": rel(final_redacted_path),
        "inner_split_hash": train_stats.get("inner_split_hash"),
        "inner_validation_progress_log_path": str(
            rel(inner_progress_path)
        ),
        "limitations": (
            "Small-subject source-label-known, target-label-requested V24 "
            "behavioral-distilled hypereditor development evidence only."
        ),
        "source_pool_audit_passed": bool(combined_audit.get("passed", False)),
        "stats_artifact_path": rel(stats_path),
        "stats_artifact_sha256": sha256_file(stats_path),
        "thresholds": THRESHOLDS,
        "thresholds_sha256": stable_hash_json(THRESHOLDS),
    })
    output_path = output_dir / "development_results.json"
    result["development_progress_log_pre_results_sha256"] = sha256_file(
        progress_log_path
    )
    return write_development_results_artifact(
        output_path=output_path,
        result=result,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at,
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
    parser.add_argument("--summary-only-stdout", action="store_true")
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
        raise SystemExit("V24 final phase is blocked until hash-bound authorization")
    if args.summary_only_stdout:
        result = development_stdout_summary(result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
