"""V25 functional editing via Jacobian-constrained rank-1/spectral edits."""

from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import resource
import sys
import threading
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

import train_four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor as v24  # noqa: E402


v23 = v24.v23
v17 = v23.v17
PATTERNS = v24.PATTERNS
SOURCE_WEIGHT_DIM = v24.SOURCE_WEIGHT_DIM
ACTIVATION_DESCRIPTOR_DIM = v24.ACTIVATION_DESCRIPTOR_DIM
PLAN_SHA256 = "50624768332c77ef85845d2c7a3919755f77e790edda4eb9a926f655e4d585b9"
V25_FULL_GRID_SHA256 = "be8de7e7f6321d0e508a7a9f408a6fbe47167b576f0e33d2211e359c2996aa40"
SEED_BEHAVIOR_STRIDE = v24.SEED_BEHAVIOR_STRIDE
SOURCE_POOL_PROGRESS_LOG_FILENAME = "source_pool_progress.jsonl"
LONG_RUN_MONITOR_LOG_FILENAME = "long_run_monitor.jsonl"

EDITOR_METHOD = "jacobian_rank1_editor_v25"
DEFAULT_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v25_pools"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runs"
    / "four_behavior_functional_weight_editing_v25_jacobian_rank1_editor"
)
V25_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"
POOL_CONFIGS = {
    "train": {
        "base_seed": 126400000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 127400000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 128400000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}

SOURCE_POOL_SCOPE = "four_behavior_functional_weight_editing_v25_source_pool"
SOURCE_AUDIT_SCOPE = "four_behavior_functional_weight_editing_v25_source_pool_construction"
FINAL_REDACTED_SCOPE = (
    "redacted_final_functional_weight_editing_v25_source_pool_audit_surface_only"
)
DEVELOPMENT_SCOPE = "four_behavior_functional_weight_editing_v25_jacobian_rank1_editor_development"
FINAL_SCOPE = "four_behavior_functional_weight_editing_v25_jacobian_rank1_editor_final"

PRIOR_FINAL_RAW_PATHS = {
    *v24.PRIOR_FINAL_RAW_PATHS,
    v24.V24_FINAL_RAW,
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
RECURSIVE_FORBIDDEN_FINAL_DETAIL_KEY_TERMS = {
    "records",
    "weights",
    "signature",
    "subject_id",
    "seed",
    "train_info",
    "support_margin",
    "heldout_margin",
    "logits",
    "descriptor",
    "jacobian",
    "delta",
}

RIDGE_GRID = [1e-5, 1e-4, 1e-3, 1e-2]
COMPAT_WEIGHT_GRID = [0.0, 0.05, 0.10, 0.20]
PROJECTION_GRID = ["none", "rank1", "spectral_rank4", "rank1_spectral_rank4"]
MATCHED_EDIT_SOURCE_GRID = ["jacobian", "empirical_centroid_task_vector"]
V26_EXPERIMENT_VARIANT = "v26_empirical_task_vector_editor"
V27_EXPERIMENT_VARIANT = "v27_localized_behavior_loss_subspace_editor"
V27_LOCALIZED_MATCHED_EDIT_SOURCE = "localized_behavior_loss_subspace"
V27_MATCHED_EDIT_SOURCE_GRID = [V27_LOCALIZED_MATCHED_EDIT_SOURCE]
V27_LOCALIZED_BASIS_GRID = [
    "spectral_train_delta_rank4",
    "target_source_logit_gradient_rank4",
    "combined_spectral_gradient_rank8",
    "output_layer_topk",
]
V27_LOCALIZED_STEPS_GRID = [25, 75]
V27_LOCALIZED_LR_GRID = [0.05, 0.01]
V27_LOCALIZED_SOURCE_MSE_WEIGHT_GRID = [0.5, 1.0]
V27_LOCALIZED_DELTA_L2_WEIGHT_GRID = [0.0, 0.01]
V27_LOCALIZED_NORM_CAP = 0.25
V27_LOCALIZED_GRID_SHA256 = (
    "b2830fc9172d4347d3fcc9b8db9639a684c822e15ef3403a35d78f8d74e6ecc0"
)
V27_LOCALIZED_OPTIMIZER_BETAS = (0.9, 0.999)
V27_LOCALIZED_OPTIMIZER_EPS = 1e-8
V27_LOCALIZED_GRAD_CLIP_NORM = 5.0
V27_LOCALIZED_NATIVE_CONTROL_CONFIG = {
    "compat_weight": 0.1,
    "projection": "rank1",
    "ridge_lambda": 1e-2,
}
V28_EXPERIMENT_VARIANT = "v28_anchor_nullspace_trust_region_editor"
V28_ANCHOR_NULLSPACE_MATCHED_EDIT_SOURCE = "anchor_nullspace_trust_region"
V28_TRUST_NORM_CAP_GRID = [0.25, 0.5]
V28_ANCHOR_COUNT_GRID = [8, 16]
V28_NULLSPACE_RTOL_GRID = [1e-3, 1e-2]
V28_COMPATIBLE_FLOOR_GRID = [0.05]
V28_CONFLICT_WEIGHT = 0.5
V28_TARGET_BCE_WEIGHT = 1.0
V28_CONFLICT_BCE_WEIGHT = 0.5
V28_COMPATIBLE_PROBE_WEIGHT = 0.1
V28_DELTA_L2_WEIGHT = 0.0
V28_ANCHOR_NULLSPACE_STEPS = 50
V28_ANCHOR_NULLSPACE_LR = 0.05
V28_ANCHOR_NULLSPACE_GRID_SHA256 = (
    "4bd940153ed647245836191fda5220f54bd11e1db230259040b191051a42c196"
)
V28_ANCHOR_NULLSPACE_NATIVE_CONTROL_CONFIG = dict(V27_LOCALIZED_NATIVE_CONTROL_CONFIG)
V29_EXPERIMENT_VARIANT = "v29_breadth_first_sparse_support_editor"
V29_BREADTH_FIRST_SPARSE_MATCHED_EDIT_SOURCE = "breadth_first_sparse_support"
V29_SPARSE_TOP_K_GRID = [16, 32]
V29_TRUST_NORM_CAP_GRID = [0.5, 1.0]
V29_COMPATIBLE_FLOOR_GRID = [0.05]
V29_EXTRA_COMPATIBLE_WEIGHT_GRID = [0.05, 0.2]
V29_CONFLICT_WEIGHT = 0.5
V29_TARGET_BCE_WEIGHT = 1.0
V29_CONFLICT_BCE_WEIGHT = 0.5
V29_COMPATIBLE_PROBE_WEIGHT = 0.2
V29_DELTA_L2_WEIGHT = 1e-4
V29_BREADTH_FIRST_EPOCHS = 75
V29_BREADTH_FIRST_LR = 0.05
V29_BREADTH_FIRST_SPARSE_GRID_SHA256 = (
    "ef40cccc68f4cf08e9e8373de9a8df7555170273f6aaa4ff4d524820859aa9d0"
)
V29_BREADTH_FIRST_SPARSE_NATIVE_CONTROL_CONFIG = dict(V27_LOCALIZED_NATIVE_CONTROL_CONFIG)
V30_EXPERIMENT_VARIANT = "v30_margin_gated_sparse_support_editor"
V30_MARGIN_GATED_SPARSE_MATCHED_EDIT_SOURCE = "margin_gated_sparse_support"
V30_SPARSE_TOP_K_GRID = [32, 64]
V30_TRUST_NORM_CAP_GRID = [1.0, 1.25]
V30_TARGET_MARGIN_FLOOR_GRID = [0.15, 0.25]
V30_COMPATIBLE_FLOOR = 0.05
V30_EXTRA_COMPATIBLE_WEIGHT = 0.05
V30_TARGET_MARGIN_WEIGHT = 0.5
V30_MARGIN_GATED_SPARSE_GRID_SHA256 = (
    "3225c6db22149aba92f1366f23010a1e87de8cde4175cd5b37ff826071cb59cc"
)
V30_MARGIN_GATED_NATIVE_CONTROL_CONFIG = dict(V27_LOCALIZED_NATIVE_CONTROL_CONFIG)
V31_EXPERIMENT_VARIANT = "v31_orthogonal_sign_sparse_support_editor"
V31_ORTHOGONAL_SIGN_SPARSE_MATCHED_EDIT_SOURCE = "orthogonal_sign_sparse_support"
V31_TRUST_NORM_CAP_GRID = [1.25, 1.5]
V31_SIGN_CONFLICT_PENALTY_GRID = [0.5, 1.0]
V31_COMPATIBLE_ORTHOGONAL_WEIGHT_GRID = [0.05, 0.15]
V31_SPARSE_TOP_K = 64
V31_TARGET_MARGIN_FLOOR = 0.25
V31_COMPATIBLE_FLOOR = 0.05
V31_EXTRA_COMPATIBLE_WEIGHT = 0.05
V31_HARD_TARGET_MARGIN_WEIGHT = 1.0
V31_ORTHOGONAL_SIGN_SPARSE_GRID_SHA256 = (
    "bf2336c5997b5f1258f407d13a687fd8923424d849632bf7e5673e0728f23337"
)
V31_ORTHOGONAL_SIGN_NATIVE_CONTROL_CONFIG = dict(V27_LOCALIZED_NATIVE_CONTROL_CONFIG)
V32_EXPERIMENT_VARIANT = "v32_support_tournament_margin_editor"
V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE = "support_tournament_margin_sparse"
V32_TOURNAMENT_MARGIN_WEIGHT_GRID = [0.5, 1.0]
V32_TOURNAMENT_MARGIN_FLOOR_GRID = [0.05, 0.15]
V32_COMPATIBLE_ORTHOGONAL_WEIGHT_GRID = [0.05, 0.15]
V32_TRUST_NORM_CAP = 1.25
V32_SIGN_CONFLICT_PENALTY = 1.0
V32_SPARSE_TOP_K = 64
V32_TARGET_MARGIN_FLOOR = 0.25
V32_COMPATIBLE_FLOOR = 0.05
V32_EXTRA_COMPATIBLE_WEIGHT = 0.05
V32_HARD_TARGET_MARGIN_WEIGHT = 1.0
V32_SUPPORT_TOURNAMENT_GRID_SHA256 = (
    "a7b866c5b8c9808a67c1aa95b063ad49288b33fc5ded869058b6cba2351eb90c"
)
V32_SUPPORT_TOURNAMENT_NATIVE_CONTROL_CONFIG = dict(V27_LOCALIZED_NATIVE_CONTROL_CONFIG)
V33_EXPERIMENT_VARIANT = "v33_proof_gate_decomposition_diagnostic"
V34_EXPERIMENT_VARIANT = "v34_locality_pressure_grid_diagnostic"
V34_LOCALITY_PRESSURE_GRID_SHA256 = (
    "4ee212a749e6db4210ce7ac096e1d5884130d38a2693a7272a23ab354229f722"
)
V35_EXPERIMENT_VARIANT = "v35_support_source_line_search_diagnostic"
V35_SUPPORT_SOURCE_LINE_SEARCH_MATCHED_EDIT_SOURCE = (
    "support_source_line_search_sparse"
)
V35_ALPHA_CANDIDATES = [1.0, 0.75, 0.5, 0.25, 0.125, 0.0]
V35_SUPPORT_SOURCE_LINE_SEARCH_GRID_SHA256 = (
    "8f405dc929810e852822a9f7f9a006051014fff026ab71387384abf941b2881c"
)
V36_EXPERIMENT_VARIANT = "v36_compatible_nullspace_projection_diagnostic"
V36_COMPATIBLE_NULLSPACE_MATCHED_EDIT_SOURCE = (
    "compatible_nullspace_projected_sparse"
)
V36_COMPATIBLE_NULLSPACE_GRID_SHA256 = (
    "23062f76db1a8b4fc2affea41ea2bee9be42db8a8ef2d2b6a0c1f75e1c5fb394"
)
V37_EXPERIMENT_VARIANT = "v37_projected_support_optimizer_diagnostic"
V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE = "projected_support_optimizer_sparse"
V37_PROJECTED_OPTIMIZER_GRID_SHA256 = (
    "e62e13a7ee50b407f9aa5364be2ddbf4dca9f60a385f4bbb0e36dd8a997243bf"
)
V38_EXPERIMENT_VARIANT = "v38_compatible_mse_gated_projected_optimizer_diagnostic"
V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE = (
    "compatible_mse_gated_projected_optimizer_sparse"
)
V38_COMPATIBLE_GATED_GRID_SHA256 = (
    "17513c7f6e091466a3aa364ef4e927591bb6191c8a3cbebb3ca3fa9f2da7895e"
)
V39_EXPERIMENT_VARIANT = "v39_target_feasible_lexicographic_projected_optimizer_diagnostic"
V39_TARGET_FEASIBLE_LEXICOGRAPHIC_MATCHED_EDIT_SOURCE = (
    "target_feasible_lexicographic_projected_optimizer_sparse"
)
V39_GRID_SHA256 = (
    "6595589d2c70e8c67943dafd5a6811b7d6d16315d2d49ce9fa6e9b8ca758fd78"
)
V40_EXPERIMENT_VARIANT = "v40_target_tolerance_locality_budget_projected_optimizer_diagnostic"
V40_TARGET_TOLERANCE_LOCALITY_BUDGET_MATCHED_EDIT_SOURCE = (
    "target_tolerance_locality_budget_projected_optimizer_sparse"
)
V40_GRID_SHA256 = (
    "ae954f7e5907ffa124d17d54a5485ed928faff8679e2fac9b8d94194768b232e"
)
V41_EXPERIMENT_VARIANT = "v41_trajectory_frontier_projected_optimizer_diagnostic"
V41_TRAJECTORY_FRONTIER_MATCHED_EDIT_SOURCE = (
    "trajectory_frontier_projected_optimizer_sparse"
)
V41_GRID_SHA256 = (
    "f2d2f77d807b2d5787599c332dd95b928a86230583490577bcc372db8447b926"
)
V42_EXPERIMENT_VARIANT = "v42_compatible_dual_frontier_projected_optimizer_diagnostic"
V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE = (
    "compatible_dual_frontier_projected_optimizer_sparse"
)
V42_GRID_SHA256 = (
    "fb87d8dffa5ed5132c5cea8925b68b72314ef7b70ca9af569eeca781fee65294"
)

RANDOM_CONTROLS_PER_RECORD = 19
PROOF_CRITICAL_CONTROL_TYPES = [
    "no_signature_ablation",
    "no_signature_trained",
    "source_behavior_target_ablation",
    "shuffled_signature",
    "v21_baseline",
    "v22_baseline",
    "v23_baseline",
    "closed_form_unprojected_jacobian",
    "rank1_random_direction",
    "spectral_basis_random_coefficients",
    "contrastive_weight_arithmetic",
]
DIAGNOSTIC_CONTROL_TYPES = [
    "nearest_train_delta",
    "teacher_oracle_delta",
    "target_only_no_source_compat",
    "activation_only_no_weight_projection",
]
EXPECTED_CONTROLS_PER_RECORD = (
    len(PROOF_CRITICAL_CONTROL_TYPES)
    + len(DIAGNOSTIC_CONTROL_TYPES)
    + RANDOM_CONTROLS_PER_RECORD
)
PER_RECORD_MIN_TARGET_MARGIN = 0.15
PER_RECORD_MIN_CONTROL_MARGIN_ADVANTAGE = 0.02
PER_RECORD_MIN_SHUFFLED_MARGIN_ADVANTAGE = 0.05
PER_RECORD_COMPATIBLE_MSE_TOLERANCE = 0.05

AGGREGATE_PROOF_GATES = {
    "individual_all_gate_pass_rate": 0.85,
    "mean_matched_minus_best_control_target_margin": 0.02,
    "mean_matched_minus_no_signature_ablation_target_margin": 0.02,
    "mean_matched_minus_no_signature_trained_target_margin": 0.02,
    "mean_matched_minus_shuffled_signature_target_margin": 0.05,
    "mean_matched_minus_v21_baseline_target_margin": 0.02,
    "mean_matched_minus_v22_baseline_target_margin": 0.02,
    "mean_matched_minus_v23_baseline_target_margin": 0.02,
    "mean_target_margin": 0.25,
    "pareto_undominated_rate": 0.85,
    "target_prediction_rate": 0.85,
}

stable_hash_json = v24.stable_hash_json
sha256_file = v24.sha256_file
write_development_results_artifact = v24.write_development_results_artifact
record_progress_event = v24.record_progress_event
write_json_atomic = v24.write_json_atomic
activation_descriptor_for_weights = v24.activation_descriptor_for_weights
record_weights_tensor = v24.record_weights_tensor
safe_mean_std = v24.safe_mean_std
tensor_to_hashable = v23.tensor_to_hashable


def build_v25_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for ridge_lambda in RIDGE_GRID:
        for compat_weight in COMPAT_WEIGHT_GRID:
            for projection in PROJECTION_GRID:
                for matched_edit_source in MATCHED_EDIT_SOURCE_GRID:
                    grid.append({
                        "compat_weight": float(compat_weight),
                        "config_index": len(grid),
                        "matched_edit_source": str(matched_edit_source),
                        "projection": str(projection),
                        "ridge_lambda": float(ridge_lambda),
                    })
    return grid


def build_v27_localized_behavior_loss_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for steps in V27_LOCALIZED_STEPS_GRID:
        for lr in V27_LOCALIZED_LR_GRID:
            for source_mse_weight in V27_LOCALIZED_SOURCE_MSE_WEIGHT_GRID:
                for delta_l2_weight in V27_LOCALIZED_DELTA_L2_WEIGHT_GRID:
                    for basis in V27_LOCALIZED_BASIS_GRID:
                        grid.append({
                            "config_index": len(grid),
                            "localized_basis": str(basis),
                            "localized_delta_l2_weight": float(delta_l2_weight),
                            "localized_lr": float(lr),
                            "localized_norm_cap": float(V27_LOCALIZED_NORM_CAP),
                            "localized_source_mse_weight": float(source_mse_weight),
                            "localized_steps": int(steps),
                            "matched_edit_source": V27_LOCALIZED_MATCHED_EDIT_SOURCE,
                        })
    return grid


def build_v28_anchor_nullspace_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for anchor_count in V28_ANCHOR_COUNT_GRID:
        for nullspace_rtol in V28_NULLSPACE_RTOL_GRID:
            for compatible_floor in V28_COMPATIBLE_FLOOR_GRID:
                for trust_norm_cap in V28_TRUST_NORM_CAP_GRID:
                    grid.append({
                        "anchor_count": int(anchor_count),
                        "compatible_floor": float(compatible_floor),
                        "config_index": len(grid),
                        "conflict_weight": float(V28_CONFLICT_WEIGHT),
                        "matched_edit_source": V28_ANCHOR_NULLSPACE_MATCHED_EDIT_SOURCE,
                        "nullspace_rtol": float(nullspace_rtol),
                        "trust_norm_cap": float(trust_norm_cap),
                    })
    return grid


def build_v29_breadth_first_sparse_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for sparse_top_k in V29_SPARSE_TOP_K_GRID:
        for trust_norm_cap in V29_TRUST_NORM_CAP_GRID:
            for compatible_floor in V29_COMPATIBLE_FLOOR_GRID:
                for extra_compatible_weight in V29_EXTRA_COMPATIBLE_WEIGHT_GRID:
                    grid.append({
                        "compatible_floor": float(compatible_floor),
                        "config_index": len(grid),
                        "extra_compatible_weight": float(extra_compatible_weight),
                        "matched_edit_source": V29_BREADTH_FIRST_SPARSE_MATCHED_EDIT_SOURCE,
                        "sparse_top_k": int(sparse_top_k),
                        "trust_norm_cap": float(trust_norm_cap),
                    })
    return grid


def build_v30_margin_gated_sparse_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for sparse_top_k in V30_SPARSE_TOP_K_GRID:
        for trust_norm_cap in V30_TRUST_NORM_CAP_GRID:
            for target_margin_floor in V30_TARGET_MARGIN_FLOOR_GRID:
                grid.append({
                    "compatible_floor": float(V30_COMPATIBLE_FLOOR),
                    "config_index": len(grid),
                    "extra_compatible_weight": float(V30_EXTRA_COMPATIBLE_WEIGHT),
                    "matched_edit_source": V30_MARGIN_GATED_SPARSE_MATCHED_EDIT_SOURCE,
                    "sparse_top_k": int(sparse_top_k),
                    "target_margin_floor": float(target_margin_floor),
                    "trust_norm_cap": float(trust_norm_cap),
                })
    return grid


def build_v31_orthogonal_sign_sparse_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for trust_norm_cap in V31_TRUST_NORM_CAP_GRID:
        for sign_conflict_penalty in V31_SIGN_CONFLICT_PENALTY_GRID:
            for compatible_orthogonal_weight in V31_COMPATIBLE_ORTHOGONAL_WEIGHT_GRID:
                grid.append({
                    "compatible_floor": float(V31_COMPATIBLE_FLOOR),
                    "compatible_orthogonal_weight": float(compatible_orthogonal_weight),
                    "config_index": len(grid),
                    "extra_compatible_weight": float(V31_EXTRA_COMPATIBLE_WEIGHT),
                    "hard_target_margin_weight": float(V31_HARD_TARGET_MARGIN_WEIGHT),
                    "matched_edit_source": V31_ORTHOGONAL_SIGN_SPARSE_MATCHED_EDIT_SOURCE,
                    "sign_conflict_penalty": float(sign_conflict_penalty),
                    "sparse_top_k": int(V31_SPARSE_TOP_K),
                    "target_margin_floor": float(V31_TARGET_MARGIN_FLOOR),
                    "trust_norm_cap": float(trust_norm_cap),
                })
    return grid


def build_v32_support_tournament_margin_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for tournament_margin_weight in V32_TOURNAMENT_MARGIN_WEIGHT_GRID:
        for tournament_margin_floor in V32_TOURNAMENT_MARGIN_FLOOR_GRID:
            for compatible_orthogonal_weight in V32_COMPATIBLE_ORTHOGONAL_WEIGHT_GRID:
                grid.append({
                    "compatible_floor": float(V32_COMPATIBLE_FLOOR),
                    "compatible_orthogonal_weight": float(compatible_orthogonal_weight),
                    "config_index": len(grid),
                    "extra_compatible_weight": float(V32_EXTRA_COMPATIBLE_WEIGHT),
                    "hard_target_margin_weight": float(V32_HARD_TARGET_MARGIN_WEIGHT),
                    "matched_edit_source": V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE,
                    "sign_conflict_penalty": float(V32_SIGN_CONFLICT_PENALTY),
                    "sparse_top_k": int(V32_SPARSE_TOP_K),
                    "target_margin_floor": float(V32_TARGET_MARGIN_FLOOR),
                    "tournament_margin_floor": float(tournament_margin_floor),
                    "tournament_margin_weight": float(tournament_margin_weight),
                    "trust_norm_cap": float(V32_TRUST_NORM_CAP),
                })
    return grid


def build_v33_proof_gate_diagnostic_config_grid() -> list[dict[str, Any]]:
    base = {
        "compatible_floor": float(V32_COMPATIBLE_FLOOR),
        "extra_compatible_weight": float(V32_EXTRA_COMPATIBLE_WEIGHT),
        "hard_target_margin_weight": float(V32_HARD_TARGET_MARGIN_WEIGHT),
        "matched_edit_source": V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE,
        "sign_conflict_penalty": float(V32_SIGN_CONFLICT_PENALTY),
        "sparse_top_k": int(V32_SPARSE_TOP_K),
        "target_margin_floor": float(V32_TARGET_MARGIN_FLOOR),
        "tournament_margin_floor": 0.15,
        "tournament_margin_weight": 1.0,
        "trust_norm_cap": float(V32_TRUST_NORM_CAP),
    }
    return [
        {**base, "compatible_orthogonal_weight": 0.15, "config_index": 0},
        {**base, "compatible_orthogonal_weight": 0.05, "config_index": 1},
    ]


def build_v34_locality_pressure_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for trust_norm_cap in [0.5, 0.75, 1.0]:
        for extra_compatible_weight in [0.5, 2.0]:
            grid.append({
                "compatible_floor": float(V32_COMPATIBLE_FLOOR),
                "compatible_orthogonal_weight": 0.15,
                "config_index": len(grid),
                "extra_compatible_weight": float(extra_compatible_weight),
                "hard_target_margin_weight": float(V32_HARD_TARGET_MARGIN_WEIGHT),
                "matched_edit_source": V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE,
                "sign_conflict_penalty": float(V32_SIGN_CONFLICT_PENALTY),
                "sparse_top_k": int(V32_SPARSE_TOP_K),
                "target_margin_floor": float(V32_TARGET_MARGIN_FLOOR),
                "tournament_margin_floor": 0.15,
                "tournament_margin_weight": 1.0,
                "trust_norm_cap": float(trust_norm_cap),
            })
    return grid


def build_v35_support_source_line_search_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for trust_norm_cap in [1.0, 1.25]:
        for alpha_target_margin_floor in [0.05, 0.10]:
            grid.append({
                "alpha_candidates": [float(value) for value in V35_ALPHA_CANDIDATES],
                "alpha_target_margin_floor": float(alpha_target_margin_floor),
                "alpha_tournament_margin_floor": 0.0,
                "compatible_floor": float(V32_COMPATIBLE_FLOOR),
                "compatible_orthogonal_weight": 0.15,
                "config_index": len(grid),
                "extra_compatible_weight": float(V32_EXTRA_COMPATIBLE_WEIGHT),
                "fallback_target_penalty": 10.0,
                "fallback_tournament_penalty": 5.0,
                "hard_target_margin_weight": float(V32_HARD_TARGET_MARGIN_WEIGHT),
                "matched_edit_source": V35_SUPPORT_SOURCE_LINE_SEARCH_MATCHED_EDIT_SOURCE,
                "sign_conflict_penalty": float(V32_SIGN_CONFLICT_PENALTY),
                "sparse_top_k": int(V32_SPARSE_TOP_K),
                "target_margin_floor": float(V32_TARGET_MARGIN_FLOOR),
                "tournament_margin_floor": 0.15,
                "tournament_margin_weight": 1.0,
                "trust_norm_cap": float(trust_norm_cap),
            })
    return grid


def build_v36_compatible_nullspace_projection_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for compatible_nullspace_rtol in [1e-4, 1e-3]:
        for projection_strength in [0.75, 1.0]:
            grid.append({
                "alpha_candidates": [float(value) for value in V35_ALPHA_CANDIDATES],
                "alpha_target_margin_floor": 0.05,
                "alpha_tournament_margin_floor": 0.0,
                "compatible_floor": float(V32_COMPATIBLE_FLOOR),
                "compatible_nullspace_rtol": float(compatible_nullspace_rtol),
                "compatible_orthogonal_weight": 0.15,
                "config_index": len(grid),
                "extra_compatible_weight": float(V32_EXTRA_COMPATIBLE_WEIGHT),
                "fallback_target_penalty": 10.0,
                "fallback_tournament_penalty": 5.0,
                "hard_target_margin_weight": float(V32_HARD_TARGET_MARGIN_WEIGHT),
                "matched_edit_source": V36_COMPATIBLE_NULLSPACE_MATCHED_EDIT_SOURCE,
                "projection_strength": float(projection_strength),
                "sign_conflict_penalty": float(V32_SIGN_CONFLICT_PENALTY),
                "sparse_top_k": int(V32_SPARSE_TOP_K),
                "target_margin_floor": float(V32_TARGET_MARGIN_FLOOR),
                "tournament_margin_floor": 0.15,
                "tournament_margin_weight": 1.0,
                "trust_norm_cap": float(V32_TRUST_NORM_CAP),
            })
    return grid


def build_v37_projected_support_optimizer_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for compatible_nullspace_rtol in [1e-3, 1e-2]:
        for projection_strength in [0.5, 0.75]:
            grid.append({
                "alpha_candidates": [float(value) for value in V35_ALPHA_CANDIDATES],
                "alpha_target_margin_floor": 0.05,
                "alpha_tournament_margin_floor": 0.0,
                "compatible_floor": float(V32_COMPATIBLE_FLOOR),
                "compatible_nullspace_rtol": float(compatible_nullspace_rtol),
                "compatible_orthogonal_weight": 0.05,
                "config_index": len(grid),
                "extra_compatible_weight": 0.05,
                "fallback_target_penalty": 10.0,
                "fallback_tournament_penalty": 5.0,
                "hard_target_margin_weight": float(V32_HARD_TARGET_MARGIN_WEIGHT),
                "matched_edit_source": V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE,
                "projected_optimizer_epochs": 80,
                "projected_optimizer_lr": float(V29_BREADTH_FIRST_LR),
                "projection_strength": float(projection_strength),
                "sign_conflict_penalty": float(V32_SIGN_CONFLICT_PENALTY),
                "sparse_top_k": int(V32_SPARSE_TOP_K),
                "target_margin_floor": float(V32_TARGET_MARGIN_FLOOR),
                "tournament_margin_floor": 0.15,
                "tournament_margin_weight": 1.0,
                "trust_norm_cap": float(V32_TRUST_NORM_CAP),
            })
    return grid


def build_v38_compatible_mse_gated_projected_optimizer_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    base = build_v37_projected_support_optimizer_config_grid()[2]
    for compatible_mse_gate in [5.0, 15.0]:
        for compatible_gate_weight in [0.5, 1.5]:
            grid.append({
                **dict(base),
                "alpha_compatible_mse_gate": float(compatible_mse_gate),
                "compatible_gate_weight": float(compatible_gate_weight),
                "compatible_mse_gate": float(compatible_mse_gate),
                "config_index": len(grid),
                "matched_edit_source": V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE,
                "projected_optimizer_event_prefix": "v38_compatible_gated_optimizer",
            })
    return grid


def build_v39_target_feasible_lexicographic_optimizer_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    base = build_v37_projected_support_optimizer_config_grid()[2]
    for compatible_mse_soft_gate in [10.0, 20.0]:
        for compatible_gate_weight in [0.25, 0.75]:
            grid.append({
                **dict(base),
                "alpha_compatible_mse_soft_gate": float(compatible_mse_soft_gate),
                "compatible_gate_weight": float(compatible_gate_weight),
                "compatible_mse_gate": float(compatible_mse_soft_gate),
                "config_index": len(grid),
                "matched_edit_source": V39_TARGET_FEASIBLE_LEXICOGRAPHIC_MATCHED_EDIT_SOURCE,
                "projected_optimizer_event_prefix": (
                    "v39_target_feasible_lexicographic_optimizer"
                ),
            })
    return grid


def build_v40_target_tolerance_locality_budget_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    base = build_v37_projected_support_optimizer_config_grid()[2]
    for compatible_mse_soft_gate in [10.0, 20.0]:
        for target_rank_score_tolerance in [0.05, 0.15]:
            grid.append({
                **dict(base),
                "alpha_compatible_mse_soft_gate": float(compatible_mse_soft_gate),
                "compatible_gate_weight": 0.25,
                "compatible_mse_gate": float(compatible_mse_soft_gate),
                "config_index": len(grid),
                "matched_edit_source": (
                    V40_TARGET_TOLERANCE_LOCALITY_BUDGET_MATCHED_EDIT_SOURCE
                ),
                "projected_optimizer_event_prefix": (
                    "v40_target_tolerance_locality_budget_optimizer"
                ),
                "target_rank_score_tolerance": float(target_rank_score_tolerance),
            })
    return grid


def build_v41_trajectory_frontier_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    base = build_v37_projected_support_optimizer_config_grid()[2]
    for compatible_mse_soft_gate in [10.0, 20.0]:
        for target_rank_score_tolerance in [0.05, 0.15]:
            grid.append({
                **dict(base),
                "alpha_compatible_mse_soft_gate": float(compatible_mse_soft_gate),
                "compatible_gate_weight": 0.25,
                "compatible_mse_gate": float(compatible_mse_soft_gate),
                "config_index": len(grid),
                "matched_edit_source": V41_TRAJECTORY_FRONTIER_MATCHED_EDIT_SOURCE,
                "projected_optimizer_event_prefix": "v41_trajectory_frontier_optimizer",
                "target_rank_score_tolerance": float(target_rank_score_tolerance),
                "trajectory_frontier_enabled": True,
                "trajectory_frontier_event_prefix": "v41_trajectory_frontier",
            })
    return grid


def build_v42_compatible_dual_frontier_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    base = build_v41_trajectory_frontier_config_grid()[0]
    for compatible_mse_budget in [10.0, 20.0]:
        for compatible_augmented_weight in [0.5, 2.0]:
            grid.append({
                **dict(base),
                "alpha_compatible_mse_soft_gate": float(compatible_mse_budget),
                "compatible_augmented_weight": float(compatible_augmented_weight),
                "compatible_dual_initial": 0.0,
                "compatible_dual_lr": 0.05,
                "compatible_dual_max": 100.0,
                "compatible_gate_weight": 0.0,
                "compatible_mse_budget": float(compatible_mse_budget),
                "compatible_mse_gate": float(compatible_mse_budget),
                "config_index": len(grid),
                "experiment_variant": V42_EXPERIMENT_VARIANT,
                "matched_edit_source": V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE,
                "projected_optimizer_event_prefix": (
                    "v42_compatible_dual_frontier_optimizer"
                ),
                "target_rank_score_tolerance": 0.15,
                "trajectory_frontier_enabled": True,
                "trajectory_frontier_event_prefix": "v42_compatible_dual_frontier",
                "v42_compatible_dual_enabled": True,
            })
    return grid


def select_v25_inner_validation_configs(
    *,
    grid_name: str,
    max_configs: int | None,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
) -> list[dict[str, Any]]:
    name = str(grid_name)
    if name in {"v25", "v26"}:
        configs = build_v25_config_grid()
    elif name == "v27-localized":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v27_localized_behavior_loss_config_grid()
        ]
    elif name == "v28-anchor-nullspace":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v28_anchor_nullspace_config_grid()
        ]
    elif name == "v29-breadth-first-sparse":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v29_breadth_first_sparse_config_grid()
        ]
    elif name == "v30-margin-gated-sparse":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v30_margin_gated_sparse_config_grid()
        ]
    elif name == "v31-orthogonal-sign-sparse":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v31_orthogonal_sign_sparse_config_grid()
        ]
    elif name == "v32-support-tournament-margin":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v32_support_tournament_margin_config_grid()
        ]
    elif name == "v33-proof-gate-diagnostic":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v33_proof_gate_diagnostic_config_grid()
        ]
    elif name == "v34-locality-pressure":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v34_locality_pressure_config_grid()
        ]
    elif name == "v35-support-source-line-search":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v35_support_source_line_search_config_grid()
        ]
    elif name == "v36-compatible-nullspace-projection":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v36_compatible_nullspace_projection_config_grid()
        ]
    elif name == "v37-projected-support-optimizer":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v37_projected_support_optimizer_config_grid()
        ]
    elif name == "v38-compatible-mse-gated-projected-optimizer":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v38_compatible_mse_gated_projected_optimizer_config_grid()
        ]
    elif name == "v39-target-feasible-lexicographic-projected-optimizer":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v39_target_feasible_lexicographic_optimizer_config_grid()
        ]
    elif name == "v40-target-tolerance-locality-budget-projected-optimizer":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v40_target_tolerance_locality_budget_config_grid()
        ]
    elif name == "v41-trajectory-frontier-projected-optimizer":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v41_trajectory_frontier_config_grid()
        ]
    elif name == "v42-compatible-dual-frontier":
        checked_train_file = require_sha256_hex(
            train_pool_file_sha256,
            field_name="train_pool_file_sha256",
        )
        checked_train_summary = require_sha256_hex(
            train_pool_summary_hash,
            field_name="train_pool_summary_hash",
        )
        configs = [
            {
                **dict(config),
                "train_pool_file_sha256": checked_train_file,
                "train_pool_summary_hash": checked_train_summary,
            }
            for config in build_v42_compatible_dual_frontier_config_grid()
        ]
    else:
        raise ValueError(f"unknown inner validation config grid: {grid_name}")
    if max_configs is not None:
        limit = int(max_configs)
        if limit <= 0:
            raise ValueError("inner_validation_max_configs must be positive")
        configs = configs[:limit]
    return configs


def v27_localized_optimization_boundary() -> dict[str, Any]:
    return {
        "allows_heldout_optimization": False,
        "optimization_split": "support",
        "proof_split": "heldout",
        "support_objective_is_proof_metric": False,
    }


def v28_anchor_nullspace_optimization_boundary() -> dict[str, Any]:
    return {
        "allows_heldout_optimization": False,
        "optimization_split": "support",
        "proof_split": "heldout",
        "support_objective_is_proof_metric": False,
    }


def v29_breadth_first_sparse_optimization_boundary() -> dict[str, Any]:
    return {
        "allows_heldout_optimization": False,
        "optimization_split": "support",
        "proof_split": "heldout",
        "support_objective_is_proof_metric": False,
    }


def v30_margin_gated_optimization_boundary() -> dict[str, Any]:
    return {
        "allows_heldout_optimization": False,
        "optimization_split": "support",
        "proof_split": "heldout",
        "support_objective_is_proof_metric": False,
    }


def v31_orthogonal_sign_optimization_boundary() -> dict[str, Any]:
    return {
        "allows_heldout_optimization": False,
        "optimization_split": "support",
        "proof_split": "heldout",
        "support_objective_is_proof_metric": False,
    }


def v32_support_tournament_optimization_boundary() -> dict[str, Any]:
    return {
        "allows_heldout_optimization": False,
        "optimization_split": "support",
        "proof_split": "heldout",
        "support_objective_is_proof_metric": False,
    }


def v35_support_source_line_search_optimization_boundary() -> dict[str, Any]:
    return {
        "allows_heldout_optimization": False,
        "optimization_split": "support",
        "proof_split": "heldout",
        "support_objective_is_proof_metric": False,
        "support_selection": "alpha_line_search",
    }


def v36_compatible_nullspace_optimization_boundary() -> dict[str, Any]:
    return {
        "allows_heldout_optimization": False,
        "optimization_split": "support",
        "proof_split": "heldout",
        "support_objective_is_proof_metric": False,
        "support_projection": "compatible_logit_jacobian_nullspace",
    }


def v37_projected_optimizer_optimization_boundary() -> dict[str, Any]:
    return {
        "allows_heldout_optimization": False,
        "optimization_split": "support",
        "proof_split": "heldout",
        "support_objective_is_proof_metric": False,
        "support_optimizer": "projected_target_tournament_loss",
        "support_projection": "compatible_logit_jacobian_nullspace",
    }


def v25_config_requires_spectral_basis(config: Mapping[str, Any]) -> bool:
    if str(config.get("matched_edit_source", "")) == V27_LOCALIZED_MATCHED_EDIT_SOURCE:
        return str(config.get("localized_basis", "")) in {
            "combined_spectral_gradient_rank8",
            "spectral_train_delta_rank4",
        }
    if str(config.get("matched_edit_source", "")) == V28_ANCHOR_NULLSPACE_MATCHED_EDIT_SOURCE:
        return False
    if str(config.get("matched_edit_source", "")) == V29_BREADTH_FIRST_SPARSE_MATCHED_EDIT_SOURCE:
        return False
    if str(config.get("matched_edit_source", "")) == V30_MARGIN_GATED_SPARSE_MATCHED_EDIT_SOURCE:
        return False
    if str(config.get("matched_edit_source", "")) == V31_ORTHOGONAL_SIGN_SPARSE_MATCHED_EDIT_SOURCE:
        return False
    if str(config.get("matched_edit_source", "")) == V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE:
        return False
    if str(config.get("matched_edit_source", "")) == (
        V35_SUPPORT_SOURCE_LINE_SEARCH_MATCHED_EDIT_SOURCE
    ):
        return False
    if str(config.get("matched_edit_source", "")) == (
        V36_COMPATIBLE_NULLSPACE_MATCHED_EDIT_SOURCE
    ):
        return False
    if str(config.get("matched_edit_source", "")) == (
        V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE
    ):
        return False
    if str(config.get("matched_edit_source", "")) == (
        V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE
    ):
        return False
    if str(config.get("matched_edit_source", "")) == (
        V39_TARGET_FEASIBLE_LEXICOGRAPHIC_MATCHED_EDIT_SOURCE
    ):
        return False
    if str(config.get("matched_edit_source", "")) == (
        V40_TARGET_TOLERANCE_LOCALITY_BUDGET_MATCHED_EDIT_SOURCE
    ):
        return False
    if str(config.get("matched_edit_source", "")) == (
        V41_TRAJECTORY_FRONTIER_MATCHED_EDIT_SOURCE
    ):
        return False
    if str(config.get("matched_edit_source", "")) == (
        V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
    ):
        return False
    return str(config["projection"]) in {"spectral_rank4", "rank1_spectral_rank4"}


def experiment_variant_for_inner_validation_grid(grid_name: str) -> str:
    if str(grid_name) == "v27-localized":
        return V27_EXPERIMENT_VARIANT
    if str(grid_name) == "v28-anchor-nullspace":
        return V28_EXPERIMENT_VARIANT
    if str(grid_name) == "v29-breadth-first-sparse":
        return V29_EXPERIMENT_VARIANT
    if str(grid_name) == "v30-margin-gated-sparse":
        return V30_EXPERIMENT_VARIANT
    if str(grid_name) == "v31-orthogonal-sign-sparse":
        return V31_EXPERIMENT_VARIANT
    if str(grid_name) == "v32-support-tournament-margin":
        return V32_EXPERIMENT_VARIANT
    if str(grid_name) == "v33-proof-gate-diagnostic":
        return V33_EXPERIMENT_VARIANT
    if str(grid_name) == "v34-locality-pressure":
        return V34_EXPERIMENT_VARIANT
    if str(grid_name) == "v35-support-source-line-search":
        return V35_EXPERIMENT_VARIANT
    if str(grid_name) == "v36-compatible-nullspace-projection":
        return V36_EXPERIMENT_VARIANT
    if str(grid_name) == "v37-projected-support-optimizer":
        return V37_EXPERIMENT_VARIANT
    if str(grid_name) == "v38-compatible-mse-gated-projected-optimizer":
        return V38_EXPERIMENT_VARIANT
    if str(grid_name) == "v39-target-feasible-lexicographic-projected-optimizer":
        return V39_EXPERIMENT_VARIANT
    if str(grid_name) == "v40-target-tolerance-locality-budget-projected-optimizer":
        return V40_EXPERIMENT_VARIANT
    if str(grid_name) == "v41-trajectory-frontier-projected-optimizer":
        return V41_EXPERIMENT_VARIANT
    if str(grid_name) == "v42-compatible-dual-frontier":
        return V42_EXPERIMENT_VARIANT
    return V26_EXPERIMENT_VARIANT


def experiment_variant_for_config(config: Mapping[str, Any]) -> str:
    if str(config.get("matched_edit_source", "")) == V27_LOCALIZED_MATCHED_EDIT_SOURCE:
        return V27_EXPERIMENT_VARIANT
    if str(config.get("matched_edit_source", "")) == V28_ANCHOR_NULLSPACE_MATCHED_EDIT_SOURCE:
        return V28_EXPERIMENT_VARIANT
    if str(config.get("matched_edit_source", "")) == V29_BREADTH_FIRST_SPARSE_MATCHED_EDIT_SOURCE:
        return V29_EXPERIMENT_VARIANT
    if str(config.get("matched_edit_source", "")) == V30_MARGIN_GATED_SPARSE_MATCHED_EDIT_SOURCE:
        return V30_EXPERIMENT_VARIANT
    if str(config.get("matched_edit_source", "")) == V31_ORTHOGONAL_SIGN_SPARSE_MATCHED_EDIT_SOURCE:
        return V31_EXPERIMENT_VARIANT
    if str(config.get("matched_edit_source", "")) == V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE:
        return V32_EXPERIMENT_VARIANT
    if str(config.get("matched_edit_source", "")) == (
        V35_SUPPORT_SOURCE_LINE_SEARCH_MATCHED_EDIT_SOURCE
    ):
        return V35_EXPERIMENT_VARIANT
    if str(config.get("matched_edit_source", "")) == (
        V36_COMPATIBLE_NULLSPACE_MATCHED_EDIT_SOURCE
    ):
        return V36_EXPERIMENT_VARIANT
    if str(config.get("matched_edit_source", "")) == (
        V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE
    ):
        return V37_EXPERIMENT_VARIANT
    if str(config.get("matched_edit_source", "")) == (
        V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE
    ):
        return V38_EXPERIMENT_VARIANT
    if str(config.get("matched_edit_source", "")) == (
        V39_TARGET_FEASIBLE_LEXICOGRAPHIC_MATCHED_EDIT_SOURCE
    ):
        return V39_EXPERIMENT_VARIANT
    if str(config.get("matched_edit_source", "")) == (
        V40_TARGET_TOLERANCE_LOCALITY_BUDGET_MATCHED_EDIT_SOURCE
    ):
        return V40_EXPERIMENT_VARIANT
    if str(config.get("matched_edit_source", "")) == (
        V41_TRAJECTORY_FRONTIER_MATCHED_EDIT_SOURCE
    ):
        return V41_EXPERIMENT_VARIANT
    if str(config.get("matched_edit_source", "")) == (
        V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
    ):
        return V42_EXPERIMENT_VARIANT
    return V26_EXPERIMENT_VARIANT


def v25_native_control_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if str(config.get("matched_edit_source", "")) == V27_LOCALIZED_MATCHED_EDIT_SOURCE:
        return dict(V27_LOCALIZED_NATIVE_CONTROL_CONFIG)
    if str(config.get("matched_edit_source", "")) == V28_ANCHOR_NULLSPACE_MATCHED_EDIT_SOURCE:
        return dict(V28_ANCHOR_NULLSPACE_NATIVE_CONTROL_CONFIG)
    if str(config.get("matched_edit_source", "")) == V29_BREADTH_FIRST_SPARSE_MATCHED_EDIT_SOURCE:
        return dict(V29_BREADTH_FIRST_SPARSE_NATIVE_CONTROL_CONFIG)
    if str(config.get("matched_edit_source", "")) == V30_MARGIN_GATED_SPARSE_MATCHED_EDIT_SOURCE:
        return dict(V30_MARGIN_GATED_NATIVE_CONTROL_CONFIG)
    if str(config.get("matched_edit_source", "")) == V31_ORTHOGONAL_SIGN_SPARSE_MATCHED_EDIT_SOURCE:
        return dict(V31_ORTHOGONAL_SIGN_NATIVE_CONTROL_CONFIG)
    if str(config.get("matched_edit_source", "")) == V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE:
        return dict(V32_SUPPORT_TOURNAMENT_NATIVE_CONTROL_CONFIG)
    if str(config.get("matched_edit_source", "")) == (
        V35_SUPPORT_SOURCE_LINE_SEARCH_MATCHED_EDIT_SOURCE
    ):
        return dict(V32_SUPPORT_TOURNAMENT_NATIVE_CONTROL_CONFIG)
    if str(config.get("matched_edit_source", "")) == (
        V36_COMPATIBLE_NULLSPACE_MATCHED_EDIT_SOURCE
    ):
        return dict(V32_SUPPORT_TOURNAMENT_NATIVE_CONTROL_CONFIG)
    if str(config.get("matched_edit_source", "")) == (
        V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE
    ):
        return dict(V32_SUPPORT_TOURNAMENT_NATIVE_CONTROL_CONFIG)
    if str(config.get("matched_edit_source", "")) == (
        V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE
    ):
        return dict(V32_SUPPORT_TOURNAMENT_NATIVE_CONTROL_CONFIG)
    if str(config.get("matched_edit_source", "")) == (
        V39_TARGET_FEASIBLE_LEXICOGRAPHIC_MATCHED_EDIT_SOURCE
    ):
        return dict(V32_SUPPORT_TOURNAMENT_NATIVE_CONTROL_CONFIG)
    if str(config.get("matched_edit_source", "")) == (
        V40_TARGET_TOLERANCE_LOCALITY_BUDGET_MATCHED_EDIT_SOURCE
    ):
        return dict(V32_SUPPORT_TOURNAMENT_NATIVE_CONTROL_CONFIG)
    if str(config.get("matched_edit_source", "")) == (
        V41_TRAJECTORY_FRONTIER_MATCHED_EDIT_SOURCE
    ):
        return dict(V32_SUPPORT_TOURNAMENT_NATIVE_CONTROL_CONFIG)
    if str(config.get("matched_edit_source", "")) == (
        V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
    ):
        return dict(V32_SUPPORT_TOURNAMENT_NATIVE_CONTROL_CONFIG)
    return dict(config)


def v25_spectral_seed_config(config: Mapping[str, Any]) -> dict[str, Any]:
    seed = v25_native_control_config(config)
    projection = str(seed["projection"])
    if projection == "spectral_rank4":
        seed["projection"] = "none"
    elif projection == "rank1_spectral_rank4":
        seed["projection"] = "rank1"
    elif projection not in {"none", "rank1"}:
        raise ValueError(f"unknown projection: {projection}")
    return seed


def build_v25_successive_halving_plan(
    *,
    configs: Sequence[Mapping[str, Any]],
    rung_job_counts: Sequence[int],
    keep_fractions: Sequence[float],
) -> dict[str, Any]:
    config_count = len(configs)
    if config_count <= 0:
        raise ValueError("successive halving requires at least one config")
    if len(rung_job_counts) <= 0:
        raise ValueError("successive halving requires at least one rung")
    if len(rung_job_counts) != len(keep_fractions):
        raise ValueError("rung_job_counts and keep_fractions must have equal length")
    active_config_count = int(config_count)
    rungs = []
    for rung_index, (rung_job_count, keep_fraction) in enumerate(
        zip(rung_job_counts, keep_fractions)
    ):
        if int(rung_job_count) <= 0:
            raise ValueError("rung job counts must be positive")
        if float(keep_fraction) <= 0.0 or float(keep_fraction) > 1.0:
            raise ValueError("keep fractions must be in (0, 1]")
        keep_config_count = max(1, math.ceil(active_config_count * float(keep_fraction)))
        rungs.append({
            "input_config_count": int(active_config_count),
            "keep_config_count": int(keep_config_count),
            "rung_index": int(rung_index),
            "rung_job_count": int(rung_job_count),
        })
        active_config_count = min(active_config_count, keep_config_count)
    config_hashes = [
        stable_hash_json(dict(config))
        for config in configs
    ]
    plan_hash = stable_hash_json({
        "config_hashes": config_hashes,
        "rungs": rungs,
        "scope": "v25_successive_halving_inner_validation_plan",
    })
    return {
        "config_count": int(config_count),
        "config_grid_hash": stable_hash_json([dict(config) for config in configs]),
        "plan_hash": plan_hash,
        "rung_count": len(rungs),
        "rungs": rungs,
    }


def stable_torch_seed(payload: Mapping[str, Any]) -> int:
    return int(stable_hash_json(payload)[:16], 16) % (2**31)


def require_sha256_hex(value: Any, *, field_name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text.lower()):
        raise ValueError(f"{field_name} must be a SHA-256 hex string")
    return text


def sign_canonicalize_basis_columns(basis: torch.Tensor) -> torch.Tensor:
    canonical = basis.detach().clone().to(dtype=torch.float32, device="cpu")
    if canonical.ndim != 2:
        raise ValueError("basis must be a matrix")
    for column_index in range(int(canonical.shape[1])):
        column = canonical[:, column_index]
        max_index = int(torch.argmax(torch.abs(column)).item())
        if float(column[max_index].item()) < 0.0:
            canonical[:, column_index] = -column
    return canonical


def validate_v27_localized_basis_matrix(basis: torch.Tensor) -> torch.Tensor:
    matrix = basis.detach().clone().to(dtype=torch.float32, device="cpu")
    if matrix.ndim != 2:
        raise ValueError("localized basis must be a matrix")
    if int(matrix.shape[0]) != SOURCE_WEIGHT_DIM:
        raise ValueError("localized basis has wrong source dimension")
    if int(matrix.shape[1]) <= 0:
        raise ValueError("localized basis rank must be positive")
    if not torch.isfinite(matrix).all():
        raise ValueError("nonfinite localized basis")
    return sign_canonicalize_basis_columns(matrix)


def v27_subject_logits_for_inputs(weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return v23.v16.v15.v10.decoder_v1.subject_forward_flat_batch(
        weights.reshape(1, -1).to(dtype=torch.float32),
        inputs.to(dtype=torch.float32),
    )[0]


def v27_support_tensors_for_source_target(
    *,
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
) -> Mapping[str, torch.Tensor]:
    return v23.v16.v15.v14.prepare_support_tensors_with_source_logits(
        source_weights=source_weights.detach().clone().to(dtype=torch.float32),
        source=str(source_behavior),
        target=str(target_behavior),
    )


def v27_support_gradient_rows(
    *,
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    source_mse_weight: float = 1.0,
) -> dict[str, Any]:
    source = source_weights.detach().clone().to(dtype=torch.float32).reshape(-1)
    support = v27_support_tensors_for_source_target(
        source_weights=source,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    rows: list[torch.Tensor] = []
    row_names: list[str] = []
    gradient_by_group: dict[str, torch.Tensor] = {}

    def append_grad(
        loss: torch.Tensor,
        name: str,
        flat: torch.Tensor,
        *,
        row_scale: float = 1.0,
    ) -> None:
        grad = torch.autograd.grad(loss, flat, retain_graph=False, allow_unused=False)[0]
        grad = grad.detach().clone().to(dtype=torch.float32, device="cpu").reshape(-1)
        if not torch.isfinite(grad).all():
            raise ValueError(f"nonfinite localized gradient row: {name}")
        scaled_grad = grad * float(row_scale)
        if not torch.isfinite(scaled_grad).all():
            raise ValueError(f"nonfinite localized scaled gradient row: {name}")
        gradient_by_group[name] = grad
        rows.append(scaled_grad)
        row_names.append(name)

    flat = source.detach().clone().requires_grad_(True)
    target_logits = v27_subject_logits_for_inputs(flat, support["target_inputs"])
    append_grad(
        F.binary_cross_entropy_with_logits(
            target_logits,
            support["target_labels"].to(dtype=torch.float32),
        ),
        "target_positive",
        flat,
    )

    flat = source.detach().clone().requires_grad_(True)
    conflict_logits = v27_subject_logits_for_inputs(flat, support["conflict_inputs"])
    append_grad(
        F.binary_cross_entropy_with_logits(
            conflict_logits,
            support["conflict_target_labels"].to(dtype=torch.float32),
        ),
        "conflict",
        flat,
    )

    flat = source.detach().clone().requires_grad_(True)
    compatible_logits = v27_subject_logits_for_inputs(flat, support["compatible_inputs"])
    append_grad(
        F.mse_loss(
            compatible_logits,
            support["compatible_source_logits"].to(dtype=torch.float32),
        ),
        "compatible",
        flat,
        row_scale=source_mse_weight,
    )

    source_l2 = torch.zeros_like(source)
    gradient_by_group["source_l2"] = source_l2
    rows.append(source_l2)
    row_names.append("source_l2")

    if not rows:
        raise ValueError("no nonzero localized gradient rows")
    return {
        "gradient_by_group": gradient_by_group,
        "gradient_rows": torch.stack(rows, dim=0).to(dtype=torch.float32),
        "row_count_by_group": {
            "compatible": 1,
            "conflict": 1,
            "source_l2": 1,
            "target_positive": 1,
        },
        "row_names": row_names,
        "support_split_counts": {
            "compatible": int(support["compatible_inputs"].shape[0]),
            "conflict": int(support["conflict_inputs"].shape[0]),
            "target": int(support["target_inputs"].shape[0]),
        },
    }


def v27_svd_basis_from_rows(rows: torch.Tensor, *, rank: int) -> torch.Tensor:
    matrix = rows.detach().clone().to(dtype=torch.float32, device="cpu")
    if matrix.ndim != 2:
        raise ValueError("gradient rows must be a matrix")
    if not torch.isfinite(matrix).all():
        raise ValueError("nonfinite localized gradient rows")
    centered = matrix - matrix.mean(dim=1, keepdim=True)
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    max_rank = min(int(rank), int(vh.shape[0]), int(vh.shape[1]))
    if max_rank <= 0:
        raise ValueError("localized gradient basis rank must be positive")
    return validate_v27_localized_basis_matrix(vh[:max_rank].T)


def build_v27_localized_behavior_loss_basis(
    *,
    config: Mapping[str, Any],
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    spectral_basis: torch.Tensor | None,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
) -> dict[str, Any]:
    basis_name = str(config.get("localized_basis", ""))
    require_sha256_hex(train_pool_file_sha256, field_name="train_pool_file_sha256")
    require_sha256_hex(train_pool_summary_hash, field_name="train_pool_summary_hash")
    source = source_weights.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(source.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("source weights have wrong dimension")
    if not torch.isfinite(source).all():
        raise ValueError("nonfinite source weights")
    source_mse_weight = float(config.get("localized_source_mse_weight", 1.0))
    support_summary: Mapping[str, Any] = {}

    if basis_name == "spectral_train_delta_rank4":
        if spectral_basis is None:
            raise ValueError("spectral basis is required for localized spectral basis")
        basis = validate_v27_localized_basis_matrix(spectral_basis[:, :4])
    elif basis_name == "target_source_logit_gradient_rank4":
        gradient_info = v27_support_gradient_rows(
            source_weights=source,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            source_mse_weight=source_mse_weight,
        )
        support_summary = {
            "row_names_hash": stable_hash_json(list(gradient_info["row_names"])),
            "support_split_counts": dict(gradient_info["support_split_counts"]),
        }
        basis = v27_svd_basis_from_rows(gradient_info["gradient_rows"], rank=4)
    elif basis_name == "combined_spectral_gradient_rank8":
        if spectral_basis is None:
            raise ValueError("spectral basis is required for localized combined basis")
        spectral = validate_v27_localized_basis_matrix(spectral_basis[:, :4])
        gradient_info = v27_support_gradient_rows(
            source_weights=source,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            source_mse_weight=source_mse_weight,
        )
        gradient = v27_svd_basis_from_rows(gradient_info["gradient_rows"], rank=4)
        q, _r = torch.linalg.qr(torch.cat([spectral, gradient], dim=1), mode="reduced")
        basis = validate_v27_localized_basis_matrix(q[:, : min(8, q.shape[1])])
        support_summary = {
            "row_names_hash": stable_hash_json(list(gradient_info["row_names"])),
            "support_split_counts": dict(gradient_info["support_split_counts"]),
        }
    elif basis_name == "output_layer_topk":
        gradient_info = v27_support_gradient_rows(
            source_weights=source,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            source_mse_weight=source_mse_weight,
        )
        gradient_by_group = dict(gradient_info["gradient_by_group"])
        gradient = (
            torch.as_tensor(
                gradient_by_group["target_positive"],
                dtype=torch.float32,
            ).reshape(-1)
            - source_mse_weight
            * torch.as_tensor(
                gradient_by_group["compatible"],
                dtype=torch.float32,
            ).reshape(-1)
        )
        start = int(v23.v16.OUTPUT_WEIGHT_START)
        end = int(v23.v16.OUTPUT_BIAS_INDEX + 1)
        output_slice = gradient[start:end]
        order = sorted(
            range(int(output_slice.numel())),
            key=lambda index: (-float(abs(output_slice[index]).item()), index),
        )[:8]
        if not order:
            raise ValueError("output layer top-k basis is empty")
        basis = torch.zeros(SOURCE_WEIGHT_DIM, len(order), dtype=torch.float32)
        for column_index, local_index in enumerate(order):
            basis[start + int(local_index), column_index] = 1.0
        basis = validate_v27_localized_basis_matrix(basis)
        support_summary = {
            "selected_coordinate_hash": stable_hash_json([
                start + int(local_index) for local_index in order
            ]),
            "support_split_counts": dict(gradient_info["support_split_counts"]),
        }
    else:
        raise ValueError(f"unknown localized basis: {basis_name}")

    basis_hash = stable_hash_json(tensor_to_hashable(basis))
    audit = {
        "basis_hash": basis_hash,
        "basis_rank": int(basis.shape[1]),
        "basis_type": basis_name,
        "experiment_variant": V27_EXPERIMENT_VARIANT,
        "script_sha256": str(script_sha256),
        "source_behavior": str(source_behavior),
        "target_behavior": str(target_behavior),
        "train_pool_file_sha256": str(train_pool_file_sha256),
        "train_pool_summary_hash": str(train_pool_summary_hash),
        **dict(support_summary),
    }
    audit["basis_provenance_hash"] = stable_hash_json(audit)
    return {
        "audit": audit,
        "basis": basis,
    }


def redact_v28_anchor_nullspace_basis_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_hash_keys = {
        "basis_hash",
        "record_id_hash",
        "selected_config_hash",
        "selected_coordinate_hash",
    }
    allowed_int_keys = {
        "anchor_count",
        "basis_rank",
        "compatible_count",
        "conflict_count",
        "jacobian_row_count",
        "preserve_rank",
        "target_count",
    }
    allowed_float_keys = {
        "compatible_energy_ratio",
        "compatible_floor",
        "nullspace_rtol",
    }
    redacted: dict[str, Any] = {}
    for key in allowed_hash_keys:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    if "failure_reason" in payload:
        redacted["failure_reason"] = str(payload["failure_reason"])
    finite_values = [
        value for key, value in redacted.items()
        if key in allowed_float_keys
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def v28_anchor_gradients_and_compatible_jacobian(
    *,
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
) -> dict[str, Any]:
    source = source_weights.detach().clone().to(dtype=torch.float32).reshape(-1)
    support = v27_support_tensors_for_source_target(
        source_weights=source,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )

    def loss_grad(loss_fn: Any) -> torch.Tensor:
        flat = source.detach().clone().requires_grad_(True)
        loss = loss_fn(flat)
        grad = torch.autograd.grad(loss, flat, retain_graph=False, allow_unused=False)[0]
        grad = grad.detach().clone().to(dtype=torch.float32, device="cpu").reshape(-1)
        if not torch.isfinite(grad).all():
            raise ValueError("nonfinite anchor gradient")
        return grad

    g_target = loss_grad(lambda flat: F.binary_cross_entropy_with_logits(
        v27_subject_logits_for_inputs(flat, support["target_inputs"]),
        support["target_labels"].to(dtype=torch.float32),
    ))
    g_conflict = loss_grad(lambda flat: F.binary_cross_entropy_with_logits(
        v27_subject_logits_for_inputs(flat, support["conflict_inputs"]),
        support["conflict_target_labels"].to(dtype=torch.float32),
    ))
    g_compatible = loss_grad(lambda flat: F.mse_loss(
        v27_subject_logits_for_inputs(flat, support["compatible_inputs"]),
        support["compatible_source_logits"].to(dtype=torch.float32),
    ))

    flat = source.detach().clone().requires_grad_(True)
    compatible_logits = v27_subject_logits_for_inputs(flat, support["compatible_inputs"])
    jacobian_rows: list[torch.Tensor] = []
    for logit_index in range(int(compatible_logits.numel())):
        grad = torch.autograd.grad(
            compatible_logits[logit_index],
            flat,
            retain_graph=logit_index < int(compatible_logits.numel()) - 1,
            allow_unused=False,
        )[0]
        grad = grad.detach().clone().to(dtype=torch.float32, device="cpu").reshape(-1)
        if not torch.isfinite(grad).all():
            raise ValueError("nonfinite compatible jacobian row")
        jacobian_rows.append(grad)
    if not jacobian_rows:
        raise ValueError("compatible jacobian requires at least one row")
    jacobian = torch.stack(jacobian_rows, dim=0).to(dtype=torch.float32)
    return {
        "g_compatible": g_compatible,
        "g_conflict": g_conflict,
        "g_target": g_target,
        "compatible_jacobian": jacobian,
        "support_split_counts": {
            "compatible": int(support["compatible_inputs"].shape[0]),
            "conflict": int(support["conflict_inputs"].shape[0]),
            "target": int(support["target_inputs"].shape[0]),
        },
    }


def select_v28_anchor_coordinates(
    *,
    source_weights: torch.Tensor,
    g_target: torch.Tensor,
    g_conflict: torch.Tensor,
    g_compatible: torch.Tensor,
    anchor_count: int,
    compatible_floor: float,
    conflict_weight: float = V28_CONFLICT_WEIGHT,
) -> list[int]:
    source = source_weights.detach().clone().to(dtype=torch.float32).reshape(-1)
    target = g_target.detach().clone().to(dtype=torch.float32).reshape(-1)
    conflict = g_conflict.detach().clone().to(dtype=torch.float32).reshape(-1)
    compatible = g_compatible.detach().clone().to(dtype=torch.float32).reshape(-1)
    if not all(int(value.numel()) == SOURCE_WEIGHT_DIM for value in [source, target, conflict, compatible]):
        raise ValueError("anchor score inputs have wrong dimension")
    if not all(torch.isfinite(value).all() for value in [source, target, conflict, compatible]):
        raise ValueError("nonfinite anchor score input")
    count = int(anchor_count)
    if count <= 0:
        raise ValueError("anchor_count must be positive")
    floor = float(compatible_floor)
    if floor <= 0.0 or not math.isfinite(floor):
        raise ValueError("compatible_floor must be positive and finite")
    score = (
        torch.abs(target + float(conflict_weight) * conflict)
        * torch.sqrt(torch.abs(source) + 1e-6)
        / (torch.abs(compatible) + floor)
    )
    if not torch.isfinite(score).all():
        raise ValueError("nonfinite anchor score")
    order = sorted(
        range(int(score.numel())),
        key=lambda index: (-float(score[index].item()), index),
    )
    return [int(index) for index in order[: min(count, len(order))]]


def build_v28_anchor_nullspace_basis(
    *,
    config: Mapping[str, Any],
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
    selected_config_hash: str | None = None,
) -> dict[str, Any]:
    require_sha256_hex(train_pool_file_sha256, field_name="train_pool_file_sha256")
    require_sha256_hex(train_pool_summary_hash, field_name="train_pool_summary_hash")
    selected_hash = (
        require_sha256_hex(selected_config_hash, field_name="selected_config_hash")
        if selected_config_hash is not None
        else None
    )
    record_hash = (
        require_sha256_hex(record_id_hash, field_name="record_id_hash")
        if record_id_hash is not None
        else None
    )
    anchor_count = int(config.get("anchor_count", 0))
    compatible_floor = float(config.get("compatible_floor", 0.0))
    nullspace_rtol = float(config.get("nullspace_rtol", 0.0))
    conflict_weight = float(config.get("conflict_weight", V28_CONFLICT_WEIGHT))
    start_extra = redact_v28_anchor_nullspace_basis_progress_event({
        **({"record_id_hash": record_hash} if record_hash else {}),
        **({"selected_config_hash": selected_hash} if selected_hash else {}),
        "anchor_count": anchor_count,
        "compatible_floor": compatible_floor,
        "nullspace_rtol": nullspace_rtol,
    })
    if progress_log_path is not None:
        record_progress_event(
            progress_log_path,
            event="anchor_nullspace_basis_start",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=start_extra,
        )
    source = source_weights.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(source.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("source weights have wrong dimension")
    if not torch.isfinite(source).all():
        raise ValueError("nonfinite source weights")
    if nullspace_rtol < 0.0 or not math.isfinite(nullspace_rtol):
        raise ValueError("nullspace_rtol must be finite and nonnegative")
    gradient_info = v28_anchor_gradients_and_compatible_jacobian(
        source_weights=source,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    selected_coordinates = select_v28_anchor_coordinates(
        source_weights=source,
        g_target=gradient_info["g_target"],
        g_conflict=gradient_info["g_conflict"],
        g_compatible=gradient_info["g_compatible"],
        anchor_count=anchor_count,
        compatible_floor=compatible_floor,
        conflict_weight=conflict_weight,
    )
    if not selected_coordinates:
        raise ValueError("anchor-nullspace basis selected no coordinates")
    jacobian = torch.as_tensor(
        gradient_info["compatible_jacobian"],
        dtype=torch.float32,
    )
    centered = jacobian - jacobian.mean(dim=1, keepdim=True)
    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    if not torch.isfinite(singular_values).all() or not torch.isfinite(vh).all():
        raise ValueError("nonfinite compatible jacobian svd")
    s_max = float(torch.max(singular_values).item()) if int(singular_values.numel()) else 0.0
    normalized = singular_values / max(s_max, 1e-12)
    preserve_mask = normalized > float(nullspace_rtol)
    preserve_rank = int(torch.count_nonzero(preserve_mask).item())
    if preserve_rank > 0:
        preserve = vh[preserve_mask].T.to(dtype=torch.float32)
        projected = torch.eye(SOURCE_WEIGHT_DIM, dtype=torch.float32)[:, selected_coordinates]
        projected = projected - preserve @ (preserve.T @ projected)
    else:
        projected = torch.eye(SOURCE_WEIGHT_DIM, dtype=torch.float32)[:, selected_coordinates]
    q, r = torch.linalg.qr(projected, mode="reduced")
    rank = int(torch.count_nonzero(torch.abs(torch.diag(r)) > 1e-6).item())
    if rank <= 0:
        raise ValueError("anchor-nullspace basis rank must be positive")
    basis = validate_v27_localized_basis_matrix(q[:, :rank])
    compatible_energy_denominator = float(torch.linalg.norm(jacobian).item())
    compatible_energy_numerator = float(torch.linalg.norm(jacobian @ basis).item())
    compatible_energy_ratio = (
        compatible_energy_numerator / max(compatible_energy_denominator, 1e-12)
    )
    if not math.isfinite(compatible_energy_ratio):
        raise ValueError("nonfinite compatible energy ratio")
    selected_coordinate_hash = stable_hash_json(selected_coordinates)
    basis_hash = stable_hash_json(tensor_to_hashable(basis))
    completed_extra = redact_v28_anchor_nullspace_basis_progress_event({
        **({"record_id_hash": record_hash} if record_hash else {}),
        **({"selected_config_hash": selected_hash} if selected_hash else {}),
        "anchor_count": anchor_count,
        "basis_hash": basis_hash,
        "basis_rank": int(basis.shape[1]),
        "compatible_count": int(gradient_info["support_split_counts"]["compatible"]),
        "compatible_energy_ratio": compatible_energy_ratio,
        "compatible_floor": compatible_floor,
        "conflict_count": int(gradient_info["support_split_counts"]["conflict"]),
        "jacobian_row_count": int(jacobian.shape[0]),
        "nullspace_rtol": nullspace_rtol,
        "preserve_rank": preserve_rank,
        "selected_coordinate_hash": selected_coordinate_hash,
        "target_count": int(gradient_info["support_split_counts"]["target"]),
    })
    if progress_log_path is not None:
        record_progress_event(
            progress_log_path,
            event="anchor_nullspace_basis_completed",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=completed_extra,
        )
    audit = {
        "anchor_count": anchor_count,
        "basis_hash": basis_hash,
        "basis_rank": int(basis.shape[1]),
        "basis_type": V28_ANCHOR_NULLSPACE_MATCHED_EDIT_SOURCE,
        "compatible_energy_ratio": compatible_energy_ratio,
        "compatible_floor": compatible_floor,
        "conflict_weight": conflict_weight,
        "experiment_variant": V28_EXPERIMENT_VARIANT,
        "jacobian_row_count": int(jacobian.shape[0]),
        "nullspace_rtol": nullspace_rtol,
        "preserve_rank": preserve_rank,
        "script_sha256": str(script_sha256),
        "selected_coordinate_hash": selected_coordinate_hash,
        "source_behavior": str(source_behavior),
        "support_split_counts": dict(gradient_info["support_split_counts"]),
        "target_behavior": str(target_behavior),
        "train_pool_file_sha256": str(train_pool_file_sha256),
        "train_pool_summary_hash": str(train_pool_summary_hash),
    }
    audit["basis_provenance_hash"] = stable_hash_json(audit)
    return {
        "audit": audit,
        "basis": basis,
    }


def redact_v29_sparse_support_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_hash_keys = {
        "coordinate_hash",
        "delta_sha256",
        "record_id_hash",
        "selected_config_hash",
    }
    allowed_int_keys = {
        "batch_count",
        "compatible_count",
        "conflict_count",
        "epoch",
        "selected_coordinate_count",
        "sparse_top_k",
        "step",
        "target_count",
    }
    allowed_float_keys = {
        "compatible_floor",
        "compatible_mse",
        "conflict_bce",
        "delta_l2",
        "delta_norm",
        "extra_compatible_weight",
        "loss",
        "target_bce",
        "trust_norm_cap",
    }
    redacted: dict[str, Any] = {}
    for key in allowed_hash_keys:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    if "failure_reason" in payload:
        redacted["failure_reason"] = str(payload["failure_reason"])
    finite_values = [
        value for key, value in redacted.items()
        if key in allowed_float_keys
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def v29_sparse_support_gradients(
    *,
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
) -> dict[str, Any]:
    source = source_weights.detach().clone().to(dtype=torch.float32).reshape(-1)
    support = v27_support_tensors_for_source_target(
        source_weights=source,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )

    def loss_grad(loss_fn: Any) -> torch.Tensor:
        flat = source.detach().clone().requires_grad_(True)
        loss = loss_fn(flat)
        grad = torch.autograd.grad(loss, flat, retain_graph=False, allow_unused=False)[0]
        grad = grad.detach().clone().to(dtype=torch.float32, device="cpu").reshape(-1)
        if not torch.isfinite(grad).all():
            raise ValueError("nonfinite sparse support gradient")
        return grad

    g_target = loss_grad(lambda flat: F.binary_cross_entropy_with_logits(
        v27_subject_logits_for_inputs(flat, support["target_inputs"]),
        support["target_labels"].to(dtype=torch.float32),
    ))
    g_conflict = loss_grad(lambda flat: F.binary_cross_entropy_with_logits(
        v27_subject_logits_for_inputs(flat, support["conflict_inputs"]),
        support["conflict_target_labels"].to(dtype=torch.float32),
    ))
    g_compatible = loss_grad(lambda flat: F.mse_loss(
        v27_subject_logits_for_inputs(flat, support["compatible_inputs"]),
        support["compatible_source_logits"].to(dtype=torch.float32),
    ))
    return {
        "g_compatible": g_compatible,
        "g_conflict": g_conflict,
        "g_target": g_target,
        "support_split_counts": {
            "compatible": int(support["compatible_inputs"].shape[0]),
            "conflict": int(support["conflict_inputs"].shape[0]),
            "target": int(support["target_inputs"].shape[0]),
        },
    }


def select_v29_sparse_coordinates(
    *,
    g_target: torch.Tensor,
    g_conflict: torch.Tensor,
    g_compatible: torch.Tensor,
    sparse_top_k: int,
    compatible_floor: float,
    conflict_weight: float = V29_CONFLICT_WEIGHT,
) -> list[int]:
    target = g_target.detach().clone().to(dtype=torch.float32).reshape(-1)
    conflict = g_conflict.detach().clone().to(dtype=torch.float32).reshape(-1)
    compatible = g_compatible.detach().clone().to(dtype=torch.float32).reshape(-1)
    if not all(int(value.numel()) == SOURCE_WEIGHT_DIM for value in [target, conflict, compatible]):
        raise ValueError("sparse coordinate score inputs have wrong dimension")
    if not all(torch.isfinite(value).all() for value in [target, conflict, compatible]):
        raise ValueError("nonfinite sparse coordinate score input")
    count = int(sparse_top_k)
    if count <= 0:
        raise ValueError("sparse_top_k must be positive")
    floor = float(compatible_floor)
    if floor <= 0.0 or not math.isfinite(floor):
        raise ValueError("compatible_floor must be positive and finite")
    score = (
        torch.abs(target + float(conflict_weight) * conflict)
        / (torch.abs(compatible) + floor)
    )
    if not torch.isfinite(score).all():
        raise ValueError("nonfinite sparse coordinate score")
    order = sorted(
        range(int(score.numel())),
        key=lambda index: (-float(score[index].item()), index),
    )
    return [int(index) for index in order[: min(count, len(order))]]


def select_v31_sign_coherent_sparse_coordinates(
    *,
    g_target: torch.Tensor,
    g_conflict: torch.Tensor,
    g_compatible: torch.Tensor,
    sparse_top_k: int,
    compatible_floor: float,
    conflict_weight: float = V29_CONFLICT_WEIGHT,
    sign_conflict_penalty: float = 1.0,
) -> list[int]:
    target = g_target.detach().clone().to(dtype=torch.float32).reshape(-1)
    conflict = g_conflict.detach().clone().to(dtype=torch.float32).reshape(-1)
    compatible = g_compatible.detach().clone().to(dtype=torch.float32).reshape(-1)
    if not all(int(value.numel()) == SOURCE_WEIGHT_DIM for value in [target, conflict, compatible]):
        raise ValueError("sign-coherent sparse coordinate score inputs have wrong dimension")
    if not all(torch.isfinite(value).all() for value in [target, conflict, compatible]):
        raise ValueError("nonfinite sign-coherent sparse coordinate score input")
    count = int(sparse_top_k)
    if count <= 0:
        raise ValueError("sparse_top_k must be positive")
    floor = float(compatible_floor)
    if floor <= 0.0 or not math.isfinite(floor):
        raise ValueError("compatible_floor must be positive and finite")
    penalty = float(sign_conflict_penalty)
    if penalty < 0.0 or not math.isfinite(penalty):
        raise ValueError("sign_conflict_penalty must be finite and nonnegative")
    sign_conflict = (target * conflict < 0.0).to(dtype=torch.float32)
    score = (
        torch.abs(target + float(conflict_weight) * conflict)
        / (torch.abs(compatible) + floor)
        / (1.0 + penalty * sign_conflict)
    )
    if not torch.isfinite(score).all():
        raise ValueError("nonfinite sign-coherent sparse coordinate score")
    order = sorted(
        range(int(score.numel())),
        key=lambda index: (-float(score[index].item()), index),
    )
    return [int(index) for index in order[: min(count, len(order))]]


def solve_v29_breadth_first_sparse_support_edit(
    *,
    coordinate_hash: str,
    selected_coordinates: Sequence[int],
    config: Mapping[str, Any],
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
    selected_config_hash: str | None = None,
) -> dict[str, Any]:
    checked_coordinate_hash = require_sha256_hex(
        coordinate_hash,
        field_name="coordinate_hash",
    )
    source = source_weights.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(source.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("source weights have wrong dimension")
    if not torch.isfinite(source).all():
        raise ValueError("nonfinite source weights")
    coordinates = [int(index) for index in selected_coordinates]
    if not coordinates:
        raise ValueError("selected_coordinates must be nonempty")
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("selected_coordinates must be unique")
    if any(index < 0 or index >= SOURCE_WEIGHT_DIM for index in coordinates):
        raise ValueError("selected coordinate out of range")
    trust_norm_cap = float(config.get("trust_norm_cap", 0.0))
    compatible_floor = float(config.get("compatible_floor", 0.0))
    extra_compatible_weight = float(config.get("extra_compatible_weight", 0.0))
    sparse_top_k = int(config.get("sparse_top_k", len(coordinates)))
    if trust_norm_cap < 0.0 or not math.isfinite(trust_norm_cap):
        raise ValueError("trust_norm_cap must be finite and nonnegative")
    if compatible_floor <= 0.0 or not math.isfinite(compatible_floor):
        raise ValueError("compatible_floor must be positive and finite")
    if extra_compatible_weight < 0.0 or not math.isfinite(extra_compatible_weight):
        raise ValueError("extra_compatible_weight must be finite and nonnegative")
    support = v27_support_tensors_for_source_target(
        source_weights=source,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    coordinate_index = torch.tensor(coordinates, dtype=torch.long)
    values = torch.zeros(len(coordinates), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [values],
        lr=V29_BREADTH_FIRST_LR,
        betas=V27_LOCALIZED_OPTIMIZER_BETAS,
        eps=V27_LOCALIZED_OPTIMIZER_EPS,
        weight_decay=0.0,
        amsgrad=False,
    )
    trace: list[dict[str, Any]] = []
    best: tuple[float, float, int, torch.Tensor, dict[str, Any]] | None = None

    def sparse_delta(current_values: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.float32)
        return delta.index_copy(0, coordinate_index, current_values)

    for epoch in range(1, V29_BREADTH_FIRST_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = sparse_delta(values)
        edited = source + delta
        target_logits = v27_subject_logits_for_inputs(edited, support["target_inputs"])
        conflict_logits = v27_subject_logits_for_inputs(edited, support["conflict_inputs"])
        compatible_logits = v27_subject_logits_for_inputs(edited, support["compatible_inputs"])
        target_bce = F.binary_cross_entropy_with_logits(
            target_logits,
            support["target_labels"].to(dtype=torch.float32),
        )
        conflict_bce = F.binary_cross_entropy_with_logits(
            conflict_logits,
            support["conflict_target_labels"].to(dtype=torch.float32),
        )
        compatible_mse = F.mse_loss(
            compatible_logits,
            support["compatible_source_logits"].to(dtype=torch.float32),
        )
        delta_l2 = torch.mean(delta.pow(2))
        loss = (
            V29_TARGET_BCE_WEIGHT * target_bce
            + V29_CONFLICT_BCE_WEIGHT * conflict_bce
            + V29_COMPATIBLE_PROBE_WEIGHT * compatible_mse
            + extra_compatible_weight * compatible_mse
            + V29_DELTA_L2_WEIGHT * delta_l2
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite breadth-first sparse support loss")
        loss.backward()
        if values.grad is None or not torch.isfinite(values.grad).all():
            raise ValueError("nonfinite breadth-first sparse gradient")
        torch.nn.utils.clip_grad_norm_([values], V27_LOCALIZED_GRAD_CLIP_NORM)
        optimizer.step()
        with torch.no_grad():
            current_delta = sparse_delta(values).detach().clone().to(dtype=torch.float32)
            clipped_delta = apply_norm_cap(current_delta, max_norm=trust_norm_cap)
            if float(torch.linalg.norm(current_delta).item()) > trust_norm_cap + 1e-8:
                values.copy_(clipped_delta[coordinate_index])
            if not torch.isfinite(values).all():
                raise ValueError("nonfinite breadth-first sparse values")
            current_delta = sparse_delta(values).detach().clone().to(dtype=torch.float32)
            current_delta_norm = float(torch.linalg.norm(current_delta).item())
            scalar_losses = {
                "compatible_mse": float(compatible_mse.detach().item()),
                "conflict_bce": float(conflict_bce.detach().item()),
                "delta_l2": float(delta_l2.detach().item()),
                "loss": float(loss.detach().item()),
                "target_bce": float(target_bce.detach().item()),
            }
            if epoch % 10 == 0 or epoch == V29_BREADTH_FIRST_EPOCHS:
                progress_event = redact_v29_sparse_support_progress_event({
                    "compatible_floor": compatible_floor,
                    "coordinate_hash": checked_coordinate_hash,
                    "delta_norm": current_delta_norm,
                    "epoch": epoch,
                    "extra_compatible_weight": extra_compatible_weight,
                    "loss": scalar_losses["loss"],
                    "selected_coordinate_count": len(coordinates),
                    "sparse_top_k": sparse_top_k,
                    "step": epoch,
                    "trust_norm_cap": trust_norm_cap,
                })
                trace.append(progress_event)
                if progress_log_path is not None:
                    event_extra = dict(progress_event)
                    if record_id_hash is not None:
                        event_extra["record_id_hash"] = require_sha256_hex(
                            record_id_hash,
                            field_name="record_id_hash",
                        )
                    if selected_config_hash is not None:
                        event_extra["selected_config_hash"] = require_sha256_hex(
                            selected_config_hash,
                            field_name="selected_config_hash",
                        )
                    record_progress_event(
                        progress_log_path,
                        event="v29_breadth_first_optimizer_progress",
                        started_at_monotonic=started_at_monotonic,
                        now_monotonic=now_monotonic,
                        extra=event_extra,
                    )
            objective_key = (
                scalar_losses["loss"],
                current_delta_norm,
                epoch,
            )
            if best is None or objective_key < best[:3]:
                best = (
                    scalar_losses["loss"],
                    current_delta_norm,
                    epoch,
                    current_delta,
                    scalar_losses,
                )
    if best is None:
        raise ValueError("breadth-first sparse optimizer produced no candidate")
    best_loss, best_delta_norm, best_epoch, best_delta, best_losses = best
    clipped_delta = apply_norm_cap(best_delta, max_norm=trust_norm_cap)
    hard_norm_clipped = bool(float(torch.linalg.norm(best_delta).item()) > trust_norm_cap + 1e-8)
    audit = {
        "best_epoch": int(best_epoch),
        "compatible_floor": compatible_floor,
        "coordinate_hash": checked_coordinate_hash,
        "delta_norm_before_hard_cap": float(best_delta_norm),
        "delta_sha256": stable_hash_json(tensor_to_hashable(clipped_delta)),
        "experiment_variant": V29_EXPERIMENT_VARIANT,
        "extra_compatible_weight": extra_compatible_weight,
        "hard_norm_clipped": hard_norm_clipped,
        "matched_edit_source": V29_BREADTH_FIRST_SPARSE_MATCHED_EDIT_SOURCE,
        "optimization_boundary": v29_breadth_first_sparse_optimization_boundary(),
        "optimization_split": "support",
        "optimizer": "AdamW",
        "optimizer_trace_hash": stable_hash_json(trace),
        "proof_split": "heldout",
        "selected_coordinate_count": len(coordinates),
        "sparse_top_k": sparse_top_k,
        "support_objective": float(best_loss),
        "support_objective_is_proof_metric": False,
        "support_scalar_losses": best_losses,
        "trust_norm_cap": trust_norm_cap,
    }
    finite_failures = recursive_numeric_finiteness_failures(audit)
    if finite_failures:
        raise ValueError("nonfinite breadth-first sparse audit: " + ", ".join(finite_failures[:5]))
    if progress_log_path is not None:
        completed_extra = redact_v29_sparse_support_progress_event({
            "coordinate_hash": checked_coordinate_hash,
            "delta_norm": float(torch.linalg.norm(clipped_delta).item()),
            "delta_sha256": audit["delta_sha256"],
            "epoch": int(best_epoch),
            "loss": float(best_loss),
            "selected_coordinate_count": len(coordinates),
            "sparse_top_k": sparse_top_k,
            "step": int(best_epoch),
            "trust_norm_cap": trust_norm_cap,
        })
        if record_id_hash is not None:
            completed_extra["record_id_hash"] = require_sha256_hex(
                record_id_hash,
                field_name="record_id_hash",
            )
        if selected_config_hash is not None:
            completed_extra["selected_config_hash"] = require_sha256_hex(
                selected_config_hash,
                field_name="selected_config_hash",
            )
        record_progress_event(
            progress_log_path,
            event="v29_breadth_first_optimizer_completed",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=completed_extra,
        )
    return {
        "audit": audit,
        "delta": clipped_delta,
    }


def redact_v30_margin_gated_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_hash_keys = {
        "coordinate_hash",
        "delta_sha256",
        "record_id_hash",
        "selected_config_hash",
    }
    allowed_int_keys = {
        "compatible_count",
        "conflict_count",
        "epoch",
        "selected_coordinate_count",
        "sparse_top_k",
        "step",
        "target_count",
    }
    allowed_float_keys = {
        "compatible_floor",
        "compatible_mse",
        "conflict_bce",
        "delta_l2",
        "delta_norm",
        "extra_compatible_weight",
        "loss",
        "target_bce",
        "target_margin_floor",
        "target_margin_hinge",
        "trust_norm_cap",
    }
    redacted: dict[str, Any] = {}
    for key in allowed_hash_keys:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    if "failure_reason" in payload:
        redacted["failure_reason"] = str(payload["failure_reason"])
    finite_values = [
        value for key, value in redacted.items()
        if key in allowed_float_keys
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def v30_target_margin_hinge_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin_floor: float,
) -> torch.Tensor:
    checked_floor = float(margin_floor)
    if checked_floor < 0.0 or not math.isfinite(checked_floor):
        raise ValueError("margin_floor must be finite and nonnegative")
    logits = logits.to(dtype=torch.float32)
    labels = labels.to(dtype=torch.float32)
    signed = (2.0 * labels - 1.0) * logits
    return torch.relu(checked_floor - signed).mean()


def solve_v30_margin_gated_sparse_support_edit(
    *,
    coordinate_hash: str,
    selected_coordinates: Sequence[int],
    config: Mapping[str, Any],
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
    selected_config_hash: str | None = None,
) -> dict[str, Any]:
    checked_coordinate_hash = require_sha256_hex(
        coordinate_hash,
        field_name="coordinate_hash",
    )
    source = source_weights.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(source.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("source weights have wrong dimension")
    if not torch.isfinite(source).all():
        raise ValueError("nonfinite source weights")
    coordinates = [int(index) for index in selected_coordinates]
    if not coordinates:
        raise ValueError("selected_coordinates must be nonempty")
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("selected_coordinates must be unique")
    if any(index < 0 or index >= SOURCE_WEIGHT_DIM for index in coordinates):
        raise ValueError("selected coordinate out of range")
    trust_norm_cap = float(config.get("trust_norm_cap", 0.0))
    compatible_floor = float(config.get("compatible_floor", V30_COMPATIBLE_FLOOR))
    extra_compatible_weight = float(
        config.get("extra_compatible_weight", V30_EXTRA_COMPATIBLE_WEIGHT)
    )
    target_margin_floor = float(config.get("target_margin_floor", 0.0))
    sparse_top_k = int(config.get("sparse_top_k", len(coordinates)))
    if trust_norm_cap < 0.0 or not math.isfinite(trust_norm_cap):
        raise ValueError("trust_norm_cap must be finite and nonnegative")
    if compatible_floor <= 0.0 or not math.isfinite(compatible_floor):
        raise ValueError("compatible_floor must be positive and finite")
    if extra_compatible_weight < 0.0 or not math.isfinite(extra_compatible_weight):
        raise ValueError("extra_compatible_weight must be finite and nonnegative")
    if target_margin_floor < 0.0 or not math.isfinite(target_margin_floor):
        raise ValueError("target_margin_floor must be finite and nonnegative")
    support = v27_support_tensors_for_source_target(
        source_weights=source,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    coordinate_index = torch.tensor(coordinates, dtype=torch.long)
    values = torch.zeros(len(coordinates), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [values],
        lr=V29_BREADTH_FIRST_LR,
        betas=V27_LOCALIZED_OPTIMIZER_BETAS,
        eps=V27_LOCALIZED_OPTIMIZER_EPS,
        weight_decay=0.0,
        amsgrad=False,
    )
    trace: list[dict[str, Any]] = []
    best: tuple[float, float, int, torch.Tensor, dict[str, Any]] | None = None

    def sparse_delta(current_values: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.float32)
        return delta.index_copy(0, coordinate_index, current_values)

    for epoch in range(1, V29_BREADTH_FIRST_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = sparse_delta(values)
        edited = source + delta
        target_logits = v27_subject_logits_for_inputs(edited, support["target_inputs"])
        conflict_logits = v27_subject_logits_for_inputs(edited, support["conflict_inputs"])
        compatible_logits = v27_subject_logits_for_inputs(edited, support["compatible_inputs"])
        target_labels = support["target_labels"].to(dtype=torch.float32)
        target_bce = F.binary_cross_entropy_with_logits(target_logits, target_labels)
        target_margin_hinge = v30_target_margin_hinge_loss(
            logits=target_logits,
            labels=target_labels,
            margin_floor=target_margin_floor,
        )
        conflict_bce = F.binary_cross_entropy_with_logits(
            conflict_logits,
            support["conflict_target_labels"].to(dtype=torch.float32),
        )
        compatible_mse = F.mse_loss(
            compatible_logits,
            support["compatible_source_logits"].to(dtype=torch.float32),
        )
        delta_l2 = torch.mean(delta.pow(2))
        loss = (
            V29_TARGET_BCE_WEIGHT * target_bce
            + V30_TARGET_MARGIN_WEIGHT * target_margin_hinge
            + V29_CONFLICT_BCE_WEIGHT * conflict_bce
            + V29_COMPATIBLE_PROBE_WEIGHT * compatible_mse
            + extra_compatible_weight * compatible_mse
            + V29_DELTA_L2_WEIGHT * delta_l2
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite margin-gated sparse support loss")
        loss.backward()
        if values.grad is None or not torch.isfinite(values.grad).all():
            raise ValueError("nonfinite margin-gated sparse gradient")
        torch.nn.utils.clip_grad_norm_([values], V27_LOCALIZED_GRAD_CLIP_NORM)
        optimizer.step()
        with torch.no_grad():
            current_delta = sparse_delta(values).detach().clone().to(dtype=torch.float32)
            clipped_delta = apply_norm_cap(current_delta, max_norm=trust_norm_cap)
            if float(torch.linalg.norm(current_delta).item()) > trust_norm_cap + 1e-8:
                values.copy_(clipped_delta[coordinate_index])
            if not torch.isfinite(values).all():
                raise ValueError("nonfinite margin-gated sparse values")
            current_delta = sparse_delta(values).detach().clone().to(dtype=torch.float32)
            current_delta_norm = float(torch.linalg.norm(current_delta).item())
            scalar_losses = {
                "compatible_mse": float(compatible_mse.detach().item()),
                "conflict_bce": float(conflict_bce.detach().item()),
                "delta_l2": float(delta_l2.detach().item()),
                "loss": float(loss.detach().item()),
                "target_bce": float(target_bce.detach().item()),
                "target_margin_hinge": float(target_margin_hinge.detach().item()),
            }
            if epoch % 10 == 0 or epoch == V29_BREADTH_FIRST_EPOCHS:
                progress_event = redact_v30_margin_gated_progress_event({
                    "compatible_floor": compatible_floor,
                    "coordinate_hash": checked_coordinate_hash,
                    "delta_norm": current_delta_norm,
                    "epoch": epoch,
                    "extra_compatible_weight": extra_compatible_weight,
                    "loss": scalar_losses["loss"],
                    "selected_coordinate_count": len(coordinates),
                    "sparse_top_k": sparse_top_k,
                    "step": epoch,
                    "target_margin_floor": target_margin_floor,
                    "target_margin_hinge": scalar_losses["target_margin_hinge"],
                    "trust_norm_cap": trust_norm_cap,
                })
                trace.append(progress_event)
                if progress_log_path is not None:
                    event_extra = dict(progress_event)
                    if record_id_hash is not None:
                        event_extra["record_id_hash"] = require_sha256_hex(
                            record_id_hash,
                            field_name="record_id_hash",
                        )
                    if selected_config_hash is not None:
                        event_extra["selected_config_hash"] = require_sha256_hex(
                            selected_config_hash,
                            field_name="selected_config_hash",
                        )
                    record_progress_event(
                        progress_log_path,
                        event="v30_margin_gated_optimizer_progress",
                        started_at_monotonic=started_at_monotonic,
                        now_monotonic=now_monotonic,
                        extra=event_extra,
                    )
            objective_key = (
                scalar_losses["loss"],
                current_delta_norm,
                epoch,
            )
            if best is None or objective_key < best[:3]:
                best = (
                    scalar_losses["loss"],
                    current_delta_norm,
                    epoch,
                    current_delta,
                    scalar_losses,
                )
    if best is None:
        raise ValueError("margin-gated sparse optimizer produced no candidate")
    best_loss, best_delta_norm, best_epoch, best_delta, best_losses = best
    clipped_delta = apply_norm_cap(best_delta, max_norm=trust_norm_cap)
    audit = {
        "best_epoch": int(best_epoch),
        "compatible_floor": compatible_floor,
        "coordinate_hash": checked_coordinate_hash,
        "delta_norm_before_hard_cap": float(best_delta_norm),
        "delta_sha256": stable_hash_json(tensor_to_hashable(clipped_delta)),
        "experiment_variant": V30_EXPERIMENT_VARIANT,
        "extra_compatible_weight": extra_compatible_weight,
        "hard_norm_clipped": bool(
            float(torch.linalg.norm(best_delta).item()) > trust_norm_cap + 1e-8
        ),
        "matched_edit_source": V30_MARGIN_GATED_SPARSE_MATCHED_EDIT_SOURCE,
        "optimization_boundary": v30_margin_gated_optimization_boundary(),
        "optimization_split": "support",
        "optimizer": "AdamW",
        "optimizer_trace_hash": stable_hash_json(trace),
        "proof_split": "heldout",
        "selected_coordinate_count": len(coordinates),
        "sparse_top_k": sparse_top_k,
        "support_objective": float(best_loss),
        "support_objective_is_proof_metric": False,
        "support_scalar_losses": best_losses,
        "target_margin_floor": target_margin_floor,
        "target_margin_weight": V30_TARGET_MARGIN_WEIGHT,
        "trust_norm_cap": trust_norm_cap,
    }
    finite_failures = recursive_numeric_finiteness_failures(audit)
    if finite_failures:
        raise ValueError("nonfinite margin-gated sparse audit: " + ", ".join(finite_failures[:5]))
    if progress_log_path is not None:
        completed_extra = redact_v30_margin_gated_progress_event({
            "coordinate_hash": checked_coordinate_hash,
            "delta_norm": float(torch.linalg.norm(clipped_delta).item()),
            "delta_sha256": audit["delta_sha256"],
            "epoch": int(best_epoch),
            "loss": float(best_loss),
            "selected_coordinate_count": len(coordinates),
            "sparse_top_k": sparse_top_k,
            "step": int(best_epoch),
            "target_margin_floor": target_margin_floor,
            "target_margin_hinge": best_losses["target_margin_hinge"],
            "trust_norm_cap": trust_norm_cap,
        })
        if record_id_hash is not None:
            completed_extra["record_id_hash"] = require_sha256_hex(
                record_id_hash,
                field_name="record_id_hash",
            )
        if selected_config_hash is not None:
            completed_extra["selected_config_hash"] = require_sha256_hex(
                selected_config_hash,
                field_name="selected_config_hash",
            )
        record_progress_event(
            progress_log_path,
            event="v30_margin_gated_optimizer_completed",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=completed_extra,
        )
    return {
        "audit": audit,
        "delta": clipped_delta,
    }


def redact_v31_orthogonal_sign_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_hash_keys = {
        "coordinate_hash",
        "delta_sha256",
        "record_id_hash",
        "selected_config_hash",
    }
    allowed_int_keys = {
        "compatible_count",
        "conflict_count",
        "epoch",
        "selected_coordinate_count",
        "sparse_top_k",
        "step",
        "target_count",
    }
    allowed_float_keys = {
        "compatible_floor",
        "compatible_mse",
        "compatible_orthogonal_loss",
        "compatible_orthogonal_weight",
        "conflict_bce",
        "delta_l2",
        "delta_norm",
        "extra_compatible_weight",
        "hard_target_margin_weight",
        "loss",
        "sign_conflict_penalty",
        "target_bce",
        "target_margin_floor",
        "target_margin_hinge",
        "target_multiplier",
        "trust_norm_cap",
    }
    redacted: dict[str, Any] = {}
    for key in allowed_hash_keys:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    if "failure_reason" in payload:
        redacted["failure_reason"] = str(payload["failure_reason"])
    finite_values = [
        value for key, value in redacted.items()
        if key in allowed_float_keys
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def v31_support_hardness_multiplier(
    *,
    source_target_logits: torch.Tensor,
    target_labels: torch.Tensor,
    target_margin_floor: float,
    hard_target_margin_weight: float,
) -> torch.Tensor:
    floor = float(target_margin_floor)
    weight = float(hard_target_margin_weight)
    if floor < 0.0 or not math.isfinite(floor):
        raise ValueError("target_margin_floor must be finite and nonnegative")
    if weight < 0.0 or not math.isfinite(weight):
        raise ValueError("hard_target_margin_weight must be finite and nonnegative")
    logits = source_target_logits.detach().clone().to(dtype=torch.float32)
    labels = target_labels.detach().clone().to(dtype=torch.float32)
    signed = (2.0 * labels - 1.0) * logits
    hardness = torch.relu(torch.tensor(floor, dtype=torch.float32) - signed).mean()
    multiplier = 1.0 + weight * hardness.detach()
    if not torch.isfinite(multiplier):
        raise ValueError("nonfinite support hardness multiplier")
    return multiplier.to(dtype=torch.float32)


def v31_compatible_gradient_orthogonal_loss(
    *,
    delta: torch.Tensor,
    g_compatible: torch.Tensor,
) -> torch.Tensor:
    delta_flat = delta.to(dtype=torch.float32).reshape(-1)
    compatible = g_compatible.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(delta_flat.numel()) != int(compatible.numel()):
        raise ValueError("orthogonal loss inputs have mismatched dimensions")
    if not torch.isfinite(delta_flat).all() or not torch.isfinite(compatible).all():
        raise ValueError("nonfinite orthogonal loss input")
    norm = torch.linalg.norm(compatible)
    unit = compatible / (norm + 1e-8)
    loss = torch.dot(delta_flat, unit).pow(2)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite compatible gradient orthogonal loss")
    return loss


def solve_v31_orthogonal_sign_sparse_support_edit(
    *,
    compatible_gradient: torch.Tensor,
    coordinate_hash: str,
    selected_coordinates: Sequence[int],
    config: Mapping[str, Any],
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
    selected_config_hash: str | None = None,
) -> dict[str, Any]:
    checked_coordinate_hash = require_sha256_hex(
        coordinate_hash,
        field_name="coordinate_hash",
    )
    source = source_weights.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(source.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("source weights have wrong dimension")
    if not torch.isfinite(source).all():
        raise ValueError("nonfinite source weights")
    compatible_grad = compatible_gradient.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(compatible_grad.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("compatible_gradient has wrong dimension")
    if not torch.isfinite(compatible_grad).all():
        raise ValueError("nonfinite compatible_gradient")
    coordinates = [int(index) for index in selected_coordinates]
    if not coordinates:
        raise ValueError("selected_coordinates must be nonempty")
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("selected_coordinates must be unique")
    if any(index < 0 or index >= SOURCE_WEIGHT_DIM for index in coordinates):
        raise ValueError("selected coordinate out of range")
    trust_norm_cap = float(config.get("trust_norm_cap", 0.0))
    compatible_floor = float(config.get("compatible_floor", V31_COMPATIBLE_FLOOR))
    extra_compatible_weight = float(
        config.get("extra_compatible_weight", V31_EXTRA_COMPATIBLE_WEIGHT)
    )
    target_margin_floor = float(config.get("target_margin_floor", V31_TARGET_MARGIN_FLOOR))
    hard_target_margin_weight = float(
        config.get("hard_target_margin_weight", V31_HARD_TARGET_MARGIN_WEIGHT)
    )
    compatible_orthogonal_weight = float(
        config.get(
            "compatible_orthogonal_weight",
            V31_COMPATIBLE_ORTHOGONAL_WEIGHT_GRID[0],
        )
    )
    sign_conflict_penalty = float(
        config.get("sign_conflict_penalty", V31_SIGN_CONFLICT_PENALTY_GRID[0])
    )
    sparse_top_k = int(config.get("sparse_top_k", len(coordinates)))
    if trust_norm_cap < 0.0 or not math.isfinite(trust_norm_cap):
        raise ValueError("trust_norm_cap must be finite and nonnegative")
    if compatible_floor <= 0.0 or not math.isfinite(compatible_floor):
        raise ValueError("compatible_floor must be positive and finite")
    if extra_compatible_weight < 0.0 or not math.isfinite(extra_compatible_weight):
        raise ValueError("extra_compatible_weight must be finite and nonnegative")
    if target_margin_floor < 0.0 or not math.isfinite(target_margin_floor):
        raise ValueError("target_margin_floor must be finite and nonnegative")
    if hard_target_margin_weight < 0.0 or not math.isfinite(hard_target_margin_weight):
        raise ValueError("hard_target_margin_weight must be finite and nonnegative")
    if compatible_orthogonal_weight < 0.0 or not math.isfinite(compatible_orthogonal_weight):
        raise ValueError("compatible_orthogonal_weight must be finite and nonnegative")
    if sign_conflict_penalty < 0.0 or not math.isfinite(sign_conflict_penalty):
        raise ValueError("sign_conflict_penalty must be finite and nonnegative")
    support = v27_support_tensors_for_source_target(
        source_weights=source,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    with torch.no_grad():
        source_target_logits = v27_subject_logits_for_inputs(source, support["target_inputs"])
        target_multiplier = v31_support_hardness_multiplier(
            source_target_logits=source_target_logits,
            target_labels=support["target_labels"].to(dtype=torch.float32),
            target_margin_floor=target_margin_floor,
            hard_target_margin_weight=hard_target_margin_weight,
        )
    coordinate_index = torch.tensor(coordinates, dtype=torch.long)
    values = torch.zeros(len(coordinates), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [values],
        lr=V29_BREADTH_FIRST_LR,
        betas=V27_LOCALIZED_OPTIMIZER_BETAS,
        eps=V27_LOCALIZED_OPTIMIZER_EPS,
        weight_decay=0.0,
        amsgrad=False,
    )
    trace: list[dict[str, Any]] = []
    best: tuple[float, float, int, torch.Tensor, dict[str, Any]] | None = None

    def sparse_delta(current_values: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.float32)
        return delta.index_copy(0, coordinate_index, current_values)

    for epoch in range(1, V29_BREADTH_FIRST_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = sparse_delta(values)
        edited = source + delta
        target_logits = v27_subject_logits_for_inputs(edited, support["target_inputs"])
        conflict_logits = v27_subject_logits_for_inputs(edited, support["conflict_inputs"])
        compatible_logits = v27_subject_logits_for_inputs(edited, support["compatible_inputs"])
        target_labels = support["target_labels"].to(dtype=torch.float32)
        target_bce = F.binary_cross_entropy_with_logits(target_logits, target_labels)
        target_margin_hinge = v30_target_margin_hinge_loss(
            logits=target_logits,
            labels=target_labels,
            margin_floor=target_margin_floor,
        )
        conflict_bce = F.binary_cross_entropy_with_logits(
            conflict_logits,
            support["conflict_target_labels"].to(dtype=torch.float32),
        )
        compatible_mse = F.mse_loss(
            compatible_logits,
            support["compatible_source_logits"].to(dtype=torch.float32),
        )
        compatible_orthogonal_loss = v31_compatible_gradient_orthogonal_loss(
            delta=delta,
            g_compatible=compatible_grad,
        )
        delta_l2 = torch.mean(delta.pow(2))
        loss = (
            target_multiplier * (
                V29_TARGET_BCE_WEIGHT * target_bce
                + V30_TARGET_MARGIN_WEIGHT * target_margin_hinge
            )
            + V29_CONFLICT_BCE_WEIGHT * conflict_bce
            + V29_COMPATIBLE_PROBE_WEIGHT * compatible_mse
            + extra_compatible_weight * compatible_mse
            + compatible_orthogonal_weight * compatible_orthogonal_loss
            + V29_DELTA_L2_WEIGHT * delta_l2
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite orthogonal sign sparse support loss")
        loss.backward()
        if values.grad is None or not torch.isfinite(values.grad).all():
            raise ValueError("nonfinite orthogonal sign sparse gradient")
        torch.nn.utils.clip_grad_norm_([values], V27_LOCALIZED_GRAD_CLIP_NORM)
        optimizer.step()
        with torch.no_grad():
            current_delta = sparse_delta(values).detach().clone().to(dtype=torch.float32)
            clipped_delta = apply_norm_cap(current_delta, max_norm=trust_norm_cap)
            if float(torch.linalg.norm(current_delta).item()) > trust_norm_cap + 1e-8:
                values.copy_(clipped_delta[coordinate_index])
            if not torch.isfinite(values).all():
                raise ValueError("nonfinite orthogonal sign sparse values")
            current_delta = sparse_delta(values).detach().clone().to(dtype=torch.float32)
            current_delta_norm = float(torch.linalg.norm(current_delta).item())
            scalar_losses = {
                "compatible_mse": float(compatible_mse.detach().item()),
                "compatible_orthogonal_loss": float(
                    compatible_orthogonal_loss.detach().item()
                ),
                "conflict_bce": float(conflict_bce.detach().item()),
                "delta_l2": float(delta_l2.detach().item()),
                "loss": float(loss.detach().item()),
                "target_bce": float(target_bce.detach().item()),
                "target_margin_hinge": float(target_margin_hinge.detach().item()),
            }
            if epoch % 10 == 0 or epoch == V29_BREADTH_FIRST_EPOCHS:
                progress_event = redact_v31_orthogonal_sign_progress_event({
                    "compatible_floor": compatible_floor,
                    "compatible_orthogonal_loss": scalar_losses[
                        "compatible_orthogonal_loss"
                    ],
                    "compatible_orthogonal_weight": compatible_orthogonal_weight,
                    "coordinate_hash": checked_coordinate_hash,
                    "delta_norm": current_delta_norm,
                    "epoch": epoch,
                    "extra_compatible_weight": extra_compatible_weight,
                    "hard_target_margin_weight": hard_target_margin_weight,
                    "loss": scalar_losses["loss"],
                    "selected_coordinate_count": len(coordinates),
                    "sign_conflict_penalty": sign_conflict_penalty,
                    "sparse_top_k": sparse_top_k,
                    "step": epoch,
                    "target_margin_floor": target_margin_floor,
                    "target_margin_hinge": scalar_losses["target_margin_hinge"],
                    "target_multiplier": float(target_multiplier.item()),
                    "trust_norm_cap": trust_norm_cap,
                })
                trace.append(progress_event)
                if progress_log_path is not None:
                    event_extra = dict(progress_event)
                    if record_id_hash is not None:
                        event_extra["record_id_hash"] = require_sha256_hex(
                            record_id_hash,
                            field_name="record_id_hash",
                        )
                    if selected_config_hash is not None:
                        event_extra["selected_config_hash"] = require_sha256_hex(
                            selected_config_hash,
                            field_name="selected_config_hash",
                        )
                    record_progress_event(
                        progress_log_path,
                        event="v31_orthogonal_sign_optimizer_progress",
                        started_at_monotonic=started_at_monotonic,
                        now_monotonic=now_monotonic,
                        extra=event_extra,
                    )
            objective_key = (
                scalar_losses["loss"],
                current_delta_norm,
                epoch,
            )
            if best is None or objective_key < best[:3]:
                best = (
                    scalar_losses["loss"],
                    current_delta_norm,
                    epoch,
                    current_delta,
                    scalar_losses,
                )
    if best is None:
        raise ValueError("orthogonal sign sparse optimizer produced no candidate")
    best_loss, best_delta_norm, best_epoch, best_delta, best_losses = best
    clipped_delta = apply_norm_cap(best_delta, max_norm=trust_norm_cap)
    audit = {
        "best_epoch": int(best_epoch),
        "compatible_floor": compatible_floor,
        "compatible_orthogonal_weight": compatible_orthogonal_weight,
        "coordinate_hash": checked_coordinate_hash,
        "delta_norm_before_hard_cap": float(best_delta_norm),
        "delta_sha256": stable_hash_json(tensor_to_hashable(clipped_delta)),
        "experiment_variant": V31_EXPERIMENT_VARIANT,
        "extra_compatible_weight": extra_compatible_weight,
        "hard_norm_clipped": bool(
            float(torch.linalg.norm(best_delta).item()) > trust_norm_cap + 1e-8
        ),
        "hard_target_margin_weight": hard_target_margin_weight,
        "matched_edit_source": V31_ORTHOGONAL_SIGN_SPARSE_MATCHED_EDIT_SOURCE,
        "optimization_boundary": v31_orthogonal_sign_optimization_boundary(),
        "optimization_split": "support",
        "optimizer": "AdamW",
        "optimizer_trace_hash": stable_hash_json(trace),
        "proof_split": "heldout",
        "selected_coordinate_count": len(coordinates),
        "sign_conflict_penalty": sign_conflict_penalty,
        "sparse_top_k": sparse_top_k,
        "support_objective": float(best_loss),
        "support_objective_is_proof_metric": False,
        "support_scalar_losses": best_losses,
        "target_margin_floor": target_margin_floor,
        "target_margin_weight": V30_TARGET_MARGIN_WEIGHT,
        "target_multiplier": float(target_multiplier.item()),
        "trust_norm_cap": trust_norm_cap,
    }
    finite_failures = recursive_numeric_finiteness_failures(audit)
    if finite_failures:
        raise ValueError("nonfinite orthogonal sign sparse audit: " + ", ".join(finite_failures[:5]))
    if progress_log_path is not None:
        completed_extra = redact_v31_orthogonal_sign_progress_event({
            "compatible_orthogonal_loss": best_losses["compatible_orthogonal_loss"],
            "compatible_orthogonal_weight": compatible_orthogonal_weight,
            "coordinate_hash": checked_coordinate_hash,
            "delta_norm": float(torch.linalg.norm(clipped_delta).item()),
            "delta_sha256": audit["delta_sha256"],
            "epoch": int(best_epoch),
            "loss": float(best_loss),
            "selected_coordinate_count": len(coordinates),
            "sign_conflict_penalty": sign_conflict_penalty,
            "sparse_top_k": sparse_top_k,
            "step": int(best_epoch),
            "target_margin_floor": target_margin_floor,
            "target_margin_hinge": best_losses["target_margin_hinge"],
            "target_multiplier": float(target_multiplier.item()),
            "trust_norm_cap": trust_norm_cap,
        })
        if record_id_hash is not None:
            completed_extra["record_id_hash"] = require_sha256_hex(
                record_id_hash,
                field_name="record_id_hash",
            )
        if selected_config_hash is not None:
            completed_extra["selected_config_hash"] = require_sha256_hex(
                selected_config_hash,
                field_name="selected_config_hash",
            )
        record_progress_event(
            progress_log_path,
            event="v31_orthogonal_sign_optimizer_completed",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=completed_extra,
        )
    return {
        "audit": audit,
        "delta": clipped_delta,
    }


def redact_v32_support_tournament_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    redacted = redact_v31_orthogonal_sign_progress_event(payload)
    allowed_float_keys = {
        "support_runner_margin",
        "support_target_margin",
        "support_tournament_margin",
        "tournament_margin_floor",
        "tournament_margin_hinge",
        "tournament_margin_weight",
    }
    allowed_hash_keys = {"support_tournament_tensor_hash"}
    for key in allowed_hash_keys:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    finite_values = [
        value for key, value in redacted.items()
        if key in allowed_float_keys
        or key in {
            "compatible_floor",
            "compatible_mse",
            "compatible_orthogonal_loss",
            "compatible_orthogonal_weight",
            "conflict_bce",
            "delta_l2",
            "delta_norm",
            "extra_compatible_weight",
            "hard_target_margin_weight",
            "loss",
            "sign_conflict_penalty",
            "target_bce",
            "target_margin_floor",
            "target_margin_hinge",
            "target_multiplier",
            "trust_norm_cap",
        }
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def select_v35_support_source_alpha_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    alpha_target_margin_floor: float,
    alpha_tournament_margin_floor: float,
    fallback_target_penalty: float,
    fallback_tournament_penalty: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("alpha candidates must be nonempty")
    target_floor = float(alpha_target_margin_floor)
    tournament_floor = float(alpha_tournament_margin_floor)
    target_penalty = float(fallback_target_penalty)
    tournament_penalty = float(fallback_tournament_penalty)
    for name, value in [
        ("alpha_target_margin_floor", target_floor),
        ("alpha_tournament_margin_floor", tournament_floor),
        ("fallback_target_penalty", target_penalty),
        ("fallback_tournament_penalty", tournament_penalty),
    ]:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if target_penalty < 0.0 or tournament_penalty < 0.0:
        raise ValueError("fallback penalties must be nonnegative")

    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = {
            "alpha": float(candidate["alpha"]),
            "candidate_index": int(index),
            "support_compatible_mse": float(candidate["support_compatible_mse"]),
            "support_runner_margin": float(candidate.get("support_runner_margin", 0.0)),
            "support_target_margin": float(candidate["support_target_margin"]),
            "support_tournament_margin": float(candidate["support_tournament_margin"]),
        }
        finite_values = [
            item["alpha"],
            item["support_compatible_mse"],
            item["support_runner_margin"],
            item["support_target_margin"],
            item["support_tournament_margin"],
        ]
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("nonfinite alpha candidate")
        if item["alpha"] < 0.0:
            raise ValueError("alpha must be nonnegative")
        item["eligible"] = bool(
            item["support_target_margin"] >= target_floor
            and item["support_tournament_margin"] >= tournament_floor
        )
        item["fallback_score"] = float(
            item["support_compatible_mse"]
            + target_penalty
            * max(0.0, target_floor - item["support_target_margin"])
            + tournament_penalty
            * max(0.0, tournament_floor - item["support_tournament_margin"])
        )
        normalized.append(item)

    eligible = [item for item in normalized if item["eligible"]]
    eligible_count = len(eligible)
    candidate_metrics_hash = stable_hash_json(normalized)
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (
                item["support_compatible_mse"],
                -item["support_tournament_margin"],
                -item["support_target_margin"],
                -item["alpha"],
            ),
        )
        selection_mode = "eligible_min_compatible_mse"
    else:
        selected = min(
            normalized,
            key=lambda item: (
                item["fallback_score"],
                item["support_compatible_mse"],
                -item["support_tournament_margin"],
                -item["support_target_margin"],
                -item["alpha"],
            ),
        )
        selection_mode = "fallback_penalized"

    result = dict(selected)
    result["selection_mode"] = selection_mode
    result["alpha_candidate_count"] = len(normalized)
    result["candidate_metrics_hash"] = candidate_metrics_hash
    result["eligible_count"] = eligible_count
    return result


def select_v38_compatible_gated_alpha_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    alpha_target_margin_floor: float,
    alpha_tournament_margin_floor: float,
    alpha_compatible_mse_gate: float,
    fallback_target_penalty: float,
    fallback_tournament_penalty: float,
    fallback_compatible_penalty: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("alpha candidates must be nonempty")
    target_floor = float(alpha_target_margin_floor)
    tournament_floor = float(alpha_tournament_margin_floor)
    compatible_gate = float(alpha_compatible_mse_gate)
    target_penalty = float(fallback_target_penalty)
    tournament_penalty = float(fallback_tournament_penalty)
    compatible_penalty = float(fallback_compatible_penalty)
    for name, value in [
        ("alpha_target_margin_floor", target_floor),
        ("alpha_tournament_margin_floor", tournament_floor),
        ("alpha_compatible_mse_gate", compatible_gate),
        ("fallback_target_penalty", target_penalty),
        ("fallback_tournament_penalty", tournament_penalty),
        ("fallback_compatible_penalty", compatible_penalty),
    ]:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if (
        compatible_gate < 0.0
        or target_penalty < 0.0
        or tournament_penalty < 0.0
        or compatible_penalty < 0.0
    ):
        raise ValueError("compatible gate and fallback penalties must be nonnegative")

    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = {
            "alpha": float(candidate["alpha"]),
            "alpha_compatible_mse_gate": compatible_gate,
            "candidate_index": int(index),
            "fallback_compatible_penalty": compatible_penalty,
            "support_compatible_mse": float(candidate["support_compatible_mse"]),
            "support_runner_margin": float(candidate.get("support_runner_margin", 0.0)),
            "support_target_margin": float(candidate["support_target_margin"]),
            "support_tournament_margin": float(candidate["support_tournament_margin"]),
        }
        item["target_pass"] = item["support_target_margin"] >= target_floor
        item["tournament_pass"] = item["support_tournament_margin"] >= tournament_floor
        item["compatible_gate_pass"] = item["support_compatible_mse"] <= compatible_gate
        item["eligible"] = (
            item["target_pass"]
            and item["tournament_pass"]
            and item["compatible_gate_pass"]
        )
        target_gap = max(0.0, target_floor - item["support_target_margin"])
        tournament_gap = max(0.0, tournament_floor - item["support_tournament_margin"])
        compatible_gap = max(0.0, item["support_compatible_mse"] - compatible_gate)
        item["fallback_score"] = (
            target_penalty * target_gap
            + tournament_penalty * tournament_gap
            + compatible_penalty * compatible_gap
            + item["support_compatible_mse"]
        )
        normalized.append(item)
    candidate_metrics_hash = stable_hash_json([
        {
            "alpha": item["alpha"],
            "alpha_compatible_mse_gate": item["alpha_compatible_mse_gate"],
            "compatible_gate_pass": item["compatible_gate_pass"],
            "eligible": item["eligible"],
            "fallback_score": item["fallback_score"],
            "support_compatible_mse": item["support_compatible_mse"],
            "support_runner_margin": item["support_runner_margin"],
            "support_target_margin": item["support_target_margin"],
            "support_tournament_margin": item["support_tournament_margin"],
        }
        for item in normalized
    ])
    eligible = [item for item in normalized if bool(item["eligible"])]
    eligible_count = len(eligible)
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (
                item["support_compatible_mse"],
                -item["support_tournament_margin"],
                -item["support_target_margin"],
                -item["alpha"],
            ),
        )
        selection_mode = "eligible_min_compatible_mse"
    else:
        selected = min(
            normalized,
            key=lambda item: (
                item["fallback_score"],
                item["support_compatible_mse"],
                -item["support_tournament_margin"],
                -item["support_target_margin"],
                -item["alpha"],
            ),
        )
        selection_mode = "fallback_penalized"
    result = dict(selected)
    result["selection_mode"] = selection_mode
    result["alpha_candidate_count"] = len(normalized)
    result["candidate_metrics_hash"] = candidate_metrics_hash
    result["eligible_count"] = eligible_count
    return result


def select_v39_target_feasible_lexicographic_alpha_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    alpha_target_margin_floor: float,
    alpha_tournament_margin_floor: float,
    alpha_compatible_mse_soft_gate: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("alpha candidates must be nonempty")
    target_floor = float(alpha_target_margin_floor)
    tournament_floor = float(alpha_tournament_margin_floor)
    compatible_gate = float(alpha_compatible_mse_soft_gate)
    for name, value in [
        ("alpha_target_margin_floor", target_floor),
        ("alpha_tournament_margin_floor", tournament_floor),
        ("alpha_compatible_mse_soft_gate", compatible_gate),
    ]:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if compatible_gate < 0.0:
        raise ValueError("alpha compatible MSE soft gate must be nonnegative")

    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = {
            "alpha": float(candidate["alpha"]),
            "alpha_compatible_mse_soft_gate": compatible_gate,
            "candidate_index": int(index),
            "support_compatible_mse": float(candidate["support_compatible_mse"]),
            "support_runner_margin": float(candidate.get("support_runner_margin", 0.0)),
            "support_target_margin": float(candidate["support_target_margin"]),
            "support_tournament_margin": float(candidate["support_tournament_margin"]),
        }
        finite_values = [
            item["alpha"],
            item["alpha_compatible_mse_soft_gate"],
            item["support_compatible_mse"],
            item["support_runner_margin"],
            item["support_target_margin"],
            item["support_tournament_margin"],
        ]
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("nonfinite alpha candidate")
        if item["alpha"] < 0.0:
            raise ValueError("alpha must be nonnegative")
        item["target_gap"] = max(0.0, target_floor - item["support_target_margin"])
        item["tournament_gap"] = max(
            0.0,
            tournament_floor - item["support_tournament_margin"],
        )
        item["compatible_gap"] = max(
            0.0,
            item["support_compatible_mse"] - compatible_gate,
        )
        item["target_rank_score"] = item["target_gap"] + item["tournament_gap"]
        item["target_feasible"] = bool(
            item["target_gap"] == 0.0 and item["tournament_gap"] == 0.0
        )
        normalized.append(item)

    candidate_metrics_hash = stable_hash_json([
        {
            "alpha": item["alpha"],
            "alpha_compatible_mse_soft_gate": item["alpha_compatible_mse_soft_gate"],
            "compatible_gap": item["compatible_gap"],
            "support_compatible_mse": item["support_compatible_mse"],
            "support_runner_margin": item["support_runner_margin"],
            "support_target_margin": item["support_target_margin"],
            "support_tournament_margin": item["support_tournament_margin"],
            "target_feasible": item["target_feasible"],
            "target_gap": item["target_gap"],
            "target_rank_score": item["target_rank_score"],
            "tournament_gap": item["tournament_gap"],
        }
        for item in normalized
    ])
    eligible = [item for item in normalized if bool(item["target_feasible"])]
    eligible_count = len(eligible)
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (
                item["support_compatible_mse"],
                -item["support_tournament_margin"],
                -item["support_target_margin"],
                -item["alpha"],
            ),
        )
        selection_mode = "target_feasible_min_compatible_mse"
    else:
        selected = min(
            normalized,
            key=lambda item: (
                item["target_rank_score"],
                -item["support_tournament_margin"],
                -item["support_target_margin"],
                item["compatible_gap"],
                item["support_compatible_mse"],
                -item["alpha"],
            ),
        )
        selection_mode = "fallback_target_feasible_lexicographic"

    result = dict(selected)
    result["selection_mode"] = selection_mode
    result["alpha_candidate_count"] = len(normalized)
    result["candidate_metrics_hash"] = candidate_metrics_hash
    result["eligible_count"] = eligible_count
    return result


def select_v40_target_tolerance_locality_budget_alpha_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    alpha_target_margin_floor: float,
    alpha_tournament_margin_floor: float,
    alpha_compatible_mse_soft_gate: float,
    target_rank_score_tolerance: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("alpha candidates must be nonempty")
    target_floor = float(alpha_target_margin_floor)
    tournament_floor = float(alpha_tournament_margin_floor)
    compatible_gate = float(alpha_compatible_mse_soft_gate)
    tolerance = float(target_rank_score_tolerance)
    for name, value in [
        ("alpha_target_margin_floor", target_floor),
        ("alpha_tournament_margin_floor", tournament_floor),
        ("alpha_compatible_mse_soft_gate", compatible_gate),
        ("target_rank_score_tolerance", tolerance),
    ]:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if compatible_gate < 0.0 or tolerance < 0.0:
        raise ValueError("compatible soft gate and target tolerance must be nonnegative")

    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = {
            "alpha": float(candidate["alpha"]),
            "alpha_compatible_mse_soft_gate": compatible_gate,
            "candidate_index": int(index),
            "support_compatible_mse": float(candidate["support_compatible_mse"]),
            "support_runner_margin": float(candidate.get("support_runner_margin", 0.0)),
            "support_target_margin": float(candidate["support_target_margin"]),
            "support_tournament_margin": float(candidate["support_tournament_margin"]),
            "target_rank_score_tolerance": tolerance,
        }
        finite_values = [
            item["alpha"],
            item["alpha_compatible_mse_soft_gate"],
            item["support_compatible_mse"],
            item["support_runner_margin"],
            item["support_target_margin"],
            item["support_tournament_margin"],
            item["target_rank_score_tolerance"],
        ]
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("nonfinite alpha candidate")
        if item["alpha"] < 0.0:
            raise ValueError("alpha must be nonnegative")
        item["target_gap"] = max(0.0, target_floor - item["support_target_margin"])
        item["tournament_gap"] = max(
            0.0,
            tournament_floor - item["support_tournament_margin"],
        )
        item["compatible_gap"] = max(
            0.0,
            item["support_compatible_mse"] - compatible_gate,
        )
        item["target_rank_score"] = item["target_gap"] + item["tournament_gap"]
        item["target_feasible"] = bool(item["target_rank_score"] == 0.0)
        normalized.append(item)

    best_target_rank_score = min(item["target_rank_score"] for item in normalized)
    feasible = [item for item in normalized if bool(item["target_feasible"])]
    if feasible:
        pool = feasible
        selection_mode = "target_feasible_min_compatible_mse"
    else:
        pool = [
            item for item in normalized
            if item["target_rank_score"] <= best_target_rank_score + tolerance
        ]
        selection_mode = "target_tolerance_min_compatible_mse"
    selected = min(
        pool,
        key=lambda item: (
            item["support_compatible_mse"],
            item["compatible_gap"],
            -item["support_tournament_margin"],
            -item["support_target_margin"],
            -item["alpha"],
        ),
    )
    candidate_metrics_hash = stable_hash_json([
        {
            "alpha": item["alpha"],
            "alpha_compatible_mse_soft_gate": item["alpha_compatible_mse_soft_gate"],
            "compatible_gap": item["compatible_gap"],
            "support_compatible_mse": item["support_compatible_mse"],
            "support_runner_margin": item["support_runner_margin"],
            "support_target_margin": item["support_target_margin"],
            "support_tournament_margin": item["support_tournament_margin"],
            "target_feasible": item["target_feasible"],
            "target_gap": item["target_gap"],
            "target_rank_score": item["target_rank_score"],
            "target_rank_score_tolerance": item["target_rank_score_tolerance"],
            "tournament_gap": item["tournament_gap"],
        }
        for item in normalized
    ])
    result = dict(selected)
    result["alpha_candidate_count"] = len(normalized)
    result["best_target_rank_score"] = float(best_target_rank_score)
    result["candidate_metrics_hash"] = candidate_metrics_hash
    result["eligible_count"] = len(feasible)
    result["selection_mode"] = selection_mode
    result["within_target_tolerance_count"] = len(pool)
    return result


def select_v41_trajectory_frontier_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    compatible_mse_soft_gate: float,
    target_margin_floor: float,
    target_rank_score_tolerance: float,
    tournament_margin_floor: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("trajectory frontier candidates must be nonempty")
    compatible_gate = float(compatible_mse_soft_gate)
    target_floor = float(target_margin_floor)
    tolerance = float(target_rank_score_tolerance)
    tournament_floor = float(tournament_margin_floor)
    for name, value in [
        ("compatible_mse_soft_gate", compatible_gate),
        ("target_margin_floor", target_floor),
        ("target_rank_score_tolerance", tolerance),
        ("tournament_margin_floor", tournament_floor),
    ]:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if compatible_gate < 0.0 or tolerance < 0.0:
        raise ValueError("compatible soft gate and target tolerance must be nonnegative")

    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = {
            "candidate_index": int(index),
            "epoch": int(candidate["epoch"]),
            "loss": float(candidate["loss"]),
            "preservation_energy_ratio": float(candidate["preservation_energy_ratio"]),
            "projected_delta_norm": float(candidate["projected_delta_norm"]),
            "support_compatible_mse": float(candidate["support_compatible_mse"]),
            "support_runner_margin": float(candidate.get("support_runner_margin", 0.0)),
            "support_target_margin": float(candidate["support_target_margin"]),
            "support_tournament_margin": float(candidate["support_tournament_margin"]),
            "target_rank_score_tolerance": tolerance,
        }
        finite_values = [
            item["epoch"],
            item["loss"],
            item["preservation_energy_ratio"],
            item["projected_delta_norm"],
            item["support_compatible_mse"],
            item["support_runner_margin"],
            item["support_target_margin"],
            item["support_tournament_margin"],
            item["target_rank_score_tolerance"],
        ]
        if not all(math.isfinite(float(value)) for value in finite_values):
            raise ValueError("nonfinite trajectory frontier candidate")
        if item["epoch"] <= 0:
            raise ValueError("frontier candidate epoch must be positive")
        item["target_gap"] = max(0.0, target_floor - item["support_target_margin"])
        item["tournament_gap"] = max(
            0.0,
            tournament_floor - item["support_tournament_margin"],
        )
        item["compatible_gap"] = max(
            0.0,
            item["support_compatible_mse"] - compatible_gate,
        )
        item["target_rank_score"] = item["target_gap"] + item["tournament_gap"]
        item["target_feasible"] = bool(item["target_rank_score"] == 0.0)
        normalized.append(item)

    best_target_rank_score = min(item["target_rank_score"] for item in normalized)
    feasible = [item for item in normalized if bool(item["target_feasible"])]
    if feasible:
        pool = feasible
        selection_mode = "frontier_target_feasible_min_compatible_mse"
    else:
        pool = [
            item for item in normalized
            if item["target_rank_score"] <= best_target_rank_score + tolerance
        ]
        selection_mode = "frontier_target_tolerance_min_compatible_mse"
    selected = min(
        pool,
        key=lambda item: (
            item["support_compatible_mse"],
            item["compatible_gap"],
            -item["support_tournament_margin"],
            -item["support_target_margin"],
            item["projected_delta_norm"],
            item["epoch"],
        ),
    )
    candidate_metrics_hash = stable_hash_json([
        {
            "epoch": item["epoch"],
            "compatible_gap": item["compatible_gap"],
            "loss": item["loss"],
            "preservation_energy_ratio": item["preservation_energy_ratio"],
            "projected_delta_norm": item["projected_delta_norm"],
            "support_compatible_mse": item["support_compatible_mse"],
            "support_runner_margin": item["support_runner_margin"],
            "support_target_margin": item["support_target_margin"],
            "support_tournament_margin": item["support_tournament_margin"],
            "target_feasible": item["target_feasible"],
            "target_gap": item["target_gap"],
            "target_rank_score": item["target_rank_score"],
            "target_rank_score_tolerance": item["target_rank_score_tolerance"],
            "tournament_gap": item["tournament_gap"],
        }
        for item in normalized
    ])
    result = dict(selected)
    result["best_target_rank_score"] = float(best_target_rank_score)
    result["candidate_metrics_hash"] = candidate_metrics_hash
    result["frontier_candidate_count"] = len(normalized)
    result["selection_mode"] = selection_mode
    result["within_target_tolerance_count"] = len(pool)
    return result


def select_v42_compatible_dual_frontier_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    compatible_mse_budget: float,
    target_margin_floor: float,
    target_rank_score_tolerance: float,
    tournament_margin_floor: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("V42 frontier candidates must be nonempty")
    budget = float(compatible_mse_budget)
    target_floor = float(target_margin_floor)
    tolerance = float(target_rank_score_tolerance)
    tournament_floor = float(tournament_margin_floor)
    for name, value in [
        ("compatible_mse_budget", budget),
        ("target_margin_floor", target_floor),
        ("target_rank_score_tolerance", tolerance),
        ("tournament_margin_floor", tournament_floor),
    ]:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if budget < 0.0 or tolerance < 0.0:
        raise ValueError("compatible budget and target tolerance must be nonnegative")

    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = {
            "candidate_index": int(index),
            "epoch": int(candidate["epoch"]),
            "loss": float(candidate["loss"]),
            "preservation_energy_ratio": float(candidate["preservation_energy_ratio"]),
            "projected_delta_norm": float(candidate["projected_delta_norm"]),
            "support_compatible_mse": float(candidate["support_compatible_mse"]),
            "support_runner_margin": float(candidate.get("support_runner_margin", 0.0)),
            "support_target_margin": float(candidate["support_target_margin"]),
            "support_tournament_margin": float(candidate["support_tournament_margin"]),
            "compatible_constraint_residual": float(
                candidate["compatible_constraint_residual"]
            ),
            "compatible_dual_lambda": float(candidate["compatible_dual_lambda"]),
            "compatible_mse_budget": budget,
            "target_rank_score_tolerance": tolerance,
        }
        if not all(math.isfinite(float(value)) for value in item.values()):
            raise ValueError("nonfinite V42 frontier candidate")
        if item["epoch"] <= 0:
            raise ValueError("V42 frontier candidate epoch must be positive")
        item["target_gap"] = max(0.0, target_floor - item["support_target_margin"])
        item["tournament_gap"] = max(
            0.0,
            tournament_floor - item["support_tournament_margin"],
        )
        item["compatible_gap"] = max(
            0.0,
            item["support_compatible_mse"] - budget,
        )
        item["target_rank_score"] = item["target_gap"] + item["tournament_gap"]
        item["target_feasible"] = bool(item["target_rank_score"] == 0.0)
        item["compatible_constraint_feasible"] = bool(item["compatible_gap"] == 0.0)
        normalized.append(item)

    best_target_rank_score = min(item["target_rank_score"] for item in normalized)
    localized = [
        item for item in normalized
        if item["target_feasible"] and item["compatible_constraint_feasible"]
    ]
    target_feasible = [item for item in normalized if item["target_feasible"]]
    if localized:
        pool = localized
        selection_mode = "frontier_target_and_compatible_feasible"
    elif target_feasible:
        pool = target_feasible
        selection_mode = "frontier_target_feasible_min_compatible_residual"
    else:
        pool = [
            item for item in normalized
            if item["target_rank_score"] <= best_target_rank_score + tolerance
        ]
        selection_mode = "frontier_target_tolerance_min_compatible_residual"

    selected = min(
        pool,
        key=lambda item: (
            item["compatible_gap"],
            item["support_compatible_mse"],
            item["target_rank_score"],
            -item["support_tournament_margin"],
            -item["support_target_margin"],
            item["projected_delta_norm"],
            item["epoch"],
        ),
    )
    candidate_metrics_hash = stable_hash_json([
        {
            "epoch": item["epoch"],
            "compatible_constraint_feasible": item["compatible_constraint_feasible"],
            "compatible_constraint_residual": item["compatible_constraint_residual"],
            "compatible_dual_lambda": item["compatible_dual_lambda"],
            "compatible_gap": item["compatible_gap"],
            "compatible_mse_budget": item["compatible_mse_budget"],
            "loss": item["loss"],
            "preservation_energy_ratio": item["preservation_energy_ratio"],
            "projected_delta_norm": item["projected_delta_norm"],
            "support_compatible_mse": item["support_compatible_mse"],
            "support_runner_margin": item["support_runner_margin"],
            "support_target_margin": item["support_target_margin"],
            "support_tournament_margin": item["support_tournament_margin"],
            "target_feasible": item["target_feasible"],
            "target_gap": item["target_gap"],
            "target_rank_score": item["target_rank_score"],
            "target_rank_score_tolerance": item["target_rank_score_tolerance"],
            "tournament_gap": item["tournament_gap"],
        }
        for item in normalized
    ])
    result = dict(selected)
    result["best_target_rank_score"] = float(best_target_rank_score)
    result["candidate_metrics_hash"] = candidate_metrics_hash
    result["frontier_candidate_count"] = len(normalized)
    result["localized_feasible_count"] = len(localized)
    result["selection_mode"] = selection_mode
    result["within_target_tolerance_count"] = len(pool)
    return result


def redact_v35_support_source_alpha_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    redacted = redact_v32_support_tournament_progress_event(payload)
    allowed_float_keys = {
        "alpha",
        "alpha_target_margin_floor",
        "alpha_tournament_margin_floor",
        "fallback_score",
        "fallback_target_penalty",
        "fallback_tournament_penalty",
        "support_compatible_mse",
    }
    allowed_int_keys = {
        "alpha_candidate_count",
        "candidate_index",
        "eligible_count",
    }
    allowed_string_keys = {
        "selection_mode",
    }
    allowed_hash_keys = {
        "alpha_candidates_hash",
        "candidate_metrics_hash",
        "selected_alpha_candidate_hash",
    }
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_string_keys:
        if key in payload:
            value = str(payload[key])
            if value not in {"eligible_min_compatible_mse", "fallback_penalized"}:
                raise ValueError(f"unknown alpha selection mode: {value}")
            redacted[key] = value
    for key in allowed_hash_keys:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    finite_keys = {
        *allowed_float_keys,
        "compatible_floor",
        "compatible_mse",
        "compatible_orthogonal_loss",
        "compatible_orthogonal_weight",
        "delta_norm",
        "extra_compatible_weight",
        "hard_target_margin_weight",
        "loss",
        "support_runner_margin",
        "support_target_margin",
        "support_tournament_margin",
        "target_margin_floor",
        "target_margin_hinge",
        "target_multiplier",
        "tournament_margin_floor",
        "tournament_margin_hinge",
        "tournament_margin_weight",
        "trust_norm_cap",
    }
    finite_values = [
        value for key, value in redacted.items()
        if key in finite_keys
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def redact_v38_compatible_gated_alpha_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    redacted = redact_v35_support_source_alpha_progress_event(payload)
    allowed_float_keys = {
        "alpha_compatible_mse_gate",
        "fallback_compatible_penalty",
    }
    allowed_bool_keys = {
        "compatible_gate_pass",
    }
    allowed_int_keys = {
        "eligible_count",
    }
    allowed_hash_keys = {
        "candidate_metrics_hash",
    }
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    for key in allowed_bool_keys:
        if key in payload:
            redacted[key] = bool(payload[key])
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_hash_keys:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    finite_keys = {
        "alpha",
        "alpha_compatible_mse_gate",
        "alpha_target_margin_floor",
        "alpha_tournament_margin_floor",
        "fallback_compatible_penalty",
        "fallback_score",
        "fallback_target_penalty",
        "fallback_tournament_penalty",
        "support_compatible_mse",
        "support_runner_margin",
        "support_target_margin",
        "support_tournament_margin",
    }
    finite_values = [
        value for key, value in redacted.items()
        if key in finite_keys
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def redact_v39_target_feasible_alpha_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_float_keys = {
        "alpha",
        "alpha_compatible_mse_soft_gate",
        "compatible_gap",
        "delta_norm",
        "support_compatible_mse",
        "support_runner_margin",
        "support_target_margin",
        "support_tournament_margin",
        "target_gap",
        "target_rank_score",
        "tournament_gap",
    }
    allowed_int_keys = {
        "alpha_candidate_count",
        "candidate_index",
        "eligible_count",
    }
    allowed_bool_keys = {
        "target_feasible",
    }
    allowed_string_keys = {
        "selection_mode",
    }
    allowed_hash_keys = {
        "alpha_candidates_hash",
        "candidate_metrics_hash",
        "record_id_hash",
        "selected_alpha_candidate_hash",
        "selected_config_hash",
    }
    allowed_selection_modes = {
        "target_feasible_min_compatible_mse",
        "fallback_target_feasible_lexicographic",
    }
    redacted: dict[str, Any] = {}
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_bool_keys:
        if key in payload:
            redacted[key] = bool(payload[key])
    for key in allowed_string_keys:
        if key in payload:
            value = str(payload[key])
            if key == "selection_mode" and value not in allowed_selection_modes:
                raise ValueError(f"unknown V39 alpha selection mode: {value}")
            redacted[key] = value
    for key in allowed_hash_keys:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    finite_values = [
        value for key, value in redacted.items()
        if key in allowed_float_keys
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def redact_v40_target_tolerance_alpha_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_float_keys = {
        "alpha",
        "alpha_compatible_mse_soft_gate",
        "best_target_rank_score",
        "compatible_gap",
        "delta_norm",
        "support_compatible_mse",
        "support_runner_margin",
        "support_target_margin",
        "support_tournament_margin",
        "target_gap",
        "target_rank_score",
        "target_rank_score_tolerance",
        "tournament_gap",
    }
    allowed_int_keys = {
        "alpha_candidate_count",
        "candidate_index",
        "eligible_count",
        "within_target_tolerance_count",
    }
    allowed_bool_keys = {
        "target_feasible",
    }
    allowed_string_keys = {
        "selection_mode",
    }
    allowed_hash_keys = {
        "alpha_candidates_hash",
        "candidate_metrics_hash",
        "record_id_hash",
        "selected_alpha_candidate_hash",
        "selected_config_hash",
    }
    allowed_selection_modes = {
        "target_feasible_min_compatible_mse",
        "target_tolerance_min_compatible_mse",
    }
    redacted: dict[str, Any] = {}
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_bool_keys:
        if key in payload:
            redacted[key] = bool(payload[key])
    for key in allowed_string_keys:
        if key in payload:
            value = str(payload[key])
            if key == "selection_mode" and value not in allowed_selection_modes:
                raise ValueError(f"unknown V40 alpha selection mode: {value}")
            redacted[key] = value
    for key in allowed_hash_keys:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    finite_values = [
        value for key, value in redacted.items()
        if key in allowed_float_keys
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def redact_v41_trajectory_frontier_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_float_keys = {
        "best_target_rank_score",
        "compatible_gap",
        "loss",
        "preservation_energy_ratio",
        "projected_delta_norm",
        "support_compatible_mse",
        "support_runner_margin",
        "support_target_margin",
        "support_tournament_margin",
        "target_gap",
        "target_rank_score",
        "target_rank_score_tolerance",
        "tournament_gap",
    }
    allowed_int_keys = {
        "candidate_index",
        "frontier_candidate_count",
        "trajectory_frontier_selected_epoch",
        "within_target_tolerance_count",
    }
    allowed_bool_keys = {
        "target_feasible",
    }
    allowed_string_keys = {
        "selection_mode",
    }
    allowed_hash_keys = {
        "candidate_metrics_hash",
        "frontier_candidates_hash",
        "record_id_hash",
        "selected_config_hash",
    }
    allowed_selection_modes = {
        "frontier_target_feasible_min_compatible_mse",
        "frontier_target_tolerance_min_compatible_mse",
    }
    redacted: dict[str, Any] = {}
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_bool_keys:
        if key in payload:
            redacted[key] = bool(payload[key])
    for key in allowed_string_keys:
        if key in payload:
            value = str(payload[key])
            if key == "selection_mode" and value not in allowed_selection_modes:
                raise ValueError(f"unknown V41 frontier selection mode: {value}")
            redacted[key] = value
    for key in allowed_hash_keys:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    finite_values = [
        value for key, value in redacted.items()
        if key in allowed_float_keys
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def redact_v42_compatible_dual_frontier_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    base_payload = dict(payload)
    if "selection_mode" in base_payload:
        base_payload = {
            **base_payload,
            "selection_mode": "frontier_target_feasible_min_compatible_mse",
        }
    redacted = redact_v41_trajectory_frontier_progress_event(base_payload)
    allowed_float_keys = {
        "compatible_constraint_residual",
        "compatible_dual_lambda",
        "compatible_gap",
        "compatible_mse_budget",
    }
    allowed_int_keys = {
        "localized_feasible_count",
    }
    allowed_bool_keys = {
        "compatible_constraint_feasible",
    }
    allowed_selection_modes = {
        "frontier_target_and_compatible_feasible",
        "frontier_target_feasible_min_compatible_residual",
        "frontier_target_tolerance_min_compatible_residual",
    }
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_bool_keys:
        if key in payload:
            redacted[key] = bool(payload[key])
    if "selection_mode" in payload:
        mode = str(payload["selection_mode"])
        if mode not in allowed_selection_modes:
            raise ValueError(f"unknown V42 frontier selection mode: {mode}")
        redacted["selection_mode"] = mode
    finite_values = [
        value for key, value in redacted.items()
        if key in allowed_float_keys
        or key in {
            "best_target_rank_score",
            "compatible_gap",
            "loss",
            "preservation_energy_ratio",
            "projected_delta_norm",
            "support_compatible_mse",
            "support_runner_margin",
            "support_target_margin",
            "support_tournament_margin",
            "target_gap",
            "target_rank_score",
            "target_rank_score_tolerance",
            "tournament_gap",
        }
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def redact_v36_compatible_nullspace_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_hash_keys = {
        "projection_audit_hash",
        "record_id_hash",
        "selected_config_hash",
    }
    allowed_int_keys = {
        "jacobian_row_count",
        "preserve_rank",
    }
    allowed_float_keys = {
        "base_preservation_energy",
        "compatible_nullspace_rtol",
        "preservation_energy_ratio",
        "projected_preservation_energy",
        "projection_removed_norm",
        "projection_retained_norm",
        "projection_strength",
    }
    redacted: dict[str, Any] = {}
    for key in allowed_hash_keys:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    finite_values = [
        value for key, value in redacted.items()
        if key in allowed_float_keys
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def redact_v37_projected_optimizer_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_hash_keys = {
        "optimizer_audit_hash",
        "record_id_hash",
        "selected_config_hash",
    }
    allowed_int_keys = {
        "jacobian_row_count",
        "optimization_steps",
        "preserve_rank",
    }
    allowed_float_keys = {
        "compatible_nullspace_rtol",
        "final_loss",
        "preservation_energy_ratio",
        "projected_delta_norm",
        "projection_strength",
        "support_compatible_mse",
        "support_runner_margin",
        "support_target_margin",
        "support_tournament_margin",
    }
    redacted: dict[str, Any] = {}
    for key in allowed_hash_keys:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    finite_values = [
        value for key, value in redacted.items()
        if key in allowed_float_keys
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def project_v36_delta_through_compatible_nullspace(
    *,
    base_delta: torch.Tensor,
    compatible_jacobian: torch.Tensor,
    compatible_nullspace_rtol: float,
    projection_strength: float,
    trust_norm_cap: float,
) -> dict[str, Any]:
    delta = base_delta.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(delta.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("base_delta has wrong dimension")
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite base_delta")
    jacobian = compatible_jacobian.detach().clone().to(dtype=torch.float32)
    if jacobian.ndim != 2 or int(jacobian.shape[1]) != SOURCE_WEIGHT_DIM:
        raise ValueError("compatible_jacobian has wrong shape")
    if not torch.isfinite(jacobian).all():
        raise ValueError("nonfinite compatible_jacobian")
    rtol = float(compatible_nullspace_rtol)
    strength = float(projection_strength)
    norm_cap = float(trust_norm_cap)
    for name, value in [
        ("compatible_nullspace_rtol", rtol),
        ("projection_strength", strength),
        ("trust_norm_cap", norm_cap),
    ]:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")

    base_energy = float(torch.linalg.norm(jacobian @ delta).item())
    if int(jacobian.shape[0]) == 0:
        preserve_rank = 0
        row_component = torch.zeros_like(delta)
    else:
        _u, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=False)
        if not torch.isfinite(singular_values).all() or not torch.isfinite(vh).all():
            raise ValueError("nonfinite compatible nullspace svd")
        s_max = float(torch.max(singular_values).item()) if int(singular_values.numel()) else 0.0
        normalized = singular_values / max(s_max, 1e-12)
        preserve_mask = normalized > rtol
        preserve_rank = int(torch.count_nonzero(preserve_mask).item())
        if preserve_rank > 0:
            preserve = vh[preserve_mask].T.to(dtype=torch.float32)
            row_component = preserve @ (preserve.T @ delta)
        else:
            row_component = torch.zeros_like(delta)
    removed = float(strength) * row_component
    projected = apply_norm_cap(delta - removed, max_norm=norm_cap)
    projected_energy = float(torch.linalg.norm(jacobian @ projected).item())
    ratio = projected_energy / max(base_energy, 1e-12)
    audit = {
        "base_preservation_energy": base_energy,
        "compatible_nullspace_rtol": rtol,
        "finite": True,
        "jacobian_row_count": int(jacobian.shape[0]),
        "preservation_energy_ratio": float(ratio),
        "preserve_rank": int(preserve_rank),
        "projected_preservation_energy": projected_energy,
        "projection_removed_norm": float(torch.linalg.norm(removed).item()),
        "projection_retained_norm": float(torch.linalg.norm(projected).item()),
        "projection_strength": strength,
    }
    finite_failures = recursive_numeric_finiteness_failures(audit)
    if finite_failures:
        raise ValueError("nonfinite compatible nullspace audit: " + ", ".join(finite_failures[:5]))
    return {
        "audit": audit,
        "delta": projected,
    }


def project_v37_delta_differentiably(
    *,
    sparse_delta: torch.Tensor,
    compatible_jacobian: torch.Tensor,
    compatible_nullspace_rtol: float,
    projection_strength: float,
    trust_norm_cap: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    delta = sparse_delta.to(dtype=torch.float32).reshape(-1)
    if int(delta.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("sparse_delta has wrong dimension")
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite sparse_delta")
    jacobian = compatible_jacobian.detach().clone().to(dtype=torch.float32)
    if jacobian.ndim != 2 or int(jacobian.shape[1]) != SOURCE_WEIGHT_DIM:
        raise ValueError("compatible_jacobian has wrong shape")
    if not torch.isfinite(jacobian).all():
        raise ValueError("nonfinite compatible_jacobian")
    rtol = float(compatible_nullspace_rtol)
    strength = float(projection_strength)
    norm_cap = float(trust_norm_cap)
    for name, value in [
        ("compatible_nullspace_rtol", rtol),
        ("projection_strength", strength),
        ("trust_norm_cap", norm_cap),
    ]:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")

    base_energy_tensor = torch.linalg.norm(jacobian @ delta)
    if int(jacobian.shape[0]) == 0:
        preserve_rank = 0
        row_component = torch.zeros_like(delta)
    else:
        _u, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=False)
        if not torch.isfinite(singular_values).all() or not torch.isfinite(vh).all():
            raise ValueError("nonfinite compatible nullspace svd")
        s_max = float(torch.max(singular_values).item()) if int(singular_values.numel()) else 0.0
        normalized = singular_values / max(s_max, 1e-12)
        preserve_mask = normalized > rtol
        preserve_rank = int(torch.count_nonzero(preserve_mask).item())
        if preserve_rank > 0:
            preserve = vh[preserve_mask].T.to(dtype=torch.float32)
            row_component = preserve @ (preserve.T @ delta)
        else:
            row_component = torch.zeros_like(delta)

    projected = delta - float(strength) * row_component
    norm = torch.linalg.norm(projected)
    scale = torch.clamp(
        torch.tensor(norm_cap, dtype=torch.float32) / torch.clamp(norm, min=1e-12),
        max=1.0,
    )
    projected = projected * scale
    projected_energy_tensor = torch.linalg.norm(jacobian @ projected)
    audit = {
        "base_preservation_energy": float(base_energy_tensor.detach().item()),
        "compatible_nullspace_rtol": rtol,
        "finite": True,
        "jacobian_row_count": int(jacobian.shape[0]),
        "preservation_energy_ratio": float(
            (projected_energy_tensor / torch.clamp(base_energy_tensor, min=1e-12))
            .detach()
            .item()
        ),
        "preserve_rank": int(preserve_rank),
        "projected_delta_norm": float(torch.linalg.norm(projected).detach().item()),
        "projected_preservation_energy": float(projected_energy_tensor.detach().item()),
        "projection_strength": strength,
    }
    finite_failures = recursive_numeric_finiteness_failures(audit)
    if finite_failures:
        raise ValueError("nonfinite V37 projection audit: " + ", ".join(finite_failures[:5]))
    return projected.to(dtype=torch.float32), audit


def v38_projected_optimizer_support_score(
    *,
    support_target_margin: float,
    support_tournament_margin: float,
    support_compatible_mse: float,
    loss: float,
    target_margin_floor: float,
    tournament_margin_floor: float,
    compatible_mse_gate: float,
    compatible_gate_weight: float,
) -> float:
    target_gap = max(0.0, float(target_margin_floor) - float(support_target_margin))
    tournament_gap = max(
        0.0,
        float(tournament_margin_floor) - float(support_tournament_margin),
    )
    compatible_gate = float(compatible_mse_gate)
    compatible_gap = max(0.0, float(support_compatible_mse) - compatible_gate)
    return float(
        100.0 * target_gap
        + 50.0 * tournament_gap
        + float(compatible_gate_weight) * compatible_gap * compatible_gap
        + float(support_compatible_mse)
        + 0.01 * float(loss)
    )


def v42_compatible_constraint_terms(
    *,
    support_compatible_mse: torch.Tensor,
    compatible_mse_budget: float,
    compatible_dual_lambda: float,
    compatible_augmented_weight: float,
) -> dict[str, torch.Tensor]:
    budget = float(compatible_mse_budget)
    dual_lambda = float(compatible_dual_lambda)
    augmented_weight = float(compatible_augmented_weight)
    for name, value in [
        ("compatible_mse_budget", budget),
        ("compatible_dual_lambda", dual_lambda),
        ("compatible_augmented_weight", augmented_weight),
    ]:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if budget < 0.0 or dual_lambda < 0.0 or augmented_weight < 0.0:
        raise ValueError("compatible constraint parameters must be nonnegative")
    residual = torch.clamp(
        support_compatible_mse - support_compatible_mse.new_tensor(budget),
        min=0.0,
    )
    penalty = dual_lambda * residual + augmented_weight * residual.pow(2)
    return {
        "compatible_constraint_penalty": penalty,
        "compatible_constraint_residual": residual,
    }


def solve_v37_projected_support_optimizer_edit(
    *,
    compatible_gradient: torch.Tensor,
    compatible_jacobian: torch.Tensor,
    coordinate_hash: str,
    selected_coordinates: Sequence[int],
    config: Mapping[str, Any],
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
    selected_config_hash: str | None = None,
) -> dict[str, Any]:
    checked_coordinate_hash = require_sha256_hex(
        coordinate_hash,
        field_name="coordinate_hash",
    )
    checked_config_hash = (
        require_sha256_hex(selected_config_hash, field_name="selected_config_hash")
        if selected_config_hash is not None
        else None
    )
    source = source_weights.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(source.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("source weights have wrong dimension")
    if not torch.isfinite(source).all():
        raise ValueError("nonfinite source weights")
    compatible_grad = compatible_gradient.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(compatible_grad.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("compatible_gradient has wrong dimension")
    if not torch.isfinite(compatible_grad).all():
        raise ValueError("nonfinite compatible_gradient")
    jacobian = compatible_jacobian.detach().clone().to(dtype=torch.float32)
    if jacobian.ndim != 2 or int(jacobian.shape[1]) != SOURCE_WEIGHT_DIM:
        raise ValueError("compatible_jacobian has wrong shape")
    if not torch.isfinite(jacobian).all():
        raise ValueError("nonfinite compatible_jacobian")

    coordinates = [int(index) for index in selected_coordinates]
    if not coordinates:
        raise ValueError("selected_coordinates must be nonempty")
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("selected_coordinates must be unique")
    if any(index < 0 or index >= SOURCE_WEIGHT_DIM for index in coordinates):
        raise ValueError("selected coordinate out of range")

    trust_norm_cap = float(config.get("trust_norm_cap", V32_TRUST_NORM_CAP))
    compatible_nullspace_rtol = float(config.get("compatible_nullspace_rtol", 1e-3))
    projection_strength = float(config.get("projection_strength", 0.5))
    projected_optimizer_epochs = int(config.get("projected_optimizer_epochs", 80))
    projected_optimizer_lr = float(config.get("projected_optimizer_lr", V29_BREADTH_FIRST_LR))
    projected_optimizer_event_prefix = str(
        config.get("projected_optimizer_event_prefix", "v37_projected_optimizer")
    )
    trajectory_frontier_enabled = bool(config.get("trajectory_frontier_enabled", False))
    trajectory_frontier_event_prefix = str(
        config.get("trajectory_frontier_event_prefix", "v41_trajectory_frontier")
    )
    compatible_mse_gate = float(config.get("compatible_mse_gate", float("inf")))
    alpha_compatible_mse_soft_gate = float(
        config.get("alpha_compatible_mse_soft_gate", compatible_mse_gate)
    )
    target_rank_score_tolerance = float(
        config.get("target_rank_score_tolerance", 0.0)
    )
    v42_compatible_dual_enabled = bool(config.get("v42_compatible_dual_enabled", False))
    compatible_mse_budget = float(config.get("compatible_mse_budget", compatible_mse_gate))
    compatible_augmented_weight = float(config.get("compatible_augmented_weight", 0.0))
    compatible_dual_lr = float(config.get("compatible_dual_lr", 0.0))
    compatible_dual_max = float(config.get("compatible_dual_max", 0.0))
    compatible_dual_lambda = float(config.get("compatible_dual_initial", 0.0))
    compatible_gate_weight = float(config.get("compatible_gate_weight", 0.0))
    extra_compatible_weight = float(config.get("extra_compatible_weight", 0.05))
    target_margin_floor = float(config.get("target_margin_floor", V32_TARGET_MARGIN_FLOOR))
    hard_target_margin_weight = float(
        config.get("hard_target_margin_weight", V32_HARD_TARGET_MARGIN_WEIGHT)
    )
    compatible_orthogonal_weight = float(config.get("compatible_orthogonal_weight", 0.05))
    tournament_margin_floor = float(
        config.get("tournament_margin_floor", V32_TOURNAMENT_MARGIN_FLOOR_GRID[0])
    )
    tournament_margin_weight = float(config.get("tournament_margin_weight", 1.0))
    for name, value, positive in [
        ("trust_norm_cap", trust_norm_cap, False),
        ("compatible_nullspace_rtol", compatible_nullspace_rtol, False),
        ("projection_strength", projection_strength, False),
        ("projected_optimizer_lr", projected_optimizer_lr, True),
        ("compatible_gate_weight", compatible_gate_weight, False),
        ("extra_compatible_weight", extra_compatible_weight, False),
        ("target_margin_floor", target_margin_floor, False),
        ("hard_target_margin_weight", hard_target_margin_weight, False),
        ("compatible_orthogonal_weight", compatible_orthogonal_weight, False),
        ("tournament_margin_floor", tournament_margin_floor, False),
        ("tournament_margin_weight", tournament_margin_weight, False),
    ]:
        if not math.isfinite(value) or (positive and value <= 0.0) or (not positive and value < 0.0):
            raise ValueError(f"{name} has invalid value")
    if v42_compatible_dual_enabled:
        for name, value in [
            ("compatible_mse_budget", compatible_mse_budget),
            ("compatible_augmented_weight", compatible_augmented_weight),
            ("compatible_dual_lr", compatible_dual_lr),
            ("compatible_dual_max", compatible_dual_max),
            ("compatible_dual_initial", compatible_dual_lambda),
        ]:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} has invalid value")
        if compatible_dual_max <= 0.0:
            raise ValueError("compatible_dual_max must be positive when V42 dual is enabled")
    if math.isnan(compatible_mse_gate) or compatible_mse_gate < 0.0:
        raise ValueError("compatible_mse_gate has invalid value")
    if trajectory_frontier_enabled:
        for name, value in [
            ("alpha_compatible_mse_soft_gate", alpha_compatible_mse_soft_gate),
            ("target_rank_score_tolerance", target_rank_score_tolerance),
        ]:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} has invalid value")
    if projected_optimizer_epochs <= 0:
        raise ValueError("projected_optimizer_epochs must be positive")

    support = v27_support_tensors_for_source_target(
        source_weights=source,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    tournament_tensors = v32_support_behavior_margin_tensors()
    tournament_tensor_hash = require_sha256_hex(
        tournament_tensors["tensor_hash"],
        field_name="support_tournament_tensor_hash",
    )
    with torch.no_grad():
        source_target_logits = v27_subject_logits_for_inputs(source, support["target_inputs"])
        target_multiplier = v31_support_hardness_multiplier(
            source_target_logits=source_target_logits,
            target_labels=support["target_labels"].to(dtype=torch.float32),
            target_margin_floor=target_margin_floor,
            hard_target_margin_weight=hard_target_margin_weight,
        )

    coordinate_index = torch.tensor(coordinates, dtype=torch.long)
    values = torch.zeros(len(coordinates), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [values],
        lr=projected_optimizer_lr,
        betas=V27_LOCALIZED_OPTIMIZER_BETAS,
        eps=V27_LOCALIZED_OPTIMIZER_EPS,
        weight_decay=0.0,
        amsgrad=False,
    )
    progress_stride = max(1, int(projected_optimizer_epochs) // 5)
    best: tuple[
        float,
        float,
        int,
        torch.Tensor,
        dict[str, Any],
        dict[str, Any],
    ] | None = None
    frontier_candidates: list[dict[str, Any]] = []

    def sparse_delta(current_values: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.float32)
        return delta.index_copy(0, coordinate_index, current_values)

    for epoch in range(1, int(projected_optimizer_epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        raw_delta = sparse_delta(values)
        delta, projection_audit = project_v37_delta_differentiably(
            sparse_delta=raw_delta,
            compatible_jacobian=jacobian,
            compatible_nullspace_rtol=compatible_nullspace_rtol,
            projection_strength=projection_strength,
            trust_norm_cap=trust_norm_cap,
        )
        edited = source + delta
        target_logits = v27_subject_logits_for_inputs(edited, support["target_inputs"])
        conflict_logits = v27_subject_logits_for_inputs(edited, support["conflict_inputs"])
        compatible_logits = v27_subject_logits_for_inputs(edited, support["compatible_inputs"])
        target_labels = support["target_labels"].to(dtype=torch.float32)
        target_bce = F.binary_cross_entropy_with_logits(target_logits, target_labels)
        target_margin_hinge = v30_target_margin_hinge_loss(
            logits=target_logits,
            labels=target_labels,
            margin_floor=target_margin_floor,
        )
        conflict_bce = F.binary_cross_entropy_with_logits(
            conflict_logits,
            support["conflict_target_labels"].to(dtype=torch.float32),
        )
        compatible_mse = F.mse_loss(
            compatible_logits,
            support["compatible_source_logits"].to(dtype=torch.float32),
        )
        if v42_compatible_dual_enabled:
            compatible_constraint = v42_compatible_constraint_terms(
                support_compatible_mse=compatible_mse,
                compatible_mse_budget=compatible_mse_budget,
                compatible_dual_lambda=compatible_dual_lambda,
                compatible_augmented_weight=compatible_augmented_weight,
            )
        else:
            compatible_constraint = {
                "compatible_constraint_penalty": compatible_mse * 0.0,
                "compatible_constraint_residual": compatible_mse * 0.0,
            }
        compatible_orthogonal_loss = v31_compatible_gradient_orthogonal_loss(
            delta=delta,
            g_compatible=compatible_grad,
        )
        tournament = v32_support_tournament_margin_loss(
            margins=v32_support_behavior_margins(
                weights=edited,
                tournament_tensors=tournament_tensors,
            ),
            target_behavior=target_behavior,
            tournament_margin_floor=tournament_margin_floor,
        )
        delta_l2 = torch.mean(delta.pow(2))
        loss = (
            target_multiplier * (
                V29_TARGET_BCE_WEIGHT * target_bce
                + V30_TARGET_MARGIN_WEIGHT * target_margin_hinge
            )
            + V29_CONFLICT_BCE_WEIGHT * conflict_bce
            + V29_COMPATIBLE_PROBE_WEIGHT * compatible_mse
            + extra_compatible_weight * compatible_mse
            + compatible_orthogonal_weight * compatible_orthogonal_loss
            + tournament_margin_weight * tournament["loss"]
            + V29_DELTA_L2_WEIGHT * delta_l2
        )
        if v42_compatible_dual_enabled:
            loss = loss + compatible_constraint["compatible_constraint_penalty"]
        if not torch.isfinite(loss):
            raise ValueError("nonfinite projected support optimizer loss")
        loss.backward()
        if values.grad is None or not torch.isfinite(values.grad).all():
            raise ValueError("nonfinite projected support optimizer gradient")
        torch.nn.utils.clip_grad_norm_([values], V27_LOCALIZED_GRAD_CLIP_NORM)
        optimizer.step()
        with torch.no_grad():
            current_raw_delta = sparse_delta(values).detach().clone().to(dtype=torch.float32)
            clipped_raw_delta = apply_norm_cap(current_raw_delta, max_norm=trust_norm_cap)
            if float(torch.linalg.norm(current_raw_delta).item()) > trust_norm_cap + 1e-8:
                values.copy_(clipped_raw_delta[coordinate_index])
            current_raw_delta = sparse_delta(values).detach().clone().to(dtype=torch.float32)
            current_delta, current_projection_audit = project_v37_delta_differentiably(
                sparse_delta=current_raw_delta,
                compatible_jacobian=jacobian,
                compatible_nullspace_rtol=compatible_nullspace_rtol,
                projection_strength=projection_strength,
                trust_norm_cap=trust_norm_cap,
            )
            current_edited = source + current_delta
            current_target_logits = v27_subject_logits_for_inputs(
                current_edited,
                support["target_inputs"],
            )
            current_conflict_logits = v27_subject_logits_for_inputs(
                current_edited,
                support["conflict_inputs"],
            )
            current_compatible_logits = v27_subject_logits_for_inputs(
                current_edited,
                support["compatible_inputs"],
            )
            current_target_bce = F.binary_cross_entropy_with_logits(
                current_target_logits,
                target_labels,
            )
            current_target_margin_hinge = v30_target_margin_hinge_loss(
                logits=current_target_logits,
                labels=target_labels,
                margin_floor=target_margin_floor,
            )
            current_conflict_bce = F.binary_cross_entropy_with_logits(
                current_conflict_logits,
                support["conflict_target_labels"].to(dtype=torch.float32),
            )
            current_compatible_mse = F.mse_loss(
                current_compatible_logits,
                support["compatible_source_logits"].to(dtype=torch.float32),
            )
            if v42_compatible_dual_enabled:
                current_compatible_constraint = v42_compatible_constraint_terms(
                    support_compatible_mse=current_compatible_mse,
                    compatible_mse_budget=compatible_mse_budget,
                    compatible_dual_lambda=compatible_dual_lambda,
                    compatible_augmented_weight=compatible_augmented_weight,
                )
            else:
                current_compatible_constraint = {
                    "compatible_constraint_penalty": current_compatible_mse * 0.0,
                    "compatible_constraint_residual": current_compatible_mse * 0.0,
                }
            current_compatible_orthogonal_loss = v31_compatible_gradient_orthogonal_loss(
                delta=current_delta,
                g_compatible=compatible_grad,
            )
            current_tournament = v32_support_tournament_margin_loss(
                margins=v32_support_behavior_margins(
                    weights=current_edited,
                    tournament_tensors=tournament_tensors,
                ),
                target_behavior=target_behavior,
                tournament_margin_floor=tournament_margin_floor,
            )
            current_delta_l2 = torch.mean(current_delta.pow(2))
            current_loss = (
                target_multiplier * (
                    V29_TARGET_BCE_WEIGHT * current_target_bce
                    + V30_TARGET_MARGIN_WEIGHT * current_target_margin_hinge
                )
                + V29_CONFLICT_BCE_WEIGHT * current_conflict_bce
                + V29_COMPATIBLE_PROBE_WEIGHT * current_compatible_mse
                + extra_compatible_weight * current_compatible_mse
                + compatible_orthogonal_weight * current_compatible_orthogonal_loss
                + tournament_margin_weight * current_tournament["loss"]
                + V29_DELTA_L2_WEIGHT * current_delta_l2
            )
            if v42_compatible_dual_enabled:
                current_loss = (
                    current_loss
                    + current_compatible_constraint["compatible_constraint_penalty"]
                )
                compatible_dual_lambda = min(
                    compatible_dual_max,
                    max(
                        0.0,
                        compatible_dual_lambda
                        + compatible_dual_lr
                        * float(
                            current_compatible_constraint[
                                "compatible_constraint_residual"
                            ].detach().item()
                        ),
                    ),
                )
            compatible_constraint_residual = float(
                current_compatible_constraint[
                    "compatible_constraint_residual"
                ].detach().item()
            )
            scalar_losses = {
                "compatible_constraint_feasible": bool(
                    float(current_compatible_mse.detach().item()) <= compatible_mse_budget
                ),
                "compatible_constraint_residual": compatible_constraint_residual,
                "compatible_dual_lambda": float(compatible_dual_lambda),
                "compatible_mse_budget": float(compatible_mse_budget),
                "compatible_mse": float(current_compatible_mse.detach().item()),
                "compatible_orthogonal_loss": float(
                    current_compatible_orthogonal_loss.detach().item()
                ),
                "conflict_bce": float(current_conflict_bce.detach().item()),
                "delta_l2": float(current_delta_l2.detach().item()),
                "loss": float(current_loss.detach().item()),
                "support_compatible_mse": float(current_compatible_mse.detach().item()),
                "support_runner_margin": float(
                    current_tournament["support_runner_margin"].detach().item()
                ),
                "support_target_margin": float(
                    current_tournament["support_target_margin"].detach().item()
                ),
                "support_tournament_margin": float(
                    current_tournament["support_tournament_margin"].detach().item()
                ),
                "target_bce": float(current_target_bce.detach().item()),
                "target_margin_hinge": float(
                    current_target_margin_hinge.detach().item()
                ),
                "tournament_margin_hinge": float(
                    current_tournament["loss"].detach().item()
                ),
            }
            support_score = v38_projected_optimizer_support_score(
                support_target_margin=scalar_losses["support_target_margin"],
                support_tournament_margin=scalar_losses["support_tournament_margin"],
                support_compatible_mse=scalar_losses["support_compatible_mse"],
                loss=scalar_losses["loss"],
                target_margin_floor=target_margin_floor,
                tournament_margin_floor=tournament_margin_floor,
                compatible_mse_gate=compatible_mse_gate,
                compatible_gate_weight=compatible_gate_weight,
            )
            candidate = (
                float(support_score),
                scalar_losses["support_compatible_mse"],
                int(epoch),
                current_delta.detach().clone(),
                dict(scalar_losses),
                dict(current_projection_audit),
            )
            if trajectory_frontier_enabled:
                frontier_candidates.append({
                    "candidate_hash": stable_hash_json({
                        "epoch": int(epoch),
                        "projection_audit": current_projection_audit,
                        "scalar_losses": scalar_losses,
                    }),
                    "delta": current_delta.detach().clone(),
                    "epoch": int(epoch),
                    "loss": scalar_losses["loss"],
                    "preservation_energy_ratio": float(
                        current_projection_audit["preservation_energy_ratio"]
                    ),
                    "projection_audit": dict(current_projection_audit),
                    "projected_delta_norm": float(
                        current_projection_audit["projected_delta_norm"]
                    ),
                    "scalar_losses": dict(scalar_losses),
                    "compatible_constraint_residual": scalar_losses[
                        "compatible_constraint_residual"
                    ],
                    "compatible_dual_lambda": scalar_losses["compatible_dual_lambda"],
                    "support_compatible_mse": scalar_losses["support_compatible_mse"],
                    "support_runner_margin": scalar_losses["support_runner_margin"],
                    "support_target_margin": scalar_losses["support_target_margin"],
                    "support_tournament_margin": (
                        scalar_losses["support_tournament_margin"]
                    ),
                })
            if best is None or candidate[:3] < best[:3]:
                best = candidate
            if progress_log_path is not None and (
                epoch % progress_stride == 0
                or epoch == int(projected_optimizer_epochs)
            ):
                progress_hash = stable_hash_json({
                    "epoch": int(epoch),
                    "projection_audit": current_projection_audit,
                    "scalar_losses": scalar_losses,
                })
                progress_extra = redact_v37_projected_optimizer_progress_event({
                    **scalar_losses,
                    **current_projection_audit,
                    **({"record_id_hash": record_id_hash} if record_id_hash else {}),
                    "final_loss": scalar_losses["loss"],
                    "optimization_steps": int(epoch),
                    "optimizer_audit_hash": progress_hash,
                    "selected_config_hash": checked_config_hash or ("0" * 64),
                })
                if v42_compatible_dual_enabled:
                    progress_extra.update({
                        "compatible_constraint_feasible": bool(
                            scalar_losses["compatible_constraint_feasible"]
                        ),
                        "compatible_constraint_residual": float(
                            scalar_losses["compatible_constraint_residual"]
                        ),
                        "compatible_dual_lambda": float(
                            scalar_losses["compatible_dual_lambda"]
                        ),
                        "compatible_mse_budget": float(
                            scalar_losses["compatible_mse_budget"]
                        ),
                    })
                record_progress_event(
                    progress_log_path,
                    event=f"{projected_optimizer_event_prefix}_progress",
                    started_at_monotonic=started_at_monotonic,
                    now_monotonic=now_monotonic,
                    extra=progress_extra,
                )

    if best is None:
        raise ValueError("projected support optimizer produced no candidate")
    frontier_selection: dict[str, Any] | None = None
    if trajectory_frontier_enabled:
        if not frontier_candidates:
            raise ValueError("trajectory frontier mode produced no candidates")
        if v42_compatible_dual_enabled:
            frontier_selection = select_v42_compatible_dual_frontier_candidate(
                candidates=frontier_candidates,
                compatible_mse_budget=compatible_mse_budget,
                target_margin_floor=target_margin_floor,
                target_rank_score_tolerance=target_rank_score_tolerance,
                tournament_margin_floor=tournament_margin_floor,
            )
        else:
            frontier_selection = select_v41_trajectory_frontier_candidate(
                candidates=frontier_candidates,
                compatible_mse_soft_gate=alpha_compatible_mse_soft_gate,
                target_margin_floor=target_margin_floor,
                target_rank_score_tolerance=target_rank_score_tolerance,
                tournament_margin_floor=tournament_margin_floor,
            )
        selected_frontier_epoch = int(frontier_selection["epoch"])
        selected_frontier_hash = str(
            frontier_candidates[int(frontier_selection["candidate_index"])]["candidate_hash"]
        )
        selected_frontier_delta = frontier_candidates[
            int(frontier_selection["candidate_index"])
        ]["delta"]
        selected_frontier_losses = dict(
            frontier_candidates[int(frontier_selection["candidate_index"])]["scalar_losses"]
        )
        selected_frontier_projection = dict(
            frontier_candidates[int(frontier_selection["candidate_index"])]["projection_audit"]
        )
        best = (
            float("inf"),
            float(frontier_selection["support_compatible_mse"]),
            selected_frontier_epoch,
            selected_frontier_delta.detach().clone(),
            selected_frontier_losses,
            selected_frontier_projection,
        )
    _score, _compatible_mse, best_epoch, best_delta, best_losses, best_projection = best
    audit = {
        **best_projection,
        "coordinate_hash": checked_coordinate_hash,
        "final_loss": float(best_losses["loss"]),
        "optimization_steps": int(best_epoch),
        "support_compatible_mse": float(best_losses["support_compatible_mse"]),
        "support_runner_margin": float(best_losses["support_runner_margin"]),
        "support_target_margin": float(best_losses["support_target_margin"]),
        "support_tournament_margin": float(best_losses["support_tournament_margin"]),
        "support_tournament_tensor_hash": tournament_tensor_hash,
    }
    if frontier_selection is not None:
        frontier_candidates_hash = stable_hash_json([
            {
                "candidate_hash": item["candidate_hash"],
                "epoch": item["epoch"],
                "loss": item["loss"],
                "preservation_energy_ratio": item["preservation_energy_ratio"],
                "projected_delta_norm": item["projected_delta_norm"],
                **({
                    "compatible_constraint_residual": item[
                        "compatible_constraint_residual"
                    ],
                    "compatible_dual_lambda": item["compatible_dual_lambda"],
                } if v42_compatible_dual_enabled else {}),
                "support_compatible_mse": item["support_compatible_mse"],
                "support_runner_margin": item["support_runner_margin"],
                "support_target_margin": item["support_target_margin"],
                "support_tournament_margin": item["support_tournament_margin"],
            }
            for item in frontier_candidates
        ])
        audit.update({
            "best_target_rank_score": float(frontier_selection["best_target_rank_score"]),
            "compatible_gap": float(frontier_selection["compatible_gap"]),
            "frontier_candidates_hash": frontier_candidates_hash,
            "target_feasible": bool(frontier_selection["target_feasible"]),
            "target_gap": float(frontier_selection["target_gap"]),
            "target_rank_score": float(frontier_selection["target_rank_score"]),
            "target_rank_score_tolerance": target_rank_score_tolerance,
            "trajectory_frontier_candidate_count": len(frontier_candidates),
            "trajectory_frontier_selected_epoch": int(frontier_selection["epoch"]),
            "trajectory_frontier_selected_hash": selected_frontier_hash,
            "trajectory_frontier_selection_hash": stable_hash_json(frontier_selection),
            "trajectory_frontier_selection_mode": str(
                frontier_selection["selection_mode"]
            ),
            "tournament_gap": float(frontier_selection["tournament_gap"]),
            "within_target_tolerance_count": int(
                frontier_selection["within_target_tolerance_count"]
            ),
        })
        if v42_compatible_dual_enabled:
            audit.update({
                "compatible_constraint_feasible": bool(
                    frontier_selection["compatible_constraint_feasible"]
                ),
                "compatible_constraint_residual": float(
                    frontier_selection["compatible_constraint_residual"]
                ),
                "compatible_dual_lambda": float(
                    frontier_selection["compatible_dual_lambda"]
                ),
                "compatible_dual_update_source": "post_step_projected_delta",
                "compatible_mse_budget": float(frontier_selection["compatible_mse_budget"]),
                "localized_feasible_count": int(
                    frontier_selection["localized_feasible_count"]
                ),
            })
        if progress_log_path is not None:
            frontier_payload = {
                **frontier_selection,
                **({"record_id_hash": record_id_hash} if record_id_hash else {}),
                "frontier_candidates_hash": frontier_candidates_hash,
                "selected_config_hash": checked_config_hash or ("0" * 64),
                "trajectory_frontier_selected_epoch": int(frontier_selection["epoch"]),
            }
            if v42_compatible_dual_enabled:
                frontier_extra = redact_v42_compatible_dual_frontier_progress_event(
                    frontier_payload
                )
            else:
                frontier_extra = redact_v41_trajectory_frontier_progress_event(
                    frontier_payload
                )
            record_progress_event(
                progress_log_path,
                event=f"{trajectory_frontier_event_prefix}_selected",
                started_at_monotonic=started_at_monotonic,
                now_monotonic=now_monotonic,
                extra=frontier_extra,
            )
    audit["optimizer_audit_hash"] = stable_hash_json(audit)
    finite_failures = recursive_numeric_finiteness_failures(audit)
    if finite_failures:
        raise ValueError("nonfinite V37 optimizer audit: " + ", ".join(finite_failures[:5]))
    if progress_log_path is not None:
        completed_extra = redact_v37_projected_optimizer_progress_event({
            **audit,
            **({"record_id_hash": record_id_hash} if record_id_hash else {}),
            "selected_config_hash": checked_config_hash or ("0" * 64),
        })
        if v42_compatible_dual_enabled:
            completed_extra.update({
                "compatible_constraint_feasible": bool(
                    audit["compatible_constraint_feasible"]
                ),
                "compatible_constraint_residual": float(
                    audit["compatible_constraint_residual"]
                ),
                "compatible_dual_lambda": float(audit["compatible_dual_lambda"]),
                "compatible_dual_update_source": str(
                    audit["compatible_dual_update_source"]
                ),
                "compatible_mse_budget": float(audit["compatible_mse_budget"]),
                "localized_feasible_count": int(audit["localized_feasible_count"]),
            })
        record_progress_event(
            progress_log_path,
            event=f"{projected_optimizer_event_prefix}_completed",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=completed_extra,
        )
    return {
        "audit": audit,
        "delta": best_delta.detach().clone().to(dtype=torch.float32),
    }


def v32_support_behavior_margin_tensor_hash(
    *,
    by_behavior: Mapping[str, Mapping[str, torch.Tensor]],
    counts_by_behavior: Mapping[str, Mapping[str, int]],
) -> str:
    payload: dict[str, Any] = {
        "counts_by_behavior": {
            str(pattern): {
                "negative": int(counts["negative"]),
                "positive": int(counts["positive"]),
            }
            for pattern, counts in sorted(counts_by_behavior.items())
        },
        "scope": "v32_support_tournament_margin_tensors",
        "tensor_hashes_by_behavior": {},
    }
    tensor_hashes_by_behavior: dict[str, dict[str, str]] = {}
    for pattern, tensors in sorted(by_behavior.items()):
        positive = tensors["positive_inputs"].detach().clone().to(dtype=torch.float32)
        negative = tensors["negative_inputs"].detach().clone().to(dtype=torch.float32)
        tensor_hashes_by_behavior[str(pattern)] = {
            "negative_inputs_sha256": stable_hash_json(tensor_to_hashable(negative)),
            "positive_inputs_sha256": stable_hash_json(tensor_to_hashable(positive)),
        }
    payload["tensor_hashes_by_behavior"] = tensor_hashes_by_behavior
    return stable_hash_json(payload)


def v32_support_behavior_margin_tensors() -> dict[str, Any]:
    suite = v23.v16.v15.v10.evaluation_suite()
    by_behavior: dict[str, dict[str, torch.Tensor]] = {}
    counts_by_behavior: dict[str, dict[str, int]] = {}
    for pattern in PATTERNS:
        support = suite["support"][pattern]
        positive_inputs = v23.v16.v15.v10.decoder_v1.sequence_tensor(
            support["positive"]
        ).to(dtype=torch.float32)
        negative_inputs = v23.v16.v15.v10.decoder_v1.sequence_tensor(
            support["negative"]
        ).to(dtype=torch.float32)
        by_behavior[str(pattern)] = {
            "negative_inputs": negative_inputs,
            "positive_inputs": positive_inputs,
        }
        counts_by_behavior[str(pattern)] = {
            "negative": int(negative_inputs.shape[0]),
            "positive": int(positive_inputs.shape[0]),
        }
    return {
        "by_behavior": by_behavior,
        "counts_by_behavior": counts_by_behavior,
        "tensor_hash": v32_support_behavior_margin_tensor_hash(
            by_behavior=by_behavior,
            counts_by_behavior=counts_by_behavior,
        ),
    }


def v32_support_behavior_margins(
    *,
    weights: torch.Tensor,
    tournament_tensors: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    margins: dict[str, torch.Tensor] = {}
    for pattern in PATTERNS:
        tensors = tournament_tensors["by_behavior"][str(pattern)]
        positive_logits = v27_subject_logits_for_inputs(weights, tensors["positive_inputs"])
        negative_logits = v27_subject_logits_for_inputs(weights, tensors["negative_inputs"])
        margin = torch.sigmoid(positive_logits).mean() - torch.sigmoid(negative_logits).mean()
        if not torch.isfinite(margin):
            raise ValueError("nonfinite support behavior margin")
        margins[str(pattern)] = margin
    return margins


def v32_support_tournament_margin_loss(
    *,
    margins: Mapping[str, torch.Tensor],
    target_behavior: str,
    tournament_margin_floor: float,
) -> dict[str, torch.Tensor]:
    target = str(target_behavior)
    if target not in margins:
        raise ValueError("target behavior missing from support margins")
    floor = float(tournament_margin_floor)
    if floor < 0.0 or not math.isfinite(floor):
        raise ValueError("tournament_margin_floor must be finite and nonnegative")
    runner_values = [
        value.to(dtype=torch.float32)
        for pattern, value in margins.items()
        if str(pattern) != target
    ]
    if not runner_values:
        raise ValueError("support tournament requires at least one runner")
    runner_margin = torch.stack(runner_values).max()
    target_margin = margins[target].to(dtype=torch.float32)
    support_tournament_margin = target_margin - runner_margin
    loss = torch.relu(
        torch.tensor(floor, dtype=torch.float32) - support_tournament_margin
    )
    if not all(
        torch.isfinite(value)
        for value in [target_margin, runner_margin, support_tournament_margin, loss]
    ):
        raise ValueError("nonfinite support tournament margin loss")
    return {
        "loss": loss,
        "support_runner_margin": runner_margin,
        "support_target_margin": target_margin,
        "support_tournament_margin": support_tournament_margin,
    }


def solve_v32_support_tournament_sparse_edit(
    *,
    compatible_gradient: torch.Tensor,
    coordinate_hash: str,
    selected_coordinates: Sequence[int],
    config: Mapping[str, Any],
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
    selected_config_hash: str | None = None,
) -> dict[str, Any]:
    checked_coordinate_hash = require_sha256_hex(
        coordinate_hash,
        field_name="coordinate_hash",
    )
    source = source_weights.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(source.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("source weights have wrong dimension")
    if not torch.isfinite(source).all():
        raise ValueError("nonfinite source weights")
    compatible_grad = compatible_gradient.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(compatible_grad.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("compatible_gradient has wrong dimension")
    if not torch.isfinite(compatible_grad).all():
        raise ValueError("nonfinite compatible_gradient")
    coordinates = [int(index) for index in selected_coordinates]
    if not coordinates:
        raise ValueError("selected_coordinates must be nonempty")
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("selected_coordinates must be unique")
    if any(index < 0 or index >= SOURCE_WEIGHT_DIM for index in coordinates):
        raise ValueError("selected coordinate out of range")
    trust_norm_cap = float(config.get("trust_norm_cap", V32_TRUST_NORM_CAP))
    compatible_floor = float(config.get("compatible_floor", V32_COMPATIBLE_FLOOR))
    extra_compatible_weight = float(
        config.get("extra_compatible_weight", V32_EXTRA_COMPATIBLE_WEIGHT)
    )
    target_margin_floor = float(config.get("target_margin_floor", V32_TARGET_MARGIN_FLOOR))
    hard_target_margin_weight = float(
        config.get("hard_target_margin_weight", V32_HARD_TARGET_MARGIN_WEIGHT)
    )
    compatible_orthogonal_weight = float(
        config.get(
            "compatible_orthogonal_weight",
            V32_COMPATIBLE_ORTHOGONAL_WEIGHT_GRID[0],
        )
    )
    sign_conflict_penalty = float(config.get("sign_conflict_penalty", V32_SIGN_CONFLICT_PENALTY))
    sparse_top_k = int(config.get("sparse_top_k", len(coordinates)))
    tournament_margin_floor = float(
        config.get("tournament_margin_floor", V32_TOURNAMENT_MARGIN_FLOOR_GRID[0])
    )
    tournament_margin_weight = float(
        config.get("tournament_margin_weight", V32_TOURNAMENT_MARGIN_WEIGHT_GRID[0])
    )
    for name, value, positive in [
        ("trust_norm_cap", trust_norm_cap, False),
        ("compatible_floor", compatible_floor, True),
        ("extra_compatible_weight", extra_compatible_weight, False),
        ("target_margin_floor", target_margin_floor, False),
        ("hard_target_margin_weight", hard_target_margin_weight, False),
        ("compatible_orthogonal_weight", compatible_orthogonal_weight, False),
        ("sign_conflict_penalty", sign_conflict_penalty, False),
        ("tournament_margin_floor", tournament_margin_floor, False),
        ("tournament_margin_weight", tournament_margin_weight, False),
    ]:
        if not math.isfinite(value) or (positive and value <= 0.0) or (not positive and value < 0.0):
            raise ValueError(f"{name} has invalid value")
    support = v27_support_tensors_for_source_target(
        source_weights=source,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    tournament_tensors = v32_support_behavior_margin_tensors()
    tournament_tensor_hash = require_sha256_hex(
        tournament_tensors["tensor_hash"],
        field_name="support_tournament_tensor_hash",
    )
    with torch.no_grad():
        source_target_logits = v27_subject_logits_for_inputs(source, support["target_inputs"])
        target_multiplier = v31_support_hardness_multiplier(
            source_target_logits=source_target_logits,
            target_labels=support["target_labels"].to(dtype=torch.float32),
            target_margin_floor=target_margin_floor,
            hard_target_margin_weight=hard_target_margin_weight,
        )
    coordinate_index = torch.tensor(coordinates, dtype=torch.long)
    values = torch.zeros(len(coordinates), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [values],
        lr=V29_BREADTH_FIRST_LR,
        betas=V27_LOCALIZED_OPTIMIZER_BETAS,
        eps=V27_LOCALIZED_OPTIMIZER_EPS,
        weight_decay=0.0,
        amsgrad=False,
    )
    trace: list[dict[str, Any]] = []
    best: tuple[float, float, int, torch.Tensor, dict[str, Any]] | None = None

    def sparse_delta(current_values: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.float32)
        return delta.index_copy(0, coordinate_index, current_values)

    for epoch in range(1, V29_BREADTH_FIRST_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = sparse_delta(values)
        edited = source + delta
        target_logits = v27_subject_logits_for_inputs(edited, support["target_inputs"])
        conflict_logits = v27_subject_logits_for_inputs(edited, support["conflict_inputs"])
        compatible_logits = v27_subject_logits_for_inputs(edited, support["compatible_inputs"])
        target_labels = support["target_labels"].to(dtype=torch.float32)
        target_bce = F.binary_cross_entropy_with_logits(target_logits, target_labels)
        target_margin_hinge = v30_target_margin_hinge_loss(
            logits=target_logits,
            labels=target_labels,
            margin_floor=target_margin_floor,
        )
        conflict_bce = F.binary_cross_entropy_with_logits(
            conflict_logits,
            support["conflict_target_labels"].to(dtype=torch.float32),
        )
        compatible_mse = F.mse_loss(
            compatible_logits,
            support["compatible_source_logits"].to(dtype=torch.float32),
        )
        compatible_orthogonal_loss = v31_compatible_gradient_orthogonal_loss(
            delta=delta,
            g_compatible=compatible_grad,
        )
        tournament = v32_support_tournament_margin_loss(
            margins=v32_support_behavior_margins(
                weights=edited,
                tournament_tensors=tournament_tensors,
            ),
            target_behavior=target_behavior,
            tournament_margin_floor=tournament_margin_floor,
        )
        delta_l2 = torch.mean(delta.pow(2))
        loss = (
            target_multiplier * (
                V29_TARGET_BCE_WEIGHT * target_bce
                + V30_TARGET_MARGIN_WEIGHT * target_margin_hinge
            )
            + V29_CONFLICT_BCE_WEIGHT * conflict_bce
            + V29_COMPATIBLE_PROBE_WEIGHT * compatible_mse
            + extra_compatible_weight * compatible_mse
            + compatible_orthogonal_weight * compatible_orthogonal_loss
            + tournament_margin_weight * tournament["loss"]
            + V29_DELTA_L2_WEIGHT * delta_l2
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite support tournament sparse loss")
        loss.backward()
        if values.grad is None or not torch.isfinite(values.grad).all():
            raise ValueError("nonfinite support tournament sparse gradient")
        torch.nn.utils.clip_grad_norm_([values], V27_LOCALIZED_GRAD_CLIP_NORM)
        optimizer.step()
        with torch.no_grad():
            current_delta = sparse_delta(values).detach().clone().to(dtype=torch.float32)
            clipped_delta = apply_norm_cap(current_delta, max_norm=trust_norm_cap)
            if float(torch.linalg.norm(current_delta).item()) > trust_norm_cap + 1e-8:
                values.copy_(clipped_delta[coordinate_index])
            if not torch.isfinite(values).all():
                raise ValueError("nonfinite support tournament sparse values")
            current_delta = sparse_delta(values).detach().clone().to(dtype=torch.float32)
            current_delta_norm = float(torch.linalg.norm(current_delta).item())
            scalar_losses = {
                "compatible_mse": float(compatible_mse.detach().item()),
                "compatible_orthogonal_loss": float(
                    compatible_orthogonal_loss.detach().item()
                ),
                "conflict_bce": float(conflict_bce.detach().item()),
                "delta_l2": float(delta_l2.detach().item()),
                "loss": float(loss.detach().item()),
                "support_runner_margin": float(
                    tournament["support_runner_margin"].detach().item()
                ),
                "support_target_margin": float(
                    tournament["support_target_margin"].detach().item()
                ),
                "support_tournament_margin": float(
                    tournament["support_tournament_margin"].detach().item()
                ),
                "target_bce": float(target_bce.detach().item()),
                "target_margin_hinge": float(target_margin_hinge.detach().item()),
                "tournament_margin_hinge": float(tournament["loss"].detach().item()),
            }
            if epoch % 10 == 0 or epoch == V29_BREADTH_FIRST_EPOCHS:
                progress_event = redact_v32_support_tournament_progress_event({
                    "compatible_floor": compatible_floor,
                    "compatible_orthogonal_loss": scalar_losses[
                        "compatible_orthogonal_loss"
                    ],
                    "compatible_orthogonal_weight": compatible_orthogonal_weight,
                    "coordinate_hash": checked_coordinate_hash,
                    "delta_norm": current_delta_norm,
                    "epoch": epoch,
                    "extra_compatible_weight": extra_compatible_weight,
                    "hard_target_margin_weight": hard_target_margin_weight,
                    "loss": scalar_losses["loss"],
                    "selected_coordinate_count": len(coordinates),
                    "sign_conflict_penalty": sign_conflict_penalty,
                    "sparse_top_k": sparse_top_k,
                    "step": epoch,
                    "support_runner_margin": scalar_losses["support_runner_margin"],
                    "support_target_margin": scalar_losses["support_target_margin"],
                    "support_tournament_margin": scalar_losses[
                        "support_tournament_margin"
                    ],
                    "support_tournament_tensor_hash": tournament_tensor_hash,
                    "target_margin_floor": target_margin_floor,
                    "target_margin_hinge": scalar_losses["target_margin_hinge"],
                    "target_multiplier": float(target_multiplier.item()),
                    "tournament_margin_floor": tournament_margin_floor,
                    "tournament_margin_hinge": scalar_losses["tournament_margin_hinge"],
                    "tournament_margin_weight": tournament_margin_weight,
                    "trust_norm_cap": trust_norm_cap,
                })
                trace.append(progress_event)
                if progress_log_path is not None:
                    event_extra = dict(progress_event)
                    if record_id_hash is not None:
                        event_extra["record_id_hash"] = require_sha256_hex(
                            record_id_hash,
                            field_name="record_id_hash",
                        )
                    if selected_config_hash is not None:
                        event_extra["selected_config_hash"] = require_sha256_hex(
                            selected_config_hash,
                            field_name="selected_config_hash",
                        )
                    record_progress_event(
                        progress_log_path,
                        event="v32_support_tournament_optimizer_progress",
                        started_at_monotonic=started_at_monotonic,
                        now_monotonic=now_monotonic,
                        extra=event_extra,
                    )
            objective_key = (scalar_losses["loss"], current_delta_norm, epoch)
            if best is None or objective_key < best[:3]:
                best = (
                    scalar_losses["loss"],
                    current_delta_norm,
                    epoch,
                    current_delta,
                    scalar_losses,
                )
    if best is None:
        raise ValueError("support tournament optimizer produced no candidate")
    best_loss, best_delta_norm, best_epoch, best_delta, best_losses = best
    clipped_delta = apply_norm_cap(best_delta, max_norm=trust_norm_cap)
    audit = {
        "best_epoch": int(best_epoch),
        "compatible_floor": compatible_floor,
        "compatible_orthogonal_weight": compatible_orthogonal_weight,
        "coordinate_hash": checked_coordinate_hash,
        "delta_norm_before_hard_cap": float(best_delta_norm),
        "delta_sha256": stable_hash_json(tensor_to_hashable(clipped_delta)),
        "experiment_variant": V32_EXPERIMENT_VARIANT,
        "extra_compatible_weight": extra_compatible_weight,
        "hard_norm_clipped": bool(
            float(torch.linalg.norm(best_delta).item()) > trust_norm_cap + 1e-8
        ),
        "hard_target_margin_weight": hard_target_margin_weight,
        "matched_edit_source": V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE,
        "optimization_boundary": v32_support_tournament_optimization_boundary(),
        "optimization_split": "support",
        "optimizer": "AdamW",
        "optimizer_trace_hash": stable_hash_json(trace),
        "proof_split": "heldout",
        "selected_coordinate_count": len(coordinates),
        "sign_conflict_penalty": sign_conflict_penalty,
        "sparse_top_k": sparse_top_k,
        "support_objective": float(best_loss),
        "support_objective_is_proof_metric": False,
        "support_scalar_losses": best_losses,
        "support_tournament_tensor_hash": tournament_tensor_hash,
        "target_margin_floor": target_margin_floor,
        "target_margin_weight": V30_TARGET_MARGIN_WEIGHT,
        "target_multiplier": float(target_multiplier.item()),
        "tournament_margin_floor": tournament_margin_floor,
        "tournament_margin_weight": tournament_margin_weight,
        "trust_norm_cap": trust_norm_cap,
    }
    finite_failures = recursive_numeric_finiteness_failures(audit)
    if finite_failures:
        raise ValueError("nonfinite support tournament audit: " + ", ".join(finite_failures[:5]))
    if progress_log_path is not None:
        completed_extra = redact_v32_support_tournament_progress_event({
            "coordinate_hash": checked_coordinate_hash,
            "delta_norm": float(torch.linalg.norm(clipped_delta).item()),
            "delta_sha256": audit["delta_sha256"],
            "epoch": int(best_epoch),
            "loss": float(best_loss),
            "selected_coordinate_count": len(coordinates),
            "sparse_top_k": sparse_top_k,
            "step": int(best_epoch),
            "support_tournament_margin": best_losses["support_tournament_margin"],
            "support_tournament_tensor_hash": tournament_tensor_hash,
            "target_margin_floor": target_margin_floor,
            "target_margin_hinge": best_losses["target_margin_hinge"],
            "target_multiplier": float(target_multiplier.item()),
            "tournament_margin_floor": tournament_margin_floor,
            "tournament_margin_hinge": best_losses["tournament_margin_hinge"],
            "tournament_margin_weight": tournament_margin_weight,
            "trust_norm_cap": trust_norm_cap,
        })
        if record_id_hash is not None:
            completed_extra["record_id_hash"] = require_sha256_hex(
                record_id_hash,
                field_name="record_id_hash",
            )
        if selected_config_hash is not None:
            completed_extra["selected_config_hash"] = require_sha256_hex(
                selected_config_hash,
                field_name="selected_config_hash",
            )
        record_progress_event(
            progress_log_path,
            event="v32_support_tournament_optimizer_completed",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=completed_extra,
        )
    return {
        "audit": audit,
        "delta": clipped_delta,
    }


def redact_v27_optimizer_progress_event(payload: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    if "basis_hash" in payload:
        redacted["basis_hash"] = require_sha256_hex(
            payload["basis_hash"],
            field_name="basis_hash",
        )
    if "delta_norm" in payload:
        redacted["delta_norm"] = float(payload["delta_norm"])
    if "loss" in payload:
        redacted["loss"] = float(payload["loss"])
    if "step" in payload:
        redacted["step"] = int(payload["step"])
    finite_values = [
        value for key, value in redacted.items()
        if key in {"delta_norm", "loss"}
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted


def shuffled_signature_derangement_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
    rung_index: int,
    source_behavior: str,
    config_hash: str,
) -> list[int]:
    if len(rows) < 2:
        raise ValueError("shuffled signature derangement requires group size >= 2")
    sorted_positions = sorted(
        range(len(rows)),
        key=lambda index: (
            str(rows[index].get("source_behavior", "")),
            str(rows[index].get("target_behavior", "")),
            str(rows[index].get("subject_id", "")),
        ),
    )
    seed = stable_torch_seed({
        "config_hash": str(config_hash),
        "rung_index": int(rung_index),
        "scope": "v25_shuffled_signature_order",
        "source": str(source_behavior),
        "split": str(split_name),
    })
    generator = torch.Generator()
    generator.manual_seed(seed)
    for _ in range(128):
        perm = torch.randperm(len(sorted_positions), generator=generator).tolist()
        if all(int(perm_index) != index for index, perm_index in enumerate(perm)):
            replacements = [0] * len(rows)
            for sorted_index, perm_index in enumerate(perm):
                original_position = sorted_positions[sorted_index]
                replacement_position = sorted_positions[int(perm_index)]
                replacements[original_position] = replacement_position
            return replacements
    raise ValueError("failed to construct shuffled signature derangement")


def assert_no_forbidden_final_raw_paths(paths: Sequence[Path | str]) -> None:
    forbidden = {V25_FINAL_RAW.resolve(), *(Path(path).resolve() for path in PRIOR_FINAL_RAW_PATHS)}
    for path in paths:
        candidate = Path(path).resolve()
        if candidate in forbidden or candidate.name == "final_subjects.json":
            raise ValueError("final raw path access is forbidden")


def load_v25_source_pool_subjects(path: Path | str) -> dict[str, Any]:
    pool_path = Path(path)
    assert_no_forbidden_final_raw_paths([pool_path])
    payload = json.loads(pool_path.read_text())
    if isinstance(payload, list):
        subjects = payload
        pool_payload_sha256 = stable_hash_json({"record_count": len(subjects)})
    elif isinstance(payload, Mapping):
        subjects = payload.get("records")
        if not isinstance(subjects, list):
            raise ValueError("source pool records payload must be a list")
        redacted_payload = {
            str(key): value
            for key, value in payload.items()
            if str(key) != "records"
        }
        redacted_payload["record_count"] = len(subjects)
        pool_payload_sha256 = stable_hash_json(redacted_payload)
    else:
        raise ValueError("source pool payload must be a list or mapping")
    counts_by_behavior: dict[str, int] = {}
    for record in subjects:
        if not isinstance(record, Mapping):
            raise ValueError("source pool records must be mappings")
        behavior = behavior_for_record(record)
        counts_by_behavior[behavior] = counts_by_behavior.get(behavior, 0) + 1
    return {
        "counts_by_behavior": counts_by_behavior,
        "path_sha256": stable_hash_json(str(pool_path.resolve())),
        "pool_file_sha256": sha256_file(pool_path),
        "pool_payload_sha256": pool_payload_sha256,
        "record_count": len(subjects),
        "subjects": subjects,
    }


def redact_v25_loaded_pool_summary(loaded_pool: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "counts_by_behavior": dict(loaded_pool.get("counts_by_behavior", {})),
        "path_sha256": str(loaded_pool["path_sha256"]),
        "pool_file_sha256": str(loaded_pool["pool_file_sha256"]),
        "record_count": int(loaded_pool["record_count"]),
    }


def prepare_v25_development_pool_inputs(
    *,
    pool_dir: Path,
    progress_log_path: Path,
    started_at_monotonic: float,
    now_monotonic: Any | None = None,
) -> dict[str, Any]:
    train = load_v25_source_pool_subjects(pool_dir / "train_subjects.json")
    development = load_v25_source_pool_subjects(pool_dir / "development_subjects.json")
    record_progress_event(
        progress_log_path,
        event="development_inputs_loaded",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "development_pool": redact_v25_loaded_pool_summary(development),
            "train_pool": redact_v25_loaded_pool_summary(train),
        },
    )
    return {
        "development": development,
        "train": train,
    }


def forbidden_final_redacted_keys(payload: Mapping[str, Any]) -> list[str]:
    forbidden: list[str] = []

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                key_text = str(key)
                path = f"{prefix}.{key_text}" if prefix else key_text
                if key_text in RECURSIVE_FORBIDDEN_FINAL_DETAIL_KEY_TERMS:
                    forbidden.append(path)
                visit(nested, path)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f"{prefix}[{index}]")

    for key, value in payload.items():
        key_text = str(key)
        if key_text not in FINAL_REDACTED_ALLOWED_TOP_LEVEL_KEYS:
            forbidden.append(key_text)
        if key_text in RECURSIVE_FORBIDDEN_FINAL_DETAIL_KEY_TERMS:
            forbidden.append(key_text)
        visit(value, key_text)

    summary = payload.get("summary", {})
    if isinstance(summary, Mapping):
        for key in summary:
            if str(key) not in FINAL_REDACTED_ALLOWED_SUMMARY_KEYS:
                forbidden.append(f"summary.{key}")
    return sorted(set(forbidden))


def forbidden_combined_final_summary_keys(payload: Mapping[str, Any]) -> list[str]:
    return sorted(key for key in payload if key not in FINAL_COMBINED_SUMMARY_ALLOWED_KEYS)


def tensor_normalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value.to(dtype=torch.float32).reshape(-1) - mean.reshape(-1)) / std.reshape(-1)


def fit_v25_activation_statistics(
    train_subjects: Sequence[Mapping[str, Any]],
    *,
    probe_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    descriptors: dict[str, torch.Tensor] = {}
    rows = []
    for record in train_subjects:
        subject_id = str(record["subject_id"])
        descriptor = activation_descriptor_for_weights(
            record_weights_tensor(record),
            probe_examples=probe_examples,
        )
        if not torch.isfinite(descriptor).all():
            raise ValueError(f"nonfinite activation descriptor for train subject {subject_id}")
        descriptors[subject_id] = descriptor.detach().to(dtype=torch.float32)
        rows.append(descriptors[subject_id])
    if not rows:
        raise ValueError("at least one train subject is required for activation statistics")
    stacked = torch.stack(rows).to(dtype=torch.float32)
    mean, std, zero_std_count = safe_mean_std(stacked)
    descriptor_norm_hash = stable_hash_json({
        "activation_descriptor_mean": tensor_to_hashable(mean),
        "activation_descriptor_std": tensor_to_hashable(std),
        "scope": "v25_activation_descriptor_normalization",
    })
    return {
        "activation_descriptor_by_subject": descriptors,
        "activation_descriptor_count": len(rows),
        "activation_descriptor_mean": mean,
        "activation_descriptor_std": std,
        "activation_descriptor_zero_std_count": int(zero_std_count),
        "descriptor_norm_hash": descriptor_norm_hash,
        "probe_examples": list(probe_examples),
        "probe_examples_hash": stable_hash_json(list(probe_examples)),
    }


def behavior_for_record(record: Mapping[str, Any]) -> str:
    pattern = record.get("pattern")
    if isinstance(pattern, str) and pattern in PATTERNS:
        return pattern
    return v24.subject_behavior(record)


def fit_v25_train_statistics(
    train_subjects: Sequence[Mapping[str, Any]],
    *,
    probe_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stats = fit_v25_activation_statistics(
        train_subjects,
        probe_examples=probe_examples,
    )
    grouped: dict[str, list[torch.Tensor]] = {pattern: [] for pattern in PATTERNS}
    for record in train_subjects:
        subject_id = str(record["subject_id"])
        behavior = behavior_for_record(record)
        grouped[behavior].append(tensor_normalize(
            stats["activation_descriptor_by_subject"][subject_id],
            stats["activation_descriptor_mean"],
            stats["activation_descriptor_std"],
        ))
    target_descriptors: dict[str, torch.Tensor] = {}
    train_counts = {}
    target_hashes = {}
    for behavior in PATTERNS:
        rows = grouped[behavior]
        train_counts[behavior] = len(rows)
        if not rows:
            raise ValueError(f"missing train subjects for behavior {behavior}")
        descriptor = torch.stack(rows).mean(dim=0).to(dtype=torch.float32)
        if not torch.isfinite(descriptor).all():
            raise ValueError(f"nonfinite target descriptor for behavior {behavior}")
        target_descriptors[behavior] = descriptor
        target_hashes[behavior] = stable_hash_json(tensor_to_hashable(descriptor))
    stats["target_activation_descriptor_by_behavior"] = target_descriptors
    stats["target_descriptor_hash_by_behavior"] = target_hashes
    stats["train_counts_by_behavior"] = train_counts
    stats["train_statistics_hash"] = stable_hash_json({
        "activation_descriptor_count": stats["activation_descriptor_count"],
        "activation_descriptor_zero_std_count": stats[
            "activation_descriptor_zero_std_count"
        ],
        "descriptor_norm_hash": stats["descriptor_norm_hash"],
        "probe_examples_hash": stats["probe_examples_hash"],
        "scope": "v25_train_statistics",
        "target_descriptor_hash_by_behavior": target_hashes,
        "train_counts_by_behavior": train_counts,
    })
    return stats


def fit_v25_development_train_statistics(
    *,
    train_subjects: Sequence[Mapping[str, Any]],
    probe_examples: Sequence[Mapping[str, Any]],
    progress_log_path: Path,
    started_at_monotonic: float,
    now_monotonic: Any | None = None,
) -> dict[str, Any]:
    record_progress_event(
        progress_log_path,
        event="train_statistics_start",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "probe_examples_hash": stable_hash_json(list(probe_examples)),
            "train_subject_count": len(train_subjects),
        },
    )
    stats = fit_v25_train_statistics(
        train_subjects,
        probe_examples=probe_examples,
    )
    record_progress_event(
        progress_log_path,
        event="train_statistics_completed",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "activation_descriptor_count": int(stats["activation_descriptor_count"]),
            "activation_descriptor_zero_std_count": int(
                stats["activation_descriptor_zero_std_count"]
            ),
            "descriptor_norm_hash": str(stats["descriptor_norm_hash"]),
            "probe_examples_hash": str(stats["probe_examples_hash"]),
            "train_counts_by_behavior": dict(stats["train_counts_by_behavior"]),
            "train_statistics_hash": str(stats["train_statistics_hash"]),
        },
    )
    return stats


def run_v25_development_setup(
    *,
    pool_dir: Path,
    output_dir: Path,
    progress_log_path: Path,
    started_at_monotonic: float,
    now_monotonic: Any | None = None,
    max_development_jobs: int = 0,
    dry_run_placeholder_controls: bool = False,
    development_job_selection: str = "prefix",
    run_inner_validation: bool = False,
    inner_validation_rung_jobs: Sequence[int] | None = None,
    inner_validation_keep_fractions: Sequence[float] | None = None,
    inner_validation_max_configs: int | None = None,
    inner_validation_config_grid: str = "v25",
) -> dict[str, Any]:
    if bool(run_inner_validation) and bool(dry_run_placeholder_controls):
        raise ValueError("inner validation cannot use placeholder dry-run controls")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_v25_development_pool_inputs(
        pool_dir=pool_dir,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
    )
    probe_examples = build_probe_examples()
    train_pool_summary = redact_v25_loaded_pool_summary(prepared["train"])
    development_pool_summary = redact_v25_loaded_pool_summary(prepared["development"])
    train_pool_summary_hash = stable_hash_json(train_pool_summary)
    train_stats = fit_v25_development_train_statistics(
        train_subjects=prepared["train"]["subjects"],
        probe_examples=probe_examples,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
    )
    development_jobs = build_v25_development_jobs(prepared["development"]["subjects"])
    development_job_summary = redact_v25_development_job_summary(development_jobs)
    inner_rung_jobs = [
        int(value) for value in (inner_validation_rung_jobs or [4, 12])
    ]
    inner_keep_fractions = [
        float(value) for value in (inner_validation_keep_fractions or [0.25, 0.25])
    ]
    selection_job_count = int(max_development_jobs)
    if bool(run_inner_validation):
        selection_job_count = max(selection_job_count, max(inner_rung_jobs))
    experiment_variant = (
        experiment_variant_for_inner_validation_grid(str(inner_validation_config_grid))
        if bool(run_inner_validation)
        else V26_EXPERIMENT_VARIANT
    )
    ordered_development_jobs, development_job_selection_summary = (
        order_v25_development_jobs_for_bounded_selection(
            development_jobs,
            max_jobs=selection_job_count,
            strategy=str(development_job_selection),
        )
    )
    record_progress_event(
        progress_log_path,
        event="development_jobs_planned",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={"development_jobs": development_job_summary},
    )
    if selection_job_count > 0:
        record_progress_event(
            progress_log_path,
            event="development_jobs_selected",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra={"development_job_selection": development_job_selection_summary},
        )
    result = {
        "development_job_selection": development_job_selection_summary,
        "development_jobs": development_job_summary,
        "development_pool": development_pool_summary,
        "descriptor_norm_hash": str(train_stats["descriptor_norm_hash"]),
        "experiment_variant": experiment_variant,
        "passed": False,
        "probe_examples_hash": str(train_stats["probe_examples_hash"]),
        "stage": "development_setup_completed",
        "train_pool": train_pool_summary,
        "train_pool_summary_hash": train_pool_summary_hash,
        "train_statistics_hash": str(train_stats["train_statistics_hash"]),
    }
    if bool(run_inner_validation):
        configs = select_v25_inner_validation_configs(
            grid_name=str(inner_validation_config_grid),
            max_configs=inner_validation_max_configs,
            train_pool_file_sha256=str(train_pool_summary["pool_file_sha256"]),
            train_pool_summary_hash=train_pool_summary_hash,
        )
        inner_validation_result = run_v25_inner_validation_successive_halving_with_progress(
            configs=configs,
            jobs=ordered_development_jobs,
            train_subjects=prepared["train"]["subjects"],
            train_stats=train_stats,
            train_pool_file_sha256=str(train_pool_summary["pool_file_sha256"]),
            train_pool_summary_hash=train_pool_summary_hash,
            rung_job_counts=inner_rung_jobs,
            keep_fractions=inner_keep_fractions,
            norm_cap=0.25,
            job_plan_hash=development_job_summary["job_plan_hash"],
            script_sha256=sha256_file(Path(__file__)),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            experiment_variant=experiment_variant,
        )
        result["inner_validation"] = inner_validation_result
        result["stage"] = "inner_validation_completed"
    elif int(max_development_jobs) > 0:
        evaluation_config = {
            "compat_weight": 0.1,
            "projection": "rank1",
            "ridge_lambda": 1e-2,
        }
        if bool(dry_run_placeholder_controls):
            dry_run_jobs = ordered_development_jobs[: int(max_development_jobs)]
            context_by_record_hash = {}
            for job in dry_run_jobs:
                record_id_hash = stable_hash_json(str(job["record_id"]))
                source_descriptor = normalized_activation_descriptor_for_weights(
                    record_weights_tensor(job["subject"]),
                    train_stats=train_stats,
                )
                context_by_record_hash[record_id_hash] = (
                    build_v25_placeholder_control_context_for_dry_run(
                        source_descriptor=source_descriptor,
                        record_id_hash=record_id_hash,
                        job_plan_hash=development_job_summary["job_plan_hash"],
                    )
                )
            dry_run_config = {
                **evaluation_config,
                "allow_pinv_fallback": True,
            }
            dry_run_result = evaluate_v25_bounded_development_dry_run_with_progress(
                jobs=ordered_development_jobs,
                max_jobs=int(max_development_jobs),
                train_stats=train_stats,
                config=dry_run_config,
                norm_cap=0.25,
                spectral_basis=torch.eye(SOURCE_WEIGHT_DIM, 4, dtype=torch.float32),
                control_context_by_record_hash=context_by_record_hash,
                selected_config_hash=stable_hash_json({
                    "config": dry_run_config,
                    "scope": "v25_placeholder_bounded_dry_run_config",
                }),
                script_sha256=sha256_file(Path(__file__)),
                progress_log_path=progress_log_path,
                started_at_monotonic=started_at_monotonic,
                now_monotonic=now_monotonic,
            )
            result["dry_run"] = {
                key: value
                for key, value in dry_run_result.items()
                if key != "proof_records"
            }
        else:
            selected_config_hash = stable_hash_json({
                "config": evaluation_config,
                "scope": "v25_real_bounded_development_config",
            })
            train_delta_bank = build_v25_train_delta_bank_with_progress(
                train_subjects=prepared["train"]["subjects"],
                train_stats=train_stats,
                config=evaluation_config,
                norm_cap=0.25,
                script_sha256=sha256_file(Path(__file__)),
                progress_log_path=progress_log_path,
                started_at_monotonic=started_at_monotonic,
                now_monotonic=now_monotonic,
            )
            train_delta_matrix = torch.stack([
                torch.as_tensor(entry["delta"], dtype=torch.float32).reshape(-1)
                for entry in train_delta_bank["entries"]
            ], dim=0)
            spectral_basis, spectral_audit = compute_train_spectral_basis(
                train_delta_matrix,
                rank=min(4, int(train_delta_matrix.shape[0]), int(train_delta_matrix.shape[1])),
            )
            spectral_summary = {
                "basis_sha256": str(spectral_audit["basis_sha256"]),
                "centered_edit_matrix_sha256": str(spectral_audit["centered_delta_sha256"]),
                "edit_vector_count": int(spectral_audit["delta_count"]),
                "explained_singular_values": list(
                    spectral_audit["explained_singular_values"]
                ),
                "rank": int(spectral_audit["rank"]),
            }
            record_progress_event(
                progress_log_path,
                event="train_edit_spectral_basis_completed",
                started_at_monotonic=started_at_monotonic,
                now_monotonic=now_monotonic,
                extra=spectral_summary,
            )
            context_by_record_hash = build_v25_train_only_control_contexts_with_progress(
                jobs=ordered_development_jobs,
                max_jobs=int(max_development_jobs),
                train_delta_bank=train_delta_bank,
                train_stats=train_stats,
                job_plan_hash=development_job_summary["job_plan_hash"],
                selected_config_hash=selected_config_hash,
                progress_log_path=progress_log_path,
                started_at_monotonic=started_at_monotonic,
                now_monotonic=now_monotonic,
            )
            record_progress_event(
                progress_log_path,
                event="development_bounded_real_start",
                started_at_monotonic=started_at_monotonic,
                now_monotonic=now_monotonic,
                extra={
                    "max_jobs": int(max_development_jobs),
                    "selected_config_hash": selected_config_hash,
                    "total_planned_jobs": len(ordered_development_jobs),
                    "train_edit_bank_hash": str(train_delta_bank["bank_hash"]),
                },
            )
            development_result = evaluate_v25_development_jobs_with_progress(
                jobs=ordered_development_jobs,
                max_jobs=int(max_development_jobs),
                train_stats=train_stats,
                config=evaluation_config,
                norm_cap=0.25,
                spectral_basis=spectral_basis,
                control_context_by_record_hash=context_by_record_hash,
                selected_config_hash=selected_config_hash,
                script_sha256=sha256_file(Path(__file__)),
                progress_log_path=progress_log_path,
                started_at_monotonic=started_at_monotonic,
                now_monotonic=now_monotonic,
            )
            result["train_edit_bank"] = redact_v25_train_delta_bank_summary(
                train_delta_bank
            )
            result["development_evaluation"] = {
                key: value
                for key, value in development_result.items()
                if key != "proof_records"
            }
            result["train_edit_spectral_basis"] = {
                "basis_sha256": str(spectral_audit["basis_sha256"]),
                "edit_vector_count": int(spectral_audit["delta_count"]),
                "rank": int(spectral_audit["rank"]),
            }
            record_progress_event(
                progress_log_path,
                event="development_bounded_real_completed",
                started_at_monotonic=started_at_monotonic,
                now_monotonic=now_monotonic,
                extra={
                    **result["development_evaluation"],
                    "stage": "development_bounded_real_completed",
                    "train_edit_bank_hash": str(train_delta_bank["bank_hash"]),
                },
            )
    record_progress_event(
        progress_log_path,
        event="development_setup_completed",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "development_pool": result["development_pool"],
            "development_jobs": result["development_jobs"],
            "experiment_variant": result["experiment_variant"],
            "probe_examples_hash": result["probe_examples_hash"],
            "train_pool": result["train_pool"],
            "train_statistics_hash": result["train_statistics_hash"],
        },
    )
    result["development_progress_log_path_sha256"] = stable_hash_json(
        str(progress_log_path.resolve())
    )
    result["development_progress_log_sha256"] = sha256_file(progress_log_path)
    return result


def build_v25_development_jobs(
    development_subjects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for subject in sorted(development_subjects, key=lambda item: str(item["subject_id"])):
        source_behavior = behavior_for_record(subject)
        for target_behavior in sorted(PATTERNS):
            if str(target_behavior) == str(source_behavior):
                continue
            record_id = (
                f"{subject['subject_id']}::{source_behavior}->{target_behavior}"
            )
            jobs.append({
                "direction": f"{source_behavior}->{target_behavior}",
                "record_id": record_id,
                "source_behavior": str(source_behavior),
                "subject": subject,
                "target_behavior": str(target_behavior),
            })
    return jobs


def redact_v25_development_job_summary(
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    direction_counts: dict[str, int] = {}
    for job in jobs:
        direction = str(job["direction"])
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    ordered_job_identity_hashes = [
        stable_hash_json({
            "direction": str(job["direction"]),
            "record_id": str(job["record_id"]),
            "source_behavior": str(job["source_behavior"]),
            "target_behavior": str(job["target_behavior"]),
        })
        for job in jobs
    ]
    return {
        "direction_counts": {
            direction: direction_counts[direction]
            for direction in sorted(direction_counts)
        },
        "job_count": len(jobs),
        "job_plan_hash": stable_hash_json({
            "direction_counts": direction_counts,
            "job_count": len(jobs),
            "ordered_job_identity_hashes": ordered_job_identity_hashes,
            "scope": "v25_development_job_plan",
        }),
    }


def order_v25_development_jobs_for_bounded_selection(
    jobs: Sequence[Mapping[str, Any]],
    *,
    max_jobs: int,
    strategy: str,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    strategy = str(strategy)
    if strategy not in {"prefix", "balanced-directions"}:
        raise ValueError(f"unsupported development job selection strategy: {strategy}")
    jobs_list = list(jobs)
    selected_count = min(max(0, int(max_jobs)), len(jobs_list))
    if selected_count == 0 or strategy == "prefix":
        ordered_jobs = jobs_list
    else:
        jobs_by_direction: dict[str, list[Mapping[str, Any]]] = {}
        for job in jobs_list:
            jobs_by_direction.setdefault(str(job["direction"]), []).append(job)
        for direction in jobs_by_direction:
            jobs_by_direction[direction].sort(key=lambda job: str(job["record_id"]))
        selected: list[Mapping[str, Any]] = []
        while len(selected) < selected_count:
            added_this_round = False
            for direction in sorted(jobs_by_direction):
                if jobs_by_direction[direction] and len(selected) < selected_count:
                    selected.append(jobs_by_direction[direction].pop(0))
                    added_this_round = True
            if not added_this_round:
                break
        selected_keys = {str(job["record_id"]) for job in selected}
        ordered_jobs = selected + [
            job for job in jobs_list
            if str(job["record_id"]) not in selected_keys
        ]
    selected_jobs = ordered_jobs[:selected_count]
    selected_direction_counts: dict[str, int] = {}
    for job in selected_jobs:
        direction = str(job["direction"])
        selected_direction_counts[direction] = selected_direction_counts.get(direction, 0) + 1
    selected_job_identity_hashes = [
        stable_hash_json({
            "direction": str(job["direction"]),
            "record_id": str(job["record_id"]),
            "source_behavior": str(job["source_behavior"]),
            "target_behavior": str(job["target_behavior"]),
        })
        for job in selected_jobs
    ]
    selection_hash = stable_hash_json({
        "scope": "v25_development_job_selection",
        "selected_direction_counts": selected_direction_counts,
        "selected_job_identity_hashes": selected_job_identity_hashes,
        "strategy": strategy,
        "total_planned_jobs": len(jobs_list),
    })
    return ordered_jobs, {
        "selected_direction_counts": {
            direction: selected_direction_counts[direction]
            for direction in sorted(selected_direction_counts)
        },
        "selected_job_count": len(selected_jobs),
        "selected_jobs_hash": stable_hash_json(selected_job_identity_hashes),
        "selection_hash": selection_hash,
        "strategy": strategy,
        "total_planned_jobs": len(jobs_list),
    }


def normalized_activation_descriptor_for_weights(
    weights: torch.Tensor,
    *,
    train_stats: Mapping[str, Any],
) -> torch.Tensor:
    descriptor = activation_descriptor_for_weights(
        weights.reshape(-1).to(dtype=torch.float32),
        probe_examples=train_stats["probe_examples"],
    )
    normalized = tensor_normalize(
        descriptor,
        train_stats["activation_descriptor_mean"],
        train_stats["activation_descriptor_std"],
    )
    if int(normalized.numel()) != ACTIVATION_DESCRIPTOR_DIM:
        raise ValueError("normalized activation descriptor has wrong dimension")
    if not torch.isfinite(normalized).all():
        raise ValueError("nonfinite normalized activation descriptor")
    return normalized.to(dtype=torch.float32)


def activation_jacobian_for_weights(
    weights: torch.Tensor,
    *,
    train_stats: Mapping[str, Any],
) -> torch.Tensor:
    flat = weights.detach().clone().to(dtype=torch.float32).reshape(-1).requires_grad_(True)
    descriptor = normalized_activation_descriptor_for_weights(flat, train_stats=train_stats)
    rows = []
    for item in descriptor:
        grad = torch.autograd.grad(item, flat, retain_graph=True, allow_unused=False)[0]
        if grad is None:
            raise RuntimeError("autograd returned None for V25 activation descriptor row")
        rows.append(grad.detach().clone().to(dtype=torch.float32))
    jacobian = torch.stack(rows).to(dtype=torch.float32)
    if jacobian.shape != (ACTIVATION_DESCRIPTOR_DIM, SOURCE_WEIGHT_DIM):
        raise ValueError(
            "activation jacobian shape mismatch: "
            f"{tuple(jacobian.shape)}"
        )
    if not torch.isfinite(jacobian).all():
        raise ValueError("nonfinite activation jacobian")
    return jacobian


def source_logits_and_jacobian_for_weights(
    weights: torch.Tensor,
    *,
    probe_examples: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = v23.v16.probe_inputs_tensor(probe_examples)
    logits, jacobian = v23.logits_and_jacobian_for_inputs(weights=weights, inputs=inputs)
    if not torch.isfinite(logits).all() or not torch.isfinite(jacobian).all():
        raise ValueError("nonfinite source logits/jacobian")
    return logits.to(dtype=torch.float32), jacobian.to(dtype=torch.float32)


def build_augmented_jacobian_system(
    *,
    activation_jacobian: torch.Tensor,
    activation_delta: torch.Tensor,
    source_logit_jacobian: torch.Tensor,
    compat_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    activation_jacobian = activation_jacobian.to(dtype=torch.float32)
    activation_delta = activation_delta.to(dtype=torch.float32).reshape(-1)
    source_logit_jacobian = source_logit_jacobian.to(dtype=torch.float32)
    if activation_jacobian.ndim != 2 or source_logit_jacobian.ndim != 2:
        raise ValueError("jacobian inputs must be rank-2")
    if int(activation_jacobian.shape[0]) != int(activation_delta.numel()):
        raise ValueError("activation jacobian rows must match activation delta")
    if int(activation_jacobian.shape[1]) != int(source_logit_jacobian.shape[1]):
        raise ValueError("activation and source-logit jacobian widths must match")
    if not torch.isfinite(activation_jacobian).all():
        raise ValueError("nonfinite activation jacobian")
    if not torch.isfinite(activation_delta).all():
        raise ValueError("nonfinite activation delta")
    if not torch.isfinite(source_logit_jacobian).all():
        raise ValueError("nonfinite source-logit jacobian")
    if float(compat_weight) <= 0.0:
        return activation_jacobian, activation_delta
    scale = math.sqrt(float(compat_weight))
    zeros = torch.zeros(
        int(source_logit_jacobian.shape[0]),
        dtype=torch.float32,
        device=activation_delta.device,
    )
    return (
        torch.cat([activation_jacobian, scale * source_logit_jacobian], dim=0),
        torch.cat([activation_delta, zeros], dim=0),
    )


def build_jacobian_cache_key(
    *,
    subject_id: str,
    source_behavior: str,
    train_stats: Mapping[str, Any],
    script_sha256: str,
) -> str:
    return stable_hash_json({
        "descriptor_norm_hash": str(train_stats["descriptor_norm_hash"]),
        "probe_examples_hash": str(train_stats["probe_examples_hash"]),
        "scope": "v25_jacobian_cache",
        "script_sha256": str(script_sha256),
        "source": str(source_behavior),
        "subject_id": str(subject_id),
    })


def compute_jacobian_cache_entry(
    record: Mapping[str, Any],
    *,
    source_behavior: str,
    train_stats: Mapping[str, Any],
    script_sha256: str,
) -> dict[str, Any]:
    started_at = time.monotonic()
    weights = record_weights_tensor(record)
    source_descriptor = normalized_activation_descriptor_for_weights(
        weights,
        train_stats=train_stats,
    )
    activation_jacobian = activation_jacobian_for_weights(weights, train_stats=train_stats)
    source_logits, source_logit_jacobian = source_logits_and_jacobian_for_weights(
        weights,
        probe_examples=train_stats["probe_examples"],
    )
    cache_key = build_jacobian_cache_key(
        subject_id=str(record["subject_id"]),
        source_behavior=str(source_behavior),
        train_stats=train_stats,
        script_sha256=str(script_sha256),
    )
    finite = all(
        bool(torch.isfinite(item).all().item())
        for item in [
            source_descriptor,
            activation_jacobian,
            source_logits,
            source_logit_jacobian,
        ]
    )
    if not finite:
        raise ValueError("nonfinite jacobian cache entry")
    audit = {
        "activation_matrix_sha256": stable_hash_json(tensor_to_hashable(activation_jacobian)),
        "activation_row_count": int(activation_jacobian.shape[0]),
        "cache_key": cache_key,
        "elapsed_seconds": float(time.monotonic() - started_at),
        "finite": True,
        "source_output_matrix_sha256": stable_hash_json(tensor_to_hashable(source_logit_jacobian)),
        "source_output_row_count": int(source_logit_jacobian.shape[0]),
        "source_output_vector_sha256": stable_hash_json(tensor_to_hashable(source_logits)),
        "source_vector_sha256": stable_hash_json(tensor_to_hashable(source_descriptor)),
        "weight_dim": int(weights.numel()),
    }
    return {
        "activation_jacobian": activation_jacobian,
        "audit": audit,
        "cache_key": cache_key,
        "source_descriptor": source_descriptor,
        "source_logit_jacobian": source_logit_jacobian,
        "source_logits": source_logits,
    }


def build_v25_seed_preflight() -> dict[str, Any]:
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


def build_probe_examples() -> list[dict[str, Any]]:
    return v23.build_probe_examples()


def read_progress_log_summary(progress_log_path: Path) -> dict[str, Any]:
    if not progress_log_path.exists():
        return {"latest": {}, "line_count": 0}
    latest: dict[str, Any] = {}
    line_count = 0
    with progress_log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            line_count += 1
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                parsed = {"event": "malformed_progress_line"}
            if isinstance(parsed, dict):
                latest = parsed
    return {"latest": latest, "line_count": line_count}


def monitor_text_has_forbidden_detail_term(text: str) -> bool:
    lowered = text.lower()
    return any(
        forbidden.lower() in lowered
        for forbidden in RECURSIVE_FORBIDDEN_FINAL_DETAIL_KEY_TERMS
    )


def build_long_run_monitor_snapshot(
    *,
    started_at_monotonic: float,
    progress_log_path: Path | None,
) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    progress_summary = (
        read_progress_log_summary(progress_log_path)
        if progress_log_path is not None
        else {"latest": {}, "line_count": 0}
    )
    latest = progress_summary["latest"]
    snapshot = {
        "cpu_system_seconds": float(usage.ru_stime),
        "cpu_user_seconds": float(usage.ru_utime),
        "elapsed_seconds": float(time.monotonic() - started_at_monotonic),
        "event": "monitor_heartbeat",
        "max_rss_platform_units": int(usage.ru_maxrss),
        "pid": int(os.getpid()),
        "progress_line_count": int(progress_summary["line_count"]),
    }
    if progress_log_path is not None:
        snapshot["progress_log_location_sha256"] = stable_hash_json(
            str(progress_log_path.resolve())
        )
    latest_event = latest.get("event")
    if latest_event is not None:
        latest_event_text = str(latest_event)
        if monitor_text_has_forbidden_detail_term(latest_event_text):
            snapshot["latest_progress_event_redacted"] = True
            snapshot["latest_progress_event_sha256"] = stable_hash_json(latest_event_text)
        else:
            snapshot["latest_progress_event"] = latest_event_text
    safe_progress_keys = {
        "candidate_count": "latest_progress_candidate_count",
        "completed_count": "latest_progress_completed_count",
        "failure_count": "latest_progress_failure_count",
        "pool": "latest_progress_pool",
        "range_count": "latest_progress_range_count",
        "record_count": "latest_progress_record_count",
        "seed_range_count": "latest_progress_range_count",
        "worker_count": "latest_progress_worker_count",
    }
    for key, safe_key in safe_progress_keys.items():
        if key in latest:
            snapshot[safe_key] = latest[key]
    if "elapsed_seconds" in latest:
        snapshot["latest_progress_elapsed_seconds"] = latest["elapsed_seconds"]
    return snapshot


def append_monitor_event(
    monitor_log_path: Path,
    *,
    event: str,
    started_at_monotonic: float,
    progress_log_path: Path | None,
) -> None:
    monitor_log_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = build_long_run_monitor_snapshot(
        started_at_monotonic=started_at_monotonic,
        progress_log_path=progress_log_path,
    )
    snapshot["event"] = event
    with monitor_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")


class LongRunMonitor:
    def __init__(
        self,
        *,
        monitor_log_path: Path,
        progress_log_path: Path | None,
        interval_seconds: float,
    ) -> None:
        self.monitor_log_path = monitor_log_path
        self.progress_log_path = progress_log_path
        self.interval_seconds = float(interval_seconds)
        self.started_at_monotonic = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        append_monitor_event(
            self.monitor_log_path,
            event="monitor_start",
            started_at_monotonic=self.started_at_monotonic,
            progress_log_path=self.progress_log_path,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds))
        append_monitor_event(
            self.monitor_log_path,
            event="monitor_stop",
            started_at_monotonic=self.started_at_monotonic,
            progress_log_path=self.progress_log_path,
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            append_monitor_event(
                self.monitor_log_path,
                event="monitor_heartbeat",
                started_at_monotonic=self.started_at_monotonic,
                progress_log_path=self.progress_log_path,
            )


def generate_pools(args: SimpleNamespace, pool_dir: Path) -> dict[str, Any]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    progress_log_path = pool_dir / SOURCE_POOL_PROGRESS_LOG_FILENAME
    seed_preflight = build_v25_seed_preflight()
    record_progress_event(
        progress_log_path,
        event="preflight_completed",
        started_at_monotonic=started_at,
        extra={
            "failure_count": len(seed_preflight["failures"]),
            "range_count": len(seed_preflight["seed_ranges"]),
        },
    )
    if seed_preflight["failures"]:
        result = {"failures": seed_preflight["failures"], "passed": False}
        write_json_atomic(pool_dir / "combined_audit.json", result)
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
        write_json_atomic(pool_path, payload)
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
    write_json_atomic(pool_dir / "final_redacted_audit.json", final_redacted)
    record_progress_event(
        progress_log_path,
        event="final_redacted_audit_written",
        started_at_monotonic=started_at,
        extra={
            "final_redacted_audit_sha256": sha256_file(
                pool_dir / "final_redacted_audit.json"
            )
        },
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
    write_json_atomic(pool_dir / "combined_audit.json", audit)
    record_progress_event(
        progress_log_path,
        event="combined_audit_written",
        started_at_monotonic=started_at,
        extra={"combined_audit_sha256": sha256_file(pool_dir / "combined_audit.json")},
    )
    return {
        "combined_audit_path": str(pool_dir / "combined_audit.json"),
        "final_redacted_audit_path": str(pool_dir / "final_redacted_audit.json"),
        "passed": bool(audit.get("passed", False)),
        "pool_dir": str(pool_dir),
        "pool_summaries": audit.get("pool_summaries", {}),
        "seed_preflight": seed_preflight,
        "source_pool_progress_log_path": str(progress_log_path),
        "source_pool_progress_log_sha256": sha256_file(progress_log_path),
    }


def inner_validation_ranking_tuple(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        bool(candidate["invalid"]),
        -float(candidate["target_prediction_rate"]),
        -float(candidate["pareto_undominated_rate"]),
        -float(candidate["mean_target_margin"]),
        -float(candidate["mean_matched_minus_best_control_target_margin"]),
        -float(candidate["mean_matched_minus_shuffled_signature_target_margin"]),
        int(candidate["proof_gate_failure_count"]),
        str(candidate["config_hash"]),
    )


def solve_jacobian_ridge_edit(
    jacobian: torch.Tensor,
    target_delta: torch.Tensor,
    *,
    ridge_lambda: float,
    allow_pinv_fallback: bool = False,
    audit: dict[str, Any] | None = None,
) -> torch.Tensor:
    jacobian = jacobian.to(dtype=torch.float32)
    target_delta = target_delta.to(dtype=torch.float32)
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be rank-2")
    if target_delta.ndim != 1:
        raise ValueError("target_delta must be rank-1")
    if int(jacobian.shape[0]) != int(target_delta.shape[0]):
        raise ValueError("jacobian row count must match target_delta")
    if not torch.isfinite(jacobian).all() or not torch.isfinite(target_delta).all():
        raise ValueError("nonfinite jacobian ridge input")
    solve_jacobian = jacobian.to(dtype=torch.float64)
    solve_target_delta = target_delta.to(dtype=torch.float64)
    eye = torch.eye(
        solve_jacobian.shape[0],
        dtype=torch.float64,
        device=solve_jacobian.device,
    )
    gram = solve_jacobian @ solve_jacobian.T + float(ridge_lambda) * eye
    fallback_used = False
    try:
        solution = torch.linalg.solve(gram, solve_target_delta)
    except RuntimeError:
        if not bool(allow_pinv_fallback):
            raise
        solution = torch.linalg.pinv(gram) @ solve_target_delta
        fallback_used = True
    if audit is not None:
        audit["pinv_fallback_allowed"] = bool(allow_pinv_fallback)
        audit["pinv_fallback_used"] = bool(fallback_used)
    delta = solve_jacobian.T @ solution
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite jacobian edit")
    return delta.to(dtype=torch.float32)


def rank1_svd_factors(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = matrix.to(dtype=torch.float32)
    if matrix.ndim != 2:
        raise ValueError("matrix must be rank-2")
    if not torch.isfinite(matrix).all():
        raise ValueError("nonfinite rank1 projection input")
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    left = u[:, 0].clone()
    right = vh[0].clone()
    pivot = int(torch.argmax(torch.abs(left)).item())
    if float(left[pivot].item()) < 0.0:
        left = -left
        right = -right
    return left.to(dtype=torch.float32), s.to(dtype=torch.float32), right.to(dtype=torch.float32)


def project_matrix_rank1(matrix: torch.Tensor) -> torch.Tensor:
    left, singular_values, right = rank1_svd_factors(matrix)
    return singular_values[0] * torch.outer(left, right)


def project_to_basis(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    flat = delta.to(dtype=torch.float32).reshape(-1)
    basis = basis.to(dtype=torch.float32)
    if basis.ndim != 2 or int(basis.shape[0]) != int(flat.numel()):
        raise ValueError("basis shape must be [delta_dim, rank]")
    if not torch.isfinite(flat).all() or not torch.isfinite(basis).all():
        raise ValueError("nonfinite basis projection input")
    coeff = basis.T @ flat
    projected = basis @ coeff
    if not torch.isfinite(projected).all():
        raise ValueError("nonfinite basis projection output")
    return projected.to(dtype=torch.float32)


def apply_norm_cap(delta: torch.Tensor, *, max_norm: float) -> torch.Tensor:
    flat = delta.to(dtype=torch.float32).reshape(-1)
    if not torch.isfinite(flat).all():
        raise ValueError("nonfinite norm-cap input")
    if float(max_norm) < 0.0:
        raise ValueError("max_norm must be nonnegative")
    norm = torch.linalg.norm(flat)
    if float(norm.item()) <= float(max_norm) or float(norm.item()) <= 1e-12:
        return flat
    return (flat * (float(max_norm) / float(norm.item()))).to(dtype=torch.float32)


def project_hidden_rank1_delta(delta: torch.Tensor) -> torch.Tensor:
    flat = delta.to(dtype=torch.float32).reshape(-1)
    if int(flat.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("delta must match source weight dimension")
    if not torch.isfinite(flat).all():
        raise ValueError("nonfinite rank1 delta input")
    projected = torch.zeros_like(flat)
    for layer_index in v23.HIDDEN_LAYERS:
        weight_spec, bias_spec = v23.hidden_layer_specs(int(layer_index))
        matrix = v17.component_from_flat(flat, weight_spec)
        bias = v17.component_from_flat(flat, bias_spec)
        v17.set_component(projected, weight_spec, project_matrix_rank1(matrix))
        v17.set_component(projected, bias_spec, bias)
    return projected.to(dtype=torch.float32)


def project_delta_for_config(
    delta: torch.Tensor,
    *,
    projection: str,
    spectral_basis: torch.Tensor | None = None,
) -> torch.Tensor:
    if projection == "none":
        return delta.to(dtype=torch.float32).reshape(-1)
    if projection == "rank1":
        return project_hidden_rank1_delta(delta)
    if projection == "spectral_rank4":
        if spectral_basis is None:
            raise ValueError("spectral basis is required for spectral projection")
        return project_to_basis(delta, spectral_basis)
    if projection == "rank1_spectral_rank4":
        if spectral_basis is None:
            raise ValueError("spectral basis is required for rank1+spectral projection")
        return project_to_basis(project_hidden_rank1_delta(delta), spectral_basis)
    raise ValueError(f"unknown projection: {projection}")


def solve_projected_jacobian_edit(
    *,
    source_descriptor: torch.Tensor,
    target_descriptor: torch.Tensor,
    activation_jacobian: torch.Tensor,
    source_logit_jacobian: torch.Tensor,
    config: Mapping[str, Any],
    norm_cap: float,
    spectral_basis: torch.Tensor | None = None,
) -> dict[str, Any]:
    source_descriptor = source_descriptor.to(dtype=torch.float32).reshape(-1)
    target_descriptor = target_descriptor.to(dtype=torch.float32).reshape(-1)
    if int(source_descriptor.numel()) != int(target_descriptor.numel()):
        raise ValueError("source and target descriptors must have the same dimension")
    if not torch.isfinite(source_descriptor).all() or not torch.isfinite(target_descriptor).all():
        raise ValueError("nonfinite edit descriptor input")
    activation_delta = target_descriptor - source_descriptor
    augmented_jacobian, augmented_delta = build_augmented_jacobian_system(
        activation_jacobian=activation_jacobian,
        activation_delta=activation_delta,
        source_logit_jacobian=source_logit_jacobian,
        compat_weight=float(config["compat_weight"]),
    )
    solve_audit: dict[str, Any] = {}
    raw_delta = solve_jacobian_ridge_edit(
        augmented_jacobian,
        augmented_delta,
        ridge_lambda=float(config["ridge_lambda"]),
        allow_pinv_fallback=bool(config.get("allow_pinv_fallback", False)),
        audit=solve_audit,
    )
    projected_delta = project_delta_for_config(
        raw_delta,
        projection=str(config["projection"]),
        spectral_basis=spectral_basis,
    )
    capped_delta = apply_norm_cap(projected_delta, max_norm=float(norm_cap))
    raw_norm = float(torch.linalg.norm(raw_delta).item())
    projected_norm = float(torch.linalg.norm(projected_delta).item())
    capped_norm = float(torch.linalg.norm(capped_delta).item())
    audit = {
        "activation_row_count": int(activation_jacobian.shape[0]),
        "augmented_row_count": int(augmented_jacobian.shape[0]),
        "compat_weight": float(config["compat_weight"]),
        "delta_sha256": stable_hash_json(tensor_to_hashable(capped_delta)),
        "norm_cap": float(norm_cap),
        "norm_cap_applied": bool(projected_norm > float(norm_cap) + 1e-8),
        "projected_delta_norm": projected_norm,
        "projection": str(config["projection"]),
        "raw_delta_norm": raw_norm,
        "ridge_lambda": float(config["ridge_lambda"]),
        "source_logit_row_count": int(source_logit_jacobian.shape[0]),
        "target_delta_norm": float(torch.linalg.norm(activation_delta).item()),
        **solve_audit,
    }
    if not all(math.isfinite(float(value)) for value in [
        audit["raw_delta_norm"],
        audit["projected_delta_norm"],
        capped_norm,
        audit["target_delta_norm"],
    ]):
        raise ValueError("nonfinite edit audit norm")
    audit["capped_delta_norm"] = capped_norm
    return {
        "audit": audit,
        "delta": capped_delta,
        "projected_delta": projected_delta,
        "raw_delta": raw_delta,
    }


def evaluate_v25_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    train_stats: Mapping[str, Any],
    cache_entry: Mapping[str, Any],
    config: Mapping[str, Any],
    norm_cap: float,
    spectral_basis: torch.Tensor | None = None,
) -> dict[str, Any]:
    source_weights = record_weights_tensor(subject)
    target_descriptor = train_stats["target_activation_descriptor_by_behavior"][
        str(target_behavior)
    ]
    edit = solve_projected_jacobian_edit(
        source_descriptor=cache_entry["source_descriptor"],
        target_descriptor=target_descriptor,
        activation_jacobian=cache_entry["activation_jacobian"],
        source_logit_jacobian=cache_entry["source_logit_jacobian"],
        config=config,
        norm_cap=float(norm_cap),
        spectral_basis=spectral_basis,
    )
    edited = source_weights + edit["delta"]
    metrics = v23.v16.v15.v14.functional_metrics(
        edited.to(dtype=torch.float32),
        str(source_behavior),
        str(target_behavior),
        source_weights.to(dtype=torch.float32),
    )
    if not all(
        math.isfinite(float(metrics[key]))
        for key in [
            "compatible_source_output_mse",
            "source_margin",
            "target_margin",
        ]
    ):
        raise ValueError("nonfinite functional metric")
    delta_norm = float(torch.linalg.norm(edit["delta"]).item())
    matched_spectral_projection_norm = None
    if spectral_basis is not None:
        matched_spectral_projection_norm = float(
            torch.linalg.norm(project_to_basis(edit["delta"], spectral_basis)).item()
        )
    editor_audit = {
        **edit["audit"],
        "cache_key": str(cache_entry["cache_key"]),
        "edited_vector_sha256": stable_hash_json(tensor_to_hashable(edited)),
    }
    if matched_spectral_projection_norm is not None:
        if not math.isfinite(matched_spectral_projection_norm):
            raise ValueError("nonfinite matched spectral projection norm")
        editor_audit["matched_spectral_projection_norm"] = matched_spectral_projection_norm
    return {
        **metrics,
        "control_type": EDITOR_METHOD,
        "delta_norm": delta_norm,
        "editor": editor_audit,
        "target_prediction_pass": metrics["predicted_behavior"] == str(target_behavior),
    }


def solve_v27_localized_behavior_loss_edit(
    *,
    basis: torch.Tensor,
    basis_audit: Mapping[str, Any],
    config: Mapping[str, Any],
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
    selected_config_hash: str | None = None,
) -> dict[str, Any]:
    basis = validate_v27_localized_basis_matrix(basis)
    basis_hash = require_sha256_hex(basis_audit["basis_hash"], field_name="basis_hash")
    source = source_weights.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(source.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("source weights have wrong dimension")
    if not torch.isfinite(source).all():
        raise ValueError("nonfinite source weights")
    support = v27_support_tensors_for_source_target(
        source_weights=source,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    steps = int(config.get("localized_steps", 0))
    if steps <= 0:
        raise ValueError("localized_steps must be positive")
    lr = float(config.get("localized_lr", 0.0))
    if lr <= 0.0 or not math.isfinite(lr):
        raise ValueError("localized_lr must be positive and finite")
    norm_cap = float(config.get("localized_norm_cap", V27_LOCALIZED_NORM_CAP))
    if norm_cap < 0.0 or not math.isfinite(norm_cap):
        raise ValueError("localized_norm_cap must be finite and nonnegative")
    source_mse_weight = float(config.get("localized_source_mse_weight", 1.0))
    delta_l2_weight = float(config.get("localized_delta_l2_weight", 0.0))
    norm_barrier_weight = float(config.get("localized_norm_barrier_weight", 10.0))
    alpha = torch.zeros(int(basis.shape[1]), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [alpha],
        lr=lr,
        betas=V27_LOCALIZED_OPTIMIZER_BETAS,
        eps=V27_LOCALIZED_OPTIMIZER_EPS,
        weight_decay=0.0,
        amsgrad=False,
    )
    trace: list[dict[str, Any]] = []
    log_every = max(1, steps // 5)
    best: tuple[float, float, int, torch.Tensor, dict[str, Any]] | None = None
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = basis @ alpha
        edited = source + delta
        target_logits = v27_subject_logits_for_inputs(edited, support["target_inputs"])
        conflict_logits = v27_subject_logits_for_inputs(edited, support["conflict_inputs"])
        compatible_logits = v27_subject_logits_for_inputs(edited, support["compatible_inputs"])
        target_bce = F.binary_cross_entropy_with_logits(
            target_logits,
            support["target_labels"].to(dtype=torch.float32),
        )
        conflict_bce = F.binary_cross_entropy_with_logits(
            conflict_logits,
            support["conflict_target_labels"].to(dtype=torch.float32),
        )
        compatible_mse = F.mse_loss(
            compatible_logits,
            support["compatible_source_logits"].to(dtype=torch.float32),
        )
        delta_l2 = torch.mean(delta.pow(2))
        delta_norm = torch.linalg.norm(delta)
        norm_barrier = torch.clamp(delta_norm - norm_cap, min=0.0).pow(2)
        loss = (
            target_bce
            + conflict_bce
            + source_mse_weight * compatible_mse
            + delta_l2_weight * delta_l2
            + norm_barrier_weight * norm_barrier
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite localized support loss")
        loss.backward()
        if alpha.grad is None or not torch.isfinite(alpha.grad).all():
            raise ValueError("nonfinite localized alpha gradient")
        torch.nn.utils.clip_grad_norm_([alpha], V27_LOCALIZED_GRAD_CLIP_NORM)
        optimizer.step()
        if not torch.isfinite(alpha).all():
            raise ValueError("nonfinite localized alpha")

        with torch.no_grad():
            current_delta = (basis @ alpha).detach().clone().to(dtype=torch.float32)
            current_delta_norm = float(torch.linalg.norm(current_delta).item())
            scalar_losses = {
                "compatible_mse": float(compatible_mse.detach().item()),
                "conflict_bce": float(conflict_bce.detach().item()),
                "delta_l2": float(delta_l2.detach().item()),
                "loss": float(loss.detach().item()),
                "target_bce": float(target_bce.detach().item()),
            }
            if step % log_every == 0 or step == steps:
                progress_event = redact_v27_optimizer_progress_event({
                    "basis_hash": basis_hash,
                    "delta_norm": current_delta_norm,
                    "loss": scalar_losses["loss"],
                    "step": step,
                })
                trace.append(progress_event)
                if progress_log_path is not None:
                    event_extra = dict(progress_event)
                    if record_id_hash is not None:
                        event_extra["record_id_hash"] = require_sha256_hex(
                            record_id_hash,
                            field_name="record_id_hash",
                        )
                    if selected_config_hash is not None:
                        event_extra["selected_config_hash"] = require_sha256_hex(
                            selected_config_hash,
                            field_name="selected_config_hash",
                        )
                    record_progress_event(
                        progress_log_path,
                        event="v27_localized_optimizer_progress",
                        started_at_monotonic=started_at_monotonic,
                        now_monotonic=now_monotonic,
                        extra=event_extra,
                    )
            objective_key = (
                scalar_losses["loss"],
                current_delta_norm,
                step,
            )
            if best is None or objective_key < best[:3]:
                best = (
                    scalar_losses["loss"],
                    current_delta_norm,
                    step,
                    current_delta,
                    scalar_losses,
                )
    if best is None:
        raise ValueError("localized optimizer produced no candidate")
    best_loss, best_delta_norm, best_step, best_delta, best_losses = best
    clipped_delta = apply_norm_cap(best_delta, max_norm=norm_cap)
    hard_norm_clipped = bool(float(torch.linalg.norm(best_delta).item()) > norm_cap + 1e-8)
    audit = {
        **dict(basis_audit),
        "basis_hash": basis_hash,
        "best_step": int(best_step),
        "delta_norm_before_hard_cap": float(best_delta_norm),
        "delta_sha256": stable_hash_json(tensor_to_hashable(clipped_delta)),
        "hard_norm_clipped": hard_norm_clipped,
        "localized_delta_l2_weight": float(delta_l2_weight),
        "localized_lr": float(lr),
        "localized_norm_cap": float(norm_cap),
        "localized_source_mse_weight": float(source_mse_weight),
        "localized_steps": int(steps),
        "optimization_boundary": v27_localized_optimization_boundary(),
        "optimization_split": "support",
        "optimizer": "AdamW",
        "optimizer_trace_hash": stable_hash_json(trace),
        "proof_split": "heldout",
        "support_objective": float(best_loss),
        "support_objective_is_proof_metric": False,
        "support_scalar_losses": best_losses,
    }
    finite_failures = recursive_numeric_finiteness_failures(audit)
    if finite_failures:
        raise ValueError("nonfinite localized optimizer audit: " + ", ".join(finite_failures[:5]))
    return {
        "audit": audit,
        "delta": clipped_delta,
    }


def evaluate_v25_localized_behavior_loss_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    config: Mapping[str, Any],
    norm_cap: float,
    selected_config_hash: str,
    spectral_basis: torch.Tensor | None,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
) -> dict[str, Any]:
    source_weights = record_weights_tensor(subject)
    localized_config = {
        **dict(config),
        "localized_norm_cap": float(config.get("localized_norm_cap", norm_cap)),
    }
    basis_result = build_v27_localized_behavior_loss_basis(
        config=localized_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        spectral_basis=spectral_basis,
        train_pool_file_sha256=train_pool_file_sha256,
        train_pool_summary_hash=train_pool_summary_hash,
        script_sha256=script_sha256,
    )
    edit = solve_v27_localized_behavior_loss_edit(
        basis=basis_result["basis"],
        basis_audit=basis_result["audit"],
        config=localized_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        record_id_hash=record_id_hash,
        selected_config_hash=selected_config_hash,
    )
    matched_spectral_projection_norm = 0.0
    if spectral_basis is not None:
        matched_spectral_projection_norm = float(
            torch.linalg.norm(project_to_basis(edit["delta"], spectral_basis)).item()
        )
    metadata = {
        **edit["audit"],
        "matched_edit_source": V27_LOCALIZED_MATCHED_EDIT_SOURCE,
        "matched_spectral_projection_norm": matched_spectral_projection_norm,
        "selected_config_hash": selected_config_hash,
    }
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=edit["delta"],
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata=metadata,
    )


def solve_v28_anchor_nullspace_edit(
    *,
    basis: torch.Tensor,
    basis_audit: Mapping[str, Any],
    config: Mapping[str, Any],
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
    selected_config_hash: str | None = None,
) -> dict[str, Any]:
    basis = validate_v27_localized_basis_matrix(basis)
    basis_hash = require_sha256_hex(basis_audit["basis_hash"], field_name="basis_hash")
    source = source_weights.detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(source.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("source weights have wrong dimension")
    if not torch.isfinite(source).all():
        raise ValueError("nonfinite source weights")
    support = v27_support_tensors_for_source_target(
        source_weights=source,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    trust_norm_cap = float(config.get("trust_norm_cap", 0.0))
    if trust_norm_cap < 0.0 or not math.isfinite(trust_norm_cap):
        raise ValueError("trust_norm_cap must be finite and nonnegative")
    alpha = torch.zeros(int(basis.shape[1]), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [alpha],
        lr=V28_ANCHOR_NULLSPACE_LR,
        betas=V27_LOCALIZED_OPTIMIZER_BETAS,
        eps=V27_LOCALIZED_OPTIMIZER_EPS,
        weight_decay=0.0,
        amsgrad=False,
    )
    trace: list[dict[str, Any]] = []
    best: tuple[float, float, int, torch.Tensor, dict[str, Any]] | None = None
    for step in range(1, V28_ANCHOR_NULLSPACE_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = basis @ alpha
        edited = source + delta
        target_logits = v27_subject_logits_for_inputs(edited, support["target_inputs"])
        conflict_logits = v27_subject_logits_for_inputs(edited, support["conflict_inputs"])
        compatible_logits = v27_subject_logits_for_inputs(edited, support["compatible_inputs"])
        target_bce = F.binary_cross_entropy_with_logits(
            target_logits,
            support["target_labels"].to(dtype=torch.float32),
        )
        conflict_bce = F.binary_cross_entropy_with_logits(
            conflict_logits,
            support["conflict_target_labels"].to(dtype=torch.float32),
        )
        compatible_mse = F.mse_loss(
            compatible_logits,
            support["compatible_source_logits"].to(dtype=torch.float32),
        )
        delta_l2 = torch.mean(delta.pow(2))
        loss = (
            V28_TARGET_BCE_WEIGHT * target_bce
            + V28_CONFLICT_BCE_WEIGHT * conflict_bce
            + V28_COMPATIBLE_PROBE_WEIGHT * compatible_mse
            + V28_DELTA_L2_WEIGHT * delta_l2
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite anchor-nullspace support loss")
        loss.backward()
        if alpha.grad is None or not torch.isfinite(alpha.grad).all():
            raise ValueError("nonfinite anchor-nullspace alpha gradient")
        torch.nn.utils.clip_grad_norm_([alpha], V27_LOCALIZED_GRAD_CLIP_NORM)
        optimizer.step()
        with torch.no_grad():
            current_delta = (basis @ alpha).detach().clone().to(dtype=torch.float32)
            clipped_delta = apply_norm_cap(current_delta, max_norm=trust_norm_cap)
            if float(torch.linalg.norm(current_delta).item()) > trust_norm_cap + 1e-8:
                alpha.copy_(basis.T @ clipped_delta)
            if not torch.isfinite(alpha).all():
                raise ValueError("nonfinite anchor-nullspace alpha")
            current_delta = (basis @ alpha).detach().clone().to(dtype=torch.float32)
            current_delta_norm = float(torch.linalg.norm(current_delta).item())
            scalar_losses = {
                "compatible_mse": float(compatible_mse.detach().item()),
                "conflict_bce": float(conflict_bce.detach().item()),
                "delta_l2": float(delta_l2.detach().item()),
                "loss": float(loss.detach().item()),
                "target_bce": float(target_bce.detach().item()),
            }
            if step % 10 == 0 or step == V28_ANCHOR_NULLSPACE_STEPS:
                progress_event = redact_v27_optimizer_progress_event({
                    "basis_hash": basis_hash,
                    "delta_norm": current_delta_norm,
                    "loss": scalar_losses["loss"],
                    "step": step,
                })
                trace.append(progress_event)
                if progress_log_path is not None:
                    event_extra = dict(progress_event)
                    if record_id_hash is not None:
                        event_extra["record_id_hash"] = require_sha256_hex(
                            record_id_hash,
                            field_name="record_id_hash",
                        )
                    if selected_config_hash is not None:
                        event_extra["selected_config_hash"] = require_sha256_hex(
                            selected_config_hash,
                            field_name="selected_config_hash",
                        )
                    record_progress_event(
                        progress_log_path,
                        event="v28_anchor_nullspace_optimizer_progress",
                        started_at_monotonic=started_at_monotonic,
                        now_monotonic=now_monotonic,
                        extra=event_extra,
                    )
            objective_key = (
                scalar_losses["loss"],
                current_delta_norm,
                step,
            )
            if best is None or objective_key < best[:3]:
                best = (
                    scalar_losses["loss"],
                    current_delta_norm,
                    step,
                    current_delta,
                    scalar_losses,
                )
    if best is None:
        raise ValueError("anchor-nullspace optimizer produced no candidate")
    best_loss, best_delta_norm, best_step, best_delta, best_losses = best
    clipped_delta = apply_norm_cap(best_delta, max_norm=trust_norm_cap)
    hard_norm_clipped = bool(float(torch.linalg.norm(best_delta).item()) > trust_norm_cap + 1e-8)
    audit = {
        **dict(basis_audit),
        "basis_hash": basis_hash,
        "best_step": int(best_step),
        "delta_norm_before_hard_cap": float(best_delta_norm),
        "delta_sha256": stable_hash_json(tensor_to_hashable(clipped_delta)),
        "hard_norm_clipped": hard_norm_clipped,
        "optimization_boundary": v28_anchor_nullspace_optimization_boundary(),
        "optimization_split": "support",
        "optimizer": "AdamW",
        "optimizer_trace_hash": stable_hash_json(trace),
        "proof_split": "heldout",
        "support_objective": float(best_loss),
        "support_objective_is_proof_metric": False,
        "support_scalar_losses": best_losses,
        "trust_norm_cap": float(trust_norm_cap),
    }
    finite_failures = recursive_numeric_finiteness_failures(audit)
    if finite_failures:
        raise ValueError("nonfinite anchor-nullspace optimizer audit: " + ", ".join(finite_failures[:5]))
    return {
        "audit": audit,
        "delta": clipped_delta,
    }


def evaluate_v25_anchor_nullspace_trust_region_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    config: Mapping[str, Any],
    norm_cap: float,
    selected_config_hash: str,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
) -> dict[str, Any]:
    source_weights = record_weights_tensor(subject)
    anchor_config = {
        **dict(config),
        "trust_norm_cap": float(config.get("trust_norm_cap", norm_cap)),
    }
    basis_result = build_v28_anchor_nullspace_basis(
        config=anchor_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        train_pool_file_sha256=train_pool_file_sha256,
        train_pool_summary_hash=train_pool_summary_hash,
        script_sha256=script_sha256,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        record_id_hash=record_id_hash,
        selected_config_hash=selected_config_hash,
    )
    edit = solve_v28_anchor_nullspace_edit(
        basis=basis_result["basis"],
        basis_audit=basis_result["audit"],
        config=anchor_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        record_id_hash=record_id_hash,
        selected_config_hash=selected_config_hash,
    )
    metadata = {
        **edit["audit"],
        "matched_edit_source": V28_ANCHOR_NULLSPACE_MATCHED_EDIT_SOURCE,
        "matched_spectral_projection_norm": 0.0,
        "selected_config_hash": selected_config_hash,
    }
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=edit["delta"],
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata=metadata,
    )


def evaluate_v25_breadth_first_sparse_support_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    config: Mapping[str, Any],
    norm_cap: float,
    selected_config_hash: str,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
) -> dict[str, Any]:
    require_sha256_hex(train_pool_file_sha256, field_name="train_pool_file_sha256")
    require_sha256_hex(train_pool_summary_hash, field_name="train_pool_summary_hash")
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    source_weights = record_weights_tensor(subject)
    sparse_config = {
        **dict(config),
        "trust_norm_cap": float(config.get("trust_norm_cap", norm_cap)),
    }
    gradients = v29_sparse_support_gradients(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    sparse_top_k = int(sparse_config.get("sparse_top_k", 0))
    compatible_floor = float(sparse_config.get("compatible_floor", 0.0))
    extra_compatible_weight = float(sparse_config.get("extra_compatible_weight", 0.0))
    selected_coordinates = select_v29_sparse_coordinates(
        g_target=gradients["g_target"],
        g_conflict=gradients["g_conflict"],
        g_compatible=gradients["g_compatible"],
        sparse_top_k=sparse_top_k,
        compatible_floor=compatible_floor,
        conflict_weight=V29_CONFLICT_WEIGHT,
    )
    if not selected_coordinates:
        raise ValueError("breadth-first sparse support selected no coordinates")
    coordinate_hash = stable_hash_json(selected_coordinates)
    if progress_log_path is not None:
        extra = redact_v29_sparse_support_progress_event({
            **({"record_id_hash": record_id_hash} if record_id_hash else {}),
            "compatible_count": int(gradients["support_split_counts"]["compatible"]),
            "compatible_floor": compatible_floor,
            "conflict_count": int(gradients["support_split_counts"]["conflict"]),
            "coordinate_hash": coordinate_hash,
            "extra_compatible_weight": extra_compatible_weight,
            "selected_config_hash": selected_hash,
            "selected_coordinate_count": len(selected_coordinates),
            "sparse_top_k": sparse_top_k,
            "target_count": int(gradients["support_split_counts"]["target"]),
            "trust_norm_cap": float(sparse_config["trust_norm_cap"]),
        })
        record_progress_event(
            progress_log_path,
            event="v29_sparse_coordinate_selection_completed",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=extra,
        )
    edit = solve_v29_breadth_first_sparse_support_edit(
        coordinate_hash=coordinate_hash,
        selected_coordinates=selected_coordinates,
        config=sparse_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        record_id_hash=record_id_hash,
        selected_config_hash=selected_hash,
    )
    metadata = {
        **edit["audit"],
        "matched_edit_source": V29_BREADTH_FIRST_SPARSE_MATCHED_EDIT_SOURCE,
        "matched_spectral_projection_norm": 0.0,
        "script_sha256": str(script_sha256),
        "selected_config_hash": selected_hash,
        "source_behavior": str(source_behavior),
        "support_split_counts": dict(gradients["support_split_counts"]),
        "target_behavior": str(target_behavior),
        "train_pool_file_sha256": str(train_pool_file_sha256),
        "train_pool_summary_hash": str(train_pool_summary_hash),
    }
    metadata["sparse_support_provenance_hash"] = stable_hash_json({
        key: value
        for key, value in metadata.items()
        if key not in {"support_scalar_losses"}
    })
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=edit["delta"],
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata=metadata,
    )


def evaluate_v25_margin_gated_sparse_support_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    config: Mapping[str, Any],
    norm_cap: float,
    selected_config_hash: str,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
) -> dict[str, Any]:
    require_sha256_hex(train_pool_file_sha256, field_name="train_pool_file_sha256")
    require_sha256_hex(train_pool_summary_hash, field_name="train_pool_summary_hash")
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    source_weights = record_weights_tensor(subject)
    margin_config = {
        **dict(config),
        "trust_norm_cap": float(config.get("trust_norm_cap", norm_cap)),
    }
    gradients = v29_sparse_support_gradients(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    sparse_top_k = int(margin_config.get("sparse_top_k", 0))
    compatible_floor = float(margin_config.get("compatible_floor", V30_COMPATIBLE_FLOOR))
    selected_coordinates = select_v29_sparse_coordinates(
        g_target=gradients["g_target"],
        g_conflict=gradients["g_conflict"],
        g_compatible=gradients["g_compatible"],
        sparse_top_k=sparse_top_k,
        compatible_floor=compatible_floor,
        conflict_weight=V29_CONFLICT_WEIGHT,
    )
    if not selected_coordinates:
        raise ValueError("margin-gated sparse support selected no coordinates")
    coordinate_hash = stable_hash_json(selected_coordinates)
    if progress_log_path is not None:
        extra = redact_v30_margin_gated_progress_event({
            **({"record_id_hash": record_id_hash} if record_id_hash else {}),
            "compatible_count": int(gradients["support_split_counts"]["compatible"]),
            "compatible_floor": compatible_floor,
            "conflict_count": int(gradients["support_split_counts"]["conflict"]),
            "coordinate_hash": coordinate_hash,
            "extra_compatible_weight": float(
                margin_config.get("extra_compatible_weight", V30_EXTRA_COMPATIBLE_WEIGHT)
            ),
            "selected_config_hash": selected_hash,
            "selected_coordinate_count": len(selected_coordinates),
            "sparse_top_k": sparse_top_k,
            "target_count": int(gradients["support_split_counts"]["target"]),
            "target_margin_floor": float(margin_config["target_margin_floor"]),
            "trust_norm_cap": float(margin_config["trust_norm_cap"]),
        })
        record_progress_event(
            progress_log_path,
            event="v30_sparse_coordinate_selection_completed",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=extra,
        )
    edit = solve_v30_margin_gated_sparse_support_edit(
        coordinate_hash=coordinate_hash,
        selected_coordinates=selected_coordinates,
        config=margin_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        record_id_hash=record_id_hash,
        selected_config_hash=selected_hash,
    )
    metadata = {
        **edit["audit"],
        "matched_edit_source": V30_MARGIN_GATED_SPARSE_MATCHED_EDIT_SOURCE,
        "matched_spectral_projection_norm": 0.0,
        "script_sha256": str(script_sha256),
        "selected_config_hash": selected_hash,
        "source_behavior": str(source_behavior),
        "support_split_counts": dict(gradients["support_split_counts"]),
        "target_behavior": str(target_behavior),
        "train_pool_file_sha256": str(train_pool_file_sha256),
        "train_pool_summary_hash": str(train_pool_summary_hash),
    }
    metadata["margin_gated_sparse_provenance_hash"] = stable_hash_json({
        key: value
        for key, value in metadata.items()
        if key not in {"support_scalar_losses"}
    })
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=edit["delta"],
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata=metadata,
    )


def evaluate_v25_orthogonal_sign_sparse_support_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    config: Mapping[str, Any],
    norm_cap: float,
    selected_config_hash: str,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
) -> dict[str, Any]:
    require_sha256_hex(train_pool_file_sha256, field_name="train_pool_file_sha256")
    require_sha256_hex(train_pool_summary_hash, field_name="train_pool_summary_hash")
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    source_weights = record_weights_tensor(subject)
    sign_config = {
        **dict(config),
        "trust_norm_cap": float(config.get("trust_norm_cap", norm_cap)),
    }
    gradients = v29_sparse_support_gradients(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    sparse_top_k = int(sign_config.get("sparse_top_k", V31_SPARSE_TOP_K))
    compatible_floor = float(sign_config.get("compatible_floor", V31_COMPATIBLE_FLOOR))
    sign_conflict_penalty = float(
        sign_config.get("sign_conflict_penalty", V31_SIGN_CONFLICT_PENALTY_GRID[0])
    )
    selected_coordinates = select_v31_sign_coherent_sparse_coordinates(
        g_target=gradients["g_target"],
        g_conflict=gradients["g_conflict"],
        g_compatible=gradients["g_compatible"],
        sparse_top_k=sparse_top_k,
        compatible_floor=compatible_floor,
        conflict_weight=V29_CONFLICT_WEIGHT,
        sign_conflict_penalty=sign_conflict_penalty,
    )
    if not selected_coordinates:
        raise ValueError("orthogonal sign sparse support selected no coordinates")
    coordinate_hash = stable_hash_json(selected_coordinates)
    if progress_log_path is not None:
        extra = redact_v31_orthogonal_sign_progress_event({
            **({"record_id_hash": record_id_hash} if record_id_hash else {}),
            "compatible_count": int(gradients["support_split_counts"]["compatible"]),
            "compatible_floor": compatible_floor,
            "compatible_orthogonal_weight": float(
                sign_config.get(
                    "compatible_orthogonal_weight",
                    V31_COMPATIBLE_ORTHOGONAL_WEIGHT_GRID[0],
                )
            ),
            "conflict_count": int(gradients["support_split_counts"]["conflict"]),
            "coordinate_hash": coordinate_hash,
            "extra_compatible_weight": float(
                sign_config.get("extra_compatible_weight", V31_EXTRA_COMPATIBLE_WEIGHT)
            ),
            "hard_target_margin_weight": float(
                sign_config.get(
                    "hard_target_margin_weight",
                    V31_HARD_TARGET_MARGIN_WEIGHT,
                )
            ),
            "selected_config_hash": selected_hash,
            "selected_coordinate_count": len(selected_coordinates),
            "sign_conflict_penalty": sign_conflict_penalty,
            "sparse_top_k": sparse_top_k,
            "target_count": int(gradients["support_split_counts"]["target"]),
            "target_margin_floor": float(sign_config["target_margin_floor"]),
            "trust_norm_cap": float(sign_config["trust_norm_cap"]),
        })
        record_progress_event(
            progress_log_path,
            event="v31_sign_coherent_coordinate_selection_completed",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=extra,
        )
    edit = solve_v31_orthogonal_sign_sparse_support_edit(
        compatible_gradient=gradients["g_compatible"],
        coordinate_hash=coordinate_hash,
        selected_coordinates=selected_coordinates,
        config=sign_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        record_id_hash=record_id_hash,
        selected_config_hash=selected_hash,
    )
    metadata = {
        **edit["audit"],
        "matched_edit_source": V31_ORTHOGONAL_SIGN_SPARSE_MATCHED_EDIT_SOURCE,
        "matched_spectral_projection_norm": 0.0,
        "script_sha256": str(script_sha256),
        "selected_config_hash": selected_hash,
        "source_behavior": str(source_behavior),
        "support_split_counts": dict(gradients["support_split_counts"]),
        "target_behavior": str(target_behavior),
        "train_pool_file_sha256": str(train_pool_file_sha256),
        "train_pool_summary_hash": str(train_pool_summary_hash),
    }
    metadata["orthogonal_sign_sparse_provenance_hash"] = stable_hash_json({
        key: value
        for key, value in metadata.items()
        if key not in {"support_scalar_losses"}
    })
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=edit["delta"],
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata=metadata,
    )


def evaluate_v25_support_tournament_sparse_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    config: Mapping[str, Any],
    norm_cap: float,
    selected_config_hash: str,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
) -> dict[str, Any]:
    require_sha256_hex(train_pool_file_sha256, field_name="train_pool_file_sha256")
    require_sha256_hex(train_pool_summary_hash, field_name="train_pool_summary_hash")
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    source_weights = record_weights_tensor(subject)
    tournament_config = {
        **dict(config),
        "trust_norm_cap": float(config.get("trust_norm_cap", norm_cap)),
    }
    gradients = v29_sparse_support_gradients(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    sparse_top_k = int(tournament_config.get("sparse_top_k", V32_SPARSE_TOP_K))
    compatible_floor = float(tournament_config.get("compatible_floor", V32_COMPATIBLE_FLOOR))
    sign_conflict_penalty = float(
        tournament_config.get("sign_conflict_penalty", V32_SIGN_CONFLICT_PENALTY)
    )
    selected_coordinates = select_v31_sign_coherent_sparse_coordinates(
        g_target=gradients["g_target"],
        g_conflict=gradients["g_conflict"],
        g_compatible=gradients["g_compatible"],
        sparse_top_k=sparse_top_k,
        compatible_floor=compatible_floor,
        conflict_weight=V29_CONFLICT_WEIGHT,
        sign_conflict_penalty=sign_conflict_penalty,
    )
    if not selected_coordinates:
        raise ValueError("support tournament sparse selected no coordinates")
    coordinate_hash = stable_hash_json(selected_coordinates)
    if progress_log_path is not None:
        extra = redact_v32_support_tournament_progress_event({
            **({"record_id_hash": record_id_hash} if record_id_hash else {}),
            "compatible_count": int(gradients["support_split_counts"]["compatible"]),
            "compatible_floor": compatible_floor,
            "compatible_orthogonal_weight": float(
                tournament_config.get(
                    "compatible_orthogonal_weight",
                    V32_COMPATIBLE_ORTHOGONAL_WEIGHT_GRID[0],
                )
            ),
            "conflict_count": int(gradients["support_split_counts"]["conflict"]),
            "coordinate_hash": coordinate_hash,
            "extra_compatible_weight": float(
                tournament_config.get(
                    "extra_compatible_weight",
                    V32_EXTRA_COMPATIBLE_WEIGHT,
                )
            ),
            "hard_target_margin_weight": float(
                tournament_config.get(
                    "hard_target_margin_weight",
                    V32_HARD_TARGET_MARGIN_WEIGHT,
                )
            ),
            "selected_config_hash": selected_hash,
            "selected_coordinate_count": len(selected_coordinates),
            "sign_conflict_penalty": sign_conflict_penalty,
            "sparse_top_k": sparse_top_k,
            "target_count": int(gradients["support_split_counts"]["target"]),
            "target_margin_floor": float(tournament_config["target_margin_floor"]),
            "tournament_margin_floor": float(
                tournament_config["tournament_margin_floor"]
            ),
            "tournament_margin_weight": float(
                tournament_config["tournament_margin_weight"]
            ),
            "trust_norm_cap": float(tournament_config["trust_norm_cap"]),
        })
        record_progress_event(
            progress_log_path,
            event="v32_support_tournament_margin_prepared",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=extra,
        )
    edit = solve_v32_support_tournament_sparse_edit(
        compatible_gradient=gradients["g_compatible"],
        coordinate_hash=coordinate_hash,
        selected_coordinates=selected_coordinates,
        config=tournament_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        record_id_hash=record_id_hash,
        selected_config_hash=selected_hash,
    )
    metadata = {
        **edit["audit"],
        "matched_edit_source": V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE,
        "matched_spectral_projection_norm": 0.0,
        "script_sha256": str(script_sha256),
        "selected_config_hash": selected_hash,
        "source_behavior": str(source_behavior),
        "support_split_counts": dict(gradients["support_split_counts"]),
        "target_behavior": str(target_behavior),
        "train_pool_file_sha256": str(train_pool_file_sha256),
        "train_pool_summary_hash": str(train_pool_summary_hash),
    }
    metadata["support_tournament_provenance_hash"] = stable_hash_json({
        key: value
        for key, value in metadata.items()
        if key not in {"support_scalar_losses"}
    })
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=edit["delta"],
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata=metadata,
    )


def evaluate_v25_support_source_line_search_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    config: Mapping[str, Any],
    norm_cap: float,
    selected_config_hash: str,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
) -> dict[str, Any]:
    require_sha256_hex(train_pool_file_sha256, field_name="train_pool_file_sha256")
    require_sha256_hex(train_pool_summary_hash, field_name="train_pool_summary_hash")
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    source_weights = record_weights_tensor(subject)
    line_search_config = {
        **dict(config),
        "trust_norm_cap": float(config.get("trust_norm_cap", norm_cap)),
    }
    gradients = v29_sparse_support_gradients(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    sparse_top_k = int(line_search_config.get("sparse_top_k", V32_SPARSE_TOP_K))
    compatible_floor = float(
        line_search_config.get("compatible_floor", V32_COMPATIBLE_FLOOR)
    )
    sign_conflict_penalty = float(
        line_search_config.get("sign_conflict_penalty", V32_SIGN_CONFLICT_PENALTY)
    )
    selected_coordinates = select_v31_sign_coherent_sparse_coordinates(
        g_target=gradients["g_target"],
        g_conflict=gradients["g_conflict"],
        g_compatible=gradients["g_compatible"],
        sparse_top_k=sparse_top_k,
        compatible_floor=compatible_floor,
        conflict_weight=V29_CONFLICT_WEIGHT,
        sign_conflict_penalty=sign_conflict_penalty,
    )
    if not selected_coordinates:
        raise ValueError("support source line search selected no coordinates")
    coordinate_hash = stable_hash_json(selected_coordinates)
    alpha_candidates = [
        float(value)
        for value in line_search_config.get("alpha_candidates", V35_ALPHA_CANDIDATES)
    ]
    if not alpha_candidates:
        raise ValueError("alpha_candidates must be nonempty")
    if any((not math.isfinite(value)) or value < 0.0 for value in alpha_candidates):
        raise ValueError("alpha_candidates must be finite and nonnegative")
    alpha_candidates_hash = stable_hash_json(alpha_candidates)
    if progress_log_path is not None:
        extra = redact_v35_support_source_alpha_progress_event({
            **({"record_id_hash": record_id_hash} if record_id_hash else {}),
            "alpha_candidate_count": len(alpha_candidates),
            "alpha_candidates_hash": alpha_candidates_hash,
            "alpha_target_margin_floor": float(
                line_search_config.get("alpha_target_margin_floor", 0.10)
            ),
            "alpha_tournament_margin_floor": float(
                line_search_config.get("alpha_tournament_margin_floor", 0.0)
            ),
            "compatible_count": int(gradients["support_split_counts"]["compatible"]),
            "compatible_floor": compatible_floor,
            "compatible_orthogonal_weight": float(
                line_search_config.get(
                    "compatible_orthogonal_weight",
                    V32_COMPATIBLE_ORTHOGONAL_WEIGHT_GRID[0],
                )
            ),
            "conflict_count": int(gradients["support_split_counts"]["conflict"]),
            "coordinate_hash": coordinate_hash,
            "extra_compatible_weight": float(
                line_search_config.get(
                    "extra_compatible_weight",
                    V32_EXTRA_COMPATIBLE_WEIGHT,
                )
            ),
            "fallback_target_penalty": float(
                line_search_config.get("fallback_target_penalty", 10.0)
            ),
            "fallback_tournament_penalty": float(
                line_search_config.get("fallback_tournament_penalty", 5.0)
            ),
            "hard_target_margin_weight": float(
                line_search_config.get(
                    "hard_target_margin_weight",
                    V32_HARD_TARGET_MARGIN_WEIGHT,
                )
            ),
            "selected_config_hash": selected_hash,
            "selected_coordinate_count": len(selected_coordinates),
            "sign_conflict_penalty": sign_conflict_penalty,
            "sparse_top_k": sparse_top_k,
            "target_count": int(gradients["support_split_counts"]["target"]),
            "target_margin_floor": float(line_search_config["target_margin_floor"]),
            "tournament_margin_floor": float(
                line_search_config["tournament_margin_floor"]
            ),
            "tournament_margin_weight": float(
                line_search_config["tournament_margin_weight"]
            ),
            "trust_norm_cap": float(line_search_config["trust_norm_cap"]),
        })
        record_progress_event(
            progress_log_path,
            event="v35_support_source_line_search_prepared",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=extra,
        )
    base_edit = solve_v32_support_tournament_sparse_edit(
        compatible_gradient=gradients["g_compatible"],
        coordinate_hash=coordinate_hash,
        selected_coordinates=selected_coordinates,
        config=line_search_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        record_id_hash=record_id_hash,
        selected_config_hash=selected_hash,
    )
    base_delta = base_edit["delta"].detach().clone().to(dtype=torch.float32).reshape(-1)
    if int(base_delta.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("base support source line search delta has wrong dimension")
    support = v27_support_tensors_for_source_target(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    tournament_tensors = v32_support_behavior_margin_tensors()
    alpha_tournament_floor = float(
        line_search_config.get("alpha_tournament_margin_floor", 0.0)
    )
    alpha_target_floor = float(
        line_search_config.get("alpha_target_margin_floor", 0.10)
    )
    fallback_target_penalty = float(
        line_search_config.get("fallback_target_penalty", 10.0)
    )
    fallback_tournament_penalty = float(
        line_search_config.get("fallback_tournament_penalty", 5.0)
    )
    candidates: list[dict[str, Any]] = []
    with torch.no_grad():
        for alpha in alpha_candidates:
            candidate_delta = base_delta * float(alpha)
            edited = source_weights.to(dtype=torch.float32).reshape(-1) + candidate_delta
            compatible_logits = v27_subject_logits_for_inputs(
                edited,
                support["compatible_inputs"],
            )
            support_compatible_mse = F.mse_loss(
                compatible_logits,
                support["compatible_source_logits"].to(dtype=torch.float32),
            )
            tournament = v32_support_tournament_margin_loss(
                margins=v32_support_behavior_margins(
                    weights=edited,
                    tournament_tensors=tournament_tensors,
                ),
                target_behavior=target_behavior,
                tournament_margin_floor=max(0.0, alpha_tournament_floor),
            )
            candidates.append({
                "alpha": float(alpha),
                "support_compatible_mse": float(support_compatible_mse.item()),
                "support_runner_margin": float(
                    tournament["support_runner_margin"].item()
                ),
                "support_target_margin": float(
                    tournament["support_target_margin"].item()
                ),
                "support_tournament_margin": float(
                    tournament["support_tournament_margin"].item()
                ),
            })
    selected_alpha = select_v35_support_source_alpha_candidate(
        candidates=candidates,
        alpha_target_margin_floor=alpha_target_floor,
        alpha_tournament_margin_floor=alpha_tournament_floor,
        fallback_target_penalty=fallback_target_penalty,
        fallback_tournament_penalty=fallback_tournament_penalty,
    )
    selected_delta = base_delta * float(selected_alpha["alpha"])
    selected_alpha_candidate_hash = stable_hash_json(selected_alpha)
    alpha_selection = {
        key: selected_alpha[key]
        for key in [
            "alpha",
            "alpha_candidate_count",
            "candidate_index",
            "candidate_metrics_hash",
            "eligible_count",
            "eligible",
            "fallback_score",
            "selection_mode",
            "support_compatible_mse",
            "support_runner_margin",
            "support_target_margin",
            "support_tournament_margin",
        ]
        if key in selected_alpha
    }
    if progress_log_path is not None:
        extra = redact_v35_support_source_alpha_progress_event({
            **selected_alpha,
            **({"record_id_hash": record_id_hash} if record_id_hash else {}),
            "alpha_candidates_hash": alpha_candidates_hash,
            "delta_norm": float(torch.linalg.norm(selected_delta).item()),
            "selected_alpha_candidate_hash": selected_alpha_candidate_hash,
            "selected_config_hash": selected_hash,
        })
        record_progress_event(
            progress_log_path,
            event="v35_support_source_alpha_selected",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=extra,
        )
    metadata = {
        "alpha_candidate_count": len(candidates),
        "alpha_candidates_hash": alpha_candidates_hash,
        "alpha_selection": alpha_selection,
        "base_delta_sha256": stable_hash_json(tensor_to_hashable(base_delta)),
        "base_support_tournament_audit_hash": stable_hash_json(base_edit["audit"]),
        "candidate_metrics_hash": selected_alpha["candidate_metrics_hash"],
        "eligible_count": int(selected_alpha["eligible_count"]),
        "matched_edit_source": V35_SUPPORT_SOURCE_LINE_SEARCH_MATCHED_EDIT_SOURCE,
        "matched_spectral_projection_norm": 0.0,
        "optimization_boundary": v35_support_source_line_search_optimization_boundary(),
        "optimization_split": "support",
        "proof_split": "heldout",
        "script_sha256": str(script_sha256),
        "selected_alpha": float(selected_alpha["alpha"]),
        "selected_alpha_candidate_hash": selected_alpha_candidate_hash,
        "selected_config_hash": selected_hash,
        "source_behavior": str(source_behavior),
        "support_objective_is_proof_metric": False,
        "support_split_counts": dict(gradients["support_split_counts"]),
        "target_behavior": str(target_behavior),
        "train_pool_file_sha256": str(train_pool_file_sha256),
        "train_pool_summary_hash": str(train_pool_summary_hash),
    }
    metadata["support_source_line_search_provenance_hash"] = stable_hash_json(metadata)
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=selected_delta,
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata=metadata,
    )


def evaluate_v25_compatible_nullspace_projected_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    config: Mapping[str, Any],
    norm_cap: float,
    selected_config_hash: str,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
) -> dict[str, Any]:
    require_sha256_hex(train_pool_file_sha256, field_name="train_pool_file_sha256")
    require_sha256_hex(train_pool_summary_hash, field_name="train_pool_summary_hash")
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    source_weights = record_weights_tensor(subject)
    projection_config = {
        **dict(config),
        "trust_norm_cap": float(config.get("trust_norm_cap", norm_cap)),
    }
    gradient_info = v28_anchor_gradients_and_compatible_jacobian(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    sparse_top_k = int(projection_config.get("sparse_top_k", V32_SPARSE_TOP_K))
    compatible_floor = float(
        projection_config.get("compatible_floor", V32_COMPATIBLE_FLOOR)
    )
    sign_conflict_penalty = float(
        projection_config.get("sign_conflict_penalty", V32_SIGN_CONFLICT_PENALTY)
    )
    selected_coordinates = select_v31_sign_coherent_sparse_coordinates(
        g_target=gradient_info["g_target"],
        g_conflict=gradient_info["g_conflict"],
        g_compatible=gradient_info["g_compatible"],
        sparse_top_k=sparse_top_k,
        compatible_floor=compatible_floor,
        conflict_weight=V29_CONFLICT_WEIGHT,
        sign_conflict_penalty=sign_conflict_penalty,
    )
    if not selected_coordinates:
        raise ValueError("compatible nullspace projection selected no coordinates")
    coordinate_hash = stable_hash_json(selected_coordinates)
    base_edit = solve_v32_support_tournament_sparse_edit(
        compatible_gradient=gradient_info["g_compatible"],
        coordinate_hash=coordinate_hash,
        selected_coordinates=selected_coordinates,
        config=projection_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        record_id_hash=record_id_hash,
        selected_config_hash=selected_hash,
    )
    projected = project_v36_delta_through_compatible_nullspace(
        base_delta=base_edit["delta"],
        compatible_jacobian=gradient_info["compatible_jacobian"],
        compatible_nullspace_rtol=float(
            projection_config.get("compatible_nullspace_rtol", 1e-4)
        ),
        projection_strength=float(projection_config.get("projection_strength", 1.0)),
        trust_norm_cap=float(projection_config["trust_norm_cap"]),
    )
    projection_audit = dict(projected["audit"])
    projection_audit_hash = stable_hash_json(projection_audit)
    if progress_log_path is not None:
        extra = redact_v36_compatible_nullspace_progress_event({
            **projection_audit,
            **({"record_id_hash": record_id_hash} if record_id_hash else {}),
            "projection_audit_hash": projection_audit_hash,
            "selected_config_hash": selected_hash,
        })
        record_progress_event(
            progress_log_path,
            event="v36_compatible_nullspace_projection_completed",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=extra,
        )

    support = v27_support_tensors_for_source_target(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    tournament_tensors = v32_support_behavior_margin_tensors()
    alpha_candidates = [
        float(value)
        for value in projection_config.get("alpha_candidates", V35_ALPHA_CANDIDATES)
    ]
    alpha_candidates_hash = stable_hash_json(alpha_candidates)
    alpha_tournament_floor = float(
        projection_config.get("alpha_tournament_margin_floor", 0.0)
    )
    alpha_target_floor = float(
        projection_config.get("alpha_target_margin_floor", 0.05)
    )
    candidates: list[dict[str, Any]] = []
    base_delta = projected["delta"].detach().clone().to(dtype=torch.float32).reshape(-1)
    with torch.no_grad():
        for alpha in alpha_candidates:
            candidate_delta = base_delta * float(alpha)
            edited = source_weights.to(dtype=torch.float32).reshape(-1) + candidate_delta
            compatible_logits = v27_subject_logits_for_inputs(
                edited,
                support["compatible_inputs"],
            )
            support_compatible_mse = F.mse_loss(
                compatible_logits,
                support["compatible_source_logits"].to(dtype=torch.float32),
            )
            tournament = v32_support_tournament_margin_loss(
                margins=v32_support_behavior_margins(
                    weights=edited,
                    tournament_tensors=tournament_tensors,
                ),
                target_behavior=target_behavior,
                tournament_margin_floor=max(0.0, alpha_tournament_floor),
            )
            candidates.append({
                "alpha": float(alpha),
                "support_compatible_mse": float(support_compatible_mse.item()),
                "support_runner_margin": float(tournament["support_runner_margin"].item()),
                "support_target_margin": float(tournament["support_target_margin"].item()),
                "support_tournament_margin": float(
                    tournament["support_tournament_margin"].item()
                ),
            })
    selected_alpha = select_v35_support_source_alpha_candidate(
        candidates=candidates,
        alpha_target_margin_floor=alpha_target_floor,
        alpha_tournament_margin_floor=alpha_tournament_floor,
        fallback_target_penalty=float(projection_config.get("fallback_target_penalty", 10.0)),
        fallback_tournament_penalty=float(
            projection_config.get("fallback_tournament_penalty", 5.0)
        ),
    )
    selected_delta = base_delta * float(selected_alpha["alpha"])
    selected_alpha_candidate_hash = stable_hash_json(selected_alpha)
    alpha_selection = {
        key: selected_alpha[key]
        for key in [
            "alpha",
            "alpha_candidate_count",
            "candidate_index",
            "candidate_metrics_hash",
            "eligible",
            "eligible_count",
            "fallback_score",
            "selection_mode",
            "support_compatible_mse",
            "support_runner_margin",
            "support_target_margin",
            "support_tournament_margin",
        ]
        if key in selected_alpha
    }
    if progress_log_path is not None:
        extra = redact_v35_support_source_alpha_progress_event({
            **selected_alpha,
            **({"record_id_hash": record_id_hash} if record_id_hash else {}),
            "alpha_candidates_hash": alpha_candidates_hash,
            "delta_norm": float(torch.linalg.norm(selected_delta).item()),
            "selected_alpha_candidate_hash": selected_alpha_candidate_hash,
            "selected_config_hash": selected_hash,
        })
        record_progress_event(
            progress_log_path,
            event="v35_support_source_alpha_selected",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=extra,
        )
    metadata = {
        "alpha_candidate_count": len(candidates),
        "alpha_candidates_hash": alpha_candidates_hash,
        "alpha_selection": alpha_selection,
        "base_delta_sha256": stable_hash_json(tensor_to_hashable(base_edit["delta"])),
        "base_support_tournament_audit_hash": stable_hash_json(base_edit["audit"]),
        "candidate_metrics_hash": selected_alpha["candidate_metrics_hash"],
        "eligible_count": int(selected_alpha["eligible_count"]),
        "matched_edit_source": V36_COMPATIBLE_NULLSPACE_MATCHED_EDIT_SOURCE,
        "matched_spectral_projection_norm": 0.0,
        "optimization_boundary": v36_compatible_nullspace_optimization_boundary(),
        "optimization_split": "support",
        "projection_audit": projection_audit,
        "projection_audit_hash": projection_audit_hash,
        "proof_split": "heldout",
        "script_sha256": str(script_sha256),
        "selected_alpha": float(selected_alpha["alpha"]),
        "selected_alpha_candidate_hash": selected_alpha_candidate_hash,
        "selected_config_hash": selected_hash,
        "source_behavior": str(source_behavior),
        "support_objective_is_proof_metric": False,
        "support_split_counts": dict(gradient_info["support_split_counts"]),
        "target_behavior": str(target_behavior),
        "train_pool_file_sha256": str(train_pool_file_sha256),
        "train_pool_summary_hash": str(train_pool_summary_hash),
    }
    metadata["compatible_nullspace_projection_provenance_hash"] = stable_hash_json(metadata)
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=selected_delta,
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata=metadata,
    )


def evaluate_v25_projected_support_optimizer_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    config: Mapping[str, Any],
    norm_cap: float,
    selected_config_hash: str,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
) -> dict[str, Any]:
    require_sha256_hex(train_pool_file_sha256, field_name="train_pool_file_sha256")
    require_sha256_hex(train_pool_summary_hash, field_name="train_pool_summary_hash")
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    source_weights = record_weights_tensor(subject)
    optimizer_config = {
        **dict(config),
        "trust_norm_cap": float(config.get("trust_norm_cap", norm_cap)),
    }
    gradient_info = v28_anchor_gradients_and_compatible_jacobian(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    sparse_top_k = int(optimizer_config.get("sparse_top_k", V32_SPARSE_TOP_K))
    compatible_floor = float(
        optimizer_config.get("compatible_floor", V32_COMPATIBLE_FLOOR)
    )
    sign_conflict_penalty = float(
        optimizer_config.get("sign_conflict_penalty", V32_SIGN_CONFLICT_PENALTY)
    )
    selected_coordinates = select_v31_sign_coherent_sparse_coordinates(
        g_target=gradient_info["g_target"],
        g_conflict=gradient_info["g_conflict"],
        g_compatible=gradient_info["g_compatible"],
        sparse_top_k=sparse_top_k,
        compatible_floor=compatible_floor,
        conflict_weight=V29_CONFLICT_WEIGHT,
        sign_conflict_penalty=sign_conflict_penalty,
    )
    if not selected_coordinates:
        raise ValueError("projected support optimizer selected no coordinates")
    coordinate_hash = stable_hash_json(selected_coordinates)
    optimizer_edit = solve_v37_projected_support_optimizer_edit(
        compatible_gradient=gradient_info["g_compatible"],
        compatible_jacobian=gradient_info["compatible_jacobian"],
        coordinate_hash=coordinate_hash,
        selected_coordinates=selected_coordinates,
        config=optimizer_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        record_id_hash=record_id_hash,
        selected_config_hash=selected_hash,
    )
    optimizer_audit = dict(optimizer_edit["audit"])
    optimizer_audit_hash = require_sha256_hex(
        optimizer_audit["optimizer_audit_hash"],
        field_name="optimizer_audit_hash",
    )

    support = v27_support_tensors_for_source_target(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    tournament_tensors = v32_support_behavior_margin_tensors()
    alpha_candidates = [
        float(value)
        for value in optimizer_config.get("alpha_candidates", V35_ALPHA_CANDIDATES)
    ]
    alpha_candidates_hash = stable_hash_json(alpha_candidates)
    alpha_tournament_floor = float(
        optimizer_config.get("alpha_tournament_margin_floor", 0.0)
    )
    alpha_target_floor = float(
        optimizer_config.get("alpha_target_margin_floor", 0.05)
    )
    candidates: list[dict[str, Any]] = []
    base_delta = optimizer_edit["delta"].detach().clone().to(dtype=torch.float32).reshape(-1)
    with torch.no_grad():
        for alpha in alpha_candidates:
            candidate_delta = base_delta * float(alpha)
            edited = source_weights.to(dtype=torch.float32).reshape(-1) + candidate_delta
            compatible_logits = v27_subject_logits_for_inputs(
                edited,
                support["compatible_inputs"],
            )
            support_compatible_mse = F.mse_loss(
                compatible_logits,
                support["compatible_source_logits"].to(dtype=torch.float32),
            )
            tournament = v32_support_tournament_margin_loss(
                margins=v32_support_behavior_margins(
                    weights=edited,
                    tournament_tensors=tournament_tensors,
                ),
                target_behavior=target_behavior,
                tournament_margin_floor=max(0.0, alpha_tournament_floor),
            )
            candidates.append({
                "alpha": float(alpha),
                "support_compatible_mse": float(support_compatible_mse.item()),
                "support_runner_margin": float(tournament["support_runner_margin"].item()),
                "support_target_margin": float(tournament["support_target_margin"].item()),
                "support_tournament_margin": float(
                    tournament["support_tournament_margin"].item()
                ),
            })
    selected_alpha = select_v35_support_source_alpha_candidate(
        candidates=candidates,
        alpha_target_margin_floor=alpha_target_floor,
        alpha_tournament_margin_floor=alpha_tournament_floor,
        fallback_target_penalty=float(optimizer_config.get("fallback_target_penalty", 10.0)),
        fallback_tournament_penalty=float(
            optimizer_config.get("fallback_tournament_penalty", 5.0)
        ),
    )
    selected_delta = base_delta * float(selected_alpha["alpha"])
    selected_alpha_candidate_hash = stable_hash_json(selected_alpha)
    alpha_selection = {
        key: selected_alpha[key]
        for key in [
            "alpha",
            "alpha_candidate_count",
            "candidate_index",
            "candidate_metrics_hash",
            "eligible",
            "eligible_count",
            "fallback_score",
            "selection_mode",
            "support_compatible_mse",
            "support_runner_margin",
            "support_target_margin",
            "support_tournament_margin",
        ]
        if key in selected_alpha
    }
    if progress_log_path is not None:
        extra = redact_v35_support_source_alpha_progress_event({
            **selected_alpha,
            **({"record_id_hash": record_id_hash} if record_id_hash else {}),
            "alpha_candidates_hash": alpha_candidates_hash,
            "delta_norm": float(torch.linalg.norm(selected_delta).item()),
            "selected_alpha_candidate_hash": selected_alpha_candidate_hash,
            "selected_config_hash": selected_hash,
        })
        record_progress_event(
            progress_log_path,
            event="v35_support_source_alpha_selected",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=extra,
        )
    metadata = {
        "alpha_candidate_count": len(candidates),
        "alpha_candidates_hash": alpha_candidates_hash,
        "alpha_selection": alpha_selection,
        "candidate_metrics_hash": selected_alpha["candidate_metrics_hash"],
        "eligible_count": int(selected_alpha["eligible_count"]),
        "matched_edit_source": V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE,
        "matched_spectral_projection_norm": 0.0,
        "optimization_boundary": v37_projected_optimizer_optimization_boundary(),
        "optimization_split": "support",
        "optimizer_audit": optimizer_audit,
        "optimizer_audit_hash": optimizer_audit_hash,
        "proof_split": "heldout",
        "projected_optimizer_provenance_hash": "",
        "script_sha256": str(script_sha256),
        "selected_alpha": float(selected_alpha["alpha"]),
        "selected_alpha_candidate_hash": selected_alpha_candidate_hash,
        "selected_config_hash": selected_hash,
        "source_behavior": str(source_behavior),
        "support_objective_is_proof_metric": False,
        "support_split_counts": dict(gradient_info["support_split_counts"]),
        "target_behavior": str(target_behavior),
        "train_pool_file_sha256": str(train_pool_file_sha256),
        "train_pool_summary_hash": str(train_pool_summary_hash),
    }
    metadata["projected_optimizer_provenance_hash"] = stable_hash_json(metadata)
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=selected_delta,
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata=metadata,
    )


def evaluate_v25_compatible_gated_projected_optimizer_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    config: Mapping[str, Any],
    norm_cap: float,
    selected_config_hash: str,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
) -> dict[str, Any]:
    require_sha256_hex(train_pool_file_sha256, field_name="train_pool_file_sha256")
    require_sha256_hex(train_pool_summary_hash, field_name="train_pool_summary_hash")
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    source_weights = record_weights_tensor(subject)
    optimizer_config = {
        **dict(config),
        "trust_norm_cap": float(config.get("trust_norm_cap", norm_cap)),
    }
    gradient_info = v28_anchor_gradients_and_compatible_jacobian(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    sparse_top_k = int(optimizer_config.get("sparse_top_k", V32_SPARSE_TOP_K))
    compatible_floor = float(
        optimizer_config.get("compatible_floor", V32_COMPATIBLE_FLOOR)
    )
    sign_conflict_penalty = float(
        optimizer_config.get("sign_conflict_penalty", V32_SIGN_CONFLICT_PENALTY)
    )
    selected_coordinates = select_v31_sign_coherent_sparse_coordinates(
        g_target=gradient_info["g_target"],
        g_conflict=gradient_info["g_conflict"],
        g_compatible=gradient_info["g_compatible"],
        sparse_top_k=sparse_top_k,
        compatible_floor=compatible_floor,
        conflict_weight=V29_CONFLICT_WEIGHT,
        sign_conflict_penalty=sign_conflict_penalty,
    )
    if not selected_coordinates:
        raise ValueError("compatible gated projected optimizer selected no coordinates")
    coordinate_hash = stable_hash_json(selected_coordinates)
    optimizer_edit = solve_v37_projected_support_optimizer_edit(
        compatible_gradient=gradient_info["g_compatible"],
        compatible_jacobian=gradient_info["compatible_jacobian"],
        coordinate_hash=coordinate_hash,
        selected_coordinates=selected_coordinates,
        config=optimizer_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        record_id_hash=record_id_hash,
        selected_config_hash=selected_hash,
    )
    optimizer_audit = dict(optimizer_edit["audit"])
    optimizer_audit_hash = require_sha256_hex(
        optimizer_audit["optimizer_audit_hash"],
        field_name="optimizer_audit_hash",
    )

    support = v27_support_tensors_for_source_target(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    tournament_tensors = v32_support_behavior_margin_tensors()
    alpha_candidates = [
        float(value)
        for value in optimizer_config.get("alpha_candidates", V35_ALPHA_CANDIDATES)
    ]
    alpha_candidates_hash = stable_hash_json(alpha_candidates)
    alpha_tournament_floor = float(
        optimizer_config.get("alpha_tournament_margin_floor", 0.0)
    )
    alpha_target_floor = float(
        optimizer_config.get("alpha_target_margin_floor", 0.05)
    )
    alpha_compatible_mse_gate = float(
        optimizer_config.get("alpha_compatible_mse_gate", float("inf"))
    )
    fallback_compatible_penalty = float(
        optimizer_config.get("fallback_compatible_penalty", 2.0)
    )
    candidates: list[dict[str, Any]] = []
    base_delta = optimizer_edit["delta"].detach().clone().to(dtype=torch.float32).reshape(-1)
    with torch.no_grad():
        for alpha in alpha_candidates:
            candidate_delta = base_delta * float(alpha)
            edited = source_weights.to(dtype=torch.float32).reshape(-1) + candidate_delta
            compatible_logits = v27_subject_logits_for_inputs(
                edited,
                support["compatible_inputs"],
            )
            support_compatible_mse = F.mse_loss(
                compatible_logits,
                support["compatible_source_logits"].to(dtype=torch.float32),
            )
            tournament = v32_support_tournament_margin_loss(
                margins=v32_support_behavior_margins(
                    weights=edited,
                    tournament_tensors=tournament_tensors,
                ),
                target_behavior=target_behavior,
                tournament_margin_floor=max(0.0, alpha_tournament_floor),
            )
            candidates.append({
                "alpha": float(alpha),
                "support_compatible_mse": float(support_compatible_mse.item()),
                "support_runner_margin": float(tournament["support_runner_margin"].item()),
                "support_target_margin": float(tournament["support_target_margin"].item()),
                "support_tournament_margin": float(
                    tournament["support_tournament_margin"].item()
                ),
            })
    selected_alpha = select_v38_compatible_gated_alpha_candidate(
        candidates=candidates,
        alpha_target_margin_floor=alpha_target_floor,
        alpha_tournament_margin_floor=alpha_tournament_floor,
        alpha_compatible_mse_gate=alpha_compatible_mse_gate,
        fallback_target_penalty=float(optimizer_config.get("fallback_target_penalty", 10.0)),
        fallback_tournament_penalty=float(
            optimizer_config.get("fallback_tournament_penalty", 5.0)
        ),
        fallback_compatible_penalty=fallback_compatible_penalty,
    )
    selected_delta = base_delta * float(selected_alpha["alpha"])
    selected_alpha_candidate_hash = stable_hash_json(selected_alpha)
    alpha_selection = {
        key: selected_alpha[key]
        for key in [
            "alpha",
            "alpha_candidate_count",
            "alpha_compatible_mse_gate",
            "candidate_index",
            "candidate_metrics_hash",
            "compatible_gate_pass",
            "eligible",
            "eligible_count",
            "fallback_compatible_penalty",
            "fallback_score",
            "selection_mode",
            "support_compatible_mse",
            "support_runner_margin",
            "support_target_margin",
            "support_tournament_margin",
        ]
        if key in selected_alpha
    }
    if progress_log_path is not None:
        extra = redact_v38_compatible_gated_alpha_progress_event({
            **selected_alpha,
            **({"record_id_hash": record_id_hash} if record_id_hash else {}),
            "alpha_candidates_hash": alpha_candidates_hash,
            "delta_norm": float(torch.linalg.norm(selected_delta).item()),
            "selected_alpha_candidate_hash": selected_alpha_candidate_hash,
            "selected_config_hash": selected_hash,
        })
        record_progress_event(
            progress_log_path,
            event="v38_compatible_gated_alpha_selected",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=extra,
        )
    metadata = {
        "alpha_candidate_count": len(candidates),
        "alpha_candidates_hash": alpha_candidates_hash,
        "alpha_compatible_mse_gate": alpha_compatible_mse_gate,
        "alpha_selection": alpha_selection,
        "candidate_metrics_hash": selected_alpha["candidate_metrics_hash"],
        "compatible_gate_weight": float(optimizer_config.get("compatible_gate_weight", 0.0)),
        "compatible_mse_gate": float(optimizer_config.get("compatible_mse_gate", float("inf"))),
        "eligible_count": int(selected_alpha["eligible_count"]),
        "fallback_compatible_penalty": fallback_compatible_penalty,
        "matched_edit_source": V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE,
        "matched_spectral_projection_norm": 0.0,
        "optimization_boundary": v37_projected_optimizer_optimization_boundary(),
        "optimization_split": "support",
        "optimizer_audit": optimizer_audit,
        "optimizer_audit_hash": optimizer_audit_hash,
        "proof_split": "heldout",
        "script_sha256": str(script_sha256),
        "selected_alpha": float(selected_alpha["alpha"]),
        "selected_alpha_candidate_hash": selected_alpha_candidate_hash,
        "selected_config_hash": selected_hash,
        "source_behavior": str(source_behavior),
        "support_objective_is_proof_metric": False,
        "support_split_counts": dict(gradient_info["support_split_counts"]),
        "target_behavior": str(target_behavior),
        "train_pool_file_sha256": str(train_pool_file_sha256),
        "train_pool_summary_hash": str(train_pool_summary_hash),
    }
    metadata["compatible_gated_projected_optimizer_provenance_hash"] = stable_hash_json(
        metadata
    )
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=selected_delta,
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata=metadata,
    )


def evaluate_v25_target_feasible_lexicographic_projected_optimizer_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    config: Mapping[str, Any],
    norm_cap: float,
    selected_config_hash: str,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
) -> dict[str, Any]:
    require_sha256_hex(train_pool_file_sha256, field_name="train_pool_file_sha256")
    require_sha256_hex(train_pool_summary_hash, field_name="train_pool_summary_hash")
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    source_weights = record_weights_tensor(subject)
    optimizer_config = {
        **dict(config),
        "trust_norm_cap": float(config.get("trust_norm_cap", norm_cap)),
    }
    gradient_info = v28_anchor_gradients_and_compatible_jacobian(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    sparse_top_k = int(optimizer_config.get("sparse_top_k", V32_SPARSE_TOP_K))
    compatible_floor = float(
        optimizer_config.get("compatible_floor", V32_COMPATIBLE_FLOOR)
    )
    sign_conflict_penalty = float(
        optimizer_config.get("sign_conflict_penalty", V32_SIGN_CONFLICT_PENALTY)
    )
    selected_coordinates = select_v31_sign_coherent_sparse_coordinates(
        g_target=gradient_info["g_target"],
        g_conflict=gradient_info["g_conflict"],
        g_compatible=gradient_info["g_compatible"],
        sparse_top_k=sparse_top_k,
        compatible_floor=compatible_floor,
        conflict_weight=V29_CONFLICT_WEIGHT,
        sign_conflict_penalty=sign_conflict_penalty,
    )
    if not selected_coordinates:
        raise ValueError("target feasible projected optimizer selected no coordinates")
    coordinate_hash = stable_hash_json(selected_coordinates)
    optimizer_edit = solve_v37_projected_support_optimizer_edit(
        compatible_gradient=gradient_info["g_compatible"],
        compatible_jacobian=gradient_info["compatible_jacobian"],
        coordinate_hash=coordinate_hash,
        selected_coordinates=selected_coordinates,
        config=optimizer_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        record_id_hash=record_id_hash,
        selected_config_hash=selected_hash,
    )
    optimizer_audit = dict(optimizer_edit["audit"])
    optimizer_audit_hash = require_sha256_hex(
        optimizer_audit["optimizer_audit_hash"],
        field_name="optimizer_audit_hash",
    )

    support = v27_support_tensors_for_source_target(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    tournament_tensors = v32_support_behavior_margin_tensors()
    alpha_candidates = [
        float(value)
        for value in optimizer_config.get("alpha_candidates", V35_ALPHA_CANDIDATES)
    ]
    alpha_candidates_hash = stable_hash_json(alpha_candidates)
    alpha_tournament_floor = float(
        optimizer_config.get("alpha_tournament_margin_floor", 0.0)
    )
    alpha_target_floor = float(
        optimizer_config.get("alpha_target_margin_floor", 0.05)
    )
    alpha_compatible_mse_soft_gate = float(
        optimizer_config.get("alpha_compatible_mse_soft_gate", float("inf"))
    )
    candidates: list[dict[str, Any]] = []
    base_delta = optimizer_edit["delta"].detach().clone().to(dtype=torch.float32).reshape(-1)
    with torch.no_grad():
        for alpha in alpha_candidates:
            candidate_delta = base_delta * float(alpha)
            edited = source_weights.to(dtype=torch.float32).reshape(-1) + candidate_delta
            compatible_logits = v27_subject_logits_for_inputs(
                edited,
                support["compatible_inputs"],
            )
            support_compatible_mse = F.mse_loss(
                compatible_logits,
                support["compatible_source_logits"].to(dtype=torch.float32),
            )
            tournament = v32_support_tournament_margin_loss(
                margins=v32_support_behavior_margins(
                    weights=edited,
                    tournament_tensors=tournament_tensors,
                ),
                target_behavior=target_behavior,
                tournament_margin_floor=max(0.0, alpha_tournament_floor),
            )
            candidates.append({
                "alpha": float(alpha),
                "support_compatible_mse": float(support_compatible_mse.item()),
                "support_runner_margin": float(tournament["support_runner_margin"].item()),
                "support_target_margin": float(tournament["support_target_margin"].item()),
                "support_tournament_margin": float(
                    tournament["support_tournament_margin"].item()
                ),
            })
    selected_alpha = select_v39_target_feasible_lexicographic_alpha_candidate(
        candidates=candidates,
        alpha_target_margin_floor=alpha_target_floor,
        alpha_tournament_margin_floor=alpha_tournament_floor,
        alpha_compatible_mse_soft_gate=alpha_compatible_mse_soft_gate,
    )
    selected_delta = base_delta * float(selected_alpha["alpha"])
    selected_alpha_candidate_hash = stable_hash_json(selected_alpha)
    alpha_selection = {
        key: selected_alpha[key]
        for key in [
            "alpha",
            "alpha_candidate_count",
            "alpha_compatible_mse_soft_gate",
            "candidate_index",
            "candidate_metrics_hash",
            "compatible_gap",
            "eligible_count",
            "selection_mode",
            "support_compatible_mse",
            "support_runner_margin",
            "support_target_margin",
            "support_tournament_margin",
            "target_feasible",
            "target_gap",
            "target_rank_score",
            "tournament_gap",
        ]
        if key in selected_alpha
    }
    if progress_log_path is not None:
        extra = redact_v39_target_feasible_alpha_progress_event({
            **selected_alpha,
            **({"record_id_hash": record_id_hash} if record_id_hash else {}),
            "alpha_candidates_hash": alpha_candidates_hash,
            "delta_norm": float(torch.linalg.norm(selected_delta).item()),
            "selected_alpha_candidate_hash": selected_alpha_candidate_hash,
            "selected_config_hash": selected_hash,
        })
        record_progress_event(
            progress_log_path,
            event="v39_target_feasible_lexicographic_alpha_selected",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=extra,
        )
    metadata = {
        "alpha_candidate_count": len(candidates),
        "alpha_candidates_hash": alpha_candidates_hash,
        "alpha_compatible_mse_soft_gate": alpha_compatible_mse_soft_gate,
        "alpha_selection": alpha_selection,
        "candidate_metrics_hash": selected_alpha["candidate_metrics_hash"],
        "compatible_gate_weight": float(optimizer_config.get("compatible_gate_weight", 0.0)),
        "compatible_mse_gate": float(optimizer_config.get("compatible_mse_gate", float("inf"))),
        "eligible_count": int(selected_alpha["eligible_count"]),
        "matched_edit_source": V39_TARGET_FEASIBLE_LEXICOGRAPHIC_MATCHED_EDIT_SOURCE,
        "matched_spectral_projection_norm": 0.0,
        "optimization_boundary": v37_projected_optimizer_optimization_boundary(),
        "optimization_split": "support",
        "optimizer_audit": optimizer_audit,
        "optimizer_audit_hash": optimizer_audit_hash,
        "proof_split": "heldout",
        "script_sha256": str(script_sha256),
        "selected_alpha": float(selected_alpha["alpha"]),
        "selected_alpha_candidate_hash": selected_alpha_candidate_hash,
        "selected_config_hash": selected_hash,
        "source_behavior": str(source_behavior),
        "support_objective_is_proof_metric": False,
        "support_split_counts": dict(gradient_info["support_split_counts"]),
        "target_behavior": str(target_behavior),
        "train_pool_file_sha256": str(train_pool_file_sha256),
        "train_pool_summary_hash": str(train_pool_summary_hash),
    }
    metadata["target_feasible_lexicographic_optimizer_provenance_hash"] = (
        stable_hash_json(metadata)
    )
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=selected_delta,
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata=metadata,
    )


def evaluate_v25_target_tolerance_locality_budget_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    config: Mapping[str, Any],
    norm_cap: float,
    selected_config_hash: str,
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    script_sha256: str,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
    record_id_hash: str | None = None,
) -> dict[str, Any]:
    require_sha256_hex(train_pool_file_sha256, field_name="train_pool_file_sha256")
    require_sha256_hex(train_pool_summary_hash, field_name="train_pool_summary_hash")
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    source_weights = record_weights_tensor(subject)
    optimizer_config = {
        **dict(config),
        "trust_norm_cap": float(config.get("trust_norm_cap", norm_cap)),
    }
    gradient_info = v28_anchor_gradients_and_compatible_jacobian(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    selected_coordinates = select_v31_sign_coherent_sparse_coordinates(
        g_target=gradient_info["g_target"],
        g_conflict=gradient_info["g_conflict"],
        g_compatible=gradient_info["g_compatible"],
        sparse_top_k=int(optimizer_config.get("sparse_top_k", V32_SPARSE_TOP_K)),
        compatible_floor=float(
            optimizer_config.get("compatible_floor", V32_COMPATIBLE_FLOOR)
        ),
        conflict_weight=V29_CONFLICT_WEIGHT,
        sign_conflict_penalty=float(
            optimizer_config.get("sign_conflict_penalty", V32_SIGN_CONFLICT_PENALTY)
        ),
    )
    if not selected_coordinates:
        raise ValueError("target tolerance optimizer selected no coordinates")
    optimizer_edit = solve_v37_projected_support_optimizer_edit(
        compatible_gradient=gradient_info["g_compatible"],
        compatible_jacobian=gradient_info["compatible_jacobian"],
        coordinate_hash=stable_hash_json(selected_coordinates),
        selected_coordinates=selected_coordinates,
        config=optimizer_config,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        record_id_hash=record_id_hash,
        selected_config_hash=selected_hash,
    )
    optimizer_audit = dict(optimizer_edit["audit"])
    optimizer_audit_hash = require_sha256_hex(
        optimizer_audit["optimizer_audit_hash"],
        field_name="optimizer_audit_hash",
    )

    support = v27_support_tensors_for_source_target(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    tournament_tensors = v32_support_behavior_margin_tensors()
    alpha_candidates = [
        float(value)
        for value in optimizer_config.get("alpha_candidates", V35_ALPHA_CANDIDATES)
    ]
    alpha_candidates_hash = stable_hash_json(alpha_candidates)
    alpha_tournament_floor = float(
        optimizer_config.get("alpha_tournament_margin_floor", 0.0)
    )
    alpha_target_floor = float(
        optimizer_config.get("alpha_target_margin_floor", 0.05)
    )
    alpha_compatible_mse_soft_gate = float(
        optimizer_config.get("alpha_compatible_mse_soft_gate", float("inf"))
    )
    target_rank_score_tolerance = float(
        optimizer_config.get("target_rank_score_tolerance", 0.0)
    )
    candidates: list[dict[str, Any]] = []
    base_delta = optimizer_edit["delta"].detach().clone().to(dtype=torch.float32).reshape(-1)
    with torch.no_grad():
        for alpha in alpha_candidates:
            candidate_delta = base_delta * float(alpha)
            edited = source_weights.to(dtype=torch.float32).reshape(-1) + candidate_delta
            compatible_logits = v27_subject_logits_for_inputs(
                edited,
                support["compatible_inputs"],
            )
            support_compatible_mse = F.mse_loss(
                compatible_logits,
                support["compatible_source_logits"].to(dtype=torch.float32),
            )
            tournament = v32_support_tournament_margin_loss(
                margins=v32_support_behavior_margins(
                    weights=edited,
                    tournament_tensors=tournament_tensors,
                ),
                target_behavior=target_behavior,
                tournament_margin_floor=max(0.0, alpha_tournament_floor),
            )
            candidates.append({
                "alpha": float(alpha),
                "support_compatible_mse": float(support_compatible_mse.item()),
                "support_runner_margin": float(tournament["support_runner_margin"].item()),
                "support_target_margin": float(tournament["support_target_margin"].item()),
                "support_tournament_margin": float(
                    tournament["support_tournament_margin"].item()
                ),
            })
    selected_alpha = select_v40_target_tolerance_locality_budget_alpha_candidate(
        candidates=candidates,
        alpha_target_margin_floor=alpha_target_floor,
        alpha_tournament_margin_floor=alpha_tournament_floor,
        alpha_compatible_mse_soft_gate=alpha_compatible_mse_soft_gate,
        target_rank_score_tolerance=target_rank_score_tolerance,
    )
    selected_delta = base_delta * float(selected_alpha["alpha"])
    selected_alpha_candidate_hash = stable_hash_json(selected_alpha)
    alpha_selection = {
        key: selected_alpha[key]
        for key in [
            "alpha",
            "alpha_candidate_count",
            "alpha_compatible_mse_soft_gate",
            "best_target_rank_score",
            "candidate_index",
            "candidate_metrics_hash",
            "compatible_gap",
            "eligible_count",
            "selection_mode",
            "support_compatible_mse",
            "support_runner_margin",
            "support_target_margin",
            "support_tournament_margin",
            "target_feasible",
            "target_gap",
            "target_rank_score",
            "target_rank_score_tolerance",
            "tournament_gap",
            "within_target_tolerance_count",
        ]
        if key in selected_alpha
    }
    if progress_log_path is not None:
        extra = redact_v40_target_tolerance_alpha_progress_event({
            **selected_alpha,
            **({"record_id_hash": record_id_hash} if record_id_hash else {}),
            "alpha_candidates_hash": alpha_candidates_hash,
            "delta_norm": float(torch.linalg.norm(selected_delta).item()),
            "selected_alpha_candidate_hash": selected_alpha_candidate_hash,
            "selected_config_hash": selected_hash,
        })
        record_progress_event(
            progress_log_path,
            event="v40_target_tolerance_locality_budget_alpha_selected",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=extra,
        )
    metadata = {
        "alpha_candidate_count": len(candidates),
        "alpha_candidates_hash": alpha_candidates_hash,
        "alpha_compatible_mse_soft_gate": alpha_compatible_mse_soft_gate,
        "alpha_selection": alpha_selection,
        "candidate_metrics_hash": selected_alpha["candidate_metrics_hash"],
        "compatible_gate_weight": float(optimizer_config.get("compatible_gate_weight", 0.0)),
        "compatible_mse_gate": float(optimizer_config.get("compatible_mse_gate", float("inf"))),
        "eligible_count": int(selected_alpha["eligible_count"]),
        "experiment_variant": experiment_variant_for_config(optimizer_config),
        "matched_edit_source": str(
            optimizer_config.get(
                "matched_edit_source",
                V40_TARGET_TOLERANCE_LOCALITY_BUDGET_MATCHED_EDIT_SOURCE,
            )
        ),
        "matched_spectral_projection_norm": 0.0,
        "optimization_boundary": v37_projected_optimizer_optimization_boundary(),
        "optimization_split": "support",
        "optimizer_audit": optimizer_audit,
        "optimizer_audit_hash": optimizer_audit_hash,
        "proof_split": "heldout",
        "script_sha256": str(script_sha256),
        "selected_alpha": float(selected_alpha["alpha"]),
        "selected_alpha_candidate_hash": selected_alpha_candidate_hash,
        "selected_config_hash": selected_hash,
        "source_behavior": str(source_behavior),
        "support_objective_is_proof_metric": False,
        "support_split_counts": dict(gradient_info["support_split_counts"]),
        "target_behavior": str(target_behavior),
        "target_rank_score_tolerance": target_rank_score_tolerance,
        "train_pool_file_sha256": str(train_pool_file_sha256),
        "train_pool_summary_hash": str(train_pool_summary_hash),
        "within_target_tolerance_count": int(
            selected_alpha["within_target_tolerance_count"]
        ),
    }
    metadata["target_tolerance_locality_budget_provenance_hash"] = (
        stable_hash_json(metadata)
    )
    if metadata["matched_edit_source"] == V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE:
        metadata["compatible_dual_frontier_provenance_hash"] = stable_hash_json(
            metadata
        )
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=selected_delta,
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata=metadata,
    )


def evaluate_v25_trajectory_frontier_matched_edit(**kwargs: Any) -> dict[str, Any]:
    return evaluate_v25_target_tolerance_locality_budget_matched_edit(**kwargs)


def evaluate_v25_compatible_dual_frontier_matched_edit(
    **kwargs: Any,
) -> dict[str, Any]:
    return evaluate_v25_target_tolerance_locality_budget_matched_edit(**kwargs)


def control_record_for_delta(
    *,
    control_type: str,
    delta: torch.Tensor,
    source_behavior: str,
    source_weights: torch.Tensor,
    target_behavior: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    delta = delta.to(dtype=torch.float32).reshape(-1)
    edited = source_weights.to(dtype=torch.float32).reshape(-1) + delta
    metrics = v23.v16.v15.v14.functional_metrics(
        edited,
        str(source_behavior),
        str(target_behavior),
        source_weights.to(dtype=torch.float32).reshape(-1),
    )
    payload = {
        **metrics,
        "control_type": str(control_type),
        "delta_norm": float(torch.linalg.norm(delta).item()),
        "editor": {
            **dict(metadata or {}),
            "delta_sha256": stable_hash_json(tensor_to_hashable(delta)),
            "edited_vector_sha256": stable_hash_json(tensor_to_hashable(edited)),
        },
    }
    finite_failures = recursive_numeric_finiteness_failures(payload)
    if finite_failures:
        raise ValueError(
            f"nonfinite control {control_type}: " + ", ".join(finite_failures[:5])
        )
    return payload


def empirical_task_vector_entry_for_direction(
    bank: Mapping[str, Any],
    *,
    source_behavior: str,
    target_behavior: str,
) -> Mapping[str, Any]:
    matches = [
        entry for entry in bank.get("entries", [])
        if str(entry["source_behavior"]) == str(source_behavior)
        and str(entry["target_behavior"]) == str(target_behavior)
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one empirical task vector for "
            f"{source_behavior}->{target_behavior}, found {len(matches)}"
        )
    return matches[0]


def evaluate_v25_empirical_task_vector_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    empirical_task_vector_bank: Mapping[str, Any],
    selected_config_hash: str,
    spectral_basis: torch.Tensor | None,
) -> dict[str, Any]:
    bank_hash = require_sha256_hex(
        empirical_task_vector_bank.get("bank_hash"),
        field_name="empirical_task_vector_bank.bank_hash",
    )
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    entry = empirical_task_vector_entry_for_direction(
        empirical_task_vector_bank,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    delta = torch.as_tensor(entry["delta"], dtype=torch.float32).reshape(-1)
    if int(delta.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("empirical matched delta has wrong dimension")
    if not torch.isfinite(delta).all():
        raise ValueError("empirical matched delta is nonfinite")
    matched_spectral_projection_norm = 0.0
    if spectral_basis is not None:
        matched_spectral_projection_norm = float(
            torch.linalg.norm(project_to_basis(delta, spectral_basis)).item()
        )
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=delta,
        source_behavior=source_behavior,
        source_weights=record_weights_tensor(subject),
        target_behavior=target_behavior,
        metadata={
            **dict(entry.get("editor_audit", {})),
            "empirical_task_vector_bank_hash": bank_hash,
            "matched_edit_source": "empirical_centroid_task_vector",
            "matched_spectral_projection_norm": matched_spectral_projection_norm,
            "selected_config_hash": selected_hash,
        },
    )


def evaluate_v25_descriptor_control(
    *,
    control_type: str,
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    source_descriptor: torch.Tensor,
    target_descriptor: torch.Tensor,
    cache_entry: Mapping[str, Any],
    config: Mapping[str, Any],
    norm_cap: float,
    spectral_basis: torch.Tensor | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    edit = solve_projected_jacobian_edit(
        source_descriptor=source_descriptor,
        target_descriptor=target_descriptor,
        activation_jacobian=cache_entry["activation_jacobian"],
        source_logit_jacobian=cache_entry["source_logit_jacobian"],
        config=config,
        norm_cap=float(norm_cap),
        spectral_basis=spectral_basis,
    )
    return control_record_for_delta(
        control_type=control_type,
        delta=edit["delta"],
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata={**edit["audit"], **dict(metadata or {})},
    )


def build_v25_random_matched_norm_controls(
    *,
    matched_delta: torch.Tensor,
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    record_id: str,
    selected_config_hash: str,
    projection: str,
    spectral_basis: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    matched_delta = matched_delta.to(dtype=torch.float32).reshape(-1)
    matched_norm = float(torch.linalg.norm(matched_delta).item())
    controls = []
    for index, control_type in enumerate(expected_v25_random_control_types()):
        seed_hash = stable_hash_json({
            "index": int(index),
            "record_id": str(record_id),
            "scope": "v25_random_control",
            "selected_config_hash": str(selected_config_hash),
        })
        seed = int(seed_hash[:16], 16) % (2**31)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        raw = torch.randn(SOURCE_WEIGHT_DIM, dtype=torch.float32, generator=generator)
        projected = project_delta_for_config(
            raw,
            projection=str(projection),
            spectral_basis=spectral_basis,
        )
        projected_norm = float(torch.linalg.norm(projected).item())
        if matched_norm <= 1e-12 or projected_norm <= 1e-12:
            final_delta = torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.float32)
            zero_norm_fallback = True
        else:
            final_delta = projected / projected_norm * matched_norm
            zero_norm_fallback = False
        controls.append(control_record_for_delta(
            control_type=control_type,
            delta=final_delta,
            source_behavior=source_behavior,
            source_weights=source_weights,
            target_behavior=target_behavior,
            metadata={
                "index": int(index),
                "matched_delta_norm": matched_norm,
                "projection": str(projection),
                "raw_delta_sha256": stable_hash_json(tensor_to_hashable(raw)),
                "seed_hash": seed_hash,
                "selected_config_hash": str(selected_config_hash),
                "zero_norm_fallback": bool(zero_norm_fallback),
            },
        ))
    return controls


def required_v25_precomputed_delta_control_types() -> list[str]:
    return [
        "v21_baseline",
        "v22_baseline",
        "v23_baseline",
        "nearest_train_delta",
        "teacher_oracle_delta",
        "contrastive_weight_arithmetic",
    ]


def build_v25_control_context(
    *,
    shuffled_target_descriptor: torch.Tensor,
    precomputed_delta_by_control_type: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    shuffled_descriptor = torch.as_tensor(
        shuffled_target_descriptor,
        dtype=torch.float32,
    ).reshape(-1)
    if int(shuffled_descriptor.numel()) != ACTIVATION_DESCRIPTOR_DIM:
        raise ValueError("shuffled target descriptor has wrong dimension")
    if not torch.isfinite(shuffled_descriptor).all():
        raise ValueError("nonfinite shuffled target descriptor")
    precomputed_deltas: dict[str, torch.Tensor] = {}
    delta_hashes: dict[str, str] = {}
    for control_type in required_v25_precomputed_delta_control_types():
        if control_type not in precomputed_delta_by_control_type:
            raise ValueError(f"missing precomputed delta for {control_type}")
        delta = torch.as_tensor(
            precomputed_delta_by_control_type[control_type],
            dtype=torch.float32,
        ).reshape(-1)
        if int(delta.numel()) != SOURCE_WEIGHT_DIM:
            raise ValueError(f"precomputed delta for {control_type} has wrong dimension")
        if not torch.isfinite(delta).all():
            raise ValueError(f"nonfinite precomputed delta for {control_type}")
        precomputed_deltas[control_type] = delta
        delta_hashes[control_type] = stable_hash_json(tensor_to_hashable(delta))
    shuffled_hash = stable_hash_json(tensor_to_hashable(shuffled_descriptor))
    context_hash = stable_hash_json({
        "precomputed_delta_hash_by_control_type": delta_hashes,
        "provenance": dict(provenance),
        "scope": "v25_train_only_control_context",
        "shuffled_target_descriptor_hash": shuffled_hash,
    })
    return {
        "context_hash": context_hash,
        "precomputed_delta_by_control_type": precomputed_deltas,
        "precomputed_delta_hash_by_control_type": delta_hashes,
        "provenance": dict(provenance),
        "shuffled_target_descriptor": shuffled_descriptor,
        "shuffled_target_descriptor_hash": shuffled_hash,
    }


def build_v25_placeholder_control_context_for_dry_run(
    *,
    source_descriptor: torch.Tensor,
    record_id_hash: str,
    job_plan_hash: str,
) -> dict[str, Any]:
    record_hash = require_sha256_hex(record_id_hash, field_name="record_id_hash")
    plan_hash = require_sha256_hex(job_plan_hash, field_name="job_plan_hash")
    return build_v25_control_context(
        shuffled_target_descriptor=torch.as_tensor(source_descriptor, dtype=torch.float32),
        precomputed_delta_by_control_type={
            control_type: torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.float32)
            for control_type in required_v25_precomputed_delta_control_types()
        },
        provenance={
            "control_context_mode": "placeholder_zero_controls",
            "dry_run_only": True,
            "job_plan_hash": plan_hash,
            "proof_valid": False,
            "record_id_hash": record_hash,
        },
    )


def redact_v25_control_context_for_progress(
    control_context: Mapping[str, Any],
) -> dict[str, Any]:
    context_hash = require_sha256_hex(
        control_context.get("context_hash"),
        field_name="control_context.context_hash",
    )
    delta_hashes = dict(control_context.get("precomputed_delta_hash_by_control_type", {}))
    missing = sorted(set(required_v25_precomputed_delta_control_types()) - set(delta_hashes))
    if missing:
        raise ValueError(f"missing precomputed delta hashes for progress: {missing}")
    for control_type, delta_hash in delta_hashes.items():
        require_sha256_hex(
            delta_hash,
            field_name=f"precomputed_delta_hash_by_control_type.{control_type}",
        )
    shuffled_hash = require_sha256_hex(
        control_context.get("shuffled_target_descriptor_hash"),
        field_name="control_context.shuffled_target_descriptor_hash",
    )
    return {
        "context_hash": context_hash,
        "precomputed_delta_hash_by_control_type": {
            control_type: delta_hashes[control_type]
            for control_type in required_v25_precomputed_delta_control_types()
        },
        "provenance": dict(control_context.get("provenance", {})),
        "shuffled_target_descriptor_hash": shuffled_hash,
    }


def build_v25_train_delta_bank(
    *,
    train_subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    config: Mapping[str, Any],
    norm_cap: float,
    script_sha256: str,
    spectral_basis: torch.Tensor | None = None,
    progress_log_path: Path | None = None,
    started_at_monotonic: float | None = None,
    now_monotonic: Any | None = None,
    progress_every_subjects: int = 16,
    train_edit_bank_role: str = "actual",
) -> dict[str, Any]:
    train_statistics_hash = require_sha256_hex(
        train_stats.get("train_statistics_hash"),
        field_name="train_statistics_hash",
    )
    entries: list[dict[str, Any]] = []
    sorted_subjects = sorted(train_subjects, key=lambda item: str(item["subject_id"]))
    if progress_log_path is not None and started_at_monotonic is None:
        raise ValueError("started_at_monotonic is required for train edit bank progress")
    progress_interval = max(1, int(progress_every_subjects))
    for subject_index, subject in enumerate(sorted_subjects, start=1):
        source_behavior = behavior_for_record(subject)
        cache_entry = compute_jacobian_cache_entry(
            subject,
            source_behavior=source_behavior,
            train_stats=train_stats,
            script_sha256=str(script_sha256),
        )
        for target_behavior in sorted(PATTERNS):
            if str(target_behavior) == str(source_behavior):
                continue
            target_descriptor = train_stats["target_activation_descriptor_by_behavior"][
                str(target_behavior)
            ]
            edit = solve_projected_jacobian_edit(
                source_descriptor=cache_entry["source_descriptor"],
                target_descriptor=target_descriptor,
                activation_jacobian=cache_entry["activation_jacobian"],
                source_logit_jacobian=cache_entry["source_logit_jacobian"],
                config=config,
                norm_cap=norm_cap,
                spectral_basis=spectral_basis,
            )
            delta = edit["delta"].to(dtype=torch.float32).reshape(-1)
            if int(delta.numel()) != SOURCE_WEIGHT_DIM:
                raise ValueError("train delta bank entry has wrong dimension")
            if not torch.isfinite(delta).all():
                raise ValueError("nonfinite train delta bank entry")
            delta_hash = stable_hash_json(tensor_to_hashable(delta))
            entries.append({
                "cache_key": str(cache_entry["cache_key"]),
                "delta": delta,
                "delta_norm": float(torch.linalg.norm(delta).item()),
                "delta_sha256": delta_hash,
                "direction": f"{source_behavior}->{target_behavior}",
                "editor_audit": dict(edit["audit"]),
                "source_behavior": str(source_behavior),
                "source_descriptor": cache_entry["source_descriptor"],
                "subject_id_hash": stable_hash_json(str(subject["subject_id"])),
                "target_behavior": str(target_behavior),
            })
        if progress_log_path is not None and (
            subject_index == len(sorted_subjects)
            or subject_index % progress_interval == 0
        ):
            record_progress_event(
                progress_log_path,
                event="train_edit_bank_progress",
                started_at_monotonic=float(started_at_monotonic),
                now_monotonic=now_monotonic,
                extra={
                    "entry_count": len(entries),
                    "processed_train_subject_count": int(subject_index),
                    "train_edit_bank_role": str(train_edit_bank_role),
                    "total_train_subject_count": len(sorted_subjects),
                },
            )
    entry_hashes = [
        stable_hash_json({
            "delta_sha256": entry["delta_sha256"],
            "direction": entry["direction"],
            "subject_id_hash": entry["subject_id_hash"],
        })
        for entry in entries
    ]
    spectral_basis_sha256 = None
    if spectral_basis is not None:
        spectral_basis_sha256 = stable_hash_json(
            tensor_to_hashable(spectral_basis.to(dtype=torch.float32))
        )
    bank_hash = stable_hash_json({
        "config": dict(config),
        "entry_hashes": entry_hashes,
        "norm_cap": float(norm_cap),
        "scope": "v25_train_delta_bank",
        "spectral_basis_sha256": spectral_basis_sha256,
        "train_statistics_hash": train_statistics_hash,
    })
    return {
        "bank_hash": bank_hash,
        "config": dict(config),
        "entries": entries,
        "entry_count": len(entries),
        "entry_hashes": entry_hashes,
        "norm_cap": float(norm_cap),
        "spectral_basis_sha256": spectral_basis_sha256,
    }


def behavior_weight_centroids(
    train_subjects: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[torch.Tensor]] = {}
    for record in train_subjects:
        behavior = behavior_for_record(record)
        grouped.setdefault(behavior, []).append(record_weights_tensor(record))
    centroids: dict[str, dict[str, Any]] = {}
    for behavior, weights in sorted(grouped.items()):
        if not weights:
            raise ValueError(f"no train weights for behavior {behavior}")
        matrix = torch.stack([
            weight.reshape(-1).to(dtype=torch.float32)
            for weight in weights
        ])
        centroid = matrix.mean(dim=0).to(dtype=torch.float32)
        if int(centroid.numel()) != SOURCE_WEIGHT_DIM:
            raise ValueError("behavior centroid has wrong dimension")
        if not torch.isfinite(centroid).all():
            raise ValueError("behavior centroid is nonfinite")
        centroids[behavior] = {"centroid": centroid, "count": len(weights)}
    return centroids


def build_v25_empirical_task_vector_bank(
    *,
    train_subjects: Sequence[Mapping[str, Any]],
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    config: Mapping[str, Any],
    norm_cap: float,
    script_sha256: str,
    spectral_basis: torch.Tensor | None = None,
) -> dict[str, Any]:
    checked_train_pool_file_sha256 = require_sha256_hex(
        train_pool_file_sha256,
        field_name="train_pool_file_sha256",
    )
    checked_train_pool_summary_hash = require_sha256_hex(
        train_pool_summary_hash,
        field_name="train_pool_summary_hash",
    )
    centroids = behavior_weight_centroids(train_subjects)
    centroid_hash_by_behavior = {
        behavior: stable_hash_json(tensor_to_hashable(payload["centroid"]))
        for behavior, payload in centroids.items()
    }
    count_by_behavior = {
        behavior: int(payload["count"])
        for behavior, payload in centroids.items()
    }
    entries: list[dict[str, Any]] = []
    for source_behavior in sorted(centroids):
        for target_behavior in sorted(centroids):
            if source_behavior == target_behavior:
                continue
            raw_delta = (
                centroids[target_behavior]["centroid"]
                - centroids[source_behavior]["centroid"]
            )
            projected_delta = project_delta_for_config(
                raw_delta,
                projection=str(config["projection"]),
                spectral_basis=spectral_basis,
            )
            delta = apply_norm_cap(projected_delta, max_norm=float(norm_cap))
            raw_norm = float(torch.linalg.norm(raw_delta).item())
            projected_norm = float(torch.linalg.norm(projected_delta).item())
            delta_hash = stable_hash_json(tensor_to_hashable(delta))
            entry = {
                "delta": delta,
                "delta_norm": float(torch.linalg.norm(delta).item()),
                "delta_sha256": delta_hash,
                "direction": f"{source_behavior}->{target_behavior}",
                "editor_audit": {
                    "delta_sha256": delta_hash,
                    "edit_source": "empirical_centroid_task_vector",
                    "norm_cap": float(norm_cap),
                    "norm_cap_applied": bool(projected_norm > float(norm_cap) + 1e-8),
                    "projected_delta_norm": projected_norm,
                    "projection": str(config["projection"]),
                    "raw_delta_norm": raw_norm,
                    "script_sha256": str(script_sha256),
                },
                "source_behavior": source_behavior,
                "source_centroid_sha256": centroid_hash_by_behavior[source_behavior],
                "source_count": int(centroids[source_behavior]["count"]),
                "target_behavior": target_behavior,
                "target_centroid_sha256": centroid_hash_by_behavior[target_behavior],
                "target_count": int(centroids[target_behavior]["count"]),
            }
            entries.append(entry)
    entry_hashes = [v25_train_delta_entry_hash(entry) for entry in entries]
    spectral_basis_sha256 = None
    if spectral_basis is not None:
        spectral_basis_sha256 = stable_hash_json(
            tensor_to_hashable(spectral_basis.to(dtype=torch.float32))
        )
    bank_hash = stable_hash_json({
        "centroid_hash_by_behavior": centroid_hash_by_behavior,
        "config": dict(config),
        "count_by_behavior": count_by_behavior,
        "entry_hashes": entry_hashes,
        "norm_cap": float(norm_cap),
        "scope": "v25_empirical_task_vector_bank",
        "script_sha256": str(script_sha256),
        "spectral_basis_sha256": spectral_basis_sha256,
        "train_pool_file_sha256": checked_train_pool_file_sha256,
        "train_pool_summary_hash": checked_train_pool_summary_hash,
    })
    return {
        "bank_hash": bank_hash,
        "centroid_hash_by_behavior": centroid_hash_by_behavior,
        "config": dict(config),
        "count_by_behavior": count_by_behavior,
        "entries": entries,
        "entry_count": len(entries),
        "entry_hashes": entry_hashes,
        "norm_cap": float(norm_cap),
        "spectral_basis_sha256": spectral_basis_sha256,
        "train_pool_file_sha256": checked_train_pool_file_sha256,
        "train_pool_summary_hash": checked_train_pool_summary_hash,
    }


def build_v25_empirical_task_vector_bank_with_progress(
    *,
    train_subjects: Sequence[Mapping[str, Any]],
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    config: Mapping[str, Any],
    norm_cap: float,
    script_sha256: str,
    spectral_basis: torch.Tensor | None = None,
    progress_log_path: Path,
    started_at_monotonic: float,
    now_monotonic: Any | None = None,
    experiment_variant: str = V26_EXPERIMENT_VARIANT,
) -> dict[str, Any]:
    checked_train_pool_file_sha256 = require_sha256_hex(
        train_pool_file_sha256,
        field_name="train_pool_file_sha256",
    )
    checked_train_pool_summary_hash = require_sha256_hex(
        train_pool_summary_hash,
        field_name="train_pool_summary_hash",
    )
    record_progress_event(
        progress_log_path,
        event="empirical_task_vector_bank_start",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "config_hash": stable_hash_json(dict(config)),
            "experiment_variant": str(experiment_variant),
            "norm_cap": float(norm_cap),
            "train_pool_file_sha256": checked_train_pool_file_sha256,
            "train_pool_summary_hash": checked_train_pool_summary_hash,
            "train_subject_count": len(train_subjects),
        },
    )
    bank = build_v25_empirical_task_vector_bank(
        train_subjects=train_subjects,
        train_pool_file_sha256=checked_train_pool_file_sha256,
        train_pool_summary_hash=checked_train_pool_summary_hash,
        config=v25_native_control_config(config),
        norm_cap=norm_cap,
        script_sha256=script_sha256,
        spectral_basis=spectral_basis,
    )
    record_progress_event(
        progress_log_path,
        event="empirical_task_vector_bank_completed",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "bank_hash": str(bank["bank_hash"]),
            "centroid_hashes_hash": stable_hash_json(bank["centroid_hash_by_behavior"]),
            "count_by_behavior": dict(bank["count_by_behavior"]),
            "entry_count": int(bank["entry_count"]),
            "entry_hashes_hash": stable_hash_json(bank["entry_hashes"]),
            "experiment_variant": str(experiment_variant),
            "train_pool_file_sha256": str(bank["train_pool_file_sha256"]),
            "train_pool_summary_hash": str(bank["train_pool_summary_hash"]),
        },
    )
    return bank


def redact_v25_train_delta_bank_summary(bank: Mapping[str, Any]) -> dict[str, Any]:
    direction_counts: dict[str, int] = {}
    delta_hashes = []
    for entry in bank["entries"]:
        direction = str(entry["direction"])
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
        delta_hashes.append(str(entry["delta_sha256"]))
    return {
        "bank_hash": str(bank["bank_hash"]),
        "direction_counts": {
            direction: direction_counts[direction]
            for direction in sorted(direction_counts)
        },
        "entry_count": int(bank["entry_count"]),
        "entry_hashes_hash": stable_hash_json(list(bank["entry_hashes"])),
        "unique_edit_vector_hash_count": len(set(delta_hashes)),
    }


def build_v25_train_delta_bank_with_progress(
    *,
    train_subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    config: Mapping[str, Any],
    norm_cap: float,
    script_sha256: str,
    spectral_basis: torch.Tensor | None = None,
    progress_log_path: Path,
    started_at_monotonic: float,
    now_monotonic: Any | None = None,
    train_edit_bank_role: str = "actual",
) -> dict[str, Any]:
    record_progress_event(
        progress_log_path,
        event="train_edit_bank_start",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "config_hash": stable_hash_json(dict(config)),
            "norm_cap": float(norm_cap),
            "train_edit_bank_role": str(train_edit_bank_role),
            "train_subject_count": len(train_subjects),
        },
    )
    bank = build_v25_train_delta_bank(
        train_subjects=train_subjects,
        train_stats=train_stats,
        config=v25_native_control_config(config),
        norm_cap=norm_cap,
        script_sha256=script_sha256,
        spectral_basis=spectral_basis,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        train_edit_bank_role=train_edit_bank_role,
    )
    record_progress_event(
        progress_log_path,
        event="train_edit_bank_completed",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            **redact_v25_train_delta_bank_summary(bank),
            "train_edit_bank_role": str(train_edit_bank_role),
        },
    )
    return bank


def v25_train_delta_entry_hash(entry: Mapping[str, Any]) -> str:
    return stable_hash_json({
        "delta_sha256": str(entry["delta_sha256"]),
        "direction": str(entry["direction"]),
        "source_behavior": str(entry["source_behavior"]),
        "target_behavior": str(entry["target_behavior"]),
    })


def v25_train_delta_entries_matching(
    entries: Sequence[Mapping[str, Any]],
    *,
    source_behavior: str | None = None,
    target_behavior: str | None = None,
) -> list[Mapping[str, Any]]:
    matched = []
    for entry in entries:
        if source_behavior is not None and str(entry["source_behavior"]) != str(source_behavior):
            continue
        if target_behavior is not None and str(entry["target_behavior"]) != str(target_behavior):
            continue
        matched.append(entry)
    return matched


def mean_v25_train_delta(
    entries: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
) -> torch.Tensor:
    if not entries:
        raise ValueError(f"{field_name} requires at least one train edit bank entry")
    deltas = [
        torch.as_tensor(entry["delta"], dtype=torch.float32).reshape(-1)
        for entry in entries
    ]
    for delta in deltas:
        if int(delta.numel()) != SOURCE_WEIGHT_DIM:
            raise ValueError(f"{field_name} train edit has wrong dimension")
        if not torch.isfinite(delta).all():
            raise ValueError(f"{field_name} train edit is nonfinite")
    return torch.stack(deltas, dim=0).mean(dim=0).to(dtype=torch.float32)


def nearest_v25_train_delta_entry(
    entries: Sequence[Mapping[str, Any]],
    *,
    source_descriptor: torch.Tensor,
) -> Mapping[str, Any]:
    if not entries:
        raise ValueError("nearest train edit requires at least one entry")
    descriptor = source_descriptor.to(dtype=torch.float32).reshape(-1)
    if int(descriptor.numel()) != ACTIVATION_DESCRIPTOR_DIM:
        raise ValueError("source descriptor has wrong dimension")
    scored = []
    for entry in entries:
        entry_descriptor = torch.as_tensor(
            entry["source_descriptor"],
            dtype=torch.float32,
        ).reshape(-1)
        if int(entry_descriptor.numel()) != ACTIVATION_DESCRIPTOR_DIM:
            raise ValueError("train edit source descriptor has wrong dimension")
        distance = float(torch.linalg.norm(entry_descriptor - descriptor).item())
        if not math.isfinite(distance):
            raise ValueError("nonfinite nearest train edit distance")
        scored.append((distance, v25_train_delta_entry_hash(entry), entry))
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][2]


def teacher_oracle_v25_train_delta_entry(
    entries: Sequence[Mapping[str, Any]],
    *,
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
) -> Mapping[str, Any]:
    if not entries:
        raise ValueError("teacher oracle train edit requires at least one entry")
    scored = []
    for entry in entries:
        delta = torch.as_tensor(entry["delta"], dtype=torch.float32).reshape(-1)
        if int(delta.numel()) != SOURCE_WEIGHT_DIM:
            raise ValueError("teacher oracle train edit has wrong dimension")
        edited = source_weights.to(dtype=torch.float32).reshape(-1) + delta
        metrics = v23.v16.v15.v14.functional_metrics(
            edited,
            str(source_behavior),
            str(target_behavior),
            source_weights.to(dtype=torch.float32).reshape(-1),
        )
        target_margin = float(metrics["target_margin"])
        compatible_mse = float(metrics["compatible_source_output_mse"])
        if not math.isfinite(target_margin) or not math.isfinite(compatible_mse):
            raise ValueError("nonfinite teacher oracle train edit metric")
        scored.append((
            -target_margin,
            compatible_mse,
            v25_train_delta_entry_hash(entry),
            entry,
        ))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored[0][3]


def deterministic_v25_shuffled_behavior(
    *,
    source_behavior: str,
    target_behavior: str,
    record_id_hash: str,
    job_plan_hash: str,
) -> str:
    candidates = [
        pattern for pattern in sorted(PATTERNS)
        if str(pattern) not in {str(source_behavior), str(target_behavior)}
    ]
    if not candidates:
        candidates = [pattern for pattern in sorted(PATTERNS) if str(pattern) != str(target_behavior)]
    if not candidates:
        raise ValueError("no shuffled behavior candidate available")
    seed = stable_hash_json({
        "job_plan_hash": str(job_plan_hash),
        "record_id_hash": str(record_id_hash),
        "scope": "v25_shuffled_signature_behavior",
        "source_behavior": str(source_behavior),
        "target_behavior": str(target_behavior),
    })
    return str(candidates[int(seed[:8], 16) % len(candidates)])


def build_v25_train_only_control_context(
    *,
    job: Mapping[str, Any],
    train_delta_bank: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    job_plan_hash: str,
    selected_config_hash: str,
) -> dict[str, Any]:
    bank_hash = require_sha256_hex(
        train_delta_bank.get("bank_hash"),
        field_name="train_delta_bank.bank_hash",
    )
    train_statistics_hash = require_sha256_hex(
        train_stats.get("train_statistics_hash"),
        field_name="train_statistics_hash",
    )
    plan_hash = require_sha256_hex(job_plan_hash, field_name="job_plan_hash")
    selected_hash = require_sha256_hex(selected_config_hash, field_name="selected_config_hash")
    entries = list(train_delta_bank.get("entries", []))
    if not entries:
        raise ValueError("train_delta_bank.entries must not be empty")
    source_behavior = str(job["source_behavior"])
    target_behavior = str(job["target_behavior"])
    record_id_hash = stable_hash_json(str(job["record_id"]))
    source_weights = record_weights_tensor(job["subject"])
    source_descriptor = normalized_activation_descriptor_for_weights(
        source_weights,
        train_stats=train_stats,
    )
    direction_entries = v25_train_delta_entries_matching(
        entries,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    source_entries = v25_train_delta_entries_matching(
        entries,
        source_behavior=source_behavior,
    )
    target_entries = v25_train_delta_entries_matching(
        entries,
        target_behavior=target_behavior,
    )
    nearest_entry = nearest_v25_train_delta_entry(
        direction_entries,
        source_descriptor=source_descriptor,
    )
    oracle_entry = teacher_oracle_v25_train_delta_entry(
        direction_entries,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    deltas = {
        "v21_baseline": mean_v25_train_delta(entries, field_name="v21_baseline"),
        "v22_baseline": mean_v25_train_delta(target_entries, field_name="v22_baseline"),
        "v23_baseline": mean_v25_train_delta(source_entries, field_name="v23_baseline"),
        "nearest_train_delta": torch.as_tensor(
            nearest_entry["delta"],
            dtype=torch.float32,
        ).reshape(-1),
        "teacher_oracle_delta": torch.as_tensor(
            oracle_entry["delta"],
            dtype=torch.float32,
        ).reshape(-1),
        "contrastive_weight_arithmetic": mean_v25_train_delta(
            direction_entries,
            field_name="contrastive_weight_arithmetic",
        ),
    }
    shuffled_behavior = deterministic_v25_shuffled_behavior(
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        record_id_hash=record_id_hash,
        job_plan_hash=plan_hash,
    )
    shuffled_descriptor = train_stats["target_activation_descriptor_by_behavior"][
        shuffled_behavior
    ]
    strategy_summary = {
        "contrastive_weight_arithmetic": {
            "entry_count": len(direction_entries),
            "strategy": "direction_mean_train_edit",
        },
        "nearest_train_delta": {
            "selected_train_entry_hash": v25_train_delta_entry_hash(nearest_entry),
            "strategy": "nearest_source_descriptor_same_direction",
        },
        "teacher_oracle_delta": {
            "diagnostic_only": True,
            "selected_train_entry_hash": v25_train_delta_entry_hash(oracle_entry),
            "strategy": "best_dev_functional_margin_same_direction",
        },
        "v21_baseline": {
            "entry_count": len(entries),
            "strategy": "global_mean_train_edit",
        },
        "v22_baseline": {
            "entry_count": len(target_entries),
            "strategy": "target_conditioned_mean_train_edit",
        },
        "v23_baseline": {
            "entry_count": len(source_entries),
            "strategy": "source_conditioned_mean_train_edit",
        },
    }
    return build_v25_control_context(
        shuffled_target_descriptor=shuffled_descriptor,
        precomputed_delta_by_control_type=deltas,
        provenance={
            "control_context_mode": "train_only_edit_bank",
            "job_plan_hash": plan_hash,
            "record_id_hash": record_id_hash,
            "selected_config_hash": selected_hash,
            "shuffled_behavior": shuffled_behavior,
            "strategy_by_control_type": strategy_summary,
            "train_edit_bank_hash": bank_hash,
            "train_statistics_hash": train_statistics_hash,
        },
    )


def build_v25_train_only_control_contexts_with_progress(
    *,
    jobs: Sequence[Mapping[str, Any]],
    max_jobs: int,
    train_delta_bank: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    job_plan_hash: str,
    selected_config_hash: str,
    progress_log_path: Path,
    started_at_monotonic: float,
    now_monotonic: Any | None = None,
) -> dict[str, dict[str, Any]]:
    capped_jobs = list(jobs)[: int(max_jobs)]
    record_progress_event(
        progress_log_path,
        event="train_only_control_contexts_start",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "max_jobs": int(max_jobs),
            "selected_config_hash": require_sha256_hex(
                selected_config_hash,
                field_name="selected_config_hash",
            ),
            "total_planned_jobs": len(jobs),
            "train_edit_bank_hash": require_sha256_hex(
                train_delta_bank.get("bank_hash"),
                field_name="train_delta_bank.bank_hash",
            ),
        },
    )
    contexts: dict[str, dict[str, Any]] = {}
    for job in capped_jobs:
        record_id_hash = stable_hash_json(str(job["record_id"]))
        contexts[record_id_hash] = build_v25_train_only_control_context(
            job=job,
            train_delta_bank=train_delta_bank,
            train_stats=train_stats,
            job_plan_hash=job_plan_hash,
            selected_config_hash=selected_config_hash,
        )
    record_progress_event(
        progress_log_path,
        event="train_only_control_contexts_completed",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "context_count": len(contexts),
            "context_hashes_hash": stable_hash_json([
                str(context["context_hash"]) for context in contexts.values()
            ]),
            "max_jobs": int(max_jobs),
            "total_planned_jobs": len(jobs),
        },
    )
    return contexts


def precomputed_v25_control_delta(
    control_context: Mapping[str, Any],
    control_type: str,
) -> torch.Tensor:
    delta_by_type = control_context.get("precomputed_delta_by_control_type")
    if not isinstance(delta_by_type, Mapping) or control_type not in delta_by_type:
        raise ValueError(f"missing precomputed delta for {control_type}")
    delta = torch.as_tensor(delta_by_type[control_type], dtype=torch.float32).reshape(-1)
    if int(delta.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError(f"precomputed delta for {control_type} has wrong dimension")
    if not torch.isfinite(delta).all():
        raise ValueError(f"nonfinite precomputed delta for {control_type}")
    return delta


def build_v25_precomputed_delta_control(
    *,
    control_type: str,
    control_context: Mapping[str, Any],
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    selected_config_hash: str,
) -> dict[str, Any]:
    delta = precomputed_v25_control_delta(control_context, control_type)
    metadata = {
        "context_hash": str(control_context.get("context_hash", "missing")),
        "precomputed_delta_sha256": stable_hash_json(tensor_to_hashable(delta)),
        "selected_config_hash": str(selected_config_hash),
        "source": "v25_train_only_control_context",
    }
    return control_record_for_delta(
        control_type=control_type,
        delta=delta,
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata=metadata,
    )


def evaluate_v25_rank1_random_direction_control(
    *,
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    source_descriptor: torch.Tensor,
    target_descriptor: torch.Tensor,
    cache_entry: Mapping[str, Any],
    config: Mapping[str, Any],
    norm_cap: float,
    selected_config_hash: str,
    subject_id: str,
    spectral_basis: torch.Tensor | None = None,
) -> dict[str, Any]:
    activation_delta = (
        target_descriptor.to(dtype=torch.float32).reshape(-1)
        - source_descriptor.to(dtype=torch.float32).reshape(-1)
    )
    seed_hash = stable_hash_json({
        "scope": "v25_rank1_random_direction_control",
        "selected_config_hash": str(selected_config_hash),
        "source": str(source_behavior),
        "subject_id": str(subject_id),
        "target": str(target_behavior),
    })
    generator = torch.Generator(device="cpu").manual_seed(int(seed_hash[:16], 16) % (2**31))
    random_delta = torch.randn(
        activation_delta.shape,
        dtype=torch.float32,
        generator=generator,
    )
    random_norm = float(torch.linalg.norm(random_delta).item())
    matched_norm = float(torch.linalg.norm(activation_delta).item())
    if random_norm <= 1e-12 or matched_norm <= 1e-12:
        normalized_random_delta = torch.zeros_like(activation_delta)
        zero_norm_fallback = True
    else:
        normalized_random_delta = random_delta / random_norm * matched_norm
        zero_norm_fallback = False
    random_target_descriptor = source_descriptor.to(dtype=torch.float32).reshape(-1) + (
        normalized_random_delta
    )
    return evaluate_v25_descriptor_control(
        control_type="rank1_random_direction",
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        source_descriptor=source_descriptor,
        target_descriptor=random_target_descriptor,
        cache_entry=cache_entry,
        config={**dict(config), "projection": "rank1"},
        norm_cap=norm_cap,
        spectral_basis=spectral_basis,
        metadata={
            "matched_activation_delta_norm": matched_norm,
            "random_activation_delta_hash": stable_hash_json(
                tensor_to_hashable(normalized_random_delta)
            ),
            "random_seed_hash": seed_hash,
            "selected_config_hash": str(selected_config_hash),
            "zero_norm_fallback": bool(zero_norm_fallback),
        },
    )


def build_v25_spectral_random_coefficients_control(
    *,
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    matched_spectral_projection_norm: float,
    selected_config_hash: str,
    subject_id: str,
    spectral_basis: torch.Tensor,
) -> dict[str, Any]:
    basis = spectral_basis.to(dtype=torch.float32)
    if basis.ndim != 2 or int(basis.shape[0]) != SOURCE_WEIGHT_DIM:
        raise ValueError("spectral basis must have shape [SOURCE_WEIGHT_DIM, rank]")
    if not torch.isfinite(basis).all():
        raise ValueError("nonfinite spectral basis control input")
    seed_hash = stable_hash_json({
        "scope": "v25_spectral_basis_random_coefficients_control",
        "selected_config_hash": str(selected_config_hash),
        "source": str(source_behavior),
        "subject_id": str(subject_id),
        "target": str(target_behavior),
    })
    generator = torch.Generator(device="cpu").manual_seed(int(seed_hash[:16], 16) % (2**31))
    coeff = torch.randn(int(basis.shape[1]), dtype=torch.float32, generator=generator)
    raw_delta = basis @ coeff
    raw_norm = float(torch.linalg.norm(raw_delta).item())
    matched_norm = float(matched_spectral_projection_norm)
    if raw_norm <= 1e-12 or matched_norm <= 1e-12:
        delta = torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.float32)
        zero_norm_fallback = True
    else:
        delta = raw_delta / raw_norm * matched_norm
        zero_norm_fallback = False
    return control_record_for_delta(
        control_type="spectral_basis_random_coefficients",
        delta=delta,
        source_behavior=source_behavior,
        source_weights=source_weights,
        target_behavior=target_behavior,
        metadata={
            "coefficient_sha256": stable_hash_json(tensor_to_hashable(coeff)),
            "matched_spectral_projection_norm": matched_norm,
            "raw_delta_sha256": stable_hash_json(tensor_to_hashable(raw_delta)),
            "random_seed_hash": seed_hash,
            "selected_config_hash": str(selected_config_hash),
            "spectral_basis_sha256": stable_hash_json(tensor_to_hashable(basis)),
            "zero_norm_fallback": bool(zero_norm_fallback),
        },
    )


def build_v25_native_controls(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    train_stats: Mapping[str, Any],
    cache_entry: Mapping[str, Any],
    config: Mapping[str, Any],
    norm_cap: float,
    matched_delta_sha256: str,
    matched_delta_norm: float,
    matched_spectral_projection_norm: float,
    selected_config_hash: str,
    spectral_basis: torch.Tensor | None = None,
    control_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    source_weights = record_weights_tensor(subject)
    source_descriptor = cache_entry["source_descriptor"]
    target_descriptor = train_stats["target_activation_descriptor_by_behavior"][
        str(target_behavior)
    ]
    context = dict(control_context or {})
    context_hash = require_sha256_hex(
        context.get("context_hash"),
        field_name="control_context.context_hash",
    )
    if "shuffled_target_descriptor" not in context:
        raise ValueError("missing shuffled_target_descriptor in control_context")
    if spectral_basis is None:
        raise ValueError("spectral_basis is required for spectral random control")
    controls_by_type: dict[str, dict[str, Any]] = {}
    zero_descriptor_norm_hash = stable_hash_json(
        tensor_to_hashable(torch.zeros_like(source_descriptor))
    )
    zero_target_controls = [
        ("no_signature_ablation", {"zero_activation_target": True}),
        ("no_signature_trained", {
            "selected_config_hash": str(selected_config_hash),
            "trained_ablation": True,
            "zero_activation_target": True,
            "zero_descriptor_norm_hash": zero_descriptor_norm_hash,
        }),
        ("source_behavior_target_ablation", {"target_behavior_override": str(source_behavior)}),
    ]
    for control_type, metadata in zero_target_controls:
        controls_by_type[control_type] = evaluate_v25_descriptor_control(
            control_type=control_type,
            source_weights=source_weights,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            source_descriptor=source_descriptor,
            target_descriptor=source_descriptor,
            cache_entry=cache_entry,
            config=config,
            norm_cap=norm_cap,
            spectral_basis=spectral_basis,
            metadata=metadata,
        )
    shuffled_descriptor = torch.as_tensor(
        context["shuffled_target_descriptor"],
        dtype=torch.float32,
    ).reshape(-1)
    if int(shuffled_descriptor.numel()) != ACTIVATION_DESCRIPTOR_DIM:
        raise ValueError("shuffled target descriptor has wrong dimension")
    controls_by_type["shuffled_signature"] = evaluate_v25_descriptor_control(
        control_type="shuffled_signature",
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        source_descriptor=source_descriptor,
        target_descriptor=shuffled_descriptor,
        cache_entry=cache_entry,
        config=config,
        norm_cap=norm_cap,
        spectral_basis=spectral_basis,
        metadata={
            "context_hash": context_hash,
            "selected_config_hash": str(selected_config_hash),
            "shuffled_target_descriptor_hash": stable_hash_json(
                tensor_to_hashable(shuffled_descriptor)
            ),
        },
    )
    for control_type in ["v21_baseline", "v22_baseline", "v23_baseline"]:
        controls_by_type[control_type] = build_v25_precomputed_delta_control(
            control_type=control_type,
            control_context=context,
            source_weights=source_weights,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            selected_config_hash=selected_config_hash,
        )
    controls_by_type["closed_form_unprojected_jacobian"] = evaluate_v25_descriptor_control(
        control_type="closed_form_unprojected_jacobian",
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        source_descriptor=source_descriptor,
        target_descriptor=target_descriptor,
        cache_entry=cache_entry,
        config={**dict(config), "projection": "none"},
        norm_cap=norm_cap,
        metadata={"matched_delta_sha256": str(matched_delta_sha256)},
    )
    controls_by_type["rank1_random_direction"] = evaluate_v25_rank1_random_direction_control(
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        source_descriptor=source_descriptor,
        target_descriptor=target_descriptor,
        cache_entry=cache_entry,
        config=config,
        norm_cap=norm_cap,
        selected_config_hash=selected_config_hash,
        subject_id=str(subject["subject_id"]),
        spectral_basis=spectral_basis,
    )
    controls_by_type["spectral_basis_random_coefficients"] = (
        build_v25_spectral_random_coefficients_control(
            source_weights=source_weights,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            matched_spectral_projection_norm=matched_spectral_projection_norm,
            selected_config_hash=selected_config_hash,
            subject_id=str(subject["subject_id"]),
            spectral_basis=spectral_basis,
        )
    )
    controls_by_type["contrastive_weight_arithmetic"] = build_v25_precomputed_delta_control(
        control_type="contrastive_weight_arithmetic",
        control_context=context,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        selected_config_hash=selected_config_hash,
    )
    for control_type in ["nearest_train_delta", "teacher_oracle_delta"]:
        controls_by_type[control_type] = build_v25_precomputed_delta_control(
            control_type=control_type,
            control_context=context,
            source_weights=source_weights,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            selected_config_hash=selected_config_hash,
        )
    controls_by_type["target_only_no_source_compat"] = evaluate_v25_descriptor_control(
        control_type="target_only_no_source_compat",
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        source_descriptor=source_descriptor,
        target_descriptor=target_descriptor,
        cache_entry=cache_entry,
        config={**dict(config), "compat_weight": 0.0},
        norm_cap=norm_cap,
        spectral_basis=spectral_basis,
    )
    controls_by_type["activation_only_no_weight_projection"] = evaluate_v25_descriptor_control(
        control_type="activation_only_no_weight_projection",
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        source_descriptor=source_descriptor,
        target_descriptor=target_descriptor,
        cache_entry=cache_entry,
        config={**dict(config), "projection": "none"},
        norm_cap=1.0e12,
        metadata={"norm_cap_disabled_by_large_finite_cap": True},
    )
    matched_delta = torch.zeros(SOURCE_WEIGHT_DIM, dtype=torch.float32)
    matched_delta[0] = float(matched_delta_norm)
    for control in build_v25_random_matched_norm_controls(
        matched_delta=matched_delta,
        source_weights=source_weights,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        record_id=str(subject["subject_id"]),
        selected_config_hash=str(selected_config_hash),
        projection=str(config["projection"]),
        spectral_basis=spectral_basis,
    ):
        controls_by_type[str(control["control_type"])] = control
    controls = [controls_by_type[control_type] for control_type in expected_v25_control_types()]
    validate_v25_controls(controls)
    return controls


def recursive_numeric_finiteness_failures(value: Any, *, prefix: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            failures.extend(recursive_numeric_finiteness_failures(nested, prefix=path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            failures.extend(
                recursive_numeric_finiteness_failures(nested, prefix=f"{prefix}[{index}]")
            )
    elif isinstance(value, tuple):
        for index, nested in enumerate(value):
            failures.extend(
                recursive_numeric_finiteness_failures(nested, prefix=f"{prefix}[{index}]")
            )
    elif isinstance(value, torch.Tensor):
        if value.ndim == 0:
            if not math.isfinite(float(value.detach().cpu().item())):
                failures.append(prefix)
        elif not bool(torch.isfinite(value).all().item()):
            failures.append(prefix)
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            failures.append(prefix)
    return failures


def expected_v25_random_control_types() -> list[str]:
    return [
        f"random_matched_norm_{index:02d}"
        for index in range(RANDOM_CONTROLS_PER_RECORD)
    ]


def expected_v25_control_types() -> list[str]:
    return [
        *PROOF_CRITICAL_CONTROL_TYPES,
        *DIAGNOSTIC_CONTROL_TYPES,
        *expected_v25_random_control_types(),
    ]


def proof_critical_control_types_for_pareto() -> set[str]:
    return set(PROOF_CRITICAL_CONTROL_TYPES)


def pareto_dominates(control: Mapping[str, Any], matched: Mapping[str, Any]) -> bool:
    control_margin = float(control["target_margin"])
    matched_margin = float(matched["target_margin"])
    control_mse = float(control["compatible_source_output_mse"])
    matched_mse = float(matched["compatible_source_output_mse"])
    return bool(
        control_margin >= matched_margin
        and control_mse <= matched_mse
        and (
            control_margin > matched_margin + 1e-8
            or control_mse < matched_mse - 1e-8
        )
    )


def control_by_type(controls: Sequence[Mapping[str, Any]], control_type: str) -> Mapping[str, Any]:
    matches = [control for control in controls if str(control["control_type"]) == control_type]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one control {control_type}, found {len(matches)}")
    return matches[0]


def validate_v25_controls(controls: Sequence[Mapping[str, Any]]) -> None:
    expected = expected_v25_control_types()
    actual = [str(control.get("control_type")) for control in controls]
    if len(actual) != EXPECTED_CONTROLS_PER_RECORD:
        raise ValueError(
            f"control count mismatch: expected {EXPECTED_CONTROLS_PER_RECORD}, got {len(actual)}"
        )
    if sorted(actual) != sorted(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"control type mismatch: missing={missing}, extra={extra}")


def build_v25_proof_gate_diagnostics(
    *,
    matched_payload: Mapping[str, Any],
    margin_pass_payloads: Sequence[Mapping[str, Any]],
    compatible_mse_pass: bool,
) -> dict[str, Any]:
    advantages = [float(payload["advantage"]) for payload in margin_pass_payloads]
    failed_control_types = sorted(
        str(payload["control_type"])
        for payload in margin_pass_payloads
        if not bool(payload["passed"])
    )
    shuffled_signature_passes = [
        bool(payload["passed"])
        for payload in margin_pass_payloads
        if str(payload["control_type"]) == "shuffled_signature"
    ]
    return {
        "compatible_mse_pass": bool(compatible_mse_pass),
        "control_margin_fail_count": len(failed_control_types),
        "control_margin_pass_count": len(margin_pass_payloads) - len(failed_control_types),
        "failed_control_types_hash": stable_hash_json(failed_control_types),
        "individual_all_gates_passed": bool(matched_payload["individual_all_gates_passed"]),
        "mean_control_margin_advantage": mean_float(advantages),
        "min_control_margin_advantage": min(advantages) if advantages else 0.0,
        "pareto_undominated": bool(matched_payload["pareto_undominated"]),
        "shuffled_signature_margin_pass": all(shuffled_signature_passes),
        "target_margin_pass": (
            float(matched_payload["target_margin"]) >= PER_RECORD_MIN_TARGET_MARGIN
        ),
        "target_prediction_pass": bool(matched_payload["target_prediction_pass"]),
    }


def build_v25_proof_record(
    *,
    source_behavior: str,
    target_behavior: str,
    matched: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_v25_controls(controls)
    matched_payload = dict(matched)
    controls_payload = [dict(control) for control in controls]
    finite_failures = recursive_numeric_finiteness_failures({
        "controls": controls_payload,
        "matched": matched_payload,
    })
    if finite_failures:
        raise ValueError("nonfinite proof record metrics: " + ", ".join(finite_failures[:5]))
    matched_payload["target_prediction_pass"] = (
        str(matched_payload["predicted_behavior"]) == str(target_behavior)
    )
    proof_controls = [
        control
        for control in controls_payload
        if str(control["control_type"]) in proof_critical_control_types_for_pareto()
    ]
    named_proof_controls = [
        control
        for control in controls_payload
        if str(control["control_type"]) in PROOF_CRITICAL_CONTROL_TYPES
    ]
    pareto_dominators = [
        control for control in proof_controls if pareto_dominates(control, matched_payload)
    ]
    best_control = max(named_proof_controls, key=lambda item: float(item["target_margin"]))
    matched_payload["pareto_undominated"] = not pareto_dominators
    matched_payload["pareto_dominator_count"] = len(pareto_dominators)
    matched_payload["pareto_dominator_types"] = sorted({
        str(control["control_type"]) for control in pareto_dominators
    })
    matched_payload["matched_minus_best_control_target_margin"] = (
        float(matched_payload["target_margin"]) - float(best_control["target_margin"])
    )
    best_compatible_mse = min(
        float(control["compatible_source_output_mse"]) for control in named_proof_controls
    )
    matched_payload["matched_minus_best_control_compatible_source_output_mse"] = (
        float(matched_payload["compatible_source_output_mse"]) - best_compatible_mse
    )
    margin_pass_payloads = []
    for control_type in PROOF_CRITICAL_CONTROL_TYPES:
        control = control_by_type(controls_payload, control_type)
        metric_key = f"matched_minus_{control_type}_target_margin"
        advantage = float(matched_payload["target_margin"]) - float(control["target_margin"])
        matched_payload[metric_key] = advantage
        threshold = (
            PER_RECORD_MIN_SHUFFLED_MARGIN_ADVANTAGE
            if control_type == "shuffled_signature"
            else PER_RECORD_MIN_CONTROL_MARGIN_ADVANTAGE
        )
        margin_pass_payloads.append({
            "advantage": advantage,
            "control_type": control_type,
            "passed": advantage >= threshold,
            "threshold": threshold,
        })
    compatible_mse_pass = (
        float(matched_payload["compatible_source_output_mse"])
        <= best_compatible_mse + PER_RECORD_COMPATIBLE_MSE_TOLERANCE
    )
    matched_payload["individual_all_gates_passed"] = bool(
        matched_payload["target_prediction_pass"]
        and float(matched_payload["target_margin"]) >= PER_RECORD_MIN_TARGET_MARGIN
        and matched_payload["pareto_undominated"]
        and compatible_mse_pass
        and all(bool(payload["passed"]) for payload in margin_pass_payloads)
    )
    proof_gate_diagnostics = build_v25_proof_gate_diagnostics(
        matched_payload=matched_payload,
        margin_pass_payloads=margin_pass_payloads,
        compatible_mse_pass=compatible_mse_pass,
    )
    summary = {
        "best_control_target_margin": float(best_control["target_margin"]),
        "best_control_type": str(best_control["control_type"]),
        "individual_all_gates_passed": matched_payload["individual_all_gates_passed"],
        "matched_minus_best_control_target_margin": matched_payload[
            "matched_minus_best_control_target_margin"
        ],
        "matched_minus_shuffled_signature_target_margin": matched_payload[
            "matched_minus_shuffled_signature_target_margin"
        ],
        "pareto_undominated": matched_payload["pareto_undominated"],
        "proof_gate_diagnostics": proof_gate_diagnostics,
        "target_prediction_pass": matched_payload["target_prediction_pass"],
    }
    for control_type in PROOF_CRITICAL_CONTROL_TYPES:
        summary[f"matched_minus_{control_type}_target_margin"] = matched_payload[
            f"matched_minus_{control_type}_target_margin"
        ]
    record = {
        "controls": controls_payload,
        "direction": f"{source_behavior}->{target_behavior}",
        "matched": matched_payload,
        "source_behavior": str(source_behavior),
        "summary": summary,
        "target_behavior": str(target_behavior),
    }
    finite_record_failures = recursive_numeric_finiteness_failures(record)
    if finite_record_failures:
        raise ValueError(
            "nonfinite proof record after summary: "
            + ", ".join(finite_record_failures[:5])
        )
    return record


def evaluate_v25_development_job(
    *,
    job: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    config: Mapping[str, Any],
    norm_cap: float,
    spectral_basis: torch.Tensor,
    control_context: Mapping[str, Any],
    selected_config_hash: str,
    script_sha256: str,
    empirical_task_vector_bank: Mapping[str, Any] | None = None,
    progress_log_path: Path | None = None,
    started_at_monotonic: float = 0.0,
    now_monotonic: Any | None = None,
) -> dict[str, Any]:
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    context_provenance = control_context.get("provenance", {})
    if bool(config.get("allow_pinv_fallback", False)):
        if not isinstance(context_provenance, Mapping) or not bool(
            context_provenance.get("dry_run_only")
        ):
            raise ValueError("pinv fallback is allowed only for dry-run-only contexts")
    subject = job["subject"]
    source_behavior = str(job["source_behavior"])
    target_behavior = str(job["target_behavior"])
    cache_entry = compute_jacobian_cache_entry(
        subject,
        source_behavior=source_behavior,
        train_stats=train_stats,
        script_sha256=str(script_sha256),
    )
    matched_edit_source = str(config.get("matched_edit_source", "jacobian"))
    if matched_edit_source == "jacobian":
        matched = evaluate_v25_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            train_stats=train_stats,
            cache_entry=cache_entry,
            config=config,
            norm_cap=norm_cap,
            spectral_basis=spectral_basis,
        )
    elif matched_edit_source == "empirical_centroid_task_vector":
        if empirical_task_vector_bank is None:
            raise ValueError("empirical_task_vector_bank is required")
        matched = evaluate_v25_empirical_task_vector_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            empirical_task_vector_bank=empirical_task_vector_bank,
            selected_config_hash=selected_hash,
            spectral_basis=spectral_basis,
        )
    elif matched_edit_source == V27_LOCALIZED_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for localized_behavior_loss_subspace"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for localized_behavior_loss_subspace"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_localized_behavior_loss_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            spectral_basis=spectral_basis,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
    elif matched_edit_source == V28_ANCHOR_NULLSPACE_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for anchor_nullspace_trust_region"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for anchor_nullspace_trust_region"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_anchor_nullspace_trust_region_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
    elif matched_edit_source == V29_BREADTH_FIRST_SPARSE_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for breadth_first_sparse_support"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for breadth_first_sparse_support"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_breadth_first_sparse_support_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
    elif matched_edit_source == V30_MARGIN_GATED_SPARSE_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for margin_gated_sparse_support"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for margin_gated_sparse_support"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_margin_gated_sparse_support_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
    elif matched_edit_source == V31_ORTHOGONAL_SIGN_SPARSE_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for orthogonal_sign_sparse_support"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for orthogonal_sign_sparse_support"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_orthogonal_sign_sparse_support_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
    elif matched_edit_source == V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for support_tournament_margin_sparse"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for support_tournament_margin_sparse"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_support_tournament_sparse_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
    elif matched_edit_source == V35_SUPPORT_SOURCE_LINE_SEARCH_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for support_source_line_search_sparse"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for support_source_line_search_sparse"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_support_source_line_search_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
    elif matched_edit_source == V36_COMPATIBLE_NULLSPACE_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for compatible_nullspace_projected_sparse"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for compatible_nullspace_projected_sparse"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_compatible_nullspace_projected_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
    elif matched_edit_source == V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for projected_support_optimizer_sparse"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for projected_support_optimizer_sparse"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_projected_support_optimizer_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
    elif matched_edit_source == V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for compatible_mse_gated_projected_optimizer_sparse"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for compatible_mse_gated_projected_optimizer_sparse"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_compatible_gated_projected_optimizer_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
    elif matched_edit_source == V39_TARGET_FEASIBLE_LEXICOGRAPHIC_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for "
                "target_feasible_lexicographic_projected_optimizer_sparse"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for "
                "target_feasible_lexicographic_projected_optimizer_sparse"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = (
            evaluate_v25_target_feasible_lexicographic_projected_optimizer_matched_edit(
                subject=subject,
                source_behavior=source_behavior,
                target_behavior=target_behavior,
                config=config,
                norm_cap=norm_cap,
                selected_config_hash=selected_hash,
                train_pool_file_sha256=train_pool_file_sha256,
                train_pool_summary_hash=train_pool_summary_hash,
                script_sha256=str(script_sha256),
                progress_log_path=progress_log_path,
                started_at_monotonic=started_at_monotonic,
                now_monotonic=now_monotonic,
                record_id_hash=stable_hash_json(str(job["record_id"])),
            )
        )
    elif matched_edit_source == V40_TARGET_TOLERANCE_LOCALITY_BUDGET_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for "
                "target_tolerance_locality_budget_projected_optimizer_sparse"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for "
                "target_tolerance_locality_budget_projected_optimizer_sparse"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_target_tolerance_locality_budget_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
    elif matched_edit_source == V41_TRAJECTORY_FRONTIER_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for "
                "trajectory_frontier_projected_optimizer_sparse"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for "
                "trajectory_frontier_projected_optimizer_sparse"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_trajectory_frontier_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
    elif matched_edit_source == V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for "
                "compatible_dual_frontier_projected_optimizer_sparse"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for "
                "compatible_dual_frontier_projected_optimizer_sparse"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_compatible_dual_frontier_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
    else:
        raise ValueError(f"unknown matched_edit_source: {matched_edit_source}")
    if "matched_spectral_projection_norm" not in matched["editor"]:
        raise ValueError("matched spectral projection norm missing")
    controls = build_v25_native_controls(
        subject=subject,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        train_stats=train_stats,
        cache_entry=cache_entry,
        config=v25_native_control_config(config),
        norm_cap=norm_cap,
        matched_delta_sha256=str(matched["editor"]["delta_sha256"]),
        matched_delta_norm=float(matched["delta_norm"]),
        matched_spectral_projection_norm=float(
            matched["editor"]["matched_spectral_projection_norm"]
        ),
        selected_config_hash=selected_hash,
        spectral_basis=spectral_basis,
        control_context=control_context,
    )
    proof_record = build_v25_proof_record(
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        matched=matched,
        controls=controls,
    )
    proof_record["cache_key"] = str(cache_entry["cache_key"])
    proof_record["control_context_hash"] = require_sha256_hex(
        control_context.get("context_hash"),
        field_name="control_context.context_hash",
    )
    proof_record["record_id_hash"] = stable_hash_json(str(job["record_id"]))
    proof_record["selected_config_hash"] = selected_hash
    finite_failures = recursive_numeric_finiteness_failures(proof_record)
    if finite_failures:
        raise ValueError("nonfinite development proof record: " + ", ".join(finite_failures[:5]))
    return proof_record


def redacted_v25_development_record_progress(
    proof_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "control_context_hash": str(proof_record["control_context_hash"]),
        "control_count": len(proof_record["controls"]),
        "direction": str(proof_record["direction"]),
        "individual_all_gates_passed": bool(
            proof_record["summary"]["individual_all_gates_passed"]
        ),
        "matched_delta_norm": float(proof_record["matched"]["delta_norm"]),
        "matched_target_margin": float(proof_record["matched"]["target_margin"]),
        "pareto_undominated": bool(proof_record["summary"]["pareto_undominated"]),
        "proof_gate_diagnostics": dict(
            proof_record["summary"]["proof_gate_diagnostics"]
        ),
        "record_id_hash": str(proof_record["record_id_hash"]),
        "selected_config_hash": str(proof_record["selected_config_hash"]),
        "target_prediction_pass": bool(proof_record["summary"]["target_prediction_pass"]),
    }


def evaluate_v25_development_job_with_progress(
    *,
    job: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    config: Mapping[str, Any],
    norm_cap: float,
    spectral_basis: torch.Tensor,
    control_context: Mapping[str, Any],
    selected_config_hash: str,
    script_sha256: str,
    progress_log_path: Path,
    started_at_monotonic: float,
    job_index: int,
    total_jobs: int,
    now_monotonic: Any | None = None,
    empirical_task_vector_bank: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record_id_hash = stable_hash_json(str(job["record_id"]))
    record_progress_event(
        progress_log_path,
        event="development_evaluation_record_start",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "direction": str(job["direction"]),
            "job_index": int(job_index),
            "record_id_hash": record_id_hash,
            "total_jobs": int(total_jobs),
        },
    )
    proof_record = evaluate_v25_development_job(
        job=job,
        train_stats=train_stats,
        config=config,
        norm_cap=norm_cap,
        spectral_basis=spectral_basis,
        control_context=control_context,
        selected_config_hash=selected_config_hash,
        script_sha256=script_sha256,
        empirical_task_vector_bank=empirical_task_vector_bank,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
    )
    record_progress_event(
        progress_log_path,
        event="development_evaluation_record_completed",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            **redacted_v25_development_record_progress(proof_record),
            "job_index": int(job_index),
            "total_jobs": int(total_jobs),
        },
    )
    return proof_record


def redact_v25_development_evaluation_summary(
    *,
    proof_records: Sequence[Mapping[str, Any]],
    max_jobs: int,
    total_planned_jobs: int,
) -> dict[str, Any]:
    proof_record_hashes = [
        stable_hash_json(proof_record)
        for proof_record in proof_records
    ]
    return {
        "evaluated_count": len(proof_records),
        "max_jobs": int(max_jobs),
        "proof_record_hashes": proof_record_hashes,
        "total_planned_jobs": int(total_planned_jobs),
    }


def evaluate_v25_development_jobs_with_progress(
    *,
    jobs: Sequence[Mapping[str, Any]],
    max_jobs: int,
    train_stats: Mapping[str, Any],
    config: Mapping[str, Any],
    norm_cap: float,
    spectral_basis: torch.Tensor,
    control_context_by_record_hash: Mapping[str, Mapping[str, Any]],
    selected_config_hash: str,
    script_sha256: str,
    progress_log_path: Path,
    started_at_monotonic: float,
    now_monotonic: Any | None = None,
    empirical_task_vector_bank: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if int(max_jobs) < 0:
        raise ValueError("max_jobs must be nonnegative")
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    capped_jobs = list(jobs)[: int(max_jobs)]
    record_progress_event(
        progress_log_path,
        event="development_evaluation_start",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "max_jobs": int(max_jobs),
            "selected_config_hash": selected_hash,
            "total_planned_jobs": len(jobs),
        },
    )
    proof_records = []
    for job_index, job in enumerate(capped_jobs):
        record_id_hash = stable_hash_json(str(job["record_id"]))
        if record_id_hash not in control_context_by_record_hash:
            raise ValueError(f"missing control context for record hash {record_id_hash}")
        proof_records.append(evaluate_v25_development_job_with_progress(
            job=job,
            train_stats=train_stats,
            config=config,
            norm_cap=norm_cap,
            spectral_basis=spectral_basis,
            control_context=control_context_by_record_hash[record_id_hash],
            selected_config_hash=selected_hash,
            script_sha256=script_sha256,
            empirical_task_vector_bank=empirical_task_vector_bank,
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            job_index=job_index,
            total_jobs=len(capped_jobs),
            now_monotonic=now_monotonic,
        ))
    summary = redact_v25_development_evaluation_summary(
        proof_records=proof_records,
        max_jobs=int(max_jobs),
        total_planned_jobs=len(jobs),
    )
    record_progress_event(
        progress_log_path,
        event="development_evaluation_completed",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra=summary,
    )
    return {
        **summary,
        "proof_records": proof_records,
    }


def v25_inner_validation_config_hash(config: Mapping[str, Any]) -> str:
    return stable_hash_json({
        "config": dict(config),
        "scope": "v25_inner_validation_config",
    })


def invalid_v25_inner_validation_candidate(
    *,
    config: Mapping[str, Any],
    config_hash: str,
    error: Exception,
) -> dict[str, Any]:
    checked_config_hash = require_sha256_hex(config_hash, field_name="config_hash")
    return {
        "config": dict(config),
        "config_hash": checked_config_hash,
        "config_index": int(config["config_index"]),
        "contract_failure_count": 1,
        "error_hash": stable_hash_json(str(error)),
        "error_type": type(error).__name__,
        "invalid": True,
        "mean_matched_minus_best_control_target_margin": 0.0,
        "mean_matched_minus_shuffled_signature_target_margin": 0.0,
        "mean_target_margin": 0.0,
        "pareto_undominated_rate": 0.0,
        "proof_gate_failure_count": 10_000,
        "proof_record_hashes_hash": stable_hash_json([]),
        "record_count": 0,
        "target_prediction_rate": 0.0,
    }


def run_v25_inner_validation_successive_halving_with_progress(
    *,
    configs: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    train_subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    rung_job_counts: Sequence[int],
    keep_fractions: Sequence[float],
    norm_cap: float,
    job_plan_hash: str,
    script_sha256: str,
    progress_log_path: Path,
    started_at_monotonic: float,
    now_monotonic: Any | None = None,
    experiment_variant: str = V26_EXPERIMENT_VARIANT,
) -> dict[str, Any]:
    plan = build_v25_successive_halving_plan(
        configs=configs,
        rung_job_counts=rung_job_counts,
        keep_fractions=keep_fractions,
    )
    checked_job_plan_hash = require_sha256_hex(job_plan_hash, field_name="job_plan_hash")
    checked_train_pool_file_sha256 = require_sha256_hex(
        train_pool_file_sha256,
        field_name="train_pool_file_sha256",
    )
    checked_train_pool_summary_hash = require_sha256_hex(
        train_pool_summary_hash,
        field_name="train_pool_summary_hash",
    )
    ordered_jobs = list(jobs)
    active_configs = [dict(config) for config in configs]
    record_progress_event(
        progress_log_path,
        event="inner_validation_start",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "experiment_variant": str(experiment_variant),
            "job_plan_hash": checked_job_plan_hash,
            "plan": plan,
            "train_pool_file_sha256": checked_train_pool_file_sha256,
            "train_pool_summary_hash": checked_train_pool_summary_hash,
            "total_planned_jobs": len(ordered_jobs),
        },
    )
    rung_results = []
    all_candidates = []
    final_ordered_candidates: list[dict[str, Any]] = []
    for rung in plan["rungs"]:
        rung_index = int(rung["rung_index"])
        rung_job_count = min(int(rung["rung_job_count"]), len(ordered_jobs))
        rung_configs = active_configs[: int(rung["input_config_count"])]
        record_progress_event(
            progress_log_path,
            event="inner_validation_rung_start",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra={
                "config_count": len(rung_configs),
                "config_hashes_hash": stable_hash_json([
                    v25_inner_validation_config_hash(config)
                    for config in rung_configs
                ]),
                "rung_index": rung_index,
                "rung_job_count": rung_job_count,
            },
        )
        rung_candidates = []
        for config_position, config in enumerate(rung_configs):
            config_hash = v25_inner_validation_config_hash(config)
            record_progress_event(
                progress_log_path,
                event="inner_validation_candidate_start",
                started_at_monotonic=started_at_monotonic,
                now_monotonic=now_monotonic,
                extra={
                    "config_hash": config_hash,
                    "config_index": int(config["config_index"]),
                    "config_position": int(config_position),
                    "rung_index": rung_index,
                    "rung_job_count": rung_job_count,
                },
            )
            try:
                spectral_basis = None
                spectral_audit = None
                if v25_config_requires_spectral_basis(config):
                    seed_config = v25_spectral_seed_config(config)
                    seed_train_delta_bank = build_v25_train_delta_bank_with_progress(
                        train_subjects=train_subjects,
                        train_stats=train_stats,
                        config=seed_config,
                        norm_cap=norm_cap,
                        script_sha256=script_sha256,
                        progress_log_path=progress_log_path,
                        started_at_monotonic=started_at_monotonic,
                        now_monotonic=now_monotonic,
                        train_edit_bank_role="spectral_seed",
                    )
                    seed_train_delta_matrix = torch.stack([
                        torch.as_tensor(entry["delta"], dtype=torch.float32).reshape(-1)
                        for entry in seed_train_delta_bank["entries"]
                    ], dim=0)
                    spectral_basis, spectral_audit = compute_train_spectral_basis(
                        seed_train_delta_matrix,
                        rank=min(
                            4,
                            int(seed_train_delta_matrix.shape[0]),
                            int(seed_train_delta_matrix.shape[1]),
                        ),
                    )
                train_delta_bank_config = v25_native_control_config(config)
                train_delta_bank = build_v25_train_delta_bank_with_progress(
                    train_subjects=train_subjects,
                    train_stats=train_stats,
                    config=train_delta_bank_config,
                    norm_cap=norm_cap,
                    script_sha256=script_sha256,
                    spectral_basis=spectral_basis,
                    progress_log_path=progress_log_path,
                    started_at_monotonic=started_at_monotonic,
                    now_monotonic=now_monotonic,
                    train_edit_bank_role="actual",
                )
                empirical_task_vector_bank = None
                if str(config.get("matched_edit_source", "jacobian")) == (
                    "empirical_centroid_task_vector"
                ):
                    empirical_task_vector_bank = (
                        build_v25_empirical_task_vector_bank_with_progress(
                            train_subjects=train_subjects,
                            train_pool_file_sha256=checked_train_pool_file_sha256,
                            train_pool_summary_hash=checked_train_pool_summary_hash,
                            config=config,
                            norm_cap=norm_cap,
                            script_sha256=script_sha256,
                            spectral_basis=spectral_basis,
                            progress_log_path=progress_log_path,
                            started_at_monotonic=started_at_monotonic,
                            now_monotonic=now_monotonic,
                            experiment_variant=experiment_variant,
                        )
                    )
                if spectral_basis is None or spectral_audit is None:
                    train_delta_matrix = torch.stack([
                        torch.as_tensor(entry["delta"], dtype=torch.float32).reshape(-1)
                        for entry in train_delta_bank["entries"]
                    ], dim=0)
                    spectral_basis, spectral_audit = compute_train_spectral_basis(
                        train_delta_matrix,
                        rank=min(
                            4,
                            int(train_delta_matrix.shape[0]),
                            int(train_delta_matrix.shape[1]),
                        ),
                    )
                context_by_record_hash = build_v25_train_only_control_contexts_with_progress(
                    jobs=ordered_jobs,
                    max_jobs=rung_job_count,
                    train_delta_bank=train_delta_bank,
                    train_stats=train_stats,
                    job_plan_hash=checked_job_plan_hash,
                    selected_config_hash=config_hash,
                    progress_log_path=progress_log_path,
                    started_at_monotonic=started_at_monotonic,
                    now_monotonic=now_monotonic,
                )
                evaluation = evaluate_v25_development_jobs_with_progress(
                    jobs=ordered_jobs,
                    max_jobs=rung_job_count,
                    train_stats=train_stats,
                    config=config,
                    norm_cap=norm_cap,
                    spectral_basis=spectral_basis,
                    control_context_by_record_hash=context_by_record_hash,
                    selected_config_hash=config_hash,
                    script_sha256=script_sha256,
                    progress_log_path=progress_log_path,
                    started_at_monotonic=started_at_monotonic,
                    now_monotonic=now_monotonic,
                    empirical_task_vector_bank=empirical_task_vector_bank,
                )
                candidate = summarize_v25_inner_validation_candidate(
                    config=config,
                    config_hash=config_hash,
                    proof_records=evaluation["proof_records"],
                    expected_record_count=rung_job_count,
                )
                candidate["evaluated_count"] = int(evaluation["evaluated_count"])
                candidate["spectral_basis_sha256"] = str(spectral_audit["basis_sha256"])
                candidate["train_edit_bank_hash"] = str(train_delta_bank["bank_hash"])
                candidate["experiment_variant"] = str(experiment_variant)
                if empirical_task_vector_bank is not None:
                    candidate["empirical_task_vector_bank_hash"] = str(
                        empirical_task_vector_bank["bank_hash"]
                    )
            except Exception as error:
                candidate = invalid_v25_inner_validation_candidate(
                    config=config,
                    config_hash=config_hash,
                    error=error,
                )
            candidate["rung_index"] = rung_index
            candidate["rung_job_count"] = rung_job_count
            rung_candidates.append(candidate)
            all_candidates.append(candidate)
            record_progress_event(
                progress_log_path,
                event="inner_validation_candidate_completed",
                started_at_monotonic=started_at_monotonic,
                now_monotonic=now_monotonic,
                extra={
                    key: value
                    for key, value in candidate.items()
                    if key != "config"
                },
            )
        ordered_candidates = sorted(
            rung_candidates,
            key=inner_validation_ranking_tuple,
        )
        valid_ordered_candidates = [
            candidate for candidate in ordered_candidates
            if not bool(candidate["invalid"])
        ]
        if not valid_ordered_candidates:
            failed_result = {
                "candidate_count": len(all_candidates),
                "experiment_variant": str(experiment_variant),
                "failed_rung_index": rung_index,
                "invalid": True,
                "plan": plan,
                "rungs": rung_results,
                "stage": "inner_validation_failed",
            }
            record_progress_event(
                progress_log_path,
                event="inner_validation_failed",
                started_at_monotonic=started_at_monotonic,
                now_monotonic=now_monotonic,
                extra={
                    "candidate_count": len(all_candidates),
                    "experiment_variant": str(experiment_variant),
                    "failed_rung_index": rung_index,
                    "invalid_candidate_count": len(rung_candidates),
                    "plan_hash": str(plan["plan_hash"]),
                    "stage": "inner_validation_failed",
                },
            )
            return failed_result
        kept_candidates = valid_ordered_candidates[: int(rung["keep_config_count"])]
        active_configs = [dict(candidate["config"]) for candidate in kept_candidates]
        final_ordered_candidates = valid_ordered_candidates
        rung_summary = {
            "candidate_count": len(rung_candidates),
            "candidate_hashes_hash": stable_hash_json([
                stable_hash_json({
                    key: value
                    for key, value in candidate.items()
                    if key != "config"
                })
                for candidate in ordered_candidates
            ]),
            "kept_config_hashes": [
                str(candidate["config_hash"]) for candidate in kept_candidates
            ],
            "kept_config_count": len(kept_candidates),
            "rung_index": rung_index,
            "rung_job_count": rung_job_count,
        }
        rung_results.append(rung_summary)
        record_progress_event(
            progress_log_path,
            event="inner_validation_rung_completed",
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            extra=rung_summary,
        )
    if not all_candidates:
        raise ValueError("inner validation produced no candidates")
    if not final_ordered_candidates:
        raise ValueError("inner validation produced no valid final rung candidates")
    best_candidate = final_ordered_candidates[0]
    result = {
        "best_candidate": best_candidate,
        "candidate_count": len(all_candidates),
        "experiment_variant": str(experiment_variant),
        "plan": plan,
        "rungs": rung_results,
        "stage": "inner_validation_completed",
    }
    record_progress_event(
        progress_log_path,
        event="inner_validation_completed",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "best_config_hash": str(best_candidate["config_hash"]),
            "best_config_index": int(best_candidate["config_index"]),
            "candidate_count": len(all_candidates),
            "experiment_variant": str(experiment_variant),
            "plan_hash": str(plan["plan_hash"]),
            "stage": "inner_validation_completed",
        },
    )
    return result


def evaluate_v25_bounded_development_dry_run_with_progress(
    *,
    jobs: Sequence[Mapping[str, Any]],
    max_jobs: int,
    train_stats: Mapping[str, Any],
    config: Mapping[str, Any],
    norm_cap: float,
    spectral_basis: torch.Tensor,
    control_context_by_record_hash: Mapping[str, Mapping[str, Any]],
    selected_config_hash: str,
    script_sha256: str,
    progress_log_path: Path,
    started_at_monotonic: float,
    now_monotonic: Any | None = None,
) -> dict[str, Any]:
    for record_hash, context in control_context_by_record_hash.items():
        require_sha256_hex(record_hash, field_name="record_id_hash")
        provenance = context.get("provenance", {})
        if not isinstance(provenance, Mapping) or not bool(provenance.get("dry_run_only")):
            raise ValueError("bounded dry run requires dry-run-only control contexts")
        if bool(provenance.get("proof_valid", True)):
            raise ValueError("bounded dry run control contexts must be proof invalid")
    record_progress_event(
        progress_log_path,
        event="development_bounded_dry_run_start",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "control_context_mode": "placeholder_zero_controls",
            "max_jobs": int(max_jobs),
            "proof_valid": False,
            "total_planned_jobs": len(jobs),
        },
    )
    evaluation = evaluate_v25_development_jobs_with_progress(
        jobs=jobs,
        max_jobs=max_jobs,
        train_stats=train_stats,
        config=config,
        norm_cap=norm_cap,
        spectral_basis=spectral_basis,
        control_context_by_record_hash=control_context_by_record_hash,
        selected_config_hash=selected_config_hash,
        script_sha256=script_sha256,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
    )
    result = {
        "control_context_mode": "placeholder_zero_controls",
        "evaluated_count": int(evaluation["evaluated_count"]),
        "max_jobs": int(evaluation["max_jobs"]),
        "proof_record_hashes": list(evaluation["proof_record_hashes"]),
        "proof_valid": False,
        "stage": "development_bounded_dry_run_completed",
        "total_planned_jobs": int(evaluation["total_planned_jobs"]),
    }
    record_progress_event(
        progress_log_path,
        event="development_bounded_dry_run_completed",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra=result,
    )
    return {
        **result,
        "proof_records": evaluation["proof_records"],
    }


def mean_float(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def summarize_v25_proof_gate_breakdown(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failure_type_counts = {control_type: 0 for control_type in PROOF_CRITICAL_CONTROL_TYPES}
    target_prediction_fail_count = 0
    target_margin_fail_count = 0
    pareto_fail_count = 0
    compatible_mse_fail_count = 0
    control_margin_fail_count = 0
    control_margin_record_fail_count = 0
    min_advantages = []
    mean_advantages = []
    for record in records:
        diagnostics = record["summary"]["proof_gate_diagnostics"]
        if not bool(diagnostics["target_prediction_pass"]):
            target_prediction_fail_count += 1
        if not bool(diagnostics["target_margin_pass"]):
            target_margin_fail_count += 1
        if not bool(diagnostics["pareto_undominated"]):
            pareto_fail_count += 1
        if not bool(diagnostics["compatible_mse_pass"]):
            compatible_mse_fail_count += 1
        record_control_fail_count = int(diagnostics["control_margin_fail_count"])
        control_margin_fail_count += record_control_fail_count
        if record_control_fail_count:
            control_margin_record_fail_count += 1
        for control_type in PROOF_CRITICAL_CONTROL_TYPES:
            if (
                float(record["summary"][f"matched_minus_{control_type}_target_margin"])
                < (
                    PER_RECORD_MIN_SHUFFLED_MARGIN_ADVANTAGE
                    if control_type == "shuffled_signature"
                    else PER_RECORD_MIN_CONTROL_MARGIN_ADVANTAGE
                )
            ):
                failure_type_counts[control_type] += 1
        min_advantages.append(float(diagnostics["min_control_margin_advantage"]))
        mean_advantages.append(float(diagnostics["mean_control_margin_advantage"]))
    return {
        "compatible_mse_fail_count": compatible_mse_fail_count,
        "control_margin_fail_count": control_margin_fail_count,
        "control_margin_failure_record_count": control_margin_record_fail_count,
        "control_margin_failure_type_counts_hash": stable_hash_json(failure_type_counts),
        "mean_control_margin_advantage": mean_float(mean_advantages),
        "min_control_margin_advantage": min(min_advantages) if min_advantages else 0.0,
        "pareto_fail_count": pareto_fail_count,
        "record_count": len(records),
        "target_margin_fail_count": target_margin_fail_count,
        "target_prediction_fail_count": target_prediction_fail_count,
    }


def summarize_v25_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_record_count: int,
) -> dict[str, Any]:
    records = list(records)
    failures = []
    if len(records) != int(expected_record_count):
        failures.append(
            f"record count mismatch: expected {expected_record_count}, got {len(records)}"
        )
    finite_failures = recursive_numeric_finiteness_failures(records)
    if finite_failures:
        failures.append("nonfinite record metrics: " + ", ".join(finite_failures[:5]))
    if not records:
        aggregate = {
            "individual_all_gate_pass_rate": 0.0,
            "mean_matched_minus_best_control_target_margin": 0.0,
            "mean_matched_minus_shuffled_signature_target_margin": 0.0,
            "mean_target_margin": 0.0,
            "pareto_undominated_rate": 0.0,
            "proof_gate_breakdown": summarize_v25_proof_gate_breakdown([]),
            "target_prediction_rate": 0.0,
        }
        return {
            "aggregate": aggregate,
            "by_direction": {},
            "failures": failures,
            "record_count": 0,
            "records": [],
        }
    aggregate = {
        "individual_all_gate_pass_rate": mean_float([
            1.0 if record["summary"]["individual_all_gates_passed"] else 0.0
            for record in records
        ]),
        "mean_matched_minus_best_control_target_margin": mean_float([
            float(record["summary"]["matched_minus_best_control_target_margin"])
            for record in records
        ]),
        "mean_target_margin": mean_float([
            float(record["matched"]["target_margin"]) for record in records
        ]),
        "pareto_undominated_rate": mean_float([
            1.0 if record["summary"]["pareto_undominated"] else 0.0
            for record in records
        ]),
        "proof_gate_breakdown": summarize_v25_proof_gate_breakdown(records),
        "target_prediction_rate": mean_float([
            1.0 if record["summary"]["target_prediction_pass"] else 0.0
            for record in records
        ]),
    }
    for control_type in PROOF_CRITICAL_CONTROL_TYPES:
        aggregate[f"mean_matched_minus_{control_type}_target_margin"] = mean_float([
            float(record["summary"][f"matched_minus_{control_type}_target_margin"])
            for record in records
        ])
    aggregate["mean_matched_minus_shuffled_signature_target_margin"] = aggregate[
        "mean_matched_minus_shuffled_signature_target_margin"
    ]
    by_direction: dict[str, dict[str, Any]] = {}
    for record in records:
        direction = str(record["direction"])
        by_direction.setdefault(direction, {"records": []})["records"].append(record)
    by_direction_summary = {}
    for direction, payload in sorted(by_direction.items()):
        direction_records = payload["records"]
        by_direction_summary[direction] = {
            "individual_all_gate_pass_rate": mean_float([
                1.0 if record["summary"]["individual_all_gates_passed"] else 0.0
                for record in direction_records
            ]),
            "mean_target_margin": mean_float([
                float(record["matched"]["target_margin"]) for record in direction_records
            ]),
            "pareto_undominated_rate": mean_float([
                1.0 if record["summary"]["pareto_undominated"] else 0.0
                for record in direction_records
            ]),
            "record_count": len(direction_records),
            "target_prediction_rate": mean_float([
                1.0 if record["summary"]["target_prediction_pass"] else 0.0
                for record in direction_records
            ]),
        }
    for metric_name, threshold in AGGREGATE_PROOF_GATES.items():
        if float(aggregate.get(metric_name, 0.0)) < float(threshold):
            failures.append(
                f"aggregate {metric_name} below gate: "
                f"{float(aggregate.get(metric_name, 0.0)):.6f} < {float(threshold):.6f}"
            )
    for direction, summary in by_direction_summary.items():
        if float(summary["target_prediction_rate"]) < 0.65:
            failures.append(f"{direction} target prediction rate below gate")
        if float(summary["pareto_undominated_rate"]) < 0.75:
            failures.append(f"{direction} pareto undominated rate below gate")
        if float(summary["mean_target_margin"]) < 0.15:
            failures.append(f"{direction} target margin below gate")
    return {
        "aggregate": aggregate,
        "by_direction": by_direction_summary,
        "failures": failures,
        "record_count": len(records),
        "records": list(records),
    }


def summarize_v25_inner_validation_candidate(
    *,
    config: Mapping[str, Any],
    config_hash: str,
    proof_records: Sequence[Mapping[str, Any]],
    expected_record_count: int,
) -> dict[str, Any]:
    checked_config_hash = require_sha256_hex(config_hash, field_name="config_hash")
    proof_records = list(proof_records)
    contract_failures = []
    if len(proof_records) != int(expected_record_count):
        contract_failures.append(
            f"record count mismatch: expected {expected_record_count}, got {len(proof_records)}"
        )
    finite_failures = recursive_numeric_finiteness_failures(proof_records)
    if finite_failures:
        contract_failures.append("nonfinite record metrics: " + ", ".join(finite_failures[:5]))
    summary = summarize_v25_records(
        proof_records,
        expected_record_count=int(expected_record_count),
    )
    aggregate = dict(summary["aggregate"])
    proof_record_hashes = [
        stable_hash_json(proof_record)
        for proof_record in proof_records
    ]
    failures = list(summary["failures"])
    proof_gate_failures = [
        failure for failure in failures
        if failure not in set(contract_failures)
    ]
    return {
        "config": dict(config),
        "config_hash": checked_config_hash,
        "config_index": int(config["config_index"]),
        "contract_failure_count": len(contract_failures),
        "invalid": bool(contract_failures),
        "mean_matched_minus_best_control_target_margin": float(
            aggregate["mean_matched_minus_best_control_target_margin"]
        ),
        "mean_matched_minus_shuffled_signature_target_margin": float(
            aggregate["mean_matched_minus_shuffled_signature_target_margin"]
        ),
        "mean_target_margin": float(aggregate["mean_target_margin"]),
        "pareto_undominated_rate": float(aggregate["pareto_undominated_rate"]),
        "proof_gate_breakdown": dict(aggregate["proof_gate_breakdown"]),
        "proof_gate_failure_count": len(proof_gate_failures),
        "proof_record_hashes_hash": stable_hash_json(proof_record_hashes),
        "record_count": int(summary["record_count"]),
        "target_prediction_rate": float(aggregate["target_prediction_rate"]),
    }


def compute_train_spectral_basis(
    deltas: torch.Tensor,
    *,
    rank: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    deltas = deltas.to(dtype=torch.float32)
    if deltas.ndim != 2:
        raise ValueError("deltas must be rank-2")
    if int(rank) <= 0 or int(rank) > int(min(deltas.shape)):
        raise ValueError("rank must be positive and no larger than min(deltas.shape)")
    if not torch.isfinite(deltas).all():
        raise ValueError("nonfinite spectral basis input")
    centered = deltas - deltas.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[: int(rank)].T.contiguous().to(dtype=torch.float32)
    audit = {
        "basis_sha256": stable_hash_json(basis.detach().cpu().tolist()),
        "centered_delta_sha256": stable_hash_json(centered.detach().cpu().tolist()),
        "delta_count": int(deltas.shape[0]),
        "explained_singular_values": singular_values[: int(rank)].detach().cpu().tolist(),
        "rank": int(rank),
    }
    return basis, audit


def stdout_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    def location_sha256(value: Any) -> str | None:
        if not value:
            return None
        return stable_hash_json(str(Path(str(value)).resolve()))

    def compact_pool_summary(pool_summary: Mapping[str, Any]) -> dict[str, Any]:
        allowed_keys = [
            "accepted_counts_by_behavior",
            "attempt_counts_by_behavior",
            "max_selected_train_vs_heldout_overlap_count",
            "pool_file_sha256",
            "pool_redacted_payload_sha256",
            "record_count",
        ]
        return {
            key: pool_summary[key]
            for key in allowed_keys
            if key in pool_summary
        }

    summary = {
        "long_run_monitor_log_sha256": result.get("long_run_monitor_log_sha256"),
        "passed": bool(result.get("passed", False)),
        "long_run_monitor_log_location_sha256": location_sha256(
            result.get("long_run_monitor_log_path")
        ),
        "pool_dir_location_sha256": location_sha256(result.get("pool_dir")),
        "source_pool_progress_log_sha256": result.get("source_pool_progress_log_sha256"),
        "source_pool_progress_log_location_sha256": location_sha256(
            result.get("source_pool_progress_log_path")
        ),
    }
    if "pool_summaries" in result:
        summary["pool_summaries"] = {
            str(pool_name): compact_pool_summary(pool_summary)
            for pool_name, pool_summary in result["pool_summaries"].items()
        }
    if "seed_preflight" in result:
        seed_preflight = result["seed_preflight"]
        summary["preflight_failure_count"] = len(seed_preflight.get("failures", []))
        summary["preflight_passed"] = bool(seed_preflight.get("passed", False))
    for key in [
        "descriptor_norm_hash",
        "development_evaluation",
        "development_job_selection",
        "development_jobs",
        "development_pool",
        "development_progress_log_sha256",
        "dry_run",
        "inner_validation",
        "probe_examples_hash",
        "stage",
        "train_edit_bank",
        "train_edit_spectral_basis",
        "train_pool",
        "train_statistics_hash",
    ]:
        if key in result:
            summary[key] = result[key]
    return summary


def parse_csv_ints(value: str, *, field_name: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{field_name} must be a comma-separated integer list") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{field_name} must contain positive integers")
    return parsed


def parse_csv_floats(value: str, *, field_name: str) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{field_name} must be a comma-separated float list") from error
    if not parsed or any(item <= 0.0 or item > 1.0 for item in parsed):
        raise ValueError(f"{field_name} must contain floats in (0, 1]")
    return parsed


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
    parser.add_argument("--max-development-jobs", type=int, default=0)
    parser.add_argument(
        "--development-job-selection",
        choices=["prefix", "balanced-directions"],
        default="prefix",
    )
    parser.add_argument("--run-inner-validation", action="store_true")
    parser.add_argument("--inner-validation-rung-jobs", default="4,12")
    parser.add_argument("--inner-validation-keep-fractions", default="0.25,0.25")
    parser.add_argument("--inner-validation-max-configs", type=int, default=None)
    parser.add_argument(
        "--inner-validation-config-grid",
        choices=[
            "v25",
            "v26",
            "v27-localized",
            "v28-anchor-nullspace",
            "v29-breadth-first-sparse",
            "v30-margin-gated-sparse",
            "v31-orthogonal-sign-sparse",
            "v32-support-tournament-margin",
            "v33-proof-gate-diagnostic",
            "v34-locality-pressure",
            "v35-support-source-line-search",
            "v36-compatible-nullspace-projection",
            "v37-projected-support-optimizer",
            "v38-compatible-mse-gated-projected-optimizer",
            "v39-target-feasible-lexicographic-projected-optimizer",
            "v40-target-tolerance-locality-budget-projected-optimizer",
            "v41-trajectory-frontier-projected-optimizer",
            "v42-compatible-dual-frontier",
        ],
        default="v25",
    )
    parser.add_argument("--dry-run-placeholder-controls", action="store_true")
    parser.add_argument("--monitor-interval-seconds", type=float, default=30.0)
    parser.add_argument("--summary-only-stdout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool_dir = REPO_ROOT / args.pool_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_dir = pool_dir if args.phase == "generate-pools" else output_dir
    phase_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = (
        pool_dir / SOURCE_POOL_PROGRESS_LOG_FILENAME
        if args.phase == "generate-pools"
        else output_dir / f"{args.phase}_progress.jsonl"
    )
    monitor_log_path = phase_dir / LONG_RUN_MONITOR_LOG_FILENAME
    if args.phase == "development":
        progress_log_path.unlink(missing_ok=True)
        monitor_log_path.unlink(missing_ok=True)
    monitor = LongRunMonitor(
        monitor_log_path=monitor_log_path,
        progress_log_path=progress_log_path,
        interval_seconds=args.monitor_interval_seconds,
    )
    monitor.start()
    result: dict[str, Any]
    try:
        if args.phase == "generate-pools":
            result = generate_pools(args, pool_dir)
        elif args.phase == "development":
            result = run_v25_development_setup(
                pool_dir=pool_dir,
                output_dir=output_dir,
                progress_log_path=progress_log_path,
                started_at_monotonic=monitor.started_at_monotonic,
                max_development_jobs=int(args.max_development_jobs),
                dry_run_placeholder_controls=bool(args.dry_run_placeholder_controls),
                development_job_selection=str(args.development_job_selection),
                run_inner_validation=bool(args.run_inner_validation),
                inner_validation_rung_jobs=parse_csv_ints(
                    args.inner_validation_rung_jobs,
                    field_name="inner_validation_rung_jobs",
                ),
                inner_validation_keep_fractions=parse_csv_floats(
                    args.inner_validation_keep_fractions,
                    field_name="inner_validation_keep_fractions",
                ),
                inner_validation_max_configs=args.inner_validation_max_configs,
                inner_validation_config_grid=str(args.inner_validation_config_grid),
            )
        else:
            raise SystemExit(
                "V25 final phase is blocked until development passes and a "
                "hash-bound reviewer authorization is recorded"
            )
    finally:
        monitor.stop()
    result["long_run_monitor_log_path"] = str(monitor_log_path)
    result["long_run_monitor_log_sha256"] = sha256_file(monitor_log_path)
    if args.summary_only_stdout:
        result = stdout_summary(result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
