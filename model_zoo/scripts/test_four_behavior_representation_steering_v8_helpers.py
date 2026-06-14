"""Direct helper tests for four-behavior representation steering V8."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_representation_steering_v8_source_conditional_tournament_correction as steer_v8  # noqa: E402


def test_v8_seed_preflight_uses_preregistered_base_seed() -> None:
    preflight = steer_v8.build_seed_preflight(
        steer_v8.POOL_CONFIGS,
        behavior_stride=steer_v8.SEED_BEHAVIOR_STRIDE,
    )
    assert preflight["passed"], preflight["failures"]
    assert preflight["seed_ranges"][0]["start_seed"] == 54300000


def test_v8_rejects_prior_pool_directories() -> None:
    pool_dirs = [
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v1_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v2_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v3_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v4_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v5_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v6_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v7_pools",
    ]
    for pool_dir in pool_dirs:
        try:
            steer_v8.assert_preregistered_pool_dir(pool_dir)
        except ValueError as error:
            assert "V8" in str(error)
        else:
            raise AssertionError(f"V8 accepted forbidden prior pool directory: {pool_dir}")


def test_v8_tournament_sources_exclude_matched_and_target() -> None:
    assert steer_v8.same_target_tournament_sources(
        source="mountain_pattern",
        target="has_majority",
    ) == ["sorted_ascending", "sorted_descending"]


def test_v8_tournament_competitors_are_ten_detached_candidates() -> None:
    z = torch.zeros(4)
    stats = {
        "calibration_coefficients": {
            steer_v8.v1.vector_key(source, target): torch.zeros(2)
            for source in steer_v8.PATTERNS
            for target in steer_v8.PATTERNS
            if source != target
        },
        "centroids": {pattern: torch.ones(4) * index for index, pattern in enumerate(steer_v8.PATTERNS)},
        "inv_sqrt_cov": {pattern: torch.eye(2) for pattern in steer_v8.PATTERNS},
        "pca_components": torch.eye(4, 2),
        "sqrt_cov": {pattern: torch.eye(2) for pattern in steer_v8.PATTERNS},
        "transport_stds": {pattern: torch.ones(4) for pattern in steer_v8.PATTERNS},
    }
    classifier = torch.nn.Linear(4, len(steer_v8.PATTERNS))
    competitors = steer_v8.build_same_target_tournament_competitors(
        z=z,
        source="mountain_pattern",
        target="has_majority",
        stats=stats,
        classifier=classifier,
    )
    assert len(competitors) == 10
    assert all(not candidate.requires_grad for candidate in competitors)


def test_v8_summary_and_gates_use_selected_pareto_metrics() -> None:
    records = []
    index = 0
    for source in steer_v8.PATTERNS:
        for target in steer_v8.PATTERNS:
            if source == target:
                continue
            for _ in range(24):
                summary = {
                    "pareto_undominated": True,
                    "selected_centroid_improvement": 0.40,
                    "selected_minus_best_control_centroid_improvement": 0.02,
                    "selected_minus_best_control_primary_target_margin": 0.02,
                    "selected_minus_v2_centroid_delta_centroid_improvement": 0.20,
                    "selected_minus_v2_centroid_delta_primary_target_margin": 0.30,
                    "selected_minus_v3_diagonal_transport_centroid_improvement": 0.20,
                    "selected_minus_v3_diagonal_transport_primary_target_margin": 0.30,
                    "selected_minus_v4_low_rank_centroid_improvement": 0.20,
                    "selected_minus_v4_low_rank_primary_target_margin": 0.30,
                    "selected_minus_v5_calibrated_centroid_improvement": 0.20,
                    "selected_minus_v5_calibrated_primary_target_margin": -5.0,
                    "selected_minus_v6_correction_centroid_improvement": 0.20,
                    "selected_minus_v6_correction_primary_target_margin": 0.30,
                    "selected_primary_target_margin": 0.60,
                    "source_primary_margin_change": -0.20,
                    "target_prediction_pass": True,
                }
                records.append({
                    "individual_all_gates_passed": True,
                    "matched": {
                        "centroid_predicted_behavior": target,
                        "pareto_undominated": True,
                        "primary_predicted_behavior": target,
                    },
                    "source_behavior": source,
                    "subject_id": f"test-{index}",
                    "summary": summary,
                    "target_behavior": target,
                })
                index += 1
    aggregate = steer_v8.summarize_records(records)
    assert aggregate["individual_all_gate_pass_count"] == 288
    assert aggregate["pareto_undominated_count"] == 288
    assert "mean_matched_primary_target_margin" not in aggregate


def main() -> None:
    tests = [
        test_v8_seed_preflight_uses_preregistered_base_seed,
        test_v8_rejects_prior_pool_directories,
        test_v8_tournament_sources_exclude_matched_and_target,
        test_v8_tournament_competitors_are_ten_detached_candidates,
        test_v8_summary_and_gates_use_selected_pareto_metrics,
    ]
    for test in tests:
        test()
    print("four-behavior representation steering V8 helper tests passed")


if __name__ == "__main__":
    main()
