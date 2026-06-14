"""Direct helper tests for four-behavior representation steering V7."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_representation_steering_v7_pareto_frontier_correction as steer_v7  # noqa: E402


def test_v7_rejects_prior_pool_directories() -> None:
    pool_dirs = [
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v1_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v2_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v3_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v4_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v5_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v6_pools",
    ]
    for pool_dir in pool_dirs:
        try:
            steer_v7.assert_preregistered_pool_dir(pool_dir)
        except ValueError as error:
            assert "V7" in str(error)
        else:
            raise AssertionError(f"V7 accepted forbidden prior pool directory: {pool_dir}")


def test_v7_seed_preflight_uses_preregistered_base_seed() -> None:
    preflight = steer_v7.build_seed_preflight(
        steer_v7.POOL_CONFIGS,
        behavior_stride=steer_v7.SEED_BEHAVIOR_STRIDE,
    )
    assert preflight["passed"], preflight["failures"]
    assert preflight["seed_ranges"][0]["start_seed"] == 51300000


def test_v7_final_redaction_rejects_forbidden_final_detail() -> None:
    combined = {
        "claim_scope": "four_behavior_representation_steering_v7_source_pool_construction",
        "overlap_counts": {},
        "passed": True,
        "pool_summaries": {
            "train": {"accepted_counts_by_behavior": {pattern: 64 for pattern in steer_v7.PATTERNS}},
            "development": {
                "accepted_counts_by_behavior": {pattern: 24 for pattern in steer_v7.PATTERNS}
            },
            "final": {
                "accepted_counts_by_behavior": {pattern: 24 for pattern in steer_v7.PATTERNS},
                "subject_id": "leaked-final-subject",
            },
        },
        "seed_preflight": {"passed": True},
    }
    final_redacted = {
        "claim_scope": "redacted_final_steering_v7_source_pool_audit_surface_only",
        "summary": {
            "accepted_counts_by_behavior": {pattern: 24 for pattern in steer_v7.PATTERNS},
            "max_selected_train_vs_heldout_overlap_count": 0,
        },
    }
    failures = steer_v7.validate_source_pool_contract(
        train_payload={
            "claim_scope": "four_behavior_representation_steering_v7_source_pool",
            "pool": "train",
        },
        eval_payload={
            "claim_scope": "four_behavior_representation_steering_v7_source_pool",
            "pool": "development",
        },
        combined_audit=combined,
        final_redacted=final_redacted,
        phase="development",
    )
    assert any("forbidden final detail" in failure for failure in failures)


def test_v7_pareto_dominance_uses_two_metrics_with_strict_epsilon() -> None:
    matched = {"primary_target_margin": 1.0, "centroid_improvement": 2.0}
    assert steer_v7.pareto_dominates(
        {"primary_target_margin": 1.0, "centroid_improvement": 2.1},
        matched,
    )
    assert steer_v7.pareto_dominates(
        {"primary_target_margin": 1.1, "centroid_improvement": 2.0},
        matched,
    )
    assert not steer_v7.pareto_dominates(
        {"primary_target_margin": 1.0, "centroid_improvement": 2.0},
        matched,
    )
    assert not steer_v7.pareto_dominates(
        {"primary_target_margin": 1.2, "centroid_improvement": 1.9},
        matched,
    )


def test_v7_radius_budgets_are_clamped_to_required_centroid_improvement() -> None:
    budgets = steer_v7.compute_radius_budgets(
        source_distance=10.0,
        v4_distance=9.0,
        v5_distance=20.0,
    )
    assert len(budgets) == 5
    assert max(budgets) <= 9.85
    assert budgets[0] == 8.95
    assert budgets[-1] == 9.85


def test_v7_project_ball_handles_inside_boundary_and_zero_distance() -> None:
    center = torch.tensor([1.0, 2.0])
    inside = torch.tensor([1.2, 2.0])
    assert torch.allclose(steer_v7.project_ball(inside, center=center, radius=1.0), inside)
    outside = torch.tensor([4.0, 2.0])
    assert torch.allclose(
        steer_v7.project_ball(outside, center=center, radius=2.0),
        torch.tensor([3.0, 2.0]),
    )
    assert torch.allclose(steer_v7.project_ball(center, center=center, radius=0.0), center)


def make_v7_passing_record(index: int, source: str, target: str) -> dict:
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
    return {
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
    }


def test_v7_summary_and_gates_use_selected_pareto_metrics() -> None:
    records = []
    index = 0
    for source in steer_v7.PATTERNS:
        for target in steer_v7.PATTERNS:
            if source == target:
                continue
            for _ in range(24):
                records.append(make_v7_passing_record(index, source, target))
                index += 1
    aggregate = steer_v7.summarize_records(records)
    assert aggregate["n"] == 288
    assert aggregate["individual_all_gate_pass_rate"] == 1.0
    assert aggregate["pareto_undominated_rate"] == 1.0
    assert aggregate["target_prediction_pass_count"] == 288
    assert abs(aggregate["mean_selected_primary_target_margin"] - 0.60) < 1e-12
    assert "mean_matched_primary_target_margin" not in aggregate

    by_target = {
        target: steer_v7.summarize_records([
            record for record in records if record["target_behavior"] == target
        ])
        for target in steer_v7.PATTERNS
    }
    by_direction = {
        steer_v7.v1.vector_key(source, target): steer_v7.summarize_records([
            record for record in records
            if record["source_behavior"] == source and record["target_behavior"] == target
        ])
        for source in steer_v7.PATTERNS
        for target in steer_v7.PATTERNS
        if source != target
    }
    audit = steer_v7.individual_gate_audit(records)
    failures = steer_v7.gate_failures(
        aggregate=aggregate,
        by_target=by_target,
        by_direction=by_direction,
        individual_audit=audit,
    )
    assert failures == []


def main() -> None:
    tests = [
        test_v7_rejects_prior_pool_directories,
        test_v7_seed_preflight_uses_preregistered_base_seed,
        test_v7_final_redaction_rejects_forbidden_final_detail,
        test_v7_pareto_dominance_uses_two_metrics_with_strict_epsilon,
        test_v7_radius_budgets_are_clamped_to_required_centroid_improvement,
        test_v7_project_ball_handles_inside_boundary_and_zero_distance,
        test_v7_summary_and_gates_use_selected_pareto_metrics,
    ]
    for test in tests:
        test()
    print("four-behavior representation steering V7 helper tests passed")


if __name__ == "__main__":
    main()
