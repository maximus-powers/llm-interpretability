"""Direct helper tests for four-behavior representation steering V9."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_representation_steering_v9_source_invariant_target_attractor as steer_v9  # noqa: E402


def test_v9_seed_preflight_uses_preregistered_base_seed() -> None:
    preflight = steer_v9.build_seed_preflight(
        steer_v9.POOL_CONFIGS,
        behavior_stride=steer_v9.SEED_BEHAVIOR_STRIDE,
    )
    assert preflight["passed"], preflight["failures"]
    assert preflight["seed_ranges"][0]["start_seed"] == 57300000


def test_v9_rejects_prior_pool_directories() -> None:
    for index in range(1, 9):
        pool_dir = REPO_ROOT / "runs" / f"four_behavior_representation_steering_v{index}_pools"
        try:
            steer_v9.assert_preregistered_pool_dir(pool_dir)
        except ValueError as error:
            assert "V9" in str(error)
        else:
            raise AssertionError(f"V9 accepted forbidden prior pool directory: {pool_dir}")


def test_v9_classifies_same_target_other_source_as_transfer_probe() -> None:
    assert steer_v9.is_transfer_probe_direction(
        matched_source="mountain_pattern",
        matched_target="has_majority",
        control_source="sorted_ascending",
        control_target="has_majority",
    )
    assert not steer_v9.is_transfer_probe_direction(
        matched_source="mountain_pattern",
        matched_target="has_majority",
        control_source="mountain_pattern",
        control_target="sorted_ascending",
    )
    assert not steer_v9.is_transfer_probe_direction(
        matched_source="mountain_pattern",
        matched_target="has_majority",
        control_source="has_majority",
        control_target="mountain_pattern",
    )


def test_v9_split_control_specs_excludes_only_transfer_probes_from_dominance() -> None:
    specs = [
        {
            "control_type": "same_target_other_source_v9_source_invariant_target_attractor",
            "metadata": {
                "control_source_behavior": "sorted_ascending",
                "control_target_behavior": "has_majority",
            },
        },
        {
            "control_type": "same_source_other_target_v9_source_invariant_target_attractor",
            "metadata": {
                "control_source_behavior": "mountain_pattern",
                "control_target_behavior": "sorted_ascending",
            },
        },
        {
            "control_type": "reverse_v9_source_invariant_target_attractor",
            "metadata": {
                "control_source_behavior": "has_majority",
                "control_target_behavior": "mountain_pattern",
            },
        },
    ]
    controls, probes = steer_v9.split_negative_controls_and_transfer_probes(
        specs,
        source="mountain_pattern",
        target="has_majority",
    )
    assert [item["control_type"] for item in probes] == [
        "same_target_other_source_v9_source_invariant_target_attractor"
    ]
    assert [item["control_type"] for item in controls] == [
        "same_source_other_target_v9_source_invariant_target_attractor",
        "reverse_v9_source_invariant_target_attractor",
    ]


def test_v9_shuffled_same_target_direction_remains_negative_control() -> None:
    specs = [
        {
            "control_type": "shuffled_v9_source_invariant_target_attractor",
            "metadata": {
                "control_source_behavior": "sorted_ascending",
                "control_target_behavior": "has_majority",
            },
        },
        {
            "control_type": "same_target_other_source_v9_source_invariant_target_attractor",
            "metadata": {
                "control_source_behavior": "sorted_descending",
                "control_target_behavior": "has_majority",
            },
        },
    ]

    controls, probes = steer_v9.split_negative_controls_and_transfer_probes(
        specs,
        source="mountain_pattern",
        target="has_majority",
    )

    assert [item["control_type"] for item in controls] == [
        "shuffled_v9_source_invariant_target_attractor"
    ]
    assert [item["control_type"] for item in probes] == [
        "same_target_other_source_v9_source_invariant_target_attractor"
    ]


def test_v9_transfer_summary_uses_all_probe_candidates() -> None:
    probes = [
        {
            "centroid_improvement": 0.3,
            "centroid_predicted_behavior": "has_majority",
            "primary_predicted_behavior": "has_majority",
            "primary_source_margin_change": -0.2,
            "primary_target_margin": 0.5,
        },
        {
            "centroid_improvement": 0.1,
            "centroid_predicted_behavior": "sorted_ascending",
            "primary_predicted_behavior": "has_majority",
            "primary_source_margin_change": -0.2,
            "primary_target_margin": 0.5,
        },
    ]
    summary = steer_v9.summarize_transfer_probes(probes, target="has_majority")
    assert summary["same_target_transfer_probe_count"] == 2
    assert summary["same_target_transfer_target_prediction_count"] == 1
    assert summary["same_target_transfer_gate_pass_count"] == 1
    assert summary["same_target_transfer_target_prediction_rate"] == 0.5
    assert summary["same_target_transfer_gate_pass_rate"] == 0.5


def test_v9_aggregate_summary_counts_transfer_probe_rates() -> None:
    records = [
        {
            "individual_all_gates_passed": True,
            "summary": {
                "pareto_undominated": True,
                "same_target_transfer_gate_pass_count": 1,
                "same_target_transfer_probe_count": 2,
                "same_target_transfer_target_prediction_count": 2,
                "selected_centroid_improvement": 3.0,
                "selected_minus_best_control_centroid_improvement": 0.2,
                "selected_minus_best_control_primary_target_margin": 0.3,
                "selected_minus_v2_centroid_delta_centroid_improvement": 0.4,
                "selected_minus_v2_centroid_delta_primary_target_margin": 0.5,
                "selected_minus_v3_diagonal_transport_centroid_improvement": 0.6,
                "selected_minus_v3_diagonal_transport_primary_target_margin": 0.7,
                "selected_minus_v4_low_rank_centroid_improvement": 0.8,
                "selected_minus_v4_low_rank_primary_target_margin": 0.9,
                "selected_minus_v5_calibrated_centroid_improvement": 1.0,
                "selected_minus_v5_calibrated_primary_target_margin": 1.1,
                "selected_minus_v6_correction_centroid_improvement": 1.2,
                "selected_minus_v6_correction_primary_target_margin": 1.3,
                "selected_primary_target_margin": 2.0,
                "source_primary_margin_change": -1.0,
                "target_prediction_pass": True,
            },
        },
        {
            "individual_all_gates_passed": False,
            "summary": {
                "pareto_undominated": False,
                "same_target_transfer_gate_pass_count": 2,
                "same_target_transfer_probe_count": 2,
                "same_target_transfer_target_prediction_count": 2,
                "selected_centroid_improvement": 4.0,
                "selected_minus_best_control_centroid_improvement": 0.3,
                "selected_minus_best_control_primary_target_margin": 0.4,
                "selected_minus_v2_centroid_delta_centroid_improvement": 0.5,
                "selected_minus_v2_centroid_delta_primary_target_margin": 0.6,
                "selected_minus_v3_diagonal_transport_centroid_improvement": 0.7,
                "selected_minus_v3_diagonal_transport_primary_target_margin": 0.8,
                "selected_minus_v4_low_rank_centroid_improvement": 0.9,
                "selected_minus_v4_low_rank_primary_target_margin": 1.0,
                "selected_minus_v5_calibrated_centroid_improvement": 1.1,
                "selected_minus_v5_calibrated_primary_target_margin": 1.2,
                "selected_minus_v6_correction_centroid_improvement": 1.3,
                "selected_minus_v6_correction_primary_target_margin": 1.4,
                "selected_primary_target_margin": 3.0,
                "source_primary_margin_change": -2.0,
                "target_prediction_pass": True,
            },
        },
    ]

    summary = steer_v9.summarize_records(records)

    assert summary["same_target_transfer_probe_count"] == 4
    assert summary["same_target_transfer_target_prediction_count"] == 4
    assert summary["same_target_transfer_gate_pass_count"] == 3
    assert summary["same_target_transfer_target_prediction_rate"] == 1.0
    assert summary["same_target_transfer_gate_pass_rate"] == 0.75


def test_v9_gate_failures_include_transfer_probe_thresholds() -> None:
    aggregate = {
        "individual_all_gate_pass_rate": 1.0,
        "mean_selected_centroid_improvement": 10.0,
        "mean_selected_minus_v2_centroid_delta_centroid_improvement": 10.0,
        "mean_selected_minus_v2_centroid_delta_primary_target_margin": 10.0,
        "mean_selected_minus_v3_diagonal_transport_centroid_improvement": 10.0,
        "mean_selected_minus_v3_diagonal_transport_primary_target_margin": 10.0,
        "mean_selected_minus_v4_low_rank_centroid_improvement": 10.0,
        "mean_selected_minus_v4_low_rank_primary_target_margin": 10.0,
        "mean_selected_minus_v5_calibrated_centroid_improvement": 10.0,
        "mean_selected_minus_v6_correction_centroid_improvement": 10.0,
        "mean_selected_minus_v6_correction_primary_target_margin": 10.0,
        "mean_selected_primary_target_margin": 10.0,
        "mean_source_primary_margin_change": -10.0,
        "n": steer_v9.THRESHOLDS["expected_record_count"],
        "pareto_undominated_rate": 1.0,
        "same_target_transfer_gate_pass_rate": 0.75,
        "same_target_transfer_target_prediction_rate": 0.85,
    }
    target_summary = {
        "individual_all_gate_pass_rate": 1.0,
        "n": steer_v9.THRESHOLDS["expected_per_target_count"],
        "pareto_undominated_rate": 1.0,
    }
    direction_summary = {
        "individual_all_gate_pass_rate": 1.0,
        "n": steer_v9.THRESHOLDS["expected_per_direction_count"],
        "pareto_undominated_rate": 1.0,
        "target_prediction_pass_count": steer_v9.THRESHOLDS[
            "min_direction_target_prediction_count"
        ],
    }

    failures = steer_v9.gate_failures(
        aggregate=aggregate,
        by_target={target: target_summary for target in steer_v9.PATTERNS},
        by_direction={
            steer_v9.v1.vector_key(source, target): direction_summary
            for source in steer_v9.PATTERNS
            for target in steer_v9.PATTERNS
            if source != target
        },
        individual_audit={"all_gate_pass_rate": 1.0},
    )

    assert any("same-target transfer target-prediction rate" in item for item in failures)
    assert any("same-target transfer gate-pass rate" in item for item in failures)


def test_v9_control_specs_match_preregistered_negative_and_transfer_counts() -> None:
    z = torch.zeros(4)
    train_stats = {
        "centroids": {pattern: torch.zeros(4) for pattern in steer_v9.PATTERNS},
        "transport_stds": {pattern: torch.ones(4) for pattern in steer_v9.PATTERNS},
    }
    originals = {
        "apply_low_rank_residual_transport": steer_v9.apply_low_rank_residual_transport,
        "apply_v5_transport": steer_v9.apply_v5_transport,
        "apply_v6_correction": steer_v9.apply_v6_correction,
        "apply_v8_frontier": steer_v9.apply_v8_frontier,
        "build_v8_frontier_candidate_specs": steer_v9.build_v8_frontier_candidate_specs,
        "random_norm_matched_vectors": steer_v9.random_norm_matched_vectors,
    }

    def frontier(**_: object) -> list[torch.Tensor]:
        return [torch.full_like(z, float(index + 1)) for index in range(5)]

    def frontier_specs(
        *,
        control_type: str,
        control_transport_key: str,
        candidate_set_name: str,
        metadata: dict[str, object],
        **_: object,
    ) -> list[dict[str, object]]:
        return [
            steer_v9.candidate_spec(
                control_type,
                control_transport_key,
                torch.full_like(z, float(index + 1)),
                {
                    **metadata,
                    "candidate_index": index,
                    "candidate_set_name": candidate_set_name,
                },
            )
            for index in range(5)
        ]

    try:
        steer_v9.apply_low_rank_residual_transport = lambda **_: z
        steer_v9.apply_v5_transport = lambda **_: z
        steer_v9.apply_v6_correction = lambda **_: z
        steer_v9.apply_v8_frontier = frontier
        steer_v9.build_v8_frontier_candidate_specs = frontier_specs
        steer_v9.random_norm_matched_vectors = lambda **kwargs: [
            torch.zeros_like(z) for _ in range(kwargs["random_controls"])
        ]

        specs = steer_v9.build_control_candidate_specs(
            classifier=None,
            subject_id="subject-1",
            z=z,
            source="mountain_pattern",
            target="has_majority",
            train_stats=train_stats,
            random_controls=32,
        )
        negative, transfer = steer_v9.split_negative_controls_and_transfer_probes(
            specs,
            source="mountain_pattern",
            target="has_majority",
        )
    finally:
        for name, value in originals.items():
            setattr(steer_v9, name, value)

    assert len(negative) == 59
    assert len(transfer) == 10
    assert all(
        item["control_type"] != "v7_pareto_frontier_correction"
        for item in negative
    )


def main() -> None:
    tests = [
        test_v9_seed_preflight_uses_preregistered_base_seed,
        test_v9_rejects_prior_pool_directories,
        test_v9_classifies_same_target_other_source_as_transfer_probe,
        test_v9_split_control_specs_excludes_only_transfer_probes_from_dominance,
        test_v9_shuffled_same_target_direction_remains_negative_control,
        test_v9_transfer_summary_uses_all_probe_candidates,
        test_v9_aggregate_summary_counts_transfer_probe_rates,
        test_v9_gate_failures_include_transfer_probe_thresholds,
        test_v9_control_specs_match_preregistered_negative_and_transfer_counts,
    ]
    for test in tests:
        test()
    print("four-behavior representation steering V9 helper tests passed")


if __name__ == "__main__":
    main()
