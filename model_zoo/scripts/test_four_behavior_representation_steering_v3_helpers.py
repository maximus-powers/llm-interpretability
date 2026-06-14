"""Direct helper tests for four-behavior representation steering V3."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_representation_steering_v3_diagonal_transport as steer_v3  # noqa: E402


def test_v3_rejects_v1_and_v2_pool_directories() -> None:
    pool_dirs = [
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v1_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v2_pools",
    ]
    for pool_dir in pool_dirs:
        try:
            steer_v3.assert_preregistered_pool_dir(pool_dir)
        except ValueError as error:
            assert "V3" in str(error)
        else:
            raise AssertionError(f"V3 accepted forbidden prior pool directory: {pool_dir}")


def test_v3_rejects_non_preregistered_cli_overrides() -> None:
    args = SimpleNamespace(
        generic_negative_cap=1024,
        hard_negative_cap=1024,
        heldout_per_class=64,
        lr=0.003,
        positive_cap=2048,
        random_controls=31,
        source_margin_gate=0.40,
        support_per_class=160,
        train_epochs=350,
    )
    try:
        steer_v3.validate_preregistered_args(args)
    except ValueError as error:
        assert "random_controls" in str(error)
    else:
        raise AssertionError("V3 accepted non-preregistered random control count")


def test_v3_seed_preflight_passes_preregistered_config() -> None:
    preflight = steer_v3.build_seed_preflight(
        steer_v3.POOL_CONFIGS,
        behavior_stride=steer_v3.SEED_BEHAVIOR_STRIDE,
    )
    assert preflight["passed"], preflight["failures"]
    assert preflight["seed_ranges"][0]["start_seed"] == 36300000


def test_v3_diagonal_transport_maps_source_centroid_to_target_centroid() -> None:
    centroids = {
        "sorted_ascending": torch.tensor([1.0, 2.0]),
        "sorted_descending": torch.tensor([5.0, -1.0]),
    }
    stds = {
        "sorted_ascending": torch.tensor([2.0, 4.0]),
        "sorted_descending": torch.tensor([6.0, 2.0]),
    }
    transported = steer_v3.apply_diagonal_transport(
        z=centroids["sorted_ascending"],
        source="sorted_ascending",
        target="sorted_descending",
        centroids=centroids,
        stds=stds,
    )
    assert torch.allclose(transported, centroids["sorted_descending"])


def test_v3_diagonal_transport_uses_clipped_train_only_ratio() -> None:
    centroids = {
        "sorted_ascending": torch.tensor([0.0, 0.0]),
        "sorted_descending": torch.tensor([0.0, 0.0]),
    }
    stds = {
        "sorted_ascending": torch.tensor([1.0, 100.0]),
        "sorted_descending": torch.tensor([10.0, 1.0]),
    }
    z = torch.tensor([1.0, 100.0])
    transported = steer_v3.apply_diagonal_transport(
        z=z,
        source="sorted_ascending",
        target="sorted_descending",
        centroids=centroids,
        stds=stds,
    )
    assert torch.allclose(transported, torch.tensor([4.0, 25.0]))


def test_v3_shuffled_direction_selection_is_deterministic_and_excludes_matched_reverse() -> None:
    first = steer_v3.select_shuffled_direction(
        subject_id="subject-a",
        source="sorted_ascending",
        target="has_majority",
    )
    second = steer_v3.select_shuffled_direction(
        subject_id="subject-a",
        source="sorted_ascending",
        target="has_majority",
    )
    assert first == second
    assert first != ("sorted_ascending", "has_majority")
    assert first != ("has_majority", "sorted_ascending")
    assert first[0] != first[1]


def test_v3_random_controls_are_deterministic_and_norm_matched() -> None:
    z = torch.zeros(4)
    controls_a = steer_v3.random_norm_matched_vectors(
        subject_id="subject-a",
        source="sorted_ascending",
        target="has_majority",
        z=z,
        matched_displacement_norm=3.0,
        random_controls=2,
    )
    controls_b = steer_v3.random_norm_matched_vectors(
        subject_id="subject-a",
        source="sorted_ascending",
        target="has_majority",
        z=z,
        matched_displacement_norm=3.0,
        random_controls=2,
    )
    assert len(controls_a) == 2
    for left, right in zip(controls_a, controls_b):
        assert torch.allclose(left, right)
        assert torch.allclose(left.norm(), torch.tensor(3.0), atol=1e-6)


def test_v3_control_specs_include_v2_centroid_delta_and_identities() -> None:
    centroids = {
        "sorted_ascending": torch.tensor([1.0, 0.0]),
        "sorted_descending": torch.tensor([4.0, -1.0]),
        "has_majority": torch.tensor([0.5, 3.0]),
        "mountain_pattern": torch.tensor([-2.0, 2.0]),
    }
    stds = {pattern: torch.ones(2) for pattern in steer_v3.PATTERNS}
    z = torch.tensor([2.0, 1.0])
    controls = steer_v3.build_control_candidate_specs(
        subject_id="subject-a",
        z=z,
        source="sorted_ascending",
        target="sorted_descending",
        centroids=centroids,
        stds=stds,
        random_controls=2,
    )
    control_types = [control["control_type"] for control in controls]
    assert "v2_centroid_delta" in control_types
    assert "same_target_other_source_diagonal_transport" in control_types
    assert control_types.count("random_norm_matched_vector") == 2
    assert all("control_transport_key" in control for control in controls)


def test_v3_record_passes_requires_v2_centroid_delta_improvement() -> None:
    passing_summary = {
        "matched_centroid_improvement": 0.2,
        "matched_minus_best_control_centroid_improvement": 0.2,
        "matched_minus_best_control_primary_target_margin": 0.2,
        "matched_minus_v2_centroid_delta_centroid_improvement": 0.2,
        "matched_minus_v2_centroid_delta_primary_target_margin": 0.2,
        "matched_primary_target_margin": 0.2,
        "source_primary_margin_change": -0.2,
    }
    matched = {
        "centroid_predicted_behavior": "sorted_descending",
        "primary_predicted_behavior": "sorted_descending",
    }
    assert steer_v3.record_passes(passing_summary, matched, target="sorted_descending")
    failing = dict(passing_summary)
    failing["matched_minus_v2_centroid_delta_centroid_improvement"] = -0.01
    assert not steer_v3.record_passes(failing, matched, target="sorted_descending")


def main() -> None:
    test_v3_rejects_v1_and_v2_pool_directories()
    test_v3_rejects_non_preregistered_cli_overrides()
    test_v3_seed_preflight_passes_preregistered_config()
    test_v3_diagonal_transport_maps_source_centroid_to_target_centroid()
    test_v3_diagonal_transport_uses_clipped_train_only_ratio()
    test_v3_shuffled_direction_selection_is_deterministic_and_excludes_matched_reverse()
    test_v3_random_controls_are_deterministic_and_norm_matched()
    test_v3_control_specs_include_v2_centroid_delta_and_identities()
    test_v3_record_passes_requires_v2_centroid_delta_improvement()
    print("four-behavior representation steering V3 helper tests passed")


if __name__ == "__main__":
    main()
