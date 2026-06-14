"""Direct helper tests for four-behavior representation steering V6."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_representation_steering_v6_centroid_constrained_primary_correction as steer_v6  # noqa: E402


def test_v6_rejects_prior_pool_directories() -> None:
    pool_dirs = [
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v1_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v2_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v3_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v4_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v5_pools",
    ]
    for pool_dir in pool_dirs:
        try:
            steer_v6.assert_preregistered_pool_dir(pool_dir)
        except ValueError as error:
            assert "V6" in str(error)
        else:
            raise AssertionError(f"V6 accepted forbidden prior pool directory: {pool_dir}")


def test_v6_seed_preflight_uses_preregistered_base_seed() -> None:
    preflight = steer_v6.build_seed_preflight(
        steer_v6.POOL_CONFIGS,
        behavior_stride=steer_v6.SEED_BEHAVIOR_STRIDE,
    )
    assert preflight["passed"], preflight["failures"]
    assert preflight["seed_ranges"][0]["start_seed"] == 45300000


def test_v6_project_ball_handles_inside_boundary_and_zero_distance() -> None:
    center = torch.tensor([1.0, 2.0])
    inside = torch.tensor([1.2, 2.0])
    assert torch.allclose(steer_v6.project_ball(inside, center=center, radius=1.0), inside)
    outside = torch.tensor([4.0, 2.0])
    assert torch.allclose(
        steer_v6.project_ball(outside, center=center, radius=2.0),
        torch.tensor([3.0, 2.0]),
    )
    assert torch.allclose(steer_v6.project_ball(center, center=center, radius=0.0), center)


def test_v6_projected_correction_is_no_farther_than_v4_target_radius() -> None:
    identity = torch.eye(2)
    stats = {
        "centroids": {
            "sorted_ascending": torch.tensor([0.0, 0.0]),
            "sorted_descending": torch.tensor([10.0, 0.0]),
            "has_majority": torch.tensor([0.0, 10.0]),
            "mountain_pattern": torch.tensor([-10.0, 0.0]),
        },
        "inv_sqrt_cov": {
            pattern: identity for pattern in steer_v6.PATTERNS
        },
        "pca_components": identity,
        "sqrt_cov": {
            pattern: identity for pattern in steer_v6.PATTERNS
        },
    }
    classifier = torch.nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]))
    z = torch.tensor([0.0, 0.0])
    v4 = steer_v6.apply_low_rank_residual_transport(
        z=z,
        source="sorted_ascending",
        target="sorted_descending",
        stats=stats,
    )
    corrected = steer_v6.apply_v6_correction(
        z=z,
        source="sorted_ascending",
        target="sorted_descending",
        stats=stats,
        classifier=classifier,
    )
    target_centroid = stats["centroids"]["sorted_descending"]
    assert (corrected - target_centroid).norm().item() <= max(
        (v4 - target_centroid).norm().item() - 0.05,
        0.0,
    ) + 1e-5


def test_v6_random_controls_are_reproducible() -> None:
    kwargs = {
        "matched_displacement_norm": 4.0,
        "random_controls": 4,
        "source": "sorted_ascending",
        "subject_id": "subject-1",
        "target": "sorted_descending",
        "z": torch.zeros(5),
    }
    first = steer_v6.random_norm_matched_vectors(**kwargs)
    second = steer_v6.random_norm_matched_vectors(**kwargs)
    assert len(first) == 4
    assert all(torch.allclose(left, right) for left, right in zip(first, second))
    assert all(abs(vector.norm().item() - 4.0) < 1e-6 for vector in first)


def main() -> None:
    tests = [
        test_v6_rejects_prior_pool_directories,
        test_v6_seed_preflight_uses_preregistered_base_seed,
        test_v6_project_ball_handles_inside_boundary_and_zero_distance,
        test_v6_projected_correction_is_no_farther_than_v4_target_radius,
        test_v6_random_controls_are_reproducible,
    ]
    for test in tests:
        test()
    print("four-behavior representation steering V6 helper tests passed")


if __name__ == "__main__":
    main()
