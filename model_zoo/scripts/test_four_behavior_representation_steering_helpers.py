"""Direct helper tests for four-behavior representation steering V1."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_representation_steering as steer  # noqa: E402


def test_forbidden_final_raw_paths_rejected() -> None:
    decoder_final = (
        REPO_ROOT
        / "runs"
        / "four_behavior_decoder_source_pools_v2"
        / "final_subjects.json"
    )
    steering_final = (
        REPO_ROOT
        / "runs"
        / "four_behavior_representation_steering_v1_pools"
        / "final_subjects.json"
    )
    for path in (decoder_final, steering_final):
        try:
            steer.assert_no_forbidden_final_raw_paths([path], allow_steering_final=False)
        except ValueError as error:
            assert "final" in str(error)
        else:
            raise AssertionError(f"final raw path was not rejected: {path}")


def test_development_input_paths_exclude_raw_final_pool() -> None:
    pool_dir = (
        REPO_ROOT
        / "runs"
        / "four_behavior_representation_steering_v1_pools"
    )
    paths = steer.development_input_paths(pool_dir)
    names = {path.name for path in paths.values()}
    assert "final_subjects.json" not in names
    assert names == {
        "combined_audit.json",
        "development_subjects.json",
        "final_redacted_audit.json",
        "train_subjects.json",
    }


def test_seed_preflight_detects_overlap() -> None:
    configs = {
        "train": {"base_seed": 100, "max_attempts_per_behavior": 8},
        "development": {"base_seed": 104, "max_attempts_per_behavior": 8},
    }
    preflight = steer.build_seed_preflight(configs, behavior_stride=1)
    assert not preflight["passed"]
    assert preflight["failures"]


def test_seed_preflight_passes_preregistered_config() -> None:
    preflight = steer.build_seed_preflight(
        steer.POOL_CONFIGS,
        behavior_stride=steer.SEED_BEHAVIOR_STRIDE,
    )
    assert preflight["passed"], preflight["failures"]


def test_centroid_delta_control_is_target_minus_source() -> None:
    centroids = {
        "source": torch.tensor([1.0, -2.0, 3.0]),
        "target": torch.tensor([4.0, 1.0, -1.0]),
    }
    expected = torch.tensor([3.0, 3.0, -4.0])
    actual = steer.centroid_delta_control(centroids, "source", "target")
    assert torch.equal(actual, expected)


def test_training_centroid_improvement_is_relative_to_no_edit_source() -> None:
    z = torch.tensor([[1.0, 0.0]])
    candidate = torch.tensor([[2.0, 0.0]])
    centroids = {
        "source": torch.tensor([0.0, 0.0]),
        "target": torch.tensor([4.0, 0.0]),
    }
    expected = torch.tensor([1.0])
    actual = steer.centroid_improvement_relative_to_no_edit(
        no_edit_z=z,
        candidate_z=candidate,
        target_behavior="target",
        centroids=centroids,
    )
    assert torch.allclose(actual, expected)


def test_shuffled_vector_selection_is_deterministic_and_excludes_source_target() -> None:
    first = steer.select_shuffled_vector_key(
        subject_id="subject-a",
        source_behavior="sorted_ascending",
        target_behavior="has_majority",
    )
    second = steer.select_shuffled_vector_key(
        subject_id="subject-a",
        source_behavior="sorted_ascending",
        target_behavior="has_majority",
    )
    assert first == second
    assert first[0] != "sorted_ascending"
    assert first[1] != "has_majority"
    assert first[0] != first[1]


def test_gate_failures_require_preregistered_direction_pass_rate() -> None:
    aggregate = {
        "n": 288,
        "mean_matched_primary_target_margin": 0.25,
        "mean_matched_minus_best_control_primary_target_margin": 0.20,
        "mean_matched_centroid_improvement": 0.20,
        "mean_matched_minus_best_control_centroid_improvement": 0.15,
        "mean_source_primary_margin_change": -0.10,
    }
    by_target = {
        pattern: {
            "n": 72,
            "mean_matched_primary_target_margin": 0.20,
            "mean_matched_minus_best_control_primary_target_margin": 0.12,
            "mean_matched_centroid_improvement": 0.12,
            "individual_all_gate_pass_rate": 0.90,
        }
        for pattern in steer.PATTERNS
    }
    by_direction = {
        f"{source}_to_{target}": {
            "n": 24,
            "mean_matched_primary_target_margin": 0.15,
            "mean_matched_minus_best_control_primary_target_margin": 0.01,
            "individual_all_gate_pass_rate": 0.89,
        }
        for source in steer.PATTERNS
        for target in steer.PATTERNS
        if source != target
    }
    individual = {"all_gate_pass_rate": 0.95}
    failures = steer.gate_failures(
        aggregate=aggregate,
        by_target=by_target,
        by_direction=by_direction,
        individual_audit=individual,
    )
    assert any("direction" in failure and "pass rate" in failure for failure in failures)
    assert any("observed 0.89 < required 0.9" in failure for failure in failures)
    assert not any("failed: 0.89 >= 0.9" in failure for failure in failures)


def main() -> None:
    test_forbidden_final_raw_paths_rejected()
    test_development_input_paths_exclude_raw_final_pool()
    test_seed_preflight_detects_overlap()
    test_seed_preflight_passes_preregistered_config()
    test_centroid_delta_control_is_target_minus_source()
    test_training_centroid_improvement_is_relative_to_no_edit_source()
    test_shuffled_vector_selection_is_deterministic_and_excludes_source_target()
    test_gate_failures_require_preregistered_direction_pass_rate()
    print("four-behavior representation steering helper tests passed")


if __name__ == "__main__":
    main()
