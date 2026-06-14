"""Direct helper tests for four-behavior representation steering V4."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_representation_steering_v4_low_rank_residual_transport as steer_v4  # noqa: E402


def test_v4_rejects_prior_pool_directories() -> None:
    pool_dirs = [
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v1_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v2_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v3_pools",
    ]
    for pool_dir in pool_dirs:
        try:
            steer_v4.assert_preregistered_pool_dir(pool_dir)
        except ValueError as error:
            assert "V4" in str(error)
        else:
            raise AssertionError(f"V4 accepted forbidden prior pool directory: {pool_dir}")


def test_v4_rejects_non_preregistered_cli_overrides() -> None:
    args = SimpleNamespace(
        generic_negative_cap=1024,
        hard_negative_cap=1024,
        heldout_per_class=64,
        lr=0.003,
        positive_cap=2048,
        random_controls=32,
        source_margin_gate=0.41,
        support_per_class=160,
        train_epochs=350,
    )
    try:
        steer_v4.validate_preregistered_args(args)
    except ValueError as error:
        assert "source_margin_gate" in str(error)
    else:
        raise AssertionError("V4 accepted non-preregistered source margin gate")


def test_v4_seed_preflight_passes_preregistered_config() -> None:
    preflight = steer_v4.build_seed_preflight(
        steer_v4.POOL_CONFIGS,
        behavior_stride=steer_v4.SEED_BEHAVIOR_STRIDE,
    )
    assert preflight["passed"], preflight["failures"]
    assert preflight["seed_ranges"][0]["start_seed"] == 39300000


def test_v4_development_paths_exclude_raw_final() -> None:
    paths = steer_v4.development_input_paths(steer_v4.DEFAULT_POOL_DIR)
    names = {path.name for path in paths.values()}
    assert "final_subjects.json" not in names
    assert names == {
        "combined_audit.json",
        "development_subjects.json",
        "final_redacted_audit.json",
        "train_subjects.json",
    }


def test_v4_forbidden_final_raw_guard_rejects_v4_final_before_open() -> None:
    final_path = steer_v4.DEFAULT_POOL_DIR / "final_subjects.json"
    try:
        steer_v4.assert_no_forbidden_final_raw_paths([final_path], allow_v4_final=False)
    except ValueError as error:
        assert "final raw" in str(error) or "final_subjects.json" in str(error)
    else:
        raise AssertionError("V4 allowed raw final path before final evaluation")


def test_v4_source_pool_file_hash_binding_detects_swapped_raw_pool() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        train_path = tmp_dir / "train_subjects.json"
        development_path = tmp_dir / "development_subjects.json"
        train_path.write_text('{"pool": "train"}')
        development_path.write_text('{"pool": "development"}')
        combined_audit = {
            "pool_file_sha256": {
                "train": steer_v4.v1.sha256_file(train_path),
                "development": "not-the-development-hash",
            },
            "pool_summaries": {},
        }
        failures = steer_v4.validate_source_pool_file_hashes(
            train_path=train_path,
            eval_path=development_path,
            combined_audit=combined_audit,
            final_redacted={},
            phase="development",
        )
    assert failures
    assert "development pool sha256" in failures[0]


def test_v4_final_redacted_audit_forbidden_keys_fail_closed() -> None:
    train_payload = {
        "claim_scope": "four_behavior_representation_steering_v4_source_pool",
        "pool": "train",
    }
    eval_payload = {
        "claim_scope": "four_behavior_representation_steering_v4_source_pool",
        "pool": "development",
    }
    counts = {pattern: 64 for pattern in steer_v4.PATTERNS}
    dev_counts = {pattern: 24 for pattern in steer_v4.PATTERNS}
    combined_audit = {
        "claim_scope": "four_behavior_representation_steering_v4_source_pool_construction",
        "overlap_counts": {},
        "passed": True,
        "pool_summaries": {
            "train": {"accepted_counts_by_behavior": counts},
            "development": {"accepted_counts_by_behavior": dev_counts},
            "final": {"accepted_counts_by_behavior": dev_counts},
        },
        "seed_preflight": {"passed": True},
    }
    final_redacted = {
        "claim_scope": "redacted_final_steering_v4_source_pool_audit_surface_only",
        "summary": {"accepted_counts_by_behavior": dev_counts},
        "leaked": {
            "subject_id": "forbidden",
            "signature_hash": "forbidden",
            "weights_hash": "forbidden",
            "source_margin": 1.0,
        },
    }
    failures = steer_v4.validate_source_pool_contract(
        train_payload=train_payload,
        eval_payload=eval_payload,
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )
    assert any("subject_id" in failure for failure in failures)
    assert any("signature_hash" in failure for failure in failures)
    assert any("weights_hash" in failure for failure in failures)
    assert any("source_margin" in failure for failure in failures)


def test_v4_low_rank_transport_maps_source_centroid_to_target_centroid() -> None:
    centroids = {
        "sorted_ascending": torch.tensor([1.0, 2.0, 3.0]),
        "sorted_descending": torch.tensor([5.0, -1.0, 4.0]),
    }
    identity = torch.eye(2)
    stats = {
        "centroids": centroids,
        "pca_components": torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]),
        "sqrt_cov": {
            "sorted_ascending": identity,
            "sorted_descending": identity,
        },
        "inv_sqrt_cov": {
            "sorted_ascending": identity,
            "sorted_descending": identity,
        },
    }
    transported = steer_v4.apply_low_rank_residual_transport(
        z=centroids["sorted_ascending"],
        source="sorted_ascending",
        target="sorted_descending",
        stats=stats,
    )
    assert torch.allclose(transported, centroids["sorted_descending"])


def test_v4_low_rank_transport_applies_covariance_map_and_drops_orthogonal_residual() -> None:
    centroids = {
        "sorted_ascending": torch.zeros(3),
        "sorted_descending": torch.zeros(3),
    }
    stats = {
        "centroids": centroids,
        "pca_components": torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]),
        "sqrt_cov": {
            "sorted_ascending": torch.eye(2),
            "sorted_descending": torch.diag(torch.tensor([2.0, 3.0])),
        },
        "inv_sqrt_cov": {
            "sorted_ascending": torch.eye(2),
            "sorted_descending": torch.eye(2),
        },
    }
    z = torch.tensor([1.0, 2.0, 9.0])
    transported = steer_v4.apply_low_rank_residual_transport(
        z=z,
        source="sorted_ascending",
        target="sorted_descending",
        stats=stats,
    )
    assert torch.allclose(transported, torch.tensor([2.0, 6.0, 0.0]))


def test_v4_transport_norm_cap_is_applied_to_displacement() -> None:
    centroids = {
        "sorted_ascending": torch.zeros(2),
        "sorted_descending": torch.zeros(2),
    }
    stats = {
        "centroids": centroids,
        "pca_components": torch.eye(2),
        "sqrt_cov": {
            "sorted_ascending": torch.eye(2),
            "sorted_descending": torch.eye(2) * 1000.0,
        },
        "inv_sqrt_cov": {
            "sorted_ascending": torch.eye(2),
            "sorted_descending": torch.eye(2),
        },
    }
    z = torch.tensor([1.0, 0.0])
    transported = steer_v4.apply_low_rank_residual_transport(
        z=z,
        source="sorted_ascending",
        target="sorted_descending",
        stats=stats,
    )
    assert torch.allclose((transported - z).norm(), torch.tensor(200.0), atol=1e-5)


def test_v4_shuffled_direction_selection_is_deterministic_and_excludes_matched_reverse() -> None:
    first = steer_v4.select_shuffled_direction(
        subject_id="subject-a",
        source="sorted_ascending",
        target="has_majority",
    )
    second = steer_v4.select_shuffled_direction(
        subject_id="subject-a",
        source="sorted_ascending",
        target="has_majority",
    )
    assert first == second
    assert first != ("sorted_ascending", "has_majority")
    assert first != ("has_majority", "sorted_ascending")
    assert first[0] != first[1]


def test_v4_random_controls_are_deterministic_and_norm_matched() -> None:
    z = torch.zeros(4)
    controls_a = steer_v4.random_norm_matched_vectors(
        subject_id="subject-a",
        source="sorted_ascending",
        target="has_majority",
        z=z,
        matched_displacement_norm=3.0,
        random_controls=2,
    )
    controls_b = steer_v4.random_norm_matched_vectors(
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


def test_v4_control_specs_include_v2_v3_and_source_conditioned_controls() -> None:
    centroids = {
        "sorted_ascending": torch.tensor([1.0, 0.0]),
        "sorted_descending": torch.tensor([4.0, -1.0]),
        "has_majority": torch.tensor([0.5, 3.0]),
        "mountain_pattern": torch.tensor([-2.0, 2.0]),
    }
    identity = torch.eye(2)
    train_stats = {
        "centroids": centroids,
        "pca_components": identity,
        "sqrt_cov": {pattern: identity for pattern in steer_v4.PATTERNS},
        "inv_sqrt_cov": {pattern: identity for pattern in steer_v4.PATTERNS},
        "transport_stds": {pattern: torch.ones(2) for pattern in steer_v4.PATTERNS},
    }
    z = torch.tensor([2.0, 1.0])
    controls = steer_v4.build_control_candidate_specs(
        subject_id="subject-a",
        z=z,
        source="sorted_ascending",
        target="sorted_descending",
        train_stats=train_stats,
        random_controls=2,
    )
    control_types = [control["control_type"] for control in controls]
    assert "v2_centroid_delta" in control_types
    assert "v3_diagonal_transport" in control_types
    assert "same_target_other_source_low_rank_residual_transport" in control_types
    assert control_types.count("random_norm_matched_vector") == 2
    assert all("control_transport_key" in control for control in controls)


def test_v4_train_only_statistics_hash_accepts_matrix_pca_components() -> None:
    train_stats = {
        "centroids": {pattern: torch.ones(2) for pattern in steer_v4.PATTERNS},
        "pca_components": torch.eye(2),
        "sig_mean": torch.zeros(2),
        "sig_std": torch.ones(2),
        "transport_stds": {pattern: torch.ones(2) for pattern in steer_v4.PATTERNS},
    }
    digest = steer_v4.train_only_statistics_hash(train_stats)
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_v4_record_passes_requires_v2_and_v3_improvement() -> None:
    passing_summary = {
        "matched_centroid_improvement": 0.2,
        "matched_minus_best_control_centroid_improvement": 0.2,
        "matched_minus_best_control_primary_target_margin": 0.2,
        "matched_minus_v2_centroid_delta_centroid_improvement": 0.2,
        "matched_minus_v2_centroid_delta_primary_target_margin": 0.2,
        "matched_minus_v3_diagonal_transport_centroid_improvement": 0.2,
        "matched_minus_v3_diagonal_transport_primary_target_margin": 0.2,
        "matched_primary_target_margin": 0.2,
        "source_primary_margin_change": -0.2,
    }
    matched = {
        "centroid_predicted_behavior": "sorted_descending",
        "primary_predicted_behavior": "sorted_descending",
    }
    assert steer_v4.record_passes(passing_summary, matched, target="sorted_descending")
    failing = dict(passing_summary)
    failing["matched_minus_v3_diagonal_transport_centroid_improvement"] = -0.01
    assert not steer_v4.record_passes(failing, matched, target="sorted_descending")


def main() -> None:
    test_v4_rejects_prior_pool_directories()
    test_v4_rejects_non_preregistered_cli_overrides()
    test_v4_seed_preflight_passes_preregistered_config()
    test_v4_development_paths_exclude_raw_final()
    test_v4_forbidden_final_raw_guard_rejects_v4_final_before_open()
    test_v4_source_pool_file_hash_binding_detects_swapped_raw_pool()
    test_v4_final_redacted_audit_forbidden_keys_fail_closed()
    test_v4_low_rank_transport_maps_source_centroid_to_target_centroid()
    test_v4_low_rank_transport_applies_covariance_map_and_drops_orthogonal_residual()
    test_v4_transport_norm_cap_is_applied_to_displacement()
    test_v4_shuffled_direction_selection_is_deterministic_and_excludes_matched_reverse()
    test_v4_random_controls_are_deterministic_and_norm_matched()
    test_v4_control_specs_include_v2_v3_and_source_conditioned_controls()
    test_v4_train_only_statistics_hash_accepts_matrix_pca_components()
    test_v4_record_passes_requires_v2_and_v3_improvement()
    print("four-behavior representation steering V4 helper tests passed")


if __name__ == "__main__":
    main()
