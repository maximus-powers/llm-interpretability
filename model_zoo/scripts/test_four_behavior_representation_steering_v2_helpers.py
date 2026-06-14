"""Direct helper tests for four-behavior representation steering V2."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_representation_steering_v2_centroid_delta as steer_v2  # noqa: E402


def test_v2_forbidden_final_raw_paths_rejected_before_final() -> None:
    forbidden_paths = [
        REPO_ROOT / "runs" / "four_behavior_decoder_source_pools_v2" / "final_subjects.json",
        REPO_ROOT
        / "runs"
        / "four_behavior_representation_steering_v1_pools"
        / "final_subjects.json",
        REPO_ROOT
        / "runs"
        / "four_behavior_representation_steering_v2_pools"
        / "final_subjects.json",
    ]
    for path in forbidden_paths:
        try:
            steer_v2.assert_no_forbidden_final_raw_paths([path], allow_v2_final=False)
        except ValueError as error:
            assert "final" in str(error)
        else:
            raise AssertionError(f"final raw path was not rejected: {path}")


def test_v2_development_input_paths_exclude_raw_final_pool() -> None:
    pool_dir = REPO_ROOT / "runs" / "four_behavior_representation_steering_v2_pools"
    paths = steer_v2.development_input_paths(pool_dir)
    names = {path.name for path in paths.values()}
    assert "final_subjects.json" not in names
    assert names == {
        "combined_audit.json",
        "development_subjects.json",
        "final_redacted_audit.json",
        "train_subjects.json",
    }


def test_v2_rejects_v1_pool_directory_for_development() -> None:
    pool_dir = REPO_ROOT / "runs" / "four_behavior_representation_steering_v1_pools"
    try:
        steer_v2.assert_preregistered_pool_dir(pool_dir)
    except ValueError as error:
        assert "V1" in str(error)
    else:
        raise AssertionError("V2 accepted the V1 steering pool directory")


def test_v2_rejects_non_preregistered_cli_overrides() -> None:
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
        steer_v2.validate_preregistered_args(args)
    except ValueError as error:
        assert "random_controls" in str(error)
    else:
        raise AssertionError("V2 accepted non-preregistered random control count")


def test_v2_source_pool_contract_rejects_non_v2_scopes() -> None:
    counts = {pattern: 64 for pattern in steer_v2.PATTERNS}
    final_counts = {pattern: 24 for pattern in steer_v2.PATTERNS}
    train_payload = {"claim_scope": "four_behavior_representation_steering_source_pool", "pool": "train"}
    eval_payload = {"claim_scope": "four_behavior_representation_steering_source_pool", "pool": "development"}
    combined_audit = {
        "claim_scope": "four_behavior_representation_steering_source_pool_construction",
        "overlap_counts": {},
        "passed": True,
        "pool_summaries": {
            "train": {"accepted_counts_by_behavior": counts},
            "development": {"accepted_counts_by_behavior": final_counts},
            "final": {"accepted_counts_by_behavior": final_counts},
        },
        "seed_preflight": {"passed": True},
    }
    final_redacted = {
        "claim_scope": "redacted_final_steering_source_pool_audit_surface_only",
        "summary": {"accepted_counts_by_behavior": final_counts},
    }
    failures = steer_v2.validate_source_pool_contract(
        train_payload=train_payload,
        eval_payload=eval_payload,
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )
    assert any("not V2-specific" in failure for failure in failures)


def test_v2_source_pool_contract_allows_accepted_counts_by_behavior_key() -> None:
    train_counts = {pattern: 64 for pattern in steer_v2.PATTERNS}
    eval_counts = {pattern: 24 for pattern in steer_v2.PATTERNS}
    train_payload = {"claim_scope": "four_behavior_representation_steering_v2_source_pool", "pool": "train"}
    eval_payload = {
        "claim_scope": "four_behavior_representation_steering_v2_source_pool",
        "pool": "development",
    }
    combined_audit = {
        "claim_scope": "four_behavior_representation_steering_v2_source_pool_construction",
        "overlap_counts": {},
        "passed": True,
        "pool_summaries": {
            "train": {"accepted_counts_by_behavior": train_counts},
            "development": {"accepted_counts_by_behavior": eval_counts},
            "final": {"accepted_counts_by_behavior": eval_counts},
        },
        "seed_preflight": {"passed": True},
    }
    final_redacted = {
        "claim_scope": "redacted_final_steering_v2_source_pool_audit_surface_only",
        "summary": {"accepted_counts_by_behavior": eval_counts},
    }
    failures = steer_v2.validate_source_pool_contract(
        train_payload=train_payload,
        eval_payload=eval_payload,
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )
    assert not failures


def test_v2_seed_preflight_passes_preregistered_config() -> None:
    preflight = steer_v2.build_seed_preflight(
        steer_v2.POOL_CONFIGS,
        behavior_stride=steer_v2.SEED_BEHAVIOR_STRIDE,
    )
    assert preflight["passed"], preflight["failures"]
    assert preflight["seed_ranges"][0]["start_seed"] == 33300000


def test_v2_centroid_delta_vectors_are_exact_target_minus_source() -> None:
    centroids = {
        "sorted_ascending": torch.tensor([1.0, 0.0]),
        "sorted_descending": torch.tensor([4.0, -1.0]),
        "has_majority": torch.tensor([0.5, 3.0]),
        "mountain_pattern": torch.tensor([-2.0, 2.0]),
    }
    vectors = steer_v2.build_centroid_delta_vectors(centroids)
    assert torch.equal(
        vectors["sorted_ascending_to_sorted_descending"],
        torch.tensor([3.0, -1.0]),
    )
    assert len(vectors) == 12


def test_v2_control_specs_report_identities_and_exclude_matched_delta_control() -> None:
    centroids = {
        "sorted_ascending": torch.tensor([1.0, 0.0]),
        "sorted_descending": torch.tensor([4.0, -1.0]),
        "has_majority": torch.tensor([0.5, 3.0]),
        "mountain_pattern": torch.tensor([-2.0, 2.0]),
    }
    controls = steer_v2.build_control_vector_specs(
        subject_id="subject-a",
        source="sorted_ascending",
        target="sorted_descending",
        centroids=centroids,
        matched_norm=3.2,
        random_controls=2,
    )
    control_types = [control["control_type"] for control in controls]
    assert "target_source_centroid_delta" not in control_types
    assert "same_source_other_target_centroid_delta" in control_types
    assert "same_target_other_source_centroid_delta" in control_types
    assert "shuffled_direction_centroid_delta" in control_types
    assert control_types.count("random_norm_matched_vector") == 2
    assert all("control_vector_key" in control for control in controls)


def test_v2_record_passes_requires_primary_and_centroid_target_predictions() -> None:
    passing_summary = {
        "matched_centroid_improvement": 0.2,
        "matched_minus_best_control_centroid_improvement": 0.2,
        "matched_minus_best_control_primary_target_margin": 0.2,
        "matched_primary_target_margin": 0.2,
        "source_primary_margin_change": -0.2,
    }
    matched = {
        "centroid_predicted_behavior": "sorted_descending",
        "primary_predicted_behavior": "sorted_descending",
    }
    assert steer_v2.record_passes(passing_summary, matched, target="sorted_descending")
    matched["centroid_predicted_behavior"] = "has_majority"
    assert not steer_v2.record_passes(passing_summary, matched, target="sorted_descending")


def test_v2_gate_failures_require_preregistered_direction_pass_rate() -> None:
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
        for pattern in steer_v2.PATTERNS
    }
    by_direction = {
        f"{source}_to_{target}": {
            "n": 24,
            "mean_matched_primary_target_margin": 0.15,
            "mean_matched_minus_best_control_primary_target_margin": 0.01,
            "individual_all_gate_pass_rate": 0.89,
        }
        for source in steer_v2.PATTERNS
        for target in steer_v2.PATTERNS
        if source != target
    }
    individual = {"all_gate_pass_rate": 0.95}
    failures = steer_v2.gate_failures(
        aggregate=aggregate,
        by_target=by_target,
        by_direction=by_direction,
        individual_audit=individual,
    )
    assert any("direction" in failure and "pass rate" in failure for failure in failures)
    assert any("observed 0.89 < required 0.9" in failure for failure in failures)


def main() -> None:
    test_v2_forbidden_final_raw_paths_rejected_before_final()
    test_v2_development_input_paths_exclude_raw_final_pool()
    test_v2_rejects_v1_pool_directory_for_development()
    test_v2_rejects_non_preregistered_cli_overrides()
    test_v2_source_pool_contract_rejects_non_v2_scopes()
    test_v2_source_pool_contract_allows_accepted_counts_by_behavior_key()
    test_v2_seed_preflight_passes_preregistered_config()
    test_v2_centroid_delta_vectors_are_exact_target_minus_source()
    test_v2_control_specs_report_identities_and_exclude_matched_delta_control()
    test_v2_record_passes_requires_primary_and_centroid_target_predictions()
    test_v2_gate_failures_require_preregistered_direction_pass_rate()
    print("four-behavior representation steering V2 helper tests passed")


if __name__ == "__main__":
    main()
