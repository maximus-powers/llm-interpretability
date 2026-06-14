"""Direct helper tests for four-behavior representation steering V5."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_representation_steering_v5_contrastive_residual_calibration as steer_v5  # noqa: E402


def test_v5_rejects_prior_pool_directories() -> None:
    pool_dirs = [
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v1_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v2_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v3_pools",
        REPO_ROOT / "runs" / "four_behavior_representation_steering_v4_pools",
    ]
    for pool_dir in pool_dirs:
        try:
            steer_v5.assert_preregistered_pool_dir(pool_dir)
        except ValueError as error:
            assert "V5" in str(error)
        else:
            raise AssertionError(f"V5 accepted forbidden prior pool directory: {pool_dir}")


def test_v5_seed_preflight_uses_preregistered_base_seed() -> None:
    preflight = steer_v5.build_seed_preflight(
        steer_v5.POOL_CONFIGS,
        behavior_stride=steer_v5.SEED_BEHAVIOR_STRIDE,
    )
    assert preflight["passed"], preflight["failures"]
    assert preflight["seed_ranges"][0]["start_seed"] == 42300000


def test_v5_development_paths_exclude_raw_final() -> None:
    paths = steer_v5.development_input_paths(steer_v5.DEFAULT_POOL_DIR)
    names = {path.name for path in paths.values()}
    assert "final_subjects.json" not in names
    assert names == {
        "combined_audit.json",
        "development_subjects.json",
        "final_redacted_audit.json",
        "train_subjects.json",
    }


def test_v5_forbidden_final_raw_guard_rejects_v5_final_before_open() -> None:
    final_path = steer_v5.DEFAULT_POOL_DIR / "final_subjects.json"
    try:
        steer_v5.assert_no_forbidden_final_raw_paths([final_path], allow_v5_final=False)
    except ValueError as error:
        assert "final raw" in str(error) or "final_subjects.json" in str(error)
    else:
        raise AssertionError("V5 allowed raw final path before final evaluation")


def test_v5_null_control_is_zero_displacement() -> None:
    z = torch.tensor([1.0, -2.0])
    spec = steer_v5.candidate_spec("null_vector", "null_vector", z)
    assert torch.equal(spec["candidate_z"], z)


def test_v5_v2_and_v3_controls_use_train_only_formulas() -> None:
    centroids = {
        "sorted_ascending": torch.tensor([1.0, 2.0]),
        "sorted_descending": torch.tensor([4.0, 6.0]),
    }
    stds = {
        "sorted_ascending": torch.tensor([2.0, 4.0]),
        "sorted_descending": torch.tensor([6.0, 2.0]),
    }
    z = torch.tensor([3.0, 10.0])
    v2 = steer_v5.apply_v2_centroid_delta(
        z=z,
        source="sorted_ascending",
        target="sorted_descending",
        centroids=centroids,
    )
    assert torch.allclose(v2, torch.tensor([6.0, 14.0]))
    v3 = steer_v5.apply_diagonal_transport(
        z=z,
        source="sorted_ascending",
        target="sorted_descending",
        centroids=centroids,
        stds=stds,
    )
    assert torch.allclose(v3, torch.tensor([10.0, 10.0]))


def test_v5_zero_calibration_matches_uncalibrated_v4_transport() -> None:
    identity = torch.eye(2)
    centroids = {
        "sorted_ascending": torch.zeros(2),
        "sorted_descending": torch.tensor([1.0, -1.0]),
    }
    stats = {
        "calibration_coefficients": {
            "sorted_ascending_to_sorted_descending": torch.zeros(2),
        },
        "centroids": centroids,
        "inv_sqrt_cov": {
            "sorted_ascending": identity,
            "sorted_descending": identity,
        },
        "pca_components": identity,
        "sqrt_cov": {
            "sorted_ascending": identity,
            "sorted_descending": identity,
        },
    }
    z = torch.tensor([0.5, 0.25])
    assert torch.allclose(
        steer_v5.apply_v5_transport(
            z=z,
            source="sorted_ascending",
            target="sorted_descending",
            stats=stats,
        ),
        steer_v5.apply_low_rank_residual_transport(
            z=z,
            source="sorted_ascending",
            target="sorted_descending",
            stats=stats,
        ),
    )


def test_v5_training_loss_detaches_control_coefficients() -> None:
    patterns = ("sorted_ascending", "sorted_descending", "has_majority", "mountain_pattern")
    coefficients = {
        steer_v5.v1.vector_key(source, target): torch.nn.Parameter(torch.zeros(2))
        for source in patterns
        for target in patterns
        if source != target
    }
    identity = torch.eye(2)
    stats = {
        "calibration_coefficients": coefficients,
        "centroids": {
            "sorted_ascending": torch.tensor([0.0, 0.0]),
            "sorted_descending": torch.tensor([1.0, 0.0]),
            "has_majority": torch.tensor([0.0, 1.0]),
            "mountain_pattern": torch.tensor([-1.0, 0.0]),
        },
        "inv_sqrt_cov": {pattern: identity for pattern in patterns},
        "pca_components": identity,
        "sqrt_cov": {pattern: identity for pattern in patterns},
        "transport_stds": {pattern: torch.ones(2) for pattern in patterns},
    }
    classifier = torch.nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]))
    loss = steer_v5.compute_train_example_loss(
        classifier=classifier,
        source="sorted_ascending",
        stats=stats,
        subject_id="subject-1",
        target="sorted_descending",
        z=torch.tensor([0.0, 0.0]),
    )
    loss.backward()
    matched_key = steer_v5.v1.vector_key("sorted_ascending", "sorted_descending")
    control_key = steer_v5.v1.vector_key("has_majority", "sorted_descending")
    assert coefficients[matched_key].grad is not None
    assert coefficients[matched_key].grad.abs().sum().item() > 0.0
    control_grad = coefficients[control_key].grad
    assert control_grad is None or control_grad.abs().sum().item() == 0.0


def test_v5_random_controls_are_reproducible_from_single_generator_stream() -> None:
    kwargs = {
        "matched_displacement_norm": 3.0,
        "random_controls": 4,
        "source": "sorted_ascending",
        "subject_id": "subject-1",
        "target": "sorted_descending",
        "z": torch.zeros(5),
    }
    first = steer_v5.random_norm_matched_vectors(**kwargs)
    second = steer_v5.random_norm_matched_vectors(**kwargs)
    assert len(first) == 4
    assert all(torch.allclose(left, right) for left, right in zip(first, second))
    assert all(abs(vector.norm().item() - 3.0) < 1e-6 for vector in first)


def test_v5_contract_failures_abort_before_training() -> None:
    train_payload = {
        "claim_scope": "wrong",
        "pool": "train",
        "subjects": [],
    }
    eval_payload = {
        "claim_scope": "four_behavior_representation_steering_v5_source_pool",
        "pool": "development",
        "subjects": [],
    }
    combined_audit = {
        "claim_scope": "four_behavior_representation_steering_v5_source_pool_construction",
        "overlap_counts": {},
        "passed": True,
        "pool_summaries": {
            "train": {"accepted_counts_by_behavior": {pattern: 64 for pattern in steer_v5.PATTERNS}},
            "development": {"accepted_counts_by_behavior": {pattern: 24 for pattern in steer_v5.PATTERNS}},
            "final": {"accepted_counts_by_behavior": {pattern: 24 for pattern in steer_v5.PATTERNS}},
        },
        "seed_preflight": {"passed": True},
    }
    final_redacted = {
        "claim_scope": "redacted_final_steering_v5_source_pool_audit_surface_only",
        "summary": {"accepted_counts_by_behavior": {pattern: 24 for pattern in steer_v5.PATTERNS}},
    }
    failures = steer_v5.validate_source_pool_contract(
        train_payload=train_payload,
        eval_payload=eval_payload,
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )
    assert any("train source pool" in failure for failure in failures)


def test_v5_final_authorization_rejects_stale_development_artifact() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        pool_dir = Path(tmp_dir_name)
        (pool_dir / "combined_audit.json").write_text('{"audit": "current"}')
        (pool_dir / "development_subjects.json").write_text('{"pool": "development"}')
        (pool_dir / "final_redacted_audit.json").write_text('{"pool": "final-redacted"}')
        (pool_dir / "train_subjects.json").write_text('{"pool": "train"}')
        development = {
            "claim_scope": "four_behavior_representation_steering_v5_contrastive_residual_calibration_development",
            "combined_audit_sha256": "stale",
            "eval_pool_sha256": steer_v5.v1.sha256_file(pool_dir / "development_subjects.json"),
            "final_redacted_audit_sha256": steer_v5.v1.sha256_file(pool_dir / "final_redacted_audit.json"),
            "next_action": "eligible_for_one_shot_final_eval_without_method_changes",
            "passed": True,
            "phase": "development",
            "train_pool_sha256": steer_v5.v1.sha256_file(pool_dir / "train_subjects.json"),
            "transport_method": "train_contrastive_residual_calibration_on_v4_low_rank_transport",
        }
        failures = steer_v5.validate_development_authorizes_final(
            development=development,
            pool_dir=pool_dir,
        )
    assert any("combined_audit_sha256" in failure for failure in failures)


def test_v5_rejects_non_preregistered_cli_overrides() -> None:
    args = SimpleNamespace(
        generic_negative_cap=1024,
        hard_negative_cap=1024,
        heldout_per_class=64,
        lr=0.003,
        positive_cap=2048,
        random_controls=32,
        source_margin_gate=0.40,
        support_per_class=160,
        train_epochs=351,
    )
    try:
        steer_v5.validate_preregistered_args(args)
    except ValueError as error:
        assert "train_epochs" in str(error)
    else:
        raise AssertionError("V5 accepted non-preregistered train epoch count")


def main() -> None:
    tests = [
        test_v5_rejects_prior_pool_directories,
        test_v5_seed_preflight_uses_preregistered_base_seed,
        test_v5_development_paths_exclude_raw_final,
        test_v5_forbidden_final_raw_guard_rejects_v5_final_before_open,
        test_v5_null_control_is_zero_displacement,
        test_v5_v2_and_v3_controls_use_train_only_formulas,
        test_v5_zero_calibration_matches_uncalibrated_v4_transport,
        test_v5_training_loss_detaches_control_coefficients,
        test_v5_random_controls_are_reproducible_from_single_generator_stream,
        test_v5_contract_failures_abort_before_training,
        test_v5_final_authorization_rejects_stale_development_artifact,
        test_v5_rejects_non_preregistered_cli_overrides,
    ]
    for test in tests:
        test()
    print("four-behavior representation steering V5 helper tests passed")


if __name__ == "__main__":
    main()
