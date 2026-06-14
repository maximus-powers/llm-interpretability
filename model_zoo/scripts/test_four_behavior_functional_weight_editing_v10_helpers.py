"""Direct helper tests for V10 functional weight editing."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_functional_weight_editing_v10_v9_conditioned_delta as v10  # noqa: E402


def test_v10_rejects_prior_final_raw_paths() -> None:
    prior = REPO_ROOT / "runs" / "four_behavior_representation_steering_v9_pools" / "final_subjects.json"
    try:
        v10.assert_no_forbidden_final_raw_paths([prior], allow_v10_final=False)
    except ValueError as error:
        assert "prior sealed final raw path" in str(error)
    else:
        raise AssertionError("V10 accepted a prior final raw pool")


def test_v10_rejects_own_final_before_authorization() -> None:
    try:
        v10.assert_no_forbidden_final_raw_paths([v10.V10_FINAL_RAW], allow_v10_final=False)
    except ValueError as error:
        assert "V10 final raw path is forbidden" in str(error)
    else:
        raise AssertionError("V10 accepted its final raw pool before authorization")


def test_v10_development_contract_does_not_hash_final_raw(monkeypatch) -> None:
    def fake_sha256(path: Path) -> str:
        if path.name == "final_subjects.json":
            raise AssertionError("development validation attempted to hash final raw")
        return {
            "train_subjects.json": "trainhash",
            "development_subjects.json": "devhash",
        }.get(path.name, "otherhash")

    monkeypatch.setattr(v10.v1, "sha256_file", fake_sha256)
    counts_64 = {pattern: 64 for pattern in v10.PATTERNS}
    counts_24 = {pattern: 24 for pattern in v10.PATTERNS}
    combined_audit = {
        "claim_scope": v10.SOURCE_AUDIT_SCOPE,
        "overlap_counts": {},
        "passed": True,
        "pool_summaries": {
            "development": {
                "accepted_counts_by_behavior": counts_24,
                "pool_file_sha256": "devhash",
            },
            "final": {
                "accepted_counts_by_behavior": counts_24,
                "pool_file_sha256": "finalhash",
            },
            "train": {
                "accepted_counts_by_behavior": counts_64,
                "pool_file_sha256": "trainhash",
            },
        },
    }
    final_redacted = {
        "claim_scope": v10.FINAL_REDACTED_SCOPE,
        "pool_file_sha256": "finalhash",
    }

    failures = v10.validate_source_pool_contract(
        train_path=Path("train_subjects.json"),
        eval_path=Path("development_subjects.json"),
        combined_audit_path=Path("combined_audit.json"),
        final_redacted_path=Path("final_redacted_audit.json"),
        train_payload={"claim_scope": v10.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v10.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )

    assert failures == []


def test_v10_contract_rejects_forbidden_final_public_detail_keys(monkeypatch) -> None:
    def fake_sha256(path: Path) -> str:
        if path.name == "final_subjects.json":
            raise AssertionError("development validation attempted to hash final raw")
        return {
            "train_subjects.json": "trainhash",
            "development_subjects.json": "devhash",
        }.get(path.name, "otherhash")

    monkeypatch.setattr(v10.v1, "sha256_file", fake_sha256)
    counts_64 = {pattern: 64 for pattern in v10.PATTERNS}
    counts_24 = {pattern: 24 for pattern in v10.PATTERNS}
    combined_audit = {
        "claim_scope": v10.SOURCE_AUDIT_SCOPE,
        "overlap_counts": {},
        "passed": True,
        "pool_summaries": {
            "development": {
                "accepted_counts_by_behavior": counts_24,
                "pool_file_sha256": "devhash",
            },
            "final": {
                "accepted_counts_by_behavior": counts_24,
                "attempt_count": 3,
                "pool_file_sha256": "finalhash",
                "records": [{"subject_id": "leaked-subject"}],
                "signature_hash": "leaked-signature-hash",
            },
            "train": {
                "accepted_counts_by_behavior": counts_64,
                "pool_file_sha256": "trainhash",
            },
        },
    }
    final_redacted = {
        "claim_scope": v10.FINAL_REDACTED_SCOPE,
        "pool_file_sha256": "finalhash",
        "summary": {
            "heldout_margin": 0.9,
            "weights_hash": "leaked-weight-hash",
        },
    }

    failures = v10.validate_source_pool_contract(
        train_path=Path("train_subjects.json"),
        eval_path=Path("development_subjects.json"),
        combined_audit_path=Path("combined_audit.json"),
        final_redacted_path=Path("final_redacted_audit.json"),
        train_payload={"claim_scope": v10.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v10.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )

    joined = "\n".join(failures)
    assert "forbidden final detail keys" in joined
    assert "attempt_count" in joined
    assert "records" in joined
    assert "signature_hash" in joined
    assert "heldout_margin" in joined
    assert "weights_hash" in joined


def test_v10_nearest_train_retrieval_uses_distance_then_stable_tiebreak() -> None:
    train_stats = {
        "train_subjects": [
            {
                "pattern": "has_majority",
                "signature_hash": "sig_b",
                "subject_id": "subject_b",
                "weights_hash": "weights_b",
            },
            {
                "pattern": "has_majority",
                "signature_hash": "sig_a",
                "subject_id": "subject_a",
                "weights_hash": "weights_a",
            },
            {
                "pattern": "sorted_ascending",
                "signature_hash": "sig_c",
                "subject_id": "subject_c",
                "weights_hash": "weights_c",
            },
        ],
        "train_weights": torch.tensor([
            [2.0, 0.0],
            [1.0, 0.0],
            [9.0, 0.0],
        ]),
        "z_train": torch.tensor([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]),
    }

    nearest = v10.nearest_train_target_retrieval(
        selected_signature_norm=torch.tensor([0.0, 0.0]),
        target="has_majority",
        train_stats=train_stats,
    )

    assert nearest["subject_id"] == "subject_a"
    assert nearest["weights_hash"] == "weights_a"
    assert torch.equal(nearest["weights"], torch.tensor([1.0, 0.0]))


def test_v10_random_weight_delta_controls_are_norm_matched_and_deterministic() -> None:
    train_stats = {
        "weight_mean": torch.zeros(345),
        "weight_std": torch.ones(345),
    }
    source_weight_norm = torch.zeros(345)
    source_weights = torch.zeros(345)
    controls_a = v10.random_weight_delta_controls(
        subject_id="subject",
        source="sorted_ascending",
        target="has_majority",
        source_weights=source_weights,
        source_weight_norm=source_weight_norm,
        matched_delta_norm=torch.tensor(3.0),
        train_stats=train_stats,
        random_controls=4,
    )
    controls_b = v10.random_weight_delta_controls(
        subject_id="subject",
        source="sorted_ascending",
        target="has_majority",
        source_weights=source_weights,
        source_weight_norm=source_weight_norm,
        matched_delta_norm=torch.tensor(3.0),
        train_stats=train_stats,
        random_controls=4,
    )

    assert len(controls_a) == 4
    assert [control["control_type"] for control in controls_a] == [
        "random_norm_matched_weight_delta:00",
        "random_norm_matched_weight_delta:01",
        "random_norm_matched_weight_delta:02",
        "random_norm_matched_weight_delta:03",
    ]
    assert [control["delta_norm"] for control in controls_a] == [
        control["delta_norm"] for control in controls_b
    ]
    for control in controls_a:
        assert abs(control["delta_norm"] - 3.0) < 1e-5


def test_v10_functional_pareto_requires_target_margin_and_source_similarity() -> None:
    matched = {"source_output_mse": 2.0, "target_margin": 0.5}
    higher_margin_worse_similarity = {"source_output_mse": 3.0, "target_margin": 0.6}
    equal_margin_better_similarity = {"source_output_mse": 1.0, "target_margin": 0.5}
    lower_margin_better_similarity = {"source_output_mse": 1.0, "target_margin": 0.4}

    assert not v10.pareto_dominates_functional(higher_margin_worse_similarity, matched)
    assert v10.pareto_dominates_functional(equal_margin_better_similarity, matched)
    assert not v10.pareto_dominates_functional(lower_margin_better_similarity, matched)


def test_v10_summarize_records_counts_core_gates() -> None:
    records = [
        {
            "individual_all_gates_passed": True,
            "matched": {
                "target_margin": 0.4,
                "target_vs_source_margin": 0.3,
            },
            "summary": {
                "matched_minus_best_control_target_margin": 0.1,
                "matched_minus_nearest_train_target_margin": 0.0,
                "matched_minus_no_edit_target_margin": 0.25,
                "nearest_train_minus_matched_source_output_mse": 1.0,
                "pareto_undominated": True,
                "source_margin_change": -0.2,
                "target_prediction_pass": True,
            },
        },
        {
            "individual_all_gates_passed": False,
            "matched": {
                "target_margin": 0.2,
                "target_vs_source_margin": 0.1,
            },
            "summary": {
                "matched_minus_best_control_target_margin": -0.1,
                "matched_minus_nearest_train_target_margin": -0.2,
                "matched_minus_no_edit_target_margin": 0.05,
                "nearest_train_minus_matched_source_output_mse": -1.0,
                "pareto_undominated": False,
                "source_margin_change": 0.0,
                "target_prediction_pass": False,
            },
        },
    ]

    summary = v10.summarize_records(records)

    assert summary["n"] == 2
    assert summary["individual_all_gate_pass_count"] == 1
    assert summary["target_prediction_count"] == 1
    assert summary["pareto_undominated_count"] == 1
    assert summary["individual_all_gate_pass_rate"] == 0.5


if __name__ == "__main__":
    test_v10_rejects_prior_final_raw_paths()
    test_v10_rejects_own_final_before_authorization()
    test_v10_nearest_train_retrieval_uses_distance_then_stable_tiebreak()
    test_v10_random_weight_delta_controls_are_norm_matched_and_deterministic()
    test_v10_functional_pareto_requires_target_margin_and_source_similarity()
    test_v10_summarize_records_counts_core_gates()
    print("V10 helper tests passed")
