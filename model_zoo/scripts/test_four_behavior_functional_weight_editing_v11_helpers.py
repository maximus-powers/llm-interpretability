"""Direct helper tests for V11 retrieval-interpolation editing."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_functional_weight_editing_v11_retrieval_interpolation as v11  # noqa: E402


def test_v11_rejects_prior_v10_final_raw() -> None:
    prior = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v10_pools" / "final_subjects.json"
    try:
        v11.assert_no_forbidden_final_raw_paths([prior], allow_v11_final=False)
    except ValueError as error:
        assert "prior sealed final raw path" in str(error)
    else:
        raise AssertionError("V11 accepted V10 final raw as an input")


def test_v11_rejects_own_final_before_authorization() -> None:
    try:
        v11.assert_no_forbidden_final_raw_paths([v11.V11_FINAL_RAW], allow_v11_final=False)
    except ValueError as error:
        assert "V11 final raw path is forbidden" in str(error)
    else:
        raise AssertionError("V11 accepted its final raw before authorization")


def test_v11_development_contract_does_not_hash_final_raw(monkeypatch) -> None:
    def fake_sha256(path: Path) -> str:
        if path.name == "final_subjects.json":
            raise AssertionError("development validation attempted to hash final raw")
        return {
            "development_subjects.json": "devhash",
            "train_subjects.json": "trainhash",
        }.get(path.name, "otherhash")

    monkeypatch.setattr(v11.v1, "sha256_file", fake_sha256)
    counts_64 = {pattern: 64 for pattern in v11.PATTERNS}
    counts_24 = {pattern: 24 for pattern in v11.PATTERNS}
    combined_audit = {
        "claim_scope": v11.SOURCE_AUDIT_SCOPE,
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
        "claim_scope": v11.FINAL_REDACTED_SCOPE,
        "pool_file_sha256": "finalhash",
    }

    failures = v11.validate_source_pool_contract(
        train_path=Path("train_subjects.json"),
        eval_path=Path("development_subjects.json"),
        combined_audit_path=Path("combined_audit.json"),
        final_redacted_path=Path("final_redacted_audit.json"),
        train_payload={"claim_scope": v11.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v11.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )

    assert failures == []


def test_v11_contract_rejects_forbidden_final_public_detail_keys(monkeypatch) -> None:
    monkeypatch.setattr(v11.v1, "sha256_file", lambda path: {
        "development_subjects.json": "devhash",
        "train_subjects.json": "trainhash",
    }.get(path.name, "otherhash"))
    counts_64 = {pattern: 64 for pattern in v11.PATTERNS}
    counts_24 = {pattern: 24 for pattern in v11.PATTERNS}
    combined_audit = {
        "claim_scope": v11.SOURCE_AUDIT_SCOPE,
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
                "records": [{"subject_id": "leaked"}],
                "weights_hash": "leaked",
            },
            "train": {
                "accepted_counts_by_behavior": counts_64,
                "pool_file_sha256": "trainhash",
            },
        },
    }
    final_redacted = {
        "claim_scope": v11.FINAL_REDACTED_SCOPE,
        "pool_file_sha256": "finalhash",
        "summary": {"signature_hash": "leaked"},
    }

    failures = v11.validate_source_pool_contract(
        train_path=Path("train_subjects.json"),
        eval_path=Path("development_subjects.json"),
        combined_audit_path=Path("combined_audit.json"),
        final_redacted_path=Path("final_redacted_audit.json"),
        train_payload={"claim_scope": v11.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v11.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )

    joined = "\n".join(failures)
    assert "forbidden final detail keys" in joined
    assert "records" in joined
    assert "weights_hash" in joined
    assert "signature_hash" in joined


def test_v11_interpolates_in_raw_weight_space() -> None:
    source = torch.tensor([0.0, 10.0, -10.0])
    target = torch.tensor([100.0, -10.0, 30.0])
    edited = v11.interpolate_weights(source_weights=source, target_weights=target, alpha=0.95)
    assert torch.allclose(edited, torch.tensor([95.0, -9.0, 28.0]))


def test_v11_shuffled_target_uses_lexicographic_candidates(monkeypatch) -> None:
    monkeypatch.setattr(v11, "stable_hash_json", lambda payload: "0000000000000001")
    selected = v11.select_shuffled_target(
        subject_id="subject",
        source="sorted_ascending",
        target="has_majority",
    )
    assert selected == "sorted_descending"


def test_v11_random_controls_are_raw_norm_matched_and_deterministic() -> None:
    source = torch.zeros(345, dtype=torch.float32)
    controls_a = v11.random_weight_delta_controls(
        subject_id="subject",
        source="sorted_ascending",
        target="has_majority",
        source_weights=source,
        matched_delta_norm=torch.tensor(7.0),
        random_controls=4,
    )
    controls_b = v11.random_weight_delta_controls(
        subject_id="subject",
        source="sorted_ascending",
        target="has_majority",
        source_weights=source,
        matched_delta_norm=torch.tensor(7.0),
        random_controls=4,
    )

    assert [control["control_type"] for control in controls_a] == [
        "random_norm_matched_weight_delta:00",
        "random_norm_matched_weight_delta:01",
        "random_norm_matched_weight_delta:02",
        "random_norm_matched_weight_delta:03",
    ]
    assert [control["random_index"] for control in controls_a] == [0, 1, 2, 3]
    assert [control["random_seed"] for control in controls_a] == [
        control["random_seed"] for control in controls_b
    ]
    for control in controls_a:
        assert abs(control["delta_norm"] - 7.0) < 1e-5


def test_v11_gate_failures_include_per_direction_source_similarity() -> None:
    aggregate = {
        "individual_all_gate_pass_rate": 1.0,
        "mean_full_retrieval_minus_matched_source_output_mse": 30.0,
        "mean_matched_minus_full_retrieval_target_margin": -0.10,
        "mean_matched_target_margin": 0.60,
        "mean_matched_target_vs_source_margin": 0.60,
        "n": 288,
        "pareto_undominated_rate": 1.0,
        "target_prediction_rate": 1.0,
    }
    by_direction = {
        "sorted_descending_to_has_majority": {
            "individual_all_gate_pass_rate": 0.90,
            "mean_full_retrieval_minus_matched_source_output_mse": -1.0,
            "mean_matched_target_margin": 0.50,
            "n": 24,
            "pareto_undominated_rate": 0.90,
            "target_prediction_rate": 1.0,
        }
    }
    records = [{
        "controls": [
            {"control_type": "full_nearest_target_retrieval"},
            *[
                {"control_type": f"dummy:{index}"}
                for index in range(v11.THRESHOLDS["expected_controls_per_record"] - 1)
            ],
        ],
        "random_control_count": 32,
        "subject_id": "subject",
    }]

    failures = v11.gate_failures(
        aggregate=aggregate,
        by_direction=by_direction,
        records=records,
    )

    assert any(
        "sorted_descending_to_has_majority full-retrieval-minus-matched source-output MSE"
        in failure
        for failure in failures
    )


if __name__ == "__main__":
    test_v11_rejects_prior_v10_final_raw()
    test_v11_rejects_own_final_before_authorization()
    test_v11_interpolates_in_raw_weight_space()
    test_v11_random_controls_are_raw_norm_matched_and_deterministic()
    test_v11_gate_failures_include_per_direction_source_similarity()
    print("V11 direct helper tests passed; run pytest for fixture-based leak checks.")
