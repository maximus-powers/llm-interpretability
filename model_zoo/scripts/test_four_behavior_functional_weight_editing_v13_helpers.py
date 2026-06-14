"""Direct helper tests for V13 hybrid signature-support editing."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_functional_weight_editing_v13_hybrid_signature_support_optimization as v13  # noqa: E402


def test_v13_rejects_prior_v11_final_raw() -> None:
    prior = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v11_pools" / "final_subjects.json"
    try:
        v13.assert_no_forbidden_final_raw_paths([prior], allow_v13_final=False)
    except ValueError as error:
        assert "prior sealed final raw path" in str(error)
    else:
        raise AssertionError("V13 accepted V11 final raw as an input")


def test_v13_rejects_prior_v12_final_raw() -> None:
    prior = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v12_pools" / "final_subjects.json"
    try:
        v13.assert_no_forbidden_final_raw_paths([prior], allow_v13_final=False)
    except ValueError as error:
        assert "prior sealed final raw path" in str(error)
    else:
        raise AssertionError("V13 accepted V12 final raw as an input")


def test_v13_rejects_own_final_before_authorization() -> None:
    try:
        v13.assert_no_forbidden_final_raw_paths([v13.V13_FINAL_RAW], allow_v13_final=False)
    except ValueError as error:
        assert "V13 final raw path is forbidden" in str(error)
    else:
        raise AssertionError("V13 accepted its final raw before authorization")


def test_v13_development_contract_does_not_hash_final_raw(monkeypatch) -> None:
    def fake_sha256(path: Path) -> str:
        if path.name == "final_subjects.json":
            raise AssertionError("development validation attempted to hash final raw")
        return {
            "development_subjects.json": "devhash",
            "train_subjects.json": "trainhash",
        }.get(path.name, "otherhash")

    monkeypatch.setattr(v13.v1, "sha256_file", fake_sha256)
    counts_64 = {pattern: 64 for pattern in v13.PATTERNS}
    counts_24 = {pattern: 24 for pattern in v13.PATTERNS}
    combined_audit = {
        "claim_scope": v13.SOURCE_AUDIT_SCOPE,
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
        "claim_scope": v13.FINAL_REDACTED_SCOPE,
        "pool_file_sha256": "finalhash",
    }

    failures = v13.validate_source_pool_contract(
        train_path=Path("train_subjects.json"),
        eval_path=Path("development_subjects.json"),
        combined_audit_path=Path("combined_audit.json"),
        final_redacted_path=Path("final_redacted_audit.json"),
        train_payload={"claim_scope": v13.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v13.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )

    assert failures == []


def test_v13_contract_rejects_forbidden_final_public_detail_keys(monkeypatch) -> None:
    monkeypatch.setattr(v13.v1, "sha256_file", lambda path: {
        "development_subjects.json": "devhash",
        "train_subjects.json": "trainhash",
    }.get(path.name, "otherhash"))
    counts_64 = {pattern: 64 for pattern in v13.PATTERNS}
    counts_24 = {pattern: 24 for pattern in v13.PATTERNS}
    combined_audit = {
        "claim_scope": v13.SOURCE_AUDIT_SCOPE,
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
        "claim_scope": v13.FINAL_REDACTED_SCOPE,
        "pool_file_sha256": "finalhash",
        "summary": {"signature_hash": "leaked"},
    }

    failures = v13.validate_source_pool_contract(
        train_path=Path("train_subjects.json"),
        eval_path=Path("development_subjects.json"),
        combined_audit_path=Path("combined_audit.json"),
        final_redacted_path=Path("final_redacted_audit.json"),
        train_payload={"claim_scope": v13.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v13.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )

    joined = "\n".join(failures)
    assert "forbidden final detail keys" in joined
    assert "records" in joined
    assert "weights_hash" in joined
    assert "signature_hash" in joined


def test_v13_interpolates_in_raw_weight_space() -> None:
    source = torch.tensor([0.0, 10.0, -10.0])
    target = torch.tensor([100.0, -10.0, 30.0])
    edited = v13.interpolate_weights(source_weights=source, target_weights=target, alpha=0.975)
    assert torch.allclose(edited, torch.tensor([97.5, -9.5, 29.0]))


def test_v13_shuffled_target_uses_lexicographic_candidates(monkeypatch) -> None:
    monkeypatch.setattr(v13, "stable_hash_json", lambda payload: "0000000000000001")
    selected = v13.select_shuffled_target(
        subject_id="subject",
        source="sorted_ascending",
        target="has_majority",
    )
    assert selected == "sorted_descending"


def test_v13_random_controls_are_raw_norm_matched_and_deterministic() -> None:
    source = torch.zeros(345, dtype=torch.float32)
    controls_a = v13.random_weight_delta_controls(
        subject_id="subject",
        source="sorted_ascending",
        target="has_majority",
        source_weights=source,
        matched_delta_norm=torch.tensor(7.0),
        random_controls=4,
    )
    controls_b = v13.random_weight_delta_controls(
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


def test_v13_source_target_split_counts_match_preregistration() -> None:
    for direction, expected in v13.EXPECTED_SPLIT_COUNTS.items():
        source, target = direction.split("_to_")
        split = v13.source_target_split(source=source, target=target)
        assert split["compatible_count"] == expected["compatible"]
        assert split["conflict_count"] == expected["conflict"]


def test_v13_support_split_counts_match_preregistration() -> None:
    for direction, expected in v13.EXPECTED_SUPPORT_SPLIT_COUNTS.items():
        source, target = direction.split("_to_")
        split = v13.source_target_support_split(source=source, target=target)
        assert split["compatible_count"] == expected["compatible"]
        assert split["conflict_count"] == expected["conflict"]


def test_v13_differentiable_signature_has_preregistered_layout() -> None:
    weights = torch.zeros(345, dtype=torch.float32)
    probe_examples = v13.build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    signature = v13.differentiable_signature(weights, probe_examples)
    assert signature.shape == (560,)
    assert torch.isfinite(signature).all()


def test_v13_support_tensors_use_source_logits_for_compatible_mse() -> None:
    source_weights = torch.zeros(345, dtype=torch.float32)
    tensors = v13.prepare_support_tensors_with_source_logits(
        source_weights=source_weights,
        source="sorted_descending",
        target="has_majority",
    )
    expected = v13.EXPECTED_SUPPORT_SPLIT_COUNTS["sorted_descending_to_has_majority"]
    assert tensors["target_inputs"].shape[0] == 320
    assert tensors["target_labels"].shape[0] == 320
    assert tensors["compatible_inputs"].shape[0] == expected["compatible"]
    assert tensors["conflict_inputs"].shape[0] == expected["conflict"]
    assert tensors["compatible_source_logits"].shape[0] == expected["compatible"]


def test_v13_alignment_preserves_target_function_under_permutation() -> None:
    base = torch.arange(345, dtype=torch.float32) / 100.0
    layers = v13.unpack_subject_weights(base)
    weights = layers["weights"]
    biases = layers["biases"]
    permutation = torch.tensor([2, 0, 1, 3, 4, 5, 6, 7], dtype=torch.long)
    weights[0] = weights[0][permutation]
    biases[0] = biases[0][permutation]
    weights[1] = weights[1][:, permutation]
    target = v13.pack_subject_weights(weights, biases)

    aligned = v13.align_target_to_source(source_weights=base, target_weights=target)
    assert torch.allclose(aligned, base)


def test_v13_gate_failures_include_conflict_and_compatible_gates() -> None:
    aggregate = {
        "individual_all_gate_pass_rate": 1.0,
        "mean_conflict_target_accuracy": 1.0,
        "mean_conflict_target_accuracy_improvement": 1.0,
        "mean_full_retrieval_minus_matched_compatible_source_output_mse": 30.0,
        "mean_matched_minus_full_retrieval_target_margin": -0.10,
        "mean_matched_minus_no_signature_target_margin": 0.10,
        "mean_matched_minus_shuffled_signature_target_margin": 0.10,
        "mean_matched_target_margin": 0.60,
        "mean_no_signature_minus_matched_compatible_source_output_mse": 3.0,
        "mean_shuffled_signature_minus_matched_compatible_source_output_mse": 3.0,
        "n": 288,
        "pareto_undominated_rate": 1.0,
        "target_prediction_rate": 1.0,
    }
    by_direction = {
        "sorted_descending_to_has_majority": {
            "mean_conflict_target_accuracy": 0.60,
            "mean_conflict_target_accuracy_improvement": 0.10,
            "individual_all_gate_pass_rate": 0.90,
            "mean_full_retrieval_minus_matched_compatible_source_output_mse": -1.0,
            "mean_matched_minus_no_signature_target_margin": 0.10,
            "mean_matched_minus_shuffled_signature_target_margin": 0.10,
            "mean_matched_target_margin": 0.50,
            "mean_no_signature_minus_matched_compatible_source_output_mse": 3.0,
            "mean_shuffled_signature_minus_matched_compatible_source_output_mse": 3.0,
            "n": 24,
            "pareto_undominated_rate": 0.90,
            "target_prediction_rate": 1.0,
        }
    }
    records = [{
        "controls": [
            {"control_type": "no_edit"},
            {"control_type": "aligned_full_nearest_target_retrieval"},
            {"control_type": "aligned_interpolation_alpha_0.975"},
            {"control_type": "no_signature_optimizer"},
            {"control_type": "source_signature_optimizer"},
            {"control_type": "shuffled_signature_optimizer:sorted_ascending"},
            {"control_type": "signature_only_optimizer"},
            {"control_type": "target_only_support_optimizer"},
            *[
                {"control_type": f"random_norm_matched_weight_delta:{index:02d}"}
                for index in range(v13.THRESHOLDS["random_controls_per_record"])
            ],
        ],
        "matched": {
            "compatible_count": v13.EXPECTED_SPLIT_COUNTS["sorted_descending_to_has_majority"]["compatible"],
            "conflict_count": v13.EXPECTED_SPLIT_COUNTS["sorted_descending_to_has_majority"]["conflict"],
        },
        "random_control_count": 16,
        "source_behavior": "sorted_descending",
        "subject_id": "subject",
        "target_behavior": "has_majority",
    }]

    failures = v13.gate_failures(
        aggregate=aggregate,
        by_direction=by_direction,
        records=records,
    )

    assert any(
        "sorted_descending_to_has_majority full-retrieval-minus-matched compatible source-output MSE"
        in failure
        for failure in failures
    )
    assert any(
        "sorted_descending_to_has_majority conflict target accuracy"
        in failure
        for failure in failures
    )


if __name__ == "__main__":
    test_v13_rejects_prior_v11_final_raw()
    test_v13_rejects_prior_v12_final_raw()
    test_v13_rejects_own_final_before_authorization()
    test_v13_interpolates_in_raw_weight_space()
    test_v13_random_controls_are_raw_norm_matched_and_deterministic()
    test_v13_source_target_split_counts_match_preregistration()
    test_v13_support_split_counts_match_preregistration()
    test_v13_differentiable_signature_has_preregistered_layout()
    test_v13_support_tensors_use_source_logits_for_compatible_mse()
    test_v13_alignment_preserves_target_function_under_permutation()
    test_v13_gate_failures_include_conflict_and_compatible_gates()
    print("V13 direct helper tests passed; run pytest for fixture-based leak checks.")
