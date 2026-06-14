"""Direct helper tests for V14 signature-gated subspace task-vector editing."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_functional_weight_editing_v14_signature_gated_subspace_task_vectors as v14  # noqa: E402


def test_v14_rejects_prior_v11_final_raw() -> None:
    prior = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v11_pools" / "final_subjects.json"
    try:
        v14.assert_no_forbidden_final_raw_paths([prior], allow_v14_final=False)
    except ValueError as error:
        assert "prior sealed final raw path" in str(error)
    else:
        raise AssertionError("V14 accepted V11 final raw as an input")


def test_v14_rejects_prior_v12_final_raw() -> None:
    prior = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v12_pools" / "final_subjects.json"
    try:
        v14.assert_no_forbidden_final_raw_paths([prior], allow_v14_final=False)
    except ValueError as error:
        assert "prior sealed final raw path" in str(error)
    else:
        raise AssertionError("V14 accepted V12 final raw as an input")


def test_v14_rejects_prior_v13_final_raw() -> None:
    prior = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v13_pools" / "final_subjects.json"
    try:
        v14.assert_no_forbidden_final_raw_paths([prior], allow_v14_final=False)
    except ValueError as error:
        assert "prior sealed final raw path" in str(error)
    else:
        raise AssertionError("V14 accepted V13 final raw as an input")


def test_v14_rejects_own_final_before_authorization() -> None:
    try:
        v14.assert_no_forbidden_final_raw_paths([v14.V14_FINAL_RAW], allow_v14_final=False)
    except ValueError as error:
        assert "V14 final raw path is forbidden" in str(error)
    else:
        raise AssertionError("V14 accepted its final raw before authorization")


def test_v14_development_contract_does_not_hash_final_raw(monkeypatch) -> None:
    def fake_sha256(path: Path) -> str:
        if path.name == "final_subjects.json":
            raise AssertionError("development validation attempted to hash final raw")
        return {
            "development_subjects.json": "devhash",
            "train_subjects.json": "trainhash",
        }.get(path.name, "otherhash")

    monkeypatch.setattr(v14.v1, "sha256_file", fake_sha256)
    counts_64 = {pattern: 64 for pattern in v14.PATTERNS}
    counts_24 = {pattern: 24 for pattern in v14.PATTERNS}
    combined_audit = {
        "claim_scope": v14.SOURCE_AUDIT_SCOPE,
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
        "claim_scope": v14.FINAL_REDACTED_SCOPE,
        "pool_file_sha256": "finalhash",
    }

    failures = v14.validate_source_pool_contract(
        train_path=Path("train_subjects.json"),
        eval_path=Path("development_subjects.json"),
        combined_audit_path=Path("combined_audit.json"),
        final_redacted_path=Path("final_redacted_audit.json"),
        train_payload={"claim_scope": v14.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v14.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )

    assert failures == []


def test_v14_contract_rejects_forbidden_final_public_detail_keys(monkeypatch) -> None:
    monkeypatch.setattr(v14.v1, "sha256_file", lambda path: {
        "development_subjects.json": "devhash",
        "train_subjects.json": "trainhash",
    }.get(path.name, "otherhash"))
    counts_64 = {pattern: 64 for pattern in v14.PATTERNS}
    counts_24 = {pattern: 24 for pattern in v14.PATTERNS}
    combined_audit = {
        "claim_scope": v14.SOURCE_AUDIT_SCOPE,
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
        "claim_scope": v14.FINAL_REDACTED_SCOPE,
        "pool_file_sha256": "finalhash",
        "summary": {"signature_hash": "leaked"},
    }

    failures = v14.validate_source_pool_contract(
        train_path=Path("train_subjects.json"),
        eval_path=Path("development_subjects.json"),
        combined_audit_path=Path("combined_audit.json"),
        final_redacted_path=Path("final_redacted_audit.json"),
        train_payload={"claim_scope": v14.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v14.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )

    joined = "\n".join(failures)
    assert "forbidden final detail keys" in joined
    assert "records" in joined
    assert "weights_hash" in joined
    assert "signature_hash" in joined


def test_v14_interpolates_in_raw_weight_space() -> None:
    source = torch.tensor([0.0, 10.0, -10.0])
    target = torch.tensor([100.0, -10.0, 30.0])
    edited = v14.interpolate_weights(source_weights=source, target_weights=target, alpha=0.975)
    assert torch.allclose(edited, torch.tensor([97.5, -9.5, 29.0]))


def test_v14_shuffled_target_uses_lexicographic_candidates(monkeypatch) -> None:
    monkeypatch.setattr(v14, "stable_hash_json", lambda payload: "0000000000000001")
    selected = v14.select_shuffled_target(
        subject_id="subject",
        source="sorted_ascending",
        target="has_majority",
    )
    assert selected == "sorted_descending"


def test_v14_random_controls_are_raw_norm_matched_and_deterministic() -> None:
    source = torch.zeros(345, dtype=torch.float32)
    controls_a = v14.random_weight_delta_controls(
        subject_id="subject",
        source="sorted_ascending",
        target="has_majority",
        source_weights=source,
        matched_delta_norm=torch.tensor(7.0),
        random_controls=4,
    )
    controls_b = v14.random_weight_delta_controls(
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


def test_v14_source_target_split_counts_match_preregistration() -> None:
    for direction, expected in v14.EXPECTED_SPLIT_COUNTS.items():
        source, target = direction.split("_to_")
        split = v14.source_target_split(source=source, target=target)
        assert split["compatible_count"] == expected["compatible"]
        assert split["conflict_count"] == expected["conflict"]


def test_v14_support_split_counts_match_preregistration() -> None:
    for direction, expected in v14.EXPECTED_SUPPORT_SPLIT_COUNTS.items():
        source, target = direction.split("_to_")
        split = v14.source_target_support_split(source=source, target=target)
        assert split["compatible_count"] == expected["compatible"]
        assert split["conflict_count"] == expected["conflict"]


def test_v14_differentiable_signature_has_preregistered_layout() -> None:
    weights = torch.zeros(345, dtype=torch.float32)
    probe_examples = v14.build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    signature = v14.differentiable_signature(weights, probe_examples)
    assert signature.shape == (560,)
    assert torch.isfinite(signature).all()


def test_v14_support_tensors_use_source_logits_for_compatible_mse() -> None:
    source_weights = torch.zeros(345, dtype=torch.float32)
    tensors = v14.prepare_support_tensors_with_source_logits(
        source_weights=source_weights,
        source="sorted_descending",
        target="has_majority",
    )
    expected = v14.EXPECTED_SUPPORT_SPLIT_COUNTS["sorted_descending_to_has_majority"]
    assert tensors["target_inputs"].shape[0] == 320
    assert tensors["target_labels"].shape[0] == 320
    assert tensors["compatible_inputs"].shape[0] == expected["compatible"]
    assert tensors["conflict_inputs"].shape[0] == expected["conflict"]
    assert tensors["compatible_source_logits"].shape[0] == expected["compatible"]


def test_v14_alignment_preserves_target_function_under_permutation() -> None:
    base = torch.arange(345, dtype=torch.float32) / 100.0
    layers = v14.unpack_subject_weights(base)
    weights = layers["weights"]
    biases = layers["biases"]
    permutation = torch.tensor([2, 0, 1, 3, 4, 5, 6, 7], dtype=torch.long)
    weights[0] = weights[0][permutation]
    biases[0] = biases[0][permutation]
    weights[1] = weights[1][:, permutation]
    target = v14.pack_subject_weights(weights, biases)

    aligned = v14.align_target_to_source(source_weights=base, target_weights=target)
    assert torch.allclose(aligned, base)


def test_v14_project_delta_to_centered_train_subspace() -> None:
    mean_delta = torch.tensor([1.0, 2.0, 3.0])
    basis = torch.tensor([[1.0, 0.0, 0.0]])
    delta = torch.tensor([4.0, 7.0, 9.0])
    projected = v14.project_delta_to_subspace(
        delta,
        {"basis": basis, "mean_delta": mean_delta},
    )
    assert torch.allclose(projected, torch.tensor([4.0, 2.0, 3.0]))


def test_v14_signature_target_weights_are_top_k_sorted_and_softmaxed() -> None:
    records = [
        {"subject_id": "b", "signature": [1.0, 0.0]},
        {"subject_id": "a", "signature": [1.0, 0.0]},
        {"subject_id": "c", "signature": [3.0, 0.0]},
    ]
    stats = {
        "sig_mean": torch.zeros(2),
        "sig_std": torch.ones(2),
    }
    result = v14.signature_target_weights(
        records,
        signature_target_norm=torch.tensor([1.0, 0.0]),
        train_stats=stats,
        top_k=2,
        temperature=1.0,
    )

    assert [item[1] for item in result["selected"]] == ["a", "b"]
    assert torch.allclose(result["softmax_weights"], torch.tensor([0.5, 0.5]))


def test_v14_scale_selection_tie_prefers_zero_scale() -> None:
    source = torch.zeros(345, dtype=torch.float32)
    edited, metadata = v14.select_scaled_task_vector_weights(
        source_weights=source,
        source="sorted_descending",
        target="has_majority",
        direction=torch.zeros_like(source),
    )

    assert torch.allclose(edited, source)
    assert metadata["selected_scale"] == 0.0


def test_v14_repair_style_interpolation_is_finite_and_shape_preserving() -> None:
    source = torch.zeros(345, dtype=torch.float32)
    target = torch.ones(345, dtype=torch.float32) * 0.01
    repaired, metadata = v14.repair_style_aligned_interpolation_weights(
        source_weights=source,
        aligned_target_weights=target,
        source="sorted_descending",
        target="has_majority",
    )

    assert repaired.shape == source.shape
    assert torch.isfinite(repaired).all()
    assert len(metadata["repair_layers"]) == 5
    assert metadata["support_union_count"] > 0


def test_v14_gate_failures_include_conflict_and_compatible_gates() -> None:
    aggregate = {
        "individual_all_gate_pass_rate": 1.0,
        "mean_conflict_target_accuracy": 1.0,
        "mean_conflict_target_accuracy_improvement": 1.0,
        "mean_full_retrieval_minus_matched_compatible_source_output_mse": 30.0,
        "mean_matched_minus_full_retrieval_target_margin": -0.10,
        "mean_matched_minus_source_signature_target_margin": 0.10,
        "mean_matched_minus_target_label_target_margin": 0.10,
        "mean_matched_minus_shuffled_signature_target_margin": 0.10,
        "mean_matched_minus_v13_no_signature_target_margin": 0.10,
        "mean_matched_target_margin": 0.60,
        "mean_source_signature_minus_matched_compatible_source_output_mse": 3.0,
        "mean_target_label_minus_matched_compatible_source_output_mse": 3.0,
        "mean_shuffled_signature_minus_matched_compatible_source_output_mse": 3.0,
        "mean_v13_no_signature_minus_matched_compatible_source_output_mse": 3.0,
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
            "mean_matched_minus_source_signature_target_margin": 0.10,
            "mean_matched_minus_target_label_target_margin": 0.10,
            "mean_matched_minus_shuffled_signature_target_margin": 0.10,
            "mean_matched_minus_v13_no_signature_target_margin": 0.10,
            "mean_matched_target_margin": 0.50,
            "mean_source_signature_minus_matched_compatible_source_output_mse": 3.0,
            "mean_target_label_minus_matched_compatible_source_output_mse": 3.0,
            "mean_shuffled_signature_minus_matched_compatible_source_output_mse": 3.0,
            "mean_v13_no_signature_minus_matched_compatible_source_output_mse": 3.0,
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
            {"control_type": "v13_no_signature_support_optimizer"},
            {"control_type": "target_label_centroid_task_vector"},
            {"control_type": "nearest_signature_task_vector"},
            {"control_type": "uniform_average_task_vector"},
            {"control_type": "shuffled_signature_weighted_task_vector"},
            {"control_type": "source_signature_weighted_task_vector"},
            {"control_type": "ties_trimmed_sign_task_vector"},
            {"control_type": "repair_style_aligned_interpolation"},
            {"control_type": "random_same_rank_subspace_task_vector"},
            {"control_type": "random_neuron_permutation_task_vector"},
            {"control_type": "no_alignment_task_vector"},
            *[
                {"control_type": f"random_norm_matched_weight_delta:{index:02d}"}
                for index in range(v14.THRESHOLDS["random_controls_per_record"])
            ],
        ],
        "matched": {
            "compatible_count": v14.EXPECTED_SPLIT_COUNTS["sorted_descending_to_has_majority"]["compatible"],
            "conflict_count": v14.EXPECTED_SPLIT_COUNTS["sorted_descending_to_has_majority"]["conflict"],
        },
        "random_control_count": 16,
        "source_behavior": "sorted_descending",
        "subject_id": "subject",
        "target_behavior": "has_majority",
    }]

    failures = v14.gate_failures(
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
    test_v14_rejects_prior_v11_final_raw()
    test_v14_rejects_prior_v12_final_raw()
    test_v14_rejects_prior_v13_final_raw()
    test_v14_rejects_own_final_before_authorization()
    test_v14_interpolates_in_raw_weight_space()
    test_v14_random_controls_are_raw_norm_matched_and_deterministic()
    test_v14_source_target_split_counts_match_preregistration()
    test_v14_support_split_counts_match_preregistration()
    test_v14_differentiable_signature_has_preregistered_layout()
    test_v14_support_tensors_use_source_logits_for_compatible_mse()
    test_v14_alignment_preserves_target_function_under_permutation()
    test_v14_project_delta_to_centered_train_subspace()
    test_v14_signature_target_weights_are_top_k_sorted_and_softmaxed()
    test_v14_scale_selection_tie_prefers_zero_scale()
    test_v14_repair_style_interpolation_is_finite_and_shape_preserving()
    test_v14_gate_failures_include_conflict_and_compatible_gates()
    print("V14 direct helper tests passed; run pytest for fixture-based leak checks.")
