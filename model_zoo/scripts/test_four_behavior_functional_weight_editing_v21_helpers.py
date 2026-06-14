import copy
from pathlib import Path

import pytest
import torch

import train_four_behavior_functional_weight_editing_v20_signature_conditioned_tangent_nullspace_editor as v20
import train_four_behavior_functional_weight_editing_v21_behavioral_probe_residual_output_editor as v21


def test_v21_fresh_scope_paths_and_pool_seeds() -> None:
    assert v21.EDITOR_METHOD == "behavioral_probe_residual_output_editor_v21"
    assert "v21" in str(v21.DEFAULT_POOL_DIR)
    assert "v21" in str(v21.DEFAULT_OUTPUT_DIR)
    assert v21.POOL_CONFIGS["train"]["base_seed"] == 114400000
    assert v21.POOL_CONFIGS["development"]["base_seed"] == 115400000
    assert v21.POOL_CONFIGS["final"]["base_seed"] == 116400000
    assert v21.POOL_CONFIGS["train"]["base_seed"] != v20.POOL_CONFIGS["train"]["base_seed"]
    assert v21.EXPECTED_CONTROLS_PER_RECORD == 27
    assert v21.RANDOM_CONTROLS_PER_RECORD == 16


def test_v21_output_layer_theta_round_trip_only_changes_output_layer() -> None:
    weights = torch.arange(v21.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    theta = v21.output_layer_theta(weights)
    assert theta.shape == (9,)
    assert torch.equal(theta[:8], weights[336:344])
    assert theta[8].item() == weights[344].item()

    replacement = torch.linspace(-1.0, 1.0, 9, dtype=torch.float32)
    edited = v21.replace_output_layer_theta(weights, replacement)
    assert torch.equal(edited[:336], weights[:336])
    assert torch.equal(edited[336:344], replacement[:8])
    assert edited[344].item() == replacement[8].item()


def test_v21_final_raw_guard_rejects_any_runs_final_subjects_path() -> None:
    with pytest.raises(ValueError):
        v21.assert_no_forbidden_final_raw_paths([v21.V21_FINAL_RAW])
    with pytest.raises(ValueError):
        v21.assert_no_forbidden_final_raw_paths([v20.V20_FINAL_RAW])
    with pytest.raises(ValueError):
        v21.assert_no_forbidden_final_raw_paths([
            Path("/Users/max/Desktop/muat/runs/other_experiment/final_subjects.json")
        ])
    v21.assert_no_forbidden_final_raw_paths([Path("runs/not_final/train_subjects.json")])


def test_v21_final_redaction_exact_allowlists_fail_closed() -> None:
    payload = {
        "behavior_suite_hashes": {},
        "candidate_pool_summary_hash": "a",
        "claim_scope": v21.FINAL_REDACTED_SCOPE,
        "config_hash": "b",
        "pool": "final",
        "pool_file_sha256": "c",
        "pool_redacted_payload_sha256": "d",
        "summary": {
            "accepted_counts_by_behavior": {},
            "max_selected_train_vs_heldout_overlap_count": 0,
        },
        "summary_payload_sha256": "e",
    }
    assert v21.forbidden_final_redacted_keys(payload) == []
    leaked = copy.deepcopy(payload)
    leaked["summary"]["subject_ids"] = ["leak"]
    assert "summary.subject_ids" in v21.forbidden_final_redacted_keys(leaked)


def test_v21_base_summary_vector_order_is_frozen(monkeypatch) -> None:
    weights = torch.zeros(v21.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    edited = weights.clone()
    edited[336:344] = 1.0
    edited[344] = 0.5

    monkeypatch.setattr(
        v21.v17,
        "support_objective_for_weights",
        lambda **_kwargs: {
            "compatible_mse": 4.0,
            "conflict_bce": 3.0,
            "objective": 1.0,
            "source_l2": 5.0,
            "target_bce": 2.0,
        },
    )
    monkeypatch.setattr(
        v21.v16.v15.v14,
        "functional_metrics",
        lambda *_args: {
            "conflict_target_accuracy": 7.0,
            "target_margin": 6.0,
        },
    )
    vector = v21.base_edit_summary_vector(
        base_weights=edited,
        source_weights=weights,
        source=v21.PATTERNS[0],
        target=v21.PATTERNS[1],
    )
    assert torch.allclose(
        vector,
        torch.tensor([
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            float(v21.output_layer_theta(edited).norm().item()),
        ]),
    )


def test_v21_raw_probe_descriptor_components_use_frozen_order(monkeypatch) -> None:
    probe_examples = [
        {"sequence": [0, 0, 0, 0, 0]},
        {"sequence": [1, 1, 1, 1, 1]},
    ]
    weights = torch.zeros(v21.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    hidden = torch.tensor(
        [[
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]],
        dtype=torch.float32,
    )
    logits = torch.tensor([[0.5, -1.5]], dtype=torch.float32)

    monkeypatch.setattr(
        v21.v16.v15,
        "hidden_activations_flat_batch",
        lambda flat_weights, inputs: [hidden],
    )
    monkeypatch.setattr(
        v21.v16.v15.v10.decoder_v1,
        "subject_forward_flat_batch",
        lambda flat_weights, inputs: logits,
    )
    components = v21.raw_probe_descriptor_components(
        weights=weights,
        probe_examples=probe_examples,
        output_signature=torch.tensor([0.25, 0.75], dtype=torch.float32),
    )
    assert torch.allclose(components["output_signature"], torch.tensor([0.25, 0.75]))
    assert torch.allclose(components["penultimate_mean"][:2], torch.tensor([2.0, 3.0]))
    assert torch.allclose(components["penultimate_std"][:2], torch.tensor([1.0, 1.0]))
    gram = hidden[0].T @ hidden[0] / 2
    expected_upper = gram[torch.triu_indices(8, 8)[0], torch.triu_indices(8, 8)[1]]
    assert torch.allclose(components["penultimate_gram_upper"], expected_upper)
    assert torch.allclose(
        components["output_logit_stats"],
        torch.tensor([-0.5, 1.0, -1.5, 0.5], dtype=torch.float32),
    )
    assert torch.allclose(components["output_logit_gram"], torch.tensor([(0.25 + 2.25) / 2]))


def test_v21_random_residual_control_is_deterministic_and_norm_matched() -> None:
    selected = torch.tensor([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    first, first_meta = v21.random_norm_matched_residual(
        selected_residual=selected,
        subject_hash="subject-a",
        source=v21.PATTERNS[0],
        target=v21.PATTERNS[1],
        index=0,
        train_statistics_hash="stats",
        decoder_config_hash="decoder",
    )
    second, second_meta = v21.random_norm_matched_residual(
        selected_residual=selected,
        subject_hash="subject-a",
        source=v21.PATTERNS[0],
        target=v21.PATTERNS[1],
        index=0,
        train_statistics_hash="stats",
        decoder_config_hash="decoder",
    )
    assert torch.allclose(first, second)
    assert first_meta["seed_hash"] == second_meta["seed_hash"]
    assert torch.isclose(first.norm(), selected.norm(), atol=1e-5)
    assert first.shape == (9,)


def test_v21_ridge_solve_excludes_intercept_from_penalty() -> None:
    x = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[2.0], [2.0]], dtype=torch.float32)
    weights = v21.solve_ridge_regression(
        x=x,
        y=y,
        ridge_lambda=10.0,
        penalize=torch.tensor([False, True]),
    )
    assert weights is not None
    assert weights.shape == (2, 1)
    assert weights[0, 0] > weights[1, 0]
    assert torch.allclose(x @ weights, torch.full((2, 1), 2.0), atol=0.2)


def test_v21_ridge_solve_fails_closed_after_one_jitter_retry(monkeypatch) -> None:
    calls = []

    def always_fail(lhs, rhs):
        calls.append((lhs, rhs))
        raise RuntimeError("singular")

    monkeypatch.setattr(v21.torch.linalg, "solve", always_fail)
    solution = v21.solve_ridge_regression(
        x=torch.eye(2, dtype=torch.float32),
        y=torch.ones(2, 1, dtype=torch.float32),
        ridge_lambda=0.0,
        penalize=torch.tensor([True, True]),
    )
    assert solution is None
    assert len(calls) == 2


def test_v21_selector_starts_from_output_layer_base_and_preserves_hidden(monkeypatch) -> None:
    source_weights = torch.arange(v21.SOURCE_WEIGHT_DIM, dtype=torch.float32) / 100.0
    base_weights = source_weights.clone()
    base_weights[336:344] += 1.0
    base_weights[344] += 0.25

    monkeypatch.setattr(
        v21.v16,
        "output_layer_no_signature_support_optimizer",
        lambda **_kwargs: (base_weights, {"base": "output"}),
    )
    edited, metadata = v21.select_behavioral_probe_residual_output_edit(
        source_weights=source_weights,
        source_signature_norm=torch.zeros(v21.SIGNATURE_DIM, dtype=torch.float32),
        source=v21.PATTERNS[0],
        target=v21.PATTERNS[1],
        subject={"subject_id": "subject-a"},
        train_stats={"train_statistics_hash": "stats"},
    )
    assert torch.equal(edited[:336], source_weights[:336])
    assert torch.equal(v21.output_layer_theta(edited), v21.output_layer_theta(base_weights))
    assert metadata["control_type"] == v21.EDITOR_METHOD
    assert metadata["residual_scale"] == 0.0
    assert metadata["base_control_type"] == "output_layer_no_signature_support_optimizer"


def test_v21_selector_applies_train_fitted_residual_decoder(monkeypatch) -> None:
    source_weights = torch.zeros(v21.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    base_weights = source_weights.clone()
    base_weights[336] = 2.0
    desired_theta = v21.output_layer_theta(base_weights).clone()
    desired_theta[0] = 3.0
    decoder_weights = torch.zeros(2, 9, dtype=torch.float32)
    decoder_weights[1, 0] = 1.0

    monkeypatch.setattr(
        v21.v16,
        "output_layer_no_signature_support_optimizer",
        lambda **_kwargs: (base_weights, {"base": "output"}),
    )
    monkeypatch.setattr(
        v21,
        "build_decoder_input_vector",
        lambda **_kwargs: torch.tensor([1.0], dtype=torch.float32),
    )
    monkeypatch.setattr(
        v21.v17,
        "support_objective_for_weights",
        lambda **kwargs: {
            "compatible_mse": 0.0,
            "conflict_bce": 0.0,
            "objective": float((v21.output_layer_theta(kwargs["weights"])[0] - 3.0).abs().item()),
            "source_l2": 0.0,
            "target_bce": 0.0,
        },
    )
    edited, metadata = v21.select_behavioral_probe_residual_output_edit(
        source_weights=source_weights,
        source_signature_norm=torch.zeros(v21.SIGNATURE_DIM, dtype=torch.float32),
        source=v21.PATTERNS[0],
        target=v21.PATTERNS[1],
        subject={"subject_id": "subject-a", "signature": [0.0] * v21.SIGNATURE_DIM},
        train_stats={
            "decoder_config_hash": "decoder",
            "decoder_input_mean": torch.zeros(1, dtype=torch.float32),
            "decoder_input_std": torch.ones(1, dtype=torch.float32),
            "decoder_weights": decoder_weights,
            "residual_norm_cap_multiplier": 1.0,
            "residual_output_mean": torch.zeros(9, dtype=torch.float32),
            "residual_output_std": torch.ones(9, dtype=torch.float32),
            "train_statistics_hash": "stats",
        },
    )
    assert torch.allclose(v21.output_layer_theta(edited), desired_theta)
    assert metadata["residual_scale"] == 1.0
    assert metadata["scale_0_selected"] is False
    assert metadata["selected_residual_norm"] > 0.0


def test_v21_descriptor_modes_change_decoder_inputs(monkeypatch) -> None:
    descriptor = torch.arange(1.0, 6.0, dtype=torch.float32)
    source_centroid = torch.ones(5, dtype=torch.float32)
    target_centroid = torch.full((5,), 3.0, dtype=torch.float32)

    monkeypatch.setattr(
        v21,
        "normalized_probe_descriptor",
        lambda **_kwargs: descriptor,
    )
    monkeypatch.setattr(
        v21,
        "base_edit_summary_vector",
        lambda **_kwargs: torch.tensor([9.0, 8.0], dtype=torch.float32),
    )

    common = {
        "subject": {"subject_id": "subject-a", "signature": [0.0] * v21.SIGNATURE_DIM},
        "source_weights": torch.zeros(v21.SOURCE_WEIGHT_DIM, dtype=torch.float32),
        "base_weights": torch.zeros(v21.SOURCE_WEIGHT_DIM, dtype=torch.float32),
        "source": v21.PATTERNS[0],
        "target": v21.PATTERNS[1],
        "train_stats": {
            "probe_descriptor_centroids": {
                v21.PATTERNS[0]: source_centroid,
                v21.PATTERNS[1]: target_centroid,
            },
            "probe_descriptor_by_subject": {
                "nearest-target": target_centroid + 0.25,
            },
            "train_by_behavior": {
                v21.PATTERNS[1]: [{"subject_id": "nearest-target"}],
            },
        },
    }
    matched = v21.build_decoder_input_vector(**common, descriptor_mode="matched_probe")
    no_probe = v21.build_decoder_input_vector(**common, descriptor_mode="no_probe")
    source_probe = v21.build_decoder_input_vector(**common, descriptor_mode="source_probe")
    target_label = v21.build_decoder_input_vector(**common, descriptor_mode="target_label_only")
    oracle_centroid = v21.build_decoder_input_vector(
        **common,
        descriptor_mode="oracle_train_centroid_probe",
    )
    nearest = v21.build_decoder_input_vector(**common, descriptor_mode="nearest_target_probe")
    shuffled = v21.build_decoder_input_vector(
        **common,
        descriptor_mode="shuffled_probe",
        shuffled_probe_descriptor=torch.full((5,), 7.0, dtype=torch.float32),
    )

    dim = int(descriptor.numel())
    assert torch.allclose(matched[:dim], descriptor)
    assert torch.allclose(matched[dim:2 * dim], target_centroid - source_centroid)
    assert torch.allclose(no_probe[:2 * dim], torch.zeros(2 * dim))
    assert torch.allclose(source_probe[:dim], descriptor)
    assert torch.allclose(source_probe[dim:2 * dim], torch.zeros(dim))
    assert torch.allclose(target_label[:2 * dim], torch.zeros(2 * dim))
    assert torch.allclose(oracle_centroid[:dim], target_centroid)
    assert torch.allclose(oracle_centroid[dim:2 * dim], torch.zeros(dim))
    assert torch.allclose(nearest[:dim], target_centroid + 0.25)
    assert torch.allclose(shuffled[:dim], torch.full((dim,), 7.0))
    assert torch.allclose(shuffled[dim:2 * dim], torch.zeros(dim))


def test_v21_build_controls_uses_residual_controls_and_expected_count(monkeypatch) -> None:
    source = v21.PATTERNS[0]
    target = v21.PATTERNS[1]
    source_weights = torch.zeros(v21.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    matched_weights = source_weights.clone()
    matched_weights[336:344] = 0.5
    matched_metadata = {
        "_selected_residual": torch.ones(9, dtype=torch.float32),
        "decoder_config_hash": "decoder",
        "residual_scale": 1.0,
        "train_statistics_hash": "stats",
    }

    monkeypatch.setattr(
        v21,
        "control_record_from_weights",
        lambda control_type, *_args: {"control_type": control_type, "metadata": _args[-1] if _args else {}},
    )
    monkeypatch.setattr(
        v21,
        "select_behavioral_probe_residual_output_edit",
        lambda **kwargs: (matched_weights, {"control_type": kwargs.get("descriptor_mode", "control")}),
    )
    monkeypatch.setattr(
        v21,
        "select_tangent_nullspace_edit",
        lambda **_kwargs: (matched_weights, {"control_type": "v20"}),
    )
    monkeypatch.setattr(
        v21.v17,
        "select_layerwise_rank1_tsv_edit",
        lambda **_kwargs: (matched_weights, {"baseline": "v17"}),
    )
    monkeypatch.setattr(v21.v16, "source_activation_stats", lambda **_kwargs: {})
    monkeypatch.setattr(v21.v16, "target_operator_grid_from_signature", lambda **_kwargs: {})
    monkeypatch.setattr(
        v21.v16,
        "select_compiled_conceptor_edit",
        lambda **_kwargs: (matched_weights, {"baseline": "v16"}),
    )
    monkeypatch.setattr(
        v21.v16,
        "output_layer_no_signature_support_optimizer",
        lambda **_kwargs: (matched_weights, {"baseline": "output"}),
    )
    controls = v21.build_controls(
        subject={"subject_id": "subject-a"},
        source=source,
        target=target,
        source_weights=source_weights,
        source_signature_norm=torch.zeros(v21.SIGNATURE_DIM, dtype=torch.float32),
        shuffled_signature_norm=torch.ones(v21.SIGNATURE_DIM, dtype=torch.float32),
        shuffled_probe_descriptor=torch.zeros(5, dtype=torch.float32),
        matched_weights=matched_weights,
        matched_metadata=matched_metadata,
        train_stats={
            "probe_examples": [{"sequence": [0, 0, 0, 0, 0]}],
            "train_statistics_hash": "stats",
            "v16_baseline_train_stats": {},
            "v16_baseline_train_statistics_hash": "v16",
            "v17_baseline_train_stats": {},
            "v17_baseline_train_statistics_hash": "v17",
        },
        random_controls=v21.RANDOM_CONTROLS_PER_RECORD,
    )
    assert len(controls) == v21.EXPECTED_CONTROLS_PER_RECORD
    control_types = [item["control_type"] for item in controls]
    assert "output_layer_no_signature_support_optimizer" in control_types
    assert "v20_tangent_nullspace_editor_recomputed" in control_types
    assert control_types[-2:] == [
        "random_norm_matched_probe_residual_14",
        "random_norm_matched_probe_residual_15",
    ]


def test_v21_random_residual_controls_attach_to_base_theta(monkeypatch) -> None:
    source = v21.PATTERNS[0]
    target = v21.PATTERNS[1]
    source_weights = torch.zeros(v21.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    base_weights = source_weights.clone()
    base_weights[336] = 2.0
    matched_weights = source_weights.clone()
    matched_weights[336] = 10.0
    matched_metadata = {
        "_selected_residual": torch.ones(9, dtype=torch.float32),
        "decoder_config_hash": "decoder",
        "residual_scale": 1.0,
        "train_statistics_hash": "stats",
    }

    def record_with_theta(control_type, weights, *_args):
        return {"control_type": control_type, "theta": v21.output_layer_theta(weights)}

    monkeypatch.setattr(v21, "control_record_from_weights", record_with_theta)
    monkeypatch.setattr(
        v21,
        "select_behavioral_probe_residual_output_edit",
        lambda **_kwargs: (matched_weights, {"control_type": "descriptor"}),
    )
    monkeypatch.setattr(
        v21,
        "random_norm_matched_residual",
        lambda **_kwargs: (torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), {}),
    )
    monkeypatch.setattr(v21, "select_tangent_nullspace_edit", lambda **_kwargs: (matched_weights, {}))
    monkeypatch.setattr(v21.v17, "select_layerwise_rank1_tsv_edit", lambda **_kwargs: (matched_weights, {}))
    monkeypatch.setattr(v21.v16, "source_activation_stats", lambda **_kwargs: {})
    monkeypatch.setattr(v21.v16, "target_operator_grid_from_signature", lambda **_kwargs: {})
    monkeypatch.setattr(v21.v16, "select_compiled_conceptor_edit", lambda **_kwargs: (matched_weights, {}))
    monkeypatch.setattr(
        v21.v16,
        "output_layer_no_signature_support_optimizer",
        lambda **_kwargs: (base_weights, {"baseline": "output"}),
    )
    controls = v21.build_controls(
        subject={"subject_id": "subject-a"},
        source=source,
        target=target,
        source_weights=source_weights,
        source_signature_norm=torch.zeros(v21.SIGNATURE_DIM, dtype=torch.float32),
        shuffled_signature_norm=torch.ones(v21.SIGNATURE_DIM, dtype=torch.float32),
        shuffled_probe_descriptor=torch.zeros(5, dtype=torch.float32),
        matched_weights=matched_weights,
        matched_metadata=matched_metadata,
        train_stats={
            "probe_examples": [{"sequence": [0, 0, 0, 0, 0]}],
            "train_statistics_hash": "stats",
            "v16_baseline_train_stats": {},
            "v16_baseline_train_statistics_hash": "v16",
            "v17_baseline_train_stats": {},
            "v17_baseline_train_statistics_hash": "v17",
        },
        random_controls=1,
    )
    random_control = controls[-1]
    assert random_control["control_type"] == "random_norm_matched_probe_residual_00"
    assert random_control["theta"][0].item() == 3.0


def test_v21_gate_failures_enforces_total_control_count() -> None:
    aggregate = {
        "mean_conflict_target_accuracy": 1.0,
        "mean_conflict_target_accuracy_improvement": 1.0,
        "mean_matched_minus_best_control_target_margin": 1.0,
        "mean_matched_minus_output_layer_no_signature_target_margin": 1.0,
        "mean_matched_minus_shuffled_signature_target_margin": 1.0,
        "mean_matched_minus_target_label_target_margin": 1.0,
        "mean_matched_minus_v17_target_margin": 1.0,
        "mean_target_margin": 1.0,
        "n": v21.THRESHOLDS["expected_record_count"],
        "individual_all_gate_pass_rate": 1.0,
        "pareto_undominated_rate": 1.0,
        "target_prediction_rate": 1.0,
    }
    records = [{
        "controls": [{"control_type": f"c{i}"} for i in range(v21.EXPECTED_CONTROLS_PER_RECORD - 1)],
        "random_control_count": v21.RANDOM_CONTROLS_PER_RECORD,
        "subject_id": "bad-total-controls",
    }]
    failures = v21.gate_failures(aggregate=aggregate, by_direction={}, records=records)
    assert any("wrong total control count" in failure for failure in failures)


def test_v21_fit_train_statistics_produces_decoder_artifacts(monkeypatch) -> None:
    subjects = []
    for behavior_index, behavior in enumerate(v21.PATTERNS):
        for item_index in range(2):
            value = float(behavior_index + item_index / 10.0)
            subjects.append({
                "pattern": behavior,
                "signature": [value] * v21.SIGNATURE_DIM,
                "subject_id": f"{behavior}-{item_index}",
                "weights": [value] * v21.SOURCE_WEIGHT_DIM,
            })

    monkeypatch.setattr(
        v21,
        "raw_probe_descriptor_components",
        lambda **kwargs: {
            "output_signature": kwargs["output_signature"].detach().clone().to(dtype=torch.float32),
            "penultimate_mean": torch.zeros(8, dtype=torch.float32),
            "penultimate_std": torch.ones(8, dtype=torch.float32),
            "penultimate_gram_upper": torch.zeros(36, dtype=torch.float32),
            "output_logit_stats": torch.zeros(4, dtype=torch.float32),
            "output_logit_gram": torch.zeros(1, dtype=torch.float32),
        },
    )
    monkeypatch.setattr(
        v21.v16,
        "fit_v16_train_statistics",
        lambda subjects, probe_examples: {"train_statistics_hash": "v16"},
    )
    monkeypatch.setattr(
        v21.v17,
        "fit_v17_train_statistics",
        lambda subjects, probe_examples, include_baseline_stats: {"train_statistics_hash": "v17"},
    )
    stats = v21.fit_v21_train_statistics(
        subjects,
        include_models=False,
        include_baseline_stats=True,
        allow_default_small_pool=True,
    )
    assert stats["decoder_weights"].shape[1] == 9
    assert stats["decoder_input_mean"].shape == stats["decoder_input_std"].shape
    assert stats["residual_output_mean"].shape == (9,)
    assert stats["residual_output_std"].shape == (9,)
    assert stats["decoder_config_hash"]
    assert stats["selected_decoder_config"]["selection_mode"] in {
        "default_small_pool",
        "inner_validation",
    }
    assert stats["train_statistics_hash"]


def test_v21_inner_split_uses_51_13_per_behavior() -> None:
    subjects = []
    for behavior in v21.PATTERNS:
        for index in range(64):
            subjects.append({
                "pattern": behavior,
                "signature": [0.0] * v21.SIGNATURE_DIM,
                "subject_id": f"{behavior}-{index:02d}",
                "weights": [0.0] * v21.SOURCE_WEIGHT_DIM,
            })
    split = v21.inner_split_by_behavior(subjects)
    for behavior, payload in split["split_payload"]["per_behavior"].items():
        assert len(payload["inner_train_subject_id_hashes"]) == 51, behavior
        assert len(payload["inner_validation_subject_id_hashes"]) == 13, behavior
    assert len(split["inner_train"]) == 204
    assert len(split["inner_validation"]) == 52


def test_v21_decoder_config_selection_uses_inner_validation(monkeypatch) -> None:
    pairs = []
    for index in range(4):
        pairs.append({
            "decoder_input": torch.tensor([float(index)], dtype=torch.float32),
            "subject_id": f"subject-{index}",
            "target_residual": torch.zeros(9, dtype=torch.float32),
        })
    split = {
        "inner_train": [{"subject_id": "subject-0"}, {"subject_id": "subject-1"}],
        "inner_validation": [{"subject_id": "subject-2"}, {"subject_id": "subject-3"}],
    }

    monkeypatch.setattr(
        v21,
        "fit_decoder_from_pairs",
        lambda pairs, ridge_lambda, residual_target_normalization: {
            "ridge_lambda": ridge_lambda,
            "residual_target_normalization": residual_target_normalization,
        },
    )

    def fake_score(pairs, *, decoder_payload, residual_norm_cap_multiplier, train_stats):
        del train_stats, pairs
        objective = abs(float(decoder_payload["ridge_lambda"]) - 10.0)
        if decoder_payload["residual_target_normalization"] == "none":
            objective += 100.0
        return {
            "mean_compatible_mse": objective,
            "mean_objective": objective,
            "mean_probe_centroid_loss": float(residual_norm_cap_multiplier),
            "mean_residual_norm": 0.0,
            "mean_target_margin": 0.0,
            "scale_0_rate": 0.0,
        }

    monkeypatch.setattr(v21, "evaluate_decoder_config_on_pairs", fake_score)
    config, diagnostics = v21.select_decoder_config(pairs, train_stats={}, split=split)
    assert config == {
        "decoder": "ridge",
        "ridge_lambda": 10.0,
        "residual_norm_cap_multiplier": 0.25,
        "residual_target_normalization": "per_component_train_std",
        "selection_mode": "inner_validation",
    }
    assert diagnostics["candidate_count"] == (
        len(v21.RESIDUAL_TARGET_NORMALIZATIONS)
        * len(v21.DECODER_RIDGE_LAMBDAS)
        * len(v21.RESIDUAL_NORM_CAP_MULTIPLIERS)
    )
