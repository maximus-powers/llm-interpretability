import copy
from pathlib import Path

import pytest
import torch

import train_four_behavior_functional_weight_editing_v20_signature_conditioned_tangent_nullspace_editor as v20
import train_four_behavior_functional_weight_editing_v22_component_activation_rank1_editor as v22


def tiny_hidden_descriptor(value: float = 0.0) -> dict[str, list[torch.Tensor]]:
    return {
        "hbar": [torch.full((8,), value + layer, dtype=torch.float32) for layer in v22.HIDDEN_LAYERS],
        "xbar": [
            torch.full((5 if layer == 0 else 8,), 1.0 + value, dtype=torch.float32)
            for layer in v22.HIDDEN_LAYERS
        ],
    }


def tiny_subjects(per_behavior: int = 2) -> list[dict[str, object]]:
    subjects = []
    for behavior_index, behavior in enumerate(v22.PATTERNS):
        for item_index in range(per_behavior):
            value = float(behavior_index + item_index / 10.0)
            subjects.append({
                "pattern": behavior,
                "signature": [value] * v22.SIGNATURE_DIM,
                "subject_id": f"{behavior}-{item_index}",
                "weights": [value] * v22.SOURCE_WEIGHT_DIM,
            })
    return subjects


def test_v22_fresh_scope_paths_and_pool_seeds() -> None:
    assert v22.EDITOR_METHOD == "component_activation_rank1_editor_v22"
    assert v22.PASSING_DEVELOPMENT_NEXT_ACTION == "run_hash_bound_final_after_reviewer_authorization"
    assert v22.FAILING_DEVELOPMENT_NEXT_ACTION == "log_negative_development_result_do_not_open_final_raw"
    assert "v22" in str(v22.DEFAULT_POOL_DIR)
    assert "v22" in str(v22.DEFAULT_OUTPUT_DIR)
    assert v22.POOL_CONFIGS["train"]["base_seed"] == 117400000
    assert v22.POOL_CONFIGS["development"]["base_seed"] == 118400000
    assert v22.POOL_CONFIGS["final"]["base_seed"] == 119400000
    assert v22.POOL_CONFIGS["train"]["base_seed"] != v20.POOL_CONFIGS["train"]["base_seed"]
    assert v22.EXPECTED_CONTROLS_PER_RECORD == 27
    assert v22.RANDOM_CONTROLS_PER_RECORD == 16


def test_v22_final_raw_guard_rejects_current_prior_and_generic_runs_final_paths() -> None:
    with pytest.raises(ValueError):
        v22.assert_no_forbidden_final_raw_paths([v22.V22_FINAL_RAW])
    with pytest.raises(ValueError):
        v22.assert_no_forbidden_final_raw_paths([v20.V20_FINAL_RAW])
    with pytest.raises(ValueError):
        v22.assert_no_forbidden_final_raw_paths([
            Path("/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v21_pools/final_subjects.json")
        ])
    with pytest.raises(ValueError):
        v22.assert_no_forbidden_final_raw_paths([
            Path("/Users/max/Desktop/muat/runs/other_experiment/final_subjects.json")
        ])
    v22.assert_no_forbidden_final_raw_paths([Path("runs/not_final/train_subjects.json")])


def test_v22_final_redaction_exact_allowlists_fail_closed() -> None:
    payload = {
        "behavior_suite_hashes": {},
        "candidate_pool_summary_hash": "a",
        "claim_scope": v22.FINAL_REDACTED_SCOPE,
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
    assert v22.forbidden_final_redacted_keys(payload) == []
    leaked = copy.deepcopy(payload)
    leaked["summary"]["subject_ids"] = ["leak"]
    assert "summary.subject_ids" in v22.forbidden_final_redacted_keys(leaked)


def test_hidden_rank1_descriptor_uses_layer_inputs_and_outputs(monkeypatch) -> None:
    weights = torch.zeros(v22.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    layer_inputs = [
        torch.full((1, 3, 5 if layer == 0 else 8), float(layer + 1), dtype=torch.float32)
        for layer in v22.HIDDEN_LAYERS
    ]
    layer_outputs = [
        torch.full((1, 3, 8), float(layer + 10), dtype=torch.float32)
        for layer in v22.HIDDEN_LAYERS
    ]
    monkeypatch.setattr(
        v22.v17,
        "hidden_inputs_and_outputs_flat_batch",
        lambda *_args: (layer_inputs, layer_outputs),
    )
    descriptor = v22.hidden_rank1_descriptor_for_weights(
        weights=weights,
        probe_examples=[{"sequence": [0, 0, 0, 0, 0]}],
    )
    assert torch.allclose(descriptor["xbar"][0], torch.full((5,), 1.0))
    assert torch.allclose(descriptor["xbar"][4], torch.full((8,), 5.0))
    assert torch.allclose(descriptor["hbar"][0], torch.full((8,), 10.0))
    assert torch.allclose(descriptor["hbar"][4], torch.full((8,), 14.0))


def test_apply_hidden_rank1_edit_only_changes_selected_hidden_component() -> None:
    base = torch.zeros(v22.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    direction = torch.arange(1.0, 9.0, dtype=torch.float32)
    xbar = torch.ones(5, dtype=torch.float32)
    edited, metadata = v22.apply_hidden_rank1_edit(
        base_weights=base,
        layer_index=0,
        direction=direction,
        xbar=xbar,
        ridge_lambda=1.0,
        scale=0.5,
        norm_cap=100.0,
    )
    assert edited is not None
    assert metadata["hidden_delta_clipped"] is False
    changed = torch.nonzero(torch.abs(edited - base) > 0, as_tuple=False).reshape(-1).tolist()
    weight_spec, bias_spec = v22.hidden_layer_specs(0)
    allowed = set(range(weight_spec["start"], weight_spec["end"])) | set(
        range(bias_spec["start"], bias_spec["end"])
    )
    assert set(changed) <= allowed
    assert torch.equal(v22.output_layer_theta(edited), v22.output_layer_theta(base))


def test_target_direction_modes_are_probe_conditioned_and_controlled() -> None:
    descriptor = tiny_hidden_descriptor(1.0)
    shuffled = tiny_hidden_descriptor(9.0)
    train_stats = {
        "global_hidden_centroids": [torch.zeros(8) for _ in v22.HIDDEN_LAYERS],
        "hidden_descriptor_by_subject": {"nearest": tiny_hidden_descriptor(3.0)},
        "hidden_target_centroids": {
            v22.PATTERNS[0]: [torch.full((8,), 2.0) for _ in v22.HIDDEN_LAYERS],
            v22.PATTERNS[1]: [torch.full((8,), 5.0) for _ in v22.HIDDEN_LAYERS],
        },
        "train_by_behavior": {v22.PATTERNS[1]: [{"subject_id": "nearest"}]},
    }
    matched = v22.target_direction_for_mode(
        subject={"subject_id": "source"},
        source=v22.PATTERNS[0],
        target=v22.PATTERNS[1],
        train_stats=train_stats,
        descriptor=descriptor,
        descriptor_mode="matched_probe",
    )
    no_probe = v22.target_direction_for_mode(
        subject={"subject_id": "source"},
        source=v22.PATTERNS[0],
        target=v22.PATTERNS[1],
        train_stats=train_stats,
        descriptor=descriptor,
        descriptor_mode="no_probe",
    )
    shuffled_direction = v22.target_direction_for_mode(
        subject={"subject_id": "source"},
        source=v22.PATTERNS[0],
        target=v22.PATTERNS[1],
        train_stats=train_stats,
        descriptor=descriptor,
        descriptor_mode="shuffled_probe",
        shuffled_hidden_descriptor=shuffled,
    )
    assert torch.allclose(matched[0], torch.full((8,), 4.0))
    assert torch.allclose(no_probe[0], torch.zeros(8))
    assert torch.allclose(shuffled_direction[0], torch.full((8,), -4.0))


def test_random_component_control_is_deterministic_and_norm_matched() -> None:
    base = torch.zeros(v22.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    first, first_meta = v22.random_norm_matched_component_edit(
        base_weights=base,
        source_weights=base,
        xbar=torch.ones(5, dtype=torch.float32),
        matched_norm=2.5,
        layer_index=0,
        ridge_lambda=1.0,
        subject_hash="subject",
        source=v22.PATTERNS[0],
        target=v22.PATTERNS[1],
        index=0,
        train_statistics_hash="stats",
        script_hash="script",
    )
    second, second_meta = v22.random_norm_matched_component_edit(
        base_weights=base,
        source_weights=base,
        xbar=torch.ones(5, dtype=torch.float32),
        matched_norm=2.5,
        layer_index=0,
        ridge_lambda=1.0,
        subject_hash="subject",
        source=v22.PATTERNS[0],
        target=v22.PATTERNS[1],
        index=0,
        train_statistics_hash="stats",
        script_hash="script",
    )
    assert torch.allclose(first, second)
    assert first_meta["seed_hash"] == second_meta["seed_hash"]
    assert abs(float((first - base).norm().item()) - 2.5) < 1e-5
    assert first_meta["control_type"] == "random_norm_matched_component_rank1_00"


def test_build_controls_uses_component_controls_and_expected_count(monkeypatch) -> None:
    source = v22.PATTERNS[0]
    target = v22.PATTERNS[1]
    source_weights = torch.zeros(v22.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    matched_metadata = {
        "_xbar": torch.ones(5, dtype=torch.float32),
        "hidden_delta_norm": 1.0,
        "layer_index": 0,
        "ridge_lambda": 1.0,
    }

    monkeypatch.setattr(
        v22,
        "control_record_from_weights",
        lambda control_type, *_args: {"control_type": control_type, "metadata": _args[-1] if _args else {}},
    )
    monkeypatch.setattr(
        v22,
        "select_component_activation_rank1_edit",
        lambda **kwargs: (source_weights, {"control_type": kwargs.get("descriptor_mode", "matched")}),
    )
    monkeypatch.setattr(v22, "select_tangent_nullspace_edit", lambda **_kwargs: (source_weights, {}))
    monkeypatch.setattr(v22.v17, "select_layerwise_rank1_tsv_edit", lambda **_kwargs: (source_weights, {}))
    monkeypatch.setattr(v22.v16, "source_activation_stats", lambda **_kwargs: {})
    monkeypatch.setattr(v22.v16, "target_operator_grid_from_signature", lambda **_kwargs: {})
    monkeypatch.setattr(v22.v16, "select_compiled_conceptor_edit", lambda **_kwargs: (source_weights, {}))
    monkeypatch.setattr(
        v22.v16,
        "output_layer_no_signature_support_optimizer",
        lambda **_kwargs: (source_weights, {}),
    )
    monkeypatch.setattr(
        v22.v21,
        "select_behavioral_probe_residual_output_edit",
        lambda **_kwargs: (source_weights, {}),
    )
    controls = v22.build_controls(
        subject={"subject_id": "subject-a"},
        source=source,
        target=target,
        source_weights=source_weights,
        source_signature_norm=torch.zeros(v22.SIGNATURE_DIM, dtype=torch.float32),
        shuffled_signature_norm=torch.ones(v22.SIGNATURE_DIM, dtype=torch.float32),
        shuffled_hidden_descriptor=tiny_hidden_descriptor(),
        matched_weights=source_weights,
        matched_metadata=matched_metadata,
        train_stats={
            "probe_examples": [{"sequence": [0, 0, 0, 0, 0]}],
            "train_statistics_hash": "stats",
            "v16_baseline_train_stats": {},
            "v16_baseline_train_statistics_hash": "v16",
            "v17_baseline_train_stats": {},
            "v17_baseline_train_statistics_hash": "v17",
            "v21_baseline_train_stats": {},
            "v21_baseline_train_statistics_hash": "v21",
        },
        random_controls=v22.RANDOM_CONTROLS_PER_RECORD,
    )
    assert len(controls) == v22.EXPECTED_CONTROLS_PER_RECORD
    control_types = [item["control_type"] for item in controls]
    assert "v21_behavioral_probe_residual_output_editor_recomputed" in control_types
    assert "no_probe_component_rank1_editor" in control_types
    assert control_types[-2:] == [
        "random_norm_matched_component_rank1_14",
        "random_norm_matched_component_rank1_15",
    ]


def test_fit_train_statistics_produces_component_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(v22, "build_probe_examples", lambda: [{"sequence": [0, 0, 0, 0, 0]}])
    monkeypatch.setattr(
        v22,
        "hidden_rank1_descriptor_for_weights",
        lambda weights, probe_examples: tiny_hidden_descriptor(float(weights.reshape(-1)[0].item())),
    )
    monkeypatch.setattr(
        v22,
        "probe_logits_for_weights",
        lambda weights, probe_examples: torch.tensor([float(weights.reshape(-1)[0].item())]),
    )
    monkeypatch.setattr(
        v22.v16,
        "fit_v16_train_statistics",
        lambda subjects, probe_examples: {"train_statistics_hash": "v16"},
    )
    monkeypatch.setattr(
        v22.v17,
        "fit_v17_train_statistics",
        lambda subjects, probe_examples, include_baseline_stats: {"train_statistics_hash": "v17"},
    )
    monkeypatch.setattr(
        v22.v21,
        "fit_v21_train_statistics",
        lambda subjects, include_models, include_baseline_stats: {"train_statistics_hash": "v21"},
    )
    stats = v22.fit_v22_train_statistics(
        tiny_subjects(),
        include_models=True,
        include_baseline_stats=True,
    )
    assert stats["component_rank1_config_hash"]
    assert len(stats["hidden_descriptor_hashes"]) == len(tiny_subjects())
    assert set(stats["hidden_target_centroids"]) == set(v22.PATTERNS)
    assert stats["v21_baseline_train_statistics_hash"] == "v21"
    assert stats["train_statistics_hash"]


def test_serializable_stats_artifact_excludes_stale_decoder_fields() -> None:
    stats = {
        "component_rank1_config_hash": "component",
        "global_hidden_centroids": [torch.zeros(8) for _ in v22.HIDDEN_LAYERS],
        "hidden_descriptor_hashes": {"a": "hash"},
        "hidden_target_centroids": {
            behavior: [torch.zeros(8) for _ in v22.HIDDEN_LAYERS]
            for behavior in v22.PATTERNS
        },
        "probe_examples_hash": "probe",
        "target_probe_logit_centroid_hashes": {"a": "hash"},
        "train_statistics_hash": "stats",
        "v16_baseline_train_statistics_hash": "v16",
        "v17_baseline_train_statistics_hash": "v17",
        "v21_baseline_train_statistics_hash": "v21",
    }
    artifact = v22.serializable_stats_artifact(stats, max_workers=2)
    assert artifact["component_rank1_config_hash"] == "component"
    assert artifact["selected_component_rank1_config"]["hidden_layers"] == v22.HIDDEN_LAYERS
    forbidden = {
        "decoder_config_hash",
        "decoder_input_zero_variance_count",
        "decoder_pair_rows_hash",
        "probe_descriptor_zero_variance_count",
        "residual_norm_cap_multiplier",
        "residual_output_zero_variance_count",
        "selected_decoder_config",
        "selected_decoder_diagnostics",
    }
    assert not (set(artifact) & forbidden)


def test_summary_reports_scale_and_layer_collapse_diagnostics() -> None:
    records = [
        {
            "individual_all_gates_passed": True,
            "matched": {
                "conflict_target_accuracy": 1.0,
                "conflict_target_accuracy_improvement": 1.0,
                "editor": {"layer_index": 0, "scale_0_selected": True},
                "pareto_undominated": True,
                "target_margin": 1.0,
                "target_prediction_pass": True,
                **{
                    f"matched_minus_{metric}_target_margin": 1.0
                    for metric in v22.ADVANTAGE_CONTROL_TYPES
                },
                **{
                    f"{metric}_minus_matched_compatible_source_output_mse": 1.0
                    for metric in v22.ADVANTAGE_CONTROL_TYPES
                },
            },
            "summary": {"matched_minus_best_control_target_margin": 1.0},
        },
        {
            "individual_all_gates_passed": False,
            "matched": {
                "conflict_target_accuracy": 0.0,
                "conflict_target_accuracy_improvement": 0.0,
                "editor": {"layer_index": 1, "scale_0_selected": False},
                "pareto_undominated": False,
                "target_margin": 0.0,
                "target_prediction_pass": False,
                **{
                    f"matched_minus_{metric}_target_margin": 0.0
                    for metric in v22.ADVANTAGE_CONTROL_TYPES
                },
                **{
                    f"{metric}_minus_matched_compatible_source_output_mse": 0.0
                    for metric in v22.ADVANTAGE_CONTROL_TYPES
                },
            },
            "summary": {"matched_minus_best_control_target_margin": 0.0},
        },
    ]
    summary = v22.summarize_records(records)
    assert summary["scale_0_selection_rate"] == 0.5
    assert summary["selected_layer_counts"] == {"0": 1, "1": 1}
    assert summary["selected_layer_entropy"] > 0.0
