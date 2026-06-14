import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train_four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor as v24


def test_v24_identity_scopes_and_seeds_are_fresh() -> None:
    assert v24.EDITOR_METHOD == "behavioral_distilled_hypereditor_v24"
    assert "v24" in str(v24.DEFAULT_POOL_DIR)
    assert "v24" in str(v24.DEFAULT_OUTPUT_DIR)
    assert v24.POOL_CONFIGS["train"]["base_seed"] == 123400000
    assert v24.POOL_CONFIGS["development"]["base_seed"] == 124400000
    assert v24.POOL_CONFIGS["final"]["base_seed"] == 125400000
    assert v24.POOL_CONFIGS["train"]["base_seed"] != v24.v23.POOL_CONFIGS["train"]["base_seed"]
    assert v24.SOURCE_POOL_SCOPE == "four_behavior_functional_weight_editing_v24_source_pool"
    assert (
        v24.SOURCE_AUDIT_SCOPE
        == "four_behavior_functional_weight_editing_v24_source_pool_construction"
    )
    assert (
        v24.FINAL_REDACTED_SCOPE
        == "redacted_final_functional_weight_editing_v24_source_pool_audit_surface_only"
    )
    assert (
        v24.DEVELOPMENT_SCOPE
        == "four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor_development"
    )
    assert (
        v24.FINAL_SCOPE
        == "four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor_final"
    )


def test_v24_final_raw_guard_rejects_all_known_final_subjects_paths() -> None:
    with pytest.raises(ValueError):
        v24.assert_no_forbidden_final_raw_paths([v24.V24_FINAL_RAW])
    with pytest.raises(ValueError):
        v24.assert_no_forbidden_final_raw_paths([v24.v23.V23_FINAL_RAW])
    with pytest.raises(ValueError):
        v24.assert_no_forbidden_final_raw_paths([
            Path("/Users/max/Desktop/muat/runs/other/final_subjects.json")
        ])
    v24.assert_no_forbidden_final_raw_paths([Path("runs/v24/train_subjects.json")])


def test_v24_final_redaction_allowlists_fail_closed() -> None:
    payload = {
        "behavior_suite_hashes": {},
        "candidate_pool_summary_hash": "a",
        "claim_scope": v24.FINAL_REDACTED_SCOPE,
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
    assert v24.forbidden_final_redacted_keys(payload) == []
    leaked = {**payload, "records": []}
    assert "top_level.records" in v24.forbidden_final_redacted_keys(leaked)
    leaked_summary = {
        **payload,
        "summary": {**payload["summary"], "subject_ids": ["leak"]},
    }
    assert "summary.subject_ids" in v24.forbidden_final_redacted_keys(leaked_summary)
    nested_leak = {
        **payload,
        "behavior_suite_hashes": {"nested": {"weights": ["leak"]}},
    }
    assert (
        "behavior_suite_hashes.nested.weights"
        in v24.forbidden_final_redacted_keys(nested_leak)
    )


def test_v24_exact_control_contract() -> None:
    required = set(v24.PROOF_CRITICAL_CONTROL_TYPES)
    named = set(v24.REQUIRED_NAMED_CONTROL_TYPES)
    assert required == {
        "no_signature_ablation_behavioral_hypereditor_v24",
        "no_signature_trained_behavioral_hypereditor_v24",
        "shuffled_signature_behavioral_hypereditor_v24",
        "source_behavior_target_ablation_behavioral_hypereditor_v24",
        "v21_behavioral_probe_residual_output_editor_recomputed",
        "v22_component_activation_rank1_editor_recomputed",
        "v23_probe_routed_sparse_subspace_editor_recomputed",
    }
    assert "no_edit" in named
    assert "nearest_train_target_signature_behavioral_hypereditor_v24" in named
    assert "teacher_oracle_support_optimizer_train_protocol_v24" in named
    assert "nearest_train_target_signature_behavioral_hypereditor_v24" not in required
    assert "teacher_oracle_support_optimizer_train_protocol_v24" not in required
    assert v24.EXPECTED_CONTROLS_PER_RECORD == 30
    assert v24.RANDOM_CONTROLS_PER_RECORD == 20


def test_v24_teacher_edit_returns_weight_sized_delta() -> None:
    source = {
        "behavior": v24.PATTERNS[0],
        "subject_id": "s0",
        "weights": [0.0] * v24.SOURCE_WEIGHT_DIM,
        "signature": [0.0] * v24.SIGNATURE_DIM,
    }

    result = v24.optimize_teacher_edit(
        source,
        target_behavior=v24.PATTERNS[1],
        config=v24.TeacherEditConfig(
            steps=2,
            lr=0.01,
            l2_weight=0.001,
            source_compat_weight=0.1,
        ),
    )

    assert result["delta"].shape == (v24.SOURCE_WEIGHT_DIM,)
    assert torch.isfinite(result["delta"]).all()
    assert result["target_behavior"] == v24.PATTERNS[1]
    assert result["source_behavior"] == v24.PATTERNS[0]
    assert result["step_count"] == 2
    assert result["invalid"] is False
    assert result["loss"] >= 0.0


def test_v24_teacher_progress_event_is_count_only(tmp_path: Path) -> None:
    progress_log_path = tmp_path / "teacher_progress.jsonl"

    v24.record_progress_event(
        progress_log_path,
        event="teacher_edit_completed",
        started_at_monotonic=10.0,
        extra={
            "completed_count": 3,
            "record_count": 12,
            "teacher_config_hash": "abc",
        },
        now_monotonic=lambda: 12.0,
    )

    text = progress_log_path.read_text()
    assert '"event": "teacher_edit_completed"' in text
    assert '"completed_count": 3' in text
    assert '"record_count": 12' in text
    assert "subject_id" not in text
    assert "weights" not in text
    assert "signature" not in text


def test_v24_teacher_loss_excludes_conflict_bce(monkeypatch: pytest.MonkeyPatch) -> None:
    source = {
        "behavior": v24.PATTERNS[0],
        "subject_id": "s0",
        "weights": [0.0] * v24.SOURCE_WEIGHT_DIM,
        "signature": [0.0] * v24.SIGNATURE_DIM,
    }
    fake_support = {
        "target_inputs": torch.zeros(2, 5),
        "target_labels": torch.ones(2),
        "conflict_inputs": torch.ones(2, 5),
        "conflict_target_labels": torch.ones(2),
        "compatible_inputs": torch.ones(2, 5) * 2,
        "compatible_source_logits": torch.zeros(2),
    }

    def fake_support_builder(*_args, **_kwargs):
        return fake_support

    def fake_logits(weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        anchor = weights.sum() * 0.0
        if torch.equal(inputs, fake_support["target_inputs"]):
            return anchor + torch.zeros(2)
        if torch.equal(inputs, fake_support["conflict_inputs"]):
            return anchor + torch.full((2,), -100.0)
        return anchor + torch.zeros(2)

    monkeypatch.setattr(
        v24.v23.v16.v15.v14,
        "prepare_support_tensors_with_source_logits",
        fake_support_builder,
    )
    monkeypatch.setattr(v24, "subject_logits_for_inputs", fake_logits)

    result = v24.optimize_teacher_edit(
        source,
        target_behavior=v24.PATTERNS[1],
        config=v24.TeacherEditConfig(
            steps=1,
            lr=0.0,
            l2_weight=0.0,
            source_compat_weight=0.0,
        ),
    )

    expected_target_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros(2),
        torch.ones(2),
    ).item()
    assert result["target_bce"] == pytest.approx(expected_target_bce)
    assert result["loss"] == pytest.approx(expected_target_bce)


def test_v24_hypereditor_input_output_shapes() -> None:
    model = v24.BehavioralDistilledHypereditor(seed=20260624)
    features = torch.zeros(4, v24.HYPEREDITOR_INPUT_DIM)

    delta, scale = model(features)

    assert delta.shape == (4, v24.SOURCE_WEIGHT_DIM)
    assert scale.shape == (4,)
    assert torch.all(scale >= 0.0)
    assert torch.all(scale <= v24.MAX_HYPEREDITOR_SCALE)


def test_v24_hypereditor_features_have_exact_layout() -> None:
    source_weights = torch.arange(v24.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    source_signature = torch.arange(v24.SIGNATURE_DIM, dtype=torch.float32) + 1000
    activation_descriptor = torch.arange(v24.ACTIVATION_DESCRIPTOR_DIM, dtype=torch.float32) + 2000

    features = v24.build_hypereditor_features(
        source_weights_norm=source_weights,
        source_signature_norm=source_signature,
        activation_descriptor_norm=activation_descriptor,
        source_behavior=v24.PATTERNS[0],
        target_behavior=v24.PATTERNS[1],
    )

    assert features.shape == (v24.HYPEREDITOR_INPUT_DIM,)
    assert torch.allclose(features[:345], source_weights)
    assert torch.allclose(features[345:905], source_signature)
    assert torch.allclose(features[905:982], activation_descriptor)
    assert features[982:986].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert features[986:990].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert features[990:].sum().item() == 1.0


def test_v24_activation_descriptor_flattens_hbar_xbar(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = {
        "hbar": [torch.full((8,), float(index)) for index in range(5)],
        "xbar": [
            torch.full((5,), 10.0),
            torch.full((8,), 11.0),
            torch.full((8,), 12.0),
            torch.full((8,), 13.0),
            torch.full((8,), 14.0),
        ],
    }

    monkeypatch.setattr(
        v24.v23,
        "hidden_rank1_descriptor_for_weights",
        lambda *, weights, probe_examples: descriptor,
    )

    actual = v24.activation_descriptor_for_weights(
        torch.zeros(v24.SOURCE_WEIGHT_DIM),
        probe_examples=[],
    )

    assert actual.shape == (v24.ACTIVATION_DESCRIPTOR_DIM,)
    assert actual[:8].tolist() == [0.0] * 8
    assert actual[32:40].tolist() == [4.0] * 8
    assert actual[40:45].tolist() == [10.0] * 5
    assert actual[69:77].tolist() == [14.0] * 8


def test_v24_train_hypereditor_logs_steps_and_is_deterministic(tmp_path: Path) -> None:
    teacher_records = []
    for index in range(4):
        teacher_records.append({
            "activation_descriptor_norm": torch.zeros(v24.ACTIVATION_DESCRIPTOR_DIM),
            "delta": torch.ones(v24.SOURCE_WEIGHT_DIM) * float(index + 1),
            "source_behavior": v24.PATTERNS[index % len(v24.PATTERNS)],
            "source_signature_norm": torch.zeros(v24.SIGNATURE_DIM),
            "source_weights_norm": torch.zeros(v24.SOURCE_WEIGHT_DIM),
            "target_behavior": v24.PATTERNS[(index + 1) % len(v24.PATTERNS)],
        })
    progress_a = tmp_path / "train_a.jsonl"
    progress_b = tmp_path / "train_b.jsonl"
    config = v24.HypereditorTrainingConfig(
        steps=2,
        batch_size=2,
        lr=1e-3,
        seed=123,
        delta_mse_weight=1.0,
        behavior_weight=0.0,
        compat_weight=0.0,
        l2_weight=0.0,
        log_every=1,
    )

    result_a = v24.train_hypereditor_on_teacher_records(
        teacher_records,
        config=config,
        progress_log_path=progress_a,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 11.0,
    )
    result_b = v24.train_hypereditor_on_teacher_records(
        teacher_records,
        config=config,
        progress_log_path=progress_b,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 11.0,
    )

    assert result_a["model_hash"] == result_b["model_hash"]
    assert result_a["step_count"] == 2
    text = progress_a.read_text()
    assert text.count('"event": "hypereditor_training_step"') == 2
    assert "subject_id" not in text
    assert "weights" not in text
    assert "signature" not in text


def test_v24_train_hypereditor_supports_behavior_and_compat_losses() -> None:
    teacher_records = [{
        "activation_descriptor_norm": torch.zeros(v24.ACTIVATION_DESCRIPTOR_DIM),
        "delta": torch.zeros(v24.SOURCE_WEIGHT_DIM),
        "source_behavior": v24.PATTERNS[0],
        "source_signature_norm": torch.zeros(v24.SIGNATURE_DIM),
        "source_weights": torch.zeros(v24.SOURCE_WEIGHT_DIM),
        "source_weights_norm": torch.zeros(v24.SOURCE_WEIGHT_DIM),
        "target_behavior": v24.PATTERNS[1],
    }]

    result = v24.train_hypereditor_on_teacher_records(
        teacher_records,
        config=v24.HypereditorTrainingConfig(
            steps=1,
            batch_size=1,
            lr=1e-3,
            seed=456,
            delta_mse_weight=0.0,
            behavior_weight=1.0,
            compat_weight=1.0,
            l2_weight=0.0,
            log_every=1,
        ),
    )

    assert result["step_count"] == 1
    assert result["behavior_loss"] >= 0.0
    assert result["compat_loss"] >= 0.0


def test_v24_fit_hypereditor_reuses_supplied_teacher_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_records = [{
        "activation_descriptor_norm": torch.zeros(v24.ACTIVATION_DESCRIPTOR_DIM),
        "delta": torch.ones(v24.SOURCE_WEIGHT_DIM),
        "source_behavior": v24.PATTERNS[0],
        "source_signature_norm": torch.zeros(v24.SIGNATURE_DIM),
        "source_weights": torch.zeros(v24.SOURCE_WEIGHT_DIM),
        "source_weights_norm": torch.zeros(v24.SOURCE_WEIGHT_DIM),
        "target_behavior": v24.PATTERNS[1],
    }]

    def fail_build(*_args, **_kwargs):
        raise AssertionError("teacher records should come from cache")

    monkeypatch.setattr(v24, "build_teacher_records", fail_build)

    result = v24.fit_hypereditor_for_config(
        subjects=[],
        config={
            "behavior_weight": 0.0,
            "compat_weight": 0.0,
            "config_hash": "config",
            "config_index": 0,
            "delta_mse_weight": 1.0,
            "hypereditor_steps": 1,
            "teacher_lr": 0.01,
            "teacher_source_compat_weight": 0.25,
            "teacher_steps": 40,
        },
        train_stats={},
        teacher_records=teacher_records,
    )

    assert result["step_count"] == 1
    assert result["model_hash"]


def test_v24_inner_validation_completion_progress_is_bounded_and_redacted() -> None:
    config = {"config_hash": "abc123"}
    candidate = {
        "inner_validation_record_count": -1,
        "invalid": True,
        "invalid_reasons": [
            "plain aggregate failure",
            "contains weights and should be redacted",
        ],
        "mean_target_margin": -0.25,
        "target_prediction_rate": 0.0,
    }

    extra = v24.inner_validation_completion_progress_extra(
        config=config,
        candidate=candidate,
        record_budget=24,
        rung_index=0,
    )

    assert extra["config_hash"] == "abc123"
    assert extra["invalid"] is True
    assert extra["invalid_reason_count"] == 2
    assert extra["invalid_reasons"] == [
        "plain aggregate failure",
        "[redacted_invalid_reason_contains_raw_term]",
    ]
    assert extra["inner_validation_record_count"] == -1
    assert extra["mean_target_margin"] == -0.25
    assert extra["target_prediction_rate"] == 0.0
    assert "weights" not in " ".join(extra["invalid_reasons"])
    assert "signature" not in " ".join(extra["invalid_reasons"])


def _v24_complete_controls() -> list[dict[str, str]]:
    return [
        {"control_type": control_type}
        for control_type in v24.REQUIRED_NAMED_CONTROL_TYPES
    ] + [
        {"control_type": f"random_matched_norm_{index:02d}"}
        for index in range(v24.RANDOM_CONTROLS_PER_RECORD)
    ]


def _v24_low_performing_eval_result() -> dict:
    aggregate = {
        "individual_all_gate_pass_rate": 0.0,
        "mean_matched_minus_best_control_target_margin": -0.35,
        "mean_target_margin": 0.0,
        "n": 24,
        "pareto_undominated_rate": 0.25,
        "target_prediction_rate": 0.25,
    }
    for metric_name in v24.ADVANTAGE_CONTROL_TYPES:
        aggregate[f"mean_matched_minus_{metric_name}_target_margin"] = -0.1
        aggregate[
            f"mean_{metric_name}_minus_matched_compatible_source_output_mse"
        ] = 0.1
    return {
        "aggregate": aggregate,
        "failures": [
            "aggregate target prediction rate 0.250000 < 0.850000",
            "aggregate individual pass rate 0.000000 < 0.850000",
            "aggregate Pareto-undominated rate 0.250000 < 0.850000",
            "mean matched target margin 0.000000 < 0.250000",
            "aggregate best-control target margin advantage -0.350000 < 0.020000",
        ],
        "record_count": 24,
        "records": [
            {
                "controls": _v24_complete_controls(),
                "random_control_count": v24.RANDOM_CONTROLS_PER_RECORD,
                "subject_id": f"subject-{index}",
            }
            for index in range(24)
        ],
    }


def test_v24_inner_validation_does_not_invalidate_low_proof_gate_performance() -> None:
    invalidity = v24.inner_validation_candidate_invalidity(
        result=_v24_low_performing_eval_result(),
        expected_record_count=24,
    )

    assert invalidity["invalid"] is False
    assert invalidity["invalid_reasons"] == []
    assert invalidity["proof_gate_failure_count"] == 5
    assert any(
        "aggregate target prediction rate" in failure
        for failure in invalidity["proof_gate_failures"]
    )


def test_v24_inner_validation_invalidates_contract_failures_only() -> None:
    wrong_count = _v24_low_performing_eval_result()
    wrong_count["aggregate"] = {**wrong_count["aggregate"], "n": 23}
    wrong_count_invalidity = v24.inner_validation_candidate_invalidity(
        result=wrong_count,
        expected_record_count=24,
    )
    assert wrong_count_invalidity["invalid"] is True
    assert any(
        "record count" in reason
        for reason in wrong_count_invalidity["invalid_reasons"]
    )

    missing_control = _v24_low_performing_eval_result()
    missing_control["records"][0] = {
        **missing_control["records"][0],
        "controls": missing_control["records"][0]["controls"][:-1],
    }
    missing_control_invalidity = v24.inner_validation_candidate_invalidity(
        result=missing_control,
        expected_record_count=24,
    )
    assert missing_control_invalidity["invalid"] is True
    assert any(
        "control_count_mismatch" in reason
        or "random_control" in reason
        or "control count" in reason
        for reason in missing_control_invalidity["invalid_reasons"]
    )

    nonfinite = _v24_low_performing_eval_result()
    nonfinite["aggregate"] = {
        **nonfinite["aggregate"],
        "mean_target_margin": float("nan"),
    }
    nonfinite_invalidity = v24.inner_validation_candidate_invalidity(
        result=nonfinite,
        expected_record_count=24,
    )
    assert nonfinite_invalidity["invalid"] is True
    assert "aggregate metric mean_target_margin nonfinite" in nonfinite_invalidity[
        "invalid_reasons"
    ]


def test_v24_inner_validation_worker_forwards_progress_logging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    progress_log_path = tmp_path / "inner_progress.jsonl"
    captured = {}

    def fake_candidate(**kwargs):
        captured.update(kwargs)
        return {"invalid": False}

    monkeypatch.setattr(v24, "inner_validation_candidate_for_config", fake_candidate)

    result = v24._inner_validation_candidate_worker({
        "config": {"config_hash": "abc123"},
        "progress_log_path": progress_log_path,
        "started_at_monotonic": 123.0,
        "teacher_records": [],
        "train_stats": {},
        "train_subjects": [],
        "validation_subjects": [],
    })

    assert result == {"invalid": False}
    assert captured["progress_log_path"] == progress_log_path
    assert captured["started_at_monotonic"] == 123.0


def test_v24_inner_split_has_no_overlap_and_is_balanced() -> None:
    subjects = [
        {
            "behavior": behavior,
            "subject_id": f"{behavior}-{index:02d}",
            "weights": [0.0] * v24.SOURCE_WEIGHT_DIM,
            "signature": [0.0] * v24.SIGNATURE_DIM,
        }
        for behavior in v24.PATTERNS
        for index in range(64)
    ]

    split = v24.inner_train_validation_split(subjects)

    train_ids = {item["subject_id"] for item in split["inner_train_subjects"]}
    valid_ids = {item["subject_id"] for item in split["inner_validation_subjects"]}
    assert not train_ids & valid_ids
    assert len(split["inner_train_subjects"]) == 204
    assert len(split["inner_validation_subjects"]) == 52
    assert all(
        len(split["inner_train_by_behavior"][behavior]) == 51
        for behavior in v24.PATTERNS
    )
    assert all(
        len(split["inner_validation_by_behavior"][behavior]) == 13
        for behavior in v24.PATTERNS
    )
    assert split["inner_split_hash"] == v24.inner_train_validation_split(subjects)[
        "inner_split_hash"
    ]


def test_v24_config_grid_subset_and_rungs_are_exact() -> None:
    full_configs = v24.iter_v24_configs()
    subset = v24.v24_evaluated_config_subset(full_configs)

    assert len(full_configs) == 432
    assert len(subset) == 48
    assert v24.v24_full_config_grid_hash(full_configs) == (
        v24.v24_full_config_grid_hash(v24.iter_v24_configs())
    )
    assert v24.INNER_VALIDATION_RUNG_RECORD_BUDGETS == [24, 72, 156]
    assert v24.INNER_VALIDATION_RUNG_SURVIVORS == [12, 3, 1]
    strata = {}
    for config in subset:
        key = (
            config["teacher_steps"],
            config["teacher_lr"],
            config["hypereditor_steps"],
            config["delta_mse_weight"],
        )
        strata[key] = strata.get(key, 0) + 1
    assert set(strata.values()) == {3}
    assert v24.v24_evaluated_config_subset_hash(subset) == (
        v24.v24_evaluated_config_subset_hash(v24.v24_evaluated_config_subset(full_configs))
    )


def test_v24_evaluate_subjects_logs_progress_and_sorts_records(tmp_path: Path) -> None:
    progress_log_path = tmp_path / "development_progress.jsonl"
    subject = {
        "behavior": v24.PATTERNS[0],
        "subject_id": "subject-0",
        "weights": [0.0] * v24.SOURCE_WEIGHT_DIM,
        "signature": [0.0] * v24.SIGNATURE_DIM,
    }

    def fake_evaluator(job, *, train_stats):
        del train_stats
        matched = {
            "compatible_source_output_mse": 0.0,
            "individual_all_gates_passed": True,
            "min_proof_critical_compatible_mse_advantage": 1.0,
            "pareto_undominated": True,
            "target_margin": 1.0,
            "target_prediction_pass": True,
        }
        for metric_name in v24.ADVANTAGE_CONTROL_TYPES:
            matched[f"matched_minus_{metric_name}_target_margin"] = 1.0
            matched[f"{metric_name}_minus_matched_compatible_source_output_mse"] = 1.0
        return {
            "controls": [
                {"control_type": control_type}
                for control_type in v24.REQUIRED_NAMED_CONTROL_TYPES
            ] + [
                {"control_type": f"random_matched_norm_{index:02d}"}
                for index in range(v24.RANDOM_CONTROLS_PER_RECORD)
            ],
            "individual_all_gates_passed": True,
            "matched": matched,
            "random_control_count": v24.RANDOM_CONTROLS_PER_RECORD,
            "source_behavior": job["source"],
            "subject_id": job["subject"]["subject_id"],
            "summary": {
                "matched_minus_best_control_target_margin": 1.0,
                **{
                    f"matched_minus_{metric_name}_target_margin": 1.0
                    for metric_name in v24.ADVANTAGE_CONTROL_TYPES
                },
                **{
                    f"{metric_name}_minus_matched_compatible_source_output_mse": 1.0
                    for metric_name in v24.ADVANTAGE_CONTROL_TYPES
                },
            },
            "target_behavior": job["target"],
        }

    result = v24.evaluate_subjects(
        subjects=[subject],
        train_stats={},
        record_evaluator=fake_evaluator,
        parallel=False,
        progress_log_path=progress_log_path,
        progress_started_at_monotonic=10.0,
        now_monotonic=lambda: 12.0,
    )

    assert result["record_count"] == len(v24.PATTERNS) - 1
    assert any("record count" in failure for failure in result["failures"])
    assert result["records"] == sorted(
        result["records"],
        key=lambda item: (
            item["source_behavior"],
            item["subject_id"],
            item["target_behavior"],
        ),
    )
    text = progress_log_path.read_text()
    assert '"event": "development_evaluation_jobs_queued"' in text
    assert text.count('"event": "development_evaluation_record_completed"') == (
        len(v24.PATTERNS) - 1
    )
    assert "weights" not in text
    assert "signature" not in text


def test_v24_evaluate_subjects_accepts_rung_record_count_override() -> None:
    subject = {
        "behavior": v24.PATTERNS[0],
        "subject_id": "subject-0",
        "weights": [0.0] * v24.SOURCE_WEIGHT_DIM,
        "signature": [0.0] * v24.SIGNATURE_DIM,
    }

    def fake_evaluator(job, *, train_stats):
        del train_stats
        matched = {
            "compatible_source_output_mse": 0.0,
            "individual_all_gates_passed": True,
            "min_proof_critical_compatible_mse_advantage": 1.0,
            "pareto_undominated": True,
            "target_margin": 1.0,
            "target_prediction_pass": True,
        }
        for metric_name in v24.ADVANTAGE_CONTROL_TYPES:
            matched[f"matched_minus_{metric_name}_target_margin"] = 1.0
            matched[f"{metric_name}_minus_matched_compatible_source_output_mse"] = 1.0
        return {
            "controls": [
                {"control_type": control_type}
                for control_type in v24.REQUIRED_NAMED_CONTROL_TYPES
            ] + [
                {"control_type": f"random_matched_norm_{index:02d}"}
                for index in range(v24.RANDOM_CONTROLS_PER_RECORD)
            ],
            "individual_all_gates_passed": True,
            "matched": matched,
            "random_control_count": v24.RANDOM_CONTROLS_PER_RECORD,
            "source_behavior": job["source"],
            "subject_id": job["subject"]["subject_id"],
            "summary": {
                "matched_minus_best_control_target_margin": 1.0,
                **{
                    f"matched_minus_{metric_name}_target_margin": 1.0
                    for metric_name in v24.ADVANTAGE_CONTROL_TYPES
                },
                **{
                    f"{metric_name}_minus_matched_compatible_source_output_mse": 1.0
                    for metric_name in v24.ADVANTAGE_CONTROL_TYPES
                },
            },
            "target_behavior": job["target"],
        }

    short_result = v24.evaluate_subjects(
        subjects=[subject],
        train_stats={},
        record_evaluator=fake_evaluator,
        parallel=False,
    )
    overridden_result = v24.evaluate_subjects(
        subjects=[subject],
        train_stats={},
        record_evaluator=fake_evaluator,
        parallel=False,
        expected_record_count=len(v24.PATTERNS) - 1,
    )

    assert any("record count" in failure for failure in short_result["failures"])
    assert not any(
        "record count" in failure for failure in overridden_result["failures"]
    )


def test_v24_control_contract_rejects_duplicate_random_indices() -> None:
    controls = [
        {"control_type": control_type}
        for control_type in v24.REQUIRED_NAMED_CONTROL_TYPES
    ] + [
        {"control_type": "random_matched_norm_00"}
        for _index in range(v24.RANDOM_CONTROLS_PER_RECORD)
    ]
    failures = v24.required_control_failures([{
        "controls": controls,
        "source_behavior": v24.PATTERNS[0],
        "subject_id": "s0",
        "target_behavior": v24.PATTERNS[1],
    }])

    assert "record_0_random_control_set_mismatch" in failures


def test_v24_development_result_payload_binds_hashes_and_negative_next_action(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train_subjects.json"
    development_path = tmp_path / "development_subjects.json"
    combined_audit_path = tmp_path / "combined_audit.json"
    final_redacted_path = tmp_path / "final_redacted_audit.json"
    inner_validation_progress_path = tmp_path / "inner_validation_progress.jsonl"
    progress_log_path = tmp_path / "development_progress.jsonl"
    for path in [
        train_path,
        development_path,
        combined_audit_path,
        final_redacted_path,
        inner_validation_progress_path,
        progress_log_path,
    ]:
        path.write_text(path.name)

    payload = v24.development_result_payload(
        eval_result={
            "aggregate": {"target_prediction_rate": 0.0},
            "by_direction": {},
            "failures": ["synthetic failure"],
            "record_count": 0,
            "records": [],
        },
        paths={
            "combined_audit": combined_audit_path,
            "development": development_path,
            "development_progress": progress_log_path,
            "final_redacted": final_redacted_path,
            "inner_validation_progress": inner_validation_progress_path,
            "train": train_path,
        },
        selected_config_hash="config-hash",
        selected_model_hash="model-hash",
        train_statistics_hash="stats-hash",
        inner_validation_selection_hash="selection-hash",
        evaluated_config_subset_hash="subset-hash",
        full_config_grid_hash="full-grid-hash",
        full_config_grid_count=432,
    )

    assert payload["passed"] is False
    assert payload["next_action"] == v24.FAILING_DEVELOPMENT_NEXT_ACTION
    assert payload["claim_scope"] == v24.DEVELOPMENT_SCOPE
    assert payload["phase"] == "development"
    assert payload["editor_method"] == v24.EDITOR_METHOD
    assert payload["train_pool_sha256"] == v24.sha256_file(train_path)
    assert payload["development_pool_sha256"] == v24.sha256_file(development_path)
    assert payload["combined_audit_sha256"] == v24.sha256_file(combined_audit_path)
    assert payload["final_redacted_audit_sha256"] == v24.sha256_file(final_redacted_path)
    assert "development_progress_log_sha256" not in payload
    assert payload["development_progress_log_pre_results_sha256"] == v24.sha256_file(
        progress_log_path
    )
    assert payload["inner_validation_progress_log_sha256"] == v24.sha256_file(
        inner_validation_progress_path
    )
    assert payload["inner_validation_full_config_grid_hash"] == "full-grid-hash"
    assert payload["inner_validation_full_config_grid_count"] == 432


def test_v24_write_development_results_artifact_logs_file_hash_without_self_hash(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "development_results.json"
    progress_log_path = tmp_path / "development_progress.jsonl"
    progress_log_path.write_text("pre-existing progress\n")
    result = {
        "development_results_sha256": "stale-self-hash",
        "passed": False,
        "record_count": 288,
    }

    written = v24.write_development_results_artifact(
        output_path=output_path,
        result=result,
        progress_log_path=progress_log_path,
        started_at_monotonic=0.0,
    )

    on_disk = json.loads(output_path.read_text())
    progress_event = json.loads(progress_log_path.read_text().splitlines()[-1])

    assert "development_results_sha256" not in on_disk
    assert "development_results_sha256" not in written
    assert "development_progress_log_sha256" not in on_disk
    assert on_disk == written
    assert on_disk["development_results_payload_sha256"] == v24.stable_hash_json(
        {
            "passed": False,
            "record_count": 288,
        }
    )
    assert progress_event["event"] == "development_results_written"
    assert progress_event["development_results_file_sha256"] == v24.sha256_file(output_path)
    assert (
        progress_event["development_results_payload_sha256"]
        == on_disk["development_results_payload_sha256"]
    )
    assert progress_event["passed"] is False


def test_v24_final_authorization_fails_closed_on_mismatch() -> None:
    development_result = {
        "claim_scope": v24.DEVELOPMENT_SCOPE,
        "combined_audit_sha256": "combined",
        "development_pool_sha256": "development",
        "development_results_sha256": "result",
        "editor_method": v24.EDITOR_METHOD,
        "evaluated_config_subset_hash": "subset",
        "final_redacted_audit_sha256": "redacted",
        "inner_validation_full_config_grid_count": 432,
        "inner_validation_full_config_grid_hash": "full-grid",
        "inner_validation_selection_hash": "selection",
        "next_action": v24.PASSING_DEVELOPMENT_NEXT_ACTION,
        "passed": True,
        "phase": "development",
        "selected_config_hash": "config",
        "selected_model_hash": "model",
        "train_pool_sha256": "train",
        "train_statistics_hash": "stats",
    }
    authorization = v24.build_final_authorization_payload(
        development_result=development_result,
        development_results_sha256="result",
        formal_preregistration_sha256="formal",
        helper_test_sha256="helper",
        reviewer_authorization_sha256="review",
        reviewer_confidence="5/5",
        script_sha256="script",
    )
    v24.validate_final_authorization_payload(
        authorization,
        development_result=development_result,
        development_results_sha256="result",
        formal_preregistration_sha256="formal",
        helper_test_sha256="helper",
        reviewer_authorization_sha256="review",
        reviewer_confidence="5/5",
        script_sha256="script",
    )
    bad = {**authorization, "selected_config_hash": "other"}
    with pytest.raises(ValueError, match="selected_config_hash"):
        v24.validate_final_authorization_payload(
            bad,
            development_result=development_result,
            development_results_sha256="result",
            formal_preregistration_sha256="formal",
            helper_test_sha256="helper",
            reviewer_authorization_sha256="review",
            reviewer_confidence="5/5",
            script_sha256="script",
        )


def test_v24_seed_preflight_has_no_pool_overlaps() -> None:
    preflight = v24.build_v24_seed_preflight()

    assert preflight["passed"] is True
    assert preflight["failures"] == []
    assert len(preflight["seed_ranges"]) == len(v24.POOL_CONFIGS) * len(v24.PATTERNS)


def test_v24_source_pool_contract_accepts_redacted_surfaces(tmp_path: Path) -> None:
    train_path = tmp_path / "train_subjects.json"
    development_path = tmp_path / "development_subjects.json"
    train_path.write_text("train")
    development_path.write_text("development")
    expected_train_counts = {pattern: 64 for pattern in v24.PATTERNS}
    expected_development_counts = {pattern: 24 for pattern in v24.PATTERNS}
    combined_audit = {
        "claim_scope": v24.SOURCE_AUDIT_SCOPE,
        "passed": True,
        "pool_summaries": {
            "train": {
                "accepted_counts_by_behavior": expected_train_counts,
                "pool_file_sha256": v24.sha256_file(train_path),
                "pool_redacted_payload_sha256": "train-redacted",
            },
            "development": {
                "accepted_counts_by_behavior": expected_development_counts,
                "pool_file_sha256": v24.sha256_file(development_path),
                "pool_redacted_payload_sha256": "development-redacted",
            },
            "final": {
                "accepted_counts_by_behavior": expected_development_counts,
                "pool_file_sha256": "sealed-final",
                "pool_redacted_payload_sha256": "final-redacted",
            },
        },
    }
    final_redacted = {
        "behavior_suite_hashes": {},
        "candidate_pool_summary_hash": "candidate",
        "claim_scope": v24.FINAL_REDACTED_SCOPE,
        "config_hash": "config",
        "pool": "final",
        "pool_file_sha256": "sealed-final",
        "pool_redacted_payload_sha256": "final-redacted",
        "summary": {
            "accepted_counts_by_behavior": expected_development_counts,
            "max_selected_train_vs_heldout_overlap_count": 0,
        },
        "summary_payload_sha256": "summary",
    }

    failures = v24.validate_source_pool_contract(
        train_path=train_path,
        eval_path=development_path,
        train_payload={"claim_scope": v24.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v24.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )

    assert failures == []
    leaked = {
        **final_redacted,
        "behavior_suite_hashes": {"nested": {"signature": ["leak"]}},
    }
    leaked_failures = v24.validate_source_pool_contract(
        train_path=train_path,
        eval_path=development_path,
        train_payload={"claim_scope": v24.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v24.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=leaked,
        phase="development",
    )
    assert any("final_redacted_audit exposes forbidden keys" in item for item in leaked_failures)
    mismatched_redacted = {
        **final_redacted,
        "pool_file_sha256": "different-final",
    }
    mismatch_failures = v24.validate_source_pool_contract(
        train_path=train_path,
        eval_path=development_path,
        train_payload={"claim_scope": v24.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v24.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=mismatched_redacted,
        phase="development",
    )
    assert "final redacted pool_file_sha256 mismatch" in mismatch_failures
    wrong_counts = {
        **final_redacted,
        "summary": {
            **final_redacted["summary"],
            "accepted_counts_by_behavior": {pattern: 0 for pattern in v24.PATTERNS},
        },
    }
    wrong_count_failures = v24.validate_source_pool_contract(
        train_path=train_path,
        eval_path=development_path,
        train_payload={"claim_scope": v24.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v24.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=wrong_counts,
        phase="development",
    )
    assert "final redacted accepted counts mismatch" in wrong_count_failures
    nonzero_overlap = {
        **final_redacted,
        "summary": {
            **final_redacted["summary"],
            "max_selected_train_vs_heldout_overlap_count": 1,
        },
    }
    overlap_failures = v24.validate_source_pool_contract(
        train_path=train_path,
        eval_path=development_path,
        train_payload={"claim_scope": v24.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v24.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=nonzero_overlap,
        phase="development",
    )
    assert "final redacted train/heldout overlap nonzero" in overlap_failures


def test_v24_generate_pools_writes_redacted_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_counts = {pattern: 24 for pattern in v24.PATTERNS}
    expected_train_counts = {pattern: 64 for pattern in v24.PATTERNS}

    monkeypatch.setattr(v24.v23.v16.v15, "build_suite", lambda *_args: {"suite": True})
    monkeypatch.setattr(v24.v23.v16.v15, "build_heldout_sequences", lambda _suite: [])
    monkeypatch.setattr(v24.v23.v16.v15, "build_candidate_pools", lambda _heldout: {})
    monkeypatch.setattr(v24.v23.v16.v15, "summarize_candidate_pools", lambda _pools: {})
    monkeypatch.setattr(v24, "build_probe_examples", lambda: [])

    def fake_generate_pool(*, pool_name, pool_config, **_kwargs):
        counts = expected_train_counts if pool_name == "train" else expected_counts
        return {
            "claim_scope": "unset",
            "config": {},
            "pool": pool_name,
            "records": [],
            "summary": {"accepted_counts_by_behavior": counts},
        }

    def fake_summarize_pool(payload):
        counts = expected_train_counts if payload["pool"] == "train" else expected_counts
        return {"accepted_counts_by_behavior": counts}

    def fake_final_redacted(_payload):
        return {
            "behavior_suite_hashes": {},
            "candidate_pool_summary_hash": "candidate",
            "claim_scope": "unset",
            "config_hash": "config",
            "pool": "final",
            "pool_redacted_payload_sha256": "final-redacted",
            "summary": {
                "accepted_counts_by_behavior": expected_counts,
                "max_selected_train_vs_heldout_overlap_count": 0,
            },
            "summary_payload_sha256": "summary",
        }

    def fake_combined_audit(*, pool_summaries, **_kwargs):
        return {
            "claim_scope": "unset",
            "passed": True,
            "pool_summaries": pool_summaries,
        }

    monkeypatch.setattr(v24.v23.v16.v15.poolgen, "generate_pool", fake_generate_pool)
    monkeypatch.setattr(
        v24.v23.v16.v15.poolgen,
        "redact_weights_and_signatures",
        lambda payload: payload,
    )
    monkeypatch.setattr(v24.v23.v16.v15.poolgen, "summarize_pool", fake_summarize_pool)
    monkeypatch.setattr(
        v24.v23.v16.v15.poolgen,
        "build_final_redacted_summary",
        fake_final_redacted,
    )
    monkeypatch.setattr(
        v24.v23.v16.v15.poolgen,
        "build_combined_audit",
        fake_combined_audit,
    )
    monkeypatch.setattr(v24.v23.v16.v15.v10, "redact_combined_audit", lambda audit: audit)

    result = v24.generate_pools(
        SimpleNamespace(
            generic_negative_cap=1,
            hard_negative_cap=1,
            heldout_per_class=1,
            lr=0.1,
            positive_cap=1,
            source_margin_gate=0.1,
            support_per_class=1,
            train_epochs=1,
        ),
        tmp_path,
    )

    assert result["passed"] is True
    assert (tmp_path / "train_subjects.json").exists()
    assert (tmp_path / "development_subjects.json").exists()
    assert (tmp_path / "final_subjects.json").exists()
    assert (tmp_path / "final_redacted_audit.json").exists()
    assert (tmp_path / "combined_audit.json").exists()
    progress_text = (tmp_path / v24.SOURCE_POOL_PROGRESS_LOG_FILENAME).read_text()
    assert '"event": "seed_preflight_completed"' in progress_text
    assert progress_text.count('"event": "pool_generation_completed"') == 3
    assert '"event": "combined_audit_written"' in progress_text
    assert "weights" not in progress_text
    assert "signature" not in progress_text
    final_redacted = v24.v23.v16.v15.v1.load_json(tmp_path / "final_redacted_audit.json")
    assert v24.forbidden_final_redacted_keys(final_redacted) == []


def test_v24_run_development_writes_hash_bound_result_without_final_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_dir = tmp_path / "pools"
    output_dir = tmp_path / "out"
    pool_dir.mkdir()
    output_dir.mkdir()
    expected_train_counts = {pattern: 64 for pattern in v24.PATTERNS}
    expected_eval_counts = {pattern: 24 for pattern in v24.PATTERNS}
    train_records = [
        {
            "accepted": True,
            "behavior": behavior,
            "signature": [0.0] * v24.SIGNATURE_DIM,
            "subject_id": f"train-{behavior}-{index}",
            "weights": [0.0] * v24.SOURCE_WEIGHT_DIM,
        }
        for behavior in v24.PATTERNS
        for index in range(64)
    ]
    development_records = [
        {
            "accepted": True,
            "behavior": behavior,
            "signature": [0.0] * v24.SIGNATURE_DIM,
            "subject_id": f"development-{behavior}-{index}",
            "weights": [0.0] * v24.SOURCE_WEIGHT_DIM,
        }
        for behavior in v24.PATTERNS
        for index in range(24)
    ]
    train_payload = {"claim_scope": v24.SOURCE_POOL_SCOPE, "records": train_records}
    development_payload = {
        "claim_scope": v24.SOURCE_POOL_SCOPE,
        "records": development_records,
    }
    (pool_dir / "train_subjects.json").write_text(
        v24.json.dumps(train_payload, indent=2, sort_keys=True)
    )
    (pool_dir / "development_subjects.json").write_text(
        v24.json.dumps(development_payload, indent=2, sort_keys=True)
    )
    combined_audit = {
        "claim_scope": v24.SOURCE_AUDIT_SCOPE,
        "passed": True,
        "pool_summaries": {
            "train": {
                "accepted_counts_by_behavior": expected_train_counts,
                "pool_file_sha256": v24.sha256_file(pool_dir / "train_subjects.json"),
                "pool_redacted_payload_sha256": "train-redacted",
            },
            "development": {
                "accepted_counts_by_behavior": expected_eval_counts,
                "pool_file_sha256": v24.sha256_file(pool_dir / "development_subjects.json"),
                "pool_redacted_payload_sha256": "development-redacted",
            },
            "final": {
                "accepted_counts_by_behavior": expected_eval_counts,
                "pool_file_sha256": "sealed-final",
                "pool_redacted_payload_sha256": "final-redacted",
            },
        },
    }
    final_redacted = {
        "behavior_suite_hashes": {},
        "candidate_pool_summary_hash": "candidate",
        "claim_scope": v24.FINAL_REDACTED_SCOPE,
        "config_hash": "config",
        "pool": "final",
        "pool_file_sha256": "sealed-final",
        "pool_redacted_payload_sha256": "final-redacted",
        "summary": {
            "accepted_counts_by_behavior": expected_eval_counts,
            "max_selected_train_vs_heldout_overlap_count": 0,
        },
        "summary_payload_sha256": "summary",
    }
    (pool_dir / "combined_audit.json").write_text(
        v24.json.dumps(combined_audit, indent=2, sort_keys=True)
    )
    (pool_dir / "final_redacted_audit.json").write_text(
        v24.json.dumps(final_redacted, indent=2, sort_keys=True)
    )
    (pool_dir / "final_subjects.json").write_text("sealed")

    selected_config = {
        "config_hash": "selected-config",
        "inner_validation_full_config_grid_count": 432,
        "inner_validation_full_config_grid_hash": "full-grid-hash",
        "inner_validation_selection_hash": "selection-hash",
        "inner_validation_evaluated_config_subset_hash": "subset-hash",
    }
    fake_model = v24.BehavioralDistilledHypereditor(seed=1)

    def fake_fit(train_subjects, *, max_workers, output_dir):
        assert len(train_subjects) == len(train_records)
        assert max_workers == 1
        assert output_dir == output_dir_arg
        return {
            "model": fake_model,
            "model_hash": "selected-model",
            "selected_config": selected_config,
            "train_statistics_hash": "stats-hash",
        }

    def fake_evaluate(*, subjects, train_stats, record_evaluator, parallel, max_workers, **_kwargs):
        assert len(subjects) == len(development_records)
        assert train_stats["model_hash"] == "selected-model"
        assert parallel is True
        assert max_workers == 1
        return {
            "aggregate": {"target_prediction_rate": 0.0},
            "by_direction": {},
            "failures": ["synthetic failure"],
            "record_count": 0,
            "records": [],
        }

    output_dir_arg = output_dir
    monkeypatch.setattr(v24, "fit_v24_train_statistics", fake_fit)
    monkeypatch.setattr(v24, "evaluate_subjects", fake_evaluate)

    result = v24.run_development(
        SimpleNamespace(max_workers=1),
        pool_dir,
        output_dir,
    )

    assert result["passed"] is False
    assert result["selected_config_hash"] == "selected-config"
    assert result["selected_model_hash"] == "selected-model"
    assert result["train_statistics_hash"] == "stats-hash"
    assert result["inner_validation_selection_hash"] == "selection-hash"
    assert result["evaluated_config_subset_hash"] == "subset-hash"
    assert result["inner_validation_full_config_grid_hash"] == "full-grid-hash"
    assert result["inner_validation_full_config_grid_count"] == 432
    assert result["train_pool_sha256"] == v24.sha256_file(pool_dir / "train_subjects.json")
    assert result["development_pool_sha256"] == v24.sha256_file(
        pool_dir / "development_subjects.json"
    )
    assert (output_dir / "development_results.json").exists()
    progress = (output_dir / "development_progress.jsonl").read_text()
    assert '"event": "development_start"' in progress
    assert '"event": "source_pool_contract_validated"' in progress
    assert '"event": "development_results_written"' in progress
    assert "final_subjects" not in progress
    assert "weights" not in progress
    assert "signature" not in progress
