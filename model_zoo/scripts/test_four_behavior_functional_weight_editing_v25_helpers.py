import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

import train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor as v25


def test_v25_identity_scopes_and_fresh_seeds() -> None:
    assert v25.EDITOR_METHOD == "jacobian_rank1_editor_v25"
    assert v25.PLAN_SHA256 == "50624768332c77ef85845d2c7a3919755f77e790edda4eb9a926f655e4d585b9"
    assert v25.POOL_CONFIGS["train"]["base_seed"] == 126400000
    assert v25.POOL_CONFIGS["development"]["base_seed"] == 127400000
    assert v25.POOL_CONFIGS["final"]["base_seed"] == 128400000
    assert "v25" in str(v25.DEFAULT_POOL_DIR)
    assert "v25" in str(v25.DEFAULT_OUTPUT_DIR)
    assert v25.SOURCE_POOL_SCOPE == "four_behavior_functional_weight_editing_v25_source_pool"
    assert (
        v25.DEVELOPMENT_SCOPE
        == "four_behavior_functional_weight_editing_v25_jacobian_rank1_editor_development"
    )


def test_v25_final_raw_guard_rejects_final_subjects_path() -> None:
    with pytest.raises(ValueError, match="final raw"):
        v25.assert_no_forbidden_final_raw_paths([v25.V25_FINAL_RAW])


def test_v25_final_raw_guard_rejects_unexpected_final_subjects_path(tmp_path: Path) -> None:
    unexpected = tmp_path / "final_subjects.json"
    unexpected.write_text("sealed")

    with pytest.raises(ValueError, match="final raw"):
        v25.assert_no_forbidden_final_raw_paths([unexpected])


def test_v25_load_source_pool_subjects_refuses_final_raw_path(tmp_path: Path) -> None:
    final_like = tmp_path / "final_subjects.json"
    final_like.write_text("[]")

    with pytest.raises(ValueError, match="final raw"):
        v25.load_v25_source_pool_subjects(final_like)


def test_v25_load_source_pool_subjects_hashes_and_counts_nonfinal_pool(tmp_path: Path) -> None:
    train_path = tmp_path / "train_subjects.json"
    train_path.write_text(json.dumps([
        {"pattern": "sorted_ascending", "subject_id": "s1", "weights": [0.0]},
        {"pattern": "sorted_descending", "subject_id": "s2", "weights": [1.0]},
    ]))

    loaded = v25.load_v25_source_pool_subjects(train_path)

    assert loaded["path_sha256"] == v25.stable_hash_json(str(train_path.resolve()))
    assert loaded["pool_file_sha256"] == v25.sha256_file(train_path)
    assert loaded["record_count"] == 2
    assert loaded["counts_by_behavior"] == {
        "sorted_ascending": 1,
        "sorted_descending": 1,
    }
    assert loaded["subjects"][0]["subject_id"] == "s1"


def test_v25_load_source_pool_subjects_accepts_wrapped_records_payload(tmp_path: Path) -> None:
    train_path = tmp_path / "train_subjects.json"
    train_path.write_text(json.dumps({
        "claim_scope": "nonfinal",
        "records": [
            {"pattern": "sorted_ascending", "subject_id": "s1", "weights": [0.0]},
            {"pattern": "sorted_ascending", "subject_id": "s2", "weights": [1.0]},
        ],
        "summary": {"record_count": 2},
    }))

    loaded = v25.load_v25_source_pool_subjects(train_path)

    assert loaded["record_count"] == 2
    assert loaded["counts_by_behavior"] == {"sorted_ascending": 2}
    assert loaded["pool_payload_sha256"] == v25.stable_hash_json({
        "claim_scope": "nonfinal",
        "record_count": 2,
        "summary": {"record_count": 2},
    })


def test_v25_redacted_loaded_pool_summary_omits_subject_records(tmp_path: Path) -> None:
    train_path = tmp_path / "train_subjects.json"
    train_path.write_text(json.dumps([
        {"pattern": "sorted_ascending", "subject_id": "s1", "weights": [0.0]},
    ]))
    loaded = v25.load_v25_source_pool_subjects(train_path)

    redacted = v25.redact_v25_loaded_pool_summary(loaded)
    redacted_text = json.dumps(redacted, sort_keys=True)

    assert redacted["record_count"] == 1
    assert redacted["pool_file_sha256"] == loaded["pool_file_sha256"]
    assert "subjects" not in redacted
    assert "weights" not in redacted_text
    assert "subject_id" not in redacted_text


def test_v25_prepare_development_pool_inputs_logs_redacted_summaries(tmp_path: Path) -> None:
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    (pool_dir / "train_subjects.json").write_text(json.dumps([
        {"pattern": "sorted_ascending", "subject_id": "train-1", "weights": [0.0]},
    ]))
    (pool_dir / "development_subjects.json").write_text(json.dumps([
        {"pattern": "sorted_descending", "subject_id": "dev-1", "weights": [1.0]},
    ]))
    progress_log = tmp_path / "development_progress.jsonl"

    prepared = v25.prepare_v25_development_pool_inputs(
        pool_dir=pool_dir,
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 12.5,
    )
    event = json.loads(progress_log.read_text().splitlines()[-1])
    event_text = json.dumps(event, sort_keys=True)

    assert prepared["train"]["record_count"] == 1
    assert prepared["development"]["record_count"] == 1
    assert event["event"] == "development_inputs_loaded"
    assert event["elapsed_seconds"] == pytest.approx(2.5)
    assert event["train_pool"]["record_count"] == 1
    assert event["development_pool"]["record_count"] == 1
    assert "subjects" not in event_text
    assert "weights" not in event_text
    assert "subject_id" not in event_text


def test_v25_fit_development_train_statistics_logs_hashes_only(tmp_path: Path) -> None:
    train_subjects = [
        {
            "pattern": pattern,
            "subject_id": f"train-{pattern}",
            "weights": torch.linspace(
                -0.05 + 0.01 * index,
                0.05 + 0.01 * index,
                v25.SOURCE_WEIGHT_DIM,
            ).tolist(),
        }
        for index, pattern in enumerate(v25.PATTERNS)
    ]
    progress_log = tmp_path / "development_progress.jsonl"
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]

    stats = v25.fit_v25_development_train_statistics(
        train_subjects=train_subjects,
        probe_examples=probe_examples,
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 14.0,
    )
    events = [json.loads(line) for line in progress_log.read_text().splitlines()]
    completed = events[-1]
    completed_text = json.dumps(completed, sort_keys=True)

    assert [event["event"] for event in events] == [
        "train_statistics_start",
        "train_statistics_completed",
    ]
    assert len(stats["train_statistics_hash"]) == 64
    assert completed["train_statistics_hash"] == stats["train_statistics_hash"]
    assert completed["descriptor_norm_hash"] == stats["descriptor_norm_hash"]
    assert completed["train_counts_by_behavior"] == {
        pattern: 1 for pattern in v25.PATTERNS
    }
    assert "weights" not in completed_text
    assert "subject_id" not in completed_text


def test_v25_run_development_setup_returns_hash_only_summary(tmp_path: Path) -> None:
    pool_dir = tmp_path / "pools"
    output_dir = tmp_path / "out"
    pool_dir.mkdir()
    output_dir.mkdir()
    train_subjects = [
        {
            "pattern": pattern,
            "subject_id": f"train-{pattern}",
            "weights": torch.linspace(
                -0.05 + 0.01 * index,
                0.05 + 0.01 * index,
                v25.SOURCE_WEIGHT_DIM,
            ).tolist(),
        }
        for index, pattern in enumerate(v25.PATTERNS)
    ]
    development_subjects = [
        {
            "pattern": pattern,
            "subject_id": f"dev-{pattern}",
            "weights": torch.linspace(
                -0.02 + 0.01 * index,
                0.02 + 0.01 * index,
                v25.SOURCE_WEIGHT_DIM,
            ).tolist(),
        }
        for index, pattern in enumerate(v25.PATTERNS)
    ]
    (pool_dir / "train_subjects.json").write_text(json.dumps(train_subjects))
    (pool_dir / "development_subjects.json").write_text(json.dumps(development_subjects))
    progress_log = output_dir / "development_progress.jsonl"

    result = v25.run_v25_development_setup(
        pool_dir=pool_dir,
        output_dir=output_dir,
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 15.0,
    )
    result_text = json.dumps(result, sort_keys=True)
    events = [json.loads(line)["event"] for line in progress_log.read_text().splitlines()]

    assert result["passed"] is False
    assert result["stage"] == "development_setup_completed"
    assert len(result["train_statistics_hash"]) == 64
    assert result["train_pool"]["record_count"] == 4
    assert result["development_pool"]["record_count"] == 4
    assert result["development_jobs"]["job_count"] == 12
    assert events == [
        "development_inputs_loaded",
        "train_statistics_start",
        "train_statistics_completed",
        "development_jobs_planned",
        "development_setup_completed",
    ]
    assert "subjects" not in result_text
    assert "weights" not in result_text
    assert "subject_id" not in result_text


def test_v25_build_development_jobs_is_deterministic_and_excludes_identity_targets() -> None:
    subjects = [
        {"pattern": "sorted_descending", "subject_id": "b", "weights": [0.0]},
        {"pattern": "sorted_ascending", "subject_id": "a", "weights": [0.0]},
    ]

    jobs = v25.build_v25_development_jobs(subjects)

    assert len(jobs) == 6
    assert [job["record_id"] for job in jobs] == [
        "a::sorted_ascending->has_majority",
        "a::sorted_ascending->mountain_pattern",
        "a::sorted_ascending->sorted_descending",
        "b::sorted_descending->has_majority",
        "b::sorted_descending->mountain_pattern",
        "b::sorted_descending->sorted_ascending",
    ]
    assert all(job["source_behavior"] != job["target_behavior"] for job in jobs)


def test_v25_redacted_development_job_summary_omits_subject_payload() -> None:
    jobs = v25.build_v25_development_jobs([
        {"pattern": "sorted_ascending", "subject_id": "a", "weights": [0.0]},
    ])

    summary = v25.redact_v25_development_job_summary(jobs)
    summary_text = json.dumps(summary, sort_keys=True)

    assert summary["job_count"] == 3
    assert summary["direction_counts"] == {
        "sorted_ascending->has_majority": 1,
        "sorted_ascending->mountain_pattern": 1,
        "sorted_ascending->sorted_descending": 1,
    }
    assert "subject" not in summary_text
    assert "weights" not in summary_text
    assert "subject_id" not in summary_text


def test_v25_development_job_plan_hash_binds_ordered_redacted_job_identities() -> None:
    first = v25.redact_v25_development_job_summary(v25.build_v25_development_jobs([
        {"pattern": "sorted_ascending", "subject_id": "a", "weights": [0.0]},
    ]))
    second = v25.redact_v25_development_job_summary(v25.build_v25_development_jobs([
        {"pattern": "sorted_ascending", "subject_id": "different", "weights": [0.0]},
    ]))

    assert first["job_count"] == second["job_count"]
    assert first["direction_counts"] == second["direction_counts"]
    assert first["job_plan_hash"] != second["job_plan_hash"]


def test_v25_balanced_development_job_selection_covers_directions() -> None:
    subjects = [
        {"pattern": pattern, "subject_id": f"{pattern}-{index}", "weights": [0.0]}
        for index in range(2)
        for pattern in v25.PATTERNS
    ]
    jobs = v25.build_v25_development_jobs(subjects)

    ordered, summary = v25.order_v25_development_jobs_for_bounded_selection(
        jobs,
        max_jobs=12,
        strategy="balanced-directions",
    )
    selected_directions = [str(job["direction"]) for job in ordered[:12]]

    assert len(set(selected_directions)) == 12
    assert all(count == 1 for count in summary["selected_direction_counts"].values())
    assert summary["selected_job_count"] == 12
    assert summary["strategy"] == "balanced-directions"
    assert len(summary["selection_hash"]) == 64
    assert len(summary["selected_jobs_hash"]) == 64
    assert len(ordered) == len(jobs)


def test_v25_final_redaction_allowlists_are_exact() -> None:
    assert v25.FINAL_REDACTED_ALLOWED_TOP_LEVEL_KEYS == {
        "behavior_suite_hashes",
        "candidate_pool_summary_hash",
        "claim_scope",
        "config_hash",
        "pool",
        "pool_file_sha256",
        "pool_redacted_payload_sha256",
        "summary",
        "summary_payload_sha256",
    }
    assert v25.FINAL_REDACTED_ALLOWED_SUMMARY_KEYS == {
        "accepted_counts_by_behavior",
        "max_selected_train_vs_heldout_overlap_count",
    }
    assert v25.FINAL_COMBINED_SUMMARY_ALLOWED_KEYS == {
        "accepted_counts_by_behavior",
        "pool_file_sha256",
        "pool_redacted_payload_sha256",
    }
    for key in [
        "records",
        "weights",
        "signature",
        "subject_id",
        "seed",
        "train_info",
        "support_margin",
        "heldout_margin",
        "logits",
        "descriptor",
        "jacobian",
        "delta",
    ]:
        assert key in v25.RECURSIVE_FORBIDDEN_FINAL_DETAIL_KEY_TERMS


def test_v25_inner_validation_ranking_tuple_orders_valid_strong_candidates() -> None:
    weak = {
        "config_hash": "b",
        "invalid": False,
        "mean_matched_minus_best_control_target_margin": 0.1,
        "mean_matched_minus_shuffled_signature_target_margin": 0.1,
        "mean_target_margin": 0.3,
        "pareto_undominated_rate": 0.6,
        "proof_gate_failure_count": 2,
        "target_prediction_rate": 0.4,
    }
    strong = {**weak, "config_hash": "a", "target_prediction_rate": 0.5}
    invalid = {**strong, "config_hash": "c", "invalid": True}

    ordered = sorted([weak, invalid, strong], key=v25.inner_validation_ranking_tuple)

    assert [item["config_hash"] for item in ordered] == ["a", "b", "c"]


def _synthetic_v25_proof_record(
    *,
    direction: str = "sorted_ascending->sorted_descending",
    target_margin: float = 0.5,
    target_prediction_pass: bool = True,
) -> dict[str, Any]:
    target_margin_pass = target_margin >= v25.PER_RECORD_MIN_TARGET_MARGIN
    summary = {
        "individual_all_gates_passed": True,
        "matched_minus_best_control_target_margin": 0.1,
        "pareto_undominated": True,
        "proof_gate_diagnostics": {
            "compatible_mse_pass": True,
            "control_margin_fail_count": 0,
            "control_margin_pass_count": len(v25.PROOF_CRITICAL_CONTROL_TYPES),
            "failed_control_types_hash": v25.stable_hash_json([]),
            "individual_all_gates_passed": True,
            "mean_control_margin_advantage": 0.1,
            "min_control_margin_advantage": 0.1,
            "pareto_undominated": True,
            "shuffled_signature_margin_pass": True,
            "target_margin_pass": target_margin_pass,
            "target_prediction_pass": target_prediction_pass,
        },
        "target_prediction_pass": target_prediction_pass,
    }
    for control_type in v25.PROOF_CRITICAL_CONTROL_TYPES:
        summary[f"matched_minus_{control_type}_target_margin"] = 0.1
    return {
        "direction": direction,
        "matched": {"target_margin": target_margin},
        "summary": summary,
    }


def test_v25_inner_validation_candidate_summary_is_hash_only_and_rankable() -> None:
    config = {
        "compat_weight": 0.1,
        "config_index": 7,
        "projection": "spectral_rank4",
        "ridge_lambda": 1e-4,
    }
    proof_records = [
        _synthetic_v25_proof_record(direction="sorted_ascending->sorted_descending"),
        _synthetic_v25_proof_record(direction="sorted_descending->sorted_ascending"),
    ]

    candidate = v25.summarize_v25_inner_validation_candidate(
        config=config,
        config_hash="a" * 64,
        proof_records=proof_records,
        expected_record_count=2,
    )
    candidate_text = json.dumps(candidate, sort_keys=True)

    assert candidate["config"] == config
    assert candidate["config_hash"] == "a" * 64
    assert candidate["config_index"] == 7
    assert candidate["invalid"] is False
    assert candidate["proof_gate_failure_count"] == 0
    assert candidate["record_count"] == 2
    assert candidate["target_prediction_rate"] == 1.0
    assert candidate["pareto_undominated_rate"] == 1.0
    assert candidate["mean_target_margin"] == 0.5
    assert candidate["mean_matched_minus_best_control_target_margin"] == 0.1
    assert candidate["mean_matched_minus_shuffled_signature_target_margin"] == 0.1
    assert len(candidate["proof_record_hashes_hash"]) == 64
    assert "records" not in candidate_text
    assert "weights" not in candidate_text
    assert "subject_id" not in candidate_text


def test_v25_inner_validation_candidate_summary_marks_missing_records_invalid() -> None:
    candidate = v25.summarize_v25_inner_validation_candidate(
        config={
            "compat_weight": 0.1,
            "config_index": 8,
            "projection": "rank1",
            "ridge_lambda": 1e-4,
        },
        config_hash="b" * 64,
        proof_records=[_synthetic_v25_proof_record()],
        expected_record_count=2,
    )

    assert candidate["invalid"] is True
    assert candidate["contract_failure_count"] >= 1


def test_v25_inner_validation_candidate_summary_keeps_weak_valid_config_rankable() -> None:
    candidate = v25.summarize_v25_inner_validation_candidate(
        config={
            "compat_weight": 0.1,
            "config_index": 9,
            "projection": "rank1",
            "ridge_lambda": 1e-4,
        },
        config_hash="c" * 64,
        proof_records=[
            _synthetic_v25_proof_record(
                target_margin=-0.1,
                target_prediction_pass=False,
            )
        ],
        expected_record_count=1,
    )

    assert candidate["invalid"] is False
    assert candidate["contract_failure_count"] == 0
    assert candidate["proof_gate_failure_count"] > 0
    assert candidate["target_prediction_rate"] == 0.0


def test_v25_full_grid_count_and_hash_are_stable() -> None:
    grid = v25.build_v25_config_grid()

    assert len(grid) == (
        len(v25.RIDGE_GRID)
        * len(v25.COMPAT_WEIGHT_GRID)
        * len(v25.PROJECTION_GRID)
        * len(v25.MATCHED_EDIT_SOURCE_GRID)
    )
    assert v25.stable_hash_json(grid) == v25.V25_FULL_GRID_SHA256
    assert [
        (
            config["ridge_lambda"],
            config["compat_weight"],
            config["projection"],
            config["matched_edit_source"],
        )
        for config in grid[:8]
    ] == [
        (1e-5, 0.0, "none", "jacobian"),
        (1e-5, 0.0, "none", "empirical_centroid_task_vector"),
        (1e-5, 0.0, "rank1", "jacobian"),
        (1e-5, 0.0, "rank1", "empirical_centroid_task_vector"),
        (1e-5, 0.0, "spectral_rank4", "jacobian"),
        (1e-5, 0.0, "spectral_rank4", "empirical_centroid_task_vector"),
        (1e-5, 0.0, "rank1_spectral_rank4", "jacobian"),
        (1e-5, 0.0, "rank1_spectral_rank4", "empirical_centroid_task_vector"),
    ]
    assert "teacher_oracle_delta" not in v25.MATCHED_EDIT_SOURCE_GRID
    assert "teacher_oracle_delta" in v25.DIAGNOSTIC_CONTROL_TYPES
    assert "teacher_oracle_delta" not in v25.PROOF_CRITICAL_CONTROL_TYPES


def test_v27_localized_behavior_loss_grid_order_and_hash_are_stable() -> None:
    grid = v25.build_v27_localized_behavior_loss_config_grid()

    assert len(grid) == 64
    assert v25.stable_hash_json(grid) == v25.V27_LOCALIZED_GRID_SHA256
    assert [
        (
            config["config_index"],
            config["localized_basis"],
            config["localized_steps"],
            config["localized_lr"],
            config["localized_source_mse_weight"],
            config["localized_delta_l2_weight"],
            config["matched_edit_source"],
        )
        for config in grid[:8]
    ] == [
        (0, "spectral_train_delta_rank4", 25, 0.05, 0.5, 0.0, "localized_behavior_loss_subspace"),
        (1, "target_source_logit_gradient_rank4", 25, 0.05, 0.5, 0.0, "localized_behavior_loss_subspace"),
        (2, "combined_spectral_gradient_rank8", 25, 0.05, 0.5, 0.0, "localized_behavior_loss_subspace"),
        (3, "output_layer_topk", 25, 0.05, 0.5, 0.0, "localized_behavior_loss_subspace"),
        (4, "spectral_train_delta_rank4", 25, 0.05, 0.5, 0.01, "localized_behavior_loss_subspace"),
        (5, "target_source_logit_gradient_rank4", 25, 0.05, 0.5, 0.01, "localized_behavior_loss_subspace"),
        (6, "combined_spectral_gradient_rank8", 25, 0.05, 0.5, 0.01, "localized_behavior_loss_subspace"),
        (7, "output_layer_topk", 25, 0.05, 0.5, 0.01, "localized_behavior_loss_subspace"),
    ]
    assert all(config["localized_norm_cap"] == 0.25 for config in grid)


def test_v27_localized_source_does_not_change_v25_proof_control_contract() -> None:
    assert "localized_behavior_loss_subspace" in v25.V27_MATCHED_EDIT_SOURCE_GRID
    assert "localized_behavior_loss_subspace" not in v25.PROOF_CRITICAL_CONTROL_TYPES
    assert "teacher_oracle_delta" in v25.DIAGNOSTIC_CONTROL_TYPES
    assert "teacher_oracle_delta" not in v25.PROOF_CRITICAL_CONTROL_TYPES
    assert v25.EXPECTED_CONTROLS_PER_RECORD == (
        len(v25.PROOF_CRITICAL_CONTROL_TYPES)
        + len(v25.DIAGNOSTIC_CONTROL_TYPES)
        + v25.RANDOM_CONTROLS_PER_RECORD
    )


def test_v27_localized_configs_use_v25_native_control_baseline() -> None:
    config = {
        "localized_basis": "spectral_train_delta_rank4",
        "matched_edit_source": "localized_behavior_loss_subspace",
    }

    expected = {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }

    assert v25.v25_config_requires_spectral_basis(config) is True
    assert v25.v25_spectral_seed_config(config) == expected
    assert v25.v25_native_control_config(config) == expected
    assert v25.v25_config_requires_spectral_basis({
        **config,
        "localized_basis": "output_layer_topk",
    }) is False


def test_v27_inner_validation_config_selection_binds_train_provenance() -> None:
    configs = v25.select_v25_inner_validation_configs(
        grid_name="v27-localized",
        max_configs=2,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert len(configs) == 2
    assert configs[0]["matched_edit_source"] == "localized_behavior_loss_subspace"
    assert configs[0]["localized_basis"] == "spectral_train_delta_rank4"
    assert configs[1]["localized_basis"] == "target_source_logit_gradient_rank4"
    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)


def test_v27_inner_validation_config_selection_rejects_unknown_grid() -> None:
    with pytest.raises(ValueError, match="unknown inner validation config grid"):
        v25.select_v25_inner_validation_configs(
            grid_name="missing",
            max_configs=1,
            train_pool_file_sha256="a" * 64,
            train_pool_summary_hash="b" * 64,
        )


def test_v27_support_optimization_boundary_is_explicit() -> None:
    boundary = v25.v27_localized_optimization_boundary()

    assert boundary["optimization_split"] == "support"
    assert boundary["proof_split"] == "heldout"
    assert boundary["allows_heldout_optimization"] is False
    assert boundary["support_objective_is_proof_metric"] is False


def test_v37_projected_optimizer_boundary_is_explicit() -> None:
    boundary = v25.v37_projected_optimizer_optimization_boundary()

    assert boundary["optimization_split"] == "support"
    assert boundary["proof_split"] == "heldout"
    assert boundary["allows_heldout_optimization"] is False
    assert boundary["support_objective_is_proof_metric"] is False
    assert boundary["support_optimizer"] == "projected_target_tournament_loss"
    assert boundary["support_projection"] == "compatible_logit_jacobian_nullspace"


def test_v27_unknown_localized_basis_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown localized basis"):
        v25.build_v27_localized_behavior_loss_basis(
            config={"localized_basis": "missing"},
            source_weights=torch.zeros(v25.SOURCE_WEIGHT_DIM),
            source_behavior="sorted_ascending",
            target_behavior="sorted_descending",
            spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
            train_pool_file_sha256="a" * 64,
            train_pool_summary_hash="b" * 64,
            script_sha256="script",
        )


def test_v27_localized_dispatch_requires_train_pool_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })

    with pytest.raises(ValueError, match="train_pool_file_sha256"):
        v25.evaluate_v25_development_job(
            job={
                "record_id": "dev-1",
                "source_behavior": "sorted_ascending",
                "subject": {
                    "pattern": "sorted_ascending",
                    "subject_id": "hidden",
                    "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
                },
                "target_behavior": "sorted_descending",
            },
            train_stats={},
            config={
                "localized_basis": "spectral_train_delta_rank4",
                "matched_edit_source": "localized_behavior_loss_subspace",
            },
            norm_cap=0.25,
            spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
            control_context={"context_hash": "a" * 64},
            selected_config_hash="b" * 64,
            script_sha256="script",
        )


def test_v27_output_layer_topk_ignores_conflict_gradient_in_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = int(v25.v23.v16.OUTPUT_WEIGHT_START)
    conflict_coordinate = start + 8
    target_gradient = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    compatible_gradient = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    conflict_gradient = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    target_gradient[start] = 1.0
    conflict_gradient[conflict_coordinate] = 1000.0

    monkeypatch.setattr(v25, "v27_support_gradient_rows", lambda **_kwargs: {
        "gradient_by_group": {
            "compatible": compatible_gradient,
            "conflict": conflict_gradient,
            "target_positive": target_gradient,
        },
        "gradient_rows": torch.stack([target_gradient, conflict_gradient, compatible_gradient]),
        "row_names": ["target_positive", "conflict", "compatible", "source_l2"],
        "support_split_counts": {"compatible": 2, "conflict": 2, "target": 2},
    })

    result = v25.build_v27_localized_behavior_loss_basis(
        config={
            "localized_basis": "output_layer_topk",
            "localized_source_mse_weight": 0.5,
        },
        source_weights=torch.zeros(v25.SOURCE_WEIGHT_DIM),
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        script_sha256="script",
    )
    selected = [
        int(torch.nonzero(result["basis"][:, column], as_tuple=False)[0].item())
        for column in range(int(result["basis"].shape[1]))
    ]

    assert start in selected
    assert conflict_coordinate not in selected


def test_v27_support_gradient_rows_records_source_l2_zero_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_support(**_kwargs: Any) -> dict[str, torch.Tensor]:
        return {
            "compatible_inputs": torch.eye(5, dtype=torch.float32)[:2],
            "compatible_source_logits": torch.zeros(2, dtype=torch.float32),
            "conflict_inputs": torch.eye(5, dtype=torch.float32)[2:4],
            "conflict_target_labels": torch.tensor([0.0, 1.0]),
            "target_inputs": torch.eye(5, dtype=torch.float32)[:2],
            "target_labels": torch.ones(2, dtype=torch.float32),
        }

    def fake_logits(weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.to(dtype=torch.float32) @ weights.reshape(-1)[:5]

    monkeypatch.setattr(v25, "v27_support_tensors_for_source_target", fake_support)
    monkeypatch.setattr(v25, "v27_subject_logits_for_inputs", fake_logits)

    info = v25.v27_support_gradient_rows(
        source_weights=torch.zeros(v25.SOURCE_WEIGHT_DIM),
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
    )

    assert "source_l2" in info["row_names"]
    assert info["row_count_by_group"]["source_l2"] == 1
    assert torch.allclose(info["gradient_by_group"]["source_l2"], torch.zeros(v25.SOURCE_WEIGHT_DIM))


def test_v27_optimizer_progress_redaction_omits_raw_vectors_and_examples() -> None:
    event = v25.redact_v27_optimizer_progress_event({
        "alpha": torch.ones(4),
        "basis": torch.ones(v25.SOURCE_WEIGHT_DIM, 4),
        "basis_hash": "c" * 64,
        "delta": torch.ones(v25.SOURCE_WEIGHT_DIM),
        "delta_norm": 0.25,
        "gradient": torch.ones(4),
        "loss": 1.5,
        "step": 5,
        "support_examples": [{"sequence": [0, 1, 2, 3, 4]}],
    })
    event_text = json.dumps(event, sort_keys=True)

    assert event == {
        "basis_hash": "c" * 64,
        "delta_norm": 0.25,
        "finite": True,
        "loss": 1.5,
        "step": 5,
    }
    for forbidden_key in ["alpha", "basis", "delta", "gradient", "support_examples"]:
        assert forbidden_key not in event
    assert "sequence" not in event_text


def test_v27_localized_optimizer_returns_finite_hash_only_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_support(**_kwargs: Any) -> dict[str, torch.Tensor]:
        return {
            "compatible_inputs": torch.eye(5, dtype=torch.float32)[:2],
            "compatible_source_logits": torch.zeros(2, dtype=torch.float32),
            "conflict_inputs": torch.eye(5, dtype=torch.float32)[2:4],
            "conflict_target_labels": torch.tensor([0.0, 1.0]),
            "target_inputs": torch.eye(5, dtype=torch.float32)[:2],
            "target_labels": torch.ones(2, dtype=torch.float32),
        }

    def fake_logits(weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.to(dtype=torch.float32) @ weights.reshape(-1)[:5]

    monkeypatch.setattr(v25, "v27_support_tensors_for_source_target", fake_support)
    monkeypatch.setattr(v25, "v27_subject_logits_for_inputs", fake_logits)

    source_weights = torch.zeros(v25.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    basis = torch.eye(v25.SOURCE_WEIGHT_DIM, 4, dtype=torch.float32)
    result = v25.solve_v27_localized_behavior_loss_edit(
        basis=basis,
        basis_audit={"basis_hash": "c" * 64, "basis_type": "test_basis"},
        config={
            "localized_delta_l2_weight": 0.0,
            "localized_lr": 0.05,
            "localized_norm_cap": 0.25,
            "localized_source_mse_weight": 0.5,
            "localized_steps": 3,
        },
        source_weights=source_weights,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        progress_log_path=tmp_path / "progress.jsonl",
        started_at_monotonic=10.0,
        now_monotonic=lambda: 12.0,
        record_id_hash="d" * 64,
        selected_config_hash="e" * 64,
    )
    progress_text = (tmp_path / "progress.jsonl").read_text()
    result_text = json.dumps(result["audit"], sort_keys=True)

    assert result["delta"].shape == (v25.SOURCE_WEIGHT_DIM,)
    assert torch.isfinite(result["delta"]).all()
    assert float(torch.linalg.norm(result["delta"]).item()) <= 0.25001
    assert len(result["audit"]["delta_sha256"]) == 64
    assert len(result["audit"]["optimizer_trace_hash"]) == 64
    assert result["audit"]["optimization_split"] == "support"
    assert result["audit"]["proof_split"] == "heldout"
    assert "v27_localized_optimizer_progress" in progress_text
    for forbidden in ["alpha", "basis_values", "delta_values", "gradient", "sequence"]:
        assert forbidden not in result_text
        assert forbidden not in progress_text


def test_v27_localized_matched_edit_dispatches_from_development_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_localized(**kwargs: Any) -> dict[str, Any]:
        called["source"] = kwargs["source_behavior"]
        called["target"] = kwargs["target_behavior"]
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": "localized_behavior_loss_subspace",
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    def fake_controls(**kwargs: Any) -> list[dict[str, Any]]:
        called["control_config"] = dict(kwargs["config"])
        return []

    def fake_proof(**kwargs: Any) -> dict[str, Any]:
        return {
            "controls": [],
            "direction": "sorted_ascending->sorted_descending",
            "matched": dict(kwargs["matched"]),
            "source_behavior": kwargs["source_behavior"],
            "summary": {
                "individual_all_gates_passed": True,
                "pareto_undominated": True,
                "target_prediction_pass": True,
            },
            "target_behavior": kwargs["target_behavior"],
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(v25, "evaluate_v25_localized_behavior_loss_matched_edit", fake_localized)
    monkeypatch.setattr(v25, "build_v25_native_controls", fake_controls)
    monkeypatch.setattr(v25, "build_v25_proof_record", fake_proof)

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            "localized_basis": "spectral_train_delta_rank4",
            "matched_edit_source": "localized_behavior_loss_subspace",
            "train_pool_file_sha256": "c" * 64,
            "train_pool_summary_hash": "e" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "a" * 64},
        selected_config_hash="b" * 64,
        script_sha256="script",
    )

    assert called == {
        "control_config": {
            "compat_weight": 0.1,
            "projection": "rank1",
            "ridge_lambda": 1e-2,
        },
        "source": "sorted_ascending",
        "target": "sorted_descending",
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        "localized_behavior_loss_subspace"
    )


def test_v28_anchor_nullspace_grid_order_and_hash_are_stable() -> None:
    grid = v25.build_v28_anchor_nullspace_config_grid()

    assert len(grid) == 8
    assert v25.stable_hash_json(grid) == v25.V28_ANCHOR_NULLSPACE_GRID_SHA256
    assert [
        (
            config["config_index"],
            config["trust_norm_cap"],
            config["anchor_count"],
            config["nullspace_rtol"],
            config["compatible_floor"],
            config["matched_edit_source"],
        )
        for config in grid
    ] == [
        (0, 0.25, 8, 1e-3, 0.05, "anchor_nullspace_trust_region"),
        (1, 0.5, 8, 1e-3, 0.05, "anchor_nullspace_trust_region"),
        (2, 0.25, 8, 1e-2, 0.05, "anchor_nullspace_trust_region"),
        (3, 0.5, 8, 1e-2, 0.05, "anchor_nullspace_trust_region"),
        (4, 0.25, 16, 1e-3, 0.05, "anchor_nullspace_trust_region"),
        (5, 0.5, 16, 1e-3, 0.05, "anchor_nullspace_trust_region"),
        (6, 0.25, 16, 1e-2, 0.05, "anchor_nullspace_trust_region"),
        (7, 0.5, 16, 1e-2, 0.05, "anchor_nullspace_trust_region"),
    ]


def test_v28_inner_validation_config_selection_binds_train_provenance() -> None:
    configs = v25.select_v25_inner_validation_configs(
        grid_name="v28-anchor-nullspace",
        max_configs=2,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert len(configs) == 2
    assert configs[0]["matched_edit_source"] == "anchor_nullspace_trust_region"
    assert configs[0]["trust_norm_cap"] == 0.25
    assert configs[1]["trust_norm_cap"] == 0.5
    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid("v28-anchor-nullspace") == (
        v25.V28_EXPERIMENT_VARIANT
    )


def test_v28_anchor_nullspace_configs_use_v25_native_control_baseline() -> None:
    config = {
        "anchor_count": 8,
        "matched_edit_source": "anchor_nullspace_trust_region",
        "trust_norm_cap": 0.5,
    }

    expected = {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }

    assert v25.v25_config_requires_spectral_basis(config) is False
    assert v25.v25_spectral_seed_config(config) == expected
    assert v25.v25_native_control_config(config) == expected
    assert "anchor_nullspace_trust_region" not in v25.PROOF_CRITICAL_CONTROL_TYPES


def test_v28_anchor_progress_redaction_omits_raw_values() -> None:
    event = v25.redact_v28_anchor_nullspace_basis_progress_event({
        "anchor_count": 8,
        "basis": torch.eye(3),
        "basis_hash": "a" * 64,
        "compatible_energy_ratio": 0.1,
        "compatible_floor": 0.05,
        "coordinates": [1, 2, 3],
        "gradient": torch.ones(3),
        "jacobian": torch.ones(2, 3),
        "nullspace_rtol": 1e-3,
        "preserve_rank": 1,
        "record_id_hash": "b" * 64,
        "selected_config_hash": "c" * 64,
        "selected_coordinate_hash": "d" * 64,
        "support_examples": [{"sequence": [0, 1, 2, 3, 4]}],
    })
    event_text = json.dumps(event, sort_keys=True)

    assert event["anchor_count"] == 8
    assert event["basis_hash"] == "a" * 64
    assert event["finite"] is True
    for forbidden in [
        "basis",
        "coordinates",
        "gradient",
        "jacobian",
        "support_examples",
    ]:
        assert forbidden not in event
    assert "sequence" not in event_text


def test_v28_anchor_selection_downweights_compatible_sensitive_coordinate() -> None:
    source = torch.ones(v25.SOURCE_WEIGHT_DIM)
    g_target = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    g_conflict = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    g_compatible = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    g_target[0] = 1.0
    g_target[1] = 10.0
    g_compatible[1] = 1000.0

    selected = v25.select_v28_anchor_coordinates(
        source_weights=source,
        g_target=g_target,
        g_conflict=g_conflict,
        g_compatible=g_compatible,
        anchor_count=1,
        compatible_floor=0.05,
    )

    assert selected == [0]


def test_v28_nullspace_basis_identity_projection_for_zero_preserve_rank(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_gradient_info(**_kwargs: Any) -> dict[str, Any]:
        g_target = torch.zeros(v25.SOURCE_WEIGHT_DIM)
        g_conflict = torch.zeros(v25.SOURCE_WEIGHT_DIM)
        g_compatible = torch.zeros(v25.SOURCE_WEIGHT_DIM)
        g_target[3] = 1.0
        g_target[5] = 0.5
        return {
            "compatible_jacobian": torch.zeros(2, v25.SOURCE_WEIGHT_DIM),
            "g_compatible": g_compatible,
            "g_conflict": g_conflict,
            "g_target": g_target,
            "support_split_counts": {"compatible": 2, "conflict": 2, "target": 2},
        }

    monkeypatch.setattr(v25, "v28_anchor_gradients_and_compatible_jacobian", fake_gradient_info)
    progress_log = tmp_path / "progress.jsonl"
    result = v25.build_v28_anchor_nullspace_basis(
        config={
            "anchor_count": 2,
            "compatible_floor": 0.05,
            "matched_edit_source": "anchor_nullspace_trust_region",
            "nullspace_rtol": 1e-3,
        },
        source_weights=torch.ones(v25.SOURCE_WEIGHT_DIM),
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        script_sha256="script",
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 12.0,
        record_id_hash="c" * 64,
        selected_config_hash="d" * 64,
    )
    events = [json.loads(line) for line in progress_log.read_text().splitlines()]
    log_text = progress_log.read_text()

    assert result["basis"].shape == (v25.SOURCE_WEIGHT_DIM, 2)
    assert result["audit"]["preserve_rank"] == 0
    assert result["audit"]["compatible_energy_ratio"] == 0.0
    assert [event["event"] for event in events] == [
        "anchor_nullspace_basis_start",
        "anchor_nullspace_basis_completed",
    ]
    assert events[1]["selected_coordinate_hash"] == result["audit"]["selected_coordinate_hash"]
    for forbidden in [
        "coordinates",
        "gradient",
        "jacobian_values",
        "weights",
        "sequence",
    ]:
        assert forbidden not in log_text


def test_v28_anchor_nullspace_dispatch_uses_v28_matched_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_anchor(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": "anchor_nullspace_trust_region",
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    def fake_controls(**kwargs: Any) -> list[dict[str, Any]]:
        called["control_config"] = dict(kwargs["config"])
        return []

    def fake_proof(**kwargs: Any) -> dict[str, Any]:
        return {
            "controls": [],
            "direction": "sorted_ascending->sorted_descending",
            "matched": dict(kwargs["matched"]),
            "source_behavior": kwargs["source_behavior"],
            "summary": {
                "individual_all_gates_passed": True,
                "pareto_undominated": True,
                "target_prediction_pass": True,
            },
            "target_behavior": kwargs["target_behavior"],
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(
        v25,
        "evaluate_v25_anchor_nullspace_trust_region_matched_edit",
        fake_anchor,
    )
    monkeypatch.setattr(v25, "build_v25_native_controls", fake_controls)
    monkeypatch.setattr(v25, "build_v25_proof_record", fake_proof)

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            "anchor_count": 8,
            "compatible_floor": 0.05,
            "matched_edit_source": "anchor_nullspace_trust_region",
            "nullspace_rtol": 1e-3,
            "train_pool_file_sha256": "c" * 64,
            "train_pool_summary_hash": "e" * 64,
            "trust_norm_cap": 0.5,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "a" * 64},
        selected_config_hash="b" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == "anchor_nullspace_trust_region"
    assert called["matched_config"]["trust_norm_cap"] == 0.5
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        "anchor_nullspace_trust_region"
    )


def test_v29_breadth_first_sparse_grid_order_and_hash_are_stable() -> None:
    grid = v25.build_v29_breadth_first_sparse_config_grid()

    assert len(grid) == 8
    assert v25.stable_hash_json(grid) == v25.V29_BREADTH_FIRST_SPARSE_GRID_SHA256
    assert [
        (
            config["config_index"],
            config["sparse_top_k"],
            config["trust_norm_cap"],
            config["compatible_floor"],
            config["extra_compatible_weight"],
            config["matched_edit_source"],
        )
        for config in grid
    ] == [
        (0, 16, 0.5, 0.05, 0.05, "breadth_first_sparse_support"),
        (1, 16, 0.5, 0.05, 0.2, "breadth_first_sparse_support"),
        (2, 16, 1.0, 0.05, 0.05, "breadth_first_sparse_support"),
        (3, 16, 1.0, 0.05, 0.2, "breadth_first_sparse_support"),
        (4, 32, 0.5, 0.05, 0.05, "breadth_first_sparse_support"),
        (5, 32, 0.5, 0.05, 0.2, "breadth_first_sparse_support"),
        (6, 32, 1.0, 0.05, 0.05, "breadth_first_sparse_support"),
        (7, 32, 1.0, 0.05, 0.2, "breadth_first_sparse_support"),
    ]


def test_v29_inner_validation_config_selection_binds_train_provenance() -> None:
    configs = v25.select_v25_inner_validation_configs(
        grid_name="v29-breadth-first-sparse",
        max_configs=3,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert len(configs) == 3
    assert configs[0]["matched_edit_source"] == "breadth_first_sparse_support"
    assert configs[0]["sparse_top_k"] == 16
    assert configs[2]["trust_norm_cap"] == 1.0
    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid("v29-breadth-first-sparse") == (
        v25.V29_EXPERIMENT_VARIANT
    )


def test_v29_sparse_configs_use_v25_native_control_baseline() -> None:
    config = {
        "matched_edit_source": "breadth_first_sparse_support",
        "sparse_top_k": 16,
        "trust_norm_cap": 1.0,
    }

    expected = {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }

    assert v25.v25_config_requires_spectral_basis(config) is False
    assert v25.v25_spectral_seed_config(config) == expected
    assert v25.v25_native_control_config(config) == expected
    assert "breadth_first_sparse_support" not in v25.PROOF_CRITICAL_CONTROL_TYPES


def test_v29_sparse_progress_redaction_omits_raw_values() -> None:
    event = v25.redact_v29_sparse_support_progress_event({
        "coordinates": [1, 2, 3],
        "coordinate_hash": "a" * 64,
        "delta": torch.ones(3),
        "delta_norm": 0.25,
        "epoch": 2,
        "final_subjects_path": str(v25.V25_FINAL_RAW),
        "gradient": torch.ones(3),
        "loss": 0.75,
        "logits": torch.ones(2),
        "record_id_hash": "b" * 64,
        "selected_config_hash": "c" * 64,
        "extra_compatible_weight": 0.2,
        "sparse_top_k": 16,
        "step": 4,
        "subject_id": "raw-subject",
        "support_examples": [{"sequence": [0, 1, 2, 3, 4]}],
        "weights": torch.ones(3),
    })
    event_text = json.dumps(event, sort_keys=True)

    assert event["coordinate_hash"] == "a" * 64
    assert event["delta_norm"] == 0.25
    assert event["finite"] is True
    for forbidden in [
        "coordinates",
        "delta",
        "final_subjects_path",
        "gradient",
        "logits",
        "subject_id",
        "support_examples",
        "weights",
    ]:
        assert forbidden not in event
    assert "sequence" not in event_text
    assert "final_subjects" not in event_text


def test_v29_sparse_selection_downweights_compatible_sensitive_coordinate() -> None:
    g_target = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    g_conflict = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    g_compatible = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    g_target[0] = 1.0
    g_target[1] = 10.0
    g_compatible[1] = 1000.0

    selected = v25.select_v29_sparse_coordinates(
        g_target=g_target,
        g_conflict=g_conflict,
        g_compatible=g_compatible,
        sparse_top_k=1,
        compatible_floor=0.05,
        conflict_weight=0.5,
    )

    assert selected == [0]


def test_v29_optimizer_respects_trust_cap_and_logs_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = {
        "compatible_inputs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "compatible_source_logits": torch.zeros(2, dtype=torch.float32),
        "conflict_inputs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "conflict_target_labels": torch.ones(2, dtype=torch.float32),
        "target_inputs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "target_labels": torch.ones(2, dtype=torch.float32),
    }
    events: list[dict[str, Any]] = []

    def fake_support_tensors(**_: Any) -> dict[str, torch.Tensor]:
        return support

    def fake_logits(flat: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return flat[0].expand(int(inputs.shape[0]))

    def fake_progress(*_: Any, event: str, extra: dict[str, Any], **__: Any) -> None:
        events.append({"event": event, "extra": dict(extra)})

    monkeypatch.setattr(v25, "v27_support_tensors_for_source_target", fake_support_tensors)
    monkeypatch.setattr(v25, "v27_subject_logits_for_inputs", fake_logits)
    monkeypatch.setattr(v25, "record_progress_event", fake_progress)

    edit = v25.solve_v29_breadth_first_sparse_support_edit(
        coordinate_hash="a" * 64,
        selected_coordinates=[0],
        config={
            "compatible_floor": 0.05,
            "extra_compatible_weight": 0.05,
            "sparse_top_k": 1,
            "trust_norm_cap": 0.05,
        },
        source_weights=torch.zeros(v25.SOURCE_WEIGHT_DIM),
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        progress_log_path=Path("progress.jsonl"),
        record_id_hash="b" * 64,
        selected_config_hash="c" * 64,
    )

    assert float(torch.linalg.norm(edit["delta"]).item()) <= 0.050001
    assert edit["audit"]["optimization_boundary"]["optimization_split"] == "support"
    assert edit["audit"]["coordinate_hash"] == "a" * 64
    assert any(item["event"] == "v29_breadth_first_optimizer_progress" for item in events)
    assert any(item["event"] == "v29_breadth_first_optimizer_completed" for item in events)
    assert all("selected_coordinates" not in item["extra"] for item in events)


def test_v29_breadth_first_dispatch_uses_v29_matched_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_cache(subject: dict[str, Any], **_: Any) -> dict[str, Any]:
        return {
            "cache_key": "cache",
            "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
        }

    def fake_matched(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": "breadth_first_sparse_support",
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    def fake_controls(**kwargs: Any) -> list[dict[str, Any]]:
        called["control_config"] = dict(kwargs["config"])
        return []

    def fake_proof(**kwargs: Any) -> dict[str, Any]:
        return {
            "controls": [],
            "direction": "sorted_ascending->sorted_descending",
            "matched": dict(kwargs["matched"]),
            "source_behavior": kwargs["source_behavior"],
            "summary": {
                "individual_all_gates_passed": True,
                "pareto_undominated": True,
                "target_prediction_pass": True,
            },
            "target_behavior": kwargs["target_behavior"],
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", fake_cache)
    monkeypatch.setattr(
        v25,
        "evaluate_v25_breadth_first_sparse_support_matched_edit",
        fake_matched,
    )
    monkeypatch.setattr(v25, "build_v25_native_controls", fake_controls)
    monkeypatch.setattr(v25, "build_v25_proof_record", fake_proof)

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            "matched_edit_source": "breadth_first_sparse_support",
            "sparse_top_k": 16,
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "e" * 64},
        selected_config_hash="c" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == "breadth_first_sparse_support"
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        "breadth_first_sparse_support"
    )


def test_v30_margin_gated_sparse_grid_order_and_hash_are_stable() -> None:
    grid = v25.build_v30_margin_gated_sparse_config_grid()

    assert len(grid) == 8
    assert v25.stable_hash_json(grid) == v25.V30_MARGIN_GATED_SPARSE_GRID_SHA256
    assert [
        (
            config["config_index"],
            config["sparse_top_k"],
            config["trust_norm_cap"],
            config["target_margin_floor"],
            config["extra_compatible_weight"],
            config["matched_edit_source"],
        )
        for config in grid
    ] == [
        (0, 32, 1.0, 0.15, 0.05, "margin_gated_sparse_support"),
        (1, 32, 1.0, 0.25, 0.05, "margin_gated_sparse_support"),
        (2, 32, 1.25, 0.15, 0.05, "margin_gated_sparse_support"),
        (3, 32, 1.25, 0.25, 0.05, "margin_gated_sparse_support"),
        (4, 64, 1.0, 0.15, 0.05, "margin_gated_sparse_support"),
        (5, 64, 1.0, 0.25, 0.05, "margin_gated_sparse_support"),
        (6, 64, 1.25, 0.15, 0.05, "margin_gated_sparse_support"),
        (7, 64, 1.25, 0.25, 0.05, "margin_gated_sparse_support"),
    ]


def test_v30_inner_validation_config_selection_binds_train_provenance() -> None:
    configs = v25.select_v25_inner_validation_configs(
        grid_name="v30-margin-gated-sparse",
        max_configs=2,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert len(configs) == 2
    assert configs[0]["matched_edit_source"] == "margin_gated_sparse_support"
    assert configs[0]["target_margin_floor"] == 0.15
    assert configs[1]["target_margin_floor"] == 0.25
    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid("v30-margin-gated-sparse") == (
        v25.V30_EXPERIMENT_VARIANT
    )


def test_v30_margin_gated_configs_use_v25_native_control_baseline() -> None:
    config = {
        "matched_edit_source": "margin_gated_sparse_support",
        "sparse_top_k": 64,
        "trust_norm_cap": 1.25,
    }

    expected = {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }

    assert v25.v25_config_requires_spectral_basis(config) is False
    assert v25.v25_spectral_seed_config(config) == expected
    assert v25.v25_native_control_config(config) == expected
    assert "margin_gated_sparse_support" not in v25.PROOF_CRITICAL_CONTROL_TYPES


def test_v30_margin_progress_redaction_omits_raw_values() -> None:
    event = v25.redact_v30_margin_gated_progress_event({
        "coordinate_hash": "a" * 64,
        "coordinates": [1, 2],
        "delta_norm": 1.0,
        "final_subjects_path": str(v25.V25_FINAL_RAW),
        "gradient": torch.ones(2),
        "logits": torch.ones(2),
        "record_id_hash": "b" * 64,
        "selected_config_hash": "c" * 64,
        "step": 10,
        "subject_id": "raw-subject",
        "support_examples": [{"sequence": [0, 1, 2, 3, 4]}],
        "target_margin_floor": 0.25,
        "target_margin_hinge": 0.125,
        "weights": torch.ones(2),
    })
    event_text = json.dumps(event, sort_keys=True)

    assert event["coordinate_hash"] == "a" * 64
    assert event["target_margin_hinge"] == 0.125
    assert event["finite"] is True
    for forbidden in [
        "coordinates",
        "final_subjects_path",
        "gradient",
        "logits",
        "subject_id",
        "support_examples",
        "weights",
    ]:
        assert forbidden not in event
    assert "final_subjects" not in event_text
    assert "sequence" not in event_text


def test_v30_target_margin_hinge_loss_rewards_margin_floor() -> None:
    satisfied = v25.v30_target_margin_hinge_loss(
        logits=torch.tensor([0.5, -0.4]),
        labels=torch.tensor([1.0, 0.0]),
        margin_floor=0.25,
    )
    below = v25.v30_target_margin_hinge_loss(
        logits=torch.tensor([0.1, -0.05]),
        labels=torch.tensor([1.0, 0.0]),
        margin_floor=0.25,
    )

    assert float(satisfied.item()) == 0.0
    assert float(below.item()) > 0.0


def test_v30_optimizer_respects_trust_cap_and_logs_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = {
        "compatible_inputs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "compatible_source_logits": torch.zeros(2, dtype=torch.float32),
        "conflict_inputs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "conflict_target_labels": torch.ones(2, dtype=torch.float32),
        "target_inputs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "target_labels": torch.ones(2, dtype=torch.float32),
    }
    events: list[dict[str, Any]] = []

    monkeypatch.setattr(
        v25,
        "v27_support_tensors_for_source_target",
        lambda **_: support,
    )
    monkeypatch.setattr(
        v25,
        "v27_subject_logits_for_inputs",
        lambda flat, inputs: flat[0].expand(int(inputs.shape[0])),
    )
    monkeypatch.setattr(
        v25,
        "record_progress_event",
        lambda *_args, event, extra, **_kwargs: events.append({
            "event": event,
            "extra": dict(extra),
        }),
    )

    edit = v25.solve_v30_margin_gated_sparse_support_edit(
        coordinate_hash="a" * 64,
        selected_coordinates=[0],
        config={
            "compatible_floor": 0.05,
            "extra_compatible_weight": 0.05,
            "sparse_top_k": 1,
            "target_margin_floor": 0.25,
            "trust_norm_cap": 0.05,
        },
        source_weights=torch.zeros(v25.SOURCE_WEIGHT_DIM),
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        progress_log_path=Path("progress.jsonl"),
        record_id_hash="b" * 64,
        selected_config_hash="c" * 64,
    )

    assert float(torch.linalg.norm(edit["delta"]).item()) <= 0.050001
    assert edit["audit"]["target_margin_floor"] == 0.25
    assert any(
        item["event"] == "v30_margin_gated_optimizer_progress"
        and "target_margin_hinge" in item["extra"]
        for item in events
    )
    assert any(item["event"] == "v30_margin_gated_optimizer_completed" for item in events)


def test_v30_margin_gated_dispatch_uses_v30_matched_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_matched(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": "margin_gated_sparse_support",
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    def fake_controls(**kwargs: Any) -> list[dict[str, Any]]:
        called["control_config"] = dict(kwargs["config"])
        return []

    def fake_proof(**kwargs: Any) -> dict[str, Any]:
        return {
            "controls": [],
            "direction": "sorted_ascending->sorted_descending",
            "matched": dict(kwargs["matched"]),
            "source_behavior": kwargs["source_behavior"],
            "summary": {
                "individual_all_gates_passed": True,
                "pareto_undominated": True,
                "target_prediction_pass": True,
            },
            "target_behavior": kwargs["target_behavior"],
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(
        v25,
        "evaluate_v25_margin_gated_sparse_support_matched_edit",
        fake_matched,
    )
    monkeypatch.setattr(v25, "build_v25_native_controls", fake_controls)
    monkeypatch.setattr(v25, "build_v25_proof_record", fake_proof)

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            "matched_edit_source": "margin_gated_sparse_support",
            "sparse_top_k": 64,
            "target_margin_floor": 0.25,
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "e" * 64},
        selected_config_hash="c" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == "margin_gated_sparse_support"
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        "margin_gated_sparse_support"
    )


def test_v31_orthogonal_sign_sparse_grid_order_and_hash_are_stable() -> None:
    grid = v25.build_v31_orthogonal_sign_sparse_config_grid()

    assert len(grid) == 8
    assert v25.stable_hash_json(grid) == v25.V31_ORTHOGONAL_SIGN_SPARSE_GRID_SHA256
    assert [
        (
            config["config_index"],
            config["sparse_top_k"],
            config["trust_norm_cap"],
            config["sign_conflict_penalty"],
            config["compatible_orthogonal_weight"],
            config["target_margin_floor"],
            config["matched_edit_source"],
        )
        for config in grid
    ] == [
        (0, 64, 1.25, 0.5, 0.05, 0.25, "orthogonal_sign_sparse_support"),
        (1, 64, 1.25, 0.5, 0.15, 0.25, "orthogonal_sign_sparse_support"),
        (2, 64, 1.25, 1.0, 0.05, 0.25, "orthogonal_sign_sparse_support"),
        (3, 64, 1.25, 1.0, 0.15, 0.25, "orthogonal_sign_sparse_support"),
        (4, 64, 1.5, 0.5, 0.05, 0.25, "orthogonal_sign_sparse_support"),
        (5, 64, 1.5, 0.5, 0.15, 0.25, "orthogonal_sign_sparse_support"),
        (6, 64, 1.5, 1.0, 0.05, 0.25, "orthogonal_sign_sparse_support"),
        (7, 64, 1.5, 1.0, 0.15, 0.25, "orthogonal_sign_sparse_support"),
    ]


def test_v31_inner_validation_config_selection_binds_train_provenance() -> None:
    configs = v25.select_v25_inner_validation_configs(
        grid_name="v31-orthogonal-sign-sparse",
        max_configs=2,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert len(configs) == 2
    assert configs[0]["matched_edit_source"] == "orthogonal_sign_sparse_support"
    assert configs[0]["compatible_orthogonal_weight"] == 0.05
    assert configs[1]["compatible_orthogonal_weight"] == 0.15
    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid("v31-orthogonal-sign-sparse") == (
        v25.V31_EXPERIMENT_VARIANT
    )


def test_v31_orthogonal_sign_configs_use_v25_native_control_baseline() -> None:
    config = {
        "matched_edit_source": "orthogonal_sign_sparse_support",
        "sparse_top_k": 64,
        "trust_norm_cap": 1.25,
    }

    expected = {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }

    assert v25.v25_config_requires_spectral_basis(config) is False
    assert v25.v25_spectral_seed_config(config) == expected
    assert v25.v25_native_control_config(config) == expected
    assert "orthogonal_sign_sparse_support" not in v25.PROOF_CRITICAL_CONTROL_TYPES


def test_v31_progress_redaction_omits_raw_values() -> None:
    event = v25.redact_v31_orthogonal_sign_progress_event({
        "compatible_orthogonal_loss": 0.25,
        "compatible_orthogonal_weight": 0.15,
        "coordinate_hash": "a" * 64,
        "coordinates": [1, 2],
        "delta_norm": 1.0,
        "final_subjects_path": str(v25.V25_FINAL_RAW),
        "gradient": torch.ones(2),
        "logits": torch.ones(2),
        "record_id_hash": "b" * 64,
        "selected_config_hash": "c" * 64,
        "sign_conflict_penalty": 1.0,
        "step": 10,
        "subject_id": "raw-subject",
        "support_examples": [{"sequence": [0, 1, 2, 3, 4]}],
        "target_multiplier": 1.5,
        "weights": torch.ones(2),
    })
    event_text = json.dumps(event, sort_keys=True)

    assert event["coordinate_hash"] == "a" * 64
    assert event["compatible_orthogonal_loss"] == 0.25
    assert event["target_multiplier"] == 1.5
    assert event["finite"] is True
    for forbidden in [
        "coordinates",
        "final_subjects_path",
        "gradient",
        "logits",
        "subject_id",
        "support_examples",
        "weights",
    ]:
        assert forbidden not in event
    assert "final_subjects" not in event_text
    assert "sequence" not in event_text


def test_v31_sign_coherent_sparse_selection_penalizes_conflicting_signs() -> None:
    g_target = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    g_conflict = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    g_compatible = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    g_target[:3] = torch.tensor([1.0, 1.0, 0.5])
    g_conflict[:3] = torch.tensor([1.0, -1.0, 0.5])
    selected = v25.select_v31_sign_coherent_sparse_coordinates(
        g_target=g_target,
        g_conflict=g_conflict,
        g_compatible=g_compatible,
        sparse_top_k=2,
        compatible_floor=0.1,
        conflict_weight=1.0,
        sign_conflict_penalty=1.0,
    )

    assert selected == [0, 2]


def test_v31_support_hardness_multiplier_and_orthogonal_loss_are_scalars() -> None:
    multiplier = v25.v31_support_hardness_multiplier(
        source_target_logits=torch.tensor([0.0, -0.5]),
        target_labels=torch.tensor([1.0, 0.0]),
        target_margin_floor=0.25,
        hard_target_margin_weight=1.0,
    )
    loss = v25.v31_compatible_gradient_orthogonal_loss(
        delta=torch.tensor([1.0, 0.0]),
        g_compatible=torch.tensor([1.0, 1.0]),
    )

    assert torch.isclose(multiplier, torch.tensor(1.125))
    assert torch.isclose(loss, torch.tensor(0.5), atol=1e-6)


def test_v31_optimizer_respects_trust_cap_and_logs_orthogonal_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = {
        "compatible_inputs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "compatible_source_logits": torch.zeros(2, dtype=torch.float32),
        "conflict_inputs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "conflict_target_labels": torch.ones(2, dtype=torch.float32),
        "target_inputs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "target_labels": torch.ones(2, dtype=torch.float32),
    }
    events: list[dict[str, Any]] = []

    monkeypatch.setattr(
        v25,
        "v27_support_tensors_for_source_target",
        lambda **_: support,
    )
    monkeypatch.setattr(
        v25,
        "v27_subject_logits_for_inputs",
        lambda flat, inputs: flat[0].expand(int(inputs.shape[0])),
    )
    monkeypatch.setattr(
        v25,
        "record_progress_event",
        lambda *_args, event, extra, **_kwargs: events.append({
            "event": event,
            "extra": dict(extra),
        }),
    )

    edit = v25.solve_v31_orthogonal_sign_sparse_support_edit(
        compatible_gradient=torch.ones(v25.SOURCE_WEIGHT_DIM),
        coordinate_hash="a" * 64,
        selected_coordinates=[0],
        config={
            "compatible_floor": 0.05,
            "compatible_orthogonal_weight": 0.15,
            "extra_compatible_weight": 0.05,
            "hard_target_margin_weight": 1.0,
            "sign_conflict_penalty": 1.0,
            "sparse_top_k": 1,
            "target_margin_floor": 0.25,
            "trust_norm_cap": 0.05,
        },
        source_weights=torch.zeros(v25.SOURCE_WEIGHT_DIM),
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        progress_log_path=Path("progress.jsonl"),
        record_id_hash="b" * 64,
        selected_config_hash="c" * 64,
    )

    assert float(torch.linalg.norm(edit["delta"]).item()) <= 0.050001
    assert edit["audit"]["compatible_orthogonal_weight"] == 0.15
    assert edit["audit"]["target_multiplier"] >= 1.0
    assert any(
        item["event"] == "v31_orthogonal_sign_optimizer_progress"
        and "compatible_orthogonal_loss" in item["extra"]
        and "target_multiplier" in item["extra"]
        for item in events
    )
    assert any(item["event"] == "v31_orthogonal_sign_optimizer_completed" for item in events)


def test_v31_orthogonal_sign_dispatch_uses_v31_matched_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_matched(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": "orthogonal_sign_sparse_support",
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    def fake_controls(**kwargs: Any) -> list[dict[str, Any]]:
        called["control_config"] = dict(kwargs["config"])
        return []

    def fake_proof(**kwargs: Any) -> dict[str, Any]:
        return {
            "controls": [],
            "direction": "sorted_ascending->sorted_descending",
            "matched": dict(kwargs["matched"]),
            "source_behavior": kwargs["source_behavior"],
            "summary": {
                "individual_all_gates_passed": True,
                "pareto_undominated": True,
                "target_prediction_pass": True,
            },
            "target_behavior": kwargs["target_behavior"],
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(
        v25,
        "evaluate_v25_orthogonal_sign_sparse_support_matched_edit",
        fake_matched,
    )
    monkeypatch.setattr(v25, "build_v25_native_controls", fake_controls)
    monkeypatch.setattr(v25, "build_v25_proof_record", fake_proof)

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            "matched_edit_source": "orthogonal_sign_sparse_support",
            "sparse_top_k": 64,
            "target_margin_floor": 0.25,
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "e" * 64},
        selected_config_hash="c" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == "orthogonal_sign_sparse_support"
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        "orthogonal_sign_sparse_support"
    )


def test_v32_support_tournament_grid_order_and_hash_are_stable() -> None:
    grid = v25.build_v32_support_tournament_margin_config_grid()

    assert len(grid) == 8
    assert v25.stable_hash_json(grid) == v25.V32_SUPPORT_TOURNAMENT_GRID_SHA256
    assert [
        (
            config["config_index"],
            config["tournament_margin_weight"],
            config["tournament_margin_floor"],
            config["compatible_orthogonal_weight"],
            config["trust_norm_cap"],
            config["matched_edit_source"],
        )
        for config in grid
    ] == [
        (0, 0.5, 0.05, 0.05, 1.25, "support_tournament_margin_sparse"),
        (1, 0.5, 0.05, 0.15, 1.25, "support_tournament_margin_sparse"),
        (2, 0.5, 0.15, 0.05, 1.25, "support_tournament_margin_sparse"),
        (3, 0.5, 0.15, 0.15, 1.25, "support_tournament_margin_sparse"),
        (4, 1.0, 0.05, 0.05, 1.25, "support_tournament_margin_sparse"),
        (5, 1.0, 0.05, 0.15, 1.25, "support_tournament_margin_sparse"),
        (6, 1.0, 0.15, 0.05, 1.25, "support_tournament_margin_sparse"),
        (7, 1.0, 0.15, 0.15, 1.25, "support_tournament_margin_sparse"),
    ]


def test_v32_inner_validation_config_selection_binds_train_provenance() -> None:
    configs = v25.select_v25_inner_validation_configs(
        grid_name="v32-support-tournament-margin",
        max_configs=2,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert len(configs) == 2
    assert configs[0]["matched_edit_source"] == "support_tournament_margin_sparse"
    assert configs[0]["tournament_margin_weight"] == 0.5
    assert configs[1]["compatible_orthogonal_weight"] == 0.15
    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid("v32-support-tournament-margin") == (
        v25.V32_EXPERIMENT_VARIANT
    )


def test_v33_diagnostic_grid_replays_two_best_v32_configs() -> None:
    grid = v25.build_v33_proof_gate_diagnostic_config_grid()

    assert len(grid) == 2
    assert [item["config_index"] for item in grid] == [0, 1]
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE
    }
    assert {item["trust_norm_cap"] for item in grid} == {1.25}
    assert {item["tournament_margin_weight"] for item in grid} == {1.0}
    assert {item["tournament_margin_floor"] for item in grid} == {0.15}
    assert [item["compatible_orthogonal_weight"] for item in grid] == [0.15, 0.05]

    configs = v25.select_v25_inner_validation_configs(
        grid_name="v33-proof-gate-diagnostic",
        max_configs=None,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid(
        "v33-proof-gate-diagnostic"
    ) == "v33_proof_gate_decomposition_diagnostic"


def test_v34_locality_pressure_grid_is_bounded_and_binds_provenance() -> None:
    grid = v25.build_v34_locality_pressure_config_grid()

    assert len(grid) == 6
    assert v25.stable_hash_json(grid) == v25.V34_LOCALITY_PRESSURE_GRID_SHA256
    assert [item["config_index"] for item in grid] == list(range(6))
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE
    }
    assert {item["compatible_orthogonal_weight"] for item in grid} == {0.15}
    assert sorted({item["trust_norm_cap"] for item in grid}) == [0.5, 0.75, 1.0]
    assert sorted({item["extra_compatible_weight"] for item in grid}) == [0.5, 2.0]

    configs = v25.select_v25_inner_validation_configs(
        grid_name="v34-locality-pressure",
        max_configs=None,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid(
        "v34-locality-pressure"
    ) == v25.V34_EXPERIMENT_VARIANT


def test_v35_alpha_selector_prefers_most_source_preserving_eligible_alpha() -> None:
    selected = v25.select_v35_support_source_alpha_candidate(
        candidates=[
            {
                "alpha": 1.0,
                "support_compatible_mse": 0.20,
                "support_target_margin": 0.40,
                "support_tournament_margin": 0.20,
            },
            {
                "alpha": 0.5,
                "support_compatible_mse": 0.03,
                "support_target_margin": 0.20,
                "support_tournament_margin": 0.06,
            },
            {
                "alpha": 0.125,
                "support_compatible_mse": 0.01,
                "support_target_margin": 0.03,
                "support_tournament_margin": -0.02,
            },
        ],
        alpha_target_margin_floor=0.10,
        alpha_tournament_margin_floor=0.00,
        fallback_target_penalty=10.0,
        fallback_tournament_penalty=5.0,
    )

    assert selected["alpha"] == 0.5
    assert selected["selection_mode"] == "eligible_min_compatible_mse"
    assert selected["eligible_count"] == 2
    assert len(selected["candidate_metrics_hash"]) == 64


def test_v35_alpha_selector_fallback_penalizes_margin_failures() -> None:
    selected = v25.select_v35_support_source_alpha_candidate(
        candidates=[
            {
                "alpha": 0.25,
                "support_compatible_mse": 0.001,
                "support_target_margin": -0.20,
                "support_tournament_margin": -0.20,
            },
            {
                "alpha": 1.0,
                "support_compatible_mse": 0.20,
                "support_target_margin": 0.09,
                "support_tournament_margin": -0.01,
            },
        ],
        alpha_target_margin_floor=0.10,
        alpha_tournament_margin_floor=0.00,
        fallback_target_penalty=10.0,
        fallback_tournament_penalty=5.0,
    )

    assert selected["alpha"] == 1.0
    assert selected["selection_mode"] == "fallback_penalized"
    assert selected["eligible_count"] == 0
    assert selected["fallback_score"] < 0.40


def test_v35_alpha_candidate_metrics_hash_binds_support_metrics() -> None:
    base_candidates = [
        {
            "alpha": 1.0,
            "support_compatible_mse": 0.20,
            "support_target_margin": 0.40,
            "support_tournament_margin": 0.20,
        },
        {
            "alpha": 0.5,
            "support_compatible_mse": 0.03,
            "support_target_margin": 0.20,
            "support_tournament_margin": 0.06,
        },
    ]
    changed_candidates = [
        dict(base_candidates[0]),
        {**base_candidates[1], "support_compatible_mse": 0.04},
    ]

    base = v25.select_v35_support_source_alpha_candidate(
        candidates=base_candidates,
        alpha_target_margin_floor=0.10,
        alpha_tournament_margin_floor=0.00,
        fallback_target_penalty=10.0,
        fallback_tournament_penalty=5.0,
    )
    changed = v25.select_v35_support_source_alpha_candidate(
        candidates=changed_candidates,
        alpha_target_margin_floor=0.10,
        alpha_tournament_margin_floor=0.00,
        fallback_target_penalty=10.0,
        fallback_tournament_penalty=5.0,
    )

    assert base["alpha"] == changed["alpha"] == 0.5
    assert base["candidate_metrics_hash"] != changed["candidate_metrics_hash"]


def test_v35_support_source_line_search_grid_is_bounded_and_binds_provenance() -> None:
    grid = v25.build_v35_support_source_line_search_config_grid()

    assert len(grid) == 4
    assert v25.stable_hash_json(grid) == v25.V35_SUPPORT_SOURCE_LINE_SEARCH_GRID_SHA256
    assert [item["config_index"] for item in grid] == list(range(4))
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V35_SUPPORT_SOURCE_LINE_SEARCH_MATCHED_EDIT_SOURCE
    }
    assert sorted({item["trust_norm_cap"] for item in grid}) == [1.0, 1.25]
    assert sorted({item["alpha_target_margin_floor"] for item in grid}) == [0.05, 0.10]
    assert {tuple(item["alpha_candidates"]) for item in grid} == {
        (1.0, 0.75, 0.5, 0.25, 0.125, 0.0)
    }

    configs = v25.select_v25_inner_validation_configs(
        grid_name="v35-support-source-line-search",
        max_configs=None,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid(
        "v35-support-source-line-search"
    ) == v25.V35_EXPERIMENT_VARIANT


def test_v35_alpha_selection_progress_redaction_omits_raw_values() -> None:
    event = v25.redact_v35_support_source_alpha_progress_event({
        "alpha": 0.5,
        "alpha_candidate_count": 6,
        "alpha_candidates_hash": "a" * 64,
        "coordinate_hash": "b" * 64,
        "delta_norm": 0.25,
        "final_subjects_path": str(v25.V25_FINAL_RAW),
        "gradient": torch.ones(2),
        "logits": torch.ones(2),
        "record_id_hash": "c" * 64,
        "selected_config_hash": "d" * 64,
        "candidate_metrics_hash": "e" * 64,
        "eligible_count": 3,
        "selection_mode": "eligible_min_compatible_mse",
        "subject_id": "raw-subject",
        "support_compatible_mse": 0.04,
        "support_examples": [{"sequence": [0, 1, 2, 3, 4]}],
        "support_runner_margin": 0.15,
        "support_target_margin": 0.25,
        "support_tournament_margin": 0.10,
        "weights": torch.ones(2),
    })
    event_text = json.dumps(event, sort_keys=True)

    assert event["alpha"] == 0.5
    assert event["support_compatible_mse"] == 0.04
    assert event["alpha_candidate_count"] == 6
    assert event["eligible_count"] == 3
    assert event["candidate_metrics_hash"] == "e" * 64
    assert event["finite"] is True
    for forbidden in [
        "final_subjects_path",
        "gradient",
        "logits",
        "subject_id",
        "support_examples",
        "weights",
    ]:
        assert forbidden not in event
    assert "final_subjects" not in event_text
    assert "sequence" not in event_text


def test_v35_matched_edit_uses_support_only_alpha_selection_and_logs_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}
    support = {
        "compatible_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "compatible_source_logits": torch.zeros(1, dtype=torch.float32),
        "conflict_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "conflict_target_labels": torch.ones(1, dtype=torch.float32),
        "target_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "target_labels": torch.ones(1, dtype=torch.float32),
    }

    monkeypatch.setattr(v25, "v29_sparse_support_gradients", lambda **_kwargs: {
        "g_compatible": torch.ones(v25.SOURCE_WEIGHT_DIM),
        "g_conflict": torch.ones(v25.SOURCE_WEIGHT_DIM),
        "g_target": torch.ones(v25.SOURCE_WEIGHT_DIM),
        "support_split_counts": {"compatible": 1, "conflict": 1, "target": 1},
    })
    monkeypatch.setattr(
        v25,
        "select_v31_sign_coherent_sparse_coordinates",
        lambda **_kwargs: [0],
    )
    monkeypatch.setattr(v25, "v27_support_tensors_for_source_target", lambda **_: support)
    monkeypatch.setattr(v25, "v32_support_behavior_margin_tensors", lambda: {
        "by_behavior": {},
        "counts_by_behavior": {},
        "tensor_hash": "e" * 64,
    })
    monkeypatch.setattr(
        v25,
        "solve_v32_support_tournament_sparse_edit",
        lambda **_kwargs: {
            "audit": {
                "base_optimizer_trace_hash": "f" * 64,
                "delta_sha256": "1" * 64,
                "support_scalar_losses": {"loss": 0.1},
            },
            "delta": torch.nn.functional.one_hot(
                torch.tensor(0),
                num_classes=v25.SOURCE_WEIGHT_DIM,
            ).to(dtype=torch.float32),
        },
    )
    monkeypatch.setattr(
        v25,
        "v27_subject_logits_for_inputs",
        lambda flat, inputs: flat[0].expand(int(inputs.shape[0])),
    )

    def fake_margins(*, weights: torch.Tensor, tournament_tensors: Mapping[str, Any]):
        target_margin = weights[0] * 0.4
        return {
            "has_majority": target_margin - 0.10,
            "mountain_pattern": target_margin - 0.20,
            "sorted_ascending": target_margin - 0.30,
            "sorted_descending": target_margin,
        }

    def fake_control_record(**kwargs: Any) -> dict[str, Any]:
        captured["delta"] = kwargs["delta"].detach().clone()
        captured["metadata"] = dict(kwargs["metadata"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": float(torch.linalg.norm(kwargs["delta"]).item()),
            "editor": {
                **dict(kwargs["metadata"]),
                "delta_sha256": "9" * 64,
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.0,
            "target_margin": 0.5,
        }

    monkeypatch.setattr(v25, "v32_support_behavior_margins", fake_margins)
    monkeypatch.setattr(v25, "control_record_for_delta", fake_control_record)
    monkeypatch.setattr(
        v25,
        "record_progress_event",
        lambda *_args, event, extra, **_kwargs: events.append({
            "event": event,
            "extra": dict(extra),
        }),
    )

    result = v25.evaluate_v25_support_source_line_search_matched_edit(
        subject={
            "pattern": "sorted_ascending",
            "subject_id": "hidden",
            "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
        },
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        config={
            **v25.build_v35_support_source_line_search_config_grid()[1],
            "alpha_candidates": [1.0, 0.5, 0.125],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=1.25,
        selected_config_hash="c" * 64,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        script_sha256="script",
        progress_log_path=Path("progress.jsonl"),
        record_id_hash="d" * 64,
    )

    assert torch.isclose(captured["delta"][0], torch.tensor(0.5))
    assert result["editor"]["matched_edit_source"] == (
        v25.V35_SUPPORT_SOURCE_LINE_SEARCH_MATCHED_EDIT_SOURCE
    )
    assert captured["metadata"]["selected_alpha"] == 0.5
    assert captured["metadata"]["alpha_selection"]["selection_mode"] == (
        "eligible_min_compatible_mse"
    )
    assert any(
        event["event"] == "v35_support_source_alpha_selected"
        and event["extra"]["alpha"] == 0.5
        and event["extra"]["selection_mode"] == "eligible_min_compatible_mse"
        for event in events
    )
    log_text = json.dumps(events, sort_keys=True)
    assert "weights" not in log_text
    assert "subject_id" not in log_text
    assert "logits" not in log_text
    assert "sequence" not in log_text


def test_v32_configs_use_v25_native_control_baseline() -> None:
    config = {
        "matched_edit_source": "support_tournament_margin_sparse",
        "sparse_top_k": 64,
        "trust_norm_cap": 1.25,
    }

    expected = {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }

    assert v25.v25_config_requires_spectral_basis(config) is False
    assert v25.v25_spectral_seed_config(config) == expected
    assert v25.v25_native_control_config(config) == expected
    assert "support_tournament_margin_sparse" not in v25.PROOF_CRITICAL_CONTROL_TYPES


def test_v32_progress_redaction_omits_raw_values() -> None:
    event = v25.redact_v32_support_tournament_progress_event({
        "coordinate_hash": "a" * 64,
        "delta_norm": 1.0,
        "final_subjects_path": str(v25.V25_FINAL_RAW),
        "gradient": torch.ones(2),
        "logits": torch.ones(2),
        "record_id_hash": "b" * 64,
        "selected_config_hash": "c" * 64,
        "step": 10,
        "subject_id": "raw-subject",
        "support_examples": [{"sequence": [0, 1, 2, 3, 4]}],
        "support_tournament_margin": 0.25,
        "tournament_margin_floor": 0.15,
        "tournament_margin_hinge": 0.0,
        "tournament_margin_weight": 1.0,
        "weights": torch.ones(2),
    })
    event_text = json.dumps(event, sort_keys=True)

    assert event["coordinate_hash"] == "a" * 64
    assert event["support_tournament_margin"] == 0.25
    assert event["finite"] is True
    for forbidden in [
        "final_subjects_path",
        "gradient",
        "logits",
        "subject_id",
        "support_examples",
        "weights",
    ]:
        assert forbidden not in event
    assert "final_subjects" not in event_text
    assert "sequence" not in event_text


def test_v32_support_tournament_margin_loss_rewards_target_runner_gap() -> None:
    satisfied = v25.v32_support_tournament_margin_loss(
        margins={
            "has_majority": torch.tensor(0.5),
            "mountain_pattern": torch.tensor(0.1),
            "sorted_ascending": torch.tensor(0.2),
            "sorted_descending": torch.tensor(0.0),
        },
        target_behavior="has_majority",
        tournament_margin_floor=0.15,
    )
    unsatisfied = v25.v32_support_tournament_margin_loss(
        margins={
            "has_majority": torch.tensor(0.2),
            "mountain_pattern": torch.tensor(0.1),
            "sorted_ascending": torch.tensor(0.18),
            "sorted_descending": torch.tensor(0.0),
        },
        target_behavior="has_majority",
        tournament_margin_floor=0.15,
    )

    assert torch.isclose(satisfied["loss"], torch.tensor(0.0))
    assert float(unsatisfied["loss"].item()) > 0.0
    assert torch.isclose(unsatisfied["support_tournament_margin"], torch.tensor(0.02))


def test_v32_support_behavior_margin_tensors_expose_only_counts_and_hash() -> None:
    tensors = v25.v32_support_behavior_margin_tensors()

    assert sorted(tensors["by_behavior"]) == sorted(v25.PATTERNS)
    assert len(tensors["tensor_hash"]) == 64
    assert all(
        tensors["counts_by_behavior"][pattern]["positive"] > 0
        and tensors["counts_by_behavior"][pattern]["negative"] > 0
        for pattern in v25.PATTERNS
    )
    assert "sequence" not in json.dumps({
        "counts_by_behavior": tensors["counts_by_behavior"],
        "tensor_hash": tensors["tensor_hash"],
    })


def test_v32_support_tournament_tensor_hash_binds_content_and_order() -> None:
    counts = {
        pattern: {"negative": 1, "positive": 1}
        for pattern in v25.PATTERNS
    }
    base = {
        pattern: {
            "negative_inputs": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            "positive_inputs": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        }
        for pattern in v25.PATTERNS
    }
    changed = {
        pattern: {
            "negative_inputs": tensors["negative_inputs"].clone(),
            "positive_inputs": tensors["positive_inputs"].clone(),
        }
        for pattern, tensors in base.items()
    }
    changed[v25.PATTERNS[0]]["positive_inputs"] = torch.tensor(
        [[0.0, 1.0]],
        dtype=torch.float32,
    )

    base_hash = v25.v32_support_behavior_margin_tensor_hash(
        by_behavior=base,
        counts_by_behavior=counts,
    )
    changed_hash = v25.v32_support_behavior_margin_tensor_hash(
        by_behavior=changed,
        counts_by_behavior=counts,
    )

    assert len(base_hash) == 64
    assert len(changed_hash) == 64
    assert base_hash != changed_hash


def test_v32_optimizer_respects_trust_cap_and_logs_tournament_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = {
        "compatible_inputs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "compatible_source_logits": torch.zeros(2, dtype=torch.float32),
        "conflict_inputs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "conflict_target_labels": torch.ones(2, dtype=torch.float32),
        "target_inputs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "target_labels": torch.ones(2, dtype=torch.float32),
    }
    tournament = {
        "by_behavior": {
            pattern: {
                "negative_inputs": torch.tensor([[0.0]], dtype=torch.float32),
                "positive_inputs": torch.tensor([[1.0]], dtype=torch.float32),
            }
            for pattern in v25.PATTERNS
        },
        "counts_by_behavior": {
            pattern: {"negative": 1, "positive": 1}
            for pattern in v25.PATTERNS
        },
        "tensor_hash": "d" * 64,
    }
    events: list[dict[str, Any]] = []

    monkeypatch.setattr(v25, "v27_support_tensors_for_source_target", lambda **_: support)
    monkeypatch.setattr(v25, "v32_support_behavior_margin_tensors", lambda: tournament)
    monkeypatch.setattr(
        v25,
        "v27_subject_logits_for_inputs",
        lambda flat, inputs: flat[0].expand(int(inputs.shape[0])),
    )
    monkeypatch.setattr(
        v25,
        "record_progress_event",
        lambda *_args, event, extra, **_kwargs: events.append({
            "event": event,
            "extra": dict(extra),
        }),
    )

    edit = v25.solve_v32_support_tournament_sparse_edit(
        compatible_gradient=torch.ones(v25.SOURCE_WEIGHT_DIM),
        coordinate_hash="a" * 64,
        selected_coordinates=[0],
        config={
            "compatible_floor": 0.05,
            "compatible_orthogonal_weight": 0.05,
            "extra_compatible_weight": 0.05,
            "hard_target_margin_weight": 1.0,
            "sign_conflict_penalty": 1.0,
            "sparse_top_k": 1,
            "target_margin_floor": 0.25,
            "tournament_margin_floor": 0.15,
            "tournament_margin_weight": 1.0,
            "trust_norm_cap": 0.05,
        },
        source_weights=torch.zeros(v25.SOURCE_WEIGHT_DIM),
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        progress_log_path=Path("progress.jsonl"),
        record_id_hash="b" * 64,
        selected_config_hash="c" * 64,
    )

    assert float(torch.linalg.norm(edit["delta"]).item()) <= 0.050001
    assert edit["audit"]["tournament_margin_floor"] == 0.15
    assert any(
        item["event"] == "v32_support_tournament_optimizer_progress"
        and "tournament_margin_hinge" in item["extra"]
        and "support_tournament_margin" in item["extra"]
        for item in events
    )
    assert any(item["event"] == "v32_support_tournament_optimizer_completed" for item in events)


def test_v32_dispatch_uses_v32_matched_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_matched(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": "support_tournament_margin_sparse",
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    def fake_controls(**kwargs: Any) -> list[dict[str, Any]]:
        called["control_config"] = dict(kwargs["config"])
        return []

    def fake_proof(**kwargs: Any) -> dict[str, Any]:
        return {
            "controls": [],
            "direction": "sorted_ascending->sorted_descending",
            "matched": dict(kwargs["matched"]),
            "source_behavior": kwargs["source_behavior"],
            "summary": {
                "individual_all_gates_passed": True,
                "pareto_undominated": True,
                "target_prediction_pass": True,
            },
            "target_behavior": kwargs["target_behavior"],
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(
        v25,
        "evaluate_v25_support_tournament_sparse_matched_edit",
        fake_matched,
    )
    monkeypatch.setattr(v25, "build_v25_native_controls", fake_controls)
    monkeypatch.setattr(v25, "build_v25_proof_record", fake_proof)

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            "matched_edit_source": "support_tournament_margin_sparse",
            "sparse_top_k": 64,
            "target_margin_floor": 0.25,
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "e" * 64},
        selected_config_hash="c" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == "support_tournament_margin_sparse"
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        "support_tournament_margin_sparse"
    )


def test_v35_dispatch_uses_line_search_matched_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_matched(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": (
                    v25.V35_SUPPORT_SOURCE_LINE_SEARCH_MATCHED_EDIT_SOURCE
                ),
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    def fake_controls(**kwargs: Any) -> list[dict[str, Any]]:
        called["control_config"] = dict(kwargs["config"])
        return []

    def fake_proof(**kwargs: Any) -> dict[str, Any]:
        return {
            "controls": [],
            "direction": "sorted_ascending->sorted_descending",
            "matched": dict(kwargs["matched"]),
            "source_behavior": kwargs["source_behavior"],
            "summary": {
                "individual_all_gates_passed": True,
                "pareto_undominated": True,
                "target_prediction_pass": True,
            },
            "target_behavior": kwargs["target_behavior"],
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(
        v25,
        "evaluate_v25_support_source_line_search_matched_edit",
        fake_matched,
    )
    monkeypatch.setattr(v25, "build_v25_native_controls", fake_controls)
    monkeypatch.setattr(v25, "build_v25_proof_record", fake_proof)

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            **v25.build_v35_support_source_line_search_config_grid()[0],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "e" * 64},
        selected_config_hash="c" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == (
        v25.V35_SUPPORT_SOURCE_LINE_SEARCH_MATCHED_EDIT_SOURCE
    )
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        v25.V35_SUPPORT_SOURCE_LINE_SEARCH_MATCHED_EDIT_SOURCE
    )


def test_v36_projection_reduces_raw_preservation_energy() -> None:
    base_delta = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    base_delta[0] = 1.0
    base_delta[1] = 2.0
    compatible_jacobian = torch.zeros(1, v25.SOURCE_WEIGHT_DIM)
    compatible_jacobian[0, 0] = 1.0

    projected = v25.project_v36_delta_through_compatible_nullspace(
        base_delta=base_delta,
        compatible_jacobian=compatible_jacobian,
        compatible_nullspace_rtol=1e-4,
        projection_strength=1.0,
        trust_norm_cap=10.0,
    )

    assert torch.isclose(projected["delta"][0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(projected["delta"][1], torch.tensor(2.0), atol=1e-6)
    assert projected["audit"]["preserve_rank"] == 1
    assert projected["audit"]["projected_preservation_energy"] < (
        projected["audit"]["base_preservation_energy"]
    )
    assert projected["audit"]["preservation_energy_ratio"] < 1e-5


def test_v36_projection_handles_zero_compatible_jacobian() -> None:
    base_delta = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    base_delta[3] = 0.5
    compatible_jacobian = torch.zeros(2, v25.SOURCE_WEIGHT_DIM)

    projected = v25.project_v36_delta_through_compatible_nullspace(
        base_delta=base_delta,
        compatible_jacobian=compatible_jacobian,
        compatible_nullspace_rtol=1e-4,
        projection_strength=1.0,
        trust_norm_cap=10.0,
    )

    assert torch.allclose(projected["delta"], base_delta)
    assert projected["audit"]["preserve_rank"] == 0
    assert projected["audit"]["base_preservation_energy"] == 0.0
    assert projected["audit"]["projected_preservation_energy"] == 0.0
    assert projected["audit"]["finite"] is True


def test_v36_projection_progress_redaction_omits_raw_values() -> None:
    event = v25.redact_v36_compatible_nullspace_progress_event({
        "base_delta": torch.ones(2),
        "base_preservation_energy": 1.0,
        "basis": torch.eye(2),
        "compatible_jacobian": torch.ones(1, 2),
        "compatible_nullspace_rtol": 1e-4,
        "final_subjects_path": str(v25.V25_FINAL_RAW),
        "jacobian_row_count": 2,
        "preservation_energy_ratio": 0.25,
        "preserve_rank": 1,
        "projected_delta": torch.ones(2),
        "projected_preservation_energy": 0.25,
        "projection_removed_norm": 0.75,
        "projection_retained_norm": 0.5,
        "projection_strength": 1.0,
        "record_id_hash": "a" * 64,
        "selected_config_hash": "b" * 64,
        "subject_id": "raw",
        "weights": torch.ones(2),
    })
    text = json.dumps(event, sort_keys=True)

    assert event["base_preservation_energy"] == 1.0
    assert event["projected_preservation_energy"] == 0.25
    assert event["preserve_rank"] == 1
    assert event["finite"] is True
    for forbidden in [
        "base_delta",
        "basis",
        "compatible_jacobian",
        "final_subjects_path",
        "projected_delta",
        "subject_id",
        "weights",
    ]:
        assert forbidden not in event
    assert "final_subjects" not in text


def test_v36_compatible_nullspace_grid_is_bounded_and_binds_provenance() -> None:
    grid = v25.build_v36_compatible_nullspace_projection_config_grid()

    assert len(grid) == 4
    assert v25.stable_hash_json(grid) == v25.V36_COMPATIBLE_NULLSPACE_GRID_SHA256
    assert [item["config_index"] for item in grid] == list(range(4))
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V36_COMPATIBLE_NULLSPACE_MATCHED_EDIT_SOURCE
    }
    assert sorted({item["compatible_nullspace_rtol"] for item in grid}) == [
        1e-4,
        1e-3,
    ]
    assert sorted({item["projection_strength"] for item in grid}) == [0.75, 1.0]

    configs = v25.select_v25_inner_validation_configs(
        grid_name="v36-compatible-nullspace-projection",
        max_configs=None,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid(
        "v36-compatible-nullspace-projection"
    ) == v25.V36_EXPERIMENT_VARIANT


def test_v37_projected_optimizer_grid_is_bounded_and_binds_provenance() -> None:
    grid = v25.build_v37_projected_support_optimizer_config_grid()

    assert len(grid) == 4
    assert v25.stable_hash_json(grid) == v25.V37_PROJECTED_OPTIMIZER_GRID_SHA256
    assert [item["config_index"] for item in grid] == list(range(4))
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE
    }
    assert sorted({item["compatible_nullspace_rtol"] for item in grid}) == [
        1e-3,
        1e-2,
    ]
    assert sorted({item["projection_strength"] for item in grid}) == [0.5, 0.75]
    assert all(item["projected_optimizer_epochs"] == 80 for item in grid)

    configs = v25.select_v25_inner_validation_configs(
        grid_name="v37-projected-support-optimizer",
        max_configs=None,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid(
        "v37-projected-support-optimizer"
    ) == v25.V37_EXPERIMENT_VARIANT


def test_v38_compatible_gated_grid_is_bounded_and_binds_provenance() -> None:
    grid = v25.build_v38_compatible_mse_gated_projected_optimizer_config_grid()

    assert len(grid) == 4
    assert v25.stable_hash_json(grid) == v25.V38_COMPATIBLE_GATED_GRID_SHA256
    assert [item["config_index"] for item in grid] == list(range(4))
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE
    }
    assert sorted({item["compatible_mse_gate"] for item in grid}) == [5.0, 15.0]
    assert sorted({item["compatible_gate_weight"] for item in grid}) == [0.5, 1.5]

    configs = v25.select_v25_inner_validation_configs(
        grid_name="v38-compatible-mse-gated-projected-optimizer",
        max_configs=None,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid(
        "v38-compatible-mse-gated-projected-optimizer"
    ) == v25.V38_EXPERIMENT_VARIANT


def test_v39_target_feasible_grid_is_bounded_and_binds_provenance() -> None:
    grid = v25.build_v39_target_feasible_lexicographic_optimizer_config_grid()

    assert len(grid) == 4
    assert v25.stable_hash_json(grid) == v25.V39_GRID_SHA256
    assert [item["config_index"] for item in grid] == list(range(4))
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V39_TARGET_FEASIBLE_LEXICOGRAPHIC_MATCHED_EDIT_SOURCE
    }
    assert sorted({item["alpha_compatible_mse_soft_gate"] for item in grid}) == [
        10.0,
        20.0,
    ]
    assert sorted({item["compatible_gate_weight"] for item in grid}) == [0.25, 0.75]
    assert {item["projected_optimizer_event_prefix"] for item in grid} == {
        "v39_target_feasible_lexicographic_optimizer"
    }

    configs = v25.select_v25_inner_validation_configs(
        grid_name="v39-target-feasible-lexicographic-projected-optimizer",
        max_configs=None,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid(
        "v39-target-feasible-lexicographic-projected-optimizer"
    ) == v25.V39_EXPERIMENT_VARIANT


def test_v40_target_tolerance_grid_is_bounded_and_binds_provenance() -> None:
    grid = v25.build_v40_target_tolerance_locality_budget_config_grid()

    assert len(grid) == 4
    assert v25.stable_hash_json(grid) == v25.V40_GRID_SHA256
    assert [item["config_index"] for item in grid] == list(range(4))
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V40_TARGET_TOLERANCE_LOCALITY_BUDGET_MATCHED_EDIT_SOURCE
    }
    assert sorted({item["alpha_compatible_mse_soft_gate"] for item in grid}) == [
        10.0,
        20.0,
    ]
    assert sorted({item["target_rank_score_tolerance"] for item in grid}) == [
        0.05,
        0.15,
    ]
    assert {item["projected_optimizer_event_prefix"] for item in grid} == {
        "v40_target_tolerance_locality_budget_optimizer"
    }

    configs = v25.select_v25_inner_validation_configs(
        grid_name="v40-target-tolerance-locality-budget-projected-optimizer",
        max_configs=None,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid(
        "v40-target-tolerance-locality-budget-projected-optimizer"
    ) == v25.V40_EXPERIMENT_VARIANT


def test_v41_trajectory_frontier_grid_is_bounded_and_binds_provenance() -> None:
    grid = v25.build_v41_trajectory_frontier_config_grid()

    assert len(grid) == 4
    assert v25.stable_hash_json(grid) == v25.V41_GRID_SHA256
    assert [item["config_index"] for item in grid] == list(range(4))
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V41_TRAJECTORY_FRONTIER_MATCHED_EDIT_SOURCE
    }
    assert all(item["trajectory_frontier_enabled"] is True for item in grid)
    assert sorted({item["alpha_compatible_mse_soft_gate"] for item in grid}) == [
        10.0,
        20.0,
    ]
    assert sorted({item["target_rank_score_tolerance"] for item in grid}) == [
        0.05,
        0.15,
    ]
    assert {item["projected_optimizer_event_prefix"] for item in grid} == {
        "v41_trajectory_frontier_optimizer"
    }
    assert {item["trajectory_frontier_event_prefix"] for item in grid} == {
        "v41_trajectory_frontier"
    }

    configs = v25.select_v25_inner_validation_configs(
        grid_name="v41-trajectory-frontier-projected-optimizer",
        max_configs=None,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid(
        "v41-trajectory-frontier-projected-optimizer"
    ) == v25.V41_EXPERIMENT_VARIANT


def test_v42_compatible_dual_frontier_grid_is_bounded_and_binds_provenance() -> None:
    grid = v25.build_v42_compatible_dual_frontier_config_grid()

    assert len(grid) == 4
    assert v25.stable_hash_json(grid) == v25.V42_GRID_SHA256
    assert [item["config_index"] for item in grid] == list(range(4))
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
    }
    assert {item["experiment_variant"] for item in grid} == {
        v25.V42_EXPERIMENT_VARIANT
    }
    assert {item["projected_optimizer_event_prefix"] for item in grid} == {
        "v42_compatible_dual_frontier_optimizer"
    }
    assert {item["trajectory_frontier_event_prefix"] for item in grid} == {
        "v42_compatible_dual_frontier"
    }
    assert all(item["v42_compatible_dual_enabled"] is True for item in grid)
    assert all(item["trajectory_frontier_enabled"] is True for item in grid)
    assert all(item["projected_optimizer_epochs"] == 80 for item in grid)
    assert {item["compatible_mse_budget"] for item in grid} == {10.0, 20.0}
    assert {item["compatible_augmented_weight"] for item in grid} == {0.5, 2.0}

    configs = v25.select_v25_inner_validation_configs(
        grid_name="v42-compatible-dual-frontier",
        max_configs=None,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)


def test_v42_variant_routing_and_native_controls_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        **v25.build_v42_compatible_dual_frontier_config_grid()[0],
        "train_pool_file_sha256": "a" * 64,
        "train_pool_summary_hash": "b" * 64,
    }

    assert v25.experiment_variant_for_inner_validation_grid(
        "v42-compatible-dual-frontier"
    ) == v25.V42_EXPERIMENT_VARIANT
    assert v25.experiment_variant_for_config(config) == v25.V42_EXPERIMENT_VARIANT
    assert v25.v25_config_requires_spectral_basis(config) is False
    assert v25.v25_native_control_config(config) == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--phase",
            "development",
            "--run-inner-validation",
            "--inner-validation-config-grid",
            "v42-compatible-dual-frontier",
        ],
    )
    args = v25.parse_args()
    assert args.inner_validation_config_grid == "v42-compatible-dual-frontier"


def test_v37_projected_optimizer_progress_redaction_omits_raw_values() -> None:
    event = v25.redact_v37_projected_optimizer_progress_event({
        "basis": torch.eye(2),
        "compatible_jacobian": torch.ones(1, 2),
        "final_subjects_path": str(v25.V25_FINAL_RAW),
        "gradient": torch.ones(2),
        "logits": torch.ones(2),
        "projected_delta": torch.ones(2),
        "raw_delta": torch.ones(2),
        "selected_coordinates": [1, 2],
        "sequence": [0, 1, 1, 0],
        "subject_id": "raw",
        "support_examples": [{"x": [1, 0], "y": 1}],
        "weights": torch.ones(2),
        "optimizer_audit_hash": "a" * 64,
        "record_id_hash": "b" * 64,
        "selected_config_hash": "c" * 64,
        "compatible_nullspace_rtol": 1e-3,
        "projection_strength": 0.5,
        "optimization_steps": 80,
        "preserve_rank": 3,
        "jacobian_row_count": 4,
        "final_loss": 0.25,
        "support_compatible_mse": 0.01,
        "support_target_margin": 0.2,
        "support_tournament_margin": 0.1,
        "preservation_energy_ratio": 0.4,
        "projected_delta_norm": 0.3,
    })
    text = json.dumps(event, sort_keys=True)

    assert event["optimization_steps"] == 80
    assert event["preserve_rank"] == 3
    assert event["finite"] is True
    for forbidden in [
        "basis",
        "compatible_jacobian",
        "final_subjects_path",
        "gradient",
        "logits",
        "projected_delta",
        "raw_delta",
        "selected_coordinates",
        "sequence",
        "subject_id",
        "support_examples",
        "weights",
    ]:
        assert forbidden not in event
    assert "final_subjects" not in text


def test_v38_alpha_redaction_keeps_gate_evidence_and_omits_raw_fields() -> None:
    event = v25.redact_v38_compatible_gated_alpha_progress_event({
        "alpha": 0.5,
        "alpha_candidate_count": 2,
        "alpha_candidates_hash": "a" * 64,
        "alpha_compatible_mse_gate": 5.0,
        "basis": torch.eye(2),
        "candidate_metrics_hash": "b" * 64,
        "compatible_gate_pass": True,
        "compatible_jacobian": torch.ones(1, 2),
        "eligible_count": 1,
        "fallback_compatible_penalty": 2.0,
        "fallback_score": 4.0,
        "final_subjects_path": str(v25.V25_FINAL_RAW),
        "gradient": torch.ones(2),
        "logits": torch.ones(2),
        "projected_delta": torch.ones(2),
        "raw_delta": torch.ones(2),
        "record_id_hash": "c" * 64,
        "selected_alpha_candidate_hash": "d" * 64,
        "selected_config_hash": "e" * 64,
        "selected_coordinates": [1, 2],
        "selection_mode": "eligible_min_compatible_mse",
        "sequence": [0, 1, 1, 0],
        "subject_id": "raw",
        "support_compatible_mse": 4.0,
        "support_examples": [{"x": [1, 0], "y": 1}],
        "support_target_margin": 0.4,
        "support_tournament_margin": 0.3,
        "weights": torch.ones(2),
    })
    text = json.dumps(event, sort_keys=True)

    assert event["compatible_gate_pass"] is True
    assert event["alpha_compatible_mse_gate"] == 5.0
    assert event["fallback_compatible_penalty"] == 2.0
    assert event["eligible_count"] == 1
    assert event["candidate_metrics_hash"] == "b" * 64
    for forbidden in [
        "basis",
        "compatible_jacobian",
        "final_subjects_path",
        "gradient",
        "logits",
        "projected_delta",
        "raw_delta",
        "selected_coordinates",
        "sequence",
        "subject_id",
        "support_examples",
        "weights",
    ]:
        assert forbidden not in event
    assert "final_subjects" not in text
    assert "raw" not in text


def test_v39_alpha_redaction_keeps_target_feasibility_audit_and_omits_raw_fields() -> None:
    event = v25.redact_v39_target_feasible_alpha_progress_event({
        "alpha": 0.75,
        "alpha_candidate_count": 3,
        "alpha_candidates_hash": "a" * 64,
        "alpha_compatible_mse_soft_gate": 10.0,
        "candidate_index": 2,
        "candidate_metrics_hash": "b" * 64,
        "compatible_gap": 1.5,
        "compatible_jacobian": torch.ones(1, 2),
        "eligible_count": 0,
        "final_subjects": [{"subject_id": "sealed"}],
        "final_subjects_path": str(v25.V25_FINAL_RAW),
        "gradient": torch.ones(2),
        "logits": torch.ones(2),
        "projected_delta": torch.ones(2),
        "raw_delta": torch.ones(2),
        "record_id_hash": "c" * 64,
        "selected_alpha_candidate_hash": "d" * 64,
        "selected_config_hash": "e" * 64,
        "selected_coordinates": [1, 2],
        "selection_mode": "fallback_target_feasible_lexicographic",
        "sequence": [0, 1, 1, 0],
        "subject_id": "raw",
        "support_compatible_mse": 11.5,
        "support_examples": [{"x": [1, 0], "y": 1}],
        "support_runner_margin": -0.1,
        "support_target_margin": 0.2,
        "support_tournament_margin": 0.15,
        "target_feasible": False,
        "target_gap": 0.0,
        "target_rank_score": 0.0,
        "tournament_gap": 0.0,
        "weights": torch.ones(2),
    })
    text = json.dumps(event, sort_keys=True)

    assert event["alpha_compatible_mse_soft_gate"] == 10.0
    assert event["compatible_gap"] == 1.5
    assert event["target_feasible"] is False
    assert event["target_gap"] == 0.0
    assert event["target_rank_score"] == 0.0
    assert event["tournament_gap"] == 0.0
    assert event["selection_mode"] == "fallback_target_feasible_lexicographic"
    assert event["candidate_metrics_hash"] == "b" * 64
    for forbidden in [
        "compatible_jacobian",
        "final_subjects",
        "final_subjects_path",
        "gradient",
        "logits",
        "projected_delta",
        "raw_delta",
        "selected_coordinates",
        "sequence",
        "subject_id",
        "support_examples",
        "weights",
    ]:
        assert forbidden not in event
    assert "final_subjects" not in text
    assert "raw" not in text


def test_v40_alpha_redaction_keeps_tolerance_audit_and_omits_raw_fields() -> None:
    event = v25.redact_v40_target_tolerance_alpha_progress_event({
        "alpha": 0.5,
        "alpha_candidate_count": 4,
        "alpha_candidates_hash": "a" * 64,
        "alpha_compatible_mse_soft_gate": 10.0,
        "best_target_rank_score": 0.2,
        "candidate_index": 1,
        "candidate_metrics_hash": "b" * 64,
        "compatible_gap": 0.0,
        "compatible_jacobian": torch.ones(1, 2),
        "eligible_count": 0,
        "final_subjects": [{"subject_id": "sealed"}],
        "final_subjects_path": str(v25.V25_FINAL_RAW),
        "gradient": torch.ones(2),
        "logits": torch.ones(2),
        "projected_delta": torch.ones(2),
        "raw_delta": torch.ones(2),
        "record_id_hash": "c" * 64,
        "selected_alpha_candidate_hash": "d" * 64,
        "selected_config_hash": "e" * 64,
        "selected_coordinates": [1, 2],
        "selection_mode": "target_tolerance_min_compatible_mse",
        "sequence": [0, 1, 1, 0],
        "subject_id": "raw",
        "support_compatible_mse": 4.0,
        "support_examples": [{"x": [1, 0], "y": 1}],
        "support_runner_margin": -0.1,
        "support_target_margin": -0.1,
        "support_tournament_margin": -0.1,
        "target_feasible": False,
        "target_gap": 0.15,
        "target_rank_score": 0.25,
        "target_rank_score_tolerance": 0.05,
        "tournament_gap": 0.1,
        "weights": torch.ones(2),
        "within_target_tolerance_count": 2,
    })
    text = json.dumps(event, sort_keys=True)

    assert event["best_target_rank_score"] == 0.2
    assert event["target_rank_score_tolerance"] == 0.05
    assert event["within_target_tolerance_count"] == 2
    assert event["selection_mode"] == "target_tolerance_min_compatible_mse"
    assert event["candidate_metrics_hash"] == "b" * 64
    for forbidden in [
        "compatible_jacobian",
        "final_subjects",
        "final_subjects_path",
        "gradient",
        "logits",
        "projected_delta",
        "raw_delta",
        "selected_coordinates",
        "sequence",
        "subject_id",
        "support_examples",
        "weights",
    ]:
        assert forbidden not in event
    assert "final_subjects" not in text
    assert "raw" not in text


def test_v41_frontier_redaction_omits_raw_fields() -> None:
    event = v25.redact_v41_trajectory_frontier_progress_event({
        "best_target_rank_score": 0.1,
        "candidate_metrics_hash": "a" * 64,
        "compatible_gap": 2.0,
        "compatible_jacobian": torch.ones(1, 2),
        "final_subjects": [{"subject_id": "sealed"}],
        "final_subjects_path": str(v25.V25_FINAL_RAW),
        "frontier_candidate_count": 5,
        "frontier_candidates_hash": "b" * 64,
        "gradient": torch.ones(2),
        "logits": torch.ones(2),
        "preservation_energy_ratio": 0.5,
        "projected_delta": torch.ones(2),
        "projected_delta_norm": 0.25,
        "raw_delta": torch.ones(2),
        "record_id_hash": "c" * 64,
        "selected_config_hash": "d" * 64,
        "selected_coordinates": [1, 2],
        "selection_mode": "frontier_target_tolerance_min_compatible_mse",
        "sequence": [0, 1, 1, 0],
        "subject_id": "raw",
        "support_compatible_mse": 4.0,
        "support_examples": [{"x": [1, 0], "y": 1}],
        "support_target_margin": -0.1,
        "support_tournament_margin": -0.1,
        "target_feasible": False,
        "target_gap": 0.15,
        "target_rank_score": 0.25,
        "target_rank_score_tolerance": 0.05,
        "trajectory_frontier_selected_epoch": 8,
        "tournament_gap": 0.1,
        "weights": torch.ones(2),
        "within_target_tolerance_count": 2,
    })
    text = json.dumps(event, sort_keys=True)

    assert event["trajectory_frontier_selected_epoch"] == 8
    assert event["frontier_candidate_count"] == 5
    assert event["within_target_tolerance_count"] == 2
    assert event["selection_mode"] == "frontier_target_tolerance_min_compatible_mse"
    assert event["frontier_candidates_hash"] == "b" * 64
    for forbidden in [
        "compatible_jacobian",
        "final_subjects",
        "final_subjects_path",
        "gradient",
        "logits",
        "projected_delta",
        "raw_delta",
        "selected_coordinates",
        "sequence",
        "subject_id",
        "support_examples",
        "weights",
    ]:
        assert forbidden not in event
    assert "final_subjects" not in text
    assert "raw" not in text


def test_v37_differentiable_projection_removes_row_component_and_backprops() -> None:
    sparse_delta = torch.zeros(v25.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    sparse_delta[0] = 1.0
    sparse_delta[1] = 2.0
    sparse_delta.requires_grad_(True)
    compatible_jacobian = torch.zeros(1, v25.SOURCE_WEIGHT_DIM)
    compatible_jacobian[0, 0] = 1.0

    projected, audit = v25.project_v37_delta_differentiably(
        sparse_delta=sparse_delta,
        compatible_jacobian=compatible_jacobian,
        compatible_nullspace_rtol=1e-4,
        projection_strength=1.0,
        trust_norm_cap=10.0,
    )
    loss = projected[1] ** 2
    loss.backward()

    assert torch.isclose(projected[0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(projected[1], torch.tensor(2.0), atol=1e-6)
    assert sparse_delta.grad is not None
    assert torch.isclose(sparse_delta.grad[1], torch.tensor(4.0), atol=1e-6)
    assert audit["preserve_rank"] == 1
    assert audit["preservation_energy_ratio"] < 1e-5


def test_v37_projected_optimizer_optimizes_target_after_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    support = {
        "compatible_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "compatible_source_logits": torch.zeros(1, dtype=torch.float32),
        "conflict_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "conflict_target_labels": torch.ones(1, dtype=torch.float32),
        "target_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "target_labels": torch.ones(1, dtype=torch.float32),
    }
    compatible_jacobian = torch.zeros(1, v25.SOURCE_WEIGHT_DIM)
    compatible_jacobian[0, 0] = 1.0

    monkeypatch.setattr(v25, "v27_support_tensors_for_source_target", lambda **_: support)
    monkeypatch.setattr(v25, "v32_support_behavior_margin_tensors", lambda: {
        "by_behavior": {},
        "counts_by_behavior": {},
        "tensor_hash": "e" * 64,
    })
    monkeypatch.setattr(
        v25,
        "v27_subject_logits_for_inputs",
        lambda flat, inputs: flat[1].expand(int(inputs.shape[0])),
    )

    def fake_margins(*, weights: torch.Tensor, tournament_tensors: Mapping[str, Any]):
        return {
            "has_majority": weights[1] - 0.30,
            "mountain_pattern": weights[1] - 0.20,
            "sorted_ascending": weights[1] - 0.10,
            "sorted_descending": weights[1],
        }

    monkeypatch.setattr(v25, "v32_support_behavior_margins", fake_margins)
    result = v25.solve_v37_projected_support_optimizer_edit(
        compatible_gradient=torch.ones(v25.SOURCE_WEIGHT_DIM),
        compatible_jacobian=compatible_jacobian,
        coordinate_hash="a" * 64,
        selected_coordinates=[0, 1],
        config={
            **v25.build_v37_projected_support_optimizer_config_grid()[0],
            "projected_optimizer_epochs": 30,
            "projected_optimizer_lr": 0.2,
            "projection_strength": 1.0,
            "trust_norm_cap": 1.0,
        },
        source_weights=source,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
    )

    assert result["delta"][0].abs() < 1e-5
    assert result["delta"][1] > 0.2
    assert result["audit"]["support_target_margin"] > 0.0
    assert result["audit"]["preservation_energy_ratio"] < 1e-5


def test_v41_projected_optimizer_returns_frontier_selected_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    support = {
        "compatible_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "compatible_source_logits": torch.zeros(1, dtype=torch.float32),
        "conflict_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "conflict_target_labels": torch.ones(1, dtype=torch.float32),
        "target_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "target_labels": torch.ones(1, dtype=torch.float32),
    }
    compatible_jacobian = torch.zeros(1, v25.SOURCE_WEIGHT_DIM)
    compatible_jacobian[0, 0] = 1.0

    monkeypatch.setattr(v25, "v27_support_tensors_for_source_target", lambda **_: support)
    monkeypatch.setattr(v25, "v32_support_behavior_margin_tensors", lambda: {
        "by_behavior": {},
        "counts_by_behavior": {},
        "tensor_hash": "e" * 64,
    })
    monkeypatch.setattr(
        v25,
        "v27_subject_logits_for_inputs",
        lambda flat, inputs: flat[1].expand(int(inputs.shape[0])),
    )
    monkeypatch.setattr(v25, "v32_support_behavior_margins", lambda **kwargs: {
        "has_majority": kwargs["weights"][1] - 0.30,
        "mountain_pattern": kwargs["weights"][1] - 0.20,
        "sorted_ascending": kwargs["weights"][1] - 0.10,
        "sorted_descending": kwargs["weights"][1],
    })

    result = v25.solve_v37_projected_support_optimizer_edit(
        compatible_gradient=torch.ones(v25.SOURCE_WEIGHT_DIM),
        compatible_jacobian=compatible_jacobian,
        coordinate_hash="a" * 64,
        selected_coordinates=[0, 1],
        config={
            **v25.build_v41_trajectory_frontier_config_grid()[0],
            "projected_optimizer_epochs": 5,
            "projected_optimizer_lr": 0.2,
            "projection_strength": 1.0,
            "trust_norm_cap": 1.0,
        },
        source_weights=source,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
    )

    assert result["audit"]["trajectory_frontier_candidate_count"] == 5
    assert 1 <= result["audit"]["trajectory_frontier_selected_epoch"] <= 5
    assert result["audit"]["trajectory_frontier_selected_hash"]
    assert "trajectory_frontier_selection_mode" in result["audit"]


def test_v37_projected_optimizer_final_loss_matches_returned_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    support = {
        "compatible_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "compatible_source_logits": torch.zeros(1, dtype=torch.float32),
        "conflict_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "conflict_target_labels": torch.ones(1, dtype=torch.float32),
        "target_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "target_labels": torch.ones(1, dtype=torch.float32),
    }
    compatible_gradient = torch.ones(v25.SOURCE_WEIGHT_DIM)
    compatible_jacobian = torch.zeros(1, v25.SOURCE_WEIGHT_DIM)
    compatible_jacobian[0, 0] = 1.0

    monkeypatch.setattr(v25, "v27_support_tensors_for_source_target", lambda **_: support)
    monkeypatch.setattr(v25, "v32_support_behavior_margin_tensors", lambda: {
        "by_behavior": {},
        "counts_by_behavior": {},
        "tensor_hash": "e" * 64,
    })
    monkeypatch.setattr(
        v25,
        "v27_subject_logits_for_inputs",
        lambda flat, inputs: flat[1].expand(int(inputs.shape[0])),
    )

    def fake_margins(*, weights: torch.Tensor, tournament_tensors: Mapping[str, Any]):
        return {
            "has_majority": weights[1] - 0.30,
            "mountain_pattern": weights[1] - 0.20,
            "sorted_ascending": weights[1] - 0.10,
            "sorted_descending": weights[1],
        }

    monkeypatch.setattr(v25, "v32_support_behavior_margins", fake_margins)
    config = {
        **v25.build_v37_projected_support_optimizer_config_grid()[0],
        "projected_optimizer_epochs": 1,
        "projected_optimizer_lr": 0.2,
        "projection_strength": 1.0,
        "trust_norm_cap": 1.0,
    }
    result = v25.solve_v37_projected_support_optimizer_edit(
        compatible_gradient=compatible_gradient,
        compatible_jacobian=compatible_jacobian,
        coordinate_hash="a" * 64,
        selected_coordinates=[0, 1],
        config=config,
        source_weights=source,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
    )

    edited = source + result["delta"]
    target_logits = v25.v27_subject_logits_for_inputs(edited, support["target_inputs"])
    conflict_logits = v25.v27_subject_logits_for_inputs(edited, support["conflict_inputs"])
    compatible_logits = v25.v27_subject_logits_for_inputs(
        edited,
        support["compatible_inputs"],
    )
    target_labels = support["target_labels"].to(dtype=torch.float32)
    target_multiplier = v25.v31_support_hardness_multiplier(
        source_target_logits=torch.zeros(1),
        target_labels=target_labels,
        target_margin_floor=float(config["target_margin_floor"]),
        hard_target_margin_weight=float(config["hard_target_margin_weight"]),
    )
    target_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        target_logits,
        target_labels,
    )
    target_margin_hinge = v25.v30_target_margin_hinge_loss(
        logits=target_logits,
        labels=target_labels,
        margin_floor=float(config["target_margin_floor"]),
    )
    conflict_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        conflict_logits,
        support["conflict_target_labels"].to(dtype=torch.float32),
    )
    compatible_mse = torch.nn.functional.mse_loss(
        compatible_logits,
        support["compatible_source_logits"].to(dtype=torch.float32),
    )
    compatible_orthogonal_loss = v25.v31_compatible_gradient_orthogonal_loss(
        delta=result["delta"],
        g_compatible=compatible_gradient,
    )
    tournament = v25.v32_support_tournament_margin_loss(
        margins=v25.v32_support_behavior_margins(
            weights=edited,
            tournament_tensors={},
        ),
        target_behavior="sorted_descending",
        tournament_margin_floor=float(config["tournament_margin_floor"]),
    )
    expected_loss = (
        target_multiplier * (
            v25.V29_TARGET_BCE_WEIGHT * target_bce
            + v25.V30_TARGET_MARGIN_WEIGHT * target_margin_hinge
        )
        + v25.V29_CONFLICT_BCE_WEIGHT * conflict_bce
        + v25.V29_COMPATIBLE_PROBE_WEIGHT * compatible_mse
        + float(config["extra_compatible_weight"]) * compatible_mse
        + float(config["compatible_orthogonal_weight"]) * compatible_orthogonal_loss
        + float(config["tournament_margin_weight"]) * tournament["loss"]
        + v25.V29_DELTA_L2_WEIGHT * torch.mean(result["delta"].pow(2))
    )

    assert result["delta"][1] > 0.0
    assert result["audit"]["final_loss"] == pytest.approx(
        float(expected_loss.item()),
        rel=1e-5,
        abs=1e-6,
    )


def test_v42_dual_update_and_residual_use_post_step_projected_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    support = {
        "compatible_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "compatible_source_logits": torch.zeros(1, dtype=torch.float32),
        "conflict_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "conflict_target_labels": torch.ones(1, dtype=torch.float32),
        "target_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "target_labels": torch.ones(1, dtype=torch.float32),
    }
    compatible_gradient = torch.ones(v25.SOURCE_WEIGHT_DIM)
    compatible_jacobian = torch.zeros(1, v25.SOURCE_WEIGHT_DIM)

    monkeypatch.setattr(v25, "v27_support_tensors_for_source_target", lambda **_: support)
    monkeypatch.setattr(v25, "v32_support_behavior_margin_tensors", lambda: {
        "by_behavior": {},
        "counts_by_behavior": {},
        "tensor_hash": "e" * 64,
    })
    monkeypatch.setattr(
        v25,
        "v27_subject_logits_for_inputs",
        lambda flat, inputs: flat[1].expand(int(inputs.shape[0])),
    )
    monkeypatch.setattr(v25, "v32_support_behavior_margins", lambda **kwargs: {
        "has_majority": kwargs["weights"][1] - 0.30,
        "mountain_pattern": kwargs["weights"][1] - 0.20,
        "sorted_ascending": kwargs["weights"][1] - 0.10,
        "sorted_descending": kwargs["weights"][1],
    })

    config = {
        **v25.build_v42_compatible_dual_frontier_config_grid()[0],
        "compatible_augmented_weight": 0.0,
        "compatible_dual_initial": 0.0,
        "compatible_dual_lr": 0.5,
        "compatible_dual_max": 100.0,
        "compatible_mse_budget": 0.0,
        "projected_optimizer_epochs": 1,
        "projected_optimizer_lr": 0.2,
        "projection_strength": 0.0,
        "trust_norm_cap": 1.0,
    }
    result = v25.solve_v37_projected_support_optimizer_edit(
        compatible_gradient=compatible_gradient,
        compatible_jacobian=compatible_jacobian,
        coordinate_hash="a" * 64,
        selected_coordinates=[0, 1],
        config=config,
        source_weights=source,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
    )

    edited = source + result["delta"]
    compatible_logits = v25.v27_subject_logits_for_inputs(
        edited,
        support["compatible_inputs"],
    )
    compatible_mse = torch.nn.functional.mse_loss(
        compatible_logits,
        support["compatible_source_logits"].to(dtype=torch.float32),
    )
    expected_residual = float(compatible_mse.item())

    assert result["audit"]["compatible_constraint_residual"] == pytest.approx(
        expected_residual,
        rel=1e-5,
        abs=1e-6,
    )
    assert result["audit"]["compatible_dual_lambda"] == pytest.approx(
        0.5 * expected_residual,
        rel=1e-5,
        abs=1e-6,
    )
    assert result["audit"]["compatible_dual_update_source"] == "post_step_projected_delta"


def test_v38_projected_optimizer_event_prefix_labels_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    source = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    support = {
        "compatible_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "compatible_source_logits": torch.zeros(1, dtype=torch.float32),
        "conflict_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "conflict_target_labels": torch.ones(1, dtype=torch.float32),
        "target_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "target_labels": torch.ones(1, dtype=torch.float32),
    }
    compatible_jacobian = torch.zeros(1, v25.SOURCE_WEIGHT_DIM)
    compatible_jacobian[0, 0] = 1.0

    monkeypatch.setattr(v25, "v27_support_tensors_for_source_target", lambda **_: support)
    monkeypatch.setattr(v25, "v32_support_behavior_margin_tensors", lambda: {
        "by_behavior": {},
        "counts_by_behavior": {},
        "tensor_hash": "e" * 64,
    })
    monkeypatch.setattr(
        v25,
        "v27_subject_logits_for_inputs",
        lambda flat, inputs: flat[1].expand(int(inputs.shape[0])),
    )
    monkeypatch.setattr(v25, "v32_support_behavior_margins", lambda **kwargs: {
        "has_majority": kwargs["weights"][1] - 0.30,
        "mountain_pattern": kwargs["weights"][1] - 0.20,
        "sorted_ascending": kwargs["weights"][1] - 0.10,
        "sorted_descending": kwargs["weights"][1],
    })
    monkeypatch.setattr(
        v25,
        "record_progress_event",
        lambda *_args, event, **_kwargs: events.append(event),
    )

    v25.solve_v37_projected_support_optimizer_edit(
        compatible_gradient=torch.ones(v25.SOURCE_WEIGHT_DIM),
        compatible_jacobian=compatible_jacobian,
        coordinate_hash="a" * 64,
        selected_coordinates=[0, 1],
        config={
            **v25.build_v38_compatible_mse_gated_projected_optimizer_config_grid()[0],
            "projected_optimizer_epochs": 2,
            "projected_optimizer_lr": 0.2,
            "projection_strength": 1.0,
            "trust_norm_cap": 1.0,
        },
        source_weights=source,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        progress_log_path=Path("progress.jsonl"),
        selected_config_hash="b" * 64,
    )

    assert "v38_compatible_gated_optimizer_progress" in events
    assert "v38_compatible_gated_optimizer_completed" in events


def test_v38_score_defaults_match_v37_formula() -> None:
    kwargs = {
        "support_target_margin": -0.1,
        "support_tournament_margin": -0.2,
        "support_compatible_mse": 12.0,
        "loss": 30.0,
        "target_margin_floor": 0.05,
        "tournament_margin_floor": 0.15,
    }
    expected = (
        100.0 * (0.05 - (-0.1))
        + 50.0 * (0.15 - (-0.2))
        + 12.0
        + 0.01 * 30.0
    )

    assert v25.v38_projected_optimizer_support_score(
        **kwargs,
        compatible_mse_gate=float("inf"),
        compatible_gate_weight=0.0,
    ) == pytest.approx(expected)


def test_v38_optimizer_score_penalizes_compatible_mse_gate() -> None:
    low = v25.v38_projected_optimizer_support_score(
        support_target_margin=0.5,
        support_tournament_margin=0.4,
        support_compatible_mse=4.0,
        loss=10.0,
        target_margin_floor=0.05,
        tournament_margin_floor=0.15,
        compatible_mse_gate=5.0,
        compatible_gate_weight=1.5,
    )
    high = v25.v38_projected_optimizer_support_score(
        support_target_margin=0.5,
        support_tournament_margin=0.4,
        support_compatible_mse=12.0,
        loss=1.0,
        target_margin_floor=0.05,
        tournament_margin_floor=0.15,
        compatible_mse_gate=5.0,
        compatible_gate_weight=1.5,
    )

    assert high > low


def test_v38_alpha_selection_requires_compatible_mse_gate() -> None:
    selected = v25.select_v38_compatible_gated_alpha_candidate(
        candidates=[
            {
                "alpha": 1.0,
                "support_compatible_mse": 20.0,
                "support_runner_margin": 0.0,
                "support_target_margin": 1.0,
                "support_tournament_margin": 1.0,
            },
            {
                "alpha": 0.5,
                "support_compatible_mse": 4.0,
                "support_runner_margin": 0.0,
                "support_target_margin": 0.4,
                "support_tournament_margin": 0.3,
            },
        ],
        alpha_target_margin_floor=0.05,
        alpha_tournament_margin_floor=0.0,
        alpha_compatible_mse_gate=5.0,
        fallback_target_penalty=10.0,
        fallback_tournament_penalty=5.0,
        fallback_compatible_penalty=2.0,
    )

    assert selected["alpha"] == 0.5
    assert selected["compatible_gate_pass"] is True
    assert selected["eligible"] is True
    assert selected["selection_mode"] == "eligible_min_compatible_mse"


def test_v39_alpha_selection_prefers_target_movement_over_noop_fallback() -> None:
    selected = v25.select_v39_target_feasible_lexicographic_alpha_candidate(
        candidates=[
            {
                "alpha": 0.0,
                "support_compatible_mse": 0.0,
                "support_runner_margin": 0.9,
                "support_target_margin": -0.6,
                "support_tournament_margin": -1.5,
            },
            {
                "alpha": 0.75,
                "support_compatible_mse": 12.0,
                "support_runner_margin": -0.1,
                "support_target_margin": 0.2,
                "support_tournament_margin": 0.1,
            },
        ],
        alpha_target_margin_floor=0.05,
        alpha_tournament_margin_floor=0.0,
        alpha_compatible_mse_soft_gate=10.0,
    )

    assert selected["alpha"] == 0.75
    assert selected["selection_mode"] == "target_feasible_min_compatible_mse"
    assert selected["target_feasible"] is True
    assert selected["compatible_gap"] == pytest.approx(2.0)


def test_v39_alpha_fallback_prefers_smaller_target_gap_over_noop_mse() -> None:
    selected = v25.select_v39_target_feasible_lexicographic_alpha_candidate(
        candidates=[
            {
                "alpha": 0.0,
                "support_compatible_mse": 0.0,
                "support_runner_margin": 0.9,
                "support_target_margin": -0.8,
                "support_tournament_margin": -1.4,
            },
            {
                "alpha": 0.75,
                "support_compatible_mse": 12.0,
                "support_runner_margin": -0.1,
                "support_target_margin": -0.1,
                "support_tournament_margin": -0.2,
            },
        ],
        alpha_target_margin_floor=0.05,
        alpha_tournament_margin_floor=0.0,
        alpha_compatible_mse_soft_gate=10.0,
    )

    assert selected["alpha"] == 0.75
    assert selected["selection_mode"] == "fallback_target_feasible_lexicographic"
    assert selected["target_feasible"] is False
    assert selected["target_rank_score"] == pytest.approx(0.35)
    assert selected["support_compatible_mse"] == 12.0


def test_v39_alpha_selection_uses_locality_among_target_feasible_candidates() -> None:
    selected = v25.select_v39_target_feasible_lexicographic_alpha_candidate(
        candidates=[
            {
                "alpha": 1.0,
                "support_compatible_mse": 14.0,
                "support_runner_margin": 0.0,
                "support_target_margin": 0.6,
                "support_tournament_margin": 0.5,
            },
            {
                "alpha": 0.5,
                "support_compatible_mse": 4.0,
                "support_runner_margin": 0.0,
                "support_target_margin": 0.3,
                "support_tournament_margin": 0.2,
            },
        ],
        alpha_target_margin_floor=0.05,
        alpha_tournament_margin_floor=0.0,
        alpha_compatible_mse_soft_gate=10.0,
    )

    assert selected["alpha"] == 0.5
    assert selected["selection_mode"] == "target_feasible_min_compatible_mse"
    assert selected["eligible_count"] == 2
    assert selected["target_feasible"] is True


def test_v40_alpha_selection_uses_locality_within_target_tolerance() -> None:
    selected = v25.select_v40_target_tolerance_locality_budget_alpha_candidate(
        candidates=[
            {
                "alpha": 1.0,
                "support_compatible_mse": 30.0,
                "support_runner_margin": 0.0,
                "support_target_margin": -0.05,
                "support_tournament_margin": -0.10,
            },
            {
                "alpha": 0.5,
                "support_compatible_mse": 4.0,
                "support_runner_margin": 0.0,
                "support_target_margin": -0.07,
                "support_tournament_margin": -0.11,
            },
            {
                "alpha": 0.25,
                "support_compatible_mse": 0.5,
                "support_runner_margin": 0.0,
                "support_target_margin": -0.30,
                "support_tournament_margin": -0.60,
            },
        ],
        alpha_target_margin_floor=0.05,
        alpha_tournament_margin_floor=0.0,
        alpha_compatible_mse_soft_gate=10.0,
        target_rank_score_tolerance=0.05,
    )

    assert selected["alpha"] == 0.5
    assert selected["selection_mode"] == "target_tolerance_min_compatible_mse"
    assert selected["within_target_tolerance_count"] == 2
    assert selected["best_target_rank_score"] == pytest.approx(0.2)


def test_v40_alpha_selection_excludes_noop_outside_target_tolerance() -> None:
    selected = v25.select_v40_target_tolerance_locality_budget_alpha_candidate(
        candidates=[
            {
                "alpha": 0.0,
                "support_compatible_mse": 0.0,
                "support_runner_margin": 0.9,
                "support_target_margin": -0.8,
                "support_tournament_margin": -1.4,
            },
            {
                "alpha": 0.75,
                "support_compatible_mse": 12.0,
                "support_runner_margin": -0.1,
                "support_target_margin": -0.1,
                "support_tournament_margin": -0.2,
            },
        ],
        alpha_target_margin_floor=0.05,
        alpha_tournament_margin_floor=0.0,
        alpha_compatible_mse_soft_gate=10.0,
        target_rank_score_tolerance=0.05,
    )

    assert selected["alpha"] == 0.75
    assert selected["selection_mode"] == "target_tolerance_min_compatible_mse"
    assert selected["target_feasible"] is False
    assert selected["within_target_tolerance_count"] == 1
    assert selected["support_compatible_mse"] == 12.0


def test_v41_frontier_selection_prefers_locality_within_target_tolerance() -> None:
    selected = v25.select_v41_trajectory_frontier_candidate(
        candidates=[
            {
                "epoch": 4,
                "loss": 20.0,
                "preservation_energy_ratio": 0.5,
                "projected_delta_norm": 1.0,
                "support_compatible_mse": 30.0,
                "support_runner_margin": 0.0,
                "support_target_margin": -0.05,
                "support_tournament_margin": -0.10,
            },
            {
                "epoch": 8,
                "loss": 25.0,
                "preservation_energy_ratio": 0.5,
                "projected_delta_norm": 0.8,
                "support_compatible_mse": 4.0,
                "support_runner_margin": 0.0,
                "support_target_margin": -0.07,
                "support_tournament_margin": -0.11,
            },
        ],
        compatible_mse_soft_gate=10.0,
        target_margin_floor=0.05,
        target_rank_score_tolerance=0.05,
        tournament_margin_floor=0.0,
    )

    assert selected["epoch"] == 8
    assert selected["selection_mode"] == "frontier_target_tolerance_min_compatible_mse"
    assert selected["within_target_tolerance_count"] == 2
    assert selected["best_target_rank_score"] == pytest.approx(0.2)


def test_v41_frontier_selection_excludes_noop_outside_target_tolerance() -> None:
    selected = v25.select_v41_trajectory_frontier_candidate(
        candidates=[
            {
                "epoch": 1,
                "loss": 0.0,
                "preservation_energy_ratio": 0.0,
                "projected_delta_norm": 0.0,
                "support_compatible_mse": 0.0,
                "support_runner_margin": 0.0,
                "support_target_margin": -0.8,
                "support_tournament_margin": -1.4,
            },
            {
                "epoch": 5,
                "loss": 20.0,
                "preservation_energy_ratio": 0.5,
                "projected_delta_norm": 0.9,
                "support_compatible_mse": 12.0,
                "support_runner_margin": -0.1,
                "support_target_margin": -0.1,
                "support_tournament_margin": -0.2,
            },
        ],
        compatible_mse_soft_gate=10.0,
        target_margin_floor=0.05,
        target_rank_score_tolerance=0.05,
        tournament_margin_floor=0.0,
    )

    assert selected["epoch"] == 5
    assert selected["selection_mode"] == "frontier_target_tolerance_min_compatible_mse"
    assert selected["target_feasible"] is False
    assert selected["within_target_tolerance_count"] == 1
    assert selected["support_compatible_mse"] == 12.0


def test_v42_compatible_constraint_penalty_zero_below_budget() -> None:
    result = v25.v42_compatible_constraint_terms(
        support_compatible_mse=torch.tensor(8.0),
        compatible_mse_budget=10.0,
        compatible_dual_lambda=3.0,
        compatible_augmented_weight=2.0,
    )

    assert result["compatible_constraint_residual"].item() == pytest.approx(0.0)
    assert result["compatible_constraint_penalty"].item() == pytest.approx(0.0)


def test_v42_compatible_constraint_penalty_positive_above_budget() -> None:
    result = v25.v42_compatible_constraint_terms(
        support_compatible_mse=torch.tensor(13.0),
        compatible_mse_budget=10.0,
        compatible_dual_lambda=2.0,
        compatible_augmented_weight=0.5,
    )

    assert result["compatible_constraint_residual"].item() == pytest.approx(3.0)
    assert result["compatible_constraint_penalty"].item() == pytest.approx(10.5)


def test_v42_frontier_selection_prefers_target_and_compatible_feasible() -> None:
    selected = v25.select_v42_compatible_dual_frontier_candidate(
        candidates=[
            {
                "epoch": 4,
                "loss": 20.0,
                "preservation_energy_ratio": 0.5,
                "projected_delta_norm": 1.0,
                "support_compatible_mse": 30.0,
                "support_runner_margin": 0.0,
                "support_target_margin": 0.20,
                "support_tournament_margin": 0.10,
                "compatible_constraint_residual": 20.0,
                "compatible_dual_lambda": 4.0,
            },
            {
                "epoch": 8,
                "loss": 25.0,
                "preservation_energy_ratio": 0.5,
                "projected_delta_norm": 0.8,
                "support_compatible_mse": 8.0,
                "support_runner_margin": 0.0,
                "support_target_margin": 0.08,
                "support_tournament_margin": 0.04,
                "compatible_constraint_residual": 0.0,
                "compatible_dual_lambda": 4.0,
            },
        ],
        compatible_mse_budget=10.0,
        target_margin_floor=0.05,
        target_rank_score_tolerance=0.05,
        tournament_margin_floor=0.0,
    )

    assert selected["epoch"] == 8
    assert selected["compatible_constraint_feasible"] is True
    assert selected["target_feasible"] is True
    assert selected["selection_mode"] == "frontier_target_and_compatible_feasible"


def test_v42_frontier_selection_reports_no_localized_candidate_when_budget_fails() -> None:
    selected = v25.select_v42_compatible_dual_frontier_candidate(
        candidates=[
            {
                "epoch": 5,
                "loss": 20.0,
                "preservation_energy_ratio": 0.5,
                "projected_delta_norm": 0.9,
                "support_compatible_mse": 12.0,
                "support_runner_margin": -0.1,
                "support_target_margin": 0.10,
                "support_tournament_margin": 0.10,
                "compatible_constraint_residual": 2.0,
                "compatible_dual_lambda": 6.0,
            },
        ],
        compatible_mse_budget=10.0,
        target_margin_floor=0.05,
        target_rank_score_tolerance=0.05,
        tournament_margin_floor=0.0,
    )

    assert selected["epoch"] == 5
    assert selected["target_feasible"] is True
    assert selected["compatible_constraint_feasible"] is False
    assert selected["localized_feasible_count"] == 0
    assert selected["selection_mode"] == "frontier_target_feasible_min_compatible_residual"


def test_v42_frontier_redaction_omits_raw_fields() -> None:
    forbidden = {
        "basis": [1.0],
        "compatible_jacobian": [1.0],
        "final_subjects": ["sealed"],
        "final_subjects_path": "/sealed/final_subjects.json",
        "gradient": [1.0],
        "logits": [1.0],
        "projected_delta": [1.0],
        "raw_delta": [3.0],
        "selected_coordinates": [0],
        "sequence": [1, 0, 1],
        "subject_id": "raw-subject",
        "support_examples": [[1, 0, 1]],
        "weights": [1.0, 2.0],
    }
    redacted = v25.redact_v42_compatible_dual_frontier_progress_event({
        **forbidden,
        "candidate_index": 1,
        "compatible_constraint_feasible": True,
        "compatible_constraint_residual": 0.0,
        "compatible_dual_lambda": 2.5,
        "compatible_gap": 0.0,
        "compatible_mse_budget": 10.0,
        "frontier_candidate_count": 3,
        "localized_feasible_count": 1,
        "loss": 4.0,
        "record_id_hash": "a" * 64,
        "selected_config_hash": "b" * 64,
        "selection_mode": "frontier_target_and_compatible_feasible",
        "support_compatible_mse": 8.0,
        "support_target_margin": 0.1,
        "support_tournament_margin": 0.2,
        "target_feasible": True,
        "target_rank_score": 0.0,
        "trajectory_frontier_selected_epoch": 5,
    })

    assert redacted["compatible_constraint_feasible"] is True
    assert redacted["selection_mode"] == "frontier_target_and_compatible_feasible"
    for key in forbidden:
        assert key not in redacted


def test_v36_matched_edit_projects_before_support_alpha_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    support = {
        "compatible_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "compatible_source_logits": torch.zeros(1, dtype=torch.float32),
        "conflict_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "conflict_target_labels": torch.ones(1, dtype=torch.float32),
        "target_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "target_labels": torch.ones(1, dtype=torch.float32),
    }
    base_delta = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    base_delta[0] = 1.0
    base_delta[1] = 0.5
    compatible_jacobian = torch.zeros(1, v25.SOURCE_WEIGHT_DIM)
    compatible_jacobian[0, 0] = 1.0

    monkeypatch.setattr(v25, "v29_sparse_support_gradients", lambda **_kwargs: {
        "g_compatible": torch.ones(v25.SOURCE_WEIGHT_DIM),
        "g_conflict": torch.ones(v25.SOURCE_WEIGHT_DIM),
        "g_target": torch.ones(v25.SOURCE_WEIGHT_DIM),
        "support_split_counts": {"compatible": 1, "conflict": 1, "target": 1},
    })
    monkeypatch.setattr(
        v25,
        "v28_anchor_gradients_and_compatible_jacobian",
        lambda **_kwargs: {
            "compatible_jacobian": compatible_jacobian,
            "g_compatible": torch.ones(v25.SOURCE_WEIGHT_DIM),
            "g_conflict": torch.ones(v25.SOURCE_WEIGHT_DIM),
            "g_target": torch.ones(v25.SOURCE_WEIGHT_DIM),
            "support_split_counts": {"compatible": 1, "conflict": 1, "target": 1},
        },
    )
    monkeypatch.setattr(
        v25,
        "select_v31_sign_coherent_sparse_coordinates",
        lambda **_kwargs: [0, 1],
    )
    monkeypatch.setattr(v25, "v27_support_tensors_for_source_target", lambda **_: support)
    monkeypatch.setattr(v25, "v32_support_behavior_margin_tensors", lambda: {
        "by_behavior": {},
        "counts_by_behavior": {},
        "tensor_hash": "e" * 64,
    })
    monkeypatch.setattr(
        v25,
        "solve_v32_support_tournament_sparse_edit",
        lambda **_kwargs: {
            "audit": {"delta_sha256": "1" * 64, "support_scalar_losses": {"loss": 0.1}},
            "delta": base_delta,
        },
    )
    monkeypatch.setattr(
        v25,
        "v27_subject_logits_for_inputs",
        lambda flat, inputs: flat[1].expand(int(inputs.shape[0])),
    )

    def fake_margins(*, weights: torch.Tensor, tournament_tensors: Mapping[str, Any]):
        target_margin = weights[1] * 0.4
        return {
            "has_majority": target_margin - 0.10,
            "mountain_pattern": target_margin - 0.20,
            "sorted_ascending": target_margin - 0.30,
            "sorted_descending": target_margin,
        }

    def fake_control_record(**kwargs: Any) -> dict[str, Any]:
        captured["delta"] = kwargs["delta"].detach().clone()
        captured["metadata"] = dict(kwargs["metadata"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": float(torch.linalg.norm(kwargs["delta"]).item()),
            "editor": {
                **dict(kwargs["metadata"]),
                "delta_sha256": "9" * 64,
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.0,
            "target_margin": 0.5,
        }

    monkeypatch.setattr(v25, "v32_support_behavior_margins", fake_margins)
    monkeypatch.setattr(v25, "control_record_for_delta", fake_control_record)
    monkeypatch.setattr(
        v25,
        "record_progress_event",
        lambda *_args, event, extra, **_kwargs: events.append({
            "event": event,
            "extra": dict(extra),
        }),
    )

    result = v25.evaluate_v25_compatible_nullspace_projected_matched_edit(
        subject={
            "pattern": "sorted_ascending",
            "subject_id": "hidden",
            "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
        },
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        config={
            **v25.build_v36_compatible_nullspace_projection_config_grid()[1],
            "alpha_candidates": [1.0],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=1.25,
        selected_config_hash="c" * 64,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        script_sha256="script",
        progress_log_path=Path("progress.jsonl"),
        record_id_hash="d" * 64,
    )

    assert torch.isclose(captured["delta"][0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(captured["delta"][1], torch.tensor(0.5), atol=1e-6)
    assert result["editor"]["matched_edit_source"] == (
        v25.V36_COMPATIBLE_NULLSPACE_MATCHED_EDIT_SOURCE
    )
    assert captured["metadata"]["projection_audit"]["preservation_energy_ratio"] < 1e-5
    assert any(
        event["event"] == "v36_compatible_nullspace_projection_completed"
        for event in events
    )


def test_v36_dispatch_uses_projected_matched_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_matched(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": v25.V36_COMPATIBLE_NULLSPACE_MATCHED_EDIT_SOURCE,
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(
        v25,
        "evaluate_v25_compatible_nullspace_projected_matched_edit",
        fake_matched,
    )
    monkeypatch.setattr(
        v25,
        "build_v25_native_controls",
        lambda **kwargs: called.setdefault("control_config", dict(kwargs["config"])) or [],
    )
    monkeypatch.setattr(v25, "build_v25_proof_record", lambda **kwargs: {
        "controls": [],
        "matched": dict(kwargs["matched"]),
        "summary": {
            "individual_all_gates_passed": True,
            "pareto_undominated": True,
            "target_prediction_pass": True,
        },
    })

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            **v25.build_v36_compatible_nullspace_projection_config_grid()[0],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "e" * 64},
        selected_config_hash="c" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == (
        v25.V36_COMPATIBLE_NULLSPACE_MATCHED_EDIT_SOURCE
    )
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        v25.V36_COMPATIBLE_NULLSPACE_MATCHED_EDIT_SOURCE
    )


def test_v37_dispatch_uses_projected_optimizer_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_matched(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": v25.V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE,
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(
        v25,
        "evaluate_v25_projected_support_optimizer_matched_edit",
        fake_matched,
    )
    monkeypatch.setattr(
        v25,
        "build_v25_native_controls",
        lambda **kwargs: called.setdefault("control_config", dict(kwargs["config"])) or [],
    )
    monkeypatch.setattr(v25, "build_v25_proof_record", lambda **kwargs: {
        "controls": [],
        "matched": dict(kwargs["matched"]),
        "summary": {
            "individual_all_gates_passed": True,
            "pareto_undominated": True,
            "target_prediction_pass": True,
        },
    })

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            **v25.build_v37_projected_support_optimizer_config_grid()[0],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "e" * 64},
        selected_config_hash="c" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == (
        v25.V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE
    )
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        v25.V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE
    )


def test_v38_dispatch_uses_compatible_gated_optimizer_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_matched(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": v25.V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE,
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(
        v25,
        "evaluate_v25_compatible_gated_projected_optimizer_matched_edit",
        fake_matched,
    )
    monkeypatch.setattr(
        v25,
        "build_v25_native_controls",
        lambda **kwargs: called.setdefault("control_config", dict(kwargs["config"])) or [],
    )
    monkeypatch.setattr(v25, "build_v25_proof_record", lambda **kwargs: {
        "controls": [],
        "matched": dict(kwargs["matched"]),
        "summary": {
            "individual_all_gates_passed": True,
            "pareto_undominated": True,
            "target_prediction_pass": True,
        },
    })

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            **v25.build_v38_compatible_mse_gated_projected_optimizer_config_grid()[0],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "e" * 64},
        selected_config_hash="c" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == (
        v25.V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE
    )
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        v25.V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE
    )


def test_v39_dispatch_uses_target_feasible_lexicographic_optimizer_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_matched(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": (
                    v25.V39_TARGET_FEASIBLE_LEXICOGRAPHIC_MATCHED_EDIT_SOURCE
                ),
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(
        v25,
        "evaluate_v25_target_feasible_lexicographic_projected_optimizer_matched_edit",
        fake_matched,
    )
    monkeypatch.setattr(
        v25,
        "build_v25_native_controls",
        lambda **kwargs: called.setdefault("control_config", dict(kwargs["config"])) or [],
    )
    monkeypatch.setattr(v25, "build_v25_proof_record", lambda **kwargs: {
        "controls": [],
        "matched": dict(kwargs["matched"]),
        "summary": {
            "individual_all_gates_passed": True,
            "pareto_undominated": True,
            "target_prediction_pass": True,
        },
    })

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            **v25.build_v39_target_feasible_lexicographic_optimizer_config_grid()[0],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "e" * 64},
        selected_config_hash="c" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == (
        v25.V39_TARGET_FEASIBLE_LEXICOGRAPHIC_MATCHED_EDIT_SOURCE
    )
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        v25.V39_TARGET_FEASIBLE_LEXICOGRAPHIC_MATCHED_EDIT_SOURCE
    )


def test_v40_dispatch_uses_target_tolerance_locality_budget_optimizer_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_matched(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": (
                    v25.V40_TARGET_TOLERANCE_LOCALITY_BUDGET_MATCHED_EDIT_SOURCE
                ),
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(
        v25,
        "evaluate_v25_target_tolerance_locality_budget_matched_edit",
        fake_matched,
    )
    monkeypatch.setattr(
        v25,
        "build_v25_native_controls",
        lambda **kwargs: called.setdefault("control_config", dict(kwargs["config"])) or [],
    )
    monkeypatch.setattr(v25, "build_v25_proof_record", lambda **kwargs: {
        "controls": [],
        "matched": dict(kwargs["matched"]),
        "summary": {
            "individual_all_gates_passed": True,
            "pareto_undominated": True,
            "target_prediction_pass": True,
        },
    })

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            **v25.build_v40_target_tolerance_locality_budget_config_grid()[0],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "e" * 64},
        selected_config_hash="c" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == (
        v25.V40_TARGET_TOLERANCE_LOCALITY_BUDGET_MATCHED_EDIT_SOURCE
    )
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        v25.V40_TARGET_TOLERANCE_LOCALITY_BUDGET_MATCHED_EDIT_SOURCE
    )


def test_v41_dispatch_uses_trajectory_frontier_optimizer_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_matched(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": v25.V41_TRAJECTORY_FRONTIER_MATCHED_EDIT_SOURCE,
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(
        v25,
        "evaluate_v25_trajectory_frontier_matched_edit",
        fake_matched,
    )
    monkeypatch.setattr(
        v25,
        "build_v25_native_controls",
        lambda **kwargs: called.setdefault("control_config", dict(kwargs["config"])) or [],
    )
    monkeypatch.setattr(v25, "build_v25_proof_record", lambda **kwargs: {
        "controls": [],
        "matched": dict(kwargs["matched"]),
        "summary": {
            "individual_all_gates_passed": True,
            "pareto_undominated": True,
            "target_prediction_pass": True,
        },
    })

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            **v25.build_v41_trajectory_frontier_config_grid()[0],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "e" * 64},
        selected_config_hash="c" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == (
        v25.V41_TRAJECTORY_FRONTIER_MATCHED_EDIT_SOURCE
    )
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        v25.V41_TRAJECTORY_FRONTIER_MATCHED_EDIT_SOURCE
    )


def test_v42_dispatch_uses_compatible_dual_frontier_optimizer_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_matched(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": (
                    v25.V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
                ),
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(
        v25,
        "evaluate_v25_compatible_dual_frontier_matched_edit",
        fake_matched,
    )
    monkeypatch.setattr(
        v25,
        "build_v25_native_controls",
        lambda **kwargs: called.setdefault("control_config", dict(kwargs["config"])) or [],
    )
    monkeypatch.setattr(v25, "build_v25_proof_record", lambda **kwargs: {
        "controls": [],
        "matched": dict(kwargs["matched"]),
        "summary": {
            "individual_all_gates_passed": True,
            "pareto_undominated": True,
            "target_prediction_pass": True,
        },
    })

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            **v25.build_v42_compatible_dual_frontier_config_grid()[0],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "e" * 64},
        selected_config_hash="c" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == (
        v25.V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
    )
    assert called["matched_config"]["v42_compatible_dual_enabled"] is True
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        v25.V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
    )


def test_v37_matched_edit_metadata_uses_projected_optimizer_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    support = {
        "compatible_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "compatible_source_logits": torch.zeros(1, dtype=torch.float32),
        "conflict_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "conflict_target_labels": torch.ones(1, dtype=torch.float32),
        "target_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "target_labels": torch.ones(1, dtype=torch.float32),
    }

    monkeypatch.setattr(
        v25,
        "v28_anchor_gradients_and_compatible_jacobian",
        lambda **_kwargs: {
            "compatible_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
            "g_compatible": torch.ones(v25.SOURCE_WEIGHT_DIM),
            "g_conflict": torch.ones(v25.SOURCE_WEIGHT_DIM),
            "g_target": torch.ones(v25.SOURCE_WEIGHT_DIM),
            "support_split_counts": {"compatible": 1, "conflict": 1, "target": 1},
        },
    )
    monkeypatch.setattr(
        v25,
        "select_v31_sign_coherent_sparse_coordinates",
        lambda **_kwargs: [0, 1],
    )
    monkeypatch.setattr(
        v25,
        "solve_v37_projected_support_optimizer_edit",
        lambda **_kwargs: {
            "audit": {
                "coordinate_hash": "1" * 64,
                "final_loss": 0.1,
                "jacobian_row_count": 1,
                "optimization_steps": 3,
                "optimizer_audit_hash": "2" * 64,
                "preservation_energy_ratio": 0.0,
                "preserve_rank": 0,
                "projected_delta_norm": 0.4,
                "support_compatible_mse": 0.0,
                "support_runner_margin": 0.1,
                "support_target_margin": 0.4,
                "support_tournament_margin": 0.3,
                "support_tournament_tensor_hash": "3" * 64,
            },
            "delta": torch.tensor([0.0, 0.4, *([0.0] * (v25.SOURCE_WEIGHT_DIM - 2))]),
        },
    )
    monkeypatch.setattr(v25, "v27_support_tensors_for_source_target", lambda **_: support)
    monkeypatch.setattr(v25, "v32_support_behavior_margin_tensors", lambda: {
        "by_behavior": {},
        "counts_by_behavior": {},
        "tensor_hash": "4" * 64,
    })
    monkeypatch.setattr(
        v25,
        "v27_subject_logits_for_inputs",
        lambda flat, inputs: flat[1].expand(int(inputs.shape[0])),
    )
    monkeypatch.setattr(v25, "v32_support_behavior_margins", lambda **kwargs: {
        "has_majority": kwargs["weights"][1] - 0.30,
        "mountain_pattern": kwargs["weights"][1] - 0.20,
        "sorted_ascending": kwargs["weights"][1] - 0.10,
        "sorted_descending": kwargs["weights"][1],
    })

    def fake_control_record(**kwargs: Any) -> dict[str, Any]:
        captured["metadata"] = dict(kwargs["metadata"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": float(torch.linalg.norm(kwargs["delta"]).item()),
            "editor": {
                **dict(kwargs["metadata"]),
                "delta_sha256": "9" * 64,
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.0,
            "target_margin": 0.5,
        }

    monkeypatch.setattr(v25, "control_record_for_delta", fake_control_record)

    result = v25.evaluate_v25_projected_support_optimizer_matched_edit(
        subject={
            "pattern": "sorted_ascending",
            "subject_id": "hidden",
            "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
        },
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        config={
            **v25.build_v37_projected_support_optimizer_config_grid()[0],
            "alpha_candidates": [1.0],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=1.0,
        selected_config_hash="c" * 64,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        script_sha256="script",
    )

    boundary = captured["metadata"]["optimization_boundary"]
    assert boundary["support_optimizer"] == "projected_target_tournament_loss"
    assert boundary["support_projection"] == "compatible_logit_jacobian_nullspace"
    assert captured["metadata"]["matched_edit_source"] == (
        v25.V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE
    )
    assert "projected_optimizer_provenance_hash" in captured["metadata"]
    assert result["editor"]["matched_edit_source"] == (
        v25.V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE
    )


def test_v42_matched_edit_metadata_stamps_dual_frontier_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    subject = {
        "pattern": "sorted_ascending",
        "subject_id": "hidden",
        "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
    }

    monkeypatch.setattr(
        v25,
        "v28_anchor_gradients_and_compatible_jacobian",
        lambda **_kwargs: {
            "compatible_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
            "g_compatible": torch.ones(v25.SOURCE_WEIGHT_DIM),
            "g_conflict": torch.ones(v25.SOURCE_WEIGHT_DIM),
            "g_target": torch.ones(v25.SOURCE_WEIGHT_DIM),
            "support_split_counts": {"compatible": 1, "conflict": 1, "target": 1},
        },
    )
    monkeypatch.setattr(
        v25,
        "select_v31_sign_coherent_sparse_coordinates",
        lambda **_kwargs: [0, 1],
    )
    monkeypatch.setattr(
        v25,
        "solve_v37_projected_support_optimizer_edit",
        lambda **_kwargs: {
            "audit": {
                "compatible_constraint_feasible": True,
                "compatible_constraint_residual": 0.0,
                "compatible_dual_lambda": 0.0,
                "compatible_dual_update_source": "post_step_projected_delta",
                "compatible_mse_budget": 10.0,
                "final_loss": 1.0,
                "optimization_steps": 1,
                "optimizer_audit_hash": "f" * 64,
                "support_compatible_mse": 0.0,
                "support_runner_margin": 0.1,
                "support_target_margin": 0.2,
                "support_tournament_margin": 0.3,
                "trajectory_frontier_selection_mode": (
                    "frontier_target_and_compatible_feasible"
                ),
            },
            "delta": torch.zeros(v25.SOURCE_WEIGHT_DIM),
        },
    )

    def fake_control_record(**kwargs: Any) -> dict[str, Any]:
        captured["metadata"] = dict(kwargs["metadata"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.0,
            "editor": {
                **dict(kwargs["metadata"]),
                "delta_sha256": "9" * 64,
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.0,
            "target_margin": 0.0,
        }

    monkeypatch.setattr(v25, "control_record_for_delta", fake_control_record)

    result = v25.evaluate_v25_compatible_dual_frontier_matched_edit(
        subject=subject,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        config={
            **v25.build_v42_compatible_dual_frontier_config_grid()[0],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        selected_config_hash="c" * 64,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        script_sha256="script",
    )

    metadata = captured["metadata"]
    assert metadata["matched_edit_source"] == (
        v25.V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
    )
    assert metadata["experiment_variant"] == v25.V42_EXPERIMENT_VARIANT
    assert metadata["optimizer_audit"]["compatible_dual_update_source"] == (
        "post_step_projected_delta"
    )
    assert metadata["optimizer_audit"]["compatible_constraint_feasible"] is True
    assert "compatible_dual_frontier_provenance_hash" in metadata
    assert result["editor"]["matched_edit_source"] == (
        v25.V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
    )


def test_v25_inner_validation_config_hash_binds_matched_edit_source() -> None:
    base = {
        "compat_weight": 0.0,
        "config_index": 0,
        "matched_edit_source": "jacobian",
        "projection": "none",
        "ridge_lambda": 1e-5,
    }
    changed = {**base, "matched_edit_source": "empirical_centroid_task_vector"}

    assert v25.v25_inner_validation_config_hash(base) != (
        v25.v25_inner_validation_config_hash(changed)
    )


def test_v25_successive_halving_plan_is_hash_bound_and_redacted() -> None:
    plan = v25.build_v25_successive_halving_plan(
        configs=v25.build_v25_config_grid(),
        rung_job_counts=[4, 12],
        keep_fractions=[0.25, 0.25],
    )
    plan_text = json.dumps(plan, sort_keys=True)

    assert plan["config_count"] == 128
    assert plan["rung_count"] == 2
    assert len(plan["plan_hash"]) == 64
    assert plan["rungs"] == [
        {
            "input_config_count": 128,
            "keep_config_count": 32,
            "rung_index": 0,
            "rung_job_count": 4,
        },
        {
            "input_config_count": 32,
            "keep_config_count": 8,
            "rung_index": 1,
            "rung_job_count": 12,
        },
    ]
    assert "weights" not in plan_text
    assert "subject_id" not in plan_text


def test_v25_inner_validation_runner_logs_rungs_and_selects_top_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = [
        {
            "compat_weight": 0.0,
            "config_index": 0,
            "projection": "rank1",
            "ridge_lambda": 1e-4,
        },
        {
            "compat_weight": 0.1,
            "config_index": 1,
            "projection": "spectral_rank4",
            "ridge_lambda": 1e-4,
        },
    ]
    jobs = [
        {
            "direction": f"sorted_ascending->sorted_descending-{index}",
            "record_id": f"job-{index}",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": f"hidden-{index}",
                "weights": [0.0],
            },
            "target_behavior": "sorted_descending",
        }
        for index in range(2)
    ]
    progress_log = tmp_path / "inner_validation_progress.jsonl"
    bank_calls = []

    def fake_bank(**kwargs):
        bank_calls.append({
            "role": str(kwargs["train_edit_bank_role"]),
            "projection": str(kwargs["config"]["projection"]),
            "has_spectral_basis": kwargs.get("spectral_basis") is not None,
        })
        return {
            "bank_hash": v25.stable_hash_json({
                "config": kwargs["config"],
                "scope": "fake_bank",
            }),
            "entries": [{"delta": torch.zeros(v25.SOURCE_WEIGHT_DIM)}],
            "entry_count": 1,
            "entry_hashes": ["e" * 64],
            "norm_cap": float(kwargs["norm_cap"]),
        }

    def fake_spectral(_matrix, *, rank):
        return (
            torch.eye(v25.SOURCE_WEIGHT_DIM, int(rank), dtype=torch.float32),
            {
                "basis_sha256": "c" * 64,
                "centered_delta_sha256": "d" * 64,
                "delta_count": 1,
                "explained_singular_values": [1.0],
                "rank": int(rank),
            },
        )

    def fake_contexts(**kwargs):
        return {
            v25.stable_hash_json(str(job["record_id"])): {
                "context_hash": v25.stable_hash_json({
                    "record_id": str(job["record_id"]),
                    "selected_config_hash": kwargs["selected_config_hash"],
                }),
            }
            for job in kwargs["jobs"][: int(kwargs["max_jobs"])]
        }

    def fake_evaluate(**kwargs):
        config_index = int(kwargs["config"]["config_index"])
        strong = config_index == 1
        return {
            "evaluated_count": int(kwargs["max_jobs"]),
            "max_jobs": int(kwargs["max_jobs"]),
            "proof_record_hashes": [str(config_index) * 64],
            "proof_records": [
                _synthetic_v25_proof_record(
                    direction=str(job["direction"]),
                    target_margin=0.5 if strong else 0.1,
                    target_prediction_pass=strong,
                )
                for job in kwargs["jobs"][: int(kwargs["max_jobs"])]
            ],
            "total_planned_jobs": len(kwargs["jobs"]),
        }

    monkeypatch.setattr(v25, "build_v25_train_delta_bank_with_progress", fake_bank)
    monkeypatch.setattr(v25, "compute_train_spectral_basis", fake_spectral)
    monkeypatch.setattr(v25, "build_v25_train_only_control_contexts_with_progress", fake_contexts)
    monkeypatch.setattr(v25, "evaluate_v25_development_jobs_with_progress", fake_evaluate)

    result = v25.run_v25_inner_validation_successive_halving_with_progress(
        configs=configs,
        jobs=jobs,
        train_subjects=[{"pattern": "sorted_ascending", "subject_id": "hidden-train"}],
        train_stats={"train_statistics_hash": "a" * 64},
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        rung_job_counts=[2],
        keep_fractions=[0.5],
        norm_cap=0.25,
        job_plan_hash="b" * 64,
        script_sha256="script",
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 12.0,
    )
    log_text = progress_log.read_text()

    assert result["best_candidate"]["config_index"] == 1
    assert result["plan"]["rung_count"] == 1
    assert result["rungs"][0]["kept_config_hashes"] == [
        result["best_candidate"]["config_hash"]
    ]
    assert "inner_validation_start" in log_text
    assert "inner_validation_rung_start" in log_text
    assert "inner_validation_candidate_completed" in log_text
    assert "inner_validation_rung_completed" in log_text
    assert "inner_validation_completed" in log_text
    assert "weights" not in log_text
    assert "subject_id" not in log_text
    assert {
        "role": "spectral_seed",
        "projection": "none",
        "has_spectral_basis": False,
    } in bank_calls
    assert {
        "role": "actual",
        "projection": "spectral_rank4",
        "has_spectral_basis": True,
    } in bank_calls


def test_v27_inner_validation_variant_label_is_v27_in_progress_and_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "config_index": 0,
        "localized_basis": "output_layer_topk",
        "localized_delta_l2_weight": 0.0,
        "localized_lr": 0.05,
        "localized_norm_cap": 0.25,
        "localized_source_mse_weight": 0.5,
        "localized_steps": 25,
        "matched_edit_source": "localized_behavior_loss_subspace",
        "train_pool_file_sha256": "a" * 64,
        "train_pool_summary_hash": "b" * 64,
    }
    job = {
        "direction": "sorted_ascending->sorted_descending",
        "record_id": "job-0",
        "source_behavior": "sorted_ascending",
        "subject": {
            "pattern": "sorted_ascending",
            "subject_id": "hidden-0",
            "weights": [0.0],
        },
        "target_behavior": "sorted_descending",
    }
    progress_log = tmp_path / "v27_inner_validation_progress.jsonl"
    bank_configs: list[dict[str, Any]] = []

    def fake_bank(**kwargs: Any) -> dict[str, Any]:
        bank_configs.append(dict(kwargs["config"]))
        return {
            "bank_hash": "c" * 64,
            "entries": [{"delta": torch.zeros(v25.SOURCE_WEIGHT_DIM)}],
            "entry_count": 1,
            "entry_hashes": ["d" * 64],
            "norm_cap": float(kwargs["norm_cap"]),
        }

    def fake_spectral(
        _matrix: torch.Tensor,
        *,
        rank: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return (
            torch.eye(v25.SOURCE_WEIGHT_DIM, int(rank), dtype=torch.float32),
            {
                "basis_sha256": "e" * 64,
                "centered_delta_sha256": "f" * 64,
                "delta_count": 1,
                "explained_singular_values": [1.0],
                "rank": int(rank),
            },
        )

    def fake_contexts(**_kwargs: Any) -> dict[str, dict[str, Any]]:
        return {v25.stable_hash_json(str(job["record_id"])): {"context_hash": "1" * 64}}

    def fake_evaluate(**_kwargs: Any) -> dict[str, Any]:
        return {
            "evaluated_count": 1,
            "proof_records": [
                _synthetic_v25_proof_record(
                    direction=str(job["direction"]),
                    target_margin=0.5,
                    target_prediction_pass=True,
                )
            ],
        }

    monkeypatch.setattr(v25, "build_v25_train_delta_bank_with_progress", fake_bank)
    monkeypatch.setattr(v25, "compute_train_spectral_basis", fake_spectral)
    monkeypatch.setattr(v25, "build_v25_train_only_control_contexts_with_progress", fake_contexts)
    monkeypatch.setattr(v25, "evaluate_v25_development_jobs_with_progress", fake_evaluate)

    result = v25.run_v25_inner_validation_successive_halving_with_progress(
        configs=[config],
        jobs=[job],
        train_subjects=[{"pattern": "sorted_ascending", "subject_id": "hidden-train"}],
        train_stats={"train_statistics_hash": "a" * 64},
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        rung_job_counts=[1],
        keep_fractions=[1.0],
        norm_cap=0.25,
        job_plan_hash="b" * 64,
        script_sha256="script",
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 12.0,
        experiment_variant=v25.V27_EXPERIMENT_VARIANT,
    )
    events = [
        json.loads(line)
        for line in progress_log.read_text().splitlines()
        if line.strip()
    ]

    assert v25.experiment_variant_for_inner_validation_grid("v27-localized") == (
        v25.V27_EXPERIMENT_VARIANT
    )
    assert result["experiment_variant"] == v25.V27_EXPERIMENT_VARIANT
    assert result["best_candidate"]["experiment_variant"] == v25.V27_EXPERIMENT_VARIANT
    assert bank_configs == [v25.V27_LOCALIZED_NATIVE_CONTROL_CONFIG]
    assert any(
        event["event"] == "inner_validation_start"
        and event["experiment_variant"] == v25.V27_EXPERIMENT_VARIANT
        for event in events
    )
    assert any(
        event["event"] == "inner_validation_candidate_completed"
        and event["experiment_variant"] == v25.V27_EXPERIMENT_VARIANT
        for event in events
    )


def test_v33_inner_validation_variant_label_overrides_reused_v32_edit_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        **v25.build_v33_proof_gate_diagnostic_config_grid()[0],
        "train_pool_file_sha256": "a" * 64,
        "train_pool_summary_hash": "b" * 64,
    }
    job = {
        "direction": "sorted_ascending->sorted_descending",
        "record_id": "job-0",
        "source_behavior": "sorted_ascending",
        "subject": {
            "pattern": "sorted_ascending",
            "subject_id": "hidden-0",
            "weights": [0.0],
        },
        "target_behavior": "sorted_descending",
    }
    progress_log = tmp_path / "v33_inner_validation_progress.jsonl"

    def fake_bank(**_kwargs: Any) -> dict[str, Any]:
        return {
            "bank_hash": "c" * 64,
            "entries": [{"delta": torch.zeros(v25.SOURCE_WEIGHT_DIM)}],
            "entry_count": 1,
            "entry_hashes": ["d" * 64],
            "norm_cap": 0.25,
        }

    def fake_spectral(
        _matrix: torch.Tensor,
        *,
        rank: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return (
            torch.eye(v25.SOURCE_WEIGHT_DIM, int(rank), dtype=torch.float32),
            {
                "basis_sha256": "e" * 64,
                "centered_delta_sha256": "f" * 64,
                "delta_count": 1,
                "explained_singular_values": [1.0],
                "rank": int(rank),
            },
        )

    def fake_contexts(**_kwargs: Any) -> dict[str, dict[str, Any]]:
        return {v25.stable_hash_json(str(job["record_id"])): {"context_hash": "1" * 64}}

    def fake_evaluate(**_kwargs: Any) -> dict[str, Any]:
        return {
            "evaluated_count": 1,
            "proof_records": [
                _synthetic_v25_proof_record(
                    direction=str(job["direction"]),
                    target_margin=0.5,
                    target_prediction_pass=True,
                )
            ],
        }

    monkeypatch.setattr(v25, "build_v25_train_delta_bank_with_progress", fake_bank)
    monkeypatch.setattr(v25, "compute_train_spectral_basis", fake_spectral)
    monkeypatch.setattr(v25, "build_v25_train_only_control_contexts_with_progress", fake_contexts)
    monkeypatch.setattr(v25, "evaluate_v25_development_jobs_with_progress", fake_evaluate)

    result = v25.run_v25_inner_validation_successive_halving_with_progress(
        configs=[config],
        jobs=[job],
        train_subjects=[{"pattern": "sorted_ascending", "subject_id": "hidden-train"}],
        train_stats={"train_statistics_hash": "a" * 64},
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        rung_job_counts=[1],
        keep_fractions=[1.0],
        norm_cap=0.25,
        job_plan_hash="b" * 64,
        script_sha256="script",
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 12.0,
        experiment_variant=v25.V33_EXPERIMENT_VARIANT,
    )
    events = [
        json.loads(line)
        for line in progress_log.read_text().splitlines()
        if line.strip()
    ]

    assert v25.experiment_variant_for_inner_validation_grid(
        "v33-proof-gate-diagnostic"
    ) == v25.V33_EXPERIMENT_VARIANT
    assert result["experiment_variant"] == v25.V33_EXPERIMENT_VARIANT
    assert result["best_candidate"]["experiment_variant"] == v25.V33_EXPERIMENT_VARIANT
    assert result["best_candidate"]["experiment_variant"] != v25.V32_EXPERIMENT_VARIANT
    assert any(
        event["event"] == "inner_validation_candidate_completed"
        and event["experiment_variant"] == v25.V33_EXPERIMENT_VARIANT
        for event in events
    )


def test_v34_inner_validation_variant_label_overrides_reused_v32_edit_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        **v25.build_v34_locality_pressure_config_grid()[0],
        "train_pool_file_sha256": "a" * 64,
        "train_pool_summary_hash": "b" * 64,
    }
    job = {
        "direction": "sorted_ascending->sorted_descending",
        "record_id": "job-0",
        "source_behavior": "sorted_ascending",
        "subject": {
            "pattern": "sorted_ascending",
            "subject_id": "hidden-0",
            "weights": [0.0],
        },
        "target_behavior": "sorted_descending",
    }
    progress_log = tmp_path / "v34_inner_validation_progress.jsonl"

    def fake_bank(**_kwargs: Any) -> dict[str, Any]:
        return {
            "bank_hash": "c" * 64,
            "entries": [{"delta": torch.zeros(v25.SOURCE_WEIGHT_DIM)}],
            "entry_count": 1,
            "entry_hashes": ["d" * 64],
            "norm_cap": 0.25,
        }

    def fake_spectral(
        _matrix: torch.Tensor,
        *,
        rank: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return (
            torch.eye(v25.SOURCE_WEIGHT_DIM, int(rank), dtype=torch.float32),
            {
                "basis_sha256": "e" * 64,
                "centered_delta_sha256": "f" * 64,
                "delta_count": 1,
                "explained_singular_values": [1.0],
                "rank": int(rank),
            },
        )

    def fake_contexts(**_kwargs: Any) -> dict[str, dict[str, Any]]:
        return {v25.stable_hash_json(str(job["record_id"])): {"context_hash": "1" * 64}}

    def fake_evaluate(**_kwargs: Any) -> dict[str, Any]:
        return {
            "evaluated_count": 1,
            "proof_records": [
                _synthetic_v25_proof_record(
                    direction=str(job["direction"]),
                    target_margin=0.5,
                    target_prediction_pass=True,
                )
            ],
        }

    monkeypatch.setattr(v25, "build_v25_train_delta_bank_with_progress", fake_bank)
    monkeypatch.setattr(v25, "compute_train_spectral_basis", fake_spectral)
    monkeypatch.setattr(v25, "build_v25_train_only_control_contexts_with_progress", fake_contexts)
    monkeypatch.setattr(v25, "evaluate_v25_development_jobs_with_progress", fake_evaluate)

    result = v25.run_v25_inner_validation_successive_halving_with_progress(
        configs=[config],
        jobs=[job],
        train_subjects=[{"pattern": "sorted_ascending", "subject_id": "hidden-train"}],
        train_stats={"train_statistics_hash": "a" * 64},
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        rung_job_counts=[1],
        keep_fractions=[1.0],
        norm_cap=0.25,
        job_plan_hash="b" * 64,
        script_sha256="script",
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 12.0,
        experiment_variant=v25.V34_EXPERIMENT_VARIANT,
    )
    events = [
        json.loads(line)
        for line in progress_log.read_text().splitlines()
        if line.strip()
    ]

    assert v25.experiment_variant_for_inner_validation_grid(
        "v34-locality-pressure"
    ) == v25.V34_EXPERIMENT_VARIANT
    assert result["experiment_variant"] == v25.V34_EXPERIMENT_VARIANT
    assert result["best_candidate"]["experiment_variant"] == v25.V34_EXPERIMENT_VARIANT
    assert result["best_candidate"]["experiment_variant"] != v25.V32_EXPERIMENT_VARIANT
    assert any(
        event["event"] == "inner_validation_candidate_completed"
        and event["experiment_variant"] == v25.V34_EXPERIMENT_VARIANT
        for event in events
    )


def test_v25_inner_validation_passes_train_pool_provenance_to_empirical_bank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}
    config = {
        "compat_weight": 0.0,
        "config_index": 0,
        "matched_edit_source": "empirical_centroid_task_vector",
        "projection": "none",
        "ridge_lambda": 1e-5,
    }
    train_subjects = [
        {
            "pattern": pattern,
            "subject_id": f"train-{pattern}",
            "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
        }
        for pattern in v25.PATTERNS
    ]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": [{"sequence": [0, 1, 2, 3, 4]}],
        "probe_examples_hash": "p" * 64,
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
        "train_statistics_hash": "s" * 64,
    }

    def fake_train_bank(**kwargs: Any) -> dict[str, Any]:
        return {
            "bank_hash": "j" * 64,
            "entries": [{
                "delta": torch.zeros(v25.SOURCE_WEIGHT_DIM),
                "delta_sha256": "d" * 64,
                "direction": "sorted_ascending->sorted_descending",
                "source_behavior": "sorted_ascending",
                "source_descriptor": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
                "target_behavior": "sorted_descending",
            }],
            "entry_count": 1,
            "entry_hashes": ["e" * 64],
        }

    def fake_empirical_bank(**kwargs: Any) -> dict[str, Any]:
        captured["train_pool_file_sha256"] = kwargs["train_pool_file_sha256"]
        captured["train_pool_summary_hash"] = kwargs["train_pool_summary_hash"]
        return {
            "bank_hash": "k" * 64,
            "entries": [{
                "delta": torch.zeros(v25.SOURCE_WEIGHT_DIM),
                "delta_sha256": "f" * 64,
                "direction": "sorted_ascending->sorted_descending",
                "editor_audit": {"edit_source": "empirical_centroid_task_vector"},
                "source_behavior": "sorted_ascending",
                "target_behavior": "sorted_descending",
            }],
            "entry_count": 1,
            "entry_hashes": ["g" * 64],
        }

    monkeypatch.setattr(v25, "build_v25_train_delta_bank_with_progress", fake_train_bank)
    monkeypatch.setattr(
        v25,
        "build_v25_empirical_task_vector_bank_with_progress",
        fake_empirical_bank,
    )
    monkeypatch.setattr(
        v25,
        "build_v25_train_only_control_contexts_with_progress",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        v25,
        "evaluate_v25_development_jobs_with_progress",
        lambda **kwargs: {"evaluated_count": 0, "proof_records": []},
    )

    v25.run_v25_inner_validation_successive_halving_with_progress(
        configs=[config],
        jobs=[{"record_id": "dev-1"}],
        train_subjects=train_subjects,
        train_stats=train_stats,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        rung_job_counts=[1],
        keep_fractions=[1.0],
        norm_cap=0.25,
        job_plan_hash="c" * 64,
        script_sha256="script",
        progress_log_path=tmp_path / "progress.jsonl",
        started_at_monotonic=1.0,
    )

    assert captured == {
        "train_pool_file_sha256": "a" * 64,
        "train_pool_summary_hash": "b" * 64,
    }


def test_v25_inner_validation_runner_selects_best_from_final_rung(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = [
        {
            "compat_weight": 0.0,
            "config_index": 0,
            "projection": "rank1",
            "ridge_lambda": 1e-4,
        },
        {
            "compat_weight": 0.1,
            "config_index": 1,
            "projection": "spectral_rank4",
            "ridge_lambda": 1e-4,
        },
    ]
    jobs = [
        {
            "direction": f"sorted_ascending->sorted_descending-{index}",
            "record_id": f"job-{index}",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": f"hidden-{index}",
                "weights": [0.0],
            },
            "target_behavior": "sorted_descending",
        }
        for index in range(2)
    ]

    monkeypatch.setattr(v25, "build_v25_train_delta_bank_with_progress", lambda **kwargs: {
        "bank_hash": v25.stable_hash_json(kwargs["config"]),
        "entries": [{"delta": torch.zeros(v25.SOURCE_WEIGHT_DIM)}],
    })
    monkeypatch.setattr(v25, "compute_train_spectral_basis", lambda _matrix, *, rank: (
        torch.eye(v25.SOURCE_WEIGHT_DIM, int(rank), dtype=torch.float32),
        {"basis_sha256": "c" * 64},
    ))
    monkeypatch.setattr(v25, "build_v25_train_only_control_contexts_with_progress", lambda **kwargs: {})

    def fake_evaluate(**kwargs):
        config_index = int(kwargs["config"]["config_index"])
        max_jobs = int(kwargs["max_jobs"])
        if max_jobs == 1:
            target_margin = 0.9 if config_index == 0 else 0.2
        else:
            target_margin = 0.1 if config_index == 0 else 0.4
        return {
            "evaluated_count": max_jobs,
            "max_jobs": max_jobs,
            "proof_records": [
                _synthetic_v25_proof_record(
                    direction=str(job["direction"]),
                    target_margin=target_margin,
                    target_prediction_pass=True,
                )
                for job in kwargs["jobs"][:max_jobs]
            ],
            "total_planned_jobs": len(kwargs["jobs"]),
        }

    monkeypatch.setattr(v25, "evaluate_v25_development_jobs_with_progress", fake_evaluate)

    result = v25.run_v25_inner_validation_successive_halving_with_progress(
        configs=configs,
        jobs=jobs,
        train_subjects=[{"pattern": "sorted_ascending", "subject_id": "hidden-train"}],
        train_stats={"train_statistics_hash": "a" * 64},
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        rung_job_counts=[1, 2],
        keep_fractions=[1.0, 0.5],
        norm_cap=0.25,
        job_plan_hash="b" * 64,
        script_sha256="script",
        progress_log_path=tmp_path / "progress.jsonl",
        started_at_monotonic=10.0,
        now_monotonic=lambda: 12.0,
    )

    assert result["best_candidate"]["config_index"] == 1
    assert result["best_candidate"]["rung_index"] == 1


def test_v25_inner_validation_runner_fails_closed_when_rung_has_no_valid_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = [
        {
            "compat_weight": 0.0,
            "config_index": 0,
            "projection": "rank1",
            "ridge_lambda": 1e-4,
        },
        {
            "compat_weight": 0.1,
            "config_index": 1,
            "projection": "spectral_rank4",
            "ridge_lambda": 1e-4,
        },
    ]
    jobs = [{
        "direction": "sorted_ascending->sorted_descending",
        "record_id": "job",
        "source_behavior": "sorted_ascending",
        "subject": {"pattern": "sorted_ascending", "subject_id": "hidden", "weights": [0.0]},
        "target_behavior": "sorted_descending",
    }]
    monkeypatch.setattr(
        v25,
        "build_v25_train_delta_bank_with_progress",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    progress_log = tmp_path / "progress.jsonl"

    result = v25.run_v25_inner_validation_successive_halving_with_progress(
        configs=configs,
        jobs=jobs,
        train_subjects=[{"pattern": "sorted_ascending", "subject_id": "hidden-train"}],
        train_stats={"train_statistics_hash": "a" * 64},
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        rung_job_counts=[1],
        keep_fractions=[0.5],
        norm_cap=0.25,
        job_plan_hash="b" * 64,
        script_sha256="script",
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 12.0,
    )
    log_text = progress_log.read_text()

    assert result["stage"] == "inner_validation_failed"
    assert result["invalid"] is True
    assert result["failed_rung_index"] == 0
    assert "inner_validation_failed" in log_text
    assert "boom" not in log_text


def test_v25_control_count_constants_match_preregistered_sets() -> None:
    assert v25.RANDOM_CONTROLS_PER_RECORD == 19
    assert len(v25.PROOF_CRITICAL_CONTROL_TYPES) == 11
    assert len(v25.DIAGNOSTIC_CONTROL_TYPES) == 4
    assert v25.EXPECTED_CONTROLS_PER_RECORD == 34


def test_v25_parse_args_exposes_explicit_bounded_dry_run_flags(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--phase",
        "development",
        "--max-development-jobs",
        "1",
        "--development-job-selection",
        "balanced-directions",
        "--run-inner-validation",
        "--inner-validation-rung-jobs",
        "2,4",
        "--inner-validation-keep-fractions",
        "0.5,1.0",
        "--inner-validation-config-grid",
        "v36-compatible-nullspace-projection",
        "--inner-validation-max-configs",
        "4",
        "--dry-run-placeholder-controls",
    ])

    args = v25.parse_args()

    assert args.max_development_jobs == 1
    assert args.development_job_selection == "balanced-directions"
    assert args.run_inner_validation is True
    assert args.inner_validation_rung_jobs == "2,4"
    assert args.inner_validation_keep_fractions == "0.5,1.0"
    assert args.inner_validation_config_grid == "v36-compatible-nullspace-projection"
    assert args.inner_validation_max_configs == 4
    assert args.dry_run_placeholder_controls is True


def test_v25_expected_control_types_include_random_controls() -> None:
    control_types = v25.expected_v25_control_types()

    assert len(control_types) == v25.EXPECTED_CONTROLS_PER_RECORD
    assert control_types[: len(v25.PROOF_CRITICAL_CONTROL_TYPES)] == (
        v25.PROOF_CRITICAL_CONTROL_TYPES
    )
    assert control_types[-1] == "random_matched_norm_18"
    assert len(set(control_types)) == len(control_types)


def test_v25_write_development_results_artifact_logs_external_file_hash(
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

    written = v25.write_development_results_artifact(
        output_path=output_path,
        result=result,
        progress_log_path=progress_log_path,
        started_at_monotonic=0.0,
    )

    on_disk = json.loads(output_path.read_text())
    progress_event = json.loads(progress_log_path.read_text().splitlines()[-1])

    assert "development_results_sha256" not in on_disk
    assert on_disk == written
    assert progress_event["event"] == "development_results_written"
    assert progress_event["development_results_file_sha256"] == v25.sha256_file(output_path)
    assert (
        progress_event["development_results_payload_sha256"]
        == on_disk["development_results_payload_sha256"]
    )


def test_v25_shuffled_signature_derangement_is_deterministic_and_nonidentity() -> None:
    rows = [
        {"source_behavior": "has_majority", "subject_id": "s1", "target_behavior": "mountain_pattern"},
        {"source_behavior": "has_majority", "subject_id": "s2", "target_behavior": "sorted_ascending"},
        {"source_behavior": "has_majority", "subject_id": "s3", "target_behavior": "sorted_descending"},
    ]

    first = v25.shuffled_signature_derangement_indices(
        rows,
        split_name="development",
        rung_index=0,
        source_behavior="has_majority",
        config_hash="config",
    )
    second = v25.shuffled_signature_derangement_indices(
        rows,
        split_name="development",
        rung_index=0,
        source_behavior="has_majority",
        config_hash="config",
    )

    assert first == second
    assert sorted(first) == [0, 1, 2]
    assert all(index != replacement for index, replacement in enumerate(first))


def test_v25_shuffled_signature_derangement_uses_sorted_row_mapping() -> None:
    rows = [
        {"source_behavior": "has_majority", "subject_id": "s3", "target_behavior": "sorted_descending"},
        {"source_behavior": "has_majority", "subject_id": "s1", "target_behavior": "mountain_pattern"},
        {"source_behavior": "has_majority", "subject_id": "s2", "target_behavior": "sorted_ascending"},
    ]

    replacements = v25.shuffled_signature_derangement_indices(
        rows,
        split_name="development",
        rung_index=0,
        source_behavior="has_majority",
        config_hash="config",
    )

    assert sorted(replacements) == [0, 1, 2]
    assert all(index != replacement for index, replacement in enumerate(replacements))
    replacement_subjects = [rows[index]["subject_id"] for index in replacements]
    assert replacement_subjects != ["s3", "s1", "s2"]


def test_v25_shuffled_signature_derangement_fails_closed_for_small_group() -> None:
    with pytest.raises(ValueError, match="derangement"):
        v25.shuffled_signature_derangement_indices(
            [{"source_behavior": "has_majority", "subject_id": "s1"}],
            split_name="development",
            rung_index=0,
            source_behavior="has_majority",
            config_hash="config",
        )


def test_v25_jacobian_ridge_edit_reduces_linear_residual() -> None:
    jacobian = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    target_delta = torch.tensor([2.0, -4.0])

    delta = v25.solve_jacobian_ridge_edit(jacobian, target_delta, ridge_lambda=1e-6)

    before = torch.linalg.norm(target_delta)
    after = torch.linalg.norm(target_delta - jacobian @ delta)
    assert after < before * 1e-3


def test_v25_jacobian_ridge_edit_rejects_nonfinite_inputs() -> None:
    jacobian = torch.tensor([[1.0, float("nan")]])
    target_delta = torch.tensor([1.0])

    with pytest.raises(ValueError, match="nonfinite"):
        v25.solve_jacobian_ridge_edit(jacobian, target_delta, ridge_lambda=1e-4)


def test_v25_jacobian_ridge_pinv_fallback_is_explicit_and_audited() -> None:
    jacobian = torch.zeros((2, v25.SOURCE_WEIGHT_DIM), dtype=torch.float32)
    target_delta = torch.ones(2, dtype=torch.float32)

    with pytest.raises(RuntimeError):
        v25.solve_jacobian_ridge_edit(
            jacobian,
            target_delta,
            ridge_lambda=0.0,
        )

    audit = {}
    delta = v25.solve_jacobian_ridge_edit(
        jacobian,
        target_delta,
        ridge_lambda=0.0,
        allow_pinv_fallback=True,
        audit=audit,
    )

    assert torch.count_nonzero(delta).item() == 0
    assert audit == {
        "pinv_fallback_allowed": True,
        "pinv_fallback_used": True,
    }


def test_v25_project_matrix_rank1_returns_rank_at_most_one() -> None:
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    projected = v25.project_matrix_rank1(matrix)

    assert projected.shape == matrix.shape
    assert torch.linalg.matrix_rank(projected).item() <= 1


def test_v25_project_matrix_rank1_canonicalizes_svd_sign() -> None:
    matrix = torch.tensor([[-3.0, -1.0], [-2.0, -4.0]])

    left, _, _ = v25.rank1_svd_factors(matrix)

    assert left[torch.argmax(torch.abs(left))].item() >= 0.0


def test_v25_project_to_basis_uses_train_only_basis_columns() -> None:
    delta = torch.tensor([1.0, 2.0, 3.0])
    basis = torch.eye(3)[:, :2]

    projected = v25.project_to_basis(delta, basis)

    assert torch.allclose(projected, torch.tensor([1.0, 2.0, 0.0]))


def test_v25_train_spectral_basis_is_orthonormal_and_hash_bound() -> None:
    deltas = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]])

    basis, audit = v25.compute_train_spectral_basis(deltas, rank=2)

    assert basis.shape == (3, 2)
    assert torch.allclose(basis.T @ basis, torch.eye(2), atol=1e-5)
    assert audit["rank"] == 2
    assert audit["delta_count"] == 3
    assert len(audit["basis_sha256"]) == 64


def test_v25_activation_statistics_are_train_only_finite_and_hash_bound() -> None:
    records = [
        {
            "subject_id": f"train-{index}",
            "weights": torch.linspace(-0.1, 0.1, v25.SOURCE_WEIGHT_DIM).add(index * 0.01).tolist(),
        }
        for index in range(3)
    ]
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]

    stats = v25.fit_v25_activation_statistics(records, probe_examples=probe_examples)

    assert stats["activation_descriptor_mean"].shape == (v25.ACTIVATION_DESCRIPTOR_DIM,)
    assert stats["activation_descriptor_std"].shape == (v25.ACTIVATION_DESCRIPTOR_DIM,)
    assert torch.isfinite(stats["activation_descriptor_mean"]).all()
    assert torch.isfinite(stats["activation_descriptor_std"]).all()
    assert stats["activation_descriptor_count"] == 3
    assert len(stats["descriptor_norm_hash"]) == 64
    assert set(stats["activation_descriptor_by_subject"]) == {"train-0", "train-1", "train-2"}


def test_v25_activation_jacobian_shape_and_cache_key_are_stable() -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
    }

    jacobian = v25.activation_jacobian_for_weights(weights, train_stats=stats)
    first_key = v25.build_jacobian_cache_key(
        subject_id="subject",
        source_behavior="sorted_ascending",
        train_stats=stats,
        script_sha256="script",
    )
    second_key = v25.build_jacobian_cache_key(
        subject_id="subject",
        source_behavior="sorted_ascending",
        train_stats=stats,
        script_sha256="script",
    )

    assert jacobian.shape == (v25.ACTIVATION_DESCRIPTOR_DIM, v25.SOURCE_WEIGHT_DIM)
    assert torch.isfinite(jacobian).all()
    assert first_key == second_key
    assert len(first_key) == 64


def test_v25_augmented_jacobian_omits_compatibility_rows_at_zero_weight() -> None:
    activation_jacobian = torch.eye(2, 3)
    activation_delta = torch.tensor([1.0, -1.0])
    source_logit_jacobian = torch.ones(4, 3)

    no_compat_j, no_compat_b = v25.build_augmented_jacobian_system(
        activation_jacobian=activation_jacobian,
        activation_delta=activation_delta,
        source_logit_jacobian=source_logit_jacobian,
        compat_weight=0.0,
    )
    compat_j, compat_b = v25.build_augmented_jacobian_system(
        activation_jacobian=activation_jacobian,
        activation_delta=activation_delta,
        source_logit_jacobian=source_logit_jacobian,
        compat_weight=0.25,
    )

    assert no_compat_j.shape == (2, 3)
    assert no_compat_b.shape == (2,)
    assert compat_j.shape == (6, 3)
    assert compat_b.shape == (6,)
    assert torch.allclose(compat_j[2:], torch.full((4, 3), 0.5))
    assert torch.allclose(compat_b[2:], torch.zeros(4))


def test_v25_rank1_projection_zeroes_output_layer_and_norm_cap() -> None:
    delta = torch.arange(v25.SOURCE_WEIGHT_DIM, dtype=torch.float32) / 100.0

    projected = v25.project_hidden_rank1_delta(delta)
    capped = v25.apply_norm_cap(projected, max_norm=0.5)

    for layer_index in range(5):
        weight_spec, _bias_spec = v25.v23.hidden_layer_specs(layer_index)
        matrix = v25.v17.component_from_flat(projected, weight_spec)
        assert torch.linalg.matrix_rank(matrix).item() <= 1
    assert torch.allclose(projected[336:345], torch.zeros(9))
    assert float(torch.linalg.norm(capped).item()) <= 0.50001


def test_v25_projected_jacobian_edit_solves_projects_and_caps_norm() -> None:
    source_descriptor = torch.tensor([0.0, 0.0])
    target_descriptor = torch.tensor([4.0, 0.0])
    activation_jacobian = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    source_logit_jacobian = torch.ones(2, 3)

    result = v25.solve_projected_jacobian_edit(
        source_descriptor=source_descriptor,
        target_descriptor=target_descriptor,
        activation_jacobian=activation_jacobian,
        source_logit_jacobian=source_logit_jacobian,
        config={"compat_weight": 0.0, "projection": "none", "ridge_lambda": 1e-6},
        norm_cap=0.5,
    )

    assert result["raw_delta"].shape == (3,)
    assert result["projected_delta"].shape == (3,)
    assert result["delta"].shape == (3,)
    assert float(torch.linalg.norm(result["delta"]).item()) <= 0.50001
    assert result["audit"]["projection"] == "none"
    assert result["audit"]["norm_cap_applied"] is True
    assert result["audit"]["augmented_row_count"] == 2
    assert len(result["audit"]["delta_sha256"]) == 64


def test_v25_projected_jacobian_edit_rank1_path_zeroes_output_layer() -> None:
    source_descriptor = torch.zeros(2)
    target_descriptor = torch.tensor([1.0, -1.0])
    activation_jacobian = torch.zeros(2, v25.SOURCE_WEIGHT_DIM)
    activation_jacobian[0, 0] = 1.0
    activation_jacobian[1, 1] = 1.0
    source_logit_jacobian = torch.zeros(1, v25.SOURCE_WEIGHT_DIM)

    result = v25.solve_projected_jacobian_edit(
        source_descriptor=source_descriptor,
        target_descriptor=target_descriptor,
        activation_jacobian=activation_jacobian,
        source_logit_jacobian=source_logit_jacobian,
        config={"compat_weight": 0.1, "projection": "rank1", "ridge_lambda": 1e-6},
        norm_cap=10.0,
    )

    assert torch.allclose(result["delta"][336:345], torch.zeros(9))
    assert result["audit"]["projection"] == "rank1"
    assert result["audit"]["augmented_row_count"] == 3


def test_v25_train_statistics_include_behavior_target_descriptors() -> None:
    records = []
    base = torch.linspace(-0.1, 0.1, v25.SOURCE_WEIGHT_DIM)
    for index, pattern in enumerate(v25.PATTERNS):
        records.append({
            "pattern": pattern,
            "subject_id": f"{pattern}-{index}",
            "weights": base.add(index * 0.01).tolist(),
        })
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]

    stats = v25.fit_v25_train_statistics(records, probe_examples=probe_examples)

    assert set(stats["target_activation_descriptor_by_behavior"]) == set(v25.PATTERNS)
    assert stats["train_counts_by_behavior"] == {pattern: 1 for pattern in v25.PATTERNS}
    assert len(stats["target_descriptor_hash_by_behavior"]) == len(v25.PATTERNS)
    assert len(stats["train_statistics_hash"]) == 64
    for descriptor in stats["target_activation_descriptor_by_behavior"].values():
        assert descriptor.shape == (v25.ACTIVATION_DESCRIPTOR_DIM,)
        assert torch.isfinite(descriptor).all()


def test_v25_jacobian_cache_entry_has_expected_tensors_and_safe_audit() -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "cache-subject",
        "weights": weights.tolist(),
    }
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
    }

    entry = v25.compute_jacobian_cache_entry(
        record,
        source_behavior="sorted_ascending",
        train_stats=train_stats,
        script_sha256="script",
    )

    assert entry["source_descriptor"].shape == (v25.ACTIVATION_DESCRIPTOR_DIM,)
    assert entry["activation_jacobian"].shape == (
        v25.ACTIVATION_DESCRIPTOR_DIM,
        v25.SOURCE_WEIGHT_DIM,
    )
    assert entry["source_logit_jacobian"].shape[1] == v25.SOURCE_WEIGHT_DIM
    assert torch.isfinite(entry["source_logits"]).all()
    assert len(entry["cache_key"]) == 64
    assert entry["audit"]["finite"] is True
    audit_text = json.dumps(entry["audit"], sort_keys=True)
    for forbidden in ["weights", "signature", "train_info", "logits", "descriptor"]:
        assert forbidden not in audit_text


def test_v25_matched_edit_evaluation_uses_functional_metrics_and_cap() -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "eval-subject",
        "weights": weights.tolist(),
    }
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
    }
    cache_entry = v25.compute_jacobian_cache_entry(
        record,
        source_behavior="sorted_ascending",
        train_stats=train_stats,
        script_sha256="script",
    )
    train_stats["target_activation_descriptor_by_behavior"]["sorted_descending"] = (
        cache_entry["source_descriptor"] + 0.1
    )

    result = v25.evaluate_v25_matched_edit(
        subject=record,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        train_stats=train_stats,
        cache_entry=cache_entry,
        config={"compat_weight": 0.0, "projection": "none", "ridge_lambda": 1e-4},
        norm_cap=0.25,
    )

    assert result["control_type"] == v25.EDITOR_METHOD
    assert result["target_prediction_pass"] == (
        result["predicted_behavior"] == "sorted_descending"
    )
    assert result["delta_norm"] <= 0.25001
    assert "compatible_source_output_mse" in result
    assert "target_margin" in result
    assert len(result["editor"]["delta_sha256"]) == 64


def test_v25_matched_edit_audits_spectral_projection_norm() -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "spectral-norm-subject",
        "weights": weights.tolist(),
    }
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
    }
    cache_entry = v25.compute_jacobian_cache_entry(
        record,
        source_behavior="sorted_ascending",
        train_stats=train_stats,
        script_sha256="script",
    )
    train_stats["target_activation_descriptor_by_behavior"]["sorted_descending"] = (
        cache_entry["source_descriptor"] + 0.05
    )
    spectral_basis = torch.eye(v25.SOURCE_WEIGHT_DIM, 4, dtype=torch.float32)

    result = v25.evaluate_v25_matched_edit(
        subject=record,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        train_stats=train_stats,
        cache_entry=cache_entry,
        config={"compat_weight": 0.1, "projection": "rank1", "ridge_lambda": 1e-4},
        norm_cap=0.25,
        spectral_basis=spectral_basis,
    )

    assert "matched_spectral_projection_norm" in result["editor"]
    assert result["editor"]["matched_spectral_projection_norm"] <= result["delta_norm"] + 1e-8


def test_v25_empirical_task_vector_matched_edit_uses_precomputed_direction_delta() -> None:
    weights = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "empirical-dev",
        "weights": weights.tolist(),
    }
    delta = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    delta[0] = 0.125
    bank = {
        "bank_hash": "b" * 64,
        "entries": [{
            "delta": delta,
            "delta_sha256": v25.stable_hash_json(v25.tensor_to_hashable(delta)),
            "direction": "sorted_ascending->sorted_descending",
            "editor_audit": {"edit_source": "empirical_centroid_task_vector"},
            "source_behavior": "sorted_ascending",
            "target_behavior": "sorted_descending",
        }],
    }

    result = v25.evaluate_v25_empirical_task_vector_matched_edit(
        subject=record,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        empirical_task_vector_bank=bank,
        selected_config_hash="c" * 64,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
    )

    assert result["control_type"] == v25.EDITOR_METHOD
    assert result["editor"]["matched_edit_source"] == "empirical_centroid_task_vector"
    assert result["editor"]["empirical_task_vector_bank_hash"] == "b" * 64
    assert result["delta_norm"] == pytest.approx(0.125)


def test_v25_random_matched_norm_controls_are_deterministic_projected_and_weightless() -> None:
    matched_delta = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    matched_delta[0] = 1.5
    source_weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)

    first = v25.build_v25_random_matched_norm_controls(
        matched_delta=matched_delta,
        source_weights=source_weights,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        record_id="record-1",
        selected_config_hash="config",
        projection="rank1",
    )
    second = v25.build_v25_random_matched_norm_controls(
        matched_delta=matched_delta,
        source_weights=source_weights,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        record_id="record-1",
        selected_config_hash="config",
        projection="rank1",
    )

    assert [control["control_type"] for control in first] == v25.expected_v25_random_control_types()
    assert v25.stable_hash_json(first) == v25.stable_hash_json(second)
    for control in first:
        assert "weights" not in control
        assert control["delta_norm"] == pytest.approx(float(torch.linalg.norm(matched_delta).item()))
        assert len(control["editor"]["delta_sha256"]) == 64


def test_v25_control_context_builder_hash_binds_deltas_and_shuffled_descriptor() -> None:
    shuffled_descriptor = torch.linspace(-0.1, 0.1, v25.ACTIVATION_DESCRIPTOR_DIM)
    delta_by_type = {
        control_type: torch.full((v25.SOURCE_WEIGHT_DIM,), 0.001 * (index + 1))
        for index, control_type in enumerate([
            "v21_baseline",
            "v22_baseline",
            "v23_baseline",
            "nearest_train_delta",
            "teacher_oracle_delta",
            "contrastive_weight_arithmetic",
        ])
    }

    first = v25.build_v25_control_context(
        shuffled_target_descriptor=shuffled_descriptor,
        precomputed_delta_by_control_type=delta_by_type,
        provenance={"split": "development", "source": "train-only"},
    )
    second = v25.build_v25_control_context(
        shuffled_target_descriptor=shuffled_descriptor,
        precomputed_delta_by_control_type=delta_by_type,
        provenance={"split": "development", "source": "train-only"},
    )
    changed = v25.build_v25_control_context(
        shuffled_target_descriptor=shuffled_descriptor,
        precomputed_delta_by_control_type={
            **delta_by_type,
            "v21_baseline": delta_by_type["v21_baseline"] + 0.001,
        },
        provenance={"split": "development", "source": "train-only"},
    )

    assert len(first["context_hash"]) == 64
    assert first["context_hash"] == second["context_hash"]
    assert first["context_hash"] != changed["context_hash"]
    assert len(first["shuffled_target_descriptor_hash"]) == 64
    assert set(first["precomputed_delta_hash_by_control_type"]) == {
        "v21_baseline",
        "v22_baseline",
        "v23_baseline",
        "nearest_train_delta",
        "teacher_oracle_delta",
        "contrastive_weight_arithmetic",
    }


def test_v25_control_context_redaction_removes_raw_delta_and_descriptor_tensors() -> None:
    context = v25.build_v25_control_context(
        shuffled_target_descriptor=torch.linspace(-0.1, 0.1, v25.ACTIVATION_DESCRIPTOR_DIM),
        precomputed_delta_by_control_type={
            control_type: torch.full((v25.SOURCE_WEIGHT_DIM,), 0.001)
            for control_type in [
                "v21_baseline",
                "v22_baseline",
                "v23_baseline",
                "nearest_train_delta",
                "teacher_oracle_delta",
                "contrastive_weight_arithmetic",
            ]
        },
        provenance={"split": "development", "source": "train-only"},
    )

    redacted = v25.redact_v25_control_context_for_progress(context)
    redacted_text = json.dumps(redacted, sort_keys=True)

    assert redacted["context_hash"] == context["context_hash"]
    assert redacted["precomputed_delta_hash_by_control_type"] == (
        context["precomputed_delta_hash_by_control_type"]
    )
    assert redacted["shuffled_target_descriptor_hash"] == (
        context["shuffled_target_descriptor_hash"]
    )
    assert "precomputed_delta_by_control_type" not in redacted
    assert "shuffled_target_descriptor" not in redacted
    assert "tensor" not in redacted_text.lower()


def test_v25_train_delta_bank_builds_nonidentity_train_only_deltas() -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "train-delta-bank-subject",
        "weights": weights.tolist(),
    }
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
        "train_statistics_hash": "a" * 64,
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
    }
    cache_entry = v25.compute_jacobian_cache_entry(
        record,
        source_behavior="sorted_ascending",
        train_stats=train_stats,
        script_sha256="script",
    )
    for index, pattern in enumerate(v25.PATTERNS):
        train_stats["target_activation_descriptor_by_behavior"][pattern] = (
            cache_entry["source_descriptor"] + 0.01 * (index + 1)
        )

    bank = v25.build_v25_train_delta_bank(
        train_subjects=[record],
        train_stats=train_stats,
        config={"compat_weight": 0.1, "projection": "none", "ridge_lambda": 1e-4},
        norm_cap=0.25,
        script_sha256="script",
    )

    assert len(bank["entries"]) == len(v25.PATTERNS) - 1
    assert len(bank["bank_hash"]) == 64
    for entry in bank["entries"]:
        assert entry["source_behavior"] == "sorted_ascending"
        assert entry["target_behavior"] != "sorted_ascending"
        assert entry["delta"].shape == (v25.SOURCE_WEIGHT_DIM,)
        assert torch.isfinite(entry["delta"]).all()
        assert len(entry["delta_sha256"]) == 64
        assert len(entry["subject_id_hash"]) == 64


def test_v25_train_delta_bank_supports_hash_bound_spectral_projection() -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "train-delta-bank-spectral-subject",
        "weights": weights.tolist(),
    }
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
        "train_statistics_hash": "a" * 64,
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
    }
    cache_entry = v25.compute_jacobian_cache_entry(
        record,
        source_behavior="sorted_ascending",
        train_stats=train_stats,
        script_sha256="script",
    )
    for index, pattern in enumerate(v25.PATTERNS):
        train_stats["target_activation_descriptor_by_behavior"][pattern] = (
            cache_entry["source_descriptor"] + 0.01 * (index + 1)
        )
    spectral_basis = torch.eye(v25.SOURCE_WEIGHT_DIM, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match="spectral basis is required"):
        v25.build_v25_train_delta_bank(
            train_subjects=[record],
            train_stats=train_stats,
            config={
                "compat_weight": 0.1,
                "projection": "spectral_rank4",
                "ridge_lambda": 1e-4,
            },
            norm_cap=0.25,
            script_sha256="script",
        )

    bank = v25.build_v25_train_delta_bank(
        train_subjects=[record],
        train_stats=train_stats,
        config={
            "compat_weight": 0.1,
            "projection": "spectral_rank4",
            "ridge_lambda": 1e-4,
        },
        norm_cap=0.25,
        script_sha256="script",
        spectral_basis=spectral_basis,
    )

    assert bank["spectral_basis_sha256"] == v25.stable_hash_json(
        v25.tensor_to_hashable(spectral_basis)
    )
    assert len(bank["bank_hash"]) == 64
    assert len(bank["entries"]) == len(v25.PATTERNS) - 1
    assert {
        entry["editor_audit"]["projection"]
        for entry in bank["entries"]
    } == {"spectral_rank4"}


def test_v25_empirical_task_vector_bank_uses_train_weight_centroids_only() -> None:
    train_subjects = [
        {
            "pattern": "sorted_ascending",
            "subject_id": "a1",
            "weights": [1.0] * v25.SOURCE_WEIGHT_DIM,
        },
        {
            "pattern": "sorted_ascending",
            "subject_id": "a2",
            "weights": [3.0] * v25.SOURCE_WEIGHT_DIM,
        },
        {
            "pattern": "sorted_descending",
            "subject_id": "d1",
            "weights": [5.0] * v25.SOURCE_WEIGHT_DIM,
        },
        {
            "pattern": "sorted_descending",
            "subject_id": "d2",
            "weights": [7.0] * v25.SOURCE_WEIGHT_DIM,
        },
    ]

    bank = v25.build_v25_empirical_task_vector_bank(
        train_subjects=train_subjects,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        config={
            "compat_weight": 0.0,
            "matched_edit_source": "empirical_centroid_task_vector",
            "projection": "none",
            "ridge_lambda": 1e-5,
        },
        norm_cap=1000.0,
        script_sha256="script",
    )

    entry = next(
        item for item in bank["entries"]
        if item["source_behavior"] == "sorted_ascending"
        and item["target_behavior"] == "sorted_descending"
    )
    assert torch.allclose(entry["delta"], torch.full((v25.SOURCE_WEIGHT_DIM,), 4.0))
    assert entry["source_count"] == 2
    assert entry["target_count"] == 2
    assert len(entry["source_centroid_sha256"]) == 64
    assert len(entry["target_centroid_sha256"]) == 64
    assert len(entry["delta_sha256"]) == 64
    assert bank["train_pool_file_sha256"] == "a" * 64
    assert bank["train_pool_summary_hash"] == "b" * 64
    assert len(bank["bank_hash"]) == 64


def test_v25_empirical_task_vector_bank_requires_train_provenance() -> None:
    with pytest.raises(ValueError, match="train_pool_file_sha256"):
        v25.build_v25_empirical_task_vector_bank(
            train_subjects=[],
            train_pool_file_sha256="not-a-sha",
            train_pool_summary_hash="b" * 64,
            config={
                "compat_weight": 0.0,
                "matched_edit_source": "empirical_centroid_task_vector",
                "projection": "none",
                "ridge_lambda": 1e-5,
            },
            norm_cap=0.25,
            script_sha256="script",
        )


def test_v25_empirical_task_vector_bank_progress_is_hash_only(tmp_path: Path) -> None:
    train_subjects = [
        {
            "pattern": "sorted_ascending",
            "subject_id": "secret-a",
            "weights": [1.0] * v25.SOURCE_WEIGHT_DIM,
        },
        {
            "pattern": "sorted_descending",
            "subject_id": "secret-d",
            "weights": [2.0] * v25.SOURCE_WEIGHT_DIM,
        },
    ]
    log_path = tmp_path / "progress.jsonl"

    bank = v25.build_v25_empirical_task_vector_bank_with_progress(
        train_subjects=train_subjects,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        config={
            "compat_weight": 0.0,
            "matched_edit_source": "empirical_centroid_task_vector",
            "projection": "none",
            "ridge_lambda": 1e-5,
        },
        norm_cap=0.25,
        script_sha256="script",
        progress_log_path=log_path,
        started_at_monotonic=1.0,
    )
    text = log_path.read_text()

    assert "empirical_task_vector_bank_start" in text
    assert "empirical_task_vector_bank_completed" in text
    assert "v26_empirical_task_vector_editor" in text
    assert "secret-a" not in text
    assert "secret-d" not in text
    assert "weights" not in text
    assert "delta" not in text
    assert len(bank["bank_hash"]) == 64


def test_v25_train_delta_bank_redaction_and_progress_are_hash_only(tmp_path: Path) -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "train-delta-progress-subject",
        "weights": weights.tolist(),
    }
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
        "train_statistics_hash": "a" * 64,
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
    }
    cache_entry = v25.compute_jacobian_cache_entry(
        record,
        source_behavior="sorted_ascending",
        train_stats=train_stats,
        script_sha256="script",
    )
    for index, pattern in enumerate(v25.PATTERNS):
        train_stats["target_activation_descriptor_by_behavior"][pattern] = (
            cache_entry["source_descriptor"] + 0.01 * (index + 1)
        )
    progress_log = tmp_path / "development_progress.jsonl"

    bank = v25.build_v25_train_delta_bank_with_progress(
        train_subjects=[record],
        train_stats=train_stats,
        config={"compat_weight": 0.1, "projection": "none", "ridge_lambda": 1e-4},
        norm_cap=0.25,
        script_sha256="script",
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 12.0,
    )
    redacted = v25.redact_v25_train_delta_bank_summary(bank)
    log_text = progress_log.read_text()
    redacted_text = json.dumps(redacted, sort_keys=True)

    assert redacted["entry_count"] == len(v25.PATTERNS) - 1
    assert redacted["bank_hash"] == bank["bank_hash"]
    assert "entries" not in redacted
    assert "train_edit_bank_progress" in log_text
    assert '"train_edit_bank_role": "actual"' in log_text
    assert "processed_train_subject_count" in log_text
    assert "delta" not in redacted_text
    assert "delta" not in log_text
    assert "weights" not in log_text
    assert "subject_id" not in log_text


def test_v25_train_delta_bank_requires_hash_bound_train_statistics() -> None:
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "missing-train-hash-subject",
        "weights": torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM).tolist(),
    }
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
    }

    with pytest.raises(ValueError, match="train_statistics_hash"):
        v25.build_v25_train_delta_bank(
            train_subjects=[record],
            train_stats=train_stats,
            config={"compat_weight": 0.1, "projection": "none", "ridge_lambda": 1e-4},
            norm_cap=0.25,
            script_sha256="script",
        )


def test_v25_train_only_control_context_builds_full_hash_bound_contract() -> None:
    source_descriptor = torch.linspace(-0.2, 0.2, v25.ACTIVATION_DESCRIPTOR_DIM)
    bank_entries = []
    for index, scale in enumerate([0.01, 0.02, 0.03]):
        delta = torch.full((v25.SOURCE_WEIGHT_DIM,), scale, dtype=torch.float32)
        bank_entries.append({
            "delta": delta,
            "delta_sha256": v25.stable_hash_json(v25.tensor_to_hashable(delta)),
            "direction": "sorted_ascending->mountain_pattern",
            "source_behavior": "sorted_ascending",
            "source_descriptor": source_descriptor + float(index),
            "subject_id_hash": v25.stable_hash_json(f"train-{index}"),
            "target_behavior": "mountain_pattern",
        })
    bank = {
        "bank_hash": "b" * 64,
        "entries": bank_entries,
        "entry_count": len(bank_entries),
        "entry_hashes": [
            v25.stable_hash_json({
                "delta_sha256": entry["delta_sha256"],
                "direction": entry["direction"],
                "subject_id_hash": entry["subject_id_hash"],
            })
            for entry in bank_entries
        ],
    }
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "probe_examples": [{"sequence": [0, 1, 2, 3, 4]}],
        "train_statistics_hash": "a" * 64,
        "target_activation_descriptor_by_behavior": {
            pattern: torch.full((v25.ACTIVATION_DESCRIPTOR_DIM,), 0.1 * index)
            for index, pattern in enumerate(v25.PATTERNS)
        },
    }
    job = {
        "direction": "sorted_ascending->mountain_pattern",
        "record_id": "development-record",
        "source_behavior": "sorted_ascending",
        "subject": {
            "pattern": "sorted_ascending",
            "subject_id": "development-record",
            "weights": torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM).tolist(),
        },
        "target_behavior": "mountain_pattern",
    }

    context = v25.build_v25_train_only_control_context(
        job=job,
        train_delta_bank=bank,
        train_stats=train_stats,
        job_plan_hash="c" * 64,
        selected_config_hash="d" * 64,
    )
    redacted = v25.redact_v25_control_context_for_progress(context)
    redacted_text = json.dumps(redacted, sort_keys=True)

    assert set(context["precomputed_delta_by_control_type"]) == set(
        v25.required_v25_precomputed_delta_control_types()
    )
    assert len(context["context_hash"]) == 64
    assert context["provenance"]["control_context_mode"] == "train_only_edit_bank"
    assert context["provenance"]["train_edit_bank_hash"] == "b" * 64
    assert context["provenance"]["train_statistics_hash"] == "a" * 64
    assert "precomputed_delta_by_control_type" not in redacted
    assert "weights" not in redacted_text
    assert "subject_id" not in redacted_text


def test_v25_train_only_control_context_requires_hash_bound_bank() -> None:
    job = {
        "direction": "sorted_ascending->mountain_pattern",
        "record_id": "development-record",
        "source_behavior": "sorted_ascending",
        "subject": {
            "pattern": "sorted_ascending",
            "subject_id": "development-record",
            "weights": torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM).tolist(),
        },
        "target_behavior": "mountain_pattern",
    }

    with pytest.raises(ValueError, match="train_delta_bank.bank_hash"):
        v25.build_v25_train_only_control_context(
            job=job,
            train_delta_bank={"entries": []},
            train_stats={
                "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
                "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
                "probe_examples": [{"sequence": [0, 1, 2, 3, 4]}],
                "train_statistics_hash": "a" * 64,
                "target_activation_descriptor_by_behavior": {
                    pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
                    for pattern in v25.PATTERNS
                },
            },
            job_plan_hash="c" * 64,
            selected_config_hash="d" * 64,
        )


def test_v25_native_controls_emit_full_control_contract_without_weights() -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "native-control-subject",
        "weights": weights.tolist(),
    }
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
    }
    cache_entry = v25.compute_jacobian_cache_entry(
        record,
        source_behavior="sorted_ascending",
        train_stats=train_stats,
        script_sha256="script",
    )
    train_stats["target_activation_descriptor_by_behavior"]["sorted_descending"] = (
        cache_entry["source_descriptor"] + 0.05
    )
    matched = v25.evaluate_v25_matched_edit(
        subject=record,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        train_stats=train_stats,
        cache_entry=cache_entry,
        config={"compat_weight": 0.1, "projection": "rank1", "ridge_lambda": 1e-4},
        norm_cap=0.25,
    )

    control_context = v25.build_v25_control_context(
        precomputed_delta_by_control_type={
            control_type: torch.full((v25.SOURCE_WEIGHT_DIM,), 0.001 * (index + 1))
            for index, control_type in enumerate([
                "v21_baseline",
                "v22_baseline",
                "v23_baseline",
                "nearest_train_delta",
                "teacher_oracle_delta",
                "contrastive_weight_arithmetic",
            ])
        },
        provenance={"split": "development", "source": "train-only"},
        shuffled_target_descriptor=cache_entry["source_descriptor"] + 0.025,
    )
    spectral_basis = torch.eye(v25.SOURCE_WEIGHT_DIM, 4, dtype=torch.float32)
    controls = v25.build_v25_native_controls(
        subject=record,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        train_stats=train_stats,
        cache_entry=cache_entry,
        config={"compat_weight": 0.1, "projection": "rank1", "ridge_lambda": 1e-4},
        norm_cap=0.25,
        matched_delta_sha256=matched["editor"]["delta_sha256"],
        matched_delta_norm=matched["delta_norm"],
        matched_spectral_projection_norm=0.123,
        selected_config_hash="config",
        spectral_basis=spectral_basis,
        control_context=control_context,
    )
    control_by_type = {control["control_type"]: control for control in controls}

    assert [control["control_type"] for control in controls] == v25.expected_v25_control_types()
    v25.validate_v25_controls(controls)
    for control_type in v25.expected_v25_control_types():
        assert control_type in control_by_type
        assert "weights" not in control_by_type[control_type]
        assert len(control_by_type[control_type]["editor"]["delta_sha256"]) == 64
    no_signature_trained = control_by_type["no_signature_trained"]
    assert no_signature_trained["editor"]["selected_config_hash"] == "config"
    assert len(no_signature_trained["editor"]["zero_descriptor_norm_hash"]) == 64
    shuffled = control_by_type["shuffled_signature"]
    assert shuffled["editor"]["context_hash"] == control_context["context_hash"]
    assert len(shuffled["editor"]["shuffled_target_descriptor_hash"]) == 64
    v21_baseline = control_by_type["v21_baseline"]
    assert v21_baseline["editor"]["context_hash"] == control_context["context_hash"]
    assert len(v21_baseline["editor"]["precomputed_delta_sha256"]) == 64
    spectral = control_by_type["spectral_basis_random_coefficients"]
    assert spectral["delta_norm"] == pytest.approx(0.123)
    assert spectral["editor"]["matched_spectral_projection_norm"] == pytest.approx(0.123)


def test_v25_native_controls_require_hash_bound_control_context() -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "missing-context-hash-subject",
        "weights": weights.tolist(),
    }
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
    }
    cache_entry = v25.compute_jacobian_cache_entry(
        record,
        source_behavior="sorted_ascending",
        train_stats=train_stats,
        script_sha256="script",
    )
    train_stats["target_activation_descriptor_by_behavior"]["sorted_descending"] = (
        cache_entry["source_descriptor"] + 0.05
    )
    control_context = {
        "precomputed_delta_by_control_type": {
            control_type: torch.zeros(v25.SOURCE_WEIGHT_DIM)
            for control_type in [
                "v21_baseline",
                "v22_baseline",
                "v23_baseline",
                "nearest_train_delta",
                "teacher_oracle_delta",
                "contrastive_weight_arithmetic",
            ]
        },
        "shuffled_target_descriptor": cache_entry["source_descriptor"] + 0.025,
    }

    with pytest.raises(ValueError, match="context_hash"):
        v25.build_v25_native_controls(
            subject=record,
            source_behavior="sorted_ascending",
            target_behavior="sorted_descending",
            train_stats=train_stats,
            cache_entry=cache_entry,
            config={"compat_weight": 0.1, "projection": "rank1", "ridge_lambda": 1e-4},
            norm_cap=0.25,
            matched_delta_sha256="b" * 64,
            matched_delta_norm=0.5,
            matched_spectral_projection_norm=0.25,
            selected_config_hash="config",
            spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4, dtype=torch.float32),
            control_context=control_context,
        )


def test_v25_development_job_evaluator_builds_full_weightless_proof_record() -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "job-eval-subject",
        "weights": weights.tolist(),
    }
    job = v25.build_v25_development_jobs([record])[0]
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
    }
    cache_entry = v25.compute_jacobian_cache_entry(
        record,
        source_behavior="sorted_ascending",
        train_stats=train_stats,
        script_sha256="script",
    )
    train_stats["target_activation_descriptor_by_behavior"][job["target_behavior"]] = (
        cache_entry["source_descriptor"] + 0.05
    )
    control_context = v25.build_v25_control_context(
        precomputed_delta_by_control_type={
            control_type: torch.zeros(v25.SOURCE_WEIGHT_DIM)
            for control_type in v25.required_v25_precomputed_delta_control_types()
        },
        provenance={"split": "development", "source": "unit-test"},
        shuffled_target_descriptor=cache_entry["source_descriptor"] + 0.025,
    )

    proof_record = v25.evaluate_v25_development_job(
        job=job,
        train_stats=train_stats,
        config={"compat_weight": 0.1, "projection": "rank1", "ridge_lambda": 1e-4},
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4, dtype=torch.float32),
        control_context=control_context,
        selected_config_hash="c" * 64,
        script_sha256="script",
    )
    proof_text = json.dumps(proof_record, sort_keys=True)

    assert proof_record["direction"] == job["direction"]
    assert proof_record["matched"]["control_type"] == v25.EDITOR_METHOD
    assert len(proof_record["controls"]) == v25.EXPECTED_CONTROLS_PER_RECORD
    assert proof_record["record_id_hash"] == v25.stable_hash_json(job["record_id"])
    assert proof_record["control_context_hash"] == control_context["context_hash"]
    assert "weights" not in proof_text
    assert "subject_id" not in proof_text


def test_v25_development_job_rejects_pinv_fallback_for_proof_context() -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "proof-context-fallback-subject",
        "weights": weights.tolist(),
    }
    job = v25.build_v25_development_jobs([record])[0]
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
    }
    cache_entry = v25.compute_jacobian_cache_entry(
        record,
        source_behavior="sorted_ascending",
        train_stats=train_stats,
        script_sha256="script",
    )
    train_stats["target_activation_descriptor_by_behavior"][job["target_behavior"]] = (
        cache_entry["source_descriptor"] + 0.05
    )
    control_context = v25.build_v25_control_context(
        precomputed_delta_by_control_type={
            control_type: torch.zeros(v25.SOURCE_WEIGHT_DIM)
            for control_type in v25.required_v25_precomputed_delta_control_types()
        },
        provenance={"split": "development", "source": "unit-test"},
        shuffled_target_descriptor=cache_entry["source_descriptor"] + 0.025,
    )

    with pytest.raises(ValueError, match="dry-run-only"):
        v25.evaluate_v25_development_job(
            job=job,
            train_stats=train_stats,
            config={
                "allow_pinv_fallback": True,
                "compat_weight": 0.1,
                "projection": "rank1",
                "ridge_lambda": 1e-4,
            },
            norm_cap=0.25,
            spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4, dtype=torch.float32),
            control_context=control_context,
            selected_config_hash="c" * 64,
            script_sha256="script",
        )


def test_v25_development_job_evaluator_progress_logs_redacted_summary(tmp_path: Path) -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "job-progress-subject",
        "weights": weights.tolist(),
    }
    job = v25.build_v25_development_jobs([record])[0]
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
    }
    cache_entry = v25.compute_jacobian_cache_entry(
        record,
        source_behavior="sorted_ascending",
        train_stats=train_stats,
        script_sha256="script",
    )
    train_stats["target_activation_descriptor_by_behavior"][job["target_behavior"]] = (
        cache_entry["source_descriptor"] + 0.05
    )
    control_context = v25.build_v25_control_context(
        precomputed_delta_by_control_type={
            control_type: torch.zeros(v25.SOURCE_WEIGHT_DIM)
            for control_type in v25.required_v25_precomputed_delta_control_types()
        },
        provenance={"split": "development", "source": "unit-test"},
        shuffled_target_descriptor=cache_entry["source_descriptor"] + 0.025,
    )
    progress_log = tmp_path / "development_progress.jsonl"

    proof_record = v25.evaluate_v25_development_job_with_progress(
        job=job,
        train_stats=train_stats,
        config={"compat_weight": 0.1, "projection": "rank1", "ridge_lambda": 1e-4},
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4, dtype=torch.float32),
        control_context=control_context,
        selected_config_hash="c" * 64,
        script_sha256="script",
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        job_index=0,
        total_jobs=1,
        now_monotonic=lambda: 12.0,
    )
    events = [json.loads(line) for line in progress_log.read_text().splitlines()]
    event_text = json.dumps(events, sort_keys=True)

    assert [event["event"] for event in events] == [
        "development_evaluation_record_start",
        "development_evaluation_record_completed",
    ]
    assert events[0]["record_id_hash"] == proof_record["record_id_hash"]
    assert events[1]["record_id_hash"] == proof_record["record_id_hash"]
    assert events[1]["control_count"] == v25.EXPECTED_CONTROLS_PER_RECORD
    assert "proof_gate_diagnostics" in events[1]
    assert "compatible_mse_pass" in events[1]["proof_gate_diagnostics"]
    assert len(events[1]["proof_gate_diagnostics"]["failed_control_types_hash"]) == 64
    assert "weights" not in event_text
    assert "subject_id" not in event_text
    assert "subject" not in event_text


def test_v25_bounded_development_jobs_runner_respects_max_jobs_and_logs(tmp_path: Path) -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "bounded-run-subject",
        "weights": weights.tolist(),
    }
    jobs = v25.build_v25_development_jobs([record])
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
    }
    cache_entry = v25.compute_jacobian_cache_entry(
        record,
        source_behavior="sorted_ascending",
        train_stats=train_stats,
        script_sha256="script",
    )
    for job in jobs:
        train_stats["target_activation_descriptor_by_behavior"][job["target_behavior"]] = (
            cache_entry["source_descriptor"] + 0.05
        )
    control_context = v25.build_v25_control_context(
        precomputed_delta_by_control_type={
            control_type: torch.zeros(v25.SOURCE_WEIGHT_DIM)
            for control_type in v25.required_v25_precomputed_delta_control_types()
        },
        provenance={"split": "development", "source": "unit-test"},
        shuffled_target_descriptor=cache_entry["source_descriptor"] + 0.025,
    )
    context_by_record_hash = {
        v25.stable_hash_json(job["record_id"]): control_context
        for job in jobs
    }
    progress_log = tmp_path / "development_progress.jsonl"

    result = v25.evaluate_v25_development_jobs_with_progress(
        jobs=jobs,
        max_jobs=1,
        train_stats=train_stats,
        config={"compat_weight": 0.1, "projection": "rank1", "ridge_lambda": 1e-4},
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4, dtype=torch.float32),
        control_context_by_record_hash=context_by_record_hash,
        selected_config_hash="c" * 64,
        script_sha256="script",
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 12.0,
    )
    events = [json.loads(line) for line in progress_log.read_text().splitlines()]
    event_text = json.dumps(events, sort_keys=True)

    assert result["evaluated_count"] == 1
    assert result["max_jobs"] == 1
    assert result["total_planned_jobs"] == 3
    assert len(result["proof_record_hashes"]) == 1
    assert [event["event"] for event in events] == [
        "development_evaluation_start",
        "development_evaluation_record_start",
        "development_evaluation_record_completed",
        "development_evaluation_completed",
    ]
    assert events[-1]["evaluated_count"] == 1
    assert events[-1]["proof_record_hashes"] == result["proof_record_hashes"]
    assert "weights" not in event_text
    assert "subject_id" not in event_text
    assert "subject" not in event_text


def test_v25_placeholder_control_context_is_marked_dry_run_only() -> None:
    source_descriptor = torch.linspace(-0.1, 0.1, v25.ACTIVATION_DESCRIPTOR_DIM)

    context = v25.build_v25_placeholder_control_context_for_dry_run(
        source_descriptor=source_descriptor,
        record_id_hash="a" * 64,
        job_plan_hash="b" * 64,
    )
    redacted = v25.redact_v25_control_context_for_progress(context)

    assert context["provenance"]["dry_run_only"] is True
    assert context["provenance"]["proof_valid"] is False
    assert context["provenance"]["control_context_mode"] == "placeholder_zero_controls"
    assert redacted["provenance"]["proof_valid"] is False
    assert all(
        torch.count_nonzero(delta).item() == 0
        for delta in context["precomputed_delta_by_control_type"].values()
    )


def test_v25_bounded_dry_run_result_is_explicitly_not_proof(tmp_path: Path) -> None:
    weights = torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "dry-run-subject",
        "weights": weights.tolist(),
    }
    jobs = v25.build_v25_development_jobs([record])
    job_summary = v25.redact_v25_development_job_summary(jobs)
    probe_examples = [{"sequence": [0, 1, 2, 3, 4]}]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": probe_examples,
        "probe_examples_hash": v25.stable_hash_json(probe_examples),
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
    }
    cache_entry = v25.compute_jacobian_cache_entry(
        record,
        source_behavior="sorted_ascending",
        train_stats=train_stats,
        script_sha256="script",
    )
    for job in jobs:
        train_stats["target_activation_descriptor_by_behavior"][job["target_behavior"]] = (
            cache_entry["source_descriptor"] + 0.05
        )
    context_by_record_hash = {}
    for job in jobs:
        record_id_hash = v25.stable_hash_json(job["record_id"])
        context_by_record_hash[record_id_hash] = (
            v25.build_v25_placeholder_control_context_for_dry_run(
                source_descriptor=cache_entry["source_descriptor"],
                record_id_hash=record_id_hash,
                job_plan_hash=job_summary["job_plan_hash"],
            )
        )
    progress_log = tmp_path / "development_progress.jsonl"

    result = v25.evaluate_v25_bounded_development_dry_run_with_progress(
        jobs=jobs,
        max_jobs=1,
        train_stats=train_stats,
        config={"compat_weight": 0.1, "projection": "rank1", "ridge_lambda": 1e-4},
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4, dtype=torch.float32),
        control_context_by_record_hash=context_by_record_hash,
        selected_config_hash="c" * 64,
        script_sha256="script",
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 12.0,
    )
    event_text = progress_log.read_text()

    assert result["stage"] == "development_bounded_dry_run_completed"
    assert result["proof_valid"] is False
    assert result["evaluated_count"] == 1
    assert "placeholder_zero_controls" in event_text
    assert "weights" not in event_text
    assert "subject_id" not in event_text


def test_v25_development_setup_runs_real_bounded_jobs_without_dry_run_flag(
    tmp_path: Path,
) -> None:
    pool_dir = tmp_path / "pools"
    output_dir = tmp_path / "out"
    pool_dir.mkdir()
    output_dir.mkdir()
    (pool_dir / "train_subjects.json").write_text(json.dumps([
        {
            "pattern": pattern,
            "subject_id": f"train-{pattern}",
            "weights": torch.linspace(
                -0.05 + 0.01 * index,
                0.05 + 0.01 * index,
                v25.SOURCE_WEIGHT_DIM,
            ).tolist(),
        }
        for index, pattern in enumerate(v25.PATTERNS)
    ]))
    (pool_dir / "development_subjects.json").write_text(json.dumps([
        {
            "pattern": "sorted_ascending",
            "subject_id": "dev",
            "weights": torch.linspace(-0.03, 0.07, v25.SOURCE_WEIGHT_DIM).tolist(),
        }
    ]))

    result = v25.run_v25_development_setup(
        pool_dir=pool_dir,
        output_dir=output_dir,
        progress_log_path=output_dir / "development_progress.jsonl",
        started_at_monotonic=10.0,
        max_development_jobs=1,
        dry_run_placeholder_controls=False,
    )
    progress_text = (output_dir / "development_progress.jsonl").read_text()

    assert "development_evaluation" in result
    assert result["development_evaluation"]["evaluated_count"] == 1
    assert result["development_job_selection"]["selected_job_count"] == 1
    assert len(result["development_job_selection"]["selection_hash"]) == 64
    assert "dry_run" not in result
    assert result["train_edit_spectral_basis"]["edit_vector_count"] == len(v25.PATTERNS) * (
        len(v25.PATTERNS) - 1
    )
    assert "delta_count" not in result["train_edit_spectral_basis"]
    assert result["train_edit_bank"]["entry_count"] == len(v25.PATTERNS) * (
        len(v25.PATTERNS) - 1
    )
    assert "development_jobs_selected" in progress_text
    assert "development_bounded_real_completed" in progress_text
    assert "placeholder_zero_controls" not in progress_text
    assert "weights" not in progress_text
    assert "subject_id" not in progress_text


def test_v25_development_setup_runs_inner_validation_path_with_redacted_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_dir = tmp_path / "pools"
    output_dir = tmp_path / "out"
    pool_dir.mkdir()
    output_dir.mkdir()
    (pool_dir / "train_subjects.json").write_text(json.dumps([
        {
            "pattern": pattern,
            "subject_id": f"train-{pattern}",
            "weights": torch.linspace(
                -0.05 + 0.01 * index,
                0.05 + 0.01 * index,
                v25.SOURCE_WEIGHT_DIM,
            ).tolist(),
        }
        for index, pattern in enumerate(v25.PATTERNS)
    ]))
    (pool_dir / "development_subjects.json").write_text(json.dumps([
        {
            "pattern": pattern,
            "subject_id": f"dev-{pattern}",
            "weights": torch.linspace(
                -0.03 + 0.01 * index,
                0.07 + 0.01 * index,
                v25.SOURCE_WEIGHT_DIM,
            ).tolist(),
        }
        for index, pattern in enumerate(v25.PATTERNS)
    ]))

    def fake_inner_validation(**kwargs):
        assert len(kwargs["configs"]) == 2
        assert kwargs["rung_job_counts"] == [2]
        assert kwargs["keep_fractions"] == [0.5]
        return {
            "best_candidate": {
                "config_hash": "a" * 64,
                "config_index": 0,
                "invalid": False,
                "mean_matched_minus_best_control_target_margin": 0.1,
                "mean_matched_minus_shuffled_signature_target_margin": 0.1,
                "mean_target_margin": 0.5,
                "pareto_undominated_rate": 1.0,
                "proof_gate_failure_count": 0,
                "record_count": 2,
                "target_prediction_rate": 1.0,
            },
            "candidate_count": 2,
            "plan": {
                "plan_hash": "b" * 64,
                "rung_count": 1,
            },
            "rungs": [],
            "stage": "inner_validation_completed",
        }

    monkeypatch.setattr(
        v25,
        "run_v25_inner_validation_successive_halving_with_progress",
        fake_inner_validation,
    )

    result = v25.run_v25_development_setup(
        pool_dir=pool_dir,
        output_dir=output_dir,
        progress_log_path=output_dir / "development_progress.jsonl",
        started_at_monotonic=10.0,
        run_inner_validation=True,
        inner_validation_rung_jobs=[2],
        inner_validation_keep_fractions=[0.5],
        inner_validation_max_configs=2,
        development_job_selection="balanced-directions",
    )
    progress_text = (output_dir / "development_progress.jsonl").read_text()

    assert result["inner_validation"]["stage"] == "inner_validation_completed"
    assert result["development_job_selection"]["selected_job_count"] == 2
    assert "development_jobs_selected" in progress_text
    assert "weights" not in progress_text
    assert "subject_id" not in progress_text


def test_v25_development_setup_runs_explicit_placeholder_dry_run(tmp_path: Path) -> None:
    pool_dir = tmp_path / "pools"
    output_dir = tmp_path / "out"
    pool_dir.mkdir()
    output_dir.mkdir()
    train_subjects = [
        {
            "pattern": pattern,
            "subject_id": f"train-{pattern}",
            "weights": torch.linspace(
                -0.05 + 0.01 * index,
                0.05 + 0.01 * index,
                v25.SOURCE_WEIGHT_DIM,
            ).tolist(),
        }
        for index, pattern in enumerate(v25.PATTERNS)
    ]
    development_subjects = [{
        "pattern": "sorted_ascending",
        "subject_id": "dev",
        "weights": torch.linspace(-0.05, 0.05, v25.SOURCE_WEIGHT_DIM).tolist(),
    }]
    (pool_dir / "train_subjects.json").write_text(json.dumps(train_subjects))
    (pool_dir / "development_subjects.json").write_text(json.dumps(development_subjects))
    progress_log = output_dir / "development_progress.jsonl"

    result = v25.run_v25_development_setup(
        pool_dir=pool_dir,
        output_dir=output_dir,
        progress_log_path=progress_log,
        started_at_monotonic=10.0,
        now_monotonic=lambda: 12.0,
        max_development_jobs=1,
        dry_run_placeholder_controls=True,
    )
    event_text = progress_log.read_text()

    assert result["dry_run"]["stage"] == "development_bounded_dry_run_completed"
    assert result["dry_run"]["proof_valid"] is False
    assert result["dry_run"]["evaluated_count"] == 1
    assert result["stage"] == "development_setup_completed"
    assert "development_bounded_dry_run_completed" in event_text
    assert "placeholder_zero_controls" in event_text
    assert "weights" not in event_text
    assert "subject_id" not in event_text


def test_v25_recursive_numeric_finiteness_detects_nested_nonfinite() -> None:
    payload = {
        "behavior_margins": {
            "sorted_ascending": 0.1,
            "sorted_descending": float("nan"),
        },
        "controls": [
            {"target_margin": 0.2},
            {"compatible_source_output_mse": float("inf")},
        ],
        "label": "not numeric",
    }

    failures = v25.recursive_numeric_finiteness_failures(payload)

    assert "behavior_margins.sorted_descending" in failures
    assert "controls[1].compatible_source_output_mse" in failures
    assert all("label" not in failure for failure in failures)


def make_v25_control(control_type: str, *, target_margin: float = 0.20) -> dict:
    return {
        "compatible_source_output_mse": 0.11,
        "control_type": control_type,
        "predicted_behavior": "sorted_ascending",
        "target_margin": target_margin,
    }


def make_v25_matched(
    *,
    target_margin: float = 0.30,
    compatible_mse: float = 0.10,
    predicted_behavior: str = "sorted_descending",
) -> dict:
    return {
        "behavior_margins": {
            "has_majority": -0.1,
            "mountain_pattern": -0.1,
            "sorted_ascending": -0.1,
            "sorted_descending": target_margin,
        },
        "compatible_source_output_mse": compatible_mse,
        "control_type": v25.EDITOR_METHOD,
        "delta_norm": 0.2,
        "predicted_behavior": predicted_behavior,
        "source_margin": -0.1,
        "target_margin": target_margin,
        "target_prediction_pass": predicted_behavior == "sorted_descending",
    }


def make_v25_controls(*, shuffled_margin: float = 0.20) -> list[dict]:
    controls = []
    for control_type in v25.expected_v25_control_types():
        margin = shuffled_margin if control_type == "shuffled_signature" else 0.20
        controls.append(make_v25_control(control_type, target_margin=margin))
    return controls


def test_v25_proof_record_computes_pareto_and_gate_summaries() -> None:
    record = v25.build_v25_proof_record(
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        matched=make_v25_matched(),
        controls=make_v25_controls(),
    )

    assert record["summary"]["target_prediction_pass"] is True
    assert record["summary"]["pareto_undominated"] is True
    assert record["summary"]["individual_all_gates_passed"] is True
    assert record["summary"]["matched_minus_best_control_target_margin"] == pytest.approx(0.10)
    assert record["summary"]["matched_minus_shuffled_signature_target_margin"] == pytest.approx(0.10)
    assert record["matched"]["matched_minus_v21_baseline_target_margin"] == pytest.approx(0.10)


def test_v25_proof_record_exposes_redacted_gate_decomposition() -> None:
    record = v25.build_v25_proof_record(
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        matched=make_v25_matched(target_margin=0.30, compatible_mse=0.20),
        controls=make_v25_controls(shuffled_margin=0.28),
    )

    diagnostics = record["summary"]["proof_gate_diagnostics"]
    diagnostic_text = json.dumps(diagnostics, sort_keys=True)

    assert diagnostics["target_prediction_pass"] is True
    assert diagnostics["target_margin_pass"] is True
    assert diagnostics["pareto_undominated"] is True
    assert diagnostics["compatible_mse_pass"] is False
    assert diagnostics["control_margin_fail_count"] >= 1
    assert diagnostics["control_margin_pass_count"] >= 1
    assert len(diagnostics["failed_control_types_hash"]) == 64
    assert "failed_control_types" not in diagnostics
    assert "weights" not in diagnostic_text
    assert "subject_id" not in diagnostic_text


def test_v25_proof_record_rejects_bad_control_count_and_nonfinite_metrics() -> None:
    controls = make_v25_controls()
    controls[0]["target_margin"] = float("nan")

    with pytest.raises(ValueError, match="nonfinite"):
        v25.build_v25_proof_record(
            source_behavior="sorted_ascending",
            target_behavior="sorted_descending",
            matched=make_v25_matched(),
            controls=controls,
        )

    with pytest.raises(ValueError, match="control count"):
        v25.build_v25_proof_record(
            source_behavior="sorted_ascending",
            target_behavior="sorted_descending",
            matched=make_v25_matched(),
            controls=make_v25_controls()[:-1],
        )


def test_v25_random_controls_do_not_participate_in_pareto_dominance() -> None:
    controls = make_v25_controls()
    random_control = next(
        control
        for control in controls
        if control["control_type"] == "random_matched_norm_00"
    )
    random_control["target_margin"] = 0.99
    random_control["compatible_source_output_mse"] = 0.0

    record = v25.build_v25_proof_record(
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        matched=make_v25_matched(),
        controls=controls,
    )

    assert record["summary"]["pareto_undominated"] is True
    assert "random_matched_norm_00" not in record["matched"]["pareto_dominator_types"]


def test_v25_proof_record_rejects_duplicate_control_types() -> None:
    controls = make_v25_controls()
    controls[-1] = {**controls[-1], "control_type": "random_matched_norm_00"}

    with pytest.raises(ValueError, match="control type mismatch"):
        v25.build_v25_proof_record(
            source_behavior="sorted_ascending",
            target_behavior="sorted_descending",
            matched=make_v25_matched(),
            controls=controls,
        )


def test_v25_aggregate_records_compute_required_gate_metrics() -> None:
    records = [
        v25.build_v25_proof_record(
            source_behavior="sorted_ascending",
            target_behavior="sorted_descending",
            matched=make_v25_matched(target_margin=0.30),
            controls=make_v25_controls(),
        ),
        v25.build_v25_proof_record(
            source_behavior="sorted_ascending",
            target_behavior="has_majority",
            matched=make_v25_matched(
                target_margin=0.10,
                predicted_behavior="sorted_ascending",
            ),
            controls=make_v25_controls(shuffled_margin=0.08),
        ),
    ]

    result = v25.summarize_v25_records(records, expected_record_count=2)

    assert result["record_count"] == 2
    assert result["aggregate"]["target_prediction_rate"] == pytest.approx(0.5)
    assert result["aggregate"]["individual_all_gate_pass_rate"] == pytest.approx(0.5)
    assert result["aggregate"]["pareto_undominated_rate"] == pytest.approx(1.0)
    assert result["aggregate"]["mean_target_margin"] == pytest.approx(0.20)
    assert result["aggregate"]["mean_matched_minus_best_control_target_margin"] == pytest.approx(0.0)
    assert result["aggregate"]["mean_matched_minus_shuffled_signature_target_margin"] == pytest.approx(0.06)
    breakdown = result["aggregate"]["proof_gate_breakdown"]
    assert breakdown["record_count"] == 2
    assert breakdown["target_prediction_fail_count"] == 1
    assert breakdown["target_margin_fail_count"] == 1
    assert breakdown["control_margin_fail_count"] >= 1
    assert len(breakdown["control_margin_failure_type_counts_hash"]) == 64
    assert result["failures"]
    assert "sorted_ascending->sorted_descending" in result["by_direction"]


def test_v25_seed_preflight_has_no_pool_overlaps() -> None:
    preflight = v25.build_v25_seed_preflight()

    assert preflight["passed"] is True
    assert preflight["failures"] == []
    assert len(preflight["seed_ranges"]) == len(v25.POOL_CONFIGS) * len(v25.PATTERNS)


def test_v25_forbidden_final_redacted_keys_detects_recursive_detail() -> None:
    payload = {
        "claim_scope": v25.FINAL_REDACTED_SCOPE,
        "pool": "final",
        "summary": {"accepted_counts_by_behavior": {}},
        "weights": [1.0],
    }

    failures = v25.forbidden_final_redacted_keys(payload)

    assert "weights" in failures


def test_v25_generate_pools_writes_redacted_audits_and_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_train_counts = {pattern: 64 for pattern in v25.PATTERNS}
    expected_eval_counts = {pattern: 24 for pattern in v25.PATTERNS}

    def fake_generate_pool(*, pool_name, pool_config, **_kwargs):
        counts = expected_train_counts if pool_name == "train" else expected_eval_counts
        return {
            "claim_scope": "unset",
            "config": {},
            "pool": pool_name,
            "records": [],
            "summary": {"accepted_counts_by_behavior": counts},
        }

    def fake_summarize_pool(payload):
        counts = expected_train_counts if payload["pool"] == "train" else expected_eval_counts
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
                "accepted_counts_by_behavior": expected_eval_counts,
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

    monkeypatch.setattr(v25.v23.v16.v15.poolgen, "generate_pool", fake_generate_pool)
    monkeypatch.setattr(
        v25.v23.v16.v15.poolgen,
        "redact_weights_and_signatures",
        lambda payload: payload,
    )
    monkeypatch.setattr(v25.v23.v16.v15.poolgen, "summarize_pool", fake_summarize_pool)
    monkeypatch.setattr(v25.v23.v16.v15.poolgen, "build_final_redacted_summary", fake_final_redacted)
    monkeypatch.setattr(v25.v23.v16.v15.poolgen, "build_combined_audit", fake_combined_audit)
    monkeypatch.setattr(v25.v23.v16.v15.v10, "redact_combined_audit", lambda audit: audit)
    monkeypatch.setattr(v25.v23.v16.v15, "build_suite", lambda *_args: {"suite": "fake"})
    monkeypatch.setattr(
        v25.v23.v16.v15,
        "build_heldout_sequences",
        lambda _suite: {"heldout": "fake"},
    )
    monkeypatch.setattr(
        v25.v23.v16.v15,
        "build_candidate_pools",
        lambda _heldout: {"candidates": "fake"},
    )
    monkeypatch.setattr(
        v25.v23.v16.v15,
        "summarize_candidate_pools",
        lambda _candidate_pools: {"candidate_summary": "fake"},
    )
    monkeypatch.setattr(v25, "build_probe_examples", lambda: [{"sequence": [0, 0, 0, 0, 0]}])

    result = v25.generate_pools(
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
    progress_text = (tmp_path / v25.SOURCE_POOL_PROGRESS_LOG_FILENAME).read_text()
    assert '"event": "preflight_completed"' in progress_text
    assert progress_text.count('"event": "pool_generation_completed"') == 3
    assert '"event": "combined_audit_written"' in progress_text
    for forbidden in v25.RECURSIVE_FORBIDDEN_FINAL_DETAIL_KEY_TERMS:
        assert forbidden not in progress_text
    final_redacted = v25.v23.v16.v15.v1.load_json(tmp_path / "final_redacted_audit.json")
    assert final_redacted["claim_scope"] == v25.FINAL_REDACTED_SCOPE
    assert v25.forbidden_final_redacted_keys(final_redacted) == []


def test_v25_long_run_monitor_snapshot_tracks_progress_without_detail(
    tmp_path: Path,
) -> None:
    progress_log = tmp_path / v25.SOURCE_POOL_PROGRESS_LOG_FILENAME
    time_start = 0.0
    v25.record_progress_event(
        progress_log,
        event="pool_generation_completed",
        started_at_monotonic=time_start,
        extra={"pool": "train", "pool_file_sha256": "abc"},
    )

    snapshot = v25.build_long_run_monitor_snapshot(
        started_at_monotonic=time_start,
        progress_log_path=progress_log,
    )

    assert snapshot["event"] == "monitor_heartbeat"
    assert snapshot["pid"] == v25.os.getpid()
    assert snapshot["progress_line_count"] == 1
    assert "progress_log_path" not in snapshot
    assert "progress_log_location_sha256" in snapshot
    assert snapshot["latest_progress_event"] == "pool_generation_completed"
    assert snapshot["latest_progress_pool"] == "train"
    snapshot_text = json.dumps(snapshot, sort_keys=True)
    for forbidden in v25.RECURSIVE_FORBIDDEN_FINAL_DETAIL_KEY_TERMS:
        assert forbidden not in snapshot_text


def test_v25_long_run_monitor_redacts_forbidden_path_and_event_terms(
    tmp_path: Path,
) -> None:
    progress_log = tmp_path / "weights_signature_progress.jsonl"
    time_start = 0.0
    v25.record_progress_event(
        progress_log,
        event="shuffled_signature_probe",
        started_at_monotonic=time_start,
        extra={"seed_range_count": 12},
    )

    snapshot = v25.build_long_run_monitor_snapshot(
        started_at_monotonic=time_start,
        progress_log_path=progress_log,
    )

    snapshot_text = json.dumps(snapshot, sort_keys=True)
    assert snapshot["latest_progress_event_redacted"] is True
    assert "latest_progress_event_sha256" in snapshot
    assert snapshot["latest_progress_range_count"] == 12
    assert "progress_log_path" not in snapshot
    for forbidden in v25.RECURSIVE_FORBIDDEN_FINAL_DETAIL_KEY_TERMS:
        assert forbidden not in snapshot_text


def test_v25_long_run_monitor_writes_start_and_stop_events(tmp_path: Path) -> None:
    monitor_log = tmp_path / v25.LONG_RUN_MONITOR_LOG_FILENAME
    progress_log = tmp_path / v25.SOURCE_POOL_PROGRESS_LOG_FILENAME

    monitor = v25.LongRunMonitor(
        monitor_log_path=monitor_log,
        progress_log_path=progress_log,
        interval_seconds=3600.0,
    )
    monitor.start()
    monitor.stop()

    events = [json.loads(line)["event"] for line in monitor_log.read_text().splitlines()]
    assert events == ["monitor_start", "monitor_stop"]


def test_v25_stdout_summary_uses_location_hashes_for_paths(tmp_path: Path) -> None:
    result = {
        "long_run_monitor_log_path": str(tmp_path / "weights_monitor.jsonl"),
        "long_run_monitor_log_sha256": "monitor",
        "passed": True,
        "pool_dir": str(tmp_path / "signature_pool"),
        "pool_summaries": {
            "train": {
                "accepted_counts_by_behavior": {"sorted_ascending": 64},
                "by_behavior": {
                    "sorted_ascending": {
                        "accepted_subject_ids": ["subject_id_with_seed_detail"],
                    },
                },
                "pool_file_sha256": "train",
                "record_count": 64,
            },
        },
        "seed_preflight": {"failures": [], "passed": True},
        "source_pool_progress_log_path": str(tmp_path / "seed_progress.jsonl"),
        "source_pool_progress_log_sha256": "progress",
    }

    summary = v25.stdout_summary(result)
    summary_text = json.dumps(summary, sort_keys=True)

    assert "long_run_monitor_log_path" not in summary
    assert "pool_dir" not in summary
    assert "source_pool_progress_log_path" not in summary
    assert summary["long_run_monitor_log_location_sha256"]
    assert summary["pool_dir_location_sha256"]
    assert summary["preflight_failure_count"] == 0
    assert summary["preflight_passed"] is True
    assert summary["source_pool_progress_log_location_sha256"]
    assert '"by_behavior"' not in summary_text
    assert "accepted_subject_ids" not in summary_text
    for forbidden in v25.RECURSIVE_FORBIDDEN_FINAL_DETAIL_KEY_TERMS:
        assert forbidden not in summary_text


def test_v25_stdout_summary_preserves_development_setup_fields() -> None:
    result = {
        "development_pool": {"record_count": 98},
        "development_jobs": {"job_count": 294, "job_plan_hash": "a" * 64},
        "development_progress_log_sha256": "b" * 64,
        "descriptor_norm_hash": "c" * 64,
        "long_run_monitor_log_sha256": "d" * 64,
        "passed": False,
        "probe_examples_hash": "e" * 64,
        "stage": "development_setup_completed",
        "train_pool": {"record_count": 264},
        "train_statistics_hash": "f" * 64,
    }

    summary = v25.stdout_summary(result)

    assert summary["stage"] == "development_setup_completed"
    assert summary["train_statistics_hash"] == "f" * 64
    assert summary["descriptor_norm_hash"] == "c" * 64
    assert summary["development_progress_log_sha256"] == "b" * 64
    assert summary["development_jobs"] == {"job_count": 294, "job_plan_hash": "a" * 64}
    assert summary["train_pool"] == {"record_count": 264}
    assert summary["development_pool"] == {"record_count": 98}
