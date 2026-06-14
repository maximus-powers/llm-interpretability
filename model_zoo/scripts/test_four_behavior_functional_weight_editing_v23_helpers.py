from pathlib import Path

import pytest
import torch

import train_four_behavior_functional_weight_editing_v22_component_activation_rank1_editor as v22
import train_four_behavior_functional_weight_editing_v23_probe_routed_sparse_subspace_editor as v23


def test_v23_fresh_scope_paths_pool_seeds_and_plan_binding() -> None:
    assert v23.EDITOR_METHOD == "probe_routed_sparse_subspace_editor_v23"
    assert v23.PLAN_SHA256 == "50a26e376b39b3f3d93d8bdf894534f1571ab444491c044de324e16ae6c671e8"
    assert "v23" in str(v23.DEFAULT_POOL_DIR)
    assert "v23" in str(v23.DEFAULT_OUTPUT_DIR)
    assert v23.POOL_CONFIGS["train"]["base_seed"] == 120400000
    assert v23.POOL_CONFIGS["development"]["base_seed"] == 121400000
    assert v23.POOL_CONFIGS["final"]["base_seed"] == 122400000
    assert v23.POOL_CONFIGS["train"]["base_seed"] != v22.POOL_CONFIGS["train"]["base_seed"]
    assert v23.EXPECTED_CONTROLS_PER_RECORD == 32
    assert v23.RANDOM_CONTROLS_PER_RECORD == 20
    assert v23.PASSING_DEVELOPMENT_NEXT_ACTION == "run_hash_bound_final_after_reviewer_authorization"
    assert v23.FAILING_DEVELOPMENT_NEXT_ACTION == "log_negative_development_result_do_not_open_final_raw"


def test_v23_final_raw_guard_rejects_prior_v22_and_any_runs_final_subjects_path() -> None:
    with pytest.raises(ValueError):
        v23.assert_no_forbidden_final_raw_paths([v23.V23_FINAL_RAW])
    with pytest.raises(ValueError):
        v23.assert_no_forbidden_final_raw_paths([v22.V22_FINAL_RAW])
    with pytest.raises(ValueError):
        v23.assert_no_forbidden_final_raw_paths([
            Path("/Users/max/Desktop/muat/runs/other_experiment/final_subjects.json")
        ])
    v23.assert_no_forbidden_final_raw_paths([Path("runs/not_final/train_subjects.json")])


def test_v23_final_redaction_allowlists_fail_closed() -> None:
    payload = {
        "behavior_suite_hashes": {},
        "candidate_pool_summary_hash": "a",
        "claim_scope": v23.FINAL_REDACTED_SCOPE,
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
    assert v23.forbidden_final_redacted_keys(payload) == []
    leaked = {**payload, "subject_ids": ["leak"]}
    assert "top_level.subject_ids" in v23.forbidden_final_redacted_keys(leaked)
    leaked_summary = {
        **payload,
        "summary": {**payload["summary"], "weights_hashes": ["leak"]},
    }
    assert "summary.weights_hashes" in v23.forbidden_final_redacted_keys(leaked_summary)


def test_v23_constants_payload_binds_sparse_subspace_grid() -> None:
    payload = v23.constants_payload()
    assert payload["editor_method"] == v23.EDITOR_METHOD
    assert payload["plan_sha256"] == v23.PLAN_SHA256
    assert payload["sparse_k_values"] == [1, 2, 3]
    assert payload["lambda_solve_grid"] == [0.01, 0.1, 1.0, 10.0]
    assert payload["compatible_weight_grid"] == [0.5, 1.0, 2.0, 4.0]
    assert payload["probe_centroid_weight_grid"] == [0.0, 0.25, 0.5, 1.0]
    assert payload["control_penalty_weight_grid"] == [0.0, 0.5, 1.0, 2.0]
    assert payload["inner_validation_rung_subjects_per_behavior"] == [1, 4, 13]
    assert payload["inner_validation_evaluated_config_count"] == 128


def test_v23_inner_validation_progress_checkpoint_round_trips_completed_candidates(
    tmp_path: Path,
) -> None:
    configs = v23.iter_sparse_subspace_configs()[:2]
    checkpoint_path = tmp_path / v23.INNER_VALIDATION_CHECKPOINT_FILENAME
    progress_log_path = tmp_path / v23.INNER_VALIDATION_PROGRESS_LOG_FILENAME
    checkpoint = v23.new_inner_validation_progress_checkpoint(configs)
    rung = v23.get_or_create_inner_validation_rung_progress(
        checkpoint=checkpoint,
        rung_index=0,
        record_budget=12,
        survivor_count=1,
        active_configs=configs,
    )
    candidate = {
        **configs[0],
        "invalid": False,
        "inner_validation_record_count": 12,
        "target_prediction_rate": 1.0,
    }

    v23.record_inner_validation_candidate_progress(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        progress_log_path=progress_log_path,
        rung=rung,
        candidate=candidate,
        event="candidate_completed",
        elapsed_seconds=0.25,
    )

    loaded = v23.load_inner_validation_progress_checkpoint(checkpoint_path)
    completed = v23.inner_validation_completed_candidates_by_hash(
        checkpoint=loaded,
        rung_index=0,
        record_budget=12,
        active_configs=configs,
    )
    assert list(completed) == [configs[0]["config_hash"]]
    assert completed[configs[0]["config_hash"]]["target_prediction_rate"] == 1.0
    assert progress_log_path.exists()
    assert '"event": "candidate_completed"' in progress_log_path.read_text()


def test_v23_inner_validation_completed_candidates_accept_progress_only_hash() -> None:
    configs = v23.iter_sparse_subspace_configs()[:2]
    checkpoint = v23.new_inner_validation_progress_checkpoint(configs)
    checkpoint["implementation_sha256"] = next(iter(
        v23.INNER_VALIDATION_PROGRESS_ONLY_COMPATIBLE_IMPLEMENTATION_SHA256
    ))
    rung = v23.get_or_create_inner_validation_rung_progress(
        checkpoint=checkpoint,
        rung_index=0,
        record_budget=12,
        survivor_count=1,
        active_configs=configs,
    )
    candidate = {
        **configs[0],
        "invalid": False,
        "inner_validation_record_count": 12,
        "target_prediction_rate": 1.0,
    }
    rung["candidates"].append(candidate)
    rung["completed_count"] = 1

    completed = v23.inner_validation_completed_candidates_by_hash(
        checkpoint=checkpoint,
        rung_index=0,
        record_budget=12,
        active_configs=configs,
    )

    assert list(completed) == [configs[0]["config_hash"]]
    assert completed[configs[0]["config_hash"]]["target_prediction_rate"] == 1.0


def test_v23_development_progress_event_appends_jsonl(tmp_path: Path) -> None:
    progress_log_path = tmp_path / v23.DEVELOPMENT_PROGRESS_LOG_FILENAME

    v23.record_development_progress_event(
        progress_log_path,
        event="development_start",
        started_at_monotonic=10.0,
        extra={"max_workers": 8},
        now_monotonic=lambda: 12.5,
    )

    text = progress_log_path.read_text()
    assert '"event": "development_start"' in text
    assert '"elapsed_seconds": 2.5' in text
    assert '"max_workers": 8' in text


def test_v23_evaluate_subjects_logs_record_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_log_path = tmp_path / v23.DEVELOPMENT_PROGRESS_LOG_FILENAME
    subject = {
        "behavior": v23.PATTERNS[0],
        "subject_id": "subject-0",
        "signature": [0.0],
        "weights": [0.0],
    }

    def fake_evaluator(job, *, train_stats, random_controls):
        del train_stats, random_controls
        return {
            "source_behavior": job["source"],
            "subject_id": job["subject"]["subject_id"],
            "target_behavior": job["target"],
        }

    monkeypatch.setattr(v23, "assign_shuffled_signatures", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(v23, "summarize_records", lambda records: {"count": len(records)})
    monkeypatch.setattr(v23, "gate_failures", lambda **_kwargs: [])

    result = v23.evaluate_subjects(
        subjects=[subject],
        train_stats={},
        parallel=False,
        record_evaluator=fake_evaluator,
        progress_log_path=progress_log_path,
        progress_started_at_monotonic=10.0,
        progress_event_prefix="development_evaluation",
    )

    text = progress_log_path.read_text()
    assert result["record_count"] == len(v23.PATTERNS) - 1
    assert '"event": "development_evaluation_jobs_queued"' in text
    assert text.count('"event": "development_evaluation_record_completed"') == (
        len(v23.PATTERNS) - 1
    )
    assert f'"record_count": {len(v23.PATTERNS) - 1}' in text


def test_v23_ordered_inner_validation_subjects_for_budget_is_balanced_by_behavior() -> None:
    subjects = [
        {"behavior": behavior, "subject_id": f"{behavior}-{index:02d}"}
        for behavior in v23.PATTERNS
        for index in range(3)
    ]

    selected = v23.ordered_inner_validation_subjects_for_budget(subjects, record_budget=12)

    assert len(selected) == len(v23.PATTERNS)
    assert [item["behavior"] for item in selected] == list(v23.PATTERNS)
    with pytest.raises(ValueError, match="divisible"):
        v23.ordered_inner_validation_subjects_for_budget(subjects, record_budget=10)


def test_v23_inner_validation_candidate_for_config_fails_closed_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = v23.iter_sparse_subspace_configs()[0]

    def boom(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(v23, "train_stats_with_selected_sparse_config", boom)

    candidate = v23.inner_validation_candidate_for_config(
        config=config,
        base_train_stats={},
        rung_subjects=[],
        record_budget=12,
    )

    assert candidate["invalid"] is True
    assert candidate["config_hash"] == config["config_hash"]
    assert candidate["invalid_reasons"] == ["exception:RuntimeError:synthetic failure"]


def test_v23_inner_validation_progress_resume_rejects_stale_scope_or_active_set(
    tmp_path: Path,
) -> None:
    configs = v23.iter_sparse_subspace_configs()[:2]
    stale_configs = list(reversed(configs))
    checkpoint = v23.new_inner_validation_progress_checkpoint(configs)
    rung = v23.get_or_create_inner_validation_rung_progress(
        checkpoint=checkpoint,
        rung_index=0,
        record_budget=12,
        survivor_count=1,
        active_configs=configs,
    )
    rung["candidates"].append({**configs[0], "inner_validation_record_count": 12})

    assert not v23.inner_validation_completed_candidates_by_hash(
        checkpoint={**checkpoint, "scope": "stale"},
        rung_index=0,
        record_budget=12,
        active_configs=configs,
    )
    assert not v23.inner_validation_completed_candidates_by_hash(
        checkpoint=checkpoint,
        rung_index=0,
        record_budget=12,
        active_configs=stale_configs,
    )
    assert not v23.load_inner_validation_progress_checkpoint(
        tmp_path / "missing.json"
    )


def test_v23_inner_validation_progress_resume_rejects_stale_implementation_or_constants() -> None:
    configs = v23.iter_sparse_subspace_configs()[:2]
    checkpoint = v23.new_inner_validation_progress_checkpoint(configs)
    rung = v23.get_or_create_inner_validation_rung_progress(
        checkpoint=checkpoint,
        rung_index=0,
        record_budget=12,
        survivor_count=1,
        active_configs=configs,
    )
    rung["candidates"].append({**configs[0], "inner_validation_record_count": 12})

    assert not v23.inner_validation_completed_candidates_by_hash(
        checkpoint={**checkpoint, "implementation_sha256": "stale"},
        rung_index=0,
        record_budget=12,
        active_configs=configs,
    )
    assert not v23.inner_validation_completed_candidates_by_hash(
        checkpoint={**checkpoint, "constants_sha256": "stale"},
        rung_index=0,
        record_budget=12,
        active_configs=configs,
    )


def test_v23_rank1_component_basis_vector_matches_formula_and_scope() -> None:
    direction = torch.arange(1, 9, dtype=torch.float32)
    xbar = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
    lambda_rank1 = 0.5

    basis = v23.rank1_component_basis_vector(
        layer_index=0,
        direction=direction,
        xbar=xbar,
        lambda_rank1=lambda_rank1,
    )

    weight_spec, bias_spec = v23.hidden_layer_specs(0)
    expected_weight = torch.outer(direction, xbar) / (float(torch.dot(xbar, xbar)) + lambda_rank1)
    expected_bias = direction
    assert basis.shape == (v23.SOURCE_WEIGHT_DIM,)
    assert torch.allclose(v23.v17.component_from_flat(basis, weight_spec), expected_weight)
    assert torch.allclose(v23.v17.component_from_flat(basis, bias_spec), expected_bias)

    outside_components = basis.clone()
    outside_components[weight_spec["start"] : weight_spec["end"]] = 0.0
    outside_components[bias_spec["start"] : bias_spec["end"]] = 0.0
    assert torch.count_nonzero(outside_components).item() == 0


def test_v23_layer_relevance_formula_penalizes_preserve_cost_and_controls() -> None:
    score = v23.layer_relevance_score(
        edit_projection=torch.tensor([1.0, 2.0]),
        edit_target=torch.tensor([1.0, 1.0]),
        preserve_projection=torch.tensor([2.0, 0.0]),
        matched_direction=torch.tensor([1.0, 0.0]),
        control_directions=[torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
        compatible_weight=0.5,
        control_penalty_weight=2.0,
    )

    expected_target_gain = 9.0 / 10.0
    expected_preserve_cost = 2.0
    expected_control_penalty = 1.0
    assert score["target_gain"] == pytest.approx(expected_target_gain)
    assert score["preserve_cost"] == pytest.approx(expected_preserve_cost)
    assert score["control_similarity_penalty"] == pytest.approx(expected_control_penalty)
    assert score["relevance"] == pytest.approx(
        expected_target_gain - 0.5 * expected_preserve_cost - 2.0 * expected_control_penalty
    )


def test_v23_layer_relevance_uses_infinite_preserve_cost_when_preserve_rows_empty() -> None:
    score = v23.layer_relevance_score(
        edit_projection=torch.tensor([1.0]),
        edit_target=torch.tensor([1.0]),
        preserve_projection=torch.tensor([], dtype=torch.float32),
        matched_direction=torch.tensor([1.0]),
        control_directions=[],
        compatible_weight=1.0,
        control_penalty_weight=0.0,
    )

    assert score["preserve_cost"] == float("inf")
    assert score["relevance"] == float("-inf")


def test_v23_select_sparse_layers_uses_preregistered_tie_breaks() -> None:
    scores = [
        {
            "layer_index": 4,
            "relevance": 1.0,
            "target_gain": 0.9,
            "preserve_cost": 0.2,
            "control_similarity_penalty": 0.1,
        },
        {
            "layer_index": 1,
            "relevance": 1.0,
            "target_gain": 0.9,
            "preserve_cost": 0.1,
            "control_similarity_penalty": 0.2,
        },
        {
            "layer_index": 2,
            "relevance": 1.0,
            "target_gain": 0.8,
            "preserve_cost": 0.0,
            "control_similarity_penalty": 0.0,
        },
        {
            "layer_index": 0,
            "relevance": 0.5,
            "target_gain": 1.0,
            "preserve_cost": 0.0,
            "control_similarity_penalty": 0.0,
        },
    ]

    selected = v23.select_sparse_layers_by_relevance(scores, k=3)

    assert [item["layer_index"] for item in selected] == [1, 4, 2]


def test_v23_sparse_ridge_solve_matches_closed_form() -> None:
    x = torch.tensor([[1.0, 2.0], [0.0, 1.0], [3.0, -1.0]], dtype=torch.float64)
    y = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    lambda_solve = 0.1

    result = v23.solve_sparse_ridge_coefficients(x, y, lambda_solve=lambda_solve)

    x32 = x.to(dtype=torch.float32)
    y32 = y.to(dtype=torch.float32)
    expected = torch.linalg.solve(
        x32.T @ x32 + lambda_solve * torch.eye(2, dtype=torch.float32),
        x32.T @ y32,
    )
    assert result["jitter_retry"] is False
    assert result["invalid"] is False
    assert result["alpha"].dtype == torch.float32
    assert torch.allclose(result["alpha"], expected)


def test_v23_sparse_ridge_solve_retries_once_with_jitter() -> None:
    x = torch.zeros((2, 2), dtype=torch.float32)
    y = torch.tensor([1.0, 2.0], dtype=torch.float32)

    result = v23.solve_sparse_ridge_coefficients(x, y, lambda_solve=0.0)

    assert result["jitter_retry"] is True
    assert result["invalid"] is False
    assert torch.allclose(result["alpha"], torch.zeros(2, dtype=torch.float32))


def test_v23_sparse_hidden_delta_scope_rejects_output_or_unselected_edits() -> None:
    delta = torch.zeros(v23.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    delta[344] = 1.0
    with pytest.raises(ValueError, match="outside selected hidden components"):
        v23.assert_sparse_hidden_delta_scope(delta, selected_layers=[0])

    unselected_weight_spec, _bias_spec = v23.hidden_layer_specs(1)
    delta = torch.zeros(v23.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    delta[unselected_weight_spec["start"]] = 1.0
    with pytest.raises(ValueError, match="outside selected hidden components"):
        v23.assert_sparse_hidden_delta_scope(delta, selected_layers=[0])


def test_v23_clip_sparse_hidden_delta_uses_selected_hidden_norm_only() -> None:
    direction = torch.ones(8, dtype=torch.float32)
    xbar = torch.ones(5, dtype=torch.float32)
    delta = 3.0 * v23.rank1_component_basis_vector(
        layer_index=0,
        direction=direction,
        xbar=xbar,
        lambda_rank1=1.0,
    )

    clipped, metadata = v23.clip_sparse_hidden_delta(
        delta,
        selected_layers=[0],
        norm_cap=1.0,
    )

    assert metadata["hidden_delta_clipped"] is True
    assert metadata["raw_hidden_delta_norm"] > 1.0
    assert metadata["hidden_delta_norm"] == pytest.approx(1.0)
    assert v23.sparse_hidden_delta_norm(clipped, selected_layers=[0]) == pytest.approx(1.0)


def test_v23_weighted_design_rows_use_preregistered_group_order_and_weights() -> None:
    basis = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    row_groups = {
        "probe_compatible": {
            "jacobian": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
            "target": torch.tensor([4.0], dtype=torch.float32),
        },
        "target_support": {
            "jacobian": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
            "target": torch.tensor([1.0, 2.0], dtype=torch.float32),
        },
        "conflict_support": {
            "jacobian": torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
            "target": torch.tensor([3.0], dtype=torch.float32),
        },
        "probe_target": {
            "jacobian": torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float32),
            "target": torch.tensor([99.0], dtype=torch.float32),
        },
    }

    design = v23.weighted_design_from_row_groups(
        row_groups,
        basis=basis,
        block_weights={
            "target_support": 1.0,
            "conflict_support": 1.0,
            "compatible_support": 4.0,
            "probe_target": 0.0,
            "probe_compatible": 4.0,
        },
    )

    target_scale = 2.0 ** -0.5
    expected_x = torch.tensor(
        [
            [target_scale, 0.0],
            [0.0, target_scale],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    expected_y = torch.tensor(
        [target_scale, 2.0 * target_scale, 3.0, 8.0],
        dtype=torch.float32,
    )
    assert torch.allclose(design["x"], expected_x)
    assert torch.allclose(design["y"], expected_y)
    assert design["row_group_names"] == [
        "target_support",
        "target_support",
        "conflict_support",
        "probe_compatible",
    ]


def test_v23_inner_split_is_train_only_hash_sorted_and_disjoint() -> None:
    subjects = [
        {"subject_id": f"{behavior}_{index:02d}", "behavior": behavior}
        for behavior in v23.PATTERNS
        for index in range(64)
    ]

    split = v23.inner_train_validation_split(subjects)

    assert {
        behavior: len(split["inner_train_by_behavior"][behavior])
        for behavior in v23.PATTERNS
    } == {behavior: 51 for behavior in v23.PATTERNS}
    assert {
        behavior: len(split["inner_validation_by_behavior"][behavior])
        for behavior in v23.PATTERNS
    } == {behavior: 13 for behavior in v23.PATTERNS}
    train_ids = {item["subject_id"] for item in split["inner_train_subjects"]}
    validation_ids = {item["subject_id"] for item in split["inner_validation_subjects"]}
    assert train_ids.isdisjoint(validation_ids)

    behavior = v23.PATTERNS[0]
    expected_sorted = sorted(
        [item for item in subjects if item["behavior"] == behavior],
        key=lambda item: (
            v23.stable_hash_json({
                "scope": "four_behavior_functional_weight_editing_v23_inner_split",
                "behavior": behavior,
                "subject_id": item["subject_id"],
            }),
            item["subject_id"],
        ),
    )
    assert split["inner_train_by_behavior"][behavior] == expected_sorted[:51]
    assert split["inner_validation_by_behavior"][behavior] == expected_sorted[51:]


def test_v23_select_inner_validation_config_uses_preregistered_lexicographic_order() -> None:
    candidates = [
        {
            "config_index": 0,
            "invalid": True,
            "target_prediction_rate": 1.0,
            "pareto_undominated_rate": 1.0,
        },
        {
            "config_index": 1,
            "invalid": False,
            "target_prediction_rate": 0.8,
            "pareto_undominated_rate": 0.9,
            "mean_matched_minus_best_control_target_margin": 0.2,
            "mean_matched_minus_shuffled_signature_target_margin": 0.2,
            "mean_target_margin": 0.5,
            "mean_compatible_source_mse": 0.1,
            "effective_zero_coefficient_rate": 0.1,
            "mean_hidden_delta_norm": 0.2,
        },
        {
            "config_index": 2,
            "invalid": False,
            "target_prediction_rate": 0.8,
            "pareto_undominated_rate": 0.9,
            "mean_matched_minus_best_control_target_margin": 0.2,
            "mean_matched_minus_shuffled_signature_target_margin": 0.2,
            "mean_target_margin": 0.5,
            "mean_compatible_source_mse": 0.1,
            "effective_zero_coefficient_rate": 0.05,
            "mean_hidden_delta_norm": 0.4,
        },
    ]

    selected = v23.select_inner_validation_config(candidates)

    assert selected["config_index"] == 2


def test_v23_inner_validation_selector_uses_inner_split_and_valid_config_metrics(monkeypatch) -> None:
    configs = [
        {"config_index": 0, "config_hash": "config0"},
        {"config_index": 1, "config_hash": "config1"},
    ]
    subjects = [{"subject_id": f"s{index}", "behavior": v23.PATTERNS[index % 4]} for index in range(256)]
    validation_subjects = [
        {"subject_id": f"v{index}", "behavior": v23.PATTERNS[index % 4]}
        for index in range(52)
    ]
    monkeypatch.setattr(v23, "iter_sparse_subspace_configs", lambda: configs)
    monkeypatch.setattr(
        v23,
        "inner_train_validation_split",
        lambda _subjects: {
            "inner_train_subjects": subjects[:204],
            "inner_validation_subjects": validation_subjects,
            "inner_train_by_behavior": {},
            "inner_validation_by_behavior": {},
        },
    )

    def fake_fit(inner_subjects, **kwargs):
        config = kwargs["selected_sparse_config"]
        return {
            "selected_sparse_config": config,
            "selected_sparse_config_hash": config["config_hash"],
            "train_statistics_hash": f"stats-{config['config_index']}",
        }

    def fake_evaluate(*, train_stats, subjects, **_kwargs):
        config_index = train_stats["selected_sparse_config"]["config_index"]
        controls = [{"control_type": control_type} for control_type in v23.PROOF_CRITICAL_CONTROL_TYPES]
        record_count = len(subjects) * 3
        records = [
            {
                "controls": controls,
                "matched": {
                    "compatible_source_output_mse": 0.2,
                    "editor": {"scale_0_selected": False, "hidden_delta_norm": 0.1 + config_index},
                    "delta_norm": 0.1 + config_index,
                },
                "summary": {},
            }
            for _index in range(record_count)
        ]
        aggregate = {
            "mean_matched_minus_best_control_target_margin": 0.1 + config_index,
            "mean_matched_minus_shuffled_signature_target_margin": 0.2 + config_index,
            "mean_target_margin": 0.3 + config_index,
            "pareto_undominated_rate": 0.4 + config_index,
            "target_prediction_rate": 0.5 + config_index,
        }
        return {"aggregate": aggregate, "record_count": record_count, "records": records}

    monkeypatch.setattr(v23, "fit_v23_train_statistics", fake_fit)
    monkeypatch.setattr(v23, "evaluate_subjects", fake_evaluate)

    selected = v23.select_sparse_config_with_inner_validation(subjects, max_workers=1)

    assert selected["config_index"] == 1
    assert selected["config_hash"] == "config1"
    assert selected["inner_validation_record_count"] == 156
    assert selected["inner_validation_total_config_count"] == 2
    assert selected["inner_validation_evaluated_config_count"] == 2
    assert selected["inner_validation_rung_record_budgets"] == [12, 48, 156]
    assert selected["inner_validation_rung_subjects_per_behavior"] == [1, 4, 13]
    assert selected["inner_validation_rung_survivors"] == [32, 8, 1]
    assert [item["record_budget"] for item in selected["inner_validation_rung_summaries"]] == [12, 48, 156]


def test_v23_sparse_config_grid_has_deterministic_indices_and_hashes() -> None:
    configs = v23.iter_sparse_subspace_configs()

    assert len(configs) == (
        len(v23.SPARSE_K_VALUES)
        * len(v23.LAMBDA_RANK1_GRID)
        * len(v23.LAMBDA_SOLVE_GRID)
        * len(v23.COMPATIBLE_WEIGHT_GRID)
        * len(v23.PROBE_CENTROID_WEIGHT_GRID)
        * len(v23.CONTROL_PENALTY_WEIGHT_GRID)
        * len(v23.CAP_MULTIPLIER_GRID)
    )
    assert v23.INNER_VALIDATION_TOTAL_CONFIG_COUNT == len(configs)
    assert [config["config_index"] for config in configs[:3]] == [0, 1, 2]
    assert configs[0] == {
        "cap_multiplier": 0.25,
        "compatible_weight": 0.5,
        "config_hash": v23.sparse_config_hash({**configs[0], "config_hash": None}),
        "config_index": 0,
        "control_penalty_weight": 0.0,
        "k": 1,
        "lambda_rank1": 0.01,
        "lambda_solve": 0.01,
        "probe_centroid_weight": 0.0,
    }
    assert configs[-1]["config_index"] == len(configs) - 1
    assert configs[-1]["k"] == 3
    assert configs[-1]["lambda_rank1"] == 10.0
    assert configs[-1]["lambda_solve"] == 10.0
    assert configs[-1]["compatible_weight"] == 4.0
    assert configs[-1]["probe_centroid_weight"] == 1.0
    assert configs[-1]["control_penalty_weight"] == 2.0
    assert configs[-1]["cap_multiplier"] == 1.0


def test_v23_inner_validation_config_subset_is_deterministic_stratified_sample() -> None:
    configs = v23.iter_sparse_subspace_configs()

    subset = v23.inner_validation_evaluated_config_subset(configs)

    assert len(subset) == v23.INNER_VALIDATION_EVALUATED_CONFIG_COUNT
    assert len({config["config_hash"] for config in subset}) == len(subset)
    assert all(config in configs for config in subset)
    strata = {(config["k"], config["cap_multiplier"]) for config in subset}
    assert strata == {
        (k, cap_multiplier)
        for k in v23.SPARSE_K_VALUES
        for cap_multiplier in v23.CAP_MULTIPLIER_GRID
    }


def test_v23_select_sparse_post_scale_candidate_uses_record_level_tie_breaks() -> None:
    candidates = [
        {
            "candidate_index": 0,
            "compatible_probe_utility_loss": 0.1,
            "hidden_delta_norm": 0.1,
            "post_scale": 0.25,
            "support_objective": 0.5,
            "target_probe_centroid_loss": 0.1,
        },
        {
            "candidate_index": 1,
            "compatible_probe_utility_loss": 0.1,
            "hidden_delta_norm": 0.2,
            "post_scale": 0.5,
            "support_objective": 0.4,
            "target_probe_centroid_loss": 0.1,
        },
        {
            "candidate_index": 2,
            "compatible_probe_utility_loss": 0.1,
            "hidden_delta_norm": 0.05,
            "post_scale": 0.75,
            "support_objective": 0.4,
            "target_probe_centroid_loss": 0.2,
        },
    ]

    selected = v23.select_sparse_post_scale_candidate(candidates)

    assert selected["candidate_index"] == 1


def test_v23_random_sparse_control_is_deterministic_and_norm_matched() -> None:
    basis_0 = v23.rank1_component_basis_vector(
        layer_index=0,
        direction=torch.ones(8),
        xbar=torch.ones(5),
        lambda_rank1=1.0,
    )
    basis_1 = v23.rank1_component_basis_vector(
        layer_index=1,
        direction=torch.arange(1, 9, dtype=torch.float32),
        xbar=torch.ones(8),
        lambda_rank1=1.0,
    )
    basis = torch.stack([basis_0, basis_1], dim=1)
    matched_delta = 0.25 * basis_0 - 0.5 * basis_1
    seed_payload = {
        "subject_hash": "subject",
        "source": "a",
        "target": "b",
        "selected_config_hash": "config",
        "train_statistics_hash": "stats",
        "index": 3,
        "selected_layers": [0, 1],
    }

    first_delta, first_meta = v23.random_sparse_subspace_delta(
        basis=basis,
        matched_hidden_delta=matched_delta,
        selected_layers=[0, 1],
        seed_payload=seed_payload,
    )
    second_delta, second_meta = v23.random_sparse_subspace_delta(
        basis=basis,
        matched_hidden_delta=matched_delta,
        selected_layers=[0, 1],
        seed_payload=seed_payload,
    )

    matched_norm = v23.sparse_hidden_delta_norm(matched_delta, selected_layers=[0, 1])
    assert torch.allclose(first_delta, second_delta)
    assert first_meta["seed_hash"] == second_meta["seed_hash"]
    assert first_meta["zero_norm_fallback"] is False
    assert first_meta["matched_hidden_delta_norm"] == pytest.approx(matched_norm)
    assert first_meta["random_hidden_delta_norm"] == pytest.approx(matched_norm)


def test_v23_random_sparse_control_zero_fallback_when_matched_delta_is_zero() -> None:
    basis = torch.stack([
        v23.rank1_component_basis_vector(
            layer_index=0,
            direction=torch.ones(8),
            xbar=torch.ones(5),
            lambda_rank1=1.0,
        )
    ], dim=1)

    random_delta, metadata = v23.random_sparse_subspace_delta(
        basis=basis,
        matched_hidden_delta=torch.zeros(v23.SOURCE_WEIGHT_DIM),
        selected_layers=[0],
        seed_payload={
            "subject_hash": "subject",
            "source": "a",
            "target": "b",
            "selected_config_hash": "config",
            "train_statistics_hash": "stats",
            "index": 0,
            "selected_layers": [0],
        },
    )

    assert torch.count_nonzero(random_delta).item() == 0
    assert metadata["zero_norm_fallback"] is True
    assert metadata["random_hidden_delta_norm"] == 0.0


def test_v23_pareto_controls_include_sparse_random_controls_not_v22_component_prefix() -> None:
    controls = [
        {"control_type": "no_edit"},
        {"control_type": "random_norm_matched_sparse_subspace_00"},
        {"control_type": "random_norm_matched_component_rank1_00"},
    ]

    pareto_types = [item["control_type"] for item in v23.pareto_controls_for_record(controls)]

    assert "no_edit" in pareto_types
    assert "random_norm_matched_sparse_subspace_00" in pareto_types
    assert "random_norm_matched_component_rank1_00" not in pareto_types


def test_v23_proof_critical_controls_include_v22_and_sparse_control_names() -> None:
    assert "v22_component_activation_rank1_editor_recomputed" in v23.PROOF_CRITICAL_CONTROL_TYPES
    for control_type in [
        "no_probe_sparse_subspace_editor",
        "source_probe_sparse_subspace_editor",
        "shuffled_probe_sparse_subspace_editor",
        "target_label_only_sparse_subspace_editor",
        "nearest_target_sparse_subspace_editor",
    ]:
        assert control_type in v23.PROOF_CRITICAL_CONTROL_TYPES
    for stale_type in [
        "no_probe_component_rank1_editor",
        "source_probe_component_rank1_editor",
        "random_norm_matched_component_rank1_00",
    ]:
        assert stale_type not in v23.PROOF_CRITICAL_CONTROL_TYPES


def test_v23_sparse_selector_returns_scoped_metadata_without_expensive_controls(monkeypatch) -> None:
    source_weights = torch.zeros(v23.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    base_weights = torch.zeros(v23.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    base_weights[336] = 2.0

    monkeypatch.setattr(
        v23.v16,
        "output_layer_no_signature_support_optimizer",
        lambda **_kwargs: (base_weights.clone(), {"stub": True}),
    )
    monkeypatch.setattr(
        v23,
        "hidden_rank1_descriptor_for_weights",
        lambda **_kwargs: {
            "hbar": [torch.zeros(8, dtype=torch.float32) for _ in v23.HIDDEN_LAYERS],
            "xbar": [
                torch.ones(5, dtype=torch.float32),
                *[torch.ones(8, dtype=torch.float32) for _ in v23.HIDDEN_LAYERS[1:]],
            ],
        },
    )
    monkeypatch.setattr(
        v23,
        "target_direction_for_mode",
        lambda **_kwargs: [torch.ones(8, dtype=torch.float32) for _ in v23.HIDDEN_LAYERS],
    )

    row = torch.ones((1, v23.SOURCE_WEIGHT_DIM), dtype=torch.float32)
    monkeypatch.setattr(
        v23,
        "sparse_subspace_row_groups",
        lambda **_kwargs: {
            "target_support": {"jacobian": row, "target": torch.tensor([1.0])},
            "conflict_support": {"jacobian": row, "target": torch.tensor([1.0])},
            "compatible_support": {"jacobian": row, "target": torch.tensor([0.0])},
            "probe_target": {"jacobian": row, "target": torch.tensor([1.0])},
            "probe_compatible": {"jacobian": row, "target": torch.tensor([0.0])},
        },
    )
    monkeypatch.setattr(
        v23,
        "sparse_post_scale_candidate_losses",
        lambda **_kwargs: {
            "compatible_probe_utility_loss": 0.0,
            "support_objective": 0.0,
            "target_probe_centroid_loss": 0.0,
        },
    )

    weights, metadata = v23.select_probe_routed_sparse_subspace_edit(
        source_weights=source_weights,
        source_signature_norm=torch.zeros(v23.SIGNATURE_DIM),
        source=v23.PATTERNS[0],
        target=v23.PATTERNS[1],
        subject={"subject_id": "stub"},
        train_stats={
            "probe_examples": [],
            "selected_sparse_config": v23.iter_sparse_subspace_configs()[0],
            "train_statistics_hash": "stats",
        },
    )

    assert metadata["control_type"] == v23.EDITOR_METHOD
    assert metadata["selected_layers"] == [0]
    assert metadata["selected_config_hash"] == v23.iter_sparse_subspace_configs()[0]["config_hash"]
    assert metadata["post_scale"] == v23.SPARSE_POST_SCALE_GRID[0]
    assert metadata["hidden_delta_norm"] <= metadata["hidden_norm_cap"] + 1e-6
    assert torch.allclose(v23.output_layer_theta(weights), v23.output_layer_theta(base_weights))
    v23.assert_sparse_hidden_delta_scope(
        metadata["_selected_hidden_delta"],
        selected_layers=metadata["selected_layers"],
    )


def _passing_gate_summary() -> dict[str, float]:
    summary = {
        "individual_all_gate_pass_rate": 1.0,
        "mean_conflict_target_accuracy": 1.0,
        "mean_conflict_target_accuracy_improvement": 1.0,
        "mean_matched_minus_best_control_target_margin": 1.0,
        "mean_target_margin": 1.0,
        "n": v23.THRESHOLDS["expected_record_count"],
        "pareto_undominated_rate": 1.0,
        "target_prediction_rate": 1.0,
    }
    for metric_name in v23.ADVANTAGE_CONTROL_TYPES:
        summary[f"mean_matched_minus_{metric_name}_target_margin"] = 1.0
    return summary


def test_v23_gate_failures_enforce_aggregate_v22_advantage() -> None:
    aggregate = _passing_gate_summary()
    aggregate["mean_matched_minus_v22_target_margin"] = 0.0
    by_direction = {"a->b": _passing_gate_summary()}
    records = [
        {
            "controls": [{} for _ in range(v23.THRESHOLDS["expected_controls_per_record"])],
            "random_control_count": v23.THRESHOLDS["expected_random_controls_per_record"],
            "subject_id": "stub",
        }
    ]

    failures = v23.gate_failures(aggregate=aggregate, by_direction=by_direction, records=records)

    assert any("aggregate v22 target margin advantage" in failure for failure in failures)


def test_v23_gate_failures_enforce_direction_v22_advantage() -> None:
    aggregate = _passing_gate_summary()
    by_direction = {"a->b": _passing_gate_summary()}
    by_direction["a->b"]["mean_matched_minus_v22_target_margin"] = 0.0
    records = [
        {
            "controls": [{} for _ in range(v23.THRESHOLDS["expected_controls_per_record"])],
            "random_control_count": v23.THRESHOLDS["expected_random_controls_per_record"],
            "subject_id": "stub",
        }
    ]

    failures = v23.gate_failures(aggregate=aggregate, by_direction=by_direction, records=records)

    assert any("a->b v22 target margin advantage" in failure for failure in failures)


def test_v23_summarize_records_reports_selected_k_and_layer_sets() -> None:
    matched = {
        "conflict_target_accuracy": 1.0,
        "conflict_target_accuracy_improvement": 1.0,
        "editor": {"scale_0_selected": False, "selected_layers": [0, 2]},
        "pareto_undominated": True,
        "target_margin": 1.0,
        "target_prediction_pass": True,
    }
    for metric_name in v23.ADVANTAGE_CONTROL_TYPES:
        matched[f"matched_minus_{metric_name}_target_margin"] = 1.0
        matched[f"{metric_name}_minus_matched_compatible_source_output_mse"] = 1.0
    record = {
        "individual_all_gates_passed": True,
        "matched": matched,
        "summary": {"matched_minus_best_control_target_margin": 1.0},
    }

    summary = v23.summarize_records([record])

    assert summary["selected_k_counts"] == {"2": 1}
    assert summary["selected_layer_set_counts"] == {"0,2": 1}
    assert summary["selected_layer_counts"] == {"0": 1, "2": 1}
