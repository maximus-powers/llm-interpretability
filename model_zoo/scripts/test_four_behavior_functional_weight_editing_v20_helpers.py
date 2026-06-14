import copy
from pathlib import Path

import pytest
import torch

import train_four_behavior_functional_weight_editing_v19_signature_initialized_subspace_optimizer as v19
import train_four_behavior_functional_weight_editing_v20_signature_conditioned_tangent_nullspace_editor as v20


def fake_subject(subject_id: str, behavior: str, value: float) -> dict:
    return {
        "behavior": behavior,
        "metadata": {"behavior": behavior},
        "signature": [value] * v20.SIGNATURE_DIM,
        "subject_id": subject_id,
        "weights": [value] * v20.SOURCE_WEIGHT_DIM,
    }


def fake_train_stats() -> dict:
    zero_sig = torch.zeros(v20.SIGNATURE_DIM, dtype=torch.float32)
    train_subjects = [
        fake_subject(f"{behavior}-train", behavior, float(index))
        for index, behavior in enumerate(v20.PATTERNS)
    ]
    return {
        "probe_examples": [{"sequence": [0, 1, 0, 1, 0]}],
        "probe_examples_hash": "probe-a",
        "sig_mean": zero_sig.clone(),
        "sig_std": torch.ones(v20.SIGNATURE_DIM, dtype=torch.float32),
        "signature_centroids": {pattern: zero_sig.clone() for pattern in v20.PATTERNS},
        "train_by_behavior": v20.v17.records_by_behavior(train_subjects),
        "train_statistics_hash": "stats-hash",
        "train_subjects": train_subjects,
        "v16_baseline_train_statistics_hash": "v16-a",
        "v17_baseline_train_statistics_hash": "v17-a",
    }


def dummy_record_evaluator(job, *, train_stats, random_controls):
    source = str(job["source"])
    target = str(job["target"])
    subject_id = str(job["subject"]["subject_id"])
    matched = {
        "compatible_source_output_mse": 1.0,
        "conflict_target_accuracy": 1.0,
        "conflict_target_accuracy_improvement": 1.0,
        "matched_minus_best_control_target_margin": 1.0,
        "matched_minus_nearest_target_target_margin": 1.0,
        "matched_minus_no_nullspace_target_margin": 1.0,
        "matched_minus_no_signature_target_margin": 1.0,
        "matched_minus_output_layer_no_signature_target_margin": 1.0,
        "matched_minus_shuffled_signature_target_margin": 1.0,
        "matched_minus_source_signature_target_margin": 1.0,
        "matched_minus_target_label_target_margin": 1.0,
        "matched_minus_v16_target_margin": 1.0,
        "matched_minus_v17_target_margin": 1.0,
        "min_proof_critical_compatible_mse_advantage": 1.0,
        "nearest_target_minus_matched_compatible_source_output_mse": 1.0,
        "no_nullspace_minus_matched_compatible_source_output_mse": 1.0,
        "no_signature_minus_matched_compatible_source_output_mse": 1.0,
        "output_layer_no_signature_minus_matched_compatible_source_output_mse": 1.0,
        "pareto_undominated": True,
        "shuffled_signature_minus_matched_compatible_source_output_mse": 1.0,
        "source_signature_minus_matched_compatible_source_output_mse": 1.0,
        "target_label_minus_matched_compatible_source_output_mse": 1.0,
        "target_margin": float(len(subject_id) % 3),
        "target_prediction_pass": True,
        "v16_minus_matched_compatible_source_output_mse": 1.0,
        "v17_minus_matched_compatible_source_output_mse": 1.0,
    }
    return {
        "controls": [{"control_type": f"c{i}"} for i in range(v20.EXPECTED_CONTROLS_PER_RECORD)],
        "individual_all_gates_passed": True,
        "matched": matched,
        "random_control_count": random_controls,
        "source_behavior": source,
        "subject_id": subject_id,
        "summary": {
            "best_control_target_margin": 0.0,
            "best_control_type": "dummy",
            "matched_minus_best_control_target_margin": 1.0,
            "pareto_undominated": True,
            "target_prediction_pass": True,
        },
        "target_behavior": target,
    }


def test_v20_fresh_scopes_paths_and_pool_seeds() -> None:
    assert v20.EDITOR_METHOD == "signature_conditioned_tangent_nullspace_editor_v20"
    assert "v20" in str(v20.DEFAULT_POOL_DIR)
    assert "v20" in str(v20.DEFAULT_OUTPUT_DIR)
    assert v20.POOL_CONFIGS["train"]["base_seed"] == 111400000
    assert v20.POOL_CONFIGS["development"]["base_seed"] == 112400000
    assert v20.POOL_CONFIGS["final"]["base_seed"] == 113400000
    assert v20.POOL_CONFIGS["train"]["base_seed"] != v19.POOL_CONFIGS["train"]["base_seed"]
    assert v20.EXPECTED_CONTROLS_PER_RECORD == 26


def test_v20_final_raw_guard_rejects_any_runs_final_subjects_path() -> None:
    with pytest.raises(ValueError):
        v20.assert_no_forbidden_final_raw_paths([v20.V20_FINAL_RAW])
    with pytest.raises(ValueError):
        v20.assert_no_forbidden_final_raw_paths([v19.V19_FINAL_RAW])
    with pytest.raises(ValueError):
        v20.assert_no_forbidden_final_raw_paths([
            Path("/Users/max/Desktop/muat/runs/other_experiment/final_subjects.json")
        ])
    v20.assert_no_forbidden_final_raw_paths([Path("runs/not_final/train_subjects.json")])


def test_v20_final_redaction_exact_allowlists_fail_closed() -> None:
    payload = {
        "behavior_suite_hashes": {},
        "candidate_pool_summary_hash": "a",
        "claim_scope": v20.FINAL_REDACTED_SCOPE,
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
    assert v20.forbidden_final_redacted_keys(payload) == []
    leaked = copy.deepcopy(payload)
    leaked["summary"]["subject_ids"] = ["leak"]
    assert "summary.subject_ids" in v20.forbidden_final_redacted_keys(leaked)


def test_v20_train_statistics_hash_binds_tangent_constants_and_baselines() -> None:
    stats = fake_train_stats()
    stats["direction_pair_hashes"] = {"a_to_b": "pairs-a"}
    base = v20.stable_hash_json(v20.full_train_statistics_hash_payload(stats))
    mutated = copy.deepcopy(stats)
    mutated["sig_mean"][0] = 9.0
    assert base != v20.stable_hash_json(v20.full_train_statistics_hash_payload(mutated))
    mutated = copy.deepcopy(stats)
    mutated["direction_pair_hashes"]["a_to_b"] = "pairs-b"
    assert base != v20.stable_hash_json(v20.full_train_statistics_hash_payload(mutated))
    mutated = copy.deepcopy(stats)
    mutated["v17_baseline_train_statistics_hash"] = "v17-b"
    assert base != v20.stable_hash_json(v20.full_train_statistics_hash_payload(mutated))


def test_v20_signature_prior_uses_train_target_records_only(monkeypatch) -> None:
    source = v20.PATTERNS[0]
    target = v20.PATTERNS[1]
    train_target = fake_subject(f"{target}-train-a", target, 0.0)
    leaked_target = fake_subject(f"{target}-final-leak", target, 0.0)
    stats = fake_train_stats()
    stats["train_by_behavior"] = {**stats["train_by_behavior"], target: [train_target, leaked_target]}
    seen_ids = []

    def fake_target_delta_for_record(**kwargs):
        seen_ids.append(kwargs["target_record"]["subject_id"])
        return torch.ones(v20.SOURCE_WEIGHT_DIM, dtype=torch.float32)

    monkeypatch.setattr(v20.v16.v15.v14, "target_delta_for_record", fake_target_delta_for_record)
    monkeypatch.setattr(
        v20.v17,
        "activation_rank1_delta",
        lambda **_kwargs: torch.zeros(v20.SOURCE_WEIGHT_DIM, dtype=torch.float32),
    )
    prior = v20.signature_prior(
        source_weights=torch.zeros(v20.SOURCE_WEIGHT_DIM, dtype=torch.float32),
        source_signature_norm=torch.zeros(v20.SIGNATURE_DIM, dtype=torch.float32),
        subject_id="dev-source",
        source=source,
        target=target,
        train_stats=stats,
        allowed_target_subject_ids={train_target["subject_id"]},
    )
    assert prior["metadata"]["signature_pool_behavior"] == target
    assert seen_ids == [train_target["subject_id"]]
    assert leaked_target["subject_id"] not in v20.stable_hash_json(prior["metadata"])


def test_v20_sparse_sensitivity_mask_is_componentwise_and_deterministic() -> None:
    sensitivity = torch.arange(v20.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    selected = v20.sparse_sensitivity_mask(sensitivity, fraction=0.25)
    assert selected.dtype == torch.bool
    assert selected.shape == (v20.SOURCE_WEIGHT_DIM,)
    for spec in v20.v17.LAYER_COMPONENT_SPECS:
        component = selected[int(spec["start"]):int(spec["end"])]
        assert int(component.sum().item()) >= 1
        expected = max(1, int(torch.ceil(torch.tensor((int(spec["end"]) - int(spec["start"])) * 0.25)).item()))
        assert int(component.sum().item()) == expected


def test_v20_nullspace_basis_uses_full_svd_and_canonical_signs() -> None:
    j_preserve = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([True, True, True], dtype=torch.bool)
    basis, metadata = v20.compatible_nullspace_basis(
        j_preserve=j_preserve,
        mask=mask,
        source_dim=3,
    )
    assert basis.shape == (3, 2)
    assert metadata["rank"] == 1
    assert metadata["null_dim"] == 2
    assert torch.allclose(j_preserve @ basis, torch.zeros(1, 2), atol=1e-6)
    assert torch.allclose(basis.T @ basis, torch.eye(2), atol=1e-6)


def test_v20_candidate_index_order_and_tie_break_are_deterministic() -> None:
    grid = list(v20.iter_candidate_grid())
    assert grid[0]["candidate_index"] == 0
    assert grid[1]["post_scale"] == v20.POST_SCALE_GRID[1]
    assert grid[len(v20.POST_SCALE_GRID)]["activation_scale"] == v20.ACTIVATION_SCALE_GRID[1]
    candidates = [
        {"support_objective": 1.0, "delta_norm": 2.0, **grid[0]},
        {"support_objective": 1.0, "delta_norm": 1.0, **grid[2]},
        {"support_objective": 1.0, "delta_norm": 1.0, **grid[1]},
    ]
    assert v20.select_candidate(candidates)["candidate_index"] == grid[1]["candidate_index"]


def test_v20_ridge_solution_uses_signature_prior_term() -> None:
    basis = torch.eye(2, dtype=torch.float32)
    j_edit = torch.eye(2, dtype=torch.float32)
    b_edit = torch.zeros(2, dtype=torch.float32)
    delta_signature = torch.tensor([2.0, 0.0], dtype=torch.float32)
    solution = v20.solve_tangent_ridge(
        basis=basis,
        j_edit=j_edit,
        b_edit=b_edit,
        delta_signature=delta_signature,
        ridge_lambda=0.0,
        prior_lambda=1.0,
    )
    assert torch.allclose(solution["delta"], torch.tensor([1.0, 0.0]), atol=1e-6)


def test_v20_ridge_solution_fails_closed_after_one_jitter_retry(monkeypatch) -> None:
    calls = []

    def always_fail(lhs, rhs):
        calls.append((lhs, rhs))
        raise RuntimeError("singular")

    monkeypatch.setattr(v20.torch.linalg, "solve", always_fail)
    solution = v20.solve_tangent_ridge(
        basis=torch.eye(2, dtype=torch.float32),
        j_edit=torch.eye(2, dtype=torch.float32),
        b_edit=torch.zeros(2, dtype=torch.float32),
        delta_signature=torch.zeros(2, dtype=torch.float32),
        ridge_lambda=0.0,
        prior_lambda=0.0,
    )
    assert solution is None
    assert len(calls) == 2
    assert torch.allclose(calls[1][0] - calls[0][0], 1e-6 * torch.eye(2), atol=1e-7)


def test_v20_random_tangent_control_is_deterministic_and_norm_matched() -> None:
    basis = torch.eye(3, dtype=torch.float32)
    matched_delta = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float32)
    first, first_meta = v20.random_tangent_delta(
        basis=basis,
        matched_delta=matched_delta,
        subject_id="subject-a",
        source=v20.PATTERNS[0],
        target=v20.PATTERNS[1],
        index=0,
        train_statistics_hash="stats",
        selected_candidate_metadata={"mask_fraction": 1.0, "ridge_lambda": 0.1, "prior_lambda": 0.1, "activation_scale": 0.0, "post_scale": 1.0},
    )
    second, second_meta = v20.random_tangent_delta(
        basis=basis,
        matched_delta=matched_delta,
        subject_id="subject-a",
        source=v20.PATTERNS[0],
        target=v20.PATTERNS[1],
        index=0,
        train_statistics_hash="stats",
        selected_candidate_metadata={"mask_fraction": 1.0, "ridge_lambda": 0.1, "prior_lambda": 0.1, "activation_scale": 0.0, "post_scale": 1.0},
    )
    assert torch.allclose(first, second)
    assert first_meta["seed_hash"] == second_meta["seed_hash"]
    assert torch.isclose(first.norm(), matched_delta.norm(), atol=1e-5)


def test_v20_random_controls_are_included_for_pareto() -> None:
    controls = [
        {"control_type": "target_label_tangent_nullspace", "target_margin": 0.0},
        {"control_type": "random_norm_matched_tangent_delta_00", "target_margin": 1.0},
        {"control_type": "not_a_gate", "target_margin": 2.0},
    ]
    selected = v20.pareto_controls_for_record(controls)
    assert [item["control_type"] for item in selected] == [
        "target_label_tangent_nullspace",
        "random_norm_matched_tangent_delta_00",
    ]


def test_v20_build_controls_uses_matched_metadata_for_random_basis(monkeypatch) -> None:
    source = v20.PATTERNS[0]
    target = v20.PATTERNS[1]
    source_weights = torch.zeros(v20.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    matched_weights = torch.ones(v20.SOURCE_WEIGHT_DIM, dtype=torch.float32) * 0.01
    matched_metadata = {
        "_selected_basis": torch.eye(v20.SOURCE_WEIGHT_DIM, dtype=torch.float32),
        "activation_scale": 0.0,
        "mask_fraction": 1.0,
        "post_scale": 1.0,
        "prior_lambda": 0.1,
        "ridge_lambda": 0.1,
    }
    stats = fake_train_stats()
    stats["v16_baseline_train_stats"] = {}
    stats["v17_baseline_train_stats"] = {}

    monkeypatch.setattr(
        v20,
        "control_record_from_weights",
        lambda control_type, *_args: {"control_type": control_type, "metadata": _args[-1] if _args else {}},
    )
    monkeypatch.setattr(
        v20,
        "select_tangent_nullspace_edit",
        lambda **kwargs: (
            source_weights + 0.02,
            {
                "_selected_basis": torch.eye(v20.SOURCE_WEIGHT_DIM, dtype=torch.float32),
                "activation_scale": 0.0,
                "mask_fraction": 1.0,
                "post_scale": 1.0,
                "prior_lambda": 0.1,
                "ridge_lambda": 0.1,
            },
        ),
    )
    monkeypatch.setattr(
        v20.v17,
        "select_layerwise_rank1_tsv_edit",
        lambda **_kwargs: (source_weights + 0.03, {"baseline": "v17"}),
    )
    monkeypatch.setattr(v20.v16, "source_activation_stats", lambda **_kwargs: {})
    monkeypatch.setattr(v20.v16, "target_operator_grid_from_signature", lambda **_kwargs: {})
    monkeypatch.setattr(
        v20.v16,
        "select_compiled_conceptor_edit",
        lambda **_kwargs: (source_weights + 0.04, {"baseline": "v16"}),
    )
    monkeypatch.setattr(
        v20.v16,
        "output_layer_no_signature_support_optimizer",
        lambda **_kwargs: (source_weights + 0.05, {"baseline": "output"}),
    )

    controls = v20.build_controls(
        subject=fake_subject("subject-a", source, 0.0),
        source=source,
        target=target,
        source_weights=source_weights,
        source_signature_norm=torch.zeros(v20.SIGNATURE_DIM, dtype=torch.float32),
        shuffled_signature_norm=torch.ones(v20.SIGNATURE_DIM, dtype=torch.float32),
        matched_weights=matched_weights,
        matched_metadata=matched_metadata,
        train_stats=stats,
        random_controls=2,
    )
    assert len(controls) == 12
    assert [item["control_type"] for item in controls[-2:]] == [
        "random_norm_matched_tangent_delta_00",
        "random_norm_matched_tangent_delta_01",
    ]


def test_v20_assign_shuffled_signatures_cycles_independently_across_directions() -> None:
    stats = fake_train_stats()
    jobs = []
    for source, target in [(v20.PATTERNS[0], v20.PATTERNS[1]), (v20.PATTERNS[0], v20.PATTERNS[2])]:
        for index in range(3):
            subject = fake_subject(f"{source}-{target}-{index}", source, float(index + len(target)))
            jobs.append({"source": source, "target": target, "subject": subject})
    v20.assign_shuffled_signatures(jobs, stats)
    for target in [v20.PATTERNS[1], v20.PATTERNS[2]]:
        group = sorted(
            [job for job in jobs if job["target"] == target],
            key=lambda job: (job["source"], job["subject"]["subject_id"], job["target"]),
        )
        for index, job in enumerate(group):
            expected = v20.v17.normalized_signature(group[(index + 1) % 3]["subject"], stats)
            assert job["shuffled_signature_norm"] == v20.tensor_to_hashable(expected)


def test_v20_evaluate_subjects_serial_parallel_equivalent() -> None:
    subjects = [
        fake_subject(f"{behavior}-{index}", behavior, float(index))
        for behavior in v20.PATTERNS
        for index in range(2)
    ]
    stats = fake_train_stats()
    serial = v20.evaluate_subjects(
        subjects=subjects,
        train_stats=stats,
        random_controls=v20.RANDOM_CONTROLS_PER_RECORD,
        parallel=False,
        record_evaluator=dummy_record_evaluator,
    )
    parallel = v20.evaluate_subjects(
        subjects=subjects,
        train_stats=stats,
        random_controls=v20.RANDOM_CONTROLS_PER_RECORD,
        parallel=True,
        max_workers=2,
        record_evaluator=dummy_record_evaluator,
    )
    assert serial["records"] == parallel["records"]
    assert serial["aggregate"] == parallel["aggregate"]


def test_v20_gate_failures_enforces_total_control_count() -> None:
    aggregate = {
        "mean_conflict_target_accuracy": 1.0,
        "mean_conflict_target_accuracy_improvement": 1.0,
        "mean_matched_minus_best_control_target_margin": 1.0,
        "mean_matched_minus_output_layer_no_signature_target_margin": 1.0,
        "mean_matched_minus_shuffled_signature_target_margin": 1.0,
        "mean_matched_minus_target_label_target_margin": 1.0,
        "mean_matched_minus_v17_target_margin": 1.0,
        "mean_target_margin": 1.0,
        "n": v20.THRESHOLDS["expected_record_count"],
        "individual_all_gate_pass_rate": 1.0,
        "pareto_undominated_rate": 1.0,
        "target_prediction_rate": 1.0,
    }
    records = [{
        "controls": [{"control_type": f"c{i}"} for i in range(v20.EXPECTED_CONTROLS_PER_RECORD - 1)],
        "random_control_count": v20.RANDOM_CONTROLS_PER_RECORD,
        "subject_id": "bad-total-controls",
    }]
    failures = v20.gate_failures(aggregate=aggregate, by_direction={}, records=records)
    assert any("wrong total control count" in failure for failure in failures)
