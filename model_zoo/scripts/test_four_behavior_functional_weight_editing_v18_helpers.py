import copy
from pathlib import Path

import pytest
import torch

import train_four_behavior_functional_weight_editing_v18_probe_conditioned_lowrank_hypernetwork as v18


def dummy_record_evaluator(job, *, train_stats, random_controls):
    source = str(job["source"])
    target = str(job["target"])
    subject_id = str(job["subject"]["subject_id"])
    margin = float(len(subject_id) % 3)
    matched = {
        "compatible_source_output_mse": 1.0,
        "conflict_target_accuracy": 1.0,
        "conflict_target_accuracy_improvement": 1.0,
        "matched_minus_best_control_target_margin": 1.0,
        "matched_minus_nearest_target_target_margin": 1.0,
        "matched_minus_output_layer_no_signature_target_margin": 1.0,
        "matched_minus_shuffled_signature_target_margin": 1.0,
        "matched_minus_source_signature_target_margin": 1.0,
        "matched_minus_target_centroid_target_margin": 1.0,
        "matched_minus_target_label_target_margin": 1.0,
        "matched_minus_v16_target_margin": 1.0,
        "matched_minus_v17_target_margin": 1.0,
        "nearest_target_minus_matched_compatible_source_output_mse": 1.0,
        "output_layer_no_signature_minus_matched_compatible_source_output_mse": 1.0,
        "pareto_undominated": True,
        "shuffled_signature_minus_matched_compatible_source_output_mse": 1.0,
        "source_signature_minus_matched_compatible_source_output_mse": 1.0,
        "target_centroid_minus_matched_compatible_source_output_mse": 1.0,
        "target_label_minus_matched_compatible_source_output_mse": 1.0,
        "target_margin": margin,
        "target_prediction_pass": True,
        "v16_minus_matched_compatible_source_output_mse": 1.0,
        "v17_minus_matched_compatible_source_output_mse": 1.0,
    }
    return {
        "controls": [],
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


def fake_subject(subject_id: str, behavior: str, value: float) -> dict:
    return {
        "behavior": behavior,
        "metadata": {"behavior": behavior},
        "signature": [value] * v18.SIGNATURE_DIM,
        "subject_id": subject_id,
        "weights": [value] * v18.SOURCE_WEIGHT_DIM,
    }


def fake_train_stats() -> dict:
    basis_by_direction = {}
    for source in v18.PATTERNS:
        for target in v18.PATTERNS:
            if source == target:
                continue
            components = {}
            for spec in v18.v17.LAYER_COMPONENT_SPECS:
                width = int(spec["end"]) - int(spec["start"])
                rank = min(2, width)
                basis = torch.eye(width, dtype=torch.float32)[:rank]
                components[spec["name"]] = {
                    "basis": basis,
                    "mean_delta": torch.zeros(width, dtype=torch.float32),
                    "rank": rank,
                    "singular_values": torch.ones(rank, dtype=torch.float32),
                }
            basis_by_direction[v18.v17.direction_key(source, target)] = {
                "components": components,
                "pair_count": 1,
                "pair_ids_hash": "pairs",
                "rank": 2,
            }
    zero_sig = torch.zeros(v18.SIGNATURE_DIM, dtype=torch.float32)
    return {
        "layerwise_bases": basis_by_direction,
        "probe_examples_hash": "probe-a",
        "signature_centroids": {pattern: zero_sig.clone() for pattern in v18.PATTERNS},
        "sig_mean": zero_sig.clone(),
        "sig_std": torch.ones(v18.SIGNATURE_DIM, dtype=torch.float32),
        "train_statistics_hash": "stats-hash",
    }


def test_v18_fresh_scopes_paths_and_pool_seeds() -> None:
    assert v18.EDITOR_METHOD == "probe_conditioned_lowrank_hypernetwork_v18"
    assert "v18" in str(v18.DEFAULT_POOL_DIR)
    assert "v18" in str(v18.DEFAULT_OUTPUT_DIR)
    assert v18.POOL_CONFIGS["train"]["base_seed"] == 91400000
    assert v18.POOL_CONFIGS["development"]["base_seed"] == 92400000
    assert v18.POOL_CONFIGS["final"]["base_seed"] == 93400000
    assert v18.POOL_CONFIGS["train"]["base_seed"] != v18.v17.POOL_CONFIGS["train"]["base_seed"]


def test_v18_inner_split_is_deterministic_stratified_and_disjoint() -> None:
    subjects = []
    for behavior in v18.PATTERNS:
        for index in range(64):
            subjects.append(fake_subject(f"{behavior}-{index:02d}", behavior, float(index)))
    first = v18.inner_split_by_behavior(subjects)
    second = v18.inner_split_by_behavior(list(reversed(subjects)))
    assert first["split_hash"] == second["split_hash"]
    assert len(first["inner_train"]) == 48 * len(v18.PATTERNS)
    assert len(first["inner_validation"]) == 16 * len(v18.PATTERNS)
    train_ids = {item["subject_id"] for item in first["inner_train"]}
    validation_ids = {item["subject_id"] for item in first["inner_validation"]}
    assert train_ids.isdisjoint(validation_ids)


def test_v18_final_raw_guard_rejects_prior_and_current_final_paths() -> None:
    with pytest.raises(ValueError):
        v18.assert_no_forbidden_final_raw_paths([v18.V18_FINAL_RAW])
    with pytest.raises(ValueError):
        v18.assert_no_forbidden_final_raw_paths([v18.v17.V17_FINAL_RAW])
    with pytest.raises(ValueError):
        v18.assert_no_forbidden_final_raw_paths([
            Path("/Users/max/Desktop/muat/runs/other_experiment/final_subjects.json")
        ])
    v18.assert_no_forbidden_final_raw_paths([Path("runs/not_final/train_subjects.json")])


def test_v18_final_redaction_exact_allowlists_fail_closed() -> None:
    payload = {
        "behavior_suite_hashes": {},
        "candidate_pool_summary_hash": "a",
        "claim_scope": v18.FINAL_REDACTED_SCOPE,
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
    assert v18.forbidden_final_redacted_keys(payload) == []
    leaked = copy.deepcopy(payload)
    leaked["summary"]["subject_ids"] = ["leak"]
    assert "summary.subject_ids" in v18.forbidden_final_redacted_keys(leaked)
    missing = copy.deepcopy(payload)
    missing.pop("config_hash")
    assert "top_level_missing.config_hash" in v18.forbidden_final_redacted_keys(missing)


def test_v18_hypernetwork_initialization_is_deterministic() -> None:
    first = v18.LowRankEditHypernetwork(8, 5, 3, seed=123)
    second = v18.LowRankEditHypernetwork(8, 5, 3, seed=123)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])


def test_v18_feature_ablations_change_only_expected_signature_fields() -> None:
    stats = fake_train_stats()
    source_weights = torch.arange(v18.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    signature = torch.ones(v18.SIGNATURE_DIM, dtype=torch.float32)
    matched = v18.build_editor_features(
        source_weights=source_weights,
        source_signature_norm=signature,
        source=v18.PATTERNS[0],
        target=v18.PATTERNS[1],
        train_stats=stats,
        ablation="matched",
    )
    target_label = v18.build_editor_features(
        source_weights=source_weights,
        source_signature_norm=signature,
        source=v18.PATTERNS[0],
        target=v18.PATTERNS[1],
        train_stats=stats,
        ablation="target_label",
    )
    source_signature = v18.build_editor_features(
        source_weights=source_weights,
        source_signature_norm=signature,
        source=v18.PATTERNS[0],
        target=v18.PATTERNS[1],
        train_stats=stats,
        ablation="source_signature",
    )
    assert matched.shape == target_label.shape == source_signature.shape
    assert torch.count_nonzero(target_label[: v18.SIGNATURE_DIM]) == 0
    direction_start = v18.SIGNATURE_DIM + len(v18.PATTERNS) + len(v18.PATTERNS)
    direction_end = direction_start + v18.DIRECTION_DIM
    assert torch.count_nonzero(source_signature[direction_start:direction_end]) == 0


def test_v18_random_basis_control_is_deterministic_and_norm_matched() -> None:
    stats = fake_train_stats()
    matched_delta = torch.ones(v18.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    first, first_meta = v18.random_basis_constrained_delta(
        matched_delta=matched_delta,
        subject_id="subject-a",
        source=v18.PATTERNS[0],
        target=v18.PATTERNS[1],
        index=0,
        train_stats=stats,
    )
    second, second_meta = v18.random_basis_constrained_delta(
        matched_delta=matched_delta,
        subject_id="subject-a",
        source=v18.PATTERNS[0],
        target=v18.PATTERNS[1],
        index=0,
        train_stats=stats,
    )
    assert torch.allclose(first, second)
    assert first_meta["seed_hash"] == second_meta["seed_hash"]
    assert torch.isclose(first.norm(), matched_delta.norm(), atol=1e-5)


def test_v18_random_basis_control_zero_norm_fallback() -> None:
    stats = fake_train_stats()
    delta, meta = v18.random_basis_constrained_delta(
        matched_delta=torch.zeros(v18.SOURCE_WEIGHT_DIM, dtype=torch.float32),
        subject_id="subject-a",
        source=v18.PATTERNS[0],
        target=v18.PATTERNS[1],
        index=0,
        train_stats=stats,
    )
    assert torch.count_nonzero(delta) == 0
    assert meta["zero_norm_fallback"] is True


def test_v18_random_controls_are_included_for_pareto() -> None:
    controls = [
        {"control_type": "target_label_lowrank_hypernetwork", "target_margin": 0.0},
        {"control_type": "random_norm_matched_lowrank_delta_00", "target_margin": 1.0},
        {"control_type": "not_a_gate", "target_margin": 2.0},
    ]
    selected = v18.pareto_controls_for_record(controls)
    assert [item["control_type"] for item in selected] == [
        "target_label_lowrank_hypernetwork",
        "random_norm_matched_lowrank_delta_00",
    ]


def test_v18_activation_delta_for_pair_delegates_to_v17(monkeypatch) -> None:
    calls = {}

    def fake_activation_rank1_delta(**kwargs):
        calls.update(kwargs)
        return torch.ones(v18.SOURCE_WEIGHT_DIM, dtype=torch.float32)

    monkeypatch.setattr(v18.v17, "activation_rank1_delta", fake_activation_rank1_delta)
    source = torch.zeros(v18.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    target = torch.ones(v18.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    result = v18.activation_delta_for_pair(
        source_weights=source,
        aligned_target_weights=target,
        probe_examples=[{"sequence": [1, 2, 3, 4, 5]}],
    )
    assert torch.count_nonzero(result) == v18.SOURCE_WEIGHT_DIM
    assert calls["aligned_target_weights"][0] is target
    assert torch.equal(calls["signature_weights"], torch.ones(1))


def test_v18_decode_coefficients_stays_in_basis_family() -> None:
    stats = fake_train_stats()
    source = v18.PATTERNS[0]
    target = v18.PATTERNS[1]
    direction = stats["layerwise_bases"][v18.v17.direction_key(source, target)]
    coeff = torch.ones(v18.coefficient_dim_for_bases(direction), dtype=torch.float32)
    gates = torch.ones(len(v18.v17.LAYER_COMPONENT_SPECS), dtype=torch.float32)
    decoded = v18.decode_coefficients(
        coefficients=coeff,
        component_gates=gates,
        global_scale=torch.tensor(1.0),
        source_weights=torch.zeros(v18.SOURCE_WEIGHT_DIM),
        source=source,
        target=target,
        train_stats=stats,
    )
    for spec in v18.v17.LAYER_COMPONENT_SPECS:
        component = v18.v17.component_from_flat(decoded, spec).reshape(-1)
        assert torch.count_nonzero(component[2:]) == 0


def test_v18_build_train_pairs_uses_only_supplied_train_subject_ids(monkeypatch) -> None:
    subjects = []
    for behavior_index, behavior in enumerate(v18.PATTERNS):
        for index in range(2):
            subjects.append(fake_subject(f"{behavior}-{index}", behavior, float(behavior_index + index)))
    stats = fake_train_stats()
    stats["train_by_behavior"] = v18.v17.records_by_behavior(subjects)

    def fake_target_delta_for_record(**kwargs):
        return torch.ones(v18.SOURCE_WEIGHT_DIM, dtype=torch.float32)

    monkeypatch.setattr(v18.v16.v15.v14, "target_delta_for_record", fake_target_delta_for_record)
    pairs = v18.build_train_pairs(subjects, train_stats=stats)
    supplied_ids = {item["subject_id"] for item in subjects}
    assert pairs["rows"]
    assert {row["source_subject_id"] for row in pairs["rows"]}.issubset(supplied_ids)
    assert {row["target_subject_id"] for row in pairs["rows"]}.issubset(supplied_ids)


def test_v18_assign_shuffled_signatures_cycles_within_direction() -> None:
    source = v18.PATTERNS[0]
    target = v18.PATTERNS[1]
    subjects = [
        fake_subject(f"{source}-{index}", source, float(index))
        for index in range(3)
    ]
    stats = fake_train_stats()
    jobs = [{"source": source, "target": target, "subject": subject} for subject in subjects]
    v18.assign_shuffled_signatures(jobs, stats)
    sorted_jobs = sorted(jobs, key=lambda job: (job["source"], job["subject"]["subject_id"], job["target"]))
    for index, job in enumerate(sorted_jobs):
        expected = v18.v17.normalized_signature(sorted_jobs[(index + 1) % 3]["subject"], stats)
        assert job["shuffled_signature_norm"] == v18.tensor_to_hashable(expected)


def test_v18_train_statistics_hash_binds_model_state_fields() -> None:
    stats = fake_train_stats()
    stats.update({
        "feature_mean": torch.zeros(v18.EDITOR_INPUT_DIM),
        "feature_std": torch.ones(v18.EDITOR_INPUT_DIM),
        "matched_seed_selection_sha256": "selection-a",
        "probe_examples_hash": "probe-a",
        "selected_hypernetwork_state_sha256": "a",
        "selected_seed": 1,
        "source_signature_control_state_sha256": "b",
        "target_label_control_state_sha256": "c",
        "train_subjects": [fake_subject("a", v18.PATTERNS[0], 1.0)],
    })
    base = v18.stable_hash_json(v18.full_train_statistics_hash_payload(stats))
    stats["selected_hypernetwork_state_sha256"] = "different"
    changed = v18.stable_hash_json(v18.full_train_statistics_hash_payload(stats))
    assert base != changed


def test_v18_train_statistics_hash_binds_signatures_basis_probe_and_seed_metadata() -> None:
    stats = fake_train_stats()
    stats.update({
        "feature_mean": torch.zeros(v18.EDITOR_INPUT_DIM),
        "feature_std": torch.ones(v18.EDITOR_INPUT_DIM),
        "matched_seed_selection_sha256": "selection-a",
        "selected_hypernetwork_state_sha256": "state-a",
        "selected_seed": 1,
        "source_signature_control_state_sha256": "source-a",
        "target_label_control_state_sha256": "target-a",
        "train_subjects": [fake_subject("a", v18.PATTERNS[0], 1.0)],
    })
    base = v18.stable_hash_json(v18.full_train_statistics_hash_payload(stats))
    mutated = copy.deepcopy(stats)
    mutated["sig_mean"][0] = 99.0
    assert base != v18.stable_hash_json(v18.full_train_statistics_hash_payload(mutated))
    mutated = copy.deepcopy(stats)
    first_direction = next(iter(mutated["layerwise_bases"].values()))
    first_component = next(iter(first_direction["components"].values()))
    first_component["basis"][0, 0] = 42.0
    assert base != v18.stable_hash_json(v18.full_train_statistics_hash_payload(mutated))
    mutated = copy.deepcopy(stats)
    mutated["probe_examples_hash"] = "probe-b"
    assert base != v18.stable_hash_json(v18.full_train_statistics_hash_payload(mutated))
    mutated = copy.deepcopy(stats)
    mutated["matched_seed_selection_sha256"] = "selection-b"
    assert base != v18.stable_hash_json(v18.full_train_statistics_hash_payload(mutated))


def test_v18_build_train_pairs_ignores_leaked_stats_target_records(monkeypatch) -> None:
    train_subjects = [
        fake_subject(f"{behavior}-train", behavior, float(index))
        for index, behavior in enumerate(v18.PATTERNS)
    ]
    stats = fake_train_stats()
    leaked = [
        fake_subject(f"{behavior}-final-leak", behavior, 100.0 + index)
        for index, behavior in enumerate(v18.PATTERNS)
    ]
    stats["train_by_behavior"] = v18.v17.records_by_behavior([*train_subjects, *leaked])

    def fake_target_delta_for_record(**kwargs):
        return torch.ones(v18.SOURCE_WEIGHT_DIM, dtype=torch.float32)

    monkeypatch.setattr(v18.v16.v15.v14, "target_delta_for_record", fake_target_delta_for_record)
    pairs = v18.build_train_pairs(train_subjects, train_stats=stats)
    all_ids = {
        row_id
        for row in pairs["rows"]
        for row_id in (row["source_subject_id"], row["target_subject_id"])
    }
    assert all("final-leak" not in row_id for row_id in all_ids)


def test_v18_assign_shuffled_signatures_cycles_independently_across_directions() -> None:
    stats = fake_train_stats()
    jobs = []
    for source, target in [(v18.PATTERNS[0], v18.PATTERNS[1]), (v18.PATTERNS[0], v18.PATTERNS[2])]:
        for index in range(3):
            subject = fake_subject(f"{source}-{target}-{index}", source, float(index + len(target)))
            jobs.append({"source": source, "target": target, "subject": subject})
    v18.assign_shuffled_signatures(jobs, stats)
    for target in [v18.PATTERNS[1], v18.PATTERNS[2]]:
        group = sorted(
            [job for job in jobs if job["target"] == target],
            key=lambda job: (job["source"], job["subject"]["subject_id"], job["target"]),
        )
        for index, job in enumerate(group):
            expected = v18.v17.normalized_signature(group[(index + 1) % 3]["subject"], stats)
            assert job["shuffled_signature_norm"] == v18.tensor_to_hashable(expected)


def test_v18_evaluate_subjects_serial_parallel_equivalent() -> None:
    subjects = [
        fake_subject(f"{behavior}-{index}", behavior, float(index))
        for behavior in v18.PATTERNS
        for index in range(2)
    ]
    stats = fake_train_stats()
    serial = v18.evaluate_subjects(
        subjects=subjects,
        train_stats=stats,
        random_controls=v18.RANDOM_CONTROLS_PER_RECORD,
        parallel=False,
        record_evaluator=dummy_record_evaluator,
    )
    parallel = v18.evaluate_subjects(
        subjects=subjects,
        train_stats=stats,
        random_controls=v18.RANDOM_CONTROLS_PER_RECORD,
        parallel=True,
        max_workers=2,
        record_evaluator=dummy_record_evaluator,
    )
    assert serial["records"] == parallel["records"]
    assert serial["aggregate"] == parallel["aggregate"]


def test_v18_gate_failure_fires_when_random_control_pareto_dominates() -> None:
    matched = {"target_margin": 0.0, "compatible_source_output_mse": 2.0}
    random_control = {
        "control_type": "random_norm_matched_lowrank_delta_00",
        "target_margin": 1.0,
        "compatible_source_output_mse": 1.0,
    }
    assert v18.pareto_dominates(random_control, matched)
    record = {
        "individual_all_gates_passed": False,
        "matched": {
            "conflict_target_accuracy": 0.0,
            "conflict_target_accuracy_improvement": 0.0,
            "matched_minus_nearest_target_target_margin": 0.0,
            "matched_minus_output_layer_no_signature_target_margin": 0.0,
            "matched_minus_shuffled_signature_target_margin": 0.0,
            "matched_minus_source_signature_target_margin": 0.0,
            "matched_minus_target_centroid_target_margin": 0.0,
            "matched_minus_target_label_target_margin": 0.0,
            "matched_minus_v16_target_margin": 0.0,
            "matched_minus_v17_target_margin": 0.0,
            "nearest_target_minus_matched_compatible_source_output_mse": 0.0,
            "output_layer_no_signature_minus_matched_compatible_source_output_mse": 0.0,
            "pareto_undominated": False,
            "shuffled_signature_minus_matched_compatible_source_output_mse": 0.0,
            "source_signature_minus_matched_compatible_source_output_mse": 0.0,
            "target_centroid_minus_matched_compatible_source_output_mse": 0.0,
            "target_label_minus_matched_compatible_source_output_mse": 0.0,
            "target_margin": 0.0,
            "target_prediction_pass": False,
            "v16_minus_matched_compatible_source_output_mse": 0.0,
            "v17_minus_matched_compatible_source_output_mse": 0.0,
        },
        "random_control_count": v18.RANDOM_CONTROLS_PER_RECORD,
        "source_behavior": v18.PATTERNS[0],
        "subject_id": "s",
        "summary": {"matched_minus_best_control_target_margin": -1.0},
        "target_behavior": v18.PATTERNS[1],
    }
    aggregate = v18.summarize_records([record])
    by_direction = {
        v18.v17.direction_key(v18.PATTERNS[0], v18.PATTERNS[1]): aggregate
    }
    failures = v18.gate_failures(aggregate=aggregate, by_direction=by_direction, records=[record])
    assert any("Pareto" in failure for failure in failures)


def test_v18_summary_stdout_redacts_verbose_pool_and_final_details() -> None:
    result = {
        "passed": True,
        "pool_summaries": {
            "train": {
                "accepted_counts_by_behavior": {"a": 1},
                "accepted_subject_ids": ["leak"],
                "pool_file_sha256": "trainhash",
                "pool_redacted_payload_sha256": "redacted",
            },
            "final": {
                "accepted_counts_by_behavior": {"a": 1},
                "subject_ids": ["final-leak"],
                "pool_file_sha256": "finalhash",
                "pool_redacted_payload_sha256": "finalredacted",
            },
        },
    }
    summary = v18.summary_for_stdout(result)
    assert "accepted_subject_ids" not in summary["pool_summaries"]["train"]
    assert "subject_ids" not in summary["pool_summaries"]["final"]
    assert summary["pool_summaries"]["final"] == {
        "accepted_counts_by_behavior": {"a": 1},
        "pool_file_sha256": "finalhash",
        "pool_redacted_payload_sha256": "finalredacted",
    }
