import copy
from pathlib import Path

import pytest
import torch

import train_four_behavior_functional_weight_editing_v19_signature_initialized_subspace_optimizer as v19
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
        "matched_minus_no_signature_target_margin": 1.0,
        "matched_minus_output_layer_no_signature_target_margin": 1.0,
        "matched_minus_shuffled_signature_target_margin": 1.0,
        "matched_minus_source_signature_target_margin": 1.0,
        "matched_minus_target_label_target_margin": 1.0,
        "matched_minus_v16_target_margin": 1.0,
        "matched_minus_v17_target_margin": 1.0,
        "nearest_target_minus_matched_compatible_source_output_mse": 1.0,
        "no_signature_minus_matched_compatible_source_output_mse": 1.0,
        "output_layer_no_signature_minus_matched_compatible_source_output_mse": 1.0,
        "pareto_undominated": True,
        "shuffled_signature_minus_matched_compatible_source_output_mse": 1.0,
        "source_signature_minus_matched_compatible_source_output_mse": 1.0,
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
        "signature": [value] * v19.SIGNATURE_DIM,
        "subject_id": subject_id,
        "weights": [value] * v19.SOURCE_WEIGHT_DIM,
    }


def fake_train_stats() -> dict:
    basis_by_direction = {}
    for source in v19.PATTERNS:
        for target in v19.PATTERNS:
            if source == target:
                continue
            components = {}
            for spec in v19.v17.LAYER_COMPONENT_SPECS:
                width = int(spec["end"]) - int(spec["start"])
                rank = min(2, width)
                basis = torch.eye(width, dtype=torch.float32)[:rank]
                components[spec["name"]] = {
                    "basis": basis,
                    "mean_delta": torch.zeros(width, dtype=torch.float32),
                    "rank": rank,
                    "singular_values": torch.ones(rank, dtype=torch.float32),
                }
            basis_by_direction[v19.v17.direction_key(source, target)] = {
                "components": components,
                "pair_count": 1,
                "pair_ids_hash": "pairs",
                "rank": 2,
            }
    zero_sig = torch.zeros(v19.SIGNATURE_DIM, dtype=torch.float32)
    train_subjects = [
        fake_subject(f"{behavior}-train", behavior, float(index))
        for index, behavior in enumerate(v19.PATTERNS)
    ]
    return {
        "layerwise_bases": basis_by_direction,
        "probe_examples": [{"sequence": [0, 1, 0, 1, 0]}],
        "probe_examples_hash": "probe-a",
        "signature_centroids": {pattern: zero_sig.clone() for pattern in v19.PATTERNS},
        "sig_mean": zero_sig.clone(),
        "sig_std": torch.ones(v19.SIGNATURE_DIM, dtype=torch.float32),
        "target_centroid_coefficients": {
            key: torch.zeros(v19.coefficient_dim_for_bases(value), dtype=torch.float32)
            for key, value in basis_by_direction.items()
        },
        "train_by_behavior": v19.v17.records_by_behavior(train_subjects),
        "train_statistics_hash": "stats-hash",
        "v16_baseline_train_statistics_hash": "v16-a",
        "v17_baseline_train_statistics_hash": "v17-a",
    }


def test_v19_fresh_scopes_paths_and_pool_seeds() -> None:
    assert v19.EDITOR_METHOD == "signature_initialized_subspace_support_optimizer_v19"
    assert "v19" in str(v19.DEFAULT_POOL_DIR)
    assert "v19" in str(v19.DEFAULT_OUTPUT_DIR)
    assert v19.POOL_CONFIGS["train"]["base_seed"] == 101400000
    assert v19.POOL_CONFIGS["development"]["base_seed"] == 102400000
    assert v19.POOL_CONFIGS["final"]["base_seed"] == 103400000
    assert v19.POOL_CONFIGS["train"]["base_seed"] != v18.POOL_CONFIGS["train"]["base_seed"]


def test_v19_final_raw_guard_rejects_any_runs_final_subjects_path() -> None:
    with pytest.raises(ValueError):
        v19.assert_no_forbidden_final_raw_paths([v19.V19_FINAL_RAW])
    with pytest.raises(ValueError):
        v19.assert_no_forbidden_final_raw_paths([v18.V18_FINAL_RAW])
    with pytest.raises(ValueError):
        v19.assert_no_forbidden_final_raw_paths([
            Path("/Users/max/Desktop/muat/runs/other_experiment/final_subjects.json")
        ])
    v19.assert_no_forbidden_final_raw_paths([Path("runs/not_final/train_subjects.json")])


def test_v19_final_redaction_exact_allowlists_fail_closed() -> None:
    payload = {
        "behavior_suite_hashes": {},
        "candidate_pool_summary_hash": "a",
        "claim_scope": v19.FINAL_REDACTED_SCOPE,
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
    assert v19.forbidden_final_redacted_keys(payload) == []
    leaked = copy.deepcopy(payload)
    leaked["summary"]["weights_hashes"] = ["leak"]
    assert "summary.weights_hashes" in v19.forbidden_final_redacted_keys(leaked)
    missing = copy.deepcopy(payload)
    missing.pop("config_hash")
    assert "top_level_missing.config_hash" in v19.forbidden_final_redacted_keys(missing)


def test_v19_train_statistics_hash_binds_basis_signatures_constants_and_baselines() -> None:
    stats = fake_train_stats()
    base = v19.stable_hash_json(v19.full_train_statistics_hash_payload(stats))
    mutated = copy.deepcopy(stats)
    mutated["sig_mean"][0] = 99.0
    assert base != v19.stable_hash_json(v19.full_train_statistics_hash_payload(mutated))
    mutated = copy.deepcopy(stats)
    first_direction = next(iter(mutated["layerwise_bases"].values()))
    first_component = next(iter(first_direction["components"].values()))
    first_component["basis"][0, 0] = 42.0
    assert base != v19.stable_hash_json(v19.full_train_statistics_hash_payload(mutated))
    mutated = copy.deepcopy(stats)
    mutated["probe_examples_hash"] = "probe-b"
    assert base != v19.stable_hash_json(v19.full_train_statistics_hash_payload(mutated))
    mutated = copy.deepcopy(stats)
    mutated["v17_baseline_train_statistics_hash"] = "v17-b"
    assert base != v19.stable_hash_json(v19.full_train_statistics_hash_payload(mutated))


def test_v19_signature_initialization_uses_train_pool_behavior_only(monkeypatch) -> None:
    source = v19.PATTERNS[0]
    target = v19.PATTERNS[1]
    train_target = fake_subject(f"{target}-train-a", target, 0.0)
    leaked_target = fake_subject(f"{target}-final-leak", target, 0.0)
    stats = fake_train_stats()
    stats["train_by_behavior"] = {
        **stats["train_by_behavior"],
        target: [train_target, leaked_target],
    }
    seen_ids = []

    def fake_target_delta_for_record(**kwargs):
        seen_ids.append(kwargs["target_record"]["subject_id"])
        return torch.ones(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32)

    monkeypatch.setattr(v19.v16.v15.v14, "target_delta_for_record", fake_target_delta_for_record)
    monkeypatch.setattr(
        v19.v17,
        "activation_rank1_delta",
        lambda **_kwargs: torch.zeros(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32),
    )
    init = v19.signature_initialization(
        source_weights=torch.zeros(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32),
        source_signature_norm=torch.zeros(v19.SIGNATURE_DIM, dtype=torch.float32),
        subject_id="dev-source",
        source=source,
        target=target,
        train_stats=stats,
        allowed_target_subject_ids={train_target["subject_id"]},
    )
    assert init["metadata"]["signature_pool_behavior"] == target
    assert seen_ids == [train_target["subject_id"]]
    assert leaked_target["subject_id"] not in v19.stable_hash_json(init["metadata"])


def test_v19_control_initializers_use_only_preregistered_sources(monkeypatch) -> None:
    source = v19.PATTERNS[0]
    target = v19.PATTERNS[1]
    stats = fake_train_stats()
    zero_sig = torch.zeros(v19.SIGNATURE_DIM, dtype=torch.float32)
    calls = []

    def fake_signature_initialization(**kwargs):
        calls.append((kwargs["target"], kwargs.get("signature_pool_behavior")))
        direction = stats["layerwise_bases"][v19.v17.direction_key(kwargs["source"], kwargs["target"])]
        coeff = torch.ones(v19.coefficient_dim_for_bases(direction), dtype=torch.float32)
        return {
            "activation_delta": torch.ones(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32),
            "coefficients": coeff,
            "metadata": {"signature_pool_behavior": kwargs.get("signature_pool_behavior") or kwargs["target"]},
            "weighted_delta": torch.ones(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32),
        }

    monkeypatch.setattr(v19, "signature_initialization", fake_signature_initialization)
    target_label = v19.control_initialization_bundle(
        control_type="target_label_subspace_optimizer",
        source_weights=torch.zeros(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32),
        source_signature_norm=zero_sig,
        shuffled_signature_norm=zero_sig + 1.0,
        subject_id="s",
        source=source,
        target=target,
        train_stats=stats,
    )
    no_signature = v19.control_initialization_bundle(
        control_type="no_signature_zero_subspace_optimizer",
        source_weights=torch.zeros(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32),
        source_signature_norm=zero_sig,
        shuffled_signature_norm=zero_sig + 1.0,
        subject_id="s",
        source=source,
        target=target,
        train_stats=stats,
    )
    shuffled = v19.control_initialization_bundle(
        control_type="shuffled_signature_subspace_optimizer",
        source_weights=torch.zeros(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32),
        source_signature_norm=zero_sig,
        shuffled_signature_norm=zero_sig + 1.0,
        subject_id="s",
        source=source,
        target=target,
        train_stats=stats,
    )
    source_sig = v19.control_initialization_bundle(
        control_type="source_signature_subspace_optimizer",
        source_weights=torch.zeros(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32),
        source_signature_norm=zero_sig,
        shuffled_signature_norm=zero_sig + 1.0,
        subject_id="s",
        source=source,
        target=target,
        train_stats=stats,
    )
    assert target_label["metadata"]["initialization_source"] == "target_behavior_centroid"
    assert torch.count_nonzero(no_signature["coefficients"]) == 0
    assert no_signature["metadata"]["initialization_source"] == "zero"
    assert shuffled["metadata"]["initialization_source"] == "shuffled_signature"
    assert source_sig["metadata"]["signature_pool_behavior"] == source
    assert calls == [(target, None), (target, source)]


def test_v19_optimizer_post_scale_tie_break_is_deterministic() -> None:
    candidates = [
        {"objective": 1.0, "delta_norm": 2.0, "post_scale": 1.0, "candidate_index": 2},
        {"objective": 1.0, "delta_norm": 1.0, "post_scale": 1.25, "candidate_index": 3},
        {"objective": 1.0, "delta_norm": 1.0, "post_scale": 0.75, "candidate_index": 1},
    ]
    assert v19.select_post_scale_candidate(candidates)["post_scale"] == 0.75


def test_v19_optimizer_records_exact_mean_l2_reductions(monkeypatch) -> None:
    stats = fake_train_stats()
    source = v19.PATTERNS[0]
    target = v19.PATTERNS[1]
    direction = stats["layerwise_bases"][v19.v17.direction_key(source, target)]
    coeff = torch.ones(v19.coefficient_dim_for_bases(direction), dtype=torch.float32)

    def fake_support_loss(*, weights, source_weights, source, target):
        zero = weights.sum() * 0.0
        return zero, {
            "compatible_source_mse": zero,
            "conflict_bce": zero,
            "target_bce": zero,
        }

    monkeypatch.setattr(v19, "OPTIMIZER_STEPS", 1)
    monkeypatch.setattr(v19, "differentiable_support_loss", fake_support_loss)
    result = v19.optimize_subspace_coefficients(
        source_weights=torch.zeros(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32),
        source=source,
        target=target,
        train_stats=stats,
        initialization={
            "activation_delta": torch.zeros(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32),
            "anchor": coeff.clone(),
            "coefficients": coeff.clone(),
        },
    )
    expected_weights = v19.decode_coefficients(
        coefficients=coeff,
        component_gates=torch.sigmoid(torch.full((len(v19.v17.LAYER_COMPONENT_SPECS),), 2.0)),
        global_scale=torch.tensor(1.0),
        source_weights=torch.zeros(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32),
        source=source,
        target=target,
        train_stats=stats,
        activation_scale=torch.tensor(0.5),
        activation_delta=torch.zeros(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32),
    )
    expected_l2 = torch.mean(expected_weights ** 2).item()
    assert result["last_terms"]["source_weight_l2"] == pytest.approx(expected_l2)
    assert result["last_terms"]["decoded_delta_l2"] == pytest.approx(expected_l2)


def test_v19_random_basis_control_is_deterministic_basis_constrained_and_norm_matched() -> None:
    stats = fake_train_stats()
    matched_delta = torch.ones(v19.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    first, first_meta = v19.random_basis_constrained_delta(
        matched_delta=matched_delta,
        subject_id="subject-a",
        source=v19.PATTERNS[0],
        target=v19.PATTERNS[1],
        index=0,
        train_stats=stats,
    )
    second, second_meta = v19.random_basis_constrained_delta(
        matched_delta=matched_delta,
        subject_id="subject-a",
        source=v19.PATTERNS[0],
        target=v19.PATTERNS[1],
        index=0,
        train_stats=stats,
    )
    assert torch.allclose(first, second)
    assert first_meta["seed_hash"] == second_meta["seed_hash"]
    assert torch.isclose(first.norm(), matched_delta.norm(), atol=1e-5)
    assert first_meta["basis_hash"] == second_meta["basis_hash"]


def test_v19_random_controls_are_included_for_pareto() -> None:
    controls = [
        {"control_type": "target_label_subspace_optimizer", "target_margin": 0.0},
        {"control_type": "random_norm_matched_lowrank_delta_00", "target_margin": 1.0},
        {"control_type": "not_a_gate", "target_margin": 2.0},
    ]
    selected = v19.pareto_controls_for_record(controls)
    assert [item["control_type"] for item in selected] == [
        "target_label_subspace_optimizer",
        "random_norm_matched_lowrank_delta_00",
    ]


def test_v19_assign_shuffled_signatures_cycles_independently_across_directions() -> None:
    stats = fake_train_stats()
    jobs = []
    for source, target in [(v19.PATTERNS[0], v19.PATTERNS[1]), (v19.PATTERNS[0], v19.PATTERNS[2])]:
        for index in range(3):
            subject = fake_subject(f"{source}-{target}-{index}", source, float(index + len(target)))
            jobs.append({"source": source, "target": target, "subject": subject})
    v19.assign_shuffled_signatures(jobs, stats)
    for target in [v19.PATTERNS[1], v19.PATTERNS[2]]:
        group = sorted(
            [job for job in jobs if job["target"] == target],
            key=lambda job: (job["source"], job["subject"]["subject_id"], job["target"]),
        )
        for index, job in enumerate(group):
            expected = v19.v17.normalized_signature(group[(index + 1) % 3]["subject"], stats)
            assert job["shuffled_signature_norm"] == v19.tensor_to_hashable(expected)


def test_v19_evaluate_subjects_serial_parallel_equivalent() -> None:
    subjects = [
        fake_subject(f"{behavior}-{index}", behavior, float(index))
        for behavior in v19.PATTERNS
        for index in range(2)
    ]
    stats = fake_train_stats()
    serial = v19.evaluate_subjects(
        subjects=subjects,
        train_stats=stats,
        random_controls=v19.RANDOM_CONTROLS_PER_RECORD,
        parallel=False,
        record_evaluator=dummy_record_evaluator,
    )
    parallel = v19.evaluate_subjects(
        subjects=subjects,
        train_stats=stats,
        random_controls=v19.RANDOM_CONTROLS_PER_RECORD,
        parallel=True,
        max_workers=2,
        record_evaluator=dummy_record_evaluator,
    )
    assert serial["records"] == parallel["records"]
    assert serial["aggregate"] == parallel["aggregate"]


def test_v19_summary_stdout_redacts_verbose_pool_and_final_details() -> None:
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
    summary = v19.summary_for_stdout(result)
    assert "accepted_subject_ids" not in summary["pool_summaries"]["train"]
    assert "subject_ids" not in summary["pool_summaries"]["final"]
    assert summary["pool_summaries"]["final"] == {
        "accepted_counts_by_behavior": {"a": 1},
        "pool_file_sha256": "finalhash",
        "pool_redacted_payload_sha256": "finalredacted",
    }
