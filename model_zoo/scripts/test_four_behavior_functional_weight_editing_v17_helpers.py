"""Direct helper tests for V17 layerwise rank-1/TSV editing."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_functional_weight_editing_v17_layerwise_rank1_tsv as v17  # noqa: E402


def lightweight_v17_record_evaluator(job, *, train_stats, random_controls):
    del train_stats
    controls = [
        {"control_type": "no_edit"},
        *[{"control_type": control_type} for control_type in sorted(v17.PROOF_CRITICAL_CONTROL_TYPES)],
    ]
    return {
        "controls": controls,
        "individual_all_gates_passed": True,
        "matched": {
            "compatible_source_output_mse": 0.0,
            "conflict_target_accuracy": 1.0,
            "conflict_target_accuracy_improvement": 1.0,
            "predicted_behavior": job["target"],
            "shuffled_signature_norm": job.get("shuffled_signature_norm"),
            "target_margin": 1.0,
        },
        "random_control_count": int(random_controls),
        "source_behavior": job["source"],
        "subject_id": str(job["subject"]["subject_id"]),
        "summary": {
            "matched_minus_best_control_target_margin": 1.0,
            "matched_minus_output_layer_no_signature_target_margin": 1.0,
            "matched_minus_shuffled_signature_target_margin": 1.0,
            "matched_minus_target_label_target_margin": 1.0,
            "output_layer_no_signature_minus_matched_compatible_source_output_mse": 3.0,
            "pareto_undominated": True,
            "shuffled_signature_minus_matched_compatible_source_output_mse": 3.0,
            "target_label_minus_matched_compatible_source_output_mse": 3.0,
            "target_prediction_pass": True,
        },
        "target_behavior": job["target"],
    }


def test_v17_fresh_scopes_paths_and_pool_seeds() -> None:
    assert v17.DEFAULT_POOL_DIR == (
        REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v17_pools"
    )
    assert v17.DEFAULT_OUTPUT_DIR == (
        REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v17_layerwise_rank1_tsv"
    )
    assert v17.POOL_CONFIGS["train"]["base_seed"] == 81300000
    assert v17.POOL_CONFIGS["development"]["base_seed"] == 82300000
    assert v17.POOL_CONFIGS["final"]["base_seed"] == 83300000
    assert v17.EDITOR_METHOD == "signature_conditioned_layerwise_rank1_tsv_v17"
    assert v17.RANDOM_CONTROLS_PER_RECORD == 16


def test_v17_layer_component_specs_cover_flat_subject_once() -> None:
    specs = v17.LAYER_COMPONENT_SPECS
    assert specs[0]["name"] == "weight_0"
    assert specs[0]["start"] == 0
    assert specs[0]["end"] == 40
    assert specs[-2]["name"] == "weight_5"
    assert specs[-2]["start"] == 336
    assert specs[-2]["end"] == 344
    assert specs[-1]["name"] == "bias_5"
    assert specs[-1]["start"] == 344
    assert specs[-1]["end"] == 345
    covered = []
    for spec in specs:
        covered.extend(range(int(spec["start"]), int(spec["end"])))
    assert covered == list(range(345))
    assert len(v17.active_component_specs("hidden_only")) == 10
    assert len(v17.active_component_specs("all_layers")) == 12


def test_v17_svd_basis_sign_canonicalization_is_deterministic() -> None:
    basis = torch.tensor([
        [-0.0, -2.0, 1.0],
        [0.0, 0.0, -3.0],
        [0.0, 0.0, 0.0],
    ])
    canonical = v17.sign_canonicalize_basis_rows(basis)
    assert canonical.tolist() == [
        [0.0, 2.0, -1.0],
        [-0.0, -0.0, 3.0],
        [0.0, 0.0, 0.0],
    ]


def test_v17_signature_topk_uses_normalized_mse_and_subject_tiebreak() -> None:
    train_stats = {
        "sig_mean": torch.tensor([1.0, 1.0]),
        "sig_std": torch.tensor([2.0, 1.0]),
    }
    target_records = [
        {"subject_id": "b", "signature": [3.0, 2.0]},
        {"subject_id": "a", "signature": [3.0, 2.0]},
        {"subject_id": "c", "signature": [5.0, 1.0]},
    ]
    result = v17.signature_topk_weights(
        target_records,
        source_signature_norm=torch.tensor([1.0, 1.0]),
        train_stats=train_stats,
        top_k=2,
    )
    selected_subjects = [subject_id for _distance, subject_id, _record in result["selected"]]
    assert selected_subjects == ["a", "b"]
    assert torch.isclose(result["weights"].sum(), torch.tensor(1.0))
    assert result["metadata"][0]["rank_order"] == 0
    assert set(result["metadata"][0]) == {
        "rank_order",
        "signature_distance",
        "subject_id_hash",
        "weight",
    }


def test_v17_pareto_orientation_uses_higher_target_margin_lower_mse() -> None:
    matched = {"target_margin": 1.0, "compatible_source_output_mse": 2.0}
    assert v17.pareto_dominates(
        {"target_margin": 1.1, "compatible_source_output_mse": 2.0},
        matched,
    )
    assert v17.pareto_dominates(
        {"target_margin": 1.0, "compatible_source_output_mse": 1.9},
        matched,
    )
    assert not v17.pareto_dominates(
        {"target_margin": 1.1, "compatible_source_output_mse": 2.1},
        matched,
    )
    assert not v17.pareto_dominates(
        {"target_margin": 1.0, "compatible_source_output_mse": 2.0},
        matched,
    )


def test_v17_random_layerwise_low_rank_delta_is_deterministic_and_norm_matched() -> None:
    source_weights = torch.zeros(345)
    kwargs = {
        "source_weights": source_weights,
        "matched_delta_norm": torch.tensor(3.0),
        "subject_id": "subject:1",
        "source": "sorted_ascending",
        "target": "has_majority",
        "index": 2,
        "rank": 2,
        "layer_mask": "hidden_only",
    }
    delta_a, meta_a = v17.random_layerwise_low_rank_delta(**kwargs)
    delta_b, meta_b = v17.random_layerwise_low_rank_delta(**kwargs)
    assert torch.allclose(delta_a, delta_b)
    assert meta_a == meta_b
    assert torch.isclose(delta_a.norm(), torch.tensor(3.0), atol=1e-5)
    assert torch.allclose(delta_a[336:], torch.zeros(9))


def test_v17_random_layerwise_low_rank_delta_zero_norm_case() -> None:
    delta, meta = v17.random_layerwise_low_rank_delta(
        source_weights=torch.zeros(345),
        matched_delta_norm=torch.tensor(0.0),
        subject_id="subject:1",
        source="sorted_ascending",
        target="has_majority",
        index=0,
        rank=4,
        layer_mask="all_layers",
    )
    assert torch.allclose(delta, torch.zeros(345))
    assert meta["zero_norm_matched_delta"] is True


def test_v17_summary_stdout_redacts_verbose_pool_details() -> None:
    result = {
        "passed": True,
        "pool_summaries": {
            "train": {
                "accepted_counts_by_behavior": {"a": 64},
                "accepted_subject_ids": ["leak"],
                "pool_file_sha256": "trainhash",
                "pool_redacted_payload_sha256": "redacted",
                "record_count": 263,
            },
            "final": {
                "accepted_counts_by_behavior": {"a": 24},
                "accepted_subject_ids": ["final-leak"],
                "pool_file_sha256": "finalhash",
            },
        },
    }

    summary = v17.summary_for_stdout(result)

    assert summary["pool_summaries"]["train"] == {
        "accepted_counts_by_behavior": {"a": 64},
        "pool_file_sha256": "trainhash",
        "pool_redacted_payload_sha256": "redacted",
        "record_count": 263,
    }
    assert summary["pool_summaries"]["final"] == {
        "accepted_counts_by_behavior": {"a": 24},
        "pool_file_sha256": "finalhash",
    }


def test_v17_train_statistics_builds_layerwise_bases_from_sorted_pairs(monkeypatch) -> None:
    subjects = []
    for behavior_index, pattern in enumerate(v17.PATTERNS):
        for subject_index in range(2):
            value = float(behavior_index * 10 + subject_index)
            subjects.append({
                "pattern": pattern,
                "signature": [value, value + 1.0],
                "subject_id": f"{pattern}:{1 - subject_index}",
                "weights": [value] * 345,
            })

    calls = []

    def fake_target_delta_for_record(
        *,
        source_weights,
        target_record,
        source,
        target,
        subject_id,
        alignment_mode,
    ):
        calls.append((source, target, subject_id, str(target_record["subject_id"]), alignment_mode))
        target_weights = torch.tensor(target_record["weights"], dtype=torch.float32)
        return target_weights - source_weights

    monkeypatch.setattr(
        v17.v16.v15.v14,
        "target_delta_for_record",
        fake_target_delta_for_record,
    )

    stats = v17.fit_v17_train_statistics(
        subjects,
        probe_examples=[{"sequence": [1, 2, 3, 4, 5]}],
    )
    direction = v17.direction_key(v17.PATTERNS[0], v17.PATTERNS[1])

    assert stats["layerwise_bases"][direction]["pair_count"] == 4
    assert stats["layerwise_bases"][direction]["components"]["weight_0"]["component_count"] == 4
    assert calls[0] == (
        v17.PATTERNS[0],
        v17.PATTERNS[1],
        f"{v17.PATTERNS[0]}:0",
        f"{v17.PATTERNS[1]}:0",
        "hungarian",
    )
    delta = torch.ones(345)
    projected = v17.project_full_delta_layerwise(
        delta,
        direction_bases=stats["layerwise_bases"][direction],
        rank=1,
        layer_mask="hidden_only",
    )
    assert projected.shape == (345,)
    assert torch.allclose(projected[336:], torch.zeros(9))


def test_v17_train_statistics_hash_binds_train_weight_values(monkeypatch) -> None:
    subjects = []
    for behavior_index, pattern in enumerate(v17.PATTERNS):
        value = float(behavior_index + 1)
        subjects.append({
            "pattern": pattern,
            "signature": [value, value + 1.0],
            "subject_id": f"{pattern}:0",
            "weights": [value] * 345,
        })

    def fake_target_delta_for_record(
        *,
        source_weights,
        target_record,
        source,
        target,
        subject_id,
        alignment_mode,
    ):
        del source, target, subject_id, alignment_mode
        target_weights = torch.tensor(target_record["weights"], dtype=torch.float32)
        return target_weights - source_weights

    monkeypatch.setattr(
        v17.v16.v15.v14,
        "target_delta_for_record",
        fake_target_delta_for_record,
    )

    base = v17.fit_v17_train_statistics(
        subjects,
        probe_examples=[{"sequence": [1, 2, 3, 4, 5]}],
    )
    mutated_subjects = [dict(subject) for subject in subjects]
    mutated_weights = list(mutated_subjects[0]["weights"])
    mutated_weights[0] += 1.0
    mutated_subjects[0]["weights"] = mutated_weights
    mutated = v17.fit_v17_train_statistics(
        mutated_subjects,
        probe_examples=[{"sequence": [1, 2, 3, 4, 5]}],
    )

    assert base["train_statistics_hash"] != mutated["train_statistics_hash"]


def test_v17_v14_baseline_hash_binds_recomputed_subspace_values(monkeypatch) -> None:
    subjects = []
    for behavior_index, pattern in enumerate(v17.PATTERNS):
        value = float(behavior_index + 1)
        subjects.append({
            "pattern": pattern,
            "signature": [value, value + 1.0],
            "subject_id": f"{pattern}:0",
            "weights": [value] * 345,
        })

    def fake_target_delta_for_record(
        *,
        source_weights,
        target_record,
        source,
        target,
        subject_id,
        alignment_mode,
    ):
        del target_record, source, target, subject_id, alignment_mode
        return torch.ones_like(source_weights)

    def fake_v14_stats(records):
        total = sum(float(record["weights"][0]) for record in records)
        return {
            "edit_subspaces": {
                "a_to_b": {
                    "basis": torch.tensor([[total, 0.0]]),
                    "explained_variance": 1.0,
                    "mean_delta": torch.tensor([total, 1.0]),
                    "mean_delta_norm": float(total),
                    "rank": 1,
                    "singular_values": torch.tensor([total]),
                }
            },
            "sig_mean": torch.tensor([total]),
            "sig_std": torch.tensor([1.0]),
        }

    monkeypatch.setattr(v17.v16.v15.v14, "target_delta_for_record", fake_target_delta_for_record)
    monkeypatch.setattr(v17.v16.v15.v14, "fit_v14_train_statistics", fake_v14_stats)
    monkeypatch.setattr(
        v17.v16,
        "fit_v16_train_statistics",
        lambda records, *, probe_examples: {"train_statistics_hash": str(len(records))},
    )

    base = v17.fit_v17_train_statistics(
        subjects,
        probe_examples=[{"sequence": [1, 2, 3, 4, 5]}],
        include_baseline_stats=True,
    )
    mutated_subjects = [dict(subject) for subject in subjects]
    mutated_weights = list(mutated_subjects[0]["weights"])
    mutated_weights[0] += 1.0
    mutated_subjects[0]["weights"] = mutated_weights
    mutated = v17.fit_v17_train_statistics(
        mutated_subjects,
        probe_examples=[{"sequence": [1, 2, 3, 4, 5]}],
        include_baseline_stats=True,
    )

    assert (
        base["v14_baseline_train_statistics_hash"]
        != mutated["v14_baseline_train_statistics_hash"]
    )


def test_v17_activation_rank1_delta_changes_hidden_layers_only() -> None:
    source_weights = torch.zeros(345)
    target_weights = torch.ones(345) * 0.01
    delta = v17.activation_rank1_delta(
        source_weights=source_weights,
        aligned_target_weights=[target_weights],
        signature_weights=torch.tensor([1.0]),
        probe_examples=[{"sequence": [1, 2, 3, 4, 5]}, {"sequence": [5, 4, 3, 2, 1]}],
    )

    assert delta.shape == (345,)
    assert torch.isfinite(delta).all()
    assert not torch.allclose(delta[:336], torch.zeros(336))
    assert torch.allclose(delta[336:], torch.zeros(9))


def test_v17_select_layerwise_edit_uses_support_objective_tiebreaks(monkeypatch) -> None:
    target = v17.PATTERNS[1]
    source = v17.PATTERNS[0]
    source_weights = torch.zeros(345)
    train_stats = {
        "layerwise_bases": {
            v17.direction_key(source, target): {
                "components": {
                    spec["name"]: {
                        "basis": torch.eye(int(spec["end"]) - int(spec["start"]))[:1],
                        "mean_delta": torch.zeros(int(spec["end"]) - int(spec["start"])),
                        "rank": 1,
                    }
                    for spec in v17.LAYER_COMPONENT_SPECS
                }
            }
        },
        "probe_examples": [{"sequence": [1, 2, 3, 4, 5]}],
        "sig_mean": torch.zeros(2),
        "sig_std": torch.ones(2),
        "train_by_behavior": {
            pattern: [
                {
                    "pattern": pattern,
                    "signature": [float(index), 0.0],
                    "subject_id": f"{pattern}:0",
                    "weights": [float(index)] * 345,
                }
            ]
            for index, pattern in enumerate(v17.PATTERNS)
        },
    }

    def fake_target_delta_for_record(
        *,
        source_weights,
        target_record,
        source,
        target,
        subject_id,
        alignment_mode,
    ):
        del target_record, source, target, subject_id, alignment_mode
        delta = torch.zeros_like(source_weights)
        delta[0] = 2.0
        return delta

    def fake_support_objective_for_weights(*, weights, source_weights, source, target):
        del source_weights, source, target
        # Prefer the smallest nonzero first-coordinate edit, then normal tie-breaks.
        objective = abs(float(weights[0].item()) - 0.5)
        return {
            "compatible_mse": 0.0,
            "conflict_bce": 0.0,
            "objective": objective,
            "source_l2": 0.0,
            "target_bce": 0.0,
        }

    monkeypatch.setattr(v17.v16.v15.v14, "target_delta_for_record", fake_target_delta_for_record)
    monkeypatch.setattr(v17, "support_objective_for_weights", fake_support_objective_for_weights)
    monkeypatch.setattr(
        v17,
        "activation_rank1_delta",
        lambda **_kwargs: torch.zeros(345),
    )

    weights, metadata = v17.select_layerwise_rank1_tsv_edit(
        source_weights=source_weights,
        subject_id="source:0",
        source=source,
        target=target,
        source_signature_norm=torch.zeros(2),
        train_stats=train_stats,
        task_scale_grid=[0.0, 0.25, 0.5],
        activation_scale_grid=[0.0],
        rank_grid=[1],
        layer_masks=["hidden_only"],
    )

    assert torch.isclose(weights[0], torch.tensor(0.5))
    assert metadata["selected_task_scale"] == 0.25
    assert metadata["selected_activation_scale"] == 0.0
    assert metadata["selected_rank"] == 1
    assert metadata["selected_layer_mask"] == "hidden_only"


def test_v17_select_layerwise_edit_has_final_lexical_tiebreak(monkeypatch) -> None:
    target = v17.PATTERNS[1]
    source = v17.PATTERNS[0]
    source_weights = torch.zeros(345)
    train_stats = {
        "layerwise_bases": {
            v17.direction_key(source, target): {
                "components": {
                    spec["name"]: {
                        "basis": torch.eye(int(spec["end"]) - int(spec["start"]))[:1],
                        "mean_delta": torch.zeros(int(spec["end"]) - int(spec["start"])),
                        "rank": 1,
                    }
                    for spec in v17.LAYER_COMPONENT_SPECS
                }
            }
        },
        "probe_examples": [{"sequence": [1, 2, 3, 4, 5]}],
        "sig_mean": torch.zeros(2),
        "sig_std": torch.ones(2),
        "train_by_behavior": {
            pattern: [
                {
                    "pattern": pattern,
                    "signature": [float(index), 0.0],
                    "subject_id": f"{pattern}:0",
                    "weights": [float(index)] * 345,
                }
            ]
            for index, pattern in enumerate(v17.PATTERNS)
        },
    }

    monkeypatch.setattr(
        v17.v16.v15.v14,
        "target_delta_for_record",
        lambda **kwargs: torch.ones_like(kwargs["source_weights"]),
    )
    monkeypatch.setattr(
        v17,
        "support_objective_for_weights",
        lambda **_kwargs: {
            "compatible_mse": 0.0,
            "conflict_bce": 0.0,
            "objective": 0.0,
            "source_l2": 0.0,
            "target_bce": 0.0,
        },
    )
    monkeypatch.setattr(v17, "activation_rank1_delta", lambda **_kwargs: torch.zeros(345))

    _weights, metadata = v17.select_layerwise_rank1_tsv_edit(
        source_weights=source_weights,
        subject_id="source:0",
        source=source,
        target=target,
        source_signature_norm=torch.zeros(2),
        train_stats=train_stats,
        task_scale_grid=[0.25, 0.25],
        activation_scale_grid=[0.0],
        rank_grid=[1],
        layer_masks=["hidden_only"],
    )

    assert metadata["selected_candidate_lexical_key"].endswith("candidate=000000")


def test_v17_evaluate_subjects_serial_parallel_equivalent() -> None:
    subjects = [
        {
            "pattern": v17.PATTERNS[0],
            "signature": [0.0, 1.0],
            "subject_id": "a",
            "weights": [0.0] * 345,
        },
        {
            "pattern": v17.PATTERNS[1],
            "signature": [1.0, 0.0],
            "subject_id": "b",
            "weights": [0.0] * 345,
        },
    ]
    train_stats = {
        "sig_mean": torch.zeros(2),
        "sig_std": torch.ones(2),
    }

    serial = v17.evaluate_subjects(
        subjects=subjects,
        train_stats=train_stats,
        random_controls=16,
        parallel=False,
        record_evaluator=lightweight_v17_record_evaluator,
    )
    parallel = v17.evaluate_subjects(
        subjects=subjects,
        train_stats=train_stats,
        random_controls=16,
        parallel=True,
        max_workers=2,
        record_evaluator=lightweight_v17_record_evaluator,
    )

    assert serial["records"] == parallel["records"]
    assert serial["aggregate"] == parallel["aggregate"]
    assert serial["by_direction"] == parallel["by_direction"]


def test_v17_evaluate_subjects_assigns_cyclic_shuffled_signatures() -> None:
    subjects = [
        {
            "pattern": v17.PATTERNS[0],
            "signature": [0.0, 1.0],
            "subject_id": "a",
            "weights": [0.0] * 345,
        },
        {
            "pattern": v17.PATTERNS[1],
            "signature": [1.0, 0.0],
            "subject_id": "b",
            "weights": [0.0] * 345,
        },
        {
            "pattern": v17.PATTERNS[0],
            "signature": [0.5, 0.5],
            "subject_id": "c",
            "weights": [0.0] * 345,
        },
        {
            "pattern": v17.PATTERNS[1],
            "signature": [0.25, 0.75],
            "subject_id": "d",
            "weights": [0.0] * 345,
        },
    ]
    train_stats = {
        "sig_mean": torch.zeros(2),
        "sig_std": torch.ones(2),
    }

    result = v17.evaluate_subjects(
        subjects=subjects,
        train_stats=train_stats,
        random_controls=16,
        parallel=False,
        record_evaluator=lightweight_v17_record_evaluator,
    )
    by_key = {
        (
            record["source_behavior"],
            record["subject_id"],
            record["target_behavior"],
        ): record["matched"]["shuffled_signature_norm"]
        for record in result["records"]
    }

    assert by_key[(v17.PATTERNS[0], "a", v17.PATTERNS[1])] == [0.5, 0.5]
    assert by_key[(v17.PATTERNS[0], "c", v17.PATTERNS[1])] == [0.0, 1.0]
    assert by_key[(v17.PATTERNS[0], "a", v17.PATTERNS[2])] == [0.5, 0.5]
    assert by_key[(v17.PATTERNS[1], "b", v17.PATTERNS[0])] == [0.25, 0.75]
    assert by_key[(v17.PATTERNS[1], "d", v17.PATTERNS[0])] == [1.0, 0.0]
