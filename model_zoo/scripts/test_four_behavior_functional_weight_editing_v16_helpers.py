"""Direct helper tests for V16 signature-conceptor output-layer editing."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_functional_weight_editing_v16_signature_conceptor_output_layer as v16  # noqa: E402


def test_v16_uses_fresh_preregistered_pool_scopes_and_seeds() -> None:
    assert v16.DEFAULT_POOL_DIR == (
        REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v16_pools"
    )
    assert v16.POOL_CONFIGS["train"]["base_seed"] == 78300000
    assert v16.POOL_CONFIGS["development"]["base_seed"] == 79300000
    assert v16.POOL_CONFIGS["final"]["base_seed"] == 80300000
    assert v16.SOURCE_POOL_SCOPE == "four_behavior_functional_weight_editing_v16_source_pool"
    assert (
        v16.SOURCE_AUDIT_SCOPE
        == "four_behavior_functional_weight_editing_v16_source_pool_construction"
    )
    assert (
        v16.FINAL_REDACTED_SCOPE
        == "redacted_final_functional_weight_editing_v16_source_pool_audit_surface_only"
    )


def test_v16_conceptor_is_symmetric_finite_and_bounded() -> None:
    activations = torch.tensor([
        [1.0, 0.0, 2.0],
        [0.5, 1.0, 1.5],
        [1.5, -1.0, 0.5],
        [0.0, 0.5, 1.0],
    ])
    conceptor = v16.conceptor_from_activations(
        activations,
        aperture=2.0,
        ridge=1e-4,
    )

    assert conceptor.shape == (3, 3)
    assert torch.isfinite(conceptor).all()
    assert torch.allclose(conceptor, conceptor.T, atol=1e-5)
    eigvals = torch.linalg.eigvalsh(conceptor)
    assert float(eigvals.min()) >= -1e-5
    assert float(eigvals.max()) <= 1.0 + 1e-5


def test_v16_output_layer_compile_matches_explicit_activation_steering() -> None:
    hidden = torch.tensor([
        [1.0, -2.0, 0.5],
        [0.25, 0.75, -1.5],
    ])
    output_weight = torch.tensor([[0.5, -0.25, 1.25]])
    output_bias = torch.tensor([0.1])
    operator = torch.tensor([
        [1.0, 0.2, 0.0],
        [0.0, 0.5, -0.1],
        [0.3, 0.0, 1.2],
    ])
    shift = torch.tensor([0.4, -0.2, 0.1])

    edited_weight, edited_bias = v16.compile_hidden_steering_to_output_layer(
        output_weight=output_weight,
        output_bias=output_bias,
        operator=operator,
        shift=shift,
    )
    explicit_logits = (hidden @ operator.T + shift) @ output_weight.T + output_bias
    compiled_logits = hidden @ edited_weight.T + edited_bias

    assert torch.allclose(compiled_logits, explicit_logits, atol=1e-6)
    assert v16.max_compile_logit_difference(
        hidden=hidden,
        output_weight=output_weight,
        output_bias=output_bias,
        operator=operator,
        shift=shift,
    ) <= 1e-6


def test_v16_compiled_flat_weights_change_only_output_layer() -> None:
    weights = torch.arange(345, dtype=torch.float32) / 100.0
    operator = torch.eye(8, dtype=torch.float32)
    operator[0, 1] = 0.25
    shift = torch.arange(8, dtype=torch.float32) / 10.0

    edited = v16.compile_hidden_steering_to_flat_weights(
        source_weights=weights,
        operator=operator,
        shift=shift,
    )

    assert torch.allclose(edited[:336], weights[:336])
    assert not torch.allclose(edited[336:], weights[336:])
    hidden = torch.randn(5, 8)
    explicit = (hidden @ operator.T + shift) @ weights[336:344].reshape(1, 8).T + weights[344]
    compiled = hidden @ edited[336:344].reshape(1, 8).T + edited[344]
    assert torch.allclose(compiled, explicit, atol=1e-6)


def test_v16_signature_weighted_conceptor_uses_only_target_behavior() -> None:
    stats = {
        "train_by_behavior": {
            "target": [
                {
                    "subject_id": "target:0",
                    "signature_norm": torch.tensor([0.0, 0.0]),
                    "activation_mean": torch.tensor([1.0, 0.0]),
                    "conceptors_by_aperture": {
                        1.0: torch.eye(2),
                    },
                },
                {
                    "subject_id": "target:1",
                    "signature_norm": torch.tensor([1.0, 1.0]),
                    "activation_mean": torch.tensor([0.0, 1.0]),
                    "conceptors_by_aperture": {
                        1.0: torch.eye(2) * 0.5,
                    },
                },
            ],
            "other": [
                {
                    "subject_id": "other:0",
                    "signature_norm": torch.tensor([0.0, 0.0]),
                    "activation_mean": torch.tensor([100.0, 100.0]),
                    "conceptors_by_aperture": {
                        1.0: torch.eye(2) * 100.0,
                    },
                },
            ],
        },
    }

    result = v16.signature_weighted_target_operator(
        train_stats=stats,
        target_behavior="target",
        target_signature_norm=torch.tensor([0.0, 0.0]),
        aperture=1.0,
    )

    assert result["target_behavior"] == "target"
    assert [set(item) for item in result["weighted_subjects"]] == [
        {"subject_id_hash", "weight", "signature_distance"},
        {"subject_id_hash", "weight", "signature_distance"},
    ]
    assert result["target_mean"][0] > result["target_mean"][1]
    assert result["target_conceptor"][0, 0] < 1.01


def test_v16_record_sorting_is_stable() -> None:
    records = [
        {"source_behavior": "b", "subject_id": "2", "target_behavior": "a"},
        {"source_behavior": "a", "subject_id": "2", "target_behavior": "b"},
        {"source_behavior": "a", "subject_id": "1", "target_behavior": "c"},
    ]

    assert v16.sort_records_for_artifact(records) == [
        {"source_behavior": "a", "subject_id": "1", "target_behavior": "c"},
        {"source_behavior": "a", "subject_id": "2", "target_behavior": "b"},
        {"source_behavior": "b", "subject_id": "2", "target_behavior": "a"},
    ]


def test_v16_train_statistics_accept_pool_pattern_key(monkeypatch) -> None:
    probe_examples = [{"sequence": [1, 2, 3, 4, 5]}, {"sequence": [5, 4, 3, 2, 1]}]
    subjects = [
        {
            "pattern": pattern,
            "subject_id": f"{pattern}:0",
            "signature": [float(index), float(index + 1), float(index + 2)],
            "weights": [0.0] * 345,
        }
        for index, pattern in enumerate(v16.PATTERNS)
    ]

    def fake_hidden_activations_flat_batch(weights, probe_inputs):
        hidden = torch.ones((weights.shape[0], probe_inputs.shape[0], 8), dtype=torch.float32)
        return [hidden]

    monkeypatch.setattr(v16.v15, "hidden_activations_flat_batch", fake_hidden_activations_flat_batch)

    stats = v16.fit_v16_train_statistics(subjects, probe_examples=probe_examples)

    assert sorted(stats["train_by_behavior"]) == sorted(v16.PATTERNS)
    assert [record["subject_id"] for record in stats["train_subjects"]] == [
        "has_majority:0",
        "mountain_pattern:0",
        "sorted_ascending:0",
        "sorted_descending:0",
    ]


def test_v16_control_contract_matches_preregistration() -> None:
    assert v16.THRESHOLDS["expected_controls_per_record"] == 26
    assert v16.THRESHOLDS["random_controls_per_record"] == 16
    assert v16.THRESHOLDS["expected_non_random_controls_per_record"] == 10
    assert v16.THRESHOLDS["min_aggregate_individual_pass_rate"] == 0.85
    assert v16.THRESHOLDS["min_aggregate_pareto_undominated_rate"] == 0.85
    assert "v13_no_signature_support_optimizer" not in v16.V16_GATING_CONTROL_TYPES
    assert "random_norm_matched_output_layer_delta:00" in v16.V16_GATING_CONTROL_TYPES
    assert len(v16.REQUIRED_NON_RANDOM_CONTROL_TYPES) == 10


def test_v16_final_summary_allowlists_are_fail_closed() -> None:
    allowed = {
        "accepted_counts_by_behavior": {pattern: 24 for pattern in v16.PATTERNS},
        "pool_file_sha256": "finalhash",
        "pool_redacted_payload_sha256": "redactedhash",
    }
    assert v16.forbidden_combined_final_summary_keys(allowed) == []
    leaked = {**allowed, "record_count": 99, "accepted_subject_ids": ["leaked"]}
    assert v16.forbidden_combined_final_summary_keys(leaked) == [
        "accepted_subject_ids",
        "record_count",
    ]

    redacted = {
        "behavior_suite_hashes": {},
        "candidate_pool_summary_hash": "candidatehash",
        "claim_scope": v16.FINAL_REDACTED_SCOPE,
        "config_hash": "confighash",
        "pool": "final",
        "pool_file_sha256": "finalhash",
        "pool_redacted_payload_sha256": "redactedhash",
        "summary": {},
        "summary_payload_sha256": "summaryhash",
    }
    assert v16.forbidden_final_redacted_keys(redacted) == []
    assert v16.forbidden_final_redacted_keys({**redacted, "subject_ids": []}) == [
        "subject_ids",
        "top_level.subject_ids",
    ]
    nested = {
        **redacted,
        "summary": {
            "subject_ids": ["leaked"],
            "nested": {"records": [], "weights_hash": "hash"},
        },
    }
    assert v16.forbidden_final_redacted_keys(nested) == [
        "summary.nested.records",
        "summary.nested.weights_hash",
        "summary.subject_ids",
    ]


def test_v16_seed_preflight_uses_v16_pool_configs() -> None:
    preflight = v16.build_v16_seed_preflight()
    assert preflight["passed"] is True
    ranges = {
        (item["pool"], item["pattern"]): item
        for item in preflight["seed_ranges"]
    }

    assert ranges[("train", v16.PATTERNS[0])]["start_seed"] == 78300000
    assert ranges[("development", v16.PATTERNS[0])]["start_seed"] == 79300000
    assert ranges[("final", v16.PATTERNS[0])]["start_seed"] == 80300000
    assert ranges[("train", v16.PATTERNS[1])]["start_seed"] == 78400000


def test_v16_rejects_prior_and_own_final_raw_before_authorization() -> None:
    prior = (
        REPO_ROOT
        / "runs"
        / "four_behavior_functional_weight_editing_v15_pools"
        / "final_subjects.json"
    )
    try:
        v16.assert_no_forbidden_final_raw_paths([prior], allow_v16_final=False)
    except ValueError as error:
        assert "prior sealed final raw path" in str(error)
    else:
        raise AssertionError("V16 accepted a prior final raw path")

    try:
        v16.assert_no_forbidden_final_raw_paths([v16.V16_FINAL_RAW], allow_v16_final=False)
    except ValueError as error:
        assert "V16 final raw path is forbidden" in str(error)
    else:
        raise AssertionError("V16 accepted its final raw before authorization")


def test_v16_contract_rejects_extra_combined_final_summary_fields(monkeypatch) -> None:
    monkeypatch.setattr(v16.v15.v1, "sha256_file", lambda path: {
        "development_subjects.json": "devhash",
        "train_subjects.json": "trainhash",
    }.get(path.name, "otherhash"))
    counts_64 = {pattern: 64 for pattern in v16.PATTERNS}
    counts_24 = {pattern: 24 for pattern in v16.PATTERNS}
    combined_audit = {
        "claim_scope": v16.SOURCE_AUDIT_SCOPE,
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
                "pool_redacted_payload_sha256": "redactedhash",
                "record_count": 96,
            },
            "train": {
                "accepted_counts_by_behavior": counts_64,
                "pool_file_sha256": "trainhash",
            },
        },
    }
    final_redacted = {
        "claim_scope": v16.FINAL_REDACTED_SCOPE,
        "pool_file_sha256": "finalhash",
    }

    failures = v16.validate_source_pool_contract(
        train_path=Path("train_subjects.json"),
        eval_path=Path("development_subjects.json"),
        train_payload={"claim_scope": v16.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v16.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )

    joined = "\n".join(failures)
    assert "combined_audit.pool_summaries.final exposes forbidden keys" in joined
    assert "record_count" in joined


def test_v16_train_statistics_include_signature_norm_and_conceptors(monkeypatch) -> None:
    probe_examples = [
        {"input": [0, 1, 2, 3, 4]},
        {"input": [4, 3, 2, 1, 0]},
    ]
    subjects = []
    for behavior_index, behavior in enumerate(v16.PATTERNS):
        for index in range(2):
            base = float(behavior_index + index) / 100.0
            subjects.append({
                "behavior": behavior,
                "subject_id": f"{behavior}:{index:02d}",
                "weights": [base] * 345,
                "signature": [base] * 560,
            })
    monkeypatch.setattr(v16, "APERTURE_GRID", [1.0])
    stats = v16.fit_v16_train_statistics(subjects, probe_examples=probe_examples)

    assert stats["probe_examples_hash"]
    assert stats["train_statistics_hash"]
    assert set(stats["train_by_behavior"]) == set(v16.PATTERNS)
    first = stats["train_by_behavior"][v16.PATTERNS[0]][0]
    assert first["signature_norm"].shape == (560,)
    assert first["activation_mean"].shape == (8,)
    assert 1.0 in first["conceptors_by_aperture"]
    assert first["conceptors_by_aperture"][1.0].shape == (8, 8)


def test_v16_serial_and_parallel_evaluation_have_same_order_and_aggregate(monkeypatch) -> None:
    monkeypatch.setattr(
        v16,
        "multiprocessing_contract",
        lambda: {
            "max_workers": 2,
            "stable_record_sort_key": ["source_behavior", "subject_id", "target_behavior"],
            "start_method": "spawn",
            "torch_threads_per_worker": 1,
            "worker_writes_result_files": False,
        },
    )
    subjects = [
        {
            "behavior": v16.PATTERNS[1],
            "subject_id": "subject:b",
            "signature": [0.0] * 560,
            "weights": [0.0] * 345,
        },
        {
            "behavior": v16.PATTERNS[0],
            "subject_id": "subject:a",
            "signature": [0.0] * 560,
            "weights": [0.0] * 345,
        },
    ]
    kwargs = {
        "subjects": subjects,
        "train_stats": {},
        "classifier": torch.nn.Identity(),
        "random_controls": 16,
        "record_evaluator": v16.lightweight_record_evaluator_for_tests,
    }

    serial = v16.evaluate_subjects(**kwargs, parallel=False)
    parallel = v16.evaluate_subjects(**kwargs, parallel=True)

    assert [record["subject_id"] for record in serial["records"]] == [
        record["subject_id"] for record in parallel["records"]
    ]
    assert [
        (record["source_behavior"], record["subject_id"], record["target_behavior"])
        for record in serial["records"]
    ] == [
        (record["source_behavior"], record["subject_id"], record["target_behavior"])
        for record in parallel["records"]
    ]
    assert serial["aggregate"] == parallel["aggregate"]
