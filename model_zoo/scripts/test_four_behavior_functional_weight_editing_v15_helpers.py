"""Direct helper tests for V15 signature-conditioned hypernetwork editing."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_functional_weight_editing_v15_signature_conditioned_hypernetwork as v15  # noqa: E402


def test_v15_uses_fresh_preregistered_pool_scopes_and_seeds() -> None:
    assert v15.DEFAULT_POOL_DIR == (
        REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v15_pools"
    )
    assert v15.POOL_CONFIGS["train"]["base_seed"] == 75300000
    assert v15.POOL_CONFIGS["development"]["base_seed"] == 76300000
    assert v15.POOL_CONFIGS["final"]["base_seed"] == 77300000
    assert v15.SOURCE_POOL_SCOPE == "four_behavior_functional_weight_editing_v15_source_pool"
    assert (
        v15.SOURCE_AUDIT_SCOPE
        == "four_behavior_functional_weight_editing_v15_source_pool_construction"
    )
    assert (
        v15.FINAL_REDACTED_SCOPE
        == "redacted_final_functional_weight_editing_v15_source_pool_audit_surface_only"
    )


def test_v15_combined_final_summary_allowlist_is_strict() -> None:
    allowed = {
        "accepted_counts_by_behavior": {pattern: 24 for pattern in v15.PATTERNS},
        "pool_file_sha256": "finalhash",
        "pool_redacted_payload_sha256": "redactedhash",
    }
    assert v15.forbidden_combined_final_summary_keys(allowed) == []
    leaked = {**allowed, "record_count": 99, "accepted_subject_ids": ["leaked"]}
    assert v15.forbidden_combined_final_summary_keys(leaked) == [
        "accepted_subject_ids",
        "record_count",
    ]


def test_v15_rejects_prior_and_own_final_raw_before_authorization() -> None:
    prior = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v14_pools" / "final_subjects.json"
    try:
        v15.assert_no_forbidden_final_raw_paths([prior], allow_v15_final=False)
    except ValueError as error:
        assert "prior sealed final raw path" in str(error)
    else:
        raise AssertionError("V15 accepted V14 final raw as an input")

    try:
        v15.assert_no_forbidden_final_raw_paths([v15.V15_FINAL_RAW], allow_v15_final=False)
    except ValueError as error:
        assert "V15 final raw path is forbidden" in str(error)
    else:
        raise AssertionError("V15 accepted its final raw before authorization")


def test_v15_training_pair_table_is_full_sorted_cross_product() -> None:
    subjects = []
    for behavior in v15.PATTERNS:
        for index in range(64):
            subjects.append({
                "behavior": behavior,
                "subject_id": f"{behavior}:{index:02d}",
                "weights": [0.0] * 345,
                "signature": [float(index)] * 560,
            })
    table = v15.build_training_pair_table(subjects)

    assert len(table) == 49152
    assert table[0] == {
        "source_behavior": v15.PATTERNS[0],
        "source_subject_id": f"{v15.PATTERNS[0]}:00",
        "target_behavior": v15.PATTERNS[1],
        "target_subject_id": f"{v15.PATTERNS[1]}:00",
    }
    assert table[-1] == {
        "source_behavior": v15.PATTERNS[-1],
        "source_subject_id": f"{v15.PATTERNS[-1]}:63",
        "target_behavior": v15.PATTERNS[-2],
        "target_subject_id": f"{v15.PATTERNS[-2]}:63",
    }


def test_v15_editor_input_dimension_and_output_shape() -> None:
    model = v15.SignatureConditionedDeltaHypernetwork(seed=20260615)
    batch = torch.zeros(3, v15.EDITOR_INPUT_DIM, dtype=torch.float32)
    delta, diagnostic_scale = model(batch)

    assert delta.shape == (3, 345)
    assert diagnostic_scale.shape == (3,)
    assert torch.all(diagnostic_scale >= 0.0)
    assert torch.all(diagnostic_scale <= 1.5)


def test_v15_random_signature_is_deterministic() -> None:
    sig_a = v15.deterministic_random_signature(
        key_parts=["subject", "source", "target"],
        dim=8,
    )
    sig_b = v15.deterministic_random_signature(
        key_parts=["subject", "source", "target"],
        dim=8,
    )
    sig_c = v15.deterministic_random_signature(
        key_parts=["subject", "source", "other"],
        dim=8,
    )

    assert torch.allclose(sig_a, sig_b)
    assert not torch.allclose(sig_a, sig_c)
    assert sig_a.shape == (8,)


def test_v15_required_control_contract_matches_preregistration() -> None:
    assert len(v15.REQUIRED_NON_RANDOM_CONTROL_TYPES) == 14
    assert v15.THRESHOLDS["expected_controls_per_record"] == 30
    assert v15.THRESHOLDS["random_controls_per_record"] == 16
    assert "v13_no_signature_support_optimizer" in v15.REQUIRED_NON_RANDOM_CONTROL_TYPES
    assert "target_label_only_hypernetwork" in v15.REQUIRED_NON_RANDOM_CONTROL_TYPES
    assert "shuffled_signature_hypernetwork" in v15.REQUIRED_NON_RANDOM_CONTROL_TYPES
    assert "signature_only_hypernetwork" in v15.REQUIRED_NON_RANDOM_CONTROL_TYPES


def test_v15_editor_features_have_preregistered_layout() -> None:
    source_weights_norm = torch.arange(345, dtype=torch.float32)
    source_signature_norm = torch.ones(560, dtype=torch.float32)
    target_signature_norm = torch.ones(560, dtype=torch.float32) * 2
    features = v15.build_editor_features(
        source_weights_norm=source_weights_norm,
        source_signature_norm=source_signature_norm,
        target_signature_norm=target_signature_norm,
        source_behavior=v15.PATTERNS[0],
        target_behavior=v15.PATTERNS[1],
    )

    assert features.shape == (v15.EDITOR_INPUT_DIM,)
    assert torch.allclose(features[:345], source_weights_norm)
    assert torch.allclose(features[345:905], source_signature_norm)
    assert torch.allclose(features[905:1465], target_signature_norm)
    assert features[1465:1469].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert features[1469:1473].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert features[1473:].sum().item() == 1.0


def test_v15_control_model_configs_freeze_losses_and_seeds() -> None:
    configs = v15.hypernetwork_control_configs()
    by_name = {config["control_type"]: config for config in configs}

    assert by_name["target_label_only_hypernetwork"]["seed"] == 20260621
    assert by_name["target_weight_mse_only_hypernetwork"]["loss_weights"] == {
        "compatible": 0.0,
        "conflict": 0.0,
        "delta_norm": 0.0001,
        "signature": 0.0,
        "source_l2": 0.0005,
        "target_bce": 0.0,
        "weight_mse": 1.0,
    }
    assert by_name["functional_only_hypernetwork"]["loss_weights"]["signature"] == 0.0
    assert by_name["signature_only_hypernetwork"]["loss_weights"]["target_bce"] == 0.0
    assert by_name["signature_only_hypernetwork"]["loss_weights"]["conflict"] == 0.0
    assert by_name["signature_only_hypernetwork"]["loss_weights"]["compatible"] == 0.0


def test_v15_sample_training_indices_are_deterministic_with_replacement() -> None:
    indices_a = v15.sample_training_indices(
        pair_count=10,
        steps=3,
        batch_size=4,
        seed=20260615,
    )
    indices_b = v15.sample_training_indices(
        pair_count=10,
        steps=3,
        batch_size=4,
        seed=20260615,
    )

    assert torch.equal(indices_a, indices_b)
    assert indices_a.shape == (3, 4)
    assert int(indices_a.max().item()) < 10
    assert int(indices_a.min().item()) >= 0


def test_v15_batch_signature_matches_v14_single_signature() -> None:
    probe_examples = v15.build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    weights = torch.stack([
        torch.zeros(345, dtype=torch.float32),
        torch.ones(345, dtype=torch.float32) * 0.01,
    ])
    batch = v15.differentiable_signature_batch(weights, probe_examples)
    single_0 = v15.v14.differentiable_signature(weights[0], probe_examples)
    single_1 = v15.v14.differentiable_signature(weights[1], probe_examples)

    assert batch.shape == (2, 560)
    assert torch.allclose(batch[0], single_0, atol=1e-5)
    assert torch.allclose(batch[1], single_1, atol=1e-5)


def test_v15_train_statistics_include_weight_norm_and_pair_hash() -> None:
    subjects = []
    for behavior_index, behavior in enumerate(v15.PATTERNS):
        for index in range(2):
            subjects.append({
                "behavior": behavior,
                "subject_id": f"{behavior}:{index:02d}",
                "weights": [float(behavior_index + index)] * 345,
                "signature": [float(index)] * 560,
            })
    stats = v15.fit_v15_train_statistics(subjects, build_v14_stats=False)

    assert stats["weight_mean"].shape == (345,)
    assert stats["weight_std"].shape == (345,)
    assert len(stats["training_pair_table"]) == 48
    assert stats["training_pair_table_hash"]
    assert set(stats["signature_centroids"]) == set(v15.PATTERNS)


def test_v15_training_batch_tensors_have_expected_shapes_for_small_pool() -> None:
    subjects = []
    for behavior_index, behavior in enumerate(v15.PATTERNS):
        for index in range(2):
            base = float(behavior_index + index)
            subjects.append({
                "behavior": behavior,
                "subject_id": f"{behavior}:{index:02d}",
                "weights": [base] * 345,
                "signature": [base] * 560,
            })
    stats = v15.fit_v15_train_statistics(subjects, build_v14_stats=False)
    batch = v15.build_training_batch_tensors(
        stats=stats,
        pair_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        target_signature_mode="paired_target_signature",
        alignment_mode="hungarian",
    )

    assert batch["features"].shape == (3, v15.EDITOR_INPUT_DIM)
    assert batch["source_weights"].shape == (3, 345)
    assert batch["target_weights"].shape == (3, 345)
    assert batch["target_signature_norm"].shape == (3, 560)
    assert batch["source_behaviors"] == [v15.PATTERNS[0]] * 3
    assert batch["target_behaviors"] == [v15.PATTERNS[1]] * 2 + [v15.PATTERNS[2]]


def test_v15_source_signature_batch_mode_replaces_target_signature() -> None:
    subjects = []
    for behavior_index, behavior in enumerate(v15.PATTERNS):
        for index in range(2):
            base = float(behavior_index + index)
            subjects.append({
                "behavior": behavior,
                "subject_id": f"{behavior}:{index:02d}",
                "weights": [base] * 345,
                "signature": [base] * 560,
            })
    stats = v15.fit_v15_train_statistics(subjects, build_v14_stats=False)
    batch = v15.build_training_batch_tensors(
        stats=stats,
        pair_indices=torch.tensor([0], dtype=torch.long),
        target_signature_mode="source_signature",
        alignment_mode="hungarian",
    )

    assert torch.allclose(batch["target_signature_norm"], batch["source_signature_norm"])


def test_v15_precomputed_training_tensors_match_batch_builder() -> None:
    subjects = []
    for behavior_index, behavior in enumerate(v15.PATTERNS):
        for index in range(2):
            base = float(behavior_index + index)
            subjects.append({
                "behavior": behavior,
                "subject_id": f"{behavior}:{index:02d}",
                "weights": [base] * 345,
                "signature": [base] * 560,
            })
    stats = v15.fit_v15_train_statistics(subjects, build_v14_stats=False)
    indices = torch.tensor([0, 3, 5], dtype=torch.long)
    direct = v15.build_training_batch_tensors(
        stats=stats,
        pair_indices=indices,
        target_signature_mode="paired_target_signature",
        alignment_mode="hungarian",
    )
    precomputed = v15.precompute_training_tensors(
        stats=stats,
        target_signature_mode="paired_target_signature",
        alignment_mode="hungarian",
    )
    gathered = v15.batch_from_precomputed_training_tensors(precomputed, indices)

    assert torch.allclose(gathered["features"], direct["features"])
    assert torch.allclose(gathered["source_weights"], direct["source_weights"])
    assert torch.allclose(gathered["target_weights"], direct["target_weights"])
    assert gathered["source_behaviors"] == direct["source_behaviors"]
    assert gathered["target_behaviors"] == direct["target_behaviors"]


def test_v15_train_hypernetwork_smoke_is_deterministic_for_weight_mse_only() -> None:
    subjects = []
    for behavior_index, behavior in enumerate(v15.PATTERNS):
        for index in range(2):
            base = float(behavior_index + index) / 100.0
            subjects.append({
                "behavior": behavior,
                "subject_id": f"{behavior}:{index:02d}",
                "weights": [base] * 345,
                "signature": [base] * 560,
            })
    stats = v15.fit_v15_train_statistics(subjects, build_v14_stats=False)
    loss_weights = {
        "compatible": 0.0,
        "conflict": 0.0,
        "delta_norm": 0.0001,
        "signature": 0.0,
        "source_l2": 0.0005,
        "target_bce": 0.0,
        "weight_mse": 1.0,
    }
    first = v15.train_hypernetwork_editor(
        stats=stats,
        seed=20260627,
        target_signature_mode="paired_target_signature",
        alignment_mode="hungarian",
        loss_weights=loss_weights,
        steps=2,
        batch_size=4,
    )
    second = v15.train_hypernetwork_editor(
        stats=stats,
        seed=20260627,
        target_signature_mode="paired_target_signature",
        alignment_mode="hungarian",
        loss_weights=loss_weights,
        steps=2,
        batch_size=4,
    )

    assert first["final_step"] == 2
    assert first["history"][-1]["loss"] >= 0.0
    for (name_a, tensor_a), (name_b, tensor_b) in zip(
        first["model"].state_dict().items(),
        second["model"].state_dict().items(),
    ):
        assert name_a == name_b
        assert torch.allclose(tensor_a, tensor_b)


def test_v15_signature_loss_normalizes_edited_signature_before_mse(monkeypatch) -> None:
    edited_weights = torch.zeros(1, 345, dtype=torch.float32)
    batch = {
        "source_weights": torch.zeros(1, 345, dtype=torch.float32),
        "target_signature_norm": torch.zeros(1, 2, dtype=torch.float32),
    }
    monkeypatch.setattr(
        v15,
        "differentiable_signature_batch",
        lambda weights, probe_examples: torch.tensor([[3.0, 8.0]], dtype=torch.float32),
    )
    loss, parts = v15.hypernetwork_training_loss(
        edited_weights=edited_weights,
        delta=torch.zeros_like(edited_weights),
        diagnostic_scale=torch.zeros(1, dtype=torch.float32),
        batch=batch,
        loss_weights={
            "compatible": 0.0,
            "conflict": 0.0,
            "delta_norm": 0.0,
            "signature": 1.0,
            "source_l2": 0.0,
            "target_bce": 0.0,
            "weight_mse": 0.0,
        },
        probe_examples=[],
        sig_mean=torch.tensor([1.0, 2.0], dtype=torch.float32),
        sig_std=torch.tensor([2.0, 3.0], dtype=torch.float32),
    )

    assert torch.allclose(loss, torch.tensor(2.5))
    assert parts["signature_mse"] == 2.5


def test_v15_seed_preflight_uses_v15_pool_configs() -> None:
    preflight = v15.build_v15_seed_preflight()
    assert preflight["passed"] is True
    ranges = {
        (item["pool"], item["pattern"]): item
        for item in preflight["seed_ranges"]
    }

    assert ranges[("train", v15.PATTERNS[0])]["start_seed"] == 75300000
    assert ranges[("development", v15.PATTERNS[0])]["start_seed"] == 76300000
    assert ranges[("final", v15.PATTERNS[0])]["start_seed"] == 77300000
    assert ranges[("train", v15.PATTERNS[1])]["start_seed"] == 75400000


def test_v15_contract_rejects_extra_combined_final_summary_fields(monkeypatch) -> None:
    monkeypatch.setattr(v15.v1, "sha256_file", lambda path: {
        "development_subjects.json": "devhash",
        "train_subjects.json": "trainhash",
    }.get(path.name, "otherhash"))
    counts_64 = {pattern: 64 for pattern in v15.PATTERNS}
    counts_24 = {pattern: 24 for pattern in v15.PATTERNS}
    combined_audit = {
        "claim_scope": v15.SOURCE_AUDIT_SCOPE,
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
        "claim_scope": v15.FINAL_REDACTED_SCOPE,
        "pool_file_sha256": "finalhash",
    }

    failures = v15.validate_source_pool_contract(
        train_path=Path("train_subjects.json"),
        eval_path=Path("development_subjects.json"),
        combined_audit_path=Path("combined_audit.json"),
        final_redacted_path=Path("final_redacted_audit.json"),
        train_payload={"claim_scope": v15.SOURCE_POOL_SCOPE},
        eval_payload={"claim_scope": v15.SOURCE_POOL_SCOPE},
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase="development",
    )

    joined = "\n".join(failures)
    assert "combined_audit.pool_summaries.final exposes forbidden keys" in joined
    assert "record_count" in joined
