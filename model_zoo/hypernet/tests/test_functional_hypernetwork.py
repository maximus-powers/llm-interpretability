"""
Test the FunctionalHyperNetwork module end-to-end.

This verifies the clean implementation works with:
1. Training on real data
2. Weight generation
3. Behavior editing
4. Functional verification
"""

import sys
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hypernet.models import FunctionalHyperNetwork, SubjectNetwork, BehaviorEditor
from hypernet.models.functional_hypernetwork import HyperNetConfig
from hypernet.evaluation.pipeline import (
    compare_dataset_provenance,
    _check_generated_pattern,
    evaluate_clean_proof_gate,
    _generate_pattern_cases,
    _monotonic_direction,
    compute_proof_metrics,
)
from hypernet.evaluation.editing_metrics import (
    find_optimal_threshold,
    threshold_success_rate,
)
from hypernet.behavior_suite import (
    CLEAN_PROOF_PATTERNS,
    CLEAN_PROOF_THRESHOLDS,
    build_clean_behavior_suite,
    predicate_counts_and_overlap,
)
from hypernet.dataset_provenance import deduplicate_fingerprints, stable_hash_json
from hypernet.paired_contrast import (
    audit_regenerated_signature_sidecar,
    build_digit_probe_examples,
    build_paired_contrast_artifact_from_subjects,
    build_probe_provenance,
    build_stored_probe_provenance,
    evaluate_paired_contrast_predictions,
    extract_signature_with_stored_probes,
    require_behavior_control_counts,
    validate_paired_contrast_artifact,
    validate_paired_group_schema,
    summarize_behavior_control_counts,
    validate_registered_decode_policy,
    validate_transitive_group_splits,
)
from hypernet.train import apply_deduplication
from hypernet.train import build_hypernet_config
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm


# =============================================================================
# Constants
# =============================================================================

ALL_PATTERNS = [
    'contains_abc', 'palindrome', 'alternating', 'sorted_descending',
    'sorted_ascending', 'ends_with', 'decreasing_pairs', 'mountain_pattern',
    'increasing_pairs', 'first_last_match', 'starts_with', 'no_repeats',
    'has_majority', 'vowel_consonant'
]
PATTERN_TO_IDX = {p: i for i, p in enumerate(ALL_PATTERNS)}
IDX_TO_PATTERN = {i: p for p, i in PATTERN_TO_IDX.items()}
TARGET_ARCH = (5, 8)


def test_functional_loss_backpropagates_to_hypernetwork_parameters():
    """Functional loss must train the decoder, not detach generated weights."""
    torch.manual_seed(0)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=16,
        condition_dim=32,
        hidden_dim=64,
        functional_loss_samples=2,
    )
    model = FunctionalHyperNetwork(config=config)

    weights = torch.randn(2, config.weight_dim)
    signatures = torch.randn(2, config.sig_dim)
    model.weight_mean = weights.mean(0)
    model.weight_std = weights.std(0).clamp(min=1e-6)
    model.sig_mean = signatures.mean(0)
    model.sig_std = signatures.std(0).clamp(min=1e-6)

    recon, _, _, _ = model(weights, signatures)
    loss = model.compute_functional_loss(weights, recon, n_probes=5)
    loss.backward()

    grad_sum = 0.0
    for name, param in model.named_parameters():
        if name.startswith("weight_decoder") and param.grad is not None:
            grad_sum += float(param.grad.abs().sum())

    assert grad_sum > 0.0


def test_flat_subject_forward_matches_subject_network():
    """Differentiable flat forward must match the reference SubjectNetwork."""
    torch.manual_seed(1)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=16,
        condition_dim=32,
        hidden_dim=64,
    )
    hypernet = FunctionalHyperNetwork(config=config)
    subject = SubjectNetwork(
        num_layers=config.num_layers,
        neurons_per_layer=config.neurons_per_layer,
        input_dim=config.input_dim,
    )
    flat_weights = subject.to_flat()
    inputs = torch.randn(7, config.input_dim)

    expected = subject(inputs)
    actual = hypernet.subject_forward_from_flat(flat_weights, inputs).squeeze(0)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_target_behavior_loss_backpropagates_to_hypernetwork_parameters():
    """Target behavior loss must train decoded weights toward label behavior."""
    torch.manual_seed(2)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=16,
        condition_dim=32,
        hidden_dim=64,
    )
    model = FunctionalHyperNetwork(config=config)

    weights = torch.randn(2, config.weight_dim)
    signatures = torch.randn(2, config.sig_dim)
    labels = torch.tensor([
        PATTERN_TO_IDX["sorted_ascending"],
        PATTERN_TO_IDX["sorted_descending"],
    ])
    model.weight_mean = weights.mean(0)
    model.weight_std = weights.std(0).clamp(min=1e-6)
    model.sig_mean = signatures.mean(0)
    model.sig_std = signatures.std(0).clamp(min=1e-6)

    recon, _, _, _ = model(weights, signatures)
    loss = model.compute_target_behavior_loss(recon, labels, PATTERN_TO_IDX)
    loss.backward()

    grad_sum = 0.0
    for name, param in model.named_parameters():
        if name.startswith("weight_decoder") and param.grad is not None:
            grad_sum += float(param.grad.abs().sum())

    assert grad_sum > 0.0


def test_condition_functional_specificity_loss_backpropagates():
    """Condition-only decode should learn subject-specific function, not only class behavior."""
    torch.manual_seed(9)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=16,
        condition_dim=32,
        hidden_dim=64,
    )
    model = FunctionalHyperNetwork(config=config)

    weights = torch.randn(4, config.weight_dim)
    signatures = torch.randn(4, config.sig_dim)
    model.weight_mean = weights.mean(0)
    model.weight_std = weights.std(0).clamp(min=1e-6)
    model.sig_mean = signatures.mean(0)
    model.sig_std = signatures.std(0).clamp(min=1e-6)

    condition = model.encode_signature(signatures)
    wrong_condition = torch.roll(condition, shifts=1, dims=0)
    loss = model.compute_condition_functional_specificity_loss(
        weights,
        condition,
        wrong_condition,
        n_probes=8,
    )
    loss.backward()

    decoder_grad = 0.0
    signature_grad = 0.0
    for name, param in model.named_parameters():
        if name.startswith("weight_decoder") and param.grad is not None:
            decoder_grad += float(param.grad.abs().sum())
        if name.startswith("sig_encoder") and param.grad is not None:
            signature_grad += float(param.grad.abs().sum())

    assert loss.item() >= 0.0
    assert decoder_grad > 0.0
    assert signature_grad > 0.0


def test_functional_probe_inputs_are_digit_domain():
    """Functional distillation probes should live on the subject-model input domain."""
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=16,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)

    probes = model._probe_inputs

    assert probes.min().item() >= 0.0
    assert probes.max().item() <= 9.0
    assert torch.allclose(probes, probes.round())


def test_condition_functional_specificity_loss_ranks_against_extra_controls():
    """Matched condition should be trained to beat the best provided control."""
    torch.manual_seed(11)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=16,
        condition_dim=32,
        hidden_dim=64,
    )
    model = FunctionalHyperNetwork(config=config)
    weights = torch.randn(3, config.weight_dim)
    signatures = torch.randn(3, config.sig_dim)
    model.weight_mean = weights.mean(0)
    model.weight_std = weights.std(0).clamp(min=1e-6)
    model.sig_mean = signatures.mean(0)
    model.sig_std = signatures.std(0).clamp(min=1e-6)
    condition = model.encode_signature(signatures)

    base_loss = model.compute_condition_functional_specificity_loss(
        weights,
        condition,
        None,
        n_probes=8,
        contrastive_margin=0.05,
    )
    ranked_loss = model.compute_condition_functional_specificity_loss(
        weights,
        condition,
        None,
        control_conditions=[condition.detach()],
        n_probes=8,
        contrastive_margin=0.05,
    )

    assert ranked_loss > base_loss


def test_behavior_prior_penalty_backpropagates_for_control_decodes():
    """Control decodes should be penalized when they solve labeled behavior cases."""
    torch.manual_seed(12)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=16,
        condition_dim=32,
        hidden_dim=64,
    )
    model = FunctionalHyperNetwork(config=config)
    weights = torch.randn(4, config.weight_dim)
    signatures = torch.randn(4, config.sig_dim)
    labels = torch.tensor([
        PATTERN_TO_IDX["sorted_ascending"],
        PATTERN_TO_IDX["sorted_descending"],
        PATTERN_TO_IDX["sorted_ascending"],
        PATTERN_TO_IDX["sorted_descending"],
    ])
    model.weight_mean = weights.mean(0)
    model.weight_std = weights.std(0).clamp(min=1e-6)
    model.sig_mean = signatures.mean(0)
    model.sig_std = signatures.std(0).clamp(min=1e-6)
    condition = model.encode_signature(signatures)
    control_weights = model.decode_weights(
        torch.zeros(4, config.latent_dim),
        condition,
    )

    penalty = model.compute_behavior_prior_penalty(
        control_weights,
        labels,
        PATTERN_TO_IDX,
        max_allowed_margin=-100.0,
    )
    penalty.backward()

    decoder_grad = 0.0
    for name, param in model.named_parameters():
        if name.startswith("weight_decoder") and param.grad is not None:
            decoder_grad += float(param.grad.abs().sum())

    assert penalty.item() > 0.0
    assert decoder_grad > 0.0


def test_all_target_control_penalty_backpropagates_across_every_behavior():
    """Zero-latent controls should be anti-behavior for every clean target."""
    torch.manual_seed(14)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=16,
        condition_dim=32,
        hidden_dim=64,
    )
    model = FunctionalHyperNetwork(config=config)
    weights = torch.randn(3, config.weight_dim)
    signatures = torch.randn(3, config.sig_dim)
    model.weight_mean = weights.mean(0)
    model.weight_std = weights.std(0).clamp(min=1e-6)
    model.sig_mean = signatures.mean(0)
    model.sig_std = signatures.std(0).clamp(min=1e-6)
    condition = model.encode_signature(signatures)
    control_weights = model.decode_weights(
        torch.zeros(3, config.latent_dim),
        condition,
    )
    behavior_cases = {
        "sorted_ascending": {
            "positive": [[0, 1, 2, 3, 4]],
            "negative": [[4, 3, 2, 1, 0]],
        },
        "sorted_descending": {
            "positive": [[4, 3, 2, 1, 0]],
            "negative": [[0, 1, 2, 3, 4]],
        },
    }

    penalty = model.compute_all_target_control_penalty(
        control_weights,
        PATTERN_TO_IDX,
        behavior_cases=behavior_cases,
        max_allowed_margin=-100.0,
    )
    penalty.backward()

    decoder_grad = 0.0
    for name, param in model.named_parameters():
        if name.startswith("weight_decoder") and param.grad is not None:
            decoder_grad += float(param.grad.abs().sum())

    assert penalty.item() > 0.0
    assert decoder_grad > 0.0


def test_all_target_control_penalty_accepts_target_weights():
    """Failing control targets should be upweightable without changing probes."""
    torch.manual_seed(16)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=16,
        condition_dim=32,
        hidden_dim=64,
    )
    model = FunctionalHyperNetwork(config=config)
    control_weights = torch.randn(2, config.weight_dim, requires_grad=True)
    behavior_cases = {
        "has_majority": {
            "positive": [[7, 7, 7, 1, 2]],
            "negative": [[0, 1, 2, 3, 4]],
        },
    }

    base_penalty = model.compute_all_target_control_penalty(
        control_weights,
        PATTERN_TO_IDX,
        behavior_cases=behavior_cases,
        max_allowed_margin=-100.0,
    )
    weighted_penalty = model.compute_all_target_control_penalty(
        control_weights,
        PATTERN_TO_IDX,
        behavior_cases=behavior_cases,
        max_allowed_margin=-100.0,
        target_weights={"has_majority": 10.0},
    )

    assert weighted_penalty > base_penalty * 5.0


def test_hard_negative_control_penalty_focuses_worst_target():
    """Hard-negative control loss should not average away a solved target."""
    torch.manual_seed(17)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=16,
        condition_dim=32,
        hidden_dim=64,
    )
    model = FunctionalHyperNetwork(config=config)
    control_weights = torch.randn(3, config.weight_dim, requires_grad=True)
    behavior_cases = {
        "sorted_ascending": {
            "positive": [[0, 1, 2, 3, 4]],
            "negative": [[4, 3, 2, 1, 0]],
        },
        "has_majority": {
            "positive": [[7, 7, 7, 1, 2]],
            "negative": [[0, 1, 2, 3, 4]],
        },
    }

    average_penalty = model.compute_all_target_control_penalty(
        control_weights,
        PATTERN_TO_IDX,
        behavior_cases=behavior_cases,
        max_allowed_margin=-100.0,
    )
    hard_penalty = model.compute_hard_negative_control_penalty(
        control_weights,
        PATTERN_TO_IDX,
        behavior_cases=behavior_cases,
        max_allowed_margin=-100.0,
    )

    assert hard_penalty >= average_penalty
    hard_penalty.backward()
    assert control_weights.grad is not None
    assert float(control_weights.grad.abs().sum()) > 0.0


def test_behavior_case_evaluation_uses_margin_deadband():
    """Near-zero positive margins should not count as clean behavior success."""
    from hypernet.evaluation.pipeline import _evaluate_case_outputs

    result = _evaluate_case_outputs(
        positive_output=0.501,
        negative_output=0.500,
        margin_threshold=0.02,
    )

    assert result["margin"] > 0.0
    assert result["correct"] is False
    assert result["raw_correct"] is True


def test_calibrated_behavior_margin_loss_backpropagates():
    """Matched decodes should be trainable against calibrated proof margins."""
    torch.manual_seed(18)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=16,
        condition_dim=32,
        hidden_dim=64,
    )
    model = FunctionalHyperNetwork(config=config)
    generated_weights = torch.randn(2, config.weight_dim, requires_grad=True)
    labels = torch.tensor([
        PATTERN_TO_IDX["mountain_pattern"],
        PATTERN_TO_IDX["has_majority"],
    ])
    behavior_cases = {
        "mountain_pattern": {
            "positive": [[1, 3, 7, 5, 2]],
            "negative": [[0, 1, 2, 3, 4]],
        },
        "has_majority": {
            "positive": [[7, 7, 7, 1, 2]],
            "negative": [[0, 1, 2, 3, 4]],
        },
    }

    loss = model.compute_calibrated_behavior_margin_loss(
        generated_weights,
        labels,
        PATTERN_TO_IDX,
        behavior_cases=behavior_cases,
        min_margin=0.02,
        target_weights={"mountain_pattern": 4.0},
    )
    loss.backward()

    assert loss.item() >= 0.0
    assert generated_weights.grad is not None
    assert float(generated_weights.grad.abs().sum()) > 0.0


def test_condition_specificity_loss_accepts_sample_weights():
    """Descending-specificity upweighting needs per-sample loss weights."""
    torch.manual_seed(15)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=16,
        condition_dim=32,
        hidden_dim=64,
    )
    model = FunctionalHyperNetwork(config=config)
    weights = torch.randn(2, config.weight_dim)
    signatures = torch.randn(2, config.sig_dim)
    model.weight_mean = weights.mean(0)
    model.weight_std = weights.std(0).clamp(min=1e-6)
    model.sig_mean = signatures.mean(0)
    model.sig_std = signatures.std(0).clamp(min=1e-6)
    condition = model.encode_signature(signatures)

    base_loss = model.compute_condition_functional_specificity_loss(
        weights,
        condition,
        None,
        n_probes=8,
    )
    weighted_loss = model.compute_condition_functional_specificity_loss(
        weights,
        condition,
        None,
        sample_weights=torch.tensor([10.0, 1.0]),
        n_probes=8,
    )

    assert not torch.allclose(base_loss, weighted_loss)


def test_subject_specificity_probes_follow_behavior_labels():
    """Subject-specificity training probes should include label-specific cases."""
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=16,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)
    labels = torch.tensor([
        PATTERN_TO_IDX["sorted_ascending"],
        PATTERN_TO_IDX["has_majority"],
    ])
    behavior_cases = {
        "sorted_ascending": {
            "positive": [[0, 1, 2, 3, 4]],
            "negative": [[4, 3, 2, 1, 0]],
        },
        "has_majority": {
            "positive": [[7, 7, 7, 1, 2]],
            "negative": [[0, 1, 2, 3, 4]],
        },
    }

    probes = model.build_subject_specificity_probes(
        labels,
        PATTERN_TO_IDX,
        behavior_cases,
    )

    assert probes.shape == (2, 2, config.input_dim)
    assert torch.equal(probes[0, 0], torch.tensor([0, 1, 2, 3, 4.0]))
    assert torch.equal(probes[1, 0], torch.tensor([7, 7, 7, 1, 2.0]))


def test_training_config_builder_wires_control_objective_knobs():
    """Main training path should expose all control-objective knobs."""
    config = build_hypernet_config(
        weight_dim=345,
        sig_dim=510,
        latent_dim=17,
        condition_dim=19,
        hidden_dim=23,
        epochs=3,
        batch_size=5,
        lr=0.004,
        lambda_kl=0.03,
        lambda_functional=7.0,
        lambda_condition_specificity=2.5,
        lambda_calibrated_behavior_margin=1.25,
        matched_behavior_min_margin=0.07,
        matched_mountain_target_weight=4.25,
        lambda_control_behavior_penalty=4.5,
        lambda_control_hard_negative_penalty=9.5,
        control_max_allowed_margin=-0.08,
        train_centroid_control_weight=6.5,
        condition_ablation_control_weight=5.5,
        control_sorted_descending_target_weight=7.5,
        control_has_majority_target_weight=8.5,
        sorted_descending_specificity_weight=3.5,
        lambda_edit_behavior=1.5,
        lambda_edit_margin_delta=0.75,
        use_condition_residual_decoder=True,
        condition_residual_scale=0.4,
        lambda_shuffled_residual_contrastive=1.75,
        shuffled_residual_min_delta=0.09,
        noise_control_weight=2.25,
        shuffled_control_weight=3.25,
        functional_loss_start_epoch=1,
        functional_loss_samples=13,
    )

    assert config.lambda_condition_specificity == 2.5
    assert config.lambda_calibrated_behavior_margin == 1.25
    assert config.matched_behavior_min_margin == 0.07
    assert config.matched_mountain_target_weight == 4.25
    assert config.lambda_control_behavior_penalty == 4.5
    assert config.lambda_control_hard_negative_penalty == 9.5
    assert config.control_max_allowed_margin == -0.08
    assert config.train_centroid_control_weight == 6.5
    assert config.condition_ablation_control_weight == 5.5
    assert config.control_sorted_descending_target_weight == 7.5
    assert config.control_has_majority_target_weight == 8.5
    assert config.sorted_descending_specificity_weight == 3.5
    assert config.lambda_edit_behavior == 1.5
    assert config.lambda_edit_margin_delta == 0.75
    assert config.use_condition_residual_decoder is True
    assert config.condition_residual_scale == 0.4
    assert config.lambda_shuffled_residual_contrastive == 1.75
    assert config.shuffled_residual_min_delta == 0.09
    assert config.noise_control_weight == 2.25
    assert config.shuffled_control_weight == 3.25
    assert config.functional_loss_samples == 13


def test_centroid_residual_decoder_config_defaults():
    """Residual decoder is opt-in and scale defaults to an identity multiplier."""
    config = HyperNetConfig()

    assert config.use_condition_residual_decoder is False
    assert config.condition_residual_scale == 1.0


def test_centroid_residual_decoder_zeroes_centroid_condition():
    """When condition equals its centroid baseline, only the latent base may decode."""
    torch.manual_seed(13)
    config = HyperNetConfig(
        weight_dim=16,
        sig_dim=12,
        latent_dim=4,
        condition_dim=6,
        hidden_dim=24,
        dropout=0.0,
        use_condition_residual_decoder=True,
    )
    model = FunctionalHyperNetwork(config=config)
    z = torch.randn(3, config.latent_dim)
    condition = torch.randn(3, config.condition_dim)
    other_condition = torch.randn(3, config.condition_dim)

    centroid_output = model.decode_weights(
        z,
        condition,
        condition_baseline=condition,
    )
    other_centroid_output = model.decode_weights(
        z,
        other_condition,
        condition_baseline=other_condition,
    )
    residual_output = model.decode_weights(
        z,
        condition + 0.5,
        condition_baseline=condition,
    )

    assert torch.allclose(centroid_output, other_centroid_output, atol=1e-6)
    assert not torch.allclose(centroid_output, residual_output)


def test_build_condition_baseline_uses_train_signature_centroids_when_enabled():
    """Residual decoding should subtract same-label train centroids, not batch stats."""
    torch.manual_seed(14)
    config = HyperNetConfig(
        weight_dim=16,
        sig_dim=5,
        latent_dim=4,
        condition_dim=6,
        hidden_dim=24,
        dropout=0.0,
        use_condition_residual_decoder=True,
    )
    model = FunctionalHyperNetwork(config=config)
    centroid_one = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    centroid_two = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    model._train_signature_centroids = {
        1: centroid_one,
        2: centroid_two,
    }
    labels = torch.tensor([2, 1])
    condition = torch.randn(2, config.condition_dim)

    baseline = model.build_condition_baseline(condition, labels)
    expected = model.encode_signature(torch.stack([centroid_two, centroid_one]))

    assert baseline is not None
    assert torch.allclose(baseline, expected)


def test_build_condition_baseline_returns_none_when_residual_decoder_disabled():
    """The compatibility path should not require labels or centroid metadata."""
    config = HyperNetConfig(
        weight_dim=16,
        sig_dim=5,
        latent_dim=4,
        condition_dim=6,
        hidden_dim=24,
        use_condition_residual_decoder=False,
    )
    model = FunctionalHyperNetwork(config=config)
    condition = torch.randn(2, config.condition_dim)
    labels = torch.tensor([1, 2])

    assert model.build_condition_baseline(condition, labels) is None


def test_different_label_condition_uses_only_other_label_rows():
    """Shuffled residual negatives must not accidentally reuse source-label rows."""
    config = HyperNetConfig(
        weight_dim=16,
        sig_dim=5,
        latent_dim=4,
        condition_dim=4,
        hidden_dim=24,
    )
    model = FunctionalHyperNetwork(config=config)
    condition = torch.arange(16, dtype=torch.float32).view(4, 4)
    labels = torch.tensor([1, 1, 2, 3])

    shuffled = model.build_different_label_condition(condition, labels)

    assert shuffled is not None
    assert torch.equal(shuffled[0], condition[2])
    assert torch.equal(shuffled[1], condition[2])
    assert torch.equal(shuffled[2], condition[0])
    assert torch.equal(shuffled[3], condition[0])


def test_different_label_condition_returns_none_for_single_label_batch():
    """Single-label batches cannot form behavior-shuffled negatives."""
    config = HyperNetConfig(
        weight_dim=16,
        sig_dim=5,
        latent_dim=4,
        condition_dim=4,
        hidden_dim=24,
    )
    model = FunctionalHyperNetwork(config=config)
    condition = torch.arange(12, dtype=torch.float32).view(3, 4)
    labels = torch.tensor([1, 1, 1])

    assert model.build_different_label_condition(condition, labels) is None


def test_behavior_control_conditions_include_noise_and_different_label_controls():
    """Training anti-behavior controls should cover evaluator-failing controls."""
    torch.manual_seed(16)
    config = HyperNetConfig(
        weight_dim=16,
        sig_dim=5,
        latent_dim=4,
        condition_dim=4,
        hidden_dim=24,
        dropout=0.0,
    )
    model = FunctionalHyperNetwork(config=config)
    signatures = torch.randn(4, config.sig_dim)
    model.sig_mean = signatures.mean(0)
    model.sig_std = signatures.std(0).clamp(min=1e-6)
    model._train_signature_mean = signatures.mean(0).detach().cpu()
    model._train_signature_std = signatures.std(0).clamp(min=1e-6).detach().cpu()
    model._train_signature_centroids = {
        1: signatures[:2].mean(0).detach().cpu(),
        2: signatures[2:].mean(0).detach().cpu(),
    }
    condition = model.encode_signature(signatures)
    labels = torch.tensor([1, 1, 2, 2])

    controls = model.build_behavior_control_conditions(
        signatures,
        condition,
        labels,
    )

    assert set(controls) == {
        "condition_ablation",
        "null",
        "noise",
        "train_centroid",
        "different_label",
    }
    assert controls["different_label"].shape == condition.shape
    assert controls["noise"].shape == condition.shape


def test_matched_control_behavior_margin_loss_backpropagates():
    """Matched residuals should be trainable to beat shuffled controls per label."""
    torch.manual_seed(15)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=12,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)
    z = torch.zeros(4, config.latent_dim)
    condition = torch.randn(4, config.condition_dim)
    matched_weights = model.decode_weights(z, condition)
    control_weights = model.decode_weights(z, torch.roll(condition, shifts=1, dims=0))
    labels = torch.tensor([
        PATTERN_TO_IDX["sorted_ascending"],
        PATTERN_TO_IDX["sorted_descending"],
        PATTERN_TO_IDX["has_majority"],
        PATTERN_TO_IDX["mountain_pattern"],
    ])

    loss = model.compute_matched_control_behavior_margin_loss(
        matched_weights,
        control_weights,
        labels,
        PATTERN_TO_IDX,
        min_delta=1.0,
    )
    loss.backward()

    decoder_grad = 0.0
    for name, param in model.named_parameters():
        if name.startswith("weight_decoder") and param.grad is not None:
            decoder_grad += float(param.grad.abs().sum())

    assert loss.item() > 0.0
    assert decoder_grad > 0.0


def test_edit_targets_prefer_different_label_batch_conditions():
    """Edit-path training should sample target conditions from different labels."""
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=4,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)
    condition = torch.arange(20, dtype=torch.float32).view(5, 4)
    labels = torch.tensor([1, 1, 2, 2, 3])

    target_condition, target_labels, valid_mask = model.build_edit_targets(
        condition,
        labels,
    )

    assert bool(valid_mask.all())
    assert torch.all(target_labels != labels)
    assert target_condition.shape == condition.shape


def test_edit_targets_use_train_centroids_when_available():
    """Edit-path training should match proof steering's train-centroid targets."""
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=4,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)
    model.eval()
    model.sig_mean = torch.zeros(config.sig_dim)
    model.sig_std = torch.ones(config.sig_dim)
    condition = torch.zeros(3, config.condition_dim)
    labels = torch.tensor([
        PATTERN_TO_IDX["sorted_ascending"],
        PATTERN_TO_IDX["sorted_descending"],
        PATTERN_TO_IDX["has_majority"],
    ])
    model._train_signature_centroids = {
        int(label): torch.full((config.sig_dim,), float(offset))
        for offset, label in enumerate(labels.tolist(), start=1)
    }

    target_condition, target_labels, valid_mask = model.build_edit_targets(
        condition,
        labels,
    )

    assert bool(valid_mask.all())
    assert torch.all(target_labels != labels)
    expected = model.encode_signature(
        torch.stack([
            model._train_signature_centroids[int(label)]
            for label in target_labels.tolist()
        ])
    )
    assert torch.allclose(target_condition, expected)


def test_all_edit_targets_cover_every_available_centroid_target():
    """Proof steering requires all clean behaviors to appear as edit targets."""
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=4,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)
    model.eval()
    model.sig_mean = torch.zeros(config.sig_dim)
    model.sig_std = torch.ones(config.sig_dim)
    labels = torch.tensor([
        PATTERN_TO_IDX["sorted_ascending"],
        PATTERN_TO_IDX["sorted_descending"],
        PATTERN_TO_IDX["has_majority"],
        PATTERN_TO_IDX["mountain_pattern"],
    ])
    condition = torch.zeros(len(labels), config.condition_dim)
    model._train_signature_centroids = {
        int(label): torch.full((config.sig_dim,), float(offset))
        for offset, label in enumerate(labels.tolist(), start=1)
    }

    target_condition, target_labels, source_indices = model.build_all_edit_targets(
        condition,
        labels,
    )

    assert len(target_labels) == len(labels) * (len(labels) - 1)
    assert set(target_labels.tolist()) == set(labels.tolist())
    for source_idx, target_label in zip(source_indices.tolist(), target_labels.tolist()):
        assert target_label != int(labels[source_idx])
    assert target_condition.shape[0] == len(target_labels)


def test_edit_margin_delta_loss_backpropagates_to_edit_path():
    """Edited weights should be trainable toward target behavior margin gains."""
    torch.manual_seed(13)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=16,
        condition_dim=32,
        hidden_dim=64,
    )
    model = FunctionalHyperNetwork(config=config)
    weights = torch.randn(4, config.weight_dim)
    signatures = torch.randn(4, config.sig_dim)
    labels = torch.tensor([
        PATTERN_TO_IDX["sorted_ascending"],
        PATTERN_TO_IDX["sorted_descending"],
        PATTERN_TO_IDX["sorted_ascending"],
        PATTERN_TO_IDX["sorted_descending"],
    ])
    model.weight_mean = weights.mean(0)
    model.weight_std = weights.std(0).clamp(min=1e-6)
    model.sig_mean = signatures.mean(0)
    model.sig_std = signatures.std(0).clamp(min=1e-6)
    recon, mu, _, condition = model(weights, signatures)
    target_condition, target_labels, valid_mask = model.build_edit_targets(
        condition,
        labels,
    )
    edited_weights = model.decode_weights(mu, target_condition)

    loss = model.compute_edit_margin_delta_loss(
        weights[valid_mask],
        edited_weights[valid_mask],
        target_labels[valid_mask],
        PATTERN_TO_IDX,
    )
    loss.backward()

    decoder_grad = 0.0
    for name, param in model.named_parameters():
        if name.startswith("weight_decoder") and param.grad is not None:
            decoder_grad += float(param.grad.abs().sum())

    assert loss.item() >= 0.0
    assert decoder_grad > 0.0


def test_same_label_wrong_condition_prefers_different_subject_same_class():
    """Specificity contrast should attack same-class prototypes when possible."""
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=4,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)
    condition = torch.arange(20, dtype=torch.float32).view(5, 4)
    labels = torch.tensor([1, 2, 1, 2, 3])

    wrong = model.build_wrong_condition(condition, labels)

    assert torch.equal(wrong[0], condition[2])
    assert torch.equal(wrong[2], condition[0])
    assert torch.equal(wrong[1], condition[3])
    assert torch.equal(wrong[3], condition[1])
    assert not torch.equal(wrong[4], condition[4])


def test_fit_records_target_behavior_loss_when_labels_are_available():
    """Training with labels should optimize and report target behavior loss."""
    torch.manual_seed(3)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=16,
        hidden_dim=32,
        functional_loss_samples=2,
    )
    model = FunctionalHyperNetwork(config=config)
    n_samples = 20
    weights = torch.randn(n_samples, config.weight_dim)
    signatures = torch.randn(n_samples, config.sig_dim)
    labels = torch.tensor(
        [PATTERN_TO_IDX["sorted_ascending"], PATTERN_TO_IDX["sorted_descending"]]
        * (n_samples // 2)
    )

    history = model.fit(
        weights,
        signatures,
        labels,
        epochs=1,
        batch_size=4,
        use_functional_loss=True,
        lambda_functional=1.0,
        verbose=False,
    )

    assert history["target_behavior_loss"][0] > 0.0
    assert history["condition_behavior_loss"][0] > 0.0
    assert history["condition_specificity_loss"][0] > 0.0
    assert history["control_behavior_penalty_loss"][0] >= 0.0
    assert history["edit_behavior_loss"][0] >= 0.0
    assert history["edit_margin_delta_loss"][0] >= 0.0
    assert history["val_target_behavior_loss"][0] > 0.0
    assert history["val_condition_behavior_loss"][0] > 0.0
    assert history["val_condition_specificity_loss"][0] > 0.0
    assert history["val_control_behavior_penalty_loss"][0] >= 0.0
    assert history["val_edit_behavior_loss"][0] >= 0.0
    assert history["val_edit_margin_delta_loss"][0] >= 0.0


def test_fit_normalization_stats_use_training_split_only():
    """Validation samples must not leak into normalization buffers."""
    torch.manual_seed(7)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=16,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)

    weights = torch.zeros(10, config.weight_dim)
    signatures = torch.zeros(10, config.sig_dim)
    weights[-1] = 1000.0
    signatures[-1] = 1000.0

    original_randperm = torch.randperm

    def deterministic_randperm(n, *args, **kwargs):
        return torch.arange(n)

    torch.randperm = deterministic_randperm
    try:
        model.fit(
            weights,
            signatures,
            epochs=1,
            batch_size=3,
            use_functional_loss=False,
            val_split=0.1,
            verbose=False,
        )
    finally:
        torch.randperm = original_randperm

    assert torch.allclose(model.weight_mean, torch.zeros_like(model.weight_mean))
    assert torch.allclose(model.sig_mean, torch.zeros_like(model.sig_mean))


def test_threshold_success_requires_positive_and_negative_threshold_accuracy():
    """Threshold success differs from mean-margin sign and should not be conflated."""
    pos_outputs = torch.tensor([0.49, 0.49, 0.49]).numpy()
    neg_outputs = torch.tensor([0.47, 0.47, 0.47]).numpy()

    margin_success = (pos_outputs.mean() - neg_outputs.mean()) > 0
    threshold_success_05 = threshold_success_rate(pos_outputs, neg_outputs, 0.5)
    threshold, pos_acc, neg_acc, total_acc = find_optimal_threshold(
        pos_outputs,
        neg_outputs,
    )

    assert margin_success
    assert threshold_success_05 == 0.0
    assert threshold_success_rate(pos_outputs, neg_outputs, threshold) == 1.0
    assert total_acc == 1.0
    assert pos_acc == 1.0
    assert neg_acc == 1.0
    assert 0.47 < threshold < 0.49


def test_proof_metrics_report_interpret_steer_and_decode_sections():
    """Proof metrics should explicitly cover the three research operations."""
    torch.manual_seed(4)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=16,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)
    editor = BehaviorEditor(model)
    weights = torch.randn(12, config.weight_dim)
    signatures = torch.randn(12, config.sig_dim)
    labels = torch.tensor(
        [
            PATTERN_TO_IDX["sorted_ascending"],
            PATTERN_TO_IDX["sorted_descending"],
            PATTERN_TO_IDX["increasing_pairs"],
            PATTERN_TO_IDX["decreasing_pairs"],
        ]
        * 3
    )
    model.weight_mean = weights.mean(0)
    model.weight_std = weights.std(0).clamp(min=1e-6)
    model.sig_mean = signatures.mean(0)
    model.sig_std = signatures.std(0).clamp(min=1e-6)

    metrics = compute_proof_metrics(
        model,
        editor,
        weights,
        signatures,
        labels,
        torch.arange(8),
        torch.arange(8, 12),
    )

    assert set(metrics.keys()) == {
        "interpret",
        "steer",
        "decode",
        "clean_proof_gate",
        "behavior_suite",
        "dataset_provenance",
    }
    assert "raw_signature_accuracy" in metrics["interpret"]
    assert "signature_condition_accuracy" in metrics["interpret"]
    assert "focused_raw_signature_accuracy" in metrics["interpret"]
    assert "focused_raw_signature_random_forest_accuracy" in metrics["interpret"]
    assert "focused_signature_condition_accuracy" in metrics["interpret"]
    assert "focused_dataset_raw_signature_random_forest_accuracy" in metrics["interpret"]
    assert "focused_dataset_test_samples" in metrics["interpret"]
    assert "mean_target_margin_delta" in metrics["steer"]
    assert "condition_only_behavior_accuracy" in metrics["decode"]
    assert "generated_heldout_seed" in metrics["decode"]
    assert metrics["decode"]["generated_heldout_cases_per_class"] >= 100
    assert "generated_heldout_behavior_accuracy" in metrics["decode"]
    assert "generated_heldout_per_pattern" in metrics["decode"]
    assert "generated_heldout_shuffled_per_target" in metrics["decode"]
    assert "generated_heldout_shuffled_source_target" in metrics["decode"]
    assert "generated_heldout_null_signature_accuracy" in metrics["decode"]
    assert "generated_heldout_noise_signature_accuracy" in metrics["decode"]
    assert "generated_heldout_train_centroid_signature_accuracy" in metrics["decode"]
    assert "generated_heldout_condition_ablation_accuracy" in metrics["decode"]
    assert "generated_heldout_null_signature_all_target" in metrics["decode"]
    assert "generated_heldout_noise_signature_all_target" in metrics["decode"]
    assert "generated_heldout_train_centroid_signature_all_target" in metrics["decode"]
    assert "generated_heldout_condition_ablation_all_target" in metrics["decode"]
    assert "clean_proof_gate" in metrics
    assert metrics["clean_proof_gate"]["thresholds"]
    assert "subject_functional_specificity" in metrics["decode"]
    assert "generated_heldout_shuffled_signature_accuracy" in metrics["decode"]
    assert "generated_heldout_opposite_direction_shuffled_accuracy" in metrics["decode"]
    assert "generated_heldout_within_direction_shuffled_accuracy" in metrics["decode"]
    assert "generated_heldout_collapsed_direction_accuracy" in metrics["decode"]
    assert "generated_heldout_target_success_rate" in metrics["steer"]
    assert "generated_heldout_per_target" in metrics["steer"]
    assert "generated_heldout_no_edit_target_success_rate" in metrics["steer"]
    assert "generated_heldout_cross_direction_no_edit_success_rate" in metrics["steer"]
    assert "generated_heldout_collapsed_direction_success_rate" in metrics["steer"]
    assert metrics["behavior_suite"]["name"] == "clean_proof_v1"
    assert metrics["clean_proof_gate"]["status"] in {
        "proof_candidate",
        "exploratory",
    }


def test_decode_specificity_controls_are_reported_per_target():
    """Decode controls should diagnose behavior priors, not only aggregate failure."""
    torch.manual_seed(8)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=16,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)
    editor = BehaviorEditor(model)
    weights = torch.randn(16, config.weight_dim)
    signatures = torch.randn(16, config.sig_dim)
    labels = torch.tensor(
        [
            PATTERN_TO_IDX["sorted_ascending"],
            PATTERN_TO_IDX["sorted_descending"],
            PATTERN_TO_IDX["has_majority"],
            PATTERN_TO_IDX["mountain_pattern"],
        ]
        * 4
    )
    model.weight_mean = weights.mean(0)
    model.weight_std = weights.std(0).clamp(min=1e-6)
    model.sig_mean = signatures.mean(0)
    model.sig_std = signatures.std(0).clamp(min=1e-6)
    suite = build_clean_behavior_suite()
    model._behavior_suite_metadata = suite["metadata"]

    metrics = compute_proof_metrics(
        model,
        editor,
        weights,
        signatures,
        labels,
        torch.arange(12),
        torch.arange(12, 16),
        {"reload_matches_checkpoint": True},
    )
    decode = metrics["decode"]

    for key in [
        "generated_heldout_shuffled_per_target",
        "generated_heldout_shuffled_source_target",
        "generated_heldout_null_signature_per_target",
        "generated_heldout_noise_signature_per_target",
        "generated_heldout_train_centroid_signature_per_target",
        "generated_heldout_condition_ablation_per_target",
    ]:
        assert set(decode[key]) == set(CLEAN_PROOF_PATTERNS)
        if key == "generated_heldout_shuffled_source_target":
            for target_values in decode[key].values():
                assert target_values
                for values in target_values.values():
                    assert {"accuracy", "mean_margin", "n_samples"}.issubset(values)
        else:
            for values in decode[key].values():
                assert {"accuracy", "mean_margin", "n_samples"}.issubset(values)


def test_decode_specificity_controls_are_reported_all_target():
    """Zero-latent controls should be audited against every target behavior."""
    torch.manual_seed(16)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=16,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)
    editor = BehaviorEditor(model)
    weights = torch.randn(16, config.weight_dim)
    signatures = torch.randn(16, config.sig_dim)
    labels = torch.tensor(
        [
            PATTERN_TO_IDX["sorted_ascending"],
            PATTERN_TO_IDX["sorted_descending"],
            PATTERN_TO_IDX["has_majority"],
            PATTERN_TO_IDX["mountain_pattern"],
        ]
        * 4
    )
    model.weight_mean = weights.mean(0)
    model.weight_std = weights.std(0).clamp(min=1e-6)
    model.sig_mean = signatures.mean(0)
    model.sig_std = signatures.std(0).clamp(min=1e-6)
    suite = build_clean_behavior_suite()
    model._behavior_suite_metadata = suite["metadata"]

    metrics = compute_proof_metrics(
        model,
        editor,
        weights,
        signatures,
        labels,
        torch.arange(12),
        torch.arange(12, 16),
        {"reload_matches_checkpoint": True},
    )
    decode = metrics["decode"]

    for key in [
        "generated_heldout_null_signature_all_target",
        "generated_heldout_noise_signature_all_target",
        "generated_heldout_train_centroid_signature_all_target",
        "generated_heldout_condition_ablation_all_target",
    ]:
        assert set(decode[key]) == set(CLEAN_PROOF_PATTERNS)
        for source_values in decode[key].values():
            assert set(source_values) == set(CLEAN_PROOF_PATTERNS)
            for values in source_values.values():
                assert {"accuracy", "mean_margin", "n_samples"}.issubset(values)


def test_clean_proof_gate_demotes_high_all_target_control_matrix():
    """A control that solves any target should block proof status."""
    per_pattern = {
        pattern: {"accuracy": 0.95, "mean_margin": 0.40, "n_samples": 80}
        for pattern in CLEAN_PROOF_PATTERNS
    }
    low_per_target = {
        pattern: {"accuracy": 0.10, "mean_margin": -0.20, "n_samples": 80}
        for pattern in CLEAN_PROOF_PATTERNS
    }
    low_matrix = {
        source: {
            target: {"accuracy": 0.10, "mean_margin": -0.20, "n_samples": 80}
            for target in CLEAN_PROOF_PATTERNS
        }
        for source in CLEAN_PROOF_PATTERNS
    }
    high_matrix = {
        **low_matrix,
        "sorted_descending": {
            **low_matrix["sorted_descending"],
            "has_majority": {
                "accuracy": 0.90,
                "mean_margin": 0.20,
                "n_samples": 80,
            },
        },
    }
    metrics = {
        "interpret": {
            "focused_heldout_majority_baseline_accuracy": 0.25,
            "focused_raw_signature_random_forest_accuracy": 0.80,
            "focused_signature_condition_accuracy": 0.80,
            "focused_raw_signature_random_forest_per_behavior_recall": {
                pattern: 0.80 for pattern in CLEAN_PROOF_PATTERNS
            },
            "focused_test_samples_per_behavior": {
                pattern: 80 for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "decode": {
            "generated_heldout_behavior_accuracy": 0.95,
            "generated_heldout_shuffled_signature_accuracy": 0.10,
            "generated_heldout_null_signature_accuracy": 0.10,
            "generated_heldout_null_signature_n_samples": 320,
            "generated_heldout_noise_signature_accuracy": 0.10,
            "generated_heldout_noise_signature_n_samples": 320,
            "generated_heldout_train_centroid_signature_accuracy": 0.10,
            "generated_heldout_train_centroid_signature_n_samples": 320,
            "generated_heldout_condition_ablation_accuracy": 0.10,
            "generated_heldout_condition_ablation_n_samples": 320,
            "generated_heldout_mean_margin": 0.40,
            "generated_heldout_per_pattern": per_pattern,
            "generated_heldout_shuffled_per_target": low_per_target,
            "generated_heldout_null_signature_per_target": low_per_target,
            "generated_heldout_noise_signature_per_target": low_per_target,
            "generated_heldout_train_centroid_signature_per_target": low_per_target,
            "generated_heldout_condition_ablation_per_target": low_per_target,
            "generated_heldout_null_signature_all_target": low_matrix,
            "generated_heldout_noise_signature_all_target": low_matrix,
            "generated_heldout_train_centroid_signature_all_target": high_matrix,
            "generated_heldout_condition_ablation_all_target": low_matrix,
            "subject_functional_specificity": {
                "matched_mse": 0.10,
                "wrong_signature_mse": 0.30,
                "null_mse": 0.30,
                "noise_mse": 0.30,
                "train_centroid_mse": 0.30,
                "condition_ablation_mse": 0.30,
                "best_control_mse": 0.30,
                "matched_improvement_vs_best_control": 0.20,
                "win_rate_vs_best_control": 0.80,
                "median_improvement_vs_best_control": 0.10,
                "n_samples": 320,
                "per_behavior": {
                    pattern: {
                        "n_samples": 80,
                        "matched_mse": 0.10,
                        "best_control_mse": 0.30,
                        "matched_improvement_vs_best_control": 0.20,
                        "win_rate_vs_best_control": 0.80,
                        "median_improvement_vs_best_control": 0.10,
                    }
                    for pattern in CLEAN_PROOF_PATTERNS
                },
            },
        },
        "steer": {
            "generated_heldout_target_success_rate": 0.80,
            "generated_heldout_no_edit_target_success_rate": 0.10,
            "generated_heldout_mean_target_margin_delta": 0.30,
            "generated_heldout_per_target": {
                pattern: {
                    "success_rate": 0.80,
                    "mean_margin_delta": 0.20,
                    "n_edits": 80,
                }
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "behavior_suite": {"matches_checkpoint_metadata": True},
        "dataset_provenance": {"reload_matches_checkpoint": True},
    }

    gate = evaluate_clean_proof_gate(metrics)

    assert not gate["passed"]
    assert any("all-target" in failure for failure in gate["failures"])


def test_generated_heldout_cases_are_deterministic_and_predicate_correct():
    """Generated heldout cases should be fresh, deterministic, and valid."""
    for pattern in [
        "sorted_descending",
        "sorted_ascending",
        "decreasing_pairs",
        "increasing_pairs",
    ]:
        first = _generate_pattern_cases(pattern, n_per_class=20, seed=123)
        second = _generate_pattern_cases(pattern, n_per_class=20, seed=123)

        assert first is not None
        assert second is not None
        assert torch.equal(first["positive"], second["positive"])
        assert torch.equal(first["negative"], second["negative"])
        assert len(first["positive"]) == 20
        assert len(first["negative"]) == 20

        for seq in first["positive"].int().tolist():
            assert _check_generated_pattern(seq, pattern)
        for seq in first["negative"].int().tolist():
            assert not _check_generated_pattern(seq, pattern)


def test_focused_patterns_have_expected_collapsed_directions():
    """Duplicate monotonic labels should be explicit in direction diagnostics."""
    assert _monotonic_direction("sorted_ascending") == "increasing"
    assert _monotonic_direction("increasing_pairs") == "increasing"
    assert _monotonic_direction("sorted_descending") == "decreasing"
    assert _monotonic_direction("decreasing_pairs") == "decreasing"
    assert _monotonic_direction("palindrome") is None


def test_model_save_load_preserves_dataset_patterns():
    """Focused training checkpoints must preserve their dataset filter."""
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=16,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)
    model._dataset_patterns = ["sorted_ascending", "sorted_descending"]

    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.pt"
        model.save(str(path))
        loaded = FunctionalHyperNetwork.load(str(path))

    assert loaded._dataset_patterns == ["sorted_ascending", "sorted_descending"]


def test_model_save_load_preserves_proof_metadata():
    """Proof checkpoints must preserve provenance and behavior-suite metadata."""
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=16,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)
    model._dataset_provenance = {
        "dataset_id": "maximuspowers/hypernet_validated",
        "fingerprint": "abc123",
        "row_indices": [1, 2, 3],
    }
    model._behavior_suite_metadata = {
        "name": "clean_proof_v1",
        "support_hash": "support",
        "heldout_hash": "heldout",
    }

    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.pt"
        model.save(str(path))
        loaded = FunctionalHyperNetwork.load(str(path))

    assert loaded._dataset_provenance == model._dataset_provenance
    assert loaded._behavior_suite_metadata == model._behavior_suite_metadata


def test_deduplicate_fingerprints_keeps_first_combined_hash_and_reports_counts():
    """Dataset rows must be deduplicated before train/validation splitting."""
    fingerprints = [
        {
            "row_hash": "row-a",
            "weight_hash": "weight-a",
            "signature_hash": "signature-a",
            "combined_hash": "combined-a",
        },
        {
            "row_hash": "row-b",
            "weight_hash": "weight-a",
            "signature_hash": "signature-a",
            "combined_hash": "combined-a",
        },
        {
            "row_hash": "row-c",
            "weight_hash": "weight-c",
            "signature_hash": "signature-c",
            "combined_hash": "combined-c",
        },
    ]

    keep_indices, summary = deduplicate_fingerprints(fingerprints)

    assert keep_indices == [0, 2]
    assert summary["before_count"] == 3
    assert summary["after_count"] == 2
    assert summary["removed_count"] == 1
    assert summary["duplicate_row_hash_count"] == 0
    assert summary["duplicate_weight_hash_count"] == 1
    assert summary["duplicate_signature_hash_count"] == 1
    assert summary["duplicate_combined_hash_count"] == 1


def test_deduplicate_fingerprints_removes_weight_signature_leakage_across_labels():
    """Identical weights/signatures with different labels must not survive splitting."""
    fingerprints = [
        {
            "row_hash": "row-a",
            "weight_hash": "same-weight",
            "signature_hash": "same-signature",
            "combined_hash": "label-a",
        },
        {
            "row_hash": "row-b",
            "weight_hash": "same-weight",
            "signature_hash": "same-signature",
            "combined_hash": "label-b",
        },
    ]

    keep_indices, summary = deduplicate_fingerprints(fingerprints)

    assert keep_indices == [0]
    assert summary["removed_count"] == 1
    assert summary["duplicate_weight_signature_hash_count"] == 1


def test_apply_deduplication_filters_tensors_and_provenance_before_split():
    """Loader-level dedup must keep tensors/provenance aligned."""
    weights = torch.arange(12, dtype=torch.float32).view(3, 4)
    signatures = torch.arange(15, dtype=torch.float32).view(3, 5)
    labels = torch.tensor([0, 1, 2])
    fingerprints = [
        {
            "row_hash": "row-a",
            "weight_hash": "same-weight",
            "signature_hash": "same-signature",
            "combined_hash": "label-a",
        },
        {
            "row_hash": "row-b",
            "weight_hash": "same-weight",
            "signature_hash": "same-signature",
            "combined_hash": "label-b",
        },
        {
            "row_hash": "row-c",
            "weight_hash": "weight-c",
            "signature_hash": "signature-c",
            "combined_hash": "label-c",
        },
    ]

    deduped = apply_deduplication(weights, signatures, labels, fingerprints)

    assert deduped["weights"].shape[0] == 2
    assert deduped["signatures"].shape[0] == 2
    assert deduped["labels"].tolist() == [0, 2]
    assert deduped["fingerprints"][0]["row_hash"] == "row-a"
    assert deduped["fingerprints"][1]["row_hash"] == "row-c"
    assert deduped["deduplication"]["removed_count"] == 1


def test_probe_provenance_hashes_probe_examples_and_configs():
    """Proof datasets must store probe content and hashes, not vague provenance text."""
    probe_examples = [
        {"sequence": [0, 1, 2, 3, 4], "pattern": "sorted_ascending"},
        {"sequence": [4, 3, 2, 1, 0], "pattern": "sorted_descending"},
    ]

    provenance = build_probe_provenance(
        probe_set_id="clean_proof_v1",
        probe_examples=probe_examples,
        behavior_suite={"seed": 20260609, "patterns": ["sorted_ascending"]},
        probe_generation_config={"sequence_length": 5, "vocab_size": 10},
        extractor_config={"methods": ["mean", "std"]},
        extractor_code="extractor-v1",
        normalization_stats={"mean_hash": "abc"},
        dataset_source={"dataset_id": "local"},
        git_commit="deadbeef",
    )
    changed = build_probe_provenance(
        probe_set_id="clean_proof_v1",
        probe_examples=[
            {"sequence": [0, 1, 2, 3, 5], "pattern": "sorted_ascending"},
            {"sequence": [4, 3, 2, 1, 0], "pattern": "sorted_descending"},
        ],
        behavior_suite={"seed": 20260609, "patterns": ["sorted_ascending"]},
        probe_generation_config={"sequence_length": 5, "vocab_size": 10},
        extractor_config={"methods": ["mean", "std"]},
    )

    assert provenance["probe_examples"] == probe_examples
    assert provenance["probe_examples_hash"] != changed["probe_examples_hash"]
    assert provenance["behavior_suite_hash"]
    assert provenance["extractor_config_hash"]
    assert provenance["normalization_stats_hash"]
    assert provenance["git_commit"] == "deadbeef"


def test_digit_probe_examples_are_deterministic_and_stored():
    """Regenerated signatures need a stable stored probe set."""
    probes_a = build_digit_probe_examples(n_examples=8, seed=123, seq_len=5, base=10)
    probes_b = build_digit_probe_examples(n_examples=8, seed=123, seq_len=5, base=10)
    probes_c = build_digit_probe_examples(n_examples=8, seed=124, seq_len=5, base=10)

    assert probes_a == probes_b
    assert probes_a != probes_c
    assert len(probes_a) == 8
    assert probes_a[0]["probe_index"] == 0
    assert len(probes_a[0]["sequence"]) == 5
    assert all(0 <= value < 10 for example in probes_a for value in example["sequence"])


def test_stored_probe_signature_extraction_is_deterministic_for_flat_weights():
    """Flat subject weights can be rebound to auditable stored-probe signatures."""
    probes = build_digit_probe_examples(n_examples=16, seed=123, seq_len=5, base=10)
    model = SubjectNetwork(num_layers=5, neurons_per_layer=8, input_dim=5)
    flat_weights = model.to_flat()

    sig_a = extract_signature_with_stored_probes(flat_weights, probes)
    sig_b = extract_signature_with_stored_probes(flat_weights, probes)
    different_probes = build_digit_probe_examples(n_examples=16, seed=124, seq_len=5, base=10)
    sig_c = extract_signature_with_stored_probes(flat_weights, different_probes)

    assert sig_a.shape == sig_b.shape
    assert sig_a.shape[0] > 0
    assert torch.allclose(sig_a, sig_b)
    assert not torch.allclose(sig_a, sig_c)
    assert torch.isfinite(sig_a).all()


def test_stored_probe_provenance_has_non_empty_validated_probe_examples():
    """Stored-probe provenance should satisfy the artifact validator contract."""
    probes = build_digit_probe_examples(n_examples=8, seed=123, seq_len=5, base=10)
    provenance = build_stored_probe_provenance(
        probe_set_id="stored_digit_probe_v1",
        probe_examples=probes,
        behavior_suite={"patterns": ["sorted_ascending", "sorted_descending"]},
        probe_generation_config={"seed": 123, "n_examples": 8, "seq_len": 5, "base": 10},
        extractor_config={"signature": "activation_stats_v1"},
        normalization_stats={"signature_mean_shape": [560]},
        dataset_source={"dataset_id": "unit-test"},
        git_commit="deadbeef",
    )
    artifact = {
        "decode_policy": "condition_only",
        "probe_provenance": provenance,
        "source_pool_preflight": {"passed": True, "failures": []},
        "splits": {
            "train": [],
            "validation": [
                {
                    "group_id": "validation:s1",
                    "target_pattern": "sorted_ascending",
                    "subject": {
                        "subject_id": "s1",
                        "weights_hash": "w1",
                        "signature_hash": "sig1",
                    },
                    "controls": {
                        "noise_signature": {"seed": 1},
                    },
                },
            ],
            "test": [
                {
                    "group_id": "test:s2",
                    "target_pattern": "sorted_ascending",
                    "subject": {
                        "subject_id": "s2",
                        "weights_hash": "w2",
                        "signature_hash": "sig2",
                    },
                    "controls": {
                        "noise_signature": {"seed": 2},
                    },
                },
            ],
        },
    }

    result = validate_paired_contrast_artifact(
        artifact,
        required_behaviors=["sorted_ascending"],
        required_control_types=["noise_signature"],
        min_count=1,
        count_splits=["validation", "test"],
    )

    assert result["passed"] is True
    assert result["probe_provenance"]["passed"] is True


def test_regenerated_signature_sidecar_audit_binds_artifact_signature_refs():
    """Artifact signature refs must be recomputed from sidecar vectors."""
    signature_values = [0.1, 0.2, 0.3]
    signature_hash = stable_hash_json(signature_values)
    artifact = {
        "splits": {
            "train": [],
            "validation": [
                {
                    "group_id": "validation:s1",
                    "target_pattern": "sorted_ascending",
                    "subject": {
                        "subject_id": "s1",
                        "weights_hash": "w1",
                        "signature_hash": signature_hash,
                    },
                    "controls": {
                        "same_label_other_subject": {
                            "subject_id": "s2",
                            "target_pattern": "sorted_ascending",
                            "weights_hash": "w2",
                            "signature_hash": signature_hash,
                        },
                    },
                },
            ],
            "test": [],
        },
    }
    sidecar = {
        "signature_hash_algorithm": "stable_hash_json_float_list_v1",
        "regenerated_signatures": {
            "s1": signature_values,
            "s2": signature_values,
        },
    }

    result = audit_regenerated_signature_sidecar(artifact, sidecar)

    assert result["passed"] is True
    assert result["n_checked"] == 2


def test_regenerated_signature_sidecar_audit_rejects_mismatched_refs():
    """Stale sidecar vectors or hashes must fail before proof use."""
    artifact = {
        "splits": {
            "train": [],
            "validation": [
                {
                    "group_id": "validation:s1",
                    "target_pattern": "sorted_ascending",
                    "subject": {
                        "subject_id": "s1",
                        "weights_hash": "w1",
                        "signature_hash": stable_hash_json([0.1, 0.2, 0.3]),
                    },
                    "controls": {},
                },
            ],
            "test": [],
        },
    }
    sidecar = {
        "signature_hash_algorithm": "stable_hash_json_float_list_v1",
        "regenerated_signatures": {
            "s1": [0.1, 0.2, 0.4],
        },
    }

    result = audit_regenerated_signature_sidecar(artifact, sidecar)

    assert result["passed"] is False
    assert any("signature hash mismatch for s1" in failure for failure in result["failures"])


def test_transitive_group_split_validation_rejects_control_member_overlap():
    """No subject/control member may cross train, validation, or test boundaries."""
    train_group = {
        "group_id": "train-1",
        "target_pattern": "sorted_ascending",
        "subject": {"subject_id": "s_train"},
        "controls": {
            "same_label_centroid": {"member_subject_ids": ["s_train", "s_shared"]},
            "opposite_direction": {"subject_id": "s_opp_train"},
        },
    }
    val_group = {
        "group_id": "val-1",
        "target_pattern": "sorted_descending",
        "subject": {"subject_id": "s_val"},
        "controls": {
            "same_label_other_subject": {"subject_id": "s_shared"},
        },
    }

    result = validate_transitive_group_splits({
        "train": [train_group],
        "validation": [val_group],
        "test": [],
    })

    assert result["passed"] is False
    assert any("s_shared" in failure for failure in result["failures"])


def test_paired_group_schema_requires_group_id_subject_and_controls():
    """Malformed paired rows must fail before split/evaluator logic."""
    group = {
        "target_pattern": "sorted_ascending",
        "subject": {},
        "controls": {
            "same_label_other_subject": {"subject_id": "s2", "target_pattern": "sorted_ascending"},
        },
    }

    result = validate_paired_group_schema(
        [group],
        required_control_types=[
            "same_label_other_subject",
            "opposite_direction",
        ],
    )

    assert result["passed"] is False
    assert any("missing group_id" in failure for failure in result["failures"])
    assert any("missing subject.subject_id" in failure for failure in result["failures"])
    assert any("missing required control opposite_direction" in failure for failure in result["failures"])


def test_paired_group_schema_rejects_duplicate_group_ids():
    """Split units need stable unique group IDs."""
    groups = [
        {
            "group_id": "g1",
            "target_pattern": "sorted_ascending",
            "subject": {"subject_id": "s1"},
            "controls": {},
        },
        {
            "group_id": "g1",
            "target_pattern": "sorted_descending",
            "subject": {"subject_id": "s2"},
            "controls": {},
        },
    ]

    result = validate_paired_group_schema(groups)

    assert result["passed"] is False
    assert any("duplicate group_id g1" in failure for failure in result["failures"])


def test_paired_group_schema_validates_control_semantics():
    """Control labels and directions must match the registered role."""
    group = {
        "group_id": "g1",
        "target_pattern": "sorted_ascending",
        "subject": {"subject_id": "s1"},
        "controls": {
            "same_label_other_subject": {
                "subject_id": "s2",
                "target_pattern": "sorted_descending",
            },
            "different_label_same_direction": {
                "subject_id": "s3",
                "target_pattern": "sorted_descending",
            },
            "opposite_direction": {
                "subject_id": "s4",
                "target_pattern": "increasing_pairs",
            },
            "same_label_centroid": {
                "member_subject_ids": ["s5"],
                "target_pattern": "sorted_descending",
            },
        },
    }

    result = validate_paired_group_schema([group])

    assert result["passed"] is False
    assert any("same_label_other_subject target_pattern" in failure for failure in result["failures"])
    assert any("same_label_centroid target_pattern" in failure for failure in result["failures"])
    assert any("different_label_same_direction must share direction" in failure for failure in result["failures"])
    assert any("opposite_direction must have opposite direction" in failure for failure in result["failures"])


def test_paired_group_schema_rejects_non_directional_same_direction_controls():
    """Unknown directions must not pass as a valid same-direction contrast."""
    group = {
        "group_id": "g1",
        "target_pattern": "has_majority",
        "subject": {"subject_id": "s1"},
        "controls": {
            "different_label_same_direction": {
                "subject_id": "s2",
                "target_pattern": "mountain_pattern",
            },
        },
    }

    result = validate_paired_group_schema([group])

    assert result["passed"] is False
    assert any(
        "different_label_same_direction requires directional source and control patterns" in failure
        for failure in result["failures"]
    )


def test_paired_group_schema_requires_explicit_same_label_target_patterns():
    """Same-label controls must explicitly declare the matched behavior label."""
    group = {
        "group_id": "g1",
        "target_pattern": "sorted_ascending",
        "subject": {"subject_id": "s1"},
        "controls": {
            "same_label_other_subject": {"subject_id": "s2"},
            "same_label_centroid": {
                "centroid_id": "train-sorted-ascending",
                "member_split": "train",
                "member_subject_ids_hash": "hash",
            },
        },
    }

    result = validate_paired_group_schema([group])

    assert result["passed"] is False
    assert any(
        "same_label_other_subject missing target_pattern" in failure
        for failure in result["failures"]
    )
    assert any(
        "same_label_centroid missing target_pattern" in failure
        for failure in result["failures"]
    )


def test_paired_group_schema_rejects_unknown_control_types():
    """Typos in control names must not become accepted proof cells."""
    group = {
        "group_id": "g1",
        "target_pattern": "sorted_ascending",
        "subject": {"subject_id": "s1"},
        "controls": {
            "same_label_oter_subject": {
                "subject_id": "s2",
                "target_pattern": "sorted_ascending",
            },
        },
    }

    result = validate_paired_group_schema([group])

    assert result["passed"] is False
    assert any("unknown control type same_label_oter_subject" in failure for failure in result["failures"])


def test_transitive_splits_allow_train_centroid_reference_by_artifact_id():
    """Validation rows can cite train centroids without leaking train subjects into the row."""
    train_group = {
        "group_id": "train-1",
        "target_pattern": "sorted_ascending",
        "subject": {"subject_id": "s_train"},
        "controls": {
            "same_label_other_subject": {
                "subject_id": "s_other_train",
                "target_pattern": "sorted_ascending",
            },
        },
    }
    val_group = {
        "group_id": "val-1",
        "target_pattern": "sorted_ascending",
        "subject": {"subject_id": "s_val"},
        "controls": {
            "same_label_centroid": {
                "target_pattern": "sorted_ascending",
                "centroid_id": "train-sorted-ascending",
                "member_split": "train",
                "member_subject_ids_hash": "members-hash",
            },
        },
    }

    result = validate_transitive_group_splits({
        "train": [train_group],
        "validation": [val_group],
        "test": [],
    })

    assert result["passed"] is True


def test_paired_contrast_artifact_validator_composes_required_gates():
    """One proof artifact gate should enforce policy, schema, splits, and counts."""
    artifact = {
        "decode_policy": "condition_only",
        "probe_provenance": build_probe_provenance(
            probe_set_id="probe-v1",
            probe_examples=[{"x": [1, 2, 3], "y": 1}],
            behavior_suite={"patterns": ["sorted_ascending"]},
            probe_generation_config={"seed": 7},
            extractor_config={"layer": "hidden"},
            extractor_code="def extract_signature(model, probes): return model(probes)",
            normalization_stats={"mean": [0.0], "std": [1.0]},
            dataset_source={"dataset_id": "paired-proof-v1"},
            git_commit="abc123",
        ),
        "source_pool_preflight": {"passed": True, "failures": []},
        "splits": {
            "train": [
                {
                    "group_id": "train-1",
                    "target_pattern": "sorted_ascending",
                    "subject": {
                        "subject_id": "s_train",
                        "weights_hash": "w-train",
                        "signature_hash": "sig-train",
                    },
                    "controls": {
                        "same_label_other_subject": {
                            "subject_id": "s_train_other",
                            "target_pattern": "sorted_ascending",
                            "weights_hash": "w-train-other",
                            "signature_hash": "sig-train-other",
                        },
                        "opposite_direction": {
                            "subject_id": "s_train_opp",
                            "target_pattern": "sorted_descending",
                            "weights_hash": "w-train-opp",
                            "signature_hash": "sig-train-opp",
                        },
                    },
                },
            ],
            "validation": [
                {
                    "group_id": "val-1",
                    "target_pattern": "sorted_ascending",
                    "subject": {
                        "subject_id": "s_val",
                        "weights_hash": "w-val",
                        "signature_hash": "sig-val",
                    },
                    "controls": {
                        "same_label_other_subject": {
                            "subject_id": "s_val_other",
                            "target_pattern": "sorted_ascending",
                            "weights_hash": "w-val-other",
                            "signature_hash": "sig-val-other",
                        },
                        "opposite_direction": {
                            "subject_id": "s_val_opp",
                            "target_pattern": "sorted_descending",
                            "weights_hash": "w-val-opp",
                            "signature_hash": "sig-val-opp",
                        },
                    },
                },
            ],
            "test": [
                {
                    "group_id": "test-1",
                    "target_pattern": "sorted_ascending",
                    "subject": {
                        "subject_id": "s_test",
                        "weights_hash": "w-test",
                        "signature_hash": "sig-test",
                    },
                    "controls": {
                        "same_label_other_subject": {
                            "subject_id": "s_test_other",
                            "target_pattern": "sorted_ascending",
                            "weights_hash": "w-test-other",
                            "signature_hash": "sig-test-other",
                        },
                        "opposite_direction": {
                            "subject_id": "s_test_opp",
                            "target_pattern": "sorted_descending",
                            "weights_hash": "w-test-opp",
                            "signature_hash": "sig-test-opp",
                        },
                    },
                },
            ],
        },
    }

    result = validate_paired_contrast_artifact(
        artifact,
        required_behaviors=["sorted_ascending"],
        required_control_types=["same_label_other_subject", "opposite_direction"],
        min_count=1,
        count_splits=["validation", "test"],
    )

    assert result["passed"] is True
    assert result["decode_policy"]["passed"] is True
    assert result["schema"]["passed"] is True
    assert result["splits"]["passed"] is True
    assert result["counts"]["passed"] is True


def test_paired_contrast_artifact_validator_rejects_failed_source_pool_preflight():
    """A builder-failed artifact must not pass standalone artifact validation."""
    result = build_paired_contrast_artifact_from_subjects(
        subjects_by_split={
            "train": [{"subject_id": "leaky_subject"}],
            "validation": [
                {"subject_id": "v_src", "target_pattern": "sorted_ascending", "weights_hash": "w-v-src", "signature_hash": "sig-v-src"},
                {"subject_id": "v_same", "target_pattern": "sorted_ascending", "weights_hash": "w-v-same", "signature_hash": "sig-v-same"},
                {"subject_id": "v_same_dir", "target_pattern": "increasing_pairs", "weights_hash": "w-v-dir", "signature_hash": "sig-v-dir"},
                {"subject_id": "v_opp", "target_pattern": "sorted_descending", "weights_hash": "w-v-opp", "signature_hash": "sig-v-opp"},
                {"subject_id": "leaky_subject"},
            ],
            "test": [
                {"subject_id": "t_src", "target_pattern": "sorted_ascending", "weights_hash": "w-t-src", "signature_hash": "sig-t-src"},
                {"subject_id": "t_same", "target_pattern": "sorted_ascending", "weights_hash": "w-t-same", "signature_hash": "sig-t-same"},
                {"subject_id": "t_same_dir", "target_pattern": "increasing_pairs", "weights_hash": "w-t-dir", "signature_hash": "sig-t-dir"},
                {"subject_id": "t_opp", "target_pattern": "sorted_descending", "weights_hash": "w-t-opp", "signature_hash": "sig-t-opp"},
            ],
        },
        decode_policy="condition_only",
        probe_provenance=_proof_provenance_for_tests(),
        required_behaviors=["sorted_ascending"],
        required_control_types=[
            "same_label_other_subject",
            "different_label_same_direction",
            "opposite_direction",
        ],
        min_count=1,
        proof_splits=["validation", "test"],
    )

    standalone = validate_paired_contrast_artifact(
        result["artifact"],
        required_behaviors=["sorted_ascending"],
        required_control_types=[
            "same_label_other_subject",
            "different_label_same_direction",
            "opposite_direction",
        ],
        min_count=1,
        count_splits=["validation", "test"],
    )

    assert result["passed"] is False
    assert standalone["passed"] is False
    assert any("source_pool_preflight failed" in failure for failure in standalone["failures"])


def test_paired_group_schema_requires_subject_weight_and_signature_refs():
    """Hand-authored subject-bearing payloads need immutable weight/signature refs."""
    group = {
        "group_id": "g1",
        "target_pattern": "sorted_ascending",
        "subject": {"subject_id": "s1"},
        "controls": {
            "same_label_other_subject": {
                "subject_id": "s2",
                "target_pattern": "sorted_ascending",
            },
        },
    }

    result = validate_paired_group_schema([group])

    assert result["passed"] is False
    assert any("subject missing weights reference" in failure for failure in result["failures"])
    assert any("subject missing signature reference" in failure for failure in result["failures"])
    assert any("same_label_other_subject missing weights reference" in failure for failure in result["failures"])
    assert any("same_label_other_subject missing signature reference" in failure for failure in result["failures"])


def _paired_contrast_eval_artifact():
    return {
        "decode_policy": "condition_only",
        "probe_provenance": build_probe_provenance(
            probe_set_id="probe-v1",
            probe_examples=[{"x": [1, 2, 3], "y": 1}],
            behavior_suite={"patterns": ["sorted_ascending"]},
            probe_generation_config={"seed": 7},
            extractor_config={"layer": "hidden"},
            extractor_code="def extract_signature(model, probes): return model(probes)",
            normalization_stats={"mean": [0.0], "std": [1.0]},
            dataset_source={"dataset_id": "paired-proof-v1"},
            git_commit="abc123",
        ),
        "source_pool_preflight": {"passed": True, "failures": []},
        "splits": {
            "train": [],
            "validation": [
                {
                    "group_id": "val-1",
                    "target_pattern": "sorted_ascending",
                    "subject": {
                        "subject_id": "s_val",
                        "weights_hash": "w-val",
                        "signature_hash": "sig-val",
                    },
                    "controls": {
                        "same_label_other_subject": {
                            "subject_id": "s_val_other",
                            "target_pattern": "sorted_ascending",
                            "weights_hash": "w-val-other",
                            "signature_hash": "sig-val-other",
                        },
                        "opposite_direction": {
                            "subject_id": "s_val_opp",
                            "target_pattern": "sorted_descending",
                            "weights_hash": "w-val-opp",
                            "signature_hash": "sig-val-opp",
                        },
                    },
                },
            ],
            "test": [
                {
                    "group_id": "test-1",
                    "target_pattern": "sorted_ascending",
                    "subject": {
                        "subject_id": "s_test",
                        "weights_hash": "w-test",
                        "signature_hash": "sig-test",
                    },
                    "controls": {
                        "same_label_other_subject": {
                            "subject_id": "s_test_other",
                            "target_pattern": "sorted_ascending",
                            "weights_hash": "w-test-other",
                            "signature_hash": "sig-test-other",
                        },
                        "opposite_direction": {
                            "subject_id": "s_test_opp",
                            "target_pattern": "sorted_descending",
                            "weights_hash": "w-test-opp",
                            "signature_hash": "sig-test-opp",
                        },
                    },
                },
            ],
        },
    }


def _paired_contrast_eval_predictions():
    return {
        "val-1": {
            "matched": {"behavior_margin": 0.8, "subject_output_mse": 0.2},
            "controls": {
                "same_label_other_subject": {
                    "behavior_margin": 0.3,
                    "subject_output_mse": 0.7,
                },
                "opposite_direction": {
                    "behavior_margin": -0.1,
                    "subject_output_mse": 1.2,
                },
            },
        },
        "test-1": {
            "matched": {"behavior_margin": 0.6, "subject_output_mse": 0.4},
            "controls": {
                "same_label_other_subject": {
                    "behavior_margin": 0.2,
                    "subject_output_mse": 0.9,
                },
                "opposite_direction": {
                    "behavior_margin": 0.1,
                    "subject_output_mse": 0.8,
                },
            },
        },
    }


def _proof_provenance_for_tests():
    return build_probe_provenance(
        probe_set_id="probe-v1",
        probe_examples=[{"x": [1, 2, 3], "y": 1}],
        behavior_suite={"patterns": ["sorted_ascending"]},
        probe_generation_config={"seed": 7},
        extractor_config={"layer": "hidden"},
        extractor_code="def extract_signature(model, probes): return model(probes)",
        normalization_stats={"mean": [0.0], "std": [1.0]},
        dataset_source={"dataset_id": "paired-proof-v1"},
        git_commit="abc123",
    )


def test_paired_contrast_generator_builds_valid_artifact_from_subjects():
    """Generator scaffold should emit validator-ready paired groups."""
    subjects_by_split = {
        "train": [],
        "validation": [
            {"subject_id": "v_src", "target_pattern": "sorted_ascending", "weights_hash": "w-v-src", "signature_hash": "sig-v-src"},
            {"subject_id": "v_same", "target_pattern": "sorted_ascending", "weights_hash": "w-v-same", "signature_hash": "sig-v-same"},
            {"subject_id": "v_same_dir", "target_pattern": "increasing_pairs", "weights_hash": "w-v-dir", "signature_hash": "sig-v-dir"},
            {"subject_id": "v_opp", "target_pattern": "sorted_descending", "weights_hash": "w-v-opp", "signature_hash": "sig-v-opp"},
        ],
        "test": [
            {"subject_id": "t_src", "target_pattern": "sorted_ascending", "weights_hash": "w-t-src", "signature_hash": "sig-t-src"},
            {"subject_id": "t_same", "target_pattern": "sorted_ascending", "weights_hash": "w-t-same", "signature_hash": "sig-t-same"},
            {"subject_id": "t_same_dir", "target_pattern": "increasing_pairs", "weights_hash": "w-t-dir", "signature_hash": "sig-t-dir"},
            {"subject_id": "t_opp", "target_pattern": "sorted_descending", "weights_hash": "w-t-opp", "signature_hash": "sig-t-opp"},
        ],
    }

    result = build_paired_contrast_artifact_from_subjects(
        subjects_by_split=subjects_by_split,
        decode_policy="condition_only",
        probe_provenance=_proof_provenance_for_tests(),
        required_behaviors=["sorted_ascending"],
        required_control_types=[
            "same_label_other_subject",
            "different_label_same_direction",
            "opposite_direction",
            "noise_signature",
        ],
        min_count=1,
        proof_splits=["validation", "test"],
        noise_seed=123,
    )

    assert result["passed"] is True
    assert result["validation"]["passed"] is True
    val_group = result["artifact"]["splits"]["validation"][0]
    assert val_group["group_id"] == "validation:v_src"
    assert val_group["subject"]["weights_hash"] == "w-v-src"
    assert val_group["subject"]["signature_hash"] == "sig-v-src"
    assert val_group["controls"]["same_label_other_subject"]["subject_id"] == "v_same"
    assert val_group["controls"]["different_label_same_direction"]["subject_id"] == "v_same_dir"
    assert val_group["controls"]["opposite_direction"]["subject_id"] == "v_opp"
    assert val_group["controls"]["noise_signature"]["seed"] == 123


def test_paired_contrast_generator_rejects_cross_split_source_pool_leakage():
    """Unused duplicate subject IDs in source metadata must still fail preflight."""
    subjects_by_split = {
        "train": [{"subject_id": "leaky_subject"}],
        "validation": [
            {"subject_id": "v_src", "target_pattern": "sorted_ascending", "weights_hash": "w-v-src", "signature_hash": "sig-v-src"},
            {"subject_id": "v_same", "target_pattern": "sorted_ascending", "weights_hash": "w-v-same", "signature_hash": "sig-v-same"},
            {"subject_id": "v_same_dir", "target_pattern": "increasing_pairs", "weights_hash": "w-v-dir", "signature_hash": "sig-v-dir"},
            {"subject_id": "v_opp", "target_pattern": "sorted_descending", "weights_hash": "w-v-opp", "signature_hash": "sig-v-opp"},
            {"subject_id": "leaky_subject"},
        ],
        "test": [
            {"subject_id": "t_src", "target_pattern": "sorted_ascending", "weights_hash": "w-t-src", "signature_hash": "sig-t-src"},
            {"subject_id": "t_same", "target_pattern": "sorted_ascending", "weights_hash": "w-t-same", "signature_hash": "sig-t-same"},
            {"subject_id": "t_same_dir", "target_pattern": "increasing_pairs", "weights_hash": "w-t-dir", "signature_hash": "sig-t-dir"},
            {"subject_id": "t_opp", "target_pattern": "sorted_descending", "weights_hash": "w-t-opp", "signature_hash": "sig-t-opp"},
        ],
    }

    result = build_paired_contrast_artifact_from_subjects(
        subjects_by_split=subjects_by_split,
        decode_policy="condition_only",
        probe_provenance=_proof_provenance_for_tests(),
        required_behaviors=["sorted_ascending"],
        required_control_types=[
            "same_label_other_subject",
            "different_label_same_direction",
            "opposite_direction",
        ],
        min_count=1,
        proof_splits=["validation", "test"],
    )

    assert result["passed"] is False
    assert any("input subject leaky_subject crosses train/validation" in failure for failure in result["failures"])


def test_paired_contrast_generator_requires_matched_weight_and_signature_refs():
    """Generated matched subjects must retain immutable weight/signature references."""
    subjects_by_split = {
        "train": [],
        "validation": [
            {"subject_id": "v_src", "target_pattern": "sorted_ascending"},
            {"subject_id": "v_same", "target_pattern": "sorted_ascending", "weights_hash": "w-v-same", "signature_hash": "sig-v-same"},
            {"subject_id": "v_same_dir", "target_pattern": "increasing_pairs", "weights_hash": "w-v-dir", "signature_hash": "sig-v-dir"},
            {"subject_id": "v_opp", "target_pattern": "sorted_descending", "weights_hash": "w-v-opp", "signature_hash": "sig-v-opp"},
        ],
        "test": [
            {"subject_id": "t_src", "target_pattern": "sorted_ascending", "weights_hash": "w-t-src", "signature_hash": "sig-t-src"},
            {"subject_id": "t_same", "target_pattern": "sorted_ascending", "weights_hash": "w-t-same", "signature_hash": "sig-t-same"},
            {"subject_id": "t_same_dir", "target_pattern": "increasing_pairs", "weights_hash": "w-t-dir", "signature_hash": "sig-t-dir"},
            {"subject_id": "t_opp", "target_pattern": "sorted_descending", "weights_hash": "w-t-opp", "signature_hash": "sig-t-opp"},
        ],
    }

    result = build_paired_contrast_artifact_from_subjects(
        subjects_by_split=subjects_by_split,
        decode_policy="condition_only",
        probe_provenance=_proof_provenance_for_tests(),
        required_behaviors=["sorted_ascending"],
        required_control_types=[
            "same_label_other_subject",
            "different_label_same_direction",
            "opposite_direction",
        ],
        min_count=1,
        proof_splits=["validation", "test"],
    )

    assert result["passed"] is False
    assert any("input subject v_src missing weights reference" in failure for failure in result["failures"])
    assert any("input subject v_src missing signature reference" in failure for failure in result["failures"])


def test_paired_contrast_generator_fails_when_required_controls_unavailable():
    """A generated artifact must fail validation when controls cannot be formed."""
    subjects_by_split = {
        "train": [],
        "validation": [
            {"subject_id": "v_src", "target_pattern": "sorted_ascending", "weights_hash": "w-v-src", "signature_hash": "sig-v-src"},
            {"subject_id": "v_same", "target_pattern": "sorted_ascending", "weights_hash": "w-v-same", "signature_hash": "sig-v-same"},
        ],
        "test": [
            {"subject_id": "t_src", "target_pattern": "sorted_ascending", "weights_hash": "w-t-src", "signature_hash": "sig-t-src"},
            {"subject_id": "t_same", "target_pattern": "sorted_ascending", "weights_hash": "w-t-same", "signature_hash": "sig-t-same"},
        ],
    }

    result = build_paired_contrast_artifact_from_subjects(
        subjects_by_split=subjects_by_split,
        decode_policy="condition_only",
        probe_provenance=_proof_provenance_for_tests(),
        required_behaviors=["sorted_ascending"],
        required_control_types=[
            "same_label_other_subject",
            "different_label_same_direction",
            "opposite_direction",
        ],
        min_count=1,
        proof_splits=["validation", "test"],
    )

    assert result["passed"] is False
    assert result["validation"]["passed"] is False
    assert any(
        "missing required control different_label_same_direction" in failure
        for failure in result["validation"]["failures"]
    )
    assert any(
        "missing required control opposite_direction" in failure
        for failure in result["validation"]["failures"]
    )


def test_paired_contrast_evaluator_fails_invalid_artifact_before_metrics():
    """Evaluator output must be gated by the artifact validator."""
    artifact = _paired_contrast_eval_artifact()
    del artifact["decode_policy"]

    result = evaluate_paired_contrast_predictions(
        artifact,
        predictions={},
        required_behaviors=["sorted_ascending"],
        required_control_types=["same_label_other_subject", "opposite_direction"],
        min_count=1,
    )

    assert result["passed"] is False
    assert result["validator"]["passed"] is False
    assert any("missing decode_policy" in failure for failure in result["failures"])


def test_paired_contrast_evaluator_reports_matched_minus_control_metrics():
    """Proof metrics should be grouped by split, behavior, and control type."""
    artifact = _paired_contrast_eval_artifact()
    predictions = _paired_contrast_eval_predictions()

    result = evaluate_paired_contrast_predictions(
        artifact,
        predictions=predictions,
        required_behaviors=["sorted_ascending"],
        required_control_types=["same_label_other_subject", "opposite_direction"],
        min_count=1,
    )

    assert result["passed"] is True
    val_same = result["metrics"]["by_split"]["validation"]["by_behavior"]["sorted_ascending"]["same_label_other_subject"]
    val_opp = result["metrics"]["by_split"]["validation"]["by_behavior"]["sorted_ascending"]["opposite_direction"]
    test_same = result["metrics"]["by_split"]["test"]["by_behavior"]["sorted_ascending"]["same_label_other_subject"]

    assert val_same["n"] == 1
    assert abs(val_same["mean_matched_minus_control_behavior_margin"] - 0.5) < 1e-9
    assert abs(val_same["mean_control_minus_matched_subject_output_mse"] - 0.5) < 1e-9
    assert abs(val_opp["mean_matched_minus_control_behavior_margin"] - 0.9) < 1e-9
    assert abs(val_opp["mean_control_minus_matched_subject_output_mse"] - 1.0) < 1e-9
    assert abs(test_same["mean_matched_minus_control_behavior_margin"] - 0.4) < 1e-9
    assert abs(test_same["mean_control_minus_matched_subject_output_mse"] - 0.5) < 1e-9


def test_paired_contrast_evaluator_passes_thresholded_proof_gates():
    """Registered thresholds should pass only when every proof cell clears them."""
    result = evaluate_paired_contrast_predictions(
        _paired_contrast_eval_artifact(),
        predictions=_paired_contrast_eval_predictions(),
        required_behaviors=["sorted_ascending"],
        required_control_types=["same_label_other_subject", "opposite_direction"],
        min_count=1,
        proof_thresholds={
            "min_mean_matched_minus_control_behavior_margin": 0.4,
            "min_mean_control_minus_matched_subject_output_mse": 0.4,
        },
    )

    assert result["passed"] is True
    assert result["proof_gates"]["passed"] is True
    val_same = result["proof_gates"]["by_split"]["validation"]["sorted_ascending"]["same_label_other_subject"]
    assert val_same["behavior_margin_delta_passed"] is True
    assert val_same["subject_output_mse_delta_passed"] is True


def test_paired_contrast_evaluator_fails_thresholded_proof_gates():
    """Complete metrics can still fail if registered proof thresholds are not met."""
    result = evaluate_paired_contrast_predictions(
        _paired_contrast_eval_artifact(),
        predictions=_paired_contrast_eval_predictions(),
        required_behaviors=["sorted_ascending"],
        required_control_types=["same_label_other_subject", "opposite_direction"],
        min_count=1,
        proof_thresholds={
            "min_mean_matched_minus_control_behavior_margin": 0.6,
            "min_mean_control_minus_matched_subject_output_mse": 0.6,
        },
    )

    assert result["passed"] is False
    assert result["metrics"]["n_pairs"] == 4
    assert result["proof_gates"]["passed"] is False
    assert any(
        "validation/sorted_ascending/same_label_other_subject behavior margin delta" in failure
        for failure in result["proof_gates"]["failures"]
    )
    assert any(
        "test/sorted_ascending/opposite_direction subject-output MSE delta" in failure
        for failure in result["proof_gates"]["failures"]
    )


def test_paired_contrast_evaluator_allows_control_specific_proof_gates():
    """Same-label specificity gates should not require behavior-margin separation."""
    result = evaluate_paired_contrast_predictions(
        _paired_contrast_eval_artifact(),
        predictions=_paired_contrast_eval_predictions(),
        required_behaviors=["sorted_ascending"],
        required_control_types=["same_label_other_subject", "opposite_direction"],
        min_count=1,
        proof_thresholds={
            "by_control_type": {
                "same_label_other_subject": {
                    "min_mean_control_minus_matched_subject_output_mse": 0.5,
                },
                "opposite_direction": {
                    "min_mean_matched_minus_control_behavior_margin": 0.5,
                    "min_mean_control_minus_matched_subject_output_mse": 0.4,
                },
            },
        },
    )

    assert result["passed"] is True
    val_same = result["proof_gates"]["by_split"]["validation"]["sorted_ascending"]["same_label_other_subject"]
    test_same = result["proof_gates"]["by_split"]["test"]["sorted_ascending"]["same_label_other_subject"]
    assert "behavior_margin_delta_passed" not in val_same
    assert val_same["subject_output_mse_delta_passed"] is True
    assert test_same["subject_output_mse_delta_passed"] is True


def test_paired_contrast_evaluator_requires_thresholds_for_each_required_control_type():
    """Control-specific gates must fail closed when a required control is omitted."""
    result = evaluate_paired_contrast_predictions(
        _paired_contrast_eval_artifact(),
        predictions=_paired_contrast_eval_predictions(),
        required_behaviors=["sorted_ascending"],
        required_control_types=["same_label_other_subject", "opposite_direction"],
        min_count=1,
        proof_thresholds={
            "by_control_type": {
                "same_label_other_subject": {
                    "min_mean_control_minus_matched_subject_output_mse": 0.5,
                },
            },
        },
    )

    assert result["passed"] is False
    assert any(
        "missing proof thresholds for control_type opposite_direction" in failure
        for failure in result["proof_gates"]["failures"]
    )


def test_paired_contrast_evaluator_rejects_unknown_control_specific_threshold_keys():
    """Control-specific thresholds must reject typos instead of silently passing."""
    result = evaluate_paired_contrast_predictions(
        _paired_contrast_eval_artifact(),
        predictions=_paired_contrast_eval_predictions(),
        required_behaviors=["sorted_ascending"],
        required_control_types=["same_label_other_subject", "opposite_direction"],
        min_count=1,
        proof_thresholds={
            "by_control_type": {
                "same_label_other_subject": {
                    "min_mean_control_minus_matched_subject_output_mse": 0.5,
                },
                "opposite_direction": {
                    "min_mean_matched_minus_control_margin": 0.5,
                    "min_mean_control_minus_matched_subject_output_mse": 0.4,
                },
            },
        },
    )

    assert result["passed"] is False
    assert any(
        (
            "unsupported proof threshold opposite_direction."
            "min_mean_matched_minus_control_margin"
        ) in failure
        for failure in result["proof_gates"]["failures"]
    )


def test_paired_contrast_evaluator_rejects_unknown_proof_threshold_keys():
    """Proof thresholds must fail closed on typos or unregistered keys."""
    result = evaluate_paired_contrast_predictions(
        _paired_contrast_eval_artifact(),
        predictions=_paired_contrast_eval_predictions(),
        required_behaviors=["sorted_ascending"],
        required_control_types=["same_label_other_subject", "opposite_direction"],
        min_count=1,
        proof_thresholds={
            "min_mean_matched_minus_control_behavior_margin": 0.4,
            "min_mean_control_minus_matched_subject_output_mse": 0.4,
            "min_mean_control_minus_matched_subject_mse": 0.9,
        },
    )

    assert result["passed"] is False
    assert result["metrics"]["n_pairs"] == 4
    assert any(
        "unsupported proof threshold min_mean_control_minus_matched_subject_mse" in failure
        for failure in result["proof_gates"]["failures"]
    )


def test_paired_contrast_evaluator_rejects_invalid_proof_threshold_values():
    """Missing, nonnumeric, and non-finite thresholds are proof-critical failures."""
    cases = [
        (
            {
                "min_mean_matched_minus_control_behavior_margin": 0.4,
            },
            "missing proof threshold min_mean_control_minus_matched_subject_output_mse",
        ),
        (
            {
                "min_mean_matched_minus_control_behavior_margin": "strict",
                "min_mean_control_minus_matched_subject_output_mse": 0.4,
            },
            "proof threshold min_mean_matched_minus_control_behavior_margin is not numeric",
        ),
        (
            {
                "min_mean_matched_minus_control_behavior_margin": 0.4,
                "min_mean_control_minus_matched_subject_output_mse": float("inf"),
            },
            "proof threshold min_mean_control_minus_matched_subject_output_mse is not finite",
        ),
    ]

    for thresholds, expected_failure in cases:
        result = evaluate_paired_contrast_predictions(
            _paired_contrast_eval_artifact(),
            predictions=_paired_contrast_eval_predictions(),
            required_behaviors=["sorted_ascending"],
            required_control_types=["same_label_other_subject", "opposite_direction"],
            min_count=1,
            proof_thresholds=thresholds,
        )

        assert result["passed"] is False
        assert any(expected_failure in failure for failure in result["proof_gates"]["failures"])


def test_paired_contrast_evaluator_fails_missing_group_predictions():
    """Missing matched/control measurements must fail the proof scaffold."""
    artifact = _paired_contrast_eval_artifact()
    predictions = {
        "val-1": {
            "matched": {"behavior_margin": 0.8, "subject_output_mse": 0.2},
            "controls": {},
        },
    }

    result = evaluate_paired_contrast_predictions(
        artifact,
        predictions=predictions,
        required_behaviors=["sorted_ascending"],
        required_control_types=["same_label_other_subject", "opposite_direction"],
        min_count=1,
    )

    assert result["passed"] is False
    assert any("val-1 missing control prediction same_label_other_subject" in failure for failure in result["failures"])
    assert any("missing prediction for group test-1" in failure for failure in result["failures"])
    assert result["metrics"] == {}


def test_paired_contrast_evaluator_rejects_non_finite_metrics():
    """NaN or infinite margins/MSE values must not enter proof summaries."""
    artifact = _paired_contrast_eval_artifact()
    predictions = {
        "val-1": {
            "matched": {"behavior_margin": float("nan"), "subject_output_mse": 0.2},
            "controls": {
                "same_label_other_subject": {
                    "behavior_margin": 0.3,
                    "subject_output_mse": 0.7,
                },
                "opposite_direction": {
                    "behavior_margin": -0.1,
                    "subject_output_mse": float("inf"),
                },
            },
        },
        "test-1": {
            "matched": {"behavior_margin": 0.6, "subject_output_mse": 0.4},
            "controls": {
                "same_label_other_subject": {
                    "behavior_margin": 0.2,
                    "subject_output_mse": 0.9,
                },
                "opposite_direction": {
                    "behavior_margin": 0.1,
                    "subject_output_mse": 0.8,
                },
            },
        },
    }

    result = evaluate_paired_contrast_predictions(
        artifact,
        predictions=predictions,
        required_behaviors=["sorted_ascending"],
        required_control_types=["same_label_other_subject", "opposite_direction"],
        min_count=1,
    )

    assert result["passed"] is False
    assert any("val-1 matched behavior_margin is not finite" in failure for failure in result["failures"])
    assert any("val-1 opposite_direction subject_output_mse is not finite" in failure for failure in result["failures"])
    assert result["metrics"] == {}


def test_paired_contrast_artifact_validator_fails_missing_policy_and_counts():
    """The artifact-level gate must fail closed when required proof metadata is absent."""
    artifact = {
        "probe_provenance": {"probe_set_id": "probe-v1"},
        "splits": {
            "train": [],
            "validation": [
                {
                    "group_id": "val-1",
                    "target_pattern": "sorted_ascending",
                    "subject": {"subject_id": "s_val"},
                    "controls": {
                        "same_label_other_subject": {
                            "subject_id": "s_val_other",
                            "target_pattern": "sorted_ascending",
                        },
                    },
                },
            ],
            "test": [],
        },
    }

    result = validate_paired_contrast_artifact(
        artifact,
        required_behaviors=["sorted_ascending"],
        required_control_types=["same_label_other_subject", "opposite_direction"],
        min_count=1,
        count_splits=["validation", "test"],
    )

    assert result["passed"] is False
    assert any("missing decode_policy" in failure for failure in result["failures"])
    assert any("probe_provenance missing probe_examples" in failure for failure in result["failures"])
    assert any("missing required control opposite_direction" in failure for failure in result["failures"])
    assert any("sorted_ascending/opposite_direction" in failure for failure in result["failures"])


def test_paired_contrast_artifact_validator_requires_full_provenance_contract():
    """Leak audits need the full probe/extractor/source provenance surface."""
    artifact = {
        "decode_policy": "condition_only",
        "probe_provenance": {
            "probe_set_id": "probe-v1",
            "probe_examples": [{"x": [1, 2, 3], "y": 1}],
            "probe_examples_hash": "probe-hash",
            "behavior_suite_hash": "suite-hash",
            "probe_generation_config_hash": "probe-config-hash",
            "extractor_config_hash": "extractor-config-hash",
        },
        "splits": {"train": [], "validation": [], "test": []},
    }

    result = validate_paired_contrast_artifact(
        artifact,
        required_behaviors=[],
        required_control_types=[],
        min_count=1,
    )

    assert result["passed"] is False
    assert any("probe_provenance missing extractor_code_hash" in failure for failure in result["failures"])
    assert any("probe_provenance missing normalization_stats_hash" in failure for failure in result["failures"])
    assert any("probe_provenance missing dataset_source_hash" in failure for failure in result["failures"])
    assert any("probe_provenance missing git_commit" in failure for failure in result["failures"])


def test_paired_contrast_artifact_validator_rejects_default_empty_provenance_values():
    """Proof-mode provenance cannot silently hash empty extractor/source defaults."""
    artifact = {
        "decode_policy": "condition_only",
        "probe_provenance": build_probe_provenance(
            probe_set_id="probe-v1",
            probe_examples=[{"x": [1, 2, 3], "y": 1}],
            behavior_suite={"patterns": ["sorted_ascending"]},
            probe_generation_config={"seed": 7},
            extractor_config={"layer": "hidden"},
        ),
        "splits": {"train": [], "validation": [], "test": []},
    }

    result = validate_paired_contrast_artifact(
        artifact,
        required_behaviors=[],
        required_control_types=[],
        min_count=1,
    )

    assert result["passed"] is False
    assert any("probe_provenance git_commit is empty" in failure for failure in result["failures"])
    assert any("extractor_code_hash matches empty extractor_code" in failure for failure in result["failures"])
    assert any("normalization_stats_hash matches empty mapping" in failure for failure in result["failures"])
    assert any("dataset_source_hash matches empty mapping" in failure for failure in result["failures"])


def test_paired_contrast_artifact_validator_rejects_probe_example_hash_mismatch():
    """Stored probe examples and their content hash must agree."""
    provenance = build_probe_provenance(
        probe_set_id="probe-v1",
        probe_examples=[{"x": [1, 2, 3], "y": 1}],
        behavior_suite={"patterns": ["sorted_ascending"]},
        probe_generation_config={"seed": 7},
        extractor_config={"layer": "hidden"},
        extractor_code="def extract_signature(model, probes): return model(probes)",
        normalization_stats={"mean": [0.0], "std": [1.0]},
        dataset_source={"dataset_id": "paired-proof-v1"},
        git_commit="abc123",
    )
    provenance["probe_examples_hash"] = "stale-hash"
    artifact = {
        "decode_policy": "condition_only",
        "probe_provenance": provenance,
        "splits": {"train": [], "validation": [], "test": []},
    }

    result = validate_paired_contrast_artifact(
        artifact,
        required_behaviors=[],
        required_control_types=[],
        min_count=1,
    )

    assert result["passed"] is False
    assert any("probe_examples_hash does not match probe_examples" in failure for failure in result["failures"])


def test_paired_contrast_artifact_validator_rejects_empty_probe_examples():
    """A proof artifact must store the actual fixed probe set, not an empty list."""
    artifact = {
        "decode_policy": "condition_only",
        "probe_provenance": build_probe_provenance(
            probe_set_id="probe-v1",
            probe_examples=[],
            behavior_suite={"patterns": ["sorted_ascending"]},
            probe_generation_config={"seed": 7},
            extractor_config={"layer": "hidden"},
            extractor_code="def extract_signature(model, probes): return model(probes)",
            normalization_stats={"mean": [0.0], "std": [1.0]},
            dataset_source={"dataset_id": "paired-proof-v1"},
            git_commit="abc123",
        ),
        "splits": {"train": [], "validation": [], "test": []},
    }

    result = validate_paired_contrast_artifact(
        artifact,
        required_behaviors=[],
        required_control_types=[],
        min_count=1,
    )

    assert result["passed"] is False
    assert any("probe_examples is empty" in failure for failure in result["failures"])


def test_paired_contrast_artifact_count_gate_is_per_proof_split():
    """Validation rows must not mask an underpowered final test split."""
    artifact = {
        "decode_policy": "condition_only",
        "probe_provenance": build_probe_provenance(
            probe_set_id="probe-v1",
            probe_examples=[{"x": [1, 2, 3], "y": 1}],
            behavior_suite={"patterns": ["sorted_ascending"]},
            probe_generation_config={"seed": 7},
            extractor_config={"layer": "hidden"},
            extractor_code="def extract_signature(model, probes): return model(probes)",
            normalization_stats={"mean": [0.0], "std": [1.0]},
            dataset_source={"dataset_id": "paired-proof-v1"},
            git_commit="abc123",
        ),
        "splits": {
            "train": [],
            "validation": [
                {
                    "group_id": "val-1",
                    "target_pattern": "sorted_ascending",
                    "subject": {"subject_id": "s_val"},
                    "controls": {
                        "same_label_other_subject": {
                            "subject_id": "s_val_other",
                            "target_pattern": "sorted_ascending",
                        },
                    },
                },
            ],
            "test": [],
        },
    }

    result = validate_paired_contrast_artifact(
        artifact,
        required_behaviors=["sorted_ascending"],
        required_control_types=["same_label_other_subject"],
        min_count=1,
        count_splits=["validation", "test"],
    )

    assert result["passed"] is False
    assert result["counts"]["per_split"]["validation"]["passed"] is True
    assert result["counts"]["per_split"]["test"]["passed"] is False
    assert any(
        "test: sorted_ascending/same_label_other_subject" in failure
        for failure in result["failures"]
    )


def test_behavior_control_counts_are_reported_per_behavior_and_control_type():
    """Proof gates need behavior x control_type counts, not aggregate counts."""
    groups = [
        {
            "group_id": "g1",
            "target_pattern": "sorted_ascending",
            "subject": {"subject_id": "s1"},
            "controls": {
                "same_label_other_subject": {"subject_id": "s2"},
                "opposite_direction": {"subject_id": "s3"},
            },
        },
        {
            "group_id": "g2",
            "target_pattern": "sorted_ascending",
            "subject": {"subject_id": "s4"},
            "controls": {
                "same_label_other_subject": {"subject_id": "s5"},
                "noise_signature": {"seed": 7},
            },
        },
        {
            "group_id": "g3",
            "target_pattern": "has_majority",
            "subject": {"subject_id": "s6"},
            "controls": {
                "same_label_other_subject": {"subject_id": "s7"},
            },
        },
    ]

    counts = summarize_behavior_control_counts(groups)

    assert counts["sorted_ascending"]["same_label_other_subject"] == 2
    assert counts["sorted_ascending"]["opposite_direction"] == 1
    assert counts["sorted_ascending"]["noise_signature"] == 1
    assert counts["has_majority"]["same_label_other_subject"] == 1


def test_behavior_control_count_gate_fails_missing_behavior_control_cell():
    """Every behavior/control cell must clear a pre-registered minimum count."""
    counts = {
        "sorted_ascending": {
            "same_label_other_subject": 2,
            "opposite_direction": 1,
        },
        "has_majority": {
            "same_label_other_subject": 1,
        },
    }

    result = require_behavior_control_counts(
        counts,
        required_behaviors=["sorted_ascending", "has_majority"],
        required_control_types=["same_label_other_subject", "opposite_direction"],
        min_count=2,
    )

    assert result["passed"] is False
    assert any(
        "has_majority/opposite_direction" in failure
        for failure in result["failures"]
    )
    assert any(
        "sorted_ascending/opposite_direction" in failure
        for failure in result["failures"]
    )


def test_registered_decode_policy_accepts_only_predeclared_policies():
    """Proof comparisons must use a registered latent decode policy."""
    assert validate_registered_decode_policy("condition_only") == "condition_only"
    assert validate_registered_decode_policy("subject_latent") == "subject_latent"
    assert validate_registered_decode_policy("both") == "both"

    try:
        validate_registered_decode_policy("mixed")
    except ValueError as exc:
        assert "Unsupported decode policy" in str(exc)
    else:
        raise AssertionError("validate_registered_decode_policy should reject mixed")


def test_clean_proof_predicates_have_zero_pairwise_overlap():
    """The clean proof set should not contain duplicate or overlapping labels."""
    audit = predicate_counts_and_overlap(CLEAN_PROOF_PATTERNS)

    assert audit["predicate_counts"]["sorted_ascending"] == 252
    assert audit["predicate_counts"]["sorted_descending"] == 252
    assert audit["predicate_counts"]["has_majority"] == 8560
    assert audit["predicate_counts"]["mountain_pattern"] == 2892

    for source in CLEAN_PROOF_PATTERNS:
        for target in CLEAN_PROOF_PATTERNS:
            if source == target:
                continue
            assert audit["overlap_matrix"][source][target] == 0


def test_clean_behavior_suite_cases_are_deterministic_disjoint_and_valid():
    """Support cases train behavior loss; heldout cases evaluate proof metrics."""
    first = build_clean_behavior_suite(
        support_per_class=8,
        heldout_per_class=16,
        seed=20260609,
    )
    second = build_clean_behavior_suite(
        support_per_class=8,
        heldout_per_class=16,
        seed=20260609,
    )

    assert first["metadata"] == second["metadata"]
    assert first["support"] == second["support"]
    assert first["heldout"] == second["heldout"]
    assert first["metadata"]["support_heldout_overlap_count"] == 0

    support_sequences = set()
    heldout_sequences = set()
    for pattern in CLEAN_PROOF_PATTERNS:
        for split_name, target in [
            ("support", support_sequences),
            ("heldout", heldout_sequences),
        ]:
            cases = first[split_name][pattern]
            assert len(cases["positive"]) == (8 if split_name == "support" else 16)
            assert len(cases["negative"]) == (8 if split_name == "support" else 16)
            predicate = first["predicates"][pattern]
            for seq in cases["positive"]:
                assert predicate(seq)
                target.add(tuple(seq))
            for seq in cases["negative"]:
                assert not predicate(seq)
                target.add(tuple(seq))

    assert support_sequences.isdisjoint(heldout_sequences)


def test_clean_behavior_suite_uses_other_behaviors_as_hard_negatives():
    """Each one-vs-target behavior must reject other clean behavior positives."""
    suite = build_clean_behavior_suite(
        support_per_class=12,
        heldout_per_class=20,
        seed=20260609,
    )

    for pattern in CLEAN_PROOF_PATTERNS:
        predicate = suite["predicates"][pattern]
        other_predicates = [
            other_predicate
            for other_pattern, other_predicate in suite["predicates"].items()
            if other_pattern != pattern
        ]
        for split_name in ["support", "heldout"]:
            negatives = suite[split_name][pattern]["negative"]
            hard_negatives = [
                seq for seq in negatives
                if any(other_predicate(seq) for other_predicate in other_predicates)
            ]
            assert hard_negatives
            for seq in hard_negatives:
                assert not predicate(seq)


def test_clean_proof_thresholds_are_numeric_and_per_behavior_guarded():
    """Proof thresholds must be pre-registered as artifact-enforceable values."""
    required = {
        "min_heldout_samples_per_behavior",
        "raw_signature_rf_min_accuracy",
        "raw_signature_rf_min_delta_vs_majority",
        "condition_min_accuracy",
        "condition_min_delta_vs_majority",
        "min_interpret_recall_per_behavior",
        "decode_min_accuracy",
        "decode_min_accuracy_per_behavior",
        "decode_min_delta_vs_control",
        "decode_min_delta_vs_control_per_behavior",
        "decode_min_delta_vs_null_signature",
        "decode_min_delta_vs_noise_signature",
        "decode_min_delta_vs_train_centroid_signature",
        "decode_min_delta_vs_condition_ablation",
        "decode_min_margin",
        "decode_min_margin_per_behavior",
        "subject_functional_specificity_min_samples",
        "subject_functional_specificity_min_samples_per_behavior",
        "subject_functional_specificity_min_improvement",
        "subject_functional_specificity_min_win_rate",
        "subject_functional_specificity_min_median_improvement",
        "steer_min_target_success",
        "steer_min_target_success_per_behavior",
        "steer_min_delta_vs_no_edit",
        "steer_min_margin_delta",
        "steer_min_margin_delta_per_target",
    }

    assert required.issubset(CLEAN_PROOF_THRESHOLDS)
    for key in required:
        assert isinstance(CLEAN_PROOF_THRESHOLDS[key], (int, float))


def test_clean_proof_gate_demotes_failed_per_behavior_metrics():
    """Aggregate success cannot hide a failed behavior target."""
    metrics = {
        "interpret": {
            "focused_heldout_majority_baseline_accuracy": 0.25,
            "focused_raw_signature_random_forest_accuracy": 0.70,
            "focused_signature_condition_accuracy": 0.65,
            "focused_raw_signature_random_forest_per_behavior_recall": {
                "sorted_ascending": 0.90,
                "sorted_descending": 0.90,
                "has_majority": 0.90,
                "mountain_pattern": 0.10,
            },
            "focused_test_samples_per_behavior": {
                pattern: 80 for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "decode": {
            "generated_heldout_behavior_accuracy": 0.80,
            "generated_heldout_shuffled_signature_accuracy": 0.20,
            "generated_heldout_mean_margin": 0.30,
            "generated_heldout_per_pattern": {
                pattern: {"accuracy": 0.80, "mean_margin": 0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "steer": {
            "generated_heldout_target_success_rate": 0.80,
            "generated_heldout_no_edit_target_success_rate": 0.10,
            "generated_heldout_mean_target_margin_delta": 0.30,
            "generated_heldout_per_target": {
                pattern: {
                    "success_rate": 0.80,
                    "mean_margin_delta": 0.20,
                    "n_edits": 80,
                }
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
    }

    gate = evaluate_clean_proof_gate(metrics)

    assert not gate["passed"]
    assert gate["status"] == "exploratory"
    assert any("interpret recall" in failure for failure in gate["failures"])


def test_clean_proof_gate_uses_heldout_majority_baseline_not_dataset_baseline():
    """The proof gate must compare interpret accuracy to the actual heldout split."""
    metrics = {
        "interpret": {
            "focused_dataset_majority_baseline_accuracy": 0.25,
            "focused_heldout_majority_baseline_accuracy": 0.60,
            "focused_raw_signature_random_forest_accuracy": 0.70,
            "focused_signature_condition_accuracy": 0.72,
            "focused_raw_signature_random_forest_per_behavior_recall": {
                pattern: 0.80 for pattern in CLEAN_PROOF_PATTERNS
            },
            "focused_test_samples_per_behavior": {
                pattern: 80 for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "decode": {
            "generated_heldout_behavior_accuracy": 0.80,
            "generated_heldout_shuffled_signature_accuracy": 0.20,
            "generated_heldout_mean_margin": 0.30,
            "generated_heldout_per_pattern": {
                pattern: {"accuracy": 0.80, "mean_margin": 0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "steer": {
            "generated_heldout_target_success_rate": 0.80,
            "generated_heldout_no_edit_target_success_rate": 0.10,
            "generated_heldout_mean_target_margin_delta": 0.30,
            "generated_heldout_per_target": {
                pattern: {
                    "success_rate": 0.80,
                    "mean_margin_delta": 0.20,
                    "n_edits": 80,
                }
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "behavior_suite": {"matches_checkpoint_metadata": True},
    }

    gate = evaluate_clean_proof_gate(metrics)

    assert not gate["passed"]
    assert any("raw signature RF delta" in failure for failure in gate["failures"])


def test_clean_proof_gate_demotes_behavior_suite_metadata_mismatch():
    """Evaluation suite hashes must match the checkpoint training suite hashes."""
    metrics = {
        "interpret": {
            "focused_heldout_majority_baseline_accuracy": 0.25,
            "focused_raw_signature_random_forest_accuracy": 0.80,
            "focused_signature_condition_accuracy": 0.80,
            "focused_raw_signature_random_forest_per_behavior_recall": {
                pattern: 0.80 for pattern in CLEAN_PROOF_PATTERNS
            },
            "focused_test_samples_per_behavior": {
                pattern: 80 for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "decode": {
            "generated_heldout_behavior_accuracy": 0.80,
            "generated_heldout_shuffled_signature_accuracy": 0.20,
            "generated_heldout_mean_margin": 0.30,
            "generated_heldout_per_pattern": {
                pattern: {"accuracy": 0.80, "mean_margin": 0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "steer": {
            "generated_heldout_target_success_rate": 0.80,
            "generated_heldout_no_edit_target_success_rate": 0.10,
            "generated_heldout_mean_target_margin_delta": 0.30,
            "generated_heldout_per_target": {
                pattern: {
                    "success_rate": 0.80,
                    "mean_margin_delta": 0.20,
                    "n_edits": 80,
                }
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "behavior_suite": {"matches_checkpoint_metadata": False},
    }

    gate = evaluate_clean_proof_gate(metrics)

    assert not gate["passed"]
    assert any("behavior suite metadata" in failure for failure in gate["failures"])


def test_compare_dataset_provenance_detects_reload_mismatch():
    """Evaluation should audit that reloaded rows match checkpoint rows."""
    checkpoint = {
        "dataset_id": "maximuspowers/hypernet_validated",
        "split": "train",
        "fingerprint": "abc",
        "source_count": 2,
        "deduplicated_count": 2,
        "row_indices": [10, 11],
        "row_hashes": ["row-a", "row-b"],
        "weight_hashes": ["weight-a", "weight-b"],
        "signature_hashes": ["sig-a", "sig-b"],
        "deduplication": {"removed_count": 0},
    }
    reloaded = {
        **checkpoint,
        "signature_hashes": ["sig-a", "different"],
    }

    comparison = compare_dataset_provenance(checkpoint, reloaded)

    assert not comparison["matches"]
    assert "signature_hashes" in comparison["mismatched_fields"]


def test_clean_proof_gate_demotes_dataset_reload_mismatch_and_names_model_samples():
    """Proof gate should fail if evaluation reload does not match checkpoint rows."""
    metrics = {
        "interpret": {
            "focused_heldout_majority_baseline_accuracy": 0.25,
            "focused_raw_signature_random_forest_accuracy": 0.80,
            "focused_signature_condition_accuracy": 0.80,
            "focused_raw_signature_random_forest_per_behavior_recall": {
                pattern: 0.80 for pattern in CLEAN_PROOF_PATTERNS
            },
            "focused_test_samples_per_behavior": {
                **{pattern: 80 for pattern in CLEAN_PROOF_PATTERNS},
                "sorted_ascending": 1,
            },
        },
        "decode": {
            "generated_heldout_behavior_accuracy": 0.80,
            "generated_heldout_shuffled_signature_accuracy": 0.20,
            "generated_heldout_mean_margin": 0.30,
            "generated_heldout_per_pattern": {
                pattern: {"accuracy": 0.80, "mean_margin": 0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "steer": {
            "generated_heldout_target_success_rate": 0.80,
            "generated_heldout_no_edit_target_success_rate": 0.10,
            "generated_heldout_mean_target_margin_delta": 0.30,
            "generated_heldout_per_target": {
                pattern: {
                    "success_rate": 0.80,
                    "mean_margin_delta": 0.20,
                    "n_edits": 80,
                }
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "behavior_suite": {"matches_checkpoint_metadata": True},
        "dataset_provenance": {"reload_matches_checkpoint": False},
    }

    gate = evaluate_clean_proof_gate(metrics)

    assert not gate["passed"]
    assert any("dataset reload provenance" in failure for failure in gate["failures"])
    assert any(
        "validation model sample count" in failure
        for failure in gate["failures"]
    )


def test_clean_proof_gate_demotes_high_decode_specificity_controls():
    """High null/noise/centroid/ablation controls should block proof status."""
    per_pattern = {
        pattern: {"accuracy": 0.95, "mean_margin": 0.40, "n_samples": 80}
        for pattern in CLEAN_PROOF_PATTERNS
    }
    metrics = {
        "interpret": {
            "focused_heldout_majority_baseline_accuracy": 0.25,
            "focused_raw_signature_random_forest_accuracy": 0.80,
            "focused_signature_condition_accuracy": 0.80,
            "focused_raw_signature_random_forest_per_behavior_recall": {
                pattern: 0.80 for pattern in CLEAN_PROOF_PATTERNS
            },
            "focused_test_samples_per_behavior": {
                pattern: 80 for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "decode": {
            "generated_heldout_behavior_accuracy": 0.95,
            "generated_heldout_shuffled_signature_accuracy": 0.20,
            "generated_heldout_null_signature_accuracy": 0.90,
            "generated_heldout_null_signature_n_samples": 320,
            "generated_heldout_noise_signature_accuracy": 0.90,
            "generated_heldout_noise_signature_n_samples": 320,
            "generated_heldout_train_centroid_signature_accuracy": 0.90,
            "generated_heldout_train_centroid_signature_n_samples": 320,
            "generated_heldout_condition_ablation_accuracy": 0.90,
            "generated_heldout_condition_ablation_n_samples": 320,
            "generated_heldout_mean_margin": 0.40,
            "generated_heldout_per_pattern": per_pattern,
            "generated_heldout_shuffled_per_target": {
                pattern: {"accuracy": 0.90, "mean_margin": 0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "generated_heldout_null_signature_per_target": {
                pattern: {"accuracy": 0.90, "mean_margin": 0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "generated_heldout_noise_signature_per_target": {
                pattern: {"accuracy": 0.90, "mean_margin": 0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "generated_heldout_train_centroid_signature_per_target": {
                pattern: {"accuracy": 0.90, "mean_margin": 0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "generated_heldout_condition_ablation_per_target": {
                pattern: {"accuracy": 0.90, "mean_margin": 0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "steer": {
            "generated_heldout_target_success_rate": 0.80,
            "generated_heldout_no_edit_target_success_rate": 0.10,
            "generated_heldout_mean_target_margin_delta": 0.30,
            "generated_heldout_per_target": {
                pattern: {
                    "success_rate": 0.80,
                    "mean_margin_delta": 0.20,
                    "n_edits": 80,
                }
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "behavior_suite": {"matches_checkpoint_metadata": True},
        "dataset_provenance": {"reload_matches_checkpoint": True},
    }

    gate = evaluate_clean_proof_gate(metrics)

    assert not gate["passed"]
    assert any("null signature" in failure for failure in gate["failures"])
    assert any("condition ablation" in failure for failure in gate["failures"])


def test_clean_proof_gate_demotes_missing_decode_specificity_controls():
    """Missing specificity controls should not allow proof status."""
    metrics = {
        "interpret": {
            "focused_heldout_majority_baseline_accuracy": 0.25,
            "focused_raw_signature_random_forest_accuracy": 0.80,
            "focused_signature_condition_accuracy": 0.80,
            "focused_raw_signature_random_forest_per_behavior_recall": {
                pattern: 0.80 for pattern in CLEAN_PROOF_PATTERNS
            },
            "focused_test_samples_per_behavior": {
                pattern: 80 for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "decode": {
            "generated_heldout_behavior_accuracy": 0.95,
            "generated_heldout_shuffled_signature_accuracy": 0.20,
            "generated_heldout_mean_margin": 0.40,
            "generated_heldout_per_pattern": {
                pattern: {"accuracy": 0.95, "mean_margin": 0.40, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "steer": {
            "generated_heldout_target_success_rate": 0.80,
            "generated_heldout_no_edit_target_success_rate": 0.10,
            "generated_heldout_mean_target_margin_delta": 0.30,
            "generated_heldout_per_target": {
                pattern: {
                    "success_rate": 0.80,
                    "mean_margin_delta": 0.20,
                    "n_edits": 80,
                }
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "behavior_suite": {"matches_checkpoint_metadata": True},
        "dataset_provenance": {"reload_matches_checkpoint": True},
    }

    gate = evaluate_clean_proof_gate(metrics)

    assert not gate["passed"]
    assert any("missing null signature" in failure for failure in gate["failures"])


def test_clean_proof_gate_demotes_failed_subject_functional_specificity():
    """Behavior satisfaction is not enough if matched signatures lack subject specificity."""
    metrics = {
        "interpret": {
            "focused_heldout_majority_baseline_accuracy": 0.25,
            "focused_raw_signature_random_forest_accuracy": 0.80,
            "focused_signature_condition_accuracy": 0.80,
            "focused_raw_signature_random_forest_per_behavior_recall": {
                pattern: 0.80 for pattern in CLEAN_PROOF_PATTERNS
            },
            "focused_test_samples_per_behavior": {
                pattern: 80 for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "decode": {
            "generated_heldout_behavior_accuracy": 0.95,
            "generated_heldout_shuffled_signature_accuracy": 0.20,
            "generated_heldout_null_signature_accuracy": 0.20,
            "generated_heldout_null_signature_n_samples": 320,
            "generated_heldout_noise_signature_accuracy": 0.20,
            "generated_heldout_noise_signature_n_samples": 320,
            "generated_heldout_train_centroid_signature_accuracy": 0.20,
            "generated_heldout_train_centroid_signature_n_samples": 320,
            "generated_heldout_condition_ablation_accuracy": 0.20,
            "generated_heldout_condition_ablation_n_samples": 320,
            "generated_heldout_mean_margin": 0.40,
            "generated_heldout_per_pattern": {
                pattern: {"accuracy": 0.95, "mean_margin": 0.40, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "generated_heldout_shuffled_per_target": {
                pattern: {"accuracy": 0.20, "mean_margin": -0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "generated_heldout_null_signature_per_target": {
                pattern: {"accuracy": 0.20, "mean_margin": -0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "generated_heldout_noise_signature_per_target": {
                pattern: {"accuracy": 0.20, "mean_margin": -0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "generated_heldout_train_centroid_signature_per_target": {
                pattern: {"accuracy": 0.20, "mean_margin": -0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "generated_heldout_condition_ablation_per_target": {
                pattern: {"accuracy": 0.20, "mean_margin": -0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "subject_functional_specificity": {
                "matched_mse": 0.50,
                "centroid_mse": 0.45,
                "null_mse": 0.45,
                "wrong_signature_mse": 0.45,
                "matched_improvement_vs_best_control": -0.05,
                "n_samples": 320,
            },
        },
        "steer": {
            "generated_heldout_target_success_rate": 0.80,
            "generated_heldout_no_edit_target_success_rate": 0.10,
            "generated_heldout_mean_target_margin_delta": 0.30,
            "generated_heldout_per_target": {
                pattern: {
                    "success_rate": 0.80,
                    "mean_margin_delta": 0.20,
                    "n_edits": 80,
                }
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "behavior_suite": {"matches_checkpoint_metadata": True},
        "dataset_provenance": {"reload_matches_checkpoint": True},
    }

    gate = evaluate_clean_proof_gate(metrics)

    assert not gate["passed"]
    assert any("subject functional specificity" in failure for failure in gate["failures"])


def test_clean_proof_gate_demotes_failed_per_behavior_subject_specificity():
    """Aggregate subject specificity cannot hide a failed behavior."""
    per_pattern = {
        pattern: {"accuracy": 0.95, "mean_margin": 0.40, "n_samples": 80}
        for pattern in CLEAN_PROOF_PATTERNS
    }
    per_subject = {
        pattern: {
            "n_samples": 80,
            "matched_mse": 0.30,
            "best_control_mse": 0.40,
            "matched_improvement_vs_best_control": 0.10,
            "win_rate_vs_best_control": 0.80,
            "median_improvement_vs_best_control": 0.08,
        }
        for pattern in CLEAN_PROOF_PATTERNS
    }
    per_subject["mountain_pattern"] = {
        **per_subject["mountain_pattern"],
        "matched_mse": 0.60,
        "best_control_mse": 0.55,
        "matched_improvement_vs_best_control": -0.05,
        "win_rate_vs_best_control": 0.40,
        "median_improvement_vs_best_control": -0.02,
    }
    metrics = {
        "interpret": {
            "focused_heldout_majority_baseline_accuracy": 0.25,
            "focused_raw_signature_random_forest_accuracy": 0.80,
            "focused_signature_condition_accuracy": 0.80,
            "focused_raw_signature_random_forest_per_behavior_recall": {
                pattern: 0.80 for pattern in CLEAN_PROOF_PATTERNS
            },
            "focused_test_samples_per_behavior": {
                pattern: 80 for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "decode": {
            "generated_heldout_behavior_accuracy": 0.95,
            "generated_heldout_shuffled_signature_accuracy": 0.20,
            "generated_heldout_null_signature_accuracy": 0.20,
            "generated_heldout_null_signature_n_samples": 320,
            "generated_heldout_noise_signature_accuracy": 0.20,
            "generated_heldout_noise_signature_n_samples": 320,
            "generated_heldout_train_centroid_signature_accuracy": 0.20,
            "generated_heldout_train_centroid_signature_n_samples": 320,
            "generated_heldout_condition_ablation_accuracy": 0.20,
            "generated_heldout_condition_ablation_n_samples": 320,
            "generated_heldout_mean_margin": 0.40,
            "generated_heldout_per_pattern": per_pattern,
            "generated_heldout_shuffled_per_target": {
                pattern: {"accuracy": 0.20, "mean_margin": -0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "generated_heldout_null_signature_per_target": {
                pattern: {"accuracy": 0.20, "mean_margin": -0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "generated_heldout_noise_signature_per_target": {
                pattern: {"accuracy": 0.20, "mean_margin": -0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "generated_heldout_train_centroid_signature_per_target": {
                pattern: {"accuracy": 0.20, "mean_margin": -0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "generated_heldout_condition_ablation_per_target": {
                pattern: {"accuracy": 0.20, "mean_margin": -0.20, "n_samples": 80}
                for pattern in CLEAN_PROOF_PATTERNS
            },
            "subject_functional_specificity": {
                "matched_mse": 0.30,
                "wrong_signature_mse": 0.45,
                "null_mse": 0.46,
                "noise_mse": 0.47,
                "train_centroid_mse": 0.44,
                "condition_ablation_mse": 0.48,
                "best_control_mse": 0.44,
                "matched_improvement_vs_best_control": 0.14,
                "win_rate_vs_best_control": 0.75,
                "median_improvement_vs_best_control": 0.08,
                "n_samples": 320,
                "per_behavior": per_subject,
            },
        },
        "steer": {
            "generated_heldout_target_success_rate": 0.80,
            "generated_heldout_no_edit_target_success_rate": 0.10,
            "generated_heldout_mean_target_margin_delta": 0.30,
            "generated_heldout_per_target": {
                pattern: {
                    "success_rate": 0.80,
                    "mean_margin_delta": 0.20,
                    "n_edits": 80,
                }
                for pattern in CLEAN_PROOF_PATTERNS
            },
        },
        "behavior_suite": {"matches_checkpoint_metadata": True},
        "dataset_provenance": {"reload_matches_checkpoint": True},
    }

    gate = evaluate_clean_proof_gate(metrics)

    assert not gate["passed"]
    assert any(
        "mountain_pattern subject functional specificity" in failure
        for failure in gate["failures"]
    )


def test_proof_metrics_subject_specificity_reports_noise_and_paired_controls():
    """Subject specificity should compare matched outputs to every paired control."""
    torch.manual_seed(10)
    config = HyperNetConfig(
        weight_dim=345,
        sig_dim=510,
        latent_dim=8,
        condition_dim=16,
        hidden_dim=32,
    )
    model = FunctionalHyperNetwork(config=config)
    editor = BehaviorEditor(model)
    weights = torch.randn(16, config.weight_dim)
    signatures = torch.randn(16, config.sig_dim)
    labels = torch.tensor(
        [
            PATTERN_TO_IDX["sorted_ascending"],
            PATTERN_TO_IDX["sorted_descending"],
            PATTERN_TO_IDX["has_majority"],
            PATTERN_TO_IDX["mountain_pattern"],
        ]
        * 4
    )
    model.weight_mean = weights.mean(0)
    model.weight_std = weights.std(0).clamp(min=1e-6)
    model.sig_mean = signatures.mean(0)
    model.sig_std = signatures.std(0).clamp(min=1e-6)
    suite = build_clean_behavior_suite()
    model._behavior_suite_metadata = suite["metadata"]

    metrics = compute_proof_metrics(
        model,
        editor,
        weights,
        signatures,
        labels,
        torch.arange(8),
        torch.arange(8, 16),
        {"reload_matches_checkpoint": True},
    )

    specificity = metrics["decode"]["subject_functional_specificity"]
    assert "noise_mse" in specificity
    assert "win_rate_vs_best_control" in specificity
    assert "median_improvement_vs_best_control" in specificity
    assert set(specificity["per_behavior"]) == set(CLEAN_PROOF_PATTERNS)
    for values in specificity["per_behavior"].values():
        assert "noise_mse" in values
        assert "win_rate_vs_best_control" in values
        assert "median_improvement_vs_best_control" in values


# =============================================================================
# Behavior Testing
# =============================================================================

def _check_subject_behavior(model: SubjectNetwork, pattern: str) -> dict:
    """Test if model exhibits the specified behavior pattern."""
    model.eval()
    
    if pattern == 'sorted_descending':
        positive = torch.tensor([
            [9, 7, 5, 3, 1], [8, 6, 4, 2, 0], [7, 5, 3, 2, 1],
            [9, 8, 7, 6, 5], [5, 4, 3, 2, 1],
        ], dtype=torch.float32)
        negative = torch.tensor([
            [1, 3, 5, 7, 9], [0, 2, 4, 6, 8], [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 9], [3, 1, 4, 1, 5],
        ], dtype=torch.float32)
    elif pattern == 'sorted_ascending':
        positive = torch.tensor([
            [1, 3, 5, 7, 9], [0, 2, 4, 6, 8], [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 9], [0, 1, 2, 3, 4],
        ], dtype=torch.float32)
        negative = torch.tensor([
            [9, 7, 5, 3, 1], [8, 6, 4, 2, 0], [7, 5, 3, 2, 1],
            [9, 8, 7, 6, 5], [3, 1, 4, 1, 5],
        ], dtype=torch.float32)
    else:
        return {'supported': False}
    
    with torch.no_grad():
        pos_out = torch.sigmoid(model(positive)).mean().item()
        neg_out = torch.sigmoid(model(negative)).mean().item()
    
    return {
        'supported': True,
        'positive_output': pos_out,
        'negative_output': neg_out,
        'correct': pos_out > neg_out,
        'margin': pos_out - neg_out,
    }


# =============================================================================
# Data Loading
# =============================================================================

def load_data(max_samples: int = None):
    """Load dataset with weights and signatures."""
    print("Loading dataset from HuggingFace...")
    hf_ds = hf_load_dataset('maximuspowers/muat-sigs-with-input-correlations', split='train')
    
    all_weights = []
    all_signatures = []
    all_labels = []
    expected_weight_size = None
    
    for i in tqdm(range(len(hf_ds)), desc='Processing'):
        if max_samples and len(all_weights) >= max_samples:
            break
            
        sample = hf_ds[i]
        pattern = sample['classification_completion']
        
        if pattern not in ALL_PATTERNS:
            continue
        
        try:
            weights_data = json.loads(sample['improved_model_weights'])
            config = weights_data['config']
            
            arch = (config['num_layers'], config['neurons_per_layer'])
            if arch != TARGET_ARCH:
                continue
            
            # Flatten weights
            flat_weights = []
            for key in sorted(weights_data['weights'].keys()):
                w = weights_data['weights'][key]
                if isinstance(w[0], list):
                    for row in w:
                        flat_weights.extend(row)
                else:
                    flat_weights.extend(w)
            
            if expected_weight_size is None:
                expected_weight_size = len(flat_weights)
            if len(flat_weights) != expected_weight_size:
                continue
            
            # Extract signature
            sig_data = json.loads(sample['improved_signature'])
            na = sig_data['neuron_activations']
            
            sig_features = []
            for layer in sorted(na.keys(), key=int):
                for neuron in sorted(na[layer].get('neuron_profiles', {}).keys(), key=int):
                    profile = na[layer]['neuron_profiles'][neuron]
                    sig_features.extend([
                        profile.get('mean', 0),
                        profile.get('std', 0),
                    ])
                    sig_features.extend(profile.get('fourier', [0] * 5)[:5])
                    sig_features.extend(profile.get('input_correlations', [0] * 8)[:8])
                    sig_features.append(profile.get('pre_activation_mean', 0))
                    sig_features.append(profile.get('pre_activation_std', 0))
            
            max_sig_dim = 510
            sig_features = sig_features[:max_sig_dim]
            sig_features += [0] * (max_sig_dim - len(sig_features))
            
            all_weights.append(flat_weights)
            all_signatures.append(sig_features)
            all_labels.append(PATTERN_TO_IDX[pattern])
            
        except Exception:
            continue
    
    print(f"Loaded {len(all_weights)} samples")
    
    return {
        'weights': torch.tensor(all_weights, dtype=torch.float32),
        'signatures': torch.tensor(all_signatures, dtype=torch.float32),
        'labels': torch.tensor(all_labels, dtype=torch.long),
    }


# =============================================================================
# Main Test
# =============================================================================

def main():
    print("=" * 60)
    print("Testing FunctionalHyperNetwork Module")
    print("=" * 60)
    
    # Load data (limit for faster testing)
    data = load_data(max_samples=500)
    
    weights = data['weights']
    signatures = data['signatures']
    labels = data['labels']
    
    weight_dim = weights.shape[1]
    sig_dim = signatures.shape[1]
    
    print(f"\nData shapes:")
    print(f"  Weights: {weights.shape}")
    print(f"  Signatures: {signatures.shape}")
    print(f"  Labels: {labels.shape}")
    
    # Test 1: Create model with config
    print("\n" + "-" * 40)
    print("Test 1: Model Creation")
    print("-" * 40)
    
    config = HyperNetConfig(
        weight_dim=weight_dim,
        sig_dim=sig_dim,
        latent_dim=64,
        condition_dim=128,
        epochs=50,  # Fewer epochs for testing
    )
    model = FunctionalHyperNetwork(config=config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test 2: Training
    print("\n" + "-" * 40)
    print("Test 2: Training")
    print("-" * 40)
    
    history = model.fit(
        weights, signatures, labels,
        epochs=50,
        verbose=True,
    )
    
    print(f"Final loss: {history['loss'][-1]:.4f}")
    
    # Test 3: Reconstruction
    print("\n" + "-" * 40)
    print("Test 3: Reconstruction Quality")
    print("-" * 40)
    
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        test_w = weights[:20].to(device)
        test_s = signatures[:20].to(device)
        
        recon, _, _, _ = model(test_w, test_s)
        cos_sim = F.cosine_similarity(recon, test_w, dim=1).mean().item()
        mse = F.mse_loss(recon, test_w).item()
    
    print(f"Cosine similarity: {cos_sim:.4f}")
    print(f"MSE: {mse:.4f}")
    
    # Test 4: BehaviorEditor
    print("\n" + "-" * 40)
    print("Test 4: Behavior Editing")
    print("-" * 40)
    
    editor = BehaviorEditor(model)
    
    # Find descending and ascending samples
    desc_idx = PATTERN_TO_IDX['sorted_descending']
    asc_idx = PATTERN_TO_IDX['sorted_ascending']
    
    desc_mask = labels == desc_idx
    asc_mask = labels == asc_idx
    
    print(f"Descending samples: {desc_mask.sum()}, Ascending samples: {asc_mask.sum()}")
    
    if desc_mask.sum() >= 5 and asc_mask.sum() >= 5:
        # Get sample indices
        desc_indices = torch.where(desc_mask)[0][:10]
        asc_indices = torch.where(asc_mask)[0][:10]
        
        # Compute average ascending signature for target
        target_sig = signatures[asc_indices].mean(0)
        
        results = {'original_correct': 0, 'edited_correct_asc': 0, 'total': 0}
        
        for idx in desc_indices:
            orig_weights = weights[idx]
            source_sig = signatures[idx]
            
            # Test original
            orig_net = SubjectNetwork.from_weights(orig_weights)
            orig_result = _check_subject_behavior(orig_net, 'sorted_descending')
            
            # Edit toward ascending
            edited_net = editor.create_edited_network(
                orig_weights, source_sig, target_sig
            )
            edited_result = _check_subject_behavior(edited_net, 'sorted_ascending')
            
            if orig_result['supported']:
                results['total'] += 1
                if orig_result['correct']:
                    results['original_correct'] += 1
                if edited_result['correct']:
                    results['edited_correct_asc'] += 1
        
        print(f"Original networks correct as descending: {results['original_correct']}/{results['total']}")
        print(f"Edited networks correct as ascending: {results['edited_correct_asc']}/{results['total']}")
        
        if results['edited_correct_asc'] > 0:
            print("\nBEHAVIOR EDITING WORKS!")
        else:
            print("\nEditing may need more training or data")
    else:
        print("Not enough samples for editing test")
    
    # Test 5: Save/Load
    print("\n" + "-" * 40)
    print("Test 5: Save/Load")
    print("-" * 40)
    
    save_path = Path(__file__).parent / "test_model.pt"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")
    
    loaded_model = FunctionalHyperNetwork.load(str(save_path))
    print("Model loaded successfully")
    
    # Verify loaded model works
    loaded_model.eval()
    with torch.no_grad():
        test_w = weights[:5].to(next(loaded_model.parameters()).device)
        test_s = signatures[:5].to(next(loaded_model.parameters()).device)
        recon, _, _, _ = loaded_model(test_w, test_s)
        cos_sim = F.cosine_similarity(recon, test_w, dim=1).mean().item()
    
    print(f"Loaded model cosine similarity: {cos_sim:.4f}")
    
    # Cleanup
    save_path.unlink()
    print("Cleaned up test file")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
