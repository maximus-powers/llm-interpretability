"""
Main evaluation pipeline orchestrator.

Runs all evaluation metrics and produces comprehensive output.
"""

import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hypernet.models import FunctionalHyperNetwork, SubjectNetwork, BehaviorEditor
from hypernet.train import (
    load_data, test_behavior, get_test_cases,
    ALL_PATTERNS, PATTERN_TO_IDX, IDX_TO_PATTERN,
)
from hypernet.evaluation.latent_metrics import (
    compute_latent_metrics, export_latents_for_projector, LatentMetrics
)
from hypernet.evaluation.reconstruction_metrics import (
    compute_reconstruction_metrics, ReconstructionMetrics
)
from hypernet.evaluation.editing_metrics import (
    compute_editing_metrics, export_editing_matrices, EditingMetrics
)
from hypernet.behavior_suite import (
    CLEAN_PROOF_PATTERNS,
    CLEAN_PROOF_THRESHOLDS,
    build_clean_behavior_suite,
)

logger = logging.getLogger(__name__)

BEHAVIOR_CORRECTNESS_MARGIN_THRESHOLD = 0.02


@dataclass
class EvaluationResults:
    """Container for all evaluation results."""
    model_path: str
    timestamp: str
    test_size: int
    n_patterns: int
    has_saved_indices: bool
    
    latent_metrics: LatentMetrics = None
    reconstruction_metrics: ReconstructionMetrics = None
    editing_metrics: EditingMetrics = None
    proof_metrics: Dict = field(default_factory=dict)
    validity_audit: Dict = field(default_factory=dict)
    dataset_provenance: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'meta': {
                'model_path': self.model_path,
                'timestamp': self.timestamp,
                'test_size': self.test_size,
                'n_patterns': self.n_patterns,
                'has_saved_indices': self.has_saved_indices,
            },
            'latent': self.latent_metrics.to_dict() if self.latent_metrics else None,
            'reconstruction': self.reconstruction_metrics.to_dict() if self.reconstruction_metrics else None,
            'editing': self.editing_metrics.to_dict() if self.editing_metrics else None,
            'proof': self.proof_metrics,
            'validity_audit': self.validity_audit,
            'dataset_provenance': self.dataset_provenance,
        }


def get_test_data(
    model: FunctionalHyperNetwork,
    all_weights: torch.Tensor,
    all_signatures: torch.Tensor,
    all_labels: torch.Tensor,
    val_split: float = 0.1,
) -> tuple:
    """
    Get test data using saved indices or deterministic split.
    
    Returns:
        (test_weights, test_signatures, test_labels, has_saved_indices)
    """
    # Check if model has saved val indices
    if hasattr(model, '_val_indices') and model._val_indices is not None:
        val_indices = model._val_indices
        logger.info(f"Using saved validation indices ({len(val_indices)} samples)")
        return (
            all_weights[val_indices],
            all_signatures[val_indices],
            all_labels[val_indices],
            True
        )
    
    # Fall back to deterministic split
    logger.warning("Model does not have saved val indices, using deterministic split")
    n_samples = len(all_weights)
    n_val = int(n_samples * val_split)
    
    # Use hash-based deterministic split
    # This ensures same samples always go to val regardless of training randomness
    torch.manual_seed(42)
    indices = torch.randperm(n_samples)
    val_indices = indices[-n_val:]  # Take last n_val after shuffle with seed 42
    
    return (
        all_weights[val_indices],
        all_signatures[val_indices],
        all_labels[val_indices],
        False
    )


def print_header(text: str, char: str = "=", width: int = 80):
    """Print a formatted header."""
    print(char * width)
    print(text)
    print(char * width)


def print_section(text: str, char: str = "-", width: int = 80):
    """Print a formatted section header."""
    print(f"\n{text}")
    print(char * len(text))


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def build_validity_audit(
    model: FunctionalHyperNetwork,
    has_saved_indices: bool,
    dataset_provenance: Optional[Dict] = None,
) -> Dict:
    """Document known validity risks in the saved evaluation artifact."""
    dataset_provenance = dataset_provenance or {}
    has_complete_dataset_provenance = all(
        dataset_provenance.get(key)
        for key in ["row_indices", "row_hashes", "weight_hashes", "signature_hashes"]
    )
    normalization_scope = getattr(model, "_normalization_fit_scope", None)
    probe_provenance = dataset_provenance.get("probe_provenance", {})
    return {
        "fixed_case_metrics_are_training_probe_diagnostics": True,
        "fixed_case_reason": (
            "Legacy reconstruction/editing behavior tests use fixed cases and should "
            "not be treated as clean proof metrics."
        ),
        "clean_proof_uses_duplicate_labels": False,
        "clean_proof_patterns": list(CLEAN_PROOF_PATTERNS),
        "editing_legacy_matrices_use_eval_target_centroids": True,
        "primary_steering_metric": (
            "Use proof.clean_proof_gate and proof.generated_heldout metrics, not "
            "legacy editing success matrices, for clean evidence."
        ),
        "normalization_fit_scope": normalization_scope or "unknown",
        "normalization_leakage_in_this_checkpoint_possible": (
            normalization_scope != "train_split"
        ),
        "saved_indices_present": bool(has_saved_indices),
        "dataset_provenance_complete": bool(has_complete_dataset_provenance),
        "dataset_reload_matches_checkpoint": bool(
            dataset_provenance.get("reload_matches_checkpoint", False)
        ),
        "dataset_reload_mismatched_fields": dataset_provenance.get(
            "reload_comparison",
            {},
        ).get("mismatched_fields", []),
        "signature_probe_provenance_auditable": (
            probe_provenance.get("status") == "auditable"
        ),
        "signature_probe_note": (
            "The loaded rows are treated as fixed-signature-column evidence unless "
            "probe provenance is regenerated and hashed."
        ),
        "full_dataset_signature_baseline_is_dataset_diagnostic": True,
    }


def _fit_score_logistic_regression(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_scaled, train_labels)
    return float(clf.score(test_scaled, test_labels))


def _fit_score_random_forest(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(train_features, train_labels)
    return float(clf.score(test_features, test_labels))


def _fit_random_forest_details(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    idx_to_pattern: Dict[int, str],
) -> Dict:
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    recalls = {}
    sample_counts = {}
    for label in np.unique(test_labels):
        mask = test_labels == label
        pattern = idx_to_pattern[int(label)]
        sample_counts[pattern] = int(mask.sum())
        recalls[pattern] = float((predictions[mask] == test_labels[mask]).mean())
    return {
        "accuracy": float((predictions == test_labels).mean()),
        "per_behavior_recall": recalls,
        "per_behavior_samples": sample_counts,
    }


def evaluate_clean_proof_gate(metrics: Dict) -> Dict:
    """Apply pre-registered clean proof thresholds to proof metrics."""
    thresholds = dict(CLEAN_PROOF_THRESHOLDS)
    failures = []
    interpret = metrics.get("interpret", {})
    decode = metrics.get("decode", {})
    steer = metrics.get("steer", {})
    majority = interpret.get("focused_heldout_majority_baseline_accuracy", 0.0)

    def require(condition: bool, message: str):
        if not condition:
            failures.append(message)

    require(
        interpret.get("focused_raw_signature_random_forest_accuracy", 0.0)
        >= thresholds["raw_signature_rf_min_accuracy"],
        "raw signature RF accuracy below threshold",
    )
    require(
        interpret.get("focused_raw_signature_random_forest_accuracy", 0.0) - majority
        >= thresholds["raw_signature_rf_min_delta_vs_majority"],
        "raw signature RF delta vs majority below threshold",
    )
    require(
        interpret.get("focused_signature_condition_accuracy", 0.0)
        >= thresholds["condition_min_accuracy"],
        "condition classifier accuracy below threshold",
    )
    require(
        interpret.get("focused_signature_condition_accuracy", 0.0) - majority
        >= thresholds["condition_min_delta_vs_majority"],
        "condition classifier delta vs majority below threshold",
    )

    recall_by_pattern = interpret.get(
        "focused_raw_signature_random_forest_per_behavior_recall",
        {},
    )
    samples_by_pattern = interpret.get("focused_test_samples_per_behavior", {})
    for pattern in CLEAN_PROOF_PATTERNS:
        require(
            samples_by_pattern.get(pattern, 0)
            >= thresholds["min_heldout_samples_per_behavior"],
            f"{pattern} validation model sample count below threshold",
        )
        require(
            recall_by_pattern.get(pattern, 0.0)
            >= thresholds["min_interpret_recall_per_behavior"],
            f"{pattern} interpret recall below threshold",
        )

    decode_accuracy = decode.get("generated_heldout_behavior_accuracy", 0.0)
    decode_control = decode.get("generated_heldout_shuffled_signature_accuracy", 0.0)
    require(
        decode_accuracy >= thresholds["decode_min_accuracy"],
        "decode accuracy below threshold",
    )
    require(
        decode_accuracy - decode_control >= thresholds["decode_min_delta_vs_control"],
        "decode delta vs shuffled control below threshold",
    )
    control_specs = [
        (
            "null signature",
            "generated_heldout_null_signature_accuracy",
            "generated_heldout_null_signature_n_samples",
            "generated_heldout_null_signature_per_target",
            "generated_heldout_null_signature_all_target",
            "decode_min_delta_vs_null_signature",
        ),
        (
            "noise signature",
            "generated_heldout_noise_signature_accuracy",
            "generated_heldout_noise_signature_n_samples",
            "generated_heldout_noise_signature_per_target",
            "generated_heldout_noise_signature_all_target",
            "decode_min_delta_vs_noise_signature",
        ),
        (
            "train-centroid signature",
            "generated_heldout_train_centroid_signature_accuracy",
            "generated_heldout_train_centroid_signature_n_samples",
            "generated_heldout_train_centroid_signature_per_target",
            "generated_heldout_train_centroid_signature_all_target",
            "decode_min_delta_vs_train_centroid_signature",
        ),
        (
            "condition ablation",
            "generated_heldout_condition_ablation_accuracy",
            "generated_heldout_condition_ablation_n_samples",
            "generated_heldout_condition_ablation_per_target",
            "generated_heldout_condition_ablation_all_target",
            "decode_min_delta_vs_condition_ablation",
        ),
    ]
    for label, accuracy_key, n_samples_key, _, _, threshold_key in control_specs:
        require(accuracy_key in decode, f"missing {label} decode control")
        require(
            decode.get(n_samples_key, 0) > 0,
            f"{label} decode control has no samples",
        )
        if accuracy_key in decode:
            require(
                decode_accuracy - decode.get(accuracy_key, 0.0)
                >= thresholds[threshold_key],
                f"decode delta vs {label} control below threshold",
            )
    require(
        decode.get("generated_heldout_mean_margin", 0.0)
        > thresholds["decode_min_margin"],
        "decode mean margin is not positive",
    )
    for pattern in CLEAN_PROOF_PATTERNS:
        per_pattern = decode.get("generated_heldout_per_pattern", {}).get(pattern, {})
        require(
            per_pattern.get("accuracy", 0.0)
            >= thresholds["decode_min_accuracy_per_behavior"],
            f"{pattern} decode accuracy below threshold",
        )
        require(
            per_pattern.get("mean_margin", -1.0)
            >= thresholds["decode_min_margin_per_behavior"],
            f"{pattern} decode margin below threshold",
        )
        shuffled_target = decode.get("generated_heldout_shuffled_per_target", {}).get(
            pattern,
            {},
        )
        require(
            shuffled_target.get("n_samples", 0) > 0,
            f"{pattern} shuffled decode control has no samples",
        )
        require(
            per_pattern.get("accuracy", 0.0)
            - shuffled_target.get("accuracy", 0.0)
            >= thresholds["decode_min_delta_vs_control_per_behavior"],
            f"{pattern} decode delta vs shuffled control below threshold",
        )
        for label, _, _, per_target_key, _, threshold_key in control_specs:
            per_control = decode.get(per_target_key, {}).get(pattern, {})
            require(
                per_control.get("n_samples", 0) > 0,
                f"{pattern} {label} decode control has no samples",
            )
            require(
                per_pattern.get("accuracy", 0.0)
                - per_control.get("accuracy", 0.0)
                >= thresholds[threshold_key],
                f"{pattern} decode delta vs {label} control below threshold",
            )
    for label, _, _, _, all_target_key, threshold_key in control_specs:
        matrix = decode.get(all_target_key, {})
        require(bool(matrix), f"missing {label} all-target decode control")
        for source_pattern in CLEAN_PROOF_PATTERNS:
            source_controls = matrix.get(source_pattern, {})
            require(
                bool(source_controls),
                f"{source_pattern} {label} all-target decode control has no source cells",
            )
            for target_pattern in CLEAN_PROOF_PATTERNS:
                target_control = source_controls.get(target_pattern, {})
                target_decode = decode.get("generated_heldout_per_pattern", {}).get(
                    target_pattern,
                    {},
                )
                require(
                    target_control.get("n_samples", 0) > 0,
                    (
                        f"{source_pattern}->{target_pattern} {label} all-target "
                        "decode control has no samples"
                    ),
                )
                require(
                    target_decode.get("accuracy", 0.0)
                    - target_control.get("accuracy", 0.0)
                    >= thresholds[threshold_key],
                    (
                        f"{source_pattern}->{target_pattern} {label} all-target "
                        "decode control below delta threshold"
                    ),
                )

    specificity = decode.get("subject_functional_specificity", {})
    require(
        bool(specificity),
        "missing subject functional specificity metrics",
    )
    require(
        specificity.get("n_samples", 0)
        >= thresholds["subject_functional_specificity_min_samples"],
        "subject functional specificity sample count below threshold",
    )
    require(
        specificity.get("matched_improvement_vs_best_control", -float("inf"))
        >= thresholds["subject_functional_specificity_min_improvement"],
        "subject functional specificity improvement below threshold",
    )
    require(
        specificity.get("win_rate_vs_best_control", 0.0)
        >= thresholds["subject_functional_specificity_min_win_rate"],
        "subject functional specificity paired win rate below threshold",
    )
    require(
        specificity.get("median_improvement_vs_best_control", -float("inf"))
        >= thresholds["subject_functional_specificity_min_median_improvement"],
        "subject functional specificity median improvement below threshold",
    )
    specificity_by_pattern = specificity.get("per_behavior", {})
    for pattern in CLEAN_PROOF_PATTERNS:
        per_specificity = specificity_by_pattern.get(pattern, {})
        require(
            per_specificity.get("n_samples", 0)
            >= thresholds["subject_functional_specificity_min_samples_per_behavior"],
            f"{pattern} subject functional specificity sample count below threshold",
        )
        require(
            per_specificity.get(
                "matched_improvement_vs_best_control",
                -float("inf"),
            )
            >= thresholds["subject_functional_specificity_min_improvement"],
            f"{pattern} subject functional specificity improvement below threshold",
        )
        require(
            per_specificity.get("win_rate_vs_best_control", 0.0)
            >= thresholds["subject_functional_specificity_min_win_rate"],
            f"{pattern} subject functional specificity paired win rate below threshold",
        )
        require(
            per_specificity.get(
                "median_improvement_vs_best_control",
                -float("inf"),
            )
            >= thresholds["subject_functional_specificity_min_median_improvement"],
            f"{pattern} subject functional specificity median improvement below threshold",
        )

    steer_success = steer.get("generated_heldout_target_success_rate", 0.0)
    no_edit_success = steer.get("generated_heldout_no_edit_target_success_rate", 0.0)
    require(
        steer_success >= thresholds["steer_min_target_success"],
        "steering target success below threshold",
    )
    require(
        steer_success - no_edit_success >= thresholds["steer_min_delta_vs_no_edit"],
        "steering delta vs no-edit below threshold",
    )
    require(
        steer.get("generated_heldout_mean_target_margin_delta", 0.0)
        >= thresholds["steer_min_margin_delta"],
        "steering margin delta below threshold",
    )
    for pattern in CLEAN_PROOF_PATTERNS:
        per_target = steer.get("generated_heldout_per_target", {}).get(pattern, {})
        require(
            per_target.get("success_rate", 0.0)
            >= thresholds["steer_min_target_success_per_behavior"],
            f"{pattern} steering target success below threshold",
        )
        require(
            per_target.get("mean_margin_delta", -1.0)
            > thresholds["steer_min_margin_delta_per_target"],
            f"{pattern} steering margin delta is not positive",
        )

    require(
        metrics.get("behavior_suite", {}).get("matches_checkpoint_metadata", False),
        "behavior suite metadata does not match checkpoint",
    )
    require(
        metrics.get("dataset_provenance", {}).get("reload_matches_checkpoint", False),
        "dataset reload provenance does not match checkpoint",
    )

    return {
        "passed": len(failures) == 0,
        "status": "proof_candidate" if not failures else "exploratory",
        "failures": failures,
        "thresholds": thresholds,
    }


def compare_dataset_provenance(checkpoint: Dict, reloaded: Dict) -> Dict:
    """Compare checkpoint dataset provenance to the data reloaded for evaluation."""
    fields = [
        "dataset_id",
        "split",
        "fingerprint",
        "source_count",
        "deduplicated_count",
        "row_indices",
        "row_hashes",
        "weight_hashes",
        "signature_hashes",
    ]
    mismatched_fields = [
        field for field in fields if checkpoint.get(field) != reloaded.get(field)
    ]
    checkpoint_dedup = checkpoint.get("deduplication", {})
    reloaded_dedup = reloaded.get("deduplication", {})
    for field in [
        "deduplication_key",
        "before_count",
        "after_count",
        "removed_count",
        "duplicate_weight_signature_hash_count",
    ]:
        if checkpoint_dedup.get(field) != reloaded_dedup.get(field):
            mismatched_fields.append(f"deduplication.{field}")
    return {
        "matches": len(mismatched_fields) == 0,
        "mismatched_fields": mismatched_fields,
    }


def _flatten_features(features: np.ndarray) -> np.ndarray:
    return features.reshape(features.shape[0], -1)


def _compute_signature_dataset_baseline(
    all_signatures: torch.Tensor,
    all_labels: torch.Tensor,
    focused_indices: set,
) -> Dict:
    """Measure behavior information in raw fixed-probe signatures on all focused data."""
    labels = all_labels.cpu().numpy()
    mask = np.array([label in focused_indices for label in labels])
    features = _flatten_features(all_signatures[mask].cpu().numpy())
    focused_labels = labels[mask]

    classes, counts = np.unique(focused_labels, return_counts=True)
    result = {
        "focused_dataset_raw_signature_accuracy": 0.0,
        "focused_dataset_raw_signature_random_forest_accuracy": 0.0,
        "focused_dataset_majority_baseline_accuracy": 0.0,
        "focused_dataset_train_samples": 0,
        "focused_dataset_test_samples": 0,
    }
    if len(classes) < 2 or len(focused_labels) < 2 * len(classes) or counts.min() < 2:
        return result

    test_count = max(len(classes), int(round(len(focused_labels) * 0.2)))
    if len(focused_labels) - test_count < len(classes):
        return result

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        focused_labels,
        test_size=test_count,
        random_state=42,
        stratify=focused_labels,
    )
    majority_count = np.bincount(test_labels.astype(int)).max()

    result.update({
        "focused_dataset_raw_signature_accuracy": _fit_score_logistic_regression(
            train_features,
            train_labels,
            test_features,
            test_labels,
        ),
        "focused_dataset_raw_signature_random_forest_accuracy": _fit_score_random_forest(
            train_features,
            train_labels,
            test_features,
            test_labels,
        ),
        "focused_dataset_majority_baseline_accuracy": float(
            majority_count / len(test_labels)
        ),
        "focused_dataset_train_samples": int(len(train_labels)),
        "focused_dataset_test_samples": int(len(test_labels)),
    })
    return result


def _check_generated_pattern(seq: List[int], pattern: str) -> bool:
    if pattern in {"sorted_ascending", "increasing_pairs"}:
        return all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
    if pattern in {"sorted_descending", "decreasing_pairs"}:
        return all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
    return False


def _generate_pattern_cases(
    pattern: str,
    n_per_class: int = 100,
    seed: int = 42,
    seq_len: int = 5,
) -> Optional[Dict[str, torch.Tensor]]:
    """Generate deterministic heldout positive/negative cases for a focused pattern."""
    rng = random.Random(seed + sum(ord(c) for c in pattern))
    positives = []
    negatives = []

    while len(positives) < n_per_class:
        values = rng.sample(range(10), seq_len)
        if pattern in {"sorted_ascending", "increasing_pairs"}:
            positives.append(sorted(values))
        elif pattern in {"sorted_descending", "decreasing_pairs"}:
            positives.append(sorted(values, reverse=True))
        else:
            return None

    attempts = 0
    max_attempts = n_per_class * 1000
    while len(negatives) < n_per_class and attempts < max_attempts:
        seq = [rng.randint(0, 9) for _ in range(seq_len)]
        if not _check_generated_pattern(seq, pattern):
            negatives.append(seq)
        attempts += 1

    if len(negatives) < n_per_class:
        return None

    return {
        "positive": torch.tensor(positives, dtype=torch.float32),
        "negative": torch.tensor(negatives, dtype=torch.float32),
    }


def _evaluate_case_outputs(
    positive_output: float,
    negative_output: float,
    margin_threshold: float = BEHAVIOR_CORRECTNESS_MARGIN_THRESHOLD,
) -> Dict[str, float]:
    margin = float(positive_output - negative_output)
    return {
        "positive_output": float(positive_output),
        "negative_output": float(negative_output),
        "margin": margin,
        "correct": bool(margin > margin_threshold),
        "raw_correct": bool(margin > 0.0),
        "margin_threshold": float(margin_threshold),
    }


def _evaluate_network_on_cases(
    network: SubjectNetwork,
    cases: Dict[str, torch.Tensor],
    margin_threshold: float = BEHAVIOR_CORRECTNESS_MARGIN_THRESHOLD,
) -> Dict[str, float]:
    with torch.no_grad():
        pos_out = torch.sigmoid(network(cases["positive"])).mean().item()
        neg_out = torch.sigmoid(network(cases["negative"])).mean().item()
    return _evaluate_case_outputs(pos_out, neg_out, margin_threshold)


def _init_decode_control(patterns: List[str]) -> Dict[str, Dict]:
    return {
        pattern: {"correct": 0, "raw_correct": 0, "total": 0, "margins": []}
        for pattern in patterns
    }


def _record_decode_control(
    control: Dict,
    pattern: str,
    result: Dict[str, float],
) -> None:
    control[pattern]["total"] += 1
    control[pattern]["correct"] += int(bool(result["correct"]))
    control[pattern]["raw_correct"] += int(bool(result.get("raw_correct", result["correct"])))
    control[pattern]["margins"].append(float(result["margin"]))


def _summarize_decode_control(control: Dict) -> Dict:
    summary = {}
    for pattern, values in control.items():
        total = int(values["total"])
        summary[pattern] = {
            "accuracy": float(values["correct"] / total) if total else 0.0,
            "raw_accuracy": float(values["raw_correct"] / total) if total else 0.0,
            "mean_margin": _safe_mean(values["margins"]),
            "n_samples": total,
        }
    return summary


def _summarize_decode_control_overall(control: Dict) -> Dict[str, float]:
    correct = sum(int(values["correct"]) for values in control.values())
    raw_correct = sum(int(values["raw_correct"]) for values in control.values())
    total = sum(int(values["total"]) for values in control.values())
    margins = []
    for values in control.values():
        margins.extend(values["margins"])
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "raw_accuracy": float(raw_correct / total) if total else 0.0,
        "mean_margin": _safe_mean(margins),
        "n_samples": int(total),
    }


def _init_all_target_decode_control(patterns: List[str]) -> Dict[str, Dict]:
    return {
        source: _init_decode_control(patterns)
        for source in patterns
    }


def _record_all_target_decode_control(
    control: Dict,
    source_pattern: str,
    network: SubjectNetwork,
    generated_cases: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    if source_pattern not in control:
        control[source_pattern] = _init_decode_control(list(generated_cases))
    for target_pattern, target_cases in generated_cases.items():
        if target_pattern not in control[source_pattern]:
            control[source_pattern][target_pattern] = {
                "correct": 0,
                "raw_correct": 0,
                "total": 0,
                "margins": [],
            }
        _record_decode_control(
            control[source_pattern],
            target_pattern,
            _evaluate_network_on_cases(network, target_cases),
        )


def _summarize_all_target_decode_control(control: Dict) -> Dict:
    return {
        source: _summarize_decode_control(targets)
        for source, targets in control.items()
    }


SUBJECT_SPECIFICITY_CONTROL_KEYS = [
    "wrong_signature_mse",
    "null_mse",
    "noise_mse",
    "train_centroid_mse",
    "condition_ablation_mse",
]


def _summarize_subject_specificity_records(records: List[Dict], patterns: List[str]) -> Dict:
    """Summarize paired subject-specific functional controls."""

    def summarize_subset(subset: List[Dict]) -> Dict:
        paired_records = [
            record for record in subset
            if any(key in record for key in SUBJECT_SPECIFICITY_CONTROL_KEYS)
        ]
        matched_values = [float(record["matched_mse"]) for record in paired_records]
        control_values = {
            key: [float(record[key]) for record in paired_records if key in record]
            for key in SUBJECT_SPECIFICITY_CONTROL_KEYS
        }
        best_control_values = [
            min(float(record[key]) for key in SUBJECT_SPECIFICITY_CONTROL_KEYS if key in record)
            for record in paired_records
        ]
        improvements = [
            best_control - matched
            for best_control, matched in zip(best_control_values, matched_values)
        ]
        return {
            "matched_mse": _safe_mean(matched_values),
            **{
                key: _safe_mean(values)
                for key, values in control_values.items()
            },
            "control_sample_counts": {
                key: int(len(values))
                for key, values in control_values.items()
            },
            "best_control_mse": _safe_mean(best_control_values),
            "matched_improvement_vs_best_control": _safe_mean(improvements),
            "win_rate_vs_best_control": (
                float(np.mean([improvement > 0.0 for improvement in improvements]))
                if improvements else 0.0
            ),
            "median_improvement_vs_best_control": (
                float(np.median(improvements)) if improvements else 0.0
            ),
            "n_samples": int(len(paired_records)),
        }

    summary = summarize_subset(records)
    summary["per_behavior"] = {
        pattern: summarize_subset([
            record for record in records
            if record.get("pattern") == pattern
        ])
        for pattern in patterns
    }
    return summary


def _monotonic_direction(pattern: str) -> Optional[str]:
    if pattern in {"sorted_ascending", "increasing_pairs"}:
        return "increasing"
    if pattern in {"sorted_descending", "decreasing_pairs"}:
        return "decreasing"
    return None


def compute_proof_metrics(
    model: FunctionalHyperNetwork,
    editor: BehaviorEditor,
    all_weights: torch.Tensor,
    all_signatures: torch.Tensor,
    all_labels: torch.Tensor,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    dataset_provenance: Optional[Dict] = None,
) -> Dict:
    """Compute metrics aligned to Interpret, Steer, and Decode claims."""
    device = next(model.parameters()).device
    train_indices = train_indices.cpu().long()
    test_indices = test_indices.cpu().long()

    train_signatures = all_signatures[train_indices].to(device)
    test_signatures = all_signatures[test_indices].to(device)
    train_labels = all_labels[train_indices].cpu().numpy()
    test_labels = all_labels[test_indices].cpu().numpy()

    with torch.no_grad():
        train_condition = model.encode_signature(train_signatures).cpu().numpy()
        test_condition = model.encode_signature(test_signatures).cpu().numpy()

    behavior_suite = build_clean_behavior_suite()
    focused_patterns = list(CLEAN_PROOF_PATTERNS)
    focused_indices = {
        PATTERN_TO_IDX[p]
        for p in focused_patterns
        if p in PATTERN_TO_IDX
    }
    train_focused_mask = np.array([label in focused_indices for label in train_labels])
    test_focused_mask = np.array([label in focused_indices for label in test_labels])
    focused_heldout_majority_baseline = 0.0
    if test_focused_mask.any():
        _, focused_heldout_counts = np.unique(
            test_labels[test_focused_mask],
            return_counts=True,
        )
        focused_heldout_majority_baseline = float(
            focused_heldout_counts.max() / focused_heldout_counts.sum()
        )

    raw_train = all_signatures[train_indices].cpu().numpy()
    raw_test = all_signatures[test_indices].cpu().numpy()
    raw_signature_accuracy = _fit_score_logistic_regression(
        raw_train,
        train_labels,
        raw_test,
        test_labels,
    )
    signature_condition_accuracy = _fit_score_logistic_regression(
        train_condition,
        train_labels,
        test_condition,
        test_labels,
    )
    focused_raw_signature_accuracy = 0.0
    focused_condition_accuracy = 0.0
    focused_raw_signature_rf_accuracy = 0.0
    if train_focused_mask.any() and test_focused_mask.any():
        focused_raw_signature_accuracy = _fit_score_logistic_regression(
            raw_train[train_focused_mask],
            train_labels[train_focused_mask],
            raw_test[test_focused_mask],
            test_labels[test_focused_mask],
        )
        focused_condition_accuracy = _fit_score_logistic_regression(
            train_condition[train_focused_mask],
            train_labels[train_focused_mask],
            test_condition[test_focused_mask],
            test_labels[test_focused_mask],
        )
        focused_rf_details = _fit_random_forest_details(
            raw_train[train_focused_mask],
            train_labels[train_focused_mask],
            raw_test[test_focused_mask],
            test_labels[test_focused_mask],
            IDX_TO_PATTERN,
        )
        focused_raw_signature_rf_accuracy = focused_rf_details["accuracy"]
        focused_raw_signature_rf_per_behavior_recall = focused_rf_details[
            "per_behavior_recall"
        ]
        focused_test_samples_per_behavior = focused_rf_details[
            "per_behavior_samples"
        ]
    else:
        focused_raw_signature_rf_per_behavior_recall = {}
        focused_test_samples_per_behavior = {}
    focused_dataset_baseline = _compute_signature_dataset_baseline(
        all_signatures,
        all_labels,
        focused_indices,
    )

    decode_correct = 0
    decode_total = 0
    decode_margins = []
    generated_decode_correct = 0
    generated_decode_raw_correct = 0
    generated_decode_total = 0
    generated_decode_margins = []
    generated_decode_by_pattern = {
        pattern: {"correct": 0, "raw_correct": 0, "total": 0, "margins": []}
        for pattern in focused_patterns
    }
    generated_decode_by_direction = {
        "increasing": {"correct": 0, "raw_correct": 0, "total": 0, "margins": []},
        "decreasing": {"correct": 0, "raw_correct": 0, "total": 0, "margins": []},
    }
    shuffled_decode_by_target = _init_decode_control(focused_patterns)
    shuffled_source_target_control: Dict[str, Dict] = {
        pattern: {} for pattern in focused_patterns
    }
    null_signature_decode_by_target = _init_decode_control(focused_patterns)
    noise_signature_decode_by_target = _init_decode_control(focused_patterns)
    train_centroid_decode_by_target = _init_decode_control(focused_patterns)
    condition_ablation_decode_by_target = _init_decode_control(focused_patterns)
    null_signature_decode_all_target = _init_all_target_decode_control(focused_patterns)
    noise_signature_decode_all_target = _init_all_target_decode_control(focused_patterns)
    train_centroid_decode_all_target = _init_all_target_decode_control(focused_patterns)
    condition_ablation_decode_all_target = _init_all_target_decode_control(
        focused_patterns
    )
    shuffled_decode_correct = 0
    shuffled_decode_total = 0
    shuffled_decode_margins = []
    opposite_shuffled_decode_correct = 0
    opposite_shuffled_decode_total = 0
    opposite_shuffled_decode_margins = []
    within_shuffled_decode_correct = 0
    within_shuffled_decode_total = 0
    within_shuffled_decode_margins = []
    steer_deltas = []
    steer_success = 0
    steer_total = 0
    generated_steer_deltas = []
    generated_steer_success = 0
    generated_steer_total = 0
    generated_no_edit_success = 0
    generated_no_edit_total = 0
    generated_cross_direction_no_edit_success = 0
    generated_cross_direction_no_edit_total = 0
    generated_collapsed_steer_success = 0
    generated_collapsed_steer_total = 0
    generated_collapsed_steer_deltas = []
    generated_steer_by_target = {
        pattern: {"success": 0, "total": 0, "deltas": []}
        for pattern in focused_patterns
    }
    generated_steer_by_direction = {
        "increasing": {"success": 0, "total": 0, "deltas": []},
        "decreasing": {"success": 0, "total": 0, "deltas": []},
    }
    subject_specificity = {
        "matched_mse": [],
        "wrong_signature_mse": [],
        "null_mse": [],
        "noise_mse": [],
        "train_centroid_mse": [],
        "condition_ablation_mse": [],
        "records": [],
    }

    generated_seed = behavior_suite["metadata"]["seed"]
    generated_cases_per_class = behavior_suite["metadata"]["heldout_per_class"]
    generated_cases = {
        pattern: {
            "positive": torch.tensor(cases["positive"], dtype=torch.float32),
            "negative": torch.tensor(cases["negative"], dtype=torch.float32),
        }
        for pattern, cases in behavior_suite["heldout"].items()
    }

    test_weights = all_weights[test_indices]
    test_sigs = all_signatures[test_indices]
    test_label_tensor = all_labels[test_indices]
    shuffled_signature_by_local_idx = {}
    shuffled_signature_source_pattern_by_local_idx = {}
    opposite_direction_signature_by_local_idx = {}
    within_direction_signature_by_local_idx = {}
    for local_idx in range(len(test_indices)):
        label = int(test_label_tensor[local_idx])
        pattern = IDX_TO_PATTERN[label]
        direction = _monotonic_direction(pattern)
        for offset in range(1, len(test_indices)):
            candidate_idx = (local_idx + offset) % len(test_indices)
            candidate_label = int(test_label_tensor[candidate_idx])
            candidate_pattern = IDX_TO_PATTERN[candidate_label]
            candidate_direction = _monotonic_direction(candidate_pattern)
            if candidate_label != label and local_idx not in shuffled_signature_by_local_idx:
                shuffled_signature_by_local_idx[local_idx] = test_sigs[candidate_idx]
                shuffled_signature_source_pattern_by_local_idx[local_idx] = (
                    candidate_pattern
                )
            if (
                direction
                and candidate_direction
                and candidate_direction != direction
                and local_idx not in opposite_direction_signature_by_local_idx
            ):
                opposite_direction_signature_by_local_idx[local_idx] = test_sigs[
                    candidate_idx
                ]
            if (
                direction
                and candidate_direction
                and candidate_direction == direction
                and candidate_label != label
                and local_idx not in within_direction_signature_by_local_idx
            ):
                within_direction_signature_by_local_idx[local_idx] = test_sigs[
                    candidate_idx
                ]
            if (
                local_idx in shuffled_signature_by_local_idx
                and local_idx in opposite_direction_signature_by_local_idx
                and local_idx in within_direction_signature_by_local_idx
            ):
                break

    target_centroids = {}
    for pattern in focused_patterns:
        pattern_idx = PATTERN_TO_IDX.get(pattern)
        if pattern_idx is None:
            continue
        mask = all_labels[train_indices] == pattern_idx
        if bool(mask.any()):
            target_centroids[pattern] = all_signatures[train_indices][mask].mean(0)

    train_signature_mean = all_signatures[train_indices].mean(0)
    train_signature_std = all_signatures[train_indices].std(0).clamp(min=1e-6)
    query_generator = torch.Generator().manual_seed(generated_seed + 17)
    subject_query_inputs = torch.randint(
        low=0,
        high=10,
        size=(128, model.config.input_dim),
        generator=query_generator,
        dtype=torch.float32,
    )

    def condition_baseline_for_pattern(
        pattern_name: str,
    ) -> Optional[torch.Tensor]:
        if not model.config.use_condition_residual_decoder:
            return None
        centroid = target_centroids.get(pattern_name)
        if centroid is None:
            return None
        with torch.no_grad():
            return model.encode_signature(centroid.unsqueeze(0).to(device))

    def generated_network_from_signature(
        signature_tensor: torch.Tensor,
        baseline_pattern: Optional[str] = None,
    ) -> SubjectNetwork:
        with torch.no_grad():
            signature_batch = signature_tensor.unsqueeze(0).to(device)
            control_condition = model.encode_signature(signature_batch)
            control_generated = model.decode_weights(
                torch.zeros(1, model.config.latent_dim, device=device),
                control_condition,
                condition_baseline=(
                    condition_baseline_for_pattern(baseline_pattern)
                    if baseline_pattern is not None
                    else None
                ),
            )[0].cpu()
        return SubjectNetwork.from_weights(
            control_generated,
            num_layers=model.config.num_layers,
            neurons_per_layer=model.config.neurons_per_layer,
            input_dim=model.config.input_dim,
        )

    def condition_ablation_network(
        baseline_pattern: Optional[str] = None,
    ) -> SubjectNetwork:
        with torch.no_grad():
            control_generated = model.decode_weights(
                torch.zeros(1, model.config.latent_dim, device=device),
                torch.zeros(1, model.config.condition_dim, device=device),
                condition_baseline=(
                    condition_baseline_for_pattern(baseline_pattern)
                    if baseline_pattern is not None
                    else None
                ),
            )[0].cpu()
        return SubjectNetwork.from_weights(
            control_generated,
            num_layers=model.config.num_layers,
            neurons_per_layer=model.config.neurons_per_layer,
            input_dim=model.config.input_dim,
        )

    def mse_to_subject_outputs(
        candidate_net: SubjectNetwork,
        reference_net: SubjectNetwork,
    ) -> float:
        with torch.no_grad():
            reference_outputs = reference_net(subject_query_inputs)
            candidate_outputs = candidate_net(subject_query_inputs)
        return float(F.mse_loss(candidate_outputs, reference_outputs).item())

    for local_idx in range(len(test_indices)):
        label = int(test_label_tensor[local_idx])
        if label not in focused_indices:
            continue

        pattern = IDX_TO_PATTERN[label]
        signature = test_sigs[local_idx:local_idx + 1].to(device)
        condition_baseline = condition_baseline_for_pattern(pattern)
        with torch.no_grad():
            condition = model.encode_signature(signature)
            generated = model.decode_weights(
                torch.zeros(1, model.config.latent_dim, device=device),
                condition,
                condition_baseline=condition_baseline,
            )[0].cpu()

        generated_net = SubjectNetwork.from_weights(
            generated,
            num_layers=model.config.num_layers,
            neurons_per_layer=model.config.neurons_per_layer,
            input_dim=model.config.input_dim,
        )
        behavior_result = test_behavior(generated_net, pattern)
        if behavior_result.get("supported", False):
            decode_total += 1
            decode_correct += int(bool(behavior_result.get("correct", False)))
            decode_margins.append(float(behavior_result.get("margin", 0.0)))

        if pattern in generated_cases:
            generated_result = _evaluate_network_on_cases(
                generated_net,
                generated_cases[pattern],
            )
            generated_decode_total += 1
            generated_decode_correct += int(generated_result["correct"])
            generated_decode_raw_correct += int(generated_result["raw_correct"])
            generated_decode_margins.append(generated_result["margin"])
            generated_decode_by_pattern[pattern]["total"] += 1
            generated_decode_by_pattern[pattern]["correct"] += int(
                generated_result["correct"]
            )
            generated_decode_by_pattern[pattern]["raw_correct"] += int(
                generated_result["raw_correct"]
            )
            generated_decode_by_pattern[pattern]["margins"].append(
                generated_result["margin"]
            )
            direction = _monotonic_direction(pattern)
            if direction:
                generated_decode_by_direction[direction]["total"] += 1
                generated_decode_by_direction[direction]["correct"] += int(
                    generated_result["correct"]
                )
                generated_decode_by_direction[direction]["raw_correct"] += int(
                    generated_result["raw_correct"]
                )
                generated_decode_by_direction[direction]["margins"].append(
                    generated_result["margin"]
                )

            source_weights = test_weights[local_idx]
            source_net_for_decode = SubjectNetwork.from_weights(
                source_weights,
                num_layers=model.config.num_layers,
                neurons_per_layer=model.config.neurons_per_layer,
                input_dim=model.config.input_dim,
            )
            subject_record = {
                "pattern": pattern,
                "matched_mse": mse_to_subject_outputs(
                    generated_net,
                    source_net_for_decode,
                ),
            }
            subject_specificity["matched_mse"].append(
                subject_record["matched_mse"]
            )

            shuffled_signature = shuffled_signature_by_local_idx.get(local_idx)
            if shuffled_signature is not None:
                with torch.no_grad():
                    shuffled_condition = model.encode_signature(
                        shuffled_signature.unsqueeze(0).to(device)
                    )
                    shuffled_generated = model.decode_weights(
                        torch.zeros(1, model.config.latent_dim, device=device),
                        shuffled_condition,
                        condition_baseline=condition_baseline,
                    )[0].cpu()
                shuffled_net = SubjectNetwork.from_weights(
                    shuffled_generated,
                    num_layers=model.config.num_layers,
                    neurons_per_layer=model.config.neurons_per_layer,
                    input_dim=model.config.input_dim,
                )
                shuffled_result = _evaluate_network_on_cases(
                    shuffled_net,
                    generated_cases[pattern],
                )
                shuffled_decode_total += 1
                shuffled_decode_correct += int(shuffled_result["correct"])
                shuffled_decode_margins.append(shuffled_result["margin"])
                _record_decode_control(
                    shuffled_decode_by_target,
                    pattern,
                    shuffled_result,
                )
                subject_specificity["wrong_signature_mse"].append(
                    mse_to_subject_outputs(shuffled_net, source_net_for_decode)
                )
                subject_record["wrong_signature_mse"] = subject_specificity[
                    "wrong_signature_mse"
                ][-1]
                source_pattern = shuffled_signature_source_pattern_by_local_idx[
                    local_idx
                ]
                shuffled_source_target_control.setdefault(pattern, {}).setdefault(
                    source_pattern,
                    {"correct": 0, "raw_correct": 0, "total": 0, "margins": []},
                )
                _record_decode_control(
                    shuffled_source_target_control[pattern],
                    source_pattern,
                    shuffled_result,
                )

            opposite_signature = opposite_direction_signature_by_local_idx.get(local_idx)
            if opposite_signature is not None:
                with torch.no_grad():
                    opposite_condition = model.encode_signature(
                        opposite_signature.unsqueeze(0).to(device)
                    )
                    opposite_generated = model.decode_weights(
                        torch.zeros(1, model.config.latent_dim, device=device),
                        opposite_condition,
                        condition_baseline=condition_baseline,
                    )[0].cpu()
                opposite_net = SubjectNetwork.from_weights(
                    opposite_generated,
                    num_layers=model.config.num_layers,
                    neurons_per_layer=model.config.neurons_per_layer,
                    input_dim=model.config.input_dim,
                )
                opposite_result = _evaluate_network_on_cases(
                    opposite_net,
                    generated_cases[pattern],
                )
                opposite_shuffled_decode_total += 1
                opposite_shuffled_decode_correct += int(opposite_result["correct"])
                opposite_shuffled_decode_margins.append(opposite_result["margin"])

            within_signature = within_direction_signature_by_local_idx.get(local_idx)
            if within_signature is not None:
                with torch.no_grad():
                    within_condition = model.encode_signature(
                        within_signature.unsqueeze(0).to(device)
                    )
                    within_generated = model.decode_weights(
                        torch.zeros(1, model.config.latent_dim, device=device),
                        within_condition,
                        condition_baseline=condition_baseline,
                    )[0].cpu()
                within_net = SubjectNetwork.from_weights(
                    within_generated,
                    num_layers=model.config.num_layers,
                    neurons_per_layer=model.config.neurons_per_layer,
                    input_dim=model.config.input_dim,
                )
                within_result = _evaluate_network_on_cases(
                    within_net,
                    generated_cases[pattern],
                )
                within_shuffled_decode_total += 1
                within_shuffled_decode_correct += int(within_result["correct"])
                within_shuffled_decode_margins.append(within_result["margin"])

            null_net = generated_network_from_signature(
                train_signature_mean.cpu(),
                baseline_pattern=pattern,
            )
            subject_specificity["null_mse"].append(
                mse_to_subject_outputs(null_net, source_net_for_decode)
            )
            subject_record["null_mse"] = subject_specificity["null_mse"][-1]
            _record_decode_control(
                null_signature_decode_by_target,
                pattern,
                _evaluate_network_on_cases(null_net, generated_cases[pattern]),
            )
            _record_all_target_decode_control(
                null_signature_decode_all_target,
                pattern,
                null_net,
                generated_cases,
            )

            noise_generator = torch.Generator().manual_seed(generated_seed + local_idx)
            noise_signature = (
                train_signature_mean.cpu()
                + torch.randn(
                    train_signature_mean.shape,
                    generator=noise_generator,
                    dtype=train_signature_mean.dtype,
                )
                * train_signature_std.cpu()
            )
            noise_net = generated_network_from_signature(
                noise_signature,
                baseline_pattern=pattern,
            )
            subject_specificity["noise_mse"].append(
                mse_to_subject_outputs(noise_net, source_net_for_decode)
            )
            subject_record["noise_mse"] = subject_specificity["noise_mse"][-1]
            _record_decode_control(
                noise_signature_decode_by_target,
                pattern,
                _evaluate_network_on_cases(noise_net, generated_cases[pattern]),
            )
            _record_all_target_decode_control(
                noise_signature_decode_all_target,
                pattern,
                noise_net,
                generated_cases,
            )

            target_centroid = target_centroids.get(pattern)
            if target_centroid is not None:
                centroid_net = generated_network_from_signature(
                    target_centroid.cpu(),
                    baseline_pattern=pattern,
                )
                subject_specificity["train_centroid_mse"].append(
                    mse_to_subject_outputs(centroid_net, source_net_for_decode)
                )
                subject_record["train_centroid_mse"] = subject_specificity[
                    "train_centroid_mse"
                ][-1]
                _record_decode_control(
                    train_centroid_decode_by_target,
                    pattern,
                    _evaluate_network_on_cases(
                        centroid_net,
                        generated_cases[pattern],
                    ),
                )
                _record_all_target_decode_control(
                    train_centroid_decode_all_target,
                    pattern,
                    centroid_net,
                    generated_cases,
                )

            ablation_net = condition_ablation_network(baseline_pattern=pattern)
            subject_specificity["condition_ablation_mse"].append(
                mse_to_subject_outputs(ablation_net, source_net_for_decode)
            )
            subject_record["condition_ablation_mse"] = subject_specificity[
                "condition_ablation_mse"
            ][-1]
            _record_decode_control(
                condition_ablation_decode_by_target,
                pattern,
                _evaluate_network_on_cases(ablation_net, generated_cases[pattern]),
            )
            _record_all_target_decode_control(
                condition_ablation_decode_all_target,
                pattern,
                ablation_net,
                generated_cases,
            )
            subject_specificity["records"].append(subject_record)

        source_weights = test_weights[local_idx]
        source_sig = test_sigs[local_idx]
        for target_pattern, target_sig in target_centroids.items():
            if target_pattern == pattern:
                continue

            source_net = SubjectNetwork.from_weights(
                source_weights,
                num_layers=model.config.num_layers,
                neurons_per_layer=model.config.neurons_per_layer,
                input_dim=model.config.input_dim,
            )
            source_target_result = test_behavior(source_net, target_pattern)

            edited_net = editor.create_edited_network(
                source_weights,
                source_sig,
                target_sig,
                source_baseline_signature=target_centroids.get(pattern),
                target_baseline_signature=target_centroids.get(target_pattern),
            )
            edited_target_result = test_behavior(edited_net, target_pattern)

            if (
                source_target_result.get("supported", False)
                and edited_target_result.get("supported", False)
            ):
                before_margin = float(source_target_result.get("margin", 0.0))
                after_margin = float(edited_target_result.get("margin", 0.0))
                steer_deltas.append(after_margin - before_margin)
                steer_total += 1
                steer_success += int(bool(edited_target_result.get("correct", False)))

            if target_pattern in generated_cases:
                source_generated_result = _evaluate_network_on_cases(
                    source_net,
                    generated_cases[target_pattern],
                )
                edited_generated_result = _evaluate_network_on_cases(
                    edited_net,
                    generated_cases[target_pattern],
                )
                generated_delta = (
                    edited_generated_result["margin"]
                    - source_generated_result["margin"]
                )
                generated_steer_deltas.append(generated_delta)
                generated_steer_total += 1
                generated_no_edit_total += 1
                generated_no_edit_success += int(
                    bool(source_generated_result["correct"])
                )
                generated_success = int(bool(edited_generated_result["correct"]))
                generated_steer_success += generated_success
                generated_steer_by_target[target_pattern]["total"] += 1
                generated_steer_by_target[target_pattern]["success"] += generated_success
                generated_steer_by_target[target_pattern]["deltas"].append(
                    generated_delta
                )
                source_direction = _monotonic_direction(pattern)
                target_direction = _monotonic_direction(target_pattern)
                if target_direction:
                    generated_steer_by_direction[target_direction]["total"] += 1
                    generated_steer_by_direction[target_direction]["success"] += (
                        generated_success
                    )
                    generated_steer_by_direction[target_direction]["deltas"].append(
                        generated_delta
                    )
                if (
                    source_direction
                    and target_direction
                    and source_direction != target_direction
                ):
                    generated_cross_direction_no_edit_total += 1
                    generated_cross_direction_no_edit_success += int(
                        bool(source_generated_result["correct"])
                    )
                    generated_collapsed_steer_total += 1
                    generated_collapsed_steer_success += generated_success
                    generated_collapsed_steer_deltas.append(generated_delta)

    generated_decode_per_pattern = {}
    for pattern, values in generated_decode_by_pattern.items():
        total = values["total"]
        generated_decode_per_pattern[pattern] = {
            "accuracy": float(values["correct"] / total) if total else 0.0,
            "raw_accuracy": float(values["raw_correct"] / total) if total else 0.0,
            "mean_margin": _safe_mean(values["margins"]),
            "n_samples": int(total),
        }

    generated_decode_per_direction = {}
    for direction, values in generated_decode_by_direction.items():
        total = values["total"]
        generated_decode_per_direction[direction] = {
            "accuracy": float(values["correct"] / total) if total else 0.0,
            "raw_accuracy": float(values["raw_correct"] / total) if total else 0.0,
            "mean_margin": _safe_mean(values["margins"]),
            "n_samples": int(total),
        }

    shuffled_decode_per_target = _summarize_decode_control(shuffled_decode_by_target)
    shuffled_source_target_summary = {
        target: _summarize_decode_control(source_controls)
        for target, source_controls in shuffled_source_target_control.items()
    }
    null_decode_summary = _summarize_decode_control_overall(
        null_signature_decode_by_target
    )
    noise_decode_summary = _summarize_decode_control_overall(
        noise_signature_decode_by_target
    )
    train_centroid_decode_summary = _summarize_decode_control_overall(
        train_centroid_decode_by_target
    )
    condition_ablation_decode_summary = _summarize_decode_control_overall(
        condition_ablation_decode_by_target
    )
    subject_specificity_summary = _summarize_subject_specificity_records(
        subject_specificity["records"],
        focused_patterns,
    )

    generated_steer_per_target = {}
    for pattern, values in generated_steer_by_target.items():
        total = values["total"]
        generated_steer_per_target[pattern] = {
            "success_rate": float(values["success"] / total) if total else 0.0,
            "mean_margin_delta": _safe_mean(values["deltas"]),
            "n_edits": int(total),
        }

    generated_steer_per_direction = {}
    for direction, values in generated_steer_by_direction.items():
        total = values["total"]
        generated_steer_per_direction[direction] = {
            "success_rate": float(values["success"] / total) if total else 0.0,
            "mean_margin_delta": _safe_mean(values["deltas"]),
            "n_edits": int(total),
        }

    checkpoint_suite_metadata = getattr(model, "_behavior_suite_metadata", None)
    suite_matches_checkpoint = bool(
        checkpoint_suite_metadata
        and checkpoint_suite_metadata.get("support_hash")
        == behavior_suite["metadata"]["support_hash"]
        and checkpoint_suite_metadata.get("heldout_hash")
        == behavior_suite["metadata"]["heldout_hash"]
    )

    metrics = {
        "interpret": {
            "raw_signature_accuracy": raw_signature_accuracy,
            "signature_condition_accuracy": signature_condition_accuracy,
            "focused_raw_signature_accuracy": focused_raw_signature_accuracy,
            "focused_raw_signature_random_forest_accuracy": focused_raw_signature_rf_accuracy,
            "focused_raw_signature_random_forest_per_behavior_recall": (
                focused_raw_signature_rf_per_behavior_recall
            ),
            "focused_signature_condition_accuracy": focused_condition_accuracy,
            "focused_test_samples": int(test_focused_mask.sum()),
            "focused_test_samples_per_behavior": focused_test_samples_per_behavior,
            "focused_heldout_majority_baseline_accuracy": (
                focused_heldout_majority_baseline
            ),
            "train_samples": int(len(train_indices)),
            "test_samples": int(len(test_indices)),
            **focused_dataset_baseline,
        },
        "steer": {
            "focused_patterns": focused_patterns,
            "mean_target_margin_delta": _safe_mean(steer_deltas),
            "target_success_rate": float(steer_success / steer_total) if steer_total else 0.0,
            "n_edits": int(steer_total),
            "generated_heldout_seed": generated_seed,
            "generated_heldout_cases_per_class": generated_cases_per_class,
            "generated_heldout_mean_target_margin_delta": _safe_mean(
                generated_steer_deltas
            ),
            "generated_heldout_target_success_rate": (
                float(generated_steer_success / generated_steer_total)
                if generated_steer_total else 0.0
            ),
            "generated_heldout_n_edits": int(generated_steer_total),
            "generated_heldout_per_target": generated_steer_per_target,
            "generated_heldout_no_edit_target_success_rate": (
                float(generated_no_edit_success / generated_no_edit_total)
                if generated_no_edit_total else 0.0
            ),
            "generated_heldout_no_edit_n_edits": int(generated_no_edit_total),
            "generated_heldout_cross_direction_no_edit_success_rate": (
                float(
                    generated_cross_direction_no_edit_success
                    / generated_cross_direction_no_edit_total
                )
                if generated_cross_direction_no_edit_total else 0.0
            ),
            "generated_heldout_cross_direction_no_edit_n_edits": int(
                generated_cross_direction_no_edit_total
            ),
            "generated_heldout_collapsed_direction_success_rate": (
                float(
                    generated_collapsed_steer_success
                    / generated_collapsed_steer_total
                )
                if generated_collapsed_steer_total else 0.0
            ),
            "generated_heldout_collapsed_direction_margin_delta": _safe_mean(
                generated_collapsed_steer_deltas
            ),
            "generated_heldout_collapsed_direction_n_edits": int(
                generated_collapsed_steer_total
            ),
            "generated_heldout_per_direction": generated_steer_per_direction,
        },
        "decode": {
            "focused_patterns": focused_patterns,
            "condition_only_behavior_accuracy": float(decode_correct / decode_total) if decode_total else 0.0,
            "mean_condition_only_margin": _safe_mean(decode_margins),
            "n_samples": int(decode_total),
            "generated_heldout_seed": generated_seed,
            "generated_heldout_cases_per_class": generated_cases_per_class,
            "behavior_correctness_margin_threshold": (
                BEHAVIOR_CORRECTNESS_MARGIN_THRESHOLD
            ),
            "generated_heldout_behavior_accuracy": (
                float(generated_decode_correct / generated_decode_total)
                if generated_decode_total else 0.0
            ),
            "generated_heldout_raw_behavior_accuracy": (
                float(generated_decode_raw_correct / generated_decode_total)
                if generated_decode_total else 0.0
            ),
            "generated_heldout_mean_margin": _safe_mean(generated_decode_margins),
            "generated_heldout_n_samples": int(generated_decode_total),
            "generated_heldout_per_pattern": generated_decode_per_pattern,
            "generated_heldout_shuffled_signature_accuracy": (
                float(shuffled_decode_correct / shuffled_decode_total)
                if shuffled_decode_total else 0.0
            ),
            "generated_heldout_shuffled_signature_mean_margin": _safe_mean(
                shuffled_decode_margins
            ),
            "generated_heldout_shuffled_signature_n_samples": int(
                shuffled_decode_total
            ),
            "generated_heldout_shuffled_per_target": shuffled_decode_per_target,
            "generated_heldout_shuffled_source_target_policy": (
                "diagnostic_only; proof gate enforces aggregate and per-target "
                "shuffled controls"
            ),
            "generated_heldout_shuffled_source_target": (
                shuffled_source_target_summary
            ),
            "generated_heldout_opposite_direction_shuffled_accuracy": (
                float(
                    opposite_shuffled_decode_correct
                    / opposite_shuffled_decode_total
                )
                if opposite_shuffled_decode_total else 0.0
            ),
            "generated_heldout_opposite_direction_shuffled_mean_margin": _safe_mean(
                opposite_shuffled_decode_margins
            ),
            "generated_heldout_opposite_direction_shuffled_n_samples": int(
                opposite_shuffled_decode_total
            ),
            "generated_heldout_within_direction_shuffled_accuracy": (
                float(within_shuffled_decode_correct / within_shuffled_decode_total)
                if within_shuffled_decode_total else 0.0
            ),
            "generated_heldout_within_direction_shuffled_mean_margin": _safe_mean(
                within_shuffled_decode_margins
            ),
            "generated_heldout_within_direction_shuffled_n_samples": int(
                within_shuffled_decode_total
            ),
            "generated_heldout_collapsed_direction_accuracy": (
                float(generated_decode_correct / generated_decode_total)
                if generated_decode_total else 0.0
            ),
            "generated_heldout_per_direction": generated_decode_per_direction,
            "generated_heldout_null_signature_accuracy": (
                null_decode_summary["accuracy"]
            ),
            "generated_heldout_null_signature_mean_margin": (
                null_decode_summary["mean_margin"]
            ),
            "generated_heldout_null_signature_n_samples": (
                null_decode_summary["n_samples"]
            ),
            "generated_heldout_null_signature_per_target": (
                _summarize_decode_control(null_signature_decode_by_target)
            ),
            "generated_heldout_null_signature_all_target": (
                _summarize_all_target_decode_control(null_signature_decode_all_target)
            ),
            "generated_heldout_noise_signature_accuracy": (
                noise_decode_summary["accuracy"]
            ),
            "generated_heldout_noise_signature_mean_margin": (
                noise_decode_summary["mean_margin"]
            ),
            "generated_heldout_noise_signature_n_samples": (
                noise_decode_summary["n_samples"]
            ),
            "generated_heldout_noise_signature_per_target": (
                _summarize_decode_control(noise_signature_decode_by_target)
            ),
            "generated_heldout_noise_signature_all_target": (
                _summarize_all_target_decode_control(noise_signature_decode_all_target)
            ),
            "generated_heldout_train_centroid_signature_accuracy": (
                train_centroid_decode_summary["accuracy"]
            ),
            "generated_heldout_train_centroid_signature_mean_margin": (
                train_centroid_decode_summary["mean_margin"]
            ),
            "generated_heldout_train_centroid_signature_n_samples": (
                train_centroid_decode_summary["n_samples"]
            ),
            "generated_heldout_train_centroid_signature_per_target": (
                _summarize_decode_control(train_centroid_decode_by_target)
            ),
            "generated_heldout_train_centroid_signature_all_target": (
                _summarize_all_target_decode_control(
                    train_centroid_decode_all_target
                )
            ),
            "generated_heldout_condition_ablation_accuracy": (
                condition_ablation_decode_summary["accuracy"]
            ),
            "generated_heldout_condition_ablation_mean_margin": (
                condition_ablation_decode_summary["mean_margin"]
            ),
            "generated_heldout_condition_ablation_n_samples": (
                condition_ablation_decode_summary["n_samples"]
            ),
            "generated_heldout_condition_ablation_per_target": (
                _summarize_decode_control(condition_ablation_decode_by_target)
            ),
            "generated_heldout_condition_ablation_all_target": (
                _summarize_all_target_decode_control(
                    condition_ablation_decode_all_target
                )
            ),
            "subject_functional_specificity": {
                **subject_specificity_summary,
                "query_seed": int(generated_seed + 17),
                "query_count": int(len(subject_query_inputs)),
            },
        },
    }
    metrics["behavior_suite"] = {
        **behavior_suite["metadata"],
        "checkpoint_metadata": checkpoint_suite_metadata,
        "matches_checkpoint_metadata": suite_matches_checkpoint,
    }
    metrics["dataset_provenance"] = {
        "reload_matches_checkpoint": bool(
            (dataset_provenance or {}).get("reload_matches_checkpoint", False)
        )
    }
    metrics["clean_proof_gate"] = evaluate_clean_proof_gate(metrics)
    return metrics


def run_evaluation(
    model_path: str,
    output_dir: Optional[str] = None,
) -> EvaluationResults:
    """
    Run comprehensive evaluation pipeline.
    
    Args:
        model_path: Path to trained model .pt file
        output_dir: Directory for output files (TSV, JSON). 
                   Defaults to model_dir/evaluation/
    
    Returns:
        EvaluationResults with all metrics
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine output directory
    if output_dir is None:
        model_dir = Path(model_path).parent
        output_dir = str(model_dir / "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    print_header("HYPERNET EVALUATION PIPELINE")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model = FunctionalHyperNetwork.load(model_path)
    model.eval()
    editor = BehaviorEditor(model)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    
    # Load data
    print("Loading dataset...")
    dataset_patterns = getattr(model, "_dataset_patterns", None)
    if dataset_patterns:
        print(f"Using checkpoint dataset patterns: {', '.join(dataset_patterns)}")
    checkpoint_provenance = getattr(model, "_dataset_provenance", None) or {}
    checkpoint_source_count = checkpoint_provenance.get("source_count")
    data_max_samples = int(checkpoint_source_count) if checkpoint_source_count else None
    data = load_data(max_samples=data_max_samples, include_patterns=dataset_patterns)
    all_weights = data['weights']
    all_signatures = data['signatures']
    all_labels = data['labels']
    reloaded_provenance = data.get("dataset_provenance", {})
    reload_comparison = compare_dataset_provenance(
        checkpoint_provenance,
        reloaded_provenance,
    )
    evaluation_provenance = {
        **(checkpoint_provenance or reloaded_provenance),
        "reload_matches_checkpoint": reload_comparison["matches"],
        "reload_comparison": reload_comparison,
    }
    print(f"Dataset: {len(all_weights)} total samples")
    
    # Get test split
    test_weights, test_signatures, test_labels, has_saved_indices = get_test_data(
        model, all_weights, all_signatures, all_labels
    )
    if has_saved_indices:
        train_indices = model._train_indices
        test_indices = model._val_indices
    else:
        torch.manual_seed(42)
        split_indices = torch.randperm(len(all_weights))
        n_val = int(len(all_weights) * 0.1)
        train_indices = split_indices[:-n_val]
        test_indices = split_indices[-n_val:]
    
    # Get patterns in test set
    unique_labels = torch.unique(test_labels).tolist()
    patterns_in_test = [IDX_TO_PATTERN[int(l)] for l in unique_labels]
    
    print(f"Test set: {len(test_weights)} samples, {len(patterns_in_test)} patterns")
    if has_saved_indices:
        print("✓ Using exact validation split from training")
    else:
        print("⚠ Using deterministic split (model lacks saved indices)")
    
    # Initialize results
    results = EvaluationResults(
        model_path=model_path,
        timestamp=timestamp,
        test_size=len(test_weights),
        n_patterns=len(patterns_in_test),
        has_saved_indices=has_saved_indices,
    )
    results.validity_audit = build_validity_audit(
        model,
        has_saved_indices,
        evaluation_provenance,
    )
    results.dataset_provenance = evaluation_provenance
    
    # =========================================================================
    # 1. LATENT SPACE ANALYSIS
    # =========================================================================
    print_section("[1/4] LATENT SPACE ANALYSIS")
    
    latent_metrics = compute_latent_metrics(
        model, test_weights, test_signatures, test_labels
    )
    results.latent_metrics = latent_metrics
    
    print(f"Silhouette Score:        {latent_metrics.silhouette_score:.4f}")
    print(f"Adjusted Rand Index:     {latent_metrics.adjusted_rand_index:.4f}")
    print(f"Linear Separability:     {latent_metrics.linear_separability*100:.1f}%")
    print(f"Inter/Intra Ratio:       {latent_metrics.inter_intra_ratio:.2f}")
    
    # Export for TensorBoard Projector
    projector_dir = os.path.join(output_dir, "projector")
    export_latents_for_projector(
        model, test_weights, test_signatures, test_labels,
        IDX_TO_PATTERN, projector_dir
    )
    print(f"\nSaved: {projector_dir}/latent_vectors.tsv, metadata.tsv")
    
    # =========================================================================
    # 2. RECONSTRUCTION QUALITY
    # =========================================================================
    print_section("[2/4] RECONSTRUCTION QUALITY")
    
    recon_metrics = compute_reconstruction_metrics(
        model, test_weights, test_signatures, test_labels,
        IDX_TO_PATTERN, test_behavior, SubjectNetwork
    )
    results.reconstruction_metrics = recon_metrics
    
    # Print per-pattern table
    print(f"\n{'Pattern':<20} | {'N':>5} | {'Cos':>6} | {'FuncMSE':>8} | {'BehavAcc':>8} | {'Margin':>8}")
    print("-" * 75)
    
    for pattern in sorted(recon_metrics.per_pattern.keys()):
        pm = recon_metrics.per_pattern[pattern]
        print(f"{pattern:<20} | {pm.n_samples:>5} | {pm.weight_cosine_mean:>6.3f} | "
              f"{pm.functional_mse_mean:>8.4f} | {pm.behavioral_accuracy*100:>7.1f}% | "
              f"{pm.margin_mean:>+8.3f}")
    
    print("-" * 75)
    print(f"{'OVERALL':<20} | {recon_metrics.n_samples:>5} | {recon_metrics.overall_weight_cosine:>6.3f} | "
          f"{recon_metrics.overall_functional_mse:>8.4f} | {recon_metrics.overall_behavioral_accuracy*100:>7.1f}% | "
          f"{recon_metrics.overall_margin_mean:>+8.3f}")
    
    # =========================================================================
    # 3. EDITING QUALITY
    # =========================================================================
    print_section("[3/4] EDITING QUALITY")
    print(f"Testing {len(patterns_in_test) * (len(patterns_in_test) - 1)} pattern pairs...")
    
    editing_metrics = compute_editing_metrics(
        model, editor,
        test_weights, test_signatures, test_labels,
        IDX_TO_PATTERN, PATTERN_TO_IDX,
        test_behavior, get_test_cases, SubjectNetwork
    )
    results.editing_metrics = editing_metrics
    
    print(f"\nThreshold 0.5:")
    print(
        "  Overall Threshold Success Rate: "
        f"{editing_metrics.overall_success_rate_05*100:.1f}%"
    )
    print(
        "  Overall Margin-Sign Success Rate: "
        f"{editing_metrics.overall_margin_success_rate*100:.1f}%"
    )
    
    print(f"\nOptimal Threshold: {editing_metrics.global_optimal_threshold:.3f}")
    print(
        "  Overall Threshold Success Rate: "
        f"{editing_metrics.overall_success_rate_optimal*100:.1f}%"
    )
    
    # Print success matrix
    print(f"\nSuccess Matrix (thresholded target positive and negative outputs):")
    print(f"{'':>15}", end="")
    for t in patterns_in_test[:8]:  # Truncate for display
        print(f"{t[:6]:>8}", end="")
    if len(patterns_in_test) > 8:
        print("  ...")
    print()
    
    for s in patterns_in_test[:8]:
        print(f"{s[:14]:<15}", end="")
        for t in patterns_in_test[:8]:
            if s == t:
                print(f"{'--':>8}", end="")
            else:
                rate = editing_metrics.success_matrix_optimal.get(s, {}).get(t, 0.0)
                print(f"{rate*100:>7.0f}%", end="")
        print()
    
    # Best and worst pairs
    print(f"\nBest pairs:")
    for src, tgt, rate in editing_metrics.best_pairs[:3]:
        print(f"  {src} → {tgt}: {rate*100:.0f}%")
    
    print(f"\nWorst pairs:")
    for src, tgt, rate in editing_metrics.worst_pairs[:3]:
        print(f"  {src} → {tgt}: {rate*100:.0f}%")
    
    # Export matrices
    export_editing_matrices(editing_metrics, output_dir, patterns_in_test)
    print(f"\nSaved: {output_dir}/editing_matrix_*.tsv")

    # =========================================================================
    # 4. PROOF METRICS
    # =========================================================================
    print_section("[4/5] PROOF METRICS")

    proof_metrics = compute_proof_metrics(
        model,
        editor,
        all_weights,
        all_signatures,
        all_labels,
        train_indices,
        test_indices,
        evaluation_provenance,
    )
    results.proof_metrics = proof_metrics

    interpret_metrics = proof_metrics["interpret"]
    steer_metrics = proof_metrics["steer"]
    decode_metrics = proof_metrics["decode"]

    print("Interpret:")
    print(
        "  Raw signature logistic accuracy: "
        f"{interpret_metrics['focused_raw_signature_accuracy'] * 100:.1f}%"
    )
    print(
        "  Raw signature random-forest accuracy: "
        f"{interpret_metrics['focused_raw_signature_random_forest_accuracy'] * 100:.1f}%"
    )
    print(
        "  Signature-condition logistic accuracy: "
        f"{interpret_metrics['focused_signature_condition_accuracy'] * 100:.1f}%"
    )
    print(
        "  All-focused raw signature RF accuracy: "
        f"{interpret_metrics['focused_dataset_raw_signature_random_forest_accuracy'] * 100:.1f}%"
    )
    print(
        "  All-focused majority baseline: "
        f"{interpret_metrics['focused_dataset_majority_baseline_accuracy'] * 100:.1f}%"
    )
    print("Steer:")
    print(
        "  Mean target margin delta: "
        f"{steer_metrics['mean_target_margin_delta']:+.3f}"
    )
    print(
        "  Focused target success: "
        f"{steer_metrics['target_success_rate'] * 100:.1f}%"
    )
    print(
        "  Generated-heldout target success: "
        f"{steer_metrics['generated_heldout_target_success_rate'] * 100:.1f}%"
    )
    print(
        "  Generated-heldout no-edit target success: "
        f"{steer_metrics['generated_heldout_no_edit_target_success_rate'] * 100:.1f}%"
    )
    print(
        "  Generated-heldout cross-direction no-edit success: "
        f"{steer_metrics['generated_heldout_cross_direction_no_edit_success_rate'] * 100:.1f}%"
    )
    print(
        "  Generated-heldout cross-direction success: "
        f"{steer_metrics['generated_heldout_collapsed_direction_success_rate'] * 100:.1f}%"
    )
    print(
        "  Generated-heldout margin delta: "
        f"{steer_metrics['generated_heldout_mean_target_margin_delta']:+.3f}"
    )
    print("Decode:")
    print(
        "  Condition-only behavior accuracy: "
        f"{decode_metrics['condition_only_behavior_accuracy'] * 100:.1f}%"
    )
    print(
        "  Mean condition-only margin: "
        f"{decode_metrics['mean_condition_only_margin']:+.3f}"
    )
    print(
        "  Generated-heldout behavior accuracy: "
        f"{decode_metrics['generated_heldout_behavior_accuracy'] * 100:.1f}%"
    )
    print(
        "  Generated-heldout shuffled-signature accuracy: "
        f"{decode_metrics['generated_heldout_shuffled_signature_accuracy'] * 100:.1f}%"
    )
    print(
        "  Opposite-direction shuffled accuracy: "
        f"{decode_metrics['generated_heldout_opposite_direction_shuffled_accuracy'] * 100:.1f}%"
    )
    print(
        "  Within-direction shuffled accuracy: "
        f"{decode_metrics['generated_heldout_within_direction_shuffled_accuracy'] * 100:.1f}%"
    )
    print(
        "  Generated-heldout mean margin: "
        f"{decode_metrics['generated_heldout_mean_margin']:+.3f}"
    )
    
    # =========================================================================
    # 5. SAVE RESULTS
    # =========================================================================
    print_section("[5/5] SAVING RESULTS")
    
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"Saved: {results_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("EVALUATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles generated:")
    print(f"  - results.json (all metrics)")
    print(f"  - projector/latent_vectors.tsv (for TensorBoard Projector)")
    print(f"  - projector/metadata.tsv")
    print(f"  - editing_matrix_05.tsv")
    print(f"  - editing_matrix_optimal.tsv")
    print(f"  - cross_pattern_response.tsv")
    
    print(f"\nKey Metrics:")
    print(f"  Latent Separability:    {latent_metrics.linear_separability*100:.1f}%")
    print(f"  Reconstruction Accuracy: {recon_metrics.overall_behavioral_accuracy*100:.1f}%")
    print(f"  Editing Success @0.5:    {editing_metrics.overall_success_rate_05*100:.1f}%")
    print(f"  Editing Success @opt:    {editing_metrics.overall_success_rate_optimal*100:.1f}%")
    print(f"  Margin-Sign Edit Success:{editing_metrics.overall_margin_success_rate*100:.1f}%")
    print(
        "  Raw Signature RF Acc:    "
        f"{interpret_metrics['focused_dataset_raw_signature_random_forest_accuracy'] * 100:.1f}%"
    )
    print(
        "  Condition Decode Acc:    "
        f"{decode_metrics['condition_only_behavior_accuracy'] * 100:.1f}%"
    )
    print(
        "  Heldout Decode Acc:      "
        f"{decode_metrics['generated_heldout_behavior_accuracy'] * 100:.1f}%"
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hypernet evaluation pipeline")
    parser.add_argument("--model", "-m", required=True, help="Path to model .pt file")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    
    args = parser.parse_args()
    
    run_evaluation(args.model, args.output)
