"""Direct helper tests for four-behavior decoder development V3."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo"))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

from hypernet.paired_contrast import (  # noqa: E402
    build_digit_probe_examples,
    extract_signature_with_stored_probes,
)
import train_four_behavior_decoder_development_v3_signature_inversion as dev  # noqa: E402


def test_final_raw_path_rejected() -> None:
    final_path = REPO_ROOT / "runs" / "four_behavior_decoder_source_pools_v2" / "final_subjects.json"
    try:
        dev.assert_no_final_raw_paths([final_path])
    except ValueError as error:
        assert "final_subjects.json" in str(error)
    else:
        raise AssertionError("final raw path was not rejected")


def test_differentiable_signature_matches_registered_extractor() -> None:
    generator = torch.Generator().manual_seed(20260618)
    weights = torch.randn(345, generator=generator) * 0.05
    probes = build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    expected = extract_signature_with_stored_probes(weights, probes)
    actual = dev.differentiable_signature(weights, probes).detach()
    max_abs_diff = torch.max(torch.abs(expected - actual)).item()
    assert max_abs_diff <= 1e-5, max_abs_diff


def test_nearest_centroid_inference_uses_train_statistics() -> None:
    centroids = {
        "a": torch.tensor([0.0, 0.0]),
        "b": torch.tensor([2.0, 0.0]),
    }
    query = torch.tensor([1.8, 0.1])
    assert dev.infer_behavior_from_centroids(query, centroids) == "b"


def test_best_control_metrics_are_adversarial() -> None:
    controls = [
        {"control_type": "weak_margin", "target_margin": 0.1, "subject_output_mse": 0.9},
        {"control_type": "strong_margin", "target_margin": 0.7, "subject_output_mse": 0.5},
        {"control_type": "low_mse", "target_margin": 0.2, "subject_output_mse": 0.05},
    ]
    summary = dev.best_control_metrics(controls)
    assert summary["best_target_margin"] == 0.7
    assert summary["best_target_margin_control_type"] == "strong_margin"
    assert summary["best_subject_output_mse"] == 0.05
    assert summary["best_subject_output_mse_control_type"] == "low_mse"


def main() -> None:
    test_final_raw_path_rejected()
    test_differentiable_signature_matches_registered_extractor()
    test_nearest_centroid_inference_uses_train_statistics()
    test_best_control_metrics_are_adversarial()
    print("v3 helper tests passed")


if __name__ == "__main__":
    main()
