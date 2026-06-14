"""Direct helper tests for four-behavior decoder development.

Run with:
python model_zoo/scripts/test_four_behavior_decoder_development_helpers.py
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

import train_four_behavior_decoder_development as dev  # noqa: E402


def test_final_raw_path_rejected() -> None:
    final_path = REPO_ROOT / "runs" / "four_behavior_decoder_source_pools_v2" / "final_subjects.json"
    try:
        dev.assert_no_final_raw_paths([final_path])
    except ValueError as error:
        assert "final_subjects.json" in str(error)
    else:
        raise AssertionError("final raw path was not rejected")


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


def test_train_control_selection_is_stable_under_input_order() -> None:
    records = [
        {
            "subject_id": "b",
            "pattern": "sorted_ascending",
            "weights_hash": "22",
            "signature_hash": "bb",
        },
        {
            "subject_id": "a",
            "pattern": "sorted_ascending",
            "weights_hash": "11",
            "signature_hash": "aa",
        },
    ]
    selected = dev.select_train_control(
        records,
        development_subject_id="dev-1",
        control_family="same_label_other_subject",
        control_behavior="sorted_ascending",
    )
    selected_reversed = dev.select_train_control(
        list(reversed(records)),
        development_subject_id="dev-1",
        control_family="same_label_other_subject",
        control_behavior="sorted_ascending",
    )
    assert selected == selected_reversed
    assert selected["subject_id"] in {"a", "b"}


def main() -> None:
    test_final_raw_path_rejected()
    test_best_control_metrics_are_adversarial()
    test_train_control_selection_is_stable_under_input_order()
    print("helper tests passed")


if __name__ == "__main__":
    main()
