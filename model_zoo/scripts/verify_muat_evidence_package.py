"""Run the MUAT evidence package verification commands.

This is intentionally an orchestration script. It does not train models, rerun
experiments, or invoke linting.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]

COMPILE_TARGETS = [
    "model_zoo/scripts/audit_muat_evidence_package.py",
    "model_zoo/scripts/build_evidence_manifest.py",
    "model_zoo/scripts/evaluate_additional_behavior_decode_feasibility.py",
    "model_zoo/scripts/evaluate_four_behavior_source_generation_feasibility.py",
    "model_zoo/scripts/generate_four_behavior_decoder_source_pools.py",
    "model_zoo/scripts/run_stored_probe_steering_robustness.py",
    "model_zoo/scripts/test_four_behavior_decoder_development_helpers.py",
    "model_zoo/scripts/test_four_behavior_decoder_development_v2_helpers.py",
    "model_zoo/scripts/test_four_behavior_decoder_development_v3_helpers.py",
    "model_zoo/scripts/test_four_behavior_representation_steering_helpers.py",
    "model_zoo/scripts/test_four_behavior_representation_steering_v2_helpers.py",
    "model_zoo/scripts/test_four_behavior_representation_steering_v3_helpers.py",
    "model_zoo/scripts/test_four_behavior_representation_steering_v4_helpers.py",
    "model_zoo/scripts/test_four_behavior_representation_steering_v5_helpers.py",
    "model_zoo/scripts/test_four_behavior_representation_steering_v6_helpers.py",
    "model_zoo/scripts/test_four_behavior_representation_steering_v7_helpers.py",
    "model_zoo/scripts/test_four_behavior_representation_steering_v8_helpers.py",
    "model_zoo/scripts/test_four_behavior_representation_steering_v9_helpers.py",
    "model_zoo/scripts/test_four_behavior_functional_weight_editing_v10_helpers.py",
    "model_zoo/scripts/train_four_behavior_decoder_development.py",
    "model_zoo/scripts/train_four_behavior_decoder_development_v2.py",
    "model_zoo/scripts/train_four_behavior_decoder_development_v3_signature_inversion.py",
    "model_zoo/scripts/train_four_behavior_representation_steering.py",
    "model_zoo/scripts/train_four_behavior_representation_steering_v2_centroid_delta.py",
    "model_zoo/scripts/train_four_behavior_representation_steering_v3_diagonal_transport.py",
    "model_zoo/scripts/train_four_behavior_representation_steering_v4_low_rank_residual_transport.py",
    "model_zoo/scripts/train_four_behavior_representation_steering_v5_contrastive_residual_calibration.py",
    "model_zoo/scripts/train_four_behavior_representation_steering_v6_centroid_constrained_primary_correction.py",
    "model_zoo/scripts/train_four_behavior_representation_steering_v7_pareto_frontier_correction.py",
    "model_zoo/scripts/train_four_behavior_representation_steering_v8_source_conditional_tournament_correction.py",
    "model_zoo/scripts/train_four_behavior_representation_steering_v9_source_invariant_target_attractor.py",
    "model_zoo/scripts/train_four_behavior_functional_weight_editing_v10_v9_conditioned_delta.py",
    "model_zoo/scripts/train_robust_signature_edit_vectors.py",
    "model_zoo/scripts/verify_muat_evidence_package.py",
]

COMMANDS = [
    {
        "name": "compile_evidence_scripts",
        "cmd": [sys.executable, "-m", "py_compile", *COMPILE_TARGETS],
    },
    {
        "name": "metric_and_scope_audit",
        "cmd": [sys.executable, "model_zoo/scripts/audit_muat_evidence_package.py"],
    },
    {
        "name": "checksum_manifest_verify",
        "cmd": [sys.executable, "model_zoo/scripts/build_evidence_manifest.py", "--verify"],
    },
]


def run_command(command: Dict) -> Dict:
    completed = subprocess.run(
        command["cmd"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "cmd": command["cmd"],
        "name": command["name"],
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0,
    }


def main() -> None:
    results: List[Dict] = []
    for command in COMMANDS:
        result = run_command(command)
        results.append(result)
        if not result["passed"]:
            break

    summary = {
        "commands": results,
        "failure_count": sum(0 if result["passed"] else 1 for result in results),
        "passed": all(result["passed"] for result in results) and len(results) == len(COMMANDS),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
