"""Build or verify SHA-256 checksums for the MUAT evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "muat_evidence_package_audit" / "manifest.json"

EVIDENCE_FILES = [
    "docs/muat_small_scale_evidence_report.md",
    "docs/preregistrations/four_behavior_stored_probe_decoder_v1.md",
    "docs/preregistrations/four_behavior_source_generation_v2_expanded_support.md",
    "docs/preregistrations/four_behavior_source_generation_v3_full_pool.md",
    "docs/preregistrations/four_behavior_source_generation_v4_accept_reject.md",
    "docs/preregistrations/four_behavior_decoder_source_pools_v1.md",
    "docs/preregistrations/four_behavior_decoder_source_pools_v2.md",
    "docs/preregistrations/four_behavior_decoder_development_v1.md",
    "docs/preregistrations/four_behavior_decoder_development_v2.md",
    "docs/preregistrations/four_behavior_decoder_development_v3_signature_inversion.md",
    "docs/preregistrations/four_behavior_representation_steering_v1.md",
    "docs/preregistrations/four_behavior_representation_steering_v2_centroid_delta.md",
    "docs/preregistrations/four_behavior_representation_steering_v3_diagonal_transport.md",
    "docs/preregistrations/four_behavior_representation_steering_v4_low_rank_residual_transport.md",
    "docs/preregistrations/four_behavior_representation_steering_v5_contrastive_residual_calibration.md",
    "docs/preregistrations/four_behavior_representation_steering_v6_centroid_constrained_primary_correction.md",
    "docs/preregistrations/four_behavior_representation_steering_v7_pareto_frontier_correction.md",
    "docs/preregistrations/four_behavior_representation_steering_v8_source_conditional_tournament_correction.md",
    "docs/preregistrations/four_behavior_representation_steering_v9_source_invariant_target_attractor.md",
    "docs/preregistrations/four_behavior_functional_weight_editing_v10_v9_conditioned_delta.md",
    "docs/representation_steering_v1_failure_diagnosis.md",
    "docs/representation_steering_v2_failure_diagnosis.md",
    "research-log.md",
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
    "runs/muat_evidence_package_audit/results.json",
    "runs/stored_probe_interpret_v1/results.json",
    "runs/stored_probe_interpret_v1/stored_probe_signatures.pt",
    "runs/paired_contrast_final_artifact_v1/paired_contrast_artifact.json",
    "runs/paired_contrast_final_artifact_v1/regenerated_signatures.json",
    "runs/paired_contrast_final_artifact_v1/sidecar_audit.json",
    "runs/paired_contrast_final_artifact_v1/summary.json",
    "runs/paired_contrast_final_artifact_v1/validation.json",
    "runs/stored_probe_functional_decoder_v2_adaptive/model.pt",
    "runs/stored_probe_functional_decoder_v2_final_eval/predictions.json",
    "runs/stored_probe_functional_decoder_v2_final_eval/results.json",
    "runs/stored_probe_signature_edit_vectors_v1_robust_external_eval/results.json",
    "runs/stored_probe_signature_edit_vectors_v2_robust_development/edit_vectors.pt",
    "runs/stored_probe_signature_edit_vectors_v2_robust_development/results.json",
    "runs/stored_probe_signature_edit_vectors_v2_robust_development/train_pool_results.json",
    "runs/stored_probe_signature_edit_vectors_v2_robust_development/validation_pool_results.json",
    "runs/stored_probe_signature_edit_vectors_v2_robust_final_eval/results.json",
    "runs/stored_probe_additional_behavior_decode_feasibility_v1/results.json",
    "runs/four_behavior_source_generation_feasibility_v1/results.json",
    "runs/four_behavior_source_generation_v2_expanded_support/results.json",
    "runs/four_behavior_source_generation_v3_full_pool/results.json",
    "runs/four_behavior_source_generation_v4_accept_reject/results.json",
    "runs/four_behavior_decoder_source_pools_v1/combined_audit.json",
    "runs/four_behavior_decoder_source_pools_v1/final_redacted_audit.json",
    "runs/four_behavior_decoder_source_pools_v2/combined_audit.json",
    "runs/four_behavior_decoder_source_pools_v2/development_subjects.json",
    "runs/four_behavior_decoder_source_pools_v2/final_redacted_audit.json",
    "runs/four_behavior_decoder_source_pools_v2/final_subjects.json",
    "runs/four_behavior_decoder_source_pools_v2/train_subjects.json",
    "runs/four_behavior_decoder_development_v1/model.pt",
    "runs/four_behavior_decoder_development_v1/results.json",
    "runs/four_behavior_decoder_development_v2/model.pt",
    "runs/four_behavior_decoder_development_v2/results.json",
    "runs/four_behavior_decoder_development_v3_signature_inversion/decoded_weights.pt",
    "runs/four_behavior_decoder_development_v3_signature_inversion/results.json",
    "runs/four_behavior_representation_steering_v1/development_results.json",
    "runs/four_behavior_representation_steering_v1/edit_vectors.pt",
    "runs/four_behavior_representation_steering_v1_pools/combined_audit.json",
    "runs/four_behavior_representation_steering_v1_pools/development_subjects.json",
    "runs/four_behavior_representation_steering_v1_pools/final_redacted_audit.json",
    "runs/four_behavior_representation_steering_v1_pools/train_subjects.json",
    "runs/four_behavior_representation_steering_v2_centroid_delta/centroid_delta_vectors.pt",
    "runs/four_behavior_representation_steering_v2_centroid_delta/development_results.json",
    "runs/four_behavior_representation_steering_v2_pools/combined_audit.json",
    "runs/four_behavior_representation_steering_v2_pools/development_subjects.json",
    "runs/four_behavior_representation_steering_v2_pools/final_redacted_audit.json",
    "runs/four_behavior_representation_steering_v2_pools/train_subjects.json",
    "runs/four_behavior_representation_steering_v3_diagonal_transport/development_results.json",
    "runs/four_behavior_representation_steering_v3_diagonal_transport/diagonal_transport_stats.pt",
    "runs/four_behavior_representation_steering_v3_pools/combined_audit.json",
    "runs/four_behavior_representation_steering_v3_pools/development_subjects.json",
    "runs/four_behavior_representation_steering_v3_pools/final_redacted_audit.json",
    "runs/four_behavior_representation_steering_v3_pools/train_subjects.json",
    "runs/four_behavior_representation_steering_v4_low_rank_residual_transport/development_results.json",
    "runs/four_behavior_representation_steering_v4_low_rank_residual_transport/low_rank_residual_transport_stats.pt",
    "runs/four_behavior_representation_steering_v4_pools/combined_audit.json",
    "runs/four_behavior_representation_steering_v4_pools/development_subjects.json",
    "runs/four_behavior_representation_steering_v4_pools/final_redacted_audit.json",
    "runs/four_behavior_representation_steering_v4_pools/train_subjects.json",
    "runs/four_behavior_representation_steering_v5_contrastive_residual_calibration/contrastive_residual_calibration_stats.pt",
    "runs/four_behavior_representation_steering_v5_contrastive_residual_calibration/development_results.json",
    "runs/four_behavior_representation_steering_v5_pools/combined_audit.json",
    "runs/four_behavior_representation_steering_v5_pools/development_subjects.json",
    "runs/four_behavior_representation_steering_v5_pools/final_redacted_audit.json",
    "runs/four_behavior_representation_steering_v5_pools/train_subjects.json",
    "runs/four_behavior_representation_steering_v6_centroid_constrained_primary_correction/centroid_constrained_primary_correction_stats.pt",
    "runs/four_behavior_representation_steering_v6_centroid_constrained_primary_correction/development_results.json",
    "runs/four_behavior_representation_steering_v6_centroid_constrained_primary_correction/posthoc_pareto_diagnosis.json",
    "runs/four_behavior_representation_steering_v6_pools/combined_audit.json",
    "runs/four_behavior_representation_steering_v6_pools/development_subjects.json",
    "runs/four_behavior_representation_steering_v6_pools/final_redacted_audit.json",
    "runs/four_behavior_representation_steering_v6_pools/train_subjects.json",
    "runs/four_behavior_representation_steering_v7_pareto_frontier_correction/development_results.json",
    "runs/four_behavior_representation_steering_v7_pareto_frontier_correction/pareto_frontier_correction_stats.pt",
    "runs/four_behavior_representation_steering_v7_pools/combined_audit.json",
    "runs/four_behavior_representation_steering_v7_pools/development_subjects.json",
    "runs/four_behavior_representation_steering_v7_pools/final_redacted_audit.json",
    "runs/four_behavior_representation_steering_v7_pools/train_subjects.json",
    "runs/four_behavior_representation_steering_v8_source_conditional_tournament_correction/development_results.json",
    "runs/four_behavior_representation_steering_v8_source_conditional_tournament_correction/source_conditional_tournament_correction_stats.pt",
    "runs/four_behavior_representation_steering_v8_pools/combined_audit.json",
    "runs/four_behavior_representation_steering_v8_pools/development_subjects.json",
    "runs/four_behavior_representation_steering_v8_pools/final_redacted_audit.json",
    "runs/four_behavior_representation_steering_v8_pools/train_subjects.json",
    "runs/four_behavior_representation_steering_v9_source_invariant_target_attractor/development_results.json",
    "runs/four_behavior_representation_steering_v9_source_invariant_target_attractor/final_results.json",
    "runs/four_behavior_representation_steering_v9_source_invariant_target_attractor/source_invariant_target_attractor_stats.pt",
    "runs/four_behavior_representation_steering_v9_pools/combined_audit.json",
    "runs/four_behavior_representation_steering_v9_pools/development_subjects.json",
    "runs/four_behavior_representation_steering_v9_pools/final_redacted_audit.json",
    "runs/four_behavior_representation_steering_v9_pools/final_subjects.json",
    "runs/four_behavior_representation_steering_v9_pools/train_subjects.json",
    "runs/four_behavior_functional_weight_editing_v10_pools/combined_audit.json",
    "runs/four_behavior_functional_weight_editing_v10_pools/development_subjects.json",
    "runs/four_behavior_functional_weight_editing_v10_pools/final_redacted_audit.json",
    "runs/four_behavior_functional_weight_editing_v10_pools/train_subjects.json",
    "runs/four_behavior_functional_weight_editing_v10_v9_conditioned_delta/development_results.json",
    "runs/four_behavior_functional_weight_editing_v10_v9_conditioned_delta/v10_ridge_editor.pt",
    "runs/fresh_external_steering_holdout_v1/subjects.json",
    "runs/fresh_external_steering_holdout_v1/summary.json",
    "runs/fresh_external_steering_holdout_v2_robust/subjects.json",
    "runs/fresh_external_steering_holdout_v2_robust/summary.json",
    "runs/fresh_robust_edit_v2_train_pool/subjects.json",
    "runs/fresh_robust_edit_v2_train_pool/summary.json",
    "runs/fresh_external_steering_holdout_v3_robust_final/subjects.json",
    "runs/fresh_external_steering_holdout_v3_robust_final/summary.json",
    "runs/fresh_additional_behavior_decode_holdout_v1/subjects.json",
    "runs/fresh_additional_behavior_decode_holdout_v1/summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT.relative_to(REPO_ROOT)),
        help="Manifest path relative to repo root.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing manifest instead of writing a new one.",
    )
    return parser.parse_args()


def build_manifest() -> Dict:
    files = []
    missing = []
    for rel_path in EVIDENCE_FILES:
        path = REPO_ROOT / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        files.append({
            "path": rel_path,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "SHA-256 manifest for the current MUAT small-scale evidence package.",
        "file_count": len(files),
        "files": files,
        "missing_files": missing,
        "passed": not missing,
    }


def verify_manifest(manifest_path: Path) -> Dict:
    manifest = json.loads(manifest_path.read_text())
    failures: List[str] = []
    for entry in manifest.get("files", []):
        rel_path = entry["path"]
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"missing file: {rel_path}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            failures.append(f"sha256 mismatch for {rel_path}")
        actual_size = path.stat().st_size
        if actual_size != entry["size_bytes"]:
            failures.append(f"size mismatch for {rel_path}")
    for rel_path in manifest.get("missing_files", []):
        failures.append(f"manifest was created with missing file: {rel_path}")
    return {
        "checked_file_count": len(manifest.get("files", [])),
        "failure_count": len(failures),
        "failures": failures,
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "passed": not failures,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    manifest_path = REPO_ROOT / args.output
    if args.verify:
        result = verify_manifest(manifest_path)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
        return

    manifest = build_manifest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({
        "file_count": manifest["file_count"],
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "missing_files": manifest["missing_files"],
        "passed": manifest["passed"],
    }, indent=2, sort_keys=True))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
