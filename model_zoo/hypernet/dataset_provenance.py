"""Dataset provenance and deduplication helpers for proof runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Dict, List, Sequence, Tuple


FINGERPRINT_KEYS: Tuple[str, ...] = (
    "row_hash",
    "weight_hash",
    "signature_hash",
    "weight_signature_hash",
    "combined_hash",
)


def stable_hash_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def make_sample_fingerprint(
    row_index: int,
    sample: Dict,
    flat_weights: Sequence[float],
    signature_features: Sequence[float],
    label: int,
) -> Dict[str, str | int]:
    """Build stable hashes for deduplication and artifact provenance."""
    weight_hash = stable_hash_json(list(flat_weights))
    signature_hash = stable_hash_json(list(signature_features))
    weight_signature_hash = stable_hash_json({
        "weight_hash": weight_hash,
        "signature_hash": signature_hash,
    })
    return {
        "row_index": int(row_index),
        "row_hash": stable_hash_json(sample),
        "weight_hash": weight_hash,
        "signature_hash": signature_hash,
        "weight_signature_hash": weight_signature_hash,
        "combined_hash": stable_hash_json({
            "weight_hash": weight_hash,
            "signature_hash": signature_hash,
            "label": int(label),
        }),
    }


def deduplicate_fingerprints(
    fingerprints: Sequence[Dict[str, str | int]],
    key: str = "weight_signature_hash",
) -> Tuple[List[int], Dict[str, int]]:
    """Keep the first row for each deduplication key and report duplicate counts."""
    if key not in FINGERPRINT_KEYS:
        raise ValueError(f"Unsupported deduplication key: {key}")

    seen = set()
    keep_indices = []
    for idx, fingerprint in enumerate(fingerprints):
        value = _fingerprint_value(fingerprint, key)
        if value in seen:
            continue
        seen.add(value)
        keep_indices.append(idx)

    summary = summarize_duplicate_hashes(fingerprints)
    summary.update({
        "deduplication_key": key,
        "before_count": int(len(fingerprints)),
        "after_count": int(len(keep_indices)),
        "removed_count": int(len(fingerprints) - len(keep_indices)),
    })
    return keep_indices, summary


def summarize_duplicate_hashes(
    fingerprints: Sequence[Dict[str, str | int]]
) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for key in FINGERPRINT_KEYS:
        counts = Counter(_fingerprint_value(fingerprint, key) for fingerprint in fingerprints)
        duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
        summary[f"duplicate_{key}_count"] = int(duplicate_count)
    return summary


def _fingerprint_value(fingerprint: Dict[str, str | int], key: str) -> str | int:
    if key == "weight_signature_hash" and key not in fingerprint:
        return stable_hash_json({
            "weight_hash": fingerprint["weight_hash"],
            "signature_hash": fingerprint["signature_hash"],
        })
    return fingerprint[key]
