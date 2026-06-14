"""Canonical behavior suite for clean hypernet proof runs."""

from __future__ import annotations

import hashlib
import json
import random
from itertools import product
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


SequenceTuple = Tuple[int, ...]
Predicate = Callable[[Sequence[int]], bool]

CLEAN_PROOF_PATTERNS: Tuple[str, ...] = (
    "sorted_ascending",
    "sorted_descending",
    "has_majority",
    "mountain_pattern",
)

CLEAN_PROOF_THRESHOLDS: Dict[str, float] = {
    "min_heldout_samples_per_behavior": 50,
    "raw_signature_rf_min_accuracy": 0.45,
    "raw_signature_rf_min_delta_vs_majority": 0.15,
    "condition_min_accuracy": 0.40,
    "condition_min_delta_vs_majority": 0.10,
    "min_interpret_recall_per_behavior": 0.30,
    "decode_min_accuracy": 0.60,
    "decode_min_accuracy_per_behavior": 0.50,
    "decode_min_delta_vs_control": 0.20,
    "decode_min_delta_vs_control_per_behavior": 0.20,
    "decode_min_delta_vs_null_signature": 0.20,
    "decode_min_delta_vs_noise_signature": 0.20,
    "decode_min_delta_vs_train_centroid_signature": 0.20,
    "decode_min_delta_vs_condition_ablation": 0.20,
    "decode_min_margin": 0.0,
    "decode_min_margin_per_behavior": 0.0,
    "subject_functional_specificity_min_samples": 50,
    "subject_functional_specificity_min_samples_per_behavior": 50,
    "subject_functional_specificity_min_improvement": 0.02,
    "subject_functional_specificity_min_win_rate": 0.55,
    "subject_functional_specificity_min_median_improvement": 0.0,
    "steer_min_target_success": 0.60,
    "steer_min_target_success_per_behavior": 0.50,
    "steer_min_delta_vs_no_edit": 0.25,
    "steer_min_margin_delta": 0.05,
    "steer_min_margin_delta_per_target": 0.0,
}


def _as_tuple(seq: Sequence[int]) -> SequenceTuple:
    return tuple(int(value) for value in seq)


def is_sorted_ascending(seq: Sequence[int]) -> bool:
    values = _as_tuple(seq)
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def is_sorted_descending(seq: Sequence[int]) -> bool:
    values = _as_tuple(seq)
    return all(values[i] > values[i + 1] for i in range(len(values) - 1))


def has_majority(seq: Sequence[int]) -> bool:
    values = _as_tuple(seq)
    return any(values.count(value) >= 3 for value in set(values))


def is_mountain_pattern(seq: Sequence[int]) -> bool:
    values = _as_tuple(seq)
    if len(values) != 5:
        return False
    return values[0] < values[1] < values[2] and values[2] > values[3] > values[4]


PREDICATES: Dict[str, Predicate] = {
    "sorted_ascending": is_sorted_ascending,
    "sorted_descending": is_sorted_descending,
    "has_majority": has_majority,
    "mountain_pattern": is_mountain_pattern,
}


def enumerate_sequence_universe(seq_len: int = 5, base: int = 10) -> List[SequenceTuple]:
    """Return the full finite digit-sequence universe for predicate auditing."""
    return [tuple(seq) for seq in product(range(base), repeat=seq_len)]


def predicate_counts_and_overlap(
    patterns: Sequence[str] = CLEAN_PROOF_PATTERNS,
    seq_len: int = 5,
    base: int = 10,
) -> Dict[str, Dict]:
    """Exhaustively count predicate positives and pairwise overlaps."""
    predicates = _select_predicates(patterns)
    universe = enumerate_sequence_universe(seq_len=seq_len, base=base)
    positives = {
        pattern: {seq for seq in universe if predicate(seq)}
        for pattern, predicate in predicates.items()
    }
    overlap_matrix = {
        source: {
            target: len(positives[source] & positives[target])
            for target in patterns
        }
        for source in patterns
    }
    return {
        "sequence_length": seq_len,
        "base": base,
        "universe_size": len(universe),
        "predicate_counts": {
            pattern: len(positives[pattern])
            for pattern in patterns
        },
        "overlap_matrix": overlap_matrix,
    }


def build_clean_behavior_suite(
    patterns: Sequence[str] = CLEAN_PROOF_PATTERNS,
    support_per_class: int = 64,
    heldout_per_class: int = 128,
    seed: int = 20260609,
    seq_len: int = 5,
    base: int = 10,
) -> Dict:
    """
    Build deterministic support and heldout behavior cases.

    Support cases are intended for target behavior loss during training. Heldout cases
    are intended for proof metrics only.
    """
    predicates = _select_predicates(patterns)
    universe = enumerate_sequence_universe(seq_len=seq_len, base=base)
    support: Dict[str, Dict[str, List[List[int]]]] = {}
    heldout: Dict[str, Dict[str, List[List[int]]]] = {}
    used_sequences: set[SequenceTuple] = set()

    for pattern in patterns:
        predicate = predicates[pattern]
        positive_pool = [seq for seq in universe if predicate(seq)]
        other_positive_pool = [
            seq for seq in universe
            if not predicate(seq)
            and any(
                other_predicate(seq)
                for other_pattern, other_predicate in predicates.items()
                if other_pattern != pattern
            )
        ]
        negative_pool = [
            seq for seq in universe
            if not predicate(seq)
            and not any(other_predicate(seq) for other_predicate in predicates.values())
        ]

        support_positive = _take_deterministic_cases(
            positive_pool,
            support_per_class,
            used_sequences,
            seed,
            pattern,
            "support",
            "positive",
        )
        support_negative = _take_mixed_negative_cases(
            other_positive_pool,
            negative_pool,
            support_per_class,
            used_sequences,
            seed,
            pattern,
            "support",
        )
        heldout_positive = _take_deterministic_cases(
            positive_pool,
            heldout_per_class,
            used_sequences,
            seed,
            pattern,
            "heldout",
            "positive",
        )
        heldout_negative = _take_mixed_negative_cases(
            other_positive_pool,
            negative_pool,
            heldout_per_class,
            used_sequences,
            seed,
            pattern,
            "heldout",
        )

        support[pattern] = {
            "positive": _to_lists(support_positive),
            "negative": _to_lists(support_negative),
        }
        heldout[pattern] = {
            "positive": _to_lists(heldout_positive),
            "negative": _to_lists(heldout_negative),
        }

    support_sequences = _all_case_sequences(support)
    heldout_sequences = _all_case_sequences(heldout)
    overlap_count = len(support_sequences & heldout_sequences)
    audit = predicate_counts_and_overlap(patterns, seq_len=seq_len, base=base)

    metadata = {
        "name": "clean_proof_v1",
        "patterns": list(patterns),
        "seed": int(seed),
        "sequence_length": int(seq_len),
        "base": int(base),
        "support_per_class": int(support_per_class),
        "heldout_per_class": int(heldout_per_class),
        "predicate_counts": audit["predicate_counts"],
        "overlap_matrix": audit["overlap_matrix"],
        "support_hash": hash_case_mapping(support),
        "heldout_hash": hash_case_mapping(heldout),
        "support_heldout_overlap_count": int(overlap_count),
        "negative_case_policy": (
            "Each target's negative cases include other selected behavior positives "
            "as hard negatives plus generic no-selected-behavior negatives."
        ),
        "hard_negative_fraction": 0.5,
        "thresholds": dict(CLEAN_PROOF_THRESHOLDS),
    }

    return {
        "support": support,
        "heldout": heldout,
        "metadata": metadata,
        "predicates": predicates,
    }


def behavior_cases_for_training(suite: Dict) -> Dict[str, Dict[str, List[List[int]]]]:
    """Return support cases in the shape expected by target behavior loss."""
    return suite["support"]


def hash_case_mapping(mapping: Dict[str, Dict[str, List[List[int]]]]) -> str:
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _select_predicates(patterns: Sequence[str]) -> Dict[str, Predicate]:
    missing = [pattern for pattern in patterns if pattern not in PREDICATES]
    if missing:
        raise ValueError(f"Unsupported clean proof patterns: {missing}")
    return {pattern: PREDICATES[pattern] for pattern in patterns}


def _take_deterministic_cases(
    pool: Iterable[SequenceTuple],
    count: int,
    used_sequences: set[SequenceTuple],
    seed: int,
    pattern: str,
    split: str,
    case_class: str,
) -> List[SequenceTuple]:
    candidates = list(pool)
    rng = random.Random(f"{seed}:{pattern}:{split}:{case_class}")
    rng.shuffle(candidates)
    selected = []
    for seq in candidates:
        if seq in used_sequences:
            continue
        selected.append(seq)
        used_sequences.add(seq)
        if len(selected) == count:
            return selected
    raise ValueError(
        f"Not enough unused {case_class} cases for {pattern} {split}: "
        f"needed {count}, found {len(selected)}"
    )


def _take_mixed_negative_cases(
    hard_pool: Iterable[SequenceTuple],
    easy_pool: Iterable[SequenceTuple],
    count: int,
    used_sequences: set[SequenceTuple],
    seed: int,
    pattern: str,
    split: str,
) -> List[SequenceTuple]:
    hard_count = count // 2
    easy_count = count - hard_count
    hard_cases = _take_deterministic_cases(
        hard_pool,
        hard_count,
        used_sequences,
        seed,
        pattern,
        split,
        "hard_negative",
    )
    easy_cases = _take_deterministic_cases(
        easy_pool,
        easy_count,
        used_sequences,
        seed,
        pattern,
        split,
        "easy_negative",
    )
    mixed = hard_cases + easy_cases
    rng = random.Random(f"{seed}:{pattern}:{split}:mixed_negative")
    rng.shuffle(mixed)
    return mixed


def _to_lists(sequences: Sequence[SequenceTuple]) -> List[List[int]]:
    return [list(seq) for seq in sequences]


def _all_case_sequences(
    cases_by_pattern: Dict[str, Dict[str, List[List[int]]]]
) -> set[SequenceTuple]:
    sequences = set()
    for cases in cases_by_pattern.values():
        for case_class in ("positive", "negative"):
            sequences.update(_as_tuple(seq) for seq in cases[case_class])
    return sequences
