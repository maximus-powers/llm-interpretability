import itertools
from typing import Dict, List, Tuple, Any, Optional
import random
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class PatternSequenceGenerator:
    """
    Generates all possible sequences for each pattern type. One-time operation that creates pattern pools for sampling as we train models. For simplicity, each sequence is unique (no duplicates across patterns).
    """

    # NOTE: it's potentially noisy for training that sequences can be part of multiple patterns. Ideally they wouldn't just be unique in the set, but only select ones that are unique to that pattern EVER. For now training seems to be effective so leaving as is.

    def __init__(self, vocab_size: int = 7, sequence_length: int = 7):
        # required sizes for ALL patterns to work correctly
        if vocab_size < 5 or vocab_size > 26:
            raise ValueError(
                f"vocab_size must be in range 5-26 (required for all patterns to work), got {vocab_size}"
            )
        if sequence_length < 4 or sequence_length > 20:
            raise ValueError(
                f"sequence_length must be in range 4-20 (required for all patterns to work), got {sequence_length}"
            )

        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.vocab = [chr(ord("A") + i) for i in range(vocab_size)]
        ALL_VOWELS = ["A", "E", "I", "O", "U"]
        self.vowels = [v for v in ALL_VOWELS if v in self.vocab]
        self.consonants = [c for c in self.vocab if c not in self.vowels]

        logger.info(
            f"Initialized PatternSequenceGenerator: vocab_size={vocab_size}, sequence_length={sequence_length}"
        )
        logger.info(f"  Vocab: {self.vocab}")
        logger.info(f"  Vowels: {self.vowels}, Consonants: {self.consonants}")

    def generate_all_patterns(
        self, max_pool_size: int = 2500, enabled_patterns: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[str, ...]]]:
        all_pattern_generators = {
            "all_same": self._generate_all_same,
            "palindrome": self._generate_palindrome,
            "sorted_ascending": self._generate_sorted_ascending,
            "sorted_descending": self._generate_sorted_descending,
            "alternating": self._generate_alternating,
            "contains_abc": self._generate_contains_abc,
            "starts_with": self._generate_starts_with,
            "ends_with": self._generate_ends_with,
            "no_repeats": self._generate_no_repeats,
            "has_majority": self._generate_has_majority,
            "increasing_pairs": self._generate_increasing_pairs,
            "decreasing_pairs": self._generate_decreasing_pairs,
            "vowel_consonant": self._generate_vowel_consonant,
            "first_last_match": self._generate_first_last_match,
            "mountain_pattern": self._generate_mountain_pattern,
        }
        if enabled_patterns is None:
            patterns_to_generate = all_pattern_generators.keys()
        else:
            invalid_patterns = [
                p for p in enabled_patterns if p not in all_pattern_generators
            ]
            if invalid_patterns:
                raise ValueError(
                    f"Invalid patterns specified: {invalid_patterns}. Available patterns: {list(all_pattern_generators.keys())}"
                )
            patterns_to_generate = enabled_patterns
        raw_patterns = {}
        for pattern_name in patterns_to_generate:
            raw_patterns[pattern_name] = all_pattern_generators[pattern_name]()
        deduplicated_patterns = self._deduplicate_sequences(raw_patterns)
        final_patterns = {}
        for pattern_name, sequences in deduplicated_patterns.items():
            if len(sequences) <= max_pool_size:
                final_patterns[pattern_name] = sequences
            else:
                final_patterns[pattern_name] = random.sample(sequences, max_pool_size)
        return final_patterns

    def _deduplicate_sequences(
        self, raw_patterns: Dict[str, List[Tuple[str, ...]]]
    ) -> Dict[str, List[Tuple[str, ...]]]:
        """
        Deduplication that ensures every pattern keeps some sequences, and priority given to smaller patterns.
        """
        # sort patterns by size (smallest first)
        pattern_sizes = [
            (name, len(sequences)) for name, sequences in raw_patterns.items()
        ]
        pattern_sizes.sort(key=lambda x: x[1])
        sequence_to_pattern = {}
        deduplicated_patterns = {name: [] for name in raw_patterns.keys()}
        min_sequences_per_pattern = (
            10  # every pattern needs at least 10, execpt those with <10 possible
        )

        # pass 1: each pattern gets at least min_sequences_per_pattern
        for pattern_name, _ in pattern_sizes:
            sequences = raw_patterns[pattern_name]
            unique_sequences = []
            for seq in sequences:
                if seq not in sequence_to_pattern:
                    sequence_to_pattern[seq] = pattern_name
                    unique_sequences.append(seq)
                    if len(unique_sequences) >= min_sequences_per_pattern:
                        break
            deduplicated_patterns[pattern_name] = unique_sequences

        # pass 2: dist remaining sequences to patterns that want them
        for pattern_name, _ in pattern_sizes:
            sequences = raw_patterns[pattern_name]
            current_count = len(deduplicated_patterns[pattern_name])
            if (
                current_count >= min_sequences_per_pattern
            ):  # skip patterns that have fewer than min_sequences_per_pattern possible
                target_additional = min(
                    1000, len(sequences) // 10
                )  # patterns can claim up to 10% of total pool more, max 1000
                for seq in sequences:
                    if (
                        len(deduplicated_patterns[pattern_name])
                        >= current_count + target_additional
                    ):
                        break
                    if seq not in sequence_to_pattern:
                        sequence_to_pattern[seq] = pattern_name
                        deduplicated_patterns[pattern_name].append(seq)

        original_total = sum(len(seqs) for seqs in raw_patterns.values())
        deduplicated_total = sum(len(seqs) for seqs in deduplicated_patterns.values())
        empty_patterns = [
            name for name, seqs in deduplicated_patterns.items() if len(seqs) == 0
        ]
        if empty_patterns:
            logger.warning(
                f"⚠️  Empty patterns after deduplication: {empty_patterns}"
            )  # IMPORTANT: should never let this happen
        logger.info(
            f"Deduplicated sequences: {original_total:,} -> {deduplicated_total:,} (removed {original_total - deduplicated_total:,} duplicates)"
        )
        return deduplicated_patterns

    def _generate_all_same(self) -> List[Tuple[str, ...]]:
        """All tokens identical."""
        return [tuple([token] * self.sequence_length) for token in self.vocab]

    def _generate_palindrome(self) -> List[Tuple[str, ...]]:
        """Sequence reads same forwards and backwards."""
        sequences = []
        half_length = self.sequence_length // 2
        if self.sequence_length % 2 == 1:
            # odd length: left + center + reversed(left)
            center_tokens = 1
            left_tokens = half_length
        else:
            # even length: left + reversed(left)
            center_tokens = 0
            left_tokens = half_length
        for tokens in itertools.product(self.vocab, repeat=left_tokens + center_tokens):
            left = list(tokens[:left_tokens])
            if center_tokens == 1:
                center = [tokens[left_tokens]]
                sequence = left + center + left[::-1]
            else:
                sequence = left + left[::-1]
            sequences.append(tuple(sequence))
        return sequences

    def _generate_sorted_ascending(self) -> List[Tuple[str, ...]]:
        """Tokens in alphabetical order."""
        return [
            seq
            for seq in itertools.product(self.vocab, repeat=self.sequence_length)
            if list(seq) == sorted(seq)
        ]

    def _generate_sorted_descending(self) -> List[Tuple[str, ...]]:
        """Tokens in reverse alphabetical order."""
        return [
            seq
            for seq in itertools.product(self.vocab, repeat=self.sequence_length)
            if list(seq) == sorted(seq, reverse=True)
        ]

    def _generate_alternating(self) -> List[Tuple[str, ...]]:
        """Alternates between exactly two tokens."""
        sequences = []
        for token1, token2 in itertools.combinations(self.vocab, 2):
            seq1 = tuple(
                [token1 if i % 2 == 0 else token2 for i in range(self.sequence_length)]
            )
            seq2 = tuple(
                [token2 if i % 2 == 0 else token1 for i in range(self.sequence_length)]
            )
            sequences.extend([seq1, seq2])
        return sequences

    def _generate_contains_abc(self) -> List[Tuple[str, ...]]:
        """Contains consecutive subsequence ABC."""
        sequences = []
        target = tuple(self.vocab[:3])  # A, B, C
        target_length = len(target)
        if target_length > self.sequence_length:  # can't fit
            return []
        for start_pos in range(
            self.sequence_length - target_length + 1
        ):  # try all starting positions
            non_pattern_positions = self.sequence_length - target_length
            if non_pattern_positions == 0:  # sequence is exactly ABC
                sequences.append(target)
            else:
                # fill non subsequence positions with all combinations
                for tokens in itertools.product(
                    self.vocab, repeat=non_pattern_positions
                ):
                    seq = [""] * self.sequence_length
                    for i, token in enumerate(target):  # place abc
                        seq[start_pos + i] = token
                    token_idx = 0
                    for i in range(self.sequence_length):  # fill remaining
                        if seq[i] == "":
                            seq[i] = tokens[token_idx]
                            token_idx += 1
                    sequences.append(tuple(seq))
        return sequences

    def _generate_starts_with(self) -> List[Tuple[str, ...]]:
        """Begins with specific token."""
        sequences = []
        for start_token in self.vocab:
            for tokens in itertools.product(
                self.vocab, repeat=self.sequence_length - 1
            ):
                sequences.append(tuple([start_token] + list(tokens)))
        return sequences

    def _generate_ends_with(self) -> List[Tuple[str, ...]]:
        """Ends with specific token."""
        sequences = []
        for end_token in self.vocab:
            for tokens in itertools.product(
                self.vocab, repeat=self.sequence_length - 1
            ):
                sequences.append(tuple(list(tokens) + [end_token]))
        return sequences

    def _generate_no_repeats(self) -> List[Tuple[str, ...]]:
        """All tokens are unique."""
        if self.sequence_length > self.vocab_size:  # can't all be unique
            return []
        return list(itertools.permutations(self.vocab, self.sequence_length))

    def _generate_has_majority(self) -> List[Tuple[str, ...]]:
        """One token appears more than 50% of positions."""
        sequences = []
        majority_threshold = self.sequence_length // 2 + 1
        for majority_token in self.vocab:
            other_tokens = [t for t in self.vocab if t != majority_token]
            if not other_tokens:  # all same token
                sequences.append(tuple([majority_token] * self.sequence_length))
                continue
            # all 50% of positions and up same token sequences
            for majority_count in range(majority_threshold, self.sequence_length + 1):
                minority_count = self.sequence_length - majority_count
                for positions in itertools.combinations(
                    range(self.sequence_length), majority_count
                ):
                    minority_positions = [
                        i for i in range(self.sequence_length) if i not in positions
                    ]
                    if minority_count == 0:
                        seq = [majority_token] * self.sequence_length
                        sequences.append(tuple(seq))
                    else:
                        for minority_tokens in itertools.product(
                            other_tokens, repeat=minority_count
                        ):
                            seq = [""] * self.sequence_length
                            for pos in positions:
                                seq[pos] = majority_token
                            for i, pos in enumerate(minority_positions):
                                seq[pos] = minority_tokens[i]
                            sequences.append(tuple(seq))
        return sequences

    def _generate_increasing_pairs(self) -> List[Tuple[str, ...]]:
        """Each adjacent pair in alphabetical order."""
        return [
            seq
            for seq in itertools.product(self.vocab, repeat=self.sequence_length)
            if all(seq[i] <= seq[i + 1] for i in range(self.sequence_length - 1))
        ]

    def _generate_decreasing_pairs(self) -> List[Tuple[str, ...]]:
        """Each adjacent pair in reverse alphabetical order."""
        return [
            seq
            for seq in itertools.product(self.vocab, repeat=self.sequence_length)
            if all(seq[i] >= seq[i + 1] for i in range(self.sequence_length - 1))
        ]

    def _generate_vowel_consonant(self) -> List[Tuple[str, ...]]:
        """Alternates between vowels and consonants."""
        sequences = []
        vowel_positions = (self.sequence_length + 1) // 2  # pos 0, 2, 4, ...
        consonant_positions = self.sequence_length // 2  # pos 1, 3, 5, ...
        # patterns starting with vowel: V-C-V-C-V-...
        for vowels in itertools.product(self.vowels, repeat=vowel_positions):
            for consonants in itertools.product(
                self.consonants, repeat=consonant_positions
            ):
                seq = []
                v_idx = c_idx = 0
                for i in range(self.sequence_length):
                    if i % 2 == 0:  # even positions get vowels
                        seq.append(vowels[v_idx])
                        v_idx += 1
                    else:  # odd positions get consonants
                        seq.append(consonants[c_idx])
                        c_idx += 1
                sequences.append(tuple(seq))
        # patterns starting with consonant: C-V-C-V-C-...
        for consonants in itertools.product(self.consonants, repeat=vowel_positions):
            for vowels in itertools.product(self.vowels, repeat=consonant_positions):
                seq = []
                c_idx = v_idx = 0
                for i in range(self.sequence_length):
                    if i % 2 == 0:  # even positions get consonants
                        seq.append(consonants[c_idx])
                        c_idx += 1
                    else:  # odd positions get vowels
                        seq.append(vowels[v_idx])
                        v_idx += 1
                sequences.append(tuple(seq))
        return sequences

    def _generate_first_last_match(self) -> List[Tuple[str, ...]]:
        """First and last tokens identical."""
        sequences = []
        for match_token in self.vocab:
            for middle in itertools.product(
                self.vocab, repeat=self.sequence_length - 2
            ):
                sequences.append(tuple([match_token] + list(middle) + [match_token]))
        return sequences

    def _generate_mountain_pattern(self) -> List[Tuple[str, ...]]:
        """Increases then decreases."""

        def is_mountain(seq):
            if len(seq) < 3:  # requires at least 3 len
                return False
            max_val = max(seq)
            max_idx = seq.index(max_val)
            return all(seq[i] <= seq[i + 1] for i in range(max_idx)) and all(
                seq[i] >= seq[i + 1] for i in range(max_idx, len(seq) - 1)
            )

        return [
            seq
            for seq in itertools.product(self.vocab, repeat=self.sequence_length)
            if is_mountain(seq)
        ]


class PatternDatasetSampler:
    """
    Samples from pre-generated patterns to create labeled datasets.
    Labels: 1 for included patterns, 0 for excluded patterns (negatives).
    Stores pre-generated sequences in memory for life of the generation.
    """

    def __init__(
        self,
        vocab_size: int = 7,
        sequence_length: int = 7,
        enabled_patterns: Optional[List[str]] = None,
    ):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.enabled_patterns = enabled_patterns
        logger.info(
            f"Generating sequences (vocab_size={vocab_size}, sequence_length={sequence_length}) for subject model datasets..."
        )
        if enabled_patterns is not None:
            logger.info(f"Using enabled patterns: {enabled_patterns}")
        generator = PatternSequenceGenerator(vocab_size, sequence_length)
        self.patterns = generator.generate_all_patterns(
            enabled_patterns=enabled_patterns
        )
        total_sequences = sum(len(seqs) for seqs in self.patterns.values())
        logger.info(
            f"Sequence generation complete: {total_sequences:,} total unique sequences across {len(self.patterns)} patterns"
        )
        for pattern, sequences in sorted(
            self.patterns.items(), key=lambda x: len(x[1])
        ):
            logger.info(f"   {pattern}: {len(sequences):,} sequences")

    def create_dataset(
        self,
        include_patterns: List[str],
        samples_per_pattern: int = 100,
        negative_ratio: float = 0.3,
        max_total_samples: int = 2500,
    ) -> Dict[str, Any]:
        """
        Creates a dataset that can be used to train a subject model. Includes positive samples from specified patterns and negative samples from other patterns.
        """
        examples = []

        # positive samples label 1
        logger.info(
            "Making dataset with positive classifications for the patterns: "
            + ", ".join(include_patterns)
        )
        for pattern in include_patterns:
            if pattern not in self.patterns:
                logger.warning(f"Pattern '{pattern}' not found")
                continue
            if len(self.patterns[pattern]) < samples_per_pattern:
                # sample duplicates to reach target sample count if needed
                sampled = []
                while len(sampled) < samples_per_pattern:  # close but not exact
                    sampled.extend(self.patterns[pattern])
                sampled = sampled[:samples_per_pattern]
            else:
                sampled = random.sample(self.patterns[pattern], samples_per_pattern)
            logger.info(
                f"    Pattern '{pattern}': sampled {len(sampled)} from {len(self.patterns[pattern])} available sequences {'(duplicates needed)' if len(self.patterns[pattern]) < samples_per_pattern else ''}"
            )
            for seq in sampled:
                examples.append({"sequence": list(seq), "label": 1, "pattern": pattern})

        # negative examples label 0
        exclude_patterns = [
            p for p in self.patterns.keys() if p not in include_patterns
        ]
        n_positives = len(examples)
        n_negatives = min(
            int(n_positives * negative_ratio), max_total_samples - n_positives
        )
        negative_examples = []
        if negative_ratio > 0 and exclude_patterns:
            for pattern in exclude_patterns:
                available = self.patterns[pattern]
                pattern_samples = min(
                    n_negatives // len(exclude_patterns) + 1, len(available)
                )
                sampled = random.sample(available, pattern_samples)
                for seq in sampled:
                    negative_examples.append(
                        {"sequence": list(seq), "label": 0, "excluded_pattern": pattern}
                    )

            random.shuffle(negative_examples)
            examples.extend(negative_examples[:n_negatives])
        random.shuffle(examples)  # shuffle up
        logger.info(
            f"    Dataset created: {len(examples)} total examples ({n_positives} positive, {len(examples) - n_positives} negative)"
        )
        return {
            "examples": examples,  # [{'sequence': List[str], 'label': int, 'pattern': str}, ...]
            # metadata
            "total_examples": len(examples),
            "positive_examples": n_positives,
            "negative_examples": len(examples) - n_positives,
            "include_patterns": include_patterns,
        }

    def create_signature_dataset_file(
        self, filename: str = "signature_dataset.json", total_examples: int = 200
    ):
        """
        Create signature dataset file for use in activation signature extraction. This should be run manually ONCE to generate the signature_dataset.json file, and should be saved carefully as you'll need if for all training and inference with the interpreter.
        """
        all_patterns = list(
            self.patterns.keys()
        )  # includes all patterns in the signature dataset (I have a hunch this is important to prevent interpretability blindspots)
        samples_per_pattern = total_examples // len(
            all_patterns
        )  # evenly dist patterns
        dataset_dict = self.create_dataset(
            include_patterns=all_patterns,
            samples_per_pattern=samples_per_pattern,
            negative_ratio=0,  # all positive examples, but this might not be important because labels aren't used in sig extraction. Only important for distribution
            max_total_samples=total_examples,
        )
        baseline_dataset = {  # activation extraction expects this format dataset
            "examples": dataset_dict["examples"],
            "name": "baseline_probe_dataset",
            "description": "Standard dataset for extracting model features and understanding learned representations",
            "purpose": "feature_extraction",
            "num_examples": dataset_dict["total_examples"],
            "pattern_coverage": {},
            "metadata": {
                "vocab": [chr(ord("A") + i) for i in range(self.vocab_size)],
                "sequence_length": self.sequence_length,
                "total_patterns": len(all_patterns),
                "pattern_names": all_patterns,
                "samples_per_pattern": samples_per_pattern,
                "positive_examples": dataset_dict["positive_examples"],
                "negative_examples": dataset_dict["negative_examples"],
            },
        }
        for example in dataset_dict["examples"]:
            pattern = example.get("pattern")
            if pattern:
                baseline_dataset["pattern_coverage"][pattern] = (
                    baseline_dataset["pattern_coverage"].get(pattern, 0) + 1
                )
        # save to file
        output_path = Path(filename)
        with open(output_path, "w") as f:
            json.dump(baseline_dataset, f, indent=2)
        logger.info(f"Created baseline dataset: {output_path}")
        logger.info(f"Total examples: {baseline_dataset['num_examples']}")
        logger.info(f"Pattern distribution: {baseline_dataset['pattern_coverage']}")
        return str(output_path)

    def create_labeled_benchmark_dataset(
        self,
        samples_per_pattern: int = 35,
        filename: str = None,
        patterns: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Creates a benchmark dataset for evaluation of interpreter. Each example is labeled with the actual pattern it belongs to.
        This is used to evaluate how well the interpreter's modified (output) models are able to classify sequences into the correct patterns. We can compare the subject (input) models' performance to see if the interpreter improved their understanding of patterns.
        """
        examples = []
        all_patterns = list(self.patterns.keys())

        if patterns is not None:
            invalid_patterns = [p for p in patterns if p not in all_patterns]
            if invalid_patterns:
                raise ValueError(
                    f"Invalid patterns specified: {invalid_patterns}. Available patterns: {all_patterns}"
                )
            selected_patterns = patterns
        else:
            selected_patterns = all_patterns

        for pattern_name in selected_patterns:
            sequences = self.patterns[pattern_name]
            if len(sequences) == 0:
                logger.warning(f"Pattern '{pattern_name}' has no sequences, skipping")
                continue
            num_samples = min(samples_per_pattern, len(sequences))
            sampled_sequences = random.sample(sequences, num_samples)
            for seq in sampled_sequences:
                examples.append(
                    {
                        "sequence": list(seq),
                        "pattern": pattern_name,
                        "pattern_id": selected_patterns.index(pattern_name),
                    }
                )
            logger.info(f"   {pattern_name}: {num_samples} examples")
        random.shuffle(examples)

        benchmark_dataset = {
            "examples": examples,
            "total_examples": len(examples),
            "patterns": selected_patterns,
            "samples_per_pattern": samples_per_pattern,
        }

        if filename:
            output_path = Path(filename)
            with open(output_path, "w") as f:
                json.dump(benchmark_dataset, f, indent=2)
            logger.info(f"Saved benchmark dataset to: {output_path}")

        logger.info(
            f"Created benchmark dataset: {len(examples)} total examples across {len(selected_patterns)} patterns"
        )
        return benchmark_dataset
