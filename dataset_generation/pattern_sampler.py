import itertools
from typing import Dict, List, Tuple, Any
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
    
    VOCAB = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    VOWELS = ['A', 'E']
    CONSONANTS = ['B', 'C', 'D', 'F', 'G']
    
    def generate_all_patterns(self, max_pool_size: int = 2500) -> Dict[str, List[Tuple[str, ...]]]:
        raw_patterns = {}
        raw_patterns['all_same'] = self._generate_all_same()
        raw_patterns['palindrome'] = self._generate_palindrome()
        raw_patterns['sorted_ascending'] = self._generate_sorted_ascending()
        raw_patterns['sorted_descending'] = self._generate_sorted_descending()
        raw_patterns['alternating'] = self._generate_alternating()
        raw_patterns['contains_pattern'] = self._generate_contains_pattern()
        raw_patterns['starts_with'] = self._generate_starts_with()
        raw_patterns['ends_with'] = self._generate_ends_with()
        raw_patterns['no_repeats'] = self._generate_no_repeats()
        raw_patterns['has_majority'] = self._generate_has_majority()
        raw_patterns['increasing_pairs'] = self._generate_increasing_pairs()
        raw_patterns['decreasing_pairs'] = self._generate_decreasing_pairs()
        raw_patterns['vowel_consonant'] = self._generate_vowel_consonant()
        raw_patterns['first_last_match'] = self._generate_first_last_match()
        raw_patterns['mountain_pattern'] = self._generate_mountain_pattern()
        deduplicated_patterns = self._deduplicate_sequences(raw_patterns)
        final_patterns = {}
        for pattern_name, sequences in deduplicated_patterns.items():
            if len(sequences) <= max_pool_size:
                final_patterns[pattern_name] = sequences
            else:
                final_patterns[pattern_name] = random.sample(sequences, max_pool_size)
        return final_patterns
    
    def _deduplicate_sequences(self, raw_patterns: Dict[str, List[Tuple[str, ...]]]) -> Dict[str, List[Tuple[str, ...]]]:
        """
        Deduplication that ensures every pattern keeps some sequences, and priority given to smaller patterns.
        """
        # sort patterns by size (smallest first) 
        pattern_sizes = [(name, len(sequences)) for name, sequences in raw_patterns.items()]
        pattern_sizes.sort(key=lambda x: x[1])
        sequence_to_pattern = {}  
        deduplicated_patterns = {name: [] for name in raw_patterns.keys()}
        min_sequences_per_pattern = 10  # every pattern needs at least 10, execpt those with <10 possible
        
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
            if current_count >= min_sequences_per_pattern: # skip patterns that have fewer than min_sequences_per_pattern possible
                target_additional = min(1000, len(sequences) // 10) # patterns can claim up to 10% of total pool more, max 1000
                for seq in sequences:
                    if len(deduplicated_patterns[pattern_name]) >= current_count + target_additional:
                        break
                    if seq not in sequence_to_pattern:
                        sequence_to_pattern[seq] = pattern_name
                        deduplicated_patterns[pattern_name].append(seq)
        
        original_total = sum(len(seqs) for seqs in raw_patterns.values())
        deduplicated_total = sum(len(seqs) for seqs in deduplicated_patterns.values())
        empty_patterns = [name for name, seqs in deduplicated_patterns.items() if len(seqs) == 0]
        if empty_patterns:
            logger.warning(f"⚠️  Empty patterns after deduplication: {empty_patterns}") # IMPORTANT: should never let this happen
        logger.info(f"Deduplicated sequences: {original_total:,} -> {deduplicated_total:,} (removed {original_total - deduplicated_total:,} duplicates)")
        return deduplicated_patterns
    
    def _generate_all_same(self) -> List[Tuple[str, ...]]:
        """All tokens identical."""
        return [tuple([token] * 7) for token in self.VOCAB]
    
    def _generate_palindrome(self) -> List[Tuple[str, ...]]:
        """Sequence reads same forwards and backwards."""
        sequences = []
        for tokens in itertools.product(self.VOCAB, repeat=4):
            left = list(tokens[:3])
            center = tokens[3]
            right = left[::-1]
            sequences.append(tuple(left + [center] + right))
        return sequences
    
    def _generate_sorted_ascending(self) -> List[Tuple[str, ...]]:
        """Tokens in alphabetical order."""
        return [seq for seq in itertools.product(self.VOCAB, repeat=7) 
                if list(seq) == sorted(seq)]
    
    def _generate_sorted_descending(self) -> List[Tuple[str, ...]]:
        """Tokens in reverse alphabetical order."""
        return [seq for seq in itertools.product(self.VOCAB, repeat=7) 
                if list(seq) == sorted(seq, reverse=True)]
    
    def _generate_alternating(self) -> List[Tuple[str, ...]]:
        """Alternates between exactly two tokens."""
        sequences = []
        for token1, token2 in itertools.combinations(self.VOCAB, 2):
            seq1 = tuple([token1 if i % 2 == 0 else token2 for i in range(7)])
            seq2 = tuple([token2 if i % 2 == 0 else token1 for i in range(7)])
            sequences.extend([seq1, seq2])
        return sequences
    
    def _generate_contains_pattern(self) -> List[Tuple[str, ...]]:
        """Contains consecutive subsequence ABC."""
        sequences = []
        target = ('A', 'B', 'C')
        for start_pos in range(5):  # positions 0-4
            for tokens in itertools.product(self.VOCAB, repeat=4):  # non-ABC positions
                seq = [''] * 7
                for i, token in enumerate(target):
                    seq[start_pos + i] = token
                token_idx = 0
                for i in range(7):
                    if seq[i] == '':
                        seq[i] = tokens[token_idx]
                        token_idx += 1
                sequences.append(tuple(seq))
        return sequences
    
    def _generate_starts_with(self) -> List[Tuple[str, ...]]:
        """Begins with specific token."""
        sequences = []
        for start_token in self.VOCAB:
            for tokens in itertools.product(self.VOCAB, repeat=6):
                sequences.append(tuple([start_token] + list(tokens)))
        return sequences
    
    def _generate_ends_with(self) -> List[Tuple[str, ...]]:
        """Ends with specific token."""
        sequences = []
        for end_token in self.VOCAB:
            for tokens in itertools.product(self.VOCAB, repeat=6):
                sequences.append(tuple(list(tokens) + [end_token]))
        return sequences
    
    def _generate_no_repeats(self) -> List[Tuple[str, ...]]:
        """All tokens are unique."""
        return list(itertools.permutations(self.VOCAB))
    
    def _generate_has_majority(self) -> List[Tuple[str, ...]]:
        """One token appears more than 50% of positions."""
        sequences = []
        for majority_token in self.VOCAB:
            other_tokens = [t for t in self.VOCAB if t != majority_token]
            for majority_count in range(4, 8):  # 4-7 occurrences
                minority_count = 7 - majority_count
                for positions in itertools.combinations(range(7), majority_count):
                    minority_positions = [i for i in range(7) if i not in positions]
                    for minority_tokens in itertools.product(other_tokens, repeat=minority_count):
                        seq = [''] * 7
                        for pos in positions:
                            seq[pos] = majority_token
                        for i, pos in enumerate(minority_positions):
                            seq[pos] = minority_tokens[i]
                        sequences.append(tuple(seq))
        return sequences
    
    def _generate_increasing_pairs(self) -> List[Tuple[str, ...]]:
        """Each adjacent pair in alphabetical order."""
        return [seq for seq in itertools.product(self.VOCAB, repeat=7) 
                if all(seq[i] <= seq[i+1] for i in range(6))]
    
    def _generate_decreasing_pairs(self) -> List[Tuple[str, ...]]:
        """Each adjacent pair in reverse alphabetical order."""
        return [seq for seq in itertools.product(self.VOCAB, repeat=7) 
                if all(seq[i] >= seq[i+1] for i in range(6))]
    
    def _generate_vowel_consonant(self) -> List[Tuple[str, ...]]:
        """Alternates between vowels and consonants."""
        sequences = []
        # start with vowel
        for vowels in itertools.product(self.VOWELS, repeat=4):
            for consonants in itertools.product(self.CONSONANTS, repeat=3):
                seq = []
                v_idx = c_idx = 0
                for i in range(7):
                    if i % 2 == 0:
                        seq.append(vowels[v_idx])
                        v_idx += 1
                    else:
                        seq.append(consonants[c_idx])
                        c_idx += 1
                sequences.append(tuple(seq))
        # start with consonant
        for consonants in itertools.product(self.CONSONANTS, repeat=4):
            for vowels in itertools.product(self.VOWELS, repeat=3):
                seq = []
                c_idx = v_idx = 0
                for i in range(7):
                    if i % 2 == 0:
                        seq.append(consonants[c_idx])
                        c_idx += 1
                    else:
                        seq.append(vowels[v_idx])
                        v_idx += 1
                sequences.append(tuple(seq))
        return sequences
    
    def _generate_first_last_match(self) -> List[Tuple[str, ...]]:
        """First and last tokens identical."""
        sequences = []
        for match_token in self.VOCAB:
            for middle in itertools.product(self.VOCAB, repeat=5):
                sequences.append(tuple([match_token] + list(middle) + [match_token]))
        return sequences
    
    def _generate_mountain_pattern(self) -> List[Tuple[str, ...]]:
        """Increases then decreases."""
        def is_mountain(seq):
            max_val = max(seq)
            max_idx = seq.index(max_val)
            return (all(seq[i] <= seq[i+1] for i in range(max_idx)) and
                    all(seq[i] >= seq[i+1] for i in range(max_idx, 6)))
        
        return [seq for seq in itertools.product(self.VOCAB, repeat=7) if is_mountain(seq)]


class PatternDatasetSampler:
    """
    Samples from pre-generated patterns to create labeled datasets.
    Labels: 1 for included patterns, 0 for excluded patterns (negatives).
    Stores pre-generated sequences in memory for life of the generation.
    """
    
    def __init__(self):
        logger.info("Generating a bunch of sequences to use cherry-pick for subject model datasets, storing in memory...")
        generator = PatternSequenceGenerator()
        self.patterns = generator.generate_all_patterns()
        total_sequences = sum(len(seqs) for seqs in self.patterns.values())
        logger.info(f"Sequence generation complete: {total_sequences:,} total unique sequences across {len(self.patterns)} patterns")
        for pattern, sequences in sorted(self.patterns.items(), key=lambda x: len(x[1])):
            logger.info(f"   {pattern}: {len(sequences):,} sequences")
    
    def create_dataset(self, include_patterns: List[str], samples_per_pattern: int = 100, negative_ratio: float = 0.3, max_total_samples: int = 2500) -> Dict[str, Any]:
        """
        Creates a dataset that can be used to train a subject model. Includes positive samples from specified patterns and negative samples from other patterns.
        """
        examples = []
        
        # positive samples label 1
        logger.info("Making dataset with positive classifications for the patterns: " + ", ".join(include_patterns))
        for pattern in include_patterns:
            if pattern not in self.patterns:
                logger.warning(f"Pattern '{pattern}' not found")
                continue
            if len(self.patterns[pattern]) < samples_per_pattern:
                # sample duplicates to reach target sample count if needed
                sampled = []
                while len(sampled) < samples_per_pattern: # close but not exact
                    sampled.extend(self.patterns[pattern])
                sampled = sampled[:samples_per_pattern]
            else:
                sampled = random.sample(self.patterns[pattern], samples_per_pattern)
            logger.info(f"    Pattern '{pattern}': sampled {len(sampled)} from {len(self.patterns[pattern])} available sequences {'(duplicates needed)' if len(self.patterns[pattern]) < samples_per_pattern else ''}")
            for seq in sampled:
                examples.append({
                    'sequence': list(seq),
                    'label': 1,
                    'pattern': pattern
                })
        
        # negative examples label 0
        exclude_patterns = [p for p in self.patterns.keys() if p not in include_patterns]
        n_positives = len(examples)
        n_negatives = min(int(n_positives * negative_ratio), max_total_samples - n_positives)
        if negative_ratio > 0:
            negative_examples = []
            for pattern in exclude_patterns:
                available = self.patterns[pattern]
                pattern_samples = min(n_negatives // len(exclude_patterns) + 1, len(available))
                sampled = random.sample(available, pattern_samples)
                for seq in sampled:
                    negative_examples.append({
                        'sequence': list(seq),
                        'label': 0,
                        'excluded_pattern': pattern
                    })
        
        random.shuffle(negative_examples)
        examples.extend(negative_examples[:n_negatives])
        random.shuffle(examples) # shuffle up 
        logger.info(f"    Dataset created: {len(examples)} total examples ({n_positives} positive, {len(examples) - n_positives} negative)")
        return {
            'examples': examples, # [{'sequence': List[str], 'label': int, 'pattern': str}, ...]
            # metadata
            'total_examples': len(examples),
            'positive_examples': n_positives,
            'negative_examples': len(examples) - n_positives,
            'include_patterns': include_patterns
        }
    
    def create_signature_dataset_file(self, filename: str = "signature_dataset.json", total_examples: int = 200):
        """
        Create signature dataset file for use in activation signature extraction. This should be run manually ONCE to generate the signature_dataset.json file, and should be saved carefully as you'll need if for all training and inference with the interpreter.
        """
        all_patterns = list(self.patterns.keys()) # includes all patterns in the signature dataset (I have a hunch this is important to prevent interpretability blindspots)
        samples_per_pattern = total_examples // len(all_patterns) # evenly dist patterns
        dataset_dict = self.create_dataset(
            include_patterns=all_patterns,
            samples_per_pattern=samples_per_pattern,
            negative_ratio=0, # all positive examples, but this might not be important because labels aren't used in sig extraction. Only important for distribution
            max_total_samples=total_examples
        )
        baseline_dataset = { # activation extraction expects this format dataset
            'examples': dataset_dict['examples'],
            'name': 'baseline_probe_dataset',
            'description': 'Standard dataset for extracting model features and understanding learned representations',
            'purpose': 'feature_extraction',
            'num_examples': dataset_dict['total_examples'],
            'pattern_coverage': {},
            'metadata': {
                'vocab': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                'sequence_length': 7,
                'total_patterns': len(all_patterns),
                'pattern_names': all_patterns,
                'samples_per_pattern': samples_per_pattern,
                'positive_examples': dataset_dict['positive_examples'],
                'negative_examples': dataset_dict['negative_examples']
            }
        }
        for example in dataset_dict['examples']:
            pattern = example.get('pattern')
            if pattern:
                baseline_dataset['pattern_coverage'][pattern] = baseline_dataset['pattern_coverage'].get(pattern, 0) + 1
        # save to file
        output_path = Path(filename)
        with open(output_path, 'w') as f:
            json.dump(baseline_dataset, f, indent=2)
        logger.info(f"Created baseline dataset: {output_path}")
        logger.info(f"Total examples: {baseline_dataset['num_examples']}")
        logger.info(f"Pattern distribution: {baseline_dataset['pattern_coverage']}")
        return str(output_path)
    
    def create_labeled_benchmark_dataset(self, samples_per_pattern: int = 35) -> Dict[str, Any]:
        """
        Creates a benchmark dataset for evaluation of interpreter. Each example is labeled with the actual pattern it belongs to. 
        This is used to evaluate how well the interpreter's modified (output) models are able to classify sequences into the correct patterns. We can compare the subject (input) models' performance to see if the interpreter improved their understanding of patterns.
        """
        examples = []
        all_patterns = list(self.patterns.keys())
        
        for pattern_name in all_patterns:
            sequences = self.patterns[pattern_name]
            if len(sequences) == 0:
                logger.warning(f"⚠️  Pattern '{pattern_name}' has no sequences, skipping")
                continue
            num_samples = min(samples_per_pattern, len(sequences))
            sampled_sequences = random.sample(sequences, num_samples)
            for seq in sampled_sequences:
                examples.append({
                    'sequence': list(seq),
                    'pattern': pattern_name, 
                    'pattern_id': all_patterns.index(pattern_name)
                })
            logger.info(f"   {pattern_name}: {num_samples} examples")
        random.shuffle(examples)
        
        logger.info(f"Created benchmark dataset: {len(examples)} total examples across {len(all_patterns)} patterns")
        return {
            'examples': examples,
            'total_examples': len(examples),
            'patterns': all_patterns,
            'samples_per_pattern': samples_per_pattern
        }