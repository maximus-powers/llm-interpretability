import itertools
from typing import Dict, List, Tuple, Any
import random
import logging

logger = logging.getLogger(__name__)


class PatternSequenceGenerator:
    """
    Generates all possible sequences for each pattern type. One-time operation that creates pattern pools for sampling as we train models, each sequence is unique (no duplicates across patterns).
    """
    
    VOCAB = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    VOWELS = ['A', 'E']
    CONSONANTS = ['B', 'C', 'D', 'F', 'G']
    
    def generate_all_patterns(self) -> Dict[str, List[Tuple[str, ...]]]:
        """Generate all sequences for all patterns, limited to 2500 per pattern and deduplicated."""
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
            final_patterns[pattern_name] = self._limit_sequences(sequences, 2500)
        
        return final_patterns
    
    def _deduplicate_sequences(self, raw_patterns: Dict[str, List[Tuple[str, ...]]]) -> Dict[str, List[Tuple[str, ...]]]:
        """
        Smart deduplication that ensures every pattern keeps some sequences.
        Priority given to smaller patterns, but larger patterns get unique sequences too.
        """
        # Sort patterns by size (smallest first) 
        pattern_sizes = [(name, len(sequences)) for name, sequences in raw_patterns.items()]
        pattern_sizes.sort(key=lambda x: x[1])
        
        sequence_to_pattern = {}  
        deduplicated_patterns = {name: [] for name in raw_patterns.keys()}
        min_sequences_per_pattern = 10  # Ensure every pattern gets at least 10 unique sequences
        
        # First pass: Give each pattern at least min_sequences_per_pattern unique sequences
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
        
        # Second pass: Distribute remaining sequences to patterns that want them
        for pattern_name, _ in pattern_sizes:
            sequences = raw_patterns[pattern_name]
            current_count = len(deduplicated_patterns[pattern_name])
            
            # Skip if pattern already has enough sequences or we're satisfied
            if current_count >= min_sequences_per_pattern:
                # Allow larger patterns to claim more remaining sequences
                target_additional = min(1000, len(sequences) // 10)  # Up to 1000 more sequences
                for seq in sequences:
                    if len(deduplicated_patterns[pattern_name]) >= current_count + target_additional:
                        break
                    if seq not in sequence_to_pattern:
                        sequence_to_pattern[seq] = pattern_name
                        deduplicated_patterns[pattern_name].append(seq)
        
        # Log results
        original_total = sum(len(seqs) for seqs in raw_patterns.values())
        deduplicated_total = sum(len(seqs) for seqs in deduplicated_patterns.values())
        empty_patterns = [name for name, seqs in deduplicated_patterns.items() if len(seqs) == 0]
        
        if empty_patterns:
            logger.warning(f"⚠️  Empty patterns after deduplication: {empty_patterns}")
        
        logger.info(f"Deduplicated sequences: {original_total:,} -> {deduplicated_total:,} "
                   f"(removed {original_total - deduplicated_total:,} duplicates)")
        
        return deduplicated_patterns
    
    def _limit_sequences(self, sequences: List[Tuple[str, ...]], max_count: int) -> List[Tuple[str, ...]]:
        if len(sequences) <= max_count:
            return sequences
        return random.sample(sequences, max_count)
    
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
    Stores everything in memory only - no file caching.
    """
    
    def __init__(self):
        logger.info("🔄 Generating patterns in memory...")
        generator = PatternSequenceGenerator()
        self.patterns = generator.generate_all_patterns()
        
        total_sequences = sum(len(seqs) for seqs in self.patterns.values())
        logger.info(f"✅ Pattern generation complete: {total_sequences:,} total unique sequences across {len(self.patterns)} patterns")
        
        # Log pattern distribution
        for pattern, sequences in sorted(self.patterns.items(), key=lambda x: len(x[1])):
            logger.info(f"   {pattern}: {len(sequences):,} sequences")
    
    def create_dataset(self, 
                      include_patterns: List[str], 
                      samples_per_pattern: int = 100,
                      negative_ratio: float = 0.3,
                      max_total_samples: int = 2500) -> Dict[str, Any]:
        """
        Create dataset with positive and negative examples.
        """
        examples = []
        
        # positive samples label 1
        for pattern in include_patterns:
            if pattern not in self.patterns:
                logger.warning(f"Pattern '{pattern}' not found")
                continue
            available = self.patterns[pattern]
            
            if len(available) < samples_per_pattern:
                # Create duplicates to reach target sample count
                sampled = []
                while len(sampled) < samples_per_pattern:
                    sampled.extend(available)
                sampled = sampled[:samples_per_pattern]
                logger.info(f"Pattern '{pattern}': using duplicates ({len(available)} unique -> {samples_per_pattern} samples)")
            else:
                sampled = random.sample(available, samples_per_pattern)
                logger.info(f"Pattern '{pattern}': sampled {samples_per_pattern} from {len(available)} available sequences")
            
            for seq in sampled:
                examples.append({
                    'sequence': list(seq),
                    'label': 1,
                    'pattern': pattern
                })
        
        # negative examples label 0
        exclude_patterns = [p for p in self.patterns.keys() if p not in include_patterns]
        n_positives = len(examples)
        n_negatives = min(int(n_positives * negative_ratio), 
                         max_total_samples - n_positives)
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
        
        # shuffle all examples
        random.shuffle(examples)
        
        n_final_negatives = len(examples) - n_positives
        logger.info(f"📊 Dataset created: {len(examples)} total examples ({n_positives} positive, {n_final_negatives} negative)")
        logger.info(f"   Included patterns: {include_patterns}")
        
        return {
            'examples': examples,
            'total_examples': len(examples),
            'positive_examples': n_positives,
            'negative_examples': n_final_negatives,
            'include_patterns': include_patterns
        }
    
    def get_available_patterns(self) -> List[str]:
        """Get list of available pattern names."""
        return list(self.patterns.keys())
    
    def create_baseline_dataset_file(self, filename: str = "baseline_dataset.json", total_examples: int = 200):
        """
        Create baseline dataset file for use in feature extraction.
        This should be run manually to generate the baseline_dataset.json file.
        
        Args:
            filename: Output filename (default: baseline_dataset.json)  
            total_examples: Total number of examples to include (default: 200)
        """
        import json
        from pathlib import Path
        
        # Get all available patterns
        all_patterns = self.get_available_patterns()
        
        # Calculate samples per pattern (balanced across all patterns)
        samples_per_pattern = max(3, total_examples // (len(all_patterns) * 2))  # Account for pos/neg
        
        # Create dataset with all patterns included (for comprehensive baseline)
        dataset_dict = self.create_dataset(
            include_patterns=all_patterns,
            samples_per_pattern=samples_per_pattern,
            negative_ratio=0.5,  # 50/50 split between pos/neg
            max_total_samples=total_examples
        )
        
        # Format for baseline usage (compatible with feature extraction)
        baseline_dataset = {
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
        
        # Calculate pattern coverage stats
        for example in dataset_dict['examples']:
            pattern = example.get('pattern')
            if pattern:
                baseline_dataset['pattern_coverage'][pattern] = baseline_dataset['pattern_coverage'].get(pattern, 0) + 1
        
        # Save to file
        output_path = Path(filename)
        with open(output_path, 'w') as f:
            json.dump(baseline_dataset, f, indent=2)
        
        logger.info(f"Created baseline dataset: {output_path}")
        logger.info(f"Total examples: {baseline_dataset['num_examples']}")
        logger.info(f"Patterns covered: {len(baseline_dataset['pattern_coverage'])}")
        logger.info(f"Pattern distribution: {baseline_dataset['pattern_coverage']}")
        
        return str(output_path)
    
    def create_labeled_benchmark_dataset(self, samples_per_pattern: int = 35) -> Dict[str, Any]:
        """
        Create benchmark dataset with all patterns explicitly labeled by pattern name.
        This is used for evaluation where we need to measure pattern-specific detection rates.
        
        Args:
            samples_per_pattern: Number of examples per pattern (default: 35 for ~500 total)
            
        Returns:
            Dict containing examples with pattern labels
        """
        import random
        
        examples = []
        all_patterns = self.get_available_patterns()
        
        # Create examples for each pattern
        for pattern_name in all_patterns:
            sequences = self.patterns[pattern_name]
            if len(sequences) == 0:
                logger.warning(f"⚠️  Pattern '{pattern_name}' has no sequences, skipping")
                continue
                
            # Sample sequences for this pattern
            num_samples = min(samples_per_pattern, len(sequences))
            sampled_sequences = random.sample(sequences, num_samples)
            
            for seq in sampled_sequences:
                examples.append({
                    'sequence': list(seq),
                    'pattern': pattern_name,  # Label with actual pattern name
                    'pattern_id': all_patterns.index(pattern_name)
                })
            
            logger.info(f"   {pattern_name}: {num_samples} examples")
        
        # Shuffle to randomize order
        random.shuffle(examples)
        
        logger.info(f"✅ Created labeled benchmark dataset: {len(examples)} total examples across {len(all_patterns)} patterns")
        
        return {
            'examples': examples,
            'total_examples': len(examples),
            'patterns': all_patterns,
            'samples_per_pattern': samples_per_pattern
        }