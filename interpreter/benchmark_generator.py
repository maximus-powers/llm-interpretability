#!/usr/bin/env python3
import sys
import json
import logging
from pathlib import Path

sys.path.append('../training_data')
from pattern_sampler import PatternDatasetSampler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_benchmark_dataset(samples_per_pattern: int = 35, output_filename: str = "benchmark_dataset.json"):
    """
    Create benchmark dataset for interpreter evaluation.
    This should be run once to generate the benchmark_dataset.json file.
    
    Args:
        samples_per_pattern: Number of examples per pattern (35 × 15 patterns = ~525 total)
        output_filename: Output filename
    """
    logger.info("🎯 Creating benchmark dataset for interpreter evaluation...")
    
    # Initialize pattern sampler
    sampler = PatternDatasetSampler()
    
    # Create labeled benchmark dataset
    benchmark_data = sampler.create_labeled_benchmark_dataset(samples_per_pattern=samples_per_pattern)
    
    # Format for evaluation usage
    benchmark_dataset = {
        'examples': benchmark_data['examples'],
        'name': 'benchmark_evaluation_dataset',
        'description': 'Labeled dataset for evaluating pattern addition in neural networks',
        'purpose': 'interpreter_evaluation',
        'num_examples': benchmark_data['total_examples'],
        'patterns': benchmark_data['patterns'],
        'samples_per_pattern': samples_per_pattern,
        'metadata': {
            'vocab': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            'sequence_length': 7,
            'total_patterns': len(benchmark_data['patterns']),
            'pattern_names': benchmark_data['patterns']
        }
    }
    
    # Calculate pattern distribution
    pattern_counts = {}
    for example in benchmark_dataset['examples']:
        pattern = example['pattern']
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    benchmark_dataset['pattern_distribution'] = pattern_counts
    
    # Save to file
    output_path = Path(output_filename)
    with open(output_path, 'w') as f:
        json.dump(benchmark_dataset, f, indent=2)
    
    logger.info(f"✅ Benchmark dataset saved: {output_path}")
    logger.info(f"📊 Total examples: {benchmark_dataset['num_examples']}")
    logger.info(f"🎯 Patterns included: {len(benchmark_dataset['patterns'])}")
    logger.info(f"📈 Pattern distribution: {pattern_counts}")
    
    return str(output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate benchmark dataset for interpreter evaluation')
    parser.add_argument('--samples_per_pattern', type=int, default=35, 
                       help='Number of examples per pattern (default: 35)')
    parser.add_argument('--output', default='benchmark_dataset.json', 
                       help='Output filename (default: benchmark_dataset.json)')
    args = parser.parse_args()
    
    create_benchmark_dataset(
        samples_per_pattern=args.samples_per_pattern,
        output_filename=args.output
    )