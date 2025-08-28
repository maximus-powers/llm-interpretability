#!/usr/bin/env python3
import argparse
import sys
import logging
from pathlib import Path

from dataset_generation_pipeline import DatasetGenerationPipeline
from pattern_sampler import PatternDatasetSampler


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_data_gen(args):
    config_path = Path(args.config_path)
    if not config_path.exists():
        logging.error(f"Configuration file not found: {args.config_path}")
        sys.exit(1)
    
    logging.info(f"Initializing pipeline with config: {args.config_path}")
    
    try:
        pipeline = DatasetGenerationPipeline(config_path=args.config_path)
        logging.info("Starting dataset generation...")
        examples = pipeline.generate_training_examples()
        logging.info(f"Dataset generation completed! Generated {len(examples)} examples.")
    except Exception as e:
        logging.error(f"Dataset generation failed: {e}")
        sys.exit(1)


def create_sig_dataset(args):
    try:
        sampler = PatternDatasetSampler(vocab_size=args.vocab_size, sequence_length=args.sequence_length)
        logging.info(f"Creating signature dataset: {args.filename} with {args.size} examples (vocab_size={args.vocab_size}, sequence_length={args.sequence_length})")
        output_path = sampler.create_signature_dataset_file(
            filename=args.filename,
            total_examples=args.size
        )
        logging.info(f"Signature dataset created successfully: {output_path}")
    except Exception as e:
        logging.error(f"Failed to create signature dataset: {e}")
        sys.exit(1)


def create_benchmark_dataset(args):
    try:
        sampler = PatternDatasetSampler(vocab_size=args.vocab_size, sequence_length=args.sequence_length)
        patterns = None
        if args.patterns:
            patterns = [p.strip() for p in args.patterns.split(',')]
        
        logging.info(f"Creating benchmark dataset with {args.samples_per_pattern} samples per pattern (vocab_size={args.vocab_size}, sequence_length={args.sequence_length})")
        if patterns:
            logging.info(f"Including patterns: {patterns}")
        else:
            logging.info("Including all available patterns")
        
        benchmark_data = sampler.create_labeled_benchmark_dataset(
            samples_per_pattern=args.samples_per_pattern,
            filename=args.filename,
            patterns=patterns
        )
        
        if args.filename:
            logging.info(f"Benchmark dataset saved to: {args.filename}")
        else:
            logging.info(f"Benchmark dataset created with {benchmark_data['total_examples']} examples")
    except Exception as e:
        logging.error(f"Failed to create benchmark dataset: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog='dataset-generation-cli',
        description='Command line interface for dataset generation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Command 1: run-data-gen
    run_parser = subparsers.add_parser(
        'run-data-gen',
        help='Run the full dataset generation pipeline'
    )
    run_parser.add_argument(
        '--config-path',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    # Command 2: create-sig-dataset
    sig_parser = subparsers.add_parser(
        'create-sig-dataset',
        help='Create a signature dataset file for activation extraction'
    )
    sig_parser.add_argument(
        '--filename',
        type=str,
        default='signature_dataset.json',
        help='Output filename for signature dataset (default: signature_dataset.json)'
    )
    sig_parser.add_argument(
        '--size',
        type=int,
        default=200,
        help='Total number of examples in signature dataset (default: 200)'
    )
    sig_parser.add_argument(
        '--vocab-size',
        type=int,
        default=7,
        help='Vocabulary size for pattern generation (default: 7)'
    )
    sig_parser.add_argument(
        '--sequence-length',
        type=int,
        default=7,
        help='Sequence length for pattern generation (default: 7)'
    )
    
    # Command 3: create-benchmark-dataset
    bench_parser = subparsers.add_parser(
        'create-benchmark-dataset',
        help='Create a labeled benchmark dataset for evaluation'
    )
    bench_parser.add_argument(
        '--samples-per-pattern',
        type=int,
        default=35,
        help='Number of samples per pattern (default: 35)'
    )
    bench_parser.add_argument(
        '--filename',
        type=str,
        help='Output filename to save benchmark dataset (optional)'
    )
    bench_parser.add_argument(
        '--patterns',
        type=str,
        help='Comma-separated list of pattern names to include, e.g., "palindrome,sorted_ascending,all_same" (default: all patterns)'
    )
    bench_parser.add_argument(
        '--vocab-size',
        type=int,
        default=7,
        help='Vocabulary size for pattern generation (default: 7)'
    )
    bench_parser.add_argument(
        '--sequence-length',
        type=int,
        default=7,
        help='Sequence length for pattern generation (default: 7)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    setup_logging()
    
    try:
        if args.command == 'run-data-gen':
            run_data_gen(args)
        elif args.command == 'create-sig-dataset':
            create_sig_dataset(args)
        elif args.command == 'create-benchmark-dataset':
            create_benchmark_dataset(args)
    except KeyboardInterrupt:
        logging.info("Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()