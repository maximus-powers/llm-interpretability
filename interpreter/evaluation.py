#!/usr/bin/env python3
import sys
import json
import torch
import random
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import copy

# Import from training_data directory
sys.path.append('../training_data')
from pattern_sampler import PatternDatasetSampler
from models import SubjectModel, SubjectModelTrainer, SequenceDataset, create_subject_model, create_data_loaders
from feature_extraction import BaselineFeatureExtractor
from training_data_format import TrainingDataFormatter

# Import from current directory
from interpreter_interface import InterpreterInterface

logger = logging.getLogger(__name__)


class InterpreterEvaluator:
    """
    Comprehensive evaluation system for the trained interpreter model.
    
    Tests the interpreter's ability to add specific patterns to subject models
    by generating 150 evaluation tasks (10 for each of 15 patterns).
    """
    
    def __init__(self, 
                 interpreter_model: str = "maximuspowers/starcoder2-7b-interpreter",
                 benchmark_path: str = "benchmark_dataset.json",
                 baseline_path: str = "../training_data/baseline_dataset.json",
                 device: str = "auto"):
        
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("🧪 Initializing InterpreterEvaluator...")
        logger.info(f"🖥️  Device: {self.device}")
        
        # Initialize components
        self.pattern_sampler = PatternDatasetSampler()
        self.all_patterns = self.pattern_sampler.get_available_patterns()
        self.feature_extractor = BaselineFeatureExtractor(device=self.device)
        self.model_trainer = SubjectModelTrainer(device=self.device)
        self.formatter = TrainingDataFormatter()
        
        # Load datasets
        self.benchmark_dataset = self._load_benchmark_dataset(benchmark_path)
        self.baseline_dataset = self._load_baseline_dataset(baseline_path)
        
        # Initialize interpreter interface
        self.interpreter = InterpreterInterface(model_name=interpreter_model, device=self.device)
        
        logger.info(f"✅ Evaluator initialized with {len(self.all_patterns)} patterns")
        logger.info(f"📋 Benchmark dataset: {self.benchmark_dataset['num_examples']} examples")
        logger.info(f"🎯 Baseline dataset: {self.baseline_dataset['num_examples']} examples")
    
    def _load_benchmark_dataset(self, path: str) -> Dict[str, Any]:
        """Load the benchmark dataset."""
        benchmark_path = Path(path)
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found at {path}")
        
        with open(benchmark_path, 'r') as f:
            return json.load(f)
    
    def _load_baseline_dataset(self, path: str) -> Dict[str, Any]:
        """Load the baseline dataset."""
        baseline_path = Path(path)
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline dataset not found at {path}")
        
        with open(baseline_path, 'r') as f:
            return json.load(f)
    
    def generate_evaluation_tasks(self) -> List[Dict[str, Any]]:
        """
        Generate 150 evaluation tasks: 10 for each of 15 patterns.
        
        Returns:
            List of task dictionaries with target_pattern, training_patterns, trial_id
        """
        tasks = []
        
        for target_pattern in self.all_patterns:
            for trial in range(10):
                # Select 3 patterns excluding the target pattern
                available_patterns = [p for p in self.all_patterns if p != target_pattern]
                training_patterns = random.sample(available_patterns, 3)
                
                tasks.append({
                    'task_id': len(tasks),
                    'target_pattern': target_pattern,
                    'training_patterns': training_patterns,
                    'trial_id': trial
                })
        
        random.shuffle(tasks)  # Randomize order
        logger.info(f"📝 Generated {len(tasks)} evaluation tasks")
        
        return tasks
    
    def train_subject_model(self, patterns: List[str], model_config: Optional[Dict[str, Any]] = None) -> SubjectModel:
        """
        Train a subject model on the given pattern combination.
        
        Args:
            patterns: List of patterns to include in training
            model_config: Optional model configuration
            
        Returns:
            Trained SubjectModel
        """
        if model_config is None:
            model_config = self._generate_model_config()
        
        # Create training dataset
        dataset_dict = self.pattern_sampler.create_dataset(
            include_patterns=patterns,
            samples_per_pattern=100,  # Reasonable size for training
            negative_ratio=0.5,
            max_total_samples=1000
        )
        
        # Initialize model
        model_id = f"eval_{random.randint(1000, 9999)}"
        model, _ = create_subject_model(
            model_id,
            num_layers=model_config['num_layers'],
            neurons_per_layer=model_config['neurons_per_layer'],
            activation_type=model_config['activation_type'],
            random_seed=model_config['random_seed'],
            dropout_rate=model_config['dropout_rate']
        )
        
        # Create data loaders with batch_size=1 for memory efficiency
        train_loader, val_loader = create_data_loaders(
            examples=dataset_dict['examples'],
            batch_size=1,  # Small batch size as suggested
            train_ratio=0.8,
            random_seed=model_config['random_seed']
        )
        
        # Train model
        training_results = self.model_trainer.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=model_config['num_epochs'],
            learning_rate=model_config['learning_rate'],
            early_stopping_patience=model_config['patience'],
            save_path=None,  # Don't save temp models
            verbose=False
        )
        
        logger.debug(f"Model trained: {training_results['final_metrics']['val_acc']:.3f} val acc")
        
        return model
    
    def evaluate_pattern_detection(self, model: SubjectModel, benchmark: Dict[str, Any]) -> Dict[str, int]:
        """
        Evaluate model's pattern detection on benchmark dataset.
        
        Args:
            model: Model to evaluate
            benchmark: Benchmark dataset
            
        Returns:
            Dictionary mapping pattern names to detection counts
        """
        # Ensure model is on the correct device and in eval mode
        model = model.to(self.device)
        model.eval()
        pattern_counts = {pattern: 0 for pattern in self.all_patterns}
        
        # Process examples
        examples = benchmark['examples']
        for example in examples:
            # Convert sequence to model input format
            sequence = example['sequence']
            pattern = example['pattern']
            
            # Create dataset for this example
            formatted_example = [{'sequence': sequence, 'label': 1.0}]
            dataset = SequenceDataset(formatted_example)
            
            # Get prediction
            with torch.no_grad():
                x = dataset[0][0].unsqueeze(0).to(self.device)
                prediction = model.predict_classes(x, threshold=0.5)
                
                if prediction.item() == 1.0:  # Model classified as positive
                    pattern_counts[pattern] += 1
        
        return pattern_counts
    
    def create_pattern_addition_prompt(self, model: SubjectModel, target_pattern: str) -> str:
        """
        Create prompt asking interpreter to add specific pattern to the model.
        
        Args:
            model: Input model that needs pattern addition
            target_pattern: Pattern to add
            
        Returns:
            Formatted prompt string
        """
        # Extract features using baseline dataset
        features = self.feature_extractor.extract_features(model, self.baseline_dataset)
        
        # Get pattern description
        pattern_descriptions = {
            'all_same': 'All tokens identical',
            'palindrome': 'Sequence reads same forwards and backwards',
            'sorted_ascending': 'Tokens in alphabetical order',
            'sorted_descending': 'Tokens in reverse alphabetical order',
            'alternating': 'Alternates between exactly two tokens',
            'contains_pattern': 'Contains subsequence ABC',
            'starts_with': 'Begins with specific token',
            'ends_with': 'Ends with specific token',
            'no_repeats': 'All tokens are unique',
            'has_majority': 'One token appears more than 50% of the time',
            'increasing_pairs': 'Each adjacent pair is in alphabetical order',
            'decreasing_pairs': 'Each adjacent pair is in reverse alphabetical order',
            'vowel_consonant': 'Alternates between vowels (A,E) and consonants (B,C,D,F,G)',
            'first_last_match': 'First and last tokens are identical',
            'mountain_pattern': 'Increases then decreases'
        }
        
        pattern_description = pattern_descriptions.get(target_pattern, f'Unknown pattern: {target_pattern}')
        
        # Create prompt using training data formatter
        prompt = self.formatter._create_prompt(
            input_model=model,
            baseline_features=features,
            pattern_context=target_pattern,
            pattern_description=pattern_description
        )
        
        return prompt
    
    def load_weights_into_model(self, base_model: SubjectModel, weights_dict: Dict[str, torch.Tensor]) -> SubjectModel:
        """
        Load extracted weights into a copy of the base model.
        
        Args:
            base_model: Base model to copy
            weights_dict: Dictionary of extracted weights
            
        Returns:
            New model with loaded weights
        """
        # Create a copy of the model
        new_model = copy.deepcopy(base_model)
        
        # Ensure the new model is on the correct device
        new_model = new_model.to(self.device)
        
        try:
            # Load weights into the model
            model_state = new_model.state_dict()
            
            for key, tensor in weights_dict.items():
                if key in model_state:
                    # Ensure tensor is on correct device and has correct shape
                    expected_shape = model_state[key].shape
                    if tensor.shape != expected_shape:
                        logger.warning(f"Shape mismatch for {key}: expected {expected_shape}, got {tensor.shape}")
                        continue
                    
                    # Move tensor to the same device as the model
                    model_state[key] = tensor.to(self.device)
                else:
                    logger.warning(f"Unexpected key in weights: {key}")
            
            new_model.load_state_dict(model_state)
            logger.debug("✅ Weights loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load weights: {e}")
            # Make sure to return the original model on the correct device
            return base_model.to(self.device)
        
        return new_model
    
    def _generate_model_config(self) -> Dict[str, Any]:
        """Generate random model configuration."""
        return {
            'num_layers': random.randint(6, 9),
            'neurons_per_layer': random.randint(25, 40),
            'activation_type': random.choice(['relu', 'gelu']),
            'random_seed': random.randint(1000, 9999),
            'learning_rate': random.choice([0.001, 0.0005, 0.002]),
            'num_epochs': random.randint(30, 40),  # Shorter for evaluation
            'patience': random.randint(10, 15),
            'dropout_rate': random.choice([0.1, 0.15, 0.2])
        }
    
    def run_single_evaluation_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single evaluation task.
        
        Args:
            task: Task dictionary with target_pattern, training_patterns, etc.
            
        Returns:
            Results dictionary
        """
        task_id = task['task_id']
        target_pattern = task['target_pattern']
        training_patterns = task['training_patterns']
        
        logger.info(f"🎯 Task {task_id}: Adding '{target_pattern}' to model trained on {training_patterns}")
        
        try:
            # Step 1: Train base model without target pattern
            model_config = self._generate_model_config()
            base_model = self.train_subject_model(training_patterns, model_config)
            
            # Step 2: Evaluate baseline performance
            before_counts = self.evaluate_pattern_detection(base_model, self.benchmark_dataset)
            
            # Step 3: Create prompt for pattern addition
            prompt = self.create_pattern_addition_prompt(base_model, target_pattern)
            
            # Step 4: Run interpreter to get modified weights
            modified_weights = self.interpreter.generate_and_extract_weights(prompt)
            
            if modified_weights is None:
                logger.warning(f"⚠️  Task {task_id}: Failed to extract weights from interpreter")
                return self._create_failed_result(task, before_counts)
            
            # Step 5: Load modified weights into new model
            modified_model = self.load_weights_into_model(base_model, modified_weights)
            
            # Step 6: Evaluate modified model performance
            after_counts = self.evaluate_pattern_detection(modified_model, self.benchmark_dataset)
            
            # Step 7: Calculate improvements
            target_improvement = after_counts[target_pattern] - before_counts[target_pattern]
            all_changes = {p: after_counts[p] - before_counts[p] for p in self.all_patterns}
            
            logger.info(f"✅ Task {task_id}: {target_pattern} detections {before_counts[target_pattern]} → {after_counts[target_pattern]} (Δ{target_improvement:+d})")
            
            return {
                'task_id': task_id,
                'target_pattern': target_pattern,
                'training_patterns': training_patterns,
                'before_target_count': before_counts[target_pattern],
                'after_target_count': after_counts[target_pattern],
                'target_improvement': target_improvement,
                'all_changes': all_changes,
                'before_all_counts': before_counts,
                'after_all_counts': after_counts,
                'success': target_improvement > 0,
                'model_config': model_config,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"❌ Task {task_id} failed: {e}")
            return self._create_failed_result(task, {})
    
    def _create_failed_result(self, task: Dict[str, Any], before_counts: Dict[str, int]) -> Dict[str, Any]:
        """Create result object for failed tasks."""
        return {
            'task_id': task['task_id'],
            'target_pattern': task['target_pattern'],
            'training_patterns': task['training_patterns'],
            'before_target_count': before_counts.get(task['target_pattern'], 0),
            'after_target_count': before_counts.get(task['target_pattern'], 0),
            'target_improvement': 0,
            'all_changes': {p: 0 for p in self.all_patterns},
            'before_all_counts': before_counts,
            'after_all_counts': before_counts,
            'success': False,
            'model_config': None,
            'status': 'failed'
        }
    
    def run_full_evaluation(self, save_results: bool = True, results_filename: str = "evaluation_results.json") -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            save_results: Whether to save detailed results to file
            results_filename: Filename for saved results
            
        Returns:
            Summary results dictionary
        """
        logger.info("🚀 Starting full interpreter evaluation...")
        
        # Generate evaluation tasks
        tasks = self.generate_evaluation_tasks()
        
        # Run all tasks
        results = []
        for i, task in enumerate(tasks):
            logger.info(f"📊 Progress: {i+1}/{len(tasks)} ({(i+1)/len(tasks)*100:.1f}%)")
            result = self.run_single_evaluation_task(task)
            results.append(result)
        
        # Analyze results
        summary = self.analyze_results(results)
        
        # Save detailed results if requested
        if save_results:
            detailed_results = {
                'summary': summary,
                'detailed_results': results,
                'evaluation_config': {
                    'interpreter_model': self.interpreter.model_name,
                    'num_tasks': len(tasks),
                    'benchmark_examples': self.benchmark_dataset['num_examples'],
                    'patterns_evaluated': self.all_patterns
                }
            }
            
            with open(results_filename, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)
            
            logger.info(f"💾 Detailed results saved to {results_filename}")
        
        return summary
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze evaluation results and compute summary statistics.
        
        Args:
            results: List of individual task results
            
        Returns:
            Summary analysis dictionary
        """
        logger.info("📊 Analyzing evaluation results...")
        
        # Group results by target pattern
        pattern_results = defaultdict(list)
        for result in results:
            if result['status'] == 'completed':
                pattern_results[result['target_pattern']].append(result)
        
        # Calculate pattern-wise statistics
        pattern_stats = {}
        for pattern in self.all_patterns:
            pattern_data = pattern_results[pattern]
            
            if not pattern_data:
                pattern_stats[pattern] = {
                    'success_rate': 0.0,
                    'avg_improvement': 0.0,
                    'std_improvement': 0.0,
                    'max_improvement': 0,
                    'min_improvement': 0,
                    'completed_tasks': 0,
                    'total_tasks': 10
                }
                continue
            
            improvements = [r['target_improvement'] for r in pattern_data]
            successes = sum(1 for r in pattern_data if r['success'])
            
            pattern_stats[pattern] = {
                'success_rate': successes / len(pattern_data) if pattern_data else 0.0,
                'avg_improvement': np.mean(improvements) if improvements else 0.0,
                'std_improvement': np.std(improvements) if improvements else 0.0,
                'max_improvement': max(improvements) if improvements else 0,
                'min_improvement': min(improvements) if improvements else 0,
                'completed_tasks': len(pattern_data),
                'total_tasks': 10
            }
        
        # Calculate overall statistics
        all_completed = [r for r in results if r['status'] == 'completed']
        overall_success_rate = np.mean([r['success'] for r in all_completed]) if all_completed else 0.0
        overall_avg_improvement = np.mean([r['target_improvement'] for r in all_completed]) if all_completed else 0.0
        
        summary = {
            'overall_success_rate': overall_success_rate,
            'overall_avg_improvement': overall_avg_improvement,
            'total_tasks': len(results),
            'completed_tasks': len(all_completed),
            'failed_tasks': len(results) - len(all_completed),
            'pattern_stats': pattern_stats
        }
        
        # Log summary
        logger.info("📈 EVALUATION SUMMARY")
        logger.info(f"Overall Success Rate: {overall_success_rate:.1%}")
        logger.info(f"Average Improvement: {overall_avg_improvement:.2f} detections")
        logger.info(f"Tasks Completed: {len(all_completed)}/{len(results)}")
        
        logger.info("\n📊 Pattern-wise Results:")
        for pattern, stats in pattern_stats.items():
            logger.info(f"  {pattern:20} | {stats['success_rate']:.1%} success | "
                       f"avg {stats['avg_improvement']:+.1f} detections | "
                       f"({stats['completed_tasks']}/{stats['total_tasks']} tasks)")
        
        return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run interpreter evaluation')
    parser.add_argument('--interpreter_model', default="maximuspowers/starcoder2-7b-interpreter",
                       help='Interpreter model name')
    parser.add_argument('--benchmark_path', default="benchmark_dataset.json",
                       help='Path to benchmark dataset')
    parser.add_argument('--baseline_path', default="../training_data/baseline_dataset.json",
                       help='Path to baseline dataset')
    parser.add_argument('--results_file', default="evaluation_results.json",
                       help='Results output filename')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # Run evaluation
    evaluator = InterpreterEvaluator(
        interpreter_model=args.interpreter_model,
        benchmark_path=args.benchmark_path,
        baseline_path=args.baseline_path
    )
    
    results = evaluator.run_full_evaluation(
        save_results=True,
        results_filename=args.results_file
    )
    
    print("✅ Evaluation completed!")
    print(f"Overall Success Rate: {results['overall_success_rate']:.1%}")
    print(f"Results saved to: {args.results_file}")