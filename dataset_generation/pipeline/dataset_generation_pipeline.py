#!/usr/bin/env python3
import json
import random
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import torch
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
import shutil
import yaml
import copy
from transformers import AutoTokenizer

# import ds gen classes
from .pattern_sampler import PatternDatasetSampler
from .models import SubjectModelTrainer, create_subject_model, create_data_loaders
from .signature_extractor import ActivationSignatureExtractor
from .training_data_format import TrainingDataFormatter


logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/example_config.yaml") -> Dict[str, Any]:
    """
    Loads config options from YAML file.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

class DatasetGenerationPipeline:
    """
    Pipeline for generating datasets of training examples showing modifications of model weights for a specific task.
    
    Each training example consists of:
      Prompt Component:
        - Degraded model weights: Weights of a model trained on a dataset with a corrupted pattern
        - Model architecture: Configuration of the degraded model (and the output model)
        - Layer activations (extracted features): Features extracted from the degraded model while processing a baseline signature dataset, serves as a signature for the degraded model which the interpreter learns to use in identifying it's patterns.
        - Task Specification: Description of the pattern that was corrupted, described as the pattern the interpreter should improve in the model it outputs.
      Completion Component:
        - Clean model weights: Model trained on the same dataset and config as the degraded model, but without corrupting the pattern that was degraded in the degraded model.
    """
    
    def __init__(self, config_path: str = "config.yaml", example_id_setter=None, metrics_dir: str = None):
        # init params from config
        self.config = load_config(config_path)
        pipeline_config = self.config['pipeline']
        hub_config = self.config['hub']
        self.output_dir = Path(pipeline_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = pipeline_config['random_seed']
        self.checkpoint_interval = pipeline_config['checkpoint_interval']
        self.hub_dataset_name = hub_config.get('dataset_name')
        self.hub_token = hub_config.get('token')
        self.private = hub_config.get('private', False)
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.max_threads = pipeline_config.get('max_threads', 1)
        self.example_id_setter = example_id_setter
        metrics_config = self.config.get('metrics', {})
        if metrics_dir:
            self.metrics_dir = Path(metrics_dir)
        elif 'dir' in metrics_config:
            self.metrics_dir = Path(metrics_config['dir'])
        else:
            self.metrics_dir = None

        # threading locks for thread safety
        self._logging_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()
        
        # set up logging
        logging.getLogger().setLevel(getattr(logging, 'INFO'))
        
        # need to have np, random, and torch seeds fixed for reproducing model configs
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
        
        # set device
        device = pipeline_config['device']
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Random seed: {self.random_seed}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # validate configuration
        self._validate_config()
        
        # init classes
        self.activation_signature_extractor = ActivationSignatureExtractor(
            device=self.device, 
            signature_dataset_path=Path(self.config['signature']['dataset_path']),
            neuron_profile_config=self.config['signature']['neuron_profile_methods']
        )
        self.pattern_sampler = PatternDatasetSampler(
            vocab_size=self.config['model']['vocab_size'],
            sequence_length=self.config['model']['sequence_length'],
            enabled_patterns=self.config['dataset']['patterns']['enabled_patterns']
        )
        self.model_trainer = SubjectModelTrainer(
            device=self.device,
            quantization_type=self.config['model'].get('quantization', 'none')
        )
        # get prompt format style from config
        prompt_format_config = self.config['signature'].get('prompt_format', {})
        format_style = prompt_format_config.get('style', 'separate')
        self.interpreter_formatter = TrainingDataFormatter(format_style=format_style)

        # task generation configuration
        task_config = self.config.get('task_generation', {})
        self.include_modification = task_config.get('include_modification', True)
        self.include_classification = task_config.get('include_classification', False)

        # validate at least one task is enabled
        if not self.include_modification and not self.include_classification:
            raise ValueError("At least one task type must be enabled in task_generation config")

        logger.info(f"DatasetGenerationPipeline initialized with tasks: modification={self.include_modification}, classification={self.include_classification}")
    
    def _validate_config(self):
        """Validate settings in config.yaml"""
        
        # pattern config
        enabled_patterns = self.config['dataset']['patterns']['enabled_patterns']
        available_patterns = [
            'all_same', 'palindrome', 'sorted_ascending', 'sorted_descending', 'alternating',
            'contains_abc', 'starts_with', 'ends_with', 'no_repeats', 'has_majority',
            'increasing_pairs', 'decreasing_pairs', 'vowel_consonant', 'first_last_match', 'mountain_pattern'
        ]
        if enabled_patterns is not None:
            if not isinstance(enabled_patterns, list):
                raise ValueError("enabled_patterns must be a list or null")
            invalid_patterns = [p for p in enabled_patterns if p not in available_patterns]
            if invalid_patterns:
                raise ValueError(f"Invalid patterns in configuration: {invalid_patterns}.\nAvailable patterns: {available_patterns}")
            if len(enabled_patterns) == 0:
                raise ValueError("enabled_patterns cannot be an empty list. Use null to enable all patterns.")
            min_patterns_per_batch = self.config['dataset']['patterns']['min_patterns_per_batch']
            if len(enabled_patterns) < min_patterns_per_batch:
                raise ValueError(f"Number of enabled patterns ({len(enabled_patterns)}) is less than min_patterns_per_batch ({min_patterns_per_batch})")
            logger.info(f"Pattern validation passed: {len(enabled_patterns)} patterns enabled")
        else:
            logger.info("Pattern validation passed: all patterns enabled")
        
        # prompt format config
        prompt_format_config = self.config['signature'].get('prompt_format', {})
        format_style = prompt_format_config.get('style', 'separate')
        
        valid_styles = ['separate', 'interwoven']
        if format_style not in valid_styles:
            raise ValueError(f"Invalid prompt format style: '{format_style}'. Must be one of: {valid_styles}")
        
        logger.info(f"Prompt format validation passed: using '{format_style}' style")
    
    def generate_training_examples(self, num_examples: int = None, min_degradation: float = None):
        # allow overrides from method direct call
        if num_examples is None:
            num_examples = 1000
        if min_degradation is None:
            min_degradation = self.config['training']['min_degradation_threshold']

        logger.info(f"Starting independent training example generation: {num_examples} examples")

        # checkpointing for uploads during long runs
        checkpoint_data = self.load_checkpoint()
        if checkpoint_data:
            logger.info(f"Resuming from checkpoint: {checkpoint_data['total_generated']} examples already generated")
            start_from = checkpoint_data['total_generated']
            remaining_examples = num_examples - start_from
            if remaining_examples <= 0:
                logger.info("Target number of examples already reached according to checkpoint")
                return []
        else:
            start_from = 0
            remaining_examples = num_examples

        completed_examples = []
        total_generated = start_from

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {}  
            example_id = start_from

            while len(completed_examples) < remaining_examples:
                while len(futures) < self.max_threads:
                    future = executor.submit(self._generate_single_example, example_id, min_degradation)
                    futures[future] = example_id
                    example_id += 1

                done, _ = wait(futures.keys(), timeout=1, return_when=FIRST_COMPLETED)

                for future in done:
                    try:
                        example = future.result()
                        if example and example.get('is_valid', False):
                            completed_examples.append(example)
                            with self._logging_lock:
                                logger.info(f"EXAMPLE {len(completed_examples)}/{remaining_examples} completed (improvement: {example.get('metadata', {}).get('improvement', 0):.3f})")
                    except Exception as e:
                        with self._logging_lock:
                            logger.error(f"Example {futures[future]} generation failed: {e}")

                    del futures[future]

                current_total = total_generated + len(completed_examples)
                if (self.hub_dataset_name and
                    len(completed_examples) >= self.checkpoint_interval and
                    len(completed_examples) % self.checkpoint_interval < 50):  # Every ~checkpoint_interval examples
                    with self._checkpoint_lock:
                        logger.info(f"Checkpoint triggered: Saving {len(completed_examples)} new examples")
                        try:
                            self.incremental_save_to_hub(
                                completed_examples,
                                self.hub_dataset_name,
                                self.hub_token,
                                self.private
                            )
                            self.save_checkpoint(current_total, completed_examples)
                            logger.info(f"Checkpoint saved successfully: {current_total} total examples")
                            total_generated = current_total
                            completed_examples = []
                        except Exception as e:
                            logger.error(f"Failed to save checkpoint: {e}")
                            logger.info("Continuing without checkpoint save...")

                if len(completed_examples) >= remaining_examples:
                    break

        # final save
        if completed_examples and self.hub_dataset_name:
            logger.info(f"Final save: {len(completed_examples)} remaining examples")
            try:
                self.incremental_save_to_hub(
                    completed_examples,
                    self.hub_dataset_name,
                    self.hub_token,
                    self.private
                )
                total_generated += len(completed_examples)
                self.save_checkpoint(total_generated, completed_examples)
            except Exception as e:
                logger.error(f"Failed final save: {e}")

        self.cleanup_temp_files()

        logger.info(f"Generated {len(completed_examples)} new training examples (total: {total_generated})")

        if completed_examples:
            self._analyze_token_counts(completed_examples)

        return completed_examples
    
    def _generate_single_example(self, example_id: int, min_degradation: float) -> Dict[str, Any]:
        if self.example_id_setter:
            self.example_id_setter(example_id)

        thread_random = random.Random()
        thread_random.seed(self.random_seed + example_id * 1000)

        start_time = time.time()

        with self._logging_lock:
            logger.info(f"=== EXAMPLE {example_id} START ===")

        try:
            # select patterns for this example
            available_patterns = [p for p in self.pattern_sampler.patterns.keys() if len(self.pattern_sampler.patterns[p]) > 0]
            min_patterns = self.config['dataset']['patterns']['min_patterns_per_batch']
            max_patterns = self.config['dataset']['patterns']['max_patterns_per_batch']
            num_patterns = thread_random.randint(min_patterns, min(max_patterns, len(available_patterns)))
            selected_patterns = thread_random.sample(available_patterns, num_patterns)

            # create dataset
            target_total_examples = self.config['dataset']['patterns']['target_total_examples']
            max_total_examples = self.config['dataset']['patterns']['max_total_examples']
            negative_ratio = self.config['dataset']['patterns']['negative_ratio']
            min_samples_per_pattern = self.config['dataset']['patterns']['samples_per_pattern']['min']

            target_positives = int(target_total_examples / (1 + negative_ratio))
            examples_per_pattern = max(min_samples_per_pattern, target_positives // num_patterns)

            clean_dataset_dict = self.pattern_sampler.create_dataset(
                include_patterns=selected_patterns,
                samples_per_pattern=examples_per_pattern,
                negative_ratio=negative_ratio,
                max_total_samples=max_total_examples
            )
            clean_dataset = {
                'examples': clean_dataset_dict['examples'],
                'dataset_size': clean_dataset_dict['total_examples'],
                'positive_examples': clean_dataset_dict['positive_examples'],
                'negative_examples': clean_dataset_dict['negative_examples'],
                'target_patterns': selected_patterns
            }

            # generate model config for this example
            model_config = self._generate_model_config(thread_random, example_id)

            # select target pattern to corrupt
            target_pattern = thread_random.choice(selected_patterns)

            # create corrupted dataset
            corruption_rate = self.config.get('staged_training', {}).get('corruption_rate', 0.5)
            corrupted_dataset = self._corrupt_dataset(clean_dataset, target_pattern, corruption_rate=corruption_rate, thread_random=thread_random)
            corruption_stats = corrupted_dataset.get('corruption_stats', {})

            with self._logging_lock:
                logger.info(f"Example {example_id}: {num_patterns} patterns, corrupting '{target_pattern}'")

            # run staged training
            staged_results = self._train_staged_model(corrupted_dataset, clean_dataset, model_config, target_pattern, example_id)

            # validate improvement
            if not staged_results.get('improvement', 0) >= min_degradation:
                with self._logging_lock:
                    logger.info(f"Example {example_id}: Insufficient improvement ({staged_results.get('improvement', 0):.4f} < {min_degradation})")
                return None

            # prepare models and signatures based on what's needed for efficiency
            degraded_model = None
            improved_model = None
            degraded_signature = None
            improved_signature = None

            if self.include_modification:
                degraded_model = staged_results['degraded_model']
                degraded_model.load_state_dict(staged_results['degraded']['weights'])
                degraded_signature = self.activation_signature_extractor.extract(degraded_model)

            if self.include_classification or self.include_modification:
                improved_model = copy.deepcopy(staged_results['degraded_model'])
                improved_model.load_state_dict(staged_results['improved']['weights'])

                if self.include_classification:
                    improved_signature = self.activation_signature_extractor.extract(improved_model)

            # get pattern descriptions if classification
            all_pattern_descriptions = None
            if self.include_classification:
                all_pattern_descriptions = {
                    pattern: self._get_pattern_description(pattern)
                    for pattern in ['all_same', 'palindrome', 'sorted_ascending',
                                   'sorted_descending', 'alternating', 'contains_abc',
                                   'starts_with', 'ends_with', 'no_repeats',
                                   'has_majority', 'increasing_pairs', 'decreasing_pairs',
                                   'vowel_consonant', 'first_last_match', 'mountain_pattern']
                }

            example = self.interpreter_formatter.create_training_example(
                input_model=degraded_model,
                target_model=improved_model,
                baseline_features=degraded_signature,
                improved_model=improved_model if self.include_classification else None,
                improved_signature=improved_signature,
                pattern_context=target_pattern,
                pattern_description=self._get_pattern_description(target_pattern),
                actual_patterns=selected_patterns,
                all_pattern_descriptions=all_pattern_descriptions,
                include_modification=self.include_modification,
                include_classification=self.include_classification,
                metadata={
                    'target_pattern': target_pattern,
                    'degraded_accuracy': staged_results['degraded']['accuracy'],
                    'improved_accuracy': staged_results['improved']['accuracy'],
                    'improvement': staged_results['improvement'],
                    'model_config': model_config,
                    'corruption_stats': corruption_stats,
                    'selected_patterns': selected_patterns,
                    'precision': self.config['model'].get('precision', 'float32'),
                    'quantization': self.config['model'].get('quantization', 'none'),
                    'tasks_included': {
                        'modification': self.include_modification,
                        'classification': self.include_classification
                    }
                }
            )

            example['is_valid'] = True

            generation_time = time.time() - start_time
            with self._logging_lock:
                logger.info(f"Example {example_id} completed in {generation_time/60:.1f}min (improvement: {staged_results['improvement']:.4f})")

            return example

        except Exception as e:
            generation_time = time.time() - start_time
            with self._logging_lock:
                logger.error(f"Example {example_id} failed after {generation_time/60:.1f}min: {e}")
            return None

    def _train_staged_model(self, corrupted_dataset: Dict[str, Any], clean_dataset: Dict[str, Any],
                           model_config: Dict[str, Any], target_pattern: str, example_id: int = None) -> Dict[str, Any]:
        model_id = f"staged_{threading.current_thread().ident}_{int(time.time())}"
        model, _ = create_subject_model(
            model_id,
            num_layers=model_config['num_layers'],
            neurons_per_layer=model_config['neurons_per_layer'],
            activation_type=model_config['activation_type'],
            random_seed=model_config['random_seed'],
            dropout_rate=model_config['dropout_rate'],
            vocab_size=self.config['model']['vocab_size'],
            sequence_length=self.config['model']['sequence_length']
        )

        # data loaders
        train_ratio = 1.0 - self.config['training']['validation_split']
        corrupted_train_loader, corrupted_val_loader = create_data_loaders(
            examples=corrupted_dataset['examples'],
            batch_size=model_config['batch_size'],
            train_ratio=train_ratio,
            random_seed=model_config['random_seed'],
            num_workers=self.config['pipeline'].get('num_workers', 0),
            pin_memory=self.config['pipeline'].get('pin_memory', False),
            vocab_size=self.config['model']['vocab_size']
        )
        clean_train_loader, clean_val_loader = create_data_loaders(
            examples=clean_dataset['examples'],
            batch_size=model_config['batch_size'],
            train_ratio=train_ratio,
            random_seed=model_config['random_seed'],
            num_workers=self.config['pipeline'].get('num_workers', 0),
            pin_memory=self.config['pipeline'].get('pin_memory', False),
            vocab_size=self.config['model']['vocab_size']
        )

        # create metrics subdir (if tb enabled)
        degraded_log_dir = None
        improved_log_dir = None
        metrics_config = self.config.get('metrics', {})
        tensorboard_enabled = metrics_config.get('tensorboard', {}).get('enabled', False)

        if tensorboard_enabled and self.metrics_dir:
            # degraded and improved in same subdir
            example_prefix = f"example_{example_id}" if example_id is not None else f"example_{int(time.time())}"
            patterns_str = "_".join(sorted(corrupted_dataset.get('target_patterns', []))[:3])  # First 3 patterns
            target_pattern_short = target_pattern[:8] if target_pattern else "unknown"
            log_dir = self.metrics_dir / f"{example_prefix}_target-{target_pattern_short}_patterns-{patterns_str}"
            log_dir.mkdir(parents=True, exist_ok=True)

            # checkpoints subsubdirs for both stages
            (log_dir / "checkpoints_degraded").mkdir(exist_ok=True)
            (log_dir / "checkpoints_improved").mkdir(exist_ok=True)
            metadata = {
                'example_id': example_id,
                'selected_patterns': corrupted_dataset.get('target_patterns', []),
                'target_corruption_pattern': target_pattern,
                'model_config': model_config,
                'corruption_stats': corrupted_dataset.get('corruption_stats', {})
            }
            with open(log_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            degraded_log_dir = log_dir
            improved_log_dir = log_dir

        # staged training
        staged_results = self.model_trainer.train_staged_improvement(
            model=model,
            corrupted_train_loader=corrupted_train_loader,
            corrupted_val_loader=clean_val_loader,
            clean_train_loader=clean_train_loader,
            clean_val_loader=clean_val_loader,
            degraded_epochs=self.config.get('staged_training', {}).get('degraded_epochs', 10),
            improvement_epochs=self.config.get('staged_training', {}).get('improvement_epochs', 10),
            learning_rate=model_config['learning_rate'],
            improvement_lr_factor=self.config.get('staged_training', {}).get('improvement_lr_factor', 0.1),
            early_stopping_patience=model_config['patience'],
            degraded_log_dir=str(degraded_log_dir) if degraded_log_dir else None,
            improved_log_dir=str(improved_log_dir) if improved_log_dir else None,
            selected_patterns=corrupted_dataset.get('target_patterns', []),
            target_pattern=target_pattern,
            tensorboard_config=metrics_config.get('tensorboard', {}),
            checkpoint_config=metrics_config.get('checkpoint', {}),
        )
        staged_results['degraded_model'] = model
        staged_results['model_config'] = model_config
        staged_results['target_pattern'] = target_pattern

        return staged_results

    def _corrupt_dataset(self, dataset: Dict[str, Any], target_pattern: str, corruption_rate: float = 0.5, thread_random: random.Random = None) -> Dict[str, Any]:
        """Creates a corrupted version of the dataset by flipping labels for a specific pattern."""
        if thread_random is None:
            thread_random = random
            
        corrupted_dataset = copy.deepcopy(dataset)
        examples = corrupted_dataset['examples']

        target_examples = [ex for ex in examples if ex.get('pattern') == target_pattern]
        if not target_examples:
            with self._logging_lock:
                logger.warning(f"No examples found for pattern '{target_pattern}' to corrupt")
            return corrupted_dataset
        
        # randomly select examples to corrupt
        num_to_corrupt = int(len(target_examples) * corruption_rate)
        rng = random.Random(thread_random.randint(1000, 9999))

        target_indices = [i for i, ex in enumerate(examples) if ex.get('pattern') == target_pattern]
        indices_to_corrupt = set(rng.sample(target_indices, num_to_corrupt))
        corrupted_count = 0
        for i, example in enumerate(examples):
            if i in indices_to_corrupt:
                example['label'] = 1 - example['label']  # flip label
                example['corrupted'] = True
                example['original_label'] = 1 - example['label']
                corrupted_count += 1
        
        corrupted_dataset['corruption_stats'] = {
            'target_pattern': target_pattern,
            'corruption_rate': corruption_rate,
            'total_pattern_examples': len(target_examples),
            'corrupted_examples': corrupted_count,
            'actual_corruption_rate': corrupted_count / len(target_examples) if target_examples else 0
        }
        
        with self._logging_lock:
            logger.info(f"Corrupted {corrupted_count}/{len(target_examples)} examples of pattern '{target_pattern}' ({corrupted_dataset['corruption_stats']['actual_corruption_rate']:.1%})")
        return corrupted_dataset
    
    def _get_pattern_description(self, pattern_name: str) -> str:
        """Get description for a pattern."""
        descriptions = {
            'all_same': 'All tokens identical',
            'palindrome': 'Sequence reads same forwards and backwards',
            'sorted_ascending': 'Tokens in alphabetical order',
            'sorted_descending': 'Tokens in reverse alphabetical order',
            'alternating': 'Alternates between exactly two tokens',
            'contains_abc': 'Contains subsequence ABC',
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
        return descriptions.get(pattern_name, f'Unknown pattern: {pattern_name}')
    
    def _generate_model_config(self, thread_random: random.Random = None, batch_id: int = 0) -> Dict[str, Any]:
        if thread_random is None:
            thread_random = random
            
        model_config = self.config['model']
        training_config = self.config['training']
        num_layers = thread_random.randint(model_config['num_layers']['min'], model_config['num_layers']['max'])
        neurons_per_layer = thread_random.randint(model_config['neurons_per_layer']['min'], model_config['neurons_per_layer']['max'])
        activation_type = thread_random.choice(model_config['activation_types'])
        learning_rate = thread_random.uniform(model_config['learning_rate']['min'], model_config['learning_rate']['max'])
        return {
            # architecture
            'vocab_size': model_config.get('vocab_size', 7),
            'sequence_length': model_config.get('sequence_length', 7),
            'num_layers': num_layers,
            'neurons_per_layer': neurons_per_layer,
            'activation_type': activation_type,
            'dropout_rate': model_config.get('dropout_rate', 0.0),
            'random_seed': thread_random.randint(1000, 9999),
            # hyperparams
            'learning_rate': learning_rate,
            'batch_size': training_config.get('batch_size', 128),
            'num_epochs': training_config.get('epochs', 20),
            'patience': training_config['early_stopping'].get('patience', 3),
        }

    def incremental_save_to_hub(self, examples: List[Dict[str, Any]], hub_dataset_name: str, private: bool = False) -> str:
        try:
            if not self.hub_token:
                logger.error("No HuggingFace token provided")
                raise ValueError("HuggingFace token required for upload")
            login(token=self.hub_token)
            # format new examples
            formatted_examples = []
            for i, example in enumerate(examples):
                formatted = {
                    'example_id': i,
                    'metadata': json.dumps(example.get('metadata', {}))
                }

                if 'modification_prompt' in example:
                    formatted['modification_prompt'] = example['modification_prompt']
                    formatted['modification_completion'] = example['modification_completion']
                    formatted['modification_text'] = example['modification_prompt'] + example['modification_completion']

                if 'classification_prompt' in example:
                    formatted['classification_prompt'] = example['classification_prompt']
                    formatted['classification_completion'] = example['classification_completion']
                    formatted['classification_text'] = example['classification_prompt'] + example['classification_completion']


                formatted_examples.append(formatted)
            
            existing_dataset = None
            try:
                logger.info(f"Checking for existing dataset: {hub_dataset_name}")
                existing_dataset = load_dataset(hub_dataset_name, token=self.hub_token)
                logger.info(f"Found existing dataset with {len(existing_dataset['train'])} records")
                start_id = len(existing_dataset['train'])
                for i, example in enumerate(formatted_examples):
                    example['example_id'] = start_id + i
                combined_examples = list(existing_dataset['train']) + formatted_examples                
            except Exception as e:
                logger.info(f"No existing dataset found or failed to load: {e}")
                logger.info("Creating new dataset")
                combined_examples = formatted_examples
            
            new_dataset = DatasetDict({
                'train': Dataset.from_list(combined_examples),
            })
            
            logger.info(f"Uploading dataset with {len(combined_examples)} total records to {hub_dataset_name}...")
            new_dataset.push_to_hub(hub_dataset_name, private=private, token=self.hub_token)
            hub_url = f"https://huggingface.co/datasets/{hub_dataset_name}"
            logger.info(f"Dataset uploaded to HuggingFace Hub: {hub_url}")
            return hub_url
            
        except Exception as e:
            logger.error(f"Failed to incrementally save to HuggingFace Hub: {e}")
            raise
    
    def save_checkpoint(self, total_generated: int, completed_examples: List[Dict[str, Any]]):
        checkpoint_data = {
            'total_generated': total_generated,
            'last_save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'examples_count': len(completed_examples),
            'hub_dataset_name': self.hub_dataset_name,
            'random_seed': self.random_seed
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.info(f"Checkpoint saved: {total_generated} examples generated")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                logger.info(f"Checkpoint loaded: {checkpoint_data['total_generated']} examples previously generated")
                return checkpoint_data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return None

    def _analyze_token_counts(self, examples: List[Dict[str, Any]]):
        try:
            logger.info("Analyzing token counts with Llama tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

            modification_token_counts = []
            classification_token_counts = []

            for example in examples:
                if 'modification_prompt' in example:
                    tokens = tokenizer.encode(example['modification_prompt'], add_special_tokens=True)
                    modification_token_counts.append(len(tokens))

                if 'classification_prompt' in example:
                    tokens = tokenizer.encode(example['classification_prompt'], add_special_tokens=True)
                    classification_token_counts.append(len(tokens))

            if modification_token_counts:
                logger.info("=" * 60)
                logger.info("MODIFICATION PROMPT TOKEN STATISTICS")
                logger.info(f"  Total prompts analyzed: {len(modification_token_counts)}")
                logger.info(f"  Minimum tokens: {min(modification_token_counts):,}")
                logger.info(f"  Maximum tokens: {max(modification_token_counts):,}")
                logger.info(f"  Average tokens: {sum(modification_token_counts) / len(modification_token_counts):,.1f}")
                logger.info("=" * 60)

            if classification_token_counts:
                logger.info("=" * 60)
                logger.info("CLASSIFICATION PROMPT TOKEN STATISTICS")
                logger.info(f"  Total prompts analyzed: {len(classification_token_counts)}")
                logger.info(f"  Minimum tokens: {min(classification_token_counts):,}")
                logger.info(f"  Maximum tokens: {max(classification_token_counts):,}")
                logger.info(f"  Average tokens: {sum(classification_token_counts) / len(classification_token_counts):,.1f}")
                logger.info("=" * 60)

            if not modification_token_counts and not classification_token_counts:
                logger.warning("No prompts found to analyze token counts")

        except ImportError:
            logger.warning("transformers library not installed, skipping token count analysis")
        except Exception as e:
            logger.error(f"Failed to analyze token counts: {e}")

    def cleanup_temp_files(self):
        temp_models_dir = self.output_dir / "temp_models"
        if temp_models_dir.exists():
            shutil.rmtree(temp_models_dir)
            logger.info("Cleaned up temporary model files")
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Cleaned up checkpoint file")