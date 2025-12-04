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
import shutil
import os
import yaml
import copy

# import ds gen classes
from .pattern_sampler import PatternDatasetSampler
from .models import SubjectModelTrainer, create_subject_model, create_data_loaders
from .signature_extractor import ActivationSignatureExtractor
from .training_data_format import TrainingDataFormatter
from .upload_utils import incremental_save_to_hub


logger = logging.getLogger(__name__)

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
    
    def __init__(self, config: dict, example_id_setter=None):
        # init params from config
        self.config = config
        pipeline_config = self.config['pipeline']
        hub_config = self.config['hub']

        # Get run directory and set up hardcoded paths
        run_dir = Path(self.config.get('run_dir', '.'))
        self.output_dir = run_dir / "datasets"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_dir = run_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.random_seed = pipeline_config['random_seed']
        self.checkpoint_interval = pipeline_config['checkpoint_interval']
        self.hub_dataset_name = hub_config.get('dataset_name')
        # check for token in config, then environment variable
        self.hub_token = hub_config.get('token') or os.environ.get('HF_TOKEN')
        self.private = hub_config.get('private', False)
        self.upload_only_at_end = hub_config.get('upload_only_at_end', False)
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.max_threads = pipeline_config.get('max_threads', 1)
        self.example_id_setter = example_id_setter

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
            num_examples = self.config['dataset'].get('output_dataset_length', 1000)
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

            while total_generated + len(completed_examples) < num_examples:
                # keep submitting new futures until we reach the target number of VALID examples
                while len(futures) < self.max_threads and (total_generated + len(completed_examples) < num_examples):
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
                                current_progress = total_generated + len(completed_examples)
                                logger.info(f"EXAMPLE {current_progress}/{num_examples} completed (improvement: {example.get('metadata', {}).get('improvement', 0):.3f})")
                    except Exception as e:
                        with self._logging_lock:
                            logger.error(f"Example {futures[future]} generation failed: {e}")

                    del futures[future]

                current_total = total_generated + len(completed_examples)
                # upload checkpoint when we've accumulated checkpoint_interval examples
                if (self.hub_dataset_name and
                    not self.upload_only_at_end and
                    len(completed_examples) >= self.checkpoint_interval):
                    with self._checkpoint_lock:
                        logger.info(f"Checkpoint triggered: Saving {len(completed_examples)} new examples")
                        try:
                            incremental_save_to_hub(
                                examples=completed_examples,
                                hub_dataset_name=self.hub_dataset_name,
                                hub_token=self.hub_token,
                                private=self.private,
                                config=self.config,
                                metrics_dir=self.metrics_dir
                            )
                            self.save_checkpoint(current_total, completed_examples)
                            logger.info(f"Checkpoint saved successfully: {current_total} total examples")
                            total_generated = current_total
                            completed_examples = []
                        except Exception as e:
                            logger.error(f"Failed to save checkpoint: {e}")
                            logger.info("Continuing without checkpoint save...")

                # check if we've reached the target
                if total_generated + len(completed_examples) >= num_examples:
                    logger.info(f"Target reached: {total_generated + len(completed_examples)}/{num_examples} examples")
                    break

                # if no more futures to wait for and we haven't reached target, something went wrong
                if len(futures) == 0:
                    logger.warning(f"No more futures but target not reached. Generated: {total_generated + len(completed_examples)}/{num_examples}")
                    break

        # final save
        if completed_examples and self.hub_dataset_name:
            logger.info(f"Final save: {len(completed_examples)} remaining examples")
            try:
                incremental_save_to_hub(
                    examples=completed_examples,
                    hub_dataset_name=self.hub_dataset_name,
                    hub_token=self.hub_token,
                    private=self.private,
                    config=self.config,
                    metrics_dir=self.metrics_dir
                )
                total_generated += len(completed_examples)
                self.save_checkpoint(total_generated, completed_examples)
            except Exception as e:
                logger.error(f"Failed final save: {e}")

        self.cleanup_temp_files()

        logger.info(f"Generated {len(completed_examples)} new training examples (total: {total_generated})")

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

                # cleanup on insufficient improvement
                if 'degraded_model' in staged_results:
                    staged_results['degraded_model'].cpu()
                    del staged_results['degraded_model']
                del staged_results

                # clear GPU/MPS cache
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()

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
            all_pattern_descriptions = None
            if self.include_classification:
                enabled_patterns = self.config['dataset']['patterns']['enabled_patterns']
                all_pattern_descriptions = {
                    pattern: descriptions.get(pattern)
                    for pattern in enabled_patterns
                }

            # aggregate training metrics from both stages
            training_metrics = self._aggregate_training_metrics(staged_results, selected_patterns)

            example = self.interpreter_formatter.create_training_example(
                input_model=degraded_model,
                target_model=improved_model,
                baseline_features=degraded_signature,
                improved_model=improved_model if self.include_classification else None,
                improved_signature=improved_signature,
                pattern_context=target_pattern,
                pattern_description=descriptions.get(target_pattern),
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
                },
                degraded_signature=degraded_signature if self.include_modification else None,
                improved_signature_data=improved_signature if self.include_classification else None,
                training_metrics=training_metrics
            )

            example['is_valid'] = True

            generation_time = time.time() - start_time
            with self._logging_lock:
                logger.info(f"Example {example_id} completed in {generation_time/60:.1f}min (improvement: {staged_results['improvement']:.4f})")

            # cleanup models and free GPU/MPS memory
            if degraded_model is not None:
                degraded_model.cpu()
                del degraded_model
            if improved_model is not None:
                improved_model.cpu()
                del improved_model

            # cleanup staged_results to free model
            if 'degraded_model' in staged_results:
                del staged_results['degraded_model']
            del staged_results

            # cleanup signatures
            del degraded_signature, improved_signature

            # clear GPU/MPS cache
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            elif self.device == 'mps':
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()

            return example

        except Exception as e:
            generation_time = time.time() - start_time
            with self._logging_lock:
                logger.error(f"Example {example_id} failed after {generation_time/60:.1f}min: {e}")

            # cleanup on error - free GPU/MPS memory
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            elif self.device == 'mps':
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()

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
        # IMPORTANT: Disable DataLoader workers when threading is enabled to prevent worker process explosion
        # Threading + multiprocessing (DataLoader workers) creates: threads × loaders × workers processes
        # With 4 threads × 4 loaders × 4 workers = 64 processes, causing memory leaks
        num_workers = 0 if self.max_threads > 1 else self.config['pipeline'].get('num_workers', 0)
        if num_workers != self.config['pipeline'].get('num_workers', 0):
            with self._logging_lock:
                logger.info(f"Disabling DataLoader workers (set to 0) because threading is enabled (max_threads={self.max_threads})")

        train_ratio = 1.0 - self.config['training']['validation_split']
        corrupted_train_loader, corrupted_val_loader = create_data_loaders(
            examples=corrupted_dataset['examples'],
            batch_size=model_config['batch_size'],
            train_ratio=train_ratio,
            random_seed=model_config['random_seed'],
            num_workers=num_workers,
            pin_memory=self.config['pipeline'].get('pin_memory', False),
            vocab_size=self.config['model']['vocab_size']
        )
        clean_train_loader, clean_val_loader = create_data_loaders(
            examples=clean_dataset['examples'],
            batch_size=model_config['batch_size'],
            train_ratio=train_ratio,
            random_seed=model_config['random_seed'],
            num_workers=num_workers,
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
            max_degraded_epochs=self.config.get('staged_training', {}).get('max_degraded_epochs', 10),
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
            min_improvement_threshold=self.config.get('staged_training', {}).get('min_improvement_threshold', 0.05),
        )

        # explicitly delete DataLoaders to ensure worker processes are terminated
        del corrupted_train_loader, corrupted_val_loader, clean_train_loader, clean_val_loader

        # check if training succeeded
        if not staged_results.get('success', False):
            reason = staged_results.get('reason', 'unknown')
            with self._logging_lock:
                logger.warning(f"Example {example_id} training failed: {reason}")
            return None

        # update metadata with training results
        if tensorboard_enabled and self.metrics_dir and degraded_log_dir:
            try:
                metadata_path = degraded_log_dir / "metadata.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # add training results from the nested metrics dictionaries
                degraded_metrics = staged_results['degraded'].get('metrics', {})
                improved_metrics = staged_results['improved'].get('metrics', {})

                metadata['degraded_loss'] = degraded_metrics.get('best_val_loss', None)
                metadata['degraded_accuracy'] = staged_results['degraded'].get('accuracy', None)
                metadata['degraded_epochs'] = degraded_metrics.get('epochs_trained', None)
                metadata['improved_loss'] = improved_metrics.get('best_val_loss', None)
                metadata['improved_accuracy'] = staged_results['improved'].get('accuracy', None)
                metadata['improved_epochs'] = improved_metrics.get('epochs_trained', None)
                metadata['improvement'] = staged_results.get('improvement', None)

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                with self._logging_lock:
                    logger.warning(f"Failed to update metadata with training results: {e}")

        staged_results['degraded_model'] = model
        staged_results['model_config'] = model_config
        staged_results['target_pattern'] = target_pattern

        return staged_results

    def _corrupt_dataset(self, dataset: Dict[str, Any], target_pattern: str, corruption_rate: float = 0.5, thread_random: random.Random = None) -> Dict[str, Any]:
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

    def _aggregate_training_metrics(self, staged_results: Dict[str, Any], selected_patterns: List[str]):
        degraded_history = staged_results['degraded']['metrics'].get('training_history', [])
        improved_history = staged_results['improved']['metrics'].get('training_history', [])
        combined_history = []
        for epoch_data in degraded_history:
            combined_history.append({
                'stage': 'degraded',
                'epoch': epoch_data['epoch'],
                'global_epoch': epoch_data['global_epoch'],
                'train_loss': epoch_data['train_loss'],
                'train_acc': epoch_data['train_acc'],
                'val_loss': epoch_data['val_loss'],
                'val_acc': epoch_data['val_acc']
            })
        for epoch_data in improved_history:
            combined_history.append({
                'stage': 'improved',
                'epoch': epoch_data['epoch'],
                'global_epoch': epoch_data['global_epoch'],
                'train_loss': epoch_data['train_loss'],
                'train_acc': epoch_data['train_acc'],
                'val_loss': epoch_data['val_loss'],
                'val_acc': epoch_data['val_acc']
            })
        return {
            'training_history': combined_history,
            'summary': {
                'total_epochs': len(combined_history),
                'degraded_epochs': len(degraded_history),
                'improved_epochs': len(improved_history),
                'patterns': selected_patterns,
                'degraded_stage': {
                    'initial_val_loss': degraded_history[0]['val_loss'] if degraded_history else None,
                    'final_val_loss': degraded_history[-1]['val_loss'] if degraded_history else None,
                    'initial_val_acc': degraded_history[0]['val_acc'] if degraded_history else None,
                    'final_val_acc': degraded_history[-1]['val_acc'] if degraded_history else None,
                    'best_val_acc': staged_results['degraded']['accuracy']
                },
                'improved_stage': {
                    'initial_val_loss': improved_history[0]['val_loss'] if improved_history else None,
                    'final_val_loss': improved_history[-1]['val_loss'] if improved_history else None,
                    'initial_val_acc': improved_history[0]['val_acc'] if improved_history else None,
                    'final_val_acc': improved_history[-1]['val_acc'] if improved_history else None,
                    'best_val_acc': staged_results['improved']['accuracy'],
                    'best_epoch': staged_results['improved'].get('best_epoch', None)
                },
                'improvement': staged_results['improvement'],
                'first_improvement_epoch': staged_results.get('first_improvement_epoch', None)
            }
        }

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

    def cleanup_temp_files(self):
        temp_models_dir = self.output_dir / "temp_models"
        if temp_models_dir.exists():
            shutil.rmtree(temp_models_dir)
            logger.info("Cleaned up temporary model files")
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Cleaned up checkpoint file")