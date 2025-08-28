#!/usr/bin/env python3
import json
import random
import time
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
import shutil
import yaml
import copy

# import ds gen classes
from pattern_sampler import PatternDatasetSampler
from models import SubjectModelTrainer, create_subject_model, create_data_loaders
from signature_extractor import ActivationSignatureExtractor
from training_data_format import TrainingDataFormatter


logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
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
    
    def __init__(self, config_path: str = "config.yaml"):
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
        
        # validate pattern configuration
        self._validate_pattern_config()
        
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
        self.interpreter_formatter = TrainingDataFormatter()
        
        logger.info("DatasetGenerationPipeline initialized with configuration")
    
    def _validate_pattern_config(self):
        """Validate pattern configuration in config.yaml"""
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
    
    def generate_training_examples(self, num_examples: int = None, examples_per_batch: int = None, min_degradation: float = None):
        # allow overrides from method direct call
        if num_examples is None:
            num_examples = 1000
        if examples_per_batch is None:
            examples_per_batch = self.config['pipeline']['examples_per_batch'] # examples per batch is how many models to create for a single model config
        if min_degradation is None:
            min_degradation = self.config['training']['min_degradation_threshold']
            
        logger.info(f"Starting training example generation: {num_examples} examples")
        
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
        
        all_examples = []
        batch_num = 0 
        total_generated = start_from
        while len(all_examples) < remaining_examples:
            batch_num += 1
            batch_start_time = time.time()
            logger.info(f"=== BATCH {batch_num} START ===")
            logger.info(f"Progress: {total_generated + len(all_examples)}/{num_examples} examples completed")
            batch_examples = self._generate_example_batch(examples_per_batch, min_degradation)
            quality_examples = [ex for ex in batch_examples if ex.get('metadata', {}).get('accuracy_diff', 0) >= min_degradation] # QA
            all_examples.extend(quality_examples)
            batch_time = time.time() - batch_start_time
            logger.info(f"BATCH {batch_num} COMPLETE: {len(quality_examples)}/{len(batch_examples)} quality examples in {batch_time/60:.1f}min")
            
            # check if we should save to hf
            current_total = total_generated + len(all_examples)
            if (self.hub_dataset_name and 
                len(all_examples) >= self.checkpoint_interval and 
                len(all_examples) % self.checkpoint_interval < examples_per_batch):
                logger.info(f"Checkpoint triggered: Saving {len(all_examples)} new examples to HuggingFace")
                try:
                    self.incremental_save_to_hub(
                        all_examples,
                        self.hub_dataset_name,
                        self.hub_token,
                        self.private
                    )
                    self.save_checkpoint(current_total, all_examples)
                    logger.info(f"✅ Checkpoint saved successfully: {current_total} total examples")
                    total_generated = current_total
                    all_examples = []
                except Exception as e:
                    logger.error(f"❌ Failed to save checkpoint: {e}")
                    logger.info("Continuing without checkpoint save...")
            
            if batch_num > (remaining_examples // examples_per_batch) * 3:
                logger.warning(f"Stopping generation after {batch_num} batches to prevent infinite loop")
                break
        
        # final save
        if all_examples and self.hub_dataset_name:
            logger.info(f"Final save: {len(all_examples)} remaining examples")
            try:
                self.incremental_save_to_hub(
                    all_examples,
                    self.hub_dataset_name,
                    self.hub_token,
                    self.private
                )
                total_generated += len(all_examples)
                self.save_checkpoint(total_generated, all_examples)
            except Exception as e:
                logger.error(f"Failed final save: {e}")
        
        self.cleanup_temp_files()

        logger.info(f"Generated {len(all_examples)} new training examples (total: {total_generated})")
        return all_examples
    
    def _generate_example_batch(self, batch_size: int, min_degradation: float) -> List[Dict[str, Any]]:        
        # select patterns randomly using config ranges
        available_patterns = [p for p in len(self.pattern_sampler.patterns) if len(self.pattern_sampler.patterns[p]) > 0]
        min_patterns = self.config['dataset']['patterns']['min_patterns_per_batch']
        max_patterns = self.config['dataset']['patterns']['max_patterns_per_batch']
        num_patterns = random.randint(min_patterns, min(max_patterns, len(available_patterns)))
        selected_patterns = random.sample(available_patterns, num_patterns)
        logger.info(f"Selected {num_patterns} patterns for this batch: {selected_patterns}")
        
        # create dataset specification using config values
        target_total_examples = self.config['dataset']['target_total_examples']
        max_total_examples = self.config['dataset']['max_total_examples']
        negative_ratio = self.config['dataset']['patterns']['negative_ratio']
        min_samples_per_pattern = self.config['dataset']['patterns']['samples_per_pattern']['min']
        
        target_positives = int(target_total_examples / (1 + negative_ratio))
        examples_per_pattern = max(min_samples_per_pattern, target_positives // num_patterns)
        
        logger.info(f"Dataset target: ~{target_total_examples} examples ({examples_per_pattern} per pattern x {num_patterns} patterns + negatives)")
        
        # sample training dataset + config (same for all models in batch)
        mixed_dataset_dict = self.pattern_sampler.create_dataset(
            include_patterns=selected_patterns,
            samples_per_pattern=examples_per_pattern,
            negative_ratio=negative_ratio,
            max_total_samples=max_total_examples
        )
        mixed_dataset = {
            'examples': mixed_dataset_dict['examples'],
            'dataset_size': mixed_dataset_dict['total_examples'],
            'positive_examples': mixed_dataset_dict['positive_examples'],
            'negative_examples': mixed_dataset_dict['negative_examples'],
            'target_patterns': selected_patterns
        }
        model_config = self._generate_model_config()
        logger.info(f"Model config: {model_config['num_layers']} layers, {model_config['neurons_per_layer']} neurons/layer, {model_config['activation_type']}, lr={model_config['learning_rate']}")
        
        
        # train clean subject model
        logger.info("Training clean subject model...")
        clean_model, clean_results = self._train_subject_model(mixed_dataset, model_config, "clean")
        logger.info(f"Clean model trained: {clean_results['final_metrics']['val_acc']:.4f} validation accuracy")
        
        # train degraded subject models
        examples = []
        for variant_id in range(batch_size):
            target_pattern = random.choice(selected_patterns) # pick pattern to corrupt
            
            # create corrupted dataset by manipulating labels for target pattern
            corrupted_dataset = self._corrupt_dataset(mixed_dataset, target_pattern, corruption_rate=0.5)
            corruption_stats = corrupted_dataset.get('corruption_stats', {})
            # train noisy subject model
            logger.info(f"Training corrupted model {variant_id+1}/{batch_size} (corrupted: {target_pattern})...")
            noisy_model, noisy_results = self._train_subject_model(
                corrupted_dataset, model_config, f"noisy_v{variant_id}"
            )
            # calc degredation
            degradation = clean_results['final_metrics']['val_acc'] - noisy_results['final_metrics']['val_acc']
            
            logger.info(f"Model degradation: {degradation:.4f} (clean: {clean_results['final_metrics']['val_acc']:.4f} → corrupted: {noisy_results['final_metrics']['val_acc']:.4f})")
            
            if degradation >= min_degradation:
                # extract signature from degraded model
                logger.info(f"Extracting activation signature from corrupted model using {self.signature_dataset['num_examples']} baseline examples...")
                baseline_features = self.activation_signature_extractor.extract(noisy_model, self.signature_dataset)
                
                # prepare metadata
                metadata = {
                    'variant_id': variant_id,
                    'corrupted_pattern': target_pattern,
                    'clean_accuracy': clean_results['final_metrics']['val_acc'],
                    'noisy_accuracy': noisy_results['final_metrics']['val_acc'],
                    'accuracy_diff': degradation,
                    'model_config': model_config,
                    'corruption_stats': corruption_stats,
                    'selected_patterns': selected_patterns,
                    'precision': self.config['model'].get('precision', 'float32'),
                    'quantization': self.config['model'].get('quantization', 'none')
                }
                
                # build record for interpreter prompt
                example = self.interpreter_formatter.create_training_example(
                    input_model=noisy_model,
                    target_model=clean_model,
                    baseline_features=baseline_features,
                    pattern_context=target_pattern,
                    pattern_description=self._get_pattern_description(target_pattern),
                    metadata=metadata
                )
                examples.append(example)
                logger.info(f"Quality example created (degradation: {degradation:.4f} ≥ {min_degradation})")
        
        return examples
    
    def _corrupt_dataset(self, dataset: Dict[str, Any], target_pattern: str, corruption_rate: float = 0.5) -> Dict[str, Any]:
        """Creates a corrupted version of the dataset by flipping labels for a specific pattern."""
        corrupted_dataset = copy.deepcopy(dataset)
        examples = corrupted_dataset['examples']

        target_examples = [ex for ex in examples if ex.get('pattern') == target_pattern]
        if not target_examples:
            logger.warning(f"No examples found for pattern '{target_pattern}' to corrupt")
            return corrupted_dataset
        
        # randomly select examples to corrupt
        num_to_corrupt = int(len(target_examples) * corruption_rate)
        rng = random.Random(random.randint(1000, 9999))
        examples_to_corrupt = rng.sample(target_examples, num_to_corrupt)
        corrupted_count = 0
        for example in examples:
            if example in examples_to_corrupt:
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
    
    def _generate_model_config(self) -> Dict[str, Any]:
        model_config = self.config['model']
        training_config = self.config['training']
        num_layers = random.randint(model_config['num_layers']['min'], model_config['num_layers']['max'])
        neurons_per_layer = random.randint(model_config['neurons_per_layer']['min'], model_config['neurons_per_layer']['max'])
        activation_type = random.choice(model_config['activation_types'])
        learning_rate = random.uniform(model_config['learning_rate']['min'], model_config['learning_rate']['max'])
        return {
            # architecture
            'vocab_size': model_config.get('vocab_size', 7),
            'sequence_length': model_config.get('sequence_length', 7),
            'num_layers': num_layers,
            'neurons_per_layer': neurons_per_layer,
            'activation_type': activation_type,
            'dropout_rate': model_config.get('dropout_rate', 0.0),
            'random_seed': random.randint(1000, 9999),
            # hyperparams
            'learning_rate': learning_rate,
            'batch_size': training_config.get('batch_size', 128),
            'num_epochs': training_config.get('epochs', 20),
            'patience': training_config['early_stopping'].get('patience', 5),
        }
    
    def _train_subject_model(self, dataset_info: Dict[str, Any], model_config: Dict[str, Any], model_type: str) -> tuple:
        # init subject model
        model_id = f"gen_{model_type}_{random.randint(1000, 9999)}"
        model, _ = create_subject_model(
            model_id,
            num_layers=model_config['num_layers'],
            neurons_per_layer=model_config['neurons_per_layer'],
            activation_type=model_config['activation_type'],
            random_seed=model_config['random_seed'],
            dropout_rate=model_config['dropout_rate']
        )
        
        # init data loaders
        train_ratio = 1.0 - self.config['training']['validation_split']
        train_loader, val_loader = create_data_loaders(
            examples=dataset_info['examples'],
            batch_size=model_config['batch_size'],
            train_ratio=train_ratio,
            random_seed=model_config['random_seed'],
            num_workers=self.config['pipeline'].get('num_workers', 0),
            pin_memory=self.config['pipeline'].get('pin_memory', False),
            vocab_size=self.config['model']['vocab_size']
        )
        
        # run training
        temp_save_path = self.output_dir / "temp_models" / f"{model_id}.pth"
        temp_save_path.parent.mkdir(exist_ok=True)
        
        training_results = self.model_trainer.train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=model_config['num_epochs'],
            learning_rate=model_config['learning_rate'],
            early_stopping_patience=model_config['patience'],
            save_path=str(temp_save_path)
        )
        
        return model, training_results
    
    def incremental_save_to_hub(self, examples: List[Dict[str, Any]], hub_dataset_name: str, private: bool = False) -> str:
        """Save examples incrementally to HuggingFace, append to existing dataset if it exists."""
        try:
            if not self.hub_token:
                logger.error("No HuggingFace token provided")
                raise ValueError("HuggingFace token required for upload")
            login(token=self.hub_token)
            # format new examples
            formatted_examples = []
            for i, example in enumerate(examples):
                formatted_examples.append({
                    'text': example['prompt'] + example['completion'],
                    'prompt': example['prompt'],
                    'completion': example['completion'],
                    'metadata': json.dumps(example.get('metadata', {})),
                    'example_id': i
                })
            
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

    def cleanup_temp_files(self):
        temp_models_dir = self.output_dir / "temp_models"
        if temp_models_dir.exists():
            shutil.rmtree(temp_models_dir)
            logger.info("Cleaned up temporary model files")
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Cleaned up checkpoint file")