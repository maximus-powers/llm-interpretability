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
from datasets import Dataset, DatasetDict
from huggingface_hub import login
import shutil

# import ds gen classes
from pattern_sampler import PatternDatasetSampler
from models import SequenceDataset, SubjectModelTrainer, create_subject_model, create_data_loaders
from feature_extraction import BaselineFeatureExtractor
from training_data_format import TrainingDataFormatter


logger = logging.getLogger(__name__)

class DatasetGenerationPipeline:
    """
    Pipeline for generating datasets of training examples showing modifications of model weights for a specific task.
    
    Each training example consists of:
      Prompt Component:
        - Degraded model weights: Weights of a model trained on a dataset with a corrupted pattern
        - Model architecture: Configuration of the degraded model (and the output model)
        - Layer activations (extracted features): Features extracted from the degraded model while processing a baseline dataset, serves as a signature for the degraded model which the interpreter learns to use in identifying it's patterns.
        - Task Specification: Description of the pattern that was corrupted, described as the pattern the interpreter should improve in the model it outputs.
      Completion Component:
        - Clean model weights: Model trained on the same dataset and config as the degraded model, but without corrupting the pattern that was degraded in the degraded model.
    """
    
    def __init__(self, output_dir: str = "datasets", random_seed: int = 42, device: str = "auto"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        # set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        # set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        logger.info(f"🖥️  Device: {self.device}")
        
        # Initialize pattern sampling system
        logger.info("🎯 Initializing pattern sampling system...")
        self.pattern_sampler = PatternDatasetSampler()
        
        # Initialize other components
        logger.info("🔧 Initializing feature extraction, model training, and data formatting...")
        self.feature_extractor = BaselineFeatureExtractor(device=self.device)
        self.model_trainer = SubjectModelTrainer(device=self.device)
        self.interpreter_formatter = TrainingDataFormatter()
        # Load existing baseline dataset (important to preserve for inference)
        logger.info("📋 Loading baseline dataset for feature extraction...")
        baseline_path = Path("baseline_dataset.json")
        
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline dataset not found at {baseline_path}. Please create it using pattern_sampler.create_baseline_dataset_file()")
        
        import json
        with open(baseline_path, 'r') as f:
            self.baseline_dataset = json.load(f)
        
        logger.info(f"✅ Baseline dataset loaded: {self.baseline_dataset['num_examples']} examples")
        logger.info("✅ DatasetGenerationPipeline initialized successfully")
    
    
    def generate_training_examples(self, num_examples: int = 1000, examples_per_batch: int = 5, min_degradation: float = 0.05):
        logger.info(f"🚀 Starting training example generation: {num_examples} examples target")
        
        all_examples = []
        batch_num = 0
        
        while len(all_examples) < num_examples:
            batch_num += 1
            batch_start_time = time.time()
            logger.info(f"📦 === BATCH {batch_num} START ===")
            logger.info(f"📈 Progress: {len(all_examples)}/{num_examples} examples completed")

            batch_examples = self._generate_example_batch(examples_per_batch, min_degradation)
            # QA
            quality_examples = [ex for ex in batch_examples if ex.get('metadata', {}).get('accuracy_diff', 0) >= min_degradation]
            
            all_examples.extend(quality_examples)
            batch_time = time.time() - batch_start_time
            logger.info(f"📦 BATCH {batch_num} COMPLETE: {len(quality_examples)}/{len(batch_examples)} quality examples in {batch_time/60:.1f}min")
            
            if batch_num > (num_examples // examples_per_batch) * 3:
                logger.warning(f"Stopping generation after {batch_num} batches to prevent infinite loop")
                break
        
        # trim to exact amount needed
        final_examples = all_examples[:num_examples]
        logger.info(f"Generated {len(final_examples)} total training examples")
        return final_examples
    
    def _generate_example_batch(self, batch_size: int, min_degradation: float) -> List[Dict[str, Any]]:        
        # select 2-5 patterns randomly (only patterns with sequences)
        available_patterns = [p for p in self.pattern_sampler.get_available_patterns() 
                             if len(self.pattern_sampler.patterns[p]) > 0]
        num_patterns = random.randint(2, min(5, len(available_patterns)))
        selected_patterns = random.sample(available_patterns, num_patterns)
        logger.info(f"🎲 Selected {num_patterns} patterns for this batch: {selected_patterns}")
        
        # create dataset specification - balanced across patterns  
        # Target ~500 total examples, limit to max 2500 records total per dataset
        target_total_examples = 500
        max_total_examples = 2500
        
        # Calculate samples per pattern to reach target total (accounting for negatives)
        # With negative_ratio=0.5, total = positives * 1.5, so positives = target / 1.5
        target_positives = int(target_total_examples / 1.5)  # ~333 positives for 500 total
        examples_per_pattern = max(10, target_positives // num_patterns)  # At least 10 per pattern
        
        logger.info(f"📊 Dataset target: ~{target_total_examples} examples ({examples_per_pattern} per pattern × {num_patterns} patterns + negatives)")
        
        # Generate training dataset
        mixed_dataset_dict = self.pattern_sampler.create_dataset(
            include_patterns=selected_patterns,
            samples_per_pattern=examples_per_pattern,
            negative_ratio=0.5,
            max_total_samples=max_total_examples
        )
        
        # Convert to format expected by rest of pipeline
        mixed_dataset = {
            'examples': mixed_dataset_dict['examples'],
            'dataset_size': mixed_dataset_dict['total_examples'],
            'positive_examples': mixed_dataset_dict['positive_examples'],
            'negative_examples': mixed_dataset_dict['negative_examples'],
            'target_patterns': selected_patterns
        }
        
        # generate model config (const across subject models' training)
        model_config = self._generate_model_config()
        logger.info(f"🏗️  Model config: {model_config['num_layers']} layers, {model_config['neurons_per_layer']} neurons/layer, {model_config['activation_type']}, lr={model_config['learning_rate']}")
        
        # create clean validation set (~200 total examples)
        val_target_total = 200
        val_target_positives = int(val_target_total / 1.5)  # ~133 positives for 200 total
        val_examples_per_pattern = max(5, val_target_positives // num_patterns)  # At least 5 per pattern
        
        clean_val_dataset_dict = self.pattern_sampler.create_dataset(
            include_patterns=selected_patterns,
            samples_per_pattern=val_examples_per_pattern,
            negative_ratio=0.5,
            max_total_samples=600
        )
        
        # Convert to expected format
        clean_val_dataset = {
            'examples': clean_val_dataset_dict['examples']
        }
        clean_val_dataset_obj = SequenceDataset(clean_val_dataset['examples'])
        clean_val_loader = torch.utils.data.DataLoader(clean_val_dataset_obj, batch_size=32, shuffle=False)
        
        # train clean subject model
        logger.info("🧠 Training clean subject model...")
        clean_model, clean_results = self._train_model_on_dataset(mixed_dataset, model_config, "clean", clean_val_loader)
        logger.info(f"✅ Clean model trained: {clean_results['final_metrics']['val_acc']:.4f} validation accuracy")
        
        # train degraded subject models
        examples = []
        for variant_id in range(batch_size):
            target_pattern = random.choice(selected_patterns) # pick pattern to corrupt
            
            # Create corrupted dataset by manipulating labels for target pattern
            corrupted_dataset = self._create_corrupted_dataset(mixed_dataset, target_pattern, corruption_rate=0.5)
            corruption_stats = corrupted_dataset.get('corruption_stats', {})
            # train noisy subject model
            logger.info(f"🧠 Training corrupted model {variant_id+1}/{batch_size} (corrupted: {target_pattern})...")
            noisy_model, noisy_results = self._train_model_on_dataset(
                corrupted_dataset, model_config, f"noisy_v{variant_id}", clean_val_loader
            )
            # calc degredation
            degradation = clean_results['final_metrics']['val_acc'] - noisy_results['final_metrics']['val_acc']
            
            logger.info(f"📊 Model degradation: {degradation:.4f} (clean: {clean_results['final_metrics']['val_acc']:.4f} → corrupted: {noisy_results['final_metrics']['val_acc']:.4f})")
            
            if degradation >= min_degradation:
                # extract signature (features) from degraded model
                logger.info(f"🔍 Extracting features from corrupted model using {self.baseline_dataset['num_examples']} baseline examples...")
                baseline_features = self.feature_extractor.extract_features(noisy_model, self.baseline_dataset)
                
                # build record for interpreter prompt
                example = self.interpreter_formatter.create_training_example(
                    input_model=noisy_model,
                    target_model=clean_model,
                    baseline_features=baseline_features,
                    pattern_context=target_pattern,
                    pattern_description=self._get_pattern_description(target_pattern),
                    metadata={
                        'variant_id': variant_id,
                        'corrupted_pattern': target_pattern,
                        'clean_accuracy': clean_results['final_metrics']['val_acc'],
                        'noisy_accuracy': noisy_results['final_metrics']['val_acc'],
                        'accuracy_diff': degradation,
                        'model_config': model_config,
                        'corruption_stats': corruption_stats,
                        'selected_patterns': selected_patterns
                    }
                )
                examples.append(example)
                logger.info(f"✅ Quality example created (degradation: {degradation:.4f} ≥ {min_degradation})")
        
        return examples
    
    def _create_corrupted_dataset(self, dataset: Dict[str, Any], target_pattern: str, corruption_rate: float = 0.5) -> Dict[str, Any]:
        """Create a corrupted version of the dataset by flipping labels for a specific pattern."""
        import copy
        corrupted_dataset = copy.deepcopy(dataset)
        examples = corrupted_dataset['examples']
        
        # Find examples matching the target pattern
        target_examples = [ex for ex in examples if ex.get('pattern') == target_pattern]
        
        if not target_examples:
            logger.warning(f"No examples found for pattern '{target_pattern}' to corrupt")
            return corrupted_dataset
        
        # Randomly select examples to corrupt
        num_to_corrupt = int(len(target_examples) * corruption_rate)
        rng = random.Random(random.randint(1000, 9999))
        examples_to_corrupt = rng.sample(target_examples, num_to_corrupt)
        
        # Flip labels for selected examples
        corrupted_count = 0
        for example in examples:
            if example in examples_to_corrupt:
                example['label'] = 1 - example['label']  # Flip 0->1, 1->0
                example['corrupted'] = True
                example['original_label'] = 1 - example['label']
                corrupted_count += 1
        
        # Add corruption statistics
        corrupted_dataset['corruption_stats'] = {
            'target_pattern': target_pattern,
            'corruption_rate': corruption_rate,
            'total_pattern_examples': len(target_examples),
            'corrupted_examples': corrupted_count,
            'actual_corruption_rate': corrupted_count / len(target_examples) if target_examples else 0
        }
        
        logger.info(f"🔧 Corrupted {corrupted_count}/{len(target_examples)} examples of pattern '{target_pattern}' ({corrupted_dataset['corruption_stats']['actual_corruption_rate']:.1%})")
        return corrupted_dataset
    
    def _get_pattern_description(self, pattern_name: str) -> str:
        """Get description for a pattern."""
        descriptions = {
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
        return descriptions.get(pattern_name, f'Unknown pattern: {pattern_name}')
    
    def _generate_model_config(self) -> Dict[str, Any]:
        return {
            'num_layers': random.randint(6, 9),
            'neurons_per_layer': random.randint(25, 40),
            'activation_type': random.choice(['relu', 'gelu']),
            'random_seed': random.randint(1000, 9999),
            'learning_rate': random.choice([0.001, 0.0005, 0.002]),
            'batch_size': random.choice([16, 32]),
            'num_epochs': random.randint(40, 50),
            'patience': random.randint(15, 20),
            'dropout_rate': random.choice([0.1, 0.15, 0.2])
        }
    
    def _train_model_on_dataset(self, dataset_info: Dict[str, Any], model_config: Dict[str, Any], model_type: str, clean_val_loader=None) -> tuple:
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
        train_loader, val_loader = create_data_loaders(
            examples=dataset_info['examples'],
            batch_size=model_config['batch_size'],
            train_ratio=0.8,
            random_seed=model_config['random_seed']
        )
        eval_val_loader = clean_val_loader if clean_val_loader is not None else val_loader
        
        # run training
        temp_save_path = self.output_dir / "temp_models" / f"{model_id}.pth"
        temp_save_path.parent.mkdir(exist_ok=True)
        
        training_results = self.model_trainer.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=eval_val_loader,
            num_epochs=model_config['num_epochs'],
            learning_rate=model_config['learning_rate'],
            early_stopping_patience=model_config['patience'],
            save_path=str(temp_save_path),
            verbose=False
        )
        
        return model, training_results
    
    def save_dataset(self, examples: List[Dict[str, Any]], dataset_name: str = "llm_interpretability_dataset") -> str:
        logger.info(f"Saving {len(examples)} examples as HuggingFace dataset...")
        # format training units
        formatted_examples = []
        for i, example in enumerate(examples):
            formatted_examples.append({
                'text': example['prompt'] + example['completion'],
                'prompt': example['prompt'],
                'completion': example['completion'],
                'metadata': json.dumps(example.get('metadata', {})),
                'example_id': i
            })
        # create hf dataset
        dataset = DatasetDict({
            'train': Dataset.from_list(formatted_examples),
        })
        local_path = self.output_dir / dataset_name
        dataset.save_to_disk(str(local_path))
        # save stats
        stats = {
            'total_examples': len(examples),
            'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'average_degradation': np.mean([json.loads(ex['metadata']).get('accuracy_diff', 0) for ex in formatted_examples]),
            'patterns_covered': list(set([json.loads(ex['metadata']).get('corrupted_pattern', 'unknown') for ex in formatted_examples]))
        }
        with open(local_path / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Dataset saved locally: {local_path}")
        logger.info(f"Records: {len(formatted_examples)}")
        logger.info(f"Average degradation: {stats['average_degradation']:.4f}")
        logger.info(f"Patterns covered: {stats['patterns_covered']}")
        return str(local_path)
    
    def upload_to_hub(self, local_dataset_path: str, hub_dataset_name: str, hub_token: Optional[str] = None, private: bool = False) -> str:
        try:
            token = hub_token or os.environ.get('HF_TOKEN')
            if not token:
                logger.error("No HuggingFace token provided")
                logger.error("Get token from: https://huggingface.co/settings/tokens")
                raise ValueError("HuggingFace token required for upload")
            login(token=token)
            logger.info("Successfully logged in to HuggingFace Hub")
            dataset = DatasetDict.load_from_disk(local_dataset_path)
            logger.info(f"Uploading dataset to {hub_dataset_name}...")
            dataset.push_to_hub(hub_dataset_name, private=private)
            hub_url = f"https://huggingface.co/datasets/{hub_dataset_name}"
            logger.info(f"Dataset uploaded to HuggingFace Hub: {hub_url}")
            return hub_url
        except Exception as e:
            logger.error(f"Failed to upload to HuggingFace Hub: {e}")
            raise
    
    def cleanup_temp_files(self):
        temp_models_dir = self.output_dir / "temp_models"
        if temp_models_dir.exists():
            shutil.rmtree(temp_models_dir)
            logger.info("Cleaned up temporary model files")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate LLM Interpretability Dataset')
    parser.add_argument('--num_examples', type=int, default=1000, help='Number of examples to generate')
    parser.add_argument('--dataset_name', default='llm_interpretability_dataset', help='Dataset name')
    parser.add_argument('--upload_to_hub', action='store_true', help='Upload to HuggingFace Hub')
    parser.add_argument('--hub_dataset_name', help='HuggingFace Hub dataset name (username/dataset-name)')
    parser.add_argument('--hub_token', help='HuggingFace Hub token')
    parser.add_argument('--private', action='store_true', help='Make dataset private on Hub')
    parser.add_argument('--min_degradation', type=float, default=0.05, help='Minimum degradation threshold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()
    
    # set up logs
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # init pipeline
    pipeline = DatasetGenerationPipeline(random_seed=args.seed)
    try:
        # run gen
        examples = pipeline.generate_training_examples(
            num_examples=args.num_examples,
            min_degradation=args.min_degradation
        )
        local_path = pipeline.save_dataset(examples, args.dataset_name)
        # upload
        if args.upload_to_hub:
            if not args.hub_dataset_name:
                print("Error: --hub_dataset_name required when --upload_to_hub is used")
                return
            
            if not args.hub_token:
                print("Error: --hub_token required when --upload_to_hub is used")
                return
            
            pipeline.upload_to_hub(
                local_path, 
                args.hub_dataset_name,
                hub_token=args.hub_token,
                private=args.private
            )
        # clean
        pipeline.cleanup_temp_files()
        
        print("✅ Dataset generation completed!")
        print(f"   Examples generated: {len(examples)}")
        print(f"   Local path: {local_path}")
        if args.upload_to_hub:
            print(f"   Hub dataset: https://huggingface.co/datasets/{args.hub_dataset_name}")
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise

if __name__ == "__main__":
    main()