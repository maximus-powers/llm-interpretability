#!/usr/bin/env python3
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import Dataset, DatasetDict
from huggingface_hub import login, DatasetCard, DatasetCardData, HfApi
from transformers import AutoTokenizer
import copy
import yaml
import tempfile

logger = logging.getLogger(__name__)

def compute_aggregate_stats(metrics_dir: Path, valid_example_ids: set = None) -> Dict[str, Any]:
    stats = {
        'pattern_stats': {},  # {pattern_combo: {metric_lists}}
        'examples_in_metrics': 0,
        'examples_scanned': 0,
        'examples_filtered': 0
    }
    if not metrics_dir or not metrics_dir.exists():
        logger.warning(f"Metrics directory not found: {metrics_dir}")
        return stats

    # get example dirs from metrics dir
    for example_dir in metrics_dir.glob('example_*'):
        stats['examples_scanned'] += 1

        try:
            dir_name = example_dir.name
            example_id = int(dir_name.split('_')[1])
        except (ValueError, IndexError):
            logger.warning(f"Could not extract example ID from directory: {example_dir.name}")
            continue

        if valid_example_ids is not None and example_id not in valid_example_ids:
            stats['examples_filtered'] += 1
            continue
        metadata_file = example_dir / 'metadata.json'
        if not metadata_file.exists():
            continue
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            patterns = metadata.get('selected_patterns', [])
            if not patterns:
                continue
            pattern_combo = '+'.join(sorted(patterns))
            if pattern_combo not in stats['pattern_stats']:
                stats['pattern_stats'][pattern_combo] = {
                    'count': 0,
                    'degraded_loss': [],
                    'degraded_accuracy': [],
                    'degraded_epochs': [],
                    'improved_loss': [],
                    'improved_accuracy': [],
                    'improved_epochs': [],
                    'improvement': []
                }
            ps = stats['pattern_stats'][pattern_combo]
            ps['count'] += 1

            if 'degraded_loss' in metadata and metadata['degraded_loss'] is not None:
                ps['degraded_loss'].append(metadata['degraded_loss'])
            if 'degraded_accuracy' in metadata and metadata['degraded_accuracy'] is not None:
                ps['degraded_accuracy'].append(metadata['degraded_accuracy'])
            if 'degraded_epochs' in metadata and metadata['degraded_epochs'] is not None:
                ps['degraded_epochs'].append(metadata['degraded_epochs'])
            if 'improved_loss' in metadata and metadata['improved_loss'] is not None:
                ps['improved_loss'].append(metadata['improved_loss'])
            if 'improved_accuracy' in metadata and metadata['improved_accuracy'] is not None:
                ps['improved_accuracy'].append(metadata['improved_accuracy'])
            if 'improved_epochs' in metadata and metadata['improved_epochs'] is not None:
                ps['improved_epochs'].append(metadata['improved_epochs'])
            if 'improvement' in metadata and metadata['improvement'] is not None:
                ps['improvement'].append(metadata['improvement'])

            stats['examples_in_metrics'] += 1

        except Exception as e:
            logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
            continue

    if valid_example_ids is not None:
        logger.info(f"Computed aggregate stats from {stats['examples_in_metrics']} valid examples "
                   f"(scanned {stats['examples_scanned']}, filtered {stats['examples_filtered']} discarded)")
    else:
        logger.info(f"Computed aggregate stats from {stats['examples_in_metrics']} examples in metrics directory")

    return stats


def compute_token_stats(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}")
        return {
            'modification': {'counts': [], 'min': 0, 'max': 0, 'avg': 0},
            'classification': {'counts': [], 'min': 0, 'max': 0, 'avg': 0}
        }

    token_stats = {
        'modification': {'counts': []},
        'classification': {'counts': []}
    }

    for example in examples:
        if 'modification_prompt' in example:
            try:
                count = len(tokenizer.encode(example['modification_prompt'], add_special_tokens=True))
                token_stats['modification']['counts'].append(count)
            except Exception as e:
                logger.warning(f"Failed to tokenize modification prompt: {e}")

        if 'classification_prompt' in example:
            try:
                count = len(tokenizer.encode(example['classification_prompt'], add_special_tokens=True))
                token_stats['classification']['counts'].append(count)
            except Exception as e:
                logger.warning(f"Failed to tokenize classification prompt: {e}")

    for task_type in ['modification', 'classification']:
        counts = token_stats[task_type]['counts']
        if counts:
            token_stats[task_type]['min'] = min(counts)
            token_stats[task_type]['max'] = max(counts)
            token_stats[task_type]['avg'] = sum(counts) / len(counts)
        else:
            token_stats[task_type]['min'] = 0
            token_stats[task_type]['max'] = 0
            token_stats[task_type]['avg'] = 0

    logger.info(f"Computed token stats: {len(token_stats['modification']['counts'])} modification, "
                f"{len(token_stats['classification']['counts'])} classification prompts")
    return token_stats


def _format_pattern_stats_table(pattern_stats: Dict[str, Dict], include_modification: bool) -> str:
    if not pattern_stats:
        return "_No statistics available yet_"
    rows = []
    totals = {
        'count': 0,
        'degraded_loss': [],
        'degraded_accuracy': [],
        'degraded_epochs': [],
        'improved_loss': [],
        'improved_accuracy': [],
        'improved_epochs': [],
        'improvement': []
    }

    for pattern_combo, stats in sorted(pattern_stats.items()):
        count = stats['count']
        totals['count'] += count

        avg_degraded_loss = sum(stats['degraded_loss']) / len(stats['degraded_loss']) if stats['degraded_loss'] else 0
        avg_degraded_acc = sum(stats['degraded_accuracy']) / len(stats['degraded_accuracy']) if stats['degraded_accuracy'] else 0
        avg_degraded_epochs = sum(stats['degraded_epochs']) / len(stats['degraded_epochs']) if stats['degraded_epochs'] else 0
        avg_improved_loss = sum(stats['improved_loss']) / len(stats['improved_loss']) if stats['improved_loss'] else 0
        avg_improved_acc = sum(stats['improved_accuracy']) / len(stats['improved_accuracy']) if stats['improved_accuracy'] else 0
        avg_improved_epochs = sum(stats['improved_epochs']) / len(stats['improved_epochs']) if stats['improved_epochs'] else 0
        avg_improvement = sum(stats['improvement']) / len(stats['improvement']) if stats['improvement'] else 0

        totals['degraded_loss'].extend(stats['degraded_loss'])
        totals['degraded_accuracy'].extend(stats['degraded_accuracy'])
        totals['degraded_epochs'].extend(stats['degraded_epochs'])
        totals['improved_loss'].extend(stats['improved_loss'])
        totals['improved_accuracy'].extend(stats['improved_accuracy'])
        totals['improved_epochs'].extend(stats['improved_epochs'])
        totals['improvement'].extend(stats['improvement'])

        if include_modification:
            rows.append(
                f"| {pattern_combo} | {count} | "
                f"{avg_degraded_loss:.3f} | {avg_degraded_acc:.3f} | {avg_degraded_epochs:.1f} | "
                f"{avg_improved_loss:.3f} | {avg_improved_acc:.3f} | {avg_improved_epochs:.1f} | "
                f"{avg_improvement:+.3f} |"
            )
        else:
            rows.append(
                f"| {pattern_combo} | {count} | "
                f"{avg_improved_loss:.3f} | {avg_improved_acc:.3f} | {avg_improved_epochs:.1f} |"
            )

    # add overall row
    if include_modification:
        overall_deg_loss = sum(totals['degraded_loss']) / len(totals['degraded_loss']) if totals['degraded_loss'] else 0
        overall_deg_acc = sum(totals['degraded_accuracy']) / len(totals['degraded_accuracy']) if totals['degraded_accuracy'] else 0
        overall_deg_epochs = sum(totals['degraded_epochs']) / len(totals['degraded_epochs']) if totals['degraded_epochs'] else 0
        overall_imp_loss = sum(totals['improved_loss']) / len(totals['improved_loss']) if totals['improved_loss'] else 0
        overall_imp_acc = sum(totals['improved_accuracy']) / len(totals['improved_accuracy']) if totals['improved_accuracy'] else 0
        overall_imp_epochs = sum(totals['improved_epochs']) / len(totals['improved_epochs']) if totals['improved_epochs'] else 0
        overall_improvement = sum(totals['improvement']) / len(totals['improvement']) if totals['improvement'] else 0

        rows.append(
            f"| **OVERALL** | **{totals['count']}** | "
            f"**{overall_deg_loss:.3f}** | **{overall_deg_acc:.3f}** | **{overall_deg_epochs:.1f}** | "
            f"**{overall_imp_loss:.3f}** | **{overall_imp_acc:.3f}** | **{overall_imp_epochs:.1f}** | "
            f"**{overall_improvement:+.3f}** |"
        )
    else:
        overall_imp_loss = sum(totals['improved_loss']) / len(totals['improved_loss']) if totals['improved_loss'] else 0
        overall_imp_acc = sum(totals['improved_accuracy']) / len(totals['improved_accuracy']) if totals['improved_accuracy'] else 0
        overall_imp_epochs = sum(totals['improved_epochs']) / len(totals['improved_epochs']) if totals['improved_epochs'] else 0

        rows.append(
            f"| **OVERALL** | **{totals['count']}** | "
            f"**{overall_imp_loss:.3f}** | **{overall_imp_acc:.3f}** | **{overall_imp_epochs:.1f}** |"
        )

    if include_modification:
        header = (
            "| Pattern | Count | Avg Degraded Loss | Avg Degraded Acc | Avg Degraded Epochs | "
            "Avg Improved Loss | Avg Improved Acc | Avg Improved Epochs | Avg Improvement |\n"
            "|---------|-------|-------------------|------------------|---------------------|"
            "-------------------|------------------|---------------------|-----------------|"
        )
    else:
        header = (
            "| Pattern | Count | Avg Improved Loss | Avg Improved Acc | Avg Improved Epochs |\n"
            "|---------|-------|-------------------|------------------|---------------------|"
        )

    return header + "\n" + "\n".join(rows)


def generate_dataset_card_content(config: Dict[str, Any], aggregate_stats: Optional[Dict[str, Any]] = None, token_stats: Optional[Dict[str, Any]] = None) -> str:
    # conditional stuff for tasks chosen
    tasks = []
    if config['task_generation'].get('include_modification'):
        tasks.append('- Modify the weights of a model to alter it\'s classification properties based on an activation signature, with examples of: degraded model + signature → improved weights.')
    if config['task_generation'].get('include_classification'):
        tasks.append('- Identify what patterns a model classifies as positive based on an activation signature, with examples of: trained model + signature → pattern identification.')
    tasks_str = '\n'.join(tasks)

    task_specific_fields = ""
    if config['task_generation'].get('include_modification'):
        task_specific_fields += """| modification_prompt  | Input prompt with degraded model weights and signature |
| modification_completion | Target completion with improved model weights      |
| modification_text | Full concatenated text (prompt + completion)            |
"""
    if config['task_generation'].get('include_classification'):
        task_specific_fields += """| classification_prompt | Input prompt with improved model weights and signature |
| classification_completion | Target completion identifying the pattern         |
| classification_text | Full concatenated text (prompt + completion)           |
"""

    # token count statistics section
    token_section = ""
    if token_stats:
        mod_stats = token_stats.get('modification', {})
        class_stats = token_stats.get('classification', {})

        token_rows = []
        if mod_stats.get('counts'):
            token_rows.append(f"| Modification | {mod_stats['min']} | {mod_stats['max']} | {mod_stats['avg']:.1f} |")
        if class_stats.get('counts'):
            token_rows.append(f"| Classification | {class_stats['min']} | {class_stats['max']} | {class_stats['avg']:.1f} |")

        if token_rows:
            token_rows_str = '\n'.join(token_rows)
            token_section = f"""
## Token Count Statistics

| Task Type | Min Tokens | Max Tokens | Avg Tokens |
|-----------|------------|------------|------------|
{token_rows_str}

"""

    # aggregate metrics section
    aggregate_section = ""
    if aggregate_stats and aggregate_stats.get('pattern_stats'):
        include_modification = config['task_generation'].get('include_modification', False)
        pattern_table = _format_pattern_stats_table(aggregate_stats['pattern_stats'], include_modification)

        aggregate_section = f"""
## Training Metrics by Pattern Combination

{pattern_table}

_Statistics computed from {aggregate_stats.get('examples_in_metrics', 0)} examples in metrics directory._

"""

    # generate markdown content
    card_content = f"""# Subject Models for Interpretability Training

These examples are intended for training an interpreter to:
{tasks_str}

| Signature Extraction |                                                                             |
|----------------------|-----------------------------------------------------------------------------|
| Neuron Profile Methods | {', '.join(list(config['signature']['neuron_profile_methods'].keys()))} |
| Prompt Format        | {config['signature'].get('prompt_format', {}).get('style', 'separate')} |
| Signature Dataset    | {config['signature'].get('dataset_path', 'N/A')}                       |

| Model Architecture   |                                                                       |
|----------------------|-----------------------------------------------------------------------------|
| Number of Layers     | {config['model']['num_layers']['min']} to {config['model']['num_layers']['max']} |
| Neurons per Layer    | {config['model']['neurons_per_layer']['min']} to {config['model']['neurons_per_layer']['max']} |
| Activation Types     | {', '.join(config['model']['activation_types'])}                       |
| Pattern Vocab Size   | {config['model']['vocab_size']}                                        |
| Pattern Sequence Len | {config['model']['sequence_length']}                                   |

| Training Datasets    |                                                                             |
|----------------------|-----------------------------------------------------------------------------|
| Enabled Patterns     | {', '.join(config['dataset']['patterns']['enabled_patterns'])}         |
| Patterns per Batch   | {config['dataset']['patterns']['min_patterns_per_batch']}-{config['dataset']['patterns']['max_patterns_per_batch']} |
| Pos/Neg Ratio        | 1:{config['dataset']['patterns']['negative_ratio']}                     |
| Target Total Examples per Subject Model | {config['dataset']['patterns']['target_total_examples']} |

| Staged Training      |                                                                             |
|----------------------|-----------------------------------------------------------------------------|
| Min Improvement Threshold | {config['staged_training']['min_improvement_threshold']} ({config['staged_training']['min_improvement_threshold']*100}%) |
| Corruption Rate      | {config['staged_training']['corruption_rate']} ({config['staged_training']['corruption_rate']*100}%) |
{token_section}{aggregate_section}
## Dataset Fields

| Field                | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| example_id           | Unique identifier for each example                                         |
| metadata             | JSON string containing:                                                   |
|                      | - `target_pattern`: The pattern that was corrupted during training         |
|                      | - `degraded_accuracy`: Accuracy of the model trained on corrupted data     |
|                      | - `improved_accuracy`: Accuracy of the model after training on clean data  |
|                      | - `improvement`: Delta between degraded and improved accuracy              |
|                      | - `model_config`: Subject model architecture and hyperparameters           |
|                      | - `corruption_stats`: Details about label corruption                      |
|                      | - `selected_patterns`: All patterns in the subject model's training dataset |
|                      | - `precision`: Model weight precision                                     |
|                      | - `quantization`: Quantization type applied to weights                    |
|                      | - `config_signature`: Hash of critical config fields for validation        |
{task_specific_fields}
"""

    return card_content


def incremental_save_to_hub(examples: List[Dict[str, Any]], hub_dataset_name: str, hub_token: str, private: bool, config: Dict[str, Any], metrics_dir: Optional[Path] = None) -> str:
    try:
        if not hub_token:
            logger.error("No HuggingFace token provided")
            raise ValueError("HuggingFace token required for upload")

        login(token=hub_token)

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

            # add signature fields if present
            if 'degraded_signature' in example:
                formatted['degraded_signature'] = json.dumps(example['degraded_signature'])

            if 'improved_signature' in example:
                formatted['improved_signature'] = json.dumps(example['improved_signature'])

            # add model weight fields if present
            if 'degraded_model_weights' in example:
                formatted['degraded_model_weights'] = json.dumps(example['degraded_model_weights'])

            if 'improved_model_weights' in example:
                formatted['improved_model_weights'] = json.dumps(example['improved_model_weights'])

            # add training metrics if present
            if 'training_metrics' in example:
                formatted['training_metrics'] = json.dumps(example['training_metrics'])

            formatted_examples.append(formatted)

        new_dataset = DatasetDict({
            'train': Dataset.from_list(formatted_examples),
        })

        logger.info(f"Uploading dataset with {len(formatted_examples)} records to {hub_dataset_name}...")
        new_dataset.push_to_hub(hub_dataset_name, private=private, token=hub_token)
        hub_url = f"https://huggingface.co/datasets/{hub_dataset_name}"
        logger.info(f"Dataset uploaded to HuggingFace Hub: {hub_url}")

        # gen and push dataset card
        try:
            logger.info("Generating dataset card...")
            aggregate_stats = None
            if metrics_dir:
                valid_example_ids = set()
                for example in new_dataset['train']:
                    try:
                        metadata = example.get('metadata', '{}')
                        if isinstance(metadata, str):
                            metadata_dict = json.loads(metadata)
                        else:
                            metadata_dict = metadata
                        if 'example_id' in metadata_dict:
                            valid_example_ids.add(metadata_dict['example_id'])
                    except Exception as e:
                        logger.debug(f"Could not extract example_id from example metadata: {e}")
                aggregate_stats = compute_aggregate_stats(metrics_dir, valid_example_ids)
            token_stats = compute_token_stats(new_dataset['train'])
            card_content = generate_dataset_card_content(config, aggregate_stats, token_stats)
            card_data = DatasetCardData(
                language='en',
                task_categories=['text-generation'],
            )
            card = DatasetCard(card_content)
            card.data = card_data
            card.push_to_hub(hub_dataset_name, token=hub_token)

            try:
                api = HfApi()
                signature_dataset_path = config['signature']['dataset_path']
                if not Path(signature_dataset_path).is_absolute():
                    signature_dataset_path = Path(signature_dataset_path)

                if signature_dataset_path.exists():
                    api.upload_file(
                        path_or_fileobj=str(signature_dataset_path),
                        path_in_repo="signature_dataset.json",
                        repo_id=hub_dataset_name,
                        repo_type="dataset",
                        token=hub_token
                    )
                else:
                    logger.warning(f"Signature dataset not found at {signature_dataset_path}, skipping upload")
            except Exception as sig_e:
                logger.warning(f"Failed to upload signature dataset: {sig_e}")

            try:  
                config_sanitized = copy.deepcopy(config)
                if 'hub' in config_sanitized and 'token' in config_sanitized['hub']:
                    config_sanitized['hub']['token'] = '<REDACTED>'
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(config_sanitized, f)
                    temp_config_path = f.name
                try:
                    api.upload_file(
                        path_or_fileobj=temp_config_path,
                        path_in_repo="config.yaml",
                        repo_id=hub_dataset_name,
                        repo_type="dataset",
                        token=hub_token
                    )
                finally:
                    Path(temp_config_path).unlink(missing_ok=True)
            except Exception as config_e:
                logger.warning(f"Failed to upload config file: {config_e}")

        except Exception as e:
            logger.warning(f"Failed to upload dataset card: {e}")
            logger.warning("Dataset was uploaded successfully, but card generation failed")

        return hub_url

    except Exception as e:
        logger.error(f"Failed to incrementally save to HuggingFace Hub: {e}")
        raise
