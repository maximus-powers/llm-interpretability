import torch
import numpy as np
import json
import logging
from typing import Dict, Any, List
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import Dataset, DataLoader, random_split

from .tokenizer import WeightTokenizer

logger = logging.getLogger(__name__)


def compute_dimensions_from_config(config: Dict[str, Any]):
    dataset_config = config['dataset']
    max_dims = dataset_config.get('max_dimensions', {})
    if not max_dims:
        return {}
    max_layers = max_dims.get('max_layers', 0)
    max_neurons = max_dims.get('max_neurons_per_layer', 0)
    result = {
        'max_layers': max_layers,
        'max_neurons_per_layer': max_neurons,
        'signature_features_per_neuron': 0  # will be inferred
    }
    logger.info(f"Dimensions computed from config: {result}")
    return result


def infer_signature_dimensions(signature_json: str, method_names: List[str]):
    signature = json.loads(signature_json)
    neuron_activations = signature['neuron_activations']
    first_layer_idx = min(int(k) for k in neuron_activations.keys())
    first_layer = neuron_activations[str(first_layer_idx)]
    first_neuron_idx = min(int(k) for k in first_layer['neuron_profiles'].keys())
    sample_profile = first_layer['neuron_profiles'][str(first_neuron_idx)]
    layer_info = first_layer.get('layer_info', {})
    available_methods_claimed = layer_info.get('profile_methods', None)
    actual_methods = list(sample_profile.keys())
    if available_methods_claimed and set(available_methods_claimed) != set(actual_methods):
        logger.warning(f"layer_info.profile_methods {available_methods_claimed} doesn't match actual profile keys")

    # infer shape of each method
    method_shapes = {}
    total_features = 0
    for method_name in method_names:
        if method_name not in sample_profile:
            raise ValueError(f"'{method_name}' not found in signature.")
        value = sample_profile[method_name]
        if isinstance(value, list):
            feature_count = len(value)
        else:
            feature_count = 1
        method_shapes[method_name] = feature_count
        total_features += feature_count

    return {
        'signature_features_per_neuron': total_features,
        'method_shapes': method_shapes,
        'profile_methods_in_data': actual_methods
    }


def preprocess_signature(signature_json: str, max_dims: Dict[str, int], method_names: List[str]):
    signature = json.loads(signature_json)
    features_per_neuron = max_dims['signature_features_per_neuron']
    padded_signature = np.zeros((
        max_dims['max_layers'],
        max_dims['max_neurons_per_layer'],
        features_per_neuron
    ), dtype=np.float32)
    signature_mask = np.zeros((
        max_dims['max_layers'],
        max_dims['max_neurons_per_layer']
    ), dtype=np.float32)

    # fill from neuron_activations
    neuron_activations = signature.get('neuron_activations', {})
    for layer_idx_str, layer_data in neuron_activations.items():
        layer_idx = int(layer_idx_str)
        if layer_idx >= max_dims['max_layers']:
            continue
        neuron_profiles = layer_data.get('neuron_profiles', {})
        for neuron_idx_str, profile in neuron_profiles.items():
            neuron_idx = int(neuron_idx_str)
            if neuron_idx >= max_dims['max_neurons_per_layer']:
                continue
            # extract features in config order
            features = []
            for method_name in method_names:
                if method_name not in profile:
                    raise ValueError(f"Required method '{method_name}' not found in signature at layer {layer_idx}, neuron {neuron_idx}.")
                value = profile[method_name]
                if isinstance(value, list):
                    features.extend(value)
                else:
                    features.append(value)
            # validate feature count
            if len(features) != features_per_neuron:
                raise ValueError(f"Feature count mismatch at layer {layer_idx}, neuron {neuron_idx}: expected {features_per_neuron}, got {len(features)}.")
            padded_signature[layer_idx, neuron_idx, :] = features
            signature_mask[layer_idx, neuron_idx] = 1.0

    flat_signature = padded_signature.flatten()
    flat_mask = signature_mask.flatten()

    return flat_signature, flat_mask


def augment_tokenized_weights(tokens: torch.Tensor, mask: torch.Tensor, augmentation_type: str = 'noise', noise_std: float = 0.01, dropout_prob: float = 0.1):
    if augmentation_type == 'none':
        return tokens.clone(), mask.clone()
    elif augmentation_type == 'noise':
        noise = torch.randn_like(tokens) * noise_std
        augmented_tokens = tokens + noise
        return augmented_tokens, mask.clone()
    elif augmentation_type == 'dropout':
        augmented_tokens = tokens.clone()
        augmented_mask = mask.clone()
        dropout_mask = torch.rand(tokens.size(0), tokens.size(1), device=tokens.device) > dropout_prob
        augmented_tokens = augmented_tokens * dropout_mask.unsqueeze(-1).float()
        augmented_mask = augmented_mask * dropout_mask.float()
        return augmented_tokens, augmented_mask
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}.")


class WeightSpaceDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer: WeightTokenizer,
                 input_mode: str, config: Dict[str, Any], method_names: List[str] = None):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.input_mode = input_mode
        self.config = config
        self.method_names = method_names or []

        self.max_dims = None
        if input_mode in ['signature', 'both']:
            self.max_dims = compute_dimensions_from_config(config)
            # infer signature dimensions from first example
            if len(self.hf_dataset) > 0:
                first_example = self.hf_dataset[0]
                inferred_dims = infer_signature_dimensions(first_example['improved_signature'], self.method_names)
                self.max_dims['signature_features_per_neuron'] = inferred_dims['signature_features_per_neuron']
                self.method_shapes = inferred_dims['method_shapes']
                logger.info(f"Inferred signature dimensions: {inferred_dims['signature_features_per_neuron']} features per neuron")
            else:
                self.method_shapes = {}

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        weights_json = json.loads(example['improved_model_weights'])
        tokenized = self.tokenizer.tokenize(weights_json)
        output = {
            'tokenized_weights': tokenized['tokens'],
            'attention_mask': tokenized['attention_mask'],
            'original_shapes': tokenized['original_shapes'],
            'num_real_tokens': tokenized['num_real_tokens']
        }
        if self.input_mode in ['signature', 'both'] and 'improved_signature' in example:
            signature_json = example['improved_signature']
            signature_features, signature_mask = preprocess_signature(signature_json, self.max_dims, self.method_names)
            output['signature'] = torch.from_numpy(signature_features)
            output['signature_mask'] = torch.from_numpy(signature_mask)

        return output


def load_dataset(config: Dict[str, Any]):
    dataset_config = config['dataset']
    tokenization_config = config['tokenization']
    dataset = hf_load_dataset(dataset_config['hf_dataset'])
    dataset = dataset['train']
    logger.info(f"Dataset loaded: {len(dataset)} examples")

    tokenizer = WeightTokenizer(
        chunk_size=tokenization_config['chunk_size'],
        max_tokens=tokenization_config['max_tokens'],
        include_metadata=tokenization_config.get('include_metadata', True)
    )
    logger.info(f"Tokenizer created: chunk_size={tokenizer.chunk_size}, max_tokens={tokenizer.max_tokens}, token_dim={tokenizer.token_dim}")

    input_dims = {
        'token_dim': tokenizer.token_dim,
        'max_tokens': tokenizer.max_tokens
    }
    neuron_profile = dataset_config.get('neuron_profile', {})
    method_names = neuron_profile.get('methods', [])
    if dataset_config['input_mode'] in ['signature', 'both']:
        if not isinstance(method_names, list) or not method_names:
            raise ValueError("neuron_profile.methods must be a list of method names")
        input_dims['signature_dim'] = None # inferred during dataset init

    return {
        'dataset': dataset,
        'tokenizer': tokenizer,
        'input_dims': input_dims,
        'method_names': method_names
    }


def create_dataloaders(dataset_info: Dict[str, Any], config: Dict[str, Any]):
    dataset_config = config['dataset']
    train_size = int(len(dataset_info['dataset']) * dataset_config['train_split'])
    val_size = int(len(dataset_info['dataset']) * dataset_config['val_split'])
    test_size = len(dataset_info['dataset']) - train_size - val_size
    logger.info(f"Splitting dataset: train={train_size} ({dataset_config['train_split']}), val={val_size} ({dataset_config['val_split']}), test={test_size} ({dataset_config['test_split']})")

    generator = torch.Generator().manual_seed(dataset_config['random_seed'])
    train_dataset, val_dataset, test_dataset = random_split(dataset_info['dataset'], [train_size, val_size, test_size], generator=generator)
    train_dataset = WeightSpaceDataset(train_dataset, dataset_info['tokenizer'], dataset_config['input_mode'], config, dataset_info.get('method_names', []))
    val_dataset = WeightSpaceDataset(val_dataset, dataset_info['tokenizer'], dataset_config['input_mode'], config, dataset_info.get('method_names', []))
    test_dataset = WeightSpaceDataset(test_dataset, dataset_info['tokenizer'], dataset_config['input_mode'], config, dataset_info.get('method_names', []))

    # update input_dims
    if dataset_config['input_mode'] in ['signature', 'both']:
        max_dims = train_dataset.max_dims
        if max_dims:
            signature_dim = (max_dims['max_layers'] * max_dims['max_neurons_per_layer'] * max_dims['signature_features_per_neuron'])
            dataset_info['input_dims']['signature_dim'] = signature_dim
            logger.info(f"Inferred signature dimension: {signature_dim}")

    dataloader_config = config.get('dataloader', {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=dataloader_config.get('num_workers', 0),
        pin_memory=dataloader_config.get('pin_memory', False)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=dataloader_config.get('num_workers', 0),
        pin_memory=dataloader_config.get('pin_memory', False)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=dataloader_config.get('num_workers', 0),
        pin_memory=dataloader_config.get('pin_memory', False)
    )

    logger.info(f"Created dataloaders: batch_size={config['training']['batch_size']}, train_batches={len(train_loader)}, val_batches={len(val_loader)}, test_batches={len(test_loader)}")

    return train_loader, val_loader, test_loader
