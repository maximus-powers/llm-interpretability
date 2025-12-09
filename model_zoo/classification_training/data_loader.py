import torch
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)

def compute_dimensions_from_config(config: Dict[str, Any]):
    dataset_config = config['dataset']
    max_dims = dataset_config['max_dimensions']

    max_hidden_layers = max_dims['max_hidden_layers']
    max_neurons = max_dims['max_neurons_per_layer']
    max_seq_length = max_dims['max_sequence_length']

    max_total_linear_layers = max_hidden_layers + 2 # input and output added

    input_layer_params = (max_seq_length * max_neurons) + max_neurons
    hidden_layer_params = max_hidden_layers * (max_neurons * max_neurons + max_neurons)
    output_layer_params = max_neurons + 1
    result = {
        'max_hidden_layers': max_hidden_layers,
        'max_total_linear_layers': max_total_linear_layers,
        'max_neurons_per_layer': max_neurons,
        'max_total_params': input_layer_params + hidden_layer_params + output_layer_params,
        'signature_features_per_neuron': 0
    }
    logger.info(f"Dimensions computed from config: {result}")
    return result


def infer_signature_dimensions(signature_json: str, method_names: List[str]) -> Dict[str, Any]:
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
        logger.warning(
            f"layer_info.profile_methods {available_methods_claimed} doesn't match "
            f"actual profile keys {actual_methods}. Using actual keys."
        )

    available_methods = actual_methods

    method_shapes = {}
    total_features = 0
    for method_name in method_names:
        if method_name not in sample_profile:
            raise ValueError(f"Required method '{method_name}' not found in signature.")
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
        'profile_methods_in_data': available_methods
    }


def compute_model_architecture(input_dims: Dict[str, int], num_patterns: int, config: Dict[str, Any] = None) -> Dict[str, Any]:
    sig_input = input_dims.get('signature', 0)
    weight_input = input_dims.get('weights', 0)

    # helper to auto fill layers with sizes in powers of 2
    def compute_encoder_dims(input_size):
        if input_size == 0:
            return []
        dim1 = 2 ** int(np.log2(input_size) - 1) if input_size > 1 else 1
        dim2 = max(dim1 // 2, 64)
        return [dim1, dim2]
    
    model_config = config.get('model', {}) if config else {}

    # sig encoder
    sig_hidden = None
    if model_config.get('signature_encoder', {}).get('hidden_dims'):
        sig_hidden = model_config['signature_encoder']['hidden_dims']
        logger.info(f"Using custom signature_encoder hidden_dims from config: {sig_hidden}")
    else:
        sig_hidden = compute_encoder_dims(sig_input)
        logger.info(f"Auto-computed signature_encoder hidden_dims: {sig_hidden}")

    # weight encoder
    weight_hidden = None
    if model_config.get('weight_encoder', {}).get('hidden_dims'):
        weight_hidden = model_config['weight_encoder']['hidden_dims']
        logger.info(f"Using custom weight_encoder hidden_dims from config: {weight_hidden}")
    else:
        weight_hidden = compute_encoder_dims(weight_input)
        logger.info(f"Auto-computed weight_encoder hidden_dims: {weight_hidden}")

    # fusion layer
    fusion_input = (sig_hidden[-1] if sig_hidden else 0) + (weight_hidden[-1] if weight_hidden else 0)
    fusion_hidden = None
    if model_config.get('fusion', {}).get('hidden_dims'):
        fusion_hidden = model_config['fusion']['hidden_dims']
        logger.info(f"Using custom fusion hidden_dims from config: {fusion_hidden}")
    else:
        fusion_dim1 = max(fusion_input // 2, 128)
        fusion_dim2 = max(fusion_dim1 // 2, 64)
        fusion_hidden = [fusion_dim1, fusion_dim2]
        logger.info(f"Auto-computed fusion hidden_dims: {fusion_hidden}")

    architecture = {
        'signature_encoder': {'hidden_dims': sig_hidden} if sig_hidden else None,
        'weight_encoder': {'hidden_dims': weight_hidden} if weight_hidden else None,
        'fusion': {'hidden_dims': fusion_hidden},
        'output': {'num_patterns': num_patterns}
    }

    logger.info("Final model architecture:")
    logger.info(f"  Signature encoder: {sig_input} -> {sig_hidden}")
    logger.info(f"  Weight encoder: {weight_input} -> {weight_hidden}")
    logger.info(f"  Fusion: {fusion_input} -> {fusion_hidden} -> {num_patterns}")

    return architecture


def preprocess_signature(signature_json: str, max_dims: Dict[str, int], method_names: List[str]):
    signature = json.loads(signature_json)
    features_per_neuron = max_dims['signature_features_per_neuron']

    max_total_linear_layers = max_dims['max_total_linear_layers']
    padded_signature = np.zeros((
        max_total_linear_layers,
        max_dims['max_neurons_per_layer'],
        features_per_neuron
    ), dtype=np.float32)

    signature_mask = np.zeros((
        max_total_linear_layers,
        max_dims['max_neurons_per_layer']
    ), dtype=np.float32)

    neuron_activations = signature['neuron_activations']
    for layer_idx_str, layer_data in neuron_activations.items():
        layer_idx = int(layer_idx_str)
        if layer_idx >= max_total_linear_layers:
            logger.warning(f"Layer index {layer_idx} exceeds max_total_linear_layers {max_total_linear_layers}")
            continue

        neuron_profiles = layer_data['neuron_profiles']
        for neuron_idx_str, profile in neuron_profiles.items():
            neuron_idx = int(neuron_idx_str)
            if neuron_idx >= max_dims['max_neurons_per_layer']:
                logger.warning(f"Neuron index {neuron_idx} exceeds max_neurons {max_dims['max_neurons_per_layer']}")
                continue

            features = []
            for method_name in method_names:
                if method_name not in profile:
                    raise ValueError(f"Required method '{method_name}' not found in signature at layer {layer_idx}, neuron {neuron_idx}.")
                value = profile[method_name]
                if isinstance(value, list):
                    features.extend(value)
                else:
                    features.append(value)
            if len(features) != features_per_neuron:
                raise ValueError(f"Feature count mismatch at layer {layer_idx}, neuron {neuron_idx}: expected {features_per_neuron}, got {len(features)}.")

            padded_signature[layer_idx, neuron_idx, :] = features
            signature_mask[layer_idx, neuron_idx] = 1.0

    return padded_signature.flatten(), signature_mask.flatten()


def preprocess_weights(weights_json: str, max_total_params: int):
    weights_data = json.loads(weights_json)
    weights = weights_data['weights']

    # flatten all params
    all_params = []
    for name in sorted(weights.keys()):
        param_data = weights[name]
        if isinstance(param_data[0], list):  # 2D weight matrix
            for row in param_data:
                all_params.extend(row)
        else:  # 1D bias vector
            all_params.extend(param_data)

    padded_weights = np.zeros(max_total_params, dtype=np.float32)
    weight_mask = np.zeros(max_total_params, dtype=np.float32)
    actual_count = len(all_params)
    if actual_count > max_total_params:
        logger.warning(f"Actual params {actual_count} exceeds max {max_total_params}, truncating")
        actual_count = max_total_params

    padded_weights[:actual_count] = all_params[:actual_count]
    weight_mask[:actual_count] = 1.0

    return padded_weights, weight_mask


class PatternClassifierDataset(Dataset):
    def __init__(self, hf_dataset, max_dims: Dict[str, int], input_mode: str, all_patterns: List[str], method_names: List[str] = None):
        self.hf_dataset = hf_dataset
        self.max_dims = max_dims
        self.input_mode = input_mode
        self.all_patterns = all_patterns
        self.method_names = method_names or []

        # infer sig dimensions
        if self.input_mode in ["signature", "both"] and len(self.hf_dataset) > 0:
            first_example = self.hf_dataset[0]
            inferred_dims = infer_signature_dimensions(
                first_example['improved_signature'],
                self.method_names
            )
            self.max_dims['signature_features_per_neuron'] = inferred_dims['signature_features_per_neuron']
            self.method_shapes = inferred_dims['method_shapes']
        else:
            self.method_shapes = {}

        logger.info(f"Initialized PatternClassifierDataset: {len(self.hf_dataset)} examples, "
                   f"mode={input_mode}, {len(self.all_patterns)} patterns")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        example = self.hf_dataset[idx]
        metadata = example['metadata']
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        pattern_list = metadata['selected_patterns']
        label = self._encode_patterns(pattern_list)
        inputs = {}
        if self.input_mode in ["signature", "both"]:
            sig, sig_mask = preprocess_signature(
                example['improved_signature'], self.max_dims, self.method_names
            )
            inputs['signature'] = torch.from_numpy(sig)
            inputs['signature_mask'] = torch.from_numpy(sig_mask)

        if self.input_mode in ["weights", "both"]:
            weights, weight_mask = preprocess_weights(
                example['improved_model_weights'],
                self.max_dims['max_total_params']
            )
            inputs['weights'] = torch.from_numpy(weights)
            inputs['weights_mask'] = torch.from_numpy(weight_mask)

        return inputs, torch.from_numpy(label)

    def _encode_patterns(self, pattern_list: List[str]) -> np.ndarray:
        # turns patterns into multi-hot vector
        label = np.zeros(len(self.all_patterns), dtype=np.float32)
        for pattern in pattern_list:
            if pattern in self.all_patterns:
                idx = self.all_patterns.index(pattern)
                label[idx] = 1.0
            else:
                logger.warning(f"Unknown pattern: {pattern}")
        return label


def load_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset_config = config['dataset']
    logger.info(f"Loading dataset from HuggingFace: {dataset_config['hf_dataset']}")
    dataset = hf_load_dataset(dataset_config['hf_dataset'])
    dataset = dataset['train']
    logger.info(f"Dataset loaded: {len(dataset)} examples")

    max_dims = compute_dimensions_from_config(config)
    all_patterns = dataset_config['patterns']
    logger.info(f"Using {len(all_patterns)} patterns from config: {all_patterns}")

    neuron_profile = dataset_config.get('neuron_profile', {})
    method_names = neuron_profile.get('methods', [])
    if not isinstance(method_names, list):
        raise ValueError("Neuron_profile.methods must be a list of method names")
    if not method_names:
        raise ValueError("neuron_profile.methods is empty. Must specify at least one method to extract.")

    logger.info(f"Will extract signature methods in order: {method_names}")

    input_mode = dataset_config['input_mode']
    input_dims = {}

    if input_mode in ["signature", "both"]:
        input_dims['signature'] = None  # will be filled after inference
    if input_mode in ["weights", "both"]:
        input_dims['weights'] = max_dims['max_total_params']

    return {
        'dataset': dataset,
        'max_dims': max_dims,
        'input_dims': input_dims,
        'all_patterns': all_patterns,
        'method_names': method_names
    }


def create_dataloaders(dataset_info: Dict[str, Any], config: Dict[str, Any]):
    dataset = dataset_info['dataset']
    max_dims = dataset_info['max_dims']
    all_patterns = dataset_info['all_patterns']
    method_names = dataset_info.get('method_names', [])

    dataset_config = config['dataset']
    dataloader_config = config['dataloader']
    train_size = int(len(dataset) * dataset_config['train_split'])
    val_size = int(len(dataset) * dataset_config['val_split'])
    test_size = len(dataset) - train_size - val_size
    logger.info(f"Splitting dataset: train={train_size}, val={val_size}, test={test_size}")
    generator = torch.Generator().manual_seed(dataset_config['random_seed'])
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    input_mode = dataset_config['input_mode']

    train_dataset = PatternClassifierDataset(train_dataset, max_dims, input_mode, all_patterns, method_names)
    val_dataset = PatternClassifierDataset(val_dataset, max_dims, input_mode, all_patterns, method_names)
    test_dataset = PatternClassifierDataset(test_dataset, max_dims, input_mode, all_patterns, method_names)

    if input_mode in ["signature", "both"]:
        signature_dim = (
            max_dims['max_total_linear_layers'] *
            max_dims['max_neurons_per_layer'] *
            max_dims['signature_features_per_neuron']
        )
        dataset_info['input_dims']['signature'] = signature_dim
        logger.info(f"Inferred signature input dimension: {signature_dim}")

    # dataloaders
    batch_size = config['training']['batch_size']
    num_workers = dataloader_config.get('num_workers', 0)
    pin_memory = dataloader_config.get('pin_memory', False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info(f"Created dataloaders: batch_size={batch_size}, "
               f"num_workers={num_workers}, pin_memory={pin_memory}")

    return train_loader, val_loader, test_loader
