import json
import numpy as np
import torch
from typing import Dict, List, Any, Union


def infer_neurons_from_weights(weights_dict: Dict[str, Any]):
    neurons_per_layer = []
    sorted_keys = sorted(weights_dict.keys())
    for key in sorted_keys:
        if "weight" in key.lower() and "bias" not in key.lower():
            weight = weights_dict[key]
            if isinstance(weight, (list, np.ndarray)):
                weight = np.array(weight)
            elif isinstance(weight, torch.Tensor):
                weight = weight.numpy()
            if hasattr(weight, "shape") and len(weight.shape) >= 2:
                neurons_per_layer.append(weight.shape[0])
    return neurons_per_layer


def extract_neuron_weights_list(weights_dict: Dict[str, Any]) -> List[np.ndarray]:
    sorted_keys = sorted(weights_dict.keys())

    # group weights and biases by layer
    layer_groups = {}
    for key in sorted_keys:
        parts = key.split(".")
        if len(parts) >= 2:
            layer_name = ".".join(parts[:-1])
            param_type = parts[-1]
        else:
            layer_name = key
            param_type = "weight"

        if layer_name not in layer_groups:
            layer_groups[layer_name] = {}
        tensor = weights_dict[key]
        if isinstance(tensor, (list, np.ndarray)):
            tensor = np.array(tensor, dtype=np.float32)
        layer_groups[layer_name][param_type] = tensor

    # extract neurons from each layer
    neurons = []
    max_neuron_size = 0
    for layer_name in sorted(layer_groups.keys()):
        layer_params = layer_groups[layer_name]
        weight = layer_params.get("weight", layer_params.get("weights", None))
        bias = layer_params.get("bias", None)
        if weight is None:
            continue
        weight = np.atleast_2d(weight)
        if bias is not None:
            bias = np.atleast_1d(bias)
        num_neurons_in_layer = weight.shape[0]
        for neuron_idx in range(num_neurons_in_layer):
            neuron_weights = weight[neuron_idx].flatten()
            if bias is not None and neuron_idx < len(bias):
                neuron_data = np.concatenate([neuron_weights, [bias[neuron_idx]]])
            else:
                neuron_data = neuron_weights
            neurons.append(neuron_data)
            max_neuron_size = max(max_neuron_size, len(neuron_data))

    # pad all neurons to same size
    padded_neurons = []
    for neuron_data in neurons:
        if len(neuron_data) < max_neuron_size:
            padded = np.pad(
                neuron_data,
                (0, max_neuron_size - len(neuron_data)),
                mode="constant",
            )
            padded_neurons.append(padded)
        else:
            padded_neurons.append(neuron_data)

    return padded_neurons


def extract_signature_features(
    signature_json: Union[str, Dict],
    method_names: List[str],
    norm_stats: Dict[str, Dict[str, float]] = None,
) -> List[np.ndarray]:
    """
    Extract signature features from a signature JSON.
    
    Args:
        signature_json: JSON string or dict containing neuron activations
        method_names: List of method names to extract (e.g., ["mean", "std", "fourier"])
        norm_stats: Optional normalization statistics to apply per-feature normalization
        
    Returns:
        List of numpy arrays, one per neuron, containing the extracted features
    """
    if isinstance(signature_json, str):
        signature_data = json.loads(signature_json)
    else:
        signature_data = signature_json

    neuron_activations = signature_data.get("neuron_activations", {})
    signature_neurons = []
    
    # Build feature order from norm_stats if available
    feature_order = norm_stats.get("_feature_order", []) if norm_stats else []

    for layer_idx_str in sorted(neuron_activations.keys(), key=int):
        layer_data = neuron_activations[layer_idx_str]
        neuron_profiles = layer_data.get("neuron_profiles", {})
        for neuron_idx_str in sorted(neuron_profiles.keys(), key=int):
            profile = neuron_profiles[neuron_idx_str]
            neuron_features = []
            feature_idx = 0
            for method_name in method_names:
                if method_name in profile:
                    value = profile[method_name]
                    if isinstance(value, list):
                        # Apply per-feature normalization for array features
                        for i, v in enumerate(value):
                            if norm_stats and feature_order:
                                feature_key = f"{method_name}_{i}"
                                if feature_key in norm_stats:
                                    mean = norm_stats[feature_key]["mean"]
                                    std = norm_stats[feature_key]["std"]
                                    v = (v - mean) / std
                            neuron_features.append(v)
                            feature_idx += 1
                    else:
                        # Apply per-feature normalization for scalar features
                        if norm_stats and method_name in norm_stats:
                            mean = norm_stats[method_name]["mean"]
                            std = norm_stats[method_name]["std"]
                            value = (value - mean) / std
                        neuron_features.append(value)
                        feature_idx += 1
            signature_neurons.append(np.array(neuron_features, dtype=np.float32))

    return signature_neurons


def flatten_signature_features(
    signature_json: Union[str, Dict],
    method_names: List[str],
    norm_stats: Dict[str, Dict[str, float]] = None,
) -> np.ndarray:
    """
    Extract and flatten signature features into a single array.
    
    Args:
        signature_json: JSON string or dict containing neuron activations
        method_names: List of method names to extract
        norm_stats: Optional normalization statistics to apply per-feature normalization
        
    Returns:
        Flattened numpy array of all signature features
    """
    signature_neurons = extract_signature_features(signature_json, method_names, norm_stats)
    if not signature_neurons:
        return np.array([], dtype=np.float32)
    return np.concatenate(signature_neurons)


def interleave_weights_signatures(
    weights_dict: Dict[str, Any],
    signature_json: Union[str, Dict],
    method_names: List[str],
    norm_stats: Dict[str, Dict[str, float]] = None,
) -> List[np.ndarray]:
    """
    Interleave weight data and signature features for each neuron.
    
    Args:
        weights_dict: Dictionary mapping parameter names to weight tensors
        signature_json: JSON string or dict containing neuron activations
        method_names: List of method names to extract (e.g., ["mean", "std", "fourier"])
        norm_stats: Optional normalization statistics to apply per-feature normalization
        
    Returns:
        List of numpy arrays, one per neuron, containing concatenated weights + features
    """
    weight_neurons = extract_neuron_weights_list(weights_dict)
    signature_neurons = extract_signature_features(signature_json, method_names, norm_stats)

    if len(weight_neurons) != len(signature_neurons):
        raise ValueError(
            f"Mismatch between weight neurons ({len(weight_neurons)}) and signature neurons ({len(signature_neurons)})"
        )

    combined_neurons = []
    for i in range(len(weight_neurons)):
        combined = np.concatenate([weight_neurons[i], signature_neurons[i]])
        combined_neurons.append(combined)

    return combined_neurons


def extract_architecture_spec(weights_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract architecture specification from a weights dictionary.

    This spec is used by the decoder for architecture-conditioned generation,
    bypassing the latent space to prevent steering vector corruption.

    Args:
        weights_dict: Dictionary mapping parameter names to weight tensors

    Returns:
        arch_spec: Dictionary containing:
            - num_layers: Number of weight layers (excluding biases)
            - neurons_per_layer: List of neuron counts per layer
            - input_dim: Input dimension of first layer
            - output_dim: Output dimension of last layer
            - layer_shapes: List of (name, shape) tuples for all parameters
    """
    # Handle nested weights dict
    if "weights" in weights_dict:
        weights_dict = weights_dict["weights"]

    sorted_keys = sorted(weights_dict.keys())
    layer_shapes = []
    weight_layers = []

    for name in sorted_keys:
        tensor = weights_dict[name]
        if isinstance(tensor, (list, np.ndarray)):
            tensor = np.array(tensor)
            shape = tuple(tensor.shape)
        elif isinstance(tensor, torch.Tensor):
            shape = tuple(tensor.shape)
        else:
            shape = ()

        layer_shapes.append((name, shape))

        # Collect weight matrices (not biases) for architecture info
        if "weight" in name.lower() and "bias" not in name.lower() and len(shape) >= 2:
            weight_layers.append({
                "name": name,
                "neurons_out": shape[0],
                "neurons_in": shape[1] if len(shape) > 1 else 1
            })

    neurons_per_layer = [layer["neurons_out"] for layer in weight_layers]

    return {
        "num_layers": len(weight_layers),
        "neurons_per_layer": neurons_per_layer,
        "input_dim": weight_layers[0]["neurons_in"] if weight_layers else 0,
        "output_dim": weight_layers[-1]["neurons_out"] if weight_layers else 0,
        "layer_shapes": layer_shapes,
    }


def compute_layer_weights(
    neurons_per_layer: List[int],
    num_tokens: int,
    weight_decay: float = 0.3,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute per-token loss weights inversely proportional to input_correlations predictive power.

    Later layers get higher weights because input_correlations is less predictive for them.
    This compensates for the information imbalance where layer 0 has ~0.96 correlation
    between input_correlations and actual weights, while later layers have ~0.55-0.65.

    Formula: weight = 1 + layer_idx * weight_decay

    For a 6-layer network with weight_decay=0.3:
    - Layer 0: weight = 1.0 (easiest to predict)
    - Layer 1: weight = 1.3
    - Layer 2: weight = 1.6
    - Layer 3: weight = 1.9
    - Layer 4: weight = 2.2
    - Layer 5: weight = 2.5 (hardest to predict)

    Args:
        neurons_per_layer: List of neuron counts per layer
        num_tokens: Total number of tokens (for padding)
        weight_decay: How much weight increases per layer (default 0.3)
        device: Target device for the tensor

    Returns:
        Tensor of shape (num_tokens,) with per-token weights
    """
    weights = []

    for layer_idx, num_neurons in enumerate(neurons_per_layer):
        layer_weight = 1.0 + layer_idx * weight_decay
        weights.extend([layer_weight] * num_neurons)

    # Pad if needed (for padding tokens, use weight 0 so they don't contribute)
    while len(weights) < num_tokens:
        weights.append(0.0)

    # Truncate if needed
    weights = weights[:num_tokens]

    result = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        result = result.to(device)

    return result
