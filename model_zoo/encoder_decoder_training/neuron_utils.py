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
) -> List[np.ndarray]:
    if isinstance(signature_json, str):
        signature_data = json.loads(signature_json)
    else:
        signature_data = signature_json

    neuron_activations = signature_data.get("neuron_activations", {})
    signature_neurons = []

    for layer_idx_str in sorted(neuron_activations.keys(), key=int):
        layer_data = neuron_activations[layer_idx_str]
        neuron_profiles = layer_data.get("neuron_profiles", {})
        for neuron_idx_str in sorted(neuron_profiles.keys(), key=int):
            profile = neuron_profiles[neuron_idx_str]
            neuron_features = []
            for method_name in method_names:
                if method_name in profile:
                    value = profile[method_name]
                    if isinstance(value, list):
                        neuron_features.extend(value)
                    else:
                        neuron_features.append(value)
            signature_neurons.append(np.array(neuron_features, dtype=np.float32))

    return signature_neurons


def flatten_signature_features(
    signature_json: Union[str, Dict],
    method_names: List[str],
) -> np.ndarray:
    signature_neurons = extract_signature_features(signature_json, method_names)
    if not signature_neurons:
        return np.array([], dtype=np.float32)
    return np.concatenate(signature_neurons)


def interleave_weights_signatures(
    weights_dict: Dict[str, Any],
    signature_json: Union[str, Dict],
    method_names: List[str],
) -> List[np.ndarray]:
    weight_neurons = extract_neuron_weights_list(weights_dict)
    signature_neurons = extract_signature_features(signature_json, method_names)

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
