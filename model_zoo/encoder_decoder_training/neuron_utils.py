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
