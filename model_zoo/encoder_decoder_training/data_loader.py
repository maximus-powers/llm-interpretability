import torch
import numpy as np
import json
import logging
from typing import Dict, Any, List
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate

from .tokenizer import WeightTokenizer

logger = logging.getLogger(__name__)


def compute_dimensions_from_config(config: Dict[str, Any]):
    dataset_config = config["dataset"]
    max_dims = dataset_config.get("max_dimensions", {})
    if not max_dims:
        return {}
    max_hidden_layers = max_dims.get("max_hidden_layers", 0)
    max_neurons = max_dims.get("max_neurons_per_layer", 0)
    max_total_linear_layers = max_hidden_layers + 2  # input and output layers
    result = {
        "max_hidden_layers": max_hidden_layers,
        "max_total_linear_layers": max_total_linear_layers,
        "max_neurons_per_layer": max_neurons,
        "signature_features_per_neuron": 0,
    }
    logger.info(f"Dimensions computed from config: {result}")
    return result


def infer_signature_dimensions(signature_json: str, method_names: List[str]):
    signature = json.loads(signature_json)
    neuron_activations = signature["neuron_activations"]
    first_layer_idx = min(int(k) for k in neuron_activations.keys())
    first_layer = neuron_activations[str(first_layer_idx)]
    first_neuron_idx = min(int(k) for k in first_layer["neuron_profiles"].keys())
    sample_profile = first_layer["neuron_profiles"][str(first_neuron_idx)]
    layer_info = first_layer.get("layer_info", {})
    available_methods_claimed = layer_info.get("profile_methods", None)
    actual_methods = list(sample_profile.keys())
    if available_methods_claimed and set(available_methods_claimed) != set(
        actual_methods
    ):
        logger.warning(
            f"layer_info.profile_methods {available_methods_claimed} doesn't match actual profile keys"
        )

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
        "signature_features_per_neuron": total_features,
        "method_shapes": method_shapes,
        "profile_methods_in_data": actual_methods,
    }


def preprocess_signature(
    signature_json: str, max_dims: Dict[str, int], method_names: List[str]
):
    signature = json.loads(signature_json)
    features_per_neuron = max_dims["signature_features_per_neuron"]

    max_total_linear_layers = max_dims["max_total_linear_layers"]
    padded_signature = np.zeros(
        (
            max_total_linear_layers,
            max_dims["max_neurons_per_layer"],
            features_per_neuron,
        ),
        dtype=np.float32,
    )
    signature_mask = np.zeros(
        (max_total_linear_layers, max_dims["max_neurons_per_layer"]), dtype=np.float32
    )

    neuron_activations = signature.get("neuron_activations", {})
    for layer_idx_str, layer_data in neuron_activations.items():
        layer_idx = int(layer_idx_str)
        if layer_idx >= max_total_linear_layers:
            continue
        neuron_profiles = layer_data.get("neuron_profiles", {})
        for neuron_idx_str, profile in neuron_profiles.items():
            neuron_idx = int(neuron_idx_str)
            if neuron_idx >= max_dims["max_neurons_per_layer"]:
                continue
            # extract features in config order
            features = []
            for method_name in method_names:
                if method_name not in profile:
                    raise ValueError(
                        f"Required method '{method_name}' not found in signature at layer {layer_idx}, neuron {neuron_idx}."
                    )
                value = profile[method_name]
                if isinstance(value, list):
                    features.extend(value)
                else:
                    features.append(value)
            # validate feature count
            if len(features) != features_per_neuron:
                raise ValueError(
                    f"Feature count mismatch at layer {layer_idx}, neuron {neuron_idx}: expected {features_per_neuron}, got {len(features)}."
                )
            padded_signature[layer_idx, neuron_idx, :] = features
            signature_mask[layer_idx, neuron_idx] = 1.0

    flat_signature = padded_signature.flatten()
    flat_mask = signature_mask.flatten()

    return flat_signature, flat_mask


def augment_tokenized_weights(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    augmentation_type: str = "noise",
    noise_std: float = 0.01,
    dropout_prob: float = 0.1,
):
    if augmentation_type == "none":
        return tokens.clone(), mask.clone()
    elif augmentation_type == "noise":
        noise = torch.randn_like(tokens) * noise_std
        augmented_tokens = tokens + noise
        return augmented_tokens, mask.clone()
    elif augmentation_type == "dropout":
        augmented_tokens = tokens.clone()
        augmented_mask = mask.clone()
        dropout_mask = (
            torch.rand(tokens.size(0), tokens.size(1), device=tokens.device)
            > dropout_prob
        )
        augmented_tokens = augmented_tokens * dropout_mask.unsqueeze(-1).float()
        augmented_mask = augmented_mask * dropout_mask.float()
        return augmented_tokens, augmented_mask
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}.")


class WeightSpaceDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer: WeightTokenizer,
        input_mode: str,
        config: Dict[str, Any],
        method_names: List[str] = None,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.input_mode = input_mode
        self.config = config
        self.method_names = method_names or []

        self.max_dims = None
        if input_mode in ["signature", "both"]:
            self.max_dims = compute_dimensions_from_config(config)
            # infer signature dimensions from first example
            if len(self.hf_dataset) > 0:
                first_example = self.hf_dataset[0]
                inferred_dims = infer_signature_dimensions(
                    first_example["improved_signature"], self.method_names
                )
                self.max_dims["signature_features_per_neuron"] = inferred_dims[
                    "signature_features_per_neuron"
                ]
                self.method_shapes = inferred_dims["method_shapes"]
                logger.info(
                    f"Inferred signature dimensions: {inferred_dims['signature_features_per_neuron']} features per neuron"
                )
            else:
                self.method_shapes = {}

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        metadata_dict = None
        if self.tokenizer.granularity == "neuron":
            # extract architecture info
            metadata_json = json.loads(example["metadata"])
            architecture = metadata_json.get("architecture", {})
            neurons_per_layer = architecture.get("neurons_per_layer", [])
            # or infer from weights
            if not neurons_per_layer:
                weights_json = json.loads(example["improved_model_weights"])
                weights_dict = weights_json["weights"] if "weights" in weights_json else weights_json
                neurons_per_layer = []
                sorted_keys = sorted(weights_dict.keys())
                for key in sorted_keys:
                    if "weight" in key and "bias" not in key:
                        weight = weights_dict[key]
                        if isinstance(weight, (list, np.ndarray)):
                            weight = np.array(weight)
                        shape = weight.shape if hasattr(weight, 'shape') else np.array(weight).shape

                        if len(shape) >= 2:
                            # For linear layers, neurons are in dimension 0
                            neurons_per_layer.append(shape[0])
            metadata_dict = {
                "neurons_per_layer": neurons_per_layer,
                "features_per_neuron":  len(self.method_names),
            }

        # tokenize weights for decoder target
        weights_json = json.loads(example["improved_model_weights"])
        weights_dict = weights_json["weights"] if "weights" in weights_json else weights_json
        weights_tokenized = self.tokenizer.tokenize(weights_dict, metadata_dict)

        # tokenize input (signature, weights, or both)
        if self.input_mode == "signature":
            if self.tokenizer.granularity == "neuron":
                # flatten signature and tokenize
                signature = json.loads(example["improved_signature"])
                neuron_activations = signature.get("neuron_activations", {})
                features_list = []
                for layer_idx_str in sorted(neuron_activations.keys(), key=int):
                    layer_data = neuron_activations[layer_idx_str]
                    neuron_profiles = layer_data.get("neuron_profiles", {})
                    for neuron_idx_str in sorted(neuron_profiles.keys(), key=int):
                        profile = neuron_profiles[neuron_idx_str]
                        neuron_features = []
                        # extract configed features
                        for method_name in self.method_names:
                            if method_name in profile:
                                value = profile[method_name]
                                if isinstance(value, list):
                                    neuron_features.extend(value)
                                else:
                                    neuron_features.append(value)
                        features_list.extend(neuron_features)
                signature_flat =  np.array(features_list, dtype=np.float32)
                encoder_tokenized = self.tokenizer.tokenize(signature_flat, metadata_dict)
            else:
                signature_features, signature_mask = preprocess_signature(
                    example["improved_signature"], self.max_dims, self.method_names
                )
                encoder_tokenized = {
                    "tokens": torch.from_numpy(signature_features.reshape(1, -1)),
                    "attention_mask": torch.from_numpy(signature_mask.reshape(1, -1)),
                    "num_real_tokens": 1,
                }

        elif self.input_mode == "both":
            if self.tokenizer.granularity == "neuron":
                combined = self._interleave_weights_signatures(
                    weights_dict,
                    example["improved_signature"],
                )
                encoder_tokenized = self.tokenizer.tokenize(combined, metadata_dict)
            else:
                # flattened weights
                weights_json = json.loads(example["improved_model_weights"])
                weights_dict_raw = weights_json["weights"] if "weights" in weights_json else weights_json
                weights_flat_list = []
                for key in sorted(weights_dict_raw.keys()):
                    weight_tensor = weights_dict_raw[key]
                    if isinstance(weight_tensor, (list, np.ndarray)):
                        weights_flat_list.extend(np.array(weight_tensor).flatten().tolist())
                # flattened signature
                signature_features, signature_mask = preprocess_signature(
                    example["improved_signature"], self.max_dims, self.method_names
                )
                signature_flat = signature_features.flatten()
                # concat
                combined_array = np.array(weights_flat_list + signature_flat.tolist(), dtype=np.float32)
                # chunk
                chunk_size = self.tokenizer.chunk_size
                num_chunks = int(np.ceil(len(combined_array) / chunk_size))
                if num_chunks > self.tokenizer.max_tokens:
                    num_chunks = self.tokenizer.max_tokens
                    combined_array = combined_array[:self.tokenizer.max_tokens * chunk_size]

                # chunked tokens
                tokens = np.zeros((self.tokenizer.max_tokens, self.tokenizer.token_dim), dtype=np.float32)
                attention_mask = np.zeros(self.tokenizer.max_tokens, dtype=np.float32)
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(combined_array))
                    chunk_data = combined_array[start_idx:end_idx]
                    # padding
                    if len(chunk_data) < chunk_size:
                        chunk_data = np.pad(chunk_data, (0, chunk_size - len(chunk_data)), mode='constant')
                    tokens[chunk_idx, :chunk_size] = chunk_data
                    # metadata
                    if self.tokenizer.include_metadata:
                        tokens[chunk_idx, chunk_size] = 0.0  # layer_idx (not meaningful here)
                        tokens[chunk_idx, chunk_size + 1] = 0.0  # param_type
                        tokens[chunk_idx, chunk_size + 2] = chunk_idx / max(num_chunks - 1, 1)  # position
                        tokens[chunk_idx, chunk_size + 3] = np.log1p(len(combined_array))  # shape_log
                        tokens[chunk_idx, chunk_size + 4] = chunk_idx / max(num_chunks - 1, 1)  # chunk_idx

                    attention_mask[chunk_idx] = 1.0

                encoder_tokenized = {
                    "tokens": torch.from_numpy(tokens),
                    "attention_mask": torch.from_numpy(attention_mask),
                    "num_real_tokens": num_chunks,
                }

        elif self.input_mode == "weights":
            encoder_tokenized = weights_tokenized
        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode}")

        # output with separate encoder/decoder data
        output = {
            "encoder_input": encoder_tokenized["tokens"],
            "encoder_mask": encoder_tokenized["attention_mask"],
            "decoder_target": weights_tokenized["tokens"],
            "decoder_mask": weights_tokenized["attention_mask"],
            "num_real_tokens": weights_tokenized["num_real_tokens"],
        }

        # original_shapes if available (for detokenization)
        if "original_shapes" in weights_tokenized:
            output["original_shapes"] = weights_tokenized["original_shapes"]

        return output

    def _interleave_weights_signatures(
        self,
        weights_dict: Dict[str, Any],
        signature_json: str,
    ) -> List[np.ndarray]:
        signature_data = json.loads(signature_json)
        neuron_activations = signature_data.get("neuron_activations", {})

        # extract neuron weights and signatures
        weight_neurons = self._extract_neuron_weights_list(weights_dict)
        signature_neurons = []
        for layer_idx_str in sorted(neuron_activations.keys(), key=int):
            layer_data = neuron_activations[layer_idx_str]
            neuron_profiles = layer_data.get("neuron_profiles", {})
            for neuron_idx_str in sorted(neuron_profiles.keys(), key=int):
                profile = neuron_profiles[neuron_idx_str]
                neuron_features = []
                for method_name in self.method_names:
                    if method_name in profile:
                        value = profile[method_name]
                        if isinstance(value, list):
                            neuron_features.extend(value)
                        else:
                            neuron_features.append(value)
                signature_neurons.append(np.array(neuron_features, dtype=np.float32))

        # [neuron0: [weights + signature], neuron1: [weights + signature], ...]
        combined_neurons = []
        if len(weight_neurons) != len(signature_neurons):
            raise ValueError(
            f"Mismatch between weight neurons ({len(weight_neurons)}) and signature neurons ({len(signature_neurons)})"
            )

        # concat within neuron
        for i in range(len(weight_neurons)):
            combined = np.concatenate([weight_neurons[i], signature_neurons[i]])
            combined_neurons.append(combined)

        return combined_neurons

    def _extract_neuron_weights_list(
        self,
        weights_dict: Dict[str, Any],
    ) -> List[np.ndarray]:
        sorted_keys = sorted(weights_dict.keys())

        # group weights and biases by layer
        layer_groups = {}
        for key in sorted_keys:
            parts = key.split('.')
            if len(parts) >= 2:
                layer_name = '.'.join(parts[:-1])
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
            weight = layer_params.get('weight', layer_params.get('weights', None))
            bias = layer_params.get('bias', None)
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
                    mode='constant'
                )
                padded_neurons.append(padded)
            else:
                padded_neurons.append(neuron_data)

        return padded_neurons


def load_dataset(config: Dict[str, Any]):
    dataset_config = config["dataset"]
    tokenization_config = config["tokenization"]
    dataset = hf_load_dataset(dataset_config["hf_dataset"])
    dataset = dataset["train"]
    logger.info(f"Dataset loaded: {len(dataset)} examples")

    tokenizer = WeightTokenizer(
        chunk_size=tokenization_config["chunk_size"],
        max_tokens=tokenization_config["max_tokens"],
        include_metadata=tokenization_config.get("include_metadata", True),
        granularity=tokenization_config.get("granularity", "chunk"),
    )
    logger.info(
        "Tokenizer created"
    )

    input_dims = {"token_dim": tokenizer.token_dim, "max_tokens": tokenizer.max_tokens}
    neuron_profile = dataset_config.get("neuron_profile", {})
    method_names = neuron_profile.get("methods", [])
    if dataset_config["input_mode"] in ["signature", "both"]:
        if not isinstance(method_names, list) or not method_names:
            raise ValueError("neuron_profile.methods must be a list of method names")
        input_dims["signature_dim"] = None  # inferred during dataset init

    return {
        "dataset": dataset,
        "tokenizer": tokenizer,
        "input_dims": input_dims,
        "method_names": method_names,
    }


def custom_collate_fn(batch):
    original_shapes_list = None
    if "original_shapes" in batch[0]:
        original_shapes_list = [item.pop("original_shapes") for item in batch]

    # max token_dim for encoder_input within batch
    max_encoder_dim = max(item["encoder_input"].shape[1] for item in batch)

    # for decoder_target, use the actual max from the batch, used to ensure consistency within the batch
    max_decoder_dim = max(item["decoder_target"].shape[1] for item in batch)

    # pad each item to max dims
    for item in batch:
        encoder_input = item["encoder_input"]
        if encoder_input.shape[1] < max_encoder_dim:
            padding = torch.zeros(encoder_input.shape[0], max_encoder_dim - encoder_input.shape[1])
            item["encoder_input"] = torch.cat([encoder_input, padding], dim=1)

        decoder_target = item["decoder_target"]
        if decoder_target.shape[1] < max_decoder_dim:
            padding = torch.zeros(decoder_target.shape[0], max_decoder_dim - decoder_target.shape[1])
            item["decoder_target"] = torch.cat([decoder_target, padding], dim=1)

    collated_batch = default_collate(batch)

    if original_shapes_list is not None:
        collated_batch["original_shapes"] = original_shapes_list

    return collated_batch


def create_dataloaders(dataset_info: Dict[str, Any], config: Dict[str, Any]):
    dataset_config = config["dataset"]
    train_size = int(len(dataset_info["dataset"]) * dataset_config["train_split"])
    val_size = int(len(dataset_info["dataset"]) * dataset_config["val_split"])
    test_size = len(dataset_info["dataset"]) - train_size - val_size
    logger.info(
        f"Splitting dataset: train={train_size} ({dataset_config['train_split']}), val={val_size} ({dataset_config['val_split']}), test={test_size} ({dataset_config['test_split']})"
    )

    generator = torch.Generator().manual_seed(dataset_config["random_seed"])
    train_dataset, val_dataset, test_dataset = random_split(
        dataset_info["dataset"], [train_size, val_size, test_size], generator=generator
    )
    train_dataset = WeightSpaceDataset(
        train_dataset,
        dataset_info["tokenizer"],
        dataset_config["input_mode"],
        config,
        dataset_info.get("method_names", []),
    )
    val_dataset = WeightSpaceDataset(
        val_dataset,
        dataset_info["tokenizer"],
        dataset_config["input_mode"],
        config,
        dataset_info.get("method_names", []),
    )
    test_dataset = WeightSpaceDataset(
        test_dataset,
        dataset_info["tokenizer"],
        dataset_config["input_mode"],
        config,
        dataset_info.get("method_names", []),
    )

    # update input_dims
    if dataset_config["input_mode"] in ["signature", "both"]:
        max_dims = train_dataset.max_dims
        if max_dims:
            signature_dim = (
                max_dims["max_total_linear_layers"]
                * max_dims["max_neurons_per_layer"]
                * max_dims["signature_features_per_neuron"]
            )
            dataset_info["input_dims"]["signature_dim"] = signature_dim
            logger.info(f"Inferred signature dimension: {signature_dim}")

    dataloader_config = config.get("dataloader", {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=dataloader_config.get("num_workers", 0),
        pin_memory=dataloader_config.get("pin_memory", False),
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=dataloader_config.get("num_workers", 0),
        pin_memory=dataloader_config.get("pin_memory", False),
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=dataloader_config.get("num_workers", 0),
        pin_memory=dataloader_config.get("pin_memory", False),
        collate_fn=custom_collate_fn,
    )

    logger.info(
        f"Created dataloaders: batch_size={config['training']['batch_size']}, train_batches={len(train_loader)}, val_batches={len(val_loader)}, test_batches={len(test_loader)}"
    )

    return train_loader, val_loader, test_loader
