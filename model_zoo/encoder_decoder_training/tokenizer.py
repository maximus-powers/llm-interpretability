import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class WeightTokenizer:
    def __init__(
        self,
        chunk_size: int = 64,
        max_tokens: int = 512,
        include_metadata: bool = True,
        granularity: str = "chunk",
    ):
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens
        self.include_metadata = include_metadata
        self.granularity = granularity
        self.metadata_features = 5 if include_metadata else 0

        # chunk mode token_dim determined here, neuron mode it's set during tokenization
        if granularity == "chunk":
            self.token_dim = chunk_size + self.metadata_features
        else:
            self.token_dim = None

        logger.info(
            f"WeightTokenizer initialized: chunk_size={chunk_size}, max_tokens={max_tokens}, include_metadata={include_metadata}, granularity={granularity}, token_dim={self.token_dim}"
        )

    def tokenize(self, input_data, metadata_dict=None) -> Dict[str, Any]:
        if self.granularity == "neuron":
            return self._tokenize_by_neuron(input_data, metadata_dict)
        else:
            return self._tokenize_by_chunk(input_data)

    def _tokenize_by_chunk(self, weights_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "weights" in weights_dict:
            state_dict = weights_dict["weights"]
        else:
            state_dict = weights_dict
        state_dict_tensors = {}
        for key, value in state_dict.items():
            if isinstance(value, (list, np.ndarray)):
                state_dict_tensors[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, torch.Tensor):
                state_dict_tensors[key] = value.float()
            else:
                raise ValueError(f"Unexpected type for weight {key}: {type(value)}")

        sorted_keys = sorted(state_dict_tensors.keys())
        original_shapes = [
            (key, tuple(state_dict_tensors[key].shape)) for key in sorted_keys
        ]

        all_weights = []
        weight_metadata = []

        for layer_idx, key in enumerate(sorted_keys):
            tensor = state_dict_tensors[key]
            flat_weights = tensor.flatten().numpy()

            param_type = 1 if "bias" in key else 0
            norm_layer_idx = layer_idx / max(len(sorted_keys) - 1, 1)
            shape_log = np.log1p(np.prod(tensor.shape))

            all_weights.extend(flat_weights)

            for pos_in_layer, _ in enumerate(flat_weights):
                norm_position = pos_in_layer / max(len(flat_weights) - 1, 1)
                weight_metadata.append(
                    {
                        "layer_idx": norm_layer_idx,
                        "param_type": param_type,
                        "position": norm_position,
                        "shape_log": shape_log,
                    }
                )

        all_weights = np.array(all_weights, dtype=np.float32)

        num_chunks = int(np.ceil(len(all_weights) / self.chunk_size))

        if num_chunks > self.max_tokens:
            logger.warning(
                f"Number of chunks ({num_chunks}) exceeds max_tokens ({self.max_tokens}). "
                f"Truncating. This may lose information."
            )
            num_chunks = self.max_tokens
            all_weights = all_weights[: self.max_tokens * self.chunk_size]
            weight_metadata = weight_metadata[: self.max_tokens * self.chunk_size]

        tokens = np.zeros((self.max_tokens, self.token_dim), dtype=np.float32)
        attention_mask = np.zeros(self.max_tokens, dtype=np.float32)

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(all_weights))
            chunk_size_actual = end_idx - start_idx

            chunk_values = all_weights[start_idx:end_idx]

            if chunk_size_actual < self.chunk_size:
                chunk_values = np.pad(
                    chunk_values,
                    (0, self.chunk_size - chunk_size_actual),
                    mode="constant",
                )

            tokens[chunk_idx, : self.chunk_size] = chunk_values

            if self.include_metadata:
                mid_idx = start_idx + chunk_size_actual // 2
                if mid_idx < len(weight_metadata):
                    meta = weight_metadata[mid_idx]
                    tokens[chunk_idx, self.chunk_size] = meta["layer_idx"]
                    tokens[chunk_idx, self.chunk_size + 1] = meta["param_type"]
                    tokens[chunk_idx, self.chunk_size + 2] = meta["position"]
                    tokens[chunk_idx, self.chunk_size + 3] = meta["shape_log"]
                    tokens[chunk_idx, self.chunk_size + 4] = chunk_idx / max(
                        num_chunks - 1, 1
                    )
            attention_mask[chunk_idx] = 1.0

        return {
            "tokens": torch.from_numpy(tokens),
            "attention_mask": torch.from_numpy(attention_mask),
            "original_shapes": original_shapes,
            "num_real_tokens": num_chunks,
        }

    def _tokenize_by_neuron(self, input_data, metadata_dict=None) -> Dict[str, Any]:
        if metadata_dict is None:
            raise ValueError("metadata_dict is required for neuron-level tokenization")

        neurons = self._extract_neurons(input_data, metadata_dict)

        # find token_dim for this batch, token_dim = neuron data size + metadata
        neuron_data_size = len(neurons[0])
        batch_token_dim = neuron_data_size + self.metadata_features
        if self.token_dim is None:
            self.token_dim = batch_token_dim

        if len(neurons) > self.max_tokens:
            raise ValueError(
                f"Number of neurons ({len(neurons)}) exceeds max_tokens ({self.max_tokens}). "
            )

        # init token array and attention mask
        tokens = np.zeros((self.max_tokens, batch_token_dim), dtype=np.float32)
        attention_mask = np.zeros(self.max_tokens, dtype=np.float32)

        # tokens for each neuron
        for neuron_idx, neuron_data in enumerate(neurons):
            tokens[neuron_idx, :neuron_data_size] = neuron_data

            # add metadata
            if self.include_metadata:
                # find layer idx
                neurons_per_layer = metadata_dict.get("neurons_per_layer", [])
                layer_idx = max(0, len(neurons_per_layer) - 1)
                cumulative = 0
                for layer_idx, num_neurons in enumerate(neurons_per_layer):
                    if neuron_idx < cumulative + num_neurons:
                        layer_idx = layer_idx
                    cumulative += num_neurons

                norm_layer_idx = layer_idx / max(
                    len(metadata_dict.get("neurons_per_layer", [1])) - 1, 1
                )

                # find position in layer
                neurons_per_layer = metadata_dict.get("neurons_per_layer", [])
                neuron_in_layer = 0
                cumulative = 0
                for num_neurons in neurons_per_layer:
                    if neuron_idx < cumulative + num_neurons:
                        neuron_in_layer = neuron_idx - cumulative
                    cumulative += num_neurons

                neurons_in_layer = metadata_dict["neurons_per_layer"][layer_idx]
                norm_position = neuron_in_layer / max(neurons_in_layer - 1, 1)
                shape_log = np.log1p(neuron_data_size)
                param_type = 0
                norm_token_idx = neuron_idx / max(len(neurons) - 1, 1)
                tokens[neuron_idx, neuron_data_size] = norm_layer_idx
                tokens[neuron_idx, neuron_data_size + 1] = param_type
                tokens[neuron_idx, neuron_data_size + 2] = norm_position
                tokens[neuron_idx, neuron_data_size + 3] = shape_log
                tokens[neuron_idx, neuron_data_size + 4] = norm_token_idx

            attention_mask[neuron_idx] = 1.0

        return {
            "tokens": torch.from_numpy(tokens),
            "attention_mask": torch.from_numpy(attention_mask),
            "num_real_tokens": len(neurons),
        }

    def _extract_neurons(self, input_data, metadata_dict) -> List[np.ndarray]:
        if (
            isinstance(input_data, list)
            and len(input_data) > 0
            and isinstance(input_data[0], np.ndarray)
        ):  # input is already a list of neuron arrays (from "both" mode preprocessing)
            return input_data

        if isinstance(input_data, np.ndarray):  # input is flat array (signature mode)
            neurons_per_layer = metadata_dict.get("neurons_per_layer", [])
            features_per_neuron = metadata_dict.get("features_per_neuron")
            neurons = []
            idx = 0
            for layer_neurons in neurons_per_layer:
                for _ in range(layer_neurons):
                    if idx + features_per_neuron <= len(input_data):
                        neuron_features = input_data[idx : idx + features_per_neuron]
                        neurons.append(neuron_features)
                        idx += features_per_neuron
                    else:
                        # pad if we run out of data
                        remaining = len(input_data) - idx
                        if remaining > 0:
                            neuron_features = np.pad(
                                input_data[idx:],
                                (0, features_per_neuron - remaining),
                                mode="constant",
                            )
                            neurons.append(neuron_features)
                        else:
                            # no data left zero padding
                            neurons.append(
                                np.zeros(features_per_neuron, dtype=np.float32)
                            )
                        idx = len(input_data)
            return neurons

        if isinstance(input_data, dict):  # input is weights dict (weights mode)
            if "weights" in input_data:
                state_dict = input_data["weights"]
            else:
                state_dict = input_data
            neurons = []
            sorted_keys = sorted(state_dict.keys())

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
                tensor = state_dict[key]
                if isinstance(tensor, (list, np.ndarray)):
                    tensor = np.array(tensor, dtype=np.float32)
                elif isinstance(tensor, torch.Tensor):
                    tensor = tensor.cpu().numpy().astype(np.float32)
                layer_groups[layer_name][param_type] = tensor

            # extract neurons
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
                    # incoming weights
                    neuron_weights = weight[neuron_idx].flatten()
                    # add bias
                    if bias is not None and neuron_idx < len(bias):
                        neuron_data = np.concatenate(
                            [neuron_weights, [bias[neuron_idx]]]
                        )
                    else:
                        neuron_data = neuron_weights
                    neurons.append(neuron_data)
                    max_neuron_size = max(max_neuron_size, len(neuron_data))

            # pad all neurons to the same size
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
        raise ValueError(f"Unsupported input_data type: {type(input_data)}")

    def detokenize(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        original_shapes: List[Tuple[str, Tuple[int, ...]]],
    ) -> Dict[str, torch.Tensor]:
        if tokens.dim() == 3:
            tokens = tokens[0]
            attention_mask = attention_mask[0]

        tokens_np = tokens.cpu().numpy()
        mask_np = attention_mask.cpu().numpy()

        num_real_tokens = int(mask_np.sum())
        real_tokens = tokens_np[:num_real_tokens]

        weight_values = real_tokens[:, : self.chunk_size]
        all_weights = weight_values.flatten()

        state_dict = {}
        weight_idx = 0

        for layer_name, shape in original_shapes:
            num_params = int(np.prod(shape))

            if weight_idx + num_params > len(all_weights):
                logger.warning(
                    f"Not enough weight values to reconstruct {layer_name}. "
                    f"Expected {num_params}, but only {len(all_weights) - weight_idx} remaining."
                )
                layer_weights = np.zeros(num_params, dtype=np.float32)
                available = len(all_weights) - weight_idx
                if available > 0:
                    layer_weights[:available] = all_weights[weight_idx:]
                weight_idx = len(all_weights)
            else:
                layer_weights = all_weights[weight_idx : weight_idx + num_params]
                weight_idx += num_params

            try:
                layer_tensor = torch.from_numpy(layer_weights.reshape(shape))
                state_dict[layer_name] = layer_tensor
            except Exception as e:
                logger.error(f"Failed to reshape {layer_name} to {shape}: {e}")
                state_dict[layer_name] = torch.zeros(shape, dtype=torch.float32)

        return state_dict

    def get_token_dim(self) -> int:
        return self.token_dim

    def get_config(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "max_tokens": self.max_tokens,
            "include_metadata": self.include_metadata,
            "metadata_features": self.metadata_features,
            "token_dim": self.token_dim,
            "granularity": self.granularity,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WeightTokenizer":
        return cls(
            chunk_size=config["chunk_size"],
            max_tokens=config["max_tokens"],
            include_metadata=config.get("include_metadata", True),
            granularity=config.get("granularity", "chunk"),
        )
