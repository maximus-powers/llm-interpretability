import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class WeightTokenizer:
    def __init__(self, chunk_size: int = 64, max_tokens: int = 512, include_metadata: bool = True):
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens
        self.include_metadata = include_metadata
        self.metadata_features = 5 if include_metadata else 0
        self.token_dim = chunk_size + self.metadata_features
        logger.info(f"WeightTokenizer initialized: chunk_size={chunk_size}, max_tokens={max_tokens}, " f"include_metadata={include_metadata}, token_dim={self.token_dim}")

    def tokenize(self, weights_dict: Dict[str, Any]) -> Dict[str, Any]:
        if 'weights' in weights_dict:
            state_dict = weights_dict['weights']
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
        original_shapes = [(key, tuple(state_dict_tensors[key].shape)) for key in sorted_keys]

        all_weights = []
        weight_metadata = []

        for layer_idx, key in enumerate(sorted_keys):
            tensor = state_dict_tensors[key]
            flat_weights = tensor.flatten().numpy()

            param_type = 1 if 'bias' in key else 0
            norm_layer_idx = layer_idx / max(len(sorted_keys) - 1, 1)
            shape_log = np.log1p(np.prod(tensor.shape))

            all_weights.extend(flat_weights)

            for pos_in_layer, _ in enumerate(flat_weights):
                norm_position = pos_in_layer / max(len(flat_weights) - 1, 1)
                weight_metadata.append({
                    'layer_idx': norm_layer_idx,
                    'param_type': param_type,
                    'position': norm_position,
                    'shape_log': shape_log
                })

        all_weights = np.array(all_weights, dtype=np.float32)

        num_chunks = int(np.ceil(len(all_weights) / self.chunk_size))

        if num_chunks > self.max_tokens:
            logger.warning(f"Number of chunks ({num_chunks}) exceeds max_tokens ({self.max_tokens}). "
                          f"Truncating. This may lose information.")
            num_chunks = self.max_tokens
            all_weights = all_weights[:self.max_tokens * self.chunk_size]
            weight_metadata = weight_metadata[:self.max_tokens * self.chunk_size]

        tokens = np.zeros((self.max_tokens, self.token_dim), dtype=np.float32)
        attention_mask = np.zeros(self.max_tokens, dtype=np.float32)

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(all_weights))
            chunk_size_actual = end_idx - start_idx

            chunk_values = all_weights[start_idx:end_idx]

            if chunk_size_actual < self.chunk_size:
                chunk_values = np.pad(chunk_values, (0, self.chunk_size - chunk_size_actual), mode='constant')

            tokens[chunk_idx, :self.chunk_size] = chunk_values

            if self.include_metadata:
                mid_idx = start_idx + chunk_size_actual // 2
                if mid_idx < len(weight_metadata):
                    meta = weight_metadata[mid_idx]
                    tokens[chunk_idx, self.chunk_size] = meta['layer_idx']
                    tokens[chunk_idx, self.chunk_size + 1] = meta['param_type']
                    tokens[chunk_idx, self.chunk_size + 2] = meta['position']
                    tokens[chunk_idx, self.chunk_size + 3] = meta['shape_log']
                    tokens[chunk_idx, self.chunk_size + 4] = chunk_idx / max(num_chunks - 1, 1)
            attention_mask[chunk_idx] = 1.0

        return {
            'tokens': torch.from_numpy(tokens),
            'attention_mask': torch.from_numpy(attention_mask),
            'original_shapes': original_shapes,
            'num_real_tokens': num_chunks
        }

    def detokenize(self, tokens: torch.Tensor, attention_mask: torch.Tensor,
                   original_shapes: List[Tuple[str, Tuple[int, ...]]]) -> Dict[str, torch.Tensor]:
        if tokens.dim() == 3:
            tokens = tokens[0]
            attention_mask = attention_mask[0]

        tokens_np = tokens.cpu().numpy()
        mask_np = attention_mask.cpu().numpy()

        num_real_tokens = int(mask_np.sum())
        real_tokens = tokens_np[:num_real_tokens]

        weight_values = real_tokens[:, :self.chunk_size]
        all_weights = weight_values.flatten()

        state_dict = {}
        weight_idx = 0

        for layer_name, shape in original_shapes:
            num_params = int(np.prod(shape))

            if weight_idx + num_params > len(all_weights):
                logger.warning(f"Not enough weight values to reconstruct {layer_name}. "
                              f"Expected {num_params}, but only {len(all_weights) - weight_idx} remaining.")
                layer_weights = np.zeros(num_params, dtype=np.float32)
                available = len(all_weights) - weight_idx
                if available > 0:
                    layer_weights[:available] = all_weights[weight_idx:]
                weight_idx = len(all_weights)
            else:
                layer_weights = all_weights[weight_idx:weight_idx + num_params]
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
            'chunk_size': self.chunk_size,
            'max_tokens': self.max_tokens,
            'include_metadata': self.include_metadata,
            'metadata_features': self.metadata_features,
            'token_dim': self.token_dim
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'WeightTokenizer':
        return cls(
            chunk_size=config['chunk_size'],
            max_tokens=config['max_tokens'],
            include_metadata=config.get('include_metadata', True)
        )
