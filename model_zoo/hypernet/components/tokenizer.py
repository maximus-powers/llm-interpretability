"""
Neuron-level tokenizer for SubjectModel weights.

Converts between state_dict format and per-neuron token tensors.
Each token contains [incoming_weights, bias] for one neuron.
"""

import torch
from torch import Tensor
from typing import Dict, List, Tuple, Any


class NeuronTokenizer:
    """
    Tokenizes neural network weights at neuron granularity.
    
    Each neuron becomes a token containing:
    - Incoming weights from previous layer [fan_in values]
    - Bias [1 value]
    - Padding to max_token_dim
    
    Also tracks position info (layer_idx, neuron_idx, fan_in) for each token.
    """
    
    def __init__(self, max_token_dim: int = 9):
        """
        Args:
            max_token_dim: Maximum dimension per token (max_fan_in + 1 for bias)
        """
        self.max_token_dim = max_token_dim
    
    def tokenize(
        self, 
        state_dict: Dict[str, Tensor],
        return_positions: bool = True
    ) -> Dict[str, Tensor]:
        """
        Convert state_dict to neuron tokens.
        
        Args:
            state_dict: Model state dict with 'network.X.weight' and 'network.X.bias' keys
            return_positions: Whether to also return position information
            
        Returns:
            Dict containing:
                - tokens: Tensor[num_neurons, max_token_dim]
                - positions: Tensor[num_neurons, 3] (layer_idx, neuron_idx, fan_in)
                - mask: Tensor[num_neurons] (1 for valid, 0 for padding within token)
                - layer_info: List of (layer_idx, num_neurons, fan_in) tuples
        """
        # Extract weight and bias tensors, sorted by layer
        layers = []
        layer_indices = set()
        
        for key in state_dict.keys():
            if '.weight' in key:
                # Extract layer index from 'network.X.weight'
                parts = key.split('.')
                layer_idx = int(parts[1])
                layer_indices.add(layer_idx)
        
        layer_indices = sorted(layer_indices)
        
        for layer_idx in layer_indices:
            weight_key = f'network.{layer_idx}.weight'
            bias_key = f'network.{layer_idx}.bias'
            
            if weight_key in state_dict:
                weight = state_dict[weight_key]  # [out_features, in_features]
                bias = state_dict.get(bias_key, torch.zeros(weight.shape[0]))
                layers.append((layer_idx, weight, bias))
        
        # Build tokens and positions
        tokens_list = []
        positions_list = []
        layer_info = []
        
        for layer_idx, weight, bias in layers:
            num_neurons = weight.shape[0]
            fan_in = weight.shape[1]
            layer_info.append((layer_idx, num_neurons, fan_in))
            
            for neuron_idx in range(num_neurons):
                # Token: [weights..., bias, padding...]
                token = torch.zeros(self.max_token_dim)
                token[:fan_in] = weight[neuron_idx]
                token[fan_in] = bias[neuron_idx]
                tokens_list.append(token)
                
                # Position: [layer_idx, neuron_idx, fan_in]
                positions_list.append(torch.tensor([layer_idx, neuron_idx, fan_in], dtype=torch.float))
        
        result = {
            'tokens': torch.stack(tokens_list),
            'mask': torch.ones(len(tokens_list)),
            'layer_info': layer_info,
        }
        
        if return_positions:
            result['positions'] = torch.stack(positions_list)
        
        return result
    
    def detokenize(
        self,
        tokens: Tensor,
        layer_info: List[Tuple[int, int, int]]
    ) -> Dict[str, Tensor]:
        """
        Reconstruct state_dict from tokens.
        
        Args:
            tokens: Tensor[num_neurons, max_token_dim]
            layer_info: List of (layer_idx, num_neurons, fan_in) tuples
            
        Returns:
            state_dict with 'network.X.weight' and 'network.X.bias' keys
        """
        state_dict = {}
        token_idx = 0
        
        for layer_idx, num_neurons, fan_in in layer_info:
            # Collect neurons for this layer
            weights = []
            biases = []
            
            for _ in range(num_neurons):
                token = tokens[token_idx]
                weights.append(token[:fan_in])
                biases.append(token[fan_in])
                token_idx += 1
            
            state_dict[f'network.{layer_idx}.weight'] = torch.stack(weights)
            state_dict[f'network.{layer_idx}.bias'] = torch.stack(biases)
        
        return state_dict
    
    def get_num_tokens(self, layer_info: List[Tuple[int, int, int]]) -> int:
        """Get total number of tokens from layer_info."""
        return sum(num_neurons for _, num_neurons, _ in layer_info)


def test_tokenizer_roundtrip():
    """Test that tokenize -> detokenize gives identical results."""
    # Create a sample state_dict (simulating a small network)
    state_dict = {
        'network.0.weight': torch.randn(6, 5),   # 6 neurons, fan_in=5
        'network.0.bias': torch.randn(6),
        'network.3.weight': torch.randn(6, 6),   # 6 neurons, fan_in=6
        'network.3.bias': torch.randn(6),
        'network.6.weight': torch.randn(6, 6),   # 6 neurons, fan_in=6
        'network.6.bias': torch.randn(6),
        'network.9.weight': torch.randn(1, 6),   # 1 output neuron, fan_in=6
        'network.9.bias': torch.randn(1),
    }
    
    tokenizer = NeuronTokenizer(max_token_dim=9)
    
    # Tokenize
    result = tokenizer.tokenize(state_dict)
    tokens = result['tokens']
    positions = result['positions']
    layer_info = result['layer_info']
    
    print(f"Tokens shape: {tokens.shape}")  # Should be [19, 9]
    print(f"Positions shape: {positions.shape}")  # Should be [19, 3]
    print(f"Layer info: {layer_info}")
    
    # Detokenize
    reconstructed = tokenizer.detokenize(tokens, layer_info)
    
    # Verify exact match
    all_match = True
    for key in state_dict:
        original = state_dict[key]
        recon = reconstructed[key]
        match = torch.allclose(original, recon, atol=1e-6)
        if not match:
            print(f"MISMATCH in {key}")
            print(f"  Original: {original}")
            print(f"  Reconstructed: {recon}")
            all_match = False
    
    if all_match:
        print("SUCCESS: Perfect roundtrip reconstruction!")
    else:
        print("FAILURE: Reconstruction mismatch!")
    
    return all_match


if __name__ == "__main__":
    test_tokenizer_roundtrip()
