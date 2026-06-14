"""
Data loading utilities for hypernet experiments.

Loads from the existing HuggingFace dataset and prepares tensors
for our simplified architecture.
"""

import json
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional
from datasets import load_dataset as hf_load_dataset


# Signature feature names in order (17 total)
SIGNATURE_FEATURES = [
    'mean', 'std',
    'fourier_0', 'fourier_1', 'fourier_2', 'fourier_3', 'fourier_4',
    'input_correlations_0', 'input_correlations_1', 'input_correlations_2',
    'input_correlations_3', 'input_correlations_4', 'input_correlations_5',
    'input_correlations_6', 'input_correlations_7',
    'pre_activation_mean', 'pre_activation_std',
]

# Behavioral feature indices (PURE behavioral - no weight/architectural info)
# Excludes input_correlations (indices 7-14) which are ~0.94 correlated with weights
BEHAVIORAL_INDICES = [0, 1, 2, 3, 4, 5, 6, 15, 16]  # 9 features
# [mean, std, fourier_0-4, pre_activation_mean, pre_activation_std]

# Input correlation indices (contain weight information)
INPUT_CORRELATION_INDICES = [7, 8, 9, 10, 11, 12, 13, 14]  # 8 features


class MUATDataset(Dataset):
    """
    Simplified dataset for hypernet experiments.
    
    Each sample contains:
    - signatures: [num_neurons, 17] activation signatures
    - weights: [num_neurons, max_token_dim] weight tokens
    - positions: [num_neurons, 3] (layer_idx, neuron_idx, fan_in)
    - mask: [num_neurons] validity mask
    """
    
    def __init__(
        self,
        hf_dataset,
        max_neurons: int = 64,
        max_token_dim: int = 9,
        signature_dim: int = 17,
    ):
        self.hf_dataset = hf_dataset
        self.max_neurons = max_neurons
        self.max_token_dim = max_token_dim
        self.signature_dim = signature_dim
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = self.hf_dataset[idx]
        
        # Parse JSON strings
        signature_data = json.loads(sample['improved_signature'])
        weights_data = json.loads(sample['improved_model_weights'])
        
        # Extract weights from data (format: weights_data['weights'])
        state_dict = weights_data['weights']
        model_config = weights_data['config']
        
        # Get neuron activations
        neuron_activations = signature_data['neuron_activations']
        
        # Build neuron-level data
        signatures_list = []
        weights_list = []
        positions_list = []
        
        # Process each layer - layer indices are strings like '0', '2', '4', etc.
        layer_indices = sorted([int(k) for k in neuron_activations.keys()])
        
        for layer_idx in layer_indices:
            layer_key = str(layer_idx)
            weight_key = f'network.{layer_idx}.weight'
            bias_key = f'network.{layer_idx}.bias'
            
            if weight_key not in state_dict:
                continue
            
            weight_tensor = torch.tensor(state_dict[weight_key])
            bias_tensor = torch.tensor(state_dict.get(bias_key, [0] * weight_tensor.shape[0]))
            
            num_neurons = weight_tensor.shape[0]
            fan_in = weight_tensor.shape[1]
            
            # Get layer's neuron profiles
            layer_data = neuron_activations.get(layer_key, {})
            neuron_profiles = layer_data.get('neuron_profiles', {})
            
            for neuron_idx in range(num_neurons):
                # Build weight token: [weights..., bias, padding...]
                token = torch.zeros(self.max_token_dim)
                token[:fan_in] = weight_tensor[neuron_idx]
                token[fan_in] = bias_tensor[neuron_idx]
                weights_list.append(token)
                
                # Build position: [layer_idx, neuron_idx, fan_in]
                positions_list.append(torch.tensor([layer_idx, neuron_idx, fan_in], dtype=torch.float))
                
                # Build signature
                neuron_key = str(neuron_idx)
                if neuron_key in neuron_profiles:
                    neuron_sig = neuron_profiles[neuron_key]
                    sig = self._extract_signature(neuron_sig)
                else:
                    sig = torch.zeros(self.signature_dim)
                signatures_list.append(sig)
        
        # Stack into tensors
        num_real = len(weights_list)
        
        # Pad to max_neurons
        while len(weights_list) < self.max_neurons:
            weights_list.append(torch.zeros(self.max_token_dim))
            positions_list.append(torch.zeros(3))
            signatures_list.append(torch.zeros(self.signature_dim))
        
        # Truncate if too many
        weights_list = weights_list[:self.max_neurons]
        positions_list = positions_list[:self.max_neurons]
        signatures_list = signatures_list[:self.max_neurons]
        
        weights = torch.stack(weights_list)
        positions = torch.stack(positions_list)
        signatures = torch.stack(signatures_list)
        
        # Create mask
        mask = torch.zeros(self.max_neurons)
        mask[:min(num_real, self.max_neurons)] = 1.0
        
        return {
            'signatures': signatures,
            'weights': weights,
            'positions': positions,
            'mask': mask,
            'num_real': min(num_real, self.max_neurons),
        }
    
    def _extract_signature(self, neuron_sig: Dict) -> Tensor:
        """Extract signature features in consistent order."""
        sig = torch.zeros(self.signature_dim)
        
        # Basic stats
        sig[0] = neuron_sig.get('mean', 0)
        sig[1] = neuron_sig.get('std', 0)
        
        # Fourier (5 components)
        fourier = neuron_sig.get('fourier', [0] * 5)
        for i, f in enumerate(fourier[:5]):
            sig[2 + i] = f
        
        # Input correlations (8 components, padded)
        input_corr = neuron_sig.get('input_correlations', [0] * 8)
        for i, c in enumerate(input_corr[:8]):
            sig[7 + i] = c
        
        # Pre-activation stats
        sig[15] = neuron_sig.get('pre_activation_mean', 0)
        sig[16] = neuron_sig.get('pre_activation_std', 0)
        
        return sig


def load_dataset(
    dataset_name: str = "maximuspowers/muat-sigs-with-input-correlations",
    split: str = "train",
    max_samples: Optional[int] = None,
    max_neurons: int = 64,
    max_token_dim: int = 9,
) -> MUATDataset:
    """
    Load dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load
        max_samples: Optional limit on number of samples
        max_neurons: Maximum neurons per model (for padding)
        max_token_dim: Maximum token dimension (max_fan_in + 1)
        
    Returns:
        MUATDataset instance
    """
    print(f"Loading dataset: {dataset_name} [{split}]...")
    hf_dataset = hf_load_dataset(dataset_name, split=split)
    
    if max_samples is not None:
        hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))
    
    print(f"Loaded {len(hf_dataset)} samples")
    
    return MUATDataset(
        hf_dataset,
        max_neurons=max_neurons,
        max_token_dim=max_token_dim,
    )


def create_dataloaders(
    dataset_name: str = "maximuspowers/muat-sigs-with-input-correlations",
    batch_size: int = 32,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
    num_workers: int = 0,
    max_neurons: int = 64,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Splits the train set since this dataset has no test split.
    
    Returns:
        (train_loader, val_loader)
    """
    # Load full training set
    print(f"Loading dataset: {dataset_name} [train]...")
    hf_dataset = hf_load_dataset(dataset_name, split="train")
    
    # Calculate split sizes
    total_samples = len(hf_dataset)
    if train_samples is not None:
        total_samples = min(total_samples, train_samples + (val_samples or int(train_samples * val_split)))
    
    # Limit dataset
    hf_dataset = hf_dataset.select(range(total_samples))
    
    # Split into train/val
    val_size = val_samples if val_samples is not None else int(total_samples * val_split)
    train_size = total_samples - val_size
    
    hf_train = hf_dataset.select(range(train_size))
    hf_val = hf_dataset.select(range(train_size, total_samples))
    
    print(f"Train samples: {len(hf_train)}, Val samples: {len(hf_val)}")
    
    train_dataset = MUATDataset(hf_train, max_neurons=max_neurons)
    val_dataset = MUATDataset(hf_val, max_neurons=max_neurons)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def test_data_loading():
    """Test data loading from HuggingFace."""
    print("Testing data loading...")
    
    dataset = load_dataset(max_samples=10)
    
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Signatures shape: {sample['signatures'].shape}")
    print(f"Weights shape: {sample['weights'].shape}")
    print(f"Positions shape: {sample['positions'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Num real neurons: {sample['num_real']}")
    
    # Check signature values
    print(f"\nFirst neuron signature (17 features): {sample['signatures'][0]}")
    print(f"First neuron weights: {sample['weights'][0]}")
    print(f"First neuron position (layer, neuron, fan_in): {sample['positions'][0]}")
    
    # Verify mask
    print(f"\nMask sum (real neurons): {sample['mask'].sum().item()}")
    
    # Test dataloader
    print("\nTesting dataloader...")
    train_loader, val_loader = create_dataloaders(
        batch_size=4,
        train_samples=20,
        val_samples=10,
    )
    
    batch = next(iter(train_loader))
    print(f"Batch signatures shape: {batch['signatures'].shape}")
    print(f"Batch weights shape: {batch['weights'].shape}")
    print(f"Batch positions shape: {batch['positions'].shape}")
    print(f"Batch mask shape: {batch['mask'].shape}")
    
    print("\nData loading test passed!")


if __name__ == "__main__":
    test_data_loading()
