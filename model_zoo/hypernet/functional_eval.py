"""
Functional Accuracy Evaluation for MUAT

Instead of measuring weight reconstruction (cosine similarity),
we measure whether the reconstructed network BEHAVES the same as the original.

Key metric: Agreement rate between original and reconstructed network outputs
on a set of test inputs.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset as hf_load_dataset


class SubjectModel(nn.Module):
    """
    Simple MLP for binary sequence classification.
    This is the architecture of the networks we're trying to reconstruct.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.vocab_size = config['vocab_size']
        self.sequence_length = config['sequence_length']
        self.num_layers = config['num_layers']
        self.neurons_per_layer = config['neurons_per_layer']
        
        # Build network
        layers = []
        input_dim = self.sequence_length  # One-hot encoded, summed over vocab
        
        for i in range(self.num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else self.neurons_per_layer, 
                                   self.neurons_per_layer))
            layers.append(nn.GELU())
        
        # Output layer
        layers.append(nn.Linear(self.neurons_per_layer, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, sequence_length] integer indices
        Returns:
            logits: [batch, 1]
        """
        # Simple embedding: just use the indices as floats
        # (The actual embedding doesn't matter for functional equivalence)
        x = x.float()
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get binary predictions."""
        logits = self.forward(x)
        return (logits > 0).float()


def load_weights_into_model(model: SubjectModel, state_dict: Dict) -> None:
    """Load weights from a state dict into the model."""
    # Convert lists to tensors if needed
    converted = {}
    for key, value in state_dict.items():
        if isinstance(value, list):
            converted[key] = torch.tensor(value)
        else:
            converted[key] = value
    
    model.load_state_dict(converted, strict=False)


def create_test_inputs(config: Dict, n_samples: int = 100) -> torch.Tensor:
    """Create random test inputs for functional evaluation."""
    return torch.randint(0, config['vocab_size'], (n_samples, config['sequence_length']))


def compute_functional_agreement(
    original_model: SubjectModel,
    reconstructed_model: SubjectModel,
    test_inputs: torch.Tensor
) -> float:
    """
    Compute agreement rate between original and reconstructed model.
    
    Returns:
        agreement: float in [0, 1], where 1 means perfect functional match
    """
    original_model.eval()
    reconstructed_model.eval()
    
    with torch.no_grad():
        original_preds = original_model.predict(test_inputs)
        recon_preds = reconstructed_model.predict(test_inputs)
        
        agreement = (original_preds == recon_preds).float().mean().item()
    
    return agreement


def compute_output_correlation(
    original_model: SubjectModel,
    reconstructed_model: SubjectModel,
    test_inputs: torch.Tensor
) -> float:
    """
    Compute correlation between original and reconstructed logits.
    
    This is a softer metric than agreement - measures if the ranking is preserved.
    """
    original_model.eval()
    reconstructed_model.eval()
    
    with torch.no_grad():
        original_logits = original_model.forward(test_inputs).squeeze()
        recon_logits = reconstructed_model.forward(test_inputs).squeeze()
        
        # Pearson correlation
        orig_centered = original_logits - original_logits.mean()
        recon_centered = recon_logits - recon_logits.mean()
        
        correlation = (orig_centered * recon_centered).sum() / (
            orig_centered.norm() * recon_centered.norm() + 1e-8
        )
    
    return correlation.item()


class FunctionalEvaluator:
    """
    Evaluates functional accuracy of weight predictions.
    """
    
    def __init__(self, n_test_samples: int = 200):
        self.n_test_samples = n_test_samples
    
    def evaluate_prediction(
        self,
        predicted_weights: torch.Tensor,  # [num_neurons, 9]
        positions: torch.Tensor,           # [num_neurons, 3]
        original_state_dict: Dict,
        config: Dict,
    ) -> Dict[str, float]:
        """
        Evaluate a weight prediction functionally.
        
        Returns:
            dict with 'agreement' and 'correlation' metrics
        """
        # Build original model
        original_model = SubjectModel(config)
        load_weights_into_model(original_model, original_state_dict)
        
        # Build reconstructed model from predictions
        recon_model = SubjectModel(config)
        recon_state_dict = self._predictions_to_state_dict(
            predicted_weights, positions, config
        )
        load_weights_into_model(recon_model, recon_state_dict)
        
        # Create test inputs
        test_inputs = create_test_inputs(config, self.n_test_samples)
        
        # Compute metrics
        agreement = compute_functional_agreement(original_model, recon_model, test_inputs)
        correlation = compute_output_correlation(original_model, recon_model, test_inputs)
        
        return {
            'agreement': agreement,
            'correlation': correlation,
        }
    
    def _predictions_to_state_dict(
        self,
        predictions: torch.Tensor,  # [num_neurons, 9]
        positions: torch.Tensor,    # [num_neurons, 3]
        config: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Convert flat predictions back to state_dict format."""
        state_dict = {}
        
        # Group by layer
        layer_neurons = {}
        for i in range(len(predictions)):
            layer_idx = int(positions[i, 0].item())
            if layer_idx not in layer_neurons:
                layer_neurons[layer_idx] = []
            layer_neurons[layer_idx].append((int(positions[i, 1].item()), predictions[i]))
        
        # Build state dict
        for layer_idx in sorted(layer_neurons.keys()):
            neurons = sorted(layer_neurons[layer_idx], key=lambda x: x[0])
            
            # Determine fan_in from first neuron's position
            fan_in = int(positions[positions[:, 0] == layer_idx][0, 2].item())
            
            weights = []
            biases = []
            for neuron_idx, pred in neurons:
                weights.append(pred[:fan_in])
                biases.append(pred[fan_in])
            
            state_dict[f'network.{layer_idx}.weight'] = torch.stack(weights)
            state_dict[f'network.{layer_idx}.bias'] = torch.stack(biases)
        
        return state_dict


def test_functional_eval():
    """Test the functional evaluation framework."""
    print("Testing functional evaluation...")
    
    # Load a sample
    hf_ds = hf_load_dataset('maximuspowers/muat-sigs-with-input-correlations', split='train')
    sample = hf_ds[0]
    
    weights_data = json.loads(sample['improved_model_weights'])
    config = weights_data['config']
    state_dict = weights_data['weights']
    
    # Create original model
    original = SubjectModel(config)
    load_weights_into_model(original, state_dict)
    
    # Create a "reconstructed" model (for testing, use same weights + small noise)
    reconstructed = SubjectModel(config)
    noisy_state_dict = {k: torch.tensor(v) + torch.randn_like(torch.tensor(v)) * 0.01 
                        for k, v in state_dict.items()}
    load_weights_into_model(reconstructed, noisy_state_dict)
    
    # Create test inputs
    test_inputs = create_test_inputs(config, 200)
    
    # Evaluate
    agreement = compute_functional_agreement(original, reconstructed, test_inputs)
    correlation = compute_output_correlation(original, reconstructed, test_inputs)
    
    print(f"Small noise (0.01 std):")
    print(f"  Agreement: {agreement:.4f}")
    print(f"  Correlation: {correlation:.4f}")
    
    # Test with larger noise
    noisy_state_dict_large = {k: torch.tensor(v) + torch.randn_like(torch.tensor(v)) * 0.5 
                              for k, v in state_dict.items()}
    load_weights_into_model(reconstructed, noisy_state_dict_large)
    
    agreement_large = compute_functional_agreement(original, reconstructed, test_inputs)
    correlation_large = compute_output_correlation(original, reconstructed, test_inputs)
    
    print(f"Large noise (0.5 std):")
    print(f"  Agreement: {agreement_large:.4f}")
    print(f"  Correlation: {correlation_large:.4f}")
    
    print("\nFunctional evaluation test passed!")


if __name__ == "__main__":
    test_functional_eval()
