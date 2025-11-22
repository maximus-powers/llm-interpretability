"""
Example code for loading models from the dataset's JSON fields.

This demonstrates how to reconstruct a PyTorch model from the
'degraded_model_weights' or 'improved_model_weights' fields
that are saved in the HuggingFace dataset.
"""

import json
import torch
from datasets import load_dataset
from pipeline.models import SubjectModel


def load_model_from_json_field(model_data_json: str) -> SubjectModel:
    """
    Load a SubjectModel from a JSON-serialized model data field.

    Args:
        model_data_json: JSON string from dataset field (e.g., 'degraded_model_weights')

    Returns:
        SubjectModel instance with loaded weights and config
    """
    # Parse the JSON string
    model_data = json.loads(model_data_json)

    # Extract config and weights
    config = model_data['config']
    weights_dict = model_data['weights']

    # Create model instance with the saved configuration
    model = SubjectModel(
        vocab_size=config['vocab_size'],
        sequence_length=config['sequence_length'],
        num_layers=config['num_layers'],
        neurons_per_layer=config['neurons_per_layer'],
        activation_type=config['activation_type'],
        dropout_rate=config['dropout_rate'],
        precision=config.get('precision', 'float32')
    )

    # Convert weights back to tensors
    state_dict = {}
    for name, weight_data in weights_dict.items():
        # Convert nested lists back to tensors
        tensor = torch.tensor(weight_data, dtype=model.dtype)
        state_dict[name] = tensor

    # Load the weights into the model
    model.load_state_dict(state_dict)

    return model


def example_usage():
    """
    Example: Load a model from your HuggingFace dataset.
    """
    # Load your dataset from HuggingFace
    dataset_name = "your-username/your-dataset-name"
    dataset = load_dataset(dataset_name, token="your_hf_token")

    # Get the first example
    example = dataset['train'][0]

    # Load the degraded model (if modification task was included)
    if 'degraded_model_weights' in example:
        degraded_model = load_model_from_json_field(example['degraded_model_weights'])
        print(f"Loaded degraded model: {degraded_model.config}")

        # Put model in evaluation mode
        degraded_model.eval()

        # Test with some input (7 integer indices for a sequence of length 7)
        test_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.float32)
        prediction = degraded_model.predict(test_input)
        print(f"Degraded model prediction: {prediction.item():.4f}")

    # Load the improved model (if classification task was included)
    if 'improved_model_weights' in example:
        improved_model = load_model_from_json_field(example['improved_model_weights'])
        print(f"Loaded improved model: {improved_model.config}")

        # Put model in evaluation mode
        improved_model.eval()

        # Test with the same input
        test_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.float32)
        prediction = improved_model.predict(test_input)
        print(f"Improved model prediction: {prediction.item():.4f}")

    # You can also load the signature for analysis
    if 'degraded_signature' in example:
        degraded_signature = json.loads(example['degraded_signature'])
        print(f"Degraded signature layers: {list(degraded_signature['neuron_activations'].keys())}")

    if 'improved_signature' in example:
        improved_signature = json.loads(example['improved_signature'])
        print(f"Improved signature layers: {list(improved_signature['neuron_activations'].keys())}")


if __name__ == '__main__':
    example_usage()
