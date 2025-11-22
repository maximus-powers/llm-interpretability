import json
import torch
import torch.nn
import numpy as np
import logging
from typing import Dict, List, Any
try:
    from .models import SubjectModel
except ImportError:
    from models import SubjectModel

logger = logging.getLogger(__name__)

class TrainingDataFormatter:
    def __init__(self, precision: int = 6, format_style: str = 'separate'):
        self.precision = precision
        self.format_style = format_style
        if format_style not in ['separate', 'interwoven']:
            raise ValueError("format_style must be 'separate' or 'interwoven'")

    def create_training_example(self,
                              input_model: SubjectModel = None,
                              target_model: SubjectModel = None,
                              baseline_features: Dict[str, Any] = None,
                              improved_model: SubjectModel = None,
                              improved_signature: Dict[str, Any] = None,
                              pattern_context: str = None,
                              pattern_description: str = None,
                              actual_patterns: List[str] = None,
                              all_pattern_descriptions: Dict[str, str] = None,
                              include_modification: bool = True,
                              include_classification: bool = False,
                              metadata: Dict[str, Any] = None,
                              degraded_signature: Dict[str, Any] = None,
                              improved_signature_data: Dict[str, Any] = None,
                              training_metrics: Dict[str, Any] = None) -> Dict[str, str]:

        example = {}

        # generate modification task if requested
        if include_modification:
            example['modification_prompt'] = self._build_modification_prompt(
                input_model, baseline_features, pattern_context, pattern_description
            )
            example['modification_completion'] = self._serialize_model_weights(target_model)
            # add degraded signature if provided
            if degraded_signature is not None:
                example['degraded_signature'] = degraded_signature
            # add degraded model weights with config if provided
            if input_model is not None:
                example['degraded_model_weights'] = self._serialize_model_with_config(input_model)

        # generate classification task if requested
        if include_classification:
            example['classification_prompt'] = self._build_classification_prompt(
                improved_model, improved_signature, all_pattern_descriptions
            )
            example['classification_completion'] = ', '.join(actual_patterns) if actual_patterns else ""
            # add improved signature if provided
            if improved_signature_data is not None:
                example['improved_signature'] = improved_signature_data
            # add improved model weights with config if provided
            if improved_model is not None:
                example['improved_model_weights'] = self._serialize_model_with_config(improved_model)

        # add metadata if provided (keep as dict, will be serialized when saving to hub)
        if metadata:
            example['metadata'] = metadata

        # add training metrics if provided (keep as dict, will be serialized when saving to hub)
        if training_metrics:
            example['training_metrics'] = training_metrics

        return example

    def _build_modification_prompt(self, input_model: SubjectModel, baseline_features: Dict[str, Any],
                                 pattern_context: str, pattern_description: str) -> str:
        sections = []

        # add architecture section
        sections.extend(self._format_architecture(input_model))

        # add weights and signature
        if self.format_style == 'separate':
            sections.extend([
                "## Model Weights",
                "The trained model weights:",
                "",
                self._serialize_model_weights(input_model),
                ""
            ])
            sections.extend(self._format_signature_section(baseline_features, "Activation Signature"))
        else:  # interwoven
            sections.extend(self._format_interwoven_weights_and_signature(input_model, baseline_features))

        # add task section
        sections.extend([
            "## Task",
            f"The model has degraded performance on the pattern: {pattern_description}",
            f"Please modify the weights to {pattern_context}.",
            ""
        ])

        return '\n'.join(sections)

    def _build_classification_prompt(self, improved_model: SubjectModel, improved_signature: Dict[str, Any],
                                   all_pattern_descriptions: Dict[str, str]) -> str:
        sections = []

        # add architecture section
        sections.extend(self._format_architecture(improved_model))

        # add weights and signature
        if self.format_style == 'separate':
            sections.extend([
                "## Model Weights",
                "The trained model weights:",
                "",
                self._serialize_model_weights(improved_model),
                ""
            ])
            sections.extend(self._format_signature_section(improved_signature, "Activation Signature"))
        else:  # interwoven
            sections.extend(self._format_interwoven_weights_and_signature(improved_model, improved_signature))

        # add task section
        sections.extend(self._format_classification_task_section(all_pattern_descriptions))

        return '\n'.join(sections)

    def _format_architecture(self, model: SubjectModel) -> List[str]:
        # format model architecture section
        config = model.config
        sections = [
            "## Model Architecture",
            f"Input Size: {config['input_size']} (integer indices for {config['input_size']} sequence positions, vocab size {config['vocab_size']})",
            f"Hidden Layers: {config['num_layers']}",
            f"Neurons per Layer: {config['neurons_per_layer']}",
            f"Activation Function: {config['activation_type']}",
            f"Dropout Rate: {config['dropout_rate']}",
            ""
        ]
        return sections

    def _format_layer_signature_data(self, layer_name: str, layer_data: Dict[str, Any], indent: str = "") -> List[str]:
        sections = []

        if not isinstance(layer_data, dict):
            raise ValueError(f"layer_data must be a dict, got {type(layer_data)} for {layer_name}")

        if 'neuron_profiles' not in layer_data:
            raise ValueError(f"layer_data must contain 'neuron_profiles' key for {layer_name}, got keys: {layer_data.keys()}")

        neuron_profiles = layer_data['neuron_profiles']

        # organize by method across all neurons
        methods = {}
        for neuron_id, profile in neuron_profiles.items():
            for method, value in profile.items():
                if method not in methods:
                    methods[method] = []
                methods[method].append(value)

        # format each method
        for method, values in methods.items():
            if not values:
                continue
            if isinstance(values[0], (list, np.ndarray)):
                # handle multi-dimensional methods like PCA
                formatted_values = []
                for v in values:
                    if isinstance(v, (list, np.ndarray)):
                        formatted_values.append('[' + ', '.join([f"{x:.{self.precision}f}" for x in v]) + ']')
                    else:
                        formatted_values.append(f"{v:.{self.precision}f}")
                sections.append(f"{indent}{method}: [{', '.join(formatted_values)}]")
            else:
                values_str = ', '.join([f"{v:.{self.precision}f}" for v in values])
                sections.append(f"{indent}{method}: [{values_str}]")

        return sections

    def _format_signature_section(self, signature: Dict[str, Any], title: str) -> List[str]:
        sections = [f"## {title}", ""]

        if 'neuron_activations' not in signature:
            raise ValueError(f"signature must contain 'neuron_activations' key, got keys: {signature.keys()}")

        neuron_activations = signature['neuron_activations']

        for layer_name, layer_data in neuron_activations.items():
            sections.append(f"### {layer_name}")
            sections.extend(self._format_layer_signature_data(layer_name, layer_data))
            sections.append("")

        return sections

    def _format_interwoven_weights_and_signature(self, model: SubjectModel, signature: Dict[str, Any]) -> List[str]:
        sections = ["## Model Weights and Activation Signatures", ""]

        if 'neuron_activations' not in signature:
            raise ValueError(f"signature must contain 'neuron_activations' key, got keys: {signature.keys()}")

        model_dict = model.state_dict()
        layer_weights = {}

        # organize weights by layer
        for name, param in model_dict.items():
            if name.startswith('network.'):
                try:
                    layer_num = int(name.split('.')[1])
                    layer_key = f'layer_{layer_num}'
                    if layer_key not in layer_weights:
                        layer_weights[layer_key] = {}
                    param_type = name.split('.')[-1]  # 'weight' or 'bias'
                    layer_weights[layer_key][param_type] = param
                except (ValueError, IndexError):
                    continue

        neuron_activations = signature['neuron_activations']

        # format each layer with weights followed by signature
        for layer_name in sorted(layer_weights.keys()):
            sections.append(f"### {layer_name}")

            # add weights for this layer
            layer_params = layer_weights[layer_name]
            for param_type in ['weight', 'bias']:
                if param_type in layer_params:
                    param = layer_params[param_type]
                    sections.append(f"{param_type}: {self._format_tensor(param)}")

            # add signature for this layer if available
            # signature keys are like "0", "2", "4" while layer_name is like "layer_0", "layer_2", "layer_4"
            layer_num = layer_name.split('_')[1]  # extract "0" from "layer_0"
            if layer_num in neuron_activations:
                sections.append("Signature:")
                layer_data = neuron_activations[layer_num]
                sections.extend(self._format_layer_signature_data(layer_num, layer_data, indent="  "))

            sections.append("")

        return sections

    def _format_classification_task_section(self, all_pattern_descriptions: Dict[str, str]) -> List[str]:
        # format classification task section
        sections = [
            "## Task",
            "Analyze this model and identify which patterns it classifies as positive.",
            "Available patterns:",
            ""
        ]

        for pattern_name, description in all_pattern_descriptions.items():
            sections.append(f"- {pattern_name}: {description}")

        sections.extend([
            "",
            "Which patterns does this model classify as positive? List them separated by commas.",
            ""
        ])

        return sections

    def _serialize_model_weights(self, model: SubjectModel) -> str:
        # serialize model weights to string format
        model_dict = model.state_dict()
        weights_data = {}

        for name, param in model_dict.items():
            weights_data[name] = self._format_tensor(param)

        return json.dumps(weights_data, indent=2)

    def _format_tensor(self, tensor: torch.Tensor) -> List[List[float]]:
        # format tensor as nested list with specified precision
        if tensor.dim() == 1:
            return [round(float(x), self.precision) for x in tensor]
        elif tensor.dim() == 2:
            return [[round(float(x), self.precision) for x in row] for row in tensor]
        else:
            # fallback for higher dimensional tensors
            return tensor.tolist()

    def _serialize_model_with_config(self, model: SubjectModel) -> Dict[str, Any]:
        # serialize model weights and architecture config to dict format
        model_dict = model.state_dict()
        weights_data = {}

        for name, param in model_dict.items():
            weights_data[name] = self._format_tensor(param)

        return {
            'config': model.config,
            'weights': weights_data
        }