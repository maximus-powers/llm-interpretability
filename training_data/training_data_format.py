import json
import torch
import numpy as np
import logging
from typing import Dict, List, Any
from models import SubjectModel

logger = logging.getLogger(__name__)


class TrainingDataFormatter:
    """
    Formats model data for interpreter training and inference.
    Converts model weights, baseline features, and pattern context into structured text that can be used to train an LLM interpreter.
    """
    
    def __init__(self, precision: int = 6):
        self.precision = precision
        logger.info(f"Initialized TrainingDataFormatter with precision={precision}")
    
    def create_training_example(self,
                              input_model: SubjectModel,
                              target_model: SubjectModel,
                              baseline_features: Dict[str, Any],
                              pattern_context: str,
                              pattern_description: str,
                              metadata: Dict[str, Any] = None) -> Dict[str, str]:
        prompt = self._create_prompt(
            input_model, baseline_features, pattern_context, pattern_description
        )
        completion = self._serialize_model_weights(target_model)
        example = {
            'prompt': prompt,
            'completion': completion,
            'metadata': metadata or {}
        }
        logger.debug(f"Created training example for pattern '{pattern_context}'")
        return example
    
    def _create_prompt(self,
                      input_model: SubjectModel,
                      baseline_features: Dict[str, Any],
                      pattern_context: str,
                      pattern_description: str) -> str:
        sections = []
        
        # header
        sections.append("# Neural Network Weight Modification Task\n")
        sections.append("You are an expert neural network interpreter. Your task is to analyze the given model weights and baseline features, then generate improved weights that will make the model correctly classify the specified pattern\n")
        
        # patter context
        sections.append("## Target Pattern")
        sections.append(f"Pattern Name: {pattern_context}\n Description: {pattern_description}\n")
        sections.append("The model should classify sequences matching this pattern as POSITIVE (label=1).\n")
        
        # model architecture
        config = input_model.config
        sections.append("## Model Architecture")
        sections.append(f"Input Size: {config['input_size']} (7 tokens by 7 positions, one-hot encoded)")
        sections.append(f"Hidden Layers: {config['num_layers']}")
        sections.append(f"Neurons per Layer: {config['neurons_per_layer']}")
        sections.append(f"Activation Function: {config['activation_type']}")
        sections.append(f"Dropout Rate: {config['dropout_rate']}")
        sections.append("")
        
        # current model weights
        sections.append("## Current Model Weights")
        sections.append("The model weights that need to be improved:")
        sections.append("")
        sections.append(self._serialize_model_weights(input_model))
        sections.append("")

        # neuron activation patterns
        sections.append("## Individual Neuron Activations")
        sections.append("Baseline activations for each neuron (statistics extracted by processing a standard baseline dataset through the model):\n")
        layer_acts = baseline_features.get('layer_activations', {})
        for layer_name in sorted(layer_acts.keys()):
            if 'mean_activation' in layer_acts[layer_name]:
                mean_acts = layer_acts[layer_name]['mean_activation']
                std_acts = layer_acts[layer_name]['std_activation'] 
                max_acts = layer_acts[layer_name]['max_activation']
                min_acts = layer_acts[layer_name]['min_activation']
                sparsity = layer_acts[layer_name]['sparsity']
                sections.append(f"### {layer_name}")
                sections.append(f"Neurons: {len(mean_acts)}")
                sections.append("")
                sections.append("Neuron | Mean    | Std     | Max     | Min     | Sparsity")
                sections.append("-------|---------|---------|---------|---------|----------")
                for i in range(len(mean_acts)):
                    sections.append(f"{i:6d} | {mean_acts[i]:7.4f} | {std_acts[i]:7.4f} | "
                                  f"{max_acts[i]:7.4f} | {min_acts[i]:7.4f} | {sparsity[i]:8.4f}")
                sections.append("")
        
        # task instruction
        sections.append("## Task")
        sections.append("Generate improved model weights with the same architecture that will:")
        sections.append(f"1. Correctly classify sequences matching the '{pattern_context}' pattern as positive")
        sections.append("2. Preserve good performance on other patterns where possible")
        sections.append("3. Use the same layer structure and neuron counts")
        sections.append("")
        sections.append("Provide the complete model weights in the same format as they were given above:")
        sections.append("")
        
        return "\n".join(sections)
    
    def _serialize_model_weights(self, model: SubjectModel) -> str:
        sections = []
        sections.append("```")
        sections.append("MODEL_WEIGHTS")
        config = model.config
        sections.append(f"CONFIG: layers={config['num_layers']}, neurons={config['neurons_per_layer']}, activation={config['activation_type']}")
        sections.append("")
        
        # serialize each parameter
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_array = param.detach().cpu().numpy()
                sections.append(f"LAYER: {name}")
                sections.append(f"SHAPE: {list(weight_array.shape)}")
                
                # format weights based on dimensions
                if weight_array.ndim == 1:  # bias vectors
                    weight_str = " ".join([f"{w:.{self.precision}f}" for w in weight_array])
                    sections.append(f"VALUES: [{weight_str}]")
                elif weight_array.ndim == 2:  # weight matrices
                    sections.append("VALUES:")
                    for row in weight_array:
                        row_str = " ".join([f"{w:.{self.precision}f}" for w in row])
                        sections.append(f"  [{row_str}]")
                else:
                    # flatten higher dimensional arrays
                    flattened = weight_array.flatten()
                    sections.append(f"VALUES: {flattened.tolist()}")
                
                sections.append("")
        
        sections.append("```")
        return "\n".join(sections)
    
    def parse_model_weights(self, weight_text: str, model_config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self._parse_weight_text(weight_text, model_config)

    def _parse_weight_text(self, weight_text: str, model_config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        weights = {}
        lines = weight_text.split('\n')
        
        current_layer = None
        current_shape = None
        current_values = []
        in_values_section = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('LAYER:'):
                # save previous layer if exists
                if current_layer and current_shape:
                    weights[current_layer] = self._create_tensor_from_values(current_values, current_shape)
                
                # start new layer
                current_layer = line.split(':', 1)[1].strip()
                current_values = []
                in_values_section = False
                
            elif line.startswith('SHAPE:'):
                shape_str = line.split(':', 1)[1].strip()
                current_shape = json.loads(shape_str)
                
            elif line.startswith('VALUES:'):
                in_values_section = True
                # check if values are on same line
                values_part = line.split(':', 1)[1].strip()
                if values_part and not values_part == '':
                    # single line format
                    values = self._parse_values_line(values_part)
                    current_values.extend(values)
                
            elif in_values_section and line.startswith('[') and line.endswith(']'):
                # multi-line format
                values = self._parse_values_line(line)
                current_values.extend(values)
            
            elif line == '```' and current_layer and current_shape:
                weights[current_layer] = self._create_tensor_from_values(current_values, current_shape)
                break
        
        if current_layer and current_shape and current_layer not in weights:
            weights[current_layer] = self._create_tensor_from_values(current_values, current_shape)
        
        return weights
    
    def _parse_values_line(self, line: str) -> List[float]:
        """Parse a line containing numerical values."""
        line = line.strip('[]')
        if not line:
            return []
        values = []
        for value_str in line.split():
            try:
                values.append(float(value_str))
            except ValueError:
                logger.warning(f"Could not parse value: {value_str}")
                values.append(0.0)
        return values
    
    def _create_tensor_from_values(self, values: List[float], shape: List[int]) -> torch.Tensor:
        """Create tensor from values and shape."""
        expected_size = np.prod(shape)
        
        if len(values) != expected_size:
            logger.warning(f"Size mismatch: expected {expected_size}, got {len(values)}")
            if len(values) < expected_size:
                values.extend([0.0] * (expected_size - len(values)))
            else:
                values = values[:expected_size]
        tensor = torch.tensor(values, dtype=torch.float32)
        return tensor.reshape(shape)
    
    def load_weights_into_model(self, model: SubjectModel, weights_dict: Dict[str, torch.Tensor]) -> SubjectModel:

        try:
            model_state_dict = model.state_dict()
            for name, tensor in weights_dict.items():
                if name in model_state_dict:
                    if tensor.shape == model_state_dict[name].shape:
                        model_state_dict[name] = tensor
                    else:
                        logger.warning(f"Shape mismatch for {name}: "
                                     f"expected {model_state_dict[name].shape}, "
                                     f"got {tensor.shape}")
                else:
                    logger.warning(f"Unknown parameter: {name}")
            model.load_state_dict(model_state_dict)
            logger.info("Successfully loaded weights into model")
        except Exception as e:
            logger.error(f"Failed to load weights into model: {e}")
        
        return model
    
    def create_inference_prompt(self, input_model: SubjectModel, baseline_features: Dict[str, Any], pattern_context: str, pattern_description: str) -> str:
        return self._create_prompt(input_model, baseline_features, pattern_context, pattern_description)