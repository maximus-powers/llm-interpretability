import json
import torch
import torch.nn
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
    
    def __init__(self, precision: int = 6, format_style: str = 'separate'):
        self.precision = precision
        self.format_style = format_style
        if format_style not in ['separate', 'interwoven']:
            raise ValueError(f"format_style must be 'separate' or 'interwoven', got {format_style}")
        logger.info(f"Initialized TrainingDataFormatter with precision={precision}, format_style={format_style}")
    
    def create_training_example(self,
                              input_model: SubjectModel,
                              target_model: SubjectModel,
                              baseline_features: Dict[str, Any],
                              pattern_context: str,
                              pattern_description: str,
                              metadata: Dict[str, Any] = None) -> Dict[str, str]:
        if self.format_style == 'interwoven':
            prompt = self._create_interwoven_prompt(
                input_model, baseline_features, pattern_context, pattern_description
            )
        else:
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
        sections.append(f"Input Size: {config['input_size']} (integer indices for {config['input_size']} sequence positions, vocab size {config['vocab_size']})")
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

        # neuron profile signatures
        sections.append("## Neuron Profile Signatures")
        neuron_activations = baseline_features.get('neuron_activations', {})
        if neuron_activations:
            sections.append("Baseline profile values extracted for each neuron (values per neuron may vary by layer):")
            sections.append("")
            
            # create compact JSON structure
            neuron_profiles_json = {}
            for layer_name in sorted(neuron_activations.keys()):
                layer_data = neuron_activations[layer_name]
                if 'neuron_profiles' not in layer_data:
                    continue
                    
                neuron_profiles = layer_data['neuron_profiles']
                layer_values = []
                
                for neuron_id in sorted(neuron_profiles.keys()):
                    profile = neuron_profiles[neuron_id]
                    values = []
                    
                    # extract values in consistent order based on profile_methods
                    if 'layer_info' in layer_data and 'profile_methods' in layer_data['layer_info']:
                        methods_order = layer_data['layer_info']['profile_methods']
                    else:
                        methods_order = sorted(profile.keys())
                    
                    for method in methods_order:
                        if method in profile:
                            value = profile[method]
                            if isinstance(value, list):
                                values.extend(value)
                            else:
                                values.append(value)
                    
                    layer_values.append(values)
                
                neuron_profiles_json[layer_name] = layer_values
            
            sections.append(json.dumps(neuron_profiles_json, separators=(',', ':')))
            sections.append("")
        else:
            sections.append("No neuron profile data available")
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
    
    def _create_interwoven_prompt(self,
                                 input_model: SubjectModel,
                                 baseline_features: Dict[str, Any],
                                 pattern_context: str,
                                 pattern_description: str) -> str:
        sections = []
        
        # header
        sections.append("# Neural Network Weight Modification Task\n")
        sections.append("You are an expert neural network interpreter. Your task is to analyze the given model weights and baseline features, then generate improved weights that will make the model correctly classify the specified pattern\n")
        
        # pattern context
        sections.append("## Target Pattern")
        sections.append(f"Pattern Name: {pattern_context}\n Description: {pattern_description}\n")
        sections.append("The model should classify sequences matching this pattern as POSITIVE (label=1).\n")
        
        # model architecture
        config = input_model.config
        sections.append("## Model Architecture")
        sections.append(f"Input Size: {config['input_size']} (integer indices for {config['input_size']} sequence positions, vocab size {config['vocab_size']})")
        sections.append(f"Hidden Layers: {config['num_layers']}")
        sections.append(f"Neurons per Layer: {config['neurons_per_layer']}")
        sections.append(f"Activation Function: {config['activation_type']}")
        sections.append(f"Dropout Rate: {config['dropout_rate']}")
        sections.append("")
        
        # interwoven weights and signatures
        sections.append("## Model Weights and Neuron Signatures")
        sections.append("Layer weights, biases, and activation signatures presented together:")
        sections.append("")
        sections.append(self._serialize_interwoven_model_and_signatures(input_model, baseline_features))
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
    
    def _map_layer_names(self, model: SubjectModel) -> Dict[str, str]:
        layer_mapping = {}
        linear_layer_count = 0
        
        for name, module in model.network.named_modules():
            if isinstance(module, torch.nn.Linear):
                signature_layer_name = f"hidden_{linear_layer_count}"
                layer_mapping[name] = signature_layer_name
                linear_layer_count += 1
        
        return layer_mapping
    
    def _serialize_interwoven_model_and_signatures(self, model: SubjectModel, baseline_features: Dict[str, Any]) -> str:
        sections = []
        sections.append("```")
        sections.append("MODEL_WEIGHTS_AND_SIGNATURES")
        config = model.config
        sections.append(f"CONFIG: layers={config['num_layers']}, neurons={config['neurons_per_layer']}, activation={config['activation_type']}")
        sections.append("")
        
        neuron_activations = baseline_features.get('neuron_activations', {})
        layer_mapping = self._map_layer_names(model)
        
        layer_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_name = name.split('.')[0]
                if layer_name not in layer_params:
                    layer_params[layer_name] = {}
                
                param_type = name.split('.')[1]  # "weight" or "bias"
                layer_params[layer_name][param_type] = param
        
        for layer_name in sorted(layer_params.keys(), key=int):
            layer_data = layer_params[layer_name]
            signature_layer_name = layer_mapping.get(layer_name, f"hidden_{layer_name}")
            
            sections.append(f"LAYER: {layer_name} ({signature_layer_name})")
            sections.append("")
            
            if 'weight' in layer_data:
                weight_array = layer_data['weight'].detach().cpu().numpy()
                sections.append(f"WEIGHTS_SHAPE: {list(weight_array.shape)}")
                sections.append("WEIGHTS_VALUES:")
                for row in weight_array:
                    row_str = " ".join([f"{w:.{self.precision}f}" for w in row])
                    sections.append(f"  [{row_str}]")
                sections.append("")
            
            if 'bias' in layer_data:
                bias_array = layer_data['bias'].detach().cpu().numpy()
                sections.append(f"BIAS_SHAPE: {list(bias_array.shape)}")
                bias_str = " ".join([f"{b:.{self.precision}f}" for b in bias_array])
                sections.append(f"BIAS_VALUES: [{bias_str}]")
                sections.append("")
            
            if signature_layer_name in neuron_activations:
                layer_signature_data = neuron_activations[signature_layer_name]
                if 'neuron_profiles' in layer_signature_data:
                    neuron_profiles = layer_signature_data['neuron_profiles']
                    layer_values = []
                    
                    for neuron_id in sorted(neuron_profiles.keys()):
                        profile = neuron_profiles[neuron_id]
                        values = []
                        
                        if 'layer_info' in layer_signature_data and 'profile_methods' in layer_signature_data['layer_info']:
                            methods_order = layer_signature_data['layer_info']['profile_methods']
                        else:
                            methods_order = sorted(profile.keys())
                        
                        for method in methods_order:
                            if method in profile:
                                value = profile[method]
                                if isinstance(value, list):
                                    values.extend(value)
                                else:
                                    values.append(value)
                        
                        layer_values.append(values)
                    
                    sections.append(f"NEURON_SIGNATURES: {json.dumps(layer_values, separators=(',', ':'))}")
                    sections.append("")
                else:
                    sections.append("NEURON_SIGNATURES: []")
                    sections.append("")
            else:
                sections.append("NEURON_SIGNATURES: []")
                sections.append("")
        
        sections.append("```")
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