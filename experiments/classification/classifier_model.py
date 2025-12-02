import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class PatternClassifierMLP(nn.Module):
    def __init__(self, config: Dict[str, Any], input_dims: Dict[str, int]):
        super().__init__()
        self.input_mode = config['dataset']['input_mode']
        self.config = config
        self.use_batch_norm = config.get('model', {}).get('use_batch_norm', False)
        logger.info(f"Initializing PatternClassifierMLP with input_mode={self.input_mode}, batch_norm={self.use_batch_norm}")

        sig_out = 0
        weight_out = 0

        # signature encoding layers
        if self.input_mode in ["signature", "both"]:
            if 'signature' not in input_dims:
                raise ValueError(f"Input mode '{self.input_mode}' requires 'signature' in input_dims")
            sig_hidden_dims = config['model']['signature_encoder']['hidden_dims']
            self.signature_encoder = self._build_mlp(
                input_dims['signature'],
                sig_hidden_dims,
                config['model']['signature_encoder']['dropout'],
                config['model']['signature_encoder']['activation']
            )
            sig_out = sig_hidden_dims[-1]
            logger.info(f"Signature encoder ({len(sig_hidden_dims)} layers): {input_dims['signature']} -> {' -> '.join(map(str, sig_hidden_dims))}")

        # weights encoding layers
        if self.input_mode in ["weights", "both"]:
            if 'weights' not in input_dims:
                raise ValueError(f"Input mode '{self.input_mode}' requires 'weights' in input_dims")
            weight_hidden_dims = config['model']['weight_encoder']['hidden_dims']
            self.weight_encoder = self._build_mlp(
                input_dims['weights'],
                weight_hidden_dims,
                config['model']['weight_encoder']['dropout'],
                config['model']['weight_encoder']['activation']
            )
            weight_out = weight_hidden_dims[-1]
            logger.info(f"Weight encoder ({len(weight_hidden_dims)} layers): {input_dims['weights']} -> {' -> '.join(map(str, weight_hidden_dims))}")

        # fusion layers
        fusion_in = sig_out + weight_out
        fusion_hidden_dims = config['model']['fusion']['hidden_dims']
        self.fusion = self._build_mlp(
            fusion_in,
            fusion_hidden_dims,
            config['model']['fusion']['dropout'],
            config['model']['fusion']['activation']
        )
        fusion_out = fusion_hidden_dims[-1]

        # output layer
        num_patterns = config['model']['output']['num_patterns']
        self.output_layer = nn.Linear(fusion_out, num_patterns)

        logger.info(f"Fusion ({len(fusion_hidden_dims)} layers): {fusion_in} -> {' -> '.join(map(str, fusion_hidden_dims))} -> {num_patterns} patterns")

        self._initialize_weights()
        logger.info("Initialized layers")

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # he or xavier init
                if hasattr(self, 'config'):
                    activation = self.config.get('model', {}).get('signature_encoder', {}).get('activation', 'relu').lower()
                    if activation in ['relu', 'leaky_relu']:
                        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    else:
                        nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.xavier_normal_(module.weight)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _build_mlp(self, input_dim: int, hidden_dims: List[int], dropout: float, activation: str) -> nn.Sequential:
        layers = []
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid()
        }
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Add batch normalization if enabled
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(activations.get(activation.lower(), nn.ReLU()))
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim
        return nn.Sequential(*layers)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = []
        # encode inputs
        if self.input_mode in ["signature", "both"]:
            if 'signature' not in inputs:
                raise ValueError(f"Input mode '{self.input_mode}' requires 'signature' in inputs")
            
            encoded.append(self.signature_encoder(inputs['signature']))
        if self.input_mode in ["weights", "both"]:
            if 'weights' not in inputs:
                raise ValueError(f"Input mode '{self.input_mode}' requires 'weights' in inputs")
            encoded.append(self.weight_encoder(inputs['weights']))

        # concat encodings, fuse, output
        if len(encoded) > 1:
            combined = torch.cat(encoded, dim=1)
        else:
            combined = encoded[0]
        fused = self.fusion(combined)
        logits = self.output_layer(fused)

        return logits  # [batch_size, num_patterns]