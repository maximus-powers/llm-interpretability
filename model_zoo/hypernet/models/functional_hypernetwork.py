"""
Functional HyperNetwork for Behavioral Weight Generation

This module implements a Conditional VAE that generates neural network weights
conditioned on behavioral signatures. The key insight is using FUNCTIONAL loss
(does the generated network behave correctly?) rather than just weight reconstruction.

Architecture:
    Signature Encoder: behavioral signatures → conditioning vector
    Weight Encoder: (weights, condition) → latent z
    Weight Decoder: (z, condition) → weights

Usage:
    # Create model
    model = FunctionalHyperNetwork(weight_dim=345, sig_dim=510)
    
    # Train with functional loss
    model.fit(weights, signatures, labels, use_functional_loss=True)
    
    # Generate weights for a behavior
    new_weights = model.generate(target_signature)
    
    # Edit behavior
    editor = BehaviorEditor(model)
    edited_weights = editor.edit(original_weights, original_sig, target_sig)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


FOCUSED_BEHAVIOR_CASES = {
    "sorted_descending": {
        "positive": [
            [9, 7, 5, 3, 1],
            [8, 6, 4, 2, 0],
            [7, 5, 3, 2, 1],
            [9, 8, 7, 6, 5],
            [5, 4, 3, 2, 1],
        ],
        "negative": [
            [1, 3, 5, 7, 9],
            [0, 2, 4, 6, 8],
            [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 9],
            [3, 1, 4, 1, 5],
        ],
    },
    "sorted_ascending": {
        "positive": [
            [1, 3, 5, 7, 9],
            [0, 2, 4, 6, 8],
            [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4],
        ],
        "negative": [
            [9, 7, 5, 3, 1],
            [8, 6, 4, 2, 0],
            [7, 5, 3, 2, 1],
            [9, 8, 7, 6, 5],
            [3, 1, 4, 1, 5],
        ],
    },
    "increasing_pairs": {
        "positive": [
            [1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4],
            [2, 3, 4, 5, 6],
            [1, 3, 5, 7, 9],
            [0, 2, 4, 6, 8],
        ],
        "negative": [
            [5, 4, 3, 2, 1],
            [2, 1, 4, 3, 5],
            [1, 1, 1, 1, 1],
            [5, 3, 4, 2, 1],
            [9, 7, 5, 3, 1],
        ],
    },
    "decreasing_pairs": {
        "positive": [
            [5, 4, 3, 2, 1],
            [9, 7, 5, 3, 1],
            [8, 6, 4, 2, 0],
            [6, 5, 4, 3, 2],
            [7, 5, 3, 1, 0],
        ],
        "negative": [
            [1, 2, 3, 4, 5],
            [1, 3, 5, 7, 9],
            [0, 2, 4, 6, 8],
            [1, 1, 1, 1, 1],
            [2, 4, 3, 5, 1],
        ],
    },
}
DEFAULT_PATTERN_TO_IDX = {
    "contains_abc": 0,
    "palindrome": 1,
    "alternating": 2,
    "sorted_descending": 3,
    "sorted_ascending": 4,
    "ends_with": 5,
    "decreasing_pairs": 6,
    "mountain_pattern": 7,
    "increasing_pairs": 8,
    "first_last_match": 9,
    "starts_with": 10,
    "no_repeats": 11,
    "has_majority": 12,
    "vowel_consonant": 13,
}


@dataclass
class HyperNetConfig:
    """Configuration for FunctionalHyperNetwork."""
    weight_dim: int = 345
    sig_dim: int = 510
    latent_dim: int = 128  # Increased from 64 - more capacity for weight info
    condition_dim: int = 128
    hidden_dim: int = 512  # Increased from 256 - more capacity
    dropout: float = 0.1
    
    # Training
    epochs: int = 150
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    lambda_kl: float = 0.01  # Reduced from 0.1 - less regularization pressure
    lambda_functional: float = 10.0  # Increased from 0.5 - prioritize functional behavior!
    lambda_condition_specificity: float = 1.0
    lambda_calibrated_behavior_margin: float = 0.0
    matched_behavior_min_margin: float = 0.02
    matched_mountain_target_weight: float = 1.0
    lambda_control_behavior_penalty: float = 1.0
    lambda_control_hard_negative_penalty: float = 0.0
    control_max_allowed_margin: float = -0.05
    train_centroid_control_weight: float = 3.0
    condition_ablation_control_weight: float = 1.0
    noise_control_weight: float = 1.0
    shuffled_control_weight: float = 1.0
    control_sorted_descending_target_weight: float = 1.0
    control_has_majority_target_weight: float = 1.0
    sorted_descending_specificity_weight: float = 2.0
    lambda_edit_behavior: float = 1.0
    lambda_edit_margin_delta: float = 1.0
    use_condition_residual_decoder: bool = False
    condition_residual_scale: float = 1.0
    lambda_shuffled_residual_contrastive: float = 0.0
    shuffled_residual_min_delta: float = 0.05
    functional_loss_start_epoch: int = 0  # Start immediately, not at epoch 50
    functional_loss_samples: int = 16  # Increased from 8 - more samples per batch
    
    # Subject network architecture
    num_layers: int = 5
    neurons_per_layer: int = 8
    input_dim: int = 5


class SubjectNetwork(nn.Module):
    """
    The target network architecture whose weights we generate.
    
    Matches the dataset_generation SubjectModel architecture exactly:
    - Input layer: input_dim -> neurons_per_layer
    - Hidden layers: (num_layers - 1) layers of neurons_per_layer -> neurons_per_layer
    - Output layer: neurons_per_layer -> 1
    
    Total Linear layers = num_layers + 1
    """
    
    def __init__(
        self,
        num_layers: int = 5,
        neurons_per_layer: int = 8,
        input_dim: int = 5,
        activation_type: str = 'gelu',
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.input_dim = input_dim
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate
        
        # Match dataset_generation's SubjectModel exactly
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
        }
        activation = activations.get(activation_type.lower(), nn.GELU())
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, neurons_per_layer))
        layers.append(activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(neurons_per_layer, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_activations(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Get activations at each layer (after activation function)."""
        activations = {}
        current = x
        layer_idx = 0
        
        for module in self.network:
            current = module(current)
            # Capture after activation (not dropout or linear alone)
            if isinstance(module, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):
                activations[layer_idx] = current.clone()
                layer_idx += 1
        
        return activations
    
    def to_flat(self) -> torch.Tensor:
        """Extract weights as flat tensor."""
        params = []
        for p in self.parameters():
            params.append(p.data.view(-1))
        return torch.cat(params)
    
    def from_flat(self, flat: torch.Tensor):
        """Load flat tensor into model parameters."""
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = flat[idx:idx + numel].view(p.shape)
            idx += numel
    
    @classmethod
    def from_weights(
        cls,
        flat_weights: torch.Tensor,
        num_layers: int = 5,
        neurons_per_layer: int = 8,
        input_dim: int = 5,
        activation_type: str = 'gelu',
        dropout_rate: float = 0.0,
    ) -> 'SubjectNetwork':
        """Create network with given weights."""
        model = cls(
            num_layers=num_layers,
            neurons_per_layer=neurons_per_layer,
            input_dim=input_dim,
            activation_type=activation_type,
            dropout_rate=dropout_rate,
        )
        model.from_flat(flat_weights)
        return model
    
    @classmethod 
    def from_config(cls, config: Dict) -> 'SubjectNetwork':
        """Create network from a config dict (as stored in dataset)."""
        return cls(
            num_layers=config['num_layers'],
            neurons_per_layer=config['neurons_per_layer'],
            input_dim=config.get('input_size', config.get('sequence_length', 5)),
            activation_type=config.get('activation_type', 'gelu'),
            dropout_rate=config.get('dropout_rate', 0.0),
        )


class SignatureEncoder(nn.Module):
    """Encode behavioral signatures into conditioning vectors."""
    
    def __init__(
        self,
        sig_dim: int = 510,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sig_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, signature: torch.Tensor) -> torch.Tensor:
        return self.net(signature)


class WeightEncoder(nn.Module):
    """Encode weights + condition to latent distribution."""
    
    def __init__(
        self,
        weight_dim: int,
        condition_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(weight_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(
        self,
        weights: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([weights, condition], dim=-1)
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class WeightDecoder(nn.Module):
    """Decode latent + condition to weights."""
    
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        weight_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_condition_residual_decoder: bool = False,
        condition_residual_scale: float = 1.0,
    ):
        super().__init__()
        self.use_condition_residual_decoder = use_condition_residual_decoder
        self.condition_residual_scale = condition_residual_scale
        if use_condition_residual_decoder:
            self.base_net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, weight_dim),
            )
            self.residual_net = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, weight_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_dim + condition_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, weight_dim),
            )
    
    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        condition_baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_condition_residual_decoder:
            if condition_baseline is None:
                condition_baseline = torch.zeros_like(condition)
            residual_condition = (
                condition - condition_baseline
            ) * self.condition_residual_scale
            zero_residual = torch.zeros_like(residual_condition)
            condition_delta = (
                self.residual_net(residual_condition)
                - self.residual_net(zero_residual)
            )
            return self.base_net(z) + condition_delta

        x = torch.cat([z, condition], dim=-1)
        return self.net(x)


class FunctionalHyperNetwork(nn.Module):
    """
    Conditional VAE for generating neural network weights.
    
    Given a behavioral signature, generates weights that exhibit
    that behavior. Supports behavior editing by modifying the
    conditioning signal.
    """
    
    def __init__(self, config: Optional[HyperNetConfig] = None, **kwargs):
        super().__init__()
        
        if config is None:
            config = HyperNetConfig(**kwargs)
        self.config = config
        
        # Components
        self.sig_encoder = SignatureEncoder(
            sig_dim=config.sig_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.condition_dim,
            dropout=config.dropout,
        )
        
        self.weight_encoder = WeightEncoder(
            weight_dim=config.weight_dim,
            condition_dim=config.condition_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        
        self.weight_decoder = WeightDecoder(
            latent_dim=config.latent_dim,
            condition_dim=config.condition_dim,
            weight_dim=config.weight_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            use_condition_residual_decoder=config.use_condition_residual_decoder,
            condition_residual_scale=config.condition_residual_scale,
        )
        
        # Normalization stats (set during training)
        self.register_buffer('weight_mean', torch.zeros(config.weight_dim))
        self.register_buffer('weight_std', torch.ones(config.weight_dim))
        self.register_buffer('sig_mean', torch.zeros(config.sig_dim))
        self.register_buffer('sig_std', torch.ones(config.sig_dim))
        
        # Fixed digit-domain probes for functional loss.
        probe_generator = torch.Generator().manual_seed(0)
        self.register_buffer(
            '_probe_inputs',
            torch.randint(
                low=0,
                high=10,
                size=(50, config.input_dim),
                generator=probe_generator,
                dtype=torch.float32,
            ),
        )
    
    def encode_signature(self, signature: torch.Tensor) -> torch.Tensor:
        """Encode behavioral signature to conditioning vector."""
        sig_norm = (signature - self.sig_mean) / self.sig_std
        return self.sig_encoder(sig_norm)
    
    def encode_weights(
        self,
        weights: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode weights to latent distribution (mu, logvar)."""
        weights_norm = (weights - self.weight_mean) / self.weight_std
        return self.weight_encoder(weights_norm, condition)
    
    def decode_weights(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        condition_baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode latent + condition to weights."""
        weights_norm = self.weight_decoder(z, condition, condition_baseline)
        return weights_norm * self.weight_std + self.weight_mean
    
    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        weights: torch.Tensor,
        signature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Returns:
            recon_weights: Reconstructed weights
            mu: Latent mean
            logvar: Latent log variance
            condition: Conditioning vector
        """
        # Normalize inputs
        weights_norm = (weights - self.weight_mean) / self.weight_std
        sig_norm = (signature - self.sig_mean) / self.sig_std
        
        # Encode
        condition = self.sig_encoder(sig_norm)
        mu, logvar = self.weight_encoder(weights_norm, condition)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_norm = self.weight_decoder(z, condition)
        recon_weights = recon_norm * self.weight_std + self.weight_mean
        
        return recon_weights, mu, logvar, condition
    
    def generate(
        self,
        signature: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate weights from a behavioral signature.
        
        Args:
            signature: Behavioral signature [batch, sig_dim] or [sig_dim]
            n_samples: Number of samples per signature
        
        Returns:
            Generated weights [batch * n_samples, weight_dim]
        """
        if signature.dim() == 1:
            signature = signature.unsqueeze(0)
        
        sig_norm = (signature - self.sig_mean) / self.sig_std
        condition = self.sig_encoder(sig_norm)
        
        # Sample from prior
        batch_size = signature.size(0)
        z = torch.randn(
            batch_size * n_samples,
            self.config.latent_dim,
            device=signature.device,
        )
        condition = condition.repeat_interleave(n_samples, dim=0)
        
        # Decode
        weights_norm = self.weight_decoder(z, condition)
        return weights_norm * self.weight_std + self.weight_mean

    def subject_forward_from_flat(
        self,
        flat_weights: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Run SubjectNetwork forward pass directly from flat weights.

        This mirrors SubjectNetwork without assigning tensors into nn.Parameters,
        so gradients from functional losses flow back to generated weights.
        """
        if flat_weights.dim() == 1:
            flat_weights = flat_weights.unsqueeze(0)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0).expand(flat_weights.size(0), -1, -1)
        elif inputs.dim() != 3:
            raise ValueError("inputs must have shape [n, d] or [batch, n, d]")

        batch_size = flat_weights.size(0)
        x = inputs.to(flat_weights.device)
        offset = 0

        for layer_idx in range(self.config.num_layers):
            in_dim = self.config.input_dim if layer_idx == 0 else self.config.neurons_per_layer
            out_dim = self.config.neurons_per_layer
            weight_count = out_dim * in_dim

            weight = flat_weights[:, offset:offset + weight_count]
            weight = weight.view(batch_size, out_dim, in_dim)
            offset += weight_count

            bias = flat_weights[:, offset:offset + out_dim]
            offset += out_dim

            x = torch.bmm(x, weight.transpose(1, 2)) + bias.unsqueeze(1)
            x = F.gelu(x)

        out_dim = 1
        in_dim = self.config.neurons_per_layer
        weight_count = out_dim * in_dim
        weight = flat_weights[:, offset:offset + weight_count]
        weight = weight.view(batch_size, out_dim, in_dim)
        offset += weight_count

        bias = flat_weights[:, offset:offset + out_dim]
        offset += out_dim

        if offset != flat_weights.size(1):
            raise ValueError(
                f"Expected {offset} flat weights, got {flat_weights.size(1)}"
            )

        logits = torch.bmm(x, weight.transpose(1, 2)) + bias.unsqueeze(1)
        return logits.squeeze(-1)
    
    def compute_functional_loss(
        self,
        original_weights: torch.Tensor,
        reconstructed_weights: torch.Tensor,
        n_probes: int = 30,
        margin_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute functional loss: do reconstructed weights produce same behavior?
        
        This compares the outputs of networks with original vs reconstructed weights
        on a fixed set of probe inputs, PLUS a margin preservation term.
        """
        batch_size = original_weights.size(0)
        probes = self._probe_inputs[:n_probes].to(original_weights.device)

        with torch.no_grad():
            orig_out = self.subject_forward_from_flat(original_weights.detach(), probes)
            orig_sorted_idx = torch.argsort(orig_out, dim=1)
            low_idx = orig_sorted_idx[:, :5]
            high_idx = orig_sorted_idx[:, -5:]
            orig_low = torch.gather(orig_out, 1, low_idx).mean(dim=1)
            orig_high = torch.gather(orig_out, 1, high_idx).mean(dim=1)
            orig_margin = orig_high - orig_low

        recon_out = self.subject_forward_from_flat(reconstructed_weights, probes)
        mse_loss = F.mse_loss(recon_out, orig_out)

        recon_low = torch.gather(recon_out, 1, low_idx).mean(dim=1)
        recon_high = torch.gather(recon_out, 1, high_idx).mean(dim=1)
        recon_margin = recon_high - recon_low
        margin_loss = F.relu(orig_margin - recon_margin).mean()

        return mse_loss + margin_weight * margin_loss

    def compute_target_behavior_loss(
        self,
        generated_weights: torch.Tensor,
        labels: torch.Tensor,
        pattern_to_idx: Dict[str, int],
        behavior_cases: Optional[Dict[str, Dict[str, List[List[float]]]]] = None,
        target_margin: float = 1.0,
    ) -> torch.Tensor:
        """Train decoded weights to satisfy their labeled behavior cases."""
        if generated_weights.dim() == 1:
            generated_weights = generated_weights.unsqueeze(0)

        cases_by_pattern = behavior_cases or FOCUSED_BEHAVIOR_CASES
        labels = labels.to(generated_weights.device)
        total_loss = generated_weights.sum() * 0.0
        n_patterns = 0

        for pattern, cases in cases_by_pattern.items():
            pattern_idx = pattern_to_idx.get(pattern)
            if pattern_idx is None:
                continue

            mask = labels == pattern_idx
            if not bool(mask.any()):
                continue

            weights_for_pattern = generated_weights[mask]
            pos_inputs = torch.tensor(
                cases["positive"],
                dtype=generated_weights.dtype,
                device=generated_weights.device,
            )
            neg_inputs = torch.tensor(
                cases["negative"],
                dtype=generated_weights.dtype,
                device=generated_weights.device,
            )

            pos_logits = self.subject_forward_from_flat(weights_for_pattern, pos_inputs)
            neg_logits = self.subject_forward_from_flat(weights_for_pattern, neg_inputs)

            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits,
                torch.ones_like(pos_logits),
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits,
                torch.zeros_like(neg_logits),
            )
            margin = pos_logits.mean(dim=1) - neg_logits.mean(dim=1)
            margin_loss = F.relu(target_margin - margin).mean()

            total_loss = total_loss + pos_loss + neg_loss + margin_loss
            n_patterns += 1

        if n_patterns == 0:
            return total_loss

        return total_loss / n_patterns

    def compute_condition_functional_specificity_loss(
        self,
        original_weights: torch.Tensor,
        matched_condition: torch.Tensor,
        wrong_condition: Optional[torch.Tensor] = None,
        control_conditions: Optional[List[torch.Tensor]] = None,
        condition_baseline: Optional[torch.Tensor] = None,
        probe_inputs: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        n_probes: int = 30,
        contrastive_margin: float = 0.05,
    ) -> torch.Tensor:
        """Make condition-only decoding match the source subject better than controls."""
        if original_weights.dim() == 1:
            original_weights = original_weights.unsqueeze(0)
        if probe_inputs is None:
            probes = self._probe_inputs[:n_probes].to(original_weights.device)
        else:
            probes = probe_inputs.to(original_weights.device)
        zero_latent = torch.zeros(
            matched_condition.size(0),
            self.config.latent_dim,
            device=matched_condition.device,
            dtype=matched_condition.dtype,
        )

        with torch.no_grad():
            reference_outputs = self.subject_forward_from_flat(
                original_weights.detach(),
                probes,
            )

        matched_weights = self.decode_weights(
            zero_latent,
            matched_condition,
            condition_baseline=condition_baseline,
        )
        matched_outputs = self.subject_forward_from_flat(matched_weights, probes)
        matched_mse_per_sample = F.mse_loss(
            matched_outputs,
            reference_outputs,
            reduction="none",
        ).mean(dim=1)
        loss_weights = self._normalize_loss_weights(
            sample_weights,
            matched_mse_per_sample,
        )
        loss = (matched_mse_per_sample * loss_weights).mean()

        controls = []
        if wrong_condition is not None:
            controls.append(wrong_condition)
        if control_conditions:
            controls.extend(control_conditions)

        control_mses = []
        for control_condition in controls:
            if control_condition.size(0) != matched_condition.size(0):
                continue
            control_weights = self.decode_weights(
                zero_latent,
                control_condition,
                condition_baseline=condition_baseline,
            )
            control_outputs = self.subject_forward_from_flat(control_weights, probes)
            control_mse_per_sample = F.mse_loss(
                control_outputs,
                reference_outputs,
                reduction="none",
            ).mean(dim=1)
            control_mses.append(control_mse_per_sample)

        if control_mses:
            best_control_mse = torch.stack(control_mses, dim=0).min(dim=0).values
            contrastive_loss = F.relu(
                contrastive_margin + matched_mse_per_sample - best_control_mse
            )
            contrastive_loss = (contrastive_loss * loss_weights).mean()
            loss = loss + contrastive_loss

        return loss

    def _normalize_loss_weights(
        self,
        sample_weights: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if sample_weights is None:
            return torch.ones_like(reference)
        weights = sample_weights.to(device=reference.device, dtype=reference.dtype)
        if weights.dim() != 1 or weights.size(0) != reference.size(0):
            raise ValueError("sample_weights must have shape [batch]")
        return weights / weights.mean().clamp(min=1e-6)

    def build_subject_specificity_sample_weights(
        self,
        labels: torch.Tensor,
        pattern_to_idx: Dict[str, int],
    ) -> torch.Tensor:
        """Build per-sample weights for behavior-specific specificity pressure."""
        weights = torch.ones(labels.size(0), dtype=torch.float32, device=labels.device)
        descending_idx = pattern_to_idx.get("sorted_descending")
        if descending_idx is not None and self.config.sorted_descending_specificity_weight != 1.0:
            weights = torch.where(
                labels.to(labels.device) == descending_idx,
                torch.full_like(weights, self.config.sorted_descending_specificity_weight),
                weights,
            )
        return weights

    def build_subject_specificity_probes(
        self,
        labels: torch.Tensor,
        pattern_to_idx: Dict[str, int],
        behavior_cases: Optional[Dict[str, Dict[str, List[List[float]]]]] = None,
    ) -> Optional[torch.Tensor]:
        """Build per-sample behavior-case probes for subject specificity."""
        if behavior_cases is None:
            return None

        idx_to_pattern = {idx: pattern for pattern, idx in pattern_to_idx.items()}
        rows = []
        expected_count = None
        for label in labels.detach().cpu().tolist():
            pattern = idx_to_pattern.get(int(label))
            cases = behavior_cases.get(pattern) if pattern else None
            if cases is None:
                return None
            case_rows = cases["positive"] + cases["negative"]
            if expected_count is None:
                expected_count = len(case_rows)
            if len(case_rows) != expected_count:
                return None
            rows.append(case_rows)

        if not rows:
            return None

        return torch.tensor(
            rows,
            dtype=torch.float32,
            device=labels.device,
        )

    def compute_behavior_prior_penalty(
        self,
        generated_weights: torch.Tensor,
        labels: torch.Tensor,
        pattern_to_idx: Dict[str, int],
        behavior_cases: Optional[Dict[str, Dict[str, List[List[float]]]]] = None,
        max_allowed_margin: float = -0.05,
    ) -> torch.Tensor:
        """Penalize control decodes that solve the labeled behavior too well."""
        if generated_weights.dim() == 1:
            generated_weights = generated_weights.unsqueeze(0)

        cases_by_pattern = behavior_cases or FOCUSED_BEHAVIOR_CASES
        labels = labels.to(generated_weights.device)
        total_loss = generated_weights.sum() * 0.0
        n_patterns = 0

        for pattern, cases in cases_by_pattern.items():
            pattern_idx = pattern_to_idx.get(pattern)
            if pattern_idx is None:
                continue

            mask = labels == pattern_idx
            if not bool(mask.any()):
                continue

            weights_for_pattern = generated_weights[mask]
            pos_inputs = torch.tensor(
                cases["positive"],
                dtype=generated_weights.dtype,
                device=generated_weights.device,
            )
            neg_inputs = torch.tensor(
                cases["negative"],
                dtype=generated_weights.dtype,
                device=generated_weights.device,
            )

            pos_prob = torch.sigmoid(
                self.subject_forward_from_flat(weights_for_pattern, pos_inputs)
            )
            neg_prob = torch.sigmoid(
                self.subject_forward_from_flat(weights_for_pattern, neg_inputs)
            )
            margin = pos_prob.mean(dim=1) - neg_prob.mean(dim=1)
            total_loss = total_loss + F.relu(margin - max_allowed_margin).mean()
            n_patterns += 1

        if n_patterns == 0:
            return total_loss

        return total_loss / n_patterns

    def compute_calibrated_behavior_margin_loss(
        self,
        generated_weights: torch.Tensor,
        labels: torch.Tensor,
        pattern_to_idx: Dict[str, int],
        behavior_cases: Optional[Dict[str, Dict[str, List[List[float]]]]] = None,
        min_margin: float = 0.02,
        target_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Train matched decodes against calibrated sigmoid behavior margins."""
        if generated_weights.dim() == 1:
            generated_weights = generated_weights.unsqueeze(0)

        cases_by_pattern = behavior_cases or FOCUSED_BEHAVIOR_CASES
        labels = labels.to(generated_weights.device)
        target_weights = target_weights or {}
        total_loss = generated_weights.sum() * 0.0
        n_patterns = 0

        for pattern, cases in cases_by_pattern.items():
            pattern_idx = pattern_to_idx.get(pattern)
            if pattern_idx is None:
                continue

            mask = labels == pattern_idx
            if not bool(mask.any()):
                continue

            weights_for_pattern = generated_weights[mask]
            pos_inputs = torch.tensor(
                cases["positive"],
                dtype=generated_weights.dtype,
                device=generated_weights.device,
            )
            neg_inputs = torch.tensor(
                cases["negative"],
                dtype=generated_weights.dtype,
                device=generated_weights.device,
            )
            pos_prob = torch.sigmoid(
                self.subject_forward_from_flat(weights_for_pattern, pos_inputs)
            )
            neg_prob = torch.sigmoid(
                self.subject_forward_from_flat(weights_for_pattern, neg_inputs)
            )
            margin = pos_prob.mean(dim=1) - neg_prob.mean(dim=1)
            target_weight = float(target_weights.get(pattern, 1.0))
            total_loss = total_loss + F.relu(min_margin - margin).mean() * target_weight
            n_patterns += 1

        if n_patterns == 0:
            return total_loss

        return total_loss / n_patterns

    def compute_matched_control_behavior_margin_loss(
        self,
        matched_weights: torch.Tensor,
        control_weights: torch.Tensor,
        labels: torch.Tensor,
        pattern_to_idx: Dict[str, int],
        behavior_cases: Optional[Dict[str, Dict[str, List[List[float]]]]] = None,
        min_delta: float = 0.05,
        target_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Require matched decodes to beat shuffled controls on source behavior."""
        if matched_weights.dim() == 1:
            matched_weights = matched_weights.unsqueeze(0)
        if control_weights.dim() == 1:
            control_weights = control_weights.unsqueeze(0)

        cases_by_pattern = behavior_cases or FOCUSED_BEHAVIOR_CASES
        labels = labels.to(matched_weights.device)
        target_weights = target_weights or {}
        total_loss = matched_weights.sum() * 0.0 + control_weights.sum() * 0.0
        n_patterns = 0

        for pattern, cases in cases_by_pattern.items():
            pattern_idx = pattern_to_idx.get(pattern)
            if pattern_idx is None:
                continue

            mask = labels == pattern_idx
            if not bool(mask.any()):
                continue

            matched_for_pattern = matched_weights[mask]
            control_for_pattern = control_weights[mask]
            pos_inputs = torch.tensor(
                cases["positive"],
                dtype=matched_weights.dtype,
                device=matched_weights.device,
            )
            neg_inputs = torch.tensor(
                cases["negative"],
                dtype=matched_weights.dtype,
                device=matched_weights.device,
            )
            matched_pos = torch.sigmoid(
                self.subject_forward_from_flat(matched_for_pattern, pos_inputs)
            )
            matched_neg = torch.sigmoid(
                self.subject_forward_from_flat(matched_for_pattern, neg_inputs)
            )
            control_pos = torch.sigmoid(
                self.subject_forward_from_flat(control_for_pattern, pos_inputs)
            )
            control_neg = torch.sigmoid(
                self.subject_forward_from_flat(control_for_pattern, neg_inputs)
            )
            matched_margin = matched_pos.mean(dim=1) - matched_neg.mean(dim=1)
            control_margin = control_pos.mean(dim=1) - control_neg.mean(dim=1)
            target_weight = float(target_weights.get(pattern, 1.0))
            total_loss = (
                total_loss
                + F.relu(min_delta - (matched_margin - control_margin)).mean()
                * target_weight
            )
            n_patterns += 1

        if n_patterns == 0:
            return total_loss

        return total_loss / n_patterns

    def compute_all_target_control_penalty(
        self,
        generated_weights: torch.Tensor,
        pattern_to_idx: Dict[str, int],
        behavior_cases: Optional[Dict[str, Dict[str, List[List[float]]]]] = None,
        max_allowed_margin: float = -0.05,
        target_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Penalize control decodes that solve any clean behavior target."""
        if generated_weights.dim() == 1:
            generated_weights = generated_weights.unsqueeze(0)

        cases_by_pattern = behavior_cases or FOCUSED_BEHAVIOR_CASES
        target_weights = target_weights or {}
        total_loss = generated_weights.sum() * 0.0
        n_patterns = 0

        for pattern, cases in cases_by_pattern.items():
            if pattern not in pattern_to_idx:
                continue

            pos_inputs = torch.tensor(
                cases["positive"],
                dtype=generated_weights.dtype,
                device=generated_weights.device,
            )
            neg_inputs = torch.tensor(
                cases["negative"],
                dtype=generated_weights.dtype,
                device=generated_weights.device,
            )
            pos_prob = torch.sigmoid(
                self.subject_forward_from_flat(generated_weights, pos_inputs)
            )
            neg_prob = torch.sigmoid(
                self.subject_forward_from_flat(generated_weights, neg_inputs)
            )
            margin = pos_prob.mean(dim=1) - neg_prob.mean(dim=1)
            target_weight = float(target_weights.get(pattern, 1.0))
            total_loss = (
                total_loss
                + F.relu(margin - max_allowed_margin).mean() * target_weight
            )
            n_patterns += 1

        if n_patterns == 0:
            return total_loss

        return total_loss / n_patterns

    def build_control_target_weights(self) -> Dict[str, float]:
        """Build target-specific weights for zero-latent anti-behavior controls."""
        weights: Dict[str, float] = {}
        if self.config.control_sorted_descending_target_weight != 1.0:
            weights["sorted_descending"] = (
                self.config.control_sorted_descending_target_weight
            )
        if self.config.control_has_majority_target_weight != 1.0:
            weights["has_majority"] = self.config.control_has_majority_target_weight
        return weights

    def build_matched_margin_target_weights(self) -> Dict[str, float]:
        """Build target-specific weights for matched calibrated margin training."""
        weights: Dict[str, float] = {}
        if self.config.matched_mountain_target_weight != 1.0:
            weights["mountain_pattern"] = self.config.matched_mountain_target_weight
        return weights

    def compute_hard_negative_control_penalty(
        self,
        generated_weights: torch.Tensor,
        pattern_to_idx: Dict[str, int],
        behavior_cases: Optional[Dict[str, Dict[str, List[List[float]]]]] = None,
        max_allowed_margin: float = -0.05,
        target_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Penalize the worst clean behavior target solved by a control decode."""
        if generated_weights.dim() == 1:
            generated_weights = generated_weights.unsqueeze(0)

        cases_by_pattern = behavior_cases or FOCUSED_BEHAVIOR_CASES
        target_weights = target_weights or {}
        losses = []

        for pattern, cases in cases_by_pattern.items():
            if pattern not in pattern_to_idx:
                continue

            pos_inputs = torch.tensor(
                cases["positive"],
                dtype=generated_weights.dtype,
                device=generated_weights.device,
            )
            neg_inputs = torch.tensor(
                cases["negative"],
                dtype=generated_weights.dtype,
                device=generated_weights.device,
            )
            pos_prob = torch.sigmoid(
                self.subject_forward_from_flat(generated_weights, pos_inputs)
            )
            neg_prob = torch.sigmoid(
                self.subject_forward_from_flat(generated_weights, neg_inputs)
            )
            margin = pos_prob.mean(dim=1) - neg_prob.mean(dim=1)
            target_weight = float(target_weights.get(pattern, 1.0))
            losses.append(F.relu(margin - max_allowed_margin).mean() * target_weight)

        if not losses:
            return generated_weights.sum() * 0.0

        return torch.stack(losses).max()

    def build_edit_targets(
        self,
        condition: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose deterministic different-label target conditions for edit training."""
        labels = labels.to(condition.device)
        target_condition = condition.clone()
        target_labels = labels.clone()
        valid_mask = torch.zeros(
            condition.size(0),
            dtype=torch.bool,
            device=condition.device,
        )

        for idx in range(condition.size(0)):
            if (
                hasattr(self, "_train_signature_centroids")
                and self._train_signature_centroids is not None
            ):
                centroid_labels = sorted(int(label) for label in self._train_signature_centroids)
                for candidate_label in centroid_labels:
                    if candidate_label == int(labels[idx].detach().cpu()):
                        continue
                    centroid_signature = self._train_signature_centroids[
                        candidate_label
                    ].to(condition.device)
                    target_condition[idx] = self.encode_signature(
                        centroid_signature.unsqueeze(0)
                    )[0]
                    target_labels[idx] = candidate_label
                    valid_mask[idx] = True
                    break
                if bool(valid_mask[idx]):
                    continue

            for offset in range(1, condition.size(0)):
                candidate_idx = (idx + offset) % condition.size(0)
                if labels[candidate_idx] != labels[idx]:
                    target_condition[idx] = condition[candidate_idx]
                    target_labels[idx] = labels[candidate_idx]
                    valid_mask[idx] = True
                    break

        return target_condition, target_labels, valid_mask

    def build_all_edit_targets(
        self,
        condition: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand edit training to every available different-label centroid target."""
        labels = labels.to(condition.device)
        if (
            not hasattr(self, "_train_signature_centroids")
            or self._train_signature_centroids is None
        ):
            target_condition, target_labels, valid_mask = self.build_edit_targets(
                condition,
                labels,
            )
            source_indices = torch.where(valid_mask)[0]
            return (
                target_condition[valid_mask],
                target_labels[valid_mask],
                source_indices,
            )

        centroid_labels = sorted(int(label) for label in self._train_signature_centroids)
        target_conditions = []
        target_labels = []
        source_indices = []
        for source_idx in range(condition.size(0)):
            source_label = int(labels[source_idx].detach().cpu())
            for target_label in centroid_labels:
                if target_label == source_label:
                    continue
                centroid_signature = self._train_signature_centroids[target_label].to(
                    condition.device
                )
                target_conditions.append(
                    self.encode_signature(centroid_signature.unsqueeze(0))[0]
                )
                target_labels.append(target_label)
                source_indices.append(source_idx)

        if not target_conditions:
            return (
                condition[:0],
                labels[:0],
                torch.empty(0, dtype=torch.long, device=condition.device),
            )

        return (
            torch.stack(target_conditions),
            torch.tensor(target_labels, dtype=labels.dtype, device=condition.device),
            torch.tensor(source_indices, dtype=torch.long, device=condition.device),
        )

    def compute_edit_margin_delta_loss(
        self,
        source_weights: torch.Tensor,
        edited_weights: torch.Tensor,
        target_labels: torch.Tensor,
        pattern_to_idx: Dict[str, int],
        behavior_cases: Optional[Dict[str, Dict[str, List[List[float]]]]] = None,
        target_delta: float = 0.10,
    ) -> torch.Tensor:
        """Train edited weights to improve target behavior margin over source."""
        if source_weights.dim() == 1:
            source_weights = source_weights.unsqueeze(0)
        if edited_weights.dim() == 1:
            edited_weights = edited_weights.unsqueeze(0)

        cases_by_pattern = behavior_cases or FOCUSED_BEHAVIOR_CASES
        target_labels = target_labels.to(edited_weights.device)
        total_loss = edited_weights.sum() * 0.0
        n_patterns = 0

        for pattern, cases in cases_by_pattern.items():
            pattern_idx = pattern_to_idx.get(pattern)
            if pattern_idx is None:
                continue

            mask = target_labels == pattern_idx
            if not bool(mask.any()):
                continue

            source_for_pattern = source_weights[mask]
            edited_for_pattern = edited_weights[mask]
            pos_inputs = torch.tensor(
                cases["positive"],
                dtype=edited_weights.dtype,
                device=edited_weights.device,
            )
            neg_inputs = torch.tensor(
                cases["negative"],
                dtype=edited_weights.dtype,
                device=edited_weights.device,
            )

            with torch.no_grad():
                source_pos = self.subject_forward_from_flat(
                    source_for_pattern.detach(),
                    pos_inputs,
                )
                source_neg = self.subject_forward_from_flat(
                    source_for_pattern.detach(),
                    neg_inputs,
                )
                source_margin = source_pos.mean(dim=1) - source_neg.mean(dim=1)

            edited_pos = self.subject_forward_from_flat(
                edited_for_pattern,
                pos_inputs,
            )
            edited_neg = self.subject_forward_from_flat(
                edited_for_pattern,
                neg_inputs,
            )
            edited_margin = edited_pos.mean(dim=1) - edited_neg.mean(dim=1)
            total_loss = total_loss + F.relu(
                target_delta - (edited_margin - source_margin)
            ).mean()
            n_patterns += 1

        if n_patterns == 0:
            return total_loss

        return total_loss / n_patterns

    def build_wrong_condition(
        self,
        condition: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build contrast conditions, preferring same-label different subjects."""
        if condition.size(0) < 2:
            return condition.clone()

        wrong_condition = torch.roll(condition, shifts=1, dims=0).clone()
        if labels is None:
            return wrong_condition

        labels = labels.to(condition.device)
        for idx in range(condition.size(0)):
            same_label_indices = torch.where(labels == labels[idx])[0]
            same_label_indices = same_label_indices[same_label_indices != idx]
            if same_label_indices.numel() > 0:
                wrong_condition[idx] = condition[same_label_indices[0]]

        return wrong_condition

    def build_different_label_condition(
        self,
        condition: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Build behavior-shuffled controls from rows with a different label."""
        if labels is None or condition.size(0) < 2:
            return None

        labels = labels.to(condition.device)
        shuffled_rows = []
        for idx in range(condition.size(0)):
            different_label_indices = torch.where(labels != labels[idx])[0]
            if different_label_indices.numel() == 0:
                return None
            shuffled_rows.append(condition[different_label_indices[0]])

        return torch.stack(shuffled_rows)

    def build_specificity_control_conditions(
        self,
        signatures: torch.Tensor,
        condition: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build train-split controls for condition-specificity ranking."""
        controls: Dict[str, torch.Tensor] = {
            "condition_ablation": torch.zeros_like(condition),
        }
        if hasattr(self, "_train_signature_mean") and self._train_signature_mean is not None:
            mean_signature = self._train_signature_mean.to(signatures.device)
            mean_signature = mean_signature.unsqueeze(0).expand_as(signatures)
            controls["null"] = self.encode_signature(mean_signature)

            if hasattr(self, "_train_signature_std") and self._train_signature_std is not None:
                std_signature = self._train_signature_std.to(signatures.device)
                noise_signature = mean_signature + torch.randn_like(signatures) * std_signature
                controls["noise"] = self.encode_signature(noise_signature)

        if (
            labels is not None
            and hasattr(self, "_train_signature_centroids")
            and self._train_signature_centroids is not None
        ):
            centroid_rows = []
            fallback = getattr(self, "_train_signature_mean", None)
            for label in labels.detach().cpu().tolist():
                centroid = self._train_signature_centroids.get(int(label), fallback)
                if centroid is None:
                    centroid = signatures.detach().mean(0).cpu()
                centroid_rows.append(centroid.to(signatures.device))
            controls["train_centroid"] = self.encode_signature(torch.stack(centroid_rows))

        return controls

    def build_behavior_control_conditions(
        self,
        signatures: torch.Tensor,
        condition: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build controls used for both specificity and anti-behavior penalties."""
        controls = self.build_specificity_control_conditions(
            signatures,
            condition,
            labels,
        )
        different_label = self.build_different_label_condition(condition, labels)
        if different_label is not None:
            controls["different_label"] = different_label
        return controls

    def weight_control_penalty(
        self,
        control_name: str,
        penalty: torch.Tensor,
    ) -> torch.Tensor:
        """Apply configured per-control penalty multipliers."""
        if control_name == "train_centroid":
            penalty = penalty * self.config.train_centroid_control_weight
        elif control_name == "condition_ablation":
            penalty = penalty * self.config.condition_ablation_control_weight
        elif control_name == "noise":
            penalty = penalty * self.config.noise_control_weight
        elif control_name == "different_label":
            penalty = penalty * self.config.shuffled_control_weight
        return penalty

    def build_condition_baseline(
        self,
        condition: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Encode same-label train centroids as residual decoder baselines."""
        if not self.config.use_condition_residual_decoder:
            return None
        if labels is None:
            return None
        if (
            not hasattr(self, "_train_signature_centroids")
            or self._train_signature_centroids is None
        ):
            return None

        baseline_rows = []
        fallback = getattr(self, "_train_signature_mean", None)
        for label in labels.detach().cpu().tolist():
            centroid = self._train_signature_centroids.get(int(label), fallback)
            if centroid is None:
                return None
            baseline_rows.append(centroid.to(condition.device))

        baseline_signatures = torch.stack(baseline_rows).to(condition.dtype)
        return self.encode_signature(baseline_signatures)
    
    def fit(
        self,
        weights: torch.Tensor,
        signatures: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        lambda_kl: Optional[float] = None,
        lambda_functional: Optional[float] = None,
        use_functional_loss: bool = False,
        functional_loss_start_epoch: Optional[int] = None,
        device: str = 'auto',
        verbose: bool = True,
        callback: Optional[Callable[[int, Dict], None]] = None,
        early_stopping_patience: Optional[int] = None,
        val_split: float = 0.1,
        behavior_cases: Optional[Dict[str, Dict[str, List[List[float]]]]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            weights: Training weights [n_samples, weight_dim]
            signatures: Training signatures [n_samples, sig_dim]
            labels: Optional labels for tracking
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            lambda_kl: KL divergence weight
            lambda_functional: Functional loss weight
            use_functional_loss: Whether to use functional loss
            functional_loss_start_epoch: Epoch to start functional loss
            device: Device to train on
            verbose: Print progress
            callback: Optional callback(epoch, metrics) called each epoch
            early_stopping_patience: Stop if val loss doesn't improve for N epochs
            val_split: Fraction of data to use for validation (default 0.1)
        
        Returns:
            Training history dict
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else \
                     'mps' if torch.backends.mps.is_available() else 'cpu'
        
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        lr = lr or self.config.lr
        lambda_kl = lambda_kl or self.config.lambda_kl
        lambda_functional = lambda_functional or self.config.lambda_functional
        functional_loss_start_epoch = functional_loss_start_epoch or self.config.functional_loss_start_epoch
        
        # Move to device
        self.to(device)
        weights = weights.to(device)
        signatures = signatures.to(device)
        
        # Train/val split
        n_samples = len(weights)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        # Shuffle indices for random split
        indices = torch.randperm(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        # Store indices for later retrieval (evaluation pipeline)
        self._train_indices = train_idx.cpu()
        self._val_indices = val_idx.cpu()
        
        train_weights = weights[train_idx]
        train_signatures = signatures[train_idx]
        val_weights = weights[val_idx]
        val_signatures = signatures[val_idx]

        # Fit normalization on train split only to avoid validation leakage.
        self.weight_mean = train_weights.mean(0)
        self.weight_std = train_weights.std(0).clamp(min=1e-6)
        self.sig_mean = train_signatures.mean(0)
        self.sig_std = train_signatures.std(0).clamp(min=1e-6)
        self._train_signature_mean = train_signatures.mean(0).detach().cpu()
        self._train_signature_std = train_signatures.std(0).clamp(min=1e-6).detach().cpu()
        self._normalization_fit_scope = "train_split"
        
        # Create dataloaders
        if labels is not None:
            labels = labels.to(device)
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]
            train_dataset = TensorDataset(train_weights, train_signatures, train_labels)
            val_dataset = TensorDataset(val_weights, val_signatures, val_labels)
            self._train_signature_centroids = {
                int(label): train_signatures[train_labels == label].mean(0).detach().cpu()
                for label in torch.unique(train_labels).tolist()
            }
        else:
            train_dataset = TensorDataset(train_weights, train_signatures)
            val_dataset = TensorDataset(val_weights, val_signatures)
            self._train_signature_centroids = None
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        if verbose:
            print(f"Train/Val split: {n_train}/{n_val} samples")
        
        # Optimizer with cosine annealing
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        history = {
            'loss': [], 'recon_loss': [], 'kl_loss': [], 'functional_loss': [],
            'target_behavior_loss': [], 'condition_behavior_loss': [],
            'calibrated_behavior_margin_loss': [],
            'shuffled_residual_contrastive_loss': [],
            'condition_specificity_loss': [], 'control_behavior_penalty_loss': [],
            'control_hard_negative_penalty_loss': [],
            'edit_behavior_loss': [], 'edit_margin_delta_loss': [],
            'val_loss': [], 'val_recon_loss': [], 'val_kl_loss': [], 'val_functional_loss': [],
            'val_target_behavior_loss': [], 'val_condition_behavior_loss': [],
            'val_calibrated_behavior_margin_loss': [],
            'val_shuffled_residual_contrastive_loss': [],
            'val_condition_specificity_loss': [], 'val_control_behavior_penalty_loss': [],
            'val_control_hard_negative_penalty_loss': [],
            'val_edit_behavior_loss': [], 'val_edit_margin_delta_loss': [],
            'lr': [],
        }
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0
            epoch_func = 0
            epoch_target = 0
            epoch_condition = 0
            epoch_calibrated_margin = 0
            epoch_shuffled_contrastive = 0
            epoch_specificity = 0
            epoch_control_penalty = 0
            epoch_control_hard = 0
            epoch_edit_behavior = 0
            epoch_edit_margin = 0
            
            use_func_this_epoch = use_functional_loss and epoch >= functional_loss_start_epoch
            
            for batch in train_loader:
                w_batch = batch[0]
                s_batch = batch[1]
                label_batch = batch[2] if len(batch) > 2 else None
                
                optimizer.zero_grad()
                
                recon, mu, logvar, condition = self(w_batch, s_batch)
                
                # Reconstruction loss
                loss_recon = F.mse_loss(recon, w_batch)
                
                # KL divergence
                loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                loss = loss_recon + lambda_kl * loss_kl
                
                # Functional loss (on subset to save compute)
                loss_func = torch.tensor(0.0, device=device)
                if use_func_this_epoch:
                    n_func = min(self.config.functional_loss_samples, len(w_batch))
                    loss_func = self.compute_functional_loss(
                        w_batch[:n_func], recon[:n_func]
                    )
                    loss = loss + lambda_functional * loss_func

                loss_target = torch.tensor(0.0, device=device)
                if use_func_this_epoch and label_batch is not None:
                    loss_target = self.compute_target_behavior_loss(
                        recon,
                        label_batch,
                        DEFAULT_PATTERN_TO_IDX,
                        behavior_cases=behavior_cases,
                    )
                    loss = loss + lambda_functional * loss_target

                loss_condition = torch.tensor(0.0, device=device)
                loss_calibrated_margin = torch.tensor(0.0, device=device)
                loss_shuffled_contrastive = torch.tensor(0.0, device=device)
                loss_control_penalty = torch.tensor(0.0, device=device)
                loss_control_hard = torch.tensor(0.0, device=device)
                loss_edit_behavior = torch.tensor(0.0, device=device)
                loss_edit_margin = torch.tensor(0.0, device=device)
                if use_func_this_epoch and label_batch is not None:
                    condition_baseline = self.build_condition_baseline(
                        condition,
                        label_batch,
                    )
                    condition_weights = self.decode_weights(
                        torch.zeros_like(mu),
                        condition,
                        condition_baseline=condition_baseline,
                    )
                    loss_condition = self.compute_target_behavior_loss(
                        condition_weights,
                        label_batch,
                        DEFAULT_PATTERN_TO_IDX,
                        behavior_cases=behavior_cases,
                    )
                    loss = loss + lambda_functional * loss_condition
                    loss_calibrated_margin = self.compute_calibrated_behavior_margin_loss(
                        condition_weights,
                        label_batch,
                        DEFAULT_PATTERN_TO_IDX,
                        behavior_cases=behavior_cases,
                        min_margin=self.config.matched_behavior_min_margin,
                        target_weights=self.build_matched_margin_target_weights(),
                    )
                    loss = loss + (
                        lambda_functional
                        * self.config.lambda_calibrated_behavior_margin
                        * loss_calibrated_margin
                    )

                    wrong_condition = self.build_wrong_condition(
                        condition,
                        label_batch,
                    )
                    behavior_controls = self.build_behavior_control_conditions(
                        s_batch,
                        condition,
                        label_batch,
                    )
                    different_label_condition = behavior_controls.get("different_label")
                    specificity_control_values = list(behavior_controls.values())
                    specificity_probes = self.build_subject_specificity_probes(
                        label_batch,
                        DEFAULT_PATTERN_TO_IDX,
                        behavior_cases,
                    )
                    specificity_weights = self.build_subject_specificity_sample_weights(
                        label_batch,
                        DEFAULT_PATTERN_TO_IDX,
                    )
                    control_target_weights = self.build_control_target_weights()
                    loss_specificity = self.compute_condition_functional_specificity_loss(
                        w_batch,
                        condition,
                        wrong_condition if len(w_batch) > 1 else None,
                        control_conditions=specificity_control_values,
                        condition_baseline=condition_baseline,
                        probe_inputs=specificity_probes,
                        sample_weights=specificity_weights,
                        n_probes=min(30, self.config.functional_loss_samples * 2),
                    )
                    loss = loss + (
                        lambda_functional
                        * self.config.lambda_condition_specificity
                        * loss_specificity
                    )

                    if (
                        different_label_condition is not None
                        and self.config.lambda_shuffled_residual_contrastive
                    ):
                        shuffled_weights = self.decode_weights(
                            torch.zeros_like(mu),
                            different_label_condition,
                            condition_baseline=condition_baseline,
                        )
                        loss_shuffled_contrastive = (
                            self.compute_matched_control_behavior_margin_loss(
                                condition_weights,
                                shuffled_weights,
                                label_batch,
                                DEFAULT_PATTERN_TO_IDX,
                                behavior_cases=behavior_cases,
                                min_delta=self.config.shuffled_residual_min_delta,
                                target_weights=self.build_matched_margin_target_weights(),
                            )
                        )
                        loss = loss + (
                            lambda_functional
                            * self.config.lambda_shuffled_residual_contrastive
                            * loss_shuffled_contrastive
                        )

                    penalty_terms = []
                    hard_penalty_terms = []
                    for control_name, control_condition in behavior_controls.items():
                        control_weights = self.decode_weights(
                            torch.zeros_like(mu),
                            control_condition,
                            condition_baseline=condition_baseline,
                        )
                        penalty = self.compute_all_target_control_penalty(
                            control_weights,
                            DEFAULT_PATTERN_TO_IDX,
                            behavior_cases=behavior_cases,
                            max_allowed_margin=self.config.control_max_allowed_margin,
                            target_weights=control_target_weights,
                        )
                        penalty = self.weight_control_penalty(control_name, penalty)
                        penalty_terms.append(penalty)
                        hard_penalty = self.compute_hard_negative_control_penalty(
                            control_weights,
                            DEFAULT_PATTERN_TO_IDX,
                            behavior_cases=behavior_cases,
                            max_allowed_margin=self.config.control_max_allowed_margin,
                            target_weights=control_target_weights,
                        )
                        hard_penalty = self.weight_control_penalty(
                            control_name,
                            hard_penalty,
                        )
                        hard_penalty_terms.append(hard_penalty)
                    if penalty_terms:
                        loss_control_penalty = torch.stack(penalty_terms).mean()
                        loss = loss + (
                            lambda_functional
                            * self.config.lambda_control_behavior_penalty
                            * loss_control_penalty
                        )
                    if hard_penalty_terms and self.config.lambda_control_hard_negative_penalty:
                        loss_control_hard = torch.stack(hard_penalty_terms).mean()
                        loss = loss + (
                            lambda_functional
                            * self.config.lambda_control_hard_negative_penalty
                            * loss_control_hard
                        )

                    target_condition, target_labels, edit_source_indices = (
                        self.build_all_edit_targets(condition, label_batch)
                    )
                    if len(edit_source_indices) > 0:
                        edited_weights = self.decode_weights(
                            mu[edit_source_indices],
                            target_condition,
                            condition_baseline=self.build_condition_baseline(
                                target_condition,
                                target_labels,
                            ),
                        )
                        source_weights_for_edit = w_batch[edit_source_indices]
                        target_labels_for_edit = target_labels
                        loss_edit_behavior = self.compute_target_behavior_loss(
                            edited_weights,
                            target_labels_for_edit,
                            DEFAULT_PATTERN_TO_IDX,
                            behavior_cases=behavior_cases,
                        )
                        loss_edit_margin = self.compute_edit_margin_delta_loss(
                            source_weights_for_edit,
                            edited_weights,
                            target_labels_for_edit,
                            DEFAULT_PATTERN_TO_IDX,
                            behavior_cases=behavior_cases,
                        )
                        loss = loss + lambda_functional * (
                            self.config.lambda_edit_behavior * loss_edit_behavior
                            + self.config.lambda_edit_margin_delta * loss_edit_margin
                        )
                else:
                    loss_specificity = torch.tensor(0.0, device=device)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += loss_recon.item()
                epoch_kl += loss_kl.item()
                epoch_func += loss_func.item() if use_func_this_epoch else 0
                epoch_target += loss_target.item() if use_func_this_epoch else 0
                epoch_condition += loss_condition.item() if use_func_this_epoch else 0
                epoch_calibrated_margin += loss_calibrated_margin.item() if use_func_this_epoch else 0
                epoch_shuffled_contrastive += loss_shuffled_contrastive.item() if use_func_this_epoch else 0
                epoch_specificity += loss_specificity.item() if use_func_this_epoch else 0
                epoch_control_penalty += loss_control_penalty.item() if use_func_this_epoch else 0
                epoch_control_hard += loss_control_hard.item() if use_func_this_epoch else 0
                epoch_edit_behavior += loss_edit_behavior.item() if use_func_this_epoch else 0
                epoch_edit_margin += loss_edit_margin.item() if use_func_this_epoch else 0
            
            scheduler.step()
            
            n_train_batches = len(train_loader)
            train_metrics = {
                'loss': epoch_loss / n_train_batches,
                'recon_loss': epoch_recon / n_train_batches,
                'kl_loss': epoch_kl / n_train_batches,
                'functional_loss': epoch_func / n_train_batches if use_func_this_epoch else 0,
                'target_behavior_loss': epoch_target / n_train_batches if use_func_this_epoch else 0,
                'condition_behavior_loss': epoch_condition / n_train_batches if use_func_this_epoch else 0,
                'calibrated_behavior_margin_loss': epoch_calibrated_margin / n_train_batches if use_func_this_epoch else 0,
                'shuffled_residual_contrastive_loss': epoch_shuffled_contrastive / n_train_batches if use_func_this_epoch else 0,
                'condition_specificity_loss': epoch_specificity / n_train_batches if use_func_this_epoch else 0,
                'control_behavior_penalty_loss': epoch_control_penalty / n_train_batches if use_func_this_epoch else 0,
                'control_hard_negative_penalty_loss': epoch_control_hard / n_train_batches if use_func_this_epoch else 0,
                'edit_behavior_loss': epoch_edit_behavior / n_train_batches if use_func_this_epoch else 0,
                'edit_margin_delta_loss': epoch_edit_margin / n_train_batches if use_func_this_epoch else 0,
                'lr': scheduler.get_last_lr()[0],
            }
            
            # Validation pass
            self.eval()
            val_loss = 0
            val_recon = 0
            val_kl = 0
            val_func = 0
            val_target = 0
            val_condition = 0
            val_calibrated_margin = 0
            val_shuffled_contrastive = 0
            val_specificity = 0
            val_control_penalty = 0
            val_control_hard = 0
            val_edit_behavior = 0
            val_edit_margin = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    w_batch = batch[0]
                    s_batch = batch[1]
                    label_batch = batch[2] if len(batch) > 2 else None
                    
                    recon, mu, logvar, condition = self(w_batch, s_batch)
                    
                    loss_recon = F.mse_loss(recon, w_batch)
                    loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = loss_recon + lambda_kl * loss_kl
                    
                    loss_func = torch.tensor(0.0, device=device)
                    if use_func_this_epoch:
                        n_func = min(self.config.functional_loss_samples, len(w_batch))
                        loss_func = self.compute_functional_loss(
                            w_batch[:n_func], recon[:n_func]
                        )
                        loss = loss + lambda_functional * loss_func

                    loss_target = torch.tensor(0.0, device=device)
                    if use_func_this_epoch and label_batch is not None:
                        loss_target = self.compute_target_behavior_loss(
                            recon,
                            label_batch,
                            DEFAULT_PATTERN_TO_IDX,
                            behavior_cases=behavior_cases,
                        )
                        loss = loss + lambda_functional * loss_target

                    loss_condition = torch.tensor(0.0, device=device)
                    loss_calibrated_margin = torch.tensor(0.0, device=device)
                    loss_shuffled_contrastive = torch.tensor(0.0, device=device)
                    loss_control_penalty = torch.tensor(0.0, device=device)
                    loss_control_hard = torch.tensor(0.0, device=device)
                    loss_edit_behavior = torch.tensor(0.0, device=device)
                    loss_edit_margin = torch.tensor(0.0, device=device)
                    if use_func_this_epoch and label_batch is not None:
                        condition_baseline = self.build_condition_baseline(
                            condition,
                            label_batch,
                        )
                        condition_weights = self.decode_weights(
                            torch.zeros_like(mu),
                            condition,
                            condition_baseline=condition_baseline,
                        )
                        loss_condition = self.compute_target_behavior_loss(
                            condition_weights,
                            label_batch,
                            DEFAULT_PATTERN_TO_IDX,
                            behavior_cases=behavior_cases,
                        )
                        loss = loss + lambda_functional * loss_condition
                        loss_calibrated_margin = self.compute_calibrated_behavior_margin_loss(
                            condition_weights,
                            label_batch,
                            DEFAULT_PATTERN_TO_IDX,
                            behavior_cases=behavior_cases,
                            min_margin=self.config.matched_behavior_min_margin,
                            target_weights=self.build_matched_margin_target_weights(),
                        )
                        loss = loss + (
                            lambda_functional
                            * self.config.lambda_calibrated_behavior_margin
                            * loss_calibrated_margin
                        )

                        wrong_condition = self.build_wrong_condition(
                            condition,
                            label_batch,
                        )
                        behavior_controls = self.build_behavior_control_conditions(
                            s_batch,
                            condition,
                            label_batch,
                        )
                        different_label_condition = behavior_controls.get("different_label")
                        specificity_control_values = list(behavior_controls.values())
                        specificity_probes = self.build_subject_specificity_probes(
                            label_batch,
                            DEFAULT_PATTERN_TO_IDX,
                            behavior_cases,
                        )
                        specificity_weights = self.build_subject_specificity_sample_weights(
                            label_batch,
                            DEFAULT_PATTERN_TO_IDX,
                        )
                        control_target_weights = self.build_control_target_weights()
                        loss_specificity = self.compute_condition_functional_specificity_loss(
                            w_batch,
                            condition,
                            wrong_condition if len(w_batch) > 1 else None,
                            control_conditions=specificity_control_values,
                            condition_baseline=condition_baseline,
                            probe_inputs=specificity_probes,
                            sample_weights=specificity_weights,
                            n_probes=min(30, self.config.functional_loss_samples * 2),
                        )
                        loss = loss + (
                            lambda_functional
                            * self.config.lambda_condition_specificity
                            * loss_specificity
                        )
                        if (
                            different_label_condition is not None
                            and self.config.lambda_shuffled_residual_contrastive
                        ):
                            shuffled_weights = self.decode_weights(
                                torch.zeros_like(mu),
                                different_label_condition,
                                condition_baseline=condition_baseline,
                            )
                            loss_shuffled_contrastive = (
                                self.compute_matched_control_behavior_margin_loss(
                                    condition_weights,
                                    shuffled_weights,
                                    label_batch,
                                    DEFAULT_PATTERN_TO_IDX,
                                    behavior_cases=behavior_cases,
                                    min_delta=self.config.shuffled_residual_min_delta,
                                    target_weights=self.build_matched_margin_target_weights(),
                                )
                            )
                            loss = loss + (
                                lambda_functional
                                * self.config.lambda_shuffled_residual_contrastive
                                * loss_shuffled_contrastive
                            )
                        penalty_terms = []
                        hard_penalty_terms = []
                        for control_name, control_condition in behavior_controls.items():
                            control_weights = self.decode_weights(
                                torch.zeros_like(mu),
                                control_condition,
                                condition_baseline=condition_baseline,
                            )
                            penalty = self.compute_all_target_control_penalty(
                                control_weights,
                                DEFAULT_PATTERN_TO_IDX,
                                behavior_cases=behavior_cases,
                                max_allowed_margin=self.config.control_max_allowed_margin,
                                target_weights=control_target_weights,
                            )
                            penalty = self.weight_control_penalty(control_name, penalty)
                            penalty_terms.append(penalty)
                            hard_penalty = self.compute_hard_negative_control_penalty(
                                control_weights,
                                DEFAULT_PATTERN_TO_IDX,
                                behavior_cases=behavior_cases,
                                max_allowed_margin=self.config.control_max_allowed_margin,
                                target_weights=control_target_weights,
                            )
                            hard_penalty = self.weight_control_penalty(
                                control_name,
                                hard_penalty,
                            )
                            hard_penalty_terms.append(hard_penalty)
                        if penalty_terms:
                            loss_control_penalty = torch.stack(penalty_terms).mean()
                            loss = loss + (
                                lambda_functional
                                * self.config.lambda_control_behavior_penalty
                                * loss_control_penalty
                            )
                        if hard_penalty_terms and self.config.lambda_control_hard_negative_penalty:
                            loss_control_hard = torch.stack(hard_penalty_terms).mean()
                            loss = loss + (
                                lambda_functional
                                * self.config.lambda_control_hard_negative_penalty
                                * loss_control_hard
                            )
                        target_condition, target_labels, edit_source_indices = (
                            self.build_all_edit_targets(condition, label_batch)
                        )
                        if len(edit_source_indices) > 0:
                            edited_weights = self.decode_weights(
                                mu[edit_source_indices],
                                target_condition,
                                condition_baseline=self.build_condition_baseline(
                                    target_condition,
                                    target_labels,
                                ),
                            )
                            source_weights_for_edit = w_batch[edit_source_indices]
                            target_labels_for_edit = target_labels
                            loss_edit_behavior = self.compute_target_behavior_loss(
                                edited_weights,
                                target_labels_for_edit,
                                DEFAULT_PATTERN_TO_IDX,
                                behavior_cases=behavior_cases,
                            )
                            loss_edit_margin = self.compute_edit_margin_delta_loss(
                                source_weights_for_edit,
                                edited_weights,
                                target_labels_for_edit,
                                DEFAULT_PATTERN_TO_IDX,
                                behavior_cases=behavior_cases,
                            )
                            loss = loss + lambda_functional * (
                                self.config.lambda_edit_behavior * loss_edit_behavior
                                + self.config.lambda_edit_margin_delta * loss_edit_margin
                            )
                    else:
                        loss_specificity = torch.tensor(0.0, device=device)
                    
                    val_loss += loss.item()
                    val_recon += loss_recon.item()
                    val_kl += loss_kl.item()
                    val_func += loss_func.item() if use_func_this_epoch else 0
                    val_target += loss_target.item() if use_func_this_epoch else 0
                    val_condition += loss_condition.item() if use_func_this_epoch else 0
                    val_calibrated_margin += loss_calibrated_margin.item() if use_func_this_epoch else 0
                    val_shuffled_contrastive += loss_shuffled_contrastive.item() if use_func_this_epoch else 0
                    val_specificity += loss_specificity.item() if use_func_this_epoch else 0
                    val_control_penalty += loss_control_penalty.item() if use_func_this_epoch else 0
                    val_control_hard += loss_control_hard.item() if use_func_this_epoch else 0
                    val_edit_behavior += loss_edit_behavior.item() if use_func_this_epoch else 0
                    val_edit_margin += loss_edit_margin.item() if use_func_this_epoch else 0
            
            n_val_batches = len(val_loader)
            val_metrics = {
                'val_loss': val_loss / n_val_batches,
                'val_recon_loss': val_recon / n_val_batches,
                'val_kl_loss': val_kl / n_val_batches,
                'val_functional_loss': val_func / n_val_batches if use_func_this_epoch else 0,
                'val_target_behavior_loss': val_target / n_val_batches if use_func_this_epoch else 0,
                'val_condition_behavior_loss': val_condition / n_val_batches if use_func_this_epoch else 0,
                'val_calibrated_behavior_margin_loss': val_calibrated_margin / n_val_batches if use_func_this_epoch else 0,
                'val_shuffled_residual_contrastive_loss': val_shuffled_contrastive / n_val_batches if use_func_this_epoch else 0,
                'val_condition_specificity_loss': val_specificity / n_val_batches if use_func_this_epoch else 0,
                'val_control_behavior_penalty_loss': val_control_penalty / n_val_batches if use_func_this_epoch else 0,
                'val_control_hard_negative_penalty_loss': val_control_hard / n_val_batches if use_func_this_epoch else 0,
                'val_edit_behavior_loss': val_edit_behavior / n_val_batches if use_func_this_epoch else 0,
                'val_edit_margin_delta_loss': val_edit_margin / n_val_batches if use_func_this_epoch else 0,
            }
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            for k, v in metrics.items():
                if k in history:
                    history[k].append(v)
            
            if callback:
                callback(epoch, metrics)
            
            if verbose and (epoch + 1) % 10 == 0:
                msg = (f"Epoch {epoch+1:3d}/{epochs} | "
                       f"Loss: {train_metrics['loss']:.4f} | "
                       f"Val: {val_metrics['val_loss']:.4f} | "
                       f"Recon: {train_metrics['recon_loss']:.4f}")
                if use_func_this_epoch:
                    msg += f" | Func: {train_metrics['functional_loss']:.4f}"
                    msg += f" | ValFunc: {val_metrics['val_functional_loss']:.4f}"
                    msg += f" | Target: {train_metrics['target_behavior_loss']:.4f}"
                    msg += f" | Cond: {train_metrics['condition_behavior_loss']:.4f}"
                    msg += f" | CalMargin: {train_metrics['calibrated_behavior_margin_loss']:.4f}"
                    msg += f" | ShufCtr: {train_metrics['shuffled_residual_contrastive_loss']:.4f}"
                    msg += f" | Spec: {train_metrics['condition_specificity_loss']:.4f}"
                    msg += f" | CtrlPen: {train_metrics['control_behavior_penalty_loss']:.4f}"
                    msg += f" | HardCtrl: {train_metrics['control_hard_negative_penalty_loss']:.4f}"
                    msg += f" | Edit: {train_metrics['edit_behavior_loss']:.4f}"
                print(msg)
            
            # Early stopping check (using validation loss)
            if early_stopping_patience is not None:
                current_val_loss = val_metrics['val_loss']
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1} (patience {early_stopping_patience}, best val_loss: {best_val_loss:.4f})")
                        # Restore best state
                        if best_state is not None:
                            self.load_state_dict(best_state)
                        break
        
        return history
    
    def save(self, path: str):
        """Save model to file."""
        # Save config as dict to avoid pickling issues
        config_dict = {
            'weight_dim': self.config.weight_dim,
            'sig_dim': self.config.sig_dim,
            'latent_dim': self.config.latent_dim,
            'condition_dim': self.config.condition_dim,
            'hidden_dim': self.config.hidden_dim,
            'dropout': self.config.dropout,
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'lr': self.config.lr,
            'weight_decay': self.config.weight_decay,
            'lambda_kl': self.config.lambda_kl,
            'lambda_functional': self.config.lambda_functional,
            'lambda_condition_specificity': self.config.lambda_condition_specificity,
            'lambda_calibrated_behavior_margin': self.config.lambda_calibrated_behavior_margin,
            'matched_behavior_min_margin': self.config.matched_behavior_min_margin,
            'matched_mountain_target_weight': self.config.matched_mountain_target_weight,
            'lambda_control_behavior_penalty': self.config.lambda_control_behavior_penalty,
            'lambda_control_hard_negative_penalty': self.config.lambda_control_hard_negative_penalty,
            'control_max_allowed_margin': self.config.control_max_allowed_margin,
            'train_centroid_control_weight': self.config.train_centroid_control_weight,
            'condition_ablation_control_weight': self.config.condition_ablation_control_weight,
            'noise_control_weight': self.config.noise_control_weight,
            'shuffled_control_weight': self.config.shuffled_control_weight,
            'control_sorted_descending_target_weight': self.config.control_sorted_descending_target_weight,
            'control_has_majority_target_weight': self.config.control_has_majority_target_weight,
            'sorted_descending_specificity_weight': self.config.sorted_descending_specificity_weight,
            'lambda_edit_behavior': self.config.lambda_edit_behavior,
            'lambda_edit_margin_delta': self.config.lambda_edit_margin_delta,
            'use_condition_residual_decoder': self.config.use_condition_residual_decoder,
            'condition_residual_scale': self.config.condition_residual_scale,
            'lambda_shuffled_residual_contrastive': self.config.lambda_shuffled_residual_contrastive,
            'shuffled_residual_min_delta': self.config.shuffled_residual_min_delta,
            'functional_loss_start_epoch': self.config.functional_loss_start_epoch,
            'functional_loss_samples': self.config.functional_loss_samples,
            'num_layers': self.config.num_layers,
            'neurons_per_layer': self.config.neurons_per_layer,
            'input_dim': self.config.input_dim,
        }
        save_dict = {
            'config_dict': config_dict,
            'state_dict': self.state_dict(),
        }
        # Save train/val indices if available (from fit())
        if hasattr(self, '_train_indices') and self._train_indices is not None:
            save_dict['train_indices'] = self._train_indices
        if hasattr(self, '_val_indices') and self._val_indices is not None:
            save_dict['val_indices'] = self._val_indices
        if hasattr(self, '_dataset_patterns') and self._dataset_patterns is not None:
            save_dict['dataset_patterns'] = self._dataset_patterns
        if hasattr(self, '_dataset_provenance') and self._dataset_provenance is not None:
            save_dict['dataset_provenance'] = self._dataset_provenance
        if (
            hasattr(self, '_behavior_suite_metadata')
            and self._behavior_suite_metadata is not None
        ):
            save_dict['behavior_suite_metadata'] = self._behavior_suite_metadata
        if (
            hasattr(self, '_normalization_fit_scope')
            and self._normalization_fit_scope is not None
        ):
            save_dict['normalization_fit_scope'] = self._normalization_fit_scope
        if hasattr(self, '_train_signature_mean') and self._train_signature_mean is not None:
            save_dict['train_signature_mean'] = self._train_signature_mean
        if hasattr(self, '_train_signature_std') and self._train_signature_std is not None:
            save_dict['train_signature_std'] = self._train_signature_std
        if (
            hasattr(self, '_train_signature_centroids')
            and self._train_signature_centroids is not None
        ):
            save_dict['train_signature_centroids'] = self._train_signature_centroids
        torch.save(save_dict, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'FunctionalHyperNetwork':
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        # Handle both old (config object) and new (config_dict) formats
        if 'config_dict' in checkpoint:
            config = HyperNetConfig(**checkpoint['config_dict'])
        else:
            config = checkpoint['config']
        model = cls(config=config)
        # Allow loading older checkpoints that may be missing new buffers
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # Load train/val indices if available (for evaluation pipeline)
        model._train_indices = checkpoint.get('train_indices', None)
        model._val_indices = checkpoint.get('val_indices', None)
        model._dataset_patterns = checkpoint.get('dataset_patterns', None)
        model._dataset_provenance = checkpoint.get('dataset_provenance', None)
        model._behavior_suite_metadata = checkpoint.get(
            'behavior_suite_metadata',
            None,
        )
        model._normalization_fit_scope = checkpoint.get(
            'normalization_fit_scope',
            None,
        )
        model._train_signature_mean = checkpoint.get('train_signature_mean', None)
        model._train_signature_std = checkpoint.get('train_signature_std', None)
        model._train_signature_centroids = checkpoint.get(
            'train_signature_centroids',
            None,
        )
        return model


class BehaviorEditor:
    """
    Edit neural network behavior using the FunctionalHyperNetwork.
    
    Workflow:
    1. Encode original weights to latent z
    2. Change the conditioning (from source to target behavior)
    3. Decode with new conditioning
    4. Verify functional behavior change
    """
    
    def __init__(self, model: FunctionalHyperNetwork):
        self.model = model
        self.model.eval()
    
    @torch.no_grad()
    def edit(
        self,
        original_weights: torch.Tensor,
        source_signature: torch.Tensor,
        target_signature: torch.Tensor,
        interpolation: float = 1.0,
        source_baseline_signature: Optional[torch.Tensor] = None,
        target_baseline_signature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Edit weights to exhibit target behavior.
        
        Args:
            original_weights: Weights of network to edit [weight_dim]
            source_signature: Original behavior signature [sig_dim]
            target_signature: Target behavior signature [sig_dim]
            interpolation: How much to move toward target (0=source, 1=target)
        
        Returns:
            Edited weights [weight_dim]
        """
        device = next(self.model.parameters()).device
        
        # Ensure proper dimensions
        if original_weights.dim() == 1:
            original_weights = original_weights.unsqueeze(0)
        if source_signature.dim() == 1:
            source_signature = source_signature.unsqueeze(0)
        if target_signature.dim() == 1:
            target_signature = target_signature.unsqueeze(0)
        if (
            source_baseline_signature is not None
            and source_baseline_signature.dim() == 1
        ):
            source_baseline_signature = source_baseline_signature.unsqueeze(0)
        if (
            target_baseline_signature is not None
            and target_baseline_signature.dim() == 1
        ):
            target_baseline_signature = target_baseline_signature.unsqueeze(0)
        
        original_weights = original_weights.to(device)
        source_signature = source_signature.to(device)
        target_signature = target_signature.to(device)
        if source_baseline_signature is not None:
            source_baseline_signature = source_baseline_signature.to(device)
        if target_baseline_signature is not None:
            target_baseline_signature = target_baseline_signature.to(device)
        
        # Get conditioning vectors
        source_cond = self.model.encode_signature(source_signature)
        target_cond = self.model.encode_signature(target_signature)
        condition_baseline = None
        if (
            source_baseline_signature is not None
            and target_baseline_signature is not None
        ):
            source_baseline = self.model.encode_signature(source_baseline_signature)
            target_baseline = self.model.encode_signature(target_baseline_signature)
            condition_baseline = (
                (1 - interpolation) * source_baseline
                + interpolation * target_baseline
            )
        
        # Encode original weights with source condition
        mu, _ = self.model.encode_weights(original_weights, source_cond)
        z = mu  # Use mean for deterministic editing
        
        # Interpolate conditioning
        edited_cond = (1 - interpolation) * source_cond + interpolation * target_cond
        
        # Decode with edited conditioning
        edited_weights = self.model.decode_weights(
            z,
            edited_cond,
            condition_baseline=condition_baseline,
        )
        
        return edited_weights.squeeze(0)
    
    @torch.no_grad()
    def swap_behavior(
        self,
        original_weights: torch.Tensor,
        source_signature: torch.Tensor,
        target_signature: torch.Tensor,
        source_baseline_signature: Optional[torch.Tensor] = None,
        target_baseline_signature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fully swap behavior (interpolation=1.0).
        
        This is a convenience wrapper around edit().
        """
        return self.edit(
            original_weights,
            source_signature,
            target_signature,
            interpolation=1.0,
            source_baseline_signature=source_baseline_signature,
            target_baseline_signature=target_baseline_signature,
        )
    
    @torch.no_grad()
    def interpolate(
        self,
        original_weights: torch.Tensor,
        source_signature: torch.Tensor,
        target_signature: torch.Tensor,
        steps: int = 5,
        source_baseline_signature: Optional[torch.Tensor] = None,
        target_baseline_signature: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Generate interpolated weights between source and target behavior.
        
        Args:
            original_weights: Starting weights
            source_signature: Source behavior
            target_signature: Target behavior
            steps: Number of interpolation steps
        
        Returns:
            List of weight tensors from source to target
        """
        results = []
        for i in range(steps + 1):
            alpha = i / steps
            edited = self.edit(
                original_weights, source_signature, target_signature,
                interpolation=alpha,
                source_baseline_signature=source_baseline_signature,
                target_baseline_signature=target_baseline_signature,
            )
            results.append(edited)
        return results
    
    def create_edited_network(
        self,
        original_weights: torch.Tensor,
        source_signature: torch.Tensor,
        target_signature: torch.Tensor,
        interpolation: float = 1.0,
        source_baseline_signature: Optional[torch.Tensor] = None,
        target_baseline_signature: Optional[torch.Tensor] = None,
    ) -> SubjectNetwork:
        """
        Edit weights and return a functional SubjectNetwork.
        
        Args:
            original_weights: Weights to edit
            source_signature: Original behavior
            target_signature: Target behavior
            interpolation: Edit strength
        
        Returns:
            SubjectNetwork with edited weights
        """
        edited_weights = self.edit(
            original_weights,
            source_signature,
            target_signature,
            interpolation,
            source_baseline_signature=source_baseline_signature,
            target_baseline_signature=target_baseline_signature,
        )
        
        return SubjectNetwork.from_weights(
            edited_weights.cpu(),
            num_layers=self.model.config.num_layers,
            neurons_per_layer=self.model.config.neurons_per_layer,
            input_dim=self.model.config.input_dim,
        )
