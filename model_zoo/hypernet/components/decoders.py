"""
Hypernetwork decoder for weight generation.

Generates weights from behavioral latent + position information.

Key hypernetwork best practices applied:
1. Position encoding separate from behavior (sinusoidal)
2. FiLM conditioning for strong behavior injection
3. NTK parameterization (1/sqrt(fan_in) scaling) for stable training
4. Chunked/per-neuron generation (not full network at once)
5. Zero initialization of output layer
6. Layer normalization throughout
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class SinusoidalPositionEncoder(nn.Module):
    """
    Sinusoidal position encoding for layer/neuron position.
    
    More robust than learned embeddings for small training sets.
    Encodes: [layer_idx, neuron_idx, fan_in]
    """
    
    def __init__(self, input_dim: int = 3, output_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Number of frequencies per input dimension
        self.freqs_per_dim = output_dim // (2 * input_dim)
        
        # Precompute frequency bands
        freqs = torch.exp(
            torch.linspace(0, math.log(100), self.freqs_per_dim)
        )
        self.register_buffer('freqs', freqs)
    
    def forward(self, positions: Tensor) -> Tensor:
        """
        Args:
            positions: [..., 3] containing [layer_idx, neuron_idx, fan_in]
            
        Returns:
            encoding: [..., output_dim]
        """
        batch_shape = positions.shape[:-1]
        
        # Normalize positions for better frequency coverage
        # layer_idx: divide by ~10 (max layers)
        # neuron_idx: divide by ~10 (max neurons per layer)
        # fan_in: divide by ~10
        scale = torch.tensor([10.0, 10.0, 10.0], device=positions.device)
        positions_normalized = positions / scale
        
        encodings = []
        for i in range(self.input_dim):
            pos_i = positions_normalized[..., i:i+1]  # [..., 1]
            
            # Apply frequencies
            pos_freq = pos_i * self.freqs  # [..., freqs_per_dim]
            
            # Sin and cos
            encodings.append(torch.sin(pos_freq))
            encodings.append(torch.cos(pos_freq))
        
        # Concatenate all encodings
        encoding = torch.cat(encodings, dim=-1)
        
        # Pad to exact output_dim if needed
        if encoding.shape[-1] < self.output_dim:
            padding = torch.zeros(*batch_shape, self.output_dim - encoding.shape[-1], 
                                  device=encoding.device)
            encoding = torch.cat([encoding, padding], dim=-1)
        elif encoding.shape[-1] > self.output_dim:
            encoding = encoding[..., :self.output_dim]
        
        return encoding


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    Applies affine transformation conditioned on external signal:
        output = gamma * input + beta
    
    where gamma, beta are derived from the conditioning signal.
    """
    
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        
        # Generate scale (gamma) and shift (beta) from conditioning
        self.film_gen = nn.Linear(cond_dim, hidden_dim * 2)
        
        # Initialize to identity transform: gamma=1, beta=0
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)
        # Set gamma bias to 1
        self.film_gen.bias.data[:hidden_dim] = 1.0
    
    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """
        Args:
            x: [..., hidden_dim] features to modulate
            condition: [..., cond_dim] conditioning signal
            
        Returns:
            modulated: [..., hidden_dim]
        """
        film_params = self.film_gen(condition)
        gamma, beta = film_params.chunk(2, dim=-1)
        return gamma * x + beta


class FiLMBlock(nn.Module):
    """
    FiLM-conditioned MLP block with residual connection.
    
    Transforms features while being conditioned on behavioral latent.
    """
    
    def __init__(self, cond_dim: int, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.film = FiLMLayer(cond_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        
        # Residual scaling (helps with deep networks)
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """
        Args:
            x: [..., input_dim]
            condition: [..., cond_dim] behavioral latent
            
        Returns:
            output: [..., input_dim]
        """
        residual = x
        
        x = self.norm(x)
        x = self.linear1(x)
        x = self.film(x, condition)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return residual + self.residual_scale * x


class HypernetDecoder(nn.Module):
    """
    Hypernetwork-style decoder for weight generation.
    
    Generates per-neuron weights from behavioral latent + position.
    
    Key design principles:
    1. Position encoding is SEPARATE from behavior (concatenated, not mixed)
    2. FiLM conditioning injects behavior at every layer
    3. NTK scaling (1/sqrt(fan_in)) stabilizes training
    4. Output layer initialized to zero (starts with zero prediction)
    
    Architecture:
        [behavior_latent, position_encoding] -> FiLM blocks -> weights
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        position_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 9,  # max_fan_in + 1
        num_layers: int = 4,
        dropout: float = 0.1,
        use_ntk_scaling: bool = True,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_ntk_scaling = use_ntk_scaling
        
        # Position encoder (sinusoidal)
        self.position_encoder = SinusoidalPositionEncoder(
            input_dim=3,  # layer_idx, neuron_idx, fan_in
            output_dim=position_dim
        )
        
        # Input projection: [latent, position] -> hidden
        self.input_proj = nn.Linear(latent_dim + position_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # FiLM-conditioned blocks
        self.blocks = nn.ModuleList([
            FiLMBlock(latent_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Zero-initialize output layer (hypernetwork best practice)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # NTK scaling factor (learnable but initialized to reasonable value)
        if use_ntk_scaling:
            self.ntk_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self,
        latent: Tensor,
        positions: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Generate weights from behavioral latent and position.
        
        Args:
            latent: [batch, num_neurons, latent_dim] behavioral features
            positions: [batch, num_neurons, 3] (layer_idx, neuron_idx, fan_in)
            mask: Optional [batch, num_neurons] validity mask
            
        Returns:
            weights: [batch, num_neurons, output_dim]
        """
        # Handle unbatched input
        squeeze_batch = False
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
            positions = positions.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            squeeze_batch = True
        
        # Encode positions
        pos_enc = self.position_encoder(positions)
        
        # Concatenate latent and position
        x = torch.cat([latent, pos_enc], dim=-1)
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        # FiLM blocks (conditioned on behavior latent)
        for block in self.blocks:
            x = block(x, latent)
        
        # Output projection
        x = self.output_norm(x)
        weights = self.output_proj(x)
        
        # NTK scaling: divide by sqrt(fan_in) for each neuron
        if self.use_ntk_scaling:
            fan_in = positions[..., 2:3].clamp(min=1)  # [batch, num_neurons, 1]
            weights = weights * self.ntk_scale / torch.sqrt(fan_in)
        
        if squeeze_batch:
            weights = weights.squeeze(0)
        
        return weights


class DualLatentDecoder(nn.Module):
    """
    Hypernetwork decoder that takes DUAL latents: behavioral + weight.
    
    Supports multiple combination strategies:
    - 'concat': [Z_behavior, Z_weight, position] concatenated
    - 'add': Z_behavior + Z_weight, then concat with position
    - 'gated': learned gating α*Z_behavior + (1-α)*Z_weight
    
    The behavioral latent is INTERPRETABLE (for rep-eng).
    The weight latent carries detailed reconstruction info.
    """
    
    def __init__(
        self,
        behavioral_latent_dim: int = 64,
        weight_latent_dim: int = 64,
        position_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 9,
        num_layers: int = 4,
        dropout: float = 0.1,
        combination_mode: str = 'concat',  # 'concat', 'add', 'gated'
        use_ntk_scaling: bool = True,
    ):
        super().__init__()
        
        self.behavioral_latent_dim = behavioral_latent_dim
        self.weight_latent_dim = weight_latent_dim
        self.output_dim = output_dim
        self.combination_mode = combination_mode
        self.use_ntk_scaling = use_ntk_scaling
        
        # Position encoder
        self.position_encoder = SinusoidalPositionEncoder(
            input_dim=3,
            output_dim=position_dim
        )
        
        # Combination-specific components
        if combination_mode == 'concat':
            input_dim = behavioral_latent_dim + weight_latent_dim + position_dim
            self.combined_latent_dim = behavioral_latent_dim + weight_latent_dim
        elif combination_mode == 'add':
            assert behavioral_latent_dim == weight_latent_dim, \
                "Add mode requires equal latent dims"
            input_dim = behavioral_latent_dim + position_dim
            self.combined_latent_dim = behavioral_latent_dim
        elif combination_mode == 'gated':
            assert behavioral_latent_dim == weight_latent_dim, \
                "Gated mode requires equal latent dims"
            # Learned gating based on both latents
            self.gate = nn.Sequential(
                nn.Linear(behavioral_latent_dim + weight_latent_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
            input_dim = behavioral_latent_dim + position_dim
            self.combined_latent_dim = behavioral_latent_dim
        else:
            raise ValueError(f"Unknown combination mode: {combination_mode}")
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # FiLM-conditioned blocks (conditioned on combined behavioral+weight latent)
        self.blocks = nn.ModuleList([
            FiLMBlock(self.combined_latent_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Zero-initialize output layer
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # NTK scaling
        if use_ntk_scaling:
            self.ntk_scale = nn.Parameter(torch.tensor(0.1))
    
    def _combine_latents(
        self,
        z_behavior: Tensor,
        z_weight: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Combine behavioral and weight latents.
        
        Returns:
            combined_for_input: latent to concat with position for input projection
            combined_for_film: latent for FiLM conditioning
        """
        if self.combination_mode == 'concat':
            combined = torch.cat([z_behavior, z_weight], dim=-1)
            return combined, combined
        
        elif self.combination_mode == 'add':
            combined = z_behavior + z_weight
            return combined, combined
        
        elif self.combination_mode == 'gated':
            gate_input = torch.cat([z_behavior, z_weight], dim=-1)
            alpha = self.gate(gate_input)  # [batch, num_neurons, 1]
            combined = alpha * z_behavior + (1 - alpha) * z_weight
            return combined, combined
        
        else:
            # Should never reach here due to __init__ validation
            raise ValueError(f"Unknown combination mode: {self.combination_mode}")
    
    def forward(
        self,
        z_behavior: Tensor,
        z_weight: Tensor,
        positions: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Generate weights from dual latents and position.
        
        Args:
            z_behavior: [batch, num_neurons, behavioral_latent_dim] behavioral latent
            z_weight: [batch, num_neurons, weight_latent_dim] weight latent
            positions: [batch, num_neurons, 3] (layer_idx, neuron_idx, fan_in)
            mask: Optional validity mask
            
        Returns:
            weights: [batch, num_neurons, output_dim]
        """
        # Handle unbatched input
        squeeze_batch = False
        if z_behavior.dim() == 2:
            z_behavior = z_behavior.unsqueeze(0)
            z_weight = z_weight.unsqueeze(0)
            positions = positions.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            squeeze_batch = True
        
        # Encode positions
        pos_enc = self.position_encoder(positions)
        
        # Combine latents
        combined_input, combined_film = self._combine_latents(z_behavior, z_weight)
        
        # Concatenate with position for input
        x = torch.cat([combined_input, pos_enc], dim=-1)
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        # FiLM blocks (conditioned on combined latent)
        for block in self.blocks:
            x = block(x, combined_film)
        
        # Output projection
        x = self.output_norm(x)
        weights = self.output_proj(x)
        
        # NTK scaling
        if self.use_ntk_scaling:
            fan_in = positions[..., 2:3].clamp(min=1)
            weights = weights * self.ntk_scale / torch.sqrt(fan_in)
        
        if squeeze_batch:
            weights = weights.squeeze(0)
        
        return weights


class WeightAutoencoder(nn.Module):
    """
    Weight autoencoder for establishing baseline reconstruction quality.
    
    Encoder: weights -> latent (position-agnostic)
    Decoder: latent + position -> weights
    
    This establishes the upper bound on reconstruction quality.
    """
    
    def __init__(
        self,
        input_dim: int = 9,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        encoder_layers: int = 3,
        decoder_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        from .encoders import WeightEncoder
        
        self.encoder = WeightEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim // 2,  # Encoder is smaller
            num_layers=encoder_layers,
            dropout=dropout,
        )
        
        self.decoder = HypernetDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=decoder_layers,
            dropout=dropout,
        )
    
    def forward(
        self,
        weight_tokens: Tensor,
        positions: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode and decode weights.
        
        Args:
            weight_tokens: [batch, num_neurons, input_dim]
            positions: [batch, num_neurons, 3]
            mask: Optional validity mask
            
        Returns:
            reconstructed: [batch, num_neurons, input_dim]
            latent: [batch, num_neurons, latent_dim]
        """
        latent = self.encoder(weight_tokens, mask)
        reconstructed = self.decoder(latent, positions, mask)
        return reconstructed, latent
    
    def encode(self, weight_tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Encode weights to latent space."""
        return self.encoder(weight_tokens, mask)
    
    def decode(
        self, 
        latent: Tensor, 
        positions: Tensor, 
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """Decode latent + position to weights."""
        return self.decoder(latent, positions, mask)


def test_decoder():
    """Test decoder components."""
    batch_size = 4
    num_neurons = 19
    latent_dim = 64
    
    # Test position encoder
    pos_enc = SinusoidalPositionEncoder(input_dim=3, output_dim=32)
    positions = torch.tensor([
        [0, 0, 5],  # Layer 0, neuron 0, fan_in=5
        [0, 1, 5],
        [3, 0, 6],
        [3, 1, 6],
    ], dtype=torch.float)
    
    pos_encoding = pos_enc(positions)
    print(f"Position encoding shape: {pos_encoding.shape}")
    assert pos_encoding.shape == (4, 32)
    
    # Different positions should have different encodings
    assert not torch.allclose(pos_encoding[0], pos_encoding[2])
    print("Position encoder test passed!")
    
    # Test FiLM layer
    film = FiLMLayer(cond_dim=64, hidden_dim=256)
    x = torch.randn(batch_size, 256)
    cond = torch.randn(batch_size, 64)
    modulated = film(x, cond)
    print(f"FiLM output shape: {modulated.shape}")
    assert modulated.shape == (batch_size, 256)
    
    # Initial FiLM should be near identity
    film_identity = FiLMLayer(cond_dim=64, hidden_dim=256)
    x_test = torch.randn(batch_size, 256)
    cond_zero = torch.zeros(batch_size, 64)
    out = film_identity(x_test, cond_zero)
    assert torch.allclose(out, x_test, atol=1e-5), "FiLM should be identity at init"
    print("FiLM layer test passed!")
    
    # Test full decoder
    decoder = HypernetDecoder(
        latent_dim=latent_dim,
        position_dim=32,
        hidden_dim=256,
        output_dim=9,
        num_layers=4,
    )
    
    latent = torch.randn(batch_size, num_neurons, latent_dim)
    positions = torch.randn(batch_size, num_neurons, 3).abs() * 5
    
    weights = decoder(latent, positions)
    print(f"Decoder output shape: {weights.shape}")
    assert weights.shape == (batch_size, num_neurons, 9)
    
    # Test gradient flow
    loss = weights.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in decoder.parameters() if p.requires_grad)
    print(f"Decoder gradients flow: {has_grad}")
    assert has_grad
    
    # Initial output should be near zero (due to zero init)
    decoder2 = HypernetDecoder(latent_dim=64, output_dim=9)
    latent2 = torch.randn(1, 10, 64)
    pos2 = torch.randn(1, 10, 3).abs() * 5
    out2 = decoder2(latent2, pos2)
    print(f"Initial output magnitude: {out2.abs().mean().item():.6f}")
    assert out2.abs().mean() < 0.1, "Initial output should be near zero"
    print("Decoder zero-init test passed!")
    
    print("\nAll decoder tests passed!")


def test_autoencoder():
    """Test weight autoencoder."""
    batch_size = 4
    num_neurons = 19
    
    ae = WeightAutoencoder(
        input_dim=9,
        latent_dim=64,
        hidden_dim=256,
    )
    
    weights = torch.randn(batch_size, num_neurons, 9)
    positions = torch.randn(batch_size, num_neurons, 3).abs() * 5
    
    reconstructed, latent = ae(weights, positions)
    
    print(f"Input shape: {weights.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    assert reconstructed.shape == weights.shape
    assert latent.shape == (batch_size, num_neurons, 64)
    
    # Test gradient flow through full model
    loss = (reconstructed - weights).pow(2).mean()
    loss.backward()
    
    enc_has_grad = all(p.grad is not None for p in ae.encoder.parameters() if p.requires_grad)
    dec_has_grad = all(p.grad is not None for p in ae.decoder.parameters() if p.requires_grad)
    print(f"Encoder gradients: {enc_has_grad}, Decoder gradients: {dec_has_grad}")
    
    print("\nAutoencoder test passed!")


def test_dual_latent_decoder():
    """Test DualLatentDecoder with different combination modes."""
    batch_size = 4
    num_neurons = 19
    latent_dim = 64
    
    for mode in ['concat', 'add', 'gated']:
        print(f"\nTesting DualLatentDecoder mode='{mode}'")
        
        decoder = DualLatentDecoder(
            behavioral_latent_dim=latent_dim,
            weight_latent_dim=latent_dim,
            hidden_dim=256,
            output_dim=9,
            combination_mode=mode,
        )
        
        z_behavior = torch.randn(batch_size, num_neurons, latent_dim)
        z_weight = torch.randn(batch_size, num_neurons, latent_dim)
        positions = torch.randn(batch_size, num_neurons, 3).abs() * 5
        
        weights = decoder(z_behavior, z_weight, positions)
        
        print(f"  Output shape: {weights.shape}")
        assert weights.shape == (batch_size, num_neurons, 9)
        
        # Test gradient flow
        loss = weights.sum()
        loss.backward()
        has_grad = all(p.grad is not None for p in decoder.parameters() if p.requires_grad)
        print(f"  Gradients flow: {has_grad}")
        assert has_grad
        
        # Reset gradients
        decoder.zero_grad()
    
    print("\nAll DualLatentDecoder tests passed!")


if __name__ == "__main__":
    test_decoder()
    test_autoencoder()
    test_dual_latent_decoder()
