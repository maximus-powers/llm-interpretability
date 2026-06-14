"""
Encoders for the MUAT hypernetwork architecture.

Two encoders that map to the same latent space:
1. WeightEncoder: weights -> latent (for training alignment)
2. SignatureEncoder: signatures -> latent (for inference)

Hypernetwork best practices applied:
- Layer normalization for stable training
- Residual connections for gradient flow
- Careful initialization
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class WeightEncoder(nn.Module):
    """
    Encodes weight tokens to behavioral latent space.
    
    Position-AGNOSTIC: only sees weight values, not layer/neuron position.
    This forces the latent to capture pure weight structure/behavior.
    
    Architecture follows hypernetwork best practices:
    - Layer normalization for stability
    - Residual connections for gradient flow
    - GELU activation (smoother gradients than ReLU)
    """
    
    def __init__(
        self,
        input_dim: int = 9,  # max_fan_in + 1
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection to latent
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Xavier init for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, weight_tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Encode weight tokens to latent space.
        
        Args:
            weight_tokens: [batch, num_neurons, input_dim] or [num_neurons, input_dim]
            mask: Optional [batch, num_neurons] or [num_neurons] validity mask
            
        Returns:
            latent: Same shape as input but last dim is latent_dim
        """
        # Handle both batched and unbatched inputs
        squeeze_batch = False
        if weight_tokens.dim() == 2:
            weight_tokens = weight_tokens.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            squeeze_batch = True
        
        # Project input
        x = self.input_proj(weight_tokens)
        x = self.input_norm(x)
        
        # Apply residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Project to latent
        latent = self.output_proj(x)
        latent = self.output_norm(latent)
        
        if squeeze_batch:
            latent = latent.squeeze(0)
        
        return latent


class SignatureEncoder(nn.Module):
    """
    Encodes activation signatures to behavioral latent space.
    
    Maps the same latent space as WeightEncoder, trained via alignment loss.
    Signatures are our universal behavioral fingerprint.
    
    Architecture follows hypernetwork best practices:
    - Layer normalization for stability
    - Residual connections for gradient flow
    - Separate handling of different signature types
    """
    
    def __init__(
        self,
        signature_dim: int = 17,  # mean, std, 5 fourier, 8 input_corr, 2 pre_act
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.signature_dim = signature_dim
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Linear(signature_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection to latent
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, signatures: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Encode signatures to latent space.
        
        Args:
            signatures: [batch, num_neurons, signature_dim] or [num_neurons, signature_dim]
            mask: Optional validity mask
            
        Returns:
            latent: Same shape as input but last dim is latent_dim
        """
        squeeze_batch = False
        if signatures.dim() == 2:
            signatures = signatures.unsqueeze(0)
            squeeze_batch = True
        
        # Project input
        x = self.input_proj(signatures)
        x = self.input_norm(x)
        
        # Apply residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Project to latent
        latent = self.output_proj(x)
        latent = self.output_norm(latent)
        
        if squeeze_batch:
            latent = latent.squeeze(0)
        
        return latent


# Behavioral feature indices (PURE behavioral - no weight/architectural info)
# Excludes input_correlations (indices 7-14) which are ~0.94 correlated with weights
BEHAVIORAL_INDICES = [0, 1, 2, 3, 4, 5, 6, 15, 16]  # 9 features
# [mean, std, fourier_0-4, pre_activation_mean, pre_activation_std]


class BehavioralEncoder(nn.Module):
    """
    Encodes PURE behavioral signatures to latent space.
    
    Only uses behavioral features (mean, std, fourier, pre_activation stats).
    Excludes input_correlations which are ~0.94 correlated with weights.
    
    This creates an INTERPRETABLE behavioral latent for representation engineering.
    """
    
    def __init__(
        self,
        behavioral_dim: int = 9,  # len(BEHAVIORAL_INDICES)
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.behavioral_dim = behavioral_dim
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Linear(behavioral_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection to latent
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, signatures: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Encode behavioral signatures to latent space.
        
        Args:
            signatures: [batch, num_neurons, 17] full signatures OR
                       [batch, num_neurons, 9] pre-extracted behavioral features
            mask: Optional validity mask
            
        Returns:
            latent: [batch, num_neurons, latent_dim]
        """
        squeeze_batch = False
        if signatures.dim() == 2:
            signatures = signatures.unsqueeze(0)
            squeeze_batch = True
        
        # Extract behavioral features if full signatures provided
        if signatures.shape[-1] == 17:
            signatures = signatures[..., BEHAVIORAL_INDICES]
        
        # Project input
        x = self.input_proj(signatures)
        x = self.input_norm(x)
        
        # Apply residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Project to latent
        latent = self.output_proj(x)
        latent = self.output_norm(latent)
        
        if squeeze_batch:
            latent = latent.squeeze(0)
        
        return latent


class ResidualMLPBlock(nn.Module):
    """
    Residual MLP block with pre-norm architecture.
    
    Follows transformer best practices:
    - Pre-normalization (LayerNorm before transformation)
    - Residual connection
    - Dropout for regularization
    - GELU activation
    """
    
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(self.norm(x))


def test_encoders():
    """Test that encoders have correct output shapes and gradients flow."""
    batch_size = 4
    num_neurons = 19
    
    # Test WeightEncoder
    weight_enc = WeightEncoder(input_dim=9, latent_dim=64)
    weights = torch.randn(batch_size, num_neurons, 9)
    weight_latent = weight_enc(weights)
    
    print(f"WeightEncoder input shape: {weights.shape}")
    print(f"WeightEncoder output shape: {weight_latent.shape}")
    assert weight_latent.shape == (batch_size, num_neurons, 64)
    
    # Test gradient flow
    loss = weight_latent.sum()
    loss.backward()
    
    has_grad = all(p.grad is not None for p in weight_enc.parameters())
    print(f"WeightEncoder gradients flow: {has_grad}")
    assert has_grad
    
    # Test SignatureEncoder
    sig_enc = SignatureEncoder(signature_dim=17, latent_dim=64)
    sigs = torch.randn(batch_size, num_neurons, 17)
    sig_latent = sig_enc(sigs)
    
    print(f"SignatureEncoder input shape: {sigs.shape}")
    print(f"SignatureEncoder output shape: {sig_latent.shape}")
    assert sig_latent.shape == (batch_size, num_neurons, 64)
    
    # Test gradient flow
    sig_enc.zero_grad()
    loss = sig_latent.sum()
    loss.backward()
    
    has_grad = all(p.grad is not None for p in sig_enc.parameters())
    print(f"SignatureEncoder gradients flow: {has_grad}")
    assert has_grad
    
    # Test unbatched input
    weights_unbatched = torch.randn(num_neurons, 9)
    latent_unbatched = weight_enc(weights_unbatched)
    print(f"Unbatched output shape: {latent_unbatched.shape}")
    assert latent_unbatched.shape == (num_neurons, 64)
    
    print("\nAll encoder tests passed!")


def test_behavioral_encoder():
    """Test BehavioralEncoder."""
    batch_size = 4
    num_neurons = 19
    
    # Test with full signatures (17 features)
    beh_enc = BehavioralEncoder(behavioral_dim=9, latent_dim=64)
    full_sigs = torch.randn(batch_size, num_neurons, 17)
    latent = beh_enc(full_sigs)
    
    print(f"BehavioralEncoder input shape (full): {full_sigs.shape}")
    print(f"BehavioralEncoder output shape: {latent.shape}")
    assert latent.shape == (batch_size, num_neurons, 64)
    
    # Test with pre-extracted behavioral features (9 features)
    beh_only = torch.randn(batch_size, num_neurons, 9)
    latent2 = beh_enc(beh_only)
    assert latent2.shape == (batch_size, num_neurons, 64)
    print(f"BehavioralEncoder input shape (behavioral only): {beh_only.shape}")
    
    # Test gradient flow
    loss = latent.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in beh_enc.parameters())
    print(f"BehavioralEncoder gradients flow: {has_grad}")
    assert has_grad
    
    print("\nBehavioralEncoder test passed!")


if __name__ == "__main__":
    test_encoders()
    test_behavioral_encoder()
