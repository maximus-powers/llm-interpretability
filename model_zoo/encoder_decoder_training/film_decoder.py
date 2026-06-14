"""
FiLM-based decoder for weight generation.

The behavioral encoding modulates position processing via Feature-wise 
Linear Modulation (FiLM). This prevents decoder collapse by making 
behavior CONTROL how position is transformed, rather than competing with it.

Key insight: Instead of adding behavior + position (where one dominates),
behavior produces (gamma, beta) parameters that TRANSFORM position features:

    output = gamma(behavior) * position_features + beta(behavior)

This ensures both signals contribute - behavior can't be ignored because
it controls the transformation, and position can't be ignored because
it's the main signal being transformed.

Reference: Perez et al. (2017) - "FiLM: Visual Reasoning with a General 
Conditioning Layer" (arXiv:1709.07871)
"""

import logging
from typing import Dict, List, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InputCorrelationBypass(nn.Module):
    """
    Layer-aware bypass pathway that maps input_correlations to weight predictions.
    
    Key insight: input_correlations[i] measures correlation between input[i] and 
    neuron activation. This is highly predictive of the corresponding incoming weight,
    BUT the relationship strength varies dramatically by layer:
    
        Layer 0: R²=0.70, cosine=0.83 (direct relationship with raw inputs)
        Layer 1: R²=0.10, cosine=0.32 (degraded - depends on layer 0 weights)
        Layer 2+: R²<0.09, cosine<0.30 (weak relationship)
    
    This bypass uses layer-aware coefficients to account for this decay.
    The layer index is extracted from position_features[0] (normalized layer index)
    or position_features[6] (is_first_layer flag).
    
    Args:
        signature_dim: Total features per neuron in signature (17 for current config)
        input_corr_start_idx: Index where input_correlations start in feature vector (7)
        input_corr_dim: Number of input_correlation features (8)
        output_dim: Weight token dimension (9 = max_neurons + 1 for bias)
        num_layers: Maximum number of layers to support (default 8)
        freeze_params: If True, bypass parameters won't be updated during training
    """
    
    def __init__(
        self,
        signature_dim: int = 17,
        input_corr_start_idx: int = 7,  # After mean(1), std(1), fourier(5)
        input_corr_dim: int = 8,
        output_dim: int = 9,
        num_layers: int = 8,
        freeze_params: bool = True,  # Freeze by default to preserve empirical coefficients
    ):
        super().__init__()
        self.signature_dim = signature_dim
        self.input_corr_start_idx = input_corr_start_idx
        self.input_corr_dim = input_corr_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.freeze_params = freeze_params
        
        # Per-layer linear transforms: input_correlations → weight prediction
        # Empirically determined coefficients from linear regression per layer:
        #   Layer 0: slope=0.81, intercept=0.007 (strong relationship)
        #   Layer 1: slope=0.22, intercept=0.003 (weaker)
        #   Layer 2: slope=0.19, intercept=0.011 (weaker)
        #   Layer 3+: slope<0.15, intercept~0.01 (weak)
        
        # Empirical per-layer slopes (from data analysis)
        self.layer_slopes = [0.81, 0.22, 0.19, 0.12, 0.04, 0.0, 0.0, 0.0]
        self.layer_intercepts = [0.007, 0.003, 0.011, 0.017, 0.007, 0.01, 0.004, 0.0]
        
        # Create per-layer transforms
        self.transforms = nn.ModuleList([
            nn.Linear(input_corr_dim, output_dim) for _ in range(num_layers)
        ])
        
        # Initialize each layer with its empirical coefficients
        for layer_idx, transform in enumerate(self.transforms):
            slope = self.layer_slopes[layer_idx] if layer_idx < len(self.layer_slopes) else 0.0
            intercept = self.layer_intercepts[layer_idx] if layer_idx < len(self.layer_intercepts) else 0.0
            
            nn.init.zeros_(transform.weight)
            nn.init.constant_(transform.bias, intercept)
            
            # Set diagonal to the layer's slope
            with torch.no_grad():
                diag_size = min(input_corr_dim, output_dim)
                for i in range(diag_size):
                    transform.weight[i, i] = slope
        
        # Freeze parameters if requested
        if freeze_params:
            for param in self.parameters():
                param.requires_grad = False
            logger.info("  Bypass parameters FROZEN (requires_grad=False)")
        
        logger.info(
            f"InputCorrelationBypass (layer-aware): extracting features [{input_corr_start_idx}:{input_corr_start_idx+input_corr_dim}] "
            f"from signature_dim={signature_dim}, mapping to output_dim={output_dim}"
        )
        logger.info(f"  Per-layer slopes: {self.layer_slopes[:4]}...")
    
    def forward(
        self, 
        raw_signature: torch.Tensor,
        num_positions: int,
        position_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Extract input_correlations and predict weights using layer-aware transforms.
        
        Args:
            raw_signature: [batch, num_tokens, signature_dim] - raw signature features
            num_positions: Number of output weight positions
            position_features: [batch, num_positions, position_dim] - contains layer info
                             position_features[..., 0] = normalized layer index
                             position_features[..., 6] = 1.0 if layer 0, else 0.0
        
        Returns:
            [batch, num_positions, output_dim] weight contribution
        """
        batch_size = raw_signature.size(0)
        device = raw_signature.device
        
        # Extract input_correlations from signature
        # Shape: [batch, num_tokens, input_corr_dim]
        input_corr = raw_signature[:, :, self.input_corr_start_idx:self.input_corr_start_idx + self.input_corr_dim]
        
        # Ensure we have the right number of positions
        if input_corr.size(1) >= num_positions:
            input_corr = input_corr[:, :num_positions, :]
        else:
            # Pad with zeros if needed
            padding = torch.zeros(
                batch_size, num_positions - input_corr.size(1), self.input_corr_dim,
                device=device
            )
            input_corr = torch.cat([input_corr, padding], dim=1)
        
        # If we have position_features, use layer-aware transforms
        if position_features is not None:
            # Get layer indices from position_features
            # position_features[..., 0] = layer_idx / (num_layers - 1)
            # We need to convert back to integer layer indices
            # Assuming max 8 layers, normalized_layer_idx * 7 gives approximate layer
            normalized_layer_idx = position_features[:, :, 0]  # [batch, num_positions]
            
            # Also check is_first_layer flag for better layer 0 detection
            is_first_layer = position_features[:, :, 6]  # [batch, num_positions]
            
            # Initialize output
            weights = torch.zeros(batch_size, num_positions, self.output_dim, device=device)
            
            # Apply per-layer transforms
            # For efficiency, we batch by layer
            for layer_idx in range(self.num_layers):
                # Identify positions belonging to this layer
                if layer_idx == 0:
                    # Use is_first_layer flag for layer 0
                    layer_mask = is_first_layer > 0.5
                else:
                    # Approximate layer from normalized index
                    # normalized = layer_idx / (num_layers - 1)
                    # For 7 layers: layer 1 = 1/6 ≈ 0.167, layer 2 = 2/6 ≈ 0.333, etc.
                    lower_bound = (layer_idx - 0.5) / 7.0
                    upper_bound = (layer_idx + 0.5) / 7.0
                    layer_mask = (normalized_layer_idx > lower_bound) & (normalized_layer_idx <= upper_bound)
                    # Exclude layer 0 (handled separately)
                    layer_mask = layer_mask & (is_first_layer < 0.5)
                
                if layer_mask.any():
                    # Get input for this layer's positions
                    # We need to handle the mask carefully
                    layer_input = input_corr[layer_mask]  # [num_layer_tokens, input_corr_dim]
                    layer_output = self.transforms[layer_idx](layer_input)  # [num_layer_tokens, output_dim]
                    weights[layer_mask] = layer_output
            
            return weights
        else:
            # Fallback: use layer 0 transform for all (backward compatibility)
            return self.transforms[0](input_corr)


class DirectWeightPath(nn.Module):
    """
    Direct pathway from behavior encoding to weight contribution.
    
    This provides a "linear-like" shortcut for behavior information to directly
    affect weight predictions, similar to how Ridge regression achieves cosine
    sim = 0.289 by directly mapping input features to weights.
    
    The FiLM pathway is more expressive but harder to optimize. This direct
    path ensures behavioral info has a guaranteed route to influence outputs,
    preventing the decoder from ignoring behavior entirely.
    
    Output is broadcast across all positions (behavior affects all weights uniformly
    in this path - position-specific modulation comes from FiLM and bilinear paths).
    """
    
    def __init__(
        self,
        behavior_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 8,
    ):
        super().__init__()
        self.output_dim = output_dim
        
        # Deeper MLP with residual connections for better gradient flow
        self.input_proj = nn.Linear(behavior_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(2)  # 2 residual blocks
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize for stable training with meaningful output."""
        # Input projection: standard Xavier
        nn.init.xavier_normal_(self.input_proj.weight, gain=1.0)
        nn.init.zeros_(self.input_proj.bias)
        
        # Residual blocks: Xavier for first linear, zeros for second (residual)
        for block in self.layers:
            linear_idx = 0
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if linear_idx == 0:  # First linear in block
                        nn.init.xavier_normal_(module.weight, gain=1.0)
                    else:  # Second linear - init small for residual
                        nn.init.xavier_normal_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    linear_idx += 1
        
        # Output projection: standard init (not zeros - we want output early)
        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, behavior: torch.Tensor, num_positions: int) -> torch.Tensor:
        """
        Generate weight contribution from behavior.
        
        Args:
            behavior: [batch, behavior_dim]
            num_positions: Number of positions to broadcast to
        
        Returns:
            [batch, num_positions, output_dim] - broadcast weight contribution
        """
        h = self.input_proj(behavior)
        
        # Residual blocks
        for layer in self.layers:
            h = h + layer(h)  # Residual connection
        
        weight_contrib = self.output_proj(h)  # [batch, output_dim]
        
        # Broadcast to all positions
        return weight_contrib.unsqueeze(1).expand(-1, num_positions, -1)


class LearnableBaseWeights(nn.Module):
    """
    Learnable base weights for residual learning.
    
    Instead of predicting absolute weights from scratch, the decoder predicts
    DELTAS from these base weights:
    
        final_weights = base_weights + predicted_delta
    
    This is analogous to:
    - LoRA: low-rank adaptation starts from pretrained weights
    - Hypernetworks: often output delta-weights
    - ResNets: residual connections make optimization easier
    
    The base weights are initialized from dataset statistics (mean weights per
    position) so the model starts with a reasonable output and learns to adjust.
    
    With zero-initialized prediction heads, the model initially outputs just
    base_weights, which should already achieve decent reconstruction (since
    weights have structure that can be partially captured by position-dependent
    means).
    """
    
    def __init__(
        self,
        max_positions: int = 48,  # 6 layers * 8 neurons
        output_dim: int = 8,
    ):
        super().__init__()
        self.max_positions = max_positions
        self.output_dim = output_dim
        
        # Learnable base weights per position
        # Initialized to zeros; can be set from data via set_from_data()
        self.base_weights = nn.Parameter(torch.zeros(max_positions, output_dim))
    
    def set_from_data(self, mean_weights: torch.Tensor):
        """
        Initialize base weights from dataset statistics.
        
        Args:
            mean_weights: [num_positions, output_dim] mean weights per position
        """
        with torch.no_grad():
            # Handle size mismatch
            num_pos = min(mean_weights.size(0), self.max_positions)
            num_dim = min(mean_weights.size(1), self.output_dim)
            self.base_weights[:num_pos, :num_dim] = mean_weights[:num_pos, :num_dim]
    
    def forward(self, num_positions: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Get base weights for given batch.
        
        If num_positions > max_positions, base weights are repeated cyclically.
        This handles cases where max_tokens (e.g., 512) exceeds the typical
        number of neurons (e.g., 48) due to padding.
        
        Args:
            num_positions: Number of positions to return
            batch_size: Batch size
            device: Target device
        
        Returns:
            [batch, num_positions, output_dim] - base weights expanded for batch
        """
        base = self.base_weights.to(device)
        
        if num_positions <= self.max_positions:
            # Simple case: just slice
            base = base[:num_positions]
        else:
            # Need to extend: repeat base weights cyclically and slice
            # This handles padding positions beyond max_positions
            repeats = (num_positions // self.max_positions) + 1
            base = base.repeat(repeats, 1)[:num_positions]
        
        return base.unsqueeze(0).expand(batch_size, -1, -1)


class FiLMGenerator(nn.Module):
    """
    Generates FiLM modulation parameters (gamma, beta) from behavioral encoding.
    
    For each FiLM layer, produces:
    - gamma: Scale parameter (multiplicative)
    - beta: Shift parameter (additive)
    
    Initialized so that gamma=1, beta=0 (identity transform) at start.
    This means the model starts by passing position through unchanged,
    and learns how behavior should modulate it during training.
    """
    
    def __init__(
        self,
        behavior_dim: int = 128,
        hidden_dim: int = 256,
        num_film_layers: int = 3,
    ):
        super().__init__()
        self.num_film_layers = num_film_layers
        self.hidden_dim = hidden_dim
        
        # Shared trunk for processing behavior
        self.trunk = nn.Sequential(
            nn.Linear(behavior_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Per-layer heads for (gamma, beta)
        self.film_heads = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * 2)  # gamma and beta concatenated
            for _ in range(num_film_layers)
        ])
        
        self._init_film_params()
    
    def _init_film_params(self):
        """Initialize FiLM to identity transform.
        
        With (1 + γ) * x + β formulation:
        - γ = 0 means scale by 1 (identity)
        - β = 0 means no shift
        """
        for i in range(len(self.film_heads)):
            head = cast(nn.Linear, self.film_heads[i])
            nn.init.zeros_(head.weight)
            # Both gamma and beta init to 0 for identity transform
            # since we use (1 + gamma) * x + beta
            if head.bias is not None:
                head.bias.data[:self.hidden_dim] = 0.0  # gamma = 0 -> scale = 1
                head.bias.data[self.hidden_dim:] = 0.0  # beta = 0 -> no shift
    
    def forward(self, behavior: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate FiLM parameters from behavioral encoding.
        
        Args:
            behavior: [batch, behavior_dim] - behavioral latent from encoder
        
        Returns:
            gammas: List of [batch, hidden_dim] scale parameters per layer
            betas: List of [batch, hidden_dim] shift parameters per layer
        """
        h = self.trunk(behavior)
        
        gammas = []
        betas = []
        for head in self.film_heads:
            params = head(h)
            gamma, beta = params.chunk(2, dim=-1)
            gammas.append(gamma)
            betas.append(beta)
        
        return gammas, betas


class FiLMBlock(nn.Module):
    """
    Single FiLM-modulated processing block.
    
    Pattern: Linear -> LayerNorm -> FiLM(gamma, beta) -> GELU
    
    The normalization before FiLM ensures stable statistics,
    then FiLM can scale/shift to any range needed.
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        gamma: torch.Tensor, 
        beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM-modulated transformation.
        
        Args:
            x: [batch, num_positions, hidden_dim] - position features
            gamma: [batch, hidden_dim] - scale parameter (will be broadcast)
            beta: [batch, hidden_dim] - shift parameter (will be broadcast)
        
        Returns:
            [batch, num_positions, hidden_dim] - modulated features
        """
        h = self.linear(x)
        h = self.norm(h)
        
        # Broadcast gamma/beta across positions
        # gamma, beta: [batch, hidden_dim] -> [batch, 1, hidden_dim]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        # FiLM modulation with StyleGAN-style scaling: (1 + γ) * x + β
        # This prevents gamma from zeroing out the signal:
        # - γ = 0 means identity transform (no scaling)
        # - γ > 0 means amplify
        # - γ < 0 means attenuate (but never zero out completely)
        h = (1 + gamma) * h + beta
        h = F.gelu(h)
        
        return h


class BilinearPath(nn.Module):
    """
    Bilinear interaction between behavior and position.
    
    Ensures both signals contribute via multiplicative interaction,
    preventing the model from ignoring one conditioning signal.
    
    If the model tried to ignore behavior, b * p would zero out.
    If the model tried to ignore position, b * p would be constant.
    Both must contribute for meaningful output.
    """
    
    def __init__(
        self,
        behavior_dim: int = 128,
        position_dim: int = 256,
        output_dim: int = 64,
    ):
        super().__init__()
        self.behavior_proj = nn.Linear(behavior_dim, output_dim)
        self.position_proj = nn.Linear(position_dim, output_dim)
    
    def forward(
        self,
        behavior: torch.Tensor,
        position_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute bilinear interaction.
        
        Args:
            behavior: [batch, behavior_dim]
            position_features: [batch, num_positions, position_dim]
        
        Returns:
            [batch, num_positions, output_dim] - multiplicative interaction
        """
        b = self.behavior_proj(behavior).unsqueeze(1)  # [batch, 1, output_dim]
        p = self.position_proj(position_features)       # [batch, num_pos, output_dim]
        
        # Element-wise multiplication - both must contribute
        return b * p


class AuxiliaryHeads(nn.Module):
    """
    Auxiliary prediction heads to prevent conditioning collapse.
    
    These heads try to predict the input conditioning signals from
    the generated weights. If the decoder ignores a conditioning signal,
    the corresponding head cannot predict it, and loss increases.
    
    - behavior_head: Predicts behavior from pooled weights
    - position_head: Predicts position features from per-position weights
    """
    
    def __init__(
        self,
        weight_dim: int,
        behavior_dim: int = 128,
        position_dim: int = 8,
    ):
        super().__init__()
        
        # Predict behavior from weights (proves weights encode behavior)
        self.behavior_head = nn.Sequential(
            nn.Linear(weight_dim, 128),
            nn.GELU(),
            nn.Linear(128, behavior_dim),
        )
        
        # Predict position from weights (proves weights encode position)
        self.position_head = nn.Sequential(
            nn.Linear(weight_dim, 64),
            nn.GELU(),
            nn.Linear(64, position_dim),
        )
    
    def predict_behavior(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Predict behavior from pooled weights.
        
        Args:
            weights: [batch, num_positions, weight_dim]
        
        Returns:
            [batch, behavior_dim] - predicted behavioral encoding
        """
        pooled = weights.mean(dim=1)  # Pool across positions
        return self.behavior_head(pooled)
    
    def predict_position(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Predict position features from per-position weights.
        
        Args:
            weights: [batch, num_positions, weight_dim]
        
        Returns:
            [batch, num_positions, position_dim] - predicted position features
        """
        return self.position_head(weights)


class FiLMDecoder(nn.Module):
    """
    FiLM-based decoder for weight generation with per-token encoder features.
    
    CRITICAL INSIGHT: The linear baseline achieves cosine_sim=0.289 because it
    uses PER-NEURON input_correlations (8 values per neuron) to predict that
    neuron's weights. Our previous FiLM decoder only used the POOLED behavioral
    latent, losing all per-position information.
    
    This version accepts per-token encoder features (one for each neuron) and
    combines them with position features for weight prediction.
    
    Architecture:
    1. **Per-Token Path**: encoder_features[i] → weight contribution for position i
       This is the key path - uses per-position behavioral info like linear regression
    2. **FiLM Path**: pooled behavior modulates position processing
    3. **Direct Path**: pooled behavior → broadcast weight contribution (optional)
    4. **Bilinear Path**: behavior × position interaction
    
    Forward accepts:
    - behavior: [batch, behavior_dim] - pooled latent (for global modulation)
    - position_features: [batch, num_positions, position_dim] - structural info
    - encoder_features: [batch, num_positions, encoder_dim] - per-token encoded signatures
    
    Final output:
        weights = per_token_path(encoder_features) + film_path(position, behavior) + ...
    """
    
    def __init__(
        self,
        behavior_dim: int = 128,
        position_dim: int = 8,
        hidden_dim: int = 256,
        num_film_layers: int = 3,
        bilinear_dim: int = 64,
        output_dim: int = 8,  # max weights per neuron (fan-in)
        max_positions: int = 48,  # 6 layers * 8 neurons
        use_direct_path: bool = True,
        use_base_weights: bool = True,
        direct_hidden_dim: int = 256,
        encoder_dim: int = 256,  # Dimension of per-token encoder features
        use_encoder_features: bool = True,  # Whether to use per-token features
        film_delta_disabled: bool = False,  # If True, FiLM delta pathway outputs zero (bypass-only mode)
    ):
        super().__init__()
        
        self.behavior_dim = behavior_dim
        self.position_dim = position_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_positions = max_positions
        self.use_direct_path = use_direct_path
        self.use_base_weights = use_base_weights
        self.encoder_dim = encoder_dim
        self.use_encoder_features = use_encoder_features
        self.input_corr_bypass = None  # Will be set if enabled
        self.film_delta_disabled = film_delta_disabled  # Bypass-only mode
        
        logger.info(
            f"Initializing FiLMDecoder: behavior_dim={behavior_dim}, "
            f"position_dim={position_dim}, hidden_dim={hidden_dim}, "
            f"num_film_layers={num_film_layers}, output_dim={output_dim}, "
            f"use_direct_path={use_direct_path}, use_base_weights={use_base_weights}, "
            f"use_encoder_features={use_encoder_features}, film_delta_disabled={film_delta_disabled}"
        )
        if film_delta_disabled:
            logger.info("  BYPASS-ONLY MODE: FiLM delta pathway disabled")
        
        # === CRITICAL: Per-token encoder feature pathway ===
        # This is analogous to linear regression: per-neuron info → per-neuron weights
        # NOTE: No LayerNorm! LayerNorm biases shift output and degrade bypass signal.
        if use_encoder_features:
            self.encoder_feature_path = nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
            # Zero-initialize ALL layers so this path starts as complete no-op
            # This is critical: we want bypass to dominate until encoder learns useful features
            for module in self.encoder_feature_path:
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)
            
            # Learnable gate: controls how much encoder_feature_path contributes
            # Initialized to -5 so sigmoid(-5) ≈ 0.007, making bypass dominate initially
            # As training progresses, gate can open if encoder learns useful features
            self.encoder_feature_gate = nn.Parameter(torch.tensor(-5.0))
            logger.info(f"  Per-token encoder feature path enabled: {encoder_dim} -> {output_dim} (gated, zero-init)")
        else:
            self.encoder_feature_path = None
            self.encoder_feature_gate = None
        
        # Note: InputCorrelationBypass is initialized separately via enable_input_corr_bypass()
        # because it needs to know the signature feature layout
        
        # === Direct path (behavior → weights shortcut) ===
        if use_direct_path:
            self.direct_path = DirectWeightPath(
                behavior_dim=behavior_dim,
                hidden_dim=direct_hidden_dim,
                output_dim=output_dim,
            )
            logger.info(f"  Direct path enabled: hidden_dim={direct_hidden_dim}")
        else:
            self.direct_path = None
        
        # === Learnable base weights (residual learning) ===
        if use_base_weights:
            self.base_weights = LearnableBaseWeights(
                max_positions=max_positions,
                output_dim=output_dim,
            )
            logger.info(f"  Base weights enabled: max_positions={max_positions}")
        else:
            self.base_weights = None
        
        # === FiLM pathway ===
        # FiLM generator: behavior -> modulation parameters
        self.film_generator = FiLMGenerator(
            behavior_dim=behavior_dim,
            hidden_dim=hidden_dim,
            num_film_layers=num_film_layers,
        )
        
        # Position encoder: project position features to hidden dim
        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # FiLM-modulated blocks
        self.film_blocks = nn.ModuleList([
            FiLMBlock(hidden_dim) for _ in range(num_film_layers)
        ])
        
        # Bilinear interaction path (collapse prevention)
        self.bilinear = BilinearPath(
            behavior_dim=behavior_dim,
            position_dim=hidden_dim,  # Takes processed position features
            output_dim=bilinear_dim,
        )
        
        # Output head: FiLM output + bilinear output -> weight DELTA
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + bilinear_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Auxiliary heads for collapse prevention
        self.aux_heads = AuxiliaryHeads(
            weight_dim=output_dim,
            behavior_dim=behavior_dim,
            position_dim=position_dim,
        )
        
        # Initialize output layers
        self._zero_init_output_layers()
    
    def _zero_init_output_layers(self):
        """
        Initialize output layers with zero weights for stable training.
        
        When input_corr_bypass is enabled, we want it to dominate initially.
        Zero-initializing the FiLM output head ensures other pathways start as no-ops.
        
        This prevents the model from outputting large random values early in training
        and allows the bypass signal to be the primary output initially.
        """
        # Get the final linear layer of output_head
        for module in reversed(list(self.output_head.modules())):
            if isinstance(module, nn.Linear):
                # Always zero-init to let bypass dominate (if enabled)
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                logger.info("  Zero-initialized output head final layer (bypass-friendly)")
                break
    
    def enable_input_corr_bypass(
        self,
        signature_dim: int = 17,
        input_corr_start_idx: int = 7,
        input_corr_dim: int = 8,
        num_layers: int = 8,
        freeze_params: bool = True,
    ):
        """
        Enable the layer-aware input correlation bypass pathway.
        
        This adds a direct pathway from input_correlations in the raw signature
        to weight predictions, bypassing the transformer encoder. Uses per-layer
        coefficients because the relationship strength varies by layer:
        
            Layer 0: R²=0.70, cosine=0.83 (strong)
            Layer 1+: R²<0.10, cosine<0.32 (degraded)
        
        Args:
            signature_dim: Total features per neuron in signature
            input_corr_start_idx: Index where input_correlations start
            input_corr_dim: Number of input_correlation features
            num_layers: Maximum number of layers to support
            freeze_params: If True, bypass parameters are frozen (recommended)
        """
        self.input_corr_bypass = InputCorrelationBypass(
            signature_dim=signature_dim,
            input_corr_start_idx=input_corr_start_idx,
            input_corr_dim=input_corr_dim,
            output_dim=self.output_dim,
            num_layers=num_layers,
            freeze_params=freeze_params,
        )
        # Move to same device as other parameters
        device = next(self.parameters()).device
        self.input_corr_bypass = self.input_corr_bypass.to(device)
        logger.info(f"  Layer-aware input correlation bypass enabled (frozen={freeze_params})")
    
    def forward(
        self,
        behavior: torch.Tensor,
        position_features: torch.Tensor,
        encoder_features: torch.Tensor = None,
        raw_signature: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate weights from behavior encoding, position features, and per-token encoder features.
        
        Combines multiple pathways:
        1. input_corr_bypass: raw input_correlations → weights (NEW, highly predictive)
        2. encoder_feature_path: per-token encoder features → per-position weights
        3. base_weights: learnable per-position base (if enabled)
        4. direct_path: behavior → weights shortcut (if enabled)
        5. film_path: position modulated by behavior
        6. bilinear_path: behavior × position interaction
        
        Args:
            behavior: [batch, behavior_dim] - pooled behavioral encoding from encoder
            position_features: [batch, num_positions, position_dim] - structural position info
            encoder_features: [batch, num_positions, encoder_dim] - per-token encoder outputs
                             This is CRITICAL for per-position weight prediction!
            raw_signature: [batch, num_tokens, signature_dim] - raw signature input (for bypass)
        
        Returns:
            weights: [batch, num_positions, output_dim] - generated weight values
        """
        batch_size = behavior.size(0)
        num_positions = position_features.size(1)
        device = behavior.device
        
        # === NEW: Input correlation bypass (most direct path) ===
        # Extracts input_correlations from raw signature and maps to weights
        # Uses layer-aware transforms based on position_features
        if self.input_corr_bypass is not None and raw_signature is not None:
            input_corr_contrib = self.input_corr_bypass(raw_signature, num_positions, position_features)
        else:
            input_corr_contrib = 0.0
        
        # === Per-token encoder features → per-position weights ===
        # This goes through the transformer encoder
        # GATED: encoder_contrib = sigmoid(gate) * encoder_path(features)
        # Gate initialized to -5 so sigmoid(-5) ≈ 0.007 → bypass dominates initially
        if self.encoder_feature_path is not None and encoder_features is not None:
            raw_encoder_contrib = self.encoder_feature_path(encoder_features)
            gate_value = torch.sigmoid(self.encoder_feature_gate)
            encoder_contrib = gate_value * raw_encoder_contrib
        else:
            encoder_contrib = 0.0
        
        # === Base weights (residual learning) ===
        if self.base_weights is not None:
            base = self.base_weights(num_positions, batch_size, device)
        else:
            base = 0.0
        
        # === Direct path (behavior → weights shortcut) ===
        if self.direct_path is not None:
            direct_contrib = self.direct_path(behavior, num_positions)
        else:
            direct_contrib = 0.0
        
        # === FiLM path (position modulated by behavior) ===
        # If film_delta_disabled, skip the FiLM computation entirely (bypass-only mode)
        if self.film_delta_disabled:
            film_delta = 0.0
        else:
            # Generate FiLM parameters from behavior
            gammas, betas = self.film_generator(behavior)
            
            # Encode positions
            h = self.position_encoder(position_features)
            
            # Apply FiLM-modulated blocks
            for i, block in enumerate(self.film_blocks):
                h = block(h, gammas[i], betas[i])
            
            # Bilinear interaction
            bilinear_out = self.bilinear(behavior, h)
            
            # Combine FiLM + bilinear → delta
            combined = torch.cat([h, bilinear_out], dim=-1)
            film_delta = self.output_head(combined)
        
        # === Combine all pathways ===
        # weights = input_corr + encoder + base + direct + film_delta
        weights = input_corr_contrib + encoder_contrib + base + direct_contrib + film_delta
        
        return weights
    
    def compute_auxiliary_loss(
        self,
        weights: torch.Tensor,
        behavior: torch.Tensor,
        position_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute auxiliary loss to prevent conditioning collapse.
        
        If the decoder ignores behavior, behavior_loss increases.
        If the decoder ignores position, position_loss increases.
        
        Args:
            weights: [batch, num_positions, output_dim] - generated weights
            behavior: [batch, behavior_dim] - input behavioral encoding
            position_features: [batch, num_positions, position_dim] - input positions
        
        Returns:
            total_aux_loss: Scalar tensor
            loss_components: Dict with individual loss values for logging
        """
        # Predict behavior from generated weights
        pred_behavior = self.aux_heads.predict_behavior(weights)
        behavior_loss = F.mse_loss(pred_behavior, behavior.detach())
        
        # Predict position from generated weights
        pred_position = self.aux_heads.predict_position(weights)
        position_loss = F.mse_loss(pred_position, position_features.detach())
        
        total_loss = behavior_loss + position_loss
        
        components = {
            "aux_behavior": behavior_loss.item(),
            "aux_position": position_loss.item(),
        }
        
        return total_loss, components
