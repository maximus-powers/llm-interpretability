import torch
import torch.nn as nn
import logging
import math
from typing import Dict, List, Any
from datasets import load_dataset

from .data_loader import infer_signature_dimensions

logger = logging.getLogger(__name__)


class WeightSpaceEncoderDecoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.latent_dim = config["architecture"]["latent_dim"]
        self.input_mode = config["dataset"]["input_mode"]
        tokenization_config = config["tokenization"]
        granularity = tokenization_config.get("granularity", "chunk")

        # Architecture bypass: metadata no longer in tokens, decoder always outputs weights only
        # Architecture info is passed directly to decoder via FiLM conditioning

        # determine token_dim (no metadata - architecture bypasses latent space)
        if granularity == "neuron":
            if self.input_mode == "signature":
                neuron_profile = config["dataset"].get("neuron_profile", {})
                features_per_neuron = neuron_profile.get("features_per_neuron")
                if features_per_neuron is None:
                    features_per_neuron = self._infer_features_per_neuron(config)

                logger.info(f"Signature mode: {features_per_neuron} features per neuron")
                self.token_dim = features_per_neuron  # No metadata

                # decoder outputs weights, not signatures
                max_neurons = config["dataset"]["max_dimensions"]["max_neurons_per_layer"]
                max_weights_per_neuron = max_neurons + 1  # incoming connections + bias
                self.weights_only_dim = max_weights_per_neuron
                self.decoder_token_dim = max_weights_per_neuron  # Always weights only

            elif self.input_mode == "weights":
                max_neurons = config["dataset"]["max_dimensions"]["max_neurons_per_layer"]
                max_weights_per_neuron = max_neurons + 1  # incoming connections + bias
                self.token_dim = max_weights_per_neuron  # No metadata
                self.weights_only_dim = max_weights_per_neuron
                self.decoder_token_dim = max_weights_per_neuron  # Always weights only

            elif self.input_mode == "both":
                neuron_profile = config["dataset"].get("neuron_profile", {})
                features_per_neuron = neuron_profile.get("features_per_neuron")
                if features_per_neuron is None:
                    features_per_neuron = self._infer_features_per_neuron(config)

                logger.info(f"Both mode: {features_per_neuron} features per neuron")
                max_neurons = config["dataset"]["max_dimensions"]["max_neurons_per_layer"]
                max_weights_per_neuron = max_neurons + 1
                self.token_dim = max_weights_per_neuron + features_per_neuron  # No metadata
                self.weights_only_dim = max_weights_per_neuron
                self.decoder_token_dim = max_weights_per_neuron  # Always weights only

            else:
                raise ValueError(f"Unknown input_mode: {self.input_mode}")

            self.max_tokens = tokenization_config["max_tokens"]

        else:
            # chunk-level tokenization
            chunk_size = tokenization_config["chunk_size"]
            self.weights_only_dim = chunk_size  # For chunk mode, weights are the chunk
            if self.input_mode == "signature" or self.input_mode == "both":
                # placeholders, gets updated by trainer
                self.token_dim = 1
                self.max_tokens = 1
                self.decoder_token_dim = chunk_size  # Always weights only
            elif self.input_mode == "weights":
                # tokenization dimensions (no metadata)
                self.token_dim = chunk_size
                self.max_tokens = tokenization_config["max_tokens"]
                self.decoder_token_dim = chunk_size  # Always weights only
            else:
                raise ValueError(f"Unknown input_mode: {self.input_mode}")

    def _infer_features_per_neuron(self, config: Dict[str, Any]) -> int:
        dataset_name = config["dataset"]["hf_dataset"]
        neuron_profile = config["dataset"].get("neuron_profile", {})
        method_names = neuron_profile.get("methods", [])
        if not method_names:
            raise ValueError(
                "neuron_profile.methods is required for signature/both mode. Specify which methods to extract (e.g., ['fourier'], ['mean', 'std'])."
            )
        logger.info(f"Inferring features_per_neuron from dataset '{dataset_name}'...")
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        first_example = next(iter(dataset))
        if "improved_signature" not in first_example:
            raise ValueError(
                f"Dataset '{dataset_name}' does not contain 'improved_signature' field. Cannot auto-infer features_per_neuron."
            )
        dims = infer_signature_dimensions(first_example["improved_signature"], method_names)
        features_per_neuron = dims["signature_features_per_neuron"]
        logger.info(
            f"Inferred features_per_neuron={features_per_neuron} from methods {dims['method_shapes']}"
        )
        return features_per_neuron

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(
        self, latent: torch.Tensor, arch_spec: Dict[str, Any], num_tokens: int
    ) -> torch.Tensor:
        """
        Decode latent representation to weight tokens.

        Args:
            latent: Behavior latent representation (batch, latent_dim)
            arch_spec: Architecture specification (bypasses latent space)
            num_tokens: Number of tokens to generate

        Returns:
            Reconstructed weight tokens (batch, num_tokens, decoder_token_dim)
        """
        raise NotImplementedError

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor, arch_spec: Dict[str, Any]
    ) -> torch.Tensor:
        latent = self.encode(tokens, mask)
        return self.decode(latent, arch_spec, tokens.size(1))


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # 3-layer MLP with BatchNorm (proven to work better for contrastive learning)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ArchitectureEncoder(nn.Module):
    """
    Encodes discrete architecture specification into a continuous embedding.

    This embedding conditions the decoder but NEVER enters the behavior latent space,
    ensuring steering vectors cannot corrupt architectural information.
    """

    def __init__(
        self,
        max_layers: int = 10,
        max_neurons: int = 256,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.max_layers = max_layers
        self.max_neurons = max_neurons
        self.embed_dim = embed_dim

        # Layer count embedding
        self.layer_count_embed = nn.Embedding(max_layers + 1, embed_dim // 4)

        # Neuron counts encoder (processes neurons_per_layer as a vector)
        self.neuron_encoder = nn.Sequential(
            nn.Linear(max_layers, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
        )

        # Final projection combining all architecture features
        # Input: layer_count_embed (embed_dim//4) + neuron_embed (embed_dim//2) + io_dims (2)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim // 4 + embed_dim // 2 + 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, arch_spec: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """
        Encode architecture specification into embedding.

        Args:
            arch_spec: Dictionary with num_layers, neurons_per_layer, input_dim, output_dim
            device: Device to create tensors on

        Returns:
            arch_embed: (1, embed_dim) tensor
        """
        # Layer count embedding
        num_layers = min(arch_spec["num_layers"], self.max_layers)
        layer_idx = torch.tensor([num_layers], device=device)
        layer_emb = self.layer_count_embed(layer_idx)  # (1, embed_dim//4)

        # Neurons per layer (pad/truncate to max_layers)
        neurons = arch_spec["neurons_per_layer"]
        neurons_padded = (neurons + [0] * self.max_layers)[: self.max_layers]
        neurons_tensor = torch.tensor(
            [neurons_padded], dtype=torch.float, device=device
        )
        neurons_normalized = neurons_tensor / self.max_neurons
        neuron_emb = self.neuron_encoder(neurons_normalized)  # (1, embed_dim//2)

        # Input/output dimensions (normalized)
        io_dims = torch.tensor(
            [
                [
                    arch_spec["input_dim"] / self.max_neurons,
                    arch_spec["output_dim"] / self.max_neurons,
                ]
            ],
            dtype=torch.float,
            device=device,
        )

        # Combine all features
        combined = torch.cat([layer_emb, neuron_emb, io_dims], dim=-1)
        return self.proj(combined)  # (1, embed_dim)


class CrossAttentionDecoderLayer(nn.Module):
    """
    Decoder layer with cross-attention only (no self-attention).

    Standard TransformerDecoderLayer has self-attention which destroys positional
    information by averaging all positions together. This layer skips self-attention
    and only uses cross-attention to memory, preserving position-specific information.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm_first = norm_first

        # Cross-attention (tgt attends to memory)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt: Target sequence [batch, seq_len, d_model]
            memory: Memory from encoder [batch, mem_len, d_model]

        Returns:
            Output sequence [batch, seq_len, d_model]
        """
        if self.norm_first:
            # Cross-attention with pre-norm
            tgt2 = self.norm1(tgt)
            tgt2, _ = self.cross_attn(tgt2, memory, memory)
            tgt = tgt + self.dropout1(tgt2)

            # Feedforward with pre-norm
            tgt2 = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout2(tgt2)
        else:
            # Cross-attention with post-norm
            tgt2, _ = self.cross_attn(tgt, memory, memory)
            tgt = self.norm1(tgt + self.dropout1(tgt2))

            # Feedforward with post-norm
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = self.norm2(tgt + self.dropout2(tgt2))

        return tgt


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Conditions decoder hidden states on architecture embedding:
        h' = gamma(arch) * h + beta(arch)

    This allows the decoder to generate architecture-appropriate outputs
    without the architecture information passing through the latent space.
    """

    def __init__(self, hidden_dim: int, arch_embed_dim: int):
        super().__init__()
        # Generate both scale (gamma) and shift (beta) from architecture embedding
        self.film_generator = nn.Linear(arch_embed_dim, hidden_dim * 2)
        self._init_identity()

    def _init_identity(self):
        """
        Initialize FiLM to be identity transform: gamma=1, beta=0.

        This ensures positional information flows through unchanged initially,
        preventing decoder collapse where all positions output identical values.
        """
        nn.init.zeros_(self.film_generator.weight)
        # Bias: first half = 1 (gamma), second half = 0 (beta)
        with torch.no_grad():
            half = self.film_generator.out_features // 2
            self.film_generator.bias[:half] = 1.0
            self.film_generator.bias[half:] = 0.0

    def forward(self, h: torch.Tensor, arch_embed: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            h: Hidden states from decoder layer (batch, seq_len, hidden_dim)
            arch_embed: Architecture embedding (batch, arch_embed_dim)

        Returns:
            Modulated hidden states (batch, seq_len, hidden_dim)
        """
        film_params = self.film_generator(arch_embed)  # (batch, hidden_dim * 2)
        gamma, beta = film_params.chunk(2, dim=-1)  # Each: (batch, hidden_dim)

        # Expand for sequence dimension
        gamma = gamma.unsqueeze(1)  # (batch, 1, hidden_dim)
        beta = beta.unsqueeze(1)  # (batch, 1, hidden_dim)

        return gamma * h + beta


class MLPEncoderDecoder(WeightSpaceEncoderDecoder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        mlp_config = config["architecture"]["mlp"]
        encoder_cfg = mlp_config["encoder"]
        decoder_cfg = mlp_config["decoder"]
        self.token_pooling = mlp_config.get("token_pooling", "mean")

        # build encoder
        if self.token_pooling == "flatten":
            encoder_input_dim = self.max_tokens * self.token_dim
        else:
            encoder_input_dim = self.token_dim
        self.encoder = self._build_mlp(
            input_dim=encoder_input_dim,
            hidden_dims=encoder_cfg["hidden_dims"],
            output_dim=self.latent_dim,
            dropout=encoder_cfg.get("dropout", 0.0),
            activation=encoder_cfg.get("activation", "relu"),
            use_batch_norm=encoder_cfg.get("batch_norm", False),
        )
        logger.info(
            f"Encoder: {encoder_input_dim} -> {encoder_cfg['hidden_dims']} -> {self.latent_dim}"
        )

        # build decoder
        decoder_output_dim = self.max_tokens * self.decoder_token_dim
        self.decoder = self._build_mlp(
            input_dim=self.latent_dim,
            hidden_dims=decoder_cfg["hidden_dims"],
            output_dim=decoder_output_dim,
            dropout=decoder_cfg.get("dropout", 0.0),
            activation=decoder_cfg.get("activation", "relu"),
            use_batch_norm=decoder_cfg.get("batch_norm", False),
        )
        logger.info(
            f"Decoder: {self.latent_dim} -> {decoder_cfg['hidden_dims']} -> {decoder_output_dim}"
        )

        self._init_weights()

    def _build_mlp(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ) -> nn.Module:
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(inplace=True))
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                activation = self.config["architecture"]["mlp"]["encoder"].get(
                    "activation", "relu"
                )
                if activation in ["relu", "leaky_relu"]:
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_in", nonlinearity="relu"
                    )
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.size(0)
        if self.token_pooling == "mean":
            masked_tokens = tokens * mask.unsqueeze(-1)
            sum_tokens = masked_tokens.sum(dim=1)
            count = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = sum_tokens / count
        elif self.token_pooling == "max":
            masked_tokens = tokens.clone()
            masked_tokens[mask == 0] = float("-inf")
            pooled = masked_tokens.max(dim=1)[0]
            pooled[pooled == float("-inf")] = 0
        elif self.token_pooling == "flatten":
            pooled = tokens.view(batch_size, -1)
        else:
            raise ValueError(f"Unknown pooling method: {self.token_pooling}")

        latent = self.encoder(pooled)

        return latent

    def decode(
        self, latent: torch.Tensor, arch_spec: Dict[str, Any], num_tokens: int
    ) -> torch.Tensor:
        # Note: MLP decoder doesn't use FiLM conditioning (arch_spec ignored)
        # This is for API consistency with TransformerEncoderDecoder
        batch_size = latent.size(0)
        flat_output = self.decoder(latent)
        reconstructed = flat_output.view(
            batch_size, self.max_tokens, self.decoder_token_dim
        )
        return reconstructed


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        latent_dim: int,
        max_tokens: int,
        encoder_cfg: Dict[str, Any],
        use_positional_encoding: bool = False,  # Default: off (architecture bypasses latent)
    ):
        super().__init__()
        self.d_model = encoder_cfg["d_model"]
        self.latent_dim = latent_dim
        self.max_tokens = max_tokens
        self.pooling_method = encoder_cfg.get("pooling", "mean")
        self.use_positional_encoding = use_positional_encoding
        self.token_projection = nn.Linear(token_dim, self.d_model)
        self.pos_encoding = None
        self.register_buffer("pos_encoding_buffer", torch.empty(0))
        if self.pooling_method == "cls_token":
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=encoder_cfg["num_heads"],
            dim_feedforward=encoder_cfg["dim_feedforward"],
            dropout=encoder_cfg["dropout"],
            activation=encoder_cfg.get("activation", "relu"),
            batch_first=True,
            norm_first=encoder_cfg.get("norm_first", True),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_cfg["num_layers"],
            enable_nested_tensor=False,  # MPS compatibility
        )
        self.latent_projection = nn.Linear(self.d_model, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.size(0)
        x = self.token_projection(tokens)

        # Only add positional encoding if enabled
        if self.use_positional_encoding:
            if self.pos_encoding is not None:
                x = x + self.pos_encoding[:, : tokens.size(1), :]
            elif self.pos_encoding_buffer.numel() > 0:
                x = x + self.pos_encoding_buffer[:, : tokens.size(1), :]

        if self.pooling_method == "cls_token":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            cls_mask = torch.ones(batch_size, 1, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        # encoding
        src_key_padding_mask = mask == 0
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # pooling
        if self.pooling_method == "mean":
            masked_encoded = encoded * mask.unsqueeze(-1)
            pooled = masked_encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(
                min=1
            )
        elif self.pooling_method == "max":
            masked_encoded = encoded.clone()
            masked_encoded[mask == 0] = float("-inf")
            pooled = masked_encoded.max(dim=1)[0]
            pooled[pooled == float("-inf")] = 0
        elif self.pooling_method == "cls_token":
            pooled = encoded[:, 0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        latent = self.latent_norm(self.latent_projection(pooled))
        return latent


class TransformerDecoder(nn.Module):
    """
    Transformer decoder with FiLM conditioning for architecture bypass.

    Architecture information bypasses the latent space and is injected via
    FiLM layers after each transformer decoder layer, ensuring steering
    vectors cannot corrupt architectural information.
    """

    def __init__(
        self,
        token_dim: int,
        latent_dim: int,
        max_tokens: int,
        decoder_cfg: Dict[str, Any],
        arch_encoder_cfg: Dict[str, Any] = None,
    ):
        super().__init__()
        self.d_model = decoder_cfg["d_model"]
        self.latent_dim = latent_dim
        self.token_dim = token_dim
        self.max_tokens = max_tokens
        self.num_layers = decoder_cfg["num_layers"]

        # Memory expansion: project latent to multiple memory tokens
        # This allows different output positions to attend to different information
        self.num_memory_tokens = decoder_cfg.get("num_memory_tokens", 8)
        self.latent_expansion = nn.Linear(latent_dim, self.d_model * self.num_memory_tokens)

        self.decoder_pos_encoding = None
        self.register_buffer("decoder_pos_encoding_buffer", torch.empty(0))

        # Architecture encoder for FiLM conditioning
        arch_cfg = arch_encoder_cfg or {}
        arch_embed_dim = arch_cfg.get("embed_dim", 64)
        self.max_neurons = arch_cfg.get("max_neurons", 256)
        self.arch_encoder = ArchitectureEncoder(
            max_layers=arch_cfg.get("max_layers", 10),
            max_neurons=self.max_neurons,
            embed_dim=arch_embed_dim,
        )

        # Architecture-aware position projection
        # Replaces generic learned positional embeddings with structure-aware positions
        num_position_features = 8  # layer_idx, neuron_idx, layer_width, fan_in, fan_out, global_pos, is_first, is_last
        self.position_projection = nn.Sequential(
            nn.Linear(num_position_features, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.d_model),
        )

        # Learnable query embeddings - gives each position a unique identity
        # This prevents positions with similar architecture features from collapsing
        # to similar outputs. These are SEPARATE from the latent (behavior) path,
        # so steering the latent won't affect position identities.
        self.query_embeddings = nn.Parameter(
            torch.randn(max_tokens, self.d_model) * 0.1
        )

        # Individual decoder layers with FiLM conditioning
        # Using CrossAttentionDecoderLayer instead of nn.TransformerDecoderLayer
        # because self-attention destroys positional information by averaging all positions
        self.decoder_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.decoder_layers.append(
                CrossAttentionDecoderLayer(
                    d_model=self.d_model,
                    nhead=decoder_cfg["num_heads"],
                    dim_feedforward=decoder_cfg["dim_feedforward"],
                    dropout=decoder_cfg["dropout"],
                    activation=decoder_cfg.get("activation", "relu"),
                    norm_first=decoder_cfg.get("norm_first", True),
                )
            )
            self.film_layers.append(FiLMLayer(self.d_model, arch_embed_dim))

        self.output_projection = nn.Linear(self.d_model, token_dim)

    def generate_arch_positions(
        self,
        arch_spec: Dict[str, Any],
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate position embeddings from architecture specification.

        Each position embedding encodes structural information:
        - Which layer this position belongs to
        - Which neuron within the layer
        - Layer width, fan-in, fan-out
        - Global position in sequence

        This provides much stronger positional signals than generic learned
        embeddings, preventing decoder collapse to uniform outputs.

        Args:
            arch_spec: Architecture specification dict
            num_tokens: Number of tokens to generate
            device: Target device

        Returns:
            position_embeddings: [num_tokens, d_model]
        """
        neurons_per_layer = arch_spec["neurons_per_layer"]
        num_layers = len(neurons_per_layer)

        position_features = []
        token_idx = 0

        for layer_idx, num_neurons in enumerate(neurons_per_layer):
            # Fan-in: inputs to this layer
            if layer_idx == 0:
                fan_in = arch_spec["input_dim"]
            else:
                fan_in = neurons_per_layer[layer_idx - 1]

            # Fan-out: outputs from this layer
            if layer_idx == num_layers - 1:
                fan_out = arch_spec["output_dim"]
            else:
                fan_out = neurons_per_layer[layer_idx + 1]

            for neuron_idx in range(num_neurons):
                features = [
                    layer_idx / max(num_layers - 1, 1),  # normalized layer index
                    neuron_idx / max(num_neurons - 1, 1),  # normalized neuron position
                    num_neurons / self.max_neurons,  # layer width
                    fan_in / self.max_neurons,  # input dimension
                    fan_out / self.max_neurons,  # output dimension
                    token_idx / max(num_tokens - 1, 1),  # global position
                    1.0 if layer_idx == 0 else 0.0,  # is first layer
                    1.0 if layer_idx == num_layers - 1 else 0.0,  # is last layer
                ]
                position_features.append(features)
                token_idx += 1
                if token_idx >= num_tokens:
                    break
            if token_idx >= num_tokens:
                break

        # Pad if needed (for architectures smaller than num_tokens)
        while len(position_features) < num_tokens:
            position_features.append([0.0] * 8)

        features_tensor = torch.tensor(
            position_features, dtype=torch.float, device=device
        )
        return self.position_projection(features_tensor)  # [num_tokens, d_model]

    def forward(
        self,
        latent: torch.Tensor,
        arch_spec: Dict[str, Any],
        num_tokens: int = None,
    ) -> torch.Tensor:
        """
        Decode latent representation with architecture conditioning.

        Args:
            latent: Behavior latent representation (batch, latent_dim)
            arch_spec: Architecture specification (bypasses latent space)
            num_tokens: Number of tokens to generate

        Returns:
            Reconstructed weight tokens (batch, num_tokens, token_dim)
        """
        batch_size = latent.size(0)
        device = latent.device
        if num_tokens is None:
            num_tokens = self.max_tokens

        # Encode architecture (separate pathway, unaffected by steering)
        arch_embed = self.arch_encoder(arch_spec, device)  # (1, arch_embed_dim)
        if arch_embed.size(0) == 1 and batch_size > 1:
            arch_embed = arch_embed.expand(batch_size, -1)  # (batch, arch_embed_dim)

        # Expand latent to multiple memory tokens for cross-attention
        # This allows different output positions to attend to different information
        memory = self.latent_expansion(latent)  # (batch, d_model * num_memory_tokens)
        memory = memory.view(batch_size, self.num_memory_tokens, self.d_model)  # (batch, num_memory_tokens, d_model)

        # Initialize target from architecture-aware position embeddings + learned query embeddings
        # arch_positions: structural info (layer, neuron index, fan-in/out)
        # query_embeddings: unique per-position identity (prevents similar positions from collapsing)
        arch_positions = self.generate_arch_positions(arch_spec, num_tokens, device)
        tgt = arch_positions + self.query_embeddings[:num_tokens]  # (num_tokens, d_model)
        tgt = tgt.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_tokens, d_model)

        # Decode with FiLM conditioning after each layer
        for decoder_layer, film_layer in zip(self.decoder_layers, self.film_layers):
            tgt = decoder_layer(tgt, memory)
            tgt = film_layer(tgt, arch_embed)  # Architecture conditioning

        reconstructed = self.output_projection(tgt)
        return reconstructed


class TransformerEncoderDecoder(WeightSpaceEncoderDecoder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        transformer_config = config["architecture"]["transformer"]
        encoder_cfg = transformer_config["encoder"]
        decoder_cfg = transformer_config["decoder"]

        # Architecture encoder config (for FiLM conditioning in decoder)
        arch_encoder_cfg = config["architecture"].get("arch_encoder", {})
        # Use dataset max_dimensions for architecture limits if not specified
        max_dims = config["dataset"].get("max_dimensions", {})
        if "max_layers" not in arch_encoder_cfg:
            arch_encoder_cfg["max_layers"] = max_dims.get("max_hidden_layers", 10)
        if "max_neurons" not in arch_encoder_cfg:
            arch_encoder_cfg["max_neurons"] = max_dims.get("max_neurons_per_layer", 256)

        # Encoder positional encoding: configurable, default off (architecture bypasses latent)
        use_positional_encoding = encoder_cfg.get("use_positional_encoding", False)

        self.encoder = TransformerEncoder(
            token_dim=self.token_dim,
            latent_dim=self.latent_dim,
            max_tokens=self.max_tokens,
            encoder_cfg=encoder_cfg,
            use_positional_encoding=use_positional_encoding,
        )
        self.decoder = TransformerDecoder(
            token_dim=self.decoder_token_dim,
            latent_dim=self.latent_dim,
            max_tokens=self.max_tokens,
            decoder_cfg=decoder_cfg,
            arch_encoder_cfg=arch_encoder_cfg,
        )

        # Setup positional encodings
        # Encoder: only if use_positional_encoding is True
        # Decoder: always needs positional encoding for target sequence
        pos_encoding_type = encoder_cfg.get("positional_encoding", "learned")
        logger.info(
            f"Encoder positional encoding: {'enabled' if use_positional_encoding else 'disabled'} "
            f"(type: {pos_encoding_type})"
        )

        if pos_encoding_type == "learned":
            if use_positional_encoding:
                self.encoder.pos_encoding = nn.Parameter(
                    torch.randn(1, self.max_tokens, encoder_cfg["d_model"]) * 0.02
                )
            # Decoder always uses positional encoding for target sequence
            # Use larger scale (0.1) to provide stronger positional signals
            self.decoder.decoder_pos_encoding = nn.Parameter(
                torch.randn(1, self.max_tokens, decoder_cfg["d_model"]) * 0.1
            )
        else:
            # Sinusoidal positional encodings
            if use_positional_encoding:
                self.encoder.register_buffer(
                    "pos_encoding_buffer",
                    self._create_sinusoidal_encoding(
                        self.max_tokens, encoder_cfg["d_model"]
                    ),
                    persistent=True,
                )
            # Decoder always uses positional encoding for target sequence
            self.decoder.register_buffer(
                "decoder_pos_encoding_buffer",
                self._create_sinusoidal_encoding(
                    self.max_tokens, decoder_cfg["d_model"]
                ),
                persistent=True,
            )

        logger.info(
            f"Encoder: d_model={encoder_cfg['d_model']}, heads={encoder_cfg['num_heads']}, "
            f"layers={encoder_cfg['num_layers']}, pooling={encoder_cfg.get('pooling', 'mean')}"
        )
        logger.info(
            f"Decoder: d_model={decoder_cfg['d_model']}, heads={decoder_cfg['num_heads']}, "
            f"layers={decoder_cfg['num_layers']}, arch_embed_dim={arch_encoder_cfg.get('embed_dim', 64)}"
        )

        self._init_weights()

    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "output_projection" in name or "latent_projection" in name:
                    nn.init.xavier_normal_(module.weight, gain=0.1)
                else:
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens, mask)

    def decode(
        self, latent: torch.Tensor, arch_spec: Dict[str, Any], num_tokens: int
    ) -> torch.Tensor:
        return self.decoder(latent, arch_spec, num_tokens)


def create_encoder_decoder(config: Dict[str, Any]) -> WeightSpaceEncoderDecoder:
    arch_type = config["architecture"]["type"]
    if arch_type == "mlp":
        return MLPEncoderDecoder(config)
    elif arch_type == "transformer":
        return TransformerEncoderDecoder(config)
    else:
        raise ValueError(
            f"Unknown architecture type: {arch_type}. Must be 'mlp' or 'transformer'"
        )
