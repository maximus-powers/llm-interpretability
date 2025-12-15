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

        # determine token_dim
        if granularity == "neuron":
            if self.input_mode == "signature":
                neuron_profile = config["dataset"].get("neuron_profile", {})
                features_per_neuron = neuron_profile.get("features_per_neuron")
                if features_per_neuron is None:
                    features_per_neuron = self._infer_features_per_neuron(config)

                logger.info(f"Signature mode: {features_per_neuron} features per neuron")

                if tokenization_config.get("include_metadata", True):
                    self.token_dim = features_per_neuron + 5  # signature + metadata
                else:
                    self.token_dim = features_per_neuron

                # decoder outputs weights, not signatures
                max_neurons = config["dataset"]["max_dimensions"]["max_neurons_per_layer"]
                max_weights_per_neuron = max_neurons + 1  # incoming connections + bias
                self.decoder_token_dim = max_weights_per_neuron + 5  # weights + metadata

            elif self.input_mode == "weights":
                max_neurons = config["dataset"]["max_dimensions"]["max_neurons_per_layer"]
                max_weights_per_neuron = max_neurons + 1  # incoming connections + bias
                self.token_dim = max_weights_per_neuron + 5  # 5 for metadata
                self.decoder_token_dim = self.token_dim

            elif self.input_mode == "both":
                neuron_profile = config["dataset"].get("neuron_profile", {})
                features_per_neuron = neuron_profile.get("features_per_neuron")
                if features_per_neuron is None:
                    features_per_neuron = self._infer_features_per_neuron(config)

                logger.info(f"Both mode: {features_per_neuron} features per neuron")
                max_neurons = config["dataset"]["max_dimensions"]["max_neurons_per_layer"]
                max_weights_per_neuron = max_neurons + 1
                self.token_dim = max_weights_per_neuron + features_per_neuron + 5
                self.decoder_token_dim = max_weights_per_neuron + 5

            else:
                raise ValueError(f"Unknown input_mode: {self.input_mode}")

            self.max_tokens = tokenization_config["max_tokens"]

        else:
            # chunk-level tokenization
            if self.input_mode == "signature" or self.input_mode == "both":
                # placeholders, gets updated by trainer
                self.token_dim = 1
                self.max_tokens = 1
                self.decoder_token_dim = 1
            elif self.input_mode == "weights":
                # tokenization dimensions
                self.token_dim = tokenization_config["chunk_size"]
                if tokenization_config.get("include_metadata", True):
                    self.token_dim += 5
                self.max_tokens = tokenization_config["max_tokens"]
                self.decoder_token_dim = self.token_dim
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
        if "signature" not in first_example:
            raise ValueError(
                f"Dataset '{dataset_name}' does not contain 'signature' field. Cannot auto-infer features_per_neuron."
            )
        dims = infer_signature_dimensions(first_example["signature"], method_names)
        features_per_neuron = dims["signature_features_per_neuron"]
        logger.info(
            f"Inferred features_per_neuron={features_per_neuron} from methods {dims['method_shapes']}"
        )
        return features_per_neuron

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, latent: torch.Tensor, num_tokens: int) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        latent = self.encode(tokens, mask)
        return self.decode(latent, tokens.size(1))


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


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

    def decode(self, latent: torch.Tensor, num_tokens: int) -> torch.Tensor:
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
    ):
        super().__init__()
        self.d_model = encoder_cfg["d_model"]
        self.latent_dim = latent_dim
        self.max_tokens = max_tokens
        self.pooling_method = encoder_cfg.get("pooling", "mean")
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
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_cfg["num_layers"],
            enable_nested_tensor=False,  # MPS compatibility
        )
        self.latent_projection = nn.Linear(self.d_model, latent_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.size(0)
        x = self.token_projection(tokens)
        if self.pos_encoding is not None:
            x = x + self.pos_encoding
        else:
            x = x + self.pos_encoding_buffer

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

        latent = self.latent_projection(pooled)
        return latent


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        latent_dim: int,
        max_tokens: int,
        decoder_cfg: Dict[str, Any],
    ):
        super().__init__()
        self.d_model = decoder_cfg["d_model"]
        self.latent_dim = latent_dim
        self.token_dim = token_dim
        self.max_tokens = max_tokens
        self.latent_expansion = nn.Linear(latent_dim, self.d_model)
        self.decoder_pos_encoding = None
        self.register_buffer("decoder_pos_encoding_buffer", torch.empty(0))

        # decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=decoder_cfg["num_heads"],
                dim_feedforward=decoder_cfg["dim_feedforward"],
                dropout=decoder_cfg["dropout"],
                activation=decoder_cfg.get("activation", "relu"),
                batch_first=True,
                norm_first=False,
            ),
            num_layers=decoder_cfg["num_layers"],
        )
        self.output_projection = nn.Linear(self.d_model, token_dim)

    def forward(self, latent: torch.Tensor, num_tokens: int = None) -> torch.Tensor:
        batch_size = latent.size(0)
        if num_tokens is None:
            num_tokens = self.max_tokens
        memory = self.latent_expansion(latent).unsqueeze(1)
        if self.decoder_pos_encoding is not None:
            tgt = self.decoder_pos_encoding[:, :num_tokens, :].expand(
                batch_size, -1, -1
            )
        else:
            tgt = self.decoder_pos_encoding_buffer[:, :num_tokens, :].expand(
                batch_size, -1, -1
            )

        # decode
        decoded = self.transformer_decoder(tgt, memory)
        reconstructed = self.output_projection(decoded)
        return reconstructed


class TransformerEncoderDecoder(WeightSpaceEncoderDecoder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        transformer_config = config["architecture"]["transformer"]
        encoder_cfg = transformer_config["encoder"]
        decoder_cfg = transformer_config["decoder"]
        self.encoder = TransformerEncoder(
            token_dim=self.token_dim,
            latent_dim=self.latent_dim,
            max_tokens=self.max_tokens,
            encoder_cfg=encoder_cfg,
        )
        self.decoder = TransformerDecoder(
            token_dim=self.decoder_token_dim,
            latent_dim=self.latent_dim,
            max_tokens=self.max_tokens,
            decoder_cfg=decoder_cfg,
        )

        # setup positional encodings (shared logic)
        pos_encoding_type = encoder_cfg.get("positional_encoding", "learned")
        logger.info(f"Positional encoding: {pos_encoding_type}")

        if pos_encoding_type == "learned":
            self.encoder.pos_encoding = nn.Parameter(
                torch.randn(1, self.max_tokens, encoder_cfg["d_model"]) * 0.02
            )
            self.decoder.decoder_pos_encoding = nn.Parameter(
                torch.randn(1, self.max_tokens, decoder_cfg["d_model"]) * 0.02
            )
        else:
            # sinusodal positional encodings
            self.encoder.register_buffer(
                "pos_encoding_buffer",
                self._create_sinusoidal_encoding(
                    self.max_tokens, encoder_cfg["d_model"]
                ),
                persistent=True,
            )
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
            f"layers={decoder_cfg['num_layers']}"
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens, mask)

    def decode(self, latent: torch.Tensor, num_tokens: int) -> torch.Tensor:
        return self.decoder(latent, num_tokens)


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
