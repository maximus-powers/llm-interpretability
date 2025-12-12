import torch
import torch.nn as nn
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


# we use build all the heads as MLPs
def _build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    dropout: float,
    activation: str,
    batch_norm: bool = False,
):
    activation_map = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(0.2),
    }
    if activation not in activation_map:
        raise ValueError(f"Unknown activation: {activation}")

    layers = []
    prev_dim = input_dim

    # build layers
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation_map[activation])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim

    return nn.Sequential(*layers)


class PatternClassificationHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_patterns: int,
        hidden_dims: List[int],
        dropout: float,
        activation: str,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_patterns = num_patterns
        if hidden_dims:
            self.mlp = _build_mlp(
                latent_dim, hidden_dims, dropout, activation, batch_norm
            )
            final_dim = hidden_dims[-1]
        else:
            self.mlp = nn.Identity()
            final_dim = latent_dim
        self.output_layer = nn.Linear(final_dim, num_patterns)
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.mlp(latent)
        logits = self.output_layer(features)
        return logits


class AccuracyPredictionHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        dropout: float,
        activation: str,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        if hidden_dims:
            self.mlp = _build_mlp(
                latent_dim, hidden_dims, dropout, activation, batch_norm
            )
            final_dim = hidden_dims[-1]
        else:
            self.mlp = nn.Identity()
            final_dim = latent_dim
        self.output_layer = nn.Sequential(nn.Linear(final_dim, 1), nn.Sigmoid())
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.mlp(latent)
        accuracy = self.output_layer(features)
        return accuracy


class HyperparameterPredictionHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        continuous_targets: Dict[str, Dict],
        discrete_targets: Dict[str, Dict],
        dropout: float,
        activation: str,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.continuous_targets = continuous_targets
        self.discrete_targets = discrete_targets

        if hidden_dims:
            self.shared_mlp = _build_mlp(
                latent_dim, hidden_dims, dropout, activation, batch_norm
            )
            final_dim = hidden_dims[-1]
        else:
            self.shared_mlp = nn.Identity()
            final_dim = latent_dim
        # continuous: one output head per target
        self.continuous_heads = nn.ModuleDict()
        for name, config in continuous_targets.items():
            self.continuous_heads[name] = nn.Sequential(nn.Linear(final_dim, 1))
        # discrete: one output head per target
        self.discrete_heads = nn.ModuleDict()
        for name, config in discrete_targets.items():
            num_classes = config.get("num_classes", len(config["values"]))
            self.discrete_heads[name] = nn.Linear(final_dim, num_classes)
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.shared_mlp(latent)
        outputs = {}
        # continuous targets
        for name, config in self.continuous_targets.items():
            raw_pred = self.continuous_heads[name](features)
            # scale to [min, max] range
            min_val = config["min"]
            max_val = config["max"]
            log_scale = config.get("log_scale", False)
            if log_scale:
                scaled_pred = torch.sigmoid(raw_pred)
                scaled_pred = min_val * ((max_val / min_val) ** scaled_pred)
            else:
                scaled_pred = torch.sigmoid(raw_pred) * (max_val - min_val) + min_val
            outputs[f"continuous_{name}"] = scaled_pred
        # discrete targets
        for name in self.discrete_targets.keys():
            logits = self.discrete_heads[name](features)
            outputs[f"discrete_{name}"] = logits
        return outputs


def create_prediction_head(
    task_type: str, latent_dim: int, task_config: Dict, model_config: Dict
):
    head_config = model_config["prediction_head"]
    hidden_dims = head_config.get("hidden_dims", [256, 128])
    dropout = head_config.get("dropout", 0.3)
    activation = head_config.get("activation", "relu")
    batch_norm = head_config.get("batch_norm", False)
    # default hidden dims
    if hidden_dims is None:
        hidden_dims = [max(latent_dim // 2, 64), max(latent_dim // 4, 32)]
        logger.info(f"Auto-computed hidden_dims: {hidden_dims}")

    if task_type == "pattern_classification":
        num_patterns = len(task_config["patterns"])
        logger.info(f"Creating PatternClassificationHead with {num_patterns} patterns")
        return PatternClassificationHead(
            latent_dim=latent_dim,
            num_patterns=num_patterns,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
        )

    elif task_type == "accuracy_prediction":
        logger.info("Creating AccuracyPredictionHead")
        return AccuracyPredictionHead(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
        )

    elif task_type == "hyperparameter_prediction":
        continuous_targets = task_config.get("continuous_targets", {})
        discrete_targets = task_config.get("discrete_targets", {})
        logger.info(
            f"Creating HyperparameterPredictionHead with {len(continuous_targets)} continuous and {len(discrete_targets)} discrete targets"
        )
        return HyperparameterPredictionHead(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            continuous_targets=continuous_targets,
            discrete_targets=discrete_targets,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
        )

    else:
        raise ValueError(f"Unknown task type: {task_type}")
