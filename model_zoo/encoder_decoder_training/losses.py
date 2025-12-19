import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional

from .encoder_decoder_model import ProjectionHead

logger = logging.getLogger(__name__)


class ReconstructionLoss:
    def __init__(self, config: Dict[str, Any], loss_type: str = "mse"):
        self.config = config
        self.loss_type = loss_type
        if loss_type not in ["mse", "mae", "cosine"]:
            raise ValueError(
                f"Unknown loss type: {loss_type}. Must be 'mse', 'mae', or 'cosine'"
            )

    def compute(
        self, predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_type == "mse":
            token_error = ((predicted - target) ** 2).mean(dim=-1)
        elif self.loss_type == "mae":
            token_error = torch.abs(predicted - target).mean(dim=-1)
        elif self.loss_type == "cosine":
            pred_norm = F.normalize(predicted, p=2, dim=-1)
            target_norm = F.normalize(target, p=2, dim=-1)
            cos_sim = (pred_norm * target_norm).sum(dim=-1)
            token_error = 1 - cos_sim
        masked_error = token_error * mask
        loss = masked_error.sum() / mask.sum().clamp(min=1)
        return loss


class CombinedReconstructionLoss:
    """Combines multiple reconstruction losses with weighted sum"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.losses = {}
        self.weights = {}
        for loss_spec in config["loss"]["components"]:
            loss_type = loss_spec["type"]
            weight = loss_spec["weight"]
            if loss_type in ["mse", "mae", "cosine"]:
                self.losses[loss_type] = ReconstructionLoss(config, loss_type=loss_type)
            else:
                raise ValueError(f"Unknown loss type in combined loss: {loss_type}")
            self.weights[loss_type] = weight

        logger.info(
            f"Combined loss initialized with components: {list(self.losses.keys())}"
        )

    def compute(
        self, predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ):
        total_loss = 0
        loss_components = {}
        for loss_name, loss_fn in self.losses.items():
            component_loss = loss_fn.compute(predicted, target, mask)
            loss_components[f"loss_{loss_name}"] = component_loss.item()
            total_loss += self.weights[loss_name] * component_loss

        return total_loss, loss_components


class SupervisedContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float,
        device: str,
        projection_head_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.device = device

        if projection_head_config is not None:
            self.projection_head = ProjectionHead(
                input_dim=projection_head_config["input_dim"],
                hidden_dim=projection_head_config["hidden_dim"],
                output_dim=projection_head_config["output_dim"],
            ).to(device)
        else:
            self.projection_head = None

    def _create_behavior_positive_mask(self, behavior_labels, device):
        batch_size = len(behavior_labels)

        # get all unique behaviors
        all_behaviors = set()
        for labels in behavior_labels:
            all_behaviors.update(labels)
        behavior_to_idx = {b: i for i, b in enumerate(all_behaviors)}
        num_behaviors = len(all_behaviors)

        # create binary label matrix: [batch_size, num_behaviors]
        label_matrix = torch.zeros(batch_size, num_behaviors, device=device)
        for i, labels in enumerate(behavior_labels):
            for label in labels:
                label_matrix[i, behavior_to_idx[label]] = 1.0

        # positive mask: samples share at least one behavior
        # (label_matrix @ label_matrix.T) > 0 means overlap exists
        overlap = torch.mm(label_matrix, label_matrix.T)  # [batch, batch]
        positive_mask = (overlap > 0).float()

        # remove self-connections
        positive_mask = positive_mask - torch.eye(batch_size, device=device)
        positive_mask = positive_mask.clamp(min=0)

        return positive_mask

    def forward(self, latents: torch.Tensor, behavior_labels) -> torch.Tensor:
        if behavior_labels is None:
            raise ValueError("behavior_labels required for supervised contrastive loss")

        if self.projection_head is not None:
            latents = self.projection_head(latents)

        device = latents.device
        batch_size = latents.size(0)

        # L2 normalize features (standard for contrastive learning)
        features = F.normalize(latents, p=2, dim=1)

        # compute similarity matrix: [batch, batch]
        sim_matrix = torch.mm(features, features.T) / self.temperature

        # create positive mask from behavior labels
        positive_mask = self._create_behavior_positive_mask(behavior_labels, device)

        # for numerical stability: subtract max from each row
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # compute log-sum-exp for denominator (all pairs except self)
        self_mask = torch.eye(batch_size, device=device)
        exp_sim = torch.exp(sim_matrix) * (1 - self_mask)  # mask out self
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # log probabilities: log(exp(sim_ij/T) / sum_k≠i exp(sim_ik/T))
        log_prob = sim_matrix - log_sum_exp

        # mean log prob over positive pairs for each anchor
        num_positives_per_anchor = positive_mask.sum(dim=1)  # [batch]
        has_positives = num_positives_per_anchor > 0

        if not has_positives.any():
            logger.warning("No positive pairs found in batch!")
            return torch.tensor(0.0, device=device)

        # sum of log_prob for positive pairs, divided by num positives
        masked_log_prob = log_prob * positive_mask
        sum_log_prob_pos = masked_log_prob.sum(dim=1)  # [batch]

        # only average over anchors that have positives
        mean_log_prob_pos = sum_log_prob_pos[has_positives] / num_positives_per_anchor[has_positives]

        # loss is negative mean log probability
        loss = -mean_log_prob_pos.mean()

        return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for behavioral clustering in latent space.

    Pulls together latent representations of models that share behavioral patterns.
    Uses supervised contrastive learning with behavior labels from dataset metadata.
    """

    def __init__(self, config: Dict[str, Any], device: str):
        super().__init__()
        contrast_cfg = config["loss"]["contrastive"]
        self.device = device

        temperature = contrast_cfg.get("temperature", 0.1)

        projection_head_config = None
        proj_cfg = contrast_cfg.get("projection_head")
        if proj_cfg is not None:
            projection_head_config = {
                "input_dim": proj_cfg["input_dim"],
                "hidden_dim": proj_cfg["hidden_dim"],
                "output_dim": proj_cfg["output_dim"],
            }

        self.loss_fn = SupervisedContrastiveLoss(
            temperature=temperature,
            device=device,
            projection_head_config=projection_head_config,
        )

        logger.info(f"ContrastiveLoss initialized: temperature={temperature}")

    def forward(self, latents: torch.Tensor, behavior_labels) -> torch.Tensor:
        """Compute contrastive loss on latent representations.

        Args:
            latents: Latent representations [batch_size, latent_dim]
            behavior_labels: List of behavior label sets for each sample

        Returns:
            Scalar contrastive loss tensor
        """
        return self.loss_fn(latents, behavior_labels)


# Keep old name as alias for backward compatibility
GammaContrastReconLoss = ContrastiveLoss


class FunctionalReconstructionLoss(nn.Module):
    """Loss that evaluates whether reconstructed weights produce functionally similar model outputs.

    Instead of just comparing token-level MSE, this loss:
    1. Detokenizes reconstructed weights into a state_dict
    2. Loads weights into SubjectModel instances
    3. Runs both original and reconstructed models on test inputs
    4. Computes MSE between their outputs

    This ensures the decoder learns to produce weights that actually work,
    not just weights that look similar at the token level.

    Uses a fixed benchmark dataset for consistent evaluation across training.
    """

    def __init__(self, config: Dict[str, Any], device: str):
        super().__init__()
        self.device = device
        func_cfg = config["loss"]["functional"]

        # test_samples: null = use all, 0.0-1.0 = percentage of benchmark
        self.test_samples_ratio = func_cfg.get("test_samples", None)

        # Load benchmark dataset for consistent test sequences
        benchmark_path = func_cfg.get("benchmark_path")
        self.benchmark_sequences = None
        self.benchmark_vocab_size = None

        if benchmark_path:
            import json
            try:
                with open(benchmark_path, "r") as f:
                    benchmark_data = json.load(f)

                # Extract sequences from benchmark
                sequences = []
                for example in benchmark_data["examples"]:
                    sequences.append(example["sequence"])

                # Store metadata
                metadata = benchmark_data["metadata"]
                self.benchmark_vocab_size = len(metadata["vocab"])
                self.benchmark_sequence_length = metadata["sequence_length"]

                # Convert sequences to tensor (one-hot or index based on vocab)
                # Sequences are lists of characters, convert to indices
                vocab = metadata["vocab"]
                char_to_idx = {c: i for i, c in enumerate(vocab)}

                indexed_sequences = []
                for seq in sequences:
                    indices = [char_to_idx.get(c, 0) for c in seq]
                    indexed_sequences.append(indices)

                self.benchmark_sequences = torch.tensor(indexed_sequences, dtype=torch.float32)

                # Calculate actual number of samples to use
                total_sequences = len(sequences)
                if self.test_samples_ratio is None:
                    self.n_test_samples = total_sequences
                else:
                    self.n_test_samples = max(1, int(total_sequences * self.test_samples_ratio))

                logger.info(
                    f"FunctionalReconstructionLoss: loaded {total_sequences} benchmark sequences, "
                    f"using {self.n_test_samples} ({self.test_samples_ratio or 1.0:.0%}) per forward pass"
                )
            except Exception as e:
                logger.warning(f"Failed to load benchmark dataset from {benchmark_path}: {e}")
                logger.warning("Falling back to random test inputs")
                self.n_test_samples = 32  # fallback default

        if self.benchmark_sequences is None:
            self.n_test_samples = 32  # default for random inputs
            logger.info(
                f"FunctionalReconstructionLoss initialized with random inputs: n_test_samples={self.n_test_samples}"
            )

    def forward(
        self,
        original_weights: Dict[str, torch.Tensor],
        reconstructed_weights: Dict[str, torch.Tensor],
        model_config: Dict[str, Any],
        test_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute functional reconstruction loss.

        Args:
            original_weights: Dict of original weight tensors (detached, no gradients needed)
            reconstructed_weights: Dict of reconstructed weight tensors (gradients preserved)
            model_config: Config dict for SubjectModel instantiation
            test_inputs: Optional test sequences. If None, uses benchmark or generates random.

        Returns:
            Scalar loss tensor (with gradients flowing back to reconstructed_weights)
        """
        # Import here to avoid circular imports
        from model_zoo.dataset_generation.models import SubjectModel

        vocab_size = model_config.get("vocab_size", 10)
        sequence_length = model_config.get("sequence_length", 5)

        # Determine test inputs to use (priority: benchmark > passed-in > random)
        if self.benchmark_sequences is not None:
            # Use benchmark sequences, sampling if we have more than n_test_samples
            if len(self.benchmark_sequences) > self.n_test_samples:
                indices = torch.randperm(len(self.benchmark_sequences))[: self.n_test_samples]
                test_inputs = self.benchmark_sequences[indices]
            else:
                test_inputs = self.benchmark_sequences
        elif test_inputs is None:
            # Generate random test inputs as fallback
            test_inputs = torch.randint(
                0, vocab_size, (self.n_test_samples, sequence_length)
            ).float()

        # Create model instances
        original_model = SubjectModel(
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            num_layers=model_config.get("num_layers", 6),
            neurons_per_layer=model_config.get("neurons_per_layer", 7),
            activation_type=model_config.get("activation_type", "relu"),
            dropout_rate=0.0,  # No dropout during evaluation
            precision="float32",
        ).to(self.device)

        recon_model = SubjectModel(
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            num_layers=model_config.get("num_layers", 6),
            neurons_per_layer=model_config.get("neurons_per_layer", 7),
            activation_type=model_config.get("activation_type", "relu"),
            dropout_rate=0.0,
            precision="float32",
        ).to(self.device)

        # Load original weights (no gradients needed)
        original_state = {k: v.detach().to(self.device) for k, v in original_weights.items()}
        original_model.load_state_dict(original_state)
        original_model.eval()

        # Load reconstructed weights - need to preserve gradients
        # We manually copy weights to the model's parameters
        recon_state = {}
        for name, param in recon_model.named_parameters():
            if name in reconstructed_weights:
                recon_state[name] = reconstructed_weights[name].to(self.device)
            else:
                # Keep initialized weights for missing keys
                recon_state[name] = param.data

        # Load state dict (this detaches by default, so we need a different approach)
        # Instead, directly assign to parameters to preserve gradients
        with torch.no_grad():
            for name, param in recon_model.named_parameters():
                if name in reconstructed_weights:
                    param.copy_(reconstructed_weights[name].to(self.device))

        # Now we need gradients to flow through. The issue is load_state_dict detaches.
        # Alternative approach: use functional_call to run the model with external weights
        # But for simplicity, we'll compute loss on the weights directly and use a proxy

        # Actually, the key insight is: we want the decoder to output weights that,
        # when loaded into a model, produce similar outputs to the original.
        # The gradient should flow: loss -> recon_output -> recon_weights -> decoder

        # To make this work, we need to manually construct the forward pass
        # using the reconstructed weights tensors directly (not via state_dict loading)

        test_inputs = test_inputs.to(self.device).float()

        # Get original model outputs (no gradients)
        with torch.no_grad():
            original_out = original_model(test_inputs)

        # For reconstructed model, we need to do a manual forward pass
        # to keep gradients flowing through the weights
        recon_out = self._forward_with_weights(recon_model, reconstructed_weights, test_inputs)

        # Numerical stability: clamp outputs to prevent inf
        recon_out = torch.clamp(recon_out, min=-1e6, max=1e6)

        # Check for inf/nan and return a large but finite loss if detected
        if torch.isnan(recon_out).any() or torch.isinf(recon_out).any():
            # Return a large loss to discourage this, but still allow gradients
            return torch.tensor(1e6, device=self.device, requires_grad=True)

        # Use smooth L1 loss for robustness to outliers
        loss = F.smooth_l1_loss(recon_out, original_out.detach())

        # Note: weight is now applied in trainer, not here
        return loss

    def _forward_with_weights(
        self,
        model: nn.Module,
        weights: Dict[str, torch.Tensor],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using external weights to preserve gradients.

        This performs the same computation as model.forward() but uses the
        weights from the dict directly, allowing gradients to flow back.
        """
        x = inputs

        # SubjectModel structure: network is Sequential with Linear, Activation, Dropout layers
        # Layer pattern: [Linear, Activation, (Dropout)] repeated, ending with Linear

        for i, module in enumerate(model.network):
            if isinstance(module, nn.Linear):
                # Get weight and bias for this layer
                weight_key = f"network.{i}.weight"
                bias_key = f"network.{i}.bias"

                if weight_key in weights and bias_key in weights:
                    w = weights[weight_key].to(inputs.device)
                    b = weights[bias_key].to(inputs.device)
                    x = F.linear(x, w, b)
                else:
                    # Fallback to module's own weights (log warning - this shouldn't happen)
                    logger.warning(
                        f"Missing weights in _forward_with_weights: {weight_key} or {bias_key}. "
                        f"Available keys: {list(weights.keys())[:5]}..."
                    )
                    x = module(x)

            elif isinstance(module, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):
                x = module(x)

            elif isinstance(module, nn.Dropout):
                # Skip dropout during functional evaluation
                pass

        return x
