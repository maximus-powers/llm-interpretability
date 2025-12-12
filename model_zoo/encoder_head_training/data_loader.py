import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


class EncoderHeadDataset(Dataset):
    def __init__(
        self,
        dataset_repo_id: str,
        tokenizer,
        task_type: str,
        task_config: Dict[str, Any],
        split: str = "train",
        cache_dir: Optional[Path] = None,
        encoder: Optional[torch.nn.Module] = None,
        device: str = "cpu",
    ):
        self.dataset_repo_id = dataset_repo_id
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.task_config = task_config
        self.split = split
        self.cache_dir = cache_dir
        self.device = device

        self.dataset = load_dataset(dataset_repo_id, split=split)
        logger.info(f"Loaded {len(self.dataset)} samples")

        # precompute representations if encoder provided
        self.cached_representations = None
        if encoder is not None and cache_dir is not None:
            self._precompute_representations(encoder)

    def _precompute_representations(self, encoder: torch.nn.Module):
        cache_file = self.cache_dir / f"{self.split}_representations.pt"
        if cache_file.exists():  # load from disk
            self.cached_representations = torch.load(cache_file, map_location="cpu")
            return

        logger.info("Pre-computing encoder representations (this may take a while)...")
        encoder.eval()
        encoder.to(self.device)
        representations = []

        with torch.no_grad():
            for i, sample in enumerate(self.dataset):
                if i % 100 == 0:
                    logger.info(f"Processing {i}/{len(self.dataset)}")

                # tokenize
                weights_dict = sample["state_dict"]
                tokens, mask = self.tokenizer.tokenize(weights_dict)
                tokens = tokens.unsqueeze(0).to(self.device)  # [1, seq_len, token_dim]
                mask = mask.unsqueeze(0).to(self.device)  # [1, seq_len]
                latent = encoder(tokens, mask)  # [1, latent_dim]
                representations.append(latent.cpu().squeeze(0))

        self.cached_representations = torch.stack(representations)
        logger.info(
            f"Cached representations shape: {self.cached_representations.shape}"
        )
        if self.cache_dir:  # save to disk
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.cached_representations, cache_file)
            logger.info(f"Saved cached representations to {cache_file}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        target = self._prepare_target(sample)

        # return tokens
        if self.cached_representations is not None:
            return {"latent": self.cached_representations[idx], "target": target}
        else:
            weights_dict = sample["state_dict"]
            tokens, mask = self.tokenizer.tokenize(weights_dict)
            return {"tokens": tokens, "mask": mask, "target": target}

    def _prepare_target(self, sample: Dict[str, Any]):
        metadata = sample["metadata"]
        if self.task_type == "pattern_classification":
            # multi-label binary vector
            patterns = self.task_config["patterns"]
            target = torch.zeros(len(patterns), dtype=torch.float32)
            for i, pattern_name in enumerate(patterns):
                if pattern_name in metadata and metadata[pattern_name]:
                    target[i] = 1.0
            return target

        elif self.task_type == "accuracy_prediction":
            # scalar regression target
            accuracy = metadata.get("accuracy", 0.0)
            return torch.tensor([accuracy], dtype=torch.float32)

        elif self.task_type == "hyperparameter_prediction":
            # multi-target dictionary
            continuous_targets = self.task_config.get("continuous_targets", {})
            discrete_targets = self.task_config.get("discrete_targets", {})
            targets = {}

            # continuous targets
            for name, config in continuous_targets.items():
                value = metadata.get(name, config.get("default", 0.0))
                targets[f"continuous_{name}"] = torch.tensor(
                    [value], dtype=torch.float32
                )

            # discrete targets
            for name, config in discrete_targets.items():
                value = metadata.get(name)
                values = config["values"]
                if value in values:
                    class_idx = values.index(value)
                else:
                    class_idx = 0
                    logger.warning(f"Unknown value '{value}' for {name}, using default")
                targets[f"discrete_{name}"] = torch.tensor(class_idx, dtype=torch.long)

            return targets

        else:
            raise ValueError(f"Unknown task type: {self.task_type}")


def create_data_loaders(
    dataset_repo_id: str,
    tokenizer,
    task_type: str,
    task_config: Dict[str, Any],
    batch_size: int,
    cache_dir: Optional[Path] = None,
    encoder: Optional[torch.nn.Module] = None,
    device: str = "cpu",
    num_workers: int = 0,
):
    train_dataset = EncoderHeadDataset(
        dataset_repo_id,
        tokenizer,
        task_type,
        task_config,
        split="train",
        cache_dir=cache_dir,
        encoder=encoder,
        device=device,
    )
    val_dataset = EncoderHeadDataset(
        dataset_repo_id,
        tokenizer,
        task_type,
        task_config,
        split="validation",
        cache_dir=cache_dir,
        encoder=encoder,
        device=device,
    )
    test_dataset = EncoderHeadDataset(
        dataset_repo_id,
        tokenizer,
        task_type,
        task_config,
        split="test",
        cache_dir=cache_dir,
        encoder=encoder,
        device=device,
    )

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if "latent" in batch[0]:
            latents = torch.stack([item["latent"] for item in batch])
            targets = _collate_targets(batch, task_type)
            return {"latent": latents, "target": targets}
        else:
            tokens = torch.stack([item["tokens"] for item in batch])
            masks = torch.stack([item["mask"] for item in batch])
            targets = _collate_targets(batch, task_type)
            return {"tokens": tokens, "mask": masks, "target": targets}

    def _collate_targets(batch: List[Dict[str, Any]], task_type: str):
        if task_type in ["pattern_classification", "accuracy_prediction"]:
            return torch.stack([item["target"] for item in batch])
        elif task_type == "hyperparameter_prediction":
            targets = {}
            sample_target = batch[0]["target"]
            for key in sample_target.keys():
                if key.startswith("continuous_"):
                    targets[key] = torch.stack([item["target"][key] for item in batch])
                elif key.startswith("discrete_"):
                    targets[key] = torch.stack([item["target"][key] for item in batch])
            return targets
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    # create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
