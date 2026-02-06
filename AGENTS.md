# AGENTS.md - AI Coding Agent Instructions

This document provides essential information for AI coding agents working in this repository.

## Project Overview

**MUAT (Meta Universal Activation Theory)** - A Python ML research project for weight-space learning and neural network interpretability. Trains "interpreter" models that understand neural network weights via activation signatures.

**Tech Stack:** Python 3.10+, PyTorch (>=2.0.0), HuggingFace, pip, Ruff (>=0.14.9), YAML configs

## Repository Structure

```
muat/
├── model_zoo/                    # Main Python package
│   ├── cli.py                    # Unified CLI entry point
│   ├── requirements.txt
│   ├── configs/                  # YAML configuration files
│   ├── dataset_generation/       # Subject model + signature dataset generation
│   ├── classification_training/  # Pattern classifier training
│   ├── encoder_decoder_training/ # Weight-space autoencoder training
│   └── representation_engineering/ # Steering vector computation
└── docs/                         # Research documentation
```

## Build & Run Commands

```bash
# Installation
cd model_zoo && pip install -r requirements.txt

# Interactive mode
python cli.py

# Dataset generation
python cli.py data run-data-gen --config-path configs/dataset_gen/my_config.yaml

# Train classifier
python cli.py experiment classifier train --config configs/classification/my_config.yaml

# Train encoder-decoder
python cli.py experiment encoder-decoder train --config configs/my_config.yaml

# Representation engineering
python cli.py experiment representation-engineering --config configs/rep_eng/my_config.yaml

# Resume from checkpoint
python cli.py experiment classifier train --config config.yaml --resume path/to/checkpoint.pt

# Batch execution (pass directory instead of file)
python cli.py data run-data-gen --config-path configs/dataset_gen/

# Linting
ruff check .
ruff check --fix .
```

**Testing:** No formal test suite. Validation via training metrics and TensorBoard.

## Code Style Guidelines

### Imports (grouped with blank lines)
```python
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .evaluator import compute_metrics
```

### Naming Conventions
- **Classes:** PascalCase (`PatternClassifierMLP`, `SubjectModel`)
- **Functions/methods:** snake_case (`create_dataloaders`)
- **Private methods:** Leading underscore (`_build_mlp`)
- **Constants:** UPPER_CASE
- **Variables:** snake_case (`train_loader`)

### Type Hints
```python
def load_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    ...
```

### Logging
```python
logger = logging.getLogger(__name__)
logger.info("Starting process...")
logger.error(f"Failed: {e}", exc_info=True)
```

### Error Handling
```python
if not config_path.exists():
    logger.error(f"Configuration file not found: {config_path}")
    sys.exit(1)

try:
    result = process_data()
except Exception as e:
    logger.error(f"Processing failed: {e}", exc_info=True)
    sys.exit(1)
```

### Class Structure
```python
class MyTrainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str):
        self.model = model.to(device)
        self.config = config
        self._setup_loss()
    
    def _setup_loss(self):
        """Private setup method."""
        ...
    
    def train(self):
        """Public training method."""
        ...
```

### Configuration Pattern
```python
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

loss_config = config.get("loss", {})
recon_enabled = loss_config.get("reconstruction", {}).get("enabled", False)
```

### Docstrings (concise)
```python
class SubjectModel(nn.Module):
    """Simple neural network for sequence binary classification."""
```

### Line Length
Target ~88-100 characters (Ruff defaults).

## Key Abstractions

| Component | Purpose |
|-----------|---------|
| `SubjectModel` | Small neural networks trained on pattern classification |
| `PatternClassifierMLP` | Classifier predicting patterns from weights+signatures |
| `MLPEncoderDecoder` / `TransformerEncoderDecoder` | Autoencoders for weight-space learning |
| `WeightTokenizer` | Neuron-level tokenization for weight sequences |
| `RepresentationPipeline` | End-to-end steering vector computation |

## Important Notes

- All outputs go to `runs/` directory (gitignored)
- TensorBoard auto-launches when configured
- HuggingFace Hub integration for model/dataset sharing
- Batch execution: pass directory paths instead of config files
