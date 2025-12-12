# Model Zoo

Core package for generating training datasets and training pattern classifiers.

## Components

- **[dataset_generation](dataset_generation/readme.md)** - Generates datasets for training LLM interpreters to understand neural network weights
- **[classification_training](classification_training/readme.md)** - Trains classifiers to identify patterns in neural network weights and activations

## Installation

```bash
cd model_zoo
pip install -r requirements.txt
```

## Usage

The Model Zoo provides a unified CLI with two modes: interactive (menu-driven) and traditional (command-line arguments).

**Note:** Navigate to the `model_zoo/` directory before running commands: `cd model_zoo`

### Interactive Mode (Recommended)

Launch the interactive menu:

```bash
python cli.py
```

Navigate with arrow keys (↑↓), press Enter to select, and follow the prompts.

### Traditional CLI Mode

Use command-line arguments for scripting and automation.

## Commands

### Dataset Generation

Generate training datasets for LLM interpreters:

**Create a signature dataset** (one-time setup per vocab/sequence configuration):

```bash
python cli.py data create-sig-dataset \
  --config-path configs/dataset_gen/my_config.yaml \
  --filename signature_dataset.json \
  --size 200
```

**Run the full dataset generation pipeline**:

```bash
python cli.py data run-data-gen \
  --config-path configs/dataset_gen/my_config.yaml
```

**Create a benchmark dataset** (for evaluation):

```bash
python cli.py data create-benchmark-dataset \
  --config-path configs/dataset_gen/my_config.yaml \
  --samples-per-pattern 35 \
  --filename benchmark_dataset.json
```

TensorBoard automatically launches at http://localhost:6006 (if configured).

### Classification Training

Train pattern classifiers:

**Train a new classifier**:

```bash
python cli.py experiment classifier train \
  --config configs/classification/my_config.yaml
```

**Resume training from checkpoint**:

```bash
python cli.py experiment classifier train \
  --config configs/classification/my_config.yaml \
  --resume path/to/checkpoint.pt
```

## Configuration

Configuration files are organized in the `configs/` directory:

### Dataset Generation (`configs/dataset_gen/`)

Configure:
- Model architectures (layers, neurons, activations)
- Training parameters (epochs, learning rate, early stopping)
- Pattern selection and corruption rates
- Dataset size and output options
- TensorBoard settings

Example: `configs/dataset_gen/example_config.yaml`

### Classification Training (`configs/classification/`)

Configure:
- Classifier architecture (encoder dimensions, dropout, activation)
- Training parameters (batch size, learning rate, optimizer)
- Dataset paths and splits
- Device settings (CPU, CUDA, MPS)
- Checkpoint and logging

Example: `configs/classification/example_config.yaml`

## Workflow

### End-to-End Training Data Generation

1. **Configure** your experiment in `configs/dataset_gen/`
2. **Create signature dataset** (one-time per configuration)
3. **Run dataset generation** to create training examples
4. Generated datasets include:
   - Subject model weights
   - Activation signatures
   - Pattern labels
   - Metadata

### Training Pattern Classifiers

1. **Configure** classifier settings in `configs/classification/`
2. **Point to generated dataset** in the config
3. **Run training** to learn pattern recognition
4. **Evaluate** on test set

## Output Structure

All CLI runs create timestamped output directories in `runs/` to keep your workspace organized:

### Data Generation Runs

Each run creates: `runs/data-generation_{config_name}_{timestamp}/`

Directory structure:
```
runs/data-generation_all_2025-12-03_18-01-42/
├── config.yaml       # Modified config used for this run
├── datasets/         # Generated training data
│   ├── metadata.json
│   ├── subject_models/
│   └── formatted_data.json
├── metrics/          # TensorBoard logs for subject model training
│   └── events.out.tfevents...
└── checkpoints/      # Subject model checkpoints during training
```

### Classification Training Runs

Each run creates: `runs/train-classifier_{config_name}_{timestamp}/`

Directory structure:
```
runs/train-classifier_muat_separate_pca_10_2025-12-03_18-05-30/
├── config.yaml       # Modified config used for this run
├── cache/            # Cached dataset files
├── logs/             # TensorBoard logs
│   └── events.out.tfevents...
└── checkpoints/      # Model checkpoints
    └── best_model.pt
```


# TODO:

- Make any config params that are necessary to have in the next step's config included in the readme of the HF uploads as tags