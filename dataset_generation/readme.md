# Training Dataset Generation

Generates datasets for training LLM interpreters to understand neural network weights.

## Setup

```bash
# Install dependencies
pip install -r pipeline/requirements.txt

# Create signature dataset (one-time)
python3 cli.py create-sig-dataset --config-path path/to/config.yaml --filename path/to/signature_dataset.json --size 200

# Generate dataset
python3 cli.py run-data-gen --config-path path/to/config.yaml

# TensorBoard runs at http://localhost:6006
```

Configure via YAML files:
- Model architectures (layers, neurons, activations)
- Training params (epochs, lr, early stopping)
- Pattern selection and corruption rates
- Output length and HuggingFace upload

## Components

**ActivationSignatureExtractor**: Processes the signature dataset through subject models to extract layer activations. Creates a fingerprint for each model that the interpreter uses as a reference when analyzing weights.

**PatternDatasetSampler**: Creates and manages sequence pools for each pattern. Samples from these pools to build training datasets with the specified pattern mix. Also generates signature datasets and benchmark datasets for eval.

**SubjectModelTrainer**: Trains models on pattern datasets. Handles different architectures, early stopping, weight quantization (int8, int4, ternary, binary), and device management (CPU/CUDA/MPS).

**TrainingDataFormatter**: Formats generated data into prompt/completion pairs for interpreter training. Handles both modification and classification tasks with configurable formatting.

## Pipeline Flow

The `DatasetGenerationPipeline` orchestrates everything:

1. **Generate each example independently**: Randomly select patterns, create mixed dataset, configure model architecture, pick a pattern to corrupt.

2. **Two-stage training**: Train on corrupted data (stage 1) until validation loss improves, then switch to clean data (stage 2) with lower learning rate. Use best checkpoint from stage 2.

3. **Validate examples**: Check that stage 2 actually improves on the corrupted pattern without breaking other patterns. Discard bad examples.

4. **Extract signatures**: Run signature dataset through degraded model (for modification tasks) or improved model (for classification tasks).

5. **Format training data**:
   - Modification task: degraded weights + signature → improved weights
   - Classification task: improved weights + signature → pattern labels

Each example runs in parallel threads for efficiency. Final dataset contains subject models, activation signatures, improved models, and metadata.
