# Representation Engineering

Linear Representation Engineering (LRE) for modifying neural network behaviors in latent space.

## Purpose

Given a pre-trained weight-space encoder, this module computes steering vectors that represent behavioral patterns in latent space and applies them to modify model capabilities. This enables targeted behavior modification without retraining - adding or enhancing patterns by navigating the learned weight representations.

## Components

**data_loader.py**: Unified loader for both steering and subject models from HuggingFace Hub. Encodes weights to latent space using pre-trained encoder, caches representations, and groups by pattern combinations. Provides filtering and sampling utilities. Same loader used for both steering vector computation and subject model testing.

**steering_vector_computer.py**: Computes steering vectors per pattern using Linear Representation Engineering: `steering_vector = mean(models_with_pattern) - mean(models_without_pattern)`. Supports vector normalization, metadata validation, and disk caching. Applies vectors sequentially for multi-pattern additions.

**model_modifier.py**: Applies steering vectors to subject model weights in latent space. Encodes weights, adds weighted steering vectors, decodes back to weight space. Supports strength scaling and multi-vector application for complex modifications.

**evaluator.py**: Evaluates modifications using pre-generated benchmark datasets. Loads benchmark JSON created via signature dataset CLI, infers vocab and sequence length from metadata. Extracts model architecture dynamically from each model's metadata to handle varying layer counts and neuron sizes. Computes per-pattern metrics (F1, precision, recall, accuracy) with contamination filtering for multi-pattern classifiers. Aggregates cumulative metrics across all patterns.

**representation_pipeline.py**: Orchestrates end-to-end workflow. Loads steering dataset to compute vectors, loads separate subject dataset for testing, validates pattern coverage, applies modifications with pattern combination expansion (tries all N-1 combinations per model), evaluates results with pre-generated benchmarks, generates reports, and optionally uploads to HuggingFace Hub.

**dataset_utils.py**: Builds HuggingFace datasets from representation engineering results. Serializes original/modified weights and metrics, generates dataset cards, and handles local/remote upload.

## Pipeline Flow

1. **Load steering dataset**: Encode models from large HuggingFace dataset, group by patterns
2. **Compute steering vectors**: Extract behavioral directions using LRE, cache with metadata validation
3. **Load subject models**: Load separate test models, ensure pattern coverage from steering dataset
4. **Pattern expansion**: For each subject model, try adding all other pattern combinations (N-1 per model)
5. **Apply modifications**: Encode weights, add steering vectors in latent space, decode back
6. **Evaluate**: Load pre-generated benchmark dataset, test modified models with contamination filtering
7. **Report**: Generate markdown summary with aggregate statistics
8. **Upload** (optional): Package results as HuggingFace dataset with before/after metrics

## Configuration

Two-dataset architecture prevents data leakage by separating steering vector computation from testing:

```yaml
dataset:
  subject_model_dataset: "maximuspowers/test-models"  # models to modify
  sample_size: 10  # number of models to test
  max_models_subject: 100  # max to load
  cache_latents: true
  cache_dir_subject: "latent_cache/subject"

steering:
  steering_vector_dataset: "maximuspowers/steering-models"  # compute vectors
  max_models_steering: null  # use all for best vectors
  normalize_vectors: false
  cache_dir: "steering_vectors_cache"
  compute_on_init: true
  force_recompute: false

encoder_decoder:
  encoder_repo_id: "maximuspowers/weight-space-encoder"
  decoder_repo_id: "maximuspowers/weight-space-encoder"  # same repo
  latent_dim: 256
  tokenization:
    chunk_size: 128
    max_tokens: 512

modification:
  strength: 1.0  # steering vector strength

evaluation:
  enabled: true
  benchmark_dataset_path: "dataset_generation/benchmarks/eval_benchmark.json"
  batch_size: 32
```

### Benchmark Dataset

Generate the benchmark dataset before running the pipeline:

```bash
python -m model_zoo.cli generate-signature-dataset \
  --output dataset_generation/benchmarks/eval_benchmark.json \
  --total-examples 1500
```

The evaluator automatically infers vocab_size and sequence_length from the benchmark metadata. Model architectures (num_layers, neurons_per_layer, etc.) are extracted dynamically from each model's metadata, allowing evaluation of models with varying architectures.

## Output

Generated files in `runs/representation-engineering-*/`:
- `modified_weights/`: Modified model state dicts per combination
- `steering_vectors.pt`: Computed steering vectors with metadata
- `evaluation_results.json`: Metrics before/after modification
- `report.md`: Summary with aggregate statistics
- `config.yaml`: Config used for this run
