# Training Dataset Generation for Interpreter

To do experiments of LLM interpretation of other neural nets, we can deterministically generate examples of subject models, activation signatures, and modified models.

---

## Dataset Record:

Each record contains:

1. **Subject Model:** A real neural network that classifies sequences of tokens (letters) based on patterns we've chosen for it to classify as positive. This model is trained on datasets of these patterns (and negative examples). For some datasets, used in the more complex "modification of weights" experiments, noise was added to one of the patterns intended to be classified as positive (to degrade it's performance). This degraded pattern is what our interpreter can be prompted to improve in the modification experiments. 
2. **Activation Signature:** A set of activation values for the subject model, found by processing the baseline dataset (baseline_dataset.json) with the subject model, extracting feature activations at each layer, and aggregating them into a single activation signature (as we'll have one for each baseline dataset record).
3. **Modified Model:** In datasets used for modification of weights experiments, we include a model trained on the same dataset as the subject model, only without corruption on any of the patterns. This can be used as the completion example, representing an improvement in the weights for a specific pattern's classifications.
4. **Metadata:** We'll keep track of metadata such as the patterns the subject model was trained to classify as positive, what pattern was corrupted/has degraded performance, evaluation metrics of each model, and subject model size/architecture.

These fields can be used to form prompts for the training of LLM interpreter models for the tasks:
- Classification: Task of identifying what patterns the subject model is trained to classify as positive.
- Modification: Task of generating weights with improved performance on a specific pattern.

---

## Generation Pipeline

To build these records, we go through the following steps, orchestrated by the `DatasetGenerationPipeline` class in `dataset_generation_pipeline.py`:

##### 1. Initialization

We start by initializing the other classes we'll need during the generation:

- `ActivationSignatureExtractor`: This gets initialized with the signature dataset (which should have already been generated), and will use each subject model we generate to process the signature dataset then extract the activations. This gives us an activation signature/model fingerprint that we essentially train our interpreter to reference when interpreting the weights of the subject model. The signature dataset underlying the activation signatures is held constant and can therefore be learned as a key to the weights. 
- `PatternDatasetSampler`: On init, this class creates pools of sequences for each pattern, which are then sampled to create datasets containing the desired patterns for training subject models. It deduplicates sequences (prioritizing smaller pool paterns), and stores them in memory so we don't get a bunch of junk files. Also, this class can be used to generate the one-time signature dataset, or benchmark datasets with pattern labels for evaluations of the interpreter's output models.
- `SubjectModelTrainer`: This class handles the actual training of subject models on pattern datasets. It supports various training configurations (learning rate, batch size, epochs), early stopping based on validation loss, and optional weight quantization (int8, int4, ternary, binary). The trainer can work with different devices (CPU, CUDA, MPS) and tracks training metrics throughout the process.
- `TrainingDataFormatter`: This class formats the generated data into the specific prompt structure needed for training the interpreter model. It can handle different formatting styles (separate vs interwoven) for organizing model weights, activation signatures, and task specifications into training prompts.

##### 2. Independent Example Generation

Each training example is generated completely independently with:

1. **Pattern Selection**: Each example randomly selects its own subset of available patterns (between min_patterns_per_batch and max_patterns_per_batch from config).

2. **Dataset Creation**: Each example uses the PatternDatasetSampler to generate its own unique mixed dataset containing positive examples of the selected patterns plus negative examples.

3. **Model Configuration**: Each example generates its own random model architecture parameters (number of layers, neurons per layer, activation type, learning rate) within the config ranges.

4. **Target Pattern Selection**: Each example randomly chooses one of its selected patterns to corrupt for the staged training process.

The pipeline uses thread pooling to generate multiple examples in parallel, with each thread working on a completely independent example. This maximizes diversity in the training data and ensures efficient parallelization.

##### 3. Staged Model Training

Each example uses a single model trained in two consecutive stages:

**Stage 1 - Degraded Training**: The model is trained on a corrupted dataset where the target pattern's labels are flipped (at the specified corruption rate). This creates a model with degraded performance specifically on that pattern while maintaining performance on other patterns.

**Stage 2 - Improvement Training**: The SAME model continues training on the clean dataset with a reduced learning rate (improvement_lr_factor from config). This teaches the model to correct its mistakes on the target pattern while preserving the existing circuits and knowledge.

This staged approach ensures the "improved" model is a genuine continuation of the degraded model, not an arbitrary replacement. The interpreter learns to make targeted improvements to fix specific issues rather than generate entirely new models.

##### 4. Pattern-Specific Validation

Each staged training example is validated to ensure the improvement is meaningful and targets the correct pattern:

1. **Improvement Magnitude**: The performance improvement from stage 1 to stage 2 must meet the minimum threshold (min_degradation_threshold from config).

2. **Pattern Specificity**: The improvement should specifically target the corrupted pattern, not just overall performance.

3. **Stability Check**: Performance on other patterns should remain stable (not degrade significantly).

Only examples passing all validation criteria are included in the final training dataset, ensuring the interpreter learns from high-quality improvement examples.

##### 5. Signature Extraction

The ActivationSignatureExtractor processes the baseline signature dataset through models based on task requirements:

- **For modification tasks**: Extracts signatures from the DEGRADED model (after stage 1) to help the interpreter understand the model's problematic behavior
- **For classification tasks**: Extracts signatures from the IMPROVED model (after stage 2) to accurately represent what patterns the clean model identifies

This approach ensures that modification tasks learn to diagnose problems from degraded states, while classification tasks learn to identify patterns from properly functioning models.

##### 6. Training Data Formatting

The TrainingDataFormatter creates structured training examples with configurable tasks:

**Modification Task** (when `include_modification: true`):
- **modification_prompt**: Contains the degraded model weights (after stage 1), degraded activation signature, architecture config, and task specification (which pattern to improve)
- **modification_completion**: Contains the improved model weights (after stage 2) representing the targeted improvement

**Classification Task** (when `include_classification: true`):
- **classification_prompt**: Contains the improved model weights, improved activation signature, architecture config, and all available pattern descriptions
- **classification_completion**: Lists the pattern names that the improved model classifies as positive

This dual-task approach enables training interpreters for both weight modification (diagnosing and fixing specific pattern issues) and pattern classification (identifying which patterns a model has learned).

##### 7. Incremental Saving and Checkpointing

The pipeline supports long-running generation with:
- **Checkpointing**: Periodic saves of progress to resume interrupted runs
- **Incremental HuggingFace Upload**: Automatic uploading of completed batches to HuggingFace Hub datasets
- **Cleanup**: Removal of temporary model files after processing

The entire process is designed to run efficiently with configurable multi-threading, allowing multiple independent examples to be processed in parallel while maintaining thread safety for logging and checkpointing operations. Each example is completely independent, maximizing diversity and enabling efficient parallelization without complex batch coordination.