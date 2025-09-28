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

##### 2. Batch Generation

For each training example batch, the pipeline:

1. **Pattern Selection**: Randomly selects a subset of available patterns (between min_patterns_per_batch and max_patterns_per_batch from config) that will be used to create the training dataset for subject models in this batch.

2. **Dataset Creation**: Uses the PatternDatasetSampler to generate a mixed dataset containing positive examples of the selected patterns plus negative examples, following the specified ratios and sample counts from the config.

3. **Model Configuration**: Generates random model architecture parameters (number of layers, neurons per layer, activation type, learning rate) within the ranges specified in the config.

##### 3. Model Training

For each batch, the pipeline trains two types of models:

1. **Clean Subject Model**: Trained on the original dataset without any corruption. This serves as the "target" model representing optimal performance on all patterns.

2. **Degraded Subject Models**: For each example in the batch, a corrupted version of the dataset is created by flipping labels for a randomly selected pattern (at the specified corruption rate). Models trained on these corrupted datasets will have degraded performance on the corrupted pattern.

##### 4. Quality Filtering

Only model pairs where the performance degradation meets the minimum threshold (min_degradation_threshold from config) are kept as training examples. This ensures the interpreter has meaningful signal to learn from.

##### 5. Signature Extraction

For each qualified degraded model, the ActivationSignatureExtractor processes the baseline signature dataset through the model to extract layer activations. These activations serve as a "fingerprint" that the interpreter can learn to associate with specific model behaviors and patterns.

##### 6. Training Data Formatting

The TrainingDataFormatter combines all components into structured training examples:
- **Prompt**: Contains the degraded model weights, architecture config, activation signature, and task specification (which pattern to improve)
- **Completion**: Contains the clean model weights representing the desired improvement

##### 7. Incremental Saving and Checkpointing

The pipeline supports long-running generation with:
- **Checkpointing**: Periodic saves of progress to resume interrupted runs
- **Incremental HuggingFace Upload**: Automatic uploading of completed batches to HuggingFace Hub datasets
- **Cleanup**: Removal of temporary model files after processing

The entire process is designed to run efficiently with configurable multi-threading, allowing multiple batches to be processed in parallel while maintaining thread safety for logging and checkpointing operations.