# Training Dataset Generation for Interpreter

To do experiments of LLM interpretation of other neural nets, we can deterministically generate examples of subject models, activation signatures, and modified models.

---

## Dataset Record:

Each record contains:

1. **Subject Model:** A real neural network that classifies sequences of 7 letters based on patterns we've chosen for it to classify as positive. This model is trained on datasets of these patterns (and negative examples). For some datasets, used in the more complex "modification of weights" experiments, noise was added to one of the patterns intended to be classified as positive (to degrade it's performance). This degraded pattern is what our interpreter can be prompted to improve in the modification experiments. 
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
- `SubjectModelTrainer`: 

