# Encoder Head Training

Trains task-specific prediction heads on top of pre-trained weight-space encoders to extract high-level properties from neural network weights.

## Purpose

Given a pre-trained encoder that compresses neural network weights into latent representations, this module trains lightweight prediction heads to classify patterns, predict accuracy, or infer hyperparameters directly from the latent codes. This enables efficient analysis of neural network properties without expensive evaluation or architectural inspection.

## Components

**encoder_loader.py**: Downloads pre-trained encoders from HuggingFace Hub. Reconstructs encoder architecture from checkpoint metadata, creates compatible tokenizers, and returns ready-to-use encoder modules for representation extraction.

**prediction_heads.py**: Implements three task-specific head architectures. PatternClassificationHead performs multi-label classification for architectural patterns (convolutions, batch norm, residual connections). AccuracyPredictionHead regresses model test accuracy. HyperparameterPredictionHead performs multi-task prediction with separate branches for continuous (learning rate, weight decay) and discrete (optimizer, activation) targets.

**model.py**: Wraps encoder and prediction head into unified EncoderWithHead model. Handles frozen encoder mode (representations only) vs fine-tuning mode (end-to-end training). Supports differential learning rates for encoder and head when fine-tuning. Automatically disables gradients for frozen encoder to save memory.

**data_loader.py**: Loads subject model datasets from HuggingFace Hub and creates task-specific targets from metadata. Implements representation caching to precompute and store encoder outputs when encoder is frozen, dramatically speeding up head training. Supports on-the-fly tokenization for fine-tuning mode. Handles multi-label targets (pattern classification), scalar targets (accuracy), and mixed targets (hyperparameters).

**evaluator.py**: Computes task-specific evaluation metrics. Pattern classification uses hamming loss, subset accuracy, per-pattern F1, and ROC-AUC. Accuracy prediction uses MSE, MAE, RMSE, R², and relative error. Hyperparameter prediction evaluates continuous targets with regression metrics and discrete targets with classification metrics, then aggregates performance across all targets.

**trainer.py**: Main training loop with early stopping, learning rate scheduling, and HuggingFace Hub integration. Supports frozen encoder training (head only) and fine-tuning (differential learning rates). Implements task-specific loss functions: BCEWithLogits for pattern classification, MSE for accuracy prediction, weighted multi-task loss for hyperparameters. Logs metrics to TensorBoard, saves best checkpoints, and uploads trained heads to Hub with model cards.

## Training Flow

1. **Load encoder**: Download pre-trained encoder from HuggingFace Hub with metadata
2. **Load dataset**: Read subject models with metadata (patterns, accuracy, hyperparameters)
3. **Optional caching**: Precompute encoder representations for frozen encoder training (saves time)
4. **Initialize head**: Create task-specific prediction head based on config
5. **Train**: Optimize head (frozen mode) or fine-tune both encoder and head (differential LR)
6. **Evaluate**: Measure task-specific metrics on validation set with early stopping
7. **Test**: Report final performance on held-out test set
8. **Upload to Hub**: Push trained prediction head with config and TensorBoard logs

## Task Types

**Pattern Classification**: Multi-label classification identifying architectural patterns (has_conv_layers, has_batch_norm, has_dropout, etc.). Uses BCEWithLogitsLoss and evaluates with hamming loss, subset accuracy, and per-pattern F1 scores.

**Accuracy Prediction**: Regression predicting model test accuracy from weights. Uses MSE loss and evaluates with MAE, RMSE, R², and relative error. Useful for model selection without evaluation.

**Hyperparameter Prediction**: Multi-task prediction inferring training hyperparameters from converged weights. Continuous targets (learning rate, weight decay) use MSE loss with optional log-space prediction. Discrete targets (optimizer, activation, architecture) use cross-entropy. Per-target loss weighting allows emphasizing important hyperparameters.

## Training Modes

**Frozen Encoder**: Encoder parameters fixed, only train prediction head. Enables representation caching for maximum training speed. Lower memory usage, faster iterations, suitable when encoder already learned good representations.

**Fine-tuning**: Train both encoder and head end-to-end with differential learning rates (encoder: 0.1x, head: 1.0x). Better task-specific performance but slower training and higher memory usage. Recommended for small datasets or specialized domains.
