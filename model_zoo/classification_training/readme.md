# Pattern Classification Training

Trains classifiers to identify patterns in neural network weights and activations.

## Purpose

Given a trained model and its activation signatures, the classifier learns to predict which patterns the model was trained to recognize. This enables automated analysis of model capabilities without manually inspecting weights or running inference.

## Components

**classifier_model.py**: Implements the PatternClassifierMLP architecture with separate encoders for weights and activation signatures. Combines both representations through a fusion layer to predict pattern labels.

**data_loader.py**: Loads generated datasets, extracts weight tensors and activation signatures from subject models, handles pattern labels, and creates train/val/test splits with proper batching.

**trainer.py**: Training loop with checkpointing, early stopping, and logging. Manages optimizer, learning rate scheduling, validation metrics, and model persistence.

**evaluator.py**: Evaluation utilities for computing metrics (accuracy, precision, recall, F1) on test sets and analyzing per-pattern performance.

## Training Flow

1. **Load dataset**: Read generated subject models with their weights, activation signatures, and pattern labels
2. **Extract features**: Convert weight tensors to flat vectors, process activation signatures
3. **Initialize model**: Configure encoder dimensions based on input feature sizes
4. **Train**: Optimize classifier to predict patterns from weight+signature inputs
5. **Evaluate**: Measure performance on held-out test set
