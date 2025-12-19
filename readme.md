# Weight-Space Learning for Interpretability

**Weights as a Modality:** Recent works in weight-space learning have effectively trained interpreter models to predict model attributes of subject models such as accuracy or the hyperparameters used to train them, taking just the weights as inputs. 

**Applied to Interpretability:** I think this concept can be extended to behavioral characteristics of models, such as predicting what they've been trained to classify as positive. Further, after encoding subject model weights into a latent space organized by behavior, we can push encodings within that space, and decode them back to weights to repair/add/remove behaviors to the weights.

>If you're looking for the experiments directory, go to [`/model_zoo/`](/model_zoo/). This readme describes the work, there is a readme in the model zoo which explains how to run experiments.

---

## Core Theory: Meta-Universal Approximation Theorem

The [Universal Approximation Theorem (UAT)](https://en.wikipedia.org/wiki/Universal_approximation_theorem) states that a neural network can approximate any continuous function, effectively learning transformations from input values to output values. **Meta-UAT extends this principle to function space, hypothesizing that a neural network can also approximate transformations between functions (or neural networks).** *Note: I'm not sure if there's an existing formal theory for this, this is just my attempt at describing the work theoretically.*

If Meta-UAT applies, it would mean we could build a model of those transformations between functions/neural networks, which would allow us to interpret their positions and apply transformations to functions/neural networks to augment them.

**Analogy:** I've found it helpful to think about this in low-dimensions. Imagine you have a black box containing an unknown 2D shape. You can't look inside, but you *can* pass other shapes through it and observe how the probe-shapes get deformed by the hidden-shape within. If you were to do this on enough black boxes, using a fixed set of probe-shapes, you could build a mapping of probe-shape deformations to hidden-shapes.

![circles/triangles analogy drawing](/docs/analogy.png)

---

## Activation Signatures Instead of Weights

Instead of training on the weights directly (which restricts the interpreter to a subject model architecture's weight space), I train on activation signatures, effectively a universal interface to weights. I extract these activation signatures by passing a fixed dataset of probes through each model in our training dataset, and observing the activations (i.e. I create a behavioral fingerprint for our models). These probes can be aribrary, but should cover the entire domain of behaviors I want to be able to interpret. By building our latent space from these signatures instead of weights, we can translate unseen architectures/weights into our latent space, by first extracting their signatures and using them as inputs.

Using activation signatures is fundamental to our interpretabiltiy task:

1. **Universal Interface:** By using signatures we can train a single interpreter to diagnose and edit models across different subject model architectures and sizes (previous work held architecture fixed).

2. **Behavior Grounded:** Two models with vastly different weights can have the same behavior. Activation-space learning encodes *what a model does*, not just *how it does something* (as weight space learning would). This aligns the latent space with human-comprehensible concepts rather than implementation details.

3. **Causal Insight:** We extract signatures at the neuron level, so there is a neuron level profile. This means our interpreter can trace location of behaviors, not just their presence. This is essential for targeted weight augmentations, rather than just generating a whole new model with desired behavior. 

---

## Training the Interpreter

*Note to self: might rewrite this later to be task agnostic, I'm describing it in terms of this work because it's clearer.*


#### Dataset of Neural Network Weights

To train an interpreter, we need a dataset of subject models to use as examples. To build that, we'll need datasets of examples with labels for certain behaviors. 

**Subject Model Task Description:** The subject models I'm training classify sequences of tokens as positive or negative based on the pattern they contain. For example, a model might be trained to identify sequences where all the tokens are the same, in which case passing in "AAAAA" would result in a positive classification, and passing in "AAABA" would be negative.

This allows me to deterministically build training datasets for subject models which contain both positive and negative examples of sequences for whatever behavior we want (i.e. what pattern it should classify as positive), and train subject models on them.

#### Extracting Signatures

For each subject model, I pass a fixed dataset of probe examples through. This dataset contains 100 examples (a few from each pattern), and after passing them into the subject model and extracting activations, I'm left with 100 signatures. I apply an aggregation method to turn that into one signature for each subject model. In theory we could use all signatures (one from each example) as our input, but it's more practical to do an aggregation across the examples' signatures to pass in a single signature with statistical profiles for each neuron.

#### Encoder/Decoder Training (Interpreter Model)

We can train an encoder-decoder which takes in activation signatures and predicts weights (using contrastive loss on our gold weights). The latent space between the encoder and decoder will have embeddings which are organized by behavior and contain implemention details. Using just the encoder, we can add classification heads to classify behavior presences or model attributes (accuracy, etc.). Using the decoder, we can decode the embeddings into weights. If we train additional layers for editing behavior in between, we can augment the embeddings to edit/add/remove/repair behaviors.

---

# TODO

I think reconstructing the decoded weights from the original shape could cause issues. Need some way to do this information internally
