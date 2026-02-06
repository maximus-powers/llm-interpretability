# Comprehensive Guide to Representation Engineering and Weight Space Learning

This guide provides a structured learning path for data science researchers seeking deep technical expertise in representation engineering (RepE) and weight space learning. The material is organized from foundational concepts to cutting-edge research, with key papers and resources for each topic.

---

## Part I: Theoretical Foundations

### 1. Linear Representation Hypothesis (LRH)

The LRH posits that high-level semantic concepts are encoded as linear directions or subspaces in neural network representations[^46]. This foundational assumption underlies most modern interpretability work.

**Core Concepts:**
- Features correspond to approximately linear directions in activation space
- High-dimensional models encode concepts as sparse linear combinations of direction vectors
- Binary concepts (e.g., truthful/untruthful) can be identified as single directions[^46]

**Key Papers:**
- **"Deep learning models are secretly (almost) linear"** - Beren Millidge's influential blog post documenting evidence for linearity across interpolation, LoRA mixing, model merging, and linear probing[^55]
- **"On the Failure of a Universal Linear Representation Hypothesis"** - Critical analysis showing LRH cannot be universally true, providing important boundary conditions[^52]
- **"Recurrent Neural Networks Learn to Store and Generate Sequences using Non-Linear Representations"** (BlackboxNLP 2024) - Counterexamples showing magnitude-based representations[^49]

### 2. Superposition Hypothesis

Superposition explains why individual neurons are often polysemantic (responding to multiple unrelated concepts). Neural networks represent more features than they have neurons by encoding features as near-orthogonal directions in activation space[^62][^68].

**Key Papers:**
- **"Toy Models of Superposition"** (Anthropic, 2022) - The seminal paper demonstrating superposition in controlled settings using toy ReLU networks[^74]. This is essential reading.
- **"Superposition is not 'just' neuron polysemanticity"** - Important clarification distinguishing polysemanticity (observation) from superposition (mechanism)[^71]

**Core Insights from Toy Models:**
- Models can store additional features by tolerating interference
- Superposition emerges when there are more sparse features than dimensions
- Models can perform computation while features are in superposition[^74]

### 3. Loss Landscape Geometry

Understanding the geometry of neural network loss landscapes provides crucial context for weight space learning.

**Key Papers:**
- **"Visualizing the Loss Landscape of Neural Nets"** (Li et al., 2018) - Introduces filter normalization for visualization; shows architecture dramatically affects landscape structure[^18]
- **"Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances"** (ICML 2021) - Studies how permutation symmetries create critical points[^24]
- **"Emergent properties of the local geometry of neural loss landscapes"** - Shows loss landscapes have C highly curved directions (C = number of classes) with flat directions elsewhere[^27]

---

## Part II: Interpretability Methods

### 4. Linear Probes and Classifiers

Linear probes are simple models trained on neural network activations to test whether specific information is linearly decodable[^16][^19].

**Key Papers:**
- **"Understanding intermediate layers using linear classifier probes"** (Alain & Bengio, 2016) - Foundational work showing linear separability increases monotonically through layers[^22]
- **"Interpreting Intentionally Flawed Models with Linear Probes"** (ICCV 2019) - Uses probes to study training dynamics and generalization[^16]

**Applications:**
- Testing if models contain specific linguistic/factual information
- Identifying where information is stored across layers
- Tracking how representations transform through the network[^19]

### 5. Logit Lens and Variants

The logit lens projects intermediate layer representations into vocabulary space to observe how predictions evolve[^76][^87].

**Key Resources:**
- **"Interpreting GPT: the logit lens"** (nostalgebraist, 2020) - Original introduction of the technique[^87]
- **"Logit Prisms: Decomposing Transformer Outputs"** - Extends logit lens with mathematically rigorous decomposition into component contributions[^76]
- **"Eliciting Latent Predictions from Transformers with the Tuned Lens"** (Belrose et al.) - Addresses limitations including representational drift[^82]

### 6. Sparse Autoencoders (SAEs)

SAEs decompose neural network activations into interpretable, monosemantic features by learning sparse overcomplete representations[^33][^36].

**Key Papers:**
- **"Towards Monosemanticity: Decomposing Language Models With Sparse Autoencoders"** (Anthropic, 2023) - Landmark paper extracting interpretable features from a one-layer transformer[^42]
- **"Sparse Autoencoders Find Highly Interpretable Features in Language Models"** (ICLR 2024) - Demonstrates SAEs on Pythia-70M/410M with automated interpretability metrics[^33]

**Core Concepts:**
- SAEs address polysemanticity by finding an overcomplete basis
- Inspired by sparse coding hypothesis in neuroscience
- Features are more interpretable than individual neurons[^36]

### 7. Mechanistic Interpretability and Circuit Analysis

Mechanistic interpretability aims to reverse-engineer the algorithms neural networks learn during training[^56].

**Foundational Resources:**
- **Transformer Circuits Thread** (Anthropic) - Ongoing research thread on transformer internals[^59]
- **TransformerLens** - Essential library for mechanistic interpretability of GPT-style models[^85]
- **"Circuit Tracing: Revealing Computational Graphs in Language Models"** (Anthropic, 2025) - Latest method for uncovering mechanisms via attribution graphs[^72]

**Key Concepts:**
- Circuits: Subgraphs of the model responsible for specific behaviors
- Attribution graphs: Causal graphs depicting computational steps
- Cross-layer transcoders: Interpretable replacements for MLPs[^72][^77]

### 8. Induction Heads

Induction heads are a well-understood circuit enabling in-context learning in transformers[^64][^67].

**Key Papers:**
- **"In-context Learning and Induction Heads"** (Olsson et al., 2022) - Presents evidence that induction heads are the mechanistic source of in-context learning[^64][^67]
- **"How Transformers Implement Induction Heads"** (ICLR 2025) - Theoretical analysis of approximation and optimization[^61]

**Core Mechanism:**
- Two attention heads across layers work together
- Implements pattern completion: [A][B]...[A] → [B]
- Formation coincides with sharp increase in in-context learning ability[^64]

---

## Part III: Representation Engineering (RepE)

### 9. Core RepE Framework

Representation Engineering is a top-down approach to AI transparency that manipulates population-level representations rather than individual neurons[^4][^1].

**Foundational Papers:**
- **"Representation Engineering: A Top-Down Approach to AI Transparency"** (Zou et al., 2023) - The defining paper introducing RepE[^4]
- **"From Representation Engineering to Circuit Breaking"** (CMU, 2024) - Shows applications to honesty, harm reduction, and adversarial robustness[^1]

**Key Principles:**
- Focus on distributed representations encoding knowledge/traits across many neurons
- Operates at Marr's algorithmic level (vs. implementational level of mechanistic interp)
- Two-step pipeline: identify concept representation, then steer model behavior[^7]

### 10. Activation Engineering and Steering Vectors

Activation engineering modifies internal activations at inference time to control model outputs[^3][^15].

**Key Papers:**
- **"Steering Language Models With Activation Engineering"** (Turner et al., 2023) - Introduces Activation Addition (ActAdd) using prompt pair contrasts[^3][^9]
- **"Steering Llama 2 via Contrastive Activation Addition"** (ACL 2024) - Scales CAA to larger models with systematic evaluation[^84][^89]
- **"Steering Large Language Models using Conceptors"** (NeurIPS 2024) - Uses ellipsoidal regions instead of single vectors for more precise control[^12]

**Methods:**
- **ActAdd**: Compute steering vector from prompt pair (e.g., "Love" - "Hate"), add during forward pass[^3]
- **CAA**: Average difference over many contrast pairs for reduced noise[^84]
- Optimal intervention typically at intermediate layers (e.g., layer 15)[^81]

### 11. Causal Tracing and Knowledge Localization

Causal tracing identifies which model components mediate specific behaviors through interventions[^63][^66].

**Key Papers:**
- **"Locating and Editing Factual Associations in GPT" (ROME)** (Meng et al., 2022) - Localizes factual recall to mid-layer MLP modules processing subject tokens[^93][^99]
- **"Towards Vision-Language Mechanistic Interpretability"** - Adapts causal tracing to multimodal models[^66]

**ROME Insights:**
- Factual associations correspond to localized, directly-editable computations
- Mid-layer feed-forward modules store factual knowledge
- Rank-one updates can modify specific facts[^93]

---

## Part IV: Weight Space Learning

### 12. Weight Space Understanding

Weight space learning treats neural network weights as data to be analyzed, embedded, and generated[^14][^8].

**Foundational Papers:**
- **"Classifying the classifier: dissecting the weight space of neural networks"** (ECAI 2020) - Trains meta-classifiers to identify training setup from weights; releases Neural Weight Space dataset[^8]
- **"Towards Scalable and Versatile Weight Space Learning" (SANE)** (ICML 2024) - Task-agnostic representations scalable to larger models[^5]

**Research Dimensions:**
- **Weight space understanding**: Geometry, symmetry, statistical properties
- **Weight space discrimination**: Embedding, retrieval, behavior prediction
- **Weight space generation**: Hypernetworks, generative models, merging[^14]

### 13. Mode Connectivity

Mode connectivity describes how different trained solutions (modes) can be connected by low-loss paths in weight space[^43][^40].

**Key Papers:**
- **"Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs"** (NeurIPS 2018) - Demonstrates mode connectivity across architectures[^43]
- **"Optimizing Mode Connectivity via Neuron Alignment"** (NeurIPS 2020) - Uses neuron alignment to find better connecting curves[^34]
- **"Revisiting Mode Connectivity with Bézier Surfaces"** (ICLR 2025) - Extends to surface connectivity[^40]

**Implications:**
- Different training runs find solutions in the same connected basin
- Enables model averaging and ensembling
- Reveals structure in high-dimensional loss landscapes[^43]

### 14. Model Merging

Model merging combines multiple trained models in weight space without additional training[^20][^17].

**Survey Papers:**
- **"Model Merging: A Survey"** (Cameron Wolfe) - Comprehensive overview from 1990s to modern LLM applications[^20]
- **"A Review of Model Merging Approaches"** (2025) - Taxonomy including permutation, direct merging, pruning-based, and LoRA merging[^23]

**Key Techniques:**
| Method | Description | Use Case |
|--------|-------------|----------|
| SLERP | Spherical linear interpolation | Smooth blending of two models[^17] |
| Task Vectors | Add/subtract fine-tuning deltas | Multi-task capability combination[^48] |
| TIES/DARE | Magnitude-based pruning merging | Conflict resolution[^17] |
| Frankenmerging | Layer stacking from different models | Architecture mixing[^26] |

### 15. Task Vectors

Task vectors encode task-specific directions in weight space, enabling arithmetic operations on model capabilities[^48][^54].

**Key Papers:**
- **"Editing Models with Task Arithmetic"** (ICLR 2023) - Foundational paper showing addition, negation, and analogy operations[^48][^51]
- **"Decomposing Task Vectors for Refined Model Editing"** - Separates shared subspaces from task-specific components[^60]

**Operations:**
- **Negation**: τ_task decreases performance on target task
- **Addition**: τ_A + τ_B improves performance on both tasks
- **Analogy**: A:B::C:D relationships can transfer capabilities[^48]

### 16. Hypernetworks

Hypernetworks are neural networks that generate weights for other networks[^32][^35][^41].

**Key Papers:**
- **"HyperNetworks"** (Ha et al., 2016) - Original work generating weights for LSTMs and CNNs[^32][^44]
- **"A Brief Review of Hypernetworks in Deep Learning"** (2024) - Comprehensive survey covering applications across problem settings[^35][^41]

**Applications:**
- Continual learning
- Transfer learning
- Weight pruning
- Uncertainty quantification
- Model compression[^35]

**Design Criteria:**
- Inputs: What conditions the weight generation
- Outputs: Which parameters are generated
- Variability: Static vs. dynamic generation
- Architecture: Structure of the hypernetwork itself[^41]

---

## Part V: Advanced Topics

### 17. Neural Tangent Kernel (NTK)

The NTK describes training dynamics of wide neural networks, linking them to kernel methods[^91][^94].

**Key Concepts:**
- In infinite-width limit, NTK becomes constant during training
- Training equivalent to kernel regression with NTK
- Provides closed-form equations for training dynamics[^94]

**Interpretability Connection:**
- Empirical NTK eigenspectrum can track phase transitions (e.g., grokking)
- Top eigenspaces may align with ground-truth features[^97]
- Limitations: equivalence theorem may not hold well in practice[^100]

### 18. Feature Visualization

Feature visualization generates inputs that maximally activate specific network units[^92][^101].

**Techniques:**
- **Visualization by optimization**: Find input maximizing unit activation
- **DeepDream**: Repeatedly add visualized features to input
- **Multifaceted visualization**: Cluster activations to identify distinct "facets"[^98]

**Insights:**
- Early layers detect simple edges/textures
- Later layers detect abstract parts/objects
- Network Dissection makes interpretability measurable[^101]

---

## Recommended Learning Progression

### Phase 1: Foundations (Weeks 1-3)
1. Read "Toy Models of Superposition" thoroughly
2. Study linear representation hypothesis evidence
3. Understand loss landscape geometry basics
4. Practice with linear probes on simple models

### Phase 2: Interpretability Methods (Weeks 4-6)
1. Implement logit lens on GPT-2 using TransformerLens
2. Train sparse autoencoders following Anthropic's guide
3. Reproduce induction head detection experiments
4. Study ROME paper and causal tracing methodology

### Phase 3: Representation Engineering (Weeks 7-9)
1. Read RepE foundational paper
2. Implement activation engineering on open models
3. Experiment with CAA for behavior steering
4. Explore circuit-tracing tools from Anthropic

### Phase 4: Weight Space Learning (Weeks 10-12)
1. Study mode connectivity literature
2. Experiment with model merging using mergekit
3. Implement task vector arithmetic
4. Explore hypernetwork architectures

---

## Essential Tools and Libraries

| Tool | Purpose | Link |
|------|---------|------|
| **TransformerLens** | Mechanistic interpretability | github.com/TransformerLensOrg/TransformerLens[^85] |
| **Mergekit** | Model merging | github.com/cg123/mergekit |
| **SAELens** | Sparse autoencoder training | github.com/jbloomAus/SAELens |
| **Neuronpedia** | Interactive feature exploration | neuronpedia.org |
| **Anthropic Circuit Tools** | Attribution graph generation | github.com/anthropics/attribution-graphs[^77] |

---

## Key Research Groups and Resources

- **Anthropic Interpretability Team**: Transformer Circuits thread, circuit tracing[^59][^77]
- **EleutherAI**: TransformerLens, open interpretability research
- **Alignment Forum / LessWrong**: Community discussions and preliminary results
- **Redwood Research**: Activation engineering, safety applications
- **MATS (ML Alignment Theory Scholars)**: Training program with interpretability focus

---

## Open Research Questions

1. **Scaling SAEs**: Do sparse autoencoders maintain interpretability at frontier model scale?
2. **Universality**: Are learned features/circuits consistent across different models?
3. **Causal Validity**: Do identified circuits actually explain model behavior or are they post-hoc rationalizations?
4. **Weight Space Geometry**: What is the true structure of the weight manifold?
5. **Compositional Steering**: Can multiple steering vectors be combined reliably?

This guide provides the foundation for rigorous research in representation engineering and weight space learning. The field is rapidly evolving—stay connected to preprint servers and research blogs for the latest developments.


---

## References

1. [From Representation Engineering to Circuit Breaking: Toward ...](https://www.cs.cmu.edu/~csd-phd-blog/2025/representation-engineering/) - Since the original paper, we have used RepE to make models more honest, to weaken harmful tendencies...

3. [[2308.10248] Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248) - We introduce activation engineering: the inference-time modification of activations in order to cont...

4. [Representation Engineering: A Top-Down Approach to AI ... - arXiv](https://arxiv.org/abs/2310.01405) - In this paper, we identify and characterize the emerging area of representation engineering (RepE), ...

5. [Towards Scalable and Versatile Weight Space Learning](https://proceedings.mlr.press/v235/schurholt24a.html) - This paper introduces the SANE approach to weight-space learning. SANE overcomes previous limitation...

7. [[PDF] arXiv:2502.19649v3 [cs.LG] 12 Mar 2025 - Jan Wehner](https://janwehner.com/files/representation_engineering.pdf) - Representation Engineering (RepE) is a novel paradigm for controlling the behavior of LLMs. Unlike t...

8. [dissecting the weight space of neural networks - arXiv](https://arxiv.org/abs/2002.05688) - This paper presents an empirical study on the weights of neural networks, where we interpret each mo...

9. [Steering Language Models with Activation Engineering - OpenReview](https://openreview.net/forum?id=2XBPdPIcFK) - We introduce a form of activation engineering: the inference-time modification of activations in ord...

12. [NeurIPS Steering Large Language Models using Conceptors](https://neurips.cc/virtual/2024/104080) - This paper explores activation engineering, where outputs of pre-trained LLMs are controlled by mani...

14. [Zehong-Wang/Awesome-Weight-Space-Learning - GitHub](https://github.com/Zehong-Wang/Awesome-Weight-Space-Learning) - Weight Space Learning is a research perspective that shifts focus from studying neural networks only...

15. [Activation Engineering - LessWrong](https://www.lesswrong.com/w/activation-engineering) - Activation Engineering is the direct manipulation of activation vectors inside of a trained machine ...

16. [[PDF] Interpreting Intentionally Flawed Models with Linear Probes](https://openaccess.thecvf.com/content_ICCVW_2019/papers/SDL-CV/Graziani_Interpreting_Intentionally_Flawed_Models_with_Linear_Probes_ICCVW_2019_paper.pdf) - The representational differences between generalizing networks and intentionally flawed models can b...

17. [Model Merging: Combining Different Fine-Tuned LLMs - Marvik — AI](https://www.marvik.ai/blog/model-merging-combining-different-fine-tuned-llms) - In this blog we explore Model Merging, which is becoming one of the most popular approaches to mix L...

18. [Visualizing the Loss Landscape of Neural Nets](https://www.cs.umd.edu/~tomg/projects/landscapes/) - In this paper, we explore the structure of neural loss functions, and the effect of loss landscapes ...

19. [Explainability methods: Linear Probes - The Carpentries Incubator](https://carpentries-incubator.github.io/fair-explainable-ml/5c-probes.html) - A probe is a simple model that uses the representations of the model as input, and tries to learn th...

20. [Model Merging: A Survey - by Cameron R. Wolfe, Ph.D.](https://cameronrwolfe.substack.com/p/model-merging) - Model merging is a popular research topic as of late, but the history of this technique is quite ext...

22. [[PDF] Understanding intermediate layers using linear classifier probes](https://arxiv.org/pdf/1610.01644.pdf) - Neural network models have a reputation for being black boxes. We propose to monitor the features at...

23. [A Review of Model Merging Approaches - arXiv](https://arxiv.org/html/2503.08998v1) - Direct Merging type is a relatively simple model merging method that avoids the complexities of reso...

24. [Geometry of the Loss Landscape in Overparameterized Neural ...](https://arxiv.org/abs/2105.12221) - We study how permutation symmetries in overparameterized multi-layer neural networks generate `symme...

26. [Model Merging and You](https://planetbanatt.net/articles/modelmerging.html) - Model merging is a weird and experimental technique which lets you take two models and combine them ...

27. [[PDF] Emergent properties of the local geometry of neural loss landscapes](https://ganguli-gang.stanford.edu/pdf/19.NeuralLossLandscapes.pdf) - Fundamen- tally, the neural network loss landscape is a scalar loss function over a very high D dime...

32. [HyperNetworks - Google Research](https://research.google/pubs/hypernetworks-2/) - This work explores hypernetworks: an approach of using a one network, also known as a hypernetwork, ...

33. [Sparse Autoencoders Find Highly Interpretable Features in ...](https://openreview.net/forum?id=F76bwRSLeK) - We use a scalable and unsupervised method called Sparse Autoencoders to find interpretable, monosema...

34. [[PDF] Optimizing Mode Connectivity via Neuron Alignment - NeurIPS](https://proceedings.neurips.cc/paper/2020/file/aecad42329922dfc97eee948606e1f8e-Paper.pdf) - We generalize learning a curve between two neural networks by optimizing both the permuta- tion of t...

35. [A brief review of hypernetworks in deep learning](https://eprints.whiterose.ac.uk/id/eprint/236619) - Hypernetworks, or hypernets for short, are neural networks that generate weights for another neural ...

36. [An Intuitive Explanation of Sparse Autoencoders for LLM ...](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html) - Sparse Autoencoders (SAEs) have recently become popular for interpretability of machine learning mod...

40. [Revisiting Mode Connectivity in Neural Networks with Bezier Surface](https://openreview.net/forum?id=1NevL7zdHS) - The paper explores the concept of mode connectivity in neural network loss landscapes, expanding it ...

41. [A Brief Review of Hypernetworks in Deep Learning - arXiv](https://arxiv.org/abs/2306.06955) - Hypernetworks, or hypernets for short, are neural networks that generate weights for another neural ...

42. [Towards Monosemanticity: Decomposing Language Models With ...](https://transformer-circuits.pub/2023/monosemantic-features) - Sparse Autoencoders extract relatively monosemantic features. · Sparse autoencoders produce interpre...

43. [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://izmailovpavel.github.io/curves_blogpost/) - In this blogpost we describe mode connectivity, a surprising property of modern neural net loss land...

44. [HyperNetworks - OpenReview](https://openreview.net/forum?id=rkpACe1lx) - This work explores hypernetworks: an approach of using one network, also known as a hypernetwork, to...

46. [Linear Representation Hypothesis - Emergent Mind](https://www.emergentmind.com/topics/linear-representation-hypothesis-lrh) - LRH is a hypothesis suggesting that high-level semantic concepts are encoded as linear directions or...

48. [[2212.04089] Editing Models with Task Arithmetic - arXiv](https://arxiv.org/abs/2212.04089) - We build task vectors by subtracting the weights of a pre-trained model from the weights of the same...

49. [Recurrent Neural Networks Learn to Store and Generate Sequences ...](https://aclanthology.org/2024.blackboxnlp-1.17/) - The Linear Representation Hypothesis (LRH) states that neural networks learn to encode concepts as d...

51. [Editing models with task arithmetic - OpenReview](https://openreview.net/forum?id=6t0Kwf8-jrj) - We build task vectors by subtracting the weights of a pre-trained model from the weights of the same...

52. [On the Failure of a Universal Linear Representation Hypothesis in...](https://openreview.net/forum?id=pmfF7wwX6W) - The Linear Representation Hypothesis (LRH) posits that semantic concepts in deep neural networks are...

54. [Task Vectors in Neural Networks - Emergent Mind](https://www.emergentmind.com/topics/task-vectors-tvs) - They enable efficient in-context learning, model editing, and multi-task merging through operations ...

55. [Deep learning models are secretly (almost) linear - Beren's Blog](https://www.beren.io/2023-04-04-DL-models-are-secretly-linear/) - People keep finding linear representations inside of neural networks when doing interpretability or ...

56. [Mechanistic Interpretability, Variables, and the Importance of ...](https://www.transformer-circuits.pub/2022/mech-interp-essay) - Mechanistic interpretability seeks to reverse engineer neural networks, similar to how one might rev...

59. [Transformer Circuits Thread](https://transformer-circuits.pub) - Mechanistic Interpretability, Variables, and the Importance of Interpretable Bases. An informal note...

60. [Decomposing Task Vectors for Refined Model Editing - Zhen Zhang](https://zzhang.org/submitted_papers/task-decomposition/) - Task vectors, defined as the difference between fine-tuned and pre-trained model parameters, provide...

61. [How Transformers Implement Induction Heads](https://openreview.net/forum?id=1lFZusYFHq) - The paper analyzes the implementation and learning of induction heads in simplified two-layer transf...

62. [The Superposition Hypothesis And How it Changed AI Interpretability](https://thesequence.substack.com/p/the-sequence-opinion-667-the-superposition) - The superposition hypothesis proposes that neural networks are not built around one-neuron-per-featu...

63. [Causal Tracing in Complex Systems - Emergent Mind](https://www.emergentmind.com/topics/causal-tracing) - Causal tracing frameworks employ a variety of mathematical, algorithmic, and infrastructural approac...

64. [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) - Induction heads are implemented by a circuit consisting of a pair of attention heads in different la...

66. [[2308.14179] Towards Vision-Language Mechanistic Interpretability](https://arxiv.org/abs/2308.14179) - In this work, we adapt a unimodal causal tracing tool to BLIP to enable the study of the neural mech...

67. [[2209.11895] In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) - We present six complementary lines of evidence, arguing that induction heads may be the mechanistic ...

68. [Superposition: What Makes it Difficult to Explain Neural Network](https://towardsdatascience.com/superposition-what-makes-it-difficult-to-explain-neural-network-565087243be4/) - Superposition refers to a specific phenomenon that one neuron in a model represents multiple overlap...

71. [Superposition is not "just" neuron polysemanticity](https://www.alignmentforum.org/posts/8EyCQKuWo6swZpagS/superposition-is-not-just-neuron-polysemanticity) - The superposition hypothesis claims that polysemanticity occurs in neural networks because of this f...

72. [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) - We introduce a method to uncover mechanisms underlying behaviors of language models. We produce grap...

74. [Toy Models of Superposition - Transformer Circuits Thread](https://transformer-circuits.pub/2022/toy_model/index.html) - The superposition hypothesis suggests that each feature in the higher-dimensional model corresponds ...

76. [Logit Prisms: Decomposing Transformer Outputs for Mechanistic ...](https://neuralblog.github.io/logit-prisms/) - The logit lens (nostalgebraist 2020) is a simple yet powerful tool for understanding how transformer...

77. [Open-sourcing circuit-tracing tools - Anthropic](https://www.anthropic.com/research/open-source-circuit-tracing) - This release enables researchers to: Trace circuits on supported models, by generating their own att...

81. [Steering Llama 2 via Contrastive Activation Addition - YouTube](https://www.youtube.com/watch?v=hAf840jD6oc) - Contrastive Activation Addition (CAA) is a method for steering language models by modifying activati...

82. [Lenses - Structure and Interpretation of Deep Networks](https://sidn.baulab.info/lenses/) - Looking Through the Lens of Interpretability ... Logit Lens in their paper Eliciting Latent Predicti...

84. [[PDF] Steering Llama 2 via Contrastive Activation Addition - ACL Anthology](https://aclanthology.org/2024.acl-long.828.pdf) - We introduce Contrastive Activation Addition. (CAA), a method for steering language models by modify...

85. [TransformerLensOrg/TransformerLens: A library for ... - GitHub](https://github.com/TransformerLensOrg/TransformerLens) - This is a library for doing mechanistic interpretability of GPT-2 Style language models. The goal of...

87. [interpreting GPT: the logit lens - LessWrong](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) - The logit lens focuses on what GPT "believes" after each step of processing, rather than how it upda...

89. [Steering Llama 2 via Contrastive Activation Addition - ACL Anthology](https://aclanthology.org/2024.acl-long.828/) - We introduce Contrastive Activation Addition (CAA), a method for steering language models by modifyi...

91. [Neural Tangent Kernel (NTK) Overview - Emergent Mind](https://www.emergentmind.com/topics/neural-tangent-kernel-framework-ntk) - In genetic risk modeling, embedding the empirical NTK into classical statistical models enables both...

92. [What is feature visualization? - AISafety.info](https://aisafety.info/questions/8HIA/What-is-feature-visualization) - Feature visualization is an interpretability technique which can generate representations to gain in...

93. [[2202.05262] Locating and Editing Factual Associations in GPT - arXiv](https://arxiv.org/abs/2202.05262) - ... Model Editing (ROME). We find that ROME is effective on a standard zero-shot relation extraction...

94. [Neural tangent kernel - Wikipedia](https://en.wikipedia.org/wiki/Neural_tangent_kernel) - The neural tangent kernel (NTK) is a kernel that describes the evolution of deep artificial neural n...

97. [Finding Features in Neural Networks with the Empirical NTK](https://www.lesswrong.com/posts/cpFqDDjhvhbaoyHnd/finding-features-in-neural-networks-with-the-empirical-ntk-1) - In interpretability, we would like to understand how neural networks represent learned features, “ca...

98. [[PDF] Visualizing and explaining neural networks](https://slazebni.cs.illinois.edu/spring21/lec12_visualization.pdf) - Overview of visualization techniques. • Mapping activations back to the image. • Synthesizing images...

99. [[PDF] Locating and Editing Factual Associations in GPT - NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/file/6f1d43d5a82a37e89b0665b33bf3a182-Paper-Conference.pdf) - We test this hypothesis by conducting a new type of intervention: modifying factual associations wit...

100. [Issues with Neural Tangent Kernel Approach to Neural Networks](https://arxiv.org/abs/2501.10929) - This theorem allows for an interpretation of neural networks as special cases of kernel regression. ...

101. [27 Learned Features – Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/cnn-features.html) - Feature visualization for a unit of a neural network is done by finding the input that maximizes the...

