# Neural Architecture Search: A Comprehensive Survey

## Abstract

Neural Architecture Search (NAS) has emerged as a transformative approach to automating
the design of deep neural network architectures. Traditional deep learning relies on
human experts to manually design network architectures — a process that is both
time-consuming and error-prone. NAS automates this process by searching through a
predefined search space of possible architectures to find optimal designs for specific
tasks. This survey covers the key components of NAS: search space design, search
strategies (including reinforcement learning, evolutionary algorithms, and gradient-based
methods), and performance estimation strategies.

## 1. Introduction

Deep learning has achieved remarkable success across many domains including computer
vision, natural language processing, speech recognition, and game playing. However,
the performance of deep learning models is highly dependent on the choice of neural
network architecture. Historically, designing these architectures has been a manual
process requiring significant domain expertise and extensive experimentation.

The challenge of architecture design is compounded by the fact that different tasks
and datasets often require different architectures. What works well for image
classification may not be optimal for object detection or semantic segmentation.
This has led researchers to explore automated methods for architecture design,
giving rise to the field of Neural Architecture Search.

### 1.1 The Problem of Manual Architecture Design

The traditional approach to neural network design follows an iterative process:
1. A human expert proposes an architecture based on intuition and experience
2. The architecture is trained on the target dataset
3. Performance is evaluated on a validation set
4. Based on the results, the expert modifies the architecture
5. The process repeats until satisfactory performance is achieved

This process suffers from several limitations. First, it requires significant human
expertise and time. Second, human designers are limited by their own biases and
experience, potentially missing novel architectural patterns. Third, the search
space of possible architectures is astronomically large, making exhaustive manual
exploration impossible.

### 1.2 Automated Machine Learning (AutoML)

NAS is part of the broader field of Automated Machine Learning (AutoML), which aims
to automate various aspects of the machine learning pipeline. AutoML encompasses
feature engineering, hyperparameter optimization, model selection, and architecture
design. NAS specifically focuses on the architecture design component, though it
often interacts with other AutoML components.

## 2. Search Space Design

The search space defines the set of architectures that can be considered during the
search process. A well-designed search space should balance expressiveness (the ability
to represent a wide variety of architectures) with tractability (the ability to
efficiently search through the space).

### 2.1 Cell-Based Search Spaces

One of the most popular approaches is to search for small building blocks called
"cells" rather than entire architectures. A cell is a small directed acyclic graph
(DAG) where each node represents a feature map and each edge represents an operation.
The final architecture is constructed by stacking cells in a predefined pattern.

The cell-based approach has several advantages: it reduces the search space
significantly, enables transfer of discovered cells to different datasets and tasks,
and produces architectures with a natural hierarchical structure similar to
handcrafted designs.

### 2.2 Operation Types

Common operations in NAS search spaces include:
- Standard convolutions (1x1, 3x3, 5x5, 7x7)
- Depthwise separable convolutions
- Dilated convolutions (atrous convolutions)
- Pooling operations (max pooling, average pooling)
- Skip connections (identity mapping)
- Squeeze-and-excitation blocks
- Multi-head self-attention

## 3. Search Strategies

### 3.1 Reinforcement Learning-Based NAS

The seminal work by Zoph and Le (2017) used a recurrent neural network (RNN)
controller to generate architecture descriptions. The controller is trained using
the REINFORCE algorithm, where the reward signal is the validation accuracy of
the generated architecture. While this approach produced state-of-the-art results,
it required enormous computational resources — approximately 800 GPU-days for the
CIFAR-10 experiment.

### 3.2 Evolutionary Approaches

Evolutionary methods maintain a population of architectures and use mutation and
crossover operations to generate new candidates. AmoebaNet demonstrated that
evolutionary search can match or exceed RL-based methods while being more
memory-efficient. The tournament selection strategy proved particularly effective,
where a subset of the population is randomly sampled and the best architecture
in the subset becomes the parent for the next generation.

### 3.3 Gradient-Based Methods (DARTS)

Differentiable Architecture Search (DARTS) relaxes the discrete search space to
be continuous, allowing architecture parameters to be optimized using gradient
descent. This dramatically reduces search cost from thousands of GPU-days to just
a few GPU-days. However, DARTS can suffer from performance collapse, where the
search converges to degenerate architectures dominated by skip connections.

### 3.4 One-Shot Methods

One-shot NAS trains a single "supernet" that contains all possible architectures
as sub-networks. Individual architectures can then be evaluated by inheriting
weights from the supernet, eliminating the need to train each candidate from
scratch. This approach is orders of magnitude more efficient than training-based
methods.

## 4. Transformers and Attention Mechanisms

The rise of transformer architectures has introduced new dimensions to the architecture
search problem. Vision Transformers (ViTs) demonstrated that pure attention-based
models can achieve competitive performance on image classification tasks. NAS has
been applied to discover optimal transformer configurations, including the number of
attention heads, embedding dimensions, feed-forward network sizes, and the depth of
the transformer stack.

Recent work has explored hybrid architectures that combine convolutional and
attention-based components. These hybrid models often achieve better accuracy-efficiency
tradeoffs than pure convolutional or pure attention-based architectures.

## 5. Efficiency-Aware NAS

Modern NAS methods increasingly consider not just accuracy but also inference
efficiency. Multi-objective NAS optimizes architectures for both accuracy and
hardware metrics such as latency, FLOPs, memory usage, and energy consumption.
This is critical for deploying models on edge devices, mobile phones, and
embedded systems where computational resources are limited.

Hardware-aware NAS goes further by considering the specific characteristics of
the target hardware platform. An operation that is efficient on a GPU may be
slow on a mobile phone's neural processing unit (NPU). By incorporating
hardware-specific latency models into the search process, NAS can discover
architectures that are optimized for specific deployment targets.

## 6. Reproducibility and Benchmarking

A significant challenge in NAS research has been fair comparison between methods.
Different papers use different search spaces, training procedures, and evaluation
protocols, making direct comparisons difficult. The NAS-Bench series of benchmarks
(NAS-Bench-101, NAS-Bench-201, NAS-Bench-301) has helped address this by providing
pre-computed performance data for large collections of architectures.

## 7. Future Directions

Several promising research directions remain:
- Zero-shot NAS: predicting architecture performance without any training
- Foundation model NAS: searching architectures for large language models
- Green NAS: minimizing the carbon footprint of architecture search
- Automated data augmentation combined with NAS
- NAS for scientific computing and domain-specific applications

## References

1. Zoph, B. and Le, Q. V. (2017). Neural Architecture Search with Reinforcement Learning.
2. Real, E. et al. (2019). Regularized Evolution for Image Classifier Architecture Search.
3. Liu, H. et al. (2019). DARTS: Differentiable Architecture Search.
4. Pham, H. et al. (2018). Efficient Neural Architecture Search via Parameter Sharing.
5. Tan, M. and Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs.
