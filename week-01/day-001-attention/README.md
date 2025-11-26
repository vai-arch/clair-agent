# Scaled Dot-Product Attention: LLM Attention System

This repository (or explanation guide) provides a comprehensive overview of the **Scaled Dot-Product Attention** mechanism, the core component powering modern Large Language Models (LLMs) like GPT, Claude, and LLaMA. The attention system allows models to weigh the importance of different parts of input sequences, enabling contextual understanding and generation.

The content is based on the original Transformer paper (*Attention is All You Need*, Vaswani et al., 2017) and includes code implementations, theoretical breakdowns, real-life examples, and visual diagrams.

## Table of Contents

- [Introduction](#introduction)
- [Core Formula](#core-formula)
- [Code Implementation](#code-implementation)
- [Key Concepts Explained](#key-concepts-explained)
- [Q, K, V Matrices: Real-Life Example](#q-k-v-matrices-real-life-example)
- [Visual Diagrams](#visual-diagrams)
- [How It Works in LLMs](#how-it-works-in-llms)
- [Extensions and Limitations](#extensions-and-limitations)
- [Running the Demo](#running-the-demo)
- [Contributing](#contributing)
- [References](#references)

## Introduction

Scaled Dot-Product Attention is the fundamental algorithm in Transformers. It computes relationships between elements in a sequence (e.g., words in a sentence) by using **Queries (Q)**, **Keys (K)**, and **Values (V)**. This enables parallel processing of long sequences, unlike older recurrent models.

In LLMs, attention helps the model "focus" on relevant context, such as resolving pronouns or capturing long-range dependencies.

## Core Formula

\[
\text{Attention}(Q, K, V) = \softmax\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
\]

- **Q**: Queries – "What am I looking for?"
- **K**: Keys – "What do I have available?"
- **V**: Values – "What information do I return?"
- **Scaling (\(\sqrt{d_k}\))**: Prevents large dot products from causing gradient issues.
- **Softmax**: Normalizes scores into probabilities.

## Code Implementation

The provided Python code (using NumPy) implements the attention mechanism. It includes a stable softmax, the main attention function, and causal masking for autoregressive models like GPT.

```python
import numpy as np

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    if mask is not None:
        scores += mask
    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)
    return output, attention_weights

def create_causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
    return mask[np.newaxis, :, :]
```

For the full code with demo, see the original script.

## Key Concepts Explained

- **batch_size**: Number of examples processed at once (e.g., 2 sentences in parallel for efficiency).
- **seq_len**: Length of the sequence (e.g., 4 tokens/words).
- **d_model**: Dimensionality of embeddings (e.g., 8 features per token for representation).

These define input shapes like (batch_size, seq_len, d_model).

Child-Friendly Analogy: Q is shouting what it wants, K are toy tags, V are the toys you grab after matching.

## Q, K, V Matrices: Real-Life Example

Using sentence: "the cat sat on mat" (tokens: ['the', 'cat', 'sat', 'on', 'mat']).

- **Input Embeddings** (simulated, shape: 1x5x4):

[[[0.1 0.2 0.3 0.4]
    [0.5 0.6 0.7 0.8]
    [0.9 1.0 1.1 1.2]
    [1.3 1.4 1.5 1.6]
    [1.7 1.8 1.9 2.0]]]

- **Q Matrix** (projected queries):

[[[-0.041 -0.663 -0.448 -0.059]
    [-0.027 -1.360 -0.433  0.446]
    [-0.013 -2.058 -0.418  0.951]
    [ 0.001 -2.755 -0.402  1.456]
    [ 0.015 -3.452 -0.387  1.961]]]

- **K Matrix** (projected keys):

  [[[-0.212 -0.097 -0.663  0.427]
    [-0.489 -0.134 -1.701  0.184]
    [-0.765 -0.171 -2.738 -0.060]
    [-1.042 -0.208 -3.775 -0.303]
    [-1.319 -0.245 -4.812 -0.547]]]

- **V Matrix** (projected values):

  [[[-0.329 -0.734 -0.402  0.250]
    [-0.547 -2.161 -0.835  0.143]
    [-0.765 -3.587 -1.268  0.035]
    [-0.983 -5.013 -1.701 -0.072]
    [-1.201 -6.440 -2.133 -0.179]]]

Q, K, V are derived from embeddings via linear projections (matrix multiplications with learned weights).

## Visual Diagrams

Here are diagrams illustrating QKV interactions:

## How It Works in LLMs

- **Unmasked**: Bidirectional (e.g., BERT encoders).
- **Masked (Causal)**: Unidirectional for generation (e.g., GPT decoders).
- Output: Context-enriched representations.

## Extensions and Limitations

- **Multi-Head Attention**: Run in parallel for richer features.
- **Limitations**: Quadratic complexity (O(seq_len²)); mitigated by sparse variants.
- **In Practice**: Use PyTorch/TensorFlow for optimized versions.

## Running the Demo

1. Install NumPy: `pip install numpy`
2. Run the script: `python attention.py`
3. Observe outputs for masked/unmasked attention.

## Contributing

Feel free to add examples, optimizations, or issues!

## References

- Vaswani et al. (2017): *Attention is All You Need*.
- Original Code: Provided in this guide.
