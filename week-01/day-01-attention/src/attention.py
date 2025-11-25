"""
Scaled Dot-Product Attention - The Core of Transformers

This is the algorithm that powers GPT, Claude, LLaMA, and every modern LLM.

Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Where:
- Q (Query): "What am I looking for?"
- K (Key): "What do I have available?"
- V (Value): "What information do I return?"
- d_k: Dimension of keys (used for scaling)
"""

import numpy as np


def softmax(x, axis=-1):
    """
    Purpose: Computes the softmax probabilities in a numerically stable way. Softmax turns raw scores into a probability distribution (sums to 1).
    
    How It Works:
    - Subtracts the max value along the axis to avoid exponential overflow (e.g., exp(1000) is huge).
    - Computes exp on the shifted values, then normalizes by the sum.

    Theory Tie-In: In attention, softmax ensures the weights are positive and normalized, acting like a "soft" selection mechanism (instead of hard max).
    
    Example: For input scores [1.0, 2.0, 3.0], softmax gives roughly [0.09, 0.24, 0.67]—higher scores get more probability mass.
    
    Args:
        x: Input array
        axis: Axis along which to compute softmax
    
    Returns:
        Softmax probabilities (sums to 1 along axis)
    """
    # Subtract max for numerical stability (prevents overflow)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention.
    
    The attention mechanism answers: "For each query, which keys (and their values) are most relevant?"
    
    Args:
        Q: Queries, shape (batch_size, seq_len_q, d_k)
        K: Keys, shape (batch_size, seq_len_k, d_k)
        V: Values, shape (batch_size, seq_len_v, d_v)
        mask: Optional mask, shape (batch_size, seq_len_q, seq_len_k)
              Use -inf for positions that should be ignored
    
    Returns:
        output: Attention output, shape (batch_size, seq_len_q, d_v)
        attention_weights: Attention probabilities, shape (batch_size, seq_len_q, seq_len_k)
    """
    # Get dimension of keys for scaling
    d_k = Q.shape[-1]
    
    # Step 1: Compute attention scores (QK^T)
    # This gives us "how much does each query attend to each key?"
    # Shape: (batch_size, seq_len_q, seq_len_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1))
    
    # Step 2: Scale by √d_k
    # Why? Prevents dot products from getting too large (which causes vanishing gradients in softmax)
    scores = scores / np.sqrt(d_k)
    
    # Step 3: Apply mask (if provided)
    # Used for things like preventing attention to future tokens (causal masking)
    if mask is not None:
        scores = scores + mask  # mask should have -inf for positions to ignore
    
    # Step 4: Apply softmax to get attention probabilities
    # This normalizes scores so they sum to 1 (making them proper probabilities)
    # Shape: (batch_size, seq_len_q, seq_len_k)
    attention_weights = softmax(scores, axis=-1)
    
    # Step 5: Apply attention weights to values
    # This is "weighted averaging" - we take a weighted sum of all values,
    # where weights are our attention probabilities
    # Shape: (batch_size, seq_len_q, d_v)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights


def create_causal_mask(seq_len):
    """
    Create a causal mask for autoregressive generation.
    
    This prevents position i from attending to positions > i.
    Used in decoder-only models like GPT.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        mask: Shape (1, seq_len, seq_len) with -inf in upper triangle
    """
    mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
    return mask[np.newaxis, :, :]  # Add batch dimension


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("SCALED DOT-PRODUCT ATTENTION - DEMO")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 4
    d_model = 8  # Embedding dimension
    ### Explaining the Concepts: batch_size, seq_len, and d_model
    """
    #### 1. **batch_size = 2**
    - **Meaning**: This is the number of independent examples (or "samples") processed simultaneously in one forward pass through the model. In machine learning, "batching" groups multiple inputs together to make training/inference faster and more efficient on hardware like GPUs (parallel processing). A batch_size of 2 means you're handling 2 separate sequences at once.
    - **Theory Tie-In**: Attention is computed in parallel across the batch, so larger batches speed up training but require more memory. In LLMs, batching helps average gradients over multiple examples during training, stabilizing learning (e.g., via stochastic gradient descent).
    - **How It Works in the Code**:
        - The inputs Q, K, and V all have shape `(batch_size, seq_len, d_model)` = `(2, 4, 8)`.
        - Operations like `np.matmul(Q, K.transpose(0, 2, 1))` are applied across the entire batch at once—no loops needed. The first dimension (batch) is independent, so batch 1's attention doesn't affect batch 2's.
        - In the output, shapes like attention_weights are `(2, 4, 4)`—one 4x4 weight matrix per batch.
    - **Example**:
        - Imagine translating two sentences: "Hello world" (batch 1) and "Bonjour monde" (batch 2). With batch_size=2, the model processes both in parallel, computing attention for each separately.
        - In Real LLMs: During training GPT, batch_size might be 512 or more (e.g., millions of tokens across batches). For inference (chatting with Claude), it's often 1 (your single prompt), but servers batch multiple user queries for efficiency.
        - Why 2 in the Demo? It's small for illustration—easy to print and debug without overwhelming output.

    #### 2. **seq_len = 4**
    - **Meaning**: This is the **sequence length**, or the number of elements (e.g., tokens like words or subwords) in each input sequence. For seq_len=4, each batch item has 4 positions/tokens. In LLMs, sequences are your input text broken into tokens (e.g., "The quick brown fox" might be 4 tokens).
    - **Theory Tie-In**: Attention scales quadratically with seq_len (O(seq_len²) due to the QK^T matrix), so longer sequences capture more context but cost more compute. LLMs like GPT-4 handle seq_len up to 128k+ (long-context models), but early Transformers used ~512.
    - **How It Works in the Code**:
        - Shapes: The middle dimension of Q/K/V is seq_len (4), representing positions in the sequence.
        - In scores = `np.matmul(Q, K.transpose(0, 2, 1))`, this creates a (batch, 4, 4) matrix—each of the 4 queries attends to all 4 keys.
        - With causal masking, it limits attention to previous positions (e.g., position 3 only sees 1-3, not 4).
        - Attention_weights is (2, 4, 4): A square matrix per batch, showing how each position attends to others.
    - **Example**:
        - For text: seq_len=4 could be the sentence "I like ice cream" (tokenized as 4 parts). Attention lets "cream" attend strongly to "ice" (contextual understanding).
        - In Real LLMs: Your prompt "Write a story about a dragon" might be ~6 tokens (seq_len=6). During generation, the model extends the sequence one token at a time, recomputing attention each step.
        - Why 4 in the Demo? Keeps matrices small (e.g., 4x4 weights are easy to print and inspect). In practice, try changing it to 10 for longer "sentences."

    #### 3. **d_model = 8**
    - **Meaning**: This is the **model dimension** or embedding size—the number of features (dimensions) used to represent each token/position in the sequence. d_model=8 means each token is a vector of 8 numbers (a point in 8D space). In LLMs, this captures rich semantics (e.g., word meanings, positions).
    - **Theory Tie-In**: Higher d_model allows more expressive representations but increases parameters/compute. In attention, d_k (from Q/K) and d_v (from V) often equal d_model. The scaling \(\sqrt{d_k}\) prevents large dot products in high dimensions.
    - **How It Works in the Code**:
        - Last dimension of Q/K/V is d_model (8).
        - In scaling: `scores / np.sqrt(d_k)` where d_k=8, so divide by ~2.83. This keeps scores reasonable before softmax.
        - Output shape: (2, 4, 8)—same as input, but now each position's vector is a weighted mix of others.
    - **Example**:
        - Think of a token like "apple" embedded as [0.1, -0.5, 0.3, 0.7, -0.2, 0.4, 0.9, -0.1] (8D vector). Attention blends these vectors based on relevance.
        - In Real LLMs: GPT-3 uses d_model=12288 (huge for nuance)! Small d_model=8 is for demo—real ones need high dims for capturing grammar, semantics, etc.
        - Why 8 in the Demo? Tiny for computation; in full models, it's split across heads (e.g., 8 heads x 64 dim/head = 512 d_model).

    ### How They Interact in Attention
    - Together, they define tensor shapes: (batch_size, seq_len, d_model).
    - In a full LLM: Input text → Tokenize (to seq_len tokens) → Embed (to d_model vectors) → Batch multiple examples → Run attention.
    - Efficiency Trade-Offs: Increase batch_size for speed; seq_len for context; d_model for quality—but balance memory (e.g., too big causes OOM errors).
    - Visual Analogy: Imagine a classroom (batch_size=2 classes), with rows of students (seq_len=4 per class), each described by 8 traits (d_model=8). Attention is students "paying attention" to each other based on similarity scores.

    If you run the code and change these (e.g., batch_size=1, seq_len=10, d_model=16), you'll see how shapes/outputs adapt. Let me know if you want examples with real text or code tweaks!
    """

    print(f"\nInput dimensions:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    
    # In practice, Q, K, V come from linear projections of input embeddings
    # For this demo, we'll use random values
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"\nShapes:")
    print(f"  Q (queries): {Q.shape}")
    print(f"  K (keys): {K.shape}")
    print(f"  V (values): {V.shape}")
    
    # Compute attention (no mask)
    print("\n" + "-" * 60)
    print("Computing attention (no mask)...")
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\nOutput:")
    print(f"  Shape: {output.shape}")
    print(f"  First position of first batch: {output[0, 0, :]}")
    
    print(f"\nAttention weights:")
    print(f"  Shape: {attention_weights.shape}")
    print(f"  First batch, all positions attending to all positions:")
    print(attention_weights[0])
    print(f"\n  Row sums (should all be 1.0): {attention_weights[0].sum(axis=-1)}")
    
    # Compute attention WITH causal mask
    print("\n" + "-" * 60)
    print("Computing attention (WITH causal mask)...")
    mask = create_causal_mask(seq_len)
    output_masked, attention_weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    print(f"\nCausal mask (upper triangle = -inf, prevents looking ahead):")
    print(mask[0])
    
    print(f"\nMasked attention weights:")
    print(f"  First batch (note zeros in upper triangle):")
    print(attention_weights_masked[0])
    print(f"\n  Row sums (still 1.0): {attention_weights_masked[0].sum(axis=-1)}")
    
    print("\n" + "=" * 60)
    print("✓ Attention mechanism working correctly!")
    print("=" * 60)

""" 
    ### Real-Life Example of Q, K, V Matrices in Scaled Dot-Product Attention

    To make this more concrete than the random matrices in the original code's demo, I'll walk you through a simplified but realistic example using an actual sentence: **"the cat sat on mat"** (a shortened version of "The cat sat on the mat" for simplicity, with seq_len=5). This mimics how Q, K, V are generated in real LLMs like GPT or Claude, but scaled down for clarity.

    In a real LLM:
    - Text is **tokenized** into words/subwords (e.g., "the", "cat", etc.).
    - Each token gets an **embedding vector** (a dense numerical representation capturing meaning, learned during training).
    - Then, these embeddings are **linearly projected** (multiplied by weight matrices) to create Q, K, and V. These projections allow the model to emphasize different aspects for querying, key-matching, and value-retrieval.
    - For this example, I used a tiny vocabulary and dummy embeddings (in practice, embeddings are high-dimensional and trained on massive data). I also used random projection weights to simulate learned ones (as in a trained model).

    I executed some code to compute this (using NumPy for matrix operations). Here's the breakdown:

    #### 1. **Sentence and Tokens**
    - Tokens: `['the', 'cat', 'sat', 'on', 'mat']`
    - batch_size=1 (just one sentence/example).
    - seq_len=5 (number of tokens).
    - d_model=4 (embedding dimension—tiny for demo; real LLMs use 512+).

    #### 2. **Input Embeddings**
    - These are the starting vectors for each token (before projection). In real LLMs, they're looked up from an embedding table.
    - I assigned simple dummy values (increasing numbers) to simulate semantic similarity (e.g., "cat" and "mat" might have related vectors in a real model).
    - Shape: (1, 5, 4) — 1 batch, 5 tokens, 4 dimensions each.
    - Matrix:
        ```
        [[[0.1 0.2 0.3 0.4]   # 'the'
        [0.5 0.6 0.7 0.8]   # 'cat'
        [0.9 1.0 1.1 1.2]   # 'sat'
        [1.3 1.4 1.5 1.6]   # 'on'
        [1.7 1.8 1.9 2.0]]] # 'mat'
        ```
    - Interpretation: Each row is a token's vector. For example, "cat" is [0.5, 0.6, 0.7, 0.8]—in a real embedding, dimensions might capture traits like "animal-ness" or grammar.

    #### 3. **Projection to Q, K, V**
    - In Transformers, Q, K, V aren't the raw embeddings—they're derived by multiplying the input embeddings by three separate weight matrices (W_q, W_k, W_v). These weights are learned during training.
    - Here, I simulated random weights (seeded for reproducibility) and computed:
        - Q = input_emb @ W_q  (queries: "What am I looking for?")
        - K = input_emb @ W_k  (keys: "What do I have?")
        - V = input_emb @ W_v  (values: "What info to return?")
    - All have shape (1, 5, 4)—same as input, but transformed.

    - **Query Matrix Q** (focuses on what each token "queries" for):
        ```
        [[[-0.04121667 -0.66319791 -0.44838102 -0.05884401]   # 'the'
        [-0.0271972  -1.36044649 -0.43295469  0.44613492]   # 'cat'
        [-0.01317774 -2.05769507 -0.41752836  0.95111384]   # 'sat'
        [ 0.00084173 -2.75494366 -0.40210203  1.45609276]   # 'on'
        [ 0.0148612  -3.45219224 -0.3866757   1.96107168]]] # 'mat'
        ```
        - Example: The query for "sat" is [-0.013, -2.058, -0.418, 0.951]. In a real model, this vector "asks" for relevant context (e.g., what "sat" relates to "cat").

    - **Key Matrix K** (what each token "offers" for matching):
        ```
        [[[-0.21172365 -0.09713125 -0.66327748  0.42744067]   # 'the'
        [-0.48860516 -0.1340513  -1.70055591  0.1838104 ]   # 'cat'
        [-0.76548666 -0.17097135 -2.73783433 -0.05981988]   # 'sat'
        [-1.04236817 -0.2078914  -3.77511276 -0.30345015]   # 'on'
        [-1.31924968 -0.24481145 -4.81239118 -0.54708042]]] # 'mat'
        ```
        - Example: The key for "on" is [-1.042, -0.208, -3.775, -0.303]. Dot products with queries measure similarity (e.g., does "sat" match "on"?).

    - **Value Matrix V** (the actual info to blend based on matches):
        ```
        [[[-0.32944583 -0.73423232 -0.40233271  0.24980566]   # 'the'
        [-0.54732144 -2.16057511 -0.83510399  0.14262011]   # 'cat'
        [-0.76519706 -3.5869179  -1.26787526  0.03543456]   # 'sat'
        [-0.98307267 -5.01326069 -1.70064654 -0.071751  ]   # 'on'
        [-1.20094829 -6.43960348 -2.13341782 -0.17893655]]] # 'mat'
        ```
        - Example: If attention weights favor "cat" for the query "sat", the output for "sat" will pull heavily from [ -0.547, -2.161, -0.835, 0.143 ].

    #### How This Works in Attention
    - Plug these into the attention formula: Scores = (Q @ K.T) / sqrt(4), then softmax, then output = weights @ V.
    - In a real LLM (e.g., processing "the cat sat on mat"), "sat" might attend strongly to "cat" (high Q-K dot product), pulling value from "cat" to understand "the cat is sitting".
    - Simplifications: Real embeddings are higher-dim (e.g., 4096 in LLaMA), trained on billions of sentences. Projections are part of the model's parameters (billions of them).

    This shows Q, K, V as transformed versions of input embeddings—tailored for attention. If you want a different sentence, larger dimensions, or to compute full attention outputs, let me know!

 """