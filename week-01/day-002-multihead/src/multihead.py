"""
Multi-Head Attention - The Secret Sauce of Transformers

Instead of one attention mechanism, we run multiple "heads" in parallel.
Each head can learn to attend to different aspects of the input.

Example:
- Head 1: Focuses on subject-verb relationships
- Head 2: Focuses on adjective-noun relationships  
- Head 3: Focuses on long-range dependencies
- etc.

Architecture:
Input → Linear projections → Split into H heads → Attention per head 
→ Concatenate → Linear projection → Output
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from attention import scaled_dot_product_attention


def split_heads(x, num_heads):
    """
    Split the last dimension into (num_heads, depth).
    
    Args:
        x: Input tensor, shape (batch_size, seq_len, d_model)
        num_heads: Number of attention heads
    
    Returns:
        Reshaped tensor, shape (batch_size, num_heads, seq_len, depth)
    """
    batch_size, seq_len, d_model = x.shape
    depth = d_model // num_heads
    
    # Reshape: (batch, seq_len, d_model) → (batch, seq_len, num_heads, depth)
    x = x.reshape(batch_size, seq_len, num_heads, depth)
    # Transpose: (batch, seq_len, num_heads, depth) → (batch, num_heads, seq_len, depth)
    x = x.transpose(0, 2, 1, 3)
    return x


def combine_heads(x):
    """
    Inverse of split_heads.
    
    Args:
        x: Input tensor, shape (batch_size, num_heads, seq_len, depth)
    
    Returns:
        Reshaped tensor, shape (batch_size, seq_len, d_model)
    """
    batch_size, num_heads, seq_len, depth = x.shape
    
    # Transpose: (batch, num_heads, seq_len, depth) → (batch, seq_len, num_heads, depth)
    x = x.transpose(0, 2, 1, 3)
    
    # Reshape: (batch, seq_len, num_heads, depth) → (batch, seq_len, d_model)
    d_model = num_heads * depth
    x = x.reshape(batch_size, seq_len, d_model)
    
    return x


class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.
    
    Architecture:
    1. Linear projections for Q, K, V (one per head)
    2. Split into multiple heads
    3. Scaled dot-product attention per head (in parallel)
    4. Concatenate head outputs
    5. Final linear projection
    """
    
    def __init__(self, d_model, num_heads, seed=42):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            seed: Random seed for weight initialization
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads  # Dimension per head
        
        # Initialize weights
        np.random.seed(seed)
        
        # Linear projections for Q, K, V
        # In practice, these are learned during training
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        
        # Final output projection
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor, shape (batch_size, seq_len_q, d_model)
            key: Key tensor, shape (batch_size, seq_len_k, d_model)
            value: Value tensor, shape (batch_size, seq_len_v, d_model)
            mask: Optional mask, shape (batch_size, 1, 1, seq_len_k)
                  or (batch_size, 1, seq_len_q, seq_len_k)
        
        Returns:
            output: Attention output, shape (batch_size, seq_len_q, d_model)
            attention_weights: Attention weights from all heads, 
                             shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.shape[0]
        
        # Step 1: Linear projections
        # (batch, seq_len, d_model) @ (d_model, d_model) → (batch, seq_len, d_model)
        Q = np.matmul(query, self.W_q)
        K = np.matmul(key, self.W_k)
        V = np.matmul(value, self.W_v)
        
        # Step 2: Split into multiple heads
        # (batch, seq_len, d_model) → (batch, num_heads, seq_len, depth)
        Q = split_heads(Q, self.num_heads)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)
        
        # Step 3: Apply attention to each head
        # Process all heads at once by treating (batch * num_heads) as batch dimension
        # Reshape: (batch, num_heads, seq_len, depth) → (batch * num_heads, seq_len, depth)
        batch_heads = batch_size * self.num_heads
        Q_reshaped = Q.reshape(batch_heads, Q.shape[2], self.depth)
        K_reshaped = K.reshape(batch_heads, K.shape[2], self.depth)
        V_reshaped = V.reshape(batch_heads, V.shape[2], self.depth)
        
        # Apply mask if provided (broadcast across heads)
        mask_reshaped = None
        if mask is not None:
            # mask shape: (batch, 1, seq_len, seq_len) or similar
            # Need to expand for heads: (batch, num_heads, seq_len, seq_len)
            if len(mask.shape) == 3:
                mask = mask[:, np.newaxis, :, :]  # Add head dimension
            # Now tile across heads and reshape
            mask_reshaped = np.tile(mask, (1, self.num_heads, 1, 1))
            mask_reshaped = mask_reshaped.reshape(batch_heads, mask.shape[-2], mask.shape[-1])
        
        # Compute attention for all heads
        attention_output, attention_weights = scaled_dot_product_attention(
            Q_reshaped, K_reshaped, V_reshaped, mask_reshaped
        )
        
        # Reshape back: (batch * num_heads, seq_len, depth) → (batch, num_heads, seq_len, depth)
        attention_output = attention_output.reshape(
            batch_size, self.num_heads, attention_output.shape[1], self.depth
        )
        attention_weights = attention_weights.reshape(
            batch_size, self.num_heads, attention_weights.shape[1], attention_weights.shape[2]
        )
        
        # Step 4: Concatenate heads
        # (batch, num_heads, seq_len, depth) → (batch, seq_len, d_model)
        concat_attention = combine_heads(attention_output)
        
        # Step 5: Final linear projection
        # (batch, seq_len, d_model) @ (d_model, d_model) → (batch, seq_len, d_model)
        output = np.matmul(concat_attention, self.W_o)
        
        return output, attention_weights
    
    def __call__(self, query, key, value, mask=None):
        """Allow calling instance like a function."""
        return self.forward(query, key, value, mask)


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-HEAD ATTENTION - DEMO")
    print("=" * 70)
    
    # Set random seed
    np.random.seed(42)
    
    # Parameters
    batch_size = 2
    seq_len = 6
    d_model = 64
    num_heads = 8
    
    print(f"\nParameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Depth per head: {d_model // num_heads}")
    
    # Create sample input (in practice, this would be token embeddings)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    
    # Initialize multi-head attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # Self-attention (Q = K = V = x)
    print("\n" + "-" * 70)
    print("Computing self-attention...")
    output, attention_weights = mha(x, x, x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"  → batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}")
    
    # Inspect attention weights from different heads
    print("\n" + "-" * 70)
    print("Attention weights from first batch, first 3 heads:")
    print("\nHead 0 (focus on position 0):")
    print(attention_weights[0, 0, :, :])
    print(f"Row sums: {attention_weights[0, 0, :, :].sum(axis=-1)}")
    
    print("\nHead 1 (might have different pattern):")
    print(attention_weights[0, 1, :, :])
    
    print("\nHead 2 (might have yet another pattern):")
    print(attention_weights[0, 2, :, :])
    
    # Verify output dimension is correct
    print("\n" + "-" * 70)
    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch!"
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len), "Attention weights shape mismatch!"
    
    print("✓ Multi-head attention working correctly!")
    print("=" * 70)
    
    # Compare: Average attention across all heads
    avg_attention = attention_weights.mean(axis=1)  # Average over head dimension
    print("\nAverage attention across all heads (first batch):")
    print(avg_attention[0])
    print("\n💡 Different heads learn different attention patterns!")
    print("   This is the power of multi-head attention.")