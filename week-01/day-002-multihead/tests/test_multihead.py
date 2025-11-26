"""
Tests for multi-head attention.
Run: python tests/test_multihead.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from multihead import MultiHeadAttention, split_heads, combine_heads


def test_split_combine_heads():
    """Test splitting and combining heads."""
    print("Testing split_heads and combine_heads...")
    
    batch_size, seq_len, d_model = 2, 4, 64
    num_heads = 8
    
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Split
    x_split = split_heads(x, num_heads)
    assert x_split.shape == (batch_size, num_heads, seq_len, d_model // num_heads), \
        f"Wrong split shape: {x_split.shape}"
    
    # Combine
    x_combined = combine_heads(x_split)
    assert x_combined.shape == (batch_size, seq_len, d_model), \
        f"Wrong combined shape: {x_combined.shape}"
    
    # Should be identical to original
    assert np.allclose(x, x_combined), "Split-combine should be identity"
    
    print("  ✓ Split and combine work correctly")


def test_multihead_shapes():
    """Test multi-head attention output shapes."""
    print("Testing multi-head attention shapes...")
    
    batch_size, seq_len, d_model = 2, 6, 64
    num_heads = 8
    
    x = np.random.randn(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model, num_heads)
    
    output, attention_weights = mha(x, x, x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Wrong output shape: {output.shape}"
    
    # Check attention weights shape
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
        f"Wrong attention weights shape: {attention_weights.shape}"
    
    print("  ✓ Shapes correct")


def test_attention_weights_sum():
    """Test that attention weights sum to 1 for each head."""
    print("Testing attention weights sum to 1...")
    
    batch_size, seq_len, d_model = 2, 4, 64
    num_heads = 8
    
    x = np.random.randn(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model, num_heads)
    
    output, attention_weights = mha(x, x, x)
    
    # Weights should sum to 1 along last dimension (for each head)
    sums = attention_weights.sum(axis=-1)
    assert np.allclose(sums, 1.0), \
        f"Attention weights don't sum to 1: {sums}"
    
    print("  ✓ Weights sum to 1 for all heads")


def test_different_qkv():
    """Test with different Q, K, V inputs (cross-attention)."""
    print("Testing with different Q, K, V...")
    
    batch_size, seq_len_q, seq_len_kv, d_model = 2, 4, 6, 64
    num_heads = 8
    
    query = np.random.randn(batch_size, seq_len_q, d_model)
    key = np.random.randn(batch_size, seq_len_kv, d_model)
    value = np.random.randn(batch_size, seq_len_kv, d_model)
    
    mha = MultiHeadAttention(d_model, num_heads)
    output, attention_weights = mha(query, key, value)
    
    # Output should match query sequence length
    assert output.shape == (batch_size, seq_len_q, d_model), \
        f"Wrong output shape: {output.shape}"
    
    # Attention weights: (batch, num_heads, seq_len_q, seq_len_kv)
    assert attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_kv), \
        f"Wrong attention shape: {attention_weights.shape}"
    
    print("  ✓ Cross-attention works")


def test_heads_are_different():
    """Test that different heads produce different attention patterns."""
    print("Testing that heads learn different patterns...")
    
    batch_size, seq_len, d_model = 1, 8, 64
    num_heads = 8
    
    x = np.random.randn(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model, num_heads)
    
    output, attention_weights = mha(x, x, x)
    
    # Compare head 0 vs head 1
    head_0 = attention_weights[0, 0, :, :]
    head_1 = attention_weights[0, 1, :, :]
    
    # They should NOT be identical (different random initializations)
    assert not np.allclose(head_0, head_1), \
        "Heads should have different attention patterns"
    
    print("  ✓ Different heads produce different patterns")


def test_divisibility_assertion():
    """Test that d_model must be divisible by num_heads."""
    print("Testing d_model divisibility assertion...")
    
    try:
        # This should fail
        mha = MultiHeadAttention(d_model=64, num_heads=7)  # 64 not divisible by 7
        assert False, "Should have raised assertion error"
    except AssertionError as e:
        assert "divisible" in str(e).lower()
        print("  ✓ Correctly enforces divisibility")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING MULTI-HEAD ATTENTION TESTS")
    print("=" * 70 + "\n")
    
    test_split_combine_heads()
    test_multihead_shapes()
    test_attention_weights_sum()
    test_different_qkv()
    test_heads_are_different()
    test_divisibility_assertion()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    