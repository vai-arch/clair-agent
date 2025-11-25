"""
Tests for attention mechanism.
Run: python tests/test_attention.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from attention import scaled_dot_product_attention, softmax, create_causal_mask

def test_softmax():
    """Test softmax function."""
    print("Testing softmax...")
    
    # Test 1: Simple case
    x = np.array([[1.0, 2.0, 3.0]])
    result = softmax(x, axis=-1)
    
    # Should sum to 1
    assert np.allclose(result.sum(axis=-1), 1.0), "Softmax should sum to 1"
    
    # Should be all positive
    assert np.all(result > 0), "Softmax should be positive"
    
    # Largest input should have largest output
    assert np.argmax(result) == np.argmax(x), "Softmax should preserve ordering"
    
    print("  ✓ Softmax works correctly")


def test_attention_shapes():
    """Test attention output shapes."""
    print("Testing attention shapes...")
    
    batch_size, seq_len, d_model = 2, 4, 8
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Wrong output shape: {output.shape}"
    
    # Check weights shape
    assert weights.shape == (batch_size, seq_len, seq_len), f"Wrong weights shape: {weights.shape}"
    
    print("  ✓ Shapes correct")


def test_attention_weights_sum():
    """Test that attention weights sum to 1."""
    print("Testing attention weights sum to 1...")
    
    batch_size, seq_len, d_model = 2, 4, 8
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    # Weights should sum to 1 along last dimension (across keys)
    sums = weights.sum(axis=-1)
    assert np.allclose(sums, 1.0), f"Attention weights don't sum to 1: {sums}"
    
    print("  ✓ Weights sum to 1")


def test_causal_mask():
    """Test causal masking."""
    print("Testing causal mask...")
    
    seq_len = 4
    batch_size, d_model = 1, 8
    
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    mask = create_causal_mask(seq_len)
    output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    # Upper triangle should be zero (can't attend to future)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert np.allclose(weights[0, i, j], 0.0), \
                f"Position {i} should not attend to future position {j}"
    
    # Lower triangle should be non-zero
    for i in range(seq_len):
        for j in range(i + 1):
            assert weights[0, i, j] > 0, \
                f"Position {i} should attend to past position {j}"
    
    print("  ✓ Causal mask working")


def test_self_attention():
    """Test that self-attention (Q=K=V) makes sense."""
    print("Testing self-attention...")
    
    batch_size, seq_len, d_model = 1, 3, 4
    X = np.array([[[1, 0, 0, 0],   # Token 1
                   [0, 1, 0, 0],   # Token 2
                   [0, 0, 1, 0]]]) # Token 3
    
    # Self-attention: Q = K = V
    output, weights = scaled_dot_product_attention(X, X, X)
    
    # Weights should be symmetric for Q=K
    # (though not exactly due to numerical precision)
    print(f"  Self-attention weights:\n{weights[0]}")
    
    print("  ✓ Self-attention computed")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING ATTENTION TESTS")
    print("=" * 60 + "\n")
    
    test_softmax()
    test_attention_shapes()
    test_attention_weights_sum()
    test_causal_mask()
    test_self_attention()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)