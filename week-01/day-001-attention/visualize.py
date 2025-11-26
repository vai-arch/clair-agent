"""
Visualize attention weights as a heatmap.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from attention import scaled_dot_product_attention, create_causal_mask

# Set random seed
np.random.seed(42)

# Create sample sequence
seq_len = 8
d_model = 16

# Create queries, keys, values
Q = np.random.randn(1, seq_len, d_model)
K = np.random.randn(1, seq_len, d_model)
V = np.random.randn(1, seq_len, d_model)

# Compute attention without mask
_, weights_no_mask = scaled_dot_product_attention(Q, K, V)

# Compute attention with causal mask
mask = create_causal_mask(seq_len)
_, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: No mask
im1 = axes[0].imshow(weights_no_mask[0], cmap='viridis', aspect='auto')
axes[0].set_title('Attention Weights (No Mask)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Key Position')
axes[0].set_ylabel('Query Position')
plt.colorbar(im1, ax=axes[0])

# Plot 2: Causal mask
im2 = axes[1].imshow(weights_masked[0], cmap='viridis', aspect='auto')
axes[1].set_title('Attention Weights (Causal Mask)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Key Position')
axes[1].set_ylabel('Query Position')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('attention_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved attention_heatmap.png")
plt.show()