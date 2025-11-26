"""
Visualize attention patterns from different heads.
Shows how each head focuses on different aspects.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multihead import MultiHeadAttention

# Set random seed
np.random.seed(42)

# Create sample input
batch_size = 1
seq_len = 8
d_model = 64
num_heads = 8

x = np.random.randn(batch_size, seq_len, d_model)

# Initialize multi-head attention
mha = MultiHeadAttention(d_model, num_heads)

# Compute attention
output, attention_weights = mha(x, x, x)

# Visualize first 4 heads
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for head_idx in range(8):
    ax = axes[head_idx]
    
    # Get attention weights for this head
    head_weights = attention_weights[0, head_idx, :, :]
    
    # Plot heatmap
    im = ax.imshow(head_weights, cmap='viridis', aspect='auto')
    ax.set_title(f'Head {head_idx}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)

plt.suptitle('Multi-Head Attention: Different Heads Learn Different Patterns', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('multihead_attention.png', dpi=150, bbox_inches='tight')
print("✓ Saved multihead_attention.png")
print("\n💡 Notice: Each head has a different attention pattern!")
print("   Some heads might focus on nearby tokens, others on distant ones.")
print("   This diversity is what makes transformers powerful.")
plt.show()
