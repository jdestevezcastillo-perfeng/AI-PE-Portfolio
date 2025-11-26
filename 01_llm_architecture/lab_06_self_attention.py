"""
SELF-ATTENTION FROM SCRATCH
---------------------------

   [Input Embeddings]  (Batch, Time, Channels)
           |
           +----------------+----------------+
           |                |                |
           v                v                v
      [Query (Q)]      [Key (K)]        [Value (V)]
      (What I look for) (What I contain) (What I pass on)
           |                |                |
           +-------+--------+                |
                   |                         |
                   v                         |
           [Dot Product] (Q @ K.T)           |
           (Similarity Scores)               |
                   |                         |
                   v                         |
             [Scale] (/ sqrt(head_size))     |
                   |                         |
                   v                         |
             [Mask] (Set future to -inf)     |
                   |                         |
                   v                         |
            [Softmax] (Normalize to 0..1)    |
                   |                         |
                   +--------+----------------+
                            |
                            v
                    [Weighted Sum] (Scores @ V)
                            |
                            v
                     [Output Context]

   >>> AIPE NOTE: The "Heart" of the Transformer
   This script implements the math of `nn.MultiheadAttention` manually.
   Understanding `Q @ K.T` is crucial for understanding how LLMs "think".
"""

import torch
import torch.nn.functional as F
import math

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

torch.manual_seed(1337)

# Hyperparameters (Small for demonstration)
B = 1  # Batch size
T = 8  # Time / Sequence length (e.g., 8 tokens)
C = 32 # Channels / Embedding dimension (e.g., 32-dimensional vectors)
head_size = 16 # Dimension of each head (usually C // n_heads)

print(f"Config: Batch={B}, Time={T}, Channels={C}, Head Size={head_size}")

# Input: A batch of token embeddings
# Shape: (B, T, C)
x = torch.randn(B, T, C)
print(f"Input shape: {x.shape}")

# ==========================================
# 2. PROJECTIONS (The Learnable Weights)
# ==========================================

# In a real model, these are nn.Linear layers.
# Here we define the raw weight matrices manually.
# We project from 'C' (embedding dim) to 'head_size'.

key_layer   = torch.nn.Linear(C, head_size, bias=False)
query_layer = torch.nn.Linear(C, head_size, bias=False)
value_layer = torch.nn.Linear(C, head_size, bias=False)

# ==========================================
# 3. CALCULATE Q, K, V
# ==========================================

# k: (B, T, head_size)
k = key_layer(x)
# q: (B, T, head_size)
q = query_layer(x)
# v: (B, T, head_size)
v = value_layer(x)

print(f"Keys shape: {k.shape}")
print(f"Queries shape: {q.shape}")
print(f"Values shape: {v.shape}")

# ==========================================
# 4. ATTENTION SCORES (The "Search")
# ==========================================

# Equation: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

# Transpose K to align dimensions for dot product
# (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
wei = q @ k.transpose(-2, -1) 

print(f"Raw weights (QK^T) shape: {wei.shape}")

# Scale by sqrt(head_size) to keep variance stable
# (Important for gradient stability)
wei = wei * (head_size ** -0.5)

# ==========================================
# 5. CAUSAL MASKING (The "Time Travel" Prevention)
# ==========================================

# We want the model to only attend to the PAST, not the FUTURE.
# We create a lower-triangular matrix of ones.
tril = torch.tril(torch.ones(T, T))

# Where tril == 0 (future positions), set weight to -infinity.
# This ensures softmax will turn them into 0.
wei = wei.masked_fill(tril == 0, float('-inf'))

print("\n>>> AIPE NOTE: Masked Weights (First row sees only itself)")
print(wei[0, 0, :]) # First token's view

# ==========================================
# 6. SOFTMAX (Normalization)
# ==========================================

# Convert scores to probabilities (sum to 1 across the last dimension)
wei = F.softmax(wei, dim=-1)

print("\n>>> AIPE NOTE: Attention Probabilities (Rows sum to 1)")
print(wei[0, 0, :]) # First token attends 100% to itself
print(wei[0, -1, :]) # Last token attends to everyone

# ==========================================
# 7. AGGREGATION (The "Retrieval")
# ==========================================

# Weighted sum of Values
# (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
out = wei @ v

print(f"\nOutput shape: {out.shape}")

# ==========================================
# 8. VERIFICATION
# ==========================================

print("\n>>> AIPE NOTE: Interpretation")
print("For the last token (index 7), the output is a weighted mix of all previous 8 tokens.")
print("The weights are determined by how similar its Query vector was to their Key vectors.")

# Let's verify the manual math against PyTorch's scaled_dot_product_attention (SDPA)
# SDPA is the optimized kernel used in production (FlashAttention).
print("\n--- Verifying against torch.nn.functional.scaled_dot_product_attention ---")

# SDPA expects (Batch, Heads, Time, Head_Dim) usually, but works with (B, T, H) if we are careful.
# For simplicity, we just pass our tensors.
# Note: SDPA handles the scaling internally.
# Note: is_causal=True handles the masking automatically.

# Re-compute Q, K, V fresh to be sure
k = key_layer(x)
q = query_layer(x)
v = value_layer(x)

# PyTorch's SDPA implementation
out_torch = F.scaled_dot_product_attention(q, k, v, is_causal=True)

print(f"Manual Output (first 3 values): {out[0, -1, :3].tolist()}")
print(f"Torch  Output (first 3 values): {out_torch[0, -1, :3].tolist()}")

# Check if they are close
diff = (out - out_torch).abs().max()
print(f"Max difference: {diff.item():.6f}")

if diff < 1e-5:
    print("SUCCESS: Manual implementation matches PyTorch optimized kernel!")
else:
    print("WARNING: Significant difference found.")
