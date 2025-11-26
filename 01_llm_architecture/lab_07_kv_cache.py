"""
KV CACHE OPTIMIZATION
---------------------

   [Naive Generation]              [With KV Cache]
   Step 1: [A, B] -> C             Step 1: [A, B] -> C
                                           Cache: {K_ab, V_ab}
   Step 2: [A, B, C] -> D          Step 2: [C] + Cache -> D
           (Recomputes A, B)               (Reuses K_ab, V_ab)
                                           Cache: {K_abc, V_abc}
   Step 3: [A, B, C, D] -> E       Step 3: [D] + Cache -> E
           (Recomputes A, B, C)            (Reuses K_abc, V_abc)

   >>> AIPE NOTE: The Speedup
   - Naive: O(N^2) complexity for generating N tokens.
   - Cached: O(N) complexity (linear time).
   - Critical for long contexts (e.g., RAG, Chatbots).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

# ==========================================
# 1. ATTENTION WITH CACHE SUPPORT
# ==========================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len=1024):
        super().__init__()
        assert d_model % n_head == 0
        self.d_head = d_model // n_head
        self.n_head = n_head
        
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        
        # Register a causal mask
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                     .view(1, 1, max_len, max_len))

    def forward(self, x, kv_cache=None):
        """
        x: (Batch, Seq_Len, Dim)
        kv_cache: Tuple(K_cache, V_cache) or None
        """
        B, T, C = x.size()
        
        # Calculate Query, Key, Value
        # q: (B, H, T, Dh)
        q = self.query(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # >>> AIPE NOTE: KV Cache Logic
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            
            # If we have a cache, we are likely generating the *next* token.
            # The input 'x' is just the single new token (T=1).
            # But we need to attend to all past tokens (in the cache).
            
            # Append new k, v to the cache
            # Cache shape: (B, H, Past_T, Dh)
            new_k = torch.cat([k_cache, k], dim=2)
            new_v = torch.cat([v_cache, v], dim=2)
            
            # Use the FULL history for attention
            k = new_k
            v = new_v
            
            # Update cache for return
            current_cache = (new_k, new_v)
        else:
            # No cache provided (first step or naive mode), initialize it
            current_cache = (k, v)

        # Attention Calculation
        # q is (B, H, 1, Dh) if cached, or (B, H, T, Dh) if naive
        # k is (B, H, Total_T, Dh)
        
        # (B, H, T_q, Dh) @ (B, H, Dh, T_k) -> (B, H, T_q, T_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Causal Masking
        # If we are using cache, we are usually at the last step, attending to everything in the past.
        # So masking is trivial (we see everything up to now).
        # If naive, we need the standard triangular mask.
        total_t = k.size(2)
        if kv_cache is None:
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, H, T, Dh)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), current_cache

# ==========================================
# 2. GENERATION LOOPS
# ==========================================

def generate_naive(model, input_ids, steps=20):
    """Re-runs the model on the full sequence every step."""
    # Start with input
    current_ids = input_ids.clone()
    
    start_time = time.time()
    for _ in range(steps):
        # Forward pass on the ENTIRE sequence
        # We discard the cache output here
        output, _ = model(current_ids, kv_cache=None)
        
        # Take the last token's embedding as the "prediction" (simplified)
        # In a real model, this would go through a LayerNorm + Head
        next_token_embed = output[:, -1, :]
        
        # Dummy "sampling": just append a zero (we only care about performance here)
        next_id = torch.zeros((1, 1, 32), device=input_ids.device) # Dummy embedding
        
        # In this simplified demo, we are appending embeddings, not IDs, 
        # because we don't have a token embedding layer.
        current_ids = torch.cat([current_ids, next_id], dim=1)
        
    end_time = time.time()
    return end_time - start_time

def generate_cached(model, input_ids, steps=20):
    """Uses KV Cache to only compute the new token."""
    
    # 1. Prefill: Process the prompt once to build the initial cache
    current_input = input_ids.clone()
    
    start_time = time.time()
    
    # Initial pass
    output, cache = model(current_input, kv_cache=None)
    
    # 2. Generation Loop
    for _ in range(steps):
        # We only feed the LAST token (dummy embedding in this demo)
        next_input = torch.zeros((1, 1, 32), device=input_ids.device)
        
        # Forward pass with cache
        # Note: We pass 'next_input' (T=1), not the full sequence
        output, cache = model(next_input, kv_cache=cache)
        
        # 'cache' is updated automatically in the forward pass
        
    end_time = time.time()
    return end_time - start_time

# ==========================================
# 3. BENCHMARK
# ==========================================

def main():
    torch.manual_seed(42)
    
    # Config
    B = 1
    T = 100 # Initial prompt length
    C = 32  # Embedding dim
    H = 4   # Heads
    STEPS = 200 # Tokens to generate
    
    print(f"Config: Prompt={T}, Gen_Steps={STEPS}, Dim={C}")
    
    # Create model and dummy input
    model = CausalSelfAttention(d_model=C, n_head=H, max_len=2048)
    # Dummy input embeddings (Batch, Time, Dim)
    input_emb = torch.randn(B, T, C)
    
    print("\n--- Running Naive Generation (Recompute All) ---")
    time_naive = generate_naive(model, input_emb, steps=STEPS)
    print(f"Time: {time_naive:.4f}s")
    
    print("\n--- Running Cached Generation (KV Cache) ---")
    time_cached = generate_cached(model, input_emb, steps=STEPS)
    print(f"Time: {time_cached:.4f}s")
    
    print(f"\n>>> Speedup: {time_naive / time_cached:.2f}x")
    
    if time_naive / time_cached > 1.5:
        print("SUCCESS: KV Cache demonstrates significant speedup!")
    else:
        print("NOTE: Speedup might be small for such tiny models/sequences, but complexity is O(N) vs O(N^2).")

if __name__ == "__main__":
    main()
