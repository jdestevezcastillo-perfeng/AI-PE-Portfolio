import torch
from transformers import AutoModel

"""
LIBRARY EXPLAINER:
------------------
1. WHAT IS PYTORCH (`torch`)?
   You guessed right! At its core, PyTorch is a library for multiplying matrices (called "Tensors").
   But it has two "superpowers" that make it better than NumPy for AI:
   
   a. GPU Acceleration: 
      Standard arrays (NumPy) live on the CPU. PyTorch Tensors can live on the GPU (NVIDIA/AMD).
      This makes matrix multiplication 100x-1000x faster, which is essential for Deep Learning.
      
   b. Autograd (Automatic Differentiation):
      PyTorch keeps a history of every math operation you do.
      If you multiply A * B = C, PyTorch remembers that.
      This allows it to automatically calculate "gradients" (how much to change A and B to improve C)
      during training. This is the "magic" behind backpropagation.

2. WHAT IS TRANSFORMERS (`transformers`)?
   Created by Hugging Face, this is a high-level library built ON TOP of PyTorch.
   Instead of writing the matrix math for "Self-Attention" from scratch every time,
   you just say `AutoModel.from_pretrained("gpt2")`, and it:
   
   a. Downloads the architecture code (the class definitions).
   b. Downloads the pre-trained weights (the gigabytes of numbers learned by reading the internet).
   c. Loads them into valid PyTorch objects ready for use.
"""

def inspect_gpt2():
    print("Loading GPT-2 model...")
    # We use GPT-2 because it's the simplest standard Transformer to inspect
    model = AutoModel.from_pretrained("gpt2")
    
    config = model.config
    print("\n" + "="*50)
    print("   GPT-2 ARCHITECTURE INSPECTION (Systems View)")
    print("="*50)
    print(f"Model Config:")
    print(f"  - Embedding Dimension (d_model): {config.n_embd}")
    print(f"    (This is the 'width' of the highway. Every token is a vector of this size.)")
    print(f"  - Number of Heads: {config.n_head}")
    print(f"  - Head Dimension: {config.n_embd // config.n_head}")
    print(f"    (d_model / n_heads. Smaller vectors allow parallel 'views' of the data.)")
    print(f"  - Number of Layers: {config.n_layer}")
    
    # Grab the first transformer block (h[0])
    block = model.h[0]
    
    print("\n" + "-"*50)
    print("   INSIDE A SINGLE TRANSFORMER BLOCK (Layer 0)")
    print("-"*50)
    print("A Block consists of two main sub-layers: Attention and MLP.")
    
    # 1. Layer Norm 1
    ln_1_params = sum(p.numel() for p in block.ln_1.parameters())
    print(f"\n[1] Layer Norm 1 (Pre-Attention)")
    print(f"    Params: {ln_1_params:,} (Tiny! Just scale & shift vectors)")
    
    # 2. Self-Attention
    # In GPT-2, c_attn combines Query, Key, and Value projections into one big matrix
    attn = block.attn
    qkv_weight_shape = attn.c_attn.weight.shape
    out_weight_shape = attn.c_proj.weight.shape
    
    print(f"\n[2] Self-Attention (The 'Mixing' Engine)")
    print(f"    Context: This is where tokens 'look at' each other.")
    print(f"    a. QKV Projection (c_attn):")
    print(f"       Shape: {qkv_weight_shape}")
    print(f"       Logic: Input ({qkv_weight_shape[0]}) -> Output ({qkv_weight_shape[1]})")
    print(f"       Why 3x? We generate Query, Key, and Value vectors simultaneously.")
    print(f"       Calculation: {config.n_embd} * 3 = {config.n_embd * 3}")
    
    print(f"    b. Output Projection (c_proj):")
    print(f"       Shape: {out_weight_shape}")
    print(f"       Logic: Projects the multi-head results back to the main highway.")
    
    attn_params = sum(p.numel() for p in attn.parameters())
    print(f"    Total Attention Params: {attn_params:,}")

    # 3. Layer Norm 2
    ln_2_params = sum(p.numel() for p in block.ln_2.parameters())
    print(f"\n[3] Layer Norm 2 (Pre-MLP)")
    print(f"    Params: {ln_2_params:,}")
    
    # 4. MLP (Feed-Forward Network)
    # GPT-2 MLP expands dimensionality by 4x (standard in Transformers)
    mlp = block.mlp
    up_proj_shape = mlp.c_fc.weight.shape
    down_proj_shape = mlp.c_proj.weight.shape
    
    print(f"\n[4] MLP / Feed-Forward (The 'Knowledge' Bank)")
    print(f"    Context: This processes each token individually. It's where 'facts' are stored.")
    print(f"    a. Up Projection (c_fc):")
    print(f"       Shape: {up_proj_shape}")
    print(f"       Logic: Expands the highway 4x wider to process features.")
    print(f"       Calculation: {config.n_embd} * 4 = {config.n_embd * 4}")
    
    print(f"    b. Down Projection (c_proj):")
    print(f"       Shape: {down_proj_shape}")
    print(f"       Logic: Projects back down to the original highway size.")
    
    mlp_params = sum(p.numel() for p in mlp.parameters())
    print(f"    Total MLP Params: {mlp_params:,}")
    print(f"    (Notice: MLP is usually ~2x larger than Attention!)")
    
    # Totals
    total_block_params = ln_1_params + attn_params + ln_2_params + mlp_params
    print("\n" + "-"*50)
    print("   BLOCK SUMMARY")
    print("-"*50)
    print(f"Total Parameters per Block: {total_block_params:,}")
    
    # Memory Calculation (FP32 = 4 bytes per param)
    mem_fp32 = total_block_params * 4 / (1024 * 1024)
    mem_fp16 = total_block_params * 2 / (1024 * 1024)
    mem_int8 = total_block_params * 1 / (1024 * 1024)
    
    print(f"Memory Usage (Weights Only):")
    print(f"  FP32 (Standard Training):   {mem_fp32:.2f} MB")
    print(f"  FP16 (Standard Inference):  {mem_fp16:.2f} MB")
    print(f"  INT8 (Quantized):           {mem_int8:.2f} MB")
    
    print("\n" + "="*50)
    print("   FULL MODEL ESTIMATION")
    print("="*50)
    total_body_params = total_block_params * config.n_layer
    print(f"Body Parameters (12 layers): {total_body_params:,}")
    print(f"Embeddings (Vocab * d_model): {config.vocab_size * config.n_embd:,}")
    print(f"Total Model Params: {total_body_params + (config.vocab_size * config.n_embd):,}")
    print("="*50)

if __name__ == "__main__":
    inspect_gpt2()
