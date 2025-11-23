"""
TRAINING PIPELINE DIAGRAM (PARQUET)
-----------------------------------

   [Parquet Files]  (Dataset/train-*.parquet)
       |
       v
   [Pandas DF]      Load & Concatenate Text
       |
       v
  [Tokenizer]       (stoi) Maps characters to integers
       |
       v
   [Integers]       [15, 42, 8, 12, ...]  (Tensor)
       |
       v
  [Data Loader]     Batches of (Context, Target) pairs
       |            x=[15,42,8], y=[42,8,12]
       v
    [Model]         GPT-2 Small (124M Params)
       |            - Embeddings
       |            - Self-Attention Blocks
       |            - Feed-Forward Networks
       v
    [Logits]        Probability distribution over next token
       |
       v
  [Optimizer]       Calculates Loss (CrossEntropy) & Updates Weights
       |
       v
  [Inference]       Generate new tokens -> [Detokenizer] (itos) -> "New Text..."
"""

import torch
import pandas as pd
import os
from shared_gpt_model import GPTLanguageModel, GPTConfig
from shared_model_config import GPT2_SMALL_CONFIG

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

cfg = GPT2_SMALL_CONFIG
train_cfg = cfg["training"]
model_cfg_dict = cfg["model"]

# >>> AIPE NOTE: Tensor Cores
# Enabling TF32 (TensorFloat-32) on NVIDIA Ampere+ GPUs (3090, A100) provides
# significant speedups for matrix multiplications with minimal precision loss.
torch.set_float32_matmul_precision('high') 

# >>> AIPE NOTE: Hardware Selection
# Ensure we are using the GPU. Training GPT-2 on CPU is not feasible.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# ==========================================
# 2. DATA PREPARATION (Parquet Loading)
# ==========================================

print("Loading data from Parquet files...")

def load_text_from_parquet(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")
    df = pd.read_parquet(file_path)
    # Concatenate all text rows into one giant string
    text = "".join(df['text'].tolist())
    return text

# Load Train and Validation data
train_text = load_text_from_parquet('Dataset/train-00000-of-00001.parquet')
val_text = load_text_from_parquet('Dataset/validation-00000-of-00001.parquet')

print(f"Training data length: {len(train_text):,} characters")
print(f"Validation data length: {len(val_text):,} characters")

# ==========================================
# 3. TOKENIZER & DETOKENIZER
# ==========================================

# We combine both texts to ensure we capture all possible characters in the vocab
all_text = train_text + val_text
chars = sorted(list(set(all_text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} characters")

# Mappings
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Encoder & Decoder
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode data to tensors
train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)

# ==========================================
# 4. DATA LOADER
# ==========================================

def get_batch(split):
    """Generates a small batch of inputs (x) and targets (y)."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - model_cfg_dict["block_size"], (train_cfg.batch_size,))
    x = torch.stack([data[i:i+model_cfg_dict["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+model_cfg_dict["block_size"]+1] for i in ix])
    
    # >>> AIPE NOTE: Data Loading Bottleneck
    # Moving data to GPU (.to(device)) inside the training loop can block computation.
    # Optimization: Use pinned_memory=True in DataLoader and prefetch data asynchronously.
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
# >>> AIPE NOTE: Memory Efficiency
# @torch.no_grad() disables gradient tracking, saving massive amounts of VRAM during eval.
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(train_cfg.eval_iters)
        for k in range(train_cfg.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ==========================================
# 5. MODEL INITIALIZATION
# ==========================================

model_cfg_dict["vocab_size"] = vocab_size
model_cfg_dict["device"] = device
config = GPTConfig(**model_cfg_dict)

model = GPTLanguageModel(config)
# >>> AIPE NOTE: VRAM Management
# Moving the model to GPU consumes VRAM.
# Optimization: For models larger than VRAM, use FSDP or CPU offloading.
m = model.to(device)
print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

# ==========================================
# 6. TRAINING LOOP
# ==========================================

print("\n--- Starting Training ---")
for iter in range(train_cfg.max_iters):
    
    # Periodic evaluation
    if iter % train_cfg.eval_interval == 0 or iter == train_cfg.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Forward pass
    xb, yb = get_batch('train')
    
    # >>> AIPE NOTE: Mixed Precision (AMP)
    # Standard training uses FP32. Use torch.amp.autocast for 2x speedup on Tensor Cores.
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n--- Training Complete ---")

# ==========================================
# 7. INFERENCE (Generation)
# ==========================================

print("\n--- Generating Text ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
