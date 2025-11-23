"""
TRAINING PIPELINE DIAGRAM
-------------------------

   [Raw Text]       "First Citizen: Before we proceed..."
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
    [Model]         GPT-2 (Transformer)
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
from shared_gpt_model import GPTLanguageModel, GPTConfig
from shared_model_config import DEMO_CONFIG

# --- Configuration ---
cfg = DEMO_CONFIG
train_cfg = cfg["training"]
model_cfg_dict = cfg["model"]

# Override device if CUDA is available
# >>> AIPE NOTE: Hardware Selection
# This is the first check. An AIPE ensures 'cuda' (NVIDIA) or 'rocm' (AMD) is active.
# Note: PyTorch uses the 'cuda' device string for both NVIDIA and AMD ROCm GPUs.
# Fallback to 'cpu' is a performance catastrophe for training.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # >>> AIPE NOTE: GPU Saturation
    # The demo model is tiny (~40k params). A powerful GPU like a 6700 XT or 3090
    # will be "starved" (idle) with a small batch size of 32.
    # We increase the batch size massively to saturate the GPU cores and hide launch overhead.
    print("Optimization: Increasing batch_size to 2048 for GPU saturation.")
    train_cfg.batch_size = 2048

torch.manual_seed(1337)

# ==========================================
# 1. PRE-TRAINING DATA (The Corpus)
# ==========================================
# This is the raw text the model will learn from.
text = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
"""

# ==========================================
# 2. TOKENIZER & DETOKENIZER
# ==========================================
# Convert raw text -> numbers (Tokenizer) and numbers -> text (Detokenizer).

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mappings
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Encoder (Tokenizer)
encode = lambda s: [stoi[c] for c in s]

# Decoder (Detokenizer)
decode = lambda l: ''.join([itos[i] for i in l])

# ==========================================
# 3. DATA SPLITS & LOADER
# ==========================================
# Convert the entire corpus into a Tensor and split into Train/Validation sets.

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# >>> AIPE NOTE: Data Loading Optimization
# For small datasets, move EVERYTHING to GPU VRAM upfront.
# This eliminates the PCIe bottleneck of copying batches during the training loop.
if device == 'cuda':
    train_data = train_data.to(device)
    val_data = val_data.to(device)

def get_batch(split):
    """Generates a small batch of inputs (x) and targets (y)."""
    data = train_data if split == 'train' else val_data
    block_size = model_cfg_dict["block_size"]
    batch_size = train_cfg.batch_size
    
    # >>> AIPE NOTE: Vectorization (Fixing the Sawtooth Pattern)
    # The previous list comprehension `[data[i:i+block_size] for i in ix]` ran in Python on CPU.
    # With batch_size=2048, this loop was too slow, causing the GPU to wait (spikes).
    # We replace it with fully vectorized tensor operations:
    
    # 1. Generate random start indices directly on GPU
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    
    # 2. Create a grid of offsets [0, 1, ..., block_size-1]
    offsets = torch.arange(block_size, device=device)
    
    # 3. Create full index matrix [batch_size, block_size] via broadcasting
    # ix: [B, 1] + offsets: [1, T] -> indices: [B, T]
    indices = ix.unsqueeze(1) + offsets.unsqueeze(0)
    
    # 4. Gather data instantly using fancy indexing
    x = data[indices]
    y = data[indices + 1] # Targets are just shifted by 1
    
    return x, y

@torch.no_grad()
# >>> AIPE NOTE: Memory Efficiency
# @torch.no_grad() is critical. It disables gradient tracking, saving massive amounts of VRAM.
# Without this, validation would likely OOM (Out of Memory) on large models.
def estimate_loss():
    """Estimates loss on train and val sets without backprop."""
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
# 4. MODEL INITIALIZATION
# ==========================================

model_cfg_dict["vocab_size"] = vocab_size
model_cfg_dict["device"] = device
config = GPTConfig(**model_cfg_dict)

model = GPTLanguageModel(config)
# >>> AIPE NOTE: VRAM Management
# Moving the model to GPU consumes VRAM.
# Optimization: For models larger than VRAM, use FSDP (Fully Sharded Data Parallel) or
# CPU offloading (accelerate library) to shard weights across devices.
m = model.to(device)
print(f"{sum(p.numel() for p in m.parameters())/1e3}k parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

# ==========================================
# 5. TRAINING LOOP
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
    # Standard training uses FP32 (float32).
    # Optimization: Use torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    # This speeds up math on Tensor Cores and reduces VRAM usage by 50%.
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n--- Training Complete ---")

# ==========================================
# 6. INFERENCE (Generation)
# ==========================================

print("\n--- Generating Text ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
