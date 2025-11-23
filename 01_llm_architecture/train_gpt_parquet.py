import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import os
from gpt_model import GPTLanguageModel, GPTConfig

# --- Hyperparameters ---
batch_size = 4       # Reduced to 4 to fit in VRAM
torch.set_float32_matmul_precision('high') # Enable TF32 for 3090
block_size = 1024    # GPT-2 context length
max_iters = 5000     # More training steps
eval_interval = 500
learning_rate = 6e-4 # Standard GPT-2 learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 768         # GPT-2 Small embedding dimension
n_head = 12          # GPT-2 Small heads
n_layer = 12         # GPT-2 Small layers
dropout = 0.2        # Dropout

torch.manual_seed(1337)

# --- Data Preparation ---
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

# Create a mapping from characters to integers (Tokenizer)
# We combine both texts to ensure we capture all possible characters in the vocab
all_text = train_text + val_text
chars = sorted(list(set(all_text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} characters")
# print(f"Vocabulary: {''.join(chars)}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Encode data
train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)

# Data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Model Initialization ---
config = GPTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    device=device
)

model = GPTLanguageModel(config)
m = model.to(device)
print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n--- Starting Training ---")
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n--- Training Complete ---")
print("\n--- Generating Text ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
