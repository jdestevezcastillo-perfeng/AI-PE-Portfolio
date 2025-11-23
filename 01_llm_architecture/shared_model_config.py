"""
Model Configurations
--------------------
Centralized configuration for GPT models to ensure consistency and separation of concerns.
"""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int
    max_iters: int
    eval_interval: int
    learning_rate: float
    eval_iters: int
    device: str = 'cuda' # Will fallback to cpu in script if not available

# --- 1. Demo / Debug Configuration ---
# Tiny model for quick CPU testing and code verification.
DEMO_CONFIG = {
    "model": {
        "block_size": 8,        # Context length
        "vocab_size": 65,       # Character-level tokenizer (Shakespeare)
        "n_layer": 3,
        "n_head": 4,
        "n_embd": 32,
        "dropout": 0.0,
    },
    "training": TrainingConfig(
        batch_size=32,
        max_iters=3000,
        eval_interval=300,
        learning_rate=1e-3,
        eval_iters=200,
        device='cpu' # Force CPU for demo to ensure it runs everywhere
    )
}

# --- 2. GPT-2 Small Configuration ---
# Standard 124M parameter model. Requires GPU (approx 6GB VRAM for training).
GPT2_SMALL_CONFIG = {
    "model": {
        "block_size": 1024,
        "vocab_size": 50257,    # Standard GPT-2 BPE vocab size (approx)
                                # Note: In our script we calculate this dynamically from data
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "dropout": 0.2,
    },
    "training": TrainingConfig(
        batch_size=4,           # Low batch size to fit in consumer GPU VRAM
        max_iters=5000,
        eval_interval=500,
        learning_rate=6e-4,
        eval_iters=20,
        device='cuda'
    )
}
