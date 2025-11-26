"""
LAB 01: BASELINE INFERENCE (Hugging Face Pipeline)
--------------------------------------------------
Goal: Establish a baseline performance metric for standard PyTorch inference.
We will measure:
1. Time to First Token (TTFT) - roughly (Latency)
2. Tokens Per Second (TPS) - (Throughput)
3. Memory Usage (Peak VRAM)

Note: This uses the standard `transformers` library, which is not optimized for
high-throughput serving (no continuous batching, paged attention, etc.).
"""

import time
import torch
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ==========================================
# 0. CONFIGURATION
# ==========================================

# Using a smaller model for the baseline to ensure it fits easily.
# Llama-3-8B is the target, but for quick dev/test we can use something smaller if needed.
# Let's stick to a standard small model or the user's preferred one.
# We'll default to GPT-2 for instant feedback, or Llama-3-8B-Instruct if available.
# Given the user has a 3090 (24GB VRAM), we can comfortably run Llama-3-8B-Instruct.

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" 
# NOTE: If you don't have access to Llama-3, swap for "gpt2" or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# ==========================================
# 1. SETUP MODEL
# ==========================================

def get_vram_usage():
    if DEVICE == "cuda":
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

print(f"Loading model: {MODEL_ID}...")
start_load = time.time()

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Load in float16 to match typical inference precision
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
except Exception as e:
    print(f"\n[ERROR] Could not load {MODEL_ID}. You might need to login to Hugging Face or choose a public model.")
    print(f"Error details: {e}")
    print("\nFalling back to 'gpt2' for demonstration...")
    MODEL_ID = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

end_load = time.time()
print(f"Model loaded in {end_load - start_load:.2f}s")
print(f"VRAM Usage: {get_vram_usage():.2f} GB")

# ==========================================
# 2. INFERENCE LOOP
# ==========================================

prompts = [
    "Explain the theory of relativity in simple terms.",
    "Write a python function to calculate the fibonacci sequence.",
    "What are the main differences between TCP and UDP?",
    "Compose a haiku about artificial intelligence."
]

print("\n--- Starting Inference Benchmark ---")

total_tokens = 0
total_time = 0

# Warmup
print("Warming up...")
_ = model.generate(**tokenizer("Hello world", return_tensors="pt").to(DEVICE), max_new_tokens=10)

for i, prompt in enumerate(prompts):
    print(f"\nPrompt {i+1}: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.shape[1]
    
    start_gen = time.time()
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.7
        )
    
    end_gen = time.time()
    
    # Metrics
    generated_ids = outputs[0][input_len:]
    num_tokens = len(generated_ids)
    duration = end_gen - start_gen
    tps = num_tokens / duration
    
    total_tokens += num_tokens
    total_time += duration
    
    print(f"Generated {num_tokens} tokens in {duration:.2f}s")
    print(f"Speed: {tps:.2f} tokens/sec")
    print(f"Output: {tokenizer.decode(generated_ids, skip_special_tokens=True)[:50]}...")

# ==========================================
# 3. SUMMARY
# ==========================================

avg_tps = total_tokens / total_time
print("\n==========================================")
print(f"BASELINE SUMMARY ({MODEL_ID})")
print("==========================================")
print(f"Average Throughput: {avg_tps:.2f} tokens/sec")
print(f"Peak VRAM Usage:    {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print("==========================================")
