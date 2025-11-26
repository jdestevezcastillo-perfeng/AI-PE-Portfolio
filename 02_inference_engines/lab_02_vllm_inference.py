"""
LAB 02: vLLM INFERENCE
----------------------
Goal: Demonstrate high-performance inference using vLLM.
We will measure:
1. Throughput (Tokens/sec)
2. Comparison vs Baseline

vLLM Features used:
- PagedAttention (Memory optimization)
- Continuous Batching (Throughput optimization)
"""

import time
import torch
from vllm import LLM, SamplingParams

# ==========================================
# 0. CONFIGURATION
# ==========================================

# Use the same model as baseline for fair comparison
# If baseline used GPT-2, we use GPT-2 here.
MODEL_ID = "gpt2" 
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" # Uncomment if available

print(f"Initializing vLLM with model: {MODEL_ID}...")

# ==========================================
# 1. SETUP vLLM
# ==========================================

# Initialize the LLM engine
# gpu_memory_utilization=0.9 ensures we use most of the GPU
llm = LLM(model=MODEL_ID, gpu_memory_utilization=0.9)

# Sampling parameters
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)

# ==========================================
# 2. INFERENCE LOOP
# ==========================================

prompts = [
    "Explain the theory of relativity in simple terms.",
    "Write a python function to calculate the fibonacci sequence.",
    "What are the main differences between TCP and UDP?",
    "Compose a haiku about artificial intelligence."
]

print("\n--- Starting vLLM Inference ---")

start_time = time.time()

# vLLM handles batching automatically! We can pass the whole list.
outputs = llm.generate(prompts, sampling_params)

end_time = time.time()
total_duration = end_time - start_time

# ==========================================
# 3. METRICS & OUTPUT
# ==========================================

total_tokens = 0

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    num_tokens = len(output.outputs[0].token_ids)
    total_tokens += num_tokens
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text[:50]}...")
    print(f"Tokens: {num_tokens}")

avg_tps = total_tokens / total_duration

print("\n==========================================")
print(f"vLLM SUMMARY ({MODEL_ID})")
print("==========================================")
print(f"Total Time:       {total_duration:.2f}s")
print(f"Total Tokens:     {total_tokens}")
print(f"Average Throughput: {avg_tps:.2f} tokens/sec")
print("==========================================")
