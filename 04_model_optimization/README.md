# Module 04: Model Compilation & Optimization

## 🎯 Objective
Go beyond simple quantization. Learn how to "compile" the model graph to fuse operations and optimize memory access patterns for your specific hardware.

## 📚 Concepts
1.  **Kernel Fusion:** Combining multiple operations (e.g., MatMul + Bias + ReLU) into a single GPU kernel launch to reduce memory overhead.
2.  **Graph Capture:** Recording the sequence of GPU operations to replay them efficiently (CUDA Graphs).
3.  **Speculative Decoding:** Using a small "draft" model to predict tokens for the large model to verify.

## 🛠️ Tools to Master
- **TensorRT-LLM:** NVIDIA's state-of-the-art optimization library (hard to setup, massive gains).
- **MIGraphX / ROCm:** The AMD equivalent for your Radeon card.
- **Torch.compile:** PyTorch 2.0's built-in compiler.

## 🧪 Lab: The Compilation Boost
**Goal:** Compile a model and measure the speedup.

### Steps:
1.  Run a standard PyTorch inference loop (baseline).
2.  Use `torch.compile(model)` and run it again.
3.  (Advanced) Attempt to build a TensorRT-LLM engine (or ROCm equivalent) for `Llama-3-8B`.
4.  **Compare:** Latency reduction and startup time (compilation cost).

## 📝 Deliverable
A chart showing "Eager Mode" vs "Compiled Mode" performance.
