# Module 06: Distributed Training

## 🎯 Objective
Understand how to scale training across multiple GPUs, even if you only have one right now. This is critical for enterprise roles.

## 📚 Concepts
1.  **DDP (Distributed Data Parallel):** Replicating the model on every GPU, splitting the data.
2.  **FSDP (Fully Sharded Data Parallel):** Sharding the model parameters across GPUs to save memory.
3.  **Gradient Accumulation:** Simulating a large batch size by running multiple small batches before updating weights (crucial for single-GPU).

## 🛠️ Tools to Master
- **DeepSpeed:** Microsoft's optimization library (Zero Redundancy Optimizer - ZeRO stages).
- **Ray Train:** For scaling across clusters.
- **Accelerate:** HuggingFace's wrapper to make distributed training easy.

## 🧪 Lab: The FSDP Simulation
**Goal:** Configure a training run as if you had a cluster.

### Steps:
1.  Set up a training script using `Accelerate`.
2.  Configure it to use **Gradient Accumulation** (steps=4) and **Mixed Precision** (fp16/bf16).
3.  Run the training and observe the memory savings compared to a naive loop.
4.  (Theory) Design a cluster architecture for training a 70B model (how many GPUs? what interconnect?).

## 📝 Deliverable
A "Cluster Architecture Proposal" document for a hypothetical 70B model training run.
