# Module 05: Parameter Efficient Fine-Tuning (PEFT)

## 🎯 Objective
Training a full 8B model requires massive VRAM. Learn how to fine-tune models on consumer hardware using PEFT techniques.

## 📚 Concepts
1.  **LoRA (Low-Rank Adaptation):** Freezing the main weights and training small "adapter" matrices.
2.  **QLoRA:** Doing LoRA on a 4-bit quantized base model (the secret to running on 1 GPU).
3.  **Catastrophic Forgetting:** Why fine-tuning can make the model dumber at general tasks.

## 🛠️ Tools to Master
- **Unsloth:** An optimized library that makes training 2-5x faster and uses 70% less memory.
- **HuggingFace PEFT/Transformers:** The standard ecosystem.
- **Axolotl:** A config-driven framework for fine-tuning.

## 🧪 Lab: Customizing Llama-3
**Goal:** Teach Llama-3 a new format or style.

### Steps:
1.  Prepare a small dataset (e.g., "Alpaca" format or a custom JSONL).
2.  Use **Unsloth** to fine-tune `Llama-3-8B-Instruct` on your dataset using QLoRA.
3.  Monitor VRAM usage during training.
4.  Save the adapters (LoRA weights).
5.  Run inference with the base model + adapters.

## 📝 Deliverable
A fine-tuned adapter file (`adapter_model.bin`) and a "Before vs After" generation example.
