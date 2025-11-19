# AI Performance Engineering Glossary

## A
- **Attention Mechanism:** The core component of Transformers that allows the model to weigh the importance of different words in the context.
- **AWQ (Activation-aware Weight Quantization):** A quantization method that preserves the precision of important weights based on activation data.

## B
- **Batch Size:** The number of samples processed before the model is updated (training) or processed simultaneously (inference).

## C
- **Context Window:** The maximum number of tokens the model can consider at one time.
- **Continuous Batching:** An inference technique that inserts new requests into a running batch as soon as previous ones finish, maximizing GPU utilization.

## D
- **DDP (Distributed Data Parallel):** A training strategy where the model is replicated on every GPU, and data is split.

## E
- **Embedding:** A vector representation of text where similar meanings are close in vector space.

## F
- **Fine-Tuning:** Taking a pre-trained model and training it further on a specific dataset.
- **FSDP (Fully Sharded Data Parallel):** A training strategy that shards model parameters, gradients, and optimizer states across GPUs to save memory.

## G
- **GGUF:** A file format for storing quantized models, optimized for CPU and Apple Silicon inference (used by llama.cpp).
- **GPU Utilization:** The percentage of time the GPU compute units are active.

## H
- **Hallucination:** When an LLM generates factually incorrect or nonsensical information confidently.

## I
- **Inference:** The process of using a trained model to generate predictions (text).

## K
- **KV Cache (Key-Value Cache):** Storing the calculated attention keys and values for previous tokens to avoid recomputing them for every new token.

## L
- **Latency:** The time it takes to receive a response (often measured as TTFT or total time).
- **LoRA (Low-Rank Adaptation):** A PEFT technique that freezes the model and trains small rank decomposition matrices.

## M
- **Model Parallelism:** Splitting a single model across multiple GPUs because it doesn't fit on one.

## O
- **Overfitting:** When a model learns the training data too well and fails to generalize to new data.

## P
- **PagedAttention:** A memory management technique (vLLM) that stores KV cache in non-contiguous memory blocks, reducing fragmentation.
- **Parameter:** A weight or bias in the neural network.
- **PEFT (Parameter-Efficient Fine-Tuning):** Methods to fine-tune models with minimal compute/memory (e.g., LoRA).
- **Perplexity:** A metric measuring how well a probability model predicts a sample (lower is better).

## Q
- **Quantization:** Reducing the precision of model weights (e.g., from 16-bit float to 4-bit integer) to save memory and speed up compute.
- **QLoRA:** Quantized LoRA.

## R
- **RAG (Retrieval-Augmented Generation):** Providing external data to the LLM as context to answer questions.
- **ROCm:** AMD's open software platform for GPU computing (competitor to NVIDIA CUDA).

## T
- **Temperature:** A hyperparameter controlling the randomness of the model's output.
- **Throughput:** The number of tokens generated per second.
- **Token:** A chunk of text (word or sub-word) that the model processes.
- **TPOT (Time Per Output Token):** The average time to generate one token (inverse of decoding speed).
- **Transformer:** The neural network architecture behind modern LLMs.
- **TTFT (Time To First Token):** The latency from sending the request to seeing the first character.

## V
- **vLLM:** A high-throughput, memory-efficient LLM serving engine.
- **VRAM (Video RAM):** The memory on the GPU, critical for loading large models.

## Z
- **Zero-Shot:** Asking the model to do a task without giving it any examples.
