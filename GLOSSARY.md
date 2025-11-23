# AI Performance Engineering Glossary

## A

- **Attention Mechanism:** The core component of Transformers that allows the model to weigh the importance of different words in the context. Computes Query, Key, Value matrices to determine token relationships.
- **AWQ (Activation-aware Weight Quantization):** A quantization method that preserves the precision of important weights based on activation data. Generally better quality than GPTQ at same bit-width.
- **Autoregressive:** Generating output one token at a time, where each new token depends on all previous tokens. The standard approach for LLM text generation.

## B

- **Batch Size:** The number of samples processed before the model is updated (training) or processed simultaneously (inference).
- **BF16 (Brain Float 16):** A 16-bit floating-point format optimized for deep learning, with the same range as FP32 but reduced precision. Preferred over FP16 for training stability.

## C

- **Compute Bound:** When GPU compute units are the bottleneck, not memory bandwidth. Larger batch sizes typically become compute bound.
- **Context Window:** The maximum number of tokens the model can consider at one time. Longer contexts require more VRAM for KV cache.
- **Continuous Batching:** An inference technique that inserts new requests into a running batch as soon as previous ones finish, maximizing GPU utilization.

## D

- **DDP (Distributed Data Parallel):** A training strategy where the model is replicated on every GPU, and data is split across GPUs.
- **Decoding:** The process of generating output tokens from the model. Can be greedy (pick highest probability) or sampling-based (temperature, top-k, top-p).

## E

- **Embedding:** A vector representation of text where similar meanings are close in vector space. The first layer of a transformer converts tokens to embeddings.
- **EXL2:** An optimized quantization format for ExLlamaV2, offering flexible bit-widths and high inference speed.

## F

- **Fine-Tuning:** Taking a pre-trained model and training it further on a specific dataset to adapt it to a particular task or domain.
- **FP16 (Half Precision):** 16-bit floating-point format. Reduces memory usage by half compared to FP32 with minimal quality loss.
- **FP32 (Full Precision):** 32-bit floating-point format. The default precision for model training.
- **FSDP (Fully Sharded Data Parallel):** A training strategy that shards model parameters, gradients, and optimizer states across GPUs to save memory.

## G

- **GGUF:** A file format for storing quantized models, optimized for CPU and Apple Silicon inference (used by llama.cpp and Ollama).
- **GPTQ:** A post-training quantization method that uses calibration data to minimize quantization error. Commonly used for 4-bit models.
- **GPU Utilization:** The percentage of time the GPU compute units are active. Low utilization often indicates memory-bound workloads.

## H

- **Hallucination:** When an LLM generates factually incorrect or nonsensical information confidently.

## I

- **Inference:** The process of using a trained model to generate predictions (text). Consists of prefill (processing prompt) and decode (generating tokens) phases.

## K

- **KV Cache (Key-Value Cache):** Storing the calculated attention keys and values for previous tokens to avoid recomputing them for every new token. Major consumer of VRAM during inference.

## L

- **Latency:** The time it takes to receive a response (often measured as TTFT or total time).
- **Loki:** A log aggregation system by Grafana Labs, designed to be cost-effective and easy to operate. Part of the PLG stack.
- **LoRA (Low-Rank Adaptation):** A PEFT technique that freezes the model and trains small rank decomposition matrices, reducing trainable parameters by 10-100x.

## M

- **Memory Bound:** When GPU memory bandwidth is the bottleneck, not compute. Small batch sizes are typically memory bound.
- **Model Parallelism:** Splitting a single model across multiple GPUs because it doesn't fit on one.
- **MLP (Multi-Layer Perceptron):** The feed-forward network in each transformer block. Typically 2/3 of model parameters.

## O

- **Ollama:** A tool for running LLMs locally with easy model management. Uses GGUF format and llama.cpp backend.
- **OpenTelemetry (OTel):** A vendor-neutral observability framework for traces, metrics, and logs. Industry standard for distributed tracing.
- **OTLP:** OpenTelemetry Protocol. The native protocol for sending telemetry data to OpenTelemetry collectors.
- **Overfitting:** When a model learns the training data too well and fails to generalize to new data.

## P

- **PagedAttention:** A memory management technique (vLLM) that stores KV cache in non-contiguous memory blocks, reducing fragmentation and enabling larger batch sizes.
- **Parameter:** A weight or bias in the neural network. A 7B model has 7 billion parameters.
- **PEFT (Parameter-Efficient Fine-Tuning):** Methods to fine-tune models with minimal compute/memory (e.g., LoRA, QLoRA).
- **Perplexity:** A metric measuring how well a probability model predicts a sample. Lower is better. Used to evaluate quantization quality loss.
- **PLG Stack:** Prometheus, Loki, Grafana - a popular open-source observability stack.
- **Prefill:** The first phase of inference where the model processes the entire input prompt in parallel. Compute-intensive.
- **Prometheus:** A time-series database and monitoring system. Scrapes metrics from exporters at regular intervals.
- **Promtail:** A log shipping agent that sends logs to Loki. Part of the Grafana ecosystem.

## Q

- **Quantization:** Reducing the precision of model weights (e.g., from 16-bit float to 4-bit integer) to save memory and speed up compute.
- **QLoRA:** Quantized LoRA - fine-tuning a quantized model using LoRA adapters, enabling fine-tuning on consumer GPUs.

## R

- **RAG (Retrieval-Augmented Generation):** Providing external data to the LLM as context to answer questions. Reduces hallucinations for domain-specific queries.
- **ROCm:** AMD's open software platform for GPU computing (competitor to NVIDIA CUDA).

## S

- **Scrape Interval:** How often Prometheus collects metrics from exporters. Typically 5-15 seconds.
- **Softmax:** A function that converts logits to probabilities summing to 1. Used in attention and final output layer.

## T

- **Temperature:** A hyperparameter controlling the randomness of the model's output. Lower = more deterministic, higher = more creative.
- **Tempo:** A distributed tracing backend by Grafana Labs. Stores and queries OpenTelemetry traces.
- **Tensor Core:** Specialized hardware on modern GPUs for accelerating matrix operations. Essential for fast transformer inference.
- **Throughput:** The number of tokens generated per second. Higher is better for batch processing.
- **Token:** A chunk of text (word or sub-word) that the model processes. Average English word is ~1.3 tokens.
- **TPOT (Time Per Output Token):** The average time to generate one token during the decode phase. Inverse of decoding speed.
- **TPS (Tokens Per Second):** Throughput metric. eval_count / (eval_duration / 1e9).
- **Transformer:** The neural network architecture behind modern LLMs. Uses self-attention to process sequences.
- **TTFT (Time To First Token):** The latency from sending the request to receiving the first token. Critical for user-perceived responsiveness.

## V

- **vLLM:** A high-throughput, memory-efficient LLM serving engine featuring PagedAttention and continuous batching.
- **VRAM (Video RAM):** The memory on the GPU, critical for loading large models. A 7B FP16 model needs ~14GB VRAM.

## Z

- **Zero-Shot:** Asking the model to do a task without giving it any examples in the prompt.
