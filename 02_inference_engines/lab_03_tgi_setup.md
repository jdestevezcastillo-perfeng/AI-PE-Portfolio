# Lab 03: Text Generation Inference (TGI) Setup

**Goal:** Run Hugging Face's production-grade inference server (TGI) on your NVIDIA RTX 3090.

## 1. Prerequisites

You need **Docker** with the **NVIDIA Container Toolkit** installed.

### Check if you have it

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

If this prints your GPU info, you are good to go. If not, you need to install the toolkit.

## 2. Running TGI

We will run TGI serving the `Llama-3-8B-Instruct` model (or a smaller one like `TinyLlama` for testing).

### Command (Run in terminal)

```bash
model=meta-llama/Llama-3.1-8B-Instruct
# Or use a gated-free model for testing:
# model=TinyLlama/TinyLlama-1.1B-Chat-v1.0

volume=$PWD/data # Share a volume with the Docker container to avoid downloading weights every time

docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $volume:/data \
  ghcr.io/huggingface/text-generation-inference:2.0 \
  --model-id $model \
  --quantize bitsandbytes # Optional: use 8-bit quantization for lower VRAM
```

## 3. Querying TGI

Once the server is running (you see "Connected" in logs), you can query it via curl or Python.

### Python Client

```python
import requests

headers = {
    "Content-Type": "application/json",
}

data = {
    "inputs": "What is the capital of France?",
    "parameters": {
        "max_new_tokens": 20,
    }
}

response = requests.post("http://127.0.0.1:8080/generate", headers=headers, json=data)
print(response.json())
```
