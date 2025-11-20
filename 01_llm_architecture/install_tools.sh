#!/bin/bash

# 01_llm_architecture/install_tools.sh
# Installs necessary tools for the LLM Architecture & Quantization module.
# Detects NVIDIA vs AMD GPU and installs appropriate versions.

set -e  # Exit immediately if a command exits with a non-zero status.

echo "================================================================="
echo "   AI Performance Engineering - Module 01 Tool Installer"
echo "================================================================="

# Check if running as root (we shouldn't, except for apt)
if [ "$EUID" -eq 0 ]; then
  echo "Please do not run this script as root (don't use sudo)."
  echo "The script will ask for sudo password when installing system packages."
  exit 1
fi

# 1. System Dependencies
echo "[*] Installing system dependencies..."
sudo apt update
sudo apt install -y build-essential cmake git python3-venv

# 2. Python Environment
echo "[*] Setting up Python environment..."
if [ -d ".venv" ]; then
    echo "    Found existing .venv. verifying ownership..."
    if [ ! -w ".venv" ]; then
        echo "    Error: .venv exists but is not writable. It might be owned by root."
        echo "    Please remove it: sudo rm -rf .venv"
        exit 1
    fi
else
    echo "    Creating .venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "    Upgrading pip..."
pip install --upgrade pip

# 3. GPU Detection
echo "[*] Detecting Hardware..."
GPU_TYPE="CPU"

if command -v nvidia-smi &> /dev/null; then
    echo "    Detected NVIDIA GPU."
    GPU_TYPE="NVIDIA"
elif command -v rocminfo &> /dev/null; then
    echo "    Detected AMD GPU (ROCm via rocminfo)."
    GPU_TYPE="AMD"
elif [ -d "/opt/rocm" ] || [ -e "/dev/kfd" ]; then
    echo "    Detected AMD GPU (Found /opt/rocm or /dev/kfd), assuming ROCm."
    echo "    Note: 'rocminfo' might have failed due to permissions. Ensure you are in the 'render' group."
    GPU_TYPE="AMD"
else
    echo "    No GPU detected (or drivers missing). Defaulting to CPU mode."
fi

# 4. Install PyTorch
echo "[*] Installing PyTorch for $GPU_TYPE..."
if [ "$GPU_TYPE" == "NVIDIA" ]; then
    # CUDA 12.1 (Adjust if needed)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [ "$GPU_TYPE" == "AMD" ]; then
    # ROCm 6.0 (Adjust if needed)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
else
    # CPU
    pip install torch torchvision torchaudio
fi

# 5. Install Common Libraries
echo "[*] Installing common libraries (transformers, accelerate, etc.)..."
pip install transformers accelerate datasets sentencepiece protobuf scipy

# 6. Install llama.cpp (llama-cpp-python)
echo "[*] Installing llama-cpp-python for $GPU_TYPE..."
if [ "$GPU_TYPE" == "NVIDIA" ]; then
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
elif [ "$GPU_TYPE" == "AMD" ]; then
    # Find ROCm path
    ROCM_PATH=""
    if [ -d "/opt/rocm" ]; then
        ROCM_PATH="/opt/rocm"
    elif ls -d /opt/rocm-* >/dev/null 2>&1; then
        ROCM_PATH=$(ls -d /opt/rocm-* | head -n 1)
    fi

    if [ -n "$ROCM_PATH" ]; then
        echo "    Found ROCm at $ROCM_PATH"
        export PATH=$ROCM_PATH/bin:$PATH
        export ROCM_PATH=$ROCM_PATH
    else
        echo "    [WARNING] Could not find ROCm installation in /opt/rocm or /opt/rocm-*. Build might fail."
    fi

    # HIPBLAS is for ROCm
    # We also need to ensure C++ compiler is set if needed, but usually hipcc handles it.
    CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
else
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
fi

# 7. Install AutoGPTQ
echo "[*] Installing AutoGPTQ for $GPU_TYPE..."
if [ "$GPU_TYPE" == "NVIDIA" ]; then
    pip install auto-gptq --no-build-isolation
elif [ "$GPU_TYPE" == "AMD" ]; then
    # AutoGPTQ on AMD can be tricky. Attempting standard install with ROCm torch present.
    echo "    Attempting AutoGPTQ install for ROCm (Experimental)..."
    if pip install auto-gptq --no-build-isolation; then
        echo "    [SUCCESS] AutoGPTQ installed."
    else
        echo "    [WARNING] AutoGPTQ installation failed. This is common on ROCm."
        echo "    [WARNING] Continuing with other tools. You can try building it from source later."
        # Do not exit script
    fi
else
    echo "    Skipping AutoGPTQ for CPU (usually requires GPU)."
fi

# 8. Install ExLlamaV2
echo "[*] Installing ExLlamaV2..."
if [ "$GPU_TYPE" == "NVIDIA" ]; then
    pip install exllamav2
elif [ "$GPU_TYPE" == "AMD" ]; then
    echo "    [WARNING] ExLlamaV2 primarily targets CUDA. Skipping installation for AMD."
    echo "    If you have a specific ROCm fork, please install it manually."
else
    echo "    Skipping ExLlamaV2 for CPU."
fi

echo "================================================================="
echo "   Installation Complete!"
echo "   To use the tools, run: source .venv/bin/activate"
echo "================================================================="
