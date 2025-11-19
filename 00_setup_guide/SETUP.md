# Module 00: Environment Setup Guide

## 1. Operating System Optimization (Linux Mint)
As a performance engineer, you know the OS is the foundation.
- **Kernel:** Ensure you are on a modern kernel (6.5+) for better hardware support.
- **Swap:** Adjust swappiness (`vm.swappiness=10`) to prefer RAM.
- **File Descriptors:** Increase limits in `/etc/security/limits.conf` for high-concurrency load testing.

## 2. GPU Drivers (AMD ROCm)
Since you are using a Radeon RX 6700 XT:
- **Install ROCm:** Follow the official AMD docs for your specific Ubuntu version (Mint base).
- **Verify:** Run `rocm-smi` to check visibility.
- **Groups:** Ensure your user is in the `render` and `video` groups (`sudo usermod -aG render,video $USER`).

## 3. Python Environment Strategy
We will use `uv` (by Astral) because it is significantly faster than pip/conda and written in Rust.
- **Install:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Usage:**
  ```bash
  uv venv .venv
  source .venv/bin/activate
  uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
  ```

## 4. Containerization
- **Docker:** Install Docker Engine.
- **Permissions:** `sudo usermod -aG docker $USER`.
- **ROCm Container Support:** You may need to run containers with specific device mapping:
  ```bash
  docker run --device=/dev/kfd --device=/dev/dri --group-add video ...
  ```
