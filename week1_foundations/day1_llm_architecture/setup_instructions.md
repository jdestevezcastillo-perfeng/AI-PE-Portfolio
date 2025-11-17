# Setup Instructions - Day 1

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or WSL2 on Windows
- **RAM**: 8GB minimum (16GB recommended for 8B models)
- **Storage**: 10GB free space
- **Python**: 3.8 or higher

---

## Step-by-Step Setup

### 1. Install Ollama

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### macOS
```bash
brew install ollama
```

#### Windows (WSL2)
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Verify installation:
```bash
ollama --version
```

### 2. Start Ollama Service

```bash
# Linux/WSL
ollama serve

# Or run in background
nohup ollama serve &
```

### 3. Pull LLM Model

In a new terminal:
```bash
# Pull llama3.1 8B model (~4.7GB)
ollama pull llama3.1:8b

# Verify model is downloaded
ollama list
```

### 4. Test Basic Inference

```bash
# Interactive mode
ollama run llama3.1:8b

# Test with a prompt
# >>> What is machine learning?
# (Wait for response, then Ctrl+D to exit)
```

### 5. Set Up Python Environment

```bash
# Create project directory
mkdir -p ~/ai-perf-labs/week1-llm-basics
cd ~/ai-perf-labs/week1-llm-basics

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install ollama openai requests numpy pandas matplotlib seaborn
pip install psutil  # For system monitoring
```

### 6. Verify Ollama API

Test the API is accessible:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

You should get a JSON response with the generated text.

### 7. Clone Benchmark Scripts

The benchmark scripts are provided in this repository. Navigate to:
```bash
cd ~/AI-PE-Portfolio/week1_foundations/day1_llm_architecture/scripts/
```

---

## Verification Checklist

- [ ] Ollama installed and running
- [ ] llama3.1:8b model downloaded
- [ ] Can run interactive chat
- [ ] Python environment created
- [ ] All pip packages installed
- [ ] Ollama API responds to curl requests
- [ ] Ready to run benchmarks

---

## Troubleshooting

### Issue: "ollama: command not found"
**Solution**: Restart terminal or add to PATH:
```bash
export PATH=$PATH:/usr/local/bin
```

### Issue: "Connection refused" on API calls
**Solution**: Ensure ollama service is running:
```bash
ps aux | grep ollama
# If not running:
ollama serve
```

### Issue: Model download is slow
**Solution**:
- Check internet connection
- Model is ~4.7GB, may take 10-30 minutes depending on speed
- Download continues from breakpoint if interrupted

### Issue: Out of memory during inference
**Solution**:
- Close other applications
- Try a smaller model: `ollama pull llama3.1:7b` or `phi3:mini`
- Monitor with: `htop` or `top`

### Issue: Python package conflicts
**Solution**:
```bash
# Create fresh environment
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# Install packages one by one
```

---

## Next: Run Your First Benchmark

Once setup is complete, proceed to:
```bash
python scripts/basic_benchmark.py
```

See `scripts/README.md` for detailed usage instructions.
