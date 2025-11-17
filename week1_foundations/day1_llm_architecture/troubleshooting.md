# Troubleshooting Guide - Day 1

Common issues and their solutions for Day 1 activities.

---

## Installation Issues

### Issue: "ollama: command not found" after installation

**Symptoms**:
```bash
$ ollama --version
bash: ollama: command not found
```

**Solutions**:

1. **Restart terminal**: Often PATH isn't updated until new shell
   ```bash
   # Close terminal and open a new one
   ```

2. **Manually add to PATH**:
   ```bash
   export PATH=$PATH:/usr/local/bin
   # Or add to ~/.bashrc or ~/.zshrc for persistence
   echo 'export PATH=$PATH:/usr/local/bin' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Check installation location**:
   ```bash
   which ollama
   # If found, note the path and add to PATH
   ```

4. **Re-install**:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

---

### Issue: Model download fails or is incomplete

**Symptoms**:
```bash
$ ollama pull llama3.1:8b
Error: connection timeout
```

**Solutions**:

1. **Check internet connection**:
   ```bash
   ping ollama.ai
   ```

2. **Try again** - Downloads resume from where they left off:
   ```bash
   ollama pull llama3.1:8b
   ```

3. **Check disk space**:
   ```bash
   df -h
   # Ensure at least 10GB free
   ```

4. **Use different network** - Sometimes corporate networks block downloads

5. **Manual retry with smaller model first**:
   ```bash
   ollama pull phi3:mini  # Only 2.3GB
   ```

---

### Issue: Python virtual environment creation fails

**Symptoms**:
```bash
$ python3 -m venv venv
Error: No module named venv
```

**Solutions**:

1. **Install python3-venv** (Ubuntu/Debian):
   ```bash
   sudo apt update
   sudo apt install python3-venv python3-pip
   ```

2. **Use conda instead**:
   ```bash
   conda create -n ai-perf python=3.10
   conda activate ai-perf
   ```

3. **Use system Python without venv** (not recommended):
   ```bash
   pip install --user ollama requests numpy pandas matplotlib seaborn psutil
   ```

---

## Runtime Issues

### Issue: "Connection refused" when running benchmarks

**Symptoms**:
```bash
$ python basic_benchmark.py
Error: Cannot connect to Ollama API
ConnectionError: Connection refused on localhost:11434
```

**Solutions**:

1. **Start Ollama service**:
   ```bash
   # In a separate terminal
   ollama serve
   ```

2. **Check if service is running**:
   ```bash
   ps aux | grep ollama
   # Should see ollama process
   ```

3. **Test with curl**:
   ```bash
   curl http://localhost:11434/api/tags
   # Should return JSON list of models
   ```

4. **Check port availability**:
   ```bash
   lsof -i :11434
   # If another process is using port 11434, stop it
   ```

---

### Issue: "Out of memory" during inference

**Symptoms**:
```bash
Error: Out of memory
Killed
```
Or system becomes unresponsive

**Solutions**:

1. **Check available RAM**:
   ```bash
   free -h
   # Need at least 8GB free for llama3.1:8b
   ```

2. **Close other applications**:
   ```bash
   # Close browsers, IDEs, etc.
   ```

3. **Use smaller model**:
   ```bash
   ollama pull phi3:mini  # Requires only 4GB
   # Update MODEL_NAME in scripts to "phi3:mini"
   ```

4. **Reduce batch size** in advanced_benchmark.py:
   ```python
   # Change from:
   requests_per_batch=10
   # To:
   requests_per_batch=5
   ```

5. **Monitor memory usage**:
   ```bash
   # In separate terminal
   watch -n 1 free -h
   # Or use htop
   htop
   ```

---

### Issue: Benchmark runs extremely slowly

**Symptoms**:
- Each request takes >30 seconds
- TPS is <1 token/second
- System feels sluggish

**Solutions**:

1. **Check CPU usage**:
   ```bash
   htop
   # Look for other processes using CPU
   ```

2. **Reduce concurrent load**:
   ```python
   # In advanced_benchmark.py
   # Reduce num_workers from 4 to 1
   ```

3. **Run fewer requests for testing**:
   ```python
   NUM_REQUESTS = 20  # Instead of 100
   ```

4. **Check for thermal throttling**:
   ```bash
   # Linux
   sensors  # Install: sudo apt install lm-sensors

   # macOS
   sudo powermetrics --samplers smc | grep temp
   ```

5. **Try smaller model**:
   ```bash
   ollama pull tinyllama  # Very small, faster
   ```

---

### Issue: Python package import errors

**Symptoms**:
```python
ModuleNotFoundError: No module named 'matplotlib'
```

**Solutions**:

1. **Ensure virtual environment is activated**:
   ```bash
   source venv/bin/activate
   # Prompt should show (venv)
   ```

2. **Install missing package**:
   ```bash
   pip install matplotlib
   # Or install all at once:
   pip install ollama requests numpy pandas matplotlib seaborn psutil
   ```

3. **Check pip is using correct environment**:
   ```bash
   which pip
   # Should show path in venv/
   ```

4. **Upgrade pip first**:
   ```bash
   pip install --upgrade pip
   pip install matplotlib seaborn
   ```

---

### Issue: Visualization script fails

**Symptoms**:
```bash
$ python visualize_results.py
Error: No such file or directory: 'benchmark_results.json'
```

**Solutions**:

1. **Run benchmark first**:
   ```bash
   python basic_benchmark.py
   # Wait for completion
   python visualize_results.py
   ```

2. **Specify file explicitly**:
   ```bash
   python visualize_results.py benchmark_results_20240115_143022.json
   ```

3. **Check for JSON files**:
   ```bash
   ls -la *.json
   ```

---

### Issue: Plots are not displayed

**Symptoms**:
- Script completes but no plots visible
- "Saved to plots/" but directory empty

**Solutions**:

1. **Check if plots/ directory was created**:
   ```bash
   ls -la plots/
   ```

2. **Check matplotlib backend**:
   ```python
   # Add to top of visualize_results.py
   import matplotlib
   matplotlib.use('Agg')  # Non-interactive backend
   import matplotlib.pyplot as plt
   ```

3. **Permission issues**:
   ```bash
   chmod +w .  # Ensure write permissions
   mkdir -p plots
   ```

---

## Benchmark Issues

### Issue: Inconsistent latency measurements

**Symptoms**:
- Large variance in latency (1s to 20s)
- P99 much higher than P95

**Causes & Solutions**:

1. **First request is slower** (model loading):
   ```python
   # In scripts, add warmup requests:
   print("Warming up...")
   for _ in range(3):
       generate_text("Hello")
   ```

2. **System background tasks**:
   - Close unnecessary applications
   - Disable system updates during benchmarking

3. **CPU frequency scaling**:
   ```bash
   # Linux - set performance mode
   sudo cpupower frequency-set -g performance
   ```

4. **Run more iterations**:
   ```python
   NUM_REQUESTS = 200  # More data = more stable stats
   ```

---

### Issue: TTFT measurements seem incorrect

**Symptoms**:
- TTFT is 0 or negative
- TTFT is larger than total latency

**Solutions**:

1. **Check Ollama version** - older versions may not report these metrics:
   ```bash
   ollama --version
   # Update if needed:
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Verify response structure**:
   ```python
   # Add debug print in script:
   print(json.dumps(result, indent=2))
   # Check for 'prompt_eval_duration' field
   ```

3. **Some models don't report metrics** - this is OK:
   - TTFT will show as 0
   - Total latency is still accurate

---

## Study Resource Issues

### Issue: Links don't load or are broken

**Solutions**:

1. **Check internet connection**:
   ```bash
   ping google.com
   ```

2. **Try alternative links**:
   - Jay Alammar's blog: Use archive.org if down
   - Papers: Try alternative sources (arxiv.org, papers.withcode.com)

3. **Offline alternatives**:
   - Download papers as PDFs for offline reading
   - Use locally saved resources in study_materials/

---

### Issue: Can't understand the transformer architecture

**Solutions**:

1. **Start with video explanations**:
   - Watch: "Attention is All You Need" explained on YouTube
   - StatQuest videos for basics

2. **Use interactive tools**:
   - Transformer Explainer: https://poloclub.github.io/transformer-explainer/
   - Play with the visualizations

3. **Break it down**:
   - Day 1: Just understand self-attention
   - Day 2: Learn multi-head attention
   - Day 3: Understand full architecture
   - Don't try to master everything in one day!

4. **Ask for help**:
   - Reddit: r/MachineLearning
   - Discord: Hugging Face, EleutherAI
   - Stack Overflow

---

## Performance Issues

### Issue: Results don't match expected values

**Expected values** (CPU inference with llama3.1:8b):
- TPS: 5-20 tokens/second
- TTFT: 0.1-2 seconds
- Latency: 2-15 seconds for typical prompts

**If your results are significantly different**:

1. **Much slower** (TPS < 5):
   - Check CPU is not throttling
   - Close background apps
   - Try smaller model first
   - Check system load: `top` or `htop`

2. **Much faster** (TPS > 50):
   - You may have GPU acceleration (great!)
   - Or model is quantized heavily
   - Check: `nvidia-smi` for GPU usage

3. **Different patterns**:
   - Every system is different - this is fine
   - Focus on relative comparisons (short vs long prompts)
   - Document your specific hardware

---

## Git/GitHub Issues

### Issue: Can't push to GitHub

**Symptoms**:
```bash
$ git push
Permission denied (publickey)
```

**Solutions**:

1. **Setup SSH key**:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # Add to GitHub: Settings → SSH and GPG keys
   ```

2. **Use HTTPS instead**:
   ```bash
   git remote set-url origin https://github.com/username/repo.git
   ```

3. **Configure git**:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your_email@example.com"
   ```

---

## Still Stuck?

If none of these solutions work:

1. **Check Day 1 README.md** for additional context

2. **Search for error message**:
   - Google the exact error
   - Check Ollama GitHub issues: https://github.com/ollama/ollama/issues
   - Check Stack Overflow

3. **Simplify and isolate**:
   ```bash
   # Test each component separately
   ollama run llama3.1:8b  # Interactive mode
   python -c "import requests; print(requests.get('http://localhost:11434').status_code)"
   ```

4. **Ask for help**:
   - Ollama Discord: https://discord.gg/ollama
   - Reddit: r/LocalLLaMA
   - GitHub Discussions on your course repo

5. **Document the issue**:
   - Create detailed bug report
   - Include error messages, system info, steps to reproduce
   - This helps others and future you!

---

## Prevention Tips

To avoid issues:

1. **Read setup instructions fully before starting**
2. **Test each step before proceeding**
3. **Keep notes of what you change**
4. **Commit working code frequently**
5. **Don't run benchmarks with low disk space (<10GB free)**
6. **Monitor system resources during runs**
7. **Start with shorter benchmark runs (20 requests) before full runs**

---

**Remember**: Issues are part of the learning process! Every error you solve teaches you more about the system.
