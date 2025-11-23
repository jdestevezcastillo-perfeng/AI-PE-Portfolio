# AI Performance Engineering - Code Guidelines

To ensure consistency and educational value across this portfolio, all Python scripts must adhere to the following guidelines.

## 1. File Structure & ASCII Art

Every script must start with a high-level ASCII art diagram illustrating the data flow or system architecture.

```python
"""
SYSTEM DIAGRAM
--------------
   [Input] -> [Process] -> [Output]
      |           |
      v           v
   (Details)   (Details)
"""
```

## 2. Section Headers

Use distinct, numbered block headers to separate logical sections of the code.

```python
# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
```

Standard sections for Training scripts:

1. Configuration & Setup
2. Data Preparation (Loading/Tokenization)
3. Data Loader (Batching)
4. Model Initialization
5. Training Loop
6. Inference / Evaluation

## 3. AIPE Performance Notes

Highlight areas ripe for performance engineering with `>>> AIPE NOTE:`. Explain **what** is happening, **why** it matters for performance, and **how** to optimize it.

**Keywords to flag:**

* **Hardware:** `device = 'cuda'` (Check for CPU fallback)
* **Data Loading:** `.to(device)` (Blocking transfers, Pinning memory)
* **Memory:** `@torch.no_grad()` (Gradient overhead), `model.to(device)` (VRAM usage)
* **Compute:** `optimizer.step()` (Precision), `autocast` (Mixed Precision)
* **Quantization:** `int8` vs `fp16` trade-offs.

**Example:**

```python
# >>> AIPE NOTE: Mixed Precision (AMP)
# Standard training uses FP32. Use torch.amp.autocast for 2x speedup on Tensor Cores.
logits, loss = model(xb, yb)
```

## 4. Coding Standards

* **Imports:** Group standard libs, third-party libs, and local modules.
* **Type Hinting:** Use `typing` where helpful for function signatures.
* **Config:** Use `model_config.py` for hyperparameters; do not hardcode them in scripts.
* **Comments:** Avoid obvious inline comments (e.g., `# increment i`). Focus on "Why", not "What".

## 5. File Naming Conventions

To ensure the repository remains organized and navigable, use the following prefixes for all files:

* **`lab_XX_name.py`**: Hands-on exercises and scripts (e.g., `lab_01_inspect_transformer.py`).
* **`setup_XX_name.sh`**: Installation and environment setup scripts (e.g., `setup_00_install_tools.sh`).
* **`doc_name.md`**: Documentation, diagrams, and references (e.g., `doc_architecture_diagrams.md`).
* **`shared_name.py`**: Reusable modules and configuration files (e.g., `shared_gpt_model.py`).

**Format:** `[Category]_[Number]_[Name].[Extension]`

* *Note: Number is optional for `doc` and `shared` files.*
