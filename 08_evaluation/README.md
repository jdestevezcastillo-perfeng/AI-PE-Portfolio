# Module 08: Evaluation & Tracing

## 🎯 Objective
Performance isn't just speed; it's quality. This module focuses on validating that your high-performance inference engines (vLLM/TGI) are maintaining accuracy while delivering high throughput.

## 🛠️ Tools
- **GSM8K Dataset**: Grade School Math dataset for testing reasoning capabilities.
- **OpenAI API**: Standard interface for querying both vLLM and TGI.
- **Datasets Library**: For easy loading of Hugging Face benchmarks.

## 🚀 Quick Start

### 1. Throughput vs Accuracy Benchmark
We have created a script that measures the "Efficiency Score" (Accuracy * Throughput) of your engines.

**Run the automated benchmark:**
```bash
cd 08_evaluation
./run_accuracy_benchmark.sh
```

This script will:
1.  Start vLLM with Llama-3.1-8B.
2.  Run 20 samples from GSM8K.
3.  Report Accuracy (%) and Throughput (tokens/s).
4.  Repeat for TGI.

### 2. Manual Evaluation
You can run the python script directly against any running OpenAI-compatible server:

```bash
# Ensure you are in the virtual environment
../venv/bin/python3 evaluate_efficiency.py \
  --url http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --samples 50
```

## 📊 Metrics
- **Accuracy**: Percentage of correct answers (exact match of numerical result).
- **Throughput**: Average tokens generated per second.
- **Efficiency Score**: A composite metric to balance speed and quality.

## 📚 Concepts
1.  **RAG Evaluation:** Retrieving the right context + Generating the right answer.
2.  **LLM-as-a-Judge:** Using a strong model (GPT-4) to grade the output of a smaller model (Llama-3).
3.  **Tracing:** Visualizing the entire chain of a complex AI request (User -> Retriever -> Reranker -> LLM -> User).

## 🧪 Lab: The Automated Grader
**Goal:** Build a CI/CD pipeline for model quality.

### Steps:
1.  Create a "Golden Dataset" of 20 questions + correct answers.
2.  Run your fine-tuned model (Module 05) on these questions.
3.  Use **Ragas** (or a simple GPT-4 script) to grade the answers.
4.  **Fail the build** if the average score drops below 80%.
