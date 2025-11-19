# Module 03: Load Testing & Concurrency

## 🎯 Objective
Learn how to stress-test an LLM endpoint to find its breaking point. "It works on my machine" is not enough for production.

## 📚 Concepts
1.  **TTFT (Time To First Token):** The user's perceived latency.
2.  **TPOT (Time Per Output Token):** The reading speed.
3.  **Queue Time:** How long a request waits before processing begins.
4.  **Concurrency vs RPS:** Why 10 users != 10 Requests Per Second.

## 🛠️ Tools to Master
- **Locust:** Python-based load testing tool (flexible).
- **GenAI-Perf (Triton):** NVIDIA's specialized tool for AI benchmarking.
- **k6:** Go-based high-performance load tester.

## 🧪 Lab: Breaking the Server
**Goal:** Find the "Knee of the Curve" where latency spikes.

### Steps:
1.  Write a `locustfile.py` that sends chat completion requests.
2.  Target your vLLM server.
3.  Ramp up users: 1, 5, 10, 20, 50.
4.  **Analyze:** Plot RPS vs Latency. Find the point where P99 latency exceeds 2 seconds.
5.  **Tune:** Adjust vLLM's `--max-num-seqs` and see if it changes.

## 📝 Deliverable
A Load Test Report showing the saturation point of your RX 6700 XT.
