# Module 07: AI Observability

## 🎯 Objective
You can't optimize what you can't measure. Learn how to instrument LLM applications to track both system performance and model behavior.

## 📚 Concepts
1.  **The 3 Pillars:** Logs, Metrics, Traces.
2.  **LLM Specific Metrics:**
    - **Token Usage:** Input vs Output tokens (cost driver).
    - **Latency Breakdown:** TTFT vs TPOT.
    - **Hallucination Rate:** (Hard to measure, but critical).
3.  **OpenTelemetry:** The industry standard for tracing requests across microservices.

## 🛠️ Tools to Master
- **Prometheus + Grafana:** The classic duo (you know this!).
- **OpenTelemetry (OTel):** For standardizing traces.
- **Arize Phoenix / LangSmith:** Specialized AI observability platforms.

## 🧪 Lab: The Ultimate Dashboard
**Goal:** Visualize the "Pulse" of your AI system.

### Steps:
1.  Instrument your vLLM/Ollama server to export metrics to Prometheus.
2.  Create a Grafana Dashboard showing:
    - Real-time Tokens/sec.
    - Queue depth.
    - GPU Temperature & Utilization.
    - Error rates.
3.  **Stress Test:** Run your load generator (Module 03) and watch the dashboard light up.

## 📝 Deliverable
A JSON export of your Grafana Dashboard.
