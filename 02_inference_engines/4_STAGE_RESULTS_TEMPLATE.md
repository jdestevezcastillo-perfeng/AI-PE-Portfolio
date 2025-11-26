# 4-Stage Benchmark Results Template

## Test Configuration

- **Date**: [YYYY-MM-DD]
- **Hardware**: NVIDIA RTX 3090 (24GB VRAM)
- **vLLM Version**: [version]
- **TGI Version**: [version]

## Results Summary

### Stage 1: Llama-3.1-8B on vLLM

#### Low Load (50 requests, concurrency 1)

| Metric | Value | Unit |
|--------|-------|------|
| Avg TTFT | | ms |
| P99 TTFT | | ms |
| Avg ITL | | ms |
| P99 ITL | | ms |
| Throughput (per-request) | | tok/s |
| Throughput (overall) | | tok/s |
| Request Rate | | req/s |
| Total Tokens | | tokens |

#### Medium Load (100 requests, concurrency 3)

| Metric | Value | Unit |
|--------|-------|------|
| Avg TTFT | | ms |
| P99 TTFT | | ms |
| Avg ITL | | ms |
| P99 ITL | | ms |
| Throughput (per-request) | | tok/s |
| Throughput (overall) | | tok/s |
| Request Rate | | req/s |
| Total Tokens | | tokens |

#### High Load (200 requests, concurrency 5)

| Metric | Value | Unit |
|--------|-------|------|
| Avg TTFT | | ms |
| P99 TTFT | | ms |
| Avg ITL | | ms |
| P99 ITL | | ms |
| Throughput (per-request) | | tok/s |
| Throughput (overall) | | tok/s |
| Request Rate | | req/s |
| Total Tokens | | tokens |

---

### Stage 2: Llama-3.1-8B on TGI

#### Low Load (50 requests, concurrency 1)

| Metric | Value | Unit |
|--------|-------|------|
| Avg TTFT | | ms |
| P99 TTFT | | ms |
| Avg ITL | | ms |
| P99 ITL | | ms |
| Throughput (per-request) | | tok/s |
| Throughput (overall) | | tok/s |
| Request Rate | | req/s |
| Total Tokens | | tokens |

#### Medium Load (100 requests, concurrency 3)

| Metric | Value | Unit |
|--------|-------|------|
| Avg TTFT | | ms |
| P99 TTFT | | ms |
| Avg ITL | | ms |
| P99 ITL | | ms |
| Throughput (per-request) | | tok/s |
| Throughput (overall) | | tok/s |
| Request Rate | | req/s |
| Total Tokens | | tokens |

#### High Load (200 requests, concurrency 5)

| Metric | Value | Unit |
|--------|-------|------|
| Avg TTFT | | ms |
| P99 TTFT | | ms |
| Avg ITL | | ms |
| P99 ITL | | ms |
| Throughput (per-request) | | tok/s |
| Throughput (overall) | | tok/s |
| Request Rate | | req/s |
| Total Tokens | | tokens |

---

### Stage 3: Mistral-7B on vLLM

#### Low Load (50 requests, concurrency 1)

| Metric | Value | Unit |
|--------|-------|------|
| Avg TTFT | | ms |
| P99 TTFT | | ms |
| Avg ITL | | ms |
| P99 ITL | | ms |
| Throughput (per-request) | | tok/s |
| Throughput (overall) | | tok/s |
| Request Rate | | req/s |
| Total Tokens | | tokens |

#### Medium Load (100 requests, concurrency 3)

| Metric | Value | Unit |
|--------|-------|------|
| Avg TTFT | | ms |
| P99 TTFT | | ms |
| Avg ITL | | ms |
| P99 ITL | | ms |
| Throughput (per-request) | | tok/s |
| Throughput (overall) | | tok/s |
| Request Rate | | req/s |
| Total Tokens | | tokens |

#### High Load (200 requests, concurrency 5)

| Metric | Value | Unit |
|--------|-------|------|
| Avg TTFT | | ms |
| P99 TTFT | | ms |
| Avg ITL | | ms |
| P99 ITL | | ms |
| Throughput (per-request) | | tok/s |
| Throughput (overall) | | tok/s |
| Request Rate | | req/s |
| Total Tokens | | tokens |

---

### Stage 4: Mistral-7B on TGI

#### Low Load (50 requests, concurrency 1)

| Metric | Value | Unit |
|--------|-------|------|
| Avg TTFT | | ms |
| P99 TTFT | | ms |
| Avg ITL | | ms |
| P99 ITL | | ms |
| Throughput (per-request) | | tok/s |
| Throughput (overall) | | tok/s |
| Request Rate | | req/s |
| Total Tokens | | tokens |

#### Medium Load (100 requests, concurrency 3)

| Metric | Value | Unit |
|--------|-------|------|
| Avg TTFT | | ms |
| P99 TTFT | | ms |
| Avg ITL | | ms |
| P99 ITL | | ms |
| Throughput (per-request) | | tok/s |
| Throughput (overall) | | tok/s |
| Request Rate | | req/s |
| Total Tokens | | tokens |

#### High Load (200 requests, concurrency 5)

| Metric | Value | Unit |
|--------|-------|------|
| Avg TTFT | | ms |
| P99 TTFT | | ms |
| Avg ITL | | ms |
| P99 ITL | | ms |
| Throughput (per-request) | | tok/s |
| Throughput (overall) | | tok/s |
| Request Rate | | req/s |
| Total Tokens | | tokens |

---

## Comparative Analysis

### Engine Comparison (Same Model)

#### Llama-3.1-8B: vLLM vs TGI

| Metric | vLLM (Stage 1) | TGI (Stage 2) | Winner |
|--------|----------------|---------------|--------|
| Avg TTFT (Low) | | | |
| Avg TTFT (Med) | | | |
| Avg TTFT (High) | | | |
| Avg ITL (Low) | | | |
| Avg ITL (Med) | | | |
| Avg ITL (High) | | | |
| Throughput (Low) | | | |
| Throughput (Med) | | | |
| Throughput (High) | | | |

**Analysis**:

- [Your observations]

#### Mistral-7B: vLLM vs TGI

| Metric | vLLM (Stage 3) | TGI (Stage 4) | Winner |
|--------|----------------|---------------|--------|
| Avg TTFT (Low) | | | |
| Avg TTFT (Med) | | | |
| Avg TTFT (High) | | | |
| Avg ITL (Low) | | | |
| Avg ITL (Med) | | | |
| Avg ITL (High) | | | |
| Throughput (Low) | | | |
| Throughput (Med) | | | |
| Throughput (High) | | | |

**Analysis**:

- [Your observations]

---

### Model Comparison (Same Engine)

#### vLLM: Llama-3.1-8B vs Mistral-7B

| Metric | Llama (Stage 1) | Mistral (Stage 3) | Winner |
|--------|-----------------|-------------------|--------|
| Avg TTFT (Low) | | | |
| Avg TTFT (Med) | | | |
| Avg TTFT (High) | | | |
| Avg ITL (Low) | | | |
| Avg ITL (Med) | | | |
| Avg ITL (High) | | | |
| Throughput (Low) | | | |
| Throughput (Med) | | | |
| Throughput (High) | | | |

**Analysis**:

- [Your observations]

#### TGI: Llama-3.1-8B vs Mistral-7B

| Metric | Llama (Stage 2) | Mistral (Stage 4) | Winner |
|--------|-----------------|-------------------|--------|
| Avg TTFT (Low) | | | |
| Avg TTFT (Med) | | | |
| Avg TTFT (High) | | | |
| Avg ITL (Low) | | | |
| Avg ITL (Med) | | | |
| Avg ITL (High) | | | |
| Throughput (Low) | | | |
| Throughput (Med) | | | |
| Throughput (High) | | | |

**Analysis**:

- [Your observations]

---

### Overall Winner

| Configuration | Avg TTFT | Avg ITL | Throughput | Overall Score |
|---------------|----------|---------|------------|---------------|
| vLLM + Llama | | | | |
| TGI + Llama | | | | |
| vLLM + Mistral | | | | |
| TGI + Mistral | | | | |

**Best Configuration**:

- **For Latency**: [configuration]
- **For Throughput**: [configuration]
- **For Consistency**: [configuration]
- **Overall**: [configuration]

---

## Key Insights

### Engine Strengths

**vLLM**:

- [Your findings]

**TGI**:

- [Your findings]

### Model Characteristics

**Llama-3.1-8B**:

- [Your findings]

**Mistral-7B**:

- [Your findings]

### Recommendations

**Use vLLM when**:

- [Your recommendations]

**Use TGI when**:

- [Your recommendations]

**Use Llama-3.1-8B when**:

- [Your recommendations]

**Use Mistral-7B when**:

- [Your recommendations]

---

## Grafana Screenshots

[Add screenshots from Grafana dashboards showing key comparisons]

---

## Conclusion

[Your overall conclusion about the 4-stage benchmark results]
