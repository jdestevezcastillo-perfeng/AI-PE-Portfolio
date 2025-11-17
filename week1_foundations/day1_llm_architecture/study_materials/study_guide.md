# Day 1 Study Guide: LLM Architecture & KV-Cache

**Study Time**: 3 hours
**Prerequisites**: Basic understanding of neural networks

---

## Hour 1: Transformer Architecture (60 minutes)

### Primary Resource
**"The Illustrated Transformer" by Jay Alammar**
- URL: https://jalammar.github.io/illustrated-transformer/
- Time: 40 minutes of focused reading

### Study Checklist

#### Core Concepts to Master:

**1. Self-Attention Mechanism (15 min)**
- [ ] Understand Query (Q), Key (K), Value (V) matrices
- [ ] How attention scores are calculated: `Attention(Q,K,V) = softmax(QK^T / √d_k)V`
- [ ] Why we divide by √d_k (scaling factor)
- [ ] How attention creates context-aware representations

**Key Questions**:
- What problem does self-attention solve?
- How does a token "attend" to other tokens?
- Why is it called "self"-attention?

**2. Multi-Head Attention (10 min)**
- [ ] Why use multiple attention heads?
- [ ] How heads are computed in parallel
- [ ] How heads are concatenated and projected
- [ ] Different heads learn different relationships

**Key Questions**:
- What's the benefit of multiple attention heads vs single head?
- How many heads does GPT-3 use? (96 heads)

**3. Position Encodings (10 min)**
- [ ] Why transformers need position information
- [ ] Sinusoidal vs learned position encodings
- [ ] Absolute vs relative position encodings
- [ ] How position info is added to embeddings

**Key Questions**:
- Why can't transformers inherently understand position?
- How is position different from sequence in RNNs?

**4. Feed-Forward Networks (5 min)**
- [ ] Two-layer FFN in each transformer block
- [ ] Expansion ratio (typically 4x)
- [ ] GELU or ReLU activation
- [ ] Applied position-wise (independently to each position)

**5. Layer Normalization & Residual Connections (5 min)**
- [ ] Pre-norm vs post-norm
- [ ] Why residual connections help training
- [ ] Role in gradient flow

### Supplementary Reading (15 min)
**"Attention Is All You Need" - Original Paper**
- URL: https://arxiv.org/abs/1706.03762
- Focus on: Section 3.1 (Attention), Section 3.2 (Multi-Head), Section 3.3 (FFN)
- **Note**: Don't try to read the whole paper, just these sections

### Take Notes!
Create a diagram in your notebook:
```
Input Tokens
    ↓
Embedding + Positional Encoding
    ↓
┌─────────────────────┐
│  Transformer Block  │
│  ┌──────────────┐  │
│  │ Multi-Head   │  │
│  │ Attention    │  │
│  └──────────────┘  │
│         ↓          │
│  ┌──────────────┐  │
│  │ Add & Norm   │  │
│  └──────────────┘  │
│         ↓          │
│  ┌──────────────┐  │
│  │ Feed-Forward │  │
│  └──────────────┘  │
│         ↓          │
│  ┌──────────────┐  │
│  │ Add & Norm   │  │
│  └──────────────┘  │
└─────────────────────┘
    ↓
(Repeat N times)
    ↓
Output Logits
```

---

## Hour 2: KV-Cache Deep Dive (60 minutes)

### Primary Resources

**1. "The Illustrated GPT-2" by Jay Alammar (30 min)**
- URL: https://jalammar.github.io/illustrated-gpt2/
- Focus on the generation/inference section
- Pay special attention to the autoregressive generation process

**2. Hugging Face KV-Cache Documentation (15 min)**
- URL: https://huggingface.co/docs/transformers/main/en/kv_cache
- Technical details on implementation

**3. Blog: "Making Deep Learning Go Brrrr" (15 min)**
- URL: https://horace.io/brrr_intro.html
- Section on memory hierarchy and caching

### Study Checklist

#### Understanding KV-Cache:

**1. The Problem: Why KV-Cache Exists (10 min)**

Without KV-cache, during autoregressive generation:
```
Prompt: "The cat sat on the"

Generation step 1: "The cat sat on the" → "mat"
  - Computes attention for ALL 5 tokens

Generation step 2: "The cat sat on the mat" → "and"
  - Re-computes attention for ALL 6 tokens (including previous 5!)

Generation step 3: "The cat sat on the mat and" → "purred"
  - Re-computes attention for ALL 7 tokens (wasteful!)
```

**Key Insight**:
- Each new token requires re-computing K and V for ALL previous tokens
- This is O(n²) computation and memory
- 99% of computation is redundant!

**2. The Solution: Cache K and V (15 min)**

With KV-cache:
```
Step 1: Compute K,V for "The cat sat on the", cache them
        Generate "mat"

Step 2: Only compute K,V for "mat"
        Retrieve cached K,V for "The cat sat on the"
        Concatenate and compute attention
        Generate "and"

Step 3: Only compute K,V for "and"
        Retrieve all cached K,V
        Generate "purred"
```

**Benefits**:
- [ ] Avoids redundant computation
- [ ] Speeds up inference dramatically
- [ ] Trade-off: Uses more memory

**3. Prefill vs Decode Phases (15 min)**

**Prefill (Processing the Prompt)**:
- All prompt tokens processed in parallel
- Compute K,V for entire prompt at once
- Store in KV-cache
- Fast but memory-bound
- **Metric**: TTFT (Time to First Token)

**Decode (Generating Tokens)**:
- One token at a time (autoregressive)
- Compute K,V only for new token
- Retrieve previous K,V from cache
- Slower per token but efficient overall
- **Metric**: TPOT (Time Per Output Token)

```
Timeline:
|←─────── Prefill ──────→|←─── Decode ─────→|
[Process entire prompt]  [Gen][Gen][Gen][Gen]
                         tok1  tok2  tok3  tok4

Prefill: Parallel, fast, high throughput
Decode:  Sequential, memory-bound
```

**4. Memory Implications (10 min)**

**KV-Cache Size Calculation**:
```
For each layer:
  K cache: [batch_size, num_heads, seq_len, head_dim]
  V cache: [batch_size, num_heads, seq_len, head_dim]

Total KV cache = 2 × num_layers × batch × heads × seq_len × head_dim × bytes_per_param

Example (LLaMA-7B):
  - 32 layers
  - 32 heads
  - head_dim = 128
  - seq_len = 2048
  - fp16 (2 bytes)

  Size = 2 × 32 × 1 × 32 × 2048 × 128 × 2 bytes
       = 1.07 GB per sequence!
```

**Key Observations**:
- [ ] KV-cache grows linearly with sequence length
- [ ] Limits maximum batch size
- [ ] Memory is the bottleneck, not compute
- [ ] Techniques like PagedAttention (vLLM) optimize this

**5. Performance Impact (10 min)**

**Without KV-Cache**:
- Latency: O(n²) where n = sequence length
- Throughput: Very poor
- Practical? No for long sequences

**With KV-Cache**:
- Latency: O(n) - linear with sequence length
- Throughput: 5-10x faster
- Memory: Must fit in VRAM/RAM
- Practical? Yes, industry standard

**Trade-offs**:
- Speed ↑ but Memory ↑
- Can run out of memory with long sequences
- Batch size must decrease as sequence length increases

---

## Hour 3: LLM Inference Optimization (60 minutes)

### Resources

**1. vLLM Paper (25 min)**
- URL: https://arxiv.org/abs/2309.06180
- Title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- Read: Abstract, Introduction, Section 3 (PagedAttention)

**2. Lilian Weng's Blog (25 min)**
- URL: https://lilianweng.github.io/posts/2023-01-10-inference-optimization/
- Sections to focus on:
  - Quantization
  - Model compression
  - Batching strategies

**3. Additional Resource: FlashAttention (10 min - skim)**
- URL: https://arxiv.org/abs/2205.14135
- Understand the high-level idea: reordering operations to be memory-efficient

### Study Checklist

#### Key Concepts:

**1. Token Generation Process**
- [ ] Understand autoregressive generation
- [ ] Sampling strategies (greedy, top-k, top-p, temperature)
- [ ] Stop conditions and max length

**2. Performance Metrics**
- [ ] **TTFT** (Time to First Token): User-perceived latency
- [ ] **TPOT** (Time Per Output Token): Generation speed
- [ ] **Throughput**: Tokens/second or requests/second
- [ ] **Latency**: End-to-end response time
- [ ] **P50, P95, P99**: Percentile latencies for SLAs

**3. Batching Strategies**
- [ ] **Static batching**: Fixed batch size, wait for all to complete
- [ ] **Dynamic batching**: Continuous batching, variable completion times
- [ ] **In-flight batching**: Add new requests to in-progress batch (vLLM)

**4. Memory Bandwidth Bottlenecks**
- [ ] Why inference is memory-bound, not compute-bound
- [ ] Arithmetic intensity and the roofline model
- [ ] How memory bandwidth limits throughput

**5. Optimization Techniques (Overview)**
- [ ] **Quantization**: FP16, INT8, INT4 (tomorrow's focus)
- [ ] **Flash Attention**: Faster attention computation
- [ ] **PagedAttention**: Efficient KV-cache memory management
- [ ] **Speculative Decoding**: Use small model to draft, large to verify
- [ ] **Continuous Batching**: Better GPU utilization

---

## Self-Assessment Quiz

Test your understanding:

### Transformer Architecture
1. What are the three components of attention (Q, K, V)?
2. Why do we use multiple attention heads?
3. Explain the role of position encodings.
4. What is the purpose of residual connections?

### KV-Cache
5. Why is KV-cache necessary for efficient inference?
6. What is the difference between prefill and decode phases?
7. How does sequence length affect KV-cache memory usage?
8. What's the trade-off of using KV-cache?

### Performance
9. What is TTFT and why does it matter?
10. Why is LLM inference memory-bound rather than compute-bound?
11. What's the difference between latency and throughput?
12. How does batching improve throughput?

### Answers
Check your answers in: `study_materials/quiz_answers.md`

---

## Practical Exercise

After reading, try this:

**Exercise: Explain to a colleague**

Imagine explaining to a fellow performance engineer:

1. "Why does generating 100 tokens take longer than processing a 100-token prompt?"

2. "We have 24GB GPU memory. Why can we only serve 4 concurrent users with 2048 token contexts?"

3. "Why does the first token take longer than subsequent tokens?"

Write your explanations in a document. This will solidify your understanding.

---

## Summary: Key Takeaways

**Transformer Architecture**:
- Self-attention allows parallel processing during training
- Multi-head attention captures different types of relationships
- Position encodings give sequence information
- Feed-forward networks provide non-linearity

**KV-Cache**:
- Essential for efficient autoregressive generation
- Trades memory for speed
- Grows linearly with sequence length
- Separates inference into prefill (parallel) and decode (sequential) phases

**Performance**:
- TTFT measures prefill latency (user experience)
- TPOT measures decode speed
- Memory bandwidth is the main bottleneck
- Batching improves throughput but increases latency

**Next Steps**:
- Tomorrow you'll explore quantization and model optimization
- These techniques reduce memory usage and increase throughput
- You'll benchmark quantized models and compare results

---

## Additional Resources (Optional)

**Videos**:
- Andrej Karpathy: "Let's build GPT" - https://www.youtube.com/watch?v=kCc8FmEb1nY
- StatQuest: Attention Mechanism - https://www.youtube.com/watch?v=PSs6nxngL6k

**Interactive**:
- Transformer Explainer - https://poloclub.github.io/transformer-explainer/
- Attention Visualizer - https://bbycroft.net/llm

**Blogs**:
- Chip Huyen: "Building LLM Applications" - https://huyenchip.com/2023/04/11/llm-engineering.html
- Eugene Yan: "Patterns for LLMs" - https://eugeneyan.com/writing/llm-patterns/

**Papers** (advanced):
- GPT-3: "Language Models are Few-Shot Learners"
- LLaMA: "Open and Efficient Foundation Language Models"
- vLLM: "Efficient Memory Management for Large Language Model Serving"

---

## Notes Section

Use this space for your own notes during study:

### Key Insights:
-

### Questions to Research:
-

### Connection to Performance Engineering:
-

### Ideas to Try Tomorrow:
-
