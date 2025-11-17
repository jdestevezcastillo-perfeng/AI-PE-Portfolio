# Study Quiz Answers - Day 1

## Transformer Architecture

### 1. What are the three components of attention (Q, K, V)?

**Answer**:
- **Query (Q)**: Represents "what I'm looking for" - the current token asking for information
- **Key (K)**: Represents "what I contain" - the searchable properties of each token
- **Value (V)**: Represents "what I'll give" - the actual information to be retrieved

The attention mechanism computes similarity between Q and K to determine weights, then uses those weights to combine V values.

### 2. Why do we use multiple attention heads?

**Answer**:
Multiple attention heads allow the model to attend to different types of relationships simultaneously:
- One head might focus on syntactic relationships (subject-verb)
- Another might capture semantic relationships (synonyms)
- Another might track long-range dependencies
- Each head learns different patterns in parallel

This is more powerful than a single large attention head because different heads can specialize in different aspects of language.

### 3. Explain the role of position encodings.

**Answer**:
Unlike RNNs that process sequentially, transformers process all tokens in parallel. This means they have no inherent notion of token order.

Position encodings solve this by:
- Adding position information to each token's embedding
- Using sinusoidal functions (or learned embeddings) to create unique position vectors
- Allowing the model to distinguish "cat chased dog" from "dog chased cat"

Without position encodings, transformers would be permutation-invariant (order wouldn't matter).

### 4. What is the purpose of residual connections?

**Answer**:
Residual connections (skip connections) serve several purposes:
- **Gradient flow**: Enable gradients to flow directly through the network during backpropagation
- **Training stability**: Prevent vanishing gradients in deep networks
- **Identity mapping**: Allow the model to learn the "identity" function (no change) easily
- **Feature preservation**: Ensure information from earlier layers isn't lost

Formula: `output = LayerNorm(input + SubLayer(input))`

---

## KV-Cache

### 5. Why is KV-cache necessary for efficient inference?

**Answer**:
During autoregressive generation, each new token requires computing attention over all previous tokens. Without KV-cache:

- Token 1: Compute attention for 1 token
- Token 2: Re-compute for tokens 1-2 (wasteful - token 1 already computed!)
- Token 3: Re-compute for tokens 1-3 (even more wasteful!)
- Token N: Re-compute for tokens 1-N

This is O(N²) redundant computation. KV-cache stores the Key and Value matrices for all previous tokens, so we only compute them once. This reduces computation from O(N²) to O(N).

**Real impact**: 5-10x faster inference

### 6. What is the difference between prefill and decode phases?

**Answer**:

**Prefill (Prompt Processing)**:
- Processes the entire input prompt at once
- Highly parallelizable - all tokens computed simultaneously
- Compute-intensive and memory-bandwidth intensive
- Fast overall but processes many tokens
- Measured by TTFT (Time to First Token)
- Example: Processing "The cat sat on the" (5 tokens in parallel)

**Decode (Token Generation)**:
- Generates one token at a time (autoregressive)
- Sequential - must wait for previous token to generate next
- Memory-bound (fetching KV-cache from memory)
- Slower per-token but necessary for generation
- Measured by TPOT (Time Per Output Token)
- Example: Generating "mat" → "and" → "purred" (one at a time)

### 7. How does sequence length affect KV-cache memory usage?

**Answer**:
KV-cache memory grows **linearly** with sequence length:

```
Memory = 2 × num_layers × batch_size × num_heads × seq_length × head_dim × bytes_per_param
```

Key points:
- **Linear growth**: Double the sequence length → double the memory
- **Per-sequence**: Each item in the batch needs its own cache
- **Layer multiplication**: Cached at every transformer layer
- **Practical limit**: Eventually run out of VRAM/RAM

Example: For LLaMA-7B with 2048 tokens = ~1GB per sequence

### 8. What's the trade-off of using KV-cache?

**Answer**:

**Benefits**:
- ✅ 5-10x faster inference
- ✅ Eliminates redundant computation
- ✅ Industry-standard technique

**Costs**:
- ❌ Increased memory usage (linear with sequence length)
- ❌ Reduces maximum batch size
- ❌ Can run out of memory with long contexts
- ❌ Memory bandwidth becomes the bottleneck

**Trade-off summary**: Speed for memory. We accept higher memory usage to avoid wasting computation.

---

## Performance

### 9. What is TTFT and why does it matter?

**Answer**:

**TTFT = Time to First Token**

It's the time from sending a request until the first generated token arrives.

**Why it matters**:
- **User experience**: This is what users perceive as "latency"
- **Responsiveness**: Determines how quickly the UI can start showing results
- **Streaming**: Critical for streaming responses (chatbots)
- **SLA target**: Often used in service-level agreements (e.g., P95 TTFT < 500ms)

**What it measures**:
- Network latency
- Queueing time (if server is busy)
- Prefill phase duration (processing the prompt)
- Scheduler overhead

**Optimization targets**:
- Reduce prefill time (faster hardware, optimized attention)
- Reduce queueing (better scheduling, more replicas)
- Use faster models for lower TTFT

### 10. Why is LLM inference memory-bound rather than compute-bound?

**Answer**:

**Memory-bound** means the bottleneck is moving data (from memory to compute units), not performing calculations.

**Reasons**:
1. **Low arithmetic intensity**: For each byte loaded, very few operations are performed
   - Loading a weight matrix: Many bytes
   - Matrix multiplication: Relatively few FLOPs per byte

2. **KV-cache fetching**: During decode, must load entire KV-cache from memory for each token
   - GPT-3 175B: Loading >100GB of KV-cache
   - Actual computation: Relatively small

3. **GPU/CPU gap**: Modern GPUs/CPUs have tons of compute but limited memory bandwidth
   - A100 GPU: 312 TFLOPS compute
   - But only: 1.5 TB/s memory bandwidth
   - Inference often saturates memory bandwidth before compute

4. **Sequential decoding**: Can't fully parallelize generation, so can't fully utilize compute units

**Implication**: Optimizations should focus on reducing memory access, not just adding more FLOPs.

### 11. What's the difference between latency and throughput?

**Answer**:

**Latency** (How fast):
- Time to process a single request from start to finish
- Measured in: seconds or milliseconds
- User-facing metric
- Lower is better
- Example: "This request took 2.5 seconds"

**Throughput** (How many):
- Number of requests processed per unit time
- Measured in: requests/second or tokens/second
- System-facing metric
- Higher is better
- Example: "The system handles 100 requests/second"

**Key insight**: Often a trade-off!
- High batch size → Higher throughput, Higher latency
- Low batch size → Lower latency, Lower throughput

**Performance engineering goal**: Optimize both within constraints
- Example: "P95 latency < 1s AND throughput > 50 req/s"

### 12. How does batching improve throughput?

**Answer**:

**Batching** processes multiple requests simultaneously on the GPU/CPU.

**Why it improves throughput**:

1. **Parallel computation**:
   - GPU has thousands of cores
   - Processing 1 sequence uses <10% of cores
   - Batching 8 sequences can use 80% of cores
   - More utilization → more throughput

2. **Amortized overhead**:
   - Model loading: Pay once for N requests
   - Kernel launch overhead: Pay once for N requests
   - Memory allocation: Shared across batch

3. **Memory efficiency**:
   - Model weights loaded once, shared across batch
   - Better cache utilization

**Example**:
```
No batching:
  Process request 1: 1s (GPU 10% utilized)
  Process request 2: 1s (GPU 10% utilized)
  Process request 8: 1s (GPU 10% utilized)
  Total: 8s for 8 requests = 1 req/s throughput

Batching (batch=8):
  Process all 8: 1.5s (GPU 80% utilized)
  Total: 1.5s for 8 requests = 5.3 req/s throughput

Throughput improved 5.3x!
```

**Trade-off**:
- Each request in the batch waits for the slowest request
- Latency increases: 1s → 1.5s in example above
- But throughput increases dramatically

---

## How did you do?

- **10-12 correct**: Excellent! You understand the fundamentals well.
- **7-9 correct**: Good! Review the areas you missed.
- **4-6 correct**: Decent start. Re-read the study guide focusing on weak areas.
- **0-3 correct**: Spend more time with the resources. Don't rush!

Remember: This is Day 1. It's OK to not understand everything perfectly yet. Come back to these concepts as you work through the practical exercises.
