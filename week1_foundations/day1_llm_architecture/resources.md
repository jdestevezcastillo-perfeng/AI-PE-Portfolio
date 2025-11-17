# Day 1 Resources & References

Comprehensive list of resources for Day 1 learning.

---

## Primary Resources (Must Read/Watch)

### Transformer Architecture

1. **The Illustrated Transformer** by Jay Alammar
   - URL: https://jalammar.github.io/illustrated-transformer/
   - Type: Blog post with visualizations
   - Time: 30-45 minutes
   - Why: Best visual explanation of transformer architecture
   - Key topics: Self-attention, multi-head attention, position encodings

2. **Attention Is All You Need** (Original Paper)
   - URL: https://arxiv.org/abs/1706.03762
   - Authors: Vaswani et al., 2017
   - Type: Research paper
   - Time: 1-2 hours (can skim sections 3.1-3.3)
   - Why: The foundational paper introducing transformers

3. **The Illustrated GPT-2** by Jay Alammar
   - URL: https://jalammar.github.io/illustrated-gpt2/
   - Type: Blog post with visualizations
   - Time: 30-40 minutes
   - Why: Shows how transformers work in autoregressive models
   - Key topics: Decoder-only architecture, token generation

### KV-Cache & Inference

4. **Hugging Face KV-Cache Documentation**
   - URL: https://huggingface.co/docs/transformers/main/en/kv_cache
   - Type: Documentation
   - Time: 15-20 minutes
   - Why: Technical details on KV-cache implementation

5. **vLLM: Easy, Fast, and Cheap LLM Serving**
   - URL: https://arxiv.org/abs/2309.06180
   - Authors: Kwon et al., 2023
   - Type: Research paper
   - Time: 45-60 minutes
   - Why: Introduces PagedAttention for efficient KV-cache management
   - Key topics: Memory management, batching strategies

6. **LLM Inference Optimization** by Lilian Weng
   - URL: https://lilianweng.github.io/posts/2023-01-10-inference-optimization/
   - Type: Blog post
   - Time: 45-60 minutes
   - Why: Comprehensive overview of optimization techniques
   - Key topics: Quantization, pruning, distillation, batching

---

## Supplementary Resources (Recommended)

### Video Tutorials

7. **Let's Build GPT** by Andrej Karpathy
   - URL: https://www.youtube.com/watch?v=kCc8FmEb1nY
   - Type: Video (1h 57min)
   - Why: Build a GPT from scratch, understand every component
   - Note: Long but extremely valuable

8. **Attention Mechanism Explained** by StatQuest
   - URL: https://www.youtube.com/watch?v=PSs6nxngL6k
   - Type: Video (15min)
   - Why: Simple, clear explanation of attention basics

9. **Transformer Neural Networks Explained** by CodeEmporium
   - URL: https://www.youtube.com/watch?v=TQQlZhbC5ps
   - Type: Video (14min)
   - Why: Quick visual overview of transformer components

### Interactive Tools

10. **Transformer Explainer**
    - URL: https://poloclub.github.io/transformer-explainer/
    - Type: Interactive visualization
    - Why: See transformer operations in real-time
    - How to use: Input text and watch attention flow

11. **LLM Visualization by bbycroft**
    - URL: https://bbycroft.net/llm
    - Type: Interactive 3D visualization
    - Why: Incredible 3D visualization of GPT-2 internals
    - Highly recommended!

12. **Attention Visualizer**
    - URL: https://github.com/jessevig/bertviz
    - Type: Tool/Library
    - Why: Visualize attention patterns in real models

### Blog Posts & Articles

13. **Making Deep Learning Go Brrrr** by Horace He
    - URL: https://horace.io/brrr_intro.html
    - Why: Understand hardware/software co-optimization
    - Key topics: Memory hierarchy, GPU optimization

14. **Building LLM Applications for Production** by Chip Huyen
    - URL: https://huyenchip.com/2023/04/11/llm-engineering.html
    - Why: Production considerations and best practices
    - Focus: Sections on inference and performance

15. **Patterns for Building LLM Apps** by Eugene Yan
    - URL: https://eugeneyan.com/writing/llm-patterns/
    - Why: Practical patterns and anti-patterns
    - Focus: Performance and cost optimization sections

16. **The Illustrated Word2vec** by Jay Alammar
    - URL: https://jalammar.github.io/illustrated-word2vec/
    - Why: Understand embeddings (foundation for transformers)
    - Time: 20-30 minutes

### Research Papers (Advanced)

17. **Flash Attention**
    - URL: https://arxiv.org/abs/2205.14135
    - Authors: Dao et al., 2022
    - Why: Faster attention computation
    - Key concept: I/O-aware algorithm design

18. **GPT-3: Language Models are Few-Shot Learners**
    - URL: https://arxiv.org/abs/2005.14165
    - Authors: Brown et al., 2020
    - Why: Architecture scaling insights
    - Focus: Section 2 (architecture) and Appendix B (hyperparameters)

19. **LLaMA: Open and Efficient Foundation Language Models**
    - URL: https://arxiv.org/abs/2302.13971
    - Authors: Touvron et al., 2023
    - Why: Modern efficient architecture design
    - Focus: Sections 2-3 (architecture and training)

20. **Speculative Decoding**
    - URL: https://arxiv.org/abs/2211.17192
    - Authors: Leviathan et al., 2022
    - Why: 2-3x speedup technique for inference
    - Key concept: Draft-verify paradigm

---

## Tools & Documentation

### Ollama

21. **Ollama Documentation**
    - URL: https://ollama.ai/
    - Type: Official docs
    - Sections to read:
      - Getting Started
      - API Reference
      - Model Library

22. **Ollama GitHub Repository**
    - URL: https://github.com/ollama/ollama
    - Why: Source code, issues, examples
    - Check: Issues for troubleshooting

### Python Libraries

23. **OpenAI Python Client**
    - URL: https://github.com/openai/openai-python
    - Why: Compatible API for LLM interaction
    - Note: Ollama supports OpenAI-compatible API

24. **Requests Documentation**
    - URL: https://requests.readthedocs.io/
    - Why: Making HTTP API calls

25. **Matplotlib Documentation**
    - URL: https://matplotlib.org/stable/index.html
    - Why: Creating performance visualizations

26. **Seaborn Gallery**
    - URL: https://seaborn.pydata.org/examples/index.html
    - Why: Beautiful statistical visualizations

### Performance Monitoring

27. **psutil Documentation**
    - URL: https://psutil.readthedocs.io/
    - Why: System resource monitoring in Python

28. **htop Tutorial**
    - URL: https://www.man7.org/linux/man-pages/man1/htop.1.html
    - Why: Interactive process viewer

29. **NVIDIA nvidia-smi**
    - URL: https://developer.nvidia.com/nvidia-system-management-interface
    - Why: GPU monitoring (if you have NVIDIA GPU)

---

## Community & Forums

### Discussion Platforms

30. **r/LocalLLaMA (Reddit)**
    - URL: https://reddit.com/r/LocalLLaMA
    - Why: Community running local LLMs, great for tips

31. **r/MachineLearning (Reddit)**
    - URL: https://reddit.com/r/MachineLearning
    - Why: Research discussions and paper summaries

32. **Hugging Face Forums**
    - URL: https://discuss.huggingface.co/
    - Why: Technical discussions on transformers and LLMs

33. **Ollama Discord**
    - URL: https://discord.gg/ollama
    - Why: Real-time help with Ollama issues

### Blogs to Follow

34. **Lilian Weng's Blog**
    - URL: https://lilianweng.github.io/
    - Why: High-quality technical posts on LLMs

35. **Jay Alammar's Blog**
    - URL: https://jalammar.github.io/
    - Why: Best visualizations of ML concepts

36. **Sebastian Raschka's Blog**
    - URL: https://sebastianraschka.com/blog/
    - Why: Practical ML and DL insights

37. **Hugging Face Blog**
    - URL: https://huggingface.co/blog
    - Why: Latest developments in transformers and LLMs

---

## Performance Engineering Specific

### Benchmarking Resources

38. **MLPerf Inference Benchmark**
    - URL: https://mlcommons.org/en/inference-edge-41/
    - Why: Industry-standard LLM benchmarks
    - Note: Reference for metrics and methodology

39. **LLM Benchmark Papers**
    - URL: https://paperswithcode.com/task/language-modelling
    - Why: Compare your results with state-of-the-art

### System Performance

40. **Brendan Gregg's Blog**
    - URL: https://www.brendangregg.com/blog/index.html
    - Why: Performance engineering bible
    - Relevant: Linux Performance, Flame Graphs

41. **The Architecture of Open Source Applications**
    - URL: https://aosabook.org/
    - Why: Understanding system design
    - Relevant: Volume II (Performance)

---

## Books (Optional, for Deep Dive)

42. **"Understanding Deep Learning"** by Simon J.D. Prince
    - URL: https://udlbook.github.io/udlbook/ (free online)
    - Chapters 12-13: Transformers
    - Time: Several hours per chapter

43. **"Deep Learning"** by Goodfellow, Bengio, Courville
    - URL: https://www.deeplearningbook.org/ (free online)
    - Chapter 10: Sequence Modeling
    - Classic reference text

44. **"Natural Language Processing with Transformers"** by Tunstall et al.
    - Publisher: O'Reilly
    - Why: Practical guide to using transformers
    - Chapter 2-4: Transformer architecture and fine-tuning

---

## Datasets & Prompts (For Testing)

45. **ShareGPT Dataset**
    - URL: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
    - Why: Real conversation prompts for benchmarking

46. **Alpaca Evaluation Set**
    - URL: https://github.com/tatsu-lab/stanford_alpaca
    - Why: Diverse task prompts

---

## Keeping Up to Date

### News & Updates

47. **Papers with Code**
    - URL: https://paperswithcode.com/
    - Why: Latest ML research with code implementations

48. **Hugging Face Papers**
    - URL: https://huggingface.co/papers
    - Why: Curated daily ML papers

49. **AI News Aggregators**
    - URL: https://www.artificialnewsdaily.com/
    - Why: Daily AI/ML news roundup

### Twitter/X Follows (Optional)

50. Key people to follow:
    - @karpathy (Andrej Karpathy)
    - @_jasonwei (Jason Wei - Chain of Thought)
    - @ylecun (Yann LeCun)
    - @sama (Sam Altman)
    - @jeremyphoward (Jeremy Howard - fast.ai)

---

## Quick Reference Cheat Sheets

51. **Transformer Math Formulas**
    ```
    Self-Attention:
    Attention(Q, K, V) = softmax(QK^T / √d_k)V

    Q = X·W_Q
    K = X·W_K
    V = X·W_V

    Multi-Head:
    head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
    ```

52. **Performance Metrics Formulas**
    ```
    TTFT = Time to First Token
    TPOT = Time Per Output Token
    TPS = Tokens Per Second = eval_count / (eval_duration_ns / 1e9)

    Latency = TTFT + (num_tokens × TPOT)
    Throughput = batch_size / latency

    KV-cache size (bytes) = 2 × layers × batch × heads × seq_len × head_dim × bytes_per_param
    ```

---

## Resource Organization Tips

**For Day 1, prioritize**:
1. Read: Resources #1, #3, #4, #6 (Alammar blogs, HF docs, Lilian's blog)
2. Watch: Resource #8 (StatQuest - quick and clear)
3. Interact: Resource #10, #11 (Transformer visualizers)
4. Reference: This resources.md file as needed

**Bookmark for later**:
- Research papers (deep dive after basics)
- Books (ongoing learning)
- Community forums (when stuck)

**Daily habit**:
- Skim Hugging Face Papers (#48)
- Check r/LocalLLaMA for tips (#30)

---

## Contributing

Found a great resource not listed here? Add it to your own fork and consider:
1. Brief description
2. URL
3. Why it's useful
4. Estimated time to review

---

**Last Updated**: [Date]
**Maintained by**: [Your Name]

---

## License & Attribution

All external resources belong to their respective owners. This compilation is for educational purposes only.
