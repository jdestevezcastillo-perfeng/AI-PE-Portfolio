# Module 01 References: Mastering LLM Architecture

A curated list of the most important resources for understanding how Large Language Models work under the hood, selected specifically for a Performance Engineer.

> **💡 Strategy Note:** As a Performance Engineer, you do **not** need to master the deep math or derivation of gradients. Focus on the **Systems View**:
> *   **Memory:** How data moves (Weights vs KV Cache).
> *   **Compute:** Arithmetic intensity (Bandwidth bound vs Compute bound).
> *   **IO:** How attention scales with context length.

## 📜 Seminal Papers (The "Must Reads")
If you only read three papers, read these. They define the modern stack.

1.  **[Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)**
    *   *The Origin Story.* Introduces the Transformer architecture.
    *   **PE Focus:** Skip the translation results. Focus on **Figure 1 (Architecture)** and the **complexity analysis** of Self-Attention ($O(N^2)$). That $O(N^2)$ is your enemy.

2.  **[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)](https://arxiv.org/abs/2205.14135)**
    *   *The Performance Breakthrough.* Explains how to optimize attention for GPU memory hierarchy (HBM vs SRAM).
    *   **PE Focus:** **CRITICAL.** This paper is pure systems engineering. Read it to understand tiling and recomputation.

3.  **[Llama 2: Open Foundation and Fine-Tuned Chat Models (2023)](https://arxiv.org/abs/2307.09288)**
    *   *The Open Standard.*
    *   **PE Focus:** Look at **GQA (Grouped Query Attention)**. It was invented specifically to reduce KV Cache size and improve inference throughput.

## 🧠 Visual Guides & Blogs
For building intuition before diving into the math.

*   **[The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)**
    *   **Priority: HIGH.** The single best visual explanation. If you can visualize how the matrices multiply, you can optimize them.
*   **[LLM Parameter Counting (Kiwibot)](https://kipp.ly/transformer-param-count/)**
    *   **Priority: CRITICAL.** A breakdown of exactly where the memory goes. This is your "Capacity Planning" bible.
*   **[Lil'Log: Prompt Engineering & Transformer Architecture](https://lilianweng.github.io/)**
    *   **Priority: Medium.** Good for deep dives, but can get theoretical.

## 📺 Videos & Courses
*   **[Andrej Karpathy: Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)**
    *   **Priority: HIGH.** He builds a GPT model line-by-line. Watch the first hour to understand the data structures.
*   **[Fast.ai: Deep Learning for Coders (Part 2)](https://course.fast.ai/)**
    *   **Priority: Low.** Good for general DL, but less focused on inference systems.
*   **[Umar Jamil: Attention is All You Need (Annotated Paper Walkthrough)](https://www.youtube.com/watch?v=bCz4OMemCcA)**
    *   **Priority: Medium.** Good if you get stuck on the paper.

## 📚 Books
*   **[Build a Large Language Model (From Scratch) by Sebastian Raschka](https://www.manning.com/books/build-a-large-language-model-from-scratch)**
    *   **Priority: Medium.** Excellent reference, but don't get bogged down in the training code unless you are doing fine-tuning.

## 💾 Quantization & Systems (Performance Engineering Specific)
*   **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)**
    *   **Priority: HIGH.** Explains how to fit big models on small cards.
*   **[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)**
    *   **Priority: Medium.** Explains the difficulty of quantizing activation outliers.
*   **[The Secret Sauce of vLLM: PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)**
    *   **Priority: CRITICAL.** Read this. It treats memory management like an OS problem (paging), which will be immediately familiar to you.

## 🗓️ Conferences to Follow
*   **MLSys (Machine Learning and Systems):** *Your home turf.* The intersection of AI and Systems Engineering/Performance.
