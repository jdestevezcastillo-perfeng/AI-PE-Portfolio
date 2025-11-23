# Transformer Architecture & Tool Mapping

## 1. Original Encoder-Decoder (The "Translator")

*As proposed in "Attention Is All You Need" (2017). Used for translation (English -> French).*

```mermaid
graph TD
    subgraph "Encoder (The Reader)"
        Input[Input Tokens] --> Emb["Embedding<br/>(torch.nn.Embedding)"]
        Emb --> Pos["Positional Encoding<br/>(Math / Tensor Add)"]
        Pos --> EncBlock["Encoder Block<br/>(transformers...BertLayer)"]
        
        subgraph "Inside Encoder Block"
            EncBlock --> SA["Self-Attention<br/>(torch.nn.MultiheadAttention)"]
            SA --> AddNorm1["Add & Norm<br/>(torch.nn.LayerNorm)"]
            AddNorm1 --> FFN["Feed Forward<br/>(Linear + GELU)"]
            FFN --> AddNorm2["Add & Norm<br/>(torch.nn.LayerNorm)"]
        end
    end

    subgraph "Decoder (The Writer)"
        Target["Target Tokens<br/>(Shifted)"] --> TEmb["Embedding<br/>(torch.nn.Embedding)"]
        TEmb --> TPos[Positional Encoding]
        TPos --> DecBlock["Decoder Block<br/>(transformers...BartDecoderLayer)"]
        
        subgraph "Inside Decoder Block"
            DecBlock --> MSA["Masked Self-Attention<br/>(torch.nn.MultiheadAttention)"]
            MSA --> DAddNorm1[Add & Norm]
            DAddNorm1 --> CA["Cross-Attention<br/>(Decoder Q, Encoder K/V)"]
            AddNorm2 -.-> CA
            CA --> DAddNorm2[Add & Norm]
            DAddNorm2 --> DFFN[Feed Forward]
            DFFN --> DAddNorm3[Add & Norm]
        end

        DAddNorm3 --> Linear["Linear Head<br/>(torch.nn.Linear)"]
        Linear --> Softmax["Softmax<br/>(torch.nn.Softmax)"] --> Output[Output Probabilities]
    end
```

---

## 2. Decoder-Only (The "Predictor")

*The standard for modern LLMs (GPT-3, Llama, Mistral). It just predicts the next token.*

```mermaid
graph LR
    subgraph "The LLM (GPT Style)"
        Input[Input Prompt] --> Emb["Embedding<br/>(torch.nn.Embedding)"]
        Emb --> Pos["Positional Encoding<br/>(RoPE)"]
        Pos --> Block["Transformer Block<br/>(transformers...LlamaDecoderLayer)"]
        
        subgraph "Inside the Block (Repeated 32x-80x)"
            Block --> Norm1["RMS Norm<br/>(LlamaRMSNorm)"]
            Norm1 --> Attn["Self-Attention<br/>(LlamaAttention)"]
            
            subgraph "Attention Internals"
                Attn -- Q, K, V --> QKV["Q, K, V Projections<br/>(torch.nn.Linear)"]
                QKV --> Matrix["Matrix Mul (Q*K)"]
                Matrix --> Soft[Softmax]
                Soft --> Context["Context * V"]
                Context --> OutProj["Output Projection<br/>(torch.nn.Linear)"]
            end
            
            Attn --> Add1["Add (Residual)"]
            Add1 --> Norm2[RMS Norm]
            Norm2 --> MLP["MLP / FFN<br/>(LlamaMLP)"]
            
            subgraph "MLP Internals (SwiGLU)"
                MLP --> Gate["Gate Proj<br/>(torch.nn.Linear)"]
                MLP --> Up["Up Proj<br/>(torch.nn.Linear)"]
                Gate & Up --> Act["SiLU Activation<br/>(functional.silu)"]
                Act --> Mult[Multiply]
                Mult --> Down["Down Proj<br/>(torch.nn.Linear)"]
            end
            
            MLP --> Add2["Add (Residual)"]
        end

        Add2 --> FinalNorm[Final Norm]
        FinalNorm --> Head["LM Head (Unembed)<br/>(torch.nn.Linear)"]
        Head --> Softmax[Softmax] --> NextToken[Next Token]
    end
```

## 🔑 Key Differences for Engineers

| Feature | Encoder-Decoder (Original) | Decoder-Only (GPT/Llama) |
| :--- | :--- | :--- |
| **Primary Use** | Translation (Seq2Seq) | Text Generation (Next Token Prediction) |
| **Attention** | Has **Cross-Attention** (Decoder looks at Encoder) | **Self-Attention Only** (Looks at past tokens) |
| **Complexity** | 2 Stacks (Encoder + Decoder) | 1 Stack (Decoder) |
| **Library Class** | `BartModel`, `T5Model` | `LlamaForCausalLM`, `GPT2LMHeadModel` |
| **KV Cache** | Caches Decoder states only | Caches **everything** (since everything is decoder) |
