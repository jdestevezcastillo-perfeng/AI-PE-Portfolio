import json
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import typer
import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from common.exporters import run_with_exporter
from common.logging_utils import get_logger, setup_logging
from common.perf_timers import PerfStats, time_block

app = typer.Typer(help="RAG pipeline (index, query, benchmark)")
log = get_logger("rag_pipeline")


def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def chunk_texts(texts: List[str], chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    for text in texts:
        for i in range(0, max(len(text) - chunk_size, 1), chunk_size - overlap):
            chunks.append(text[i : i + chunk_size])
    return chunks


def synthetic_docs(n: int, seed: int = 42) -> List[str]:
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        docs.append(f"Doc {i}: " + " ".join([f"token{rng.integers(0, 1000)}" for _ in range(100)]))
    return docs


def embed_texts(texts: List[str], model_name: str, use_gpu: bool):
    try:
        device = "cuda" if use_gpu else "cpu"
        model = SentenceTransformer(model_name, device=device)
    except Exception as exc:  # noqa: BLE001
        log.warning("Embedding model load failed, falling back to CPU", error=str(exc))
        model = SentenceTransformer(model_name, device="cpu")
    return np.array(model.encode(texts, convert_to_numpy=True, show_progress_bar=False))


def build_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index


def retrieve(index, query_vec: np.ndarray, top_k: int):
    distances, ids = index.search(query_vec.astype(np.float32), top_k)
    return ids[0]


def generate_answer(context: List[str], question: str, model_name: str, max_new_tokens: int, use_gpu: bool) -> str:
    try:
        device = 0 if use_gpu else -1
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        gen_pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device=device)
        prompt = f"Context: {' '.join(context)}\nQuestion: {question}\nAnswer:"
        output = gen_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return output[0]["generated_text"]
    except Exception as exc:  # noqa: BLE001
        log.warning("Generation failed, returning synthetic answer", error=str(exc))
        return f"[synthetic answer] {question}"


@app.command()
def index(config: Path = typer.Option(..., exists=True, readable=True), log_level: str = "INFO"):
    setup_logging(log_level)
    cfg = load_config(config)
    docs = synthetic_docs(cfg["index"]["n_chunks"], seed=cfg["data"].get("synthetic_seed", 42))
    chunks = chunk_texts(docs, cfg["index"]["chunk_size"], cfg["index"]["overlap"])
    embeddings = embed_texts(chunks, cfg["index"]["embeddings_model"], cfg["index"]["use_gpu"])
    index_obj = build_index(embeddings)
    typer.echo(f"Built index with {index_obj.ntotal} vectors")
    idx_path = Path("data/index.faiss")
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index_obj, str(idx_path))
    Path("data/chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    typer.echo(f"Saved index to {idx_path}")


@app.command()
def query(
    q: str = typer.Option(..., help="Query string"),
    config: Path = typer.Option(..., exists=True, readable=True),
    log_level: str = "INFO",
):
    setup_logging(log_level)
    cfg = load_config(config)
    index_path = Path("data/index.faiss")
    if not index_path.exists():
        raise FileNotFoundError("Index not found, run `index` first")
    index_obj = faiss.read_index(str(index_path))
    chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
    q_vec = embed_texts([q], cfg["index"]["embeddings_model"], cfg["index"]["use_gpu"])
    ids = retrieve(index_obj, q_vec, cfg["retrieval"]["top_k"])
    context = [chunks[i] for i in ids]
    answer = generate_answer(context, q, cfg["generation"]["model_name"], cfg["generation"]["max_new_tokens"], cfg["generation"]["use_gpu"])
    typer.echo(answer)


@app.command()
def benchmark(config: Path = typer.Option(..., exists=True, readable=True), log_level: str = "INFO", exporter_port: Optional[int] = None):
    setup_logging(log_level)
    cfg = load_config(config)
    queries = [f"Query {i}" for i in range(20)]

    def one_query():
        q = queries.pop() if queries else "repeat query"
        index_obj = faiss.read_index("data/index.faiss")
        chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
        q_vec = embed_texts([q], cfg["index"]["embeddings_model"], cfg["index"]["use_gpu"])
        ids = retrieve(index_obj, q_vec, cfg["retrieval"]["top_k"])
        context = [chunks[i] for i in ids]
        _ = generate_answer(context, q, cfg["generation"]["model_name"], cfg["generation"]["max_new_tokens"], cfg["generation"]["use_gpu"])

    if exporter_port:
        summary = run_with_exporter(one_query, port=int(exporter_port), runs=10)
    else:
        stats = PerfStats()
        for _ in range(10):
            with time_block(stats):
                one_query()
        summary = stats.summary()
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
