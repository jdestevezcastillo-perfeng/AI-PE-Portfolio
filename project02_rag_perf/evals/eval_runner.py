import json
from pathlib import Path

import typer
import yaml

from project02_rag_perf.rag_pipeline import embed_texts, retrieve, synthetic_docs
from common.perf_timers import PerfStats, time_block

app = typer.Typer(help="RAG evaluation scaffold")


def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.command()
def main(config: Path = typer.Option(..., exists=True, readable=True)):
    cfg = load_config(config)
    docs = synthetic_docs(cfg["index"]["n_chunks"], seed=cfg["data"].get("synthetic_seed", 42))
    # Simple self-retrieval evaluation: query is a doc prefix
    queries = [doc.split(":")[0] for doc in docs[:10]]
    chunks = docs  # already chunk-like in synthetic form
    embeddings = embed_texts(chunks, cfg["index"]["embeddings_model"], cfg["index"]["use_gpu"])

    # Build temporary index in-memory
    import faiss
    import numpy as np

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    stats = PerfStats()
    hits = 0
    for i, q in enumerate(queries):
        with time_block(stats):
            q_vec = embed_texts([q], cfg["index"]["embeddings_model"], cfg["index"]["use_gpu"])
            ids = retrieve(index, q_vec, cfg["retrieval"]["top_k"])
            hits += 1 if i in ids else 0
    latency = stats.summary()
    recall = hits / len(queries) if queries else 0.0
    result = {"latency": latency, "recall@k": recall}
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
