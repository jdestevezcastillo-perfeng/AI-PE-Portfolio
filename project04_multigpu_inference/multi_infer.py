import json
import queue
import threading
from pathlib import Path
from typing import Dict, List, Optional

import typer
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from common.exporters import run_with_exporter
from common.logging_utils import get_logger, setup_logging
from common.perf_timers import PerfStats, time_block

app = typer.Typer(help="Multi-GPU inference scaffold with CPU fallback")
log = get_logger("multi_infer")


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pipe(model_name: str, device: str):
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name, device_map=None if device == "cpu" else {0: device})
        device_id = -1 if device == "cpu" else 0
        return pipeline("text-generation", model=mdl, tokenizer=tok, device=device_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("Pipeline load failed, falling back to CPU", error=str(exc))
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        return pipeline("text-generation", model=mdl, tokenizer=tok, device=-1)


def worker(task_q: "queue.Queue[str]", results: List[str], pipe, max_new_tokens: int):
    while True:
        try:
            prompt = task_q.get_nowait()
        except queue.Empty:
            break
        _ = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        results.append("ok")
        task_q.task_done()


def run_batches(cfg: Dict):
    prompts = [f"Prompt {i}" for i in range(cfg["batch_size"] * cfg["runs"])]
    stats = PerfStats()
    devices = cfg.get("devices", ["cpu"])
    use_gpu = cfg.get("use_gpu", False)

    for _ in range(cfg["runs"]):
        batch = [prompts.pop() for _ in range(cfg["batch_size"])]
        with time_block(stats):
            # TODO: proper device assignment, real tensor parallel when GPU present
            device = devices[0] if use_gpu else "cpu"
            pipe = load_pipe(cfg["model_name"], device)
            task_q: queue.Queue[str] = queue.Queue()
            for p in batch:
                task_q.put(p)
            threads = []
            results: List[str] = []
            for _ in range(len(batch)):
                t = threading.Thread(target=worker, args=(task_q, results, pipe, cfg["max_new_tokens"]))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
    return stats.summary()


@app.command()
def main(config: Path = typer.Option(..., exists=True, readable=True), log_level: str = "INFO", exporter_port: Optional[int] = None):
    setup_logging(log_level)
    cfg = load_config(config)
    exporter = exporter_port or cfg.get("exporter_port")
    if exporter:
        summary = run_with_exporter(lambda: run_batches(cfg), port=int(exporter), runs=1)
    else:
        summary = run_batches(cfg)
    log.info("Multi infer summary", summary=json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
