import json
import random
import time
from pathlib import Path
from typing import Dict, Optional

import typer
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from common.exporters import run_with_exporter
from common.logging_utils import get_logger, setup_logging
from common.perf_timers import PerfStats, time_block

app = typer.Typer(help="LLM benchmark runner")
log = get_logger("bench_runner")


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def synthetic_step(prompt_len: int, gen_tokens: int):
    # No GPU needed; pure CPU math to mimic generation cost
    import numpy as np

    _ = np.random.randn(prompt_len, gen_tokens) @ np.random.randn(gen_tokens, prompt_len)


def load_model(model_name: str, use_gpu: bool):
    device = 0 if use_gpu else -1
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto" if use_gpu else None)
        pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device=device)
        return pipe
    except Exception as exc:  # noqa: BLE001
        log.warning("Model load failed, falling back to synthetic mode", error=str(exc))
        return None


def run_generation(pipe, prompt: str, max_new_tokens: int) -> str:
    if pipe is None:
        # synthetic path
        synthetic_step(len(prompt), max_new_tokens)
        return "synthetic-output"
    outputs = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return outputs[0]["generated_text"]


def run_benchmark(cfg: Dict, exporter_port: Optional[int] = None):
    stats = PerfStats()
    pipe = None

    if cfg.get("task") != "synthetic":
        pipe = load_model(cfg["model_name"], cfg.get("use_gpu", False))

    prompts = [f"Benchmark prompt {i} with randomness {random.random():.3f}" for i in range(cfg["runs"])]
    for i in range(cfg["runs"]):
        prompt = prompts[i][-cfg.get("prompt_length", 128) :]
        with time_block(stats):
            _ = run_generation(pipe, prompt, cfg.get("generation_tokens", 64))
    summary = stats.summary()
    summary["config"] = cfg
    return summary


@app.command()
def main(
    config: Path = typer.Option(..., exists=True, readable=True),
    log_level: str = "INFO",
    exporter_port: Optional[int] = None,
):
    setup_logging(log_level)
    cfg = load_config(config)
    exporter = exporter_port or cfg.get("exporter_port")
    if exporter:
        summary = run_with_exporter(lambda: run_benchmark(cfg), port=int(exporter), runs=1)
    else:
        summary = run_benchmark(cfg, exporter)
    log.info("Benchmark summary", summary=json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
