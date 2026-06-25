# -*- coding: utf-8 -*-
"""
Benchmark two Finnish->English translation models on per-translation latency:

  1. Helsinki-NLP/opus-mt-fi-en        (smaller, classic OPUS-MT)
  2. Helsinki-NLP/opus-mt-tc-big-fi-en (larger "tc-big" model)

Each of the 100 Finnish medical queries (see queries.py) is translated one at a
time (batch size 1) so the measured time reflects real per-query latency such as
a doctor would experience typing one search at a time. The first few translations
per model are treated as warm-up and excluded from the average.

Usage:
    .venv/bin/python benchmark.py
"""

import argparse
import platform
import statistics
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from queries import QUERIES

MODELS = [
    "Helsinki-NLP/opus-mt-fi-en",
    "Helsinki-NLP/opus-mt-tc-big-fi-en",
]

WARMUP = 3  # translations excluded from timing stats (model/threadpool warm-up)


def load(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def translate(text, tokenizer, model):
    """Full per-query pipeline: tokenize -> generate -> decode."""
    # Inputs must live on the same device as the model; .to() is a no-op on CPU.
    batch = tokenizer([text], return_tensors="pt", truncation=True).to(model.device)
    # The model's generation_config already sets max_length=512, which is far
    # longer than any of these queries needs. We rely on it rather than also
    # passing max_new_tokens, which would make transformers warn on every call.
    generated = model.generate(**batch)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def benchmark_model(model_name, queries, device, save_handle=None):
    print(f"\nLoading {model_name} ...", flush=True)
    t0 = time.perf_counter()
    tokenizer, model = load(model_name, device)
    load_secs = time.perf_counter() - t0
    print(f"  loaded in {load_secs:.1f}s", flush=True)

    times = []          # timed (post-warmup) per-query seconds
    translations = []   # (query, english, secs) for every query

    for i, q in enumerate(queries):
        t = time.perf_counter()
        en = translate(q, tokenizer, model)
        dt = time.perf_counter() - t
        translations.append((q, en, dt))
        if i >= WARMUP:
            times.append(dt)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(queries)} translated", flush=True)

    if save_handle is not None:
        save_handle.write(f"\n===== {model_name} =====\n")
        for q, en, dt in translations:
            save_handle.write(f"[{dt * 1000:7.1f} ms] {q}\n           -> {en}\n")

    return {
        "model": model_name,
        "load_secs": load_secs,
        "n_timed": len(times),
        "avg": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "total": sum(times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=None,
                        help="torch CPU thread count (default: torch default)")
    parser.add_argument("--save", default="translations.txt",
                        help="file to write all translations to (default: translations.txt)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_note = "GPU present" if device == "cuda" else "no GPU"
    print("Finnish -> English translation benchmark")
    print("(run environment & method summary shown with the results below)\n")

    save_handle = open(args.save, "w", encoding="utf-8") if args.save else None
    try:
        results = [benchmark_model(m, QUERIES, device, save_handle) for m in MODELS]
    finally:
        if save_handle:
            save_handle.close()

    print(render_summary(results, device, gpu_note,
                          save_path=args.save if save_handle else None))


def render_summary(results, device, gpu_note, save_path=None):
    """Build the results + environment/method block.

    Separator lines are sized to the widest content line so they always span
    the full text. The environment/method info sits beside the results here
    (rather than at the top of the run) so it reads as a footnote to the numbers.
    """
    results_title = ("RESULTS — average time per translation (batch size 1, "
                     f"{results[0]['n_timed']} timed queries each)")
    table_header = (f"{'model':40s} {'avg/query':>11s} {'median':>9s} "
                    f"{'min':>8s} {'max':>8s}")
    rows = [
        f"{r['model']:40s} {r['avg'] * 1000:8.1f} ms {r['median'] * 1000:6.1f} ms "
        f"{r['min'] * 1000:5.0f} ms {r['max'] * 1000:5.0f} ms"
        for r in results
    ]

    headline = ["Headline:"] + [
        f"  {r['model']:40s}  {r['avg'] * 1000:7.1f} ms/translation "
        f"(±{r['stdev'] * 1000:.1f} ms)"
        for r in results
    ]
    if len(results) == 2:
        slower, faster = sorted(results, key=lambda r: r["avg"], reverse=True)
        ratio = slower["avg"] / faster["avg"]
        headline += ["", f"  -> {faster['model'].split('/')[-1]} is {ratio:.2f}x "
                         f"faster per translation than {slower['model'].split('/')[-1]}."]

    env = [
        "Environment & method",
        f"  queries      : {len(QUERIES)} (warm-up excluded: {WARMUP})",
        f"  device       : {device} ({gpu_note})",
        f"  torch threads: {torch.get_num_threads()}",
        f"  python       : {platform.python_version()}",
        f"  torch        : {torch.__version__}",
        "",
        (f"  Run environment: Python {platform.python_version()}, torch "
         f"{torch.__version__} on {device.upper()}, "
         f"{torch.get_num_threads()} threads, {gpu_note}."),
        "  Timing covers the full per-query pipeline (tokenize -> beam-search",
        "  generate -> decode), one query at a time to mirror a doctor typing",
        f"  one search at a time. First {WARMUP} translations per model excluded "
        "as warm-up.",
    ]
    if save_path:
        env += ["", f"  All translations written to: {save_path}"]

    width = max(len(line) for line in
                [results_title, table_header, *rows, *headline, *env])
    eq, dash = "=" * width, "-" * width

    block = [eq, results_title, eq, table_header, dash, *rows, dash,
             *headline, dash, *env, eq]
    return "\n" + "\n".join(block)


if __name__ == "__main__":
    main()
