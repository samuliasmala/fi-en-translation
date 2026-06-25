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


def load(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def translate(text, tokenizer, model):
    """Full per-query pipeline: tokenize -> generate -> decode."""
    batch = tokenizer([text], return_tensors="pt", truncation=True)
    # The model's generation_config already sets max_length=512, which is far
    # longer than any of these queries needs. We rely on it rather than also
    # passing max_new_tokens, which would make transformers warn on every call.
    generated = model.generate(**batch)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def benchmark_model(model_name, queries, save_handle=None):
    print(f"\nLoading {model_name} ...", flush=True)
    t0 = time.perf_counter()
    tokenizer, model = load(model_name)
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
    print("=" * 70)
    print("Finnish -> English translation benchmark")
    print(f"  queries     : {len(QUERIES)} (warm-up excluded: {WARMUP})")
    print(f"  device      : {device}")
    print(f"  torch threads: {torch.get_num_threads()}")
    print(f"  torch        : {torch.__version__}")
    print("=" * 70)

    save_handle = open(args.save, "w", encoding="utf-8") if args.save else None
    try:
        results = [benchmark_model(m, QUERIES, save_handle) for m in MODELS]
    finally:
        if save_handle:
            save_handle.close()

    # ---- summary table ----
    print("\n" + "=" * 70)
    print("RESULTS — average time per translation (batch size 1, "
          f"{results[0]['n_timed']} timed queries each)")
    print("=" * 70)
    header = f"{'model':40s} {'avg/query':>11s} {'median':>9s} {'min':>8s} {'max':>8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['model']:40s} "
              f"{r['avg'] * 1000:8.1f} ms "
              f"{r['median'] * 1000:6.1f} ms "
              f"{r['min'] * 1000:5.0f} ms "
              f"{r['max'] * 1000:5.0f} ms")

    print("\nHeadline:")
    for r in results:
        print(f"  {r['model']:40s}  {r['avg'] * 1000:7.1f} ms/translation "
              f"(±{r['stdev'] * 1000:.1f} ms)")

    if len(results) == 2:
        slower, faster = sorted(results, key=lambda r: r["avg"], reverse=True)
        ratio = slower["avg"] / faster["avg"]
        print(f"\n  -> {faster['model'].split('/')[-1]} is {ratio:.2f}x faster "
              f"per translation than {slower['model'].split('/')[-1]}.")

    if save_handle:
        print(f"\nAll translations written to: {args.save}")


if __name__ == "__main__":
    main()
