"""
Phase 4 — KV cache: run both versions, prove identity, measure speedup.

Run:
    python phase4_kvcache.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import load_weights, precompute_rope, greedy_decode, greedy_decode_cached
from phase2_tokenizer import BPETokenizer
from benchmark import Benchmark

MODEL_DIR = Path(__file__).parent / "model" / "qwen2.5-0.5b-instruct"

PROMPT = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "What is 2 + 2?<|im_end|>\n"
    "<|im_start|>assistant\n"
)
MAX_NEW_TOKENS = 20  # long enough to show speedup, short enough to finish fast


def main():
    print("Loading tokenizer...")
    tokenizer = BPETokenizer(MODEL_DIR)

    print("Loading weights...")
    t0 = time.perf_counter()
    weights, config = load_weights(MODEL_DIR)
    rope_cos, rope_sin = precompute_rope(
        config.head_dim, config.max_position_embeddings, config.rope_theta
    )
    print(f"  done in {time.perf_counter()-t0:.1f}s\n")

    prompt_ids = tokenizer.encode(PROMPT)
    print(f"Prompt: {len(prompt_ids)} tokens\n")

    # ── No-cache baseline ──────────────────────────────────────────────────
    print(f"Running WITHOUT cache (max {MAX_NEW_TOKENS} tokens)...")
    t_nc = time.perf_counter()
    gen_nocache, tps_nocache = greedy_decode(
        prompt_ids, weights, config, rope_cos, rope_sin,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    t_nc = time.perf_counter() - t_nc
    text_nocache = tokenizer.decode(gen_nocache)
    print(f"  Output : {repr(text_nocache)}")
    print(f"  Tokens : {len(gen_nocache)}   Time: {t_nc:.1f}s   Speed: {tps_nocache:.2f} tok/s\n")

    # ── Cached ─────────────────────────────────────────────────────────────
    print(f"Running WITH cache (max {MAX_NEW_TOKENS} tokens)...")
    t_c = time.perf_counter()
    gen_cached, prefill_tps, decode_tps = greedy_decode_cached(
        prompt_ids, weights, config, rope_cos, rope_sin,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    t_c = time.perf_counter() - t_c
    text_cached = tokenizer.decode(gen_cached)
    print(f"  Output : {repr(text_cached)}")
    print(f"  Tokens : {len(gen_cached)}   Time: {t_c:.1f}s")
    print(f"  Prefill: {prefill_tps:.2f} tok/s   Decode: {decode_tps:.2f} tok/s\n")

    # ── Proof of identity ──────────────────────────────────────────────────
    print("Checking identity (cached == uncached)...")
    if gen_nocache == gen_cached:
        print("  PASS — token sequences are identical\n")
    else:
        min_len = min(len(gen_nocache), len(gen_cached))
        first_diff = next(
            (i for i in range(min_len) if gen_nocache[i] != gen_cached[i]), min_len
        )
        print(f"  FAIL — first divergence at token {first_diff}")
        print(f"    no-cache: {gen_nocache[:first_diff+3]}")
        print(f"    cached  : {gen_cached[:first_diff+3]}")
        sys.exit(1)

    # ── Speedup ────────────────────────────────────────────────────────────
    speedup = decode_tps / tps_nocache if tps_nocache > 0 else float("inf")
    print(f"Speedup: {speedup:.1f}x  ({tps_nocache:.2f} -> {decode_tps:.2f} tok/s)")

    prompt_t  = len(prompt_ids)
    gen_t     = len(gen_cached)

    # Record both phases in the benchmark log
    bm = Benchmark("phase4-kvcache-decode")
    bm.record(prompt_t, gen_t, t_c, note=f"decode={decode_tps:.2f}t/s prefill={prefill_tps:.2f}t/s")
    bm.report()
    bm.save()

    Benchmark.compare_phases()


if __name__ == "__main__":
    main()
