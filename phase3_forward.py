"""
Phase 3 — Naive float32 forward pass with greedy decoding.

Run:
    python phase3_forward.py

Loads weights, tokenizes a prompt, runs the full forward pass for each new
token (no KV cache), and prints the generated text.  Also records tok/s in
the benchmark log.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import load_weights, precompute_rope, greedy_decode
from phase2_tokenizer import BPETokenizer
from benchmark import Benchmark, Timer

MODEL_DIR = Path(__file__).parent / "model" / "qwen2.5-0.5b-instruct"

PROMPT = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "What is 2 + 2?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def main():
    print("Loading tokenizer...")
    tokenizer = BPETokenizer(MODEL_DIR)

    print("Loading weights (BF16 -> FP32)...")
    t0 = time.perf_counter()
    weights, config = load_weights(MODEL_DIR)
    load_time = time.perf_counter() - t0
    print(f"  done in {load_time:.1f}s")

    print("Precomputing RoPE tables...")
    rope_cos, rope_sin = precompute_rope(
        config.head_dim,
        config.max_position_embeddings,
        config.rope_theta,
    )

    print(f"\nPrompt: {repr(PROMPT)}")
    prompt_ids = tokenizer.encode(PROMPT)
    print(f"Prompt tokens: {len(prompt_ids)}")

    print("\nGenerating (greedy, no cache)...\n")
    generated_ids, tps = greedy_decode(
        prompt_ids,
        weights,
        config,
        rope_cos,
        rope_sin,
        max_new_tokens=50,
    )

    generated_text = tokenizer.decode(generated_ids)
    print(f"Assistant: {generated_text}")
    print(f"\nGenerated {len(generated_ids)} tokens at {tps:.2f} tok/s")

    # Record in benchmark log
    bm = Benchmark("phase3-naive-fp32")
    bm.record(
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated_ids),
        elapsed_sec=len(generated_ids) / tps if tps > 0 else 0,
        note=f"load={load_time:.1f}s",
    )
    bm.report()
    bm.save()


if __name__ == "__main__":
    main()
