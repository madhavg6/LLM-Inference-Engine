"""
Phase 5 — Sampling demo.

Shows the same prompt answered with different strategies:
greedy, nucleus sampling at multiple temperatures, and high-temperature chaos.

Run:
    python phase5_sampling.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import load_weights, precompute_rope
from phase2_tokenizer import BPETokenizer
from sampler import generate
from benchmark import Benchmark

MODEL_DIR = Path(__file__).parent / "model" / "qwen2.5-0.5b-instruct"

PROMPT = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Write a one-sentence description of the moon.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
MAX_NEW_TOKENS = 40

STRATEGIES = [
    dict(temperature=1.0, top_k=1,  top_p=1.0, seed=0,  name="greedy        (T=1.0, k=1)"),
    dict(temperature=0.7, top_k=50, top_p=0.9, seed=1,  name="focused        (T=0.7, k=50, p=0.9)"),
    dict(temperature=0.7, top_k=50, top_p=0.9, seed=2,  name="focused        (T=0.7, k=50, p=0.9, seed=2)"),
    dict(temperature=1.0, top_k=50, top_p=0.9, seed=1,  name="balanced       (T=1.0, k=50, p=0.9)"),
    dict(temperature=1.5, top_k=0,  top_p=0.9, seed=1,  name="creative       (T=1.5, p=0.9)"),
]


def main():
    print("Loading weights...")
    t0 = time.perf_counter()
    weights, config = load_weights(MODEL_DIR)
    rope_cos, rope_sin = precompute_rope(
        config.head_dim, config.max_position_embeddings, config.rope_theta
    )
    tokenizer = BPETokenizer(MODEL_DIR)
    print(f"  done in {time.perf_counter()-t0:.1f}s\n")

    prompt_ids = tokenizer.encode(PROMPT)
    print(f"Prompt: {repr(PROMPT[-50:])}")
    print(f"Tokens: {len(prompt_ids)}\n")
    print("=" * 70)

    bm = Benchmark("phase5-sampling")
    for cfg in STRATEGIES:
        name = cfg.pop("name")
        gen, tps = generate(
            prompt_ids, weights, config, rope_cos, rope_sin,
            max_new_tokens=MAX_NEW_TOKENS, **cfg,
        )
        text = tokenizer.decode(gen)
        print(f"\n[{name}]")
        print(f"  {repr(text)}")
        print(f"  ({len(gen)} tokens, {tps:.2f} tok/s)")
        bm.record(len(prompt_ids), len(gen), len(gen)/tps if tps>0 else 0, note=name)

    print("\n" + "=" * 70)
    bm.report()
    bm.save()


if __name__ == "__main__":
    main()
