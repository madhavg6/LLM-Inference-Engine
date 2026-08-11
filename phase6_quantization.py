"""
Phase 6 — Quantization and perplexity comparison.

Measures:
  - FP32  baseline perplexity  (weights as loaded)
  - INT8  perplexity           (dequantize before matmul)
  - INT4  perplexity           (dequantize before matmul)

Also reports compression ratio and memory savings for each level.

Run:
    python phase6_quantization.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import load_weights, precompute_rope
from phase2_tokenizer import BPETokenizer
from quantizer import (
    quantize_int8, quantize_int4,
    compute_perplexity, compression_report,
)
from benchmark import Benchmark

MODEL_DIR = Path(__file__).parent / "model" / "qwen2.5-0.5b-instruct"

# ── Evaluation corpus ───────────────────────────────────────────────────────
# ~350 words of English prose — consistent, public-domain text for perplexity.
# This corpus is fixed so FP32/INT8/INT4 numbers are directly comparable.
CORPUS = """
The transformer architecture introduced by Vaswani et al. in 2017 revolutionized
natural language processing. Unlike recurrent networks, which process tokens
sequentially, transformers process all tokens simultaneously using a mechanism
called self-attention. Each token attends to every other token, computing a
weighted sum of their representations. The weights are determined by the
compatibility between query and key vectors.

Modern large language models build on this foundation. They typically consist of
dozens of transformer layers, each refining the token representations. Residual
connections allow gradients to flow directly through layers, enabling training of
very deep networks. Layer normalization stabilizes activations and makes
optimization more reliable.

The feedforward network within each transformer layer applies a two-layer
nonlinear transformation to each token independently. This component is
responsible for most of the model's capacity to store factual knowledge. Recent
architectures use the SwiGLU activation function, which combines a gating
mechanism with the SiLU nonlinearity.

Grouped query attention reduces the number of key-value heads relative to query
heads, significantly reducing the memory required for the key-value cache during
generation. This technique maintains nearly the same quality as full multi-head
attention while enabling much faster inference on long sequences.

Quantization compresses model weights from 32-bit floating point to fewer bits.
Post-training quantization requires no additional training data or gradient
computation. For weights, symmetric absmax quantization scales each channel by
the maximum absolute value, dividing by the number of representable levels. This
preserves the most significant information while reducing storage requirements.
Eight-bit quantization typically incurs negligible accuracy loss. Four-bit
quantization shows larger degradation but remains useful for deployment on
memory-constrained hardware. Group quantization, which applies separate scales
to small groups of weights, significantly improves four-bit accuracy by adapting
to local weight distributions rather than using a single scale per channel.
"""

CONTEXT_LEN = 256   # tokens per evaluation window
STRIDE      = 128   # non-overlapping tokens evaluated per window


def run_perplexity(label, weights, config, rope_cos, rope_sin, tokenizer, tokens):
    print(f"\n  [{label}] computing perplexity on {len(tokens)} tokens ...")
    t0 = time.perf_counter()
    ppl = compute_perplexity(tokens, weights, config, rope_cos, rope_sin,
                             context_len=CONTEXT_LEN, stride=STRIDE)
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] perplexity = {ppl:.2f}  ({elapsed:.0f}s)")
    return ppl


def main():
    print("Loading tokenizer ...")
    tokenizer = BPETokenizer(MODEL_DIR)

    print("Loading weights (BF16 -> FP32) ...")
    t0 = time.perf_counter()
    weights, config = load_weights(MODEL_DIR)
    rope_cos, rope_sin = precompute_rope(
        config.head_dim, config.max_position_embeddings, config.rope_theta
    )
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    # Tokenize evaluation corpus
    corpus_tokens = tokenizer.encode(CORPUS)
    print(f"\nCorpus: {len(corpus_tokens)} tokens  "
          f"(context={CONTEXT_LEN}, stride={STRIDE})")

    # ── FP32 baseline ─────────────────────────────────────────────────────
    ppl_fp32 = run_perplexity("FP32 ", weights, config, rope_cos, rope_sin,
                              tokenizer, corpus_tokens)

    # ── INT8 ──────────────────────────────────────────────────────────────
    print("\n  Quantizing to INT8 ...")
    t0 = time.perf_counter()
    store_q8, weights_q8 = quantize_int8(weights)
    print(f"  done in {time.perf_counter()-t0:.2f}s")
    compression_report(weights, store_q8, "INT8 compression")

    ppl_q8 = run_perplexity("INT8 ", weights_q8, config, rope_cos, rope_sin,
                             tokenizer, corpus_tokens)

    # ── INT4 ──────────────────────────────────────────────────────────────
    print("\n  Quantizing to INT4 (group_size=128) ...")
    t0 = time.perf_counter()
    store_q4, weights_q4 = quantize_int4(weights, group_size=128)
    print(f"  done in {time.perf_counter()-t0:.2f}s")
    compression_report(weights, store_q4, "INT4 compression")

    ppl_q4 = run_perplexity("INT4 ", weights_q4, config, rope_cos, rope_sin,
                             tokenizer, corpus_tokens)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  {'Precision':<12} {'Perplexity':>12} {'Delta vs FP32':>15}")
    print(f"  {'-'*12} {'-'*12} {'-'*15}")
    for label, ppl in [("FP32", ppl_fp32), ("INT8", ppl_q8), ("INT4", ppl_q4)]:
        delta = f"+{ppl - ppl_fp32:.3f}" if ppl > ppl_fp32 else f"{ppl - ppl_fp32:.3f}"
        print(f"  {label:<12} {ppl:>12.2f} {delta:>15}")
    print("=" * 55)

    # Benchmark log
    bm = Benchmark("phase6-quantization")
    bm.record(len(corpus_tokens), 0, 0, note=f"ppl fp32={ppl_fp32:.2f} int8={ppl_q8:.2f} int4={ppl_q4:.2f}")
    bm.save()

    Benchmark.compare_phases()


if __name__ == "__main__":
    main()
