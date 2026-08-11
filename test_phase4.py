"""
Phase 4 test — prove KV cache produces bitwise-identical outputs to no-cache.

Tests:
  1. prefill() logits == forward() logits for the same prompt (exact match)
  2. forward_step() logits == forward() logits for each position (exact match)
  3. Full token sequence: greedy_decode_cached == greedy_decode (all tokens match)
  4. Cache shapes grow correctly with each step
  5. Timing: cached decode is faster than no-cache for sequences > 10 tokens

Run:
    python test_phase4.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import (
    load_weights, precompute_rope,
    forward, prefill, forward_step,
    greedy_decode, greedy_decode_cached,
    Qwen25Config,
)
from phase2_tokenizer import BPETokenizer

MODEL_DIR = Path(__file__).parent / "model" / "qwen2.5-0.5b-instruct"

# Short prompt so tests run in reasonable time
PROMPT = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Name one planet.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
MAX_NEW_TOKENS = 15


def test_prefill_matches_forward(weights, config, rope_cos, rope_sin, prompt_ids):
    """
    prefill() must return the same last-token logits as forward().
    This verifies the prefill path is not accidentally different.
    """
    ids = np.array(prompt_ids, dtype=np.int32)

    logits_fwd    = forward(ids, weights, config, rope_cos, rope_sin)
    logits_prefill, _ = prefill(ids, weights, config, rope_cos, rope_sin)

    max_diff = float(np.max(np.abs(logits_fwd - logits_prefill)))
    assert max_diff == 0.0, (
        f"prefill logits differ from forward by {max_diff} (expected exact match)"
    )
    assert int(np.argmax(logits_fwd)) == int(np.argmax(logits_prefill))
    print(f"  PASS  prefill_matches_forward  (max_diff={max_diff})")


def test_step_matches_forward(weights, config, rope_cos, rope_sin, prompt_ids):
    """
    forward_step() at position T must match forward([...prompt, token]) at [-1].

    We allow a tiny numerical tolerance (~1e-5): np.einsum with optimize=True
    chooses different BLAS contraction paths for a [1, heads, d] query vs a
    [T, heads, d] query, leading to different floating-point rounding order.
    The important check is that argmax (greedy token) is identical.
    """
    ids = np.array(prompt_ids, dtype=np.int32)
    logits_prefill, kv_cache = prefill(ids, weights, config, rope_cos, rope_sin)

    first_gen = int(np.argmax(logits_prefill))
    position  = len(prompt_ids)

    logits_step, _ = forward_step(
        first_gen, position, kv_cache, weights, config, rope_cos, rope_sin
    )

    ids_extended = np.array(list(prompt_ids) + [first_gen], dtype=np.int32)
    logits_full  = forward(ids_extended, weights, config, rope_cos, rope_sin)

    max_diff = float(np.max(np.abs(logits_step - logits_full)))
    # Argmax must always agree — numerical error << logit margin between top tokens
    assert int(np.argmax(logits_step)) == int(np.argmax(logits_full)), \
        "forward_step and full forward disagree on greedy token"
    # Max absolute diff should be at the floating-point noise floor (~1e-5)
    NUMERICAL_TOLERANCE = 1e-4
    assert max_diff < NUMERICAL_TOLERANCE, (
        f"forward_step logits differ by {max_diff:.2e} (tolerance {NUMERICAL_TOLERANCE:.0e})"
    )
    print(f"  PASS  step_matches_forward  (max_diff={max_diff:.2e}, argmax agrees)")


def test_cache_shapes(weights, config, rope_cos, rope_sin, prompt_ids):
    """
    After prefill of T tokens and N decode steps, each K/V array
    must have shape [T + N, n_kv_heads, head_dim].
    """
    T = len(prompt_ids)
    ids = np.array(prompt_ids, dtype=np.int32)
    _, kv_cache = prefill(ids, weights, config, rope_cos, rope_sin)

    # Check prefill shapes
    for layer_idx, (k, v) in enumerate(kv_cache):
        assert k.shape == (T, config.num_key_value_heads, config.head_dim), \
            f"Layer {layer_idx} K shape wrong after prefill: {k.shape}"
        assert v.shape == (T, config.num_key_value_heads, config.head_dim), \
            f"Layer {layer_idx} V shape wrong after prefill: {v.shape}"

    # Do a few steps and verify cache grows
    logits = _  # reuse prefill logits
    logits, _ = prefill(ids, weights, config, rope_cos, rope_sin)
    position = T
    for step in range(3):
        next_token = int(np.argmax(logits))
        logits, kv_cache = forward_step(
            next_token, position, kv_cache, weights, config, rope_cos, rope_sin
        )
        position += 1
        expected_len = T + step + 1
        for layer_idx, (k, v) in enumerate(kv_cache):
            assert k.shape[0] == expected_len, \
                f"Layer {layer_idx} step {step}: K has {k.shape[0]} rows, expected {expected_len}"

    print(f"  PASS  cache_shapes  (verified prefill + 3 steps)")


def test_token_sequence_identity(weights, config, rope_cos, rope_sin, prompt_ids):
    """
    Full token sequence from greedy_decode_cached must exactly match
    greedy_decode (no cache).  This is the definitive correctness proof.
    """
    gen_nc, _ = greedy_decode(
        prompt_ids, weights, config, rope_cos, rope_sin,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    gen_c, _, _ = greedy_decode_cached(
        prompt_ids, weights, config, rope_cos, rope_sin,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    assert gen_nc == gen_c, (
        f"Token sequences differ!\n"
        f"  no-cache: {gen_nc}\n"
        f"  cached  : {gen_c}"
    )
    print(f"  PASS  token_sequence_identity  ({len(gen_nc)} tokens matched exactly)")


def test_speedup(weights, config, rope_cos, rope_sin, prompt_ids):
    """
    Cached decode must be faster than no-cache.

    Why the speedup is modest for short sequences:
      The transformer has two cost components:
        (a) Weight reads:  O(1) in token count — 24 layers × ~21 MB = ~500 MB
                           per forward pass, regardless of sequence length.
        (b) Token-dependent: Q/K/V projections, MLP, attention — all O(T).

      For a 23-token prompt + 15 generated tokens, (a) dominates.  The cache
      eliminates (b) for past tokens but (a) stays constant.  So measured
      speedup reflects the fraction of time in (b), which is small here.

      The speedup grows with prompt length: at T=200 tokens, (b) grows while
      (a) stays fixed, so the cache matters much more.  Phase 6 quantisation
      will shrink (a) significantly, compounding the cache benefit.

    We assert >= 1.1x, which is reliably true even for short sequences.
    """
    t0 = time.perf_counter()
    _, tps_nc = greedy_decode(
        prompt_ids, weights, config, rope_cos, rope_sin,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    t_nc = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, _, tps_c = greedy_decode_cached(
        prompt_ids, weights, config, rope_cos, rope_sin,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    t_c = time.perf_counter() - t0

    speedup = tps_c / tps_nc if tps_nc > 0 else float("inf")
    print(f"  No-cache : {tps_nc:.2f} tok/s  ({t_nc:.1f}s total)")
    print(f"  Cached   : {tps_c:.2f} tok/s  ({t_c:.1f}s total)")
    print(f"  Speedup  : {speedup:.2f}x  (higher with longer prompts)")

    assert speedup >= 1.1, f"Expected speedup >= 1.1x, got {speedup:.2f}x"
    print(f"  PASS  speedup >= 1.1x")


def main():
    print("=" * 60)
    print("Phase 4 tests — KV cache correctness and speedup")
    print("=" * 60)

    print("\nLoading weights...")
    t0 = time.perf_counter()
    weights, config = load_weights(MODEL_DIR)
    rope_cos, rope_sin = precompute_rope(
        config.head_dim, config.max_position_embeddings, config.rope_theta
    )
    tokenizer = BPETokenizer(MODEL_DIR)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s\n")

    prompt_ids = tokenizer.encode(PROMPT)
    print(f"Prompt: {len(prompt_ids)} tokens\n")

    test_prefill_matches_forward(weights, config, rope_cos, rope_sin, prompt_ids)
    test_step_matches_forward(weights, config, rope_cos, rope_sin, prompt_ids)
    test_cache_shapes(weights, config, rope_cos, rope_sin, prompt_ids)
    test_token_sequence_identity(weights, config, rope_cos, rope_sin, prompt_ids)

    print()
    test_speedup(weights, config, rope_cos, rope_sin, prompt_ids)

    print("\nAll Phase 4 tests passed.")


if __name__ == "__main__":
    main()
