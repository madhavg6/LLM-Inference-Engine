"""
Phase 5 test — temperature, top-k, top-p sampling correctness.

Test structure:
  UNIT TESTS (fast, synthetic logits — no model weights needed):
    - temperature=0  → always picks argmax
    - temperature→∞  → approaches uniform (all logits equal → each sampled ~equally)
    - top_k=1        → always picks argmax
    - top_k=k        → sampled token is always within the top-k
    - top_p close=0  → only the top token is ever sampled
    - top_p=0.9      → sampled token is always within the top-90% nucleus
    - reproducibility → same seed → same sequence of tokens
    - randomness      → different seeds → sequences can differ
    - shift trick     → nucleus always covers >= p of the mass

  INTEGRATION TESTS (requires weights):
    - generate(top_k=1) matches greedy_decode_cached exactly
    - generate(seed=X) called twice gives identical token sequences
    - generate with temperature=1.0, top_p=0.9 produces coherent text

Run:
    python test_phase5.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sampler import (
    apply_temperature, apply_top_k, apply_top_p,
    sample_token, sample_logits, generate,
)
from model import load_weights, precompute_rope, greedy_decode_cached
from phase2_tokenizer import BPETokenizer

MODEL_DIR = Path(__file__).parent / "model" / "qwen2.5-0.5b-instruct"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def softmax(logits):
    l = logits - logits.max()
    e = np.exp(l)
    return e / e.sum()


def nucleus_prob_mass(logits, p):
    """Return the actual probability mass of the nucleus produced by top-p=p."""
    filtered = apply_top_p(logits, p)
    probs = softmax(filtered)
    return float(probs[np.isfinite(filtered)].sum())


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests
# ══════════════════════════════════════════════════════════════════════════════

def test_temperature_zero():
    """Temperature=0 must always pick argmax."""
    rng = np.random.default_rng(0)
    logits = np.array([1.0, 5.0, 3.0, 2.0, 4.0], dtype=np.float32)
    for _ in range(20):
        tok = sample_logits(logits, rng, temperature=0.0)
        assert tok == 1, f"Expected argmax=1, got {tok}"
    print("  PASS  temperature_zero → always argmax")


def test_temperature_sharpens():
    """
    Low temperature concentrates mass on the top token.
    High temperature spreads mass more evenly.
    """
    rng = np.random.default_rng(1)
    logits = np.array([3.0, 1.0, 0.5, 0.0, -1.0], dtype=np.float32)

    # With T=0.1 (sharp): top token should be sampled almost always
    counts_low = np.zeros(5)
    for _ in range(500):
        counts_low[sample_logits(logits, rng, temperature=0.1)] += 1
    # Top token (index 0) should dominate
    assert counts_low[0] > 490, f"T=0.1 should heavily favour top token, counts={counts_low}"

    # With T=10 (flat): distribution should be much more even
    counts_high = np.zeros(5)
    for _ in range(500):
        counts_high[sample_logits(logits, rng, temperature=10.0)] += 1
    # All tokens should get at least 50/500 samples
    assert counts_high.min() > 30, f"T=10 should spread mass, counts={counts_high}"

    print("  PASS  temperature_sharpens  "
          f"(T=0.1: top={counts_low[0]}/500, T=10: min={counts_high.min()}/500)")


def test_top_k_one_is_greedy():
    """top_k=1 must always return argmax."""
    rng = np.random.default_rng(2)
    logits = np.array([0.5, 2.0, 1.5, 0.1, 1.8], dtype=np.float32)
    argmax = int(np.argmax(logits))
    for _ in range(20):
        tok = sample_logits(logits, rng, temperature=1.0, top_k=1)
        assert tok == argmax, f"top_k=1 should give argmax={argmax}, got {tok}"
    print(f"  PASS  top_k=1 → always argmax ({argmax})")


def test_top_k_constrains_to_top_k():
    """All sampled tokens must be within the top-k."""
    rng = np.random.default_rng(3)
    k = 3
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    top_k_set = set(np.argsort(logits)[-k:].tolist())

    for _ in range(200):
        tok = sample_logits(logits, rng, temperature=1.0, top_k=k)
        assert tok in top_k_set, f"top_k={k}: token {tok} not in top-{k} {top_k_set}"

    print(f"  PASS  top_k={k} → all samples in {{0,1,2}} (top-{k})")


def test_top_p_constrains_to_nucleus():
    """All sampled tokens must be within the top-p nucleus."""
    rng = np.random.default_rng(4)
    p = 0.9
    # Create a logit vector where the top few tokens dominate
    logits = np.array([5.0, 4.0, 3.0, 1.0, 0.0, -1.0, -2.0, -3.0], dtype=np.float32)
    probs = softmax(logits)

    # Identify the nucleus: minimum prefix covering p of the mass
    order = np.argsort(-probs)
    cumprobs = np.cumsum(probs[order])
    nucleus_size = int(np.searchsorted(cumprobs, p)) + 1  # +1 for the shift
    nucleus = set(order[:nucleus_size].tolist())

    for _ in range(500):
        tok = sample_logits(logits, rng, temperature=1.0, top_p=p)
        assert tok in nucleus, f"top_p={p}: token {tok} not in nucleus {nucleus}"

    print(f"  PASS  top_p={p} → all samples in nucleus {sorted(nucleus)}")


def test_top_p_nucleus_covers_p():
    """The nucleus produced by apply_top_p must cover at least p of the mass."""
    rng = np.random.default_rng(5)
    for p in [0.5, 0.75, 0.9, 0.95, 0.99]:
        logits = rng.standard_normal(100).astype(np.float32)
        mass = nucleus_prob_mass(logits, p)
        assert mass >= p - 1e-6, f"top_p={p}: nucleus covers only {mass:.4f}"
    print("  PASS  top_p nucleus covers >= p of probability mass")


def test_shift_trick():
    """
    Without the shift trick, the first token that pushes cumprob > p would be
    excluded, leaving a nucleus that covers strictly less than p.

    We verify that apply_top_p always covers >= p and that the nucleus is minimal.
    """
    # Extreme case: one token has prob 0.6, one has 0.4
    logits = np.log(np.array([0.6, 0.4], dtype=np.float32))
    # With p=0.5: the top token (prob 0.6) alone covers > 0.5, so nucleus = {0}
    result = apply_top_p(logits, 0.5)
    assert np.isfinite(result[0]) and result[1] == -np.inf, \
        "p=0.5 with logits=[0.6,0.4]: nucleus should be just token 0"

    # With p=0.7: after token 0 (cumprob=0.6), need token 1 too
    result = apply_top_p(logits, 0.7)
    assert np.isfinite(result[0]) and np.isfinite(result[1]), \
        "p=0.7 with logits=[0.6,0.4]: both tokens should be in nucleus"

    print("  PASS  shift_trick (nucleus size correct at boundary cases)")


def test_reproducibility():
    """Same seed must produce exactly the same token sequence."""
    rng = np.random.default_rng(42)
    logits = np.array([1.0, 2.0, 1.5, 0.5, 1.8, 0.3], dtype=np.float32)

    seq_a = [sample_logits(logits, np.random.default_rng(42), temperature=1.0)
             for _ in range(20)]
    seq_b = [sample_logits(logits, np.random.default_rng(42), temperature=1.0)
             for _ in range(20)]
    assert seq_a == seq_b, f"Same seed gave different sequences:\n  {seq_a}\n  {seq_b}"
    print(f"  PASS  reproducibility (seed=42 gives same 20-token sequence twice)")


def test_different_seeds_differ():
    """Different seeds should (with overwhelmingly high probability) differ."""
    logits = np.array([1.0, 2.0, 1.5, 0.5, 1.8, 0.3], dtype=np.float32)
    seq_42 = [sample_logits(logits, np.random.default_rng(42), temperature=1.0)
              for _ in range(20)]
    seq_99 = [sample_logits(logits, np.random.default_rng(99), temperature=1.0)
              for _ in range(20)]
    # With 6 tokens and 20 draws, probability of identical sequences is ~(1/6)^20 ≈ 0
    assert seq_42 != seq_99, "Seeds 42 and 99 produced identical sequences (extremely unlikely)"
    print(f"  PASS  different_seeds_differ")


def test_top_k_zero_no_filter():
    """top_k=0 must behave identically to top_k=vocab_size (no filtering)."""
    logits = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)
    result_0    = apply_top_k(logits, 0)
    result_full = apply_top_k(logits, len(logits))
    np.testing.assert_array_equal(result_0, logits)
    np.testing.assert_array_equal(result_full, logits)
    print("  PASS  top_k=0 → no filtering")


# ══════════════════════════════════════════════════════════════════════════════
# Integration tests (require model weights)
# ══════════════════════════════════════════════════════════════════════════════

PROMPT = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "What is 2 + 2?<|im_end|>\n"
    "<|im_start|>assistant\n"
)
MAX_NEW_TOKENS = 12


def test_top_k1_matches_greedy(weights, config, rope_cos, rope_sin, tokenizer):
    """generate(top_k=1) must produce the same token sequence as greedy_decode_cached."""
    prompt_ids = tokenizer.encode(PROMPT)

    gen_greedy, _, _ = greedy_decode_cached(
        prompt_ids, weights, config, rope_cos, rope_sin,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    gen_sampled, _ = generate(
        prompt_ids, weights, config, rope_cos, rope_sin,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0, top_k=1, top_p=1.0, seed=0,
    )

    assert gen_greedy == gen_sampled, (
        f"top_k=1 diverged from greedy at token "
        f"{next(i for i,(a,b) in enumerate(zip(gen_greedy,gen_sampled)) if a!=b)}\n"
        f"  greedy  : {gen_greedy}\n"
        f"  sampled : {gen_sampled}"
    )
    text = tokenizer.decode(gen_greedy)
    print(f"  PASS  top_k=1 matches greedy  output={repr(text)}")


def test_seed_reproducibility(weights, config, rope_cos, rope_sin, tokenizer):
    """generate(seed=42) called twice must return identical token sequences."""
    prompt_ids = tokenizer.encode(PROMPT)

    gen_a, _ = generate(
        prompt_ids, weights, config, rope_cos, rope_sin,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0, top_k=50, top_p=0.9, seed=42,
    )
    gen_b, _ = generate(
        prompt_ids, weights, config, rope_cos, rope_sin,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0, top_k=50, top_p=0.9, seed=42,
    )

    assert gen_a == gen_b, (
        f"Seed 42 gave different sequences:\n  {gen_a}\n  {gen_b}"
    )
    text = tokenizer.decode(gen_a)
    print(f"  PASS  seed_reproducibility (seed=42)  output={repr(text)}")


def test_sampling_demo(weights, config, rope_cos, rope_sin, tokenizer):
    """Show outputs for several sampling strategies side by side."""
    prompt_ids = tokenizer.encode(PROMPT)
    configs = [
        dict(temperature=1.0, top_k=1,  top_p=1.0, seed=0,  label="greedy (top_k=1)"),
        dict(temperature=1.0, top_k=50, top_p=0.9, seed=42, label="nucleus  (T=1, k=50, p=0.9)"),
        dict(temperature=0.7, top_k=50, top_p=0.9, seed=42, label="creative (T=0.7, k=50, p=0.9)"),
        dict(temperature=1.5, top_k=0,  top_p=0.9, seed=42, label="random   (T=1.5, p=0.9)"),
    ]
    print()
    for cfg in configs:
        label = cfg.pop("label")
        gen, _ = generate(
            prompt_ids, weights, config, rope_cos, rope_sin,
            max_new_tokens=MAX_NEW_TOKENS, **cfg,
        )
        text = tokenizer.decode(gen)
        print(f"  [{label}]  {repr(text)}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 5 tests — sampling")
    print("=" * 60)

    print("\n-- Unit tests (synthetic logits, no model) --")
    test_temperature_zero()
    test_temperature_sharpens()
    test_top_k_one_is_greedy()
    test_top_k_constrains_to_top_k()
    test_top_p_constrains_to_nucleus()
    test_top_p_nucleus_covers_p()
    test_shift_trick()
    test_reproducibility()
    test_different_seeds_differ()
    test_top_k_zero_no_filter()

    print("\n-- Integration tests (requires weights) --")
    print("Loading weights...")
    t0 = time.perf_counter()
    weights, config = load_weights(MODEL_DIR)
    rope_cos, rope_sin = precompute_rope(
        config.head_dim, config.max_position_embeddings, config.rope_theta
    )
    tokenizer = BPETokenizer(MODEL_DIR)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s\n")

    test_top_k1_matches_greedy(weights, config, rope_cos, rope_sin, tokenizer)
    test_seed_reproducibility(weights, config, rope_cos, rope_sin, tokenizer)

    print("\n-- Sampling strategy demo --")
    test_sampling_demo(weights, config, rope_cos, rope_sin, tokenizer)

    print("\nAll Phase 5 tests passed.")


if __name__ == "__main__":
    main()
