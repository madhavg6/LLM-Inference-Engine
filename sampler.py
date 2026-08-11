"""
Phase 5 — Temperature, top-k, and top-p sampling with a seedable RNG.

Sampling pipeline (applied in this order):
  1. apply_temperature(logits, T)   — rescale logit differences
  2. apply_top_k(logits, k)         — zero out all but the k highest
  3. apply_top_p(logits, p)         — zero out below-nucleus tokens
  4. sample_token(logits, rng)      — draw from the remaining distribution

Each function takes raw logits and returns logits (not probabilities), so
they compose cleanly: you can apply any subset in any order without losing
precision to an intermediate softmax.

Conventions:
  temperature = 1.0   → no change to logits
  temperature = 0.0   → greedy (return one-hot at argmax)
  top_k = 0           → no top-k filtering (keep all)
  top_p = 1.0         → no top-p filtering (keep all)
  seed = None         → non-reproducible (OS entropy)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from model import Qwen25Config, prefill, forward_step


# ── Temperature ─────────────────────────────────────────────────────────────

def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Scale logits by 1/temperature.

    Mathematically: softmax(logits / T).
      T < 1  → sharpens the distribution (more confident)
      T = 1  → unchanged
      T > 1  → flattens the distribution (more random)
      T → 0  → approaches argmax (greedy)

    T = 0 is handled as a special case: return a one-hot at argmax position.
    This avoids division-by-zero and keeps the dtype consistent.
    """
    if temperature <= 0.0:
        one_hot = np.full_like(logits, -np.inf)
        one_hot[int(np.argmax(logits))] = 0.0
        return one_hot
    if temperature == 1.0:
        return logits
    return logits / temperature


# ── Top-k ───────────────────────────────────────────────────────────────────

def apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """
    Keep only the k highest logits; set the rest to -inf.

    k = 0 or k >= vocab_size  → no filtering (return logits unchanged).
    Ties at the k-th position: all tied tokens are kept (same as HuggingFace).

    Implementation: np.partition finds the k-th largest in O(n) (quickselect),
    much faster than a full sort.
    """
    if k <= 0 or k >= len(logits):
        return logits
    # np.partition(-k) puts the k-th largest element at position -k
    threshold = np.partition(logits, -k)[-k]
    return np.where(logits >= threshold, logits, -np.inf)


# ── Top-p (nucleus) ─────────────────────────────────────────────────────────

def apply_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    """
    Keep the minimum set of tokens whose cumulative probability >= p.

    Algorithm:
      1. Sort tokens by logit value, descending.
      2. Compute their softmax probabilities (in sorted order).
      3. Compute cumulative probabilities.
      4. Mark for removal any token where the cumulative probability BEFORE
         that token already exceeds p (the "shift trick").
      5. Always keep at least the highest-probability token (never remove index 0).
      6. Set removed tokens to -inf.

    The "shift trick" in step 4 is critical: without it, we might exclude the
    token that first pushes cumprob over p, leaving us with a nucleus that
    covers only slightly less than p of the mass.

    p = 1.0  → no filtering (equivalent to returning logits unchanged).
    p = 0.0  → only the top token is kept (equivalent to greedy).
    """
    if p >= 1.0:
        return logits

    # Sort descending
    order = np.argsort(-logits)           # indices sorted by descending logit
    sorted_logits = logits[order]

    # Numerically stable softmax
    shifted = sorted_logits - sorted_logits[sorted_logits > -np.inf].max()
    exp_l = np.exp(shifted)
    exp_l[sorted_logits == -np.inf] = 0.0  # keep -inf tokens at prob 0
    probs = exp_l / exp_l.sum()

    # Cumulative probabilities in descending-probability order
    cumprobs = np.cumsum(probs)

    # The shift trick: a token at sorted position i should be REMOVED if
    # the cumulative probability of all HIGHER-ranked tokens already >= p.
    # That is: cumprobs[i-1] >= p  (the previous cumsum already covered p).
    to_remove = np.zeros(len(logits), dtype=bool)
    to_remove[1:] = cumprobs[:-1] >= p   # shift right by 1
    # Index 0 (highest probability token) is NEVER removed
    to_remove[0] = False

    # Apply mask: remove in original index space
    result = logits.copy()
    result[order[to_remove]] = -np.inf
    return result


# ── Categorical sampling ─────────────────────────────────────────────────────

def sample_token(logits: np.ndarray, rng: np.random.Generator) -> int:
    """
    Draw one token from the categorical distribution defined by logits.

    Steps:
      1. Subtract max for numerical stability (doesn't change distribution).
      2. Exponentiate.
      3. Normalize to get probabilities.
      4. Draw via rng.choice (uses the Alias method internally — O(1)).

    Handles the case where logits contains -inf values (prob = 0 for those).
    """
    finite_mask = np.isfinite(logits)
    if not np.any(finite_mask):
        raise ValueError("All logits are -inf; cannot sample.")

    # Stable softmax
    l = logits.copy()
    l[~finite_mask] = -np.inf
    l -= l[finite_mask].max()
    probs = np.exp(l)
    probs[~finite_mask] = 0.0
    probs /= probs.sum()

    return int(rng.choice(len(probs), p=probs))


# ── Full sampling pipeline ───────────────────────────────────────────────────

def sample_logits(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Apply the full temperature→top_k→top_p→sample pipeline."""
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    return sample_token(logits, rng)


# ── generate() — cached forward pass + sampling ────────────────────────────

def generate(
    prompt_ids: List[int],
    weights: Dict[str, np.ndarray],
    config: Qwen25Config,
    rope_cos: np.ndarray,
    rope_sin: np.ndarray,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    eos_token_id: int = 151645,
) -> Tuple[List[int], float]:
    """
    Generate tokens using the KV cache + configurable sampling.

    With temperature=1.0, top_k=1 (or temperature=0.0):
        Equivalent to greedy decoding.

    Returns (generated_token_ids, decode_tokens_per_second).
    """
    rng = np.random.default_rng(seed)

    # Prefill: process the prompt, build KV cache
    logits, kv_cache = prefill(
        np.array(prompt_ids, dtype=np.int32),
        weights, config, rope_cos, rope_sin,
    )

    generated: List[int] = []
    position = len(prompt_ids)

    t_start = time.perf_counter()
    while len(generated) < max_new_tokens:
        next_token = sample_logits(logits, rng, temperature, top_k, top_p)
        generated.append(next_token)
        if next_token == eos_token_id:
            break
        logits, kv_cache = forward_step(
            next_token, position, kv_cache, weights, config, rope_cos, rope_sin
        )
        position += 1
    elapsed = time.perf_counter() - t_start

    tps = len(generated) / elapsed if elapsed > 0 else 0.0
    return generated, tps
