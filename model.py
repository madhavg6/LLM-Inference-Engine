"""
Qwen2.5 forward pass — NumPy only, float32.
Phases 3 (no cache) and 4 (KV cache) live here together.

Every function is a direct translation of the math.  No framework magic.

Architecture (from config.json):
  hidden_size          = 896
  num_hidden_layers    = 24
  num_attention_heads  = 14   (Q heads)
  num_key_value_heads  = 2    (KV heads, GQA)
  head_dim             = 64   (= hidden_size / num_attention_heads)
  intermediate_size    = 4864 (MLP hidden width)
  vocab_size           = 151936
  rope_theta           = 1_000_000
  rms_norm_eps         = 1e-6
  tie_word_embeddings  = True  (lm_head shares embed_tokens weights)

Forward pass call sequence for one token sequence:
  x = embed_tokens[token_ids]          # [T, H]
  for each layer N:
    h = rms_norm(x)                    # pre-attention norm
    q, k, v = qkv_proj(h)             # linear projections (with bias for q, k, v)
    q, k = apply_rope(q, k, pos)      # rotary position embeddings
    k, v = repeat_kv(k, v)            # expand KV heads to match Q heads (GQA)
    attn = softmax(q @ k.T / sqrt(d)) @ v
    x = x + o_proj(attn)              # residual
    h = rms_norm(x)                   # post-attention norm
    x = x + swiglu_mlp(h)            # residual
  x = rms_norm(x)                     # final norm
  logits = x @ embed_tokens.T        # tied unembedding [T, vocab]
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from safetensors_parser import load_all_tensors


# ── Config ─────────────────────────────────────────────────────────────────

@dataclass
class Qwen25Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, path: Path) -> "Qwen25Config":
        with open(path) as f:
            c = json.load(f)
        return cls(
            hidden_size=c["hidden_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"],
            intermediate_size=c["intermediate_size"],
            vocab_size=c["vocab_size"],
            rope_theta=c["rope_theta"],
            rms_norm_eps=c["rms_norm_eps"],
            max_position_embeddings=c["max_position_embeddings"],
        )


# ── Weight loading ──────────────────────────────────────────────────────────

def load_weights(model_dir: Path) -> Tuple[Dict[str, np.ndarray], Qwen25Config]:
    """
    Load all safetensors weights into float32 numpy arrays.
    Returns (weights_dict, config).
    """
    model_dir = Path(model_dir)
    config = Qwen25Config.from_json(model_dir / "config.json")

    shard_paths = sorted(model_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    weights: Dict[str, np.ndarray] = {}
    for p in shard_paths:
        weights.update(load_all_tensors(p))

    return weights, config


# ── RMSNorm ────────────────────────────────────────────────────────────────
#
# RMSNorm(x, w) = x / sqrt(mean(x²) + eps) * w
#
# Unlike LayerNorm there is no mean subtraction — just scale by the RMS.
# This is ~15% faster to compute and works just as well in practice.
# Qwen2 uses eps=1e-6.

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # x: [..., hidden_size]
    rms = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


# ── RoPE (Rotary Position Embeddings) ──────────────────────────────────────
#
# RoPE encodes position by rotating query and key vectors in 2D planes.
# For a head of dimension D, we treat it as D/2 pairs of values.
# Pair i is rotated by angle:  θᵢ = position / (rope_theta ^ (2i / D))
#
# HuggingFace uses the "split-half" convention:
#   rotate_half(x) = concat([-x_second_half, x_first_half])
#   x_rotated = x * cos(θ) + rotate_half(x) * sin(θ)
#
# This is mathematically equivalent to the original paired-dimension rotation
# but with the cos/sin broadcast across the full head_dim after repeating:
#   cos = [cos(θ₀), ..., cos(θ_{D/2-1}), cos(θ₀), ..., cos(θ_{D/2-1})]
#   sin = [sin(θ₀), ..., sin(θ_{D/2-1}), sin(θ₀), ..., sin(θ_{D/2-1})]

def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (cos, sin) each of shape [max_seq_len, head_dim].
    Call once, then index by position for each forward pass.
    """
    # Frequencies: one per pair of head dimensions
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (np.arange(0, half, dtype=np.float32) / half)
    )  # [head_dim/2]

    # Outer product: position × frequency = angle
    positions = np.arange(max_seq_len, dtype=np.float32)   # [T]
    angles = np.outer(positions, inv_freq)                   # [T, head_dim/2]

    cos = np.cos(angles)   # [T, head_dim/2]
    sin = np.sin(angles)   # [T, head_dim/2]

    # Repeat to full head_dim (split-half convention)
    cos = np.concatenate([cos, cos], axis=-1)  # [T, head_dim]
    sin = np.concatenate([sin, sin], axis=-1)  # [T, head_dim]

    return cos.astype(np.float32), sin.astype(np.float32)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    """[-x_second_half, x_first_half] — the "rotation partner" of x."""
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rope(
    q: np.ndarray,          # [T, n_q_heads, head_dim]
    k: np.ndarray,          # [T, n_kv_heads, head_dim]
    cos: np.ndarray,        # [max_seq, head_dim]
    sin: np.ndarray,        # [max_seq, head_dim]
    positions: np.ndarray,  # [T]  — integer position indices
) -> Tuple[np.ndarray, np.ndarray]:
    c = cos[positions][:, np.newaxis, :]   # [T, 1, head_dim]
    s = sin[positions][:, np.newaxis, :]   # [T, 1, head_dim]
    q_rot = q * c + _rotate_half(q) * s
    k_rot = k * c + _rotate_half(k) * s
    return q_rot, k_rot


# ── Grouped Query Attention ─────────────────────────────────────────────────
#
# GQA uses fewer KV heads than Q heads to save memory and computation.
# Qwen2.5-0.5B: 14 Q heads, 2 KV heads → each KV head serves 7 Q heads.
#
# We expand K and V by repeating each KV head n_rep times before computing
# attention scores.  The expansion is:
#   K[kv_head] -> [K[kv_head], K[kv_head], ... (n_rep times)]
# so Q heads 0..6 attend to KV head 0, Q heads 7..13 attend to KV head 1.

def _repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """
    x: [T, n_kv_heads, head_dim]  →  [T, n_kv_heads * n_rep, head_dim]
    np.repeat repeats each element n_rep times along axis=1, which gives
    the correct grouping: head 0 repeated n_rep times, then head 1, etc.
    """
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=1)


def attention(
    q: np.ndarray,  # [T_q, n_q_heads, head_dim]
    k: np.ndarray,  # [T_kv, n_kv_heads, head_dim]
    v: np.ndarray,  # [T_kv, n_kv_heads, head_dim]
    mask: np.ndarray,  # [T_q, T_kv]  — additive mask (0 or -inf)
    n_rep: int,
) -> np.ndarray:       # [T_q, n_q_heads, head_dim]
    """
    Scaled dot-product attention with GQA expansion.

    Score(q, k) = q @ k.T / sqrt(head_dim)   [n_heads, T_q, T_kv]
    attn = softmax(Score + mask, dim=-1)
    output = attn @ v                          [n_heads, T_q, head_dim]
    """
    head_dim = q.shape[-1]
    scale = 1.0 / np.sqrt(head_dim)

    # Expand KV for GQA
    k = _repeat_kv(k, n_rep)  # [T_kv, n_q_heads, head_dim]
    v = _repeat_kv(v, n_rep)  # [T_kv, n_q_heads, head_dim]

    # Compute scores: [n_q_heads, T_q, T_kv]
    # einsum 'qhd,khd->hqk': for each head h, q[q,h,:] · k[k,h,:]
    scores = np.einsum("qhd,khd->hqk", q, k, optimize=True) * scale

    # Add causal mask (mask[q, k] = -inf when k > q, 0 otherwise)
    scores = scores + mask[np.newaxis, :, :]  # broadcast over heads

    # Numerically stable softmax: subtract max before exp
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)  # [n_heads, T_q, T_kv]

    # Weighted sum of values: [T_q, n_q_heads, head_dim]
    # einsum 'hqk,khd->qhd': for each head h, weighted sum of v over k positions
    out = np.einsum("hqk,khd->qhd", weights, v, optimize=True)

    return out


# ── SwiGLU MLP ────────────────────────────────────────────────────────────
#
# SwiGLU(x) = silu(gate_proj(x)) ⊙ up_proj(x)
# output    = down_proj(SwiGLU(x))
#
# silu(x) = x * sigmoid(x) = x / (1 + e^{-x})
# The gate branch controls which elements of the up-projection pass through.
# This gating mechanism outperforms ReLU and GELU in practice.
#
# Note on numerical stability: for x << 0, silu(x) → 0 naturally because
# sigmoid(x) → 0.  NumPy handles float32 overflow (exp → inf) gracefully here
# because inf in the denominator collapses the sigmoid to zero.

def _silu(x: np.ndarray) -> np.ndarray:
    # exp(-x) overflows to inf for x < -88 in float32, but x/inf = 0 correctly.
    # Suppress the overflow RuntimeWarning — it's expected and handled.
    with np.errstate(over="ignore"):
        return x / (1.0 + np.exp(-x))


def swiglu_mlp(
    x: np.ndarray,         # [T, hidden_size]
    gate_w: np.ndarray,    # [intermediate_size, hidden_size]
    up_w: np.ndarray,      # [intermediate_size, hidden_size]
    down_w: np.ndarray,    # [hidden_size, intermediate_size]
) -> np.ndarray:           # [T, hidden_size]
    gate = x @ gate_w.T    # [T, intermediate_size]
    up   = x @ up_w.T      # [T, intermediate_size]
    hidden = _silu(gate) * up
    return hidden @ down_w.T  # [T, hidden_size]


# ── Full forward pass ───────────────────────────────────────────────────────

def forward(
    token_ids: np.ndarray,              # [T]  integer token IDs
    weights: Dict[str, np.ndarray],
    config: Qwen25Config,
    rope_cos: np.ndarray,               # precomputed [max_seq, head_dim]
    rope_sin: np.ndarray,
    return_all_logits: bool = False,
) -> np.ndarray:
    """
    Full transformer forward pass.

    Returns logits of shape:
      [vocab_size]           if return_all_logits=False  (last token only)
      [T, vocab_size]        if return_all_logits=True
    """
    T = len(token_ids)
    positions = np.arange(T)

    # Token embeddings: [T, hidden_size]
    x = weights["model.embed_tokens.weight"][token_ids].astype(np.float32)

    # Causal mask: [T, T], 0 on and below diagonal, -inf above
    # mask[i, j] = -inf means token i cannot attend to position j (future)
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)

    for layer_idx in range(config.num_hidden_layers):
        p = f"model.layers.{layer_idx}"

        # ── Self-attention ─────────────────────────────────────────────────
        h = rms_norm(x, weights[f"{p}.input_layernorm.weight"], config.rms_norm_eps)

        # QKV projections (Qwen2 has biases on q, k, v — unusual but true)
        q = h @ weights[f"{p}.self_attn.q_proj.weight"].T + weights[f"{p}.self_attn.q_proj.bias"]
        k = h @ weights[f"{p}.self_attn.k_proj.weight"].T + weights[f"{p}.self_attn.k_proj.bias"]
        v = h @ weights[f"{p}.self_attn.v_proj.weight"].T + weights[f"{p}.self_attn.v_proj.bias"]

        # Reshape to [T, n_heads, head_dim]
        q = q.reshape(T, config.num_attention_heads, config.head_dim)
        k = k.reshape(T, config.num_key_value_heads, config.head_dim)
        v = v.reshape(T, config.num_key_value_heads, config.head_dim)

        # Rotary embeddings
        q, k = apply_rope(q, k, rope_cos, rope_sin, positions)

        # Attention (includes GQA expansion internally)
        attn_out = attention(q, k, v, mask, config.num_kv_groups)

        # Merge heads and project: [T, hidden_size]
        attn_out = attn_out.reshape(T, config.hidden_size)
        attn_out = attn_out @ weights[f"{p}.self_attn.o_proj.weight"].T

        x = x + attn_out  # residual

        # ── MLP ────────────────────────────────────────────────────────────
        h = rms_norm(x, weights[f"{p}.post_attention_layernorm.weight"], config.rms_norm_eps)
        mlp_out = swiglu_mlp(
            h,
            weights[f"{p}.mlp.gate_proj.weight"],
            weights[f"{p}.mlp.up_proj.weight"],
            weights[f"{p}.mlp.down_proj.weight"],
        )
        x = x + mlp_out  # residual

    # Final norm
    x = rms_norm(x, weights["model.norm.weight"], config.rms_norm_eps)

    # Unembedding (weight-tied: same matrix as embed_tokens, transposed)
    # embed_tokens: [vocab, hidden]  →  x @ embed_tokens.T: [T, vocab]
    embed = weights["model.embed_tokens.weight"]
    logits = x @ embed.T  # [T, vocab_size]

    if return_all_logits:
        return logits
    return logits[-1]  # [vocab_size] — only the last token's logits matter for generation


# ── Greedy decoding ─────────────────────────────────────────────────────────

def greedy_decode(
    prompt_ids: list,
    weights: Dict[str, np.ndarray],
    config: Qwen25Config,
    rope_cos: np.ndarray,
    rope_sin: np.ndarray,
    max_new_tokens: int = 50,
    eos_token_id: int = 151645,  # <|im_end|>
) -> Tuple[list, float]:
    """
    Naive greedy decoding: no cache, full forward pass for every new token.

    Returns (generated_token_ids, tokens_per_second).
    This is O(T²) in sequence length — correct but slow.  Phase 4 adds KV cache.
    """
    tokens = list(prompt_ids)
    generated = []

    t_start = time.perf_counter()
    for _ in range(max_new_tokens):
        ids = np.array(tokens, dtype=np.int32)
        logits = forward(ids, weights, config, rope_cos, rope_sin)
        next_token = int(np.argmax(logits))
        tokens.append(next_token)
        generated.append(next_token)
        if next_token == eos_token_id:
            break
    elapsed = time.perf_counter() - t_start

    tps = len(generated) / elapsed if elapsed > 0 else 0.0
    return generated, tps


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — KV cache
# ══════════════════════════════════════════════════════════════════════════════
#
# Why the cache is correct (the causality argument):
#
#   The causal mask ensures position i attends only to positions 0..i.
#   Adding token at position T cannot alter the attention output of any
#   earlier position, so the residual stream and the K/V projections for
#   positions 0..T-1 are identical whether or not T is in the input.
#
#   Consequence: K and V for past positions can be computed once during the
#   prefill pass and reused forever.  Each new decode step only computes K/V
#   for ONE new token (one row per layer), then attends to the full cache.
#
# Cache layout:
#   kv_cache  : list[layer_idx]  →  (K_array, V_array)
#   K_array   : [T_cached, n_kv_heads, head_dim]  — stored after RoPE
#   V_array   : [T_cached, n_kv_heads, head_dim]  — V has no RoPE
#
# After prefill of a T-token prompt, each K/V array has T rows.
# After each decode step the arrays grow by one row (via np.concatenate).
# ══════════════════════════════════════════════════════════════════════════════


def _layer_forward(
    x: np.ndarray,         # [T, hidden_size]   — T=1 during decode steps
    layer_idx: int,
    weights: Dict[str, np.ndarray],
    config: Qwen25Config,
    rope_cos: np.ndarray,
    rope_sin: np.ndarray,
    positions: np.ndarray,  # [T] integer positions for RoPE
    kv_cache_entry: Optional[Tuple[np.ndarray, np.ndarray]],
    mask: np.ndarray,       # [T_q, T_kv]  additive causal mask
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Single transformer layer with optional KV cache.

    If kv_cache_entry is None (prefill, first call):
      — Compute K, V normally; return them as the new cache entry.
    If kv_cache_entry is (K_past, V_past):
      — Append new K, V to the cache before attending.

    Returns (x_out, (K_full, V_full)) where K_full/V_full include the new row.
    """
    T = x.shape[0]
    p = f"model.layers.{layer_idx}"

    # Pre-attention RMSNorm
    h = rms_norm(x, weights[f"{p}.input_layernorm.weight"], config.rms_norm_eps)

    # QKV projections — Qwen2 has biases on Q, K, V
    q = h @ weights[f"{p}.self_attn.q_proj.weight"].T + weights[f"{p}.self_attn.q_proj.bias"]
    k = h @ weights[f"{p}.self_attn.k_proj.weight"].T + weights[f"{p}.self_attn.k_proj.bias"]
    v = h @ weights[f"{p}.self_attn.v_proj.weight"].T + weights[f"{p}.self_attn.v_proj.bias"]

    q = q.reshape(T, config.num_attention_heads, config.head_dim)
    k = k.reshape(T, config.num_key_value_heads, config.head_dim)
    v = v.reshape(T, config.num_key_value_heads, config.head_dim)

    # RoPE: applied to Q and K, not V
    q, k = apply_rope(q, k, rope_cos, rope_sin, positions)

    # Extend the cache with new K and V rows
    if kv_cache_entry is not None:
        k_past, v_past = kv_cache_entry
        k_full = np.concatenate([k_past, k], axis=0)  # [T_past + T, n_kv, d]
        v_full = np.concatenate([v_past, v], axis=0)
    else:
        k_full, v_full = k, v  # prefill: cache is just the prompt's K, V

    # Attention: Q attends to the full K/V cache
    attn_out = attention(q, k_full, v_full, mask, config.num_kv_groups)

    # Merge heads, project back to hidden_size
    attn_out = attn_out.reshape(T, config.hidden_size)
    attn_out = attn_out @ weights[f"{p}.self_attn.o_proj.weight"].T
    x = x + attn_out  # residual

    # Post-attention RMSNorm + SwiGLU MLP
    h = rms_norm(x, weights[f"{p}.post_attention_layernorm.weight"], config.rms_norm_eps)
    mlp_out = swiglu_mlp(
        h,
        weights[f"{p}.mlp.gate_proj.weight"],
        weights[f"{p}.mlp.up_proj.weight"],
        weights[f"{p}.mlp.down_proj.weight"],
    )
    x = x + mlp_out  # residual

    return x, (k_full, v_full)


def prefill(
    token_ids: np.ndarray,              # [T] prompt token IDs
    weights: Dict[str, np.ndarray],
    config: Qwen25Config,
    rope_cos: np.ndarray,
    rope_sin: np.ndarray,
) -> Tuple[np.ndarray, list]:
    """
    Process the prompt in one shot and build the KV cache.

    Returns (logits_for_last_token [vocab_size], kv_cache).
    The kv_cache is a list of (K, V) arrays, one per layer.
    Each K/V has shape [T, n_kv_heads, head_dim].
    """
    T = len(token_ids)
    positions = np.arange(T, dtype=np.int64)
    x = weights["model.embed_tokens.weight"][token_ids].astype(np.float32)
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)

    kv_cache = []
    for layer_idx in range(config.num_hidden_layers):
        x, kv = _layer_forward(
            x, layer_idx, weights, config, rope_cos, rope_sin,
            positions, kv_cache_entry=None, mask=mask,
        )
        kv_cache.append(kv)

    x = rms_norm(x, weights["model.norm.weight"], config.rms_norm_eps)
    logits = x @ weights["model.embed_tokens.weight"].T  # [T, vocab]
    return logits[-1], kv_cache


def forward_step(
    token_id: int,
    position: int,
    kv_cache: list,
    weights: Dict[str, np.ndarray],
    config: Qwen25Config,
    rope_cos: np.ndarray,
    rope_sin: np.ndarray,
) -> Tuple[np.ndarray, list]:
    """
    Single-token decode step using the KV cache.

    The new token at `position` attends to all cached positions via the full cache.
    The cache is extended by one row per layer and returned.

    Returns (logits [vocab_size], updated_kv_cache).
    """
    pos_arr = np.array([position], dtype=np.int64)
    x = weights["model.embed_tokens.weight"][[token_id]].astype(np.float32)  # [1, H]

    # No causal masking: the new token is at the end, can see all past positions.
    # mask shape [T_q=1, T_kv] = all zeros (add-zero = no masking effect).
    T_kv = kv_cache[0][0].shape[0] + 1  # past + this new token
    mask = np.zeros((1, T_kv), dtype=np.float32)

    new_cache = []
    for layer_idx in range(config.num_hidden_layers):
        x, kv = _layer_forward(
            x, layer_idx, weights, config, rope_cos, rope_sin,
            pos_arr, kv_cache_entry=kv_cache[layer_idx], mask=mask,
        )
        new_cache.append(kv)

    x = rms_norm(x, weights["model.norm.weight"], config.rms_norm_eps)
    logits = x @ weights["model.embed_tokens.weight"].T  # [1, vocab]
    return logits[0], new_cache


def greedy_decode_cached(
    prompt_ids: list,
    weights: Dict[str, np.ndarray],
    config: Qwen25Config,
    rope_cos: np.ndarray,
    rope_sin: np.ndarray,
    max_new_tokens: int = 200,
    eos_token_id: int = 151645,
) -> Tuple[list, float, float]:
    """
    Greedy decoding with KV cache.

    Returns (generated_ids, prefill_tps, decode_tps).
    Separating prefill and decode throughput is important: they have very
    different computational profiles (prefill is compute-bound, decode is
    memory-bandwidth-bound once the model fits in cache).
    """
    # ── Prefill ──────────────────────────────────────────────────────────
    t_prefill_start = time.perf_counter()
    logits, kv_cache = prefill(
        np.array(prompt_ids, dtype=np.int32), weights, config, rope_cos, rope_sin
    )
    t_prefill = time.perf_counter() - t_prefill_start
    prefill_tps = len(prompt_ids) / t_prefill

    # ── Decode ───────────────────────────────────────────────────────────
    generated = []
    position = len(prompt_ids)

    t_decode_start = time.perf_counter()
    while len(generated) < max_new_tokens:
        next_token = int(np.argmax(logits))
        generated.append(next_token)
        if next_token == eos_token_id:
            break
        logits, kv_cache = forward_step(
            next_token, position, kv_cache, weights, config, rope_cos, rope_sin
        )
        position += 1
    t_decode = time.perf_counter() - t_decode_start

    decode_tps = len(generated) / t_decode if t_decode > 0 else 0.0
    return generated, prefill_tps, decode_tps
