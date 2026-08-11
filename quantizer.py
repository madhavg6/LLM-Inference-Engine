"""
Phase 6 — Post-training quantization: INT8 and INT4.

Design
------
Both quantizers produce a dequantized float32 weight dict that the existing
forward() function accepts without modification.  This is correct-by-design:
the accuracy loss comes from rounding during quantize→dequantize, not from
changing how matmul works.

Which weights are quantized
---------------------------
Only 2-D weight matrices in the attention and MLP blocks:
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

Excluded (kept at FP32):
  embed_tokens.weight  — embedding table, accessed by lookup not matmul
  *.norm.weight        — 1-D scale vectors, tiny
  *.bias               — 1-D bias vectors, tiny

INT8 scheme: symmetric per-channel (per output row)
  scale[i] = max(|W[i,:]|) / 127
  W_q[i,j] = round(W[i,j] / scale[i])  clipped to [-128, 127]
  W_deq[i,j] = W_q[i,j] * scale[i]
  Max error per element: scale[i] / 2  (half an LSB)

INT4 scheme: symmetric per-group (group_size=128 elements per group)
  Groups span the flattened weight matrix.  Each group of 128 contiguous
  values gets its own scale:
    scale[g] = max(|W_flat[g*128 : (g+1)*128]|) / 7
    W_q_flat[i] = round(W_flat[i] / scale[g(i)])  clipped to [-7, 7]
  Storage: packed as unsigned nibbles in [0, 14] with offset = 7.
    stored_nibble = W_q + 7      (maps -7..7 → 0..14)
    Two nibbles per byte: byte = (nibble_even << 4) | nibble_odd
  Max error per element: scale[g] / 2
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# ── Which tensors to quantize ───────────────────────────────────────────────

def _should_quantize(key: str, tensor: np.ndarray) -> bool:
    return (
        tensor.ndim == 2
        and key.endswith(".weight")
        and "embed_tokens" not in key
        and "norm" not in key      # catches input_layernorm, post_attention_layernorm, model.norm
    )


# ══════════════════════════════════════════════════════════════════════════════
# INT8
# ══════════════════════════════════════════════════════════════════════════════

def quantize_int8(
    weights: Dict[str, np.ndarray],
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """
    Quantize all eligible weight matrices to INT8 (per-channel absmax).

    Returns:
      (quantized_store, dequantized_weights)
      quantized_store  : internal representation (data + scale per tensor)
      dequantized_weights : float32 dict, ready to pass to forward()
    """
    store: Dict = {}
    deq: Dict[str, np.ndarray] = {}

    for key, w in weights.items():
        if not _should_quantize(key, w):
            deq[key] = w
            continue

        # Per-channel scale: one scale per output row
        scale = np.max(np.abs(w), axis=1, keepdims=True) / 127.0   # [out, 1]
        scale = np.where(scale == 0.0, 1.0, scale)                  # avoid ÷0

        w_q = np.round(w / scale).clip(-128, 127).astype(np.int8)   # [out, in]
        w_deq = w_q.astype(np.float32) * scale                      # [out, in]

        store[key] = {"data": w_q, "scale": scale.squeeze(1), "shape": w.shape}
        deq[key] = w_deq

    return store, deq


# ══════════════════════════════════════════════════════════════════════════════
# INT4
# ══════════════════════════════════════════════════════════════════════════════

_INT4_MAX  = 7        # symmetric range [-7, 7]
_INT4_OFS  = 7        # offset encoding: stored = q + 7 → [0, 14]
_GROUP_SZ  = 128      # elements per quantization group


def _pack_nibbles(values: np.ndarray) -> np.ndarray:
    """
    Pack int8 values in [-7, 7] into uint8 nibbles (2 per byte).

    Encoding: stored_nibble = value + 7  (maps -7..7 → 0..14).
    Pairs of stored nibbles are packed as:  byte = (nibble_even << 4) | nibble_odd.

    Input length must be even (caller pads if needed).
    """
    u = (values + _INT4_OFS).astype(np.uint8)   # [0, 14]
    return (u[::2] << 4) | u[1::2]               # pack pairs → uint8


def _unpack_nibbles(packed: np.ndarray, n: int) -> np.ndarray:
    """
    Unpack uint8 nibbles back to int8 values in [-7, 7].

    `n` is the original number of elements (before any padding).
    """
    high = (packed >> 4).astype(np.int16) - _INT4_OFS   # even positions
    low  = (packed & 0xF).astype(np.int16) - _INT4_OFS  # odd positions
    interleaved = np.empty(len(packed) * 2, dtype=np.int16)
    interleaved[::2]  = high
    interleaved[1::2] = low
    return interleaved[:n].astype(np.int8)


def _quantize_tensor_int4(
    w: np.ndarray, group_size: int = _GROUP_SZ
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize a single 2-D weight matrix to INT4 with per-group absmax.

    Returns (packed_uint8, scales_float32).
    packed_uint8 : flat array, 2 values per byte
    scales_float32 : [n_groups], one scale per group of `group_size` elements
    """
    flat = w.flatten().astype(np.float32)   # [N]
    N = len(flat)

    # Pad to a multiple of group_size
    pad = (-N) % group_size
    if pad:
        flat_padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    else:
        flat_padded = flat

    groups = flat_padded.reshape(-1, group_size)      # [n_groups, group_size]
    scales = np.max(np.abs(groups), axis=1) / _INT4_MAX  # [n_groups]
    scales = np.where(scales == 0.0, 1.0, scales)

    q_groups = np.round(groups / scales[:, None]).clip(-_INT4_MAX, _INT4_MAX)
    q_flat = q_groups.flatten().astype(np.int8)[:N]    # remove padding

    # Pack pairs into bytes; pad to even length if needed
    if N % 2:
        q_flat = np.append(q_flat, np.int8(0))
    packed = _pack_nibbles(q_flat)

    return packed, scales


def _dequantize_tensor_int4(
    packed: np.ndarray,
    scales: np.ndarray,
    original_shape: Tuple,
    group_size: int = _GROUP_SZ,
) -> np.ndarray:
    """Dequantize INT4 packed data back to float32."""
    N = 1
    for d in original_shape:
        N *= d

    q_flat = _unpack_nibbles(packed, N)   # int8, length N

    # Assign each element to its group
    pad = (-N) % group_size
    if pad:
        q_flat_padded = np.concatenate([q_flat, np.zeros(pad, dtype=np.int8)])
    else:
        q_flat_padded = q_flat

    groups = q_flat_padded.reshape(-1, group_size).astype(np.float32)
    deq_flat = (groups * scales[:, None]).flatten()[:N]
    return deq_flat.reshape(original_shape)


def quantize_int4(
    weights: Dict[str, np.ndarray],
    group_size: int = _GROUP_SZ,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """
    Quantize all eligible weight matrices to INT4 (per-group absmax, packed).

    Returns:
      (quantized_store, dequantized_weights)
    """
    store: Dict = {}
    deq: Dict[str, np.ndarray] = {}

    for key, w in weights.items():
        if not _should_quantize(key, w):
            deq[key] = w
            continue

        packed, scales = _quantize_tensor_int4(w, group_size)
        w_deq = _dequantize_tensor_int4(packed, scales, w.shape, group_size)

        store[key] = {
            "data": packed, "scale": scales,
            "shape": w.shape, "group_size": group_size,
        }
        deq[key] = w_deq

    return store, deq


# ══════════════════════════════════════════════════════════════════════════════
# Compression reporting
# ══════════════════════════════════════════════════════════════════════════════

def _tensor_bytes_fp32(w: np.ndarray) -> int:
    return w.nbytes  # dtype is float32 = 4 bytes per element


def _store_bytes_int8(store_entry: dict) -> int:
    return store_entry["data"].nbytes + store_entry["scale"].nbytes * 4


def _store_bytes_int4(store_entry: dict) -> int:
    return store_entry["data"].nbytes + store_entry["scale"].nbytes * 4


def compression_report(
    weights_fp32: Dict[str, np.ndarray],
    store_q: Dict,
    label: str,
) -> None:
    fp32_total = 0
    q_total    = 0
    n_quantized = 0

    for key, w in weights_fp32.items():
        fp32_bytes = _tensor_bytes_fp32(w)
        fp32_total += fp32_bytes
        if key in store_q:
            entry = store_q[key]
            q_bytes = entry["data"].nbytes + entry["scale"].nbytes * 4
            q_total += q_bytes
            n_quantized += 1
        else:
            q_total += fp32_bytes

    ratio = fp32_total / q_total if q_total else float("inf")
    print(f"\n  {label}")
    print(f"    FP32 size    : {fp32_total / 1024**2:.1f} MB")
    print(f"    Quantized    : {q_total  / 1024**2:.1f} MB  ({ratio:.2f}x compression)")
    print(f"    Tensors quantized: {n_quantized}")


# ══════════════════════════════════════════════════════════════════════════════
# Perplexity
# ══════════════════════════════════════════════════════════════════════════════

def compute_perplexity(
    text_tokens: list,
    weights: Dict[str, np.ndarray],
    config,
    rope_cos: np.ndarray,
    rope_sin: np.ndarray,
    context_len: int = 256,
    stride: int = 128,
) -> float:
    """
    Sliding-window perplexity over `text_tokens`.

    For each window of `context_len` tokens:
      - Run the full forward pass to get logits for all positions.
      - Evaluate negative log-likelihood on the last `stride` positions
        (the non-overlapping portion of this window).

    Using a stride < context_len means each token gets evaluated with
    a full `context_len`-token context (except the first window).

    perplexity = exp( -1/N * Σ log P(token_i | token_{0..i-1}) )
    """
    # Import here to avoid circular import at module level
    from model import forward

    tokens = list(text_tokens)
    T = len(tokens)
    if T < 2:
        return float("inf")

    total_nll = 0.0
    total_n   = 0

    for start in range(0, T - 1, stride):
        end = min(start + context_len, T)
        ctx = np.array(tokens[start:end], dtype=np.int32)
        if len(ctx) < 2:
            break

        # Full forward pass: logits for ALL positions
        all_logits = forward(ctx, weights, config, rope_cos, rope_sin,
                             return_all_logits=True)  # [len(ctx), vocab]

        # Evaluate positions in the non-overlapping stride
        # (skip the first `context_len - stride` positions to avoid double-counting)
        eval_start = max(0, len(ctx) - stride) if start > 0 else 0

        # For each evaluation position i, the target is ctx[i+1]
        for i in range(eval_start, len(ctx) - 1):
            target = int(ctx[i + 1])
            logits_i = all_logits[i]

            # Numerically stable log-softmax
            logits_shifted = logits_i - logits_i.max()
            log_z = np.log(np.sum(np.exp(logits_shifted)))
            log_p_target = logits_shifted[target] - log_z

            total_nll -= log_p_target
            total_n   += 1

        if end >= T:
            break

    if total_n == 0:
        return float("inf")
    return float(np.exp(total_nll / total_n))
