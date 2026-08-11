"""
Phase 1 — Download Qwen2.5-0.5B-Instruct and inspect the weight files.

Run:
    python phase1_inspect.py

What this does
--------------
1. Downloads the model from Hugging Face into ./model/qwen2.5-0.5b-instruct/
   using huggingface_hub (download only — no transformers).
2. Parses every .safetensors shard with our own parser (see safetensors_parser.py).
3. Prints a table: layer name | shape | dtype | param count.
4. Prints total parameter count and memory footprint at float32.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── 1. Make sure huggingface_hub is installed ──────────────────────────────
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub not found — install it with:")
    print("  pip install huggingface-hub")
    sys.exit(1)

from safetensors_parser import list_tensors, _parse_header


# ── 2. Model config ────────────────────────────────────────────────────────
MODEL_ID  = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_DIR = Path(__file__).parent / "model" / "qwen2.5-0.5b-instruct"

# Files we don't need for inference (keep download small)
IGNORE_PATTERNS = [
    "*.bin",          # old PyTorch shards — model is safetensors-only
    "flax_model*",
    "tf_model*",
    "rust_model*",
    "onnx/**",
    "original/**",
]


def download_model() -> None:
    if MODEL_DIR.exists() and any(MODEL_DIR.glob("*.safetensors")):
        print(f"Model already present at {MODEL_DIR}")
        return

    print(f"Downloading {MODEL_ID} -> {MODEL_DIR}")
    print("(This will take a few minutes on first run — ~1 GB of weights)\n")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"\nDownload complete.")


def inspect_weights() -> None:
    shard_paths = sorted(MODEL_DIR.glob("*.safetensors"))
    if not shard_paths:
        print("No .safetensors files found — did the download succeed?")
        sys.exit(1)

    print(f"\nFound {len(shard_paths)} shard(s):")
    for p in shard_paths:
        size_mb = p.stat().st_size / 1024**2
        print(f"  {p.name:<50}  {size_mb:7.1f} MB")

    # ── Collect all tensor records across shards ───────────────────────────
    all_records = []
    for p in shard_paths:
        recs = list_tensors(p)
        for r in recs:
            r["shard"] = p.name
        all_records.extend(recs)

    # Sort by layer name so the table reads top-to-bottom naturally
    all_records.sort(key=lambda r: r["name"])

    # ── Print table ────────────────────────────────────────────────────────
    print(f"\n{'Tensor name':<60} {'Shape':<30} {'dtype':<6} {'params':>12}")
    print(f"{'-'*60} {'-'*30} {'-'*6} {'-'*12}")

    total_params  = 0
    total_bytes   = 0

    for r in all_records:
        shape_str = "×".join(str(d) for d in r["shape"]) if r["shape"] else "(scalar)"
        print(
            f"{r['name']:<60} {shape_str:<30} {r['dtype']:<6} "
            f"{r['num_params']:>12,}"
        )
        total_params += r["num_params"]
        total_bytes  += r["nbytes"]

    # ── Summary ────────────────────────────────────────────────────────────
    fp32_bytes = total_params * 4
    print(f"\n{'='*110}")
    print(f"  Total tensors   : {len(all_records)}")
    print(f"  Total params    : {total_params:,}   ({total_params/1e6:.1f}M)")
    print(f"  On-disk size    : {total_bytes/1024**2:.1f} MB  (BF16 weights)")
    print(f"  FP32 footprint  : {fp32_bytes/1024**2:.1f} MB  (what we'll load into numpy)")
    print()

    # ── Architecture quick-summary ─────────────────────────────────────────
    # Count layers by inspecting the names
    layer_nums = set()
    for r in all_records:
        parts = r["name"].split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_nums.add(int(parts[i + 1]))
                except ValueError:
                    pass

    if layer_nums:
        print(f"  Transformer layers: {max(layer_nums) + 1}")

    # Print a quick key about what the layer names mean
    print("""
Layer name anatomy for Qwen2.5:
  model.embed_tokens.weight              — token embedding table [vocab, hidden]
  model.layers.N.self_attn.q_proj.weight — query projection for layer N
  model.layers.N.self_attn.k_proj.weight — key   projection for layer N  (GQA: fewer heads)
  model.layers.N.self_attn.v_proj.weight — value projection for layer N  (GQA: fewer heads)
  model.layers.N.self_attn.o_proj.weight — output projection for layer N
  model.layers.N.mlp.gate_proj.weight    — SwiGLU gate branch
  model.layers.N.mlp.up_proj.weight      — SwiGLU up branch
  model.layers.N.mlp.down_proj.weight    — SwiGLU down branch
  model.layers.N.input_layernorm.weight  — RMSNorm before self-attention
  model.layers.N.post_attention_layernorm.weight — RMSNorm before MLP
  model.norm.weight                      — final RMSNorm
  lm_head.weight                         — unembedding [vocab, hidden]  (may be tied to embed_tokens)
""")


def main() -> None:
    download_model()
    inspect_weights()


if __name__ == "__main__":
    main()
