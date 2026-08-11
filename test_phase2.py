"""
Phase 2 test — exact match against the reference tokenizer on 10,000 strings.

Reference: tokenizers library (Rust-based HuggingFace backend, NOT transformers).
           Loaded directly from tokenizer.json so it uses the exact same weights.

Test corpus covers:
  - Plain English prose
  - Code (Python, C, JSON)
  - Numbers and arithmetic
  - Unicode: CJK, emoji, Arabic, Devanagari, mixed scripts
  - Whitespace edge cases (leading space, tabs, newlines, mixed)
  - Special tokens embedded in text
  - Very short strings (1 char, empty — though empty typically returns [])
  - Very long strings (512+ chars)
  - Repeated characters
  - Byte sequences that would be unusual in UTF-8

Run:
    python test_phase2.py
"""

import random
import string
import sys
import unicodedata
from pathlib import Path
from typing import List

# Reference tokenizer (Rust, not transformers)
from tokenizers import Tokenizer as RefTokenizer

# Our implementation
sys.path.insert(0, str(Path(__file__).parent))
from phase2_tokenizer import BPETokenizer

MODEL_DIR = Path(__file__).parent / "model" / "qwen2.5-0.5b-instruct"

# ── Test string generators ─────────────────────────────────────────────────

def _prose_samples(n: int, rng: random.Random) -> List[str]:
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world!",
        "This is a test of the emergency broadcast system.",
        "NumPy is a fundamental package for scientific computing.",
        "Transformers are attention-based sequence models.",
        "The capital of France is Paris.",
        "    leading spaces matter   ",
        "\ttabbed\tcontent\there",
        "line one\nline two\nline three",
        "mixed\r\nline\r\nendings",
        "it's don't can't won't shouldn't",
        "Dr. Smith went to Washington, D.C.",
        "e.g., i.e., etc.",
        "42",
        "3.14159",
        "1,000,000",
        "1e-6",
        "-273.15",
        "0xFF",
        "0b1010",
    ]
    result = []
    for _ in range(n):
        base = rng.choice(sentences)
        # Occasionally concatenate two
        if rng.random() < 0.3:
            base = base + " " + rng.choice(sentences)
        result.append(base)
    return result


def _code_samples(n: int, rng: random.Random) -> List[str]:
    snippets = [
        "def foo(x):\n    return x * 2",
        "for i in range(10):\n    print(i)",
        "import numpy as np\nx = np.array([1, 2, 3])",
        "int main() { return 0; }",
        '#include <stdio.h>\nint main() {\n    printf("hello\\n");\n}',
        '{"key": "value", "num": 42}',
        "SELECT * FROM users WHERE id = 1;",
        "git commit -m 'fix: handle edge case'",
        "https://example.com/path?q=hello+world&lang=en",
        "model.layers[0].self_attn.q_proj.weight",
        "x = y if y > 0 else -y",
        "lambda x: x**2 + 2*x + 1",
        "    " * 4 + "deeply_nested_code()",
        "# This is a comment\nresult = compute()",
        "assert abs(a - b) < 1e-6, f'Expected {a}, got {b}'",
    ]
    result = []
    for _ in range(n):
        result.append(rng.choice(snippets))
    return result


def _unicode_samples(n: int, rng: random.Random) -> List[str]:
    samples = [
        # CJK
        "中文测试",              # 中文测试
        "日本語テスト",  # 日本語テスト
        "한국어",                     # 한국어
        # Arabic
        "مرحبا",         # مرحبا
        # Devanagari
        "नमस्ते",   # नमस्ते
        # Emoji
        "\U0001f600\U0001f604\U0001f970",
        "Hello \U0001f44b World",
        "score: 100\U0001f3c6",
        # Mixed
        "café",                              # café
        "naïve",                             # naïve
        "élève",                        # élève
        "Straße",                            # Straße
        "Zürich",                            # Zürich
        # Mathematical
        "α + β = γ",              # α + β = γ
        "∫ f(x) dx",                        # ∫ f(x) dx
        "∀ε > 0, ∃δ",        # ∀ε > 0, ∃δ
        # Control / edge
        "​",                                  # zero-width space
        "à",                                # a + combining grave -> à (unnormalized)
    ]
    result = []
    for _ in range(n):
        s = rng.choice(samples)
        if rng.random() < 0.3:
            s = s + " " + rng.choice(samples)
        result.append(s)
    return result


def _random_strings(n: int, rng: random.Random) -> List[str]:
    result = []
    for _ in range(n):
        length = rng.randint(1, 200)
        # Mix of printable ASCII + some unicode
        chars = []
        for _ in range(length):
            r = rng.random()
            if r < 0.7:
                chars.append(rng.choice(string.printable))
            elif r < 0.85:
                chars.append(chr(rng.randint(0x80, 0x7FF)))   # 2-byte UTF-8
            else:
                chars.append(chr(rng.randint(0x4E00, 0x9FFF))) # CJK
        result.append("".join(chars))
    return result


def _long_strings(n: int, rng: random.Random) -> List[str]:
    result = []
    words = "the quick brown fox jumps over lazy dog numpy array tensor attention head".split()
    for _ in range(n):
        length = rng.randint(100, 512)
        result.append(" ".join(rng.choice(words) for _ in range(length)))
    return result


def _special_token_strings() -> List[str]:
    return [
        "<|im_start|>user\nhello<|im_end|>",
        "<|endoftext|>",
        "before<|im_end|>after",
        "<|im_start|>assistant\nI am helpful<|im_end|>\n",
        "no special tokens here",
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>",
    ]


# ── Build the 10k corpus ───────────────────────────────────────────────────

def build_corpus(seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    corpus = []
    corpus.extend(_prose_samples(3000, rng))
    corpus.extend(_code_samples(2000, rng))
    corpus.extend(_unicode_samples(2000, rng))
    corpus.extend(_random_strings(2000, rng))
    corpus.extend(_long_strings(500, rng))
    corpus.extend(_special_token_strings())
    # Pad to exactly 10_000
    while len(corpus) < 10_000:
        corpus.append(rng.choice(corpus))
    return corpus[:10_000]


# ── Run the comparison ─────────────────────────────────────────────────────

def run_test():
    print("Loading reference tokenizer (tokenizers library)...")
    ref = RefTokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    # The reference tokenizer does NOT add special tokens in encode() by default.
    # We need to match behavior: no auto-prepend of BOS/EOS.
    ref.no_truncation()
    ref.no_padding()

    print("Loading our BPE tokenizer...")
    ours = BPETokenizer(MODEL_DIR)

    print("Building test corpus (10,000 strings)...")
    corpus = build_corpus(seed=42)

    print("Running comparison...\n")
    mismatches = []
    total = len(corpus)

    for i, text in enumerate(corpus):
        ref_ids  = ref.encode(text).ids
        our_ids  = ours.encode(text)

        if ref_ids != our_ids:
            mismatches.append({
                "index": i,
                "text":  repr(text[:80]),
                "ref":   ref_ids[:20],
                "ours":  our_ids[:20],
            })
            if len(mismatches) <= 5:
                print(f"MISMATCH #{len(mismatches)} (index {i})")
                print(f"  text : {repr(text[:80])}")
                print(f"  ref  : {ref_ids[:20]}")
                print(f"  ours : {our_ids[:20]}")
                print()

        if (i + 1) % 1000 == 0:
            status = "ok" if not mismatches else f"{len(mismatches)} mismatches so far"
            print(f"  [{i+1:>5}/{total}]  {status}")

    print()
    if not mismatches:
        print(f"PASS  — all {total} strings matched exactly.")
    else:
        pct = 100 * len(mismatches) / total
        print(f"FAIL  — {len(mismatches)}/{total} mismatches ({pct:.2f}%)")
        print(f"\nFirst mismatch details saved above.")
        if len(mismatches) > 5:
            print(f"({len(mismatches) - 5} additional mismatches not shown)")

    return len(mismatches) == 0


if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)
