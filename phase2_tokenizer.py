"""
Phase 2 — BPE tokenizer for Qwen2.5, implemented from scratch.

Pipeline (mirrors HuggingFace tokenizers library exactly):
  1. Match and extract special tokens (verbatim, no byte-encoding)
  2. NFC-normalize each non-special segment
  3. Regex-split each segment into pre-token chunks
  4. ByteLevel-encode each chunk (byte -> unicode char, GPT-2 style)
  5. Apply BPE merges to the character sequence
  6. Vocab lookup: token string -> integer ID

Why ByteLevel?  BPE was originally designed for text, but text has ~100k unicode
characters.  GPT-2 solved this by working at the byte level: every possible input
is first expressed as UTF-8 bytes (256 possible values), then each byte is mapped
to a printable unicode character.  That gives a closed 256-symbol alphabet before
any merges, so BPE can combine bytes into longer tokens without ever hitting an
unknown symbol.

The mapping (bytes_to_unicode) is:
  - Bytes 33-126 (printable ASCII ! .. ~)  -> themselves
  - Bytes 161-172, 174-255 (latin printable) -> themselves
  - Everything else (0-32, 127-160, 173)   -> chr(256), chr(257), ...

So byte 32 (space) -> chr(256+n) where n is its position in the "unmapped" list.
The HuggingFace ByteLevel pre-tokenizer uses 'Ġ' (U+0120 = 288) for space, which
matches: space is byte 32, and in the sorted unmapped list [0,1,...,32,...] it's
position 32, so chr(256+32) = chr(288) = 'Ġ'.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import regex  # Unicode-property-aware regex (\p{L}, \p{N})


# ── 1. Byte encoder (GPT-2 style) ─────────────────────────────────────────

def _build_byte_encoder() -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Returns (byte->char dict, char->byte dict).

    Bytes that are already printable map to themselves.
    The remaining 33 bytes map to chr(256), chr(257), ...
    """
    # Printable bytes that map to themselves
    bs: List[int] = (
        list(range(ord('!'), ord('~') + 1))   # 33-126
        + list(range(ord('¡'), ord('¬') + 1)) # 161-172
        + list(range(ord('®'), ord('ÿ') + 1)) # 174-255
    )
    cs: List[int] = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    enc = {b: chr(c) for b, c in zip(bs, cs)}
    dec = {v: k for k, v in enc.items()}
    return enc, dec


BYTE_ENC, BYTE_DEC = _build_byte_encoder()


def bytes_to_chars(b: bytes) -> str:
    """Encode a bytes object into the unicode-char representation."""
    return "".join(BYTE_ENC[byte] for byte in b)


def chars_to_bytes(s: str) -> bytes:
    """Decode a unicode-char representation back to bytes."""
    return bytes(BYTE_DEC[c] for c in s)


# ── 2. BPE merge engine ────────────────────────────────────────────────────

def _get_pairs(seq: List[str]) -> List[Tuple[str, str]]:
    """Return all adjacent pairs in seq (with duplicates preserved, order matters)."""
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


def bpe_merge(chars: List[str], ranks: Dict[Tuple[str, str], int]) -> List[str]:
    """
    Apply BPE merges to a list of single characters.

    Algorithm:
      - Find the pair (a, b) with the lowest rank in the merge table.
      - Scan left-to-right, replacing every occurrence of (a, b) with (a+b).
      - Repeat until no pair exists in the merge table.

    This is O(n * m) where n = initial length and m = number of merges applied,
    but in practice vocab tokens are short so it's fast.
    """
    if len(chars) == 1:
        return chars

    while True:
        pairs = _get_pairs(chars)
        if not pairs:
            break
        # Pick the pair with the smallest rank (highest priority)
        best = min(pairs, key=lambda p: ranks.get(p, 10**9))
        if best not in ranks:
            break  # No more mergeable pairs

        a, b = best
        merged: List[str] = []
        i = 0
        while i < len(chars):
            # Find next occurrence of `a` at or after i
            try:
                j = chars.index(a, i)
            except ValueError:
                merged.extend(chars[i:])
                break
            merged.extend(chars[i:j])
            i = j
            if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(chars[i])
                i += 1
        chars = merged

    return chars


# ── 3. The tokenizer class ─────────────────────────────────────────────────

# Qwen2.5 pre-tokenizer regex (from tokenizer.json pre_tokenizer.pretokenizers[0])
# Uses \p{L} (Unicode letters) and \p{N} (Unicode numbers) — requires `regex`.
_PRETOK_PATTERN = regex.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class BPETokenizer:
    """
    Qwen2.5 BPE tokenizer — no dependencies on transformers or tokenizers lib.

    Special tokens are matched literally before the normal pipeline runs.
    Non-special text goes through: NFC -> regex split -> byte-encode -> BPE -> vocab lookup.
    """

    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)

        with open(model_dir / "tokenizer.json", encoding="utf-8") as f:
            tok_json = json.load(f)

        # Vocab: token_string -> id
        self.vocab: Dict[str, int] = tok_json["model"]["vocab"]

        # Reverse vocab: id -> token_string
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        # Merge rules: (str_a, str_b) -> rank (lower = higher priority)
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        for rank, merge_str in enumerate(tok_json["model"]["merges"]):
            a, b = merge_str.split(" ", 1)
            self.merge_ranks[(a, b)] = rank

        # Special tokens (matched verbatim, bypassing the normal pipeline)
        # Load from tokenizer_config.json's added_tokens_decoder
        with open(model_dir / "tokenizer_config.json", encoding="utf-8") as f:
            tok_cfg = json.load(f)

        self.special_tokens: Dict[str, int] = {}
        for id_str, info in tok_cfg.get("added_tokens_decoder", {}).items():
            self.special_tokens[info["content"]] = int(id_str)

        # Sort longest-first so we match <|im_start|> before <|im|>
        self._special_sorted = sorted(
            self.special_tokens.keys(), key=len, reverse=True
        )
        # Regex that splits on any special token
        escaped = [re.escape(s) for s in self._special_sorted]
        self._special_re = re.compile("(" + "|".join(escaped) + ")")

        # Decoded special tokens (for decode())
        self.id_to_special: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}

    # ── Encode ──────────────────────────────────────────────────────────────

    def _encode_chunk(self, text: str) -> List[int]:
        """Encode one non-special text segment: NFC -> regex split -> bpe -> ids."""
        text = unicodedata.normalize("NFC", text)
        ids: List[int] = []
        for chunk in _PRETOK_PATTERN.findall(text):
            # ByteLevel: encode each UTF-8 byte as its unicode char
            char_seq: List[str] = list(bytes_to_chars(chunk.encode("utf-8")))
            # BPE merges
            merged = bpe_merge(char_seq, self.merge_ranks)
            # Vocab lookup
            for tok in merged:
                if tok in self.vocab:
                    ids.append(self.vocab[tok])
                else:
                    # Fallback: encode byte by byte (should be rare / impossible
                    # with a complete byte-level vocab)
                    for c in tok:
                        byte_tok = bytes_to_chars(c.encode("utf-8"))
                        ids.append(self.vocab[byte_tok])
        return ids

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Tokenize text into a list of integer IDs.

        add_special_tokens=True prepends <|im_start|>system ... but we skip that
        for now — the caller is responsible for formatting the chat template.

        Special tokens embedded in the text (e.g. '<|im_end|>') are always
        matched verbatim.
        """
        ids: List[int] = []
        # Split on special tokens first
        parts = self._special_re.split(text)
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_chunk(part))
        return ids

    # ── Decode ──────────────────────────────────────────────────────────────

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert a list of IDs back to a string.

        Special tokens are omitted by default (skip_special_tokens=True).
        Non-special tokens are byte-decoded: reverse byte encoder -> UTF-8.
        """
        char_buf = ""
        result_bytes = bytearray()

        for token_id in ids:
            if token_id in self.id_to_special:
                if char_buf:
                    result_bytes.extend(chars_to_bytes(char_buf))
                    char_buf = ""
                if not skip_special_tokens:
                    result_bytes.extend(
                        self.id_to_special[token_id].encode("utf-8")
                    )
            elif token_id in self.id_to_token:
                char_buf += self.id_to_token[token_id]
            # unknown ids: skip

        if char_buf:
            result_bytes.extend(chars_to_bytes(char_buf))

        return result_bytes.decode("utf-8", errors="replace")

    # ── Convenience ─────────────────────────────────────────────────────────

    def apply_chat_template(self, messages: List[Dict]) -> str:
        """
        Minimal chat template for Qwen2.5-Instruct (ChatML format):

          <|im_start|>system
          You are a helpful assistant.<|im_end|>
          <|im_start|>user
          {content}<|im_end|>
          <|im_start|>assistant
        """
        parts = []
        for msg in messages:
            parts.append(
                f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            )
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)
