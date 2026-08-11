"""
safetensors binary format
=========================

A safetensors file has exactly three regions:

  [0:8]        uint64 LE  — byte length of the JSON header that follows
  [8:8+N]      UTF-8 JSON — tensor metadata
  [8+N:]       raw bytes  — packed tensor data (no padding between tensors)

The JSON header is a flat dict.  Each key is a tensor name; the value is:

    {
        "dtype":        "F32" | "BF16" | "F16" | "I32" | "I64" | "U8" ...
        "shape":        [d0, d1, ...]
        "data_offsets": [start, end]   # byte range inside the DATA region
    }

There is one special key, "__metadata__", whose value is a string→string dict
(e.g. {"format": "pt"}).  Ignore it for tensor loading.

data_offsets are relative to the START of the data region (byte 8+N), NOT to
the start of the file.  So the absolute file offset of tensor t is:

    file_byte = 8 + header_len + t["data_offsets"][0]

dtype strings map to numpy dtypes like this:
    F32  → float32,  F16 → float16,  BF16 → bfloat16 (load as uint16, recast)
    I32  → int32,    I64 → int64,    U8  → uint8,    BOOL → bool
"""

import json
import mmap
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DTYPE_MAP = {
    "F32":  np.float32,
    "F16":  np.float16,
    "BF16": np.uint16,    # numpy has no bfloat16; load raw then view/cast
    "I32":  np.int32,
    "I64":  np.int64,
    "U8":   np.uint8,
    "I8":   np.int8,
    "BOOL": np.bool_,
}


def _parse_header(data: bytes) -> Tuple[int, Dict]:
    """
    Read the 8-byte length prefix, then parse the JSON header.
    Returns (header_byte_length, header_dict).
    """
    header_len = struct.unpack_from("<Q", data, 0)[0]   # little-endian uint64
    header_json = data[8 : 8 + header_len].decode("utf-8")
    header = json.loads(header_json)
    return header_len, header


def list_tensors(path: Path) -> List[Dict]:
    """
    Return a list of dicts — one per tensor — sorted by data offset.
    Each dict: {"name", "dtype", "shape", "num_params", "nbytes"}
    """
    raw = path.read_bytes()
    header_len, header = _parse_header(raw)

    records = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype_str = meta["dtype"]
        shape = meta["shape"]
        start, end = meta["data_offsets"]
        num_params = 1
        for d in shape:
            num_params *= d
        records.append({
            "name":       name,
            "dtype":      dtype_str,
            "shape":      shape,
            "num_params": num_params,
            "nbytes":     end - start,
            "_start":     start,
        })

    records.sort(key=lambda r: r["_start"])
    return records


def load_tensor(path: Path, name: str) -> np.ndarray:
    """
    Load a single tensor by name into a numpy array.
    For BF16 tensors the returned array is float32 (upcast on the fly).
    """
    raw = path.read_bytes()
    header_len, header = _parse_header(raw)

    meta = header[name]
    dtype_str  = meta["dtype"]
    shape      = meta["shape"]
    start, end = meta["data_offsets"]

    data_region_offset = 8 + header_len
    buf = raw[data_region_offset + start : data_region_offset + end]

    np_dtype = DTYPE_MAP[dtype_str]
    arr = np.frombuffer(buf, dtype=np_dtype).reshape(shape)

    if dtype_str == "BF16":
        # BF16 is identical to F32 with the low 16 bits zeroed.
        # Upcast: left-shift the 16-bit pattern into the high half of float32.
        arr = arr.astype(np.uint32) << 16
        arr = arr.view(np.float32)

    return arr


def load_all_tensors(path: Path, verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Load every tensor from a single .safetensors file.
    Uses mmap so the OS can page-cache efficiently for repeated access.
    """
    records = list_tensors(path)
    raw = path.read_bytes()
    header_len, header = _parse_header(raw)
    data_region_offset = 8 + header_len

    tensors = {}
    for rec in records:
        name      = rec["name"]
        meta      = header[name]
        dtype_str = meta["dtype"]
        shape     = meta["shape"]
        start, end = meta["data_offsets"]

        buf = raw[data_region_offset + start : data_region_offset + end]
        np_dtype = DTYPE_MAP[dtype_str]
        arr = np.frombuffer(buf, dtype=np_dtype).reshape(shape).copy()

        if dtype_str == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)

        tensors[name] = arr
        if verbose:
            print(f"  loaded {name:60s} {str(shape):30s} {dtype_str}")

    return tensors


def load_multi_file(paths: List[Path], verbose: bool = False) -> Dict[str, np.ndarray]:
    """Load tensors spread across multiple .safetensors shards."""
    tensors = {}
    for p in paths:
        tensors.update(load_all_tensors(p, verbose=verbose))
    return tensors
