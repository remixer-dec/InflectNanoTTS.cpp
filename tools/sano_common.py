#!/usr/bin/env python3
"""Small dependency-free contracts shared by the Sano host tools."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Iterable


SNL1_HEADER = struct.Struct("<4sHH32sIIQQQ8sHHI")
SNL1_BUCKET = struct.Struct("<QIIHHI")
SNL1_VERSION = 1
SNL1_HEADER_SIZE = SNL1_HEADER.size
SNL1_BUCKET_SIZE = SNL1_BUCKET.size
SNL1_MAGIC = b"SNL1"

# SNL2 uses compact u24 directories and a fixed 88-byte header.  Keep these
# contracts here so the compiler and dependency-free validator agree exactly.
SNL2_HEADER = struct.Struct("<4sHHI32sIIIIIIII8sBBH")
SNL2_VERSION = 1
SNL2_HEADER_SIZE = SNL2_HEADER.size
SNL2_MAGIC = b"SNL2"


def load_phoneme_config(path: Path) -> tuple[dict, dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("phoneme_type", "espeak") not in (None, "espeak"):
        raise ValueError(f"{path}: only phoneme_type='espeak' is supported")
    espeak = payload.get("espeak") or {}
    voice = espeak.get("voice")
    if not voice:
        raise ValueError(f"{path}: missing espeak.voice")
    raw_map = payload.get("phoneme_id_map")
    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError(f"{path}: missing/empty phoneme_id_map")
    id_map: dict[str, int] = {}
    for symbol, values in raw_map.items():
        if len(symbol) != 1 or not isinstance(values, list) or len(values) != 1:
            raise ValueError(
                f"{path}: expected one codepoint and one id for {symbol!r}"
            )
        value = int(values[0])
        if not 0 <= value <= 255:
            raise ValueError(f"{path}: phoneme id out of uint8 range: {value}")
        id_map[symbol] = value
    for symbol, expected in (("_", 0), ("^", 1), ("$", 2)):
        if id_map.get(symbol) != expected:
            raise ValueError(
                f"{path}: {symbol!r} must map to {expected}, got {id_map.get(symbol)}"
            )
    return payload, id_map


def phoneme_map_hash(id_map: dict[str, int]) -> bytes:
    """Hash the sorted, compact codepoint-to-id map used by SNL1."""
    canonical = json.dumps(
        {key: int(id_map[key]) for key in sorted(id_map)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).digest()


def phoneme_map_hash_hex(id_map: dict[str, int]) -> str:
    return phoneme_map_hash(id_map).hex()


def normalize_language(value: str) -> str:
    value = value.strip().lower().replace("_", "-")
    return value.split("-", 1)[0] if value else ""


def read_cmd2_words(path: Path) -> Iterable[str]:
    """Read words from the legacy CMD2 dictionary without decoding phones."""
    with path.open("rb") as source:
        marker = source.read(4)
        if len(marker) != 4:
            raise ValueError(f"{path}: truncated dictionary header")
        if marker == b"CMD2":
            raw_count = source.read(4)
            if len(raw_count) != 4:
                raise ValueError(f"{path}: truncated CMD2 count")
            (count,) = struct.unpack("<I", raw_count)
        else:
            (count,) = struct.unpack("<I", marker)
        for index in range(count):
            raw_size = source.read(1)
            if len(raw_size) != 1:
                raise ValueError(f"{path}: truncated word at entry {index}")
            word = source.read(raw_size[0])
            if len(word) != raw_size[0]:
                raise ValueError(f"{path}: truncated word at entry {index}")
            raw_phones = source.read(1)
            if len(raw_phones) != 1:
                raise ValueError(f"{path}: truncated phones at entry {index}")
            phone_bytes = source.read(raw_phones[0] * 2)
            if len(phone_bytes) != raw_phones[0] * 2:
                raise ValueError(f"{path}: truncated phones at entry {index}")
            yield word.decode("utf-8")


def read_word_file(path: Path) -> Iterable[str]:
    """Read a plain word list, tolerating CMU-style ``WORD  PHONES`` lines."""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";;;"):
            continue
        if "  " in line:
            line = line.split(None, 1)[0]
        yield line
