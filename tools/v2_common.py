#!/usr/bin/env python3
"""Shared immutable format contracts for Inflect Nano v2 tooling."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Iterable


PAD = "_"
PUNCTUATION = ';:,.!?¡¿—…"«»“” '
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LETTERS_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴ"
    "øɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃ"
    "ˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)
SYMBOLS = [PAD, *PUNCTUATION, *LETTERS, *LETTERS_IPA]
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
SYMBOL_BYTES = "".join(SYMBOLS).encode("utf-8")
SYMBOL_HASH = hashlib.sha256(SYMBOL_BYTES).digest()
SYMBOL_HASH_HEX = SYMBOL_HASH.hex()

assert len(SYMBOLS) == 178
assert SYMBOL_HASH_HEX == "3e81afeec2d0906de3d7acf2214d32fbc066be8218d2edafe355255391ea92f7"

IVL2_HEADER = struct.Struct("<4sHH32sIIIQQ8s")
IVI2_HEADER = struct.Struct("<4sHH32sIII")
IVL2_VERSION = 1
IVL2_HEADER_SIZE = IVL2_HEADER.size
IVI2_HEADER_SIZE = IVI2_HEADER.size


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phoneme_ids(text: str) -> list[int]:
    try:
        return [SYMBOL_TO_ID[character] for character in text]
    except KeyError as error:
        codepoint = ord(error.args[0])
        raise ValueError(
            f"phoneme output contains symbol outside released table: "
            f"U+{codepoint:04X} {error.args[0]!r}"
        ) from error


def canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def common_prefix_bytes(left: bytes, right: bytes, limit: int = 255) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b or count == limit:
            break
        count += 1
    return count


def read_cmd2_words(path: Path) -> Iterable[str]:
    with path.open("rb") as source:
        marker = source.read(4)
        if len(marker) != 4:
            raise ValueError(f"truncated CMU dictionary: {path}")
        raw = struct.unpack("<I", marker)[0]
        if marker == b"CMD2":
            count_bytes = source.read(4)
            if len(count_bytes) != 4:
                raise ValueError(f"truncated CMD2 count: {path}")
            count = struct.unpack("<I", count_bytes)[0]
        else:
            count = raw
        for index in range(count):
            size_bytes = source.read(1)
            if not size_bytes:
                raise ValueError(f"truncated dictionary at entry {index}/{count}")
            word_size = size_bytes[0]
            word = source.read(word_size).decode("utf-8")
            phone_count_bytes = source.read(1)
            if not phone_count_bytes:
                raise ValueError(f"truncated dictionary phones at entry {index}")
            phone_count = phone_count_bytes[0]
            if len(source.read(phone_count * 2)) != phone_count * 2:
                raise ValueError(f"truncated dictionary payload at entry {index}")
            yield word
