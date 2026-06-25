#!/usr/bin/env python3
"""Compiles cmudict.rep into cmudict.bin and writes a sparse lookup index."""

import os
import re
import struct
import sys

# English phone IDs from the full TinyTTS symbols.py table used by the model.
_symbol_to_id = {
    "_": 0,
    "V": 14,
    "aa": 21, "ae": 22, "ah": 23, "ao": 27, "aw": 28, "ay": 29,
    "b": 30, "ch": 33, "d": 34, "dh": 35, "eh": 39, "er": 43,
    "ey": 44, "f": 45, "g": 46, "hh": 49, "ih": 59, "iy": 65,
    "jh": 67, "k": 68, "l": 70, "m": 71, "n": 73, "ng": 74,
    "ow": 80, "oy": 81, "p": 82, "r": 85, "s": 87, "sh": 88,
    "t": 89, "th": 90, "uh": 99, "uw": 103, "w": 108, "y": 110,
    "z": 111, "zh": 112,
    "!": 208, "?": 209, "\u2026": 210, ",": 211, ".": 212, "'": 213,
    "-": 214, "\u00bf": 215, "\u00a1": 216, "SP": 217, "UNK": 218,
}

# ─────────────────────────────────────────────────────────────────────────
# 2. Phoneme Parsing
# ─────────────────────────────────────────────────────────────────────────

def parse_phoneme(phn_str):
    """Extracts phoneme and tone. E.g., 'EH2' -> ('eh', 3)"""
    tone = 0
    match = re.search(r"(\d)$", phn_str)
    if match:
        tone = int(match.group(1)) + 1
        phn_str = phn_str[:-1]
    return phn_str.lower(), tone

def process_syllables(syllables_str):
    """Processes 'EH2 K - S K L' -> list of (phone_id, tone)"""
    phonemes = []
    # Split by space, ignore the '-' syllable separator
    for phn in syllables_str.split(" "):
        if phn == "-" or not phn:
            continue
        phn_lower, tone = parse_phoneme(phn)
        if phn_lower in _symbol_to_id:
            phonemes.append((_symbol_to_id[phn_lower], tone))
        else:
            # Fallback for unknown characters in dictionary
            phonemes.append((_symbol_to_id["UNK"], 0))
    return phonemes

# ─────────────────────────────────────────────────────────────────────────
# 3. Main Compilation
# ─────────────────────────────────────────────────────────────────────────

def write_index_records(records, index_path, total_count, stride=256):
    with open(index_path, "wb") as f:
        f.write(b"CMY1")
        index_count = len(records)
        f.write(struct.pack("<III", stride, total_count, index_count))
        for entry_index, offset, word in records:
            word_bytes = word.encode("ascii")
            f.write(struct.pack("<IQB", entry_index, offset, len(word_bytes)))
            f.write(word_bytes)


def write_index(entries, index_path, stride=256):
    records = []
    offset = 8  # CMD2 magic + entry count.
    for entry_index, (word, phns) in enumerate(entries):
        if entry_index % stride == 0:
            records.append((entry_index, offset, word))
        offset += 1 + len(word.encode("ascii")) + 1 + 2 * len(phns)
    write_index_records(records, index_path, len(entries), stride)


def write_index_from_bin(bin_path, index_path, stride=256):
    records = []
    with open(bin_path, "rb") as f:
        marker = f.read(4)
        if len(marker) != 4:
            raise ValueError(f"{bin_path}: invalid header")
        if marker == b"CMD2":
            raw_count = f.read(4)
            if len(raw_count) != 4:
                raise ValueError(f"{bin_path}: invalid CMD2 count")
            count = struct.unpack("<I", raw_count)[0]
        else:
            count = struct.unpack("<I", marker)[0]

        for entry_index in range(count):
            offset = f.tell()
            raw_word_len = f.read(1)
            if len(raw_word_len) != 1:
                raise ValueError(f"{bin_path}: truncated word length at entry {entry_index}")
            word_len = raw_word_len[0]
            word = f.read(word_len).decode("ascii")
            raw_phone_count = f.read(1)
            if len(raw_phone_count) != 1:
                raise ValueError(f"{bin_path}: truncated phone count at entry {entry_index}")
            phone_count = raw_phone_count[0]
            f.seek(2 * phone_count, os.SEEK_CUR)
            if entry_index % stride == 0:
                records.append((entry_index, offset, word))

    write_index_records(records, index_path, count, stride)
    return count, len(records)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--index-only":
        bin_path = sys.argv[2]
        index_path = sys.argv[3] if len(sys.argv) >= 4 else os.path.splitext(bin_path)[0] + ".idx"
        count, index_count = write_index_from_bin(bin_path, index_path)
        index_size = os.path.getsize(index_path)
        print(
            f"Wrote sparse index for {count} words, {index_count} entries "
            f"to {index_path} ({index_size/1024:.1f} KB)"
        )
        return

    if len(sys.argv) < 3:
        print("Usage: python compile_cmudict.py <cmudict.rep> <cmudict.bin>")
        print("       python compile_cmudict.py --index-only <cmudict.bin> [cmudict.idx]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    if not os.path.exists(in_path):
        print(f"Error: {in_path} not found.")
        sys.exit(1)

    entries = []

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(";;;"):  # Skip comments
                continue

            # Split by 2 or more spaces (word  phonemes)
            parts = re.split(r"\s{2,}", line.strip(), maxsplit=1)
            if len(parts) != 2:
                continue

            word = parts[0].upper()
            # Some words have alternative pronunciations like "READ(1)"
            # We strip the (1) and just use the primary mapping
            word = re.sub(r"\(\d+\)$", "", word)

            # Remove non-alphanumeric from word (like apostrophes)
            # Keep it simple: A-Z
            if not re.match(r"^[A-Z]+$", word):
                continue

            phonemes = process_syllables(parts[1])
            if not phonemes:
                continue

            entries.append((word, phonemes))

    # Sort alphabetically for binary search
    entries.sort(key=lambda x: x[0])

    # Remove duplicates (keep first occurrence after sorting)
    unique_entries = []
    last_word = ""
    for word, phns in entries:
        if word != last_word:
            unique_entries.append((word, phns))
            last_word = word

    # Write binary file
    with open(out_path, "wb") as f:
        # Magic: marks full TinyTTS model-vocabulary phone IDs.
        f.write(b"CMD2")

        # Header: Number of entries (4 bytes)
        f.write(struct.pack("<I", len(unique_entries)))

        for word, phns in unique_entries:
            word_bytes = word.encode("ascii")

            # Word: [len: 1 byte][word: chars]
            f.write(struct.pack("<B", len(word_bytes)))
            f.write(word_bytes)

            # Phonemes: [count: 1 byte][phone_id: 1 byte, tone: 1 byte] * count
            f.write(struct.pack("<B", len(phns)))
            for phone_id, tone in phns:
                f.write(struct.pack("<BB", phone_id, tone))

    out_size = os.path.getsize(out_path)
    print(f"Compiled {len(unique_entries)} words to {out_path} ({out_size/1024:.1f} KB)")

    index_path = os.path.splitext(out_path)[0] + ".idx"
    write_index(unique_entries, index_path)
    index_size = os.path.getsize(index_path)
    print(f"Wrote sparse index to {index_path} ({index_size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
