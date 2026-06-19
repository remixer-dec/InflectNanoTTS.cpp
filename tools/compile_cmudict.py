#!/usr/bin/env python3
"""Compiles the text-based cmudict.rep into a compressed, binary cmudict.bin
for zero-RAM lookup on microcontrollers."""

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

def main():
    if len(sys.argv) < 3:
        print("Usage: python compile_cmudict.py <cmudict.rep> <cmudict.bin>")
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

if __name__ == "__main__":
    main()
