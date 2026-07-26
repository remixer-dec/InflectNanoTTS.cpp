#!/usr/bin/env python3
"""Compile one self-indexed, flash-backed Inflect v2 IVL2 lexicon.

Run this where the released Inflect-Nano-v2 eSpeak frontend is available.
The emitted .bin has no eSpeak or Python runtime dependency.
"""

from __future__ import annotations

import argparse
import importlib
import json
import struct
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from v2_common import (
    IVI2_HEADER,
    IVI2_HEADER_SIZE,
    IVL2_HEADER,
    IVL2_HEADER_SIZE,
    IVL2_VERSION,
    SYMBOL_HASH,
    SYMBOL_HASH_HEX,
    SYMBOL_TO_ID,
    canonical_json,
    common_prefix_bytes,
    read_cmd2_words,
)


def load_frontend(reference_root: Path):
    module_path = reference_root / "inflect_nano_v2_frontend.py"
    vits_path = reference_root / "inflect_vits_frontend.py"
    if not module_path.is_file() or not vits_path.is_file():
        raise FileNotFoundError(
            "reference root must contain inflect_nano_v2_frontend.py and "
            "inflect_vits_frontend.py"
        )
    sys.path.insert(0, str(reference_root))
    # Import normally so dataclasses and other runtime introspection can find
    # the module in sys.modules while its classes are being defined.
    return importlib.import_module("inflect_vits_frontend")


def collect_words(cmudict: Path, supplemental: list[Path], normalize) -> list[str]:
    raw_words = set(read_cmd2_words(cmudict))
    for path in supplemental:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("{"):
                    row = json.loads(line)
                    line = str(
                        row.get("word")
                        or row.get("text")
                        or row.get("target_text")
                        or ""
                    )
                raw_words.update(line.split())

    # The runtime normalizes before lookup. Compile normalized output words,
    # including the released override expansions and spoken letter names.
    words: set[str] = set()
    for raw in raw_words:
        # CMD2 stores dictionary keys in uppercase. Feeding those directly to
        # v2 normalization makes ordinary words look like acronyms (for
        # example, HELLO becomes "aitch ee ell ell oh"), collapsing the
        # dictionary into a few letter-name entries.
        normalized = normalize(raw.lower())
        current = []
        for character in normalized:
            if character.isascii() and (
                character.isalpha() or character == "'"
            ):
                current.append(character.lower())
            elif current:
                words.add("".join(current))
                current.clear()
        if current:
            words.add("".join(current))
    words.update(
        {
            "ay", "bee", "see", "dee", "ee", "eff", "gee", "aitch", "eye",
            "jay", "kay", "ell", "em", "en", "oh", "pee", "cue", "ar", "ess",
            "tee", "you", "vee", "double", "ex", "why", "zee",
        }
    )
    return sorted(word for word in words if word)


def write_assets(
    entries: list[tuple[str, bytes]],
    output: Path,
    stride: int,
) -> None:
    sparse_count = (len(entries) + stride - 1) // stride
    output.parent.mkdir(parents=True, exist_ok=True)
    sparse_rows: list[tuple[int, int, bytes]] = []
    with output.open("wb") as destination:
        header = (
            b"IVL2",
            IVL2_VERSION,
            IVL2_HEADER_SIZE,
            SYMBOL_HASH,
            len(entries),
            stride,
            sparse_count,
            IVL2_HEADER_SIZE,
            0,
            b"\0" * 8,
        )
        destination.write(IVL2_HEADER.pack(*header))
        previous = b""
        for entry_index, (word, tokens) in enumerate(entries):
            encoded = word.encode("utf-8")
            if len(encoded) > 255 or len(tokens) > 255:
                raise ValueError(f"IVL2 entry exceeds uint8 limits: {word!r}")
            if entry_index % stride == 0:
                prefix = 0
                sparse_rows.append((entry_index, destination.tell(), encoded))
            else:
                prefix = common_prefix_bytes(previous, encoded)
            suffix = encoded[prefix:]
            destination.write(struct.pack("<BB", prefix, len(suffix)))
            destination.write(suffix)
            destination.write(struct.pack("<B", len(tokens)))
            destination.write(tokens)
            previous = encoded

        index_offset = destination.tell()
        destination.write(
            IVI2_HEADER.pack(
                b"IVI2",
                IVL2_VERSION,
                IVI2_HEADER_SIZE,
                SYMBOL_HASH,
                len(entries),
                stride,
                len(sparse_rows),
            )
        )
        for entry_index, offset, word in sparse_rows:
            destination.write(struct.pack("<IQH", entry_index, offset, len(word)))
            destination.write(word)
        destination.seek(0)
        destination.write(
            IVL2_HEADER.pack(
                *header[:8],
                index_offset,
                header[9],
            )
        )


def encode_pronunciation(
    word: str,
    pronunciation: str,
    removed_marks: Counter[str],
) -> bytes | None:
    """Map released symbols and discard only unsupported combining marks.

    eSpeak occasionally adds pronunciation diacritics (notably U+0303) that
    the released 178-symbol model cannot consume. Dropping such a mark retains
    the encodable base phone. Unknown non-combining symbols are reported by
    returning None so one obscure entry cannot abort the complete lexicon.
    """
    token_ids = []
    for character in pronunciation:
        token_id = SYMBOL_TO_ID.get(character)
        if token_id is not None:
            token_ids.append(token_id)
        elif unicodedata.combining(character):
            removed_marks[character] += 1
        else:
            codepoint = ord(character)
            print(
                f"warning: skipping {word!r}: unsupported phoneme symbol "
                f"U+{codepoint:04X} {character!r}",
                file=sys.stderr,
            )
            return None
    return bytes(token_ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--cmudict", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--index-output",
        type=Path,
        help="deprecated and ignored; the sparse index is embedded in --output",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.stride <= 65536:
        raise SystemExit("--stride must be in [1, 65536]")

    frontend = load_frontend(args.reference_root.resolve())
    words = collect_words(args.cmudict, args.supplemental, frontend.normalize_text)
    # One batch is intentional: this is the only eSpeak generation pass.
    phonemes = frontend.phonemize_normalized_batch(words, jobs=args.jobs)
    entries = []
    removed_marks: Counter[str] = Counter()
    skipped_words = []
    for word, pronunciation in zip(words, phonemes, strict=True):
        corrected = frontend._apply_phoneme_overrides(pronunciation)
        token_bytes = encode_pronunciation(word, corrected, removed_marks)
        if token_bytes is None:
            skipped_words.append(word)
            continue
        if not token_bytes:
            print(
                f"warning: skipping {word!r}: empty encodable pronunciation",
                file=sys.stderr,
            )
            skipped_words.append(word)
            continue
        entries.append((word, token_bytes))

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    write_assets(entries, args.output, args.stride)
    manifest = {
        "format": "inflect_v2_lexicon_manifest_v2",
        "symbol_count": 178,
        "symbol_hash": SYMBOL_HASH_HEX,
        "entry_count": len(entries),
        "sparse_stride": args.stride,
        "source_word_count": len(words),
        "skipped_words": skipped_words,
        "removed_combining_marks": {
            f"U+{ord(mark):04X}": count
            for mark, count in sorted(removed_marks.items())
        },
        "self_indexed": True,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json(manifest))
    print(f"wrote {len(entries)} IVL2 entries to {args.output}")
    if args.index_output:
        print(
            f"ignored deprecated --index-output {args.index_output}; "
            "the sparse index is embedded in the IVL2 file"
        )
    for mark, count in sorted(removed_marks.items()):
        print(
            f"removed unsupported combining mark U+{ord(mark):04X} "
            f"{mark!r} from {count} pronunciations"
        )
    if skipped_words:
        print(f"skipped {len(skipped_words)} entries with unsupported base symbols")
    print(f"symbol hash: {SYMBOL_HASH_HEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
