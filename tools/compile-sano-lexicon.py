#!/usr/bin/env python3
"""Compile an SNL1 or SNL2 lexicon for sanoTTS Piperlite.

SNL1 is a separate format from Inflect v2 IVL2.  Existing IVL2 files and the
V2Frontend reader are not modified.  SNL1 stores raw per-word Piper phoneme
ids (no BOS/PAD/EOS); the C++ SanoFrontend inserts spaces/punctuation and
applies global Piper framing at runtime.

Pronunciations are generated offline with the same eSpeak/phonemizer contract
as sanoTTS/Piper: stress enabled, untied IPA, preserved Piper punctuation,
language-switch flags removed, trailing whitespace stripped, then NFD mapping
through the voice's phoneme_id_map.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import struct
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from sano_common import (
    SNL1_BUCKET as BUCKET,
    SNL1_BUCKET_SIZE as BUCKET_SIZE,
    SNL1_HEADER as HEADER,
    SNL1_HEADER_SIZE as HEADER_SIZE,
    SNL1_MAGIC as MAGIC,
    SNL1_VERSION as VERSION,
    normalize_language,
    read_cmd2_words,
)
from v2_common import canonical_json
from sano_g2p import phonetic_units


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
PUNCTUATION_MARKS = "!'(),-.:;?\""

assert HEADER_SIZE == 88
assert BUCKET_SIZE == 24


def normalize_id_map(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("phoneme_id_map is missing or empty")
    result: dict[str, int] = {}
    for symbol, raw_id in raw.items():
        if len(symbol) != 1:
            raise ValueError(f"phoneme map key must be one codepoint: {symbol!r}")
        if isinstance(raw_id, list):
            if len(raw_id) != 1:
                raise ValueError(f"phoneme map value must contain one id: {symbol!r}")
            raw_id = raw_id[0]
        value = int(raw_id)
        if not 0 <= value <= 255:
            raise ValueError(f"SNL1 ids are uint8: {symbol!r} -> {value}")
        result[symbol] = value
    for symbol, expected in (("_", 0), ("^", 1), ("$", 2)):
        if result.get(symbol) != expected:
            raise ValueError(
                f"framing symbol {symbol!r} maps to {result.get(symbol)!r}; "
                f"expected {expected}"
            )
    return result


def map_hash(id_map: dict[str, int]) -> str:
    payload = json.dumps(
        id_map, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = normalize_id_map(payload.get("phoneme_id_map"))

    espeak = payload.get("espeak") if isinstance(payload.get("espeak"), dict) else {}

    espeak_voice = str(
        payload.get("espeak_voice")
        or espeak.get("voice")
        or ""
    )

    if not espeak_voice:
        raise ValueError("config is missing espeak_voice / espeak.voice")

    # SNL1 language is a compact runtime identifier.
    # eSpeak voice remains separate and may be more specific.
    language = normalize_language(espeak_voice.split("-", 1)[0])

    return ids, espeak_voice, language


def configure_espeak():
    import os

    try:
        import espeakng_loader
    except ImportError as exc:
        raise RuntimeError(
            "compile-sano-lexicon requires espeakng-loader and phonemizer-fork"
        ) from exc

    library = espeakng_loader.get_library_path()
    data_path = espeakng_loader.get_data_path()

    # Configure the bundled native eSpeak library before phonemizer imports
    # or initializes an EspeakBackend. This matches the known-working
    # Inflect-Nano-v2 frontend.
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = library
    os.environ["ESPEAK_DATA_PATH"] = data_path

    espeakng_loader.make_library_available()

    loaded = espeakng_loader.load_library()
    if loaded is None:
        raise RuntimeError(
            f"could not load bundled espeak-ng library: {library}"
        )

    # phonemizer-fork additionally supports explicitly passing the data
    # directory into espeak_Initialize. Keep this as a second layer.
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(library)

    if hasattr(EspeakWrapper, "set_data_path"):
        EspeakWrapper.set_data_path(data_path)

    # Probe the exact configuration we're going to use.
    from phonemizer.backend import EspeakBackend

    try:
        probe = EspeakBackend(
            "en-us",
            preserve_punctuation=True,
            punctuation_marks=PUNCTUATION_MARKS,
            with_stress=True,
            tie=False,
            language_switch="remove-flags",
        )
        probe.phonemize(["a"], strip=False, separator=None)
    except Exception as exc:
        raise RuntimeError(
            f"could not initialize bundled espeak-ng "
            f"(library={library!r}, data={data_path!r})"
        ) from exc


def make_backend(espeak_voice: str):
    configure_espeak()
    from phonemizer.backend import EspeakBackend

    candidates = [espeak_voice]
    if espeak_voice == "en":
        candidates.extend(("en-us", "en-gb"))
    last_error = None
    for candidate in candidates:
        try:
            return EspeakBackend(
                candidate,
                preserve_punctuation=True,
                punctuation_marks=PUNCTUATION_MARKS,
                with_stress=True,
                tie=False,
                language_switch="remove-flags",
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"no eSpeak voice for {espeak_voice!r}") from last_error


def ascii_lower(text: str) -> str:
    return "".join(
        chr(ord(char) + 32) if "A" <= char <= "Z" else char
        for char in text
    )


def runtime_word_key(word: str) -> str:
    word = unicodedata.normalize("NFC", word.strip())
    word = word.replace("’", "'").replace("‘", "'")
    return ascii_lower(word)


def extract_words(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for char in unicodedata.normalize("NFC", text):
        category = unicodedata.category(char)
        if char.isalnum() or category.startswith(("L", "M")) or (
            char == "'" and current
        ):
            current.append(char)
        else:
            if current:
                words.append(runtime_word_key("".join(current)))
                current.clear()
    if current:
        words.append(runtime_word_key("".join(current)))
    return [word for word in words if word]


def collect_words(
    cmudict: Path | None,
    word_lists: list[Path],
    supplementals: list[Path],
    language: str,
    texts: list[str] | None = None,
) -> list[str]:
    words: set[str] = set()
    if cmudict is not None:
        for word in read_cmd2_words(cmudict):
            words.add(runtime_word_key(word))
    for path in [*word_lists, *supplementals]:
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
                words.update(extract_words(line))
    for text in texts or []:
        words.update(extract_words(text))
    # The runtime spells OOV words using character pronunciations in this voice.
    words.update("abcdefghijklmnopqrstuvwxyz0123456789")
    if language == "vi":
        # Vietnamese vowel/letter variants, including uppercase and tone marks.
        letters = "ăâđêôơưáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ"
        letters += "".join(chr(codepoint) for codepoint in range(0x1ea0, 0x1efa))
        words.update(runtime_word_key(char) for char in letters + letters.upper())
    if language == "en":
        # Runtime OOV spelling uses these names.  Keeping them in the same
        # voice lexicon avoids a second symbol/pronunciation table.
        words.update(
            {
                "ay", "bee", "see", "dee", "ee", "eff", "gee", "aitch", "eye",
                "jay", "kay", "ell", "em", "en", "oh", "pee", "cue", "ar", "ess",
                "tee", "you", "vee", "double", "ex", "why", "zee",
            }
        )
    return sorted(word for word in words if word)


def encode_pronunciation(
    word: str,
    pronunciation: str,
    id_map: dict[str, int],
    missing: Counter[str],
) -> bytes | None:
    result: list[int] = []
    for symbol in unicodedata.normalize("NFD", pronunciation.rstrip()):
        token = id_map.get(symbol)
        if token is not None:
            # Word entries deliberately exclude inter-word separator spaces;
            # SanoFrontend inserts the voice's space id between text words.
            if symbol != " ":
                result.append(token)
        else:
            missing[symbol] += 1
            # Piper skips symbols missing from phoneme_id_map.  Do the same;
            # one unsupported combining detail should not remove a word.
    return bytes(result) if result else None


def phonemize_words(backend, words: list[str], jobs: int) -> list[str]:
    try:
        return backend.phonemize(
            words, strip=False, separator=None, njobs=max(1, jobs)
        )
    except TypeError:
        # Older phonemizer-fork versions do not expose njobs on the backend
        # method.  The list call still amortizes backend initialization.
        return backend.phonemize(words, strip=False, separator=None)


def fnv1a(word: bytes) -> int:
    value = FNV_OFFSET
    for byte in word:
        if 65 <= byte <= 90:
            byte += 32
        value ^= byte
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value or 1


def next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result <<= 1
    return result


def write_snl1(
    entries: list[tuple[str, bytes]],
    output: Path,
    language: str,
    frontend_hash: str,
    load_factor: float,
) -> tuple[int, int]:
    if not entries:
        raise ValueError("cannot write an empty SNL1")
    if not (0.25 <= load_factor <= 0.80):
        raise ValueError("load factor must be in [0.25, 0.80]")
    language_bytes = language.encode("ascii")
    if len(language_bytes) > 7:
        raise ValueError("SNL1 language code must fit 7 ASCII bytes")

    minimum = int(len(entries) / load_factor) + 1
    bucket_count = next_power_of_two(max(2, minimum))
    bucket_rows: list[tuple[int, int, int, int, int, int] | None] = [
        None
    ] * bucket_count
    blob = bytearray()
    max_word = 0
    max_tokens = 0

    for word, tokens in entries:
        encoded = word.encode("utf-8")
        if len(encoded) > 0xFFFF or len(tokens) > 0xFFFF:
            raise ValueError(f"SNL1 entry exceeds uint16 limits: {word!r}")
        word_offset = len(blob)
        blob.extend(encoded)
        token_offset = len(blob)
        blob.extend(tokens)
        hash_value = fnv1a(encoded)
        slot = hash_value & (bucket_count - 1)
        while bucket_rows[slot] is not None:
            slot = (slot + 1) & (bucket_count - 1)
        bucket_rows[slot] = (
            hash_value,
            word_offset,
            token_offset,
            len(encoded),
            len(tokens),
            0,
        )
        max_word = max(max_word, len(encoded))
        max_tokens = max(max_tokens, len(tokens))

    bucket_offset = HEADER_SIZE
    blob_offset = bucket_offset + bucket_count * BUCKET_SIZE
    file_size = blob_offset + len(blob)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as destination:
        destination.write(
            HEADER.pack(
                MAGIC,
                VERSION,
                HEADER_SIZE,
                bytes.fromhex(frontend_hash),
                len(entries),
                bucket_count,
                bucket_offset,
                blob_offset,
                file_size,
                language_bytes.ljust(8, b"\0"),
                max_word,
                max_tokens,
                0,
            )
        )
        empty = BUCKET.pack(0, 0, 0, 0, 0, 0)
        for row in bucket_rows:
            destination.write(empty if row is None else BUCKET.pack(*row))
        destination.write(blob)
    return bucket_count, len(blob)


def pack_u24(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFF:
        raise ValueError(f"value does not fit u24: {value}")
    return value.to_bytes(3, "little")


def fnv1a_snl2(word: bytes) -> int:
    value = FNV_OFFSET
    for byte in word:
        value ^= byte
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def write_snl2(
    entries: list[tuple[str, bytes]],
    output: Path,
    language: str,
    frontend_hash: str,
) -> tuple[int, int, int]:
    """Write the deterministic block-compressed SNL2 format."""
    if not entries:
        raise ValueError("cannot write an empty SNL2")

    language_bytes = language.encode("ascii")
    if len(language_bytes) > 7:
        raise ValueError("SNL2 language code must fit 7 ASCII bytes")

    # Normalize and deduplicate at the format boundary as required by SNL2.
    # The compiler's input is already normalized, but doing it here prevents a
    # second caller from silently emitting a non-conforming file.
    normalized: dict[str, bytes] = {}
    for word, tokens in entries:
        key = runtime_word_key(word)
        key_bytes = key.encode("utf-8")
        if not key or len(key_bytes) > 255 or len(tokens) > 255:
            raise ValueError(f"SNL2 entry exceeds uint8 limits: {word!r}")
        previous = normalized.setdefault(key, tokens)
        if previous != tokens:
            raise ValueError(f"conflicting pronunciations for normalized word {key!r}")

    sorted_entries = sorted(
        ((word.encode("utf-8"), tokens) for word, tokens in normalized.items()),
        key=lambda item: item[0],
    )
    entry_count = len(sorted_entries)
    if entry_count > 0xFFFFFF:
        raise ValueError("SNL2 entry count exceeds u24 limit")

    block_count = (entry_count + 15) // 16
    blob = bytearray()
    block_offsets = [0]
    for block_start in range(0, entry_count, 16):
        previous = b""
        for index in range(block_start, min(block_start + 16, entry_count)):
            word, tokens = sorted_entries[index]
            prefix = 0 if index % 16 == 0 else _common_prefix(previous, word)
            suffix = word[prefix:]
            if len(suffix) > 255:
                raise ValueError(f"SNL2 suffix exceeds uint8 limit: {word!r}")
            blob.extend((prefix, len(suffix), len(tokens)))
            blob.extend(suffix)
            blob.extend(tokens)
            previous = word
        block_offsets.append(len(blob))

    if len(blob) > 0xFFFFFF:
        raise ValueError("SNL2 compressed blob exceeds u24 limit")

    bucket_count = max(1, (entry_count + 1) // 2)
    while True:
        buckets: list[list[tuple[int, int]]] = [[] for _ in range(bucket_count)]
        for index, (word, _) in enumerate(sorted_entries):
            hash_value = fnv1a_snl2(word)
            locator = ((index // 16) << 4) | (index & 0x0F)
            buckets[hash_value % bucket_count].append(
                ((hash_value >> 48) & 0xFFFF, locator)
            )
        if all(len(bucket) <= 16 for bucket in buckets):
            break
        bucket_count *= 2

    bucket_directory = [0]
    hash_records: list[tuple[int, int]] = []
    for bucket in buckets:
        hash_records.extend(sorted(bucket))
        bucket_directory.append(len(hash_records))

    header_size = 88
    bucket_dir_offset = header_size
    hash_index_offset = bucket_dir_offset + 3 * (bucket_count + 1)
    block_dir_offset = hash_index_offset + 5 * entry_count
    blob_offset = block_dir_offset + 3 * (block_count + 1)
    file_size = blob_offset + len(blob)
    for value, label in (
        (bucket_dir_offset, "bucket directory offset"),
        (hash_index_offset, "hash index offset"),
        (block_dir_offset, "block directory offset"),
        (blob_offset, "blob offset"),
        (file_size, "file size"),
    ):
        if value > 0xFFFFFFFF:
            raise ValueError(f"SNL2 {label} exceeds uint32 limit")

    header = struct.pack(
        "<4sHHI32sIIIIIIII8sBBH",
        b"SNL2", 1, header_size, 0, bytes.fromhex(frontend_hash),
        entry_count, bucket_count, block_count,
        bucket_dir_offset, hash_index_offset, block_dir_offset, blob_offset,
        file_size, language_bytes.ljust(8, b"\0"), 16, 16, 0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as destination:
        destination.write(header)
        for value in bucket_directory:
            destination.write(pack_u24(value))
        for fingerprint, locator in hash_records:
            destination.write(struct.pack("<H", fingerprint))
            destination.write(pack_u24(locator))
        for value in block_offsets:
            destination.write(pack_u24(value))
        destination.write(blob)
    return bucket_count, block_count, len(blob)


def _common_prefix(left: bytes, right: bytes) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phoneme-config", type=Path, required=True,
                        help="Piper phoneme config or sano config with phoneme_id_map")
    parser.add_argument("--cmudict", type=Path,
                        help="optional compiled CMD2/legacy CMU word source")
    parser.add_argument("--word-list", type=Path, action="append", default=[])
    parser.add_argument("--supplemental", type=Path, action="append", default=[])
    parser.add_argument("--text", action="append", default=[],
                        help="include words from this text (repeatable)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--format", "-format", choices=("snl1", "snl2"), default="snl2",
        help="lexicon format to emit (default: snl2)",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--load-factor", type=float, default=0.65)
    parser.add_argument("--expected-map-hash")
    args = parser.parse_args(argv)

    if args.cmudict is None and not args.word_list and not args.supplemental and not args.text:
        parser.error("provide --cmudict, --word-list, --supplemental, or --text")
    id_map, espeak_voice, language = load_config(args.phoneme_config)
    frontend_hash = map_hash(id_map)
    if (args.format == "snl1" and args.expected_map_hash and
            frontend_hash != args.expected_map_hash.lower()):
        raise SystemExit(
            f"phoneme map hash mismatch: expected {args.expected_map_hash}, "
            f"got {frontend_hash}"
        )

    words = collect_words(
        args.cmudict, args.word_list, args.supplemental, language, args.text
    )
    backend = make_backend(espeak_voice)
    pronunciations = phonemize_words(backend, words, args.jobs)
    if len(pronunciations) != len(words):
        raise RuntimeError(
            f"phonemizer returned {len(pronunciations)} rows for {len(words)} words"
        )

    missing: Counter[str] = Counter()
    entries: list[tuple[str, bytes]] = []
    skipped: list[str] = []
    for word, pronunciation in zip(words, pronunciations, strict=True):
        encoded = encode_pronunciation(word, pronunciation, id_map, missing)
        if encoded:
            entries.append((word, encoded))
        else:
            skipped.append(word)
    units = phonetic_units(language, id_map)
    entries.extend(units)
    if args.format == "snl2":
        bucket_count, block_count, blob_bytes = write_snl2(
            entries, args.output, language, frontend_hash
        )
    else:
        bucket_count, blob_bytes = write_snl1(
            entries, args.output, language, frontend_hash, args.load_factor
        )
        block_count = 0

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "format": f"sanotts_{args.format}_manifest_v1",
        "frontend": "piperlite",
        "language": language,
        "espeak_voice": espeak_voice,
        "phoneme_map_hash": frontend_hash,
        "entry_count": len(entries),
        "source_word_count": len(words),
        "phonetic_unit_count": len(units),
        "bucket_count": bucket_count,
        "block_count": block_count,
        "load_factor": len(entries) / bucket_count,
        "blob_bytes": blob_bytes,
        "skipped_words": skipped,
        "missing_phoneme_symbols": {
            f"U+{ord(symbol):04X}": count
            for symbol, count in sorted(missing.items())
        },
        "lookup": "fnv1a64_bounded_buckets" if args.format == "snl2"
        else "fnv1a64_open_addressing",
        "raw_word_phoneme_ids": True,
        "piper_framing_at_runtime": True,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json(manifest))
    print(
        f"wrote {len(entries)} {args.format.upper()} entries to {args.output} "
        f"({bucket_count} buckets, load={len(entries) / bucket_count:.3f})"
    )
    print(f"phoneme map hash: {frontend_hash}")
    if skipped:
        print(f"skipped {len(skipped)} empty/unencodable entries", file=sys.stderr)
    if missing:
        print(
            "missing map symbols were skipped to match Piper behavior: "
            + ", ".join(
                f"U+{ord(symbol):04X}={count}"
                for symbol, count in sorted(missing.items())
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
