#!/usr/bin/env python3
"""Bridge a sanoTTS raw Piperlite voice package to canonical GGUF.

Input directory layout (the format used by current sanoTTS voice packages):

    manifest.json
    weights.fp16.bin
    piper-phoneme-config.json

The script reconstructs each named tensor from manifest offsets and emits a
plain F32 (default) or F16 GGUF plus a normalized config JSON compatible with
InflectNanoTTS.cpp's tools/convert_sano.py.

Typical two-step conversion:

    python tools/convert_sano_raw.py \
      --voice-dir amy-en-1p1m \
      --output amy-1p1m-f32.gguf \
      --config-out amy-1p1m-inflect.json

    python tools/convert_sano.py \
      --input amy-1p1m-f32.gguf \
      --config amy-1p1m-inflect.json \
      --output amy-1p1m-q4_0_e.gguf \
      --quantize q4_0_e

Use --quantize q8_0 in the second command for the Q8_0 variant.

The intermediate GGUF is intentionally *canonical*, not runtime-packed: Conv1d
weights remain [out,in,kernel] and ConvTranspose1d weights remain
[in,out,kernel]. convert_sano.py performs the ESP32-oriented transpose/flatten,
padding, quantization and runtime metadata emission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_FORMAT = "roota.raw-fp16.v1"
COMPONENTS = ("duration", "acoustic", "decoder")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _int(cfg: dict[str, Any], key: str, where: str) -> int:
    value = int(require(cfg, key, where))
    if value <= 0:
        raise ValueError(f"{where}.{key} must be positive, got {value}")
    return value


def _kernel(cfg: dict[str, Any], where: str) -> int:
    raw = cfg.get("kernel", cfg.get("kernel_size"))
    if raw is None:
        raise ValueError(f"{where}: missing kernel/kernel_size")
    value = int(raw)
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{where}: kernel must be a positive odd integer, got {value}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def component(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    comps = require(manifest, "components", "manifest")
    if not isinstance(comps, dict):
        raise ValueError("manifest.components must be an object")
    comp = comps.get(name)
    if not isinstance(comp, dict):
        raise ValueError(f"manifest.components.{name} is missing or not an object")
    return comp


def merged_decoder_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Handle both direct decoder configs and exporter metadata wrappers.

    Some sanoTTS manifests carry a `matched_config` snapshot in addition to
    top-level fields. Prefer explicit top-level values and fill missing values
    from matched_config.
    """
    result: dict[str, Any] = {}
    nested = raw.get("matched_config")
    if isinstance(nested, dict):
        result.update(nested)
    result.update(raw)
    return result


def normalize_phoneme_map(raw: Any) -> dict[str, int | list[int]]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("piper-phoneme-config.json: missing phoneme_id_map")
    out: dict[str, int | list[int]] = {}
    for symbol, value in raw.items():
        if not isinstance(symbol, str) or len(symbol) != 1:
            raise ValueError(f"phoneme map key must be one codepoint: {symbol!r}")
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"phoneme map entry must contain exactly one id: {symbol!r}")
            pid = int(value[0])
            out[symbol] = [pid]
        else:
            pid = int(value)
            out[symbol] = pid
        if not 0 <= pid <= 255:
            raise ValueError(f"phoneme id does not fit uint8: {symbol!r} -> {pid}")
    # convert_sano.py accepts ints or singleton lists, but framing must match.
    def first(v: int | list[int]) -> int:
        return int(v[0] if isinstance(v, list) else v)
    for symbol, expected in (("_", 0), ("^", 1), ("$", 2)):
        if symbol not in out or first(out[symbol]) != expected:
            raise ValueError(
                f"Piper framing symbol {symbol!r} must map to {expected}, got {out.get(symbol)!r}"
            )
    return out


def normalize_config(
    manifest: dict[str, Any], phoneme: dict[str, Any] | None
) -> dict[str, Any]:
    dur_comp = component(manifest, "duration")
    ac_comp = component(manifest, "acoustic")
    dec_comp = component(manifest, "decoder")

    dur = require(dur_comp, "config", "manifest.components.duration")
    ac = require(ac_comp, "config", "manifest.components.acoustic")
    dec_raw = require(dec_comp, "config", "manifest.components.decoder")
    if not isinstance(dur, dict) or not isinstance(ac, dict) or not isinstance(dec_raw, dict):
        raise ValueError("duration/acoustic/decoder component config must be JSON objects")
    dec = merged_decoder_config(dec_raw)

    if str(dur.get("architecture", "")) != "duration_conv":
        raise ValueError(f"unsupported duration architecture: {dur.get('architecture')!r}")
    if str(ac.get("architecture", "")) != "token_context":
        raise ValueError(
            "this bridge currently targets the adapter-free Piperlite token_context acoustic path; "
            f"got acoustic architecture {ac.get('architecture')!r}"
        )
    if str(dec.get("variant", "")) != "piperlite":
        raise ValueError(f"unsupported decoder variant: {dec.get('variant')!r}")

    channels = [int(x) for x in require(dec, "channels", "decoder config")]
    if len(channels) != 4 or any(x <= 0 for x in channels):
        raise ValueError(f"decoder.channels must contain four positive integers, got {channels!r}")

    out: dict[str, Any] = {
        "voice": str(manifest.get("voice") or manifest.get("package_name") or "sano-piperlite"),
        "language": str(require(manifest, "language", "manifest")),
        "sample_rate": int(require(manifest, "sample_rate", "manifest")),
        "duration_length_scale": float(
            manifest.get("inference", {}).get("duration_length_scale", 1.0)
            if isinstance(manifest.get("inference"), dict)
            else 1.0
        ),
        "duration": {
            "vocab_size": _int(dur, "vocab_size", "duration config"),
            "hidden": _int(dur, "hidden", "duration config"),
            "depth": _int(dur, "depth", "duration config"),
            "kernel": _kernel(dur, "duration config"),
            "max_tokens": _int(dur, "max_tokens", "duration config"),
            "max_duration": _int(dur, "max_duration", "duration config"),
        },
        "acoustic": {
            "vocab_size": _int(ac, "vocab_size", "acoustic config"),
            "hidden": _int(ac, "hidden", "acoustic config"),
            "token_depth": _int(ac, "token_depth", "acoustic config"),
            "depth": _int(ac, "depth", "acoustic config"),
            "kernel": _kernel(ac, "acoustic config"),
            "out_channels": _int(ac, "out_channels", "acoustic config"),
        },
        "decoder": {
            "channels": channels,
            "stage0_branches": [int(x) for x in dec.get("stage0_branches", [0, 1, 2])],
            "stage1_branches": [int(x) for x in dec.get("stage1_branches", [0, 1, 2])],
            "stage2_branches": [int(x) for x in dec.get("stage2_branches", [0, 1, 2])],
            "post_filter_channels": int(dec.get("post_filter_channels", 0) or 0),
            "post_filter_layers": int(dec.get("post_filter_layers", 0) or 0),
            "post_filter_kernel": int(dec.get("post_filter_kernel", 9) or 9),
            "post_filter_scale": float(dec.get("post_filter_scale", 0.0) or 0.0),
        },
    }
    if out["sample_rate"] <= 0 or out["duration_length_scale"] <= 0:
        raise ValueError("sample_rate and duration_length_scale must be positive")

    if phoneme is not None:
        raw_espeak = phoneme.get("espeak")
        espeak_voice = raw_espeak.get("voice") if isinstance(raw_espeak, dict) else None
        if espeak_voice:
            out["espeak_voice"] = str(espeak_voice)
        out["phoneme_id_map"] = normalize_phoneme_map(phoneme.get("phoneme_id_map"))
    return out


def expected_tensor_bytes(shape: tuple[int, ...], dtype: str) -> int:
    count = math.prod(shape)
    if dtype == "float16":
        return count * 2
    if dtype == "float32":
        return count * 4
    raise ValueError(f"unsupported model tensor dtype {dtype!r}; expected float16/float32")


def read_tensors(
    manifest: dict[str, Any], weights_path: Path, verify_tensor_hashes: bool
) -> dict[str, np.ndarray]:
    file_size = weights_path.stat().st_size
    tensors: dict[str, np.ndarray] = {}

    with weights_path.open("rb") as f:
        for component_name in COMPONENTS:
            comp = component(manifest, component_name)
            raw_tensors = require(comp, "tensors", f"manifest.components.{component_name}")
            if not isinstance(raw_tensors, list):
                raise ValueError(f"manifest.components.{component_name}.tensors must be an array")
            for desc in raw_tensors:
                if not isinstance(desc, dict):
                    raise ValueError(f"{component_name}: tensor descriptor is not an object")
                local_name = str(require(desc, "name", f"{component_name} tensor"))
                full_name = f"{component_name}.{local_name}"
                if full_name in tensors:
                    raise ValueError(f"duplicate tensor name: {full_name}")

                shape = tuple(int(x) for x in require(desc, "shape", full_name))
                if not shape or any(x <= 0 for x in shape):
                    raise ValueError(f"{full_name}: invalid shape {shape!r}")
                dtype = str(require(desc, "dtype", full_name))
                offset = int(require(desc, "offset_bytes", full_name))
                nbytes = int(require(desc, "nbytes", full_name))
                expected = expected_tensor_bytes(shape, dtype)
                if nbytes != expected:
                    raise ValueError(
                        f"{full_name}: manifest nbytes={nbytes} but shape/dtype imply {expected}"
                    )
                if offset < 0 or nbytes <= 0 or offset + nbytes > file_size:
                    raise ValueError(
                        f"{full_name}: byte range [{offset}, {offset+nbytes}) is outside "
                        f"weights file size {file_size}"
                    )

                f.seek(offset)
                raw = f.read(nbytes)
                if len(raw) != nbytes:
                    raise ValueError(f"{full_name}: truncated tensor payload")
                if verify_tensor_hashes and desc.get("sha256"):
                    actual = hashlib.sha256(raw).hexdigest()
                    expected_sha = str(desc["sha256"])
                    if actual != expected_sha:
                        raise ValueError(
                            f"{full_name}: sha256 mismatch; manifest={expected_sha}, actual={actual}"
                        )

                np_dtype = "<f2" if dtype == "float16" else "<f4"
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
                # Canonical bridge stores host-owned contiguous fp32 tensors.
                tensors[full_name] = np.ascontiguousarray(arr, dtype=np.float32)

    if not tensors:
        raise ValueError("manifest contains no model tensors")
    return tensors


def write_gguf(output: Path, tensors: dict[str, np.ndarray], manifest: dict[str, Any], dtype: str) -> None:
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'gguf' is required. Use the same Python environment as "
            "tools/convert_sano.py / convert_v2.py."
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output), "sanotts")
    writer.add_string("sanotts.graph", "piperlite")
    writer.add_string("general.name", str(manifest.get("voice") or output.stem))
    writer.add_string("sanotts.source_format", str(manifest.get("format", "")))
    writer.add_uint32("sanotts.sample_rate", int(manifest.get("sample_rate", 22050)))
    writer.add_uint32("sanotts.hop_length", int(manifest.get("hop_length", 256)))

    raw_type = (
        gguf.GGMLQuantizationType.F32
        if dtype == "f32"
        else gguf.GGMLQuantizationType.F16
    )
    for name, source in tensors.items():
        payload = source if dtype == "f32" else source.astype(np.float16)
        writer.add_tensor(name, np.ascontiguousarray(payload), raw_dtype=raw_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voice-dir", type=Path, required=True,
        help="Local sanoTTS Piperlite package containing manifest.json + weights.fp16.bin",
    )
    parser.add_argument("--output", type=Path, required=True, help="Canonical intermediate GGUF")
    parser.add_argument(
        "--config-out", type=Path,
        help="Normalized config for convert_sano.py (default: <output>.config.json)",
    )
    parser.add_argument(
        "--dtype", choices=("f32", "f16"), default="f32",
        help="Intermediate GGUF storage type (default: f32)",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip manifest whole-file and per-tensor SHA-256 checks",
    )
    args = parser.parse_args(argv)

    voice_dir = args.voice_dir.expanduser().resolve()
    manifest_path = voice_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"error: missing {manifest_path}")
    manifest = load_json(manifest_path)
    if manifest.get("format") != SUPPORTED_FORMAT:
        raise SystemExit(
            f"error: unsupported manifest format {manifest.get('format')!r}; "
            f"expected {SUPPORTED_FORMAT!r}"
        )

    weights_file = str(require(manifest, "weights_file", "manifest"))
    weights_path = voice_dir / weights_file
    if not weights_path.is_file():
        raise SystemExit(f"error: missing {weights_path}")

    expected_size = manifest.get("weights_size_bytes")
    if expected_size is not None and weights_path.stat().st_size != int(expected_size):
        raise SystemExit(
            f"error: weights size {weights_path.stat().st_size} != manifest {expected_size}"
        )
    if not args.no_verify and manifest.get("weights_sha256"):
        actual = sha256_file(weights_path)
        expected = str(manifest["weights_sha256"])
        if actual != expected:
            raise SystemExit(
                f"error: weights sha256 mismatch\n  manifest: {expected}\n  actual:   {actual}"
            )

    phoneme_path = voice_dir / str(
        (manifest.get("frontend") or {}).get("included_config", "piper-phoneme-config.json")
        if isinstance(manifest.get("frontend"), dict)
        else "piper-phoneme-config.json"
    )
    phoneme = load_json(phoneme_path) if phoneme_path.is_file() else None
    if phoneme is None:
        print(f"warning: {phoneme_path.name} not found; generated config will require --phoneme-config later")

    normalized = normalize_config(manifest, phoneme)
    tensors = read_tensors(manifest, weights_path, verify_tensor_hashes=not args.no_verify)

    # Catch obvious family mismatches before writing the intermediate file.
    for prefix in ("duration.", "acoustic.", "decoder."):
        if not any(name.startswith(prefix) for name in tensors):
            raise SystemExit(f"error: no {prefix} tensors found")

    write_gguf(args.output, tensors, manifest, args.dtype)

    config_out = args.config_out or args.output.with_suffix(args.output.suffix + ".config.json")
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    params = sum(int(t.size) for t in tensors.values())
    print(f"wrote {args.output} ({len(tensors)} tensors, {params:,} parameters, {args.dtype})")
    print(f"wrote {config_out}")
    if phoneme is None:
        print("next: pass --phoneme-config <voice-dir>/piper-phoneme-config.json to convert_sano.py")
    else:
        print("next: convert_sano.py can use the generated config directly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
