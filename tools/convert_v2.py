#!/usr/bin/env python3
"""Convert and quantize Inflect Nano/Micro v2 models as GGUF.

The converter can write a folded FP32 GGUF directly from the released PyTorch
checkpoint, or quantize an existing canonical FP32 GGUF without PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from v2_common import SYMBOL_HASH_HEX, sha256_file


EXPECTED_TENSORS = 410
EXPECTED_WEIGHT_NORM_PAIRS = 108
REQUIRED_NAMESPACES = ("enc_p.", "dp.", "flow.", "dec.")
QUANT_PROFILES = ("f16", "q8_0", "mixed-q8_0", "q4_0_e")
ROOT_DIR = Path(__file__).resolve().parent.parent
GGML_QUANTIZER = ROOT_DIR / "build" / "tools" / "ggml_quantize_stdin"


def require(config: dict, dotted: str):
    value = config
    for component in dotted.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"config is missing required value {dotted!r}")
        value = value[component]
    return value


def load_state(checkpoint_path: Path):
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("expected released checkpoint mapping with a 'model' state dict")
    return payload["model"]


def fold_weight_norm(v_tensor, g_tensor) -> np.ndarray:
    v = v_tensor.detach().cpu().float().numpy()
    g = g_tensor.detach().cpu().float().numpy()
    norm = np.linalg.norm(v, axis=tuple(range(1, v.ndim)), keepdims=True)
    if np.any(norm == 0):
        raise ValueError("weight normalization tensor contains a zero norm")
    return np.ascontiguousarray(v * (g / norm), dtype=np.float32)


def converted_tensors(state: dict) -> dict[str, np.ndarray]:
    names = set(state)
    pairs: dict[str, tuple[str, str]] = {}
    for name in names:
        if name.endswith(".weight_v"):
            base = name[: -len(".weight_v")]
            g_name = base + ".weight_g"
            if g_name not in names:
                raise ValueError(f"missing matching weight_g for {name}")
            pairs[base] = (name, g_name)
        elif name.endswith(".weight_g"):
            base = name[: -len(".weight_g")]
            if base + ".weight_v" not in names:
                raise ValueError(f"missing matching weight_v for {name}")
    if len(pairs) != EXPECTED_WEIGHT_NORM_PAIRS:
        raise ValueError(
            f"expected {EXPECTED_WEIGHT_NORM_PAIRS} weight-norm pairs, "
            f"found {len(pairs)}"
        )

    result: dict[str, np.ndarray] = {}
    for name, tensor in state.items():
        if name.endswith(".weight_v") or name.endswith(".weight_g"):
            continue
        result[name] = np.ascontiguousarray(
            tensor.detach().cpu().float().numpy(), dtype=np.float32
        )
    for base, (v_name, g_name) in pairs.items():
        plain_name = base + ".weight"
        if plain_name in result:
            raise ValueError(f"folded tensor would overwrite {plain_name}")
        result[plain_name] = fold_weight_norm(state[v_name], state[g_name])
    return result


def validate_config(config: dict) -> dict[str, object]:
    expected = {
        "data.sampling_rate": 24000,
        "data.add_blank": True,
        "model.n_layers": 3,
        "model.kernel_size": 3,
        "model.resblock": "1",
        "model.resblock_kernel_sizes": [3, 7, 11],
        "model.resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "model.upsample_rates": [8, 8, 2, 2],
        "model.upsample_kernel_sizes": [16, 16, 4, 4],
        "model.use_sdp": False,
        "model.inference_only": True,
    }
    for key, expected_value in expected.items():
        actual = require(config, key)
        if actual != expected_value:
            raise ValueError(f"incompatible {key}: expected {expected_value!r}, got {actual!r}")
    dimensions = {
        key: require(config, key)
        for key in (
            "model.inter_channels",
            "model.hidden_channels",
            "model.filter_channels",
            "model.n_heads",
            "model.upsample_initial_channel",
        )
    }
    if any(not isinstance(value, int) or value <= 0
           for value in dimensions.values()):
        raise ValueError(f"invalid v2 model dimensions: {dimensions}")
    if dimensions["model.inter_channels"] % 2:
        raise ValueError("model.inter_channels must be even for coupling flow")
    if (dimensions["model.hidden_channels"] %
            dimensions["model.n_heads"]):
        raise ValueError("model.hidden_channels must divide evenly into n_heads")
    if dimensions["model.upsample_initial_channel"] % 16:
        raise ValueError(
            "model.upsample_initial_channel must support four halving stages"
        )
    return {
        **{key: require(config, key) for key in expected},
        **dimensions,
    }


def starts_with_any(value: str, prefixes: list[str]) -> bool:
    return any(value.startswith(prefix) for prefix in prefixes)


def is_v2_weight(name: str, tensor: np.ndarray) -> bool:
    return (
        name.endswith(".weight")
        and tensor.ndim >= 2
        and name.startswith(REQUIRED_NAMESPACES)
    )


def is_conv_weight(name: str) -> bool:
    return (
        name != "enc_p.emb.weight"
        and name.endswith(".weight")
        and name.startswith(REQUIRED_NAMESPACES)
    )


def is_decoder_upsample(name: str) -> bool:
    return name.startswith("dec.ups.") and name.endswith(".weight")


def protected_mixed_tensor(name: str) -> bool:
    return (
        name.endswith(".bias")
        or ".norm_" in name
        or ".norms_" in name
        or ".norm_layers_" in name
        or name == "enc_p.proj.weight"
        or name == "dp.proj.weight"
        or (name.startswith("flow.") and ".post." in name)
        or name.startswith("dec.resblocks.")
        or name == "dec.conv_post.weight"
    )


def mixed_quantizable(name: str, tensor: np.ndarray) -> bool:
    if protected_mixed_tensor(name) or not name.endswith(".weight"):
        return False
    if name.startswith(("enc_p.", "dp.")):
        return tensor.ndim >= 2
    if name.startswith("dec."):
        if name != "dec.ups.0.weight":
            return False
    elif not name.startswith("flow."):
        return False
    return tensor.ndim >= 2 and tensor.size >= 4096


def pad_last_dimension(tensor: np.ndarray, multiple: int) -> np.ndarray:
    padded = ((tensor.shape[-1] + multiple - 1) // multiple) * multiple
    if padded == tensor.shape[-1]:
        return tensor
    widths = [(0, 0)] * tensor.ndim
    widths[-1] = (0, padded - tensor.shape[-1])
    return np.pad(tensor, widths, mode="constant", constant_values=0)


def trim_ggml_dimensions(tensor: np.ndarray) -> np.ndarray:
    while tensor.ndim > 1 and tensor.shape[0] == 1:
        tensor = tensor.reshape(tensor.shape[1:])
    return tensor


def quantizer_is_usable(path: Path) -> bool:
    try:
        process = subprocess.run(
            [path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return False
    return process.returncode == 2 and b"usage:" in process.stderr


def build_ggml_quantizer() -> Path:
    sources = [
        ROOT_DIR / "tools" / "ggml_quantize_stdin.c",
        ROOT_DIR / "ggml" / "src" / "ggml-quants.c",
        ROOT_DIR / "ggml" / "src" / "ggml-quants.h",
        ROOT_DIR / "ggml" / "include" / "ggml.h",
    ]
    if (
        GGML_QUANTIZER.exists()
        and all(GGML_QUANTIZER.stat().st_mtime >= source.stat().st_mtime
                for source in sources)
        and quantizer_is_usable(GGML_QUANTIZER)
    ):
        return GGML_QUANTIZER

    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    compiler = compiler or shutil.which("clang")
    if not compiler:
        raise RuntimeError(
            "v2 quantization requires a C compiler for "
            "tools/ggml_quantize_stdin.c"
        )
    GGML_QUANTIZER.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(GGML_QUANTIZER) + ".tmp")
    command = [
        compiler,
        "-O2",
        "-DNDEBUG",
        "-DGGML_USE_CPU",
        '-DGGML_VERSION="0.17.0"',
        '-DGGML_COMMIT="vendored"',
        f"-I{ROOT_DIR}",
        f"-I{ROOT_DIR / 'ggml' / 'include'}",
        f"-I{ROOT_DIR / 'ggml' / 'src'}",
        str(sources[0]),
        str(sources[1]),
        "-lm",
        "-o",
        str(temporary),
    ]
    if os.uname().sysname == "Linux":
        command.insert(-2, "-ldl")
    try:
        subprocess.run(command, check=True)
        temporary.replace(GGML_QUANTIZER)
    finally:
        temporary.unlink(missing_ok=True)
    if not quantizer_is_usable(GGML_QUANTIZER):
        raise RuntimeError(
            f"built quantizer is not executable on this host: {GGML_QUANTIZER}"
        )
    return GGML_QUANTIZER


def ggml_quantize(tensor: np.ndarray, quant_name: str) -> np.ndarray:
    quantizer = build_ggml_quantizer()
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    rows = int(np.prod(tensor.shape[:-1]))
    columns = tensor.shape[-1]
    process = subprocess.run(
        [quantizer, quant_name, str(rows), str(columns)],
        input=tensor.tobytes(order="C"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            process.stderr.decode("utf-8", errors="replace").strip()
        )
    row_bytes = len(process.stdout) // rows
    if row_bytes * rows != len(process.stdout):
        raise RuntimeError(
            f"invalid quantizer output size {len(process.stdout)} "
            f"for {rows} row(s)"
        )
    return np.frombuffer(process.stdout, dtype=np.uint8).reshape(
        (*tensor.shape[:-1], row_bytes)
    ).copy()


def quantize_tensor(
    name: str,
    source: np.ndarray,
    profile: str,
    keep_f16: list[str],
    keep_q8: list[str],
):
    import gguf

    source = np.ascontiguousarray(source, dtype=np.float32)
    if profile == "f16":
        return (
            trim_ggml_dimensions(source.astype(np.float16)),
            gguf.GGMLQuantizationType.F16,
        )

    force_q8 = starts_with_any(name, keep_q8)
    quant_name = "q4_0" if profile == "q4_0_e" and not force_q8 else "q8_0"
    quantize = (
        (profile == "q8_0" and is_v2_weight(name, source))
        or (profile == "mixed-q8_0" and mixed_quantizable(name, source))
        or (profile == "q4_0_e" and is_v2_weight(name, source))
    )
    if source.ndim < 2 or (
        profile == "mixed-q8_0" and protected_mixed_tensor(name)
    ):
        quantize = False
    if starts_with_any(name, keep_f16):
        quantize = False
    elif force_q8 and profile == "q4_0_e" and is_v2_weight(name, source):
        quantize = True

    if not quantize:
        return (
            trim_ggml_dimensions(source.astype(np.float16)),
            gguf.GGMLQuantizationType.F16,
        )

    if is_conv_weight(name):
        if is_decoder_upsample(name):
            source = source.transpose(1, 2, 0).reshape(
                source.shape[1], source.shape[2] * source.shape[0]
            )
        else:
            source = source.reshape(source.shape[0], -1)
    source = pad_last_dimension(source, 32)
    quant_type = (
        gguf.GGMLQuantizationType.Q4_0
        if quant_name == "q4_0"
        else gguf.GGMLQuantizationType.Q8_0
    )
    return trim_ggml_dimensions(ggml_quantize(source, quant_name)), quant_type


def load_canonical_gguf(path: Path):
    import gguf

    reader = gguf.GGUFReader(path)
    architecture = reader.fields.get("general.architecture")
    symbol_hash = reader.fields.get("inflect.v2.symbol_hash")
    if (
        architecture is None
        or architecture.contents() != "inflect-v2"
        or symbol_hash is None
        or symbol_hash.contents() != SYMBOL_HASH_HEX
    ):
        raise ValueError("input is not a compatible canonical Inflect v2 GGUF")

    metadata = []
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        subtype = field.types[1] if len(field.types) > 1 else None
        metadata.append((key, field.contents(), field.types[0], subtype))

    tensors: dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        if tensor.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError(
                f"canonical input tensor must be F32: {tensor.name} "
                f"({tensor.tensor_type.name})"
            )
        tensors[tensor.name] = np.ascontiguousarray(tensor.data, dtype=np.float32)
    return tensors, metadata


def add_checkpoint_metadata(
    writer,
    tensors: dict[str, np.ndarray],
    config: dict,
    checkpoint_hash: str,
    config_hash: str,
    revision: str,
    source_parameter_count: int,
) -> None:
    if revision:
        writer.add_string("inflect.source.revision", revision)
    writer.add_string("inflect.source.checkpoint_sha256", checkpoint_hash)
    writer.add_string("inflect.source.config_sha256", config_hash)
    writer.add_string("inflect.v2.symbol_hash", SYMBOL_HASH_HEX)
    writer.add_uint32("inflect.v2.symbol_count", 178)
    writer.add_uint64(
        "inflect.v2.parameter_count",
        sum(int(tensor.size) for tensor in tensors.values()),
    )
    writer.add_uint64(
        "inflect.v2.reference_deployed_parameter_count",
        source_parameter_count,
    )
    writer.add_uint32("inflect.v2.source_tensor_count", EXPECTED_TENSORS)
    writer.add_uint32(
        "inflect.v2.folded_weight_norm_pairs", EXPECTED_WEIGHT_NORM_PAIRS
    )

    data = config["data"]
    model = config["model"]
    writer.add_uint32("inflect.v2.sample_rate", data["sampling_rate"])
    writer.add_bool("inflect.v2.add_blank", data["add_blank"])
    for name in (
        "inter_channels", "hidden_channels", "filter_channels", "n_heads",
        "n_layers", "kernel_size", "upsample_initial_channel",
    ):
        writer.add_uint32(f"inflect.v2.{name}", model[name])
    writer.add_array(
        "inflect.v2.resblock_kernel_sizes", model["resblock_kernel_sizes"]
    )
    writer.add_array(
        "inflect.v2.resblock_dilation_sizes",
        [item for group in model["resblock_dilation_sizes"] for item in group],
    )
    writer.add_array("inflect.v2.upsample_rates", model["upsample_rates"])
    writer.add_array(
        "inflect.v2.upsample_kernel_sizes", model["upsample_kernel_sizes"]
    )
    writer.add_string("inflect.v2.activation", "leaky_relu")


def write_gguf(
    output: Path,
    tensors: dict[str, np.ndarray],
    *,
    config: dict | None = None,
    checkpoint_hash: str = "",
    config_hash: str = "",
    revision: str = "",
    source_parameter_count: int = 0,
    metadata=None,
    quantize: str | None = None,
    keep_f16: list[str] | None = None,
    keep_q8: list[str] | None = None,
) -> None:
    import gguf

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output), "inflect-v2")
    if metadata is not None:
        for key, value, value_type, subtype in metadata:
            writer.add_key_value(key, value, value_type, subtype)
    elif config is not None:
        add_checkpoint_metadata(
            writer, tensors, config, checkpoint_hash, config_hash, revision,
            source_parameter_count,
        )
    else:
        raise ValueError("GGUF metadata source is required")

    keep_f16 = keep_f16 or []
    keep_q8 = keep_q8 or []
    if quantize:
        writer.add_uint32("general.quantization_version", 2)
        writer.add_string("inflect.v2.quantization", quantize)
        writer.add_key_value(
            "inflect.v2.quantization_keep_f16",
            ",".join(keep_f16),
            gguf.GGUFValueType.STRING,
        )
        writer.add_key_value(
            "inflect.v2.quantization_keep_q8",
            ",".join(keep_q8),
            gguf.GGUFValueType.STRING,
        )

    type_bytes: dict[str, int] = {}
    for name in sorted(tensors):
        if quantize:
            payload, raw_type = quantize_tensor(
                name, tensors[name], quantize, keep_f16, keep_q8
            )
            writer.add_tensor(name, payload, raw_dtype=raw_type)
            type_name = raw_type.name.lower()
        else:
            payload = tensors[name].astype(np.float32, copy=False)
            writer.add_tensor(name, payload)
            type_name = "f32"
        type_bytes[type_name] = type_bytes.get(type_name, 0) + payload.nbytes
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    summary = " ".join(
        f"{name}={size}" for name, size in sorted(type_bytes.items())
    )
    print(f"wrote {output}: {summary}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--input",
        type=Path,
        help="Canonical folded FP32 GGUF to quantize without PyTorch.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-revision", default="")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--config-sha256")
    parser.add_argument("--quantize", choices=QUANT_PROFILES)
    parser.add_argument("--keep-f16", action="append", default=[], metavar="PREFIX")
    parser.add_argument("--keep-q8", action="append", default=[], metavar="PREFIX")
    args = parser.parse_args(argv)

    if args.input:
        if args.config:
            parser.error("--config is only valid with --checkpoint")
        if not args.quantize:
            parser.error("--input requires --quantize")
        tensors, metadata = load_canonical_gguf(args.input)
        write_gguf(
            args.output,
            tensors,
            metadata=metadata,
            quantize=args.quantize,
            keep_f16=args.keep_f16,
            keep_q8=args.keep_q8,
        )
        return 0

    if not args.config:
        parser.error("--checkpoint requires --config")
    checkpoint_hash = sha256_file(args.checkpoint)
    if args.checkpoint_sha256 and checkpoint_hash != args.checkpoint_sha256.lower():
        raise SystemExit(
            f"checkpoint hash mismatch: expected {args.checkpoint_sha256}, "
            f"got {checkpoint_hash}"
        )
    config_hash = sha256_file(args.config)
    if args.config_sha256 and config_hash != args.config_sha256.lower():
        raise SystemExit(
            f"config hash mismatch: expected {args.config_sha256}, got {config_hash}"
        )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    state = load_state(args.checkpoint)
    if len(state) != EXPECTED_TENSORS:
        raise SystemExit(f"expected {EXPECTED_TENSORS} checkpoint tensors, got {len(state)}")
    if not all(any(name.startswith(prefix) for prefix in REQUIRED_NAMESPACES) for name in state):
        raise SystemExit("checkpoint does not contain all required v2 namespaces")
    tensors = converted_tensors(state)
    source_parameters = sum(int(tensor.numel()) for tensor in state.values())
    deployed = sum(int(array.size) for array in tensors.values())
    if tensors.get("dec.conv_post.weight") is None:
        raise SystemExit("missing bias-free decoder final convolution weight")
    if "dec.conv_post.bias" in tensors:
        raise SystemExit("unexpected decoder final convolution bias")
    write_gguf(
        args.output,
        tensors,
        config=config,
        checkpoint_hash=checkpoint_hash,
        config_hash=config_hash,
        revision=args.reference_revision,
        source_parameter_count=source_parameters,
        quantize=args.quantize,
        keep_f16=args.keep_f16,
        keep_q8=args.keep_q8,
    )
    if not args.quantize:
        print(
            f"canonical folded FP32: {len(tensors)} tensors, "
            f"{deployed} folded parameters; {source_parameters} source parameters"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
