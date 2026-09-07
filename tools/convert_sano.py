#!/usr/bin/env python3
"""Convert a canonical sanoTTS Piperlite F32/F16 GGUF to edge GGUF.

The emitted layout is designed for InflectNanoTTS.cpp's packed ESP32-S3
kernels:

* ordinary Conv1d quantized rows are [out, in * kernel] (channel-major taps),
  padded to 32 elements;
* ConvTranspose1d Q4/Q8 rows are [out, kernel * in] (tap-major channels),
  padded to 32 elements;
* duration-sensitive weights remain F16 in q4_0_e by default;
* explicit Q2_K/Q3_K/Q4_1 overrides use GGML's generic quantized path;
* decoder tensors are physically ordered by runtime stage so ModelLoader's
  prefix selection can use bulk sequential reads instead of scattered seeks.

The source GGUF is expected to use the sanoTTS/audio.cpp tensor names.  The
JSON config is the adjacent sanoTTS Piperlite config.json.  A separate Piper
phoneme config can be supplied when phoneme_id_map is not embedded in it.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path

from sano_common import normalize_language
from v2_common import sha256_file
import numpy as np

QUANT_PROFILES = (
    "f16", "q2_0", "q2_k", "q3_k", "q4_0", "q4_0_e", "q4_1", "q8_0",
)
ARCHITECTURE = "sanotts-piperlite"
LAYOUT_VERSION = 1
FRONT_QUANT_LAYERS = (
    "duration.blocks.0.net.0.weight",
    "duration.blocks.0.net.2.weight",
    "duration.blocks.1.net.0.weight",
    "duration.blocks.1.net.2.weight",
    "duration.blocks.2.net.0.weight",
    "duration.blocks.2.net.2.weight",
    "acoustic.output.weight",
)
FRONT_QUANT_LAYER_SET = frozenset(FRONT_QUANT_LAYERS)
QUANT_OVERRIDE_TYPES = {
    "f16": None,
    "none": None,
    "q2_0": "q2_0",
    "q2_k": "q2_k",
    "q3_k": "q3_k",
    "q4_0": "q4_0",
    "q4_0_e": "q4_0",
    "q4_1": "q4_1",
    "q8_0": "q8_0",
}


def require(mapping: dict, key: str):
    if key not in mapping:
        raise ValueError(f"missing required config key {key!r}")
    return mapping[key]


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
            raise ValueError(f"phoneme id must fit uint8 for SNL1: {symbol!r} -> {value}")
        result[symbol] = value
    for symbol, expected in (("_", 0), ("^", 1), ("$", 2)):
        if result.get(symbol) != expected:
            raise ValueError(
                f"Piper framing symbol {symbol!r} must map to {expected}, "
                f"got {result.get(symbol)!r}"
            )
    return result


def map_hash(id_map: dict[str, int]) -> str:
    payload = json.dumps(
        id_map, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _component_config(payload: dict, name: str) -> dict:
    """Return the small runtime config from either a manifest or flat JSON.

    The shipped Amy/HFC/Kristin packages call this file ``manifest.json`` and
    keep the model configs under ``components``.  Early converter prototypes
    used a flat test config, so accepting both shapes keeps the host tool
    useful for those fixtures without weakening the tensor checks below.
    """
    components = payload.get("components")
    if isinstance(components, dict):
        component = components.get(name)
        if not isinstance(component, dict) or not isinstance(component.get("config"), dict):
            raise ValueError(f"manifest is missing components.{name}.config")
        return dict(component["config"])
    component = payload.get(name)
    if not isinstance(component, dict):
        raise ValueError(f"config is missing {name}")
    return dict(component)


def normalize_config(payload: dict) -> dict:
    """Normalize a sanoTTS manifest into the converter's runtime schema."""
    duration = _component_config(payload, "duration")
    acoustic = _component_config(payload, "acoustic")
    decoder = _component_config(payload, "decoder")

    if duration.get("architecture", "duration_conv") != "duration_conv":
        raise ValueError(
            "unsupported Sano duration architecture: "
            f"{duration.get('architecture')!r} (expected 'duration_conv')"
        )
    if acoustic.get("architecture", "token_context") != "token_context":
        raise ValueError(
            "unsupported Sano acoustic architecture: "
            f"{acoustic.get('architecture')!r} (expected 'token_context')"
        )
    if decoder.get("variant", "piperlite") != "piperlite":
        raise ValueError(
            "unsupported Sano decoder variant: "
            f"{decoder.get('variant')!r} (expected 'piperlite')"
        )
    if str(decoder.get("activation", "leaky_relu")) != "leaky_relu":
        raise ValueError("only leaky_relu Sano decoders are supported")
    if int(decoder.get("res_layers", 1)) != 1:
        raise ValueError("only one residual layer per Sano decoder stage is supported")
    if int(decoder.get("pre_tanh_repair_channels", 0) or 0) > 0:
        raise ValueError("Sano pre_tanh_repair is not supported by this runtime")

    def renamed(source: dict, old: str, new: str) -> None:
        if new not in source and old in source:
            source[new] = source[old]

    renamed(duration, "kernel_size", "kernel")
    renamed(acoustic, "kernel_size", "kernel")

    inference = payload.get("inference")
    if not isinstance(inference, dict):
        inference = {}
    sample_rate = payload.get("sample_rate")
    hop_length = payload.get("hop_length", 256)
    if sample_rate is None:
        audio = payload.get("audio")
        if isinstance(audio, dict):
            sample_rate = audio.get("sample_rate")
    if sample_rate is None:
        sample_rate = 22050

    language = payload.get("language")
    if not language:
        language = payload.get("lang")
    if not language:
        language = (payload.get("espeak") or {}).get("voice", "")

    result = {
        "voice": str(payload.get("voice") or payload.get("name") or "sano-voice"),
        "language": normalize_language(str(language)),
        "sample_rate": int(sample_rate),
        "hop_length": int(hop_length),
        "duration_length_scale": float(
            inference.get("duration_length_scale", payload.get("duration_length_scale", 1.0))
        ),
        "duration": duration,
        "acoustic": acoustic,
        "decoder": decoder,
    }
    if not result["language"]:
        raise ValueError("manifest/config is missing a language")
    return result


def load_frontend_config(config: dict, phoneme_config_path: Path | None):
    piper = {}
    if phoneme_config_path is not None:
        piper = json.loads(phoneme_config_path.read_text(encoding="utf-8"))
    raw_map = (
        config.get("phoneme_id_map")
        or config.get("phoneme_map")
        or piper.get("phoneme_id_map")
    )
    ids = normalize_id_map(raw_map)
    espeak = piper.get("espeak") if isinstance(piper.get("espeak"), dict) else {}
    config_espeak = config.get("espeak")
    if not isinstance(config_espeak, dict):
        config_espeak = {}
    espeak_voice = str(
        config.get("espeak_voice")
        or config_espeak.get("voice")
        or espeak.get("voice")
        or ""
    )
    if not espeak_voice:
        raise ValueError("missing espeak_voice (or phoneme-config espeak.voice)")
    return ids, espeak_voice


def validate_config(config: dict) -> dict[str, object]:
    duration = require(config, "duration")
    acoustic = require(config, "acoustic")
    decoder = require(config, "decoder")
    if not all(isinstance(value, dict) for value in (duration, acoustic, decoder)):
        raise ValueError("duration/acoustic/decoder config entries must be objects")

    channels = list(require(decoder, "channels"))
    if len(channels) != 4 or any(int(value) <= 0 for value in channels):
        raise ValueError("decoder.channels must contain four positive integers")
    branch_masks = []
    for stage in range(3):
        branches = list(require(decoder, f"stage{stage}_branches"))
        mask = 0
        for branch in branches:
            branch = int(branch)
            if branch not in (0, 1, 2):
                raise ValueError(f"invalid stage{stage} residual branch: {branch}")
            mask |= 1 << branch
        if mask == 0:
            raise ValueError(f"decoder stage {stage} has no active branches")
        branch_masks.append(mask)

    def positive(section: dict, key: str) -> int:
        value = int(require(section, key))
        if value <= 0:
            raise ValueError(f"{key} must be positive")
        return value

    values = {
        "voice": str(require(config, "voice")),
        "language": str(require(config, "language")),
        "sample_rate": int(require(config, "sample_rate")),
        "hop_length": int(require(config, "hop_length")),
        "duration_length_scale": float(require(config, "duration_length_scale")),
        "duration_vocab": positive(duration, "vocab_size"),
        "duration_hidden": positive(duration, "hidden"),
        "duration_depth": positive(duration, "depth"),
        "duration_kernel": positive(duration, "kernel"),
        "duration_max_tokens": positive(duration, "max_tokens"),
        "duration_max_duration": positive(duration, "max_duration"),
        "acoustic_vocab": positive(acoustic, "vocab_size"),
        "acoustic_hidden": positive(acoustic, "hidden"),
        "acoustic_token_depth": positive(acoustic, "token_depth"),
        "acoustic_depth": positive(acoustic, "depth"),
        "acoustic_kernel": positive(acoustic, "kernel"),
        "acoustic_out_channels": positive(acoustic, "out_channels"),
        "channels": [int(value) for value in channels],
        "branch_masks": branch_masks,
        "post_filter_channels": int(decoder.get("post_filter_channels", 0)),
        "post_filter_layers": int(decoder.get("post_filter_layers", 0)),
        "post_filter_kernel": int(decoder.get("post_filter_kernel", 9)),
        "post_filter_scale": float(decoder.get("post_filter_scale", 0.0)),
    }
    if values["sample_rate"] <= 0 or values["duration_length_scale"] <= 0:
        raise ValueError("sample rate and duration_length_scale must be positive")
    if values["hop_length"] != 256:
        raise ValueError("Sano Piperlite requires hop_length=256")
    if values["duration_kernel"] % 2 == 0 or values["acoustic_kernel"] % 2 == 0:
        raise ValueError("front kernels must be odd")
    if (values["post_filter_channels"] == 0) != (values["post_filter_layers"] == 0):
        raise ValueError("post-filter channels/layers must both be zero or both positive")
    return values


def _gguf_field_value(reader, key: str, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    try:
        return field.contents()
    except (AttributeError, TypeError):
        return default


def load_canonical(path: Path) -> dict[str, np.ndarray]:
    import gguf

    reader = gguf.GGUFReader(path)
    architecture = _gguf_field_value(reader, "general.architecture")
    graph = _gguf_field_value(reader, "sanotts.graph")
    if architecture != "sanotts" or graph != "piperlite":
        raise ValueError(
            "input is not a canonical sanoTTS Piperlite GGUF: "
            f"general.architecture={architecture!r}, sanotts.graph={graph!r}"
        )
    tensors: dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        if tensor.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError(
                f"canonical source tensor must be F32, got {tensor.tensor_type.name}: "
                f"{tensor.name}"
            )
        tensors[tensor.name] = np.ascontiguousarray(tensor.data, dtype=np.float32)
    if not tensors:
        raise ValueError("source GGUF contains no tensors")
    for prefix in ("duration.", "acoustic.", "decoder."):
        if not any(name.startswith(prefix) for name in tensors):
            raise ValueError(f"source GGUF is missing {prefix} tensors")
    return tensors


def validate_tensors(
    tensors: dict[str, np.ndarray], config: dict[str, object]
) -> None:
    expected: dict[str, tuple[int, ...]] = {}

    def tensor(name: str, shape: tuple[int, ...]) -> None:
        expected[name] = shape

    def conv(name: str, out_ch: int, in_ch: int, kernel: int) -> None:
        tensor(name + ".weight", (out_ch, in_ch, kernel))
        tensor(name + ".bias", (out_ch,))

    dh = int(config["duration_hidden"])
    tensor(
        "duration.embedding.weight",
        (int(config["duration_vocab"]), dh),
    )
    conv("duration.input_proj", dh, dh + 3, 1)
    for block in range(int(config["duration_depth"])):
        root = f"duration.blocks.{block}"
        tensor(root + ".scale", (1,))
        conv(root + ".net.0", dh, dh, int(config["duration_kernel"]))
        conv(root + ".net.2", dh, dh, int(config["duration_kernel"]))
    conv("duration.output", 1, dh, 1)

    ah = int(config["acoustic_hidden"])
    tensor(
        "acoustic.embedding.weight",
        (int(config["acoustic_vocab"]), ah),
    )
    conv("acoustic.token_input_proj", ah, ah + 2, 1)
    for block in range(int(config["acoustic_token_depth"])):
        root = f"acoustic.token_blocks.{block}"
        tensor(root + ".scale", (1,))
        conv(root + ".net.0", ah, ah, int(config["acoustic_kernel"]))
        conv(root + ".net.2", ah, ah, int(config["acoustic_kernel"]))
    conv("acoustic.frame_input_proj", ah, ah + 3, 1)
    for block in range(int(config["acoustic_depth"])):
        root = f"acoustic.frame_blocks.{block}"
        tensor(root + ".scale", (1,))
        conv(root + ".net.0", ah, ah, int(config["acoustic_kernel"]))
        conv(root + ".net.2", ah, ah, int(config["acoustic_kernel"]))
    conv("acoustic.output", int(config["acoustic_out_channels"]), ah, 1)

    channels = [int(value) for value in config["channels"]]
    pre = tensors.get("decoder.pre.weight")
    post = tensors.get("decoder.post.weight")
    if pre is None or pre.ndim != 3 or pre.shape[:2] != (
        channels[0], int(config["acoustic_out_channels"])
    ) or pre.shape[2] <= 0 or pre.shape[2] % 2 == 0:
        raise ValueError(
            f"invalid decoder.pre.weight shape: "
            f"{None if pre is None else pre.shape}"
        )
    conv("decoder.pre", channels[0], int(config["acoustic_out_channels"]), int(pre.shape[2]))

    up_kernels = (16, 16, 8)
    bank_kernels = (3, 5, 7)
    for stage in range(3):
        in_ch = channels[stage]
        out_ch = channels[stage + 1]
        tensor(
            f"decoder.up{stage}.weight",
            (in_ch, out_ch, up_kernels[stage]),
        )
        tensor(f"decoder.up{stage}.bias", (out_ch,))
        mask = int(config["branch_masks"][stage])
        for branch, kernel in enumerate(bank_kernels):
            if not (mask & (1 << branch)):
                continue
            root = f"decoder.res{stage}.0.blocks.{branch}"
            conv(root + ".conv1", out_ch, out_ch, kernel)
            conv(root + ".conv2", out_ch, out_ch, kernel)

    if post is None or post.ndim != 3 or post.shape[:2] != (1, channels[3]) \
            or post.shape[2] <= 0 or post.shape[2] % 2 == 0:
        raise ValueError(
            f"invalid decoder.post.weight shape: "
            f"{None if post is None else post.shape}"
        )
    conv("decoder.post", 1, channels[3], int(post.shape[2]))

    pf = int(config["post_filter_channels"])
    if pf > 0:
        pf_kernel = int(config["post_filter_kernel"])
        conv("decoder.post_filter.in_conv", pf, 1, pf_kernel)
        for layer in range(int(config["post_filter_layers"])):
            root = f"decoder.post_filter.units.{layer}"
            tensor(root + ".scale", (1,))
            conv(root + ".conv1", pf, pf, 3)
            conv(root + ".conv2", pf, pf, 3)
        conv("decoder.post_filter.out_conv", 1, pf, pf_kernel)

    missing = sorted(set(expected) - set(tensors))
    extra = sorted(set(tensors) - set(expected))
    wrong = [
        (name, tensors[name].shape, shape)
        for name, shape in expected.items()
        if name in tensors and tuple(tensors[name].shape) != tuple(shape)
    ]
    if missing or extra or wrong:
        details = []
        if missing:
            details.append(f"missing={missing[:8]}")
        if extra:
            details.append(f"extra={extra[:8]}")
        if wrong:
            details.append(
                "wrong=" + repr(
                    [(name, tuple(got), want) for name, got, want in wrong[:8]]
                )
            )
        raise ValueError("incompatible sanoTTS Piperlite tensor inventory: " + "; ".join(details))


def starts_with_any(name: str, prefixes: list[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def parse_quant_overrides(specs: list[str]) -> list[tuple[str, str | None, str]]:
    """Parse comma-separated pattern/type overrides in command-line order.

    Both ``PATTERN[,PATTERN...]=TYPE`` and
    ``PATTERN=TYPE,PATTERN=TYPE`` are accepted.
    """
    overrides: list[tuple[str, str | None, str]] = []
    for raw_spec in specs:
        clauses = [clause.strip() for clause in raw_spec.split(",")]
        if len(clauses) > 1 and all("=" in clause for clause in clauses):
            entries = clauses
        else:
            entries = [raw_spec]
        for entry in entries:
            if "=" not in entry:
                raise ValueError(
                    "Sano quantization override must be PATTERN[,PATTERN...]=TYPE "
                    "or PATTERN=TYPE,PATTERN=TYPE: "
                    f"{raw_spec}"
                )
            patterns, raw_type = entry.rsplit("=", 1)
            raw_type = raw_type.strip().lower()
            if raw_type not in QUANT_OVERRIDE_TYPES:
                choices = ", ".join(sorted(QUANT_OVERRIDE_TYPES))
                raise ValueError(
                    f"invalid Sano quantization override type {raw_type!r}; "
                    f"choices: {choices}"
                )
            for pattern in patterns.split(","):
                pattern = pattern.strip()
                if not pattern:
                    raise ValueError(f"empty Sano quantization override pattern: {raw_spec}")
                overrides.append((pattern, QUANT_OVERRIDE_TYPES[raw_type], raw_type))
    return overrides


def quant_override(
    name: str, overrides: list[tuple[str, str | None, str]]
) -> tuple[bool, str | None]:
    selected = (False, None)
    for pattern, quant_name, _ in overrides:
        if fnmatch.fnmatchcase(name, pattern):
            selected = (True, quant_name)
    return selected


def is_embedding(name: str) -> bool:
    return name in ("duration.embedding.weight", "acoustic.embedding.weight")


def is_conv_weight(name: str, tensor: np.ndarray) -> bool:
    return name.endswith(".weight") and tensor.ndim == 3 and not is_embedding(name)


def is_conv_transpose(name: str) -> bool:
    return name.startswith(("decoder.up0.", "decoder.up1.", "decoder.up2.")) and \
        name.endswith(".weight")


def q4_e_quant_type(
    name: str, tensor: np.ndarray, quantize_front: bool = False
) -> str | None:
    if not is_conv_weight(name, tensor):
        return None
    if quantize_front and name in FRONT_QUANT_LAYER_SET:
        return "q4_0"
    # Duration is deliberately F16: exp + round converts tiny logit errors to
    # discrete frame changes.  Front interface projections stay conservative;
    # the expensive acoustic residual stack and decoder are the Q4 target.
    if name.startswith("duration."):
        return None
    if name.startswith("acoustic."):
        if ".token_blocks." in name or ".frame_blocks." in name:
            return "q4_0"
        return None
    if name.startswith("decoder.post_filter."):
        return None
    if name.startswith("decoder."):
        return "q4_0"
    return None


def quantize_tensor(
    name: str,
    source: np.ndarray,
    profile: str,
    keep_f16: list[str],
    keep_q8: list[str],
    quantize_front: bool = False,
    overrides: list[tuple[str, str | None, str]] | None = None,
):
    # Keep importing this module dependency-free: the current target/container
    # does not carry numpy/gguf, while --help and static checks should still
    # work.  The host conversion environment supplies these packages.
    from convert_v2 import ggml_quantize, pad_last_dimension, trim_ggml_dimensions
    import gguf

    source = np.ascontiguousarray(source, dtype=np.float32)
    matched_override, quant_name = quant_override(name, overrides or [])
    if not matched_override:
        quant_name = None
        if profile in ("q2_0", "q2_k", "q3_k", "q4_0", "q4_1", "q8_0") and \
                is_conv_weight(name, source):
            quant_name = profile
        elif profile == "q4_0_e":
            quant_name = q4_e_quant_type(name, source, quantize_front)

    if not matched_override:
        if starts_with_any(name, keep_f16):
            quant_name = None
        elif starts_with_any(name, keep_q8) and is_conv_weight(name, source):
            quant_name = "q8_0"

    if quant_name is None:
        return (
            trim_ggml_dimensions(source.astype(np.float16)),
            gguf.GGMLQuantizationType.F16,
        )

    if not is_conv_weight(name, source):
        raise ValueError(
            f"quantization override selected non-convolution tensor {name}; "
            "quantized overrides require a rank-3 .weight tensor"
        )

    if is_conv_transpose(name):
        # [in,out,K] -> [out,K,in] -> [out,K*in]
        source = source.transpose(1, 2, 0).reshape(
            source.shape[1], source.shape[2] * source.shape[0]
        )
    else:
        # [out,in,K] -> [out,in*K].  The packed Conv1d writer gathers one
        # channel's K taps contiguously, matching this row ordering.
        source = source.reshape(source.shape[0], -1)
    block_size = (
        256 if quant_name in ("q2_k", "q3_k")
        else 64 if quant_name == "q2_0"
        else 32
    )
    source = pad_last_dimension(source, block_size)
    raw_type = {
        "q2_0": gguf.GGMLQuantizationType.Q2_0,
        "q2_k": gguf.GGMLQuantizationType.Q2_K,
        "q3_k": gguf.GGMLQuantizationType.Q3_K,
        "q4_0": gguf.GGMLQuantizationType.Q4_0,
        "q4_1": gguf.GGMLQuantizationType.Q4_1,
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
    }[quant_name]
    return trim_ggml_dimensions(ggml_quantize(source, quant_name)), raw_type


def runtime_group(name: str) -> tuple[int, str]:
    if name.startswith("duration."):
        return 0, name
    if name.startswith("acoustic."):
        return 1, name
    if name.startswith("decoder.pre."):
        return 2, name
    for stage in range(3):
        if name.startswith((f"decoder.up{stage}.", f"decoder.res{stage}.")):
            return 3 + stage, name
    if name.startswith(("decoder.post.", "decoder.post_filter.")):
        return 6, name
    return 7, name


def tensor_kernel(tensors: dict[str, np.ndarray], name: str) -> int:
    tensor = tensors.get(name)
    if tensor is None or tensor.ndim != 3:
        raise ValueError(f"missing/rank-invalid convolution: {name}")
    return int(tensor.shape[2])


def add_metadata(
    writer,
    tensors: dict[str, np.ndarray],
    config: dict[str, object],
    id_map: dict[str, int],
    espeak_voice: str,
    source_hash: str,
    config_hash: str,
    frontend_hash: str,
    profile: str,
    keep_f16: list[str],
    keep_q8: list[str],
    quantize_front: bool,
    overrides: list[tuple[str, str | None, str]],
) -> None:
    writer.add_uint32("sanotts.layout_version", LAYOUT_VERSION)
    writer.add_string("sanotts.voice", config["voice"])
    writer.add_string("sanotts.language", config["language"])
    writer.add_string("sanotts.espeak_voice", espeak_voice)
    writer.add_string("sanotts.frontend.map_hash", frontend_hash)
    writer.add_uint32("sanotts.sample_rate", config["sample_rate"])
    writer.add_uint32("sanotts.hop_length", config["hop_length"])
    writer.add_float32(
        "sanotts.duration_length_scale", config["duration_length_scale"]
    )

    for key, value in (
        ("vocab_size", config["duration_vocab"]),
        ("hidden", config["duration_hidden"]),
        ("depth", config["duration_depth"]),
        ("kernel", config["duration_kernel"]),
        ("max_tokens", config["duration_max_tokens"]),
        ("max_duration", config["duration_max_duration"]),
    ):
        writer.add_uint32(f"sanotts.duration.{key}", value)
    for key, value in (
        ("vocab_size", config["acoustic_vocab"]),
        ("hidden", config["acoustic_hidden"]),
        ("token_depth", config["acoustic_token_depth"]),
        ("depth", config["acoustic_depth"]),
        ("kernel", config["acoustic_kernel"]),
        ("out_channels", config["acoustic_out_channels"]),
    ):
        writer.add_uint32(f"sanotts.acoustic.{key}", value)

    for index, channel in enumerate(config["channels"]):
        writer.add_uint32(f"sanotts.decoder.channel{index}", channel)
    writer.add_uint32(
        "sanotts.decoder.pre_kernel", tensor_kernel(tensors, "decoder.pre.weight")
    )
    writer.add_uint32(
        "sanotts.decoder.post_kernel", tensor_kernel(tensors, "decoder.post.weight")
    )
    strides = (8, 8, 4)
    paddings = (4, 4, 2)
    for stage in range(3):
        writer.add_uint32(
            f"sanotts.decoder.up{stage}_kernel",
            tensor_kernel(tensors, f"decoder.up{stage}.weight"),
        )
        writer.add_uint32(f"sanotts.decoder.up{stage}_stride", strides[stage])
        writer.add_uint32(f"sanotts.decoder.up{stage}_padding", paddings[stage])
        writer.add_uint32(
            f"sanotts.decoder.stage{stage}_branch_mask",
            config["branch_masks"][stage],
        )
    writer.add_uint32(
        "sanotts.decoder.post_filter_channels", config["post_filter_channels"]
    )
    writer.add_uint32(
        "sanotts.decoder.post_filter_layers", config["post_filter_layers"]
    )
    writer.add_uint32(
        "sanotts.decoder.post_filter_kernel", config["post_filter_kernel"]
    )
    writer.add_float32(
        "sanotts.decoder.post_filter_scale", config["post_filter_scale"]
    )

    punctuation = {
        "space_id": " ",
        "punc.apostrophe": "'",
        "punc.bang": "!",
        "punc.lparen": "(",
        "punc.rparen": ")",
        "punc.comma": ",",
        "punc.dash": "-",
        "punc.period": ".",
        "punc.colon": ":",
        "punc.semicolon": ";",
        "punc.question": "?",
        "punc.quote": '"',
    }
    for key, symbol in punctuation.items():
        value = id_map.get(symbol, -1)
        writer.add_uint32(
            f"sanotts.frontend.{key}", value if value >= 0 else 0xFFFFFFFF
        )

    writer.add_string("sanotts.source.gguf_sha256", source_hash)
    writer.add_string("sanotts.source.config_sha256", config_hash)
    writer.add_uint64(
        "sanotts.parameter_count", sum(int(value.size) for value in tensors.values())
    )
    writer.add_uint32("general.quantization_version", 2)
    writer.add_string("sanotts.quantization", profile)
    writer.add_string("sanotts.quantization_keep_f16", ",".join(keep_f16))
    writer.add_string("sanotts.quantization_keep_q8", ",".join(keep_q8))
    writer.add_bool("sanotts.quantization_front", quantize_front)
    writer.add_string(
        "sanotts.quantization_overrides",
        ",".join(f"{pattern}={raw_type}" for pattern, _, raw_type in overrides),
    )


def write_gguf(
    output: Path,
    tensors: dict[str, np.ndarray],
    config: dict[str, object],
    id_map: dict[str, int],
    espeak_voice: str,
    source_hash: str,
    config_hash: str,
    frontend_hash: str,
    profile: str,
    keep_f16: list[str],
    keep_q8: list[str],
    quantize_front: bool,
    overrides: list[tuple[str, str | None, str]],
) -> None:
    import gguf

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output), ARCHITECTURE)
    add_metadata(
        writer, tensors, config, id_map, espeak_voice, source_hash,
        config_hash, frontend_hash, profile, keep_f16, keep_q8, quantize_front,
        overrides,
    )

    totals: dict[str, int] = {}
    for name in sorted(tensors, key=runtime_group):
        payload, raw_type = quantize_tensor(
            name, tensors[name], profile, keep_f16, keep_q8, quantize_front,
            overrides,
        )
        writer.add_tensor(name, payload, raw_dtype=raw_type)
        totals[raw_type.name] = totals.get(raw_type.name, 0) + payload.nbytes
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    summary = " ".join(
        f"{kind.lower()}={size}" for kind, size in sorted(totals.items())
    )
    print(f"wrote {output}: {summary}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="sanoTTS Piperlite canonical F32/F16 GGUF")
    parser.add_argument("--config", type=Path, required=True,
                        help="sanoTTS Piperlite manifest.json (or flat config JSON)")
    parser.add_argument("--phoneme-config", type=Path,
                        help="Piper phoneme config when config.json omits phoneme_id_map")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantize", choices=QUANT_PROFILES, default="q4_0_e")
    parser.add_argument(
        "--quantize-front", action="store_true",
        help="also quantize the seven duration/acoustic front tensors in q4_0_e",
    )
    parser.add_argument(
        "--quantize-override", action="append", default=[], metavar="PATTERNS=TYPE",
        help=(
            "override tensor quantization; use PATTERN[,PATTERN...]=TYPE or "
            "PATTERN=TYPE,PATTERN=TYPE; TYPE is f16/none, q2_0, q2_k, q3_k, q4_0, "
            "q4_1, or q8_0; repeatable "
            "and later matches win"
        ),
    )
    parser.add_argument("--keep-f16", action="append", default=[], metavar="PREFIX")
    parser.add_argument("--keep-q8", action="append", default=[], metavar="PREFIX")
    args = parser.parse_args(argv)
    overrides = parse_quant_overrides(args.quantize_override)

    config_json = json.loads(args.config.read_text(encoding="utf-8"))
    normalized_json = normalize_config(config_json)
    config = validate_config(normalized_json)
    phoneme_config = args.phoneme_config
    if phoneme_config is None:
        sibling = args.config.with_name("piper-phoneme-config.json")
        if sibling.is_file():
            phoneme_config = sibling
    id_map, espeak_voice = load_frontend_config(config_json, phoneme_config)
    frontend_hash = map_hash(id_map)
    tensors = load_canonical(args.input)
    validate_tensors(tensors, config)
    write_gguf(
        args.output,
        tensors,
        config,
        id_map,
        espeak_voice,
        sha256_file(args.input),
        sha256_file(args.config),
        frontend_hash,
        args.quantize,
        args.keep_f16,
        args.keep_q8,
        args.quantize_front,
        overrides,
    )
    print(f"frontend map hash: {frontend_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
