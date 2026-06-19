#!/usr/bin/env python3
"""Convert Inflect-Nano (Acoustic + Vocoder) PyTorch checkpoints to GGUF format.

Reads acoustic.pt and vocoder.pt, transposes/folds weights to match the
C++ GGML implementation, and writes two separate .gguf files.

Usage:
    python convert.py --acoustic acoustic.pt --vocoder vocoder.pt --out_dir models/
    python convert.py --acoustic acoustic.pt --vocoder vocoder.pt --out_dir models/ --quantize q8_0
    python convert.py --acoustic acoustic.pt --vocoder vocoder.pt --out_dir models/ --quantize q4_0 --vocoder-quantize f16
"""

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import torch
import numpy as np
import gguf

QUANT_MAP = {}


def add_quant_alias(name, enum_name):
    quant_type = getattr(gguf.GGMLQuantizationType, enum_name, None)
    if quant_type is not None:
        QUANT_MAP[name] = quant_type
    return quant_type


Q1_0_TYPE = add_quant_alias("q1_0", "Q1_0")
if Q1_0_TYPE is not None:
    QUANT_MAP["q1_k"] = Q1_0_TYPE

QUANT_MAP.update({
    "q2_k": gguf.GGMLQuantizationType.Q2_K,
    "q3_k": gguf.GGMLQuantizationType.Q3_K,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
    "q4_k": gguf.GGMLQuantizationType.Q4_K,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q5_k": gguf.GGMLQuantizationType.Q5_K,
    "q6_k": gguf.GGMLQuantizationType.Q6_K,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "f16": gguf.GGMLQuantizationType.F16,
})

QUANT_BLOCK_SIZE = {
    gguf.GGMLQuantizationType.Q2_K: 256,
    gguf.GGMLQuantizationType.Q3_K: 256,
    gguf.GGMLQuantizationType.Q4_0: 32,
    gguf.GGMLQuantizationType.Q5_0: 32,
    gguf.GGMLQuantizationType.Q8_0: 32,
    # K-quants are stored in super-blocks. Padding K to 256 keeps gguf
    # quantization from rejecting hidden sizes like 168.
    gguf.GGMLQuantizationType.Q4_K: 256,
    gguf.GGMLQuantizationType.Q5_K: 256,
    gguf.GGMLQuantizationType.Q6_K: 256,
    gguf.GGMLQuantizationType.F16: 1,
}
if Q1_0_TYPE is not None:
    QUANT_BLOCK_SIZE[Q1_0_TYPE] = 128

F32_QUANT_NAMES = {"none", "f32"}
EXTERNAL_QUANT_NAMES = {"q1_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"}
QUANT_NAME_BY_TYPE = {}
for _name, _type in QUANT_MAP.items():
    QUANT_NAME_BY_TYPE.setdefault(_type, _name)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GGML_QUANTIZER = os.path.join(ROOT_DIR, "build", "tools", "ggml_quantize_stdin")


class QuantOverride:
    def __init__(self, pattern, quant_type, raw_quant):
        self.pattern = pattern
        self.quant_type = quant_type
        self.raw_quant = raw_quant
        self.count = 0
        if pattern.startswith("re:"):
            self.regex = re.compile(pattern[3:])
            self.glob = None
        else:
            self.regex = None
            self.glob = pattern

    def matches(self, *names):
        for name in names:
            if self.regex is not None:
                if self.regex.search(name):
                    return True
            elif fnmatch.fnmatchcase(name, self.glob):
                return True
        return False


def parse_quant_name(name, where):
    q = name.strip().lower()
    if q in F32_QUANT_NAMES:
        return None
    if q not in QUANT_MAP:
        choices = ", ".join(sorted([*QUANT_MAP.keys(), *F32_QUANT_NAMES]))
        raise ValueError(f"invalid {where} quantization '{name}'; choices: {choices}")
    return QUANT_MAP[q]


def parse_quant_overrides(specs, where):
    overrides = []
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"{where} override must be PATTERN=TYPE: {spec}")
        pattern, quant_name = spec.rsplit("=", 1)
        pattern = pattern.strip()
        quant_name = quant_name.strip()
        if not pattern or not quant_name:
            raise ValueError(f"{where} override must be PATTERN=TYPE: {spec}")
        overrides.append(QuantOverride(pattern, parse_quant_name(quant_name, where), quant_name.lower()))
    return overrides


def read_override_files(paths):
    specs = []
    for path in paths or []:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                prefix = "--acoustic-quantize-override "
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                prefix = "--vocoder-quantize-override "
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                if (
                    (line.startswith("'") and line.endswith("'"))
                    or (line.startswith('"') and line.endswith('"'))
                ):
                    line = line[1:-1].strip()
                # Also handle shell-ready snippets such as:
                # --acoustic-quantize-override 'encoder.0.ff.0.weight=q4_0'
                if len(line) >= 2 and line[0] == line[-1] and line[0] in {"'", '"'}:
                    line = line[1:-1]
                specs.append(line)
    return specs


def resolve_quant_type(name, default_type, overrides, *aliases):
    quant_type = default_type
    for override in overrides or []:
        if override.matches(name, *aliases):
            quant_type = override.quant_type
            override.count += 1
    return quant_type


def has_block_quant(default_type, overrides):
    if default_type and default_type != gguf.GGMLQuantizationType.F16:
        return True
    return any(o.quant_type and o.quant_type != gguf.GGMLQuantizationType.F16 for o in overrides or [])


def print_override_summary(label, overrides):
    for override in overrides or []:
        print(f"{label} override {override.pattern}={override.raw_quant} matched {override.count} tensor(s)")


class TensorStats:
    def __init__(self, label):
        self.label = label
        self.counts = {}
        self.bytes = {}
        self.fallbacks = []
        self.quantized = 0
        self.requested = 0

    def add(self, dtype, nbytes, requested=False, fallback_from=None, name=None, reason=None):
        self.counts[dtype] = self.counts.get(dtype, 0) + 1
        self.bytes[dtype] = self.bytes.get(dtype, 0) + int(nbytes)
        if requested:
            self.requested += 1
        if fallback_from:
            self.fallbacks.append((name, fallback_from, dtype, reason))
        elif requested and dtype not in {"F16", "F32"}:
            self.quantized += 1

    def print(self):
        total = sum(self.counts.values())
        total_bytes = sum(self.bytes.values())
        print(f"{self.label} tensor dtypes:")
        for dtype in sorted(self.counts):
            pct = (100.0 * self.counts[dtype] / total) if total else 0.0
            mb = self.bytes[dtype] / (1024 * 1024)
            print(f"  {dtype}: {self.counts[dtype]} tensor(s), {pct:.1f}%, {mb:.2f} MiB")
        if self.requested:
            print(f"  requested quantization: {self.quantized}/{self.requested} tensor(s) block-quantized")
        if self.fallbacks:
            print(f"  quantization fallbacks: {len(self.fallbacks)} tensor(s)")
            for name, src, dst, reason in self.fallbacks[:20]:
                print(f"    {name}: {src} -> {dst} ({reason})")
            if len(self.fallbacks) > 20:
                print(f"    ... {len(self.fallbacks) - 20} more")


def to_numpy(t):
    return t.detach().cpu().float().numpy()


def fold_wn(weight_v, weight_g):
    """Folds PyTorch weight_norm (v and g) into a single weight tensor."""
    v = to_numpy(weight_v)
    g = to_numpy(weight_g)
    dims = list(range(1, v.ndim))
    norm = np.linalg.norm(v, axis=tuple(dims), keepdims=True)
    return v * (g / (norm + 1e-8))


def quant_block_size(quant_type):
    return QUANT_BLOCK_SIZE.get(quant_type, 32) if quant_type else 1


def quant_type_name(quant_type):
    if quant_type is None:
        return "f32"
    return QUANT_NAME_BY_TYPE.get(quant_type, str(quant_type).split(".")[-1].lower())


def is_k_quant(quant_type):
    return quant_type_name(quant_type) in EXTERNAL_QUANT_NAMES


def pad_dim(arr, dim, multiple):
    if multiple <= 1:
        return arr
    size = arr.shape[dim]
    padded = ((size + multiple - 1) // multiple) * multiple
    if padded == size:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[dim] = (0, padded - size)
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def build_ggml_quantizer():
    sources = [
        os.path.join(ROOT_DIR, "tools", "ggml_quantize_stdin.c"),
        os.path.join(ROOT_DIR, "ggml", "src", "ggml-quants.c"),
        os.path.join(ROOT_DIR, "ggml", "src", "ggml-quants.h"),
        os.path.join(ROOT_DIR, "ggml", "include", "ggml.h"),
    ]
    if os.path.exists(GGML_QUANTIZER) and all(os.path.getmtime(GGML_QUANTIZER) >= os.path.getmtime(src) for src in sources):
        return GGML_QUANTIZER
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("K-quant requested but no C compiler was found to build tools/ggml_quantize_stdin.c")
    os.makedirs(os.path.dirname(GGML_QUANTIZER), exist_ok=True)
    cmd = [
        cc,
        "-O2",
        "-DNDEBUG",
        "-DGGML_USE_CPU",
        '-DGGML_VERSION="0.15.1"',
        '-DGGML_COMMIT="vendored"',
        "-I" + ROOT_DIR,
        "-I" + os.path.join(ROOT_DIR, "ggml", "include"),
        "-I" + os.path.join(ROOT_DIR, "ggml", "src"),
        sources[0],
        sources[1],
        "-lm",
        "-o",
        GGML_QUANTIZER,
    ]
    if os.uname().sysname == "Linux":
        cmd.insert(-2, "-ldl")
    subprocess.run(cmd, check=True)
    return GGML_QUANTIZER


def ggml_quantize(arr, quant_type):
    qname = quant_type_name(quant_type)
    quantizer = build_ggml_quantizer()
    rows = int(np.prod(arr.shape[:-1]))
    cols = int(arr.shape[-1])
    proc = subprocess.run(
        [quantizer, qname, str(rows), str(cols)],
        input=np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace").strip())
    row_bytes = len(proc.stdout) // rows
    if row_bytes * rows != len(proc.stdout):
        raise RuntimeError(f"invalid quantizer output size {len(proc.stdout)} for {rows} row(s)")
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape((*arr.shape[:-1], row_bytes)).copy()


def is_graph_linear_weight(name):
    """Weights consumed by GGML mul_mat in acoustic graphs."""
    if not name.endswith(".weight"):
        return False
    if name == "speaker_proj.weight":
        return True
    if name.endswith("point.weight"):
        return True
    graph_prefixes = (
        "encoder.",
        "decoder.",
        "duration_head.",
        "energy_head.",
        "bright_head.",
        "pitch_head.",
        "mel_head.",
    )
    if not name.startswith(graph_prefixes):
        return False
    return (
        name.endswith("ff.0.weight")
        or name.endswith("ff.3.weight")
        or name.endswith("head.1.weight")
        or name.endswith("head.3.weight")
        or name == "mel_head.1.weight"
        or name == "mel_head.3.weight"
    )


def is_embedding_weight(name):
    return name in {"phone.weight", "tone.weight", "lang.weight", "speaker.weight", "abs_frame.weight"}


def is_gru_weight(name):
    return name.startswith("frame_gru.weight")


def is_depthwise_weight(name):
    return name.endswith("depth.weight")


def is_postnet_weight(name):
    return name.startswith("postnet.") and name.endswith(".weight")


def is_bridge_linear_weight(name):
    return (
        name in {
            "energy_proj.weight",
            "bright_proj.weight",
            "pitch_proj.0.weight",
            "pitch_proj.2.weight",
            "frame_proj.0.weight",
            "frame_proj.2.weight",
            "local_ctx.0.weight",
            "local_ctx.2.weight",
        }
    )


def parse_scope_list(value):
    scopes = set()
    for item in (value or "graph").split(","):
        item = item.strip()
        if item:
            scopes.add(item)
    if "all" in scopes:
        scopes.update({"graph", "embeddings", "gru", "bridge", "depthwise", "postnet"})
    valid = {"graph", "embeddings", "gru", "bridge", "depthwise", "postnet", "all"}
    unknown = sorted(scopes - valid)
    if unknown:
        raise ValueError(f"invalid acoustic quantize scope(s): {', '.join(unknown)}")
    return scopes


def add_tensor(writer, name, arr, quant_type=None, stats=None, unquantized_weight_dtype="f16"):
    """Adds a tensor to the GGUF writer, applying quantization if possible."""
    arr = np.ascontiguousarray(arr)
    requested = bool(quant_type and "log_alpha" not in name and arr.size > 1)

    # Snake alphas and scalars must remain F32 for precision
    if requested:
        block = quant_block_size(quant_type)
        # GGUF reverses numpy dimensions into GGML ne[] order. Quantized
        # tensors require GGML ne[0], i.e. numpy's last dimension, to align.
        if arr.ndim > 1 and arr.shape[-1] % block == 0:
            try:
                if quant_type == gguf.GGMLQuantizationType.F16:
                    writer.add_tensor(name, arr.astype(np.float16), raw_dtype=quant_type)
                    if stats:
                        stats.add("F16", arr.size * 2, requested=True)
                else:
                    qbytes = ggml_quantize(arr, quant_type) if is_k_quant(quant_type) else gguf.quants.quantize(arr, quant_type)
                    writer.add_tensor(name, qbytes, raw_dtype=quant_type)
                    if stats:
                        stats.add(quant_type_name(quant_type).upper(), qbytes.nbytes, requested=True)
                return
            except Exception as e:
                print(f"Warning: could not quantize {name} to {quant_type}: {e}")

        # Fallback to F16 if quantization is requested but dimension doesn't align
        writer.add_tensor(name, arr.astype(np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
        if stats:
            reason = f"last dim {arr.shape[-1]} not multiple of {block}" if not (arr.ndim > 1 and arr.shape[-1] % block == 0) else "quantizer failed"
            stats.add("F16", arr.size * 2, requested=True, fallback_from=quant_type_name(quant_type).upper(), name=name, reason=reason)
        return

    if name.endswith(".weight") and arr.ndim > 1 and "log_alpha" not in name and unquantized_weight_dtype == "f16":
        writer.add_tensor(name, arr.astype(np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
        if stats:
            stats.add("F16", arr.size * 2)
        return

    writer.add_tensor(name, arr.astype(np.float32))
    if stats:
        stats.add("F32", arr.size * 4)


def is_quantizable(name, scopes):
    """Determine if a tensor should be quantized based on its name."""
    return (
        ("graph" in scopes and is_graph_linear_weight(name))
        or ("embeddings" in scopes and is_embedding_weight(name))
        or ("gru" in scopes and is_gru_weight(name))
        or ("bridge" in scopes and is_bridge_linear_weight(name))
        or ("depthwise" in scopes and is_depthwise_weight(name))
        or ("postnet" in scopes and is_postnet_weight(name))
    )


def is_unsafe_block_quant_layout(name):
    # Depthwise weights store kernel as GGML ne[0]. Block quantization would
    # pad a 7-tap kernel to 32 and change convolution behavior.
    return is_depthwise_weight(name) or is_postnet_weight(name)


def strip_generator_prefix(name):
    return name[len("generator.") :] if name.startswith("generator.") else name


def is_vocoder_conv1d_weight(bare_name):
    return bare_name in {"conv_pre.weight", "conv_post.weight"} or "convs1." in bare_name or "convs2." in bare_name


def flatten_vocoder_conv1d_weight(arr, quant_type):
    # PyTorch Conv1d is [out, in, K]. Store as a GGML matrix
    # [in*K padded, out] so quantized mul_mat can consume im2col output.
    flat = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])
    return pad_dim(flat, -1, quant_block_size(quant_type))


def process_acoustic(pt_path, out_path, quant_type=None, quant_overrides=None, unquantized_weight_dtype="f16", quant_scopes=None):
    print(f"Loading acoustic model from {pt_path}...")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_state = ckpt["model"]

    w = gguf.GGUFWriter(out_path, "inflect-acoustic")
    quant_overrides = quant_overrides or []
    quant_scopes = quant_scopes or {"graph"}
    stats = TensorStats("Acoustic")
    if has_block_quant(quant_type, quant_overrides):
        w.add_uint32("general.quantization_version", 2)

    # Write metadata
    w.add_uint32("vocab_size", cfg["vocab_size"])
    w.add_uint32("tone_size", cfg["tone_size"])
    w.add_uint32("lang_size", cfg["lang_size"])
    w.add_uint32("n_mels", cfg["n_mels"])
    w.add_uint32("hidden", cfg["hidden"])
    w.add_uint32("encoder_layers", cfg["encoder_layers"])
    w.add_uint32("decoder_layers", cfg["decoder_layers"])
    w.add_uint32("encoder_ff_mult", cfg.get("encoder_ff_mult", 4))
    w.add_uint32("decoder_ff_mult", cfg.get("decoder_ff_mult", 3))
    w.add_uint32("kernel_size", cfg["kernel_size"])
    w.add_uint32("speaker_count", cfg["speaker_count"])
    w.add_uint32("speaker_dim", cfg["speaker_dim"])
    w.add_uint32("abs_frame_bins", cfg["abs_frame_bins"])
    w.add_uint32("sample_rate", cfg["sample_rate"])
    w.add_uint32("max_frames", cfg.get("max_frames", 1400))
    w.add_float32("postnet_scale", cfg["postnet_scale"])

    H = cfg["hidden"]

    for name, tensor in model_state.items():
        arr = to_numpy(tensor)
        can_quantize = is_quantizable(name, quant_scopes)
        qt = resolve_quant_type(name, quant_type, quant_overrides) if can_quantize else None
        if qt and qt != gguf.GGMLQuantizationType.F16 and is_unsafe_block_quant_layout(name):
            print(f"Warning: keeping {name} F16; block quantization would pad the convolution kernel axis")
            qt = gguf.GGMLQuantizationType.F16

        # GGUF reverses numpy dimensions into GGML ne[] order. For a PyTorch
        # Linear [out, in], writing it unchanged loads as GGML [in, out].
        # For Conv1d [out, in, K], writing it unchanged loads as [K, in, out].

        # 1. Linear layers
        if (
            name.endswith("ff.0.weight")
            or name.endswith("ff.3.weight")
            or name.endswith("head.1.weight")
            or name.endswith("head.3.weight")
            or name.endswith("proj.0.weight")
            or name.endswith("proj.2.weight")
            or name == "speaker_proj.weight"
            or name == "mel_head.1.weight"
            or name == "mel_head.3.weight"
            or name == "abs_frame.weight"
            or name == "local_ctx.0.weight"
            or name == "local_ctx.2.weight"
            or name.startswith("frame_gru.weight")
        ):
            if can_quantize:
                arr = pad_dim(arr, -1, quant_block_size(qt))

        # 2. Pointwise Conv1d 1x1
        elif name.endswith("point.weight"):
            arr = arr.squeeze(-1)
            if is_graph_linear_weight(name):
                arr = pad_dim(arr, -1, quant_block_size(qt))

        # 3. Depthwise Conv1d: [2*H, 1, K] -> store [2, H, K], load [K, H, 2]
        elif name.endswith("depth.weight"):
            arr = arr.reshape(2, H, -1)
            if can_quantize:
                arr = pad_dim(arr, -1, quant_block_size(qt))

        # 4. Postnet Conv1d
        elif name.startswith("postnet.") and name.endswith(".weight"):
            if can_quantize:
                arr = pad_dim(arr, -1, quant_block_size(qt))

        # 5. Token embeddings
        elif name in ["phone.weight", "tone.weight", "lang.weight"]:
            if can_quantize:
                arr = pad_dim(arr, -1, quant_block_size(qt))

        # 6. Speaker embedding is [speaker_count, speaker_dim] in PyTorch;
        # writing unchanged loads as GGML [speaker_dim, speaker_count].
        elif name == "speaker.weight":
            if can_quantize:
                arr = pad_dim(arr, -1, quant_block_size(qt))

        add_tensor(w, name, arr, qt, stats, unquantized_weight_dtype)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print_override_summary("Acoustic", quant_overrides)
    stats.print()
    print(f"Wrote {out_path}")


def process_vocoder(pt_path, out_path, quant_type=None, quant_scope="all", quant_overrides=None):
    print(f"Loading vocoder model from {pt_path}...")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    generator_state = ckpt["generator"]

    w = gguf.GGUFWriter(out_path, "inflect-vocoder")
    quant_overrides = quant_overrides or []
    stats = TensorStats("Vocoder")
    if has_block_quant(quant_type, quant_overrides):
        w.add_uint32("general.quantization_version", 2)

    # Write metadata
    w.add_uint32("sample_rate", cfg["sample_rate"])
    w.add_uint32("num_mels", cfg["num_mels"])
    w.add_uint32("upsample_initial_channel", cfg["upsample_initial_channel"])
    w.add_string("activation", cfg["activation"])
    w.add_array("upsample_rates", list(cfg["upsample_rates"]))
    w.add_array("upsample_kernel_sizes", list(cfg["upsample_kernel_sizes"]))
    w.add_array("resblock_kernel_sizes", list(cfg["resblock_kernel_sizes"]))

    # Flatten resblock dilations for GGUF array
    dilations = [int(x) for d in cfg["resblock_dilation_sizes"] for x in d]
    w.add_array("resblock_dilation_sizes", dilations)

    emitted_weights = set()

    def should_quantize_vocoder_conv(bare_name):
        if not quant_type or quant_type == gguf.GGMLQuantizationType.F16:
            return False
        if not is_vocoder_conv1d_weight(bare_name):
            return False
        if quant_scope == "all":
            return True
        if quant_scope == "resblocks":
            return "convs1." in bare_name or "convs2." in bare_name
        if quant_scope == "no_post":
            return bare_name != "conv_post.weight"
        if quant_scope == "pre_post":
            return bare_name in {"conv_pre.weight", "conv_post.weight"}
        return False

    def emit_weight(name, arr):
        bare_name = strip_generator_prefix(name)
        default_qt = quant_type if should_quantize_vocoder_conv(bare_name) else None
        can_quantize = is_vocoder_conv1d_weight(bare_name)
        qt = resolve_quant_type(name, default_qt, quant_overrides, bare_name) if can_quantize else None

        # GGUF reverses numpy dimensions into GGML ne[] order. Keep PyTorch
        # Conv1d [out, in, K] and ConvTranspose1d [in, out, K] contiguous.
        if is_vocoder_conv1d_weight(bare_name) and qt and qt != gguf.GGMLQuantizationType.F16:
            arr = flatten_vocoder_conv1d_weight(arr, qt)
            add_tensor(w, name, arr, qt, stats, "f16")
            emitted_weights.add(name)
            return
        elif bare_name.startswith("ups.") or ".ups." in bare_name:
            pass

        # Keep vocoder convolutions F16. GGML conv kernels in the host runtime
        # do not accept GGUF block-quantized conv kernels here.
        qt = None
        if qt is None:
            w.add_tensor(name, np.ascontiguousarray(arr).astype(np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
            stats.add("F16", arr.size * 2)
        else:
            add_tensor(w, name, arr, qt, stats, "f16")
        emitted_weights.add(name)

    for name, tensor in generator_state.items():
        # Skip weight_norm parameters, we will fold them
        if ".weight_v" in name or ".weight_g" in name:
            continue

        # Squeeze Snake activation alphas: [1, C, 1] -> [C]
        if name.endswith(".log_alpha"):
            arr = to_numpy(tensor).squeeze()
            add_tensor(w, name, arr, stats=stats)
            continue

        if name.endswith(".weight"):
            base_name = name[: -len(".weight")]
            v_name = base_name + ".weight_v"
            g_name = base_name + ".weight_g"

            if v_name in generator_state and g_name in generator_state:
                arr = fold_wn(generator_state[v_name], generator_state[g_name])
            else:
                arr = to_numpy(tensor)

            emit_weight(name, arr)
        else:
            arr = to_numpy(tensor)
            add_tensor(w, name, arr, stats=stats)

    # Weight-normalized checkpoints often contain only .weight_v/.weight_g,
    # not a materialized .weight tensor. Emit folded weights for those pairs.
    for v_name in sorted(k for k in generator_state if k.endswith(".weight_v")):
        base_name = v_name[: -len(".weight_v")]
        g_name = base_name + ".weight_g"
        weight_name = base_name + ".weight"
        if weight_name in emitted_weights or g_name not in generator_state:
            continue
        arr = fold_wn(generator_state[v_name], generator_state[g_name])
        emit_weight(weight_name, arr)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print_override_summary("Vocoder", quant_overrides)
    stats.print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Inflect-Nano PyTorch models to GGUF")
    parser.add_argument("--acoustic", default=None, help="Path to inflect_nano_v1_acoustic.pt")
    parser.add_argument("--vocoder", default=None, help="Path to inflect_nano_v1_vocoder.pt")
    parser.add_argument("--out_dir", default=".", help="Output directory for .gguf files")
    parser.add_argument("--skip-acoustic", action="store_true", help="Do not write inflect_acoustic.gguf.")
    parser.add_argument("--skip-vocoder", action="store_true", help="Do not write inflect_vocoder.gguf.")
    parser.add_argument(
        "--quantize",
        default=None,
        choices=sorted(QUANT_MAP.keys()),
        help="Acoustic quantization type (e.g., q8_0, q4_k). Falls back to F16 for unaligned dims.",
    )
    parser.add_argument(
        "--vocoder-quantize",
        default=None,
        choices=sorted(QUANT_MAP.keys()),
        help="Vocoder quantization type. Defaults to --quantize when set, otherwise F16. Upsample transposed convs stay F16.",
    )
    parser.add_argument(
        "--vocoder-quantize-scope",
        default="all",
        choices=["all", "resblocks", "no_post", "pre_post"],
        help="Experimental vocoder Conv1d subset to quantize when --vocoder-quantize is not f16.",
    )
    parser.add_argument(
        "--acoustic-quantize-override",
        action="append",
        default=[],
        metavar="PATTERN=TYPE",
        help="Override acoustic quantization for matching tensor names. PATTERN is glob or re:REGEX. TYPE may be f32/none, f16, q2_k, q4_0, etc. Repeatable; later matches win.",
    )
    parser.add_argument(
        "--acoustic-quantize-scope",
        default="graph",
        help="Comma-separated acoustic tensor classes eligible for --quantize. Choices: graph,embeddings,gru,bridge,depthwise,postnet,all. Overrides can still target eligible tensors only.",
    )
    parser.add_argument(
        "--acoustic-quantize-override-file",
        action="append",
        default=[],
        help="Read acoustic PATTERN=TYPE overrides from a file. Lines may also be full --acoustic-quantize-override 'PATTERN=TYPE' snippets.",
    )
    parser.add_argument(
        "--vocoder-quantize-override",
        action="append",
        default=[],
        metavar="PATTERN=TYPE",
        help="Override vocoder Conv1d quantization for matching full or generator-stripped tensor names. PATTERN is glob or re:REGEX. Repeatable; later matches win.",
    )
    parser.add_argument(
        "--unquantized-weight-dtype",
        default="f16",
        choices=["f16", "f32"],
        help="Storage dtype for unquantized multi-dimensional weight tensors. Biases, norms, scalars, and alphas stay F32.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.skip_acoustic and args.skip_vocoder:
        raise SystemExit("nothing to convert: both --skip-acoustic and --skip-vocoder were provided")
    if not args.skip_acoustic and not args.acoustic:
        raise SystemExit("--acoustic is required unless --skip-acoustic is provided")
    if not args.skip_vocoder and not args.vocoder:
        raise SystemExit("--vocoder is required unless --skip-vocoder is provided")

    q_type = QUANT_MAP[args.quantize] if args.quantize else None
    vocoder_quantize = args.vocoder_quantize if args.vocoder_quantize is not None else args.quantize
    vocoder_q_type = QUANT_MAP[vocoder_quantize] if vocoder_quantize else None
    acoustic_override_specs = read_override_files(args.acoustic_quantize_override_file) + args.acoustic_quantize_override
    acoustic_overrides = parse_quant_overrides(acoustic_override_specs, "acoustic")
    acoustic_quant_scopes = parse_scope_list(args.acoustic_quantize_scope)
    vocoder_overrides = parse_quant_overrides(args.vocoder_quantize_override, "vocoder")

    if not args.skip_acoustic:
        process_acoustic(
            args.acoustic,
            os.path.join(args.out_dir, "inflect_acoustic.gguf"),
            q_type,
            acoustic_overrides,
            args.unquantized_weight_dtype,
            acoustic_quant_scopes,
        )
    if not args.skip_vocoder:
        process_vocoder(
            args.vocoder,
            os.path.join(args.out_dir, "inflect_vocoder.gguf"),
            vocoder_q_type,
            args.vocoder_quantize_scope,
            vocoder_overrides,
        )

    print("\nConversion complete!")
