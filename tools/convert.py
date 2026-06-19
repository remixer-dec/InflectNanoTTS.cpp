#!/usr/bin/env python3
"""Convert Inflect-Nano (Acoustic + Vocoder) PyTorch checkpoints to GGUF format.

Reads acoustic.pt and vocoder.pt, transposes/folds weights to match the
C++ GGML implementation, and writes two separate .gguf files.

Usage:
    python convert.py --acoustic acoustic.pt --vocoder vocoder.pt --out_dir models/
    python convert.py --acoustic acoustic.pt --vocoder vocoder.pt --out_dir models/ --quantize q8_0
"""

import argparse
import os
import torch
import numpy as np
import gguf

QUANT_MAP = {
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
    "q4_k": gguf.GGMLQuantizationType.Q4_K,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q5_k": gguf.GGMLQuantizationType.Q5_K,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "f16": gguf.GGMLQuantizationType.F16,
}


def to_numpy(t):
    return t.detach().cpu().float().numpy()


def fold_wn(weight_v, weight_g):
    """Folds PyTorch weight_norm (v and g) into a single weight tensor."""
    v = to_numpy(weight_v)
    g = to_numpy(weight_g)
    dims = list(range(1, v.ndim))
    norm = np.linalg.norm(v, axis=tuple(dims), keepdims=True)
    return v * (g / (norm + 1e-8))


def add_tensor(writer, name, arr, quant_type=None):
    """Adds a tensor to the GGUF writer, applying quantization if possible."""
    arr = np.ascontiguousarray(arr)

    # Snake alphas and scalars must remain F32 for precision
    if quant_type and "log_alpha" not in name and arr.size > 1:
        # GGML block quantization (Q4_K, Q8_0) requires the innermost dimension (ne[0])
        # to be a multiple of 32.
        if arr.ndim > 1 and arr.shape[-1] % 32 == 0:
            try:
                qbytes = gguf.quants.quantize(arr, quant_type)
                writer.add_tensor(name, qbytes, raw_dtype=quant_type)
                return
            except Exception as e:
                print(f"Warning: could not quantize {name} to {quant_type}: {e}")

        # Fallback to F16 if quantization is requested but dimension doesn't align
        writer.add_tensor(name, arr.astype(np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
        return

    # Default to F32 for small tensors, biases, and alphas
    writer.add_tensor(name, arr.astype(np.float32))


def is_quantizable(name):
    """Determine if a tensor should be quantized based on its name."""
    if "log_alpha" in name:
        return False
    if name.endswith(".bias"):
        return False
    # We do not quantize depthwise convs, they are small and custom-handled
    if "depth" in name:
        return False
    return True


def strip_generator_prefix(name):
    return name[len("generator.") :] if name.startswith("generator.") else name


def process_acoustic(pt_path, out_path, quant_type=None):
    print(f"Loading acoustic model from {pt_path}...")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_state = ckpt["model"]

    w = gguf.GGUFWriter(out_path, "inflect-acoustic")

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
            pass

        # 2. Pointwise Conv1d 1x1
        elif name.endswith("point.weight"):
            pass

        # 3. Depthwise Conv1d: [2*H, 1, K] -> store [2, H, K], load [K, H, 2]
        elif name.endswith("depth.weight"):
            arr = arr.reshape(2, H, -1)

        # 4. Postnet Conv1d
        elif name.startswith("postnet.") and name.endswith(".weight"):
            pass

        # 5. Token embeddings
        elif name in ["phone.weight", "tone.weight", "lang.weight"]:
            pass

        # 6. Speaker embedding is [speaker_count, speaker_dim] in PyTorch;
        # writing unchanged loads as GGML [speaker_dim, speaker_count].
        elif name == "speaker.weight":
            pass

        qt = quant_type if is_quantizable(name) else None
        add_tensor(w, name, arr, qt)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"Wrote {out_path}")


def process_vocoder(pt_path, out_path, quant_type=None):
    print(f"Loading vocoder model from {pt_path}...")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    generator_state = ckpt["generator"]

    w = gguf.GGUFWriter(out_path, "inflect-vocoder")

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

    def emit_weight(name, arr):
        bare_name = strip_generator_prefix(name)

        # GGUF reverses numpy dimensions into GGML ne[] order. Keep PyTorch
        # Conv1d [out, in, K] and ConvTranspose1d [in, out, K] contiguous.
        if bare_name in ["conv_pre.weight", "conv_post.weight"] or "convs1." in bare_name or "convs2." in bare_name:
            pass
        elif bare_name.startswith("ups.") or ".ups." in bare_name:
            pass

        # Do not quantize transposed convs to prevent audio static. Modern GGML
        # CPU conv_transpose_1d expects the kernel tensor to be F16.
        qt = quant_type if (is_quantizable(name) and "ups." not in bare_name) else None
        if qt is None:
            w.add_tensor(name, np.ascontiguousarray(arr).astype(np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
        else:
            add_tensor(w, name, arr, qt)
        emitted_weights.add(name)

    for name, tensor in generator_state.items():
        # Skip weight_norm parameters, we will fold them
        if ".weight_v" in name or ".weight_g" in name:
            continue

        # Squeeze Snake activation alphas: [1, C, 1] -> [C]
        if name.endswith(".log_alpha"):
            arr = to_numpy(tensor).squeeze()
            add_tensor(w, name, arr)
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
            add_tensor(w, name, arr)

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
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Inflect-Nano PyTorch models to GGUF")
    parser.add_argument("--acoustic", required=True, help="Path to inflect_nano_v1_acoustic.pt")
    parser.add_argument("--vocoder", required=True, help="Path to inflect_nano_v1_vocoder.pt")
    parser.add_argument("--out_dir", default=".", help="Output directory for .gguf files")
    parser.add_argument(
        "--quantize",
        default=None,
        choices=sorted(QUANT_MAP.keys()),
        help="Quantization type (e.g., q8_0, q4_k). Falls back to F16 for unaligned dims.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.quantize:
        print("Note: ignoring --quantize for now; current host runtime expects acoustic F32 and vocoder F16 conv kernels.")

    process_acoustic(args.acoustic, os.path.join(args.out_dir, "inflect_acoustic.gguf"), None)
    process_vocoder(args.vocoder, os.path.join(args.out_dir, "inflect_vocoder.gguf"), None)

    print("\nConversion complete!")
