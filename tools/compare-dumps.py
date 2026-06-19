#!/usr/bin/env python3
"""Compare Inflect-Nano Python golden tensors with C++ debug dumps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / denom)


def compare_f32(name: str, golden: Path, candidate: Path) -> str:
    a = np.fromfile(golden, dtype="<f4")
    b = np.fromfile(candidate, dtype="<f4")
    if a.shape != b.shape:
        return f"{name}: shape mismatch {a.shape[0]} != {b.shape[0]}"
    diff = np.abs(a - b)
    max_i = int(np.argmax(diff))
    cos = cosine(a, b)
    lines = [
        f"{name}: count={a.size} max_abs={float(diff.max(initial=0.0)):.6g} "
        f"mean_abs={float(diff.mean() if diff.size else 0.0):.6g} cosine={cos:.9f}",
    ]
    lines.append(f"  max_diff at [{max_i}]: golden={float(a.flat[max_i]):.6g} candidate={float(b.flat[max_i]):.6g} diff={float(diff.flat[max_i]):.6g}")
    lines.append(f"  golden first: {a.flat[:5]}")
    lines.append(f"  candidate first: {b.flat[:5]}")
    return "\n".join(lines)


def compare_i32(name: str, golden: Path, candidate: Path) -> str:
    a = np.fromfile(golden, dtype="<i4")
    b = np.fromfile(candidate, dtype="<i4")
    if a.shape != b.shape:
        return f"{name}: shape mismatch {a.shape[0]} != {b.shape[0]}"
    mismatches = int(np.count_nonzero(a != b))
    first = ""
    if mismatches:
        idx = int(np.flatnonzero(a != b)[0])
        first = f" first_mismatch={idx} golden={int(a[idx])} candidate={int(b[idx])}"
    return f"{name}: count={a.size} mismatches={mismatches}{first}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python and C++ tensor dumps.")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()

    names = [
        "phone_ids.i32",
        "tone_ids.i32",
        "lang_ids.i32",
        "encoded.f32",
        "log_durations.f32",
        "energy.f32",
        "bright.f32",
        "pitch.f32",
        "regulated.f32",
        "mel.f32",
        "vocoder_conv_pre.f32",
        "vocoder_upsample_0.f32",
        "vocoder_resblock_0_0.f32",
        "vocoder_resblock_0_1.f32",
        "vocoder_resblock_0_2.f32",
        "vocoder_resblock_avg_0.f32",
        "vocoder_upsample_1.f32",
        "vocoder_resblock_1_0.f32",
        "vocoder_resblock_1_1.f32",
        "vocoder_resblock_1_2.f32",
        "vocoder_resblock_avg_1.f32",
        "vocoder_upsample_2.f32",
        "vocoder_resblock_2_0.f32",
        "vocoder_resblock_2_1.f32",
        "vocoder_resblock_2_2.f32",
        "vocoder_resblock_avg_2.f32",
        "vocoder_upsample_3.f32",
        "vocoder_resblock_3_0.f32",
        "vocoder_resblock_3_1.f32",
        "vocoder_resblock_3_2.f32",
        "vocoder_resblock_avg_3.f32",
        "vocoder_conv_post.f32",
        "vocoder_tanh.f32",
        "audio_raw.f32",
        "audio_normalized.f32",
    ]

    for filename in names:
        golden = args.golden / filename
        candidate = args.candidate / filename
        if not golden.exists() or not candidate.exists():
            print(f"{filename}: missing")
            continue
        name = filename.rsplit(".", 1)[0]
        if filename.endswith(".f32"):
            print(compare_f32(name, golden, candidate))
        else:
            print(compare_i32(name, golden, candidate))


if __name__ == "__main__":
    main()
