#!/usr/bin/env python3
"""End-to-end Inflect-Nano parity test for CI.

This script clones the original reference repository, dumps Python golden
tensors from the PyTorch checkpoints, converts those checkpoints to GGUF,
builds the C++ binary, dumps C++ tensors, and compares the two runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


REPO_URL = "https://huggingface.co/owensong/Inflect-Nano-v1/"
TEXT = "Hello, this is a test."


@dataclass
class Metric:
    name: str
    dtype: str
    count: int
    candidate_count: int
    ok: bool
    finite: bool = True
    cosine: float | None = None
    max_abs: float | None = None
    mean_abs: float | None = None
    mismatches: int | None = None
    note: str = ""


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(cmd, cwd=cwd, env=merged_env, text=True, check=True)


def normalized_os() -> str:
    name = platform.system().lower()
    return {"darwin": "macos"}.get(name, name)


def normalized_arch() -> str:
    name = platform.machine().lower()
    return {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(name, name)


def default_binary_path(tts_dir: Path) -> Path:
    return tts_dir / "build" / f"{normalized_os()}-{normalized_arch()}" / "inflect-nano"


def clone_reference(repo_url: str, ref_dir: Path, refresh: bool) -> None:
    if refresh and ref_dir.exists():
        shutil.rmtree(ref_dir)
    if not ref_dir.exists():
        ref_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", repo_url, str(ref_dir)], cwd=ref_dir.parent)
    else:
        run(["git", "fetch", "--all", "--tags"], cwd=ref_dir)

    if shutil.which("git-lfs"):
        run(["git", "lfs", "pull"], cwd=ref_dir)


def ensure_weight(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    if path.stat().st_size < 1024:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "git-lfs" in text:
            raise RuntimeError(
                f"{path} is still a Git LFS pointer. Install git-lfs and rerun, or run `git lfs pull` in the clone."
            )


def prepare_dictionary(tts_dir: Path, ref_dir: Path, cmudict_out: Path) -> None:
    cmudict_src = ref_dir / "third_party" / "tiny_tts_frontend" / "tiny_tts" / "text" / "cmudict.rep"
    if cmudict_src.exists():
        cmudict_out.parent.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "compile_cmudict.py", str(cmudict_src), str(cmudict_out)], cwd=tts_dir)
        return

    raise FileNotFoundError(f"missing cmudict source: {cmudict_src}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / denom)


def valid_f32(a: np.ndarray) -> bool:
    if not bool(np.all(np.isfinite(a))):
        return False
    if a.size == 0:
        return False
    return True


def compare_f32(name: str, golden: Path, candidate: Path, min_cosine: float, max_mean_abs: float) -> Metric:
    if not golden.exists() or not candidate.exists():
        return Metric(name, "f32", 0, 0, False, note="missing")
    a = np.fromfile(golden, dtype="<f4")
    b = np.fromfile(candidate, dtype="<f4")
    if a.shape != b.shape:
        return Metric(name, "f32", int(a.size), int(b.size), False, note="shape mismatch")
    finite = valid_f32(a) and valid_f32(b)
    diff = np.abs(a - b)
    cos = cosine(a, b)
    mean_abs = float(diff.mean() if diff.size else math.inf)
    max_abs = float(diff.max(initial=0.0))
    ok = finite and cos >= min_cosine and mean_abs <= max_mean_abs
    return Metric(name, "f32", int(a.size), int(b.size), ok, finite, cos, max_abs, mean_abs)


def compare_i32(name: str, golden: Path, candidate: Path) -> Metric:
    if not golden.exists() or not candidate.exists():
        return Metric(name, "i32", 0, 0, False, note="missing")
    a = np.fromfile(golden, dtype="<i4")
    b = np.fromfile(candidate, dtype="<i4")
    if a.shape != b.shape:
        return Metric(name, "i32", int(a.size), int(b.size), False, note="shape mismatch")
    mismatches = int(np.count_nonzero(a != b))
    return Metric(name, "i32", int(a.size), int(b.size), mismatches == 0, mismatches=mismatches)


def print_metric(metric: Metric) -> None:
    status = "ok" if metric.ok else "FAIL"
    if metric.dtype == "i32":
        print(f"{status:4} {metric.name}: count={metric.count} mismatches={metric.mismatches} {metric.note}")
    else:
        print(
            f"{status:4} {metric.name}: count={metric.count}/{metric.candidate_count} "
            f"cosine={metric.cosine} mean_abs={metric.mean_abs} max_abs={metric.max_abs} "
            f"finite={metric.finite} {metric.note}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clone, convert, dump, and compare Inflect-Nano parity tensors.")
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--work-dir", type=Path, default=Path("build/parity"))
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--speaker", default="mark")
    parser.add_argument("--binary", type=Path, default=None, help="Use this inflect-nano executable instead of the default build/<os>-<arch>/inflect-nano.")
    parser.add_argument("--skip-build", action="store_true", help="Do not run tools/build.sh; requires an existing --binary or default build/<os>-<arch>/inflect-nano.")
    parser.add_argument("--refresh", action="store_true", help="Delete and reclone the reference repo.")
    parser.add_argument("--keep-going", action="store_true", help="Print all diagnostics before returning failure.")
    parser.add_argument("--min-acoustic-cosine", type=float, default=0.99999)
    parser.add_argument("--max-acoustic-mean-abs", type=float, default=1e-3)
    parser.add_argument("--min-audio-cosine", type=float, default=0.80)
    parser.add_argument("--max-audio-mean-abs", type=float, default=0.05)
    args = parser.parse_args(argv)

    tts_dir = Path(__file__).resolve().parents[1]
    work_dir = (tts_dir / args.work_dir).resolve() if not args.work_dir.is_absolute() else args.work_dir.resolve()
    ref_dir = work_dir / "Inflect-Nano-v1"
    py_dir = work_dir / "golden_python"
    cpp_dir = work_dir / "golden_cpp"
    converted_dir = work_dir / "converted"
    cmudict_path = work_dir / "cmudict.bin"
    wav_path = work_dir / "output.wav"

    clone_reference(args.repo_url, ref_dir, args.refresh)

    acoustic_pt = ref_dir / "weights" / "inflect_nano_v1_acoustic.pt"
    vocoder_pt = ref_dir / "weights" / "inflect_nano_v1_vocoder.pt"
    ensure_weight(acoustic_pt)
    ensure_weight(vocoder_pt)
    prepare_dictionary(tts_dir, ref_dir, cmudict_path)

    shutil.rmtree(py_dir, ignore_errors=True)
    shutil.rmtree(cpp_dir, ignore_errors=True)
    shutil.rmtree(converted_dir, ignore_errors=True)
    py_dir.mkdir(parents=True, exist_ok=True)
    cpp_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "tools/dump-golden.py",
            "--text",
            args.text,
            "--speaker",
            args.speaker,
            "--out",
            str(py_dir),
            "--reference-root",
            str(ref_dir),
            "--acoustic",
            str(acoustic_pt),
            "--vocoder",
            str(vocoder_pt),
        ],
        cwd=tts_dir,
    )
    run(
        [
            sys.executable,
            "convert.py",
            "--acoustic",
            str(acoustic_pt),
            "--vocoder",
            str(vocoder_pt),
            "--out_dir",
            str(converted_dir),
        ],
        cwd=tts_dir,
    )
    if args.binary is not None:
        binary = args.binary.resolve()
    else:
        binary = default_binary_path(tts_dir)

    if not args.skip_build:
        run(["./tools/build.sh"], cwd=tts_dir)
    if not binary.exists():
        raise FileNotFoundError(f"missing inflect-nano executable: {binary}")

    run(
        [
            str(binary),
            "-a",
            str(converted_dir / "inflect_acoustic.gguf"),
            "-v",
            str(converted_dir / "inflect_vocoder.gguf"),
            "-d",
            str(cmudict_path),
            "-t",
            args.text,
            "-o",
            str(wav_path),
        ],
        cwd=tts_dir,
        env={"INFLECT_DUMP_DIR": str(cpp_dir)},
    )

    metrics: list[Metric] = []
    for name in ("phone_ids", "tone_ids", "lang_ids", "durations"):
        metrics.append(compare_i32(name, py_dir / f"{name}.i32", cpp_dir / f"{name}.i32"))

    acoustic_names = (
        "embed_sum",
        "enc_block_0",
        "enc_block_1",
        "enc_block_2",
        "enc_block_3",
        "enc_block_4",
        "encoded",
        "log_durations",
        "energy",
        "bright",
        "pitch",
        "regulated",
        "mel",
    )
    for name in acoustic_names:
        metrics.append(
            compare_f32(
                name,
                py_dir / f"{name}.f32",
                cpp_dir / f"{name}.f32",
                args.min_acoustic_cosine,
                args.max_acoustic_mean_abs,
            )
        )

    for name in ("audio_raw", "audio_normalized"):
        metrics.append(
            compare_f32(
                name,
                py_dir / f"{name}.f32",
                cpp_dir / f"{name}.f32",
                args.min_audio_cosine,
                args.max_audio_mean_abs,
            )
        )

    print("\nParity diagnostics")
    for metric in metrics:
        print_metric(metric)

    report_path = work_dir / "parity_report.json"
    report_path.write_text(json.dumps([asdict(m) for m in metrics], indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")
    print(f"Wrote {wav_path}")

    failed = [m for m in metrics if not m.ok]
    if failed:
        print("\nFailed metrics:", ", ".join(m.name for m in failed), file=sys.stderr)
        if args.keep_going:
            return 1
        return 1
    return 0


def test_tts_parity() -> None:
    """Pytest entry point."""
    rc = main([])
    assert rc == 0


if __name__ == "__main__":
    raise SystemExit(main())
