#!/usr/bin/env python3
"""Quantize acoustic graph weights to Q2 two-at-a-time and run inference.

The output CSV ranks pairs by audio difference from a baseline run. This is a
coarse screening tool; use the vocoder/acoustic tensor dumps for final diagnosis.
"""

import argparse
import atexit
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import fnmatch
import hashlib
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

from convert import QUANT_MAP, is_graph_linear_weight


DEFAULT_TEXT = "Hello, this is a test of the Inflect Nano text to speech system."
WORKER_BASELINE_AUDIO = None


def normalized_os():
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def normalized_arch():
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    if machine in {"x86_64", "amd64"}:
        return "x64"
    return machine


def default_binary_path(tts_dir):
    return tts_dir / "build" / f"{normalized_os()}-{normalized_arch()}" / "inflect-nano"


def acquire_lock(out_dir):
    lock_path = Path(out_dir) / ".sweep.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        detail = lock_path.read_text(encoding="utf-8", errors="ignore").strip()
        raise SystemExit(f"refusing to run: sweep lock exists at {lock_path} ({detail})")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()}\n")

    def cleanup_lock():
        try:
            if lock_path.exists() and f"pid={os.getpid()}" in lock_path.read_text(encoding="utf-8", errors="ignore"):
                lock_path.unlink()
        except OSError:
            pass

    atexit.register(cleanup_lock)
    return lock_path


def run(cmd, cwd, env=None, log_path=None, quiet=False):
    started = time.monotonic()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    if log_path is None:
        if not quiet:
            print("+ " + " ".join(str(x) for x in cmd), flush=True)
        subprocess.run(cmd, cwd=cwd, env=merged_env, check=True)
        return time.monotonic() - started

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("+ " + " ".join(str(x) for x in cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=cwd, env=merged_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.monotonic() - started


def split_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def pattern_matches(pattern, name):
    if pattern.startswith("re:"):
        import re

        return re.search(pattern[3:], name) is not None
    return fnmatch.fnmatchcase(name, pattern)


def select_layers(acoustic_pt, patterns):
    ckpt = torch.load(acoustic_pt, map_location="cpu", weights_only=False)
    names = [name for name in ckpt["model"] if is_graph_linear_weight(name)]
    names.sort()
    if not patterns:
        return names
    selected = []
    for name in names:
        if any(pattern_matches(pattern, name) for pattern in patterns):
            selected.append(name)
    return selected


def safe_case_name(index, layers):
    joined = "__".join(layers)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    compact = "__".join(layer.replace(".", "_") for layer in layers)
    compact = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in compact)
    return f"case_{index:04d}_{compact[:80]}_{digest}"


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM WAV: {path}")
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data


def compare_audio(reference, candidate):
    n = min(reference.size, candidate.size)
    if n == 0:
        return {"samples": 0, "cosine": 0.0, "mean_abs": float("inf"), "max_abs": float("inf")}
    ref = reference[:n]
    cand = candidate[:n]
    diff = cand - ref
    denom = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    cosine = float(np.dot(ref, cand) / denom) if denom > 0 else 0.0
    return {
        "samples": int(n),
        "cosine": cosine,
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def worker_env(tts_threads=None):
    env = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if tts_threads and tts_threads > 0:
        env["INFLECT_THREADS"] = str(tts_threads)
    return env


def convert_case_config(
    tts_dir,
    acoustic,
    vocoder,
    out_dir,
    base_quantize,
    q2_type,
    vocoder_quantize,
    overrides,
    log_path=None,
    quiet=False,
    skip_vocoder=False,
):
    cmd = [
        sys.executable,
        str(tts_dir / "tools" / "convert.py"),
        "--acoustic",
        str(acoustic),
        "--vocoder",
        str(vocoder),
        "--out_dir",
        str(out_dir),
        "--vocoder-quantize",
        vocoder_quantize,
    ]
    if skip_vocoder:
        cmd.append("--skip-vocoder")
    if base_quantize != "none":
        cmd += ["--quantize", base_quantize]
    for layer in overrides:
        cmd += ["--acoustic-quantize-override", f"{layer}={q2_type}"]
    return run(cmd, tts_dir, env=worker_env(), log_path=log_path, quiet=quiet)


def convert_case(tts_dir, args, out_dir, overrides, log_path=None, quiet=False, skip_vocoder=False):
    return convert_case_config(
        tts_dir,
        args.acoustic,
        args.vocoder,
        out_dir,
        args.base_quantize,
        args.q2_type,
        args.vocoder_quantize,
        overrides,
        log_path=log_path,
        quiet=quiet,
        skip_vocoder=skip_vocoder,
    )


def synthesize(tts_dir, binary, cmudict, model_dir, text, wav_path, log_path=None, quiet=False, tts_threads=1):
    cmd = [
        str(binary),
        "-a",
        str(model_dir / "inflect_acoustic.gguf"),
        "-v",
        str(model_dir / "inflect_vocoder.gguf"),
        "-d",
        str(cmudict),
        "-t",
        text,
        "-o",
        str(wav_path),
    ]
    if tts_threads and tts_threads > 0:
        cmd += ["--threads", str(tts_threads)]
    return run(
        cmd,
        tts_dir,
        env=worker_env(tts_threads),
        log_path=log_path,
        quiet=quiet,
    )


def reuse_vocoder(src, dst, mode):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    os.symlink(src.resolve(), dst)


def write_reports(out_dir, rows, verbose=False):
    jsonl_path = out_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    csv_path = out_dir / "results.csv"
    fieldnames = [
        "rank",
        "case",
        "layers",
        "cosine",
        "mean_abs",
        "max_abs",
        "samples",
        "convert_seconds",
        "link_seconds",
        "synth_seconds",
        "compare_seconds",
        "total_seconds",
        "wav",
        "log",
        "error",
    ]
    ranked = sorted(rows, key=lambda r: (r.get("cosine", float("inf")), -r.get("mean_abs", -float("inf"))))
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "case": row["case"],
                    "layers": "|".join(row["layers"]),
                    "cosine": row.get("cosine", ""),
                    "mean_abs": row.get("mean_abs", ""),
                    "max_abs": row.get("max_abs", ""),
                    "samples": row.get("samples", ""),
                    "convert_seconds": row.get("convert_seconds", ""),
                    "link_seconds": row.get("link_seconds", ""),
                    "synth_seconds": row.get("synth_seconds", ""),
                    "compare_seconds": row.get("compare_seconds", ""),
                    "total_seconds": row.get("total_seconds", ""),
                    "wav": row["wav"],
                    "log": row.get("log", ""),
                    "error": row.get("error", ""),
                }
            )
    if verbose:
        print(f"Wrote {jsonl_path}")
        print(f"Wrote {csv_path}")


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def progress_line(done, total, started_at, recent_times=None):
    elapsed = time.monotonic() - started_at
    pct = (done / total) * 100.0 if total else 100.0
    rate = 0.0
    if recent_times and len(recent_times) >= 2:
        window_elapsed = recent_times[-1] - recent_times[0]
        if window_elapsed > 0:
            rate = (len(recent_times) - 1) / window_elapsed
    elif done > 0 and elapsed > 0:
        rate = done / elapsed
    eta = "estimating" if rate <= 0 else format_duration((total - done) / rate)
    return (
        f"[pid={os.getpid()} {done}/{total} {pct:5.1f}%] "
        f"elapsed={format_duration(elapsed)} eta={eta}"
    )


def init_worker(baseline_wav):
    global WORKER_BASELINE_AUDIO
    WORKER_BASELINE_AUDIO = read_wav(Path(baseline_wav))


def run_pair_case(payload):
    if WORKER_BASELINE_AUDIO is None:
        raise RuntimeError("worker baseline audio was not initialized")

    tts_dir = Path(payload["tts_dir"])
    case_dir = Path(payload["case_dir"])
    wav_path = case_dir / "output.wav"
    log_path = case_dir / "run.log"
    pair = payload["pair"]

    case_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.monotonic()
    convert_seconds = convert_case_config(
        tts_dir,
        Path(payload["acoustic"]),
        Path(payload["vocoder"]),
        case_dir,
        payload["base_quantize"],
        payload["q2_type"],
        payload["vocoder_quantize"],
        pair,
        log_path=log_path,
        quiet=True,
        skip_vocoder=payload["reuse_vocoder"],
    )
    link_seconds = 0.0
    if payload["reuse_vocoder"]:
        link_started = time.monotonic()
        reuse_vocoder(payload["baseline_vocoder"], case_dir / "inflect_vocoder.gguf", payload["reuse_vocoder_mode"])
        link_seconds = time.monotonic() - link_started
    synth_seconds = synthesize(
        tts_dir,
        Path(payload["binary"]),
        Path(payload["cmudict"]),
        case_dir,
        payload["text"],
        wav_path,
        log_path=log_path,
        quiet=True,
        tts_threads=payload["tts_threads"],
    )
    compare_started = time.monotonic()
    metrics = compare_audio(WORKER_BASELINE_AUDIO, read_wav(wav_path))
    compare_seconds = time.monotonic() - compare_started
    if payload["delete_case_models"]:
        for name in ("inflect_acoustic.gguf", "inflect_vocoder.gguf"):
            model_path = case_dir / name
            if model_path.exists():
                model_path.unlink()
    return {
        "case": payload["case"],
        "layers": list(pair),
        "wav": str(wav_path),
        "log": str(log_path),
        "convert_seconds": convert_seconds,
        "link_seconds": link_seconds,
        "synth_seconds": synth_seconds,
        "compare_seconds": compare_seconds,
        "total_seconds": time.monotonic() - total_started,
        **metrics,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run pairwise acoustic Q2 quantization sweep.")
    parser.add_argument("--acoustic", required=True, type=Path, help="Path to acoustic .pt checkpoint.")
    parser.add_argument("--vocoder", required=True, type=Path, help="Path to vocoder .pt checkpoint.")
    parser.add_argument("--cmudict", required=True, type=Path, help="Path to compiled cmudict.bin.")
    parser.add_argument("--out-dir", type=Path, default=Path("build/q2_pair_sweep"))
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--base-quantize", default="f16", choices=sorted([*QUANT_MAP.keys(), "none"]))
    parser.add_argument("--q2-type", default="q2_k", choices=sorted(QUANT_MAP.keys()))
    parser.add_argument("--vocoder-quantize", default="f16", choices=sorted(QUANT_MAP.keys()))
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer globs/regexes to limit the sweep, e.g. 'encoder.*.ff.*.weight,mel_head.*'.",
    )
    parser.add_argument("--start", type=int, default=0, help="Skip this many layer pairs.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of layer pairs to run.")
    parser.add_argument("--clean", action="store_true", help="Delete --out-dir before running.")
    parser.add_argument("--build", action="store_true", help="Run tools/build.sh before inference.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after failed cases.")
    parser.add_argument("--list-layers", action="store_true", help="Print selected acoustic layers and exit.")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1), help="Number of pair cases to run concurrently.")
    parser.add_argument("--tts-threads", type=int, default=1, help="GGML CPU threads per inflect-nano inference process.")
    parser.add_argument("--progress-interval", type=float, default=10.0, help="Seconds between progress heartbeat lines while jobs are running.")
    parser.add_argument("--delete-case-models", action="store_true", help="Delete per-case GGUF files after each inference; keeps WAV, log, and result rows.")
    parser.add_argument("--no-reuse-vocoder", action="store_true", help="Reconvert the vocoder for every pair case instead of reusing the baseline vocoder.")
    parser.add_argument("--reuse-vocoder-mode", choices=["symlink", "hardlink", "copy"], default="symlink", help="How pair cases should reuse the baseline vocoder GGUF.")
    args = parser.parse_args(argv)

    if not args.q2_type.startswith("q2"):
        raise SystemExit("--q2-type should be a Q2 variant for this sweep")

    tts_dir = Path(__file__).resolve().parents[1]
    if args.clean:
        shutil.rmtree(args.out_dir, ignore_errors=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = acquire_lock(args.out_dir)

    binary = args.binary.resolve() if args.binary else default_binary_path(tts_dir)
    if args.build:
        run(["./tools/build.sh"], tts_dir)
    if not binary.exists():
        raise FileNotFoundError(f"missing inflect-nano executable: {binary}")

    layers = select_layers(args.acoustic, split_csv(args.layers))
    if args.list_layers:
        for name in layers:
            print(name)
        print(f"{len(layers)} layer(s)")
        return 0

    pairs = list(itertools.combinations(layers, 2))
    pairs = pairs[args.start :]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("no layer pairs selected")

    manifest = {
        "acoustic": str(args.acoustic),
        "vocoder": str(args.vocoder),
        "cmudict": str(args.cmudict),
        "binary": str(binary),
        "text": args.text,
        "base_quantize": args.base_quantize,
        "q2_type": args.q2_type,
        "vocoder_quantize": args.vocoder_quantize,
        "jobs": args.jobs,
        "tts_threads": args.tts_threads,
        "delete_case_models": args.delete_case_models,
        "reuse_vocoder": not args.no_reuse_vocoder,
        "reuse_vocoder_mode": args.reuse_vocoder_mode,
        "layers": layers,
        "num_pairs": len(pairs),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    baseline_dir = args.out_dir / "baseline"
    baseline_wav = baseline_dir / "output.wav"
    baseline_log = baseline_dir / "run.log"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    print("[baseline] converting and synthesizing reference audio", flush=True)
    baseline_convert_seconds = convert_case(tts_dir, args, baseline_dir, [], log_path=baseline_log, quiet=True)
    baseline_synth_seconds = synthesize(
        tts_dir,
        binary,
        args.cmudict,
        baseline_dir,
        args.text,
        baseline_wav,
        log_path=baseline_log,
        quiet=True,
        tts_threads=args.tts_threads,
    )
    read_wav(baseline_wav)
    print(
        f"[baseline] convert={baseline_convert_seconds:.2f}s synth={baseline_synth_seconds:.2f}s "
        f"vocoder={baseline_dir / 'inflect_vocoder.gguf'}",
        flush=True,
    )

    rows = []
    started_at = time.monotonic()
    total = len(pairs)
    jobs = max(1, args.jobs)
    print(
        f"[sweep pid={os.getpid()}] running {total} pair cases with {jobs} worker(s); "
        f"lock={lock_path}; logs are under {args.out_dir}/case_*/run.log",
        flush=True,
    )

    next_offset = 0
    pending = set()
    future_meta = {}

    def submit_next(executor):
        nonlocal next_offset
        if next_offset >= total:
            return None
        pair = pairs[next_offset]
        index = args.start + next_offset
        case = safe_case_name(index, pair)
        case_dir = args.out_dir / case
        payload = {
            "tts_dir": str(tts_dir),
            "acoustic": str(args.acoustic),
            "vocoder": str(args.vocoder),
            "cmudict": str(args.cmudict),
            "binary": str(binary),
            "text": args.text,
            "tts_threads": args.tts_threads,
            "base_quantize": args.base_quantize,
            "q2_type": args.q2_type,
            "vocoder_quantize": args.vocoder_quantize,
            "case": case,
            "case_dir": str(case_dir),
            "pair": pair,
            "reuse_vocoder": not args.no_reuse_vocoder,
            "baseline_vocoder": str(baseline_dir / "inflect_vocoder.gguf"),
            "reuse_vocoder_mode": args.reuse_vocoder_mode,
        }
        payload["delete_case_models"] = args.delete_case_models
        future = executor.submit(run_pair_case, payload)
        future_meta[future] = {"case": case, "pair": pair, "case_dir": str(case_dir)}
        next_offset += 1
        return future

    with ProcessPoolExecutor(max_workers=jobs, initializer=init_worker, initargs=(str(baseline_wav),)) as executor:
        for _ in range(min(jobs, total)):
            future = submit_next(executor)
            if future is not None:
                pending.add(future)

        completed = 0
        recent_times = []
        while pending:
            done_set, pending = wait(pending, timeout=args.progress_interval, return_when=FIRST_COMPLETED)
            if not done_set:
                queued = total - completed - len(pending)
                print(
                    f"{progress_line(completed, total, started_at, recent_times)} "
                    f"active={len(pending)} queued={queued} jobs={jobs}",
                    flush=True,
                )
                continue

            for future in done_set:
                completed += 1
                now = time.monotonic()
                recent_times.append(now)
                recent_times = recent_times[-32:]
                meta = future_meta[future]
                try:
                    row = future.result()
                    rows.append(row)
                    print(
                        f"{progress_line(completed, total, started_at, recent_times)} {row['case']}: "
                        f"cosine={row['cosine']:.6f} mean_abs={row['mean_abs']:.6f} "
                        f"max_abs={row['max_abs']:.6f} "
                        f"convert={row['convert_seconds']:.2f}s synth={row['synth_seconds']:.2f}s",
                        flush=True,
                    )
                except Exception as exc:
                    row = {
                        "case": meta["case"],
                        "layers": list(meta["pair"]),
                        "wav": str(Path(meta["case_dir"]) / "output.wav"),
                        "log": str(Path(meta["case_dir"]) / "run.log"),
                        "error": str(exc),
                    }
                    rows.append(row)
                    print(
                        f"{progress_line(completed, total, started_at, recent_times)} {meta['case']}: failed; "
                        f"log={row['log']} error={exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if not args.keep_going:
                        for pending_future in pending:
                            pending_future.cancel()
                        write_reports(args.out_dir, rows)
                        raise

                write_reports(args.out_dir, rows)
                future = submit_next(executor)
                if future is not None:
                    pending.add(future)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
