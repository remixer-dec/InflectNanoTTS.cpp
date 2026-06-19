#!/usr/bin/env python3
"""Dump Python reference tensors for Inflect-Nano host parity checks."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch


def add_reference_to_path(ref_root: Path) -> None:
    sys.path.insert(0, str(ref_root))
    sys.path.insert(0, str(ref_root / "third_party" / "tiny_tts_frontend"))


def save_manifest_line(out_dir: Path, line: str) -> None:
    with (out_dir / "manifest.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_f32(out_dir: Path, name: str, tensor: torch.Tensor | np.ndarray, shape: str) -> None:
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().contiguous().numpy()
    else:
        arr = np.asarray(tensor)
    arr.astype("<f4", copy=False).tofile(out_dir / f"{name}.f32")
    save_manifest_line(out_dir, f"{name} f32 {shape} count={arr.size}")


def save_i32(out_dir: Path, name: str, tensor: torch.Tensor | np.ndarray, shape: str) -> None:
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().contiguous().numpy()
    else:
        arr = np.asarray(tensor)
    arr.astype("<i4", copy=False).tofile(out_dir / f"{name}.i32")
    save_manifest_line(out_dir, f"{name} i32 {shape} count={arr.size}")


def save_vocoder_f32(out_dir: Path, name: str, tensor: torch.Tensor) -> None:
    arr = tensor.detach().cpu()
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(f"unexpected vocoder tensor shape for {name}: {tuple(arr.shape)}")
    save_f32(out_dir, name, arr.contiguous(), f"[{arr.shape[0]},{arr.shape[1]}]")


def normalize_audio(audio: np.ndarray, target_rms_db: float = -20.0, peak_db: float = -1.0) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)
    audio = audio - float(audio.mean())
    rms = 20.0 * math.log10(float(np.sqrt(np.mean(audio**2, dtype=np.float64))) + 1e-9)
    audio *= 10 ** ((target_rms_db - rms) / 20.0)
    peak = float(np.max(np.abs(audio)) + 1e-9)
    peak_limit = 10 ** (peak_db / 20.0)
    if peak > peak_limit:
        audio *= peak_limit / peak
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


@torch.inference_mode()
def dump(args: argparse.Namespace) -> None:
    ref_root = args.reference_root.resolve()
    add_reference_to_path(ref_root)

    from inference import load_acoustic, load_vocoder, text_to_tokens

    device = torch.device(args.device)
    acoustic, speakers, _ = load_acoustic(args.acoustic, device)
    vocoder, _ = load_vocoder(args.vocoder, device)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.txt").write_text("inflect-nano Python golden dump\n", encoding="utf-8")
    save_manifest_line(out_dir, f"text {args.text}")
    save_manifest_line(out_dir, f"reference_root {ref_root}")
    save_manifest_line(out_dir, f"acoustic {args.acoustic.resolve()}")
    save_manifest_line(out_dir, f"vocoder {args.vocoder.resolve()}")

    phone, tone, lang = text_to_tokens(args.text)
    save_i32(out_dir, "phone_ids", phone, f"[{phone.numel()}]")
    save_i32(out_dir, "tone_ids", tone, f"[{tone.numel()}]")
    save_i32(out_dir, "lang_ids", lang, f"[{lang.numel()}]")

    phone = phone.unsqueeze(0).to(device)
    tone = tone.unsqueeze(0).to(device)
    lang = lang.unsqueeze(0).to(device)
    speaker_id = int(speakers.get(args.speaker, next(iter(speakers.values()), 0)))
    speaker = torch.LongTensor([speaker_id]).to(device)

    token_mask = torch.ones_like(phone, dtype=torch.bool)

    # Inline encode with per-block dumps for parity debugging
    x = (
        acoustic.phone(phone)
        + acoustic.tone(tone.clamp_max(acoustic.cfg.tone_size - 1))
        + acoustic.lang(lang.clamp_max(acoustic.cfg.lang_size - 1))
    )
    x = x + acoustic.speaker_proj(acoustic.speaker(speaker)).unsqueeze(1)
    x = x * token_mask.unsqueeze(-1)
    save_f32(out_dir, "embed_sum", x[0].transpose(0, 1), f"[{x.shape[-1]},{phone.shape[1]}]")

    for i, block in enumerate(acoustic.encoder):
        x = block(x, token_mask)
        save_f32(out_dir, f"enc_block_{i}", x[0].transpose(0, 1), f"[{x.shape[-1]},{phone.shape[1]}]")

    encoded = x
    log_dur, energy, bright, pitch = acoustic.predict_prosody(encoded, token_mask)

    if getattr(acoustic.cfg, "use_group_duration_planner", False):
        durations = acoustic.apply_group_duration_plan(
            phone, log_dur, encoded, float(args.length_scale), int(args.max_duration)
        )
        durations = durations.masked_fill(~token_mask, 0)
    else:
        pred_dur = torch.expm1(log_dur).clamp(0, int(args.max_duration)) * float(args.length_scale)
        durations = torch.round(pred_dur).long().clamp_min(int(args.min_duration)).masked_fill(~token_mask, 0)

    energy_scaled = energy * float(args.energy_scale)
    pitch_scaled = torch.stack(
        [pitch[..., 0] * float(args.pitch_scale), pitch[..., 1].clamp(0.0, 1.0)],
        dim=-1,
    )
    conditioned = encoded + acoustic.energy_proj(energy_scaled.unsqueeze(-1)) + acoustic.bright_proj(bright.unsqueeze(-1))
    frames, frame_meta, frame_mask = acoustic.regulate(conditioned, durations)
    x = frames + acoustic.frame_proj(frame_meta) + acoustic.add_local_context(conditioned, durations)

    pos = torch.arange(x.shape[1], device=x.device)
    pos = torch.div(
        pos * acoustic.cfg.abs_frame_bins,
        max(1, acoustic.cfg.max_frames),
        rounding_mode="floor",
    ).clamp_max(acoustic.cfg.abs_frame_bins - 1)
    x = x + acoustic.abs_frame(pos).unsqueeze(0)

    if acoustic.cfg.use_frame_pitch:
        pitch_frame = acoustic.expand_token_feature(pitch_scaled, durations)[:, : x.shape[1]]
        x = x + acoustic.pitch_proj(pitch_frame)

    regulated = x

    for block in acoustic.decoder:
        x = block(x, frame_mask)
    x = x + acoustic.frame_gru(x)[0]
    mel_base = acoustic.mel_head(x).transpose(1, 2)
    mel = mel_base + acoustic.cfg.postnet_scale * acoustic.postnet(mel_base)
    v = vocoder
    vx = v.conv_pre(mel)
    save_vocoder_f32(out_dir, "vocoder_conv_pre", vx)

    ups = list(v.ups)
    resblocks = list(v.resblocks)
    n_res = len(resblocks) // max(1, len(ups))
    for i, up in enumerate(ups):
        vx = v.up_acts[i](vx)
        vx = up(vx)
        save_vocoder_f32(out_dir, f"vocoder_upsample_{i}", vx)

        rb_sum = None
        for j in range(n_res):
            rb = resblocks[i * n_res + j]
            rb_out = rb(vx)
            save_vocoder_f32(out_dir, f"vocoder_resblock_{i}_{j}", rb_out)
            rb_sum = rb_out if rb_sum is None else rb_sum + rb_out
        vx = rb_sum / n_res
        save_vocoder_f32(out_dir, f"vocoder_resblock_avg_{i}", vx)

    vx = v.post_act(vx)
    vx = v.conv_post(vx)
    save_vocoder_f32(out_dir, "vocoder_conv_post", vx)
    vx = torch.tanh(vx)
    save_vocoder_f32(out_dir, "vocoder_tanh", vx)

    audio_raw = vx.squeeze().detach().cpu().numpy().astype(np.float32)
    audio_norm = normalize_audio(audio_raw)

    t = int(phone.shape[1])
    h = int(acoustic.cfg.hidden)
    frames_n = int(mel.shape[-1])
    n_mels = int(acoustic.cfg.n_mels)

    save_f32(out_dir, "encoded", encoded[0].transpose(0, 1), f"[{h},{t}]")
    save_f32(out_dir, "log_durations", log_dur[0], f"[{t}]")
    save_f32(out_dir, "energy", energy[0], f"[{t}]")
    save_f32(out_dir, "bright", bright[0], f"[{t}]")
    save_f32(out_dir, "pitch", pitch[0].transpose(0, 1), f"[2,{t}]")
    save_i32(out_dir, "durations", durations[0].to(torch.int32), f"[{t}]")
    save_f32(out_dir, "regulated", regulated[0].transpose(0, 1), f"[{h},{frames_n}]")
    save_f32(out_dir, "mel", mel[0], f"[{n_mels},{frames_n}]")
    save_f32(out_dir, "audio_raw", audio_raw, f"[{audio_raw.size}]")
    save_f32(out_dir, "audio_normalized", audio_norm, f"[{audio_norm.size}]")

    print(f"Wrote golden tensors to {out_dir}")
    print(f"Frames: {frames_n}, samples: {audio_raw.size}, speaker: {speaker_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump Python reference tensors for C++ parity checks.")
    parser.add_argument("--text", default="Hello, this is a test.")
    parser.add_argument("--out", type=Path, default=Path("build/golden/python"))
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--acoustic", type=Path, required=True)
    parser.add_argument("--vocoder", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speaker", default="mark")
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--pitch-scale", type=float, default=1.0)
    parser.add_argument("--energy-scale", type=float, default=1.0)
    parser.add_argument("--min-duration", type=int, default=1)
    parser.add_argument("--max-duration", type=int, default=80)
    dump(parser.parse_args())


if __name__ == "__main__":
    main()
