"""Shared audio normalization helpers used by the VAD and STT stages."""

import io
import subprocess

import numpy as np
import soundfile as sf
import torch
import torchaudio


class AudioDecodeError(Exception):
    """Raised when ffmpeg can't make sense of uploaded audio bytes."""


def decode_browser_audio(raw_bytes: bytes, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Transcode arbitrary browser-recorded audio (webm/opus from Chrome, mp4/aac from
    Safari, ogg, ...) into mono float32 PCM at target_sr via ffmpeg. soundfile/libsndfile
    can't read most of those containers/codecs directly, but ffmpeg auto-detects the input
    format from the byte stream itself (the claimed Content-Type is irrelevant to it)."""
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",
                "-f", "wav", "-ar", str(target_sr), "-ac", "1",
                "pipe:1",
            ],
            input=raw_bytes,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioDecodeError(f"ffmpeg failed to run: {exc}") from exc
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip() or "ffmpeg produced no output"
        raise AudioDecodeError(detail)
    data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    return data, sr


def to_mono_tensor(audio: np.ndarray, sr: int, target_sr: int) -> torch.Tensor:
    """Convert arbitrary-shape/dtype audio to a mono float32 tensor at target_sr."""
    tensor = torch.from_numpy(np.asarray(audio)).float()
    if tensor.ndim > 1:
        tensor = tensor.mean(dim=-1)
    if tensor.abs().max() > 1.5:  # looks like int16 range, normalize to [-1, 1]
        tensor = tensor / 32768.0
    if sr != target_sr:
        tensor = torchaudio.functional.resample(tensor, sr, target_sr)
    return tensor
