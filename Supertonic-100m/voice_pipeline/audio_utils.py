"""Shared audio normalization helpers used by the VAD and STT stages."""

import numpy as np
import torch
import torchaudio


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
