"""Preprocessing utilities shared between train.py and inference.py.

Cross-subject EEG normalization is the key challenge. Different subjects have
very different EEG amplitude scales, so a global z-score wipes out the
amplitude information the model needs. We z-score per trial (per-channel),
which is robust across subjects.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


@dataclass
class PreprocessConfig:
    fs: float = 250.0
    band_low: float = 8.0
    band_high: float = 30.0
    notch_freqs: tuple[float, ...] = (50.0, 60.0)
    crop_start: int = 125     # 0.5 s @ 250 Hz
    crop_end: int = 875       # 3.5 s @ 250 Hz
    normalize: str = "per_trial"  # "per_trial" or "per_subject"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PreprocessConfig":
        clean = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "notch_freqs" in clean:
            clean["notch_freqs"] = tuple(clean["notch_freqs"])
        return cls(**clean)


def _bandpass(x: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    nyq = fs / 2.0
    high = min(high, nyq - 1.0)
    b, a = butter(N=4, Wn=[low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1).astype(np.float32)


def _notch(x: np.ndarray, freq: float, fs: float, q: float = 30.0) -> np.ndarray:
    nyq = fs / 2.0
    if freq >= nyq:
        return x
    b, a = iirnotch(w0=freq / nyq, Q=q)
    return filtfilt(b, a, x, axis=-1).astype(np.float32)


def _zscore_per_trial(x: np.ndarray) -> np.ndarray:
    """Per-trial, per-channel z-score. x: (N, C, T)."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def _zscore_per_subject(x: np.ndarray) -> np.ndarray:
    """Per-subject z-score: treat the input as one subject's trials."""
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def preprocess_array(x: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Apply notch + bandpass + crop + normalize. x shape: (N, C, T)."""
    x = x.astype(np.float32, copy=True)

    for f in cfg.notch_freqs:
        x = _notch(x, f, cfg.fs)

    x = _bandpass(x, cfg.band_low, cfg.band_high, cfg.fs)

    s, e = cfg.crop_start, cfg.crop_end
    s = max(0, min(s, x.shape[-1]))
    e = max(s + 1, min(e, x.shape[-1]))
    x = x[..., s:e]

    if cfg.normalize == "per_trial":
        x = _zscore_per_trial(x)
    elif cfg.normalize == "per_subject":
        x = _zscore_per_subject(x)
    else:
        raise ValueError(f"Unknown normalize: {cfg.normalize}")

    return x


def filter_and_crop(x: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Notch + bandpass + crop only (no normalize).

    Used by the Riemannian + sliding-window pipeline, which needs raw filtered
    signal so that sliding windows can be extracted before covariance estimation.
    """
    x = x.astype(np.float32, copy=True)
    for f in cfg.notch_freqs:
        x = _notch(x, f, cfg.fs)
    x = _bandpass(x, cfg.band_low, cfg.band_high, cfg.fs)

    s, e = cfg.crop_start, cfg.crop_end
    s = max(0, min(s, x.shape[-1]))
    e = max(s + 1, min(e, x.shape[-1]))
    return x[..., s:e]


def sliding_windows(x: np.ndarray, window_size: int, stride: int) -> tuple[np.ndarray, int]:
    """Split each trial into overlapping windows along the time axis.

    Args:
        x: (N, C, T) float array.
        window_size: window length in samples.
        stride: hop in samples.

    Returns:
        (N * n_windows, C, window_size), n_windows
    """
    n, c, t = x.shape
    if window_size > t:
        raise ValueError(f"window_size {window_size} > T {t}")
    starts = np.arange(0, t - window_size + 1, stride)
    if len(starts) == 0:
        starts = np.array([0])
    pieces = [x[:, :, s:s + window_size] for s in starts]
    out = np.stack(pieces, axis=1)  # (N, W, C, win)
    out = out.reshape(n * len(starts), c, window_size)
    return out.astype(np.float32), len(starts)


def estimate_covs(x: np.ndarray) -> np.ndarray:
    """OAS shrinkage covariance for each trial. x: (N, C, T) -> (N, C, C)."""
    from pyriemann.estimation import Covariances
    return Covariances(estimator="oas").transform(x.astype(np.float64))


def align_covs(
    covs: np.ndarray, ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean Alignment whitening for cross-subject covariance harmonisation.

    For one subject's trial covariances ``covs`` (N, C, C), compute
    ``ref = mean(covs)`` (or use the provided ref), then whiten:

        aligned[i] = ref^(-1/2) @ covs[i] @ ref^(-1/2)

    After alignment, ``mean(aligned) ~= I``, which removes subject-specific
    baseline differences while preserving discriminative class structure.

    Reference:
        He & Wu, "Transfer Learning for Brain-Computer Interfaces: A Euclidean
        Space Data Alignment Approach", IEEE TBME 2020.

    Args:
        covs: (N, C, C) covariance matrices for one subject.
        ref: optional precomputed reference matrix (C, C). If None, computed as
            the arithmetic mean of covs.

    Returns:
        (aligned_covs, ref_used)
    """
    if ref is None:
        ref = covs.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(ref)
    eigvals = np.maximum(eigvals, 1e-6)
    ref_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
    aligned = np.array([ref_inv_sqrt @ c @ ref_inv_sqrt for c in covs])
    return aligned, ref


def compute_ea_whitening(x: np.ndarray) -> np.ndarray:
    """Compute the Euclidean Alignment whitening matrix R^(-1/2) for raw EEG.

    Unlike ``align_covs`` (which whitens covariance matrices for the Riemann
    pipeline), this whitens the *raw signal* so a deep model like EEGNet sees
    subject-harmonised trials.

    For one subject's filtered+cropped trials ``x`` (N, C, T):
        1. R = mean over trials of (x_i @ x_i^T / T)   -- reference covariance
        2. return R^(-1/2)

    Applying R^(-1/2) to every trial makes the subject's mean spatial
    covariance equal to the identity, removing subject-specific scale/rotation.

    Reference:
        He & Wu, "Transfer Learning for Brain-Computer Interfaces: A Euclidean
        Space Data Alignment Approach", IEEE TBME 2020.

    Args:
        x: (N, C, T) raw (filtered + cropped) trials for ONE subject.

    Returns:
        (C, C) whitening matrix R^(-1/2), float32.
    """
    x = x.astype(np.float64)
    covs = np.einsum("nct,ndt->ncd", x, x) / x.shape[-1]  # (N, C, C)
    ref = covs.mean(axis=0)                               # (C, C)
    eigvals, eigvecs = np.linalg.eigh(ref)
    eigvals = np.maximum(eigvals, 1e-6)
    r_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
    return r_inv_sqrt.astype(np.float32)


def apply_ea_whitening(x: np.ndarray, whitening: np.ndarray) -> np.ndarray:
    """Apply an EA whitening matrix to raw trials.

    Args:
        x: (N, C, T) raw trials.
        whitening: (C, C) matrix from ``compute_ea_whitening``.

    Returns:
        (N, C, T) aligned trials, float32.
    """
    aligned = np.einsum(
        "cd,ndt->nct", whitening.astype(np.float64), x.astype(np.float64)
    )
    return aligned.astype(np.float32)
