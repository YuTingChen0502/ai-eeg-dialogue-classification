"""Task 2 inference.

Compatible CLI:
python inference.py --data <data_dir_or_test_file> --checkpoint <checkpoint_path> [--output <output_path>]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

import joblib
import numpy as np
from scipy.signal import butter, sosfiltfilt

EPS = 1e-10


@dataclass(frozen=True)
class Config:
    name: str
    band_set: str
    feature_set: str
    raw_norm: str
    feature_norm: str
    classifier: str
    C: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 2 inference and write a Kaggle submission CSV.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    return parser.parse_args()


def resolve_test_file(data_path: Path) -> Path:
    if data_path.is_file():
        return data_path
    candidates = [data_path / "test.npz", data_path / "task2" / "test.npz"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate Task 2 test data. Checked: {[str(path) for path in candidates]}")


def load_test_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    test_file = resolve_test_file(data_path)
    arrays = np.load(test_file, allow_pickle=True)
    if "x" not in arrays:
        raise KeyError(f"Missing 'x' in {test_file}")
    x = np.asarray(arrays["x"], dtype=np.float64)
    ids = arrays["id"] if "id" in arrays else np.arange(len(x), dtype=np.int64)
    return x, ids.astype(np.int64)


def config_from_checkpoint(checkpoint: dict[str, Any]) -> Config:
    if "feature_extractor_config" in checkpoint:
        d = checkpoint["feature_extractor_config"]
        return Config(
            name=d["name"],
            band_set=d["band_set"],
            feature_set=d["feature_set"],
            raw_norm=d["raw_norm"],
            feature_norm=d["feature_norm"],
            classifier=d["classifier"],
            C=d.get("C"),
        )
    # Backward-compatible path for stable v1 checkpoints.
    return Config("legacy_v1", "checkpoint_bands", "logvar", "none", "none", "logreg", None)


def apply_raw_norm(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "trial_zscore":
        mu = x.mean(axis=-1, keepdims=True)
        sigma = x.std(axis=-1, keepdims=True)
        sigma = np.where(sigma < EPS, 1.0, sigma)
        return (x - mu) / sigma
    raise ValueError(f"Unknown raw normalization: {mode}")


def bandpass(x: np.ndarray, low: float, high: float, sfreq: float, order: int) -> np.ndarray:
    nyq = sfreq / 2.0
    wn = (low / nyq, high / nyq)
    if not (0.0 < wn[0] < wn[1] < 1.0):
        raise ValueError(f"Invalid band cutoffs {low}-{high} Hz for sfreq={sfreq}")
    sos = butter(order, wn, btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def covariance_upper_features(filtered: np.ndarray) -> np.ndarray:
    covs = []
    iu = np.triu_indices(filtered.shape[1])
    for trial in filtered:
        centered = trial - trial.mean(axis=-1, keepdims=True)
        cov = centered @ centered.T / max(1, centered.shape[-1] - 1)
        trace = np.trace(cov)
        if trace > EPS:
            cov = cov / trace
        covs.append(cov[iu])
    return np.asarray(covs, dtype=np.float64)


def band_features(filtered: np.ndarray, feature_set: str) -> np.ndarray:
    var = np.maximum(np.var(filtered, axis=-1), EPS)
    parts = [np.log(var)]
    if feature_set == "logvar":
        return np.concatenate(parts, axis=1)
    if feature_set == "logvar_meanabs":
        parts.append(np.mean(np.abs(filtered), axis=-1))
    elif feature_set == "logvar_bandpower":
        parts.append(np.mean(filtered**2, axis=-1))
    elif feature_set == "logvar_meanstd":
        parts.append(np.mean(filtered, axis=-1))
        parts.append(np.std(filtered, axis=-1))
    elif feature_set == "cov_upper":
        return covariance_upper_features(filtered)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return np.concatenate(parts, axis=1).astype(np.float64)


def bands_from_checkpoint(checkpoint: dict[str, Any]) -> Sequence[Tuple[float, float]]:
    if "band_metadata" in checkpoint:
        return [(float(b["actual_low"]), float(b["actual_high"])) for b in checkpoint["band_metadata"]]
    return [(float(low), float(high)) for low, high in checkpoint["bands"]]


def extract_features(x: np.ndarray, checkpoint: dict[str, Any]) -> np.ndarray:
    config = config_from_checkpoint(checkpoint)
    x = apply_raw_norm(x, config.raw_norm)
    parts = []
    for low, high in bands_from_checkpoint(checkpoint):
        filtered = bandpass(x, low, high, float(checkpoint["sfreq"]), int(checkpoint["filter_order"]))
        parts.append(band_features(filtered, config.feature_set))
    return np.concatenate(parts, axis=1).astype(np.float64)


def apply_subject_feature_normalization(features: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return features
    if mode == "subject":
        mu = features.mean(axis=0)
        sigma = features.std(axis=0)
        sigma = np.where(sigma < EPS, 1.0, sigma)
        return (features - mu) / sigma
    raise ValueError(f"Unknown feature normalization: {mode}")


def preprocess_for_inference(x: np.ndarray, checkpoint: dict[str, Any]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (N, C, T), got {x.shape}")
    expected_channels = checkpoint.get("n_channels")
    if expected_channels is not None and x.shape[1] != expected_channels:
        raise ValueError(f"Channel count mismatch: checkpoint expects {expected_channels}, got {x.shape[1]}")
    features = extract_features(x, checkpoint)
    config = config_from_checkpoint(checkpoint)
    features = apply_subject_feature_normalization(features, config.feature_norm)
    return checkpoint["scaler"].transform(features)


def validate_predictions(pred: np.ndarray, num_examples: int) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.shape != (num_examples,):
        raise ValueError(f"Expected predictions with shape ({num_examples},), got {pred.shape}")
    if not np.issubdtype(pred.dtype, np.integer):
        raise TypeError(f"Predictions must be integers, got dtype {pred.dtype}")
    if np.any((pred < 0) | (pred > 3)):
        raise ValueError("Predicted labels must be integers in {0, 1, 2, 3}")
    return pred.astype(np.int64)


def write_submission(rows: Iterable[Tuple[int, int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    x_test, ids = load_test_data(args.data)
    checkpoint = joblib.load(args.checkpoint)
    x_features = preprocess_for_inference(x_test, checkpoint)
    pred = checkpoint["model"].predict(x_features)
    pred = validate_predictions(pred, len(ids))
    rows = [(int(sample_id), int(label)) for sample_id, label in zip(ids, pred)]
    write_submission(rows, args.output)
    print(f"Wrote {len(rows)} predictions to {args.output}")
    print("Prediction distribution:", dict(zip(*np.unique(pred, return_counts=True))))


if __name__ == "__main__":
    main()
