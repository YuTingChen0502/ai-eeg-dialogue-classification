import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import joblib
import numpy as np
from scipy.signal import butter, filtfilt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 2 inference and write a Kaggle submission CSV.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the task2 data directory (recommended: ./data) or directly to test.npz.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to your saved model checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Output CSV path. Defaults to ./submission.csv",
    )
    return parser.parse_args()


def resolve_test_file(data_path: Path) -> Path:
    if data_path.is_file():
        return data_path

    candidates = [
        data_path / "test.npz",
        data_path / "task2" / "test.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate Task 2 test data. "
        "If you run from part1/task2, use --data data or --data data/test.npz. "
        f"Checked: {[str(path) for path in candidates]}"
    )


def load_test_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    test_file = resolve_test_file(data_path)
    arrays = np.load(test_file, allow_pickle=True)

    if "x" not in arrays:
        raise KeyError(f"Missing 'x' in {test_file}")

    x = arrays["x"]
    ids = arrays["id"] if "id" in arrays else np.arange(len(x), dtype=np.int64)
    return x, ids.astype(np.int64)


def design_filters(
    bands: Sequence[Tuple[float, float]], sfreq: float, order: int
) -> list:
    nyq = sfreq / 2.0
    filters = []
    for low, high in bands:
        wn = (low / nyq, high / nyq)
        if not (0.0 < wn[0] < wn[1] < 1.0):
            raise ValueError(
                f"Band {low}-{high} Hz invalid for sfreq={sfreq} Hz; "
                f"normalized cutoffs were {wn}."
            )
        b, a = butter(order, wn, btype="bandpass")
        filters.append((b, a))
    return filters


def extract_log_variance(x: np.ndarray, filters) -> np.ndarray:
    parts = []
    for b, a in filters:
        filtered = filtfilt(b, a, x, axis=-1)
        var = np.maximum(np.var(filtered, axis=-1), 1e-10)
        parts.append(np.log(var))
    return np.concatenate(parts, axis=1).astype(np.float64)


def load_checkpoint(checkpoint_path: Path):
    return joblib.load(checkpoint_path)


def preprocess_for_inference(x: np.ndarray, checkpoint) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (N, C, T), got {x.shape}")

    expected_channels = checkpoint.get("n_channels")
    if expected_channels is not None and x.shape[1] != expected_channels:
        raise ValueError(
            f"Channel count mismatch: checkpoint expects {expected_channels}, "
            f"got {x.shape[1]}"
        )

    filters = design_filters(
        checkpoint["bands"], checkpoint["sfreq"], checkpoint["filter_order"]
    )
    feats = extract_log_variance(x, filters)
    return checkpoint["scaler"].transform(feats)


def build_model(checkpoint):
    return checkpoint["model"]


def predict(model, x: np.ndarray) -> np.ndarray:
    pred = model.predict(x)
    pred = np.asarray(pred).astype(np.int64)
    return pred


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
    checkpoint = load_checkpoint(args.checkpoint)
    x_test = preprocess_for_inference(x_test, checkpoint)
    model = build_model(checkpoint)
    pred = predict(model, x_test)
    pred = validate_predictions(pred, len(ids))

    rows = [(int(sample_id), int(label)) for sample_id, label in zip(ids, pred)]
    write_submission(rows, args.output)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
