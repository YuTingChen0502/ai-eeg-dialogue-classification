"""Task 2: cross-subject EEG classification training pipeline.

Pipeline:
- load all subject npz files from --data
- bandpass filter into multiple frequency bands
- per-band channel-wise log-variance features, concatenated
- standardize, train LogisticRegression
- save joblib checkpoint with all preprocessing metadata
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

DEFAULT_BANDS: Tuple[Tuple[float, float], ...] = (
    (8.0, 13.0),
    (13.0, 30.0),
    (4.0, 40.0),
)
CHECKPOINT_VERSION = "task2-logvar-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task 2 cross-subject EEG classifier.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory containing subjectXX.npz files (e.g., data/train).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/task2_train"),
        help="Directory for run artifacts.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to <output-dir>/task2_model.pkl.",
    )
    parser.add_argument("--sfreq", type=float, default=250.0, help="Assumed sampling frequency (Hz).")
    parser.add_argument("--filter-order", type=int, default=4, help="Butterworth filter order.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for LogisticRegression.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum solver iterations for LogisticRegression.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run leave-one-subject-out validation before final training.",
    )
    return parser.parse_args()


def load_subjects(data_dir: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    files = sorted(data_dir.glob("subject*.npz"))
    if not files:
        raise FileNotFoundError(f"No subjectXX.npz files found in {data_dir}")
    subjects = []
    for f in files:
        arr = np.load(f, allow_pickle=True)
        if "x" not in arr or "y" not in arr:
            raise KeyError(f"{f} is missing required 'x' or 'y' arrays")
        x = np.asarray(arr["x"], dtype=np.float64)
        y = np.asarray(arr["y"], dtype=np.int64)
        if x.ndim != 3:
            raise ValueError(f"{f}: expected x with shape (N, C, T), got {x.shape}")
        if len(x) != len(y):
            raise ValueError(f"{f}: x has {len(x)} trials but y has {len(y)} labels")
        subjects.append((f.stem, x, y))
    return subjects


def design_filters(
    bands: Sequence[Tuple[float, float]], sfreq: float, order: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
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


def extract_log_variance(
    x: np.ndarray, filters: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Return (N, n_bands * n_channels) log-variance features."""
    parts = []
    for b, a in filters:
        filtered = filtfilt(b, a, x, axis=-1)
        var = np.var(filtered, axis=-1)
        var = np.maximum(var, 1e-10)
        parts.append(np.log(var))
    return np.concatenate(parts, axis=1).astype(np.float64)


def fit_classifier(
    x_features: np.ndarray, y: np.ndarray, *, C: float, max_iter: int, seed: int
) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(x_features)
    x_scaled = scaler.transform(x_features)
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=seed,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=None,
    )
    clf.fit(x_scaled, y)
    return scaler, clf


def loso_validate(
    subjects: List[Tuple[str, np.ndarray, np.ndarray]],
    filters,
    *,
    C: float,
    max_iter: int,
    seed: int,
) -> None:
    accs, f1s = [], []
    for i in range(len(subjects)):
        train_pool = [s for j, s in enumerate(subjects) if j != i]
        held = subjects[i]

        x_tr = np.concatenate([s[1] for s in train_pool], axis=0)
        y_tr = np.concatenate([s[2] for s in train_pool], axis=0)
        x_va, y_va = held[1], held[2]

        f_tr = extract_log_variance(x_tr, filters)
        f_va = extract_log_variance(x_va, filters)
        scaler, clf = fit_classifier(f_tr, y_tr, C=C, max_iter=max_iter, seed=seed)
        pred = clf.predict(scaler.transform(f_va))

        acc = accuracy_score(y_va, pred)
        f1 = f1_score(y_va, pred, average="macro")
        accs.append(acc)
        f1s.append(f1)
        print(f"  LOSO fold {i:02d} (held-out: {held[0]}): acc={acc:.4f}  macro_f1={f1:.4f}")
    print(
        f"LOSO mean: acc={np.mean(accs):.4f} ± {np.std(accs):.4f}  "
        f"macro_f1={np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or (args.output_dir / "task2_model.pkl")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    bands = list(DEFAULT_BANDS)
    filters = design_filters(bands, args.sfreq, args.filter_order)

    subjects = load_subjects(args.data)
    n_trials = sum(len(s[1]) for s in subjects)
    n_channels = subjects[0][1].shape[1]
    n_time = subjects[0][1].shape[2]
    print(
        f"Loaded {len(subjects)} subjects, {n_trials} trials total, "
        f"shape per trial=({n_channels}, {n_time})"
    )

    if args.validate:
        print("Running leave-one-subject-out validation...")
        loso_validate(
            subjects,
            filters,
            C=args.C,
            max_iter=args.max_iter,
            seed=args.seed,
        )

    x_all = np.concatenate([s[1] for s in subjects], axis=0)
    y_all = np.concatenate([s[2] for s in subjects], axis=0)

    print(f"Fitting final model on {len(x_all)} trials...")
    f_all = extract_log_variance(x_all, filters)
    scaler, clf = fit_classifier(
        f_all, y_all, C=args.C, max_iter=args.max_iter, seed=args.seed
    )

    train_pred = clf.predict(scaler.transform(f_all))
    print(
        f"Train acc={accuracy_score(y_all, train_pred):.4f}  "
        f"macro_f1={f1_score(y_all, train_pred, average='macro'):.4f}"
    )

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "feature_type": "log_variance",
        "bands": [tuple(b) for b in bands],
        "sfreq": float(args.sfreq),
        "filter_order": int(args.filter_order),
        "n_channels": int(n_channels),
        "n_time_points": int(n_time),
        "scaler": scaler,
        "model": clf,
        "classes": clf.classes_.astype(np.int64).tolist(),
        "label_mapping": None,
        "train_args": {
            "C": float(args.C),
            "max_iter": int(args.max_iter),
            "seed": int(args.seed),
            "data_dir": str(args.data),
        },
    }
    joblib.dump(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
