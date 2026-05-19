"""Proba-level ensemble inference across multiple EA+sliding checkpoints.

Each checkpoint corresponds to a different frequency band (or hyperparameter
configuration). For every test trial:
  1. Apply each checkpoint's preprocessing (notch + bandpass + crop).
  2. Sliding-window split + cov estimation + Euclidean Alignment.
  3. Pipeline predict_proba per window, average across windows -> per-trial proba.
  4. Average per-trial proba across all checkpoints (optionally weighted).
  5. Argmax -> final integer label in {0,1,2,3}.

This script is a development helper for Kaggle submission generation; the
spec-required interface remains the single-checkpoint inference.py.

Run from part1/task2:
    python ensemble_inference.py --data data \\
        --checkpoints outputs/task2_train/task2_model.pt \\
                      outputs/task2_train_beta/task2_model.pt \\
                      outputs/task2_train_broad/task2_model.pt \\
        --output submission_ensemble.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np

from preprocess import (
    PreprocessConfig,
    align_covs,
    apply_ea_whitening,
    compute_ea_whitening,
    estimate_covs,
    filter_and_crop,
    sliding_windows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-checkpoint proba-level ensemble inference."
    )
    p.add_argument("--data", type=Path, required=True,
                   help="Path to task2 data dir (contains test.npz) or test.npz itself.")
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                   help="One or more joblib checkpoints from train.py.")
    p.add_argument("--output", type=Path, default=Path("submission_ensemble.csv"))
    p.add_argument("--weights", type=float, nargs="+", default=None,
                   help="Optional ensemble weights (one per checkpoint). "
                        "If omitted, uniform average.")
    return p.parse_args()


def resolve_test_file(data_path: Path) -> Path:
    if data_path.is_file():
        return data_path
    candidates = [data_path / "test.npz", data_path / "task2" / "test.npz"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate test.npz under {data_path}. Checked: "
        f"{[str(c) for c in candidates]}"
    )


def load_test(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    test_file = resolve_test_file(data_path)
    arrays = np.load(test_file, allow_pickle=True)
    if "x" not in arrays:
        raise KeyError(f"Missing 'x' in {test_file}")
    x = arrays["x"]
    ids = arrays["id"] if "id" in arrays else np.arange(len(x), dtype=np.int64)
    return x, ids.astype(np.int64)


def _per_window_zscore(x: np.ndarray) -> np.ndarray:
    """Per-window, per-channel z-score. x: (N, C, T). Matches train_dl.py."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def proba_for_checkpoint(checkpoint_path: Path, x_raw: np.ndarray) -> np.ndarray:
    """Run one checkpoint's full pipeline on raw test data. Returns (N, n_classes) proba.

    Dispatches on ``ckpt["model_type"]``:
        riemann_ea_sliding, mdm_ea  -> filter + slide + cov + EA + predict_proba
        csp_lda, xdawn_ts           -> filter + slide + predict_proba (pipeline
                                       handles cov / xDAWN internally)
        eegnet_ea_mixup             -> filter + EA-whiten + slide + z-score +
                                       EEGNet softmax
        (missing model_type)        -> assumes the cov + EA path (legacy ckpts)
    """
    ckpt = joblib.load(checkpoint_path)
    pre_cfg = PreprocessConfig.from_dict(ckpt["preprocess_cfg"])
    sliding_cfg = ckpt.get("sliding_cfg", {"window_size": 500, "stride": 100})
    window_size = int(sliding_cfg["window_size"])
    stride = int(sliding_cfg["stride"])
    model_type = ckpt.get("model_type", "riemann_ea_sliding")

    x = filter_and_crop(x_raw, pre_cfg)
    n_trials = len(x)

    if model_type == "eegnet_ea_mixup":
        # EEGNet: test-self EA whitening -> slide -> per-window z-score -> softmax.
        import torch
        from model import EEGNet, ModelConfig

        whitening = compute_ea_whitening(x)
        x_aligned = apply_ea_whitening(x, whitening)
        x_w, n_w = sliding_windows(x_aligned, window_size, stride)
        x_w = _per_window_zscore(x_w)

        net = EEGNet(ModelConfig.from_dict(ckpt["model_kwargs"]))
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        tx = torch.from_numpy(x_w).float().unsqueeze(1)
        probs = []
        with torch.no_grad():
            for i in range(0, len(tx), 128):
                logits = net(tx[i:i + 128].to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        proba_w = np.concatenate(probs, axis=0)
        proba = proba_w.reshape(n_trials, n_w, -1).mean(axis=1)
        return proba

    x_w, n_w = sliding_windows(x, window_size, stride)

    if model_type in ("csp_lda", "xdawn_ts"):
        # Pipeline operates directly on (N, C, T) raw windows.
        proba_w = ckpt["pipeline"].predict_proba(x_w.astype(np.float64))
    else:
        # Cov + EA path for riemann_ea_sliding, mdm_ea, and any legacy ckpt.
        covs = estimate_covs(x_w)
        covs_a, _ = align_covs(covs)
        proba_w = ckpt["pipeline"].predict_proba(covs_a)

    proba = proba_w.reshape(n_trials, n_w, -1).mean(axis=1)
    return proba


def write_submission(rows: Iterable[Tuple[int, int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        raise ValueError(
            f"--weights count ({len(args.weights)}) must match "
            f"--checkpoints count ({len(args.checkpoints)})"
        )

    x_raw, ids = load_test(args.data)
    print(f"[ensemble] test trials: {len(ids)}")

    probas = []
    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"[ensemble] checkpoint {i+1}/{len(args.checkpoints)}: {ckpt_path}")
        proba = proba_for_checkpoint(ckpt_path, x_raw)
        probas.append(proba)
        print(f"  proba shape: {proba.shape}, "
              f"argmax distribution: {np.bincount(proba.argmax(1), minlength=4).tolist()}")

    stacked = np.stack(probas, axis=0)  # (K, N, n_classes)
    if args.weights is None:
        avg = stacked.mean(axis=0)
    else:
        w = np.asarray(args.weights, dtype=np.float64)
        w = w / w.sum()
        avg = (stacked * w[:, None, None]).sum(axis=0)

    pred = avg.argmax(axis=1).astype(np.int64)
    print(f"[ensemble] final label distribution: "
          f"{np.bincount(pred, minlength=4).tolist()}")

    rows = [(int(i), int(l)) for i, l in zip(ids, pred)]
    write_submission(rows, args.output)
    print(f"[ensemble] wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
