"""Task 2 inference: confidence-gated ensemble (default) + legacy single-model fallback.

Spec-required CLI (part1_spec.md §5):
    python inference.py --data <data_dir_or_test_file> --checkpoint <ckpt> [--output <csv>]

Optional extension:
    --gate-k <K>   Override the checkpoint's default_gate_k (gated_ensemble only).

Behaviour:
  * If checkpoint["model_type"] == "gated_ensemble":
        Run the strong sub-model (Riemann broad_dense) AND the average of the
        diverse sub-models (3 EEGNet seeds). Keep the strong prediction
        everywhere EXCEPT the K trials where the strong model is least
        confident -- on those, switch to the diverse-average prediction.
        Default K = checkpoint["default_gate_k"] (= 3 from train.py).
        With K=3 this reproduces gated_k3 (Kaggle public LB 0.8750).
        Pass --gate-k 7 to reproduce gated_k7 (also 0.8750).

  * If checkpoint is a single-model legacy format (riemann_ea_sliding /
    eegnet_ea_mixup / mdm_ea / csp_lda / xdawn_ts), inference runs that
    model directly. This branch is kept for backward compatibility.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np

from SUBMISSION.task2_submission.preprocess import (
    PreprocessConfig,
    align_covs,
    apply_ea_whitening,
    compute_ea_whitening,
    estimate_covs,
    filter_and_crop,
    sliding_windows,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Task 2 inference and write a Kaggle submission CSV.",
    )
    p.add_argument("--data", type=Path, required=True,
                   help="Path to the task2 data directory (recommended: ./data) "
                        "or directly to test.npz.")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to the bundled checkpoint produced by train.py.")
    p.add_argument("--output", type=Path, default=Path("submission.csv"),
                   help="Output CSV path. Defaults to ./submission.csv")
    p.add_argument("--gate-k", type=int, default=None,
                   help="(Optional, gated_ensemble checkpoints only) Override "
                        "the checkpoint's default_gate_k. With the released "
                        "checkpoint, default 3 reproduces gated_k3 (LB 0.8750); "
                        "use 7 to reproduce gated_k7 (also LB 0.8750).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


def load_checkpoint(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return joblib.load(checkpoint_path)


# ---------------------------------------------------------------------------
# Helpers shared by both sub-model types
# ---------------------------------------------------------------------------

def _per_window_zscore(x: np.ndarray) -> np.ndarray:
    """Per-window, per-channel z-score. x: (N, C, T). Matches train.py."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def _proba_riemann_subckpt(sub_ckpt: dict, x_raw: np.ndarray) -> np.ndarray:
    """Riemann pipeline path: filter+crop -> slide -> cov -> EA -> predict_proba.

    Returns trial-level (N, n_classes) proba.
    """
    pre_cfg = PreprocessConfig.from_dict(sub_ckpt["preprocess_cfg"])
    sliding_cfg = sub_ckpt.get("sliding_cfg", {"window_size": 400, "stride": 50})
    window_size = int(sliding_cfg["window_size"])
    stride = int(sliding_cfg["stride"])

    x = filter_and_crop(x_raw, pre_cfg)
    n_trials = len(x)
    x_w, n_w = sliding_windows(x, window_size, stride)
    covs = estimate_covs(x_w)
    covs_a, _ = align_covs(covs)                # test-self EA on covariances
    proba_w = sub_ckpt["pipeline"].predict_proba(covs_a)
    return proba_w.reshape(n_trials, n_w, -1).mean(axis=1)


def _proba_eegnet_subckpt(sub_ckpt: dict, x_raw: np.ndarray) -> np.ndarray:
    """EEGNet path: filter+crop -> EA-whiten -> slide -> zscore -> EEGNet softmax.

    Returns trial-level (N, n_classes) proba.
    """
    import torch
    from SUBMISSION.task2_submission.model import EEGNet, ModelConfig

    pre_cfg = PreprocessConfig.from_dict(sub_ckpt["preprocess_cfg"])
    sliding_cfg = sub_ckpt.get("sliding_cfg", {"window_size": 500, "stride": 100})
    window_size = int(sliding_cfg["window_size"])
    stride = int(sliding_cfg["stride"])

    x = filter_and_crop(x_raw, pre_cfg)
    whitening = compute_ea_whitening(x)         # test-self EA on raw signal
    x_aligned = apply_ea_whitening(x, whitening)
    n_trials = len(x_aligned)
    x_w, n_w = sliding_windows(x_aligned, window_size, stride)
    x_w = _per_window_zscore(x_w)

    model_cfg = ModelConfig.from_dict(sub_ckpt["model_kwargs"])
    net = EEGNet(model_cfg)
    net.load_state_dict(sub_ckpt["state_dict"])
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
    return proba_w.reshape(n_trials, n_w, -1).mean(axis=1)


def _proba_for_subckpt(sub_ckpt: dict, x_raw: np.ndarray) -> np.ndarray:
    """Dispatch a single sub-checkpoint to the right proba routine."""
    sub_type = sub_ckpt.get("model_type", "riemann_ea_sliding")
    if sub_type == "eegnet_ea_mixup":
        return _proba_eegnet_subckpt(sub_ckpt, x_raw)
    # All cov+EA paths (riemann_ea_sliding / mdm_ea / etc.) share this routine.
    return _proba_riemann_subckpt(sub_ckpt, x_raw)


# ---------------------------------------------------------------------------
# Gated ensemble (main path for our bundled checkpoint)
# ---------------------------------------------------------------------------

def predict_gated(checkpoint: dict, x_raw: np.ndarray, gate_k: int) -> np.ndarray:
    """Confidence-gated ensemble inference.

    Steps:
      1. Run the strong sub-model -> proba_strong, strong_pred, strong_conf.
      2. Run each diverse sub-model -> average proba_diverse, diverse_pred.
      3. Take the K trials where strong_conf is lowest.
      4. Replace strong_pred at those K trials with diverse_pred.
      5. Return final integer labels in {0,1,2,3}.
    """
    strong = checkpoint["strong"]
    diverse_list = checkpoint["diverse"]
    n_trials = len(x_raw)
    k = max(0, min(gate_k, n_trials))

    print(f"[inference] gated_ensemble | gate_k = {k} of {n_trials} trials")
    print(f"[inference] strong checkpoint type: {strong.get('model_type', '?')}")
    proba_strong = _proba_for_subckpt(strong, x_raw)
    strong_pred = proba_strong.argmax(axis=1)
    strong_conf = proba_strong.max(axis=1)
    print(f"[inference] strong  argmax dist: "
          f"{np.bincount(strong_pred, minlength=4).tolist()}")

    diverse_probas = []
    for i, d in enumerate(diverse_list, 1):
        print(f"[inference] diverse #{i} (seed={d.get('seed', '?')}, "
              f"type={d.get('model_type', '?')})")
        diverse_probas.append(_proba_for_subckpt(d, x_raw))
    proba_diverse = np.mean(diverse_probas, axis=0)
    diverse_pred = proba_diverse.argmax(axis=1)
    print(f"[inference] diverse avg argmax dist: "
          f"{np.bincount(diverse_pred, minlength=4).tolist()}")

    # Pick the K trials where the strong model is least confident.
    low_conf_idx = np.argsort(strong_conf)[:k]
    final = strong_pred.copy()
    n_changed = 0
    for idx in low_conf_idx:
        if diverse_pred[idx] != strong_pred[idx]:
            n_changed += 1
        final[idx] = diverse_pred[idx]

    print(f"[inference] gated {k} least-confident strong predictions "
          f"(confidences {np.sort(strong_conf)[:k].round(3).tolist()})")
    print(f"[inference] of those, {n_changed} predictions actually changed")
    print(f"[inference] final argmax dist: "
          f"{np.bincount(final, minlength=4).tolist()}")
    return final.astype(np.int64)


# ---------------------------------------------------------------------------
# Legacy single-model fallback (kept for backward compat with old checkpoints)
# ---------------------------------------------------------------------------

def predict_single(checkpoint: dict, x_raw: np.ndarray) -> np.ndarray:
    """Single-checkpoint inference (no gated ensemble). Returns int labels."""
    print(f"[inference] single-model | type = {checkpoint.get('model_type', '?')}")
    proba = _proba_for_subckpt(checkpoint, x_raw)
    pred = proba.argmax(axis=1).astype(np.int64)
    print(f"[inference] argmax dist: {np.bincount(pred, minlength=4).tolist()}")
    return pred


# ---------------------------------------------------------------------------
# Output validation + CSV
# ---------------------------------------------------------------------------

def validate_predictions(pred: np.ndarray, num_examples: int) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.shape != (num_examples,):
        raise ValueError(
            f"Expected predictions with shape ({num_examples},), got {pred.shape}")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    x_test, ids = load_test_data(args.data)
    print(f"[inference] loaded {len(ids)} test trials from {args.data}")
    checkpoint = load_checkpoint(args.checkpoint)
    model_type = checkpoint.get("model_type", "riemann_ea_sliding")
    print(f"[inference] checkpoint model_type: {model_type}")

    if model_type == "gated_ensemble":
        default_k = int(checkpoint.get("default_gate_k", 3))
        gate_k = int(args.gate_k) if args.gate_k is not None else default_k
        pred = predict_gated(checkpoint, x_test, gate_k)
    else:
        pred = predict_single(checkpoint, x_test)

    pred = validate_predictions(pred, len(ids))
    rows = [(int(sid), int(lbl)) for sid, lbl in zip(ids, pred)]
    write_submission(rows, args.output)
    print(f"[inference] wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
