"""Alternative cross-subject MI training methods.

The Riemann+EA+LR pipeline in train.py plateaus at Kaggle 0.6875 across very
diverse ensembles, meaning ~5/16 test trials require a fundamentally different
feature space to classify. This script provides three alternative pipelines
that live in different feature spaces, designed to disagree with Riemann+EA
on those hard trials:

  --method csp_lda   : Covariances -> CSP (log-variance) -> LDA
                       Classical BCI baseline. Feature space = CSP-projected
                       channel variance. Very different from Riemann tangent.

  --method xdawn_ts  : XdawnCovariances -> TangentSpace -> LogisticRegression
                       Supervised spatial filtering (class-conditional means)
                       combined with Riemannian tangent-space features.

  --method mdm_ea    : Same EA preprocessing as train.py, but classifies by
                       Riemannian distance to class means (MDM) instead of LR.
                       Different decision boundary mechanism.

All three save a checkpoint with a distinct `model_type` value that
ensemble_inference.py can dispatch on.

Run from part1/task2:
    python train_alt.py --method csp_lda --band-low 13 --band-high 30 \\
        --data data/train --output-dir outputs/task2_train_csp_beta \\
        --checkpoint outputs/task2_train_csp_beta/task2_model.pt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import joblib
import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.spatialfilters import CSP as RiemannCSP
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from SUBMISSION.task2_submission.preprocess import (
    PreprocessConfig,
    align_covs,
    estimate_covs,
    filter_and_crop,
    sliding_windows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train alternative cross-subject MI methods.",
    )
    p.add_argument("--method", type=str, required=True,
                   choices=["csp_lda", "xdawn_ts", "mdm_ea"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-size", type=int, default=500)
    p.add_argument("--stride", type=int, default=100)
    p.add_argument("--band-low", type=float, default=8.0)
    p.add_argument("--band-high", type=float, default=30.0)
    p.add_argument("--n-csp", type=int, default=6)
    p.add_argument("--n-xdawn", type=int, default=4)
    p.add_argument("--lr-c", type=float, default=0.1)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_subject(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    arrs = np.load(npz_path, allow_pickle=True)
    if "x" not in arrs or "y" not in arrs:
        raise KeyError(f"{npz_path} missing 'x' or 'y'")
    return arrs["x"].astype(np.float32), arrs["y"].astype(np.int64)


def load_all_subjects(
    data_dir: Path, cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_parts, y_parts, sid_parts = [], [], []
    for sid in range(1, 11):
        f = data_dir / f"subject{sid:02d}.npz"
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")
        x, y = load_subject(f)
        x = filter_and_crop(x, cfg)
        x_parts.append(x)
        y_parts.append(y)
        sid_parts.append(np.full(len(y), sid, dtype=np.int64))
    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(sid_parts, axis=0),
    )


def make_csp_lda_pipeline(n_csp: int):
    """Cov -> CSP (log-var features) -> LDA. Input: (N, C, T) windows."""
    return make_pipeline(
        Covariances(estimator="oas"),
        RiemannCSP(nfilter=n_csp, metric="euclid", log=True),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def make_xdawn_ts_pipeline(n_xdawn: int, c: float):
    """XdawnCovariances -> TS -> LR. Input: (N, C, T) windows."""
    return make_pipeline(
        XdawnCovariances(nfilter=n_xdawn, estimator="oas"),
        TangentSpace(),
        LogisticRegression(C=c, max_iter=2000, solver="lbfgs"),
    )


def make_mdm_pipeline():
    """MDM on already-aligned covariances. Input: (N, C, C)."""
    return make_pipeline(MDM(metric="riemann"))


def loso_cv_csp_or_xdawn(
    x: np.ndarray, y: np.ndarray, sid: np.ndarray,
    window_size: int, stride: int, pipe_factory,
) -> tuple[float, float, list[tuple[int, float]]]:
    """LOSO with sliding-window TTA, pipeline takes raw windows."""
    subjects = sorted(np.unique(sid).tolist())
    fold_accs: list[tuple[int, float]] = []
    for held in subjects:
        tr_mask = sid != held
        va_mask = sid == held

        x_tr_w, n_w_tr = sliding_windows(x[tr_mask], window_size, stride)
        y_tr_w = np.repeat(y[tr_mask], n_w_tr)
        x_va_w, n_w_va = sliding_windows(x[va_mask], window_size, stride)

        pipe = pipe_factory()
        pipe.fit(x_tr_w.astype(np.float64), y_tr_w)
        proba_w = pipe.predict_proba(x_va_w.astype(np.float64))
        proba = proba_w.reshape(int(va_mask.sum()), n_w_va, -1).mean(axis=1)
        acc = float((proba.argmax(axis=1) == y[va_mask]).mean())
        fold_accs.append((int(held), acc))
        print(f"[LOSO] hold subject{held:02d}: val_acc = {acc:.4f}")

    accs = np.array([a for _, a in fold_accs])
    return float(accs.mean()), float(accs.std()), fold_accs


def loso_cv_mdm_ea(
    x: np.ndarray, y: np.ndarray, sid: np.ndarray,
    window_size: int, stride: int,
) -> tuple[float, float, list[tuple[int, float]]]:
    """LOSO with per-subject EA, then MDM on aligned covs."""
    subjects = sorted(np.unique(sid).tolist())
    fold_accs: list[tuple[int, float]] = []
    for held in subjects:
        tr_mask = sid != held
        va_mask = sid == held

        covs_parts, y_parts = [], []
        for s in subjects:
            if s == held:
                continue
            s_mask = sid == s
            x_s_w, n_w = sliding_windows(x[s_mask], window_size, stride)
            covs_s = estimate_covs(x_s_w)
            covs_s_a, _ = align_covs(covs_s)
            covs_parts.append(covs_s_a)
            y_parts.append(np.repeat(y[s_mask], n_w))
        covs_tr = np.concatenate(covs_parts, axis=0)
        y_tr = np.concatenate(y_parts, axis=0)

        x_va_w, n_w_va = sliding_windows(x[va_mask], window_size, stride)
        covs_va = estimate_covs(x_va_w)
        covs_va_a, _ = align_covs(covs_va)

        pipe = make_mdm_pipeline()
        pipe.fit(covs_tr, y_tr)
        proba_w = pipe.predict_proba(covs_va_a)
        proba = proba_w.reshape(int(va_mask.sum()), n_w_va, -1).mean(axis=1)
        acc = float((proba.argmax(axis=1) == y[va_mask]).mean())
        fold_accs.append((int(held), acc))
        print(f"[LOSO+EA+MDM] hold subject{held:02d}: val_acc = {acc:.4f}")

    accs = np.array([a for _, a in fold_accs])
    return float(accs.mean()), float(accs.std()), fold_accs


def fit_csp_or_xdawn_final(x, y, window_size, stride, pipe_factory):
    x_w, n_w = sliding_windows(x, window_size, stride)
    y_w = np.repeat(y, n_w)
    pipe = pipe_factory()
    pipe.fit(x_w.astype(np.float64), y_w)
    return pipe, n_w


def fit_mdm_ea_final(x, y, sid, window_size, stride):
    """Per-subject EA across all 10 subjects, then MDM."""
    covs_parts, y_parts = [], []
    n_w_used = None
    for s in sorted(np.unique(sid).tolist()):
        s_mask = sid == s
        x_s_w, n_w = sliding_windows(x[s_mask], window_size, stride)
        covs_s = estimate_covs(x_s_w)
        covs_s_a, _ = align_covs(covs_s)
        covs_parts.append(covs_s_a)
        y_parts.append(np.repeat(y[s_mask], n_w))
        n_w_used = n_w
    covs_all = np.concatenate(covs_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    pipe = make_mdm_pipeline()
    pipe.fit(covs_all, y_all)
    return pipe, int(n_w_used)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    pre_cfg = PreprocessConfig(
        band_low=args.band_low,
        band_high=args.band_high,
    )
    print(f"[train_alt] method={args.method}, preprocess={pre_cfg.to_dict()}")
    print(f"[train_alt] sliding: ws={args.window_size}, stride={args.stride}")

    x_all, y_all, sid_all = load_all_subjects(args.data, pre_cfg)
    print(f"[train_alt] all subjects: x={x_all.shape}, classes={np.bincount(y_all)}")

    if args.method == "csp_lda":
        factory = lambda: make_csp_lda_pipeline(args.n_csp)
        cv_mean, cv_std, per_subj = loso_cv_csp_or_xdawn(
            x_all, y_all, sid_all, args.window_size, args.stride, factory,
        )
        final_pipe, n_w = fit_csp_or_xdawn_final(
            x_all, y_all, args.window_size, args.stride, factory,
        )
    elif args.method == "xdawn_ts":
        factory = lambda: make_xdawn_ts_pipeline(args.n_xdawn, args.lr_c)
        cv_mean, cv_std, per_subj = loso_cv_csp_or_xdawn(
            x_all, y_all, sid_all, args.window_size, args.stride, factory,
        )
        final_pipe, n_w = fit_csp_or_xdawn_final(
            x_all, y_all, args.window_size, args.stride, factory,
        )
    elif args.method == "mdm_ea":
        cv_mean, cv_std, per_subj = loso_cv_mdm_ea(
            x_all, y_all, sid_all, args.window_size, args.stride,
        )
        final_pipe, n_w = fit_mdm_ea_final(
            x_all, y_all, sid_all, args.window_size, args.stride,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"[train_alt] LOSO CV = {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"[train_alt] fitted final pipeline, n_w={n_w}")

    log_lines = [
        f"Method: {args.method}",
        f"LOSO CV mean = {cv_mean:.4f}",
        f"LOSO CV std  = {cv_std:.4f}",
        "Per-subject accuracies:",
    ]
    log_lines += [f"  subject{s:02d}: {a:.4f}" for s, a in per_subj]
    (args.output_dir / "training_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8",
    )

    payload = {
        "model_type": args.method,
        "pipeline": final_pipe,
        "preprocess_cfg": pre_cfg.to_dict(),
        "sliding_cfg": {
            "window_size": args.window_size,
            "stride": args.stride,
        },
        "loso_cv_acc_mean": cv_mean,
        "loso_cv_acc_std": cv_std,
        "loso_per_subject": per_subj,
    }
    joblib.dump(payload, args.checkpoint)
    print(f"[train_alt] wrote checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
