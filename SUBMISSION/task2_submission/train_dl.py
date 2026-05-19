"""Task 2 deep-learning training: EEGNet + Euclidean Alignment + Mixup.

Combo 1 from the SOTA literature survey (target 0.85-0.92 on cross-subject MI):

  1. Per-subject Euclidean Alignment (EA) on the *raw* signal -- whiten each
     subject's trials so the deep model sees subject-harmonised input.
  2. Sliding-window augmentation -- each trial -> several 2s windows.
  3. Per-window z-score so EEGNet gets normalised input.
  4. EEGNet backbone (from model.py).
  5. Mixup augmentation in the training loop.
  6. Cosine-annealing LR schedule.
  7. LOSO validation (hold one subject out) for an honest cross-subject CV.

Train one checkpoint per --seed; combine multiple seeds with
ensemble_inference.py for a multi-seed ensemble.

Run from part1/task2:
    python train_dl.py --data data/train \\
        --output-dir outputs/task2_dl_s42 \\
        --checkpoint outputs/task2_dl_s42/task2_model.pt \\
        --epochs 200 --batch-size 64 --learning-rate 0.001 \\
        --weight-decay 0.0001 --seed 42 --device auto \\
        --band-low 8 --band-high 30 --window-size 500 --stride 100 \\
        --mixup-alpha 0.2
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from SUBMISSION.task2_submission.model import EEGNet, ModelConfig
from SUBMISSION.task2_submission.preprocess import (
    PreprocessConfig,
    apply_ea_whitening,
    compute_ea_whitening,
    filter_and_crop,
    sliding_windows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task 2 EEGNet + EA + Mixup deep-learning training.",
    )
    p.add_argument("--data", type=Path, required=True,
                   help="Directory containing subject01.npz ... subject10.npz")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--val-subject", type=int, default=10,
                   help="Subject id (1-10) held out for LOSO validation.")
    p.add_argument("--band-low", type=float, default=8.0)
    p.add_argument("--band-high", type=float, default=30.0)
    p.add_argument("--window-size", type=int, default=500)
    p.add_argument("--stride", type=int, default=100)
    p.add_argument("--mixup-alpha", type=float, default=0.2,
                   help="Beta distribution parameter for Mixup. 0 disables Mixup.")
    p.add_argument("--dropout", type=float, default=0.5)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def load_subject(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    arrs = np.load(npz_path, allow_pickle=True)
    if "x" not in arrs or "y" not in arrs:
        raise KeyError(f"{npz_path} missing 'x' or 'y'")
    return arrs["x"].astype(np.float32), arrs["y"].astype(np.int64)


def _per_window_zscore(x: np.ndarray) -> np.ndarray:
    """Per-window, per-channel z-score. x: (N, C, T)."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def load_subject_windows(
    data_dir: Path, sid: int, cfg: PreprocessConfig,
    window_size: int, stride: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load one subject -> filter+crop -> per-subject EA -> sliding windows
    -> per-window z-score. Returns (windows, labels_expanded, n_windows)."""
    f = data_dir / f"subject{sid:02d}.npz"
    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")
    x, y = load_subject(f)
    x = filter_and_crop(x, cfg)
    whitening = compute_ea_whitening(x)        # per-subject EA reference
    x = apply_ea_whitening(x, whitening)
    x_w, n_w = sliding_windows(x, window_size, stride)
    x_w = _per_window_zscore(x_w)
    y_w = np.repeat(y, n_w)
    return x_w, y_w, n_w


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool,
                num_workers: int) -> DataLoader:
    tx = torch.from_numpy(x).float().unsqueeze(1)  # (N, 1, C, T)
    ty = torch.from_numpy(y).long()
    return DataLoader(
        TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=False,
    )


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Mixup: returns mixed_x, y_a, y_b, lam.

    If alpha <= 0, returns the batch unchanged with lam = 1.0.
    """
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def train_epoch(model, loader, optimizer, criterion, device, mixup_alpha):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        mixed_x, y_a, y_b, lam = mixup_batch(xb, yb, mixup_alpha)
        logits = model(mixed_x)
        loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * yb.size(0)
        # Accuracy reported against the original (un-permuted) label.
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_n += yb.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * yb.size(0)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_n += yb.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def eval_per_trial(model, x_w, y_trial, n_w, device, batch_size=128):
    """Window-level TTA: average proba across the n_w windows of each trial."""
    model.eval()
    tx = torch.from_numpy(x_w).float().unsqueeze(1)
    probs = []
    for i in range(0, len(tx), batch_size):
        logits = model(tx[i:i + batch_size].to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    proba_w = np.concatenate(probs, axis=0)
    proba = proba_w.reshape(len(y_trial), n_w, -1).mean(axis=1)
    pred = proba.argmax(axis=1)
    return float((pred == y_trial).mean())


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    pre_cfg = PreprocessConfig(band_low=args.band_low, band_high=args.band_high)
    print(f"[train_dl] device={device}, seed={args.seed}")
    print(f"[train_dl] preprocess: {pre_cfg.to_dict()}")
    print(f"[train_dl] sliding: ws={args.window_size}, stride={args.stride}, "
          f"mixup_alpha={args.mixup_alpha}")

    # ---- Load all subjects: per-subject EA + sliding windows ----
    train_x_parts, train_y_parts = [], []
    val_x_w = val_y_trial = None
    val_n_w = 0
    for sid in range(1, 11):
        x_w, y_w, n_w = load_subject_windows(
            args.data, sid, pre_cfg, args.window_size, args.stride,
        )
        if sid == args.val_subject:
            val_x_w = x_w
            val_y_trial = y_w.reshape(-1, n_w)[:, 0]  # one label per trial
            val_n_w = n_w
        else:
            train_x_parts.append(x_w)
            train_y_parts.append(y_w)
    x_tr = np.concatenate(train_x_parts, axis=0)
    y_tr = np.concatenate(train_y_parts, axis=0)
    print(f"[train_dl] train windows: {x_tr.shape}, classes={np.bincount(y_tr)}")
    print(f"[train_dl] val subject {args.val_subject}: "
          f"{val_x_w.shape} -> {len(val_y_trial)} trials, n_w={val_n_w}")

    n_channels, n_samples = x_tr.shape[1], x_tr.shape[2]
    model_cfg = ModelConfig(
        n_channels=n_channels, n_samples=n_samples,
        n_classes=4, dropout=args.dropout,
    )
    print(f"[train_dl] model: {model_cfg.to_dict()}")

    train_loader = make_loader(x_tr, y_tr, args.batch_size, True, args.num_workers)
    val_loader = make_loader(
        val_x_w, np.repeat(val_y_trial, val_n_w),
        args.batch_size, False, args.num_workers,
    )

    model = EEGNet(model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_trial_acc = -1.0
    best_state = None
    log_lines = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.mixup_alpha,
        )
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        trial_acc = eval_per_trial(model, val_x_w, val_y_trial, val_n_w, device)
        scheduler.step()
        line = (f"epoch {epoch:3d}/{args.epochs} | "
                f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                f"val(window) loss={va_loss:.4f} acc={va_acc:.4f} | "
                f"val(trial-TTA) acc={trial_acc:.4f}")
        if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs:
            print(line)
        log_lines.append(line)
        if trial_acc > best_trial_acc:
            best_trial_acc = trial_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}

    (args.output_dir / "training_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8",
    )

    payload = {
        "model_type": "eegnet_ea_mixup",
        "state_dict": best_state,
        "model_kwargs": model_cfg.to_dict(),
        "preprocess_cfg": pre_cfg.to_dict(),
        "sliding_cfg": {"window_size": args.window_size, "stride": args.stride},
        "seed": args.seed,
        "mixup_alpha": args.mixup_alpha,
        "best_val_acc": best_trial_acc,
        "val_subject": args.val_subject,
    }
    joblib.dump(payload, args.checkpoint)
    print(f"[train_dl] best LOSO trial-TTA acc = {best_trial_acc:.4f}")
    print(f"[train_dl] wrote checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
