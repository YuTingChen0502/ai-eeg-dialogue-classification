"""Task 2 unified trainer: produces the gated-ensemble bundled checkpoint.

ONE invocation trains four sub-models and bundles them into one checkpoint
that ``inference.py`` can load:

    [1/4] STRONG model -- Riemann + per-subject EA on covariances + dense
          sliding window + TangentSpace + Logistic Regression on broad band
          (8-30 Hz, ws=400, stride=50). This is the verified Kaggle-0.8125
          single model (`broad_dense`).

    [2/4]..[4/4] DIVERSE models -- three EEGNet seeds (42, 1, 7) trained with
          per-subject Euclidean Alignment on the raw signal, Mixup
          augmentation, cosine LR. These supply the "second opinion" for the
          confidence-gated inference path.

The bundled checkpoint follows this schema:

    {
        "model_type": "gated_ensemble",
        "default_gate_k": 3,
        "strong":  {<one Riemann sub-checkpoint dict>},
        "diverse": [{<EEGNet seed 42>}, {<seed 1>}, {<seed 7>}],
    }

inference.py loads this bundle, runs the strong model and the average of the
three diverse models, then keeps the strong-model prediction everywhere
EXCEPT the K=3 trials where the strong model is least confident -- on those
the average-diverse prediction takes over. This reproduces our Kaggle
public-LB 0.8750 submission (gated_k3) by default.

Run from part1/task2:
    bash train_command.txt
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from SUBMISSION.task2_submission.model import EEGNet, ModelConfig
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
# CLI -- mirrors the spec example flag set so train_command.txt stays a
# one-line drop-in. The values are used for the EEGNet sub-models; the Riemann
# strong model has its own (hard-coded) hyperparameters embedded below.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task 2 unified gated-ensemble trainer."
    )
    p.add_argument("--data", type=Path, required=True,
                   help="Directory containing subject01.npz ... subject10.npz")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=120,
                   help="EEGNet training epochs.")
    p.add_argument("--batch-size", type=int, default=32,
                   help="EEGNet mini-batch size.")
    p.add_argument("--learning-rate", type=float, default=5e-4,
                   help="EEGNet Adam learning rate.")
    p.add_argument("--weight-decay", type=float, default=5e-4,
                   help="EEGNet Adam weight decay (L2).")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; the diverse seeds are fixed at [42, 1, 7] "
                        "for reproducibility regardless of this flag.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    # Internal -- override only if you know what you are doing.
    p.add_argument("--diverse-dropout", type=float, default=0.6)
    p.add_argument("--diverse-mixup-alpha", type=float, default=0.3)
    p.add_argument("--default-gate-k", type=int, default=3,
                   help="Stored in the checkpoint as the default gate-k for "
                        "inference.py (override with inference.py --gate-k).")
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


# ---------------------------------------------------------------------------
# STRONG sub-model: Riemann + per-subject EA on covariances + dense sliding
# ---------------------------------------------------------------------------

STRONG_BAND = (8.0, 30.0)
STRONG_WINDOW = 400
STRONG_STRIDE = 50
STRONG_LR_C = 0.1


def train_strong(data_dir: Path) -> dict:
    """Train the Riemann broad_dense pipeline (Kaggle-verified 0.8125 alone)."""
    pre_cfg = PreprocessConfig(band_low=STRONG_BAND[0], band_high=STRONG_BAND[1])
    print(f"  band: {STRONG_BAND}, window={STRONG_WINDOW}, stride={STRONG_STRIDE}, "
          f"LR-C={STRONG_LR_C}")

    covs_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    n_w_used: int | None = None
    for sid in range(1, 11):
        f = data_dir / f"subject{sid:02d}.npz"
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")
        x, y = load_subject(f)
        x = filter_and_crop(x, pre_cfg)
        x_w, n_w = sliding_windows(x, STRONG_WINDOW, STRONG_STRIDE)
        covs = estimate_covs(x_w)
        covs_a, _ = align_covs(covs)            # per-subject EA on covariances
        covs_parts.append(covs_a)
        y_parts.append(np.repeat(y, n_w))
        if n_w_used is None:
            n_w_used = n_w

    covs_all = np.concatenate(covs_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    print(f"  fit pipeline on {covs_all.shape[0]} windows "
          f"({covs_all.shape[0] // (n_w_used or 1)} trials x {n_w_used} windows)")

    pipeline = make_pipeline(
        TangentSpace(),
        LogisticRegression(C=STRONG_LR_C, max_iter=2000, solver="lbfgs"),
    )
    pipeline.fit(covs_all, y_all)

    return {
        "model_type": "riemann_ea_sliding",
        "pipeline": pipeline,
        "preprocess_cfg": pre_cfg.to_dict(),
        "sliding_cfg": {"window_size": STRONG_WINDOW, "stride": STRONG_STRIDE},
    }


# ---------------------------------------------------------------------------
# DIVERSE sub-model: EEGNet + per-subject raw-signal EA + Mixup
# ---------------------------------------------------------------------------

DIVERSE_BAND = (8.0, 30.0)
DIVERSE_WINDOW = 500
DIVERSE_STRIDE = 100
DIVERSE_SEEDS = [42, 1, 7]


def _per_window_zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int,
                 shuffle: bool, num_workers: int) -> DataLoader:
    tx = torch.from_numpy(x).float().unsqueeze(1)   # (N, 1, C, T)
    ty = torch.from_numpy(y).long()
    return DataLoader(
        TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=False,
    )


def _load_diverse_subject_windows(
    data_dir: Path, sid: int, cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Filter + crop + per-subject EA whitening + sliding + zscore."""
    f = data_dir / f"subject{sid:02d}.npz"
    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")
    x, y = load_subject(f)
    x = filter_and_crop(x, cfg)
    whitening = compute_ea_whitening(x)
    x = apply_ea_whitening(x, whitening)
    x_w, n_w = sliding_windows(x, DIVERSE_WINDOW, DIVERSE_STRIDE)
    x_w = _per_window_zscore(x_w)
    y_w = np.repeat(y, n_w)
    return x_w, y_w, n_w


def train_diverse_seed(data_dir: Path, seed: int, args, device: torch.device,
                       val_subject: int = 10) -> tuple[dict, list[str]]:
    """Train one EEGNet seed with EA + Mixup + cosine-LR; hold subject 10 for val."""
    set_seed(seed)

    pre_cfg = PreprocessConfig(band_low=DIVERSE_BAND[0], band_high=DIVERSE_BAND[1])

    train_x_parts: list[np.ndarray] = []
    train_y_parts: list[np.ndarray] = []
    val_x_w: np.ndarray | None = None
    val_y_trial: np.ndarray | None = None
    val_n_w = 0
    for sid in range(1, 11):
        x_w, y_w, n_w = _load_diverse_subject_windows(data_dir, sid, pre_cfg)
        if sid == val_subject:
            val_x_w = x_w
            val_y_trial = y_w.reshape(-1, n_w)[:, 0]
            val_n_w = n_w
        else:
            train_x_parts.append(x_w)
            train_y_parts.append(y_w)
    x_tr = np.concatenate(train_x_parts, axis=0)
    y_tr = np.concatenate(train_y_parts, axis=0)
    assert val_x_w is not None and val_y_trial is not None

    n_channels, n_samples = x_tr.shape[1], x_tr.shape[2]
    model_cfg = ModelConfig(
        n_channels=n_channels, n_samples=n_samples,
        n_classes=4, dropout=args.diverse_dropout,
    )

    train_loader = _make_loader(
        x_tr, y_tr, args.batch_size, True, args.num_workers,
    )

    model = EEGNet(model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    best_trial_acc = -1.0
    best_state = None
    log_lines: list[str] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mixed_x, y_a, y_b, lam = _mixup_batch(xb, yb, args.diverse_mixup_alpha)
            logits = model(mixed_x)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            tx = torch.from_numpy(val_x_w).float().unsqueeze(1)
            probs_batches = []
            for i in range(0, len(tx), 128):
                logits = model(tx[i:i + 128].to(device))
                probs_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
            proba_w = np.concatenate(probs_batches, axis=0)
            proba = proba_w.reshape(len(val_y_trial), val_n_w, -1).mean(axis=1)
            trial_acc = float((proba.argmax(axis=1) == val_y_trial).mean())

        if trial_acc > best_trial_acc:
            best_trial_acc = trial_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        line = (f"  [seed={seed}] epoch {epoch:3d}/{args.epochs} | "
                f"val(trial-TTA)={trial_acc:.4f} | best={best_trial_acc:.4f}")
        log_lines.append(line)
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(line)

    if best_state is None:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}

    sub_ckpt = {
        "model_type": "eegnet_ea_mixup",
        "state_dict": best_state,
        "model_kwargs": model_cfg.to_dict(),
        "preprocess_cfg": pre_cfg.to_dict(),
        "sliding_cfg": {"window_size": DIVERSE_WINDOW, "stride": DIVERSE_STRIDE},
        "seed": seed,
        "best_val_acc": best_trial_acc,
    }
    return sub_ckpt, log_lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # Determinism nudge (cuDNN). On CPU it's already deterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_total = 1 + len(DIVERSE_SEEDS)
    print(f"[train] device={device}, base_seed={args.seed}")
    print(f"[train] strong  : Riemann + EA + dense sliding, band {STRONG_BAND}, "
          f"ws={STRONG_WINDOW}, stride={STRONG_STRIDE}")
    print(f"[train] diverse : EEGNet + EA + Mixup, band {DIVERSE_BAND}, "
          f"ws={DIVERSE_WINDOW}, stride={DIVERSE_STRIDE}, "
          f"seeds={DIVERSE_SEEDS}, mixup_alpha={args.diverse_mixup_alpha}, "
          f"dropout={args.diverse_dropout}, epochs={args.epochs}, "
          f"bs={args.batch_size}, lr={args.learning_rate}, wd={args.weight_decay}")
    print(f"[train] default_gate_k = {args.default_gate_k}")
    print()

    log_all: list[str] = []

    # --- Strong ---
    t0 = time.time()
    print("=" * 60)
    print(f"[1/{n_total}] Training STRONG (Riemann broad_dense) ...")
    print("=" * 60)
    strong = train_strong(args.data)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s")
    log_all.append(f"strong: trained in {dt:.1f}s")
    print()

    # --- Diverse seeds ---
    diverse: list[dict] = []
    for i, seed in enumerate(DIVERSE_SEEDS, start=1):
        t0 = time.time()
        print("=" * 60)
        print(f"[{i + 1}/{n_total}] Training DIVERSE seed={seed} (EEGNet+EA+Mixup) ...")
        print("=" * 60)
        sub_ckpt, seed_log = train_diverse_seed(args.data, seed, args, device)
        diverse.append(sub_ckpt)
        dt = time.time() - t0
        print(f"  best val trial-TTA acc = {sub_ckpt['best_val_acc']:.4f}  "
              f"({dt:.1f}s)")
        log_all.append(
            f"diverse seed={seed}: best val acc = {sub_ckpt['best_val_acc']:.4f} "
            f"in {dt:.1f}s"
        )
        log_all.extend(seed_log)
        print()

    # --- Bundle ---
    bundle = {
        "model_type": "gated_ensemble",
        "default_gate_k": args.default_gate_k,
        "strong": strong,
        "diverse": diverse,
    }
    joblib.dump(bundle, args.checkpoint)

    (args.output_dir / "training_log.txt").write_text(
        "\n".join(log_all), encoding="utf-8",
    )

    print(f"[done] wrote bundled checkpoint to {args.checkpoint}")
    print(f"[done] inference.py will produce gated_k3 (Public LB 0.8750) by default,")
    print(f"       use 'inference.py --gate-k 7' to reproduce gated_k7 (Public LB 0.8750).")


if __name__ == "__main__":
    main()
