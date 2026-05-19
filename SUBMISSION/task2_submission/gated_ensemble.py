"""Confidence-gated ensemble for Kaggle public-LB exploration.

Idea: a strong model (Riemann broad_dense, public LB 0.8125) is correct on
most trials; its mistakes tend to concentrate on its *least confident*
predictions. So keep the strong model's prediction everywhere EXCEPT on its
K least-confident trials, where we defer to a diverse model (the EEGNet
ensemble). This targets exactly the trials the strong model is most likely
wrong on, without risking the ones it is confident (and usually right) about.

This is a development helper for Kaggle submission generation -- the
spec-required interface remains the single-checkpoint inference.py.

Run from part1/task2:
    python gated_ensemble.py --data data \\
        --strong-checkpoint outputs/task2_train_broad_dense/task2_model.pt \\
        --diverse-checkpoints outputs/task2_dl_tuned/task2_model.pt \\
                              outputs/task2_dl_tuned_s1/task2_model.pt \\
                              outputs/task2_dl_tuned_s7/task2_model.pt \\
        --gate-k 4 \\
        --output submission_gated_k4.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Reuse the proven helpers from ensemble_inference.py so the per-checkpoint
# preprocessing / inference path is identical to the rest of the pipeline.
from SUBMISSION.task2_submission.ensemble_inference import (
    load_test,
    proba_for_checkpoint,
    write_submission,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Confidence-gated ensemble: strong model + diverse fallback."
    )
    p.add_argument("--data", type=Path, required=True,
                   help="Path to task2 data dir (contains test.npz) or test.npz itself.")
    p.add_argument("--strong-checkpoint", type=Path, required=True,
                   help="The high-accuracy checkpoint (e.g. Riemann broad_dense).")
    p.add_argument("--diverse-checkpoints", type=Path, nargs="+", required=True,
                   help="One or more diverse checkpoints (e.g. EEGNet seeds). "
                        "Their probabilities are averaged into one fallback model.")
    p.add_argument("--gate-k", type=int, default=4,
                   help="Number of least-confident strong-model trials to defer "
                        "to the diverse fallback. Default 4.")
    p.add_argument("--output", type=Path, default=Path("submission_gated.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    x_raw, ids = load_test(args.data)
    n_trials = len(ids)
    print(f"[gated] test trials: {n_trials}, gate-k: {args.gate_k}")

    # ---- Strong model probabilities ----
    print(f"[gated] strong checkpoint: {args.strong_checkpoint}")
    proba_strong = proba_for_checkpoint(args.strong_checkpoint, x_raw)
    strong_pred = proba_strong.argmax(axis=1)
    strong_conf = proba_strong.max(axis=1)

    # ---- Diverse fallback: average probabilities across all diverse checkpoints ----
    diverse_probas = []
    for ckpt in args.diverse_checkpoints:
        print(f"[gated] diverse checkpoint: {ckpt}")
        diverse_probas.append(proba_for_checkpoint(ckpt, x_raw))
    proba_diverse = np.mean(diverse_probas, axis=0)
    diverse_pred = proba_diverse.argmax(axis=1)

    # ---- Gate: replace the K least-confident strong predictions ----
    gate_k = max(0, min(args.gate_k, n_trials))
    low_conf_idx = np.argsort(strong_conf)[:gate_k]  # K lowest-confidence trials

    final_pred = strong_pred.copy()
    n_changed = 0
    for idx in low_conf_idx:
        if diverse_pred[idx] != strong_pred[idx]:
            n_changed += 1
        final_pred[idx] = diverse_pred[idx]

    print(f"[gated] strong-model label dist : "
          f"{np.bincount(strong_pred, minlength=4).tolist()}")
    print(f"[gated] gated {gate_k} least-confident trials "
          f"(confidences {np.sort(strong_conf)[:gate_k].round(3).tolist()})")
    print(f"[gated] of those, {n_changed} predictions actually changed")
    print(f"[gated] final label dist       : "
          f"{np.bincount(final_pred, minlength=4).tolist()}")

    rows = [(int(i), int(l)) for i, l in zip(ids, final_pred.astype(np.int64))]
    write_submission(rows, args.output)
    print(f"[gated] wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
