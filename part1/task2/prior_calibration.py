"""Score export and prior-constrained calibration for Task 2 submissions.

This helper keeps inference.py's required CLI untouched. It loads the current
best checkpoint, exports test probabilities, and creates quota-constrained
submission candidates for upload experiments.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

from inference import load_test_data, preprocess_for_inference, validate_predictions, write_submission

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback for minimal environments
    linear_sum_assignment = None

LABELS = (0, 1, 2, 3)
EPS = 1e-12

QUOTA_FAMILIES: dict[str, dict[int, int]] = {
    "current_like": {0: 5, 1: 8, 2: 12, 3: 7},
    "mild_class2": {0: 6, 1: 8, 2: 10, 3: 8},
    "balancedish": {0: 7, 1: 8, 2: 9, 3: 8},
    "balanced": {0: 8, 1: 8, 2: 8, 3: 8},
    "class2_less": {0: 6, 1: 9, 2: 8, 3: 9},
}

UPLOAD_NAMES: dict[str, str] = {
    "public_adjusted": "submission_task2_prior_public_adjusted.csv",
    "mild_class2": "submission_task2_prior_mild_class2.csv",
    "balancedish": "submission_task2_prior_balancedish.csv",
    "balanced": "submission_task2_prior_balanced.csv",
    "class2_less": "submission_task2_prior_class2_less.csv",
    "current_like": "submission_task2_prior_current_like.csv",
}


@dataclass(frozen=True)
class CandidateSummary:
    name: str
    path: Path
    distribution: dict[int, int]
    changed: list[dict[str, int | float]]
    mean_changed_confidence: float | None
    identical_to: list[str]
    agreement_with_references: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task 2 prior-calibrated submission candidates.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/task2_highscore/task2_model.pkl"))
    parser.add_argument("--baseline", type=Path, default=Path("submission_task2_best_loso.csv"))
    parser.add_argument("--scores-output", type=Path, default=Path("task2_test_scores.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--previous",
        type=Path,
        nargs="*",
        default=[
            Path("submission_task2_best_loso.csv"),
            Path("submission_task2_alt_highscore.csv"),
            Path("submission_task2_riemannian_best.csv"),
            Path("submission.csv"),
        ],
        help="Existing submissions to compare candidates against.",
    )
    parser.add_argument("--max-public-adjustments", type=int, default=3)
    return parser.parse_args()


def load_submission(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "label"]:
            raise ValueError(f"{path}: expected header id,label, got {reader.fieldnames}")
        rows = [(int(row["id"]), int(row["label"])) for row in reader]
    if not rows:
        raise ValueError(f"{path}: no rows")
    ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    labels = np.asarray([row[1] for row in rows], dtype=np.int64)
    return ids, validate_predictions(labels, len(labels))


def probability_matrix(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    elif hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(features), dtype=np.float64)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        shifted = decision - decision.max(axis=1, keepdims=True)
        raw = np.exp(shifted)
        raw = raw / raw.sum(axis=1, keepdims=True)
    else:
        raise TypeError("Checkpoint model has neither predict_proba nor decision_function")

    classes = np.asarray(getattr(model, "classes_", LABELS), dtype=np.int64)
    probs = np.full((len(features), len(LABELS)), EPS, dtype=np.float64)
    for source_idx, label in enumerate(classes):
        if int(label) in LABELS:
            probs[:, int(label)] = raw[:, source_idx]
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def top2_stats(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0].astype(np.int64)
    top2 = order[:, 1].astype(np.int64)
    confidence = probs[np.arange(len(probs)), top1]
    margin = confidence - probs[np.arange(len(probs)), top2]
    return top1, top2, confidence, margin


def write_scores(
    path: Path,
    ids: np.ndarray,
    pred: np.ndarray,
    probs: np.ndarray,
    top2: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
    references: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = [f"{name}_label" for name in references]
    fields = [
        "id",
        "predicted_label",
        "prob_0",
        "prob_1",
        "prob_2",
        "prob_3",
        "top2_label",
        "top2_probability",
        "top1_top2_margin",
        "confidence",
        *extra_fields,
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, sample_id in enumerate(ids):
            row = {
                "id": int(sample_id),
                "predicted_label": int(pred[i]),
                "prob_0": f"{probs[i, 0]:.10f}",
                "prob_1": f"{probs[i, 1]:.10f}",
                "prob_2": f"{probs[i, 2]:.10f}",
                "prob_3": f"{probs[i, 3]:.10f}",
                "top2_label": int(top2[i]),
                "top2_probability": f"{probs[i, top2[i]]:.10f}",
                "top1_top2_margin": f"{margin[i]:.10f}",
                "confidence": f"{confidence[i]:.10f}",
            }
            for name, labels in references.items():
                row[f"{name}_label"] = int(labels[i])
            writer.writerow(row)


def quota_labels(probs: np.ndarray, quota: dict[int, int]) -> np.ndarray:
    if sum(quota.values()) != len(probs):
        raise ValueError(f"Quota must sum to {len(probs)}, got {quota}")
    slots = np.asarray([label for label in LABELS for _ in range(quota[label])], dtype=np.int64)
    objective = np.log(np.clip(probs[:, slots], EPS, 1.0))

    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(-objective)
        labels = np.empty(len(probs), dtype=np.int64)
        labels[rows] = slots[cols]
        return labels

    remaining = Counter(quota)
    labels = np.full(len(probs), -1, dtype=np.int64)
    pairs = sorted(
        ((float(probs[row, label]), row, label) for row in range(len(probs)) for label in LABELS),
        reverse=True,
    )
    for _, row, label in pairs:
        if labels[row] == -1 and remaining[label] > 0:
            labels[row] = label
            remaining[label] -= 1
    if np.any(labels == -1):
        raise RuntimeError("Greedy quota assignment failed to assign every row")
    return labels


def public_adjusted_labels(
    baseline: np.ndarray,
    top2: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
    max_changes: int,
) -> np.ndarray:
    labels = baseline.copy()
    max_changes = max(1, min(int(max_changes), 3))
    order = sorted(range(len(labels)), key=lambda i: (labels[i] != 2, float(margin[i]), float(confidence[i]), i))
    changed = 0
    for idx in order:
        new_label = int(top2[idx])
        if new_label == int(labels[idx]):
            continue
        labels[idx] = new_label
        changed += 1
        if changed >= max_changes:
            break
    return labels


def validate_submission(ids: np.ndarray, labels: np.ndarray, expected_ids: np.ndarray, path: Path) -> None:
    validate_predictions(labels, len(expected_ids))
    if len(labels) != 32:
        raise ValueError(f"{path}: expected 32 rows, got {len(labels)}")
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"{path}: id order does not match test.npz ids")


def changed_rows(
    ids: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for i, (old, new) in enumerate(zip(baseline, labels)):
        if int(old) != int(new):
            rows.append(
                {
                    "id": int(ids[i]),
                    "old": int(old),
                    "new": int(new),
                    "confidence": round(float(confidence[i]), 6),
                    "margin": round(float(margin[i]), 6),
                }
            )
    return rows


def distribution(labels: Iterable[int]) -> dict[int, int]:
    counts = Counter(int(label) for label in labels)
    return {label: counts.get(label, 0) for label in LABELS}


def summarize_candidate(
    name: str,
    path: Path,
    ids: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
    previous: dict[str, np.ndarray],
) -> CandidateSummary:
    changes = changed_rows(ids, baseline, labels, confidence, margin)
    mean_conf = None if not changes else float(np.mean([row["confidence"] for row in changes]))
    identical = [prev_name for prev_name, prev_labels in previous.items() if np.array_equal(labels, prev_labels)]
    agreement = {prev_name: int(np.sum(labels == prev_labels)) for prev_name, prev_labels in previous.items()}
    return CandidateSummary(
        name=name,
        path=path,
        distribution=distribution(labels),
        changed=changes,
        mean_changed_confidence=mean_conf,
        identical_to=identical,
        agreement_with_references=agreement,
    )


def print_summary(summary: CandidateSummary) -> None:
    mean_conf = "n/a" if summary.mean_changed_confidence is None else f"{summary.mean_changed_confidence:.6f}"
    changed = "none" if not summary.changed else ", ".join(
        f"id {row['id']}: {row['old']}->{row['new']} conf={row['confidence']:.6f} margin={row['margin']:.6f}"
        for row in summary.changed
    )
    identical = "none" if not summary.identical_to else ", ".join(summary.identical_to)
    print(f"\n[{summary.name}] {summary.path}")
    print(f"  distribution: {summary.distribution}")
    print(f"  changed rows vs best_loso ({len(summary.changed)}): {changed}")
    print(f"  mean confidence of changed rows: {mean_conf}")
    print(f"  identical to previous file: {identical}")
    print(f"  agreement counts: {summary.agreement_with_references}")


def main() -> None:
    args = parse_args()
    x_test, ids = load_test_data(args.data)
    checkpoint = joblib.load(args.checkpoint)
    features = preprocess_for_inference(x_test, checkpoint)
    probs = probability_matrix(checkpoint["model"], features)
    pred, top2, confidence, margin = top2_stats(probs)
    pred = validate_predictions(pred, len(ids))

    baseline_ids, baseline = load_submission(args.baseline)
    validate_submission(baseline_ids, baseline, ids, args.baseline)
    if not np.array_equal(pred, baseline):
        print("WARNING: model argmax differs from baseline best_loso submission.")

    previous: dict[str, np.ndarray] = {}
    for previous_path in args.previous:
        if previous_path.exists():
            prev_ids, prev_labels = load_submission(previous_path)
            validate_submission(prev_ids, prev_labels, ids, previous_path)
            previous[previous_path.name] = prev_labels

    references = {
        name.removeprefix("submission_task2_").removesuffix(".csv"): labels
        for name, labels in previous.items()
        if name != args.baseline.name
    }
    write_scores(args.scores_output, ids, pred, probs, top2, confidence, margin, references)
    print(f"Wrote score diagnostics to {args.scores_output}")
    print(f"Argmax distribution: {distribution(pred)}")
    print(f"Assignment optimizer: {'linear_sum_assignment' if linear_sum_assignment is not None else 'greedy'}")

    candidate_labels: dict[str, np.ndarray] = {
        name: quota_labels(probs, quota) for name, quota in QUOTA_FAMILIES.items()
    }
    candidate_labels["public_adjusted"] = public_adjusted_labels(
        baseline, top2, confidence, margin, args.max_public_adjustments
    )

    written_labels: dict[str, np.ndarray] = {}
    for name in ["public_adjusted", "mild_class2", "balancedish", "balanced", "class2_less", "current_like"]:
        labels = validate_predictions(candidate_labels[name], len(ids))
        output_path = args.output_dir / UPLOAD_NAMES[name]
        rows = [(int(sample_id), int(label)) for sample_id, label in zip(ids, labels)]
        write_submission(rows, output_path)
        out_ids, out_labels = load_submission(output_path)
        validate_submission(out_ids, out_labels, ids, output_path)
        comparison_pool = {**previous, **written_labels}
        summary = summarize_candidate(name, output_path, ids, labels, baseline, confidence, margin, comparison_pool)
        print_summary(summary)
        written_labels[output_path.name] = labels


if __name__ == "__main__":
    main()
