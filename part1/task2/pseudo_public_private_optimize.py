"""Pseudo-public/private validation for Task 2 cross-subject EEG models.

This script simulates a Kaggle public/private split on each training subject.
For every held-out subject, models are trained on the other subjects, the
held-out 32 trials are split into pseudo-public and pseudo-private halves, and
adaptation/calibration methods are evaluated only on the pseudo-private half.

Oracle pseudo-public labels are used only for upper-bound diagnostics inside
this simulation. Real-test candidates are generated only from legal unlabeled
or pseudo-label methods, and only when the internal simulation clears a
conservative threshold.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from cross_subject_optimize import (
    EPS,
    LABELS,
    Config,
    FittedPipeline,
    apply_feature_normalizer,
    apply_subject_transform,
    changed_rows,
    config_to_dict,
    distribution,
    filter_configs,
    fit_pipeline,
    fit_subject_feature_normalizer,
    fit_subject_transform,
    guardrail_notes,
    load_existing_submission,
    load_subjects,
    load_test_data,
    make_configs,
    prediction_scores,
    prepare_subjects,
    quota_assign,
    validate_submission,
    write_submission,
    transform_features,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "task2_pseudo_public_private"
DEFAULT_CONFIGS = "tangent_broad_lda,cov_align_subject_lda_class2_up"
DEFAULT_REFERENCE_FILES = (
    "submission_task2_prior_public_adjusted.csv",
    "submission_task2_prior_class2_less.csv",
)
GRID_BIASES = (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2)
BIAS_GRID = np.asarray(
    [
        np.asarray(raw_bias, dtype=np.float64) - np.mean(raw_bias)
        for raw_bias in itertools.product(GRID_BIASES, repeat=len(LABELS))
    ],
    dtype=np.float64,
)
BIAS_GRID_MAGNITUDE = np.sum(np.abs(BIAS_GRID), axis=1)


@dataclass(frozen=True)
class SplitSpec:
    subject: str
    subject_idx: int
    split_idx: int
    public_idx: np.ndarray
    private_idx: np.ndarray


@dataclass(frozen=True)
class MethodSpec:
    name: str
    category: str
    legal_for_test: bool
    uses_public_labels: bool


@dataclass(frozen=True)
class ModelBundle:
    name: str
    config_names: tuple[str, ...]


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("baseline_private_only", "baseline", True, False),
    MethodSpec("unlabeled_batch32_norm", "legal_unlabeled", True, False),
    MethodSpec("unlabeled_em_prior32", "legal_unlabeled", True, False),
    MethodSpec("unlabeled_conservative_quota32", "legal_pseudolabel", True, False),
    MethodSpec("unlabeled_uniform_quota32", "legal_prior", True, False),
    MethodSpec("oracle_public_prior_ratio", "oracle_public", False, True),
    MethodSpec("oracle_public_confusion", "oracle_public", False, True),
    MethodSpec("oracle_public_bias_grid", "oracle_public", False, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 2 pseudo-public/private validation.")
    parser.add_argument("--train-data", type=Path, default=SCRIPT_DIR / "data" / "train")
    parser.add_argument("--test-data", type=Path, default=SCRIPT_DIR / "data" / "test.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=SCRIPT_DIR / "task2_pseudo_public_private_report.md")
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", type=int, default=12)
    parser.add_argument("--public-size", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--config-names", default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--candidate-min-delta",
        type=float,
        default=0.01,
        help="Minimum mean macro-F1 delta vs baseline required to emit one legal real-test candidate.",
    )
    parser.add_argument(
        "--candidate-min-win-rate",
        type=float,
        default=0.55,
        help="Minimum split-level win rate vs baseline required to emit one legal real-test candidate.",
    )
    return parser.parse_args()


def balanced_quota(total: int) -> dict[int, int]:
    raw = {label: total / len(LABELS) for label in LABELS}
    quota = {label: int(np.floor(raw[label])) for label in LABELS}
    remaining = total - sum(quota.values())
    for label in LABELS[:remaining]:
        quota[label] += 1
    return quota


def normalized(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    return probs / np.maximum(probs.sum(axis=1, keepdims=True), EPS)


def margins(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def predict_batch(
    pipeline: FittedPipeline,
    x_batch: np.ndarray,
    target_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Predict a target subset after fitting subject transforms on x_batch."""
    transform = fit_subject_transform(x_batch, pipeline.config.raw_norm, pipeline.config.align)
    x_norm = apply_subject_transform(x_batch, transform)
    feats = transform_features(x_norm, pipeline.state)
    normalizer = fit_subject_feature_normalizer(feats, pipeline.config.feature_norm)
    feats = apply_feature_normalizer(feats, normalizer)
    feats = pipeline.scaler.transform(feats)
    probs = prediction_scores(pipeline.model, feats)
    if target_positions is not None:
        probs = probs[target_positions]
    return normalized(probs)


def em_prior_adjust(probs: np.ndarray, iterations: int = 50) -> np.ndarray:
    """Saerens-style label-shift prior update using only unlabeled probabilities."""
    probs = normalized(probs)
    n_classes = probs.shape[1]
    train_prior = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    test_prior = probs.mean(axis=0)
    test_prior = np.maximum(test_prior, EPS)
    test_prior = test_prior / test_prior.sum()
    for _ in range(iterations):
        weights = test_prior / train_prior
        adjusted = normalized(probs * weights)
        new_prior = adjusted.mean(axis=0)
        new_prior = np.maximum(new_prior, EPS)
        new_prior = new_prior / new_prior.sum()
        if np.max(np.abs(new_prior - test_prior)) < 1e-5:
            break
        test_prior = new_prior
    return normalized(probs * (test_prior / train_prior))


def conservative_quota_labels(probs: np.ndarray, quota: dict[int, int], max_changed_margin: float = 0.20) -> np.ndarray:
    """Quota decoding that keeps high-margin argmax labels fixed."""
    base = np.argmax(probs, axis=1).astype(np.int64)
    quota_labels = quota_assign(probs, quota)
    margin = margins(probs)
    out = base.copy()
    low_margin_changed = (quota_labels != base) & (margin <= max_changed_margin)
    out[low_margin_changed] = quota_labels[low_margin_changed]
    return out


def oracle_prior_ratio_adjust(probs: np.ndarray, public_idx: np.ndarray, y_public: np.ndarray) -> np.ndarray:
    public_counts = np.asarray([np.sum(y_public == label) for label in LABELS], dtype=np.float64) + 0.75
    public_prior = public_counts / public_counts.sum()
    soft_counts = probs[public_idx].sum(axis=0) + 0.75
    soft_prior = soft_counts / soft_counts.sum()
    return normalized(probs * (public_prior / np.maximum(soft_prior, EPS)))


def oracle_confusion_adjust(probs: np.ndarray, public_idx: np.ndarray, y_public: np.ndarray) -> np.ndarray:
    pred_public = np.argmax(probs[public_idx], axis=1).astype(np.int64)
    # Rows are predicted labels, columns are corrected true labels.
    correction = np.full((len(LABELS), len(LABELS)), 0.5, dtype=np.float64)
    for y_pred, y_true in zip(pred_public, y_public):
        correction[int(y_pred), int(y_true)] += 1.0
    correction = correction / correction.sum(axis=1, keepdims=True)
    return normalized(probs @ correction)


def oracle_bias_grid_adjust(
    probs: np.ndarray,
    public_idx: np.ndarray,
    y_public: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    log_probs = np.log(np.clip(probs, EPS, 1.0))
    public_log_probs = log_probs[public_idx]
    pred_grid = np.argmax(public_log_probs[None, :, :] + BIAS_GRID[:, None, :], axis=2)
    public_f1 = macro_f1_grid(y_public, pred_grid)
    public_acc = np.mean(pred_grid == y_public[None, :], axis=1)
    base_public = np.argmax(public_log_probs, axis=1)
    changes = np.sum(pred_grid != base_public[None, :], axis=1)
    order = np.lexsort((BIAS_GRID_MAGNITUDE, changes, -public_acc, -public_f1))
    best_idx = int(order[0])
    best_bias = BIAS_GRID[best_idx]
    return normalized(np.exp(log_probs + best_bias)), best_bias, float(public_f1[best_idx])


def macro_f1_grid(y_true: np.ndarray, pred_grid: np.ndarray) -> np.ndarray:
    scores = np.zeros(pred_grid.shape[0], dtype=np.float64)
    for label in LABELS:
        true_mask = y_true[None, :] == label
        pred_mask = pred_grid == label
        tp = np.sum(pred_mask & true_mask, axis=1)
        fp = np.sum(pred_mask & ~true_mask, axis=1)
        fn = np.sum(~pred_mask & true_mask, axis=1)
        denom = 2 * tp + fp + fn
        scores += np.where(denom > 0, (2 * tp) / denom, 0.0)
    return scores / len(LABELS)


def score_labels(y_true: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    class_f1 = f1_score(y_true, labels, average=None, labels=LABELS, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, labels, average="macro", labels=LABELS, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, labels)),
        "class_f1": {label: float(class_f1[idx]) for idx, label in enumerate(LABELS)},
        "confusion": confusion_matrix(y_true, labels, labels=LABELS),
    }


def make_splits(subjects: Sequence[Any], public_size: int, n_splits: int, seed: int) -> list[SplitSpec]:
    rng = np.random.default_rng(seed)
    specs: list[SplitSpec] = []
    for subject_idx, subject in enumerate(subjects):
        n_trials = len(subject.y)
        if public_size <= 0 or public_size >= n_trials:
            raise ValueError(f"public_size must be in 1..{n_trials - 1}, got {public_size}")
        seen: set[tuple[int, ...]] = set()
        attempts = 0
        while len([s for s in specs if s.subject_idx == subject_idx]) < n_splits:
            attempts += 1
            if attempts > n_splits * 100:
                raise RuntimeError(f"Could not create enough unique splits for {subject.name}")
            public_idx = np.sort(rng.choice(n_trials, size=public_size, replace=False))
            key = tuple(int(i) for i in public_idx)
            if key in seen:
                continue
            seen.add(key)
            private_idx = np.asarray([idx for idx in range(n_trials) if idx not in set(key)], dtype=np.int64)
            split_idx = len([s for s in specs if s.subject_idx == subject_idx])
            specs.append(SplitSpec(subject.name, subject_idx, split_idx, public_idx, private_idx))
    return specs


def fit_loso_pipelines(
    subjects: Sequence[Any],
    configs: Sequence[Config],
    sfreq: float,
    filter_order: int,
    seed: int,
    max_iter: int,
) -> dict[tuple[int, str], FittedPipeline]:
    pipelines: dict[tuple[int, str], FittedPipeline] = {}
    for config in configs:
        x_all, y_all, names = prepare_subjects(subjects, config)
        for hold_idx, subject_name in enumerate(names):
            train_x = [x for idx, x in enumerate(x_all) if idx != hold_idx]
            train_y = [y for idx, y in enumerate(y_all) if idx != hold_idx]
            pipelines[(hold_idx, config.name)] = fit_pipeline(
                train_x,
                train_y,
                config,
                sfreq,
                filter_order,
                seed,
                max_iter,
            )
            print(f"fitted {config.name} with held-out {subject_name}")
    return pipelines


def build_model_bundles(configs: Sequence[Config]) -> list[ModelBundle]:
    config_names = {config.name for config in configs}
    bundles = [ModelBundle(name=config.name, config_names=(config.name,)) for config in configs]
    if {"tangent_broad_lda", "cov_align_subject_lda_class2_up"}.issubset(config_names):
        bundles.append(
            ModelBundle(
                name="ensemble_tangent_cov2",
                config_names=("tangent_broad_lda", "cov_align_subject_lda_class2_up"),
            )
        )
    return bundles


def apply_method_all(
    method: str,
    probs_all32: np.ndarray,
    public_idx: np.ndarray,
    y_public: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    quota = balanced_quota(len(probs_all32))
    extra: dict[str, Any] = {}
    if method == "unlabeled_batch32_norm":
        labels_all = np.argmax(probs_all32, axis=1).astype(np.int64)
    elif method == "unlabeled_em_prior32":
        adjusted = em_prior_adjust(probs_all32)
        labels_all = np.argmax(adjusted, axis=1).astype(np.int64)
        extra["estimated_prior"] = adjusted.mean(axis=0).round(6).tolist()
    elif method == "unlabeled_uniform_quota32":
        labels_all = quota_assign(probs_all32, quota)
    elif method == "unlabeled_conservative_quota32":
        labels_all = conservative_quota_labels(probs_all32, quota)
    elif method == "oracle_public_prior_ratio":
        adjusted = oracle_prior_ratio_adjust(probs_all32, public_idx, y_public)
        labels_all = np.argmax(adjusted, axis=1).astype(np.int64)
    elif method == "oracle_public_confusion":
        adjusted = oracle_confusion_adjust(probs_all32, public_idx, y_public)
        labels_all = np.argmax(adjusted, axis=1).astype(np.int64)
    elif method == "oracle_public_bias_grid":
        adjusted, bias, public_fit = oracle_bias_grid_adjust(probs_all32, public_idx, y_public)
        labels_all = np.argmax(adjusted, axis=1).astype(np.int64)
        extra["bias"] = bias.round(6).tolist()
        extra["public_fit_macro_f1"] = public_fit
    else:
        raise ValueError(f"Unknown method: {method}")
    return labels_all.astype(np.int64), extra


def run_simulation(
    subjects: Sequence[Any],
    configs: Sequence[Config],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[ModelBundle]]:
    splits = make_splits(subjects, args.public_size, args.splits, args.seed)
    pipelines = fit_loso_pipelines(subjects, configs, args.sfreq, args.filter_order, args.seed, args.max_iter)
    bundles = build_model_bundles(configs)
    method_map = {method.name: method for method in METHODS}
    records: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    all32_cache: dict[tuple[int, str], np.ndarray] = {}
    private_cache: dict[tuple[int, int, str], np.ndarray] = {}

    for split in splits:
        subject = subjects[split.subject_idx]
        y_private = subject.y[split.private_idx]
        y_public = subject.y[split.public_idx]
        for config in configs:
            pipeline = pipelines[(split.subject_idx, config.name)]
            all_key = (split.subject_idx, config.name)
            if all_key not in all32_cache:
                all32_cache[all_key] = predict_batch(pipeline, subject.x)
            private_key = (split.subject_idx, split.split_idx, config.name)
            if private_key not in private_cache:
                private_cache[private_key] = predict_batch(pipeline, subject.x[split.private_idx])

        for bundle in bundles:
            probs_all32 = np.mean([all32_cache[(split.subject_idx, name)] for name in bundle.config_names], axis=0)
            probs_private_only = np.mean(
                [private_cache[(split.subject_idx, split.split_idx, name)] for name in bundle.config_names],
                axis=0,
            )
            baseline_labels = np.argmax(probs_private_only, axis=1).astype(np.int64)
            baseline_score = score_labels(y_private, baseline_labels)
            public_base_labels = np.argmax(probs_all32[split.public_idx], axis=1).astype(np.int64)
            public_base_f1 = f1_score(
                y_public,
                public_base_labels,
                average="macro",
                labels=LABELS,
                zero_division=0,
            )

            for method_spec in METHODS:
                if method_spec.name == "baseline_private_only":
                    labels = baseline_labels
                    public_labels_for_method = public_base_labels
                    extra: dict[str, Any] = {}
                else:
                    labels_all, extra = apply_method_all(
                        method_spec.name,
                        probs_all32,
                        split.public_idx,
                        y_public,
                    )
                    labels = labels_all[split.private_idx]
                    public_labels_for_method = labels_all[split.public_idx]
                score = score_labels(y_private, labels)
                public_f1 = f1_score(
                    y_public,
                    public_labels_for_method,
                    average="macro",
                    labels=LABELS,
                    zero_division=0,
                )
                delta = score["macro_f1"] - baseline_score["macro_f1"]
                changed = int(np.sum(labels != baseline_labels))
                record = {
                    "model": bundle.name,
                    "method": method_spec.name,
                    "category": method_spec.category,
                    "legal_for_test": method_spec.legal_for_test,
                    "uses_public_labels": method_spec.uses_public_labels,
                    "subject": split.subject,
                    "subject_idx": split.subject_idx,
                    "split_idx": split.split_idx,
                    "private_macro_f1": score["macro_f1"],
                    "private_accuracy": score["accuracy"],
                    "private_class_f1": score["class_f1"],
                    "public_macro_f1": float(public_f1),
                    "public_base_macro_f1": float(public_base_f1),
                    "delta_macro_f1": float(delta),
                    "changed_vs_baseline": changed,
                    "private_true_distribution": distribution(y_private),
                    "private_pred_distribution": distribution(labels),
                    "confusion": score["confusion"].tolist(),
                    "extra": extra,
                }
                records.append(record)
                split_rows.append(flatten_record(record))
    return records, split_rows, bundles


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: record[key]
        for key in [
            "model",
            "method",
            "category",
            "legal_for_test",
            "uses_public_labels",
            "subject",
            "subject_idx",
            "split_idx",
            "private_macro_f1",
            "private_accuracy",
            "public_macro_f1",
            "public_base_macro_f1",
            "delta_macro_f1",
            "changed_vs_baseline",
        ]
    }
    row["private_class_f1"] = json.dumps(record["private_class_f1"], sort_keys=True)
    row["private_true_distribution"] = json.dumps(record["private_true_distribution"], sort_keys=True)
    row["private_pred_distribution"] = json.dumps(record["private_pred_distribution"], sort_keys=True)
    row["confusion"] = json.dumps(record["confusion"])
    row["extra"] = json.dumps(record["extra"], sort_keys=True)
    return row


def aggregate_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["method"])].append(record)

    summaries: list[dict[str, Any]] = []
    for (model, method), rows in grouped.items():
        method_spec = next(spec for spec in METHODS if spec.name == method)
        macro = np.asarray([row["private_macro_f1"] for row in rows], dtype=np.float64)
        acc = np.asarray([row["private_accuracy"] for row in rows], dtype=np.float64)
        delta = np.asarray([row["delta_macro_f1"] for row in rows], dtype=np.float64)
        changed = np.asarray([row["changed_vs_baseline"] for row in rows], dtype=np.float64)
        public_macro = np.asarray([row["public_macro_f1"] for row in rows], dtype=np.float64)
        class_f1 = {
            label: float(np.mean([row["private_class_f1"][label] for row in rows])) for label in LABELS
        }
        pred_counts = Counter()
        true_counts = Counter()
        cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
        by_subject: dict[str, list[float]] = defaultdict(list)
        by_subject_delta: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            pred_counts.update(row["private_pred_distribution"])
            true_counts.update(row["private_true_distribution"])
            cm += np.asarray(row["confusion"], dtype=np.int64)
            by_subject[row["subject"]].append(row["private_macro_f1"])
            by_subject_delta[row["subject"]].append(row["delta_macro_f1"])
        subject_macro = {subject: float(np.mean(values)) for subject, values in by_subject.items()}
        subject_delta = {subject: float(np.mean(values)) for subject, values in by_subject_delta.items()}
        summaries.append(
            {
                "model": model,
                "method": method,
                "category": method_spec.category,
                "legal_for_test": method_spec.legal_for_test,
                "uses_public_labels": method_spec.uses_public_labels,
                "n_pairs": len(rows),
                "mean_macro_f1": float(np.mean(macro)),
                "std_macro_f1": float(np.std(macro)),
                "mean_accuracy": float(np.mean(acc)),
                "std_accuracy": float(np.std(acc)),
                "class_f1": class_f1,
                "class2_f1": class_f1[2],
                "class3_f1": class_f1[3],
                "worst_subject_macro_f1": float(min(subject_macro.values())),
                "win_rate_vs_baseline": float(np.mean(delta > 1e-12)),
                "mean_delta_macro_f1": float(np.mean(delta)),
                "median_delta_macro_f1": float(np.median(delta)),
                "catastrophic_regressions": int(np.sum(delta < -0.05)),
                "mean_changed_vs_baseline": float(np.mean(changed)),
                "public_macro_f1": float(np.mean(public_macro)),
                "true_distribution": {label: int(true_counts.get(label, 0)) for label in LABELS},
                "pred_distribution": {label: int(pred_counts.get(label, 0)) for label in LABELS},
                "confusion": cm.tolist(),
                "subject_macro_f1": subject_macro,
                "subject_delta_macro_f1": subject_delta,
                "subjects_positive_delta": int(np.sum([value > 1e-12 for value in subject_delta.values()])),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["model"],
            0 if row["method"] == "baseline_private_only" else 1,
            row["uses_public_labels"],
            -row["mean_delta_macro_f1"],
            row["method"],
        ),
    )


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def serialize_summary_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        row = dict(summary)
        for key in [
            "class_f1",
            "true_distribution",
            "pred_distribution",
            "confusion",
            "subject_macro_f1",
            "subject_delta_macro_f1",
        ]:
            row[key] = json.dumps(row[key], sort_keys=True)
        rows.append(row)
    return rows


def load_references(expected_ids: np.ndarray) -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    for filename in DEFAULT_REFERENCE_FILES:
        path = SCRIPT_DIR / filename
        labels = load_existing_submission(path, expected_ids)
        if labels is not None:
            refs[filename] = labels
    return refs


def select_candidate_summary(summaries: Sequence[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any] | None:
    eligible = [
        row
        for row in summaries
        if row["legal_for_test"]
        and row["method"] != "baseline_private_only"
        and row["mean_delta_macro_f1"] >= args.candidate_min_delta
        and row["win_rate_vs_baseline"] >= args.candidate_min_win_rate
        and row["catastrophic_regressions"] <= max(2, int(0.12 * row["n_pairs"]))
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            row["model"].startswith("ensemble"),
            row["mean_delta_macro_f1"],
            row["win_rate_vs_baseline"],
            -row["catastrophic_regressions"],
        ),
        reverse=True,
    )
    return eligible[0]


def fit_final_model_probs(
    subjects: Sequence[Any],
    configs: Sequence[Config],
    bundles: Sequence[ModelBundle],
    x_test: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    config_map = {config.name: config for config in configs}
    probs_by_config: dict[str, np.ndarray] = {}
    for config in configs:
        x_all, y_all, _ = prepare_subjects(subjects, config)
        pipeline = fit_pipeline(x_all, y_all, config, args.sfreq, args.filter_order, args.seed, args.max_iter)
        probs_by_config[config.name] = predict_batch(pipeline, x_test)
        print(f"fitted final {config.name} on all training subjects")
    probs_by_bundle: dict[str, np.ndarray] = {}
    for bundle in bundles:
        missing = [name for name in bundle.config_names if name not in config_map]
        if missing:
            continue
        probs_by_bundle[bundle.name] = np.mean([probs_by_config[name] for name in bundle.config_names], axis=0)
    return probs_by_bundle


def labels_for_real_method(method: str, probs: np.ndarray) -> np.ndarray:
    if method == "unlabeled_batch32_norm":
        return np.argmax(probs, axis=1).astype(np.int64)
    if method == "unlabeled_em_prior32":
        return np.argmax(em_prior_adjust(probs), axis=1).astype(np.int64)
    if method == "unlabeled_uniform_quota32":
        return quota_assign(probs, balanced_quota(len(probs)))
    if method == "unlabeled_conservative_quota32":
        return conservative_quota_labels(probs, balanced_quota(len(probs)))
    raise ValueError(f"Cannot apply non-real-test method {method}")


def maybe_write_real_candidate(
    selected: dict[str, Any] | None,
    subjects: Sequence[Any],
    configs: Sequence[Config],
    bundles: Sequence[ModelBundle],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    x_test, test_ids = load_test_data(args.test_data)
    refs = load_references(test_ids)
    candidate_rows: list[dict[str, Any]] = []
    if selected is None:
        return candidate_rows

    probs_by_bundle = fit_final_model_probs(subjects, configs, bundles, x_test, args)
    probs = probs_by_bundle[selected["model"]]
    labels = labels_for_real_method(selected["method"], probs)
    safe_model = selected["model"].replace("/", "_")
    path = args.output_dir / f"submission_task2_ppp_{safe_model}_{selected['method']}.csv"
    write_submission(path, test_ids, labels)
    validated = validate_submission(path, test_ids)
    if not np.array_equal(validated, labels):
        raise ValueError(f"Candidate validation changed labels for {path}")
    row: dict[str, Any] = {
        "filename": str(path),
        "model": selected["model"],
        "method": selected["method"],
        "distribution": distribution(labels),
        "guardrails": guardrail_notes(test_ids, labels),
        "csv_valid": True,
    }
    for ref_name, ref_labels in refs.items():
        changes = changed_rows(test_ids, ref_labels, labels)
        row[f"hamming_vs_{Path(ref_name).stem}"] = len(changes)
        row[f"changed_rows_vs_{Path(ref_name).stem}"] = changes
    candidate_rows.append(row)
    return candidate_rows


def write_candidate_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    serializable = []
    for row in rows:
        flat = dict(row)
        for key, value in list(flat.items()):
            if isinstance(value, (dict, list)):
                flat[key] = json.dumps(value, sort_keys=True)
        serializable.append(flat)
    write_csv(serializable, path)


def format_float(value: float) -> str:
    return f"{value:.4f}"


def write_report(
    summaries: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    configs: Sequence[Config],
    bundles: Sequence[ModelBundle],
    n_subjects: int,
) -> None:
    lines = [
        "# Task 2 pseudo-public/private validation",
        "",
        "Oracle pseudo-public methods below use labels only inside held-out training-subject simulation.",
        "They are not legal real-test procedures.",
        "",
        f"- held-out subjects: {n_subjects}",
        f"- random splits per subject: {args.splits}",
        f"- pseudo-public size: {args.public_size}",
        f"- configs: {', '.join(config.name for config in configs)}",
        f"- model bundles: {', '.join(bundle.name for bundle in bundles)}",
        "",
        "| model | method | legal | macro F1 | acc | class F1 0/1/2/3 | worst subj | win | mean delta | median delta | cats | mean changed | pred dist |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        class_text = "/".join(format_float(row["class_f1"][label]) for label in LABELS)
        legal = "yes" if row["legal_for_test"] else "oracle"
        lines.append(
                "| {model} | {method} | {legal} | {macro}+/-{std} | {acc}+/-{acc_std} | {class_text} | "
            "{worst} | {win} | {delta} | {median} | {cats} | {changed} | {dist} |".format(
                model=row["model"],
                method=row["method"],
                legal=legal,
                macro=format_float(row["mean_macro_f1"]),
                std=format_float(row["std_macro_f1"]),
                acc=format_float(row["mean_accuracy"]),
                acc_std=format_float(row["std_accuracy"]),
                class_text=class_text,
                worst=format_float(row["worst_subject_macro_f1"]),
                win=format_float(row["win_rate_vs_baseline"]),
                delta=format_float(row["mean_delta_macro_f1"]),
                median=format_float(row["median_delta_macro_f1"]),
                cats=row["catastrophic_regressions"],
                changed=format_float(row["mean_changed_vs_baseline"]),
                dist=json.dumps(row["pred_distribution"], sort_keys=True),
            )
        )
    lines.extend(["", "## Candidate diagnostics", ""])
    if not candidate_rows:
        lines.append(
            "No real-test candidate was emitted because no legal method cleared the configured validation threshold."
        )
    else:
        for row in candidate_rows:
            lines.append(f"- `{row['filename']}` from {row['model']} / {row['method']}")
            lines.append(f"  - distribution: {json.dumps(row['distribution'], sort_keys=True)}")
            lines.append(f"  - guardrails: {', '.join(row['guardrails']) if row['guardrails'] else 'none'}")
            for key, value in row.items():
                if key.startswith("hamming_vs_"):
                    lines.append(f"  - {key}: {value}")
                if key.startswith("changed_rows_vs_"):
                    lines.append(f"  - {key}: {json.dumps(value)}")
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = load_subjects(args.train_data)
    configs = filter_configs(make_configs(), args.config_names)
    print(f"loaded {len(subjects)} subjects")
    print(f"configs: {', '.join(config.name for config in configs)}")
    records, split_rows, bundles = run_simulation(subjects, configs, args)
    summaries = aggregate_records(records)

    write_csv(split_rows, args.output_dir / "pseudo_private_split_results.csv")
    write_csv(serialize_summary_rows(summaries), args.output_dir / "pseudo_private_method_summary.csv")
    selected = select_candidate_summary(summaries, args)
    candidate_rows = maybe_write_real_candidate(selected, subjects, configs, bundles, args)
    write_candidate_csv(candidate_rows, args.output_dir / "real_test_candidate_summary.csv")
    write_report(summaries, candidate_rows, args, configs, bundles, len(subjects))

    print("\n=== Pseudo-private method summary ===")
    for row in summaries:
        class_text = "/".join(format_float(row["class_f1"][label]) for label in LABELS)
        legal = "legal" if row["legal_for_test"] else "oracle"
        print(
            f"{row['model']} | {row['method']} ({legal}) | "
            f"macro={row['mean_macro_f1']:.4f}+/-{row['std_macro_f1']:.4f} "
            f"acc={row['mean_accuracy']:.4f}+/-{row['std_accuracy']:.4f} "
            f"class={class_text} win={row['win_rate_vs_baseline']:.3f} "
            f"delta={row['mean_delta_macro_f1']:.4f} cats={row['catastrophic_regressions']}"
        )
    if selected is None:
        print("\nNo legal real-test candidate cleared candidate thresholds.")
    else:
        print(f"\nSelected one legal real-test diagnostic candidate: {selected['model']} / {selected['method']}")
    print(f"Wrote split results to {args.output_dir / 'pseudo_private_split_results.csv'}")
    print(f"Wrote method summary to {args.output_dir / 'pseudo_private_method_summary.csv'}")
    print(f"Wrote candidate summary to {args.output_dir / 'real_test_candidate_summary.csv'}")
    print(f"Wrote report to {args.report_path}")


if __name__ == "__main__":
    main()
