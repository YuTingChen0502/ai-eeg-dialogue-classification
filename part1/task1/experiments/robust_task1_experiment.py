"""Robust within-subject Task 1 experiments.

Run from the repository root:
    python part1/task1/experiments/robust_task1_experiment.py

Or from part1/task1:
    python experiments/robust_task1_experiment.py

The script intentionally writes only new robust candidate CSVs and diagnostics.
It does not overwrite the existing public-score hedge submissions.
"""

from __future__ import annotations

import csv
import json
import random
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SEED = 42
CLASSES = np.array([0, 1, 2, 3], dtype=int)

REQUIRED_BANDS = {
    "alpha_mu_8_13": [(8.0, 13.0)],
    "beta_13_30": [(13.0, 30.0)],
    "broad_4_40": [(4.0, 40.0)],
    "high_gamma_70_125": [(70.0, 125.0)],
}

FILTERBANKS = {
    "fb_required_all": [
        (8.0, 13.0),
        (13.0, 30.0),
        (4.0, 40.0),
        (70.0, 125.0),
    ],
    "fb_motor_low": [
        (8.0, 13.0),
        (13.0, 30.0),
        (4.0, 40.0),
    ],
    "fb_broad_gamma": [
        (4.0, 40.0),
        (70.0, 125.0),
    ],
}

BAND_SETS = {**REQUIRED_BANDS, **FILTERBANKS}
PREPROCESS_MODES = ("demean", "car")
CLASSIFIER_KEYS = ("lda", "logreg:0.25", "linsvc:0.10")
REFERENCE_NAMES = (
    "task1_submission_ensemble_best.csv",
    "task1_submission_multiband_balanced.csv",
)
CANDIDATE_NAMES = {
    "robust_csp": "task1_submission_robust_csp.csv",
    "robust_filterbank": "task1_submission_robust_filterbank.csv",
    "robust_ensemble": "task1_submission_robust_ensemble.csv",
    "stable_csp_ensemble": "task1_submission_stable_csp_ensemble.csv",
    "private_hedge": "task1_submission_private_hedge.csv",
}
PUBLIC_DIAGNOSTICS = {
    "task1_submission_ensemble_best.csv": {
        "public_score": 0.62500,
        "status": "preferred_current_hedge",
        "note": "Current best public hedge; keep preferred unless stronger private-safe evidence appears.",
    },
    "task1_submission_multiband_balanced.csv": {
        "public_score": 0.62500,
        "status": "preferred_current_hedge",
        "note": "Current best public hedge; useful diversity against ensemble_best.",
    },
    "task1_submission_robust_csp.csv": {
        "public_score": 0.50000,
        "status": "failed_diagnostic_not_preferred",
        "note": "Local validation did not transfer to the tiny public split; do not treat as final candidate.",
    },
}
DEFAULT_KAGGLE_RECOMMENDATION = (
    "do_not_upload: current 0.625 hedge files remain preferred; avoid further public probing."
)


@dataclass(frozen=True)
class FeatureSpec:
    family: str
    feature_set: str
    preprocess: str
    bands: tuple[tuple[float, float], ...]
    csp_components: int = 0
    csp_shrinkage: float = 0.0

    @property
    def key(self) -> tuple:
        return (
            self.family,
            self.feature_set,
            self.preprocess,
            self.csp_components,
            round(self.csp_shrinkage, 6),
            self.bands,
        )


def find_data_dir() -> Path:
    candidates = [Path("data"), Path("part1") / "task1" / "data", Path.cwd() / "data"]
    for candidate in candidates:
        if (candidate / "train.npz").exists() and (candidate / "test.npz").exists():
            return candidate
    raise FileNotFoundError("Expected Task 1 data/train.npz and data/test.npz.")


def read_scalar(npz_file: np.lib.npyio.NpzFile, key: str, default: float) -> float:
    if key not in npz_file.files:
        return default
    return float(np.asarray(npz_file[key]).reshape(-1)[0])


def effective_band(low: float, high: float, sfreq: float) -> tuple[float, float]:
    nyq = sfreq / 2.0
    return float(low), float(min(high, nyq * 0.999))


def bandpass(x: np.ndarray, band: tuple[float, float], sfreq: float, order: int = 4) -> np.ndarray:
    nyq = sfreq / 2.0
    low, high = effective_band(*band, sfreq=sfreq)
    wn = (low / nyq, high / nyq)
    if not (0.0 < wn[0] < wn[1] < 1.0):
        raise ValueError(f"Invalid band {band} for sfreq={sfreq}; effective={low}-{high}.")
    sos = butter(order, wn, btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def preprocess_trials(x: np.ndarray, mode: str) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64).copy()
    z -= z.mean(axis=-1, keepdims=True)
    if mode == "car":
        z -= z.mean(axis=1, keepdims=True)
    elif mode != "demean":
        raise ValueError(f"Unknown preprocess mode: {mode}")
    return z


def log_variance(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.var(x, axis=-1), 1e-12))


def shrink_covariance(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    n_channels = cov.shape[0]
    target = np.eye(n_channels) / n_channels
    return (1.0 - shrinkage) * cov + shrinkage * target


def trial_covariance(trial: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    centered = trial - trial.mean(axis=-1, keepdims=True)
    cov = centered @ centered.T
    trace = max(float(np.trace(cov)), eps)
    return cov / trace


def mean_covariance(trials: np.ndarray, shrinkage: float) -> np.ndarray:
    cov = np.stack([trial_covariance(trial) for trial in trials], axis=0).mean(axis=0)
    return shrink_covariance(cov, shrinkage)


class RegularizedOVRCSP:
    def __init__(self, n_components: int = 2, shrinkage: float = 0.20, eps: float = 1e-5):
        self.n_components = int(n_components)
        self.shrinkage = float(shrinkage)
        self.eps = float(eps)
        self.filters_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RegularizedOVRCSP":
        filters = []
        n_top = int(np.ceil(self.n_components / 2))
        n_bottom = int(np.floor(self.n_components / 2))
        eye = np.eye(x.shape[1])
        for cls in CLASSES:
            cov_pos = mean_covariance(x[y == cls], self.shrinkage)
            cov_neg = mean_covariance(x[y != cls], self.shrinkage)
            composite = cov_pos + cov_neg + self.eps * eye
            try:
                vals, vecs = eigh(cov_pos + self.eps * eye, composite)
            except LinAlgError:
                vals, vecs = eigh(cov_pos + 100.0 * self.eps * eye, composite + 100.0 * self.eps * eye)
            order = np.argsort(vals)[::-1]
            selected = list(order[:n_top]) + (list(order[-n_bottom:]) if n_bottom else [])
            filters.append(vecs[:, selected].T)
        self.filters_ = np.vstack(filters)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.filters_ is None:
            raise RuntimeError("CSP has not been fitted.")
        rows = []
        for trial in x:
            projected = self.filters_ @ trial
            var = np.maximum(np.var(projected, axis=1), 1e-12)
            rows.append(np.log(var / np.maximum(var.sum(), 1e-12)))
        return np.asarray(rows, dtype=np.float64)


def make_classifier(key: str):
    if key == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    kind, value = key.split(":", 1)
    if kind == "logreg":
        return LogisticRegression(
            C=float(value),
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
            solver="lbfgs",
        )
    if kind == "linsvc":
        return LinearSVC(C=float(value), class_weight="balanced", max_iter=20000, random_state=SEED)
    raise ValueError(f"Unknown classifier: {key}")


def build_feature_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for preprocess in PREPROCESS_MODES:
        for feature_set, bands in BAND_SETS.items():
            family = "filterbank_logvar" if feature_set in FILTERBANKS else "single_band_logvar"
            specs.append(FeatureSpec(family, feature_set, preprocess, tuple(bands)))

    csp_sets = {
        "csp_alpha_mu_8_13": REQUIRED_BANDS["alpha_mu_8_13"],
        "csp_beta_13_30": REQUIRED_BANDS["beta_13_30"],
        "csp_broad_4_40": REQUIRED_BANDS["broad_4_40"],
        "csp_high_gamma_70_125": REQUIRED_BANDS["high_gamma_70_125"],
        "csp_fb_motor_low": FILTERBANKS["fb_motor_low"],
        "csp_fb_required_all": FILTERBANKS["fb_required_all"],
    }
    for preprocess in PREPROCESS_MODES:
        for feature_set, bands in csp_sets.items():
            for n_components in (2, 4):
                specs.append(
                    FeatureSpec(
                        "regularized_csp",
                        feature_set,
                        preprocess,
                        tuple(bands),
                        csp_components=n_components,
                        csp_shrinkage=0.25,
                    )
                )
    return specs


def feature_dim(spec: FeatureSpec, n_channels: int) -> int:
    if spec.family == "regularized_csp":
        return len(CLASSES) * spec.csp_components * len(spec.bands)
    return n_channels * len(spec.bands)


def extract_features(
    spec: FeatureSpec,
    x_fit_raw: np.ndarray,
    y_fit: np.ndarray,
    x_eval_raw: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_fit = preprocess_trials(x_fit_raw, spec.preprocess)
    x_eval = preprocess_trials(x_eval_raw, spec.preprocess)
    fit_parts = []
    eval_parts = []
    for band in spec.bands:
        x_fit_band = bandpass(x_fit, band, sfreq)
        x_eval_band = bandpass(x_eval, band, sfreq)
        if spec.family == "regularized_csp":
            csp = RegularizedOVRCSP(spec.csp_components, spec.csp_shrinkage).fit(x_fit_band, y_fit)
            fit_parts.append(csp.transform(x_fit_band))
            eval_parts.append(csp.transform(x_eval_band))
        else:
            fit_parts.append(log_variance(x_fit_band))
            eval_parts.append(log_variance(x_eval_band))
    return np.hstack(fit_parts), np.hstack(eval_parts)


def extract_features_from_band_arrays(
    spec: FeatureSpec,
    x_fit_bands: list[np.ndarray],
    y_fit: np.ndarray,
    x_eval_bands: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    fit_parts = []
    eval_parts = []
    for x_fit_band, x_eval_band in zip(x_fit_bands, x_eval_bands):
        if spec.family == "regularized_csp":
            csp = RegularizedOVRCSP(spec.csp_components, spec.csp_shrinkage).fit(x_fit_band, y_fit)
            fit_parts.append(csp.transform(x_fit_band))
            eval_parts.append(csp.transform(x_eval_band))
        else:
            fit_parts.append(log_variance(x_fit_band))
            eval_parts.append(log_variance(x_eval_band))
    return np.hstack(fit_parts), np.hstack(eval_parts)


def aligned_scores(clf, x_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    if hasattr(clf, "decision_function"):
        train_scores = clf.decision_function(x_train)
        eval_scores = clf.decision_function(x_eval)
    elif hasattr(clf, "predict_proba"):
        train_scores = np.log(np.maximum(clf.predict_proba(x_train), 1e-12))
        eval_scores = np.log(np.maximum(clf.predict_proba(x_eval), 1e-12))
    else:
        raise TypeError(f"Classifier {type(clf).__name__} does not expose scores.")

    train_scores = np.asarray(train_scores, dtype=np.float64)
    eval_scores = np.asarray(eval_scores, dtype=np.float64)
    if train_scores.ndim == 1:
        train_scores = np.column_stack([-train_scores, train_scores])
        eval_scores = np.column_stack([-eval_scores, eval_scores])

    train_aligned = np.full((train_scores.shape[0], len(CLASSES)), -1e6, dtype=np.float64)
    eval_aligned = np.full((eval_scores.shape[0], len(CLASSES)), -1e6, dtype=np.float64)
    for source_idx, cls in enumerate(clf.classes_):
        target_idx = int(np.where(CLASSES == int(cls))[0][0])
        train_aligned[:, target_idx] = train_scores[:, source_idx]
        eval_aligned[:, target_idx] = eval_scores[:, source_idx]

    mu = train_aligned.mean(axis=0, keepdims=True)
    sigma = train_aligned.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (eval_aligned - mu) / sigma


def metrics_from_predictions(y_true: list[int], y_pred: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)),
        "class_f1": {
            str(cls): float(score)
            for cls, score in zip(
                CLASSES,
                f1_score(y_true, y_pred, average=None, labels=CLASSES, zero_division=0),
            )
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=CLASSES).astype(int).tolist(),
    }


def prediction_distribution(pred: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(pred, dtype=int), minlength=len(CLASSES))
    return {str(cls): int(counts[i]) for i, cls in enumerate(CLASSES)}


def distribution_delta(pred: np.ndarray, ref: np.ndarray) -> dict[str, int]:
    pred_counts = np.bincount(np.asarray(pred, dtype=int), minlength=len(CLASSES))
    ref_counts = np.bincount(np.asarray(ref, dtype=int), minlength=len(CLASSES))
    return {str(cls): int(pred_counts[i] - ref_counts[i]) for i, cls in enumerate(CLASSES)}


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(np.asarray(a, dtype=int) != np.asarray(b, dtype=int)))


def changed_row_ids(ids: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> list[int]:
    pred = np.asarray(pred, dtype=int)
    ref = np.asarray(ref, dtype=int)
    return [int(sample_id) for sample_id, p, r in zip(ids.tolist(), pred.tolist(), ref.tolist()) if p != r]


def public_diagnostic_for(filename: str) -> dict:
    diagnostic = PUBLIC_DIAGNOSTICS.get(filename)
    if diagnostic is not None:
        return dict(diagnostic)
    if filename == "task1_submission_robust_ensemble.csv":
        return {
            "public_score": None,
            "status": "high_risk_not_submitted",
            "note": "High-risk candidate because test predictions remain feet-heavy.",
        }
    if filename == "task1_submission_robust_filterbank.csv":
        return {
            "public_score": None,
            "status": "skipped_class_collapse",
            "note": "Skipped because the test predictions collapse away from class 1.",
        }
    return {
        "public_score": None,
        "status": "not_submitted_not_preferred",
        "note": "Not preferred over the current 0.625 hedge files without stronger non-public evidence.",
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    data_dir = find_data_dir()
    output_dir = data_dir.parent
    train_npz = np.load(data_dir / "train.npz", allow_pickle=True)
    test_npz = np.load(data_dir / "test.npz", allow_pickle=True)

    x_train = np.asarray(train_npz["x"], dtype=np.float64)
    y_train = np.asarray(train_npz["y"], dtype=int)
    x_test = np.asarray(test_npz["x"], dtype=np.float64)
    test_ids = np.asarray(test_npz["id"], dtype=int) if "id" in test_npz.files else np.arange(len(x_test), dtype=int)
    sfreq = read_scalar(train_npz, "sfreq", 250.0)

    if x_train.shape != (16, 45, 1125) or x_test.shape != (16, 45, 1125):
        print(f"Warning: unexpected shape x_train={x_train.shape}, x_test={x_test.shape}")
    if sorted(np.unique(y_train).tolist()) != CLASSES.tolist():
        raise ValueError(f"Unexpected train labels: {np.unique(y_train).tolist()}")

    existing_csvs = {
        path.name: pd.read_csv(path)
        for path in sorted(output_dir.glob("*.csv"))
        if path.name not in CANDIDATE_NAMES.values()
    }
    reference_predictions = {}
    for name in REFERENCE_NAMES:
        path = output_dir / name
        if path.exists():
            sub = pd.read_csv(path)
            if list(sub.columns) == ["id", "label"] and sub["id"].tolist() == test_ids.tolist():
                reference_predictions[name] = sub["label"].to_numpy(dtype=int)

    min_class_count = int(np.min(np.unique(y_train, return_counts=True)[1]))
    splits = {
        "skf4": list(StratifiedKFold(n_splits=min(4, min_class_count), shuffle=True, random_state=SEED).split(x_train, y_train)),
        "rskf12": list(
            RepeatedStratifiedKFold(n_splits=min(4, min_class_count), n_repeats=12, random_state=SEED).split(
                x_train,
                y_train,
            )
        ),
        "loo": list(LeaveOneOut().split(x_train, y_train)),
    }

    specs = build_feature_specs()
    all_bands = sorted({band for spec in specs for band in spec.bands})
    filtered_train: dict[tuple[str, tuple[float, float]], np.ndarray] = {}
    filtered_test: dict[tuple[str, tuple[float, float]], np.ndarray] = {}
    print(f"Precomputing {len(all_bands)} required bands for {len(PREPROCESS_MODES)} preprocessing modes...", flush=True)
    for preprocess in PREPROCESS_MODES:
        x_train_pre = preprocess_trials(x_train, preprocess)
        x_test_pre = preprocess_trials(x_test, preprocess)
        for band in all_bands:
            filtered_train[(preprocess, band)] = bandpass(x_train_pre, band, sfreq)
            filtered_test[(preprocess, band)] = bandpass(x_test_pre, band, sfreq)

    feature_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def split_features(spec: FeatureSpec, tr_idx: np.ndarray, va_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = (spec.key, tuple(tr_idx.tolist()), tuple(va_idx.tolist()))
        if key not in feature_cache:
            x_fit_bands = [filtered_train[(spec.preprocess, band)][tr_idx] for band in spec.bands]
            x_eval_bands = [filtered_train[(spec.preprocess, band)][va_idx] for band in spec.bands]
            feature_cache[key] = extract_features_from_band_arrays(spec, x_fit_bands, y_train[tr_idx], x_eval_bands)
        return feature_cache[key]

    def fit_predict_split(
        spec: FeatureSpec,
        clf_key: str,
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
        return_scores: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        f_train, f_eval = split_features(spec, tr_idx, va_idx)
        scaler = StandardScaler().fit(f_train)
        f_train_s = scaler.transform(f_train)
        f_eval_s = scaler.transform(f_eval)
        clf = make_classifier(clf_key)
        clf.fit(f_train_s, y_train[tr_idx])
        pred = clf.predict(f_eval_s).astype(int)
        scores = aligned_scores(clf, f_train_s, f_eval_s) if return_scores else None
        return y_train[va_idx], pred, scores

    def evaluate_config(spec: FeatureSpec, clf_key: str, split_name: str) -> dict:
        fold_f1 = []
        fold_acc = []
        all_true: list[int] = []
        all_pred: list[int] = []
        for tr_idx, va_idx in splits[split_name]:
            y_eval, pred, _ = fit_predict_split(spec, clf_key, tr_idx, va_idx)
            fold_f1.append(f1_score(y_eval, pred, average="macro", labels=CLASSES, zero_division=0))
            fold_acc.append(accuracy_score(y_eval, pred))
            all_true.extend(y_eval.tolist())
            all_pred.extend(pred.tolist())
        aggregate = metrics_from_predictions(all_true, all_pred)
        return {
            "fold_macro_f1_mean": float(np.mean(fold_f1)),
            "fold_macro_f1_std": float(np.std(fold_f1)),
            "fold_accuracy_mean": float(np.mean(fold_acc)),
            "fold_accuracy_std": float(np.std(fold_acc)),
            "aggregate": aggregate,
        }

    def evaluate_ensemble(rows: list[dict], split_name: str) -> dict:
        fold_f1 = []
        fold_acc = []
        all_true: list[int] = []
        all_pred: list[int] = []
        for tr_idx, va_idx in splits[split_name]:
            score_parts = []
            y_eval_ref = None
            for row in rows:
                spec = spec_from_record(row)
                y_eval, _, scores = fit_predict_split(spec, row["classifier"], tr_idx, va_idx, return_scores=True)
                y_eval_ref = y_eval
                score_parts.append(scores)
            pred = CLASSES[np.argmax(np.mean(score_parts, axis=0), axis=1)]
            fold_f1.append(f1_score(y_eval_ref, pred, average="macro", labels=CLASSES, zero_division=0))
            fold_acc.append(accuracy_score(y_eval_ref, pred))
            all_true.extend(y_eval_ref.tolist())
            all_pred.extend(pred.tolist())
        aggregate = metrics_from_predictions(all_true, all_pred)
        return {
            "fold_macro_f1_mean": float(np.mean(fold_f1)),
            "fold_macro_f1_std": float(np.std(fold_f1)),
            "fold_accuracy_mean": float(np.mean(fold_acc)),
            "fold_accuracy_std": float(np.std(fold_acc)),
            "aggregate": aggregate,
        }

    def spec_from_record(row: dict) -> FeatureSpec:
        return FeatureSpec(
            family=row["family"],
            feature_set=row["feature_set"],
            preprocess=row["preprocess"],
            bands=tuple(tuple(band) for band in row["bands"]),
            csp_components=int(row["csp_components"]),
            csp_shrinkage=float(row["csp_shrinkage"]),
        )

    records = []
    print(f"Data dir: {data_dir.resolve()}", flush=True)
    print(f"x_train={x_train.shape}, y_train={y_train.shape}, x_test={x_test.shape}, sfreq={sfreq}", flush=True)
    print(
        f"Validation: SKF4 folds={len(splits['skf4'])}, repeated folds={len(splits['rskf12'])}, "
        f"LOO folds={len(splits['loo'])}",
        flush=True,
    )
    print(f"Feature specs={len(specs)}, classifiers={len(CLASSIFIER_KEYS)}", flush=True)

    for spec_index, spec in enumerate(specs, start=1):
        for clf_key in CLASSIFIER_KEYS:
            skf = evaluate_config(spec, clf_key, "skf4")
            rskf = evaluate_config(spec, clf_key, "rskf12")
            loo = evaluate_config(spec, clf_key, "loo")
            selection_score = (
                rskf["fold_macro_f1_mean"]
                - 0.25 * rskf["fold_macro_f1_std"]
                + 0.20 * loo["aggregate"]["macro_f1"]
                + 0.10 * skf["fold_macro_f1_mean"]
            )
            records.append(
                {
                    **asdict(spec),
                    "bands": [list(band) for band in spec.bands],
                    "classifier": clf_key,
                    "feature_dim": feature_dim(spec, x_train.shape[1]),
                    "skf4_macro_f1_mean": skf["fold_macro_f1_mean"],
                    "skf4_macro_f1_std": skf["fold_macro_f1_std"],
                    "skf4_accuracy_mean": skf["fold_accuracy_mean"],
                    "rskf12_macro_f1_mean": rskf["fold_macro_f1_mean"],
                    "rskf12_macro_f1_std": rskf["fold_macro_f1_std"],
                    "rskf12_accuracy_mean": rskf["fold_accuracy_mean"],
                    "rskf12_agg_macro_f1": rskf["aggregate"]["macro_f1"],
                    "loo_accuracy": loo["aggregate"]["accuracy"],
                    "loo_macro_f1": loo["aggregate"]["macro_f1"],
                    "loo_class_f1": loo["aggregate"]["class_f1"],
                    "loo_confusion": loo["aggregate"]["confusion"],
                    "selection_score": float(selection_score),
                }
            )
        print(
            f"Finished spec {spec_index:02d}/{len(specs)}: {spec.family} | {spec.feature_set} | {spec.preprocess}",
            flush=True,
        )

    results = pd.DataFrame(records).sort_values(
        ["selection_score", "rskf12_macro_f1_mean", "loo_macro_f1", "rskf12_macro_f1_std"],
        ascending=[False, False, False, True],
    )
    results_path = output_dir / "task1_robust_results.csv"
    results.to_csv(results_path, index=False)

    single_band_table = (
        results[results["family"] == "single_band_logvar"]
        .sort_values(["feature_set", "selection_score"], ascending=[True, False])
        .groupby("feature_set", as_index=False)
        .head(1)
        .sort_values("selection_score", ascending=False)
    )

    print("\nBest required single-band log-variance configs:")
    print(
        single_band_table[
            [
                "feature_set",
                "preprocess",
                "classifier",
                "rskf12_macro_f1_mean",
                "rskf12_macro_f1_std",
                "loo_accuracy",
                "loo_macro_f1",
                "selection_score",
            ]
        ].to_string(index=False)
    )

    print("\nTop validation configs:")
    print(
        results[
            [
                "family",
                "feature_set",
                "preprocess",
                "csp_components",
                "classifier",
                "feature_dim",
                "rskf12_macro_f1_mean",
                "rskf12_macro_f1_std",
                "loo_accuracy",
                "loo_macro_f1",
                "selection_score",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    def row_to_dict(row: pd.Series) -> dict:
        out = row.to_dict()
        out["bands"] = [list(band) for band in out["bands"]]
        return out

    best_csp = row_to_dict(results[results["family"] == "regularized_csp"].iloc[0])
    best_filterbank = row_to_dict(results[results["family"] == "filterbank_logvar"].iloc[0])

    diverse_rows: list[dict] = []
    seen_groups = set()
    for _, row in results.iterrows():
        group = (row["family"], row["feature_set"], row["preprocess"], int(row["csp_components"]))
        if group in seen_groups:
            continue
        diverse_rows.append(row_to_dict(row))
        seen_groups.add(group)
        if len(diverse_rows) == 5:
            break

    ensemble_metrics = {
        "skf4": evaluate_ensemble(diverse_rows, "skf4"),
        "rskf12": evaluate_ensemble(diverse_rows, "rskf12"),
        "loo": evaluate_ensemble(diverse_rows, "loo"),
    }

    used_signatures = {
        (best_csp["family"], best_csp["feature_set"], best_csp["preprocess"], best_csp["classifier"]),
        (best_filterbank["family"], best_filterbank["feature_set"], best_filterbank["preprocess"], best_filterbank["classifier"]),
    }
    hedge_row = None
    for _, row in results.iterrows():
        signature = (row["family"], row["feature_set"], row["preprocess"], row["classifier"])
        if signature not in used_signatures:
            hedge_row = row_to_dict(row)
            break

    def fit_full_predict(row: dict, return_scores: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        spec = spec_from_record(row)
        x_fit_bands = [filtered_train[(spec.preprocess, band)] for band in spec.bands]
        x_eval_bands = [filtered_test[(spec.preprocess, band)] for band in spec.bands]
        f_train, f_test = extract_features_from_band_arrays(spec, x_fit_bands, y_train, x_eval_bands)
        scaler = StandardScaler().fit(f_train)
        f_train_s = scaler.transform(f_train)
        f_test_s = scaler.transform(f_test)
        clf = make_classifier(row["classifier"])
        clf.fit(f_train_s, y_train)
        pred = clf.predict(f_test_s).astype(int)
        scores = aligned_scores(clf, f_train_s, f_test_s) if return_scores else None
        return pred, scores

    def fit_full_ensemble(rows: list[dict]) -> np.ndarray:
        scores = []
        for row in rows:
            _, score = fit_full_predict(row, return_scores=True)
            scores.append(score)
        return CLASSES[np.argmax(np.mean(scores, axis=0), axis=1)]

    full_score_cache: dict[tuple, np.ndarray] = {}

    def full_scores_for_row(row: dict) -> np.ndarray:
        key = (
            row["family"],
            row["feature_set"],
            row["preprocess"],
            int(row["csp_components"]),
            row["classifier"],
        )
        if key not in full_score_cache:
            _, score = fit_full_predict(row, return_scores=True)
            full_score_cache[key] = score
        return full_score_cache[key]

    def average_score_prediction(rows: list[dict]) -> np.ndarray:
        return CLASSES[np.argmax(np.mean([full_scores_for_row(row) for row in rows], axis=0), axis=1)]

    stable_pool = (
        results[
            (results["family"] == "regularized_csp")
            & (results["rskf12_macro_f1_mean"] >= 0.45)
            & (results["loo_macro_f1"] >= 0.40)
            & (results["feature_dim"] <= 64)
        ]
        .sort_values(["selection_score", "rskf12_macro_f1_mean", "loo_macro_f1"], ascending=[False, False, False])
        .head(10)
    )

    stable_candidates = []
    stable_pool_rows = [row_to_dict(row) for _, row in stable_pool.iterrows()]
    for combo_size in (3, 5):
        if len(stable_pool_rows) < combo_size:
            continue
        for combo in combinations(stable_pool_rows, combo_size):
            combo_rows = list(combo)
            pred = average_score_prediction(combo_rows)
            dist = prediction_distribution(pred)
            label2_count = dist["2"]
            max_count = max(dist.values())
            mean_selection = float(np.mean([row["selection_score"] for row in combo_rows]))
            distinct_feature_sets = len({row["feature_set"] for row in combo_rows})
            distinct_preprocess = len({row["preprocess"] for row in combo_rows})
            # This is a soft private-safety penalty, not a row-level distribution force:
            # it rejects obvious feet-heavy collapse while keeping validation strength first.
            risk_penalty = 0.04 * max(0, label2_count - 6) + 0.03 * max(0, max_count - 7)
            private_safe_score = mean_selection + 0.015 * distinct_feature_sets + 0.01 * distinct_preprocess - risk_penalty
            stable_candidates.append(
                {
                    "rows": combo_rows,
                    "pred": pred,
                    "distribution": dist,
                    "label2_count": label2_count,
                    "max_count": max_count,
                    "mean_selection": mean_selection,
                    "private_safe_score": private_safe_score,
                }
            )

    stable_csp_rows = None
    stable_csp_pred = None
    stable_csp_metrics = None
    if stable_candidates:
        stable_candidates.sort(
            key=lambda item: (
                item["private_safe_score"],
                -item["label2_count"],
                -item["max_count"],
                item["mean_selection"],
            ),
            reverse=True,
        )
        stable_choice = stable_candidates[0]
        stable_csp_rows = stable_choice["rows"]
        stable_csp_pred = stable_choice["pred"]
        stable_csp_metrics = {
            "skf4": evaluate_ensemble(stable_csp_rows, "skf4"),
            "rskf12": evaluate_ensemble(stable_csp_rows, "rskf12"),
            "loo": evaluate_ensemble(stable_csp_rows, "loo"),
            "selection": {
                "private_safe_score": float(stable_choice["private_safe_score"]),
                "mean_member_selection_score": float(stable_choice["mean_selection"]),
                "label2_count": int(stable_choice["label2_count"]),
                "max_label_count": int(stable_choice["max_count"]),
            },
        }
        print("\nStable CSP ensemble selection:")
        print(
            f"distribution={stable_choice['distribution']} "
            f"private_safe_score={stable_choice['private_safe_score']:.4f} "
            f"mean_member_selection={stable_choice['mean_selection']:.4f}"
        )

    raw_candidates: list[dict] = []
    csp_pred, _ = fit_full_predict(best_csp)
    raw_candidates.append({"key": "robust_csp", "kind": "single", "row": best_csp, "pred": csp_pred, "metrics": None})

    filterbank_pred, _ = fit_full_predict(best_filterbank)
    raw_candidates.append(
        {"key": "robust_filterbank", "kind": "single", "row": best_filterbank, "pred": filterbank_pred, "metrics": None}
    )

    ensemble_pred = fit_full_ensemble(diverse_rows)
    raw_candidates.append(
        {
            "key": "robust_ensemble",
            "kind": "ensemble",
            "members": diverse_rows,
            "pred": ensemble_pred,
            "metrics": ensemble_metrics,
        }
    )

    if stable_csp_rows is not None and stable_csp_pred is not None:
        raw_candidates.append(
            {
                "key": "stable_csp_ensemble",
                "kind": "ensemble",
                "members": stable_csp_rows,
                "pred": stable_csp_pred,
                "metrics": stable_csp_metrics,
            }
        )

    if hedge_row is not None:
        hedge_pred, _ = fit_full_predict(hedge_row)
        raw_candidates.append({"key": "private_hedge", "kind": "single", "row": hedge_row, "pred": hedge_pred, "metrics": None})

    unique_candidates: list[dict] = []
    for candidate in raw_candidates:
        pred = candidate["pred"]
        if any(np.array_equal(pred, old["pred"]) for old in unique_candidates):
            print(f"Skipping {candidate['key']}: identical to another generated candidate.")
            continue
        unique_candidates.append(candidate)

    def validate_submission(path: Path) -> pd.DataFrame:
        sub = pd.read_csv(path)
        if list(sub.columns) != ["id", "label"]:
            raise ValueError(f"Bad header in {path.name}: {sub.columns.tolist()}")
        if len(sub) != len(test_ids):
            raise ValueError(f"Bad row count in {path.name}: {len(sub)}")
        if sub["id"].duplicated().any():
            raise ValueError(f"Duplicate IDs in {path.name}")
        if sub["id"].tolist() != test_ids.tolist():
            raise ValueError(f"Test order changed in {path.name}")
        labels = sub["label"].to_numpy()
        if not set(labels.tolist()).issubset(set(CLASSES.tolist())):
            raise ValueError(f"Bad labels in {path.name}: {sorted(set(labels.tolist()))}")
        return sub

    candidate_summaries = []
    previous_predictions = {
        name: df["label"].to_numpy(dtype=int)
        for name, df in existing_csvs.items()
        if list(df.columns) == ["id", "label"] and len(df) == len(test_ids) and df["id"].tolist() == test_ids.tolist()
    }

    for candidate in unique_candidates[:6]:
        filename = CANDIDATE_NAMES[candidate["key"]]
        public_diagnostic = public_diagnostic_for(filename)
        path = output_dir / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label"])
            for sample_id, label in zip(test_ids.tolist(), candidate["pred"].tolist()):
                writer.writerow([int(sample_id), int(label)])
        validate_submission(path)

        identical_to = sorted(
            name for name, previous_pred in previous_predictions.items() if np.array_equal(previous_pred, candidate["pred"])
        )
        reference_hamming = {
            name: (hamming(candidate["pred"], ref_pred) if name in reference_predictions else None)
            for name, ref_pred in {name: reference_predictions.get(name) for name in REFERENCE_NAMES}.items()
        }
        reference_changed_ids = {
            name: (changed_row_ids(test_ids, candidate["pred"], ref_pred) if name in reference_predictions else None)
            for name, ref_pred in {name: reference_predictions.get(name) for name in REFERENCE_NAMES}.items()
        }
        reference_distribution_delta = {
            name: (distribution_delta(candidate["pred"], ref_pred) if name in reference_predictions else None)
            for name, ref_pred in {name: reference_predictions.get(name) for name in REFERENCE_NAMES}.items()
        }
        if candidate["kind"] == "single":
            row = candidate["row"]
            validation = {
                "selection_score": float(row["selection_score"]),
                "rskf12_macro_f1_mean": float(row["rskf12_macro_f1_mean"]),
                "rskf12_macro_f1_std": float(row["rskf12_macro_f1_std"]),
                "loo_accuracy": float(row["loo_accuracy"]),
                "loo_macro_f1": float(row["loo_macro_f1"]),
                "loo_class_f1": row["loo_class_f1"],
                "loo_confusion": row["loo_confusion"],
                "model": {
                    "family": row["family"],
                    "feature_set": row["feature_set"],
                    "preprocess": row["preprocess"],
                    "classifier": row["classifier"],
                    "feature_dim": int(row["feature_dim"]),
                    "csp_components": int(row["csp_components"]),
                },
            }
        else:
            validation = {
                "selection_score": None,
                "rskf12_macro_f1_mean": float(candidate["metrics"]["rskf12"]["fold_macro_f1_mean"]),
                "rskf12_macro_f1_std": float(candidate["metrics"]["rskf12"]["fold_macro_f1_std"]),
                "loo_accuracy": float(candidate["metrics"]["loo"]["aggregate"]["accuracy"]),
                "loo_macro_f1": float(candidate["metrics"]["loo"]["aggregate"]["macro_f1"]),
                "loo_class_f1": candidate["metrics"]["loo"]["aggregate"]["class_f1"],
                "loo_confusion": candidate["metrics"]["loo"]["aggregate"]["confusion"],
                "members": [
                    {
                        "family": row["family"],
                        "feature_set": row["feature_set"],
                        "preprocess": row["preprocess"],
                        "classifier": row["classifier"],
                        "csp_components": int(row["csp_components"]),
                    }
                    for row in candidate["members"]
                ],
            }

        candidate_summaries.append(
            {
                "filename": filename,
                "candidate_key": candidate["key"],
                "distribution": prediction_distribution(candidate["pred"]),
                "validation": validation,
                "hamming_vs_task1_submission_ensemble_best": reference_hamming["task1_submission_ensemble_best.csv"],
                "hamming_vs_task1_submission_multiband_balanced": reference_hamming[
                    "task1_submission_multiband_balanced.csv"
                ],
                "changed_ids_vs_task1_submission_ensemble_best": reference_changed_ids[
                    "task1_submission_ensemble_best.csv"
                ],
                "changed_ids_vs_task1_submission_multiband_balanced": reference_changed_ids[
                    "task1_submission_multiband_balanced.csv"
                ],
                "distribution_delta_vs_task1_submission_ensemble_best": reference_distribution_delta[
                    "task1_submission_ensemble_best.csv"
                ],
                "distribution_delta_vs_task1_submission_multiband_balanced": reference_distribution_delta[
                    "task1_submission_multiband_balanced.csv"
                ],
                "identical_to_existing": identical_to,
                "public_diagnostic": public_diagnostic,
                "kaggle_recommendation": DEFAULT_KAGGLE_RECOMMENDATION,
            }
        )
        previous_predictions[filename] = candidate["pred"]
        print(f"Wrote {filename}: distribution={prediction_distribution(candidate['pred'])}, identical_to={identical_to}")

    for summary in candidate_summaries:
        diagnostic_status = summary["public_diagnostic"]["status"]
        if diagnostic_status == "failed_diagnostic_not_preferred":
            summary["risk_assessment"] = "failed diagnostic: public score 0.50000, not preferred"
        elif summary["filename"] == "task1_submission_robust_ensemble.csv":
            summary["risk_assessment"] = "high: feet-heavy test distribution, keep as analysis only"
        elif summary["filename"] == "task1_submission_robust_filterbank.csv":
            summary["risk_assessment"] = "high: class-collapse pattern, skipped"
        elif summary["validation"]["rskf12_macro_f1_mean"] >= 0.45 and summary["validation"]["loo_macro_f1"] >= 0.25:
            summary["risk_assessment"] = "medium: locally competitive but still high variance with 16 trials"
        else:
            summary["risk_assessment"] = "high: weak or unstable local evidence"

    ranked_candidates = sorted(
        candidate_summaries,
        key=lambda item: (
            item["validation"]["rskf12_macro_f1_mean"],
            item["validation"]["loo_macro_f1"],
            -len(item["identical_to_existing"]),
        ),
        reverse=True,
    )
    for priority, summary in enumerate(ranked_candidates, start=1):
        summary["local_validation_rank"] = priority
        summary["recommended_upload_priority"] = None

    candidate_summary_path = output_dir / "task1_robust_candidate_summary.csv"
    pd.DataFrame(
        [
            {
                "filename": item["filename"],
                "distribution": json.dumps(item["distribution"], sort_keys=True),
                "rskf12_macro_f1_mean": item["validation"]["rskf12_macro_f1_mean"],
                "rskf12_macro_f1_std": item["validation"]["rskf12_macro_f1_std"],
                "loo_accuracy": item["validation"]["loo_accuracy"],
                "loo_macro_f1": item["validation"]["loo_macro_f1"],
                "hamming_vs_ensemble_best": item["hamming_vs_task1_submission_ensemble_best"],
                "hamming_vs_multiband_balanced": item["hamming_vs_task1_submission_multiband_balanced"],
                "changed_ids_vs_ensemble_best": json.dumps(
                    item["changed_ids_vs_task1_submission_ensemble_best"],
                    sort_keys=True,
                ),
                "changed_ids_vs_multiband_balanced": json.dumps(
                    item["changed_ids_vs_task1_submission_multiband_balanced"],
                    sort_keys=True,
                ),
                "dist_delta_vs_ensemble_best": json.dumps(
                    item["distribution_delta_vs_task1_submission_ensemble_best"],
                    sort_keys=True,
                ),
                "dist_delta_vs_multiband_balanced": json.dumps(
                    item["distribution_delta_vs_task1_submission_multiband_balanced"],
                    sort_keys=True,
                ),
                "identical_to_existing": ",".join(item["identical_to_existing"]),
                "public_score": item["public_diagnostic"]["public_score"],
                "public_status": item["public_diagnostic"]["status"],
                "public_note": item["public_diagnostic"]["note"],
                "local_validation_rank": item["local_validation_rank"],
                "recommended_upload_priority": item["recommended_upload_priority"],
                "kaggle_recommendation": item["kaggle_recommendation"],
                "risk_assessment": item["risk_assessment"],
            }
            for item in ranked_candidates
        ]
    ).to_csv(candidate_summary_path, index=False)

    run_summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "shape": {
            "x_train": list(x_train.shape),
            "y_train": list(y_train.shape),
            "x_test": list(x_test.shape),
            "sfreq": sfreq,
        },
        "required_single_band_best": single_band_table[
            [
                "feature_set",
                "preprocess",
                "classifier",
                "rskf12_macro_f1_mean",
                "rskf12_macro_f1_std",
                "loo_accuracy",
                "loo_macro_f1",
                "selection_score",
            ]
        ].to_dict(orient="records"),
        "top_configs": results.head(20)[
            [
                "family",
                "feature_set",
                "preprocess",
                "csp_components",
                "classifier",
                "feature_dim",
                "rskf12_macro_f1_mean",
                "rskf12_macro_f1_std",
                "loo_accuracy",
                "loo_macro_f1",
                "selection_score",
            ]
        ].to_dict(orient="records"),
        "ensemble_members": [
            {
                "family": row["family"],
                "feature_set": row["feature_set"],
                "preprocess": row["preprocess"],
                "classifier": row["classifier"],
                "csp_components": int(row["csp_components"]),
                "selection_score": float(row["selection_score"]),
            }
            for row in diverse_rows
        ],
        "candidate_summaries": ranked_candidates,
        "reference_files_found": sorted(reference_predictions.keys()),
        "reference_files_missing": [name for name in REFERENCE_NAMES if name not in reference_predictions],
        "public_diagnostics": PUBLIC_DIAGNOSTICS,
        "overall_recommendation": {
            "preferred_files": [
                "task1_submission_ensemble_best.csv",
                "task1_submission_multiband_balanced.csv",
            ],
            "kaggle_action": "do_not_submit_more_candidates_yet",
            "rationale": (
                "robust_csp scored 0.50000 publicly, so local validation is not transferring reliably; "
                "current 0.625 hedge files remain preferred."
            ),
        },
        "reference_distributions": {
            name: prediction_distribution(pred)
            for name, pred in reference_predictions.items()
        },
        "stable_csp_ensemble_members": [
            {
                "family": row["family"],
                "feature_set": row["feature_set"],
                "preprocess": row["preprocess"],
                "classifier": row["classifier"],
                "csp_components": int(row["csp_components"]),
                "selection_score": float(row["selection_score"]),
            }
            for row in (stable_csp_rows or [])
        ],
    }

    summary_path = output_dir / "task1_robust_run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("\nCandidate summary:")
    print(pd.read_csv(candidate_summary_path).to_string(index=False))
    print(f"\nWrote diagnostics: {results_path.name}, {candidate_summary_path.name}, {summary_path.name}")


if __name__ == "__main__":
    main()
