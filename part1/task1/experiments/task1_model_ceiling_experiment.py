"""Task 1 model-ceiling experiments.

This helper is intentionally internal-only. It uses train labels plus the
unlabeled test signals for unsupervised normalization/alignment, writes a small
set of candidate CSVs for review, and records diagnostics without using public
leaderboard feedback as labels.

Run from the repository root:
    python part1/task1/experiments/task1_model_ceiling_experiment.py

Or from part1/task1:
    python experiments/task1_model_ceiling_experiment.py
"""

from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

import robust_task1_experiment as robust


SEED = 42
CLASSES = robust.CLASSES
REPEATS = 3

REQUIRED_BANDS = {
    "alpha_mu_8_13": ((8.0, 13.0),),
    "beta_13_30": ((13.0, 30.0),),
    "broad_4_40": ((4.0, 40.0),),
    "high_gamma_70_125": ((70.0, 125.0),),
}
FILTERBANKS = {
    "fb_required_all": (
        (8.0, 13.0),
        (13.0, 30.0),
        (4.0, 40.0),
        (70.0, 125.0),
    ),
    "fb_motor_low": (
        (8.0, 13.0),
        (13.0, 30.0),
        (4.0, 40.0),
    ),
}
BAND_SETS = {**REQUIRED_BANDS, **FILTERBANKS}

WINDOWS_SECONDS = {
    "full": None,
    "0.5-2.5s": (0.5, 2.5),
    "1.0-3.0s": (1.0, 3.0),
    "1.5-3.5s": (1.5, 3.5),
}
REFERENCE_FILES = [
    "task1_submission_ensemble_best.csv",
    "task1_submission_multiband_balanced.csv",
    "task1_submission_hybrid_two_public_winners.csv",
    "task1_submission_robust_csp.csv",
    "task1_submission_multiband_best.csv",
]
CANDIDATE_FILES = {
    "transductive": "task1_submission_model_ceiling_transductive.csv",
    "ovo_csp": "task1_submission_model_ceiling_ovo_csp.csv",
    "consensus": "task1_submission_model_ceiling_consensus.csv",
    "private_hedge": "task1_submission_model_ceiling_private_hedge.csv",
}


@dataclass(frozen=True)
class CeilingConfig:
    family: str
    feature_set: str
    preprocess: str
    adaptation: str
    window: str
    bands: tuple[tuple[float, float], ...]
    csp_components: int = 2
    csp_shrinkage: float = 0.25

    @property
    def key(self) -> tuple:
        return (
            self.family,
            self.feature_set,
            self.preprocess,
            self.adaptation,
            self.window,
            self.csp_components,
            round(self.csp_shrinkage, 6),
            self.bands,
        )

    @property
    def short_name(self) -> str:
        return "|".join(
            [
                self.family,
                self.feature_set,
                self.preprocess,
                self.adaptation,
                self.window,
                f"csp{self.csp_components}",
            ]
        )


class BinaryRegularizedCSP:
    def __init__(self, n_components: int = 2, shrinkage: float = 0.25, eps: float = 1e-5):
        self.n_components = int(n_components)
        self.shrinkage = float(shrinkage)
        self.eps = float(eps)
        self.filters_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y_positive: np.ndarray) -> "BinaryRegularizedCSP":
        y_positive = np.asarray(y_positive, dtype=bool)
        if len(np.unique(y_positive)) != 2:
            raise ValueError("Binary CSP needs both classes.")
        cov_pos = robust.mean_covariance(x[y_positive], self.shrinkage)
        cov_neg = robust.mean_covariance(x[~y_positive], self.shrinkage)
        eye = np.eye(x.shape[1])
        composite = cov_pos + cov_neg + self.eps * eye
        try:
            vals, vecs = eigh(cov_pos + self.eps * eye, composite)
        except LinAlgError:
            vals, vecs = eigh(cov_pos + 100.0 * self.eps * eye, composite + 100.0 * self.eps * eye)
        order = np.argsort(vals)[::-1]
        n_top = int(math.ceil(self.n_components / 2))
        n_bottom = int(math.floor(self.n_components / 2))
        selected = list(order[:n_top]) + (list(order[-n_bottom:]) if n_bottom else [])
        self.filters_ = vecs[:, selected].T
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.filters_ is None:
            raise RuntimeError("Binary CSP has not been fitted.")
        rows = []
        for trial in x:
            projected = self.filters_ @ trial
            var = np.maximum(np.var(projected, axis=1), 1e-12)
            rows.append(np.log(var / np.maximum(var.sum(), 1e-12)))
        return np.asarray(rows, dtype=np.float64)


class PairwiseCSPEnsemble:
    def __init__(self, n_components: int = 2, shrinkage: float = 0.25):
        self.n_components = int(n_components)
        self.shrinkage = float(shrinkage)
        self.pair_models_: list[dict] = []
        self.score_mu_: np.ndarray | None = None
        self.score_sigma_: np.ndarray | None = None

    def fit(self, x_bands: list[np.ndarray], y: np.ndarray) -> "PairwiseCSPEnsemble":
        self.pair_models_ = []
        for cls_a, cls_b in combinations(CLASSES.tolist(), 2):
            pair_mask = np.isin(y, [cls_a, cls_b])
            pair_y = y[pair_mask]
            csps = []
            feature_parts = []
            for x_band in x_bands:
                pair_x = x_band[pair_mask]
                csp = BinaryRegularizedCSP(self.n_components, self.shrinkage).fit(pair_x, pair_y == cls_b)
                csps.append(csp)
                feature_parts.append(csp.transform(pair_x))
            f_pair = np.hstack(feature_parts)
            scaler = StandardScaler().fit(f_pair)
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(scaler.transform(f_pair), pair_y)
            self.pair_models_.append(
                {
                    "classes": (int(cls_a), int(cls_b)),
                    "csps": csps,
                    "scaler": scaler,
                    "clf": clf,
                }
            )
        raw_train_scores = self._raw_scores(x_bands)
        self.score_mu_ = raw_train_scores.mean(axis=0, keepdims=True)
        self.score_sigma_ = raw_train_scores.std(axis=0, keepdims=True)
        self.score_sigma_ = np.where(self.score_sigma_ < 1e-8, 1.0, self.score_sigma_)
        return self

    def _raw_scores(self, x_bands: list[np.ndarray]) -> np.ndarray:
        if not self.pair_models_:
            raise RuntimeError("Pairwise CSP ensemble has not been fitted.")
        n_eval = x_bands[0].shape[0]
        log_scores = np.zeros((n_eval, len(CLASSES)), dtype=np.float64)
        votes = np.zeros((n_eval, len(CLASSES)), dtype=np.float64)
        exposure = np.zeros(len(CLASSES), dtype=np.float64)

        for model in self.pair_models_:
            feature_parts = [
                csp.transform(x_band)
                for csp, x_band in zip(model["csps"], x_bands)
            ]
            features = np.hstack(feature_parts)
            features = model["scaler"].transform(features)
            clf = model["clf"]
            probs = np.maximum(clf.predict_proba(features), 1e-12)
            pair_pred = clf.predict(features).astype(int)
            for source_idx, cls in enumerate(clf.classes_):
                target_idx = int(np.where(CLASSES == int(cls))[0][0])
                log_scores[:, target_idx] += np.log(probs[:, source_idx])
                exposure[target_idx] += 1.0
            for row_idx, cls in enumerate(pair_pred):
                target_idx = int(np.where(CLASSES == int(cls))[0][0])
                votes[row_idx, target_idx] += 1.0

        exposure = np.where(exposure < 1.0, 1.0, exposure)
        return log_scores / exposure.reshape(1, -1) + 0.05 * votes

    def predict_scores(self, x_bands: list[np.ndarray]) -> np.ndarray:
        if self.score_mu_ is None or self.score_sigma_ is None:
            raise RuntimeError("Pairwise CSP ensemble has not been fitted.")
        return (self._raw_scores(x_bands) - self.score_mu_) / self.score_sigma_


def load_data() -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    data_dir = robust.find_data_dir()
    train_npz = np.load(data_dir / "train.npz", allow_pickle=True)
    test_npz = np.load(data_dir / "test.npz", allow_pickle=True)
    x_train = np.asarray(train_npz["x"], dtype=np.float64)
    y_train = np.asarray(train_npz["y"], dtype=int)
    x_test = np.asarray(test_npz["x"], dtype=np.float64)
    test_ids = np.asarray(test_npz["id"], dtype=int) if "id" in test_npz.files else np.arange(len(x_test), dtype=int)
    sfreq = robust.read_scalar(train_npz, "sfreq", 250.0)
    return data_dir, x_train, y_train, x_test, test_ids, sfreq


def window_slice(window_name: str, n_times: int, sfreq: float) -> slice:
    seconds = WINDOWS_SECONDS[window_name]
    if seconds is None:
        return slice(0, n_times)
    start = max(0, int(round(seconds[0] * sfreq)))
    stop = min(n_times, int(round(seconds[1] * sfreq)))
    if stop <= start:
        raise ValueError(f"Invalid window {window_name}: {start}:{stop}")
    return slice(start, stop)


def joint_channel_zscore(x_fit: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([x_fit, x_eval], axis=0)
    mu = combined.mean(axis=(0, 2), keepdims=True)
    sigma = combined.std(axis=(0, 2), keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (x_fit - mu) / sigma, (x_eval - mu) / sigma


def euclidean_align(x_fit: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([x_fit, x_eval], axis=0)
    cov = np.stack([robust.trial_covariance(trial) for trial in combined], axis=0).mean(axis=0)
    eye = np.eye(cov.shape[0])
    vals, vecs = eigh(cov + 1e-6 * eye)
    vals = np.maximum(vals, 1e-8)
    whitening = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return (
        np.einsum("ij,njt->nit", whitening, x_fit),
        np.einsum("ij,njt->nit", whitening, x_eval),
    )


def prepare_band_arrays(
    cfg: CeilingConfig,
    x_fit_raw: np.ndarray,
    x_eval_raw: np.ndarray,
    sfreq: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    x_fit = robust.preprocess_trials(x_fit_raw, cfg.preprocess)
    x_eval = robust.preprocess_trials(x_eval_raw, cfg.preprocess)
    win = window_slice(cfg.window, x_fit.shape[-1], sfreq)
    fit_bands = []
    eval_bands = []
    for band in cfg.bands:
        x_fit_band = robust.bandpass(x_fit, band, sfreq)[..., win]
        x_eval_band = robust.bandpass(x_eval, band, sfreq)[..., win]
        if cfg.adaptation == "joint_zscore":
            x_fit_band, x_eval_band = joint_channel_zscore(x_fit_band, x_eval_band)
        elif cfg.adaptation == "ea_joint":
            x_fit_band, x_eval_band = euclidean_align(x_fit_band, x_eval_band)
        elif cfg.adaptation != "none":
            raise ValueError(f"Unknown adaptation: {cfg.adaptation}")
        fit_bands.append(x_fit_band)
        eval_bands.append(x_eval_band)
    return fit_bands, eval_bands


def fit_predict_config(
    cfg: CeilingConfig,
    x_fit_raw: np.ndarray,
    y_fit: np.ndarray,
    x_eval_raw: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    fit_bands, eval_bands = prepare_band_arrays(cfg, x_fit_raw, x_eval_raw, sfreq)
    if cfg.family == "ovr_csp":
        train_parts = []
        eval_parts = []
        for x_fit_band, x_eval_band in zip(fit_bands, eval_bands):
            csp = robust.RegularizedOVRCSP(cfg.csp_components, cfg.csp_shrinkage).fit(x_fit_band, y_fit)
            train_parts.append(csp.transform(x_fit_band))
            eval_parts.append(csp.transform(x_eval_band))
        f_train = np.hstack(train_parts)
        f_eval = np.hstack(eval_parts)
        scaler = StandardScaler().fit(f_train)
        f_train_s = scaler.transform(f_train)
        f_eval_s = scaler.transform(f_eval)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(f_train_s, y_fit)
        scores = robust.aligned_scores(clf, f_train_s, f_eval_s)
        pred = CLASSES[np.argmax(scores, axis=1)]
        return pred.astype(int), scores

    if cfg.family == "ovo_csp":
        model = PairwiseCSPEnsemble(cfg.csp_components, cfg.csp_shrinkage).fit(fit_bands, y_fit)
        scores = model.predict_scores(eval_bands)
        pred = CLASSES[np.argmax(scores, axis=1)]
        return pred.astype(int), scores

    raise ValueError(f"Unknown family: {cfg.family}")


def fit_predict_model_set(
    configs: list[CeilingConfig],
    x_fit_raw: np.ndarray,
    y_fit: np.ndarray,
    x_eval_raw: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_parts = []
    pred_parts = []
    for cfg in configs:
        pred, scores = fit_predict_config(cfg, x_fit_raw, y_fit, x_eval_raw, sfreq)
        pred_parts.append(pred)
        score_parts.append(scores)
    mean_scores = np.mean(score_parts, axis=0)
    pred = CLASSES[np.argmax(mean_scores, axis=1)]
    return pred.astype(int), mean_scores, np.vstack(pred_parts)


def aggregate_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    return robust.metrics_from_predictions(y_true, y_pred)


def evaluate_model_set(
    configs: list[CeilingConfig],
    x_train: np.ndarray,
    y_train: np.ndarray,
    sfreq: float,
) -> dict:
    min_class_count = int(np.min(np.unique(y_train, return_counts=True)[1]))
    splitters = {
        f"rskf{REPEATS}": list(
            RepeatedStratifiedKFold(
                n_splits=min(4, min_class_count),
                n_repeats=REPEATS,
                random_state=SEED,
            ).split(x_train, y_train)
        ),
        "loo": list(LeaveOneOut().split(x_train, y_train)),
    }
    out = {}
    for split_name, splits in splitters.items():
        fold_f1 = []
        fold_acc = []
        all_true: list[int] = []
        all_pred: list[int] = []
        for tr_idx, va_idx in splits:
            pred, _, _ = fit_predict_model_set(configs, x_train[tr_idx], y_train[tr_idx], x_train[va_idx], sfreq)
            y_eval = y_train[va_idx]
            fold_f1.append(f1_score(y_eval, pred, average="macro", labels=CLASSES, zero_division=0))
            fold_acc.append(accuracy_score(y_eval, pred))
            all_true.extend(y_eval.tolist())
            all_pred.extend(pred.tolist())
        out[split_name] = {
            "fold_macro_f1_mean": float(np.mean(fold_f1)),
            "fold_macro_f1_std": float(np.std(fold_f1)),
            "fold_accuracy_mean": float(np.mean(fold_acc)),
            "fold_accuracy_std": float(np.std(fold_acc)),
            "aggregate": aggregate_metrics(all_true, all_pred),
        }
    return out


def selection_score(metrics: dict) -> float:
    rskf = metrics[f"rskf{REPEATS}"]
    loo = metrics["loo"]["aggregate"]
    return float(
        rskf["fold_macro_f1_mean"]
        - 0.30 * rskf["fold_macro_f1_std"]
        + 0.25 * loo["macro_f1"]
    )


def build_configs() -> list[CeilingConfig]:
    configs: dict[tuple, CeilingConfig] = {}

    def add(cfg: CeilingConfig) -> None:
        configs[cfg.key] = cfg

    for feature_set in REQUIRED_BANDS:
        for adaptation in ("none", "ea_joint"):
            add(
                CeilingConfig(
                    family="ovr_csp",
                    feature_set=feature_set,
                    preprocess="car",
                    adaptation=adaptation,
                    window="full",
                    bands=BAND_SETS[feature_set],
                )
            )

    add(
        CeilingConfig(
            family="ovr_csp",
            feature_set="alpha_mu_8_13",
            preprocess="car",
            adaptation="joint_zscore",
            window="full",
            bands=BAND_SETS["alpha_mu_8_13"],
        )
    )

    for window in ("0.5-2.5s", "1.0-3.0s", "1.5-3.5s"):
        for feature_set in ("alpha_mu_8_13", "high_gamma_70_125"):
            add(
                CeilingConfig(
                    family="ovr_csp",
                    feature_set=feature_set,
                    preprocess="car",
                    adaptation="ea_joint",
                    window=window,
                    bands=BAND_SETS[feature_set],
                )
            )

    for adaptation in ("none", "ea_joint"):
        add(
            CeilingConfig(
                family="ovr_csp",
                feature_set="fb_required_all",
                preprocess="car",
                adaptation=adaptation,
                window="full",
                bands=BAND_SETS["fb_required_all"],
            )
        )

    for adaptation in ("none", "ea_joint"):
        add(
            CeilingConfig(
                family="ovo_csp",
                feature_set="alpha_mu_8_13",
                preprocess="car",
                adaptation=adaptation,
                window="full",
                bands=BAND_SETS["alpha_mu_8_13"],
                )
            )

    return list(configs.values())


def prediction_distribution(pred: np.ndarray) -> dict[str, int]:
    return robust.prediction_distribution(np.asarray(pred, dtype=int))


def load_references(output_dir: Path, test_ids: np.ndarray) -> tuple[dict[str, np.ndarray], list[str]]:
    references = {}
    missing = []
    for name in REFERENCE_FILES:
        path = output_dir / name
        if not path.exists():
            missing.append(name)
            continue
        df = pd.read_csv(path)
        if list(df.columns) != ["id", "label"] or df["id"].tolist() != test_ids.tolist():
            missing.append(f"{name} (invalid format/order)")
            continue
        labels = df["label"].to_numpy(dtype=int)
        if not set(labels.tolist()).issubset(set(CLASSES.tolist())):
            missing.append(f"{name} (invalid labels)")
            continue
        references[name] = labels
    return references, missing


def write_submission(path: Path, test_ids: np.ndarray, pred: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for sample_id, label in zip(test_ids.tolist(), pred.tolist()):
            writer.writerow([int(sample_id), int(label)])
    validate_submission(path, test_ids)


def validate_submission(path: Path, test_ids: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "label"]:
        raise ValueError(f"{path.name}: expected header id,label")
    if df["id"].tolist() != test_ids.tolist():
        raise ValueError(f"{path.name}: id order mismatch")
    labels = df["label"].to_numpy(dtype=int)
    if len(labels) != len(test_ids) or not set(labels.tolist()).issubset(set(CLASSES.tolist())):
        raise ValueError(f"{path.name}: labels must be in {CLASSES.tolist()}")
    return df


def class_counts(pred: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(pred, dtype=int), minlength=len(CLASSES))


def score_margins(scores: np.ndarray) -> np.ndarray:
    sorted_scores = np.sort(scores, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


def reference_comparison(
    pred: np.ndarray,
    test_ids: np.ndarray,
    references: dict[str, np.ndarray],
) -> tuple[dict[str, int], dict[str, list[int]], dict[str, dict[str, int]]]:
    hamming = {}
    changed = {}
    deltas = {}
    for name, ref_pred in references.items():
        hamming[name] = robust.hamming(pred, ref_pred)
        changed[name] = robust.changed_row_ids(test_ids, pred, ref_pred)
        deltas[name] = robust.distribution_delta(pred, ref_pred)
    return hamming, changed, deltas


def risk_flags(
    pred: np.ndarray,
    metrics: dict,
    references: dict[str, np.ndarray],
    hamming: dict[str, int],
) -> list[str]:
    counts = class_counts(pred)
    flags = []
    if int(counts.min()) == 0:
        flags.append("class_collapse")
    if int(counts[2]) >= 7:
        flags.append("feet_heavy")
    if int(counts.max()) >= 8:
        flags.append("extreme_distribution")
    if metrics[f"rskf{REPEATS}"]["fold_macro_f1_std"] > 0.25:
        flags.append("high_cv_variance")
    if metrics["loo"]["aggregate"]["macro_f1"] < 0.45:
        flags.append("weak_loo")
    if "task1_submission_robust_csp.csv" in references:
        if hamming.get("task1_submission_robust_csp.csv", 99) <= 2:
            flags.append("near_failed_robust_csp")
    if hamming.get("task1_submission_ensemble_best.csv", 0) >= 11:
        flags.append("large_shift_vs_ensemble_best")
    if not flags:
        flags.append("none")
    return flags


def private_safe_score(record: dict) -> float:
    flags = set(record["risk_flags"])
    penalty = 0.0
    penalty += 0.18 if "class_collapse" in flags else 0.0
    penalty += 0.12 if "feet_heavy" in flags else 0.0
    penalty += 0.08 if "extreme_distribution" in flags else 0.0
    penalty += 0.05 if "large_shift_vs_ensemble_best" in flags else 0.0
    penalty += 0.05 if "near_failed_robust_csp" in flags else 0.0
    return float(record["selection_score"] - penalty)


def summarize_config_record(
    cfg: CeilingConfig,
    metrics: dict,
    pred: np.ndarray,
    scores: np.ndarray,
    test_ids: np.ndarray,
    references: dict[str, np.ndarray],
) -> dict:
    hamming, changed, deltas = reference_comparison(pred, test_ids, references)
    flags = risk_flags(pred, metrics, references, hamming)
    margins = score_margins(scores)
    record = {
        **asdict(cfg),
        "bands": [list(band) for band in cfg.bands],
        "config_name": cfg.short_name,
        "selection_score": selection_score(metrics),
        f"rskf{REPEATS}_macro_f1_mean": metrics[f"rskf{REPEATS}"]["fold_macro_f1_mean"],
        f"rskf{REPEATS}_macro_f1_std": metrics[f"rskf{REPEATS}"]["fold_macro_f1_std"],
        "loo_macro_f1": metrics["loo"]["aggregate"]["macro_f1"],
        "loo_class_f1": metrics["loo"]["aggregate"]["class_f1"],
        "loo_confusion": metrics["loo"]["aggregate"]["confusion"],
        "prediction_distribution": prediction_distribution(pred),
        "score_margin_mean": float(np.mean(margins)),
        "score_margin_min": float(np.min(margins)),
        "hamming": hamming,
        "changed_ids": changed,
        "distribution_delta": deltas,
        "risk_flags": flags,
        "prediction": pred,
        "scores": scores,
        "metrics": metrics,
    }
    record["private_safe_score"] = private_safe_score(record)
    return record


def select_records(records: list[dict]) -> dict[str, list[dict]]:
    records_sorted = sorted(records, key=lambda row: row["private_safe_score"], reverse=True)
    transductive = [
        row
        for row in records_sorted
        if row["family"] == "ovr_csp"
        and row["adaptation"] != "none"
        and "class_collapse" not in row["risk_flags"]
        and "feet_heavy" not in row["risk_flags"]
    ]
    ovo = [
        row
        for row in records_sorted
        if row["family"] == "ovo_csp"
        and "class_collapse" not in row["risk_flags"]
        and "feet_heavy" not in row["risk_flags"]
    ]
    safer = [
        row
        for row in records_sorted
        if "class_collapse" not in row["risk_flags"]
        and "feet_heavy" not in row["risk_flags"]
        and "weak_loo" not in row["risk_flags"]
    ]

    consensus: list[dict] = []
    seen_predictions: set[tuple[int, ...]] = set()
    for required in [
        transductive[:2],
        ovo[:1],
        [row for row in safer if row["adaptation"] == "none"][:1],
        [row for row in safer if row["feature_set"] == "fb_required_all"][:1],
    ]:
        for row in required:
            key = tuple(row["prediction"].tolist())
            if key not in seen_predictions:
                consensus.append(row)
                seen_predictions.add(key)
    for row in safer:
        if len(consensus) >= 4:
            break
        key = tuple(row["prediction"].tolist())
        if key not in seen_predictions:
            consensus.append(row)
            seen_predictions.add(key)

    return {
        "transductive": transductive[:1] or records_sorted[:1],
        "ovo_csp": ovo[:1] or records_sorted[:1],
        "consensus": consensus,
    }


def candidate_summary(
    candidate_key: str,
    configs: list[CeilingConfig],
    metrics: dict,
    pred: np.ndarray,
    scores: np.ndarray,
    member_preds: np.ndarray,
    test_ids: np.ndarray,
    references: dict[str, np.ndarray],
) -> dict:
    hamming, changed, deltas = reference_comparison(pred, test_ids, references)
    flags = risk_flags(pred, metrics, references, hamming)
    margins = score_margins(scores)
    member_vote_fraction = []
    if member_preds.ndim == 2 and member_preds.shape[0] > 1:
        for col in range(member_preds.shape[1]):
            counts = np.bincount(member_preds[:, col], minlength=len(CLASSES))
            member_vote_fraction.append(float(counts.max() / member_preds.shape[0]))
    else:
        member_vote_fraction = [1.0] * len(pred)
    return {
        "candidate_key": candidate_key,
        "filename": CANDIDATE_FILES[candidate_key],
        "member_count": len(configs),
        "members": [cfg.short_name for cfg in configs],
        "prediction_distribution": prediction_distribution(pred),
        f"rskf{REPEATS}_macro_f1_mean": metrics[f"rskf{REPEATS}"]["fold_macro_f1_mean"],
        f"rskf{REPEATS}_macro_f1_std": metrics[f"rskf{REPEATS}"]["fold_macro_f1_std"],
        "loo_macro_f1": metrics["loo"]["aggregate"]["macro_f1"],
        "loo_class_f1": metrics["loo"]["aggregate"]["class_f1"],
        "loo_confusion": metrics["loo"]["aggregate"]["confusion"],
        "score_margin_mean": float(np.mean(margins)),
        "score_margin_min": float(np.min(margins)),
        "member_vote_fraction_mean": float(np.mean(member_vote_fraction)),
        "member_vote_fraction_min": float(np.min(member_vote_fraction)),
        "hamming": hamming,
        "changed_ids": changed,
        "distribution_delta": deltas,
        "risk_flags": flags,
        "prediction": pred,
        "scores": scores,
        "member_preds": member_preds,
        "metrics": metrics,
        "selection_score": selection_score(metrics),
    }


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    columns = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_jsonish(value) -> str:
    return json.dumps(value, sort_keys=True)


def get_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def write_report(
    report_path: Path,
    data_dir: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    sfreq: float,
    references: dict[str, np.ndarray],
    missing_refs: list[str],
    records: list[dict],
    candidates: list[dict],
    recommendation: str,
) -> None:
    top_configs_df = pd.DataFrame(
        [
            {
                "rank": idx + 1,
                "family": row["family"],
                "feature_set": row["feature_set"],
                "preprocess": row["preprocess"],
                "adaptation": row["adaptation"],
                "window": row["window"],
                "score": row["selection_score"],
                f"rskf{REPEATS}_f1": row[f"rskf{REPEATS}_macro_f1_mean"],
                f"rskf{REPEATS}_std": row[f"rskf{REPEATS}_macro_f1_std"],
                "loo_f1": row["loo_macro_f1"],
                "dist": format_jsonish(row["prediction_distribution"]),
                "risk": ",".join(row["risk_flags"]),
            }
            for idx, row in enumerate(sorted(records, key=lambda item: item["private_safe_score"], reverse=True)[:12])
        ]
    )
    candidate_df = pd.DataFrame(
        [
            {
                "candidate": item["candidate_key"],
                "file": item["filename"],
                "members": item["member_count"],
                "dist": format_jsonish(item["prediction_distribution"]),
                f"rskf{REPEATS}_f1": item[f"rskf{REPEATS}_macro_f1_mean"],
                f"rskf{REPEATS}_std": item[f"rskf{REPEATS}_macro_f1_std"],
                "loo_f1": item["loo_macro_f1"],
                "class_f1": format_jsonish(item["loo_class_f1"]),
                "hamming": format_jsonish(item["hamming"]),
                "changed_rows": format_jsonish(item["changed_ids"]),
                "risk": ",".join(item["risk_flags"]),
            }
            for item in candidates
        ]
    )

    method_lines = [
        "- Transductive OVR CSP: per-trial demeaning or common-average reference, bandpass filtering, compact one-vs-rest CSP, shrinkage LDA, and optional unlabeled-test joint channel z-scoring or Euclidean covariance alignment.",
        "- Pairwise OVO CSP: six binary CSP plus shrinkage-LDA pair classifiers, combined by calibrated pair log-probability scores.",
        "- Time-window robustness: full window plus 0.5-2.5s, 1.0-3.0s, and 1.5-3.5s motor-imagery windows; the window grid is intentionally small.",
        "- Consensus-stable ensemble: model members selected by repeated CV, LOO F1, non-extreme prediction distribution, and avoidance of known failed patterns.",
    ]
    text = "\n".join(
        [
            "# Task 1 Model-Ceiling Experiment",
            "",
            "This is an internal review artifact. It does not use test labels, infer public labels, or submit to Kaggle.",
            "",
            "## Environment",
            "",
            f"- Worktree: `{Path.cwd()}`",
            f"- Branch: `{get_branch()}`",
            f"- Data dir: `{data_dir}`",
            f"- Shapes: x_train={tuple(x_train.shape)}, y_train={tuple(y_train.shape)}, x_test={tuple(x_test.shape)}",
            f"- Sampling rate: {sfreq}",
            f"- Class counts: `{format_jsonish(prediction_distribution(y_train))}`",
            "",
            "## Reference Files",
            "",
            f"- Found: `{', '.join(references.keys()) if references else 'none'}`",
            f"- Missing or invalid: `{', '.join(missing_refs) if missing_refs else 'none'}`",
            "",
            "## Methods",
            "",
            "\n".join(method_lines),
            "",
            "The unlabeled test set is used only inside label-free normalization/alignment transforms. During validation, the held-out fold is treated analogously as unlabeled evaluation data for those transforms.",
            "",
            "## Search Scope",
            "",
            f"- Configs evaluated: {len(records)}",
            f"- Repeated CV: Stratified 4-fold x {REPEATS} repeats",
            "- LOO: aggregate predictions across all 16 held-out trials",
            f"- Required bands covered: `{', '.join(REQUIRED_BANDS.keys())}`",
            "",
            "## Top Internal Configs",
            "",
            dataframe_to_markdown(top_configs_df),
            "",
            "## Candidate Results",
            "",
            dataframe_to_markdown(candidate_df),
            "",
            "## Recommendation",
            "",
            recommendation,
            "",
            "Generated CSVs and diagnostics are review-only artifacts and should remain uncommitted.",
            "",
        ]
    )
    report_path.write_text(text, encoding="utf-8")


def recommendation_from_candidates(candidates: list[dict]) -> str:
    viable = []
    for item in candidates:
        flags = set(item["risk_flags"])
        stable = (
            item[f"rskf{REPEATS}_macro_f1_mean"] >= 0.62
            and item[f"rskf{REPEATS}_macro_f1_std"] <= 0.22
            and item["loo_macro_f1"] >= 0.58
        )
        if stable and flags == {"none"}:
            viable.append(item)
    if viable:
        best = max(viable, key=lambda item: item["selection_score"])
        return (
            f"`{best['filename']}` appears stronger internally, but it should still be reviewed before any upload. "
            "The current 0.625 hedge remains the default until this evidence is accepted."
        )

    sparse = []
    for item in candidates:
        flags = set(item["risk_flags"])
        stable_enough = item[f"rskf{REPEATS}_macro_f1_mean"] >= 0.58 and item["loo_macro_f1"] >= 0.55
        if stable_enough and flags == {"none"}:
            sparse.append(item)
    if sparse:
        best = max(sparse, key=lambda item: item["selection_score"])
        return (
            f"`{best['filename']}` is the only candidate worth sparse Kaggle diagnostic later, "
            "but not as an immediate replacement for the current 0.625 hedge."
        )

    return (
        "No new candidate is worth uploading from this phase. The current preferred hedge remains "
        "`task1_submission_ensemble_best.csv` plus `task1_submission_multiband_balanced.csv`."
    )


def write_diagnostics(output_dir: Path, records: list[dict], candidates: list[dict], test_ids: np.ndarray) -> None:
    config_rows = []
    for row in records:
        config_rows.append(
            {
                "config_name": row["config_name"],
                "family": row["family"],
                "feature_set": row["feature_set"],
                "preprocess": row["preprocess"],
                "adaptation": row["adaptation"],
                "window": row["window"],
                "bands": json.dumps(row["bands"]),
                "selection_score": row["selection_score"],
                "private_safe_score": row["private_safe_score"],
                f"rskf{REPEATS}_macro_f1_mean": row[f"rskf{REPEATS}_macro_f1_mean"],
                f"rskf{REPEATS}_macro_f1_std": row[f"rskf{REPEATS}_macro_f1_std"],
                "loo_macro_f1": row["loo_macro_f1"],
                "loo_class_f1": json.dumps(row["loo_class_f1"]),
                "loo_confusion": json.dumps(row["loo_confusion"]),
                "prediction_distribution": json.dumps(row["prediction_distribution"]),
                "score_margin_mean": row["score_margin_mean"],
                "score_margin_min": row["score_margin_min"],
                "hamming": json.dumps(row["hamming"]),
                "changed_ids": json.dumps(row["changed_ids"]),
                "risk_flags": ",".join(row["risk_flags"]),
            }
        )
    pd.DataFrame(config_rows).sort_values("private_safe_score", ascending=False).to_csv(
        output_dir / "task1_model_ceiling_config_results.csv",
        index=False,
    )

    candidate_rows = []
    for item in candidates:
        candidate_rows.append(
            {
                "candidate_key": item["candidate_key"],
                "filename": item["filename"],
                "member_count": item["member_count"],
                "members": json.dumps(item["members"]),
                "prediction_distribution": json.dumps(item["prediction_distribution"]),
                f"rskf{REPEATS}_macro_f1_mean": item[f"rskf{REPEATS}_macro_f1_mean"],
                f"rskf{REPEATS}_macro_f1_std": item[f"rskf{REPEATS}_macro_f1_std"],
                "loo_macro_f1": item["loo_macro_f1"],
                "loo_class_f1": json.dumps(item["loo_class_f1"]),
                "loo_confusion": json.dumps(item["loo_confusion"]),
                "hamming": json.dumps(item["hamming"]),
                "changed_ids": json.dumps(item["changed_ids"]),
                "distribution_delta": json.dumps(item["distribution_delta"]),
                "score_margin_mean": item["score_margin_mean"],
                "score_margin_min": item["score_margin_min"],
                "member_vote_fraction_mean": item["member_vote_fraction_mean"],
                "member_vote_fraction_min": item["member_vote_fraction_min"],
                "risk_flags": ",".join(item["risk_flags"]),
            }
        )
    pd.DataFrame(candidate_rows).to_csv(output_dir / "task1_model_ceiling_candidate_summary.csv", index=False)

    row_records = []
    for row_idx, sample_id in enumerate(test_ids.tolist()):
        record = {"id": int(sample_id)}
        for item in candidates:
            margins = score_margins(item["scores"])
            member_preds = item["member_preds"]
            if member_preds.shape[0] > 1:
                counts = np.bincount(member_preds[:, row_idx], minlength=len(CLASSES))
                vote_fraction = float(counts.max() / member_preds.shape[0])
            else:
                vote_fraction = 1.0
            record[f"{item['candidate_key']}_label"] = int(item["prediction"][row_idx])
            record[f"{item['candidate_key']}_margin"] = float(margins[row_idx])
            record[f"{item['candidate_key']}_member_vote_fraction"] = vote_fraction
        row_records.append(record)
    pd.DataFrame(row_records).to_csv(output_dir / "task1_model_ceiling_row_stability.csv", index=False)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    data_dir, x_train, y_train, x_test, test_ids, sfreq = load_data()
    output_dir = data_dir.parent
    if x_train.shape != (16, 45, 1125) or x_test.shape != (16, 45, 1125):
        print(f"Warning: unexpected shape x_train={x_train.shape}, x_test={x_test.shape}")
    if sorted(np.unique(y_train).tolist()) != CLASSES.tolist():
        raise ValueError(f"Unexpected Task 1 labels: {np.unique(y_train).tolist()}")

    references, missing_refs = load_references(output_dir, test_ids)
    configs = build_configs()
    print(f"Data dir: {data_dir.resolve()}")
    print(f"x_train={x_train.shape}, y_train={y_train.shape}, x_test={x_test.shape}, sfreq={sfreq}")
    print(f"Reference files found: {list(references.keys())}")
    print(f"Evaluating {len(configs)} compact model-ceiling configs...")

    records = []
    for idx, cfg in enumerate(configs, start=1):
        metrics = evaluate_model_set([cfg], x_train, y_train, sfreq)
        pred, scores, _ = fit_predict_model_set([cfg], x_train, y_train, x_test, sfreq)
        record = summarize_config_record(cfg, metrics, pred, scores, test_ids, references)
        records.append(record)
        print(
            f"[{idx:02d}/{len(configs)}] {cfg.short_name} "
            f"score={record['selection_score']:.4f} safe={record['private_safe_score']:.4f} "
            f"dist={record['prediction_distribution']} risk={','.join(record['risk_flags'])}",
            flush=True,
        )

    selected = select_records(records)
    candidates = []
    seen_candidate_predictions: set[tuple[int, ...]] = set()
    for candidate_key, selected_records in selected.items():
        configs_for_candidate = [
            CeilingConfig(
                family=row["family"],
                feature_set=row["feature_set"],
                preprocess=row["preprocess"],
                adaptation=row["adaptation"],
                window=row["window"],
                bands=tuple(tuple(band) for band in row["bands"]),
                csp_components=int(row["csp_components"]),
                csp_shrinkage=float(row["csp_shrinkage"]),
            )
            for row in selected_records
        ]
        metrics = evaluate_model_set(configs_for_candidate, x_train, y_train, sfreq)
        pred, scores, member_preds = fit_predict_model_set(configs_for_candidate, x_train, y_train, x_test, sfreq)
        pred_key = tuple(pred.tolist())
        if pred_key in seen_candidate_predictions:
            print(f"Skipping duplicate candidate {candidate_key}: prediction already written.")
            continue
        seen_candidate_predictions.add(pred_key)
        item = candidate_summary(
            candidate_key,
            configs_for_candidate,
            metrics,
            pred,
            scores,
            member_preds,
            test_ids,
            references,
        )
        write_submission(output_dir / item["filename"], test_ids, pred)
        candidates.append(item)
        print(
            f"Wrote {item['filename']}: dist={item['prediction_distribution']} "
            f"rskf{REPEATS}={item[f'rskf{REPEATS}_macro_f1_mean']:.4f}+/-"
            f"{item[f'rskf{REPEATS}_macro_f1_std']:.4f} loo={item['loo_macro_f1']:.4f} "
            f"risk={','.join(item['risk_flags'])}",
            flush=True,
        )

    recommendation = recommendation_from_candidates(candidates)
    write_diagnostics(output_dir, records, candidates, test_ids)
    write_report(
        output_dir / "task1_model_ceiling_report.md",
        data_dir,
        x_train,
        y_train,
        x_test,
        sfreq,
        references,
        missing_refs,
        records,
        candidates,
        recommendation,
    )
    print(f"Wrote diagnostics under {output_dir}")
    print(f"Wrote report: {output_dir / 'task1_model_ceiling_report.md'}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
