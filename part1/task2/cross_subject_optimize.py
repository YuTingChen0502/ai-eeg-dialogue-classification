"""Task 2 cross-subject optimization experiments.

This script focuses on validation-backed generalization rather than public-row
editing. It evaluates preprocessing, feature, and model variants with LOSO,
then fits selected models on all training subjects and writes Kaggle candidate
CSVs. Generated CSV/checkpoint/output files are intentionally ignored by git.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

LABELS = (0, 1, 2, 3)
EPS = 1e-10

BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
    "broad": (4.0, 40.0),
    "gamma": (70.0, 124.875),
}

BAND_SETS: dict[str, tuple[str, ...]] = {
    "broad_only": ("broad",),
    "alpha_beta_broad": ("alpha", "beta", "broad"),
    "expanded_filterbank": ("theta", "alpha", "low_beta", "high_beta", "broad", "gamma"),
    "alpha_beta_broad_gamma": ("alpha", "beta", "broad", "gamma"),
    "broad_gamma": ("broad", "gamma"),
}


@dataclass(frozen=True)
class Subject:
    name: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Config:
    name: str
    raw_norm: str
    align: str
    band_set: str
    feature_family: str
    feature_norm: str
    model: str
    C: float | None = None
    csp_components: int = 2


@dataclass
class SubjectTransform:
    raw_norm: str
    align: str
    raw_center: np.ndarray | None = None
    raw_scale: np.ndarray | None = None
    align_matrix: np.ndarray | None = None


@dataclass
class FeatureNormalizer:
    mode: str
    center: np.ndarray | None = None
    scale: np.ndarray | None = None


@dataclass
class FeatureState:
    config: Config
    sfreq: float
    filter_order: int
    csp_filters: dict[str, np.ndarray] | None = None
    tangent_refs: dict[str, np.ndarray] | None = None


@dataclass
class FittedPipeline:
    config: Config
    sfreq: float
    filter_order: int
    state: FeatureState
    scaler: StandardScaler
    model: Any


@dataclass(frozen=True)
class EvalRecord:
    config: Config
    mean_macro_f1: float
    std_macro_f1: float
    mean_accuracy: float
    std_accuracy: float
    confusion: np.ndarray
    true_distribution: dict[int, int]
    pred_distribution: dict[int, int]
    per_subject: list[dict[str, Any]]
    oof_ids: list[str]
    oof_true: np.ndarray
    oof_pred: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 2 LOSO cross-subject optimization.")
    parser.add_argument("--train-data", type=Path, default=Path("data/train"))
    parser.add_argument("--test-data", type=Path, default=Path("data/test.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--top-k-ensemble", type=int, default=4)
    parser.add_argument("--results-csv", type=Path, default=Path("task2_cross_loso_results.csv"))
    parser.add_argument("--oof-csv", type=Path, default=Path("task2_cross_oof_predictions.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/task2_cross_subject/task2_model.pkl"))
    return parser.parse_args()


def distribution(labels: Iterable[int]) -> dict[int, int]:
    counts = Counter(int(label) for label in labels)
    return {label: counts.get(label, 0) for label in LABELS}


def load_subjects(data_dir: Path) -> list[Subject]:
    subjects = []
    for path in sorted(data_dir.glob("subject*.npz")):
        arrays = np.load(path, allow_pickle=True)
        if "x" not in arrays or "y" not in arrays:
            raise KeyError(f"{path} must contain x and y arrays")
        x = np.asarray(arrays["x"], dtype=np.float64)
        y = np.asarray(arrays["y"], dtype=np.int64)
        if x.ndim != 3:
            raise ValueError(f"{path}: expected x shape (N,C,T), got {x.shape}")
        if len(x) != len(y):
            raise ValueError(f"{path}: x/y length mismatch")
        subjects.append(Subject(path.stem, x, y))
    if not subjects:
        raise FileNotFoundError(f"No subject*.npz files found under {data_dir}")
    return subjects


def resolve_test_file(data_path: Path) -> Path:
    if data_path.is_file():
        return data_path
    for candidate in [data_path / "test.npz", data_path / "task2" / "test.npz"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate test.npz from {data_path}")


def load_test_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.load(resolve_test_file(data_path), allow_pickle=True)
    if "x" not in arrays:
        raise KeyError(f"{data_path}: missing x")
    x = np.asarray(arrays["x"], dtype=np.float64)
    ids = arrays["id"] if "id" in arrays else np.arange(len(x), dtype=np.int64)
    return x, np.asarray(ids, dtype=np.int64)


def effective_band(name: str, sfreq: float) -> tuple[float, float]:
    low, high = BANDS[name]
    return float(low), float(min(high, (sfreq / 2.0) * 0.999))


def bandpass(x: np.ndarray, low: float, high: float, sfreq: float, order: int) -> np.ndarray:
    wn = (low / (sfreq / 2.0), high / (sfreq / 2.0))
    if not (0.0 < wn[0] < wn[1] < 1.0):
        raise ValueError(f"Invalid band {low}-{high} for sfreq={sfreq}")
    sos = butter(order, wn, btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def safe_scale(scale: np.ndarray) -> np.ndarray:
    return np.where(np.abs(scale) < EPS, 1.0, scale)


def fit_subject_transform(x: np.ndarray, raw_norm: str, align: str) -> SubjectTransform:
    transform = SubjectTransform(raw_norm=raw_norm, align=align)
    if raw_norm == "subject_standard":
        transform.raw_center = x.mean(axis=(0, 2), keepdims=True)
        transform.raw_scale = safe_scale(x.std(axis=(0, 2), keepdims=True))
    elif raw_norm == "robust_subject":
        transform.raw_center = np.median(x, axis=(0, 2), keepdims=True)
        q75 = np.percentile(x, 75, axis=(0, 2), keepdims=True)
        q25 = np.percentile(x, 25, axis=(0, 2), keepdims=True)
        transform.raw_scale = safe_scale((q75 - q25) / 1.349)
    elif raw_norm in {"none", "trial_zscore", "channel_zscore"}:
        pass
    else:
        raise ValueError(f"Unknown raw_norm: {raw_norm}")

    normalized = apply_subject_transform(x, transform, include_alignment=False)
    if align == "euclidean":
        transform.align_matrix = euclidean_alignment_matrix(normalized)
    elif align != "none":
        raise ValueError(f"Unknown alignment: {align}")
    return transform


def apply_subject_transform(x: np.ndarray, transform: SubjectTransform, *, include_alignment: bool = True) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    if transform.raw_norm == "none":
        out = out.copy()
    elif transform.raw_norm == "trial_zscore":
        center = out.mean(axis=(1, 2), keepdims=True)
        scale = safe_scale(out.std(axis=(1, 2), keepdims=True))
        out = (out - center) / scale
    elif transform.raw_norm == "channel_zscore":
        center = out.mean(axis=-1, keepdims=True)
        scale = safe_scale(out.std(axis=-1, keepdims=True))
        out = (out - center) / scale
    elif transform.raw_norm in {"subject_standard", "robust_subject"}:
        out = (out - transform.raw_center) / transform.raw_scale
    else:
        raise ValueError(f"Unknown raw_norm: {transform.raw_norm}")

    if include_alignment and transform.align == "euclidean":
        out = np.einsum("cd,ndt->nct", transform.align_matrix, out, optimize=True)
    return out


def trial_covariances(x: np.ndarray, *, normalize_trace: bool = True, eps: float = 1e-3) -> np.ndarray:
    covs = []
    n_channels = x.shape[1]
    eye = np.eye(n_channels)
    for trial in x:
        centered = trial - trial.mean(axis=-1, keepdims=True)
        cov = centered @ centered.T / max(1, centered.shape[-1] - 1)
        trace = np.trace(cov)
        if normalize_trace and trace > EPS:
            cov = cov / trace
        if eps > 0:
            scale = np.trace(cov) / n_channels
            cov = cov + eps * max(float(scale), EPS) * eye
        covs.append((cov + cov.T) * 0.5)
    return np.asarray(covs, dtype=np.float64)


def euclidean_alignment_matrix(x: np.ndarray) -> np.ndarray:
    cov = np.mean(trial_covariances(x, normalize_trace=False, eps=1e-3), axis=0)
    vals, vecs = eigh((cov + cov.T) * 0.5)
    vals = np.maximum(vals, EPS)
    return (vecs / np.sqrt(vals)) @ vecs.T


def fit_subject_feature_normalizer(features: np.ndarray, mode: str) -> FeatureNormalizer:
    if mode == "none":
        return FeatureNormalizer(mode)
    if mode == "subject":
        center = features.mean(axis=0)
        scale = safe_scale(features.std(axis=0))
        return FeatureNormalizer(mode, center, scale)
    if mode == "robust_subject":
        center = np.median(features, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        q25 = np.percentile(features, 25, axis=0)
        scale = safe_scale((q75 - q25) / 1.349)
        return FeatureNormalizer(mode, center, scale)
    raise ValueError(f"Unknown feature_norm: {mode}")


def apply_feature_normalizer(features: np.ndarray, normalizer: FeatureNormalizer) -> np.ndarray:
    if normalizer.mode == "none":
        return features
    return (features - normalizer.center) / normalizer.scale


def logvar_meanabs_features(filtered: np.ndarray) -> np.ndarray:
    var = np.maximum(np.var(filtered, axis=-1), EPS)
    return np.concatenate([np.log(var), np.mean(np.abs(filtered), axis=-1)], axis=1)


def bandpower_logvar_features(filtered: np.ndarray) -> np.ndarray:
    var = np.maximum(np.var(filtered, axis=-1), EPS)
    power = np.maximum(np.mean(filtered**2, axis=-1), EPS)
    rms = np.sqrt(power)
    return np.concatenate([np.log(var), np.log(power), rms], axis=1)


def covariance_upper_features(filtered: np.ndarray) -> np.ndarray:
    covs = trial_covariances(filtered, normalize_trace=True, eps=1e-3)
    iu = np.triu_indices(filtered.shape[1])
    return covs[:, iu[0], iu[1]]


def matrix_invsqrt(mat: np.ndarray) -> np.ndarray:
    vals, vecs = eigh((mat + mat.T) * 0.5)
    vals = np.maximum(vals, EPS)
    return (vecs / np.sqrt(vals)) @ vecs.T


def matrix_log(mat: np.ndarray) -> np.ndarray:
    vals, vecs = eigh((mat + mat.T) * 0.5)
    vals = np.maximum(vals, EPS)
    return (vecs * np.log(vals)) @ vecs.T


def vectorize_symmetric(mats: np.ndarray, *, scale_offdiag: bool) -> np.ndarray:
    iu = np.triu_indices(mats.shape[1])
    feats = mats[:, iu[0], iu[1]].copy()
    if scale_offdiag:
        feats[:, iu[0] != iu[1]] *= np.sqrt(2.0)
    return feats


def fit_csp_filters(filtered_by_band: dict[str, np.ndarray], y: np.ndarray, n_components: int) -> dict[str, np.ndarray]:
    filters = {}
    for band_name, x_band in filtered_by_band.items():
        covs = trial_covariances(x_band, normalize_trace=True, eps=1e-3)
        band_filters = []
        for cls in LABELS:
            cls_cov = np.mean(covs[y == cls], axis=0)
            rest_cov = np.mean(covs[y != cls], axis=0)
            composite = cls_cov + rest_cov
            vals, vecs = eigh((cls_cov + cls_cov.T) * 0.5, (composite + composite.T) * 0.5)
            order = np.argsort(vals)
            selected = np.concatenate([order[:n_components], order[-n_components:]])
            band_filters.append(vecs[:, selected].T)
        filters[band_name] = np.concatenate(band_filters, axis=0)
    return filters


def csp_features(filtered_by_band: dict[str, np.ndarray], filters: dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for band_name, x_band in filtered_by_band.items():
        w = filters[band_name]
        projected = np.einsum("kc,nct->nkt", w, x_band, optimize=True)
        var = np.maximum(np.var(projected, axis=-1), EPS)
        parts.append(np.log(var / np.maximum(var.sum(axis=1, keepdims=True), EPS)))
    return np.concatenate(parts, axis=1).astype(np.float64)


def filtered_bands(x: np.ndarray, config: Config, sfreq: float, filter_order: int) -> dict[str, np.ndarray]:
    out = {}
    for band_name in BAND_SETS[config.band_set]:
        low, high = effective_band(band_name, sfreq)
        out[band_name] = bandpass(x, low, high, sfreq, filter_order)
    return out


def fit_feature_state(train_x: np.ndarray, y: np.ndarray, config: Config, sfreq: float, filter_order: int) -> FeatureState:
    state = FeatureState(config=config, sfreq=sfreq, filter_order=filter_order)
    if config.feature_family == "csp":
        train_bands = filtered_bands(train_x, config, sfreq, filter_order)
        state.csp_filters = fit_csp_filters(train_bands, y, config.csp_components)
    elif config.feature_family == "tangent":
        refs = {}
        for band_name, x_band in filtered_bands(train_x, config, sfreq, filter_order).items():
            refs[band_name] = np.mean(trial_covariances(x_band, normalize_trace=False, eps=1e-3), axis=0)
        state.tangent_refs = refs
    return state


def transform_features(x: np.ndarray, state: FeatureState) -> np.ndarray:
    config = state.config
    bands = filtered_bands(x, config, state.sfreq, state.filter_order)
    if config.feature_family == "logvar_meanabs":
        return np.concatenate([logvar_meanabs_features(bands[name]) for name in bands], axis=1).astype(np.float64)
    if config.feature_family == "bandpower_logvar":
        return np.concatenate([bandpower_logvar_features(bands[name]) for name in bands], axis=1).astype(np.float64)
    if config.feature_family == "cov_upper":
        return np.concatenate([covariance_upper_features(bands[name]) for name in bands], axis=1).astype(np.float64)
    if config.feature_family == "csp":
        return csp_features(bands, state.csp_filters)
    if config.feature_family == "tangent":
        parts = []
        for band_name, x_band in bands.items():
            ref_inv = matrix_invsqrt(state.tangent_refs[band_name])
            logs = []
            for cov in trial_covariances(x_band, normalize_trace=False, eps=1e-3):
                logs.append(matrix_log(ref_inv @ cov @ ref_inv))
            parts.append(vectorize_symmetric(np.asarray(logs, dtype=np.float64), scale_offdiag=True))
        return np.concatenate(parts, axis=1).astype(np.float64)
    raise ValueError(f"Unknown feature_family: {config.feature_family}")


def make_model(config: Config, seed: int, max_iter: int):
    if config.model == "logreg":
        return LogisticRegression(
            C=1.0 if config.C is None else float(config.C),
            class_weight="balanced",
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed,
        )
    if config.model == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if config.model == "linear_svc":
        return LinearSVC(
            C=1.0 if config.C is None else float(config.C),
            class_weight="balanced",
            max_iter=max_iter * 5,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {config.model}")


def prediction_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    elif hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x), dtype=np.float64)
        if raw.ndim == 1:
            raw = np.column_stack([-raw, raw])
        raw = raw - raw.max(axis=1, keepdims=True)
        raw = np.exp(raw)
        raw = raw / raw.sum(axis=1, keepdims=True)
    else:
        pred = np.asarray(model.predict(x), dtype=np.int64)
        raw = np.full((len(pred), len(LABELS)), EPS, dtype=np.float64)
        raw[np.arange(len(pred)), pred] = 1.0

    classes = np.asarray(getattr(model, "classes_", LABELS), dtype=np.int64)
    probs = np.full((len(x), len(LABELS)), EPS, dtype=np.float64)
    for source_idx, label in enumerate(classes):
        if int(label) in LABELS:
            probs[:, int(label)] = raw[:, source_idx]
    return probs / probs.sum(axis=1, keepdims=True)


def prepare_subjects(subjects: Sequence[Subject], config: Config) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    x_list, y_list, names = [], [], []
    for subject in subjects:
        transform = fit_subject_transform(subject.x, config.raw_norm, config.align)
        x_list.append(apply_subject_transform(subject.x, transform))
        y_list.append(subject.y)
        names.append(subject.name)
    return x_list, y_list, names


def fit_pipeline(
    x_by_subject: Sequence[np.ndarray],
    y_by_subject: Sequence[np.ndarray],
    config: Config,
    sfreq: float,
    filter_order: int,
    seed: int,
    max_iter: int,
) -> FittedPipeline:
    train_x = np.concatenate(x_by_subject, axis=0)
    y_train = np.concatenate(y_by_subject, axis=0)
    state = fit_feature_state(train_x, y_train, config, sfreq, filter_order)
    features_by_subject = [transform_features(x, state) for x in x_by_subject]
    normalizers = [fit_subject_feature_normalizer(feats, config.feature_norm) for feats in features_by_subject]
    normalized = [apply_feature_normalizer(feats, norm) for feats, norm in zip(features_by_subject, normalizers)]
    x_train = np.concatenate(normalized, axis=0)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    model = make_model(config, seed, max_iter)
    model.fit(x_train, y_train)
    return FittedPipeline(config, sfreq, filter_order, state, scaler, model)


def transform_subject_for_pipeline(x: np.ndarray, pipeline: FittedPipeline) -> np.ndarray:
    transform = fit_subject_transform(x, pipeline.config.raw_norm, pipeline.config.align)
    x_norm = apply_subject_transform(x, transform)
    feats = transform_features(x_norm, pipeline.state)
    normalizer = fit_subject_feature_normalizer(feats, pipeline.config.feature_norm)
    feats = apply_feature_normalizer(feats, normalizer)
    return pipeline.scaler.transform(feats)


def evaluate_config(
    subjects: Sequence[Subject],
    config: Config,
    sfreq: float,
    filter_order: int,
    seed: int,
    max_iter: int,
) -> EvalRecord:
    x_all, y_all, names = prepare_subjects(subjects, config)
    static_feature_family = config.feature_family in {"logvar_meanabs", "bandpower_logvar", "cov_upper"}
    if static_feature_family:
        static_state = FeatureState(config=config, sfreq=sfreq, filter_order=filter_order)
        static_features = [transform_features(x, static_state) for x in x_all]
    else:
        static_features = None
    per_subject, f1s, accs = [], [], []
    all_true, all_pred, oof_ids = [], [], []
    total_confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)

    for hold_idx, subject_name in enumerate(names):
        train_y = [y for idx, y in enumerate(y_all) if idx != hold_idx]
        if static_feature_family:
            train_features = [f for idx, f in enumerate(static_features) if idx != hold_idx]
            train_norms = [fit_subject_feature_normalizer(feats, config.feature_norm) for feats in train_features]
            train_features = [
                apply_feature_normalizer(feats, norm) for feats, norm in zip(train_features, train_norms)
            ]
            x_train = np.concatenate(train_features, axis=0)
            y_train = np.concatenate(train_y, axis=0)
            scaler = StandardScaler().fit(x_train)
            model = make_model(config, seed, max_iter)
            model.fit(scaler.transform(x_train), y_train)
            va_features = static_features[hold_idx]
            va_norm = fit_subject_feature_normalizer(va_features, config.feature_norm)
            va_features = apply_feature_normalizer(va_features, va_norm)
            pred = np.asarray(model.predict(scaler.transform(va_features)), dtype=np.int64)
        else:
            train_x = [x for idx, x in enumerate(x_all) if idx != hold_idx]
            pipeline = fit_pipeline(train_x, train_y, config, sfreq, filter_order, seed, max_iter)
            va_features = transform_features(x_all[hold_idx], pipeline.state)
            va_norm = fit_subject_feature_normalizer(va_features, config.feature_norm)
            va_features = apply_feature_normalizer(va_features, va_norm)
            va_features = pipeline.scaler.transform(va_features)
            pred = np.asarray(pipeline.model.predict(va_features), dtype=np.int64)
        y_true = y_all[hold_idx]
        f1 = f1_score(y_true, pred, average="macro", labels=LABELS)
        acc = accuracy_score(y_true, pred)
        cm = confusion_matrix(y_true, pred, labels=LABELS)
        total_confusion += cm
        f1s.append(float(f1))
        accs.append(float(acc))
        all_true.append(y_true)
        all_pred.append(pred)
        oof_ids.extend([f"{subject_name}:{i}" for i in range(len(pred))])
        per_subject.append(
            {
                "subject": subject_name,
                "macro_f1": round(float(f1), 6),
                "accuracy": round(float(acc), 6),
                "true_distribution": distribution(y_true),
                "pred_distribution": distribution(pred),
                "confusion": cm.tolist(),
            }
        )
    true = np.concatenate(all_true)
    pred = np.concatenate(all_pred)
    return EvalRecord(
        config=config,
        mean_macro_f1=float(np.mean(f1s)),
        std_macro_f1=float(np.std(f1s)),
        mean_accuracy=float(np.mean(accs)),
        std_accuracy=float(np.std(accs)),
        confusion=total_confusion,
        true_distribution=distribution(true),
        pred_distribution=distribution(pred),
        per_subject=per_subject,
        oof_ids=oof_ids,
        oof_true=true,
        oof_pred=pred,
    )


def make_configs() -> list[Config]:
    configs = [
        Config("baseline_logvar_meanabs_subject_lr", "none", "none", "alpha_beta_broad", "logvar_meanabs", "subject", "logreg", 1.0),
        Config("trialz_logvar_meanabs_subject_lr", "trial_zscore", "none", "alpha_beta_broad", "logvar_meanabs", "subject", "logreg", 1.0),
        Config("channelz_logvar_meanabs_subject_lr", "channel_zscore", "none", "alpha_beta_broad", "logvar_meanabs", "subject", "logreg", 1.0),
        Config("subjectz_logvar_meanabs_subject_lr", "subject_standard", "none", "alpha_beta_broad", "logvar_meanabs", "subject", "logreg", 1.0),
        Config("robust_logvar_meanabs_robust_lr", "robust_subject", "none", "alpha_beta_broad", "logvar_meanabs", "robust_subject", "logreg", 1.0),
        Config("align_logvar_meanabs_subject_lr", "subject_standard", "euclidean", "alpha_beta_broad", "logvar_meanabs", "subject", "logreg", 1.0),
        Config("expanded_bandpower_subject_lr", "none", "none", "expanded_filterbank", "bandpower_logvar", "subject", "logreg", 0.5),
        Config("expanded_bandpower_subject_lda", "none", "none", "expanded_filterbank", "bandpower_logvar", "subject", "lda", None),
        Config("csp_alpha_beta_broad_lda", "channel_zscore", "none", "alpha_beta_broad", "csp", "none", "lda", None, 1),
        Config("csp_alpha_beta_broad_lr", "channel_zscore", "none", "alpha_beta_broad", "csp", "none", "logreg", 0.5, 1),
        Config("cov_subject_lda", "none", "none", "alpha_beta_broad", "cov_upper", "subject", "lda", None),
        Config("cov_robust_lda", "robust_subject", "none", "alpha_beta_broad", "cov_upper", "robust_subject", "lda", None),
        Config("tangent_broad_lda", "none", "none", "broad_only", "tangent", "subject", "lda", None),
        Config("baseline_logvar_meanabs_subject_svc", "none", "none", "alpha_beta_broad", "logvar_meanabs", "subject", "linear_svc", 0.5),
    ]
    return configs


def config_to_dict(config: Config) -> dict[str, Any]:
    return {
        "name": config.name,
        "raw_norm": config.raw_norm,
        "align": config.align,
        "band_set": config.band_set,
        "feature_family": config.feature_family,
        "feature_norm": config.feature_norm,
        "model": config.model,
        "C": config.C,
        "csp_components": config.csp_components,
    }


def write_results(records: Sequence[EvalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "config_name",
        "mean_macro_f1",
        "std_macro_f1",
        "mean_accuracy",
        "std_accuracy",
        "true_distribution",
        "pred_distribution",
        "confusion",
        "config",
        "per_subject",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, record in enumerate(records, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "config_name": record.config.name,
                    "mean_macro_f1": f"{record.mean_macro_f1:.8f}",
                    "std_macro_f1": f"{record.std_macro_f1:.8f}",
                    "mean_accuracy": f"{record.mean_accuracy:.8f}",
                    "std_accuracy": f"{record.std_accuracy:.8f}",
                    "true_distribution": json.dumps(record.true_distribution, sort_keys=True),
                    "pred_distribution": json.dumps(record.pred_distribution, sort_keys=True),
                    "confusion": json.dumps(record.confusion.tolist()),
                    "config": json.dumps(config_to_dict(record.config), sort_keys=True),
                    "per_subject": json.dumps(record.per_subject, sort_keys=True),
                }
            )


def write_oof(records: Sequence[EvalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config_name", "example_id", "true", "pred"])
        for record in records:
            for example_id, y_true, y_pred in zip(record.oof_ids, record.oof_true, record.oof_pred):
                writer.writerow([record.config.name, example_id, int(y_true), int(y_pred)])


def fit_final_pipeline(subjects: Sequence[Subject], config: Config, args: argparse.Namespace) -> FittedPipeline:
    x_all, y_all, _ = prepare_subjects(subjects, config)
    return fit_pipeline(x_all, y_all, config, args.sfreq, args.filter_order, args.seed, args.max_iter)


def predict_test(pipeline: FittedPipeline, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_features = transform_subject_for_pipeline(x_test, pipeline)
    probs = prediction_scores(pipeline.model, x_features)
    return np.argmax(probs, axis=1).astype(np.int64), probs


def choose_diverse_records(records: Sequence[EvalRecord], top_k: int) -> list[EvalRecord]:
    selected = []
    seen_families = set()
    for record in records:
        family_key = (record.config.feature_family, record.config.model)
        if family_key in seen_families and len(selected) >= 2:
            continue
        selected.append(record)
        seen_families.add(family_key)
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        for record in records:
            if record not in selected:
                selected.append(record)
            if len(selected) >= top_k:
                break
    return selected


def quota_from_distribution(dist: dict[int, int], total: int) -> dict[int, int]:
    observed_total = sum(dist.values())
    raw = {label: (dist[label] / observed_total) * total for label in LABELS}
    quota = {label: int(np.floor(raw[label])) for label in LABELS}
    remaining = total - sum(quota.values())
    for label in sorted(LABELS, key=lambda lab: raw[lab] - quota[lab], reverse=True)[:remaining]:
        quota[label] += 1
    return quota


def quota_assign(probs: np.ndarray, quota: dict[int, int]) -> np.ndarray:
    if sum(quota.values()) != len(probs):
        raise ValueError(f"Quota must sum to {len(probs)}, got {quota}")
    slots = np.asarray([label for label in LABELS for _ in range(quota[label])], dtype=np.int64)
    objective = np.log(np.clip(probs[:, slots], EPS, 1.0))
    rows, cols = linear_sum_assignment(-objective)
    labels = np.empty(len(probs), dtype=np.int64)
    labels[rows] = slots[cols]
    return labels


def supported_class2_reduced_quota(records: Sequence[EvalRecord], base_labels: np.ndarray) -> dict[int, int] | None:
    top = records[0]
    true_dist = top.true_distribution
    pred_dist = top.pred_distribution
    base_dist = distribution(base_labels)
    if base_dist[2] <= 8 and pred_dist[2] <= true_dist[2]:
        return None
    target = {label: 8 for label in LABELS}
    if base_dist[2] > 8:
        target[2] = max(6, min(8, base_dist[2] - 2))
    deficits = {label: max(0, true_dist[label] - pred_dist[label]) for label in LABELS if label != 2}
    if sum(deficits.values()) == 0:
        deficits = {0: 1, 1: 1, 3: 1}
    while sum(target.values()) < len(base_labels):
        label = max(deficits, key=lambda lab: (deficits[lab], -target[lab]))
        target[label] += 1
        deficits[label] = max(0, deficits[label] - 1)
    while sum(target.values()) > len(base_labels):
        label = max([0, 1, 3], key=lambda lab: target[lab])
        target[label] -= 1
    return target


def validate_submission(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "label"]:
            raise ValueError(f"{path}: expected header id,label, got {reader.fieldnames}")
        rows = [(int(row["id"]), int(row["label"])) for row in reader]
    if len(rows) != len(expected_ids):
        raise ValueError(f"{path}: expected {len(expected_ids)} rows, got {len(rows)}")
    ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    labels = np.asarray([row[1] for row in rows], dtype=np.int64)
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"{path}: id order does not match test.npz")
    if np.any((labels < 0) | (labels > 3)):
        raise ValueError(f"{path}: labels must be in {{0,1,2,3}}")
    return labels


def write_submission(path: Path, ids: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows((int(sample_id), int(label)) for sample_id, label in zip(ids, labels))
    validate_submission(path, ids)


def load_existing_submission(path: Path, expected_ids: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        return None
    return validate_submission(path, expected_ids)


def changed_rows(ids: np.ndarray, base: np.ndarray, labels: np.ndarray) -> list[str]:
    return [
        f"id {int(ids[i])}: {int(old)}->{int(new)}"
        for i, (old, new) in enumerate(zip(base, labels))
        if int(old) != int(new)
    ]


def print_candidate_summary(
    name: str,
    path: Path,
    ids: np.ndarray,
    labels: np.ndarray,
    references: dict[str, np.ndarray],
) -> None:
    print(f"\n{name}: {path}")
    print(f"  distribution: {distribution(labels)}")
    identical = [ref_name for ref_name, ref_labels in references.items() if np.array_equal(labels, ref_labels)]
    print(f"  identical to previous: {', '.join(identical) if identical else 'none'}")
    for ref_name, ref_labels in references.items():
        changes = changed_rows(ids, ref_labels, labels)
        print(f"  changed rows vs {ref_name} ({len(changes)}): {', '.join(changes) if changes else 'none'}")


def save_checkpoint(path: Path, pipeline: FittedPipeline, eval_record: EvalRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": "task2-cross-subject-optimize-v1",
        "note": "Generated by cross_subject_optimize.py and supported by inference.py.",
        "config": config_to_dict(pipeline.config),
        "sfreq": pipeline.sfreq,
        "filter_order": pipeline.filter_order,
        "feature_state": {
            "csp_filters": pipeline.state.csp_filters,
            "tangent_refs": pipeline.state.tangent_refs,
        },
        "scaler": pipeline.scaler,
        "model": pipeline.model,
        "loso_summary": {
            "mean_macro_f1": eval_record.mean_macro_f1,
            "std_macro_f1": eval_record.std_macro_f1,
            "mean_accuracy": eval_record.mean_accuracy,
            "std_accuracy": eval_record.std_accuracy,
            "confusion": eval_record.confusion.tolist(),
            "pred_distribution": eval_record.pred_distribution,
            "true_distribution": eval_record.true_distribution,
        },
    }
    joblib.dump(checkpoint, path)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    subjects = load_subjects(args.train_data)
    x_test, test_ids = load_test_data(args.test_data)
    print(f"Loaded {len(subjects)} training subjects and {len(test_ids)} test trials.")
    print(f"Training label distribution: {distribution(np.concatenate([s.y for s in subjects]))}")

    records = []
    configs = make_configs()
    for idx, config in enumerate(configs, start=1):
        record = evaluate_config(subjects, config, args.sfreq, args.filter_order, args.seed, args.max_iter)
        records.append(record)
        print(
            f"[{idx:02d}/{len(configs)}] {config.name:38s} "
            f"macro_f1={record.mean_macro_f1:.4f}+/-{record.std_macro_f1:.4f} "
            f"acc={record.mean_accuracy:.4f}+/-{record.std_accuracy:.4f} "
            f"pred_dist={record.pred_distribution}",
            flush=True,
        )
    records.sort(key=lambda rec: (rec.mean_macro_f1, -rec.std_macro_f1, rec.mean_accuracy), reverse=True)
    write_results(records, args.results_csv)
    write_oof(records, args.oof_csv)

    print("\n=== Top LOSO configs ===")
    for rank, record in enumerate(records[:8], start=1):
        print(
            f"{rank}. {record.config.name}: macro_f1={record.mean_macro_f1:.4f}+/-{record.std_macro_f1:.4f}, "
            f"acc={record.mean_accuracy:.4f}+/-{record.std_accuracy:.4f}, pred_dist={record.pred_distribution}"
        )
        print(f"   confusion={record.confusion.tolist()}")

    best_record = records[0]
    best_pipeline = fit_final_pipeline(subjects, best_record.config, args)
    single_labels, single_probs = predict_test(best_pipeline, x_test)
    save_checkpoint(args.checkpoint, best_pipeline, best_record)

    ensemble_records = choose_diverse_records(records, args.top_k_ensemble)
    ensemble_probs = np.zeros_like(single_probs)
    print("\nEnsemble members:")
    for record in ensemble_records:
        pipeline = fit_final_pipeline(subjects, record.config, args)
        _, probs = predict_test(pipeline, x_test)
        weight = max(record.mean_macro_f1, EPS)
        ensemble_probs += weight * probs
        print(f"  {record.config.name}: weight={weight:.4f}")
    ensemble_probs /= np.maximum(ensemble_probs.sum(axis=1, keepdims=True), EPS)
    ensemble_labels = np.argmax(ensemble_probs, axis=1).astype(np.int64)

    conservative_quota = {label: len(test_ids) // len(LABELS) for label in LABELS}
    conservative_labels = quota_assign(ensemble_probs, conservative_quota)
    reduced_quota = supported_class2_reduced_quota(records, ensemble_labels)
    reduced_labels = quota_assign(ensemble_probs, reduced_quota) if reduced_quota is not None else None

    candidates: list[tuple[str, Path, np.ndarray]] = [
        ("strongest_loso_single", args.output_dir / "submission_task2_cross_loso_single.csv", single_labels),
        ("diverse_ensemble", args.output_dir / "submission_task2_cross_diverse_ensemble.csv", ensemble_labels),
        ("conservative_calibrated", args.output_dir / "submission_task2_cross_conservative_calibrated.csv", conservative_labels),
    ]
    if reduced_labels is not None:
        print(f"Class-2-reduced quota supported by LOSO/test overprediction signal: {reduced_quota}")
        candidates.append(("class2_reduced_calibrated", args.output_dir / "submission_task2_cross_class2_reduced.csv", reduced_labels))
    else:
        print("Class-2-reduced calibrated candidate skipped: not supported by LOSO/test overprediction signal.")

    reference_paths = [
        Path("submission_task2_best_loso.csv"),
        Path("submission_task2_prior_public_adjusted.csv"),
        Path("submission_task2_prior_class2_less.csv"),
    ]
    references = {
        path.name: labels
        for path in reference_paths
        if (labels := load_existing_submission(path, test_ids)) is not None
    }

    print("\n=== Generated candidates ===")
    for name, path, labels in candidates:
        write_submission(path, test_ids, labels)
        print_candidate_summary(name, path, test_ids, labels, references)

    print("\nSuggested upload order:")
    order = [
        "submission_task2_cross_diverse_ensemble.csv",
        "submission_task2_cross_loso_single.csv",
        "submission_task2_cross_class2_reduced.csv",
        "submission_task2_cross_conservative_calibrated.csv",
    ]
    available = {path.name for _, path, _ in candidates}
    for idx, file_name in enumerate([name for name in order if name in available], start=1):
        print(f"  {idx}. {file_name}")
    print(f"\nWrote LOSO table to {args.results_csv}")
    print(f"Wrote OOF predictions to {args.oof_csv}")
    print(f"Wrote best single-model checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
