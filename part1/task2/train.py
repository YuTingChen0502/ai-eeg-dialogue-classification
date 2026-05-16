"""Task 2 high-score cross-subject EEG training.

The stable CLI is preserved, but --validate now performs true
leave-one-subject-out (LOSO) model selection. The saved checkpoint contains the
full feature/normalization/classifier configuration needed by inference.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

CHECKPOINT_VERSION = "task2-highscore-v2"
EPS = 1e-10

BAND_LIBRARY: dict[str, Tuple[float, float]] = {
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "broad": (4.0, 40.0),
    "gamma": (70.0, 124.875),
}
BAND_SETS: dict[str, Tuple[str, ...]] = {
    "alpha": ("alpha",),
    "beta": ("beta",),
    "broad": ("broad",),
    "gamma": ("gamma",),
    "alpha_beta": ("alpha", "beta"),
    "alpha_beta_broad": ("alpha", "beta", "broad"),
    "alpha_beta_broad_gamma": ("alpha", "beta", "broad", "gamma"),
    "broad_gamma": ("broad", "gamma"),
}


@dataclass(frozen=True)
class Config:
    name: str
    band_set: str
    feature_set: str
    raw_norm: str
    feature_norm: str
    classifier: str
    C: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task 2 cross-subject EEG classifier.")
    parser.add_argument("--data", type=Path, required=True, help="Directory containing subjectXX.npz files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2_train"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0, help="Used by the legacy default config.")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--validate", action="store_true", help="Run true leave-one-subject-out validation/search.")
    parser.add_argument(
        "--config-name",
        default=None,
        help="Optional exact config name to train. If omitted with --validate, the best LOSO config is used.",
    )
    return parser.parse_args()


def load_subjects(data_dir: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    files = sorted(data_dir.glob("subject*.npz"))
    if not files:
        raise FileNotFoundError(f"No subjectXX.npz files found in {data_dir}")
    subjects = []
    for f in files:
        arr = np.load(f, allow_pickle=True)
        if "x" not in arr or "y" not in arr:
            raise KeyError(f"{f} is missing required 'x' or 'y' arrays")
        x = np.asarray(arr["x"], dtype=np.float64)
        y = np.asarray(arr["y"], dtype=np.int64)
        if x.ndim != 3:
            raise ValueError(f"{f}: expected x with shape (N, C, T), got {x.shape}")
        if len(x) != len(y):
            raise ValueError(f"{f}: x has {len(x)} trials but y has {len(y)} labels")
        subjects.append((f.stem, x, y))
    return subjects


def effective_band(low: float, high: float, sfreq: float) -> Tuple[float, float]:
    nyq = sfreq / 2.0
    return float(low), float(min(high, nyq * 0.999))


def bands_for_config(config: Config, sfreq: float) -> List[Tuple[str, float, float, float, float]]:
    out = []
    for name in BAND_SETS[config.band_set]:
        nominal_low, nominal_high = BAND_LIBRARY[name]
        low, high = effective_band(nominal_low, nominal_high, sfreq)
        out.append((name, nominal_low, nominal_high, low, high))
    return out


def apply_raw_norm(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if mode == "none":
        return x
    if mode == "trial_zscore":
        mu = x.mean(axis=-1, keepdims=True)
        sigma = x.std(axis=-1, keepdims=True)
        sigma = np.where(sigma < EPS, 1.0, sigma)
        return (x - mu) / sigma
    raise ValueError(f"Unknown raw normalization: {mode}")


def bandpass(x: np.ndarray, low: float, high: float, sfreq: float, order: int) -> np.ndarray:
    nyq = sfreq / 2.0
    wn = (low / nyq, high / nyq)
    if not (0.0 < wn[0] < wn[1] < 1.0):
        raise ValueError(f"Invalid band cutoffs {low}-{high} Hz for sfreq={sfreq}")
    sos = butter(order, wn, btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def covariance_upper_features(filtered: np.ndarray) -> np.ndarray:
    covs = []
    iu = np.triu_indices(filtered.shape[1])
    for trial in filtered:
        centered = trial - trial.mean(axis=-1, keepdims=True)
        cov = centered @ centered.T / max(1, centered.shape[-1] - 1)
        trace = np.trace(cov)
        if trace > EPS:
            cov = cov / trace
        covs.append(cov[iu])
    return np.asarray(covs, dtype=np.float64)


def band_features(filtered: np.ndarray, feature_set: str) -> np.ndarray:
    var = np.maximum(np.var(filtered, axis=-1), EPS)
    parts = [np.log(var)]
    if feature_set == "logvar":
        return np.concatenate(parts, axis=1)
    if feature_set == "logvar_meanabs":
        parts.append(np.mean(np.abs(filtered), axis=-1))
    elif feature_set == "logvar_bandpower":
        parts.append(np.mean(filtered**2, axis=-1))
    elif feature_set == "logvar_meanstd":
        parts.append(np.mean(filtered, axis=-1))
        parts.append(np.std(filtered, axis=-1))
    elif feature_set == "cov_upper":
        return covariance_upper_features(filtered)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return np.concatenate(parts, axis=1).astype(np.float64)


def extract_features(x: np.ndarray, config: Config, sfreq: float, filter_order: int) -> np.ndarray:
    x = apply_raw_norm(x, config.raw_norm)
    parts = []
    for _, _, _, low, high in bands_for_config(config, sfreq):
        filtered = bandpass(x, low, high, sfreq, filter_order)
        parts.append(band_features(filtered, config.feature_set))
    return np.concatenate(parts, axis=1).astype(np.float64)


def fit_subject_feature_normalizers(
    features_by_subject: Sequence[np.ndarray], mode: str
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    means: list[np.ndarray | None] = []
    scales: list[np.ndarray | None] = []
    if mode == "none":
        return [None for _ in features_by_subject], [None for _ in features_by_subject]
    if mode != "subject":
        raise ValueError(f"Unknown feature normalization: {mode}")
    for feats in features_by_subject:
        mu = feats.mean(axis=0)
        sigma = feats.std(axis=0)
        sigma = np.where(sigma < EPS, 1.0, sigma)
        means.append(mu)
        scales.append(sigma)
    return means, scales


def apply_subject_feature_normalization(
    features_by_subject: Sequence[np.ndarray], mode: str
) -> List[np.ndarray]:
    means, scales = fit_subject_feature_normalizers(features_by_subject, mode)
    out = []
    for feats, mu, sigma in zip(features_by_subject, means, scales):
        out.append(feats if mu is None else (feats - mu) / sigma)
    return out


def make_classifier(config: Config, seed: int, max_iter: int):
    if config.classifier == "logreg":
        return LogisticRegression(
            C=float(config.C),
            class_weight="balanced",
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed,
        )
    if config.classifier == "svc":
        return LinearSVC(C=float(config.C), class_weight="balanced", max_iter=max_iter * 5, random_state=seed)
    if config.classifier == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if config.classifier == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if config.classifier == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier: {config.classifier}")


def make_search_configs(legacy_C: float) -> List[Config]:
    configs: list[Config] = []
    for band_set in [
        "alpha",
        "beta",
        "broad",
        "gamma",
        "alpha_beta",
        "alpha_beta_broad",
        "alpha_beta_broad_gamma",
        "broad_gamma",
    ]:
        configs.append(Config(f"{band_set}_logvar_lr_C1", band_set, "logvar", "none", "none", "logreg", 1.0))
        configs.append(Config(f"{band_set}_logvar_meanabs_lr_C1", band_set, "logvar_meanabs", "none", "none", "logreg", 1.0))
        configs.append(Config(f"{band_set}_logvar_meanabs_svc_C1", band_set, "logvar_meanabs", "none", "none", "svc", 1.0))

    for C in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        configs.append(Config(f"alpha_beta_broad_logvar_lr_C{C:g}", "alpha_beta_broad", "logvar", "none", "none", "logreg", C))
        configs.append(Config(f"alpha_beta_broad_logvar_svc_C{C:g}", "alpha_beta_broad", "logvar", "none", "none", "svc", C))

    configs.extend(
        [
            Config("alpha_beta_broad_logvar_bandpower_lr_C1", "alpha_beta_broad", "logvar_bandpower", "none", "none", "logreg", 1.0),
            Config("alpha_beta_broad_logvar_meanstd_lr_C1", "alpha_beta_broad", "logvar_meanstd", "none", "none", "logreg", 1.0),
            Config("alpha_beta_broad_gamma_logvar_meanabs_lr_C05", "alpha_beta_broad_gamma", "logvar_meanabs", "none", "none", "logreg", 0.5),
            Config("alpha_beta_broad_gamma_logvar_meanabs_svc_C05", "alpha_beta_broad_gamma", "logvar_meanabs", "none", "none", "svc", 0.5),
            Config("alpha_beta_broad_logvar_trialz_lr_C1", "alpha_beta_broad", "logvar", "trial_zscore", "none", "logreg", 1.0),
            Config("alpha_beta_broad_logvar_meanabs_trialz_lr_C1", "alpha_beta_broad", "logvar_meanabs", "trial_zscore", "none", "logreg", 1.0),
            Config("alpha_beta_broad_logvar_subjectnorm_lr_C1", "alpha_beta_broad", "logvar", "none", "subject", "logreg", 1.0),
            Config("alpha_beta_broad_logvar_meanabs_subjectnorm_lr_C1", "alpha_beta_broad", "logvar_meanabs", "none", "subject", "logreg", 1.0),
            Config("alpha_beta_broad_logvar_lda", "alpha_beta_broad", "logvar", "none", "none", "lda", None),
            Config("alpha_beta_broad_logvar_meanabs_lda", "alpha_beta_broad", "logvar_meanabs", "none", "none", "lda", None),
            Config("alpha_beta_broad_cov_upper_lr_C025", "alpha_beta_broad", "cov_upper", "none", "none", "logreg", 0.25),
            Config("alpha_beta_broad_cov_upper_svc_C025", "alpha_beta_broad", "cov_upper", "none", "none", "svc", 0.25),
            Config("broad_gamma_cov_upper_lr_C025", "broad_gamma", "cov_upper", "none", "none", "logreg", 0.25),
            Config("alpha_beta_broad_logvar_extra_trees", "alpha_beta_broad", "logvar", "none", "none", "extra_trees", None),
            Config("alpha_beta_broad_logvar_meanabs_extra_trees", "alpha_beta_broad", "logvar_meanabs", "none", "none", "extra_trees", None),
            Config("alpha_beta_broad_logvar_random_forest", "alpha_beta_broad", "logvar", "none", "none", "random_forest", None),
            Config("legacy_cli_C", "alpha_beta_broad", "logvar", "none", "none", "logreg", legacy_C),
        ]
    )
    seen = set()
    unique = []
    for cfg in configs:
        if cfg.name not in seen:
            unique.append(cfg)
            seen.add(cfg.name)
    return unique


def config_to_dict(config: Config) -> dict[str, Any]:
    return {
        "name": config.name,
        "band_set": config.band_set,
        "feature_set": config.feature_set,
        "raw_norm": config.raw_norm,
        "feature_norm": config.feature_norm,
        "classifier": config.classifier,
        "C": config.C,
    }


def config_from_dict(d: dict[str, Any]) -> Config:
    return Config(
        name=d["name"],
        band_set=d["band_set"],
        feature_set=d["feature_set"],
        raw_norm=d["raw_norm"],
        feature_norm=d["feature_norm"],
        classifier=d["classifier"],
        C=d.get("C"),
    )


def prepare_train_matrix(
    subject_feature_list: Sequence[np.ndarray], y_by_subject: Sequence[np.ndarray], feature_norm: str
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    normalized_subjects = apply_subject_feature_normalization(subject_feature_list, feature_norm)
    x = np.concatenate(normalized_subjects, axis=0)
    y = np.concatenate(y_by_subject, axis=0)
    scaler = StandardScaler().fit(x)
    return scaler.transform(x), y, scaler


def prepare_eval_matrix(features: np.ndarray, feature_norm: str, scaler: StandardScaler) -> np.ndarray:
    if feature_norm == "subject":
        normalized = apply_subject_feature_normalization([features], "subject")[0]
    elif feature_norm == "none":
        normalized = features
    else:
        raise ValueError(f"Unknown feature normalization: {feature_norm}")
    return scaler.transform(normalized)


def evaluate_config(
    subjects: List[Tuple[str, np.ndarray, np.ndarray]], config: Config, args: argparse.Namespace
) -> dict[str, Any]:
    subject_features = [
        extract_features(x, config, args.sfreq, args.filter_order)
        for _, x, _ in subjects
    ]
    y_by_subject = [y for _, _, y in subjects]
    accs, f1s, per_subject = [], [], []
    for i, (subject_name, _, y_va) in enumerate(subjects):
        train_features = [f for j, f in enumerate(subject_features) if j != i]
        train_labels = [y for j, y in enumerate(y_by_subject) if j != i]
        x_tr, y_tr, scaler = prepare_train_matrix(train_features, train_labels, config.feature_norm)
        x_va = prepare_eval_matrix(subject_features[i], config.feature_norm, scaler)
        clf = make_classifier(config, args.seed, args.max_iter)
        clf.fit(x_tr, y_tr)
        pred = clf.predict(x_va)
        acc = accuracy_score(y_va, pred)
        f1 = f1_score(y_va, pred, average="macro")
        accs.append(acc)
        f1s.append(f1)
        per_subject.append({"subject": subject_name, "macro_f1": round(float(f1), 4), "accuracy": round(float(acc), 4)})
    return {
        "config_name": config.name,
        "bands": config.band_set,
        "feature_set": config.feature_set,
        "normalization": f"raw={config.raw_norm}, feature={config.feature_norm}",
        "classifier": config.classifier if config.C is None else f"{config.classifier}(C={config.C:g})",
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s)),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "per_subject_scores": per_subject,
    }


def loso_search(subjects: List[Tuple[str, np.ndarray, np.ndarray]], args: argparse.Namespace, configs: Sequence[Config]):
    records = []
    for idx, config in enumerate(configs, start=1):
        rec = evaluate_config(subjects, config, args)
        records.append(rec)
        print(
            f"[{idx:02d}/{len(configs)}] {config.name:48s} "
            f"macro_f1={rec['mean_macro_f1']:.4f}+/-{rec['std_macro_f1']:.4f} "
            f"acc={rec['mean_accuracy']:.4f}+/-{rec['std_accuracy']:.4f}"
        )
        print("  per-subject:", rec["per_subject_scores"])
    df = pd.DataFrame(records).sort_values(
        ["mean_macro_f1", "std_macro_f1", "mean_accuracy"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    print("\n=== LOSO leaderboard sorted by mean Macro-F1 ===")
    print(df.to_string(index=False))
    return df


def fit_final(subjects: List[Tuple[str, np.ndarray, np.ndarray]], config: Config, args: argparse.Namespace):
    features_by_subject = [extract_features(x, config, args.sfreq, args.filter_order) for _, x, _ in subjects]
    y_by_subject = [y for _, _, y in subjects]
    x_train, y_train, scaler = prepare_train_matrix(features_by_subject, y_by_subject, config.feature_norm)
    clf = make_classifier(config, args.seed, args.max_iter)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_train)
    return scaler, clf, x_train, y_train, pred


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or (args.output_dir / "task2_model.pkl")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    subjects = load_subjects(args.data)
    n_trials = sum(len(s[1]) for s in subjects)
    n_channels = subjects[0][1].shape[1]
    n_time = subjects[0][1].shape[2]
    print(f"Loaded {len(subjects)} subjects, {n_trials} trials total, shape per trial=({n_channels}, {n_time})")

    configs = make_search_configs(args.C)
    config_map = {cfg.name: cfg for cfg in configs}
    if args.config_name:
        if args.config_name not in config_map:
            raise KeyError(f"Unknown --config-name {args.config_name}. Available: {sorted(config_map)}")
        selected = config_map[args.config_name]
        leaderboard = None
    elif args.validate:
        print(f"Running true leave-one-subject-out search over {len(configs)} configs...")
        leaderboard = loso_search(subjects, args, configs)
        selected = config_map[leaderboard.iloc[0]["config_name"]]
    else:
        selected = Config("legacy_cli_default", "alpha_beta_broad", "logvar", "none", "none", "logreg", args.C)
        leaderboard = None

    print(f"\nSelected final config: {selected}")
    scaler, clf, x_all, y_all, train_pred = fit_final(subjects, selected, args)
    print(
        f"Final train sanity: acc={accuracy_score(y_all, train_pred):.4f} "
        f"macro_f1={f1_score(y_all, train_pred, average='macro'):.4f}"
    )

    band_meta = bands_for_config(selected, args.sfreq)
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "feature_extractor_config": config_to_dict(selected),
        "feature_type": selected.feature_set,
        "bands": [(low, high) for _, _, _, low, high in band_meta],
        "band_metadata": [
            {
                "name": name,
                "nominal_low": nominal_low,
                "nominal_high": nominal_high,
                "actual_low": low,
                "actual_high": high,
            }
            for name, nominal_low, nominal_high, low, high in band_meta
        ],
        "sfreq": float(args.sfreq),
        "filter_order": int(args.filter_order),
        "normalization_config": {
            "raw_norm": selected.raw_norm,
            "feature_norm": selected.feature_norm,
            "standard_scaler": True,
        },
        "scaler": scaler,
        "model": clf,
        "classifier": selected.classifier,
        "n_channels": int(n_channels),
        "n_time_points": int(n_time),
        "classes": np.unique(y_all).astype(np.int64).tolist(),
        "train_args": {
            "seed": int(args.seed),
            "max_iter": int(args.max_iter),
            "data_dir": str(args.data),
            "selected_by_validate": bool(args.validate and leaderboard is not None),
        },
        "loso_leaderboard": None if leaderboard is None else leaderboard.to_dict(orient="records"),
    }
    joblib.dump(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
