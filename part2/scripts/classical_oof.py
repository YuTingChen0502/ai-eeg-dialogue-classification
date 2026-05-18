from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from part2_utils import ROOT, assert_finite, labels_from_probs, load_data, tune_threshold_from_values, write_json


CONFIGS = [
    {
        "name": "word_tfidf_linearsvc",
        "vectorizer": TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
        "model": LinearSVC(C=1.0, class_weight=None, random_state=42),
    },
    {
        "name": "char_wb_tfidf_linearsvc",
        "vectorizer": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        "model": LinearSVC(C=1.0, class_weight=None, random_state=42),
    },
]


def main() -> None:
    start = time.time()
    train_df, test_df = load_data()
    y = train_df["label"].values.astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_features = np.zeros((len(train_df), len(CONFIGS)), dtype=np.float32)
    test_features = np.zeros((len(test_df), len(CONFIGS)), dtype=np.float32)
    feature_names = [cfg["name"] for cfg in CONFIGS]
    config_meta: list[dict] = []

    for feature_idx, cfg in enumerate(CONFIGS):
        print(f"Running {cfg['name']}")
        test_fold_scores: list[np.ndarray] = []
        fold_scores: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_df["text"], y)):
            pipe = Pipeline([("tfidf", cfg["vectorizer"]), ("svc", cfg["model"])])
            pipe.fit(train_df.iloc[tr_idx]["text"], y[tr_idx])
            valid_score = pipe.decision_function(train_df.iloc[va_idx]["text"])
            test_score = pipe.decision_function(test_df["text"])
            train_features[va_idx, feature_idx] = valid_score.astype(np.float32)
            test_fold_scores.append(test_score.astype(np.float32))
            threshold, fold_f1 = tune_threshold_from_values(y[va_idx], valid_score)
            fold_scores.append(float(fold_f1))
            print(f"  fold {fold_idx + 1}: tuned threshold={threshold:.4f} Macro-F1={fold_f1:.4f}")
        test_features[:, feature_idx] = np.mean(np.vstack(test_fold_scores), axis=0)
        threshold, tuned_f1 = tune_threshold_from_values(y, train_features[:, feature_idx])
        default_f1 = f1_score(y, (train_features[:, feature_idx] >= 0).astype(int), average="macro")
        config_meta.append(
            {
                "name": cfg["name"],
                "fold_macro_f1_tuned": fold_scores,
                "oof_macro_f1_default_threshold_zero": float(default_f1),
                "oof_threshold": float(threshold),
                "oof_macro_f1_tuned": float(tuned_f1),
                "prediction_distribution_at_threshold": pd.Series(
                    (train_features[:, feature_idx] >= threshold).astype(int)
                )
                .value_counts()
                .sort_index()
                .astype(int)
                .to_dict(),
            }
        )
        print(f"{cfg['name']} OOF Macro-F1@0={default_f1:.4f} tuned={tuned_f1:.4f}")

    avg_score = train_features.mean(axis=1)
    avg_threshold, avg_tuned_f1 = tune_threshold_from_values(y, avg_score)
    avg_default_f1 = f1_score(y, (avg_score >= 0).astype(int), average="macro")
    assert_finite("classical_train", train_features)
    assert_finite("classical_test", test_features)

    out_dir = ROOT / "outputs" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "classical_train.npy", train_features)
    np.save(out_dir / "classical_test.npy", test_features)
    write_json(out_dir / "classical_feature_names.json", {"feature_names": feature_names})
    meta = {
        "shape_train": list(train_features.shape),
        "shape_test": list(test_features.shape),
        "feature_names": feature_names,
        "models": config_meta,
        "average_score_default_macro_f1": float(avg_default_f1),
        "average_score_threshold": float(avg_threshold),
        "average_score_tuned_macro_f1": float(avg_tuned_f1),
        "runtime_seconds": float(time.time() - start),
        "validation": "5-fold StratifiedKFold on train only; test scores averaged across folds.",
    }
    write_json(out_dir / "classical_meta.json", meta)
    print(f"classical_train shape={train_features.shape}")
    print(f"classical_test shape={test_features.shape}")
    print(f"classical average tuned Macro-F1={avg_tuned_f1:.4f} threshold={avg_threshold:.4f}")


if __name__ == "__main__":
    main()
