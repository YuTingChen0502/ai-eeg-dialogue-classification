"""Run auditable HW3 Part 2 candidate experiments.

The notebook remains the submission artifact. This helper is intentionally
portable back into the notebook: it uses the same train/test CSVs, Macro-F1,
Stratified K-fold validation, Random Over-sampling, and cost-sensitive variants.
Optional frozen DistilBERT features use only locally cached model files. These
are diagnostic comparisons only; the final selected Part 2 model is tuned
RoBERTa-base seed 42 with Kaggle public score 0.9686.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import resample


TEXT_COL_CANDIDATES = ("text", "content", "utterance", "sentence")
REFERENCE_NAMES = (
    "submission_bert_threshold.csv",
    "submission_bert.csv",
    "submission_classical_best_single.csv",
)


@dataclass(frozen=True)
class TextConfig:
    name: str
    model_kind: str
    c_value: float
    class_weight: str | None
    balance: str
    use_word: bool = True
    word_range: tuple[int, int] = (1, 3)
    use_char: bool = True
    char_range: tuple[int, int] = (3, 6)
    use_endpoint: bool = True


@dataclass(frozen=True)
class EmbeddingConfig:
    name: str
    model_kind: str
    c_value: float
    class_weight: str | None


class EndpointFeatureExtractor(BaseEstimator, TransformerMixin):
    """Small numeric features for endpoint/disfluency cues."""

    trailing_incomplete = {
        "a",
        "about",
        "actually",
        "also",
        "an",
        "and",
        "are",
        "as",
        "at",
        "because",
        "but",
        "by",
        "for",
        "from",
        "if",
        "in",
        "is",
        "like",
        "of",
        "on",
        "or",
        "so",
        "than",
        "that",
        "the",
        "then",
        "to",
        "uh",
        "um",
        "wait",
        "well",
        "when",
        "while",
        "with",
    }
    request_prefixes = (
        "can you",
        "could you",
        "would you",
        "please",
        "what is",
        "what are",
        "how do",
        "how can",
        "tell me",
        "show me",
        "i need",
        "i want",
    )

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        rows = [self._one(str(text)) for text in X]
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float32))

    def _one(self, text: str) -> list[float]:
        raw = text if isinstance(text, str) else ""
        stripped = raw.strip()
        lower = stripped.lower()
        tokens = re.findall(r"[a-z0-9']+", lower)
        last = tokens[-1] if tokens else ""
        char_len = len(stripped)
        word_count = len(tokens)
        comma_count = stripped.count(",")
        punct_count = sum(stripped.count(ch) for ch in ".?!;:")
        filler_count = sum(tok in {"um", "uh", "erm", "hmm", "wait", "actually"} for tok in tokens)
        conjunction_count = sum(tok in {"and", "or", "but", "because", "so"} for tok in tokens)
        return [
            np.log1p(char_len),
            np.log1p(word_count),
            np.log1p(comma_count),
            np.log1p(punct_count),
            float(stripped.endswith(".")),
            float(stripped.endswith("?")),
            float(stripped.endswith("!")),
            float(stripped.endswith(",")),
            float(stripped.endswith("-") or stripped.endswith("...")),
            float(last in self.trailing_incomplete),
            float(any(lower.startswith(prefix) for prefix in self.request_prefixes)),
            float("?" in stripped),
            float("," in stripped),
            float(filler_count),
            float(conjunction_count),
            float(word_count <= 4),
            float(char_len <= 24),
            float(word_count > 0 and char_len / max(word_count, 1) > 6.5),
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 2 candidate experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--write-candidates", action="store_true")
    parser.add_argument("--frozen-distilbert", action="store_true")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def find_csv(data_dir: Path, name: str) -> Path:
    candidates = [
        data_dir / name,
        Path.cwd() / name,
        Path.cwd() / "part2" / name,
        Path(__file__).resolve().parent / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not locate {name}. Tried: {[str(p) for p in candidates]}")


def standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    text_col = next((col for col in TEXT_COL_CANDIDATES if col in df.columns), None)
    if text_col is None:
        raise KeyError(f"No text column found. Columns: {df.columns.tolist()}")
    df = df.copy()
    if text_col != "text":
        df = df.rename(columns={text_col: "text"})
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    df["text"] = df["text"].fillna("").astype(str)
    return df


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = find_csv(data_dir, "train.csv")
    test_path = find_csv(data_dir, "test.csv")
    train_df = standardize_frame(pd.read_csv(train_path))
    test_df = standardize_frame(pd.read_csv(test_path))
    train_df["label"] = train_df["label"].astype(int)
    print(f"train: {train_path.resolve()} shape={train_df.shape}")
    print(f"test : {test_path.resolve()} shape={test_df.shape}")
    print(f"label distribution: {train_df['label'].value_counts().sort_index().to_dict()}")
    print(f"test id unique: {test_df['id'].nunique()} / {len(test_df)}")
    return train_df, test_df


def oversample_training(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = df["label"].value_counts()
    majority_label = counts.idxmax()
    minority_label = counts.idxmin()
    majority = df[df["label"] == majority_label]
    minority = df[df["label"] == minority_label]
    minority_up = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=seed,
    )
    return (
        pd.concat([majority, minority_up], axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def make_text_pipeline(cfg: TextConfig) -> Pipeline:
    features = []
    if cfg.use_word:
        features.append(
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=cfg.word_range,
                    min_df=2,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            )
        )
    if cfg.use_char:
        features.append(
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=cfg.char_range,
                    min_df=2,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            )
        )
    if cfg.use_endpoint:
        features.append(
            (
                "endpoint",
                Pipeline(
                    [
                        ("features", EndpointFeatureExtractor()),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
            )
        )
    if not features:
        raise ValueError("At least one feature branch is required.")

    if cfg.model_kind == "svc":
        clf = LinearSVC(
            C=cfg.c_value,
            class_weight=cfg.class_weight,
            random_state=42,
            max_iter=12000,
        )
    elif cfg.model_kind == "lr":
        clf = LogisticRegression(
            C=cfg.c_value,
            class_weight=cfg.class_weight,
            random_state=42,
            max_iter=3000,
            solver="liblinear",
        )
    else:
        raise ValueError(f"Unknown model kind: {cfg.model_kind}")
    return Pipeline([("features", FeatureUnion(features)), ("clf", clf)])


def make_embedding_pipeline(cfg: EmbeddingConfig) -> Pipeline:
    if cfg.model_kind == "svc":
        clf = LinearSVC(
            C=cfg.c_value,
            class_weight=cfg.class_weight,
            random_state=42,
            max_iter=12000,
        )
    elif cfg.model_kind == "lr":
        clf = LogisticRegression(
            C=cfg.c_value,
            class_weight=cfg.class_weight,
            random_state=42,
            max_iter=3000,
            solver="liblinear",
        )
    else:
        raise ValueError(f"Unknown model kind: {cfg.model_kind}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def model_scores(model, X) -> np.ndarray:  # noqa: N803 - sklearn API
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        scores = model.predict_proba(X)[:, 1]
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def labels_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores) >= threshold).astype(int)


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(scores, [1, 99])
    if low == high:
        low, high = float(scores.min()), float(scores.max())
    thresholds = np.linspace(float(low), float(high), 401)
    thresholds = np.unique(np.concatenate([thresholds, np.array([0.0])]))
    best_threshold = 0.0
    best_f1 = -1.0
    for threshold in thresholds:
        macro_f1 = f1_score(y_true, labels_from_scores(scores, threshold), average="macro")
        if macro_f1 > best_f1:
            best_f1 = float(macro_f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def summarize_oof(name: str, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = labels_from_scores(scores, threshold)
    class_f1 = f1_score(y_true, pred, average=None, labels=[0, 1])
    return {
        "name": name,
        "threshold": float(threshold),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "class0_f1": float(class_f1[0]),
        "class1_f1": float(class_f1[1]),
        "confusion": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "prediction_distribution": dict(zip(*np.unique(pred, return_counts=True))),
        "report": classification_report(
            y_true,
            pred,
            labels=[0, 1],
            target_names=["Incomplete(0)", "Complete(1)"],
            digits=4,
        ),
    }


def evaluate_text_config(cfg: TextConfig, df: pd.DataFrame, n_splits: int, seed: int) -> dict:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    texts = df["text"].astype(str).values
    labels = df["label"].values
    oof_scores = np.zeros(len(df), dtype=np.float64)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(texts, labels), start=1):
        train_fold = df.iloc[tr_idx].reset_index(drop=True)
        if cfg.balance == "random":
            train_fold = oversample_training(train_fold, seed + fold)
        model = make_text_pipeline(cfg)
        model.fit(train_fold["text"].astype(str).values, train_fold["label"].values)
        scores = model_scores(model, texts[va_idx])
        oof_scores[va_idx] = scores
        fold_scores.append(f1_score(labels[va_idx], labels_from_scores(scores, 0.0), average="macro"))
    tuned_threshold, tuned_f1 = tune_threshold(labels, oof_scores)
    default_summary = summarize_oof(cfg.name, labels, oof_scores, 0.0)
    tuned_summary = summarize_oof(f"{cfg.name} tuned", labels, oof_scores, tuned_threshold)
    return {
        "config": cfg,
        "fold_scores": fold_scores,
        "mean_fold_macro_f1": float(np.mean(fold_scores)),
        "std_fold_macro_f1": float(np.std(fold_scores)),
        "default": default_summary,
        "tuned": tuned_summary,
        "tuned_threshold": tuned_threshold,
        "tuned_macro_f1": tuned_f1,
        "oof_scores": oof_scores,
    }


def evaluate_embedding_config(
    cfg: EmbeddingConfig,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> dict:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_scores = np.zeros(len(y), dtype=np.float64)
    fold_scores = []
    for tr_idx, va_idx in splitter.split(X, y):
        model = make_embedding_pipeline(cfg)
        model.fit(X[tr_idx], y[tr_idx])
        scores = model_scores(model, X[va_idx])
        oof_scores[va_idx] = scores
        fold_scores.append(f1_score(y[va_idx], labels_from_scores(scores, 0.0), average="macro"))
    tuned_threshold, tuned_f1 = tune_threshold(y, oof_scores)
    return {
        "config": cfg,
        "fold_scores": fold_scores,
        "mean_fold_macro_f1": float(np.mean(fold_scores)),
        "std_fold_macro_f1": float(np.std(fold_scores)),
        "default": summarize_oof(cfg.name, y, oof_scores, 0.0),
        "tuned": summarize_oof(f"{cfg.name} tuned", y, oof_scores, tuned_threshold),
        "tuned_threshold": tuned_threshold,
        "tuned_macro_f1": tuned_f1,
        "oof_scores": oof_scores,
    }


def result_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        cfg = res["config"]
        rows.append(
            {
                "name": cfg.name,
                "balance": getattr(cfg, "balance", "none"),
                "default_macro_f1": res["default"]["macro_f1"],
                "tuned_macro_f1": res["tuned"]["macro_f1"],
                "threshold": res["tuned_threshold"],
                "class0_f1": res["tuned"]["class0_f1"],
                "class1_f1": res["tuned"]["class1_f1"],
                "fold_mean_default": res["mean_fold_macro_f1"],
                "fold_std_default": res["std_fold_macro_f1"],
                "fold_scores_default": [round(x, 4) for x in res["fold_scores"]],
            }
        )
    return pd.DataFrame(rows).sort_values("tuned_macro_f1", ascending=False).reset_index(drop=True)


def validate_submission(path: Path, test_df: pd.DataFrame) -> pd.DataFrame:
    sub = pd.read_csv(path)
    if list(sub.columns) != ["id", "label"]:
        raise AssertionError(f"{path.name}: header is {sub.columns.tolist()}, expected ['id', 'label']")
    if len(sub) != len(test_df):
        raise AssertionError(f"{path.name}: row count {len(sub)} != {len(test_df)}")
    if sub["id"].tolist() != test_df["id"].tolist():
        raise AssertionError(f"{path.name}: ID order does not match test.csv")
    if sub["id"].duplicated().any():
        raise AssertionError(f"{path.name}: duplicate IDs")
    if set(sub["label"].unique()) - {0, 1}:
        raise AssertionError(f"{path.name}: labels must be integers in {{0,1}}")
    if not np.allclose(sub["label"].values, sub["label"].values.astype(int)):
        raise AssertionError(f"{path.name}: labels must be integer-valued")
    return sub


def write_submission(path: Path, test_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test_df["id"].values, "label": labels.astype(int)}).to_csv(path, index=False)
    return validate_submission(path, test_df)


def find_reference_files(output_dir: Path) -> list[Path]:
    roots = [output_dir, Path.cwd(), Path.cwd() / "part2"]
    found = []
    for name in REFERENCE_NAMES:
        for root in roots:
            path = root / name
            if path.exists() and path not in found:
                found.append(path)
                break
    return found


def compare_candidate(path: Path, test_df: pd.DataFrame, refs: list[Path]) -> dict:
    candidate = validate_submission(path, test_df)
    out = {}
    for ref in refs:
        try:
            ref_sub = validate_submission(ref, test_df)
            changed = int((candidate["label"].values != ref_sub["label"].values).sum())
            out[ref.name] = {"changed_rows": changed, "identical": changed == 0}
        except Exception as exc:  # noqa: BLE001 - report unusable references without aborting
            out[ref.name] = {"error": str(exc)}
    return out


def fit_text_full_and_predict(cfg: TextConfig, train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> np.ndarray:
    fit_df = train_df
    if cfg.balance == "random":
        fit_df = oversample_training(train_df, seed)
    model = make_text_pipeline(cfg)
    model.fit(fit_df["text"].astype(str).values, fit_df["label"].values)
    return model_scores(model, test_df["text"].astype(str).values)


def fit_embedding_full_and_predict(
    cfg: EmbeddingConfig,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    model = make_embedding_pipeline(cfg)
    model.fit(train_x, train_y)
    return model_scores(model, test_x)


def extract_distilbert_embeddings(
    train_texts: list[str],
    test_texts: list[str],
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.eval()

    def encode(texts: list[str], label: str) -> np.ndarray:
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                encoded = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                output = model(**encoded)
                hidden = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                mean_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                cls_pool = hidden[:, 0, :]
                pooled = torch.cat([cls_pool, mean_pool], dim=1).cpu().numpy()
                batches.append(pooled)
                done = min(start + batch_size, len(texts))
                if done == len(texts) or done % (batch_size * 10) == 0:
                    print(f"{label}: encoded {done}/{len(texts)}", flush=True)
        return np.vstack(batches)

    return encode(train_texts, "train"), encode(test_texts, "test")


def print_best_details(label: str, result: dict) -> None:
    print(f"\n=== {label} Best Details ===")
    print(f"name: {result['config'].name}")
    print(f"default Macro-F1: {result['default']['macro_f1']:.4f}")
    print(f"tuned Macro-F1: {result['tuned']['macro_f1']:.4f}")
    print(f"threshold: {result['tuned_threshold']:.6f}")
    print(f"class-wise F1 tuned: class0={result['tuned']['class0_f1']:.4f} class1={result['tuned']['class1_f1']:.4f}")
    print(f"confusion tuned [rows true 0/1, cols pred 0/1]: {result['tuned']['confusion']}")
    print("classification report tuned:")
    print(result["tuned"]["report"])


def main() -> int:
    args = parse_args()
    train_df, test_df = load_data(args.data_dir)
    y = train_df["label"].values

    text_configs = [
        TextConfig("word_char_endpoint_svc_c0.5", "svc", 0.5, None, "none"),
        TextConfig("word_char_endpoint_svc_c1.0", "svc", 1.0, None, "none"),
        TextConfig("word_char_endpoint_svc_c0.5_balanced", "svc", 0.5, "balanced", "none"),
        TextConfig("word_char_endpoint_svc_c0.5_random_os", "svc", 0.5, None, "random"),
        TextConfig(
            "word_only_svc_c0.5",
            "svc",
            0.5,
            None,
            "none",
            use_char=False,
            use_endpoint=False,
        ),
        TextConfig(
            "char_endpoint_svc_c0.5",
            "svc",
            0.5,
            None,
            "none",
            use_word=False,
            use_char=True,
            use_endpoint=True,
        ),
        TextConfig("word_char_endpoint_lr_c2_balanced", "lr", 2.0, "balanced", "none"),
    ]

    print("\nRunning classical text feature experiments...")
    text_results = [evaluate_text_config(cfg, train_df, args.n_splits, args.seed) for cfg in text_configs]
    text_table = result_table(text_results)
    print("\n=== Classical Validation Table ===")
    print(text_table.to_string(index=False))
    best_text = max(text_results, key=lambda res: res["tuned"]["macro_f1"])
    print_best_details("Classical", best_text)

    refs = find_reference_files(args.output_dir)
    if refs:
        print(f"reference files found: {[str(p) for p in refs]}")
    else:
        print("reference files found: none")

    generated = []
    if args.write_candidates:
        scores = fit_text_full_and_predict(best_text["config"], train_df, test_df, args.seed)
        base_path = args.output_dir / "submission_classical_endpoint_svc.csv"
        tuned_path = args.output_dir / "submission_classical_endpoint_svc_threshold.csv"
        for path, threshold in ((base_path, 0.0), (tuned_path, best_text["tuned_threshold"])):
            sub = write_submission(path, test_df, labels_from_scores(scores, threshold))
            comparisons = compare_candidate(path, test_df, refs)
            generated.append((path, threshold, sub["label"].value_counts().sort_index().to_dict(), comparisons))

    if args.frozen_distilbert:
        print("\nRunning frozen offline DistilBERT feature experiment...")
        train_x, test_x = extract_distilbert_embeddings(
            train_df["text"].astype(str).tolist(),
            test_df["text"].astype(str).tolist(),
            args.max_length,
            args.batch_size,
        )
        embed_configs = [
            EmbeddingConfig("frozen_distilbert_lr_c1", "lr", 1.0, None),
            EmbeddingConfig("frozen_distilbert_lr_c1_balanced", "lr", 1.0, "balanced"),
            EmbeddingConfig("frozen_distilbert_svc_c0.5", "svc", 0.5, None),
            EmbeddingConfig("frozen_distilbert_svc_c0.5_balanced", "svc", 0.5, "balanced"),
        ]
        embed_results = [evaluate_embedding_config(cfg, train_x, y, args.n_splits, args.seed) for cfg in embed_configs]
        embed_table = result_table(embed_results)
        print("\n=== Frozen DistilBERT Validation Table ===")
        print(embed_table.to_string(index=False))
        best_embed = max(embed_results, key=lambda res: res["tuned"]["macro_f1"])
        print_best_details("Frozen DistilBERT", best_embed)
        if args.write_candidates:
            embed_scores = fit_embedding_full_and_predict(best_embed["config"], train_x, y, test_x)
            base_path = args.output_dir / "submission_frozen_distilbert.csv"
            tuned_path = args.output_dir / "submission_frozen_distilbert_threshold.csv"
            for path, threshold in ((base_path, 0.0), (tuned_path, best_embed["tuned_threshold"])):
                sub = write_submission(path, test_df, labels_from_scores(embed_scores, threshold))
                comparisons = compare_candidate(path, test_df, refs)
                generated.append((path, threshold, sub["label"].value_counts().sort_index().to_dict(), comparisons))

    if generated:
        print("\n=== Generated Candidate CSVs ===")
        seen = {}
        for path, threshold, distribution, comparisons in generated:
            labels = pd.read_csv(path)["label"].astype(int).to_numpy()
            duplicate_of = next((name for name, prev in seen.items() if np.array_equal(labels, prev)), None)
            seen[path.name] = labels
            print(f"{path.name}: threshold={threshold:.6f} distribution={distribution}")
            print(f"  identical_to_generated={duplicate_of if duplicate_of else 'none'}")
            print(f"  reference_comparisons={comparisons if comparisons else 'no reference files'}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
