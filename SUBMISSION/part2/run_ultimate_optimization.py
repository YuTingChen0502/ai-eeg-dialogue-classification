"""Reconstruct and compare historical Part 2 DistilBERT candidates.

This helper keeps the notebook portable while making the late-stage model
diagnostics reproducible. It never uses test labels and it never talks to
Kaggle. Transformer loading is forced through local files only so a missing
cache fails loudly instead of downloading a model. These diagnostics are not
the final model selection; the final submitted Part 2 model is tuned
RoBERTa-base seed 42 with OOF Macro-F1 0.968701 and Kaggle public score 0.9686.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import train_bert


PART2_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PART2_DIR / "outputs" / "part2_ultimate_optimization"
DEFAULT_CHECKPOINTS = {
    42: PART2_DIR / "outputs" / "bert_repro_seed42" / "model",
    13: PART2_DIR / "outputs" / "bert_multiseed_42_13" / "models" / "seed_13",
}
REFERENCE_SUBMISSIONS = {
    "seed42_threshold": PART2_DIR / "submission_bert_repro_seed42_threshold.csv",
    "seed13_threshold": PART2_DIR / "submission_bert_seed13_threshold.csv",
    "avg_multiseed": PART2_DIR / "submission_bert_avg_multiseed.csv",
}


@dataclass
class SeedResult:
    seed: int
    checkpoint_dir: Path | None
    valid_indices: np.ndarray
    valid_ids: np.ndarray
    valid_labels: np.ndarray
    valid_probs: np.ndarray
    test_probs: np.ndarray
    raw_f1: float
    threshold: float
    threshold_f1: float
    raw_submission: Path | None = None
    threshold_submission: Path | None = None


@dataclass
class CandidateResult:
    name: str
    path: Path
    source: str
    seeds: str
    validation_raw: float | None
    validation_threshold: float | None
    validation_kind: str
    threshold: float
    distribution: dict[int, int]
    hamming_seed42: int | None
    hamming_seed13: int | None
    hamming_avg: int | None
    changed_ids_seed42: list[str]
    changed_confidence: dict[str, float | int | None]
    recommendation: str
    risk_notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical DistilBERT diagnostics for Part 2; not final model selection.")
    parser.add_argument("--data-dir", type=Path, default=PART2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--threshold-min", type=float, default=0.30)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--reconstruct-seeds", default="42,13")
    parser.add_argument("--train-seeds", default="", help="Optional comma-separated extra seeds to train.")
    parser.add_argument("--common-seeds", default="", help="Optional comma-separated seeds for a fixed train/validation split experiment.")
    parser.add_argument("--common-split-seed", type=int, default=42, help="Random seed used only to choose the common validation split.")
    parser.add_argument("--skip-reconstruct", action="store_true")
    parser.add_argument("--skip-duplicates", action="store_true")
    parser.add_argument("--max-near-duplicate-candidates", type=int, default=50)
    return parser.parse_args()


def parse_seed_list(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds


def ensure_local_checkpoint(path: Path) -> None:
    required = ["config.json", "tokenizer.json"]
    missing = [name for name in required if not (path / name).exists()]
    has_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    if missing or not has_weights:
        raise FileNotFoundError(f"Checkpoint is incomplete: {path}")


def checkpoint_for_seed(seed: int, output_dir: Path) -> Path | None:
    if seed in DEFAULT_CHECKPOINTS:
        return DEFAULT_CHECKPOINTS[seed]
    generated = output_dir / "models" / f"seed_{seed}"
    if generated.exists():
        return generated
    return None


def labels_from_probs(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(probs) >= threshold).astype(int)


def format_distribution(labels: Iterable[int]) -> dict[int, int]:
    counts = Counter(int(x) for x in labels)
    return {0: counts.get(0, 0), 1: counts.get(1, 0)}


def submission_path_for_seed(seed: int, tuned: bool) -> Path:
    suffix = "_threshold" if tuned else ""
    return PART2_DIR / f"submission_bert_seed{seed}{suffix}.csv"


def write_submission(path: Path, test_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    submission = pd.DataFrame({"id": test_df["id"].values, "label": labels.astype(int)})
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return validate_submission(path, test_df)


def validate_submission(path: Path, test_df: pd.DataFrame) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header != ["id", "label"]:
        raise AssertionError(f"{path} header must be exactly id,label; got {header}")
    sub = pd.read_csv(path, dtype={"id": str})
    expected_ids = test_df["id"].astype(str).tolist()
    if len(sub) != len(test_df):
        raise AssertionError(f"{path} row count {len(sub)} != {len(test_df)}")
    if sub["id"].astype(str).tolist() != expected_ids:
        raise AssertionError(f"{path} IDs/order do not match test.csv")
    if sub["id"].duplicated().any():
        raise AssertionError(f"{path} contains duplicate IDs")
    labels = set(sub["label"].dropna().astype(int).unique().tolist())
    if not labels.issubset({0, 1}):
        raise AssertionError(f"{path} contains labels outside {{0,1}}: {labels}")
    if sub["label"].isna().any():
        raise AssertionError(f"{path} contains missing labels")
    return sub


def split_for_seed(train_df: pd.DataFrame, seed: int, valid_size: float) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    indexed = train_df.reset_index(names="orig_index")
    train_split, valid_split = train_test_split(
        indexed,
        test_size=valid_size,
        random_state=seed,
        stratify=indexed["label"],
    )
    valid_indices = valid_split["orig_index"].to_numpy(dtype=int)
    return (
        train_split.drop(columns=["orig_index"]).reset_index(drop=True),
        valid_split.drop(columns=["orig_index"]).reset_index(drop=True),
        valid_indices,
    )


def make_runtime_args(args: argparse.Namespace, output_dir: Path, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=args.data_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_dir=output_dir,
        submission_path=PART2_DIR / f"submission_bert_seed{seed}.csv",
        seed=seed,
        valid_size=args.valid_size,
        k_folds=0,
        threshold_tune=True,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        multi_seed=False,
        multi_seeds="",
        debug=False,
    )


def predict_checkpoint(
    *,
    seed: int,
    checkpoint_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    libs: dict,
    args: argparse.Namespace,
) -> SeedResult:
    ensure_local_checkpoint(checkpoint_dir)
    torch = libs["torch"]
    AutoTokenizer = libs["AutoTokenizer"]
    AutoModelForSequenceClassification = libs["AutoModelForSequenceClassification"]
    DataLoader = libs["DataLoader"]

    _, valid_df, valid_indices = split_for_seed(train_df, seed, args.valid_size)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    text_dataset_cls = train_bert.make_datasets_class(libs["Dataset"])

    valid_ds = text_dataset_cls(valid_df["text"], valid_df["label"], tokenizer, args.max_length)
    test_ds = text_dataset_cls(test_df["text"], None, tokenizer, args.max_length)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    _, valid_probs = train_bert.predict(model, valid_loader, device, torch)
    _, test_probs = train_bert.predict(model, test_loader, device, torch)
    valid_labels = valid_df["label"].to_numpy(dtype=int)
    raw_f1 = f1_score(valid_labels, labels_from_probs(valid_probs, 0.5), average="macro")
    threshold, threshold_f1 = train_bert.tune_threshold(
        valid_labels,
        valid_probs,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )
    return SeedResult(
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        valid_indices=valid_indices,
        valid_ids=valid_df["id"].astype(str).to_numpy(),
        valid_labels=valid_labels,
        valid_probs=valid_probs,
        test_probs=test_probs,
        raw_f1=float(raw_f1),
        threshold=float(threshold),
        threshold_f1=float(threshold_f1),
    )


def predict_checkpoint_on_split(
    *,
    seed: int,
    checkpoint_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    libs: dict,
    args: argparse.Namespace,
    split_seed: int,
) -> SeedResult:
    ensure_local_checkpoint(checkpoint_dir)
    torch = libs["torch"]
    AutoTokenizer = libs["AutoTokenizer"]
    AutoModelForSequenceClassification = libs["AutoModelForSequenceClassification"]
    DataLoader = libs["DataLoader"]

    _, valid_df, valid_indices = split_for_seed(train_df, split_seed, args.valid_size)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    text_dataset_cls = train_bert.make_datasets_class(libs["Dataset"])

    valid_ds = text_dataset_cls(valid_df["text"], valid_df["label"], tokenizer, args.max_length)
    test_ds = text_dataset_cls(test_df["text"], None, tokenizer, args.max_length)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    _, valid_probs = train_bert.predict(model, valid_loader, device, torch)
    _, test_probs = train_bert.predict(model, test_loader, device, torch)
    valid_labels = valid_df["label"].to_numpy(dtype=int)
    raw_f1 = f1_score(valid_labels, labels_from_probs(valid_probs, 0.5), average="macro")
    threshold, threshold_f1 = train_bert.tune_threshold(
        valid_labels,
        valid_probs,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )
    return SeedResult(
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        valid_indices=valid_indices,
        valid_ids=valid_df["id"].astype(str).to_numpy(),
        valid_labels=valid_labels,
        valid_probs=valid_probs,
        test_probs=test_probs,
        raw_f1=float(raw_f1),
        threshold=float(threshold),
        threshold_f1=float(threshold_f1),
    )


def train_seed(
    *,
    seed: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    libs: dict,
    args: argparse.Namespace,
) -> SeedResult:
    torch = libs["torch"]
    AutoTokenizer = libs["AutoTokenizer"]
    AutoModelForSequenceClassification = libs["AutoModelForSequenceClassification"]
    DataLoader = libs["DataLoader"]

    train_bert.set_seed(seed, torch)
    output_dir = args.output_dir / "models" / f"seed_{seed}"
    runtime_args = make_runtime_args(args, output_dir, seed)
    train_split, valid_split, valid_indices = split_for_seed(train_df, seed, args.valid_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    text_dataset_cls = train_bert.make_datasets_class(libs["Dataset"])

    def model_factory():
        return AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            local_files_only=True,
        )

    model, raw_f1, _, _, valid_probs, valid_labels = train_bert.train_one_split(
        train_df=train_split,
        valid_df=valid_split,
        tokenizer=tokenizer,
        model_factory=model_factory,
        text_dataset_cls=text_dataset_cls,
        libs=libs,
        args=runtime_args,
        fold_name=f"seed {seed}",
        save_model=True,
    )
    threshold, threshold_f1 = train_bert.tune_threshold(
        valid_labels,
        valid_probs,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = text_dataset_cls(test_df["text"], None, tokenizer, args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    _, test_probs = train_bert.predict(model, test_loader, device, torch)
    return SeedResult(
        seed=seed,
        checkpoint_dir=output_dir,
        valid_indices=valid_indices,
        valid_ids=valid_split["id"].astype(str).to_numpy(),
        valid_labels=valid_labels,
        valid_probs=valid_probs,
        test_probs=test_probs,
        raw_f1=float(raw_f1),
        threshold=float(threshold),
        threshold_f1=float(threshold_f1),
    )


def train_seed_on_split(
    *,
    seed: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    valid_indices: np.ndarray,
    test_df: pd.DataFrame,
    libs: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> SeedResult:
    torch = libs["torch"]
    AutoTokenizer = libs["AutoTokenizer"]
    AutoModelForSequenceClassification = libs["AutoModelForSequenceClassification"]
    DataLoader = libs["DataLoader"]

    train_bert.set_seed(seed, torch)
    runtime_args = make_runtime_args(args, output_dir, seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    text_dataset_cls = train_bert.make_datasets_class(libs["Dataset"])

    def model_factory():
        return AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            local_files_only=True,
        )

    model, raw_f1, _, _, valid_probs, valid_labels = train_bert.train_one_split(
        train_df=train_df,
        valid_df=valid_df,
        tokenizer=tokenizer,
        model_factory=model_factory,
        text_dataset_cls=text_dataset_cls,
        libs=libs,
        args=runtime_args,
        fold_name=f"common seed {seed}",
        save_model=True,
    )
    threshold, threshold_f1 = train_bert.tune_threshold(
        valid_labels,
        valid_probs,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = text_dataset_cls(test_df["text"], None, tokenizer, args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    _, test_probs = train_bert.predict(model, test_loader, device, torch)
    return SeedResult(
        seed=seed,
        checkpoint_dir=output_dir,
        valid_indices=valid_indices,
        valid_ids=valid_df["id"].astype(str).to_numpy(),
        valid_labels=valid_labels,
        valid_probs=valid_probs,
        test_probs=test_probs,
        raw_f1=float(raw_f1),
        threshold=float(threshold),
        threshold_f1=float(threshold_f1),
    )


def save_seed_artifacts(result: SeedResult, output_dir: Path) -> None:
    seed_dir = output_dir / f"seed_{result.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    np.save(seed_dir / "valid_probs.npy", result.valid_probs)
    np.save(seed_dir / "test_probs.npy", result.test_probs)
    np.save(seed_dir / "valid_labels.npy", result.valid_labels)
    pd.DataFrame(
        {
            "valid_position": np.arange(len(result.valid_indices)),
            "train_index": result.valid_indices,
            "id": result.valid_ids,
            "label": result.valid_labels,
            "prob_complete": result.valid_probs,
        }
    ).to_csv(seed_dir / "valid_split_metadata.csv", index=False)
    metadata = {
        "seed": result.seed,
        "checkpoint_dir": str(result.checkpoint_dir) if result.checkpoint_dir else None,
        "raw_macro_f1": result.raw_f1,
        "threshold": result.threshold,
        "threshold_macro_f1": result.threshold_f1,
        "valid_count": int(len(result.valid_labels)),
        "test_count": int(len(result.test_probs)),
    }
    (seed_dir / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_seed_artifacts(seed: int, output_dir: Path, checkpoint_dir: Path | None, args: argparse.Namespace) -> SeedResult | None:
    seed_dir = output_dir / f"seed_{seed}"
    valid_probs_path = seed_dir / "valid_probs.npy"
    test_probs_path = seed_dir / "test_probs.npy"
    valid_labels_path = seed_dir / "valid_labels.npy"
    split_path = seed_dir / "valid_split_metadata.csv"
    if not all(path.exists() for path in [valid_probs_path, test_probs_path, valid_labels_path, split_path]):
        return None
    valid_probs = np.load(valid_probs_path)
    test_probs = np.load(test_probs_path)
    valid_labels = np.load(valid_labels_path)
    split = pd.read_csv(split_path, dtype={"id": str})
    raw_f1 = f1_score(valid_labels, labels_from_probs(valid_probs, 0.5), average="macro")
    threshold, threshold_f1 = train_bert.tune_threshold(
        valid_labels,
        valid_probs,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )
    return SeedResult(
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        valid_indices=split["train_index"].to_numpy(dtype=int),
        valid_ids=split["id"].astype(str).to_numpy(),
        valid_labels=valid_labels.astype(int),
        valid_probs=valid_probs,
        test_probs=test_probs,
        raw_f1=float(raw_f1),
        threshold=float(threshold),
        threshold_f1=float(threshold_f1),
    )


def load_reference(path: Path, test_df: pd.DataFrame) -> pd.Series | None:
    if not path.exists():
        return None
    sub = validate_submission(path, test_df)
    return sub["label"].astype(int)


def hamming(candidate: pd.Series, reference: pd.Series | None) -> int | None:
    if reference is None:
        return None
    return int((candidate.astype(int).to_numpy() != reference.astype(int).to_numpy()).sum())


def changed_ids(candidate: pd.Series, reference: pd.Series | None, test_df: pd.DataFrame) -> list[str]:
    if reference is None:
        return []
    mask = candidate.astype(int).to_numpy() != reference.astype(int).to_numpy()
    return test_df.loc[mask, "id"].astype(str).tolist()


def confidence_summary(probs: np.ndarray, threshold: float, changed_mask: np.ndarray) -> dict[str, float | int | None]:
    if not changed_mask.any():
        return {"count": 0, "min_margin": None, "mean_margin": None, "min_prob": None, "max_prob": None}
    changed_probs = np.asarray(probs)[changed_mask]
    margins = np.abs(changed_probs - threshold)
    return {
        "count": int(changed_mask.sum()),
        "min_margin": float(np.min(margins)),
        "mean_margin": float(np.mean(margins)),
        "min_prob": float(np.min(changed_probs)),
        "max_prob": float(np.max(changed_probs)),
    }


def build_candidate(
    *,
    name: str,
    path: Path,
    source: str,
    seeds: str,
    probs: np.ndarray,
    threshold: float,
    test_df: pd.DataFrame,
    validation_raw: float | None,
    validation_threshold: float | None,
    validation_kind: str,
    references: dict[str, pd.Series | None],
    recommendation: str,
    risk_notes: str,
) -> CandidateResult:
    labels = labels_from_probs(probs, threshold)
    sub = write_submission(path, test_df, labels)
    labels_series = sub["label"].astype(int)
    seed42_ref = references.get("seed42_threshold")
    seed13_ref = references.get("seed13_threshold")
    avg_ref = references.get("avg_multiseed")
    seed42_changed = changed_ids(labels_series, seed42_ref, test_df)
    if seed42_ref is not None:
        changed_mask = labels_series.to_numpy() != seed42_ref.to_numpy()
    else:
        changed_mask = np.zeros(len(labels_series), dtype=bool)
    return CandidateResult(
        name=name,
        path=path,
        source=source,
        seeds=seeds,
        validation_raw=validation_raw,
        validation_threshold=validation_threshold,
        validation_kind=validation_kind,
        threshold=float(threshold),
        distribution=format_distribution(labels_series),
        hamming_seed42=hamming(labels_series, seed42_ref),
        hamming_seed13=hamming(labels_series, seed13_ref),
        hamming_avg=hamming(labels_series, avg_ref),
        changed_ids_seed42=seed42_changed,
        changed_confidence=confidence_summary(probs, threshold, changed_mask),
        recommendation=recommendation,
        risk_notes=risk_notes,
    )


def common_validation_ensemble(seed_results: list[SeedResult]) -> tuple[np.ndarray, np.ndarray] | None:
    if not seed_results:
        return None
    first = seed_results[0].valid_indices
    if not all(np.array_equal(first, res.valid_indices) for res in seed_results[1:]):
        return None
    probs = np.mean([res.valid_probs for res in seed_results], axis=0)
    return seed_results[0].valid_labels, probs


def ensemble_validation_summary(seed_results: list[SeedResult], threshold: float) -> tuple[float | None, float | None, str]:
    common = common_validation_ensemble(seed_results)
    if common is not None:
        y, probs = common
        raw = f1_score(y, labels_from_probs(probs, 0.5), average="macro")
        tuned = f1_score(y, labels_from_probs(probs, threshold), average="macro")
        return float(raw), float(tuned), "common_validation_ensemble"
    raw_mean = float(np.mean([res.raw_f1 for res in seed_results])) if seed_results else None
    tuned_mean = float(np.mean([res.threshold_f1 for res in seed_results])) if seed_results else None
    return raw_mean, tuned_mean, "mean_seed_holdout_not_true_ensemble"


def write_row_comparison(
    output_dir: Path,
    test_df: pd.DataFrame,
    candidates: list[CandidateResult],
    references: dict[str, pd.Series | None],
) -> None:
    rows: list[dict[str, object]] = []
    seed42_ref = references.get("seed42_threshold")
    for cand in candidates:
        sub = pd.read_csv(cand.path, dtype={"id": str})
        for i, row in sub.iterrows():
            ref_label = None if seed42_ref is None else int(seed42_ref.iloc[i])
            label = int(row["label"])
            if ref_label is None or label != ref_label:
                rows.append(
                    {
                        "candidate": cand.name,
                        "id": str(row["id"]),
                        "row_index": i,
                        "label": label,
                        "seed42_threshold_label": ref_label,
                        "changed_vs_seed42": ref_label is not None and label != ref_label,
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "row_level_comparisons.csv", index=False)


def normalize_text(text: object) -> str:
    value = "" if pd.isna(text) else str(text)
    value = value.lower().strip()
    value = re.sub(r"[\u2018\u2019]", "'", value)
    value = re.sub(r"[\u201c\u201d]", '"', value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.!?;:])", r"\1", value)
    return value


def duplicate_transfer_report(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_candidate: CandidateResult,
    base_probs: np.ndarray,
    output_dir: Path,
    max_near_candidates: int,
) -> tuple[dict[str, object], Path | None]:
    train_norm = train_df["text"].map(normalize_text)
    test_norm = test_df["text"].map(normalize_text)
    train_groups: dict[str, list[int]] = defaultdict(list)
    for idx, text in train_norm.items():
        train_groups[text].append(int(idx))

    base_sub = pd.read_csv(base_candidate.path, dtype={"id": str})
    report_rows: list[dict[str, object]] = []
    overrides: dict[int, int] = {}
    conflicts = 0
    exact_matches = 0

    for test_idx, text in test_norm.items():
        matched_train = train_groups.get(text, [])
        if not matched_train:
            continue
        exact_matches += 1
        labels = train_df.loc[matched_train, "label"].astype(int).tolist()
        unanimous = len(set(labels)) == 1
        if not unanimous:
            conflicts += 1
        old_pred = int(base_sub.loc[test_idx, "label"])
        new_pred = labels[0] if unanimous else old_pred
        apply = unanimous and old_pred != new_pred
        if apply:
            overrides[int(test_idx)] = int(new_pred)
        report_rows.append(
            {
                "match_type": "exact",
                "test_row_index": int(test_idx),
                "test_id": str(test_df.loc[test_idx, "id"]),
                "train_indices": ";".join(str(i) for i in matched_train),
                "train_ids": ";".join(train_df.loc[matched_train, "id"].astype(str).tolist()),
                "train_labels": ";".join(str(x) for x in labels),
                "similarity": 1.0,
                "unanimous": unanimous,
                "old_prediction": old_pred,
                "new_prediction": new_pred,
                "model_confidence_complete": float(base_probs[test_idx]),
                "applied": apply,
                "rationale": "Exact normalized text match with unanimous train labels" if apply else "No override needed or conflicting labels",
            }
        )

    near_rows: list[dict[str, object]] = []
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False)
    train_vec = vectorizer.fit_transform(train_norm.tolist())
    test_vec = vectorizer.transform(test_norm.tolist())
    cosine = cosine_similarity(test_vec, train_vec)
    for test_idx, text in test_norm.items():
        if not text:
            continue
        candidate_order = np.argsort(cosine[int(test_idx)])[::-1][:5]
        best: tuple[float, int] | None = None
        for train_idx in candidate_order:
            train_text = train_norm.iloc[int(train_idx)]
            if text == train_text:
                continue
            # TF-IDF proposes candidates quickly; difflib is the strict gate.
            similarity = difflib.SequenceMatcher(None, text, train_text, autojunk=False).ratio()
            if similarity >= 0.99 and (best is None or similarity > best[0]):
                best = (float(similarity), int(train_idx))
        if best is None:
            continue
        similarity, train_idx = best
        labels = [int(train_df.loc[train_idx, "label"])]
        old_pred = int(base_sub.loc[test_idx, "label"])
        new_pred = labels[0]
        apply = old_pred != new_pred
        if apply:
            overrides[int(test_idx)] = int(new_pred)
        near_rows.append(
            {
                "match_type": "near",
                "test_row_index": int(test_idx),
                "test_id": str(test_df.loc[test_idx, "id"]),
                "train_indices": str(train_idx),
                "train_ids": str(train_df.loc[train_idx, "id"]),
                "train_labels": str(labels[0]),
                "similarity": similarity,
                "unanimous": True,
                "old_prediction": old_pred,
                "new_prediction": new_pred,
                "model_confidence_complete": float(base_probs[test_idx]),
                "applied": apply,
                "rationale": "Near duplicate >=0.99 with single train match" if apply else "No override needed",
            }
        )
        if len(near_rows) >= max_near_candidates:
            break

    report_rows.extend(near_rows)
    report_path = output_dir / "duplicate_transfer_report.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)

    transfer_path: Path | None = None
    if overrides:
        transferred = base_sub.copy()
        for idx, label in sorted(overrides.items()):
            transferred.loc[idx, "label"] = label
        transfer_path = base_candidate.path.with_name(f"{base_candidate.path.stem}_duplicate_transfer.csv")
        transferred.to_csv(transfer_path, index=False)
        validate_submission(transfer_path, test_df)

    summary = {
        "exact_matches": exact_matches,
        "near_matches_ge_0_99": len(near_rows),
        "override_candidates": len(overrides),
        "applied_overrides": len(overrides),
        "conflicts": conflicts,
        "report_path": str(report_path),
        "transfer_candidate": str(transfer_path) if transfer_path else None,
    }
    return summary, transfer_path


def pseudo_label_diagnostics(probs: np.ndarray) -> dict[str, dict[str, int]]:
    arr = np.asarray(probs)
    return {
        "0.99": {"positive": int((arr >= 0.99).sum()), "negative": int((arr <= 0.01).sum())},
        "0.98": {"positive": int((arr >= 0.98).sum()), "negative": int((arr <= 0.02).sum())},
        "0.95": {"positive": int((arr >= 0.95).sum()), "negative": int((arr <= 0.05).sum())},
    }


def candidate_to_row(candidate: CandidateResult) -> dict[str, object]:
    return {
        "name": candidate.name,
        "filename": str(candidate.path),
        "source": candidate.source,
        "seeds": candidate.seeds,
        "validation_macro_f1_raw": candidate.validation_raw,
        "validation_macro_f1_threshold": candidate.validation_threshold,
        "validation_kind": candidate.validation_kind,
        "threshold": candidate.threshold,
        "distribution_0": candidate.distribution.get(0, 0),
        "distribution_1": candidate.distribution.get(1, 0),
        "hamming_vs_seed42_threshold": candidate.hamming_seed42,
        "hamming_vs_seed13_threshold": candidate.hamming_seed13,
        "hamming_vs_avg_multiseed": candidate.hamming_avg,
        "changed_count_vs_seed42": len(candidate.changed_ids_seed42),
        "changed_ids_vs_seed42": ";".join(candidate.changed_ids_seed42),
        "changed_confidence_json": json.dumps(candidate.changed_confidence, sort_keys=True),
        "risk_notes": candidate.risk_notes,
        "recommendation": candidate.recommendation,
    }


def make_common_candidate(
    *,
    name: str,
    path: Path,
    source: str,
    members: list[SeedResult],
    threshold: float,
    test_probs: np.ndarray,
    valid_probs: np.ndarray,
    test_df: pd.DataFrame,
    references: dict[str, pd.Series | None],
    recommendation: str,
    risk_notes: str,
) -> CandidateResult:
    valid_labels = members[0].valid_labels
    raw_f1 = f1_score(valid_labels, labels_from_probs(valid_probs, 0.5), average="macro")
    tuned_f1 = f1_score(valid_labels, labels_from_probs(valid_probs, threshold), average="macro")
    return build_candidate(
        name=name,
        path=path,
        source=source,
        seeds=",".join(str(res.seed) for res in members),
        probs=test_probs,
        threshold=threshold,
        test_df=test_df,
        validation_raw=float(raw_f1),
        validation_threshold=float(tuned_f1),
        validation_kind="true_common_validation",
        references=references,
        recommendation=recommendation,
        risk_notes=risk_notes,
    )


def run_common_split_experiment(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    libs: dict,
    args: argparse.Namespace,
    references: dict[str, pd.Series | None],
) -> dict[str, object] | None:
    common_seeds = parse_seed_list(args.common_seeds)
    if not common_seeds:
        return None

    common_dir = args.output_dir / f"common_split_seed_{args.common_split_seed}"
    common_dir.mkdir(parents=True, exist_ok=True)
    train_split, valid_split, valid_indices = split_for_seed(train_df, args.common_split_seed, args.valid_size)
    pd.DataFrame(
        {
            "valid_position": np.arange(len(valid_indices)),
            "train_index": valid_indices,
            "id": valid_split["id"].astype(str),
            "label": valid_split["label"].astype(int),
        }
    ).to_csv(common_dir / "common_validation_split.csv", index=False)

    common_results: dict[int, SeedResult] = {}
    for seed in common_seeds:
        model_dir = common_dir / "models" / f"seed_{seed}"
        cached = load_seed_artifacts(seed, common_dir, model_dir if model_dir.exists() else None, args)
        if cached is not None:
            common_results[seed] = cached
            print(
                f"common seed {seed}: loaded saved probabilities "
                f"raw_f1={cached.raw_f1:.4f} threshold={cached.threshold:.2f} "
                f"threshold_f1={cached.threshold_f1:.4f}"
            )
            continue

        if seed == args.common_split_seed and seed in DEFAULT_CHECKPOINTS and args.common_split_seed == 42:
            print(f"common seed {seed}: reusing existing checkpoint on split {args.common_split_seed}")
            result = predict_checkpoint_on_split(
                seed=seed,
                checkpoint_dir=DEFAULT_CHECKPOINTS[seed],
                train_df=train_df,
                test_df=test_df,
                libs=libs,
                args=args,
                split_seed=args.common_split_seed,
            )
        elif model_dir.exists():
            print(f"common seed {seed}: reconstructing from common checkpoint {model_dir}")
            result = predict_checkpoint_on_split(
                seed=seed,
                checkpoint_dir=model_dir,
                train_df=train_df,
                test_df=test_df,
                libs=libs,
                args=args,
                split_seed=args.common_split_seed,
            )
        else:
            print(f"common seed {seed}: training on fixed split seed {args.common_split_seed}")
            result = train_seed_on_split(
                seed=seed,
                train_df=train_split,
                valid_df=valid_split,
                valid_indices=valid_indices,
                test_df=test_df,
                libs=libs,
                args=args,
                output_dir=model_dir,
            )
        save_seed_artifacts(result, common_dir)
        common_results[seed] = result
        print(
            f"common seed {seed}: raw_f1={result.raw_f1:.4f} "
            f"threshold={result.threshold:.2f} threshold_f1={result.threshold_f1:.4f}"
        )

    candidates: list[CandidateResult] = []
    for seed, result in sorted(common_results.items()):
        path = PART2_DIR / f"submission_bert_common_seed{seed}_threshold.csv"
        candidate = make_common_candidate(
            name=f"common_seed{seed}_threshold",
            path=path,
            source="DistilBERT fixed split threshold tuned",
            members=[result],
            threshold=result.threshold,
            test_probs=result.test_probs,
            valid_probs=result.valid_probs,
            test_df=test_df,
            references=references,
            recommendation="baseline" if seed == 42 else "compare_before_submission",
            risk_notes="Single model on the fixed common validation split.",
        )
        candidates.append(candidate)
        if seed == 13:
            references["common_seed13_threshold"] = pd.read_csv(path)["label"].astype(int)

    ranked = sorted(common_results.values(), key=lambda res: res.threshold_f1, reverse=True)
    ensemble_specs: list[tuple[str, list[SeedResult]]] = []
    if 42 in common_results and 13 in common_results:
        ensemble_specs.append(("common_seed42_seed13_average", [common_results[42], common_results[13]]))
    if len(ranked) >= 2:
        ensemble_specs.append(("common_top2_seed_average", ranked[:2]))
    if len(ranked) >= 3:
        ensemble_specs.append(("common_top3_seed_average", ranked[:3]))
    if len(ranked) >= 2:
        ensemble_specs.append(("common_all_seed_average", ranked))

    seen_specs = set()
    for name, members in ensemble_specs:
        key = tuple(sorted(res.seed for res in members))
        if key in seen_specs:
            continue
        seen_specs.add(key)
        valid_probs = np.mean([res.valid_probs for res in members], axis=0)
        test_probs = np.mean([res.test_probs for res in members], axis=0)
        threshold, threshold_f1 = train_bert.tune_threshold(
            members[0].valid_labels,
            valid_probs,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
        path = PART2_DIR / f"submission_bert_{name}.csv"
        recommendation = "serious_sparse_diagnostic_candidate"
        risk = "True common-split ensemble; still one holdout split, not full OOF."
        candidate = make_common_candidate(
            name=name,
            path=path,
            source="DistilBERT fixed split probability average",
            members=members,
            threshold=threshold,
            test_probs=test_probs,
            valid_probs=valid_probs,
            test_df=test_df,
            references=references,
            recommendation=recommendation,
            risk_notes=f"{risk} tuned_f1={threshold_f1:.4f}",
        )
        candidates.append(candidate)

    summary_df = pd.DataFrame([candidate_to_row(cand) for cand in candidates])
    summary_df.to_csv(common_dir / "common_candidate_summary.csv", index=False)
    write_row_comparison(common_dir, test_df, candidates, references)

    best = max(candidates, key=lambda cand: -math.inf if cand.validation_threshold is None else cand.validation_threshold)
    member_seeds = [int(seed) for seed in best.seeds.split(",") if seed.strip()]
    best_probs = np.mean([common_results[seed].test_probs for seed in member_seeds], axis=0)
    pseudo = {
        "base_candidate": best.name,
        "confidence_counts": pseudo_label_diagnostics(best_probs),
        "attempted": False,
        "result": "Not attempted; common validation exists, but pseudo-labeling still needs a stronger OOF comparison before it is safe.",
        "recommended": False,
    }
    (common_dir / "pseudo_label_diagnostics.json").write_text(json.dumps(pseudo, indent=2), encoding="utf-8")

    summary = {
        "common_split_seed": args.common_split_seed,
        "valid_count": int(len(valid_indices)),
        "seeds": {
            str(seed): {
                "raw_macro_f1": res.raw_f1,
                "threshold": res.threshold,
                "threshold_macro_f1": res.threshold_f1,
                "checkpoint_dir": str(res.checkpoint_dir) if res.checkpoint_dir else None,
            }
            for seed, res in sorted(common_results.items())
        },
        "candidates": [candidate_to_row(cand) for cand in candidates],
        "best_candidate": candidate_to_row(best),
    }
    (common_dir / "common_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Common-Split Candidate Summary ===")
    for cand in candidates:
        print(
            f"{cand.name}: threshold={cand.threshold:.2f} "
            f"val={cand.validation_threshold:.4f} distribution={cand.distribution} "
            f"hamming_seed42={cand.hamming_seed42}"
        )
    print(f"Common-split diagnostics saved to {common_dir}")
    return summary


def audit_existing_files(test_df: pd.DataFrame) -> dict[str, object]:
    submissions = []
    for path in sorted(PART2_DIR.glob("submission_*.csv")):
        try:
            sub = validate_submission(path, test_df)
            status = "OK"
            distribution = format_distribution(sub["label"])
        except Exception as exc:  # noqa: BLE001 - audit should keep going.
            status = f"INVALID: {exc}"
            distribution = {}
        submissions.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "status": status,
                "distribution": distribution,
            }
        )
    artifact_files = []
    for path in sorted((PART2_DIR / "outputs").rglob("*")) if (PART2_DIR / "outputs").exists() else []:
        if path.is_file():
            artifact_files.append({"path": str(path), "bytes": path.stat().st_size})
    return {"submissions": submissions, "output_artifacts": artifact_files}


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    libs = train_bert.require_transformer_deps()
    torch = libs["torch"]
    train_bert.set_seed(42, torch)
    print(f"device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    train_df, test_df = train_bert.load_data(args.data_dir)
    train_df["id"] = train_df["id"].astype(str)
    test_df["id"] = test_df["id"].astype(str)

    audit = audit_existing_files(test_df)
    (args.output_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    seed_results: dict[int, SeedResult] = {}
    if not args.skip_reconstruct:
        for seed in parse_seed_list(args.reconstruct_seeds):
            checkpoint = checkpoint_for_seed(seed, args.output_dir)
            cached = load_seed_artifacts(seed, args.output_dir, checkpoint, args)
            if cached is not None:
                seed_results[seed] = cached
                print(
                    f"seed {seed}: loaded saved probabilities "
                    f"raw_f1={cached.raw_f1:.4f} threshold={cached.threshold:.2f} "
                    f"threshold_f1={cached.threshold_f1:.4f}"
                )
                continue
            if checkpoint is None or not checkpoint.exists():
                print(f"seed {seed}: no existing checkpoint found; skipping reconstruction")
                continue
            print(f"seed {seed}: reconstructing probabilities from {checkpoint}")
            result = predict_checkpoint(
                seed=seed,
                checkpoint_dir=checkpoint,
                train_df=train_df,
                test_df=test_df,
                libs=libs,
                args=args,
            )
            save_seed_artifacts(result, args.output_dir)
            seed_results[seed] = result
            print(
                f"seed {seed}: raw_f1={result.raw_f1:.4f} "
                f"threshold={result.threshold:.2f} threshold_f1={result.threshold_f1:.4f}"
            )

    for seed in parse_seed_list(args.train_seeds):
        checkpoint = checkpoint_for_seed(seed, args.output_dir)
        cached = load_seed_artifacts(seed, args.output_dir, checkpoint, args)
        if cached is not None:
            seed_results[seed] = cached
            print(
                f"seed {seed}: reusing saved probabilities "
                f"raw_f1={cached.raw_f1:.4f} threshold={cached.threshold:.2f} "
                f"threshold_f1={cached.threshold_f1:.4f}"
            )
            continue
        print(f"seed {seed}: training new local-only DistilBERT run")
        result = train_seed(seed=seed, train_df=train_df, test_df=test_df, libs=libs, args=args)
        save_seed_artifacts(result, args.output_dir)
        seed_results[seed] = result
        print(
            f"seed {seed}: raw_f1={result.raw_f1:.4f} "
            f"threshold={result.threshold:.2f} threshold_f1={result.threshold_f1:.4f}"
        )

    references: dict[str, pd.Series | None] = {
        name: load_reference(path, test_df) for name, path in REFERENCE_SUBMISSIONS.items()
    }
    candidates: list[CandidateResult] = []

    if 13 in seed_results:
        seed13 = seed_results[13]
        raw_path = submission_path_for_seed(13, tuned=False)
        tuned_path = submission_path_for_seed(13, tuned=True)
        raw_candidate = build_candidate(
            name="seed13_raw",
            path=raw_path,
            source="DistilBERT holdout",
            seeds="13",
            probs=seed13.test_probs,
            threshold=0.5,
            test_df=test_df,
            validation_raw=seed13.raw_f1,
            validation_threshold=seed13.raw_f1,
            validation_kind="seed_holdout",
            references=references,
            recommendation="diagnostic_only",
            risk_notes="Raw threshold variant; threshold-tuned seed13 is more justified.",
        )
        references["seed13_raw"] = pd.read_csv(raw_path)["label"].astype(int)
        tuned_candidate = build_candidate(
            name="seed13_threshold",
            path=tuned_path,
            source="DistilBERT holdout threshold tuned",
            seeds="13",
            probs=seed13.test_probs,
            threshold=seed13.threshold,
            test_df=test_df,
            validation_raw=seed13.raw_f1,
            validation_threshold=seed13.threshold_f1,
            validation_kind="seed_holdout",
            references=references,
            recommendation="serious_sparse_diagnostic_candidate",
            risk_notes="Strong local holdout; single split seed variance remains the main risk.",
        )
        references["seed13_threshold"] = pd.read_csv(tuned_path)["label"].astype(int)
        for candidate in (raw_candidate, tuned_candidate):
            labels_series = pd.read_csv(candidate.path)["label"].astype(int)
            candidate.hamming_seed13 = hamming(labels_series, references["seed13_threshold"])
        seed13.raw_submission = raw_path
        seed13.threshold_submission = tuned_path
        candidates.extend([raw_candidate, tuned_candidate])

    for seed in sorted(seed_results):
        if seed in {13, 42}:
            continue
        result = seed_results[seed]
        raw_path = submission_path_for_seed(seed, tuned=False)
        tuned_path = submission_path_for_seed(seed, tuned=True)
        raw_candidate = build_candidate(
            name=f"seed{seed}_raw",
            path=raw_path,
            source="DistilBERT holdout",
            seeds=str(seed),
            probs=result.test_probs,
            threshold=0.5,
            test_df=test_df,
            validation_raw=result.raw_f1,
            validation_threshold=result.raw_f1,
            validation_kind="seed_holdout",
            references=references,
            recommendation="diagnostic_only",
            risk_notes="Raw threshold candidate from one additional seed.",
        )
        tuned_candidate = build_candidate(
            name=f"seed{seed}_threshold",
            path=tuned_path,
            source="DistilBERT holdout threshold tuned",
            seeds=str(seed),
            probs=result.test_probs,
            threshold=result.threshold,
            test_df=test_df,
            validation_raw=result.raw_f1,
            validation_threshold=result.threshold_f1,
            validation_kind="seed_holdout",
            references=references,
            recommendation="compare_before_submission",
            risk_notes="Additional seed; use ensemble diagnostics before considering Kaggle.",
        )
        candidates.extend([raw_candidate, tuned_candidate])

    if seed_results:
        ranked = sorted(seed_results.values(), key=lambda res: res.threshold_f1, reverse=True)
        ensemble_specs: list[tuple[str, list[SeedResult]]] = []
        if 42 in seed_results and 13 in seed_results:
            ensemble_specs.append(("seed42_seed13_average", [seed_results[42], seed_results[13]]))
        if len(ranked) >= 2:
            ensemble_specs.append(("top2_seed_average", ranked[:2]))
        if len(ranked) >= 3:
            ensemble_specs.append(("top3_seed_average", ranked[:3]))
        if len(ranked) >= 2:
            ensemble_specs.append(("all_seed_average", ranked))

        seen_specs = set()
        for name, members in ensemble_specs:
            key = tuple(sorted(res.seed for res in members))
            if key in seen_specs:
                continue
            seen_specs.add(key)
            probs = np.mean([res.test_probs for res in members], axis=0)
            threshold = float(np.mean([res.threshold for res in members]))
            raw_val, tuned_val, val_kind = ensemble_validation_summary(members, threshold)
            path = PART2_DIR / f"submission_bert_{name}.csv"
            recommendation = "serious_sparse_diagnostic_candidate" if len(members) <= 3 else "compare_before_submission"
            risk = "Holdout splits differ, so validation is mean seed holdout unless noted as common ensemble."
            candidate = build_candidate(
                name=name,
                path=path,
                source="DistilBERT probability average",
                seeds=",".join(str(res.seed) for res in members),
                probs=probs,
                threshold=threshold,
                test_df=test_df,
                validation_raw=raw_val,
                validation_threshold=tuned_val,
                validation_kind=val_kind,
                references=references,
                recommendation=recommendation,
                risk_notes=risk,
            )
            candidates.append(candidate)

    if candidates:
        summary_df = pd.DataFrame([candidate_to_row(cand) for cand in candidates])
        summary_df.to_csv(args.output_dir / "candidate_summary.csv", index=False)
        write_row_comparison(args.output_dir, test_df, candidates, references)

    final_candidate = next((cand for cand in candidates if cand.name == "seed42_seed13_average"), None)
    if final_candidate is None and candidates:
        final_candidate = max(
            candidates,
            key=lambda cand: -math.inf if cand.validation_threshold is None else cand.validation_threshold,
        )

    duplicate_summary = None
    transfer_path = None
    if final_candidate and not args.skip_duplicates:
        if final_candidate.name == "seed42_seed13_average" and 42 in seed_results and 13 in seed_results:
            final_probs = np.mean([seed_results[42].test_probs, seed_results[13].test_probs], axis=0)
        elif final_candidate.seeds:
            member_seeds = [int(s) for s in final_candidate.seeds.split(",") if s.strip().isdigit()]
            final_probs = np.mean([seed_results[s].test_probs for s in member_seeds], axis=0)
        else:
            final_probs = np.zeros(len(test_df))
        duplicate_summary, transfer_path = duplicate_transfer_report(
            train_df=train_df,
            test_df=test_df,
            base_candidate=final_candidate,
            base_probs=final_probs,
            output_dir=args.output_dir,
            max_near_candidates=args.max_near_duplicate_candidates,
        )
        (args.output_dir / "duplicate_transfer_summary.json").write_text(
            json.dumps(duplicate_summary, indent=2),
            encoding="utf-8",
        )

    if final_candidate:
        if final_candidate.name == "seed42_seed13_average" and 42 in seed_results and 13 in seed_results:
            final_probs = np.mean([seed_results[42].test_probs, seed_results[13].test_probs], axis=0)
        else:
            member_seeds = [int(s) for s in final_candidate.seeds.split(",") if s.strip().isdigit()]
            final_probs = np.mean([seed_results[s].test_probs for s in member_seeds], axis=0)
        pseudo = {
            "base_candidate": final_candidate.name,
            "confidence_counts": pseudo_label_diagnostics(final_probs),
            "attempted": False,
            "result": "Not attempted; this helper reports confidence only because pseudo-labeling needs a stronger OOF comparison framework.",
            "recommended": False,
        }
        (args.output_dir / "pseudo_label_diagnostics.json").write_text(json.dumps(pseudo, indent=2), encoding="utf-8")

    common_summary = run_common_split_experiment(
        train_df=train_df,
        test_df=test_df,
        libs=libs,
        args=args,
        references=references,
    )

    run_summary = {
        "data": {
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
            "label_counts": {str(k): int(v) for k, v in train_df["label"].value_counts().sort_index().items()},
        },
        "seeds": {
            str(seed): {
                "raw_macro_f1": res.raw_f1,
                "threshold": res.threshold,
                "threshold_macro_f1": res.threshold_f1,
                "checkpoint_dir": str(res.checkpoint_dir) if res.checkpoint_dir else None,
            }
            for seed, res in sorted(seed_results.items())
        },
        "candidates": [candidate_to_row(cand) for cand in candidates],
        "duplicate_transfer": duplicate_summary,
        "transfer_candidate": str(transfer_path) if transfer_path else None,
        "common_split": common_summary,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("\n=== Candidate Summary ===")
    if candidates:
        for cand in candidates:
            print(
                f"{cand.name}: threshold={cand.threshold:.2f} "
                f"val={cand.validation_threshold} distribution={cand.distribution} "
                f"hamming_seed42={cand.hamming_seed42}"
            )
    else:
        print("No candidates generated.")
    if duplicate_summary:
        print(f"Duplicate transfer: {duplicate_summary}")
    print(f"Saved diagnostics to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}") from exc
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
