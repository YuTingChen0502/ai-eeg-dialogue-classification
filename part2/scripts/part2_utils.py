from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


TEXT_COL_CANDIDATES = ("text", "content", "utterance", "sentence")
ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = r"C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe"
BASELINE_SUBMISSION = "submission_bert_common_top3_seed_average.csv"


def ensure_dirs() -> None:
    for rel in [
        "scripts",
        "outputs",
        "outputs/oof",
        "outputs/features",
        "outputs/checkpoints",
        "outputs/reports",
        "submissions",
        "submissions/backup",
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def find_csv(name: str, data_dir: Path | None = None) -> Path:
    base = data_dir or ROOT
    candidates = [base / name, ROOT / name, Path.cwd() / name, Path.cwd() / "part2" / name]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not locate {name}. Tried: {[str(p) for p in candidates]}")


def standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    text_col = next((col for col in TEXT_COL_CANDIDATES if col in df.columns), None)
    if text_col is None:
        raise KeyError(f"No text column found. Columns: {df.columns.tolist()}")
    out = df.copy()
    if text_col != "text":
        out = out.rename(columns={text_col: "text"})
    if "id" not in out.columns:
        out["id"] = np.arange(len(out))
    out["text"] = out["text"].fillna("").astype(str)
    return out


def load_data(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = standardize_frame(pd.read_csv(find_csv("train.csv", data_dir)))
    test = standardize_frame(pd.read_csv(find_csv("test.csv", data_dir)))
    if "label" not in train.columns:
        raise KeyError("train.csv must contain a label column.")
    train["label"] = train["label"].astype(int)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def validate_submission(path: Path, test_df: pd.DataFrame) -> dict:
    sub = pd.read_csv(path)
    errors: list[str] = []
    if list(sub.columns) != ["id", "label"]:
        errors.append(f"header must be exactly ['id', 'label'], got {list(sub.columns)}")
    if len(sub) != len(test_df):
        errors.append(f"row count must be {len(test_df)}, got {len(sub)}")
    if "id" in sub.columns and not sub["id"].equals(test_df["id"].reset_index(drop=True)):
        errors.append("ids do not match test.csv order")
    if "label" in sub.columns:
        labels = set(sub["label"].dropna().astype(int).unique().tolist())
        if not labels.issubset({0, 1}):
            errors.append(f"labels must be subset of {{0,1}}, got {sorted(labels)}")
        if sub["label"].isna().any():
            errors.append("labels contain missing values")
    result = {
        "path": str(path),
        "valid": not errors,
        "errors": errors,
        "shape": list(sub.shape),
        "distribution": sub["label"].value_counts().sort_index().astype(int).to_dict()
        if "label" in sub.columns
        else {},
    }
    if errors:
        raise ValueError(f"Invalid submission {path}: {errors}")
    return result


def labels_from_probs(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(probs) >= threshold).astype(int)


def tune_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold_min: float = 0.30,
    threshold_max: float = 0.70,
    threshold_step: float = 0.01,
) -> tuple[float, float]:
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2.0, threshold_step)
    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        pred = labels_from_probs(scores, float(threshold))
        macro_f1 = f1_score(y_true, pred, average="macro")
        if macro_f1 > best_f1:
            best_f1 = float(macro_f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def tune_threshold_from_values(y_true: np.ndarray, scores: np.ndarray, num: int = 201) -> tuple[float, float]:
    finite = np.asarray(scores, dtype=float)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if math.isclose(lo, hi):
        return lo, f1_score(y_true, np.zeros_like(y_true), average="macro")
    thresholds = np.linspace(lo, hi, num)
    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        pred = (finite >= threshold).astype(int)
        macro_f1 = f1_score(y_true, pred, average="macro")
        if macro_f1 > best_f1:
            best_f1 = float(macro_f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def safe_float(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=safe_float), encoding="utf-8")


def assert_finite(name: str, array: np.ndarray) -> None:
    arr = np.asarray(array)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf")


def write_submission(path: Path, test_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": test_df["id"].values, "label": labels.astype(int)})
    sub.to_csv(path, index=False)
    validate_submission(path, test_df)
    return sub


def normalize_text(text: str, strip_trailing_punct: bool = False) -> str:
    value = "" if text is None else str(text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    if strip_trailing_punct:
        value = value.rstrip(".,!?:;").strip()
    return value


def token_count(text: str) -> int:
    return len(str(text).split())


def format_dist(counts: dict | pd.Series) -> str:
    if isinstance(counts, pd.Series):
        counts = counts.sort_index().astype(int).to_dict()
    return "{" + ", ".join(f"{int(k)}: {int(v)}" for k, v in sorted(counts.items())) + "}"


def read_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def list_existing(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]
